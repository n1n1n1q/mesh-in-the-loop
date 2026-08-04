from typing import Dict, Any, Optional
import torch
from scene.gaussian_model import GaussianModel


def compute_screen_space_residual(
    mesh_depth: torch.Tensor,
    gaussians_depth: torch.Tensor,
    mesh_normal_view: torch.Tensor,
    gaussians_normal_view: torch.Tensor,
    rasterization_mask: torch.Tensor,
    spatial_lr_scale: float,
    depth_weight: float = 1.0,
    normal_weight: float = 1.0,
) -> torch.Tensor:
    """
    Per-pixel screen-space residual between the mesh render and the Gaussian volumetric render.

    Reuses the same depth/normal terms as the mesh depth/normal losses, so this map is
    consistent with what is actually backpropagated to the Gaussians (see mesh.py):
    it is a diagnostic view of the same signal that already drives densification via
    autograd, not a separate accumulation path.

    Returns:
        torch.Tensor: Residual map of shape (H, W), zero outside mesh coverage.
    """
    depth_term = torch.log(1. + (mesh_depth - gaussians_depth).abs() / spatial_lr_scale)
    normal_term = 1. - (mesh_normal_view * gaussians_normal_view).sum(dim=-1).abs()
    residual = (depth_weight * depth_term + normal_weight * normal_term) * rasterization_mask
    return residual


def get_per_gaussian_residual_weight(
    gaussians: GaussianModel,
    default: float = 1.0,
) -> torch.Tensor:
    """
    Per-Gaussian weight proxying local mesh/Gaussian disagreement.

    Built from the mesh-phase densification gradient accumulator (mesh_xyz_gradient_accum):
    Gaussians that the mesh depth/normal loss backward pass pushes the hardest are exactly
    the ones sitting where the mesh currently disagrees the most with the Gaussians, since
    that gradient flows through the same differentiable pivot -> Delaunay -> marching
    tetrahedra -> mesh-render chain that produces the mesh loss. This avoids re-deriving a
    separate per-Gaussian residual signal from the per-pixel map.

    Returns a tensor of shape (N_gaussians,), normalized to have mean 1.
    """
    n_gaussians = gaussians.get_xyz.shape[0]
    accum = gaussians.mesh_xyz_gradient_accum
    denom = gaussians.mesh_denom
    if (accum.shape[0] != n_gaussians) or (denom.sum() <= 0):
        return torch.full((n_gaussians,), default, device=gaussians.get_xyz.device)
    grad = (accum / denom.clamp(min=1.)).squeeze(-1)  # (N,)
    weight = grad / grad.mean().clamp(min=1e-12)
    return weight


def compute_adaptive_pivot_budget(iteration: int, config: Dict[str, Any]) -> int:
    """
    Growing budget for the number of Delaunay points (n_max_points_in_delaunay),
    from pivot_budget_init up to pivot_budget_max, doubling^growth_factor every
    pivot_growth_interval iterations since mesh regularization started.
    """
    start_iter = config["start_iter"]
    n_steps = max(0, (iteration - start_iter) // config["pivot_growth_interval"])
    budget = config["pivot_budget_init"] * (config["pivot_growth_factor"] ** n_steps)
    return int(min(config["pivot_budget_max"], budget))


def compute_erosion_anneal_scale(iteration: int, config: Dict[str, Any]) -> float:
    """
    Linear decay factor (1.0 -> 0.0) for the erosion hinge loss over the final
    erosion_anneal_last_iters iterations of the mesh phase.

    Early in mesh regularization, the residual-driven pull-back is doing real work:
    correcting genuinely eroded regions where the mesh has collapsed. But once the
    geometry has mostly converged, the same pull-back is reacting to whatever residual
    noise remains (see get_per_gaussian_residual_weight) rather than a real erosion
    signal, and keeps nudging pivots for the rest of training -- adding late-stage
    noise instead of correcting anything. Decaying the hinge to 0 over the last stretch
    (mirroring how pivot_freeze_last_iters already freezes pivot growth in the same
    window) lets the geometry settle instead of chasing residual noise until the end.
    """
    anneal_last_iters = config.get("erosion_anneal_last_iters", 0)
    if anneal_last_iters <= 0:
        return 1.0
    anneal_start = config["stop_iter"] - anneal_last_iters
    if iteration <= anneal_start:
        return 1.0
    progress = (iteration - anneal_start) / anneal_last_iters
    return max(0.0, 1.0 - progress)


def compute_erosion_loss(
    current_occupancy: torch.Tensor,
    gaussian_idx: Optional[torch.Tensor],
    gaussians: GaussianModel,
    config: Dict[str, Any],
) -> torch.Tensor:
    """
    Residual-weighted erosion hinge.

    Extends the paper's occupied-centers loss (Eq. 8, `max(0, iso - sdf_center)`, which
    only ever looks at Gaussian centers) by weighting the hinge with the local residual
    proxy, so the pull-back pressure concentrates where the mesh is currently wrong,
    instead of being applied uniformly everywhere. The caller controls which pivots are
    included: mesh.py passes all 9 pivots by default, or just the center pivot when
    erosion_hinge_center_only is set (matching the paper's original center-only scope).

    Args:
        current_occupancy (torch.Tensor): Occupancy of the sampled pivots. Shape (N, 9)
            or (N, 1) if only the center pivot is passed.
        gaussian_idx (torch.Tensor, optional): Indices of the sampled Gaussians into the
            full Gaussian set, or None if all Gaussians are sampled.
    """
    isosurface = config["sdf_default_isosurface"]
    hinge = (isosurface - current_occupancy).clamp(min=0.)  # (N, 9)

    weight = get_per_gaussian_residual_weight(gaussians)  # (N_total,)
    if gaussian_idx is not None:
        weight = weight[gaussian_idx]
    weight = weight.clamp(max=config.get("erosion_residual_weight_clip", 5.0)).unsqueeze(-1)  # (N, 1)

    return config["occupied_centers_weight"] * (hinge * weight).mean()


def compute_erosion_reseed_alpha(
    previous_occupancy: torch.Tensor,
    gaussian_idx: Optional[torch.Tensor],
    gaussians: GaussianModel,
    config: Dict[str, Any],
    default_alpha: float,
) -> torch.Tensor:
    """
    Per-pivot EMA blend factor for the periodic SDF reset.

    Normally, resetting SDFs blends the fresh depth-fusion estimate with the existing
    learned value via a fixed EMA factor. But once a pivot has eroded (occupancy
    saturated near 1, i.e. the tet flipped fully "outside"), the fresh depth-fusion
    estimate is itself computed from the current -- already eroded -- Gaussian render,
    so a smooth EMA blend just perpetuates the collapse. For pivots that look eroded
    (saturated occupancy) in a region with high residual (mesh disagrees with Gaussians),
    we instead hard-reseed (alpha -> erosion_hard_reseed_alpha, default 1.0, i.e. fully
    replaced) to break the self-reinforcing loop and let gradient flow back in.
    """
    weight = get_per_gaussian_residual_weight(gaussians)
    if gaussian_idx is not None:
        weight = weight[gaussian_idx]
    weight = weight.unsqueeze(-1).expand_as(previous_occupancy)  # (N, 9)

    eroded = (
        (previous_occupancy > config["erosion_occupancy_threshold"])
        & (weight > config["erosion_residual_threshold"])
    )
    alpha = torch.where(
        eroded,
        torch.full_like(previous_occupancy, config.get("erosion_hard_reseed_alpha", 1.0)),
        torch.full_like(previous_occupancy, default_alpha),
    )
    return alpha


def compute_anti_saturation_loss(
    occupancy_shift: torch.Tensor,
    gaussian_idx: Optional[torch.Tensor],
    gaussians: GaussianModel,
    config: Dict[str, Any],
) -> torch.Tensor:
    """
    dSoft regularizer pulling the learned occupancy shift back from extreme magnitudes
    for high-residual pivots.

    The paper identifies erosion's root cause as: once sigmoid(base_occupancy + shift)
    saturates near 1, the mesh rasterization is sharp enough that gradients vanish and
    geometry can't recover. This penalizes |shift| beyond a margin (where the sigmoid is
    already saturated) so pivots don't drift into the dead zone in the first place,
    concentrated on Gaussians where the mesh currently disagrees with the render.
    """
    weight = get_per_gaussian_residual_weight(gaussians)
    if gaussian_idx is not None:
        weight = weight[gaussian_idx]
    weight = weight.clamp(max=config.get("erosion_residual_weight_clip", 5.0)).unsqueeze(-1)  # (N, 1)

    margin = config["erosion_anti_saturation_margin"]
    excess = (occupancy_shift.abs() - margin).clamp(min=0.)
    return config["erosion_anti_saturation_weight"] * (weight * excess.pow(2)).mean()
