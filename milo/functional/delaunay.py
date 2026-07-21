from typing import Union
import torch
from tetranerf.utils.extension import cpp as delaunay_cpp
from functional.pivots import extract_gaussian_pivots

# dmesh2's CGAL regular (weighted / Laguerre) triangulation. Optional: only needed when
# `use_weighted_triangulation` is enabled in a mesh config. Guarded so plain-Delaunay runs
# (and environments without dmesh2 built) are unaffected.
try:
    from mindiffdt import _C as _wdt_cpp
except Exception as _wdt_import_err:  # pragma: no cover
    _wdt_cpp = None
    _WDT_IMPORT_ERR = _wdt_import_err


def _rank_normalize(x: torch.Tensor) -> torch.Tensor:
    """Map values to their rank fraction in [0, 1] (ties broken arbitrarily, monotone in x).
    Robust to the heavy-tailed dynamic range of imp_score/distCUDA2, and faithful to density
    pruning's *top-k* selection: the bottom-ranked sites get the smallest multiplier."""
    x = x.reshape(-1).float()
    n = x.numel()
    if n <= 1:
        return torch.ones_like(x)
    order = torch.argsort(x)
    ranks = torch.empty_like(x)
    ranks[order] = torch.arange(n, device=x.device, dtype=x.dtype)
    return ranks / (n - 1)


def compute_wdt_weights(
    pivot_scales: torch.Tensor,
    mode: str = "none",
    weight_scale: float = 0.0,
    pivot_opacity: Union[torch.Tensor, None] = None,
    pivot_density: Union[torch.Tensor, None] = None,
) -> torch.Tensor:
    """Static per-pivot power weights for the weighted (regular) triangulation.

    A weight has units of length^2 and competes with squared pivot spacing in the power
    distance ||x-p||^2 - w. `pivot_scales` is a per-pivot length (e.g. the max-axis Gaussian
    scale returned alongside the pivots by `get_tetra_points`). No gradients (Phase 1/2 are
    non-differentiable).

    Modes:
        none     -> zeros (regular triangulation == Delaunay; invariant control).
        scale_sq -> (weight_scale * s_i)^2, bury pivots against their own Gaussian extent.
        aniso    -> same as scale_sq at pivot granularity (pivot scale is already max-axis).
        opacity  -> (weight_scale * s_i)^2 * opacity_i. Still locally scaled, but survival is
                    tied to *photometric* importance: low-opacity pivots get small cells and are
                    buried first. Tests whether geometry-blind burying (scale_sq, which lost
                    recall on Truck) failed because it removed load-bearing sites.
        density  -> (weight_scale * s_i)^2 * rank01(imp_score_i / distCUDA2_i). Survival tied to the
                    `--prune_rule density` signal (the project's best on-surface signal, F1 0.6372):
                    high density-importance pivots keep large power cells and survive; low ones are
                    buried first. `pivot_density` is the raw imp/dist score (rank-normalized here).
    """
    s = pivot_scales.detach().reshape(-1).float()
    if mode == "none":
        return torch.zeros_like(s)
    if mode in ("scale_sq", "aniso"):
        return (float(weight_scale) * s) ** 2
    if mode == "opacity":
        if pivot_opacity is None:
            raise ValueError("wdt_weight_mode='opacity' requires pivot_opacity.")
        o = pivot_opacity.detach().reshape(-1).float()
        if o.shape != s.shape:
            raise ValueError(f"pivot_opacity shape {tuple(o.shape)} != pivot_scales {tuple(s.shape)}")
        return ((float(weight_scale) * s) ** 2) * o
    if mode == "density":
        if pivot_density is None:
            raise ValueError("wdt_weight_mode='density' requires pivot_density (imp_score/distCUDA2).")
        d = pivot_density.detach().reshape(-1).float()
        if d.shape != s.shape:
            raise ValueError(f"pivot_density shape {tuple(d.shape)} != pivot_scales {tuple(s.shape)}")
        return ((float(weight_scale) * s) ** 2) * _rank_normalize(d)
    raise ValueError(f"Unknown wdt_weight_mode: {mode!r}")


@torch.no_grad()
def compute_triangulation(
    pivots: torch.Tensor,
    config: Union[dict, None] = None,
    pivot_scales: Union[torch.Tensor, None] = None,
    return_num_hidden: bool = False,
    pivot_opacity: Union[torch.Tensor, None] = None,
    pivot_density: Union[torch.Tensor, None] = None,
):
    """Tetrahedralize `pivots` (P, 3), returning tets (T, 4) long on CUDA.

    When `config["use_weighted_triangulation"]` is falsy (or config is None) this is the plain
    Delaunay path — byte-identical to upstream MILo. When enabled it builds a CGAL regular
    (weighted) triangulation via dmesh2, with per-pivot weights from `compute_wdt_weights`.
    With `wdt_weight_mode == "none"` the weighted path reproduces the Delaunay tet set.
    """
    use_wdt = bool(config) and config.get("use_weighted_triangulation", False)
    num_hidden = 0
    if use_wdt:
        if _wdt_cpp is None:
            raise ImportError(
                "use_weighted_triangulation is set but dmesh2 (mindiffdt._C) is not importable: "
                f"{_WDT_IMPORT_ERR}"
            )
        if pivot_scales is None:
            raise ValueError("Weighted triangulation requires pivot_scales for the weight heuristic.")
        weights = compute_wdt_weights(
            pivot_scales,
            mode=config.get("wdt_weight_mode", "none"),
            weight_scale=config.get("wdt_weight_scale", 0.0),
            pivot_opacity=pivot_opacity,
            pivot_density=pivot_density,
        )
        pts = pivots.detach().cpu().contiguous().float()
        weights = weights.detach().cpu().contiguous().float()
        tets, _cc, _t = _wdt_cpp.compute_wdt(pts, weights, False)
        tets = tets.cuda().long()
        if return_num_hidden:
            num_hidden = int(pivots.shape[0] - torch.unique(tets).numel())
    else:
        tets = delaunay_cpp.triangulate(pivots.detach()).cuda().long()
    torch.cuda.empty_cache()
    if return_num_hidden:
        return tets, num_hidden
    return tets


@torch.no_grad()
def compute_delaunay_triangulation(
    means:Union[torch.Tensor, None]=None,
    scales:Union[torch.Tensor, None]=None,
    rotations:Union[torch.Tensor, None]=None,
    gaussian_idx:Union[torch.Tensor, None]=None,
    scale_pivots_with_downsample_ratio:bool=True,
    scale_pivots_factor:float=None,
    override_pivots:Union[torch.Tensor, None]=None,
) -> torch.Tensor:
    """Compute Delaunay tetrahedralization for a set of Gaussian pivots.
    Override pivots can be provided; otherwise, pivots will be extracted from Gaussians in gaussian_idx.
    Either (means, scales, rotations) or override_pivots must be provided.

    Args:
        means (Union[torch.Tensor, None], optional): Means of the Gaussians. Shape: (N, 3). Defaults to None.
        scales (Union[torch.Tensor, None], optional): Scales of the Gaussians. Shape: (N, 3). Defaults to None.
        rotations (Union[torch.Tensor, None], optional): Rotations of the Gaussians as quaternions. Shape: (N, 4). Defaults to None.
        gaussian_idx (Union[torch.Tensor, None], optional): Indices of the Gaussians to be used for generating pivots. 
            Shape: (N_selected,). Defaults to None.
        scale_pivots_with_downsample_ratio (bool, optional): If True, the scale of the pivots will be adjusted to match the downsample ratio. Defaults to True.
        scale_pivots_factor (float, optional): If provided, the scale of the pivots will be multiplied by this factor. Defaults to None.
        override_pivots (Union[torch.Tensor, None], optional): Override pivots. Shape: (N_pivots, 3). Defaults to None.

    Returns:
        torch.Tensor: Indices of the tetrahedra. Shape: (N_tetrahedra, 4).
    """
    assert (
        (
            (means is not None) and (scales is not None) and (rotations is not None)
        ) or (
            override_pivots is not None
        )
    )
    
    # Extract pivots from Gaussians.
    # If override_pivots is provided, use it instead of extracting pivots from Gaussians.
    if override_pivots is None:
        pivots, _ = extract_gaussian_pivots(
            means=means,
            scales=scales,
            rotations=rotations,
            gaussian_idx=gaussian_idx,
            scale_pivots_with_downsample_ratio=scale_pivots_with_downsample_ratio,
            scale_pivots_factor=scale_pivots_factor
        )
    else:
        pivots = override_pivots

    print(f"[INFO] Computing Delaunay tetrahedralization for {pivots.shape[0]} points...")
    delaunay_tets = delaunay_cpp.triangulate(pivots.detach()).cuda().long()
    torch.cuda.empty_cache()
    return delaunay_tets
