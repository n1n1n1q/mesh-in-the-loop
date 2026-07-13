# A large portion of this script is adapted from MiniSplatting2.
# Link: https://github.com/fatPeter/mini-splatting2
#
from typing import Union, Dict
import math
import torch
from diff_gaussian_rasterization_ms import GaussianRasterizationSettings as MiniSplatting2RasterizationSettings
from diff_gaussian_rasterization_ms import GaussianRasterizer as MiniSplatting2Rasterizer
from scene.cameras import Camera


def _init_cdf_mask(
    importance:torch.Tensor, 
    thres:float=1.0
) -> torch.Tensor:
    importance = importance.flatten()   
    if thres!=1.0:
        percent_sum = thres
        vals,idx = torch.sort(importance+(1e-6))
        cumsum_val = torch.cumsum(vals, dim=0)
        split_index = ((cumsum_val/vals.sum()) > (1-percent_sum)).nonzero().min()
        split_val_nonprune = vals[split_index]

        non_prune_mask = importance>split_val_nonprune 
    else: 
        non_prune_mask = torch.ones_like(importance).bool()
        
    return non_prune_mask


def _render_simp(
    viewpoint_camera:Camera, 
    means:torch.Tensor,
    opacities:torch.Tensor,
    scales:torch.Tensor,
    rotations:torch.Tensor,
    bg_color:torch.Tensor,
    dc:Union[torch.Tensor, None]=None,
    shs:Union[torch.Tensor, None]=None,
    override_color:Union[torch.Tensor, None]=None, 
    scaling_modifier:float=1.0, 
    culling:Union[torch.Tensor, None]=None
) -> Dict[str, torch.Tensor]:
    """
    Background tensor (bg_color) must be on GPU!
    """
    
    assert (dc is not None and shs is not None) or (override_color is not None)
    if override_color is not None:
        active_sh_degree = 0
    else:
        active_sh_degree = int(math.sqrt(shs.shape[1] + 1) - 1)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(means, dtype=means.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = MiniSplatting2RasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = MiniSplatting2Rasterizer(raster_settings=raster_settings)
    means2D = screenspace_points

    if culling==None:
        culling=torch.zeros(means.shape[0], dtype=torch.bool, device='cuda')

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    (
        rendered_image, radii, 
        accum_weights_ptr, accum_weights_count, accum_max_count
    )  = rasterizer.render_simp(
        means3D=means,
        means2D=means2D,
        dc=dc,
        shs=shs,
        culling=culling,
        colors_precomp=override_color,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    # The Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : (radii > 0).nonzero(),
            "radii": radii,
            "accum_weights": accum_weights_ptr,
            "area_proj": accum_weights_count,
            "area_max": accum_max_count,
        }


def _part1by2_64(x:torch.Tensor) -> torch.Tensor:
    """Spread the low 21 bits of `x` out so that two zero bits sit between each."""
    x = x & 0x1FFFFF
    x = (x | (x << 32)) & 0x1F00000000FFFF
    x = (x | (x << 16)) & 0x1F0000FF0000FF
    x = (x | (x << 8))  & 0x100F00F00F00F00F
    x = (x | (x << 4))  & 0x10C30C30C30C30C3
    x = (x | (x << 2))  & 0x1249249249249249
    return x


def _morton_code_3d(coords_norm:torch.Tensor, bits:int=21) -> torch.Tensor:
    """
    Morton (Z-order) codes for points already normalized to [0, 1]^3.

    Args:
        coords_norm: (N, 3) float tensor with values in [0, 1].
        bits: Quantization bits per axis. 21 keeps the code inside a signed int64.

    Returns:
        (N,) int64 tensor of interleaved codes. Sorting by this code makes spatial
        neighbours contiguous in 1D.
    """
    assert 1 <= bits <= 21, f"bits must be in [1, 21] to fit in int64, got {bits}"
    max_q = (1 << bits) - 1
    q = (coords_norm.clamp(0.0, 1.0) * max_q).long().clamp(0, max_q)
    return (
        _part1by2_64(q[:, 0])
        | (_part1by2_64(q[:, 1]) << 1)
        | (_part1by2_64(q[:, 2]) << 2)
    )


@torch.no_grad()
def stratified_importance_sample(
    means:torch.Tensor,
    imp_score:torch.Tensor,
    num_sampled:int,
    alpha:float=1.0,
    bits:int=21,
) -> torch.Tensor:
    """
    Z-curve stratified importance sampling, without replacement.

    Candidates (`imp_score > 0`) are sorted along a Morton curve and split into
    `num_sampled` equal-count strata. Exactly one candidate is drawn from each stratum
    with probability proportional to `imp_score ** alpha`. This is stratification plus
    importance sampling: it guarantees spatial spread (one pick per stratum, and strata
    are contiguous along a space-filling curve) while still preferring high-importance
    Gaussians *within* each stratum.

        alpha = 1.0 -> importance-weighted within stratum
        alpha = 0.0 -> uniform within stratum (pure spatial stratification)

    Fully vectorized: the per-stratum draw uses the exponential race trick
    (key = -log(u) / w, smallest key wins => P(win) proportional to w) resolved with a
    segmented argmin via `scatter_reduce_`.

    Args:
        means: (N, 3) Gaussian centers.
        imp_score: (N,) non-negative importance. Zeros are excluded from sampling.
        num_sampled: Number of samples to draw.
        alpha: Importance exponent applied within each stratum.
        bits: Morton quantization bits per axis.

    Returns:
        1D int64 tensor of indices into the original N-length arrays. If
        `num_sampled >= n_candidates`, every candidate is returned.
    """
    device = means.device
    imp = imp_score.reshape(-1).float()
    candidates = torch.nonzero(imp > 0, as_tuple=True)[0]
    n_candidates = candidates.numel()

    num_sampled = int(num_sampled)
    if num_sampled <= 0:
        return candidates[:0]
    if num_sampled >= n_candidates:
        return candidates

    # Morton-sort the candidates so spatial neighbours become contiguous.
    pts = means[candidates]
    lo = pts.amin(dim=0)
    extent = (pts.amax(dim=0) - lo).clamp_min(1e-12)
    codes = _morton_code_3d((pts - lo) / extent, bits=bits)
    sorted_idx = candidates[torch.argsort(codes)]

    # Equal-count strata: rank r lands in stratum (r * num_sampled) // n_candidates.
    # Every stratum is non-empty because num_sampled < n_candidates.
    ranks = torch.arange(n_candidates, device=device)
    strata = (ranks * num_sampled) // n_candidates

    weights = imp[sorted_idx]
    if alpha != 1.0:
        weights = weights.pow(alpha)
    weights = weights.clamp_min(torch.finfo(weights.dtype).tiny)

    # Exponential race within each stratum.
    u = torch.rand(n_candidates, device=device).clamp_min(torch.finfo(weights.dtype).tiny)
    keys = -torch.log(u) / weights

    best = torch.full((num_sampled,), float("inf"), device=device, dtype=keys.dtype)
    best.scatter_reduce_(0, strata, keys, reduce="amin", include_self=True)

    # Break exact-tie keys deterministically by taking the lowest rank among the minima.
    tie_rank = torch.where(keys == best[strata], ranks, torch.full_like(ranks, n_candidates))
    winner_rank = torch.full((num_sampled,), n_candidates, device=device, dtype=ranks.dtype)
    winner_rank.scatter_reduce_(0, strata, tie_rank, reduce="amin", include_self=True)

    return sorted_idx[winner_rank]