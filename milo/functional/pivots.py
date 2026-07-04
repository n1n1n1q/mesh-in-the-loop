# A large portion of this script is adapted from MiniSplatting2.
# Link: https://github.com/fatPeter/mini-splatting2
# 
# The function for sampling pivots from selected Gaussians is inspired by GOF.
# Link: https://github.com/autonomousvision/gaussian-opacity-fields
#
from typing import List, Union, Tuple
import numpy as np
import torch
import trimesh
from scene.cameras import Camera
from utils.general_utils import build_rotation, compute_pivot_keep_mask
from functional.func_utils import _init_cdf_mask, _render_simp


@torch.no_grad()
def sample_gaussians_on_surface(
    views:List[Camera],
    means:torch.Tensor,
    scales:torch.Tensor,
    rotations:torch.Tensor,
    opacities:torch.Tensor,
    n_max_samples:int,
    scene_type:str,
    sampling_mask:Union[torch.Tensor, None]=None,
    sampling_method:str="surface",
) -> torch.Tensor:
    """This function aims to identify what Gaussians are located close to or on the surface.
    These Gaussians should be prioritized for generating pivots efficiently, then applying Delaunay triangulation.
    
    This function is slow and should not be called frequently:
    We typically propose to call this function just once mid-training,
    and then use the same set of Gaussians for generating pivots for all iterations.
    This is because the set of Gaussians that are used as pivots is not expected to change much.

    Args:
        views (List[Camera]): List of cameras.
        means (torch.Tensor): Means of the Gaussians. Shape: (N, 3).
        scales (torch.Tensor): Scales of the Gaussians. Shape: (N, 3).
        rotations (torch.Tensor): Rotations of the Gaussians as quaternions. Shape: (N, 4).
        opacities (torch.Tensor): Opacities of the Gaussians. Shape: (N, 1).
        n_max_samples (int): Maximum number of Gaussians to sample.
        scene_type (str): Type of the scene. Should be either 'outdoor' or 'indoor'.
        sampling_mask (Union[torch.Tensor, None], optional): Sampling mask. Shape: (N,).
            If provided, only the Gaussians in the sampling mask will be sampled. 
            Defaults to None.
        sampling_method (str, optional): Method to sample Gaussians. Should be either 'surface' or 'surface+opacity'.
            Defaults to 'surface'. If 'surface', only Gaussians likely to be on the surface will be sampled.
            Such Gaussians can be limited in number, and may not reach the desired number of samples n_max_samples.
            If 'surface+opacity', Gaussians likely to be on the surface will be sampled, 
            and then Gaussians with high opacity will be sampled to reach n_max_samples.

    Raises:
        ValueError: Invalid scene type.

    Returns:
        torch.Tensor: Indices of the sampled Gaussians. Shape: (N_sampled,).
    """
    
    assert scene_type in ['outdoor', 'indoor']
    assert sampling_method in ['surface', 'surface+opacity']
    
    imp_score = torch.zeros(means.shape[0]).cuda()
    accum_area_max = torch.zeros(means.shape[0]).cuda()

    culling=torch.zeros((means.shape[0], len(views)), dtype=torch.bool, device='cuda')

    count_rad = torch.zeros((means.shape[0],1)).cuda()
    count_vis = torch.zeros((means.shape[0],1)).cuda()

    for view in views:
        render_pkg = _render_simp(            
            viewpoint_camera=view, 
            means=means,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            bg_color=torch.tensor([0., 0., 0.], device=means.device),
            dc=None,
            shs=None,
            override_color=torch.zeros_like(means),
            scaling_modifier=1.0, 
            culling=culling[:,view.uid]
        )
        accum_weights = render_pkg["accum_weights"]
        area_proj = render_pkg["area_proj"]
        area_max = render_pkg["area_max"]

        accum_area_max = accum_area_max+area_max

        # The importance score is computed differently for indoor and outdoor scenes.
        if scene_type == 'outdoor':
            mask_t=area_max!=0 
            temp=imp_score+accum_weights/area_proj
            imp_score[mask_t] = temp[mask_t]
        elif scene_type == 'indoor':
            imp_score=imp_score+accum_weights
        else:
            raise ValueError(f"Invalid scene type: {scene_type}")

        non_prune_mask = _init_cdf_mask(importance=accum_weights, thres=0.99)

        culling[:,view.uid]=(non_prune_mask==False)

        count_rad[render_pkg["radii"]>0] += 1
        count_vis[non_prune_mask] += 1

    # The probability of each Gaussian to be sampled
    # is proportional to its importance score.
    imp_score[accum_area_max==0]=0
    if sampling_mask is not None:
        imp_score[~sampling_mask] = 0.0
    prob = imp_score/imp_score.sum()
    prob = prob.cpu().numpy()

    # Some Gaussians may have zero importance score,
    # resulting in a zero probability.
    N_xyz=means.shape[0]
    N_nonzero_prob = (prob !=0 ).sum()
    
    # The number of samples is bounded by 
    # the number of non-zero probability Gaussians.
    num_sampled = min(n_max_samples, N_nonzero_prob)
    
    # Sample num_sampled Gaussians without replacement
    indices = np.random.choice(N_xyz, size=num_sampled, p=prob, replace=False)
    
    # TODO: Following lines are not necessary and are leftover from the original code.

    non_prune_mask = np.zeros(N_xyz, dtype=bool)
    non_prune_mask[indices] = True
    factor_culling=count_vis/(count_rad+1e-1)

    # Non-sampled Gaussians
    prune_mask = (count_vis<=1)[:,0]
    prune_mask = torch.logical_or(prune_mask, torch.tensor(non_prune_mask==False, device='cuda'))
    
    # Sampled Gaussians
    sampled_idx = torch.arange(N_xyz, device='cuda')[~prune_mask]
    
    if sampling_method == 'surface+opacity':
        n_remaining_gaussians_to_sample = n_max_samples - sampled_idx.shape[0]
        if n_remaining_gaussians_to_sample > 0:
            # Creating a sample mask by removing the already sampled Gaussians
            opacity_sample_mask = torch.ones(means.shape[0], device="cuda", dtype=torch.bool)
            opacity_sample_mask[sampled_idx] = False
            
            # Filtering out the already sampled Gaussians by setting their opacities to 0
            opacities_filtered = opacities.clone()
            opacities_filtered[~opacity_sample_mask] = 0.0
            
            # Converting the opacities to probabilities
            prob = opacities_filtered / opacities_filtered.sum()
            prob = prob.cpu().numpy()
            
            # Sampling Gaussians with high opacity
            N_xyz=means.shape[0]
            N_nonzero_prob = (prob !=0 ).sum()
            
            num_sampled = min(n_remaining_gaussians_to_sample, N_nonzero_prob)
            
            indices = np.random.choice(N_xyz, size=num_sampled, p=prob, replace=False)
            opacity_sampled_idx = torch.tensor(indices, device=opacities_filtered.device)
            
            # Concatenating the surface Gaussians with the sampled Gaussians with high opacity
            sampled_idx = torch.cat(
                [
                    sampled_idx,
                    opacity_sampled_idx,
                ],
                dim=0,
            )
            sampled_idx = torch.sort(sampled_idx, dim=0)[0]
    
    return sampled_idx


def extract_gaussian_pivots(
    means:torch.Tensor,
    scales:torch.Tensor,
    rotations:torch.Tensor,
    gaussian_idx:Union[torch.Tensor, None]=None,
    scale_pivots_with_downsample_ratio:bool=True,
    scale_pivots_factor:float=None,
    use_adaptive_pivot_sampling:bool=False,
    gaussian_type_linear_ratio:float=3.0,
    gaussian_type_planar_ratio:float=3.0,
    override_keep_mask:Union[torch.Tensor, None]=None,
) -> Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, None]]:
    """Extract pivots from Gaussians, in a differentiable manner.
    Each Gaussian will spawn 9 pivots, unless use_adaptive_pivot_sampling is True, in which case
    each Gaussian will spawn 9 (volumetric), 5 (planar) or 3 (linear) pivots depending on its shape.
    A list of indices can be provided to generate pivots only for a subset of Gaussians.
    We recommend to use only Gaussians that are located on or near the surface;
    indices of such Gaussians can be obtained by calling sample_gaussians_on_surface(...).

    Args:
        means (torch.Tensor): Means of the Gaussians. Shape: (N, 3).
        scales (torch.Tensor): Scales of the Gaussians. Shape: (N, 3).
        rotations (torch.Tensor): Rotations of the Gaussians as quaternions. Shape: (N, 4).
        gaussian_idx (Union[torch.Tensor, None], optional): Indices of the Gaussians to be used for generating pivots.
            Shape: (N_selected,). Defaults to None.
        scale_pivots_with_downsample_ratio (bool, optional): If True, the scale of the pivots will be adjusted to match the downsample ratio. Defaults to True.
        scale_pivots_factor (float, optional): If provided, the scale of the pivots will be multiplied by this factor. Defaults to None.
        use_adaptive_pivot_sampling (bool, optional): If True, classify each Gaussian as volumetric, planar
            or linear from its scale, and only emit the subset of the 8 bounding-box corners relevant to
            its shape (8 for volumetric, 4 for planar, 2 for linear), always keeping the center.
            Defaults to False (legacy behavior: always 8 corners + center).
        gaussian_type_linear_ratio (float, optional): Ratio threshold for "linear" classification.
            Only used if use_adaptive_pivot_sampling is True. Defaults to 3.0.
        gaussian_type_planar_ratio (float, optional): Ratio threshold for "planar" classification.
            Only used if use_adaptive_pivot_sampling is True. Defaults to 3.0.
        override_keep_mask (Union[torch.Tensor, None], optional): If provided, use this (N_selected, 9)
            boolean mask directly instead of reclassifying Gaussians from their current scale. Required to
            keep the corner/center layout consistent with a previously computed Delaunay triangulation
            between two triangulation updates. Only used if use_adaptive_pivot_sampling is True.
            Defaults to None (recompute the mask from the current scale).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, None]]: Pivots, their scales, and a keep_mask.
            Pivots: Shape: (9*N_selected, 3), or (M, 3) if use_adaptive_pivot_sampling is True.
            Pivots_scale: Shape: (9*N_selected, 1), or (M, 1) if use_adaptive_pivot_sampling is True.
            keep_mask: None if use_adaptive_pivot_sampling is False (legacy corners-then-centers layout).
                Otherwise, boolean mask of shape (N_selected, 9) indicating which of the 8 corners + center
                were emitted for each Gaussian, in row-major (gaussian-major, slot-minor) order.
    """
    M = trimesh.creation.box()
    M.vertices *= 2

    xyz = means.clone()
    scale = scales.clone() * 3.
    rots = build_rotation(rotations.clone())

    if gaussian_idx is not None:
        # Compute downsample ratio between the total number of Gaussians
        # and the number of Gaussians to be used for generating pivots.
        downsample_ratio = gaussian_idx.shape[0] / xyz.shape[0]

        # Select the Gaussians to be used for generating pivots.
        xyz = xyz[gaussian_idx]
        scale = scale[gaussian_idx]
        rots = rots[gaussian_idx]

        # Adjust the scale of the pivots to match the downsample ratio.
        if scale_pivots_with_downsample_ratio:
            scale = scale / (downsample_ratio ** (1/3))
        elif scale_pivots_factor is not None:
            scale = scale * scale_pivots_factor

    if use_adaptive_pivot_sampling:
        if override_keep_mask is not None:
            keep_mask = override_keep_mask
        else:
            log_scale = torch.log(scale.clamp(min=1e-12))
            corner_signs = torch.from_numpy(M.vertices).float().to(xyz.device)
            keep_mask = compute_pivot_keep_mask(
                log_scale, corner_signs,
                linear_ratio=gaussian_type_linear_ratio,
                planar_ratio=gaussian_type_planar_ratio,
            )  # (N, 9): 8 corners + center

        corners = M.vertices.T
        corners = torch.from_numpy(corners).float().cuda().unsqueeze(0).repeat(xyz.shape[0], 1, 1)
        corners = corners * scale.unsqueeze(-1)
        corners = torch.bmm(rots, corners).squeeze(-1) + xyz.unsqueeze(-1)
        corners = corners.permute(0, 2, 1).contiguous()  # (N, 8, 3)

        scale_max = scale.max(dim=-1, keepdim=True)[0]  # (N, 1)
        pivots_full = torch.cat([corners, xyz.unsqueeze(1)], dim=1)  # (N, 9, 3)
        scale_full = scale_max.unsqueeze(1).expand(-1, 9, -1)  # (N, 9, 1)

        pivots = pivots_full[keep_mask].contiguous()
        pivots_scale = scale_full[keep_mask].contiguous()

        return pivots, pivots_scale, keep_mask

    pivots = M.vertices.T
    pivots = torch.from_numpy(pivots).float().cuda().unsqueeze(0).repeat(xyz.shape[0], 1, 1)

    # Reparameterization trick
    pivots = pivots * scale.unsqueeze(-1)
    pivots = torch.bmm(rots, pivots).squeeze(-1) + xyz.unsqueeze(-1)
    pivots = pivots.permute(0, 2, 1).reshape(-1, 3).contiguous()

    # Concatenate center points
    pivots = torch.cat([pivots, xyz], dim=0)

    # Scale is not a good solution but use it for now
    scale = scale.max(dim=-1, keepdim=True)[0]
    scale_corner = scale.repeat(1, 8).reshape(-1, 1)
    pivots_scale = torch.cat([scale_corner, scale], dim=0)

    return pivots, pivots_scale, None
