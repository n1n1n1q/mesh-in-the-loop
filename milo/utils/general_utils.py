#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
import math
from datetime import datetime
import numpy as np
import random

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def classify_gaussian_types(
    log_scale:torch.Tensor,
    linear_ratio:float=3.0,
    planar_ratio:float=3.0,
) -> torch.Tensor:
    """Classify Gaussians as volumetric, planar or linear from their (log-space) scale.

    Gaussians are classified by comparing the gaps between sorted log-scales
    (l0 >= l1 >= l2) against epsilon thresholds derived from the given ratios
    (a gap of eps=log(ratio) in log-space corresponds to a ratio-x difference
    in linear scale space). Comparing gaps instead of ratios avoids division.

    Args:
        log_scale (torch.Tensor): Log-space scale of the Gaussians. Shape: (N, 3).
        linear_ratio (float, optional): Ratio threshold for the largest scale to dominate
            both other axes for a Gaussian to be classified as "linear". Defaults to 3.0.
        planar_ratio (float, optional): Ratio threshold for the two largest scales to dominate
            the smallest axis for a Gaussian to be classified as "planar". Defaults to 3.0.

    Returns:
        torch.Tensor: Gaussian type codes. Shape: (N,). 0=volumetric, 1=planar, 2=linear.
    """
    sorted_log_scale = torch.sort(log_scale, dim=-1, descending=True)[0]
    gap_major = sorted_log_scale[:, 0] - sorted_log_scale[:, 1]
    gap_minor = sorted_log_scale[:, 1] - sorted_log_scale[:, 2]

    eps_linear = math.log(linear_ratio)
    eps_planar = math.log(planar_ratio)

    is_linear = gap_major > eps_linear
    is_planar = (~is_linear) & (gap_minor > eps_planar)

    gaussian_type = torch.zeros(log_scale.shape[0], dtype=torch.int8, device=log_scale.device)
    gaussian_type[is_planar] = 1
    gaussian_type[is_linear] = 2
    return gaussian_type

def compute_pivot_keep_mask(
    log_scale:torch.Tensor,
    corner_signs:torch.Tensor,
    linear_ratio:float=3.0,
    planar_ratio:float=3.0,
    gaussian_types:torch.Tensor=None,
) -> torch.Tensor:
    """Compute a per-Gaussian keep-mask selecting which of the 9 canonical pivot slots
    (8 bounding-box corners + 1 center, in that order) should be used as Delaunay sites.

    Volumetric Gaussians keep all 8 corners. Planar and linear Gaussians keep a corner
    subset chosen to stay symmetric around the center along the axes being collapsed,
    rather than collapsing to one side of the Gaussian -- a one-sided reduction removes
    the local 3D thickness that keeps nearby tetrahedra well-conditioned, which is
    especially risky for planar Gaussians since they are extremely common on flat
    surfaces and a whole contiguous flat region losing its off-plane spread can degrade
    the Delaunay triangulation into slivers there:
      - Planar Gaussians keep the 4 corners forming a regular tetrahedron inscribed in
        the bounding box (the corners where sign_dominant * sign_mid * sign_minor == 1).
        This covers all 4 (dominant, mid) sign combinations exactly once, each paired
        with an alternating minor-axis sign, so the kept points remain balanced on both
        sides of the collapsed (minor) axis instead of only one.
      - Linear Gaussians keep 1 pair of antipodal corners (all three signs identical,
        i.e. (+1,+1,+1) and (-1,-1,-1)), which differ maximally along the dominant axis
        while also being symmetric (through the center) along the two collapsed axes.
    The center (slot 8) is always kept.

    Args:
        log_scale (torch.Tensor): Log-space scale of the Gaussians. Shape: (N, 3).
        corner_signs (torch.Tensor): Sign pattern of the 8 canonical box corners
            (the same corners used to build the pivot points), values in {-1, 1}.
            Shape: (8, 3).
        linear_ratio (float, optional): See `classify_gaussian_types`. Defaults to 3.0.
        planar_ratio (float, optional): See `classify_gaussian_types`. Defaults to 3.0.
        gaussian_types (torch.Tensor, optional): Precomputed per-Gaussian type codes of shape (N,)
            (0=volumetric, 1=planar, 2=linear), e.g. a frozen classification. If provided, it is
            used directly instead of reclassifying from the current scale (linear_ratio/planar_ratio
            are then ignored); the per-axis ranking used to select corners is still derived from the
            current log_scale. Defaults to None (classify from log_scale).

    Returns:
        torch.Tensor: Boolean keep-mask. Shape: (N, 9).
    """
    n_gaussians = log_scale.shape[0]
    device = log_scale.device

    if gaussian_types is not None:
        gaussian_type = gaussian_types.to(device)
    else:
        gaussian_type = classify_gaussian_types(log_scale, linear_ratio=linear_ratio, planar_ratio=planar_ratio)
    # rank[:, 0] = index of the dominant (largest-scale) axis, rank[:, 2] = most-minor axis
    rank = torch.argsort(log_scale, dim=-1, descending=True)

    # For each Gaussian and each of the 8 corners, reorder the corner's sign vector
    # so that column order becomes [dominant axis sign, mid axis sign, minor axis sign].
    signs = corner_signs.to(device).unsqueeze(0).expand(n_gaussians, 8, 3)
    rank_expanded = rank.unsqueeze(1).expand(n_gaussians, 8, 3)
    signs_ranked = torch.gather(signs, dim=2, index=rank_expanded)

    keep_volumetric = torch.ones(n_gaussians, 8, dtype=torch.bool, device=device)
    # Alternating (tetrahedral) subset: even number of -1 signs, i.e. product of signs == 1.
    keep_planar = (signs_ranked[..., 0] * signs_ranked[..., 1] * signs_ranked[..., 2]) == 1
    # Antipodal pair: all three signs identical (either all +1 or all -1).
    keep_linear = (signs_ranked[..., 0] == signs_ranked[..., 1]) & (signs_ranked[..., 1] == signs_ranked[..., 2])

    keep_corners = torch.where(
        (gaussian_type == 2).unsqueeze(-1),
        keep_linear,
        torch.where((gaussian_type == 1).unsqueeze(-1), keep_planar, keep_volumetric),
    )

    keep_center = torch.ones(n_gaussians, 1, dtype=torch.bool, device=device)
    return torch.cat([keep_corners, keep_center], dim=1)

def compute_type_shape_regularization(
    log_scale:torch.Tensor,
    gaussian_types:torch.Tensor,
    linear_target_ratio:float=5.0,
    planar_target_ratio:float=5.0,
) -> torch.Tensor:
    """Type-specific shape regularization pushing each Gaussian toward its (frozen) type.

    Works on the same sorted log-scale gaps as `classify_gaussian_types` (l0 >= l1 >= l2):
      gap_major = l0 - l1   (dominance of the largest axis over the second)
      gap_minor = l1 - l2   (dominance of the second axis over the smallest)
    Both targets are one-sided hinges in log-space, so a Gaussian already "anisotropic enough"
    (gap >= log(target_ratio)) contributes exactly zero and is never pushed further -- the loss
    only sharpens under-shaped Gaussians up to the target ratio, bounding scale drift.

      - Linear (type 2): penalize relu(log(linear_target_ratio) - gap_major). The gradient grows
        the largest log-scale (one axis dominates further, "one eigenvalue bigger") and shrinks
        the second-largest -> more needle-like.
      - Planar (type 1): penalize relu(log(planar_target_ratio) - gap_minor). The gradient shrinks
        the smallest log-scale ("minimum eigenvalue smaller") and slightly grows the second
        -> more disk-like.
      - Volumetric (type 0): no regularization.

    Args:
        log_scale (torch.Tensor): Log-space scale of the Gaussians (grad-enabled). Shape: (N, 3).
        gaussian_types (torch.Tensor): Per-Gaussian frozen type codes. Shape: (N,).
            0=volumetric, 1=planar, 2=linear.
        linear_target_ratio (float, optional): Target l0/l1 ratio for linear Gaussians. Defaults to 5.0.
        planar_target_ratio (float, optional): Target l1/l2 ratio for planar Gaussians. Defaults to 5.0.

    Returns:
        torch.Tensor: Scalar regularization loss (mean over each populated type, summed across types).
    """
    sorted_log_scale = torch.sort(log_scale, dim=-1, descending=True)[0]
    gap_major = sorted_log_scale[:, 0] - sorted_log_scale[:, 1]
    gap_minor = sorted_log_scale[:, 1] - sorted_log_scale[:, 2]

    gaussian_types = gaussian_types.to(log_scale.device)
    is_linear = gaussian_types == 2
    is_planar = gaussian_types == 1

    loss = log_scale.new_zeros(())
    if is_linear.any():
        eps_linear = math.log(linear_target_ratio)
        loss = loss + (eps_linear - gap_major[is_linear]).clamp(min=0.).mean()
    if is_planar.any():
        eps_planar = math.log(planar_target_ratio)
        loss = loss + (eps_planar - gap_minor[is_planar]).clamp(min=0.).mean()
    return loss

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
