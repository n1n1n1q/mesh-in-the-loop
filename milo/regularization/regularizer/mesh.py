from typing import Callable, Dict, Any, Tuple
from functools import partial
import gc
import math
import numpy as np
import torch
from arguments import PipelineParams
from scene import Scene
from scene.cameras import Camera
from gaussian_renderer import render_simp
from scene.mesh import Meshes, MeshRasterizer, MeshRenderer, ScalableMeshRenderer
from scene.gaussian_model import GaussianModel
from utils.tetmesh import marching_tetrahedra
from utils.camera_utils import get_cameras_spatial_extent
from utils.geometry_utils import is_in_view_frustum
from utils.geometry_utils import depth_to_normal as depth_double_to_normal
from regularization.sdf.integration import (
    evaluate_cull_sdf_values,
)
from regularization.sdf.depth_fusion import (
    evaluate_mesh_occupancy,
)
from regularization.sdf.depth_fusion import evaluate_sdf_values as evaluate_sdf_values_depth_fusion
from regularization.sdf.learnable import (
    compute_initial_sdf_with_binary_search,
    convert_sdf_to_occupancy,
    convert_occupancy_to_sdf,
)
from utils.geometry_utils import (
    unflatten_voronoi_features,
    flatten_voronoi_features,
    depths_to_points,
)

# Try importing cpp extension, handle potential ImportError
try:
    from tetranerf.utils.extension import cpp
except ImportError:
    cpp = None
    print("[WARNING] Could not import 'tetranerf.utils.extension.cpp'. Mesh regularization requires this.")

# Config-gated triangulation dispatcher (plain Delaunay, or weighted/regular when enabled).
from functional.delaunay import compute_triangulation
from simple_knn._C import distCUDA2  # for the WDT `density` weight mode (imp_score / distCUDA2)


from diffmesh.adapter import normalize_points, nearest_excluding_face
from mindiffdt.minball import MB3_V0
from dmesh2_renderer import Renderer as _DMeshRenderer


C0_SH = 0.28209479177387814  # SH DC -> base RGB


def unique_delaunay_faces(tets: torch.Tensor) -> torch.Tensor:
    """
    Unique triangular faces of the tetrahedralization for DMesh candidates
    ."""
    tets = tets.long()
    faces = torch.cat([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]],
                       tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]], dim=0)
    faces = torch.sort(faces, dim=1)[0]
    return torch.unique(faces, dim=0)


def pivot_base_colors(gaussians, xyz_idx):
    """
    Per-pivot base RGB (SH-DC) laid out to match get_tetra_points: [8N corners (g-major), N centers].
    """
    gc = (0.5 + C0_SH * gaussians.get_features[:, 0, :]).clamp(0.0, 1.0).detach()  # (Ng, 3)
    if xyz_idx is not None:
        gc = gc[xyz_idx]
    corners = gc.repeat_interleave(8, dim=0)  # g0 x8, g1 x8, ...  == flatten_voronoi corner order
    return torch.cat([corners, gc], dim=0).contiguous()  # (9N, 3)


def target_depth_in_renderer_space(viewpoint_cam, median_depth, proj):
    """Convert a radegs metric (view-space z) depth map into the *same* normalized-NDC depth space
    the dmesh2 renderer outputs — `1 - (ndc_z + 1)/2` — so it can supervise the soft mesh depth
    pixel-wise. Steps: unproject the depth to view-space points (`depths_to_points`), project them
    through the projection matrix exactly as the renderer does (clip = view_hom @ projection_matrix,
    where the renderer's `proj` arg is projection_matrix.T so we use `proj.t()`), take ndc_z=clip_z/w,
    and apply the renderer's `(1 - (ndc_z+1)/2)` transform. Returns ((H,W) depth, (H,W) valid mask)."""
    dm = median_depth.squeeze()                                   # (H, W) view-space z (metric)
    H, W = dm.shape
    pview = depths_to_points(viewpoint_cam, dm).reshape(3, -1).t()  # (H*W, 3) view-space points
    pv_h = torch.cat([pview, torch.ones_like(pview[:, :1])], dim=1)  # (H*W, 4)
    clip = pv_h @ proj.t()                                        # (H*W, 4); proj.t() = projection_matrix
    w = clip[:, 3]
    w = torch.where(w.abs() < 1e-6, torch.full_like(w, 1e-6), w)
    ndc_z = (clip[:, 2] / w).reshape(H, W)
    tgt_dn = (1.0 - ndc_z) / 2.0                                  # renderer depth = 1 - (ndc_z+1)/2
    valid = (dm > 0) & torch.isfinite(ndc_z)
    return tgt_dn, valid


def viewz_to_normal(viewpoint_cam, view_z, H, W):
    """
    View-space normal map (3,H,W) from a metric view-z depth map (H,W).
    """
    dev = view_z.device
    fx = W / (2 * math.tan(viewpoint_cam.FoVx / 2.))
    fy = H / (2 * math.tan(viewpoint_cam.FoVy / 2.))
    intrins_inv = torch.tensor(
        [[1 / fx, 0., -W / (2 * fx)],
         [0., 1 / fy, -H / (2 * fy)],
         [0., 0., 1.0]], device=dev).float()
    gx, gy = torch.meshgrid(torch.arange(W, device=dev) + 0.5,
                            torch.arange(H, device=dev) + 0.5, indexing='xy')
    pix = torch.stack([gx, gy, torch.ones_like(gx)], dim=0).reshape(3, -1).float()  # (3, H*W)
    rays = intrins_inv @ pix                                                          # (3, H*W)
    pts = (view_z.reshape(1, -1) * rays).reshape(3, H, W)
    dx = pts[:, 2:, 1:-1] - pts[:, :-2, 1:-1]
    dy = pts[:, 1:-1, 2:] - pts[:, 1:-1, :-2]
    n = torch.nn.functional.normalize(torch.cross(dx, dy, dim=0), dim=0)
    out = torch.zeros_like(pts)
    out[:, 1:-1, 1:-1] = n
    return out                                                                        # (3, H, W)


def compute_dmesh_soft_regularization(
    voronoi_points, delaunay_tets, delaunay_xyz_idx, gaussians, viewpoint_cam,
    target_image, config, mesh_state, target_depth=None,
):
    """
    Faithful DMesh++ differentiable mesh loss (replaces marching-tets + the depth/normal loss).

    Lambda(F) = Lambda_min(positions) * Lambda_real(realness), rendered with Lambda as per-face
    opacity through the dmesh2 soft renderer; 
    
    supervised by RGB self-distillation vs the Gaussians' render over the mesh-covered pixels. 
    Gradient flows to Gaussian geometry through BOTH Lambda_min and the vertex positions, and
    to the per-pivot realness.
    """

    # Candidate faces
    faces = mesh_state.get("dmesh_faces")
    if faces is None:
        faces = unique_delaunay_faces(delaunay_tets).int()
        mesh_state["dmesh_faces"] = faces
        mesh_state["dmesh_vcolor"] = None
        mesh_state["dmesh_nearest"] = None
        print(f"[INFO] (dmesh-soft) candidate faces: {faces.shape[0]}")
    nF = faces.shape[0]

    # Lambda_min(positions): sigmoid(alpha * minball empty-ball margin). Detached similarity transform
    # -> normalized coords for numerically-stable minball; gradient still flows to positions.
    with torch.no_grad():
        _, tf = normalize_points(voronoi_points.detach())

    pts_n = (voronoi_points - tf["center"].cuda()) * tf["scale"].cuda()
    facesL = faces.long()
    ball, stable = MB3_V0.forward(pts_n[facesL[:, 0]], pts_n[facesL[:, 1]], pts_n[facesL[:, 2]])

    # The nearest-point (excluding the face's verts) index is combinatorial/detached and drifts
    # slowly; cache it per Delaunay reset (cKDTree is CPU + O(F) so must not run every iteration).
    nearest = mesh_state.get("dmesh_nearest")
    if nearest is None or nearest.shape[0] != faces.shape[0]:
        with torch.no_grad():
            nearest = nearest_excluding_face(ball.center, facesL, pts_n.detach())
        mesh_state["dmesh_nearest"] = nearest
    sdist = torch.norm(pts_n[nearest] - ball.center, dim=-1) - ball.radius  # differentiable in pts
    lam_min = torch.sigmoid(config["dmesh_alpha"] * sdist) * stable.float().detach()

    # Lambda_real(realness): product over the 3 verts of sigmoid(occupancy logit), aligned to pivots.
    occ_logit = gaussians.get_occupancy_logit
    if delaunay_xyz_idx is not None:
        occ_logit = occ_logit[delaunay_xyz_idx]
    preal = flatten_voronoi_features(occ_logit)          # (P,), matches voronoi_points order
    lam_real = torch.sigmoid(preal)[faces.long()].prod(dim=-1)
    lam = (lam_min * lam_real).contiguous()              # Lambda(F)

    # Per-pivot colour (cached per Delaunay reset).
    vcolor = mesh_state.get("dmesh_vcolor")
    if vcolor is None or vcolor.shape[0] != voronoi_points.shape[0]:
        vcolor = pivot_base_colors(gaussians, delaunay_xyz_idx)
        mesh_state["dmesh_vcolor"] = vcolor

    with torch.no_grad():
        fL = faces.long()
        vp = voronoi_points.detach()
        frust = is_in_view_frustum(vp, viewpoint_cam)[fL].all(dim=1)
        ones = torch.ones((vp.shape[0], 1), device=vp.device)
        clip = torch.cat([vp, ones], dim=1) @ viewpoint_cam.full_proj_transform.cuda().float()
        w = clip[:, 3]
        ndc = clip[:, :2] / w.unsqueeze(-1).clamp_min(1e-6)          # (P, 2)
        in_front = (w > 1e-4)[fL].all(dim=1)
        a, b, c = ndc[fL[:, 0]], ndc[fL[:, 1]], ndc[fL[:, 2]]
        area2 = ((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0]).abs()
        non_degen = area2 > 1e-7
        finite = torch.isfinite(ndc).all(dim=1)[fL].all(dim=1)
        vis = frust & in_front & non_degen & finite
    faces_v = faces[vis].contiguous()
    lam_v = lam[vis].contiguous()
    nF_v = faces_v.shape[0]

    # Render one camera through the soft renderer with Lambda as per-face opacity.
    aa_temp = float(config.get("dmesh_aa_temperature", 1.0))
    rscale = float(config.get("dmesh_render_scale", 1.0))
    depth_w = float(config.get("dmesh_depth_weight", 0.0))
    normal_w = float(config.get("dmesh_normal_weight", 0.0))
    mv = viewpoint_cam.world_view_transform.transpose(0, 1).cuda().float()
    proj = viewpoint_cam.projection_matrix.transpose(0, 1).cuda().float()
    W, H = viewpoint_cam.image_width, viewpoint_cam.image_height
    Wr = max(1, int(round(W * rscale)))
    Hr = max(1, int(round(H * rscale)))
    renderer = _DMeshRenderer(mv[None], proj[None], Wr, Hr, device="cuda")
    fintense = torch.ones((1, nF_v), device="cuda")
    patch_min = torch.zeros((1, 2), dtype=torch.int32, device="cuda")
    bg = torch.zeros(3, device="cuda")

    color, depth = renderer.forward([0], patch_min, Wr, Hr, voronoi_points.contiguous(),
                                    faces_v, vcolor, lam_v, fintense, bg, aa_temp)
    mesh_img = color[0]                                   # (Hr, Wr, 3)
    mesh_depth = depth[0].squeeze()                       # (Hr, Wr), normalized 1-(ndc_z+1)/2

    # RGB self-distillation
    tgt = target_image.clamp(0, 1).detach()              # (3, H, W)
    if (Hr, Wr) != (H, W):
        tgt = torch.nn.functional.interpolate(tgt[None], size=(Hr, Wr), mode="bilinear", align_corners=False)[0]
    tgt = tgt.permute(1, 2, 0)                            # (Hr, Wr, 3)
    cover_full = (mesh_depth > 0).detach()               # (legacy: uncovered sentinel is 0.5 -> ~all pixels)
    l1 = (mesh_img - tgt).abs().mean(dim=-1)
    rgb_loss = (l1 * cover_full).sum() / cover_full.sum().clamp(min=1)
    mesh_loss = config["dmesh_rgb_weight"] * rgb_loss

    # Depth self-distillation 
    depth_loss = torch.zeros((), device=mesh_loss.device)
    if depth_w > 0.0 and target_depth is not None:
        with torch.no_grad():
            tgt_dn, valid = target_depth_in_renderer_space(viewpoint_cam, target_depth, proj)
            if (Hr, Wr) != (H, W):
                tgt_dn = torch.nn.functional.interpolate(tgt_dn[None, None], size=(Hr, Wr), mode="nearest")[0, 0]
                valid = torch.nn.functional.interpolate(valid[None, None].float(), size=(Hr, Wr), mode="nearest")[0, 0] > 0.5
        mesh_covered = (mesh_depth - 0.5).abs() > 1e-4
        dmask = mesh_covered & valid
        depth_loss = ((mesh_depth - tgt_dn).abs() * dmask).sum() / dmask.sum().clamp(min=1)
        mesh_loss = mesh_loss + depth_w * depth_loss

    # Normal self-distillation
    normal_loss = torch.zeros((), device=mesh_loss.device)
    if normal_w > 0.0 and target_depth is not None:
        znear, zfar = float(viewpoint_cam.znear), float(viewpoint_cam.zfar)
        k = zfar / (zfar - znear)                                  # ndc_z = k*(z-znear)/z
        ndc_z_mesh = 1.0 - 2.0 * mesh_depth                        # invert renderer's (1-(ndc_z+1)/2)
        vz_mesh = k * znear / (k - ndc_z_mesh).clamp_min(1e-6)     # metric view-z, differentiable
        with torch.no_grad():
            vz_tgt = target_depth.squeeze()
            if (Hr, Wr) != (H, W):
                vz_tgt = torch.nn.functional.interpolate(
                    vz_tgt[None, None], size=(Hr, Wr), mode="nearest")[0, 0]
            n_tgt = viewz_to_normal(viewpoint_cam, vz_tgt, Hr, Wr)
            tgt_ok = (n_tgt.norm(dim=0) > 0.5) & (vz_tgt > 0)
        n_mesh = viewz_to_normal(viewpoint_cam, vz_mesh, Hr, Wr)
        mesh_covered_n = (mesh_depth - 0.5).abs() > 1e-4
        nmask = mesh_covered_n & tgt_ok
        cosim = (n_mesh * n_tgt).sum(dim=0)
        normal_loss = ((1.0 - cosim) * nmask).sum() / nmask.sum().clamp(min=1)
        mesh_loss = mesh_loss + normal_w * normal_loss

    cover = cover_full
    return {
        "mesh_loss": mesh_loss,
        "depth_loss": float(depth_loss),
        "normal_loss": float(normal_loss),
        "lam_min_on": float((lam_min > 0.5).float().mean()),
        "lam_real_mean": float(lam_real.mean()),
        "n_faces": nF,
        "n_faces_visible": nF_v,
        "coverage": float(cover.float().mean()),
    }


def centroid_band_faces(
    tets: torch.Tensor,
    pivot_tsdf: torch.Tensor,
    pool_band: float,
    cut: float,
) -> torch.Tensor:
    """
    Face-centroid TSDF band extractor for mesh-in-the-loop.
    """
    tets = tets.long()
    # Process the 4 triangular faces of each tet sequentially and keep only survivors, to avoid
    # materialising all ~4*N_tets faces at once inside the training loop.
    kept = []
    for combo in ([0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]):
        f = tets[:, combo]              # (N_tets, 3)
        tf = pivot_tsdf[f]             # (N_tets, 3)
        keep = (tf.abs() < pool_band).all(dim=1) & (tf.mean(dim=1).abs() < cut)
        kept.append(f[keep])
    faces = torch.cat(kept, dim=0)
    faces = torch.sort(faces, dim=1)[0]
    faces = torch.unique(faces, dim=0)  # dedup faces shared by adjacent tets
    return faces


def initialize_mesh_regularization(
    scene: Scene, 
    config: Dict[str, Any], 
) -> Tuple[MeshRenderer, Dict[str, Any]]:
    """
    Initializes components required for mesh regularization.

    Args:
        scene: The scene object containing training cameras.
        config: Configuration dictionary for mesh regularization.

    Returns:
        A dictionary containing initialized components:
        - mesh_renderer: The initialized MeshRenderer.
        - state: A dictionary to hold mesh regularization state variables.
    """
    if cpp is None:
        raise ImportError("Mesh regularization requires 'tetranerf.utils.extension.cpp'. Please ensure it's compiled.")

    print("[INFO] Mesh regularization enabled.")
    print(f"         > Mesh depth loss type: {config['mesh_depth_loss_type']}")
    print(f"         > Occupancy mode: {config['occupancy_mode']}")
        
    if config.get("use_dmesh_soft_mesh", False):
        print("[INFO] DMesh-soft meshing: skipping nvdiffrast MeshRasterizer (uses dmesh2 renderer).")
        mesh_renderer = None
    else:
        mesh_rasterizer = MeshRasterizer(cameras=scene.getTrainCameras().copy())
        if config["use_scalable_renderer"]:
            print("[INFO] Using scalable mesh renderer.")
            mesh_renderer = ScalableMeshRenderer(mesh_rasterizer)
        else:
            mesh_renderer = MeshRenderer(mesh_rasterizer)

    # Initialize state dictionary
    mesh_state = {
        "delaunay_tets": None,
        "voronoi_occupancy_labels": None,
        "delaunay_xyz_idx": None,
        "surface_delaunay_xyz_idx": None,
        "reset_delaunay_samples": True,
        "reset_sdf_values": True,
    }

    return mesh_renderer, mesh_state


def compute_mesh_regularization(
    iteration: int,
    render_pkg: Dict[str, torch.Tensor],
    viewpoint_cam: Camera,
    viewpoint_idx: int,
    gaussians: GaussianModel,
    scene: Scene,
    pipe: PipelineParams,
    background: torch.Tensor,
    kernel_size: float,
    config: Dict[str, Any],
    mesh_renderer: MeshRenderer,
    mesh_state: Dict[str, Any],
    render_func: Callable,
    weight_adjustment: float=0.005,
    args: Any=None,
    integrate_func:Callable=None,
) -> Dict[str, Any]:
    """
    Computes the mesh regularization loss and updates the mesh state.

    Args:
        iteration: Current training iteration.
        render_pkg: Dictionary containing rendering results.
            - render: The rendered image.
            - median_depth: The median depth of the rendered image.
            - expected_depth: The expected depth of the rendered image.
        viewpoint_cam: The current viewpoint camera.
        viewpoint_idx: Index of the current viewpoint camera.
        gaussians: The GaussianModel object.
        scene: The scene object.
        pipe: Pipeline parameters.
        background: Background color tensor.
        kernel_size: Kernel size for rendering.
        config: Configuration dictionary for mesh regularization.
        mesh_renderer: The MeshRenderer object.
        mesh_state: Dictionary holding the current state of mesh regularization.
        render_func: Function to render the scene. 
            Takes as input: 
            - A camera viewpoint_camera
            - A GaussianModel pc
            - A pipeline pipe
            - A background color bg_color
            Returns a dictionary with the following keys:
            - render: The rendered image.
            - median_depth: The median depth of the rendered image.

    Returns:
        A dictionary containing:
        - mesh_loss: The computed total mesh regularization loss (torch.Tensor).
        - mesh_depth_loss: The depth component of the mesh loss (torch.Tensor).
        - mesh_normal_loss: The normal component of the mesh loss (torch.Tensor).
        - updated_state: The updated mesh_state dictionary.
        - mesh_render_pkg: Dictionary containing mesh rendering results (depth, normals).
        - voronoi_points_count: Number of points used for Delaunay triangulation.
    """

    lambda_mesh_depth = config["depth_weight"]
    lambda_mesh_normal = config["normal_weight"]

    # DMesh++ meshing replaces marching-tetrahedra: instead of extracting the isosurface of a
    # learnable SDF, the mesh IS the face-centroid-band selection of the Delaunay faces of the
    # pivots (verts = pivots, so the render loss still flows to Gaussian geometry)
    use_cband = config.get("use_centroid_band_mesh", False)
    # Faithful DMesh++ differentiable meshing: soft Lambda(F) rendered with per-face opacity, RGB
    # self-distillation loss, gradient to positions (via Lambda_min + vertex pos) and realness.
    use_dmesh_soft = config.get("use_dmesh_soft_mesh", False)

    # --- State Management ---
    # For filtering Delaunay points
    delaunay_xyz_idx = mesh_state["delaunay_xyz_idx"]  # (N_voronoi_Gaussians,)
    # For SDF regularization by occupancy labels
    voronoi_occupancy_labels = mesh_state["voronoi_occupancy_labels"]  # (N_voronoi_points, )
    # Delaunay tetrahedralization
    delaunay_tets = mesh_state["delaunay_tets"]  # (N_tets, 4)
    # Flags
    reset_delaunay_samples = mesh_state["reset_delaunay_samples"]
    reset_sdf_values = mesh_state["reset_sdf_values"]
    # For logging
    voronoi_points_count = 0
    
    # Used for filtering Delaunay points
    use_delaunay_downsampling = (
        (config["n_max_points_in_delaunay"] is not None)
        and (config["n_max_points_in_delaunay"] > 0)
    )

    # Check for resets based on intervals
    if iteration < config['stop_iter']:
        if iteration % config["delaunay_reset_interval"] == 0:
            delaunay_tets = None
            reset_delaunay_samples = True

        if iteration % config["sdf_reset_interval"] == 0:
            reset_sdf_values = True

        # DMesh-soft has no SDF/occupancy reset, realness is trained by the render loss
        if use_dmesh_soft:
            reset_sdf_values = False

        if config["fix_set_of_learnable_sdfs"] and (iteration > config["start_iter"]):
            reset_delaunay_samples = False
            
        if (not use_cband) and (config["learnable_sdf_reset_mode"] == "none") and (iteration > config["start_iter"]):
            reset_sdf_values = False  # TODO: Maybe not needed?

        if (not use_cband) and (iteration >= config["learnable_sdf_reset_stop_iter"]):
            assert iteration > config["start_iter"]
            reset_sdf_values = False
    else:
        print(f"[INFO] Stopping mesh regularization at iteration {iteration}.")
        print(f"          > Skipping Delaunay and SDF resets for last iterations.")

    if delaunay_tets is None:
        print(f"[INFO] Resetting Delaunay state at iteration {iteration}.")
    if reset_sdf_values:
        print(f"[INFO] Resetting SDF state at iteration {iteration}.")

    # Start mesh regularization logic
    if iteration == config["start_iter"]:
        print("[INFO] Starting mesh regularization at iteration {}".format(iteration))
        print(f"          > Spatial scale for mesh depth loss: {gaussians.spatial_lr_scale}")
        print(f"          > Use Delaunay downsampling: {use_delaunay_downsampling}")
        print(f"          > Use foreground culling: {config['radius_culling'] > 0.0}")
        assert gaussians.spatial_lr_scale > 0.

        # Force resets on the first iteration
        delaunay_tets = None
        reset_sdf_values = True
        reset_delaunay_samples = True

        gaussians.set_occupancy_mode(config["occupancy_mode"])
        print(f"          > Occupancy mode: {gaussians._occupancy_mode}")
        print(f"          > Filter large edges: {config['filter_large_edges']}")
        print(f"          > Fixing set of learnable SDFs: {config['fix_set_of_learnable_sdfs']}")
        print(f"          > Method to reset SDF: {config['method_to_reset_sdf']}")
        print(f"          > Number of binary steps to reset SDF: {config['n_binary_steps_to_reset_sdf']}")
        print(f"          > Number of linearization steps to reset SDF: {config['sdf_reset_linearization_n_steps']}")
        print(f"          > Learnable SDF reset mode: {config['learnable_sdf_reset_mode']}")
        if config["learnable_sdf_reset_mode"] == "ema":
            print(f"             > Learnable SDF reset alpha EMA: {config['learnable_sdf_reset_alpha_ema']}")
        print(f"          > Enforcing occupied centers: {config['enforce_occupied_centers']}")
        print(f"          > Using occupancy labels loss: {config['use_occupancy_labels_loss']}")
        if config["use_occupancy_labels_loss"]:
            print(f"             > Reset occupancy labels every: {config['reset_occupancy_labels_every']}")
        print(f"          > Initializing SDF values by integrating Gaussian occupancy values...")
        # TODO: Do a reset here?

    # --- Main Mesh Regularization Computation ---
    if iteration >= config["start_iter"]:
        # Reset Delaunay samples if needed
        reset_occupancy_labels_for_new_delaunay_sites = False
        if use_delaunay_downsampling:
            if reset_delaunay_samples:
                n_gaussians_to_sample_from = gaussians._xyz.shape[0]
                    
                if config["radius_culling"] > 0.0:
                    cam_spatial_extent = get_cameras_spatial_extent(scene.getTrainCameras().copy())
                    delaunay_radius = cam_spatial_extent["radius"]
                    delaunay_center = cam_spatial_extent["avg_cam_center"].view(1, 3)            
                    with torch.no_grad():
                        delaunay_sampling_radius_mask = (
                            (gaussians._xyz - delaunay_center).norm(dim=-1) <= delaunay_radius
                        ).view(-1)
                    n_gaussians_to_sample_from = int(delaunay_sampling_radius_mask.sum().item())
                else:
                    delaunay_sampling_radius_mask = None

                n_max_gaussians_for_delaunay = int(config["n_max_points_in_delaunay"] / 9.)
                downsample_gaussians_for_delaunay = n_max_gaussians_for_delaunay < n_gaussians_to_sample_from

                if downsample_gaussians_for_delaunay:
                    print(f"[INFO] Downsampling Delaunay Gaussians from {n_gaussians_to_sample_from} to {n_max_gaussians_for_delaunay}.")                        
                    if config["delaunay_sampling_method"] == "random":
                        delaunay_xyz_idx = torch.randperm(
                            n_gaussians_to_sample_from, device="cuda"
                        )[:n_max_gaussians_for_delaunay]
                    elif config["delaunay_sampling_method"] == "surface":
                        delaunay_xyz_idx = gaussians.sample_surface_gaussians(
                            scene=scene,
                            render_simp=render_simp,
                            iteration=iteration,
                            args=args,
                            pipe=pipe,
                            background=background,
                            n_samples=n_max_gaussians_for_delaunay,
                            sampling_mask=delaunay_sampling_radius_mask,
                        )
                    elif config["delaunay_sampling_method"] == "surface+opacity":
                        delaunay_xyz_idx = gaussians.sample_surface_gaussians(
                            scene=scene,
                            render_simp=render_simp,
                            iteration=iteration,
                            args=args,
                            pipe=pipe,
                            background=background,
                            n_samples=n_max_gaussians_for_delaunay,
                            sampling_mask=delaunay_sampling_radius_mask,
                        )
                        n_remaining_gaussians_to_sample = n_max_gaussians_for_delaunay - delaunay_xyz_idx.shape[0]
                        if n_remaining_gaussians_to_sample > 0:
                            mesh_state["surface_delaunay_xyz_idx"] = delaunay_xyz_idx.clone()
                            opacity_sample_mask = torch.ones(gaussians._xyz.shape[0], device="cuda", dtype=torch.bool)
                            opacity_sample_mask[delaunay_xyz_idx] = False
                            delaunay_xyz_idx = torch.cat(
                                [
                                    delaunay_xyz_idx,
                                    gaussians.sample_opacity_gaussians(
                                        n_samples=n_remaining_gaussians_to_sample,
                                        sampling_mask=opacity_sample_mask,
                                    )
                                ],
                                dim=0,
                            )
                            delaunay_xyz_idx = torch.sort(delaunay_xyz_idx, dim=0)[0]
                    else:
                        raise ValueError(f"Invalid Delaunay sampling method: {config['delaunay_sampling_method']}")
                    print(f"[INFO] Downsampled Delaunay Gaussians from {n_gaussians_to_sample_from} to {len(delaunay_xyz_idx)}.")
                    reset_occupancy_labels_for_new_delaunay_sites = True
                else:
                    if delaunay_sampling_radius_mask is not None:
                        delaunay_xyz_idx = torch.where(delaunay_sampling_radius_mask)[0]
                        print(f"[INFO] Using foreground culling with radius {config['radius_culling']}.")
                    else:
                        delaunay_xyz_idx = None
                        print(f"[INFO] No need to downsample Delaunay Gaussians.")

                torch.cuda.empty_cache()
                reset_delaunay_samples = False # Reset flag after computation
                delaunay_tets = None # If downsampling, we need to recompute the tetrahedra
        else:
            delaunay_xyz_idx = None # Ensure it's None if not used

        # Compute Voronoi generators
        # Pass delaunay_xyz_idx which might be None (use all), or indices after opacity/downsampling
        # The `opacity`/`density` WDT weight modes additionally need a pivot-aligned per-Gaussian
        # feature (opacity, or imp_score/distCUDA2) broadcast to pivots.
        wdt_mode = config.get("wdt_weight_mode") if config.get("use_weighted_triangulation", False) else None
        voronoi_opacity = None
        voronoi_density = None
        if wdt_mode == "opacity":
            voronoi_points, voronoi_scale, voronoi_opacity = gaussians.get_tetra_points(
                downsample_ratio=None, let_gradients_flow=True,
                xyz_idx=delaunay_xyz_idx, return_opacity=True,
            )
        elif wdt_mode == "density":
            # density signal = imp_score / distCUDA2 (the --prune_rule density signal)
            imp_score = gaussians.compute_importance_score(
                scene, render_simp, iteration, args, pipe, background)
            dist2 = torch.clamp_min(distCUDA2(gaussians.get_xyz.detach()), 1e-7)
            density_score = imp_score / dist2
            voronoi_points, voronoi_scale, voronoi_density = gaussians.get_tetra_points(
                downsample_ratio=None, let_gradients_flow=True,
                xyz_idx=delaunay_xyz_idx, extra_gaussian_feature=density_score,
            )
        else:
            voronoi_points, voronoi_scale = gaussians.get_tetra_points(
                downsample_ratio=None, let_gradients_flow=True,
                xyz_idx=delaunay_xyz_idx, # Pass the computed indices
            )
        voronoi_points_count = voronoi_points.shape[0]
        # Recompute Delaunay tetrahedralization if needed
        if delaunay_tets is None:
            print(f"[INFO] Recomputing tetrahedralization for {voronoi_points.shape[0]} points...")
            with torch.no_grad():
                # Ensure points are detached before passing to C++ extension.
                # Plain Delaunay unless config['use_weighted_triangulation'] (then CGAL regular/WDT).
                if config.get("use_weighted_triangulation", False):
                    delaunay_tets, n_hidden = compute_triangulation(
                        voronoi_points, config=config, pivot_scales=voronoi_scale,
                        return_num_hidden=True, pivot_opacity=voronoi_opacity,
                        pivot_density=voronoi_density,
                    )
                    print(f"[INFO] Weighted triangulation ({config.get('wdt_weight_mode')}, "
                          f"scale={config.get('wdt_weight_scale')}): {delaunay_tets.shape[0]} tets, "
                          f"{n_hidden} hidden pivots ({100.0 * n_hidden / max(voronoi_points.shape[0], 1):.1f}%).")
                else:
                    delaunay_tets = compute_triangulation(voronoi_points, config=config)
            torch.cuda.empty_cache()
            if use_cband:
                mesh_state["cband_faces"] = None  # connectivity changed -> rebuild face set
            if use_dmesh_soft:
                mesh_state["dmesh_faces"] = None  # connectivity changed -> rebuild candidate faces

        # DMesh++ differentiable meshing: build soft Lambda mesh, render, RGB self-distill loss,
        # and return early 
        if use_dmesh_soft:
            z = torch.zeros(size=(), device=gaussians._xyz.device)
            soft_interval = int(config.get("dmesh_soft_interval", 1))
            is_soft_iter = (
                (soft_interval <= 1)
                or (iteration % soft_interval == 0)
                or (iteration == config["start_iter"])
                or reset_delaunay_samples
                or (delaunay_tets is not None and mesh_state.get("dmesh_faces") is None)
            )
            if is_soft_iter:
                mesh_state["_dmesh_iter"] = iteration
                soft = compute_dmesh_soft_regularization(
                    voronoi_points=voronoi_points,
                    delaunay_tets=delaunay_tets,
                    delaunay_xyz_idx=delaunay_xyz_idx,
                    gaussians=gaussians,
                    viewpoint_cam=viewpoint_cam,
                    target_image=render_pkg["render"],
                    target_depth=render_pkg.get("median_depth"),
                    config=config,
                    mesh_state=mesh_state,
                )
                soft_mesh_loss = soft["mesh_loss"]
                if iteration % 200 == 0:
                    print(f"[INFO] (dmesh-soft) it {iteration}: loss {soft['mesh_loss'].item():.5f}, "
                          f"depth_loss {soft.get('depth_loss', 0.0):.5f}, "
                          f"normal_loss {soft.get('normal_loss', 0.0):.5f}, "
                          f"faces {soft['n_faces_visible']}/{soft['n_faces']} vis, "
                          f"Lambda_min>0.5 {100*soft['lam_min_on']:.1f}%, "
                          f"Lambda_real mean {soft['lam_real_mean']:.3f}, coverage {100*soft['coverage']:.1f}%")
            else:
                soft_mesh_loss = z  # skipped iter: no mesh gradient this step
            mesh_state["delaunay_xyz_idx"] = delaunay_xyz_idx
            mesh_state["delaunay_tets"] = delaunay_tets
            mesh_state["reset_delaunay_samples"] = reset_delaunay_samples
            mesh_state["reset_sdf_values"] = reset_sdf_values
            return {
                "mesh_loss": soft_mesh_loss,
                "mesh_depth_loss": z,
                "mesh_normal_loss": z,
                "occupied_centers_loss": z,
                "occupancy_labels_loss": z,
                "updated_state": mesh_state,
                "mesh_render_pkg": {
                    "depth": torch.zeros(viewpoint_cam.image_height, viewpoint_cam.image_width),
                    "normals": torch.zeros(viewpoint_cam.image_height, viewpoint_cam.image_width, 3),
                },
                "voronoi_points_count": voronoi_points_count,
            }

        # --- Compute SDF values ---
        # Check if an SDF reset has to be enforced because of a shape mismatch
        if not reset_sdf_values:
            n_voronoi_sdf = voronoi_points.shape[0]
            if n_voronoi_sdf != voronoi_points.shape[0]:
                print(f"[WARNING] Delaunay SDFs ({n_voronoi_sdf}) and points ({voronoi_points.shape[0]}) count mismatch. Resetting SDFs.")
                reset_sdf_values = True
        
        if reset_sdf_values and use_cband:
            # DMesh++ meshing: no learnable SDF. Just fuse the per-pivot depth-fusion TSDF
            # that drives the face-centroid band, cache it, and force a face rebuild.
            with torch.no_grad():
                pivot_tsdf = evaluate_sdf_values_depth_fusion(
                    points=voronoi_points,
                    views=scene.getTrainCameras().copy(),
                    masks=None,
                    gaussians=gaussians,
                    pipeline=pipe,
                    background=background,
                    kernel_size=kernel_size,
                    return_colors=False,
                    trunc_margin=None,
                    render_func=render_func,
                )
                mesh_state["pivot_tsdf"] = pivot_tsdf.detach().float()
                mesh_state["cband_faces"] = None
                print(f"[INFO] (cband) Re-fused per-pivot TSDF for {voronoi_points.shape[0]} pivots.")
            reset_sdf_values = False

        if reset_sdf_values:
            with torch.no_grad():
                # Get base occupancy values for all voronoi points
                if config["method_to_reset_sdf"] == 'integration':
                    sdf_function = partial(
                        evaluate_cull_sdf_values,
                        views=scene.getTrainCameras().copy(), 
                        masks=None, 
                        gaussians=gaussians, 
                        pipeline=pipe, 
                        background=background, 
                        kernel_size=kernel_size, 
                        return_colors=False, 
                        isosurface_value=config["sdf_default_isosurface"], 
                        transform_sdf_to_linear_space=config["transform_sdf_to_linear_space"], 
                        min_occupancy_value=config["min_occupancy_value"],
                        integrate_func=integrate_func,
                    )
                    
                elif config["method_to_reset_sdf"] == 'depth_fusion':                        
                    sdf_function = partial(
                        evaluate_sdf_values_depth_fusion,
                        views=scene.getTrainCameras().copy(), 
                        masks=None, 
                        gaussians=gaussians, 
                        pipeline=pipe, 
                        background=background, 
                        kernel_size=kernel_size, 
                        return_colors=False,
                        trunc_margin=None, 
                        render_func=render_func,
                    )
                
                # Compute and linearize initial occupancy values with binary search if needed
                base_occupancy = compute_initial_sdf_with_binary_search(
                    voronoi_points=voronoi_points,
                    voronoi_scales=voronoi_scale,
                    delaunay_tets=delaunay_tets,
                    sdf_function=sdf_function,
                    n_binary_steps=config["n_binary_steps_to_reset_sdf"],
                    n_linearization_steps=config["sdf_reset_linearization_n_steps"],
                    enforce_std=config["sdf_reset_linearization_enforce_std"] if config["n_binary_steps_to_reset_sdf"] > 0 else None,
                )  # Between -1 and 1
                base_occupancy = convert_sdf_to_occupancy(base_occupancy)  # Between 0.005 and 0.995
                
                # Reshape base occupancy to make it (N_sampled_gaussians, 9)
                base_occupancy = unflatten_voronoi_features(
                    base_occupancy, 
                    n_voronoi_per_gaussians=9
                )  # (N_sampled_gaussians, 9)
                
                # Logic for resetting occupancy values
                if config["learnable_sdf_reset_mode"] == "ema":
                    print(f"[INFO] Resetting learnable SDF with EMA.")
                    _n_ema = (gaussians._base_occupancy[delaunay_xyz_idx] != 0.).sum().item()
                    _n_voronoi = base_occupancy.view(-1).shape[0]
                    print(f"          > Number of points to reset with EMA: {_n_ema}/{_n_voronoi}")
                    sdf_alpha_ema = config["learnable_sdf_reset_alpha_ema"]
                    new_occupancy =  torch.where(
                        gaussians._base_occupancy[delaunay_xyz_idx] != 0.,  # Points that have been sampled before
                        (sdf_alpha_ema * base_occupancy 
                            + (1. - sdf_alpha_ema) * gaussians.get_occupancy[delaunay_xyz_idx]),
                        base_occupancy,
                    ).clamp(min=0.005, max=0.995)
                    
                elif config["learnable_sdf_reset_mode"] == "none":
                    new_occupancy = None
                    
                else:
                    raise ValueError(f"Invalid learnable SDF reset mode: {config['learnable_sdf_reset_mode']}")
                
                # Reset occupancy values for the sampled gaussians
                gaussians.reset_occupancy(
                    base_occupancy=base_occupancy, 
                    occupancy=new_occupancy,
                    gaussian_idx=delaunay_xyz_idx, 
                )
                
                # Clear cache
                torch.cuda.empty_cache()
                gc.collect()
            
                reset_sdf_values = False # Reset flag after computation
        
        if use_cband:
            # --- DMesh++ meshing (replaces Marching Tetrahedra) ---
            # verts ARE the pivots (differentiable w.r.t. Gaussian geometry); faces are the
            # Delaunay faces kept by the face-centroid TSDF band. The selection is a hard,
            # periodically-recomputed mask cached in mesh_state (rebuilt on Delaunay/TSDF reset),
            # exactly like the tet connectivity. The render loss below flows verts -> Gaussians.
            verts = voronoi_points
            cband_faces = mesh_state.get("cband_faces", None)
            if cband_faces is None:
                with torch.no_grad():
                    cband_faces = centroid_band_faces(
                        delaunay_tets,
                        mesh_state["pivot_tsdf"],
                        pool_band=config.get("centroid_pool_band", 1.2),
                        cut=config.get("centroid_band_cut", 0.4),
                    )
                mesh_state["cband_faces"] = cband_faces
                print(f"[INFO] (cband) Built {cband_faces.shape[0]} faces "
                      f"(pool {config.get('centroid_pool_band', 1.2)}, cut {config.get('centroid_band_cut', 0.4)}).")
            faces = cband_faces
        else:
            # Convert learnable occupancy values to SDF
            if delaunay_xyz_idx is not None:
                current_occupancy = gaussians.get_occupancy[delaunay_xyz_idx]  # (N_sampled_gaussians, 9)
            else:
                current_occupancy = gaussians.get_occupancy  # (N_gaussians, 9)
            current_voronoi_sdf = convert_occupancy_to_sdf(
                flatten_voronoi_features(current_occupancy)
            )  # (N_voronoi_points, )

            # --- Marching Tetrahedra ---
            verts_list, scale_list, faces_list, _ = marching_tetrahedra(
                vertices=voronoi_points[None],
                tets=delaunay_tets,
                sdf=current_voronoi_sdf.reshape(1, -1), # Use the computed SDF for this iteration
                scales=voronoi_scale[None]
            )
            end_points, end_sdf = verts_list[0]  # (N_verts, 2, 3) and (N_verts, 2, 1)
            end_scales = scale_list[0]  # (N_verts, 2, 1)

            norm_sdf = end_sdf.abs() / end_sdf.abs().sum(dim=1, keepdim=True)
            verts = end_points[:, 0, :] * norm_sdf[:, 1, :] + end_points[:, 1, :] * norm_sdf[:, 0, :]
            faces = faces_list[0]  # (N_faces, 3)

        # --- Filtering ---
        # Frustum filtering
        faces_mask = is_in_view_frustum(verts, viewpoint_cam)[faces].any(axis=1)
        
        # GOF filtering for large edges (marching-tets only; cband has no tet end-points)
        if (not use_cband) and (config["filter_large_edges"] or config["collapse_large_edges"]):
            dmtet_distance = torch.norm(end_points[:, 0, :] - end_points[:, 1, :], dim=-1)
            dmtet_scale = end_scales[:, 0, 0] + end_scales[:, 1, 0]
            dmtet_vertex_mask = (dmtet_distance <= dmtet_scale)
            
        if (not use_cband) and config["filter_large_edges"]:
            dmtet_face_mask = dmtet_vertex_mask[faces].all(axis=1)
            faces_mask = faces_mask & dmtet_face_mask

        if (not use_cband) and config["collapse_large_edges"]:
            min_end_points = end_points[
                np.arange(end_points.shape[0]), 
                end_sdf.argmin(dim=1).flatten().cpu().numpy()
            ]  # TODO: Do the computation only for filtered vertices
            verts = torch.where(dmtet_vertex_mask[:, None], verts, min_end_points)

        # --- Build and Render Mesh ---
        mesh = Meshes(verts=verts, faces=faces[faces_mask])

        mesh_render_pkg = mesh_renderer(
            mesh,
            cam_idx=viewpoint_idx,
            return_depth=config["use_depth_loss"],
            return_normals=config["use_normal_loss"],
            use_antialiasing=True,
        )
        mesh_depth = (
            mesh_render_pkg["depth"].squeeze() 
            if config["use_depth_loss"] 
            else torch.zeros(viewpoint_cam.image_height, viewpoint_cam.image_width)
        )  # (H, W)
        mesh_normal_view = (
            mesh_render_pkg["normals"].squeeze() @ viewpoint_cam.world_view_transform[:3,:3] 
            if config["use_normal_loss"] 
            else torch.zeros(viewpoint_cam.image_height, viewpoint_cam.image_width, 3)
        )  # (H, W, 3)
        rasterization_mask = mesh_depth > 0.  # (H, W)
        
        # Reset occupancy labels
        if (
            config["use_occupancy_labels_loss"] 
            and (
                (iteration % config["reset_occupancy_labels_every"] == 0)  # Every N iterations
                or (iteration == config["start_iter"])  # First iteration
                or reset_occupancy_labels_for_new_delaunay_sites  # If not fixing sites and downsampling, compute labels for sampled sites
            )
        ):
            print(f"[INFO] Resetting occupancy labels at iteration {iteration}.")
            voronoi_occupancy_labels, _ = evaluate_mesh_occupancy(
                points=voronoi_points,
                views=scene.getTrainCameras().copy(),
                mesh=Meshes(verts=verts, faces=faces),
                masks=None,
                return_colors=True,
                use_scalable_renderer=config["use_scalable_renderer"],
            )
            print(f"[INFO] Points with label > 0.5: {torch.sum(voronoi_occupancy_labels > 0.5) / voronoi_occupancy_labels.numel()}")

        # --- Compute Losses ---
        mesh_depth_ratio = config["depth_ratio"]
        
        # Mesh Depth Loss
        if config["use_depth_loss"]:
            gaussians_depth = (
                (1. - mesh_depth_ratio) * render_pkg["expected_depth"] 
                + mesh_depth_ratio * render_pkg["median_depth"]
            ).squeeze()  # (H, W)
            
            if config["mesh_depth_loss_type"] == "log":
                mesh_depth_loss = torch.log(1. + (mesh_depth - gaussians_depth).abs() / gaussians.spatial_lr_scale)  # (H, W)

            elif config["mesh_depth_loss_type"] == "normal":
                mesh_depth_loss = depth_double_to_normal(
                    viewpoint_cam,
                    mesh_depth.squeeze()[None],
                    gaussians_depth.squeeze()[None],
                )  # (2, 3, H, W)
                mesh_depth_loss = 1. - (mesh_depth_loss[0] * mesh_depth_loss[1]).sum(dim=0)  # (H, W)

            else:
                raise ValueError(f"Invalid mesh depth loss type: {config['mesh_depth_loss_type']}")
            
            mesh_depth_loss = lambda_mesh_depth * (mesh_depth_loss * rasterization_mask).mean()
        else:
            mesh_depth_loss = torch.zeros(size=(), device=gaussians._xyz.device)

        # Mesh Normal Loss
        if config["use_normal_loss"]:
            if config["use_depth_normal"]:
                # Compute normals from Gaussian depth map
                depth_middepth_normal = depth_double_to_normal(
                    viewpoint_cam,
                    render_pkg["expected_depth"],
                    render_pkg["median_depth"]
                )
                gaussians_normal_view = (
                    (1. - mesh_depth_ratio) * depth_middepth_normal[0]
                    + mesh_depth_ratio * depth_middepth_normal[1]
                ).permute(1, 2, 0) # (H, W, 3)
            else:
                # Use rendered normals directly (already in view space)
                gaussians_normal_view = render_pkg["normal"].permute(1, 2, 0)  # (H, W, 3)

            # Compute cosine similarity loss (1 - |dot_product|).
            #
            # To do this, we flip the mesh normals to make the loss invariant to the sign of the mesh normal.
            # Indeed, we just want to make sure the planes of both the mesh and the gaussians are aligned,
            # so we don't really care about the direction of the normal.
            #
            # This might be needed in scenarios where the Delaunay triangulation is not updated for a while,
            # so that the mesh could self-intersect and have flipped normals.
            #
            # For computing the loss, we just need to use .abs() on the dot product.
            # We also explicitly flip the mesh normals for logging purposes.

            normal_dot_product = (mesh_normal_view * gaussians_normal_view).sum(dim=-1, keepdim=True)  # (H, W, 1)
            mesh_normal_loss = 1. - normal_dot_product.abs()  # (H, W, 1)
            mesh_normal_loss = lambda_mesh_normal * (mesh_normal_loss * rasterization_mask.unsqueeze(-1)).mean()
        else:
            mesh_normal_loss = torch.zeros(size=(), device=gaussians._xyz.device)
            
        # Enforce occupied centers
        if config["enforce_occupied_centers"]:
            # Get sdf values for centers of sampled Gaussians
            if mesh_state["surface_delaunay_xyz_idx"] is not None:
                gaussians_occupancy = gaussians.get_occupancy[mesh_state["surface_delaunay_xyz_idx"]]  # (N_surface_gaussians, 9)
                gaussians_occupancy = gaussians_occupancy[:, -1]  # (N_surface_gaussians, )
            else:
                gaussians_occupancy = current_occupancy[:, -1]
            occupied_centers_loss = config["occupied_centers_weight"] * (config["sdf_default_isosurface"] - gaussians_occupancy).clamp(min=0.).mean()
        else:
            occupied_centers_loss = torch.zeros(size=(), device=gaussians._xyz.device)
            
        # Occupancy labels loss
        if config["use_occupancy_labels_loss"]:
            occupancy_labels_loss = config["occupancy_labels_loss_weight"] * (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    flatten_voronoi_features(
                        gaussians.get_occupancy_logit if delaunay_xyz_idx is None
                        else gaussians.get_occupancy_logit[delaunay_xyz_idx]
                    ),
                    voronoi_occupancy_labels
                )
            ) * (voronoi_occupancy_labels > 0.5).float()
            occupancy_labels_loss = occupancy_labels_loss.mean()
        else:
            occupancy_labels_loss = torch.zeros(size=(), device=gaussians._xyz.device)

    # --- Return Results ---
    total_mesh_loss = (
        mesh_depth_loss 
        + mesh_normal_loss 
        + occupied_centers_loss 
        + occupancy_labels_loss
    )
    
    # --- Update State ---
    # Store updated filter parameters
    mesh_state["delaunay_xyz_idx"] = delaunay_xyz_idx
    # Store updated occupancy labels
    mesh_state["voronoi_occupancy_labels"] = voronoi_occupancy_labels
    # Store Updated Delaunay tetrahedra
    mesh_state["delaunay_tets"] = delaunay_tets
    # Reset flags were potentially set back to False inside the logic
    mesh_state["reset_delaunay_samples"] = reset_delaunay_samples
    mesh_state["reset_sdf_values"] = reset_sdf_values

    return {
        "mesh_loss": total_mesh_loss,
        "mesh_depth_loss": mesh_depth_loss.detach(),
        "mesh_normal_loss": mesh_normal_loss.detach(),
        "occupied_centers_loss": occupied_centers_loss.detach(),
        "occupancy_labels_loss": occupancy_labels_loss.detach(),
        "updated_state": mesh_state,
        "mesh_render_pkg": {
            "depth": mesh_depth,
            "normals": mesh_normal_view,
        }, # Contains depth/normals for logging
        "voronoi_points_count": voronoi_points_count,
    }


def reset_mesh_state_at_next_iteration(mesh_state):
    mesh_state["reset_delaunay_samples"] = True
    mesh_state["reset_sdf_values"] = True
    mesh_state["delaunay_tets"] = None
    return mesh_state
