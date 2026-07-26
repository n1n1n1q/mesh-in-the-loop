import os
import time
import random
from argparse import ArgumentParser

import numpy as np
import torch
import trimesh

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render_simp
from scene import Scene
from regularization.sdf.depth_fusion import evaluate_sdf_values as compute_sdf_with_depth_fusion

from diffmesh import (
    normalize_points,
    denormalize_points,
    extract_pivot_realness,
    dt_faces_from_points,
    select_real_faces,
)


def load_gaussians(dataset, iteration):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    gaussians.load_ply(
        os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    )
    return gaussians, scene


def extract_points_and_tsdf(dataset, iteration, pipe, args, render):
    gaussians, scene = load_gaussians(dataset, iteration)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print("[INFO] Computing per-Gaussian importance score...")
    t0 = time.time()
    imp_score = gaussians.compute_importance_score(scene, render_simp, iteration, args, pipe, background)
    print(f"[INFO] imp_score: {gaussians.get_xyz.shape[0]} Gaussians in {time.time() - t0:.1f}s.")

    pivots, _, _ = extract_pivot_realness(
        gaussians, imp_score, max_pivots=args.max_pivots, psi_from="rank", pivot_mode=args.pivot_mode,
    )

    print("[INFO] Computing depth-fusion TSDF per pivot...")
    t0 = time.time()
    tsdf = compute_sdf_with_depth_fusion(
        points=pivots, views=scene.getTrainCameras().copy(), masks=None, gaussians=gaussians,
        pipeline=pipe, background=background, kernel_size=dataset.kernel_size,
        return_colors=False, trunc_margin=None, render_func=render,
    ).reshape(-1)
    q = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], device=tsdf.device)
    print(f"[INFO] TSDF quantiles {torch.quantile(tsdf, q).tolist()} in {time.time() - t0:.1f}s.")

    psi_vertex = torch.exp(-((tsdf / args.psi_band) ** 2))  # vertex near-surface prior (vertex-AND)
    pivots_norm, transform = normalize_points(pivots, domain=1.0, margin=0.95)
    return pivots_norm.cuda(), tsdf.cuda(), psi_vertex.cuda(), transform


def write_mesh(pivots_norm, faces, transform, model_path, out_name):
    verts_world = denormalize_points(pivots_norm, transform)
    mesh = trimesh.Trimesh(
        vertices=verts_world.detach().cpu().numpy(),
        faces=faces.detach().cpu().numpy(), process=False,
    )
    mesh.remove_unreferenced_vertices()
    out_path = os.path.join(model_path, out_name)
    mesh.export(out_path)
    print(f"[INFO] Wrote {out_path}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces.")
    return out_path


def extract_mesh(pivots_norm, tsdf, psi_vertex, transform, args):
    faces = dt_faces_from_points(pivots_norm)
    if args.centroid_pool_band is not None:  # drop faces with any wildly-off vertex first
        pool = torch.exp(-((tsdf / args.centroid_pool_band) ** 2)) > args.psi_thresh
        faces = select_real_faces(faces, pool)
    tface = tsdf[faces.long()].mean(dim=-1).abs()
    real_faces = faces[tface < args.centroid_cut]
    print(f"[INFO] Delaunay {faces.shape[0]} (pool-filtered) -> centroid |tsdf|<{args.centroid_cut}: "
            f"{real_faces.shape[0]} faces.")
    return write_mesh(pivots_norm, real_faces, transform, args.model_path, args.out_name)


if __name__ == "__main__":
    parser = ArgumentParser(description="DMesh++ static mesh extraction (on-surface Delaunay faces)")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=18000, type=int)
    parser.add_argument("--rasterizer", default="radegs", type=str, choices=["radegs", "gof"])
    parser.add_argument("--imp_metric", default="outdoor", type=str, choices=["indoor", "outdoor"])
    parser.add_argument("--warn_until_iter", default=3000, type=int)
    parser.add_argument("--max-pivots", dest="max_pivots", default=3000000, type=int)
    parser.add_argument("--psi-band", dest="psi_band", default=1.1, type=float,
                        help="Band for the vertex realness prior psi=exp(-(tsdf/band)^2); 1.1 = static peak.")
    parser.add_argument("--psi-thresh", dest="psi_thresh", default=0.5, type=float)
    parser.add_argument("--pivot-mode", dest="pivot_mode", default="full", choices=["full", "centers"])
    parser.add_argument("--centroid-cut", dest="centroid_cut", default=0.4, type=float,
                        help="Keep faces with |mean signed tsdf| < this (face-select=centroid). Peak ~0.4.")
    parser.add_argument("--centroid-pool-band", dest="centroid_pool_band", default=1.2, type=float,
                        help="Pre-filter to the vertex-band pool before the centroid test (helps slightly).")
    parser.add_argument("--out-name", dest="out_name", default="mesh_dmeshpp.ply", type=str)
    args = get_combined_args(parser)
    print("[INFO] Model: " + args.model_path)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

    if args.rasterizer == "radegs":
        from gaussian_renderer.radegs import render_radegs as render
    else:
        from gaussian_renderer.gof import render_gof as render

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    pivots_norm, tsdf, psi_vertex, transform = extract_points_and_tsdf(
        dataset, args.iteration, pipe, args, render
    )
    args.model_path = dataset.model_path

    extract_mesh(pivots_norm, tsdf, psi_vertex, transform, args)
