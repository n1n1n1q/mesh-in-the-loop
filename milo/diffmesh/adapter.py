"""Pivot <-> DMesh++ point-set adapter (P0).

Maps MILo's Gaussian pivots into the DMesh++ world and back:
  * `normalize_points` / `denormalize_points` - fit pivots into DMesh++'s [-1, 1]^3 domain
    (`dmesh2/input/common.py: DOMAIN = 1.0`) with an invertible uniform similarity transform.
  * `rank` - rank-normalize a per-pivot signal (e.g. imp_score) to [0, 1]; this seeds realness psi.
    (DMesh++'s 3D pipeline has NO per-point weight; psi is the free per-point variable. See
    context/differentiable_meshing.md - "imp_score seeds realness, not a weight".)
  * `extract_pivot_realness` - extract pivots + a per-pivot imp_score-derived psi from a GaussianModel.
  * `dt_faces_from_points` / `select_real_faces` - the no-optimization DMesh++ face export: plain
    Delaunay of the points, keep triangular faces whose vertices are all "real" (this is exactly the
    hard mesh DMesh++ exports, `mvrecon_3d.py:_adjacency_check`).

The `mindiffdt` (dmesh2) import is guarded, mirroring `functional/delaunay.py`: it is only needed for
the DT export, so importing this module never breaks environments without dmesh2 built. Requires
`export LD_LIBRARY_PATH=$CONDA_PREFIX/lib` at runtime (gmp/mpfr/CGAL) - see the dmesh2 install note.
"""

from typing import Tuple, Dict, Union
import torch

from scipy.spatial import cKDTree

from mindiffdt.cgaldt import CGALDTStruct
from mindiffdt.minball import MB3_V0



def rank(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(-1).float()
    n = x.numel()
    if n <= 1:
        return torch.ones_like(x)
    order = torch.argsort(x)
    ranks = torch.empty_like(x)
    ranks[order] = torch.arange(n, device=x.device, dtype=x.dtype)
    return ranks / (n - 1)


def normalize_points(
    points: torch.Tensor, domain: float = 1.0, margin: float = 0.95
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Uniformly scale+translate `points` (P, 3) so their bbox fits inside [-domain*margin, +...]^3.

    Uniform (single) scale preserves aspect ratio, which the minball / Delaunay tests depend on.
    Returns the normalized points and an invertible transform dict {center, scale, domain}."""
    p = points.float()
    lo = p.min(dim=0).values
    hi = p.max(dim=0).values
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo).max().clamp_min(1e-8)
    scale = (float(domain) * float(margin)) / half
    pts = (p - center) * scale
    return pts, {"center": center, "scale": scale, "domain": torch.tensor(float(domain))}


def denormalize_points(points: torch.Tensor, transform: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Invert `normalize_points` - map DMesh++-space points back to world space for export/eval."""
    center = transform["center"].to(points.device)
    scale = transform["scale"].to(points.device)
    return points.float() / scale + center


@torch.no_grad()
def extract_pivot_realness(
    gaussians,
    imp_score: torch.Tensor,
    downsample_ratio: Union[float, None] = None,
    max_pivots: Union[int, None] = None,
    psi_from: str = "rank",
    pivot_mode: str = "full",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_gauss = gaussians.get_xyz.shape[0]
    per_gauss = 9 if pivot_mode == "full" else 1
    if max_pivots is not None:
        downsample_ratio = min(float(max_pivots) / (per_gauss * n_gauss), 1.0)

    scale = None
    if pivot_mode == "centers":
        xyz = gaussians.get_xyz.detach()
        pivot_imp = imp_score.detach().reshape(-1).float()
        if downsample_ratio is not None and downsample_ratio < 1.0:
            idx = torch.randperm(xyz.shape[0], device=xyz.device)[: int(xyz.shape[0] * downsample_ratio)]
            xyz, pivot_imp = xyz[idx], pivot_imp[idx]
        pivots = xyz
    elif pivot_mode == "full":
        pivots, scale, pivot_imp = gaussians.get_tetra_points(
            downsample_ratio=downsample_ratio,
            let_gradients_flow=False,
            extra_gaussian_feature=imp_score,
        )
        pivot_imp = pivot_imp.reshape(-1).float()
        scale = scale.reshape(-1).float()

    if psi_from == "rank":
        psi = rank(pivot_imp)
    elif psi_from == "raw":
        psi = pivot_imp

    # scale is the per-pivot Gaussian scale used by MILo's large-edge filter (None in centers mode).
    return pivots.contiguous(), psi, pivot_imp, scale


def dt_faces_from_points(points: torch.Tensor) -> torch.Tensor:
    dt = CGALDTStruct.forward(points.detach())
    tets = dt.dsimp_point_id.to(dtype=torch.long, device=points.device)  # (T, 4)
    combs = torch.tensor([0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3], device=points.device)
    faces = tets[:, combs].reshape(-1, 3)
    faces = torch.sort(faces, dim=-1).values
    faces = torch.unique(faces, dim=0)
    return faces


def select_real_faces(faces: torch.Tensor, real_mask: torch.Tensor) -> torch.Tensor:
    keep = real_mask[faces].all(dim=-1)
    return faces[keep]


@torch.no_grad()
def nearest_excluding_face(
    centers: torch.Tensor, faces: torch.Tensor, points: torch.Tensor, k: int = 5
) -> torch.Tensor:

    pts_np = points.detach().cpu().numpy()
    tree = cKDTree(pts_np)
    _d, nn = tree.query(centers.detach().cpu().numpy(), k=k)  # (F, k)
    nn = torch.from_numpy(nn).to(faces.device).long()
    # first neighbor not in the face; fall back to column 0 if all k collide (rare)
    is_self = (nn.unsqueeze(-1) == faces.unsqueeze(1)).any(dim=-1)  # (F, k)
    first_valid = torch.argmax((~is_self).int(), dim=-1)  # (F,)
    return nn[torch.arange(nn.shape[0], device=nn.device), first_valid]


def minball_sdist(points: torch.Tensor, faces: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    p1 = points[faces[:, 0]]
    p2 = points[faces[:, 1]]
    p3 = points[faces[:, 2]]
    ball, stable = MB3_V0.forward(p1, p2, p3)
    nearest = nearest_excluding_face(ball.center, faces, points)
    d1 = torch.norm(points[nearest] - ball.center, dim=-1, p=2)
    sdist = d1 - ball.radius
    return sdist, stable


def minball_surface_faces(
    points: torch.Tensor, faces: torch.Tensor, tol: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Static DMesh++ surface subset: keep faces whose minimum circumscribing ball is (near-)empty.
    """

    sdist, stable = minball_sdist(points, faces)
    if tol > 0.0:
        p1, p2, p3 = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
        ball, _ = MB3_V0.forward(p1, p2, p3)
        keep = (sdist > -tol * ball.radius) & stable
    else:
        keep = (sdist > 0) & stable
    return faces[keep], keep
