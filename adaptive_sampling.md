# Type-Adaptive Gaussian Pivot Sampling for Delaunay Triangulation

## Motivation

MILo extracts a Delaunay triangulation every training iteration from **Gaussian
Pivots**: each sampled Gaussian spawns 9 points — its center plus the 8 corners
of the bounding box aligned to its principal axes/scale (paper Sec. 4.1, eq. 1).
This is fixed regardless of the Gaussian's shape.

In practice, optimized Gaussians are rarely isotropic ("volumetric"). Most end up:
- **planar** (disk-like: 2 large scale components, 1 tiny) — common on flat surfaces,
- **linear** (needle-like: 1 large component, 2 tiny) — common on thin structures/edges.

For these, several of the 8 canonical corners nearly coincide (the extent along a
collapsed axis is negligible), producing redundant, near-duplicate Delaunay sites
that waste compute/memory in CGAL triangulation, Marching Tetrahedra, and mesh
rasterization. The paper's "Future Work" (Sec. 6.7) explicitly calls out "novel
adaptive sampling strategies for Delaunay sites" as an open direction.

**Idea:** classify each Gaussian as volumetric / planar / linear from its scale, and
emit fewer corners for anisotropic ones — **8 corners (volumetric), 4 (planar),
2 (linear)** — always keeping the center (so 9 / 5 / 3 total pivots). This cuts
redundant points without touching the rest of the pipeline (learnable per-corner
SDF/occupancy, triangulation, Marching Tetrahedra all keep operating on a flat
point list). Gated behind a config flag, defaulting to the legacy fixed-9 behavior.

## Results (Tanks&Temples / Truck, RaDe-GS rasterizer, `default` vs `default_adaptive`)

| Metric                        | Baseline (fixed-9) | Adaptive (9/5/3) | Change            |
|-------------------------------|--------------------|------------------|-------------------|
| Delaunay sites per iteration  | 2,924,532          | ~1,740,000       | **−40.5%**        |
| Avg pivots per Gaussian       | 9.0                | ~5.36            | reflects planar/linear-heavy scene |
| Training time (18k iters)     | 1:36:56            | 1:13:46          | **−23.8%**        |
| Final total training loss     | 0.0733             | 0.0737           | +0.5% (parity)    |
| # Gaussians (final)           | 324,948            | 329,326          | ~equal            |

Both runs completed the full 18,000 iterations with **zero NaN/errors**. The ~5.36
average pivots/Gaussian confirms the core hypothesis: the surface Gaussians in a real
scene are overwhelmingly planar/linear, so most were emitting redundant corners.

wandb project: `ets-3dgs` (entity `basystyo-ukrainian-catholic-university`).
Outputs: `output/Truck_adaptive_pivot_sampling/` and `output/Truck_baseline/`.

> Note: this is the training-loss comparison. The geometric F1 / mesh-vertex-count /
> Mesh-Based-NVS numbers (the paper's actual quality metrics) still need to be computed
> by running mesh extraction + evaluation on both checkpoints — see "Next steps".

## Design

### 1. Classification (volumetric / planar / linear)

Gaussian scale is stored as `_scaling` (log-space; `get_scaling = exp(_scaling)`).
Working in log-space avoids ratios/division: sort each Gaussian's 3 log-scales
descending (`l0 ≥ l1 ≥ l2`) and compare consecutive **gaps** against epsilon
thresholds (a gap of `eps = log(ratio)` in log-space is a `ratio`× difference in
linear scale):

```
gap_major = l0 - l1
gap_minor = l1 - l2
is_linear     = gap_major > log(linear_ratio)          # one axis dominates both others
is_planar     = (not linear) and gap_minor > log(planar_ratio)  # two axes dominate the third
is_volumetric = otherwise
```

Helper: `classify_gaussian_types(log_scale, linear_ratio, planar_ratio)` in
`milo/utils/general_utils.py` (int8 codes: 0=volumetric, 1=planar, 2=linear),
fully vectorized.

### 2. Corner keep-mask — SYMMETRIC selection (important)

The naive reduction (fix the collapsed axes' signs to +1) keeps corners all on **one
side** of the center along the short axis. That destroys the local 3D thickness that
keeps nearby tetrahedra well-conditioned and causes **sliver/degenerate tetrahedra →
NaN** (see Bug 2 below). The corners are therefore chosen to stay **symmetric through
the center** along the collapsed axes:

- **Volumetric:** all 8 corners.
- **Planar:** the 4 corners forming a regular tetrahedron inscribed in the box —
  `sign_dominant · sign_mid · sign_minor == 1`. Covers all 4 (dominant, mid) sign
  combinations, balanced 2-and-2 across the minor axis.
- **Linear:** 1 antipodal pair (all three signs equal: `(+,+,+)` and `(-,-,-)`) —
  maximally separated along the dominant axis while symmetric through the center on
  the two collapsed axes.

Helper: `compute_pivot_keep_mask(log_scale, corner_signs, linear_ratio, planar_ratio)`
→ `(N, 9)` bool mask (corners 0..7, center at index 8, always True). The sign pattern
is read from the trimesh box at runtime (`torch.sign(box.vertices)`), never hardcoded.

### 3. Emit masked pivots

`_get_tetra_points` (`milo/scene/gaussian_model.py`) and its standalone twin
`extract_gaussian_pivots` (`milo/functional/pivots.py`) build the full `(N, 9, 3)`
points and `(N, 9, 1)` scales as before, then apply the mask: `points_full[keep_mask]`
(row-major, gaussian-major). Both now **return `keep_mask`** as an extra value (None on
the legacy path). Legacy output is byte-for-byte identical when the flag is off.

### 4. Mask-aware occupancy flatten/unflatten

The learnable occupancy/SDF tensors (`_base_occupancy` / `_occupancy_shift`) stay shape
`(N, 9)`; only which slots become Delaunay sites changes. `flatten_voronoi_features` /
`unflatten_voronoi_features` (`milo/utils/geometry_utils.py`) gained an optional
`keep_mask` (and `unflatten` a `fill_value`). `keep_mask=None` = exact legacy behavior.

## Config

`use_adaptive_pivot_sampling`, `gaussian_type_linear_ratio`, `gaussian_type_planar_ratio`
were added to all `milo/configs/mesh/*.yaml` (default **off**, ratios 3.0). A dedicated
`milo/configs/mesh/default_adaptive.yaml` is a copy of `default.yaml` with the flag on.

```yaml
use_adaptive_pivot_sampling: true    # 9/5/3 corner-pivot counts for volumetric/planar/linear
gaussian_type_linear_ratio: 3.0      # log-scale gap threshold (as a ratio) for "linear"
gaussian_type_planar_ratio: 3.0      # log-scale gap threshold (as a ratio) for "planar"
```

## Files changed

- `milo/utils/general_utils.py` — `classify_gaussian_types`, `compute_pivot_keep_mask`.
- `milo/utils/geometry_utils.py` — optional `keep_mask` / `fill_value` on
  `flatten_voronoi_features` / `unflatten_voronoi_features`.
- `milo/scene/gaussian_model.py` — `_get_tetra_points`: compute/apply/return `keep_mask`,
  plus `override_keep_mask` (see Bug 1).
- `milo/functional/pivots.py` — `extract_gaussian_pivots`: same treatment; callers in
  `functional/{delaunay,mesh,sdf}.py` updated for the 3-tuple return.
- `milo/regularization/regularizer/mesh.py` — thread `keep_mask` through; cache it in
  `mesh_state`; lockstep occupancy-label refresh; `fill_value=0.5`.
- `milo/mesh_extract_sdf.py` — same threading + `fill_value=0.5`.
- `milo/mesh_extract_integration.py` — capture (ignore) the extra return value.
- `milo/configs/mesh/*.yaml` + new `default_adaptive.yaml`.

## Bugs found & fixed during bring-up

This feature interacts with MILo's *cached* per-iteration state in three non-obvious
ways. All three only manifested at/after iteration 8001 (when mesh regularization
starts) and are now covered by regression tests.

**Bug 1 — cached-topology desync (crash: CUDA assert in Marching Tetrahedra).**
The per-Gaussian corner count depends on current scale, which drifts every iteration,
but the Delaunay triangulation (`delaunay_tets`) is cached and only recomputed every
`delaunay_reset_interval` (500) iterations — it references vertices by index. Recomputing
the mask fresh each iteration desynced the point count from the cached tets → OOB index.
**Fix:** cache `voronoi_keep_mask` in `mesh_state`; pass it back via `override_keep_mask`
so topology stays frozen between triangulations (positions still update every iteration
for differentiability). `need_fresh_keep_mask = (delaunay_tets is None)`.

**Bug 2 — one-sided corner collapse → degenerate tets → NaN in mesh losses.**
The first corner-selection scheme kept reduced corners all on one side of the Gaussian's
short axis, collapsing local thickness in flat/thin regions and producing sliver
tetrahedra whose SDF-edge interpolation divides by ~0 → NaN, poisoning the loss.
**Fix:** the symmetric (antipodal / tetrahedral) selection in §2.

**Bug 3 — `inverse_sigmoid(0.0) = -inf` → NaN in occupancy-labels loss.**
`reset_occupancy` applies `inverse_sigmoid` to values in *occupancy* space (0.005–0.995).
`unflatten_voronoi_features` originally filled unused slots with `0.0`, and
`inverse_sigmoid(0.0) = -inf`; the EMA path then computes `-inf - (-inf) = NaN`. The
poison sat dormant in masked slots until a keep-mask refresh promoted one to an active
site — detonating `OccLabLoss` exactly at a reset boundary.
**Fix:** fill masked slots with the neutral occupancy **0.5** (`inverse_sigmoid(0.5) = 0`)
at the two occupancy-space reset call sites. Also force the occupancy-label refresh in
lockstep with every keep-mask refresh (two Gaussians can flip classification in opposite
directions, leaving the total count unchanged while reshuffling which label maps where —
a size check alone is insufficient).

## Verification

Regression tests (in the session scratchpad; re-runnable against the `milo` conda env):
- `test_adaptive_pivots.py` — classification codes; 9/5/3 counts; symmetry properties
  (linear pair antipodal; planar corners balanced across the minor axis); mask-aware
  flatten/unflatten round-trip; legacy byte-identical.
- `test_get_tetra_points.py` — real `GaussianModel._get_tetra_points`: correct counts,
  fewer points than legacy, volumetric set matches legacy exactly.
- `test_keep_mask_consistency.py` — `override_keep_mask` freezes topology under scale
  drift while positions still update (Bug 1).
- `test_occupancy_reset_nan.py` — reset → mask change activating a previously-masked slot
  → occupancy logits / BCE loss stay finite (Bug 3).

End-to-end: full 18k-iteration training on Truck completes with zero NaN for both
`default` (baseline) and `default_adaptive`.

## How to run

```bash
conda activate milo
cd milo
# Adaptive:
python train.py -s <DATASET> -m <OUT_adaptive> --imp_metric outdoor --rasterizer radegs \
  --mesh_config default_adaptive --wandb_project ets-3dgs --wandb_entity <ENTITY> --log_interval 200
# Baseline:
python train.py -s <DATASET> -m <OUT_baseline> --imp_metric outdoor --rasterizer radegs \
  --mesh_config default --wandb_project ets-3dgs --wandb_entity <ENTITY> --log_interval 200
```

## Enhancement: Frozen-Type Shape Regularization (`use_type_shape_regularization`)

Instead of reclassifying every Gaussian on every Delaunay reset, the successor **freezes** the
volumetric/planar/linear classification once, at the first mesh-in-the-loop iteration
(`start_iter=8001`), and then **regularizes each Gaussian's shape toward its frozen type**. This is
safe to freeze there because the Gaussian population is already fixed by that point (no
densification past `densify_until_iter`, last simplification at `simp_iteration2=8000`), so a
per-Gaussian `(N,)` type array stays valid for the whole remaining run.

**Shape-reg loss (log-gap hinge, `utils/general_utils.py::compute_type_shape_regularization`).**
Using the same sorted log-scale gaps as the classifier (`gap_major=l0−l1`, `gap_minor=l1−l2`):
- **Linear:** `relu(log(linear_target_ratio) − gap_major)` — gradient grows the dominant axis,
  shrinks the second → more needle-like ("one eigenvalue bigger").
- **Planar:** `relu(log(planar_target_ratio) − gap_minor)` — gradient shrinks the smallest axis
  → flatter ("min eigenvalue smaller").
- **Volumetric:** no term. The hinge is **zero once a Gaussian is anisotropic enough**, so it only
  sharpens under-shaped Gaussians up to the target ratio (default 5.0, beyond the 3.0 classification
  threshold), bounding scale drift. Folded into `total_mesh_loss` with weight `shape_reg_weight`
  (default 0.01) and logged as `ShapeRegLoss`.

**Freezing lets the earlier caching fixes be reverted.** With frozen types the per-Gaussian pivot
*count* depends only on the type (8/4/2 corners + center) — invariant under scale drift and axis
rank flips. So:
- **Bug 1 reverted:** removed `override_keep_mask` / `need_fresh_keep_mask` / the cached
  `voronoi_keep_mask` replay. The keep-mask is recomputed fresh each iteration from the frozen
  types (`compute_pivot_keep_mask(..., gaussian_types=...)`, and
  `_get_tetra_points(..., override_gaussian_types=...)`); its total point count stays consistent
  with the cached `delaunay_tets` (which still refresh on Delaunay resample).
- **Bug 3 coupling reverted:** removed the `need_fresh_keep_mask` term from the occupancy-label
  refresh (no reclassification → no slot reshuffle). The `fill_value=0.5` fills are kept as cheap,
  now non-load-bearing, insurance.
- **Bug 2 kept:** the symmetric antipodal/tetrahedral corner selection is a geometric anti-sliver
  fix, independent of freezing.

**Config:** `use_type_shape_regularization` (+ `shape_reg_weight`,
`shape_reg_linear_target_ratio`, `shape_reg_planar_target_ratio`) added to all
`configs/mesh/*.yaml` (off). New `configs/mesh/default_adaptive_shapereg.yaml` turns it on (it also
requires `use_adaptive_pivot_sampling: true`, else it warns and no-ops). Files touched:
`utils/general_utils.py`, `scene/gaussian_model.py`, `regularization/regularizer/mesh.py`,
`train.py` / `train_regular_densification.py`, `utils/log_utils.py`.

**Bring-up verification (Truck, `default_adaptive_shapereg`):** types froze at 8001
(115,751 volumetric / 136,803 planar / 74,142 linear of 326,696) → 1,948,200 Delaunay points
(−34% vs fixed-9's 2,940,264). `ShapeRegLoss` logged and trending down (0.00354 → 0.00066 over
8010→8400). The post-freeze Delaunay reset at 8500 recomputed the **identical** 1,948,200-point
triangulation (confirming the count-stability invariant), and the run crossed the freeze boundary
and reset with **zero NaN/errors**. Unit tests: `scratchpad/test_shape_reg.py` (loss directions +
zero-past-target, override==live keep-mask, count stability under drift).

## Next steps

1. **Compute the paper's quality metrics** (not just training loss) on both checkpoints:
   mesh extraction (`mesh_extract_sdf.py`) → F1 vs GT, mesh vertex/triangle counts,
   Mesh-Based NVS. Expectation to validate: equal-or-better F1 at lower mesh complexity.
2. **Sweep `linear_ratio` / `planar_ratio`** (currently 3.0/3.0). More conservative
   thresholds reduce the anisotropic population; more aggressive ones cut more points.
3. **Optionally tighten the Delaunay budget:** `n_max_gaussians_for_delaunay =
   n_max_points_in_delaunay / 9` in `mesh.py` is now conservative (actual ≈ 5.36/Gaussian),
   so the effective site budget is under-used — left as-is for safety.
4. **Run more T&T / DTU scenes** to confirm the compute/quality trade-off generalizes.
