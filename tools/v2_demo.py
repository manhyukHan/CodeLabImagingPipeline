"""
v2 end to end on real crops, rendered so the fit can be judged by eye.

WHAT v2 IS, ASSEMBLED
---------------------
Every piece below was measured separately on the 16-allele bench first;
this puts them together and draws the result.

  1. BASELINE, fit-free. Each hybe's argmax z, mapped to the shared frame
     with its own cell_z_offset, then the MEDIAN across hybes. Measured:
     1.05 planes of box-placement error at 0.09 s, against 4.65 planes at
     106 s for a full pillar fit per hybe -- the expensive baseline is
     WORSE, because a pillar fit is the degenerate fit being fixed here,
     so building a baseline out of it propagates the damage.
  2. BOX, not pillar. Each hybe's box is centred at baseline minus that
     hybe's own offset, NaN-padded where it runs off the slab (0.1% of
     crops) rather than clipped, so every box is the same volume.
  3. CENTROID SEED. Intensity-weighted first moment inside the box,
     weighted above the box median.
  4. LINEAR BACKGROUND, so the gradient is not paid for out of the
     position estimate.
  5. LOOSE BOUNDS, 5 px lateral and 10 planes axial, separately.

Measured combined: occupancy 0.354 -> 0.818, blank-region fits 34% -> 0%.

WHAT THE PNG SHOWS
------------------
One row per hybe, v1 and v2 side by side on the SAME crop, YX above ZX,
with each engine's fitted centroid circled. A centroid sitting on empty
background is the failure this exists to show; a centroid on the emitter
is the fix. The occupancy number is printed per tile so the picture and
the metric can be checked against each other.

Usage:
    python tools/v2_demo.py bench.npz [--allele 5] [--hybes 10]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline.localization import fit3d_um as U        # noqa: E402
from codelab_pipeline.localization import localization as L    # noqa: E402

DEFAULT_OUT = os.path.join('notes', 'chromatin_tracing_optimization')
VOXEL = (0.208, 0.208, 0.2)
BOX_HALF = (5, 5, 15)            # voxels
PB_XY_UM = 5 * VOXEL[0]          # 5 px lateral
PB_Z_UM = 10 * VOXEL[2]          # 10 planes axial


def load_offsets(bench):
    off = {}
    for s in bench['__zpred__']:
        k, _zp, o = s.rsplit('|', 2)
        off[k] = float(o)
    return off


def baseline_shared_z(cubes, offsets, aid):
    """Fit-free consensus depth in the shared frame (see module docstring)."""
    vals = []
    for hybe, c in cubes.items():
        if not np.isfinite(c).any():
            continue
        iz = int(np.unravel_index(int(np.nanargmax(c)), c.shape)[2])
        vals.append(iz + offsets.get(f'a{aid}|{hybe}', 0.0))
    return float(np.median(vals)) if vals else None


def local_bg(cube, centre):
    """Plane fitted to the shell around the centre, emitter excluded."""
    dy, dx, dz = VOXEL
    Y, X, Z = np.indices(cube.shape)
    Y, X, Z = Y * dy, X * dx, Z * dz
    cy, cx, cz = centre[0] * dy, centre[1] * dx, centre[2] * dz
    ins = (np.abs(Y - cy) <= 1.0) & (np.abs(X - cx) <= 1.0) & (np.abs(Z - cz) <= 3.0)
    core = (np.abs(Y - cy) <= .45) & (np.abs(X - cx) <= .45) & (np.abs(Z - cz) <= 1.2)
    sh = ins & ~core & np.isfinite(cube)
    if sh.sum() < 50:
        return np.full(cube.shape, float(np.nanmedian(cube[np.isfinite(cube)])))
    A = np.column_stack([np.ones(sh.sum()), Y[sh], X[sh], Z[sh]])
    q, *_ = np.linalg.lstsq(A, cube[sh], rcond=None)
    return q[0] + q[1] * Y + q[2] * X + q[3] * Z


def _at(a, idx):
    return float(a[int(np.clip(round(idx[0]), 0, a.shape[0] - 1)),
                   int(np.clip(round(idx[1]), 0, a.shape[1] - 1)),
                   int(np.clip(round(idx[2]), 0, a.shape[2] - 1))])


def occupancy(cube, fit_yxz, arg_yxz):
    bg = local_bg(cube, arg_yxz)
    denom = _at(cube, arg_yxz) - _at(bg, arg_yxz)
    if denom <= 0:
        return np.nan
    return (_at(cube, fit_yxz) - _at(bg, fit_yxz)) / denom


def fit_v1(cube):
    """v1 exactly as it ships: XY seeded at the crop centre, Z at the
    argmax, on the full pillar."""
    cy, cx = (cube.shape[0] - 1) / 2.0, (cube.shape[1] - 1) / 2.0
    iz = int(np.unravel_index(int(np.nanargmax(cube)), cube.shape)[2])
    r = L.fit_gaussian_3d(cube, float(cx), float(cy), float(iz))
    if r is None:
        return None
    return (r[2], r[1], r[3])


def fit_v2(cube, z_centre):
    """v2: box, centroid seed, linear background, loose separate bounds.
    Returns the centroid in FULL-CROP voxel indices."""
    cy, cx = (cube.shape[0] - 1) / 2.0, (cube.shape[1] - 1) / 2.0
    box, (oy, ox, oz) = U.extract_box(cube, (cy, cx, z_centre), BOX_HALF)
    bc = tuple((s - 1) / 2.0 for s in box.shape)
    seed = U.intensity_centroid(box, bc, BOX_HALF, voxel_um=VOXEL)
    if seed is None:
        return None
    f = U.fit_gaussian_3d_um(box, seed[0], seed[1], seed[2], voxel_um=VOXEL,
                             peak_bound_um=PB_XY_UM, peak_bound_z_um=PB_Z_UM,
                             max_sigma_xy_um=3.0, max_sigma_z_um=6.0,
                             fit_radius_um=(1.0, 1.0, 3.0), background='linear',
                             apply_gates=False)
    if f is None:
        return None
    return (f.y + oy, f.x + ox, f.z + oz)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bench')
    ap.add_argument('--allele', type=int, default=None)
    ap.add_argument('--hybes', type=int, default=10)
    ap.add_argument('--kind', default='readout', choices=['fiducial', 'readout'])
    ap.add_argument('--out', default=DEFAULT_OUT)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from canvas import spot_fit_status

    b = np.load(a.bench, allow_pickle=False)
    offsets = load_offsets(b)
    labels = dict(s.split('|', 1) for s in b['__labels__'])
    meta = b['__meta__']
    aid = a.allele if a.allele is not None else int(meta[0][0])

    fid_cubes = {k.split('|')[1]: b[k].astype(float)
                 for k in b.files if k.startswith(f'a{aid}|') and k.endswith('|fiducial')}
    base = baseline_shared_z(fid_cubes, offsets, aid)
    if base is None:
        raise SystemExit(f'allele a{aid}: no fiducial crops')
    print(f'allele a{aid}  {labels.get(f"a{aid}", "?")}')
    print(f'fit-free consensus depth (shared frame): {base:.1f} planes '
          f'from {len(fid_cubes)} hybes\n')

    keys = sorted(k for k in b.files
                  if k.startswith(f'a{aid}|') and k.endswith(f'|{a.kind}'))
    keys = keys[:a.hybes]
    n = len(keys)
    fig, axes = plt.subplots(4, n, figsize=(2.05 * n, 8.6), squeeze=False)
    occ1, occ2 = [], []
    for j, k in enumerate(keys):
        hybe = k.split('|')[1]
        cube = b[k].astype(float)
        off = offsets.get(f'a{aid}|{hybe}', 0.0)
        arg = np.unravel_index(int(np.nanargmax(cube)), cube.shape)
        p1, p2 = fit_v1(cube), fit_v2(cube, base - off)
        o1 = occupancy(cube, p1, arg) if p1 else np.nan
        o2 = occupancy(cube, p2, arg) if p2 else np.nan
        occ1.append(o1)
        occ2.append(o2)
        # spot_fit_status wants centroid as (x, y, z)
        c1 = [(p1[1], p1[0], p1[2])] if p1 else None
        c2 = [(p2[1], p2[0], p2[2])] if p2 else None
        spot_fit_status.draw_spot_fit_status(
            axes[0][j], axes[1][j], cube, centroid=c1,
            title=f'{hybe}\nv1 pillar  occ={o1:.2f}' if p1 else f'{hybe}\nv1 NO FIT',
            title_fontsize=7)
        spot_fit_status.draw_spot_fit_status(
            axes[2][j], axes[3][j], cube, centroid=c2,
            title=f'v2 box  occ={o2:.2f}' if p2 else 'v2 NO FIT',
            title_fontsize=7)
    o1 = np.array([v for v in occ1 if np.isfinite(v)])
    o2 = np.array([v for v in occ2 if np.isfinite(v)])
    fig.suptitle(
        f'{labels.get(f"a{aid}", "")}  --  {a.kind} channel\n'
        f'rows 1-2: v1 on the full pillar (median occupancy {np.median(o1):.2f})   |   '
        f'rows 3-4: v2 box+centroid+linear bg (median occupancy {np.median(o2):.2f})\n'
        'yellow circle = fitted centroid. occ=1.0 means it sits on the emitter; '
        'occ near 0 means it sits in background.', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f'v2_vs_v1_a{aid}_{a.kind}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'v1 median occupancy {np.median(o1):.3f}   ({len(o1)}/{n} fitted)')
    print(f'v2 median occupancy {np.median(o2):.3f}   ({len(o2)}/{n} fitted)')
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
