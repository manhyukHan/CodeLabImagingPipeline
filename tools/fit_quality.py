"""
Is the FIT good, separately from whether the GATE is good?

The gate question -- which acceptance thresholds to use -- is answered by
tools/fit_gates.py against the replicate pairs. This asks the prior
question: when the fit returns a centroid, is that centroid ON the
emitter at all?

The complaint this exists to test is concrete: the fit often reports a
centroid in a BLANK region rather than near the argmax. That is checkable
without any ground truth, from two facts about the experiment:

  1. Every hybe's FIDUCIAL images the same physical markers, and every
     fiducial crop is cut at the same shared-frame point. So across the
     hybes of one allele the fiducial should sit at nearly the SAME
     crop-local position. Spread across hybes is therefore error, and can
     be compared directly against the spread of the raw argmax.

  2. A Hyb/Rep pair images the same locus, so their READOUT crops should
     be similar and their fitted positions close -- in 3D, not just
     laterally.

Three measurements, all in micrometres and all including Z:

  occupancy   intensity at the fitted centroid, as a fraction of the
              intensity at the argmax, both above the local background.
              ~1 means the fit landed on the emitter. Near 0 means it
              landed in background -- the blank-region failure, measured
              rather than eyeballed.
  drift       distance from the fitted centroid to the argmax voxel.
  spread      per allele, the scatter of fitted positions across hybes
              (fiducial) or within a replicate pair (readout).

Usage:
    python tools/fit_quality.py bench.npz [--out DIR]
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
FIT_RADIUS = (1.0, 1.0, 3.0)

REPLICATES = [('Hyb_023', 'Rep_023'), ('Hyb_032', 'Rep_032'),
              ('Hyb_037', 'Rep_037'), ('Hyb_040', 'Rep_040'),
              ('Hyb_081', 'Rep_081'), ('Hyb_042', 'Rep_042'),
              ('Hyb_070', 'Rep_070')]


def local_background(cube, peak_idx, radius_um=(1.0, 1.0, 3.0)):
    """Plane fitted to the shell inside the fit radius but outside the
    core, so the emitter cannot lift its own background."""
    dy, dx, dz = VOXEL
    iy, ix, iz = peak_idx
    Y, X, Z = np.indices(cube.shape)
    Y, X, Z = Y * dy, X * dx, Z * dz
    cy, cx, cz = iy * dy, ix * dx, iz * dz
    ry, rx, rz = radius_um
    inside = (np.abs(Y - cy) <= ry) & (np.abs(X - cx) <= rx) & (np.abs(Z - cz) <= rz)
    core = (np.abs(Y - cy) <= 0.45) & (np.abs(X - cx) <= 0.45) & (np.abs(Z - cz) <= 1.2)
    shell = inside & ~core & np.isfinite(cube)
    if shell.sum() < 50:
        return np.full(cube.shape, float(np.nanmedian(cube)))
    A = np.column_stack([np.ones(shell.sum()), Y[shell], X[shell], Z[shell]])
    try:
        c, *_ = np.linalg.lstsq(A, cube[shell], rcond=None)
    except Exception:
        return np.full(cube.shape, float(np.nanmedian(cube)))
    return c[0] + c[1] * Y + c[2] * X + c[3] * Z


def sample(cube, yxz_voxel):
    """Trilinear-ish sample: nearest voxel, clamped. Good enough to ask
    'is there signal here', which is all occupancy needs."""
    y, x, z = yxz_voxel
    iy = int(np.clip(round(y), 0, cube.shape[0] - 1))
    ix = int(np.clip(round(x), 0, cube.shape[1] - 1))
    iz = int(np.clip(round(z), 0, cube.shape[2] - 1))
    return float(cube[iy, ix, iz]), (iy, ix, iz)


def measure(cube, engine):
    """(occupancy, drift_um, fitted_yxz_um, argmax_yxz_um) or None."""
    dy, dx, dz = VOXEL
    iy, ix, iz = np.unravel_index(int(np.nanargmax(cube)), cube.shape)
    cy, cx = (cube.shape[0] - 1) / 2.0, (cube.shape[1] - 1) / 2.0
    if engine == 'v1':
        r = L.fit_gaussian_3d(cube, float(cx), float(cy), float(iz))
        if r is None:
            return None
        _a, fx, fy, fz = r[:4]
        fit_vox = (fy, fx, fz)
    else:
        f = U.fit_gaussian_3d_um(cube, cy, cx, float(iz), voxel_um=VOXEL,
                                 max_sigma_xy_um=3.0, max_sigma_z_um=6.0,
                                 fit_radius_um=FIT_RADIUS, background='linear',
                                 apply_gates=False)
        if f is None:
            return None
        fit_vox = (f.y, f.x, f.z)

    bg = local_background(cube, (iy, ix, iz))
    i_peak, _ = sample(cube, (iy, ix, iz))
    b_peak, _ = sample(bg, (iy, ix, iz))
    i_fit, fit_idx = sample(cube, fit_vox)
    b_fit, _ = sample(bg, fit_vox)
    denom = i_peak - b_peak
    occupancy = ((i_fit - b_fit) / denom) if denom > 0 else np.nan

    fit_um = np.array([fit_vox[0] * dy, fit_vox[1] * dx, fit_vox[2] * dz])
    arg_um = np.array([iy * dy, ix * dx, iz * dz])
    return occupancy, float(np.linalg.norm(fit_um - arg_um)), fit_um, arg_um


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bench')
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--limit', type=int, default=400)
    a = ap.parse_args()

    b = np.load(a.bench, allow_pickle=False)
    meta = b['__meta__']
    res = {}
    for kind in ('fiducial', 'readout'):
        keys = [k for k in b.files if k.endswith('|' + kind)][:a.limit]
        for engine in ('v1', 'v2'):
            occ, drift, dz_err = [], [], []
            for k in keys:
                out = measure(b[k].astype(float), engine)
                if out is None:
                    continue
                o, d, fit_um, arg_um = out
                occ.append(o)
                drift.append(d)
                dz_err.append(abs(fit_um[2] - arg_um[2]))
            res[(kind, engine)] = (np.array(occ), np.array(drift), np.array(dz_err))

    print('OCCUPANCY -- intensity at the fitted centroid, as a fraction of the')
    print('intensity at the argmax, both above local background.')
    print('1.0 = the fit landed on the emitter.  <=0 = it landed in background.\n')
    hdr = (f'{"":<10}{"engine":<7}{"n":>5}{"median":>9}{"frac <0.2":>11}'
           f'{"frac <=0":>10}{"drift um":>10}{"|dz| um":>9}')
    print(hdr)
    print('-' * len(hdr))
    for kind in ('fiducial', 'readout'):
        for engine in ('v1', 'v2'):
            o, d, z = res[(kind, engine)]
            o = o[np.isfinite(o)]
            print(f'{kind:<10}{engine:<7}{len(o):>5}{np.median(o):>9.3f}'
                  f'{100 * np.mean(o < 0.2):>10.0f}%{100 * np.mean(o <= 0):>9.0f}%'
                  f'{np.median(d):>10.3f}{np.median(z):>9.3f}')

    # -- fact 1: fiducials of one allele must agree across hybes --
    print('\nFIDUCIAL CONSISTENCY -- all hybes image the same markers, and every')
    print('crop is cut at the same shared-frame point, so crop-local positions')
    print('should agree across hybes. Spread is error. (3D, um)\n')
    print(f'{"":<10}{"argmax":>10}{"v1 fit":>10}{"v2 fit":>10}')
    print('-' * 40)
    spreads = {'argmax': [], 'v1': [], 'v2': []}
    for aid, *_r in meta[:8]:
        pos = {'argmax': [], 'v1': [], 'v2': []}
        for k in [k for k in b.files if k.startswith(f'a{aid}|') and k.endswith('|fiducial')]:
            cube = b[k].astype(float)
            iy, ix, iz = np.unravel_index(int(np.nanargmax(cube)), cube.shape)
            pos['argmax'].append([iy * VOXEL[0], ix * VOXEL[1], iz * VOXEL[2]])
            for engine in ('v1', 'v2'):
                out = measure(cube, engine)
                if out is not None:
                    pos[engine].append(out[2])
        for name in pos:
            p = np.array(pos[name])
            if len(p) > 3:
                spreads[name].append(float(np.median(np.linalg.norm(
                    p - np.median(p, axis=0), axis=1))))
    print(f'{"median":<10}{np.median(spreads["argmax"]):>10.3f}'
          f'{np.median(spreads["v1"]):>10.3f}{np.median(spreads["v2"]):>10.3f}')

    render(res, spreads, a.out)


def render(res, spreads, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    fig.suptitle('Is the fitted centroid ON the emitter?  '
                 '(occupancy 1.0 = yes, <=0 = it is in background)', fontsize=13)
    for ax, kind in zip(axes[:2], ('fiducial', 'readout')):
        for engine, colour in (('v1', '#d62728'), ('v2', '#2ca02c')):
            o = res[(kind, engine)][0]
            o = o[np.isfinite(o)]
            ax.hist(np.clip(o, -0.5, 1.5), bins=60, range=(-0.5, 1.5), alpha=.6,
                    color=colour, label=f'{engine} (median {np.median(o):.2f})')
        ax.axvline(0, color='k', lw=1)
        ax.axvline(1, color='k', ls=':', lw=1)
        ax.set_title(f'{kind} channel')
        ax.set_xlabel('occupancy: signal at fit / signal at argmax')
        ax.legend(fontsize=9)
        ax.grid(alpha=.3)
    ax = axes[2]
    names = ['argmax', 'v1', 'v2']
    vals = [np.median(spreads[n]) if spreads[n] else np.nan for n in names]
    ax.bar(names, vals, color=['#999999', '#d62728', '#2ca02c'])
    ax.set_ylabel('median 3D spread across hybes (um)')
    ax.set_title('fiducial consistency\n(same markers -> spread is error)')
    ax.grid(alpha=.3, axis='y')
    path = os.path.join(out_dir, 'fit_quality.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
