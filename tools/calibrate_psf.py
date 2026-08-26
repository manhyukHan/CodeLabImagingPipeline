"""
Calibrate this experiment's PSF from its own reference-hybe spots.

There is no bead stack for this scheme, so the PSF is recovered from the
data. The reference hybe is the right source for two reasons: it images
many independent point-like emitters in ONE optical configuration, and
being the reference it needs no alignment first, so nothing about the
calibration depends on the alignment being right.

Candidate shapes (psf.py: gaussian / moffat / lorentzian) are fitted
JOINTLY across many reference crops -- one shared shape, with amplitude,
centre and a linear background profiled out per crop -- and scored by
residual per voxel so families stay comparable. The winner is a property
of the experiment and is reused for every hybe, channel and allele.

Renders the comparison as well as printing it: a residual number says
which candidate won, a picture says whether the winner actually looks
like the data, which is the part worth checking by eye.

Usage:
    python tools/calibrate_psf.py bench.npz [--hybe Hyb_016] [--crops 40]
    python tools/calibrate_psf.py bench.npz --save G:/.../MP58/DNA
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline.localization import psf as P            # noqa: E402

DEFAULT_OUT = os.path.join('notes', 'chromatin_tracing_optimization')


def gather(bench, hybe, kind, limit):
    """Reference-hybe crops, brightest first -- calibration wants the
    cleanest emitters available, not a random sample."""
    keys = [k for k in bench.files
            if k.endswith(f'|{hybe}|{kind}')]
    scored = []
    for k in keys:
        c = bench[k].astype(float)
        if not np.isfinite(c).any():
            continue
        contrast = float(np.nanmax(c) - np.nanmedian(c))
        scored.append((contrast, k))
    scored.sort(reverse=True)
    chosen = [k for _s, k in scored[:limit]]
    return [bench[k].astype(float) for k in chosen], chosen


def _local_background(cube, peak_idx, voxel_um, radius_um=(1.0, 1.0, 3.0)):
    """
    A plane b0 + by*y + bx*x + bz*z fitted to the crop's OUTER shell --
    the voxels inside the fit radius but outside a core region around the
    peak, so the emitter itself does not pull the background up.

    Returns the plane evaluated over the whole crop. Falls back to the
    global median if the shell is too small to constrain a plane.
    """
    dy, dx, dz = voxel_um
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
        coef, *_ = np.linalg.lstsq(A, cube[shell], rcond=None)
    except Exception:
        return np.full(cube.shape, float(np.nanmedian(cube)))
    return coef[0] + coef[1] * Y + coef[2] * X + coef[3] * Z


def render(crops, results, voxel_um, kind, hybe, out_dir):
    """Radial and axial profiles: data vs every candidate."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    dy, dx, dz = voxel_um
    # Empirical profiles, each crop normalised and centred on its own
    # peak, with the LOCAL LINEAR BACKGROUND subtracted -- the same
    # background the calibration itself fits.
    #
    # Subtracting a global median instead (the obvious thing, and what
    # this did first) leaves a pedestal that reads as a heavy PSF tail:
    # the axial profile appeared to flatten at ~0.09 of peak all the way
    # out to 3 um, which made a Lorentzian look like the right shape.
    # With the fitted local background removed the same profile decays to
    # -0.003 by 2.8 um. The plateau was background, not signal.
    lat_r, lat_v, ax_r, ax_v = [], [], [], []
    for c in crops:
        iy, ix, iz = np.unravel_index(int(np.nanargmax(c)), c.shape)
        base = _local_background(c, (iy, ix, iz), voxel_um)
        peak = float(c[iy, ix, iz]) - float(base[iy, ix, iz])
        if peak <= 0:
            continue
        yy = (np.arange(c.shape[0]) - iy) * dy
        xx = (np.arange(c.shape[1]) - ix) * dx
        zz = (np.arange(c.shape[2]) - iz) * dz
        lat_r.extend(yy)
        lat_v.extend((c[:, ix, iz] - base[:, ix, iz]) / peak)
        lat_r.extend(xx)
        lat_v.extend((c[iy, :, iz] - base[iy, :, iz]) / peak)
        ax_r.extend(zz)
        ax_v.extend((c[iy, ix, :] - base[iy, ix, :]) / peak)
    lat_r, lat_v = np.array(lat_r), np.array(lat_v)
    ax_r, ax_v = np.array(ax_r), np.array(ax_v)

    fig, axes2 = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    best = results['best']
    fig.suptitle(f'PSF calibrated from {len(crops)} {hybe} {kind} crops '
                 f'(voxel {dy}x{dx}x{dz} um)   --   best by rss: {best}\n'
                 f'top: linear (the core, which dominates the residual)   '
                 f'bottom: log (the tails, which the residual barely sees)',
                 fontsize=12)

    panels = [(axes2[0][0], lat_r, lat_v, 'lateral offset (um)', 1.2, False),
              (axes2[0][1], ax_r, ax_v, 'axial offset (um)', 3.0, False),
              (axes2[1][0], lat_r, lat_v, 'lateral offset (um)', 1.2, True),
              (axes2[1][1], ax_r, ax_v, 'axial offset (um)', 3.0, True)]
    for ax, r, v, label, lim, logy in panels:
        m = np.abs(r) <= lim
        # binned median of the real data, so the cloud is readable
        bins = np.linspace(-lim, lim, 41)
        idx = np.digitize(r[m], bins)
        bx, by = [], []
        for i in range(1, len(bins)):
            sel = v[m][idx == i]
            if sel.size:
                bx.append((bins[i - 1] + bins[i]) / 2)
                by.append(np.median(sel))
        ax.plot(r[m], v[m], '.', color='#cccccc', ms=1.5, alpha=.5, zorder=1)
        ax.plot(bx, by, 'o-', color='k', ms=4, lw=1.6, label='data (binned median)', zorder=3)
        grid = np.linspace(-lim, lim, 300)
        zero = np.zeros_like(grid)
        for family, colour in (('gaussian', '#1f77b4'), ('moffat', '#2ca02c'),
                               ('lorentzian', '#d62728'),
                               ('gaussian_halo', '#9467bd')):
            if family not in results:
                continue
            pr = results[family]['params']
            shape = tuple(pr[k] for k in P.FAMILIES[family][1])
            if 'lateral' in label:
                curve = P.evaluate(family, shape, grid, zero, zero)
            else:
                curve = P.evaluate(family, shape, zero, zero, grid)
            style = '-' if family == best else '--'
            ax.plot(grid, curve, style, color=colour, lw=2.2 if family == best else 1.4,
                    label=f'{family} (rss/vox {results[family]["score"]:.0f})')
        ax.set_xlabel(label)
        ax.set_ylabel('normalised intensity')
        if logy:
            ax.set_yscale('log')
            ax.set_ylim(2e-3, 1.6)
        else:
            ax.set_ylim(-0.1, 1.15)
        ax.grid(alpha=.3, which='both')
        ax.legend(fontsize=8)
    path = os.path.join(out_dir, f'psf_calibration_{hybe}_{kind}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bench')
    ap.add_argument('--hybe', default='Hyb_016')
    ap.add_argument('--kind', action='append', default=None,
                    choices=['fiducial', 'readout'])
    ap.add_argument('--crops', type=int, default=40)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--voxel', default='0.208,0.208,0.2')
    ap.add_argument('--save', default=None,
                    help='storage_path to write analysis/psf.json into')
    a = ap.parse_args()

    voxel = tuple(float(v) for v in a.voxel.split(','))
    bench = np.load(a.bench, allow_pickle=False)
    for kind in (a.kind or ['fiducial', 'readout']):
        crops, keys = gather(bench, a.hybe, kind, a.crops)
        if not crops:
            print(f'{kind}: no {a.hybe} crops in this bench')
            continue
        print(f'--- {kind}: calibrating on {len(crops)} {a.hybe} crops '
              f'(voxel {voxel}) ---', flush=True)
        results = P.calibrate(crops, voxel_um=voxel)
        best = results['best']
        print(f'   best: {best}  {results[best]["params"]}')
        fig = render(crops, results, voxel, kind, a.hybe, a.out)
        print(f'   wrote {fig}')
        if a.save and kind == 'fiducial':
            scores = {f: results[f] for f in results if f != 'best'}
            path = P.save(a.save, best, results[best]['params'], voxel,
                          source={'hybe': a.hybe, 'channel_kind': kind,
                                  'n_crops': len(crops), 'bench': os.path.basename(a.bench)},
                          scores=scores)
            print(f'   saved {path}')
        print()


if __name__ == '__main__':
    main()
