"""
Which gate actually shows a trade-off, now that the FIT is fixed?

The earlier sweep (tools/fit_gates.py) ran on fits that were mostly
stopping on constraints, so no threshold from it was trustworthy. With
v2 (consensus box, centroid seed, linear background, calibrated PSF) the
fit lands on the emitter in ~98% of fiducial crops, so a gate is finally
being asked to do only its own job: reject crops with nothing in them.

CANDIDATES, and why each might be the one
-----------------------------------------
occupancy         signal at the fitted centroid over signal at the
                  argmax, both above local background. Measures directly
                  what the inherited heuristics only approximate: did the
                  fit land on the emitter.
max_uncert_xy_nm  95% CI half-width on lateral position. A statement
                  about the ESTIMATOR, in a physical length. Swept
                  independently of z, since a plane and a pixel are
                  different distances.
max_uncert_z_nm   the same, axially.
min_hb_ratio      v1 heritage: raw seed voxel over background.
min_ah_ratio      v1 heritage: fitted amplitude over raw seed voxel.
at_bound          not a threshold but a filter: a fit that stopped on a
                  constraint reports the constraint, not a measurement,
                  and its CI does not describe it either.

SCORED, never gated on: the replicate distance -- 3D, fiducial-corrected,
Hyb-vs-Rep only (toehold rounds are displacement controls). Using it as a
gate input would be circular.

A gate is only interesting if it trades: tightening should buy accuracy
and cost coverage. One that loses both is simply wrong, and one that
changes neither is inert.

EQUAL-COVERAGE COMPARISON
-------------------------
v1 scores 26 of 336 possible pairs -- it rejects 92% of readout crops and
keeps the easiest. Comparing its median against v2's at full coverage is
a selection effect, not a result. So each v2 gate is also reported at the
threshold that leaves it 26 pairs, which is the only apples-to-apples
number.

Usage:
    python tools/gate_sweep_v2.py bench.npz [--out DIR]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline.localization import fit3d_mle as M       # noqa: E402
from codelab_pipeline.localization import fit3d_um as U        # noqa: E402
from codelab_pipeline.localization import localization as L    # noqa: E402
from codelab_pipeline.localization import psf as P             # noqa: E402

DEFAULT_OUT = os.path.join('notes', 'chromatin_tracing_optimization')
VOXEL = (0.208, 0.208, 0.2)
BOX_HALF = (5, 5, 15)
PB_XY, PB_Z = 5 * VOXEL[0], 10 * VOXEL[2]
FIT_RADIUS = (1.0, 1.0, 3.0)
REPLICATES = [('Hyb_023', 'Rep_023'), ('Hyb_032', 'Rep_032'),
              ('Hyb_037', 'Rep_037'), ('Hyb_040', 'Rep_040'),
              ('Hyb_081', 'Rep_081'), ('Hyb_042', 'Rep_042'),
              ('Hyb_070', 'Rep_070')]


def offsets_of(bench):
    out = {}
    for s in bench['__zpred__']:
        k, _z, o = s.rsplit('|', 2)
        out[k] = float(o)
    return out


def baseline(bench, offsets, aid):
    vals = []
    for k in bench.files:
        if k.startswith(f'a{aid}|') and k.endswith('|fiducial'):
            c = bench[k]
            if np.isfinite(c).any():
                iz = int(np.unravel_index(int(np.nanargmax(c)), c.shape)[2])
                vals.append(iz + offsets.get(f'a{aid}|' + k.split('|')[1], 0.0))
    return float(np.median(vals)) if vals else None


def _bg(cube, centre):
    dy, dx, dz = VOXEL
    Y, X, Z = np.indices(cube.shape)
    Y, X, Z = Y * dy, X * dx, Z * dz
    cy, cx, cz = centre[0] * dy, centre[1] * dx, centre[2] * dz
    ins = (np.abs(Y - cy) <= 1.) & (np.abs(X - cx) <= 1.) & (np.abs(Z - cz) <= 3.)
    core = (np.abs(Y - cy) <= .45) & (np.abs(X - cx) <= .45) & (np.abs(Z - cz) <= 1.2)
    sh = ins & ~core & np.isfinite(cube)
    if sh.sum() < 50:
        f = cube[np.isfinite(cube)]
        return np.full(cube.shape, float(np.median(f)) if f.size else 0.0)
    A = np.column_stack([np.ones(sh.sum()), Y[sh], X[sh], Z[sh]])
    q, *_ = np.linalg.lstsq(A, cube[sh], rcond=None)
    return q[0] + q[1] * Y + q[2] * X + q[3] * Z


def _at(a, i):
    return float(a[int(np.clip(round(i[0]), 0, a.shape[0] - 1)),
                   int(np.clip(round(i[1]), 0, a.shape[1] - 1)),
                   int(np.clip(round(i[2]), 0, a.shape[2] - 1))])


def fit_all(bench, kind, offsets, shape, engine='v2'):
    """Every crop of one kind, fitted ONCE, ungated, with its gate
    quantities attached."""
    out, t0 = {}, time.perf_counter()
    aids = sorted({int(k.split('|')[0][1:]) for k in bench.files
                   if not k.startswith('__')})
    for aid in aids:
        base = baseline(bench, offsets, aid)
        if base is None:
            continue
        for k in [k for k in bench.files
                  if k.startswith(f'a{aid}|') and k.endswith(f'|{kind}')]:
            hybe = k.split('|')[1]
            cube = bench[k].astype(float)
            off = offsets.get(f'a{aid}|{hybe}', 0.0)
            cy, cx = (cube.shape[0] - 1) / 2., (cube.shape[1] - 1) / 2.
            arg = np.unravel_index(int(np.nanargmax(cube)), cube.shape)
            if engine == 'v1':
                iz = int(arg[2])
                r = L.fit_gaussian_3d(cube, float(cx), float(cy), float(iz))
                if r is None:
                    continue
                pos = (r[2], r[1], r[3])
                rec = {'pos': np.array(pos) * np.array(VOXEL),
                       'occ': np.nan, 'ci_xy': np.nan, 'ci_z': np.nan,
                       'hb': np.nan, 'ah': np.nan, 'bound': False}
            else:
                box, (oy, ox, oz) = U.extract_box(cube, (cy, cx, base - off), BOX_HALF)
                bc = tuple((s - 1) / 2. for s in box.shape)
                seed = U.intensity_centroid(box, bc, BOX_HALF, voxel_um=VOXEL)
                if seed is None:
                    continue
                f = M.fit_gaussian_3d_mle(box, seed[0], seed[1], seed[2],
                                          voxel_um=VOXEL, family=shape['family'],
                                          shape_params=shape['params'],
                                          noise='gaussian', peak_bound_um=PB_XY,
                                          peak_bound_z_um=PB_Z,
                                          fit_radius_um=FIT_RADIUS,
                                          background='linear')
                if f is None:
                    continue
                pos = (f.y + oy, f.x + ox, f.z + oz)
                rec = {'pos': np.array(pos) * np.array(VOXEL),
                       'ci_xy': 2000 * max(f.ci_y_um, f.ci_x_um),
                       'ci_z': 2000 * f.ci_z_um,
                       'hb': f.peak_bg_ratio, 'ah': f.amp_h_ratio,
                       'bound': any(s in ('x', 'y', 'z') for s in f.at_bound)}
                bg = _bg(cube, arg)
                den = _at(cube, arg) - _at(bg, arg)
                rec['occ'] = ((_at(cube, pos) - _at(bg, pos)) / den) if den > 0 else np.nan
            out[k] = rec
    print(f'   {kind}/{engine}: {len(out)} fits in {time.perf_counter() - t0:.0f}s',
          flush=True)
    return out


def score(read, fid, keep):
    """(pairs, median_um, p90) over Hyb/Rep pairs surviving `keep`,
    3D and fiducial-corrected."""
    d = []
    aids = sorted({int(k.split('|')[0][1:]) for k in read})
    for aid in aids:
        for x, y in REPLICATES:
            ra, rb = read.get(f'a{aid}|{x}|readout'), read.get(f'a{aid}|{y}|readout')
            ga, gb = fid.get(f'a{aid}|{x}|fiducial'), fid.get(f'a{aid}|{y}|fiducial')
            if not all([ra, rb, ga, gb]):
                continue
            if not (keep(ra) and keep(rb)):
                continue
            d.append(float(np.linalg.norm((ra['pos'] - ga['pos'])
                                          - (rb['pos'] - gb['pos']))))
    if not d:
        return 0, float('nan'), float('nan')
    d = np.array(d)
    return len(d), float(np.median(d)), float(np.percentile(d, 90))


GATES = {
    'occupancy':        (np.round(np.arange(-0.2, 1.01, 0.05), 3),
                         lambda r, t: np.isfinite(r['occ']) and r['occ'] >= t),
    'max_uncert_xy_nm': (np.arange(20, 820, 20),
                         lambda r, t: np.isfinite(r['ci_xy']) and r['ci_xy'] < t),
    'max_uncert_z_nm':  (np.arange(20, 1620, 40),
                         lambda r, t: np.isfinite(r['ci_z']) and r['ci_z'] < t),
    'min_hb_ratio':     (np.round(np.arange(1.0, 3.01, 0.1), 2),
                         lambda r, t: np.isfinite(r['hb']) and r['hb'] >= t),
    'min_ah_ratio':     (np.round(np.arange(0.0, 1.01, 0.05), 2),
                         lambda r, t: np.isfinite(r['ah']) and r['ah'] >= t),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bench')
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--calib-crops', type=int, default=40)
    a = ap.parse_args()

    b = np.load(a.bench, allow_pickle=False)
    offsets = offsets_of(b)
    shapes = {}
    for kind in ('fiducial', 'readout'):
        keys = [k for k in b.files if k.endswith(f'|Hyb_016|{kind}')][:a.calib_crops]
        zc = []
        for k in keys:
            aid = int(k.split('|')[0][1:])
            base = baseline(b, offsets, aid)
            zc.append(None if base is None
                      else base - offsets.get(f'a{aid}|Hyb_016', 0.0))
        res = P.calibrate([b[k].astype(float) for k in keys], voxel_um=VOXEL,
                          verbose=False, z_centres=zc)
        best = res['best']
        shapes[kind] = {'family': best,
                        'params': tuple(res[best]['params'][n]
                                        for n in P.FAMILIES[best][1])}
        print(f'PSF [{kind}]: {best} '
              + '  '.join(f'{n}={res[best]["params"][n]:.4f}'
                          for n in P.FAMILIES[best][1]))

    read = fit_all(b, 'readout', offsets, shapes['readout'])
    fid = fit_all(b, 'fiducial', offsets, shapes['fiducial'])
    v1r = fit_all(b, 'readout', offsets, shapes['readout'], engine='v1')
    v1f = fit_all(b, 'fiducial', offsets, shapes['fiducial'], engine='v1')

    n_all, med_all, p90_all = score(read, fid, lambda r: True)
    n_v1, med_v1, p90_v1 = score(v1r, v1f, lambda r: True)
    print(f'\nungated: v2 {n_all} pairs @ {med_all:.4f} um   |   '
          f'v1 {n_v1} pairs @ {med_v1:.4f} um')
    n_bound, med_bound, _ = score(read, fid, lambda r: not r['bound'])
    print(f'at_bound filter alone: {n_bound} pairs @ {med_bound:.4f} um\n')

    curves = {}
    print(f'{"gate":<18}{"thr":>8}{"pairs":>8}{"median um":>12}{"p90":>9}   note')
    print('-' * 74)
    for name, (grid, keep) in GATES.items():
        rows = []
        for t in grid:
            n, med, p90 = score(read, fid, lambda r, t=t: keep(r, t))
            rows.append((float(t), n, med, p90))
        curves[name] = rows
        # the threshold that matches v1's coverage, for a fair head-to-head
        eq = min((r for r in rows if r[1] > 0),
                 key=lambda r: abs(r[1] - n_v1), default=None)
        best_med = min((r for r in rows if r[1] >= 30), key=lambda r: r[2],
                       default=None)
        if eq:
            print(f'{name:<18}{eq[0]:>8.3g}{eq[1]:>8}{eq[2]:>12.4f}{eq[3]:>9.4f}'
                  f'   <- v1-equal coverage ({n_v1})')
        if best_med:
            print(f'{name:<18}{best_med[0]:>8.3g}{best_med[1]:>8}{best_med[2]:>12.4f}'
                  f'{best_med[3]:>9.4f}   <- best median at >=30 pairs')
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'gate_sweep_v2.json'), 'w') as f:
        json.dump({'curves': curves, 'v1': [n_v1, med_v1, p90_v1],
                   'v2_ungated': [n_all, med_all, p90_all]}, f, indent=1,
                  default=float)
    render(curves, n_v1, med_v1, n_all, med_all, a.out)


def render(curves, n_v1, med_v1, n_all, med_all, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(curves)
    fig, axes = plt.subplots(2, len(names), figsize=(4.3 * len(names), 7.6),
                             constrained_layout=True)
    fig.suptitle('Gates on the FIXED v2 fit -- which one actually trades?\n'
                 f'ungated v2: {n_all} pairs @ {med_all:.3f} um    |    '
                 f'v1: {n_v1} pairs @ {med_v1:.3f} um (8% coverage)',
                 fontsize=12)
    for j, name in enumerate(names):
        r = np.array(curves[name], dtype=float)
        t, n, med, p90 = r.T
        ax = axes[0][j]
        ax.plot(t, n, 'o-', color='#1f77b4', ms=3)
        ax.axhline(n_v1, color='crimson', ls=':', lw=1.5)
        ax.set_title(name, fontsize=10)
        ax.set_ylabel('pairs scored')
        ax.grid(alpha=.3)
        ax = axes[1][j]
        ok = n > 0
        ax.plot(t[ok], med[ok], 'o-', color='#d62728', ms=3, label='median')
        ax.plot(t[ok], p90[ok], '^--', color='#ff9896', ms=3, label='p90')
        ax.axhline(med_v1, color='crimson', ls=':', lw=1.5, label='v1 median')
        ax.set_xlabel(name)
        ax.set_ylabel('replicate distance (um)')
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    path = os.path.join(out_dir, 'gate_sweep_v2.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
