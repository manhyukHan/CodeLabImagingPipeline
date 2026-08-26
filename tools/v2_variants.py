"""
What does each piece of v2 actually buy?

v2 is four independent changes stacked on v1: geometry (box instead of
pillar, centroid seed, loose separate bounds), a linear background, a
CALIBRATED PSF shape, and a POISSON noise model. Reporting them as one
number would make it impossible to tell which are earning their place --
and two of them (PSF, MLE) were built but never actually run in the
figure that showed the fit fixed, so their contribution is so far
entirely unmeasured on real data.

This runs the ladder, adding one thing at a time:

    v1                    as it ships: pillar, centre-seeded XY, LSQ
    v2-geom               box + centroid + linear bg, free Gaussian, LSQ
    v2-geom+psf           the same, with sigma FIXED to the calibration
    v2-geom+psf+mle       the same, with Poisson deviance residuals

Scored on three things that measure different failures:

    occupancy   did the centroid land on the emitter (fit quality)
    at_bound    did the fit stop on a constraint instead of an optimum
    replicate   3D distance between two rounds of one locus, with the
                pipeline's own fiducial correction applied -- the SCORE,
                never a gate input, since a gate judged by it would be
                circular

Cost is reported too, including what the PSF calibration itself costs,
since a calibration that has to be redone per experiment is only worth
having if it pays for itself.

Usage:
    python tools/v2_variants.py bench.npz [--limit 400]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline.localization import fit3d_mle as M       # noqa: E402
from codelab_pipeline.localization import fit3d_um as U        # noqa: E402
from codelab_pipeline.localization import localization as L    # noqa: E402
from codelab_pipeline.localization import psf as P             # noqa: E402

VOXEL = (0.208, 0.208, 0.2)
BOX_HALF = (5, 5, 15)
PB_XY = 5 * VOXEL[0]
PB_Z = 10 * VOXEL[2]
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


def baseline(cubes, offsets, aid):
    """Fit-free consensus depth: argmax per hybe -> shared frame -> median."""
    vals = []
    for hybe, c in cubes.items():
        if np.isfinite(c).any():
            iz = int(np.unravel_index(int(np.nanargmax(c)), c.shape)[2])
            vals.append(iz + offsets.get(f'a{aid}|{hybe}', 0.0))
    return float(np.median(vals)) if vals else None


def _bg_plane(cube, centre):
    dy, dx, dz = VOXEL
    Y, X, Z = np.indices(cube.shape)
    Y, X, Z = Y * dy, X * dx, Z * dz
    cy, cx, cz = centre[0] * dy, centre[1] * dx, centre[2] * dz
    ins = (np.abs(Y - cy) <= 1.0) & (np.abs(X - cx) <= 1.0) & (np.abs(Z - cz) <= 3.0)
    core = (np.abs(Y - cy) <= .45) & (np.abs(X - cx) <= .45) & (np.abs(Z - cz) <= 1.2)
    sh = ins & ~core & np.isfinite(cube)
    if sh.sum() < 50:
        finite = cube[np.isfinite(cube)]
        return np.full(cube.shape, float(np.median(finite)) if finite.size else 0.0)
    A = np.column_stack([np.ones(sh.sum()), Y[sh], X[sh], Z[sh]])
    q, *_ = np.linalg.lstsq(A, cube[sh], rcond=None)
    return q[0] + q[1] * Y + q[2] * X + q[3] * Z


def _at(a, i):
    return float(a[int(np.clip(round(i[0]), 0, a.shape[0] - 1)),
                   int(np.clip(round(i[1]), 0, a.shape[1] - 1)),
                   int(np.clip(round(i[2]), 0, a.shape[2] - 1))])


def occupancy(cube, fit_yxz, arg_yxz):
    bg = _bg_plane(cube, arg_yxz)
    den = _at(cube, arg_yxz) - _at(bg, arg_yxz)
    if den <= 0:
        return np.nan
    return (_at(cube, fit_yxz) - _at(bg, fit_yxz)) / den


def run_variant(cube, z_centre, variant, shape):
    """(y, x, z) in full-crop voxel indices, plus at_bound flag, or None."""
    cy, cx = (cube.shape[0] - 1) / 2.0, (cube.shape[1] - 1) / 2.0
    if variant == 'v1':
        iz = int(np.unravel_index(int(np.nanargmax(cube)), cube.shape)[2])
        r = L.fit_gaussian_3d(cube, float(cx), float(cy), float(iz))
        if r is None:
            return None
        railed = abs(abs(r[3] - iz) - 2.0) < 1e-6
        return (r[2], r[1], r[3]), railed

    box, (oy, ox, oz) = U.extract_box(cube, (cy, cx, z_centre), BOX_HALF)
    bc = tuple((s - 1) / 2.0 for s in box.shape)
    seed = U.intensity_centroid(box, bc, BOX_HALF, voxel_um=VOXEL)
    if seed is None:
        return None
    if variant == 'v2-geom':
        f = U.fit_gaussian_3d_um(box, seed[0], seed[1], seed[2], voxel_um=VOXEL,
                                 peak_bound_um=PB_XY, peak_bound_z_um=PB_Z,
                                 max_sigma_xy_um=3.0, max_sigma_z_um=6.0,
                                 fit_radius_um=FIT_RADIUS, background='linear',
                                 apply_gates=False)
        if f is None:
            return None
        railed = any(s in ('x', 'y', 'z') for s in f.at_bound)
        return (f.y + oy, f.x + ox, f.z + oz), railed

    noise = 'poisson' if variant.endswith('mle') else 'gaussian'
    f = M.fit_gaussian_3d_mle(box, seed[0], seed[1], seed[2], voxel_um=VOXEL,
                              family=shape['family'], shape_params=shape['params'],
                              free_shape=False, noise=noise,
                              peak_bound_um=PB_XY, peak_bound_z_um=PB_Z,
                              fit_radius_um=FIT_RADIUS, background='linear')
    if f is None:
        return None
    railed = any(s in ('x', 'y', 'z') for s in f.at_bound)
    return (f.y + oy, f.x + ox, f.z + oz), railed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bench')
    ap.add_argument('--limit', type=int, default=None, help='alleles to use')
    ap.add_argument('--calib-crops', type=int, default=40)
    a = ap.parse_args()

    b = np.load(a.bench, allow_pickle=False)
    meta = b['__meta__'][:a.limit] if a.limit else b['__meta__']
    offsets = offsets_of(b)

    # -- what does calibrating the PSF cost, and on how much data? --
    calib = {}
    for kind in ('fiducial', 'readout'):
        keys = [k for k in b.files if k.endswith(f'|Hyb_016|{kind}')][:a.calib_crops]
        crops = [b[k].astype(float) for k in keys]
        # Place each calibration box by the SAME consensus depth the fit
        # uses, not by that crop's own pillar argmax. Calibrating on
        # boxes placed by the method known to misplace them would bias
        # the shape being measured.
        z_centres = []
        for k in keys:
            aid = int(k.split('|')[0][1:])
            fid = {kk.split('|')[1]: b[kk].astype(float) for kk in b.files
                   if kk.startswith(f'a{aid}|') and kk.endswith('|fiducial')}
            base = baseline(fid, offsets, aid)
            z_centres.append(None if base is None
                             else base - offsets.get(f'a{aid}|Hyb_016', 0.0))
        t0 = time.perf_counter()
        res = P.calibrate(crops, voxel_um=VOXEL, verbose=False, z_centres=z_centres)
        dt = time.perf_counter() - t0
        best = res['best']
        names = P.FAMILIES[best][1]
        calib[kind] = {'family': best,
                       'params': tuple(res[best]['params'][n] for n in names),
                       'n': len(crops), 'seconds': dt}
        pretty = '  '.join(f'{n}={res[best]["params"][n]:.4f}' for n in names)
        print(f'PSF calibration [{kind}]: {best}  {pretty}')
        print(f'   from {len(crops)} crops in {dt:.1f}s  '
              f'({dt / max(len(crops), 1):.2f}s per crop, ONCE per experiment)')
    print()

    variants = ['v1', 'v2-geom', 'v2-geom+psf', 'v2-geom+psf+mle']
    # positions are collected for BOTH channels before scoring, because the
    # replicate score needs each variant's OWN fiducial fits: the pipeline
    # reports a readout as position + (fiducial(ref) - fiducial(hybe)), and
    # in a pair difference the reference and the crop origins cancel, so
    # the correction is each readout minus its own hybe's fiducial. Scoring
    # v2 readouts against v1 fiducials would measure neither variant.
    acc = {(v, k): {'occ': [], 'rail': [], 'n': 0, 't': 0.0, 'pos': {}}
           for v in variants for k in ('fiducial', 'readout')}
    for aid, *_r in meta:
        fid = {k.split('|')[1]: b[k].astype(float)
               for k in b.files
               if k.startswith(f'a{aid}|') and k.endswith('|fiducial')}
        base = baseline(fid, offsets, aid)
        if base is None:
            continue
        for kind in ('fiducial', 'readout'):
            shape = calib[kind]
            for k in [k for k in b.files
                      if k.startswith(f'a{aid}|') and k.endswith(f'|{kind}')]:
                hybe = k.split('|')[1]
                cube = b[k].astype(float)
                off = offsets.get(f'a{aid}|{hybe}', 0.0)
                arg = np.unravel_index(int(np.nanargmax(cube)), cube.shape)
                for v in variants:
                    slot = acc[(v, kind)]
                    t0 = time.perf_counter()
                    out = run_variant(cube, base - off, v, shape)
                    slot['t'] += time.perf_counter() - t0
                    if out is None:
                        continue
                    pos, railed = out
                    slot['n'] += 1
                    slot['occ'].append(occupancy(cube, pos, arg))
                    slot['rail'].append(railed)
                    slot['pos'][(aid, hybe)] = np.array(pos) * np.array(VOXEL)

    for kind in ('fiducial', 'readout'):
        print(f'--- {kind} ---')
        hdr = (f'{"variant":<18}{"fits":>7}{"occupancy":>11}{"occ<0.2":>9}'
               f'{"at bound":>10}{"time":>9}')
        print(hdr)
        print('-' * len(hdr))
        for v in variants:
            slot = acc[(v, kind)]
            o = np.array([x for x in slot['occ'] if np.isfinite(x)])
            r = np.array(slot['rail'])
            if not len(o):
                print(f'{v:<18}{"none":>7}')
                continue
            print(f'{v:<18}{slot["n"]:>7}{np.median(o):>11.3f}'
                  f'{100 * np.mean(o < 0.2):>8.0f}%{100 * np.mean(r):>9.0f}%'
                  f'{slot["t"]:>8.1f}s')
        print()

    print('REPLICATE SCORE -- readout, 3D, fiducial-corrected (never a gate input)')
    print(f'{"variant":<18}{"pairs":>7}{"median um":>12}{"p90":>9}')
    print('-' * 46)
    for v in variants:
        rd, fd = acc[(v, 'readout')]['pos'], acc[(v, 'fiducial')]['pos']
        d = []
        for aid, *_r in meta:
            for x, y in REPLICATES:
                ra, rb = rd.get((aid, x)), rd.get((aid, y))
                ga, gb = fd.get((aid, x)), fd.get((aid, y))
                if ra is None or rb is None or ga is None or gb is None:
                    continue
                d.append(float(np.linalg.norm((ra - ga) - (rb - gb))))
        d = np.array(d) if d else np.array([np.nan])
        print(f'{v:<18}{len(d):>7}{np.nanmedian(d):>12.4f}'
              f'{np.nanpercentile(d, 90):>9.4f}')


if __name__ == '__main__':
    main()
