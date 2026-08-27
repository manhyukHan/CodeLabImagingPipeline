"""
Tests for the shape-free fiducial estimators.

The properties worth pinning are the ones a wrong answer still looks
right for: a SIGN convention (a delta applied backwards doubles the drift
instead of removing it, and still produces numbers), NaN padding treated
as dark signal, and a size estimate that silently reports the box rather
than the object.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_shapefree.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                             # noqa: E402

from codelab_pipeline.localization import shapefree as SF      # noqa: E402

PASS, FAIL = [], []
VOXEL = (0.208, 0.208, 0.2)


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}' + (f'   [{detail}]' if detail else ''))


def blob(shape=(21, 21, 41), centre=(10.0, 10.0, 20.0),
         sigma=(2.0, 2.0, 4.0), amp=500.0, bg=100.0, seed=0, extended=0.0):
    """A blob, optionally with a broad extended pedestal on top of it --
    the thing a PSF family cannot describe and this module must not care
    about."""
    rng = np.random.default_rng(seed)
    iy, ix, iz = np.indices(shape)
    cy, cx, cz = centre
    sy, sx, sz = sigma
    g = amp * np.exp(-(((iy - cy) / sy) ** 2 + ((ix - cx) / sx) ** 2
                       + ((iz - cz) / sz) ** 2) / 2.0)
    if extended:
        g = g + extended * amp * np.exp(
            -(((iy - cy) / (sy * 4)) ** 2 + ((ix - cx) / (sx * 4)) ** 2
              + ((iz - cz) / (sz * 3)) ** 2) / 2.0)
    return g + bg + rng.normal(0, 3.0, shape)


def main():
    print('shapefree.shift_yxz')

    # -- recovers a known shift, sub-voxel ------------------------------
    truth = (1.30, -2.20, 3.40)
    ref = blob(seed=1)
    mov = blob(centre=(10.0 - truth[0], 10.0 - truth[1], 20.0 - truth[2]), seed=2)
    got, q = SF.shift_yxz(ref, mov)
    err = max(abs(g - t) for g, t in zip(got, truth))
    check('recovers a known sub-voxel shift to better than 0.15 voxel',
          got is not None and err < 0.15,
          f'want {truth}, got {tuple(round(v, 2) for v in got)}, max err {err:.3f}')
    check('reports a high correlation for a genuine match', q > 0.9, f'q={q:.3f}')

    # -- THE SIGN. Applied backwards this doubles the error ------------
    #
    # The contract: the returned shift is what must be ADDED to a position
    # measured in `moving` to express it in `reference`'s frame. The blob
    # sits at 10-truth in moving and at 10 in reference, so adding the
    # shift to the moving centroid must land on the reference centroid.
    m_mov = SF.moments_yxz(mov, VOXEL, background=0.9)
    m_ref = SF.moments_yxz(ref, VOXEL, background=0.9)
    moved = [c + s for c, s in zip(m_mov['centroid_yxz'], got)]
    resid = max(abs(a - b) for a, b in zip(moved, m_ref['centroid_yxz']))
    check('the sign is such that ADDING the shift maps moving -> reference',
          resid < 0.2, f'residual {resid:.3f} voxel (backwards would be ~2x truth)')

    # -- identical input means no shift ---------------------------------
    same, qs = SF.shift_yxz(ref, ref)
    check('identical crops give a zero shift',
          max(abs(v) for v in same) < 1e-6, str(tuple(round(v, 4) for v in same)))
    check('identical crops correlate at 1.0', qs > 0.999, f'q={qs:.4f}')

    # -- an EXTENDED object registers at least as well as a point -------
    #
    # This is the whole argument for the approach: a PSF fit is degraded
    # by structure it cannot model, registration is helped by it.
    t2 = (0.7, 1.4, -2.1)
    for label, ext in (('point-like', 0.0), ('extended', 0.8)):
        r = blob(seed=11, extended=ext)
        m = blob(centre=(10.0 - t2[0], 10.0 - t2[1], 20.0 - t2[2]),
                 seed=12, extended=ext)
        g, _q = SF.shift_yxz(r, m)
        e = max(abs(a - b) for a, b in zip(g, t2))
        check(f'registers an {label} object to better than 0.2 voxel', e < 0.2,
              f'max err {e:.3f}')

    # -- NaN padding is not dark signal ---------------------------------
    #
    # The hole has to overlap the OBJECT for this to test anything. Placed
    # in the background the two treatments agree exactly (both measured
    # 0.000), because there is nothing there for either to get wrong --
    # which is what the first version of this check did, and it proved
    # nothing while looking like it passed.
    ref2 = blob(seed=4)

    # (a) padding in the BACKGROUND -- the routine case at a slab edge --
    #     must not disturb the answer at all.
    edge = blob(seed=3)
    edge[:, :, :6] = np.nan
    g_edge, _ = SF.shift_yxz(ref2, edge)
    check('NaN padding in the background leaves the shift unchanged',
          g_edge is not None and max(abs(v) for v in g_edge) < 0.1,
          str(tuple(round(v, 3) for v in g_edge)) if g_edge else 'None')
    check('and that crop is scored as fully covered',
          SF.signal_coverage(ref2, edge) > 0.99,
          f'coverage {SF.signal_coverage(ref2, edge):.3f}')

    # (b) a hole through the OBJECT is NOT repairable. Filling it with the
    #     background is byte-identical to filling it with zero once
    #     negatives are clipped, and both bias the axial shift by -1.3
    #     voxels. So the requirement is that it be REJECTED, not that some
    #     cleverer fill rescue it.
    holed = blob(seed=3)
    holed[:, :, 14:18] = np.nan
    cov = SF.signal_coverage(ref2, holed)
    check('a hole through the object is scored as low coverage',
          cov < 0.9, f'coverage {cov:.3f}')
    g_hole, q_hole = SF.shift_yxz(ref2, holed)
    check('and is REJECTED rather than returning a confident wrong shift',
          g_hole is None and q_hole == 0.0,
          str(g_hole))
    check('with the rejection lifted, it would indeed have been wrong',
          max(abs(v) for v in SF.shift_yxz(ref2, holed, min_coverage=None)[0]) > 0.5,
          str(tuple(round(v, 2) for v in
                    SF.shift_yxz(ref2, holed, min_coverage=None)[0])))

    # -- all-NaN and mismatched shapes fail loudly, not silently --------
    bad, qb = SF.shift_yxz(ref, np.full_like(ref, np.nan))
    check('an all-NaN crop returns None, not a number', bad is None and qb == 0.0)
    bad2, _ = SF.shift_yxz(ref, blob(shape=(11, 11, 21)))
    check('mismatched shapes return None', bad2 is None)

    # -- max_shift rejects rather than returning a wild answer ----------
    far = blob(centre=(3.0, 17.0, 34.0), seed=5)
    lim, _ = SF.shift_yxz(ref, far, max_shift_vox=(2, 2, 2))
    check('a shift beyond max_shift_vox is rejected', lim is None)

    print('\nshapefree.moments_yxz')

    # -- a bigger object measures bigger --------------------------------
    small = SF.moments_yxz(blob(sigma=(2.0, 2.0, 4.0), seed=6), VOXEL, background=0.9)
    big = SF.moments_yxz(blob(sigma=(4.0, 4.0, 4.0), seed=6), VOXEL, background=0.9)
    check('a laterally wider object gives a larger lateral Rg',
          big['rg_xy_um'] > small['rg_xy_um'] * 1.2,
          f'{small["rg_xy_um"]*1000:.0f} nm -> {big["rg_xy_um"]*1000:.0f} nm')

    # -- the background threshold is load-bearing ------------------------
    #
    # Without it the crop's own background dominates the second moment and
    # every object measures the size of the BOX, which is the failure this
    # estimator exists to avoid.
    a_thr = SF.moments_yxz(blob(sigma=(2., 2., 4.), seed=7), VOXEL, background=0.9)
    b_thr = SF.moments_yxz(blob(sigma=(4., 4., 4.), seed=7), VOXEL, background=0.9)
    a_no = SF.moments_yxz(blob(sigma=(2., 2., 4.), seed=7), VOXEL, background=None)
    b_no = SF.moments_yxz(blob(sigma=(4., 4., 4.), seed=7), VOXEL, background=None)
    sep_thr = b_thr['rg_xy_um'] / a_thr['rg_xy_um']
    sep_no = b_no['rg_xy_um'] / a_no['rg_xy_um']
    check('thresholding separates two sizes better than not thresholding',
          sep_thr > sep_no,
          f'ratio with threshold {sep_thr:.2f} vs without {sep_no:.2f}')

    # -- centroid agrees with the truth ---------------------------------
    m = SF.moments_yxz(blob(centre=(11.5, 9.25, 22.75), seed=8), VOXEL,
                       background=0.9)
    e = max(abs(a - b) for a, b in zip(m['centroid_yxz'], (11.5, 9.25, 22.75)))
    check('the centroid lands on the true centre within 0.2 voxel', e < 0.2,
          f'max err {e:.3f}')

    # -- an aperture is honoured and reported ---------------------------
    m2 = SF.moments_yxz(blob(seed=9, extended=1.0), VOXEL,
                        aperture_um=(0.6, 0.6, 1.5), background=0.9)
    m3 = SF.moments_yxz(blob(seed=9, extended=1.0), VOXEL,
                        aperture_um=(1.2, 1.2, 3.0), background=0.9)
    check('a wider aperture measures a heavy-tailed object as larger',
          m3['rg_xy_um'] > m2['rg_xy_um'],
          f'{m2["rg_xy_um"]*1000:.0f} -> {m3["rg_xy_um"]*1000:.0f} nm')
    check('the aperture is reported alongside the size, never bare',
          m2['aperture_um'] == (0.6, 0.6, 1.5))

    check('an all-NaN crop gives None rather than a size',
          SF.moments_yxz(np.full((9, 9, 9), np.nan), VOXEL) is None)

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        for f in FAIL:
            print('  FAILED:', f)
        return 1
    print('ALL GOOD')
    return 0


if __name__ == '__main__':
    sys.exit(main())
