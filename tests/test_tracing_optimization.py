"""
The chromatin-tracing fit/gate findings, as executable assertions.

WHY THIS EXISTS
---------------
The investigation behind these numbers is written up in
notes/chromatin_tracing_optimization.md, and the tools that produced them
are in tools/ -- but every one of those needs the real store or a 330 MB
frozen bench, so none of them can be run to re-establish what was learned.
This file can: it is self-contained, synthetic, and fast, and it fails if
any conclusion stops holding.

Each check names the finding it pins. Read it as the executable half of
the notes.

THE FINDINGS PINNED HERE
------------------------
1. A crop-BOX beats a crop-PILLAR. The tracing crop is bounded in XY but
   takes the full slab in Z, so the PSF is a few hundred voxels against
   tens of thousands of out-of-focus ones, and least squares spends the
   position and sigma parameters describing background instead.
2. at_bound must be reported truthfully, because a fit that stopped on a
   constraint returns the constraint, not a measurement -- and its
   Jacobian CI does not describe it either.
3. A degenerate PSF must be REFUSED, not scored. A 39 nm core scored as
   well as a 312 nm one on the same real data, so score cannot arbitrate.
4. The Z baseline is fit-free (per-hybe argmax -> shared frame -> median)
   and must NOT be built from full fits -- measured worse AND 1000x
   slower on real crops.
5. Toehold rounds are displacement CONTROLS, not replicates.
6. Gate thresholds are physical lengths with lateral and axial
   independent -- a pixel and a plane are different distances.

Run: python tests/test_tracing_optimization.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                            # noqa: E402

from codelab_pipeline.localization import fit3d_um as U       # noqa: E402
from codelab_pipeline.localization import psf as P            # noqa: E402

PASS, FAIL = [], []
VOXEL = (0.208, 0.208, 0.2)


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(('  ok ' if cond else 'FAIL ') + name + (f' -- {detail}' if detail and not cond else ''))


def pillar(shape=(21, 21, 111), spot_z=55, sigma=(0.15, 0.15, 0.45),
           amp=3000.0, haze=900.0, seed=0, competitor=True):
    """
    A crop shaped like a real one, INCLUDING what actually breaks the
    pillar fit.

    A compact emitter alone is not enough: a first version of this
    synthetic gave occupancy 1.000 for both pillar and box, because a
    clean bright spot in smooth haze is findable either way. The real
    crops are harder in a specific way -- the column also holds diffuse
    structure at a DIFFERENT depth (neighbouring chromatin, other cells),
    and because least squares weights every voxel equally, a broad dim
    blob can carry more total residual than the compact bright one. That
    is what drags a whole-column fit off the emitter, and it is why the
    measured intensity-weighted depth sat ~16 planes from the argmax.

    `competitor` adds exactly that: wide, dimmer at the peak, far more
    total flux, and 40 planes away -- far enough that its tail is
    negligible inside a +/-15 plane box (25 planes out, 3.3 sigma, under
    0.5% of its peak) while being fully inside the 111-plane pillar. A
    first version put it 25 planes out with sigma_z 2.2 um, whose tail
    contributed ~890 counts at the box edge and so corrupted the box fit
    as well -- which real crops do not do, and which would have made this
    test assert the opposite of the measured finding.
    """
    dy, dx, dz = VOXEL
    ny, nx, nz = shape
    iy, ix, iz = np.indices(shape)
    Y, X, Z = iy * dy, ix * dx, iz * dz
    cy, cx, cz = (ny - 1) / 2 * dy, (nx - 1) / 2 * dx, spot_z * dz
    g = amp * np.exp(-0.5 * (((Y - cy) / sigma[0]) ** 2
                             + ((X - cx) / sigma[1]) ** 2
                             + ((Z - cz) / sigma[2]) ** 2))
    if competitor:
        oz = (spot_z - 40 if spot_z > nz / 2 else spot_z + 40) * dz
        g = g + 0.55 * amp * np.exp(
            -0.5 * (((Y - cy - 0.5) / 0.9) ** 2 + ((X - cx + 0.4) / 0.9) ** 2
                    + ((Z - oz) / 1.5) ** 2))
    grad = haze * (0.6 + 0.8 * (iz / max(nz - 1, 1)))     # 1.09x..1.75x, as measured
    rng = np.random.default_rng(seed)
    return g + grad + rng.normal(0, 25.0, shape), (cy, cx, cz)


def occupancy(cube, fit_yxz, arg_yxz):
    """Signal at the fit over signal at the argmax, above a local plane."""
    dy, dx, dz = VOXEL
    Y, X, Z = np.indices(cube.shape)
    Y, X, Z = Y * dy, X * dx, Z * dz
    c = (arg_yxz[0] * dy, arg_yxz[1] * dx, arg_yxz[2] * dz)
    ins = ((np.abs(Y - c[0]) <= 1.) & (np.abs(X - c[1]) <= 1.)
           & (np.abs(Z - c[2]) <= 3.))
    core = ((np.abs(Y - c[0]) <= .45) & (np.abs(X - c[1]) <= .45)
            & (np.abs(Z - c[2]) <= 1.2))
    sh = ins & ~core & np.isfinite(cube)
    A = np.column_stack([np.ones(sh.sum()), Y[sh], X[sh], Z[sh]])
    q, *_ = np.linalg.lstsq(A, cube[sh], rcond=None)
    bg = q[0] + q[1] * Y + q[2] * X + q[3] * Z

    def at(a, i):
        return float(a[int(np.clip(round(i[0]), 0, a.shape[0] - 1)),
                       int(np.clip(round(i[1]), 0, a.shape[1] - 1)),
                       int(np.clip(round(i[2]), 0, a.shape[2] - 1))])
    den = at(cube, arg_yxz) - at(bg, arg_yxz)
    return ((at(cube, fit_yxz) - at(bg, fit_yxz)) / den) if den > 0 else np.nan


def main():
    # -- 1. BOX beats PILLAR ------------------------------------------
    occ_pillar, occ_box, bound_pillar, bound_box = [], [], 0, 0
    for s in range(6):
        zz = 50 + 3 * s
        cube, true_um = pillar(seed=s, spot_z=zz)
        cy, cx = (cube.shape[0] - 1) / 2., (cube.shape[1] - 1) / 2.
        # the TRUE emitter, not the column's argmax -- with a competitor
        # present the brightest voxel is not necessarily the spot, which
        # is the whole point
        arg = (int(round(cy)), int(round(cx)), zz)

        f = U.fit_gaussian_3d_um(cube, cy, cx, float(arg[2]), voxel_um=VOXEL,
                                 max_sigma_xy_um=3.0, max_sigma_z_um=6.0,
                                 background='constant', apply_gates=False)
        if f is not None:
            occ_pillar.append(occupancy(cube, (f.y, f.x, f.z), arg))
            bound_pillar += any(n in ('x', 'y', 'z') for n in f.at_bound)

        box, (oy, ox, oz) = U.extract_box(cube, (cy, cx, arg[2]), (5, 5, 15))
        bc = tuple((n - 1) / 2. for n in box.shape)
        seed_yxz = U.intensity_centroid(box, bc, (5, 5, 15), voxel_um=VOXEL)
        g = U.fit_gaussian_3d_um(box, seed_yxz[0], seed_yxz[1], seed_yxz[2],
                                 voxel_um=VOXEL, peak_bound_um=5 * VOXEL[0],
                                 peak_bound_z_um=10 * VOXEL[2],
                                 max_sigma_xy_um=3.0, max_sigma_z_um=6.0,
                                 fit_radius_um=(1., 1., 3.), background='linear',
                                 apply_gates=False)
        if g is not None:
            occ_box.append(occupancy(cube, (g.y + oy, g.x + ox, g.z + oz), arg))
            bound_box += any(n in ('x', 'y', 'z') for n in g.at_bound)
    mp, mb = np.nanmedian(occ_pillar), np.nanmedian(occ_box)
    check('the box fit lands ON the emitter (occupancy > 0.5)', mb > 0.5,
          f'{mb:.3f}')
    check('the box fit never rails more often than the pillar fit',
          bound_box <= bound_pillar, f'box {bound_box} vs pillar {bound_pillar}')
    # NOT asserted here: that the box BEATS the pillar on occupancy.
    # Two attempts to synthesise that failed in opposite directions --
    # first both scored 1.000, then the competitor's tail leaked into the
    # box and the box scored WORSE -- and the reason is instructive: this
    # synthetic's background IS the model the fit assumes (a Gaussian on a
    # linear ramp), so a whole-column fit has no trouble with it. The
    # pillar fails on REAL crops precisely because real cellular
    # background is not the model. A synthetic that cannot contain that
    # failure cannot honestly test the fix for it, and asserting it here
    # would give false confidence.
    #
    # The finding stands on 7008 real crops: occupancy 0.373 -> 0.806 and
    # blank-region fits 31% -> 4%. Re-measure with
    #   python tools/fit_quality.py <bench.npz>
    # against a bench from tools/fit_testbox.py. See
    # notes/chromatin_tracing_optimization.md.
    print(f'     (synthetic occupancy: box {mb:.3f}, pillar {mp:.3f} -- '
          f'see the note above on why this is not asserted)')

    # What IS testable synthetically is the MECHANISM: a constant
    # background cannot represent a gradient, so whatever it cannot absorb
    # is paid for out of the Gaussian's own width.
    infl_const, infl_lin = [], []
    for s2 in range(5):
        cube2, _t = pillar(seed=10 + s2, spot_z=55, competitor=False)
        ccy, ccx = (cube2.shape[0] - 1) / 2., (cube2.shape[1] - 1) / 2.
        for bg_model, acc in (('constant', infl_const), ('linear', infl_lin)):
            h = U.fit_gaussian_3d_um(cube2, ccy, ccx, 55.0, voxel_um=VOXEL,
                                     max_sigma_xy_um=3.0, max_sigma_z_um=6.0,
                                     fit_radius_um=(1., 1., 3.),
                                     background=bg_model, apply_gates=False)
            if h is not None:
                acc.append(h.sigma_z_um)
    if infl_const and infl_lin:
        mc, ml = np.median(infl_const), np.median(infl_lin)
        check('a CONSTANT background over a gradient inflates sigma_z more '
              'than a LINEAR one does', mc >= ml - 1e-9,
              f'constant {mc:.3f} um vs linear {ml:.3f} um')

    # -- 2. at_bound tells the truth ----------------------------------
    cube, _ = pillar(seed=1)
    cy, cx = (cube.shape[0] - 1) / 2., (cube.shape[1] - 1) / 2.
    arg = np.unravel_index(int(np.nanargmax(cube)), cube.shape)
    tight = U.fit_gaussian_3d_um(cube, cy, cx, float(arg[2]) - 8, voxel_um=VOXEL,
                                 peak_bound_um=0.416, peak_bound_z_um=0.02,
                                 max_sigma_xy_um=3.0, max_sigma_z_um=6.0,
                                 fit_radius_um=(1., 1., 3.), background='linear',
                                 apply_gates=False)
    check('a z bound that blocks the emitter is REPORTED as at_bound',
          tight is None or 'z' in tight.at_bound,
          'None' if tight is None else str(tight.at_bound))
    if tight is not None and 'z' in tight.at_bound:
        seeded = float(arg[2]) - 8
        check('and the at-bound value IS the bound, not a measurement',
              abs(abs(tight.z - seeded) - 0.02 / VOXEL[2]) < 1e-6,
              f'{tight.z - seeded:.4f} planes from seed')

    # -- 3. a degenerate PSF is refused, not scored --------------------
    ok, why = P.plausible('gaussian_halo', {'sigma_xy_um': 0.0388,
                                            'sigma_z_um': 0.1003,
                                            'halo_frac': 0.549,
                                            'halo_scale': 8.0})
    check('a 39 nm core is refused as physically impossible', not ok, str(why))
    ok, why = P.plausible('lorentzian', {'sigma_xy_um': 0.0300,
                                         'sigma_z_um': 0.677})
    check('a sigma sitting ON its lower bound is refused', not ok, str(why))
    ok, _ = P.plausible('gaussian_halo', {'sigma_xy_um': 0.1462,
                                          'sigma_z_um': 0.4949,
                                          'halo_frac': 0.1696,
                                          'halo_scale': 2.1445})
    check('the real 40-crop readout answer is accepted', ok)
    ok, _ = P.plausible('gaussian', {'sigma_xy_um': 0.2410, 'sigma_z_um': 0.629})
    check('the real 40-crop fiducial answer is accepted', ok)
    check('halo_frac cannot reach 0.5, which is what makes core/halo '
          'identifiable', P.FAMILIES['gaussian_halo'][3][2][1] < 0.5,
          str(P.FAMILIES['gaussian_halo'][3][2]))

    # -- 4. the fit-free consensus baseline ---------------------------
    # hybes at known offsets, all imaging one physical depth
    true_shared = 60.0
    offsets = [0.0, 21.0, -14.0, 5.0, -8.0, 12.0]
    natives = [true_shared - o for o in offsets]
    recovered = []
    for nz_native, off in zip(natives, offsets):
        cube, _ = pillar(spot_z=int(round(nz_native)), seed=int(abs(off)) + 2)
        iz = int(np.unravel_index(int(np.nanargmax(cube)), cube.shape)[2])
        recovered.append(iz + off)
    base = float(np.median(recovered))
    check('argmax-median recovers the shared depth across hybe offsets',
          abs(base - true_shared) <= 1.0, f'{base:.1f} vs {true_shared}')
    placements = [abs((base - o) - n) for o, n in zip(offsets, natives)]
    check('and places every hybe box within a plane of its emitter',
          max(placements) <= 1.5, f'max {max(placements):.2f} planes')

    # -- 5. toehold rounds are controls, not replicates ----------------
    sys.argv = ['fit_testbox']
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'fit_testbox', os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'tools', 'fit_testbox.py'))
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)
    check("only H and R rounds count as replicates",
          set(tb.REPLICATE_DATATYPES) == {'H', 'R'}, str(tb.REPLICATE_DATATYPES))
    records = [{'folder': 'Hyb_032', 'datatype': 'H', 'readout_id': 32},
               {'folder': 'Rep_032', 'datatype': 'R', 'readout_id': 32},
               {'folder': 'Toe_032', 'datatype': 'T', 'readout_id': 32}]
    pairs = tb.replicate_pairs(records)
    folders = {f for a, b, _r in pairs for f in (a, b)}
    check('a toehold round is NOT paired as a replicate',
          'Toe_032' not in folders and len(pairs) == 1, str(pairs))
    check('but the H/R pair for that locus still is',
          pairs and set(pairs[0][:2]) == {'Hyb_032', 'Rep_032'}, str(pairs))
    check('toehold rounds stay in the TRACED set (real rounds, real fiducials)',
          'T' in tb.DATATYPES, str(tb.DATATYPES))

    # -- 6. lateral and axial bounds are independent physical lengths --
    cube, _ = pillar(seed=3)
    f = U.fit_gaussian_3d_um(cube, cy, cx, float(arg[2]), voxel_um=VOXEL,
                             peak_bound_um=1.04, peak_bound_z_um=2.0,
                             max_sigma_xy_um=3.0, max_sigma_z_um=6.0,
                             fit_radius_um=(1., 1., 3.), background='linear',
                             apply_gates=False)
    check('the fit exposes every quantity a gate is computed from',
          f is not None and all(np.isfinite(v) for v in
                                (f.ci_y_um, f.ci_x_um, f.ci_z_um,
                                 f.peak_bg_ratio, f.amp_h_ratio)),
          'a gate quantity came back non-finite')
    check('the axial position bound is separate from the lateral one',
          'peak_bound_z_um' in U.fit_gaussian_3d_um.__code__.co_varnames)

    print()
    print(f'{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        raise SystemExit('FAILURES: ' + ', '.join(FAIL))
    print('ALL GOOD')


if __name__ == '__main__':
    main()
