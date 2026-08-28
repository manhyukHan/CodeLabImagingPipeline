"""
Tests for the v2 tracing path.

The failures worth pinning are the ones that still produce numbers:
routing to the wrong engine, deriving the consensus depth in the wrong
frame, and applying one channel's gates to the other.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_tracing_v2.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                              # noqa: E402

from codelab_pipeline.localization import tracing_v2 as V2      # noqa: E402
from codelab_pipeline.localization import psf_library as LIB    # noqa: E402

PASS, FAIL = [], []
VOXEL = (0.208, 0.208, 0.2)


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}' + (f'   [{detail}]' if detail else ''))


def blob(shape=(17, 17, 61), centre=(8., 8., 30.), sigma=(2., 2., 4.),
         amp=900., bg=120., seed=0):
    rng = np.random.default_rng(seed)
    iy, ix, iz = np.indices(shape)
    cy, cx, cz = centre
    sy, sx, sz = sigma
    g = amp * np.exp(-(((iy - cy) / sy) ** 2 + ((ix - cx) / sx) ** 2
                       + ((iz - cz) / sz) ** 2) / 2)
    return g + bg + rng.normal(0, 4.0, shape)


def main():
    print('engine routing')
    for name, want in (('v1 (ChrTracer3 port)', False), ('v2', True),
                       ('v2 (calibrated PSF)', True), ('V2', True),
                       ('', False), (None, False), ('v10', False)):
        check(f'is_v2({name!r}) is {want}', V2.is_v2(name) is want)

    # The dispatcher must actually CALL the other module for v1 -- routing
    # that silently falls through to v2 would look like a working switch.
    from unittest import mock
    from codelab_pipeline.localization import localization as L
    with mock.patch.object(L, 'build_chromatin_trace_allele',
                           return_value=('V1', None)) as m:
        got, _ = V2.trace_allele('v1 (ChrTracer3 port)', None, [], 'R', {}, {},
                                 '/nowhere', 1, 'DNA', None, {})
    check('v1 routes to localization.build_chromatin_trace_allele',
          got == 'V1' and m.call_count == 1)
    with mock.patch.object(L, 'build_chromatin_trace_allele') as m2:
        allele = mock.Mock(coordinate=(0., 0., 0.), fiducial_trace_adj={},
                           polymer_adj={}, rejected_hybes={})
        V2.trace_allele('v2', allele, [], 'R', {}, {}, '/nowhere', 1, 'DNA',
                        None, {})
    check('v2 does NOT call the v1 path', m2.call_count == 0)

    print('\nconsensus depth (the frame bug that still produces numbers)')
    # Two hybes whose fiducials sit at the SAME shared depth but different
    # native depths. A median taken in the native frame would return one
    # number for both; the shared-frame derivation must give them back
    # their own depths.
    a = blob(centre=(8., 8., 30.), seed=1)     # native peak 30, offset 0
    b = blob(centre=(8., 8., 20.), seed=2)     # native peak 20, offset +10
    z = V2.consensus_native_z({'A': a, 'B': b}, {'A': 0.0, 'B': 10.0})
    check('each hybe gets its OWN expected depth back',
          abs(z['A'] - 30) < 0.6 and abs(z['B'] - 20) < 0.6,
          f'A={z["A"]:.1f} (want 30)  B={z["B"]:.1f} (want 20)')
    check('the two differ by exactly the offset difference',
          abs((z['A'] - z['B']) - 10.0) < 1e-9, f'{z["A"] - z["B"]:.3f}')
    check('a hybe with no offset is simply absent, not defaulted to 0',
          'C' not in V2.consensus_native_z({'A': a}, {'A': 0.0}))
    check('no usable crops gives an empty mapping',
          V2.consensus_native_z({'A': None}, {'A': 0.0}) == {})

    print('\nthe two channels are gated separately')
    p = V2.V2Params(voxel_um=VOXEL)
    check('fiducial and readout have DIFFERENT default gates',
          p.fiducial_gates != p.readout_gates,
          f'fid occ {p.fiducial_gates["min_occupancy"]} vs '
          f'readout {p.readout_gates["min_occupancy"]}')
    check('the fiducial gate is the looser of the two (extended object)',
          p.fiducial_gates['min_occupancy'] < p.readout_gates['min_occupancy'])
    # at_bound is a READOUT gate. It was on for both until it was measured
    # on the fiducial side: over 9007 permissively-fitted readouts (MP58
    # FOV1, 127 alleles), a readout z-uncert <= 600 nm gate reached 375
    # pairs at p90 735 nm where fiducial at_bound reached 378 at 794 --
    # same coverage, better tail, and tunable. Of the 798 readouts the
    # fiducial gate removed, only 20% were the ones a readout gate would
    # remove; the rest were not the damaging ones.
    check('at_bound rejects on the READOUT, whose measurement it is',
          p.readout_gates['reject_at_bound'] is True)
    check('and NOT on the fiducial, where it measured worse than a '
          'readout uncertainty gate at matched coverage',
          p.fiducial_gates['reject_at_bound'] is False)
    check('neither inherits v1 min_hb_ratio / min_ah_ratio',
          'min_hb_ratio' not in p.readout_gates
          and 'min_ah_ratio' not in p.readout_gates)
    check('overrides merge over the defaults rather than replacing them',
          V2.V2Params(readout_gates={'min_occupancy': 0.9}
                      ).readout_gates['reject_at_bound'] is True)

    print('\ngating behaviour')
    cube = blob(seed=3)
    f = V2.fit_fiducial(cube, 30, p)
    ok, why = V2.gate(f, cube, p.fiducial_gates, VOXEL)
    check('a clean fiducial passes its own gates', ok, str(why))
    check('occupancy on a clean fit is near 1', V2.occupancy(cube, f) > 0.9,
          f'{V2.occupancy(cube, f):.3f}')
    check('a failed fit is rejected with a reason, not an exception',
          V2.gate(None, cube, p.readout_gates)[0] is False
          and 'fit failed' in V2.gate(None, cube, p.readout_gates)[1])

    # Parameter names as fit3d_um.py:245 emits them. ON the emitter (the
    # crop's peak is at 8, 8, 30), so the ONLY thing wrong is the bound --
    # a stub in background would fail the occupancy gate instead and the
    # at_bound test would pass for the wrong reason.
    def _railed(*names):
        class _R(object):
            at_bound = names
            y = x = 8.0
            z = 30.0
            ci_y_um = ci_x_um = ci_z_um = 0.001
            amplitude = 100.0
        return _R()

    # LATERAL position only. The readout box is pre-placed at the
    # fiducial-derived consensus depth, so a railed z measures how far
    # this locus sits from the median fiducial plane -- allele geometry,
    # not data quality (a real case railed on z while reporting CIs of
    # xy 18 / z 57 nm). Measured at the 2.8 um bound: making z fatal
    # keeps 438 pairs at p90 1553 against 443 at 1605 -- ~1-3%, which the
    # z-uncertainty gate covers properly when wanted.
    ok2, why2 = V2.gate(_railed('y'), cube, p.readout_gates, VOXEL)
    check('readout: a railed LATERAL position is rejected before any other gate',
          ok2 is False and 'at bound' in why2, str(why2))
    ok2z, why2z = V2.gate(_railed('z'), cube, p.readout_gates, VOXEL)
    check('readout: a railed z alone is NOT fatal -- it reflects distance '
          'from the consensus depth, not fit quality',
          ok2z is True, str(why2z))
    ok2f, why2f = V2.gate(_railed('z'), cube, p.fiducial_gates, VOXEL)
    check('fiducial: a railed POSITION is NOT rejected -- it passes to the '
          'readout gates, which measure round quality better',
          ok2f is True, str(why2f))
    # Switched back on, the fiducial must still reject POSITION and only
    # position -- the key survives the default change for exactly this.
    on = dict(p.fiducial_gates, reject_at_bound=True)
    check('fiducial: with the filter switched back on, position IS fatal',
          V2.gate(_railed('z'), cube, on, VOXEL)[0] is False)
    for label, g in (('fiducial', on), ('readout', p.readout_gates)):
        # sigma at a bound is NOT fatal for either channel: the swept filter
        # was `any(s in ('x','y','z') ...)` (gate_sweep_v2.py:159), and a
        # fiducial's sigma pins to whatever ceiling exists while its
        # centroid stays on the emitter.
        ok3, why3 = V2.gate(_railed('sigma_y', 'sigma_x', 'offset'), cube, g, VOXEL)
        check(f'{label}: a railed SIGMA or OFFSET is not, on its own, fatal',
              ok3 is True, str(why3))
    ok4, _ = V2.gate(_railed('z'), cube,
                     dict(p.readout_gates, reject_at_bound=False), VOXEL)
    check('and the at_bound filter can be turned off deliberately', ok4 is True)

    tight = dict(p.readout_gates, max_uncert_xy_nm=0.001)
    check('the lateral uncertainty gate is applied in NANOMETRES',
          V2.gate(f, cube, tight, VOXEL)[0] is False
          and 'lateral uncertainty' in V2.gate(f, cube, tight, VOXEL)[1])
    tightz = dict(p.readout_gates, max_uncert_z_nm=0.001)
    check('the axial gate is independent of the lateral one',
          V2.gate(f, cube, tightz, VOXEL)[0] is False
          and 'axial uncertainty' in V2.gate(f, cube, tightz, VOXEL)[1])

    print('\nz placement')
    # A/B-tested at the shipping gates (MP58 FOV1, 127 alleles): consensus
    # 230 pairs @ 129/320 nm against self 204 @ 126/399, and self kept
    # FEWER fiducials (7288 vs 7694) -- the single-pillar centroid is
    # noisier than the pooled median. The switch stays for re-testing.
    check('fiducial boxes are placed by the cross-hybe consensus by default',
          V2.Z_PLACEMENT == 'consensus', V2.Z_PLACEMENT)
    zc = np.zeros((17, 17, 110))
    zc[8, 8, 80] = 900.0
    zc[7:10, 7:10, 78:83] += 120.0
    check('own_native_z finds an emitter far off the stack middle',
          abs(V2.own_native_z(zc, VOXEL) - 80.0) <= 2.0,
          f'{V2.own_native_z(zc, VOXEL):.1f}')
    check('own_native_z answers mid-stack for an all-NaN pillar, not a raise',
          V2.own_native_z(np.full((5, 5, 9), np.nan), VOXEL) == 4.0)

    print('\nthe readout PSF')
    p_free = V2.V2Params(voxel_um=VOXEL)
    check('no PSF means sigma is fitted per spot', not p_free.has_psf)
    doc = LIB.read('universal-default')
    if doc:
        fam, shape = LIB.shape_tuple(doc)
        p_psf = V2.V2Params(voxel_um=VOXEL, psf_family=fam, psf_shape=shape,
                            psf_label='universal-default')
        check('a library entry gives a usable fixed shape', p_psf.has_psf)
        r_free = V2.fit_readout(cube, 30, p_free)
        r_psf = V2.fit_readout(cube, 30, p_psf)
        check('both readout paths localize the same emitter',
              r_free is not None and r_psf is not None
              and abs(r_free.y - r_psf.y) < 1.0 and abs(r_free.z - r_psf.z) < 2.0,
              f'free ({r_free.y:.2f},{r_free.z:.2f}) vs '
              f'fixed ({r_psf.y:.2f},{r_psf.z:.2f})')
        check('describe() names the PSF actually in use',
              'universal-default' in p_psf.describe(), p_psf.describe())
    # Assert what the line must CARRY, not how it is worded. describe() is
    # the one record of what a run actually was, so the requirements are
    # that a free-sigma fallback is identifiable and the voxel size is
    # present in BOTH branches -- the earlier version omitted the voxel
    # size here, so a fallback run left no record of its configuration.
    free = p_free.describe()
    check('describe() names the free-sigma fallback when there is no PSF',
          'FREE' in free, free)
    check('and carries the voxel size in that branch too',
          '0.208' in free and '0.2' in free, free)
    rejected = V2.V2Params(voxel_um=VOXEL,
                           psf_label='some-entry [REJECTED: sigma below limit]')
    check('a REJECTED psf is reported as rejected, not as absent',
          'REJECTED' in rejected.describe(), rejected.describe())

    # from_panel must prefer the INSTALLED copy over the library, or a run
    # would silently follow the library when it moved on.
    store = tempfile.mkdtemp(prefix='v2store_')
    if doc:
        LIB.install('universal-default', store)
        pp = V2.V2Params.from_panel(
            {'voxel_um': VOXEL, 'readout_psf': 'no-such-entry'}, store)
        check('from_panel uses the INSTALLED psf, not the config label',
              pp.has_psf and pp.psf_label == 'universal-default', pp.psf_label)

    print('\nslab extraction')
    s = V2._slab(cube, 30, 15)
    check('a slab has the requested depth', s is not None and s.shape[2] == 31,
          str(None if s is None else s.shape))
    edge = V2._slab(cube, 2, 15)
    check('a slab running off the stack is NaN-PADDED, not clipped',
          edge is not None and edge.shape[2] == 31 and np.isnan(edge).any(),
          str(None if edge is None else edge.shape))
    check('a slab entirely outside the stack is None',
          V2._slab(cube, -500, 5) is None)

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        for f_ in FAIL:
            print('  FAILED:', f_)
        return 1
    print('ALL GOOD')
    return 0


if __name__ == '__main__':
    sys.exit(main())
