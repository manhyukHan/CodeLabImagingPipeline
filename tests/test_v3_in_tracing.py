"""
v3 in chromatin tracing: the learned engine replaces the READOUT fit only.

WHY. With v3 selected, every dispatch asked is_v2(engine) and sent the
allele down the v1 tracer -- which has no readout engine. v1's gates
fired ('no readout peak accepted'), no multispot ever formed, and the
readout grid showed v1's numbers. The panel's own docstring had
predicted it: "a route the others do not know about is exactly how a v3
selection routes to v1 with no error".

WHAT v3 KEEPS FROM v2 is the point of the design: the fiducial fitted by
v2's Gaussian (one major spot, never a multispot search), the drift and
z-drift gates against the reference, and the readout crop cut around the
fiducial-mapped seed -- so a readout candidate is judged inside the SAME
box the fiducial anchors. On top, a candidate outside the fiducial's own
axial reach is dropped: a spot with a different z-drift from the
fiducial's is not this locus.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_v3_in_tracing.py
"""
import os
import pickle
import sys

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline.localization import tracing_v2 as T2          # noqa: E402
from codelab_pipeline.localization.engine import LocalizedSpot       # noqa: E402

CHECKS = [0, 0]
MODEL = 'D:/models/mp58_rna'
_APP = None


def app():
    # HELD AT MODULE LEVEL. A QApplication bound to a local dies with the
    # function, and the next widget built with none alive fail-fasts the
    # process with no traceback at all (0xC0000409).
    global _APP
    from PyQt5 import QtWidgets
    _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return _APP


def check(name, ok, note=''):
    CHECKS[0] += 1
    CHECKS[1] += 1 if ok else 0
    print(('  ok   ' if ok else '  FAIL ') + name
          + (('  -- ' + note) if note else ''))


def spot(y, x, z, p_exist, p=0.9, amp=100.0):
    nan = float('nan')
    return LocalizedSpot(y=y, x=x, z=z, p=p, amplitude=amp,
                         sigma_y=nan, sigma_x=nan, sigma_z=nan, offset=0.0,
                         p_exist=p_exist)


class FakeEngine(object):
    """Returns whatever it was given; records the crop it saw."""

    def __init__(self, spots):
        self.spots = list(spots)
        self.seen = None
        self.multispot_cal = None

    def localize(self, cube, seed_yxz=None, n_max=None):
        self.seen = np.asarray(cube).shape
        return list(self.spots)


def test_routing():
    print('routing')
    check('v1 does not use the v2 tracer', not T2.uses_v2_tracer(T2.ROUTE_V1))
    check('v2 does', T2.uses_v2_tracer(T2.ROUTE_V2))
    check('and v3 DOES -- it is the v2 path with a learned readout',
          T2.uses_v2_tracer(T2.ROUTE_V3) and T2.is_v3(T2.ROUTE_V3)
          and not T2.is_v2(T2.ROUTE_V3))
    check('a decorated label routes the same',
          T2.uses_v2_tracer('v3-psfmatcher (learned: classifier + matched PSF)'))
    import inspect
    src = inspect.getsource(T2.trace_allele)
    check('trace_allele dispatches on the tracer, not on is_v2',
          'if not uses_v2_tracer(engine):' in src)
    bsrc = inspect.getsource(T2.build_chromatin_trace_allele)
    check("the caller hands _readout_multi the fiducial's own position",
          'seed_yx=seed_yx' in bsrc and 'fid_local[hybe][0]' in bsrc)
    check('and a search crop wider than the display crop by the engine box',
          'pad=spad + BOX_R' in bsrc and 'display_offset=(oy, ox)' in bsrc)

    # v2 honours the two z controls too.
    pp = T2.V2Params.from_panel({'engine': T2.ROUTE_V2, 'v2': {},
                                 'z_window': 9, 'z_boundary_trim': 7}, None)
    check('from_panel carries z_window and the trim onto V2Params',
          pp.z_window == 9 and pp.z_reach() == 9 and pp.z_boundary_trim == 7
          and pp.readout_gates.get('z_boundary_trim') == 7
          and pp.fiducial_gates.get('z_boundary_trim') == 7)

    class Fit(object):
        at_bound = ()
        ci_y_um = ci_x_um = ci_z_um = 0.01

        def __init__(self, z):
            self.y, self.x, self.z = 8.0, 8.0, float(z)
    cube = np.random.RandomState(0).normal(300.0, 5.0, (17, 17, 105))
    ok_edge, why_edge = T2.gate(Fit(2.0), cube, {'z_boundary_trim': 10,
                                                 'min_occupancy': None},
                                T2.DEFAULT_VOXEL_UM)
    check("v2's gate rejects a fit inside the shaved planes",
          not ok_edge and 'of the stack end' in str(why_edge), str(why_edge))
    check('and stamps v3 as v3', "'v3' if is_v3(engine)" in src)
    for fn in (T2.allele_task, T2.allele_task_with_debug):
        s = inspect.getsource(fn)
        check(f'{fn.__name__} stamps the engine from the params',
              "params.is_learned" in s and "'readout_model_dir'" in s)


def test_params():
    print('V2Params carries a model directory, not an engine')
    p = T2.V2Params.from_panel(
        {'engine': T2.ROUTE_V3, 'engine_label': 'v3-psfmatcher (learned)',
         'v3': {'model_dir': MODEL, 'min_p_exist': 0.6}, 'v2': {}}, None)
    check('from_panel reads the v3 block under a v3 engine',
          p.readout_model_dir == MODEL and p.min_p_exist == 0.6
          and p.is_learned)
    check('describe() names it', p.describe().startswith('v3,')
          and '0.6' in p.describe())
    q = T2.V2Params.from_panel(
        {'engine': T2.ROUTE_V2, 'v3': {'model_dir': MODEL, 'min_p_exist': 0.6},
         'v2': {}}, None)
    check('but not under a v2 engine -- the v3 block is ignored',
          q.readout_model_dir is None and q.min_p_exist is None
          and not q.is_learned and q.readout_engine is None)

    # PICKLING: the engine never crosses; the directory does.
    p.readout_engine = FakeEngine([])
    d = pickle.loads(pickle.dumps(p))
    check('a pickled V2Params drops the built engine',
          d._readout_engine is None and d.readout_model_dir == MODEL)
    if os.path.isdir(MODEL):
        eng = d.readout_engine
        check('and rebuilds it lazily from the directory in the child',
              eng is not None and type(eng).__name__ == 'PsfMatcherV3Engine')
        check('once', d.readout_engine is eng)
    else:
        check('the shipped model is present', False, 'skipped')


def _run(engine, z_r=20.0, min_p_exist=0.5, cube=None, debug=None,
         seed_yx=(8.0, 8.0), z_window=None, z_boundary_trim=0,
         display_offset=(0, 0), display_shape=(17, 17)):
    from codelab_pipeline.models.allele import AnAllele
    a = AnAllele()
    a.polymer_adj, a.polymer_raw = {}, {}
    p = T2.V2Params(min_p_exist=min_p_exist, z_window=z_window,
                    z_boundary_trim=z_boundary_trim)
    p.readout_engine = engine
    if cube is None:
        cube = np.random.RandomState(0).normal(300.0, 5.0, (17, 17, 105))
    ok, why = T2._readout_multi(
        a, 'H', cube, z_r, p, 0.0, 0.0, 0.0, 100, 200,
        lambda h, y, x, z, ymin, xmin: (y + ymin, x + xmin, z), debug,
        seed_yx=seed_yx, display_offset=display_offset,
        display_shape=display_shape)
    return a, ok, why


def _slab_z0(z_r, reach, trim=0, depth=105):
    """Where the search slab starts -- the same rule _readout_multi uses,
    so a fake engine that echoes slab-local z can be fed absolute z."""
    from codelab_pipeline.localization.psfmatcher import BOX_RZ
    off = T2._z_trim_offset(depth, trim)
    return max(off, int(round(z_r)) - reach - BOX_RZ)


def test_readout_multi_gates():
    print('_readout_multi: the fiducial\'s reach, then p_exist, then refinement')
    z_half = T2._seed_z_half(T2.READOUT_FIT_RADIUS_UM, T2.DEFAULT_VOXEL_UM)
    r_lat = T2._lateral_reach_px(T2.READOUT_FIT_RADIUS_UM, T2.DEFAULT_VOXEL_UM)
    check('the axial reach is v2\'s own readout search half-depth',
          z_half >= 1, f'{z_half} planes')
    check('and the lateral reach is the same radius in pixels',
          r_lat >= 1, f'{r_lat} px')
    # THE ENGINE SEES THE SLAB, so a fake that echoes its inputs must
    # speak slab-local z. The slab is the reach PADDED by the engine's
    # own box (BOX_RZ) each way: a hit at the reach still needs its
    # 25-plane feature box and 11-plane template.
    from codelab_pipeline.localization.psfmatcher import BOX_RZ
    z0 = _slab_z0(20.0, z_half)
    spots = [spot(8, 8, 20.0 - z0, 0.9),                    # kept, at z_r
             spot(8, 12, 22.0 - z0, 0.8),                   # kept: a second locus
             spot(8, 4, 20.0 + z_half + 5 - z0, 0.95),      # beyond the reach in z
             spot(8, 16, 20.0 - z0, 0.88),                  # beyond the reach laterally
             spot(4, 8, 20.0 - z0, 0.3),                    # below the threshold
             spot(12, 8, 21.0 - z0, 0.7, p=float('nan'))]   # never refined
    eng = FakeEngine(spots)
    debug = {'H': {}}
    a, ok, why = _run(eng, z_r=20.0, min_p_exist=0.5, debug=debug)
    z1 = min(105, 20 + z_half + BOX_RZ + 1)
    check("the engine saw the fiducial's slab, padded by its own box",
          eng.seen == (17, 17, z1 - z0), f'{eng.seen} vs z0 {z0} z1 {z1}')
    check('and the slab is recorded for the record',
          debug['H']['readout_search'] == {'z0': z0, 'z1': z1, 'reach': z_half,
                                           'lateral_px': r_lat, 'trim': 0},
          str(debug['H'].get('readout_search')))
    check('it wrote', ok, why)
    adj = a.polymer_adj['H']
    check('TWO loci written as a list -- the multispot the panel never got',
          len(adj) == 2 and all(len(t) == 4 for t in adj))
    check('their z is back in the CROP\'s planes, not the slab\'s',
          sorted(round(t[2]) for t in adj) == [20, 22],
          str([t[2] for t in adj]))
    check('the far-in-z spot is NOT one of them, however confident',
          all(abs(t[2] - 20.0) <= z_half + 2.5 for t in adj))
    d = debug['H']
    check('every candidate reaches the grid with its p_exist',
          d['readout_engine'] == 'v3' and len(d['readout_p_exist']) == 6)
    check('kept ones carry labels', d['readout_labels'] == ['0.90', '0.80'])
    check('dropped ones carry theirs, tagged with why, and their reasons',
          len(d['readout_rejected_centroids']) == 4
          and d['readout_rejected_labels'] == ['0.95 z', '0.88 xy', '0.30 <',
                                                '0.70 ~']
          and any('planes from the fiducial' in w for w in d['readout_dropped_why'])
          and any('px from the fiducial' in w for w in d['readout_dropped_why'])
          and any('p_exist 0.30 < 0.5' in w for w in d['readout_dropped_why'])
          and any('sub-voxel' in w for w in d['readout_dropped_why']),
          str(d['readout_dropped_why']))
    check('and the count of out-of-reach ones is recorded',
          d['readout_n_out_of_reach'] == 2)

    # No threshold: p_exist is not cut at all (the reach still is).
    a2, ok2, _ = _run(FakeEngine(spots), z_r=20.0, min_p_exist=None)
    check('with no threshold, only reach and refinement cut',
          ok2 and len(a2.polymer_adj['H']) == 3)

    # Every candidate out of reach: the reason says so.
    far = [spot(8, 8, 2 * z_half + 20.0, 0.9), spot(8, 12, 2 * z_half + 25.0, 0.9)]
    a3, ok3, why3 = _run(FakeEngine(far), z_r=20.0)
    check('all out of reach -> rejected with the breakdown naming the reach',
          not ok3 and why3 == 'readout: none kept -- 2 z', why3)

    # THE PANEL'S Z WINDOW IS THE REACH. Six planes: a hit eight away is
    # dropped that would have been kept at the measured fourteen.
    z0w = _slab_z0(20.0, 6)
    debug = {'H': {}}
    a4, ok4, _ = _run(FakeEngine([spot(8, 8, 20.0 - z0w, 0.9),
                                  spot(8, 12, 28.0 - z0w, 0.9)]),
                      z_r=20.0, z_window=6, debug=debug)
    check('z_window from the panel sets the reach',
          ok4 and len(a4.polymer_adj['H']) == 1
          and any('> 6' in w for w in debug['H']['readout_dropped_why']),
          str(debug['H'].get('readout_dropped_why')))

    # THE TRIM SHAVES THE SLAB and drops what lands in the shaved planes.
    off = T2._z_trim_offset(105, 10)
    z0t = _slab_z0(12.0, z_half, trim=10)
    debug = {'H': {}}
    # Plane 8 is INSIDE a 10-plane trim (planes 0-9 are shaved); plane 11
    # is not -- the first draft of this check used 11 and proved nothing.
    a5, ok5, _ = _run(FakeEngine([spot(8, 8, 12.0 - z0t, 0.9),
                                  spot(8, 12, 8.0 - z0t, 0.9)]),
                      z_r=12.0, z_boundary_trim=10, debug=debug)
    check('the search slab starts at the trim, never before it',
          debug['H']['readout_search']['z0'] == off == 10
          and debug['H']['readout_search']['trim'] == 10)
    # slab-local 8 - z0t is -2 -> below the slab: the real engine cannot
    # return that, the fake does, and the trim gate is what catches it.
    check('a hit inside the shaved planes is dropped, and says so',
          ok5 and len(a5.polymer_adj['H']) == 1
          and any('of the stack end' in w for w in debug['H']['readout_dropped_why']),
          str(debug['H'].get('readout_dropped_why')))

    # THE DISPLAY OFFSET: hits come back in the display crop's frame.
    debug = {'H': {}}
    _run(FakeEngine([spot(8, 8, 20.0 - z0, 0.9)]), z_r=20.0,
         seed_yx=(8.0, 8.0), display_offset=(3, 4), debug=debug)
    cx, cy, _cz = debug['H']['readout_centroids'][0]
    check('debug centroids are shifted into the display crop',
          (cx, cy) == (4.0, 5.0), str((cx, cy)))

    # And nothing more to check on the earlier out-of-reach run.
    a3, ok3, why3 = _run(FakeEngine(far), z_r=20.0)
    check('all out of reach -> rejected with the breakdown naming the reach',
          not ok3 and why3 == 'readout: none kept -- 2 z', why3)


def test_rejected_labels_say_why():
    """A rejected candidate's label carries a tag for WHY it lost, and a
    readout that kept nothing says the breakdown, not the last reason."""
    print('rejected labels say why')
    M = T2.alt_marker
    check('one tag per kind',
          M('beyond the fiducial window (20.3 > 17 planes from expected)') == ' z'
          and M('z 17.6 planes from the fiducial > 14') == ' z'
          and M('7.0 px from the fiducial > 5') == ' xy'
          and M('z 3.0 within 10 planes of the stack end') == ' e'
          and M('no sub-voxel position') == ' ~'
          and M('p_exist 0.30 < 0.5') == ' <'
          and M('refined by the Gaussian fit') == ' g'
          and M('not the best') == '' and M(None) == '')
    import inspect
    src = inspect.getsource(T2)
    check('both label lists carry the tag',
          src.count('{alt_marker(w)}') == 2)
    z0 = _slab_z0(20.0, T2._seed_z_half(T2.READOUT_FIT_RADIUS_UM,
                                        T2.DEFAULT_VOXEL_UM))
    debug = {'H': {}}
    a, ok, why = _run(FakeEngine([spot(8, 15, 20.0 - z0, 0.99),   # 7 px out
                                  spot(8, 8, 20.0 - z0, 0.30)]),  # below p
                      debug=debug)
    check('nothing kept -> the breakdown by kind, short enough for a tile',
          not ok and why == 'readout: none kept -- 1 xy, 1 p<0.5', why)
    check('and the labels wear the tags',
          debug['H']['readout_rejected_labels'] == ['0.99 xy', '0.30 <'],
          str(debug['H']['readout_rejected_labels']))


def test_display_box_is_a_hard_boundary():
    """A candidate whose localized position lies outside the display crop
    is removed silently -- never counted, never listed, never drawn."""
    print('the display box is a hard boundary')
    z0 = _slab_z0(20.0, T2._seed_z_half(T2.READOUT_FIT_RADIUS_UM,
                                        T2.DEFAULT_VOXEL_UM))
    # search crop 27 wide (display 17 + BOX_R 7 each side): display offset
    # (7, 7). A hit at search x=2 is display x=-5: outside.
    spots = [spot(15, 15, 20.0 - z0, 0.9),      # display (8, 8): kept
             spot(15, 2, 20.0 - z0, 0.99),      # display x=-5: gone
             spot(26, 15, 20.0 - z0, 0.99)]     # display y=19: gone
    debug = {'H': {}}
    cube = np.random.RandomState(0).normal(300.0, 5.0, (31, 31, 105))
    a, ok, why = _run(FakeEngine(spots), z_r=20.0, cube=cube,
                      seed_yx=(15.0, 15.0), display_offset=(7, 7),
                      debug=debug)
    check('one written', ok and len(a.polymer_adj['H']) == 1, why)
    d = debug['H']
    check('the two outside the box were never counted or listed',
          d['readout_n_before_p_gate'] == 1 and d['readout_p_exist'] == [0.9]
          and not d.get('readout_rejected_centroids'),
          str((d['readout_n_before_p_gate'], d['readout_p_exist'])))
    # _run passes no display_shape by default; here it must.


def test_learned_fiducial_best_of_one():
    print('the learned fiducial: best of one, in the Gaussian\'s own window')
    p = T2.V2Params(min_p_exist=0.5, z_boundary_trim=0)
    window = T2.fiducial_window_planes(p.voxel_um)
    check("the window is the Gaussian fiducial's seed window (17 planes at "
          "0.2 um), not the readout reach (14)",
          window == 17 and window > p.z_reach(), f'{window} vs {p.z_reach()}')
    z0 = _slab_z0(20.0, window)
    cands = [spot(8, 8, 20.0 - z0, 0.7),                 # good
             spot(4, 8, 21.0 - z0, 0.95),                # BEST
             spot(12, 8, 20.0 - z0, 0.99, p=float('nan')),  # unrefined
             spot(8, 12, 20.0 + window + 3 - z0, 0.98),  # beyond the window
             spot(8, 4, 20.0 - z0, 0.3),                 # below threshold
             spot(8, 30, 20.0 - z0, 0.99)]               # outside the crop
    p.fiducial_engine = FakeEngine(cands)
    cube = np.random.RandomState(0).normal(300.0, 5.0, (17, 17, 105))
    f, why, alts, how = T2._fiducial_learned(cube, 20.0, p)
    check('the highest p_exist inside the window and above threshold wins',
          f is not None and (f.y, f.x) == (4.0, 8.0) and f.p_exist == 0.95
          and how == 'v3', str((f, how)))
    check('its z is back in the crop\'s planes', abs(f.z - 21.0) < 1e-9)
    check('ONE answer -- the fit has no list', isinstance(f, T2.LearnedFit))
    whys = [w for _a, w in alts]
    check('every other candidate is an alternative with a reason',
          len(alts) == 4 and 'not the best' in whys
          and any('no sub-voxel' in w for w in whys)
          and any('beyond the fiducial window' in w for w in whys)
          and any('p_exist 0.30 < 0.5' in w for w in whys), str(whys))
    check('the one outside the crop never existed', len(alts) + 1 == 5)
    check('the slab the engine saw spans the window plus its box',
          p.fiducial_engine.seen[2] >= 2 * window + 1)

    # 16 PLANES OFF IS INSIDE THE WINDOW. At the readout reach it was
    # refused, and on MP58/DNA that refusal cost two thirds of the
    # learned fiducial's losses.
    p.fiducial_engine = FakeEngine([spot(8, 8, 20.0 + 16 - z0, 0.9)])
    f16, _w, _a, how16 = T2._fiducial_learned(cube, 20.0, p)
    check('a candidate 16 planes from the expected depth is taken',
          f16 is not None and abs(f16.z - 36.0) < 1e-9 and how16 == 'v3')

    # NOTHING INSIDE THE WINDOW: the best in the slab, and phase 3's
    # drift gate against the reference decides, as it does for v2.
    p.fiducial_engine = FakeEngine([spot(8, 8, 20.0 + window + 3 - z0, 0.8),
                                    spot(6, 6, 20.0 + window + 5 - z0, 0.6)])
    fb, _w, ab, howb = T2._fiducial_learned(cube, 20.0, p)
    check('beyond the window, the best p_exist is still handed on -- and '
          'says so', fb is not None and fb.p_exist == 0.8
          and howb == 'v3 beyond window' and len(ab) == 1
          and 'not the best' in ab[0][1], str((fb, howb, ab)))

    # UNREFINED, BELIEVED: the Gaussian fit takes the seed. A bright
    # Gaussian blob is planted where the engine points.
    yy, xx, zz = np.mgrid[0:17, 0:17, 0:105].astype(float)
    blob = 3000.0 * np.exp(-0.5 * (((yy - 8) / 1.4) ** 2 + ((xx - 8) / 1.4) ** 2
                                   + ((zz - 22) / 3.0) ** 2))
    cube_b = cube + blob
    p.fiducial_engine = FakeEngine([spot(8, 8, 22.0 - z0, 0.9, p=float('nan'))])
    fg, wg, ag, howg = T2._fiducial_learned(cube_b, 20.0, p)
    check('an unrefined candidate is refined by the Gaussian fit, seeded '
          'where the engine pointed',
          fg is not None and howg == 'v3+gauss'
          and abs(fg.y - 8) < 0.5 and abs(fg.x - 8) < 0.5 and abs(fg.z - 22) < 1.0,
          str((None if fg is None else (fg.y, fg.x, fg.z), howg, wg)))
    check('and the engine\'s own candidate is listed as what was refined',
          ag and 'refined by the Gaussian' in ag[0][1], str(ag[:1]))
    # THE GATE IS FORCED, not hoped for: v2's fiducial gate is generous
    # enough to pass a Gaussian fitted to flat noise (it did here), so
    # the failure branch is exercised by making the gate say no.
    p.fiducial_engine = FakeEngine([spot(8, 8, 22.0 - z0, 0.9, p=float('nan'))])
    real_gate = T2.gate
    T2.gate = lambda *a, **k: (False, 'occupancy 0.10 < 0.25')
    try:
        fn, wn, an, hown = T2._fiducial_learned(cube_b, 20.0, p)
    finally:
        T2.gate = real_gate
    check('but a Gaussian fit that fails its gate does not rescue it -- and '
          'the reason is short enough for a tile title',
          fn is None and hown is None and wn.startswith('unrefined; refit')
          and 'occupancy 0.10 < 0.25' in wn and len(wn) < 45
          and an and 'Gaussian refinement failed' in an[0][1], wn)

    p.fiducial_engine = FakeEngine([spot(8, 8, 20.0 - z0, 0.2)])
    f2, why2, alts2, _h = T2._fiducial_learned(cube, 20.0, p)
    check('none above the threshold -> None with the reason',
          f2 is None and 'p_exist >= 0.5' in why2 and len(alts2) == 1, why2)
    p.fiducial_engine = FakeEngine([])
    f3, why3, _, _h = T2._fiducial_learned(cube, 20.0, p)
    check('nothing found -> None, says so', f3 is None and 'nothing' in why3)

    # V2Params carries the second model the same way as the first.
    pp = T2.V2Params.from_panel(
        {'engine': T2.ROUTE_V3, 'v2': {},
         'v3': {'model_dir': MODEL, 'fiducial_model_dir': MODEL,
                'min_p_exist': 0.5}}, None)
    check('from_panel reads the fiducial model', pp.fiducial_model_dir == MODEL)
    check('describe() says best-of-one', 'best-of-one' in pp.describe())
    pp.fiducial_engine = FakeEngine([])
    d = pickle.loads(pickle.dumps(pp))
    check('pickling drops the built fiducial engine, keeps the dir',
          d._fiducial_engine is None and d.fiducial_model_dir == MODEL)
    if os.path.isdir(MODEL):
        check('and it rebuilds lazily',
              type(d.fiducial_engine).__name__ == 'PsfMatcherV3Engine')
    q = T2.V2Params.from_panel({'engine': T2.ROUTE_V3, 'v2': {},
                                'v3': {'model_dir': MODEL}}, None)
    check('no fiducial model -> v2 Gaussian fiducial (engine None)',
          q.fiducial_model_dir is None and q.fiducial_engine is None)

    import inspect
    bsrc = inspect.getsource(T2.build_chromatin_trace_allele)
    check('the builder branches on the fiducial engine and records how',
          'if p.fiducial_engine is not None:' in bsrc
          and 'f, why, alts, how = _fiducial_learned(cube, z0, p)' in bsrc
          and "debug[hybe]['fiducial_how'] = how" in bsrc)
    check("and v2's seed fallback does not run for it",
          'FIDUCIAL_SEED_FALLBACK and p.fiducial_engine is None' in bsrc)


def test_panel():
    print('the panel')
    from PyQt5 import QtWidgets
    app()
    from ui.chromatin_tracing_panel import ChromatinTracingPanelUI, V3_DEFAULTS
    w = QtWidgets.QWidget()
    ui = ChromatinTracingPanelUI()
    ui.setupUi(w)
    check('a third page exists for v3',
          ui.FitParamsStackedWidget.count() == 3)
    for i in range(ui.EngineComboBox.count()):
        if ui.EngineComboBox.itemData(i) == T2.ROUTE_V3:
            ui.EngineComboBox.setCurrentIndex(i)
    ui.apply_engine_visibility()
    check('selecting v3 shows the v3 page, not v1\'s',
          ui.FitParamsStackedWidget.currentIndex() == 2)
    ui.populate_models([{'name': 'r1', 'path': '/m/r1', 'is_default': True,
                         'has_multispot': True, 'problems': []}],
                       select='/m/r1')
    ui.V3MinPExistSpinBox.setValue(0.65)
    pr = ui.params()
    check("params() carries the v3 block",
          pr['v3'] == {'model_dir': '/m/r1', 'fiducial_model_dir': None,
                       'min_p_exist': 0.65}
          and T2.route(pr['engine']) == T2.ROUTE_V3)
    ui.populate_models([{'name': 'r1', 'path': '/m/r1', 'is_default': True,
                         'has_multispot': True, 'problems': []}],
                       select='/m/r1', select_fiducial='/m/r1')
    check('the fiducial combo offers v2 Gaussian first, then the runs',
          ui.V3FiducialModelComboBox.itemData(0) is None
          and ui.V3FiducialModelComboBox.count() == 2)
    check('and params() carries the chosen fiducial model',
          ui.params()['v3']['fiducial_model_dir'] == '/m/r1')
    ui.V3FiducialModelComboBox.setCurrentIndex(0)
    check('back to v2 Gaussian -> None',
          ui.params()['v3']['fiducial_model_dir'] is None)
    check('the v3 page shows no Gaussian-fit gate',
          not any(k in dir(ui) for k in ('V3PeakBoundSpinBox',
                                          'V3MaxSigmaSpinBox')))
    # THE TWO Z CONTROLS ARE SHARED, not v1's: they must not live on the
    # v1 page (index 0), or a v3 selection hides them.
    v1page = ui.FitParamsStackedWidget.widget(0)
    check('Z search window and Z boundary trim are off the v1 page',
          not v1page.isAncestorOf(ui.ZWindowSpinBox)
          and not v1page.isAncestorOf(ui.ZBoundaryTrimSpinBox))
    check('and are read into params() for every engine',
          pr['z_window'] == ui.ZWindowSpinBox.value()
          and pr['z_boundary_trim'] == ui.ZBoundaryTrimSpinBox.value()
          and ui.ZWindowSpinBox.suffix() == ' planes')
    ui.reset_defaults()
    check('Reset restores the p_exist default',
          abs(ui.V3MinPExistSpinBox.value() - V3_DEFAULTS['min_p_exist'])
          < 1e-9)
    w.deleteLater()


def test_grid():
    print('the grid: every hit ringed and labelled')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from canvas import spot_fit_status as F
    cube = np.random.RandomState(0).normal(300.0, 5.0, (17, 17, 41))
    fig = plt.figure()
    a1 = fig.add_subplot(211)
    a2 = fig.add_subplot(212, sharex=a1)
    F.draw_spot_fit_status(a1, a2, cube,
                           centroid=[(8.0, 8.0, 20.0), (12.0, 8.0, 22.0)],
                           rejected=[(4.0, 8.0, 20.0)],
                           labels=['0.90', '0.80'], rejected_labels=['0.30'],
                           all_primary=True)
    cols = [tuple(np.round(c.get_edgecolor()[0], 2)) for c in a1.collections]
    check('with all_primary every kept ring is yellow',
          cols[:2] == [(1.0, 1.0, 0.0, 1.0)] * 2, str(cols))
    check('the rejected one is blue', cols[2] == (0.0, 0.75, 1.0, 1.0))
    texts = sorted(t.get_text() for t in a1.texts)
    check('each ring carries its p_exist', texts == ['0.30', '0.80', '0.90'],
          str(texts))
    plt.close(fig)

    from canvas.chromatin_trace_grid_displayer import ChromatinTraceGridDisplayer
    app()
    g = ChromatinTraceGridDisplayer('Readout')
    g.show_fit_status_grid([
        (cube, [(8.0, 8.0, 20.0)], 'H1\\np_exist 0.90', None,
         {'labels': ['0.90'], 'all_primary': True}),
        (cube, [(8.0, 8.0, 20.0)], 'H2', None)])
    check('the grid accepts a 5-tuple beside a 4-tuple',
          len(g.canvas.figure.axes) == 4)


def test_main_window_wiring():
    print('the app')
    import inspect
    import sys as _sys
    from windows.main_window import MainWindow as MW
    # The MODULE, not the class: one dispatch site is in the tracing
    # worker class, which MainWindow's own source does not contain.
    src = inspect.getsource(_sys.modules[MW.__module__])
    check('no dispatch site asks is_v2 for the tracer any more',
          'tracing_v2.is_v2(self.engine)' not in src
          and "tracing_v2.is_v2(full_params.get('engine'))" not in src
          and src.count('tracing_v2.uses_v2_tracer(') >= 3)
    cp = inspect.getsource(MW._chromatin_v2_params)
    check('v3 without a model is refused with the reason',
          'v3 needs a trained model' in cp)
    check('the readout tile for v3 lists p_exist and rings every kept hit',
          "d.get('readout_engine') == 'v3'" in src
          and "'all_primary': True" in src)
    check('and its title is cut to fit the tile',
          'def _short_reason(why):' in src and "[:4]" in src)
    check('the tracing panel gets the same model list as Spot Localization',
          'chp.populate_models(' in inspect.getsource(MW._refresh_model_list))
    check('the fiducial tile shows best-of-one with its p_exist',
          "d.get('fiducial_engine') == 'v3'" in src and 'best of' in src)
    check("a fiducial reason on a readout tile follows the readout title's "
          "own convention, so its threshold survives the 34-character cut",
          "('occupancy', 'occ')" in src and "('; refit ', ', ')" in src
          and "w = 'fid ' + w[len('fiducial '):]" in src)
    check('every tile, both engines, carries the depth window for the XZ '
          'panel', src.count("_z_extras(d, 'fiducial')") == 2
          and src.count("_z_extras(d, 'readout')") == 2
          and "out['z_display_pad'] = int(half) + 2" in src)
    bsrc2 = inspect.getsource(T2.build_chromatin_trace_allele)
    check('and the builder records the expected depth and the reach',
          "debug[hybe]['fiducial_zexp'] = float(z0)" in bsrc2
          and "debug[hybe]['fiducial_z_window']" in bsrc2
          and "debug[hybe]['readout_z_reach'] = int(p.z_reach())" in bsrc2)


def main():
    test_routing()
    test_params()
    test_readout_multi_gates()
    test_rejected_labels_say_why()
    test_display_box_is_a_hard_boundary()
    test_learned_fiducial_best_of_one()
    test_panel()
    test_grid()
    test_main_window_wiring()
    print()
    print('%d/%d checks passed' % (CHECKS[1], CHECKS[0]))
    return 0 if CHECKS[1] == CHECKS[0] else 1


if __name__ == '__main__':
    sys.exit(main())
