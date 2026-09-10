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
         seed_yx=(8.0, 8.0)):
    from codelab_pipeline.models.allele import AnAllele
    a = AnAllele()
    a.polymer_adj, a.polymer_raw = {}, {}
    p = T2.V2Params(min_p_exist=min_p_exist)
    p.readout_engine = engine
    if cube is None:
        cube = np.random.RandomState(0).normal(300.0, 5.0, (17, 17, 105))
    ok, why = T2._readout_multi(
        a, 'H', cube, z_r, p, 0.0, 0.0, 0.0, 100, 200,
        lambda h, y, x, z, ymin, xmin: (y + ymin, x + xmin, z), debug,
        seed_yx=seed_yx)
    return a, ok, why


def test_readout_multi_gates():
    print('_readout_multi: the fiducial\'s reach, then p_exist, then refinement')
    z_half = T2._seed_z_half(T2.READOUT_FIT_RADIUS_UM, T2.DEFAULT_VOXEL_UM)
    r_lat = T2._lateral_reach_px(T2.READOUT_FIT_RADIUS_UM, T2.DEFAULT_VOXEL_UM)
    check('the axial reach is v2\'s own readout search half-depth',
          z_half >= 1, f'{z_half} planes')
    check('and the lateral reach is the same radius in pixels',
          r_lat >= 1, f'{r_lat} px')
    # THE ENGINE SEES THE SLAB, so a fake that echoes its inputs must
    # speak slab-local z: z0 = z_r - z_half, and a hit at slab z_half is
    # at the fiducial's own plane.
    z0 = 20 - z_half
    spots = [spot(8, 8, float(z_half), 0.9),               # kept, at z_r
             spot(8, 12, float(z_half + 2), 0.8),          # kept: a second locus
             spot(8, 4, float(2 * z_half + 5), 0.95),      # beyond the reach in z
             spot(8, 16, float(z_half), 0.88),             # beyond the reach laterally
             spot(4, 8, float(z_half), 0.3),               # below the threshold
             spot(12, 8, float(z_half + 1), 0.7, p=float('nan'))]  # never refined
    eng = FakeEngine(spots)
    debug = {'H': {}}
    a, ok, why = _run(eng, z_r=20.0, min_p_exist=0.5, debug=debug)
    check("the engine saw the fiducial's SLAB, not the whole column",
          eng.seen == (17, 17, 2 * z_half + 1), str(eng.seen))
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
    check('dropped ones carry theirs and their reasons',
          len(d['readout_rejected_centroids']) == 4
          and d['readout_rejected_labels'] == ['0.95', '0.88', '0.30', '0.70']
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
    check('all out of reach -> rejected with the reach named',
          not ok3 and 'from the fiducial in z' in why3, why3)


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
          pr['v3'] == {'model_dir': '/m/r1', 'min_p_exist': 0.65}
          and T2.route(pr['engine']) == T2.ROUTE_V3)
    check('the v3 page shows no Gaussian-fit gate',
          not any(k in dir(ui) for k in ('V3PeakBoundSpinBox',
                                          'V3MaxSigmaSpinBox')))
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


def main():
    test_routing()
    test_params()
    test_readout_multi_gates()
    test_panel()
    test_grid()
    test_main_window_wiring()
    print()
    print('%d/%d checks passed' % (CHECKS[1], CHECKS[0]))
    return 0 if CHECKS[1] == CHECKS[0] else 1


if __name__ == '__main__':
    sys.exit(main())
