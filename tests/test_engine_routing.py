"""
One vocabulary for the thing a person picks, and one rule for reading it.

TWO SELECTORS OFFERED THE SAME IDEA IN INCOMPATIBLE WORDS. The tracing
panel stored 'v1' / 'v2'; the 3D-localization popup stored 'gaussian' /
'v2' -- while its own combo read addItem('v1 gaussian', 'gaussian'), so
the LABEL already said v1 and only the stored value disagreed. A value
copied from one panel to the other did not route, and 'gaussian' collided
by name with a real make_engine key meaning the bare Gaussian ENGINE
rather than that route.

AND THREE COPIES OF THE READING RULE, one of which had already drifted:
tracing_v2.is_v2 matched a PREFIX, ChromatinTracingPanel re-implemented
the prefix inline, and localization.refine_spots_batch matched on
EQUALITY. A decorated label routed correctly in tracing and fell back to
v1 in the popup, and the only symptom would have been slightly worse
numbers. Adding a third route to that arrangement is how a v3 selection
silently runs as v1.

DATA ON DISK OUTLIVES A RENAME. Five checked-in configs carry
engine="v2"; every old spelling still lands on the route it always meant.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_engine_routing.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from PyQt5 import QtWidgets                                  # noqa: E402

from codelab_pipeline.localization import engine as E        # noqa: E402
from codelab_pipeline.localization import tracing_v2 as R    # noqa: E402

PASS, FAIL = [], []
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}'
          + (f'   [{detail}]' if detail else ''))


def src(*parts):
    with open(os.path.join(HERE, *parts), encoding='utf-8') as f:
        return f.read()


def test_every_old_spelling_still_lands():
    print('\nroute(): a rename must not move what a stored config means')
    check('the three routes are named', R.ROUTES ==
          (R.ROUTE_V1, R.ROUTE_V2, R.ROUTE_V3), str(R.ROUTES))
    # The five checked-in configs say engine="v2".
    check("engine=\"v2\" -- what 5 configs carry -- is the anchor-fit route",
          R.route('v2') == R.ROUTE_V2)
    # The popup stored this for the v1 route while labelling it 'v1 gaussian'.
    check("'gaussian' was never a third engine: it is v1",
          R.route('gaussian') == R.ROUTE_V1)
    check("and so is 'v1'", R.route('v1') == R.ROUTE_V1)
    check('the new names round-trip',
          R.route(R.ROUTE_V2) == R.ROUTE_V2
          and R.route(R.ROUTE_V3) == R.ROUTE_V3)
    # PREFIX, not equality: a label may gain explanation.
    check('a decorated label still routes',
          R.route('v2 (calibrated PSF)') == R.ROUTE_V2
          and R.route('V3-PSFMATCHER  ') == R.ROUTE_V3)
    check('an unknown value falls to v1 rather than raising',
          R.route('banana') == R.ROUTE_V1 and R.route(None) == R.ROUTE_V1)
    check('is_v2 / is_v3 agree with route',
          R.is_v2('v2') and not R.is_v2('gaussian')
          and R.is_v3('v3-psfmatcher') and not R.is_v3('v2'))


def test_both_panels_speak_it():
    print('\nthe two selectors: one vocabulary, stored as itemData')
    app = (QtWidgets.QApplication.instance()
           or QtWidgets.QApplication(sys.argv[:1]))
    import ui.chromatin_tracing_panel as P
    import canvas.localize_3d_displayer as L

    w = QtWidgets.QWidget()
    ui = P.ChromatinTracingPanelUI()
    ui.setupUi(w)
    cb = ui.EngineComboBox
    data = [cb.itemData(i) for i in range(cb.count())]
    check('the tracing panel offers all three routes',
          set(data) == set(R.ROUTES), str(data))
    check('and stores routes, never display text',
          all(d in R.ROUTES for d in data))
    check('v3 is reachable from the panel at all',
          R.ROUTE_V3 in data)
    check('its default is a real route',
          R.route(P.DEFAULT_PARAMS['engine']) == P.DEFAULT_PARAMS['engine'],
          P.DEFAULT_PARAMS['engine'])

    ps = src('canvas', 'localize_3d_displayer.py')
    check("the popup no longer stores 'gaussian'",
          "'gaussian')" not in ps, 'still there' if "'gaussian')" in ps else '')
    check('the popup default is the same word the panel uses',
          L.DEFAULT_PARAMS['engine'] == P.DEFAULT_PARAMS['engine'],
          f"{L.DEFAULT_PARAMS['engine']} vs {P.DEFAULT_PARAMS['engine']}")
    check('and both are the anchor-fit route',
          R.route(L.DEFAULT_PARAMS['engine']) == R.ROUTE_V2)


def test_one_reading_rule_not_three():
    print('\nthe consumers: one matcher, and the drifted copy is gone')
    ls = src('codelab_pipeline', 'localization', 'localization.py')
    check("refine_spots_batch no longer matches on equality",
          "if engine == 'v2':" not in ls)
    check('it calls the shared matcher instead',
          'is_v2(engine)' in ls)
    ps = src('ui', 'chromatin_tracing_panel.py')
    check('the panel no longer re-implements the prefix rule',
          ".startswith('v2')" not in ps)
    check('it calls the shared matcher too', 'R.is_v2(' in ps)
    # THE ROUTES NOW NAME ENGINES. Before this, no config key, widget or
    # flag ever supplied make_engine's `name` -- every in-app callsite
    # passed the literal 'gaussian'.
    for r in R.ROUTES:
        check(f'make_engine accepts the route {r!r}',
              type(E.make_engine(r)).__name__.endswith('Engine'),
              type(E.make_engine(r)).__name__)
    check('and the original engine names still resolve',
          all(n in E.ENGINES for n in ('gaussian', 'anchor-v1', 'anchor-v2',
                                       'psf-match')))


def test_the_popup_choice_survives_a_restart():
    print('\nconfig: the 3D-localization engine was never written down')
    ms = src('windows', 'main_window.py')
    i = ms.index("'spot_localization': {")
    block = ms[i:i + 900]
    check("spot_localization now carries an 'engine' key",
          "'engine': (" in block, block.split('\n')[0])
    check('mapped to the popup that owns it',
          'Localize3DDisplayer' in block)


def main():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    for t in (test_every_old_spelling_still_lands,
              test_both_panels_speak_it,
              test_one_reading_rule_not_three,
              test_the_popup_choice_survives_a_restart):
        t()
    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    for f in FAIL:
        print('  FAILED:', f)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
