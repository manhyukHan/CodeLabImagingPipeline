"""The Pipeline tab's per-step parameter dialog.

WHAT THIS PINS. The dialog does not own a parameter model: it reads
`MainWindow._CONFIG_PARAM_MAP` through `status.STEP_SECTIONS`, and it
hands its values back through `_apply_config_params` -- the same door a
loaded config comes through. Both halves are easy to "simplify" into
direct widget writes, and the moment that happens the dialog and a
config file apply the same value differently (the map's aliases, the
readout/fiducial translation and the two-pass retry all live behind that
door, not in the widgets).

The other thing pinned here is the TWO target sets. A chosen subset of
localization rounds narrows the RUN and deliberately does not narrow the
EXPECTATION, so a store missing rounds keeps reporting them missing. One
filtered call in the wrong place turns the status grid into a mirror of
what someone chose to run today.

Run: QT_QPA_PLATFORM=offscreen python tests/test_step_params_dialog.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt5 import QtWidgets                                  # noqa: E402

from codelab_pipeline.pipeline import status as S            # noqa: E402

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

TARGETS = {'DNA': [('Hyb_104', 635), ('Hyb_105', 635), ('Toe_113', 635)]}


class FakeWindow(QtWidgets.QMainWindow):
    """A window with the three things the dialog asks of MainWindow."""

    def __init__(self):
        super().__init__()
        from ui.main_window_ui import MainWindowUI
        from windows.main_window import MainWindow
        self._CONFIG_PARAM_MAP = MainWindow._CONFIG_PARAM_MAP
        self.ui = MainWindowUI()
        self.ui.setupUi(self)
        self.applied = None

    def _config_widget(self, panel, widget):
        owner = getattr(self.ui, panel, None)
        if owner is None:
            owner = getattr(self, panel)
        return getattr(owner, widget)

    def _apply_config_params(self, params):
        self.applied = params


def _dialog(win, step, targets=None):
    from ui import step_params_dialog as SPD
    return SPD.StepParamsDialog(win, step, targets=targets, parent=win)


def test_the_fields_come_from_the_config_map():
    """Not from a list kept in the dialog: every field it offers must be
    a parameter the map names for that step's section."""
    win = FakeWindow()
    dlg = _dialog(win, 'segmentation')
    offered = {(section, param) for section, param, _ed, _o in dlg._fields}
    assert offered, 'the segmentation dialog offered nothing'
    for section, param in offered:
        assert param in win._CONFIG_PARAM_MAP[section], (section, param)
    assert ('cell_segmentation', 'diameter') in offered, sorted(offered)


def test_an_edit_reaches_apply_in_the_config_spelling():
    """values() must be {section: {param: str}} -- the capture format --
    and apply() must route it through _apply_config_params."""
    win = FakeWindow()
    dlg = _dialog(win, 'segmentation')
    editor = next(ed for s, p, ed, _o in dlg._fields
                  if (s, p) == ('cell_segmentation', 'diameter'))
    editor.setValue(int(editor.value()) + 7)
    want = str(editor.value())

    changed = dlg.apply()
    assert win.applied is not None, 'apply() bypassed _apply_config_params'
    assert win.applied['cell_segmentation']['diameter'] == want, win.applied
    assert 'cell_segmentation/diameter' in changed, changed


def test_the_dialog_does_not_write_the_widget_itself():
    """The write goes through the one door. If the dialog ever sets the
    tab's widget directly, the map's aliases and retry pass are skipped
    and a config load and an OK stop agreeing."""
    win = FakeWindow()
    src = win._config_widget('CellSegmentPanel', 'DiameterSpinBox')
    before = src.value()
    dlg = _dialog(win, 'segmentation')
    editor = next(ed for s, p, ed, _o in dlg._fields
                  if (s, p) == ('cell_segmentation', 'diameter'))
    editor.setValue(int(before) + 7)
    dlg.apply()
    assert src.value() == before, (
        'the dialog moved the tab widget itself; only '
        '_apply_config_params may')


def test_restore_puts_the_opening_values_back():
    win = FakeWindow()
    dlg = _dialog(win, 'segmentation')
    editor = next(ed for s, p, ed, _o in dlg._fields
                  if (s, p) == ('cell_segmentation', 'diameter'))
    before = editor.value()
    editor.setValue(int(before) + 7)
    dlg._restore()
    assert editor.value() == before, (editor.value(), before)
    assert dlg.apply() == '', 'restore left the dialog reporting a change'


def test_all_rounds_is_stored_as_empty_not_as_every_name():
    """Empty keeps following the config: a tracing round added later is
    localized without anyone opening this dialog again."""
    from ui.step_params_dialog import RoundsPicker
    picker = RoundsPicker(TARGETS, '')
    assert len(picker.boxes) == 3
    assert picker.value() == '', picker.value()
    picker._clear_toeholds()
    assert picker.value() == 'DNA|Hyb_104|635,DNA|Hyb_105|635', picker.value()
    picker._set_all(False)
    assert picker.value() == ''  # nothing checked is not a subset either
    # and a stored subset comes back checked exactly as it was
    again = RoundsPicker(TARGETS, 'DNA|Toe_113|635')
    assert [cb.isChecked() for cb in again.boxes] == [False, False, True]


def test_a_chosen_subset_narrows_the_run_and_not_the_expectation():
    project = {'modalities': {'DNA': {'records': []}},
               'params': {'spot_localization': {'hybe': 'Hyb_104 (DNA)',
                                                'channel': '635'},
                          'chromatin_tracing': {'readout_channel': '635',
                                                'hybes': 'Hyb_104|DNA,'
                                                         'Hyb_105|DNA,'
                                                         'Toe_113|DNA'}}}
    full = S.localization_targets(project)
    assert sorted(full['DNA']) == [('Hyb_104', 635), ('Hyb_105', 635),
                                   ('Toe_113', 635)], full

    assert S.localization_run_targets(project) == full, 'empty must mean all'

    project['params']['localization_run'] = {
        'rounds': 'DNA|Hyb_104|635,DNA|Hyb_105|635'}
    run = S.localization_run_targets(project)
    assert sorted(run['DNA']) == [('Hyb_104', 635), ('Hyb_105', 635)], run
    assert sorted(S.localization_targets(project)['DNA']) == sorted(full['DNA']), (
        'the chosen subset narrowed the EXPECTATION too, so a store '
        'missing rounds would stop reporting them')


def test_every_step_opens(capsys=None):
    """A map entry naming a widget that does not exist must skip that one
    parameter, never take the dialog down with it."""
    win = FakeWindow()
    for step in S.STEPS:
        dlg = _dialog(win, step, targets=TARGETS if step == 'localization' else None)
        assert dlg.windowTitle(), step


def test_the_version_box_hides_the_other_versions():
    """A v2 run has no v3 parameters. Listing every version side by side
    is how a value gets set on an engine that is not the one running."""
    win = FakeWindow()
    dlg = _dialog(win, 'tracing')
    engine = next(ed for s, p, ed, _o in dlg._fields
                  if (s, p) == ('chromatin_tracing', 'engine'))
    titles = [engine.itemText(i) for i in range(engine.count())]
    boxes = {c.title(): c for c in dlg.findChildren(QtWidgets.QGroupBox)
             if c.title() in titles}
    assert len(boxes) == len(titles), (sorted(boxes), titles)

    for i, title in enumerate(titles):
        engine.setCurrentIndex(i)
        for name, box in boxes.items():
            want = (name == title)
            assert box.isVisibleTo(dlg) == want, (title, name,
                                                  box.isVisibleTo(dlg))


def test_the_engine_boxes_are_the_panel_s_own_pages():
    """Not prefix families. `readout_*` and `fiducial_*` are the v1
    page's fit columns, and grouping by name put them in boxes that
    stayed visible under v3, where they do nothing."""
    win = FakeWindow()
    dlg = _dialog(win, 'tracing')
    titles = {c.title() for c in dlg.findChildren(QtWidgets.QGroupBox)}
    assert 'readout' not in titles, sorted(titles)
    assert 'fiducial' not in titles, sorted(titles)


def test_a_measured_parameter_is_not_offered_as_a_setting():
    """The focal plane is detected per FOV before each segmentation, so
    the dialog must not offer it as a number to fix."""
    win = FakeWindow()
    dlg = _dialog(win, 'segmentation')
    offered = {param for _s, param, _ed, _o in dlg._fields}
    for param in ('z_plane', 'z_start', 'z_end'):
        assert param not in offered, param
    assert 'diameter' in offered, sorted(offered)


def test_the_localization_dialog_offers_what_the_batch_takes():
    """Not the Spot Localization tab's interactive parameters: the
    headless batch is v3 only and reads a model, a floor and a pad."""
    win = FakeWindow()
    dlg = _dialog(win, 'localization', targets=TARGETS)
    offered = {param for _s, param, _ed, _o in dlg._fields}
    assert offered == {'model', 'min_p_exist', 'pad'}, sorted(offered)
    assert dlg.rounds is not None, 'no rounds picker'
    assert 'rounds' not in offered, 'the picker edits rounds, not a text field'


def _run_all():
    tests = {k: v for k, v in globals().items() if k.startswith('test_')}
    failures = []
    for name in sorted(tests):
        try:
            tests[name]()
            print(f'  PASS  {name}')
        except AssertionError as e:
            failures.append(name); print(f'  FAIL  {name}: {e}')
        except Exception as e:                               # noqa: BLE001
            failures.append(name); print(f'  ERROR {name}: {type(e).__name__}: {e}')
    print(f'\n{len(tests) - len(failures)}/{len(tests)} passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(_run_all())
