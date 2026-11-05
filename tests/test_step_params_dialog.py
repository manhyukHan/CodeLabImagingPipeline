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
ROUNDS = {'DNA': ['Hyb_104', 'Hyb_105', 'Toe_113']}


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
    picker = RoundsPicker(ROUNDS, '')
    assert len(picker.boxes) == 3
    assert picker.value() == '', picker.value()
    picker._clear_toeholds()
    assert picker.value() == 'DNA|Hyb_104,DNA|Hyb_105', picker.value()
    picker._set_all(False)
    # nothing checked is NOT empty: empty means "follow the config", so
    # it used to come back as every expected round -- None meaning All
    assert picker.value() == S.NONE_CHOSEN, picker.value()
    # and a stored subset comes back checked exactly as it was
    again = RoundsPicker(ROUNDS, 'DNA|Toe_113')
    assert [cb.isChecked() for cb in again.boxes] == [False, False, True]
    # an older config spelling the channel out still names its rounds
    old = RoundsPicker(ROUNDS, 'DNA|Toe_113|635')
    assert [cb.isChecked() for cb in old.boxes] == [False, False, True]


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
        'rounds': 'DNA|Hyb_104,DNA|Hyb_105'}
    run = S.localization_run_targets(project)
    assert sorted(run['DNA']) == [('Hyb_104', 635), ('Hyb_105', 635)], run
    assert sorted(S.localization_targets(project)['DNA']) == sorted(full['DNA']), (
        'the chosen subset narrowed the EXPECTATION too, so a store '
        'missing rounds would stop reporting them')


def test_the_spot_localization_channel_is_a_choice_not_a_derivation():
    """The step had no channel of its own: which channels it looked in
    fell out of what each contributor named, and for the rounds naming
    none, out of which channel was somebody else's fiducial. One
    selector decides it now, and it FILTERS -- the rounds are untouched.
    """
    recs = [{'folder': 'Hyb_001', 'channels': [555, 635, 747],
             'fiducial_channel': 555, 'readout_name': 'MCM2_mRNA'},
            {'folder': 'Hyb_002', 'channels': [555, 635, 747],
             'fiducial_channel': 555, 'readout_name': 'CCNB1_exon'}]
    project = {'modalities': {'RNA': {'records': recs}}, 'params': {}}
    # unset: the list this step has always had, so no status moves
    every = sorted(S.localization_targets(project)['RNA'])
    assert every == [('Hyb_001', 635), ('Hyb_001', 747),
                     ('Hyb_002', 635), ('Hyb_002', 747)], every
    assert S.default_localization_channels(project) == [635, 747], (
        S.default_localization_channels(project))
    assert S.localization_channels(project) == {'RNA': [555, 635, 747]}, (
        'the selector must offer every channel, the fiducial included')

    project['params']['localization_run'] = {'channels': '635'}
    one = sorted(S.localization_targets(project)['RNA'])
    assert one == [('Hyb_001', 635), ('Hyb_002', 635)], one
    project['params']['localization_run'] = {'channels': '635,747'}
    assert sorted(S.localization_targets(project)['RNA']) == every

    # the rounds are untouched by it: the same two, at fewer channels
    project['params']['localization_run'] = {'channels': '635'}
    assert S.localization_rounds(project) == {'RNA': ['Hyb_001', 'Hyb_002']}, (
        S.localization_rounds(project))


def test_the_channel_selector_offers_every_channel_and_starts_on_the_old_set():
    """A selector, not a rule. The fiducial channel is offered because
    spot localization finds spots in an image, and which channel an
    alignment uses is a different question -- on MAZ the 555 fiducial is
    the beads whose spots we wanted."""
    from ui.step_params_dialog import ChannelPicker
    picker = ChannelPicker([555, 635, 747], [635, 747], [])
    assert [cb.key for cb in picker.boxes] == [555, 635, 747]
    assert picker.value_list() == [635, 747], 'it must start on the default'
    assert picker.value() == '', 'the default set keeps following the config'

    next(cb for cb in picker.boxes if cb.key == 555).setChecked(True)
    assert picker.value() == '555,635,747', picker.value()

    picker = ChannelPicker([555, 635, 747], [635, 747], [635])
    assert picker.value_list() == [635]
    assert picker.value() == '635', picker.value()

    for cb in picker.boxes:
        cb.setChecked(False)
    assert picker.value_list() == []
    assert 'not localized' in picker.Note.text(), picker.Note.text()


def test_none_is_none_and_not_all():
    """Empty means FOLLOW THE CONFIG, so an empty selection came back as
    every expected round: pressing None ran everything."""
    from ui.step_params_dialog import RoundsPicker
    picker = RoundsPicker(ROUNDS, '')
    picker._set_all(False)
    assert picker.value() == S.NONE_CHOSEN, picker.value()
    project = {'modalities': {'DNA': {'records': []}},
               'params': {'chromatin_tracing': {'hybes': 'Hyb_104|DNA',
                                                'readout_channel': '635'},
                          'localization_run': {'rounds': S.NONE_CHOSEN}}}
    assert S.localization_run_targets(project) == {'DNA': []}, (
        S.localization_run_targets(project))
    assert S.localization_targets(project)['DNA'] == [('Hyb_104', 635)], (
        'naming no round changed the expectation')
    picker._set_all(True)
    assert picker.value() == '', picker.value()


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
    headless batch is v3 only and reads a model, a floor and a pad --
    plus two pickers: the rounds, and the CHANNELS the step looks in --
    which the batch takes per group and nothing could state."""
    win = FakeWindow()
    dlg = _dialog(win, 'localization', targets=TARGETS)
    offered = {param for _s, param, _ed, _o in dlg._fields}
    assert offered == {'model', 'min_p_exist', 'pad'}, sorted(offered)
    assert dlg.rounds is not None, 'no rounds picker'
    assert 'rounds' not in offered, 'the picker edits rounds, not a text field'
    assert 'channels' not in offered, (
        'the channel picker edits channels, not a text field')


def test_the_celltype_dialog_shows_one_mode_at_a_time():
    """FOV mode and barcode mode are the panel's own stacked pages. Which
    one runs decides what every other parameter in the section means, so
    listing both reads as if a run used both."""
    win = FakeWindow()
    ct = win.ui.CelltypeDeterminationPanel
    ct.BarcodeModeRadioButton.setChecked(True)
    dlg = _dialog(win, 'celltype')
    boxes = {c.title(): c for c in dlg.findChildren(QtWidgets.QGroupBox)
             if c.title() in ('FOV ranges', 'Barcode channel')}
    assert 'Barcode channel' in boxes, sorted(boxes)
    assert boxes['Barcode channel'].isVisibleTo(dlg)

    mode = next(ed for s_, p_, ed, _o in dlg._fields
                if (s_, p_) == ('celltype', 'fov_mode'))
    mode.setChecked(True)                      # FOV
    assert not boxes['Barcode channel'].isVisibleTo(dlg)


def test_the_mode_is_a_config_parameter_at_last():
    """A saved experiment could not say which mode produced its
    celltypes. The radio goes through the same capture/apply pair as
    every other widget."""
    from windows.main_window import MainWindow
    win = FakeWindow()
    radio = win._config_widget('CelltypeDeterminationPanel',
                               'FOVModeRadioButton')
    win.ui.CelltypeDeterminationPanel.BarcodeModeRadioButton.setChecked(True)
    assert MainWindow._widget_value(radio) == '0'
    MainWindow._apply_widget_value(radio, '1')
    assert radio.isChecked() and MainWindow._widget_value(radio) == '1'
    assert not win.ui.CelltypeDeterminationPanel.BarcodeModeRadioButton.isChecked(), (
        'the button group did not answer for the other half')


def test_the_celltype_names_can_be_removed_not_only_added():
    """ensure_celltype_names only ever adds, on purpose. A name taken out
    deliberately has to stay out -- and taking it out must not touch a
    cell that already carries it."""
    win = FakeWindow()
    ct = win.ui.CelltypeDeterminationPanel
    ct.ensure_celltype_names(['A', 'B', 'C'])
    assert ct.celltype_names() == ['A', 'B', 'C']
    ct.ensure_celltype_names(['B'])
    assert ct.celltype_names() == ['A', 'B', 'C'], 'ensure_ removed one'
    assert ct.set_celltype_names(['C', 'A']) == ['C', 'A']
    assert ct.celltype_names() == ['C', 'A'], ct.celltype_names()


def test_a_new_celltype_starts_as_a_training_target():
    """None checked means 'every cell', which on an experiment whose
    celltypes ARE defined is the weaker choice. A name appearing for the
    first time arrives checked; unchecking it is the decision."""
    win = FakeWindow()
    cc = win.ui.CellCyclePanel
    cc.set_celltype_names(['GM2-5', 'C2-6'])
    assert cc.training_celltypes() == ['GM2-5', 'C2-6'], cc.training_celltypes()
    cc.set_training_celltypes(['GM2-5'])
    cc.set_celltype_names(['GM2-5', 'C2-6', 'HU_2mM'])
    assert cc.training_celltypes() == ['GM2-5', 'HU_2mM'], (
        'a name already decided on was re-checked, or the new one was not')


def test_an_arrested_looking_name_is_named_but_never_dropped():
    from codelab_pipeline.analysis import cellcycle as CC
    assert CC.arrested_looking(['GM2-5', 'HU_2mM', 'Noco 3h', 'C2-6']) == [
        'HU_2mM', 'Noco 3h']
    assert CC.arrested_looking(['GM2-5', 'C2-6']) == []
    assert CC.arrested_looking([]) == []
    # a vehicle control is the CYCLING arm of an arrest experiment,
    # so warning about it would point at the one population the fit
    # wants
    assert CC.arrested_looking(['DMSO', 'DMSO 0.1%']) == []


def test_an_auto_choice_greys_out_the_number_it_chooses():
    # an editable 100 beside a ticked 'auto' is a lie about what runs
    win = FakeWindow()
    dlg = _dialog(win, 'cellcycle')
    by = {p_: ed for _s, p_, ed, _o in dlg._fields}
    assert 'alpha' in by and 'alpha_auto' in by, sorted(by)
    by['alpha_auto'].setChecked(True)
    assert not by['alpha'].isEnabled()
    by['alpha_auto'].setChecked(False)
    assert by['alpha'].isEnabled()


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
