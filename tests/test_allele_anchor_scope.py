"""
The anchor round is configuration, and nothing is allowed to invent it.

WHAT IT COST. The Alleles section picks which round and channel every
spot becomes an allele from. Neither combo was in
MainWindow._CONFIG_PARAM_MAP, so the choice was never saved and never
restored -- 0 of 20 checked-in configs carried it. The pipeline runner's
tracing step builds anchors for any FOV that has none, and it read the
choice off the live widgets, which headless held their own defaults: the
FIRST hybe in the list, and the first READOUT channel. On MP58 that is
Hyb_015 ch635 while the store's own anchors are Hyb_038 ch555, so a
re-run turned every ch635 spot of an unrelated round into an allele --
696 added across 4 FOVs, on top of anchors that were already there.

WHY THE OLD MITIGATION WAS NOT ENOUGH. It skipped FOVs that already had
alleles. That protects a populated store and does nothing for the case
it deliberately keeps: a FOV with none still gets them, which is exactly
the first run of a new experiment -- the thing the runner exists for.
And `Quiet` replaces QMessageBox.question with an automatic Yes while a
step runs, so the tab's own "turn EVERY spot in <hybe> ch<channel> into
an allele" confirmation cannot stop it either.

NEITHER CHANNEL IS A SAFE GUESS. The old default avoided the fiducial
channel, on the reasoning that index 0 is often the fiducial and picking
it by accident is the trap the combo exists to avoid. But MP58's real
anchors ARE on the fiducial channel. The safe-looking guess is wrong for
the store this project knows best, so there is no default now: an
unstated scope stays unstated, and the runner refuses rather than
guessing (the user's call, 2026-11-10).

WHAT IS PINNED HERE. The two map entries and which of text/itemData each
round-trips; that `_apply_widget_value` compares a STRINGIFIED itemData,
without which the int-valued channel could be captured but never
restored; that neither populate function selects anything it was not
given; that a stated scope still survives a repopulation; and that the
runner raises instead of building.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_allele_anchor_scope.py
"""
import glob
import os
import sys
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from PyQt5 import QtWidgets                                    # noqa: E402

PASS, FAIL = [], []
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: two rounds, fiducial-first, so index 0 IS the fiducial -- the shape
#: both old defaults were reasoning about. MP58's anchor scheme is the
#: fiducial one (Hyb_038 ch555), so neither end is privileged.
RECORDS = [({'folder': 'Hyb_015', 'channels': [555, 635], 'fiducial_channel': 555}, 'DNA'),
           ({'folder': 'Hyb_038', 'channels': [555, 635], 'fiducial_channel': 555}, 'DNA')]


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}'
          + (f'   [{detail}]' if detail else ''), flush=True)


#: HELD AT MODULE LEVEL ON PURPOSE. A QApplication whose only reference
#: was a discarded return value gets collected, and the next widget
#: built on it takes the process down in C++ with no Python traceback --
#: exit 127 and no output after the last print.
_APP = None
_ALIVE = []


def app():
    global _APP
    _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    return _APP


def panel():
    """A real ChromatinTracingPanel, so the populate functions under test
    are the ones the app runs."""
    import ui.chromatin_tracing_panel as P
    app()
    w = QtWidgets.QWidget()
    ui = P.ChromatinTracingPanelUI()
    ui.setupUi(w)
    _ALIVE.append((w, ui))
    return ui


def test_the_scope_is_configuration():
    print('\nthe anchor scope round-trips through the config')
    app()
    from windows.main_window import MainWindow as M

    entries = M._CONFIG_PARAM_MAP['chromatin_tracing']
    check('allele_hybe is in the map', 'allele_hybe' in entries,
          str(entries.get('allele_hybe')))
    check('allele_channel is in the map', 'allele_channel' in entries,
          str(entries.get('allele_channel')))
    check('both name the Alleles combos',
          entries.get('allele_hybe') == ('ChromatinTracingPanel', 'AlleleHybeComboBox')
          and entries.get('allele_channel') == ('ChromatinTracingPanel', 'AlleleChannelComboBox'))

    p = panel()
    p.populate_allele_hybe_choices(RECORDS)
    p.populate_allele_channel_choices(RECORDS[0][0])
    # The hybe saves TEXT -- the same shape reference_hybe already uses.
    check('the hybe combo round-trips its text, like reference_hybe',
          not M._combo_uses_data(p.AlleleHybeComboBox))
    # The channel's text is decorated ('555 (fiducial)'), and the tag
    # moves with the layout's fiducial assignment, so it must not be the
    # key. See MainWindow._CONFIG_DATA_PROPERTY.
    check('the channel combo round-trips its itemData, not its decorated text',
          M._combo_uses_data(p.AlleleChannelComboBox))
    check('and its text really is decorated (so text would be the wrong key)',
          '(' in p.AlleleChannelComboBox.itemText(0),
          p.AlleleChannelComboBox.itemText(0))

    # THE SYMMETRY. capture writes str(currentData()); apply has to
    # compare str(itemData) or an int-valued combo can never come back.
    p.AlleleChannelComboBox.setCurrentIndex(1)
    captured = M._widget_value(p.AlleleChannelComboBox)
    check('the channel captures as a bare number', captured == '635', captured)
    p.AlleleChannelComboBox.setCurrentIndex(0)
    landed = M._apply_widget_value(p.AlleleChannelComboBox, captured)
    check('and applying that string finds the int item again',
          landed and p.AlleleChannelComboBox.currentData() == 635,
          f'landed={landed} data={p.AlleleChannelComboBox.currentData()!r}')

    hp = panel()
    hp.populate_allele_hybe_choices(RECORDS)
    hp.AlleleHybeComboBox.setCurrentIndex(1)
    cap_h = M._widget_value(hp.AlleleHybeComboBox)
    check('the hybe captures as "Hyb_038 (DNA)"', cap_h == 'Hyb_038 (DNA)', cap_h)
    hp.AlleleHybeComboBox.setCurrentIndex(0)
    check('and applying it comes back to the same round',
          M._apply_widget_value(hp.AlleleHybeComboBox, cap_h)
          and hp.current_allele_hybe_key() == ('Hyb_038', 'DNA'),
          str(hp.current_allele_hybe_key()))


def test_nothing_is_invented():
    print('\npopulating offers choices; it does not make one')
    app()
    p = panel()

    p.populate_allele_hybe_choices(RECORDS)
    check('the hybe list is filled', p.AlleleHybeComboBox.count() == 2,
          str(p.AlleleHybeComboBox.count()))
    check('but NOTHING is selected -- not index 0, the old default',
          p.AlleleHybeComboBox.currentIndex() == -1,
          f'index {p.AlleleHybeComboBox.currentIndex()}, '
          f'{p.AlleleHybeComboBox.currentText()!r}')
    check('so the key reads as unstated', p.current_allele_hybe_key() == (None, None),
          str(p.current_allele_hybe_key()))

    p.populate_allele_channel_choices(RECORDS[0][0])
    check('the channel list is filled', p.AlleleChannelComboBox.count() == 2)
    check('and nothing is selected there either -- not the first readout',
          p.AlleleChannelComboBox.currentIndex() == -1,
          f'index {p.AlleleChannelComboBox.currentIndex()}, '
          f'{p.AlleleChannelComboBox.currentText()!r}')
    check('so the channel reads as unstated', p.current_allele_channel() is None,
          repr(p.current_allele_channel()))

    # A STATED SCOPE MUST STILL SURVIVE. The restore path is the whole
    # reason these functions take `current` into account; removing the
    # fallback must not remove that.
    p.AlleleHybeComboBox.setCurrentIndex(1)
    p.populate_allele_channel_choices(RECORDS[0][0])
    p.AlleleChannelComboBox.setCurrentIndex(0)          # the FIDUCIAL, as MP58 uses
    before = (p.current_allele_hybe_key(), p.current_allele_channel())
    p.populate_allele_hybe_choices(RECORDS)
    p.populate_allele_channel_choices(RECORDS[0][0])
    check('a stated hybe survives a repopulation',
          p.current_allele_hybe_key() == before[0], str(p.current_allele_hybe_key()))
    check('and so does a stated FIDUCIAL channel (MP58 anchors on ch555)',
          p.current_allele_channel() == before[1] == 555,
          repr(p.current_allele_channel()))


def test_the_runner_refuses_an_unstated_scope():
    print('\nthe runner refuses to build anchors it was not given')
    app()
    from windows import pipeline_wiring as W

    def runner_for(p):
        calls = []
        mw = SimpleNamespace(
            ui=SimpleNamespace(ChromatinTracingPanel=p),
            log=lambda *a, **k: None,
            _chromatin_storage_path_and_modality=lambda: ('DNA', ''),
            _build_chromatin_alleles_all_fovs=lambda: calls.append('build'),
            _run_chromatin_tracing_fit_all=lambda: calls.append('trace'),
            _chromatin_worker=None)
        r = W.PipelineRunner(mw)
        r.project = None                 # skip the tracing-configured gate
        return r, calls

    # -- unstated: refuse --
    p = panel()
    p.populate_allele_hybe_choices(RECORDS)
    p.populate_allele_channel_choices(RECORDS[0][0])
    r, calls = runner_for(p)
    try:
        r._run_tracing([1, 2, 3], 'append')
        raised = None
    except ValueError as exc:
        raised = str(exc)
    check('it raises instead of building', raised is not None,
          '' if raised else 'it returned normally -- anchors would have been built')
    check('and built nothing', 'build' not in calls,
          '' if 'build' not in calls else f'it built: {calls}')
    if raised:
        check('the message says what to do about it',
              'Alleles section' in raised and 'Save Config' in raised,
              raised[:90])
        check('and names how many FOVs it is refusing for', '3 FOV(s)' in raised,
              raised[:90])

    # -- stated: proceed --
    p2 = panel()
    p2.populate_allele_hybe_choices(RECORDS)
    p2.AlleleHybeComboBox.setCurrentIndex(1)                    # Hyb_038
    p2.populate_allele_channel_choices(RECORDS[1][0])
    p2.AlleleChannelComboBox.setCurrentIndex(0)                 # ch555
    r2, calls2 = runner_for(p2)
    try:
        r2._run_tracing([1, 2, 3], 'append')
        err = None
    except ValueError as exc:
        err = str(exc)
    check('a STATED scope is built and traced as before',
          err is None and calls2 == ['build', 'trace'], f'{err or ""} {calls2}')


def test_the_configs_premise():
    print('\nthe configs this refusal is live for')
    files = sorted(glob.glob(os.path.join(HERE, 'configs', '*.xml')))
    check('there are configs to check', bool(files), f'{len(files)} file(s)')
    both, one, neither = [], [], []
    for f in files:
        with open(f, encoding='utf-8', errors='replace') as fh:
            s = fh.read()
        h, c = 'allele_hybe=' in s, 'allele_channel=' in s
        (both if h and c else one if (h or c) else neither).append(os.path.basename(f))
    print(f'    carry both: {len(both)}   exactly one: {len(one)}   neither: {len(neither)}')
    # NEVER ONE WITHOUT THE OTHER. A hybe with no channel, or the
    # reverse, is half a scope -- and _apply_config_params only touches
    # keys a file carries, so the missing half would keep whatever the
    # widget held. That is the defect, re-entered through the config.
    check('no config carries half a scope', not one, ', '.join(one))
    # Informational: every config predating this change carries neither,
    # which is exactly why the runner now refuses rather than guesses.
    check('the ones with neither are the refusal path, as expected',
          len(neither) + len(both) == len(files))


def main():
    for t in (test_the_scope_is_configuration,
              test_nothing_is_invented,
              test_the_runner_refuses_an_unstated_scope,
              test_the_configs_premise):
        t()
    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    for f in FAIL:
        print('  FAILED:', f)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
