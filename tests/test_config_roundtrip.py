"""
Save Config captures the running app; loading it reproduces the app exactly.

The verification is the feature's own definition (per explicit request): tweak
EVERY parameter _CONFIG_PARAM_MAP names -- every combo to a different real
item, every spinbox to a different in-range value, every checkbox toggled,
plus the two dynamic ones (per-modality cell-alignment reference hybes and
chromatin tracing's checked-hybe set) -- press Save Config, open the saved
file in a FRESH MainWindow, and require _capture_config_params of both apps
to compare EQUAL. Any parameter that saves but doesn't restore (or restores
into the wrong widget) fails the dict comparison by name.

Also pins the file's human-readable contract: per-stage section tags with
biological parameter names, no Qt widget names anywhere in the XML.

Run: CODELAB_RT_CONFIG=<config> python tests/test_config_roundtrip.py
(default: configs/chr19_downstream_debug.xml against the real local store)
"""
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest import mock

from PyQt5 import QtWidgets

CONFIG = os.environ.get('CODELAB_RT_CONFIG', 'configs/chr19_downstream_debug.xml')
OUT = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'config_roundtrip_saved.xml')


def _tweak(w):
    """Move a widget off its current value, staying within what the widget
    itself allows -- so the value is guaranteed to be reproducible in a
    fresh app fed the same store."""
    if isinstance(w, QtWidgets.QComboBox):
        if w.count() > 1:
            w.setCurrentIndex((w.currentIndex() + 1) % w.count())
    elif isinstance(w, QtWidgets.QCheckBox):
        w.setChecked(not w.isChecked())
    elif isinstance(w, QtWidgets.QLineEdit):
        w.setText('7' if w.text() != '7' else '8')
    else:  # QSpinBox / QDoubleSpinBox
        step = w.singleStep() or 1
        w.setValue(w.value() - step if w.value() + step > w.maximum() else w.value() + step)


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from windows.main_window import MainWindow

    dialogs = []

    def capture_dialog(parent, title, text, *a, **k):
        dialogs.append((title, text))
        return QtWidgets.QMessageBox.Ok

    with mock.patch.object(QtWidgets.QMessageBox, 'information', side_effect=capture_dialog), \
            mock.patch.object(QtWidgets.QMessageBox, 'warning', side_effect=capture_dialog), \
            mock.patch.object(QtWidgets.QMessageBox, 'critical', side_effect=capture_dialog), \
            mock.patch.object(QtWidgets.QMessageBox, 'question',
                              return_value=QtWidgets.QMessageBox.Yes):
        a = MainWindow(CONFIG)
        assert not dialogs, f'dialogs during initial load: {dialogs}'

        # tweak every mapped scalar parameter, in map order (hybe combos
        # come before their dependent channel combos there, so a channel
        # tweak always picks from the repopulated list)
        n_tweaked = 0
        for section, entries in a._CONFIG_PARAM_MAP.items():
            for param, (panel, widget) in entries.items():
                _tweak(getattr(getattr(a.ui, panel), widget))
                n_tweaked += 1
        # the two dynamic parameters
        chp = a.ui.ChromatinTracingPanel
        keys = [chp.HybeListWidget.item(i).data(0x0100)  # QtCore.Qt.UserRole
                for i in range(chp.HybeListWidget.count())]
        chp.set_checked_hybes(keys[:2])
        ap = a.ui.AlignmentPanel
        for name in a.ui.IngestionPanel.modality_names:
            combo = ap.CellReferenceHybeComboBoxes.get(name)
            if combo is not None and combo.count() > 1:
                combo.setCurrentIndex((combo.currentIndex() + 1) % combo.count())

        with mock.patch.object(QtWidgets.QFileDialog, 'getSaveFileName',
                               return_value=(OUT, '')):
            a._save_config_dialog()
        assert not dialogs, f'dialogs during save: {dialogs}'
        state_a = a._capture_config_params()

        xml = open(OUT).read()
        for fragment in ('SpinBox', 'ComboBox', 'CheckBox', 'LineEdit', 'Panel'):
            assert fragment not in xml, f'widget name leaked into the config: {fragment}'
        for section in a._CONFIG_PARAM_MAP:
            assert f'<{section}' in xml, f'section missing from saved file: {section}'

        b = MainWindow(OUT)
        assert not dialogs, f'dialogs during reload: {dialogs}'
        state_b = b._capture_config_params()

    diffs = []
    for section in state_a:
        for param in state_a[section]:
            va, vb = state_a[section][param], state_b.get(section, {}).get(param)
            if va != vb:
                diffs.append(f'{section}.{param}: saved {va!r} -> reloaded {vb!r}')
    if diffs:
        print('\n'.join(diffs))
        print(f'{len(diffs)} parameter(s) failed the roundtrip')
        return 1
    n_params = sum(len(v) for v in state_a.values())
    print(f'{n_tweaked} widgets tweaked; {n_params} parameters captured; '
          f'fresh app reproduces every one. ALL GOOD')
    return 0


if __name__ == '__main__':
    sys.exit(main())
