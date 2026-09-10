"""
Every _CONFIG_PARAM_MAP entry resolves on a window built with no config.

WHY. Save Config raised AttributeError on every press: the map's
3D-localization entry named a pop-up that lives on the WINDOW
(self.localize_3d_displayer), and both the capture and the apply side
did getattr(getattr(self.ui, panel), widget) inline. The round-trip
test that walks the map could not have caught it on this machine -- its
default config points at a Mac fixture path and it fails at load -- so
this guard needs no store and no config at all: a config-less
MainWindow, offscreen, and the map checked entry by entry through the
one resolver the app uses.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_config_map_resolves.py
"""
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKS = [0, 0]
_APP = None


def check(name, ok, note=''):
    CHECKS[0] += 1
    CHECKS[1] += 1 if ok else 0
    print(('  ok   ' if ok else '  FAIL ') + name
          + (('  -- ' + note) if note else ''))


def main():
    global _APP
    from PyQt5 import QtWidgets
    # HELD at module level: a QApplication dropped on the same line it is
    # made leaves the next widget with no live app and a silent
    # fail-fast, which an earlier test in this tree learned by tracing.
    _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from windows.main_window import MainWindow

    print('a config-less window')
    w = MainWindow(None)
    entries = [(s, p, panel, widget)
               for s, e in w._CONFIG_PARAM_MAP.items()
               for p, (panel, widget) in e.items()]
    check('the map is not empty', len(entries) > 50, str(len(entries)))
    bad = []
    for s, p, panel, widget in entries:
        try:
            w._config_widget(panel, widget)
        except Exception as exc:                            # noqa: BLE001
            bad.append((s, p, panel, widget, type(exc).__name__))
    check('EVERY entry resolves through the resolver the app uses',
          not bad, str(bad[:3]))
    check('the 3D-localization engine names the window attribute',
          w._CONFIG_PARAM_MAP['spot_localization']['engine'][0]
          == 'localize_3d_displayer')

    cap = w._capture_config_params()
    check('Save Config\'s capture runs', isinstance(cap, dict) and cap)
    check('and carries the 3D engine',
          bool(cap.get('spot_localization', {}).get('engine')))

    raised = False
    try:
        w._config_widget('NoSuchPanel', 'NoSuchWidget')
    except AttributeError:
        raised = True
    check('a panel on neither self.ui nor the window is a LOUD failure',
          raised)

    import inspect
    src = inspect.getsource(MainWindow._capture_config_params)
    src2 = inspect.getsource(MainWindow._apply_config_params)
    check('capture and apply both go through _config_widget',
          'self._config_widget(' in src and 'self._config_widget(' in src2
          and 'getattr(getattr(self.ui' not in src
          and 'getattr(getattr(self.ui' not in src2)
    print()
    print('%d/%d checks passed' % (CHECKS[1], CHECKS[0]))
    return 0 if CHECKS[1] == CHECKS[0] else 1


if __name__ == '__main__':
    sys.exit(main())
