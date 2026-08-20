"""
Call every no-argument MainWindow method and report which ones raise.

Exists because grep-based refactors kept leaving crashes that only appeared
when the user launched the app: a renamed key, a stripped type, a variable
whose definition moved. Import succeeding proves nothing -- these are all
runtime failures on paths a smoke test never walked.

Not a correctness test. It asserts only that a method does not blow up, which
is exactly the class of bug that was reaching the user.
"""
import os
import sys
import traceback

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5 import QtWidgets
from unittest import mock

CONFIG = os.environ.get('CODELAB_SMOKE_CONFIG', 'configs/chr19_downstream_debug.xml')
FOV = 1

# Methods that legitimately need a user, a running worker, or would mutate
# real data. Skipped by name so the list of exclusions is explicit and small.
SKIP = {
    'close', 'show', 'hide', 'raise_', 'update', 'repaint', 'deleteLater',
    'setFocus', 'showFullScreen', 'showMaximized', 'showMinimized',
    'showNormal', 'activateWindow', 'grabMouse', 'grabKeyboard', 'destroy',
}
SKIP_PREFIX = ('_save', '_run_', '_write', '_delete', '_remove', '_accept',
               '_reject', '_start', '_stop', '_export', '_import', '_load_config')


def build():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from windows.main_window import MainWindow
    # EXACTLY main.py's launch form: the constructor itself loads the
    # config. A NameError inside config load once blocked a real launch
    # while this harness, loading the config only post-construction and
    # only pre-verification, never saw it -- the sweep must BE the
    # clean-run launch test, and must be re-run after any verification
    # that writes params/matrices (persisted state arms new load paths).
    w = MainWindow(CONFIG)
    w._activate_fov(FOV)
    return app, w


def candidates(w):
    out = []
    for name in sorted(dir(w)):
        if name in SKIP or name.startswith(SKIP_PREFIX) or name.startswith('__'):
            continue
        try:
            attr = getattr(w, name)
        except Exception:
            continue
        if not callable(attr):
            continue
        code = getattr(attr, '__func__', attr).__code__ if hasattr(attr, '__code__') or hasattr(attr, '__func__') else None
        if code is None:
            continue
        required = code.co_argcount - 1 - len(getattr(attr, '__defaults__', None) or ())
        if required == 0 and getattr(attr, '__module__', '').startswith('windows'):
            out.append((name, attr))
    return out


def tb_is_call_site(tb):
    """True when the TypeError came from OUR call, not from inside the method
    -- i.e. the signature needs arguments we did not supply, which is a skip
    rather than a failure."""
    return traceback.extract_tb(tb)[-1].filename.endswith('smoke_main_window.py')


def main():
    app, w = build()
    failures = []
    skipped = []
    with mock.patch.object(QtWidgets.QMessageBox, 'information'), \
            mock.patch.object(QtWidgets.QMessageBox, 'warning'), \
            mock.patch.object(QtWidgets.QMessageBox, 'critical'), \
            mock.patch.object(QtWidgets.QMessageBox, 'question'):
        methods = candidates(w)
        for name, fn in methods:
            try:
                fn()
            except TypeError as e:
                if 'required positional argument' in str(e) and tb_is_call_site(sys.exc_info()[2]):
                    skipped.append(name)   # staticmethod: co_argcount has no self
                    continue
                tb = traceback.extract_tb(sys.exc_info()[2])
                where = f'{os.path.basename(tb[-1].filename)}:{tb[-1].lineno}'
                failures.append((name, where, f'{type(e).__name__}: {e}'))
            except Exception as e:
                tb = traceback.extract_tb(sys.exc_info()[2])
                where = f'{os.path.basename(tb[-1].filename)}:{tb[-1].lineno}'
                failures.append((name, where, f'{type(e).__name__}: {e}'))
    print(f'called {len(methods) - len(skipped)} no-arg methods, '
          f'{len(failures)} raised ({len(skipped)} skipped: need args)\n')
    for name, where, err in failures:
        print(f'  {name}')
        print(f'      {where}  {err[:110]}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
