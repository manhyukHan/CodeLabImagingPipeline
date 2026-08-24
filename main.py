import os
# Must be set before numpy is ever imported (transitively, by anything below
# this line) -- OpenBLAS's internal thread pool can SIGSEGV when a
# multi-threaded BLAS routine (e.g. np.linalg.inv, used by
# compute_cell_alignment) is invoked from a non-main thread (a QThread
# worker, here), which is exactly the pattern this app's alignment/cell
# workers use. Confirmed via a real crash: SIGSEGV inside
# dgetrf_parallel/dgesv_64_ (libopenblas) called from
# CellAlignmentWorker.run() -> compute_cell_alignment -> la.inv(H1). Forcing
# every BLAS backend to single-threaded avoids the race entirely.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import sys
import warnings
warnings.filterwarnings('ignore')

from PyQt5 import QtWidgets, QtGui

from config import path
from windows.main_window import MainWindow

# The CODE Lab 'O' mark (tools/make_app_icon.py regenerates it) -- set as
# the application icon so every window and the taskbar/dock entry carry
# it, on every platform Qt runs on. PNG here (Qt loads it everywhere);
# the .ico/.icns siblings exist for OS-level shortcuts.
ICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'codelab_o.png')

def _install_error_dialog_hook():
    """
    Show an uncaught exception in a dialog and KEEP RUNNING.

    Without this, any unhandled exception in a slot propagates out of the Qt
    event loop and aborts the process, losing whatever was in memory --
    unsaved spots, a loaded cell container, the current view. An uncaught bug
    is a bug either way, but it should cost a dialog, not the session.

    Deliberately does not catch KeyboardInterrupt/SystemExit: those are the
    ways the app is legitimately asked to stop.
    """
    import traceback

    def hook(exc_type, exc, tb):
        if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            sys.__excepthook__(exc_type, exc, tb)
            return
        detail = ''.join(traceback.format_exception(exc_type, exc, tb))
        sys.stderr.write(detail)          # still on the console for debugging
        try:
            box = QtWidgets.QMessageBox()
            box.setIcon(QtWidgets.QMessageBox.Critical)
            box.setWindowTitle('Unexpected error')
            box.setText(f'{exc_type.__name__}: {exc}')
            box.setInformativeText('The app is still running. This operation did '
                                   'not complete -- your in-memory work is intact.')
            box.setDetailedText(detail)
            box.exec_()
        except Exception:
            pass                          # a dialog failure must not re-raise

    sys.excepthook = hook


if __name__ == '__main__':
    _install_error_dialog_hook()
    question_app = QtWidgets.QApplication(sys.argv)
    question_app.setWindowIcon(QtGui.QIcon(ICON_PATH))
    question_window = QtWidgets.QMainWindow()
    question_window.show()
    config_file = QtWidgets.QFileDialog.getOpenFileName(question_window, 'Load configuration file (Cancel to start fresh)', path, 'configuration file (*.xml)')[0]
    question_window.close()

    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(ICON_PATH))
    window = MainWindow(config_file if config_file != '' else None)
    # the combined log comes up with the app; shown first so the main
    # window lands on top of it
    window.show_log_window()
    window.show()
    sys.exit(app.exec_())
