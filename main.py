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

# The GUI stack is imported INSIDE the __main__ guard, not here.
#
# On Windows every multiprocessing 'spawn' child re-imports the parent's
# __main__ module (as __mp_main__) so pickled references resolve. With
# these imports at module level, each of the N children of every pool --
# ingestion, FOV alignment, cell alignment, tracing, overlays -- paid for
# PyQt5 and matplotlib before it could run a single task.
#
# Measured: 8 children became usable in 2222 ms with the GUI stack at
# module level (1552 modules each) vs 137 ms without it (133 modules).
# That startup storm is what made the app hitch AT POOL CREATION, before
# any fitting had begun, and it got worse the more workers a pool asked
# for. The children need none of it: their entry points live in Qt-free
# modules and take only plain data.

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

    from PyQt5 import QtWidgets     # local: keeps module import Qt-free

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
    # BEFORE any pool can be created: put this process under an OS-level
    # guarantee that its children cannot outlive it. Confirmed real --
    # a force-kill once left 97 pool workers alive for 2.7 days, holding
    # 5 GB and locking the conda env. See codelab_pipeline/process_guard.py.
    from codelab_pipeline import process_guard
    _guard_state = process_guard.install_parent_guard()

    # Everything below is GUI-only, so it is imported HERE rather than at
    # module level -- see the note at the top: spawn children re-import
    # this file and must not pay for any of it.
    from PyQt5 import QtWidgets, QtGui
    from config import path

    _install_error_dialog_hook()
    # ONE QApplication. Constructing a second while the first is alive is
    # undefined in Qt -- it happened to work, leaving two live objects with
    # QApplication.instance() pointing at the newer one, which is exactly
    # the sort of thing that works until a Qt upgrade decides it does not.
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(ICON_PATH))
    question_window = QtWidgets.QMainWindow()
    question_window.show()
    config_file = QtWidgets.QFileDialog.getOpenFileName(question_window, 'Load configuration file (Cancel to start fresh)', path, 'configuration file (*.xml)')[0]
    question_window.close()

    # The application itself is imported only NOW, after the dialog has
    # been answered. It is 1555 modules and ~2.5 s on an idle machine
    # (measured; it inflates to tens of seconds while a cell-alignment run
    # is competing for the disk), and none of it is needed to ask which
    # config to open -- paying it first meant the user clicked the icon and
    # watched nothing happen for those seconds. Same total work, but the
    # window that asks a question now appears immediately.
    from windows.main_window import MainWindow

    window = MainWindow(config_file if config_file != '' else None)
    # the combined log comes up with the app; shown first so the main
    # window lands on top of it
    window.show_log_window()
    # Logged, not silent: a guard that failed to install is worse than
    # none, because nobody would go looking for orphaned workers.
    window.log(f'Worker-process guard: {_guard_state}')
    window.show()
    sys.exit(app.exec_())
