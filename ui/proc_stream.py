"""
Run an external command and stream its stdout back to Qt.

IT LIVES IN ui/ BECAUSE BOTH SIDES NEED IT and only one direction
of import is allowed: windows/ already imports ui/, and nothing in
ui/ imports windows/. The main window uses it for bundle builds
started from the Spot Localization panel; ui/model_build_dialog.py
uses it for the same builds and for training. Duplicating it would
have been two progress parsers to keep in step with one tool's
output format.
"""
import re
import subprocess

from PyQt5 import QtCore


class StreamingProcWorker(QtCore.QThread):
    """
    Run ONE external command and stream its stdout back, line by line.

    The third worker shape here, and the one the other two cannot cover:
    FnWorker runs a callable in this process, ProcWorker runs picklable
    jobs in a pool, and neither can host a command-line tool that
    reports its own progress on stdout. Bundle building is exactly that
    -- tools/build_bundle.py already prints `[done/total]` lines, and it
    is a SEPARATE PROGRAM whose whole point is that it does not share
    this process's h5py lock.

    WHY NOT subprocess.run. This started as a blocking `run()` on the GUI
    thread with a comment admitting it. On a real six-FOV MP58 bundle
    that freezes the entire application for the length of the build --
    every panel, every viewer, the log itself -- for work that has
    nothing to do with the rest of the app and holds none of its state.

    `-u` is added to the interpreter by the caller: without it Python
    block-buffers a pipe and the progress lines arrive in 8 KB clumps at
    the end, which is a progress bar that jumps from 0 to done.

    Cancellation is deliberately NOT offered. The child writes a bundle
    directory and its own manifest; killing it midway would leave a
    half-written cache that looks finished. It is cheap to rebuild and
    harmless to leave running.
    """
    line = QtCore.pyqtSignal(str)                # one stdout line
    progress = QtCore.pyqtSignal(int, int)       # (done, total)
    finished_ok = QtCore.pyqtSignal(int)         # exit code
    failed = QtCore.pyqtSignal(str)

    # '  [  12/ 666] fov007 Hyb_101 ...' -- build_bundle's own format
    _PROGRESS = re.compile(r'^\s*\[\s*(\d+)\s*/\s*(\d+)\s*\]')

    def __init__(self, cmd, cwd=None):
        super().__init__()
        self.cmd = list(cmd)
        self.cwd = cwd

    def run(self):
        try:
            proc = subprocess.Popen(
                self.cmd, cwd=self.cwd, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
                encoding='utf-8', errors='replace')
        except Exception as e:                              # noqa: BLE001
            self.failed.emit(f'{type(e).__name__}: {e}')
            return
        try:
            for raw in proc.stdout:
                text = raw.rstrip()
                m = self._PROGRESS.match(text)
                if m:
                    self.progress.emit(int(m.group(1)), int(m.group(2)))
                if text:
                    self.line.emit(text)
            proc.wait()
        except Exception as e:                              # noqa: BLE001
            self.failed.emit(f'{type(e).__name__}: {e}')
            return
        finally:
            try:
                if proc.stdout is not None:
                    proc.stdout.close()
            except Exception:                               # noqa: BLE001
                pass
        self.finished_ok.emit(int(proc.returncode or 0))
