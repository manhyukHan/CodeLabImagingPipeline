"""
Run an external command and stream its output back to Qt.

IT LIVES IN ui/ BECAUSE BOTH SIDES NEED IT and only one direction of
import is allowed: windows/ already imports ui/, and nothing in ui/
imports windows/.

TWO WAYS TO RUN A CHILD, and the difference is whether it survives us.
  pipe   stdout is a pipe this thread reads. Simple, and FATAL to the
         child the moment this process dies: its next print hits a
         closed pipe and raises. MEASURED: a child printing every 0.3 s
         whose parent exited after 1 s never finished.
  file   stdout goes to a log file the child owns, the child is
         started in its own process group, detached, and this thread
         TAILS the file. The same measurement: the child finished, all
         15 lines in its log. A bundle build launched this way keeps
         running after the app quits, and the file is what a relaunched
         app reads to catch up (LogTailWorker).
"""
import os
import re
import subprocess
import sys
import time

from PyQt5 import QtCore

# '  [  12/ 666] fov007 Hyb_101 ...' -- build_bundle's own format
_PROGRESS = re.compile(r'^\s*\[\s*(\d+)\s*/\s*(\d+)\s*\]')


def _detached_kwargs():
    """Popen keywords that make the child outlive this process.

    CREATE_NO_WINDOW, NOT DETACHED_PROCESS. Detached means the child has
    no console at all -- and every console-subsystem process IT then
    starts (a build's spawn pool: 32 python.exe workers) is given a
    brand-new console of its own, which on a real build put thirty-odd
    empty black windows on the desktop. A child started with a HIDDEN
    console keeps one its workers inherit, so nothing appears; the
    console is the child's own, not this process's, so the child still
    outlives us (the survival test in tests/test_bundle_background.py
    runs with exactly these flags).
    """
    if sys.platform.startswith('win'):
        return {'creationflags': (subprocess.CREATE_NEW_PROCESS_GROUP
                                  | subprocess.CREATE_NO_WINDOW),
                'close_fds': True}
    return {'start_new_session': True, 'close_fds': True}


class _Tail(object):
    """Incremental line reader over a file another process appends to."""

    def __init__(self, path):
        self.path = str(path)
        self.pos = 0
        self.buf = ''

    def lines(self):
        try:
            with open(self.path, 'r', encoding='utf-8', errors='replace') as f:
                f.seek(self.pos)
                chunk = f.read()
                self.pos = f.tell()
        except OSError:
            return []
        if not chunk:
            return []
        self.buf += chunk
        out = self.buf.split('\n')
        self.buf = out.pop()          # a partial last line waits
        return [t.rstrip('\r') for t in out]


class StreamingProcWorker(QtCore.QThread):
    """Run ONE command; emit its lines, its [done/total] progress, its exit.

    log_path=None  -> pipe mode (see module docstring).
    log_path=PATH  -> file mode: the child appends to PATH, detached, and
                      this thread tails PATH until the child exits.

    Cancellation is deliberately NOT offered: the child writes a bundle
    and its own manifest, and killing it midway is what the .part
    discipline exists to survive -- but there is no reason to cause it.
    """
    line = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int, int)
    finished_ok = QtCore.pyqtSignal(int)
    failed = QtCore.pyqtSignal(str)

    _PROGRESS = _PROGRESS

    def __init__(self, cmd, cwd=None, log_path=None):
        super().__init__()
        self.cmd = list(cmd)
        self.cwd = cwd
        self.log_path = str(log_path) if log_path else None

    def _emit(self, text):
        m = self._PROGRESS.match(text)
        if m:
            self.progress.emit(int(m.group(1)), int(m.group(2)))
        if text:
            self.line.emit(text)

    def run(self):
        if self.log_path:
            self._run_file()
        else:
            self._run_pipe()

    def _run_pipe(self):
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
                self._emit(raw.rstrip())
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

    def _run_file(self):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)) or '.',
                        exist_ok=True)
            tail = _Tail(self.log_path)
            # Start tailing from the END of an existing log: an appended
            # build should not replay the previous run's lines.
            try:
                tail.pos = os.path.getsize(self.log_path)
            except OSError:
                tail.pos = 0
            log = open(self.log_path, 'a', encoding='utf-8')
            try:
                log.write('--- ' + time.strftime('%Y-%m-%d %H:%M:%S') + '  '
                          + ' '.join(self.cmd) + '\n')
                log.flush()
                proc = subprocess.Popen(
                    self.cmd, cwd=self.cwd, stdout=log,
                    stderr=subprocess.STDOUT, **_detached_kwargs())
            finally:
                log.close()          # the child holds its own handle now
        except Exception as e:                              # noqa: BLE001
            self.failed.emit(f'{type(e).__name__}: {e}')
            return
        try:
            while True:
                for t in tail.lines():
                    self._emit(t)
                if proc.poll() is not None:
                    for t in tail.lines():
                        self._emit(t)
                    break
                time.sleep(0.25)
        except Exception as e:                              # noqa: BLE001
            self.failed.emit(f'{type(e).__name__}: {e}')
            return
        self.finished_ok.emit(int(proc.returncode or 0))


class LogTailWorker(QtCore.QThread):
    """Follow a build log this process did not start.

    For a relaunched app finding a bundle whose manifest is not yet
    complete: the build may still be running somewhere. There is no
    process handle to wait on, so completion is read from the world --
    `is_done()` (the manifest says complete) -- and abandonment from
    silence: a log that has not grown for `stale_seconds` is reported
    as stalled, not finished. finished_ok(1) means done, (0) stalled.
    """
    line = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int, int)
    finished_ok = QtCore.pyqtSignal(int)

    def __init__(self, log_path, is_done, stale_seconds=180, replay=40):
        super().__init__()
        self.log_path = str(log_path)
        self.is_done = is_done
        self.stale = float(stale_seconds)
        self.replay = int(replay)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        tail = _Tail(self.log_path)
        # Replay the last few lines so the reader sees where it is.
        try:
            with open(self.log_path, encoding='utf-8', errors='replace') as f:
                last = f.read().splitlines()[-self.replay:]
            for t in last:
                m = _PROGRESS.match(t)
                if m:
                    self.progress.emit(int(m.group(1)), int(m.group(2)))
                self.line.emit(t)
            tail.pos = os.path.getsize(self.log_path)
        except OSError:
            tail.pos = 0
        quiet_since = time.time()
        while not self._stop:
            got = tail.lines()
            for t in got:
                m = _PROGRESS.match(t)
                if m:
                    self.progress.emit(int(m.group(1)), int(m.group(2)))
                self.line.emit(t)
            if got:
                quiet_since = time.time()
            try:
                if self.is_done():
                    self.finished_ok.emit(1)
                    return
            except Exception:                               # noqa: BLE001
                pass
            if time.time() - quiet_since > self.stale:
                self.finished_ok.emit(0)
                return
            time.sleep(0.5)
