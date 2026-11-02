"""
MainWindow.log() may be called from a worker thread, and must not touch
the log widget from there.

WHAT WENT WRONG. Every compute module in this repo takes a `log=`
callback, and the pipeline runner hands the localization batch one that
ends in MainWindow.log() -- called from inside an FnWorker. log() wrote
straight to a QPlainTextEdit. Appending there re-enters the widget's
QTextDocument, which connects its own layout signals; off the GUI thread
that connection is cross-thread, Qt cannot marshal its arguments, and the
app printed

    QObject::connect: Cannot queue arguments of type 'QTextBlock'
    QObject::connect: Cannot queue arguments of type 'QTextCursor'

and then died -- mid-run, with the shell prompt back and no traceback.
The warnings name the document's own signals, so they point at the widget
and not at the logging, which is why this was easy to read as noise.

WHY THE TEST IS HERE AND NOT AT THE CALL SITE. The fault is reachable
from any `log=` callback anyone ever passes into a worker, so the sink is
the only place a fix stays fixed. These checks therefore pin the sink's
contract, not the pipeline's use of it:

  * a GUI-thread call still appends immediately (ordering against
    everything else that thread does is unchanged -- a queued-always
    sink would have reordered the log against the work it describes);
  * a worker-thread call is delivered ON THE GUI THREAD;
  * the line still arrives, intact, exactly once.

The third matters as much as the second: a "fix" that drops off-thread
lines silently would pass a thread check and lose the log.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_log_thread_safety.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt5 import QtCore, QtWidgets                      # noqa: E402

from windows.main_window import MainWindow               # noqa: E402

# Before any widget exists: constructing a MainWindow without one crashes
# the interpreter outright (STATUS_STACK_BUFFER_OVERRUN, no traceback).
_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}' + (f'   [{detail}]' if detail else ''))


class LogCaller(QtCore.QThread):
    """One mw.log() from off the GUI thread, the way a `log=` callback
    handed to a compute module reaches it."""

    def __init__(self, mw, message):
        super().__init__()
        self.mw, self.message = mw, message
        self.tid = None

    def run(self):
        self.tid = int(QtCore.QThread.currentThreadId())
        self.mw.log(self.message)


def main():
    mw = MainWindow()
    gui_tid = int(QtCore.QThread.currentThreadId())

    # Where the append actually happens, recorded per call. The real
    # LogWindow.append is kept and wrapped, so the test exercises the
    # widget write it is about rather than a stand-in for it.
    seen = []
    real_append = mw.log_window.append

    def recording_append(message):
        seen.append((str(message), int(QtCore.QThread.currentThreadId())))
        return real_append(message)
    mw.log_window.append = recording_append

    print('\n-- from the GUI thread --')
    mw.log('on the gui thread')
    check('appends without waiting for the event loop', len(seen) == 1, f'{len(seen)} append(s)')
    check('and appends on the GUI thread', bool(seen) and seen[-1][1] == gui_tid,
          f'{seen[-1][1] if seen else None} vs gui {gui_tid}')

    print('\n-- from a worker thread --')
    seen.clear()
    worker = LogCaller(mw, 'from the worker')
    worker.start()
    ok = worker.wait(10000)
    check('the worker finished', ok)
    check('nothing was appended while the worker ran', not seen,
          f'{len(seen)} append(s) before the queue drained')
    for _ in range(200):                      # let the queued emit arrive
        _app.processEvents()
        if seen:
            break
    check('the line is not lost', len(seen) == 1, f'{len(seen)} append(s): {seen}')
    check('it arrives intact', bool(seen) and seen[0][0] == 'from the worker',
          repr(seen[0][0]) if seen else 'nothing')
    check('and it is appended on the GUI THREAD, not the worker',
          bool(seen) and seen[0][1] == gui_tid and seen[0][1] != worker.tid,
          f'appended on {seen[0][1] if seen else None}, worker {worker.tid}, gui {gui_tid}')

    print('\n-- the whole line survives the trip --')
    seen.clear()
    long_line = 'localize DNA ch635: ' + 'x' * 400
    w2 = LogCaller(mw, long_line)
    w2.start()
    w2.wait(10000)
    for _ in range(200):
        _app.processEvents()
        if seen:
            break
    check('a long line is not truncated', bool(seen) and seen[0][0] == long_line,
          f'{len(seen[0][0]) if seen else 0} of {len(long_line)} chars')

    mw.log_window.append = real_append
    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        print('FAILED:', FAIL)
        sys.exit(1)
    print('ALL GOOD')


if __name__ == '__main__':
    main()
