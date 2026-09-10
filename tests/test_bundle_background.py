"""
Bundle building runs in the background and reports on the progress bar.

WHY. 'Make new model...' called subprocess.run on the GUI thread, with a
comment admitting it was blocking on purpose. On a real six-FOV MP58
bundle that freezes the whole application -- every panel, every viewer,
the log itself -- for the length of the build, for work that shares no
state with the app at all: the child reads the store and writes a new
directory, and touches nothing the session holds.

These tests drive StreamingProcWorker against REAL child processes, not
a mock, because the two things that actually go wrong are buffering (the
child's progress lines sitting in a pipe until it exits, so the bar
jumps 0 -> done) and the exit path. Neither is visible against a fake.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_bundle_background.py
"""
import os
import sys
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5 import QtCore, QtWidgets                          # noqa: E402

from windows.main_window import StreamingProcWorker          # noqa: E402
from ui.proc_stream import LogTailWorker                     # noqa: E402

CHECKS = [0, 0]


def check(name, ok, note=''):
    CHECKS[0] += 1
    CHECKS[1] += 1 if ok else 0
    print(('  ok   ' if ok else '  FAIL ') + name
          + (('  -- ' + note) if note else ''))


def run_worker(cmd, timeout=25.0, log_path=None):
    """Drive one worker to completion on a real event loop."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    got = {'lines': [], 'progress': [], 'code': None, 'fail': None,
           'stamps': []}
    w = StreamingProcWorker(cmd, log_path=log_path)
    w.line.connect(lambda t: got['lines'].append(t))
    w.progress.connect(
        lambda d, t: (got['progress'].append((d, t)),
                      got['stamps'].append(time.perf_counter())))
    w.finished_ok.connect(lambda c: got.__setitem__('code', c))
    w.failed.connect(lambda why: got.__setitem__('fail', why))
    w.start()
    t0 = time.perf_counter()
    while (got['code'] is None and got['fail'] is None
           and time.perf_counter() - t0 < timeout):
        app.processEvents(QtCore.QEventLoop.AllEvents, 20)
    w.wait(3000)
    got['wall'] = time.perf_counter() - t0
    return got


CHILD = r'''
import sys, time
print("work    3 FOVs x 2 hybes = 6 tasks", flush=True)
for i in range(1, 7):
    time.sleep(0.25)
    print("  [%4d/%4d] fov007 Hyb_10%d   3 crops   12 cand" % (i, 6, i),
          flush=True)
print("done in 0.1 min", flush=True)
sys.exit(0)
'''


def test_progress_arrives_while_it_runs():
    print('progress streams, it does not arrive in one lump')
    import tempfile
    fd, path = tempfile.mkstemp(suffix='.py')
    os.write(fd, CHILD.encode('utf-8'))
    os.close(fd)
    try:
        got = run_worker([sys.executable, '-u', path])
        check('the child finished cleanly', got['code'] == 0
              and got['fail'] is None, str(got['fail']))
        check('every [done/total] line was parsed',
              got['progress'] == [(i, 6) for i in range(1, 7)],
              str(got['progress']))
        check('the non-progress lines came through too',
              any('work    3 FOVs' in t for t in got['lines'])
              and any('done in' in t for t in got['lines']))

        # THE BUFFERING TEST, and the reason this uses a real process.
        # The child sleeps 0.25 s between lines; if the pipe were block-
        # buffered every stamp would land within a few ms of the last,
        # at the end. Spread over >= 1 s means they arrived as they were
        # printed.
        st = got['stamps']
        spread = (st[-1] - st[0]) if len(st) > 1 else 0.0
        check('the updates are spread over the run, not clumped at the end',
              spread > 0.9, f'{spread:.2f} s between first and last of 6')
    finally:
        os.unlink(path)


def test_a_failing_child_is_reported_not_raised():
    print('a child that fails is reported, never raised into the loop')
    import tempfile
    fd, path = tempfile.mkstemp(suffix='.py')
    os.write(fd, b'import sys\nprint("boom", flush=True)\nsys.exit(3)\n')
    os.close(fd)
    try:
        got = run_worker([sys.executable, '-u', path])
        check('the exit code comes back as itself', got['code'] == 3,
              str(got['code']))
        check('and its output is still delivered',
              any('boom' in t for t in got['lines']))
        check('failed was not used for a non-zero exit',
              got['fail'] is None)
    finally:
        os.unlink(path)


def test_a_command_that_cannot_start():
    print('a command that cannot start')
    got = run_worker(['this-command-does-not-exist-at-all'], timeout=10)
    check('it reports through failed, and does not raise',
          got['fail'] is not None and got['code'] is None,
          str(got['fail'])[:60])


def test_file_mode_streams_and_keeps_a_log():
    print('file mode: the child writes its own log, and we tail it')
    import tempfile
    fd, path = tempfile.mkstemp(suffix='.py')
    os.write(fd, CHILD.encode('utf-8'))
    os.close(fd)
    d = tempfile.mkdtemp(prefix='buildlog_')
    log = os.path.join(d, 'build.log')
    try:
        with open(log, 'w', encoding='utf-8') as f:
            f.write('an earlier run\nwith two lines\n')
        got = run_worker([sys.executable, '-u', path], log_path=log)
        check('the child finished cleanly', got['code'] == 0
              and got['fail'] is None, str(got['fail']))
        check('every [done/total] line was parsed from the FILE',
              got['progress'] == [(i, 6) for i in range(1, 7)],
              str(got['progress']))
        check("an earlier run's lines are NOT replayed",
              not any('earlier run' in t for t in got['lines']))
        st = got['stamps']
        spread = (st[-1] - st[0]) if len(st) > 1 else 0.0
        check('the updates still arrive as they happen',
              spread > 0.9, f'{spread:.2f} s')
        text = open(log, encoding='utf-8').read()
        check('the log on disk holds the earlier run, the command and the new run',
              'earlier run' in text and '--- ' in text and 'done in' in text)
    finally:
        os.unlink(path)
        import shutil
        shutil.rmtree(d, ignore_errors=True)


ORPHAN_CHILD = r"""
import sys, time
for i in range(12):
    print('tick', i, flush=True)
    time.sleep(0.25)
open(sys.argv[1], 'w').write('done')
"""

ORPHAN_LAUNCHER = r"""
import os, sys, time
sys.path.insert(0, %(repo)r)
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt5 import QtCore, QtWidgets
app = QtWidgets.QApplication([])
from ui.proc_stream import StreamingProcWorker
w = StreamingProcWorker([sys.executable, '-u', %(child)r, %(marker)r],
                        log_path=%(log)r)
w.start()
t0 = time.time()
while time.time() - t0 < 0.8:
    app.processEvents(QtCore.QEventLoop.AllEvents, 20)
os._exit(0)     # the app 'quits' with the build one fifth done
"""


def test_the_child_outlives_its_launcher():
    """THE MEASUREMENT THAT DECIDED THE DESIGN. A child on a pipe died
    with its parent; a child on a file, detached, finished. Here the
    launcher is a real process that exits 0.8 s into a 3 s child."""
    print('a file-mode child outlives the process that started it')
    import subprocess
    import tempfile
    d = tempfile.mkdtemp(prefix='orphan_')
    child = os.path.join(d, 'child.py')
    launcher = os.path.join(d, 'launcher.py')
    marker = os.path.join(d, 'marker.txt')
    log = os.path.join(d, 'build.log')
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        open(child, 'w', encoding='utf-8').write(ORPHAN_CHILD)
        open(launcher, 'w', encoding='utf-8').write(
            ORPHAN_LAUNCHER % {'repo': here, 'child': child,
                               'marker': marker, 'log': log})
        t0 = time.perf_counter()
        rc = subprocess.call([sys.executable, launcher], cwd=here)
        gone = time.perf_counter() - t0
        check('the launcher really exited early', rc == 0 and gone < 2.5,
              f'{gone:.1f} s')
        check('and its child was still running then',
              not os.path.exists(marker))
        deadline = time.perf_counter() + 8
        while not os.path.exists(marker) and time.perf_counter() < deadline:
            time.sleep(0.2)
        check('the child FINISHED without its parent',
              os.path.exists(marker))
        lines = open(log, encoding='utf-8').read().splitlines()
        check('and wrote every line to its own log',
              sum(1 for t in lines if t.startswith('tick')) == 12,
              f'{len(lines)} lines')
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def test_log_tail_worker():
    print('following a log this process did not start')
    import tempfile
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    d = tempfile.mkdtemp(prefix='tail_')
    log = os.path.join(d, 'build.log')
    state = {'done': False}
    try:
        with open(log, 'w', encoding='utf-8') as f:
            f.write('old 1\nold 2\n  [   3/  10] fov001 H  1 crops  2 cand\n')
        got = {'lines': [], 'progress': [], 'fin': None}
        t = LogTailWorker(log, lambda: state['done'], stale_seconds=2.0,
                          replay=2)
        t.line.connect(got['lines'].append)
        t.progress.connect(lambda a, b: got['progress'].append((a, b)))
        t.finished_ok.connect(lambda v: got.__setitem__('fin', v))
        t.start()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 0.6:
            app.processEvents(QtCore.QEventLoop.AllEvents, 20)
        check('it replays the last lines so the reader sees where it is',
              got['lines'][:2] == ['old 2',
                                   '  [   3/  10] fov001 H  1 crops  2 cand'],
              str(got['lines'][:2]))
        check('and the progress in them', got['progress'] == [(3, 10)])
        with open(log, 'a', encoding='utf-8') as f:
            f.write('  [   4/  10] fov001 H  1 crops  2 cand\n')
        t0 = time.perf_counter()
        while len(got['lines']) < 3 and time.perf_counter() - t0 < 3:
            app.processEvents(QtCore.QEventLoop.AllEvents, 20)
        check('a line appended by someone else arrives',
              got['progress'][-1:] == [(4, 10)], str(got['progress']))
        state['done'] = True
        t0 = time.perf_counter()
        while got['fin'] is None and time.perf_counter() - t0 < 3:
            app.processEvents(QtCore.QEventLoop.AllEvents, 20)
        check('is_done() ends it as FINISHED', got['fin'] == 1, str(got['fin']))
        t.wait(2000)

        # stale: nothing appends and is_done never fires
        got2 = {'fin': None}
        t2 = LogTailWorker(log, lambda: False, stale_seconds=0.8, replay=0)
        t2.finished_ok.connect(lambda v: got2.__setitem__('fin', v))
        t2.start()
        t0 = time.perf_counter()
        while got2['fin'] is None and time.perf_counter() - t0 < 4:
            app.processEvents(QtCore.QEventLoop.AllEvents, 20)
        check('a log that goes quiet is reported STALLED, not finished',
              got2['fin'] == 0, str(got2['fin']))
        t2.wait(2000)
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def test_the_gui_wiring():
    print('the GUI side')
    import inspect
    from windows.main_window import MainWindow as MW
    from ui import model_build_dialog as MB

    src = inspect.getsource(MW._make_new_model)
    check('the blocking subprocess.run is gone from the main window',
          'subprocess.run' not in inspect.getsource(MW))
    check('Build model... opens the dialog and caches it',
          'ModelBuildDialog' in src and '_model_build_dialog' in src)
    check('the dialog logs into the main log and refreshes the model list',
          'dlg.logged.connect(self.log)' in src
          and 'model_trained.connect' in src)

    dsrc = inspect.getsource(MB.ModelBuildDialog)
    check('the dialog builds with the streaming worker',
          'StreamingProcWorker' in dsrc)
    check("the child interpreter is unbuffered ('-u')", "'-u'" in dsrc)
    check('a second job is refused while one runs',
          'one at a time' in dsrc)
    check('the real FOV pool is passed, not the 1-41 default',
          "'--fov-pool'" in dsrc)
    check('the same explicit FOV list goes to every channel',
          "'--fovs'" in dsrc)
    check('training runs from the app now',
          'train_spotmodel.py' in dsrc and "'--reviewer'" in dsrc)
    check('closing hides; a running build keeps reporting',
          'event.ignore()' in dsrc and 'self.hide()' in dsrc)

    # The regex has to match build_bundle's real format.
    fmt = '  [%4d/%4d] fov007 Hyb_101   3 crops   12 cand' % (12, 666)
    m = StreamingProcWorker._PROGRESS.match(fmt)
    check("it parses build_bundle's own line format",
          m is not None and (int(m.group(1)), int(m.group(2))) == (12, 666),
          fmt.strip())


def main():
    test_progress_arrives_while_it_runs()
    test_a_failing_child_is_reported_not_raised()
    test_a_command_that_cannot_start()
    test_file_mode_streams_and_keeps_a_log()
    test_the_child_outlives_its_launcher()
    test_log_tail_worker()
    test_the_gui_wiring()
    print()
    print('%d/%d checks passed' % (CHECKS[1], CHECKS[0]))
    return 0 if CHECKS[1] == CHECKS[0] else 1


if __name__ == '__main__':
    sys.exit(main())
