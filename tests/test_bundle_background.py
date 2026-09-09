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

CHECKS = [0, 0]


def check(name, ok, note=''):
    CHECKS[0] += 1
    CHECKS[1] += 1 if ok else 0
    print(('  ok   ' if ok else '  FAIL ') + name
          + (('  -- ' + note) if note else ''))


def run_worker(cmd, timeout=25.0):
    """Drive one worker to completion on a real event loop."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    got = {'lines': [], 'progress': [], 'code': None, 'fail': None,
           'stamps': []}
    w = StreamingProcWorker(cmd)
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


def test_the_gui_wiring():
    print('the GUI side')
    import inspect
    from windows.main_window import MainWindow as MW

    src = inspect.getsource(MW._start_bundle_build)
    check('the blocking subprocess.run is gone',
          'subprocess.run' not in src)
    check('it uses the streaming worker', 'StreamingProcWorker' in src)
    check("the child interpreter is unbuffered ('-u')", "'-u'" in src)
    check('it drives the Spot Localization progress bar',
          'SpotLocalizationPanel' in src and 'ProgressBar' in src)
    check('a second build is refused while one runs',
          '_bundle_worker' in src and 'already running' in src)
    check('the button is disabled for the duration',
          'MakeModelPushButton.setEnabled(False)' in src)

    done = inspect.getsource(MW._on_bundle_done)
    check('the button comes back on every exit path',
          'MakeModelPushButton.setEnabled(True)' in done)
    check('spot check is offered only on success',
          done.index('if int(code) != 0') < done.index('_offer_spotcheck'))
    check('the worker slot is cleared so another build can start',
          '_bundle_worker = None' in done)

    prog = inspect.getsource(MW._on_bundle_progress)
    check('progress sets both the maximum and the value',
          'setMaximum' in prog and 'setValue' in prog)

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
    test_the_gui_wiring()
    print()
    print('%d/%d checks passed' % (CHECKS[1], CHECKS[0]))
    return 0 if CHECKS[1] == CHECKS[0] else 1


if __name__ == '__main__':
    sys.exit(main())
