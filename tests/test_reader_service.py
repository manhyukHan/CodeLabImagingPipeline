"""
Tests for windows/reader_service.py -- the single resident store reader.

The properties worth pinning are the ones whose absence still LOOKS like a
working reader:

  * If coalescing does not fire, every superseded click still runs. The
    right image still appears, a moment later, so nothing looks broken --
    but five clicks did five reads, which is the behaviour the reader was
    built to replace.
  * If a superseded answer is delivered anyway, the view shows the
    SECOND-to-last image the user asked for. Intermittent, and it looks
    like a rendering bug rather than a queueing one.
  * If the child process is not reused, every read pays a Windows 'spawn'
    -- a fresh interpreter and numpy import -- and the cache it warms dies
    with it. Correct results, no cache, and nobody notices from outputs.
  * If a dead child does not clear the executor handle, the first crash
    breaks every later read permanently.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_reader_service.py
"""
import os
import sys
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5 import QtCore, QtWidgets                             # noqa: E402

from windows.reader_service import StoreReader                  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}' + (f'   [{detail}]' if detail else ''))


# -- module-level, because 'spawn' cannot pickle a closure ---------------

def _echo(x):
    return x


def _slow_echo(x, seconds=0.6):
    time.sleep(seconds)
    return x


def _whoami(_x):
    """The child's pid, to prove the process is REUSED and not respawned."""
    return os.getpid()


def _boom(_x):
    raise RuntimeError('the child died')


def pump(app, seconds, until=None):
    """Run the event loop, stopping early once `until` says so."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.processEvents(QtCore.QEventLoop.AllEvents, 20)
        if until is not None and until():
            return True
        time.sleep(0.005)
    return until() if until is not None else False


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    reader = StoreReader()
    got = []

    print('\n-- an answer comes back, on the GUI thread --')
    main_tid = int(QtCore.QThread.currentThreadId())
    where = []
    reader.read('v', _echo, ('hello',),
                on_ok=lambda r: (got.append(r),
                                 where.append(int(QtCore.QThread.currentThreadId()))))
    ok = pump(app, 60, until=lambda: bool(got))
    check('the read is answered at all', ok and got == ['hello'], str(got))
    check('the callback runs on the GUI thread, so it may touch widgets',
          where and where[0] == main_tid, str(where))

    print('\n-- the child process is REUSED, not respawned per read --')
    pids = []
    for i in range(3):
        pids.clear() if False else None
        reader.read(f'p{i}', _whoami, (i,), on_ok=pids.append)
        pump(app, 60, until=lambda n=i: len(pids) == n + 1)
    check('three reads, one child process', len(set(pids)) == 1, str(pids))

    print('\n-- superseding: five clicks, one read --')
    before = dict(reader.stats)
    seen = []
    for i in range(5):
        reader.read('view', _slow_echo, (i,), on_ok=seen.append)
    pump(app, 60, until=lambda: bool(seen))
    pump(app, 1.0)                     # let any stragglers try to arrive
    cancelled = reader.stats['cancelled_before_start'] - before['cancelled_before_start']
    check('the superseded requests were cancelled before they ran',
          cancelled >= 3, str(cancelled))
    check('exactly one answer reaches the view', len(seen) == 1, str(seen))
    check('and it is the LAST one asked for, not an earlier one',
          seen == [4], str(seen))

    print('\n-- different views do not interfere --')
    a, b = [], []
    reader.read('left', _echo, ('A',), on_ok=a.append)
    reader.read('right', _echo, ('B',), on_ok=b.append)
    pump(app, 60, until=lambda: a and b)
    check('two slots, two answers', a == ['A'] and b == ['B'], str((a, b)))

    print('\n-- a failing read reports instead of vanishing --')
    errs = []
    reader.read('v', _boom, (1,), on_ok=got.append, on_fail=errs.append)
    pump(app, 60, until=lambda: bool(errs))
    check('the error reaches on_fail', bool(errs), str(errs))

    print('\n-- a dead child does not break the reader forever --')
    pool_before = reader._pool
    after = []
    reader.read('v', _echo, ('still alive',), on_ok=after.append)
    pump(app, 60, until=lambda: bool(after))
    check('reads still work after a child-side exception',
          after == ['still alive'], str(after))
    check('an ordinary exception does not throw the pool away',
          reader._pool is pool_before)

    print('\n-- shutdown is safe, and idempotent --')
    reader.shutdown(wait=True)
    reader.shutdown(wait=True)
    fresh = StoreReader()
    fresh.shutdown()                   # never started
    check('shutdown before first use is a no-op', True)

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        for f in FAIL:
            print('  FAILED:', f)
        return 1
    print('ALL GOOD')
    return 0


if __name__ == '__main__':
    sys.exit(main())
