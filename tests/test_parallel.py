"""
Tests for codelab_pipeline/parallel.py.

The properties worth pinning are the ones whose absence is INVISIBLE:
a reordered result set still looks like a result set, an unpinned BLAS
still returns the right numbers, and a swallowed exception still leaves a
list of the right length. Each of those is asserted directly.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_parallel.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline import parallel as PL                     # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}' + (f'   [{detail}]' if detail else ''))


# -- module-level workers: Windows spawn cannot pickle a closure ---------

def _square(x):
    return x * x


def _slow_then_square(x):
    # Deliberately anti-correlated with input order: the LAST item
    # finishes FIRST. If ordering were taken from completion order this
    # would come back reversed, which is the whole point.
    time.sleep((10 - x) * 0.01)
    return x * x


def _raise_on_three(x):
    if x == 3:
        raise ValueError('item three is bad')
    return x * x


def _read_blas_env(_x):
    return {v: os.environ.get(v) for v in
            ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}


def _blas_env_and_pid(_x):
    # Slow enough that the pool actually distributes. With instant items
    # the first worker to start drains the whole queue before the others
    # are ready, which says nothing about pinning or about spreading.
    time.sleep(0.05)
    return (os.getpid(), _read_blas_env(None))


def _numpy_thread_count(_x):
    """What a threaded BLAS actually reports INSIDE the worker."""
    try:
        import numpy  # noqa: F401  (must be imported for the pools to exist)
        from threadpoolctl import threadpool_info
        return [d.get('num_threads') for d in threadpool_info()]
    except ImportError:
        return 'threadpoolctl-missing'


def _make_state():
    # A marker unique to the worker process, so reuse across items is
    # detectable from the parent.
    return {'pid': os.getpid(), 'calls': 0}


def _use_state(x, state):
    state['calls'] += 1
    time.sleep(0.02)          # see _blas_env_and_pid on why this is here
    return (x, state['pid'], state['calls'])


def main():
    print('parallel.pmap')

    # -- ordering ------------------------------------------------------
    items = list(range(10))
    got = PL.pmap(_slow_then_square, items, jobs=4)
    check('results come back in INPUT order, not completion order',
          got == [x * x for x in items], f'{got[:4]}...')

    # -- serial and parallel agree -------------------------------------
    ser = PL.pmap(_square, items, jobs=1)
    par = PL.pmap(_square, items, jobs=4)
    check('jobs=1 and jobs=4 give identical results', ser == par)

    # -- failure isolation ---------------------------------------------
    res = PL.pmap(_raise_on_three, items, jobs=4)
    fails = PL.failures(res)
    check('one item raising does not lose the others',
          len(res) == 10 and len(fails) == 1 and len(PL.ok(res)) == 9)
    check('the failure lands in the FAILING item\'s slot',
          isinstance(res[3], PL.Failure) and res[3].index == 3)
    check('the failure carries the child traceback',
          bool(fails) and 'ValueError' in fails[0].traceback
          and 'item three is bad' in fails[0].error)
    check('serial path reports failures the same way',
          isinstance(PL.pmap(_raise_on_three, items, jobs=1)[3], PL.Failure))

    # -- BLAS pinning ---------------------------------------------------
    # Enough items to genuinely need the pool. A ONE-item call is capped
    # to one worker and runs inline -- correct behaviour, but it reads the
    # PARENT's environment, so it cannot tell you anything about what a
    # child inherits. The first version of this test did exactly that and
    # reported unpinned workers that were never spawned.
    spread = PL.pmap(_blas_env_and_pid, list(range(16)), jobs=4)
    envs = [e for _pid, e in spread]
    check('workers are spawned with BLAS pinned to one thread',
          all(e.get('OMP_NUM_THREADS') == '1' and e.get('MKL_NUM_THREADS') == '1'
              for e in envs), str(envs[0]))
    check('every worker is pinned, not just the first',
          len({pid for pid, _e in spread}) > 1,
          f'{len({pid for pid, _e in spread})} distinct worker pids')
    counts = PL.pmap(_numpy_thread_count, list(range(8)), jobs=4)[0]
    if counts == 'threadpoolctl-missing':
        print('  --  BLAS thread count unverifiable (threadpoolctl not installed)')
    else:
        check('numpy\'s own thread pools report 1 thread inside a worker',
              all(c == 1 for c in counts if c is not None), str(counts))

    # -- the parent's environment is not left modified -------------------
    before = os.environ.get('OMP_NUM_THREADS')
    PL.pmap(_square, items, jobs=2)
    check('pmap restores the parent environment afterwards',
          os.environ.get('OMP_NUM_THREADS') == before,
          f'{before!r} -> {os.environ.get("OMP_NUM_THREADS")!r}')

    # -- worker state is built ONCE per worker, not once per item -------
    out = PL.pmap(_use_state, list(range(12)), jobs=3, initializer=_make_state)
    pids = {pid for _x, pid, _c in out}
    maxcalls = max(c for _x, _p, c in out)
    check('initializer state is REUSED across items in a worker',
          maxcalls > 1, f'max calls per worker = {maxcalls}')
    check('work is spread over more than one worker',
          len(pids) > 1, f'{len(pids)} distinct worker pids')
    check('state-taking fn works on the serial path too',
          [x for x, _p, _c in PL.pmap(_use_state, [1, 2], jobs=1,
                                      initializer=_make_state)] == [1, 2])

    # -- degenerate inputs ----------------------------------------------
    check('empty input returns empty, spawns nothing', PL.pmap(_square, [], jobs=8) == [])
    check('pool is never larger than the item count',
          PL.cpu_budget('cpu', jobs=64, n_items=3) == 3)
    check('a caller-specified job count wins over the kind default',
          PL.cpu_budget('io', jobs=5, n_items=100) == 5)
    check('io and cpu defaults differ (measured, not guessed)',
          PL.cpu_budget('io', n_items=1000) != PL.cpu_budget('cpu', n_items=1000)
          or PL.physical_cores() == PL.DEFAULT_IO_JOBS,
          f'io={PL.cpu_budget("io", n_items=1000)} '
          f'cpu={PL.cpu_budget("cpu", n_items=1000)}')

    # -- progress callback ----------------------------------------------
    seen = []
    PL.pmap(_square, list(range(6)), jobs=3,
            on_done=lambda d, t, i, r: seen.append((d, t)))
    check('on_done fires once per item with a rising count',
          [d for d, _t in seen] == list(range(1, 7)) and all(t == 6 for _d, t in seen),
          str(seen))

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        for f in FAIL:
            print('  FAILED:', f)
        return 1
    print('ALL GOOD')
    return 0


if __name__ == '__main__':
    sys.exit(main())
