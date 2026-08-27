"""
One process pool, with the two mistakes this repo has already made in it
fixed by construction.

MISTAKE 1: MORE WORKERS IS FASTER
---------------------------------
It is not. Ingestion measured 66 MB/s at 36 workers against 117.6 MB/s at
12 -- three times the processes for HALF the throughput, because the work
was NAS reads and the share contended with itself. So `jobs` is never
guessed from `cpu_count()` alone: callers say what KIND of work it is
(`kind='cpu'` or `kind='io'`) and get a default that was measured for
that kind, and any specific number a caller passes wins over both.

MISTAKE 2: EVERY WORKER TAKES THE WHOLE MACHINE
-----------------------------------------------
numpy/scipy link a threaded BLAS that sizes its pool from the machine, not
from the caller's share of it. On this box that is 64 threads per process;
32 processes would ask for 2048 threads on 32 physical cores. The threads
do not fail, they just fight -- context switching and cache thrash on work
that was already saturating its core. Workers are therefore PINNED to one
BLAS thread each, and the pinning is asserted in the tests rather than
assumed, because it is set through the environment and environments are
easy to get wrong on spawn.

WHAT IT GUARANTEES
------------------
* results come back in INPUT ORDER, always, whatever order they finish in
* one item raising does not lose the others -- it comes back as a Failure
  and the rest complete
* jobs=1 runs INLINE, in this process, with no pool at all, so a parallel
  path can always be compared against the identical serial one (and
  debugged with a breakpoint that actually stops)

WINDOWS SPAWN
-------------
Every child re-imports the module that defines `fn`, so `fn` and anything
it closes over must be importable top-level names -- no lambdas, no
closures, no bound methods of live objects. Per-worker state that is
expensive to build (a MainWindow, a store handle, an array of prepared
crops) goes through `initializer`, which runs ONCE per worker, not once
per item.
"""
import os
import sys
import traceback

# Measured defaults, not guesses. Both come from this repo's own numbers.
#
#   io  : ingestion, NAS-backed -- 12 workers 117.6 MB/s, 36 workers 66 MB/s
#   cpu : PSF calibration, no I/O in the inner loop, so it scales with
#         PHYSICAL cores; hyperthreads share an FPU and this work is
#         float-heavy, which is the case where they buy least.
DEFAULT_IO_JOBS = 12

_BLAS_VARS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


class Failure(object):
    """What an item that raised comes back as.

    A sentinel rather than a re-raise: one bad crop out of forty should
    not lose the other thirty-nine, and the caller is usually better
    placed to decide whether the run is still meaningful. Carries the
    child's traceback as text because a traceback object cannot cross a
    process boundary.
    """

    __slots__ = ('index', 'error', 'traceback')

    def __init__(self, index, error, tb):
        self.index = index
        self.error = error
        self.traceback = tb

    def __repr__(self):
        return f'<Failure item {self.index}: {self.error}>'


def physical_cores():
    """Physical cores, falling back to logical, falling back to 1."""
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:
        pass
    return int(os.cpu_count() or 1)


def cpu_budget(kind='cpu', jobs=None, n_items=None):
    """How many workers to actually start.

    Capped by the item count, because a pool larger than the work is
    pure startup cost -- on Windows spawn that is a fresh interpreter and
    a fresh numpy import per worker, which is not cheap.
    """
    if jobs is None:
        jobs = DEFAULT_IO_JOBS if kind == 'io' else physical_cores()
    jobs = max(1, int(jobs))
    if n_items is not None:
        jobs = max(1, min(jobs, int(n_items)))
    return jobs


def blas_env(threads=1):
    """The environment a pinned worker should be spawned with."""
    return {v: str(threads) for v in _BLAS_VARS}


def _pin_blas(threads=1):
    """Pin this process's BLAS to `threads`.

    Sets the environment for libraries that read it at import time, then
    ALSO calls threadpoolctl for any already imported -- numpy may well
    be loaded before this runs, and by then the env var has missed its
    chance. Neither alone is sufficient.
    """
    for v in _BLAS_VARS:
        os.environ[v] = str(threads)
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(threads)
    except Exception:
        pass


_WORKER_STATE = {}


def _worker_init(user_init, user_args, threads):
    _pin_blas(threads)
    if user_init is not None:
        _WORKER_STATE['state'] = user_init(*user_args)


def _worker_call(payload):
    index, fn, item, wants_state = payload
    try:
        if wants_state:
            return index, fn(item, _WORKER_STATE.get('state'))
        return index, fn(item)
    except BaseException as e:                      # noqa: BLE001 -- deliberate
        return index, Failure(index, f'{type(e).__name__}: {e}',
                              traceback.format_exc())


def pmap(fn, items, kind='cpu', jobs=None, initializer=None, initargs=(),
         threads_per_worker=1, on_done=None, chunksize=1):
    """
    Map `fn` over `items` and return the results IN INPUT ORDER.

    fn(item) -- or fn(item, state) when `initializer` is given, where
    `state` is whatever the initializer returned in that worker.

    `on_done(n_done, n_total, index, result)` is called on the PARENT as
    each result arrives, for progress. It is called in completion order,
    not input order; the returned list is still ordered.

    An item that raises comes back as a Failure in its slot.
    """
    items = list(items)
    n = len(items)
    if n == 0:
        return []

    wants_state = initializer is not None
    workers = cpu_budget(kind, jobs, n)

    # jobs==1 is the SERIAL path on purpose: no pool, no pickling, no
    # spawn. It is what the parallel path is tested against, so it has to
    # be the same code shape -- including the initializer and the Failure
    # handling, or the comparison would not be like for like.
    if workers == 1:
        state = initializer(*initargs) if wants_state else None
        out = []
        for i, item in enumerate(items):
            try:
                r = fn(item, state) if wants_state else fn(item)
            except BaseException as e:              # noqa: BLE001
                r = Failure(i, f'{type(e).__name__}: {e}', traceback.format_exc())
            out.append(r)
            if on_done:
                on_done(i + 1, n, i, r)
        return out

    import multiprocessing as mp

    # Children inherit the environment AT SPAWN, so the BLAS vars have to
    # be set here, in the parent, before the pool exists -- setting them
    # inside the child's initializer alone would be too late for a numpy
    # that the interpreter imports on the way in. _pin_blas runs there
    # too, for the libraries already loaded by then.
    saved = {v: os.environ.get(v) for v in _BLAS_VARS}
    os.environ.update(blas_env(threads_per_worker))
    try:
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=workers, initializer=_worker_init,
                      initargs=(initializer, initargs, threads_per_worker)) as pool:
            payloads = [(i, fn, item, wants_state) for i, item in enumerate(items)]
            out = [None] * n
            done = 0
            for index, result in pool.imap_unordered(_worker_call, payloads,
                                                     chunksize=chunksize):
                out[index] = result
                done += 1
                if on_done:
                    on_done(done, n, index, result)
        return out
    finally:
        for v, old in saved.items():
            if old is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = old


def failures(results):
    """The Failure entries in a pmap result, for reporting."""
    return [r for r in results if isinstance(r, Failure)]


def ok(results):
    """The non-Failure entries, order preserved."""
    return [r for r in results if not isinstance(r, Failure)]
