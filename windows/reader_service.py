"""
One long-lived child process that owns every store read the GUI needs.

WHY A CHILD AT ALL
------------------
h5py holds one lock per PROCESS (`h5py._objects.phil`), taken for the whole
call -- and since the stacks are gzip+shuffle (measured: 222.7 MB on disk
against 444 MB raw, chunks (32,32,64)), "the whole call" includes
decompressing up to 220 MB. It is a CPU lock, not merely an I/O one, so it
bites even when the disk is idle.

The size of that bite is already measured in this repo: one background
THREAD doing slab reads took a 16.5 ms MIP open to 2043 ms -- 124x -- while
the same reads in separate PROCESSES left it at 16.8 ms, untouched. That is
why ProcWorker exists.

WHY ONE CHILD, WHEN SEPARATE LOCKS WERE THE POINT
-------------------------------------------------
Because escaping the lock and answering a click quickly are different
problems, and ProcWorker only solves the first.

ProcWorker spawns a FRESH pool per request. On Windows 'spawn' that is a new
interpreter and a new numpy import before a single byte is read, the number
alive at once is unbounded, and the MIP cache the child warms up dies with
it -- so every read starts cold. Worse, when the user clicks through
hybes/channels, every superseded request still runs to completion in its own
process, and they contend with each other for the same disk.

A single resident reader fixes all four:

  * the spawn is paid once, at first use, not per image
  * its MIP cache stays warm across requests
  * one process, so no unbounded fan-out and one cache budget
  * there is a QUEUE, which is the real win: a new request for the same
    view CANCELS the superseded one. Clicking through five hybes reads
    one, not five.

Serialising the GUI's own reads costs nothing -- the window shows one image
at a time -- and the batch pools (alignment's 16 processes) are untouched
and stay parallel. Same principle, opposite conclusion, because the work is
different: throughput there, latency here.

The reader deliberately does NOT lower its I/O priority. The batch children
sit at VERYLOW so they stay out of the way; putting the process that answers
the user's clicks behind them would be backwards.
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from PyQt5 import QtCore

from codelab_pipeline import process_guard


def _reader_child_init():
    """The reader child's own initializer.

    Deliberately NOT process_guard.child_initializer, and the two
    differences are the point:

      * no lower_io_priority() -- see the module docstring; this child
        serves clicks, and the batch pools are the ones that must yield.
      * no CODELAB_MIP_CACHE_GB cap -- that cap exists because a pool has
        42 children and 42 x 4 GB is not a budget. There is exactly one
        reader, so analysis_store's own sizing (5% of available RAM,
        clamped to [1, 4] GB) is already the right answer for it.

    The parent-death guard is kept, because an orphaned reader is exactly
    the failure process_guard was written for.
    """
    process_guard.child_guard()


class StoreReader(QtCore.QObject):
    """Ask for a store read; get the answer back on the GUI thread.

        reader = StoreReader(main_window)
        reader.read('fov_display',
                    analysis_store.read_hybe_mip, (sp, fov, hybe, ch),
                    on_ok=self._show_mip, on_fail=self.log)

    `slot` names the VIEW the answer is for, and is the whole coalescing
    mechanism: a second read on the same slot means the first answer is no
    longer wanted, so it is cancelled if it has not started and discarded
    if it has. Two different slots never interfere.

    `fn` must be a module-level function and `args` picklable -- 'spawn'
    start method, same rule as every other pool here.
    """

    # Emitted from a pool thread, delivered on the GUI thread. Private:
    # callers use the on_ok/on_fail callbacks, which is what makes the
    # slot bookkeeping possible at all.
    _arrived = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pool = None
        self._seq = {}          # slot -> the only request whose answer is wanted
        self._pending = {}      # slot -> (future, seq, on_ok, on_fail)
        self.stats = {'requested': 0, 'served': 0, 'superseded': 0,
                      'cancelled_before_start': 0, 'failed': 0}
        self._arrived.connect(self._deliver, QtCore.Qt.QueuedConnection)

    # -- lifecycle ------------------------------------------------------

    def _ensure_pool(self):
        """Start the reader on first use, not at launch.

        Launch latency was cut from 8.3 s to 0.25 s and from 7.68 s to
        0.88 s by removing exactly this kind of eager work; a 'spawn' of a
        fresh interpreter plus numpy would put a chunk of it straight back
        for a user who never opens an image.
        """
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context('spawn'),
                initializer=_reader_child_init)
        return self._pool

    def shutdown(self, wait=False):
        """Stop the reader. Safe to call more than once, and safe when it
        was never started."""
        pool, self._pool = self._pool, None
        self._pending.clear()
        if pool is not None:
            pool.shutdown(wait=wait, cancel_futures=True)

    # -- the request ----------------------------------------------------

    def read(self, slot, fn, args=(), on_ok=None, on_fail=None):
        """Queue one read for `slot`, superseding whatever it had pending.

        Returns the sequence number of this request, which is only useful
        for tests -- callers get their answer through on_ok.
        """
        self.stats['requested'] += 1
        self._supersede(slot)
        seq = self._seq[slot] = self._seq.get(slot, 0) + 1
        try:
            pool = self._ensure_pool()
            future = pool.submit(fn, *args)
        except Exception as e:
            # A pool that will not start must not take the click with it.
            # Dropping the handle matters as much as reporting: a reader
            # whose child died leaves a permanently broken executor, and
            # every later click would fail against the corpse. Clearing it
            # makes the next request spawn a fresh one.
            self._pool = None
            self.stats['failed'] += 1
            if on_fail:
                on_fail(f'{type(e).__name__}: {e}')
            return seq
        self._pending[slot] = (future, seq, on_ok, on_fail)
        # add_done_callback fires on a pool thread; the signal hop is what
        # gets the answer onto the GUI thread, where the callbacks may
        # legally touch widgets.
        future.add_done_callback(
            lambda fut, s=slot, n=seq: self._arrived.emit((s, n, fut)))
        return seq

    def _supersede(self, slot):
        """Drop the answer `slot` is currently waiting for.

        Cancel wins when the request has not started -- the read never
        happens at all, which is the point. Once it is running there is
        nothing to cancel, so the answer is discarded on arrival instead
        by the sequence check in _deliver.
        """
        prev = self._pending.pop(slot, None)
        if prev is None:
            return
        future = prev[0]
        if future.cancel():
            self.stats['cancelled_before_start'] += 1
        else:
            self.stats['superseded'] += 1

    # -- the answer, back on the GUI thread ------------------------------

    def _deliver(self, payload):
        slot, seq, future = payload
        if seq != self._seq.get(slot):
            return                    # superseded while in flight; drop it
        entry = self._pending.pop(slot, None)
        on_ok = entry[2] if entry else None
        on_fail = entry[3] if entry else None
        if future.cancelled():
            return
        try:
            result = future.result()
        except Exception as e:
            # A dead child poisons the executor for every later request,
            # so the handle goes with the error rather than after it.
            if 'BrokenProcessPool' in type(e).__name__:
                self._pool = None
            self.stats['failed'] += 1
            if on_fail:
                on_fail(f'{type(e).__name__}: {e}')
            return
        self.stats['served'] += 1
        if on_ok:
            on_ok(result)
