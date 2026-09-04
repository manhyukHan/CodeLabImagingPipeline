"""
No alignment runs on a FOV whose ingestion is unfinished -- in EITHER mode.

Append mode has refused such a FOV all along. Overwrite did not, in any of
the three alignment types, so the same button was safe in one mode and not
the other. That asymmetry is the bug this pins.

Why it matters more than "the fit would fail": the fit does not fail. A
hybe whose MIP is not on disk reaches chain._cell_native_crop, which
returns a bare None -- the SAME None it returns for a cell that genuinely
projects off-frame -- and an identity residual is persisted with
provenance asserting the off-frame cause. Nothing afterwards can separate
the two. Append then decides the hybe is done purely by the key existing,
so a hybe that was merely LATE is permanently marked aligned, and because
its H2 is identity it scores 0.0 px on _cell_max_residual_shift, the
lowest possible: the one entry guaranteed never to be flagged for review.

So the check belongs before any work starts, it must cover overwrite, and
it must name what it excluded -- a silent exclusion is how "only one FOV
was calculated" got reported twice without anyone being able to say why.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_alignment_ingestion_gate.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from unittest import mock                                # noqa: E402
from PyQt5 import QtWidgets                              # noqa: E402

import windows.main_window as MW                         # noqa: E402
from windows.main_window import MainWindow               # noqa: E402

# Before any widget exists: constructing a MainWindow without one crashes
# the interpreter outright (STATUS_STACK_BUFFER_OVERRUN, no traceback).
_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}' + (f'   [{detail}]' if detail else ''))


def main():
    mw = MainWindow()

    print('\n-- the gate itself --')
    ready = {(m, f): set(h) for (m, f), h in {
        ('DNA', 1): ['H0', 'H1'],
        ('DNA', 2): ['H0'],              # H1 still arriving
        ('RNA', 1): ['R0'],
        ('RNA', 2): ['R0'],
    }.items()}
    with mock.patch.object(mw, '_ready_hybes',
                           side_effect=lambda m, f: ready.get((m, f), set())):
        out = mw._incomplete_fovs([1, 2], [('DNA', ['H0', 'H1'])])
        check('a FOV missing a hybe is reported', list(out) == [2], str(list(out)))
        check('and the reason names the modality and the count',
              'DNA' in out[2][0] and '1 of 2' in out[2][0], str(out.get(2)))

        out = mw._incomplete_fovs([1, 2], [('DNA', ['H0'])])
        check('a FOV is judged only on what the run will actually read',
              out == {}, str(out))

        multi = mw._incomplete_fovs([1, 2], [('DNA', ['H0', 'H1']), ('RNA', ['R0'])])
        check('every modality the run touches is checked',
              list(multi) == [2], str(list(multi)))

    print('\n-- a check that cannot be made never excludes --')
    # The gate only ever removes work it can positively show is not ready.
    with mock.patch.object(mw, '_ready_hybes', side_effect=RuntimeError('no scan')):
        check('a failing readiness scan excludes nothing',
              mw._incomplete_fovs([1, 2], [('DNA', ['H0'])]) == {})
    with mock.patch.object(mw, '_ready_hybes', side_effect=lambda m, f: set()):
        check('a pass with no modality excludes nothing',
              mw._incomplete_fovs([1], [(None, ['H0'])]) == {})
        check('a pass with no hybes excludes nothing',
              mw._incomplete_fovs([1], [('DNA', [])]) == {})
        check('but a real requirement with nothing ready DOES exclude',
              list(mw._incomplete_fovs([1], [('DNA', ['H0'])])) == [1])

    print('\n-- the exclusion is announced, not silent --')
    notices = []
    with mock.patch.object(QtWidgets.QMessageBox, 'information',
                           side_effect=lambda *a, **k: notices.append(a[2] if len(a) > 2 else '')):
        detail = mw._report_excluded_fovs(
            'Run X', {3: ['DNA: 1 of 2 hybe(s) not ingested yet'],
                      7: ['DNA: 2 of 2 hybe(s) not ingested yet']},
            'Consequence sentence.')
    check('the notice names each excluded FOV',
          'FOV003' in detail and 'FOV007' in detail, detail)
    check('and reaches the user, with the consequence spelled out',
          notices and 'FOV003' in notices[0] and 'Consequence sentence.' in notices[0],
          str(notices[:1]))

    long = {f: ['DNA: 1 of 2 hybe(s) not ingested yet'] for f in range(1, 21)}
    with mock.patch.object(QtWidgets.QMessageBox, 'information',
                           side_effect=lambda *a, **k: None):
        detail = mw._report_excluded_fovs('Run X', long, 'c')
    check('a long list is truncated rather than unreadable',
          '+12 more' in detail and detail.count('FOV') == 8, detail[-40:])

    print('\n-- an OVERWRITE ingestion takes the stores to itself --')
    # The other direction of the same rule. Overwrite rewrites stacks and
    # MIPs that other work is mid-read of; on Windows os.replace fails
    # outright while a reader holds the file (measured: PermissionError
    # [WinError 5]), and the case that is worse than failing is succeeding
    # -- an alignment running across the swap fits some hybes against the
    # old pixels and some against the new, and records that as a result.
    class FakeWorker:
        def __init__(self):
            self.running = True
            self.terminated = False

        def isRunning(self):
            return self.running

        def terminate(self):
            self.terminated = True
            self.running = False

        def wait(self, _ms=0):
            return True

    for attr in mw.STORE_WORKER_ATTRS:
        setattr(mw, attr, None)
    mw._fov_overlay_workers = []
    check('nothing running means nothing to stop', mw._store_workers_running() == [])
    check('and an overwrite proceeds without asking anything',
          mw._clear_for_overwrite_ingestion(ask=True) is True)

    align = FakeWorker()
    overlay = FakeWorker()
    mw._cell_alignment_worker = align
    mw._fov_overlay_workers = [overlay]
    running = mw._store_workers_running()
    check('a live alignment and a live overlay render are both seen',
          len(running) == 2, str([lbl for lbl, _w in running]))

    asked = []
    with mock.patch.object(QtWidgets.QMessageBox, 'question',
                           side_effect=lambda *a, **k: (asked.append(a[2]),
                                                        QtWidgets.QMessageBox.Cancel)[1]):
        proceed = mw._clear_for_overwrite_ingestion(ask=True)
    check('cancelling leaves the run alone', proceed is False)
    check('and stops nothing', align.running and overlay.running)
    check('the prompt says what is running and why it must stop',
          asked and 'cell alignment worker' in asked[0] and 'replaced' in asked[0],
          str(asked[:1])[:120])

    with mock.patch.object(QtWidgets.QMessageBox, 'question',
                           return_value=QtWidgets.QMessageBox.Yes), \
         mock.patch.object(MW.multiprocessing, 'active_children', return_value=[]):
        proceed = mw._clear_for_overwrite_ingestion(ask=True)
    check('confirming proceeds', proceed is True)
    check('and every reader is actually stopped',
          align.terminated and overlay.terminated)

    # A queued run was authorised when the queue started; it must not stall
    # on a dialog, but it must not be silent either.
    align2 = FakeWorker()
    mw._cell_alignment_worker = align2
    mw._fov_overlay_workers = []
    logged = []
    with mock.patch.object(mw, 'log', side_effect=logged.append), \
         mock.patch.object(QtWidgets.QMessageBox, 'question',
                           side_effect=AssertionError('a queued run must not prompt')), \
         mock.patch.object(MW.multiprocessing, 'active_children', return_value=[]):
        proceed = mw._clear_for_overwrite_ingestion(ask=False)
    check('a queued overwrite never prompts', proceed is True)
    check('but it does stop the reader', align2.terminated)
    check('and says so in the log',
          any('stopped 1 running operation' in m for m in logged), str(logged[:2]))

    print('\n-- APPEND is never interrupted --')
    # Append only ADDS hybes nobody is reading yet, so a concurrent
    # alignment sees a store that only grows. That is the whole point of
    # the mid-ingestion append workflow, and the guard must stay out of it.
    src = __import__('inspect').getsource(MW.MainWindow._run_ingestion)
    check('the guard is reached only for overwrite',
          "overwrite_mode == 'overwrite' and not self._clear_for_overwrite_ingestion" in src,
          'guard is not mode-gated')
    qsrc = __import__('inspect').getsource(MW.MainWindow._run_next_queued_job)
    check('and in the queue only when the queue itself is overwrite',
          'if self._job_queue_overwrite:' in qsrc)

    for attr in mw.STORE_WORKER_ATTRS:
        setattr(mw, attr, None)
    mw._fov_overlay_workers = []

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        for f in FAIL:
            print('  FAILED:', f)
        return 1
    print('ALL GOOD')
    return 0


if __name__ == '__main__':
    sys.exit(main())
