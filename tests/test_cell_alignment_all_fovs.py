"""
Cell-level residual alignment across ALL FOVs (auto-save), and what it
does NOT disturb.

Four properties:

1. Every FOV in the Ingestion tab's list is aligned in ONE run, one job
   per FOV, with cells HYDRATED first -- a FOV never visited this
   session has no cells resident, and skipping that would silently align
   only whatever happened to be on screen.
2. Append is passed through to the worker (fit only per-(cell, hybe)
   residuals not already present), exactly as the per-FOV run does.
3. It is auto-save, and it leaves a staged single-cell preview COMPLETELY
   alone -- _pending_per_cell_alignment is neither read nor cleared.
4. A staged preview survives moving the FOV/cell spinboxes (only Accept
   and Reject clear it), but Accept/Reject are live only while that
   result is the one on screen -- otherwise the buttons offered a commit
   for something invisible.

Run: python tests/test_cell_alignment_all_fovs.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np                                   # noqa: E402
from unittest import mock                            # noqa: E402
from PyQt5 import QtWidgets                          # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
for _m in ('critical', 'warning', 'information', 'question'):
    mock.patch.object(QtWidgets.QMessageBox, _m,
                      return_value=QtWidgets.QMessageBox.Yes).start()

import windows.main_window as MW                     # noqa: E402
from windows.main_window import MainWindow           # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(('  ok ' if cond else 'FAIL ') + name + (f' -- {detail}' if detail and not cond else ''))


def cell(cid):
    return types.SimpleNamespace(id=cid, fov=0, reference_modality='DNA',
                                 reference_hybe='H1', matrices={},
                                 matrix_anchors={}, matrix_provenance={})


class FakeContainer:
    def __init__(self, by_fov):
        self.data = by_fov

    def get_cells(self, fov):
        return self.data.get(fov, [])


def main():
    mw = MainWindow()
    ap, ip = mw.ui.AlignmentPanel, mw.ui.IngestionPanel
    ip.FovListLineEdit.setText('1,2,3')

    # FOV 1 and 3 have cells; FOV 2 has none (must be skipped, not fatal)
    container = FakeContainer({1: [cell(1), cell(2)], 3: [cell(9)]})
    mw.cell_container_permanent = container
    mw.cell_container = None
    activated = []
    mw._activate_fov = lambda fov: activated.append(fov)
    mw._storage_path_for_modality = lambda m: '/store/DNA'
    mw._cell_alignment_passes = lambda modality, sp, fov: [{'fov': fov}]
    ap.build_cell_reference_hybe_fields(['DNA'])
    mw._all_analysis_storage_paths = lambda: ['/store/DNA']

    started = {}

    class FakeWorker:
        def __init__(self, jobs, **kw):
            started['jobs'] = jobs
            started['kw'] = kw
            self.progress = mock.MagicMock()
            self.finished_ok = mock.MagicMock()
            self.failed = mock.MagicMock()

        def start(self):
            started['started'] = True

    # a staged single-cell preview is sitting there, mid-review
    real, staged = cell(5), cell(5)
    mw._pending_per_cell_alignment = (real, staged)
    mw._pending_per_cell_alignment_fov = 2
    mw._pending_per_cell_alignment_params = {'reference_hybe': 'H1', 'modality': 'DNA'}

    with mock.patch.object(MW, 'CellAlignmentWorker', FakeWorker), \
         mock.patch.object(mw, '_confirm_batch_mode', return_value='append'):
        mw._run_cell_alignment_all_fovs()

    check('every FOV in the list was hydrated first', activated == [1, 2, 3], str(activated))
    check('the run started', started.get('started') is True)
    jobs = started.get('jobs', [])
    check('one job per FOV that has cells', [j[0] for j in jobs] == [1, 3], str([j[0] for j in jobs]))
    check('FOV with no cells is skipped, not fatal', 2 not in [j[0] for j in jobs])
    check('every cell of each FOV is included',
          [len(j[1]) for j in jobs] == [2, 1], str([len(j[1]) for j in jobs]))
    check('append reaches the worker', started.get('kw', {}).get('append') is True,
          str(started.get('kw')))

    check('a staged preview is left untouched by the batch',
          mw._pending_per_cell_alignment is not None
          and mw._pending_per_cell_alignment_fov == 2,
          'batch cleared it')

    # ---- persistence side: every FOV written, spots recast ----
    written, recast = [], []
    mw._recast_persisted_spots = lambda fov: recast.append(fov)
    mw._refresh_cell_fov_panels = lambda fov: None
    mw._cell_overlay_draw_args = lambda *a, **k: None      # no overlays in this check
    ap.CellOverlayAutoSaveThresholdSpinBox.setValue(500)   # nothing crosses it
    with mock.patch.object(MW.analysis_store, 'mirror_write_cells',
                           side_effect=lambda paths, fov, cont: written.append(fov)), \
         mock.patch.object(MW.analysis_store, 'write_global_params'):
        mw._on_cell_alignment_all_finished(
            [(1, container.get_cells(1)), (3, container.get_cells(3))],
            {1: container, 3: container}, '/store/DNA', 'H1', 'DNA', 'fiducial', 10)
    check('every FOV persisted', written == [1, 3], str(written))
    check('every FOV had its spots recast', recast == [1, 3], str(recast))
    check('staged preview STILL untouched after the finish handler',
          mw._pending_per_cell_alignment is not None, 'finish handler cleared it')

    # ---- staged-result lifecycle across spinbox moves ----
    mw._pending_per_cell_alignment = (real, staged)
    mw._pending_per_cell_alignment_fov = 3
    mw._cell_per_hybe_context = None
    ap.PerCellAcceptPushButton.setEnabled(True)
    ap.PerCellRejectPushButton.setEnabled(True)
    ap.CellFovSpinBox.setValue(3)
    ap.CellIdSpinBox.setValue(5)
    mw._refresh_cell_per_hybe_results(3, 5)
    on_screen = ap.PerCellAcceptPushButton.isEnabled()
    ap.CellFovSpinBox.setValue(1)
    mw._refresh_cell_per_hybe_results(1, 5)
    check('staged result SURVIVES an FOV change (only Accept/Reject clear it)',
          mw._pending_per_cell_alignment is not None
          and mw._pending_per_cell_alignment_fov == 3)
    check('Accept is live while its own result is on screen', on_screen)
    check('Accept is disabled while the staged result is off-screen',
          not ap.PerCellAcceptPushButton.isEnabled() and not ap.PerCellRejectPushButton.isEnabled())

    print()
    print(f'{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        raise SystemExit('FAILURES: ' + ', '.join(FAIL))
    print('ALL GOOD')


if __name__ == '__main__':
    main()
