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
    # cells for a non-resident FOV now come from a plain store READ, not
    # from the GUI's _activate_fov (measured: 84.8 s of GUI-thread
    # staging for 40 FOVs before any fit started)
    read_fovs = []
    MW.analysis_store.read_cells = (
        lambda sp, fov: (read_fovs.append(fov), (None, ''))[1])
    mw._storage_path_for_modality = lambda m: '/store/DNA'
    # ORDER OF OPERATIONS, pinned. _cell_alignment_passes puts a
    # fov_matrices OBJECT into each pass -- a snapshot, not a live view --
    # so the FOV-level backfill has to happen BEFORE the passes are built.
    # With it below, a freshly started app logged "loaded FOV-level
    # matrices for 49/49 FOV(s)" and then skipped 3600 of 3600 cells: only
    # the one FOV resident before the run had a populated snapshot, and
    # every other FOV's records were filtered down to nothing.
    order = []
    mw._fov_matrices_for = lambda sp, fov: None       # nothing resident
    mw._hybe_records_for_storage_path = lambda sp: [{'folder': 'H1'}]
    MW.alignment.read_same_modality_matrices = lambda sp, fov, recs: {('H1', 'DNA'): 1}
    mw._merge_fov_matrices = lambda fov, fm: order.append(('backfill', fov))
    mw._cell_alignment_passes = (
        lambda modality, sp, fov: (order.append(('passes', fov)), [{'fov': fov}])[1])
    ap.build_cell_reference_hybe_fields(['DNA'])
    mw._all_analysis_storage_paths = lambda: ['/store/DNA']

    started = {}

    class FakeWorker:
        def __init__(self, jobs, **kw):
            started['jobs'] = jobs
            started['kw'] = kw
            self.progress = mock.MagicMock()
            self.fov_done = mock.MagicMock()
            self.finished_ok = mock.MagicMock()
            self.failed = mock.MagicMock()
            # QThread's own signal, which the real worker inherits and the
            # run report connects to as its catch-all ending. The double
            # has to carry it for the same reason it carries the other
            # four: it stands in for a QThread, and a stand-in missing a
            # signal the caller connects fails on wiring rather than on
            # anything this test is about.
            self.finished = mock.MagicMock()

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

    check('preparation does NOT run the GUI activation path', activated == [],
          str(activated))
    kinds = [k for k, _fov in order]
    check('FOV-level matrices are backfilled BEFORE any pass is built',
          'backfill' in kinds and 'passes' in kinds
          and kinds.index('passes') > max(i for i, k in enumerate(kinds) if k == 'backfill'),
          str(order))
    check('every FOV without resident cells is READ from the store instead',
          read_fovs and read_fovs[0] == 2, str(read_fovs))
    # ...and then EVERY FOV with cells is read again, because this run is in
    # APPEND mode and append is no longer allowed to ask the in-memory cell
    # what is already done. A run mutates cells and only then writes the
    # FOV, so anything stopped in between leaves matrices that exist in the
    # process and nowhere else; trusting those skipped the work permanently.
    # These reads are mtime-cached (analysis_store._cached), and preparation
    # has just read the same files, so they cost a real read only where the
    # file changed.
    check('append also reads the persisted state of every FOV with cells',
          sorted(read_fovs[1:]) == [1, 3], str(read_fovs))
    check('and the worker is given it',
          isinstance(started.get('kw', {}).get('persisted'), dict),
          str(type(started.get('kw', {}).get('persisted'))))
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
    # ORDER matters: matrices must all be on disk BEFORE any overlay work
    # starts. Resolving one cell's overlay args costs ~55 ms on the real
    # store, so interleaving them made later FOVs' saves queue behind
    # earlier FOVs' rendering prep.
    events = []
    # the recast is BACKGROUNDED now (it re-reads/rewrites every spot
    # slice: ~1 s CPU + NAS round-trips per FOV, and doing it here on
    # the GUI thread is what made Spot Localization and Cell
    # Segmentation crawl during an all-FOVs run). The handler's job is
    # to HAND every persisted FOV over.
    mw._start_spot_recast = lambda fovs, label: [
        events.append(('recast', f)) for f in fovs]
    mw._recast_persisted_spots = lambda fov: events.append(('inline', fov))
    mw._refresh_cell_fov_panels = lambda fov: None
    mw._cell_overlay_draw_args = lambda c, fov, *a, **k: events.append(('resolve', fov)) or {'save_path': 'x'}
    ap.CellOverlayAutoSaveThresholdSpinBox.setValue(0)     # every cell wants one
    mw._cell_max_residual_shift = staticmethod(lambda c: 1.0)

    # ---- a FOV is SAVED THE MOMENT IT LANDS ----
    # Persisting only at the end meant stopping a whole-project run (a
    # day of fitting) threw away every completed FOV: the residuals were
    # in Results because the cells are mutated in memory, while the store
    # had never been written, so the status detail showed nothing.
    landed = []
    with mock.patch.object(MW.analysis_store, 'mirror_write_cells',
                           side_effect=lambda paths, fov, cont: landed.append(fov)):
        mw._on_cell_alignment_fov_done(1, container.get_cells(1), {1: container})
    check('a finished FOV is written immediately, not at the end of the run',
          landed == [1], str(landed))
    check('and is remembered as saved', 1 in mw._cell_align_saved_fovs,
          str(mw._cell_align_saved_fovs))
    check('its overlays are collected for the end of the run',
          len(mw._cell_align_pending_overlays) == 2,
          str(len(mw._cell_align_pending_overlays)))

    # a FOV that cannot be written must not take the rest of the run
    # down with it -- the remaining FOVs are still worth hours
    with mock.patch.object(MW.analysis_store, 'mirror_write_cells',
                           side_effect=OSError('store went away')):
        mw._on_cell_alignment_fov_done(3, container.get_cells(3), {3: container})
    check('a FOV that fails to save is reported, not raised',
          3 not in mw._cell_align_saved_fovs, 'marked saved despite failing')

    # the finish handler then writes only what never landed (FOV3 here,
    # whose save failed; in append mode, a FOV whose cells were all
    # skipped has no tasks and so never fires fov_done at all)
    with mock.patch.object(MW.analysis_store, 'mirror_write_cells',
                           side_effect=lambda paths, fov, cont: events.append(('save', fov))), \
         mock.patch.object(MW.analysis_store, 'write_global_params'):
        mw._on_cell_alignment_all_finished(
            [(1, container.get_cells(1)), (3, container.get_cells(3))],
            {1: container, 3: container}, '/store/DNA', 'H1', 'DNA', 'fiducial', 10)
    check('the finish handler does not rewrite a FOV already saved',
          [f for k, f in events if k == 'save'] == [3],
          str([f for k, f in events if k == 'save']))
    check('overlays from both paths reach the queue', mw._overlay_total == 3,
          str(mw._overlay_total))

    # ---- and the whole-run path still holds when nothing landed early ----
    events.clear()
    mw._cell_align_saved_fovs = set()
    mw._cell_align_pending_overlays = []
    mw._overlay_pending, mw._overlay_ready, mw._overlay_total = [], [], 0
    with mock.patch.object(MW.analysis_store, 'mirror_write_cells',
                           side_effect=lambda paths, fov, cont: events.append(('save', fov))), \
         mock.patch.object(MW.analysis_store, 'write_global_params'):
        mw._on_cell_alignment_all_finished(
            [(1, container.get_cells(1)), (3, container.get_cells(3))],
            {1: container, 3: container}, '/store/DNA', 'H1', 'DNA', 'fiducial', 10)
    written = [f for k, f in events if k == 'save']
    recast = [f for k, f in events if k == 'recast']
    check('every FOV persisted', written == [1, 3], str(written))
    check('every FOV is handed to the recast', recast == [1, 3], str(recast))
    check('and NOT recast inline on the GUI thread',
          [f for k, f in events if k == 'inline'] == [])
    kinds = [k for k, _f in events]
    check('ALL saves happen before ANY overlay work',
          'resolve' not in kinds or kinds.index('resolve') > max(
              i for i, k in enumerate(kinds) if k == 'save'),
          str(kinds))
    check('overlays are queued, not resolved inline',
          len(mw._overlay_pending) + len(mw._overlay_ready) > 0 or 'resolve' not in kinds,
          f'pending={len(mw._overlay_pending)} ready={len(mw._overlay_ready)}')
    check('queued overlay count matches the cells that wanted one',
          mw._overlay_total == 3, str(mw._overlay_total))

    # the drip resolves a few per tick and never all at once
    before = len(mw._overlay_pending)
    with mock.patch.object(mw, '_start_cell_overlay_save_worker') as starter:
        mw._resolve_overlay_chunk()
        after = len(mw._overlay_pending)
        check('a tick resolves at most one chunk',
              before - after <= mw.OVERLAY_RESOLVE_CHUNK, f'{before} -> {after}')
        check('a resolved batch is handed to the render pool',
              starter.called, 'pool never started')

    # ---- a FOV whose ingestion is unfinished is excluded ENTIRELY ----
    #
    # Running cell alignment during ingestion used to record an IDENTITY
    # residual for every not-yet-arrived hybe, with provenance reading
    # "cell does not overlap this hybe's frame" -- the same message the
    # genuinely-off-frame case gets, because _cell_native_crop returns a
    # bare None for both. Append then treats the key's existence as done,
    # so the hybe is permanently marked aligned and no later run revisits
    # it; and since the fabricated entry has H2 = identity it scores 0.0 px
    # on _cell_max_residual_shift, the lowest possible, so it is also the
    # one entry guaranteed never to be flagged for review.
    started.clear()
    notices = []
    ready_by_fov = {1: {'H0', 'H1'}, 3: {'H0'}}        # FOV3 is missing H1
    with mock.patch.object(MW, 'CellAlignmentWorker', FakeWorker), \
         mock.patch.object(mw, '_confirm_batch_mode', return_value='overwrite'), \
         mock.patch.object(mw, '_ready_hybes',
                           side_effect=lambda mod, fov: ready_by_fov.get(fov, set())), \
         mock.patch.object(mw, '_cell_alignment_passes',
                           side_effect=lambda mod, sp, fov: [
                               {'modality': 'DNA', 'storage_path': sp,
                                'hybe_records': [{'folder': 'H0'}, {'folder': 'H1'}],
                                'fov_matrices': {}, 'reference_hybe': 'H0',
                                'cellref_fov_matrices': {}}]), \
         mock.patch.object(QtWidgets.QMessageBox, 'information',
                           side_effect=lambda *a, **k: notices.append(a[2] if len(a) > 2 else '')), \
         mock.patch.object(QtWidgets.QMessageBox, 'question',
                           return_value=QtWidgets.QMessageBox.Yes):
        mw._run_cell_alignment_all_fovs()
    jobs = started.get('jobs', [])
    check('a FOV with an un-ingested hybe never reaches the worker',
          [j[0] for j in jobs] == [1], str([j[0] for j in jobs]))
    check('and the user is told which FOV and why',
          notices and 'FOV003' in notices[0] and 'not ingested' in notices[0],
          str(notices[:1]))
    check('the complete FOV still runs', bool(jobs))

    # every FOV incomplete -> nothing starts at all
    started.clear()
    warned = []
    with mock.patch.object(MW, 'CellAlignmentWorker', FakeWorker), \
         mock.patch.object(mw, '_confirm_batch_mode', return_value='overwrite'), \
         mock.patch.object(mw, '_ready_hybes', side_effect=lambda mod, fov: set()), \
         mock.patch.object(mw, '_cell_alignment_passes',
                           side_effect=lambda mod, sp, fov: [
                               {'modality': 'DNA', 'storage_path': sp,
                                'hybe_records': [{'folder': 'H0'}],
                                'fov_matrices': {}, 'reference_hybe': 'H0',
                                'cellref_fov_matrices': {}}]), \
         mock.patch.object(QtWidgets.QMessageBox, 'information',
                           side_effect=lambda *a, **k: None), \
         mock.patch.object(QtWidgets.QMessageBox, 'warning',
                           side_effect=lambda *a, **k: warned.append(a[2] if len(a) > 2 else '')), \
         mock.patch.object(QtWidgets.QMessageBox, 'question',
                           return_value=QtWidgets.QMessageBox.Yes):
        mw._run_cell_alignment_all_fovs()
    check('with every FOV incomplete the run never starts',
          not started.get('started'), str(started.get('started')))
    check('and it says so rather than reporting an empty success',
          warned and 'unfinished ingestion' in warned[-1], str(warned[-1:]))

    # ---- FOV-level matrices are read from the STORE, not just the session ----
    #
    # The confirmed cause of "Align All Cells in All FOVs only did one FOV",
    # reported twice. _fov_matrices_for reads self.fov_matrices, which
    # _activate_fov fills ONE FOV at a time; the bulk backfill in
    # _refresh_params_from_vlinks skips the store's own modality, so on a
    # single-modality project it never runs. Every unvisited FOV then ships
    # hybe_records filtered to [], and append reports its cells as "already
    # fully aligned" while overwrite raises -- both silently.
    resident = set()                          # what self.fov_matrices holds
    read_for = []

    def run_once():
        read_for.clear()

        def _read(sp, fov, records):
            read_for.append(fov)
            return {('H0', 'DNA'): np.eye(3)}

        with mock.patch.object(MW, 'CellAlignmentWorker', FakeWorker), \
             mock.patch.object(mw, '_confirm_batch_mode', return_value='append'), \
             mock.patch.object(mw, '_fov_matrices_for',
                               side_effect=lambda sp, fov: {'H0': 1} if fov in resident else {}), \
             mock.patch.object(mw, '_merge_fov_matrices',
                               side_effect=lambda fov, m: resident.add(fov)), \
             mock.patch.object(MW.alignment, 'read_same_modality_matrices',
                               side_effect=_read), \
             mock.patch.object(mw, '_hybe_records_for_storage_path',
                               return_value=[{'folder': 'H0'}]):
            mw._run_cell_alignment_all_fovs()

    run_once()
    check('matrices are read from the STORE for every FOV the session lacks',
          sorted(read_for) == [1, 3], str(read_for))
    check('and those FOVs are then resident', resident == {1, 3}, str(resident))

    # A FOV already resident is NOT re-read: the session copy can hold
    # in-session edits that the disk copy does not.
    run_once()
    check('a FOV already resident is not re-read from the store',
          read_for == [], str(read_for))

    # ---- an append run that fits NOTHING must do nothing downstream ----
    #
    # Observed twice in one morning on a fully-aligned 50-FOV project:
    #   append: 3600 cell(s) already fully aligned -- skipped
    #   ... | 0 passes in 0.3 s
    #   Cell alignment complete -- 3600 cell(s) across 50 FOV(s) aligned and SAVED.
    # followed by a 50-FOV spot recast and ~700 overlay PNGs, seven minutes
    # of background rendering for work that never happened -- which is also
    # the load that was starving the GUI's own redraws. The handler had no
    # way to tell "every FOV was IN the run" from "every FOV was CHANGED by
    # it"; the worker now publishes fitted_fovs, and this pins that it is
    # honoured. The user's report was "only one FOV was calculated": in
    # append mode that is the same failure seen from the other side.
    events.clear()
    mw._cell_align_saved_fovs = set()
    mw._cell_align_pending_overlays = []
    mw._overlay_pending, mw._overlay_ready, mw._overlay_total = [], [], 0
    notices = []
    mw._cell_alignment_worker = types.SimpleNamespace(
        fitted_fovs=set(), n_skipped=3600, pool_shape='serial', n_cells_fitted=0)
    with mock.patch.object(MW.analysis_store, 'mirror_write_cells',
                           side_effect=lambda paths, fov, cont: events.append(('save', fov))), \
         mock.patch.object(MW.analysis_store, 'write_global_params'), \
         mock.patch.object(mw, '_notify_complete',
                           side_effect=lambda t, m, **k: notices.append(m)):
        mw._on_cell_alignment_all_finished(
            [(1, container.get_cells(1)), (3, container.get_cells(3))],
            {1: container, 3: container}, '/store/DNA', 'H1', 'DNA', 'fiducial', 10)
    check('a no-op append writes no cells',
          [f for k, f in events if k == 'save'] == [],
          str([f for k, f in events if k == 'save']))
    check('a no-op append queues no spot recast',
          [f for k, f in events if k == 'recast'] == [],
          str([f for k, f in events if k == 'recast']))
    check('a no-op append queues no overlays', mw._overlay_total == 0,
          str(mw._overlay_total))
    check('and it says nothing was refitted, not that it aligned',
          notices and 'Nothing was refitted' in notices[0]
          and 'aligned and SAVED' not in notices[0],
          str(notices[:1]))
    check('it names the way out (Overwrite)',
          notices and 'Overwrite' in notices[0], str(notices[:1]))

    # a PARTIAL append still processes exactly the FOVs that were fitted
    events.clear()
    mw._cell_align_saved_fovs = set()
    mw._cell_align_pending_overlays = []
    mw._overlay_pending, mw._overlay_ready, mw._overlay_total = [], [], 0
    mw._cell_alignment_worker = types.SimpleNamespace(
        fitted_fovs={3}, n_skipped=12, pool_shape='across-cells x4', n_cells_fitted=1)
    with mock.patch.object(MW.analysis_store, 'mirror_write_cells',
                           side_effect=lambda paths, fov, cont: events.append(('save', fov))), \
         mock.patch.object(MW.analysis_store, 'write_global_params'):
        mw._on_cell_alignment_all_finished(
            [(1, container.get_cells(1)), (3, container.get_cells(3))],
            {1: container, 3: container}, '/store/DNA', 'H1', 'DNA', 'fiducial', 10)
    check('a partial append touches only the FOV it fitted',
          [f for k, f in events if k == 'save'] == [3],
          str([f for k, f in events if k == 'save']))
    check('and recasts only that FOV',
          [f for k, f in events if k == 'recast'] == [3],
          str([f for k, f in events if k == 'recast']))
    mw._cell_alignment_worker = None

    # quit reporting
    mw._overlay_pending = [(cell(1), 1)] * 5
    mw._overlay_ready = []
    q, rendering = mw._overlays_in_flight()
    check('unfinished overlays are visible to the quit path', q == 5 and not rendering,
          f'{q} {rendering}')
    mw._overlay_pending = []
    q, _r = mw._overlays_in_flight()
    check('nothing in flight when the queue is empty', q == 0, str(q))
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
