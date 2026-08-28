"""
Append-mode (per-(FOV, hybe) delta, manual trigger) functional check:

1. vlinks_store.aligned_hybes: persisted-matrix set, distinguishable from
   the identity-defaulting reader; empty store -> frozenset().
2. fov_ready_hybes registry: fed per-(fov, hybe) by task_done (including
   the second FOV of an already-active hybe), seeded by real scans;
   _ready_hybes answers from the registry mid-run (zero disk) and from a
   scan when idle.
3. AlignmentWorker per-FOV record subsets drive align_same_modality with
   exactly each FOV's own delta.
4. _run_fov_alignment_all append: composes records_by_fov = ready minus
   persisted, skips FOVs whose reference isn't ready, early-outs when
   nothing is missing.
5. CellAlignmentWorker append: per-(cell, hybe) filter against
   cell.matrices; fully-aligned cells skipped.
6. build_chromatin_trace_allele append: merge (no reset) vs default full
   replace; requested hybes' stale entries cleared for re-derive.
7. ChromatinTracingWorker append: fits EVERY checked hybe for every allele
   it is handed, always with append=False. Which ALLELES run is decided
   upstream by membership (AlleleContainer.has_traced on the permanent
   tier), not by which hybes an allele happens to be missing.

   This section asserted the opposite until 2026-08-28. The old rule --
   append fits the hybes not yet traced on an allele -- is what allowed an
   allele half-traced by one engine to be finished by another, leaving one
   polymer built from two estimators with nothing on disk recording it.
   Section 6 stays: the ENGINE still supports merge, only the caller
   stopped asking for it.
"""
import json
import os
import sys
import tempfile
import types

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

FAILS = []


def check(name, cond, detail=''):
    print(('  PASS  ' if cond else '  FAIL  ') + name + (f'  ({detail})' if detail else ''))
    if not cond:
        FAILS.append(name)


def rec(folder, n=1):
    return {'folder': folder, 'readout_id': n, 'datatype': 'x', 'hybe_num': n,
            'channels': [635], 'fiducial_channel': 405, 'channel_layout': '', 'total_frames': 2,
            'readout_name': None}


def main():
    from PyQt5 import QtWidgets
    from unittest import mock
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from windows.main_window import (MainWindow, IngestionWorker, AlignmentWorker,
                                     CellAlignmentWorker, ChromatinTracingWorker)
    from codelab_pipeline.io import analysis_store as vlinks_store, paths
    from codelab_pipeline.alignment import chain, frames
    from codelab_pipeline.localization import localization

    tmp = tempfile.mkdtemp(prefix='append_')
    root = os.path.join(tmp, 'proj')
    for m in ('DNA', 'RNA'):
        os.makedirs(os.path.join(root, m))
    os.makedirs(os.path.join(root, 'analysis'))
    with open(os.path.join(root, 'manifest.json'), 'w') as f:
        json.dump({'layout_version': 2,
                   'modalities': {m: {'layout_path': '', 'dax_directory': ''} for m in ('DNA', 'RNA')}}, f)
    dna_sp = os.path.join(root, 'DNA')

    # -- 1. aligned_hybes ----------------------------------------------------
    check('empty store -> frozenset()', vlinks_store.aligned_hybes(dna_sp, 1) == frozenset())
    vlinks_store.write_same_modality_matrices(dna_sp, 1, {'H1': np.eye(3), 'H2': np.eye(3) + 0.0}, 'H1')
    got = vlinks_store.aligned_hybes(dna_sp, 1)
    check('persisted matrices reported by name', got == {'H1', 'H2'}, str(got))
    check('other FOV unaffected', vlinks_store.aligned_hybes(dna_sp, 2) == frozenset())

    # -- 2. readiness registry + _ready_hybes --------------------------------
    w = MainWindow()
    ip = w.ui.IngestionPanel
    ip.modality_names = ['DNA', 'RNA']
    ip.modality_data = {
        'DNA': {'layout_path': '', 'dax_directory': '', 'storage_path': dna_sp, 'active_hybe_list': []},
        'RNA': {'layout_path': '', 'dax_directory': '', 'storage_path': os.path.join(root, 'RNA'),
                'active_hybe_list': []},
    }
    records = [rec('H1', 1), rec('H2', 2), rec('H3', 3)]
    dna_job = {'fov_list': [1, 2], 'hybe_records': records, 'dax_directory': '/x',
               'storage_path': dna_sp, 'modality': 'DNA'}
    worker = IngestionWorker([dna_job], max_workers=1)
    w._repopulate_hybe_choice_widgets = lambda: None
    w._on_ingestion_task_ready(worker, 1, 'H1', 'DNA', True)
    w._on_ingestion_task_ready(worker, 2, 'H1', 'DNA', True)   # dedup path for active_hybe_list...
    w._on_ingestion_task_ready(worker, 1, 'H2', 'DNA', True)
    w._on_ingestion_task_ready(worker, 1, 'H3', 'DNA', False)  # failure never marks ready
    check('registry records every (fov, hybe), incl. the deduped second FOV',
          w.fov_ready_hybes[('DNA', 1)] == {'H1', 'H2'} and w.fov_ready_hybes[('DNA', 2)] == {'H1'},
          str(dict(w.fov_ready_hybes)))

    scan_spy_calls = []
    def scan_spy(sp, fov, recs):
        scan_spy_calls.append(fov)
        return (['H1', 'H2', 'H3'], [], [])
    w._ingested_hybes_for_fov = scan_spy
    w.hybe_records_by_modality = {'DNA': records}
    w._active_ingestions.append(worker)   # mid-run
    # v2 store: the mid-run scan is NOT skipped any more. Readiness here
    # is one directory listing per (modality, FOV) -- 7 ms for the whole
    # real store -- and skipping it was the confirmed cause of an
    # already-ingested FOV looking un-analysed while ingestion ran (see
    # tests/test_analysis_during_ingestion.py). The registry-only answer
    # survives for v1, whose readiness costs one HDF5 open per hybe.
    check('mid-run on v2: _ready_hybes scans disk (cheap) and re-seeds',
          w._ready_hybes('DNA', 1) == {'H1', 'H2', 'H3'} and scan_spy_calls == [1]
          and w.fov_ready_hybes[('DNA', 1)] == {'H1', 'H2', 'H3'})
    w._active_ingestions.clear()          # idle
    check('idle: _ready_hybes scans disk and re-seeds',
          w._ready_hybes('DNA', 1) == {'H1', 'H2', 'H3'} and scan_spy_calls == [1, 1]
          and w.fov_ready_hybes[('DNA', 1)] == {'H1', 'H2', 'H3'})

    # -- 3. AlignmentWorker per-FOV subsets ----------------------------------
    # specs are per (FOV, modality) now -- one pool per FOV covering every
    # modality (see tests/test_fov_alignment_multimodality.py).
    calls = []
    def fake_align(fov, specs, **kw):
        for sp in specs:
            calls.append((fov, tuple(r['folder'] for r in sp['hybe_records'])))
            if kw.get('progress'):
                kw['progress'](len(sp['hybe_records']), len(sp['hybe_records']), fov, 'x')
        return {sp['modality']: {r['folder']: np.eye(3) for r in sp['hybe_records']}
                for sp in specs}

    def _spec(recs):
        return [{'modality': 'DNA', 'storage_path': dna_sp,
                 'hybe_records': recs, 'reference_hybe': 'H1'}]
    aw = AlignmentWorker([1, 2], {1: _spec([records[2]]), 2: _spec(records[:2])}, write=False)
    with mock.patch.object(chain, 'align_fov_all_modalities', side_effect=fake_align):
        aw.run()   # synchronous call, no thread start
    check('worker hands each FOV exactly its own subset',
          calls == [(1, ('H3',)), (2, ('H1', 'H2'))], str(calls))

    # -- 4. _run_fov_alignment_all append composition -------------------------
    ap = w.ui.AlignmentPanel
    ap.same_modality_references = lambda: {'DNA': 'H1'}
    ip.modality_data['DNA']['active_hybe_list'] = records
    ip.FovListLineEdit.setText('1,2,3')
    w._ready_hybes = lambda m, fov: {1: {'H1', 'H2', 'H3'}, 2: {'H1', 'H2'}, 3: {'H2'}}[fov]
    captured = {}

    class FakeAW(AlignmentWorker):
        def start(self):
            captured['specs_by_fov'] = self.specs_by_fov
            captured['fov_list'] = self.fov_list
    with mock.patch.object(w, '_confirm_batch_mode', return_value='append'), \
         mock.patch('windows.main_window.AlignmentWorker', FakeAW):
        w._run_fov_alignment_all()
    # persisted for FOV1: H1, H2 -> FOV1 delta = H3; FOV2 nothing persisted,
    # ready H1+H2 -> both; FOV3 reference H1 not ready -> skipped whole.
    rbf = {fov: tuple(r['folder'] for sp in specs for r in sp['hybe_records'])
           for fov, specs in captured.get('specs_by_fov', {}).items()}
    check('append delta: ready minus persisted, per FOV',
          rbf == {1: ('H3',), 2: ('H1', 'H2')}, str(rbf))
    check('FOV without a ready reference skipped whole', 3 not in rbf and captured['fov_list'] == [1, 2])
    ap.RunAllFovAlignmentPushButton.setEnabled(True)

    # nothing-to-do early-out
    w._ready_hybes = lambda m, fov: {'H1', 'H2'}
    infos = {}
    with mock.patch.object(w, '_confirm_batch_mode', return_value='append'), \
         mock.patch.object(QtWidgets.QMessageBox, 'information',
                           side_effect=lambda *a, **k: infos.setdefault('msg', a[2] if len(a) > 2 else '')):
        vlinks_store.write_same_modality_matrices(dna_sp, 2, {'H1': np.eye(3), 'H2': np.eye(3)}, 'H1')
        vlinks_store.write_same_modality_matrices(dna_sp, 3, {'H1': np.eye(3), 'H2': np.eye(3)}, 'H1')
        w._run_fov_alignment_all()
    check('fully-persisted project -> "nothing to append" early-out',
          'Nothing to append' in infos.get('msg', ''), infos.get('msg', ''))

    # -- 5. CellAlignmentWorker append ---------------------------------------
    fm = frames.FrameMatrices(modality='DNA')
    for h in ('H1', 'H2', 'H3'):
        fm[(h, 'DNA')] = np.eye(3)
    cell_done = types.SimpleNamespace(id=1, reference_modality='DNA', reference_hybe='H1',
                                      matrices={('H1', 'DNA'): 1, ('H2', 'DNA'): 1, ('H3', 'DNA'): 1},
                                      matrix_anchors={}, matrix_provenance={})
    cell_part = types.SimpleNamespace(id=2, reference_modality='DNA', reference_hybe='H1',
                                      matrices={('H1', 'DNA'): 1}, matrix_anchors={}, matrix_provenance={})
    passes = [{'modality': 'DNA', 'storage_path': dna_sp, 'hybe_records': records,
               'fov_matrices': fm, 'reference_hybe': 'H1',
               'cellref_fov_matrices': {'DNA': fm}}]
    caw = CellAlignmentWorker([(7, [cell_done, cell_part], passes)], workers=1, append=True)
    fitted = []
    def fake_cell_align(cell, sp, fov, hybe_records, fovm, **kw):
        fitted.append((cell.id, tuple(r['folder'] for r in hybe_records)))
    with mock.patch.object(chain, 'compute_cell_alignment', side_effect=fake_cell_align):
        caw.run()
    # The reference hybe (H1) ALWAYS rides along even though cell 2 already
    # has its matrix: compute_cell_alignment raises without its anchor
    # record -- filtering it out was the real "append stuck" bug. The
    # delta is the NON-reference records (H2, H3); the complete cell's
    # pass has only the reference left and is pruned entirely.
    check('append fits only the missing (cell, hybe) pairs; complete cell skipped',
          fitted == [(2, ('H1', 'H2', 'H3'))], str(fitted))

    # -- 6. build_chromatin_trace_allele append semantics ---------------------
    allele = types.SimpleNamespace(coordinate=(5.0, 6.0, 7.0),
                                   fiducial_trace={'R': (1, 2, 3), 'A': (1, 2, 3)},
                                   polymer={'A': ['kept']}, rejected_hybes={'B': 'why'})
    localization.build_chromatin_trace_allele(allele, [], 'R', {}, {}, '/nowhere', 1, 'DNA', None, {},
                                              append=True)
    check('append=True with no hybes leaves everything untouched',
          allele.polymer == {'A': ['kept']} and allele.rejected_hybes == {'B': 'why'}
          and allele.fiducial_trace.get('R') == (1, 2, 3))
    localization.build_chromatin_trace_allele(allele, [], 'R', {}, {}, '/nowhere', 1, 'DNA', None, {})
    check('default (overwrite) still resets the three dicts',
          allele.polymer == {} and allele.rejected_hybes == {} and allele.fiducial_trace == {})

    # -- 7. ChromatinTracingWorker: append is a MEMBERSHIP question ----------
    #
    # This section used to assert the opposite, and the behaviour it
    # defended is the one that had to go: append filtered HYBES per allele,
    # so an allele half-traced by one engine could have the rest of its
    # hybes filled in by another, leaving ONE polymer built from two
    # estimators with nothing on disk saying so.
    #
    # Which ALLELES run is now decided before the worker starts, by
    # AlleleContainer.has_traced on the permanent tier (see
    # MainWindow._run_chromatin_tracing_fit_all). The worker's remaining
    # job is simple and is what these checks pin: fit EVERY checked hybe,
    # for every allele it is handed, with append=False.
    from codelab_pipeline.localization import tracing_v2

    def _spy(traced):
        def record(engine, allele, hybes, *a, **k):
            traced.append((allele.id, tuple(hybes), k.get('append')))
            return allele, None
        return record

    # a1 is partly traced, a2 is fully traced -- under the OLD rule a1
    # would have been fitted for H3 only and a2 skipped entirely.
    a1 = types.SimpleNamespace(id=1, cell=-1, coordinate=(0, 0, 0),
                               polymer={'H1': 1}, rejected_hybes={'H2': 'r'}, fiducial_trace={})
    a2 = types.SimpleNamespace(id=2, cell=-1, coordinate=(0, 0, 0),
                               polymer={'H1': 1, 'H3': 1}, rejected_hybes={'H2': 'r'}, fiducial_trace={})
    # The worker no longer ACCEPTS ready_hybes_by_fov. A parameter it does
    # not read would imply a per-hybe filter that no longer exists, so its
    # absence is asserted rather than assumed.
    import inspect
    sig = inspect.signature(ChromatinTracingWorker.__init__)
    check('the worker no longer takes ready_hybes_by_fov -- readiness is a '
          'FOV-level decision made on the GUI thread',
          'ready_hybes_by_fov' not in sig.parameters, str(list(sig.parameters)))

    ctw = ChromatinTracingWorker([(dna_sp, 9, [a1, a2])], ['H1', 'H2', 'H3'], 'H1', {}, {}, 'DNA',
                                 {}, lambda fov, cid: None, 5.0, 8, 15, {}, {},
                                 workers=1, append=True)
    traced = []
    with mock.patch.object(tracing_v2, 'trace_allele', side_effect=_spy(traced)):
        ctw.run()
    check('every allele handed to the worker is fitted for EVERY checked hybe',
          traced == [(1, ('H1', 'H2', 'H3'), False),
                     (2, ('H1', 'H2', 'H3'), False)], str(traced))
    check('and always with append=False -- a selected allele has no committed '
          'trace to merge into, so a full re-derivation is correct',
          all(entry[2] is False for entry in traced), str(traced))

    # An allele with NOTHING traced gets the full hybe list too -- there is
    # no path left by which the worker fits a subset.
    a3 = types.SimpleNamespace(id=3, cell=-1, coordinate=(0, 0, 0),
                               polymer={}, rejected_hybes={}, fiducial_trace={})
    ctw2 = ChromatinTracingWorker([(dna_sp, 9, [a3])], ['H1', 'H2', 'H3'], 'H1', {}, {}, 'DNA',
                                  {}, lambda fov, cid: None, 5.0, 8, 15, {}, {},
                                  workers=1, append=True)
    traced.clear()
    with mock.patch.object(tracing_v2, 'trace_allele', side_effect=_spy(traced)):
        ctw2.run()
    check('an untraced allele is also fitted for every checked hybe',
          traced == [(3, ('H1', 'H2', 'H3'), False)], str(traced))
    # The source itself: no surviving expression that narrows the hybe list.
    import inspect as _inspect
    src = _inspect.getsource(ChromatinTracingWorker.run)
    check('the worker body contains no per-hybe append filter at all',
          'not in traced' not in src and 'set(allele.polymer' not in src
          and 'ready_hybes_by_fov' not in src,
          'a hybe-narrowing expression survives in ChromatinTracingWorker.run')

    # -- 7b. the membership rule itself --------------------------------------
    from codelab_pipeline.models.allele import AnAllele
    from codelab_pipeline.models.allele_container import AlleleContainer
    key = (dna_sp, 9)
    perm = AlleleContainer()
    for aid, traced_flag in ((1, True), (2, False)):
        a = AnAllele()
        a.set_metadata(id=aid, fov=9, cell=-1, anchor_uid=aid, anchor_hybe='H1',
                       anchor_channel=555, coordinate=(0.0, 0.0, 0.0),
                       raw_coordinate=(0.0, 0.0, 0.0))
        if traced_flag:
            a.polymer = {'H1': [(1.0, 2.0, 3.0, 9.0)]}
        perm.add(key, a)
    check('a committed allele WITH a trace is skipped by append',
          perm.has_traced(key, 1))
    check('a committed allele with NO trace is still fitted by append -- '
          'Add then Save must not become a trap',
          not perm.has_traced(key, 2))
    check('an allele that was never saved is fitted by append',
          not perm.has_traced(key, 3))

    print(f"\n{'ALL GOOD' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
