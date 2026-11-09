"""
Failing to draw a picture must never cost the data.

THE LOSS. MP58-Q, 2026-11-09: FOV025 held 18,584 spots and every one of
them carried cell = -1 in every slice but a single Hyb_113. Re-running
spot detection and saving again reproduced it exactly. The cells were
fine -- 94 of them, on disk, projecting correctly; only the `cell` field
was wrong, and re-assigning from the store put 97.7% of the spots back
inside cells in 13 seconds.

THE PATH. _try_show_existing_cells does two things: it STAGES this FOV's
persisted cells into the transient container, and it DRAWS them. The
staging used to sit BELOW the reference-MIP read, so:

    cells loaded into cell_container_permanent      <- the data arrives
    ...
    reference_image = fiducial_channel_mip(...)     <- display resource
    if reference_image is None:
        return False                                <- and the data is lost
    ...
    self.cell_container.sync_from(permanent, fov)   <- never reached

_reassign_fov_spots reads the TRANSIENT container, and by explicit spec
treats "no cells" as "every spot is unassigned" (emptying a FOV is a
legitimate thing to save). So a display failure silently turned every
save of that FOV into a wholesale unassignment.

WHY THE MIP READ FAILED, MEASURED. Every MP58-Q FOV's cells carry the
reference frame (Hyb_113, RNA), and Hyb_113 exists ONLY in the RNA
store: mip_meta under .../DNA returns None for FOV001 and FOV025 alike,
under .../RNA both read fine. The old lookup ended `or storage_path`, so
one missing Ingestion-tab entry for RNA sent every lookup to the DNA
store and failed for EVERY FOV -- which is why four FOVs were reported
at once and the per-FOV variation was only which ones got saved in that
state.

WHAT IS PINNED HERE. Order (stage before draw), the verdict a failed
draw returns (True -- staged, just not shown; False would leave the FOV
unlatched and re-sync it from permanent forever, the very clobber
_activated_fovs exists to prevent), the removal of the wrong-store
fallback, and the second line of defence in _reassign_fov_spots.

The last test re-measures the premise against the real store and says so
explicitly when it cannot reach it, rather than passing quietly.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_cell_staging_survives_display.py
      CODELAB_STAGE_STORE_ROOT=<project_root>   (optional; default below)
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

PASS, FAIL = [], []
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STORE_ROOT = os.environ.get(
    'CODELAB_STAGE_STORE_ROOT',
    r'Q:\Seonghyeok\ORCA3\20251130_MP58_RNAandDNA_\Project_Root')
STORE_FOVS = (1, 25)


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}'
          + (f'   [{detail}]' if detail else ''))


def src(*parts):
    with open(os.path.join(HERE, *parts), encoding='utf-8') as f:
        return f.read()


def method(text, name):
    """The body of one method: from its `def` to the next one at the same
    indent. Slicing beats importing here -- these are MainWindow methods
    and the whole point is to check them without standing up the app."""
    i = text.index(f'    def {name}(')
    j = text.find('\n    def ', i + 1)
    return text[i:j if j != -1 else len(text)]


def test_stage_before_draw():
    print('\n_try_show_existing_cells: the data arrives before the picture')
    body = method(src('windows', 'main_window.py'), '_try_show_existing_cells')

    i_stage = body.find('sync_from(self.cell_container_permanent')
    i_draw = body.find('fiducial_channel_mip(')
    check('it still stages and still draws', i_stage != -1 and i_draw != -1,
          f'stage@{i_stage} draw@{i_draw}')
    check('STAGING COMES FIRST -- a failed draw cannot cost the cells',
          -1 < i_stage < i_draw,
          'staging sits after the MIP read again' if i_stage > i_draw else '')

    i_disp = body.find('self.cell_displayer.set_data(')
    check('and the displayer is still fed after both', i_disp > i_stage)

    # The branch a missing MIP takes.
    m = re.search(r'if reference_image is None:(.*?)\n\n', body, re.S)
    check('the missing-MIP branch is still there', m is not None)
    if m:
        branch = m.group(1)
        check('it returns True -- STAGED, just not displayed',
              'return True' in branch and 'return False' not in branch,
              'returns False, so _activated_fovs never latches'
              if 'return False' in branch else '')


def test_no_fallback_into_the_wrong_store():
    print('\nthe reference frame is resolved in ITS OWN modality, or not at all')
    body = method(src('windows', 'main_window.py'), '_try_show_existing_cells')

    fallback_back = ".get('storage_path') or storage_path)" in body
    check("the wrong-store fallback is gone", not fallback_back,
          'the `or storage_path` fallback is back' if fallback_back else '')
    check('a NAMED modality with no configured path refuses to guess',
          'ref_modality' in body and 'has no storage path configured' in body)
    # An unnamed modality has nothing to resolve: the current path is the
    # only answer there is, so that fallback must survive.
    check('an UNNAMED modality still falls back to the current path',
          'cell_storage_path = storage_path' in body)


def test_reassign_has_a_second_line():
    print('\n_reassign_fov_spots: "no cells" means two different things')
    body = method(src('windows', 'main_window.py'), '_reassign_fov_spots')

    i_guard = body.find('_transient_staged_fovs')
    i_read = body.find('read_cells(')
    i_assign = body.find('assignment.assign_spots(')
    check('an unedited FOV is told apart from a deliberately emptied one',
          i_guard != -1, '' if i_guard != -1 else 'no _transient_staged_fovs test')
    check('and re-reads its cells from the store', i_read != -1)
    check('before assignment runs, not after',
          -1 < i_guard < i_read < i_assign,
          f'guard@{i_guard} read@{i_read} assign@{i_assign}')
    # The deliberate case must still work: emptying a FOV is saveable.
    check('a DELIBERATELY emptied FOV still unassigns (spec, not a bug)',
          'not cells and int(fov) not in self._transient_staged_fovs' in body)


def test_the_premise_on_the_real_store():
    print('\nthe measurement this fix rests on (real store)')
    from codelab_pipeline.io import analysis_store as A

    dna, rna = os.path.join(STORE_ROOT, 'DNA'), os.path.join(STORE_ROOT, 'RNA')
    if not os.path.isdir(dna):
        print(f'  SKIPPED -- {STORE_ROOT} is not reachable from this machine, so '
              f'the premise below is UNVALIDATED here. It was measured on '
              f'2026-11-09; set CODELAB_STAGE_STORE_ROOT to re-measure.')
        return

    for fov in STORE_FOVS:
        dicts, _ = A.read_cells(dna, fov)
        if not dicts:
            print(f'  SKIPPED FOV{fov:03d} -- no cells in this store')
            continue
        ref_hybe = dicts[0].get('reference_hybe')
        ref_mod = dicts[0].get('reference_modality')
        check(f'FOV{fov:03d} cells name their own modality',
              bool(ref_mod), f'{ref_hybe} / {ref_mod}')
        right = A.fiducial_channel_mip(
            os.path.join(STORE_ROOT, ref_mod), fov, ref_hybe)
        wrong_root = dna if ref_mod != 'DNA' else rna
        wrong = A.fiducial_channel_mip(wrong_root, fov, ref_hybe)
        check(f'FOV{fov:03d} {ref_hybe} READS in {ref_mod}', right is not None)
        check(f'FOV{fov:03d} {ref_hybe} is MISSING in the other store -- '
              f'which is what the old `or storage_path` walked into',
              wrong is None)


def main():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    for t in (test_stage_before_draw,
              test_no_fallback_into_the_wrong_store,
              test_reassign_has_a_second_line,
              test_the_premise_on_the_real_store):
        t()
    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    for f in FAIL:
        print('  FAILED:', f)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
