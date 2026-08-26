"""
FOV alignment across modalities simultaneously.

Two properties, both of which used to be impossible with one shared
reference combo:

1. Each modality aligns to ITS OWN reference hybe. Alignment is
   per-modality maths -- a hybe is comparable only to another hybe of
   its own modality, fiducial to fiducial -- so a single shared anchor
   could only ever align one modality.

2. Scheduling is FOV-MAJOR: for a given FOV, every (hybe, modality)
   combination is done before the next FOV starts, out of ONE pool.
   Same rule ingestion follows -- finish a FOV completely so it becomes
   usable, rather than advancing every modality in lockstep.

Run: python tests/test_fov_alignment_multimodality.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np                                   # noqa: E402
from unittest import mock                            # noqa: E402

from codelab_pipeline.alignment import chain          # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(('  ok ' if cond else 'FAIL ') + name + (f' -- {detail}' if detail and not cond else ''))


def rec(folder, channel=555):
    return {'folder': folder, 'fiducial_channel': channel, 'channels': [channel],
            'datatype': 'H', 'readout_id': 0, 'readout_name': '', 'total_frames': 2}


DNA = [rec('D_ref'), rec('D_a'), rec('D_b')]
RNA = [rec('R_ref'), rec('R_a')]
SPECS = [
    {'modality': 'DNA', 'storage_path': '/store/DNA',
     'hybe_records': DNA, 'reference_hybe': 'D_ref'},
    {'modality': 'RNA', 'storage_path': '/store/RNA',
     'hybe_records': RNA, 'reference_hybe': 'R_ref'},
]


def main():
    # workers=1 keeps the serial path, which exercises the same task list
    # and the same per-modality reference resolution without spawning.
    seen = []

    def fake_align(moving, reference, lb, ub, border_trim=0, max_shift=None):
        seen.append((float(moving[0, 0]), float(reference[0, 0])))
        return np.eye(3) * 2

    def fake_mip(storage_path, fov, hybe, channel):
        # encode identity in the pixel so the fit can be checked
        return np.full((4, 4), hash((storage_path, hybe)) % 100, dtype=np.uint16)

    with mock.patch.object(chain, 'align_readout_to_reference', side_effect=fake_align), \
         mock.patch.object(chain.analysis_store, 'read_hybe_mip', side_effect=fake_mip), \
         mock.patch.object(chain, 'write_same_modality_matrices') as writer:
        out = chain.align_fov_all_modalities(1, SPECS, write=True, workers=1)

    check('both modalities returned', set(out) == {'DNA', 'RNA'}, str(set(out)))
    check('every DNA hybe aligned', set(out['DNA']) == {'D_ref', 'D_a', 'D_b'}, str(set(out['DNA'])))
    check('every RNA hybe aligned', set(out['RNA']) == {'R_ref', 'R_a'}, str(set(out['RNA'])))
    check('each reference is identity by construction',
          np.allclose(out['DNA']['D_ref'], np.eye(3)) and np.allclose(out['RNA']['R_ref'], np.eye(3)))
    check('non-reference hybes carry a real fit',
          np.allclose(out['DNA']['D_a'], np.eye(3) * 2) and np.allclose(out['RNA']['R_a'], np.eye(3) * 2))

    # every fit compared against ITS OWN modality's reference
    dna_ref = float(fake_mip('/store/DNA', 1, 'D_ref', 555)[0, 0])
    rna_ref = float(fake_mip('/store/RNA', 1, 'R_ref', 555)[0, 0])
    refs_used = {r for _m, r in seen}
    check('each modality fitted against its own reference',
          refs_used == {dna_ref, rna_ref}, f'{refs_used} vs {{{dna_ref}, {rna_ref}}}')

    # per-modality persistence, each under its own storage path
    written = {c.args[0]: c.args[3] for c in writer.call_args_list}
    check('each modality persisted under its own path with its own reference',
          written == {'/store/DNA': 'D_ref', '/store/RNA': 'R_ref'}, str(written))

    # ---- FOV-major scheduling, through the real worker ----
    from windows.main_window import AlignmentWorker
    order = []

    def spy_align(fov, specs, **kw):
        for sp in specs:
            order.append((fov, sp['modality']))
        return {sp['modality']: {r['folder']: np.eye(3) for r in sp['hybe_records']}
                for sp in specs}

    worker = AlignmentWorker([1, 2, 3], {f: SPECS for f in (1, 2, 3)}, write=False)
    with mock.patch.object(chain, 'align_fov_all_modalities', side_effect=spy_align):
        worker.run()
    fov_sequence = [f for f, _m in order]
    check('FOV-major: a FOV finishes across every modality before the next',
          fov_sequence == [1, 1, 2, 2, 3, 3], str(fov_sequence))
    check('every FOV covered both modalities',
          {f: sorted(m for ff, m in order if ff == f) for f in (1, 2, 3)}
          == {1: ['DNA', 'RNA'], 2: ['DNA', 'RNA'], 3: ['DNA', 'RNA']}, str(order))

    # a missing reference is a loud error, not a silent skip
    bad = [{'modality': 'DNA', 'storage_path': '/store/DNA',
            'hybe_records': DNA, 'reference_hybe': 'nope'}]
    try:
        chain.align_fov_all_modalities(1, bad, write=False, workers=1)
        check('unknown reference raises', False, 'no error')
    except ValueError as e:
        check('unknown reference raises', 'nope' in str(e))

    print()
    print(f'{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        raise SystemExit('FAILURES: ' + ', '.join(FAIL))
    print('ALL GOOD')


if __name__ == '__main__':
    main()
