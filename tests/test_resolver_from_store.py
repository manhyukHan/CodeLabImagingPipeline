"""
resolver_for: the FOV alignment layer is assembled from the STORE, never
from a path outside the project root.

The manifest records layout_path/dax_directory on the acquisition share,
with whichever drive letter the last session to open the project had the
NAS mounted on -- Z: for one user, Q: for the next, the same file either
way. resolver_for used to source its hybe list from that file, so an
unreachable layout left the FOV layer EMPTY, and to_shared's per-hybe
identity fallback then dropped the whole FOV translation silently. On the
MP58 store that was 3116 stored matrices ignored across 41 FOVs, up to
13.7 px per FOV, arriving as fiducial drift the gate refuses.

Run: python tests/test_resolver_from_store.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline.analysis import resolvers                # noqa: E402
from codelab_pipeline.io import analysis_store as A            # noqa: E402
from codelab_pipeline.io import paths                          # noqa: E402

PASS, FAIL = [], []

UNREACHABLE = 'Z:/not-mounted-here/ExperimentLayout.xlsx'
FOV_T = (7.60, -0.81)        # the FOV leg, as the store holds it
RES_T = (0.76, 0.59)         # the cell's residual on top of it


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(('  ok ' if cond else 'FAIL ') + name
          + (f'   [{detail}]' if detail and not cond else ''))


def translate(ty, tx):
    H = np.eye(3)
    H[0, 2], H[1, 2] = ty, tx
    return H


class StubCell:
    """to_shared reads .matrices and .id; nothing else."""

    def __init__(self, matrices, cid=1):
        self.matrices = matrices
        self.id = cid


def cell_dict(cid, fov, residual):
    return {'id': cid, 'fov': fov, 'reference_hybe': 'H1',
            'reference_modality': 'DNA',
            'nucleus': (np.array([1., 2.]), np.array([3., 4.])),
            'nucleus_hybe': 'H1', 'nucleus_modality': 'DNA',
            'celltype': 'typeA',
            'area': (np.array([5., 6.]), np.array([7., 8.])),
            'frame_shape': (1024, 1024),
            'matrices': {('H2', 'DNA'): {'yx': translate(*RES_T), 'dz': 1.5,
                                         'yx_is_residual': residual}},
            'matrix_anchors': {'DNA': np.eye(3)},
            'matrix_provenance': {},
            'distmap': np.array([]), 'linked': False, 'linked_at': None}


def touch_mips(sp, fov, hybes):
    d = paths.mips_dir(sp, fov)
    os.makedirs(d, exist_ok=True)
    for h in hybes:
        open(os.path.join(d, f'{h}.h5'), 'wb').close()


def main():
    root = tempfile.mkdtemp(prefix='resolver_')
    try:
        dp = os.path.join(root, 'proj')
        # one modality, so the hub inference is unambiguous without bridges
        paths.write_manifest(dp, ['DNA'], layout_paths={'DNA': UNREACHABLE})
        sp = os.path.join(dp, 'DNA')

        # FOV 7: a real FOV leg in the store, and a cell whose entry is a
        # bare residual -- the shape every aligned cell carries today.
        touch_mips(sp, 7, ['H1', 'H2'])
        A.write_same_modality_matrices(
            sp, 7, {'H1': np.eye(3), 'H2': translate(*FOV_T)},
            reference_hybe='H1')
        A.write_cell_dicts(sp, 7, [cell_dict(1, 7, residual=True)])

        print('the FOV layer survives an unreachable layout_path')
        check('layout_path is in fact unreachable',
              not os.path.exists(UNREACHABLE))
        r = resolvers.resolver_for(sp, 7)
        check('within holds every aligned hybe, not zero',
              sorted(r.within.get('DNA', {})) == ['H1', 'H2'],
              f'{sorted(r.within.get("DNA", {}))}')
        H = np.asarray(r.within['DNA']['H2'], float)
        check('the stored translation arrives intact',
              np.allclose((H[0, 2], H[1, 2]), FOV_T), f'{H[0, 2]}, {H[1, 2]}')

        print('\nto_shared composes BOTH legs, not the residual alone')
        cell = StubCell({('H2', 'DNA'): {'yx': translate(*RES_T), 'dz': 1.5,
                                        'yx_is_residual': True}})
        T = np.asarray(r.to_shared('H2', 'DNA', cell), float)
        want = (FOV_T[0] + RES_T[0], FOV_T[1] + RES_T[1])
        check('cell route = FOV leg + residual',
              np.allclose((T[0, 2], T[1, 2]), want), f'{T[0, 2]}, {T[1, 2]}')
        check('and it is NOT the bare residual (the old silent answer)',
              not np.allclose((T[0, 2], T[1, 2]), RES_T))
        check('cell=None gives the FOV leg alone',
              np.allclose((np.asarray(r.to_shared('H2', 'DNA'), float)[0, 2],
                           np.asarray(r.to_shared('H2', 'DNA'), float)[1, 2]),
                          FOV_T))

        print('\nthe answer does not depend on what layout_path says')
        seen = []
        for lp in (UNREACHABLE, '', os.path.abspath(__file__)):
            paths.write_manifest(dp, ['DNA'], layout_paths={'DNA': lp})
            rr = resolvers.resolver_for(sp, 7)
            seen.append(np.asarray(rr.to_shared('H2', 'DNA', cell), float))
        check('three different layout_path values, one answer',
              all(np.allclose(seen[0], s) for s in seen[1:]))
        paths.write_manifest(dp, ['DNA'], layout_paths={'DNA': UNREACHABLE})

        print('\na residual with no FOV leg is refused, not averaged in')
        # FOV 8: cells carry residuals, the store holds no FOV matrices.
        A.write_cell_dicts(sp, 8, [cell_dict(1, 8, residual=True)])
        try:
            resolvers.resolver_for(sp, 8)
            check('refuses an uncomposable pair', False, 'no raise')
        except ValueError as e:
            check('refuses an uncomposable pair', True)
            check('the message names the modality and the FOV',
                  'DNA' in str(e) and 'FOV 8' in str(e), str(e))

        print('\nbut an unaligned FOV with no residuals is not an error')
        # FOV 9: nothing aligned, and the cell says so (not a residual).
        A.write_cell_dicts(sp, 9, [cell_dict(1, 9, residual=False)])
        try:
            r9 = resolvers.resolver_for(sp, 9)
            check('no FOV matrices and no residuals: still builds', True)
            check('its FOV layer is honestly empty',
                  r9.within.get('DNA') == {})
        except Exception as e:                                # noqa: BLE001
            check('no FOV matrices and no residuals: still builds', False,
                  f'{type(e).__name__}: {e}')
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        for n in FAIL:
            print(f'  FAILED: {n}')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
