"""
Single-source-of-truth test for coordinate transforms.

"Two resolvers silently disagreeing" is this codebase's documented core
failure class (see ACell._require_composable). This test pins every
transform helper -- the session wrappers AND the two compositions that
still own independent math (_fov_matrices_in_frame's [within, bridge]
compose, chain.hybe_to_cellref_matrix) -- to frames.FrameResolver
NUMERICALLY, on real data, for every (hybe, modality) and for both a
residual-matrix cell and the no-cell case. If any helper drifts from
the resolver by more than float noise, this fails and names it.

Config via CODELAB_EQ_CONFIG (point it at a clone when the live app
holds the store lock).
"""
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import numpy.linalg as la
from unittest import mock
from PyQt5 import QtWidgets

CONFIG = os.environ.get('CODELAB_EQ_CONFIG', 'configs/chr19_downstream_debug.xml')
FOV = 1
ATOL = 1e-9

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
from windows.main_window import MainWindow
from codelab_pipeline.alignment import chain as alignment
from codelab_pipeline.alignment import spot_mapper

failures = []
n_checks = 0


def check(name, A, B):
    global n_checks
    n_checks += 1
    if A is None or B is None:
        if A is not B:
            failures.append(f'{name}: one side None ({A is None} vs {B is None})')
        return
    if not np.allclose(np.asarray(A, float), np.asarray(B, float), atol=ATOL):
        d = np.abs(np.asarray(A, float) - np.asarray(B, float)).max()
        failures.append(f'{name}: max diff {d:.3e}')


def main():
    with mock.patch.object(QtWidgets.QMessageBox, 'information'), \
            mock.patch.object(QtWidgets.QMessageBox, 'warning'), \
            mock.patch.object(QtWidgets.QMessageBox, 'critical'), \
            mock.patch.object(QtWidgets.QMessageBox, 'question',
                              return_value=QtWidgets.QMessageBox.Yes):
        w = MainWindow(CONFIG)
        w._activate_fov(FOV)
        keys = sorted(w.fov_matrices.get(FOV, {}).keys())
        assert keys, 'no FOV matrices loaded -- fixture missing'
        cells = list(w.cell_container_permanent.get_cells(FOV))
        res_cell = next((c for c in cells
                         if any(v.get('yx_is_residual') for v in c.matrices.values())), None)
        assert res_cell is not None, 'no residual-matrix cell -- fixture missing arming state'
        shared = w._shared_frame_modality()

        for cell in (None, res_cell):
            tag = 'cell=None' if cell is None else f'cell={cell.id}(residual)'
            resolver = w._frame_resolver(cell, FOV)
            for (h, m) in keys:
                # 1. wrapper == resolver, same arguments
                check(f'_matrix_to_shared[{h},{m},{tag}]',
                      w._matrix_to_shared(h, m, cell, FOV),
                      resolver.to_shared(h, m, cell))
                # 2. the dict-based composition == resolver, per entry
                composed = w._fov_matrices_in_frame(m, shared or m, FOV)
                check(f'_fov_matrices_in_frame[{h},{m}]',
                      composed[(h, m)] if composed and (h, m) in composed else None,
                      resolver.to_shared(h, m, None) if composed and (h, m) in composed else None)
                # 3. spot_mapper with resolver == resolver, as a point map
                p = np.array([412.0, 633.0, 1.0])
                H = resolver.to_shared(h, m, cell)
                got = spot_mapper.raw_to_reference((p[0], p[1]), h, None,
                                                   modality=m, cell=cell, resolver=resolver) \
                    if cell is not None else None
                if cell is not None:
                    check(f'spot_mapper.raw_to_reference[{h},{m},{tag}]',
                          np.asarray(got), (H @ p)[:2])
                if cell is not None:
                    # 4. transform identity: to-cellref == inv(dst) @ src
                    ref = (cell.reference_hybe, cell.reference_modality)
                    check(f'_matrix_to_cellref[{h},{m},{tag}]',
                          w._matrix_to_cellref(h, m, cell, FOV),
                          la.inv(resolver.to_shared(*ref, cell)) @ resolver.to_shared(h, m, cell))

        # 5. chain.hybe_to_cellref_matrix (independent math, used by the
        # legacy no-resolver fallbacks + compute_cell_alignment) == the
        # resolver's own frame-to-frame transform, over composed dicts.
        ref = (res_cell.reference_hybe, res_cell.reference_modality)
        resolver0 = w._frame_resolver(None, FOV)
        for modality in {m for (_h, m) in keys}:
            composed = w._fov_matrices_in_frame(modality, shared or modality, FOV)
            if not composed:
                continue
            ref_composed = w._fov_matrices_in_frame(ref[1], shared or ref[1], FOV) or {}
            H_ref = ref_composed.get(ref, np.eye(3))
            for (h, m) in composed.keys():
                check(f'hybe_to_cellref_matrix[{h},{m}]',
                      alignment.hybe_to_cellref_matrix(composed, H_ref, h),
                      resolver0.transform((h, m), ref, None)[0])

    print(f'{n_checks} equivalence checks, {len(failures)} failed')
    for f in failures:
        print('  FAIL', f)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
