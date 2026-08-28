"""
Per-FOV analysis store: roundtrip, append semantics, manifest self-heal,
migration guard, and the vlinks.h5 -> capsule migration tool -- all on a
temp v2 project (no real data touched).
Run: python tests/test_analysis_store.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))

from codelab_pipeline.io import analysis_store as A          # noqa: E402
from legacy import vlinks_store as LEGACY                  # noqa: E402
from codelab_pipeline.io import paths                        # noqa: E402
import migrate_vlinks                                        # noqa: E402

PASS = []
FAIL = []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(('  ok ' if cond else 'FAIL ') + name + (f' -- {detail}' if detail and not cond else ''))


def make_project(root, name):
    dp = os.path.join(root, name)
    paths.write_manifest(dp, ['DNA', 'RNA'])
    return dp


class FakeSpot:
    def __init__(self, uid, fov, modality, hybe, channel, cell=-1):
        self.uid = uid
        self.d = {'uid': uid, 'fov': fov, 'modality': modality, 'hybe': hybe,
                  'channel': channel, 'cell': cell, 'celltype': '',
                  'adj_coordinate': (1.5, 2.5, 3.5), 'raw_coordinate': (1.0, 2.0, 3.0),
                  'size': 4.0, 'brightness': 100.0, 'linked': False,
                  'linked_at': None, 'mixture_centroids': ((1.0, 2.0, 3.0, 4.0),)}

    def save(self):
        self.d['uid'] = self.uid
        return dict(self.d)


class FakeContainer:
    def __init__(self, dicts):
        self.dicts = dicts

    def get_cells(self, fov):
        class C:
            def __init__(self, d):
                self.d = d

            def save(self):
                return dict(self.d)
        return [C(d) for d in self.dicts]


def cell_dict(cid, fov):
    return {'id': cid, 'fov': fov, 'reference_hybe': 'H1',
            'reference_modality': 'DNA', 'nucleus': (np.array([1., 2.]), np.array([3., 4.])),
            'nucleus_hybe': 'H1', 'nucleus_modality': 'DNA', 'celltype': 'typeA',
            'area': (np.array([5., 6.]), np.array([7., 8.])),
            'frame_shape': (1024, 1024),
            'matrices': {('H2', 'DNA'): {'yx': np.eye(3), 'dz': 1.5, 'yx_is_residual': True}},
            'matrix_anchors': {'DNA': np.eye(3) * 2},
            'matrix_provenance': {('H2', 'DNA'): {'reference_sequence': 'H2->H1',
                                                  'steps': np.eye(3)[None]}},
            'distmap': np.array([]), 'linked': False, 'linked_at': None}


def allele_dict(aid, fov):
    return {'id': aid, 'fov': fov, 'cell': 1, 'anchor_uid': 7, 'anchor_channel': 1,
            'anchor_hybe': 'H1', 'coordinate': (1., 2., 3.), 'raw_coordinate': (1., 2., 3.),
            'fiducial_trace_adj': {'H2': np.array([1., 2., 3., 4.])},
            'polymer_adj': {'H2': [np.array([1., 2., 3., 4.])]},
            'rejected_hybes': {'H3': 'drift'},
            'final_polymer': np.array([[1., 2., 3.]]),
            'linked': False, 'linked_at': None}


def main():
    root = tempfile.mkdtemp(prefix='astore_')
    try:
        dp = make_project(root, 'proj')
        sp = os.path.join(dp, 'DNA')
        sp_rna = os.path.join(dp, 'RNA')
        fov = 7

        # -- params routing ------------------------------------------------
        A.write_global_params(sp, layout_path='L_DNA', cell_pad=12, none_key=None)
        A.write_global_params(sp_rna, layout_path='L_RNA')
        p_dna, p_rna = A.read_global_params(sp), A.read_global_params(sp_rna)
        check('params: modality-scoped key does not clobber',
              p_dna.get('layout_path') == 'L_DNA' and p_rna.get('layout_path') == 'L_RNA')
        check('params: shared key visible from both',
              p_dna.get('cell_pad') == 12 and p_rna.get('cell_pad') == 12)
        check('params: None skipped', 'none_key' not in p_dna)

        # -- celltype config ----------------------------------------------
        cal = {'scale': {('H1', 1, 'DNA'): {7: 1.5}}, 'lower_bound': {}, 'upper_bound': {}}
        A.write_celltype_config(sp, {'A': '1-3'}, {'A': ('H1', 1, 'DNA')}, cal, 'Vote')
        fr, bc, c2, meth = A.read_celltype_config(sp_rna)   # same store via RNA path
        check('celltype config roundtrip (cross-modality path)',
              fr == {'A': '1-3'} and bc == {'A': ('H1', 1, 'DNA')}
              and c2 == cal and meth == 'Vote')

        # -- uids ----------------------------------------------------------
        u1 = A.allocate_spot_uids(sp, fov, 3)
        u2 = A.allocate_spot_uids(sp_rna, fov, 2)   # same FOV counter via RNA path
        check('uids monotonic across modality paths',
              u1 == [1, 2, 3] and u2 == [4, 5], f'{u1} {u2}')

        # -- spots ---------------------------------------------------------
        spots = [FakeSpot(0, fov, 'DNA', 'H2', 1), FakeSpot(0, fov, 'DNA', 'H2', 1)]
        A.write_spots(sp, fov, 'DNA', 'H2', 1, spots)
        check('write_spots allocates uids', all(s.uid for s in spots))
        back = A.read_spots(sp, fov, 'DNA', 'H2', 1)
        check('spot slice roundtrip', len(back) == 2
              and back[0]['adj_coordinate'] == (1.5, 2.5, 3.5)
              and back[0]['mixture_centroids'] == ((1.0, 2.0, 3.0, 4.0),))
        A.write_spots(sp_rna, fov, 'RNA', 'H2', 2, [FakeSpot(0, fov, 'RNA', 'H2', 2)])
        allspots = A.read_spots(sp, fov)
        check('whole-FOV read crosses modalities', len(allspots) == 3)
        sl = A.spot_slices(sp, fov)
        check('spot_slices from listdir', sorted(sl) == [('DNA', 'H2', 1), ('RNA', 'H2', 2)], str(sl))
        A.write_spots(sp, fov, 'DNA', 'H2', 1, [])   # clear slice
        check('empty-slice replace persists', A.read_spots(sp, fov, 'DNA', 'H2', 1) == []
              and ('DNA', 'H2', 1) in A.spot_slices(sp, fov))

        # -- cells / alleles ----------------------------------------------
        A.write_cells(sp, fov, FakeContainer([cell_dict(1, fov), cell_dict(2, fov)]))
        cells, m = A.read_cells(sp, fov)
        check('cells roundtrip', len(cells) == 2 and m == ''
              and cells[0]['matrices'][('H2', 'DNA')]['dz'] == 1.5
              and cells[0]['celltype'] == 'typeA')
        check('cells missing FOV -> (None, "")', A.read_cells(sp, 99) == (None, ''))

        A.write_fov_alleles(sp, fov, FakeContainer([allele_dict(1, fov)]).get_cells(fov))
        al = A.read_fov_alleles(sp, fov)
        check('alleles roundtrip', len(al) == 1 and al[0]['rejected_hybes'] == {'H3': 'drift'}
              and np.allclose(al[0]['final_polymer'], [[1., 2., 3.]]))

        # -- matrices: merge + ingested gate + aligned_hybes ---------------
        os.makedirs(paths.mips_dir(sp, fov), exist_ok=True)
        for h in ('H1', 'H2', 'H3'):
            open(paths.mip_path(sp, fov, h), 'wb').close()
        A.write_same_modality_matrices(sp, fov, {'H2': np.eye(3) * 2}, 'H1')
        A.write_same_modality_matrices(sp, fov, {'H3': np.eye(3) * 3}, 'H1')  # append merge
        ah = A.aligned_hybes(sp, fov)
        check('aligned_hybes after merge', ah == frozenset({'H2', 'H3'}), str(ah))
        fm = A.read_same_modality_matrices(sp, fov, ['H1', 'H2', 'H3', 'H9'])
        check('read matrices: ingested-unaligned=identity, merge kept, not-ingested skipped',
              np.allclose(fm[('H1', 'DNA')], np.eye(3))
              and np.allclose(fm[('H2', 'DNA')], np.eye(3) * 2)
              and np.allclose(fm[('H3', 'DNA')], np.eye(3) * 3)
              and ('H9', 'DNA') not in fm)
        check('aligned_hybes empty for RNA', A.aligned_hybes(sp_rna, fov) == frozenset())

        # -- cross-modal ----------------------------------------------------
        A.write_cross_modal_matrix(sp, fov, np.eye(3) * 5, modality='DNA')
        A.write_cross_modal_z(sp, fov, -3.0, modality='DNA')
        A.write_cross_modal_quality(sp, fov, {'residual_after': 0.5, 'z_quality': 0.9},
                                    modality='DNA')
        check('cross-modal keyed roundtrip',
              np.allclose(A.read_cross_modal_matrix(sp, fov, modality='DNA'), np.eye(3) * 5)
              and A.read_cross_modal_z(sp, fov, modality='DNA') == -3.0
              and A.read_cross_modal_quality(sp, fov, modality='DNA')
              == {'residual_after': 0.5, 'z_quality': 0.9})
        check('cross-modal other modality empty',
              A.read_cross_modal_matrix(sp, fov, modality='RNA') is None
              and A.read_cross_modal_z(sp, fov, modality='RNA') == 0.0)

        # -- counts ---------------------------------------------------------
        counts = A.fov_counts(sp, [fov, 99])
        check('fov_counts from manifest',
              counts[fov] == {'cells': 2, 'spots': 1, 'alleles': 1}
              and counts[99] == {'cells': 0, 'spots': 0, 'alleles': 0}, str(counts))

        # -- manifest self-heal ---------------------------------------------
        fd = paths.analysis_fov_dir(sp, fov)
        os.remove(os.path.join(fd, 'manifest.json'))
        counts2 = A.fov_counts(sp, [fov])
        check('manifest rebuild restores counts',
              counts2[fov] == {'cells': 2, 'spots': 1, 'alleles': 1}, str(counts2))
        check('manifest rebuild restores aligned_hybes',
              A.aligned_hybes(sp, fov) == frozenset({'H2', 'H3'}))
        u3 = A.allocate_spot_uids(sp, fov, 1)
        check('uid floor survives manifest loss (no reuse)', u3[0] > max(s.uid for s in spots),
              str(u3))

        # -- migration guard -------------------------------------------------
        dp2 = make_project(root, 'proj2')
        os.makedirs(os.path.join(dp2, 'analysis'), exist_ok=True)
        open(os.path.join(dp2, 'analysis', 'vlinks.h5'), 'wb').close()
        try:
            A.read_cells(os.path.join(dp2, 'DNA'), 1)
            check('unmigrated store refused', False)
        except RuntimeError as e:
            check('unmigrated store refused', 'migrate_vlinks' in str(e))

        # -- migration tool ---------------------------------------------------
        dp3 = make_project(root, 'proj3')
        sp3, sp3r = os.path.join(dp3, 'DNA'), os.path.join(dp3, 'RNA')
        LEGACY.declare_modality(sp3, 'DNA')
        LEGACY.declare_modality(sp3r, 'RNA')
        LEGACY.write_cells(sp3, 3, FakeContainer([cell_dict(1, 3)]))
        sp_obj = [FakeSpot(0, 3, 'DNA', 'H2', 1)]
        LEGACY.write_spots(sp3, 3, 'DNA', 'H2', 1, sp_obj)
        LEGACY.write_spots(sp3r, 3, 'RNA', 'H5', 2, [FakeSpot(0, 3, 'RNA', 'H5', 2)])
        LEGACY.write_fov_alleles(sp3, 3, FakeContainer([allele_dict(1, 3)]).get_cells(3))
        LEGACY.write_same_modality_matrices(sp3, 3, {'H2': np.eye(3) * 7}, 'H1')
        LEGACY.write_cross_modal_matrix(sp3, 3, np.eye(3) * 9, modality='DNA')
        LEGACY.write_cross_modal_z(sp3, 3, 2.0)     # legacy flat
        LEGACY.write_global_params(sp3, layout_path='L3', cell_pad=5)
        LEGACY.write_celltype_config(sp3, {'B': '1'}, {}, {'scale': {}, 'lower_bound': {},
                                                          'upper_bound': {}}, 'Median')
        migrate_vlinks.main(dp3)
        check('migration: retired file exists',
              os.path.exists(os.path.join(dp3, 'analysis', 'vlinks.h5.retired'))
              and not os.path.exists(os.path.join(dp3, 'analysis', 'vlinks.h5')))
        cells3, _ = A.read_cells(sp3, 3)
        check('migration: cells', cells3 is not None and len(cells3) == 1
              and cells3[0]['celltype'] == 'typeA')
        check('migration: spots per modality',
              len(A.read_spots(sp3, 3, 'DNA', 'H2', 1)) == 1
              and len(A.read_spots(sp3, 3, 'RNA', 'H5', 2)) == 1)
        check('migration: alleles', len(A.read_fov_alleles(sp3, 3)) == 1)
        os.makedirs(paths.mips_dir(sp3, 3), exist_ok=True)
        open(paths.mip_path(sp3, 3, 'H2'), 'wb').close()
        fm3 = A.read_same_modality_matrices(sp3, 3, ['H2'])
        check('migration: matrices', np.allclose(fm3[('H2', 'DNA')], np.eye(3) * 7))
        check('migration: cross-modal keyed + flat-z fallback',
              np.allclose(A.read_cross_modal_matrix(sp3, 3, modality='DNA'), np.eye(3) * 9)
              and A.read_cross_modal_z(sp3, 3, modality='DNA') == 2.0)
        check('migration: params', A.read_global_params(sp3).get('layout_path') == 'L3'
              and A.read_global_params(sp3).get('cell_pad') == 5)
        fr3, _bc3, _cal3, meth3 = A.read_celltype_config(sp3)
        check('migration: celltype config', fr3 == {'B': '1'} and meth3 == 'Median')
        u4 = A.allocate_spot_uids(sp3, 3, 1)
        check('migration: uid counter carried', u4[0] > sp_obj[0].uid, str(u4))

        print()
        print(f'{len(PASS)} passed, {len(FAIL)} failed')
        if FAIL:
            raise SystemExit('FAILURES: ' + ', '.join(FAIL))
        print('ALL GOOD')
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == '__main__':
    main()
