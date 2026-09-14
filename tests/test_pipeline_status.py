"""
The pipeline status module against a temporary v2 store: every step's
state per FOV read the way the app's append modes read it, the plan a
policy implies, and the CLI parsers.

Run:  python tests/test_pipeline_status.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                             # noqa: E402

from codelab_pipeline.io import analysis_store as A            # noqa: E402
from codelab_pipeline.io import paths                          # noqa: E402
from codelab_pipeline.models.allele import AnAllele            # noqa: E402
from codelab_pipeline.pipeline import status as S              # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}' + (f'   [{detail}]' if detail else ''), flush=True)


def cell_dict(cid, fov, celltype, matrices=None):
    return {'id': cid, 'fov': fov, 'reference_hybe': 'Hyb_001', 'reference_modality': 'RNA',
            'nucleus': (np.array([1., 2.]), np.array([3., 4.])), 'nucleus_hybe': 'Hyb_001', 'nucleus_modality': 'RNA',
            'celltype': celltype, 'area': (np.array([5., 6.]), np.array([7., 8.])), 'frame_shape': (1024, 1024),
            'matrices': matrices or {}, 'matrix_anchors': {}, 'matrix_provenance': {}, 'linked': False, 'linked_at': None}


def spot_dict(uid, fov, hybe, cell):
    return {'uid': uid, 'fov': fov, 'modality': 'RNA', 'hybe': hybe, 'channel': 635, 'cell': cell, 'celltype': '',
            'adj_coordinate': (1.5, 2.5, 3.5), 'raw_coordinate': (1.0, 2.0, 3.0), 'size': 4.0, 'brightness': 100.0,
            'linked': False, 'linked_at': None, 'mixture_centroids': (), 'p_exist': 0.9}


def allele(aid, fov, cell, traced):
    a = AnAllele()
    a.id, a.fov, a.cell = aid, fov, cell
    a.anchor_hybe, a.anchor_channel = 'Hyb_001', 555
    if traced:
        a.polymer_adj = {'Hyb_001': [(1.0, 2.0, 3.0, 100.0)]}
    return a


def touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'wb').close()


def main():
    root = tempfile.mkdtemp(prefix='pipestatus_')
    try:
        dp = os.path.join(root, 'proj')
        os.makedirs(dp)
        paths.write_manifest(dp, ['DNA', 'RNA'])
        rna, dna = os.path.join(dp, 'RNA'), os.path.join(dp, 'DNA')
        records_rna = [{'folder': 'Hyb_001', 'channels': [635, 555], 'fiducial_channel': 555, 'readout_name': 'MCM2_mRNA'},
                       {'folder': 'Hyb_002', 'channels': [635, 555], 'fiducial_channel': 555, 'readout_name': 'CCNB1_exon'},
                       {'folder': 'Hyb_003', 'channels': [635, 555], 'fiducial_channel': 555, 'readout_name': 'MCM2_nascent'}]
        records_dna = [{'folder': 'Hyb_001', 'channels': [555, 488], 'fiducial_channel': 488, 'readout_name': None},
                       {'folder': 'Hyb_002', 'channels': [555, 488], 'fiducial_channel': 488, 'readout_name': None}]
        project = {'config': 'test', 'project_root': dp, 'fovs': [1, 2, 3],
                   'modalities': {'RNA': {'storage_path': rna, 'layout_path': '', 'records': records_rna, 'fields': {}},
                                  'DNA': {'storage_path': dna, 'layout_path': '', 'records': records_dna, 'fields': {}}},
                   'params': {'fov_alignment': {'reference_hybe_RNA': 'Hyb_001', 'reference_hybe_DNA': 'Hyb_001'},
                              'spot_localization': {'hybe': 'Hyb_001 (DNA)', 'channel': '555'}},
                   'celltype_names': ['A', 'B']}
        # FOV 1: everything; FOV 2: ingested + cells only; FOV 3: nothing
        for h in ('Hyb_001', 'Hyb_002', 'Hyb_003'):
            touch(os.path.join(paths.mips_dir(rna, 1), h + '.h5'))
        for h in ('Hyb_001', 'Hyb_002'):
            touch(os.path.join(paths.mips_dir(dna, 1), h + '.h5'))
        touch(os.path.join(paths.mips_dir(rna, 2), 'Hyb_001.h5'))       # partial ingestion
        touch(os.path.join(paths.mips_dir(dna, 2), 'Hyb_001.h5'))
        mats = {('Hyb_002', 'RNA'): {'yx': np.eye(3), 'dz': 0.0}}
        A.write_cell_dicts(rna, 1, [cell_dict(1, 1, 'A', mats), cell_dict(2, 1, 'B', mats)])
        A.write_cell_dicts(rna, 2, [cell_dict(1, 2, 'A'), cell_dict(2, 2, '')])
        A.write_same_modality_matrices(rna, 1, {'Hyb_002': np.eye(3), 'Hyb_003': np.eye(3)}, 'Hyb_001')
        A.write_same_modality_matrices(dna, 1, {'Hyb_002': np.eye(3)}, 'Hyb_001')
        A.write_cross_modal_matrix(rna, 1, np.eye(3), modality='DNA')
        A.write_spot_dicts(rna, 1, 'RNA', 'Hyb_001', 635, [spot_dict(1, 1, 'Hyb_001', 1)])
        A.write_spot_dicts(rna, 1, 'RNA', 'Hyb_002', 635, [spot_dict(2, 1, 'Hyb_002', 1)])
        A.write_spot_dicts(dna, 1, 'DNA', 'Hyb_001', 555, [dict(spot_dict(3, 1, 'Hyb_001', 1), modality='DNA', channel=555)])
        A.write_fov_cellcycle(rna, 1, {'version': 1, 'model': 'm', 'rows': [{'cell': 1, 'theta_deg': 10.0}, {'cell': 2, 'theta_deg': 20.0}]})
        A.write_fov_alleles(rna, 1, [allele(1, 1, 1, True), allele(2, 1, 2, False)])

        print('per-step states')
        st1 = S.fov_status(project, 1)
        st2 = S.fov_status(project, 2)
        st3 = S.fov_status(project, 3)
        exp1 = {'ingestion': 'done', 'segmentation': 'done', 'fov_alignment': 'done', 'cross_modal': 'done',
                'cell_alignment': 'done', 'localization': 'done', 'celltype': 'done', 'cellcycle': 'done', 'tracing': 'partial'}
        for s, e in exp1.items():
            check(f'fov 1 {s} is {e}', st1[s]['state'] == e, f'{st1[s]}')
        exp2 = {'ingestion': 'partial', 'segmentation': 'done', 'fov_alignment': 'missing', 'cross_modal': 'missing',
                'cell_alignment': 'missing', 'localization': 'missing', 'celltype': 'partial', 'cellcycle': 'missing', 'tracing': 'missing'}
        for s, e in exp2.items():
            check(f'fov 2 {s} is {e}', st2[s]['state'] == e, f'{st2[s]}')
        check('fov 3: nothing ingested, no cells -> deep steps n/a', st3['ingestion']['state'] == 'missing'
              and st3['segmentation']['state'] == 'missing' and st3['cell_alignment']['state'] == 'n/a'
              and st3['celltype']['state'] == 'n/a' and st3['cellcycle']['state'] == 'n/a' and st3['tracing']['state'] == 'n/a', str(st3))
        check('tracing counts traced alleles', st1['tracing']['have'] == 1 and st1['tracing']['want'] == 2, str(st1['tracing']))
        check('fov alignment leaves the reference hybes out of the want', st1['fov_alignment']['want'] == 3, str(st1['fov_alignment']))
        check('localization wants the countable RNA rounds and the config anchor round, not the nascent or the other DNA round',
              st1['localization']['want'] == 3
              and S.localization_targets(project) == {'RNA': [('Hyb_001', 635), ('Hyb_002', 635)], 'DNA': [('Hyb_001', 555)]},
              str(S.localization_targets(project)))
        with_genes = dict(project, params=dict(project['params'], cellcycle={'genes': 'RNA|Hyb_003|635|MCM2_nascent|-'}))
        check('cell-cycle gene sources from the config are expected too', ('Hyb_003', 635) in S.localization_targets(with_genes)['RNA'])
        shallow = S.fov_status(project, 1, deep=False)
        check('shallow status leaves the deep steps n/a without opening capsules', shallow['cell_alignment']['state'] == 'n/a'
              and shallow['segmentation']['state'] == 'done')

        print('table, summary, plan')
        rows = S.status_table(project)
        check('one row per FOV', [r['fov'] for r in rows] == [1, 2, 3])
        summ = S.summarize(rows)
        check('summary counts', summ['ingestion'] == {'done': 1, 'partial': 1, 'missing': 1, 'n/a': 0}, str(summ['ingestion']))
        p = S.plan(rows)
        check('existing policy runs only what is not done', p['ingestion'] == [2, 3] and p['segmentation'] == [3]
              and p['cross_modal'] == [2, 3] and p['tracing'] == [1, 2] and p['cell_alignment'] == [2], str(p))
        p2 = S.plan(rows, {'localization': 'redo', 'segmentation': 'skip'})
        check('redo runs every applicable FOV, skip none', p2['localization'] == [1, 2, 3] and p2['segmentation'] == [])
        try:
            S.plan(rows, {'tracing': 'sometimes'})
            check('an unknown policy is refused', False)
        except ValueError:
            check('an unknown policy is refused', True)
        grid = S.format_grid(rows)
        check('grid has a line per FOV', grid.count('\n') == 1 + 3, grid)
        check('parse_policy', S.parse_policy('segmentation=skip, localization=redo') == {'segmentation': 'skip', 'localization': 'redo'})
        check('parse_fovs', S.parse_fovs('1-3, 7, 9-10') == [1, 2, 3, 7, 9, 10])
        lvl = S.project_level(project)
        check('project level: no model, no celltype config yet', lvl == {'cellcycle_model': False, 'celltype_config': False}, str(lvl))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        print('FAILED:', FAIL)
        sys.exit(1)
    print('ALL GOOD')


if __name__ == '__main__':
    main()
