"""
The Cell Cycle stage's store contract: counts from stored spots (p_exist
gate, soft count, candidates), the per-FOV placement capsule, the
experiment-level model file, the Population join and the phase gates.
All on a temp v2 project -- no real data touched.

Run:  python tests/test_cellcycle_store.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                             # noqa: E402
import pandas as pd                                            # noqa: E402

from codelab_pipeline.io import analysis_store as A            # noqa: E402
from codelab_pipeline.io import paths                          # noqa: E402
from codelab_pipeline.analysis import cellcycle as CC          # noqa: E402
from codelab_pipeline.analysis import gate                     # noqa: E402
from codelab_pipeline.analysis import population as popmod     # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}' + (f'   [{detail}]' if detail else ''), flush=True)


def cell_dict(cid, fov, celltype):
    return {'id': cid, 'fov': fov, 'reference_hybe': 'Hyb_001',
            'reference_modality': 'RNA',
            'nucleus': (np.array([1., 2.]), np.array([3., 4.])),
            'nucleus_hybe': 'Hyb_001', 'nucleus_modality': 'RNA',
            'celltype': celltype,
            'area': (np.array([5., 6.]), np.array([7., 8.])),
            'frame_shape': (1024, 1024), 'matrices': {}, 'matrix_anchors': {},
            'matrix_provenance': {}, 'linked': False, 'linked_at': None}


def spot_dict(uid, fov, hybe, cell, p_exist):
    d = {'uid': uid, 'fov': fov, 'modality': 'RNA', 'hybe': hybe, 'channel': 635,
         'cell': cell, 'celltype': '', 'adj_coordinate': (1.5, 2.5, 3.5),
         'raw_coordinate': (1.0, 2.0, 3.0), 'size': 4.0, 'brightness': 100.0,
         'linked': False, 'linked_at': None, 'mixture_centroids': ()}
    if p_exist is not None:
        d['p_exist'] = p_exist
    return d


def main():
    root = tempfile.mkdtemp(prefix='ccstore_')
    try:
        dp = os.path.join(root, 'proj')
        paths.write_manifest(dp, ['DNA', 'RNA'])
        sp = os.path.join(dp, 'RNA')
        os.makedirs(sp, exist_ok=True)
        # two FOVs, two cells each; FOV 2's second cell is Nocodazole
        c2 = cell_dict(2, 1, 'Unsynchronized')
        c2['area'] = (np.array([50., 51.]), np.array([60., 61.]))       # away from the bright square below
        A.write_cell_dicts(sp, 1, [cell_dict(1, 1, 'Unsynchronized'), c2])
        A.write_cell_dicts(sp, 2, [cell_dict(1, 2, 'Unsynchronized'), cell_dict(2, 2, 'Nocodazole')])
        # gene A (Hyb_001): cell 1 has 3 spots at p 0.9/0.6/0.2, cell 2 one
        # spot without p_exist (manual keep), one homeless spot
        A.write_spot_dicts(sp, 1, 'RNA', 'Hyb_001', 635, [
            spot_dict(1, 1, 'Hyb_001', 1, 0.9), spot_dict(2, 1, 'Hyb_001', 1, 0.6),
            spot_dict(3, 1, 'Hyb_001', 1, 0.2), spot_dict(4, 1, 'Hyb_001', 2, None),
            spot_dict(5, 1, 'Hyb_001', -1, 0.95)])
        # gene B (Hyb_002) only in FOV 1: cell 2 two spots at 0.5 (the gate is inclusive)
        A.write_spot_dicts(sp, 1, 'RNA', 'Hyb_002', 635, [
            spot_dict(6, 1, 'Hyb_002', 2, 0.5), spot_dict(7, 1, 'Hyb_002', 2, 0.5)])
        A.write_spot_dicts(sp, 2, 'RNA', 'Hyb_001', 635, [
            spot_dict(1, 2, 'Hyb_001', 1, 0.7), spot_dict(2, 2, 'Hyb_001', 2, 0.8)])

        print('counts from stored spots')
        sources = [('RNA', 'Hyb_001', 635), ('RNA', 'Hyb_002', 635)]
        t, fails = CC.count_table(sp, [1, 2], sources, gate=0.5, jobs=1)
        check('no FOV failed', fails == [], str(fails))
        check('one row per (cell, source)', len(t) == 8, str(len(t)))
        r = t[(t['fov'] == 1) & (t['cell'] == 1) & (t['hybe'] == 'Hyb_001')].iloc[0]
        check('gated count, soft count and candidates for a cell with p_exist',
              r['n'] == 2 and abs(r['soft'] - 1.7) < 1e-9 and r['candidates'] == 3,
              f"n {r['n']} soft {r['soft']} cand {r['candidates']}")
        r = t[(t['fov'] == 1) & (t['cell'] == 2) & (t['hybe'] == 'Hyb_001')].iloc[0]
        check('a stored spot without p_exist counts once with soft weight 1',
              r['n'] == 1 and r['soft'] == 1.0 and r['candidates'] == 1)
        r = t[(t['fov'] == 1) & (t['cell'] == 2) & (t['hybe'] == 'Hyb_002')].iloc[0]
        check('the gate is inclusive at p_exist == gate', r['n'] == 2 and r['p_min'] == 0.5)
        r = t[(t['fov'] == 2) & (t['hybe'] == 'Hyb_002')]
        check('a source the FOV never stored reads zero with n_stored 0, not a missing row',
              len(r) == 2 and (r['n'] == 0).all() and (r['n_stored'] == 0).all() and np.isnan(r['p_min']).all())
        check('homeless spots are not counted for anybody',
              t[(t['fov'] == 1) & (t['hybe'] == 'Hyb_001')]['n'].sum() == 3)
        check('celltype rides along', set(t[t['fov'] == 2]['celltype']) == {'Unsynchronized', 'Nocodazole'})
        X, ct = CC.gene_table(t, {('RNA', 'Hyb_001', 635): 'GA', ('RNA', 'Hyb_002', 635): 'GB'}, metric='n')
        check('gene_table pivots the count column into cells x genes', list(X.columns) == ['GA', 'GB'] and len(X) == 4
              and X.loc[(1, 1), 'GA'] == 2 and X.loc[(1, 2), 'GB'] == 2)
        Xs, _ = CC.gene_table(t, {('RNA', 'Hyb_001', 635): 'GA', ('RNA', 'Hyb_002', 635): 'GB'}, metric='soft')
        check('and the soft column', abs(Xs.loc[(1, 1), 'GA'] - 1.7) < 1e-9)

        print('the placement capsule and the model file')
        rows1 = CC.capsule_rows({'theta': np.radians([10.0, 350.0]), 'R': [0.9, 0.4],
                                 'fit_z': [0.1, -2.5], 'bf_ring': [5.0, -1.0],
                                 'radius': [1.0, 0.4]}, [1, 2], [120, 40])
        check('capsule rows carry every column', all(set(r) == set(CC.CAPSULE_COLUMNS) for r in rows1)
              and rows1[1]['theta_deg'] == 350.0)
        A.write_fov_cellcycle(sp, 1, {'version': CC.CAPSULE_VERSION, 'model': 'm1', 'stamp': {}, 'rows': rows1})
        rows2 = CC.capsule_rows({'theta': np.radians([200.0]), 'R': [0.95], 'fit_z': [0.0],
                                 'bf_ring': [3.0], 'radius': [1.1]}, [2], [80])
        A.write_fov_cellcycle(sp, 2, {'version': CC.CAPSULE_VERSION, 'model': 'm1', 'stamp': {}, 'rows': rows2})
        back = A.read_fov_cellcycle(sp, 1)
        check('capsule round-trips', back['model'] == 'm1' and back['rows'] == rows1)
        check('a FOV without a capsule reads None', A.read_fov_cellcycle(sp, 3) is None)
        A.write_cellcycle_model(sp, {'version': 1, 'model': {'K': 2}, 'categories': [{'name': 'G1', 'start_deg': 300, 'end_deg': 60}]})
        m = A.read_cellcycle_model(sp)
        check('model file round-trips at the project level',
              m['model'] == {'K': 2} and A.read_cellcycle_model(os.path.join(dp, 'DNA')) == m)
        A.write_fov_cellcycle(sp, 1, {'version': CC.CAPSULE_VERSION, 'model': 'm2', 'stamp': {}, 'rows': rows1})
        check('a rewrite is seen at once (no stale cache)', A.read_fov_cellcycle(sp, 1)['model'] == 'm2')

        print('the population join and the gates')
        pop = popmod.Population.build(sp, [1, 2], jobs=1, mask_intensity=False)
        check('build attaches the placements', pop.cellcycle is not None and len(pop.cellcycle) == 3
              and set(pop.cellcycle['fov']) == {1, 2})
        check('summary mentions them', 'placed on the cell cycle' in pop.summary())
        # cells row order: (1,1) (1,2) (2,1) (2,2); (2,1) was never placed
        v = gate.PhaseRange(0, 360).values(pop)
        check('values follow the cells row order, NaN for the unplaced cell',
              np.allclose(v[[0, 1, 3]], [10.0, 350.0, 200.0]) and np.isnan(v[2]))
        check('an arc through zero', list(gate.PhaseRange(300, 60).mask(pop)) == [True, True, False, False])
        check('an arc that does not', list(gate.PhaseRange(100, 250).mask(pop)) == [False, False, False, True])
        check('lo == hi is the whole circle, still failing the unplaced cell',
              list(gate.PhaseRange(90, 90).mask(pop)) == [True, True, False, True])
        check('min_r demands the concentration', list(gate.PhaseRange(300, 60, min_r=0.5).mask(pop)) == [True, False, False, False])
        check('CycleRange on bf_ring', list(gate.CycleRange('bf_ring', lo=0).mask(pop)) == [True, False, False, True])
        check('CycleRange on radius (inside the ring)', list(gate.CycleRange('radius', hi=0.6).mask(pop)) == [False, True, False, False])
        try:
            gate.CycleRange('theta_deg')
            check('CycleRange refuses a non-verdict metric', False)
        except ValueError:
            check('CycleRange refuses a non-verdict metric', True)
        cond = gate.Condition([gate.CelltypeIn(['Unsynchronized']), gate.PhaseRange(300, 60, min_r=0.5),
                               gate.CycleRange('fit_z', lo=-2)])
        d = cond.to_dict()
        cond2 = gate.Condition.from_dict(d)
        check('the new predicates round-trip through plain data',
              cond2.to_dict() == d and list(cond2.mask(pop)) == [True, False, False, False])
        cats = CC.categorize(v, [{'name': 'G1', 'start_deg': 300, 'end_deg': 60},
                                 {'name': 'G2M', 'start_deg': 100, 'end_deg': 250}])
        check('categories from arcs; unplaced and uncovered read empty', list(cats) == ['G1', 'G1', '', 'G2M'])
        print("the stage's arcs and gates become the category at read time")
        cat0 = pop.cellcycle.set_index(['fov', 'cell'])['category']
        check('with the stored arc and no gates the category follows the arc alone',
              cat0[(1, 1)] == 'G1' and cat0[(1, 2)] == 'G1' and cat0[(2, 2)] == '', str(dict(cat0)))
        A.write_cellcycle_model(sp, {'version': 1, 'model': {'K': 2},
                                     'categories': [{'name': 'G1', 'start_deg': 300, 'end_deg': 60},
                                                    {'name': 'G2M', 'start_deg': 100, 'end_deg': 250}],
                                     'gates': {'min_R': 0.5, 'min_bf_ring': 0.0}})
        pop = popmod.Population.build(sp, [1, 2], jobs=1, mask_intensity=False)
        cat = pop.cellcycle.set_index(['fov', 'cell'])['category']
        check('arcs name the cells that pass every gate; a failed gate reads unassigned',
              cat[(1, 1)] == 'G1' and cat[(1, 2)] == '' and cat[(2, 2)] == 'G2M', str(dict(cat)))
        check('CycleCategoryIn gates like CelltypeIn, never placed matches nothing',
              list(gate.CycleCategoryIn(['G1']).mask(pop)) == [True, False, False, False]
              and list(gate.CycleCategoryIn(['Unassigned']).mask(pop)) == [False, True, False, False]
              and list(gate.CycleCategoryIn(['G1', 'G2M']).mask(pop)) == [True, False, False, True])
        d = gate.Condition([gate.CycleCategoryIn(['G2M'])]).to_dict()
        check('CycleCategoryIn round-trips', list(gate.Condition.from_dict(d).mask(pop)) == [False, False, False, True])
        a = CC.assign(pd.DataFrame({'theta_deg': [10.0, 10.0, np.nan], 'R': [0.9, 0.3, 0.9], 'fit_z': [0.0, 0.0, 0.0]}),
                      [{'name': 'G1', 'start_deg': 300, 'end_deg': 60}], {'min_R': 0.5, 'min_bf_ring': 0.0})
        check('assign: a gated verdict the table lacks fails every cell', list(a) == ['', '', ''])
        a = CC.assign(pd.DataFrame({'theta_deg': [10.0, 10.0, np.nan], 'R': [0.9, 0.3, 0.9]}),
                      [{'name': 'G1', 'start_deg': 300, 'end_deg': 60}], {'min_R': 0.5})
        check('assign: gates and NaN phases', list(a) == ['G1', '', ''])
        print('DAPI inside the mask')
        # a MIP with a bright square where cell 1 sits and background elsewhere
        mip = np.full((1024, 1024), 100.0)
        mip[4:8, 6:10] = 400.0
        A.write_hybe_mip(sp, 1, 'Hyb_500', {405: mip.astype(np.uint16)})
        tab_d, fails_d = CC.mask_intensity_table(sp, [1], ('RNA', 'Hyb_500', 405), jobs=1)
        c1 = tab_d[tab_d['cell'] == 1].iloc[0] if len(tab_d) else None
        check('mask intensity rows for the FOV, no failure', fails_d == [] and len(tab_d) == 2, f'{fails_d} {len(tab_d)}')
        check('the cell on the bright square sums above the background, the other does not',
              c1 is not None and c1['sum_above_bg'] > 0 and float(tab_d[tab_d['cell'] == 2]['sum_above_bg'].iloc[0]) == 0.0,
              str(tab_d[['cell', 'area', 'mask_mean', 'sum_above_bg', 'background']].to_dict('records')))
        crops = CC.gallery_crops(sp, ('RNA', 'Hyb_500', 405), {1: [1, 2]}, size=16, jobs=1)
        check('gallery crops: one tile per wanted cell, the mask inside, on the FOV scale',
              set(crops) == {(1, 1), (1, 2)} and crops[(1, 1)][0].shape == (16, 16) and crops[(1, 1)][1].sum() == 2
              and crops[(1, 1)][2] <= crops[(1, 1)][3], str({k: (v[0].shape, int(v[1].sum())) for k, v in crops.items()}))
        empty = popmod.Population(sp, [1], (0.208, 0.208, 0.2), pop.cells, None, None, None, [])
        try:
            gate.PhaseRange(0, 90).mask(empty)
            check('a population without placements refuses with a message', False)
        except ValueError as e:
            check('a population without placements refuses with a message', 'Cell Cycle stage' in str(e))
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        print('FAILED:', FAIL)
        sys.exit(1)
    print('ALL GOOD')


if __name__ == '__main__':
    main()
