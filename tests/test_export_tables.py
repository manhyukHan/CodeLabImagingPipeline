"""
Flat spot/allele export: the shape of the tables, and the one thing that
would silently corrupt every downstream analysis if it were wrong.

THE COORDINATE MAPPING IS THE POINT. Everything inside this pipeline is
(y, x, z); every column in these files is named _x/_y/_z. If that swap is
ever inverted the files still look perfectly reasonable -- same numbers,
same ranges, plausible plots -- and every distance computed from them is
wrong. So it is asserted directly, with y and x deliberately far apart.

Also pinned: the allele table is LONG (one row per allele x bin x
candidate), a bin with no accepted readout still gets a row (a missing
bin is a result), is_selected marks exactly one candidate per bin, raw
candidates are never paired positionally with adj ones unless the two
lists agree in length.

Run: python tests/test_export_tables.py
"""
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                   # noqa: E402

from codelab_pipeline.analysis import export as E    # noqa: E402

PASSED, FAILED = [], []


def check(label, cond, detail=''):
    (PASSED if cond else FAILED).append(label)
    print(('  ok   ' if cond else '  FAIL ') + label + ('' if cond else f'   {detail}'))


# (y, x, z) with y and x far apart, so a swap cannot hide
SPOT = {'uid': 5, 'fov': 3, 'modality': 'DNA', 'hybe': 'Hyb_015', 'channel': 555,
        'cell': 7, 'celltype': 'stale', 'size': 2.5, 'brightness': 900.0,
        'raw_coordinate': (10.0, 800.0, 40.0),
        'adj_coordinate': (11.0, 801.0, 41.0),
        'mixture_centroids': ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
        'linked': True, 'linked_at': 'now'}


def test_spot_row_coordinate_mapping():
    rows = E.spot_rows([SPOT], fov=3, celltype_of={7: 'real'},
                       voxel_um=(0.208, 0.208, 0.2))
    r = rows[0]
    check('one row per spot', len(rows) == 1)
    check('y (index 0) lands in _y, NOT _x',
          r['raw_y'] == 10.0 and r['raw_x'] == 800.0,
          f"x={r['raw_x']} y={r['raw_y']}")
    check('z (index 2) lands in _z', r['raw_z'] == 40.0)
    check('adj coordinates map the same way',
          r['adj_y'] == 11.0 and r['adj_x'] == 801.0 and r['adj_z'] == 41.0)
    check('um scaling is PER AXIS and applied after the swap',
          abs(r['raw_x_um'] - 800.0 * 0.208) < 1e-9
          and abs(r['raw_z_um'] - 40.0 * 0.2) < 1e-9,
          f"x_um={r['raw_x_um']} z_um={r['raw_z_um']}")
    check("the CELL's celltype wins over the spot's own stale copy",
          r['celltype'] == 'real', r['celltype'])
    check('mixture candidates are counted, not flattened into columns',
          r['n_mixture_candidates'] == 2)
    check('every declared column is present',
          set(E.SPOT_COLUMNS) <= set(r), str(set(E.SPOT_COLUMNS) - set(r)))


def _allele(**over):
    a = {'id': 42, 'fov': 3, 'cell': 7, 'anchor_uid': 5,
         'anchor_hybe': 'Hyb_015', 'anchor_channel': 555,
         'coordinate': (10.0, 800.0, 40.0),
         'raw_coordinate': (12.0, 802.0, 42.0),
         'fiducial_trace_adj': {'Hyb_015': (1.0, 2.0, 3.0, 111.0)},
         'fiducial_trace_raw': {'Hyb_015': (4.0, 5.0, 6.0, 222.0)},
         'polymer_adj': {'Hyb_015': [(20.0, 900.0, 50.0, 10.0),
                                     (21.0, 901.0, 51.0, 99.0)]},
         'polymer_raw': {'Hyb_015': [(22.0, 902.0, 52.0, 10.0),
                                     (23.0, 903.0, 53.0, 99.0)]},
         'rejected_hybes': {'Hyb_017': 'no fiducial peak'},
         'final_polymer': np.empty((0, 3))}
    a.update(over)
    return a


BINS = [(15, 'Hyb_015'), (16, 'Hyb_016'), (17, 'Hyb_017')]


def test_allele_long_format():
    rows = E.allele_rows([_allele()], fov=3, bins=BINS, modality='DNA',
                         celltype_of={7: 'real'})
    check('every bin appears, even ones the allele never saw',
          {r['bin_index'] for r in rows} == {0, 1, 2},
          str(sorted({r['bin_index'] for r in rows})))
    check('a 2-candidate bin yields 2 rows, the empty bins 1 each',
          len(rows) == 4, f'{len(rows)} rows')
    check('all rows share one allele_id',
          {r['allele_id'] for r in rows} == {42})
    b0 = [r for r in rows if r['bin_index'] == 0]
    check('readout candidates keep the (y,x,z)->_x/_y/_z mapping',
          b0[0]['readout_adj_y'] == 20.0 and b0[0]['readout_adj_x'] == 900.0,
          f"x={b0[0]['readout_adj_x']} y={b0[0]['readout_adj_y']}")
    check('the 4th element is amplitude, not a coordinate',
          b0[0]['readout_adj_amplitude'] == 10.0
          and b0[1]['readout_adj_amplitude'] == 99.0)
    check('raw candidates pair by index with adj ones',
          b0[0]['readout_raw_y'] == 22.0 and b0[1]['readout_raw_y'] == 23.0,
          f"{b0[0]['readout_raw_y']} {b0[1]['readout_raw_y']}")
    check('fiducial adj and raw are separate columns, with amplitude',
          b0[0]['fiducial_adj_y'] == 1.0 and b0[0]['fiducial_adj_amplitude'] == 111.0
          and b0[0]['fiducial_raw_y'] == 4.0 and b0[0]['fiducial_raw_amplitude'] == 222.0)
    check('the anchor is carried on every row of the allele',
          all(r['anchor_adj_y'] == 10.0 and r['anchor_adj_x'] == 800.0
              for r in rows))
    finals = [r for r in b0 if r['is_selected']]
    check('exactly ONE candidate per bin is marked final', len(finals) == 1)
    check('and it is the BRIGHTEST one (the collapse rule)',
          finals[0]['readout_adj_amplitude'] == 99.0)
    check('selection_rule says the position was derived, not stored',
          finals[0]['selection_rule'] == 'computed', finals[0]['selection_rule'])
    rej = [r for r in rows if r['bin_index'] == 2][0]
    check('a rejected bin carries its reason',
          rej['rejected_reason'] == 'no fiducial peak', rej['rejected_reason'])
    check('and blank coordinates rather than a fabricated position',
          np.isnan(rej['readout_adj_x']) and rej['candidate_index'] == -1)
    check('every declared column is present',
          set(E.ALLELE_COLUMNS) <= set(rows[0]),
          str(set(E.ALLELE_COLUMNS) - set(rows[0])))


def test_stored_final_polymer_wins():
    fp = np.array([[70.0, 700.0, 7.0], [0, 0, 0], [0, 0, 0]])
    rows = E.allele_rows([_allele(final_polymer=fp)], fov=3, bins=BINS,
                         modality='DNA')
    b0 = [r for r in rows if r['bin_index'] == 0][0]
    check('a stored final_polymer is used as-is',
          b0['final_y'] == 70.0 and b0['final_x'] == 700.0,
          f"x={b0['final_x']} y={b0['final_y']}")
    check('and is labelled stored', b0['selection_rule'] == 'stored')
    check('no candidate matches it, so none is marked final',
          not any(r['is_selected'] for r in rows if r['bin_index'] == 0))


def test_mismatched_candidate_lists_are_not_paired():
    a = _allele(polymer_raw={'Hyb_015': [(22.0, 902.0, 52.0, 10.0)]})
    rows = [r for r in E.allele_rows([a], fov=3, bins=BINS, modality='DNA')
            if r['bin_index'] == 0]
    check('a length mismatch blanks raw rather than mis-pairing it',
          all(np.isnan(r['readout_raw_x']) for r in rows),
          str([r['readout_raw_x'] for r in rows]))
    check('adj candidates are still exported in full',
          [r['readout_adj_amplitude'] for r in rows] == [10.0, 99.0])


def test_layout_join_and_channel_role():
    """The join that makes the file readable -- and the one that stops a
    newcomer reading alignment beads as signal."""
    layout = {'DNA': {'Hyb_015': {'folder': 'Hyb_015', 'readout_id': 15,
                                  'datatype': 'H', 'fiducial_channel': 555,
                                  'readout_name': 'Gorab_exon'}}}
    fid = E.spot_rows([SPOT], layout_by_modality=layout)[0]
    check('a spot in the fiducial channel is labelled fiducial',
          fid['channel_role'] == 'fiducial', fid['channel_role'])
    readout = E.spot_rows([dict(SPOT, channel=635)],
                          layout_by_modality=layout)[0]
    check('a spot in another channel is labelled readout',
          readout['channel_role'] == 'readout', readout['channel_role'])
    check('the layout supplies the human-readable name',
          fid['readout_name'] == 'Gorab_exon' and fid['datatype'] == 'H'
          and fid['readout_id'] == 15)
    # -- the per-spot quality: what every count in this project is gated on ---
    q = E.spot_rows([dict(SPOT, p_exist=0.87, z_status='accepted')])[0]
    check('p_exist is exported as a number', q['p_exist'] == 0.87, str(q['p_exist']))
    check('z_status is exported', q['z_status'] == 'accepted', str(q['z_status']))
    check('a spot with a calibrated p_exist and a fitted z is marked learned',
          q['engine'] == 'learned', str(q['engine']))
    old = E.spot_rows([SPOT])[0]
    check('a spot from before the field reads NaN, not an error',
          old['p_exist'] != old['p_exist'], str(old['p_exist']))
    check('and its z_status reads not_fit rather than empty',
          old['z_status'] == 'not_fit', str(old['z_status']))
    check('with no calibrated probability it is not called learned', old['engine'] == '')
    v2 = E.spot_rows([dict(SPOT, p_exist=float('nan'), z_status='accepted')])[0]
    check('a fitted z alone does not make it learned', v2['engine'] == '')

    bare = E.spot_rows([SPOT])[0]
    check('without a layout the columns are blank, not wrong',
          bare['readout_name'] == '' and bare['channel_role'] == '',
          f"{bare['readout_name']!r} {bare['channel_role']!r}")


def test_fov_comes_from_the_spot_not_the_caller():
    r = E.spot_rows([SPOT], fov=999)[0]
    check("the spot's own fov wins over a caller's wrong one",
          r['fov'] == 3 and r['spot_key'] == 'F003-U5',
          f"fov={r['fov']} key={r['spot_key']}")
    check('cell_key is FOV-scoped too', r['cell_key'] == 'F003-C7',
          r['cell_key'])
    homeless = E.spot_rows([dict(SPOT, cell=-1)])[0]
    check('a homeless spot is kept, with a blank cell_key',
          homeless['cell'] == -1 and homeless['cell_key'] == '')


def test_qc_rounds_are_not_discarded():
    """polymer_adj also holds R/T rounds; a bin-only export loses them."""
    a = _allele(polymer_adj={'Hyb_015': [(20.0, 900.0, 50.0, 10.0)],
                             'Rep_032': [(30.0, 930.0, 60.0, 12.0)]},
                polymer_raw={'Hyb_015': [(22.0, 902.0, 52.0, 10.0)],
                             'Rep_032': [(32.0, 932.0, 62.0, 12.0)]})
    recs = {'Rep_032': {'folder': 'Rep_032', 'readout_id': 32, 'datatype': 'R'}}
    rows = E.allele_rows([a], fov=3, bins=BINS, modality='DNA',
                         records_by_hybe=recs)
    qc = [r for r in rows if r['hybe'] == 'Rep_032']
    check('a traced QC round still gets a row', len(qc) == 1)
    check('marked bin_index -1 -- a measurement, not a genomic bin',
          qc and qc[0]['bin_index'] == -1)
    check('and carries its datatype so it can be filtered out',
          qc and qc[0]['datatype'] == 'R', str(qc and qc[0]['datatype']))
    check('its coordinates are still exported',
          qc and qc[0]['readout_adj_y'] == 30.0)


def test_modality_prefers_provenance():
    a = _allele(provenance={'modality': 'RNA'})
    r = E.allele_rows([a], fov=3, bins=BINS, modality='DNA')[0]
    check('provenance beats the caller-supplied modality',
          r['modality'] == 'RNA' and r['modality_source'] == 'provenance',
          f"{r['modality']} {r['modality_source']}")
    r2 = E.allele_rows([_allele()], fov=3, bins=BINS, modality='DNA')[0]
    check('without provenance the label is marked assumed',
          r2['modality'] == 'DNA' and r2['modality_source'] == 'assumed')


def test_modalities_sharing_one_analysis_dir_are_read_once():
    """The bug this guards against doubled every count in the file.

    analysis/ is SHARED by every modality of a project -- analysis_dir()
    of '<proj>/DNA' and of '<proj>/RNA' is the same directory -- so
    reading per modality returns the identical spots twice. Modality is a
    property of each spot, not of the path it was read through.
    """
    from unittest import mock
    from codelab_pipeline.io import analysis_store

    calls = []

    def fake_read_spots(sp, fov, **kw):
        calls.append((sp, fov))
        return [dict(SPOT, fov=fov)]

    with mock.patch.object(analysis_store, 'read_spots', side_effect=fake_read_spots), \
         mock.patch.object(analysis_store, 'read_cells', return_value=([], '')), \
         mock.patch.object(analysis_store, 'read_fov_alleles', return_value=[]), \
         mock.patch.object(E.polymer, 'records_for', return_value=[]):
        t = E.collect({'DNA': '/proj/DNA', 'RNA': '/proj/RNA'},
                      fovs=[1, 2])
    summary = t.summary

    check('each FOV is read ONCE, not once per modality',
          len(calls) == 2, f'{len(calls)} reads: {calls}')
    check('so the row count is not doubled',
          summary['n_spots'] == 2, str(summary['n_spots']))
    check('and every FOV still appears', sorted(summary['fovs']) == [1, 2],
          str(summary['fovs']))


def test_writers_are_atomic_and_round_trip():
    d = tempfile.mkdtemp(prefix='exporttest_')
    try:
        rows = E.spot_rows([SPOT], fov=3)
        p = E.write_csv(rows, E.SPOT_COLUMNS, os.path.join(d, 'spots.csv'))
        check('csv written', os.path.exists(p))
        check('no .part left behind',
              not [f for f in os.listdir(d) if '.part' in f],
              str(os.listdir(d)))
        import pandas as pd
        back = pd.read_csv(p)
        check('csv round-trips every column',
              list(back.columns) == list(E.SPOT_COLUMNS))
        check('csv preserves the coordinate mapping',
              back['raw_x'][0] == 800.0 and back['raw_y'][0] == 10.0)
        xp = E.write_excel([('spots', rows, E.SPOT_COLUMNS)],
                           os.path.join(d, 'x.xlsx'))
        check('xlsx written with no .part left',
              os.path.exists(xp)
              and not [f for f in os.listdir(d) if '.part' in f])
        back2 = pd.read_excel(xp, sheet_name='spots')
        check('xlsx round-trips the coordinate mapping',
              back2['raw_x'][0] == 800.0 and back2['raw_y'][0] == 10.0)
        # too many rows for Excel must REFUSE, not truncate
        class _Many(list):
            def __len__(self):
                return E.EXCEL_MAX_ROWS + 1
        try:
            E.write_excel([('big', _Many(rows), E.SPOT_COLUMNS)],
                          os.path.join(d, 'big.xlsx'))
            check('an over-long sheet is refused, not silently truncated', False,
                  'no error raised')
        except ValueError:
            check('an over-long sheet is refused, not silently truncated', True)
    finally:
        shutil.rmtree(d, ignore_errors=True)


CELL = {
    'id': 7, 'fov': 3, 'celltype': 'real',
    'reference_hybe': 'Hyb_IFdapi', 'reference_modality': 'DNA',
    'nucleus_hybe': 'Hyb_IFdapi', 'nucleus_modality': 'DNA',
    # y and x far apart again, for the same reason as SPOT
    'nucleus': (np.array([10.0, 11.0, 12.0]), np.array([800.0, 801.0, 802.0])),
    'area': (np.array([9.0, 13.0]), np.array([799.0, 803.0])),
    'frame_shape': (1024, 1024),
    'matrices': {('Hyb_IFdapi', 'DNA'): {'yx': np.eye(3), 'dz': 0.0,
                                         'yx_is_residual': True},
                 ('Hyb_015', 'DNA'): {'yx': np.array([[1.0, 0.0, -2.0],
                                                      [0.0, 1.0, -5.0],
                                                      [0.0, 0.0, 1.0]]),
                                      'dz': -3.0, 'yx_is_residual': True}},
    'matrix_anchors': {'DNA': np.eye(3)},
    'matrix_provenance': {},
    'distmap': np.zeros(4),
    'linked': False, 'linked_at': None,
}


def test_cell_row_shape_and_masks():
    arrays = {}
    rows = E.cell_rows([CELL], fov=3, voxel_um=(0.208, 0.208, 0.2),
                       n_spots_of={7: 11}, n_readout_spots_of={7: 4},
                       n_alleles_of={7: 2}, arrays=arrays)
    r = rows[0]
    check('one row per cell', len(rows) == 1)
    check('cell_key matches the one the spot row builds',
          r['cell_key'] == 'F003-C7', r['cell_key'])
    check('the nucleus centroid takes y from index 0, x from index 1',
          r['nucleus_centroid_y'] == 11.0
          and r['nucleus_centroid_x'] == 801.0,
          f"x={r['nucleus_centroid_x']} y={r['nucleus_centroid_y']}")
    check('nucleus and area are counted SEPARATELY',
          r['nucleus_px'] == 3 and r['area_px'] == 2,
          f"nucleus={r['nucleus_px']} area={r['area_px']}")
    check('area in um2 uses both in-plane axes',
          abs(r['nucleus_um2'] - 3 * 0.208 * 0.208) < 1e-12,
          str(r['nucleus_um2']))
    check('the counts come from the caller, not the cell',
          (r['n_spots'], r['n_spots_readout'], r['n_alleles']) == (11, 4, 2))
    check('a cell away from every border does not read as touching it',
          r['touches_frame_edge'] is False, str(r['touches_frame_edge']))


def test_cell_alignment_is_stored_not_resolved():
    rows = E.cell_rows([CELL], fov=3)
    r = rows[0]
    check('the home hybe is named, so a reader can resolve against it',
          r['home_hybe'] == 'Hyb_IFdapi' and r['home_modality'] == 'DNA')
    check('and the row SAYS the stored matrix is a bare residual',
          r['home_yx_is_residual'] is True)
    check('every aligned hybe is counted', r['n_hybes_aligned'] == 2)
    # the decomposition itself, on the one entry that is not identity
    th, dx, dy, dz, sc = E._affine_parts(
        CELL['matrices'][('Hyb_015', 'DNA')]['yx'], -3.0)
    check('a translation decomposes to dx/dy in OUTPUT order, not y-major',
          dx == -5.0 and dy == -2.0, f'dx={dx} dy={dy}')
    check('a pure translation has no rotation and unit scale',
          abs(th) < 1e-12 and abs(sc - 1.0) < 1e-12, f'theta={th} scale={sc}')
    check('z travels beside the matrix, never inside it', dz == -3.0)
    # a rotation, so the decomposition is not only exercised on identity
    import math
    a = math.radians(30.0)
    R = np.array([[math.cos(a), -math.sin(a), 0.0],
                  [math.sin(a), math.cos(a), 0.0],
                  [0.0, 0.0, 1.0]])
    th2, _dx2, _dy2, _dz2, sc2 = E._affine_parts(R, 0.0)
    check('a rotation comes back in degrees', abs(th2 - 30.0) < 1e-9,
          str(th2))
    check('and is not mistaken for a scale', abs(sc2 - 1.0) < 1e-12,
          str(sc2))
    check('a missing matrix is NaN, not identity',
          E._affine_parts(None)[0] != E._affine_parts(None)[0])


def test_unknown_fields_survive_the_export():
    """The whole point: export.py must not be a list of one project's
    fields. Nothing below is known to it."""
    c = dict(CELL)
    c['n_speckle'] = 23
    c['speckle_frac'] = 0.064
    c['genotype'] = 'C'
    c['adj_nucleus'] = (np.array([10.5]), np.array([800.5]))
    arrays = {}
    rows = E.cell_rows([c], fov=3, arrays=arrays)
    r = rows[0]
    check('a scalar field export.py never heard of becomes a column',
          r.get('n_speckle') == 23 and r.get('genotype') == 'C',
          str({k: r.get(k) for k in ('n_speckle', 'genotype')}))
    cols = E.columns_for(rows, E.CELL_COLUMNS)
    check('and columns_for puts it after the curated ones, not instead',
          cols[:len(E.CELL_COLUMNS)] == tuple(E.CELL_COLUMNS)
          and 'n_speckle' in cols)
    check('a non-scalar one goes to the arrays sidecar',
          'F003-C7/adj_nucleus' in arrays, str(sorted(arrays)[:4]))
    check('WITHOUT overwriting the original mask',
          'F003-C7/nucleus_y' in arrays and 'F003-C7/nucleus_x' in arrays)
    check('and the row says what was kept beside it',
          'adj_nucleus' in r['arrays'] and 'nucleus' in r['arrays'],
          r['arrays'])


def test_combined_joins_cell_onto_spot():
    class _T(object):
        pass
    t = _T()
    # a project field on each side; the cache must keep BOTH
    t.cells = E.cell_rows([dict(CELL, n_speckle=23)], fov=3,
                          n_spots_of={7: 2})
    homeless = dict(SPOT, uid=6, cell=-1)
    t.spots = E.spot_rows([dict(SPOT, d_surface_um=0.37), homeless], fov=3)
    rows = E.combined_rows(t)
    check('one row per SPOT, not per cell', len(rows) == 2)
    r = [x for x in rows if x['cell'] == 7][0]
    check('the cell fields arrive prefixed',
          r.get('cell_nucleus_um2') is not None,
          str(sorted(k for k in r if k.startswith('cell_'))[:6]))
    check('fov/celltype are NOT duplicated -- one column, from the spot',
          'cell_fov' not in r and 'cell_celltype' not in r)
    # the compact rule: the project's own fields are what the cache is FOR
    check('a field the schema does not declare is KEPT, both sides',
          r.get('d_surface_um') == 0.37 and r.get('cell_n_speckle') == 23,
          str((r.get('d_surface_um'), r.get('cell_n_speckle'))))
    check('while the pipeline bookkeeping already in the other sheets '
          'is dropped',
          'cell_nucleus_px' not in r and 'raw_x' not in r
          and 'cell_home_yx_dx' not in r,
          str([k for k in ('cell_nucleus_px', 'raw_x', 'cell_home_yx_dx')
               if k in r]))
    full = E.combined_rows(t, compact=False)
    r2 = [x for x in full if x['cell'] == 7][0]
    check('compact=False still gives the whole join',
          r2.get('cell_nucleus_px') == 3 and r2.get('raw_x') == 800.0)
    h = [x for x in rows if x['cell'] == -1][0]
    check('a homeless spot keeps its row',
          h['spot_key'].endswith('U6') and 'cell_nucleus_um2' not in h)


def main():
    for fn in (test_spot_row_coordinate_mapping,
               test_layout_join_and_channel_role,
               test_fov_comes_from_the_spot_not_the_caller,
               test_allele_long_format,
               test_qc_rounds_are_not_discarded,
               test_modality_prefers_provenance,
               test_modalities_sharing_one_analysis_dir_are_read_once,
               test_stored_final_polymer_wins,
               test_mismatched_candidate_lists_are_not_paired,
               test_writers_are_atomic_and_round_trip,
               test_cell_row_shape_and_masks,
               test_cell_alignment_is_stored_not_resolved,
               test_unknown_fields_survive_the_export,
               test_combined_joins_cell_onto_spot):
        print(f'\n{fn.__name__}')
        fn()
    print(f'\n{len(PASSED)} passed, {len(FAILED)} failed')
    if FAILED:
        for f in FAILED:
            print('  FAILED:', f)
        return 1
    print('ALL GOOD')
    return 0


if __name__ == '__main__':
    sys.exit(main())
