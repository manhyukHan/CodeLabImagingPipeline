"""
The storage contract for traced spots, end to end:

    every spot is (y, x, z, amplitude, quality)   -- AnAllele's docstring
    a fiducial that was looked for and refused is None under its key
    fiducial_drift[hybe] = fiducial(reference) - fiducial(hybe), or None

and every reader -- the columnar store (new files and files written
before the quality slot existed), reconcile, export, the polymer
collapse -- sees one width.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_spot_quality_storage.py
"""
import math
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from codelab_pipeline.models.allele import AnAllele, SPOT_WIDTH        # noqa: E402
from codelab_pipeline.io import columnar                               # noqa: E402
from codelab_pipeline.localization import tracing_v2 as T2             # noqa: E402
from codelab_pipeline.localization import fiducial_xcorr as FX         # noqa: E402
from codelab_pipeline.analysis import reconcile as REC                 # noqa: E402
from codelab_pipeline.analysis import export as E                      # noqa: E402
from codelab_pipeline.analysis import polymer as POLY                  # noqa: E402

CHECKS = [0, 0]


def check(name, ok, note=''):
    CHECKS[0] += 1
    CHECKS[1] += 1 if ok else 0
    print(('  ok   ' if ok else '  FAIL ') + name + (('  -- ' + note) if note else ''))


def isnan(v):
    return isinstance(v, float) and math.isnan(v)


def same(a, b, tol=1e-9):
    """Tuple equality with NaN == NaN."""
    if a is None or b is None:
        return a is b
    if len(a) != len(b):
        return False
    return all((isnan(x) and isnan(y)) or (not isnan(x) and not isnan(y)
                                            and abs(float(x) - float(y)) <= tol)
               for x, y in zip(a, b))


def test_allele_widening():
    print('AnAllele: one width, quality NaN where there was none')
    check('SPOT_WIDTH is 5', SPOT_WIDTH == 5)
    a = AnAllele()
    a.set_metadata(fiducial_trace_adj={'A': (1.0, 2.0, 3.0, 4.0), 'B': None},
                   fiducial_trace_raw={'A': (5.0, 6.0, 7.0, 8.0, 0.9)},
                   polymer_adj={'A': [(1.0, 2.0, 3.0, 4.0), (1.5, 2.5, 3.5, 4.5, 0.7)]},
                   fiducial_drift={'A': (0.5, -0.5, 1.0), 'B': None})
    t = a.fiducial_trace_adj['A']
    check('a 4-wide fiducial loads 5 wide with quality NaN',
          len(t) == 5 and t[:4] == (1.0, 2.0, 3.0, 4.0) and isnan(t[4]), str(t))
    check('None stays None (looked for, refused)', a.fiducial_trace_adj['B'] is None)
    check('a 5-wide fiducial is kept as it is',
          a.fiducial_trace_raw['A'] == (5.0, 6.0, 7.0, 8.0, 0.9))
    c = a.polymer_adj['A']
    check('polymer candidates widen the same way, each on its own',
          len(c) == 2 and len(c[0]) == 5 and isnan(c[0][4]) and c[1][4] == 0.7, str(c))
    check('fiducial_drift is 3 wide and keeps None',
          a.fiducial_drift == {'A': (0.5, -0.5, 1.0), 'B': None}, str(a.fiducial_drift))
    d = a.save()
    check('save() carries fiducial_drift and the full width',
          'fiducial_drift' in d and d['fiducial_drift']['B'] is None
          and len(d['polymer_adj']['A'][0]) == 5 and len(d['fiducial_trace_adj']['A']) == 5)
    b = AnAllele()
    b.set_metadata(**d)
    check('save -> set_metadata round trip',
          same(b.fiducial_trace_adj['A'], a.fiducial_trace_adj['A'])
          and b.fiducial_trace_adj['B'] is None
          and same(b.polymer_adj['A'][1], a.polymer_adj['A'][1])
          and b.fiducial_drift == a.fiducial_drift)
    s = AnAllele()
    s.set_metadata(polymer_adj={'A': ['kept']})
    check("a staged marker (tests use ['kept']) passes through untouched",
          s.polymer_adj == {'A': ['kept']})
    check('a fresh allele has an empty fiducial_drift', AnAllele().fiducial_drift == {})


def test_columnar_round_trip_and_legacy():
    print('\ncolumnar store: new files, and files written with 4 columns')
    import h5py
    a = AnAllele()
    a.set_metadata(id=1, fov=2, cell=3, anchor_hybe='H1', anchor_channel=555,
                   coordinate=(1.0, 2.0, 3.0), raw_coordinate=(1.0, 2.0, 3.0),
                   fiducial_trace_adj={'H1': (1.0, 2.0, 3.0, 4.0, 0.8), 'H2': None},
                   fiducial_trace_raw={'H1': (1.5, 2.5, 3.5, 4.5, 0.8), 'H2': None},
                   polymer_adj={'H1': [(1.0, 2.0, 3.0, 4.0, 0.66), (2.0, 3.0, 4.0, 5.0)]},
                   polymer_raw={'H1': [(1.1, 2.1, 3.1, 4.1, 0.66), (2.1, 3.1, 4.1, 5.1)]},
                   fiducial_drift={'H1': (0.0, 0.0, 0.0), 'H2': None},
                   rejected_hybes={'H2': 'fiducial ncc 0.31 < 0.50'})
    d = a.save()
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'alleles.h5')
    with h5py.File(path, 'w') as f:
        g = f.create_group('alleles')
        columnar.pack_alleles(g, [d])
        out = columnar.unpack_alleles(g)[0]
        check('pl_vals is written 5 wide', g['pl_vals'].shape[1] == 5)
        check('fd_* datasets exist', all(k in g for k in ('fd_allele', 'fd_hybe', 'fd_isnone', 'fd_vals')))
        # a file written BEFORE the quality slot: the same group with
        # every *_vals sliced to 4 columns and no fd_* datasets
        f.copy(g, 'legacy')
        g2 = f['legacy']
        for name in ('fd_allele', 'fd_hybe', 'fd_isnone', 'fd_vals'):
            del g2[name]
        for name in ('tr_vals', 'trr_vals', 'pl_vals', 'plr_vals'):
            data = g2[name][()][:, :4]
            del g2[name]
            g2.create_dataset(name, data=data)
        old = columnar.unpack_alleles(g2)[0]
    check('polymer candidates round-trip 5 wide, NaN where quality was absent',
          same(out['polymer_adj']['H1'][0], (1.0, 2.0, 3.0, 4.0, 0.66))
          and same(out['polymer_adj']['H1'][1], (2.0, 3.0, 4.0, 5.0, float('nan')))
          and same(out['polymer_raw']['H1'][0], (1.1, 2.1, 3.1, 4.1, 0.66)),
          str(out['polymer_adj']))
    check('fiducials round-trip with their quality, None kept under its key',
          same(out['fiducial_trace_adj']['H1'], (1.0, 2.0, 3.0, 4.0, 0.8))
          and out['fiducial_trace_adj']['H2'] is None
          and out['fiducial_trace_raw']['H2'] is None, str(out['fiducial_trace_adj']))
    check('fiducial_drift round-trips, None included',
          out['fiducial_drift'] == {'H1': (0.0, 0.0, 0.0), 'H2': None}, str(out['fiducial_drift']))
    check('rejected reason rides along', out['rejected_hybes'] == {'H2': 'fiducial ncc 0.31 < 0.50'})
    check('a 4-column file reads WIDENED: quality NaN, drift empty',
          len(old['polymer_adj']['H1'][0]) == 5 and isnan(old['polymer_adj']['H1'][0][4])
          and old['polymer_adj']['H1'][0][:4] == (1.0, 2.0, 3.0, 4.0)
          and len(old['fiducial_trace_adj']['H1']) == 5 and isnan(old['fiducial_trace_adj']['H1'][4])
          and old['fiducial_trace_adj']['H2'] is None
          and old['fiducial_drift'] == {},
          f"{old['polymer_adj']} {old['fiducial_trace_adj']} {old['fiducial_drift']}")


def test_fiducial_quality():
    print('\nfiducial_quality: the number the method gates on')
    lf = T2.LearnedFit(1.0, 2.0, 3.0, 100.0, 0.72, ())
    check('a learned fiducial stores p_exist', T2.fiducial_quality(lf) == 0.72)
    xf = FX.XcorrFit(1.0, 2.0, 3.0, 100.0, 0.83, (0.0, 0.0, 0.0), (), 'cc')
    check('a correlation fiducial stores its peak NCC', T2.fiducial_quality(xf) == 0.83)

    class G(object):
        y, x, z, amplitude = 1.0, 2.0, 3.0, 100.0
    check('a Gaussian fit stores NaN', isnan(T2.fiducial_quality(G())))
    lf2 = T2.LearnedFit(1.0, 2.0, 3.0, 100.0, float('nan'), ())
    check('a NaN p_exist stays NaN, never 0', isnan(T2.fiducial_quality(lf2)))


def test_readout_writes_p_exist():
    print('\nthe learned readout writes p_exist in the fifth slot')
    from test_v3_in_tracing import _run, FakeEngine, spot, _slab_z0
    z0 = _slab_z0(20.0, T2.V2Params().z_reach())
    eng = FakeEngine([spot(8.0, 8.0, 20.0 - z0, 0.91), spot(9.0, 8.5, 21.0 - z0, 0.63)])
    a, ok, why = _run(eng, min_p_exist=0.5)
    cands = a.polymer_adj.get('H') or []
    check('two candidates kept', ok and len(cands) == 2, str(why))
    check('each carries its own p_exist as quality',
          all(len(c) == 5 for c in cands)
          and sorted(round(c[4], 2) for c in cands) == [0.63, 0.91], str(cands))
    check('raw carries the same quality',
          sorted(round(c[4], 2) for c in a.polymer_raw['H']) == [0.63, 0.91])
    import inspect
    bsrc = inspect.getsource(T2.build_chromatin_trace_allele)
    check("v2's Gaussian readout writes NaN there (no gate number of its own)",
          bsrc.count("float(r.amplitude), float('nan'))]") == 2)
    check('a refused fiducial is written as None under its key, with a None drift',
          'allele.fiducial_trace_adj[hybe] = None' in bsrc
          and 'allele.fiducial_trace_raw[hybe] = None' in bsrc
          and 'allele.fiducial_drift[hybe] = None' in bsrc)
    check('the drift is recorded before the gates, as data',
          'allele.fiducial_drift[hybe] = (float(dy), float(dx), float(dz))' in bsrc
          and bsrc.index('allele.fiducial_drift[hybe] = (float(dy)') < bsrc.index("f'drift {drift:.1f}px"))
    check('the fiducial rows carry fiducial_quality(f)',
          'q = fiducial_quality(f)' in bsrc and '(float(f.amplitude), q)' in bsrc)
    # the child -> parent hand-off carries the drift
    al = AnAllele()
    T2.apply_allele_result(al, (7, {'H1': (1., 2., 3., 4., 0.5)}, {}, {}, {}, {}, None, {},
                                {'H1': (0., 0., 0.), 'H2': None}))
    check('apply_allele_result installs fiducial_drift',
          al.fiducial_drift == {'H1': (0., 0., 0.), 'H2': None})


class StubResolver(object):
    """Every hybe projects by the same shift (2, 3) and z + 5, so a
    re-derived correction is exactly the raw fiducial difference."""
    within = {}
    shared = 'DNA'

    def to_shared(self, hybe, modality, cell):
        return np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]])

    def z_to_shared(self, hybe, modality, cell):
        return 5.0

    def bridge(self, a, b):
        return np.eye(3)


def test_reconcile_carries_quality_and_drift():
    print('\nreconcile: quality rides along, fiducial_drift re-derived')
    al = [{'id': 1, 'cell': -1, 'anchor_hybe': 'H1',
           'raw_coordinate': (0.0, 0.0, 10.0), 'coordinate': (9., 9., 9.),
           'provenance': {'reference_hybe': 'H1', 'modality': 'DNA'},
           'fiducial_trace_raw': {'H1': (0., 0., 10., 5., 0.9),
                                  'H2': (1., 1., 12., 5., 0.8),
                                  'H3': None},
           'fiducial_trace_adj': {},
           'fiducial_drift': {'H2': (99., 99., 99.)},      # stale, must be replaced
           'polymer_raw': {'H2': [(4.0, 4.0, 12.0, 99.0, 0.77)]},
           'polymer_adj': {}}]
    st = REC.reconcile_allele_dicts(al, StubResolver(), {})
    got = al[0]['polymer_adj']['H2'][0]
    check('polymer adj = projected raw + correction, quality untouched',
          same(got, (4.0 + 2 - 1, 4.0 + 3 - 1, 12.0 + 5 - 2, 99.0, 0.77)), str(got))
    check('fiducial adj keeps amplitude and quality',
          same(al[0]['fiducial_trace_adj']['H2'], (3.0, 4.0, 17.0, 5.0, 0.8))
          and al[0]['fiducial_trace_adj']['H3'] is None, str(al[0]['fiducial_trace_adj']))
    check('fiducial_drift re-derived: reference 0, H2 = ref - H2, None where missing',
          al[0]['fiducial_drift'] == {'H1': (0.0, 0.0, 0.0), 'H2': (-1.0, -1.0, -2.0), 'H3': None},
          str(al[0]['fiducial_drift']))
    check('one allele updated', st['updated'] == 1)


def test_export_columns():
    print('\nexport: quality and drift columns')
    bins = [(15, 'Hyb_015'), (16, 'Hyb_016')]
    a = {'id': 42, 'fov': 3, 'cell': 7, 'anchor_uid': 5,
         'anchor_hybe': 'Hyb_015', 'anchor_channel': 555,
         'coordinate': (10.0, 800.0, 40.0), 'raw_coordinate': (12.0, 802.0, 42.0),
         'fiducial_trace_adj': {'Hyb_015': (1.0, 2.0, 3.0, 111.0, 0.85), 'Hyb_016': None},
         'fiducial_trace_raw': {'Hyb_015': (4.0, 5.0, 6.0, 222.0, 0.85), 'Hyb_016': None},
         'fiducial_drift': {'Hyb_015': (1.0, 2.0, 3.0), 'Hyb_016': None},
         'polymer_adj': {'Hyb_015': [(20.0, 900.0, 50.0, 10.0, 0.61),
                                     (21.0, 901.0, 51.0, 99.0)]},
         'polymer_raw': {'Hyb_015': [(22.0, 902.0, 52.0, 10.0, 0.61),
                                     (23.0, 903.0, 53.0, 99.0)]},
         'rejected_hybes': {'Hyb_016': 'fiducial ncc 0.2 < 0.5'},
         'final_polymer': np.empty((0, 3))}
    rows = E.allele_rows([a], fov=3, bins=bins, modality='DNA')
    r15 = [r for r in rows if r['hybe'] == 'Hyb_015']
    r16 = [r for r in rows if r['hybe'] == 'Hyb_016']
    check('one row per candidate plus one for the empty bin', len(r15) == 2 and len(r16) == 1)
    check('readout_quality is the fifth slot, NaN where absent',
          r15[0]['readout_quality'] == 0.61 and isnan(r15[1]['readout_quality']))
    check('fiducial_quality and the drift, drift in x/y reader order',
          r15[0]['fiducial_quality'] == 0.85 and r15[0]['fiducial_drift_x'] == 2.0
          and r15[0]['fiducial_drift_y'] == 1.0 and r15[0]['fiducial_drift_z'] == 3.0,
          str({k: r15[0][k] for k in ('fiducial_quality', 'fiducial_drift_x', 'fiducial_drift_y')}))
    check('a refused fiducial: found False, NaN quality and drift, reason kept',
          r16[0]['fiducial_found'] is False and isnan(r16[0]['fiducial_quality'])
          and isnan(r16[0]['fiducial_drift_x']) and isnan(r16[0]['readout_quality'])
          and r16[0]['rejected_reason'] == 'fiducial ncc 0.2 < 0.5')
    check('4-wide tuples still export (older stores), amplitude intact',
          r15[1]['readout_adj_amplitude'] == 99.0)


def test_polymer_collapse():
    print('\npolymer collapse with five-wide candidates')
    pos, amp, n = POLY.collapse_polymer(
        {'polymer_adj': {'A': [(1.0, 2.0, 3.0, 10.0, 0.5), (4.0, 5.0, 6.0, 20.0, 0.9)]}},
        ['A', 'B'])
    check('the brightest wins and its position is (y, x, z)',
          tuple(pos[0]) == (4.0, 5.0, 6.0) and amp[0] == 20.0 and n[0] == 2
          and np.isnan(pos[1]).all() and n[1] == 0, str((pos, amp, n)))


def main():
    test_allele_widening()
    test_columnar_round_trip_and_legacy()
    test_fiducial_quality()
    test_readout_writes_p_exist()
    test_reconcile_carries_quality_and_drift()
    test_export_columns()
    test_polymer_collapse()
    print(f'\n{CHECKS[1]}/{CHECKS[0]} checks passed')
    return 0 if CHECKS[1] == CHECKS[0] else 1


if __name__ == '__main__':
    sys.exit(main())
