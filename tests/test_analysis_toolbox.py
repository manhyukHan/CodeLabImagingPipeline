"""
The analysis toolbox: polymer collapse, ensembles, gates, detection nulls.

Everything here is synthetic and pins the CONVENTIONS, not the store:
um scaling per axis, NaN honesty, selector pluggability, gate
composition, and the two completeness nulls' calibration on truly
independent data.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_analysis_toolbox.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                             # noqa: E402
import pandas as pd                                            # noqa: E402

from codelab_pipeline.analysis import (polymer, ensemble, expression,
                                       gate, distances, detection)  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}' + (f'   [{detail}]' if detail else ''))


RECORDS = [
    {'folder': 'Hyb_003', 'readout_id': 3, 'datatype': 'H'},
    {'folder': 'Hyb_001', 'readout_id': 1, 'datatype': 'H'},
    {'folder': 'Rep_001', 'readout_id': 1, 'datatype': 'R'},
    {'folder': 'Hyb_002', 'readout_id': 2, 'datatype': 'H'},
    {'folder': 'Toe_050', 'readout_id': 50, 'datatype': 'T'},
]

print('bins')
check('bins are H-only, readout_id-ordered',
      polymer.bin_hybes(RECORDS) == ['Hyb_001', 'Hyb_002', 'Hyb_003'])
qcr = polymer.qc_rounds(RECORDS)
check('qc_rounds lists R and T rounds, never as polymer bins',
      qcr == {'repeats': [(1, 'Rep_001')], 'toes': [(50, 'Toe_050')]})

print('\npolymer collapse')
allele = {'polymer_adj': {
    'Hyb_001': [(1.0, 2.0, 3.0, 100.0), (9.0, 9.0, 9.0, 500.0)],
    'Hyb_003': [(4.0, 5.0, 6.0, 50.0)],
}}
pos, amp, n_cand = polymer.collapse_polymer(allele, ['Hyb_001', 'Hyb_002', 'Hyb_003'])
check('brightest candidate wins', tuple(pos[0]) == (9.0, 9.0, 9.0) and amp[0] == 500.0)
check('candidate counts kept', list(n_cand) == [2, 0, 1])
check('missing bin is NaN, not zero', np.isnan(pos[1]).all())

sel = lambda cands: min(cands, key=lambda c: c[3])
pos2, amp2, _ = polymer.collapse_polymer(allele, ['Hyb_001'], selector=sel)
check('the Selector is swappable (dimmest wins here)',
      tuple(pos2[0]) == (1.0, 2.0, 3.0) and amp2[0] == 100.0)

print('\ndistance maps in um')
p = np.full((1, 3, 3), np.nan)
p[0, 0] = (0.0, 0.0, 0.0)
p[0, 1] = (0.0, 0.208, 0.0)      # one PIXEL apart in x, already um-scaled
d = polymer.polymer_distmaps(p)
check('distances are euclidean um (float32: memory is the design point)',
      abs(float(d[0, 0, 1]) - 0.208) < 2e-6 and d.dtype == np.float32)
check('NaN position poisons its row and column',
      np.isnan(d[0, 2]).all() and np.isnan(d[0, :, 2]).all())
check('diagonal is zero where observed', d[0, 0, 0] == 0.0)

print('\nQC: the elementwise-brightness bug stays dead')
table = {'pos_um': np.random.default_rng(0).normal(5, 1, (20, 6, 3)),
         'amp': np.full((20, 6), 1000.0)}
# coordinates are ~5, far below any brightness quantile -- the ORCA bug
# would have flagged every bin through the elementwise comparison
dmaps = polymer.polymer_distmaps(table['pos_um'])
thr = polymer.qc_thresholds(table, dmaps)
out = polymer.apply_qc(table, dmaps, thr)
check('uniform-amplitude data loses nothing to the brightness gate',
      not (np.isfinite(table['amp']) & out['bads']).all()
      and out['bads'].sum() < table['amp'].size * 0.5,
      f"{out['bads'].sum()} bad of {table['amp'].size}")
check('efficacy/completeness shapes', len(polymer.efficacy(out['pos_um'])) == 6
      and len(polymer.completeness(out['pos_um'])) == len(out['pos_um']))

print('\nensemble maps')
dm = np.full((4, 3, 3), np.nan)
for i, v in enumerate((1.0, 2.0, 3.0, 100.0)):
    dm[i] = v
    np.fill_diagonal(dm[i], 0)
m, counts = ensemble.ensemble_map(dm, reducer='median')
check('median ensemble', m[0, 1] == 2.5, m[0, 1])
m2, _ = ensemble.ensemble_map(dm, mask=np.array([1, 1, 1, 0], bool))
check('the mask gates alleles', m2[0, 1] == 2.0)
dm[2:, 0, 1] = dm[2:, 1, 0] = np.nan
m3, c3 = ensemble.ensemble_map(dm, min_n=3)
check('min_n masks under-observed pixels to NaN',
      np.isnan(m3[0, 1]) and c3[0, 1] == 2)
sub, _, _ = ensemble.subtraction_map(dm, np.array([1, 1, 0, 0], bool),
                                     np.array([0, 0, 1, 1], bool), min_n=1)
check('subtraction map masks where either group is empty', np.isnan(sub[0, 1]))

print('\nSCC')
rng = np.random.default_rng(1)
base = np.abs(rng.normal(3, 1, (12, 12)))
base = (base + base.T) / 2
noisy = base + rng.normal(0, 0.05, base.shape)
far = np.abs(rng.normal(3, 1, (12, 12)))
far = (far + far.T) / 2
s_same = ensemble.scc(base, noisy, h=0)
s_diff = ensemble.scc(base, far, h=0)
check('similar maps score high, unrelated low',
      s_same > 0.8 and s_diff < 0.5, f'{s_same:.2f} vs {s_diff:.2f}')

print('\nFOV MSD test')
rng_f = np.random.default_rng(3)
n_per, nb = 40, 10
base_map = np.abs(rng_f.normal(3, 1, (nb, nb)))
base_map = (base_map + base_map.T) / 2


def fov_stack(offset, n):
    out = np.repeat(base_map[None], n, 0) + rng_f.normal(0, 0.15, (n, nb, nb))
    return out + offset


homog = np.concatenate([fov_stack(0, n_per) for _ in range(3)])
fov_ids = np.repeat([1, 2, 3], n_per)
r_h = ensemble.fov_msd_test(homog, fov_ids, seed=0)
check('homogeneous FOVs: in-FOV and cross-FOV MSD agree (p not small)',
      r_h['p'] > 0.001, f"p={r_h['p']:.3g}")
deviant = np.concatenate([fov_stack(0, n_per), fov_stack(0, n_per),
                          fov_stack(1.5, n_per)])
r_d = ensemble.fov_msd_test(deviant, fov_ids, seed=0)
worst = max(r_d['per_fov'], key=lambda d: d['signed_neglog10p'])
check('a deviant FOV separates in-FOV from cross-FOV MSD',
      r_d['p'] < 1e-6, f"p={r_d['p']:.3g}")
scores = {d['fov']: d['signed_neglog10p'] for d in r_d['per_fov']}
check('and the SIGNED per-FOV verdict NAMES the deviant FOV alone',
      worst['fov'] == 3 and scores[3] > 3
      and scores[1] < scores[3] and scores[2] < scores[3],
      str({k: round(v, 1) for k, v in scores.items()}))
r_e = ensemble.fov_msd_test(homog[:2], fov_ids[:2])
check('too few alleles reports NaN, not a crash', np.isnan(r_e['p']))

print('\nallele differences (only differences exist, not indices)')
fovs = np.array([1, 1, 1, 2])
cells = np.array([5, 5, 7, -1])
res = ensemble.allele_difference(dm, fovs, cells, n_null=50)
check('one within-cell pair found (cell 5 twice)',
      len(res['within']) == 1 and res['within_cells'] == [(1, 5)])
check('homeless alleles form no within pair', res['n_multi_allelic'] == 1)

print('\ngates')


class FakePop:
    pass


pop = FakePop()
pop.cells = pd.DataFrame({'fov': [1, 1, 2, 2], 'cell': [1, 2, 1, 2],
                          'celltype': ['WT', 'KI', 'WT', '']})
pop.expression = pd.DataFrame({
    'fov': [1, 1, 2, 2], 'cell': [1, 2, 1, 2],
    'celltype': ['WT', 'KI', 'WT', ''],
    'modality': ['RNA'] * 4, 'hybe': ['Hyb_101'] * 4, 'channel': [555] * 4,
    'n_spots': [10, 3, 7, 0],
    'brightness_median': [200.0, 150.0, 180.0, np.nan],
    'brightness_total': [2000.0, 450.0, 1260.0, 0.0]})
pop.alleles = {'fov': np.array([1, 1, 2]), 'cell': np.array([1, 1, 1]),
               'n_traced': np.array([50, 40, 1])}

m = gate.CelltypeIn(['WT']).mask(pop)
check('celltype gate', list(m) == [True, False, True, False])
# distance_histogram with a gate on ZERO pairs must return an empty
# histogram, not KeyError (the pandas empty-list column-selection trap)
m = gate.ExpressionRange(('RNA', 'Hyb_101', 555), 'n_spots', lo=5).mask(pop)
check('expression range gate', list(m) == [True, False, True, False])
m = gate.ExpressionRange(('RNA', 'Hyb_101', 555), 'brightness_median',
                         lo=0).mask(pop)
check('a cell with no value FAILS a range gate (NaN is not in range)',
      list(m) == [True, True, True, False])
m = gate.AlleleCount(lo=2, min_bins=2).mask(pop)
check('allele count gate needs traced alleles',
      list(m) == [True, False, False, False],
      f'{list(m)}')
cond = gate.Condition([gate.CelltypeIn(['WT']),
                       gate.ExpressionRange(('RNA', 'Hyb_101', 555),
                                            'n_spots', lo=5)])
check('conditions AND sequentially', list(cond.mask(pop)) == [True, False, True, False])
check('report gives sequential survivor counts',
      [n for _r, n in cond.report(pop)] == [2, 2])
d = cond.to_dict()
cond2 = gate.Condition.from_dict(d)
check('a gate round-trips through plain data',
      cond2.to_dict() == d and list(cond2.mask(pop)) == list(cond.mask(pop)))

print('\nOR composition and allele-level presence')
from codelab_pipeline.analysis import population as popmod   # noqa: E402
FakePop.allele_mask_from_cells = popmod.Population.allele_mask_from_cells
pop.alleles = {'fov': np.array([1, 1, 2]), 'cell': np.array([1, 1, 1]),
               'n_traced': np.array([50, 40, 30]),
               'bin_hybes': ['A', 'B', 'C'],
               'pos_um': np.array([
                   [[1., 1, 1], [2, 2, 2], [np.nan] * 3],   # has A, B
                   [[np.nan] * 3, [2, 2, 2], [3, 3, 3]],    # missing A
                   [[1., 1, 1], [np.nan] * 3, [3, 3, 3]]])}  # has A
am = gate.BarcodePresence(['A']).allele_mask(pop)
check('presence gate is ALLELE-level (heterogeneous modification)',
      list(am) == [True, False, True])
check('its cell-level projection: cell holds >= 1 qualifying allele',
      list(gate.BarcodePresence(['A']).mask(pop)) == [True, False, True, False])
check('absent=True inverts per allele',
      list(gate.BarcodePresence(['A'], absent=True).allele_mask(pop))
      == [False, True, False])
orc = gate.Condition(clauses=[[gate.CelltypeIn(['WT'])],
                              [gate.CelltypeIn(['KI'])]])
check('clauses OR together: (WT) OR (KI)',
      list(orc.mask(pop)) == [True, True, True, False])
check('the report marks clause boundaries',
      any(r == '-- OR --' for r, _n in orc.report(pop)))
mixed = gate.Condition(clauses=[
    [gate.CelltypeIn(['WT']), gate.BarcodePresence(['A'])]])
check('allele_mask: cell predicates project, presence applies per allele',
      list(mixed.allele_mask(pop)) == [True, False, True])
mixed2 = gate.Condition(clauses=[
    [gate.CelltypeIn(['WT']), gate.BarcodePresence(['A'])],
    [gate.BarcodePresence(['C'])]])
check('OR at the allele level too: ... OR presence(C)',
      list(mixed2.allele_mask(pop)) == [True, True, True])
d_or = mixed2.to_dict()
check('OR-composed gates round-trip through plain data',
      gate.Condition.from_dict(d_or).to_dict() == d_or)

print('\ncompleteness gate (allele-level, integer bounds)')
cr = gate.CompletenessRange(lo=35)
check('CompletenessRange gates ALLELES on n_traced',
      list(cr.allele_mask(pop)) == [True, True, False])
check('its cell projection: cell holds >= 1 qualifying allele',
      list(cr.mask(pop)) == [True, False, False, False])
check('hi bound is inclusive',
      list(gate.CompletenessRange(lo=30, hi=40).allele_mask(pop))
      == [False, True, True])
d_cr = gate.CompletenessRange(lo=30, hi=40).to_dict()
check('completeness gate round-trips through plain data',
      gate.Predicate.from_dict(d_cr).to_dict() == d_cr)

print('\nexpression normalization')
t = pop.expression.copy()
# modality-wide per-cell denominators, as the population build computes
t['mod_n_spots'] = [20, 3, 7, 0]
n1 = expression.normalize(t, 'n_spots', 'by_modality')
check('by_modality divides by the modality-wide per-cell quantity',
      n1['n_spots_norm'].iloc[0] == 0.5)
check("legacy alias 'by_total_count' lands on by_modality",
      expression.normalize(t, 'n_spots',
                           'by_total_count')['n_spots_norm'].iloc[0] == 0.5)
check('zero denominator is NaN, never fabricated',
      np.isnan(n1['n_spots_norm'].iloc[3]))
try:
    expression.normalize(pop.expression, 'n_spots', 'by_modality')
    refused = False
except ValueError:
    refused = True
check('a table without mod_* columns refuses, not silently normalizes',
      refused)
n2 = expression.normalize(t, 'brightness_total', 'by_source',
                          ref_source=('RNA', 'Hyb_101', 555))
check('by_source normalizes against the reference source, NaN-honest',
      n2['brightness_total_norm'].iloc[0] == 1.0)

print('\npair distances')
pop.spots = pd.DataFrame({
    'fov': [1, 1, 1, 1], 'cell': [1, 1, 1, 2],
    'celltype': ['WT'] * 3 + ['KI'],
    'modality': ['RNA', 'RNA', 'DNA', 'RNA'],
    'hybe': ['Hyb_101', 'Hyb_101', 'Hyb_016', 'Hyb_101'],
    'channel': [555, 555, 555, 555],
    'y_um': [0.0, 3.0, 4.0, 0.0], 'x_um': [0.0, 0.0, 0.0, 0.0],
    'z_um': [0.0, 4.0, 3.0, 0.0], 'brightness': [1.0] * 4})
pairs = distances.pair_distances(pop, ('RNA', 'Hyb_101', 555),
                                 ('DNA', 'Hyb_016', 555))
check('cross-set pairs within the cell only, um euclidean',
      len(pairs) == 2 and abs(sorted(pairs['d_um'])[0] - np.sqrt(2)) < 1e-9
      and abs(sorted(pairs['d_um'])[1] - 5.0) < 1e-9,
      str(list(pairs['d_um'])))
same = distances.pair_distances(pop, ('RNA', 'Hyb_101', 555),
                                ('RNA', 'Hyb_101', 555))
check('same-source excludes self-pairs and double counting',
      len(same) == 1 and abs(same['d_um'].iloc[0] - 5.0) < 1e-9)
per_cell = distances.pair_distance_per_cell(pop, ('RNA', 'Hyb_101', 555),
                                            ('DNA', 'Hyb_016', 555))
check('per-cell collapse is median', abs(per_cell.loc[(1, 1)]
      - np.median(pairs['d_um'])) < 1e-9)
counts_h, _edges = distances.distance_histogram(
    pop, ('RNA', 'Hyb_101', 555), ('DNA', 'Hyb_016', 555),
    mask=np.array([False, False, False, False]))
check('a gate leaving zero pairs yields an EMPTY histogram, not KeyError',
      counts_h.sum() == 0)
empty_tab = {'pos_um': np.empty((0, 5, 3)), 'amp': np.empty((0, 5))}
empty_dm = polymer.polymer_distmaps(empty_tab['pos_um'])
out0 = polymer.apply_qc(empty_tab, empty_dm,
                        polymer.qc_thresholds(empty_tab, empty_dm))
check('apply_qc on zero alleles returns empties, not IndexError',
      len(out0['kept']) == 0 and out0['dmaps'].shape == (0, 5, 5))

print('\ndetection nulls')
rng = np.random.default_rng(7)
n, m_bins = 300, 12
u = rng.normal(0, 1.0, n)
b = rng.normal(0.5, 0.5, m_bins)
X_indep = (rng.random((n, m_bins))
           < 1 / (1 + np.exp(-(b[None, :] + u[:, None])))).astype(np.uint8)
fit = detection.fit_quality_model(X_indep, n_nodes=15, maxiter=200)
check('quality model recovers a real tau', 0.5 < fit['tau'] < 2.0,
      f"tau={fit['tau']:.2f}")
u_mean, _sd, q = detection.posterior_u(X_indep, fit['b'], fit['tau'], n_nodes=15)
check('posterior efficiency tracks the truth',
      np.corrcoef(u_mean, u)[0, 1] > 0.6,
      f'r={np.corrcoef(u_mean, u)[0, 1]:.2f}')
Z, p1, ratio, C, Ex = detection.cooccurrence_z(X_indep, q, fit['b'],
                                               fit['tau'])
off = Z[np.triu_indices(m_bins, 1)]
check('independent bins: co-occurrence z is calibrated (|median| < 1)',
      abs(np.nanmedian(off)) < 1.0, f'{np.nanmedian(off):.2f}')
# now COUPLE two bins hard and expect the null to flag exactly them
X_dep = X_indep.copy()
X_dep[:, 5] = X_dep[:, 4]
fit2 = detection.fit_quality_model(X_dep, n_nodes=15, maxiter=200)
u2, _s2, q2 = detection.posterior_u(X_dep, fit2['b'], fit2['tau'], n_nodes=15)
Z2, _p, _r, _c, _e = detection.cooccurrence_z(X_dep, q2, fit2['b'],
                                              fit2['tau'])
others = Z2[np.triu_indices(m_bins, 1)]
check('a duplicated bin pair stands out against the quality null',
      Z2[4, 5] > np.nanquantile(others, 0.99), f'z={Z2[4, 5]:.1f}')
Zc, _pc, info = detection.count_stratified_null_z(X_dep, n_samples=100,
                                                  seed=0)
check('and against the margin-preserving count null too',
      Zc[4, 5] > 3.0, f'z={Zc[4, 5]:.1f}')
# THE FIXED DEFECT, pinned: strongly heterogeneous per-bin efficacy on
# INDEPENDENT data. The uniform-subset null ignored column margins and
# called every high-efficacy pair significant; the curveball null
# preserves both margins and must stay calibrated here.
b_het = np.linspace(-2.5, 2.5, m_bins)
X_het = (rng.random((n, m_bins))
         < 1 / (1 + np.exp(-(b_het[None, :] + u[:, None])))).astype(np.uint8)
Zh, _ph, _ih = detection.count_stratified_null_z(X_het, n_samples=100,
                                                 seed=1)
off_h = Zh[np.triu_indices(m_bins, 1)]
check('heterogeneous-efficacy independent data stays CALIBRATED under '
      'the margin-preserving null',
      abs(np.nanmedian(off_h)) < 1.0 and np.nanmax(np.abs(off_h)) < 6.0,
      f'median {np.nanmedian(off_h):.2f}, max |z| {np.nanmax(np.abs(off_h)):.1f}')
q_bh = detection.bh_fdr(np.array([0.001, 0.02, 0.5, np.nan]))
check('bh_fdr is monotone and NaN-transparent',
      q_bh[0] <= q_bh[1] <= q_bh[2] and np.isnan(q_bh[3]))
od = detection.count_overdispersion(X_indep, fit['b'], fit['tau'],
                                    n_boot=200, n_nodes=15)
check('independent data is not overdispersed', od['p'] > 0.05,
      f"p={od['p']:.3f} ratio={od['ratio']:.2f}")

print('\nreconcile: adj re-derived from raw under CURRENT matrices')
from codelab_pipeline.analysis import reconcile as rec               # noqa: E402


class StubResolver:
    """+2 y, +3 x for every projection; +5 planes on the z chain."""

    def to_shared(self, hybe, modality, cell):
        return np.array([[1., 0, 2], [0, 1, 3], [0, 0, 1]])

    def z_to_shared(self, hybe, modality, cell):
        return 5.0


spots_r = [{'hybe': 'H1', 'modality': 'DNA', 'cell': -1,
            'raw_coordinate': (10.0, 20.0, 30.0),
            'adj_coordinate': (0.0, 0.0, 0.0)},
           {'hybe': 'H1', 'modality': 'DNA', 'cell': 7,
            'raw_coordinate': (1.0, 1.0, 0.0),
            'adj_coordinate': (1.0, 1.0, 0.0)}]
st = rec.reconcile_spot_dicts(spots_r, StubResolver(), {})
check('spot adj = H @ raw, z = raw z + z chain',
      spots_r[0]['adj_coordinate'] == (12.0, 23.0, 35.0))
check('the 2D placeholder z (raw 0, adj 0) is NEVER minted into depth',
      spots_r[1]['adj_coordinate'] == (3.0, 4.0, 0.0)
      and st['z_placeholder'] == 1)

al_r = [{'id': 1, 'cell': -1, 'anchor_hybe': 'H1',
         'raw_coordinate': (0.0, 0.0, 10.0), 'coordinate': (9., 9., 9.),
         'provenance': {'reference_hybe': 'H1', 'modality': 'DNA'},
         'fiducial_trace_raw': {'H1': (0., 0., 10., 5.),
                                'H2': (1., 1., 12., 5.)},
         'fiducial_trace_adj': {},
         'polymer_raw': {'H2': [(4.0, 4.0, 12.0, 99.0)]},
         'polymer_adj': {}},
        {'id': 2, 'cell': -1, 'anchor_hybe': 'H1',
         'raw_coordinate': (0., 0., 0.), 'coordinate': (0., 0., 0.),
         'provenance': {},
         'fiducial_trace_raw': {'H1': (0., 0., 0., 1.)},
         'fiducial_trace_adj': {}, 'polymer_raw': {}, 'polymer_adj': {}},
        {'id': 3, 'cell': -1, 'anchor_hybe': 'H1',
         'raw_coordinate': (0., 0., 0.), 'coordinate': (0., 0., 0.),
         'provenance': {}, 'fiducial_trace_raw': {},
         'fiducial_trace_adj': {}, 'polymer_raw': {}, 'polymer_adj': {}}]
st = rec.reconcile_allele_dicts(al_r, StubResolver(), {})
# stub projects both fiducials by the same shift, so the re-derived
# correction is exactly the RAW fiducial difference: H1 - H2 = (-1,-1,-2)
got = al_r[0]['polymer_adj']['H2'][0]
check('polymer adj = projected raw + re-derived ref-relative correction',
      got == (4.0 + 2 - 1, 4.0 + 3 - 1, 12.0 + 5 - 2, 99.0), str(got))
check('fiducial adj re-derived from raw',
      al_r[0]['fiducial_trace_adj']['H2'] == (3.0, 4.0, 17.0, 5.0))
check('an allele naming no reference is SKIPPED, never guessed',
      st['skipped']['no_reference'] == 1)
check('an allele with no raw at all is skipped and counted',
      st['skipped']['no_raw'] == 1)
check('exactly the reconcilable allele updated', st['updated'] == 1)
st2 = rec.reconcile_allele_dicts([al_r[1]], StubResolver(), {},
                                 modality='DNA', reference_hybe='H1')
check('arguments stand in when provenance is unstamped (old traces)',
      st2['updated'] == 1)

print('\nreconcile: cell anchors refresh under current matrices')


class StubResolver2(StubResolver):
    shared = 'DNA'
    within = {'RNA': {'Hyb_101': np.array([[1., 0, 10], [0, 1, 20],
                                           [0, 0, 1]])}}

    def bridge(self, modality, shared):
        return np.array([[1., 0, 1], [0, 1, 2], [0, 0, 1]])


cell_r = [{'id': 1,
           'matrix_anchors': {'RNA': np.eye(3)},
           'matrix_provenance': {('Hyb_103', 'RNA'): {
               'reference_sequence': 'Hyb_103(cell 3)->Hyb_101 [z=3.0px]'}}},
          {'id': 2, 'matrix_anchors': {'RNA': np.eye(3)},
           'matrix_provenance': {}}]
st = rec.reconcile_cell_dicts(cell_r, StubResolver2())
new_a = cell_r[0]['matrix_anchors']['RNA']
check('anchor = bridge @ within[anchor hybe], anchor hybe parsed from '
      'the residuals\' own provenance',
      new_a[0, 2] == 11.0 and new_a[1, 2] == 22.0,
      f'({new_a[0, 2]}, {new_a[1, 2]})')
check('a modality nothing testifies for is skipped, never guessed',
      st['skipped']['anchor_hybe_unknown'] == 1 and st['updated'] == 1)

print('\ncomputed-attribute cache (append mode)')
# item 3's contract, pinned end to end on a real temp capsule: build
# persists per-source rows + per-modality aggregates; a rebuild reuses
# them WITHOUT touching spots; appending a source computes only the new
# one; overwrite recomputes; celltype is never trusted from the cache.
import shutil                                                  # noqa: E402
import tempfile                                                # noqa: E402
from codelab_pipeline.analysis import population as popmod2    # noqa: E402
from codelab_pipeline.io import analysis_store as ASt          # noqa: E402
from codelab_pipeline.io import paths as iopaths               # noqa: E402

_croot = tempfile.mkdtemp(prefix='exprcache_')
iopaths.write_manifest(os.path.join(_croot, 'proj'), ['RNA'])
_csp = os.path.join(_croot, 'proj', 'RNA')


def _cells_v1(s, f):
    return [{'id': 1, 'celltype': 'WT'}, {'id': 2, 'celltype': 'KI'}], ''


def _spots_ok(s, f, modality=None, hybe=None, channel=None):
    return [{'cell': 1, 'brightness': 100.0},
            {'cell': 1, 'brightness': 50.0},
            {'cell': -1, 'brightness': 1.0}]


def _spots_boom(s, f, modality=None, hybe=None, channel=None):
    raise AssertionError('cached build must not read spots')


_orig_rc, _orig_rs = ASt.read_cells, ASt.read_spots
try:
    ASt.read_cells, ASt.read_spots = _cells_v1, _spots_ok
    SRC1, SRC2 = ('RNA', 'Hyb_101', 555), ('RNA', 'Hyb_102', 555)

    def _item(srcs, overwrite=False):
        return (_csp, 1, None, srcs, None, (0.208, 0.208, 0.2), False,
                None, None, overwrite)

    b1 = popmod2._fov_bundle(_item([SRC1]))
    check('first build computes, persists the capsule, joins mod_*',
          ASt.read_fov_expression(_csp, 1) is not None
          and b1['expr_cache'] == {'cached': 0, 'computed': 1}
          and b1['expression']['n_spots'].tolist() == [2, 0]
          and b1['expression']['mod_n_spots'].tolist()[0] == 2)
    ASt.read_spots = _spots_boom
    b2 = popmod2._fov_bundle(_item([SRC1]))
    check('rebuild reuses the capsule -- ZERO spot reads',
          b2['expr_cache'] == {'cached': 1, 'computed': 0}
          and b2['expression']['n_spots'].tolist() == [2, 0]
          and b2['expression']['mod_n_spots'].tolist()[0] == 2)
    ASt.read_cells = lambda s, f: ([{'id': 1, 'celltype': 'NEW'},
                                    {'id': 2, 'celltype': 'KI'}], '')
    b3 = popmod2._fov_bundle(_item([SRC1]))
    check('cached rows refresh celltype from the CURRENT cells',
          b3['expression']['celltype'].tolist() == ['NEW', 'KI'])
    ASt.read_spots = _spots_ok
    b4 = popmod2._fov_bundle(_item([SRC1, SRC2]))
    check('appending a source computes ONLY the new one',
          b4['expr_cache'] == {'cached': 1, 'computed': 1}
          and len(b4['expression']) == 4)
    b5 = popmod2._fov_bundle(_item([SRC1, SRC2], overwrite=True))
    check('overwrite recomputes everything requested',
          b5['expr_cache'] == {'cached': 0, 'computed': 2})
finally:
    ASt.read_cells, ASt.read_spots = _orig_rc, _orig_rs
    shutil.rmtree(_croot, ignore_errors=True)

print('\nrepeat/toe QC figure')
import matplotlib                                              # noqa: E402
matplotlib.use('Agg')
from codelab_pipeline.analysis import figures                  # noqa: E402


class QcPop:
    pass


qp = QcPop()
qp.alleles = {
    'bin_ids': [1, 2, 3],
    'pos_um': np.array([[[0., 0, 0], [1, 1, 1], [2, 2, 2]],
                        [[0., 0, 0], [np.nan] * 3, [2, 2, 2]]]),
    # repeat of bin 2: allele 0 re-imaged 1 um off; allele 1 has no H fix
    'repeat_pos_um': np.array([[[1., 2, 1]], [[5., 5, 5]]]),
    'repeat_ids': [2],
    # two toe rounds: first seen in both alleles, second in one
    'toe_pos_um': np.array([[[1., 1, 1], [np.nan] * 3],
                            [[2., 2, 2], [3., 3, 3]]]),
    'toe_ids': [50, 51],
}
figq = figures.fig_repeat_toe_qc(qp)
check('repeat/toe QC figure renders from build-time R/T positions',
      len(figq.axes) == 2)
import matplotlib.pyplot as plt                                # noqa: E402
plt.close(figq)
# the H-vs-R distance the left panel histograms: allele 0 only
# (allele 1's H bin 2 is NaN -> its pair is NaN-honestly dropped)
d_hr = np.sqrt(((qp.alleles['pos_um'][:, 1, :]
                 - qp.alleles['repeat_pos_um'][:, 0, :]) ** 2).sum(1))
check('repeat distance is NaN-honest per allele',
      d_hr[0] == 1.0 and np.isnan(d_hr[1]))

print('\nthe headless contract')
# THE toolbox promise, pinned: importing the whole analysis package --
# resolvers included -- loads no Qt and no app module. A fresh
# subprocess, because this test file itself may run beside app tests.
import subprocess                                              # noqa: E402
r = subprocess.run(
    [sys.executable, '-c',
     'import sys; import codelab_pipeline.analysis; '
     'import codelab_pipeline.analysis.resolvers; '
     'bad = [m for m in sys.modules if "PyQt5" in m '
     'or m.split(".")[0] in ("windows", "ui", "canvas")]; '
     'sys.exit(1 if bad else 0)'],
    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    capture_output=True, text=True)
check('importing the toolbox loads no Qt and no app module',
      r.returncode == 0, r.stdout + r.stderr)

print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
if FAIL:
    for nme in FAIL:
        print('  FAILED:', nme)
    sys.exit(1)
print('ALL GOOD')
