"""
The cell-cycle phase model: recovery, the bridge, placement, the three
verdicts, persistence. Everything synthetic; the real-data validation is
recorded in the design document.

Run:  python tests/test_cellcycle.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                             # noqa: E402
import pandas as pd                                            # noqa: E402

from codelab_pipeline.analysis import cellcycle as CC          # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}' + (f'   [{detail}]' if detail else ''), flush=True)


def deg(x):
    return x * 360.0


# -- 1. one experiment: recovery, fast M-step vs exact -------------------------
print('single experiment')
X, theta, coef, a, genes = CC.simulate(n=1200, seed=1)
t0 = time.time()
m = CC.CycleModel().fit([CC.Dataset('A', X, genes, alpha=100.0)])
t_fast = time.time() - t0
th, R, _ = m.phase('A', X)
med, sgn, shift = CC.circular_agreement(theta, th)
check('phases recovered within 10 deg (fast M-step, DM E-step)', deg(med) < 10, f'{deg(med):.1f} deg, {t_fast:.1f}s')
check('evidence never fell below its best (keep-best)', m.history[-1] >= max(m.history) - 1e-9)
me = CC.CycleModel(mstep='dm').fit([CC.Dataset('A', X, genes, alpha=100.0)])
the, _, _ = me.phase('A', X)
med2, _, _ = CC.circular_agreement(th, the)
check('exact DM M-step agrees with the fast path within 8 deg', deg(med2) < 8, f'{deg(med2):.1f} deg')
check('centre is softmax of the intercepts and the CLR ring mean',
      np.allclose(CC.clr(m.centre('A')[None, :])[0], CC.clr(np.exp(m.log_pi('A'))).mean(0), atol=1e-6))

# -- 2. orientation by roles ----------------------------------------------------
peaks, _ = m.peak_phase()
early = [genes[i] for i in np.argsort(peaks)[:3]]
late = [genes[i] for i in np.argsort(peaks)[4:7]]
shift_, flip_ = m.orient(early, late)
peaks2, _ = m.peak_phase()
e = np.degrees(np.angle(np.mean(np.exp(1j * peaks2[[m.gi[g] for g in early]])))) % 360
l_ = np.degrees(np.angle(np.mean(np.exp(1j * peaks2[[m.gi[g] for g in late]])))) % 360
check('after orient the early mean peak is 0 and the late one is ahead of it', min(e, 360 - e) < 3 and 0 < l_ < 180, f'early {e:.0f}, late {l_:.0f}')
th_o, _, _ = m.phase('A', X)
med3, _, _ = CC.circular_agreement(theta, th_o)
check('orientation keeps the recovery', deg(med3) < 10, f'{deg(med3):.1f} deg')

# -- 3. three experiments, two disjoint panels, one bridge ----------------------
print('bridge')
rng = np.random.default_rng(2)
allg = [f'g{i}' for i in range(12)]
coef12 = np.zeros((12, 4))
coef12[:, :2] = rng.normal(0, 1.0, (12, 2))
coef12[:, 2:] = rng.normal(0, 0.3, (12, 2))
def make(name, cols, n, seed, arrested=None):
    th = np.random.default_rng(seed).uniform(0, CC.TWO_PI, n)
    groups = np.array(['cycling'] * n, dtype=object)
    if arrested is not None:
        k = np.random.default_rng(seed + 1).random(n) < 0.5
        th[k] = (arrested + np.random.default_rng(seed + 2).normal(0, 0.2, k.sum())) % CC.TWO_PI
        groups[k] = 'arrested'
    Xd, thd, _, _, _ = CC.simulate(n=n, genes=[allg[c] for c in cols], coef=coef12[cols], seed=seed, theta=th)
    return Xd, thd, groups, [allg[c] for c in cols]
XA, thA, gA, genA = make('A', [0, 1, 2, 3, 4, 5], 900, 10, arrested=1.0)
XB, thB, gB, genB = make('B', [6, 7, 8, 9, 10, 11], 900, 20, arrested=4.0)
XC, thC, gC, genC = make('C', [0, 1, 2, 6, 7, 8], 900, 30)
cyc = lambda Xd, gd: (Xd[gd == 'cycling'], gd[gd == 'cycling'])            # noqa: E731
XA_c, gA_c = cyc(XA, gA)
XB_c, gB_c = cyc(XB, gB)
mj = CC.CycleModel().fit([CC.Dataset('A', XA_c, genA, gA_c, alpha=100.0),
                          CC.Dataset('B', XB_c, genB, gB_c, alpha=100.0),
                          CC.Dataset('C', XC, genC, gC, alpha=100.0)])
check('bridge report names the shared genes', mj.bridge_report()[('A', 'B')] == [] and len(mj.bridge_report()[('A', 'C')]) == 3)
th_all = np.concatenate([mj.phase(n, Xd)[0] for n, Xd in (('A', XA), ('B', XB), ('C', XC))])
tr_all = np.concatenate([thA, thB, thC])
medj, sgnj, shiftj = CC.circular_agreement(tr_all, th_all)
check('one rotation aligns all three experiments within 10 deg', deg(medj) < 10, f'{deg(medj):.1f} deg')
for n, Xd, thd in (('A', XA, thA), ('B', XB, thB)):
    thp = mj.phase(n, Xd)[0]
    err = np.abs(np.angle(np.exp(1j * (thd - sgnj * thp - shiftj))))
    check(f'{n} placed in the common frame within 12 deg', deg(np.median(err) / CC.TWO_PI) < 12, f'{deg(np.median(err) / CC.TWO_PI):.1f} deg')
sA = mj.group_summary('A', XA, gA)
sB = mj.group_summary('B', XB, gB)
check('arrested groups concentrate, cycling groups spread',
      sA['arrested']['concentration'] > 0.9 and sB['arrested']['concentration'] > 0.9
      and sA['cycling']['concentration'] < 0.3 and sB['cycling']['concentration'] < 0.3,
      f"A {sA['arrested']['concentration']:.2f}/{sA['cycling']['concentration']:.2f}, B {sB['arrested']['concentration']:.2f}/{sB['cycling']['concentration']:.2f}")
gap = abs(np.angle(np.exp(1j * np.radians(sA['arrested']['mean_deg'] - sB['arrested']['mean_deg']))))
check('the two arrest points keep their true separation (3.0 rad) within 0.3 rad', abs(gap - 3.0) < 0.3, f'{gap:.2f} rad')

# -- 3b. the reflection fixed by gene roles: the same bridge, no flip search ----
print('roles')
pk_true = np.degrees(np.arctan2(coef12[:, 1], coef12[:, 0])) % 360           # true peak of the first harmonic
early_g = [allg[i] for i in np.argsort(pk_true)[:3]]
late_g = [allg[i] for i in np.argsort(pk_true)[5:8]]
mr = CC.CycleModel().fit([CC.Dataset('A', XA_c, genA, gA_c, alpha=100.0),
                          CC.Dataset('B', XB_c, genB, gB_c, alpha=100.0),
                          CC.Dataset('C', XC, genC, gC, alpha=100.0)], orient_by=(early_g, late_g))
fixed = [r.get('reflection_fixed_by_roles') for r in mr.align_report.values() if 'flip' in r]
th_r = np.concatenate([mr.phase(n, Xd)[0] for n, Xd in (('A', XA), ('B', XB), ('C', XC))])
medr, _, _ = CC.circular_agreement(tr_all, th_r)
check('with orient_by every bridge skips the reflection search', all(fixed) and len(fixed) == 2, str(fixed))
check('and the three experiments still align within 10 deg', deg(medr) < 10, f'{deg(medr):.1f} deg')

print('verdicts')
# cells generated from the FITTED ring of A (its own profiles and intercepts),
# cells at its centre, and cells pushed off it
Xr, thr, _, _, _ = CC.simulate(n=600, genes=genA, coef=mj.coef[mj.cols['A']], a=mj.a['A'],
                               seed=40, total=(150, 400))
pc = mj.centre('A')
rng = np.random.default_rng(41)
Xc = np.stack([rng.multinomial(int(s), pc) for s in rng.integers(150, 400, 600)])       # centre cells
Xo = Xr.copy()
Xo[:, 0] = Xo[:, 0] * 8 + 50                                                              # one gene blown up: outside
out_r, out_c, out_o = (mj.place('A', Xd) for Xd in (Xr, Xc, Xo))
check('ring cells: log bf_ring high, radius ~1, fit_z ~0',
      np.median(out_r['bf_ring']) > 3 and 0.7 < np.median(out_r['radius']) < 1.3 and abs(np.nanmedian(out_r['fit_z'])) < 1.0,
      f"bf {np.median(out_r['bf_ring']):.1f}, radius {np.median(out_r['radius']):.2f}, z {np.nanmedian(out_r['fit_z']):.2f}")
check('centre cells: log bf_ring far below the ring cells, radius small',
      np.median(out_c['bf_ring']) < np.median(out_r['bf_ring']) - 3 and np.median(out_c['radius']) < 0.5,
      f"bf {np.median(out_c['bf_ring']):.1f}, radius {np.median(out_c['radius']):.2f}, z {np.nanmedian(out_c['fit_z']):.2f}")
check('centre cells still get a confident-looking R (why bf_ring exists)', np.median(out_c['R']) > 0.5, f"R {np.median(out_c['R']):.2f}")
check('outside cells: fit_z far below the cycling reference', np.nanmedian(out_o['fit_z']) < -2, f"z {np.nanmedian(out_o['fit_z']):.2f}")
# the real-data regime: a SMALL ring (profile amplitude ~0.4), where the
# centre lies inside the dispersion band -- fit_z is blind there and
# only bf_ring / radius see the centre cells
Xs, ths, coefs, a_s, gens = CC.simulate(n=1200, seed=50, total=(150, 500))
coefs = coefs * 0.3
Xs, ths, _, _, _ = CC.simulate(n=1200, seed=51, total=(150, 500), coef=coefs, a=a_s, genes=gens)
ms = CC.CycleModel().fit([CC.Dataset('S', Xs, gens, alpha=100.0)])
pcs = ms.centre('S')
Xsc = np.stack([rng.multinomial(int(s), pcs) for s in rng.integers(150, 500, 400)])
Xsr, _, _, _, _ = CC.simulate(n=400, genes=gens, coef=ms.coef, a=ms.a['S'], seed=52, total=(150, 500))
o_sr, o_sc = ms.place('S', Xsr), ms.place('S', Xsc)
check('small ring: fit_z cannot tell centre cells from ring cells',
      abs(np.nanmedian(o_sc['fit_z']) - np.nanmedian(o_sr['fit_z'])) < 1.5,
      f"z centre {np.nanmedian(o_sc['fit_z']):.2f} vs ring {np.nanmedian(o_sr['fit_z']):.2f}")
check('small ring: bf_ring and radius still separate them',
      np.median(o_sc['bf_ring']) < np.median(o_sr['bf_ring']) - 2 and np.median(o_sc['radius']) < 0.6 < np.median(o_sr['radius']),
      f"bf centre {np.median(o_sc['bf_ring']):.1f} vs ring {np.median(o_sr['bf_ring']):.1f}; radius {np.median(o_sc['radius']):.2f} vs {np.median(o_sr['radius']):.2f}")

# -- 4b. two experiments of very different count depth share every gene:
#        the fast path must not let the deep one own the profiles
print('depth')
Xd1, thd1, coefd, ad, gend = CC.simulate(n=700, seed=60, total=(60, 160))
Xd2, thd2, _, _, _ = CC.simulate(n=700, seed=61, total=(800, 1600), coef=coefd, a=ad + 0.3, genes=gend)
md = CC.CycleModel().fit([CC.Dataset('shallow', Xd1, gend, alpha=100.0), CC.Dataset('deep', Xd2, gend, alpha=100.0)])
me_ = CC.CycleModel(mstep='dm').fit([CC.Dataset('shallow', Xd1, gend, alpha=100.0), CC.Dataset('deep', Xd2, gend, alpha=100.0)])
e1, _, _ = CC.circular_agreement(thd1, md.phase('shallow', Xd1)[0])
e2, _, _ = CC.circular_agreement(thd2, md.phase('deep', Xd2)[0])
check('fast path recovers the shallow and the deep experiment', deg(e1) < 12 and deg(e2) < 8, f'{deg(e1):.1f} / {deg(e2):.1f} deg')
amp_f = np.log(np.exp(md.B @ md.coef.T).max(0) / np.exp(md.B @ md.coef.T).min(0))
amp_e = np.log(np.exp(me_.B @ me_.coef.T).max(0) / np.exp(me_.B @ me_.coef.T).min(0))
check('fast path profile amplitudes match the exact DM fit within 15%', np.median(np.abs(amp_f - amp_e) / amp_e) < 0.15, f'median rel. diff {np.median(np.abs(amp_f - amp_e) / amp_e):.2f}')
rad_s = np.median(md.radius('shallow', Xd1)[0]); rad_d = np.median(md.radius('deep', Xd2)[0])
check('cells of both experiments sit on the fitted ring (radius ~1)', 0.75 < rad_s < 1.3 and 0.75 < rad_d < 1.3, f'{rad_s:.2f} / {rad_d:.2f}')

# -- 4c. the origin at division: the steepest drop of the panel total ----------
print('origin')
rng_o = np.random.default_rng(70)
th_o = rng_o.uniform(0, 360, 3000)
tot_o = np.where(((th_o - 200) % 360) < 180, 900.0, 300.0) * rng_o.lognormal(0, 0.15, 3000)   # high on 200..20, drops at 20
ang, fac = CC.total_drop_angle(th_o, tot_o)
check('the drop angle is found within a bin', min(abs(ang - 20), 360 - abs(ang - 20)) <= 12, f'{ang:.0f} deg, x{fac:.1f}')
check('and the drop factor is about three', 2.0 < fac < 4.5, f'x{fac:.1f}')
check('too few cells or no drop gives None / a factor near 1',
      CC.total_drop_angle(th_o[:50], tot_o[:50]) == (None, None) and CC.total_drop_angle(th_o, np.full(3000, 500.0))[1] < 1.2)
mo = CC.CycleModel().fit([CC.Dataset('A', X, genes, alpha=100.0)])
pk_before, _ = mo.peak_phase()
mo.rotate_to_zero(np.degrees(pk_before[0]))
pk_after, _ = mo.peak_phase()
check('rotate_to_zero moves the named angle to 0 and keeps every other gap',
      abs(np.angle(np.exp(1j * pk_after[0]))) < 1e-6
      and np.allclose(np.angle(np.exp(1j * ((pk_after - pk_after[0]) - (pk_before - pk_before[0])))), 0, atol=1e-6))

# -- 5. persistence, gene table, helpers --------------------------------------
print('helpers')
m2 = CC.CycleModel.from_dict(mj.to_dict())
o1, o2 = mj.place('A', XA), m2.place('A', XA)
check('to_dict/from_dict reproduces placement', np.allclose(o1['theta'], o2['theta']) and np.allclose(o1['fit_z'], o2['fit_z'], equal_nan=True))
share = mj.fisher_share('A')
check('fisher shares are a distribution over the panel', abs(sum(share.values()) - 1) < 1e-9 and set(share) == set(genA))
rows = []
for f in (1, 2):
    for c in (1, 2, 3):
        for src, g in ((('RNA', 'Hyb_101', 635), 'GMNN'), (('RNA', 'Hyb_102', 635), 'HMGB2')):
            rows.append({'fov': f, 'cell': c, 'celltype': 'WT', 'modality': src[0], 'hybe': src[1],
                         'channel': src[2], 'n_spots': f * 10 + c})
rows.append({'fov': 3, 'cell': 1, 'celltype': 'WT', 'modality': 'RNA', 'hybe': 'Hyb_101', 'channel': 635, 'n_spots': 5})
rows.append({'fov': 1, 'cell': 1, 'celltype': 'WT', 'modality': 'RNA', 'hybe': 'Hyb_130', 'channel': 635, 'n_spots': 99})
tbl, ct = CC.gene_table(pd.DataFrame(rows), {('RNA', 'Hyb_101', 635): 'GMNN', ('RNA', 'Hyb_102', 635): 'HMGB2'})
check('gene_table pivots named sources, drops incomplete cells and unnamed sources',
      list(tbl.columns) == ['GMNN', 'HMGB2'] and len(tbl) == 6 and (3, 1) not in tbl.index and ct.iloc[0] == 'WT')
sat = CC.saturation_table([{'gene': 'G', 'n_all': 60, 'n_p50': 4}, {'gene': 'H', 'n_all': 40, 'n_p50': 30}])
check('saturation_table flags low acceptance with many candidates', bool(sat.loc['G', 'flag']) and not bool(sat.loc['H', 'flag']))
elim = CC.backward_elimination(mj, 'A', XA, gA, train_group='cycling')
check('backward elimination walks the panel down to two genes with rising error',
      elim[-1]['n_genes'] == 2 and elim[-1]['err_deg'] >= elim[0]['err_deg'] and 'dropped' in elim[1])

print('the per-(dataset, gene) gain')
Xg, thg, _cg, _ag, genesg = CC.simulate(n=400, seed=11, total=(80, 300))
gA = list(genesg)[:6]
gB = list(genesg)[2:8]
dA = CC.Dataset('A', Xg[:, [genesg.index(g) for g in gA]], gA, alpha=100.0)
dB = CC.Dataset('B', Xg[:, [genesg.index(g) for g in gB]], gB, alpha=100.0)

m_off = CC.CycleModel().fit([dA, dB])
check('gains are 1 unless asked for',
      all(abs(v - 1.0) < 1e-12 for d in m_off.gains().values() for v in d.values()))

m_solo = CC.CycleModel().fit([dA], gene_gains=True)
check('a single dataset gets no free gain: there a gain is redundant with (b, c)',
      all(abs(v - 1.0) < 1e-12 for v in m_solo.gains('A').values()),
      str(m_solo.gains('A')))

m_on = CC.CycleModel().fit([dA, dB], gene_gains=True)
shared = [g for g in gA if g in gB]
only_a = [g for g in gA if g not in gB]
check('only the SHARED genes get a free gain',
      all(abs(m_on.gains('A')[g] - 1.0) < 1e-12 for g in only_a),
      str({g: round(m_on.gains('A')[g], 3) for g in only_a}))
check('a shared gene does move off 1',
      any(abs(m_on.gains('A')[g] - 1.0) > 1e-6 for g in shared),
      str({g: round(m_on.gains('A')[g], 3) for g in shared}))
for name in ('A', 'B'):
    lg = [np.log(m_on.gains(name)[g]) for g in shared]
    check(f'{name}: the free log gains average to zero, so only RATIOS are claimed',
          abs(float(np.mean(lg))) < 1e-6, f'{np.mean(lg):.2e}')

spec = m_on.to_dict()
back = CC.CycleModel.from_dict(spec)
check('the gains survive a round trip',
      all(abs(back.gains(n)[g] - m_on.gains(n)[g]) < 1e-12 for n in ('A', 'B') for g in back.panels[n]))
old = {k: v for k, v in spec.items() if k != 's'}
check('a model written before gains existed reads back with every gain at 1',
      all(abs(v - 1.0) < 1e-12 for d in CC.CycleModel.from_dict(old).gains().values() for v in d.values()))
# WHEN NOT TO USE IT. On synthetic pairs built WITH a known gain, the
# fitted ratios track the truth at correlation 0.07 (3 shared genes),
# 0.11 (5), 0.66 (8), 0.29 (12) over three repeats each at 1500 cells
# per dataset -- and at 12 genes two of three repeats reached a LOWER
# evidence than the same fit without gains. The parameter is right in
# principle (the five genes chr19 and JP_001 share do differ in depth
# by up to 2.4x, measured on 20 bootstrap refits) but is not estimable
# at this depth, and it destabilises the EM. Holding the gains at 1
# until the angles settle (gain_warmup=10) repairs the EM: with 12
# shared genes the correlation goes 0.41 -> 0.90 and the evidence is
# then always above the fit without gains; with 8 it is 0.63 -> 0.67,
# with 5 only -0.06 -> 0.35. On the real pair (5 shared) the warm-up
# changes nothing: TPX2 still comes out at 0.2 against a measured 1.24.
# Hence: opt-in, off, and not for panels sharing fewer than ~12 genes.


print('DAPI as an observation of the joint fit')
rng_d = np.random.default_rng(5)
Xd, thd, _cd, _ad, gd = CC.simulate(n=500, seed=21, total=(80, 300))
Xe, the, _ce, _ae, _ge = CC.simulate(n=500, seed=22, total=(80, 300))
gD, gE = list(gd)[:6], list(gd)[2:8]
def _dapi(th):
    return np.log1p((np.asarray(th) % (2 * np.pi)) / (2 * np.pi)) + rng_d.normal(0, 0.1, len(th))
dD = CC.Dataset('D', Xd[:, [gd.index(g) for g in gD]], gD, alpha=100.0, dapi=_dapi(thd))
dE = CC.Dataset('E', Xe[:, [gd.index(g) for g in gE]], gE, alpha=100.0, dapi=_dapi(the))
m_nod = CC.CycleModel().fit([dD, dE])
check('without dapi_weight the fit never looks at DAPI', m_nod.h is None and m_nod.dapi_weight == 0.0)
m_d = CC.CycleModel().fit([dD, dE], dapi_weight=0.5)
check('with dapi_weight the shared curve exists on the grid with mean 0',
      m_d.h is not None and len(m_d.h) == m_d.T and abs(float(m_d.h.mean())) < 1e-9)
# the q-weighted mean over broad posteriors, smoothed like the spectrum,
# flattens a log-2 sawtooth to about 0.3 here; what matters is that a
# cycle came out at all, not its full height
check('the curve is a cycle, not flat (a log-2 sawtooth reads about 0.3 after the smoothing)',
      0.15 < float(m_d.h.max() - m_d.h.min()) < 1.0, f'{m_d.h.max() - m_d.h.min():.2f}')
check('every dataset gets an offset and a residual scale',
      set(m_d.c) == {'D', 'E'} and all(v > 0 for v in m_d.sigma.values()))
th_pl = np.degrees(m_d.place('D', dD.X)['theta']) % 360.0
check('placement stays count-only: it runs without DAPI and returns an angle per cell',
      len(th_pl) == len(dD.X) and np.all(np.isfinite(th_pl)))
back_d = CC.CycleModel.from_dict(m_d.to_dict())
check('the DAPI curve survives a round trip',
      back_d.h is not None and np.allclose(back_d.h, m_d.h) and back_d.c == m_d.c and back_d.dapi_weight == 0.5)


# -- 12. a ring that folds through its centre ---------------------------------
print('the on-ring gate at a fold')
# A ring that passes exactly through its own centre at 90 deg: with K=2
# the composition there is c1 - b2 per gene, so b2 = c1 puts every gene
# at its intercept at that angle. Both real panels without a gene that
# separates post-M from G2/M have this shape right after division, and
# the old fixed gate (bf_ring >= 0, radius >= 0.5) empties those angles:
# it compares the cell with the centre, and at a fold the centre IS the
# ring.
fold = np.zeros((6, 4))
fold[:, 0] = np.random.default_rng(5).normal(0, 1.0, 6)
fold[:, 1] = np.random.default_rng(6).normal(0, 1.0, 6)
fold[:, 2] = fold[:, 1]
Xf, thf, _, _, _ = CC.simulate(n=600, genes=genA, coef=fold, seed=7, alpha=100.0)
mf = CC.CycleModel().fit([CC.Dataset('F', Xf, genA, alpha=100.0)])
mf.coef[mf.cols['F']] = fold                       # the model as truth, fold included
ringf = CC.clr(np.exp(mf.log_pi('F')))
dist = np.linalg.norm(ringf - ringf.mean(0), axis=1)
check('the ring folds through its centre', dist.min() / np.median(dist) < 0.3,
      f'{dist.min() / np.median(dist):.2f} of the radius at {deg(mf.grid[int(np.argmin(dist))] / CC.TWO_PI):.0f} deg')
mf.ring_reference('F', np.full(400, 150.0), n_sim=200, seed=1)
rngf = np.random.default_rng(2)
pisf = np.exp(mf.log_pi('F'))
keep_fixed, keep_gate = [], []
for t in range(0, mf.T, 6):
    Xs = CC._simulate_cells(rngf, pisf[t], 100.0, 150, 200)
    pl = mf.place('F', Xs)
    keep_fixed.append(((pl['bf_ring'] >= 0) & (pl['radius'] >= 0.5)).mean())
    keep_gate.append(mf.on_ring('F', Xs, miss=0.05, placed=pl)[0].mean())
check('the old fixed gate empties the fold', min(keep_fixed) < 0.5, f'worst angle keeps {min(keep_fixed):.2f}')
check('the gate keeps true on-ring cells at every angle, the fold included',
      min(keep_gate) >= 0.8, f'worst angle keeps {min(keep_gate):.2f}, mean {np.mean(keep_gate):.2f}')
mf2 = CC.CycleModel.from_dict(mf.to_dict())
Xchk = CC._simulate_cells(rngf, pisf[0], 100.0, 150, 100)
check('the reference survives to_dict / from_dict',
      'F' in mf2.ring_ref and np.array_equal(mf2.on_ring('F', Xchk)[0], mf.on_ring('F', Xchk)[0]))
plchk = mf.place('F', Xchk)
check('without a reference on_ring falls back to the fixed gate',
      np.array_equal(CC.CycleModel.from_dict({**mf.to_dict(), 'ring_ref': {}}).on_ring('F', Xchk)[0],
                     (plchk['bf_ring'] >= 0) & (plchk['radius'] >= 0.5)))
check('place carries the level and its tail', np.isfinite(plchk['ring_level']).all() and np.isfinite(plchk['ring_tail']).all())

# -- 13. a ring that does NOT fold, and cells that are not cells --------------
print('the on-ring gate on a circular ring')
# an exact circle in CLR: log pi_t = a + u cos t + v sin t with u, v
# mean-zero, orthogonal and of equal length, so every angle sits the same
# distance from the centre. Here the fixed gate misses almost nothing --
# but it also passes compositions that have nothing to do with the ring,
# because 'far from the centre' is all it ever asked.
rngc = np.random.default_rng(11)
u = rngc.normal(0, 1, 6); u -= u.mean()
v = rngc.normal(0, 1, 6); v -= v.mean()
v -= (v @ u) / (u @ u) * u
v *= np.linalg.norm(u) / np.linalg.norm(v)
circ = np.zeros((6, 4)); circ[:, 0] = u; circ[:, 1] = v
Xc0, _thc, _, _, _ = CC.simulate(n=600, genes=genA, coef=circ, seed=12, alpha=100.0)
mc = CC.CycleModel().fit([CC.Dataset('C1', Xc0, genA, alpha=100.0)])
mc.coef[mc.cols['C1']] = circ
rc = CC.clr(np.exp(mc.log_pi('C1')))
dc = np.linalg.norm(rc - rc.mean(0), axis=1)
check('the ring is a circle: every angle the same distance from the centre',
      (dc.max() - dc.min()) / np.median(dc) < 0.1, f'spread {(dc.max() - dc.min()) / np.median(dc):.3f} of the radius')
mc.ring_reference('C1', np.full(400, 150.0), n_sim=200, seed=2)
rngc2 = np.random.default_rng(13)
pisc = np.exp(mc.log_pi('C1'))
keep_c = []
for t in range(0, mc.T, 6):
    Xs = CC._simulate_cells(rngc2, pisc[t], 100.0, 150, 200)
    keep_c.append(mc.on_ring('C1', Xs, miss=0.05)[0].mean())
check('a sound panel keeps its cells at every angle', min(keep_c) >= 0.85,
      f'worst angle keeps {min(keep_c):.2f}, mean {np.mean(keep_c):.2f}')
Xrand = np.array([rngc2.multinomial(150, p) for p in rngc2.dirichlet(np.ones(6), 400)], float)
perm = rngc2.permutation(6)
Xscr = CC._simulate_cells(rngc2, pisc[0][perm], 100.0, 150, 400)
for labl, Xbad in (('a random composition', Xrand), ('the ring composition with its genes scrambled', Xscr)):
    plb = mc.place('C1', Xbad)
    fx = ((plb['bf_ring'] >= 0) & (plb['radius'] >= 0.5)).mean()
    gt = mc.on_ring('C1', Xbad, miss=0.05, placed=plb)[0].mean()
    check(f'the gate rejects {labl}', gt <= 0.25, f'gate passes {gt:.2f}, the old fixed gate passed {fx:.2f}')

# -- 14. orient_by must hold for a single dataset too -------------------------
print('orientation of a single fit')
# orient_by used to be applied only inside stagewise_init, which a
# single-dataset fit never reaches, so the handedness came from the PCA
# seed: two fits of the same cells could come out mirrored.
pk_true = np.degrees(np.arctan2(coef12[:, 1], coef12[:, 0])) % 360
early1 = [allg[i] for i in np.argsort(pk_true)[:3]]
late1 = [allg[i] for i in np.argsort(pk_true)[5:8]]
gens = [g for g in genA if g in early1 + late1] or genA
m1 = CC.CycleModel().fit([CC.Dataset('S1', XA_c, genA, alpha=100.0)], orient_by=(early1, late1))
pk1, _ = m1.peak_phase()
e1 = [m1.gi[g] for g in early1 if g in m1.gi]
l1 = [m1.gi[g] for g in late1 if g in m1.gi]
sep = float(np.degrees(np.angle(np.mean(np.exp(1j * pk1[l1]))) - np.angle(np.mean(np.exp(1j * pk1[e1]))))) % 360
check('a single fit honours orient_by: the early mean peak sits at 0', abs(deg(np.angle(np.mean(np.exp(1j * pk1[e1]))) / CC.TWO_PI)) < 6,
      f'{deg(np.angle(np.mean(np.exp(1j * pk1[e1]))) / CC.TWO_PI):.1f} deg')
check('and the late genes lie in the forward half turn', sep < 180, f'{sep:.0f} deg')
m2 = CC.CycleModel().fit([CC.Dataset('S1', XA_c, genA, alpha=100.0)], orient_by=(early1, late1))
med2, sgn2, _sh2 = CC.circular_agreement(m1.place('S1', XA_c)['theta'], m2.place('S1', XA_c)['theta'])
check('two fits of the same cells keep the same handedness', sgn2 > 0, 'mirrored' if sgn2 < 0 else 'same')

# -- 15. depth-dependent dispersion: alpha_i = alpha (s_i / s_ref)^beta --------
print('depth-dependent alpha')
from scipy.special import gammaln as _gl, logsumexp as _lse   # noqa: E402


def manual_evidence(model, name, Xm, alpha_i):
    pi_ = np.exp(model.log_pi(name))
    A = alpha_i[:, None, None] * pi_[None]
    ll = (_gl(Xm[:, None, :] + A) - _gl(A)).sum(2)
    s_ = Xm.sum(1)
    return (_lse(ll, axis=1) - np.log(model.T) + _gl(s_ + 1) - _gl(Xm + 1).sum(1)
            + _gl(alpha_i) - _gl(s_ + alpha_i))


Xs20 = X[:20]
ev_scalar = m.evidence('A', Xs20, prior='uniform')
check('beta 0 is the single-alpha likelihood exactly',
      np.allclose(ev_scalar, manual_evidence(m, 'A', Xs20, np.full(20, 100.0)), atol=1e-8))
m.alpha_beta['A'], m.alpha_sref['A'] = 0.7, 150.0
ai = 100.0 * (Xs20.sum(1) / 150.0) ** 0.7
check('a per-cell alpha is read at each cell\'s own depth',
      np.allclose(m.evidence('A', Xs20, prior='uniform'), manual_evidence(m, 'A', Xs20, ai), atol=1e-8))
bf_b = m.bf_ring('A', Xs20)
check('bf_ring takes the per-cell alpha too (finite, differs from beta 0)',
      np.isfinite(bf_b).all() and not np.allclose(bf_b, (m.alpha_beta.__setitem__('A', 0.0) or m.bf_ring('A', Xs20))))
m.alpha_beta['A'] = 0.0
check('back at beta 0 the evidence is unchanged', np.allclose(m.evidence('A', Xs20, prior='uniform'), ev_scalar))

# synthetic panels: one whose dispersion grows with depth, one with a single alpha
Xd, thd, _cd, _ad, gd = CC.simulate(n=1500, total=(30, 900), alpha=100.0, alpha_beta=1.0, seed=7)
Xf, thf, _cf, _af, gf = CC.simulate(n=1500, total=(30, 900), alpha=100.0, seed=7)
fov_lab = np.random.default_rng(3).integers(0, 10, 1500)
t0 = time.time()
md = CC.CycleModel().fit([CC.Dataset('D', Xd, gd, alpha=100.0)])
mf = CC.CycleModel().fit([CC.Dataset('F', Xf, gf, alpha=100.0)])
GRID = dict(alphas=(10, 30, 100, 300, 1000), betas=(0.0, 0.5, 1.0, 1.5), folds=3, jobs=1)
rep_d = CC.calibrate_dispersion(md, 'D', Xd, fov_lab, **GRID)
rep_f = CC.calibrate_dispersion(mf, 'F', Xf, fov_lab, **GRID)
check('the depth-dependent panel calibrates to beta > 0 and gains over one alpha',
      rep_d['adopted'][1] >= 0.5 and rep_d['gain'] > CC.DISPERSION_MIN_GAIN,
      f"adopted {rep_d['adopted']}, gain {rep_d['gain']:+.3f}/cell")
check('the single-alpha panel keeps one alpha (its gain does not clear the threshold)',
      rep_f['adopted'][1] == 0.0 and rep_f['gain'] < CC.DISPERSION_MIN_GAIN,
      f"adopted {rep_f['adopted']}, gain {rep_f['gain']:+.3f}/cell, {time.time() - t0:.0f}s")
check('the report carries the per-depth table and the reference depth',
      all(len(v) == 5 and np.isfinite(v).all() for v in rep_d['by_depth'].values())
      and rep_d['s_ref'] > 0)
check('applying the calibration drops the ring reference (the caller rebuilds it)',
      'D' not in md.ring_ref and md.alpha_beta['D'] == rep_d['adopted'][1])

a_b, b_b = rep_d['adopted']

mb = md                                            # the calibrated model, profiles fitted at one alpha
ms = CC.CycleModel().fit([CC.Dataset('D', Xd, gd, alpha=rep_d['best_scalar'])])
medb, _, _ = CC.circular_agreement(thd, mb.phase('D', Xd)[0])
meds, _, _ = CC.circular_agreement(thd, ms.phase('D', Xd)[0])
check('the calibration leaves the placements where one alpha put them', deg(medb) <= deg(meds) + 1.0,
      f'{deg(medb):.1f} vs {deg(meds):.1f} deg against the truth')
sd_ = Xd.sum(1)
tert = np.searchsorted(np.quantile(sd_, [1 / 3, 2 / 3]), sd_, side='right')
onr = {}
for lab_, mm in (('alpha(s)', mb), ('one alpha', ms)):
    mm.ring_reference('D', sd_)
    ok_, _t = mm.on_ring('D', Xd)
    onr[lab_] = [float(ok_[tert == j].mean()) for j in range(3)]
check('with alpha(s) the gate keeps true cells at every depth (>= 0.88 in each tertile)', min(onr['alpha(s)']) >= 0.88,
      f"alpha(s) {np.round(onr['alpha(s)'], 2).tolist()}, one alpha {np.round(onr['one alpha'], 2).tolist()}")
check('and more evenly than one alpha does',
      np.ptp(onr['alpha(s)']) < np.ptp(onr['one alpha']), f"spread {np.ptp(onr['alpha(s)']):.2f} vs {np.ptp(onr['one alpha']):.2f}")
mr = CC.CycleModel.from_dict(mb.to_dict())
check('alpha_beta and alpha_sref survive to_dict / from_dict',
      mr.alpha_beta['D'] == mb.alpha_beta['D'] and mr.alpha_sref['D'] == mb.alpha_sref['D']
      and np.allclose(mr.evidence('D', Xd[:50], prior='uniform'), mb.evidence('D', Xd[:50], prior='uniform')))
old = mb.to_dict()
old.pop('alpha_beta')
old.pop('alpha_sref')
check('a model saved before the exponent existed reads as one alpha',
      CC.CycleModel.from_dict(old).alpha_of('D', Xd[:5]) == a_b)
me_b = CC.CycleModel(mstep='dm').fit([CC.Dataset('D', Xd, gd, alpha=a_b, alpha_beta=b_b,
                                                 alpha_sref=rep_d['s_ref'])], n_iter=15)
mede, _, _ = CC.circular_agreement(mb.phase('D', Xd)[0], me_b.phase('D', Xd)[0])
check('the exact DM M-step runs with a per-cell alpha and agrees with the fast path', deg(mede) < 8, f'{deg(mede):.1f} deg')


print(f'\n{len(PASS)} passed, {len(FAIL)} failed' + (f': {FAIL}' if FAIL else ''))
sys.exit(1 if FAIL else 0)
