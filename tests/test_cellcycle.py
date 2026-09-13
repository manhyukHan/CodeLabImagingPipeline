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

print(f'\n{len(PASS)} passed, {len(FAIL)} failed' + (f': {FAIL}' if FAIL else ''))
sys.exit(1 if FAIL else 0)
