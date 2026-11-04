"""The origin's on-ring subset is its own rule, not the verdict gate.

WHAT THIS PINS. The division origin is measured on the cells that are
on the ring, because cells that are not do not divide on this circle
and mixing them in dilutes the step (chr19: x1.39 at 0 deg over every
training cell, x1.63 at 40 deg over the on-ring ones).

That subset used to be read out of the panel's VERDICT gate dict. The
two are different decisions -- a verdict gate says what an angle may be
CALLED, this says which cells a STEP is measured ON -- and reading one
off the other had a consequence: the moment the verdict gates became
off by default, the subset became every cell, while the stored origin
still reported 'on-ring training cells'.

So: no gates parameter, and the origin dict says which rule actually
ran.

WHICH rule that is remains open (ORIGIN_SUBSET, provisionally
`radius`), because on chr19 the four candidates disagree not only on
the verdict but on the angle, by 50 deg -- the mark of a proxy with no
step in it rather than of one right subset. A fit therefore does not
settle it quietly: it measures every candidate and stores them all
under `subset_survey`, so the choice can be made later from the record
instead of from whoever happened to run the fit.

Run: QT_QPA_PLATFORM=offscreen python tests/test_origin_subset.py
"""
import inspect
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from codelab_pipeline.analysis import cellcycle as CC            # noqa: E402
from windows import cellcycle_wiring as CW                       # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print('  %s %s%s' % ('ok' if cond else 'FAIL', name,
                         ('   [%s]' % detail) if detail else ''), flush=True)


# -- the signature -------------------------------------------------------------
params = list(inspect.signature(CW.CellCycleWiring._settle_origin).parameters)
check('_settle_origin takes no verdict gates', 'gates' not in params, ', '.join(params))


# -- a population half of which is not on the ring -----------------------------
NAME = 'E'
X0, theta0, _coef, _a, genes = CC.simulate(n=900, seed=3)
m = CC.CycleModel().fit([CC.Dataset(NAME, X0, genes, alpha=100.0)])
rng = np.random.default_rng(11)

th0 = np.degrees(m.place(NAME, X0)['theta']) % 360.0
# the SAME compositions redrawn at two depths, so the panel total steps
# x2 at one angle and the angles themselves are untouched
tot = np.where(th0 < 200.0, 300, 150)
p0 = X0 / np.maximum(X0.sum(1, keepdims=True), 1e-9)
X_on = np.array([rng.multinomial(t, p) for t, p in zip(tot, p0)], float)

# cells drawn from the ring's CENTRE: a composition with no angle
centre = np.asarray(m.centre(NAME), float)
centre = centre / centre.sum()
tot_off = rng.choice([150, 300], 350)
X_off = np.array([rng.multinomial(t, centre) for t in tot_off], float)

X_tr = np.vstack([X_on, X_off])
is_off = np.zeros(len(X_tr), bool)
is_off[len(X_on):] = True
fov_tr = np.ones(len(X_tr), int)
cell_tr = np.arange(len(X_tr))

m.ring_reference(NAME, X_tr.sum(1), n_sim=200, seed=1)
on, _tail = m.on_ring(NAME, X_tr)
check('the centre-drawn cells are the ones ring_level drops',
      on[~is_off].mean() > 0.80 and on[is_off].mean() < 0.35,
      'on-ring %.2f, centre-drawn %.2f' % (on[~is_off].mean(), on[is_off].mean()))

# the surviving half of the withdrawn pair, on the same cells. This
# ring does not fold through its own centre, and there radius is not
# worse -- it keeps slightly MORE on-ring cells than the calibrated
# gate. That is the whole shape of the thing: a centre-referenced rule
# is wrong only where the centre stops being a reference, which is at a
# fold, and test_cellcycle's fold fixture is where that shows.
rad = np.asarray(m.place(NAME, X_tr)['radius'], float) >= 0.5
check('on an unfolded ring radius alone agrees with the calibrated gate',
      rad[is_off].mean() < 0.35 and abs(rad[~is_off].mean() - on[~is_off].mean()) < 0.10,
      'radius keeps on-ring %.2f / centre %.2f; on_ring keeps %.2f / %.2f'
      % (rad[~is_off].mean(), rad[is_off].mean(), on[~is_off].mean(), on[is_off].mean()))


# -- the origin itself ---------------------------------------------------------
origin, birth_used, dapi_tab = CW.CellCycleWiring._settle_origin(
    None, m, NAME, X_tr, fov_tr, cell_tr, None, True, None, 0.0)

check('the step was found', origin['mode'].startswith('division (total drop)'),
      '%s, drop x%.2f' % (origin['mode'], origin.get('drop_factor', float('nan'))))
check('the subset is a strict subset and still big enough',
      CW.MIN_CELLS_FOR_ORIGIN <= origin.get('n_cells', 0) < len(X_tr),
      '%s of %d' % (origin.get('n_cells'), len(X_tr)))
check('a found division puts birth at 0 and reads no DAPI table',
      birth_used == 0.0 and dapi_tab is None)


# -- the choice is recorded, and so are the ones not taken ---------------------
# ORIGIN_SUBSET is provisional: chr19's candidates disagree on the
# VERDICT and on the ANGLE by 50 deg, which says the proxy has no step
# in it rather than that one subset is right. So a fit does not get to
# make that decision quietly -- it measures them all and stores them.
# There were four until 2026-11-04; the fourth was the bf_ring pair,
# which went when bf_ring did.
survey = origin.get('subset_survey') or {}
check('every candidate subset is surveyed, not only the one used',
      set(survey) == {'radius', 'on_ring', 'all'}
      and sum(bool(r['used']) for r in survey.values()) == 1,
      ', '.join('%s n=%d%s' % (k, v['n'], ' USED' if v['used'] else '')
                for k, v in sorted(survey.items())))
check('measured_on names the rule that actually ran, and the survey agrees',
      origin.get('measured_on') == survey.get(CW.ORIGIN_SUBSET, {}).get('rule')
      and survey[CW.ORIGIN_SUBSET]['used'],
      '%s   (ORIGIN_SUBSET = %r)' % (origin.get('measured_on'), CW.ORIGIN_SUBSET))
check('the used row carries the very number the origin acted on',
      abs(survey[CW.ORIGIN_SUBSET]['factor'] - origin['drop_factor']) < 1e-9
      and survey[CW.ORIGIN_SUBSET]['angle_deg'] == origin['angle_before_deg'],
      'x%.4f at %.0f deg' % (survey[CW.ORIGIN_SUBSET]['factor'],
                             survey[CW.ORIGIN_SUBSET]['angle_deg']))
check('and the subsets NOT chosen carry their own verdicts',
      all(v['factor'] is not None for v in survey.values()),
      ' | '.join('%s x%.2f@%.0f' % (k, v['factor'], v['angle_deg'])
                 for k, v in sorted(survey.items())))


# -- a model with no reference builds one, and the survey says so -------------
# There is no fallback rule to label any more. Until 2026-11-04 a model
# with no reference answered the withdrawn pair's question under the
# `on_ring` key, so this very check existed to stop that passing
# silently -- and on the real stores it was not hypothetical: JP_001 and
# JP_002 both carried no reference.
m_noref = CC.CycleModel.from_dict({**m.to_dict(), 'ring_ref': {}})
origin2, _b, _d = CW.CellCycleWiring._settle_origin(
    None, m_noref, NAME, X_tr, fov_tr, cell_tr, None, True, None, 0.0)
rule2 = (origin2.get('subset_survey') or {}).get('on_ring', {}).get('rule', '')
check('without a reference the survey says the reference was built here',
      'reference built here' in rule2 and 'fallback' not in rule2 and NAME in m_noref.ring_ref,
      rule2)
check('and it is the same question as a model that came with one',
      abs(origin2['subset_survey']['on_ring']['n']
          - survey['on_ring']['n']) <= 0.05 * len(X_tr),
      'n %d (with a reference: %d)' % (origin2['subset_survey']['on_ring']['n'],
                                       survey['on_ring']['n']))


print('')
print('%d/%d passed' % (len(PASS), len(PASS) + len(FAIL)))
if FAIL:
    print('failed: ' + ', '.join(FAIL))
sys.exit(1 if FAIL else 0)
