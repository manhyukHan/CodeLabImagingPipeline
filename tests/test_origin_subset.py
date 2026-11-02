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

So: no gates parameter, the subset comes from ring_level (the test that
replaced the withdrawn bf_ring pair), and the origin dict says which
rule actually ran.

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

# what the OLD rule would have done with the gates the panel now sends:
# every key present and None, so g.get(key, default) never reaches its
# default and no cell is dropped
old_gates = {'min_bf_ring': None, 'min_radius': None}
old_on = np.ones(len(X_tr), bool)
for key, col in (('min_bf_ring', 'bf_ring'), ('min_radius', 'radius')):
    v = old_gates.get(key, {'min_bf_ring': 0.0, 'min_radius': 0.5}[key])
    if v is not None:
        old_on &= m.place(NAME, X_tr)[col] >= float(v)
check('the old rule would have kept every cell once the gates went off',
      old_on.all() and not on.all(),
      'old %d/%d, new %d/%d' % (old_on.sum(), len(X_tr), on.sum(), len(X_tr)))


# -- the origin itself ---------------------------------------------------------
origin, birth_used, dapi_tab = CW.CellCycleWiring._settle_origin(
    None, m, NAME, X_tr, fov_tr, cell_tr, None, True, None, 0.0)

check('the step was found', origin['mode'].startswith('division (total drop)'),
      '%s, drop x%.2f' % (origin['mode'], origin.get('drop_factor', float('nan'))))
check('it was measured on the on-ring subset, and says so',
      origin.get('measured_on', '').startswith('on-ring training cells (ring_level tail)'),
      origin.get('measured_on', ''))
check('the subset is a strict subset and still big enough',
      CW.MIN_CELLS_FOR_ORIGIN <= origin.get('n_cells', 0) < len(X_tr),
      '%s of %d' % (origin.get('n_cells'), len(X_tr)))
check('a found division puts birth at 0 and reads no DAPI table',
      birth_used == 0.0 and dapi_tab is None)


# -- the honest label when the model carries no reference ----------------------
m_noref = CC.CycleModel.from_dict({**m.to_dict(), 'ring_ref': {}})
origin2, _b, _d = CW.CellCycleWiring._settle_origin(
    None, m_noref, NAME, X_tr, fov_tr, cell_tr, None, True, None, 0.0)
check('without a reference the origin names the fallback rule instead',
      'no ring reference' in origin2.get('measured_on', ''),
      origin2.get('measured_on', ''))


print('')
print('%d/%d passed' % (len(PASS), len(PASS) + len(FAIL)))
if FAIL:
    print('failed: ' + ', '.join(FAIL))
sys.exit(1 if FAIL else 0)
