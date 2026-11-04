"""A refused origin is refused, not answered with a different question.

WHAT THIS PINS. The origin door offers three settings and they are not
approximations of each other:

    DAPI halving    the DNA itself, which doubles over S and halves at
                    division. This is the one that means division.
    panel total     the panel's mRNA, which falls at division but also
                    for its own reasons (G2/M transcripts leaving, the
                    panel's own depth).
    S mean peak     not division at all -- a landmark named after the S
                    genes, used when nothing else is available.

Asked for DAPI, the old code consulted the panel total when DAPI
refused, and then settled for the S mean peak when that refused too.
Measured on chr19 those three angles are 280, 20 and 0 degrees: a
fall-through does not approximate the answer, it replaces it. Worse,
the second branch OVERWROTE the first, so the stored origin carried the
panel total's factor under a mode that named neither measurement, and
the DAPI number that had made the decision was destroyed.

So: one measurement answers, the one that was asked for. A refusal
returns, carrying what it measured, on how many cells, by which rule,
and what 0 deg means instead.

Run: QT_QPA_PLATFORM=offscreen python tests/test_origin_refusal.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from codelab_pipeline.analysis import cellcycle as CC            # noqa: E402
from windows import cellcycle_wiring as CW                       # noqa: E402

PASS, FAIL = [], []
DAPI_SRC = ('DNA', 'DAPI', 635)
BIRTH = 37.0                      # anything but 0, so a rotation shows


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print('  %s %s%s' % ('ok' if cond else 'FAIL', name,
                         ('   [%s]' % detail) if detail else ''), flush=True)


# -- a population whose PANEL TOTAL has a clean x2 step at 200 deg -------------
# the same construction as test_origin_subset: identical compositions
# redrawn at two depths, so the total steps and the angles do not move.
NAME = 'E'
X0, _theta0, _coef, _a, genes = CC.simulate(n=900, seed=3)
m = CC.CycleModel().fit([CC.Dataset(NAME, X0, genes, alpha=100.0)])
rng = np.random.default_rng(11)

th0 = np.degrees(m.place(NAME, X0)['theta']) % 360.0
tot = np.where(th0 < 200.0, 300, 150)
p0 = X0 / np.maximum(X0.sum(1, keepdims=True), 1e-9)
X_tr = np.array([rng.multinomial(t, p) for t, p in zip(tot, p0)], float)
fov_tr = np.ones(len(X_tr), int)
cell_tr = np.arange(len(X_tr))
m.ring_reference(NAME, X_tr.sum(1), n_sim=200, seed=1)

TH = np.degrees(m.place(NAME, X_tr)['theta']) % 360.0
on, _tail = m.on_ring(NAME, X_tr)

# THE CONTROL THAT MAKES THIS TEST MEAN ANYTHING: the panel total does
# have a drop deep enough to pass. So if the DAPI refusal still falls
# through, it lands on a real answer -- the wrong one -- rather than on
# an obvious failure.
t_angle, t_factor = CC.total_drop_angle(TH[on], X_tr[on].sum(1))
check('the panel total WOULD have passed, at its own angle',
      t_factor >= CW.TOTAL_MIN_DROP,
      'total x%.2f at %.0f deg (needs x%.1f)' % (t_factor, t_angle, CW.TOTAL_MIN_DROP))


def dapi_table(values):
    """What mask_intensity_table returns, with sum_above_bg given."""
    return pd.DataFrame({'fov': fov_tr, 'cell': cell_tr,
                         'celltype': '', 'area': 100.0, 'mask_mean': 1.0,
                         'mask_median': 1.0, 'sum_above_bg': np.asarray(values, float),
                         'background': 0.0})


def settle(values, fails=(), model=None):
    """_settle_origin with mask_intensity_table standing in for the store."""
    real = CC.mask_intensity_table
    CC.mask_intensity_table = lambda *a, **k: (dapi_table(values), list(fails))
    try:
        mm = model if model is not None else CC.CycleModel.from_dict(m.to_dict())
        before = np.degrees(mm.place(NAME, X_tr)['theta']) % 360.0
        out = CW.CellCycleWiring._settle_origin(
            None, mm, NAME, X_tr, fov_tr, cell_tr, None, True, DAPI_SRC, BIRTH)
        after = np.degrees(mm.place(NAME, X_tr)['theta']) % 360.0
        return out + (float(np.abs(((after - before + 180) % 360) - 180).max()),)
    finally:
        CC.mask_intensity_table = real


# -- 1. DAPI refuses, and that is the end of it -------------------------------
print('a flat DAPI, while the panel total would have passed')
flat = np.ones(len(X_tr))
origin, birth_used, _tab, moved = settle(flat)

check('the mode says REFUSED', origin.get('mode', '').startswith('REFUSED'),
      origin.get('mode', ''))
check('and carries the flag a reader can branch on', origin.get('refused') is True)
check('the model was NOT rotated', moved < 1e-9, 'max angle moved %.2e deg' % moved)
check('the birth angle is left where it was', birth_used == BIRTH,
      '%s (not 0.0)' % birth_used)
check('DAPI\'s own number survives, under its own name',
      origin.get('dapi_factor') is not None and origin['dapi_factor'] < CW.DAPI_MIN_DROP,
      'dapi_factor %s, needed %s' % (origin.get('dapi_factor'), origin.get('needed')))
check('THE PANEL TOTAL WAS NEVER ASKED', 'total_factor' not in origin,
      'keys: ' + ', '.join(sorted(origin)))
check('and nothing is reported as the drop the origin was placed at',
      'drop_factor' not in origin and 'angle_before_deg' not in origin)
check('the refusal records its subset and its n, which only successes used to',
      origin.get('measured_on') == (origin.get('subset_survey') or {})
      .get(CW.ORIGIN_SUBSET, {}).get('rule')
      and origin.get('n_cells', 0) >= CW.MIN_CELLS_FOR_ORIGIN,
      '%s on %s cells' % (origin.get('measured_on'), origin.get('n_cells')))
check('and a REFUSAL surveys the other subsets too -- which is the one '
      'case where that record is the whole point',
      set(origin.get('subset_survey') or {}) == {'radius', 'on_ring', 'all'},
      ' | '.join('%s x%s' % (k, v['factor'])
                 for k, v in sorted((origin.get('subset_survey') or {}).items())))
check('it says what 0 deg is instead', 'NOT division' in origin.get('zero_is', ''),
      origin.get('zero_is', ''))

txt = CW.origin_text(origin)
check('and the line a user reads says so first', txt.startswith('*** ORIGIN NOT SET ***'), txt[:92])
check('that line carries the number and the bar',
      ('x%.2f' % origin['dapi_factor']) in txt and ('x%.1f' % CW.DAPI_MIN_DROP) in txt)


# -- 2. unreadable FOVs are no longer dropped on the floor --------------------
print('')
print('a DAPI source the store could not read everywhere')
origin_f, _b, _t, _mv = settle(flat, fails=[(7, 'no such hybe'), (9, 'no such hybe')])
check('the failures reach the origin dict', origin_f.get('n_dapi_fovs_unread') == 2
      and origin_f.get('dapi_unread') == [7, 9], str(origin_f.get('dapi_unread')))
check('and the line names them', 'FOV007' in CW.origin_text(origin_f),
      CW.origin_text(origin_f)[-70:])


# -- 3. a real halving is still taken, at DAPI's angle not the total's --------
print('')
print('a DAPI that halves at 60 deg, where the panel total does not step')
# high before 60 deg, low after it: the detector looks for the FALL
halved = np.where(TH < 60.0, 2.0, 1.0)
mm = CC.CycleModel.from_dict(m.to_dict())
origin_ok, birth_ok, tab_ok, moved_ok = settle(halved, model=mm)
check('the mode is the DAPI halving', origin_ok.get('mode') == 'division (DAPI halving)',
      origin_ok.get('mode', ''))
check('the model was rotated', moved_ok > 1.0, 'max angle moved %.0f deg' % moved_ok)
check('to DAPI\'s angle, not the panel total\'s',
      abs(((origin_ok['angle_before_deg'] - 60.0 + 180) % 360) - 180) <= 20.0
      and abs(((origin_ok['angle_before_deg'] - t_angle + 180) % 360) - 180) > 20.0,
      'rotated from %.0f deg; DAPI steps at 60, the total at %.0f'
      % (origin_ok['angle_before_deg'], t_angle))
check('birth goes to 0 and the DAPI table comes back',
      birth_ok == 0.0 and tab_ok is not None)
check('both the placed drop and DAPI\'s own factor are stored',
      origin_ok.get('drop_factor') is not None and origin_ok.get('dapi_factor') is not None,
      'drop x%.2f, dapi x%.2f' % (origin_ok['drop_factor'], origin_ok['dapi_factor']))
check('and the line does not shout', not CW.origin_text(origin_ok).startswith('***'),
      CW.origin_text(origin_ok)[:92])


# -- 4. asked for the panel total, a refusal is also the end ------------------
print('')
print('no DAPI source, and a panel total with no step')
flat_tot = np.array([rng.multinomial(200, p) for p in p0], float)
m2 = CC.CycleModel().fit([CC.Dataset(NAME, flat_tot, genes, alpha=100.0)])
m2.ring_reference(NAME, flat_tot.sum(1), n_sim=200, seed=1)
origin_t, birth_t, tab_t = CW.CellCycleWiring._settle_origin(
    None, m2, NAME, flat_tot, fov_tr, cell_tr, None, True, None, BIRTH)
check('the mode says REFUSED and names the panel total',
      origin_t.get('mode', '').startswith('REFUSED') and 'panel total' in origin_t['mode'],
      origin_t.get('mode', ''))
check('the total\'s number is kept, and no DAPI key is invented',
      origin_t.get('total_factor') is not None and 'dapi_factor' not in origin_t,
      'total x%.2f' % origin_t['total_factor'])
check('birth untouched, no DAPI table read',
      birth_t == BIRTH and tab_t is None)


print('')
print('%d/%d passed' % (len(PASS), len(PASS) + len(FAIL)))
if FAIL:
    print('failed: ' + ', '.join(FAIL))
sys.exit(1 if FAIL else 0)
