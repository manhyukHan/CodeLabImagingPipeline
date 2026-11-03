"""The DAPI arc cut reads 2N off the curve's trough, not off birth.

WHAT THIS PINS. S begins where the DNA content starts to rise above
2N. That needs a 2N level, and the rule used to take it from the first
sixth of the cycle after birth -- which assumes the content is already
2N as soon as division is behind it.

It often is not. Measured on three stores, the stretch just after
birth sits ABOVE the curve's own trough every time, because the
division step is smeared: chr19 reads 1.09 against a trough of 0.86,
JP_001 0.92 against 0.77. The inflated reference pushed S late and
narrow -- chr19's stored S was 20 deg wide and JP_001's 30, on curves
that run 1.68 and 2.13-fold from trough to top. chr19's S then began
80 deg AFTER GMNN and GINS2 peak, which is where S entry actually is.

Two changes, and both are needed. Referencing the trough alone moves
the fault to the other side: those same elevated post-birth bins clear
a trough-referenced threshold FROM ABOVE, so a search that still
starts at birth returns S = 0 deg. The cycle runs birth -> 2N trough
-> rise, so the rise that ends G1 is the one after the trough.

Run: python tests/test_dapi_arcs.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib                                              # noqa: E402
matplotlib.use('Agg')

from codelab_pipeline.analysis import figures_cellcycle as FC  # noqa: E402

PASS, FAIL = [], []

#: the fixture, as bin levels. A SMEARED DIVISION: the first 50 deg
#: after birth have not reached 2N yet, which is what every real store
#: here does and what the old rule mistook for the 2N level.
POST_BIRTH, TROUGH, TOP = 1.10, 0.85, 1.60
RISE_FROM, RISE_TO = 180.0, 280.0


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print('  %s %s%s' % ('ok' if cond else 'FAIL', name,
                         ('   [%s]' % detail) if detail else ''), flush=True)


def level(a):
    if a < 50.0:
        return POST_BIRTH
    if a < 150.0:
        return TROUGH
    if a < RISE_FROM:
        return 0.90
    if a < RISE_TO:
        return TROUGH + (a - RISE_FROM) / (RISE_TO - RISE_FROM) * (TOP - TROUGH)
    return TOP


def population(levels=level, n_per_bin=40, seed=0):
    rng = np.random.default_rng(seed)
    th, dna = [], []
    for a in np.arange(0.0, 360.0, 10.0):
        th.extend([a + 5.0] * n_per_bin)
        dna.extend(levels(a) + rng.normal(0.0, 0.002, n_per_bin))
    return np.array(th), np.array(dna)


th, dna = population()
arcs, info = FC.propose_arcs_from_dapi(th, dna, birth_deg=0.0)
by = {a['name']: a for a in arcs}

# where the rule SHOULD land, from the fixture's own definition
thr = TROUGH * 1.15
want_s = next(a for a in np.arange(RISE_FROM, 360.0, 10.0) if level(a) >= thr)
target = TROUGH + 0.85 * (TOP - TROUGH)
want_g = next(a for a in np.arange(RISE_FROM, 360.0, 10.0) if level(a) >= target)

print('the 2N level')
check('it is the trough, not the stretch after birth',
      abs(info['g1_level'] - TROUGH) < 0.02,
      'g1_level %.3f (trough %.2f, post-birth %.2f)' % (info['g1_level'], TROUGH, POST_BIRTH))
check('and the trough\'s angle is reported', 50.0 <= info['trough_deg'] <= 140.0,
      '%.0f deg' % info['trough_deg'])

print('')
print('where S starts')
check('THE BINS JUST AFTER BIRTH DO NOT START S, though they clear the threshold',
      POST_BIRTH >= thr and by['S']['start_deg'] > 100.0,
      'post-birth %.2f >= threshold %.3f, yet S starts at %.0f deg'
      % (POST_BIRTH, thr, by['S']['start_deg']))
check('S starts at the rise', abs(by['S']['start_deg'] - want_s) < 1e-6,
      '%.0f deg (the fixture crosses %.3f at %.0f)' % (by['S']['start_deg'], thr, want_s))
check('G1 runs from birth to there',
      by['G1']['start_deg'] == 0.0 and by['G1']['end_deg'] == by['S']['start_deg'])

print('')
print('where G2/M starts -- the plateau rule, unchanged')
check('S ends at the plateau level', abs(by['S']['end_deg'] - want_g) < 1e-6,
      '%.0f deg (the fixture reaches %.3f at %.0f)' % (by['S']['end_deg'], target, want_g))
check('G2/M runs from there back to birth',
      by['G2/M']['start_deg'] == by['S']['end_deg'] and by['G2/M']['end_deg'] == 0.0)

print('')
print('the old rule would have failed this fixture')
# the first sixth after birth is bins 0-50, every one of them POST_BIRTH
old_g1 = POST_BIRTH
old_thr = old_g1 * 1.15
old_s = next((a for a in np.arange(0.0, 360.0, 10.0) if level(a) >= old_thr), None)
check('its 2N level would be the post-birth plateau', abs(old_g1 - POST_BIRTH) < 1e-9)
check('and its S would start late, or not at all',
      old_s is None or old_s > want_s + 20.0,
      'old %s vs new %.0f deg' % ('never' if old_s is None else '%.0f' % old_s, want_s))

print('')
print('a curve with no rise is still refused, and says what it compared')
flat_th, flat_dna = population(levels=lambda a: 1.0)
try:
    FC.propose_arcs_from_dapi(flat_th, flat_dna, birth_deg=0.0)
    check('a flat curve raises', False, 'it returned arcs')
except ValueError as e:
    check('a flat curve raises ValueError', True)
    check('and the message names the trough, not a post-birth level',
          'trough' in str(e) and 'post-birth' not in str(e), str(e)[:88])

print('')
print('%d/%d passed' % (len(PASS), len(PASS) + len(FAIL)))
if FAIL:
    print('failed: ' + ', '.join(FAIL))
sys.exit(1 if FAIL else 0)
