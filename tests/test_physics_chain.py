"""The physics term behaves the way the loss needs it to.

NOT A RE-FIT. The constants were measured once (physics/MEASUREMENT.md)
on 7 million pairs across five panels; nothing here re-derives them.
What these pin is the SHAPE of the law, because the shape is what the
path loss relies on and what a careless edit would quietly break:

  - the scale moves with genomic separation. A constant radius is the
    defect being replaced -- `reach` was 1.0 um at every s, too generous
    where two adjacent loci cannot be a micron apart and too tight where
    1.3 um is ordinary.
  - the cost SATURATES. A far locus must be expensive and not
    impossible, or one bad assignment dominates a path and we are back
    to `d = where(d <= reach, d, inf)`.
  - z is wider than xy in the instrument and in the contaminant, and
    NOT in the polymer.

Run: python tests/test_physics_chain.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from codelab_pipeline import physics as PH
from codelab_pipeline.physics import chain as C

_n = {'pass': 0, 'fail': 0}


def check(name, ok, extra=''):
    _n['pass' if ok else 'fail'] += 1
    print('  %-4s %s%s' % ('ok' if ok else 'FAIL', name,
                           ('   [%s]' % extra) if extra else ''))


def along(r, axis=0):
    d = np.zeros(3)
    d[axis] = r
    return d


def half_point(s, cal, axis=0):
    """|d| where P(same chain) crosses 0.5 -- the s-dependent 'reach'."""
    lo, hi = 0.0, 8.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if PH.p_same_chain(along(mid, axis), s, cal) > 0.5:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_the_scale_follows_the_genome():
    print('\nthe scale is a function of s, which is the whole point')
    cal = PH.Calibration()
    sp = [cal.sigma_poly(s) for s in (25, 100, 1000, 4000)]
    check('sigma_poly rises with s', all(a < b for a, b in zip(sp, sp[1:])),
          ' '.join('%.3f' % v for v in sp))
    # the exponent is sub-diffusive: confined chromatin, not an ideal
    # chain (0.5) and not a rod (1.0)
    check('alpha is in the confined-chromatin range',
          0.15 < C.SIGMA_POLY_ALPHA < 0.40, '%.3f' % C.SIGMA_POLY_ALPHA)
    hp = [half_point(s, cal) for s in (25, 100, 1000, 2000)]
    check('the half-probability distance rises with s too',
          all(a < b for a, b in zip(hp, hp[1:])),
          ' '.join('%.2f' % v for v in hp))
    # the defect being replaced, stated as a test: a single radius
    # cannot sit where the law does at both ends
    check('it spans the old constant radius rather than matching it',
          hp[0] < 1.0 < hp[-1], '%.2f .. %.2f um' % (hp[0], hp[-1]))


def test_the_cost_saturates():
    print('\na far locus is expensive, not impossible')
    cal = PH.Calibration()
    s = 100.0
    r = np.array([0.2, 0.5, 1.0, 2.0, 4.0, 8.0])
    cost = np.array([float(PH.pair_cost(along(x), s, cal)) for x in r])
    check('cost increases with distance',
          all(a < b for a, b in zip(cost, cost[1:])),
          ' '.join('%.1f' % v for v in cost))
    # a pure Gaussian doubles its exponent fourfold when r doubles; the
    # mixture must grow far slower once the contaminant takes over
    quad = (cost[-1] - cost[-2]) / max(cost[2] - cost[1], 1e-9)
    pure = float(PH.pair_cost(along(8.0), s, cal)) / max(
        0.5 * (8.0 ** 2) / cal.chain_var(s)[0], 1e-9)
    check('growth is sub-quadratic far out', pure < 0.25, 'ratio %.3f' % pure)
    check('and every cost is finite -- no `inf` wall',
          bool(np.isfinite(cost).all()))
    far = float(PH.pair_cost(along(50.0), s, cal))
    check('even an absurd separation stays finite', np.isfinite(far),
          '%.1f at 50 um' % far)


def test_the_posterior_is_a_probability():
    print('\nP(same chain) is a probability a reviewer can read')
    cal = PH.Calibration()
    p = [float(PH.p_same_chain(along(x), 100.0, cal))
         for x in (0.05, 0.3, 0.7, 1.5, 5.0)]
    check('it is bounded', all(0.0 <= v <= 1.0 for v in p),
          ' '.join('%.3f' % v for v in p))
    check('it falls with distance', all(a > b for a, b in zip(p, p[1:])))
    check('a near pair is almost certainly one chain', p[0] > 0.95,
          '%.4f' % p[0])
    check('a far pair is almost certainly not', p[-1] < 0.05,
          '%.4f' % p[-1])


def test_z_is_wider_where_it_should_be():
    print('\nanisotropy sits in the instrument and the contaminant, not the chain')
    check('the instrument is worse in z', C.SIGMA_LOC_Z > C.SIGMA_LOC_XY,
          '%.3f vs %.3f' % (C.SIGMA_LOC_Z, C.SIGMA_LOC_XY))
    # MEASURED on four panels: Tz/Txy 1.69-2.47, no exception. That is
    # the second allele stacked in z.
    check('the contaminant is stretched in z', C.TAU_Z > 1.5 * C.TAU_XY,
          'ratio %.2f' % (C.TAU_Z / C.TAU_XY))
    cal = PH.Calibration()
    # the POLYMER is isotropic: the only difference between the axes of
    # the chain component is the instrument term
    vxy, vz = cal.chain_var(400.0)
    gap = vz - vxy
    want = 2 * (cal.sigma_loc_z ** 2 - cal.sigma_loc_xy ** 2)
    check('the chain differs between axes ONLY by the instrument',
          abs(gap - want) < 1e-12, '%.5f vs %.5f' % (gap, want))


def test_the_calibration_is_the_store_s_own():
    print('\nsigma_loc comes from the store, robustly')
    rng = np.random.default_rng(7)
    true_xy, true_z = 0.020, 0.060
    n = 4000
    v = rng.normal(0, np.sqrt(2) * true_xy, (n, 3))
    v[:, 2] = rng.normal(0, np.sqrt(2) * true_z, n)
    # a fifth of the repeats land on the wrong allele: the tail that
    # made a plain moment report sigma_loc an order of magnitude high
    bad = rng.random(n) < 0.2
    v[bad] += rng.normal(0, 1.2, (int(bad.sum()), 3))
    cal = PH.calibration_from_repeats(v)
    check('lateral is recovered despite the tail',
          abs(cal.sigma_loc_xy - true_xy) < 0.2 * true_xy,
          '%.4f vs %.4f' % (cal.sigma_loc_xy, true_xy))
    check('axial is recovered too',
          abs(cal.sigma_loc_z - true_z) < 0.2 * true_z,
          '%.4f vs %.4f' % (cal.sigma_loc_z, true_z))
    check('a plain mean square would NOT be',
          abs(np.sqrt(np.mean(v[:, 0] ** 2) / 2) - true_xy) > 0.5 * true_xy,
          'rms gives %.4f' % np.sqrt(np.mean(v[:, 0] ** 2) / 2))
    check('it says where it came from', 'repeats' in cal.source, cal.source)
    try:
        PH.calibration_from_repeats(np.zeros((5, 3)))
        check('too few repeats raises', False)
    except ValueError:
        check('too few repeats raises', True)


def test_saturation_reports_the_degenerate_regime():
    print('\nsaturation_um says when a pair carries no information')
    cal = PH.Calibration()
    near = PH.saturation_um(50.0, cal)
    far = PH.saturation_um(4000.0, cal)
    check('a short-range pair has a crossing', np.isfinite(near),
          '%.3f um' % near)
    # at long range the chain is broad enough that nothing is better
    # explained by the contaminant -- the same degeneracy that makes the
    # mixture weight unidentifiable there
    check('a long-range pair does not', far == float('inf'), str(far))


def main():
    for t in (test_the_scale_follows_the_genome,
              test_the_cost_saturates,
              test_the_posterior_is_a_probability,
              test_z_is_wider_where_it_should_be,
              test_the_calibration_is_the_store_s_own,
              test_saturation_reports_the_degenerate_regime):
        t()
    print('\n%d passed, %d failed' % (_n['pass'], _n['fail']))
    return 1 if _n['fail'] else 0


if __name__ == '__main__':
    sys.exit(main())
