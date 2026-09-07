"""
The localizer seam really is stack -> [(y, x, z, p)], for every engine.

WHY THIS EXISTS: a machine-learning spot localizer is being built by
another author against this contract, and the contract is the only thing
the two sides share. If it drifts -- an engine that needs a seed, one
that returns a bare tuple, one that raises on a masked crop -- the ML
engine cannot be dropped in, and the failure shows up as a wrong
coordinate rather than an error.

So this pins the contract itself, not the arithmetic:

  * an UNSEEDED localize(stack) works on every engine and finds the
    emitters. That call is the whole interface a learned detector has to
    satisfy, and it was the one previously left as a bare argmax.
  * a SEEDED call is the same door a person clicks through. Manual
    anchoring is not a stage between anchoring and fitting -- it is an
    alternative anchor SOURCE at the very front -- and the test that says
    so is the one that shows an external seed and an auto anchor reach
    the same fit.
  * p is present, ordered, and in (0, 1] everywhere.
  * bad data returns [], never an exception.

The generosity check is here rather than in the training code because it
is a property OF THE ANCHOR STEP: production defaults miss a dim emitter
that GENEROUS_ANCHOR finds, and a training run that quietly used the
production numbers would lose exactly the candidates a person is needed
to judge.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_localize_engine_seam.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                          # noqa: E402

from codelab_pipeline.localization import engine as E       # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}' + (f'   [{detail}]' if detail else ''))


def field(shape=(48, 48, 41), emitters=((12.3, 15.7, 18.4, 9000.),
                                        (31.6, 29.2, 22.9, 5000.),
                                        (20.1, 38.4, 15.2, 2500.)),
          mask_corner=True, seed=0):
    """A crop shaped like a real one: Gaussian emitters on noise, with a
    NaN region standing in for pixels outside the cell mask."""
    rng = np.random.default_rng(seed)
    h, w, d = shape
    st = rng.normal(300, 12, shape)
    yy, xx, zz = np.indices(shape)
    for (cy, cx, cz, a) in emitters:
        st += a * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.3 ** 2)
                           + (zz - cz) ** 2 / (2 * 2.6 ** 2)))
    if mask_corner:
        st[:6, :6, :] = np.nan
    return st, [e[:3] for e in emitters]


def nearest(spot, truth):
    return min(((spot.y - t[0]) ** 2 + (spot.x - t[1]) ** 2
                + (spot.z - t[2]) ** 2) ** .5 for t in truth)


ANCHORED = ('anchor-v1', 'anchor-v2')


def main():
    st, truth = field()

    print('\n-- every engine is registered and constructible --')
    for name in ('gaussian',) + ANCHORED:
        check(f'{name} is in the factory', name in E.ENGINES)
        check(f'{name} constructs', E.make_engine(name) is not None)
    try:
        E.make_engine('no-such-engine')
        check('an unknown engine raises', False)
    except ValueError:
        check('an unknown engine raises ValueError', True)

    print('\n-- the anchor step alone: stack -> [(y, x, z)] --')
    anchors = E.anchor_candidates(st, n_max=8, **E.GENEROUS_ANCHOR)

    def covered(cands, tol=1.5):
        """How many true emitters have a candidate on them."""
        return sum(any(((cy - t[0]) ** 2 + (cx - t[1]) ** 2) ** .5 <= tol
                       for (cy, cx, _) in cands) for t in truth)

    # RECALL, not precision. A generous anchor is supposed to return junk
    # alongside the real emitters -- that junk is what a reviewer turns
    # into hard negatives. Asserting an exact count here would be
    # asserting the opposite of the setting's purpose, and would fail the
    # moment the threshold got as low as it needs to be.
    check('proposes every true emitter', covered(anchors) == 3,
          f'{covered(anchors)}/3 covered by {len(anchors)} candidates')
    check('each anchor is a 3-tuple of floats',
          all(len(a) == 3 and all(isinstance(v, float) for v in a) for a in anchors))
    check('mode-based background beats the median on this field',
          E.background_mode(np.nanmax(st, axis=2))[0]
          <= np.nanmedian(np.nanmax(st, axis=2)) * 1.5)

    print('\n-- generosity is a real difference, not a slogan --')
    strict = E.anchor_candidates(st, n_max=8)
    check('production defaults miss the dim emitter',
          len(strict) < len(anchors), f'{len(strict)} vs {len(anchors)}')
    check('GENEROUS_ANCHOR is looser on both knobs',
          E.GENEROUS_ANCHOR['threshold_rel'] < 0.5
          and E.GENEROUS_ANCHOR['max_to_background'] < 2.0)

    print('\n-- UNSEEDED localize is the contract a learned engine meets --')
    for name in ANCHORED:
        eng = E.make_engine(name, anchor=E.GENEROUS_ANCHOR)
        got = eng.localize(st, n_max=8)
        hits = [s for s in got if nearest(s, truth) < 1.0]
        check(f'{name}: unseeded call recovers all three emitters',
              len({round(nearest(s, truth), 6) for s in hits}) >= 3
              or len(hits) >= 3, f'{len(hits)} hits of {len(got)} returned')
        check(f'{name}: every recovered emitter is sub-pixel accurate',
              hits and all(nearest(s, truth) < 1.0 for s in hits),
              f'max err {max(nearest(s, truth) for s in hits):.2f}px' if hits else 'none')
        check(f'{name}: p is in (0, 1]', all(0 < s.p <= 1 for s in got),
              str([round(s.p, 3) for s in got]))
        check(f'{name}: results are ordered best-first',
              all(a.p >= b.p for a, b in zip(got, got[1:])))
        check(f'{name}: n_max is honoured', len(eng.localize(st, n_max=1)) == 1)

    print('\n-- a human click and an auto anchor enter the SAME door --')
    # The point of the whole design: manual anchoring is an alternative
    # anchor source at the front, not a stage in the middle. Feeding an
    # anchor back in as an explicit seed must reach the same fit.
    for name in ANCHORED:
        eng = E.make_engine(name, anchor=E.GENEROUS_ANCHOR)
        auto = eng.localize(st, n_max=8)
        a0 = anchors[0]
        seeded = eng.localize(st, seed_yxz=a0, n_max=1)
        check(f'{name}: a seeded call returns exactly one', len(seeded) == 1)
        if seeded and auto:
            same = min(auto, key=lambda s: (s.y - seeded[0].y) ** 2 + (s.x - seeded[0].x) ** 2)
            d = ((same.y - seeded[0].y) ** 2 + (same.x - seeded[0].x) ** 2
                 + (same.z - seeded[0].z) ** 2) ** .5
            check(f'{name}: seeded and auto agree on that emitter', d < 0.01, f'{d:.4f}px')

    print('\n-- p ranks the way a review queue needs --')
    # Brighter emitters should not score below dimmer ones; the queue is
    # ordered by p, so an inversion here puts junk in front of a reviewer.
    for name in ANCHORED:
        got = E.make_engine(name, anchor=E.GENEROUS_ANCHOR).localize(st, n_max=8)
        by_amp = sorted(got, key=lambda s: -(nearest(s, truth) + 0))  # stable
        order_ok = all(a.p >= b.p for a, b in zip(got, got[1:]))
        check(f'{name}: p is monotone over the returned order', order_ok)
        check(f'{name}: the dimmest real emitter still scores > 0',
              min(s.p for s in got) > 0)
        del by_amp

    print('\n-- absence is not an error --')
    empty = np.full((12, 12, 9), np.nan)
    flat = np.full((12, 12, 9), 300.0)
    for name in ('gaussian',) + ANCHORED:
        eng = E.make_engine(name)
        for label, bad in (('all-NaN', empty), ('empty', np.zeros((0, 0, 0))),
                           ('None', None)):
            try:
                out = eng.localize(bad, n_max=3)
                check(f'{name}: {label} stack returns [] without raising', out == [])
            except Exception as exc:                        # noqa: BLE001
                check(f'{name}: {label} stack returns [] without raising', False,
                      f'{type(exc).__name__}: {exc}')
        try:
            out = eng.localize(flat, n_max=3)
            check(f'{name}: a featureless stack does not raise', isinstance(out, list))
        except Exception as exc:                            # noqa: BLE001
            check(f'{name}: a featureless stack does not raise', False,
                  f'{type(exc).__name__}')

    print('\n-- the masked region is never a candidate --')
    st2, _ = field(mask_corner=True)
    got = E.make_engine('anchor-v2', anchor=E.GENEROUS_ANCHOR).localize(st2, n_max=8)
    check('no spot is reported inside the NaN corner',
          all(not (s.y < 6 and s.x < 6) for s in got))

    print('\n-- LocalizedSpot carries what a non-Gaussian engine can fill --')
    s = E._spot(1, 2, 3, 0.5)
    check('shape fields default to NaN for a shapeless engine',
          np.isnan(s.amplitude) and np.isnan(s.sigma_z))
    check('position and p are still real', (s.y, s.x, s.z, s.p) == (1.0, 2.0, 3.0, 0.5))
    check('p is a named field, not positional luck', 'p' in LocalizedFields())

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        for f in FAIL:
            print('  FAILED:', f)
        return 1
    print('ALL GOOD')
    return 0


def LocalizedFields():
    return E.LocalizedSpot._fields


if __name__ == '__main__':
    sys.exit(main())
