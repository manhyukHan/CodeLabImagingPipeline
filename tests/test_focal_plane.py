"""The focal plane must refuse an edge argmax.

WHAT THIS PINS. The variance of the Laplacian has an interior maximum
only when the stack brackets the focus. When it does not -- a stack
that starts inside the sample, a channel with no structure, brightfield
whose contrast keeps climbing -- the profile is monotonic or flat, and
argmax returns the FIRST OR LAST PLANE: the two planes where a nucleus
is least in focus. The GUI then reported that as a detected focal
plane, and segmentation ran on it.

The middle is not a better estimate of focus. It is the honest answer
when the measurement failed, and the caller is told which it got.

Run: python tests/test_focal_plane.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline.segmentation import segment            # noqa: E402

ZS = np.arange(129)


def _peaked(centre, width=12.0, base=0.2):
    return base + np.exp(-0.5 * ((ZS - centre) / width) ** 2)


def test_a_real_peak_is_kept():
    z, how = segment.focal_plane(ZS, _peaked(76))
    assert (z, how) == (76, 'peak'), (z, how)


def test_a_rising_profile_goes_to_the_middle_not_the_last_plane():
    """The measured case: fov026 of the SG store, argmax at z=0."""
    rising = np.linspace(1.0, 4.0, ZS.size)
    z, how = segment.focal_plane(ZS, rising)
    assert how == 'no interior peak', how
    assert z == 64, z
    assert int(np.argmax(rising)) == ZS.size - 1, 'the fixture is not monotonic'


def test_a_falling_profile_too():
    z, how = segment.focal_plane(ZS, np.linspace(4.0, 1.0, ZS.size))
    assert (z, how) == (64, 'no interior peak'), (z, how)


def test_a_flat_profile_is_refused_even_with_an_interior_argmax():
    """Noise has an argmax. A peak that barely clears its own baseline
    is not a focal plane, wherever it happens to sit."""
    rng = np.random.default_rng(7)
    flat = 100.0 + rng.normal(0.0, 0.5, ZS.size)
    z, how = segment.focal_plane(ZS, flat)
    assert how == 'flat profile', (how, z)
    assert z == 64, z


def test_a_peak_near_the_edge_counts_as_an_edge():
    """fov060 of the SG store peaks at z=1, not z=0. A test for the
    literal first plane would have let it through, and fov060 is one of
    the four fields the store already showed as broken."""
    vals = np.full(ZS.size, 1.0)
    vals[1] = 3.0
    z, how = segment.focal_plane(ZS, vals)
    assert (z, how) == (64, 'no interior peak'), (z, how)


def test_the_measured_store_cases_split_the_way_they_did():
    """The whole point of the thresholds, as (argmax index, prominence)
    pairs taken from the 68-FOV scan of Hyb_IFdapi ch390.

    REJECTED are exactly the four fields the store had already shown as
    broken; the three borderline ones are kept, because their peaks are
    interior and plausible and an earlier, stricter prominence test
    threw them away.
    """
    reject = {'fov009': (0, 0.087), 'fov026': (0, 0.077),
              'fov043': (0, 0.131), 'fov060': (1, 0.075)}
    keep = {'fov001': (47, 0.110), 'fov006': (54, 0.142),
            'fov033': (41, 0.145), 'fov016': (45, 0.155),
            'fov030': (64, 0.227)}
    for name, (idx, prom) in {**reject, **keep}.items():
        vals = np.full(ZS.size, 1.0)
        # a profile whose 10th percentile is 1.0 and whose peak clears
        # it by `prom` of the peak, with the peak at `idx`
        vals[idx] = 1.0 / (1.0 - prom)
        z, how = segment.focal_plane(ZS, vals)
        if name in reject:
            assert how != 'peak', (name, idx, prom, how)
            assert z == 64, (name, z)
        else:
            assert (z, how) == (idx, 'peak'), (name, z, how)


def _run_all():
    tests = {k: v for k, v in globals().items() if k.startswith('test_')}
    failures = []
    for name in sorted(tests):
        try:
            tests[name]()
            print('  PASS  ' + name)
        except AssertionError as e:
            failures.append(name); print('  FAIL  %s: %s' % (name, e))
        except Exception as e:                               # noqa: BLE001
            failures.append(name); print('  ERROR %s: %r' % (name, e))
    print('')
    print('%d/%d passed' % (len(tests) - len(failures), len(tests)))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(_run_all())
