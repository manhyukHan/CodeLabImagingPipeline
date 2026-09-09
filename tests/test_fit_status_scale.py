"""
The fit-status tile's grey scale follows the MARKED spot.

WHY. On a real grid several tiles showed a yellow circle over what
looked like empty black while an off-centre blob was saturated white.
The scale was the cause, not the fit: ub=0.9999 on a 15x15 crop (225
pixels) is that crop's own maximum for all practical purposes, so one
bright neighbour set the white point and everything dimmer -- including
the spot the tile is about -- collapsed into the bottom of the range.

The tile knows which spot it is about: the one that gets the circle. So
the white point comes from a small box around that spot, and a brighter
neighbour clips to white instead of setting the exposure.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_fit_status_scale.py
"""
import os
import sys

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from canvas import spot_fit_status as F           # noqa: E402

CHECKS = [0, 0]


def check(name, ok, note=''):
    CHECKS[0] += 1
    CHECKS[1] += 1 if ok else 0
    print(('  ok   ' if ok else '  FAIL ') + name + (('  -- ' + note)
                                                     if note else ''))


def blob(cube, y, x, z, amp, sy=1.0, sz=2.4):
    h, w, d = cube.shape
    yy, xx, zz = np.ogrid[0:h, 0:w, 0:d]
    cube += amp * np.exp(-(((yy - y) ** 2 + (xx - x) ** 2) / (2 * sy ** 2)
                           + ((zz - z) ** 2) / (2 * sz ** 2)))
    return cube


def a_crop_with_a_brighter_neighbour():
    """The screenshot's case: dim marked spot, bright blob off-centre."""
    rng = np.random.RandomState(0)
    cube = rng.normal(100.0, 8.0, (15, 15, 41))
    blob(cube, 7, 7, 20, 120.0)          # the marked spot
    blob(cube, 2, 3, 22, 1400.0)         # the neighbour that ruined the tile
    return cube


def axes():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    return fig, fig.add_subplot(211), fig.add_subplot(212)


def drawn(ax):
    """The uint8 image an axes was given."""
    return np.asarray(ax.images[-1].get_array())


def test_marked_spot_is_visible():
    print('the marked spot survives a brighter neighbour')
    cube = a_crop_with_a_brighter_neighbour()

    # BEFORE: whole-crop quantiles, which is what the old tile did.
    from codelab_pipeline.io import preprocess
    old = preprocess.normalize_to_uint8(cube.max(axis=2), 0.3, 0.9999)
    old_spot = int(old[7, 7])

    fig, ax_yx, ax_xz = axes()
    F.draw_spot_fit_status(ax_yx, ax_xz, cube, centroid=(7.0, 7.0, 20.0))
    new = drawn(ax_yx)
    new_spot = int(new[7, 7])

    check('the old scale crushed the marked spot', old_spot < 60,
          'was %d/255' % old_spot)
    check('the new scale makes it read', new_spot > 200,
          'now %d/255' % new_spot)
    check('the neighbour clips to white instead of setting the scale',
          int(new[2, 3]) == 255)
    check('the noise floor is still dark',
          float(np.median(new)) < 40, 'median %d' % np.median(new))

    # The ZX panel gets the same treatment.
    zx = drawn(ax_xz)
    old_zx = preprocess.normalize_to_uint8(
        cube[:, :, 5:36].max(axis=0).T, 0.3, 0.9999)
    # THE NEIGHBOUR IS 4 px AWAY AND 12x BRIGHTER, so its tail is inside
    # the marked spot's own box and legitimately shares the white point.
    # The claim is not 'full scale', it is 'plainly visible, and far
    # better than before'.
    check('ZX is scaled to the marked spot too',
          int(zx[15, 7]) > 100 and int(zx[15, 7]) > 3 * int(old_zx[15, 7]),
          'was %d/255, now %d/255' % (old_zx[15, 7], zx[15, 7]))
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_no_marker_falls_back():
    print('no marker -> the old behaviour, exactly')
    from codelab_pipeline.io import preprocess
    cube = a_crop_with_a_brighter_neighbour()
    fig, ax_yx, ax_xz = axes()
    F.draw_spot_fit_status(ax_yx, ax_xz, cube, centroid=None)
    got = drawn(ax_yx)
    # NO MARKER STILL SCALES TO THE BRIGHTEST VOXEL, which with no spot
    # to be about is the only defensible choice -- and is what the
    # whole-crop scale did anyway.
    ref = preprocess.normalize_to_uint8(cube.max(axis=2), 0.3, 0.9999)
    # <=3/255: the two take the same white point by different routes
    # (this one the peak box's max, the old one the 0.9999 quantile of
    # 225 pixels), which differ only by quantile interpolation.
    check('an unmarked tile matches the whole-crop scale',
          int(np.abs(got.astype(int) - ref.astype(int)).max()) <= 3,
          'max difference %d/255'
          % np.abs(got.astype(int) - ref.astype(int)).max())
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_rejected_marker_is_used():
    print('a gate-rejected fit still sets the scale')
    cube = a_crop_with_a_brighter_neighbour()
    fig, ax_yx, ax_xz = axes()
    F.draw_spot_fit_status(ax_yx, ax_xz, cube, centroid=None,
                           rejected=(7.0, 7.0, 20.0))
    check('rejected fit scales like an accepted one',
          int(drawn(ax_yx)[7, 7]) > 200)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_degenerate_inputs():
    print('degenerate crops do not raise')
    import matplotlib.pyplot as plt
    flat = np.full((15, 15, 41), 300.0)
    fig, ax_yx, ax_xz = axes()
    F.draw_spot_fit_status(ax_yx, ax_xz, flat, centroid=(7.0, 7.0, 20.0))
    check('a flat crop draws', drawn(ax_yx).shape == (15, 15))
    plt.close(fig)

    holed = a_crop_with_a_brighter_neighbour()
    holed[0:3, :, :] = np.nan
    fig, ax_yx, ax_xz = axes()
    F.draw_spot_fit_status(ax_yx, ax_xz, holed, centroid=(7.0, 7.0, 20.0))
    check('NaN outside a cell mask draws', int(drawn(ax_yx)[7, 7]) > 200)
    plt.close(fig)

    fig, ax_yx, ax_xz = axes()
    # A CENTROID OUTSIDE THE CROP is possible (a fit that walked out).
    F.draw_spot_fit_status(ax_yx, ax_xz, a_crop_with_a_brighter_neighbour(),
                           centroid=(99.0, 99.0, 20.0))
    check('an off-crop centroid falls back rather than raising',
          drawn(ax_yx).shape == (15, 15))
    plt.close(fig)


def test_helpers():
    print('helpers')
    img = np.zeros((11, 11))
    img[5, 5] = 10.0
    img[0, 0] = 99.0
    check('_peak reads only its own box', F._peak(img, 5, 5, 2, 2) == 10.0)
    check('_peak clips to the image', F._peak(img, 0, 0, 2, 2) == 99.0)
    check('_peak off the image is None', F._peak(img, 50, 50, 2, 2) is None)
    check('_peak of all-NaN is None',
          F._peak(np.full((5, 5), np.nan), 2, 2, 1, 1) is None)
    # THE BOX IS ANISOTROPIC: tall in z, narrow in x.
    tall = np.zeros((21, 21))
    tall[5 + 4, 5] = 50.0        # 4 rows away  -> inside (half_row 5)
    tall[5, 5 + 4] = 70.0        # 4 cols away  -> outside (half_col 2)
    check('_peak reaches further along rows than columns',
          F._peak(tall, 5, 5, 5, 2) == 50.0)
    check('_floor ignores NaN',
          abs(F._floor(np.array([1.0, 2.0, 3.0, np.nan]), 0.5) - 2.0) < 1e-9)
    out = F._scaled(np.array([[0.0, 5.0, 20.0]]), 0.0, 10.0, 0.3, 0.9999)
    check('_scaled clips above the white point', list(out[0]) == [0, 127, 255])
    check('hi <= lo falls back to quantiles',
          F._scaled(np.array([[0.0, 1.0]]), 5.0, 5.0, 0.3, 0.9999) is not None)


def main():
    test_marked_spot_is_visible()
    test_no_marker_falls_back()
    test_rejected_marker_is_used()
    test_degenerate_inputs()
    test_helpers()
    print()
    print('%d/%d checks passed' % (CHECKS[1], CHECKS[0]))
    return 0 if CHECKS[1] == CHECKS[0] else 1


if __name__ == '__main__':
    sys.exit(main())
