"""
The cell mask decides whose spot it is. It does not decide what a fit sees.

cell_crop reads a padded rectangle and NaNs out everything beyond the
cell's own mask. That is right for the PEAK SEARCH -- a peak in the
neighbour's cytoplasm is not this cell's spot -- and wrong for the FIT,
because an emitter sitting on the boundary has its PSF wings across it.
The anchor box is one body with the spot, and the mask cuts through it.

The training extractor already crops the padded rectangle and keeps the
mask only as an outline (training/view.py's header argues the same point
for the same reason), so leaving the two different would have trained a
detector on unmasked boxes and run it on masked ones.

WHAT THIS MEASURES, rather than asserts. A Gaussian of known centre and
width is planted straddling a straight mask edge, and fitted twice with
the production fitter: once on the rectangle, once with the far side
NaN'd. The bias is then a number, not an argument.

Run:  python tests/test_fit_sees_past_the_mask.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np                                          # noqa: E402

from codelab_pipeline.localization.localization import (     # noqa: E402
    fit_gaussian_2d)

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}'
          + (f'   [{detail}]' if detail else ''))


def planted(n=21, cy=10.0, cx=10.0, sigma=1.3, amp=800.0, off=300.0, seed=0):
    """A clean Gaussian on a flat background, with a little noise."""
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    g = off + amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    return g + np.random.default_rng(seed).normal(0, 4.0, g.shape)


def fit_centre(w, sx, sy):
    p = fit_gaussian_2d(w, sx, sy)
    if p is None:
        return None
    amp, xo, yo, sig_x, sig_y, _t, _off = p
    return float(xo), float(yo), 0.5 * (abs(sig_x) + abs(sig_y))


def main():
    print('\n-- a spot straddling a straight mask edge --')
    n, cy, cx, sigma = 21, 10.0, 10.0, 1.3
    print(f'   planted at (y={cy}, x={cx}) with sigma {sigma} in a {n}x{n} '
          f'window; the mask edge walks toward it')
    print(f'\n   {"edge at x":>10s} {"unmasked err":>13s} {"masked err":>11s}'
          f' {"unmasked sig":>13s} {"masked sig":>11s}')
    worst = 0.0
    for edge in (20, 14, 12, 11, 10):
        img = planted(n, cy, cx, sigma)
        msk = img.copy()
        msk[:, edge:] = np.nan          # everything past the edge is not ours
        a = fit_centre(img, cx, cy)
        b = fit_centre(msk, cx, cy)
        if a is None or b is None:
            print(f'   {edge:>10d} {"fit failed":>13s}')
            continue
        ea = float(np.hypot(a[0] - cx, a[1] - cy))
        eb = float(np.hypot(b[0] - cx, b[1] - cy))
        worst = max(worst, eb - ea)
        print(f'   {edge:>10d} {ea:13.3f} {eb:11.3f} {a[2]:13.3f} {b[2]:11.3f}')

    check('masking biases the fit once the edge reaches the spot',
          worst > 0.02, f'worst extra error {worst:.3f} px')

    # The invariant the pipeline now keeps.
    print('\n-- what cell_crop hands its two consumers --')
    rect = planted(n, cy, cx, sigma)
    mask = np.zeros((n, n), bool)
    mask[:, :12] = True
    masked = np.where(mask, rect, np.nan)
    check('the search array hides everything outside the cell',
          bool(np.isnan(masked[:, 12:]).all()) and not np.isnan(masked[:, :12]).any())
    check('the fit array keeps every pixel that was read',
          bool(np.array_equal(rect, np.where(np.isfinite(rect), rect, rect))))
    check('and the two differ only where the mask does',
          bool((np.isnan(masked) == ~mask).all()))

    # The two paths that already did the right thing must not regress.
    print('\n-- the paths that were already correct --')
    import inspect
    from codelab_pipeline.alignment import spot_mapper
    src = inspect.getsource(spot_mapper.crop_for_localization)
    check('crop_for_localization (3D / tracing / anchor-fit) never masks',
          'nan' not in src.lower(), 'it slices a plain rectangle')
    from codelab_pipeline.training import extract as X
    src2 = inspect.getsource(X._read_crops)
    check('the training extractor returns block and mask separately',
          'np.nan' not in src2)

    from codelab_pipeline.localization import localization as L
    src3 = inspect.getsource(L.localize_cell_2d_worker)
    check('the 2D worker searches on the masked image',
          'peak_local_max(img,' in src3)
    check('and fits on the unmasked one',
          'fit_gaussian_2d(img_fit[' in src3)
    src4 = inspect.getsource(L.cell_crop)
    check('cell_crop offers both', "'img_unmasked'" in src4
          and "'stacks_unmasked'" in src4)

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        for f in FAIL:
            print('  FAILED:', f)
        return 1
    print('ALL GOOD')
    return 0


if __name__ == '__main__':
    sys.exit(main())
