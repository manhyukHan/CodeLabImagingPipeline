"""
The Fiducial Overlay grid's ZX row must compare the same PHYSICAL depth
in every tile.

Each hybe's tracing crop is cut in that hybe's OWN raw z frame -- the
crop box spans the full stack depth and is never z-shifted -- while the
fit stored in allele.fiducial_trace_adj is in the SHARED frame (native z plus
the FOV, cross-modal and CELL-level residual dz). The overlay builder
took its z-window from the shared-frame reference fit and then sliced
every raw cubic at those same indices, so the two sides of a tile showed
two different depths.

Confirmed real: a cell whose Hyb_039 residual was dz=21 rendered a ZX
tile whose moving fiducial fell clean outside the +/-15 window (pinned at
row 0, 15 rows from the reference) while the tile's own printed
d=(+0.42,+0.28,+1.36) correctly reported a drift of about one plane. The
numbers were right; the picture was not.

Two properties, and the second is why the first cannot be fixed by
re-centering each tile on its own fit:

1. The ZX peaks coincide when the shared-frame drift is zero, WHATEVER
   the cell-level dz is.
2. A real shared-frame drift still separates them by exactly that much --
   the ZX row exists to show drift, so the fix must not flatten it.

Run: python tests/test_fiducial_overlay_z_frame.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np                                      # noqa: E402
from PyQt5 import QtWidgets                             # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

from windows.main_window import MainWindow              # noqa: E402

PASS, FAIL = [], []
DEPTH, H, W = 120, 21, 21
REF_NATIVE = 70.0
Z_PAD = 15          # the builder's own display window


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(('  ok ' if cond else 'FAIL ') + name + (f' -- {detail}' if detail and not cond else ''))


def cube(z_native):
    """A clean fiducial at native plane z_native, on realistic noise."""
    c = np.random.default_rng(0).normal(100, 3, (H, W, DEPTH))
    zz, yy, xx = np.meshgrid(np.arange(DEPTH), np.arange(H), np.arange(W), indexing='ij')
    g = 4000 * np.exp(-(((yy - 10) ** 2 + (xx - 10) ** 2) / (2 * 1.3 ** 2)
                        + ((zz - z_native) ** 2) / (2 * 2.5 ** 2)))
    return c + np.transpose(g, (1, 2, 0))


def peak_row(rgb_img, channel):
    """Row of the brightest depth in one colour channel of a ZX tile
    (depth is the ROW axis there -- the view is transposed)."""
    return int(np.argmax(np.nanmax(rgb_img[..., channel], axis=1)))


def build(cell_dz, drift):
    """
    One reference + one moving hybe, where the moving hybe carries
    `cell_dz` planes of already-corrected cell-level residual and `drift`
    planes of genuine leftover shared-frame drift.
    """
    mov_native = REF_NATIVE - cell_dz + drift
    allele = types.SimpleNamespace(fiducial_trace_adj={
        # shared frame: native + that hybe's own offset
        'R': (10.0, 10.0, REF_NATIVE, 500.0),
        'M': (10.0, 10.0, mov_native + cell_dz, 500.0)})
    debug = {
        # debug centroids are the SAME fits in each hybe's own raw frame
        'R': {'fiducial_cubic': cube(REF_NATIVE), 'fiducial_centroid': (10.0, 10.0, REF_NATIVE)},
        'M': {'fiducial_cubic': cube(mov_native), 'fiducial_centroid': (10.0, 10.0, mov_native)}}
    entries, total = MainWindow._build_fiducial_overlay_entries(allele, 'R', debug)
    return entries, total


def main():
    # -- 1. zero drift must render as coincident, at any cell-level dz --
    for cell_dz in (0.0, 21.0, -18.0, 40.0):
        entries, _ = build(cell_dz, 0.0)
        _yb, _ya, zx_b, _za, _title = entries[0]
        sep = peak_row(zx_b, 1) - peak_row(zx_b, 0)
        check(f'cell dz={cell_dz:+.0f}, no drift -> ZX peaks coincide',
              abs(sep) <= 1, f'{sep} rows apart')

    # -- 2. real drift must survive, undiminished, at any cell-level dz --
    for cell_dz in (0.0, 21.0, -18.0):
        for drift in (4.0, -6.0):
            entries, _ = build(cell_dz, drift)
            _yb, _ya, zx_b, _za, title = entries[0]
            sep = peak_row(zx_b, 1) - peak_row(zx_b, 0)
            check(f'cell dz={cell_dz:+.0f}, drift={drift:+.0f} -> ZX shows the drift',
                  abs(sep - drift) <= 1, f'{sep} rows apart, expected {drift:+.0f}')

    # -- 3. the reported number and the picture must agree --
    entries, _ = build(21.0, 4.0)
    _yb, _ya, zx_b, _za, title = entries[0]
    reported = float(title.rsplit(',', 1)[1].rstrip(')'))
    sep = peak_row(zx_b, 1) - peak_row(zx_b, 0)
    check('the printed dz matches what the ZX tile shows',
          abs(reported - sep) <= 1, f'printed {reported:+.2f}, drawn {sep}')

    # -- 4. tiles stay renderable: no NaN reaches the RGB compositing --
    entries, total = build(200.0, 0.0)      # window entirely off this crop
    ok = all(np.isfinite(np.asarray(img)).all()
             for tile in entries for img in tile[:4])
    check('a window off the end of the crop still renders (no NaN leaks)', ok)
    check('the total overlay is built too', total is not None)

    # -- 5. every tile shares one row grid, so rows are comparable --
    entries, _ = build(21.0, 0.0)
    _yb, _ya, zx_b, zx_a, _t = entries[0]
    check('ZX tiles are exactly the shared window tall',
          zx_b.shape[0] == 2 * Z_PAD + 1 and zx_a.shape[0] == 2 * Z_PAD + 1,
          f'{zx_b.shape[0]} rows, expected {2 * Z_PAD + 1}')

    print()
    print(f'{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        raise SystemExit('FAILURES: ' + ', '.join(FAIL))
    print('ALL GOOD')


if __name__ == '__main__':
    main()
