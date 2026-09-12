"""A label mask with more than 255 cells survives _filter_and_relabel.

Regression for a real loss: the relabelled mask was cast to uint8, so on
a dense brightfield field (JP_001, 2026-09-12) label 256 became
background and 257 onward wrapped onto cells 1, 2, ... -- 12 of 30 FOVs
reported exactly 255 cells, and the wrapped cells' pixels were merged
into other cells' masks. Run as a script, like the other tests here.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline.segmentation import segment  # noqa: E402


def main():
    n = 300
    side = 20                                   # 20x20 = 400 px per cell
    grid = 20                                   # 20 cells per row -> 15 rows
    mask = np.zeros((grid * side, grid * side), float)
    for k in range(1, n + 1):
        r, c = divmod(k - 1, grid)
        mask[r * side:(r + 1) * side, c * side:(c + 1) * side] = k * 3   # sparse ids
    out = segment._filter_and_relabel(mask, min_size=100, max_size=10000)
    labels = np.unique(out)[1:]
    ok = True
    checks = [
        ('every one of the 300 cells keeps its own label',
         len(labels) == n and labels.min() == 1 and labels.max() == n),
        ('labels are sequential from 1',
         np.array_equal(labels, np.arange(1, n + 1))),
        ('no two input cells share an output label',
         all((out[mask == k * 3] == out[mask == k * 3][0]).all()
             and len(np.unique(out[mask == k * 3])) == 1 for k in range(1, n + 1))
         and len({int(out[mask == k * 3][0]) for k in range(1, n + 1)}) == n),
        ('the dtype can hold more than 255 labels', np.iinfo(out.dtype).max >= 65535),
    ]
    for name, passed in checks:
        print(('PASS' if passed else 'FAIL'), name)
        ok &= bool(passed)
    print('ALL PASS' if ok else 'FAILURES')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
