"""Edge cells are found on the container's own masks, and the cell-level
Z bound defaults to what the runs use.

CellContainer.edge_cell_ids is the batch form of the displayer's Remove
Edge Cells: a cell with a mask pixel on the first or last row or column
of its frame. Run as a script, like the other tests here.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline.alignment import chain                      # noqa: E402
from codelab_pipeline.models.cell_container import CellContainer  # noqa: E402


def main():
    mask = np.zeros((100, 100), np.uint16)
    mask[0:10, 40:60] = 1          # touches the top row
    mask[40:60, 90:100] = 2        # touches the right column
    mask[40:60, 40:60] = 3         # interior
    mask[90:100, 0:10] = 4         # bottom-left corner: both edges
    c = CellContainer([1])
    c.load_new_cells(1, mask, 'Hyb_500', reference_modality='RNA')
    ids = c.edge_cell_ids(1)
    ok = True
    checks = [
        ('the three clipped cells are found, the interior one is not', ids == [1, 2, 4]),
        ('removing them leaves the interior cell',
         [x.id for x in (c.remove(1, ids), c.get_cells(1))[1]] == [3]),
        ('a FOV with no cells reports none', c.edge_cell_ids(2) == []),
        ('the cell-level Z bound defaults to 55 planes',
         chain.MAX_CELL_Z_SHIFT_PLANES == 55.0),
    ]
    for name, passed in checks:
        print(('PASS' if passed else 'FAIL'), name)
        ok &= bool(passed)
    print('ALL PASS' if ok else 'FAILURES')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
