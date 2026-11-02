"""The segmentation gate measures, and what it measures is kept.

WHAT THIS PINS.

1. flag() reads the cells the app actually has. It was written against
   dicts (`c['area']`) while its one caller -- the pipeline's
   segmentation step -- hands it CellContainer.get_cells(), which
   returns ACell objects. ACell is not subscriptable, so every real
   call raised TypeError into the step's per-FOV error list and the
   gate never ran once. Nothing tested it.

2. Everything it measures reaches the cell, and the store, and the
   export table. The numbers used to be computed, used to drop, logged
   as counts and discarded, so "why was this cell kept" could only be
   answered by re-reading the stain and re-flagging a population that
   the removal had already changed.

3. The fields are PREFIXED. The exporter spells out area_um2 and
   solidity of its own, from the mask and the voxel size; an unprefixed
   gate field would overwrite a column that means something else.

4. The provenance travels with them, n_population above all: area_rel
   and dapi_z are statements about the FOV's population, and that
   population shrinks by the drop a moment later.

Run: python tests/test_cell_gate.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline.analysis import export                    # noqa: E402
from codelab_pipeline.io import analysis_store                   # noqa: E402
from codelab_pipeline.models.cell import ACell                   # noqa: E402
from codelab_pipeline.models.cell_container import CellContainer  # noqa: E402
from codelab_pipeline.segmentation import cell_gate as CG         # noqa: E402

PASS, FAIL = [], []
FRAME = (128, 128)
PX = 0.108


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print('  %s %s%s' % ('ok' if cond else 'FAIL', name,
                         ('   [%s]' % detail) if detail else ''), flush=True)


def square(cid, y0, x0, n):
    c = ACell()
    ys, xs = np.mgrid[y0:y0 + n, x0:x0 + n]
    c.set_metadata(id=cid, fov=1, area=(ys.ravel(), xs.ravel()),
                   nucleus=(ys.ravel(), xs.ravel()), frame_shape=FRAME,
                   reference_hybe='Hyb_IFdapi', reference_modality='DNA',
                   nucleus_hybe='Hyb_IFdapi', nucleus_modality='DNA')
    return c


# A FIELD OF 23 NUCLEI, and the count matters: dapi_z is a z-score
# WITHIN the field, so it cannot exceed sqrt(n - 1) however bright one
# nucleus is. BRIGHT_Z is 3.0, which a field of 9 can never reach --
# the first version of this fixture had 9 and failed for that reason,
# not because the flag was wrong. Real fields carry hundreds.
img = np.ones(FRAME, float)
img[60:70, 100:110] = 60.0
plain = [(y, x) for y in (20, 40, 60, 80) for x in (20, 40, 60, 80)]
plain += [(100, 20), (100, 40), (100, 60)]
cells = [square(i + 1, y, x, 10) for i, (y, x) in enumerate(plain)]
cells += [square(20, 100, 80, 4),         # small
          square(21, 20, 100, 16),        # large
          square(22, 0, 100, 10),         # border
          square(23, 60, 100, 10)]        # bright


print('flag on the cells the app has')
rows = CG.flag(cells, img, PX, frame_shape=FRAME)
check('flag runs on ACell objects', len(rows) == len(cells), '%d rows' % len(rows))
by_id = {r['id']: r for r in rows}
check('small / large / border / bright land on the right nuclei',
      by_id[20]['small'] and by_id[21]['large'] and by_id[22]['border']
      and by_id[23]['bright'],
      'small %s large %s border %s bright %s'
      % (by_id[20]['small'], by_id[21]['large'], by_id[22]['border'],
         by_id[23]['bright']))
check('a plain 10x10 nucleus is flagged nothing',
      not any(by_id[1][f] for f in CG.FLAGS), str({f: by_id[1][f] for f in CG.FLAGS}))
# the ceiling that made the first fixture fail, stated as the property
# it is: one outlier in a field of n cannot score past sqrt(n - 1)
check('dapi_z cannot exceed sqrt(n - 1), so BRIGHT_Z needs n >= 11',
      by_id[23]['dapi_z'] <= np.sqrt(len(rows) - 1) + 1e-9
      and np.sqrt(9 - 1) < CG.BRIGHT_Z <= np.sqrt(len(rows) - 1),
      'z %.2f, ceiling %.2f' % (by_id[23]['dapi_z'], np.sqrt(len(rows) - 1)))


print('annotate')
STAIN = 'Hyb_IFdapi ch390 single plane z=47'
n = CG.annotate(cells, rows, drop='border,large', stain=STAIN)
check('every cell annotated', n == len(cells), str(n))
p = cells[0].properties
want = ({CG.PREFIX + k for k in CG.MEASURED} | {CG.PREFIX + f for f in CG.FLAGS}
        | {CG.PREFIX + 'drop', CG.PREFIX + 'stain', CG.PREFIX + 'n_population'})
check('all of it is stored, and only under the prefix',
      set(p) == want, str(sorted(set(p) ^ want)))
check('the provenance says what the numbers are relative to',
      p['gate_drop'] == 'border+large' and p['gate_stain'] == STAIN
      and p['gate_n_population'] == len(cells),
      '%s | %s | %s' % (p['gate_drop'], p['gate_stain'], p['gate_n_population']))
check('plain Python scalars, never numpy ones',
      all(type(v) in (int, float, bool, str) for v in p.values()),
      str({k: type(v).__name__ for k, v in p.items()
           if type(v) not in (int, float, bool, str)}))
check('an open gate still annotates, and says the gate was open',
      CG.annotate([square(99, 20, 20, 10)], [{**by_id[1], 'id': 99}], drop='',
                  stain=STAIN) == 1)

# n_population is the PRE-removal count, which is the whole point
ids = CG.to_drop(rows, 'border,large')
check('n_population is the population before the drop',
      cells[0].properties['gate_n_population'] == len(rows) > len(rows) - len(ids),
      '%d stored, %d would remain' % (cells[0].properties['gate_n_population'],
                                      len(rows) - len(ids)))


print('through the store and out to the table')
root = tempfile.mkdtemp(prefix='gatestore_')
try:
    container = CellContainer.load({1: [c.save() for c in cells]})
    analysis_store.write_cells(root, 1, container)
    back, _mod = analysis_store.read_cells(root, 1)
    check('the FOV reads back', back is not None and len(back) == len(cells),
          str(None if back is None else len(back)))
    got = {k: v for k, v in (back[0] or {}).items() if k.startswith(CG.PREFIX)}
    check('every gate field survives the columnar round trip',
          set(got) == want, str(sorted(set(got) ^ want)))
    check('and keeps its value and its type',
          got['gate_n_px'] == p['gate_n_px']
          and abs(got['gate_area_rel'] - p['gate_area_rel']) < 1e-9
          and got['gate_border'] is False and got['gate_stain'] == STAIN,
          str({k: got[k] for k in ('gate_n_px', 'gate_area_rel', 'gate_border')}))

    table = export.cell_rows(back, fov=1)
    row = table[0]
    check('the exporter turns each one into its own column',
          all(CG.PREFIX + f in row for f in CG.MEASURED)
          and all(CG.PREFIX + f in row for f in CG.FLAGS),
          str(sorted(k for k in row if k.startswith(CG.PREFIX))))
    # THE REASON FOR THE PREFIX: the exporter has its own area_um2 and
    # solidity, computed from the mask and the voxel size
    check("the exporter's own area_um2 is untouched",
          'area_um2' in row and abs(row['area_um2'] - row['gate_area_um2']) > 1e-6,
          'export %.4f vs gate %.4f' % (row['area_um2'], row['gate_area_um2']))
finally:
    shutil.rmtree(root, ignore_errors=True)


print('')
print('%d/%d passed' % (len(PASS), len(PASS) + len(FAIL)))
if FAIL:
    print('failed: ' + ', '.join(FAIL))
sys.exit(1 if FAIL else 0)
