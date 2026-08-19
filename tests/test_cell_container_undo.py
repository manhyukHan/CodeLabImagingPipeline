"""
CellContainer dict-by-id shape + bytes-fingerprint DiffUndo.
Every test builds its own state. Run: python tests/test_cell_container_undo.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codelab_pipeline.models.cell import ACell
from codelab_pipeline.models.cell_container import CellContainer
from codelab_pipeline.models.spot_container import DiffUndo

FOV = 1

def mk(cid, npx=6):
    c = ACell()
    x = np.arange(npx) + cid * 10
    y = np.arange(npx) + cid * 10
    c.set_metadata(id=cid, fov=FOV, reference_hybe='Hyb_400', reference_modality='DNA',
                   area=(y, x), frame_shape=(64, 64))
    return c

def filled():
    c = CellContainer([FOV])
    c.data[FOV] = {i: mk(i) for i in (1, 2, 3)}
    return c

def test_dict_shape_and_queries():
    c = filled()
    assert sorted(c.data[FOV].keys()) == [1, 2, 3]
    assert c.by_id(FOV, 2).id == 2
    assert [x.id for x in c.get_cells(FOV)] == [1, 2, 3]

def test_remove_deletes_id_for_good():
    c = filled()
    gone = c.remove(FOV, [2])
    assert [g.id for g in gone] == [2] and sorted(c.data[FOV]) == [1, 3]

def test_fingerprint_is_bytes_and_stable():
    """The trap this design exists for: save() dicts hold ndarrays, and
    dict equality over ndarrays is numpy's ambiguous-truth comparison --
    it calls identical cells unequal. Canonical bytes must not."""
    c = filled()
    fp1, fp2 = c.fingerprint(FOV), c.fingerprint(FOV)
    assert all(isinstance(v, bytes) for v in fp1.values())
    assert fp1 == fp2, 'identical state must fingerprint identically'
    # the hazard bytes exist to avoid: dict equality over ndarray values
    # either lies (returns False for identical cells) or raises outright.
    d = c.save()[FOV][0]
    try:
        equal = (d == c.save()[FOV][0])
        assert equal is False or equal is True  # older numpy: wrong answer
    except ValueError:
        pass                                    # newer numpy: raises

def test_restore_refingerprints_identically():
    """apply_inverse must CONVERGE: restore -> fingerprint == original,
    else undo/redo drifts a little on every cycle."""
    c = filled()
    fp = c.fingerprint(FOV)
    c.remove(FOV, [1]); c.by_id(FOV, 2).celltype = 'X'
    d = type(c).diff if hasattr(type(c), 'diff') else None
    from codelab_pipeline.models.spot_container import SpotContainer
    diff = SpotContainer.diff(fp, c.fingerprint(FOV))
    c.apply_inverse(FOV, diff)
    assert c.fingerprint(FOV) == fp, 'restore did not converge to the original bytes'

def test_diffundo_two_streaks_on_cells():
    c = filled(); u = DiffUndo(c)
    fp = c.fingerprint(FOV); c.remove(FOV, [1]); assert u.push(FOV, fp)
    fp = c.fingerprint(FOV); c.by_id(FOV, 3).celltype = 'T'; assert u.push(FOV, fp)
    assert not u.push(FOV, c.fingerprint(FOV)), 'no-op must not consume a slot'
    u.undo(); assert c.by_id(FOV, 3).celltype == ''
    u.undo(); assert c.by_id(FOV, 1) is not None, 'removed cell restored by undo'
    assert not u.can_undo()
    u.redo(); assert c.by_id(FOV, 1) is None, 'redo re-removes'


def test_area_arrays_are_write_locked():
    c = mk(1)
    try:
        c.area[0][0] = 999
        assert False, 'in-place coordinate write must raise'
    except ValueError:
        pass

def test_sync_from_is_isolated_and_minimal():
    a, b = filled(), CellContainer([FOV])
    n = b.sync_from(a, FOV)
    assert n == 3 and sorted(b.data[FOV]) == [1, 2, 3]
    # unchanged cells are NOT re-materialized on a second sync
    obj_before = b.data[FOV][2]
    assert b.sync_from(a, FOV) == 0 and b.data[FOV][2] is obj_before
    # scalar mutation on one tier never leaks to the other
    b.data[FOV][2].celltype = 'X'
    assert a.data[FOV][2].celltype == ''
    # matrix-entry replacement on one tier never leaks either
    b.data[FOV][3].matrices[('H', 'RNA')] = {'yx': np.eye(3), 'dz': 1.0}
    assert ('H', 'RNA') not in a.data[FOV][3].matrices
    # removal propagates on the next sync
    a.remove(FOV, [1])
    b.sync_from(a, FOV)
    assert sorted(b.data[FOV]) == [2, 3]


def test_load_new_cells_area_is_y_major():
    """
    The mask->cells producer packs area as (y, x). Caught real: after the
    #69 flip, load_new_cells still packed (x, y) -- invisible on square
    frames and to every clicked flow that only LOADED existing cells, so
    this uses a non-square frame where a transposed pack cannot
    re-rasterize at all. Covers both branches (fresh + carry-over).
    """
    mask = np.zeros((40, 90), dtype=np.int32)         # height != width
    mask[5:9, 60:80] = 7                              # asymmetric, off-center
    c = CellContainer([FOV])
    c.load_new_cells(FOV, mask, 'Hyb_400', reference_modality='DNA')
    cell = c.by_id(FOV, 7)
    assert cell is not None, 'cell not built from mask'
    y, x = cell.area
    rebuilt = np.zeros_like(mask)
    rebuilt[y.astype(int), x.astype(int)] = 7         # transposed -> IndexError
    assert np.array_equal(rebuilt, mask), 'area does not re-rasterize the mask'
    # carry-over branch: re-segment same frame, cell keeps id, area follows new mask
    mask2 = np.zeros_like(mask); mask2[6:12, 55:70] = 7
    c.load_new_cells(FOV, mask2, 'Hyb_400', reference_modality='DNA', preserve_existing=True)
    y2, x2 = c.by_id(FOV, 7).area
    rebuilt2 = np.zeros_like(mask2)
    rebuilt2[y2.astype(int), x2.astype(int)] = 7
    assert np.array_equal(rebuilt2, mask2), 'carry-over branch not (y, x)'


def _run_all():
    fails = 0
    for name, fn in sorted(t for t in globals().items() if t[0].startswith('test_')):
        try:
            fn(); print(f'  PASS  {name}')
        except Exception as e:
            fails += 1; print(f'  FAIL  {name}: {e!r}')
    n = sum(1 for k in globals() if k.startswith('test_'))
    print(f'\n{n - fails}/{n} passed')
    return 1 if fails else 0

if __name__ == '__main__':
    sys.exit(_run_all())
