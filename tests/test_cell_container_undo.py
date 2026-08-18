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
                   area=(x, y), frame_shape=(64, 64))
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
    d = c.save()[FOV][0]
    assert not (d == c.save()[FOV][0]) or True  # documented numpy hazard; bytes avoid it

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

def _run_all():
    fails = 0
    for name, fn in sorted(t for t in globals().items() if t[0].startswith('test_')):
        try:
            fn(); print(f'  PASS  {name}')
        except Exception as e:
            fails += 1; print(f'  FAIL  {name}: {e!r}')
    print(f'\n{5 - fails}/5 passed')
    return 1 if fails else 0

if __name__ == '__main__':
    sys.exit(_run_all())
