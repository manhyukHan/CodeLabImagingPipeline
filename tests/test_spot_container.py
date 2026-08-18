"""
SpotContainer + DiffUndo unit tests. Every test builds its own state; no
ordering dependencies (the lesson test_spot_store_roundtrip learned).
Run: python tests/test_spot_container.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codelab_pipeline.models.spot import ASpot
from codelab_pipeline.models.spot_container import SpotContainer, DiffUndo

FOV = 1

def mk(uid, hybe='Hyb_101', mod='RNA', ch=635, cell=-1, x=1.0):
    s = ASpot()
    s.set_metadata(uid=uid, fov=FOV, modality=mod, hybe=hybe, channel=ch, cell=cell,
                   raw_coordinate=(x, x, 0.0), coordinate=(x, x, 0.0))
    return s

def filled():
    c = SpotContainer()
    c.add_many(FOV, [mk(1), mk(2, cell=7), mk(3, hybe='Hyb_105'),
                     mk(4, mod='DNA', hybe='Hyb_002', ch=555, cell=9)])
    return c

def test_uid_zero_rejected():
    c = SpotContainer()
    try:
        c.add(FOV, mk(0)); assert False, 'uid=0 must be rejected'
    except ValueError: pass

def test_duplicate_uid_rejected():
    c = filled()
    try:
        c.add(FOV, mk(1)); assert False, 'duplicate uid must be rejected'
    except ValueError: pass

def test_views_are_filters_not_locations():
    c = filled()
    assert len(c.all(FOV)) == 4
    assert {s.uid for s in c.slice(FOV, 'RNA', 'Hyb_101', 635)} == {1, 2}
    assert {s.uid for s in c.of_cell(FOV, 7)} == {2}
    assert {s.uid for s in c.unassigned(FOV)} == {1, 3}
    # assigned and unassigned coexist in one slice, differing only in .cell
    kinds = {s.cell != -1 for s in c.slice(FOV, 'RNA', 'Hyb_101', 635)}
    assert kinds == {True, False}

def test_replace_slice_is_scoped():
    c = filled()
    c.replace_slice(FOV, 'RNA', 'Hyb_101', 635, [mk(10)])
    assert {s.uid for s in c.all(FOV)} == {3, 4, 10}, 'other slices must survive'

def test_tier_transfer_deep_copies():
    trans, perm = filled(), SpotContainer()
    perm.copy_slice_from(trans, FOV, 'RNA', 'Hyb_101', 635)
    assert {s.uid for s in perm.all(FOV)} == {1, 2}
    perm.data[FOV][1].cell = 99
    assert trans.data[FOV][1].cell == -1, 'tiers must never share objects'

def test_diff_add_remove_change():
    c = filled()
    fp = c.fingerprint(FOV)
    c.remove(FOV, [1]); c.add(FOV, mk(5)); c.data[FOV][2].cell = -1
    d = SpotContainer.diff(fp, c.fingerprint(FOV))
    assert set(d['added']) == {5} and set(d['removed']) == {1} and set(d['changed']) == {2}

def test_undo_redo_roundtrip():
    c = filled(); u = DiffUndo(c)
    fp = c.fingerprint(FOV)
    c.remove(FOV, [1]); c.add(FOV, mk(5)); c.data[FOV][2].cell = -1
    assert u.push(FOV, fp)
    after = c.fingerprint(FOV)
    u.undo()
    assert c.fingerprint(FOV) == fp, 'undo must restore the exact before-state'
    u.redo()
    assert c.fingerprint(FOV) == after, 'redo must restore the exact after-state'

def test_two_streaks_and_noop_drop():
    c = filled(); u = DiffUndo(c)
    for k in (20, 21, 22):                      # three edits -> only last two undoable
        fp = c.fingerprint(FOV); c.add(FOV, mk(k)); u.push(FOV, fp)
    fp = c.fingerprint(FOV)
    assert not u.push(FOV, fp), 'a no-op edit must not consume a slot'
    assert u.undo() is not None and u.undo() is not None
    assert not u.can_undo(), 'depth is exactly two'
    assert {s.uid for s in c.all(FOV)} == {1, 2, 3, 4, 20}, 'first edit is beyond the streak'

def _run_all():
    fails = 0
    for name, fn in sorted(t for t in globals().items() if t[0].startswith('test_')):
        try:
            fn(); print(f'  PASS  {name}')
        except Exception as e:
            fails += 1; print(f'  FAIL  {name}: {e!r}')
    print(f'\n{8 - fails}/8 passed')
    return 1 if fails else 0

if __name__ == '__main__':
    sys.exit(_run_all())
