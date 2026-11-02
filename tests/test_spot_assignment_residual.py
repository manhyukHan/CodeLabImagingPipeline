"""
An aligned cell must still appear in the label mask.

THE BUG THIS PINS. `label_mask_for_frame` projects each cell's mask into
the spot's own frame, and its session-free default is
`cell.get_area_in_readout`, which RAISES BY DESIGN on a bare cell-level
residual -- that is, on every cell that has run cell alignment. The
`except: continue` beside it then leaves that cell OUT of the mask, and
every spot inside it is assigned to nothing: `cell = -1`, silently, as a
normal result.

Two of the three callers did not pass the resolver-backed projection the
docstring asks for, and both are the post-alignment recast: the
disk-scoped `MainWindow._recast_persisted_spots` and the child-process
`assignment.recast_fov_spots_worker`. On C7 that emptied 32 of 42 FOVs
(1,413 of 6,051 cells left countable) with no error anywhere -- the FOVs
open in the session took the third, correct path and were fine.

So the checks below are about one thing: the mask must contain a
residual-form cell, and the default must still refuse to guess.

Run: python tests/test_spot_assignment_residual.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline.alignment.frames import FrameResolver
from codelab_pipeline.localization import assignment
from codelab_pipeline.models.cell import ACell
from codelab_pipeline.models.spot import ASpot

FRAME = (64, 64)
REF, HYB = 'Hyb_001', 'Hyb_002'
MOD = 'DNA'
#: the FOV leg for HYB, and the cell's own residual on top of it. Both
#: are translations, which is what this pipeline actually fits.
H_FOV = np.array([[1.0, 0.0, 4.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]])
H_RESIDUAL = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])


def _cell():
    """One 10x10 block of a cell, aligned: its HYB entry is a BARE
    residual, the form every cell carries after cell alignment."""
    ys, xs = np.mgrid[20:30, 20:30]
    c = ACell()
    c.set_metadata(id=7, fov=1, reference_hybe=REF, reference_modality=MOD,
                   nucleus_hybe=REF, nucleus_modality=MOD, celltype='',
                   nucleus=(ys.ravel(), xs.ravel()),
                   area=(ys.ravel(), xs.ravel()),
                   frame_shape=FRAME,
                   matrices={(REF, MOD): {'yx': np.eye(3), 'dz': 0.0,
                                          'yx_is_residual': True},
                             (HYB, MOD): {'yx': H_RESIDUAL, 'dz': -2.0,
                                          'yx_is_residual': True}},
                   matrix_anchors={MOD: np.eye(3)},
                   matrix_provenance={}, distmap=np.array([]),
                   linked=False, linked_at=None)
    return c


def _resolver():
    return FrameResolver({MOD: {REF: np.eye(3), HYB: H_FOV}}, shared=MOD,
                         anchors={MOD: np.eye(3)}, bridges={})


def _spots_inside(cell, resolver):
    """One spot per mask pixel of the cell AS SEEN IN HYB -- so every one
    of them is inside the cell by construction, and any that comes back
    unassigned is a fault in the mask, not in the geometry."""
    y, x = assignment.area_in_frame_via_resolver(resolver)(cell, HYB, MOD)
    spots = []
    for iy, ix in list(zip(np.asarray(y, int), np.asarray(x, int)))[:40]:
        s = ASpot()
        s.set_metadata(uid=len(spots) + 1, fov=1, modality=MOD, hybe=HYB,
                       channel=635, cell=7, celltype='',
                       raw_coordinate=(float(iy), float(ix), 50.0),
                       adj_coordinate=(float(iy), float(ix), 50.0),
                       size=1.0, brightness=1.0,
                       linked=False, linked_at=None, mixture_centroids=())
        spots.append(s)
    return spots


def test_the_cell_refuses_to_compose_its_own_residual():
    """The premise. If this ever stops raising, the silent-drop path below
    is gone and the rest of this file is about nothing."""
    c = _cell()
    try:
        c.get_area_in_readout(HYB, MOD)
    except ValueError as e:
        assert 'residual' in str(e), e
        return
    raise AssertionError('get_area_in_readout composed a bare residual')


def test_the_default_projection_drops_an_aligned_cell():
    """Pins the mechanism, so a future change cannot quietly restore it:
    with no area_in_frame, an aligned cell contributes NOTHING."""
    mask = assignment.label_mask_for_frame([_cell()], HYB, MOD, FRAME)
    assert mask.max() == 0, 'the default projection somehow kept the cell'


def test_the_resolver_projection_keeps_it():
    r = _resolver()
    mask = assignment.label_mask_for_frame(
        [_cell()], HYB, MOD, FRAME,
        area_in_frame=assignment.area_in_frame_via_resolver(r))
    assert (mask == 7).sum() == 100, (mask == 7).sum()


def test_the_mask_sits_where_both_legs_put_it():
    """Not just non-empty -- in the right place. HYB's frame is the
    reference frame moved by the FOV leg and then the cell's residual, so
    the mask's corner is the literal corner minus both translations (the
    projection is the INVERSE: reference -> hybe)."""
    r = _resolver()
    y, x = assignment.area_in_frame_via_resolver(r)(_cell(), HYB, MOD)
    # H maps hybe -> reference, so reference -> hybe subtracts both legs.
    # Row 0 of these matrices is y and row 1 is x (rasterised order, see
    # alignment/convention.py), so y moves by 4 + 1 and x by -3 + 2.
    assert int(np.min(y)) == 20 - (4 + 1), int(np.min(y))
    assert int(np.min(x)) == 20 - (-3 + 2), int(np.min(x))


def test_spots_inside_an_aligned_cell_keep_their_owner():
    """The consequence, end to end: the same spots, the same cells, and
    the only difference is the projection."""
    r, c = _resolver(), _cell()
    spots = _spots_inside(c, r)
    assert len(spots) == 40

    lost, _ = assignment.assign_spots(list(spots), [c], FRAME, {7: c})
    broken = assignment.count_unassigned(list(spots))

    good = [ASpot() for _ in spots]
    for s, d in zip(spots, good):
        d.set_metadata(**s.save())
    n_ok, n_bad = assignment.assign_spots(
        good, [c], FRAME, {7: c},
        area_in_frame=assignment.area_in_frame_via_resolver(r))

    assert (lost, broken) == (0, 40), (lost, broken)
    assert (n_ok, n_bad) == (40, 0), (n_ok, n_bad)
    assert all(int(s.cell) == 7 for s in good)


def _run_all():
    tests = {k: v for k, v in globals().items() if k.startswith('test_')}
    failures = []
    for name in sorted(tests):
        try:
            tests[name]()
            print(f'  PASS  {name}')
        except AssertionError as e:
            failures.append(name); print(f'  FAIL  {name}: {e}')
        except Exception as e:
            failures.append(name); print(f'  ERROR {name}: {e!r}')
    print(f'\n{len(tests) - len(failures)}/{len(tests)} passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(_run_all())
