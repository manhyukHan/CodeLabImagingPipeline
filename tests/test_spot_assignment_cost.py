"""
Cost and correctness of assignment.assign_spots, on real cells.

The open design question this answers: can spot assignment simply be re-run
on every save, or does it need to be incremental? Re-running is far simpler
-- assignment stops being state that can go stale and becomes a pure function
of (spots, cells, matrices) -- but only if it is cheap at real scale.

Loads FOV01's real cells, rasterises them once, and assigns synthetic spot
sets of increasing size. Synthetic because the question is throughput, not
detection: positions are drawn across the frame so a realistic mix lands
inside and outside cells.

Run: python tests/test_spot_assignment_cost.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline.io import vlinks_store as V
from codelab_pipeline.localization import assignment
from codelab_pipeline.models.cell_container import CellContainer
from codelab_pipeline.models.spot import ASpot

STORAGE = os.environ.get('CODELAB_COST_STORE', 'data/chr19_downstream_new/RNA_queue')
FOV = 1
HYBE = 'Hyb_400'        # whichever frame the bulk of the real cells live in
MODALITY = 'DNA'
SIZES = [1_000, 10_000, 50_000, 100_000]


def load_cells():
    dicts, modality = V.read_cells(STORAGE, FOV)
    container = CellContainer.load({FOV: dicts}, modality=modality)
    return container.get_cells(FOV)


def make_spots(n, shape, rng):
    ys = rng.integers(0, shape[0], size=n)
    xs = rng.integers(0, shape[1], size=n)
    out = []
    for i in range(n):
        s = ASpot()
        s.modality = MODALITY
        s.set_metadata(uid=i + 1, fov=FOV, hybe=HYBE, channel=635,
                       raw_coordinate=(float(ys[i]), float(xs[i]), 0.0),
                       coordinate=(float(ys[i]), float(xs[i]), 0.0))
        out.append(s)
    return out


_cells = None
_mask = None


def setup():
    global _cells, _mask
    if _cells is None:
        _cells = load_cells()
        assert _cells, 'no cells in the real store to assign against'
        ref = _cells[0]
        _mask = assignment.label_mask_for_frame(
            _cells, ref.reference_hybe, ref.reference_modality, ref.frame_shape)
    return _cells, _mask


def test_assignment_is_correct_before_it_is_fast():
    """A spot placed on a known cell pixel must land on that cell."""
    cells, mask = setup()
    by_id = {c.id: c for c in cells}
    target = cells[0]
    y, x = target.area           # (y, x) -- rasterized order
    px, py = int(np.asarray(x).ravel()[0]), int(np.asarray(y).ravel()[0])

    # The pixel comes from target.area, which is native to the cell's OWN
    # reference frame -- so the spot must claim that frame too. The old
    # hard-coded 'RNA' only ever worked incidentally against legacy
    # composed matrices; on residual-form cells the mismatched frame has
    # no session-free projection at all.
    inside = ASpot(); inside.modality = target.reference_modality
    inside.set_metadata(uid=1, fov=FOV, hybe=target.reference_hybe, channel=635,
                        raw_coordinate=(float(py), float(px), 0.0))
    outside = ASpot(); outside.modality = target.reference_modality
    outside.set_metadata(uid=2, fov=FOV, hybe=target.reference_hybe, channel=635,
                         raw_coordinate=(-50.0, -50.0, 0.0))

    n_a, n_u = assignment.assign_spots([inside, outside], cells, mask.shape, by_id)
    assert inside.cell == target.id, f'expected cell {target.id}, got {inside.cell}'
    assert outside.cell == -1, 'a spot outside the frame must stay unassigned'
    assert (n_a, n_u) == (1, 1)


def test_unresolvable_frame_leaves_spots_unassigned():
    """A None matrix must not be treated as identity."""
    cells, mask = setup()
    s = ASpot(); s.modality = MODALITY
    s.set_metadata(uid=1, fov=FOV, hybe='Hyb_999', channel=635,
                   raw_coordinate=(10.0, 10.0, 0.0))
    n_a, n_u = assignment.assign_spots([s], cells, mask.shape,
                                       {c.id: c for c in cells})
    assert s.cell == -1 and (n_a, n_u) == (0, 1)


def test_reassignment_drops_a_stale_owner():
    """
    Assignment is recomputed, not patched. A spot carrying an owner it no
    longer sits inside must lose it -- keeping it is how assignment and
    geometry drift apart.
    """
    cells, mask = setup()
    s = ASpot(); s.modality = MODALITY
    s.set_metadata(uid=1, fov=FOV, hybe=HYBE, channel=635, cell=cells[0].id,
                   celltype='something', raw_coordinate=(-99.0, -99.0, 0.0))
    assignment.assign_spots([s], cells, mask.shape, {c.id: c for c in cells})
    assert s.cell == -1 and s.celltype == ''


def test_cost_scales_linearly_and_is_affordable_per_save():
    cells, mask = setup()
    by_id = {c.id: c for c in cells}
    rng = np.random.default_rng(0)
    timings = {}
    for n in SIZES:
        spots = make_spots(n, mask.shape, rng)
        t0 = time.perf_counter()
        assignment.assign_spots(spots, cells, mask.shape, by_id)
        timings[n] = time.perf_counter() - t0

    print(f'\n  {len(cells)} real cells, mask {mask.shape}')
    for n, dt in timings.items():
        print(f'    {n:7,d} spots  {dt*1000:8.1f} ms   {dt/n*1e6:6.2f} us/spot')

    # Cost model is "one mask build per frame, then O(1) per spot", so
    # per-spot cost FALLS with N as the fixed build amortises. What must
    # stay linear is the MARGINAL cost, so compare the two largest sizes:
    # doubling the spots may not double the time (the fixed part is already
    # paid) but must never more than double it.
    big, small = sorted(SIZES)[-1], sorted(SIZES)[-2]
    ratio = timings[big] / timings[small]
    assert ratio <= (big / small) * 1.3, \
        f'{small:,}->{big:,} spots cost {ratio:.2f}x, worse than the {big/small:.0f}x linear bound'
    per_spot = [dt / n for n, dt in timings.items()]
    assert per_spot[-1] <= per_spot[0], \
        f'per-spot cost should amortise downward, got {per_spot}'
    assert timings[big] < 10.0, \
        f'{big:,} spots took {timings[big]:.1f}s -- too slow to re-run per save'


def test_matrix_resolution_is_memoized_per_cell():
    """
    The transform is a cheap matmul; the caller's matrix RESOLUTION is not
    (MainWindow builds a FrameResolver per call -- measured 540 us/spot,
    27 s for a 50k-spot save, before the memo). assign_spots and
    recast_spots_to_shared must therefore resolve at most once per
    (hybe, modality, cell), never once per spot.
    """
    cells, mask = setup()
    by_id = {c.id: c for c in cells}
    rng = np.random.default_rng(1)
    ids = list(by_id)
    spots = []
    for i in range(5000):
        s = ASpot(); s.modality = MODALITY
        s.set_metadata(uid=i + 1, fov=FOV, hybe=HYBE, channel=635, cell=int(rng.choice(ids)),
                       raw_coordinate=(float(rng.uniform(0, 1024)), float(rng.uniform(0, 1024)), 0.0),
                       coordinate=(0.0, 0.0, 0.0))
        spots.append(s)
    calls = {'n': 0}

    def to_shared(hybe, modality, owner):
        calls['n'] += 1
        return np.eye(3)
    n = assignment.recast_spots_to_shared(spots, to_shared, by_id)
    assert n == 5000
    assert calls['n'] <= len(by_id) + 1, \
        f'{calls["n"]} matrix resolutions for {len(by_id)} cells -- memo is broken'
    calls['n'] = 0
    assignment.assign_spots(spots, cells, mask.shape, by_id, matrix_to_shared=to_shared)
    # +1: owner=None is a real memo entry now (unassigned spots recast
    # through the same transform, one resolution per (hybe, modality)).
    assert calls['n'] <= len(by_id) + 1, \
        f'assign_spots resolved {calls["n"]}x -- per-spot resolution is back'


def test_unassigned_spots_are_recast_too():
    """No owner means the cell layer is identity, not that nothing moves:
    the FOV/cross-modal legs still change with matrices, so unassigned
    spots' shared coordinates must refresh like everyone else's."""
    cells, mask = setup()
    s = ASpot(); s.modality = MODALITY
    s.set_metadata(uid=1, fov=FOV, hybe=HYBE, channel=635, cell=-1,
                   raw_coordinate=(10.0, 20.0, 0.0), coordinate=(10.0, 20.0, 0.0))
    H = np.eye(3); H[0, 2], H[1, 2] = 5.0, -3.0
    n = assignment.recast_spots_to_shared([s], lambda h, m, o: H, {c.id: c for c in cells})
    assert n == 1, 'the unassigned spot must be recast'
    assert s.coordinate[:2] == (15.0, 17.0), s.coordinate


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
