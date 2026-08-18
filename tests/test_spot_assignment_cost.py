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

STORAGE = 'data/chr19_downstream_new/RNA_queue'
FOV = 1
SIZES = [1_000, 10_000, 50_000, 100_000]


def load_cells():
    dicts, modality = V.read_cells(STORAGE, FOV)
    container = CellContainer.load({FOV: dicts}, modality=modality)
    return container.data[FOV]


def make_spots(n, shape, rng):
    ys = rng.integers(0, shape[0], size=n)
    xs = rng.integers(0, shape[1], size=n)
    out = []
    for i in range(n):
        s = ASpot()
        s.modality = 'RNA'
        s.set_metadata(uid=i + 1, fov=FOV, hybe='Hyb_101', channel=635,
                       raw_coordinate=(float(xs[i]), float(ys[i]), 0.0),
                       coordinate=(float(xs[i]), float(ys[i]), 0.0))
        out.append(s)
    return out


_cells = None
_mask = None


def setup():
    global _cells, _mask
    if _cells is None:
        _cells = load_cells()
        assert _cells, 'no cells in the real store to assign against'
        _mask = assignment.rasterize_cells(_cells, _cells[0].frame_shape)
    return _cells, _mask


def test_assignment_is_correct_before_it_is_fast():
    """A spot placed on a known cell pixel must land on that cell."""
    cells, mask = setup()
    by_id = {c.id: c for c in cells}
    target = cells[0]
    x, y = target.area
    px, py = int(np.asarray(x).ravel()[0]), int(np.asarray(y).ravel()[0])

    inside = ASpot(); inside.modality = 'RNA'
    inside.set_metadata(uid=1, fov=FOV, hybe='Hyb_101', channel=635,
                        raw_coordinate=(float(px), float(py), 0.0))
    outside = ASpot(); outside.modality = 'RNA'
    outside.set_metadata(uid=2, fov=FOV, hybe='Hyb_101', channel=635,
                         raw_coordinate=(-50.0, -50.0, 0.0))

    n_a, n_u = assignment.assign_spots([inside, outside], mask,
                                       lambda h, m: np.eye(3), by_id)
    assert inside.cell == target.id, f'expected cell {target.id}, got {inside.cell}'
    assert outside.cell == -1, 'a spot outside the frame must stay unassigned'
    assert (n_a, n_u) == (1, 1)


def test_unresolvable_frame_leaves_spots_unassigned():
    """A None matrix must not be treated as identity."""
    cells, mask = setup()
    s = ASpot(); s.modality = 'RNA'
    s.set_metadata(uid=1, fov=FOV, hybe='Hyb_999', channel=635,
                   raw_coordinate=(10.0, 10.0, 0.0))
    n_a, n_u = assignment.assign_spots([s], mask, lambda h, m: None,
                                       {c.id: c for c in cells})
    assert s.cell == -1 and (n_a, n_u) == (0, 1)


def test_reassignment_drops_a_stale_owner():
    """
    Assignment is recomputed, not patched. A spot carrying an owner it no
    longer sits inside must lose it -- keeping it is how assignment and
    geometry drift apart.
    """
    cells, mask = setup()
    s = ASpot(); s.modality = 'RNA'
    s.set_metadata(uid=1, fov=FOV, hybe='Hyb_101', channel=635, cell=cells[0].id,
                   celltype='something', raw_coordinate=(-99.0, -99.0, 0.0))
    assignment.assign_spots([s], mask, lambda h, m: np.eye(3), {c.id: c for c in cells})
    assert s.cell == -1 and s.celltype == ''


def test_cost_scales_linearly_and_is_affordable_per_save():
    cells, mask = setup()
    by_id = {c.id: c for c in cells}
    rng = np.random.default_rng(0)
    timings = {}
    for n in SIZES:
        spots = make_spots(n, mask.shape, rng)
        t0 = time.perf_counter()
        assignment.assign_spots(spots, mask, lambda h, m: np.eye(3), by_id)
        timings[n] = time.perf_counter() - t0

    print(f'\n  {len(cells)} real cells, mask {mask.shape}')
    for n, dt in timings.items():
        print(f'    {n:7,d} spots  {dt*1000:8.1f} ms   {dt/n*1e6:6.2f} us/spot')

    per_spot = [dt / n for n, dt in timings.items()]
    assert max(per_spot) / min(per_spot) < 4.0, \
        f'per-spot cost is not flat -- looks worse than linear: {per_spot}'
    assert timings[max(SIZES)] < 10.0, \
        f'{max(SIZES):,} spots took {timings[max(SIZES)]:.1f}s -- too slow to re-run per save'


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
