"""
The one spot-assignment function.

Assignment answers a single question for each spot -- which cell, if any,
contains it -- and then records the consequences: ASpot.cell, ASpot.celltype,
and the spot's position in the shared frame. It is deliberately ONE function
rather than logic spread across the save path, the localization workers and
the FOV viewer, because those drifted apart before: each grew its own notion
of what "assigned" meant and they disagreed.

It is pure. It takes spots, a label mask and two lookups, mutates the spots
and returns a count. It never reads or writes storage, never touches Qt, and
does not care whether a spot was assigned before -- assignment is a field on
a spot, not a location, so re-running it is always safe and always
idempotent for unchanged input.

COST. The mask is a rasterised label image (0 = background, otherwise a cell
id), so containment is one array index per spot rather than a polygon test
against every cell: O(N) in spots and independent of the number of cells.
The per-(hybe, modality) matrix is resolved once and reused for every spot in
that group. See tests/test_spot_assignment_cost.py for measured numbers.
"""
import numpy as np
import numpy.linalg as la


def label_mask_for_frame(cells, hybe, modality, frame_shape, area_in_frame=None):
    """
    A label image of every cell AS SEEN IN (hybe, modality)'s own frame --
    0 background, otherwise the cell's id.

    Each cell is projected from ITS OWN reference hybe, so cells segmented
    in different hybes (or different modalities) compose correctly into one
    mask. That per-cell reference does not need to agree with anything: the
    destination is the SPOT's frame, and every cell is mapped into it
    independently.

    area_in_frame: optional callable (cell, hybe, modality) -> (x, y) that
    does the projection. Session callers MUST pass one backed by the live
    FrameResolver (MainWindow._cell_area_in_readout): the default,
    cell.get_area_in_readout, RAISES by design on residual-form matrices
    (any cell that has run cell alignment), and the except-continue below
    would then silently drop exactly the best-aligned cells from the mask
    -- spots inside them would quietly come back unassigned. The default
    exists only for session-free contexts (tests, legacy composed
    matrices).

    Built once per (hybe, modality) and then indexed per spot, which is what
    makes assignment O(N) in spots with no per-spot matrix work at all --
    the transform cost is paid once per frame, not once per spot.

    Later cells win where masks overlap. Segmentation does not normally
    produce overlap, and a deterministic rule beats an ambiguous one.
    """
    labels = np.zeros(frame_shape[:2], dtype=np.int32)
    height, width = labels.shape
    for cell in cells:
        try:
            if area_in_frame is not None:
                x, y = area_in_frame(cell, hybe, modality)
            else:
                x, y = cell.get_area_in_readout(hybe, modality)
        except Exception:
            continue          # this cell has no path into that frame
        x = np.asarray(x).astype(int).ravel()
        y = np.asarray(y).astype(int).ravel()
        keep = (x >= 0) & (y >= 0) & (x < width) & (y < height)
        labels[y[keep], x[keep]] = int(cell.id)
    return labels


def count_unassigned(spots):
    """How many of `spots` currently belong to no cell."""
    return sum(1 for s in spots if int(getattr(s, 'cell', -1)) == -1)


def assign_spots(spots, cells, frame_shape, cells_by_id=None,
                 matrix_to_shared=None, area_in_frame=None):
    """
    Assign every spot in `spots` against `cells`, in place.

    Spots are grouped by their own (hybe, modality); each group gets one
    label mask built in THAT frame (see label_mask_for_frame), and each spot
    is then a single integer lookup: mask[int(y), int(x)]. Zero means no
    cell. No per-spot matrix work, so cost is one projection per frame plus
    O(N) lookups.

    Takes ALL spots, never just the unassigned ones. Cells get deleted,
    resegmented, or replaced by differently-shaped ones, so a spot that
    already has an owner may no longer sit inside it -- assignment is
    recomputed from scratch and a stale owner is dropped. Walking only the
    unassigned pool could never notice, which is how a dead owner used to
    survive indefinitely.

    matrix_to_shared: optional callable (hybe, modality, cell) -> 3x3 into
    the shared frame; when given, an assigned spot's `coordinate` is
    recomputed through it so shared-frame positions reflect the CURRENT
    matrices rather than whatever they were at localization time.

    area_in_frame: forwarded to label_mask_for_frame (see its docstring --
    session callers must pass the resolver-backed projection).

    Returns (n_assigned, n_unassigned).
    """
    if cells_by_id is None:
        cells_by_id = {c.id: c for c in cells}
    if matrix_to_shared is not None:
        matrix_to_shared = _memoized(matrix_to_shared)
    groups = {}
    for spot in spots:
        groups.setdefault((spot.hybe, getattr(spot, 'modality', '') or ''), []).append(spot)

    n_assigned = n_unassigned = 0
    for (hybe, modality), group in groups.items():
        mask = label_mask_for_frame(cells, hybe, modality, frame_shape,
                                    area_in_frame=area_in_frame)
        height, width = mask.shape
        for spot in group:
            ix = int(spot.raw_coordinate[0])
            iy = int(spot.raw_coordinate[1])
            label = int(mask[iy, ix]) if (0 <= iy < height and 0 <= ix < width) else 0
            owner = cells_by_id.get(label) if label else None
            if owner is None:
                spot.cell = -1
                spot.celltype = ''
                if matrix_to_shared is not None:
                    H = matrix_to_shared(hybe, modality, None)
                    if H is not None:
                        cx, cy = (np.asarray(H, dtype=float)[:2]
                                  @ np.array([float(spot.raw_coordinate[0]),
                                              float(spot.raw_coordinate[1]), 1.0]))
                        spot.coordinate = (float(cx), float(cy), float(spot.coordinate[2]))
                n_unassigned += 1
                continue
            spot.cell = int(owner.id)
            spot.celltype = str(getattr(owner, 'celltype', '') or '')
            if matrix_to_shared is not None:
                H = matrix_to_shared(hybe, modality, owner)
                if H is not None:
                    cx, cy = (np.asarray(H, dtype=float)[:2]
                              @ np.array([float(spot.raw_coordinate[0]),
                                          float(spot.raw_coordinate[1]), 1.0]))
                    spot.coordinate = (float(cx), float(cy), float(spot.coordinate[2]))
            n_assigned += 1
    return n_assigned, n_unassigned


def _memoized(matrix_to_shared):
    """
    Per-run memo over (hybe, modality, owner.id). The transform itself is a
    2x3 matmul; the COST is the caller's matrix resolution (MainWindow's
    matrix_to_shared builds a FrameResolver per call -- measured 540 us per
    spot, 27 SECONDS for a 50k-spot save). One matrix per (hybe, modality,
    cell) is all a run can ever need; memoized here so every caller
    inherits the fix rather than each remembering to cache. Never outlives
    one call: matrices legitimately change between saves.
    """
    cache = {}

    def resolve(hybe, modality, owner):
        key = (hybe, modality, int(owner.id) if owner is not None else -1)
        if key not in cache:
            cache[key] = matrix_to_shared(hybe, modality, owner)
        return cache[key]
    return resolve


def recast_spots_to_shared(spots, matrix_to_shared, cells_by_id):
    """
    Rewrite every ASSIGNED spot's shared-frame `coordinate` from its own
    raw_coordinate under the CURRENT matrices -- ownership untouched.

    The coordinates-only counterpart of assign_spots (which recasts as a
    side effect of assignment): matrix accepts already run full
    reassignment, so this exists for callers that need coordinates
    refreshed without re-deciding ownership -- and as the primitive a lazy
    per-spot map() could delegate to later.

    UNASSIGNED spots are recast too, through the SAME general transform
    with owner=None: no owner just means the cell layer contributes
    identity (FrameResolver's own documented default), not that there is
    nothing to compose -- the FOV and cross-modal legs still move when
    matrices change, and skipping these spots left their shared
    coordinates stale after every matrix accept.
    """
    n = 0
    resolve = _memoized(matrix_to_shared)
    for spot in spots:
        owner = cells_by_id.get(int(spot.cell)) if int(spot.cell) != -1 else None
        H = resolve(spot.hybe, getattr(spot, 'modality', '') or '', owner)
        if H is None:
            continue
        cx, cy = (np.asarray(H, dtype=float)[:2]
                  @ np.array([float(spot.raw_coordinate[0]),
                              float(spot.raw_coordinate[1]), 1.0]))
        spot.coordinate = (float(cx), float(cy), float(spot.coordinate[2]))
        n += 1
    return n
