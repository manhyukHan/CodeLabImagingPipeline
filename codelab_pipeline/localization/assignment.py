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
                y, x = area_in_frame(cell, hybe, modality)
            else:
                y, x = cell.get_area_in_readout(hybe, modality)
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

    matrix_to_shared: optional callable (hybe, modality, cell) ->
    (H_yx 3x3, dz planes) into the shared frame -- a bare 3x3 is also
    accepted (dz = 0.0, kept for older callers/tests). When given, a
    spot's `adj_coordinate` is recomputed IN FULL 3D: (y, x) through H, and
    z as raw_z + dz. Per confirmed real bug: the recast used to carry z
    over untouched, so a cell's own dz and the cross-modal z drift never
    reached any persisted coordinate no matter how many saves ran --
    coordinate[2] stayed equal to raw[2] forever.

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
            # raw_coordinate is (y, x, z) -- rasterized order.
            iy = int(spot.raw_coordinate[0])
            ix = int(spot.raw_coordinate[1])
            label = int(mask[iy, ix]) if (0 <= iy < height and 0 <= ix < width) else 0
            owner = cells_by_id.get(label) if label else None
            if owner is None:
                spot.cell = -1
                spot.celltype = ''
                if matrix_to_shared is not None:
                    _apply_transform(spot, matrix_to_shared(hybe, modality, None))
                n_unassigned += 1
                continue
            spot.cell = int(owner.id)
            spot.celltype = str(getattr(owner, 'celltype', '') or '')
            if matrix_to_shared is not None:
                _apply_transform(spot, matrix_to_shared(hybe, modality, owner))
            n_assigned += 1
    return n_assigned, n_unassigned


def _split_transform(resolved):
    """(H, dz) from a callback result -- bare 3x3 means dz = 0.0."""
    if isinstance(resolved, tuple):
        H, dz = resolved
        return (None, 0.0) if H is None else (np.asarray(H, dtype=float), float(dz))
    return (None, 0.0) if resolved is None else (np.asarray(resolved, dtype=float), 0.0)


def _apply_transform(spot, resolved):
    H, dz = _split_transform(resolved)
    if H is None:
        return False
    cy, cx = H[:2] @ np.array([float(spot.raw_coordinate[0]),
                               float(spot.raw_coordinate[1]), 1.0])
    spot.adj_coordinate = (float(cy), float(cx), float(spot.raw_coordinate[2]) + dz)
    return True


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
    Rewrite every ASSIGNED spot's shared-frame `adj_coordinate` from its own
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
        if _apply_transform(spot, resolve(spot.hybe, getattr(spot, 'modality', '') or '', owner)):
            n += 1
    return n


def recast_fov_spots_worker(payload):
    """One FOV's full post-matrix-write spot recast, in a CHILD PROCESS.

    Top-level and Qt-free by contract (spawn pickles it), and a PROCESS
    rather than a thread because every read/write here is HDF5: h5py
    takes one lock per PROCESS for every call, so doing this work in a
    background THREAD of the GUI process still stalls the GUI's own
    image loads (measured on the real store: a 16.5 ms MIP open became
    2043 ms). That is precisely what made Spot Localization and Cell
    Segmentation crawl during an all-FOVs cell alignment, whatever FOV
    was on screen.

    payload: (any_path, fov, storage_path_by_modality, resolver) --
    `resolver` is a plain-data FrameResolver built by the caller for
    THIS fov (it pickles by design; see analysis/resolvers.py).

    Returns (fov, n_spots, error_or_None). Never raises: one bad FOV
    must not abort a batch.
    """
    from codelab_pipeline.io import analysis_store
    from codelab_pipeline.models.cell_container import CellContainer
    from codelab_pipeline.models.spot import ASpot
    any_path, fov, storage_path_by_modality, resolver = payload
    try:
        dicts = analysis_store.read_spots(any_path, fov)
        if not dicts:
            return fov, 0, None
        spots = []
        for d in dicts:
            spot = ASpot()
            spot.set_metadata(**d)
            spots.append(spot)
        cell_dicts, cells_modality = analysis_store.read_cells(any_path, fov)
        cells = (CellContainer.load({fov: cell_dicts}, modality=cells_modality)
                 .get_cells(fov) if cell_dicts else [])
        cells_by_id = {c.id: c for c in cells}

        def to_shared(hybe, modality, owner):
            # ONE resolver for the whole FOV; the cell leg is an argument
            # to to_shared, so per-cell resolvers were never needed
            fallback = owner.reference_modality if owner is not None else None
            m = modality or fallback
            if resolver.shared is None:
                return None
            return (resolver.to_shared(hybe, m, owner),
                    resolver.z_to_shared(hybe, m, owner))

        if cells:
            assign_spots(spots, cells, cells[0].frame_shape, cells_by_id,
                         matrix_to_shared=to_shared)
        else:
            recast_spots_to_shared(spots, to_shared, cells_by_id)

        by_slice = {}
        for spot in spots:
            by_slice.setdefault(
                (spot.modality, spot.hybe, int(spot.channel)), []).append(spot)
        for (modality, hybe, channel), slice_spots in by_slice.items():
            path = storage_path_by_modality.get(modality) or any_path
            analysis_store.write_spots(path, fov, modality, hybe, channel,
                                       slice_spots)
        return fov, len(spots), None
    except Exception as e:
        return fov, 0, f'{type(e).__name__}: {e}'
