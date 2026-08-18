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


def assign_spots(spots, mask, matrix_to_mask_frame, cells_by_id,
                 matrix_to_shared=None):
    """
    Assign every spot in `spots` against `mask`, in place.

    mask: (height, width) integer label image in the cells' own frame --
        0 background, any other value a cell id. This is the segmentation
        output, reused rather than re-derived.
    matrix_to_mask_frame: callable (hybe, modality) -> 3x3 mapping that
        readout's RAW coordinates into the mask's frame, or None when no
        transform is known. None means "cannot place this spot" and leaves
        it unassigned rather than guessing.
    cells_by_id: {cell id: ACell}, for reading celltype and for the shared
        -frame transform.
    matrix_to_shared: optional callable (hybe, modality, cell) -> 3x3
        mapping raw coordinates into the pipeline's shared frame. When
        given, an assigned spot's `coordinate` is recomputed through it, so
        a spot's shared-frame position always reflects the CURRENT matrices
        rather than whatever they were when it was first localized.

    A spot that lands on background, or whose frame cannot be resolved, is
    set to cell = -1: assignment is recomputed from scratch, so a spot that
    used to belong to a cell and no longer does must lose it. Silently
    keeping a stale owner is how assignment and geometry drift apart.

    Returns (n_assigned, n_unassigned).
    """
    height, width = mask.shape[:2]
    matrix_cache = {}
    n_assigned = n_unassigned = 0

    for spot in spots:
        key = (spot.hybe, getattr(spot, 'modality', '') or '')
        if key not in matrix_cache:
            matrix_cache[key] = matrix_to_mask_frame(*key)
        H = matrix_cache[key]
        owner = None
        if H is not None:
            raw_x, raw_y = float(spot.raw_coordinate[0]), float(spot.raw_coordinate[1])
            mx, my = (np.asarray(H, dtype=float)[:2] @ np.array([raw_x, raw_y, 1.0]))
            ix, iy = int(round(mx)), int(round(my))
            if 0 <= iy < height and 0 <= ix < width:
                label = int(mask[iy, ix])
                if label:
                    owner = cells_by_id.get(label)

        if owner is None:
            spot.cell = -1
            spot.celltype = ''
            n_unassigned += 1
            continue

        spot.cell = int(owner.id)
        spot.celltype = str(getattr(owner, 'celltype', '') or '')
        if matrix_to_shared is not None:
            H_shared = matrix_to_shared(spot.hybe, key[1], owner)
            if H_shared is not None:
                raw_x, raw_y = float(spot.raw_coordinate[0]), float(spot.raw_coordinate[1])
                cx, cy = (np.asarray(H_shared, dtype=float)[:2]
                          @ np.array([raw_x, raw_y, 1.0]))
                spot.coordinate = (float(cx), float(cy), float(spot.coordinate[2]))
        n_assigned += 1

    return n_assigned, n_unassigned


def rasterize_cells(cells, frame_shape):
    """
    A label image from a set of cells, in the frame their masks are native
    to -- 0 background, otherwise the cell's id.

    Built once per assignment run and indexed per spot, which is what makes
    assign_spots O(N) in spots rather than O(N x cells). Later cells win
    where masks overlap; segmentation does not normally produce overlap, and
    a deterministic rule beats an ambiguous one.
    """
    labels = np.zeros(frame_shape[:2], dtype=np.int32)
    height, width = labels.shape
    for cell in cells:
        x, y = cell.area
        x = np.asarray(x).astype(int).ravel()
        y = np.asarray(y).astype(int).ravel()
        keep = (x >= 0) & (y >= 0) & (x < width) & (y < height)
        labels[y[keep], x[keep]] = int(cell.id)
    return labels
