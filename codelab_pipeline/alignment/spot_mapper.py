import os
import numpy as np
import numpy.linalg as la
import h5py

from ..io import vlinks_store

"""
Generic single-coordinate mapper between a hybe's own native (raw,
unmodified) pixel frame and the shared reference frame -- the same "H @
[x, y, 1]" forward transform already used throughout chain.py/cell.py
(align_cell, ACell.get_area_in_readout, localize_cell_2d_worker), pulled
out as a reusable point-level API instead of a mask-array one. No
consumer exists yet (no spot-localization panel), but the math matches
what compute_cell_alignment/localize_cell_2d_worker already do internally,
so it's built and tested now as groundwork.

"localize as-is, move after": a spot is always localized on a hybe's own
untouched image; only its resulting (x, y) coordinate ever moves, via
raw_to_reference below, never the pixel data itself.

Fiducial-spot convention (matches ChrTracer's selectSpots.csv / AnAllele):
a fiducial spot is selected once, on a single reference readout's fiducial
channel -- never re-detected independently per hybe. reference_to_raw then
answers "where does that same physical point sit in this other hybe's own
raw frame", so a crop can be taken there (crop_for_localization) and a
localizer run on the untouched image.
"""


def _resolve_matrix(hybe, fov_matrices, modality=None, cell=None):
    """
    The single 3x3 'yx' matrix mapping `hybe`'s own native (raw) frame to
    the reference frame for this coordinate. If `cell` is given, that
    reference frame is the pipeline's ONE shared reference frame (RNA's
    own same-modality reference hybe -- see ACell.matrix_to_shared's own
    docstring for why this, not cell.reference_hybe, is the destination
    every spot coordinate should land in) -- ACell.matrix_to_shared
    resolves this from cell.matrices/matrix_anchors, with graceful
    identity fallback for either leg per "no no-alignment". Without a
    cell, the reference frame is whatever fov_matrices itself targets,
    which the caller is expected to have already composed with H_across
    when applicable (see main_window._composed_fov_matrices_for_cell_
    alignment) -- this function never re-derives or infers that
    composition itself.

    modality: which modality `hybe` belongs to -- required to look up
    cell.matrices correctly, since it's keyed by (hybe, modality), not
    bare hybe (the cross-modal bridge hybe, e.g. Hyb_130, is a real,
    distinct file in both modalities and would otherwise collide).
    Defaults to cell.modality when cell is given and modality is omitted
    -- correct for the common case of a same-modality lookup.
    """
    if cell is not None:
        key_modality = modality if modality is not None else cell.modality
        return cell.matrix_to_shared(hybe, key_modality)
    if hybe not in fov_matrices:
        raise KeyError(f"No alignment matrix for hybe '{hybe}' -- pass a fov_matrices entry "
                        f"(or a cell with its own residual for this hybe/modality)")
    return fov_matrices[hybe]


def raw_to_reference(coordinate, hybe, fov_matrices, modality=None, cell=None):
    """
    coordinate: (x, y) in `hybe`'s own native/raw pixel frame -- e.g.
    exactly where a spot was localized on that hybe's own unmodified image.
    Returns (x, y) in the shared reference frame by forward-applying H:
    ref_point = H @ raw_point. Only the coordinate moves.
    """
    x, y = coordinate
    H = _resolve_matrix(hybe, fov_matrices, modality, cell)
    rx, ry, _ = H @ np.array([x, y, 1.0])
    return float(rx), float(ry)


def reference_to_raw(coordinate, hybe, fov_matrices, modality=None, cell=None):
    """
    Inverse of raw_to_reference: coordinate is a point already known in the
    shared reference frame (e.g. a fiducial spot selected once on a single
    reference readout's fiducial channel -- never re-derived per hybe).
    Returns where that same physical point sits in `hybe`'s own native/raw
    frame, so a crop can be taken there for localization.
    """
    x, y = coordinate
    H = _resolve_matrix(hybe, fov_matrices, modality, cell)
    rx, ry, _ = la.inv(H) @ np.array([x, y, 1.0])
    return float(rx), float(ry)


def crop_for_localization(storage_path, fov, hybe, channel, native_coordinate, pad=5, use_stack=False):
    """
    native_coordinate: (x, y) already in `hybe`'s own raw frame (typically
    from reference_to_raw). Returns a small crop centered there -- the
    "crop nearby" step, ready for a localizer to run on as-is. use_stack=
    False (default, 2D localization) reads vlinks.h5's real MIP copy, never
    the raw stack file. use_stack=True (3D localization -- the legitimate
    exception) reads the full (height, width, depth) Z-stack from the raw
    stack file instead.

    Returns (crop, (ymin, xmin)): the offset lets a caller convert a
    within-crop peak back to `hybe`'s native frame (raw_x = x + xmin,
    raw_y = y + ymin), matching localize_cell_2d_worker's own convention,
    before handing it to raw_to_reference.
    """
    x, y = native_coordinate
    if not use_stack:
        mip = vlinks_store.read_hybe_mip(storage_path, fov, hybe, channel)
        if mip is None:
            raise ValueError(f'FOV{fov:02d} {hybe} not in vlinks.h5 -- ingest it first.')
        height, width = mip.shape[0], mip.shape[1]
        ymin, ymax = max(0, int(round(y)) - pad), min(height, int(round(y)) + pad + 1)
        xmin, xmax = max(0, int(round(x)) - pad), min(width, int(round(x)) + pad + 1)
        return mip[ymin:ymax, xmin:xmax], (ymin, xmin)

    h5path = os.path.join(storage_path, f'FOV{fov:02d}', f'{hybe}_stack.h5')
    with h5py.File(h5path, 'r') as f:
        dataset = f[f'/stack/ch{channel}']
        height, width = dataset.shape[0], dataset.shape[1]
        ymin, ymax = max(0, int(round(y)) - pad), min(height, int(round(y)) + pad + 1)
        xmin, xmax = max(0, int(round(x)) - pad), min(width, int(round(x)) + pad + 1)
        crop = dataset[ymin:ymax, xmin:xmax, :]
    return crop, (ymin, xmin)
