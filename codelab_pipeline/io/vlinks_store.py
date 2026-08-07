import os
import pickle
from datetime import datetime

import h5py
import numpy as np


def _cells_group_path(fov):
    return f'/FOV{fov:02d}/cells'


def _unassigned_spots_group_path(fov):
    return f'/FOV{fov:02d}/unassigned_spots'


def write_cells(storage_path, fov, cell_container):
    """
    Real, on-disk persistence for one FOV's cells (with their nested spots
    and per-hybe alignment matrices -- ACell.save() already includes both,
    see codelab_pipeline/models/cell.py) into that experiment's own
    vlinks.h5, at /FOV##/cells/blob.

    A single pickled blob, not a hand-rolled HDF5-native array layout --
    /FOV##/cells/ already exists (created by preprocess.py's aggregate
    vlinks builder) but confirmed empty on every real vlinks.h5 on disk;
    the legacy resizable-array/id-indexed schema the old Jupyter-widget
    classes (segment.py's SegmentWidget etc, both dead code, never
    instantiated by this GUI) would have used is not worth replicating --
    nothing else reads /FOV##/cells/ today, so the simplest fully
    round-trippable choice (reusing CellContainer/ACell's own already-
    correct save()/load()) wins over inventing a schema with no second
    consumer.
    """
    cells = cell_container.data.get(fov, [])
    payload = {'modality': cell_container.modality, 'cells': [cell.save() for cell in cells]}
    blob = np.void(pickle.dumps(payload))
    vlinks_path = os.path.join(storage_path, 'vlinks.h5')
    with h5py.File(vlinks_path, 'a') as f:
        grp = f.require_group(_cells_group_path(fov))
        if 'blob' in grp:
            del grp['blob']
        grp.create_dataset('blob', data=blob)
        grp.attrs['saved_at'] = datetime.now().isoformat()
        grp.attrs['n_cells'] = len(cells)
        grp.attrs['n_spots'] = sum(len(c.spots) for c in cells)


def write_single_cell(storage_path, fov, cell):
    """
    Merges exactly one cell's current save() state into this FOV's
    on-disk cell list, leaving every other cell's persisted data
    untouched. Unlike write_cells (which always serializes the FULL
    in-memory cell_container.data[fov] and overwrites the whole blob),
    this reads the existing disk list first, replaces the matching
    cell.id entry (or appends it, if this cell was never saved before),
    and writes that back -- the narrow "just this one cell" persistence
    "Save Current Spot" needs, since the transient container otherwise
    holds every cell in the FOV and a full-container write would
    silently also push out whatever's currently in memory for all of
    them.
    """
    existing, modality = read_cells(storage_path, fov)
    if existing is None:
        existing = []
    cell_dict = cell.save()
    for i, d in enumerate(existing):
        if d.get('id') == cell.id:
            existing[i] = cell_dict
            break
    else:
        existing.append(cell_dict)
    payload = {'modality': modality or cell.modality, 'cells': existing}
    blob = np.void(pickle.dumps(payload))
    vlinks_path = os.path.join(storage_path, 'vlinks.h5')
    with h5py.File(vlinks_path, 'a') as f:
        grp = f.require_group(_cells_group_path(fov))
        if 'blob' in grp:
            del grp['blob']
        grp.create_dataset('blob', data=blob)
        grp.attrs['saved_at'] = datetime.now().isoformat()
        grp.attrs['n_cells'] = len(existing)
        grp.attrs['n_spots'] = sum(len(d.get('spots', [])) for d in existing)


def read_cells(storage_path, fov):
    """
    Returns (cell_dicts, modality) -- cell_dicts is a list of ACell.save()-
    shaped dicts (feed to CellContainer.load's per-FOV reconstruction), or
    (None, '') if nothing has been persisted for this FOV yet (a
    freshly-ingested experiment, or a FOV never segmented).
    """
    vlinks_path = os.path.join(storage_path, 'vlinks.h5')
    if not os.path.exists(vlinks_path):
        return None, ''
    grp_path = _cells_group_path(fov)
    with h5py.File(vlinks_path, 'r') as f:
        if grp_path not in f or 'blob' not in f[grp_path]:
            return None, ''
        raw = bytes(f[grp_path]['blob'][()])
    payload = pickle.loads(raw)
    return payload['cells'], payload.get('modality', '')


def summarize_fov(storage_path, fov):
    """{'n_cells':, 'n_spots':, 'saved_at':} straight from the group attrs
    (no need to unpickle the whole blob just to count) -- (None if nothing
    persisted for this FOV yet."""
    vlinks_path = os.path.join(storage_path, 'vlinks.h5')
    if not os.path.exists(vlinks_path):
        return None
    grp_path = _cells_group_path(fov)
    with h5py.File(vlinks_path, 'r') as f:
        if grp_path not in f or 'blob' not in f[grp_path]:
            return None
        attrs = f[grp_path].attrs
        return {'n_cells': int(attrs.get('n_cells', 0)),
                'n_spots': int(attrs.get('n_spots', 0)),
                'saved_at': attrs.get('saved_at', '')}


def summarize_all_fovs(storage_path, fov_list):
    return {fov: summarize_fov(storage_path, fov) for fov in fov_list}


def mirror_write_cells(storage_paths, fov, cell_container):
    """write_cells into every distinct storage path given -- the "same
    cell/spot/matrix data lives in both DNA vlinks and RNA vlinks" mirror
    the user explicitly asked for, since a cell's spots span whichever
    hybes/modalities were actually localized for it, not just the modality
    it happened to be segmented in."""
    seen = set()
    for path in storage_paths:
        if not path or path in seen:
            continue
        seen.add(path)
        write_cells(path, fov, cell_container)


def mirror_write_single_cell(storage_paths, fov, cell):
    """write_single_cell into every distinct storage path given -- the
    narrow-write counterpart to mirror_write_cells, same dual-modality
    mirroring rationale (a cell's spots span whichever hybes/modalities
    were actually localized for it)."""
    seen = set()
    for path in storage_paths:
        if not path or path in seen:
            continue
        seen.add(path)
        write_single_cell(path, fov, cell)


def write_fov_spots(storage_path, fov, spots):
    """
    Persists Whole FOV auto-detect spots that don't belong to any cell
    (ASpot.cell left at its model default, -1 -- no cell to link to) at
    a separate top-level location from /FOV##/cells, so this write can
    never collide with or clobber write_cells/write_single_cell's own
    per-cell blob. Full-replace of the whole FOV's unassigned-spot list
    (the caller, _replace_fov_unassigned_spots, already merges per-
    (hybe, channel) in memory before this is called) -- same shape as
    write_cells itself, just a plain list of ASpot.save() dicts instead
    of a {'modality', 'cells'} payload (there's no per-cell modality to
    carry here).
    """
    payload = [spot.save() for spot in spots]
    blob = np.void(pickle.dumps(payload))
    vlinks_path = os.path.join(storage_path, 'vlinks.h5')
    with h5py.File(vlinks_path, 'a') as f:
        grp = f.require_group(_unassigned_spots_group_path(fov))
        if 'blob' in grp:
            del grp['blob']
        grp.create_dataset('blob', data=blob)
        grp.attrs['saved_at'] = datetime.now().isoformat()
        grp.attrs['n_spots'] = len(spots)


def read_fov_spots(storage_path, fov):
    """Returns a list of ASpot.save()-shaped dicts (feed to ASpot().set_metadata(**d)),
    or [] if nothing's been persisted for this FOV yet."""
    vlinks_path = os.path.join(storage_path, 'vlinks.h5')
    if not os.path.exists(vlinks_path):
        return []
    grp_path = _unassigned_spots_group_path(fov)
    with h5py.File(vlinks_path, 'r') as f:
        if grp_path not in f or 'blob' not in f[grp_path]:
            return []
        raw = bytes(f[grp_path]['blob'][()])
    return pickle.loads(raw)


def mirror_write_fov_spots(storage_paths, fov, spots):
    """write_fov_spots into every distinct storage path given -- same
    dual-modality mirroring rationale as mirror_write_cells/
    mirror_write_single_cell."""
    seen = set()
    for path in storage_paths:
        if not path or path in seen:
            continue
        seen.add(path)
        write_fov_spots(path, fov, spots)


def _params_group_path():
    return '/params'


def _fov_params_group_path(fov):
    return f'/params/FOV{fov:02d}'


def write_global_params(storage_path, **params):
    """
    Whole-experiment-scope metadata -- layout_path, same-modality
    alignment's reference hybe/channel, cross-modal alignment's RNA/DNA
    reference hybes/channel/paired storage path, cell-based alignment's
    reference hybe/channel/pad -- as plain HDF5 attrs on a single
    dedicated /params group. Deliberately ONE shallow, well-known
    location (not scattered across /FOV##/ or buried under /cells/) --
    per explicit request, metadata belongs at the top where it's cheaply
    and uniformly reachable, not "at the very last layer" of the data.
    Reading it back is one dict(f['/params'].attrs), no pickling, no
    per-FOV/per-hybe-file scanning. Merges into whatever's already
    stored -- only overwrites the keys actually passed (None values are
    skipped), so each caller (same-modality alignment accept, cross-
    modal accept, cell-alignment run/accept) writes just the slice of
    state its own operation just established, without clobbering the
    others'. Segmentation is deliberately NOT written here -- see
    write_fov_params, it can legitimately differ per FOV.
    """
    vlinks_path = os.path.join(storage_path, 'vlinks.h5')
    with h5py.File(vlinks_path, 'a') as f:
        grp = f.require_group(_params_group_path())
        for k, v in params.items():
            if v is None:
                continue
            grp.attrs[k] = v


def read_global_params(storage_path):
    """{key: value} of whatever's been written via write_global_params,
    or {} if nothing yet / no vlinks.h5 at this storage path -- the read
    half of "parse every current metadata from the storage path" (used
    to refresh session state on activation, before any config-file
    default is allowed to apply)."""
    vlinks_path = os.path.join(storage_path, 'vlinks.h5')
    if not os.path.exists(vlinks_path):
        return {}
    grp_path = _params_group_path()
    with h5py.File(vlinks_path, 'r') as f:
        if grp_path not in f:
            return {}
        return dict(f[grp_path].attrs)


def write_celltype_config(storage_path, fov_ranges_by_celltype, barcode_channel_by_celltype, calibration,
                          barcode_method=None):
    """
    Persists Celltype Determination's ENTIRE setup work -- FOV mode's
    {celltype: range_string} map, Barcode mode's {celltype: (hybe,
    channel)} assignment, the actual computed {'scale'/'lower_bound'/
    'upper_bound': {(hybe,channel): {fov: value}}} calibration (real
    per-FOV bound values, not just the input widget settings that
    produced them), and the classification method (Vote/Median) -- so a
    later session can reconstruct all of it and run/view results
    without re-running Set FOV Ranges / Assign to Selected Celltype /
    Apply Calibration from scratch, same "just usable, no extra step"
    standard cells/spots/alignment matrices already meet elsewhere in
    this app. A single pickled blob at /params/celltype_config_blob
    (same "no HDF5-native schema, nothing else reads this" reasoning as
    write_cells/write_fov_spots) -- (hybe, channel) tuple keys aren't
    representable as plain HDF5 attrs the way write_global_params's flat
    scalars are.
    """
    payload = {
        'fov_ranges_by_celltype': dict(fov_ranges_by_celltype),
        'barcode_channel_by_celltype': dict(barcode_channel_by_celltype),
        'calibration': calibration,
        'barcode_method': barcode_method,
    }
    blob = np.void(pickle.dumps(payload))
    vlinks_path = os.path.join(storage_path, 'vlinks.h5')
    with h5py.File(vlinks_path, 'a') as f:
        grp = f.require_group(_params_group_path())
        if 'celltype_config_blob' in grp:
            del grp['celltype_config_blob']
        grp.create_dataset('celltype_config_blob', data=blob)


def read_celltype_config(storage_path):
    """
    (fov_ranges_by_celltype, barcode_channel_by_celltype, calibration,
    barcode_method) as written by write_celltype_config, or ({}, {},
    {'scale': {}, 'lower_bound': {}, 'upper_bound': {}}, None) if
    nothing's been persisted for this storage path yet.
    """
    empty_calibration = {'scale': {}, 'lower_bound': {}, 'upper_bound': {}}
    vlinks_path = os.path.join(storage_path, 'vlinks.h5')
    if not os.path.exists(vlinks_path):
        return {}, {}, empty_calibration, None
    grp_path = _params_group_path()
    with h5py.File(vlinks_path, 'r') as f:
        if grp_path not in f or 'celltype_config_blob' not in f[grp_path]:
            return {}, {}, empty_calibration, None
        raw = bytes(f[grp_path]['celltype_config_blob'][()])
    payload = pickle.loads(raw)
    return (payload.get('fov_ranges_by_celltype', {}), payload.get('barcode_channel_by_celltype', {}),
            payload.get('calibration', empty_calibration), payload.get('barcode_method'))


def mirror_write_celltype_config(storage_paths, fov_ranges_by_celltype, barcode_channel_by_celltype, calibration,
                                 barcode_method=None):
    """write_celltype_config into every distinct storage path given --
    same dual-modality mirroring rationale as mirror_write_cells (a
    celltype's barcode channel can belong to either modality, so the
    whole config belongs in both)."""
    seen = set()
    for path in storage_paths:
        if not path or path in seen:
            continue
        seen.add(path)
        write_celltype_config(path, fov_ranges_by_celltype, barcode_channel_by_celltype, calibration, barcode_method)


def write_fov_params(storage_path, fov, **params):
    """
    Per-FOV metadata that can legitimately differ FOV-to-FOV within one
    experiment -- today just segmentation_reference_hybe (a user can, and
    did, segment different FOVs against different hybes). Still lives
    under the SAME top-level /params tree as write_global_params (at
    /params/FOV##), not inside /FOV##/cells/ alongside the actual pickled
    cell/spot blob -- one dedicated metadata area, cheap to read
    regardless of whether any cells have ever been saved for this FOV.
    """
    vlinks_path = os.path.join(storage_path, 'vlinks.h5')
    with h5py.File(vlinks_path, 'a') as f:
        grp = f.require_group(_fov_params_group_path(fov))
        for k, v in params.items():
            if v is None:
                continue
            grp.attrs[k] = v


def read_fov_params(storage_path, fov):
    """{key: value} of whatever's been written via write_fov_params for
    this FOV, or {} if nothing yet / no vlinks.h5."""
    vlinks_path = os.path.join(storage_path, 'vlinks.h5')
    if not os.path.exists(vlinks_path):
        return {}
    grp_path = _fov_params_group_path(fov)
    with h5py.File(vlinks_path, 'r') as f:
        if grp_path not in f:
            return {}
        return dict(f[grp_path].attrs)


def write_cross_modal_matrix(storage_path, fov, H):
    """
    Mirrors an already-computed H_across (see
    codelab_pipeline.alignment.chain.write_cross_modal_matrix) into
    vlinks.h5, at /params/FOV##/matrix_across -- same dedicated /params
    tree as every other piece of metadata here. The per-hybe-stack-file
    copy that function writes can only be LOCATED if the DNA reference
    hybe is already known -- a chicken-and-egg problem when
    reconstructing session state from a fresh app launch with nothing
    yet loaded. This copy is reachable from vlinks.h5 alone, no
    reference hybe required to find it.
    """
    vlinks_path = os.path.join(storage_path, 'vlinks.h5')
    with h5py.File(vlinks_path, 'a') as f:
        grp = f.require_group(_fov_params_group_path(fov))
        if 'matrix_across' in grp:
            del grp['matrix_across']
        grp.create_dataset('matrix_across', data=np.asarray(H, dtype='float64'))


def read_cross_modal_matrix(storage_path, fov):
    """The vlinks.h5-mirrored H_across for this FOV, or None if nothing's
    been written here yet (see write_cross_modal_matrix)."""
    vlinks_path = os.path.join(storage_path, 'vlinks.h5')
    if not os.path.exists(vlinks_path):
        return None
    grp_path = _fov_params_group_path(fov)
    with h5py.File(vlinks_path, 'r') as f:
        if grp_path not in f or 'matrix_across' not in f[grp_path]:
            return None
        return f[grp_path]['matrix_across'][:]
