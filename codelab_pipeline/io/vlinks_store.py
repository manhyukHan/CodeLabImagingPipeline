import contextlib
import glob
import os
import pickle
import threading
from datetime import datetime

import h5py

from . import columnar
from . import paths
import numpy as np


_MODALITY_CACHE = {}


def _vlinks_path(storage_path):
    """
    The ONE vlinks.h5 for the project a per-modality stack directory belongs
    to. `storage_path` still names that modality's own raw-stack directory
    (that is genuinely per-modality -- the DAX lands there); vlinks lives one
    level up and is shared by every modality in the project.

    Unifying the file is what lets modality stop being a system-wide mode:
    everything inside is keyed by (modality, hybe), so a caller asks for the
    pair it wants rather than first switching the whole program into a
    modality. See _modality_of for where that key comes from.
    """
    # Layout-versioned (io/paths.py): v2 projects keep analysis state in
    # <dp>/analysis/vlinks.h5, small and single-writer -- MIPs live in
    # their own per-hybe files there, so ingestion never contends with
    # this store at all.
    return paths.vlinks_path(storage_path)



# -- one open handle at a time (see _open_vlinks) --------------------------
#
# HDF5 permits exactly ONE open handle per file per process, and every open
# must agree on flags. This module had 21 independent h5py.File() sites and
# no coordination between them, so two overlapping opens simply raced. That
# is not theoretical -- it was confirmed in a real ingestion run as
#
#     FOV06 Hyb_073: ERROR: ingested but failed to write vlinks.h5 MIP:
#     Unable to open file (file is already open for read-only)
#
# when the Cell/Spot status viewer (GUI thread, several hundred reads in one
# refresh) overlapped IngestionWorker's per-task MIP write on the
# coordinator thread. The conversion itself had already succeeded; only the
# MIP copy was lost, and Append mode cannot repair it afterwards because
# convert_dax_to_h5_worker skips on the stack file merely EXISTING. So a
# collision here costs real, hard-to-notice data.
#
# Every vlinks.h5 access now goes through _open_vlinks, which serializes on
# a re-entrant module lock: a writer that arrives mid-read WAITS instead of
# failing. The lock is deliberately held for as long as a handle is open --
# that is what makes the one-handle guarantee hold, and it is what lets
# vlinks_session below batch many reads into a single open.

_VLINKS_LOCK = threading.RLock()

# {abs path: [file, depth, mode]}. Only ever holds one entry at a time; the
# lock guarantees any nested lookup is on the thread that opened it.
_OPEN_VLINKS = {}


@contextlib.contextmanager
def _open_vlinks(vlinks_path, mode='r'):
    """
    The single door to vlinks.h5. Same use as h5py.File(path, mode), but
    serialized process-wide and re-entrant, so a nested open reuses the
    handle its enclosing caller already holds instead of asking HDF5 for a
    second one it will refuse to give.
    """
    key = os.path.abspath(vlinks_path)
    with _VLINKS_LOCK:
        entry = _OPEN_VLINKS.get(key)
        if entry is not None:
            # Re-entrant reuse. Only reachable on the thread already inside
            # the lock, so no other thread can observe a half-open handle.
            if mode != 'r' and entry[2] == 'r':
                raise RuntimeError(
                    f'{key} is already open read-only further up this call '
                    f'stack (a vlinks_session, most likely) -- a {mode!r} '
                    f'write cannot be nested inside a read session')
            entry[1] += 1
            try:
                yield entry[0]
            finally:
                entry[1] -= 1
            return

        f = h5py.File(key, mode)
        _OPEN_VLINKS[key] = [f, 1, mode]
        try:
            yield f
        finally:
            entry = _OPEN_VLINKS[key]
            entry[1] -= 1
            if entry[1] == 0:
                del _OPEN_VLINKS[key]
                f.close()


@contextlib.contextmanager
def vlinks_session(storage_path, mode='r'):
    """
    Hold ONE vlinks.h5 handle open across a batch of reads.

    Every read_* call made inside the block reuses this handle rather than
    opening and closing the file for itself. That matters because the reads
    are not the expensive part -- the opens are. One Cell/Spot status
    refresh issues several hundred of them (one per FOV per panel), and each
    one is both a syscall storm and a window in which the ingestion
    coordinator's MIP write can collide.

    Callers need change nothing else: read_spots/read_cells/... are unaware
    of the session and pick the open handle up through _open_vlinks.

    The module lock is held for the whole block, so a concurrent writer
    waits here rather than failing. Keep a session to a BOUNDED batch of
    reads for that reason -- an unbounded one stalls ingestion's MIP writes
    for its duration. Nothing is lost when that happens (the writes queue
    and catch up) but progress stops moving, so this is a knob to use
    deliberately, not to wrap the whole app in.

    Yields None if the store does not exist yet; every read_* already
    guards for that itself, so a caller can ignore the yielded value and
    simply use the session for its batching effect.
    """
    path = _vlinks_path(storage_path)
    if not os.path.exists(path):
        yield None
        return
    with _open_vlinks(path, mode) as f:
        yield f


# -- coordinate-order schema guard (convention.py) -------------------------

def _stamp_order(f):
    """Every write stamps the store as rasterized (y, x). A BRAND-NEW
    store is additionally stamped analysis_schema='columnar' (phase-2
    typed datasets, io/columnar.py); existing stores keep whatever they
    are until tools/migrate_analysis_columnar.py converts them."""
    f.attrs['coordinate_order'] = 'yx'
    if 'analysis_schema' not in f.attrs and len(f.keys()) == 0:
        f.attrs['analysis_schema'] = 'columnar'


def _schema(f):
    v = f.attrs.get('analysis_schema', 'pickle')
    return v.decode() if isinstance(v, bytes) else str(v)


def _reset_group(f, path):
    if path in f:
        del f[path]
    return f.require_group(path)


def _require_yx(f, vlinks_path):
    """
    Refuse to read a store that predates the Y/X unification (or was
    written x-major). Loud by design: silently reading swapped
    coordinates/matrices produces plausible-looking, wrong positions
    everywhere. Run tools/migrate_store_to_yx.py once per store (it
    conjugates every matrix and swaps every coordinate tuple), or remake
    the store from raw data.
    """
    order = f.attrs.get('coordinate_order')
    if order != 'yx':
        raise ValueError(
            f"{vlinks_path} is not stamped coordinate_order='yx' "
            f"(found {order!r}) -- this store predates the Y/X unification. "
            f"Run tools/migrate_store_to_yx.py on it once, or remake the data.")

def modality_of(storage_path):
    """
    Which modality owns this stack directory, read from the `modality` attr
    ingestion writes onto every {hybe}_stack.h5.

    Deliberately NOT taken from app config or a directory-name convention:
    ingestion is the one place modality is genuinely authoritative (it is
    the only step that knows which ExperimentLayout a DAX came from), so
    every later reader derives it from that record instead of carrying its
    own notion. Cached per directory -- this reads attrs only, never pixels.
    """
    key = os.path.abspath(storage_path)
    if key in _MODALITY_CACHE:
        return _MODALITY_CACHE[key]
    declared_v2 = paths.modality_from_path(storage_path)
    if declared_v2:
        # v2 project: the manifest IS the registry -- no HDF5, no
        # bootstrapping order, storage_path's basename names the modality.
        _MODALITY_CACHE[key] = declared_v2
        return declared_v2
    for fov_dir in sorted(glob.glob(os.path.join(key, 'FOV*'))):
        for stack in sorted(glob.glob(os.path.join(fov_dir, '*_stack.h5'))):
            try:
                with h5py.File(stack, 'r') as f:
                    m = f.attrs.get('modality')
            except OSError:
                continue
            if m is not None:
                m = m.decode() if isinstance(m, bytes) else str(m)
                _MODALITY_CACHE[key] = m
                return m
    declared = _DECLARED_MODALITY.get(key)
    if declared:
        # UI-declared mapping (declare_modality): lets every modality-
        # scoped read/write work on a COMPLETELY FRESH storage path,
        # before anything was ever ingested -- parse-layout, global
        # params, MIP reads. Deliberately NOT cached: once a real
        # ingested stack lands, its own attr (data truth) wins above.
        return declared
    raise ValueError(
        f'cannot determine modality for {storage_path}: nothing ingested '
        f'there carries a `modality` attr and no modality was declared '
        f'for it (MainWindow declares every configured storage path via '
        f'declare_modality).')


_DECLARED_MODALITY = {}


def declare_modality(storage_path, name):
    """
    Register which configured modality owns storage_path, from the UI's
    own Ingestion state. This is what lets a FRESH project work: before
    anything is ingested, no stack file exists to carry the modality
    attr, and every modality-scoped path (params, MIPs, matrices) used
    to raise -- a boot gate on the very first Parse Layout of a new
    dataset (confirmed real, single-modality Windows session
    2026-08-20). Data truth still wins: modality_of prefers an ingested
    stack's own attr and uses this only as the pre-ingestion fallback.
    """
    if storage_path and name:
        _DECLARED_MODALITY[os.path.abspath(storage_path)] = str(name)


def _cells_group_path(fov):
    return f'/FOV{fov:02d}/cells'


def _spots_group_path(fov):
    """
    The ONE spot store for a FOV: every spot, assigned or not. ASpot.cell
    carries the assignment (-1 = unassigned), so assignment is a field
    write, never a move between stores. Replaces the old split, where
    assigned spots were serialized inside the /FOV##/cells blob and
    unassigned ones lived in /FOV##/unassigned_spots -- a partition BY
    assignment state, which made every reassignment a cross-store move and
    made any control that wrote one store unable to see the other.
    """
    return f'/FOV{fov:02d}/spots'


def allocate_spot_uids(storage_path, fov, count):
    """
    `count` fresh, never-before-used spot uids for this FOV, as a list.

    A per-FOV monotonic counter kept in the spot group's `next_uid` attr.
    uids are never reused, so a uid identifies one spot for its whole life:
    that is what lets a save merge by identity, an undo diff distinguish
    "moved" from "deleted and re-added", and a staleness mark survive a
    refit. Nothing derived from a mutable field could do that -- `cell`
    changes on assignment, `raw_coordinate` changes when 3D localization
    refines the fit, display numbering is recomputed per view.

    On every call the counter is floored above the highest uid actually
    present, so a counter that was lost or reset can never hand out a uid
    that is already in use -- silently reusing one would make two different
    spots indistinguishable to exactly the machinery uid exists to serve.
    """
    vlinks_path = _vlinks_path(storage_path)
    grp_path = _spots_group_path(fov)
    with _open_vlinks(vlinks_path, 'a') as f:
        _stamp_order(f)
        grp = f.require_group(grp_path)
        next_uid = int(grp.attrs.get('next_uid', 1))
        highest = int(grp.attrs.get('highest_uid_seen', 0))
        start = max(next_uid, highest + 1, 1)
        grp.attrs['next_uid'] = start + int(count)
        grp.attrs['highest_uid_seen'] = start + int(count) - 1
        return list(range(start, start + int(count)))


def _spot_slice_path(fov, modality, hybe, channel):
    """
    One blob per (modality, hybe, channel) inside the FOV's spot group.

    That is exactly the scope a save writes and a removal clears, so a save
    replaces one blob and can never touch a hybe the user never opened.
    Assigned and unassigned spots share the blob -- they differ only in
    ASpot.cell -- so no operation here needs to know or care which is which.
    """
    return f'{_spots_group_path(fov)}/{modality}/{hybe}/ch{int(channel)}'


def write_spots(storage_path, fov, modality, hybe, channel, spots):
    """
    Full replace of ONE (modality, hybe, channel) slice with `spots`.

    Full replace within the slice, so deletions propagate: a spot the user
    removed is simply absent from `spots` and therefore gone from disk.
    Scoped to the slice, so it cannot delete anything outside it. Both
    properties are needed together -- a FOV-wide replace would propagate
    deletions the user never made, in hybes they never opened.

    Any spot still carrying uid 0 is allocated one here, so nothing ever
    reaches storage without a stable identity.
    """
    spots = list(spots)
    unallocated = [sp for sp in spots if not getattr(sp, 'uid', 0)]
    if unallocated:
        for sp, uid in zip(unallocated, allocate_spot_uids(storage_path, fov, len(unallocated))):
            sp.uid = uid
    payload = [sp.save() for sp in spots]
    seen = {}
    for d in payload:
        if d['uid'] in seen:
            raise ValueError(
                f'duplicate spot uid {d["uid"]} in FOV{fov:02d} '
                f'{modality}/{hybe}/ch{channel} -- uid must identify one spot')
        seen[d['uid']] = True
    with _open_vlinks(_vlinks_path(storage_path), 'a') as f:
        _stamp_order(f)
        if _schema(f) == 'columnar':
            grp = _reset_group(f, _spot_slice_path(fov, modality, hybe, channel))
            columnar.pack_spots(grp, payload)
        else:
            grp = f.require_group(_spot_slice_path(fov, modality, hybe, channel))
            if 'blob' in grp:
                del grp['blob']
            grp.create_dataset('blob', data=np.void(pickle.dumps(payload)))
        grp.attrs['saved_at'] = datetime.now().isoformat()
        grp.attrs['n_spots'] = len(payload)


def read_spots(storage_path, fov, modality=None, hybe=None, channel=None):
    """
    ASpot.save()-shaped dicts. With modality/hybe/channel given, just that
    slice; with none, every spot in the FOV across every slice. Assigned and
    unassigned come back together -- filter on 'cell' (-1 = unassigned) if a
    caller wants one or the other.
    """
    vlinks_path = _vlinks_path(storage_path)
    if not os.path.exists(vlinks_path):
        return []
    out = []
    with _open_vlinks(vlinks_path, 'r') as f:
        _require_yx(f, vlinks_path)
        if modality is not None and hybe is not None and channel is not None:
            gp = _spot_slice_path(fov, modality, hybe, channel)
            if gp in f:
                if 'table' in f[gp]:
                    out.extend(columnar.unpack_spots(f[gp]))
                elif 'blob' in f[gp]:
                    out.extend(pickle.loads(bytes(f[gp]['blob'][()])))
            return out
        root = _spots_group_path(fov)
        if root not in f:
            return out
        for mod in f[root]:
            if not isinstance(f[f'{root}/{mod}'], h5py.Group):
                continue
            for hy in f[f'{root}/{mod}']:
                for ch in f[f'{root}/{mod}/{hy}']:
                    g = f[f'{root}/{mod}/{hy}/{ch}']
                    if 'table' in g:
                        out.extend(columnar.unpack_spots(g))
                    elif 'blob' in g:
                        out.extend(pickle.loads(bytes(g['blob'][()])))
    return out


def _alleles_group_path(fov):
    return f'/FOV{fov:02d}/alleles'


def write_cells(storage_path, fov, cell_container):
    """
    Real, on-disk persistence for one FOV's cells (with their per-hybe
    alignment matrices -- see ACell.save() in codelab_pipeline/models/
    cell.py; spots are NOT included, they live in the FOV's own spot
    store, see write_spots/read_spots) into that experiment's own
    vlinks.h5, at /FOV##/cells/blob.

    A single pickled blob, not a hand-rolled HDF5-native array layout --
    /FOV##/cells/ already exists (created by preprocess.py's aggregate
    vlinks builder) but confirmed empty on every real vlinks.h5 on disk;
    the legacy resizable-array/id-indexed schema the old Jupyter-widget
    classes (legacy/segment_widgets.py's SegmentWidget etc, dead code,
    never instantiated by this GUI) would have used is not worth replicating --
    nothing else reads /FOV##/cells/ today, so the simplest fully
    round-trippable choice (reusing CellContainer/ACell's own already-
    correct save()/load()) wins over inventing a schema with no second
    consumer.
    """
    cells = cell_container.get_cells(fov)
    dicts = [cell.save() for cell in cells]
    vlinks_path = _vlinks_path(storage_path)
    with _open_vlinks(vlinks_path, 'a') as f:
        _stamp_order(f)
        if _schema(f) == 'columnar':
            grp = _reset_group(f, _cells_group_path(fov))
            columnar.pack_cells(grp, dicts)
        else:
            grp = f.require_group(_cells_group_path(fov))
            if 'blob' in grp:
                del grp['blob']
            grp.create_dataset('blob', data=np.void(pickle.dumps({'cells': dicts})))
        grp.attrs['saved_at'] = datetime.now().isoformat()
        grp.attrs['n_cells'] = len(cells)
        # No n_spots attr: spots are not in this blob any more, and a
        # count taken here would be permanently 0 -- read_spots on the FOV's
        # own store is the real count.


        # No n_spots attr: spots are not in this blob any more, and a
        # count taken here would be permanently 0 -- read_spots on the FOV's
        # own store is the real count.


def read_cells(storage_path, fov):
    """
    Returns (cell_dicts, modality) -- cell_dicts is a list of ACell.save()-
    shaped dicts (feed to CellContainer.load's per-FOV reconstruction), or
    (None, '') if nothing has been persisted for this FOV yet (a
    freshly-ingested experiment, or a FOV never segmented).
    """
    vlinks_path = _vlinks_path(storage_path)
    if not os.path.exists(vlinks_path):
        return None, ''
    grp_path = _cells_group_path(fov)
    with _open_vlinks(vlinks_path, 'r') as f:
        _require_yx(f, vlinks_path)
        if grp_path not in f:
            return None, ''
        if 'table' in f[grp_path]:
            return columnar.unpack_cells(f[grp_path]), ''
        if 'blob' not in f[grp_path]:
            return None, ''
        raw = bytes(f[grp_path]['blob'][()])
    payload = pickle.loads(raw)
    # Second element is a legacy slot: containers/cells no longer carry a
    # modality (frame identity is each cell's own (reference_hybe,
    # reference_modality) pair). Kept as '' so `cells, _ = read_cells(...)`
    # unpacking keeps working while call sites migrate.
    return payload['cells'], ''



def distinct_stores(storage_paths):
    """
    One representative storage_path per DISTINCT physical vlinks.h5.

    Since the vlinks unification, every storage path inside one project
    resolves (via _vlinks_path, one level up) to the SAME file -- so
    deduping by the raw path STRING, as the mirrors used to, still wrote
    the identical payload once per modality (two writes of the same blob
    to the same file on every save). Deduping by the RESOLVED file keeps
    the mirror concept correct for a hypothetical multi-project session
    while collapsing the normal case to exactly one write. Shared by
    every mirror_* writer AND by main_window's own per-path read/write
    loops, so no caller re-derives the rule.
    """
    seen, out = set(), []
    for path in storage_paths:
        if not path:
            continue
        resolved = os.path.abspath(_vlinks_path(path))
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(path)
    return out


def mirror_write_cells(storage_paths, fov, cell_container):
    """write_cells into every DISTINCT vlinks store among the given paths
    (see distinct_stores) -- the "same cell/spot/matrix data lives in both
    DNA vlinks and RNA vlinks" mirror the user explicitly asked for, since
    a cell's spots span whichever hybes/modalities were actually localized
    for it, not just the modality it happened to be segmented in."""
    for path in distinct_stores(storage_paths):
        write_cells(path, fov, cell_container)



def write_fov_alleles(storage_path, fov, alleles):
    """
    Persists one FOV's chromatin-tracing alleles (AnAllele.save()-shaped
    dicts -- id/anchor_hybe/anchor_channel/coordinate/raw_coordinate/
    fiducial_trace/polymer/rejected_hybes/final_polymer) at their own
    top-level location, same shape/rationale as write_fov_spots: a plain
    pickled list, full-replace of the whole FOV's allele list, never
    touching /FOV##/cells or /FOV##/unassigned_spots.
    """
    payload = [allele.save() for allele in alleles]
    vlinks_path = _vlinks_path(storage_path)
    with _open_vlinks(vlinks_path, 'a') as f:
        _stamp_order(f)
        if _schema(f) == 'columnar':
            grp = _reset_group(f, _alleles_group_path(fov))
            columnar.pack_alleles(grp, payload)
        else:
            grp = f.require_group(_alleles_group_path(fov))
            if 'blob' in grp:
                del grp['blob']
            grp.create_dataset('blob', data=np.void(pickle.dumps(payload)))
        grp.attrs['saved_at'] = datetime.now().isoformat()
        grp.attrs['n_alleles'] = len(alleles)


def read_fov_alleles(storage_path, fov):
    """Returns a list of AnAllele.save()-shaped dicts (feed to
    AnAllele().set_metadata(**d)), or [] if nothing's been persisted for
    this FOV yet."""
    vlinks_path = _vlinks_path(storage_path)
    if not os.path.exists(vlinks_path):
        return []
    grp_path = _alleles_group_path(fov)
    with _open_vlinks(vlinks_path, 'r') as f:
        _require_yx(f, vlinks_path)
        if grp_path not in f:
            return []
        if 'table' in f[grp_path]:
            return columnar.unpack_alleles(f[grp_path])
        if 'blob' not in f[grp_path]:
            return []
        raw = bytes(f[grp_path]['blob'][()])
    return pickle.loads(raw)


def mirror_write_fov_alleles(storage_paths, fov, alleles):
    """write_fov_alleles into every DISTINCT vlinks store among the given
    paths (see distinct_stores) -- same dual-modality mirroring rationale
    as mirror_write_cells."""
    for path in distinct_stores(storage_paths):
        write_fov_alleles(path, fov, alleles)


MODALITY_SCOPED_PARAMS = frozenset({
    'layout_path',
    'dax_directory',
    'same_modality_reference_hybe',
    'same_modality_channel_type',
})
"""
Params that describe ONE modality rather than the project. They live under
/params/modalities/{modality}/ so two modalities cannot overwrite each
other -- with a single unified vlinks.h5 every storage_path resolves to the
same file, so a per-modality fact written to the shared /params group is
destroyed by whichever modality is written second. Everything not listed
here describes the project or the RELATIONSHIP between modalities (the
cross-modal reference-hybe pair and channel type, cell-alignment settings)
and is correctly stored once.
"""


def _fov_params_group_path(fov):
    """Per-FOV metadata area. No longer holds user params (the only one,
    segmentation_reference_hybe, was write-only and is now per-cell) -- it
    remains the home of the reference-hybe-independent cross-modal matrix
    and Z-shift, which are genuinely per-FOV measurements."""
    return f'/params/FOV{fov:02d}'


def _modality_params_group_path(modality):
    return f'/params/modalities/{modality}'


def _params_group_path():
    return '/params'


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
    others'.

    Keys in MODALITY_SCOPED_PARAMS are routed to
    /params/modalities/{modality} instead of the shared group -- see that
    table for why. Segmentation reference hybe is NOT stored here or
    anywhere: it lives on each cell (ACell.reference_hybe /
    nucleus_hybe with their modalities), and cells in one FOV can
    legitimately disagree, so no single FOV-level value can represent it.
    """
    vlinks_path = _vlinks_path(storage_path)
    modality = None
    with _open_vlinks(vlinks_path, 'a') as f:
        _stamp_order(f)
        shared = f.require_group(_params_group_path())
        for k, v in params.items():
            if v is None:
                continue
            if k in MODALITY_SCOPED_PARAMS:
                if modality is None:
                    modality = modality_of(storage_path)
                f.require_group(_modality_params_group_path(modality)).attrs[k] = v
            else:
                shared.attrs[k] = v


def read_global_params(storage_path):
    """{key: value} of whatever's been written via write_global_params,
    or {} if nothing yet / no vlinks.h5 at this storage path -- the read
    half of "parse every current metadata from the storage path" (used
    to refresh session state on activation, before any config-file
    default is allowed to apply)."""
    vlinks_path = _vlinks_path(storage_path)
    if not os.path.exists(vlinks_path):
        return {}
    out = {}
    with _open_vlinks(vlinks_path, 'r') as f:
        _require_yx(f, vlinks_path)
        grp_path = _params_group_path()
        if grp_path in f:
            out.update(dict(f[grp_path].attrs))
        mod_path = _modality_params_group_path(modality_of(storage_path))
        if mod_path in f:
            out.update(dict(f[mod_path].attrs))
    return out


def write_celltype_config(storage_path, fov_ranges_by_celltype, barcode_channel_by_celltype, calibration,
                          barcode_method=None):
    """
    Persists Celltype Determination's ENTIRE setup work -- FOV mode's
    {celltype: range_string} map, Barcode mode's {celltype: (hybe,
    channel, modality)} assignment, the actual computed {'scale'/
    'lower_bound'/'upper_bound': {(hybe,channel,modality): {fov: value}}}
    calibration (real per-FOV bound values, not just the input widget
    settings that produced them), and the classification method (Vote/
    Median) -- so a later session can reconstruct all of it and run/view
    results without re-running Set FOV Ranges / Assign to Selected
    Celltype / Apply Calibration from scratch, same "just usable, no
    extra step" standard cells/spots/alignment matrices already meet
    elsewhere in this app. A single pickled blob at /params/
    celltype_config_blob (same "no HDF5-native schema, nothing else
    reads this" reasoning as write_cells/write_fov_spots) -- (hybe,
    channel, modality) tuple keys aren't representable as plain HDF5
    attrs the way write_global_params's flat scalars are.

    The modality element is what lets a barcode channel be identified
    unambiguously even when two modalities happen to share a hybe folder
    name -- read_celltype_config's own caller is responsible for
    dropping any OLDER-format 2-tuple (hybe, channel) entry it encounters
    (no modality tag to recover, same "drop rather than guess wrong"
    policy codelab_pipeline.models.cell_container's own
    _drop_legacy_matrix_keys uses for its analogous pre-tuple-key
    format); this function itself just round-trips whatever it's given.
    """
    payload = {
        'fov_ranges_by_celltype': dict(fov_ranges_by_celltype),
        'barcode_channel_by_celltype': dict(barcode_channel_by_celltype),
        'calibration': calibration,
        'barcode_method': barcode_method,
    }
    blob = np.void(pickle.dumps(payload))
    vlinks_path = _vlinks_path(storage_path)
    with _open_vlinks(vlinks_path, 'a') as f:
        _stamp_order(f)
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
    vlinks_path = _vlinks_path(storage_path)
    if not os.path.exists(vlinks_path):
        return {}, {}, empty_calibration, None
    grp_path = _params_group_path()
    with _open_vlinks(vlinks_path, 'r') as f:
        _require_yx(f, vlinks_path)
        if grp_path not in f or 'celltype_config_blob' not in f[grp_path]:
            return {}, {}, empty_calibration, None
        raw = bytes(f[grp_path]['celltype_config_blob'][()])
    payload = pickle.loads(raw)
    return (payload.get('fov_ranges_by_celltype', {}), payload.get('barcode_channel_by_celltype', {}),
            payload.get('calibration', empty_calibration), payload.get('barcode_method'))


def mirror_write_celltype_config(storage_paths, fov_ranges_by_celltype, barcode_channel_by_celltype, calibration,
                                 barcode_method=None):
    """write_celltype_config into every DISTINCT vlinks store among the
    given paths (see distinct_stores) -- same dual-modality mirroring
    rationale as mirror_write_cells (a celltype's barcode channel can
    belong to either modality, so the whole config belongs in both)."""
    for path in distinct_stores(storage_paths):
        write_celltype_config(path, fov_ranges_by_celltype, barcode_channel_by_celltype, calibration, barcode_method)


def _mip_group_path(fov, modality, hybe):
    return f'/FOV{fov:02d}/mip/{modality}/{hybe}'


def _fov_matrix_group_path(fov, modality):
    return f'/FOV{fov:02d}/matrix/{modality}'


def write_hybe_mip(storage_path, fov, hybe, channel_mips, fiducial_channel=None):
    """
    Real (non-virtual) copy of one hybe's MIP, one dataset per channel,
    into vlinks.h5 at /FOV##/mip/{hybe}/ch{channel} -- the fix for
    principle 4 ("store MIP in vlinks"): the old aggregate vlinks builder
    (legacy/preprocess_legacy.py's dax_vlinks_h5/vlinks_h5, dead code) only
    ever wrote h5py.VirtualSource entries here, which still require the
    original per-hybe {hybe}_stack.h5 files to physically exist at the
    SAME absolute path recorded at creation time -- not self-contained,
    and silently broken the moment a data folder moves (confirmed on real
    data: every existing vlinks.h5's /mip and /stack are_virtual=True,
    pointing at an absolute path). This writes an actual copy of the pixel
    data instead, so vlinks.h5 alone is enough to visualize/inspect a
    hybe's MIP with no dependency on the raw stack file at all.

    Deliberately keyed by hybe (/mip/{hybe}/ch{c}), not a single aggregate
    per-FOV array indexed by hybe position the way the old builder's
    /mip/ch{c} (shape (n_hybes, h, w)) was -- self-contained per hybe, no
    dependency on a shared hybe_list ordering being correct or even
    present, and no need to rewrite every other hybe's entry when one new
    hybe is ingested.

    Also seeds /FOV##/matrix/{hybe} = identity here if nothing is written
    there yet, mirroring preprocess.convert_dax_to_h5_worker's own
    per-hybe-stack-file convention of seeding /matrix/{hybe} to identity at
    ingestion time -- "no alignment run yet" must default to identity,
    never be treated as an error or a missing hybe (see
    read_same_modality_matrices below).

    channel_mips: {channel (int or str): 2D ndarray}. fiducial_channel
    (optional): that hybe's own fiducial channel (from its ExperimentLayout
    record), stashed as a group attr so fiducial_channel_mip/
    readout_channel_mip below can resolve "which channel is fiducial" from
    vlinks.h5 alone -- the same per-hybe attr chain.py's
    _fiducial_channel_mip/_readout_channel_mip read directly off the raw
    stack file's own .attrs, now mirrored here so display code never needs
    that raw file just to answer this.
    """
    if paths.is_v2(storage_path):
        # v2: one small standalone file per (modality, FOV, hybe), written
        # ATOMICALLY (.part + replace) so existence == completeness --
        # this is what lets the ingestion WORKER write it with no shared-
        # file contention and the checkup become a directory listing.
        # vlinks.h5 is never touched here (no identity seeding either:
        # absence of a matrix is identity by the pipeline's own rule).
        target = paths.mip_path(storage_path, fov, hybe)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        tmp = target + '.part'
        with h5py.File(tmp, 'w') as f:
            f.attrs['coordinate_order'] = 'yx'
            if fiducial_channel is not None:
                f.attrs['fiducial_channel'] = int(fiducial_channel)
            for ch, mip in channel_mips.items():
                f.create_dataset(f'ch{ch}', data=np.asarray(mip),
                                 chunks=True, compression='gzip', compression_opts=1)
        os.replace(tmp, target)
        return
    vlinks_path = _vlinks_path(storage_path)
    with _open_vlinks(vlinks_path, 'a') as f:
        _stamp_order(f)
        grp = f.require_group(_mip_group_path(fov, modality_of(storage_path), hybe))
        for ch, mip in channel_mips.items():
            name = f'ch{ch}'
            if name in grp:
                del grp[name]
            grp.create_dataset(name, data=np.asarray(mip))
        if fiducial_channel is not None:
            grp.attrs['fiducial_channel'] = int(fiducial_channel)
        mgrp = f.require_group(_fov_matrix_group_path(fov, modality_of(storage_path)))
        if hybe not in mgrp:
            mgrp.create_dataset(hybe, data=np.eye(3, dtype='float32'))


def read_hybe_mip(storage_path, fov, hybe, channel):
    """The vlinks.h5-stored MIP for one hybe/channel (see write_hybe_mip),
    or None if this hybe hasn't been ingested yet / no vlinks.h5 yet."""
    if paths.is_v2(storage_path):
        try:
            with h5py.File(paths.mip_path(storage_path, fov, hybe), 'r') as f:
                name = f'ch{channel}'
                return f[name][:] if name in f else None
        except OSError:
            return None
    vlinks_path = _vlinks_path(storage_path)
    if not os.path.exists(vlinks_path):
        return None
    grp_path = _mip_group_path(fov, modality_of(storage_path), hybe)
    with _open_vlinks(vlinks_path, 'r') as f:
        _require_yx(f, vlinks_path)
        name = f'ch{channel}'
        if grp_path not in f or name not in f[grp_path]:
            return None
        return f[grp_path][name][:]


def fiducial_channel_mip(storage_path, fov, hybe):
    """
    The fiducial channel's MIP for a hybe, read entirely from vlinks.h5 --
    vlinks-based counterpart to codelab_pipeline.alignment.chain's
    _fiducial_channel_mip, which reads the raw stack file's own .attrs to
    resolve the channel. Returns None if this hybe (or its fiducial_channel
    attr -- only present for hybes ingested after write_hybe_mip started
    stashing it) isn't in vlinks.h5 yet.
    """
    if paths.is_v2(storage_path):
        try:
            with h5py.File(paths.mip_path(storage_path, fov, hybe), 'r') as f:
                if 'fiducial_channel' not in f.attrs:
                    return None
                name = f"ch{int(f.attrs['fiducial_channel'])}"
                return f[name][:] if name in f else None
        except OSError:
            return None
    vlinks_path = _vlinks_path(storage_path)
    if not os.path.exists(vlinks_path):
        return None
    grp_path = _mip_group_path(fov, modality_of(storage_path), hybe)
    with _open_vlinks(vlinks_path, 'r') as f:
        _require_yx(f, vlinks_path)
        if grp_path not in f or 'fiducial_channel' not in f[grp_path].attrs:
            return None
        channel = int(f[grp_path].attrs['fiducial_channel'])
        name = f'ch{channel}'
        return f[grp_path][name][:] if name in f[grp_path] else None


def readout_channel_mip(storage_path, fov, hybe):
    """
    The one non-fiducial channel's MIP for a hybe, read entirely from
    vlinks.h5 -- vlinks-based counterpart to
    codelab_pipeline.alignment.chain's _readout_channel_mip. Returns None
    if this hybe/its fiducial_channel attr isn't in vlinks.h5 yet, or if it
    genuinely has no non-fiducial channel.
    """
    if paths.is_v2(storage_path):
        try:
            with h5py.File(paths.mip_path(storage_path, fov, hybe), 'r') as f:
                if 'fiducial_channel' not in f.attrs:
                    return None
                fid = str(int(f.attrs['fiducial_channel']))
                chans = [k[2:] for k in f.keys() if k.startswith('ch')]
                readout = [c for c in chans if c != fid]
                name = f'ch{readout[0]}' if readout else f'ch{fid}'
                return f[name][:] if name in f else None
        except OSError:
            return None
    vlinks_path = _vlinks_path(storage_path)
    if not os.path.exists(vlinks_path):
        return None
    grp_path = _mip_group_path(fov, modality_of(storage_path), hybe)
    with _open_vlinks(vlinks_path, 'r') as f:
        _require_yx(f, vlinks_path)
        if grp_path not in f or 'fiducial_channel' not in f[grp_path].attrs:
            return None
        fiducial_ch = int(f[grp_path].attrs['fiducial_channel'])
        channels = [name[2:] for name in f[grp_path].keys() if name.startswith('ch')]
        readout = [c for c in channels if c != str(fiducial_ch)]
        name = f'ch{readout[0]}' if readout else f'ch{fiducial_ch}'
        return f[grp_path][name][:] if name in f[grp_path] else None


def mip_channels_present(storage_path, fov, hybe):
    """
    {channel(str): True} for whatever channels this hybe's MIP actually has
    in vlinks.h5 right now, or None if this hybe was never ingested at all
    (no /FOV##/mip/{hybe} group whatsoever) -- lets a caller distinguish
    "never ingested" from "ingested but incomplete" (e.g. write_hybe_mip
    was interrupted partway through its channel loop), the same
    distinction windows/main_window.py's ingestion-status check already
    surfaces to the user (MISSING vs INCOMPLETE/UNREADABLE).

    v2 stores: the per-hybe MIP file is written atomically, so its
    existence is completeness -- one open only when the file exists.
    """
    if paths.is_v2(storage_path):
        try:
            with h5py.File(paths.mip_path(storage_path, fov, hybe), 'r') as f:
                return {k[2:]: True for k in f.keys() if k.startswith('ch')}
        except OSError:
            return None
    vlinks_path = _vlinks_path(storage_path)
    if not os.path.exists(vlinks_path):
        return None
    grp_path = _mip_group_path(fov, modality_of(storage_path), hybe)
    with _open_vlinks(vlinks_path, 'r') as f:
        _require_yx(f, vlinks_path)
        if grp_path not in f:
            return None
        return {name[2:]: True for name in f[grp_path].keys() if name.startswith('ch')}



def write_same_modality_matrices(storage_path, fov, matrices, reference_hybe):
    """
    Persists an already-computed {hybe: matrix} dict into vlinks.h5's
    /FOV##/matrix/{hybe} (real datasets, with reference_sequence/steps
    provenance attrs mirroring the per-hybe-stack-file convention this
    replaces) -- the vlinks-based counterpart to
    codelab_pipeline.alignment.chain's identically-named function, which
    now delegates here instead of writing into each hybe's own raw
    {hybe}_stack.h5. vlinks.h5 must be the authoritative store for this,
    not scattered across N raw per-hybe files, so that "has this FOV been
    aligned" is answerable -- cheaply, and correctly -- from vlinks.h5
    alone (matches how write_cross_modal_matrix below already mirrors the
    cross-modal case into vlinks.h5, for the same reason).
    """
    vlinks_path = _vlinks_path(storage_path)
    with _open_vlinks(vlinks_path, 'a') as f:
        _stamp_order(f)
        grp = f.require_group(_fov_matrix_group_path(fov, modality_of(storage_path)))
        for hybe, H in matrices.items():
            if hybe in grp:
                del grp[hybe]
            ds = grp.create_dataset(hybe, data=np.asarray(H, dtype='float32'))
            ds.attrs['reference_sequence'] = np.array([f'{hybe}->{reference_hybe}'], dtype='S')
            ds.attrs['steps'] = np.asarray(H, dtype='float32')[None, ...]


def read_same_modality_matrices(storage_path, fov, hybe_list):
    """
    Reads back whatever's in vlinks.h5's /FOV##/matrix/{hybe} for each hybe
    in hybe_list -- the vlinks-based counterpart to
    codelab_pipeline.alignment.chain's identically-named function.

    A hybe already ingested (real MIP present, see MainWindow.
    _ingested_hybes_for_fov)
    but with no matrix entry yet legitimately gets an identity default
    (write_hybe_mip seeds this at ingestion time, so in practice this only
    matters for a vlinks.h5 written before this seeding existed). A hybe
    not yet ingested at all is silently SKIPPED, never given a fake
    identity entry -- same "don't claim a non-existent hybe is processable"
    contract the old per-file-based version had (see that function's own
    docstring in chain.py for why this distinction matters downstream).
    """
    vlinks_path = _vlinks_path(storage_path)
    if not os.path.exists(vlinks_path):
        return {}
    from ..alignment.frames import FrameMatrices
    matrices = FrameMatrices(modality=modality_of(storage_path))
    with _open_vlinks(vlinks_path, 'r') as f:
        _require_yx(f, vlinks_path)
        modality = modality_of(storage_path)
        matrix_grp_path = _fov_matrix_group_path(fov, modality)
        # ingested-gate is layout-aware: v2 keeps MIPs as standalone
        # per-hybe files (paths.mips_present -- one directory listing),
        # not /mip groups inside this store. Gating on the in-store group
        # returned {} for every v2 hybe (confirmed on the migrated clone).
        ingested = paths.mips_present(storage_path, fov) if paths.is_v2(storage_path) else None
        for hybe in hybe_list:
            if (hybe not in ingested) if ingested is not None else                     (_mip_group_path(fov, modality, hybe) not in f):
                continue  # not ingested yet
            key = (hybe, modality)
            if matrix_grp_path in f and hybe in f[matrix_grp_path]:
                matrices[key] = f[matrix_grp_path][hybe][:]
            else:
                matrices[key] = np.eye(3)
    return matrices


def write_cross_modal_z(storage_path, fov, dz):
    """
    FOV-level cross-modal Z drift in PLANES, stored beside the 2D
    /matrix_across as its own /params/FOV##/z_across scalar.

    A separate key, NOT a reshape of matrix_across into 4x4: every consumer
    of that matrix (compose_chain, align_cell, hybe_to_cellref_matrix,
    matrix_anchors, the matrix viewer, the preview) assumes 3x3/2D, and z
    already lives in this codebase as an additive scalar channel alongside
    the affine (see ACell.matrices' own {'yx','zx'} split). Extending the
    affine would touch every one of those; adding a parallel scalar
    touches none. Old files simply have no z_across and read back as 0.
    """
    vlinks_path = _vlinks_path(storage_path)
    with _open_vlinks(vlinks_path, 'a') as f:
        _stamp_order(f)
        grp = f.require_group(_fov_params_group_path(fov))
        grp.attrs['z_across'] = float(dz)


def read_cross_modal_z(storage_path, fov):
    """Planes, DNA frame -> RNA frame. 0.0 when never written (see write_cross_modal_z)."""
    vlinks_path = _vlinks_path(storage_path)
    if not os.path.exists(vlinks_path):
        return 0.0
    grp_path = _fov_params_group_path(fov)
    with _open_vlinks(vlinks_path, 'r') as f:
        _require_yx(f, vlinks_path)
        if grp_path not in f:
            return 0.0
        return float(f[grp_path].attrs.get('z_across', 0.0))


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
    vlinks_path = _vlinks_path(storage_path)
    with _open_vlinks(vlinks_path, 'a') as f:
        _stamp_order(f)
        grp = f.require_group(_fov_params_group_path(fov))
        if 'matrix_across' in grp:
            del grp['matrix_across']
        grp.create_dataset('matrix_across', data=np.asarray(H, dtype='float64'))


def read_cross_modal_matrix(storage_path, fov):
    """The vlinks.h5-mirrored H_across for this FOV, or None if nothing's
    been written here yet (see write_cross_modal_matrix)."""
    vlinks_path = _vlinks_path(storage_path)
    if not os.path.exists(vlinks_path):
        return None
    grp_path = _fov_params_group_path(fov)
    with _open_vlinks(vlinks_path, 'r') as f:
        _require_yx(f, vlinks_path)
        if grp_path not in f or 'matrix_across' not in f[grp_path]:
            return None
        return f[grp_path]['matrix_across'][:]
