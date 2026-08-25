"""
Per-FOV analysis store -- the retirement of the single analysis/vlinks.h5.

One directory per FOV under <dp>/analysis/ (paths.analysis_fov_dir):

    <dp>/analysis/fov007/
        cells.h5                        columnar cells, full-replace
        alleles.h5                      columnar alleles, full-replace
        spots/{mod}__{hybe}__ch{c}.h5   columnar spots, one file per slice
        matrices__{modality}.h5         same-modality matrices, one dataset
                                        per hybe (merge-on-write)
        crossmodal.json                 per-bridging-modality H/z/quality
        manifest.json                   counts + aligned hybes + spot-uid
                                        counter: the human-readable flag
                                        file, so status panels and append
                                        planning read ONE tiny JSON per FOV
                                        instead of opening HDF5 at all
    <dp>/analysis/params.json           experiment params (shared +
                                        per-modality), human-readable
    <dp>/analysis/celltype_config.pkl   celltype setup (tuple-keyed)

Why this replaces the single file: vlinks.h5 was a global lock on a
per-FOV world. Every worker is FOV-major, but every read and write
funneled through one RLock into one HDF5 handle, so a backend writing
FOV 20 stalled a GUI read of FOV 3, and one interrupted write could
corrupt the whole experiment's analysis (HDF5 fails silently: a
truncated file opens happily and lies about its shape). Here every
write goes to its own small file via '<name>.part' + os.replace, so:

  - a reader NEVER sees a partial file (a crash leaves only .part,
    which nothing reads) -- the truncation hazard cannot occur;
  - there are no locks: concurrent analysis, ingestion, and GUI reads
    touch different files, and the one writer per (FOV, kind) at a time
    is already guaranteed by the FOV-major workers;
  - blast radius of any failure is one FOV's one kind of data.

os.replace over a file a reader currently holds open raises
PermissionError on Windows/SMB; writers retry briefly (_replace),
which is sufficient because every read here is open-read-close over a
small file.

The per-FOV manifest.json is DERIVED data, written after (never
before) the data file it describes, so a crash between the two leaves
it conservatively stale: a count lags, an aligned hybe is re-fit on the
next append pass -- false negatives only, never a claim about data that
is not there. tools/verify_store.py-style rebuilding lives in
_rebuild_manifest and runs automatically when a manifest is missing.

v1 (pre-manifest) stores: every public function delegates to the
frozen legacy module (vlinks_store) when storage_path is not a v2
project -- v1 stays readable, and is simply never created any more.
A v2 project that still carries analysis/vlinks.h5 refuses loudly and
names tools/migrate_vlinks.py, rather than silently reading an empty
new-layout store beside a full old one.

Model-facing shapes are unchanged: functions here accept and return
exactly what their vlinks_store namesakes did (ACell/ASpot/AnAllele
.save() dicts, FrameMatrices, plain params dicts), serialized through
the same columnar packers, so tests/test_columnar_roundtrip.py's
fidelity contract carries over verbatim.
"""
import collections
import contextlib
import json
import os
import pickle
import threading
import time
from datetime import datetime

import h5py
import numpy as np

from . import columnar
from . import paths
from . import vlinks_store as _legacy
from ..alignment.frames import FrameMatrices

# Layout-agnostic pieces the legacy module already owns (modality comes
# from the manifest for v2, from ingested stack attrs / declaration for
# v1) -- one registry, not two.
modality_of = _legacy.modality_of
declare_modality = _legacy.declare_modality

CROSS_MODAL_QUALITY_KEYS = _legacy.CROSS_MODAL_QUALITY_KEYS
MODALITY_SCOPED_PARAMS = _legacy.MODALITY_SCOPED_PARAMS


# -- atomic file primitives ----------------------------------------------

def _replace(tmp, target):
    """os.replace with a bounded retry: on Windows/SMB, replacing a file
    a reader briefly holds open raises PermissionError; reads here are
    open-read-close over small files, so waiting out the reader is both
    correct and short. Gives up loudly after ~5 s."""
    for attempt in range(20):
        try:
            os.replace(tmp, target)
            return
        except PermissionError:
            time.sleep(0.02 * (attempt + 1))
    os.replace(tmp, target)


def _tmp_name(target):
    """Per-thread .part name, so two threads full-replacing the SAME
    target never collide on one temp file -- each builds its own and the
    replaces land last-writer-wins, which is exactly the semantics a
    full-replace writer already promises. Still ends in '.part' so
    tools/verify_store.py's stray scan keeps catching leftovers."""
    return f'{target}.{threading.get_ident()}.part'


def _atomic_h5(target, build):
    """Write a capsule h5 file atomically: build(f) fills a fresh .part
    file, then one os.replace makes it visible whole."""
    os.makedirs(os.path.dirname(target), exist_ok=True)
    tmp = _tmp_name(target)
    with h5py.File(tmp, 'w') as f:
        f.attrs['coordinate_order'] = 'yx'
        f.attrs['saved_at'] = datetime.now().isoformat()
        build(f)
    _replace(tmp, target)


def _atomic_json(target, payload):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    tmp = _tmp_name(target)
    with open(tmp, 'w') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    _replace(tmp, target)


def _atomic_bytes(target, blob):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    tmp = _tmp_name(target)
    with open(tmp, 'wb') as f:
        f.write(blob)
    _replace(tmp, target)


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except OSError:
        return None


def _require_yx(f, path):
    order = f.attrs.get('coordinate_order')
    if order != 'yx':
        raise ValueError(
            f"{path} is not stamped coordinate_order='yx' (found {order!r}) "
            f"-- refusing to read swapped coordinates silently.")


# -- mtime-keyed read cache ----------------------------------------------
#
# Same contract as the legacy store's _mtime_cached: refreshes re-read
# unchanged files constantly (status panels, combo switches); a ~1 ms
# stat serves the cached result whenever mtime_ns is unchanged, and ANY
# write -- this process or another machine -- bumps mtime and
# invalidates naturally. Results are defensively copied / returned
# read-only so a caller cannot poison the cache.

_READ_CACHE = collections.OrderedDict()
_READ_CACHE_MAX = 512
# Readers run on the GUI thread AND on every worker QThread at once;
# OrderedDict reorder/evict is not safe under concurrent mutation.
_CACHE_LOCK = threading.Lock()


def _cache_copy(v):
    if isinstance(v, FrameMatrices):
        return FrameMatrices(v, modality=v.modality)
    if isinstance(v, np.ndarray):
        v = v.view()
        v.flags.writeable = False
        return v
    if isinstance(v, list):
        return [dict(d) if isinstance(d, dict) else d for d in v]
    if isinstance(v, tuple):
        return tuple(_cache_copy(x) for x in v)
    if isinstance(v, dict):
        return dict(v)
    return v


def _cached(path, extra, loader):
    """Serve loader() from the cache while `path`'s mtime is unchanged.
    `extra` distinguishes different reads of the same backing file. The
    loader itself runs outside the lock -- only cache bookkeeping is
    serialized, so a slow NAS read never blocks other threads' cache
    hits (two threads racing the same cold key just both load; the
    second store wins, harmlessly)."""
    try:
        mtime = os.stat(path).st_mtime_ns
    except OSError:
        return loader()
    key = (os.path.abspath(path), extra)
    with _CACHE_LOCK:
        hit = _READ_CACHE.get(key)
        if hit is not None and hit[0] == mtime:
            _READ_CACHE.move_to_end(key)
            return _cache_copy(hit[1])
    result = loader()
    with _CACHE_LOCK:
        _READ_CACHE[key] = (mtime, result)
        _READ_CACHE.move_to_end(key)
        while len(_READ_CACHE) > _READ_CACHE_MAX:
            _READ_CACHE.popitem(last=False)
    return _cache_copy(result)


# -- layout routing and the migration gate -------------------------------

_MIGRATION_CHECKED = set()


def _analysis_root(storage_path):
    """<dp>/analysis, after refusing an unmigrated store: a v2 project
    still carrying analysis/vlinks.h5 (or a half-done .migrating) has its
    truth in the OLD layout, and reading the new one would silently
    report an empty experiment beside a full one."""
    root = paths.analysis_dir(storage_path)
    dp = os.path.dirname(root)
    if dp not in _MIGRATION_CHECKED:
        if os.path.exists(os.path.join(root, 'vlinks.h5.migrating')):
            raise RuntimeError(
                f'{root} holds an interrupted vlinks.h5 migration '
                f'(vlinks.h5.migrating) -- re-run tools/migrate_vlinks.py '
                f'on this project before using it.')
        if os.path.exists(os.path.join(root, 'vlinks.h5')):
            raise RuntimeError(
                f'{root} still holds the retired single-file store '
                f'(vlinks.h5) -- run tools/migrate_vlinks.py once on this '
                f'project to convert it to the per-FOV layout.')
        _MIGRATION_CHECKED.add(dp)
    return root


def _fov_dir(storage_path, fov):
    _analysis_root(storage_path)
    return paths.analysis_fov_dir(storage_path, int(fov))


def _routed(fn):
    """v2 -> this module's implementation; anything else -> the frozen
    legacy vlinks_store function of the same name (v1 readability is a
    standing commitment)."""
    legacy_fn = getattr(_legacy, fn.__name__)

    def wrapped(storage_path, *args, **kwargs):
        if paths.is_v2(storage_path):
            return fn(storage_path, *args, **kwargs)
        return legacy_fn(storage_path, *args, **kwargs)

    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    wrapped.__wrapped__ = fn
    return wrapped


def distinct_stores(storage_paths):
    """One representative storage_path per distinct physical analysis
    store (v2: the project root; v1: the resolved vlinks.h5) -- the
    mirror_* dedup rule, same contract as the legacy function."""
    seen, out = set(), []
    for path in storage_paths:
        if not path:
            continue
        if paths.is_v2(path):
            resolved = ('v2', os.path.abspath(paths.project_root(path)))
        else:
            resolved = ('v1', os.path.abspath(paths.vlinks_path(path)))
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(path)
    return out


# -- per-FOV manifest (the flag file) ------------------------------------

_MANIFEST_LOCK = threading.Lock()

_EMPTY_MANIFEST = {'cells': {'n': 0}, 'alleles': {'n': 0}, 'spots': {},
                   'matrices': {}, 'next_uid': 1, 'highest_uid_seen': 0}


def _manifest_path(fov_dir):
    return os.path.join(fov_dir, 'manifest.json')


def _read_fov_manifest(fov_dir, fresh=False):
    """The FOV's manifest dict (a fresh default when the FOV has no
    analysis at all). A manifest missing UNDER existing capsule files is
    rebuilt from the files once -- self-healing, since the manifest is
    derived data.

    fresh=True bypasses the mtime cache and reads the file itself --
    REQUIRED inside any read-modify-write: Windows file-time granularity
    is coarser than the write rate, so two writes can land within one
    timestamp tick and the cache would serve the pre-first-write state
    to the second writer, silently dropping the first's update
    (confirmed real: duplicate spot uids under 4-thread allocation in
    the concurrency stress). The cache stays correct for PURE readers
    because _refresh_cached_manifest pokes every fresh write into it.
    """
    if fresh:
        m = _read_json(_manifest_path(fov_dir))
    else:
        m = _cached(_manifest_path(fov_dir), 'manifest',
                    lambda: _read_json(_manifest_path(fov_dir)))
    if m is not None:
        return m
    if os.path.isdir(fov_dir) and any(
            n != 'manifest.json' and not n.endswith('.part')
            for n in os.listdir(fov_dir)):
        return _rebuild_manifest(fov_dir)
    return json.loads(json.dumps(_EMPTY_MANIFEST))


def _poke_cache(path, extra, value):
    """Poke a just-written value into the read cache, so a pure reader
    arriving within the same file-time tick still sees the new value
    instead of a stale cache hit (the write-side half of the fresh=True
    rule -- see _read_fov_manifest)."""
    try:
        mtime = os.stat(path).st_mtime_ns
    except OSError:
        return
    with _CACHE_LOCK:
        _READ_CACHE[(os.path.abspath(path), extra)] = (mtime, value)


def _refresh_cached_manifest(path, m):
    _poke_cache(path, 'manifest', json.loads(json.dumps(m)))


def _update_fov_manifest(fov_dir, mutate):
    """Read-modify-write the manifest atomically under the module lock.
    Call AFTER the data write it describes, so a crash between the two
    leaves the manifest conservatively stale, never ahead of the data.
    Reads FRESH (never the mtime cache -- see _read_fov_manifest) and
    refreshes the cache after writing."""
    with _MANIFEST_LOCK:
        path = _manifest_path(fov_dir)
        m = _read_fov_manifest(fov_dir, fresh=True)
        mutate(m)
        _atomic_json(path, m)
        _refresh_cached_manifest(path, m)


def _rebuild_manifest(fov_dir):
    """Reconstruct a lost manifest from the capsule files themselves --
    counts from each file's own attrs, aligned hybes from the matrices
    files' keys, the uid floor from each spot slice's max_uid."""
    m = json.loads(json.dumps(_EMPTY_MANIFEST))
    highest = 0
    for name, kind in (('cells.h5', 'cells'), ('alleles.h5', 'alleles')):
        p = os.path.join(fov_dir, name)
        if os.path.exists(p):
            try:
                with h5py.File(p, 'r') as f:
                    m[kind] = {'n': int(f.attrs.get(f'n_{kind}', 0)),
                               'saved_at': str(f.attrs.get('saved_at', ''))}
            except OSError:
                pass
    spots_dir = os.path.join(fov_dir, 'spots')
    if os.path.isdir(spots_dir):
        for name in os.listdir(spots_dir):
            if not name.endswith('.h5'):
                continue
            try:
                with h5py.File(os.path.join(spots_dir, name), 'r') as f:
                    m['spots'][name[:-3]] = int(f.attrs.get('n_spots', 0))
                    highest = max(highest, int(f.attrs.get('max_uid', 0)))
            except OSError:
                continue
    for name in os.listdir(fov_dir) if os.path.isdir(fov_dir) else []:
        if name.startswith('matrices__') and name.endswith('.h5'):
            try:
                with h5py.File(os.path.join(fov_dir, name), 'r') as f:
                    m['matrices'][name[len('matrices__'):-3]] = sorted(f.keys())
            except OSError:
                continue
    m['highest_uid_seen'] = highest
    m['next_uid'] = highest + 1
    _atomic_json(_manifest_path(fov_dir), m)
    return m


# -- spots ---------------------------------------------------------------

def _slice_name(modality, hybe, channel):
    modality, hybe = str(modality), str(hybe)
    for part in (modality, hybe):
        if '__' in part:
            raise ValueError(
                f'{part!r} contains "__", the slice-filename separator -- '
                f'modality and hybe names must not.')
    return f'{modality}__{hybe}__ch{int(channel)}'


def _parse_slice_name(stem):
    """'{mod}__{hybe}__ch{c}' -> (modality, hybe, channel) or None."""
    left, sep, ch = stem.rpartition('__ch')
    if not sep:
        return None
    mod, sep, hybe = left.partition('__')
    if not sep:
        return None
    try:
        return mod, hybe, int(ch)
    except ValueError:
        return None


def _spots_dir(storage_path, fov):
    return os.path.join(_fov_dir(storage_path, fov), 'spots')


@_routed
def allocate_spot_uids(storage_path, fov, count):
    """`count` fresh, never-before-used spot uids for this FOV.

    The per-FOV monotonic counter lives in the FOV's manifest.json; on
    every call it is floored above the highest uid ever seen, so a
    counter lost with its manifest (rebuilt from the slice files'
    max_uid attrs) can never re-issue a uid already in use -- identity
    (save-merge, undo diffs, staleness marks) depends on that.
    """
    fov_dir = _fov_dir(storage_path, fov)
    out = []

    def bump(m):
        start = max(int(m.get('next_uid', 1)),
                    int(m.get('highest_uid_seen', 0)) + 1, 1)
        m['next_uid'] = start + int(count)
        m['highest_uid_seen'] = start + int(count) - 1
        out.extend(range(start, start + int(count)))

    _update_fov_manifest(fov_dir, bump)
    return out


def write_spot_dicts(storage_path, fov, modality, hybe, channel, payload):
    """Full atomic replace of ONE (modality, hybe, channel) slice file
    with already-.save()d spot dicts (the migration tool's entry point;
    write_spots below is the object-level door)."""
    seen = {}
    for d in payload:
        if d['uid'] in seen:
            raise ValueError(
                f'duplicate spot uid {d["uid"]} in FOV{int(fov):03d} '
                f'{modality}/{hybe}/ch{channel} -- uid must identify one spot')
        seen[d['uid']] = True
    name = _slice_name(modality, hybe, channel)
    target = os.path.join(_spots_dir(storage_path, fov), name + '.h5')

    def build(f):
        columnar.pack_spots(f, payload)
        f.attrs['n_spots'] = len(payload)
        f.attrs['max_uid'] = max((int(d['uid']) for d in payload), default=0)

    _atomic_h5(target, build)
    _update_fov_manifest(
        _fov_dir(storage_path, fov),
        lambda m: m.setdefault('spots', {}).__setitem__(name, len(payload)))


@_routed
def write_spots(storage_path, fov, modality, hybe, channel, spots):
    """
    Full replace of ONE (modality, hybe, channel) slice with `spots` --
    its own small file, so deletions propagate within the slice and
    nothing outside the slice can be touched, and a hundred slices can
    be written by a hundred workers with zero contention. Any spot still
    carrying uid 0 is allocated one here.
    """
    spots = list(spots)
    unallocated = [sp for sp in spots if not getattr(sp, 'uid', 0)]
    if unallocated:
        for sp, uid in zip(unallocated,
                           allocate_spot_uids(storage_path, fov, len(unallocated))):
            sp.uid = uid
    write_spot_dicts(storage_path, fov, modality, hybe, channel,
                     [sp.save() for sp in spots])


def _read_slice_file(path):
    def load():
        try:
            with h5py.File(path, 'r') as f:
                _require_yx(f, path)
                return columnar.unpack_spots(f)
        except OSError:
            return []
    return _cached(path, 'spots', load)


@_routed
def read_spots(storage_path, fov, modality=None, hybe=None, channel=None):
    """
    ASpot.save()-shaped dicts. With modality/hybe/channel given, just
    that slice (one small file); with none, every spot in the FOV (one
    listdir + one read per slice file). Assigned and unassigned come
    back together -- filter on 'cell' (-1 = unassigned).
    """
    spots_dir = _spots_dir(storage_path, fov)
    if modality is not None and hybe is not None and channel is not None:
        return _read_slice_file(
            os.path.join(spots_dir, _slice_name(modality, hybe, channel) + '.h5'))
    out = []
    try:
        names = sorted(os.listdir(spots_dir))
    except OSError:
        return out
    for name in names:
        if name.endswith('.h5'):
            out.extend(_read_slice_file(os.path.join(spots_dir, name)))
    return out


@_routed
def spot_slices(storage_path, fov):
    """[(modality, hybe, channel), ...] of every spot slice this FOV
    holds -- ONE directory listing, no file opened."""
    out = []
    try:
        names = sorted(os.listdir(_spots_dir(storage_path, fov)))
    except OSError:
        return out
    for name in names:
        if name.endswith('.h5'):
            parsed = _parse_slice_name(name[:-3])
            if parsed:
                out.append(parsed)
    return out


# -- cells ---------------------------------------------------------------

def write_cell_dicts(storage_path, fov, dicts):
    """Full atomic replace of the FOV's cells capsule with already-
    .save()d cell dicts."""
    target = os.path.join(_fov_dir(storage_path, fov), 'cells.h5')

    def build(f):
        columnar.pack_cells(f, dicts)
        f.attrs['n_cells'] = len(dicts)

    _atomic_h5(target, build)
    _update_fov_manifest(
        _fov_dir(storage_path, fov),
        lambda m: m.__setitem__('cells', {'n': len(dicts),
                                          'saved_at': datetime.now().isoformat()}))


@_routed
def write_cells(storage_path, fov, cell_container):
    """One FOV's cells (with their per-hybe matrices; spots live in the
    FOV's own slice files) into the FOV's cells.h5 capsule."""
    write_cell_dicts(storage_path, fov,
                     [cell.save() for cell in cell_container.get_cells(fov)])


@_routed
def read_cells(storage_path, fov):
    """
    (cell_dicts, '') -- ACell.save()-shaped dicts for
    CellContainer.load, or (None, '') if nothing persisted for this FOV
    yet. (The second element is the legacy modality slot, always ''.)
    """
    path = os.path.join(_fov_dir(storage_path, fov), 'cells.h5')

    def load():
        try:
            with h5py.File(path, 'r') as f:
                _require_yx(f, path)
                return columnar.unpack_cells(f), ''
        except OSError:
            return None, ''
    return _cached(path, 'cells', load)


def mirror_write_cells(storage_paths, fov, cell_container):
    """write_cells into every DISTINCT analysis store among the paths --
    one write for a normal single-project session."""
    for path in distinct_stores(storage_paths):
        write_cells(path, fov, cell_container)


# -- alleles -------------------------------------------------------------

def write_allele_dicts(storage_path, fov, payload):
    target = os.path.join(_fov_dir(storage_path, fov), 'alleles.h5')

    def build(f):
        columnar.pack_alleles(f, payload)
        f.attrs['n_alleles'] = len(payload)

    _atomic_h5(target, build)
    _update_fov_manifest(
        _fov_dir(storage_path, fov),
        lambda m: m.__setitem__('alleles', {'n': len(payload),
                                            'saved_at': datetime.now().isoformat()}))


@_routed
def write_fov_alleles(storage_path, fov, alleles):
    """One FOV's chromatin-tracing alleles (AnAllele objects), full
    replace of the FOV's alleles.h5 capsule."""
    write_allele_dicts(storage_path, fov, [a.save() for a in alleles])


@_routed
def read_fov_alleles(storage_path, fov):
    """AnAllele.save()-shaped dicts, or [] if none persisted yet."""
    path = os.path.join(_fov_dir(storage_path, fov), 'alleles.h5')

    def load():
        try:
            with h5py.File(path, 'r') as f:
                _require_yx(f, path)
                return columnar.unpack_alleles(f)
        except OSError:
            return []
    return _cached(path, 'alleles', load)


def mirror_write_fov_alleles(storage_paths, fov, alleles):
    for path in distinct_stores(storage_paths):
        write_fov_alleles(path, fov, alleles)


# -- counts (status panels) ----------------------------------------------

@_routed
def fov_counts(storage_path, fovs):
    """{fov: {'cells': n, 'spots': n, 'alleles': n}} for many FOVs --
    one manifest.json read per FOV, no HDF5 opened."""
    out = {}
    for fov in fovs:
        m = _read_fov_manifest(_fov_dir(storage_path, fov))
        out[int(fov)] = {
            'cells': int(m.get('cells', {}).get('n', 0)),
            'spots': int(sum(m.get('spots', {}).values())),
            'alleles': int(m.get('alleles', {}).get('n', 0)),
        }
    return out


# -- same-modality matrices ----------------------------------------------

# The one read-MERGE-write in this store: AlignmentWorker (a QThread)
# writes each FOV's matrices as it finishes, and the GUI thread's
# manual-accept door writes the same file shape -- two concurrent
# merges without this lock could each read the same base and drop the
# other's hybes on rewrite. Every other writer here is a pure full
# replace (last-writer-wins is already its contract) or is guarded by
# its own RMW lock (manifest, crossmodal, params).
_MATRICES_LOCK = threading.Lock()


def _matrices_path(storage_path, fov, modality):
    return os.path.join(_fov_dir(storage_path, fov), f'matrices__{modality}.h5')


def _read_matrices_raw(path, fresh=False):
    """{hybe: {'H', 'reference_sequence', 'steps'}} straight off one
    matrices file, or {}. fresh=True bypasses the mtime cache -- required
    inside the merge RMW, same file-time-granularity hazard as
    _read_fov_manifest(fresh=True)."""
    def load():
        out = {}
        try:
            with h5py.File(path, 'r') as f:
                _require_yx(f, path)
                for hybe in f:
                    ds = f[hybe]
                    out[hybe] = {
                        'H': ds[:],
                        'reference_sequence': ds.attrs.get('reference_sequence'),
                        'steps': (np.asarray(ds.attrs['steps'])
                                  if 'steps' in ds.attrs else None),
                    }
        except OSError:
            return {}
        return out
    if fresh:
        return load()
    return _cached(path, 'matrices', load)


@_routed
def write_same_modality_matrices(storage_path, fov, matrices, reference_hybe):
    """
    Merge an already-computed {hybe: matrix} dict into this (FOV,
    modality)'s matrices file -- existing hybes not in `matrices` are
    preserved (append runs write only what they fit), each entry carries
    reference_sequence/steps provenance, and the whole file is rewritten
    atomically (it is ~KB even at a hundred hybes).
    """
    modality = modality_of(storage_path)
    target = _matrices_path(storage_path, fov, modality)
    with _MATRICES_LOCK:
        merged = {h: dict(v) for h, v in _read_matrices_raw(target, fresh=True).items()}
        for hybe, H in matrices.items():
            H = np.asarray(H, dtype='float32')
            merged[hybe] = {
                'H': H,
                'reference_sequence': np.array([f'{hybe}->{reference_hybe}'], dtype='S'),
                'steps': H[None, ...],
            }

        def build(f):
            for hybe, entry in merged.items():
                ds = f.create_dataset(hybe, data=np.asarray(entry['H'], dtype='float32'))
                if entry.get('reference_sequence') is not None:
                    ds.attrs['reference_sequence'] = entry['reference_sequence']
                if entry.get('steps') is not None:
                    ds.attrs['steps'] = np.asarray(entry['steps'], dtype='float32')

        _atomic_h5(target, build)
        _update_fov_manifest(
            _fov_dir(storage_path, fov),
            lambda m: m.setdefault('matrices', {}).__setitem__(
                modality, sorted(merged.keys())))


@_routed
def read_same_modality_matrices(storage_path, fov, hybe_list):
    """
    FrameMatrices keyed (hybe, modality) for each hybe in hybe_list. An
    ingested hybe with no persisted matrix defaults to identity ("not
    aligned yet" is identity, never an error); a hybe not ingested at
    all is silently skipped, never given a fake entry.
    """
    modality = modality_of(storage_path)
    stored = _read_matrices_raw(_matrices_path(storage_path, fov, modality))
    ingested = paths.mips_present(storage_path, fov)
    matrices = FrameMatrices(modality=modality)
    for hybe in hybe_list:
        if hybe not in ingested:
            continue
        entry = stored.get(hybe)
        # np.array (copy), not asarray: `stored` inner arrays are shared
        # with the read cache and must not be mutable through the result.
        matrices[(hybe, modality)] = (np.array(entry['H'], dtype='float64')
                                      if entry is not None else np.eye(3))
    return matrices


@_routed
def aligned_hybes(storage_path, fov):
    """The set of hybe names with a REAL persisted matrix for this
    (modality, FOV) -- the append-mode primitive, answered from the
    FOV's manifest (which only ever lags a write, so append re-fits
    rather than skips on any drift)."""
    modality = modality_of(storage_path)
    m = _read_fov_manifest(_fov_dir(storage_path, fov))
    return frozenset(m.get('matrices', {}).get(modality, []))


# -- cross-modal results -------------------------------------------------
#
# crossmodal.json: {bridging_modality: {"matrix": [[...]x3], "z": float,
# "quality": {...}}}. "_" holds a value migrated from the legacy flat
# (pre-star) keys, and is the read fallback -- same contract as the
# legacy store's flat-attr fallback.

_CROSSMODAL_LOCK = threading.Lock()


def _crossmodal_path(storage_path, fov):
    return os.path.join(_fov_dir(storage_path, fov), 'crossmodal.json')


def _crossmodal_read(storage_path, fov):
    path = _crossmodal_path(storage_path, fov)
    return _cached(path, 'crossmodal', lambda: _read_json(path)) or {}


def _crossmodal_update(storage_path, fov, modality, key, value):
    with _CROSSMODAL_LOCK:
        path = _crossmodal_path(storage_path, fov)
        data = _read_json(path) or {}
        data.setdefault(modality or '_', {})[key] = value
        _atomic_json(path, data)
        _poke_cache(path, 'crossmodal', json.loads(json.dumps(data)))


@_routed
def write_cross_modal_matrix(storage_path, fov, H, modality=None):
    """The accepted H_across for one FOV's bridge, keyed by the bridging
    modality (star topology -- two bridges never overwrite each other)."""
    _crossmodal_update(storage_path, fov, modality, 'matrix',
                       np.asarray(H, dtype='float64').tolist())


@_routed
def read_cross_modal_matrix(storage_path, fov, modality=None):
    """The persisted H_across (ndarray) or None. modality-keyed entry
    first, migrated-flat '_' fallback."""
    data = _crossmodal_read(storage_path, fov)
    for key in ((modality, '_') if modality is not None else ('_',)):
        entry = data.get(key, {})
        if 'matrix' in entry:
            return np.asarray(entry['matrix'], dtype='float64')
    return None


@_routed
def write_cross_modal_z(storage_path, fov, dz, modality=None):
    """FOV-level cross-modal Z drift in PLANES, beside the 2D matrix as
    its own scalar (z is an additive channel alongside the affine
    everywhere in this codebase, never a 4x4 reshape)."""
    _crossmodal_update(storage_path, fov, modality, 'z', float(dz))


@_routed
def read_cross_modal_z(storage_path, fov, modality=None):
    """Planes, bridging frame -> shared frame; 0.0 when never written."""
    data = _crossmodal_read(storage_path, fov)
    for key in ((modality, '_') if modality is not None else ('_',)):
        entry = data.get(key, {})
        if 'z' in entry:
            return float(entry['z'])
    return 0.0


@_routed
def write_cross_modal_quality(storage_path, fov, quality, modality=None):
    """The measured fit quality of one FOV's cross-modal result --
    whichever CROSS_MODAL_QUALITY_KEYS `quality` carries."""
    if not quality:
        return
    payload = {k: float(quality[k]) for k in CROSS_MODAL_QUALITY_KEYS
               if quality.get(k) is not None}
    if payload:
        _crossmodal_update(storage_path, fov, modality, 'quality', payload)


@_routed
def read_cross_modal_quality(storage_path, fov, modality=None):
    """{key: float} of the persisted quality components, or {} -- never
    fabricated."""
    data = _crossmodal_read(storage_path, fov)
    for key in ((modality, '_') if modality is not None else ('_',)):
        entry = data.get(key, {})
        if 'quality' in entry:
            return {k: float(v) for k, v in entry['quality'].items()}
    return {}


# -- experiment params ---------------------------------------------------

_PARAMS_LOCK = threading.Lock()


def _params_path(storage_path):
    return os.path.join(_analysis_root(storage_path), 'params.json')


def _jsonable(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, bytes):
        return v.decode()
    return v


@_routed
def write_global_params(storage_path, **params):
    """
    Whole-experiment metadata as human-readable JSON at
    analysis/params.json: {"shared": {...}, "modalities": {name: {...}}}.
    Merges -- only the keys actually passed are overwritten (None values
    skipped); keys in MODALITY_SCOPED_PARAMS route to this
    storage_path's modality section so two modalities never clobber
    each other's facts.
    """
    modality = None
    with _PARAMS_LOCK:
        path = _params_path(storage_path)
        data = _read_json(path) or {'shared': {}, 'modalities': {}}
        for k, v in params.items():
            if v is None:
                continue
            if k in MODALITY_SCOPED_PARAMS:
                if modality is None:
                    modality = modality_of(storage_path)
                data.setdefault('modalities', {}).setdefault(modality, {})[k] = _jsonable(v)
            else:
                data.setdefault('shared', {})[k] = _jsonable(v)
        _atomic_json(path, data)
        _poke_cache(path, 'params', json.loads(json.dumps(data)))


@_routed
def read_global_params(storage_path):
    """{key: value} -- the shared section merged with this
    storage_path's own modality section, or {}."""
    path = _params_path(storage_path)

    def load():
        return _read_json(path)
    data = _cached(path, 'params', load)
    if not data:
        return {}
    out = dict(data.get('shared', {}))
    try:
        modality = modality_of(storage_path)
    except ValueError:
        modality = None
    if modality:
        out.update(data.get('modalities', {}).get(modality, {}))
    return out


# -- celltype config -----------------------------------------------------

def _celltype_config_path(storage_path):
    return os.path.join(_analysis_root(storage_path), 'celltype_config.pkl')


@_routed
def write_celltype_config(storage_path, fov_ranges_by_celltype,
                          barcode_channel_by_celltype, calibration,
                          barcode_method=None):
    """Celltype Determination's entire setup (FOV ranges, (hybe,
    channel, modality) barcode assignments, computed calibration,
    method), pickled whole -- tuple keys and per-FOV float maps are not
    JSON-shaped, and fidelity beats readability for a config the panel
    itself re-renders."""
    payload = {
        'fov_ranges_by_celltype': dict(fov_ranges_by_celltype),
        'barcode_channel_by_celltype': dict(barcode_channel_by_celltype),
        'calibration': calibration,
        'barcode_method': barcode_method,
    }
    _atomic_bytes(_celltype_config_path(storage_path), pickle.dumps(payload))


@_routed
def read_celltype_config(storage_path):
    """(fov_ranges, barcode_channels, calibration, method) or the empty
    shape if nothing persisted yet."""
    empty_calibration = {'scale': {}, 'lower_bound': {}, 'upper_bound': {}}
    path = _celltype_config_path(storage_path)

    def load():
        try:
            with open(path, 'rb') as f:
                return pickle.loads(f.read())
        except OSError:
            return None
    payload = _cached(path, 'celltype_config', load)
    if not payload:
        return {}, {}, empty_calibration, None
    return (payload.get('fov_ranges_by_celltype', {}),
            payload.get('barcode_channel_by_celltype', {}),
            payload.get('calibration', empty_calibration),
            payload.get('barcode_method'))


def mirror_write_celltype_config(storage_paths, fov_ranges_by_celltype,
                                 barcode_channel_by_celltype, calibration,
                                 barcode_method=None):
    for path in distinct_stores(storage_paths):
        write_celltype_config(path, fov_ranges_by_celltype,
                              barcode_channel_by_celltype, calibration,
                              barcode_method)


# -- MIPs (v2: standalone per-hybe files under <modality>/mips/) ---------

@_routed
def write_hybe_mip(storage_path, fov, hybe, channel_mips, fiducial_channel=None):
    """One small standalone MIP file per (modality, FOV, hybe), written
    atomically by the ingestion worker itself -- existence ==
    completeness, no shared-file contention (v1 delegates to the legacy
    in-vlinks copy)."""
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
    _replace(tmp, target)


@_routed
def read_hybe_mip(storage_path, fov, hybe, channel, window=None):
    """One hybe/channel's stored MIP (or None). window=(ymin, ymax,
    xmin, xmax) reads only the covering chunks -- measured 58x cheaper
    than inflating the full gzip-chunked frame for a cell-sized crop."""
    try:
        with h5py.File(paths.mip_path(storage_path, fov, hybe), 'r') as f:
            name = f'ch{channel}'
            if name not in f:
                return None
            if window is None:
                return f[name][:]
            ymin, ymax, xmin, xmax = window
            return f[name][ymin:ymax, xmin:xmax]
    except OSError:
        return None


@_routed
def fiducial_channel_mip(storage_path, fov, hybe):
    """The fiducial channel's MIP for a hybe, resolved from the MIP
    file's own fiducial_channel attr; None if not ingested/stamped."""
    try:
        with h5py.File(paths.mip_path(storage_path, fov, hybe), 'r') as f:
            if 'fiducial_channel' not in f.attrs:
                return None
            name = f"ch{int(f.attrs['fiducial_channel'])}"
            return f[name][:] if name in f else None
    except OSError:
        return None


@_routed
def readout_channel_mip(storage_path, fov, hybe):
    """The one non-fiducial channel's MIP for a hybe (falls back to the
    fiducial when the hybe genuinely has no other channel); None if not
    ingested/stamped."""
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


@_routed
def mip_channels_present(storage_path, fov, hybe):
    """{channel(str): True} for the channels this hybe's MIP holds, or
    None if never ingested. The MIP file is written atomically, so its
    existence is completeness -- one open only when it exists."""
    try:
        with h5py.File(paths.mip_path(storage_path, fov, hybe), 'r') as f:
            return {k[2:]: True for k in f.keys() if k.startswith('ch')}
    except OSError:
        return None
