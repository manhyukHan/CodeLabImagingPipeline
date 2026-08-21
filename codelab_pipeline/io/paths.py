"""
THE storage-layout resolver: every file path in a project goes through
here, so the on-disk layout is a versioned, swappable fact instead of a
convention smeared across call sites.

Two layouts exist:

v1 (legacy)  storage_path IS a per-modality queue directory:
               storage_path/FOV##/{hybe}_stack.h5
               <parent>/vlinks.h5            (stacks + MIPs + analysis)
               storage_path/FOV##/*.png      (figures, dumped beside data)

v2           storage_path = <dp>/{modality} inside a project root <dp>
             carrying a manifest.json (see write_manifest):
               <dp>/{modality}/stacks/FOV##/{hybe}.h5   chunked, write-once
               <dp>/{modality}/mips/FOV##/{hybe}.h5     per-hybe, written
                                                        ATOMICALLY by the
                                                        ingestion worker
                                                        (existence == complete)
               <dp>/analysis/vlinks.h5                  cells/spots/alleles/
                                                        matrices/params only
               <dp>/figures/{category}/FOV##/*.png      all rendered outputs

Why modality-major (dp/{modality}/...) rather than FOV-major: the whole
pipeline keys modality by storage path (modality_of and its ~20 callers;
every stack/crop/segment signature takes a modality-scoped
storage_path). With storage_path = dp/{modality} every existing
signature keeps working and the cross-modal bridge hybe can never
become ambiguous through a path. Per explicit decision the order was
left free ("or change modality-FOV order").

Why per-hybe MIP files: they are written by the ingestion WORKER itself
(each file has exactly one writer ever -- no shared-file contention, no
coordinator bottleneck), and a hybe becomes visible to the UI the moment
its own file lands. Writers write '<name>.part' then os.replace, so a
plain existence check IS the completeness check -- the ingestion
checkup becomes one directory listing instead of thousands of vlinks
opens.

Detection is by manifest: a storage_path whose PARENT holds
manifest.json is v2. Everything else is v1. No behavior of v1 stores
changes.
"""
import json
import os

MANIFEST_NAME = 'manifest.json'
LAYOUT_VERSION = 2

_MANIFEST_CACHE = {}   # {dp: (mtime_ns, dict)}


def project_root(storage_path):
    """The <dp> a storage_path belongs to: its parent directory."""
    return os.path.dirname(os.path.abspath(storage_path).rstrip(os.sep))


def manifest_path(dp):
    return os.path.join(dp, MANIFEST_NAME)


def read_manifest(dp):
    """The project manifest dict, or None -- mtime-cached."""
    p = manifest_path(dp)
    try:
        mtime = os.stat(p).st_mtime_ns
    except OSError:
        return None
    hit = _MANIFEST_CACHE.get(dp)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    with open(p) as f:
        m = json.load(f)
    _MANIFEST_CACHE[dp] = (mtime, m)
    return m


def write_manifest(dp, modalities, layout_paths=None, dax_directories=None):
    """
    modalities: [name, ...]; layout_paths/dax_directories: optional
    {name: path}. Overwrites atomically. The manifest is the project's
    modality registry -- readable with no HDF5 at all, which is what
    retires the modality_of bootstrapping problem for v2 stores.
    """
    m = {'layout_version': LAYOUT_VERSION,
         'modalities': {name: {'layout_path': (layout_paths or {}).get(name, ''),
                               'dax_directory': (dax_directories or {}).get(name, '')}
                        for name in modalities}}
    os.makedirs(dp, exist_ok=True)
    tmp = manifest_path(dp) + '.part'
    with open(tmp, 'w') as f:
        json.dump(m, f, indent=2)
    os.replace(tmp, manifest_path(dp))
    _MANIFEST_CACHE.pop(dp, None)
    return m


def is_v2(storage_path):
    return read_manifest(project_root(storage_path)) is not None


def looks_like_v1_store(storage_path):
    """
    True when storage_path ALREADY holds v1 data (FOV##/{hybe}_stack.h5).

    Exists to tell an existing v1 store apart from a merely empty path, so
    that new work can default to v2 without touching old datasets. v1 stays
    readable for as long as anyone has one; it is simply never created any
    more. Nothing here converts a store -- that is
    tools/migrate_store_v2.py's job, deliberately explicit and one-shot.

    Cheap on purpose: one listdir of the store plus one per FOV directory
    until the first hit, and it stops at the first _stack.h5 it sees. It is
    called from Parse Layout, not from a loop.
    """
    if not os.path.isdir(storage_path):
        return False
    try:
        entries = sorted(os.listdir(storage_path))
    except OSError:
        return False
    for name in entries:
        if not name.startswith('FOV'):
            continue
        fov_dir = os.path.join(storage_path, name)
        if not os.path.isdir(fov_dir):
            continue
        try:
            if any(f.endswith('_stack.h5') for f in os.listdir(fov_dir)):
                return True
        except OSError:
            continue
    return False


def modality_from_path(storage_path):
    """v2 only: the modality a storage_path names, validated against the
    manifest; None when not a v2 store or not a declared modality."""
    m = read_manifest(project_root(storage_path))
    if m is None:
        return None
    name = os.path.basename(os.path.abspath(storage_path).rstrip(os.sep))
    return name if name in m.get('modalities', {}) else None


# -- file resolution -----------------------------------------------------

def stack_path(storage_path, fov, hybe):
    if is_v2(storage_path):
        return os.path.join(storage_path, 'stacks', f'FOV{fov:02d}', f'{hybe}.h5')
    return os.path.join(storage_path, f'FOV{fov:02d}', f'{hybe}_stack.h5')


def mip_path(storage_path, fov, hybe):
    """v2 only -- v1 keeps MIPs inside vlinks.h5 (callers branch on is_v2)."""
    return os.path.join(storage_path, 'mips', f'FOV{fov:02d}', f'{hybe}.h5')


def mips_dir(storage_path, fov):
    return os.path.join(storage_path, 'mips', f'FOV{fov:02d}')


def vlinks_path(storage_path):
    dp = project_root(storage_path)
    if read_manifest(dp) is not None:
        return os.path.join(dp, 'analysis', 'vlinks.h5')
    return os.path.join(dp, 'vlinks.h5')


def figure_path(storage_path, category, fov, filename):
    """
    All rendered outputs live under <dp>/figures/{category}/FOV##/ in a
    v2 store (per explicit decision: theoretically every analysis output
    is stored, so categories keep them distinguishable -- 'alignment',
    'cells', 'celltype', 'chromatin', ...). v1 keeps the legacy
    dump-beside-the-data location.
    """
    if is_v2(storage_path):
        d = os.path.join(project_root(storage_path), 'figures', category, f'FOV{fov:02d}')
    else:
        d = os.path.join(storage_path, f'FOV{fov:02d}')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, filename)


def mips_present(storage_path, fov):
    """
    v2 ingestion checkup primitive: the set of hybe names whose MIP file
    exists for this FOV -- ONE directory listing, no HDF5 opens. MIP
    files are written atomically (.part + os.replace), so existence is
    completeness.
    """
    try:
        return {f[:-3] for f in os.listdir(mips_dir(storage_path, fov)) if f.endswith('.h5')}
    except OSError:
        return set()
