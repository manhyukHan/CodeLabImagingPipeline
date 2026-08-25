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
               <dp>/{modality}/stacks/fov###/{hybe}.h5  chunked, write-once
               <dp>/{modality}/mips/fov###/{hybe}.h5    per-hybe, written
                                                        ATOMICALLY by the
                                                        ingestion worker
                                                        (existence == complete)
               <dp>/analysis/fov###/...                 per-FOV analysis
                                                        capsules (see
                                                        io/analysis_store.py;
                                                        replaces the single
                                                        analysis/vlinks.h5,
                                                        now retired)
               <dp>/analysis/params.json                experiment params
               <dp>/figures/{modality}/{category}/fov###/*.png
                                                        all rendered outputs
                                                        ({category}:
                                                        'alignment', 'cells',
                                                        'alleles', ...)

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

def fov_dir_name(fov):
    """THE v2 FOV directory name: fov### everywhere (stacks, mips,
    analysis, figures) -- 3-digit because experiments already exceed 99
    FOVs. v1's FOV## naming survives only inside the frozen legacy
    layout and the legacy vlinks.h5 group paths."""
    return f'fov{int(fov):03d}'


_FOV_NAMING_CHECKED = set()


def _require_fov3(storage_path):
    """Refuse a v2 store whose stacks/mips still carry 2-digit FOV##
    directories: resolving fov### paths against them would silently
    report every FOV as un-ingested (Windows' case-insensitive matching
    hides the case difference, but not the digit count). One listdir per
    root, once per process per storage_path."""
    key = os.path.abspath(storage_path)
    if key in _FOV_NAMING_CHECKED:
        return
    for sub in ('stacks', 'mips'):
        root = os.path.join(storage_path, sub)
        try:
            entries = os.listdir(root)
        except OSError:
            continue
        for name in entries:
            if name[:3].lower() == 'fov' and name[3:].isdigit() and len(name[3:]) != 3:
                raise RuntimeError(
                    f'{root} still uses the retired {name!r}-style FOV '
                    f'directory naming -- run tools/migrate_fov_naming.py '
                    f'once on this project to rename every FOV directory '
                    f'to fov###.')
    _FOV_NAMING_CHECKED.add(key)


def stack_path(storage_path, fov, hybe):
    if is_v2(storage_path):
        _require_fov3(storage_path)
        return os.path.join(storage_path, 'stacks', fov_dir_name(fov), f'{hybe}.h5')
    return os.path.join(storage_path, f'FOV{fov:02d}', f'{hybe}_stack.h5')


def mip_path(storage_path, fov, hybe):
    """v2 only -- v1 keeps MIPs inside vlinks.h5 (callers branch on is_v2)."""
    _require_fov3(storage_path)
    return os.path.join(storage_path, 'mips', fov_dir_name(fov), f'{hybe}.h5')


def mips_dir(storage_path, fov):
    _require_fov3(storage_path)
    return os.path.join(storage_path, 'mips', fov_dir_name(fov))


def vlinks_path(storage_path):
    dp = project_root(storage_path)
    if read_manifest(dp) is not None:
        return os.path.join(dp, 'analysis', 'vlinks.h5')
    return os.path.join(dp, 'vlinks.h5')


def analysis_dir(storage_path):
    """v2 only: the project's analysis root, <dp>/analysis."""
    return os.path.join(project_root(storage_path), 'analysis')


def analysis_fov_dir(storage_path, fov):
    """
    v2 only: one FOV's analysis capsule directory,
    <dp>/analysis/fov### (fov_dir_name -- the one v2 FOV naming).
    """
    return os.path.join(analysis_dir(storage_path), fov_dir_name(fov))


def figure_path(storage_path, category, fov, filename):
    """
    All rendered outputs live under <dp>/figures/{modality}/{category}/
    fov###/ in a v2 store -- modality-major like the data tree itself
    (per explicit decision: figures did not distinguish modalities, so
    two modalities sharing a hybe name overwrote each other's overlays),
    then by category ('alignment' for FOV-level overlays, 'cells',
    'alleles', 'celltype', ...). The modality comes from the
    storage_path the figure was rendered for; a figure rendered outside
    any single modality lands under 'shared'. v1 keeps the legacy
    dump-beside-the-data location.

    Figures already on disk under the pre-modality layout
    (figures/{category}/FOV##/) are regenerable outputs and are simply
    left where they are -- nothing reads figures back.
    """
    d = figure_dir(storage_path, category, fov)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, filename)


def figure_dir(storage_path, category, fov):
    """The directory figure_path resolves into, WITHOUT creating it --
    for callers that only offer it as a default save location."""
    if is_v2(storage_path):
        modality = modality_from_path(storage_path) or 'shared'
        return os.path.join(project_root(storage_path), 'figures', modality,
                            category, fov_dir_name(fov))
    return os.path.join(storage_path, f'FOV{fov:02d}')


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
