"""
THE storage-layout resolver: every file path in a project goes through
here, so the on-disk layout is a versioned, swappable fact instead of a
convention smeared across call sites.

The layout:

storage_path = <dp>/{modality} inside a project root <dp>
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

A project is identified by manifest.json in the parent of every
modality store. The pre-manifest v1 layout is gone
from the live pipeline; its reader is frozen in legacy/vlinks_store.py
and tools/migrate_vlinks.py converts an old store once.
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
    target = manifest_path(dp)

    # WRITE ONLY IF IT WOULD CHANGE.
    #
    # Every session rewrites this at startup from its config, so the
    # content is almost always byte-identical to what is already there.
    # Two sessions on one project -- which the launcher explicitly
    # supports, it makes one log per launch precisely because two copies
    # can run at once -- then race on os.replace, and on Windows the loser
    # gets PermissionError/WinError 5 replacing a file the other has open.
    # Observed repeatedly on the real NAS store.
    #
    # The manifest survived (atomic replace: the old file stays whole), but
    # the exception escaped into a session that had done nothing wrong.
    # Skipping the no-op write removes the race in the case that causes it,
    # and removes pointless NAS churn at every startup as a side effect.
    try:
        with open(target, 'r') as f:
            if json.load(f) == m:
                _MANIFEST_CACHE.pop(dp, None)
                return m
    except (OSError, ValueError):
        pass    # missing, unreadable or malformed -- write it

    tmp = target + '.part'
    with open(tmp, 'w') as f:
        json.dump(m, f, indent=2)
    try:
        os.replace(tmp, target)
    except OSError:
        # Another session replaced it first. That is only tolerable if the
        # file now says what we were going to say -- otherwise the caller
        # must hear about it, because a manifest that disagrees with the
        # session's own configuration is not a cosmetic problem.
        try:
            with open(target, 'r') as f:
                same = json.load(f) == m
        except (OSError, ValueError):
            same = False
        try:
            os.remove(tmp)      # never leave a .part behind
        except OSError:
            pass
        if not same:
            raise
    _MANIFEST_CACHE.pop(dp, None)
    return m



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
    _require_fov3(storage_path)
    return os.path.join(storage_path, 'stacks', fov_dir_name(fov), f'{hybe}.h5')


def mip_path(storage_path, fov, hybe):
    """One standalone MIP file per (modality, FOV, hybe)."""
    _require_fov3(storage_path)
    return os.path.join(storage_path, 'mips', fov_dir_name(fov), f'{hybe}.h5')


def mips_dir(storage_path, fov):
    _require_fov3(storage_path)
    return os.path.join(storage_path, 'mips', fov_dir_name(fov))



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
    modality = modality_from_path(storage_path) or 'shared'
    return os.path.join(project_root(storage_path), 'figures', modality,
                        category, fov_dir_name(fov))


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
