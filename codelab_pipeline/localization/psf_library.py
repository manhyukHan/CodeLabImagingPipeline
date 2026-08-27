"""
The PSF library: every readout PSF ever calibrated, kept and reusable.

WHY A LIBRARY AND NOT ONE FILE PER EXPERIMENT
---------------------------------------------
The readout PSF is close to universal. Four experiments spanning 64x in
genomic scope (0.29 to 18.5 Mb) agreed on family and to within 20 nm in
sigma_xy -- tighter than the ~40 nm one experiment moves when the crops
fed to its own calibration are reselected. So the most useful thing a
calibration produces is not a number for that experiment, it is another
data point about the microscope.

A per-experiment file throws that away. This folder is GIT-TRACKED on
purpose: the library accumulates across experiments and people, and a
calibration run months from now can be compared against every earlier one
instead of standing alone.

READOUT ONLY. THERE IS NO FIDUCIAL PSF.
---------------------------------------
Nothing in here describes a fiducial, and nothing should. The fiducial is
an extended object -- the whole genomic region the readouts trace -- and
a Gaussian fitted to it returns the FIT WINDOW, not a width: measured
sigma ~ r^0.5 across radii 0.6/0.8/1.2/1.6 um on all four experiments,
with no plateau anywhere. Storing a "fiducial PSF" would file a number
that does not exist, and everything downstream would then trust it. The
fiducial is fitted per hybe with sigma FREE and generous bounds, and its
width is a QC observation, never a stored constant.

HOW IT IS USED
--------------
    library (git, here)  ->  pick one  ->  copied INTO the experiment
                                           at <project>/analysis/psf.json

The copy is what the run reads, so a run is reproducible from its own
store even if the library moves on. The config records the LABEL, so the
choice is visible in the experiment description rather than implied.
"""
import json
import os
import re
import time

LIBRARY_DIRNAME = 'psf'
SUFFIX = '.json'
# <experiment>-<hybe>-ch<channel>-<YYYYMMDD>, e.g. MP58-Hyb_016-ch555-20260827
LABEL_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.+-]{0,80}$')


def library_dir():
    """<repo>/psf -- tracked, so the library travels with the code."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    return os.path.join(root, LIBRARY_DIRNAME)


def default_label(experiment='', hybe='', channel='', when=None):
    """The naming convention. Callers may override it entirely.

    Every part is optional so a hand-made entry can still be named
    sensibly; empty parts are dropped rather than leaving '--' holes.
    """
    stamp = time.strftime('%Y%m%d', time.localtime(when)) if when is not None \
        else time.strftime('%Y%m%d')
    parts = [str(p) for p in (experiment, hybe) if str(p).strip()]
    if str(channel).strip():
        parts.append(f'ch{channel}')
    parts.append(stamp)
    label = '-'.join(parts)
    return sanitize_label(label)


def sanitize_label(label):
    """A label is a FILENAME. Anything that could escape the folder or
    collide with a path separator is replaced, not rejected -- a user
    renaming an entry should not have to learn the rules."""
    out = re.sub(r'[^A-Za-z0-9_.+-]+', '-', str(label).strip()).strip('-.')
    return out[:80] or 'psf'


def entry_path(label):
    return os.path.join(library_dir(), sanitize_label(label) + SUFFIX)


def write(label, family, params, voxel_um, source, scores=None,
          converged=None, notes=''):
    """
    Add (or replace) a library entry. Atomic, like every other write in
    this project: .part then os.replace, never delete-then-rewrite.

    `converged` should carry the evidence that the fit had settled --
    e.g. {'n_crops': [80, 160], 'sigma_xy_nm': [147, 146], 'delta_nm': 1}.
    A calibration still moving at its largest crop count is a waypoint,
    not an answer, and the reader deserves to see which it got.
    """
    os.makedirs(library_dir(), exist_ok=True)
    target = entry_path(label)
    doc = {
        'label': sanitize_label(label),
        'kind': 'readout',            # the only kind; see the module note
        'family': family,
        'params': dict(params),
        'voxel_um': list(voxel_um),
        'source': dict(source or {}),
        'converged': converged,
        'notes': notes,
        'candidates': scores or {},
        'written': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    tmp = target + '.part'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    os.replace(tmp, target)
    return target


def read(label):
    """One entry, or None. Never raises on a malformed file -- a broken
    entry must not stop the others being listed or used."""
    p = entry_path(label)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def list_entries():
    """Every entry, newest first, each with its label. Malformed files are
    skipped silently here and reported by `problems()`."""
    d = library_dir()
    if not os.path.isdir(d):
        return []
    out = []
    for name in os.listdir(d):
        if not name.endswith(SUFFIX):
            continue
        doc = read(name[:-len(SUFFIX)])
        if doc is None:
            continue
        doc.setdefault('label', name[:-len(SUFFIX)])
        out.append(doc)
    out.sort(key=lambda d: (d.get('written') or ''), reverse=True)
    return out


def problems():
    """Labels whose files exist but could not be parsed."""
    d = library_dir()
    if not os.path.isdir(d):
        return []
    bad = []
    for name in os.listdir(d):
        if name.endswith(SUFFIX) and read(name[:-len(SUFFIX)]) is None:
            bad.append(name[:-len(SUFFIX)])
    return sorted(bad)


def install(label, storage_path):
    """Copy the chosen entry INTO the experiment, at <project>/analysis/psf.json.

    The run reads the copy, never the library, so an experiment stays
    reproducible from its own store after the library has moved on. The
    installed copy records which label it came from, so a store found
    later can be traced back.
    """
    from codelab_pipeline.io import paths
    doc = read(label)
    if doc is None:
        return None
    doc = dict(doc)
    doc['installed_from'] = doc.get('label', sanitize_label(label))
    doc['installed_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    target = os.path.join(paths.analysis_dir(storage_path), 'psf.json')
    os.makedirs(os.path.dirname(target), exist_ok=True)
    tmp = target + '.part'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    os.replace(tmp, target)
    return target


def installed(storage_path):
    """What this experiment is actually using, or None."""
    from codelab_pipeline.io import paths
    p = os.path.join(paths.analysis_dir(storage_path), 'psf.json')
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def shape_tuple(doc):
    """(family, shape_params) ready for psf.evaluate, or None.

    Ordered by the family's own parameter list rather than by dict order,
    because a shape passed in the wrong order is still a valid-looking
    tuple and would silently describe a different PSF.
    """
    from codelab_pipeline.localization import psf as P
    if not doc:
        return None
    fam = doc.get('family')
    if fam not in P.FAMILIES:
        return None
    names = P.FAMILIES[fam][1]
    params = doc.get('params') or {}
    if any(n not in params for n in names):
        return None
    return fam, tuple(float(params[n]) for n in names)
