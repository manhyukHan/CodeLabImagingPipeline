"""
Trained models live in the app, one folder per run, each with a manifest.

WHERE. `<repo>/models/<name>/`, mirroring psf_library.library_dir()'s
`<repo>/psf` -- so a model travels with the code the way the PSF library
does, and a checkout is a working install. The first models were written
to D:/models, which put a thing the app depends on outside the app.

ONE FOLDER PER RUN, NEVER OVERWRITTEN. Re-training is a new folder, so
the model that produced last week's spots is still on disk to compare
against and to re-run. A UI lists them and lets a person pick.

WHAT A RUN CONTAINS. Three learned artefacts, and a manifest that binds
them:

    spot_classifier_<head>.json   weights, Platt (a, b), the standardiser
                                  and the feature list it was fitted on
    psf_bank.h5                   the measured template, its voxel grid,
                                  and the analytic PSF it was compared to
    psf_multispot.json            Platt (a, b) for the matcher's score,
                                  and the template size it was fitted at
    report.json                   every number the run printed
    manifest.json                 THIS FILE -- see below

THE MANIFEST EXISTS BECAUSE A FOLDER IS NOT A GUARANTEE. Each artefact
already refuses the one mismatch it can see for itself: the classifier
refuses a changed features.NAMES, the bank refuses another voxel grid,
the calibration refuses another template size. None of them can see that
they came from DIFFERENT RUNS -- drop one run's classifier beside
another's bank and every individual check passes. The manifest records a
content hash for each file and load() verifies them, so that mixture is
an error rather than a silent one.

IT IS ALSO WHAT A STORED RESULT POINTS AT. `model_id` is the hash of the
manifest, short enough to write beside a spot; see `stamp`.

NOTHING HERE IS NEEDED TO TRAIN OR TO INFER FROM A BUNDLE. The artefacts
are numbers and arrays -- MEASURED by copying a model directory somewhere
with no bundle in sight and localizing from it. `source.bundle` in the
bank and `bundle` in the report are PROVENANCE strings; a bundle is a
deletable cache and deleting it costs a model nothing.
"""
import hashlib
import json
import os
import time

DIRNAME = 'models'
MANIFEST = 'manifest.json'

# The files a run is expected to write. A missing OPTIONAL one is a
# state, not a fault: psf_multispot.json is absent whenever the training
# bundle carried no multispot verdicts, which is the ordinary bootstrap
# -- review pass/fail, train, use, then review multispot and re-train.
REQUIRED = ('psf_bank.h5', 'report.json')
OPTIONAL = ('psf_multispot.json',)


DEFAULT_MARKER = 'DEFAULT'


def run_name(reviewer, when=None):
    """<reviewer>_<YYYYMMDD-HHMMSS> -- the name a training run gets.

    THE REVIEWER IS IN THE NAME because a model is calibrated to a
    person. Both Platt fits in a run are fitted against one reviewer's
    keep/drop decisions, so p is P(THIS reviewer keeps it | features) and
    not some reviewer-free truth. Two people labelling the same bundle
    produce two legitimately different models, and a folder listing that
    does not say whose is a listing nobody can choose from.

    The timestamp makes it unique without a counter, and re-training is
    a NEW FOLDER rather than an overwrite -- the model that produced last
    week's spots stays on disk to compare against and to re-run. Deleting
    one is deleting its folder; nothing else refers to it.
    """
    safe = ''.join(c if (c.isalnum() or c in '._-') else '-'
                   for c in str(reviewer or '')).strip('-') or 'anon'
    return f'{safe}_{time.strftime("%Y%m%d-%H%M%S", when or time.localtime())}'


def default_model(root=None):
    """The run used when nobody names one, or None.

    A `DEFAULT` file naming a folder if there is one -- so a person can
    PIN the model their lab uses and a fresh training run does not
    silently become everyone's default. Failing that, the newest run,
    which is the only defensible guess.
    """
    d = models_dir(root)
    marker = os.path.join(d, DEFAULT_MARKER)
    if os.path.exists(marker):
        try:
            with open(marker, encoding='utf-8') as f:
                name = f.read().strip()
            if name and os.path.isdir(os.path.join(d, name)):
                return os.path.join(d, name)
        except OSError:
            pass
    runs = available(root)
    return runs[0]['path'] if runs else None


def set_default(model_dir, root=None):
    """Pin a run as the default. Writes <repo>/models/DEFAULT."""
    d = models_dir(root)
    name = os.path.basename(os.path.abspath(str(model_dir)))
    if not os.path.isdir(os.path.join(d, name)):
        raise ValueError(f'{name} is not a run under {d}')
    tmp = os.path.join(d, DEFAULT_MARKER + '.part')
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(name + '\n')
    os.replace(tmp, os.path.join(d, DEFAULT_MARKER))
    return name


def models_dir(root=None):
    """<repo>/models -- the same rule psf_library.library_dir() follows."""
    if root:
        return str(root)
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(os.path.dirname(here)), DIRNAME)


def _sha(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def _files(d):
    return sorted(n for n in os.listdir(d)
                  if os.path.isfile(os.path.join(d, n)) and n != MANIFEST
                  and n != DEFAULT_MARKER and not n.endswith('.part'))


def write_manifest(model_dir, extra=None):
    """Hash every artefact and bind them. Returns the manifest dict.

    Written LAST, after the artefacts, so a manifest never describes a
    run that did not finish.
    """
    d = str(model_dir)
    files = {n: _sha(os.path.join(d, n)) for n in _files(d)}
    doc = {'name': os.path.basename(os.path.abspath(d)),
           'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'files': files}
    doc.update(extra or {})
    # The id is the hash of everything that identifies the run. It does
    # NOT include `created`, so re-writing a manifest over identical
    # artefacts gives the same id rather than a new one.
    doc['model_id'] = hashlib.sha256(
        json.dumps({'name': doc['name'], 'files': files},
                   sort_keys=True).encode()).hexdigest()[:12]
    tmp = os.path.join(d, MANIFEST + '.part')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, os.path.join(d, MANIFEST))
    return doc


def read_manifest(model_dir):
    p = os.path.join(str(model_dir), MANIFEST)
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def verify(model_dir):
    """[] when the folder matches its manifest, else a list of complaints.

    Never raises on a missing manifest -- a hand-made folder is allowed,
    it simply carries no guarantee, and saying so is more useful than
    refusing to load it.
    """
    d = str(model_dir)
    man = read_manifest(d)
    if man is None:
        return ['no manifest -- this folder was not written by a training '
                'run, so nothing binds its artefacts to each other']
    out = []
    for n, want in (man.get('files') or {}).items():
        p = os.path.join(d, n)
        if not os.path.exists(p):
            out.append(f'{n} is missing')
        elif _sha(p) != want:
            out.append(f'{n} does not match the manifest -- it was replaced '
                       f'or came from another run')
    for n in _files(d):
        if n not in (man.get('files') or {}):
            out.append(f'{n} is not in the manifest')
    for n in REQUIRED:
        if not os.path.exists(os.path.join(d, n)):
            out.append(f'{n} is required and absent')
    return out


def model_id(model_dir):
    """The short hash a stored result points at, or None."""
    man = read_manifest(model_dir)
    return (man or {}).get('model_id')


def available(root=None):
    """[{name, path, model_id, created, heads, has_multispot, problems}]

    Every run under <repo>/models, newest first. A folder with problems
    is LISTED with them rather than hidden: a person choosing a model is
    exactly who should see that one of its files was replaced.
    """
    d = models_dir(root)
    if not os.path.isdir(d):
        return []
    out = []
    marker = os.path.basename(default_model(root) or '')
    for n in sorted(os.listdir(d)):
        p = os.path.join(d, n)
        if not os.path.isdir(p):
            continue
        man = read_manifest(p) or {}
        rep = {}
        rp = os.path.join(p, 'report.json')
        if os.path.exists(rp):
            try:
                with open(rp, encoding='utf-8') as f:
                    rep = json.load(f)
            except (OSError, ValueError):
                rep = {}
        out.append({
            'name': n, 'path': p,
            'model_id': man.get('model_id'),
            'created': man.get('created'),
            'heads': sorted((rep.get('heads') or {})),
            'bundle': rep.get('bundle'),
            'has_multispot': os.path.exists(
                os.path.join(p, 'psf_multispot.json')),
            'reviewer': man.get('reviewer'),
            'is_default': (n == marker),
            'problems': verify(p)})
    out.sort(key=lambda r: (r.get('created') or ''), reverse=True)
    return out


def stamp(model_dir):
    """{model, model_id} -- what a result records about what produced it.

    SMALL ON PURPOSE. The alternative is a per-spot column, and the
    granularity that actually matters is the WRITE: every spot in one
    localization run came from one model, so recording it once per write
    answers "which model produced these" without a string on every row.
    """
    man = read_manifest(model_dir) or {}
    return {'model': man.get('name') or os.path.basename(
        os.path.abspath(str(model_dir))),
        'model_id': man.get('model_id')}
