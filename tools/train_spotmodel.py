"""
Turn a reviewed bundle into the two spot models, and say what it did.

    python tools/train_spotmodel.py D:/bundles/MP58_RNA_all_4fov \
        --out D:/models/mp58_rna --heads linear,mlp,conv --psf-components 0

WHAT COMES OUT of --out:

    spot_classifier_<head>.json   pass/fail, p calibrated on held-out cells
    psf_bank.h5                   the measured PSF, with the labels that
                                  made it recorded inside
    report.json                   every number printed below, kept

THE SPLIT IS BY CELL. Spots in one cell share a background, a
segmentation, an alignment residual and a focus position; splitting at
random puts siblings on both sides and every score comes out inflated.

WHAT THE NUMBERS ARE FOR. On a small or single-reviewer set they say the
pipeline RUNS and does not collapse to all-fail -- not that the model is
good. This prints the size and shape of the labelled set first, and
refuses to pretend a set of one hybe from one FOV is a benchmark. Read
`--min-groups` as the line below which it will warn loudly.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                          # noqa: E402

from codelab_pipeline.training import dataset as D           # noqa: E402
from codelab_pipeline.training import features as F          # noqa: E402
from codelab_pipeline.localization import psf_bank as PB     # noqa: E402
from codelab_pipeline.training import model_store as MS  # noqa: E402
from codelab_pipeline.localization import psf as PSF         # noqa: E402

DEFAULT_VOXEL = (0.208, 0.208, 0.2)


def analytic_reference(mean, voxel_um, storage_path=None):
    """How far the measured template is from the calibrated analytic one,
    and which analytic family describes the template ITSELF.

    A cheap and worthwhile check: if the installed calibration and the
    template agree, that is confidence in both. If they do not, the
    family fits say which kind of disagreement it is: a template that a
    fitted gaussian describes at cosine 0.95 while the installed readout
    gaussian_halo sits at 0.78 was measured on other emitters (the
    fiducial channel), not on bad labels or bad optics. The installed
    keys (family, params, cosine_to_measured) are {} when this project
    has never been calibrated; the fits are always there.
    """
    out = {}
    try:
        ff = PB.fit_families(mean, voxel_um)
        out['fits'] = ff['fits']
        b = ff['best']
        out['best_fit'] = {'family': b, 'params': ff['fits'][b]['params'],
                           'cosine': ff['fits'][b]['cosine']}
    except Exception as exc:                                 # noqa: BLE001
        out['fits_error'] = f'{type(exc).__name__}: {exc}'
    if not storage_path:
        return out
    try:
        doc = PSF.load(storage_path)
    except Exception:                                        # noqa: BLE001
        return out
    if not doc:
        return out
    # THE PARAMETERS ARE A TUPLE IN THE FAMILY'S OWN ORDER, and getting
    # that from the dict is what psf_library.shape_tuple is for. Passing
    # the dict straight to evaluate() raises -- and the first version of
    # this function did exactly that inside a bare `except`, so the
    # comparison never ran and reported "no analytic calibration" for a
    # store that has one. A silent fallback that looks like an absence is
    # worse than the error it was hiding.
    from codelab_pipeline.localization import psf_library as PL
    st = PL.shape_tuple(doc)
    if st is None:
        return out
    family, shape_params = st
    ny, nx, nz = mean.shape
    yy, xx, zz = np.mgrid[0:ny, 0:nx, 0:nz].astype(float)
    dy = (yy - ny // 2) * voxel_um[0]
    dx = (xx - nx // 2) * voxel_um[1]
    dz = (zz - nz // 2) * voxel_um[2]
    vol = PSF.evaluate(family, shape_params, dy, dx, dz)
    a = PB.normalise(np.asarray(vol, float)).ravel()
    out.update({'family': family, 'params': doc.get('params'),
                'cosine_to_measured': float(a @ PB.normalise(mean).ravel())})
    return out


# What one training run writes. ONLY these are cleared on an overwrite:
# a run folder may hold notes, and a person's file is never a stale
# artefact.
RUN_ARTEFACTS = ('psf_bank.h5', 'psf_multispot.json', 'report.json',
                 MS.MANIFEST)


def clear_run(out):
    """Remove a previous run's artefacts before writing over it.

    WHY NOT JUST OVERWRITE. A retrain writes the heads it was asked for,
    a bank, a report and a manifest -- and only writes a multispot
    calibration when this bundle has multispot verdicts. A calibration
    left from the previous run would then sit beside a NEW bank, pass
    the template-size check (same 7x7x11), be bound into the new
    manifest, and quietly calibrate p3 against a bank it was never
    fitted on. A head file for a head no longer trained would be bound
    the same way. So the run's own artefacts go first, named, and
    nothing else in the folder is touched.
    """
    gone = []
    for n in sorted(os.listdir(out)):
        if n in RUN_ARTEFACTS or (n.startswith('spot_classifier_')
                                  and n.endswith('.json')):
            os.remove(os.path.join(out, n))
            gone.append(n)
    return gone


def calibrate_into(a):
    """Fit ONLY the multispot calibration, into an existing run.

    THE CALIBRATION BELONGS TO THE BANK THAT SCORED THE VERDICTS. Each
    judged match carries the p the reviewer saw, and that p was the NCC
    against whichever psf_bank.h5 Spot Check was matching with. A fresh
    retrain rebuilds a bank from the same positives and usually gets the
    same template -- but 'usually' is not a binding, and a retrain also
    re-fits the classifier, which is the M1 a person may have chosen and
    wants kept. So: the run stays, the calibration is added, the
    manifest is rebound over the new file set.
    """
    run = os.path.abspath(a.calibrate_into)
    if not os.path.isdir(run):
        print(f'no such run directory: {run}')
        return 1
    from codelab_pipeline.training import classify as C
    rp = os.path.join(run, 'report.json')
    try:
        with open(rp, encoding='utf-8') as f:
            report = json.load(f)
    except Exception as exc:                                 # noqa: BLE001
        print(f'{run} has no readable report.json ({exc}); a run to '
              f'calibrate into must be one train_spotmodel wrote.')
        return 1
    r = int(report.get('template_r', a.template_r))
    rz = int(report.get('template_rz', a.template_rz))
    tpl_shape = (2 * r + 1, 2 * r + 1, 2 * rz + 1)
    man = MS.read_manifest(run) or {}
    print(f'run     {run}')
    print(f"model   {man.get('model_id', '?')}  template {tpl_shape}")
    bundles = [os.path.abspath(str(b)) for b in a.bundle_dir]
    cal, calrep = C.fit_multispot(bundles, template=tpl_shape)
    report['multispot'] = calrep
    if cal is None:
        print(f"multispot calibration: {calrep.get('skipped')}")
        print('nothing written')
        return 1
    calrep['bank_model_id'] = man.get('model_id')
    cp = cal.save(os.path.join(run, C.MULTISPOT_NAME))
    print(f"multispot calibration from {calrep['n']} judged matches over "
          f"{calrep['pillars']} pillars:")
    print(f"   raw PR-AUC {calrep['raw_pr_auc']:.3f}  ->  at p 0.5: "
          f"precision {calrep['precision_at_half']:.3f}, "
          f"recall {calrep['recall_at_half']:.3f}")
    print(f'   -> {os.path.basename(cp)}')
    with open(rp, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1)
    # REBOUND, with the run's own provenance kept: write_manifest hashes
    # every file afresh (so the new calibration is bound) but knows
    # nothing of reviewer or bundle unless told.
    # RECORDED AS GIVEN, not absolutised: a path a person typed is what
    # they will look for in the manifest.
    extra = {'bundle': man.get('bundle') or a.bundle_dir[0],
             'reviewer': man.get('reviewer') or a.reviewer,
             'calibrated_from': (list(a.bundle_dir) if len(a.bundle_dir) > 1
                                 else a.bundle_dir[0]),
             'calibrated_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
    new = MS.write_manifest(run, extra=extra)
    print(f"   manifest {man.get('model_id', '?')} -> {new['model_id']} "
          f"over {len(new['files'])} files  (the classifier and the bank "
          f"are untouched; the id changed because the file set did)")
    if a.set_default:
        print(f'   pinned as the default model: {MS.set_default(run)}')
    print(f'writing the run to {run}')
    print(f'\nreport  {rp}')
    return 0


def _short_bundle(b):
    """'<experiment>/<bundle>' -- the two path parts that tell bundles
    apart, for reports and run listings."""
    b = os.path.normpath(str(b))
    exp = os.path.basename(os.path.dirname(os.path.dirname(b)))
    return f'{exp}/{os.path.basename(b)}' if exp else os.path.basename(b)


def _manifest_value(bundle_dir, key):
    """One value from a bundle's own manifest, or None."""
    try:
        with open(os.path.join(bundle_dir, 'bundle_manifest.json'),
                  encoding='utf-8') as f:
            return json.load(f).get(key)
    except Exception:                                        # noqa: BLE001
        return None


def _manifest_store(bundle_dir):
    """The store a bundle was cut from, from its own manifest."""
    return _manifest_value(bundle_dir, 'storage_path')


def _per_bundle_val(clf, head, X, cores, y, groups, kept, bundles, a):
    """The pooled head's validation numbers, per bundle.

    The split is recomputed exactly as classify.train made it (same rows,
    same seed), so 'validation' here is the same held-out cells -- just
    read off one bundle at a time. This is the number that says whether
    the pooled model serves each experiment, not only their average.
    """
    from codelab_pipeline.training import classify as C
    tr, va = D.split_by_group([{'group': g} for g in groups],
                              frac=a.val_frac, seed=a.seed)
    p = clf.score(np.asarray(X) if head != 'conv' else None,
                  np.asarray(cores) if head == 'conv' else None)
    out = {}
    for bi, b in enumerate(bundles):
        idx = [i for i in va if kept[i].get('bundle') == bi]
        yy = np.asarray([y[i] for i in idx], int)
        e = {'n_val': len(idx), 'positive': int(yy.sum()) if len(idx) else 0}
        if len(idx) and 0 < yy.sum() < len(idx):
            pp = np.asarray([p[i] for i in idx], float)
            e['val_pr_auc'] = float(C.pr_auc(yy, pp))
            e['val_roc_auc'] = float(C.roc_auc(yy, pp))
        else:
            e['skipped'] = 'no held-out cells of both classes'
        out[_short_bundle(b)] = e
    return out


def _transfer(X, cores, y, groups, kept, bundles, heads, a):
    """A head trained on ONE bundle, scored on each of the others.

    This is the direct measure of how far two experiments' spots differ:
    a model that has never seen experiment B, judged on all of B's
    labels. Its own held-out number is beside it, so 'transfers worse
    than it validates' is read off one line. Heads that read pixels
    (conv) are left out -- this is about the feature vectors.
    """
    from codelab_pipeline.training import classify as C
    X = np.asarray(X)
    y = np.asarray(y, int)
    out = {}
    for head in [h for h in heads if h != 'conv']:
        out[head] = {}
        for bi, b in enumerate(bundles):
            src = [i for i in range(len(kept)) if kept[i].get('bundle') == bi]
            ys = y[src]
            if len(set(tuple(groups[i]) for i in src)) < 4 or ys.sum() == 0 \
                    or ys.sum() == len(src):
                out[head][_short_bundle(b)] = {'skipped': 'too few cells or '
                                                          'one class only'}
                continue
            try:
                # ONE BUNDLE: any context column is constant here, so the
                # transfer heads read the box features only.
                nb = len(F.NAMES)
                clf, rp = C.train(X[src][:, :nb], [cores[i] for i in src], ys,
                                  [groups[i] for i in src], head=head,
                                  epochs=a.epochs, seed=a.seed,
                                  val_frac=a.val_frac, verbose=False)
            except Exception as exc:                         # noqa: BLE001
                out[head][_short_bundle(b)] = {'skipped': f'{type(exc).__name__}: {exc}'}
                continue
            e = {'own_val_pr_auc': rp['val_pr_auc'],
                 'own_val_roc_auc': rp['val_roc_auc'], 'n_train_rows': len(src),
                 'to': {}}
            for bj, b2 in enumerate(bundles):
                if bj == bi:
                    continue
                dst = [i for i in range(len(kept)) if kept[i].get('bundle') == bj]
                yd = y[dst]
                if not len(dst) or yd.sum() == 0 or yd.sum() == len(dst):
                    e['to'][_short_bundle(b2)] = {'skipped': 'one class only'}
                    continue
                pd = clf.score(X[dst][:, :nb], None)
                e['to'][_short_bundle(b2)] = {
                    'n': len(dst), 'positive_frac': float(yd.mean()),
                    'pr_auc': float(C.pr_auc(yd, pd)),
                    'roc_auc': float(C.roc_auc(yd, pd)),
                    'says_yes_frac': float((pd >= 0.5).mean())}
            out[head][_short_bundle(b)] = e
            line = (f'   {head:6s} trained on {_short_bundle(b)} '
                    f"(own val PR-AUC {rp['val_pr_auc']:.3f})")
            for name, t in e['to'].items():
                line += (f'  ->  {name}: '
                         + (f"PR-AUC {t['pr_auc']:.3f}  ROC {t['roc_auc']:.3f}"
                            f"  (n {t['n']}, {100 * t['positive_frac']:.0f}% "
                            f"positive, says yes to {100 * t['says_yes_frac']:.0f}%)"
                            if 'pr_auc' in t else t['skipped']))
            print(line)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('bundle_dir', nargs='+',
                    help='one bundle, or several to POOL into one model. '
                         'Each keeps its own bundle and verdicts; the run '
                         'records all of them, validates the pooled heads '
                         'per bundle, measures how a head trained on one '
                         'transfers to the others, and fits a PSF template '
                         'per bundle beside the pooled one -- the '
                         'experiment-to-experiment difference, measured.')
    ap.add_argument('--reviewer', default=None,
                    help='WHO LABELLED THIS. It names the run -- '
                         '<reviewer>_<YYYYMMDD-HHMMSS> under <repo>/models '
                         '-- because a model is calibrated to a person: '
                         "both Platt fits are against one reviewer's "
                         'keep/drop, so p is P(THIS reviewer keeps it), not '
                         'a reviewer-free truth. Two people labelling one '
                         'bundle make two legitimately different models and '
                         'a listing that does not say whose is unusable.')
    ap.add_argument('--out', default=None,
                    help='where the run is written. Defaults to '
                         '<repo>/models/<reviewer>_<timestamp>, so a model '
                         'travels with the code the way the psf library '
                         'does. A bare name lands under <repo>/models; an '
                         'absolute path is used as given. Re-training never '
                         'overwrites -- delete a folder to remove a run.')
    ap.add_argument('--set-default', action='store_true',
                    help='pin this run as the one v3 uses when nobody names '
                         'a model (writes <repo>/models/DEFAULT)')
    ap.add_argument('--heads', default='linear,mlp')
    ap.add_argument('--psf-components', type=int, default=0,
                    help='0 stores only the mean template, which is usually '
                         'the honest answer -- see psf_bank\'s header')
    ap.add_argument('--voxel-um', default=','.join(str(v) for v in DEFAULT_VOXEL))
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--val-frac', type=float, default=0.25)
    ap.add_argument('--min-groups', type=int, default=40,
                    help='below this many labelled CELLS the run is treated '
                         'as plumbing, not as a result')
    ap.add_argument('--template-r', type=int, default=PB.DEFAULT_R,
                    help='SEARCH template half-width in y and x. The bank is '
                         'stored at the classifier box size and cut to this '
                         'for matching (psf_bank.centre_crop). It was a '
                         'module constant with no way to vary it, which left '
                         'the open question of 5x5x9 against 7x7x11 '
                         'unmeasurable. The multispot calibration records '
                         'the size it was fitted at and refuses another, so '
                         'a run at a different size gets its own.')
    ap.add_argument('--template-rz', type=int, default=PB.DEFAULT_RZ,
                    help='SEARCH template half-depth. See --template-r.')
    ap.add_argument('--bank-from', default=None, metavar='RUN_DIR',
                    help='when no confirmed spot has pixels (the verdicts '
                         'carry vectors and the shards are gone), reuse '
                         'this run\'s psf_bank.h5 instead of failing.')
    ap.add_argument('--calibrate-into', default=None, metavar='RUN_DIR',
                    help='ADD the multispot calibration to an EXISTING run '
                         'and touch nothing else in it. The classifier and '
                         'the PSF bank stay as they are; psf_multispot.json '
                         'is fitted from this bundle\'s multispot verdicts '
                         'against that run\'s template, report.json gains '
                         'its multispot block, and the manifest is rebound. '
                         'This is how a second-stage review lands on the '
                         'model whose PSF it was judged with, instead of a '
                         'retrain whose bank merely happens to match.')
    ap.add_argument('--genomic-resolution-kb', action='append', default=None,
                    type=float,
                    help='kb per readout step, ONE PER BUNDLE in order, '
                         'overriding what each bundle\'s manifest records. '
                         'When the pooled bundles carry different values '
                         'it becomes a feature of the model '
                         '(features.CONTEXT_NAMES); one value across all '
                         'rows is metadata, recorded but not learned on.')
    ap.add_argument('--channel', type=int, default=None,
                    help='train on this channel\'s crops only. A bundle may '
                         'hold several channels (each build adds its own '
                         'shards) and a reviewer may have judged a few of '
                         'the other one; a model is a model of ONE channel. '
                         'Rows of other channels are counted and left out.')
    ap.add_argument('--storage-path', action='append', default=None,
                    help='to compare the measured PSF against the analytic '
                         'calibration in that store. Default: the store '
                         'each bundle\'s own manifest names.')
    a = ap.parse_args(argv)
    voxel = tuple(float(v) for v in a.voxel_um.split(','))
    if a.calibrate_into:
        return calibrate_into(a)

    bundles = [os.path.abspath(str(b)) for b in a.bundle_dir]
    rows, per_bundle = [], []
    for bi, b in enumerate(bundles):
        rb = D.rows(b)
        if a.channel is not None:
            other = [r for r in rb if int(r.get('channel', -1)) != int(a.channel)]
            if other:
                print(f'channel {a.channel}: {len(other)} labelled row(s) of '
                      f'other channel(s) in {_short_bundle(b)} left out '
                      f'({sorted({int(r["channel"]) for r in other})})')
            rb = [r for r in rb if int(r.get('channel', -1)) == int(a.channel)]
        for r in rb:
            # ONE POOL, DISTINCT CELLS. A cell is (fov, cell) inside its
            # bundle, and two experiments both have a FOV 7 cell 12; a
            # split by cell that could not tell them apart would put one
            # experiment's cell on the other's validation side.
            r['bundle'] = bi
            r['group'] = (bi,) + tuple(r['group'])
        sb = D.summary(rb)
        per_bundle.append({'bundle': b, 'name': _short_bundle(b),
                           'storage_path': _manifest_store(b), 'labels': sb,
                           'channel': a.channel})
        print(f'bundle  {b}')
        print(f'labels  {sb["positive"]} positive, {sb["negative"]} negative, '
              f'{sb["contested"]} contested, {sb["added"]} added')
        print(f'        over {sb["crops"]} crops and {sb["groups"]} cells, '
              f'{len(sb["hybes"])} hybe(s), FOVs {sb["fovs"]}')
        rows.extend(rb)
    s = D.summary(rows)
    if len(bundles) > 1:
        print(f'pooled  {s["positive"]} positive, {s["negative"]} negative '
              f'over {s["crops"]} crops and {s["groups"]} cells from '
              f'{len(bundles)} bundles')
    if not rows:
        print('nothing labelled here yet')
        return 1
    provisional = s['groups'] < a.min_groups or len(s['hybes']) < 2
    if provisional:
        print(f'\n  ** {s["groups"]} cells and {len(s["hybes"])} hybe(s). '
              'Treat everything below as a check that the pipeline runs,')
        print('  ** not as a measure of how well it works.')

    train_rows = [r for r in rows if r['label'] in (0, 1)]
    kept, X, cores = D.features_for_rows(train_rows)
    n_cut = sum(1 for c in cores if c is not None)
    print(f'\nrows    {len(kept)} of {len(train_rows)} (the rest ran too far '
          f'off their crop): {n_cut} cut from shards, {len(kept) - n_cut} '
          f'from vectors stored in the verdicts')
    if s.get('inconsistent'):
        print(f"   ** {s['inconsistent']} candidate(s) EXCLUDED: reviewers' "
              f"stored vectors disagree, so they were judged against "
              f"different pixels")
    if not len(kept):
        print('nothing to train on')
        return 1
    y = np.array([r['label'] for r in kept])
    groups = [tuple(r['group']) for r in kept]

    # THE CONTEXT FEATURE, when it varies. Each bundle's genomic
    # resolution (the manifest's, or --genomic-resolution-kb per bundle)
    # is one value for every row of that bundle. With two or more
    # distinct values across the pooled rows it is appended as
    # log10_genomic_resolution_kb and the model learns on it; with one
    # value it is a constant column -- unidentifiable here and wild on
    # the next experiment after standardising -- so it is recorded and
    # not trained on, and the run says which.
    res_by_bundle = {}
    for bi, b in enumerate(bundles):
        given = (a.genomic_resolution_kb or [])
        v = given[bi] if bi < len(given) else None
        if v is None:
            v = _manifest_value(b, 'genomic_resolution_kb')
        res_by_bundle[bi] = float(v) if v else None
        per_bundle[bi]['genomic_resolution_kb'] = res_by_bundle[bi]
    feature_names = list(F.NAMES)
    context_used = False
    distinct = {v for v in res_by_bundle.values() if v}
    if any(v is None for v in res_by_bundle.values()):
        missing = [_short_bundle(bundles[bi]) for bi, v in res_by_bundle.items()
                   if v is None]
        print(f'\ncontext genomic resolution unknown for {missing} -- the '
              f'model has no resolution feature (set it on the Ingestion '
              f'tab before building, or pass --genomic-resolution-kb)')
    elif len(distinct) < 2:
        print(f'\ncontext genomic resolution {sorted(distinct)} kb, one value '
              f'for every row: recorded, not a feature (nothing to learn '
              f'from a constant)')
    else:
        col = np.array([[np.log10(res_by_bundle[r['bundle']])] for r in kept])
        X = np.hstack([np.asarray(X, float), col])
        feature_names = list(F.NAMES) + list(F.CONTEXT_NAMES)
        context_used = True
        print(f'\ncontext genomic resolution per bundle '
              f'{ {_short_bundle(bundles[bi]): v for bi, v in res_by_bundle.items()} } '
              f'kb -> log10_genomic_resolution_kb is the 17th feature')
    report_context = {'genomic_resolution_kb': {
        _short_bundle(bundles[bi]): v for bi, v in res_by_bundle.items()},
        'as_feature': context_used}
    if any(c is None for c in cores) and 'conv' in a.heads:
        print('   conv head skipped: it reads pixels, and some rows here '
              'have only their stored vector')
        a.heads = ','.join(h for h in a.heads.split(',')
                           if h.strip() != 'conv')
    if not a.out:
        if not a.reviewer:
            ap.error('--reviewer is required (or pass --out). A model is '
                     'calibrated to whoever labelled it, so a run without '
                     'a name for that person cannot be told apart from '
                     "anyone else's.")
        a.out = os.path.join(MS.models_dir(), MS.run_name(a.reviewer))
    elif not os.path.isabs(a.out) and os.sep not in a.out and '/' not in a.out:
        a.out = os.path.join(MS.models_dir(), a.out)
    existed = os.path.isdir(a.out) and bool(os.listdir(a.out))
    os.makedirs(a.out, exist_ok=True)
    if existed:
        gone = clear_run(a.out)
        print(f'OVERWRITING {a.out}: removed its previous '
              f'{len(gone)} artefact(s) ({", ".join(gone)}) so nothing '
              f'from the old run is bound into the new one')
    print(f'writing the run to {a.out}')

    bundle_label = (a.bundle_dir[0] if len(bundles) == 1
                    else ' + '.join(_short_bundle(b) for b in bundles))
    report = {'bundle': bundle_label, 'bundles': per_bundle, 'labels': s,
              'channel': a.channel,
              'context': report_context, 'features': feature_names,
              'provisional': provisional,
              'n_boxes': len(kept), 'heads': {}, 'voxel_um': list(voxel),
              'template_r': a.template_r, 'template_rz': a.template_rz}

    from codelab_pipeline.training import classify as C
    print('\nclassifier (split by cell):')
    for head in [h.strip() for h in a.heads.split(',') if h.strip()]:
        try:
            clf, rep = C.train(X, (np.asarray(cores) if head == 'conv'
                                   else cores), y, groups, head=head,
                               epochs=a.epochs, seed=a.seed,
                               val_frac=a.val_frac,
                               feature_names=feature_names)
        except Exception as exc:                             # noqa: BLE001
            print(f'   {head:6s} FAILED {type(exc).__name__}: {exc}')
            report['heads'][head] = {'error': str(exc)}
            continue
        p = clf.score(X if head != 'conv' else None,
                      np.asarray(cores) if head == 'conv' else None)
        rep['degenerate'] = bool(float(p.max() - p.min()) < 1e-3)
        rep['all_fail'] = bool((p >= clf.threshold).sum() == 0)
        rep['all_pass'] = bool((p < clf.threshold).sum() == 0)
        path = clf.save(os.path.join(a.out, f'spot_classifier_{head}.json'))
        rep['path'] = path
        if len(bundles) > 1:
            rep['per_bundle'] = _per_bundle_val(clf, head, X, cores, y, groups,
                                                kept, bundles, a)
            for name, e in rep['per_bundle'].items():
                print(f'          on {name}: '
                      + (f"val PR-AUC {e['val_pr_auc']:.3f}  ROC "
                         f"{e['val_roc_auc']:.3f}  (n {e['n_val']}, "
                         f"{e['positive']} positive)"
                         if 'val_pr_auc' in e else e['skipped']))
        report['heads'][head] = rep
        flag = ('  DEGENERATE' if rep['degenerate'] else
                '  ALL-FAIL' if rep['all_fail'] else
                '  ALL-PASS' if rep['all_pass'] else '')
        print(f'          -> {os.path.basename(path)}{flag}')

    if len(bundles) > 1:
        print('\ntransfer (a head trained on ONE bundle, scored on all of '
              'another):')
        report['transfer'] = _transfer(
            X, cores, y, groups, kept, bundles,
            [h.strip() for h in a.heads.split(',') if h.strip()], a)

    pos = [c for c, r in zip(cores, kept)
           if r['label'] == 1 and c is not None]
    n_pos_all = sum(1 for r in kept if r['label'] == 1)
    print(f'\nPSF bank from {len(pos)} confirmed spots'
          + (f' ({n_pos_all - len(pos)} more have only their vector; a bank '
             f'is averaged PIXELS and cannot use those)'
             if n_pos_all > len(pos) else '') + ':')
    if not pos and a.bank_from:
        import shutil
        shutil.copyfile(os.path.join(a.bank_from, 'psf_bank.h5'),
                        os.path.join(a.out, 'psf_bank.h5'))
        print(f'   no pixels to build one from -- REUSED the bank of '
              f'{a.bank_from}')
        report['psf'] = {'reused_from': a.bank_from}
    elif not pos:
        print('   no positives with pixels -- nothing to measure a PSF '
              'from. Pass --bank-from RUN to reuse an earlier run\'s bank.')
    else:
        mean, comps, var = PB.build(pos, voxel, n_components=a.psf_components)
        store0 = ((a.storage_path or [None])[0]
                  or per_bundle[0].get('storage_path'))
        ref = analytic_reference(mean, voxel, store0)
        # PER BUNDLE, BESIDE THE POOLED ONE: each experiment's own
        # template, its best analytic fit, its cosine to the pooled
        # template, and the templates' cosines to each other -- the
        # experiment-to-experiment difference of the emitters, measured.
        psf_per, psf_between = [], {}
        if len(bundles) > 1:
            means = {}
            print('   per bundle:')
            for bi, b in enumerate(bundles):
                pos_b = [c for c, r in zip(cores, kept)
                         if r['label'] == 1 and c is not None
                         and r.get('bundle') == bi]
                if len(pos_b) < 20:
                    psf_per.append({'bundle': _short_bundle(b),
                                    'n_spots': len(pos_b),
                                    'skipped': 'fewer than 20 confirmed '
                                               'spots with pixels'})
                    print(f'      {_short_bundle(b)}: {len(pos_b)} spots, '
                          f'too few for a template of its own')
                    continue
                mb, _cb, _vb = PB.build(pos_b, voxel)
                ff = PB.fit_families(mb, voxel)
                bf = ff['best']
                means[bi] = mb
                cos_pooled = float(PB.normalise(mb).ravel()
                                   @ PB.normalise(mean).ravel())
                prm = ff['fits'][bf]['params']
                psf_per.append({'bundle': _short_bundle(b), 'n_spots': len(pos_b),
                                'best_fit': {'family': bf,
                                             'cosine': ff['fits'][bf]['cosine'],
                                             'params': prm},
                                'fits': {k: {'cosine': v['cosine'],
                                             'params': v['params']}
                                         for k, v in ff['fits'].items()},
                                'cosine_to_pooled': cos_pooled})
                print(f"      {_short_bundle(b)}: {len(pos_b)} spots, "
                      f"best fit {bf} (cosine {ff['fits'][bf]['cosine']:.3f}, "
                      f"sigma_xy {1000 * prm['sigma_xy_um']:.0f} nm, "
                      f"sigma_z {1000 * prm['sigma_z_um']:.0f} nm), "
                      f"cosine to the pooled template {cos_pooled:.3f}")
            keys = sorted(means)
            for i in keys:
                for j in keys:
                    if j > i:
                        c = float(PB.normalise(means[i]).ravel()
                                  @ PB.normalise(means[j]).ravel())
                        psf_between[f'{_short_bundle(bundles[i])} | '
                                    f'{_short_bundle(bundles[j])}'] = c
                        print(f'      {_short_bundle(bundles[i])} vs '
                              f'{_short_bundle(bundles[j])}: cosine {c:.3f}')
        reviewers = sorted({rv for r in rows
                            for rv in ([r.get('reviewer')] if r.get('reviewer')
                                       else [])})
        path = PB.save(
            os.path.join(a.out, 'psf_bank.h5'), mean, voxel,
            components=comps, explained_var=var, n_spots=len(pos),
            source={'bundle': bundle_label, 'bundles': bundles,
                    'store': kept[0].get('store'),
                    'hybes': s['hybes'], 'fovs': s['fovs'],
                    'channel': kept[0]['channel']},
            labels={'reviewers': reviewers, 'n_positive': len(pos),
                    'n_negative': s['negative'], 'n_contested': s['contested'],
                    'selection': 'labels() positive bucket',
                    'provisional': provisional},
            analytic_ref=ref)
        # THE MATCHER'S OWN OPERATING POINT, fitted here so it SHIPS.
        # Without it the matched filter has no calibrated threshold and a
        # caller either hard-codes one experiment's answer as a universal
        # constant or assumes every experiment arrives with its own
        # multispot review. Neither is acceptable; a Platt pair beside
        # the template is the same kind of artefact as the classifier.
        # Absent multispot verdicts this simply does not write, and the
        # matcher falls back to its sigma threshold.
        tpl_shape = (2 * a.template_r + 1, 2 * a.template_r + 1,
                     2 * a.template_rz + 1)
        cal, calrep = C.fit_multispot(bundles, template=tpl_shape)
        report['multispot'] = calrep
        if cal is None:
            print(f"\nmultispot calibration: {calrep.get('skipped')}")
        else:
            cp = cal.save(os.path.join(a.out, C.MULTISPOT_NAME))
            print(f"\nmultispot calibration from {calrep['n']} judged "
                  f"matches over {calrep['pillars']} pillars:")
            print(f"   raw PR-AUC {calrep['raw_pr_auc']:.3f}  ->  at p 0.5: "
                  f"precision {calrep['precision_at_half']:.3f}, "
                  f"recall {calrep['recall_at_half']:.3f}")
            print(f'   -> {os.path.basename(cp)}')
        warn = PB.cosine_warning(ref, n_spots=len(pos))
        if warn:
            print('\n   WARNING: ' + warn)
            report.setdefault('warnings', []).append(warn)
        d = PB.drift(pos, mean)
        print(f'   template {mean.shape}, components {comps.shape[0]}'
              + (f', explained {np.round(var * 100, 1).tolist()}%'
                 if len(var) else ''))
        print(f'   its own spots score median {float(np.median(d)):.3f} '
              f'(p10 {float(np.percentile(d, 10)):.3f})')
        if ref.get('cosine_to_measured') is not None:
            print(f'   vs the installed {ref.get("family")}: '
                  f'cosine {ref["cosine_to_measured"]:.3f}')
        for fam, fit in (ref.get('fits') or {}).items():
            pr = fit.get('params') or {}
            print(f'   fitted {fam:<13} cosine {fit["cosine"]:.3f}   '
                  + '  '.join(f'{k}={v:.4f}' for k, v in pr.items())
                  + ('' if fit.get('plausible') else '   (implausible)')
                  + ('   <- best' if fam == (ref.get('best_fit') or {}).get('family')
                     else ''))
        print(f'   -> {os.path.basename(path)}')
        report['psf'] = {'path': path, 'n_spots': len(pos),
                         'explained_var': [float(v) for v in var],
                         'drift_median': float(np.median(d)),
                         'analytic_ref': ref}
        if psf_per:
            report['psf']['per_bundle'] = psf_per
            report['psf']['between'] = psf_between

    # THE MANIFEST IS WRITTEN LAST, over finished artefacts, so it can
    # never describe a run that did not complete. It binds the three
    # learned files to each other -- each already refuses the one
    # mismatch it can see for itself, and none of them can see that they
    # came from different runs.
    rp = os.path.join(a.out, 'report.json')
    with open(rp, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1)
    man = MS.write_manifest(a.out, extra={'bundle': bundle_label,
                                          'bundles': list(a.bundle_dir),
                                          'reviewer': a.reviewer})
    if a.set_default:
        print(f'   pinned as the default model: {MS.set_default(a.out)}')
    print(f"   manifest {man['model_id']} over {len(man['files'])} files")
    print(f'\nreport  {rp}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
