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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                          # noqa: E402

from codelab_pipeline.training import dataset as D           # noqa: E402
from codelab_pipeline.training import features as F          # noqa: E402
from codelab_pipeline.localization import psf_bank as PB     # noqa: E402
from codelab_pipeline.localization import psf as PSF         # noqa: E402

DEFAULT_VOXEL = (0.208, 0.208, 0.2)


def analytic_reference(mean, voxel_um, storage_path=None):
    """How far the measured template is from the calibrated analytic one.

    A cheap and worthwhile check: if the two agree, that is confidence in
    both; if they do not, the difference is what an analytic family
    cannot express, which is the reason to measure a template at all.
    Returns {} when this project has never been analytically calibrated.
    """
    if not storage_path:
        return {}
    try:
        doc = PSF.load(storage_path)
    except Exception:                                        # noqa: BLE001
        return {}
    if not doc:
        return {}
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
        return {}
    family, shape_params = st
    ny, nx, nz = mean.shape
    yy, xx, zz = np.mgrid[0:ny, 0:nx, 0:nz].astype(float)
    dy = (yy - ny // 2) * voxel_um[0]
    dx = (xx - nx // 2) * voxel_um[1]
    dz = (zz - nz // 2) * voxel_um[2]
    vol = PSF.evaluate(family, shape_params, dy, dx, dz)
    a = PB.normalise(np.asarray(vol, float)).ravel()
    return {'family': family, 'params': doc.get('params'),
            'cosine_to_measured': float(a @ PB.normalise(mean).ravel())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('bundle_dir')
    ap.add_argument('--out', required=True)
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
    ap.add_argument('--storage-path', default=None,
                    help='to compare the measured PSF against the analytic '
                         'calibration in that store')
    a = ap.parse_args(argv)
    voxel = tuple(float(v) for v in a.voxel_um.split(','))

    rows = D.rows(a.bundle_dir)
    s = D.summary(rows)
    print(f'bundle  {a.bundle_dir}')
    print(f'labels  {s["positive"]} positive, {s["negative"]} negative, '
          f'{s["contested"]} contested, {s["added"]} added')
    print(f'        over {s["crops"]} crops and {s["groups"]} cells, '
          f'hybes {s["hybes"]}, FOVs {s["fovs"]}')
    if not rows:
        print('nothing labelled here yet')
        return 1
    provisional = s['groups'] < a.min_groups or len(s['hybes']) < 2
    if provisional:
        print(f'\n  ** {s["groups"]} cells and {len(s["hybes"])} hybe(s). '
              'Treat everything below as a check that the pipeline runs,')
        print('  ** not as a measure of how well it works.')

    train_rows = [r for r in rows if r['label'] in (0, 1)]
    kept, cores, cols = D.boxes(train_rows)
    print(f'\nboxes   {len(kept)} of {len(train_rows)} (the rest ran too far '
          f'off their crop), core {cores.shape[1:]}, column {cols.shape[1]}')
    X = F.many(cores, cols, kept)
    y = np.array([r['label'] for r in kept])
    groups = [tuple(r['group']) for r in kept]
    os.makedirs(a.out, exist_ok=True)

    report = {'bundle': a.bundle_dir, 'labels': s, 'provisional': provisional,
              'n_boxes': len(kept), 'heads': {}, 'voxel_um': list(voxel)}

    from codelab_pipeline.training import classify as C
    print('\nclassifier (split by cell):')
    for head in [h.strip() for h in a.heads.split(',') if h.strip()]:
        try:
            clf, rep = C.train(X, cores, y, groups, head=head,
                               epochs=a.epochs, seed=a.seed,
                               val_frac=a.val_frac)
        except Exception as exc:                             # noqa: BLE001
            print(f'   {head:6s} FAILED {type(exc).__name__}: {exc}')
            report['heads'][head] = {'error': str(exc)}
            continue
        p = clf.score(X if head != 'conv' else None,
                      cores if head == 'conv' else None)
        rep['degenerate'] = bool(float(p.max() - p.min()) < 1e-3)
        rep['all_fail'] = bool((p >= clf.threshold).sum() == 0)
        rep['all_pass'] = bool((p < clf.threshold).sum() == 0)
        path = clf.save(os.path.join(a.out, f'spot_classifier_{head}.json'))
        rep['path'] = path
        report['heads'][head] = rep
        flag = ('  DEGENERATE' if rep['degenerate'] else
                '  ALL-FAIL' if rep['all_fail'] else
                '  ALL-PASS' if rep['all_pass'] else '')
        print(f'          -> {os.path.basename(path)}{flag}')

    pos = [c for c, r in zip(cores, kept) if r['label'] == 1]
    print(f'\nPSF bank from {len(pos)} confirmed spots:')
    if not pos:
        print('   no positives -- nothing to measure a PSF from')
    else:
        mean, comps, var = PB.build(pos, voxel, n_components=a.psf_components)
        ref = analytic_reference(mean, voxel, a.storage_path)
        reviewers = sorted({rv for r in rows
                            for rv in ([r.get('reviewer')] if r.get('reviewer')
                                       else [])})
        path = PB.save(
            os.path.join(a.out, 'psf_bank.h5'), mean, voxel,
            components=comps, explained_var=var, n_spots=len(pos),
            source={'bundle': a.bundle_dir, 'store': kept[0].get('store'),
                    'hybes': s['hybes'], 'fovs': s['fovs'],
                    'channel': kept[0]['channel']},
            labels={'reviewers': reviewers, 'n_positive': len(pos),
                    'n_negative': s['negative'], 'n_contested': s['contested'],
                    'selection': 'labels() positive bucket',
                    'provisional': provisional},
            analytic_ref=ref)
        d = PB.drift(pos, mean)
        print(f'   template {mean.shape}, components {comps.shape[0]}'
              + (f', explained {np.round(var * 100, 1).tolist()}%'
                 if len(var) else ''))
        print(f'   its own spots score median {float(np.median(d)):.3f} '
              f'(p10 {float(np.percentile(d, 10)):.3f})')
        if ref:
            print(f'   vs the analytic {ref.get("family")}: '
                  f'cosine {ref["cosine_to_measured"]:.3f}')
        print(f'   -> {os.path.basename(path)}')
        report['psf'] = {'path': path, 'n_spots': len(pos),
                         'explained_var': [float(v) for v in var],
                         'drift_median': float(np.median(d)),
                         'analytic_ref': ref}

    rp = os.path.join(a.out, 'report.json')
    with open(rp, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1)
    print(f'\nreport  {rp}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
