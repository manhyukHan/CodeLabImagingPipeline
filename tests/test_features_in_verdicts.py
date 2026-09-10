"""
The classifier's input lives in the verdict, so M1 retrains without pixels.

WHY. A retrain read every shard to cut a 15x15x25 box per label and
compute 16 features from it -- the classifier's actual input. Spot Check
has those very pixels on screen when a verdict is made, so it can store
the 16 numbers in the record; from then on the classifier retrains from
verdicts alone, and a bundle really is a cache. Two things stay
pixel-bound and are said so: the PSF bank (an aligned average of the
boxes -- not a feature) and the conv head.

AND THE VECTORS ARE A CORRUPTION GATE. Two reviewers who judged the same
candidate against the same pixels stored the same 16 numbers; a
disagreement means one of them was looking at different pixels under
the same key. labels() files such a candidate as 'inconsistent' and
nobody trains on it.

Everything here runs on a REAL shard from the two-channel bundle at
D:/claude-tmp/twoch, through Spot Check's own load/commit path; skips
with a reason when it is absent.

Run:  python tests/test_features_in_verdicts.py
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline.training import bundle as B          # noqa: E402
from codelab_pipeline.training import dataset as D         # noqa: E402
from codelab_pipeline.training import features as F        # noqa: E402
from codelab_pipeline.training import verdicts as V        # noqa: E402

CHECKS = [0, 0]
TWOCH = 'D:/claude-tmp/twoch'
MODEL = 'D:/models/mp58_rna'
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check(name, ok, note=''):
    CHECKS[0] += 1
    CHECKS[1] += 1 if ok else 0
    print(('  ok   ' if ok else '  FAIL ') + name
          + (('  -- ' + note) if note else ''))


def a_shard():
    """One real shard with candidates, or None."""
    if not os.path.isdir(TWOCH):
        return None
    for p in B.shard_paths(TWOCH):
        idx = B.read_index(p)
        for row in idx:
            if int(row['n_candidates']) >= 6:
                return p, row
    return None


def test_featurize_is_boxes():
    print('featurize_crop is boxes(), by construction and by number')
    got = a_shard()
    if got is None:
        check('a real shard is present', False, 'skipped')
        return
    shard, row = got
    stack, _m, cands, _w = B.read_crop(shard, row['key'])
    pts = [(float(c['y']), float(c['x']), float(c['z'])) for c in cands[:8]]
    per, b0, sg = D.featurize_crop(stack, pts)
    check('one vector per point, len(NAMES) wide',
          len(per) == 8 and all(len(f['feat']) == len(F.NAMES) for f in per))
    rows = [dict(key=str(row['key']), shard=shard, y=y, x=x, z=z, label=1,
                 group=(1, 1)) for (y, x, z) in pts]
    kept, cores, cols = D.boxes(rows)
    X = F.many(cores, cols, kept)
    keep_ix = [i for i, f in enumerate(per) if f['border_frac'] <= D.MAX_BORDER_FRAC]
    check('boxes() keeps exactly the points inside the border gate',
          len(kept) == len(keep_ix))
    d = max(float(np.max(np.abs(np.asarray(per[i]['feat']) - X[k])))
            for k, i in enumerate(keep_ix)) if keep_ix else 0.0
    check('and its features are featurize_crop\'s to the last bit',
          d == 0.0, f'max|diff| {d:.2e}')


def _judge(bundle_dir, reviewer, accepted, perturb=None, which=0):
    """One page judged through Spot Check's own PassFail path.

    which: 0 = the crop with the most candidates, 1 = the next. A shard's
    first row can be a cell that had none, and a page of nothing judges
    nothing."""
    from spotcheck import modes as M
    shard = B.shard_paths(bundle_dir)[0]
    rows_ = sorted(B.read_index(shard),
                   key=lambda r: -int(r['n_candidates']))
    row = rows_[which]
    mode = M.PassFail()
    log = V.VerdictLog(bundle_dir, reviewer)
    s = mode.load((shard, row, 0, list(range(6))), log, 6, None)
    s['accepted'] = set(accepted)
    s['added'] = []
    if perturb is not None:
        # A DIFFERENT PICTURE UNDER THE SAME KEY. Not a gain: the features
        # are in sigma units after background subtraction, so a global
        # x1.5 leaves every one of them EXACTLY as it was (the first draft
        # of this test learned that the hard way). A lateral shift moves
        # every spot off its box centre, which no normalisation undoes.
        st = np.asarray(s['stack'], float).copy()
        s['stack'] = np.roll(st, int(perturb), axis=0)
    rec = mode.commit(log, s, 1.0)
    return rec, s


def test_end_to_end_without_the_shard():
    print('a verdict carries its vectors; the shard can then go')
    got = a_shard()
    if got is None:
        check('a real shard is present', False, 'skipped')
        return
    shard, _row = got
    d = tempfile.mkdtemp(prefix='featverd_')
    try:
        shutil.copy(shard, d)
        rec, s = _judge(d, 'alice', accepted=[0, 2, 4])
        e0 = rec['shown'][0]
        check('the record stores a 16-vector per shown candidate',
              all(len(e['feat']) == len(F.NAMES) for e in rec['shown']))
        check('and the z it was cut at, the border fraction, the depth-edge',
              all(k in e0 for k in ('z_used', 'border_frac',
                                    'planes_from_stack_end')))
        check('and what the numbers mean',
              rec['feat_meta']['names_sha'] == F.names_sha()
              and rec['feat_meta']['r'] == D.DEFAULT_R
              and 'bg' in rec['feat_meta'])
        size = len(json.dumps(rec))
        check('a six-candidate record stays small', size < 6000, f'{size} B')

        # WITH the shard: rows cut from pixels.
        rows_cut = D.rows(d)
        kept_c, X_cut, cores_c = D.features_for_rows(
            [r for r in rows_cut if r['label'] in (0, 1)])
        check('with the shard, rows are cut from pixels',
              len(kept_c) == 6 and all(c is not None for c in cores_c))

        # WITHOUT the shard: the same rows, from the verdict.
        os.remove(os.path.join(d, os.path.basename(shard)))
        rows_v = D.rows(d)
        check('without the shard the rows are still there',
              len(rows_v) == 6 and all(r['shard'] is None for r in rows_v))
        s_ = D.summary(rows_v)
        check('summary counts them as feature-only',
              s_['feature_only'] == 6 and s_['positive'] == 3
              and s_['negative'] == 3)
        kept_v, X_feat, cores_v = D.features_for_rows(
            [r for r in rows_v if r['label'] in (0, 1)])
        check('features_for_rows uses the stored vectors',
              len(kept_v) == 6 and all(c is None for c in cores_v))
        # same order: sort both by (y, x)
        oc = np.argsort([(r['y'], r['x']) for r in kept_c], axis=0)[:, 0]
        ov = np.argsort([(r['y'], r['x']) for r in kept_v], axis=0)[:, 0]
        dmax = float(np.max(np.abs(X_cut[oc] - X_feat[ov])))
        check('and they equal the pixel-cut features to the stored precision',
              dmax < 2e-5, f'max|diff| {dmax:.2e}')
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_two_reviewers_and_the_gate():
    print('two reviewers: agreement passes, a different picture is caught')
    got = a_shard()
    if got is None:
        check('a real shard is present', False, 'skipped')
        return
    shard, _row = got
    d = tempfile.mkdtemp(prefix='featgate_')
    try:
        shutil.copy(shard, d)
        _judge(d, 'alice', accepted=[0, 2, 4])
        _judge(d, 'bob', accepted=[0, 2, 4])
        lab = list(V.labels(d).values())[0]
        check('two reviewers on the same pixels: nothing inconsistent',
              lab['inconsistent'] == [] and len(lab['positive']) == 3
              and len(lab['feat']) == 6)

        # Carol judged a different picture under the same key.
        _judge(d, 'carol', accepted=[0, 2, 4], perturb=3)
        lab = list(V.labels(d).values())[0]
        check('a reviewer whose vectors disagree makes the candidates '
              'INCONSISTENT', len(lab['inconsistent']) == 6, str(len(lab['inconsistent'])))
        check('and they leave the training buckets',
              lab['positive'] == [] and lab['negative'] == [])
        s_ = D.summary(D.rows(d))
        check('summary says how many', s_['inconsistent'] == 6
              and s_['positive'] == 0)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_training_from_verdicts_alone():
    print('train_spotmodel on a shard-less bundle')
    got = a_shard()
    if got is None or not os.path.isdir(MODEL):
        check('a real shard and the shipped run are present', False,
              'skipped')
        return
    import subprocess
    shard, _row = got
    d = tempfile.mkdtemp(prefix='feattrain_')
    out = tempfile.mkdtemp(prefix='feattrain_run_')
    try:
        shutil.copy(shard, d)
        # TWO CELLS, so the cell-held-out split has one to train on and
        # one to validate on; a single group puts everything in val.
        _judge(d, 'alice', accepted=[0, 2, 4], which=0)
        _judge(d, 'alice', accepted=[1, 3], which=1)
        os.remove(os.path.join(d, os.path.basename(shard)))
        r = subprocess.run(
            [sys.executable, '-u', os.path.join(REPO, 'tools',
                                                'train_spotmodel.py'),
             d, '--reviewer', 'alice', '--out', out, '--epochs', '20',
             '--min-groups', '1', '--bank-from', MODEL],
            capture_output=True, text=True, timeout=600)
        failed = [t for t in r.stdout.splitlines() if 'FAILED' in t]
        check('it trains', r.returncode == 0 and not failed,
              ' | '.join(failed)[:300] or r.stdout[-200:])
        check('every row came from a stored vector',
              '0 cut from shards, 12 from vectors stored' in r.stdout,
              str([t for t in r.stdout.splitlines() if t.startswith('rows')][:1]))
        check('the heads were written',
              os.path.exists(os.path.join(out, 'spot_classifier_linear.json'))
              and os.path.exists(os.path.join(out, 'spot_classifier_mlp.json')))
        check('the bank was REUSED, and it says so',
              'REUSED the bank of' in r.stdout
              and os.path.exists(os.path.join(out, 'psf_bank.h5')))
    finally:
        shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)


def test_wiring():
    print('wiring')
    import inspect
    from spotcheck import modes as M
    import tools.train_spotmodel as T
    src = inspect.getsource(M.PassFail.commit)
    check('Spot Check featurises from the pixels on screen at commit',
          'featurize_crop(st' in src and 'features=feats' in src)
    tsrc = inspect.getsource(T.main)
    check('training takes stored vectors where it has them',
          'features_for_rows' in tsrc)
    check('the conv head is skipped when any row has no pixels',
          "conv head skipped" in tsrc)
    check('the bank says it cannot use vectors',
          'averaged PIXELS' in tsrc)


def main():
    test_featurize_is_boxes()
    test_end_to_end_without_the_shard()
    test_two_reviewers_and_the_gate()
    test_training_from_verdicts_alone()
    test_wiring()
    print()
    print('%d/%d checks passed' % (CHECKS[1], CHECKS[0]))
    return 0 if CHECKS[1] == CHECKS[0] else 1


if __name__ == '__main__':
    sys.exit(main())
