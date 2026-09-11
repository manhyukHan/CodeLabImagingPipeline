"""
Several bundles, one model -- and what differs between the experiments.

WHY. Bundles and verdicts stay per experiment, which is the point of
keeping them apart from the model: a model can be trained again from
any set of them. Early on that means one ch555 model per experiment;
later, one pooled model from all of them. Pooling has to keep the
experiments' cells apart in the validation split, say how the pooled
heads validate on EACH experiment, measure how a head trained on one
experiment transfers to another, and fit each experiment's PSF template
beside the pooled one -- otherwise 'one model' hides exactly the
difference a person wants to see.

Verified on the two real DNA ch555 bundles (MP58 and JP chr19), read
only; the run is written under D:/claude-tmp. Skips with a reason when
they are absent.

Run:  python tests/test_multibundle_training.py
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKS = [0, 0]
MP58 = 'G:/Seonghyeok/2025-11-30-MP58/review_bundles/DNA_ch555'
JP = 'G:/JP/2026-01-26-JP-chr19_downstream/review_bundles/DNA_ch555'
OUT = 'D:/claude-tmp/models/multibundle_test'


def check(name, ok, note=''):
    CHECKS[0] += 1
    CHECKS[1] += 1 if ok else 0
    print(('  ok   ' if ok else '  FAIL ') + name
          + (('  -- ' + note) if note else ''))


def test_source():
    print('the plumbing, by source')
    import inspect
    import tools.train_spotmodel as T
    from codelab_pipeline.training import classify as C
    src = inspect.getsource(T.main)
    check('the bundle argument takes several', "nargs='+'" in src)
    check('cells are qualified by bundle before the split',
          "r['group'] = (bi,) + tuple(r['group'])" in src)
    check('the pooled heads are validated per bundle, and a head trained '
          'on one bundle is scored on the others',
          '_per_bundle_val(' in src and "report['transfer'] = _transfer(" in src)
    check('a template per bundle, and their mutual cosine',
          "report['psf']['per_bundle'] = psf_per" in src
          and "report['psf']['between'] = psf_between" in src)
    check('the multispot calibration pools every bundle that has verdicts',
          'C.fit_multispot(bundles, template=tpl_shape)' in src
          and 'used.append(b)' in inspect.getsource(C.fit_multispot))
    check('the manifest records every bundle',
          "'bundles': bundles," in src)
    check('a model is a model of ONE channel: --channel leaves other '
          "channels' rows out, counted", "'--channel'" in src
          and "rb = [r for r in rb if int(r.get('channel', -1)) == int(a.channel)]" in src)
    check('the pooled multispot report names the banks the verdicts were '
          'judged with', "'banks': banks" in inspect.getsource(C.fit_multispot))


def test_pooled_multispot():
    print('multispot verdicts pooled from several bundles')
    if not (os.path.isdir(MP58) and os.path.isdir(JP)):
        check('both real bundles are present', False, 'skipped')
        return
    import tempfile
    from codelab_pipeline.training import classify as C
    a, ra = C.fit_multispot(MP58, template=(7, 7, 11))
    b, rb = C.fit_multispot(JP, template=(7, 7, 11))
    both, r2 = C.fit_multispot([JP, MP58], template=(7, 7, 11))
    have = [x for x, r in ((MP58, ra), (JP, rb)) if r.get('n')]
    check('the pool is the sum of the bundles that have verdicts, and '
          'names them',
          both is not None and r2['n'] == ra.get('n', 0) + rb.get('n', 0)
          and set(r2['bundles']) == set(have)
          and isinstance(r2.get('banks'), list),
          str((ra.get('n'), rb.get('n'), r2.get('n'), r2.get('bundles'))))
    empty = tempfile.mkdtemp(prefix='nomulti_', dir='D:/claude-tmp')
    none, r3 = C.fit_multispot([empty], template=(7, 7, 11))
    check('and a bundle without any says so',
          none is None and 'no multispot verdicts' in r3['skipped'])
    alone, r4 = C.fit_multispot([empty, MP58], template=(7, 7, 11))
    check('pooling with an empty bundle changes nothing',
          alone is not None and r4['n'] == ra['n'] and r4['bundles'] == [MP58]
          and r4['platt'] == ra['platt'])
    os.rmdir(empty)


def test_the_two_real_bundles():
    print('one model from MP58 + JP DNA ch555')
    if not (os.path.isdir(MP58) and os.path.isdir(JP)):
        check('both real bundles are present', False, 'skipped')
        return
    out = subprocess.run(
        [sys.executable, '-u', 'tools/train_spotmodel.py', MP58, JP,
         '--reviewer', 'Manhyuk', '--out', OUT, '--heads', 'linear,mlp',
         '--epochs', '200', '--channel', '555',
         '--genomic-resolution-kb', '50', '--genomic-resolution-kb', '200'],
        capture_output=True, text=True, timeout=1800,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    text = out.stdout
    check('the run finishes', out.returncode == 0, (out.stderr or text)[-600:])
    rp = os.path.join(OUT, 'report.json')
    if not os.path.exists(rp):
        check('a report was written', False, text[-400:])
        return
    r = json.load(open(rp, encoding='utf-8'))
    bs = r.get('bundles') or []
    check('the report names both bundles with their own labels',
          len(bs) == 2 and all(b['labels']['positive'] > 0 for b in bs)
          and ' + ' in r.get('bundle', ''), r.get('bundle'))
    tot = r['labels']['groups']
    check('cells of the two experiments never merge: pooled cells = sum',
          tot == sum(b['labels']['groups'] for b in bs),
          f"{tot} vs {[b['labels']['groups'] for b in bs]}")
    heads = r.get('heads') or {}
    ok_pb = all('per_bundle' in h and len(h['per_bundle']) == 2
                for h in heads.values() if 'error' not in h)
    check('each pooled head reports validation per bundle', ok_pb,
          str({k: list((h.get('per_bundle') or {}).keys()) for k, h in heads.items()}))
    tr = r.get('transfer') or {}
    check('a head trained on each bundle is scored on the other',
          all(any('pr_auc' in t for e in tr.get(h, {}).values()
                  for t in (e.get('to') or {}).values()) for h in ('linear', 'mlp')),
          json.dumps({h: {s: list((e.get('to') or {}).keys())
                          for s, e in tr.get(h, {}).items()} for h in tr})[:200])
    psf = r.get('psf') or {}
    per = psf.get('per_bundle') or []
    check('a PSF template per bundle, with its best fit and its cosine to '
          'the pooled one', len(per) == 2
          and all('best_fit' in e and 'cosine_to_pooled' in e for e in per),
          str([(e.get('bundle'), e.get('n_spots'), e.get('skipped')) for e in per]))
    check('and the two templates\' cosine to each other',
          len(psf.get('between') or {}) == 1, str(psf.get('between')))
    ms = r.get('multispot') or {}
    check('the multispot calibration pooled the bundles that have verdicts, '
          'and names their banks',
          set(ms.get('bundles') or []) <= {os.path.abspath(MP58), os.path.abspath(JP)}
          and 'platt' in ms and isinstance(ms.get('banks'), list),
          str((ms.get('bundles'), ms.get('banks'))))
    check("JP's two judged ch635 crops were left out of the ch555 model",
          r.get('channel') == 555 and all(
              b.get('channel') == 555 for b in r.get('bundles') or []))
    for line in text.splitlines():
        if line.startswith('   linear trained on') or line.startswith('   mlp    trained on') \
                or 'cosine to the pooled' in line or ' vs ' in line:
            print('    ' + line.strip())
    from codelab_pipeline.training import classify as C
    clf, why = C.load_best(OUT)
    check('and the run loads like any other', clf is not None
          and why['head'] in ('linear', 'mlp'), str(why.get('head')))
    from codelab_pipeline.localization import psf_bank as PB
    cands = PB.candidates(os.path.join(OUT, 'psf_bank.h5'))
    check('the bank carries a candidate template per bundle, tagged with its '
          'resolution', sorted(m['genomic_resolution_kb'] for _t, m in cands) == [50.0, 200.0]
          and all(m.get('sigma_z_um') for _t, m in cands), str([m for _t, m in cands]))
    cal = C.MultispotCalibration.load(os.path.join(OUT, 'psf_multispot.json'))
    check('and the calibration has a Platt per resolution where a bundle had '
          '150+ judged matches (MP58 does)', '50' in cal.per_resolution
          and cal.per_resolution['50']['n'] >= 150, str(list(cal.per_resolution)))


def main():
    test_source()
    test_pooled_multispot()
    test_the_two_real_bundles()
    print()
    print('%d/%d checks passed' % (CHECKS[1], CHECKS[0]))
    return 0 if CHECKS[1] == CHECKS[0] else 1


if __name__ == '__main__':
    sys.exit(main())
