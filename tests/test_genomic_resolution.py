"""
The genomic resolution: one field, into the config, the bundle, the
model and the tracer.

WHY. The fiducial is the whole genomic region the readouts trace, so its
apparent size tracks the DESIGN -- the kb of DNA a readout step spans --
not the optics. The two DNA ch555 bundles pooled first (MP58, 50 kb;
chr19 downstream) had fiducial templates of sigma_z 760 and 1072 nm, and
an MLP trained on one alone fell to PR-AUC 0.65 on the other. The target
span in Mb is the physical quantity but rarely a clean number; the step
in kb is, and scales with it. So it is typed once, on the Ingestion tab
under the voxel size, saved with the config, written into every bundle
manifest, appended at training as log10 when the pooled bundles differ
in it, and supplied at scoring time by the tracer from the same field.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_genomic_resolution.py
"""
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                          # noqa: E402

CHECKS = [0, 0]
_APP = None
MP58 = 'G:/Seonghyeok/2025-11-30-MP58/review_bundles/DNA_ch555'
JP = 'G:/JP/2026-01-26-JP-chr19_downstream/review_bundles/DNA_ch555'
OUT = 'D:/claude-tmp/models/resolution_test'


def check(name, ok, note=''):
    CHECKS[0] += 1
    CHECKS[1] += 1 if ok else 0
    print(('  ok   ' if ok else '  FAIL ') + name
          + (('  -- ' + note) if note else ''))


def test_field_and_config():
    print('the field, the config map, the tracing params')
    global _APP
    from PyQt5 import QtWidgets
    _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from windows.main_window import MainWindow
    w = MainWindow(None)
    ip = w.ui.IngestionPanel
    check('empty is unknown, not zero', ip.genomic_resolution_kb() is None
          and 'unknown' in ip.ResolutionStatusLabel.text())
    ip.GenomicResolutionLineEdit.setText('50')
    check('typed is not yet in force', ip.genomic_resolution_kb() is None)
    check('Apply commits it, and says so',
          ip.apply_genomic_resolution() and ip.genomic_resolution_kb() == 50.0
          and 'in force: 50 kb' in ip.ResolutionStatusLabel.text())
    ip.GenomicResolutionLineEdit.setText('abc')
    check('a non-number is REFUSED and the value in force stays',
          not ip.apply_genomic_resolution() and ip.genomic_resolution_kb() == 50.0
          and 'NOT applied' in ip.ResolutionStatusLabel.text())
    ip.GenomicResolutionLineEdit.setText('')
    check('empty + Apply is unknown again',
          ip.apply_genomic_resolution() and ip.genomic_resolution_kb() is None)
    check('loading a config commits it like the voxel size',
          'IngestionPanel.apply_genomic_resolution()' in inspect.getsource(
              MainWindow._apply_config_params))
    m = MainWindow._CONFIG_PARAM_MAP['acquisition']
    check('it is config-shaped state beside the voxel size',
          m.get('genomic_resolution_kb') == ('IngestionPanel',
                                             'GenomicResolutionLineEdit'))
    src = inspect.getsource(MainWindow)
    check('the window injects it into the tracing panel like the voxel size',
          src.count('chp._genomic_resolution_kb = self._genomic_resolution_kb()') == 2)
    check('and hands it to the Build model window',
          'genomic_resolution_kb=self._genomic_resolution_kb' in src)
    ip.GenomicResolutionLineEdit.setText('200')
    ip.apply_genomic_resolution()
    check("the window reads the committed value", w._genomic_resolution_kb() == 200.0)
    chp = w.ui.ChromatinTracingPanel
    chp._genomic_resolution_kb = w._genomic_resolution_kb()
    check("the tracing panel's params carry it",
          chp.params().get('genomic_resolution_kb') == 200.0)
    for cfg, kb in (('configs/2025-11-30-MP58-testbox.xml', '50'),
                    ('configs/2026-07-22-DI-DNA-HoxA.xml', '5'),
                    ('configs/2025-12-08-JY-Chr19_Downstream.xml', '200')):
        if os.path.exists(cfg):
            check(f'{os.path.basename(cfg)} records {kb} kb',
                  f'genomic_resolution_kb="{kb}"' in open(cfg, encoding='utf-8').read())


def test_bundle_and_dialog():
    print('the bundle manifest and the Build model window')
    import tools.build_bundle as BB
    src = inspect.getsource(BB.main)
    check('build_bundle takes it and records it',
          "'--genomic-resolution-kb'" in src and 'genomic_resolution_kb=' in src)
    from ui.model_build_dialog import ModelBuildDialog
    from tests.test_model_build_dialog import SOURCES, STORES, REPO, set_checked
    recall = ModelBuildDialog._recall_bundle_path
    ModelBuildDialog._recall_bundle_path = lambda self: None
    try:
        d = ModelBuildDialog(SOURCES, [7, 14], STORES.get, REPO,
                             genomic_resolution_kb=lambda: 50.0)
    finally:
        ModelBuildDialog._recall_bundle_path = recall
    set_checked(d, lambda m, h, c: m == 'RNA')
    d.FovListLineEdit.setText('7,14')
    d.BundlePathLineEdit.setText('D:/claude-tmp/whatever')
    cmds = d.build_commands()
    check('every build command carries the resolution',
          cmds and all(c[c.index('--genomic-resolution-kb') + 1] == '50'
                       for c in cmds))
    d._genomic_resolution_kb = lambda: None
    check('and none when it is unknown',
          all('--genomic-resolution-kb' not in c for c in d.build_commands()))


def test_feature_contract():
    print('the feature contract: NAMES plus context, and the engine')
    from codelab_pipeline.training import features as F, classify as C
    check('one context feature, named', F.CONTEXT_NAMES == ('log10_genomic_resolution_kb',))
    v = F.context_vector({'genomic_resolution_kb': 100.0})
    check('log10: 100 kb -> 2', v.shape == (1,) and abs(v[0] - 2.0) < 1e-12)
    raised = ''
    try:
        F.context_vector({})
    except ValueError as exc:
        raised = str(exc)
    check('missing -> a refusal that says where to type it',
          'Ingestion tab' in raised, raised[:60])
    # a head trained on 17 columns, scored with context; one on 16, without
    rng = np.random.RandomState(0)
    n = 240
    X16 = rng.normal(size=(n, len(F.NAMES)))
    y = (X16[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    groups = [(i % 24,) for i in range(n)]
    res = np.where(np.arange(n) % 2 == 0, 50.0, 200.0)
    X17 = np.hstack([X16, np.log10(res)[:, None]])
    clf17, _r = C.train(X17, None, y, groups, head='linear', epochs=30,
                        verbose=False,
                        feature_names=list(F.NAMES) + list(F.CONTEXT_NAMES))
    clf16, _r = C.train(X16, None, y, groups, head='linear', epochs=30,
                        verbose=False)
    check('a head records what it was trained on',
          clf17.context_names == F.CONTEXT_NAMES and clf16.context_names == ())
    p = clf17.score(clf17.with_context(X16[:5], {'genomic_resolution_kb': 50}))
    check('with_context appends the column and the head scores',
          p.shape == (5,) and np.all((p > 0) & (p < 1)))
    check('a head without context leaves X alone',
          clf16.with_context(X16[:5], None).shape == (5, len(F.NAMES)))
    raised = ''
    try:
        clf17.with_context(X16[:5], None)
    except ValueError as exc:
        raised = str(exc)
    check('and a head with context refuses to score without it',
          'Ingestion tab' in raised)
    with tempfile.TemporaryDirectory(dir='D:/claude-tmp') as d:
        path = clf17.save(os.path.join(d, 'spot_classifier_linear.json'))
        back = C.SpotClassifier.load(path)
        check('it round-trips through disk with its 17 inputs',
              back.feature_names == clf17.feature_names
              and np.allclose(back.score(back.with_context(
                  X16[:5], {'genomic_resolution_kb': 50})), p))
        rep = {'heads': {'linear': {'path': path, 'val_pr_auc': 0.9}}}
        json.dump(rep, open(os.path.join(d, 'report.json'), 'w'))
        got, why = C.load_best(d)
        check('load_best accepts NAMES + known context', why['head'] == 'linear')
        doc = json.load(open(path))
        doc['meta']['features'] = list(F.NAMES) + ['something_else']
        json.dump(doc, open(path, 'w'))
        try:
            C.load_best(d)
            refused = ''
        except ValueError as exc:
            refused = str(exc)
        check('and still refuses a context it does not know',
              'features' in refused)
    from codelab_pipeline.localization import psfmatcher as PM, tracing_v2 as T2
    src = inspect.getsource(PM.PsfMatcherV3Engine.localize)
    check('the engine scores with its context',
          'self.classifier.with_context(feats, self.context)' in src)
    p2 = T2.V2Params.from_panel({'engine': T2.ROUTE_V3, 'v2': {}, 'v3': {},
                                 'genomic_resolution_kb': 50.0}, None)
    check('V2Params carries it from the panel and gives it to its engines',
          p2.genomic_resolution_kb == 50.0
          and p2.context() == {'genomic_resolution_kb': 50.0}
          and '_give_context(self._fiducial_engine)' in inspect.getsource(T2.V2Params))


def test_training_paths():
    print('training: a feature when it varies, metadata when it does not')
    import tools.train_spotmodel as T
    src = inspect.getsource(T.main)
    check('one value -> recorded, not a feature; two -> the 17th column',
          "len(distinct) < 2" in src and "list(F.NAMES) + list(F.CONTEXT_NAMES)" in src
          and 'feature_names=feature_names' in src)
    check('the transfer heads read the box features only',
          'X[src][:, :nb]' in inspect.getsource(T._transfer))
    if not (os.path.isdir(MP58) and os.path.isdir(JP)):
        check('both real bundles are present', False, 'skipped')
        return
    out = subprocess.run(
        [sys.executable, '-u', 'tools/train_spotmodel.py', MP58, JP,
         '--reviewer', 'Manhyuk', '--out', OUT, '--heads', 'linear,mlp',
         '--epochs', '150', '--genomic-resolution-kb', '50',
         '--genomic-resolution-kb', '200'],
        capture_output=True, text=True, timeout=1800,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    check('the pooled run with two resolutions finishes', out.returncode == 0,
          (out.stderr or out.stdout)[-500:])
    rp = os.path.join(OUT, 'report.json')
    if not os.path.exists(rp):
        return
    r = json.load(open(rp, encoding='utf-8'))
    check('the resolution became the 17th feature',
          r.get('features', [])[-1:] == ['log10_genomic_resolution_kb']
          and r.get('context', {}).get('as_feature') is True,
          str(r.get('context')))
    doc = json.load(open(os.path.join(OUT, 'spot_classifier_mlp.json')))
    check('the saved head has 17 inputs', len(doc['meta']['features']) == 17)
    for line in out.stdout.splitlines():
        if line.startswith('context') or 'val PR-AUC' in line[:24]:
            print('    ' + line.strip())
    from codelab_pipeline.localization import engine as E
    eng = E.make_engine('v3-psfmatcher', model_dir=OUT)
    rng = np.random.RandomState(1)
    cube = rng.normal(300, 5, (31, 31, 40))
    cube[15, 15, 20] += 3000
    eng.context = {}
    raised = ''
    try:
        eng.localize(cube, seed_yxz=None, n_max=None)
    except ValueError as exc:
        raised = str(exc)
    check('that model refuses to run without the resolution',
          'Ingestion tab' in raised, raised[:70])
    eng.context = {'genomic_resolution_kb': 50.0}
    got = eng.localize(cube, seed_yxz=None, n_max=None)
    check('and runs with it', isinstance(got, list))


def main():
    test_field_and_config()
    test_bundle_and_dialog()
    test_feature_contract()
    test_training_paths()
    print()
    print('%d/%d checks passed' % (CHECKS[1], CHECKS[0]))
    return 0 if CHECKS[1] == CHECKS[0] else 1


if __name__ == '__main__':
    sys.exit(main())
