"""
The Build model window: sources, FOVs, bundle, review, train, result.

WHY. 'Make new model...' ended at step two of three -- the training
command was printed for a person to type -- and a bundle was one
channel wide, so an experiment with two readout channels meant the
whole chain twice into two folders. The window owns every step and
runs each long one in the background.

Everything here that reads a bundle reads the REAL two-channel bundle
built from G:/Seonghyeok/2025-11-30-MP58/RNA into D:/claude-tmp/twoch
(116 crops in ch555 and 116 in ch635, FOV007 Hyb_101), and skips with
a reason when it is absent. The report rendering reads the shipped
model under D:/models/mp58_rna.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_model_build_dialog.py
"""
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5 import QtCore, QtWidgets                          # noqa: E402

from ui.model_build_dialog import ModelBuildDialog, _parse_fovs  # noqa: E402

CHECKS = [0, 0]
TWOCH = 'D:/claude-tmp/twoch'
MODEL = 'D:/models/mp58_rna'
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCES = [
    ('RNA', 'Hyb_101', 'geneA', 635, False),
    ('RNA', 'Hyb_101', 'geneA', 555, True),
    ('RNA', 'Hyb_103', 'geneB', 635, False),
    ('RNA', 'Hyb_103', 'geneB', 555, True),
    ('DNA', 'Hyb_201', '', 635, False),
]
STORES = {'RNA': 'G:/Seonghyeok/2025-11-30-MP58/RNA', 'DNA': 'D:/nope/DNA'}


def check(name, ok, note=''):
    CHECKS[0] += 1
    CHECKS[1] += 1 if ok else 0
    print(('  ok   ' if ok else '  FAIL ') + name
          + (('  -- ' + note) if note else ''))


_APP = None


def make(fov_pool=(7, 8, 9, 14, 19)):
    # HELD, NOT DISCARDED. `QApplication.instance() or QApplication([])`
    # as a bare expression creates the app and drops it on the same
    # line; the QDialog built next has no live application and Qt
    # fail-fasts (0xC0000409) inside super().__init__ with no Python
    # traceback at all. Found by tracing; the dialog was never at fault.
    global _APP
    _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    d = ModelBuildDialog(SOURCES, list(fov_pool), STORES.get, REPO)
    # START FROM AN EMPTY PATH. The dialog recalls the last bundle path
    # from spotcheck/build_model_last.json in THIS repo, which the real
    # app writes -- so with a build running on this machine every test
    # here inherited G:/.../DNA_ch555 and the path-suggestion check
    # failed on the app's memory, not on the code. The recall itself is
    # tested in test_memory_and_target against its own file.
    d.BundlePathLineEdit.setText('')
    return d


def set_checked(d, pred):
    lw = d.SourceListWidget
    for i in range(lw.count()):
        it = lw.item(i)
        m, h, c = it.data(QtCore.Qt.UserRole)
        it.setCheckState(QtCore.Qt.Checked if pred(m, h, c)
                         else QtCore.Qt.Unchecked)


def test_sources():
    print('1  sources')
    d = make()
    lw = d.SourceListWidget
    check('one row per (modality, hybe, channel)', lw.count() == len(SOURCES))
    check("rows read like the Analysis tab's",
          lw.item(0).text() == 'RNA | Hyb_101 (geneA) | ch635'
          and lw.item(1).text().endswith('(fiducial)'),
          lw.item(1).text())
    check('every row starts unchecked',
          all(lw.item(i).checkState() == QtCore.Qt.Unchecked
              for i in range(lw.count())))
    check('UserRole carries [modality, folder, channel]',
          lw.item(0).data(QtCore.Qt.UserRole) == ['RNA', 'Hyb_101', 635])

    # Check Modality+Channel
    d.BulkModalityComboBox.setCurrentText('RNA')
    d.BulkChannelComboBox.setCurrentText('635')
    d._check_modality_channel()
    check('Check Modality+Channel checks every RNA ch635 row',
          d.checked_sources() == [('RNA', 'Hyb_101', 635),
                                  ('RNA', 'Hyb_103', 635)])
    # Check/Uncheck Selected
    lw.item(1).setSelected(True)
    d._set_selected(True)
    check('Check Selected adds the selected row',
          ('RNA', 'Hyb_101', 555) in d.checked_sources())
    d._set_selected(False)
    check('Uncheck Selected removes it again',
          ('RNA', 'Hyb_101', 555) not in d.checked_sources())

    # ONE STORE PER BUNDLE
    lw.item(4).setCheckState(QtCore.Qt.Checked)      # DNA
    check('two modalities -> no store, with the reason logged',
          d._store() is None
          and any('ONE store' in d.LogListWidget.item(i).text()
                  for i in range(d.LogListWidget.count())))
    lw.item(4).setCheckState(QtCore.Qt.Unchecked)
    check('one modality -> its store', d._store() == STORES['RNA'])
    check('a bundle path was suggested beside the store',
          'review_bundles' in d.BundlePathLineEdit.text()
          and 'ch635' in d.BundlePathLineEdit.text(),
          d.BundlePathLineEdit.text())


def test_fovs():
    print('2  FOVs')
    check('parser: ranges, lists, spaces, dupes',
          _parse_fovs('1-3, 7 7,10-11') == [1, 2, 3, 7, 10, 11])
    d = make(fov_pool=(7, 8, 9, 14, 19))
    d.RandomCountLineEdit.setText('3')
    # LEFT FILLS RIGHT from the FOVs that HAVE masks, never from the
    # whole declared pool: 8 and 9 carry none here.
    d._fill_fovs(3, [(7, 5), (14, 8), (19, 3)])
    got = d.fovs()
    check('the draw fills the right box from the masked FOVs only',
          sorted(got) == got and len(got) == 3
          and set(got) <= {7, 14, 19}, str(got))
    check('the left box is untouched by the draw',
          d.RandomCountLineEdit.text() == '3')
    d._fill_fovs(9, [(7, 5), (14, 8)])
    check('asking for more than exist gives what exists, and says so',
          d.fovs() == [7, 14]
          and any('only 2 had masks' in d.LogListWidget.item(i).text()
                  for i in range(d.LogListWidget.count())))
    # RIGHT NEVER CHANGES LEFT
    d.FovListLineEdit.setText('1-4')
    check('hand-editing the right box is honoured', d.fovs() == [1, 2, 3, 4])
    check('and does not touch the left', d.RandomCountLineEdit.text() == '3')


def test_build_commands():
    print('3  bundle commands')
    d = make(fov_pool=(7, 8, 9, 14, 19))
    set_checked(d, lambda m, h, c: m == 'RNA')
    d.FovListLineEdit.setText('7,14')
    d.BundlePathLineEdit.setText('D:/claude-tmp/whatever')
    cmds = d.build_commands()
    check('one command per checked CHANNEL', len(cmds) == 2)
    ch = [c[c.index('--channel') + 1] for c in cmds]
    check('channels in order', ch == ['555', '635'], str(ch))
    hy = [c[c.index('--hybes') + 1] for c in cmds]
    check("each channel gets its own hybes", hy == ['Hyb_101,Hyb_103'] * 2,
          str(hy))
    fv = {c[c.index('--fovs') + 1] for c in cmds}
    check('the SAME explicit FOV list goes to every channel',
          fv == {'7,14'}, str(fv))
    pool = {c[c.index('--fov-pool') + 1] for c in cmds}
    check("the declared pool is passed, not build_bundle's 1-41 default",
          pool == {'7,8,9,14,19'}, str(pool))
    check('every command targets the one bundle directory',
          all(c[c.index('--out') + 1] == 'D:/claude-tmp/whatever'
              for c in cmds))
    check('every command runs the interpreter unbuffered',
          all(c[1] == '-u' for c in cmds))
    wk = {c[c.index('--workers') + 1] for c in cmds}
    check('the worker count is a setting, default 6',
          wk == {'6'} and d.WorkersSpinBox.value() == 6, str(wk))
    d.WorkersSpinBox.setValue(4)
    check('and it is what the command carries',
          all(c[c.index('--workers') + 1] == '4' for c in d.build_commands()))
    # A COUNT OF CROPS FOR THE WHOLE BUNDLE, shared out by source. Four
    # RNA sources are checked here (two hybes x two channels), so each
    # channel's run gets half of the 10,000 and splits it over its two
    # hybes: 2,500 crops a source.
    nc = [c[c.index('--n-crops') + 1] for c in cmds]
    check('10,000 crops, default, shared out to each channel by its hybes',
          nc == ['5000', '5000'] and d.CropsSpinBox.value() == 10000, str(nc))
    note = d.CropsNote.text()
    check('and the note says what it comes to per source and per FOV',
          '10,000 crops' in note and '2,500 a source' in note
          and '1,250 cells a FOV a source' in note, note)
    d.CropsSpinBox.setValue(0)
    check('0 reads "all", every command carries 0, the note says every cell',
          d.CropsSpinBox.text() == 'all'
          and all(c[c.index('--n-crops') + 1] == '0'
                  for c in d.build_commands())
          and 'every cell' in d.CropsNote.text(), d.CropsNote.text())
    d.CropsSpinBox.setValue(10000)

    d.FovListLineEdit.setText('')
    check('no FOVs -> no commands, with a reason', d.build_commands() == [])
    d.FovListLineEdit.setText('7')
    d.BundlePathLineEdit.setText('')
    check('no path -> no commands', d.build_commands() == [])


def test_train_during_build():
    """A bundle is usable while it is being built -- every shard on disk
    is complete and .h5.part is never listed -- so training and the
    build are independent children with independent gates, and
    training never touches the build's progress bar."""
    print('train during a build')
    d = make()

    class Fake:
        def deleteLater(self):
            pass

    d._build_worker = Fake()
    d._set_build_busy(True)
    check('a running build leaves Train enabled',
          not d.BuildBundlePushButton.isEnabled()
          and d.TrainPushButton.isEnabled())
    check('and only a second BUILD is refused',
          d._build_busy() and not d._train_busy())
    d._on_progress(7, 1380)
    d._set_train_busy(True)
    check("training shows its own indicator and leaves the build's bar alone",
          not d.TrainProgressBar.isHidden() and d.TrainProgressBar.maximum() == 0
          and d.ProgressBar.maximum() == 1380 and d.ProgressBar.value() == 7
          and not d.TrainPushButton.isEnabled()
          and not d.BuildBundlePushButton.isEnabled())
    d._train_worker = Fake()
    check('a second training run is refused while the build still is',
          d._train_busy() and d._build_busy())
    d._set_train_busy(False)
    d._release_train_worker()
    check("training's end frees Train only, and the build's bar is untouched",
          d.TrainPushButton.isEnabled() and not d.BuildBundlePushButton.isEnabled()
          and d.ProgressBar.value() == 7 and d.ProgressBar.maximum() == 1380
          and d.TrainProgressBar.isHidden() and d._train_worker is None)
    d._set_build_busy(False)
    d._release_build_worker()
    check("the build's end frees Build and resets its bar",
          d.BuildBundlePushButton.isEnabled() and d.ProgressBar.maximum() == 1
          and d._build_worker is None)
    d._build_worker = Fake()
    d.close()
    check('closing while a build runs says so and hides',
          d.isHidden() and any('build continue' in d.LogListWidget.item(i).text()
                               for i in range(d.LogListWidget.count())))
    d._build_worker = None

    import inspect
    src = inspect.getsource(ModelBuildDialog)
    check('no single shared slot is left',
          'self._worker' not in src and '_set_busy(' not in src
          and '_release_worker(' not in src)
    check('_train gates on the train slot, _build_bundle on the build slot',
          'if self._train_busy():' in inspect.getsource(ModelBuildDialog._train)
          and 'if self._build_busy():'
          in inspect.getsource(ModelBuildDialog._build_bundle))
    import tools.build_bundle as BB
    check('the build manifest is rewritten atomically, so a mid-build '
          'reader never sees half of it',
          inspect.getsource(BB.main).count('_write_json_atomic(mpath') == 2
          and 'os.replace(tmp, path)' in inspect.getsource(BB._write_json_atomic))


def test_review_status():
    print('4  review status on the real two-channel bundle')
    if not os.path.isdir(TWOCH):
        check('the two-channel bundle is present', False,
              'skipped: ' + TWOCH + ' absent')
        return
    d = make()
    d.BundlePathLineEdit.setText(TWOCH)
    st = d.review_status()
    check('both channels are seen', sorted(st['channels']) == [555, 635],
          str(st['channels']))
    check('all crops are counted', st['crops'] == 232, str(st['crops']))
    check('FOV and hybe come from the shard indexes',
          st['fovs'] == {7} and st['hybes'] == {'Hyb_101'})
    check('an unreviewed bundle reports nobody, not an error',
          st['passfail']['crops'] == 0 and st['passfail']['reviewers'] == [])
    d._refresh_review()
    txt = d.ReviewStatusLabel.text()
    check('the label says what is there and who has reviewed',
          '232 crops' in txt and 'ch555, ch635' in txt and 'nobody yet' in txt,
          txt.splitlines()[0])
    d.BundlePathLineEdit.setText('D:/claude-tmp/does-not-exist')
    check('a missing directory is None, not a crash',
          d.review_status() is None)


def test_train_command():
    print('5  train command')
    d = make()
    d.BundlePathLineEdit.setText(TWOCH if os.path.isdir(TWOCH) else REPO)
    # The dialog recalls the last session's bundle at startup and, when
    # that bundle has exactly one reviewer, prefills the name -- so 'no
    # reviewer' has to be made true rather than assumed.
    d.ReviewerLineEdit.setText('')
    d.TargetRunComboBox.setCurrentIndex(0)
    check('no reviewer -> no command, with the reason',
          d.train_command() is None
          and any('Name the reviewer' in d.LogListWidget.item(i).text()
                  for i in range(d.LogListWidget.count())))
    d.ReviewerLineEdit.setText('tester')
    cmd = d.train_command()
    check('the command names the bundle and the reviewer',
          cmd is not None and cmd[3] == d.bundle_dir()
          and cmd[cmd.index('--reviewer') + 1] == 'tester')
    check('the run is NAMED, not timestamped: --out defaults to the reviewer',
          cmd[cmd.index('--out') + 1] == 'tester')
    d.RunNameLineEdit.setText('mp58_v2')
    cmd = d.train_command()
    check('a run name overrides it', cmd[cmd.index('--out') + 1] == 'mp58_v2')
    check('the button says new run when the name is free',
          'new run mp58_v2' in d.TrainPushButton.text(),
          d.TrainPushButton.text())
    d.RunNameLineEdit.setText('')
    if os.path.isdir(TWOCH):
        check("the store comes from the bundle's own manifest",
              '--storage-path' in cmd
              and cmd[cmd.index('--storage-path') + 1]
              == 'G:/Seonghyeok/2025-11-30-MP58/RNA')
    check('--set-default only when asked', '--set-default' not in cmd)
    d.SetDefaultCheckBox.setChecked(True)
    check('and present when asked', '--set-default' in d.train_command())


def test_render_report():
    print('6  the result view')
    if not os.path.isdir(MODEL):
        check('the shipped model is present', False, 'skipped')
        return
    txt = ModelBuildDialog.render_report(MODEL)
    check('labels line', 'labels' in txt and 'positive' in txt)
    check('every head has a line with its validation numbers',
          'linear' in txt and 'mlp' in txt and 'PR-AUC' in txt)
    check('the multispot calibration is reported',
          'multispot' in txt and 'pillars' in txt)
    check('the PSF is reported', 'psf' in txt and 'confirmed spots' in txt)
    bad = ModelBuildDialog.render_report('D:/claude-tmp/no-such-run')
    check('a missing report is a message, not a crash',
          'no readable report.json' in bad)


def test_close_hides():
    print('closing')
    d = make()
    d.show()
    d.close()
    check('close hides the window rather than destroying it',
          not d.isVisible() and d.SourceListWidget.count() == len(SOURCES))


def test_log_collapses_repeats():
    print('log flooding')
    d = make()
    got = []
    d.logged.connect(got.append)
    n0 = d.LogListWidget.count()
    for _ in range(3):
        d._log('RuntimeWarning: invalid value encountered in sqrt')
    d._log('a different line')
    items = [d.LogListWidget.item(i).text()
             for i in range(n0, d.LogListWidget.count())]
    check('the same line three times is ONE item with a count',
          len(items) == 2 and items[0].endswith('(x 3)'), str(items))
    check('the main log received it once, not three times',
          got.count('RuntimeWarning: invalid value encountered in sqrt') == 1
          and 'a different line' in got)


def test_multispot_gets_the_trained_bank():
    """Multispot review needs the PSF a training run measured, and Spot
    Check only looks inside the bundle folder for it. The dialog hands
    the run's bank over instead of asking a person to copy a file."""
    print('the hinge between the two reviews')
    import shutil
    import tempfile
    from codelab_pipeline.training import model_store as MS
    d = make()
    root = tempfile.mkdtemp(prefix='runs_')
    try:
        run = os.path.join(root, 'tester_20260910-120000')
        os.makedirs(run)
        open(os.path.join(run, 'psf_bank.h5'), 'wb').close()
        empty = os.path.join(root, 'tester_20260910-130000')
        os.makedirs(empty)                     # a run with no bank

        real_default, real_avail = MS.default_model, MS.available
        MS.default_model = lambda root=None: None
        MS.available = lambda root=None: []
        try:
            d._last_run_dir = None
            check('no run anywhere -> no bank, pass/fail only',
                  d.psf_bank_for_review() is None)
            d._last_run_dir = empty
            check('a just-trained run WITHOUT a bank is skipped',
                  d.psf_bank_for_review() is None)
            d._last_run_dir = run
            check('the just-trained run wins',
                  d.psf_bank_for_review() == os.path.join(run, 'psf_bank.h5'))
            d._last_run_dir = None
            MS.default_model = lambda root=None: run
            check('then the pinned default',
                  d.psf_bank_for_review() == os.path.join(run, 'psf_bank.h5'))
            MS.default_model = lambda root=None: None
            MS.available = lambda root=None: [{'path': empty}, {'path': run}]
            check('then the newest on disk that actually has one',
                  d.psf_bank_for_review() == os.path.join(run, 'psf_bank.h5'))
        finally:
            MS.default_model, MS.available = real_default, real_avail
    finally:
        shutil.rmtree(root, ignore_errors=True)
    import inspect
    src = inspect.getsource(d._open_spotcheck)
    check("Open Spot Check passes --psf-bank when there is one",
          "'--psf-bank'" in src)
    check('and says pass/fail-only when there is not',
          'only the pass/fail review' in src)


def test_bundle_state_and_relaunch():
    """What a relaunched app sees on disk, and what it does about it."""
    print('bundle state across a relaunch')
    import json
    import shutil
    import tempfile
    import time
    d = make()
    root = tempfile.mkdtemp(prefix='state_')
    try:
        d.BundlePathLineEdit.setText(os.path.join(root, 'nope'))
        check("no directory -> 'none'", d.bundle_state() == 'none')
        b = os.path.join(root, 'b')
        os.makedirs(b)
        d.BundlePathLineEdit.setText(b)
        check("a directory without a manifest -> 'none'",
              d.bundle_state() == 'none')
        mp = os.path.join(b, 'bundle_manifest.json')
        json.dump({'channel': 635}, open(mp, 'w'))
        check("a manifest without complete and no log -> 'incomplete'",
              d.bundle_state() == 'incomplete')
        lp = os.path.join(b, 'build.log')
        open(lp, 'w').write('  [  1/ 10] fov001 H 1 crops 1 cand\n')
        check("...and a log written just now -> 'building'",
              d.bundle_state() == 'building')
        old = time.time() - 3600
        os.utime(lp, (old, old))
        check("...and a log an hour quiet -> 'incomplete' (interrupted)",
              d.bundle_state() == 'incomplete')
        d._refresh_review()
        check('the state line says the shards are usable and Build appends',
              'INCOMPLETE' in d.BundleStateLabel.text()
              and 'appends' in d.BundleStateLabel.text())
        json.dump({'channel': 635, 'complete': True}, open(mp, 'w'))
        check("complete=True -> 'complete'", d.bundle_state() == 'complete')

        # A build in progress is FOLLOWED, not re-run.
        json.dump({'channel': 635}, open(mp, 'w'))
        now = time.time()
        os.utime(lp, (now, now))
        d._reattach_if_building()
        check('a running build gets a tail and Build is disabled',
              d._tail is not None
              and not d.BuildBundlePushButton.isEnabled())
        d._tail.stop()
        d._tail.wait(2000)
        d._on_tail_done(0)
        check('when it goes quiet, Build comes back and says why',
              d.BuildBundlePushButton.isEnabled()
              and any('interrupted' in d.LogListWidget.item(i).text()
                      for i in range(d.LogListWidget.count())))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_memory_and_target():
    print('memory across relaunch, and where training lands')
    import json
    import shutil
    import tempfile
    root = tempfile.mkdtemp(prefix='mem_')
    try:
        mem = os.path.join(root, 'build_model_last.json')
        b = os.path.join(root, 'bundle')
        os.makedirs(b)
        d = make()
        d._memory_path = lambda: mem
        d.BundlePathLineEdit.setText(b)
        d._on_path_edited()
        check('the bundle path is remembered', json.load(open(mem))['bundle'] == b)
        d2 = make()
        d2._memory_path = lambda: mem
        d2.BundlePathLineEdit.setText('')
        d2._recall_bundle_path()
        check('and recalled on the next open', d2.BundlePathLineEdit.text() == b)

        # A run to calibrate into, chosen from the verdicts themselves.
        run = os.path.join(root, 'tester_20260910-100000')
        os.makedirs(run)
        open(os.path.join(run, 'psf_bank.h5'), 'wb').close()
        rec = {'kind': 'multispot', 'reviewer': 'tester', 'key': 'k1',
               'page': 0, 'shown': [{'i': 0, 'y': 1, 'x': 1, 'z': 1,
                                     'p': 0.9, 'keep': 1}],
               'bank': 'psf_bank.h5',
               'bank_path': os.path.join(run, 'psf_bank.h5'),
               'bank_sha': 'abc'}
        with open(os.path.join(b, 'multispot_tester__s1.jsonl'), 'w') as f:
            f.write(json.dumps(rec) + '\n')
        check('the verdicts name the run whose bank scored them',
              d2.bank_that_scored_the_verdicts() == run)
        d2.TargetRunComboBox.addItem('tester_20260910-100000', run)
        d2.TargetRunComboBox.setCurrentIndex(d2.TargetRunComboBox.count() - 1)
        check('an existing target -> calibrate into it, no reviewer needed',
              d2.target_run() == run
              and '--calibrate-into' in d2.train_command()
              and '--reviewer' not in d2.train_command())
        check('and the button says what will and will not change',
              'Add multispot calibration' in d2.TrainPushButton.text()
              and 'untouched' in d2.TrainPushButton.text())
        d2.TargetRunComboBox.setCurrentIndex(0)
        check("'new run' -> the full training command",
              d2.target_run() is None
              and 'Train' in d2.TrainPushButton.text())
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_overwrite_is_said_and_cleared():
    """The same name replaces the run -- said in the log, and the run's
    own artefacts cleared first so nothing stale is bound in."""
    print('a named run overwrites')
    import shutil
    import tempfile
    from codelab_pipeline.training import model_store as MS
    import tools.train_spotmodel as T
    root = tempfile.mkdtemp(prefix='named_')
    try:
        real = MS.models_dir
        MS.models_dir = lambda root=None: root_dir
        root_dir = root
        run = os.path.join(root, 'mine')
        os.makedirs(run)
        for n in ('spot_classifier_linear.json', 'spot_classifier_conv.json',
                  'psf_bank.h5', 'psf_multispot.json', 'report.json',
                  MS.MANIFEST):
            open(os.path.join(run, n), 'w').write('{}')
        open(os.path.join(run, 'my_notes.txt'), 'w').write('keep me')
        d = make()
        d.BundlePathLineEdit.setText(REPO)
        d.ReviewerLineEdit.setText('tester')
        d.RunNameLineEdit.setText('mine')
        d.TargetRunComboBox.setCurrentIndex(0)
        check('the button warns of the overwrite',
              'OVERWRITE run mine' in d.TrainPushButton.text(),
              d.TrainPushButton.text())
        cmd = d.train_command()
        check('the command still goes out (no modal stop)',
              cmd is not None and cmd[cmd.index('--out') + 1] == 'mine')
        check('and the log says what will be replaced',
              any('will be OVERWRITTEN' in d.LogListWidget.item(i).text()
                  for i in range(d.LogListWidget.count())))
        gone = T.clear_run(run)
        check("clear_run removes exactly the run's artefacts",
              sorted(gone) == sorted(['spot_classifier_linear.json',
                                      'spot_classifier_conv.json',
                                      'psf_bank.h5', 'psf_multispot.json',
                                      'report.json', MS.MANIFEST]),
              str(gone))
        check("and leaves a person's file alone",
              os.path.exists(os.path.join(run, 'my_notes.txt')))
        import inspect
        src = inspect.getsource(T.main)
        check('main clears an existing run before writing',
              'clear_run(a.out)' in src and 'OVERWRITING' in src)
    finally:
        MS.models_dir = real
        shutil.rmtree(root, ignore_errors=True)


def main():
    test_sources()
    test_fovs()
    test_build_commands()
    test_train_during_build()
    test_review_status()
    test_train_command()
    test_render_report()
    test_close_hides()
    test_log_collapses_repeats()
    test_multispot_gets_the_trained_bank()
    test_bundle_state_and_relaunch()
    test_memory_and_target()
    test_overwrite_is_said_and_cleared()
    print()
    print('%d/%d checks passed' % (CHECKS[1], CHECKS[0]))
    return 0 if CHECKS[1] == CHECKS[0] else 1


if __name__ == '__main__':
    sys.exit(main())
