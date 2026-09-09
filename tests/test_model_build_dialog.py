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
    return ModelBuildDialog(SOURCES, list(fov_pool), STORES.get, REPO)


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

    d.FovListLineEdit.setText('')
    check('no FOVs -> no commands, with a reason', d.build_commands() == [])
    d.FovListLineEdit.setText('7')
    d.BundlePathLineEdit.setText('')
    check('no path -> no commands', d.build_commands() == [])


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
    check('no reviewer -> no command, with the reason',
          d.train_command() is None
          and any('Name the reviewer' in d.LogListWidget.item(i).text()
                  for i in range(d.LogListWidget.count())))
    d.ReviewerLineEdit.setText('tester')
    cmd = d.train_command()
    check('the command names the bundle and the reviewer',
          cmd is not None and cmd[3] == d.bundle_dir()
          and cmd[cmd.index('--reviewer') + 1] == 'tester')
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


def main():
    test_sources()
    test_fovs()
    test_build_commands()
    test_review_status()
    test_train_command()
    test_render_report()
    test_close_hides()
    print()
    print('%d/%d checks passed' % (CHECKS[1], CHECKS[0]))
    return 0 if CHECKS[1] == CHECKS[0] else 1


if __name__ == '__main__':
    sys.exit(main())
