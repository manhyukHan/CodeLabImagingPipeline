"""
Build a model: bundle, review, train, look at the result -- one window.

WHY ONE WINDOW. 'Make new model...' was a chain of three QInputDialogs
that ended at step 2 of 3: it built a bundle (blocking the app while it
did), offered to open Spot Check, and then printed the training command
to the log for a person to type. Training never ran from the app at
all. And a bundle was ONE channel wide, so an experiment with two
readout channels meant the whole chain twice into two folders.

This dialog owns the whole sequence, and every long step runs in the
background: the app stays live, the dialog is modeless, and closing it
does not stop a build -- the worker is the dialog's and the dialog is
cached on the main window, so it simply keeps receiving progress.

A BUNDLE IS ONE STORE. Sources from two modalities are two stores, and
build_bundle takes one -- so a mixed selection is refused here with
the reason, rather than silently building the first modality's half.
Within one store, every checked (hybe, channel) goes into ONE bundle
directory: build_bundle runs once per channel with that channel's
hybes and the SAME explicit FOV list, and the shards are named by
channel (training/extract.py) so the runs coexist. Before that naming
fix the second channel's build overwrote the first's shards through
os.replace and the bundle still looked complete.
"""
import json
import os
import random
import sys
import time

from PyQt5 import QtCore, QtGui, QtWidgets

from ui.proc_stream import LogTailWorker, StreamingProcWorker

# A build's own log, inside its bundle: what a relaunched app reads.
BUILD_LOG = 'build.log'
# How long a build log may go quiet before it is called stalled.
STALE_SECONDS = 180


class MaskScanWorker(QtCore.QThread):
    """Which of a store's FOVs carry a segmentation -- off the GUI thread.

    A random FOV draw has to draw from FOVs that HAVE cells (a FOV with
    no masks is one nobody segmented yet, and drawing it would silently
    shrink the sample), and knowing that means opening masks on the
    NAS: one listdir-and-open per FOV. That is exactly the work that
    used to freeze the app for the length of a build.
    """
    done = QtCore.pyqtSignal(list)           # [(fov, n_cells), ...]
    failed = QtCore.pyqtSignal(str)

    def __init__(self, storage_path, pool):
        super().__init__()
        self.storage_path = str(storage_path)
        self.pool = [int(f) for f in pool]

    def run(self):
        try:
            from codelab_pipeline.training import extract as X
            out = []
            for fov in self.pool:
                try:
                    ids = [c[0] for c in X.cell_masks(self.storage_path, fov)]
                except Exception:                           # noqa: BLE001
                    continue
                if ids:
                    out.append((int(fov), len(ids)))
            self.done.emit(out)
        except Exception as exc:                            # noqa: BLE001
            self.failed.emit(f'{type(exc).__name__}: {exc}')


def _parse_fovs(text):
    """'1-10,15 20-25' -> [1..10, 15, 20..25]; order kept, dupes dropped."""
    import re
    out, seen = [], set()
    for chunk in re.split(r'[,\s]+', str(text).strip()):
        if not chunk:
            continue
        if '-' in chunk:
            lo, hi = chunk.split('-', 1)
            rng = range(int(lo), int(hi) + 1)
        else:
            rng = [int(chunk)]
        for f in rng:
            if f not in seen:
                seen.add(f)
                out.append(f)
    return out


class ModelBuildDialog(QtWidgets.QDialog):
    """See the module docstring.

    sources      [(modality, folder, readout_name, channel, is_fiducial)]
    fov_pool     the experiment's declared FOVs (the Ingestion tab's list)
    storage_for  modality -> storage path ('' if none)
    repo_root    where tools/ lives; the child interpreter is this one
    """
    logged = QtCore.pyqtSignal(str)          # every line, for the main log
    model_trained = QtCore.pyqtSignal(str)   # a new run directory exists

    def __init__(self, sources, fov_pool, storage_for, repo_root,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle('Build model')
        self.setWindowFlags(self.windowFlags()
                            & ~QtCore.Qt.WindowContextHelpButtonHint)
        self.resize(760, 900)
        self._sources = list(sources)
        self._fov_pool = [int(f) for f in fov_pool]
        self._storage_for = storage_for
        self._repo = str(repo_root)
        self._worker = None          # the running build/train child
        self._scan = None            # the running mask scan
        self._queue = []             # commands still to run, this build
        self._have = {}              # storage_path -> [(fov, n_cells)]
        self._last_run_dir = None
        self._tail = None            # following a build we did not start
        self._build()
        self._populate_sources()
        self._recall_bundle_path()
        self._refresh_runs()
        self._refresh_review()

    # -- layout --------------------------------------------------------

    def _build(self):
        lay = QtWidgets.QVBoxLayout(self)

        # 1  sources ----------------------------------------------------
        g = QtWidgets.QGroupBox('1   Sources  (check to include)')
        v = QtWidgets.QVBoxLayout(g)
        self.SourceListWidget = QtWidgets.QListWidget()
        self.SourceListWidget.setMinimumHeight(150)
        self.SourceListWidget.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)
        v.addWidget(self.SourceListWidget)
        row = QtWidgets.QHBoxLayout()
        self.CheckSelectedPushButton = QtWidgets.QPushButton('Check Selected')
        self.UncheckSelectedPushButton = QtWidgets.QPushButton(
            'Uncheck Selected')
        self.BulkModalityComboBox = QtWidgets.QComboBox()
        self.BulkChannelComboBox = QtWidgets.QComboBox()
        self.CheckModalityChannelPushButton = QtWidgets.QPushButton(
            'Check Modality+Channel')
        self.CheckModalityChannelPushButton.setToolTip(
            'Check every hybe of the picked modality at the picked '
            'channel -- fiducial-channel entries included: the fiducial '
            'role is a per-hybe fact, not a property of the channel.')
        for w in (self.CheckSelectedPushButton, self.UncheckSelectedPushButton,
                  self.BulkModalityComboBox, self.BulkChannelComboBox,
                  self.CheckModalityChannelPushButton):
            row.addWidget(w)
        row.addStretch(1)
        v.addLayout(row)
        self.SourceNote = QtWidgets.QLabel(
            'One bundle is cut from ONE store, so every checked source '
            'must share a modality. Several channels are fine: each '
            'gets its own shards in the same bundle.')
        self.SourceNote.setWordWrap(True)
        self.SourceNote.setStyleSheet('color:#555;')
        v.addWidget(self.SourceNote)
        lay.addWidget(g)

        # 2  FOVs -------------------------------------------------------
        g = QtWidgets.QGroupBox('2   FOVs')
        f = QtWidgets.QGridLayout(g)
        f.addWidget(QtWidgets.QLabel('draw at random:'), 0, 0)
        self.RandomCountLineEdit = QtWidgets.QLineEdit()
        self.RandomCountLineEdit.setPlaceholderText('how many, then Enter')
        self.RandomCountLineEdit.setValidator(
            QtGui.QIntValidator(1, 100000, self))
        self.RandomCountLineEdit.setMaximumWidth(160)
        f.addWidget(self.RandomCountLineEdit, 0, 1)
        f.addWidget(QtWidgets.QLabel('FOVs to use:'), 0, 2)
        self.FovListLineEdit = QtWidgets.QLineEdit()
        self.FovListLineEdit.setPlaceholderText(
            'e.g. 7,12,19  or  1-4,9   -- edit by hand, or let the left '
            'box fill it')
        f.addWidget(self.FovListLineEdit, 0, 3)
        f.setColumnStretch(3, 1)
        self.FovNote = QtWidgets.QLabel(
            'The draw is from the declared FOVs that carry cell masks '
            '(scanned in the background on Enter). Left fills right; '
            'right never changes left.')
        self.FovNote.setWordWrap(True)
        self.FovNote.setStyleSheet('color:#555;')
        f.addWidget(self.FovNote, 1, 0, 1, 4)
        lay.addWidget(g)

        # 3  bundle -----------------------------------------------------
        g = QtWidgets.QGroupBox('3   Bundle')
        f = QtWidgets.QGridLayout(g)
        f.addWidget(QtWidgets.QLabel('path:'), 0, 0)
        self.BundlePathLineEdit = QtWidgets.QLineEdit()
        f.addWidget(self.BundlePathLineEdit, 0, 1)
        self.BrowsePushButton = QtWidgets.QPushButton('Browse...')
        f.addWidget(self.BrowsePushButton, 0, 2)
        self.BuildBundlePushButton = QtWidgets.QPushButton('Build bundle')
        self.BuildBundlePushButton.setToolTip(
            'Runs tools/build_bundle.py once per checked channel, in the '
            'background, into this one directory. The app stays usable.')
        f.addWidget(self.BuildBundlePushButton, 1, 0, 1, 3)
        f.setColumnStretch(1, 1)
        lay.addWidget(g)

        # 4  review -----------------------------------------------------
        g = QtWidgets.QGroupBox('4   Review')
        f = QtWidgets.QGridLayout(g)
        self.OpenSpotCheckPushButton = QtWidgets.QPushButton(
            'Open Spot Check on this bundle')
        self.OpenSpotCheckPushButton.setToolTip(
            'Opens spotcheck as its own program. You type your OWN name '
            'there -- it goes on your verdict file, so several people '
            'can share one bundle.')
        f.addWidget(self.OpenSpotCheckPushButton, 0, 0)
        self.RefreshReviewPushButton = QtWidgets.QPushButton(
            'Refresh review status')
        f.addWidget(self.RefreshReviewPushButton, 0, 1)
        self.BundleStateLabel = QtWidgets.QLabel('')
        self.BundleStateLabel.setWordWrap(True)
        self.BundleStateLabel.setStyleSheet('color:#7a5200;')
        f.addWidget(self.BundleStateLabel, 2, 0, 1, 2)
        self.ReviewStatusLabel = QtWidgets.QLabel('')
        self.ReviewStatusLabel.setWordWrap(True)
        self.ReviewStatusLabel.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse)
        f.addWidget(self.ReviewStatusLabel, 1, 0, 1, 2)
        lay.addWidget(g)

        # 5  train ------------------------------------------------------
        g = QtWidgets.QGroupBox('5   Train')
        f = QtWidgets.QGridLayout(g)
        f.addWidget(QtWidgets.QLabel('reviewer:'), 0, 0)
        self.ReviewerLineEdit = QtWidgets.QLineEdit()
        self.ReviewerLineEdit.setPlaceholderText(
            'who labelled it -- recorded in the manifest')
        f.addWidget(self.ReviewerLineEdit, 0, 1)
        self.SetDefaultCheckBox = QtWidgets.QCheckBox('pin as default model')
        f.addWidget(self.SetDefaultCheckBox, 0, 2)
        f.addWidget(QtWidgets.QLabel('run name:'), 1, 0)
        self.RunNameLineEdit = QtWidgets.QLineEdit()
        self.RunNameLineEdit.setPlaceholderText(
            'folder under models/ -- defaults to the reviewer; an existing '
            'name is OVERWRITTEN')
        self.RunNameLineEdit.setToolTip(
            'THE NAME IS THE FOLDER. No timestamp: train again under the '
            'same name and the run is replaced -- its classifier, bank, '
            'calibration, report and manifest are removed first, so '
            'nothing of the old run is bound into the new one. Anything '
            'else in that folder is left alone. A different name keeps '
            'the old run.')
        f.addWidget(self.RunNameLineEdit, 1, 1, 1, 2)
        f.addWidget(QtWidgets.QLabel('into:'), 2, 0)
        self.TargetRunComboBox = QtWidgets.QComboBox()
        self.TargetRunComboBox.setToolTip(
            'NEW RUN trains everything from the pass/fail verdicts: the '
            'classifier (M1), the PSF bank, and -- if multispot verdicts '
            'exist -- the calibration.\n\n'
            'AN EXISTING RUN adds ONLY the multispot calibration to it and '
            'touches nothing else. Pick the run whose PSF the multispot '
            'review matched with: its verdicts carry the p that bank '
            'scored, so the calibration belongs to that bank, and a fresh '
            'retrain would also re-fit the M1 you may want kept.')
        f.addWidget(self.TargetRunComboBox, 2, 1, 1, 2)
        self.TrainPushButton = QtWidgets.QPushButton('Train')
        self.TrainPushButton.setToolTip(
            'Runs tools/train_spotmodel.py in the background. The result '
            'lands below and in the model list.')
        f.addWidget(self.TrainPushButton, 3, 0, 1, 3)
        self.ResultTextEdit = QtWidgets.QPlainTextEdit()
        self.ResultTextEdit.setReadOnly(True)
        self.ResultTextEdit.setMinimumHeight(170)
        self.ResultTextEdit.setFont(QtGui.QFont('Consolas', 9))
        self.ResultTextEdit.setPlaceholderText(
            'Training result appears here: labels, each head\'s '
            'validation numbers, the multispot calibration and the PSF.')
        f.addWidget(self.ResultTextEdit, 4, 0, 1, 3)
        f.setColumnStretch(1, 1)
        lay.addWidget(g)

        # progress + log --------------------------------------------------
        self.ProgressBar = QtWidgets.QProgressBar()
        self.ProgressBar.setRange(0, 1)
        self.ProgressBar.setValue(0)
        lay.addWidget(self.ProgressBar)
        self.LogListWidget = QtWidgets.QListWidget()
        self.LogListWidget.setMinimumHeight(110)
        lay.addWidget(self.LogListWidget, 1)
        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        self.ClosePushButton = QtWidgets.QPushButton('Close')
        btns.addWidget(self.ClosePushButton)
        lay.addLayout(btns)

        # wiring ---------------------------------------------------------
        self.CheckSelectedPushButton.clicked.connect(
            lambda: self._set_selected(True))
        self.UncheckSelectedPushButton.clicked.connect(
            lambda: self._set_selected(False))
        self.CheckModalityChannelPushButton.clicked.connect(
            self._check_modality_channel)
        self.SourceListWidget.itemChanged.connect(
            lambda _i: self._suggest_bundle_path())
        self.RandomCountLineEdit.returnPressed.connect(self._draw_fovs)
        self.BrowsePushButton.clicked.connect(self._browse)
        self.BuildBundlePushButton.clicked.connect(self._build_bundle)
        self.OpenSpotCheckPushButton.clicked.connect(self._open_spotcheck)
        self.RefreshReviewPushButton.clicked.connect(self._refresh_review)
        self.BundlePathLineEdit.editingFinished.connect(self._on_path_edited)
        self.TargetRunComboBox.currentIndexChanged.connect(
            lambda _i: self._explain_target())
        self.RunNameLineEdit.textChanged.connect(
            lambda _t: self._explain_target())
        self.ReviewerLineEdit.textChanged.connect(
            lambda _t: self._explain_target())
        self.TrainPushButton.clicked.connect(self._train)
        self.ClosePushButton.clicked.connect(self.close)

    # -- log ----------------------------------------------------------

    def _log(self, message):
        text = str(message)
        lw = self.LogListWidget
        # A LINE THAT REPEATS BECOMES A COUNT. A child that prints one
        # warning per fit hands this list hundreds of thousands of
        # identical items, and a QListWidget that size is what makes
        # the dialog itself stop responding. The same line again is
        # shown once with 'x N'; the main log gets it once.
        last = lw.item(lw.count() - 1) if lw.count() else None
        base = last.data(QtCore.Qt.UserRole) if last is not None else None
        if last is not None and base == text:
            n = int(last.data(QtCore.Qt.UserRole + 1) or 1) + 1
            last.setData(QtCore.Qt.UserRole + 1, n)
            last.setText(f'{text}   (x {n})')
            return
        item = QtWidgets.QListWidgetItem(text)
        item.setData(QtCore.Qt.UserRole, text)
        item.setData(QtCore.Qt.UserRole + 1, 1)
        lw.addItem(item)
        lw.scrollToBottom()
        self.logged.emit(text)

    # -- 1  sources -----------------------------------------------------

    def _populate_sources(self):
        lw = self.SourceListWidget
        lw.blockSignals(True)
        lw.clear()
        self.BulkModalityComboBox.clear()
        self.BulkChannelComboBox.clear()
        mods, chans = [], set()
        for modality, folder, name, ch, fid in self._sources:
            if modality not in mods:
                mods.append(modality)
            chans.add(int(ch))
            shown = f'{folder} ({name})' if name else folder
            label = f'{modality} | {shown} | ch{ch}' + (
                '  (fiducial)' if fid else '')
            item = QtWidgets.QListWidgetItem(label)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            item.setData(QtCore.Qt.UserRole, [modality, folder, int(ch)])
            lw.addItem(item)
        for m in mods:
            self.BulkModalityComboBox.addItem(m)
        for ch in sorted(chans):
            self.BulkChannelComboBox.addItem(str(ch))
        lw.blockSignals(False)

    def _set_selected(self, checked):
        for item in self.SourceListWidget.selectedItems():
            item.setCheckState(QtCore.Qt.Checked if checked
                               else QtCore.Qt.Unchecked)

    def _check_modality_channel(self):
        modality = self.BulkModalityComboBox.currentText()
        ch_text = self.BulkChannelComboBox.currentText()
        if not modality or not ch_text:
            return
        channel = int(ch_text)
        lw = self.SourceListWidget
        n = 0
        for i in range(lw.count()):
            item = lw.item(i)
            m, _h, ch = item.data(QtCore.Qt.UserRole)
            if m == modality and int(ch) == channel:
                item.setCheckState(QtCore.Qt.Checked)
                n += 1
        self._log(f'checked {n} source(s): {modality} ch{channel}')

    def checked_sources(self):
        """[(modality, folder, channel)] in list order."""
        out = []
        lw = self.SourceListWidget
        for i in range(lw.count()):
            item = lw.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                m, h, ch = item.data(QtCore.Qt.UserRole)
                out.append((str(m), str(h), int(ch)))
        return out

    def _one_modality(self, quiet=False):
        """The single modality the checked sources share, or None."""
        mods = []
        for m, _h, _c in self.checked_sources():
            if m not in mods:
                mods.append(m)
        if len(mods) == 1:
            return mods[0]
        if not quiet:
            if not mods:
                self._log('Check at least one source first.')
            else:
                self._log('Checked sources span ' + ', '.join(mods)
                          + ' -- a bundle is cut from ONE store, so pick '
                            'one modality.')
        return None

    def _store(self, quiet=False):
        m = self._one_modality(quiet=quiet)
        if m is None:
            return None
        sp = self._storage_for(m)
        if not sp:
            if not quiet:
                self._log(f'{m} has no storage path configured.')
            return None
        return sp

    def _suggest_bundle_path(self):
        """A default next to the store, once the store is known."""
        if self.BundlePathLineEdit.text().strip():
            return
        sp = self._store(quiet=True)
        if not sp:
            return
        name = os.path.basename(os.path.normpath(str(sp))) or 'store'
        chans = sorted({c for _m, _h, c in self.checked_sources()})
        tag = '_'.join(f'ch{c}' for c in chans) if chans else 'bundle'
        self.BundlePathLineEdit.setText(os.path.join(
            os.path.dirname(os.path.abspath(str(sp))), 'review_bundles',
            f'{name}_{tag}'))

    # -- 2  FOVs --------------------------------------------------------

    def _draw_fovs(self):
        text = self.RandomCountLineEdit.text().strip()
        if not text:
            return
        n = int(text)
        sp = self._store()
        if not sp:
            return
        if not self._fov_pool:
            self._log('No FOVs declared on the Ingestion tab -- nothing '
                      'to draw from.')
            return
        if sp in self._have:
            self._fill_fovs(n, self._have[sp])
            return
        if self._scan is not None and self._scan.isRunning():
            self._log('still scanning for cell masks -- one moment')
            return
        self._log(f'scanning {len(self._fov_pool)} declared FOV(s) for cell '
                  f'masks in the background...')
        self.ProgressBar.setRange(0, 0)
        self._scan = MaskScanWorker(sp, self._fov_pool)
        self._scan.done.connect(lambda have: self._on_scanned(sp, n, have))
        self._scan.failed.connect(self._on_scan_failed)
        self._scan.start()

    def _on_scanned(self, sp, n, have):
        self.ProgressBar.setRange(0, 1)
        self.ProgressBar.setValue(0)
        self._have[sp] = list(have)
        self._log(f'{len(have)} of {len(self._fov_pool)} declared FOV(s) '
                  f'carry cell masks')
        self._fill_fovs(n, have)

    def _on_scan_failed(self, why):
        self.ProgressBar.setRange(0, 1)
        self.ProgressBar.setValue(0)
        self._log('mask scan failed: ' + why)

    def _fill_fovs(self, n, have):
        pool = sorted(f for f, _n in have)
        if not pool:
            self._log('no FOV carries a segmentation -- segment first')
            return
        seed = int(time.time())
        # SORTED POOL AND A LOGGED SEED, so the same seed gives the same
        # draw on any machine -- build_bundle's own rule.
        picked = sorted(random.Random(seed).sample(pool, min(n, len(pool))))
        self.FovListLineEdit.setText(','.join(str(f) for f in picked))
        self._log(f'drew {len(picked)} FOV(s) at random (seed {seed}): '
                  + ','.join(str(f) for f in picked)
                  + (f'  -- only {len(pool)} had masks' if n > len(pool)
                     else ''))

    def fovs(self):
        try:
            return _parse_fovs(self.FovListLineEdit.text())
        except ValueError:
            return []

    # -- 3  bundle ------------------------------------------------------

    def _browse(self):
        start = self.BundlePathLineEdit.text().strip() or os.path.expanduser('~')
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Bundle directory', start)
        if d:
            self.BundlePathLineEdit.setText(d)
            self._on_path_edited()

    def _on_path_edited(self):
        self._remember_bundle_path()
        self._refresh_review()
        self._reattach_if_building()

    # -- memory across relaunch ----------------------------------------

    def _memory_path(self):
        # Beside Spot Check's own last_session.json, same idea, same
        # lifetime: a small file in the repo that says where we were.
        return os.path.join(self._repo, 'spotcheck', 'build_model_last.json')

    def _remember_bundle_path(self):
        b = self.bundle_dir()
        if not b:
            return
        try:
            with open(self._memory_path(), 'w', encoding='utf-8') as f:
                json.dump({'bundle': b}, f)
        except OSError:
            pass

    def _recall_bundle_path(self):
        """Prefill the bundle path from the last session -- ours, else
        Spot Check's. A relaunch mid-workflow lands on the same bundle
        instead of asking a person to find it again."""
        if self.BundlePathLineEdit.text().strip():
            return
        for path in (self._memory_path(),
                     os.path.join(self._repo, 'spotcheck', 'last_session.json')):
            try:
                with open(path, encoding='utf-8') as f:
                    b = (json.load(f) or {}).get('bundle')
            except Exception:                               # noqa: BLE001
                continue
            if b and os.path.isdir(str(b)):
                self.BundlePathLineEdit.setText(str(b))
                self._log('bundle path recalled from the last session: ' + str(b))
                return

    # -- what state a bundle directory is in ------------------------------

    def bundle_state(self):
        """'none' | 'building' | 'incomplete' | 'complete', from disk.

        The manifest is written before a build and stamped complete=True
        after it, and the build appends to build.log as it goes. So: no
        manifest -> none; complete -> complete; otherwise the log's
        mtime says whether something is still writing (recent) or the
        build was interrupted (quiet longer than STALE_SECONDS)."""
        b = self.bundle_dir()
        if not b or not os.path.isdir(b):
            return 'none'
        mp = os.path.join(b, 'bundle_manifest.json')
        if not os.path.exists(mp):
            return 'none'
        try:
            with open(mp, encoding='utf-8') as f:
                if (json.load(f) or {}).get('complete'):
                    return 'complete'
        except Exception:                                   # noqa: BLE001
            return 'incomplete'
        lp = os.path.join(b, BUILD_LOG)
        try:
            if time.time() - os.path.getmtime(lp) < STALE_SECONDS:
                return 'building'
        except OSError:
            pass
        return 'incomplete'

    def _manifest_complete(self):
        return self.bundle_state() == 'complete'

    def _reattach_if_building(self):
        """A build this app did not start is followed, not re-run."""
        if self._worker is not None or self._tail is not None:
            return
        state = self.bundle_state()
        if state != 'building':
            return
        self._log('a build is writing to this bundle right now (its log '
                  'changed within the last %d s) -- following it. Build '
                  'is disabled until it finishes.' % STALE_SECONDS)
        self.BuildBundlePushButton.setEnabled(False)
        self.ProgressBar.setRange(0, 0)
        t = LogTailWorker(os.path.join(self.bundle_dir(), BUILD_LOG),
                          self._manifest_complete, stale_seconds=STALE_SECONDS)
        self._tail = t
        t.line.connect(lambda x: self._log('  ' + x))
        t.progress.connect(self._on_progress)
        t.finished_ok.connect(self._on_tail_done)
        t.start()

    def _on_tail_done(self, done):
        t = self._tail
        self._tail = None
        if t is not None:
            t.deleteLater()
        self.BuildBundlePushButton.setEnabled(True)
        self.ProgressBar.setRange(0, 1)
        self.ProgressBar.setValue(0)
        if done:
            self._log('the build we were following finished.')
        else:
            self._log('the build we were following went quiet for %d s -- '
                      'treating it as interrupted. Its shards are all '
                      'usable; Build appends the missing ones.'
                      % STALE_SECONDS)
        self._refresh_review()

    def bundle_dir(self):
        return self.BundlePathLineEdit.text().strip()

    def build_commands(self):
        """The build_bundle command per checked channel, or [] with a
        logged reason. Pure: reads widgets, runs nothing."""
        sp = self._store()
        if not sp:
            return []
        out = self.bundle_dir()
        if not out:
            self._log('Give the bundle a path (3).')
            return []
        fovs = self.fovs()
        if not fovs:
            self._log('List the FOVs to use (2), or draw some.')
            return []
        by_ch = {}
        for _m, folder, ch in self.checked_sources():
            by_ch.setdefault(int(ch), [])
            if folder not in by_ch[ch]:
                by_ch[ch].append(folder)
        cmds = []
        for ch in sorted(by_ch):
            cmds.append([sys.executable, '-u',
                         os.path.join(self._repo, 'tools', 'build_bundle.py'),
                         str(sp), '--out', out, '--channel', str(ch),
                         '--hybes', ','.join(by_ch[ch]),
                         '--fovs', ','.join(str(f) for f in fovs),
                         # THE REAL POOL, not build_bundle's '1-41'
                         # default, which silently capped every draw
                         # the old chain made at FOV 41.
                         '--fov-pool', ','.join(str(f) for f in
                                                (self._fov_pool or fovs))])
        return cmds

    def _build_bundle(self):
        if self._busy():
            return
        cmds = self.build_commands()
        if not cmds:
            return
        if self._tail is not None:
            self._log('a build is already writing to this bundle -- wait '
                      'for it, or pick another path.')
            return
        state = self.bundle_state()
        if state == 'incomplete':
            self._log('this bundle is incomplete -- the build APPENDS: '
                      'shards already on disk are kept, only the missing '
                      'ones are built.')
        elif state == 'complete':
            self._log('this bundle is complete -- a build here adds '
                      'channels or FOVs not yet in it and skips the rest.')
        self._queue = list(cmds)
        self._log(f'building {len(cmds)} channel(s) into {self.bundle_dir()} '
                  f'-- in the background, one channel after another. '
                  f'Closing the app does not stop it.')
        self._set_busy(True)
        # Show the status line NOW, not only when the build ends: a
        # suggested path is set programmatically, which fires no
        # editingFinished, so until this the line sat empty for the
        # whole build and read as if the button did nothing.
        self._refresh_review()
        self._next_build()

    def _next_build(self):
        if not self._queue:
            self._set_busy(False)
            self._log('bundle ready: ' + self.bundle_dir())
            self._refresh_review()
            return
        cmd = self._queue.pop(0)
        ch = cmd[cmd.index('--channel') + 1]
        self._log(f'--- channel {ch} ---   ' + ' '.join(cmd[2:]))
        self.ProgressBar.setRange(0, 0)
        # FILE-BACKED AND DETACHED, so the build outlives this app. Its
        # output goes to <bundle>/build.log, which a relaunched app
        # follows (see _reattach_if_building). MEASURED: a child on a
        # pipe died with its parent; one on a file finished.
        w = StreamingProcWorker(
            cmd, cwd=self._repo,
            log_path=os.path.join(self.bundle_dir(), BUILD_LOG))
        self._worker = w
        w.line.connect(lambda t: self._log('  ' + t))
        w.progress.connect(self._on_progress)
        w.finished_ok.connect(self._on_build_step_done)
        w.failed.connect(lambda why: self._on_build_step_done(-1, why))
        w.start()

    def _on_build_step_done(self, code, why=''):
        self._release_worker()
        if why:
            self._log('  could not start: ' + why)
        if int(code) != 0 or why:
            self._queue = []
            self._set_busy(False)
            self._log(f'bundle build stopped (exit {code}); the lines above '
                      f'are the build\'s own output.')
            self._refresh_review()
            return
        self._next_build()

    # -- 4  review ------------------------------------------------------

    def psf_bank_for_review(self):
        """The PSF bank multispot review should match with, or None.

        THE TWO REVIEWS ARE SEQUENTIAL, and this is the hinge between
        them. Multispot review judges the matches a PSF finds in the
        pillar around each spot pass/fail already confirmed -- so it
        needs a PSF, and the PSF comes from training on the pass/fail
        verdicts. Spot Check looks for psf_bank.h5 INSIDE the bundle
        folder and nothing ever put it there: after training M1 a person
        opened multispot mode and was told to copy a file by hand. The
        dialog knows which run was just trained, so it hands that run's
        bank over. Newest run first, then the pinned default, then the
        newest on disk; None means only pass/fail can run yet.
        """
        from codelab_pipeline.training import model_store as MS
        cands = [self._last_run_dir, MS.default_model()]
        try:
            cands += [r['path'] for r in MS.available()]
        except Exception:                                   # noqa: BLE001
            pass
        for run in cands:
            if not run:
                continue
            bank = os.path.join(str(run), 'psf_bank.h5')
            if os.path.exists(bank):
                return bank
        return None

    def _open_spotcheck(self):
        import subprocess
        b = self.bundle_dir()
        if not b or not os.path.isdir(b):
            self._log('No bundle directory to review yet.')
            return
        cmd = [sys.executable, '-m', 'spotcheck.app', b]
        bank = self.psf_bank_for_review()
        if bank:
            cmd += ['--psf-bank', bank]
            self._log('multispot review will match with the PSF from '
                      + os.path.basename(os.path.dirname(bank)))
        else:
            self._log('no trained run yet, so only the pass/fail review '
                      'can run: multispot needs the PSF a training run '
                      'measures from confirmed spots. Train (5) first, '
                      'then open Spot Check again for multispot.')
        try:
            subprocess.Popen(cmd, cwd=self._repo)
        except Exception as exc:                            # noqa: BLE001
            self._log(f'could not launch Spot Check: '
                      f'{type(exc).__name__}: {exc}')
            return
        self._log('Spot Check opened as its own program. Verdicts land in '
                  'the bundle folder; press "Refresh review status" here '
                  'to see them.')

    def review_status(self):
        """Counts from the bundle on disk, or None. Cheap: verdict JSONL
        plus one index header per shard, no pixels."""
        b = self.bundle_dir()
        if not b or not os.path.isdir(b):
            return None
        from codelab_pipeline.training import bundle as B
        from codelab_pipeline.training import verdicts as V
        st = {'crops': 0, 'channels': set(), 'hybes': set(), 'fovs': set()}
        for p in B.shard_paths(b):
            try:
                idx = B.read_index(p)
            except Exception:                               # noqa: BLE001
                continue
            st['crops'] += len(idx)
            for row in idx:
                st['channels'].add(int(row['channel']))
                st['hybes'].add(str(row['hybe']))
                st['fovs'].add(int(row['fov']))
        for kind, key in ((V.DEFAULT_KIND, 'passfail'),
                          (V.MULTISPOT_KIND, 'multispot')):
            try:
                recs, _agree = V.merge(b, kind=kind)
            except Exception:                               # noqa: BLE001
                recs = []
            st[key] = {
                'records': len(recs),
                'crops': len({r['key'] for r in recs}),
                'pages': len({(r['key'], int(r.get('page', 0)))
                              for r in recs}),
                'reviewers': sorted({str(r.get('reviewer') or '')
                                     for r in recs} - {''}),
            }
        return st

    def _refresh_review(self):
        state = self.bundle_state()
        self.BundleStateLabel.setText({
            'none': '',
            'building': 'a build is writing to this bundle now',
            'incomplete': 'INCOMPLETE build -- every shard on disk is '
                          'complete and usable; Build appends the missing '
                          'ones. Review and training work on what is there.',
            'complete': '',
        }[state])
        st = self.review_status()
        if st is None:
            self.ReviewStatusLabel.setText(
                'no bundle at that path yet -- Build bundle (3) writes one'
                if self.bundle_dir() else
                'give the bundle a path (3) to see its review status here')
            return
        pf, ms = st['passfail'], st['multispot']
        chans = ', '.join(f'ch{c}' for c in sorted(st['channels'])) or '--'
        self.ReviewStatusLabel.setText(
            f"bundle   {st['crops']:,} crops   {chans}   "
            f"{len(st['hybes'])} hybe(s)   FOVs "
            f"{','.join(str(f) for f in sorted(st['fovs'])) or '--'}\n"
            f"pass/fail   {pf['crops']:,} of {st['crops']:,} crops judged "
            f"({pf['pages']:,} pages) by "
            f"{', '.join(pf['reviewers']) or 'nobody yet'}\n"
            f"multispot   {ms['crops']:,} crops ({ms['pages']:,} pillar pages)"
            f" by {', '.join(ms['reviewers']) or 'nobody yet'}")
        if not self.ReviewerLineEdit.text().strip():
            names = pf['reviewers'] or ms['reviewers']
            if len(names) == 1:
                self.ReviewerLineEdit.setText(names[0])
        if ms['records'] and self.target_run() is None:
            self.suggest_target()

    # -- 5  train -------------------------------------------------------

    def _refresh_runs(self, select=None):
        """Fill the target combo: new run, then every run on disk."""
        from codelab_pipeline.training import model_store as MS
        cb = self.TargetRunComboBox
        want = select or cb.currentData()
        cb.blockSignals(True)
        cb.clear()
        cb.addItem('new run  (<reviewer>_<timestamp>)', None)
        try:
            runs = MS.available()
        except Exception:                                   # noqa: BLE001
            runs = []
        for r in runs:
            tag = ('  [default]' if r.get('is_default') else '') + (
                '  +multispot' if r.get('has_multispot') else '')
            cb.addItem(f"{r['name']}{tag}", r['path'])
        idx = cb.findData(want) if want else -1
        cb.setCurrentIndex(idx if idx >= 0 else 0)
        cb.blockSignals(False)
        self._explain_target()

    def target_run(self):
        """The existing run to calibrate into, or None for a new run."""
        return self.TargetRunComboBox.currentData()

    def run_name(self):
        """The run folder name: the run-name field, else the reviewer."""
        name = self.RunNameLineEdit.text().strip()
        return name or self.ReviewerLineEdit.text().strip()

    def run_dir_for_name(self):
        from codelab_pipeline.training import model_store as MS
        name = self.run_name()
        return os.path.join(MS.models_dir(), name) if name else None

    def _explain_target(self):
        run = self.target_run()
        if run:
            self.TrainPushButton.setText(
                f'Add multispot calibration to {os.path.basename(run)}  '
                f'(M1 and PSF untouched)')
            return
        name = self.run_name()
        target = self.run_dir_for_name()
        if not name:
            self.TrainPushButton.setText('Train  (name the run first)')
        elif target and os.path.isdir(target) and os.listdir(target):
            self.TrainPushButton.setText(
                f'Train -> OVERWRITE run {name}  (M1 + PSF + calibration '
                f'if reviewed; the old run is replaced)')
        else:
            self.TrainPushButton.setText(
                f'Train -> new run {name}  (M1 + PSF + calibration if '
                f'reviewed)')

    def bank_that_scored_the_verdicts(self):
        """The run whose PSF the multispot verdicts were matched with,
        read from the verdicts themselves (records carry `bank_path`),
        or None for verdicts older than that field."""
        b = self.bundle_dir()
        if not b or not os.path.isdir(b):
            return None
        from codelab_pipeline.training import verdicts as V
        try:
            recs, _a = V.merge(b, kind=V.MULTISPOT_KIND)
        except Exception:                                   # noqa: BLE001
            return None
        banks = {str(r.get('bank_path')) for r in recs if r.get('bank_path')}
        if len(banks) == 1:
            run = os.path.dirname(next(iter(banks)))
            return run if os.path.isdir(run) else None
        return None

    def suggest_target(self):
        """Preselect the run the multispot verdicts belong to."""
        run = self.bank_that_scored_the_verdicts()
        if run is None:
            st = self.review_status() or {}
            if (st.get('multispot') or {}).get('records'):
                bank = self.psf_bank_for_review()
                run = os.path.dirname(bank) if bank else None
        if run:
            self._refresh_runs(select=run)

    def train_command(self):
        """The train_spotmodel command, or None with a logged reason."""
        b = self.bundle_dir()
        if not b or not os.path.isdir(b):
            self._log('No bundle to train from.')
            return None
        run = self.target_run()
        if run:
            # CALIBRATE INTO: the classifier and the bank in that run
            # stay; only psf_multispot.json is added and the manifest
            # rebound. No reviewer needed -- the run already has one.
            cmd = [sys.executable, '-u',
                   os.path.join(self._repo, 'tools', 'train_spotmodel.py'),
                   b, '--calibrate-into', str(run)]
            if self.SetDefaultCheckBox.isChecked():
                cmd.append('--set-default')
            return cmd
        who = self.ReviewerLineEdit.text().strip()
        if not who:
            self._log('Name the reviewer (5): a model is calibrated to '
                      'whoever labelled it, and the manifest records them.')
            return None
        name = self.run_name()
        target = self.run_dir_for_name()
        if target and os.path.isdir(target) and os.listdir(target):
            # SAID, NOT ASKED. A modal question here is a stop, and the
            # name field's whole point is that the same name replaces
            # the run. The log says what is about to be replaced.
            self._log(f'run "{name}" exists -- it will be OVERWRITTEN: its '
                      f'classifier, PSF bank, calibration, report and '
                      f'manifest are removed first; other files in the '
                      f'folder are left alone.')
        cmd = [sys.executable, '-u',
               os.path.join(self._repo, 'tools', 'train_spotmodel.py'),
               b, '--reviewer', who, '--out', name]
        # THE STORE THE BUNDLE CAME FROM, read from its own manifest --
        # analytic_ref and resolution_bound are measured there, and
        # without it they silently fall back to another experiment's
        # defaults. The checked modality is the fallback for a bundle
        # built before manifests carried it.
        sp = None
        try:
            with open(os.path.join(b, 'bundle_manifest.json'),
                      encoding='utf-8') as f:
                sp = json.load(f).get('storage_path')
        except Exception:                                   # noqa: BLE001
            sp = None
        sp = sp or self._store(quiet=True)
        if sp:
            cmd += ['--storage-path', str(sp)]
        else:
            self._log('WARNING: no storage path known for this bundle -- '
                      'the PSF analytic reference and resolution bound '
                      'will fall back to defaults.')
        if self.SetDefaultCheckBox.isChecked():
            cmd.append('--set-default')
        return cmd

    def _train(self):
        if self._busy():
            return
        cmd = self.train_command()
        if not cmd:
            return
        self._log('training in the background:   ' + ' '.join(cmd[2:]))
        self._set_busy(True)
        self.ProgressBar.setRange(0, 0)
        self._last_run_dir = None
        w = StreamingProcWorker(cmd, cwd=self._repo)
        self._worker = w
        w.line.connect(self._on_train_line)
        w.finished_ok.connect(self._on_train_done)
        w.failed.connect(lambda why: self._on_train_done(-1, why))
        w.start()

    def _on_train_line(self, text):
        self._log('  ' + text)
        marker = 'writing the run to '
        if text.strip().startswith(marker):
            self._last_run_dir = text.strip()[len(marker):].strip()

    def _on_train_done(self, code, why=''):
        self._release_worker()
        self._set_busy(False)
        if why:
            self._log('  could not start: ' + why)
            return
        if int(code) != 0:
            self._log(f'training stopped (exit {code}); the lines above are '
                      f'its own output.')
            return
        run = self._last_run_dir
        if not run or not os.path.isdir(run):
            from codelab_pipeline.training import model_store as MS
            runs = MS.available()
            run = runs[0]['path'] if runs else None
        if not run:
            self._log('training finished but no run directory was found.')
            return
        self.ResultTextEdit.setPlainText(self.render_report(run))
        self._log('model ready: ' + run)
        self._refresh_runs(select=run)
        self.model_trained.emit(run)

    @staticmethod
    def render_report(run_dir):
        """report.json as a person reads it."""
        rp = os.path.join(run_dir, 'report.json')
        try:
            with open(rp, encoding='utf-8') as f:
                r = json.load(f)
        except Exception as exc:                            # noqa: BLE001
            return f'{run_dir}\n(no readable report.json: {exc})'
        from codelab_pipeline.training import model_store as MS
        man = MS.read_manifest(run_dir) or {}
        s = r.get('labels') or {}
        lines = [f"run        {os.path.basename(run_dir)}"
                 f"{'   (default model)' if MS.default_model() == run_dir else ''}",
                 f"model_id   {man.get('model_id', '?')}    reviewer "
                 f"{man.get('reviewer', '?')}",
                 f"bundle     {r.get('bundle', '?')}",
                 f"labels     {s.get('positive', 0)} positive, "
                 f"{s.get('negative', 0)} negative, "
                 f"{s.get('contested', 0)} contested, "
                 f"{s.get('undetermined', 0)} undetermined, "
                 f"{s.get('added', 0)} added",
                 f"           over {s.get('crops', 0)} crops, "
                 f"{s.get('groups', 0)} cells, "
                 f"{len(s.get('hybes') or [])} hybe(s), FOVs "
                 f"{s.get('fovs')}",
                 f"boxes      {r.get('n_boxes', '?')}    template "
                 f"r={r.get('template_r')} rz={r.get('template_rz')}"]
        if r.get('provisional'):
            lines.append('PROVISIONAL: too few cells or hybes -- a check '
                         'that the pipeline runs, not a measure of it.')
        lines.append('')
        lines.append('heads      (validation, split by cell)')
        for head, rep in (r.get('heads') or {}).items():
            if 'error' in rep:
                lines.append(f"   {head:7s} FAILED {rep['error']}")
                continue
            flag = ('DEGENERATE' if rep.get('degenerate') else
                    'ALL-FAIL' if rep.get('all_fail') else
                    'ALL-PASS' if rep.get('all_pass') else '')
            lines.append(
                f"   {head:7s} PR-AUC {rep.get('val_pr_auc', float('nan')):.3f}"
                f"  ROC {rep.get('val_roc_auc', float('nan')):.3f}"
                f"  p in [{rep.get('p_min', 0):.3f}, {rep.get('p_max', 0):.3f}]"
                f"  says yes {100 * rep.get('predicted_positive_frac', 0):.0f}%"
                f"  ({rep.get('n_train', '?')} train / "
                f"{rep.get('n_val', '?')} val)  {flag}")
        ms = r.get('multispot') or {}
        if not ms:
            # A RUN OLDER THAN THE REPORT'S multispot BLOCK still ships
            # the calibration file itself, and its meta says what it
            # was fitted on. Read that rather than print nothing about
            # an artefact that is there.
            try:
                with open(os.path.join(run_dir, 'psf_multispot.json'),
                          encoding='utf-8') as f:
                    meta = (json.load(f) or {}).get('meta') or {}
                if meta:
                    ms = {'n': meta.get('n'), 'pillars': meta.get('pillars'),
                          'from_file': True}
            except Exception:                               # noqa: BLE001
                ms = {}
        if ms.get('skipped'):
            lines.append(f"multispot  skipped: {ms['skipped']}")
        elif ms.get('from_file'):
            lines.append(
                f"multispot  {ms.get('n', '?')} judged matches over "
                f"{ms.get('pillars', '?')} pillars   (from psf_multispot.json; "
                f"this run's report predates the multispot block)")
        elif ms:
            lines.append(
                f"multispot  {ms.get('n', '?')} judged matches over "
                f"{ms.get('pillars', '?')} pillars   raw PR-AUC "
                f"{ms.get('raw_pr_auc', float('nan')):.3f}   at 0.5: precision "
                f"{ms.get('precision_at_half', float('nan')):.3f}, recall "
                f"{ms.get('recall_at_half', float('nan')):.3f}")
        psf = r.get('psf') or {}
        if psf:
            ref = psf.get('analytic_ref') or {}
            lines.append(
                f"psf        {psf.get('n_spots', '?')} confirmed spots   "
                f"drift median {psf.get('drift_median', float('nan')):.3f}"
                + (f"   vs analytic {ref.get('family')}: cosine "
                   f"{ref.get('cosine_to_measured', float('nan')):.3f}"
                   if ref else ''))
        for w in r.get('warnings') or []:
            lines.append('WARNING    ' + str(w))
        if man.get('calibrated_at'):
            lines.append(f"calibrated {man['calibrated_at']} from "
                         f"{man.get('calibrated_from', '?')}  -- M1 and PSF "
                         f"are the original run's")
        return '\n'.join(lines)

    # -- busy state -----------------------------------------------------

    def _busy(self):
        if self._worker is not None:
            self._log('something is already running here -- one at a time.')
            return True
        return False

    def _set_busy(self, busy):
        for w in (self.BuildBundlePushButton, self.TrainPushButton):
            w.setEnabled(not busy)
        if not busy:
            self.ProgressBar.setRange(0, 1)
            self.ProgressBar.setValue(0)

    def _on_progress(self, done, total):
        self.ProgressBar.setRange(0, max(int(total), 1))
        self.ProgressBar.setValue(int(done))

    def _release_worker(self):
        w = self._worker
        self._worker = None
        if w is not None:
            w.deleteLater()

    def closeEvent(self, event):
        """Closing HIDES; a running build keeps running and keeps
        reporting here. The main window caches this dialog, so Build
        model... brings it back with its log intact."""
        if self._worker is not None:
            self._log('closed while running -- the build continues in the '
                      'background; reopen Build model... to watch it.')
        event.ignore()
        self.hide()
