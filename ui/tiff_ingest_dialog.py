"""
The from-TIFF ingestion pop-up: the notebook's working style, as a form.

Round-based TIFF experiments differ enough from DAX (one file per FOV
per CHANNEL per session, rounds identified only by the operator's
ordered code list) that the main Ingestion tab's layout-driven flow
doesn't fit -- so this dialog IS the trial table: point at the
experiment directory, every TIFF-holding trial directory becomes a row,
and EVERY parameter -- modality, opener, job code, depth, channels,
fiducial -- lives per row (per explicit redesign: sessions genuinely
differ in job code, channels, even modality). The fields at the top are
a GLOBAL ASSIGNER: set them once, press Assign, then edit individual
rows where a session deviates. Validate before anything writes, then
ingest into the ordinary v2 store. The durable artifacts are GENERATED
ExperimentLayout XLSX files (one per modality) + manifest entries,
after which the app treats this store exactly like a DAX one.
"""
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

from PyQt5 import QtCore, QtWidgets

from codelab_pipeline import process_guard
from codelab_pipeline.io import paths, preprocess
from codelab_pipeline.io import tiff_ingestion as ti

DEFAULT_OPENER = 'M_and_F'

# table columns
COL_DIR, COL_CODES, COL_MODALITY, COL_OPENER, COL_JOB, COL_DEPTH, \
    COL_CHANNELS, COL_FIDUCIAL = range(8)
_HEADERS = ['Trial directory (check to include)',
            'Readout codes (7 / r5 / t10 / b130 / DAPI; 0=skip; '
            'ranges 1-10, r104-110)',
            'Modality', 'Opener', 'Job code', 'Depth', 'Channels',
            'Fiducial']


class TiffIngestionWorker(QtCore.QThread):
    """PER-ROUND tasks through one spawn pool, FOV-major -- so progress
    flows continuously (a whole-trial task surfaced nothing for minutes;
    reported). Each task: (fov, spec, round_index, storage_path,
    modality)."""
    progress = QtCore.pyqtSignal(int, int, str)
    finished_ok = QtCore.pyqtSignal(int, int, list)   # n_errors, n_total, lines
    failed = QtCore.pyqtSignal(str)

    def __init__(self, tasks, overwrite, max_workers):
        super().__init__()
        self.tasks = tasks
        self.overwrite = overwrite
        self.max_workers = max_workers

    def run(self):
        try:
            done, errors = 0, []

            def consume(results):
                nonlocal done
                for fov, folder, err in results:
                    done += 1
                    if err is not None:
                        errors.append(f'FOV{fov:03d} {folder}: {err}')
                    self.progress.emit(done, len(self.tasks),
                                       f'FOV{fov:03d} {folder}: '
                                       + ('ok' if err is None else err))

            if self.max_workers > 1 and len(self.tasks) > 1:
                with ProcessPoolExecutor(
                        max_workers=min(self.max_workers, len(self.tasks)),
                        mp_context=multiprocessing.get_context('spawn'),
                        initializer=process_guard.child_initializer) as ex:
                    futures = [
                        ex.submit(ti.convert_tiff_trial_fov_worker,
                                  fov, spec, storage_path, modality,
                                  overwrite=self.overwrite, rounds=[r])
                        for fov, spec, r, storage_path, modality in self.tasks]
                    for future in as_completed(futures):
                        try:
                            consume(future.result())
                        except Exception as e:
                            done += 1
                            errors.append(f'worker process failed: {e}')
            else:
                for fov, spec, r, storage_path, modality in self.tasks:
                    consume(ti.convert_tiff_trial_fov_worker(
                        fov, spec, storage_path, modality,
                        overwrite=self.overwrite, rounds=[r]))
            self.finished_ok.emit(len(errors), len(self.tasks), errors)
        except Exception as e:
            self.failed.emit(f'{type(e).__name__}: {e}')


class TiffIngestDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Ingest From TIFF (round-based trials)')
        self.resize(1250, 760)
        self._worker = None
        self._validator = None
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self.ProjectRootLineEdit = QtWidgets.QLineEdit()
        form.addRow('Project root (v2 store):',
                    self._with_browse(self.ProjectRootLineEdit))
        self.ExperimentDirLineEdit = QtWidgets.QLineEdit()
        self.ExperimentDirLineEdit.editingFinished.connect(
            self._populate_trials)
        form.addRow('Experiment dir (holds trial dirs):',
                    self._with_browse(self.ExperimentDirLineEdit,
                                      on_set=self._populate_trials))
        layout.addLayout(form)

        # -- the GLOBAL ASSIGNER: one set of values, pushed onto every
        # trial row by the button; rows stay individually editable --
        assignGroup = QtWidgets.QGroupBox(
            'Global parameters (Assign fills every trial row; edit rows '
            'individually afterward)')
        aRow = QtWidgets.QHBoxLayout(assignGroup)
        self.ModalityLineEdit = QtWidgets.QLineEdit('DNA')
        self.OpenerLineEdit = QtWidgets.QLineEdit(DEFAULT_OPENER)
        self.JobNameLineEdit = QtWidgets.QLineEdit('Job 2')
        self.DepthLineEdit = QtWidgets.QLineEdit('129')
        self.ChannelsLineEdit = QtWidgets.QLineEdit('555, 647, 488')
        self.ChannelsLineEdit.setToolTip(
            'Channel names for slots ch00, ch01, ... in order. Numeric '
            'only. The FIRST non-fiducial channel becomes THE readout '
            'channel for alignment/tracing pickers.')
        self.FiducialLineEdit = QtWidgets.QLineEdit('555')
        for label, w in (('Modality', self.ModalityLineEdit),
                         ('Opener', self.OpenerLineEdit),
                         ('Job code', self.JobNameLineEdit),
                         ('Depth', self.DepthLineEdit),
                         ('Channels (=ch00,ch01,..)', self.ChannelsLineEdit),
                         ('Fiducial', self.FiducialLineEdit)):
            aRow.addWidget(QtWidgets.QLabel(label + ':'))
            aRow.addWidget(w, stretch=2 if w is self.ChannelsLineEdit else 1)
        self.AssignPushButton = QtWidgets.QPushButton('Assign to all trials')
        self.AssignPushButton.clicked.connect(self.assign_globals)
        aRow.addWidget(self.AssignPushButton)
        layout.addWidget(assignGroup)

        self.TrialTableWidget = QtWidgets.QTableWidget(0, len(_HEADERS))
        self.TrialTableWidget.setHorizontalHeaderLabels(_HEADERS)
        hdr = self.TrialTableWidget.horizontalHeader()
        hdr.setSectionResizeMode(COL_DIR, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(COL_CODES, QtWidgets.QHeaderView.Stretch)
        layout.addWidget(self.TrialTableWidget, stretch=2)

        fovRow = QtWidgets.QHBoxLayout()
        fovRow.addWidget(QtWidgets.QLabel('FOVs:'))
        self.FovListLineEdit = QtWidgets.QLineEdit()
        self.FovListLineEdit.setPlaceholderText(
            'e.g. 1-50  (Auto-discover reads the first checked trial)')
        fovRow.addWidget(self.FovListLineEdit, stretch=1)
        self.DiscoverFovsPushButton = QtWidgets.QPushButton('Auto-discover')
        self.DiscoverFovsPushButton.clicked.connect(self._discover_fovs)
        fovRow.addWidget(self.DiscoverFovsPushButton)
        fovRow.addWidget(QtWidgets.QLabel('Mode:'))
        self.ModeComboBox = QtWidgets.QComboBox()
        self.ModeComboBox.addItems(['append (skip complete rounds)',
                                    'overwrite (rebuild everything)'])
        fovRow.addWidget(self.ModeComboBox)
        fovRow.addWidget(QtWidgets.QLabel('Workers:'))
        self.WorkersSpinBox = QtWidgets.QSpinBox()
        ceiling = preprocess.max_ingestion_workers()
        self.WorkersSpinBox.setRange(1, ceiling)
        self.WorkersSpinBox.setValue(min(12, ceiling))
        fovRow.addWidget(self.WorkersSpinBox)
        layout.addLayout(fovRow)

        btnRow = QtWidgets.QHBoxLayout()
        self.ValidatePushButton = QtWidgets.QPushButton(
            'Validate (dry run: files + page counts)')
        self.ValidatePushButton.clicked.connect(self.validate)
        btnRow.addWidget(self.ValidatePushButton)
        self.IngestPushButton = QtWidgets.QPushButton(
            'Ingest (generate layouts + convert)')
        self.IngestPushButton.clicked.connect(self.ingest)
        btnRow.addWidget(self.IngestPushButton)
        layout.addLayout(btnRow)

        self.ProgressBar = QtWidgets.QProgressBar()
        layout.addWidget(self.ProgressBar)
        self.LogListWidget = QtWidgets.QListWidget()
        layout.addWidget(self.LogListWidget, stretch=1)

    # -- helpers -----------------------------------------------------------
    def _with_browse(self, line_edit, on_set=None):
        row = QtWidgets.QHBoxLayout()
        row.addWidget(line_edit, stretch=1)
        btn = QtWidgets.QPushButton('Browse...')

        def browse():
            path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Pick')
            if path:
                line_edit.setText(path)
                if on_set:
                    on_set()
        btn.clicked.connect(browse)
        row.addWidget(btn)
        return row

    def _log(self, msg):
        self.LogListWidget.addItem(msg)
        self.LogListWidget.scrollToBottom()

    def _cell(self, r, c):
        item = self.TrialTableWidget.item(r, c)
        return item.text().strip() if item else ''

    def _set_cell(self, r, c, text):
        self.TrialTableWidget.setItem(r, c,
                                      QtWidgets.QTableWidgetItem(text))

    def _populate_trials(self):
        exp = self.ExperimentDirLineEdit.text().strip()
        self.TrialTableWidget.setRowCount(0)
        if not exp or not os.path.isdir(exp):
            return
        names = ti.discover_trials(exp)
        if not names:
            self._log('no directory here (or its subdirectories) holds '
                      'any .tif files')
        for name in names:
            r = self.TrialTableWidget.rowCount()
            self.TrialTableWidget.insertRow(r)
            shown = '(this directory)' if name == '.' else name
            item = QtWidgets.QTableWidgetItem(shown)
            item.setData(QtCore.Qt.UserRole, name)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked)
            self.TrialTableWidget.setItem(r, COL_DIR, item)
            for c in range(1, len(_HEADERS)):
                self._set_cell(r, c, '')

    def assign_globals(self):
        """Push the global fields onto EVERY trial row -- the one-set/
        many-rows working style; per-row deviations are edited after."""
        for r in range(self.TrialTableWidget.rowCount()):
            self._set_cell(r, COL_MODALITY, self.ModalityLineEdit.text())
            self._set_cell(r, COL_OPENER, self.OpenerLineEdit.text())
            self._set_cell(r, COL_JOB, self.JobNameLineEdit.text())
            self._set_cell(r, COL_DEPTH, self.DepthLineEdit.text())
            self._set_cell(r, COL_CHANNELS, self.ChannelsLineEdit.text())
            self._set_cell(r, COL_FIDUCIAL, self.FiducialLineEdit.text())
        self._log(f'assigned globals to '
                  f'{self.TrialTableWidget.rowCount()} trial row(s)')

    def _trial_specs(self):
        """Harvest one SELF-CONTAINED spec per checked row; raises
        ValueError naming the row for anything missing/unparseable."""
        exp = self.ExperimentDirLineEdit.text().strip()
        specs = []
        for r in range(self.TrialTableWidget.rowCount()):
            item = self.TrialTableWidget.item(r, COL_DIR)
            if item is None or item.checkState() != QtCore.Qt.Checked:
                continue
            name = item.data(QtCore.Qt.UserRole) or item.text()
            label = item.text()
            codes_text = self._cell(r, COL_CODES)
            if not codes_text:
                raise ValueError(f'{label}: readout codes are empty -- type '
                                 f'the ordered rounds (or uncheck the row)')
            try:
                codes = ti.parse_readout_codes(codes_text)
            except ValueError as e:
                raise ValueError(f'{label}: {e}')
            channels = [c.strip() for c in
                        self._cell(r, COL_CHANNELS).split(',') if c.strip()]
            spec = {'path': os.path.normpath(os.path.join(exp, name)),
                    'codes': codes,
                    'modality': self._cell(r, COL_MODALITY),
                    'opener': self._cell(r, COL_OPENER),
                    'job_name': self._cell(r, COL_JOB),
                    'depth': self._cell(r, COL_DEPTH),
                    'channels': channels,
                    'fiducial_channel': self._cell(r, COL_FIDUCIAL)}
            try:
                ti.check_spec(spec)
            except ValueError as e:
                raise ValueError(str(e))
            specs.append(spec)
        if not specs:
            raise ValueError('no trial rows checked')
        return specs

    def _fovs(self):
        fovs, seen = [], set()
        for chunk in re.split(r'[,\s]+', self.FovListLineEdit.text().strip()):
            if not chunk:
                continue
            if '-' in chunk:
                a, b = (int(x) for x in chunk.split('-', 1))
                new = range(a, b + 1)
            else:
                new = [int(chunk)]
            for f in new:
                if f not in seen:
                    seen.add(f)
                    fovs.append(f)
        if not fovs:
            raise ValueError('FOV list is empty -- Auto-discover or type it')
        return fovs

    def _discover_fovs(self):
        try:
            spec = self._trial_specs()[0]
        except ValueError as e:
            self._log(f'auto-discover: {e}')
            return
        fovs = ti.discover_fovs(spec['path'], spec['opener'],
                                spec['job_name'])
        if not fovs:
            self._log(f'auto-discover: no files in '
                      f'{os.path.basename(spec["path"])} match '
                      f'{spec["opener"]}_Pos*__{spec["job_name"]}_*_RAW_'
                      f'ch*.tif')
            return
        self.FovListLineEdit.setText(
            f'{fovs[0]}-{fovs[-1]}' if fovs == list(
                range(fovs[0], fovs[-1] + 1))
            else ','.join(map(str, fovs)))
        self._log(f'discovered {len(fovs)} FOV(s) in '
                  f'{os.path.basename(spec["path"])}')

    # -- actions -----------------------------------------------------------
    def validate(self):
        """BACKGROUND, always (an earlier draft froze the GUI for the
        NAS walk; reported): per trial, one ~1 s arithmetic page count
        plus listdir/size checks."""
        try:
            specs = self._trial_specs()
            fovs = self._fovs()
            ti.synthesize_hybe_records(specs)   # codes/duplicates/channels
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, 'Validate', str(e))
            return
        self.ValidatePushButton.setEnabled(False)
        self.IngestPushButton.setEnabled(False)
        self._log(f'validating {len(specs)} trial(s) x {len(fovs)} FOV(s) '
                  f'in the background (listdir + one ~1 s page count per '
                  f'trial + size checks)...')

        class _Validate(QtCore.QThread):
            line = QtCore.pyqtSignal(str)
            done = QtCore.pyqtSignal(int)

            def run(worker):
                n_bad = 0
                try:
                    for spec in specs:
                        problems, note = ti.validate_trial(spec, fovs)
                        worker.line.emit(
                            f'{os.path.basename(spec["path"])}: {note}')
                        for fov, msg in problems:
                            n_bad += 1
                            worker.line.emit(
                                f'{os.path.basename(spec["path"])} '
                                f'FOV{fov:03d}: {msg}')
                except Exception as e:
                    n_bad += 1
                    worker.line.emit(f'validate FAILED: '
                                     f'{type(e).__name__}: {e}')
                worker.done.emit(n_bad)

        self._validator = _Validate()
        self._validator.line.connect(self._log)
        self._validator.done.connect(self._on_validated)
        self._validator.start()

    def _on_validated(self, n_bad):
        self.ValidatePushButton.setEnabled(True)
        self.IngestPushButton.setEnabled(True)
        self._log(f'validate: {n_bad} problem(s)')
        if n_bad == 0:
            self._log('validate: all files present with the declared '
                      'rounds -- ready to ingest')

    def ingest(self):
        try:
            specs = self._trial_specs()
            fovs = self._fovs()
            dp = self.ProjectRootLineEdit.text().strip()
            if not dp:
                raise ValueError('set the project root')
            records_by_modality = ti.synthesize_hybe_records(specs)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, 'Ingest', str(e))
            return
        # the durable description first: one generated layout PER
        # MODALITY + manifest entries -- after this the store is
        # indistinguishable from a DAX one
        os.makedirs(dp, exist_ok=True)
        manifest = paths.read_manifest(dp) or {}
        names = list(manifest.get('modalities', {}))
        layout_paths = {m: manifest.get('modalities', {}).get(m, {})
                        .get('layout_path', '') for m in names}
        dax_dirs = {m: manifest.get('modalities', {}).get(m, {})
                    .get('dax_directory', '') for m in names}
        try:
            for modality, records in records_by_modality.items():
                xlsx = ti.write_layout_xlsx(
                    records,
                    os.path.join(dp, f'{modality}_ExperimentLayout.xlsx'))
                if modality not in names:
                    names.append(modality)
                layout_paths[modality] = xlsx
                dax_dirs[modality] = self.ExperimentDirLineEdit.text().strip()
                self._log(f'generated layout: {xlsx}')
        except PermissionError:
            # the reported "nothing proceeding": the previous run's
            # generated xlsx was open in Excel, and the replace died
            # inside a Qt slot
            QtWidgets.QMessageBox.warning(
                self, 'Ingest',
                'A generated ExperimentLayout xlsx is locked -- close it '
                '(Excel keeps it open) and press Ingest again.')
            return
        paths.write_manifest(dp, names, layout_paths=layout_paths,
                             dax_directories=dax_dirs)

        overwrite = self.ModeComboBox.currentIndex() == 1
        # named rounds' auto-assigned Readouts indices, so worker-written
        # stack attrs agree with the generated layout
        for spec in specs:
            spec['auto_ids'] = ti.auto_ids_for(
                records_by_modality[spec['modality']])
        # PER-ROUND tasks, FOV-major: progress flows round by round
        tasks = []
        for fov in fovs:
            for spec in specs:
                storage_path = os.path.join(dp, spec['modality'])
                for r, code in enumerate(spec['codes']):
                    if ti.folder_for(code) is None:
                        continue
                    tasks.append((fov, spec, r, storage_path,
                                  spec['modality']))
        n_ch = max(len(s['channels']) for s in specs)
        mb = max(int(s['depth']) for s in specs) * n_ch * 2  # ~MB/round @1k^2
        self.ProgressBar.setRange(0, 0)      # busy until the first result
        self._log(f'submitted {len(tasks)} round-task(s) to '
                  f'{self.WorkersSpinBox.value()} worker(s); each round '
                  f'reads ~{mb} MB from the NAS, so the first results '
                  f'land after a minute or two of quiet -- the window '
                  f'stays live')
        self.IngestPushButton.setEnabled(False)
        self._worker = TiffIngestionWorker(tasks, overwrite,
                                           self.WorkersSpinBox.value())
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress(self, done, total, msg):
        if self.ProgressBar.maximum() == 0:
            self.ProgressBar.setRange(0, total)
        self.ProgressBar.setValue(done)
        if 'ok' not in msg or done % 25 == 0 or done == total:
            self._log(f'[{done}/{total}] {msg}')

    def _on_finished(self, n_errors, n_total, error_lines):
        self.IngestPushButton.setEnabled(True)
        self.ProgressBar.setRange(0, max(n_total, 1))
        self.ProgressBar.setValue(n_total)
        for line in error_lines[:30]:
            self._log('ERROR ' + line)
        self._log(f'ingestion finished: {n_total - n_errors}/{n_total} '
                  f'round(s) ok, {n_errors} error(s)')
        QtWidgets.QMessageBox.information(
            self, 'Ingest From TIFF',
            f'{n_total - n_errors}/{n_total} rounds ingested'
            + (f', {n_errors} ERROR(S) -- see the log' if n_errors else '')
            + '.\n\nThe generated ExperimentLayout(s) are registered in the '
              'manifest: activate the project root from the Ingestion tab '
              'and the store behaves like any other.')

    def _on_failed(self, message):
        self.IngestPushButton.setEnabled(True)
        self.ProgressBar.setRange(0, 1)
        QtWidgets.QMessageBox.critical(self, 'Ingest From TIFF', message)
