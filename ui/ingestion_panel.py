import os
from concurrent.futures import ThreadPoolExecutor

from PyQt5 import QtWidgets, QtCore, QtGui

from codelab_pipeline.io import preprocess

# The microscope's voxel size, in micrometres. Experiment-level, so it
# lives here rather than on whichever analysis panel happened to need it
# first. Defaults are this lab's own; every store should set its own.
VOXEL_DEFAULTS = {'voxel_xy_um': 0.208, 'voxel_z_um': 0.2}


# {(dax_directory, folder): bool} -- session cache of "does this hybe
# folder hold real .dax data". Raw acquisition data essentially never
# disappears mid-session, and the probe is the measured bottleneck of
# Parse Layouts: one listdir per hybe folder over the DAX NAS share, 43 ms
# a round-trip on the real store, 222 of them serial for a two-modality
# 111-hybe project = ~9.5 s of every click (~55% of the whole parse) --
# for information that was identical to the click before. Cached, a
# re-parse pays zero DAX round-trips; the ONE explicit invalidation is
# re-confirming a modality's DAX field (editingFinished ->
# refresh_hybe_checks probes fresh and overwrites the cache).
_DAX_PRESENCE_CACHE = {}


def _probe_dax_presence(pairs, refresh=False):
    """
    {(dax_directory, folder): bool} for every requested pair, probing
    uncached (or, with refresh=True, ALL) pairs through a thread pool --
    the probes are pure SMB metadata waits (os.listdir releases the GIL),
    so overlapping them is where the parse-speed win lives; 16 in flight
    matches the storage-bound worker counts used elsewhere in this app.
    Callers pass only pairs with a REAL dax_directory -- the blank-
    directory default (checked, never probed, see _has_dax_data) is the
    caller's own branch, kept out of the cache so a later real directory
    can't be shadowed by a cached blank-default True.
    """
    unique = list(dict.fromkeys(pairs))
    todo = [p for p in unique if refresh or p not in _DAX_PRESENCE_CACHE]
    if todo:
        with ThreadPoolExecutor(max_workers=min(16, len(todo))) as pool:
            for pair, present in zip(todo, pool.map(
                    lambda p: IngestionPanelUI._has_dax_data(p[0], p[1]), todo)):
                _DAX_PRESENCE_CACHE[pair] = present
    return {p: _DAX_PRESENCE_CACHE[p] for p in unique}


class IngestionPanelUI(object):
    """
    The combinatorial ingestion form: EVERY modality's own
    (ExperimentLayout.xlsx, DAX repository) input pair is visible at once,
    with ONE project root (storage) and ONE FOV list shared by all of
    them -- per explicit design, there is no per-modality storage path or
    FOV list left to mismatch (storage_path IS <project root>/<modality>,
    the v2 layout contract in io/paths.py). Parse Layouts parses every
    configured modality; the combined hybe checklist below concatenates
    each modality's hybes (each row tagged + keyed by (folder, modality));
    Run Ingestion ingests exactly the checked set, every modality in ONE
    FOV-major combined run (see IngestionWorker).
    """
    def setupUi(self, Widget):
        Widget.setObjectName('IngestionPanel')
        layout = QtWidgets.QVBoxLayout(Widget)

        # -- modality setup: how many modalities and what they're named.
        # Count/names default to 2 (DNA, RNA) so nothing changes for
        # anyone who never touches this group -- Set/Activate only matter
        # for a different count or custom names. --
        modalityGroup = QtWidgets.QGroupBox('Modality Setup')
        modalityLayout = QtWidgets.QVBoxLayout(modalityGroup)

        countRow = QtWidgets.QWidget()
        countRowLayout = QtWidgets.QHBoxLayout(countRow)
        countRowLayout.setContentsMargins(0, 0, 0, 0)
        countRowLayout.addWidget(QtWidgets.QLabel('# modalities:'))
        self.NumModalitiesLineEdit = QtWidgets.QLineEdit('2')
        self.NumModalitiesLineEdit.setValidator(QtGui.QIntValidator(1, 20))
        self.NumModalitiesLineEdit.setMaximumWidth(60)
        self.SetNumModalitiesPushButton = QtWidgets.QPushButton('Set')
        countRowLayout.addWidget(self.NumModalitiesLineEdit)
        countRowLayout.addWidget(self.SetNumModalitiesPushButton)
        countRowLayout.addStretch()
        modalityLayout.addWidget(countRow)

        self.ModalityNamesContainer = QtWidgets.QWidget()
        self.ModalityNamesLayout = QtWidgets.QFormLayout(self.ModalityNamesContainer)
        self.ModalityNamesLayout.setContentsMargins(0, 0, 0, 0)
        modalityLayout.addWidget(self.ModalityNamesContainer)
        self.ModalityNameComboBoxes = []

        self.ActivateModalitiesPushButton = QtWidgets.QPushButton('Activate Modalities')
        modalityLayout.addWidget(self.ActivateModalitiesPushButton)

        # -- The modality registry lives HERE, on the ingestion panel, not
        # on MainWindow. Per explicit principle: modality is an INGESTION
        # concept -- each modality needs its own ExperimentLayout + DAX
        # repository -- and nothing anywhere may hold a "current
        # modality". There IS no current_modality any more: the
        # combinatorial form shows every modality's inputs at once (per
        # explicit design), so even the panel-display-state version of
        # the concept is gone. Non-ingestion code reads the REGISTRY
        # (modality_names order + modality_data paths) through
        # MainWindow's accessor helpers, and derives any per-datum
        # modality from the datum itself ((hybe, modality) keys,
        # cell.modality, spot.modality, vlinks_store.modality_of).
        # modality_names ORDER is semantic and must be preserved: the
        # first configured entry is the cross-modal hub frame
        # (_shared_frame_modality), and a loaded config's modality order
        # means "first = hub".
        self.modality_names = ['DNA', 'RNA']
        self.modality_data = {}           # {name: modality-state dict}

        layout.addWidget(modalityGroup)
        self.build_modality_name_fields(2)

        # -- combinatorial inputs: one (layout, DAX) row pair per
        # modality, all always visible; rebuilt wholesale by
        # build_modality_input_rows on activation, so MainWindow re-wires
        # the rows' signals after every rebuild (same
        # rebuild-then-reconnect convention as AlignmentPanel's
        # build_cell_reference_hybe_fields combos). --
        self.ModalityInputsContainer = QtWidgets.QWidget()
        self.ModalityInputsLayout = QtWidgets.QFormLayout(self.ModalityInputsContainer)
        self.ModalityInputsLayout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.ModalityInputsContainer)
        self.modality_input_rows = {}     # {name: {'layout': QLineEdit, 'dax': QLineEdit}}
        self.build_modality_input_rows(self.modality_names)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.ProjectRootLineEdit, projectRootRow = self._path_row('Select project root (storage) directory',
                                                                  is_file=False)
        form.addRow('Project root (storage):', projectRootRow)

        self.FovListLineEdit = QtWidgets.QLineEdit()
        self.FovListLineEdit.setPlaceholderText('e.g. 1,2,3 or 1-10,15,20-25 or 1 2 3 (comma and/or space separated)')
        form.addRow('FOV list (all modalities):', self.FovListLineEdit)

        # VOXEL SIZE IS A PROPERTY OF THE MICROSCOPE, not of one analysis
        # step. It lived on the chromatin-tracing panel because v2 was the
        # first thing to need it, which made it look like a tracing
        # setting; it is entered once here instead, travels in the config,
        # and every consumer reads it from one place.
        #
        # Lateral and axial are SEPARATE inputs and neither is derived from
        # the other. On this microscope a pixel is 0.208 um and a plane
        # 0.2 um -- close enough that assuming a ratio looks harmless, and
        # v1's axial gate did exactly that (2x the lateral, in pixels) and
        # was wrong by construction.
        voxelRow = QtWidgets.QWidget()
        voxelLayout = QtWidgets.QHBoxLayout(voxelRow)
        voxelLayout.setContentsMargins(0, 0, 0, 0)
        self.VoxelXYSpinBox = QtWidgets.QDoubleSpinBox()
        self.VoxelXYSpinBox.setDecimals(4)
        self.VoxelXYSpinBox.setRange(0.0001, 10.0)
        self.VoxelXYSpinBox.setSingleStep(0.001)
        self.VoxelXYSpinBox.setValue(VOXEL_DEFAULTS['voxel_xy_um'])
        self.VoxelXYSpinBox.setSuffix(' um/px')
        self.VoxelZSpinBox = QtWidgets.QDoubleSpinBox()
        self.VoxelZSpinBox.setDecimals(4)
        self.VoxelZSpinBox.setRange(0.0001, 20.0)
        self.VoxelZSpinBox.setSingleStep(0.001)
        self.VoxelZSpinBox.setValue(VOXEL_DEFAULTS['voxel_z_um'])
        self.VoxelZSpinBox.setSuffix(' um/plane')
        voxelLayout.addWidget(QtWidgets.QLabel('lateral'))
        voxelLayout.addWidget(self.VoxelXYSpinBox)
        voxelLayout.addWidget(QtWidgets.QLabel('axial'))
        voxelLayout.addWidget(self.VoxelZSpinBox)
        voxelLayout.addStretch(1)
        form.addRow('Voxel size:', voxelRow)

        self.ParseLayoutPushButton = QtWidgets.QPushButton('Parse Layouts (all modalities)')
        layout.addWidget(self.ParseLayoutPushButton)

        layout.addWidget(QtWidgets.QLabel('Hybes to ingest (all modalities combined -- [modality] tags each row):'))
        self.HybeListWidget = QtWidgets.QListWidget()
        # ExtendedSelection (not NoSelection) -- lets the user click+drag,
        # shift-click, or ctrl-click to highlight many rows at once, then
        # batch-toggle them via the two buttons below instead of clicking
        # each checkbox individually.
        self.HybeListWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.HybeListWidget)

        hybeCheckButtonsRow = QtWidgets.QWidget()
        hybeCheckButtonsLayout = QtWidgets.QHBoxLayout(hybeCheckButtonsRow)
        hybeCheckButtonsLayout.setContentsMargins(0, 0, 0, 0)
        self.CheckSelectedHybesPushButton = QtWidgets.QPushButton('Check Selected')
        self.UncheckSelectedHybesPushButton = QtWidgets.QPushButton('Uncheck Selected')
        hybeCheckButtonsLayout.addWidget(self.CheckSelectedHybesPushButton)
        hybeCheckButtonsLayout.addWidget(self.UncheckSelectedHybesPushButton)
        layout.addWidget(hybeCheckButtonsRow)
        self.CheckSelectedHybesPushButton.clicked.connect(lambda: self._set_selected_hybe_check_state(QtCore.Qt.Checked))
        self.UncheckSelectedHybesPushButton.clicked.connect(lambda: self._set_selected_hybe_check_state(QtCore.Qt.Unchecked))

        workersRow = QtWidgets.QWidget()
        workersLayout = QtWidgets.QHBoxLayout(workersRow)
        workersLayout.setContentsMargins(0, 0, 0, 0)
        workersLayout.addWidget(QtWidgets.QLabel('Parallel workers:'))
        self.IngestWorkersSpinBox = QtWidgets.QSpinBox()
        # The ceiling is derived from THIS machine's cores and RAM rather
        # than the flat 16 it used to be -- see
        # preprocess.max_ingestion_workers. The DEFAULT deliberately stays
        # 4: the ceiling describes what the box can host, which is a
        # different question from what the storage wants, and the storage
        # is what actually limits this work.
        worker_ceiling = preprocess.max_ingestion_workers()
        self.IngestWorkersSpinBox.setRange(1, worker_ceiling)
        self.IngestWorkersSpinBox.setValue(min(4, worker_ceiling))
        self.IngestWorkersSpinBox.setToolTip(
            'DAX->H5 conversions run in this many separate processes. The work '
            'is I/O-bound (huge sequential reads, often from a network share), '
            'so 3-4 overlaps read latency well off a share; a local NVMe on a '
            'many-core box takes considerably more before extra workers stop '
            'helping. 1 = the old sequential behavior.\n\n'
            'The maximum here ({}) comes from this machine: cores minus two, '
            'and 60% of RAM divided by the ~5 GB one conversion holds '
            '(read_dax loads an entire DAX at once).'.format(worker_ceiling))
        workersLayout.addWidget(self.IngestWorkersSpinBox)
        workersLayout.addStretch(1)
        layout.addWidget(workersRow)

        self.RunIngestionPushButton = QtWidgets.QPushButton('Run Ingestion')
        self.RunIngestionPushButton.setEnabled(False)
        self.RunIngestionPushButton.setToolTip(
            'Ingests exactly the checked hybes above -- every modality in ONE combined run: each '
            "FOV's hybes are converted across every checked modality before any modality's hybes "
            'for the next FOV, so early FOVs finish across every modality together (they can be '
            'analyzed while the rest still ingests). One shared Parallel workers pool bounds the '
            'whole run regardless of how many modalities are checked. To ingest a single modality, '
            "uncheck the other modalities' rows.")
        layout.addWidget(self.RunIngestionPushButton)

        self.CheckIngestionStatusPushButton = QtWidgets.QPushButton('Check Ingestion Status')
        layout.addWidget(self.CheckIngestionStatusPushButton)
        self.IngestionStatusTextEdit = QtWidgets.QTextEdit()
        self.IngestionStatusTextEdit.setReadOnly(True)
        self.IngestionStatusTextEdit.setPlaceholderText(
            'Per-modality, per-FOV hybe readiness (is each hybe really ingested under '
            '{project root}/{modality}?) shows here after Parse Layouts + Check Ingestion Status.')
        self.IngestionStatusTextEdit.setMaximumHeight(140)
        layout.addWidget(self.IngestionStatusTextEdit)

        self.ShowCellSpotStatusDisplayerPushButton = QtWidgets.QPushButton('Show Cell/Spot Status Detail...')
        layout.addWidget(self.ShowCellSpotStatusDisplayerPushButton)

        viewerGroup = QtWidgets.QGroupBox('MIP Viewer (visually spot-check an ingested FOV/hybe/channel)')
        viewerForm = QtWidgets.QFormLayout(viewerGroup)
        self.ViewerFovSpinBox = QtWidgets.QSpinBox()
        self.ViewerFovSpinBox.setRange(1, 100000)
        self.ViewerFovSpinBox.setValue(1)
        viewerForm.addRow('FOV:', self.ViewerFovSpinBox)
        self.ViewerHybeComboBox = QtWidgets.QComboBox()
        viewerForm.addRow('Hybe:', self.ViewerHybeComboBox)
        self.ViewerChannelComboBox = QtWidgets.QComboBox()
        viewerForm.addRow('Channel:', self.ViewerChannelComboBox)
        self.ShowMipViewerPushButton = QtWidgets.QPushButton('Show MIP Viewer')
        viewerForm.addRow(self.ShowMipViewerPushButton)
        layout.addWidget(viewerGroup)
        self.ViewerHybeComboBox.currentIndexChanged.connect(self._on_viewer_hybe_changed)

        layout.addWidget(QtWidgets.QLabel('Job queue (layout + directory + storage path, batched):'))
        jobButtonsRow = QtWidgets.QWidget()
        jobButtonsLayout = QtWidgets.QHBoxLayout(jobButtonsRow)
        jobButtonsLayout.setContentsMargins(0, 0, 0, 0)
        self.AddJobPushButton = QtWidgets.QPushButton('Add to Queue')
        self.RemoveJobPushButton = QtWidgets.QPushButton('Remove Selected')
        self.RunQueuePushButton = QtWidgets.QPushButton('Run Queued Jobs')
        jobButtonsLayout.addWidget(self.AddJobPushButton)
        jobButtonsLayout.addWidget(self.RemoveJobPushButton)
        jobButtonsLayout.addWidget(self.RunQueuePushButton)
        layout.addWidget(jobButtonsRow)

        self.JobQueueListWidget = QtWidgets.QListWidget()
        self.JobQueueListWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.JobQueueListWidget)

        self.ProgressBar = QtWidgets.QProgressBar()
        layout.addWidget(self.ProgressBar)

        # The panel log boxes moved into the one combined log window (see
        # ui/log_window.py) -- the stretch keeps the controls top-anchored
        # where the log box used to soak up the leftover height.
        layout.addStretch(1)

    def _path_row(self, dialog_title, is_file, name_filter='All files (*)'):
        row = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        line_edit = QtWidgets.QLineEdit()
        browse_button = QtWidgets.QPushButton('Browse...')

        def browse():
            if is_file:
                path, _ = QtWidgets.QFileDialog.getOpenFileName(row, dialog_title, '', name_filter)
            else:
                path = QtWidgets.QFileDialog.getExistingDirectory(row, dialog_title)
            if path:
                line_edit.setText(path)
                # setText() alone doesn't fire editingFinished (only Enter/
                # focus-loss does) -- emit it explicitly so a Browse-picked
                # path behaves identically to a typed-then-confirmed one for
                # anything connected to editingFinished (e.g. the DAX
                # directory's hybe-checkbox refresh).
                line_edit.editingFinished.emit()

        browse_button.clicked.connect(browse)
        row_layout.addWidget(line_edit)
        row_layout.addWidget(browse_button)
        return line_edit, row

    def build_modality_name_fields(self, n):
        """
        Rebuild the N modality-name entry combos (editable -- DNA/RNA
        quick-pick, or type anything). Defaults: n==1 -> RNA; n==2 -> DNA,
        RNA (in that order); n>2 -> blank, placeholder-only (no sane
        default name beyond the first two).
        """
        while self.ModalityNamesLayout.rowCount():
            self.ModalityNamesLayout.removeRow(0)
        self.ModalityNameComboBoxes = []
        for i in range(n):
            combo = QtWidgets.QComboBox()
            combo.setEditable(True)
            combo.addItems(['DNA', 'RNA'])
            if n == 1:
                combo.setCurrentText('RNA')
            elif n == 2:
                combo.setCurrentText('DNA' if i == 0 else 'RNA')
            else:
                combo.setCurrentText('')
                combo.lineEdit().setPlaceholderText(f'Modality {i + 1} name')
            self.ModalityNameComboBoxes.append(combo)
            self.ModalityNamesLayout.addRow(f'Modality {i + 1}:', combo)

    def modality_name_values(self):
        return [c.currentText().strip() for c in self.ModalityNameComboBoxes]

    def build_modality_input_rows(self, names):
        """
        Rebuild the per-modality (ExperimentLayout.xlsx, DAX directory)
        row pairs -- one pair per modality, ALL always visible: the
        combinatorial form has no current-modality switching, per
        explicit design. Rebuilding replaces the row widgets wholesale,
        so MainWindow must re-wire their signals after every call
        (_wire_modality_input_rows -- same rebuild-then-reconnect
        convention as AlignmentPanel's build_cell_reference_hybe_fields).
        """
        while self.ModalityInputsLayout.rowCount():
            self.ModalityInputsLayout.removeRow(0)
        self.modality_input_rows = {}
        for name in names:
            layout_edit, layout_row = self._path_row(f'Select {name} ExperimentLayout.xlsx', is_file=True,
                                                     name_filter='Excel files (*.xlsx)')
            self.ModalityInputsLayout.addRow(f'{name} ExperimentLayout.xlsx:', layout_row)
            dax_edit, dax_row = self._path_row(f'Select {name} DAX directory', is_file=False)
            self.ModalityInputsLayout.addRow(f'{name} DAX directory:', dax_row)
            self.modality_input_rows[name] = {'layout': layout_edit, 'dax': dax_edit}

    def modality_input_values(self):
        """{name: {'layout_path':..., 'dax_directory':...}} read live off the N row pairs."""
        return {name: {'layout_path': edits['layout'].text().strip(),
                       'dax_directory': edits['dax'].text().strip()}
                for name, edits in self.modality_input_rows.items()}

    def set_modality_input_values(self, fields_by_name):
        """Fill the row pairs from {name: {'layout_path', 'dax_directory'}} -- names without a row (not activated) are ignored."""
        for name, fields in (fields_by_name or {}).items():
            edits = self.modality_input_rows.get(name)
            if not edits:
                continue
            edits['layout'].setText(fields.get('layout_path', ''))
            edits['dax'].setText(fields.get('dax_directory', ''))

    def lock_modality_setup(self):
        self.NumModalitiesLineEdit.setEnabled(False)
        self.SetNumModalitiesPushButton.setEnabled(False)
        for combo in self.ModalityNameComboBoxes:
            combo.setEnabled(False)
        self.ActivateModalitiesPushButton.setEnabled(False)

    def _set_selected_hybe_check_state(self, check_state):
        """Batch-apply check_state to every currently-selected (highlighted) row in HybeListWidget."""
        for item in self.HybeListWidget.selectedItems():
            item.setCheckState(check_state)

    def hybe_checkbox_items(self):
        """
        Currently-checked (hybe_folder, modality) pairs from the combined
        HybeListWidget. The modality element is load-bearing, not
        cosmetic: hybe folder names legitimately collide across
        modalities (the cross-modal bridge hybe exists in BOTH layouts),
        so a bare folder name cannot identify a checked row.
        """
        checked = []
        for i in range(self.HybeListWidget.count()):
            item = self.HybeListWidget.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                checked.append(item.data(QtCore.Qt.UserRole))
        return checked

    def populate_hybe_list(self, tagged_records, dax_directories=None):
        """
        Rebuild the COMBINED checklist. tagged_records: [(hybe_record,
        modality), ...] -- every modality's parsed hybes concatenated in
        modality order, each row labeled with and UserRole-keyed by
        (folder, modality) (see hybe_checkbox_items for why the pair).
        Default-checks each row iff its OWN modality's DAX directory
        (dax_directories: {name: dir}) has real data for that folder --
        still opt-out-by-default overall (an absent/blank directory
        defaults to checked, see _has_dax_data), just evaluated per
        modality instead of against one shared directory. Presence is
        answered by _probe_dax_presence (pooled probes + session cache --
        see its own comment for the measured 9.5 s this took serially),
        so a re-parse costs zero DAX round-trips; re-confirm a DAX field
        to force a fresh probe of that modality (refresh_hybe_checks).
        """
        dax_directories = dax_directories or {}
        presence = _probe_dax_presence(
            [(dax_directories[modality], record['folder'])
             for record, modality in tagged_records if dax_directories.get(modality)])
        self.HybeListWidget.clear()
        for record, modality in tagged_records:
            folder = record['folder']
            label = f"{folder} [{modality}]  (readout_id={record['readout_id']}, datatype={record['datatype']}, " \
                   f"channels={record['channels']}, name={record['readout_name'] or '-'})"
            item = QtWidgets.QListWidgetItem(label)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            dax_directory = dax_directories.get(modality)
            has_data = presence.get((dax_directory, folder), True) if dax_directory else True
            item.setCheckState(QtCore.Qt.Checked if has_data else QtCore.Qt.Unchecked)
            item.setData(QtCore.Qt.UserRole, (folder, modality))
            self.HybeListWidget.addItem(item)

    def refresh_hybe_checks(self, modality, dax_directory):
        """Re-evaluate checked state for ONLY `modality`'s rows against a
        (possibly just-changed) DAX directory -- other modalities' rows
        are left exactly as they are, since their own DAX repositories
        did not change. Probes FRESH (refresh=True), overwriting the
        session cache: this is the one explicit invalidation for DAX
        presence, so re-confirming the DAX field is how newly-copied
        data becomes visible without restarting."""
        rows = []
        for i in range(self.HybeListWidget.count()):
            item = self.HybeListWidget.item(i)
            folder, row_modality = item.data(QtCore.Qt.UserRole)
            if row_modality != modality:
                continue
            rows.append((item, folder))
        presence = _probe_dax_presence([(dax_directory, folder) for _item, folder in rows],
                                       refresh=True) if dax_directory else {}
        for item, folder in rows:
            has_data = presence.get((dax_directory, folder), True) if dax_directory else True
            item.setCheckState(QtCore.Qt.Checked if has_data else QtCore.Qt.Unchecked)

    @staticmethod
    def _has_dax_data(dax_directory, folder):
        if not dax_directory:
            return True  # no directory to check against yet -- default to checked, not excluded
        hybe_dir = os.path.join(dax_directory, folder)
        return os.path.isdir(hybe_dir) and any(f.endswith('.dax') for f in os.listdir(hybe_dir))

    def populate_viewer_hybe_choices(self, total_active_hybe_list):
        """
        total_active_hybe_list: [(hybe_record, modality_name), ...] --
        every configured modality's active hybes at once, so MIP Viewer
        can show a hybe from ANY modality directly (there is no
        current-modality switcher left to flip first). Same
        itemData-tagged, selection-preserving pattern as
        SpotLocalizationPanel.populate_hybe_choices.
        """
        current = self.current_viewer_hybe_key()
        self.ViewerHybeComboBox.blockSignals(True)
        self.ViewerHybeComboBox.clear()
        for record, modality in total_active_hybe_list:
            self.ViewerHybeComboBox.addItem(f"{record['folder']} ({modality})", (record, modality))
        if self.ViewerHybeComboBox.count():
            restore_index = next((i for i in range(self.ViewerHybeComboBox.count())
                                  if self._viewer_hybe_item_key(i) == current), 0)
            self.ViewerHybeComboBox.setCurrentIndex(restore_index)
        self.ViewerHybeComboBox.blockSignals(False)
        self._on_viewer_hybe_changed()

    def _viewer_hybe_item_key(self, index):
        data = self.ViewerHybeComboBox.itemData(index)
        return (data[0]['folder'], data[1]) if data is not None else (None, None)

    def current_viewer_hybe_key(self):
        data = self.ViewerHybeComboBox.currentData()
        return (data[0]['folder'], data[1]) if data is not None else (None, None)

    def current_viewer_hybe(self):
        """Real hybe folder name for whatever's currently selected, or '' if nothing is."""
        data = self.ViewerHybeComboBox.currentData()
        return data[0]['folder'] if data is not None else ''

    def current_viewer_modality(self):
        """Owning modality name for whatever's currently selected, or None if nothing is."""
        data = self.ViewerHybeComboBox.currentData()
        return data[1] if data is not None else None

    def _on_viewer_hybe_changed(self):
        data = self.ViewerHybeComboBox.currentData()
        record = data[0] if data is not None else None
        # clear()+addItems() is one logical "channel list changed" update --
        # without blockSignals, clear() alone fires currentIndexChanged(-1)
        # with a momentarily-empty combo, which anything downstream
        # listening for "channel changed" (e.g. an already-open MIP
        # Viewer's live-refresh) would see as "no channel selected" and
        # misreport as an error, even though the real new selection lands
        # a moment later once addItems() runs.
        self.ViewerChannelComboBox.blockSignals(True)
        self.ViewerChannelComboBox.clear()
        if record is not None:
            self.ViewerChannelComboBox.addItems([str(c) for c in record['channels']])
        self.ViewerChannelComboBox.blockSignals(False)
        self.ViewerChannelComboBox.currentIndexChanged.emit(self.ViewerChannelComboBox.currentIndex())

    def add_job_item(self, job):
        """job: {'jobs': [{'fov_list', 'hybe_records', 'dax_directory',
        'storage_path', 'modality'}, ...], 'fov_list': list[int]} -- the
        SAME multi-job shape IngestionWorker takes (one inner dict per
        modality with checked hybes), snapshotted whole at Add time so
        Run Queued Jobs hands each queue entry straight to the worker
        with no re-derivation from live form state."""
        modalities = '+'.join(j['modality'] for j in job['jobs'])
        n_hybes = sum(len(j['hybe_records']) for j in job['jobs'])
        fov_text = ','.join(str(f) for f in job['fov_list'])
        label = f"{modalities} | FOV {fov_text} | {n_hybes} hybe(s)"
        item = QtWidgets.QListWidgetItem(label)
        item.setData(QtCore.Qt.UserRole, job)
        self.JobQueueListWidget.addItem(item)

    def queued_jobs(self):
        return [self.JobQueueListWidget.item(i).data(QtCore.Qt.UserRole) for i in range(self.JobQueueListWidget.count())]

    def remove_selected_jobs(self):
        for item in self.JobQueueListWidget.selectedItems():
            self.JobQueueListWidget.takeItem(self.JobQueueListWidget.row(item))
