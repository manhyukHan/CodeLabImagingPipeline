import os
from PyQt5 import QtWidgets, QtCore, QtGui


class IngestionPanelUI(object):
    """
    Pick an ExperimentLayout.xlsx + DAX directory + storage path + modality,
    parse the layout, select which hybes/FOVs to actually ingest (real
    datasets have far too many to always ingest in full), and run.
    """
    def setupUi(self, Widget):
        Widget.setObjectName('IngestionPanel')
        layout = QtWidgets.QVBoxLayout(Widget)

        # -- modality setup: how many modalities, what they're named, and
        # which one the rest of this tab (+ every downstream panel's own
        # mirrored ModalityComboBox) is currently pointed at. Count/names
        # default to 2 (DNA, RNA) so nothing changes for anyone who never
        # touches this group -- Set/Activate only matter for a different
        # count or custom names. --
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

        self.ModalityComboBox = QtWidgets.QComboBox()
        modalityLayout.addWidget(QtWidgets.QLabel('Current modality (switches everything below that is not global):'))
        modalityLayout.addWidget(self.ModalityComboBox)

        layout.addWidget(modalityGroup)
        self.build_modality_name_fields(2)
        self.ModalityComboBox.addItems(['DNA', 'RNA'])

        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.LayoutPathLineEdit, layoutPathRow = self._path_row('Select ExperimentLayout.xlsx', is_file=True,
                                                                 name_filter='Excel files (*.xlsx)')
        form.addRow('ExperimentLayout.xlsx:', layoutPathRow)

        self.DaxDirectoryLineEdit, daxDirRow = self._path_row('Select DAX directory', is_file=False)
        form.addRow('DAX directory:', daxDirRow)

        self.StoragePathLineEdit, storagePathRow = self._path_row('Select storage directory', is_file=False)
        form.addRow('Storage path:', storagePathRow)

        self.FovListLineEdit = QtWidgets.QLineEdit()
        self.FovListLineEdit.setPlaceholderText('e.g. 1,2,3 or 1-10,15,20-25 or 1 2 3 (comma and/or space separated)')
        form.addRow('FOV list:', self.FovListLineEdit)

        self.ParseLayoutPushButton = QtWidgets.QPushButton('Parse Layout')
        layout.addWidget(self.ParseLayoutPushButton)

        layout.addWidget(QtWidgets.QLabel('Hybes to ingest:'))
        self.HybeListWidget = QtWidgets.QListWidget()
        self.HybeListWidget.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        layout.addWidget(self.HybeListWidget)

        self.RunIngestionPushButton = QtWidgets.QPushButton('Run Ingestion')
        self.RunIngestionPushButton.setEnabled(False)
        layout.addWidget(self.RunIngestionPushButton)

        self.CheckIngestionStatusPushButton = QtWidgets.QPushButton('Check Ingestion Status')
        layout.addWidget(self.CheckIngestionStatusPushButton)
        self.IngestionStatusTextEdit = QtWidgets.QTextEdit()
        self.IngestionStatusTextEdit.setReadOnly(True)
        self.IngestionStatusTextEdit.setPlaceholderText(
            'Per-FOV/per-hybe readiness (does {storage path}/FOV##/{hybe}_stack.h5 exist with real data?) '
            'shows here after Parse Layout + Check Ingestion Status.')
        self.IngestionStatusTextEdit.setMaximumHeight(140)
        layout.addWidget(self.IngestionStatusTextEdit)

        self.ShowMemoryViewerPushButton = QtWidgets.QPushButton('Show Cell/Spot Memory Status...')
        layout.addWidget(self.ShowMemoryViewerPushButton)

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

        layout.addWidget(QtWidgets.QLabel('Log:'))
        self.LogTextEdit = QtWidgets.QTextEdit()
        self.LogTextEdit.setReadOnly(True)
        layout.addWidget(self.LogTextEdit)

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

    def lock_modality_setup(self):
        self.NumModalitiesLineEdit.setEnabled(False)
        self.SetNumModalitiesPushButton.setEnabled(False)
        for combo in self.ModalityNameComboBoxes:
            combo.setEnabled(False)
        self.ActivateModalitiesPushButton.setEnabled(False)

    def hybe_checkbox_items(self):
        """Currently-checked hybe folder names from HybeListWidget."""
        checked = []
        for i in range(self.HybeListWidget.count()):
            item = self.HybeListWidget.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                checked.append(item.data(QtCore.Qt.UserRole))
        return checked

    def populate_hybe_list(self, hybe_records, dax_directory=None):
        """
        Default-checks every hybe -- there's no reason to make the user
        manually re-check each one every time. If dax_directory is given
        (already set when Parse Layout runs), a hybe only gets checked if
        it actually has real DAX data present ({dax_directory}/{folder}/
        contains at least one .dax file) -- still opt-out-by-default
        overall, just not for hybes that couldn't possibly ingest anyway.
        """
        self.HybeListWidget.clear()
        for record in hybe_records:
            folder = record['folder']
            label = f"{folder}  (readout_id={record['readout_id']}, datatype={record['datatype']}, " \
                   f"channels={record['channels']}, name={record['readout_name'] or '-'})"
            item = QtWidgets.QListWidgetItem(label)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if self._has_dax_data(dax_directory, folder) else QtCore.Qt.Unchecked)
            item.setData(QtCore.Qt.UserRole, folder)
            self.HybeListWidget.addItem(item)

    def refresh_hybe_checks(self, dax_directory):
        """Re-evaluate checked state for the already-populated list against a (possibly just-changed) DAX directory."""
        for i in range(self.HybeListWidget.count()):
            item = self.HybeListWidget.item(i)
            item.setCheckState(QtCore.Qt.Checked if self._has_dax_data(dax_directory, item.data(QtCore.Qt.UserRole)) else QtCore.Qt.Unchecked)

    @staticmethod
    def _has_dax_data(dax_directory, folder):
        if not dax_directory:
            return True  # no directory to check against yet -- default to checked, not excluded
        hybe_dir = os.path.join(dax_directory, folder)
        return os.path.isdir(hybe_dir) and any(f.endswith('.dax') for f in os.listdir(hybe_dir))

    def populate_viewer_hybe_choices(self, hybe_records):
        self._viewer_hybe_records = list(hybe_records)
        self.ViewerHybeComboBox.clear()
        self.ViewerHybeComboBox.addItems([r['folder'] for r in hybe_records])

    def _on_viewer_hybe_changed(self):
        folder = self.ViewerHybeComboBox.currentText()
        record = next((r for r in getattr(self, '_viewer_hybe_records', []) if r['folder'] == folder), None)
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
        """job: dict with layout_path, dax_directory, storage_path, modality,
        fov_list (list[int]), selected_records (list of hybe record dicts),
        hybe_records (full parsed layout for the job's ExperimentLayout.xlsx)."""
        hybe_names = ','.join(r['folder'] for r in job['selected_records'])
        fov_text = ','.join(str(f) for f in job['fov_list'])
        label = f"{job['modality']} | {job['layout_path'].split('/')[-1]} | FOV {fov_text} | {hybe_names}"
        item = QtWidgets.QListWidgetItem(label)
        item.setData(QtCore.Qt.UserRole, job)
        self.JobQueueListWidget.addItem(item)

    def queued_jobs(self):
        return [self.JobQueueListWidget.item(i).data(QtCore.Qt.UserRole) for i in range(self.JobQueueListWidget.count())]

    def remove_selected_jobs(self):
        for item in self.JobQueueListWidget.selectedItems():
            self.JobQueueListWidget.takeItem(self.JobQueueListWidget.row(item))
