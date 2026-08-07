from PyQt5 import QtWidgets, QtCore


class CelltypeDeterminationPanelUI(object):
    """
    Two celltype-determination modes -- FOV (a static FOV-number -> celltype
    lookup table) and Barcode (per-cell/per-spot classification against a
    calibrated barcode-channel readout, ported from CellClassifier's
    classify_cell_barcode -- see codelab_pipeline/models/celltype.py).
    Barcode is the default mode (per explicit request), differing from
    CellClassifier's own XML-driven default of FOV.

    Celltype identity is a single shared, ordered list (CelltypeNamesListWidget)
    used by both modes -- FOV mode attaches an FOV-range string to the
    selected celltype, Barcode mode attaches a (hybe, channel) barcode
    readout + calibration to it. This sidesteps needing two independently-
    ordered lists that have to be kept in sync by position (which is how
    CellClassifier's own barcode_channel/barcode_celltype pairing works,
    and is easy to get silently out of sync) -- here, "which channel
    belongs to which celltype" is always an explicit per-celltype
    assignment, not implicit list position.

    This panel only exposes widget state -- it has no storage_path and
    can't read real images, so it never computes an actual quantile/scale
    itself. windows/main_window.py reads the configured inputs via
    get_barcode_calibration_inputs() and does the real per-FOV image I/O
    and calibration-value computation, same separation used everywhere
    else in this app (panels are dumb, the controller does I/O).
    """
    def setupUi(self, Widget):
        Widget.setObjectName('CelltypeDeterminationPanel')
        layout = QtWidgets.QVBoxLayout(Widget)

        self.ModalityComboBox = QtWidgets.QComboBox()
        self.ModalityComboBox.addItems(['DNA', 'RNA'])
        modalityRow = QtWidgets.QWidget()
        modalityRowLayout = QtWidgets.QHBoxLayout(modalityRow)
        modalityRowLayout.setContentsMargins(0, 0, 0, 0)
        modalityRowLayout.addWidget(QtWidgets.QLabel('Modality:'))
        modalityRowLayout.addWidget(self.ModalityComboBox)
        layout.addWidget(modalityRow)

        celltypeGroup = QtWidgets.QGroupBox('Celltypes (shared identity for both modes)')
        celltypeLayout = QtWidgets.QVBoxLayout(celltypeGroup)
        self.CelltypeNamesListWidget = QtWidgets.QListWidget()
        self.CelltypeNamesListWidget.setMaximumHeight(100)
        celltypeLayout.addWidget(self.CelltypeNamesListWidget)
        addRow = QtWidgets.QWidget()
        addLayout = QtWidgets.QHBoxLayout(addRow)
        addLayout.setContentsMargins(0, 0, 0, 0)
        self.NewCelltypeNameLineEdit = QtWidgets.QLineEdit()
        self.NewCelltypeNameLineEdit.setPlaceholderText('new celltype name')
        self.AddCelltypeNamePushButton = QtWidgets.QPushButton('Add')
        self.RemoveCelltypeNamePushButton = QtWidgets.QPushButton('Remove Selected')
        addLayout.addWidget(self.NewCelltypeNameLineEdit)
        addLayout.addWidget(self.AddCelltypeNamePushButton)
        addLayout.addWidget(self.RemoveCelltypeNamePushButton)
        celltypeLayout.addWidget(addRow)
        layout.addWidget(celltypeGroup)
        self.AddCelltypeNamePushButton.clicked.connect(self._add_celltype_name)
        self.RemoveCelltypeNamePushButton.clicked.connect(self._remove_celltype_name)

        modeGroup = QtWidgets.QGroupBox('Mode')
        modeLayout = QtWidgets.QHBoxLayout(modeGroup)
        self.FOVModeRadioButton = QtWidgets.QRadioButton('FOV')
        self.BarcodeModeRadioButton = QtWidgets.QRadioButton('Barcode')
        self.BarcodeModeRadioButton.setChecked(True)
        self.ModeButtonGroup = QtWidgets.QButtonGroup(Widget)
        self.ModeButtonGroup.addButton(self.FOVModeRadioButton)
        self.ModeButtonGroup.addButton(self.BarcodeModeRadioButton)
        modeLayout.addWidget(self.FOVModeRadioButton)
        modeLayout.addWidget(self.BarcodeModeRadioButton)
        layout.addWidget(modeGroup)

        self.ModeStackedWidget = QtWidgets.QStackedWidget()
        layout.addWidget(self.ModeStackedWidget)
        self.ModeStackedWidget.addWidget(self._build_fov_page())
        self.ModeStackedWidget.addWidget(self._build_barcode_page())
        self.ModeStackedWidget.setCurrentIndex(1)  # Barcode is default
        self.FOVModeRadioButton.toggled.connect(lambda on: self.ModeStackedWidget.setCurrentIndex(0) if on else None)
        self.BarcodeModeRadioButton.toggled.connect(lambda on: self.ModeStackedWidget.setCurrentIndex(1) if on else None)

        self.IncludeTransientCheckBox = QtWidgets.QCheckBox('Also classify unsaved (transient) cells, not just saved/permanent')
        self.IncludeTransientCheckBox.setChecked(True)
        layout.addWidget(self.IncludeTransientCheckBox)

        self.RunCelltypeDeterminationPushButton = QtWidgets.QPushButton('Run Celltype Determination')
        layout.addWidget(self.RunCelltypeDeterminationPushButton)

        self.ShowCelltypeResultPushButton = QtWidgets.QPushButton('Show Celltype Result (Reference vs Celltype)')
        layout.addWidget(self.ShowCelltypeResultPushButton)

        layout.addWidget(QtWidgets.QLabel('Log:'))
        self.LogTextEdit = QtWidgets.QTextEdit()
        self.LogTextEdit.setReadOnly(True)
        layout.addWidget(self.LogTextEdit)

        self._hybe_records = []

    def _build_fov_page(self):
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        self.FovRangesLineEdit = QtWidgets.QLineEdit()
        self.FovRangesLineEdit.setPlaceholderText('e.g. 1-46, 50-93 (applies to the selected celltype above)')
        form.addRow('FOV ranges:', self.FovRangesLineEdit)
        self.SetFovRangesPushButton = QtWidgets.QPushButton('Set FOV Ranges for Selected Celltype')
        form.addRow(self.SetFovRangesPushButton)
        self.FovRangesSummaryTextEdit = QtWidgets.QTextEdit()
        self.FovRangesSummaryTextEdit.setReadOnly(True)
        self.FovRangesSummaryTextEdit.setMaximumHeight(80)
        form.addRow('Current mapping:', self.FovRangesSummaryTextEdit)
        return page

    def _build_barcode_page(self):
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)

        self.BarcodeHybeComboBox = QtWidgets.QComboBox()
        form.addRow('Barcode hybe:', self.BarcodeHybeComboBox)
        self.BarcodeChannelComboBox = QtWidgets.QComboBox()
        form.addRow('Barcode channel:', self.BarcodeChannelComboBox)
        self.AssignBarcodeChannelPushButton = QtWidgets.QPushButton('Assign to Selected Celltype')
        form.addRow(self.AssignBarcodeChannelPushButton)

        self.BarcodeMethodComboBox = QtWidgets.QComboBox()
        self.BarcodeMethodComboBox.addItems(['Vote (200-sample)', 'Median'])
        form.addRow('Classification method:', self.BarcodeMethodComboBox)

        self.CalibrationFovLineEdit = QtWidgets.QLineEdit()
        self.CalibrationFovLineEdit.setPlaceholderText('blank = all ingested FOVs, or space-separated e.g. 1 2 3')
        form.addRow('Calibrate for FOV(s):', self.CalibrationFovLineEdit)

        self.ScaleDoubleSpinBox = QtWidgets.QDoubleSpinBox()
        self.ScaleDoubleSpinBox.setRange(0, 1e6)
        self.ScaleDoubleSpinBox.setValue(1.0)
        self.ScaleDoubleSpinBox.setDecimals(4)
        form.addRow('Scale:', self.ScaleDoubleSpinBox)

        self.LowerBoundModeComboBox = QtWidgets.QComboBox()
        self.LowerBoundModeComboBox.addItems(['Quantile', 'Absolute'])
        self.LowerBoundValueDoubleSpinBox = QtWidgets.QDoubleSpinBox()
        self.LowerBoundValueDoubleSpinBox.setRange(0, 1e9)
        self.LowerBoundValueDoubleSpinBox.setDecimals(4)
        self.LowerBoundValueDoubleSpinBox.setSingleStep(0.001)
        self.LowerBoundValueDoubleSpinBox.setValue(0.3)
        lowerRow = QtWidgets.QWidget()
        lowerLayout = QtWidgets.QHBoxLayout(lowerRow)
        lowerLayout.setContentsMargins(0, 0, 0, 0)
        lowerLayout.addWidget(self.LowerBoundModeComboBox)
        lowerLayout.addWidget(self.LowerBoundValueDoubleSpinBox)
        form.addRow('Lower bound:', lowerRow)

        self.UpperBoundModeComboBox = QtWidgets.QComboBox()
        self.UpperBoundModeComboBox.addItems(['Quantile', 'Absolute'])
        self.UpperBoundValueDoubleSpinBox = QtWidgets.QDoubleSpinBox()
        self.UpperBoundValueDoubleSpinBox.setRange(0, 1e9)
        self.UpperBoundValueDoubleSpinBox.setDecimals(4)
        self.UpperBoundValueDoubleSpinBox.setSingleStep(0.001)
        self.UpperBoundValueDoubleSpinBox.setValue(0.999)
        upperRow = QtWidgets.QWidget()
        upperLayout = QtWidgets.QHBoxLayout(upperRow)
        upperLayout.setContentsMargins(0, 0, 0, 0)
        upperLayout.addWidget(self.UpperBoundModeComboBox)
        upperLayout.addWidget(self.UpperBoundValueDoubleSpinBox)
        form.addRow('Upper bound:', upperRow)

        self.ApplyCalibrationPushButton = QtWidgets.QPushButton('Apply Calibration to Selected Celltype\'s Channel')
        form.addRow(self.ApplyCalibrationPushButton)

        self.CalibrationSummaryTextEdit = QtWidgets.QTextEdit()
        self.CalibrationSummaryTextEdit.setReadOnly(True)
        self.CalibrationSummaryTextEdit.setMaximumHeight(100)
        form.addRow('Current calibration:', self.CalibrationSummaryTextEdit)

        self.OverviewFovLineEdit = QtWidgets.QLineEdit()
        self.OverviewFovLineEdit.setPlaceholderText('FOV to preview, e.g. 1')
        form.addRow('Overview FOV:', self.OverviewFovLineEdit)
        self.ShowBarcodeOverviewPushButton = QtWidgets.QPushButton('Show Barcode Channel Overview')
        form.addRow(self.ShowBarcodeOverviewPushButton)
        return page

    def populate_hybe_choices(self, hybe_records):
        """
        Restricted to datatype == 'B' (ExperimentLayout's own barcode-
        readout classification, e.g. Hyb_130/Hyb_131 in the DNA layout)
        -- a barcode channel is specifically a barcode-type readout, not
        just any active hybe. hybe_records is already scoped to the
        currently active modality (see MainWindow._refresh_active_hybe_
        lists); switching the (mirrored) ModalityComboBox to whichever
        modality actually has 'B' hybes is how those become available
        here -- this panel doesn't re-derive modality on its own.
        """
        self._hybe_records = [r for r in hybe_records if r.get('datatype') == 'B']
        self.BarcodeHybeComboBox.clear()
        self.BarcodeHybeComboBox.addItems([r['folder'] for r in self._hybe_records])
        self.BarcodeHybeComboBox.currentIndexChanged.connect(self._on_barcode_hybe_changed)
        self._on_barcode_hybe_changed()

    def _on_barcode_hybe_changed(self):
        folder = self.BarcodeHybeComboBox.currentText()
        record = next((r for r in self._hybe_records if r['folder'] == folder), None)
        # blockSignals so clear()+addItems() reads as one atomic "channel
        # list changed" update, not a transient empty-then-refilled state
        self.BarcodeChannelComboBox.blockSignals(True)
        self.BarcodeChannelComboBox.clear()
        if record is not None:
            self.BarcodeChannelComboBox.addItems([str(c) for c in record['channels']])
            # default to the readout channel, not the fiducial one --
            # same convention as ui/spot_localization_panel.py's own
            # _on_hybe_changed (a barcode channel is a signal channel,
            # fiducial is only ever used for alignment/segmentation).
            fiducial = record.get('fiducial_channel')
            readout_channels = [c for c in record['channels'] if c != fiducial]
            if readout_channels:
                idx = self.BarcodeChannelComboBox.findText(str(readout_channels[0]))
                if idx >= 0:
                    self.BarcodeChannelComboBox.setCurrentIndex(idx)
        self.BarcodeChannelComboBox.blockSignals(False)
        self.BarcodeChannelComboBox.currentIndexChanged.emit(self.BarcodeChannelComboBox.currentIndex())

    def selected_celltype(self):
        item = self.CelltypeNamesListWidget.currentItem()
        return item.text() if item is not None else None

    def celltype_names(self):
        return [self.CelltypeNamesListWidget.item(i).text() for i in range(self.CelltypeNamesListWidget.count())]

    def mode(self):
        return 'fov' if self.FOVModeRadioButton.isChecked() else 'barcode'

    def barcode_method(self):
        return 'median' if self.BarcodeMethodComboBox.currentText() == 'Median' else 'vote'

    def get_barcode_calibration_inputs(self):
        """Everything needed to compute+store one (hybe,channel)'s calibration -- main_window does the actual image I/O."""
        return {
            'hybe': self.BarcodeHybeComboBox.currentText(),
            'channel': int(self.BarcodeChannelComboBox.currentText()) if self.BarcodeChannelComboBox.currentText() else None,
            'fov_scope_text': self.CalibrationFovLineEdit.text().strip(),
            'scale': self.ScaleDoubleSpinBox.value(),
            'lower_is_quantile': self.LowerBoundModeComboBox.currentText() == 'Quantile',
            'lower_value': self.LowerBoundValueDoubleSpinBox.value(),
            'upper_is_quantile': self.UpperBoundModeComboBox.currentText() == 'Quantile',
            'upper_value': self.UpperBoundValueDoubleSpinBox.value(),
        }

    def ensure_celltype_names(self, names):
        """
        Adds any of `names` not already present as items -- used to seed
        the shared celltype identity list from a loaded config's default
        names and/or real classified celltypes already found in vlinks.h5
        (see MainWindow._refresh_celltype_names_from_vlinks), without
        clobbering anything the user already typed in this session
        (existing items are never removed or reordered). If nothing is
        currently selected, selects the first item afterward so the list
        is immediately usable (e.g. Set FOV Ranges) without requiring a
        manual click first -- per explicit request, the listview should
        already be "active" the moment real data justifies it.
        """
        existing = set(self.celltype_names())
        for name in names:
            if name and name not in existing:
                self.CelltypeNamesListWidget.addItem(name)
                existing.add(name)
        if self.CelltypeNamesListWidget.currentItem() is None and self.CelltypeNamesListWidget.count() > 0:
            self.CelltypeNamesListWidget.setCurrentRow(0)

    def _add_celltype_name(self):
        name = self.NewCelltypeNameLineEdit.text().strip()
        if not name or name in self.celltype_names():
            return
        self.CelltypeNamesListWidget.addItem(name)
        self.NewCelltypeNameLineEdit.clear()

    def _remove_celltype_name(self):
        item = self.CelltypeNamesListWidget.currentItem()
        if item is None:
            return
        self.CelltypeNamesListWidget.takeItem(self.CelltypeNamesListWidget.row(item))
