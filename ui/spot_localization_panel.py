from PyQt5 import QtWidgets, QtCore


class SpotLocalizationPanelUI(object):
    """
    Interactive spot localization: pick a cell (from the transient
    CellContainer that Cell Segmentation already manages) + hybe/channel,
    then either auto-detect (peak_local_max) or manually click spots on
    the pop-up crop displayer (canvas/spot_crop_displayer.py). Spots are
    appended directly into cell.spots the moment they're detected/clicked
    -- ACell already owns its spots as real, persistent objects (unlike
    CellClassifier's lazy per-call views, which is why that project needs
    a separate SpotContainer + a later link step and this one doesn't), so
    there's no separate staging container or "commit to cells" step here.
    Save/Discard/Send for spots is the SAME transient/permanent promotion
    the Cell Segmentation tab already provides (spots ride along inside
    cell.spots automatically) -- deliberately not duplicated here.

    Scope has two modes, both real (neither is a UI-only stand-in for the
    other):
    - Current Cell (default): peak_local_max over just the selected cell's
      own padded crop, threshold relative to that crop's own max.
    - Whole FOV: peak_local_max over the full FOV MIP, threshold relative
      to the FOV's own max -- a genuine global search+threshold, not
      "whatever happens to be on screen." Detected peaks are assigned to
      whichever cell's mask they land in (direct label lookup); peaks
      outside every cell are discarded (this app has no "homeless spot"
      storage the way CellClassifier's array-backed SpotContainer does).

    Manual click mode is always single-cell (never whole-FOV) -- a
    manually-placed point needs an unambiguous owning cell to attach to,
    and the crop displayer only ever shows one cell's own crop.
    """
    def setupUi(self, Widget):
        Widget.setObjectName('SpotLocalizationPanel')
        layout = QtWidgets.QVBoxLayout(Widget)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.ModalityComboBox = QtWidgets.QComboBox()
        self.ModalityComboBox.addItems(['DNA', 'RNA'])
        form.addRow('Modality:', self.ModalityComboBox)

        self.CellListWidget = QtWidgets.QListWidget()
        self.CellListWidget.setMaximumHeight(120)
        form.addRow('Cell (transient, this FOV):', self.CellListWidget)

        self.RefreshCellListPushButton = QtWidgets.QPushButton('Refresh Cell List')
        layout.addWidget(self.RefreshCellListPushButton)

        self.HybeComboBox = QtWidgets.QComboBox()
        form.addRow('Hybe:', self.HybeComboBox)

        self.ChannelComboBox = QtWidgets.QComboBox()
        form.addRow('Channel:', self.ChannelComboBox)

        self.ScopeComboBox = QtWidgets.QComboBox()
        self.ScopeComboBox.addItems(['Current Cell', 'Whole FOV'])
        self.ScopeComboBox.setCurrentText('Whole FOV')
        form.addRow('Scope:', self.ScopeComboBox)

        self.ThresholdPercentSpinBox = QtWidgets.QSpinBox()
        self.ThresholdPercentSpinBox.setRange(0, 100)
        self.ThresholdPercentSpinBox.setValue(50)
        self.ThresholdPercentSpinBox.setSuffix(' %')
        form.addRow('Threshold (% of scope max):', self.ThresholdPercentSpinBox)

        self.MinDistanceSpinBox = QtWidgets.QSpinBox()
        self.MinDistanceSpinBox.setRange(1, 100)
        self.MinDistanceSpinBox.setValue(3)
        form.addRow('Min distance between spots (px):', self.MinDistanceSpinBox)

        self.PadSpinBox = QtWidgets.QSpinBox()
        self.PadSpinBox.setRange(0, 100)
        self.PadSpinBox.setValue(10)
        form.addRow('Cell crop padding (px, Current Cell scope only):', self.PadSpinBox)

        self.AutoDetectPushButton = QtWidgets.QPushButton('Run Auto-Detect')
        layout.addWidget(self.AutoDetectPushButton)

        self.ShowDisplayerPushButton = QtWidgets.QPushButton('Show Spot Crop Displayer')
        self.ShowDisplayerPushButton.setCheckable(True)
        layout.addWidget(self.ShowDisplayerPushButton)

        infoLabel = QtWidgets.QLabel(
            'Spots are attached to cells immediately. Use the Cell Segmentation '
            'tab\'s Save/Discard/Send buttons to promote or discard them -- '
            'saving cells saves their spots too.')
        infoLabel.setWordWrap(True)
        layout.addWidget(infoLabel)

        self.ProgressBar = QtWidgets.QProgressBar()
        layout.addWidget(self.ProgressBar)

        layout.addWidget(QtWidgets.QLabel('Log:'))
        self.LogTextEdit = QtWidgets.QTextEdit()
        self.LogTextEdit.setReadOnly(True)
        layout.addWidget(self.LogTextEdit)

        self._hybe_records = []
        self.HybeComboBox.currentIndexChanged.connect(self._on_hybe_changed)

    def populate_hybe_choices(self, hybe_records):
        self._hybe_records = list(hybe_records)
        self.HybeComboBox.clear()
        self.HybeComboBox.addItems([r['folder'] for r in hybe_records])

    def _on_hybe_changed(self):
        folder = self.HybeComboBox.currentText()
        record = next((r for r in self._hybe_records if r['folder'] == folder), None)
        # blockSignals so clear()+addItems() reads as one atomic "channel
        # list changed" update, not a transient empty-then-refilled state
        self.ChannelComboBox.blockSignals(True)
        self.ChannelComboBox.clear()
        if record is not None:
            self.ChannelComboBox.addItems([str(c) for c in record['channels']])
            # default to the readout channel, not the fiducial one -- spots
            # live in the actual signal channel, fiducial is only ever used
            # for alignment/segmentation elsewhere in this app
            fiducial = record.get('fiducial_channel')
            readout_channels = [c for c in record['channels'] if c != fiducial]
            if readout_channels:
                idx = self.ChannelComboBox.findText(str(readout_channels[0]))
                if idx >= 0:
                    self.ChannelComboBox.setCurrentIndex(idx)
        self.ChannelComboBox.blockSignals(False)
        self.ChannelComboBox.currentIndexChanged.emit(self.ChannelComboBox.currentIndex())

    def populate_cell_choices(self, cells):
        """cells: list of ACell -- shows id + current total_num_spots, selection preserved by cell id where possible."""
        previous_id = self.selected_cell_id()
        self.CellListWidget.clear()
        for cell in cells:
            item = QtWidgets.QListWidgetItem(f'Cell {cell.id}: {cell.total_num_spots} spot(s)')
            item.setData(QtCore.Qt.UserRole, cell.id)
            self.CellListWidget.addItem(item)
            if cell.id == previous_id:
                self.CellListWidget.setCurrentItem(item)

    def selected_cell_id(self):
        item = self.CellListWidget.currentItem()
        return item.data(QtCore.Qt.UserRole) if item is not None else None

    def scope(self):
        return 'current_cell' if self.ScopeComboBox.currentText() == 'Current Cell' else 'whole_fov'
