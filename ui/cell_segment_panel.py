from PyQt5 import QtWidgets, QtCore


class CellSegmentPanelUI(object):
    """
    Cell segmentation: pick a reference hybe/channel + a method (Cellpose /
    Classical threshold+watershed / Manual polygon draw, each with its own
    parameter page), run in bulk (off the main thread, except Manual which
    has nothing to compute), review/hand-correct the mask in the pop-up
    displayer (canvas/cell_displayer.py -- not embedded here, see that
    module's docstring for why), then stage/save/discard through a
    transient/permanent CellContainer pair -- mirrors CellClassifier's own
    proven cell-segmentation temp/approve pattern. All three methods funnel
    into the same mask -> CellContainer.load_new_cells path; see
    cell_displayer.py's docstring for how Manual-mode drawing composes
    additively with whatever a Cellpose/Classical run already produced.
    """
    def setupUi(self, Widget):
        Widget.setObjectName('CellSegmentPanel')
        layout = QtWidgets.QVBoxLayout(Widget)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.ModalityComboBox = QtWidgets.QComboBox()
        self.ModalityComboBox.addItems(['DNA', 'RNA'])
        form.addRow('Modality:', self.ModalityComboBox)

        self.FovSpinBox = QtWidgets.QSpinBox()
        self.FovSpinBox.setRange(1, 100000)
        self.FovSpinBox.setValue(1)
        form.addRow('FOV:', self.FovSpinBox)

        self.ReferenceHybeComboBox = QtWidgets.QComboBox()
        form.addRow('Reference hybe:', self.ReferenceHybeComboBox)

        self.ChannelComboBox = QtWidgets.QComboBox()
        form.addRow('Channel:', self.ChannelComboBox)

        self.MethodComboBox = QtWidgets.QComboBox()
        self.MethodComboBox.addItems(['Cellpose', 'Classical', 'Manual'])
        form.addRow('Method:', self.MethodComboBox)

        self.MethodStackedWidget = QtWidgets.QStackedWidget()
        layout.addWidget(self.MethodStackedWidget)
        self.MethodStackedWidget.addWidget(self._build_cellpose_page())
        self.MethodStackedWidget.addWidget(self._build_classical_page())
        self.MethodStackedWidget.addWidget(self._build_manual_page())
        self.MethodComboBox.currentIndexChanged.connect(self.MethodStackedWidget.setCurrentIndex)

        self.AppendModeCheckBox = QtWidgets.QCheckBox('Append Mode')
        layout.addWidget(self.AppendModeCheckBox)
        appendModeLabel = QtWidgets.QLabel(
            'Add to the current mask instead of replacing it -- e.g. segment one hybe, then a '
            'different cell-barcode hybe, and accumulate cells from both.')
        appendModeLabel.setWordWrap(True)
        layout.addWidget(appendModeLabel)

        self.RunSegmentationPushButton = QtWidgets.QPushButton('Run Segmentation')
        layout.addWidget(self.RunSegmentationPushButton)

        self.ShowDisplayerPushButton = QtWidgets.QPushButton('Show Cell Displayer')
        self.ShowDisplayerPushButton.setCheckable(True)
        layout.addWidget(self.ShowDisplayerPushButton)

        transientGroup = QtWidgets.QGroupBox('Transient Cell Container (this FOV, staged)')
        transientLayout = QtWidgets.QVBoxLayout(transientGroup)
        buttonsRow = QtWidgets.QWidget()
        buttonsLayout = QtWidgets.QHBoxLayout(buttonsRow)
        buttonsLayout.setContentsMargins(0, 0, 0, 0)
        self.SaveCellsPushButton = QtWidgets.QPushButton('Save')
        self.DiscardCellsPushButton = QtWidgets.QPushButton('Discard')
        self.SendPermanentPushButton = QtWidgets.QPushButton('Send Permanent to Transient')
        buttonsLayout.addWidget(self.SaveCellsPushButton)
        buttonsLayout.addWidget(self.DiscardCellsPushButton)
        buttonsLayout.addWidget(self.SendPermanentPushButton)
        transientLayout.addWidget(buttonsRow)

        transientInfoLabel = QtWidgets.QLabel(
            'Saved cells are written to vlinks.h5 automatically (both RNA and DNA storage paths, '
            'if the Alignment tab\'s Cross-Modal fields are set) and activate automatically when you '
            'return to this FOV -- no separate file save/load needed.')
        transientInfoLabel.setWordWrap(True)
        transientLayout.addWidget(transientInfoLabel)
        layout.addWidget(transientGroup)

        self.ProgressBar = QtWidgets.QProgressBar()
        layout.addWidget(self.ProgressBar)

        layout.addWidget(QtWidgets.QLabel('Log:'))
        self.LogTextEdit = QtWidgets.QTextEdit()
        self.LogTextEdit.setReadOnly(True)
        layout.addWidget(self.LogTextEdit)

        self._hybe_records = []
        self.ReferenceHybeComboBox.currentIndexChanged.connect(self._on_reference_hybe_changed)

    def _build_cellpose_page(self):
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        self.DiameterSpinBox = QtWidgets.QSpinBox()
        self.DiameterSpinBox.setRange(1, 500)
        self.DiameterSpinBox.setValue(40)
        form.addRow('Diameter (px):', self.DiameterSpinBox)
        self.MinSizeSpinBox = QtWidgets.QSpinBox()
        self.MinSizeSpinBox.setRange(0, 1000000)
        self.MinSizeSpinBox.setValue(1000)
        form.addRow('Min size (px):', self.MinSizeSpinBox)
        self.MaxSizeSpinBox = QtWidgets.QSpinBox()
        self.MaxSizeSpinBox.setRange(1, 1000000)
        self.MaxSizeSpinBox.setValue(10000)
        form.addRow('Max size (px):', self.MaxSizeSpinBox)
        return page

    def _build_classical_page(self):
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        self.ClassicalAlgorithmComboBox = QtWidgets.QComboBox()
        self.ClassicalAlgorithmComboBox.addItems(['Otsu', 'Yen', 'Li', 'Triangle', 'Absolute'])
        form.addRow('Algorithm:', self.ClassicalAlgorithmComboBox)
        self.ClassicalAbsoluteCutoffSpinBox = QtWidgets.QDoubleSpinBox()
        self.ClassicalAbsoluteCutoffSpinBox.setRange(0, 1e9)
        self.ClassicalAbsoluteCutoffSpinBox.setValue(1000)
        self.ClassicalAbsoluteCutoffSpinBox.setEnabled(False)
        form.addRow('Absolute cutoff:', self.ClassicalAbsoluteCutoffSpinBox)
        self.ClassicalAlgorithmComboBox.currentTextChanged.connect(
            lambda text: self.ClassicalAbsoluteCutoffSpinBox.setEnabled(text == 'Absolute'))
        self.ClassicalMinDistanceSpinBox = QtWidgets.QSpinBox()
        self.ClassicalMinDistanceSpinBox.setRange(1, 1000)
        self.ClassicalMinDistanceSpinBox.setValue(7)
        form.addRow('Min distance between cells (px):', self.ClassicalMinDistanceSpinBox)
        self.ClassicalMinSizeSpinBox = QtWidgets.QSpinBox()
        self.ClassicalMinSizeSpinBox.setRange(0, 1000000)
        self.ClassicalMinSizeSpinBox.setValue(500)
        form.addRow('Min size (px):', self.ClassicalMinSizeSpinBox)
        self.ClassicalMaxSizeSpinBox = QtWidgets.QSpinBox()
        self.ClassicalMaxSizeSpinBox.setRange(1, 1000000)
        self.ClassicalMaxSizeSpinBox.setValue(10000)
        form.addRow('Max size (px):', self.ClassicalMaxSizeSpinBox)
        return page

    def _build_manual_page(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        label = QtWidgets.QLabel(
            'Opens the displayer with an empty mask, Manual Add Mode pre-enabled.\n'
            'In the displayer: left-click to place polygon vertices, "A" to commit\n'
            'the current polygon as a new cell, "D" to undo the last vertex.')
        label.setWordWrap(True)
        layout.addWidget(label)
        layout.addStretch()
        return page

    def populate_reference_hybe_choices(self, hybe_records):
        self._hybe_records = list(hybe_records)
        self.ReferenceHybeComboBox.clear()
        self.ReferenceHybeComboBox.addItems([r['folder'] for r in hybe_records])

    def current_method(self):
        """'cellpose' / 'classical' / 'manual' -- lowercase, matches segment.py's own method kwarg casing."""
        return self.MethodComboBox.currentText().lower()

    def _on_reference_hybe_changed(self):
        folder = self.ReferenceHybeComboBox.currentText()
        record = next((r for r in self._hybe_records if r['folder'] == folder), None)
        # blockSignals so clear()+addItems() reads as one atomic "channel
        # list changed" update, not a transient empty-then-refilled state
        # any downstream listener could observe mid-update
        self.ChannelComboBox.blockSignals(True)
        self.ChannelComboBox.clear()
        if record is not None:
            self.ChannelComboBox.addItems([str(c) for c in record['channels']])
            # default to the readout channel, not fiducial -- fiducial is
            # only ever the default for FOV/within-experiment alignment's
            # own computation, everywhere else (segmentation included)
            # readout is the default per explicit principle
            fiducial = record.get('fiducial_channel')
            readout_channels = [c for c in record['channels'] if c != fiducial]
            if readout_channels:
                idx = self.ChannelComboBox.findText(str(readout_channels[0]))
                if idx >= 0:
                    self.ChannelComboBox.setCurrentIndex(idx)
        self.ChannelComboBox.blockSignals(False)
        self.ChannelComboBox.currentIndexChanged.emit(self.ChannelComboBox.currentIndex())
