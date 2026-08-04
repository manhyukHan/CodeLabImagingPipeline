from PyQt5 import QtWidgets


class AlignmentPanelUI(object):
    """
    The full 4-layer alignment chain: FOV (within-experiment), experiment
    (cross-modal), cell-based, and spot-based (scaffolded here, disabled
    until the spot-localization panel exists to unlock it). One panel-wide
    manual/automatic mode applies uniformly to all four: automatic computes
    and immediately persists; manual computes into a staged/pending result
    that only persists on explicit Accept (Reject discards it) -- each
    section gets its own Accept/Reject pair, enabled only when it actually
    has a pending manual-mode result.

    Every preview -- FOV, cross-modal, cell, interactive or auto-saved --
    is a pop-up (canvas/alignment_preview_window.py), never embedded here.
    An earlier version embedded the FOV/cross-modal preview directly in
    this panel; a real screenshot showed the controls column squeezing it
    into an unreadably stretched aspect ratio, so nothing gets embedded now.
    """
    def setupUi(self, Widget):
        Widget.setObjectName('AlignmentPanel')
        controls_layout = QtWidgets.QVBoxLayout(Widget)

        # -- mode: applies uniformly to all four layers below --
        modeGroup = QtWidgets.QGroupBox('Mode')
        modeLayout = QtWidgets.QHBoxLayout(modeGroup)
        self.ModeAutomaticRadioButton = QtWidgets.QRadioButton('Automatic (compute + save)')
        self.ModeManualRadioButton = QtWidgets.QRadioButton('Manual (review before saving)')
        self.ModeAutomaticRadioButton.setChecked(True)
        self.ModeButtonGroup = QtWidgets.QButtonGroup(Widget)
        self.ModeButtonGroup.addButton(self.ModeAutomaticRadioButton)
        self.ModeButtonGroup.addButton(self.ModeManualRadioButton)
        modeLayout.addWidget(self.ModeAutomaticRadioButton)
        modeLayout.addWidget(self.ModeManualRadioButton)
        controls_layout.addWidget(modeGroup)

        # -- within-experiment (FOV) --
        sameModalityGroup = QtWidgets.QGroupBox('1. Same-Modality (FOV) Alignment')
        sameModalityLayout = QtWidgets.QFormLayout(sameModalityGroup)
        self.ModalityComboBox = QtWidgets.QComboBox()
        self.ModalityComboBox.addItems(['DNA', 'RNA'])
        sameModalityLayout.addRow('Modality:', self.ModalityComboBox)
        self.ReferenceHybeComboBox = QtWidgets.QComboBox()
        sameModalityLayout.addRow('Reference hybe:', self.ReferenceHybeComboBox)
        self.SameModalityBorderTrimSpinBox = QtWidgets.QSpinBox()
        self.SameModalityBorderTrimSpinBox.setRange(0, 500)
        self.SameModalityBorderTrimSpinBox.setValue(0)
        self.SameModalityBorderTrimSpinBox.setSuffix(' px')
        sameModalityLayout.addRow('Border trim:', self.SameModalityBorderTrimSpinBox)
        self.SameModalityMaxShiftSpinBox = QtWidgets.QSpinBox()
        self.SameModalityMaxShiftSpinBox.setRange(0, 100000)
        self.SameModalityMaxShiftSpinBox.setValue(0)
        self.SameModalityMaxShiftSpinBox.setSpecialValueText('unbounded')
        self.SameModalityMaxShiftSpinBox.setSuffix(' px')
        sameModalityLayout.addRow('Max shift (0 = unbounded):', self.SameModalityMaxShiftSpinBox)
        self.RunFovAlignmentPushButton = QtWidgets.QPushButton('Run FOV Alignment')
        sameModalityLayout.addRow(self.RunFovAlignmentPushButton)
        self.SameModalityResultsListWidget = QtWidgets.QListWidget()
        sameModalityLayout.addRow('Results:', self.SameModalityResultsListWidget)
        self.SameModalityOverlayFovLineEdit = QtWidgets.QLineEdit()
        self.SameModalityOverlayFovLineEdit.setPlaceholderText('blank = FOV just aligned above, or type any FOV, e.g. 1')
        sameModalityLayout.addRow('Overlay FOV:', self.SameModalityOverlayFovLineEdit)
        self.SameModalityChannelTypeComboBox = QtWidgets.QComboBox()
        self.SameModalityChannelTypeComboBox.addItems(['readout', 'fiducial'])
        sameModalityLayout.addRow('Overlay channel:', self.SameModalityChannelTypeComboBox)
        self.SameModalityShowOverlayPushButton = QtWidgets.QPushButton('Show All-Readouts Overlay')
        sameModalityLayout.addRow(self.SameModalityShowOverlayPushButton)
        self.SameModalityAcceptPushButton, self.SameModalityRejectPushButton, sameModalityAcceptRow = self._accept_reject_row()
        sameModalityLayout.addRow(sameModalityAcceptRow)
        controls_layout.addWidget(sameModalityGroup)

        # -- cross-experiment --
        crossGroup = QtWidgets.QGroupBox('2. Cross-Modality Alignment')
        crossLayout = QtWidgets.QFormLayout(crossGroup)

        self.RnaStoragePathLineEdit, rnaRow = self._path_row('Select RNA storage directory')
        crossLayout.addRow('RNA storage path:', rnaRow)
        self.RnaReferenceHybeComboBox = QtWidgets.QComboBox()
        self.RnaReferenceHybeComboBox.setEditable(True)
        self.RnaReferenceHybeComboBox.lineEdit().setPlaceholderText('e.g. Hyb_500')
        crossLayout.addRow('RNA reference hybe:', self.RnaReferenceHybeComboBox)

        self.DnaStoragePathLineEdit, dnaRow = self._path_row('Select DNA storage directory')
        crossLayout.addRow('DNA storage path:', dnaRow)
        self.DnaReferenceHybeComboBox = QtWidgets.QComboBox()
        self.DnaReferenceHybeComboBox.setEditable(True)
        self.DnaReferenceHybeComboBox.lineEdit().setPlaceholderText('e.g. Hyb_400')
        crossLayout.addRow('DNA reference hybe:', self.DnaReferenceHybeComboBox)

        self.ChannelTypeComboBox = QtWidgets.QComboBox()
        self.ChannelTypeComboBox.addItems(['readout', 'fiducial'])
        crossLayout.addRow('Channel type:', self.ChannelTypeComboBox)

        self.CrossModalBorderTrimSpinBox = QtWidgets.QSpinBox()
        self.CrossModalBorderTrimSpinBox.setRange(0, 500)
        self.CrossModalBorderTrimSpinBox.setValue(0)
        self.CrossModalBorderTrimSpinBox.setSuffix(' px')
        crossLayout.addRow('Border trim:', self.CrossModalBorderTrimSpinBox)
        self.CrossModalMaxShiftSpinBox = QtWidgets.QSpinBox()
        self.CrossModalMaxShiftSpinBox.setRange(0, 100000)
        self.CrossModalMaxShiftSpinBox.setValue(0)
        self.CrossModalMaxShiftSpinBox.setSpecialValueText('unbounded')
        self.CrossModalMaxShiftSpinBox.setSuffix(' px')
        crossLayout.addRow('Max shift (0 = unbounded):', self.CrossModalMaxShiftSpinBox)

        self.RunCrossModalPushButton = QtWidgets.QPushButton('Run Cross-Modal Alignment')
        crossLayout.addRow(self.RunCrossModalPushButton)
        self.CrossModalResultLabel = QtWidgets.QLabel('')
        crossLayout.addRow('Result:', self.CrossModalResultLabel)
        self.CrossModalOverlayFovComboBox = QtWidgets.QComboBox()
        self.CrossModalOverlayFovComboBox.setEditable(True)
        self.CrossModalOverlayFovComboBox.lineEdit().setPlaceholderText('blank = FOV just aligned above, or pick/type any FOV')
        crossLayout.addRow('Overlay FOV:', self.CrossModalOverlayFovComboBox)
        self.CrossModalShowOverlayPushButton = QtWidgets.QPushButton('Show Overlay')
        crossLayout.addRow(self.CrossModalShowOverlayPushButton)
        self.CrossModalAcceptPushButton, self.CrossModalRejectPushButton, crossAcceptRow = self._accept_reject_row()
        crossLayout.addRow(crossAcceptRow)

        controls_layout.addWidget(crossGroup)

        # -- cell-based --
        cellGroup = QtWidgets.QGroupBox('3. Cell-Based Alignment')
        cellLayout = QtWidgets.QFormLayout(cellGroup)
        self.CellFovSpinBox = QtWidgets.QSpinBox()
        self.CellFovSpinBox.setRange(1, 100000)
        self.CellFovSpinBox.setValue(1)
        cellLayout.addRow('FOV:', self.CellFovSpinBox)
        self.CellReferenceHybeComboBox = QtWidgets.QComboBox()
        cellLayout.addRow('Reference hybe:', self.CellReferenceHybeComboBox)
        self.CellChannelTypeComboBox = QtWidgets.QComboBox()
        self.CellChannelTypeComboBox.addItems(['readout', 'fiducial'])
        cellLayout.addRow('Channel type:', self.CellChannelTypeComboBox)
        self.CellPadSpinBox = QtWidgets.QSpinBox()
        self.CellPadSpinBox.setRange(0, 500)
        self.CellPadSpinBox.setValue(10)
        self.CellPadSpinBox.setSuffix(' px')
        # Real compute_cell_alignment parameter (crop size the phase
        # correlation itself runs on), not a display-only margin -- a
        # bigger pad gives the algorithm more room to detect a larger
        # shift. The preview reuses this same value so it always shows
        # the actual window alignment computed against.
        cellLayout.addRow('Pad:', self.CellPadSpinBox)
        self.CellOverlayAutoSaveThresholdSpinBox = QtWidgets.QSpinBox()
        self.CellOverlayAutoSaveThresholdSpinBox.setRange(0, 500)
        self.CellOverlayAutoSaveThresholdSpinBox.setValue(5)
        self.CellOverlayAutoSaveThresholdSpinBox.setSuffix(' px')
        # Automatic-mode overlay PNGs are no longer generated for every
        # cell (too slow -- drawing+saving one is ~9x the cost of the
        # actual per-cell alignment fit itself). Instead a cell's overlay
        # is auto-saved only when its own cell-level residual shift
        # exceeds this threshold -- flags the cells worth a human look
        # without paying the cost for the (usual) well-behaved majority.
        # Use Save All Cell Overlays below to generate every cell's
        # overlay on demand regardless of this threshold.
        cellLayout.addRow('Auto-save overlay if shift >', self.CellOverlayAutoSaveThresholdSpinBox)
        self.RunCellAlignmentPushButton = QtWidgets.QPushButton('Run Cell Alignment')
        cellLayout.addRow(self.RunCellAlignmentPushButton)
        self.CellResultsListWidget = QtWidgets.QListWidget()
        cellLayout.addRow('Results (per cell):', self.CellResultsListWidget)

        self.CellPreviewHybeComboBox = QtWidgets.QComboBox()
        cellLayout.addRow('Preview hybe:', self.CellPreviewHybeComboBox)

        self.CellOverlayFovLineEdit = QtWidgets.QLineEdit()
        self.CellOverlayFovLineEdit.setPlaceholderText('FOV, e.g. 1')
        cellLayout.addRow('Overlay FOV:', self.CellOverlayFovLineEdit)
        self.CellOverlayCellListWidget = QtWidgets.QListWidget()
        self.CellOverlayCellListWidget.setMaximumHeight(100)
        cellLayout.addRow('Overlay cell:', self.CellOverlayCellListWidget)
        self.CellShowOverlayPushButton = QtWidgets.QPushButton('Show All-Readouts Overlay')
        cellLayout.addRow(self.CellShowOverlayPushButton)
        self.SaveAllCellOverlaysPushButton = QtWidgets.QPushButton('Save All Cell Overlays')
        # On-demand batch generation of every cell's overlay PNG (for
        # skimming the whole run's alignment quality by eye), independent
        # of the auto-save-on-large-shift threshold above -- covers the
        # cells that didn't trip the threshold too.
        cellLayout.addRow(self.SaveAllCellOverlaysPushButton)
        self.CellAcceptPushButton, self.CellRejectPushButton, cellAcceptRow = self._accept_reject_row()
        cellLayout.addRow(cellAcceptRow)
        controls_layout.addWidget(cellGroup)

        # -- spot-based (scaffold only -- unlocked once spot localization exists) --
        self.SpotGroupBox = QtWidgets.QGroupBox('4. Spot-Based Alignment')
        spotLayout = QtWidgets.QVBoxLayout(self.SpotGroupBox)
        spotLayout.addWidget(QtWidgets.QLabel('Requires spot localization -- not available yet.'))
        self.SpotGroupBox.setEnabled(False)
        controls_layout.addWidget(self.SpotGroupBox)

        controls_layout.addStretch()

    def _accept_reject_row(self):
        row = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        accept_button = QtWidgets.QPushButton('Accept')
        reject_button = QtWidgets.QPushButton('Reject')
        accept_button.setEnabled(False)
        reject_button.setEnabled(False)
        row_layout.addWidget(accept_button)
        row_layout.addWidget(reject_button)
        return accept_button, reject_button, row

    def _path_row(self, dialog_title):
        row = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        line_edit = QtWidgets.QLineEdit()
        browse_button = QtWidgets.QPushButton('Browse...')

        def browse():
            path = QtWidgets.QFileDialog.getExistingDirectory(row, dialog_title)
            if path:
                line_edit.setText(path)

        browse_button.clicked.connect(browse)
        row_layout.addWidget(line_edit)
        row_layout.addWidget(browse_button)
        return line_edit, row

    def populate_reference_hybe_choices(self, hybe_records):
        self.ReferenceHybeComboBox.clear()
        self.ReferenceHybeComboBox.addItems([r['folder'] for r in hybe_records])

    def populate_cell_reference_hybe_choices(self, hybe_records):
        """
        Separate from populate_reference_hybe_choices (which only ever
        reflects the single currently-active modality) -- cell-based
        alignment now processes hybes from BOTH modalities (see
        MainWindow._other_modality_cell_alignment_inputs), so its own
        anchor-hybe combo needs to offer choices from both too, not just
        whichever one happens to be current.
        """
        self._repopulate_editable_combo(self.CellReferenceHybeComboBox, [r['folder'] for r in hybe_records])

    def populate_rna_reference_hybe_choices(self, hybe_records):
        self._repopulate_editable_combo(self.RnaReferenceHybeComboBox, [r['folder'] for r in hybe_records])

    def populate_dna_reference_hybe_choices(self, hybe_records):
        self._repopulate_editable_combo(self.DnaReferenceHybeComboBox, [r['folder'] for r in hybe_records])

    def populate_overlay_fov_choices(self, fov_list):
        self._repopulate_editable_combo(self.CrossModalOverlayFovComboBox, [str(f) for f in fov_list])

    @staticmethod
    def _repopulate_editable_combo(combo, items):
        """clear()+addItems() on an editable combo wipes its typed/selected
        text too -- restore whatever was there before, so repopulating
        choices (e.g. after re-parsing a layout) never silently discards
        a value the user already set or loaded from config."""
        current = combo.currentText()
        combo.clear()
        combo.addItems(items)
        if current:
            combo.setCurrentText(current)

    def is_manual_mode(self):
        return self.ModeManualRadioButton.isChecked()
