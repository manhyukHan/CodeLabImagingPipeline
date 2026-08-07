from PyQt5 import QtWidgets


class AlignmentPanelUI(object):
    """
    The full 4-layer alignment chain: FOV (within-experiment), experiment
    (cross-modal), cell-based, and spot-based (scaffolded here, disabled
    until the spot-localization panel exists to unlock it).

    No panel-wide mode toggle -- per explicit request, each of the first
    three layers instead exposes its own explicit "current" vs "all"
    pair: the current-FOV/current-cell button computes into a staged
    result with its own Accept/Reject (nothing persists until Accept,
    review it in the preview first), while the all-FOV/all-cells button
    always computes AND immediately saves, no staging -- once the current
    one's result looks right with a given set of parameters, running it
    against everything is the explicit next step, not a separate global
    setting to remember to flip.

    Every preview -- FOV, cross-modal, cell, interactive or auto-saved --
    is a pop-up (canvas/alignment_preview_window.py), never embedded here.
    An earlier version embedded the FOV/cross-modal preview directly in
    this panel; a real screenshot showed the controls column squeezing it
    into an unreadably stretched aspect ratio, so nothing gets embedded now.
    """
    def setupUi(self, Widget):
        Widget.setObjectName('AlignmentPanel')
        controls_layout = QtWidgets.QVBoxLayout(Widget)

        # -- within-experiment (FOV) --
        sameModalityGroup = QtWidgets.QGroupBox('1. Same-Modality (FOV) Alignment')
        sameModalityLayout = QtWidgets.QFormLayout(sameModalityGroup)
        self.ModalityComboBox = QtWidgets.QComboBox()
        self.ModalityComboBox.addItems(['DNA', 'RNA'])
        sameModalityLayout.addRow('Modality:', self.ModalityComboBox)
        self.SameModalityFovSpinBox = QtWidgets.QSpinBox()
        self.SameModalityFovSpinBox.setRange(1, 100000)
        self.SameModalityFovSpinBox.setValue(1)
        sameModalityLayout.addRow('FOV:', self.SameModalityFovSpinBox)
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
        self.RunFovAlignmentPushButton = QtWidgets.QPushButton('Run Current FOV Alignment')
        sameModalityLayout.addRow(self.RunFovAlignmentPushButton)
        self.SameModalityAcceptPushButton, self.SameModalityRejectPushButton, sameModalityAcceptRow = self._accept_reject_row()
        sameModalityLayout.addRow(sameModalityAcceptRow)
        self.RunAllFovAlignmentPushButton = QtWidgets.QPushButton('Run All FOV Alignment (Auto-Save)')
        # Every FOV in the Ingestion tab's FOV list, computed AND saved
        # immediately -- no staging, no Accept step. Use Run Current FOV
        # Alignment above first to confirm the parameters on one real FOV.
        sameModalityLayout.addRow(self.RunAllFovAlignmentPushButton)
        self.SameModalityResultsListWidget = QtWidgets.QListWidget()
        sameModalityLayout.addRow('Results:', self.SameModalityResultsListWidget)
        self.SameModalityOverlayFovSpinBox = QtWidgets.QSpinBox()
        self.SameModalityOverlayFovSpinBox.setRange(1, 100000)
        self.SameModalityOverlayFovSpinBox.setValue(1)
        sameModalityLayout.addRow('Overlay FOV:', self.SameModalityOverlayFovSpinBox)
        self.SameModalityChannelTypeComboBox = QtWidgets.QComboBox()
        self.SameModalityChannelTypeComboBox.addItems(['readout', 'fiducial'])
        sameModalityLayout.addRow('Overlay channel:', self.SameModalityChannelTypeComboBox)
        self.SameModalityShowOverlayPushButton = QtWidgets.QPushButton('Show All-Readouts Overlay')
        sameModalityLayout.addRow(self.SameModalityShowOverlayPushButton)
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

        self.CrossModalFovSpinBox = QtWidgets.QSpinBox()
        self.CrossModalFovSpinBox.setRange(1, 100000)
        self.CrossModalFovSpinBox.setValue(1)
        crossLayout.addRow('FOV:', self.CrossModalFovSpinBox)
        self.RunCrossModalPushButton = QtWidgets.QPushButton('Run Cross-Modal Alignment for Current FOV')
        crossLayout.addRow(self.RunCrossModalPushButton)
        self.CrossModalResultsListWidget = QtWidgets.QListWidget()
        # Every FOV with either a disk-persisted or in-session cross-modal
        # result -- not just whichever FOV was aligned most recently.
        # Clicking a row shows that FOV's overlay.
        self.CrossModalResultsListWidget.setMaximumHeight(120)
        crossLayout.addRow('Results:', self.CrossModalResultsListWidget)
        self.CrossModalAcceptPushButton, self.CrossModalRejectPushButton, crossAcceptRow = self._accept_reject_row()
        crossLayout.addRow(crossAcceptRow)
        self.RunAllCrossModalPushButton = QtWidgets.QPushButton('Run Cross-Modal Alignment for All FOVs (Auto-Save)')
        # Every FOV in the Ingestion tab's FOV list, computed AND saved
        # immediately -- no staging, no Accept step.
        crossLayout.addRow(self.RunAllCrossModalPushButton)
        self.CrossModalOverlayFovSpinBox = QtWidgets.QSpinBox()
        self.CrossModalOverlayFovSpinBox.setRange(1, 100000)
        self.CrossModalOverlayFovSpinBox.setValue(1)
        crossLayout.addRow('Overlay FOV:', self.CrossModalOverlayFovSpinBox)
        self.CrossModalShowOverlayPushButton = QtWidgets.QPushButton('Show Overlay')
        crossLayout.addRow(self.CrossModalShowOverlayPushButton)

        controls_layout.addWidget(crossGroup)

        # -- cell-based --
        # Three tiers, top to bottom: (1) per-cell tuning -- compute+preview
        # ONE cell, cheap enough to re-run after every parameter tweak, its
        # own Accept/Reject so accepting it can never be confused with
        # accepting a whole-FOV batch; (2) whole-FOV commit -- once the
        # parameters look right on real cells, run every cell in the FOV;
        # (3) visualization -- pure read-only browsing of whatever's already
        # saved/staged, driven entirely by FOV/reference-hybe selection,
        # never triggers computation or a write.
        cellGroup = QtWidgets.QGroupBox('3. Cell-Based Alignment')
        cellLayout = QtWidgets.QFormLayout(cellGroup)

        # -- tier 1: per-cell tuning --
        self.CellFovSpinBox = QtWidgets.QSpinBox()
        self.CellFovSpinBox.setRange(1, 100000)
        self.CellFovSpinBox.setValue(1)
        cellLayout.addRow('FOV:', self.CellFovSpinBox)
        self.CellIdSpinBox = QtWidgets.QSpinBox()
        self.CellIdSpinBox.setRange(1, 100000)
        self.CellIdSpinBox.setValue(1)
        cellLayout.addRow('Cell ID:', self.CellIdSpinBox)
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
        self.PreviewThisCellPushButton = QtWidgets.QPushButton("Preview This Cell's Alignment")
        # Computes cell-based alignment for JUST (FOV, Cell ID) above using
        # the live Reference hybe/Channel/Pad values -- lets a parameter
        # tweak be checked against one real cell in well under a second,
        # instead of paying Run Cell Alignment's whole-FOV cost just to see
        # if it helped. Stages into ITS OWN pending slot, separate from the
        # whole-FOV run below -- so its Accept/Reject can never accidentally
        # apply to (or be confused with) a whole-FOV batch. Nothing is
        # written to vlinks.h5 or drawn to a PNG until Accept.
        cellLayout.addRow(self.PreviewThisCellPushButton)
        self.CellResultsListWidget = QtWidgets.QListWidget()
        # Each row is one cell in whichever FOV is picked via Overlay FOV
        # below; its text re-derives against whichever hybe is currently
        # picked in Preview reference hybe below (see MainWindow._refresh_
        # cell_fov_panels) -- not a fixed label. Clicking a row draws the
        # 2-hybe (reference vs. target), 3-column preview. A pure read of
        # already-saved/staged data (tier 3) -- positioned here, right
        # after Preview This Cell's Alignment, so the just-previewed
        # cell's context is visible before deciding Accept/Reject.
        cellLayout.addRow('Results (per cell, per hybe):', self.CellResultsListWidget)
        self.PerCellAcceptPushButton, self.PerCellRejectPushButton, perCellAcceptRow = self._accept_reject_row()
        cellLayout.addRow(perCellAcceptRow)

        # -- tier 2: whole-FOV commit --
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
        self.RunCellAlignmentPushButton = QtWidgets.QPushButton('Align All Cells in FOV')
        # Uses the SAME FOV/Reference hybe/Channel type/Pad values as the
        # per-cell tool above -- once those check out on a real cell there,
        # this commits to every cell in the FOV. Always computes AND saves
        # immediately, no staging/Accept step -- the per-cell tool above
        # already gives a cheap, reviewable way to validate parameters
        # before committing to the whole FOV, so a second review step here
        # would just be redundant.
        cellLayout.addRow(self.RunCellAlignmentPushButton)

        # -- tier 3: visualization only (never computes or writes) --
        self.CellOverlayFovSpinBox = QtWidgets.QSpinBox()
        self.CellOverlayFovSpinBox.setRange(1, 100000)
        self.CellOverlayFovSpinBox.setValue(1)
        # Drives BOTH result lists (per-hybe above, overlay below) purely
        # by selection -- reading whatever's already saved/staged for that
        # FOV, never computing or writing anything.
        cellLayout.addRow('Overlay FOV:', self.CellOverlayFovSpinBox)
        self.CellOverlayCellListWidget = QtWidgets.QListWidget()
        self.CellOverlayCellListWidget.setMaximumHeight(100)
        # Clicking a row draws the one-vs-all sequential all-readouts
        # overlay (3 columns) for that cell -- independent of whatever's
        # selected in Preview reference hybe below.
        cellLayout.addRow('Results (per cell, overlay):', self.CellOverlayCellListWidget)
        self.CellPreviewReferenceHybeComboBox = QtWidgets.QComboBox()
        # FOV-wide (union of every hybe ANY cell in the current Overlay
        # FOV has data for) -- free choice of reference, independent of
        # whatever hybe actually anchored a target's own phase-correlation
        # at compute time (see ACell.matrix_between). Drives BOTH what
        # each row in "Results (per cell, per hybe)" above displays AND
        # (as the reference side) what clicking one of those rows
        # previews -- the row itself supplies the target hybe, no
        # separate target combo needed.
        cellLayout.addRow('Preview reference hybe:', self.CellPreviewReferenceHybeComboBox)
        self.SaveAllCellOverlaysPushButton = QtWidgets.QPushButton('Save All Cell Overlays')
        # On-demand batch generation of every cell's overlay PNG (for
        # skimming the whole run's alignment quality by eye), independent
        # of the auto-save-on-large-shift threshold above -- covers the
        # cells that didn't trip the threshold too. A read of already-saved
        # data, same as the rest of this tier -- computes no new alignment.
        cellLayout.addRow(self.SaveAllCellOverlaysPushButton)
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
