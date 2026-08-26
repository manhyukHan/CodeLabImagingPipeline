from PyQt5 import QtWidgets

from codelab_pipeline.alignment import chain as alignment


class AlignmentPanelUI(object):
    """
    The 3-layer alignment chain: FOV (within-experiment), experiment
    (cross-modal), and cell-based. A 4th, spot-based layer was scaffolded
    here early on but never built out -- chromatin tracing (ui/chromatin_
    tracing_panel.py) supersedes it as its own independent panel instead.

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
        self.SameModalityFovSpinBox = QtWidgets.QSpinBox()
        self.SameModalityFovSpinBox.setRange(1, 100000)
        self.SameModalityFovSpinBox.setValue(1)
        sameModalityLayout.addRow('FOV:', self.SameModalityFovSpinBox)
        # One reference hybe PER MODALITY, exactly like cell-based
        # alignment below. Alignment is per-modality maths -- a hybe is
        # only comparable to another hybe of its own modality, fiducial
        # to fiducial -- so every modality needs its own anchor, and a
        # single shared combo could only ever align one of them.
        self.SameModalityReferenceHybeFormLayout = QtWidgets.QFormLayout()
        self.SameModalityReferenceHybeFormLayout.setContentsMargins(0, 0, 0, 0)
        self.SameModalityReferenceHybeComboBoxes = {}
        _refHost = QtWidgets.QWidget()
        _refHost.setLayout(self.SameModalityReferenceHybeFormLayout)
        sameModalityLayout.addRow(_refHost)
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
        self.SameModalityChannelTypeComboBox = QtWidgets.QComboBox()
        # fiducial FIRST, so it is the default. The matrices on this tab are
        # always fitted fiducial-to-fiducial regardless of this setting (see
        # canvas.draw_fov_all_readouts_overlay) -- the fiducial images the same
        # physical object in every hybe, which is exactly what makes a
        # before/after overlay readable as "did alignment work". A readout
        # overlay shows genuinely different biological content per hybe, so
        # residual misregistration is hard to distinguish from real signal
        # change. Showing readout by default meant the default view was the one
        # that answers the question least directly.
        #
        # Sits with the run settings rather than down by the overlay
        # controls: the current-FOV run pops its own overlay immediately,
        # so this is read at RUN time too, not only when Show is clicked.
        self.SameModalityChannelTypeComboBox.addItems(['fiducial', 'readout'])
        sameModalityLayout.addRow('Overlay channel:', self.SameModalityChannelTypeComboBox)
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
        # One row per (hybe, modality) for ONE FOV -- 100+ rows on the real
        # project, so a default-height list showed a handful at a time.
        # Same minimum as the per-cell results list below.
        self.SameModalityResultsListWidget.setMinimumHeight(320)
        sameModalityLayout.addRow('Results:', self.SameModalityResultsListWidget)
        self.SameModalityOverlayFovSpinBox = QtWidgets.QSpinBox()
        self.SameModalityOverlayFovSpinBox.setRange(1, 100000)
        self.SameModalityOverlayFovSpinBox.setValue(1)
        sameModalityLayout.addRow('Overlay FOV:', self.SameModalityOverlayFovSpinBox)
        self.SameModalityShowOverlayPushButton = QtWidgets.QPushButton('Show All-Readouts Overlay')
        sameModalityLayout.addRow(self.SameModalityShowOverlayPushButton)
        controls_layout.addWidget(sameModalityGroup)

        # -- cross-experiment --
        crossGroup = QtWidgets.QGroupBox('2. Cross-Modality Alignment')
        crossLayout = QtWidgets.QFormLayout(crossGroup)

        # No storage-path rows and no fixed RNA/DNA combos, per explicit
        # direction: storage paths come from the project manifest (there is
        # nothing to type), and the bridge reference-hybe selectors are
        # rebuilt per ACTIVATED modality -- the same generalized-name
        # principle as the cell-level residual section's own per-modality
        # combos below. The first activated modality is the shared (hub)
        # frame; every other modality bridges into it with its own hybe.
        self.CrossModalSharedFrameLabel = QtWidgets.QLabel('-')
        crossLayout.addRow('Shared frame (hub):', self.CrossModalSharedFrameLabel)
        self.CrossModalReferenceHybeComboBoxes = {}
        self.CrossModalReferenceHybeFormLayout = QtWidgets.QFormLayout()
        crossLayout.addRow(self.CrossModalReferenceHybeFormLayout)

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

        # Z drift is deliberately NOT a control here. It is a measured
        # component of the cross-modal result, exactly like dx/dy -- so it
        # is computed by Run Cross-Modal Alignment, shown in Results, and
        # persisted by Accept. Exposing it as an editable spinbox framed a
        # measurement as a parameter and detached it from the run that
        # produces it.
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
        # Scoped to ONE FOV (the spinbox above) across every modality, so
        # it is short -- but give it room to show every bridge at once
        # rather than a two-row window.
        self.CrossModalResultsListWidget.setMinimumHeight(160)
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
        # No modality picker here -- per explicit decision, this app only
        # ever holds ONE modality's cells resident in memory at a time
        # (self.cell_container/cell_container_permanent, each carrying a
        # single .modality), so a separate "which modality" selector was
        # pure redundant friction: whichever cell (FOV, Cell ID) resolves
        # to already tells you its own modality directly. Cell-based
        # alignment against the OTHER modality's hybes was also dropped
        # entirely (it consistently gave worse results than same-modality
        # cell alignment), so each modality has its own independent,
        # explicit reference-hybe setting below instead of one shared
        # combo whose OWN picked hybe's modality used to silently decide
        # which modality's cells got processed.
        self.CellReferenceHybeFormLayout = QtWidgets.QFormLayout()
        # One combo per configured modality, ALL shown simultaneously
        # (mirrors Cross-Modality Alignment's own RNA/DNA fields, always
        # visible side by side, never toggled by a selector) -- rebuilt
        # by build_cell_reference_hybe_fields whenever the configured
        # modality set changes. self.CellReferenceHybeComboBoxes is a
        # dict[modality] -> QComboBox, populated there.
        self.CellReferenceHybeComboBoxes = {}
        cellLayout.addRow(self.CellReferenceHybeFormLayout)
        self.CellChannelTypeComboBox = QtWidgets.QComboBox()
        # fiducial FIRST, so it is the default -- and here it is not only a
        # display choice: this value selects the channel compute_cell_alignment
        # actually FITS on, both the YX phase correlation and the Z leg. The
        # cell-level step is a small residual refinement measured between two
        # crops of the same physical location, which needs a channel showing
        # the same physical content on both sides. Fiducial does; a readout
        # round images different targets per hybe, so a real residual and a
        # real change in signal look alike to the fit.
        self.CellChannelTypeComboBox.addItems(['fiducial', 'readout'])
        cellLayout.addRow('Channel type:', self.CellChannelTypeComboBox)
        self.CellZMaxShiftSpinBox = QtWidgets.QSpinBox()
        self.CellZMaxShiftSpinBox.setRange(0, 500)
        self.CellZMaxShiftSpinBox.setValue(int(alignment.MAX_CELL_Z_SHIFT_PLANES))
        self.CellZMaxShiftSpinBox.setSuffix(' planes')
        # How far the cell-level Z refinement may move a hybe. Was derived from
        # pad (pad/2 = 5 planes), which bounded a DEPTH correction by an XY
        # search radius -- unrelated quantities. Measured on a real run, that
        # cap was truncating genuine drift: applied shifts piled up exactly at
        # 5 while 16.4% of hybes were rejected on magnitude alone with |z| up
        # to 11. Raise it if hybes are still being rejected with plausible
        # shifts; the reconstruction-residual gate beside it, not this bound,
        # is what actually rejects bad fits.
        cellLayout.addRow('Max Z shift:', self.CellZMaxShiftSpinBox)
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
        # One row per (hybe, modality) -- 111+ rows on the real project, so
        # a default-height list showed a handful at a time (explicitly
        # reported too short). Minimum raised so a real per-hybe review is
        # possible without fighting the scrollbar; still grows with the
        # panel.
        self.CellResultsListWidget.setMinimumHeight(320)
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
        # 0 px = save EVERY cell's overlay, and that is now the default
        # (lowered from 10). The old default existed only because drawing
        # was on the GUI thread and dominated the run; overlay rendering
        # now happens in a background worker on its own Agg figure, so the
        # run itself is no longer held up by it. Saving as many as
        # possible is what makes viewing cheap: the "Results (per cell,
        # overlay)" viewer SHOWS the saved PNG instead of recomputing a
        # ~35 s composite (see MainWindow._show_cell_all_readouts_overlay),
        # so an overlay that exists on disk is an overlay that opens
        # instantly. Raise this if a run's background overlay pass is
        # taking longer than you want to leave running.
        self.CellOverlayAutoSaveThresholdSpinBox.setValue(0)
        self.CellOverlayAutoSaveThresholdSpinBox.setSuffix(' px')
        self.CellOverlayAutoSaveThresholdSpinBox.setSpecialValueText('every cell')
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
        self.RunCellAlignmentAllPushButton = QtWidgets.QPushButton(
            'Align All Cells in ALL FOVs (Auto-Save)')
        # Every FOV in the Ingestion tab's list, computed AND saved
        # immediately -- no staging, no Accept, exactly like Run All FOV
        # Alignment one section up. Validate the parameters with Preview
        # This Cell / Align All Cells in FOV first; this is the commit.
        cellLayout.addRow(self.RunCellAlignmentAllPushButton)

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

        # -- progress + log, the parity this panel was missing --
        # Ingestion, Cell Segmentation and Spot Localization all have both;
        # alignment had neither, and its workers reported only through a
        # transient statusBar message. Same-modality alignment is the
        # longest silent operation in the app: measured on real 1024x1024
        # MIPs it is ~3.5 s PER HYBE (Powell converges in ~155 objective
        # evaluations at ~20 ms each), so a 78-hybe FOV is ~4.5 minutes
        # during which AlignmentWorker used to emit exactly ONE signal --
        # after the whole FOV finished. Nothing was wrong; it just looked
        # that way.
        self.ProgressBar = QtWidgets.QProgressBar()
        controls_layout.addWidget(self.ProgressBar)

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

    def build_cell_reference_hybe_fields(self, modality_names):
        """
        Rebuilds self.CellReferenceHybeComboBoxes to have exactly one
        combo per name in modality_names, all shown as separate rows in
        self.CellReferenceHybeFormLayout -- called whenever the configured
        modality set changes (MainWindow._activate_modalities, same hook
        point the Ingestion panel's modality setup already uses). Closest existing
        precedent: ingestion_panel.build_modality_name_fields (a
        positional-list rebuild of the same kind), adapted here to a
        name-keyed dict since each combo needs to be addressed by its own
        modality afterward, not by position.
        """
        while self.CellReferenceHybeFormLayout.rowCount():
            self.CellReferenceHybeFormLayout.removeRow(0)
        self.CellReferenceHybeComboBoxes = {}
        for name in modality_names:
            combo = QtWidgets.QComboBox()
            self.CellReferenceHybeComboBoxes[name] = combo
            self.CellReferenceHybeFormLayout.addRow(f'Reference hybe ({name}):', combo)

    def build_same_modality_reference_hybe_fields(self, modality_names):
        """One reference-hybe combo per configured modality for FOV
        alignment -- same rebuild-on-modality-change contract as
        build_cell_reference_hybe_fields."""
        while self.SameModalityReferenceHybeFormLayout.rowCount():
            self.SameModalityReferenceHybeFormLayout.removeRow(0)
        self.SameModalityReferenceHybeComboBoxes = {}
        for name in modality_names:
            combo = QtWidgets.QComboBox()
            self.SameModalityReferenceHybeComboBoxes[name] = combo
            self.SameModalityReferenceHybeFormLayout.addRow(f'Reference hybe ({name}):', combo)

    def populate_same_modality_reference_hybe_choices(self, total_active_hybe_list):
        """Route each hybe into its own modality's FOV-alignment combo --
        same scoping as populate_cell_reference_hybe_choices."""
        by_modality = {}
        for record, modality in total_active_hybe_list:
            by_modality.setdefault(modality, []).append(record)
        for modality, combo in self.SameModalityReferenceHybeComboBoxes.items():
            current = combo.currentData()['folder'] if combo.currentData() is not None else None
            combo.blockSignals(True)
            combo.clear()
            for record in by_modality.get(modality, []):
                combo.addItem(record['folder'], record)
            if combo.count():
                restore_index = next((i for i in range(combo.count())
                                      if combo.itemData(i)['folder'] == current), 0)
                combo.setCurrentIndex(restore_index)
            combo.blockSignals(False)

    def current_same_modality_reference_hybe(self, modality):
        """Selected reference hybe for `modality`, or '' if none."""
        combo = self.SameModalityReferenceHybeComboBoxes.get(modality)
        if combo is None:
            return ''
        data = combo.currentData()
        return data['folder'] if data is not None else ''

    def select_same_modality_reference_hybe(self, modality, folder):
        """Select `folder` in `modality`'s FOV-alignment combo. No-op if
        either doesn't exist."""
        combo = self.SameModalityReferenceHybeComboBoxes.get(modality)
        if combo is None:
            return
        for i in range(combo.count()):
            if combo.itemData(i)['folder'] == folder:
                combo.setCurrentIndex(i)
                return

    def same_modality_references(self):
        """{modality: hybe} for every modality with a real pick -- the
        FOV-alignment counterpart of cell_align_references()."""
        return {m: self.current_same_modality_reference_hybe(m)
                for m in self.SameModalityReferenceHybeComboBoxes
                if self.current_same_modality_reference_hybe(m)}

    def populate_cell_reference_hybe_choices(self, total_active_hybe_list):
        """
        total_active_hybe_list: [(hybe_record, modality_name), ...] --
        routed into each hybe's OWN modality's combo (self.
        CellReferenceHybeComboBoxes, built by build_cell_reference_hybe_
        fields) -- each combo is already modality-scoped, so its own
        items are bare hybe records (no more (record, modality) tuple-
        tagging needed, unlike populate_reference_hybe_choices, whose one
        shared combo still needs that to disambiguate).
        """
        by_modality = {}
        for record, modality in total_active_hybe_list:
            by_modality.setdefault(modality, []).append(record)
        for modality, combo in self.CellReferenceHybeComboBoxes.items():
            current = combo.currentData()['folder'] if combo.currentData() is not None else None
            combo.blockSignals(True)
            combo.clear()
            for record in by_modality.get(modality, []):
                combo.addItem(record['folder'], record)
            if combo.count():
                restore_index = next((i for i in range(combo.count())
                                      if combo.itemData(i)['folder'] == current), 0)
                combo.setCurrentIndex(restore_index)
            combo.blockSignals(False)

    def current_cell_reference_hybe(self, modality):
        """Real hybe folder name currently selected for `modality`'s own combo, or '' if none/unconfigured."""
        combo = self.CellReferenceHybeComboBoxes.get(modality)
        if combo is None:
            return ''
        data = combo.currentData()
        return data['folder'] if data is not None else ''

    def select_cell_reference_hybe(self, modality, folder):
        """Finds and selects `folder` in `modality`'s own combo. No-op if either doesn't exist."""
        combo = self.CellReferenceHybeComboBoxes.get(modality)
        if combo is None:
            return
        for i in range(combo.count()):
            if combo.itemData(i)['folder'] == folder:
                combo.setCurrentIndex(i)
                return

    def cell_align_references(self):
        """{modality: hybe} for every configured modality that currently has a real pick."""
        return {m: self.current_cell_reference_hybe(m) for m in self.CellReferenceHybeComboBoxes
               if self.current_cell_reference_hybe(m)}

    def build_cross_modal_reference_hybe_fields(self, modality_names):
        """
        Cross-modal counterpart of build_cell_reference_hybe_fields: one
        bridge-hybe combo per ACTIVATED modality, rebuilt whenever the
        configured modality set changes -- replacing the fixed RNA/DNA
        pair per the generalized-name principle.
        """
        while self.CrossModalReferenceHybeFormLayout.rowCount():
            self.CrossModalReferenceHybeFormLayout.removeRow(0)
        self.CrossModalReferenceHybeComboBoxes = {}
        for name in modality_names:
            combo = QtWidgets.QComboBox()
            self.CrossModalReferenceHybeComboBoxes[name] = combo
            self.CrossModalReferenceHybeFormLayout.addRow(f'Bridge hybe ({name}):', combo)

    def populate_cross_modal_reference_hybe_choices(self, total_active_hybe_list):
        """Route each hybe into its own modality's bridge combo -- same
        per-modality scoping as populate_cell_reference_hybe_choices."""
        by_modality = {}
        for record, modality in total_active_hybe_list:
            by_modality.setdefault(modality, []).append(record)
        for modality, combo in self.CrossModalReferenceHybeComboBoxes.items():
            current = combo.currentData()['folder'] if combo.currentData() is not None else None
            combo.blockSignals(True)
            combo.clear()
            for record in by_modality.get(modality, []):
                combo.addItem(record['folder'], record)
            if combo.count():
                restore_index = next((i for i in range(combo.count())
                                      if combo.itemData(i)['folder'] == current), 0)
                combo.setCurrentIndex(restore_index)
            combo.blockSignals(False)

    def current_cross_modal_reference_hybe(self, modality):
        """Real hybe folder currently selected as `modality`'s bridge hybe, or ''."""
        combo = self.CrossModalReferenceHybeComboBoxes.get(modality)
        if combo is None:
            return ''
        data = combo.currentData()
        return data['folder'] if data is not None else ''

    def select_cross_modal_reference_hybe(self, modality, folder):
        """Finds and selects `folder` in `modality`'s bridge combo. No-op if either doesn't exist."""
        combo = self.CrossModalReferenceHybeComboBoxes.get(modality)
        if combo is None:
            return
        for i in range(combo.count()):
            if combo.itemData(i)['folder'] == folder:
                combo.setCurrentIndex(i)
                return

    def cross_modal_references(self):
        """{modality: bridge hybe} for every modality with a real pick."""
        return {m: self.current_cross_modal_reference_hybe(m)
                for m in self.CrossModalReferenceHybeComboBoxes
                if self.current_cross_modal_reference_hybe(m)}

