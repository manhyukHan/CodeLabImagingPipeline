from PyQt5 import QtWidgets, QtCore


class SpotLocalizationPanelUI(object):
    """
    Interactive spot localization: pick a cell (or the FOV pseudo-row) +
    hybe/channel, then either auto-detect (peak_local_max) or manually
    click spots on the pop-up crop displayer (canvas/spot_crop_displayer.py).

    No separate Scope control -- the CellListWidget itself carries the
    view selector: its first row is always a special "FOV" pseudo-row
    (FOV_ROW_MARKER), everything after it is a real cell. Selecting the
    FOV row shows/edits the FOV-level unassigned-spot pool
    (fov_unassigned_spots on MainWindow); selecting a cell shows/edits
    that cell's own cell.spots. Both are real, editable views -- neither
    is a UI-only stand-in for the other.

    FOV-view spots start OUT unassigned (ASpot.cell == -1, its model
    default) regardless of how they were found (auto-detect or manual
    click) -- identification against the current cell mask only happens
    when the user explicitly saves the view (Save View), not at
    detect/click time. This keeps "browsing/editing the FOV pool" and
    "committing spots to their owning cells" as two distinct steps, and
    means Save View must be told to also persist whatever's left
    unassigned after identification.

    Cell-view spots are unambiguous from the moment they're created
    (ASpot.cell is always the selected cell's own id) -- ACell already
    owns its spots as real, persistent objects (unlike CellClassifier's
    lazy per-call views), so there's no separate staging container for
    them either.

    Manual click mode now works in BOTH views (not cell-only): the crop
    displayer's spots_edited signal always has a live crop context to
    write into (see MainWindow._spot_crop_context's 'kind' field),
    fixing the earlier gap where FOV-view clicks were drawn but silently
    discarded.

    Append Mode (AppendModeCheckBox): when checked, Run Auto-Detect adds
    newly found spots on top of whatever's already shown in the current
    view for this (hybe, channel), instead of replacing them -- useful
    for combining multiple detection passes (e.g. different thresholds)
    without losing earlier results.

    Remove Transient Spots / Remove spots in view: both scoped to the
    current view's current (hybe, channel), matching auto-detect's own
    scoping. "Transient" means "not yet written to vlinks.h5" -- Remove
    Transient Spots re-reads the on-disk state for this (hybe, channel)
    and reverts the in-memory view to exactly that (so anything already
    saved survives; anything added/edited since the last save doesn't).
    Remove spots in view clears the view outright (both permanent and
    transient) for this (hybe, channel) -- if Save View is then clicked,
    the now-empty state is what gets written, i.e. this really does
    delete them from vlinks.h5 once saved.

    Remove All Spots (bottom, FOV view only): a much bigger, separate
    action -- a COMPLETE wipe of every spot in the current FOV, both the
    unassigned pool AND every cell's own spots, across every hybe/
    channel (not just the one currently selected). Confirmed via a
    warning dialog and immediately persisted as part of the same click
    (not staged for a later Save View), since a plain in-memory clear
    here wouldn't actually reach already-saved cells the way Save View's
    FOV-view branch normally would (that branch only ever writes cells
    it just identified new spots INTO, not ones being emptied out).
    Disabled outside FOV view -- there's no equivalent single-cell
    "wipe everything" button; Remove spots in view already covers that
    narrower case for a selected cell.

    Two list views (a third, per-cell "Spot" breakdown, is informational
    only and not a selector):
    - FOV (this FOV, all cells): per-(hybe, channel) spot COUNTS across
      every cell currently in this FOV, PLUS FOV-level unassigned counts
      -- "see all spots in FOV" at a glance.
    - Cell (transient, this FOV): the view selector described above --
      row 0 is always the FOV pseudo-row, rows 1+ are real cells.
    """
    FOV_ROW_MARKER = '__fov_view__'

    def setupUi(self, Widget):
        Widget.setObjectName('SpotLocalizationPanel')
        layout = QtWidgets.QVBoxLayout(Widget)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        # -- 1. Modality --
        self.ModalityComboBox = QtWidgets.QComboBox()
        self.ModalityComboBox.addItems(['DNA', 'RNA'])
        form.addRow('Modality:', self.ModalityComboBox)

        # -- 2. FOV spinbox --
        fovRow = QtWidgets.QWidget()
        fovRowLayout = QtWidgets.QHBoxLayout(fovRow)
        fovRowLayout.setContentsMargins(0, 0, 0, 0)
        self.FovSpinBox = QtWidgets.QSpinBox()
        self.FovSpinBox.setRange(1, 100000)
        self.FovSpinBox.setValue(1)
        fovRowLayout.addWidget(QtWidgets.QLabel('FOV:'))
        fovRowLayout.addWidget(self.FovSpinBox)
        fovRowLayout.addStretch()
        layout.addWidget(fovRow)

        # -- 3. Refresh cell list | Show spot crop displayer --
        actionRow = QtWidgets.QWidget()
        actionLayout = QtWidgets.QHBoxLayout(actionRow)
        actionLayout.setContentsMargins(0, 0, 0, 0)
        self.RefreshCellListPushButton = QtWidgets.QPushButton('Refresh Cell List')
        actionLayout.addWidget(self.RefreshCellListPushButton)
        self.ShowDisplayerPushButton = QtWidgets.QPushButton('Show Spot Crop Displayer')
        self.ShowDisplayerPushButton.setCheckable(True)
        actionLayout.addWidget(self.ShowDisplayerPushButton)
        layout.addWidget(actionRow)

        # -- 4. FOV listview | Cell/FOV-view listview | Spot listview --
        listsRow = QtWidgets.QWidget()
        listsLayout = QtWidgets.QHBoxLayout(listsRow)
        listsLayout.setContentsMargins(0, 0, 0, 0)

        fovListCol = QtWidgets.QVBoxLayout()
        fovListCol.addWidget(QtWidgets.QLabel('FOV (all spots, this FOV):'))
        self.FovListWidget = QtWidgets.QListWidget()
        self.FovListWidget.setMaximumHeight(120)
        fovListCol.addWidget(self.FovListWidget)
        listsLayout.addLayout(fovListCol)

        cellListCol = QtWidgets.QVBoxLayout()
        cellListCol.addWidget(QtWidgets.QLabel('View (FOV pool / Cell, this FOV):'))
        self.CellListWidget = QtWidgets.QListWidget()
        self.CellListWidget.setMaximumHeight(120)
        cellListCol.addWidget(self.CellListWidget)
        listsLayout.addLayout(cellListCol)

        spotListCol = QtWidgets.QVBoxLayout()
        spotListCol.addWidget(QtWidgets.QLabel('Spot (transient, this view):'))
        self.SpotListWidget = QtWidgets.QListWidget()
        self.SpotListWidget.setMaximumHeight(120)
        spotListCol.addWidget(self.SpotListWidget)
        listsLayout.addLayout(spotListCol)

        layout.addWidget(listsRow)

        # -- 5. Hybe | Channel --
        self.HybeComboBox = QtWidgets.QComboBox()
        form.addRow('Hybe:', self.HybeComboBox)

        self.ChannelComboBox = QtWidgets.QComboBox()
        form.addRow('Channel:', self.ChannelComboBox)

        # -- 6. Threshold (%) | Threshold (absolute) -- kept live-linked,
        # see MainWindow._sync_threshold_from_percent/_from_absolute.
        thresholdRow = QtWidgets.QWidget()
        thresholdLayout = QtWidgets.QHBoxLayout(thresholdRow)
        thresholdLayout.setContentsMargins(0, 0, 0, 0)
        self.ThresholdPercentLineEdit = QtWidgets.QLineEdit('50')
        self.ThresholdPercentLineEdit.setPlaceholderText('% of scope max, e.g. 50')
        thresholdLayout.addWidget(QtWidgets.QLabel('Threshold (% of scope max):'))
        thresholdLayout.addWidget(self.ThresholdPercentLineEdit)
        self.ThresholdAbsoluteLineEdit = QtWidgets.QLineEdit()
        self.ThresholdAbsoluteLineEdit.setPlaceholderText('absolute value -- kept in sync with % above')
        thresholdLayout.addWidget(QtWidgets.QLabel('Absolute value:'))
        thresholdLayout.addWidget(self.ThresholdAbsoluteLineEdit)
        form.addRow(thresholdRow)

        # -- 7. Min distance --
        self.MinDistanceSpinBox = QtWidgets.QSpinBox()
        self.MinDistanceSpinBox.setRange(1, 100)
        self.MinDistanceSpinBox.setValue(3)
        form.addRow('Min distance between spots (px):', self.MinDistanceSpinBox)

        # -- 8. Cell crop padding --
        self.PadSpinBox = QtWidgets.QSpinBox()
        self.PadSpinBox.setRange(0, 100)
        self.PadSpinBox.setValue(10)
        form.addRow('Cell crop padding (px, Cell view only):', self.PadSpinBox)

        # -- Append mode --
        self.AppendModeCheckBox = QtWidgets.QCheckBox(
            'Append Mode (Run Auto-Detect adds to the current view instead of replacing it)')
        layout.addWidget(self.AppendModeCheckBox)

        # -- 9. Run auto-detect --
        self.AutoDetectPushButton = QtWidgets.QPushButton('Run Auto-Detect')
        layout.addWidget(self.AutoDetectPushButton)

        # -- Remove transient / Remove spots in view (current view + current hybe/channel only) --
        removeRow = QtWidgets.QWidget()
        removeRowLayout = QtWidgets.QHBoxLayout(removeRow)
        removeRowLayout.setContentsMargins(0, 0, 0, 0)
        self.RemoveTransientSpotsPushButton = QtWidgets.QPushButton('Remove Transient Spots')
        removeRowLayout.addWidget(self.RemoveTransientSpotsPushButton)
        self.RemoveSpotsInViewPushButton = QtWidgets.QPushButton('Remove spots in view')
        removeRowLayout.addWidget(self.RemoveSpotsInViewPushButton)
        layout.addWidget(removeRow)

        # -- 10. Save view --
        self.SaveViewPushButton = QtWidgets.QPushButton('Save View')
        layout.addWidget(self.SaveViewPushButton)

        # -- Complete wipe: FOV view only, every cell + the unassigned pool,
        # confirmed + immediately persisted (see MainWindow._remove_all_spots_in_fov).
        # Deliberately below Save View and disabled outside FOV view -- this
        # is a much bigger, irreversible action than Remove spots in view.
        self.RemoveAllSpotsPushButton = QtWidgets.QPushButton('Remove All Spots')
        self.RemoveAllSpotsPushButton.setEnabled(False)
        layout.addWidget(self.RemoveAllSpotsPushButton)

        infoLabel = QtWidgets.QLabel(
            'Spots are attached to the current view immediately. Save View writes ONLY the current '
            'view to vlinks.h5: for a Cell view, just that cell (every other cell on disk untouched); '
            'for the FOV view, its unassigned spots are first identified against the current cell mask '
            '(newly-identified ones are added to their owning cell, the rest stay unassigned) and both '
            'the affected cells and the remaining unassigned pool are saved. The Cell Segmentation '
            'tab\'s Send/Save buttons still promote/persist transient cells more broadly.')
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
        """
        Row 0 is always the FOV pseudo-row (the FOV-level unassigned-spot
        pool view); rows 1+ are real cells (id + current total_num_spots).
        Selection preserved by cell id where possible, defaulting to the
        FOV row when nothing else was previously selected (matches the
        old Scope combobox's own default of Whole FOV).
        """
        previous_id = self.selected_cell_id()
        had_selection = self.CellListWidget.currentItem() is not None
        self.CellListWidget.clear()
        fov_item = QtWidgets.QListWidgetItem('FOV (unassigned spot pool)')
        fov_item.setData(QtCore.Qt.UserRole, self.FOV_ROW_MARKER)
        self.CellListWidget.addItem(fov_item)
        selected_item = fov_item
        for cell in cells:
            item = QtWidgets.QListWidgetItem(f'Cell {cell.id}: {cell.total_num_spots} spot(s)')
            item.setData(QtCore.Qt.UserRole, cell.id)
            self.CellListWidget.addItem(item)
            if had_selection and cell.id == previous_id:
                selected_item = item
        self.CellListWidget.setCurrentItem(selected_item)

    def selected_cell_id(self):
        item = self.CellListWidget.currentItem()
        if item is None:
            return None
        data = item.data(QtCore.Qt.UserRole)
        return None if data == self.FOV_ROW_MARKER else data

    def current_view(self):
        """'fov' or 'cell' -- see class docstring. Defaults to 'fov' when
        nothing is selected yet (mirrors the old Scope combobox default)."""
        item = self.CellListWidget.currentItem()
        if item is not None and item.data(QtCore.Qt.UserRole) != self.FOV_ROW_MARKER:
            return 'cell'
        return 'fov'

    def threshold_abs(self, scope_max):
        """
        Resolves the actual threshold_abs value peak_local_max should use:
        the absolute value field when it's filled in (a real number,
        non-empty), otherwise (% of scope max). Raises ValueError with a
        human-readable message on unparseable input, so the caller can
        surface it directly rather than a bare exception. The two fields
        are kept live-synced (see MainWindow), so in practice they always
        agree -- this still prefers Absolute so a manually-typed override
        (before any sync has happened, e.g. no crop loaded yet) still works.
        """
        abs_text = self.ThresholdAbsoluteLineEdit.text().strip()
        if abs_text:
            try:
                return float(abs_text)
            except ValueError:
                raise ValueError(f'Absolute value {abs_text!r} is not a number.')
        pct_text = self.ThresholdPercentLineEdit.text().strip()
        try:
            pct = float(pct_text) if pct_text else 50.0
        except ValueError:
            raise ValueError(f'Threshold percent {pct_text!r} is not a number.')
        return (pct / 100.0) * scope_max
