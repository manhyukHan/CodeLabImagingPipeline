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

    Remove Transient Spots / Remove Unassigned spots: both scoped to the
    current view's current (hybe, channel), matching auto-detect's own
    scoping. "Transient" means "not yet written to vlinks.h5" -- Remove
    Transient Spots re-reads the on-disk state for this (hybe, channel)
    and reverts the in-memory view to exactly that (so anything already
    saved survives; anything added/edited since the last save doesn't).
    Remove Unassigned spots clears the view outright (both permanent and
    transient) for this (hybe, channel) -- if Save View is then clicked,
    the now-empty state is what gets written, i.e. this really does
    delete them from vlinks.h5 once saved. (Named for its primary FOV-view
    use -- clearing the unassigned pool; in Cell view it clears that
    cell's own spots for this hybe/channel instead.)

    Remove all spots (FOV view only): a bigger, separate action -- clears
    every spot for the CURRENT (hybe, channel), both the unassigned pool
    AND every cell's own spots (not just whichever single category
    "Remove Unassigned spots" would touch from this view). Confirmed via
    a warning dialog. In-memory only, same "Save View to persist"
    convention as every other edit here -- a cell's own spots removed
    this way still need THAT cell's own Cell view + Save View to actually
    reach vlinks.h5. Disabled outside FOV view -- there's no equivalent
    single-cell "wipe everything" button; Remove Unassigned spots already
    covers that narrower case for a selected cell.

    Undo/Redo: a generalized snapshot stack over this view's own spot
    list (see MainWindow._push_spot_undo_snapshot) -- every mutating
    action (auto-detect, manual click add/remove, 3D localization, the
    Remove buttons above) pushes a pre-action snapshot, with no
    distinction between "was this an add or a remove." Undo restores the
    previous snapshot and pushes the current state onto the redo stack;
    any new action after an Undo clears the redo stack (standard linear
    undo/redo, not branching history).

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
        """
        Layout order (per explicit request): Modality -> Hybe -> Channel ->
        Threshold%/Absolute pair -> Min distance/Pad pair -> FOV -> Append
        Mode -> Run Auto-Detect -> [Refresh/Show Displayer/3D Localization]
        -> the 3 list views -> [Remove Transient/Remove Unassigned/Remove
        all] -> [Undo/Redo] -> Save View -> (info label, progress, log).
        """
        Widget.setObjectName('SpotLocalizationPanel')
        layout = QtWidgets.QVBoxLayout(Widget)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        # -- Hybe | Channel -- no separate Modality selector: HybeComboBox
        # itself offers every configured modality's hybes at once (see
        # populate_hybe_choices), each item tagged with its own owning
        # modality, so there's nothing left for a modality picker to do
        # here.
        self.HybeComboBox = QtWidgets.QComboBox()
        form.addRow('Hybe:', self.HybeComboBox)

        self.ChannelComboBox = QtWidgets.QComboBox()
        form.addRow('Channel:', self.ChannelComboBox)

        # -- Threshold (%) | Threshold (absolute) -- kept live-linked,
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

        # -- Min distance | Cell crop padding --
        distancePadRow = QtWidgets.QWidget()
        distancePadLayout = QtWidgets.QHBoxLayout(distancePadRow)
        distancePadLayout.setContentsMargins(0, 0, 0, 0)
        self.MinDistanceSpinBox = QtWidgets.QSpinBox()
        self.MinDistanceSpinBox.setRange(1, 100)
        self.MinDistanceSpinBox.setValue(3)
        distancePadLayout.addWidget(QtWidgets.QLabel('Min distance (px):'))
        distancePadLayout.addWidget(self.MinDistanceSpinBox)
        self.PadSpinBox = QtWidgets.QSpinBox()
        self.PadSpinBox.setRange(0, 100)
        self.PadSpinBox.setValue(10)
        distancePadLayout.addWidget(QtWidgets.QLabel('Cell crop padding (px, Cell view only):'))
        distancePadLayout.addWidget(self.PadSpinBox)
        form.addRow(distancePadRow)

        # -- FOV spinbox --
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

        # -- Append mode --
        self.AppendModeCheckBox = QtWidgets.QCheckBox(
            'Append Mode (Run Auto-Detect adds to the current view instead of replacing it)')
        layout.addWidget(self.AppendModeCheckBox)

        # -- Run auto-detect --
        self.AutoDetectPushButton = QtWidgets.QPushButton('Run Auto-Detect')
        layout.addWidget(self.AutoDetectPushButton)

        # -- Refresh cell list | Show spot crop displayer | 3D localization --
        actionRow = QtWidgets.QWidget()
        actionLayout = QtWidgets.QHBoxLayout(actionRow)
        actionLayout.setContentsMargins(0, 0, 0, 0)
        self.RefreshCellListPushButton = QtWidgets.QPushButton('Refresh Cell List')
        actionLayout.addWidget(self.RefreshCellListPushButton)
        self.ShowDisplayerPushButton = QtWidgets.QPushButton('Show Spot Crop Displayer')
        self.ShowDisplayerPushButton.setCheckable(True)
        actionLayout.addWidget(self.ShowDisplayerPushButton)
        self.Show3DLocalizationPushButton = QtWidgets.QPushButton('3D Localization...')
        self.Show3DLocalizationPushButton.setCheckable(True)
        actionLayout.addWidget(self.Show3DLocalizationPushButton)
        layout.addWidget(actionRow)

        # -- FOV listview | Cell/FOV-view listview | Spot listview --
        listsRow = QtWidgets.QWidget()
        listsLayout = QtWidgets.QHBoxLayout(listsRow)
        listsLayout.setContentsMargins(0, 0, 0, 0)

        fovListCol = QtWidgets.QVBoxLayout()
        fovListCol.addWidget(QtWidgets.QLabel('FOV (all spots, this FOV) -- click to jump hybe/channel:'))
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

        # -- Remove transient (current hybe/channel/modality only) --
        # There is deliberately NO FOV-wide wipe button. Saving is scoped to
        # the current (hybe, channel), so a control that clears every hybe at
        # once could destroy spots for hybes the user never opened -- the
        # remove and save scopes are kept identical on purpose.
        removeRow = QtWidgets.QWidget()
        removeRowLayout = QtWidgets.QHBoxLayout(removeRow)
        removeRowLayout.setContentsMargins(0, 0, 0, 0)
        self.RemoveTransientSpotsPushButton = QtWidgets.QPushButton('Revert This Hybe/Channel')
        removeRowLayout.addWidget(self.RemoveTransientSpotsPushButton)
        # One operation over one store: clears assigned and unassigned alike
        # for the current hybe/channel/modality -- the capability the old
        # "Remove Unassigned spots" button could not provide because the
        # split store hid assigned spots from it. In-memory; Save persists
        # the emptied slice.
        self.ClearHybeChannelPushButton = QtWidgets.QPushButton('Clear This Hybe/Channel')
        removeRowLayout.addWidget(self.ClearHybeChannelPushButton)
        layout.addWidget(removeRow)

        # -- Undo | Redo -- see MainWindow._push_spot_undo_snapshot/_undo_spot_edit/_redo_spot_edit --
        undoRedoRow = QtWidgets.QWidget()
        undoRedoLayout = QtWidgets.QHBoxLayout(undoRedoRow)
        undoRedoLayout.setContentsMargins(0, 0, 0, 0)
        self.UndoPushButton = QtWidgets.QPushButton('Undo')
        self.UndoPushButton.setEnabled(False)
        undoRedoLayout.addWidget(self.UndoPushButton)
        self.RedoPushButton = QtWidgets.QPushButton('Redo')
        self.RedoPushButton.setEnabled(False)
        undoRedoLayout.addWidget(self.RedoPushButton)
        layout.addWidget(undoRedoRow)

        # -- Save current spots --
        self.SaveCurrentSpotsPushButton = QtWidgets.QPushButton('Save Current Spots')
        layout.addWidget(self.SaveCurrentSpotsPushButton)
        self.SaveAllFovSpotsPushButton = QtWidgets.QPushButton('Save ALL FOV Spots (every hybe/channel)')
        self.SaveAllFovSpotsPushButton.setToolTip(
            'Reassigns every spot in this FOV against the current cells and persists '
            'EVERY (modality, hybe, channel) slice at once -- not just the currently '
            'viewed one. Takes seconds (full reassignment + every slice write), but '
            'nothing is left behind, so it needs re-running far less often.')
        layout.addWidget(self.SaveAllFovSpotsPushButton)

        infoLabel = QtWidgets.QLabel(
            'Spots are attached to the current view immediately. Save Current Spots writes EVERY '
            'cell\'s current spots for this FOV to vlinks.h5 in one pass -- not just whichever cell/view '
            'happens to be open -- so picking spots across many cells in Cell view and then saving once '
            'is enough. The FOV-level unassigned pool is identified against the current cell mask first '
            '(newly-identified ones join their owning cell, the rest stay unassigned) and saved too. The '
            'Cell Segmentation tab\'s Send/Save buttons still promote/persist transient cells more broadly.')
        infoLabel.setWordWrap(True)
        layout.addWidget(infoLabel)

        self.ProgressBar = QtWidgets.QProgressBar()
        layout.addWidget(self.ProgressBar)

        # The panel log boxes moved into the one combined log window (see
        # ui/log_window.py) -- the stretch keeps the controls top-anchored
        # where the log box used to soak up the leftover height.
        layout.addStretch(1)

        self.HybeComboBox.currentIndexChanged.connect(self._on_hybe_changed)

    def populate_hybe_choices(self, total_active_hybe_list):
        """
        total_active_hybe_list: [(hybe_record, modality_name), ...] -- the
        union of every configured modality's active hybes (MainWindow.
        total_active_hybe_list), NOT one modality's own list. Each combo
        item's data is the (record, modality) pair itself, so a hybe
        folder name that happens to collide across modalities still
        resolves unambiguously (see current_hybe_folder/current_hybe_
        modality) and this panel never needs its own modality selector.

        Preserves the current selection across a refresh by matching
        (folder, modality) -- same pattern MainWindow.
        _refresh_cell_preview_reference_choices already uses for
        CellPreviewReferenceHybeComboBox -- so re-populating with THE SAME
        content (e.g. every redundant call after a modality that isn't
        even part of this list changes) never resets what's selected.
        This is the fix for a real bug: the old version's unconditional
        clear()+addItems() reset HybeComboBox's selection on every single
        modality switch, silently navigating the crop displayer away from
        whatever hybe the user's in-progress spot picks were on.
        """
        current_key = self._current_key()
        self.HybeComboBox.blockSignals(True)
        self.HybeComboBox.clear()
        for record, modality in total_active_hybe_list:
            label = f"{record['folder']} ({modality})"
            self.HybeComboBox.addItem(label, (record, modality))
        if self.HybeComboBox.count():
            restore_index = next((i for i in range(self.HybeComboBox.count())
                                  if self._item_key(i) == current_key), 0)
            self.HybeComboBox.setCurrentIndex(restore_index)
        self.HybeComboBox.blockSignals(False)
        self._on_hybe_changed()

    def _item_key(self, index):
        data = self.HybeComboBox.itemData(index)
        return (data[0]['folder'], data[1]) if data is not None else (None, None)

    def _current_key(self):
        data = self.HybeComboBox.currentData()
        return (data[0]['folder'], data[1]) if data is not None else (None, None)

    def current_hybe_folder(self):
        """Real hybe folder name for whatever's currently selected, or '' if nothing is."""
        data = self.HybeComboBox.currentData()
        return data[0]['folder'] if data is not None else ''

    def current_hybe_modality(self):
        """Owning modality name for whatever's currently selected, or None if nothing is."""
        data = self.HybeComboBox.currentData()
        return data[1] if data is not None else None

    def select_hybe(self, folder, modality):
        """Finds and selects the combo item matching (folder, modality) exactly. No-op if not found."""
        for i in range(self.HybeComboBox.count()):
            if self._item_key(i) == (folder, modality):
                self.HybeComboBox.setCurrentIndex(i)
                return

    def _on_hybe_changed(self):
        data = self.HybeComboBox.currentData()
        record = data[0] if data is not None else None
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

    def populate_cell_choices(self, cells, n_spots_by_cell=None):
        """
        Row 0 is always the FOV pseudo-row (the FOV-level unassigned-spot
        pool view); rows 1+ are real cells. n_spots_by_cell: {cell.id: n}
        supplied by the caller from the session's SpotContainer -- cells
        hold no spot lists of their own.
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
        n_spots_by_cell = n_spots_by_cell or {}
        for cell in cells:
            item = QtWidgets.QListWidgetItem(
                f'Cell {cell.id}: {n_spots_by_cell.get(cell.id, 0)} spot(s)')
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
