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
        Layout order, rebuilt around the fact that there are now TWO KINDS
        of localization and they need different controls:

          1  Hybe | Channel | FOV        what to look at
          2  Append mode | Cell padding  global, both kinds
          3  Mode + a stacked page       v1/v2 want anchor thresholds;
                                         v3/v4 want a model and a p-gate
          -  Run                         one button, two connections
          4  Refresh | Crop displayer | 3D Spot Viewer
          -  the three lists, remove/undo/save, info, progress

        THE THRESHOLDS WERE NEVER GLOBAL. Threshold %, absolute and min
        distance are the ANCHOR step's parameters -- where to start a
        Gaussian fit -- and a learned engine has no anchor step to point
        them at. Leaving them on screen under v3 would offer numbers that
        do nothing. Append mode and cell padding are genuinely shared: one
        says whether a run replaces the view, the other how much context a
        cell crop carries, and both are true of any engine.
        """
        Widget.setObjectName('SpotLocalizationPanel')
        layout = QtWidgets.QVBoxLayout(Widget)

        # ---- 1. what to look at -----------------------------------------
        # Hybe, channel and FOV name the pixels. They were three rows with
        # a form label each; one row is what they are.
        scopeRow = QtWidgets.QWidget()
        scopeLayout = QtWidgets.QHBoxLayout(scopeRow)
        scopeLayout.setContentsMargins(0, 0, 0, 0)
        scopeLayout.addWidget(QtWidgets.QLabel('Hybe:'))
        # No separate Modality selector: HybeComboBox itself offers every
        # configured modality's hybes at once (see populate_hybe_choices),
        # each item tagged with its own owning modality.
        self.HybeComboBox = QtWidgets.QComboBox()
        self.HybeComboBox.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                        QtWidgets.QSizePolicy.Fixed)
        scopeLayout.addWidget(self.HybeComboBox, 3)
        scopeLayout.addWidget(QtWidgets.QLabel('Channel:'))
        self.ChannelComboBox = QtWidgets.QComboBox()
        scopeLayout.addWidget(self.ChannelComboBox, 1)
        scopeLayout.addWidget(QtWidgets.QLabel('FOV:'))
        self.FovSpinBox = QtWidgets.QSpinBox()
        # BOUNDED TO WHAT EXISTS, once the store is known -- see
        # set_fov_range. The range starts at 1..1 rather than 1..100000
        # because an unbounded spinbox lets a person walk to a FOV the
        # experiment does not have, and every read after that fails
        # somewhere further down with a message about a missing file.
        self.FovSpinBox.setRange(1, 1)
        self.FovSpinBox.setValue(1)
        scopeLayout.addWidget(self.FovSpinBox)
        layout.addWidget(scopeRow)

        # ---- 2. global to every engine ----------------------------------
        globalRow = QtWidgets.QWidget()
        globalLayout = QtWidgets.QHBoxLayout(globalRow)
        globalLayout.setContentsMargins(0, 0, 0, 0)
        self.AppendModeCheckBox = QtWidgets.QCheckBox(
            'Append Mode (a run adds to the current view instead of '
            'replacing it)')
        globalLayout.addWidget(self.AppendModeCheckBox)
        globalLayout.addStretch(1)
        globalLayout.addWidget(
            QtWidgets.QLabel('Cell crop padding (px, Cell view only):'))
        self.PadSpinBox = QtWidgets.QSpinBox()
        self.PadSpinBox.setRange(0, 100)
        self.PadSpinBox.setValue(10)
        globalLayout.addWidget(self.PadSpinBox)
        layout.addWidget(globalRow)

        # ---- 3. mode, and the controls that belong to it ----------------
        modeRow = QtWidgets.QWidget()
        modeLayout = QtWidgets.QHBoxLayout(modeRow)
        modeLayout.setContentsMargins(0, 0, 0, 0)
        modeLayout.addWidget(QtWidgets.QLabel('Mode:'))
        self.EngineComboBox = QtWidgets.QComboBox()
        # ONE VOCABULARY WITH THE TRACING PANEL -- tracing_v2.ROUTE_*. The
        # 3D-localization popup's combo used to store 'gaussian' for the v1
        # route while its own label read 'v1 gaussian'.
        from codelab_pipeline.localization import tracing_v2 as _R
        self.EngineComboBox.addItem('v1 (anchor + bounded gaussian)',
                                    _R.ROUTE_V1)
        self.EngineComboBox.addItem('v2-anchor-fit (anchor + PSF fit)',
                                    _R.ROUTE_V2)
        self.EngineComboBox.addItem('v3-psfmatcher (learned, automatic)',
                                    _R.ROUTE_V3)
        self.EngineComboBox.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                          QtWidgets.QSizePolicy.Fixed)
        modeLayout.addWidget(self.EngineComboBox, 1)
        layout.addWidget(modeRow)

        # A PAGE SWAP, NOT A GREY-OUT. Each page keeps its own values while
        # hidden, so switching away and back restores exactly what was
        # there -- the same rule ChromatinTracingPanel follows, and for the
        # same reason: a pixel threshold and a model choice are not two
        # spellings of one number.
        self.ModeStackedWidget = QtWidgets.QStackedWidget()
        layout.addWidget(self.ModeStackedWidget)

        # page 0: the anchor engines (v1, v2)
        anchorPage = QtWidgets.QWidget()
        anchorLayout = QtWidgets.QVBoxLayout(anchorPage)
        anchorLayout.setContentsMargins(0, 0, 0, 0)
        thresholdRow = QtWidgets.QWidget()
        thresholdLayout = QtWidgets.QHBoxLayout(thresholdRow)
        thresholdLayout.setContentsMargins(0, 0, 0, 0)
        # Threshold % and absolute stay live-linked, see MainWindow's
        # _sync_threshold_from_percent / _from_absolute.
        thresholdLayout.addWidget(
            QtWidgets.QLabel('Threshold (% of scope max):'))
        self.ThresholdPercentLineEdit = QtWidgets.QLineEdit('50')
        self.ThresholdPercentLineEdit.setPlaceholderText(
            '% of scope max, e.g. 50')
        thresholdLayout.addWidget(self.ThresholdPercentLineEdit)
        thresholdLayout.addWidget(QtWidgets.QLabel('Absolute:'))
        self.ThresholdAbsoluteLineEdit = QtWidgets.QLineEdit()
        self.ThresholdAbsoluteLineEdit.setPlaceholderText(
            'absolute value -- kept in sync with % above')
        thresholdLayout.addWidget(self.ThresholdAbsoluteLineEdit)
        thresholdLayout.addWidget(QtWidgets.QLabel('Min distance (px):'))
        self.MinDistanceSpinBox = QtWidgets.QSpinBox()
        self.MinDistanceSpinBox.setRange(1, 100)
        self.MinDistanceSpinBox.setValue(3)
        thresholdLayout.addWidget(self.MinDistanceSpinBox)
        anchorLayout.addWidget(thresholdRow)
        self.ModeStackedWidget.addWidget(anchorPage)

        # page 1: the learned engines (v3, and v4 when it arrives)
        learnedPage = QtWidgets.QWidget()
        learnedLayout = QtWidgets.QHBoxLayout(learnedPage)
        learnedLayout.setContentsMargins(0, 0, 0, 0)
        learnedLayout.addWidget(QtWidgets.QLabel('Model:'))
        self.ModelComboBox = QtWidgets.QComboBox()
        self.ModelComboBox.setToolTip(
            'A trained run under <repo>/models, named for whoever labelled '
            'it. Both calibrations in a run are fitted against ONE '
            "reviewer's keep/drop, so whose it is matters.")
        self.ModelComboBox.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                         QtWidgets.QSizePolicy.Fixed)
        learnedLayout.addWidget(self.ModelComboBox, 1)
        self.PreviewPGatePushButton = QtWidgets.QPushButton(
            'Preview p histogram...')
        self.PreviewPGatePushButton.setToolTip(
            'The posterior gate. Shows the distribution of p over the spots '
            'currently in view, with examples from either side of a '
            'threshold, and removes what falls below it -- the same shape '
            'as Remove Z-Rejected, on a different number.')
        learnedLayout.addWidget(self.PreviewPGatePushButton)
        self.MakeModelPushButton = QtWidgets.QPushButton('Make new model...')
        self.MakeModelPushButton.setToolTip(
            'Build a review bundle, review it, and train a model on your own '
            'experiment and your own judgement.')
        learnedLayout.addWidget(self.MakeModelPushButton)
        self.ModeStackedWidget.addWidget(learnedPage)

        # ---- the run button ---------------------------------------------
        # ONE BUTTON, TWO CONNECTIONS. What it runs depends on the mode; a
        # second button would let a person press the one the mode is not on.
        self.AutoDetectPushButton = QtWidgets.QPushButton('Run Auto-Detect')
        layout.addWidget(self.AutoDetectPushButton)

        # ---- 4. viewers --------------------------------------------------
        actionRow = QtWidgets.QWidget()
        actionLayout = QtWidgets.QHBoxLayout(actionRow)
        actionLayout.setContentsMargins(0, 0, 0, 0)
        self.RefreshCellListPushButton = QtWidgets.QPushButton(
            'Refresh Cell List')
        actionLayout.addWidget(self.RefreshCellListPushButton)
        self.ShowDisplayerPushButton = QtWidgets.QPushButton(
            'Show Spot Crop Displayer')
        self.ShowDisplayerPushButton.setCheckable(True)
        actionLayout.addWidget(self.ShowDisplayerPushButton)
        # '3D Localization' named a v1/v2 ACTION. A learned engine localizes
        # in 3D on its own, so for those spots this window is a VIEWER --
        # and it is a viewer for the others too, because whatever produced
        # a spot, the window shows a coordinate and a z_status and those
        # are the same objects either way. It can still re-fit, which is
        # why the name had to stop claiming that is what it is for.
        self.Show3DLocalizationPushButton = QtWidgets.QPushButton(
            '3D Spot Viewer...')
        self.Show3DLocalizationPushButton.setCheckable(True)
        actionLayout.addWidget(self.Show3DLocalizationPushButton)
        layout.addWidget(actionRow)

        self.EngineComboBox.currentIndexChanged.connect(
            lambda _i: self.apply_mode_visibility())
        self.apply_mode_visibility()

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
        # Removes ONLY spots 3D localization ran on and REJECTED
        # (ASpot.z_status == 'rejected'). Spots nobody has fitted are
        # 'not_fit' and are deliberately left alone -- "the fit said no"
        # and "nobody asked" are different facts, and only the first is a
        # measured negative safe to sweep. Same (hybe, channel) scope as
        # every other removal here, in memory, persisted by Save.
        self.RemoveZRejectedPushButton = QtWidgets.QPushButton('Remove Z-rejected')
        self.RemoveZRejectedPushButton.setToolTip(
            'Remove spots whose 3D fit was rejected, for this hybe/channel.\n'
            'Spots that have never been 3D-localized are left untouched.\n'
            'In memory -- Save Current Spots persists it.')
        removeRowLayout.addWidget(self.RemoveZRejectedPushButton)
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

    # -- mode ---------------------------------------------------------------

    def selected_engine(self):
        """The route this panel names -- tracing_v2.ROUTE_*."""
        from codelab_pipeline.localization import tracing_v2 as R
        data = self.EngineComboBox.currentData()
        return R.route(data if data else self.EngineComboBox.currentText())

    def selected_engine_is_learned(self):
        """True for v3 (and v4 when it lands): no anchor step to configure."""
        from codelab_pipeline.localization import tracing_v2 as R
        return R.is_v3(self.selected_engine())

    def apply_mode_visibility(self):
        """Show the page that belongs to the chosen mode."""
        learned = self.selected_engine_is_learned()
        self.ModeStackedWidget.setCurrentIndex(1 if learned else 0)
        self.AutoDetectPushButton.setText(
            'Run Automatic Spot Localization' if learned
            else 'Run Auto-Detect')
        return learned

    def set_fov_range(self, n_fovs, keep=True):
        """Bound the FOV spinbox to the FOVs this experiment HAS.

        Every FOV spinbox in this app was 1..100000, so a person could
        walk to a FOV the store does not contain and the failure surfaced
        somewhere much further down as a missing file. A spinbox that
        cannot name a FOV that does not exist is the cheapest place to
        say so.

        `keep` clamps the current value into the new range rather than
        letting Qt do it silently at the next edit.
        """
        n = max(1, int(n_fovs or 1))
        cur = int(self.FovSpinBox.value())
        self.FovSpinBox.setRange(1, n)
        if keep:
            self.FovSpinBox.setValue(min(max(cur, 1), n))
        self.FovSpinBox.setToolTip(
            f'1 to {n} -- the FOVs this experiment has')
        return n

    def populate_models(self, runs, select=None):
        """Fill the model combo from model_store.available().

        A run with PROBLEMS is listed with them rather than hidden: a
        person choosing a model is exactly who should see that one of its
        files no longer matches its manifest.
        """
        self.ModelComboBox.blockSignals(True)
        self.ModelComboBox.clear()
        for r in runs or ():
            bits = [r['name']]
            if r.get('is_default'):
                bits.append('(default)')
            if not r.get('has_multispot'):
                bits.append('- no multispot calibration')
            if r.get('problems'):
                bits.append('!! ' + r['problems'][0])
            self.ModelComboBox.addItem('  '.join(bits), r['path'])
        if select:
            i = self.ModelComboBox.findData(select)
            if i >= 0:
                self.ModelComboBox.setCurrentIndex(i)
        self.ModelComboBox.blockSignals(False)
        return self.ModelComboBox.count()

    def selected_model_dir(self):
        return self.ModelComboBox.currentData()

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
        # PRESERVE the chosen channel across hybe changes (per request:
        # switching the hybe used to reset the channel every time) --
        # only when the new hybe lacks it does the selection move, and
        # then to the first READOUT channel, never the fiducial (spots
        # live in the signal channel).
        previous = self.ChannelComboBox.currentText()
        # blockSignals so clear()+addItems() reads as one atomic "channel
        # list changed" update, not a transient empty-then-refilled state
        self.ChannelComboBox.blockSignals(True)
        self.ChannelComboBox.clear()
        if record is not None:
            self.ChannelComboBox.addItems([str(c) for c in record['channels']])
            idx = self.ChannelComboBox.findText(previous) if previous else -1
            if idx < 0:
                fiducial = record.get('fiducial_channel')
                readout_channels = [c for c in record['channels']
                                    if c != fiducial]
                if readout_channels:
                    idx = self.ChannelComboBox.findText(
                        str(readout_channels[0]))
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
