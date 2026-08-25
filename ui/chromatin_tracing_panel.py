from PyQt5 import QtWidgets, QtCore

# ChrTracer3's own real values (see codelab_pipeline.localization.
# localize_chromatin_trace_hybe / build_chromatin_trace_allele), not
# arbitrary starting points -- same single-source-of-truth pattern
# DEFAULT_PARAMS already follows in canvas/localize_3d_displayer.py, so the
# spinboxes' initial values and ResetDefaultsPushButton can never drift
# apart. multi_mode defaults OFF, per explicit request -- same "opt-in for
# the crowded-crop case" default 3D Localization's own MultiModeCheckBox
# already uses, not the earlier "expected common case" assumption.
#
# Per explicit request: spad/z_window/max_fiducial_drift are "cross-mode"
# -- crop placement, the mixture seed-search Z-window, and the drift-
# rejection gate are all cross-cutting between fiducial and readout
# fitting, never something the two would want to disagree about, so they
# stay single, shared fields. peak_bound/max_sigma/max_uncert/min_hb_ratio/
# min_ah_ratio are independently configurable per channel -- a fiducial
# bead and a real genomic-locus probe can have genuinely different
# brightness/PSF characteristics worth tuning separately. Both channels
# now START at ChrTracer3 FitPsf3D's own gate values (minHBratio 1.2,
# minAHratio 0.25), per explicit request (2026-08-20), superseding the
# earlier readout relaxation to 1.05; either stays freely re-tunable
# here if real traces start losing hybes. min_sep/multi_mode (mixture-mode
# search) are READOUT-ONLY, per explicit request -- a fiducial's whole
# purpose is ONE per-hybe drift-correction anchor (see localization.
# _localize_fiducial_hybe), so there's no legitimate multi-component case
# for it the way a real genomic-locus readout has.
# z_boundary_trim: planes shaved off EACH end of a crop's depth before the
# Gaussian fit runs -- boundary trimming, deliberately NOT a center window
# (+-N around a seed), per explicit decision: an allele near the top or
# bottom of its cell can have its real peak outside a seed-centered window,
# while a stack's outermost planes are out-of-focus junk regardless of where
# the allele sits. Display crops stay full-depth; only the fit domain shrinks.
CROSS_MODE_DEFAULTS = {'spad': 8, 'z_window': 15, 'max_fiducial_drift': 5.0,
                       'max_fiducial_drift_z': 10.0, 'z_boundary_trim': 10}
SHARED_FIT_DEFAULTS = {'peak_bound': 2.0, 'max_sigma': 2.5, 'max_uncert': 2.0, 'min_ah_ratio': 0.25}
READOUT_ONLY_FIT_DEFAULTS = {'min_sep': 3.0, 'multi_mode': False}
DEFAULT_PARAMS = {**CROSS_MODE_DEFAULTS,
                  'fiducial': {**SHARED_FIT_DEFAULTS, 'min_hb_ratio': 1.2},
                  'readout': {**SHARED_FIT_DEFAULTS, 'min_hb_ratio': 1.2, **READOUT_ONLY_FIT_DEFAULTS}}

# (attribute prefix, param key, row label, kind) -- single source of truth
# for both building the two-column grid and reading/resetting it, so the
# UI and the two columns' own widgets can never drift apart or disagree
# about which row is which. _SHARED_FIT_ROWS apply to both fiducial and
# readout columns; _READOUT_ONLY_FIT_ROWS render in the readout column
# only (the fiducial column's cell at that row is left blank).
_SHARED_FIT_ROWS = [
    ('PeakBound', 'peak_bound', 'Peak bound (px from seed):', 'double', 0.5, 10.0),
    ('MaxSigma', 'max_sigma', 'Max sigma (px, xy):', 'double', 0.5, 10.0),
    ('MaxUncert', 'max_uncert', 'Max uncertainty (95% CI, px):', 'double', 0.1, 10.0),
    ('MinHBRatio', 'min_hb_ratio', 'Min peak/background ratio:', 'double', 1.0, 10.0),
    ('MinAHRatio', 'min_ah_ratio', 'Min amplitude/peak ratio:', 'double', 0.0, 1.0),
]
_READOUT_ONLY_FIT_ROWS = [
    ('MinSep', 'min_sep', 'Min separation (px, merge closer peaks):', 'double', 0.5, 20.0),
    ('MultiMode', 'multi_mode', 'Multi-Gaussian mixture mode:', 'bool', None, None),
]


class ChromatinTracingPanelUI(object):
    """
    Chromatin tracing: an allele's (x,y) is already known (a spot the user
    already selected in Spot Localization -- no fresh detection, no
    fiducial-channel requirement for that seed click, no 3D localization
    prerequisite, per explicit request). This panel just configures which
    hybes participate, builds AnAllele records from the current spot
    selection, tunes the same kind of fit parameters 3D Localization
    already exposes (plus one new one -- max_fiducial_drift, the shared-
    frame rejection gate), previews one allele's crop grids, and runs the
    batch fit across every FOV in the background.

    Five sections, top to bottom:
     1. Hybes Involved -- checkbox list (same ExtendedSelection + Check/
        Uncheck Selected pattern as ingestion_panel.HybeListWidget) across
        every configured modality's hybes, default-checked wherever
        modality=='DNA' and datatype in ('H','R','T') (see MainWindow._
        default_chromatin_tracing_hybes) -- freely re-checkable for a
        different modality/hybe set. Reference hybe combo feeds the
        drift-rejection baseline (chromatin_tracing_reference_hybe).
     2. Alleles -- its own FOV/Hybe/Channel pickers (per confirmed real
        bug: this used to silently inherit whatever Spot Localization's
        own Hybe/Channel comboboxes happened to be showing, which could
        easily be left on the fiducial channel by habit, making a built
        allele's readout_channel collide with its own fiducial_channel
        and render two identical grids), a spot list scoped to that exact
        (FOV, hybe, channel) read straight from vlinks.h5 (same "reads
        real, persisted data" principle as CellSpotStatusDisplayer, not
        live in-memory session state), and "Build/Refresh Alleles from
        Selected Spots".
     3. Fit Parameters -- fiducial/readout columns, same 5 shared fields
        as Localize3DDisplayer (peak_bound/max_sigma/max_uncert/
        min_hb_ratio/min_ah_ratio) plus max_fiducial_drift (shared) and
        min_sep/multi_mode (readout-only -- no mixture mode for fiducial).
     4. Preview One Allele -- pick one allele, View Crop opens the
        fiducial/readout grid pop-ups.
     5. Fit All FOVs -- background batch run + progress bar.
    """
    def setupUi(self, Widget):
        Widget.setObjectName('ChromatinTracingPanel')
        layout = QtWidgets.QVBoxLayout(Widget)

        # -- 1. hybes involved --
        hybesGroup = QtWidgets.QGroupBox('1. Hybes Involved')
        hybesLayout = QtWidgets.QVBoxLayout(hybesGroup)

        self.HybeListWidget = QtWidgets.QListWidget()
        self.HybeListWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        hybesLayout.addWidget(self.HybeListWidget)

        hybeCheckButtonsRow = QtWidgets.QWidget()
        hybeCheckButtonsLayout = QtWidgets.QHBoxLayout(hybeCheckButtonsRow)
        hybeCheckButtonsLayout.setContentsMargins(0, 0, 0, 0)
        self.CheckSelectedHybesPushButton = QtWidgets.QPushButton('Check Selected')
        self.UncheckSelectedHybesPushButton = QtWidgets.QPushButton('Uncheck Selected')
        hybeCheckButtonsLayout.addWidget(self.CheckSelectedHybesPushButton)
        hybeCheckButtonsLayout.addWidget(self.UncheckSelectedHybesPushButton)
        hybesLayout.addWidget(hybeCheckButtonsRow)
        self.CheckSelectedHybesPushButton.clicked.connect(lambda: self._set_selected_hybe_check_state(QtCore.Qt.Checked))
        self.UncheckSelectedHybesPushButton.clicked.connect(lambda: self._set_selected_hybe_check_state(QtCore.Qt.Unchecked))

        refForm = QtWidgets.QFormLayout()
        self.ReferenceHybeComboBox = QtWidgets.QComboBox()
        refForm.addRow('Reference hybe (drift baseline):', self.ReferenceHybeComboBox)
        hybesLayout.addLayout(refForm)
        layout.addWidget(hybesGroup)

        # -- 2. alleles --
        allelesGroup = QtWidgets.QGroupBox('2. Alleles')
        allelesLayout = QtWidgets.QVBoxLayout(allelesGroup)

        # FOV | Hybe | Channel in one row -- this panel's OWN scope for
        # picking seed spots, independent of Spot Localization's own
        # current view (see class docstring on why).
        scopeRow = QtWidgets.QWidget()
        scopeLayout = QtWidgets.QHBoxLayout(scopeRow)
        scopeLayout.setContentsMargins(0, 0, 0, 0)
        scopeLayout.addWidget(QtWidgets.QLabel('FOV:'))
        self.AlleleFovSpinBox = QtWidgets.QSpinBox()
        self.AlleleFovSpinBox.setRange(1, 100000)
        self.AlleleFovSpinBox.setValue(1)
        scopeLayout.addWidget(self.AlleleFovSpinBox)
        scopeLayout.addWidget(QtWidgets.QLabel('Hybe:'))
        self.AlleleHybeComboBox = QtWidgets.QComboBox()
        scopeLayout.addWidget(self.AlleleHybeComboBox)
        scopeLayout.addWidget(QtWidgets.QLabel('Channel:'))
        self.AlleleChannelComboBox = QtWidgets.QComboBox()
        scopeLayout.addWidget(self.AlleleChannelComboBox)
        allelesLayout.addWidget(scopeRow)

        allelesLayout.addWidget(QtWidgets.QLabel('Spots in this FOV/hybe/channel (select which become alleles):'))
        self.SpotListWidget = QtWidgets.QListWidget()
        self.SpotListWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.SpotListWidget.setMaximumHeight(120)
        allelesLayout.addWidget(self.SpotListWidget)

        self.BuildAllelesPushButton = QtWidgets.QPushButton('Build/Refresh Alleles from Selected Spots')
        allelesLayout.addWidget(self.BuildAllelesPushButton)

        self.AlleleCountLabel = QtWidgets.QLabel('0 allele(s) in this FOV.')
        allelesLayout.addWidget(self.AlleleCountLabel)

        self.AlleleListWidget = QtWidgets.QListWidget()
        self.AlleleListWidget.setMaximumHeight(120)
        allelesLayout.addWidget(self.AlleleListWidget)
        layout.addWidget(allelesGroup)

        # -- 3. fit parameters --
        paramsGroup = QtWidgets.QGroupBox('3. Fit Parameters')
        paramsOuter = QtWidgets.QVBoxLayout(paramsGroup)

        def double_spin(default, minv, maxv, step=0.1, decimals=2):
            sb = QtWidgets.QDoubleSpinBox()
            sb.setDecimals(decimals)
            sb.setRange(minv, maxv)
            sb.setSingleStep(step)
            sb.setValue(default)
            return sb

        # -- cross-mode: one shared value, used identically by both
        # fiducial and readout fitting --
        crossForm = QtWidgets.QFormLayout()
        paramsOuter.addLayout(crossForm)

        self.SpadSpinBox = QtWidgets.QSpinBox()
        self.SpadSpinBox.setRange(1, 100)
        self.SpadSpinBox.setValue(CROSS_MODE_DEFAULTS['spad'])
        crossForm.addRow('Crop half-width (px):', self.SpadSpinBox)

        self.ZWindowSpinBox = QtWidgets.QSpinBox()
        self.ZWindowSpinBox.setRange(1, 200)
        self.ZWindowSpinBox.setValue(CROSS_MODE_DEFAULTS['z_window'])
        crossForm.addRow('Z search window (+/-px, mixture mode only):', self.ZWindowSpinBox)

        self.ZBoundaryTrimSpinBox = QtWidgets.QSpinBox()
        self.ZBoundaryTrimSpinBox.setRange(0, 100)
        self.ZBoundaryTrimSpinBox.setValue(CROSS_MODE_DEFAULTS['z_boundary_trim'])
        self.ZBoundaryTrimSpinBox.setSuffix(' planes')
        crossForm.addRow('Z boundary trim (each end):', self.ZBoundaryTrimSpinBox)

        self.MaxFiducialDriftSpinBox = double_spin(CROSS_MODE_DEFAULTS['max_fiducial_drift'], 0.5, 100.0)
        # The rejection gate, per explicit request: a hybe whose own
        # fiducial fit lands more than this many px (shared frame) from the
        # reference hybe's own fiducial is rejected outright -- readout
        # never even fit for that hybe.
        crossForm.addRow('Max fiducial drift vs. reference (px):', self.MaxFiducialDriftSpinBox)

        self.MaxFiducialDriftZSpinBox = double_spin(CROSS_MODE_DEFAULTS['max_fiducial_drift_z'], 0.5, 300.0)
        # Z gated separately, in PLANES: a fiducial fit can pass the XY
        # bound while landing on different content in depth (confirmed
        # real case: 20 planes off at only 1.4px XY), and such a fit
        # would "correct" every readout in that hybe by a bogus dz.
        crossForm.addRow('Max fiducial Z drift vs. reference (planes):', self.MaxFiducialDriftZSpinBox)

        # -- per-channel: independently tunable fiducial vs. readout
        # columns, per explicit request -- a fiducial bead and a real
        # genomic-locus probe can have genuinely different brightness/PSF
        # characteristics worth tuning separately. min_sep/multi_mode are
        # readout-only (no mixture mode for fiducial, per explicit
        # request) -- the fiducial column simply has no cell at those rows.
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel('Fiducial'), 0, 1)
        grid.addWidget(QtWidgets.QLabel('Readout'), 0, 2)
        self.FiducialSpinBoxes, self.ReadoutSpinBoxes = {}, {}

        def add_widget(row, column_name, boxes, attr, key, kind, minv, maxv, col):
            default = DEFAULT_PARAMS[column_name][key]
            if kind == 'double':
                widget = double_spin(default, minv, maxv)
            else:
                widget = QtWidgets.QCheckBox()
                widget.setChecked(default)
            setattr(self, f'{column_name.capitalize()}{attr}CheckBox' if kind == 'bool'
                   else f'{column_name.capitalize()}{attr}SpinBox', widget)
            boxes[key] = widget
            grid.addWidget(widget, row, col)

        row = 1
        for attr, key, label, kind, minv, maxv in _SHARED_FIT_ROWS:
            grid.addWidget(QtWidgets.QLabel(label), row, 0)
            add_widget(row, 'fiducial', self.FiducialSpinBoxes, attr, key, kind, minv, maxv, 1)
            add_widget(row, 'readout', self.ReadoutSpinBoxes, attr, key, kind, minv, maxv, 2)
            row += 1
        for attr, key, label, kind, minv, maxv in _READOUT_ONLY_FIT_ROWS:
            grid.addWidget(QtWidgets.QLabel(label), row, 0)
            add_widget(row, 'readout', self.ReadoutSpinBoxes, attr, key, kind, minv, maxv, 2)
            row += 1
        paramsOuter.addLayout(grid)

        self.ResetDefaultsPushButton = QtWidgets.QPushButton('Reset to Defaults')
        self.ResetDefaultsPushButton.clicked.connect(self.reset_defaults)
        paramsOuter.addWidget(self.ResetDefaultsPushButton)
        layout.addWidget(paramsGroup)

        # -- 4. preview one allele --
        previewGroup = QtWidgets.QGroupBox('4. Preview One Allele')
        previewLayout = QtWidgets.QVBoxLayout(previewGroup)
        previewForm = QtWidgets.QFormLayout()
        self.PreviewAlleleComboBox = QtWidgets.QComboBox()
        previewForm.addRow('Allele:', self.PreviewAlleleComboBox)
        previewLayout.addLayout(previewForm)
        self.ViewCropPushButton = QtWidgets.QPushButton('View Crop (fiducial + readout grids)')
        previewLayout.addWidget(self.ViewCropPushButton)
        layout.addWidget(previewGroup)

        # -- 5. fit all fovs --
        fitAllGroup = QtWidgets.QGroupBox('5. Fit All FOVs')
        fitAllLayout = QtWidgets.QVBoxLayout(fitAllGroup)
        self.FitAllFovsPushButton = QtWidgets.QPushButton('Fit All FOVs')
        fitAllLayout.addWidget(self.FitAllFovsPushButton)
        self.ProgressBar = QtWidgets.QProgressBar()
        fitAllLayout.addWidget(self.ProgressBar)
        self.StatusLabel = QtWidgets.QLabel('')
        self.StatusLabel.setWordWrap(True)
        fitAllLayout.addWidget(self.StatusLabel)
        layout.addWidget(fitAllGroup)

        layout.addStretch()

    # -- hybe checklist --

    def _set_selected_hybe_check_state(self, check_state):
        for item in self.HybeListWidget.selectedItems():
            item.setCheckState(check_state)

    def populate_hybe_list(self, total_active_hybe_list, default_checked):
        """
        total_active_hybe_list: [(hybe_record, modality_name), ...] -- every
        configured modality's hybes at once (same shape AlignmentPanelUI.
        populate_reference_hybe_choices already consumes), so chromatin
        tracing isn't locked to whichever modality happens to be "current"
        elsewhere. default_checked(record, modality) -> bool decides each
        item's initial check state (MainWindow._default_chromatin_tracing_
        hybes: modality=='DNA' and datatype in ('H','R','T')) -- freely
        re-checkable afterward via the Check/Uncheck Selected buttons.
        """
        self.HybeListWidget.clear()
        for record, modality in total_active_hybe_list:
            label = f"{record['folder']} ({modality}, datatype={record['datatype']})"
            item = QtWidgets.QListWidgetItem(label)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if default_checked(record, modality) else QtCore.Qt.Unchecked)
            item.setData(QtCore.Qt.UserRole, (record['folder'], modality))
            self.HybeListWidget.addItem(item)

    def checked_hybes(self):
        """[(folder, modality), ...] currently checked, in list order."""
        checked = []
        for i in range(self.HybeListWidget.count()):
            item = self.HybeListWidget.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                checked.append(item.data(QtCore.Qt.UserRole))
        return checked

    def set_checked_hybes(self, keys):
        """Restore HybeListWidget check states by (folder, modality) key --
        the inverse of checked_hybes, for config save/load. Keys not in the
        current list are ignored (the list is rebuilt from the live active-
        hybe set; a stale config must not error)."""
        want = {tuple(k) for k in keys}
        for i in range(self.HybeListWidget.count()):
            item = self.HybeListWidget.item(i)
            item.setCheckState(QtCore.Qt.Checked if tuple(item.data(QtCore.Qt.UserRole)) in want
                               else QtCore.Qt.Unchecked)

    def populate_reference_hybe_choices(self, total_active_hybe_list):
        current = self.current_reference_hybe_key()
        self.ReferenceHybeComboBox.blockSignals(True)
        self.ReferenceHybeComboBox.clear()
        for record, modality in total_active_hybe_list:
            self.ReferenceHybeComboBox.addItem(f"{record['folder']} ({modality})", (record['folder'], modality))
        if self.ReferenceHybeComboBox.count():
            restore_index = next((i for i in range(self.ReferenceHybeComboBox.count())
                                  if self.ReferenceHybeComboBox.itemData(i) == current), 0)
            self.ReferenceHybeComboBox.setCurrentIndex(restore_index)
        self.ReferenceHybeComboBox.blockSignals(False)

    def current_reference_hybe_key(self):
        return self.ReferenceHybeComboBox.currentData()

    def current_reference_hybe(self):
        data = self.ReferenceHybeComboBox.currentData()
        return data[0] if data is not None else ''

    # -- alleles --

    def populate_allele_hybe_choices(self, total_active_hybe_list):
        """
        total_active_hybe_list: [(hybe_record, modality_name), ...] -- same
        shape/source as the Hybes Involved checklist, but this combo isn't
        restricted to the checked subset: the seed spot's own anchor hybe
        doesn't need to be one of the traced rounds. itemData is the full
        (record, modality) pair (not just the folder name) so Channel
        choices can be derived from record['channels']/['fiducial_channel']
        without a second lookup.
        """
        current = self.current_allele_hybe_key()
        self.AlleleHybeComboBox.blockSignals(True)
        self.AlleleHybeComboBox.clear()
        for record, modality in total_active_hybe_list:
            self.AlleleHybeComboBox.addItem(f"{record['folder']} ({modality})", (record, modality))
        if self.AlleleHybeComboBox.count():
            restore_index = next((i for i in range(self.AlleleHybeComboBox.count())
                                  if self._allele_hybe_item_key(i) == current), 0)
            self.AlleleHybeComboBox.setCurrentIndex(restore_index)
        self.AlleleHybeComboBox.blockSignals(False)

    def _allele_hybe_item_key(self, index):
        data = self.AlleleHybeComboBox.itemData(index)
        return (data[0]['folder'], data[1]) if data is not None else (None, None)

    def current_allele_hybe_key(self):
        data = self.AlleleHybeComboBox.currentData()
        return (data[0]['folder'], data[1]) if data is not None else (None, None)

    def current_allele_hybe_record_and_modality(self):
        """(hybe_record, modality) for whatever's currently picked, or (None, None)."""
        data = self.AlleleHybeComboBox.currentData()
        return data if data is not None else (None, None)

    def populate_allele_channel_choices(self, record):
        """
        record: the hybe_record currently picked in AlleleHybeComboBox (or
        None to clear). Labels mark which channel is fiducial vs readout
        (record['fiducial_channel']) -- per confirmed real bug, nothing
        previously distinguished them, making it easy to accidentally pick
        the fiducial channel as an allele's own anchor/readout channel.
        """
        current = self.current_allele_channel()
        self.AlleleChannelComboBox.blockSignals(True)
        self.AlleleChannelComboBox.clear()
        fiducial_channel = record.get('fiducial_channel') if record is not None else None
        for ch in (record.get('channels', []) if record is not None else []):
            tag = 'fiducial' if ch == fiducial_channel else 'readout'
            self.AlleleChannelComboBox.addItem(f'{ch} ({tag})', int(ch))
        if self.AlleleChannelComboBox.count():
            # Prefer restoring the previous selection; failing that,
            # default to the first READOUT channel rather than index 0 --
            # channels are frequently listed fiducial-first, and index 0
            # defaulting straight to the fiducial channel is exactly the
            # trap this combobox exists to avoid (see class docstring).
            restore_index = next((i for i in range(self.AlleleChannelComboBox.count())
                                  if self.AlleleChannelComboBox.itemData(i) == current), None)
            if restore_index is None:
                restore_index = next((i for i in range(self.AlleleChannelComboBox.count())
                                      if self.AlleleChannelComboBox.itemData(i) != fiducial_channel), 0)
            self.AlleleChannelComboBox.setCurrentIndex(restore_index)
        self.AlleleChannelComboBox.blockSignals(False)

    def current_allele_channel(self):
        return self.AlleleChannelComboBox.currentData()

    def populate_spot_choices(self, indexed):
        """
        indexed: [(global_index, spot_dict), ...] (see MainWindow.
        _ordered_spot_dicts_for_scope) -- read straight from vlinks.h5,
        same "real, persisted data" source CellSpotStatusDisplayer already
        uses, not live in-memory session state. itemData is the spot_dict
        itself, so Build/Refresh can read it back without a second lookup.
        """
        self.SpotListWidget.clear()
        for global_index, d in indexed:
            cell_tag = 'unassigned' if d['cell'] == -1 else f"cell {d['cell']}"
            x, y, z = d['adj_coordinate']
            item = QtWidgets.QListWidgetItem(f'Spot {global_index} | {cell_tag} | ({x:.1f}, {y:.1f}, {z:.1f})')
            item.setData(QtCore.Qt.UserRole, d)
            self.SpotListWidget.addItem(item)

    def selected_spot_dicts(self):
        return [item.data(QtCore.Qt.UserRole) for item in self.SpotListWidget.selectedItems()]

    def populate_allele_list(self, rows):
        """rows: [(allele_id, label), ...] -- pure display, caller (MainWindow)
        already resolved the label text from the real AnAllele objects."""
        self.AlleleListWidget.clear()
        for _, label in rows:
            self.AlleleListWidget.addItem(label)
        self.AlleleCountLabel.setText(f'{len(rows)} allele(s) in this FOV.')

    def populate_preview_allele_choices(self, rows):
        """rows: [(allele_id, label), ...]."""
        current = self.PreviewAlleleComboBox.currentData()
        self.PreviewAlleleComboBox.blockSignals(True)
        self.PreviewAlleleComboBox.clear()
        for allele_id, label in rows:
            self.PreviewAlleleComboBox.addItem(label, allele_id)
        if self.PreviewAlleleComboBox.count():
            restore_index = next((i for i in range(self.PreviewAlleleComboBox.count())
                                  if self.PreviewAlleleComboBox.itemData(i) == current), 0)
            self.PreviewAlleleComboBox.setCurrentIndex(restore_index)
        self.PreviewAlleleComboBox.blockSignals(False)

    def current_preview_allele_id(self):
        return self.PreviewAlleleComboBox.currentData()

    # -- fit parameters --

    @staticmethod
    def _read_channel_params(boxes):
        return {key: (widget.isChecked() if isinstance(widget, QtWidgets.QCheckBox) else widget.value())
                for key, widget in boxes.items()}

    def params(self):
        """
        {'spad':, 'z_window':, 'max_fiducial_drift':, 'fiducial': {...},
        'readout': {...}} -- the cross-mode keys apply identically to both
        channels. 'fiducial' holds its own independently-tunable
        peak_bound/max_sigma/max_uncert/min_hb_ratio/min_ah_ratio only (no
        mixture mode); 'readout' holds those same five plus min_sep/
        multi_mode, straight from localization.build_chromatin_trace_
        allele's own fiducial_params/readout_params shape (multi_mode here
        maps to that function's own use_mixture key -- MainWindow's own
        caller renames it, matching Localize3DDisplayer's own params()
        convention).
        """
        return {'spad': self.SpadSpinBox.value(),
                'z_window': self.ZWindowSpinBox.value(),
                'max_fiducial_drift': self.MaxFiducialDriftSpinBox.value(),
                'max_fiducial_drift_z': self.MaxFiducialDriftZSpinBox.value(),
                'z_boundary_trim': self.ZBoundaryTrimSpinBox.value(),
                'fiducial': self._read_channel_params(self.FiducialSpinBoxes),
                'readout': self._read_channel_params(self.ReadoutSpinBoxes)}

    def reset_defaults(self):
        self.SpadSpinBox.setValue(CROSS_MODE_DEFAULTS['spad'])
        self.ZWindowSpinBox.setValue(CROSS_MODE_DEFAULTS['z_window'])
        self.ZBoundaryTrimSpinBox.setValue(CROSS_MODE_DEFAULTS['z_boundary_trim'])
        self.MaxFiducialDriftSpinBox.setValue(CROSS_MODE_DEFAULTS['max_fiducial_drift'])
        for column_name, boxes in (('fiducial', self.FiducialSpinBoxes), ('readout', self.ReadoutSpinBoxes)):
            for key, widget in boxes.items():
                default = DEFAULT_PARAMS[column_name][key]
                if isinstance(widget, QtWidgets.QCheckBox):
                    widget.setChecked(default)
                else:
                    widget.setValue(default)
