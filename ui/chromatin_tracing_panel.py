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
# The two engines. v1 stays the reference implementation -- it is a direct
# port of ChrTracer3's FitPsf3D and must remain runnable so any v2 claim
# can be checked against it rather than asserted.
ENGINE_V1 = 'v1 (ChrTracer3 port)'
ENGINE_V2 = 'v2'

# Voxel size, in micrometres. An INPUT with a default, not a constant:
# v2 reasons in physical length, and a lateral pixel (0.208 um) and an
# axial plane (0.2 um) are different lengths that happen to be close on
# this microscope. Anything that assumes a fixed ratio between them is
# wrong somewhere else.
VOXEL_DEFAULTS = {'voxel_xy_um': 0.208, 'voxel_z_um': 0.2}

# 'universal-default' is the mean of the converged per-experiment readout
# calibrations in <repo>/psf. Justified because the between-experiment
# spread (28 nm over 4 experiments spanning 64x in genomic scope) is
# SMALLER than the ~40 nm a single experiment moves when its own
# calibration crops are reselected.
DEFAULT_READOUT_PSF = 'universal-default'

CROSS_MODE_DEFAULTS = {'spad': 8, 'z_window': 15, 'max_fiducial_drift': 5.0,
                       'max_fiducial_drift_z': 10.0, 'z_boundary_trim': 10}
SHARED_FIT_DEFAULTS = {'peak_bound': 2.0, 'max_sigma': 2.5, 'max_uncert': 2.0, 'min_ah_ratio': 0.25}
READOUT_ONLY_FIT_DEFAULTS = {'min_sep': 3.0, 'multi_mode': False}
DEFAULT_PARAMS = {**CROSS_MODE_DEFAULTS, **VOXEL_DEFAULTS,
                  'engine': ENGINE_V1,
                  'readout_psf': DEFAULT_READOUT_PSF,
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

        # ADD, not refresh: the transient container accumulates across
        # clicks now, and Remove Selected is how a mistake is undone.
        self.BuildAllelesPushButton = QtWidgets.QPushButton('Add Alleles from Selected Spots')
        allelesLayout.addWidget(self.BuildAllelesPushButton)

        self.AlleleCountLabel = QtWidgets.QLabel('0 allele(s) in this FOV.')
        allelesLayout.addWidget(self.AlleleCountLabel)

        # Selectable and removable. This is the ONLY allele selector --
        # it drives Remove, Save and the single-allele preview fit alike.
        # A second, independent selector for the same job is the exact bug
        # class this panel's own docstring already records (the Alleles
        # section used to inherit Spot Localization's comboboxes).
        self.AlleleListWidget = QtWidgets.QListWidget()
        self.AlleleListWidget.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)
        self.AlleleListWidget.setMaximumHeight(140)
        allelesLayout.addWidget(self.AlleleListWidget)

        # Transient/permanent controls, same shape as the cell panel's.
        alleleButtonsRow = QtWidgets.QWidget()
        alleleButtons = QtWidgets.QHBoxLayout(alleleButtonsRow)
        alleleButtons.setContentsMargins(0, 0, 0, 0)
        self.RemoveAllelesPushButton = QtWidgets.QPushButton('Remove Selected')
        self.RemoveAllelesPushButton.setToolTip(
            'Drops the selected alleles from this session only. '
            'The store is untouched until you press Save.')
        self.SaveAllelesPushButton = QtWidgets.QPushButton('Save')
        self.SaveAllelesPushButton.setToolTip(
            'Writes this FOV staged alleles to the store, REPLACING the FOV. '
            'Alleles removed since the last save are deleted from disk.')
        self.RevertAllelesPushButton = QtWidgets.QPushButton('Revert')
        self.RevertAllelesPushButton.setToolTip(
            'Discards staged edits and restores this FOV from the store.')
        for b in (self.RemoveAllelesPushButton, self.SaveAllelesPushButton,
                  self.RevertAllelesPushButton):
            alleleButtons.addWidget(b)
        alleleButtons.addStretch(1)
        allelesLayout.addWidget(alleleButtonsRow)
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

        # -- engine, voxel size, and the readout PSF --
        #
        # Voxel size is an INPUT, not a constant. v2 works in micrometres
        # throughout, so a lateral pixel and an axial plane are separate
        # physical lengths instead of being silently equated: v1 wrote its
        # axial gate as 2x the lateral one in pixels, assuming a plane is
        # twice a pixel, when here a plane is 0.2 um against a pixel's
        # 0.208 -- they differ by 4%, not 2x.
        engineForm = QtWidgets.QFormLayout()
        paramsOuter.addLayout(engineForm)

        # Filled by MainWindow from the Ingestion panel before params()
        # is read. A default here only covers a panel constructed alone,
        # as the tests do.
        self._voxel_um = (VOXEL_DEFAULTS['voxel_xy_um'],
                          VOXEL_DEFAULTS['voxel_xy_um'],
                          VOXEL_DEFAULTS['voxel_z_um'])

        self.EngineComboBox = QtWidgets.QComboBox()
        # itemData, not display text. The config round-trips on this, so
        # renaming a label ("v2" -> "v2 (calibrated PSF)") must not turn
        # every saved v2 config into a v1 run silently carrying v2 numbers.
        self.EngineComboBox.addItem(ENGINE_V1, 'v1')
        self.EngineComboBox.addItem(ENGINE_V2, 'v2')
        self.EngineComboBox.setProperty('config_uses_item_data', True)
        self.EngineComboBox.setCurrentText(DEFAULT_PARAMS['engine'])
        self.EngineComboBox.setToolTip(
            'v1 is the direct port of ChrTracer3 FitPsf3D and is kept as the\n'
            'reference implementation. v2 fits in micrometres with a boxed\n'
            'fit domain, a linear background, and a calibrated readout PSF.')
        engineForm.addRow('Fit engine:', self.EngineComboBox)

        # The readout PSF is CHOSEN, not derived here. Entries come from
        # the tracked library in <repo>/psf, which accumulates every
        # calibration ever run -- so a new experiment can start from the
        # universal default (4 experiments over 64x in genomic scope agreed
        # to within 28 nm) and only re-fit if it has reason to.
        #
        # There is deliberately no fiducial entry: a Gaussian fitted to a
        # fiducial returns the FIT WINDOW rather than a width (sigma ~
        # r^0.5, no plateau), so a stored fiducial PSF would be a number
        # that does not exist.
        psfRow = QtWidgets.QHBoxLayout()
        self.ReadoutPsfComboBox = QtWidgets.QComboBox()
        self.ReadoutPsfComboBox.setMinimumWidth(240)
        # Round-trip through the config on itemData (the stable label), not
        # on the decorated display text. See MainWindow._widget_value.
        self.ReadoutPsfComboBox.setProperty('config_uses_item_data', True)
        self.ReadoutPsfComboBox.setToolTip(
            'Readout PSF shape, from the tracked library in <repo>/psf.\n'
            'The chosen entry is COPIED into this experiment at\n'
            '<project>/analysis/psf.json when tracing runs, so a store\n'
            'stays reproducible after the library moves on.')
        psfRow.addWidget(self.ReadoutPsfComboBox, 1)
        self.FitReadoutPsfPushButton = QtWidgets.QPushButton('Fit Readout PSF...')
        self.FitReadoutPsfPushButton.setToolTip(
            'Calibrate a readout PSF from this experiment\'s own reference-hybe\n'
            'spots and add it to the library as a new entry. Never overwrites\n'
            'an existing one -- every calibration is kept, so the library is a\n'
            'history rather than a current value.')
        psfRow.addWidget(self.FitReadoutPsfPushButton)
        engineForm.addRow('Readout PSF (v2):', psfRow)

        # -- cross-mode: shared by BOTH engines, so it sits OUTSIDE the
        # stack. v2 really does read all four: spad reaches
        # crop_for_localization, and both drift gates are applied in
        # tracing_v2.build_chromatin_trace_allele. z_window and
        # z_boundary_trim are v1-only and move onto the v1 page below.
        crossForm = QtWidgets.QFormLayout()
        paramsOuter.addLayout(crossForm)

        self.SpadSpinBox = QtWidgets.QSpinBox()
        self.SpadSpinBox.setRange(1, 100)
        self.SpadSpinBox.setValue(CROSS_MODE_DEFAULTS['spad'])
        crossForm.addRow('Crop half-width (px):', self.SpadSpinBox)

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
        # -- the stack: one page per engine ------------------------------
        #
        # Greying v1's rows under v2 was honest but useless -- it hid the
        # wrong parameters without showing the right ones, and v2's actual
        # gate (occupancy) had no widget at all. Each engine now shows its
        # OWN parameters, in its own units.
        self.FitParamsStackedWidget = QtWidgets.QStackedWidget()

        v1Page = QtWidgets.QWidget()
        v1Outer = QtWidgets.QVBoxLayout(v1Page)
        v1Outer.setContentsMargins(0, 0, 0, 0)
        v1Form = QtWidgets.QFormLayout()
        v1Outer.addLayout(v1Form)

        self.ZWindowSpinBox = QtWidgets.QSpinBox()
        self.ZWindowSpinBox.setRange(1, 200)
        self.ZWindowSpinBox.setValue(CROSS_MODE_DEFAULTS['z_window'])
        v1Form.addRow('Z search window (+/-px, mixture mode only):', self.ZWindowSpinBox)

        self.ZBoundaryTrimSpinBox = QtWidgets.QSpinBox()
        self.ZBoundaryTrimSpinBox.setRange(0, 100)
        self.ZBoundaryTrimSpinBox.setValue(CROSS_MODE_DEFAULTS['z_boundary_trim'])
        self.ZBoundaryTrimSpinBox.setSuffix(' planes')
        v1Form.addRow('Z boundary trim (each end):', self.ZBoundaryTrimSpinBox)

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
        v1Outer.addLayout(grid)
        self.FitParamsStackedWidget.addWidget(v1Page)          # index 0 = v1
        self.FitParamsStackedWidget.addWidget(self._build_v2_page(double_spin))
        paramsOuter.addWidget(self.FitParamsStackedWidget)

        self.ResetDefaultsPushButton = QtWidgets.QPushButton('Reset to Defaults')
        self.ResetDefaultsPushButton.clicked.connect(self.reset_defaults)
        paramsOuter.addWidget(self.ResetDefaultsPushButton)
        layout.addWidget(paramsGroup)

        # -- 4. preview one allele --
        previewGroup = QtWidgets.QGroupBox('4. Preview One Allele')
        previewLayout = QtWidgets.QVBoxLayout(previewGroup)
        # NO allele picker here. The allele listview in section 2 is the
        # single selector -- a second, independent one for the same job is
        # exactly what this class's docstring records going wrong before,
        # when the Alleles section silently inherited Spot Localization's
        # comboboxes.
        previewLayout.addWidget(QtWidgets.QLabel(
            'Select ONE allele in the list above, then:'))
        self.ViewCropPushButton = QtWidgets.QPushButton('View Crop (fiducial + readout grids)')
        previewLayout.addWidget(self.ViewCropPushButton)
        layout.addWidget(previewGroup)

        # -- 5. fit all fovs --
        fitAllGroup = QtWidgets.QGroupBox('5. Fit All FOVs')
        fitAllLayout = QtWidgets.QVBoxLayout(fitAllGroup)
        self.FitThisFovPushButton = QtWidgets.QPushButton('Fit This FOV')
        self.FitThisFovPushButton.setToolTip(
            'Traces only the FOV shown in the Alleles section. Same append '
            'and overwrite semantics as Fit All FOVs -- useful while '
            'ingestion is still running, since a FOV whose hybes have not '
            'all landed is skipped in append mode.')
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
            # adj_coordinate is (y, x, z) -- rasterized order, models/spot.py:31.
            # Unpacking it as x, y, z printed the two transposed, so a spot at
            # y=300, x=700 read as "(700.0, 300.0, ...)" and anyone matching a
            # listed spot against a coordinate seen elsewhere got the mirror.
            # Display-only, unlike the two transposes found in the fit path,
            # but the same mistake and worth not leaving in place.
            y, x, z = d['adj_coordinate']
            item = QtWidgets.QListWidgetItem(
                f'Spot {global_index} | {cell_tag} | y={y:.1f}, x={x:.1f}, z={z:.1f}')
            item.setData(QtCore.Qt.UserRole, d)
            self.SpotListWidget.addItem(item)

    def selected_spot_dicts(self):
        return [item.data(QtCore.Qt.UserRole) for item in self.SpotListWidget.selectedItems()]

    def populate_allele_list(self, rows):
        """rows: [(allele_id, label), ...] -- caller resolved the text.

        The id is carried as itemData, not parsed back out of the label:
        every consumer (Remove, the preview fit) keys on it, and a label
        is display text that is meant to change. Selection is preserved
        across a repopulate BY ID, so refreshing after a Build or a fit
        does not silently move which allele is selected.
        """
        keep = set(self.selected_allele_ids())
        self.AlleleListWidget.blockSignals(True)
        self.AlleleListWidget.clear()
        for allele_id, label in rows:
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.UserRole, int(allele_id))
            self.AlleleListWidget.addItem(item)
            if int(allele_id) in keep:
                item.setSelected(True)
        self.AlleleListWidget.blockSignals(False)
        self.AlleleCountLabel.setText(f'{len(rows)} allele(s) in this FOV.')

    def selected_allele_ids(self):
        """Ids of the selected rows, from itemData -- never from the text."""
        return [int(i.data(QtCore.Qt.UserRole))
                for i in self.AlleleListWidget.selectedItems()
                if i.data(QtCore.Qt.UserRole) is not None]

    def current_allele_id(self):
        """The one allele a single-allele action works on, or None."""
        ids = self.selected_allele_ids()
        return ids[0] if len(ids) == 1 else None

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
                'engine': (self.EngineComboBox.currentData()
                           or self.EngineComboBox.currentText()),
                'engine_label': self.EngineComboBox.currentText(),
                'v2': self.v2_params(),
                # NOT read here any more: voxel size is experiment-level
                # and lives on the Ingestion panel. MainWindow injects it,
                # so there is exactly one widget pair for it in the app.
                'voxel_um': self._voxel_um,
                'readout_psf': self.ReadoutPsfComboBox.currentData()
                               or self.ReadoutPsfComboBox.currentText(),
                'fiducial': self._read_channel_params(self.FiducialSpinBoxes),
                'readout': self._read_channel_params(self.ReadoutSpinBoxes)}

    def _build_v2_page(self, double_spin):
        """v2's own parameters, in v2's own units.

        EVERY DEFAULT IS READ FROM THE ENGINE, never typed here. The values
        in tracing_v2 are measurements -- occupancy 0.25/0.40, the 1.0/3.0
        um fit domain the whole 43-68% result was taken at -- and a literal
        retyped in the UI that drifts by one digit would replace the
        validated configuration with a plausible-looking different one,
        silently. Importing the constants makes that impossible rather
        than unlikely, which is the same reason DEFAULT_PARAMS exists for
        the v1 page.

        The page separates TUNABLE gates from MEASURED constants. The
        constants are shown read-only rather than hidden: a person needs
        to know the fit domain is 1.0 um to understand why a 1.04 um
        position bound can rail, and hiding it is how the v1 gates ended
        up inherited without provenance.
        """
        from codelab_pipeline.localization import tracing_v2 as V2

        page = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)

        self.V2EngineStatusLabel = QtWidgets.QLabel('')
        self.V2EngineStatusLabel.setWordWrap(True)
        self.V2EngineStatusLabel.setStyleSheet('color: #555;')
        outer.addWidget(self.V2EngineStatusLabel)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel('Fiducial'), 0, 1)
        grid.addWidget(QtWidgets.QLabel('Readout'), 0, 2)
        self.V2FiducialBoxes, self.V2ReadoutBoxes = {}, {}

        def occ_box(default):
            w = double_spin(default, -0.5, 1.0, step=0.05, decimals=2)
            w.setToolTip(
                'Intensity at the fitted centroid over intensity at the argmax,\n'
                'both above a LOCAL PLANE background. 1.0 = the fit is on the\n'
                'emitter, <= 0 = it is in background.\n\n'
                'The tunable gate, and the best-behaved: it degrades smoothly\n'
                'instead of falling off a cliff. Measured v1 -> v2: 0.354 ->\n'
                '0.838 fiducial, 0.561 -> 0.677 readout. The fiducial default\n'
                'is looser on purpose -- an extended object spreads its peak.')
            return w

        def ci_box(default_nm):
            w = QtWidgets.QDoubleSpinBox()
            w.setDecimals(0)
            w.setRange(0.0, 2000.0)
            # step 1.0 at 0 decimals matters: test_config_roundtrip nudges
            # each spinbox by one singleStep, and a 0.1 step at 0 decimals
            # is a silent no-op that would exercise nothing while passing.
            w.setSingleStep(1.0)
            w.setValue(0.0 if default_nm is None else float(default_nm))
            w.setSpecialValueText('off (measured default)')
            w.setToolTip(
                'FULL 95% confidence interval in nanometres; 0 = gate off.\n\n'
                'Off by default deliberately. At v1\'s own coverage these reach\n'
                '61-79 nm against v1\'s 183 nm, but those thresholds were derived\n'
                'on ONE dataset, and a silently inherited constant is how the v1\n'
                'gates became wrong in the first place.\n\n'
                'FULL width, not half: the sweep that produced those numbers used\n'
                '2000 * max(ci_y_um, ci_x_um).')
            return w

        rows = [
            ('Min occupancy (fit on emitter; 1.0 = perfect):',
             'min_occupancy', occ_box, 'MinOccupancy'),
            ('Max lateral uncertainty (full 95% CI, nm; 0 = off):',
             'max_uncert_xy_nm', ci_box, 'MaxUncertXY'),
            ('Max axial uncertainty (full 95% CI, nm; 0 = off):',
             'max_uncert_z_nm', ci_box, 'MaxUncertZ'),
        ]
        for r, (label, key, factory, attr) in enumerate(rows, start=1):
            grid.addWidget(QtWidgets.QLabel(label), r, 0)
            for col, (name, gates, boxes) in enumerate(
                    (('Fiducial', V2.FIDUCIAL_GATES, self.V2FiducialBoxes),
                     ('Readout', V2.READOUT_GATES, self.V2ReadoutBoxes)), start=1):
                w = factory(gates[key])
                setattr(self, f'V2{name}{attr}SpinBox', w)
                boxes[key] = w
                grid.addWidget(w, r, col)
        outer.addLayout(grid)

        qcRow = QtWidgets.QFormLayout()
        self.V2QcShiftCheckBox = QtWidgets.QCheckBox()
        self.V2QcShiftCheckBox.setChecked(bool(V2.V2Params().qc_shift))
        self.V2QcShiftCheckBox.setToolTip(
            'An independent estimate of each hybe\'s drift by image registration.\n'
            'QC ONLY -- it never changes a stored position.\n\n'
            'Measured 14.8% WORSE than the fit on the median of 305 replicate\n'
            'pairs, and closer on only 37% of them, so it is not the estimator.\n'
            'It is here because it has the better tail (p90 1.858 vs 1.945) and\n'
            'is independent: two estimates of one displacement that disagree is\n'
            'a strong outlier signal. Costs roughly 20 s per 48 alleles.')
        qcRow.addRow('Cross-check drift by image registration (QC only):',
                     self.V2QcShiftCheckBox)
        outer.addLayout(qcRow)

        # -- measured, not tunable ---------------------------------------
        frozen = QtWidgets.QGroupBox('Measured constants (not tunable)')
        fl = QtWidgets.QFormLayout(frozen)

        def frozen_row(label, text, tip):
            w = QtWidgets.QLabel(text)
            w.setStyleSheet('color: #555;')
            w.setToolTip(tip)
            fl.addRow(label, w)

        fr, rr = V2.FIDUCIAL_FIT_RADIUS_UM, V2.READOUT_FIT_RADIUS_UM
        frozen_row('Fit domain (half-extent, um):',
                   f'fiducial {fr[0]:g}, {fr[1]:g}, {fr[2]:g}   '
                   f'readout {rr[0]:g}, {rr[1]:g}, {rr[2]:g}',
                   'A BOX, not a pillar: occupancy 0.373 -> 0.806 and\n'
                   'blank-region fits 31% -> 4%. The SAME domain for both\n'
                   'channels because that is what was measured. It briefly read\n'
                   '(0.8, 0.8, 2.0) -- 1029 voxels instead of 2511, 41% of the\n'
                   'data every readout number was taken on.')
        frozen_row('Position bound, lateral / axial (um):',
                   f'{V2.FIDUCIAL_PEAK_BOUND_UM:g} (~5 px)  /  '
                   f'{V2.FIDUCIAL_PEAK_BOUND_Z_UM:g} (~10 planes)',
                   'Loose and SEPARATE. Tight bounds put 75-100% of fits on a\n'
                   'constraint -- which is why every dz in a fiducial overlay\n'
                   'used to print as a whole number.\n\n'
                   'Note the lateral bound (1.04 um) slightly exceeds the lateral\n'
                   'fit domain (1.0 um). Both are the measured values; a lateral\n'
                   'rail is fatal, so such fits are rejected rather than trusted.')
        frozen_row('Sigma bounds (um):',
                   f'min {V2.FIDUCIAL_MIN_SIGMA_UM:g}   '
                   f'max xy {V2.FIDUCIAL_MAX_SIGMA_XY_UM:g} / z '
                   f'{V2.FIDUCIAL_MAX_SIGMA_Z_UM:g}   '
                   '(readout sigma is FIXED by the calibrated PSF)',
                   'The ceilings every v2 measurement was taken with. Tightening\n'
                   'was tried twice (1.20, then 0.60) and only moved the number:\n'
                   'a single-crop fiducial fit pins sigma to whatever ceiling\n'
                   'exists -- 14 consecutive HoxA rounds returned exactly 600 nm.\n'
                   'The floor is psf.plausible\'s own diffraction limit, 70 nm.')
        frozen_row('At-bound rejection:',
                   'on -- fatal on position (y, x, z) only',
                   'Free, with no threshold to choose: 295/311 pairs at 0.218 um\n'
                   'against 311 at 0.294 ungated -- 95% of pairs kept for a 26%\n'
                   'better median.\n\n'
                   'Sigma railing is NOT fatal for a fiducial: its width has no\n'
                   'value to converge to, while its centroid stays on the emitter\n'
                   '(occupancy 0.44-0.69 on exactly those railed fits).')
        frozen_row('v1 gates v2 does not implement:',
                   'min peak/background, min amplitude/peak',
                   'min_hb_ratio is untunable -- 311 pairs at 1.0, 40 at 1.2,\n'
                   '~10 by 1.6, so a 0.1 change swings coverage by an order of\n'
                   'magnitude. min_ah_ratio is dominated by occupancy, which\n'
                   'measures the same intent properly.')
        outer.addWidget(frozen)
        return page

    def v2_params(self):
        """The v2 page's tunable values, shaped for V2Params.

        0 in a CI box means "off" -- the widget's special value -- and must
        become None, not 0.0. A 0 nm threshold would reject every fit ever
        made, which is the opposite of off and would look like the engine
        had broken.
        """
        def read(boxes):
            out = {}
            for key, w in boxes.items():
                v = w.value()
                out[key] = None if (key.endswith('_nm') and v <= 0) else v
            return out
        return {'fiducial': read(self.V2FiducialBoxes),
                'readout': read(self.V2ReadoutBoxes),
                'qc_shift': self.V2QcShiftCheckBox.isChecked()}

    def apply_engine_visibility(self):
        """Show the SELECTED engine's parameters.

        A page swap, not a grey-out. Each engine's widgets keep their own
        values while hidden, so switching v1 -> v2 -> v1 restores exactly
        what was there: a pixel peak-bound and a micrometre one are not
        the same number, and carrying values across would be a unit bug
        waiting to happen.
        """
        v2 = self.selected_engine_is_v2()
        self.FitParamsStackedWidget.setCurrentIndex(1 if v2 else 0)
        return v2

    def selected_engine_is_v2(self):
        """Match tracing_v2.is_v2 -- prefix, and on itemData when present."""
        data = self.EngineComboBox.currentData()
        name = data if data else self.EngineComboBox.currentText()
        return str(name or '').strip().lower().startswith('v2')

    def refresh_psf_entries(self, select=None):
        """Repopulate the PSF combo from the library on disk.

        The label is carried as itemData, not just as display text: the
        text shows provenance (sigma, source, whether it converged) and is
        meant to change as the library grows, while the label is what the
        config stores and must stay exactly what was written.
        """
        from codelab_pipeline.localization import psf_library as LIB
        want = select or (self.ReadoutPsfComboBox.currentData()
                          or DEFAULT_READOUT_PSF)
        self.ReadoutPsfComboBox.blockSignals(True)
        self.ReadoutPsfComboBox.clear()
        for e in LIB.list_entries():
            p = e.get('params', {})
            sxy = p.get('sigma_xy_um')
            conv = (e.get('converged') or {})
            mark = '' if conv.get('converged', True) else '  [not converged]'
            bits = [e['label']]
            if sxy:
                bits.append(f'{1000 * sxy:.0f} nm')
            bits.append(str(e.get('family', '')))
            self.ReadoutPsfComboBox.addItem('  --  '.join(bits) + mark,
                                            e['label'])
        # Matching on DATA, never on display text: findData compares
        # non-QVariant payloads by object identity in PyQt5, so this walks
        # the items instead (the same bug that broke the hybe combo).
        idx = -1
        for i in range(self.ReadoutPsfComboBox.count()):
            if self.ReadoutPsfComboBox.itemData(i) == want:
                idx = i
                break
        if idx >= 0:
            self.ReadoutPsfComboBox.setCurrentIndex(idx)
        elif want:
            # THE LABEL IS NOT IN THIS MACHINE'S LIBRARY. Leaving the combo
            # where it was meant the config named one PSF, the run used
            # another, the log confidently named the wrong one, and pressing
            # Save Config then overwrote the record of which calibration
            # actually produced the traces.
            #
            # <repo>/psf is git-tracked but a fresh calibration lands there
            # untracked, so "calibrated on machine A, opened on machine B"
            # is the normal way to reach this, not an exotic one.
            #
            # Show it, select it, and mark it. V2Params cannot resolve a
            # shape for it, so v2 falls back to free sigma and SAYS so
            # rather than silently substituting a different shape.
            self.ReadoutPsfComboBox.addItem(f'{want}  --  MISSING FROM LIBRARY', want)
            self.ReadoutPsfComboBox.setCurrentIndex(
                self.ReadoutPsfComboBox.count() - 1)
        self.ReadoutPsfComboBox.blockSignals(False)
        return self.ReadoutPsfComboBox.count()

    def reset_defaults(self):
        self.EngineComboBox.setCurrentText(DEFAULT_PARAMS['engine'])
        # BOTH pages reset, from the engine's own constants -- Reset means
        # "the documented configuration", and the hidden page is part of it.
        from codelab_pipeline.localization import tracing_v2 as V2
        for gates, boxes in ((V2.FIDUCIAL_GATES, self.V2FiducialBoxes),
                             (V2.READOUT_GATES, self.V2ReadoutBoxes)):
            for key, w in boxes.items():
                v = gates[key]
                w.setValue(0.0 if v is None else float(v))
        self.V2QcShiftCheckBox.setChecked(bool(V2.V2Params().qc_shift))
        self.apply_engine_visibility()
        self.refresh_psf_entries(select=DEFAULT_READOUT_PSF)
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
