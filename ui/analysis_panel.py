from PyQt5 import QtWidgets, QtCore

# The Analysis tab is a THIN SHELL over codelab_pipeline/analysis -- the
# toolbox is headless by contract, the app USES it and adds nothing to
# it. This panel only gathers widget values into toolbox calls and shows
# the returned figures; every computation is reproducible in a notebook
# from the sidecar JSON the saver writes.
#
# Two composition axes, mirrored in the layout per the founding spec:
#   CONDITIONS (section 2) narrow the cell set -- an ordered AND list,
#   with the sequential survivor counts and the per-celltype gated/total
#   always on screen, because a gated figure without its population is
#   not a result.
#   FLAGS (section 3) multiply the figures -- celltype decomposition
#   splits every view into per-celltype panels; it never re-gates.

# metric choices for ExpressionRange -- mask_median appears only when the
# population was built with mask intensity
EXPR_METRICS = ['n_spots', 'brightness_median', 'brightness_total',
                'mask_median']
NORM_MODES = ['none', 'by_modality', 'by_source']


class AnalysisPanelUI(object):
    def setupUi(self, Widget):
        layout = QtWidgets.QVBoxLayout(Widget)

        # -- 1. population -------------------------------------------------
        popGroup = QtWidgets.QGroupBox('1. Population (read from the store)')
        popLayout = QtWidgets.QFormLayout(popGroup)
        self.FovListLineEdit = QtWidgets.QLineEdit()
        self.FovListLineEdit.setPlaceholderText(
            'FOVs, e.g. 1-5,8  (empty = the Ingestion tab list)')
        popLayout.addRow('FOVs:', self.FovListLineEdit)
        # sources: one row per (modality, hybe, channel) the user checks;
        # MainWindow populates from the layouts' hybe records
        self.SourceListWidget = QtWidgets.QListWidget()
        self.SourceListWidget.setMaximumHeight(120)
        self.SourceListWidget.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)
        popLayout.addRow('Expression/spot sources\n(check to include):',
                         self.SourceListWidget)
        srcBtnRow = QtWidgets.QHBoxLayout()
        self.CheckSelectedSourcesPushButton = QtWidgets.QPushButton(
            'Check Selected')
        srcBtnRow.addWidget(self.CheckSelectedSourcesPushButton)
        self.UncheckSelectedSourcesPushButton = QtWidgets.QPushButton(
            'Uncheck Selected')
        srcBtnRow.addWidget(self.UncheckSelectedSourcesPushButton)
        self.CheckSpotSourcesPushButton = QtWidgets.QPushButton(
            'Check All With Spots')
        self.CheckSpotSourcesPushButton.setToolTip(
            'Check every (modality | hybe | channel) that has localized '
            'spots in the store -- the simple default start.')
        srcBtnRow.addWidget(self.CheckSpotSourcesPushButton)
        popLayout.addRow('', srcBtnRow)
        self.MaskIntensityCheckBox = QtWidgets.QCheckBox(
            'mask-based intensity (median MIP over each cell mask; slower)')
        popLayout.addRow('', self.MaskIntensityCheckBox)
        self.OverwriteCacheCheckBox = QtWidgets.QCheckBox(
            'overwrite cached cell attributes (recompute already-built '
            'sources; use after re-detection or re-alignment)')
        popLayout.addRow('', self.OverwriteCacheCheckBox)
        self.BuildPopulationPushButton = QtWidgets.QPushButton(
            'Build / Refresh Population')
        popLayout.addRow(self.BuildPopulationPushButton)
        self.PopulationStatusLabel = QtWidgets.QLabel('no population built')
        self.PopulationStatusLabel.setWordWrap(True)
        popLayout.addRow(self.PopulationStatusLabel)
        layout.addWidget(popGroup)

        # -- 1b. polymer QC ------------------------------------------------
        # A VISIBLE stage, not a hidden default: thresholds are derived
        # from the data (quantiles, the ORCA way), shown in EDITABLE
        # fields the user can override, previewed as histograms with the
        # threshold lines drawn, and applied explicitly. Everything
        # downstream (gates, views) then runs on the QC-filtered alleles,
        # and the sidecar JSON records the thresholds used.
        qcGroup = QtWidgets.QGroupBox(
            '1b. Polymer QC (derive, inspect, adjust, apply)')
        qcLayout = QtWidgets.QVBoxLayout(qcGroup)
        qcForm = QtWidgets.QGridLayout()
        self.QcFields = {}
        for col, (key, label) in enumerate((
                ('min_brightness', 'min amp'),
                ('max_brightness', 'max amp'),
                ('max_jump_um', 'max jump (um)'),
                ('max_dist_um', 'max median dist (um)'))):
            qcForm.addWidget(QtWidgets.QLabel(label), 0, col)
            w = QtWidgets.QLineEdit()
            w.setPlaceholderText('derive first')
            self.QcFields[key] = w
            qcForm.addWidget(w, 1, col)
        qcForm.addWidget(QtWidgets.QLabel('min traced bins'), 0, 4)
        self.QcMinTracedSpinBox = QtWidgets.QSpinBox()
        self.QcMinTracedSpinBox.setRange(2, 1000)
        self.QcMinTracedSpinBox.setValue(2)
        qcForm.addWidget(self.QcMinTracedSpinBox, 1, 4)
        qcLayout.addLayout(qcForm)
        qcRow = QtWidgets.QHBoxLayout()
        self.DeriveQcPushButton = QtWidgets.QPushButton(
            'Derive Thresholds (quantiles of this population)')
        qcRow.addWidget(self.DeriveQcPushButton)
        self.PreviewQcPushButton = QtWidgets.QPushButton(
            'Preview QC (histograms + efficacy/completeness)')
        qcRow.addWidget(self.PreviewQcPushButton)
        self.ApplyQcCheckBox = QtWidgets.QCheckBox(
            'apply QC to all views and gates')
        self.ApplyQcCheckBox.setChecked(True)
        qcRow.addWidget(self.ApplyQcCheckBox)
        qcLayout.addLayout(qcRow)
        self.QcStatusLabel = QtWidgets.QLabel('QC not derived')
        self.QcStatusLabel.setWordWrap(True)
        self.QcStatusLabel.setStyleSheet('font-family: monospace')
        qcLayout.addWidget(self.QcStatusLabel)
        layout.addWidget(qcGroup)

        # -- 2. conditions -------------------------------------------------
        condGroup = QtWidgets.QGroupBox(
            '2. Conditions (each ANDs in; narrows the cell set)')
        condLayout = QtWidgets.QVBoxLayout(condGroup)
        form = QtWidgets.QFormLayout()
        self.PredicateKindComboBox = QtWidgets.QComboBox()
        self.PredicateKindComboBox.addItems(
            ['ExpressionRange', 'PairDistanceRange', 'BarcodePresence',
             'CompletenessRange', 'CelltypeIn', 'FovIn', 'AlleleCount'])
        form.addRow('Kind:', self.PredicateKindComboBox)
        self.SourceAComboBox = QtWidgets.QComboBox()
        form.addRow('Source (A):', self.SourceAComboBox)
        self.SourceBComboBox = QtWidgets.QComboBox()
        form.addRow('Source B (pair distance):', self.SourceBComboBox)
        self.MetricComboBox = QtWidgets.QComboBox()
        self.MetricComboBox.addItems(EXPR_METRICS)
        form.addRow('Metric:', self.MetricComboBox)
        self.NormalizeComboBox = QtWidgets.QComboBox()
        self.NormalizeComboBox.addItems(NORM_MODES)
        form.addRow('Normalize:', self.NormalizeComboBox)
        self.CollapseComboBox = QtWidgets.QComboBox()
        self.CollapseComboBox.addItems(['median', 'min'])
        form.addRow('Distance collapse:', self.CollapseComboBox)
        self.LoLineEdit = QtWidgets.QLineEdit()
        self.LoLineEdit.setPlaceholderText('min (empty = open)')
        self.HiLineEdit = QtWidgets.QLineEdit()
        self.HiLineEdit.setPlaceholderText('max (empty = open)')
        rng = QtWidgets.QHBoxLayout()
        rng.addWidget(self.LoLineEdit)
        rng.addWidget(self.HiLineEdit)
        form.addRow('Range:', rng)
        self.ValuesLineEdit = QtWidgets.QLineEdit()
        self.ValuesLineEdit.setPlaceholderText(
            'celltypes / FOVs / barcode hybes, comma-separated')
        form.addRow('Values:', self.ValuesLineEdit)
        self.AbsentCheckBox = QtWidgets.QCheckBox(
            'ABSENT (barcodes must be missing from the allele)')
        form.addRow('', self.AbsentCheckBox)
        condLayout.addLayout(form)
        row = QtWidgets.QHBoxLayout()
        self.PreviewHistogramPushButton = QtWidgets.QPushButton(
            'Preview Histogram (shows the range on the distribution)')
        row.addWidget(self.PreviewHistogramPushButton)
        self.AddConditionPushButton = QtWidgets.QPushButton('Add Condition')
        row.addWidget(self.AddConditionPushButton)
        self.NewOrGroupPushButton = QtWidgets.QPushButton('New OR Group')
        self.NewOrGroupPushButton.setToolTip(
            'Conditions after this divider form a new AND-clause; clauses '
            'OR together: (A AND B) OR (C AND D).')
        row.addWidget(self.NewOrGroupPushButton)
        condLayout.addLayout(row)
        self.ConditionListWidget = QtWidgets.QListWidget()
        self.ConditionListWidget.setMaximumHeight(90)
        condLayout.addWidget(self.ConditionListWidget)
        row2 = QtWidgets.QHBoxLayout()
        self.RemoveConditionPushButton = QtWidgets.QPushButton('Remove Selected')
        row2.addWidget(self.RemoveConditionPushButton)
        self.ClearConditionsPushButton = QtWidgets.QPushButton('Clear All')
        row2.addWidget(self.ClearConditionsPushButton)
        condLayout.addLayout(row2)
        # the provenance block: sequential survivors + per-celltype counts
        self.GateSummaryLabel = QtWidgets.QLabel('gate: all cells')
        self.GateSummaryLabel.setWordWrap(True)
        self.GateSummaryLabel.setStyleSheet('font-family: monospace')
        condLayout.addWidget(self.GateSummaryLabel)
        layout.addWidget(condGroup)

        # -- 3. views + flags ----------------------------------------------
        viewGroup = QtWidgets.QGroupBox(
            '3. Views (flags multiply figures; they never re-gate)')
        viewLayout = QtWidgets.QVBoxLayout(viewGroup)
        flagRow = QtWidgets.QHBoxLayout()
        self.CelltypeDecomposeCheckBox = QtWidgets.QCheckBox(
            'decompose by celltype')
        self.CelltypeDecomposeCheckBox.setChecked(True)
        flagRow.addWidget(self.CelltypeDecomposeCheckBox)
        flagRow.addWidget(QtWidgets.QLabel('min alleles per map pixel:'))
        self.MinNSpinBox = QtWidgets.QSpinBox()
        self.MinNSpinBox.setRange(1, 10000)
        self.MinNSpinBox.setValue(5)
        flagRow.addWidget(self.MinNSpinBox)
        flagRow.addWidget(QtWidgets.QLabel('allele gate mode:'))
        self.AlleleModeComboBox = QtWidgets.QComboBox()
        self.AlleleModeComboBox.addItems(
            ['All (pool gated cells)', 'Presence vs Absence',
             'Full decompose (3 groups)'])
        flagRow.addWidget(self.AlleleModeComboBox)
        self.ShowFovMapsCheckBox = QtWidgets.QCheckBox(
            'FOV consistency: show per-FOV maps')
        flagRow.addWidget(self.ShowFovMapsCheckBox)
        flagRow.addStretch(1)
        viewLayout.addLayout(flagRow)
        # the views' OWN inputs, per explicit decision: expression and
        # distance histograms are final-layer callers like the ensemble
        # map; they read the GATED cells but never the condition form.
        viewForm = QtWidgets.QFormLayout()
        self.ViewExprSourceComboBox = QtWidgets.QComboBox()
        viewForm.addRow('Expression source:', self.ViewExprSourceComboBox)
        self.ViewExprMetricComboBox = QtWidgets.QComboBox()
        self.ViewExprMetricComboBox.addItems(EXPR_METRICS)
        viewForm.addRow('Expression metric:', self.ViewExprMetricComboBox)
        self.ViewDistSourceAComboBox = QtWidgets.QComboBox()
        viewForm.addRow('Distance source A:', self.ViewDistSourceAComboBox)
        self.ViewDistSourceBComboBox = QtWidgets.QComboBox()
        viewForm.addRow('Distance source B:', self.ViewDistSourceBComboBox)
        self.ViewDistCollapseComboBox = QtWidgets.QComboBox()
        self.ViewDistCollapseComboBox.addItems(['all', 'median', 'min'])
        viewForm.addRow('Distance collapse:', self.ViewDistCollapseComboBox)
        viewLayout.addLayout(viewForm)
        grid = QtWidgets.QGridLayout()
        self.EnsembleMapPushButton = QtWidgets.QPushButton('Ensemble Distance Map')
        self.FovConsistencyPushButton = QtWidgets.QPushButton('FOV Consistency (SCC + MSD test)')
        self.AlleleDifferencePushButton = QtWidgets.QPushButton('Allele Differences')
        self.ExpressionHistPushButton = QtWidgets.QPushButton('Expression Histogram (source A)')
        self.BrightnessVsCountPushButton = QtWidgets.QPushButton('Brightness vs Count (source A)')
        self.DistanceHistPushButton = QtWidgets.QPushButton('Distance Histogram (A vs B)')
        self.RepeatToeQcPushButton = QtWidgets.QPushButton('Repeat / Toe QC')
        for i, b in enumerate((self.EnsembleMapPushButton,
                               self.FovConsistencyPushButton,
                               self.AlleleDifferencePushButton,
                               self.ExpressionHistPushButton,
                               self.BrightnessVsCountPushButton,
                               self.DistanceHistPushButton,
                               self.RepeatToeQcPushButton)):
            grid.addWidget(b, i // 2, i % 2)
        viewLayout.addLayout(grid)
        layout.addWidget(viewGroup)
        layout.addStretch(1)

        self.PredicateKindComboBox.currentTextChanged.connect(
            self._sync_kind_fields)
        self._sync_kind_fields(self.PredicateKindComboBox.currentText())

    # -- helpers the wiring uses ------------------------------------------
    def _sync_kind_fields(self, kind):
        is_expr = kind == 'ExpressionRange'
        is_pair = kind == 'PairDistanceRange'
        is_list = kind in ('CelltypeIn', 'FovIn')
        is_pres = kind == 'BarcodePresence'
        is_allele = kind in ('AlleleCount', 'CompletenessRange')
        self.SourceAComboBox.setEnabled(is_expr or is_pair)
        self.SourceBComboBox.setEnabled(is_pair)
        self.MetricComboBox.setEnabled(is_expr)
        self.NormalizeComboBox.setEnabled(is_expr)
        self.CollapseComboBox.setEnabled(is_pair)
        self.LoLineEdit.setEnabled(is_expr or is_pair or is_allele)
        self.HiLineEdit.setEnabled(is_expr or is_pair or is_allele)
        self.ValuesLineEdit.setEnabled(is_list or is_pres)
        self.AbsentCheckBox.setEnabled(is_pres)
        self.PreviewHistogramPushButton.setEnabled(is_expr or is_pair)

    def checked_sources(self):
        """[(modality, hybe, channel)] the user checked in section 1."""
        out = []
        for i in range(self.SourceListWidget.count()):
            item = self.SourceListWidget.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                out.append(tuple(item.data(QtCore.Qt.UserRole)))
        return out

    def combo_source(self, combo):
        d = combo.currentData()
        return tuple(d) if d else None

    def range_values(self):
        def parse(w):
            t = w.text().strip()
            return float(t) if t else None
        return parse(self.LoLineEdit), parse(self.HiLineEdit)

    def qc_threshold_values(self):
        """The QC fields as a thresholds dict, or None when not derived.
        Raises ValueError on unparseable text -- an edited threshold that
        does not parse must refuse, not silently revert."""
        vals = {}
        for key, w in self.QcFields.items():
            t = w.text().strip()
            if not t:
                return None
            try:
                vals[key] = float(t)
            except ValueError:
                raise ValueError(f'QC threshold {key} = {t!r} is not a number')
        return vals

    def set_qc_thresholds(self, thresholds):
        for key, w in self.QcFields.items():
            w.setText(f'{thresholds[key]:.4g}')

    def predicate_dict(self):
        """Harvest the form into a gate.Predicate dict, or raise ValueError
        with a human-readable message."""
        kind = self.PredicateKindComboBox.currentText()
        lo, hi = self.range_values()
        if kind == 'ExpressionRange':
            src = self.combo_source(self.SourceAComboBox)
            if src is None:
                raise ValueError('pick a source (build the population with '
                                 'at least one checked source first)')
            norm = self.NormalizeComboBox.currentText()
            normalize = None
            if norm == 'by_total_count':
                normalize = ['by_total_count']
            elif norm == 'by_source':
                normalize = ['by_source', list(src)]
            return {'kind': 'expression_range', 'source': list(src),
                    'metric': self.MetricComboBox.currentText(),
                    'lo': lo, 'hi': hi, 'normalize': normalize}
        if kind == 'PairDistanceRange':
            a = self.combo_source(self.SourceAComboBox)
            b = self.combo_source(self.SourceBComboBox)
            if a is None or b is None:
                raise ValueError('pick both sources')
            return {'kind': 'pair_distance_range', 'source_a': list(a),
                    'source_b': list(b), 'lo': lo, 'hi': hi,
                    'collapse': self.CollapseComboBox.currentText()}
        if kind == 'BarcodePresence':
            hybes = [v.strip() for v in self.ValuesLineEdit.text().split(',')
                     if v.strip()]
            if not hybes:
                raise ValueError('list the barcode hybes, comma-separated')
            return {'kind': 'barcode_presence', 'hybes': hybes,
                    'absent': self.AbsentCheckBox.isChecked()}
        if kind == 'CelltypeIn':
            names = [v.strip() for v in self.ValuesLineEdit.text().split(',')
                     if v.strip()]
            if not names:
                raise ValueError('list the celltypes, comma-separated')
            return {'kind': 'celltype_in', 'names': names}
        if kind == 'FovIn':
            try:
                fovs = [int(v) for v in self.ValuesLineEdit.text().split(',')
                        if v.strip()]
            except ValueError:
                raise ValueError('FOVs must be integers, comma-separated')
            if not fovs:
                raise ValueError('list the FOVs')
            return {'kind': 'fov_in', 'fovs': fovs}
        if kind == 'CompletenessRange':
            return {'kind': 'completeness_range',
                    'lo': int(lo) if lo is not None else None,
                    'hi': int(hi) if hi is not None else None}
        if kind == 'AlleleCount':
            return {'kind': 'allele_count',
                    'lo': int(lo) if lo is not None else 1,
                    'hi': int(hi) if hi is not None else None,
                    'min_bins': 2}
        raise ValueError(f'unknown kind {kind}')
