"""The Pipeline tab: every step of a store's workflow with a policy,
the FOV x step status grid read from the store, and one Start.

Policies per step: 'existing' runs only the FOVs the store does not
hold yet (the app's append), 'redo' runs every FOV (overwrite), 'skip'
leaves the step out. Segmentation defaults to 'skip': it is the step
the user does by hand, and the runner never starts it unless it is
chosen here. Ingestion is never overwritten from here (a redo runs as
append). The FOV list is the Ingestion tab's, as for every batch button.

CLICKING A STEP opens its parameters (ui/step_params_dialog), generated
from the config map and written straight back to the tab they belong
to. The 'settings' column is a reminder of what that dialog holds, and
STEP_NOTES -- which used to be a column of prose wider than the table --
is what the dialog says about the step.
"""
from PyQt5 import QtCore, QtGui, QtWidgets

from codelab_pipeline.pipeline import status as S

#: (min_p_exist, pad) seeded from the batch module itself, so the panel
#: opens on what a run would have done anyway and nobody's first Save
#: Config silently changes the floor.
try:
    from codelab_pipeline.localization import batch as _batch
    _BATCH_DEFAULTS = (float(_batch.DEFAULT_MIN_P_EXIST), int(_batch.DEFAULT_PAD))
except Exception:                                           # noqa: BLE001
    _BATCH_DEFAULTS = (0.1, 10)

STEP_LABELS = {'ingestion': 'Ingestion', 'segmentation': 'Cell segmentation', 'fov_alignment': 'FOV alignment',
               'cross_modal': 'Cross-modal alignment', 'cell_alignment': 'Cell alignment', 'localization': 'Spot localization',
               'celltype': 'Celltype determination', 'cellcycle': 'Cell cycle', 'tracing': 'Chromatin tracing'}
STEP_NOTES = {'ingestion': 'append only from here (an overwrite clears stacks: do that on the Ingestion tab)',
              'segmentation': 'your own step: runs only when chosen here, FOV by FOV (run, save)',
              'fov_alignment': 'the Alignment tab\'s "all FOVs" batch, append = missing (FOV, hybe) pairs',
              'cross_modal': 'n/a with one modality',
              'cell_alignment': 'the slowest step; append = cells without matrices',
              'localization': 'headless v3 batch on the selected model; expected rounds = every tracing round + the localization round + cell-cycle genes + countable readouts',
              'celltype': 'FOV ranges or barcode config from the store',
              'cellcycle': 'build counts, then fit (redo, or no model) or place with the stored model',
              'tracing': 'build alleles for every FOV, then fit all (append = alleles without a trace)'}
COMBO_NAMES = {'ingestion': 'IngestionPolicyComboBox', 'segmentation': 'SegmentationPolicyComboBox',
               'fov_alignment': 'FovAlignmentPolicyComboBox', 'cross_modal': 'CrossModalPolicyComboBox',
               'cell_alignment': 'CellAlignmentPolicyComboBox', 'localization': 'LocalizationPolicyComboBox',
               'celltype': 'CelltypePolicyComboBox', 'cellcycle': 'CellcyclePolicyComboBox', 'tracing': 'TracingPolicyComboBox'}
DEFAULTS = dict(S.DEFAULT_POLICY, segmentation='skip')
STATE_COLOURS = {'done': QtGui.QColor(198, 239, 206), 'partial': QtGui.QColor(255, 235, 156),
                 'missing': QtGui.QColor(230, 230, 230), 'n/a': QtGui.QColor(245, 245, 245)}
STATE_TEXT = {'done': 'done', 'partial': 'partial', 'missing': '-', 'n/a': 'n/a'}


class PipelinePanelUI:
    def setupUi(self, widget):
        outer = QtWidgets.QVBoxLayout(widget)
        intro = QtWidgets.QLabel(
            'One run over the whole store, in pipeline order. '
            '<b>Click a step to set its parameters.</b>')
        intro.setWordWrap(True)
        outer.addWidget(intro)

        # -- 1. steps and policies ---------------------------------------------------
        box1 = QtWidgets.QGroupBox('1. Steps and policies')
        l1 = QtWidgets.QVBoxLayout(box1)
        self.StepsTable = QtWidgets.QTableWidget(len(S.STEPS), 4)
        self.StepsTable.setHorizontalHeaderLabels(['step', 'policy', 'to run', 'settings'])
        self.StepsTable.verticalHeader().setVisible(False)
        self.StepsTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.StepsTable.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.StepsTable.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.policy_combos = {}
        for i, step in enumerate(S.STEPS):
            item = QtWidgets.QTableWidgetItem(STEP_LABELS[step])
            item.setToolTip(STEP_NOTES[step])
            font = item.font()
            font.setUnderline(True)
            item.setFont(font)
            item.setForeground(QtGui.QBrush(QtGui.QColor(30, 70, 140)))
            self.StepsTable.setItem(i, 0, item)
            combo = QtWidgets.QComboBox()
            combo.addItems(list(S.POLICIES))
            combo.setCurrentText(DEFAULTS[step])
            setattr(self, COMBO_NAMES[step], combo)
            self.policy_combos[step] = combo
            self.StepsTable.setCellWidget(i, 1, combo)
            to_run = QtWidgets.QTableWidgetItem('')
            to_run.setTextAlignment(QtCore.Qt.AlignCenter)
            self.StepsTable.setItem(i, 2, to_run)
            settings = QtWidgets.QTableWidgetItem('')
            # 70 rather than 90: at this size subpixel antialiasing puts
            # a colour fringe on light grey text, which reads as a dull
            # red -- the thing that looked like a palette choice in the
            # old note column and was not one
            settings.setForeground(QtGui.QBrush(QtGui.QColor(70, 70, 70)))
            settings.setToolTip(STEP_NOTES[step])
            self.StepsTable.setItem(i, 3, settings)
        self.StepsTable.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        # the settings column is a stretched one holding left-aligned
        # text; a centred header over it floats in empty space
        self.StepsTable.horizontalHeaderItem(3).setTextAlignment(
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.StepsTable.resizeColumnsToContents()
        self.StepsTable.setMinimumHeight(30 * (len(S.STEPS) + 1) + 8)
        l1.addWidget(self.StepsTable)
        # THE PIPELINE'S LOCALIZATION RUN SETTINGS, config section
        # 'localization_run'. WIDGETS because _CONFIG_PARAM_MAP's entries
        # are widgets -- that is what makes them save and load like every
        # other parameter. Their editor is the step dialog, so they are
        # not shown on the tab itself.
        #
        # They exist because localization/batch.run read its p_exist
        # floor and its pad from module defaults that no run could state
        # or change: this experiment settled on p_exist 0.2 and the
        # pipeline went on localizing at batch.py's 0.5.
        self.LocalizationRoundsLineEdit = QtWidgets.QLineEdit()
        self.LocalizationRoundsLineEdit.setReadOnly(True)
        self.LocalizationMinPExistSpinBox = QtWidgets.QDoubleSpinBox()
        self.LocalizationMinPExistSpinBox.setRange(0.0, 1.0)
        self.LocalizationMinPExistSpinBox.setSingleStep(0.05)
        self.LocalizationMinPExistSpinBox.setDecimals(2)
        self.LocalizationMinPExistSpinBox.setValue(_BATCH_DEFAULTS[0])
        self.LocalizationMinPExistSpinBox.setToolTip(
            'A spot is written only if the model gives it at least this '
            'p_exist. The floor is applied AT WRITE TIME, so lowering it '
            'later means re-running the round -- which is why the default '
            'is a minimal floor rather than a review threshold. Gate for '
            'analysis afterwards instead; p, p1 and p_exist are all '
            'stored.')
        self.LocalizationChannelsLineEdit = QtWidgets.QLineEdit()
        self.LocalizationChannelsLineEdit.setPlaceholderText(
            'every non-fiducial channel')
        self.LocalizationChannelsLineEdit.setToolTip(
            'Readout channel(s) to localize the DERIVED rounds at, comma '
            'separated (e.g. 635 or 635,747). Empty means every '
            'non-fiducial channel the round carries, which is what this '
            'did before it could be stated. It governs only the rounds '
            'nobody named a channel for -- the countable readouts. The '
            'anchor round, the cell-cycle genes and the tracing rounds '
            'carry their own channel and are left as they say.')
        self.LocalizationPadSpinBox = QtWidgets.QSpinBox()
        self.LocalizationPadSpinBox.setRange(0, 64)
        self.LocalizationPadSpinBox.setValue(_BATCH_DEFAULTS[1])
        self.LocalizationPadSpinBox.setToolTip(
            'Half-width of the crop the model sees around each candidate.')
        for w in (self.LocalizationRoundsLineEdit,
                  self.LocalizationChannelsLineEdit,
                  self.LocalizationMinPExistSpinBox,
                  self.LocalizationPadSpinBox):
            w.setVisible(False)
            l1.addWidget(w)
        outer.addWidget(box1)

        # -- 2. status ----------------------------------------------------------------------
        box2 = QtWidgets.QGroupBox('2. What the store holds')
        l2 = QtWidgets.QVBoxLayout(box2)
        row = QtWidgets.QHBoxLayout()
        self.RefreshPushButton = QtWidgets.QPushButton('Refresh status')
        self.DeepCheckBox = QtWidgets.QCheckBox('open capsules (cells, alleles, cell cycle) -- slower, exact')
        self.DeepCheckBox.setChecked(True)
        self.FovListLabel = QtWidgets.QLabel('FOVs: (Ingestion tab list)')
        row.addWidget(self.RefreshPushButton)
        row.addWidget(self.DeepCheckBox)
        row.addStretch(1)
        row.addWidget(self.FovListLabel)
        l2.addLayout(row)
        self.StatusTable = QtWidgets.QTableWidget(0, len(S.STEPS) + 1)
        self.StatusTable.setHorizontalHeaderLabels(['FOV'] + [S.LABEL.get(s, s) for s in S.STEPS])
        self.StatusTable.verticalHeader().setVisible(False)
        self.StatusTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.StatusTable.setMinimumHeight(220)
        l2.addWidget(self.StatusTable)
        self.SummaryLabel = QtWidgets.QLabel('')
        self.SummaryLabel.setWordWrap(True)
        l2.addWidget(self.SummaryLabel)
        outer.addWidget(box2)

        # -- 3. run --------------------------------------------------------------------------
        box3 = QtWidgets.QGroupBox('3. Run')
        l3 = QtWidgets.QVBoxLayout(box3)
        row3 = QtWidgets.QHBoxLayout()
        self.StartPushButton = QtWidgets.QPushButton('Start (in pipeline order)')
        self.StopPushButton = QtWidgets.QPushButton('Stop after the current step')
        self.StopPushButton.setEnabled(False)
        self.ProgressLabel = QtWidgets.QLabel('')
        row3.addWidget(self.StartPushButton)
        row3.addWidget(self.StopPushButton)
        row3.addWidget(self.ProgressLabel, 1)
        l3.addLayout(row3)
        self.LogPlainTextEdit = QtWidgets.QPlainTextEdit()
        self.LogPlainTextEdit.setReadOnly(True)
        self.LogPlainTextEdit.setMaximumBlockCount(2000)
        self.LogPlainTextEdit.setMinimumHeight(140)
        l3.addWidget(self.LogPlainTextEdit)
        outer.addWidget(box3)
        outer.addStretch(1)

    # -- values --------------------------------------------------------------------

    def policies(self):
        return {step: combo.currentText() for step, combo in self.policy_combos.items()}

    def set_policies(self, policies):
        for step, pol in (policies or {}).items():
            if step in self.policy_combos and pol in S.POLICIES:
                self.policy_combos[step].setCurrentText(pol)

    def set_settings(self, texts):
        """The 'settings' column: one line per step of the parameters it
        would run with right now, read from the tabs themselves."""
        for i, step in enumerate(S.STEPS):
            item = self.StepsTable.item(i, 3)
            if item is not None:
                text = texts.get(step, '')
                item.setText(text)
                item.setToolTip((text + '\n\n' if text else '') + STEP_NOTES[step])

    def step_at(self, row):
        return S.STEPS[row] if 0 <= row < len(S.STEPS) else None

    def explicit_steps(self):
        """The steps whose policy a person chose away from 'skip' among
        those the runner never starts on its own."""
        return tuple(s for s in S.NEVER_AUTO if self.policy_combos[s].currentText() != 'skip')

    def set_status_rows(self, rows, plan=None):
        """Fill the FOV x step grid from status.status_table rows and the
        'to run' column from a plan."""
        self.StatusTable.setRowCount(len(rows))
        for i, r in enumerate(rows):
            self.StatusTable.setItem(i, 0, QtWidgets.QTableWidgetItem(str(r['fov'])))
            for j, step in enumerate(S.STEPS):
                st = r[step]
                item = QtWidgets.QTableWidgetItem(STATE_TEXT[st['state']] + (f' {st["have"]}/{st["want"]}' if st['state'] == 'partial' else ''))
                item.setBackground(QtGui.QBrush(STATE_COLOURS[st['state']]))
                item.setToolTip(st.get('detail', ''))
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.StatusTable.setItem(i, j + 1, item)
        self.StatusTable.resizeColumnsToContents()
        summ = S.summarize(rows) if rows else {}
        parts = []
        for step in S.STEPS:
            c = summ.get(step)
            if c:
                parts.append(f'{S.LABEL.get(step, step)}: {c["done"]} done, {c["partial"]} partial, {c["missing"]} missing')
        self.SummaryLabel.setText('; '.join(parts))
        if plan is not None:
            for i, step in enumerate(S.STEPS):
                n = len(plan.get(step, []))
                self.StepsTable.item(i, 2).setText(str(n) if n else '')

    def set_running(self, running):
        self.StartPushButton.setEnabled(not running)
        self.StopPushButton.setEnabled(running)
        self.RefreshPushButton.setEnabled(not running)
        for combo in self.policy_combos.values():
            combo.setEnabled(not running)

    def append_log(self, text):
        self.LogPlainTextEdit.appendPlainText(text)
