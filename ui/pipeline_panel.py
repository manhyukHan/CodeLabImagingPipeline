"""The Pipeline tab: every step of a store's workflow with a policy,
the FOV x step status grid read from the store, and one Start.

Policies per step: 'existing' runs only the FOVs the store does not
hold yet (the app's append), 'redo' runs every FOV (overwrite), 'skip'
leaves the step out. Segmentation defaults to 'skip': it is the step
the user does by hand, and the runner never starts it unless it is
chosen here. Ingestion is never overwritten from here (a redo runs as
append). The FOV list is the Ingestion tab's, as for every batch button.
"""
from PyQt5 import QtCore, QtGui, QtWidgets

from codelab_pipeline.pipeline import status as S

STEP_LABELS = {'ingestion': 'Ingestion', 'segmentation': 'Cell segmentation', 'fov_alignment': 'FOV alignment',
               'cross_modal': 'Cross-modal alignment', 'cell_alignment': 'Cell alignment', 'localization': 'Spot localization',
               'celltype': 'Celltype determination', 'cellcycle': 'Cell cycle', 'tracing': 'Chromatin tracing'}
STEP_NOTES = {'ingestion': 'append only from here (an overwrite clears stacks: do that on the Ingestion tab)',
              'segmentation': 'your own step: runs only when chosen here, FOV by FOV (run, save)',
              'fov_alignment': 'the Alignment tab\'s "all FOVs" batch, append = missing (FOV, hybe) pairs',
              'cross_modal': 'n/a with one modality',
              'cell_alignment': 'the slowest step; append = cells without matrices',
              'localization': 'headless v3 batch on the selected model; expected rounds = localization round + cell-cycle genes + countable readouts',
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
            'One run over the whole store, in pipeline order, using each tab\'s current settings. '
            'The FOV list is the Ingestion tab\'s. Refresh reads what the store already holds; a policy of '
            '"existing" runs only what is missing, "redo" runs every FOV, "skip" leaves the step out.')
        intro.setWordWrap(True)
        outer.addWidget(intro)

        # -- 1. steps and policies ---------------------------------------------------
        box1 = QtWidgets.QGroupBox('1. Steps and policies')
        l1 = QtWidgets.QVBoxLayout(box1)
        self.StepsTable = QtWidgets.QTableWidget(len(S.STEPS), 4)
        self.StepsTable.setHorizontalHeaderLabels(['step', 'policy', 'to run', 'note'])
        self.StepsTable.verticalHeader().setVisible(False)
        self.StepsTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.StepsTable.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.policy_combos = {}
        for i, step in enumerate(S.STEPS):
            item = QtWidgets.QTableWidgetItem(STEP_LABELS[step])
            item.setToolTip(step)
            self.StepsTable.setItem(i, 0, item)
            combo = QtWidgets.QComboBox()
            combo.addItems(list(S.POLICIES))
            combo.setCurrentText(DEFAULTS[step])
            setattr(self, COMBO_NAMES[step], combo)
            self.policy_combos[step] = combo
            self.StepsTable.setCellWidget(i, 1, combo)
            self.StepsTable.setItem(i, 2, QtWidgets.QTableWidgetItem(''))
            note = QtWidgets.QTableWidgetItem(STEP_NOTES[step])
            note.setForeground(QtGui.QBrush(QtGui.QColor(90, 90, 90)))
            self.StepsTable.setItem(i, 3, note)
        self.StepsTable.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        self.StepsTable.resizeColumnsToContents()
        self.StepsTable.setMinimumHeight(30 * (len(S.STEPS) + 1) + 8)
        l1.addWidget(self.StepsTable)
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
