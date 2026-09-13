from PyQt5 import QtWidgets, QtCore

from codelab_pipeline.analysis import cellcycle as CC


class CellCyclePanelUI(object):
    """
    The Cell Cycle stage: per-cell phase on a circle from an RNA panel,
    between Celltype Determination and Chromatin Tracing (user decision
    2026-09-13: the phase is a cell attribute like celltype, so it is
    determined here, and the Analysis tab only gates on it).

    Four sections, top to bottom in the order the work happens:
      1. the panel -- which RNA sources count as which genes, with the
         biological role each gene plays in ORIENTING the circle;
      2. the model -- training population (the cycling celltypes), the
         dispersion, other experiments' exported counts as bridges, and
         the fit / place / load / save doors;
      3. the figures the design records as the stage's validation;
      4. categories and verdict gates -- the arcs the user draws on the
         circle and the gates a placement must pass; stored with the
         model, applied when placements are read.

    The panel only exposes widget state; windows/cellcycle_wiring.py
    reads it and runs the toolbox (codelab_pipeline/analysis/cellcycle).
    """
    ROLE_COLUMN = 5

    def setupUi(self, Widget):
        Widget.setObjectName('CellCyclePanel')
        layout = QtWidgets.QVBoxLayout(Widget)

        # -- 1. panel -----------------------------------------------------
        panelGroup = QtWidgets.QGroupBox('1. Panel: RNA sources counted as genes')
        panelLayout = QtWidgets.QVBoxLayout(panelGroup)
        self.SourceTableWidget = QtWidgets.QTableWidget(0, 6)
        self.SourceTableWidget.setHorizontalHeaderLabels(['use', 'modality', 'hybe', 'channel', 'gene', 'role'])
        self.SourceTableWidget.horizontalHeader().setStretchLastSection(True)
        self.SourceTableWidget.verticalHeader().setVisible(False)
        self.SourceTableWidget.setMinimumHeight(180)
        self.SourceTableWidget.setToolTip(
            'One row per (modality, hybe, channel) the layouts declare. "use" makes the row a\n'
            'panel gene; "gene" is the name the model reports (defaults to the readout name\n'
            'without its _mRNA/_exon suffix); "role" only orients the circle: the S genes\'\n'
            'mean peak becomes 0 and the G2/M genes\' mean peak lies in the forward half turn.\n'
            'Housekeeping genes never bridge experiments. Nascent/intron, repeat and toe\n'
            'rounds are not counts (design 3b) and start unchecked. The fiducial channel is '
            'never a gene and is not listed; every other channel of a hybe is.')
        panelLayout.addWidget(self.SourceTableWidget)
        row = QtWidgets.QWidget()
        rowLayout = QtWidgets.QHBoxLayout(row)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        self.CheckMrnaRoundsPushButton = QtWidgets.QPushButton('Check mRNA/exon rounds')
        self.UncheckAllSourcesPushButton = QtWidgets.QPushButton('Uncheck all')
        self.RolesFromListsPushButton = QtWidgets.QPushButton('Roles from known lists')
        self.RolesFromListsPushButton.setToolTip('Seurat/Tirosh S and G2/M lists plus CCNE1, CDT1 (S) and CCNB1 (G2/M); '
                                                 'GAPDH/ACTB/TUBB/RPLP0/HPRT1 housekeeping.')
        self.ShowKnownListsPushButton = QtWidgets.QPushButton('Show known lists')
        for b in (self.CheckMrnaRoundsPushButton, self.UncheckAllSourcesPushButton, self.RolesFromListsPushButton,
                  self.ShowKnownListsPushButton):
            rowLayout.addWidget(b)
        panelLayout.addWidget(row)
        form = QtWidgets.QFormLayout()
        self.FovListLineEdit = QtWidgets.QLineEdit()
        self.FovListLineEdit.setPlaceholderText('blank = the Ingestion tab\'s FOV list')
        form.addRow('FOVs:', self.FovListLineEdit)
        self.ProxyComboBox = QtWidgets.QComboBox()
        self.ProxyComboBox.addItems(['count (spots at p_exist >= 0.5)', 'soft count (sum of p_exist)'])
        self.ProxyComboBox.setToolTip('Measured (design 4.2): counts at the classifier\'s own 0.5 and the soft count give '
                                      'the same phase within ~8 deg; brightness weighting and mask intensity do not.')
        form.addRow('Expression proxy:', self.ProxyComboBox)
        self.MinTotalSpinBox = QtWidgets.QSpinBox()
        self.MinTotalSpinBox.setRange(0, 100000)
        self.MinTotalSpinBox.setValue(10)
        self.MinTotalSpinBox.setToolTip('Cells with fewer panel counts than this are neither trained on nor placed.')
        form.addRow('Min panel total per cell:', self.MinTotalSpinBox)
        panelLayout.addLayout(form)
        btnRow = QtWidgets.QWidget()
        btnLayout = QtWidgets.QHBoxLayout(btnRow)
        btnLayout.setContentsMargins(0, 0, 0, 0)
        self.BuildCountsPushButton = QtWidgets.QPushButton('Build counts from the spot store')
        self.ExportCountsPushButton = QtWidgets.QPushButton('Export counts for another project...')
        self.ExportCountsPushButton.setToolTip('Writes this experiment\'s cells x genes counts (training celltypes, roles, '
                                               'alpha) as JSON, to be added as a bridge in another project\'s fit.')
        btnLayout.addWidget(self.BuildCountsPushButton)
        btnLayout.addWidget(self.ExportCountsPushButton)
        panelLayout.addWidget(btnRow)
        self.CountStatusLabel = QtWidgets.QLabel('no counts yet')
        self.CountStatusLabel.setWordWrap(True)
        panelLayout.addWidget(self.CountStatusLabel)
        layout.addWidget(panelGroup)

        # -- 2. model -------------------------------------------------------
        modelGroup = QtWidgets.QGroupBox('2. Model: fit on the cycling population, place every cell')
        modelLayout = QtWidgets.QVBoxLayout(modelGroup)
        trainRow = QtWidgets.QWidget()
        trainLayout = QtWidgets.QHBoxLayout(trainRow)
        trainLayout.setContentsMargins(0, 0, 0, 0)
        trainCol = QtWidgets.QVBoxLayout()
        trainCol.addWidget(QtWidgets.QLabel('Training celltypes (the CYCLING populations; none checked = every cell):'))
        self.TrainingCelltypeListWidget = QtWidgets.QListWidget()
        self.TrainingCelltypeListWidget.setMaximumHeight(90)
        self.TrainingCelltypeListWidget.setToolTip('Measured (design 4.3): fitting the profiles on arrested populations '
                                                   'collapses the conditions into clusters. Arrested cells are PLACED, never trained on.')
        trainCol.addWidget(self.TrainingCelltypeListWidget)
        trainLayout.addLayout(trainCol, 1)
        bridgeCol = QtWidgets.QVBoxLayout()
        bridgeCol.addWidget(QtWidgets.QLabel('Bridge experiments (exported counts; shared genes align the frames):'))
        self.BridgeListWidget = QtWidgets.QListWidget()
        self.BridgeListWidget.setMaximumHeight(90)
        bridgeCol.addWidget(self.BridgeListWidget)
        bridgeBtns = QtWidgets.QHBoxLayout()
        self.AddBridgePushButton = QtWidgets.QPushButton('Add exported counts...')
        self.RemoveBridgePushButton = QtWidgets.QPushButton('Remove')
        bridgeBtns.addWidget(self.AddBridgePushButton)
        bridgeBtns.addWidget(self.RemoveBridgePushButton)
        bridgeCol.addLayout(bridgeBtns)
        trainLayout.addLayout(bridgeCol, 1)
        modelLayout.addWidget(trainRow)
        mform = QtWidgets.QFormLayout()
        alphaRow = QtWidgets.QWidget()
        alphaLayout = QtWidgets.QHBoxLayout(alphaRow)
        alphaLayout.setContentsMargins(0, 0, 0, 0)
        self.AlphaSpinBox = QtWidgets.QDoubleSpinBox()
        self.AlphaSpinBox.setRange(1.0, 100000.0)
        self.AlphaSpinBox.setDecimals(0)
        self.AlphaSpinBox.setValue(100.0)
        self.AlphaSpinBox.setToolTip('Dirichlet-multinomial dispersion: a cell carries at most ~alpha effective counts. '
                                     'Measured by held-out evidence: chr19 100, JP_001 300, JP_002 100.')
        self.AlphaHeldOutPushButton = QtWidgets.QPushButton('Choose by held-out evidence')
        alphaLayout.addWidget(self.AlphaSpinBox)
        alphaLayout.addWidget(self.AlphaHeldOutPushButton)
        mform.addRow('Dispersion alpha:', alphaRow)
        modelLayout.addLayout(mform)
        fitRow = QtWidgets.QWidget()
        fitLayout = QtWidgets.QHBoxLayout(fitRow)
        fitLayout.setContentsMargins(0, 0, 0, 0)
        self.FitPushButton = QtWidgets.QPushButton('Fit model and place all cells')
        self.FitPushButton.setToolTip('EM on the training cells with the bridges (reflection fixed by the roles), '
                                      'then every cell of every FOV is placed and written to analysis/fov###/cellcycle.json; '
                                      'the model with its provenance to analysis/cellcycle_model.json.')
        self.PlacePushButton = QtWidgets.QPushButton('Place all cells with the current model')
        self.LoadModelPushButton = QtWidgets.QPushButton('Load model...')
        self.SaveModelPushButton = QtWidgets.QPushButton('Save model as...')
        for b in (self.FitPushButton, self.PlacePushButton, self.LoadModelPushButton, self.SaveModelPushButton):
            fitLayout.addWidget(b)
        modelLayout.addWidget(fitRow)
        self.ModelStatusLabel = QtWidgets.QLabel('no model')
        self.ModelStatusLabel.setWordWrap(True)
        modelLayout.addWidget(self.ModelStatusLabel)
        layout.addWidget(modelGroup)

        # -- 3. figures -----------------------------------------------------
        figGroup = QtWidgets.QGroupBox('3. Figures (each opens its own window; Save Result writes PNG + CSV)')
        figLayout = QtWidgets.QGridLayout(figGroup)
        self.SpectrumPushButton = QtWidgets.QPushButton('Spectrum and total count per condition')
        self.ProfilesPushButton = QtWidgets.QPushButton('Gene profiles by role')
        self.EmbeddingsPushButton = QtWidgets.QPushButton('The ring: PCA / tSNE / UMAP')
        self.VerdictsPushButton = QtWidgets.QPushButton('Verdict histograms (R, fit_z, BF_ring, radius)')
        self.ContributionPushButton = QtWidgets.QPushButton('Gene contribution (Fisher share)')
        self.GroupTablePushButton = QtWidgets.QPushButton('Group table (anchors)')
        self.HalfPanelPushButton = QtWidgets.QPushButton('Reproducibility: two panel halves by depth')
        self.CycleTimePushButton = QtWidgets.QPushButton('Angle as cycle time')
        for i, b in enumerate((self.SpectrumPushButton, self.ProfilesPushButton, self.EmbeddingsPushButton,
                               self.VerdictsPushButton, self.ContributionPushButton, self.GroupTablePushButton,
                               self.HalfPanelPushButton, self.CycleTimePushButton)):
            figLayout.addWidget(b, i // 2, i % 2)
        layout.addWidget(figGroup)

        # -- 4. categories and gates ----------------------------------------
        catGroup = QtWidgets.QGroupBox('4. Categories and verdict gates (stored with the model, applied when placements are read)')
        catLayout = QtWidgets.QVBoxLayout(catGroup)
        arcRow = QtWidgets.QWidget()
        arcLayout = QtWidgets.QHBoxLayout(arcRow)
        arcLayout.setContentsMargins(0, 0, 0, 0)
        self.ArcsTableWidget = QtWidgets.QTableWidget(0, 3)
        self.ArcsTableWidget.setHorizontalHeaderLabels(['category', 'start (deg)', 'end (deg)'])
        self.ArcsTableWidget.horizontalHeader().setStretchLastSection(True)
        self.ArcsTableWidget.verticalHeader().setVisible(False)
        self.ArcsTableWidget.setMaximumHeight(120)
        self.ArcsTableWidget.setToolTip('Each arc runs FORWARD from start to end (so 300 -> 60 crosses 0 = S-phase). '
                                        'Overlaps resolve to the first row; a phase in no arc reads Unassigned.')
        arcLayout.addWidget(self.ArcsTableWidget, 1)
        arcBtns = QtWidgets.QVBoxLayout()
        self.AddArcPushButton = QtWidgets.QPushButton('Add arc')
        self.RemoveArcPushButton = QtWidgets.QPushButton('Remove arc')
        self.ProposeArcsPushButton = QtWidgets.QPushButton('Propose from roles')
        self.ProposeArcsPushButton.setToolTip('S around the S genes\' mean peak (0), G2/M around the G2/M genes\' mean peak, '
                                              'G1 between them: a starting point to edit, not a verdict.')
        for b in (self.AddArcPushButton, self.RemoveArcPushButton, self.ProposeArcsPushButton):
            arcBtns.addWidget(b)
        arcBtns.addStretch(1)
        arcLayout.addLayout(arcBtns)
        catLayout.addWidget(arcRow)
        gates = QtWidgets.QGridLayout()
        self.MinRCheckBox = QtWidgets.QCheckBox('min R (posterior concentration)')
        self.MinRSpinBox = QtWidgets.QDoubleSpinBox(); self.MinRSpinBox.setRange(0.0, 1.0); self.MinRSpinBox.setSingleStep(0.05); self.MinRSpinBox.setValue(0.5)
        self.MinBfCheckBox = QtWidgets.QCheckBox('min log BF_ring (ring vs centre)')
        self.MinBfSpinBox = QtWidgets.QDoubleSpinBox(); self.MinBfSpinBox.setRange(-100.0, 100.0); self.MinBfSpinBox.setValue(0.0)
        self.MinFitZCheckBox = QtWidgets.QCheckBox('min fit_z (evidence vs training cells)')
        self.MinFitZSpinBox = QtWidgets.QDoubleSpinBox(); self.MinFitZSpinBox.setRange(-100.0, 100.0); self.MinFitZSpinBox.setValue(-3.0)
        self.MaxRadiusCheckBox = QtWidgets.QCheckBox('max centre ratio (plane radius)')
        self.MaxRadiusSpinBox = QtWidgets.QDoubleSpinBox(); self.MaxRadiusSpinBox.setRange(0.0, 100.0); self.MaxRadiusSpinBox.setValue(2.0)
        self.MinTotalGateCheckBox = QtWidgets.QCheckBox('min panel total')
        self.MinTotalGateSpinBox = QtWidgets.QDoubleSpinBox(); self.MinTotalGateSpinBox.setRange(0.0, 1e6); self.MinTotalGateSpinBox.setDecimals(0); self.MinTotalGateSpinBox.setValue(10.0)
        self.MinBfCheckBox.setChecked(True)
        for i, (cb, sb) in enumerate(((self.MinRCheckBox, self.MinRSpinBox), (self.MinBfCheckBox, self.MinBfSpinBox),
                                      (self.MinFitZCheckBox, self.MinFitZSpinBox), (self.MaxRadiusCheckBox, self.MaxRadiusSpinBox),
                                      (self.MinTotalGateCheckBox, self.MinTotalGateSpinBox))):
            gates.addWidget(cb, i // 2, 2 * (i % 2))
            gates.addWidget(sb, i // 2, 2 * (i % 2) + 1)
        catLayout.addLayout(gates)
        bform = QtWidgets.QFormLayout()
        self.BirthDegSpinBox = QtWidgets.QDoubleSpinBox()
        self.BirthDegSpinBox.setRange(0.0, 360.0)
        self.BirthDegSpinBox.setValue(215.0)
        self.BirthDegSpinBox.setToolTip('Where the cycle-time clock starts (cell birth, M exit). Proposed from the G2/M '
                                        'profiles\' descent to their mean; the clock itself is the training spectrum.')
        bform.addRow('Birth angle for cycle time (deg):', self.BirthDegSpinBox)
        catLayout.addLayout(bform)
        self.ApplyCategoriesPushButton = QtWidgets.QPushButton('Apply categories and gates to the stored model')
        catLayout.addWidget(self.ApplyCategoriesPushButton)
        self.CategoryStatusLabel = QtWidgets.QLabel('')
        self.CategoryStatusLabel.setWordWrap(True)
        catLayout.addWidget(self.CategoryStatusLabel)
        layout.addWidget(catGroup)
        layout.addStretch(1)

        self.CheckMrnaRoundsPushButton.clicked.connect(self._check_mrna_rounds)
        self.UncheckAllSourcesPushButton.clicked.connect(lambda: self._set_all_use(False))
        self.RolesFromListsPushButton.clicked.connect(self._roles_from_lists)
        self.ShowKnownListsPushButton.clicked.connect(lambda: self.show_known_lists(Widget))
        self.AddArcPushButton.clicked.connect(lambda: self._add_arc_row('', 0.0, 0.0))
        self.RemoveArcPushButton.clicked.connect(self._remove_arc_row)

    # -- 1. panel state ---------------------------------------------------

    def populate_sources(self, records_by_modality):
        """One row per (modality, hybe, channel) of the layouts' records
        (folder, readout_name, channels, fiducial_channel). mRNA/exon
        rounds on a non-fiducial channel start checked, with the gene
        name and its default role."""
        t = self.SourceTableWidget
        t.setRowCount(0)
        for modality, records in (records_by_modality or {}).items():
            for r in records:
                fid = r.get('fiducial_channel')
                name = str(r.get('readout_name') or '')
                for ch in r.get('channels', []):
                    # the fiducial channel is never a gene: no row (user
                    # request 2026-09-13); every other channel of a
                    # multichannel readout gets one
                    if fid is not None and ch == fid:
                        continue
                    gene = CC.gene_from_readout(name) if name else ''
                    use = bool(name) and CC.countable_round(name)
                    self._add_source_row((modality, r['folder'], int(ch)), gene, CC.default_role(gene), use)
        t.resizeColumnsToContents()

    def _add_source_row(self, src, gene, role, use):
        t = self.SourceTableWidget
        i = t.rowCount()
        t.insertRow(i)
        useItem = QtWidgets.QTableWidgetItem()
        useItem.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
        useItem.setCheckState(QtCore.Qt.Checked if use else QtCore.Qt.Unchecked)
        useItem.setData(QtCore.Qt.UserRole, list(src))
        # whether the row is a countable round on a readout channel --
        # what "Check mRNA/exon rounds" re-checks, decided once from the
        # layout, not guessed from the (editable) gene name later
        useItem.setData(QtCore.Qt.UserRole + 1, bool(use))
        t.setItem(i, 0, useItem)
        for col, text in ((1, src[0]), (2, src[1]), (3, str(src[2]))):
            it = QtWidgets.QTableWidgetItem(text)
            it.setFlags(QtCore.Qt.ItemIsEnabled)
            t.setItem(i, col, it)
        t.setItem(i, 4, QtWidgets.QTableWidgetItem(gene))
        combo = QtWidgets.QComboBox()
        combo.addItems(list(CC.ROLES))
        combo.setCurrentIndex(max(0, list(CC.ROLES).index(role) if role in CC.ROLES else 0))
        t.setCellWidget(i, self.ROLE_COLUMN, combo)

    def show_known_lists(self, parent=None):
        """A pop-up table of the role lists the defaults come from."""
        dlg = QtWidgets.QDialog(parent)
        dlg.setWindowTitle('Known gene roles')
        dlg.resize(520, 620)
        layout = QtWidgets.QVBoxLayout(dlg)
        note = QtWidgets.QLabel(
            'Roles only ORIENT the circle (S mean peak -> 0, G2/M mean peak forward). '
            'S and G2/M: the Seurat / Tirosh 2016 lists, plus CCNE1 and CDT1 (S) and CCNB1 (G2/M). '
            'Housekeeping genes never bridge experiments. Edit any role in the panel table.')
        note.setWordWrap(True)
        layout.addWidget(note)
        rows = ([(g, 'S') for g in CC.S_GENES] + [(g, 'G2/M') for g in CC.G2M_GENES]
                + [(g, 'housekeeping') for g in CC.HOUSEKEEPING])
        table = QtWidgets.QTableWidget(len(rows), 2)
        table.setHorizontalHeaderLabels(['gene', 'role'])
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        for i, (g, role) in enumerate(rows):
            table.setItem(i, 0, QtWidgets.QTableWidgetItem(g))
            table.setItem(i, 1, QtWidgets.QTableWidgetItem(role))
        table.resizeColumnsToContents()
        layout.addWidget(table)
        close = QtWidgets.QPushButton('Close')
        close.clicked.connect(dlg.accept)
        layout.addWidget(close)
        self._known_lists_dialog = dlg
        dlg.show()
        return dlg

    def gene_rows(self):
        """[{'source': (modality, hybe, channel), 'gene': str, 'role': str,
        'use': bool}] for every row."""
        t = self.SourceTableWidget
        out = []
        for i in range(t.rowCount()):
            useItem = t.item(i, 0)
            gene = (t.item(i, 4).text() if t.item(i, 4) else '').strip()
            combo = t.cellWidget(i, self.ROLE_COLUMN)
            out.append({'source': tuple(useItem.data(QtCore.Qt.UserRole)),
                        'gene': gene, 'role': combo.currentText() if combo else '-',
                        'use': useItem.checkState() == QtCore.Qt.Checked})
        return out

    def checked_genes(self):
        """[(source, gene, role)] of the rows in use with a gene name."""
        return [(r['source'], r['gene'], r['role']) for r in self.gene_rows() if r['use'] and r['gene']]

    def _set_all_use(self, on):
        t = self.SourceTableWidget
        for i in range(t.rowCount()):
            t.item(i, 0).setCheckState(QtCore.Qt.Checked if on else QtCore.Qt.Unchecked)

    def _check_mrna_rounds(self):
        t = self.SourceTableWidget
        for i in range(t.rowCount()):
            countable = bool(t.item(i, 0).data(QtCore.Qt.UserRole + 1))
            t.item(i, 0).setCheckState(QtCore.Qt.Checked if countable else QtCore.Qt.Unchecked)

    def _roles_from_lists(self):
        t = self.SourceTableWidget
        for i in range(t.rowCount()):
            gene = (t.item(i, 4).text() if t.item(i, 4) else '').strip()
            combo = t.cellWidget(i, self.ROLE_COLUMN)
            if combo is not None:
                combo.setCurrentIndex(list(CC.ROLES).index(CC.default_role(gene)))

    def gene_config(self):
        """The checked rows as one config string: 'mod|hybe|ch|gene|role,...'."""
        return ','.join(f'{s[0]}|{s[1]}|{s[2]}|{g}|{r}' for s, g, r in self.checked_genes())

    def set_gene_config(self, text):
        """Apply a gene_config string: rows named in it are checked with
        the given gene and role, every other row unchecked. Rows the
        layouts no longer declare are skipped."""
        want = {}
        for part in str(text or '').split(','):
            bits = part.split('|')
            if len(bits) == 5:
                want[(bits[0], bits[1], int(bits[2]))] = (bits[3], bits[4])
        t = self.SourceTableWidget
        for i in range(t.rowCount()):
            src = tuple(t.item(i, 0).data(QtCore.Qt.UserRole))
            hit = want.get(src)
            t.item(i, 0).setCheckState(QtCore.Qt.Checked if hit else QtCore.Qt.Unchecked)
            if hit:
                t.item(i, 4).setText(hit[0])
                combo = t.cellWidget(i, self.ROLE_COLUMN)
                if combo is not None and hit[1] in CC.ROLES:
                    combo.setCurrentIndex(list(CC.ROLES).index(hit[1]))

    def proxy_metric(self):
        """'n' (count at p_exist >= 0.5) or 'soft' (sum of p_exist)."""
        return 'soft' if self.ProxyComboBox.currentIndex() == 1 else 'n'

    # -- 2. model state ---------------------------------------------------

    def set_celltype_names(self, names):
        """Refresh the training list, keeping what was checked."""
        lw = self.TrainingCelltypeListWidget
        checked = set(self.training_celltypes())
        lw.clear()
        for n in names:
            item = QtWidgets.QListWidgetItem(str(n))
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if n in checked else QtCore.Qt.Unchecked)
            lw.addItem(item)

    def training_celltypes(self):
        lw = self.TrainingCelltypeListWidget
        return [lw.item(i).text() for i in range(lw.count())
                if lw.item(i).checkState() == QtCore.Qt.Checked]

    def set_training_celltypes(self, names):
        lw = self.TrainingCelltypeListWidget
        names = set(names)
        for i in range(lw.count()):
            lw.item(i).setCheckState(QtCore.Qt.Checked if lw.item(i).text() in names else QtCore.Qt.Unchecked)

    def bridges(self):
        lw = self.BridgeListWidget
        return [lw.item(i).text() for i in range(lw.count())]

    # -- 4. categories and gates ----------------------------------------------

    def _add_arc_row(self, name, start, end):
        t = self.ArcsTableWidget
        i = t.rowCount()
        t.insertRow(i)
        t.setItem(i, 0, QtWidgets.QTableWidgetItem(str(name)))
        t.setItem(i, 1, QtWidgets.QTableWidgetItem(f'{float(start):g}'))
        t.setItem(i, 2, QtWidgets.QTableWidgetItem(f'{float(end):g}'))

    def _remove_arc_row(self):
        t = self.ArcsTableWidget
        rows = sorted({ix.row() for ix in t.selectedIndexes()}, reverse=True)
        for r in rows:
            t.removeRow(r)

    def arcs(self):
        """[{'name', 'start_deg', 'end_deg'}] from the table; rows with no
        name or an unreadable angle are skipped."""
        t = self.ArcsTableWidget
        out = []
        for i in range(t.rowCount()):
            name = (t.item(i, 0).text() if t.item(i, 0) else '').strip()
            try:
                start = float(t.item(i, 1).text())
                end = float(t.item(i, 2).text())
            except (AttributeError, ValueError):
                continue
            if name:
                out.append({'name': name, 'start_deg': start % 360.0, 'end_deg': end % 360.0})
        return out

    def set_arcs(self, arcs):
        t = self.ArcsTableWidget
        t.setRowCount(0)
        for a in arcs or []:
            self._add_arc_row(a['name'], a['start_deg'], a['end_deg'])

    def gates(self):
        """{gate key: value or None} over cellcycle.GATE_KEYS' keys the
        panel offers."""
        return {'min_R': self.MinRSpinBox.value() if self.MinRCheckBox.isChecked() else None,
                'min_bf_ring': self.MinBfSpinBox.value() if self.MinBfCheckBox.isChecked() else None,
                'min_fit_z': self.MinFitZSpinBox.value() if self.MinFitZCheckBox.isChecked() else None,
                'max_radius': self.MaxRadiusSpinBox.value() if self.MaxRadiusCheckBox.isChecked() else None,
                'min_total': self.MinTotalGateSpinBox.value() if self.MinTotalGateCheckBox.isChecked() else None}

    def set_gates(self, gates):
        gates = gates or {}
        for key, cb, sb in (('min_R', self.MinRCheckBox, self.MinRSpinBox),
                            ('min_bf_ring', self.MinBfCheckBox, self.MinBfSpinBox),
                            ('min_fit_z', self.MinFitZCheckBox, self.MinFitZSpinBox),
                            ('max_radius', self.MaxRadiusCheckBox, self.MaxRadiusSpinBox),
                            ('min_total', self.MinTotalGateCheckBox, self.MinTotalGateSpinBox)):
            v = gates.get(key)
            cb.setChecked(v is not None)
            if v is not None:
                sb.setValue(float(v))
