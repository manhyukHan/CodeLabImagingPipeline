"""
The Analysis tab's wiring: widget values -> toolbox calls -> displayer.

A separate module on purpose, and the seam is the point: everything
below composes codelab_pipeline/analysis functions that run identically
in a notebook -- MainWindow grew past 11k lines, and this tab's whole
job is to prove the toolbox needs nothing from the app. MainWindow
instantiates AnalysisWiring(self) and forwards nothing else.

Population.build and every view run inside FnWorker threads; the
HDF5-heavy extraction inside build already fans out to CHILD processes
(pmap), so the thread never holds h5py's lock on the GUI's behalf.
"""
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtWidgets

from codelab_pipeline.analysis import (polymer, ensemble, expression,
                                       distances, gate, population,
                                       figures, report)
from canvas.analysis_figure_displayer import AnalysisFigureDisplayer


class AnalysisWiring(QtCore.QObject):
    def __init__(self, mw):
        super().__init__(mw)
        self.mw = mw
        self.panel = mw.ui.AnalysisPanel
        self.pop = None
        self.qc = None            # apply_qc output when applied
        self.qc_thresholds = None
        self.displayers = []   # ADDITIVE windows: every view pops a
        # new one and old ones stay (per explicit request)
        p = self.panel
        p.BuildPopulationPushButton.clicked.connect(self.build_population)
        p.CheckSelectedSourcesPushButton.clicked.connect(
            lambda: self._set_selected_sources(True))
        p.UncheckSelectedSourcesPushButton.clicked.connect(
            lambda: self._set_selected_sources(False))
        p.CheckSpotSourcesPushButton.clicked.connect(self.check_spot_sources)
        p.CheckModalityChannelPushButton.clicked.connect(
            self.check_modality_channel)
        p.DeriveQcPushButton.clicked.connect(self.derive_qc)
        p.PreviewQcPushButton.clicked.connect(self.preview_qc)
        p.ApplyQcCheckBox.toggled.connect(self._refresh_gate_summary)
        p.DistanceDimsComboBox.currentIndexChanged.connect(
            self._on_dims_changed)
        p.IsotropyQcPushButton.clicked.connect(self.view_isotropy)
        p.AddConditionPushButton.clicked.connect(self.add_condition)
        p.NewOrGroupPushButton.clicked.connect(self.new_or_group)
        p.RemoveConditionPushButton.clicked.connect(self.remove_condition)
        p.ClearConditionsPushButton.clicked.connect(self.clear_conditions)
        p.PreviewHistogramPushButton.clicked.connect(self.preview_histogram)
        p.EnsembleMapPushButton.clicked.connect(self.view_ensemble)
        p.FovConsistencyPushButton.clicked.connect(self.view_fov_consistency)
        p.AlleleDifferencePushButton.clicked.connect(self.view_allele_difference)
        p.ExpressionHistPushButton.clicked.connect(self.view_expression_hist)
        p.BrightnessVsCountPushButton.clicked.connect(self.view_brightness_vs_count)
        p.DistanceHistPushButton.clicked.connect(self.view_distance_hist)
        p.RepeatToeQcPushButton.clicked.connect(self.view_repeat_toe_qc)

    # -- population --------------------------------------------------------
    def _hybe_name(self, modality, hybe):
        """The layout's common name for a round (readout_name, e.g.
        'Gorab_exon' for Hyb_111), or '' when the layout gives none."""
        for r in (self.mw.hybe_records_by_modality or {}).get(modality, []):
            if r.get('folder') == hybe:
                return str(r.get('readout_name') or '')
        return ''

    def _source_label(self, src):
        """Display name for figure titles: the common hybe name shown
        beside the folder when the layout has one, per request. A traced
        source names the POLYMER position, not a channel."""
        m, h, ch = src
        name = self._hybe_name(m, h)
        head = f'{m}/{h} ({name})' if name else f'{m}/{h}'
        return f'{head}/traced' if str(ch) == 'traced' else f'{head}/ch{ch}'

    def _norm_label(self, normalize):
        """Title line naming the applied normalization, per request:
        'normalized: by_source, Hyb_133 (SO57_exon)' / 'by_modality'."""
        if not normalize:
            return None
        mode = normalize[0]
        if mode == 'by_source' and len(normalize) > 1:
            rm, rh, _rch = normalize[1]
            name = self._hybe_name(rm, rh)
            ref = f'{rh} ({name})' if name else rh
            return f'normalized: by_source, {ref}'
        return 'normalized: by_modality'

    def populate_sources(self):
        """Fill the source list + combos from the layouts' hybe records.
        Called by MainWindow after layouts parse."""
        p = self.panel
        p.SourceListWidget.clear()
        p.BulkModalityComboBox.clear()
        p.BulkChannelComboBox.clear()
        all_channels = set()
        tree = {}
        for modality, records in (self.mw.hybe_records_by_modality or {}).items():
            p.BulkModalityComboBox.addItem(modality)
            entries = []
            for r in records:
                fid = r.get('fiducial_channel')
                name = str(r.get('readout_name') or '')
                entries.append((r['folder'], name,
                                [(int(ch), ch == fid)
                                 for ch in r.get('channels', [])]))
                for ch in r.get('channels', []):
                    all_channels.add(int(ch))
                    shown = f'{r["folder"]} ({name})' if name else r['folder']
                    label = f'{modality} | {shown} | ch{ch}' + (
                        '  (fiducial)' if ch == fid else '')
                    src = (modality, r['folder'], int(ch))
                    item = QtWidgets.QListWidgetItem(label)
                    item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                    item.setCheckState(QtCore.Qt.Unchecked)
                    item.setData(QtCore.Qt.UserRole, list(src))
                    p.SourceListWidget.addItem(item)
            tree[modality] = entries
        for picker in (p.SourceAPicker, p.SourceBPicker,
                       p.ViewExprSourcePicker, p.ViewExprNormRefPicker,
                       p.ViewDistSourceAPicker, p.ViewDistSourceBPicker):
            picker.populate(tree)
        for ch in sorted(all_channels):
            p.BulkChannelComboBox.addItem(str(ch))

    def _set_selected_sources(self, checked):
        lw = self.panel.SourceListWidget
        for item in lw.selectedItems():
            item.setCheckState(QtCore.Qt.Checked if checked
                               else QtCore.Qt.Unchecked)

    def check_modality_channel(self):
        """Check EVERY source of the picked modality at the picked
        channel -- general in channel, per correction.

        The earlier version excluded hybes where this channel happens to
        be the fiducial one. That is a bad assumption twice over: the
        fiducial role is a PER-HYBE fact (with three channels the same
        wavelength is fiducial in one round and a real readout in
        another), and a fiducial-channel source is legitimately wanted
        anyway -- as a mask-intensity source, or simply as spots the
        user localized there. The picker names a channel; it should
        select that channel."""
        p = self.panel
        modality = p.BulkModalityComboBox.currentText()
        ch_text = p.BulkChannelComboBox.currentText()
        if not modality or not ch_text:
            return
        channel = int(ch_text)
        lw = p.SourceListWidget
        n = 0
        for i in range(lw.count()):
            item = lw.item(i)
            m, h, ch = item.data(QtCore.Qt.UserRole)
            if m == modality and int(ch) == channel:
                item.setCheckState(QtCore.Qt.Checked)
                n += 1
        self.mw.log(f'Analysis: checked {n} {modality} ch{channel} '
                    f'source(s).')

    def check_spot_sources(self):
        """Check every source with localized spots in the store -- the
        simple default start, per explicit request."""
        from codelab_pipeline.io import analysis_store
        modality = next(iter(self.mw.hybe_records_by_modality or {}), None)
        sp = self.mw._storage_path_for_modality(modality)
        if not sp:
            return
        have = set()
        for f in self._fovs():
            for sl in analysis_store.spot_slices(sp, int(f)):
                have.add((sl[0], sl[1], int(sl[2])))
        lw = self.panel.SourceListWidget
        n = 0
        for i in range(lw.count()):
            item = lw.item(i)
            src = tuple(item.data(QtCore.Qt.UserRole))
            src = (src[0], src[1], int(src[2]))
            if src in have:
                item.setCheckState(QtCore.Qt.Checked)
                n += 1
        self.mw.log(f'Analysis: checked {n} source(s) with spots '
                    f'({len(have)} slices in the store).')

    def _fovs(self):
        text = self.panel.FovListLineEdit.text().strip()
        if not text:
            text = self.mw.ui.IngestionPanel.FovListLineEdit.text()
        return self.mw._parse_fov_list(text)

    def build_population(self):
        p = self.panel
        # the first configured modality's storage path holds the analysis
        # capsules (they are shared across the store's modalities)
        modality = next(iter(self.mw.hybe_records_by_modality or {}), None)
        storage_path = self.mw._storage_path_for_modality(modality)
        if not storage_path:
            QtWidgets.QMessageBox.warning(self.mw, 'Analysis',
                                          'No storage path -- set up Ingestion first.')
            return
        fovs = self._fovs()
        if not fovs:
            QtWidgets.QMessageBox.warning(self.mw, 'Analysis', 'No FOVs.')
            return
        sources = p.checked_sources()
        if not sources:
            # THE DEFAULT, per explicit request: nothing checked means
            # every source that has spots in the store. Resolved here,
            # not at layout parse -- the scan reads the store (one
            # listdir per FOV) and startup stays lazy.
            self.check_spot_sources()
            sources = p.checked_sources()
        records = self.mw._active_hybe_records_for_modality(modality) or []
        voxel = self.mw._voxel_um()
        mask_int = p.MaskIntensityCheckBox.isChecked()
        # EXACT mask projection: one plain-data resolver per FOV (they
        # pickle into the extraction children); the cell is supplied at
        # transform time, so cell-level residuals still apply. Without
        # these, every post-cell-alignment cell falls to the flagged
        # reference-frame mask -- measured 'reference' on 386/386 cells,
        # and for cross-modal sources that is the whole bridge of error.
        resolvers = {}
        if mask_int:
            for f in fovs:
                try:
                    resolvers[int(f)] = self.mw._frame_resolver(None, f)
                except Exception:
                    pass
        p.BuildPopulationPushButton.setEnabled(False)
        p.PopulationStatusLabel.setText('building population...')

        def _compute():
            return population.Population.build(
                storage_path, fovs, records=records, modality=modality,
                sources=sources or None, spot_sources=sources or None,
                voxel_um=voxel, mask_intensity=mask_int,
                resolvers=resolvers or None,
                overwrite_cache=p.OverwriteCacheCheckBox.isChecked())

        def _done(pop):
            p.BuildPopulationPushButton.setEnabled(True)
            self.pop = pop
            self.qc = None
            self.qc_thresholds = None
            p.QcStatusLabel.setText('QC not derived (population rebuilt)')
            p.PopulationStatusLabel.setText(pop.summary())
            self._refresh_gate_summary()

        def _fail(msg):
            p.BuildPopulationPushButton.setEnabled(True)
            p.PopulationStatusLabel.setText(f'FAILED: {msg}')

        self._start(_compute, _done, _fail)

    def _start(self, fn, ok, fail):
        from windows.main_window import FnWorker
        self._worker = FnWorker(fn)
        self._worker.finished_ok.connect(ok)
        self._worker.failed.connect(fail)
        self._worker.start()

    def _need_pop(self, alleles=False, expr=False, spots=False):
        if self.pop is None:
            raise ValueError('build the population first (section 1)')
        if alleles and self.pop.alleles is None:
            raise ValueError('population has no alleles -- are any traced?')
        if expr and self.pop.expression is None:
            raise ValueError('population has no expression table -- check '
                             'sources in section 1 and rebuild')
        if spots and self.pop.spots is None:
            raise ValueError('population has no spot table -- check sources '
                             'in section 1 and rebuild')
        return self.pop

    # -- polymer QC --------------------------------------------------------
    def derive_qc(self):
        try:
            pop = self._need_pop(alleles=True)
            thr = polymer.qc_thresholds(pop.alleles, pop.dmaps(self.dims()),
                                        dims=self.dims())
            self.panel.set_qc_thresholds(thr)
            self._run_qc(thr)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self.mw, 'Polymer QC', str(e))

    def _run_qc(self, thr):
        pop = self.pop
        # STAMP the thresholds with the dimensionality actually in force.
        # They may have arrived from the edit boxes (which carry only the
        # four numbers) or from a config, so the stamp cannot be trusted
        # to have survived -- and an unstamped threshold set is exactly
        # what lets a 3D bound be applied to 2D data unnoticed.
        thr = dict(thr)
        thr['dims'] = self.dims()
        out = polymer.apply_qc(pop.alleles, pop.dmaps(self.dims()), thr,
                               dims=self.dims(),
                               min_traced=self.panel.QcMinTracedSpinBox.value())
        self.qc = out
        self.qc_thresholds = thr
        kept = int(out['kept'].sum())
        # the dimensionality is named FIRST and the numeric thresholds
        # follow: they are micrometre lengths that mean different cuts in
        # the two metrics, so the label that reports them has to say
        # which one they are. (It is also why this formats only the
        # numeric entries -- the 'dims' stamp is a string.)
        self.panel.QcStatusLabel.setText(
            f'QC [{thr.get("dims", "xyz")} distances]: '
            f'{kept}/{len(pop.alleles["amp"])} alleles kept, '
            f'{int(out["bads"].sum())} bins removed  |  '
            + '  '.join(f'{k}={v:.3g}' for k, v in thr.items()
                        if isinstance(v, (int, float))))
        self._refresh_gate_summary()

    def preview_qc(self):
        try:
            pop = self._need_pop(alleles=True)
            thr = self.panel.qc_threshold_values()
            if thr is None:
                raise ValueError('derive (or type) the thresholds first')
            self._run_qc(thr)
            fig = figures.fig_polymer_qc(pop.alleles, pop.dmaps(self.dims()), thr,
                                         qc_result=self.qc,
                                         bin_ids=pop.alleles.get('bin_ids'))
            eff = polymer.efficacy(self.qc['pos_um'])
            self._show(fig, 'polymer_qc',
                       tables={'efficacy_per_bin': eff,
                               'completeness_per_allele':
                                   polymer.completeness(self.qc['pos_um']),
                               'thresholds': pd.DataFrame([thr])},
                       params={'thresholds': thr,
                               'min_traced': self.panel.QcMinTracedSpinBox.value()})
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self.mw, 'Polymer QC', str(e))

    def dims(self):
        """'xyz' or 'xy' -- the distance every map, gate and histogram in
        this tab is computed from. ONE accessor, so no view can disagree
        with the gate that selected its cells."""
        return self.panel.DistanceDimsComboBox.currentData() or 'xyz'

    def _on_dims_changed(self):
        """The dimensionality changed, so every DERIVED number is stale.

        Same tier as Apply QC: it changes the numbers, not the view. That
        means the QC result and its thresholds must go -- they are
        micrometre lengths fitted to the other metric, and keeping them
        would gate bins in one dimensionality and display another. It
        does NOT rebuild the population: positions are dimension-free, so
        the expensive part (the store read, the expression cache) stays
        valid and only the distances downstream of it are recomputed.
        """
        self.qc = None
        self.qc_thresholds = None
        self.panel.QcStatusLabel.setText(
            f'QC cleared -- thresholds are {self.dims()}-specific and must '
            f'be re-derived')
        self._refresh_gate_summary()

    def _allele_state(self):
        """(dmaps, allele_index) honoring the Apply-QC toggle and the
        distance dimensionality -- the ONE funnel every map view uses."""
        pop = self._need_pop(alleles=True)
        if self.panel.ApplyQcCheckBox.isChecked():
            if self.qc is None:
                raise ValueError('Apply QC is on, but QC has not been run -- '
                                 'derive or preview it first')
            # the QC result stamps the dimensionality it was computed in;
            # refuse a mismatch rather than plot one metric under another
            if str(self.qc.get('dims', 'xyz')) != str(self.dims()):
                raise ValueError(
                    f"QC was run in {self.qc.get('dims')} but the distance "
                    f"setting is now {self.dims()} -- re-derive QC first")
            return self.qc['dmaps'], self.qc['index']
        dm = pop.dmaps(self.dims())
        return dm, np.arange(len(dm))

    def _qc_alleles(self):
        """The allele dict TRACED distance sources should read: the
        QC-filtered positions when Apply QC is on (so a traced distance
        and a map are computed from the same alleles), else the
        population's own. Returns None when nothing is filtered, which
        distances reads as 'use pop.alleles'."""
        pop = self._need_pop(alleles=True)
        if not self.panel.ApplyQcCheckBox.isChecked() or self.qc is None:
            return None
        idx = self.qc['index']
        al = pop.alleles
        out = {'pos_um': self.qc['pos_um'],
               'bin_hybes': list(al.get('bin_hybes') or []),
               'fov': np.asarray(al['fov'])[idx],
               'cell': np.asarray(al['cell'])[idx],
               'celltype': [al['celltype'][i] for i in idx]}
        return out

    # -- conditions --------------------------------------------------------
    OR_MARKER = '__or__'

    def condition(self):
        p = self.panel
        clauses, cur = [], []
        for i in range(p.ConditionListWidget.count()):
            d = p.ConditionListWidget.item(i).data(QtCore.Qt.UserRole)
            if d == self.OR_MARKER:
                if cur:
                    clauses.append(cur)
                cur = []
            else:
                cur.append(d)
        if cur:
            clauses.append(cur)
        return gate.Condition.from_dict({'clauses': clauses})

    def new_or_group(self):
        item = QtWidgets.QListWidgetItem('---------- OR ----------')
        item.setData(QtCore.Qt.UserRole, self.OR_MARKER)
        self.panel.ConditionListWidget.addItem(item)
        self._refresh_gate_summary()

    def gate_dict(self):
        """The gate as plain data -- what Save Config persists. The gate
        is a DESCRIPTION (clauses of predicate dicts); which cells pass
        is derived data, re-computed against whatever population is
        built, and deliberately never saved as state -- the Save Result
        sidecar snapshots membership at figure-save time instead."""
        return self.condition().to_dict()

    def set_gate_dict(self, d):
        """Rebuild the condition list from a saved gate (config load).
        Binds to nothing: predicates only touch data when mask(pop)
        runs, so restoring at app start cannot go stale."""
        p = self.panel
        p.ConditionListWidget.clear()
        for ci, clause in enumerate((d or {}).get('clauses') or []):
            if ci:
                item = QtWidgets.QListWidgetItem('---------- OR ----------')
                item.setData(QtCore.Qt.UserRole, self.OR_MARKER)
                p.ConditionListWidget.addItem(item)
            for pd in clause:
                pred = gate.Predicate.from_dict(pd)   # validate, or raise
                item = QtWidgets.QListWidgetItem(repr(pred))
                # store the NORMALIZED dict (to_dict of the parsed
                # predicate), so save->load->save is byte-stable
                item.setData(QtCore.Qt.UserRole, pred.to_dict())
                p.ConditionListWidget.addItem(item)
        self._refresh_gate_summary()

    def add_condition(self):
        try:
            d = self.panel.predicate_dict()
            pred = gate.Predicate.from_dict(d)
            item = QtWidgets.QListWidgetItem(repr(pred))
            item.setData(QtCore.Qt.UserRole, d)
            self.panel.ConditionListWidget.addItem(item)
            self._refresh_gate_summary()
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self.mw, 'Add Condition', str(e))

    def remove_condition(self):
        lw = self.panel.ConditionListWidget
        for item in lw.selectedItems():
            lw.takeItem(lw.row(item))
        self._refresh_gate_summary()

    def clear_conditions(self):
        self.panel.ConditionListWidget.clear()
        self._refresh_gate_summary()

    def _refresh_gate_summary(self, *_a):
        p = self.panel
        if self.pop is None:
            p.GateSummaryLabel.setText('gate: build the population first')
            return
        try:
            summary = report.gate_summary(self.pop, self.condition())
        except Exception as e:
            # this runs from slots (add/remove/toggle) -- an escaped
            # exception here is a qFatal abort, not a traceback
            p.GateSummaryLabel.setText(f'gate: {type(e).__name__}: {e}')
            return
        lines = [f"gated {summary['n_cells_gated']}/{summary['n_cells_total']} cells"]
        for r, n in summary['sequential']:
            lines.append(f'  {n:5d} after {r}')
        lines.append('  by celltype: ' + '  '.join(
            f'{k} {v[0]}/{v[1]}' for k, v in summary['by_celltype'].items()))
        p.GateSummaryLabel.setText('\n'.join(lines))

    # -- views -------------------------------------------------------------
    def _show(self, fig, name, tables=None, params=None, rebuild=None):
        """rebuild(bins) -> a new figure over the SAME rows, for the
        displayer's view-range panel. Presentation only: it re-renders,
        it never re-gates (per explicit requirement)."""
        d = AnalysisFigureDisplayer(title=f'Analysis -- {name}',
                                    parent=self.mw)
        d.set_figure(fig, name=name, tables=tables, pop=self.pop,
                     condition=self.condition(), params=params,
                     default_dir=self._default_dir(), rebuild=rebuild)
        d.show()
        d.raise_()
        self.displayers.append(d)
        self.displayer = d      # tests/tools peek at the latest

    def _default_dir(self):
        try:
            from codelab_pipeline.io import paths
            modality = next(iter(self.mw.hybe_records_by_modality or {}), None)
            sp = self.mw._storage_path_for_modality(modality)
            return paths.figure_dir(sp, 'analysis', 0) if sp else ''
        except Exception:
            return ''

    def _guard(self, fn):
        """EVERY exception contained. A slot that lets one escape is not a
        crash report -- PyQt turns it into qFatal/abort (0xC0000409, no
        traceback), which is exactly how a missing dict key in a figure
        label killed the whole app during the headless drive. ValueError
        stays the polite channel for precondition messages; anything else
        is a bug and says so, but in a dialog, not a core dump.
        """
        try:
            fn()
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self.mw, 'Analysis', str(e))
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self.mw, 'Analysis error', f'{type(e).__name__}: {e}')

    def _groups(self, allele_index):
        """The FLAG axis: {name: allele mask} for the current gate +
        celltype decomposition, over dmaps[allele_index]."""
        pop = self.pop
        # ALLELE-level gate: cell predicates project through the cell,
        # BarcodePresence applies per allele -- a heterozygous cell
        # passes while only its qualifying allele feeds the map.
        amask = self.condition().allele_mask(pop)[allele_index]
        traced = pop.alleles['n_traced'][allele_index] >= 2
        base = amask & traced
        if not self.panel.CelltypeDecomposeCheckBox.isChecked():
            return {'gated': base}
        cts = np.array(pop.alleles['celltype'])[allele_index]
        groups = {}
        for name in sorted(set(cts)):
            label = name if name else 'Unassigned'
            groups[label] = base & (cts == name)
        return groups or {'gated': base}

    def view_ensemble(self):
        def go():
            dm, idx = self._allele_state()
            groups = self._groups(idx)
            min_n = self.panel.MinNSpinBox.value()
            fig = figures.fig_ensemble(
                dm, group_masks=groups, title='ensemble distance map',
                min_n=min_n, bin_ids=self.pop.alleles.get('bin_ids'))
            tables = {}
            for name, gmask in groups.items():
                m, counts = ensemble.ensemble_map(dm, gmask, 'median', min_n)
                tables[f'map_{name}'] = m
                tables[f'counts_{name}'] = counts
            self._show(fig, 'ensemble_map', tables,
                       params={'min_n': min_n, 'qc_applied':
                               self.panel.ApplyQcCheckBox.isChecked(),
                               'qc_thresholds': self.qc_thresholds})
        self._guard(go)

    def view_fov_consistency(self):
        def go():
            dm, idx = self._allele_state()
            fovs = self.pop.alleles['fov'][idx]
            groups = self._groups(idx)
            fig = figures.fig_fov_consistency(
                dm, fovs, group_masks=groups,
                min_n=self.panel.MinNSpinBox.value(),
                show_maps=self.panel.ShowFovMapsCheckBox.isChecked(),
                bin_ids=self.pop.alleles.get('bin_ids'))
            tables = {}
            for name, gmask in groups.items():
                t = ensemble.fov_msd_test(dm, fovs, gmask)
                tables[f'msd_per_fov_{name}'] = pd.DataFrame(t['per_fov'])
            self._show(fig, 'fov_consistency', tables,
                       params={'qc_applied': self.panel.ApplyQcCheckBox.isChecked()})
        self._guard(go)

    def view_allele_difference(self):
        def go():
            dm, idx = self._allele_state()
            al = self.pop.alleles
            amask = self.condition().allele_mask(self.pop)[idx]
            fig = figures.fig_allele_difference(dm, al['fov'][idx],
                                                al['cell'][idx], amask)
            res = ensemble.allele_difference(dm, al['fov'][idx],
                                             al['cell'][idx], amask)
            self._show(fig, 'allele_difference',
                       {'within': res['within'], 'null': res['null']})
        self._guard(go)

    def _allele_cell_groups(self):
        """items 14: gated cells partitioned by allele-gate status.

        Cell gates narrow first; the allele-level predicates then label
        each gated cell: 'no gated allele' / 'mixed' / 'all gated'. The
        mode combo picks the view: pool everything, presence-vs-absence,
        or the full three-way decompose. Returns {name: cells DataFrame}.
        """
        pop = self.pop
        cond = self.condition()
        mode = self.panel.AlleleModeComboBox.currentIndex()
        has_allele_pred = any(getattr(p_, 'level', 'cell') == 'allele'
                              for p_ in cond.predicates)
        if mode == 0 or not has_allele_pred or pop.alleles is None:
            cmask = cond.mask(pop)
            return {'gated': pop.cells.loc[np.asarray(cmask, bool),
                                           ['fov', 'cell', 'celltype']]}
        # decompose modes base on the CELL-LEVEL predicates only (per
        # request): the full gate projects allele predicates onto cells,
        # which empties the Absence group BY CONSTRUCTION -- a cell
        # whose alleles all lack the barcode never passed the gate at
        # all. Cell gates narrow first; the allele gate then PARTITIONS,
        # absence included.
        cell_only = gate.Condition(clauses=[
            [p_ for p_ in clause if getattr(p_, 'level', 'cell') == 'cell']
            for clause in cond.clauses])
        cmask = cell_only.mask(pop)
        gated = pop.cells.loc[np.asarray(cmask, bool), ['fov', 'cell',
                                                        'celltype']]
        amask = cond.allele_mask(pop)
        al = pop.alleles
        adf = pd.DataFrame({'fov': al['fov'], 'cell': al['cell'],
                            'gated': amask,
                            'traced': al['n_traced'] >= 2})
        adf = adf[(adf['cell'] >= 0) & adf['traced']]
        counts = adf.groupby(['fov', 'cell'])['gated'].agg(['sum', 'count'])
        idx = pd.MultiIndex.from_frame(gated[['fov', 'cell']])
        got = counts.reindex(idx)
        n_g = got['sum'].fillna(0).to_numpy()
        n_t = got['count'].fillna(0).to_numpy()
        if mode == 1:
            return {'>=1 gated allele': gated[n_g >= 1],
                    'no gated allele': gated[n_g == 0]}
        return {'all alleles gated': gated[(n_t > 0) & (n_g == n_t)],
                'mixed': gated[(n_g >= 1) & (n_g < n_t)],
                'no gated allele': gated[n_g == 0]}

    def _rows_for_cells(self, table, cells_df):
        return table.merge(cells_df[['fov', 'cell']].drop_duplicates(),
                           on=['fov', 'cell'], how='inner')

    def view_expression_hist(self):
        def go():
            pop = self._need_pop(expr=True)
            src = self.panel.ViewExprSourcePicker.current()
            if src is None:
                raise ValueError('pick the expression source (section 3)')
            metric = self.panel.ViewExprMetricComboBox.currentText()
            # the view's OWN normalization (per request) -- same semantics
            # as the gate's: by_modality / by_source with its reference
            norm_mode = self.panel.ViewExprNormComboBox.currentText()
            normalize = None
            if norm_mode == 'by_modality':
                normalize = ['by_modality']
            elif norm_mode == 'by_source':
                ref = self.panel.ViewExprNormRefPicker.current()
                if ref is None:
                    raise ValueError("normalize 'by_source' divides by the "
                                     'same metric of the norm reference -- '
                                     'pick it (section 3)')
                normalize = ['by_source', list(ref)]
            table = pop.expression
            metric_col = metric
            if normalize:
                ref_t = (tuple(normalize[1]) if len(normalize) > 1 else None)
                table = expression.normalize(table, metric, normalize[0],
                                             ref_source=ref_t)
                metric_col = f'{metric}_norm'
            label = self._source_label(src)
            nl = self._norm_label(normalize)
            if nl:
                label = f'{label}\n{nl}'
            groups = self._allele_cell_groups()
            import matplotlib.pyplot as plt
            per_ct = self.panel.CelltypeDecomposeCheckBox.isChecked()

            def _render(bins=None):
                # SAME rows, different binning -- presentation only, so
                # the displayer's view-range panel can re-bin without
                # touching the gate or the prepared data
                n = len(groups)
                f, axes = plt.subplots(1, n, figsize=(5.6 * n, 4),
                                       squeeze=False)
                for ax, (gname, cells_df) in zip(axes[0], groups.items()):
                    figures.expression_hist_ax(
                        ax, self._rows_for_cells(table, cells_df),
                        src, metric_col, label=label,
                        bins=bins or 30, per_celltype=per_ct)
                    ax.set_title(f'{gname} ({len(cells_df)} cells)\n'
                                 + ax.get_title(), fontsize=9)
                return f

            fig = _render()
            self._show(fig, 'expression_hist',
                       {'expression': table}, rebuild=_render,
                       params={'source': list(src), 'metric': metric,
                               'normalize': normalize,
                               'allele_mode':
                               self.panel.AlleleModeComboBox.currentText()})
        self._guard(go)

    def view_brightness_vs_count(self):
        def go():
            pop = self._need_pop(expr=True)
            src = self.panel.ViewExprSourcePicker.current()
            if src is None:
                raise ValueError('pick the expression source (section 3)')
            fig = figures.fig_brightness_vs_count(
                pop.expression, src, source_label=self._source_label(src))
            self._show(fig, 'brightness_vs_count',
                       {'expression': pop.expression},
                       params={'source': list(src)})
        self._guard(go)

    def view_distance_hist(self):
        def go():
            pop = self._need_pop(spots=True)
            a = self.panel.ViewDistSourceAPicker.current()
            b = self.panel.ViewDistSourceBPicker.current()
            if a is None or b is None:
                raise ValueError('pick distance sources A and B (section 3)')
            collapse = self.panel.ViewDistCollapseComboBox.currentText()
            per_ct = self.panel.CelltypeDecomposeCheckBox.isChecked()
            groups = self._allele_cell_groups()
            # traced sources read the SAME alleles the views use, so
            # Apply QC governs them exactly as it governs the maps
            alleles = None
            if (distances.is_traced_source(a) or distances.is_traced_source(b)):
                alleles = self._qc_alleles()
            pairs = distances.pair_distances(pop, a, b, alleles, dims=self.dims())
            import matplotlib.pyplot as plt

            def _render(bins=None):
                # SAME pairs, different binning -- presentation only
                f, axes2 = plt.subplots(1, len(groups),
                                        figsize=(5.6 * len(groups), 4),
                                        squeeze=False)
                for ax, (gname, cells_df) in zip(axes2[0], groups.items()):
                    sub = self._rows_for_cells(pairs, cells_df)
                    if collapse != 'all' and len(sub):
                        agg = {'median': 'median', 'min': 'min'}[collapse]
                        sub = sub.groupby(['fov', 'cell'], as_index=False)                             .agg({'d_um': agg, 'celltype': 'first'})
                    figures.distance_hist_ax(ax, sub, bins=bins or 40,
                                             per_celltype=per_ct)
                    unit = 'pairs' if collapse == 'all' else 'cells'
                    ax.set_title(f'{gname} ({len(cells_df)} cells, '
                                 f'{len(sub)} {unit}, {collapse})', fontsize=9)
                f.suptitle(f'{self._source_label(a)} to '
                           f'{self._source_label(b)} within gated cells',
                           fontsize=11)
                return f

            fig = _render()
            self._show(fig, 'distance_hist', {'pairs': pairs}, rebuild=_render,
                       params={'source_a': list(a), 'source_b': list(b),
                               'collapse': collapse,
                               'allele_mode':
                               self.panel.AlleleModeComboBox.currentText()})
        self._guard(go)

    def view_repeat_toe_qc(self):
        def go():
            pop = self._need_pop(alleles=True)
            fig = figures.fig_repeat_toe_qc(pop, dims=self.dims())
            self._show(fig, 'repeat_toe_qc',
                       params={'dims': self.dims()})
        self._guard(go)

    def view_isotropy(self):
        """Is Z as trustworthy as X and Y?

        Deliberately NOT routed through self.dims(): this needs the same
        pairs measured BOTH ways, and in XY mode a dims-following version
        would compare XY against XY, draw a perfect diagonal and report
        isotropy on data whose Z is badly inflated -- the QC would hide
        the very problem it exists to find.

        Reads the QC-filtered positions when Apply QC is on, so the
        answer describes the alleles the maps are actually built from.
        """
        def go():
            pop = self._need_pop(alleles=True)
            qa = self._qc_alleles()
            pos = (qa['pos_um'] if qa is not None
                   else pop.alleles['pos_um'])
            fig = figures.fig_anisotropy(pos, title='isotropy QC')
            self._show(fig, 'isotropy_qc',
                       params={'measured_in': 'xy and xyz (both, always)',
                               'apply_qc': self.panel.ApplyQcCheckBox.isChecked()})
        self._guard(go)

    def preview_histogram(self):
        """The gate-range picker's view: the distribution the range gates,
        with the currently typed range drawn on it."""
        def go():
            kind = self.panel.PredicateKindComboBox.currentText()
            lo, hi = self.panel.range_values()
            picked = None if lo is None and hi is None else (
                lo if lo is not None else -np.inf,
                hi if hi is not None else np.inf)
            if kind == 'ExpressionRange':
                pop = self._need_pop(expr=True)
                src = self.panel.SourceAPicker.current()
                if src is None:
                    raise ValueError('pick source A')
                # the preview must show the DISTRIBUTION THE GATE READS:
                # harvesting through predicate_dict keeps normalization
                # identical to the gate's (previewing raw while gating
                # normalized draws the range on the wrong axis)
                d = self.panel.predicate_dict()
                table = pop.expression
                metric = d['metric']
                if d.get('normalize'):
                    from codelab_pipeline.analysis import expression as E
                    ref = (tuple(d['normalize'][1])
                           if len(d['normalize']) > 1 else None)
                    table = E.normalize(table, metric, d['normalize'][0],
                                        ref_source=ref)
                    metric = f'{metric}_norm'
                fig = figures.fig_expression_hist(
                    table, src, metric, picked_range=picked,
                    source_label=self._source_label(src),
                    norm_label=self._norm_label(d.get('normalize')))
                self._show(fig, 'gate_range_preview',
                           params={'source': list(src),
                                   'normalize': d.get('normalize')})
            elif kind == 'PairDistanceRange':
                pop = self._need_pop(spots=True)
                a = self.panel.SourceAPicker.current()
                b = self.panel.SourceBPicker.current()
                if a is None or b is None:
                    raise ValueError('pick sources A and B')
                per_cell = distances.pair_distance_per_cell(pop, a, b, dims=self.dims())
                vals = per_cell.to_numpy()
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(5.6, 4))
                ax.hist(vals[np.isfinite(vals)], bins=40)
                if picked:
                    ax.axvspan(picked[0], picked[1], color='gold', alpha=0.25)
                ax.set_xlabel('per-cell median pair distance (um)')
                ax.set_ylabel('cells')
                ax.set_title(f'{a[1]}-{b[1]} per-cell distance', fontsize=10)
                self._show(fig, 'gate_range_preview',
                           params={'source_a': list(a), 'source_b': list(b)})
            else:
                raise ValueError('range preview applies to ExpressionRange '
                                 'and PairDistanceRange')
        self._guard(go)
