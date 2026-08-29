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
        self.displayer = AnalysisFigureDisplayer(parent=mw)
        p = self.panel
        p.BuildPopulationPushButton.clicked.connect(self.build_population)
        p.DeriveQcPushButton.clicked.connect(self.derive_qc)
        p.PreviewQcPushButton.clicked.connect(self.preview_qc)
        p.ApplyQcCheckBox.toggled.connect(self._refresh_gate_summary)
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

    # -- population --------------------------------------------------------
    def populate_sources(self):
        """Fill the source list + combos from the layouts' hybe records.
        Called by MainWindow after layouts parse."""
        p = self.panel
        p.SourceListWidget.clear()
        p.SourceAComboBox.clear()
        p.SourceBComboBox.clear()
        for modality, records in (self.mw.hybe_records_by_modality or {}).items():
            for r in records:
                fid = r.get('fiducial_channel')
                for ch in r.get('channels', []):
                    label = f'{modality} | {r["folder"]} | ch{ch}' + (
                        '  (fiducial)' if ch == fid else '')
                    src = (modality, r['folder'], int(ch))
                    item = QtWidgets.QListWidgetItem(label)
                    item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                    item.setCheckState(QtCore.Qt.Unchecked)
                    item.setData(QtCore.Qt.UserRole, list(src))
                    p.SourceListWidget.addItem(item)
                    p.SourceAComboBox.addItem(label, list(src))
                    p.SourceBComboBox.addItem(label, list(src))

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
                storage_path, fovs, records=records,
                sources=sources or None, spot_sources=sources or None,
                voxel_um=voxel, mask_intensity=mask_int,
                resolvers=resolvers or None)

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
            thr = polymer.qc_thresholds(pop.alleles, pop.dmaps())
            self.panel.set_qc_thresholds(thr)
            self._run_qc(thr)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self.mw, 'Polymer QC', str(e))

    def _run_qc(self, thr):
        pop = self.pop
        out = polymer.apply_qc(pop.alleles, pop.dmaps(), thr,
                               min_traced=self.panel.QcMinTracedSpinBox.value())
        self.qc = out
        self.qc_thresholds = thr
        kept = int(out['kept'].sum())
        self.panel.QcStatusLabel.setText(
            f'QC: {kept}/{len(pop.alleles["amp"])} alleles kept, '
            f'{int(out["bads"].sum())} bins removed  |  '
            + '  '.join(f'{k}={v:.3g}' for k, v in thr.items()))
        self._refresh_gate_summary()

    def preview_qc(self):
        try:
            pop = self._need_pop(alleles=True)
            thr = self.panel.qc_threshold_values()
            if thr is None:
                raise ValueError('derive (or type) the thresholds first')
            self._run_qc(thr)
            fig = figures.fig_polymer_qc(pop.alleles, pop.dmaps(), thr,
                                         qc_result=self.qc)
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

    def _allele_state(self):
        """(dmaps, allele_index) honoring the Apply-QC toggle."""
        pop = self._need_pop(alleles=True)
        if self.panel.ApplyQcCheckBox.isChecked():
            if self.qc is None:
                raise ValueError('Apply QC is on, but QC has not been run -- '
                                 'derive or preview it first')
            return self.qc['dmaps'], self.qc['index']
        dm = pop.dmaps()
        return dm, np.arange(len(dm))

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
    def _show(self, fig, name, tables=None, params=None):
        self.displayer.set_figure(
            fig, name=name, tables=tables, pop=self.pop,
            condition=self.condition(), params=params,
            default_dir=self._default_dir())
        self.displayer.show()
        self.displayer.raise_()

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
            fig = figures.fig_ensemble(dm, group_masks=groups,
                                       title='ensemble distance map',
                                       min_n=min_n)
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
            fig = figures.fig_fov_consistency(dm, fovs, group_masks=groups,
                                              min_n=self.panel.MinNSpinBox.value())
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

    def view_expression_hist(self):
        def go():
            pop = self._need_pop(expr=True)
            src = self.panel.combo_source(self.panel.SourceAComboBox)
            if src is None:
                raise ValueError('pick source A')
            metric = self.panel.MetricComboBox.currentText()
            lo, hi = self.panel.range_values()
            picked = (lo, hi) if lo is not None and hi is not None else None
            fig = figures.fig_expression_hist(
                pop.expression, src, metric,
                per_celltype=self.panel.CelltypeDecomposeCheckBox.isChecked(),
                picked_range=picked)
            self._show(fig, 'expression_hist',
                       {'expression': pop.expression},
                       params={'source': list(src), 'metric': metric})
        self._guard(go)

    def view_brightness_vs_count(self):
        def go():
            pop = self._need_pop(expr=True)
            src = self.panel.combo_source(self.panel.SourceAComboBox)
            if src is None:
                raise ValueError('pick source A')
            fig = figures.fig_brightness_vs_count(pop.expression, src)
            self._show(fig, 'brightness_vs_count',
                       {'expression': pop.expression},
                       params={'source': list(src)})
        self._guard(go)

    def view_distance_hist(self):
        def go():
            pop = self._need_pop(spots=True)
            a = self.panel.combo_source(self.panel.SourceAComboBox)
            b = self.panel.combo_source(self.panel.SourceBComboBox)
            if a is None or b is None:
                raise ValueError('pick sources A and B')
            mask = self.condition().mask(pop)
            per_ct = self.panel.CelltypeDecomposeCheckBox.isChecked()
            h = distances.distance_histogram(pop, a, b, mask=mask,
                                             per_celltype=per_ct)
            fig = figures.fig_distance_hist(
                h, f'{a[1]}({a[0]}) to {b[1]}({b[0]}) within gated cells')
            pairs = distances.pair_distances(pop, a, b)
            self._show(fig, 'distance_hist', {'pairs': pairs},
                       params={'source_a': list(a), 'source_b': list(b)})
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
                src = self.panel.combo_source(self.panel.SourceAComboBox)
                if src is None:
                    raise ValueError('pick source A')
                fig = figures.fig_expression_hist(
                    pop.expression, src, self.panel.MetricComboBox.currentText(),
                    picked_range=picked)
                self._show(fig, 'gate_range_preview',
                           params={'source': list(src)})
            elif kind == 'PairDistanceRange':
                pop = self._need_pop(spots=True)
                a = self.panel.combo_source(self.panel.SourceAComboBox)
                b = self.panel.combo_source(self.panel.SourceBComboBox)
                if a is None or b is None:
                    raise ValueError('pick sources A and B')
                per_cell = distances.pair_distance_per_cell(pop, a, b)
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
