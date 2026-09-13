"""
The Cell Cycle tab's wiring: widget values -> codelab_pipeline/analysis/
cellcycle -> the store and the figure displayer.

A separate module like analysis_wiring.py, for the same reason: every
computation below is a toolbox call that runs identically headless,
and MainWindow only instantiates CellCycleWiring(self). Counting, the
fit and the heavier figures run inside FnWorker threads; count_table's
per-FOV store reads fan out to child processes (pmap) inside.

What the stage persists (design 7, store contract of 2026-09-13):
  analysis/fov###/cellcycle.json   every placed cell of the FOV --
                                   theta_deg, R, fit_z, bf_ring,
                                   radius, total
  analysis/cellcycle_model.json    the one model with its provenance
                                   (sources -> genes, roles, proxy,
                                   alpha, training celltypes, bridges)
                                   and the user's category arcs and
                                   verdict gates
The category itself is never stored per cell: population.py derives it
from the arcs and gates when placements are read, so re-drawing a
boundary is a re-read, not a refit.
"""
import datetime
import json
import os

import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtWidgets

from codelab_pipeline.analysis import cellcycle as CC
from codelab_pipeline.analysis import figures_cellcycle as FC
from codelab_pipeline.io import analysis_store, paths
from canvas.analysis_figure_displayer import AnalysisFigureDisplayer

TAB_TITLE = 'Cell Cycle'


class CellCycleWiring(QtCore.QObject):
    def __init__(self, mw):
        super().__init__(mw)
        self.mw = mw
        self.panel = mw.ui.CellCyclePanel
        self.tidy = None          # count_table's tidy rows
        self.names = None         # {(modality, hybe, channel): gene}
        self.roles = None         # {gene: role}
        self.model = None         # CycleModel
        self.spec = None          # the model file payload
        self.placed = None        # DataFrame: fov, cell, celltype, theta_deg, R, fit_z, bf_ring, radius, total
        self.q = None             # (n, T) posteriors of the placed cells
        self.X = None             # (n, G) counts of the placed cells
        self.displayers = []
        self.n_usable = 0        # cells reaching the minimum panel total, from the last count
        p = self.panel
        p.BuildCountsPushButton.clicked.connect(lambda: self._guard(self.build_counts))
        p.ExportCountsPushButton.clicked.connect(lambda: self._guard(self.export_counts))
        p.AlphaHeldOutPushButton.clicked.connect(lambda: self._guard(self.alpha_held_out))
        p.AddBridgePushButton.clicked.connect(lambda: self._guard(self.add_bridge))
        p.RemoveBridgePushButton.clicked.connect(self.remove_bridge)
        p.FitPushButton.clicked.connect(lambda: self._guard(self.fit))
        p.PlacePushButton.clicked.connect(lambda: self._guard(self.place))
        p.LoadModelPushButton.clicked.connect(lambda: self._guard(self.load_model))
        p.SaveModelPushButton.clicked.connect(lambda: self._guard(self.save_model))
        p.SpectrumPushButton.clicked.connect(lambda: self._guard(self.view_spectrum))
        p.ProfilesPushButton.clicked.connect(lambda: self._guard(self.view_profiles))
        p.EmbeddingsPushButton.clicked.connect(lambda: self._guard(self.view_embeddings))
        p.VerdictsPushButton.clicked.connect(lambda: self._guard(self.view_verdicts))
        p.ContributionPushButton.clicked.connect(lambda: self._guard(self.view_contribution))
        p.GroupTablePushButton.clicked.connect(lambda: self._guard(self.view_groups))
        p.HalfPanelPushButton.clicked.connect(lambda: self._guard(self.view_half_panel))
        p.CycleTimePushButton.clicked.connect(lambda: self._guard(self.view_cycle_time))
        p.ProposeArcsPushButton.clicked.connect(lambda: self._guard(self.propose_arcs))
        p.ProposeFromTimePushButton.clicked.connect(lambda: self._guard(self.propose_arcs_from_time))
        p.FovOverlayPushButton.clicked.connect(lambda: self._guard(self.view_fov_overlay))
        p.DapiPushButton.clicked.connect(lambda: self._guard(self.view_dapi))
        p.DapiGalleryPushButton.clicked.connect(lambda: self._guard(self.view_dapi_gallery))
        p.SetOriginPushButton.clicked.connect(lambda: self._guard(self.set_origin))
        p.ProposeFromDapiPushButton.clicked.connect(lambda: self._guard(self.propose_arcs_from_dapi))
        p.ApplyCategoriesPushButton.clicked.connect(lambda: self._guard(self.apply_categories))
        mw.ui.tabWidget.currentChanged.connect(self._on_tab_changed)

    # -- plumbing -----------------------------------------------------------

    def _guard(self, fn):
        """Every exception contained (see analysis_wiring._guard: an
        escaping one is a qFatal, not a traceback)."""
        try:
            fn()
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self.mw, TAB_TITLE, str(e))
        except Exception as e:                                  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self.mw, f'{TAB_TITLE} error', f'{type(e).__name__}: {e}')

    def _start(self, fn, ok, fail):
        from windows.main_window import FnWorker
        self._worker = FnWorker(fn)
        self._worker.finished_ok.connect(lambda r: self._guard(lambda: ok(r)))
        self._worker.failed.connect(fail)
        self._worker.start()

    def _on_tab_changed(self, index):
        tabs = self.mw.ui.tabWidget
        if tabs.tabText(index) != TAB_TITLE:
            return
        self.panel.set_celltype_names(list(self.mw.current_celltype_list or []))
        if self.model is None:
            try:
                self._restore_from_store()
            except Exception as e:                              # noqa: BLE001
                self.mw.log(f'{TAB_TITLE}: stored model not restored: {type(e).__name__}: {e}')

    def _storage(self):
        modality = next(iter(self.mw.hybe_records_by_modality or {}), None)
        sp = self.mw._storage_path_for_modality(modality) if modality else None
        if not sp:
            raise ValueError('No storage path -- set up Ingestion first.')
        return sp

    def _experiment_name(self):
        return os.path.basename(os.path.normpath(paths.project_root(self._storage())))

    def _fovs(self):
        text = self.panel.FovListLineEdit.text().strip()
        if not text:
            text = self.mw.ui.IngestionPanel.FovListLineEdit.text()
        fovs = self.mw._parse_fov_list(text)
        if not fovs:
            raise ValueError('No FOVs.')
        return [int(f) for f in fovs]

    def _log(self, msg):
        self.mw.log(f'{TAB_TITLE}: {msg}')

    def populate_sources(self):
        """Called by MainWindow after the layouts parse."""
        self.panel.populate_sources(self.mw.hybe_records_by_modality or {})

    # -- 1. counts --------------------------------------------------------------

    def _panel_genes(self):
        rows = self.panel.checked_genes()
        if len(rows) < 3:
            raise ValueError('Check at least three sources as genes in section 1.')
        genes = [g for _s, g, _r in rows]
        dup = {g for g in genes if genes.count(g) > 1}
        if dup:
            raise ValueError(f'One gene name per source: {sorted(dup)} appear more than once.')
        names = {tuple(s): g for s, g, _r in rows}
        roles = {g: r for _s, g, r in rows}
        return names, roles

    def build_counts(self):
        p = self.panel
        sp = self._storage()
        fovs = self._fovs()
        names, roles = self._panel_genes()
        sources = list(names)
        p.BuildCountsPushButton.setEnabled(False)
        p.CountStatusLabel.setText(f'counting {len(sources)} sources over {len(fovs)} FOVs...')

        def _compute():
            return CC.count_table(sp, fovs, sources, gate=CC.COUNT_GATE)

        def _done(res):
            tidy, fails = res
            p.BuildCountsPushButton.setEnabled(True)
            self.tidy, self.names, self.roles = tidy, names, roles
            n_cells = tidy.groupby(['fov', 'cell']).ngroups if len(tidy) else 0
            notes = []
            if fails:
                notes.append(f'{len(fails)} FOV(s) FAILED: {fails[0][1]}')
            # COVERAGE, said plainly: a store with no RNA spots for the
            # panel still yields a table (every count 0), and a fit on
            # the handful of cells that happen to clear the minimum is
            # numerically fine and scientifically nothing (seen: 86 of
            # 2853 cells, evidence -0.000). Count the empty (FOV, source)
            # slices and the cells that reach the minimum, and say so.
            slices = tidy.groupby(['fov', 'hybe', 'channel'])['n_stored'].first() if len(tidy) else pd.Series(dtype=float)
            n_empty, n_slices = int((slices == 0).sum()), int(len(slices))
            self.n_usable = 0
            if len(tidy):
                table, _ct = CC.gene_table(tidy, names, metric=p.proxy_metric())
                self.n_usable = int((table.sum(axis=1) >= p.MinTotalSpinBox.value()).sum())
            if n_empty:
                empty = sorted({h for h, n in zip(tidy['hybe'], tidy['n_stored']) if n == 0})
                notes.append(f'{n_empty} of {n_slices} (FOV, source) slices hold NO stored spots ({empty[:6]}'
                             f'{"..." if len(empty) > 6 else ""}) -- run Spot Localization for those hybes first')
            notes.append(f'{self.n_usable} cells reach the minimum panel total of {p.MinTotalSpinBox.value()}')
            pmin = tidy.groupby('hybe')['p_min'].min() if len(tidy) else pd.Series(dtype=float)
            # a store gated AT 0.5 holds p_exist values from 0.500x up; only
            # a clear margin above the count gate means a stricter store gate
            above = sorted(pmin[pmin > CC.COUNT_GATE + 0.02].index)
            if above:
                notes.append(f'{len(above)} source(s) were stored with a p_exist gate above {CC.COUNT_GATE}: '
                             f'their count is the store\'s gate ({above[:6]})')
            head = f'{n_cells} cells x {len(sources)} sources over {len(fovs)} FOVs'
            if n_slices and n_empty > 0.5 * n_slices:
                head = 'NOT USABLE FOR A FIT: ' + head
            p.CountStatusLabel.setText(head + (' | ' + ' | '.join(notes) if notes else ''))
            self._log(p.CountStatusLabel.text())

        def _fail(msg):
            p.BuildCountsPushButton.setEnabled(True)
            p.CountStatusLabel.setText(f'FAILED: {msg}')

        self._start(_compute, _done, _fail)

    def _dataset(self, names=None):
        """(X, genes, celltype, keys) of every cell with at least the
        minimum panel total, from the built counts."""
        if self.tidy is None:
            raise ValueError('Build counts first (section 1).')
        names = names or self.names
        metric = self.panel.proxy_metric()
        table, celltype = CC.gene_table(self.tidy, names, metric=metric)
        keep = table.sum(axis=1) >= self.panel.MinTotalSpinBox.value()
        table, celltype = table[keep], celltype[keep]
        if len(table) == 0:
            raise ValueError('No cell reaches the minimum panel total.')
        return (table.to_numpy(float), list(table.columns),
                celltype.to_numpy().astype(object), table.index)

    def export_counts(self):
        X, genes, ct, keys = self._dataset()
        name = self._experiment_name()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.mw, 'Export counts', os.path.join(self._default_dir(), f'{name}_cellcycle_counts.json'), 'JSON (*.json)')
        if not path:
            return
        payload = {'version': 1, 'name': name, 'genes': genes, 'X': X.astype(int).tolist(),
                   'groups': [str(c) for c in ct], 'alpha': float(self.panel.AlphaSpinBox.value()),
                   'roles': self.roles, 'train_celltypes': self.panel.training_celltypes(),
                   'proxy': self.panel.proxy_metric(), 'fov': [int(f) for f, _c in keys],
                   'cell': [int(c) for _f, c in keys]}
        with open(path, 'w') as f:
            json.dump(payload, f)
        self._log(f'exported {len(X)} cells x {len(genes)} genes to {path}')

    # -- 2. model ----------------------------------------------------------------

    def _role_lists(self, roles):
        early = [g for g, r in roles.items() if r == 'S']
        late = [g for g, r in roles.items() if r == 'G2/M']
        hk = [g for g, r in roles.items() if r == 'housekeeping'] + list(CC.HOUSEKEEPING)
        return early, late, hk

    @staticmethod
    def load_bridge(path):
        """An exported counts file -> a Dataset of its training cells."""
        with open(path) as f:
            d = json.load(f)
        X = np.asarray(d['X'], float)
        groups = np.asarray(d.get('groups', ['all'] * len(X)), dtype=object)
        train = d.get('train_celltypes') or []
        k = np.isin(groups, train) if train else np.ones(len(X), bool)
        if k.sum() < 20:
            raise ValueError(f'{os.path.basename(path)}: only {int(k.sum())} training cells')
        return CC.Dataset(d['name'], X[k], d['genes'], groups[k], alpha=d.get('alpha', CC.DEFAULT_ALPHA)), d

    def add_bridge(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self.mw, 'Add exported counts', self._default_dir(), 'JSON (*.json)')
        if not path:
            return
        ds, d = self.load_bridge(path)
        self.panel.BridgeListWidget.addItem(path)
        self._log(f'bridge {ds.name}: {len(ds.X)} training cells x {len(ds.genes)} genes '
                  f'({", ".join(ds.genes[:8])}{"..." if len(ds.genes) > 8 else ""})')

    def remove_bridge(self):
        lw = self.panel.BridgeListWidget
        for item in lw.selectedItems():
            lw.takeItem(lw.row(item))

    def alpha_held_out(self):
        X, genes, ct, keys = self._dataset()
        train = self.panel.training_celltypes()
        k = np.isin(ct, train) if train else np.ones(len(X), bool)
        Xt = X[k]
        p = self.panel
        p.AlphaHeldOutPushButton.setEnabled(False)
        p.ModelStatusLabel.setText(f'held-out evidence over alpha on {len(Xt)} training cells...')

        def _compute():
            return CC.select_alpha(Xt)

        def _done(scores):
            # {alpha: held-out mean evidence}, None = multinomial
            p.AlphaHeldOutPushButton.setEnabled(True)
            best = max(scores, key=lambda a: scores[a])
            table = ', '.join(f'{"multinomial" if a is None else f"{a:g}"}: {v:.4f}' for a, v in scores.items())
            if best is None:
                p.ModelStatusLabel.setText(f'held-out evidence prefers the multinomial (no over-dispersion); alpha left as is [{table}]')
            else:
                p.AlphaSpinBox.setValue(float(best))
                p.ModelStatusLabel.setText(f'alpha by held-out evidence: {best:g} [{table}]')
            self._log(p.ModelStatusLabel.text())

        def _fail(msg):
            p.AlphaHeldOutPushButton.setEnabled(True)
            p.ModelStatusLabel.setText(f'FAILED: {msg}')

        self._start(_compute, _done, _fail)

    def fit(self):
        p = self.panel
        sp = self._storage()
        fovs = self._fovs()
        X, genes, ct, keys = self._dataset()
        name = self._experiment_name()
        train = p.training_celltypes()
        k = np.isin(ct, train) if train else np.ones(len(X), bool)
        slices = self.tidy.groupby(['fov', 'hybe', 'channel'])['n_stored'].first()
        n_empty = int((slices == 0).sum())
        if n_empty > 0.5 * len(slices):
            raise ValueError(f'{n_empty} of {len(slices)} (FOV, source) slices hold no stored spots: the store has not '
                             "been localized for this panel. Run Spot Localization for the panel hybes first.")
        if k.sum() < 100:
            raise ValueError(f'Only {int(k.sum())} training cells reach the minimum panel total -- too few for a 72-point '
                             f'circle. Check the training celltypes, the minimum total, and that the store holds spots '
                             f'for every panel hybe.')
        alpha = float(p.AlphaSpinBox.value())
        early, late, hk = self._role_lists(self.roles)
        bridges = []
        for path in p.bridges():
            ds, _d = self.load_bridge(path)
            bridges.append(ds)
        datasets = [CC.Dataset(name, X[k], genes, ct[k], alpha=alpha)] + bridges
        orient_by = (early, late) if early and late else None
        if orient_by is None:
            self._log('no S and G2/M roles set: the circle keeps an arbitrary orientation and the bridges choose the reflection')
        roles, names_map, metric, min_total = dict(self.roles), dict(self.names), p.proxy_metric(), p.MinTotalSpinBox.value()
        gates, birth = p.gates(), float(p.BirthDegSpinBox.value())
        origin_choice = int(p.OriginComboBox.currentIndex())        # 0 DAPI, 1 total drop, 2 S peak
        origin_division = origin_choice in (0, 1)
        dapi_src = p.dapi_source() if origin_choice == 0 else None
        if origin_choice == 0 and dapi_src is None:
            self._log('origin: no DAPI source in the layouts -- falling back to the panel-total drop')
        fov_of = np.array([int(f) for f, _c in keys])
        cell_of = np.array([int(c) for _f, c in keys])
        shares = p.phase_shares()
        p.FitPushButton.setEnabled(False)
        p.ModelStatusLabel.setText(f'fitting on {int(k.sum())} training cells' + (f' with {len(bridges)} bridge(s)' if bridges else '') + '...')

        def _compute():
            m = CC.CycleModel().fit(datasets, bridge_exclude=hk, orient_by=orient_by)
            if orient_by is not None:
                m.orient(early, late)
            origin, birth_used, dapi_tab = self._settle_origin(m, name, X[k], fov_of[k], cell_of[k], sp,
                                                               origin_division, dapi_src, birth)
            placed = m.place(name, X)
            # the arcs are RE-PROPOSED after every fit, never carried over:
            # a fit can move the frame (the origin, a bridge), and arcs
            # drawn in the old frame would silently name the wrong cells
            # (seen: the time arcs of an S-peak frame applied to a
            # division frame put half the cells in G2/M). DAPI when it was
            # measured, else the educated guess from cycle time; the roles
            # can replace either, and Apply stores the final choice.
            arcs, arcs_from = self._propose_after(m, name, X, k, placed, fov_of, cell_of, dapi_tab, birth_used, shares)
            spec = {'version': 1, 'model': m.to_dict(), 'experiment': name,
                    'sources': {CC.source_key(s): g for s, g in names_map.items()},
                    'roles': roles, 'proxy': metric, 'gate': CC.COUNT_GATE, 'alpha': alpha,
                    'train_celltypes': list(train), 'fovs': fovs, 'min_total': int(min_total),
                    'n_train': int(k.sum()), 'n_placed': int(len(X)),
                    'bridges': [d.name for d in bridges], 'orient_by': [early, late],
                    'align_report': {f'{a}->{b}': {kk: vv for kk, vv in r.items() if kk != 'residual_deg'}
                                     for (a, b), r in m.align_report.items()},
                    'fitted_at': datetime.datetime.now().isoformat(timespec='seconds'),
                    'origin': origin, 'categories_from': arcs_from,
                    'categories': arcs, 'gates': gates, 'birth_deg': birth_used}
            self._persist(sp, spec, placed, keys, X)
            return m, placed, spec

        def _done(res):
            m, placed, spec = res
            p.FitPushButton.setEnabled(True)
            self._adopt(m, spec, placed, keys, ct, X)
            rep = '; '.join(f'{k_}: flip {v.get("flip")} shift {v.get("shift_deg", float("nan")):.0f} '
                            f'residual {v.get("err_deg", float("nan")):.0f} deg'
                            for k_, v in spec['align_report'].items() if 'flip' in v)
            o = spec.get('origin') or {}
            otxt = o.get('mode', '')
            if 'drop_factor' in o and o.get('drop_factor') is not None:
                otxt += f' (drop x{o["drop_factor"]:.1f}' + (f' at {o["angle_before_deg"]:.0f} deg before the rotation)' if 'angle_before_deg' in o else ')')
            p.ModelStatusLabel.setText(
                f'fitted {spec["fitted_at"]}: {spec["n_train"]} training cells, {spec["n_placed"]} placed over '
                f'{len(fovs)} FOVs; alpha {alpha:g}; evidence {m.history[-1]:.3f}; origin {otxt}'
                + (f'; categories proposed from {spec["categories_from"]}' if spec.get('categories_from') else '')
                + (f'; bridges {rep}' if rep else ''))
            self._log(p.ModelStatusLabel.text())

        def _fail(msg):
            p.FitPushButton.setEnabled(True)
            p.ModelStatusLabel.setText(f'FAILED: {msg}')

        self._start(_compute, _done, _fail)

    def _persist(self, sp, spec, placed, keys, X):
        """The per-FOV capsules and the model file, written atomically by
        the store. Runs in the worker (no widgets)."""
        fov_of = np.array([int(f) for f, _c in keys])
        cell_of = np.array([int(c) for _f, c in keys])
        totals = X.sum(1)
        for fov in sorted(set(fov_of.tolist())):
            k = fov_of == fov
            sub = {key: np.asarray(v)[k] for key, v in placed.items() if key != 'posterior'}
            rows = CC.capsule_rows(sub, cell_of[k], totals[k])
            analysis_store.write_fov_cellcycle(sp, fov, {
                'version': CC.CAPSULE_VERSION, 'model': spec['fitted_at'],
                'stamp': analysis_store.fov_input_stamp(sp, fov), 'rows': rows})
        analysis_store.write_cellcycle_model(sp, spec)

    def _adopt(self, m, spec, placed, keys, ct, X):
        self.model, self.spec, self.X = m, spec, X
        self.q = placed['posterior']
        self.placed = pd.DataFrame({
            'fov': [int(f) for f, _c in keys], 'cell': [int(c) for _f, c in keys],
            'celltype': [str(c) for c in ct],
            'theta_deg': np.degrees(placed['theta']) % 360.0, 'R': placed['R'],
            'fit_z': placed['fit_z'], 'bf_ring': placed['bf_ring'], 'radius': placed['radius'],
            'total': X.sum(1)})
        self.panel.set_arcs(spec.get('categories') or [])
        self.panel.set_gates(spec.get('gates') or {})
        if spec.get('birth_deg') is not None:
            self.panel.BirthDegSpinBox.setValue(float(spec['birth_deg']))
        self._refresh_category_counts()

    def place(self):
        """Every cell of the built counts through the current model."""
        p = self.panel
        if self.model is None:
            raise ValueError('No model: fit one or load one first.')
        sp = self._storage()
        names = self._names_from_spec() if self.tidy is not None and self.spec is not None else self.names
        X, genes, ct, keys = self._dataset(names)
        name = self.spec.get('experiment', self._experiment_name()) if self.spec else self._experiment_name()
        if name not in self.model.panels:
            raise ValueError(f'the model carries no panel named {name!r} (it has {list(self.model.panels)})')
        want = self.model.panels[name]
        if list(genes) != list(want):
            missing = [g for g in want if g not in genes]
            if missing:
                raise ValueError(f'the built counts lack the model\'s genes {missing}')
            X = X[:, [genes.index(g) for g in want]]
        m, spec = self.model, dict(self.spec or {})
        spec['placed_at'] = datetime.datetime.now().isoformat(timespec='seconds')
        spec.setdefault('fitted_at', spec['placed_at'])
        p.PlacePushButton.setEnabled(False)
        p.ModelStatusLabel.setText(f'placing {len(X)} cells...')

        def _compute():
            placed = m.place(name, X)
            self._persist(sp, spec, placed, keys, X)
            return placed

        def _done(placed):
            p.PlacePushButton.setEnabled(True)
            self._adopt(m, spec, placed, keys, ct, X)
            p.ModelStatusLabel.setText(f'placed {len(X)} cells with the model of {spec.get("fitted_at")}')
            self._log(p.ModelStatusLabel.text())

        def _fail(msg):
            p.PlacePushButton.setEnabled(True)
            p.ModelStatusLabel.setText(f'FAILED: {msg}')

        self._start(_compute, _done, _fail)

    # -- the origin and the arcs, shared by fit and by the origin door ---------------

    def _settle_origin(self, m, name, X_tr, fov_tr, cell_tr, sp, origin_division, dapi_src, birth):
        """Rotate `m` so 0 deg is division, measured on the training cells:
        the DAPI halving when a DAPI source is given, else the panel-
        total drop, else nothing (0 stays the S genes' mean peak).
        Returns (origin dict, birth angle to use, the DAPI table or None).
        Runs in the worker: no widgets."""
        origin = {'mode': 'S mean peak'}
        birth_used = birth
        dapi_tab = None
        done = False
        if dapi_src is not None:
            th_tr = np.degrees(m.phase(name, X_tr)[0]) % 360.0
            tr = pd.DataFrame({'fov': fov_tr, 'cell': cell_tr, 'theta_deg': th_tr})
            dapi_tab, _fails = CC.mask_intensity_table(sp, sorted(set(tr['fov'].tolist())), dapi_src)
            merged = tr.merge(dapi_tab[['fov', 'cell', 'sum_above_bg']], on=['fov', 'cell'], how='inner')
            dna = merged['sum_above_bg'].to_numpy(float).copy()
            fv = merged['fov'].to_numpy()
            for f_ in np.unique(fv):
                kk = (fv == f_) & np.isfinite(dna)
                med = np.median(dna[kk]) if kk.sum() >= 5 else np.nan
                dna[fv == f_] = dna[fv == f_] / med if np.isfinite(med) and med > 0 else np.nan
            angle, factor = CC.total_drop_angle(merged['theta_deg'].to_numpy(), dna)
            if angle is not None and factor >= 1.3:
                m.rotate_to_zero(angle)
                origin = {'mode': 'division (DAPI halving)', 'angle_before_deg': angle, 'drop_factor': factor,
                          'dapi_source': list(dapi_src)}
                birth_used = 0.0
                done = True
            else:
                origin = {'mode': 'DAPI showed no halving', 'drop_factor': factor}
        if origin_division and not done:
            th_tr = np.degrees(m.phase(name, X_tr)[0]) % 360.0
            angle, factor = CC.total_drop_angle(th_tr, X_tr.sum(1))
            # the drop must be real: measured, the Tirosh-list panel
            # (JP_001) drops x5.4 within 30 deg, chr19 x1.9, the cyclin/CDK
            # panel (JP_002) x2.1 -- and JP_002's DAPI halves at the same
            # place, so that gentler drop IS division too
            if angle is not None and factor >= 1.8:
                m.rotate_to_zero(angle)
                origin = {'mode': 'division (total drop)' + (' after DAPI showed no halving' if dapi_src is not None else ''),
                          'angle_before_deg': angle, 'drop_factor': factor}
                birth_used = 0.0
            else:
                origin = {'mode': 'S mean peak (no clear drop)', 'drop_factor': factor}
        return origin, birth_used, dapi_tab

    def _propose_after(self, m, name, X, k, placed, fov_of, cell_of, dapi_tab, birth_used, shares):
        """The arcs to store with a (re)placed model: from DAPI when it was
        measured, else from cycle time. Returns (arcs, source label)."""
        arcs, arcs_from = [], ''
        w_train = m.spectrum(placed['posterior'][k])
        if dapi_tab is not None:
            try:
                th_new = np.degrees(placed['theta'][k]) % 360.0
                tr2 = pd.DataFrame({'fov': fov_of[k], 'cell': cell_of[k], 'theta_deg': th_new})
                merged2 = tr2.merge(dapi_tab[['fov', 'cell', 'sum_above_bg']], on=['fov', 'cell'], how='inner')
                arcs, _info = FC.propose_arcs_from_dapi(merged2['theta_deg'].to_numpy(), merged2['sum_above_bg'].to_numpy(),
                                                         birth_deg=birth_used, fov=merged2['fov'].to_numpy())
                arcs_from = 'DAPI'
            except Exception as exc:                            # noqa: BLE001
                self.mw.log(f'{TAB_TITLE}: arcs from DAPI not proposed: {type(exc).__name__}: {exc}')
        if not arcs:
            try:
                arcs = FC.propose_arcs_from_time(w_train, m.grid, birth_used, shares=shares)
                arcs_from = 'cycle time'
            except Exception:                                   # noqa: BLE001
                arcs = []
        if arcs and arcs_from != 'cycle time' and self.panel.PostMCheckBox.isChecked():
            # the first few percent of the cycle after division as its own
            # arc (2N, mitotic transcripts may remain) -- the share from the
            # shares line, the angle from the training clock; the cycle-time
            # proposal carries post-M as its own first share already
            arcs = FC.with_post_m(arcs, w_train, m.grid, birth_used, share=self._post_m_share(shares))
        return arcs, arcs_from

    def set_origin(self):
        """The origin door: rotate the CURRENT model to the chosen origin,
        re-place every cell, re-propose the arcs -- no EM."""
        p = self.panel
        m = self._need_model()
        sp = self._storage()
        names = self._names_from_spec() if self.spec is not None else self.names
        X, genes, ct, keys = self._dataset(names)
        name = (self.spec or {}).get('experiment', self._experiment_name())
        if name not in m.panels:
            raise ValueError(f'the model carries no panel named {name!r}')
        want = m.panels[name]
        if list(genes) != list(want):
            missing = [g for g in want if g not in genes]
            if missing:
                raise ValueError(f'the built counts lack the model\'s genes {missing}')
            X = X[:, [genes.index(g) for g in want]]
        train = (self.spec or {}).get('train_celltypes') or p.training_celltypes()
        k = np.isin(ct, train) if train else np.ones(len(X), bool)
        if k.sum() < 100:
            raise ValueError(f'Only {int(k.sum())} training cells to measure the origin on.')
        fov_of = np.array([int(f) for f, _c in keys])
        cell_of = np.array([int(c) for _f, c in keys])
        origin_choice = int(p.OriginComboBox.currentIndex())
        origin_division = origin_choice in (0, 1)
        dapi_src = p.dapi_source() if origin_choice == 0 else None
        if origin_choice == 0 and dapi_src is None:
            self._log('origin: no DAPI source in the layouts -- falling back to the panel-total drop')
        birth = float(p.BirthDegSpinBox.value())
        shares = p.phase_shares()
        spec = dict(self.spec or {})
        if origin_choice == 2:
            early = [g for g, r in (self.roles or {}).items() if r == 'S']
            late = [g for g, r in (self.roles or {}).items() if r == 'G2/M']
            if not (early and late):
                raise ValueError('the S-peak origin needs S and G2/M roles')
        p.SetOriginPushButton.setEnabled(False)
        p.ModelStatusLabel.setText('setting the origin on the current model...')

        def _compute():
            if origin_choice == 2:
                m.orient(early, late)
                origin, birth_used, dapi_tab = {'mode': 'S mean peak'}, birth, None
            else:
                origin, birth_used, dapi_tab = self._settle_origin(m, name, X[k], fov_of[k], cell_of[k], sp,
                                                                   origin_division, dapi_src, birth)
            placed = m.place(name, X)
            arcs, arcs_from = self._propose_after(m, name, X, k, placed, fov_of, cell_of, dapi_tab, birth_used, shares)
            spec['model'] = m.to_dict()
            spec['origin'] = origin
            spec['origin_set_at'] = datetime.datetime.now().isoformat(timespec='seconds')
            spec.setdefault('fitted_at', spec['origin_set_at'])
            spec['categories'], spec['categories_from'], spec['birth_deg'] = arcs, arcs_from, birth_used
            self._persist(sp, spec, placed, keys, X)
            return placed, spec

        def _done(res):
            placed, spec_ = res
            p.SetOriginPushButton.setEnabled(True)
            self._adopt(m, spec_, placed, keys, ct, X)
            o = spec_.get('origin') or {}
            otxt = o.get('mode', '')
            if o.get('drop_factor') is not None:
                otxt += f' (x{o["drop_factor"]:.1f}' + (f' at {o["angle_before_deg"]:.0f} deg before the rotation)' if 'angle_before_deg' in o else ')')
            p.ModelStatusLabel.setText(f'origin set on the model of {spec_.get("fitted_at")}: {otxt}; {len(X)} cells re-placed'
                                       + (f'; categories proposed from {spec_["categories_from"]}' if spec_.get('categories_from') else ''))
            self._log(p.ModelStatusLabel.text())

        def _fail(msg):
            p.SetOriginPushButton.setEnabled(True)
            p.ModelStatusLabel.setText(f'origin FAILED: {msg}')

        self._start(_compute, _done, _fail)

    def _names_from_spec(self):
        out = {}
        for key, gene in (self.spec.get('sources') or {}).items():
            m, h, ch = key.split('|')
            out[(m, h, int(ch))] = gene
        return out

    def _restore_from_store(self):
        """A stored model file restores the stage's state on the first
        visit: the model, the panel's genes/roles/arcs/gates."""
        sp = self._storage()
        spec = analysis_store.read_cellcycle_model(sp)
        if not spec or 'model' not in spec:
            return
        self._adopt_spec(spec)
        self.panel.ModelStatusLabel.setText(
            f'model of {spec.get("fitted_at")} restored from analysis/cellcycle_model.json '
            f'({spec.get("n_placed", "?")} cells placed; origin {(spec.get("origin") or {}).get("mode", "S mean peak")}); '
            f'build counts and Place to refresh')

    def _adopt_spec(self, spec):
        self.model = CC.CycleModel.from_dict(spec['model'])
        self.spec = spec
        self.roles = dict(spec.get('roles') or {})
        self.names = self._names_from_spec()
        p = self.panel
        p.set_gene_config(','.join(f'{m}|{h}|{ch}|{g}|{self.roles.get(g, "-")}' for (m, h, ch), g in self.names.items()))
        p.set_celltype_names(list(self.mw.current_celltype_list or []))
        p.set_training_celltypes(spec.get('train_celltypes') or [])
        if spec.get('alpha'):
            p.AlphaSpinBox.setValue(float(spec['alpha']))
        p.set_arcs(spec.get('categories') or [])
        p.set_gates(spec.get('gates') or {})
        if spec.get('birth_deg') is not None:
            p.BirthDegSpinBox.setValue(float(spec['birth_deg']))
        p.ProxyComboBox.setCurrentIndex(1 if spec.get('proxy') == 'soft' else 0)
        if spec.get('min_total') is not None:
            p.MinTotalSpinBox.setValue(int(spec['min_total']))

    def _model_dir(self):
        """Where model copies go by default: the project root (the user's
        own choice), beside the store's analysis/ directory."""
        try:
            return paths.project_root(self._storage())
        except Exception:                                       # noqa: BLE001
            return ''

    def load_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self.mw, 'Load cell-cycle model', self._model_dir(), 'JSON (*.json)')
        if not path:
            return
        with open(path) as f:
            spec = json.load(f)
        if 'model' not in spec:
            raise ValueError('not a cell-cycle model file (no "model" entry)')
        self._adopt_spec(spec)
        self.placed = self.q = self.X = None
        self.panel.ModelStatusLabel.setText(f'loaded {os.path.basename(path)} (fitted {spec.get("fitted_at")}, '
                                            f'experiment {spec.get("experiment")}, origin '
                                            f'{(spec.get("origin") or {}).get("mode", "S mean peak")}); '
                                            f'the frame travels with the model -- Place to write placements')
        self._log(self.panel.ModelStatusLabel.text())

    def save_model(self):
        if self.spec is None:
            raise ValueError('No model to save.')
        spec = dict(self.spec)
        spec['categories'], spec['gates'], spec['birth_deg'] = self.panel.arcs(), self.panel.gates(), float(self.panel.BirthDegSpinBox.value())
        stamp = str(spec.get('fitted_at') or '').replace(':', '').replace('-', '').replace('T', '-')
        default = os.path.join(self._model_dir(), f'{spec.get("experiment", "model")}_cellcycle_model_{stamp or "unfitted"}.json')
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self.mw, 'Save cell-cycle model', default, 'JSON (*.json)')
        if not path:
            return
        with open(path, 'w') as f:
            json.dump(spec, f)
        self._log(f'model saved to {path}')

    # -- 3. figures ------------------------------------------------------------------

    def _need_model(self, placed=False):
        if self.model is None:
            raise ValueError('No model: fit one or load one first (section 2).')
        if placed and self.placed is None:
            raise ValueError('No placements in memory: Fit, or Place with the loaded model.')
        return self.model

    def _name(self):
        return (self.spec or {}).get('experiment', self._experiment_name())

    def _marks(self):
        """The role marks for the phase axes, from the current model."""
        try:
            return FC.role_marks(self.model, self.roles or {}) if self.model is not None else None
        except Exception:                                       # noqa: BLE001
            return None

    def _default_dir(self):
        try:
            return paths.figure_dir(self._storage(), 'cellcycle', 0)
        except Exception:
            return ''

    def _show(self, fig, name, tables=None, params=None):
        d = AnalysisFigureDisplayer(title=f'{TAB_TITLE} -- {name}', parent=self.mw)
        d.set_figure(fig, name=name, tables=tables, pop=None, condition=None,
                     params=params or {'model': (self.spec or {}).get('fitted_at')},
                     default_dir=self._default_dir())
        d.show()
        d.raise_()
        self.displayers.append(d)

    def _view(self, label, compute, name, tables_of=None):
        """Run `compute` -> (fig, tables) in a worker, then show."""
        self.panel.ModelStatusLabel.setText(f'{label}...')

        def _done(res):
            fig, tables = res if isinstance(res, tuple) else (res, None)
            self.panel.ModelStatusLabel.setText(f'{label}: shown')
            self._show(fig, name, tables=tables)

        def _fail(msg):
            self.panel.ModelStatusLabel.setText(f'{label} FAILED: {msg}')

        self._start(compute, _done, _fail)

    def _training_mask(self):
        train = (self.spec or {}).get('train_celltypes') or self.panel.training_celltypes()
        ct = self.placed['celltype'].to_numpy()
        return np.isin(ct, train) if train else np.ones(len(ct), bool)

    def view_spectrum(self):
        m = self._need_model(placed=True)
        groups, q, totals, name = self.placed['celltype'].to_numpy(), self.q, self.placed['total'].to_numpy(), self._name()
        marks = self._marks()
        self._view('spectrum', lambda: FC.fig_spectrum_and_totals(m, groups, q, totals, title=name, marks=marks),
                   'spectrum_and_totals')

    def view_profiles(self):
        m = self._need_model()
        roles = self.roles or {}
        rd = {'S indicators': [g for g in m.genes if roles.get(g) == 'S'],
              'G2/M indicators': [g for g in m.genes if roles.get(g) == 'G2/M'],
              'unassigned': [g for g in m.genes if roles.get(g) not in ('S', 'G2/M')]}
        marks = self._marks()
        self._view('gene profiles', lambda: FC.fig_profiles_by_role(m, rd, title=f'{self._name()}: fitted gene profiles',
                                                                     marks=marks), 'profiles_by_role')

    def view_embeddings(self):
        m = self._need_model(placed=True)
        k = self._training_mask()
        X, th, name = self.X[k], self.placed['theta_deg'].to_numpy()[k], self._name()

        marks = self._marks()
        tsne = self.panel.TsneCheckBox.isChecked()

        def _compute():
            fig, notes = FC.fig_embeddings(m, name, X, th, tsne=tsne, marks=marks,
                                           title=f'{name}: {int(k.sum())} training cells, coloured by phase')
            if notes:
                self.mw.log(f'{TAB_TITLE}: embeddings: ' + '; '.join(notes))
            return fig, None
        self._view('the ring' + (' (tSNE takes a while)' if tsne else ''), _compute, 'embeddings')

    def view_verdicts(self):
        self._need_model(placed=True)
        placed = self.placed.copy()

        def _compute():
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
            conds = list(dict.fromkeys(placed['celltype']))
            pal = FC.gene_palette(len(conds))
            specs = (('R', 'R (posterior concentration)', (0, 1)), ('fit_z', 'fit_z (vs training cells)', (-8, 4)),
                     ('bf_ring', 'log BF_ring (ring vs centre)', (-10, 20)), ('radius', 'centre ratio (plane radius)', (0, 3)))
            for ax, (col, lab, rng) in zip(axes, specs):
                for cond, c in zip(conds, pal):
                    v = placed.loc[placed['celltype'] == cond, col].to_numpy(float)
                    v = v[np.isfinite(v)]
                    if len(v):
                        FC.line_hist(ax, np.clip(v, *rng), bins=40, range=rng, color=c, lw=1.5,
                                     label=f'{cond or "Unassigned"} (n={len(v)})')
                ax.set_xlabel(lab)
                ax.set_yticks([])
            axes[0].legend(fontsize=7, frameon=False)
            fig.tight_layout(rect=(0, 0, 1, 0.9))
            FC.finish(fig, f'{self._name()}: per-cell verdicts by condition')
            return fig, {'placements': placed}
        self._view('verdicts', _compute, 'verdicts')

    def view_contribution(self):
        m = self._need_model()
        name = self._name()

        def _compute():
            import matplotlib.pyplot as plt
            share = m.fisher_share(name)
            pk, f = m.peak_phase()
            amp = np.exp(f.max(0) - f.min(0))
            rows = [{'gene': g, 'role': (self.roles or {}).get(g, '-'), 'fisher_share_%': round(100 * float(share[g]), 1),
                     'peak_deg': int(round(np.degrees(pk[m.gi[g]]) % 360)), 'amplitude': round(float(amp[m.gi[g]]), 2)}
                    for g in m.panels[name]]
            tab = pd.DataFrame(rows).sort_values('fisher_share_%', ascending=False)
            fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(tab)), 3.6))
            cols = ['#0072B2' if r == 'S' else '#D55E00' if r == 'G2/M' else '#999999' for r in tab['role']]
            ax.bar(tab['gene'], tab['fisher_share_%'], color=cols)
            ax.set_ylabel('Fisher information share (%)')
            ax.tick_params(axis='x', rotation=60)
            fig.tight_layout(rect=(0, 0, 1, 0.9))
            FC.finish(fig, f'{name}: what each gene contributes to the phase (blue S, orange G2/M, grey other)')
            return fig, {'contribution': tab}
        self._view('contribution', _compute, 'contribution')

    def view_groups(self):
        m = self._need_model(placed=True)
        name, X, groups, q = self._name(), self.X, self.placed['celltype'].to_numpy(), self.q

        def _compute():
            rows = []
            for c, r in m.group_summary(name, X, groups).items():
                r.update({'condition': c or 'Unassigned'})
                rows.append(r)
            tab = pd.DataFrame(rows)
            entries = [(name, c or 'Unassigned', q[groups == c]) for c in dict.fromkeys(groups) if (groups == c).sum() >= 5]
            fig = FC.fig_anchor_spectra(m, entries, [c or 'Unassigned' for c in dict.fromkeys(groups)],
                                        title=f'{name}: spectrum per condition (mean direction marked)',
                                        marks=self._marks())
            return fig, {'groups': tab}
        self._view('group table', _compute, 'groups')

    def view_half_panel(self):
        m = self._need_model(placed=True)
        name = self._name()
        k = self._training_mask()
        X = self.X[k]

        def _compute():
            share = m.fisher_share(name)
            order = sorted(share, key=lambda g: -share[g])
            halves = {'half A': order[0::2], 'half B': order[1::2]}
            th, st = FC.subpanel_agreement(m, name, X, halves)
            fig = FC.fig_subpanel(th, 'half A', 'half B', marks=self._marks(),
                                  title=f'{name} training cells: half A {halves["half A"]} vs half B {halves["half B"]}')
            d = np.abs((th['half A'] - th['half B'] + 180) % 360 - 180)
            s = X.sum(1)
            edges = np.quantile(s, np.linspace(0, 1, 5))
            b = np.clip(np.searchsorted(edges, s, side='right') - 1, 0, 3)
            depth = pd.DataFrame([{'total_count_range': f'{edges[j]:.0f}-{edges[j + 1]:.0f}', 'n': int((b == j).sum()),
                                   'median_abs_diff_deg': round(float(np.median(d[b == j])), 1) if (b == j).any() else np.nan,
                                   'frac_within_45': round(float(np.mean(d[b == j] <= 45)), 2) if (b == j).any() else np.nan}
                                  for j in range(4)])
            return fig, {'agreement': st, 'by_depth': depth}
        self._view('half-panel reproducibility', _compute, 'half_panel')

    def _training_spectrum(self):
        m = self._need_model(placed=True)
        return m.spectrum(self.q[self._training_mask()])

    def view_cycle_time(self):
        m = self._need_model(placed=True)
        k = self._training_mask()
        name, birth = self._name(), float(self.panel.BirthDegSpinBox.value())
        groups = self.placed['celltype'].to_numpy()
        theta = self.placed['theta_deg'].to_numpy()
        conds = [c for c in dict.fromkeys(groups) if (groups == c).sum() >= 5]
        spectra = {(c or 'Unassigned'): m.spectrum(self.q[groups == c]) for c in conds}
        training = 'training (' + ', '.join(self.spec.get('train_celltypes') or ['all']) + ')'
        spectra[training] = m.spectrum(self.q[k])
        groups_theta = {(c or 'Unassigned'): theta[groups == c] for c in conds}
        marks = self._marks()
        self._view('cycle time', lambda: (FC.fig_cycle_time(
            m, spectra, birth_deg=birth, training=training, groups_theta=groups_theta, marks=marks,
            title=f'{name}: the angle as cycle time (birth at {birth:.0f} deg)'), None),
            'cycle_time')

    @staticmethod
    def _post_m_share(shares):
        for n, s in shares:
            if n.lower().replace('_', '-') in ('post-m', 'postm', 'post m'):
                return float(s)
        return 0.05

    def propose_arcs_from_dapi(self):
        """Arcs from the measured DNA content of the training cells."""
        p = self.panel
        self._need_model(placed=True)
        src = p.dapi_source()
        if src is None:
            raise ValueError('No DAPI source: parse the layouts first, then pick the DAPI round.')
        sp = self._storage()
        k = self._training_mask()
        placed = self.placed[k]
        fovs = sorted(set(int(f) for f in placed['fov']))
        birth = float(p.BirthDegSpinBox.value())
        p.ProposeFromDapiPushButton.setEnabled(False)
        p.CategoryStatusLabel.setText(f'DAPI over {len(fovs)} FOVs for the arcs...')

        def _compute():
            tab, _fails = CC.mask_intensity_table(sp, fovs, src)
            merged = placed.merge(tab[['fov', 'cell', 'sum_above_bg']], on=['fov', 'cell'], how='inner')
            return FC.propose_arcs_from_dapi(merged['theta_deg'].to_numpy(), merged['sum_above_bg'].to_numpy(),
                                             birth_deg=birth, fov=merged['fov'].to_numpy())

        def _done(res):
            p.ProposeFromDapiPushButton.setEnabled(True)
            arcs, info = res
            if p.PostMCheckBox.isChecked():
                arcs = FC.with_post_m(arcs, self._training_spectrum(), self.model.grid, birth, share=self._post_m_share(p.phase_shares()))
            p.set_arcs(arcs)
            p.CategoryStatusLabel.setText(
                f'proposed from DAPI (G1 level {info["g1_level"]:.2f}, plateau {info["top"]:.2f}, ratio {info["ratio"]:.2f}): '
                + ', '.join(f'{a["name"]} {a["start_deg"]:.0f}-{a["end_deg"]:.0f}' for a in arcs) + '. Edit, then Apply.')
            self._refresh_category_counts()

        def _fail(msg):
            p.ProposeFromDapiPushButton.setEnabled(True)
            p.CategoryStatusLabel.setText(f'DAPI proposal FAILED: {msg}')

        self._start(_compute, _done, _fail)

    def propose_arcs_from_time(self):
        m = self._need_model(placed=True)
        shares = self.panel.phase_shares()
        birth = float(self.panel.BirthDegSpinBox.value())
        arcs = FC.propose_arcs_from_time(self._training_spectrum(), m.grid, birth, shares=shares)
        self.panel.set_arcs(arcs)
        self.panel.CategoryStatusLabel.setText(
            'proposed from cycle time: ' + ', '.join(f'{a["name"]} {a["start_deg"]:.0f}-{a["end_deg"]:.0f}' for a in arcs)
            + '. Edit, then Apply.')
        self._refresh_category_counts()

    def view_dapi(self):
        """The routine verification: DAPI inside each mask against the
        phase and the categories."""
        p = self.panel
        self._need_model(placed=True)
        src = p.dapi_source()
        if src is None:
            raise ValueError('No DAPI source: parse the layouts first, then pick the DAPI round.')
        sp = self._storage()
        fovs = sorted(set(int(f) for f in self.placed['fov']))
        placed = self.placed.copy()
        arcs, gates, marks, name = p.arcs(), p.gates(), self._marks(), self._name()
        try:
            voxel = self.mw._voxel_um()
        except Exception:                                       # noqa: BLE001
            voxel = None
        p.DapiPushButton.setEnabled(False)
        p.ModelStatusLabel.setText(f'DAPI over {len(fovs)} FOVs...')

        def _compute():
            tab, fails = CC.mask_intensity_table(sp, fovs, src)
            if fails:
                self.mw.log(f'{TAB_TITLE}: DAPI: {len(fails)} FOV(s) failed: {fails[0][1]}')
            merged = placed.merge(tab[['fov', 'cell', 'area', 'mask_mean', 'sum_above_bg']], on=['fov', 'cell'], how='left')
            cat = CC.assign(merged, arcs, gates) if arcs else None
            order = [a['name'] for a in arcs] if arcs else None
            train = (self.spec or {}).get('train_celltypes') or []
            tr = np.isin(merged['celltype'].to_numpy(), train) if train else None
            area, unit = merged['area'].to_numpy(float), 'px'
            if voxel is not None:
                area, unit = area * float(voxel[0]) * float(voxel[1]), 'um^2'
            fig = FC.fig_dapi_vs_phase(merged['theta_deg'].to_numpy(), merged['sum_above_bg'].to_numpy(),
                                       groups=merged['celltype'].to_numpy(), categories=cat, order=order, marks=marks,
                                       fov=merged['fov'].to_numpy(), training=tr, area=area, area_unit=unit,
                                       title=f'{name}: DAPI ({src[1]} ch{src[2]}) as the routine verification')
            if cat is not None:
                merged = merged.assign(category=cat)
            return fig, {'dapi': merged}

        def _done(res):
            p.DapiPushButton.setEnabled(True)
            fig, tables = res
            p.ModelStatusLabel.setText('DAPI: shown')
            self._show(fig, 'dapi_vs_phase', tables=tables)

        def _fail(msg):
            p.DapiPushButton.setEnabled(True)
            p.ModelStatusLabel.setText(f'DAPI FAILED: {msg}')

        self._start(_compute, _done, _fail)

    def view_dapi_gallery(self, n_bins=8, per_row=10, size=72, seed=0):
        """Example DAPI images per phase bin: the routine verification by
        eye."""
        p = self.panel
        self._need_model(placed=True)
        src = p.dapi_source()
        if src is None:
            raise ValueError('No DAPI source: parse the layouts first, then pick the DAPI round.')
        sp = self._storage()
        placed = self.placed
        k = self._training_mask() & (placed['R'].to_numpy() >= 0.5)
        pool = placed[k]
        if len(pool) < n_bins:
            raise ValueError('Too few confident training cells for a gallery.')
        rng = np.random.default_rng(seed)
        edges = np.linspace(0, 360, n_bins + 1)
        picks, labels = [], []
        for i in range(n_bins):
            sub = pool[(pool['theta_deg'] >= edges[i]) & (pool['theta_deg'] < edges[i + 1])]
            take = sub.iloc[rng.choice(len(sub), min(per_row, len(sub)), replace=False)] if len(sub) else sub
            picks.append([(int(f), int(c)) for f, c in zip(take['fov'], take['cell'])])
            labels.append(f'{edges[i]:.0f}-{edges[i + 1]:.0f} deg (n={len(sub)})')
        wanted = {}
        for row in picks:
            for f, c in row:
                wanted.setdefault(f, []).append(c)
        name = self._name()
        p.DapiGalleryPushButton.setEnabled(False)
        p.ModelStatusLabel.setText(f'DAPI gallery: {sum(len(r) for r in picks)} crops over {len(wanted)} FOVs...')

        def _compute():
            crops = CC.gallery_crops(sp, src, wanted, size=size)
            rows = [(lab, [crops[key] for key in row if key in crops]) for lab, row in zip(labels, picks)]
            fig = FC.fig_gallery(rows, size=size, title=f'{name}: DAPI ({src[1]} ch{src[2]}) examples per phase bin, training cells with R >= 0.5')
            return fig, None

        def _done(res):
            p.DapiGalleryPushButton.setEnabled(True)
            fig, _t = res
            p.ModelStatusLabel.setText('DAPI gallery: shown')
            self._show(fig, 'dapi_gallery')

        def _fail(msg):
            p.DapiGalleryPushButton.setEnabled(True)
            p.ModelStatusLabel.setText(f'DAPI gallery FAILED: {msg}')

        self._start(_compute, _done, _fail)

    def view_fov_overlay(self):
        p = self.panel
        sp = self._storage()
        fov = int(p.OverlayFovSpinBox.value())
        mode = p.OverlayModeComboBox.currentText()
        cells, _ = analysis_store.read_cells(sp, fov)
        if not cells:
            raise ValueError(f'FOV {fov} has no cells in the store.')
        if self.placed is not None and (self.placed['fov'] == fov).any():
            sub = self.placed[self.placed['fov'] == fov]
            theta = dict(zip(sub['cell'].astype(int), sub['theta_deg'].astype(float)))
            table = sub
        else:
            cap = analysis_store.read_fov_cellcycle(sp, fov)
            if not cap or not cap.get('rows'):
                raise ValueError(f'FOV {fov} has no placements: Fit or Place first.')
            table = pd.DataFrame(cap['rows'])
            theta = dict(zip(table['cell'].astype(int), table['theta_deg'].astype(float)))
        if mode == 'category':
            arcs = p.arcs()
            if not arcs:
                raise ValueError('No category arcs: propose or add them (section 4) first.')
            cat = CC.assign(table, arcs, p.gates())
            values = dict(zip(table['cell'].astype(int), cat))
            order = [a['name'] for a in arcs]
        else:
            values, order = theta, None
        # the reference MIP: the cells' own reference hybe in its modality,
        # the segmentation channel when the panel names one
        ref_hybe = str(cells[0].get('reference_hybe') or '')
        ref_mod = str(cells[0].get('reference_modality') or '')
        sp_ref = self.mw._storage_path_for_modality(ref_mod) or sp
        channel = None
        try:
            channel = int(self.mw.ui.CellSegmentPanel.ChannelComboBox.currentText())
        except Exception:                                       # noqa: BLE001
            pass
        if channel is not None:
            mip = analysis_store.read_hybe_mip(sp_ref, fov, ref_hybe, channel)
        else:
            mip = analysis_store.fiducial_channel_mip(sp_ref, fov, ref_hybe)
        marks = self._marks()
        title = f'{self._name()} FOV {fov}: cells by {mode} on {ref_hybe}' + (f' ch{channel}' if channel else '')
        self._view('FOV overlay', lambda: (FC.fig_fov_overlay(mip, cells, values, mode=mode, categories=order,
                                                              title=title, marks=marks), None), f'fov{fov:03d}_overlay')

    # -- 4. categories -----------------------------------------------------------------

    def propose_arcs(self):
        """A phase begins where its indicator genes' mean profile rises
        through the cycle mean; G1 begins at birth."""
        m = self._need_model()
        birth = float(self.panel.BirthDegSpinBox.value())
        arcs = FC.propose_arcs_from_profiles(m, self.roles or {}, birth_deg=birth)
        if self.panel.PostMCheckBox.isChecked() and self.placed is not None:
            arcs = FC.with_post_m(arcs, self._training_spectrum(), m.grid, birth, share=self._post_m_share(self.panel.phase_shares()))
        self.panel.set_arcs(arcs)
        self.panel.CategoryStatusLabel.setText(
            'proposed from the roles (a phase starts where its genes rise above their cycle mean; G1 at birth '
            f'{birth:.0f} deg): ' + ', '.join(f'{a["name"]} {a["start_deg"]:.0f}-{a["end_deg"]:.0f}' for a in arcs)
            + '. Edit, then Apply.')
        self._refresh_category_counts()

    def apply_categories(self):
        sp = self._storage()
        spec = analysis_store.read_cellcycle_model(sp)
        if not spec:
            if self.spec is None:
                raise ValueError('No stored model to attach categories to: fit or place first.')
            spec = dict(self.spec)
        arcs, gates = self.panel.arcs(), self.panel.gates()
        if not arcs:
            raise ValueError('No category arcs (add rows or Propose from roles).')
        spec['categories'], spec['gates'], spec['birth_deg'] = arcs, gates, float(self.panel.BirthDegSpinBox.value())
        analysis_store.write_cellcycle_model(sp, spec)
        self.spec = spec
        self._refresh_category_counts()
        self._log('categories and gates stored: ' + ', '.join(f'{a["name"]} {a["start_deg"]:.0f}-{a["end_deg"]:.0f}' for a in arcs)
                  + ' | gates ' + ', '.join(f'{k}={v}' for k, v in gates.items() if v is not None))

    def _refresh_category_counts(self):
        p = self.panel
        arcs = p.arcs()
        if self.placed is None or not arcs:
            p.CategoryStatusLabel.setText('' if self.placed is None else 'no arcs yet')
            return
        cat = CC.assign(self.placed, arcs, p.gates())
        counts = pd.Series(cat).replace('', 'Unassigned').value_counts()
        p.CategoryStatusLabel.setText('cells per category: ' + ', '.join(f'{k} {v}' for k, v in counts.items()))
