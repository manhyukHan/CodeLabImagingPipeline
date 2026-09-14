"""The pipeline runner's execution half: a sequencer over the app's own
batch actions.

Five of the nine steps exist only as MainWindow actions (celltype, cell
cycle, chromatin tracing, and the three alignment batches with their
append deltas), so the runner does not re-implement them: it sets the
Ingestion tab's FOV list to the step's FOVs, answers the batch-mode
dialog from the policy ('existing' -> append, 'redo' -> overwrite),
mutes the message boxes into the log, calls the action, and polls the
action's worker until it is idle. Localization uses the headless
batch module directly. Segmentation is looped FOV by FOV through the
Cell Segmentation tab (run, then save) and only when a person chose it.

The same object drives the tab inside the live app (the timer runs on
the GUI event loop) and the CLI (an offscreen MainWindow whose event
loop the CLI pumps). Ingestion is never overwritten by the runner: a
'redo' on ingestion runs as append and says so.
"""
import time

from PyQt5 import QtCore, QtWidgets

from codelab_pipeline.pipeline import status as S

TAB_TITLE = 'Pipeline'
POLL_MS = 250


def _running(worker):
    try:
        return worker is not None and worker.isRunning()
    except RuntimeError:                                        # a deleted QThread
        return False


class Quiet:
    """Message boxes and the two batch-mode dialogs answered from the
    policy while a step runs; everything they would have said goes to
    the log. Restores on exit."""

    def __init__(self, mw, mode, log):
        self.mw, self.mode, self.log = mw, mode, log
        self._saved = {}

    def __enter__(self):
        def _box(kind):
            def _show(*args, **kwargs):
                text = ' | '.join(str(a) for a in args[1:3] if isinstance(a, str))
                self.log(f'[{kind}] {text}')
                return QtWidgets.QMessageBox.Yes if kind == 'question' else QtWidgets.QMessageBox.Ok
            return _show
        for kind in ('information', 'warning', 'critical', 'question'):
            self._saved[kind] = getattr(QtWidgets.QMessageBox, kind)
            setattr(QtWidgets.QMessageBox, kind, staticmethod(_box(kind)))
        self._saved['_confirm_batch_mode'] = self.mw._confirm_batch_mode
        self._saved['_confirm_overwrite'] = self.mw._confirm_overwrite
        mode = self.mode
        self.mw._confirm_batch_mode = lambda *a, **k: mode
        # ingestion is never overwritten from here: an overwrite clears
        # stacks first, and that is a person's decision at the tab
        self.mw._confirm_overwrite = lambda *a, **k: 'append'
        return self

    def __exit__(self, *exc):
        for kind in ('information', 'warning', 'critical', 'question'):
            setattr(QtWidgets.QMessageBox, kind, staticmethod(self._saved[kind]))
        self.mw._confirm_batch_mode = self._saved['_confirm_batch_mode']
        self.mw._confirm_overwrite = self._saved['_confirm_overwrite']
        return False


class PipelineRunner(QtCore.QObject):
    """start(project, plan, policies, explicit) runs the plan's steps in
    pipeline order, one at a time, polling each step's worker; emits
    step_started / step_finished / finished; stop() ends after the
    current step (running workers are not killed)."""
    step_started = QtCore.pyqtSignal(str, list)
    step_finished = QtCore.pyqtSignal(str, str)
    finished = QtCore.pyqtSignal(dict)
    logged = QtCore.pyqtSignal(str)

    def __init__(self, mw, parent=None):
        super().__init__(parent)
        self.mw = mw
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._tick)
        self._queue = []
        self._current = None
        self._stopping = False
        self._report = {'steps': [], 'failed': []}
        self._fov_text = None
        self.project = None
        self.running = False

    # -- driving --------------------------------------------------------------

    def log(self, msg):
        self.logged.emit(msg)
        try:
            self.mw.log(f'{TAB_TITLE}: {msg}')
        except Exception:                                       # noqa: BLE001
            pass

    def start(self, project, plan, policies=None, explicit=()):
        """project: status.load_project(); plan: status.plan(); policies
        {step: policy}; explicit: the steps a person set a policy for
        (segmentation runs only when it is among them)."""
        if self.running:
            raise RuntimeError('a pipeline run is already in progress')
        pol = dict(S.DEFAULT_POLICY)
        pol.update(policies or {})
        self.project = project
        self._queue = []
        for step in S.STEPS:
            fovs = list(plan.get(step, []))
            if not fovs or pol[step] == 'skip':
                continue
            if step in S.NEVER_AUTO and step not in set(explicit):
                self.log(f'{step}: {len(fovs)} FOV(s) not done, left alone (choose its policy explicitly to run it)')
                continue
            self._queue.append((step, fovs, 'overwrite' if pol[step] == 'redo' else 'append'))
        self._report = {'steps': [], 'failed': [], 'started': time.time()}
        self._stopping = False
        self._current = None
        ip = self.mw.ui.IngestionPanel
        self._fov_text = ip.FovListLineEdit.text()
        self.running = True
        self.log('run: ' + (', '.join(f'{s} ({len(f)} FOV, {m})' for s, f, m in self._queue) or 'nothing to do'))
        self._timer.start()

    def stop(self):
        self._stopping = True
        self.log('stop requested: finishing the current step, then stopping')

    def _tick(self):
        if self._current is not None:
            step, t0, busy, finish, quiet = self._current
            try:
                if busy():
                    return
            except Exception as exc:                            # noqa: BLE001
                self.log(f'{step}: busy check failed: {type(exc).__name__}: {exc}')
            self._finish_current()
        if self._stopping or not self._queue:
            self._finalize()
            return
        self._start_next()

    def _start_next(self):
        step, fovs, mode = self._queue.pop(0)
        ip = self.mw.ui.IngestionPanel
        ip.FovListLineEdit.setText(','.join(str(f) for f in fovs))
        self.step_started.emit(step, fovs)
        self.log(f'{step}: {len(fovs)} FOV(s), mode {mode}')
        quiet = Quiet(self.mw, mode, self.log)
        quiet.__enter__()
        t0 = time.time()
        try:
            busy, finish = getattr(self, f'_run_{step}')(fovs, mode)
        except Exception as exc:                                # noqa: BLE001
            quiet.__exit__(None, None, None)
            msg = f'{step}: could not start: {type(exc).__name__}: {exc}'
            self.log(msg)
            self._report['failed'].append((step, msg))
            self._report['steps'].append({'step': step, 'fovs': fovs, 'mode': mode, 'seconds': 0.0, 'error': msg})
            self.step_finished.emit(step, msg)
            return
        self._current = (step, t0, busy, finish, quiet)

    def _finish_current(self):
        step, t0, busy, finish, quiet = self._current
        self._current = None
        quiet.__exit__(None, None, None)
        summary = ''
        try:
            summary = finish() or ''
        except Exception as exc:                                # noqa: BLE001
            summary = f'finish hook failed: {type(exc).__name__}: {exc}'
        seconds = time.time() - t0
        after = self._after(step)
        entry = {'step': step, 'seconds': round(seconds, 1), 'summary': summary, 'after': after}
        self._report['steps'].append(entry)
        text = f'{step}: {seconds:.0f}s; {summary}' + (f'; store now {after}' if after else '')
        self.log(text)
        self.step_finished.emit(step, text)

    def _after(self, step):
        """The step's store state over its FOVs after it ran."""
        if self.project is None:
            return ''
        try:
            fovs = self._fovs_of(step)
            counts = {'done': 0, 'partial': 0, 'missing': 0, 'n/a': 0}
            targets = S.localization_targets(self.project)
            for f in fovs:
                st = S.fov_status(self.project, f, deep=(step in S.DEEP_STEPS), targets=targets)[step]
                counts[st['state']] += 1
            return ', '.join(f'{k} {v}' for k, v in counts.items() if v)
        except Exception as exc:                                # noqa: BLE001
            return f'(status unreadable: {type(exc).__name__})'

    def _fovs_of(self, step):
        ip = self.mw.ui.IngestionPanel
        return self.mw._parse_fov_list(ip.FovListLineEdit.text())

    def _finalize(self):
        self._timer.stop()
        if self._fov_text is not None:
            self.mw.ui.IngestionPanel.FovListLineEdit.setText(self._fov_text)
        self.running = False
        self._report['seconds'] = round(time.time() - self._report.get('started', time.time()), 1)
        self._report['stopped'] = self._stopping
        self.log(f'run finished in {self._report["seconds"]:.0f}s; {len(self._report["steps"])} step(s), '
                 f'{len(self._report["failed"])} failed' + (' (stopped early)' if self._stopping else ''))
        self.finished.emit(dict(self._report))

    # -- the steps: each returns (busy, finish) -----------------------------------------

    def _run_ingestion(self, fovs, mode):
        mw = self.mw
        if mode == 'overwrite':
            self.log('ingestion: redo requested; the runner never overwrites stacks -- running as append')
        mw._run_ingestion()
        return (lambda: bool(getattr(mw, '_active_ingestions', []))), (lambda: 'ingestion run ended')

    def _run_segmentation(self, fovs, mode):
        """FOV by FOV through the Cell Segmentation tab: run, wait, save."""
        mw = self.mw
        cp = mw.ui.CellSegmentPanel
        state = {'todo': list(fovs), 'started': None, 'saved': 0, 'errors': []}

        def _launch():
            fov = state['todo'].pop(0)
            state['started'] = fov
            cp.FovSpinBox.setValue(int(fov))
            mw._run_cell_segmentation()

        def busy():
            w = getattr(mw, '_segment_worker', None)
            if state['started'] is not None:
                if _running(w):
                    return True
                try:
                    mw._save_cells()
                    state['saved'] += 1
                except Exception as exc:                        # noqa: BLE001
                    state['errors'].append(f'FOV {state["started"]}: {type(exc).__name__}: {exc}')
                state['started'] = None
            if state['todo']:
                _launch()
                return True
            return False
        _launch()
        return busy, (lambda: f'{state["saved"]} FOV(s) segmented and saved' + (f'; errors: {state["errors"][:5]}' if state['errors'] else ''))

    def _run_fov_alignment(self, fovs, mode):
        mw = self.mw
        mw._run_fov_alignment_all()
        return (lambda: _running(getattr(mw, '_alignment_worker', None))), (lambda: 'FOV alignment batch ended')

    def _run_cross_modal(self, fovs, mode):
        mw = self.mw
        mw._run_cross_modal_alignment_all()

        def busy():
            return bool(getattr(mw, '_cross_modal_queue', [])) or _running(getattr(mw, '_cross_modal_worker', None))
        return busy, (lambda: 'cross-modal batch ended')

    def _run_cell_alignment(self, fovs, mode):
        mw = self.mw
        mw._run_cell_alignment_all_fovs()
        return (lambda: _running(getattr(mw, '_cell_alignment_worker', None))), (lambda: 'cell alignment batch ended')

    def _run_localization(self, fovs, mode):
        """The headless batch module, one group per (modality, channel)
        of the expected sources, on the panel's selected v3 model."""
        from codelab_pipeline.localization import batch
        from windows.main_window import FnWorker
        mw = self.mw
        sp_panel = mw.ui.SpotLocalizationPanel
        model_dir = sp_panel.selected_model_dir()
        if not model_dir:
            raise ValueError('no localization model selected on the Spot Localization tab')
        targets = S.localization_targets(self.project)
        groups = []
        for modality, m in self.project['modalities'].items():
            by_ch = {}
            for hybe, ch in targets.get(modality, []):
                by_ch.setdefault(int(ch), []).append(hybe)
            for ch, hybes in by_ch.items():
                groups.append((m['storage_path'], modality, ch, hybes))
        if not groups:
            raise ValueError('no localization targets for this project')
        jobs = None
        try:
            jobs = int(mw.ui.IngestionPanel.IngestWorkersSpinBox.value()) or None
        except Exception:                                       # noqa: BLE001
            jobs = None
        overwrite = (mode == 'overwrite')
        log = self.log
        state = {'result': None}

        def _work():
            out = []
            for storage_path, modality, ch, hybes in groups:
                res = batch.run(storage_path, fovs, hybes, ch, model_dir, jobs=jobs, overwrite=overwrite,
                                log=lambda m: log(f'localize {modality} ch{ch}: {m}'))
                out.append((modality, ch, res))
            return out
        worker = FnWorker(_work)
        worker.finished_ok.connect(lambda r: state.__setitem__('result', r))
        worker.failed.connect(lambda m: state.__setitem__('result', f'FAILED: {m}'))
        self._loc_worker = worker
        worker.start()

        def finish():
            r = state['result']
            if isinstance(r, str):
                self._report['failed'].append(('localization', r))
                return r
            parts = []
            for modality, ch, res in (r or []):
                parts.append(f'{modality} ch{ch}: {res["written"]} written, {res["skipped"]} skipped, {len(res["failed"])} failed')
                for (fov, hybe), msg in res['failed'][:5]:
                    self.log(f'localize {modality} ch{ch} FOV {fov} {hybe}: {msg}')
            return '; '.join(parts) or 'nothing to localize'
        return (lambda: _running(worker)), finish

    def _run_celltype(self, fovs, mode):
        """The store's celltype config first (FOV ranges, barcode channels
        -- the app only reads it on a tab visit), then FOV mode when ranges
        exist, else the panel's own mode."""
        mw = self.mw
        refresh = getattr(mw, '_refresh_celltype_config_from_vlinks', None)
        if refresh is not None:
            refresh()
        ranges = {k: v for k, v in (getattr(mw, '_fov_ranges_by_celltype', {}) or {}).items() if v}
        ctp = getattr(mw.ui, 'CelltypeDeterminationPanel', None)
        if ctp is not None and hasattr(ctp, 'FOVModeRadioButton'):
            if ranges:
                ctp.FOVModeRadioButton.setChecked(True)
                self.log(f'celltype: FOV mode, ranges for {sorted(ranges)}')
            elif not (getattr(mw, '_barcode_channel_by_celltype', None) or {}):
                raise ValueError('no celltype configuration in the store: set FOV ranges or barcode channels on the Celltype tab first')
        mw._run_celltype_determination()
        return (lambda: _running(getattr(mw, '_celltype_worker', None))), (lambda: 'celltype determination ended')

    def _run_cellcycle(self, fovs, mode):
        """Build counts, then fit (redo, or no model yet) or place with
        the stored model; each through the tab's own guarded worker."""
        mw = self.mw
        cc = mw.cellcycle
        p = mw.ui.CellCyclePanel
        if cc.model is None:
            cc._restore_from_store()
        p.FovListLineEdit.setText(','.join(str(f) for f in fovs))
        stages = ['counts', 'fit' if (mode == 'overwrite' or cc.model is None) else 'place']
        state = {'stages': stages, 'launched': None, 'done': []}

        def _launch(name):
            state['launched'] = name
            cc._guard({'counts': cc.build_counts, 'fit': cc.fit, 'place': cc.place}[name])

        def busy():
            w = getattr(cc, '_worker', None)
            if state['launched'] is not None and _running(w):
                return True
            if state['launched'] is not None:
                state['done'].append(state['launched'])
                state['launched'] = None
            if state['stages']:
                _launch(state['stages'].pop(0))
                return True
            return False
        _launch(state['stages'].pop(0))

        def finish():
            return ', '.join(state['done']) + ': ' + (p.ModelStatusLabel.text() or p.CountStatusLabel.text())
        return busy, finish

    def _run_tracing(self, fovs, mode):
        mw = self.mw
        if self.project is not None and not S.tracing_configured(self.project):
            raise ValueError('chromatin tracing is not configured (no tracing hybes checked): not building alleles')
        mw._build_chromatin_alleles_all_fovs()
        mw._run_chromatin_tracing_fit_all()
        return (lambda: _running(getattr(mw, '_chromatin_worker', None))), (lambda: 'chromatin tracing batch ended')


def project_from_app(mw):
    """The status module's project dict from the live app: storage paths
    and layouts from the Ingestion tab, the parsed hybe records, every
    tab's current parameters (the config the app would save), the
    Ingestion FOV list."""
    from codelab_pipeline.io import paths
    ip = mw.ui.IngestionPanel
    modalities = {}
    root = ''
    for name, data in (getattr(ip, 'modality_data', {}) or {}).items():
        sp = str((data or {}).get('storage_path', '') or '')
        if not sp:
            continue
        root = root or paths.project_root(sp)
        modalities[name] = {'storage_path': sp, 'layout_path': str((data or {}).get('layout_path', '') or ''),
                            'records': list((getattr(mw, 'hybe_records_by_modality', {}) or {}).get(name, [])),
                            'fields': dict(data or {})}
    try:
        params = mw._capture_config_params()
    except Exception:                                           # noqa: BLE001
        params = {}
    return {'config': 'app', 'project_root': root, 'fovs': mw._parse_fov_list(ip.FovListLineEdit.text()),
            'modalities': modalities, 'params': params, 'celltype_names': list(getattr(mw, 'current_celltype_list', None) or [])}


class PipelineTab:
    """The Pipeline tab over PipelineRunner: Refresh reads the store's
    status (a worker), Start runs the plan the policies imply, Stop ends
    after the current step; the runner's log lands in the tab."""

    def __init__(self, mw):
        self.mw = mw
        self.panel = mw.ui.PipelinePanel
        self.runner = PipelineRunner(mw)
        self.rows = None
        self.project = None
        self._worker = None
        p = self.panel
        p.RefreshPushButton.clicked.connect(lambda: self._guard(self.refresh))
        p.StartPushButton.clicked.connect(lambda: self._guard(self.start))
        p.StopPushButton.clicked.connect(lambda: self._guard(self.runner.stop))
        for combo in p.policy_combos.values():
            combo.currentTextChanged.connect(lambda _t: self._update_plan())
        self.runner.logged.connect(p.append_log)
        self.runner.step_started.connect(lambda s, f: p.ProgressLabel.setText(f'running {s} on {len(f)} FOV(s)...'))
        self.runner.step_finished.connect(lambda s, t: p.ProgressLabel.setText(t))
        self.runner.finished.connect(self._on_finished)

    def _guard(self, fn):
        try:
            fn()
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self.mw, TAB_TITLE, str(e))
        except Exception as e:                                  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self.mw, f'{TAB_TITLE} error', f'{type(e).__name__}: {e}')

    def _project(self):
        project = project_from_app(self.mw)
        if not project['modalities']:
            raise ValueError('No storage path set on the Ingestion tab.')
        if not project['fovs']:
            raise ValueError('The Ingestion tab has no FOV list.')
        if not any(m['records'] for m in project['modalities'].values()):
            raise ValueError('Parse the layouts on the Ingestion tab first.')
        return project

    def refresh(self, then=None):
        from windows.main_window import FnWorker
        p = self.panel
        project = self._project()
        deep = p.DeepCheckBox.isChecked()
        p.FovListLabel.setText(f'FOVs: {len(project["fovs"])} ({project["fovs"][0]}..{project["fovs"][-1]})')
        p.RefreshPushButton.setEnabled(False)
        p.SummaryLabel.setText('reading the store...')

        def _done(rows):
            p.RefreshPushButton.setEnabled(True)
            self.rows, self.project = rows, project
            self._update_plan()
            if then is not None:
                self._guard(then)

        def _fail(msg):
            p.RefreshPushButton.setEnabled(True)
            p.SummaryLabel.setText(f'status failed: {msg}')
        self._worker = FnWorker(lambda: S.status_table(project, deep=deep))
        self._worker.finished_ok.connect(lambda r: self._guard(lambda: _done(r)))
        self._worker.failed.connect(_fail)
        self._worker.start()

    def _update_plan(self):
        if self.rows is None:
            return
        plan = S.plan(self.rows, self.panel.policies())
        self.panel.set_status_rows(self.rows, plan)

    def start(self):
        if self.runner.running:
            raise ValueError('A run is already in progress.')
        if self.rows is None or self.project is None:
            self.refresh(then=self.start)
            return
        pol = self.panel.policies()
        plan = S.plan(self.rows, pol)
        if not any(plan[s] for s in S.STEPS if pol[s] != 'skip'):
            raise ValueError('Nothing to run: every step is done for these FOVs, or skipped.')
        self.panel.set_running(True)
        self.panel.append_log('---- run started ----')
        self.runner.start(self.project, plan, pol, explicit=self.panel.explicit_steps())

    def _on_finished(self, report):
        self.panel.set_running(False)
        self.panel.ProgressLabel.setText(f'finished in {report.get("seconds", 0):.0f}s; {len(report.get("failed", []))} failed'
                                         + (' (stopped early)' if report.get('stopped') else ''))
        # the store changed: read it again
        try:
            self.refresh()
        except Exception as exc:                                # noqa: BLE001
            self.panel.append_log(f'refresh after the run failed: {type(exc).__name__}: {exc}')


def run_headless(config_path, fovs=None, policies=None, explicit=(), log=print, timeout_s=None):
    """The CLI's door: an offscreen MainWindow, the runner, the event loop
    pumped until it finishes. Returns the report."""
    import os
    from unittest import mock
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    project = S.load_project(config_path)
    if fovs:
        project['fovs'] = list(fovs)
    rows = S.status_table(project)
    plan = S.plan(rows, policies)
    from windows.main_window import MainWindow

    def _dialog(kind):
        def _show(*args, **kwargs):
            log(f'[dialog {kind}] ' + ' | '.join(str(a) for a in args[1:3] if isinstance(a, str)))
            return QtWidgets.QMessageBox.Yes if kind == 'question' else QtWidgets.QMessageBox.Ok
        return _show
    patches = [mock.patch.object(QtWidgets.QMessageBox, k, staticmethod(_dialog(k))) for k in ('information', 'warning', 'critical', 'question')]
    for pt in patches:
        pt.start()
    try:
        mw = MainWindow(config_path)
        app.processEvents()
        if not mw.hybe_records_by_modality:
            mw._parse_layouts()
            app.processEvents()
        try:
            mw.cellcycle.populate_sources()
        except Exception as exc:                                # noqa: BLE001
            log(f'cellcycle sources not populated: {type(exc).__name__}: {exc}')
        runner = PipelineRunner(mw)
        runner.logged.connect(log)
        done = {'report': None}
        runner.finished.connect(lambda r: done.__setitem__('report', r))
        runner.start(project, plan, policies, explicit)
        t0 = time.time()
        while done['report'] is None:
            app.processEvents()
            time.sleep(0.05)
            if timeout_s and time.time() - t0 > timeout_s:
                runner.stop()
                log(f'timeout after {timeout_s}s: stopping after the current step')
                timeout_s = None
        return done['report']
    finally:
        for pt in patches:
            pt.stop()
