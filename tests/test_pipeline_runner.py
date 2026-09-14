"""
The pipeline sequencer against a fake MainWindow: steps run in pipeline
order one at a time, each step's FOVs go into the Ingestion tab's FOV
list and the original text comes back, the batch-mode dialog answers
from the policy and message boxes go to the log, segmentation runs
only when chosen explicitly, stop() ends after the current step, and
the report carries every step.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_pipeline_runner.py
"""
import os
import sys
import time
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5 import QtWidgets                                    # noqa: E402

from codelab_pipeline.pipeline import status as S              # noqa: E402
from windows import pipeline_wiring as W                       # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}' + (f'   [{detail}]' if detail else ''), flush=True)


class FakeWorker:
    """isRunning() for `ticks` polls, then idle."""
    def __init__(self, ticks):
        self.ticks = ticks

    def isRunning(self):
        self.ticks -= 1
        return self.ticks >= 0


class FakeMW:
    def __init__(self):
        self.ui = SimpleNamespace(IngestionPanel=SimpleNamespace(FovListLineEdit=QtWidgets.QLineEdit('1,2,3'),
                                                                 IngestWorkersSpinBox=QtWidgets.QSpinBox()),
                                  CellSegmentPanel=SimpleNamespace(FovSpinBox=QtWidgets.QSpinBox()))
        self.calls = []
        self.modes = []
        self.logs = []
        self._active_ingestions = []
        self._alignment_worker = None
        self._cross_modal_queue = []
        self._cross_modal_worker = None
        self._cell_alignment_worker = None
        self._chromatin_worker = None
        self._celltype_worker = None
        self._segment_worker = None

    def log(self, m):
        self.logs.append(m)

    def _parse_fov_list(self, text):
        return [int(t) for t in text.replace(' ', '').split(',') if t]

    def _confirm_batch_mode(self, *a):
        raise AssertionError('the real dialog must not be reached')

    def _confirm_overwrite(self, *a):
        raise AssertionError('the real dialog must not be reached')

    # the batch actions: record the FOV list text and the answered mode
    def _run_ingestion(self):
        self.calls.append(('ingestion', self.ui.IngestionPanel.FovListLineEdit.text()))
        self.modes.append(self._confirm_overwrite())
        w = FakeWorker(2)
        self._active_ingestions.append(w)
        self._ingest_worker = w

    def _run_fov_alignment_all(self):
        self.calls.append(('fov_alignment', self.ui.IngestionPanel.FovListLineEdit.text()))
        self.modes.append(self._confirm_batch_mode('t', 'w'))
        QtWidgets.QMessageBox.information(None, 'FOV alignment', 'Nothing to append -- muted into the log')
        self._alignment_worker = FakeWorker(3)

    def _run_cross_modal_alignment_all(self):
        self.calls.append(('cross_modal', self.ui.IngestionPanel.FovListLineEdit.text()))
        self.modes.append(self._confirm_batch_mode('t', 'w'))
        self._cross_modal_worker = FakeWorker(1)

    def _run_cell_alignment_all_fovs(self):
        self.calls.append(('cell_alignment', self.ui.IngestionPanel.FovListLineEdit.text()))
        self.modes.append(self._confirm_batch_mode('t', 'w'))
        self._cell_alignment_worker = FakeWorker(2)

    def _run_celltype_determination(self):
        self.calls.append(('celltype', self.ui.IngestionPanel.FovListLineEdit.text()))
        if QtWidgets.QMessageBox.question(None, 'q', 'proceed?') != QtWidgets.QMessageBox.Yes:
            raise AssertionError('question must answer Yes')

    def _build_chromatin_alleles_all_fovs(self):
        self.calls.append(('build_alleles', self.ui.IngestionPanel.FovListLineEdit.text()))

    def _run_chromatin_tracing_fit_all(self):
        self.calls.append(('tracing', self.ui.IngestionPanel.FovListLineEdit.text()))
        self.modes.append(self._confirm_batch_mode('t', 'w'))
        self._chromatin_worker = FakeWorker(2)

    def _run_cell_segmentation(self):
        self.calls.append(('segment', self.ui.CellSegmentPanel.FovSpinBox.value()))
        self._segment_worker = FakeWorker(1)

    def _save_cells(self):
        self.calls.append(('save_cells', self.ui.CellSegmentPanel.FovSpinBox.value()))


def drive(app, runner, max_s=20):
    done = {'r': None}
    runner.finished.connect(lambda r: done.__setitem__('r', r))
    t0 = time.time()
    while done['r'] is None and time.time() - t0 < max_s:
        app.processEvents()
        time.sleep(0.01)
        # the fake ingestion worker 'finishes' by leaving the live list
        mw = runner.mw
        if mw._active_ingestions and not mw._active_ingestions[0].isRunning():
            mw._active_ingestions.clear()
    return done['r']


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])   # noqa: F841
    W.POLL_MS = 5
    print('order, FOV lists, modes, dialogs')
    mw = FakeMW()
    runner = W.PipelineRunner(mw)
    plan = {'ingestion': [4, 5], 'segmentation': [4, 5], 'fov_alignment': [4, 5], 'cross_modal': [5],
            'cell_alignment': [4], 'localization': [], 'celltype': [4, 5], 'cellcycle': [], 'tracing': [4]}
    runner.project = None
    runner.start(project=None, plan=plan, policies={'cell_alignment': 'redo', 'tracing': 'skip'}, explicit=('cell_alignment', 'tracing'))
    r = drive(app, runner)
    check('the run finishes', r is not None and not r['stopped'])
    steps = [c[0] for c in mw.calls]
    check('steps run in pipeline order, tracing skipped, segmentation not chosen -> not run',
          steps == ['ingestion', 'fov_alignment', 'cross_modal', 'cell_alignment', 'celltype'], str(steps))
    check('each step sees its own FOV list', dict(mw.calls) == {'ingestion': '4,5', 'fov_alignment': '4,5', 'cross_modal': '5',
                                                              'cell_alignment': '4', 'celltype': '4,5'}, str(mw.calls))
    check('the batch-mode dialog answers from the policy (append by default, overwrite for redo), ingestion always append',
          mw.modes == ['append', 'append', 'append', 'overwrite'], str(mw.modes))
    check('the FOV list text is restored', mw.ui.IngestionPanel.FovListLineEdit.text() == '1,2,3')
    check('message boxes went to the log', any('Nothing to append' in m for m in mw.logs), str([m for m in mw.logs if 'Nothing' in m]))
    check('segmentation left-alone is logged', any('left alone' in m for m in mw.logs))
    check('the report lists the steps', [e['step'] for e in r['steps']] == ['ingestion', 'fov_alignment', 'cross_modal', 'cell_alignment', 'celltype'])
    check('dialogs are restored after the run', mw._confirm_batch_mode.__func__ is FakeMW._confirm_batch_mode
          and QtWidgets.QMessageBox.information is not None)

    print('segmentation when chosen explicitly, and stop() after the current step')
    mw = FakeMW()
    runner = W.PipelineRunner(mw)
    plan = {'segmentation': [7, 8], 'fov_alignment': [7, 8], 'celltype': [7]}
    runner.start(project=None, plan=plan, policies={'segmentation': 'existing'}, explicit=('segmentation',))
    runner.step_finished.connect(lambda s, t: runner.stop() if s == 'segmentation' else None)
    r = drive(app, runner)
    seq = [c for c in mw.calls]
    check('segmentation runs FOV by FOV with a save after each', seq[:4] == [('segment', 7), ('save_cells', 7), ('segment', 8), ('save_cells', 8)], str(seq))
    check('stop() ends the run after the current step', r['stopped'] and 'fov_alignment' not in [c[0] for c in seq], str(seq))

    print('a step that raises at start is reported, the run continues')
    mw = FakeMW()
    mw._run_fov_alignment_all = lambda: (_ for _ in ()).throw(ValueError('no reference hybe'))
    runner = W.PipelineRunner(mw)
    runner.start(project=None, plan={'fov_alignment': [1], 'celltype': [1]}, policies={})
    r = drive(app, runner)
    check('the failure is in the report and the next step still ran', r['failed'] and r['failed'][0][0] == 'fov_alignment'
          and [c[0] for c in mw.calls] == ['celltype'], str((r['failed'], mw.calls)))

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        print('FAILED:', FAIL)
        sys.exit(1)
    print('ALL GOOD')


if __name__ == '__main__':
    main()
