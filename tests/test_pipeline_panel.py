"""
The Pipeline tab's widget contract: policies (segmentation defaults to
skip, the round trip through set_policies), the explicit-steps rule, the
FOV x step grid from status rows with the plan's 'to run' counts, the
config-map names, and the tab's guard against a store-less app.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_pipeline_panel.py
"""
import os
import sys
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5 import QtWidgets                                    # noqa: E402

from codelab_pipeline.pipeline import status as S              # noqa: E402
from ui.pipeline_panel import PipelinePanelUI, COMBO_NAMES     # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}' + (f'   [{detail}]' if detail else ''), flush=True)


def st(state, have=0, want=1):
    return {'state': state, 'have': have, 'want': want, 'detail': f'{state} detail'}


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])   # noqa: F841
    w = QtWidgets.QWidget()
    p = PipelinePanelUI()
    p.setupUi(w)
    print('policies')
    pol = p.policies()
    check('every step has a policy; segmentation defaults to skip, the rest to existing',
          set(pol) == set(S.STEPS) and pol['segmentation'] == 'skip' and all(v == 'existing' for k, v in pol.items() if k != 'segmentation'), str(pol))
    check('segmentation is not explicit until chosen', p.explicit_steps() == ())
    p.set_policies({'segmentation': 'existing', 'tracing': 'redo', 'bogus': 'redo', 'celltype': 'sometimes'})
    pol = p.policies()
    check('set_policies round-trips known steps and ignores unknown steps/policies',
          pol['segmentation'] == 'existing' and pol['tracing'] == 'redo' and pol['celltype'] == 'existing', str(pol))
    check('segmentation becomes explicit once chosen', p.explicit_steps() == ('segmentation',))
    check('the config-map widget names exist on the panel', all(hasattr(p, n) for n in COMBO_NAMES.values()))

    print('status grid and plan')
    rows = [{'fov': 1, **{s: st('done') for s in S.STEPS}},
            {'fov': 2, **{s: st('missing') for s in S.STEPS}, 'localization': st('partial', 3, 9), 'cross_modal': st('n/a', 0, 0)}]
    plan = S.plan(rows, p.policies())
    p.set_status_rows(rows, plan)
    check('one grid row per FOV, one column per step plus the FOV', p.StatusTable.rowCount() == 2 and p.StatusTable.columnCount() == len(S.STEPS) + 1)
    check('a partial cell shows have/want', p.StatusTable.item(1, 1 + S.STEPS.index('localization')).text() == 'partial 3/9')
    check('the summary counts the states', 'localize: 1 done, 1 partial, 0 missing' in p.SummaryLabel.text(), p.SummaryLabel.text())
    to_run = {s: p.StepsTable.item(i, 2).text() for i, s in enumerate(S.STEPS)}
    check('the to-run column follows the plan (existing -> FOV 2 only; cross-modal n/a there -> none; segmentation existing -> 1; tracing redo -> 2)',
          to_run['ingestion'] == '1' and to_run['cross_modal'] == '' and to_run['tracing'] == '2' and to_run['segmentation'] == '1', str(to_run))
    p.set_running(True)
    check('running disables Start/Refresh/policies and enables Stop',
          not p.StartPushButton.isEnabled() and p.StopPushButton.isEnabled() and not p.IngestionPolicyComboBox.isEnabled())
    p.set_running(False)
    p.append_log('hello')
    check('log appends', 'hello' in p.LogPlainTextEdit.toPlainText())

    print('the tab wiring guards')
    from windows import pipeline_wiring as W
    ip = SimpleNamespace(FovListLineEdit=QtWidgets.QLineEdit('1,2'), modality_data={}, IngestWorkersSpinBox=QtWidgets.QSpinBox())
    mw = SimpleNamespace(ui=SimpleNamespace(PipelinePanel=p, IngestionPanel=ip), log=lambda m: None, hybe_records_by_modality={},
                         _parse_fov_list=lambda t: [int(x) for x in t.split(',') if x], _capture_config_params=lambda: {},
                         current_celltype_list=[], _confirm_batch_mode=lambda *a: None, _confirm_overwrite=lambda *a: None)
    tab = W.PipelineTab(mw)
    try:
        tab._project()
        check('a store-less app is refused with a ValueError', False)
    except ValueError as e:
        check('a store-less app is refused with a ValueError', 'storage path' in str(e), str(e))
    proj = W.project_from_app(mw)
    check('project_from_app returns the status shape', set(proj) >= {'project_root', 'fovs', 'modalities', 'params', 'celltype_names'} and proj['fovs'] == [1, 2])

    print('the main window shell')
    from ui.main_window_ui import MainWindowUI
    mwin = QtWidgets.QMainWindow()
    ui = MainWindowUI()
    ui.setupUi(mwin)
    titles = [ui.tabWidget.tabText(i) for i in range(ui.tabWidget.count())]
    check('the Pipeline tab is the last tab', titles[-1] == 'Pipeline', str(titles))

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        print('FAILED:', FAIL)
        sys.exit(1)
    print('ALL GOOD')


if __name__ == '__main__':
    main()
