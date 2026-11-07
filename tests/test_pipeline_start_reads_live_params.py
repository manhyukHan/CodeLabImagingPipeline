"""Start reads the parameters as they stand, not as Refresh left them.

THE BUG THIS PINS. `PipelineTab.refresh()` cached a project dict, and
every parameter dialog afterwards wrote to the TABS -- so a run used
whatever the parameters were when someone last pressed Refresh status.
That was invisible for most settings and wrong for the round picker,
because an empty `localization_run/rounds` means EVERY target by design
and a stale snapshot reads empty. Reported from the real pipeline on
MP58: Refresh status, pick Hyb_107 alone, Start -- and the run began
Hyb_025, localizing all 84 rounds over the NAS.

Two checks, both on the method itself with a stub self, because the
whole of the fix is "which dict does start() hand the runner":

  1. a parameter changed after refresh reaches the runner
  2. a FOV list or storage path changed after refresh is REFUSED, since
     `self.rows` came from the store under the old one and
     project_from_app cannot re-read the store

Run:  QT_QPA_PLATFORM=offscreen python tests/test_pipeline_start_reads_live_params.py
"""
import os
import sys
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline.pipeline import status as S              # noqa: E402
from windows import pipeline_wiring as W                       # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}'
          + (f'   [{detail}]' if detail else ''), flush=True)


def project(rounds, fovs=(1, 2, 3), sp='S:/store/DNA'):
    return {'config': 'app', 'project_root': 'S:/store', 'fovs': list(fovs),
            'modalities': {'DNA': {'storage_path': sp, 'layout_path': '',
                                   'records': [], 'fields': {}}},
            'params': {'localization_run': {'rounds': rounds}},
            'celltype_names': []}


class StubRunner:
    running = False

    def __init__(self):
        self.got = None

    def start(self, project, plan, pol, explicit=None):
        self.got = project


def make_tab(live, cached):
    """A stub self carrying only what start() touches."""
    tab = SimpleNamespace(
        mw=object(), runner=StubRunner(), rows=[object()], project=cached,
        panel=SimpleNamespace(
            policies=lambda: {s: ('existing' if s == 'localization' else 'skip')
                              for s in S.STEPS},
            explicit_steps=lambda: (),
            set_running=lambda _v: None,
            append_log=lambda _m: None))
    tab.refresh = lambda then=None: None
    W.project_from_app = lambda _mw: live          # patched for the test
    W.S = S
    return tab


def run():
    real, orig_plan = W.project_from_app, S.plan
    # the plan comes from the STORE rows, which this test does not have
    # and does not need: it is about which project dict reaches the
    # runner, so the plan is stubbed non-empty for the whole run
    S.plan = lambda rows, pol: {s: ([1] if s == 'localization' else [])
                                for s in S.STEPS}
    try:
        print('a parameter changed after Refresh reaches the run')
        # the store was refreshed while every round was still selected
        tab = make_tab(live=project('DNA|Hyb_107'), cached=project(''))
        W.PipelineTab.start(tab)
        got = tab.runner.got
        chosen = (got or {}).get('params', {}).get(
            'localization_run', {}).get('rounds')
        check('the runner is handed the LIVE rounds, not the cached empty one',
              chosen == 'DNA|Hyb_107', f'runner got {chosen!r}')
        check('start() replaced its own cached project',
              tab.project.get('params', {}).get('localization_run', {}).get(
                  'rounds') == 'DNA|Hyb_107')

        print('a FOV list or storage path changed after Refresh is refused')
        for name, live in (('FOV list', project('', fovs=(1, 2, 3, 4))),
                           ('storage path', project('', sp='T:/other/DNA'))):
            tab = make_tab(live=live, cached=project(''))
            try:
                W.PipelineTab.start(tab)
                check(f'{name} change refuses the run', False, 'no error raised')
            except ValueError as exc:
                check(f'{name} change refuses the run',
                      'Refresh status' in str(exc), str(exc)[:60])
            check(f'{name} change does not start the runner',
                  tab.runner.got is None)
    finally:
        W.project_from_app, S.plan = real, orig_plan

    print()
    print(f'{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        print('failed: ' + ', '.join(FAIL))
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(run())
