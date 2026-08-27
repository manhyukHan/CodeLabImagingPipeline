"""
v1 against v2 driven through the APP, not through the functions.

WHY NOT A FUNCTION-LEVEL COMPARISON
-----------------------------------
Because the wiring is part of what is being tested. Calling
tracing_v2.trace_allele directly skips everything the panel does on the
way there: reading the engine and the voxel size off the widgets,
resolving and INSTALLING the readout PSF into the store, and -- the part
that actually bit -- using each experiment's OWN tuned parameters from
its config file instead of a harness author's defaults.

The first version of this comparison passed hardcoded defaults and got
127 of 160 hybe-fits rejected on HoxA for v1. That is a harness result,
not an engine result.

So this drives a real MainWindow headless: load the config, check the
hybes, build the alleles through the panel's own handler, set the engine
combo, press Fit All FOVs, and pump the Qt event loop until the worker
signals done. Both arms differ in exactly one widget.

Usage:
    python tools/engine_ab_app.py --exp MP58 --alleles 24
    python tools/engine_ab_app.py --all
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.experiments import EXPERIMENTS, config_fovs           # noqa: E402
from tools.fit_testbox import replicate_pairs                    # noqa: E402

VOXEL = (0.208, 0.208, 0.2)
DEFAULT_OUT = os.path.join('notes', 'chromatin_tracing_optimization')
_APP = None


def _session(exp):
    """A real MainWindow on this experiment's config, dialogs suppressed.

    Every QMessageBox is patched to its affirmative answer BEFORE the
    window is built: a modal dialog in a headless run blocks forever with
    nothing on screen to answer, which looks exactly like a hang.
    """
    global _APP
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from unittest import mock
    from PyQt5 import QtWidgets
    _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    for m in ('critical', 'warning', 'information', 'question'):
        mock.patch.object(QtWidgets.QMessageBox, m,
                          return_value=QtWidgets.QMessageBox.Yes).start()
    from windows.main_window import MainWindow
    mw = MainWindow(exp.config)
    # Fit All FOVs asks replace-or-append. Answer 'replace' every time:
    # append would merge into whatever the previous arm left behind and
    # the second engine would be scored on the first engine's fits.
    mock.patch.object(MainWindow, '_confirm_batch_mode',
                      return_value='replace').start()
    return mw


def _pump(mw, timeout_s):
    """Run the Qt loop until the tracing worker finishes.

    The worker is a QThread that reports through signals, so there has to
    be a loop for them to arrive on. Polling isFinished() without one
    would spin forever with the queued signals never delivered.
    """
    from PyQt5 import QtCore
    w = getattr(mw, '_chromatin_worker', None)
    if w is None:
        return 'no worker started'
    loop = QtCore.QEventLoop()
    state = {'msg': None}

    def _ok(_r):
        state['msg'] = 'ok'
        loop.quit()

    def _fail(m):
        state['msg'] = f'failed: {m}'
        loop.quit()

    w.finished_ok.connect(_ok)
    w.failed.connect(_fail)
    timer = QtCore.QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(lambda: (state.update(msg='TIMEOUT'), loop.quit()))
    timer.start(int(timeout_s * 1000))
    loop.exec_()
    timer.stop()
    w.wait(5000)
    # finished_ok fires before the per-FOV results have necessarily been
    # DELIVERED: fov_done is a queued signal, so its slot runs on the next
    # trip through the loop. Reading chromatin_alleles without draining the
    # queue first reports the alleles as they were BEFORE the run -- traced
    # 0, rejected 0, which reads as "the engine did nothing" rather than
    # "the results have not arrived yet".
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance()
    for _ in range(20):
        if app is not None:
            app.processEvents()
    return state['msg']


def _prepare(mw, exp, fovs, n_alleles):
    """Check the traced hybes, pick the reference, build the alleles."""
    chp = mw.ui.ChromatinTracingPanel
    from PyQt5 import QtCore
    lw = chp.HybeListWidget
    wanted, ref_item = 0, None
    # By DATATYPE, from the layout -- not by name prefix. A prefix match on
    # ('Hyb', 'Rep', 'Toe') swept in HoxA's five BLANK rounds, which are
    # named like the rest and are not part of the traced set: 5 of the 45
    # "rejections" were rounds that should never have been attempted.
    sp0 = mw._storage_path_for_modality(exp.modality)
    traced = {r['folder'] for r in mw._hybe_records_for_storage_path(sp0)
              if str(r['datatype']).upper() in exp.datatypes}
    for i in range(lw.count()):
        item = lw.item(i)
        name = item.text().split()[0]
        keep = name in traced
        item.setCheckState(QtCore.Qt.Checked if keep else QtCore.Qt.Unchecked)
        wanted += 1 if keep else 0
    for i in range(chp.ReferenceHybeComboBox.count()):
        if chp.ReferenceHybeComboBox.itemText(i).startswith(exp.reference_hybe):
            chp.ReferenceHybeComboBox.setCurrentIndex(i)
            ref_item = chp.ReferenceHybeComboBox.itemText(i)
            break
    built = 0
    per = max(1, n_alleles // max(len(fovs), 1))
    sp = mw._storage_path_for_modality(exp.modality)
    for fov in fovs:
        chp.AlleleFovSpinBox.setValue(fov)
        mw._on_chromatin_allele_fov_changed()
        for i in range(chp.AlleleHybeComboBox.count()):
            if chp.AlleleHybeComboBox.itemText(i).startswith(exp.anchor_hybe):
                chp.AlleleHybeComboBox.setCurrentIndex(i)
                break
        mw._on_chromatin_allele_hybe_changed()
        for i in range(chp.AlleleChannelComboBox.count()):
            if str(exp.anchor_channel) in chp.AlleleChannelComboBox.itemText(i):
                chp.AlleleChannelComboBox.setCurrentIndex(i)
                break
        mw._refresh_chromatin_allele_spot_choices()
        # Build Alleles reads the panel's own SPOT LIST SELECTION -- it is
        # not driven by the combos alone. Selecting rows here is what a
        # person clicking in the list does, and skipping it is why the
        # first attempt built zero alleles from a fully-populated panel.
        lst = chp.SpotListWidget
        lst.clearSelection()
        n = min(per, lst.count())
        for i in range(n):
            lst.item(i).setSelected(True)
        if n:
            mw._build_chromatin_alleles_from_selection()
        got = mw.chromatin_alleles.get((sp, fov), [])
        built += len(got)
    return wanted, ref_item, built


def _score(mw, exp, fovs, pairs):
    sp = mw._storage_path_for_modality(exp.modality)
    dy, dx, dz = VOXEL
    out, rejected, traced = {}, 0, 0
    for fov in fovs:
        for allele in mw.chromatin_alleles.get((sp, fov), []):
            poly = {}
            for h, comps in (allele.polymer or {}).items():
                if comps:
                    best = max(comps, key=lambda c: c[3])
                    poly[h] = (float(best[0]), float(best[1]), float(best[2]))
            rejected += len(allele.rejected_hybes or {})
            traced += len(poly)
            for a, b, _rid in pairs:
                if a in poly and b in poly:
                    d = (np.array(poly[a]) - np.array(poly[b])) * np.array([dx, dy, dz])
                    out[(fov, allele.id, a, b)] = (float(np.linalg.norm(d)),
                                                   float(np.linalg.norm(d[:2])))
    return out, rejected, traced


def run_one(name, n_alleles, psf_label, timeout_s):
    exp = EXPERIMENTS[name]
    mw = _session(exp)
    chp = mw.ui.ChromatinTracingPanel
    sp = mw._storage_path_for_modality(exp.modality)
    records = mw._hybe_records_for_storage_path(sp)
    pairs = [(a, b, r) for a, b, r in replicate_pairs(records)]
    fovs = list(exp.fovs) if exp.fovs else list(config_fovs(exp.config))

    n_checked, ref, built = _prepare(mw, exp, fovs, n_alleles)
    print(f'=== {name} ({exp.scope_mb} Mb) ===')
    print(f'  store    : {sp}')
    print(f'  hybes    : {n_checked} checked   reference: {ref}')
    print(f'  alleles  : {built} over FOV {fovs}   pairs/allele: {len(pairs)}')
    if not built:
        print('  no alleles built -- skipping\n')
        return None

    chp.refresh_psf_entries(select=psf_label)
    row = {'experiment': name, 'scope_mb': exp.scope_mb, 'alleles': built,
           'psf': psf_label, 'pairs_per_allele': len(pairs)}
    scores = {}
    for engine_text, arm in ((None, 'v1'), (None, 'v2')):
        # exactly one widget differs between the arms
        target = [t for t in (chp.EngineComboBox.itemText(i)
                              for i in range(chp.EngineComboBox.count()))
                  if (t.lower().startswith('v2') if arm == 'v2'
                      else not t.lower().startswith('v2'))]
        chp.EngineComboBox.setCurrentText(target[0])
        print(f'  running {arm}: engine combo = {chp.EngineComboBox.currentText()!r}',
              flush=True)
        t0 = time.perf_counter()
        mw._run_chromatin_tracing_fit_all()
        msg = _pump(mw, timeout_s)
        dt = time.perf_counter() - t0
        d, rej, traced = _score(mw, exp, fovs, pairs)
        scores[arm] = d
        row[f'{arm}_status'] = msg
        row[f'{arm}_wall_s'] = round(dt, 1)
        row[f'{arm}_rejected'] = rej
        row[f'{arm}_traced'] = traced
        row[f'{arm}_pairs'] = len(d)
        if d:
            a3 = np.array([v[0] for v in d.values()])
            row[f'{arm}_median_3d_um'] = round(float(np.median(a3)), 4)
            row[f'{arm}_median_xy_um'] = round(float(np.median(
                [v[1] for v in d.values()])), 4)
            row[f'{arm}_p90_3d_um'] = round(float(np.percentile(a3, 90)), 4)
        print(f'    {arm}: {msg}  {dt:.0f}s  traced {traced}  rejected {rej}  '
              f'pairs {len(d)}'
              + (f'  median {row.get(f"{arm}_median_3d_um")} um' if d else ''),
              flush=True)

    common = sorted(set(scores['v1']) & set(scores['v2']))
    row['common_pairs'] = len(common)
    if common:
        m1 = float(np.median([scores['v1'][k][0] for k in common]))
        m2 = float(np.median([scores['v2'][k][0] for k in common]))
        row['common_v1_median_3d_um'] = round(m1, 4)
        row['common_v2_median_3d_um'] = round(m2, 4)
        row['v2_change_pct'] = round(100 * (m2 - m1) / m1, 1)
        row['v2_closer_pairs'] = sum(1 for k in common
                                     if scores['v2'][k][0] < scores['v1'][k][0])
    print(flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exp', action='append', default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--alleles', type=int, default=None)
    ap.add_argument('--psf', default='universal-default')
    ap.add_argument('--timeout', type=float, default=5400)
    ap.add_argument('--out', default=DEFAULT_OUT)
    a = ap.parse_args()

    names = list(EXPERIMENTS) if a.all else (a.exp or ['MP58'])
    # Comparable PAIRS, not comparable alleles: HoxA has 2 replicate pairs
    # per allele against CrossMod's 8.
    sizes = {'MP58': 24, 'Chr19': 32, 'CrossMod': 20, 'HoxA': 60}
    rows = []
    for name in names:
        try:
            r = run_one(name, a.alleles or sizes.get(name, 24), a.psf, a.timeout)
            if r:
                rows.append(r)
        except Exception as e:
            import traceback
            print(f'  {name} FAILED: {type(e).__name__}: {e}')
            traceback.print_exc()

    if not rows:
        return
    print('=' * 104)
    print(f'{"experiment":<11}{"Mb":>6}  {"common":>8}  {"v1 median":>11}'
          f'{"v2 median":>11}{"change":>9}  {"v2 closer":>11}  '
          f'{"v1 traced":>10}{"v2 traced":>10}')
    print('=' * 104)
    for r in rows:
        c = r.get('common_pairs', 0)
        print(f'{r["experiment"]:<11}{r["scope_mb"]:>6}  {c:>8}  '
              f'{r.get("common_v1_median_3d_um", float("nan")):>10.4f}u'
              f'{r.get("common_v2_median_3d_um", float("nan")):>10.4f}u'
              f'{r.get("v2_change_pct", float("nan")):>8.1f}%  '
              f'{r.get("v2_closer_pairs", 0):>4}/{c:<5}  '
              f'{r.get("v1_traced", 0):>10}{r.get("v2_traced", 0):>10}')
    print('\nnegative change = v2 better. traced counts are shown so neither '
          'engine\ncan look good by rejecting the hard rounds.')
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, 'engine_ab_app.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
