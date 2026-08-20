"""
End-to-end clicked-flow for celltype determination on the migrated clone,
through the REAL doors: _show_barcode_overview -> _run_celltype_
determination (barcode mode) -> _show_celltype_result.

Armed deliberately with MIXED-modality barcode channels including the
cross-modal bridge hybe Hyb_130 (RNA side) -- the exact (hybe, modality)
ambiguity that raised the FrameMatrices TypeError -- plus a DNA channel.
Asserts: no error dialogs, cells actually classified, and the result
mask is orientation-correct (equal to the (y, x) ground truth, NOT equal
to its transpose).

CODELAB_CT_CONFIG is REQUIRED and must point at a config whose
storage_paths are a CLONE: the run persists celltypes through the real
mirror_write_cells door, so pointing it at real data would write there.
"""
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from unittest import mock
from PyQt5 import QtWidgets

CONFIG = os.environ.get('CODELAB_CT_CONFIG')
assert CONFIG and os.path.exists(CONFIG), \
    'set CODELAB_CT_CONFIG to a config whose storage_paths point at a CLONE (this flow WRITES celltypes)'
FOV = 1

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
from windows.main_window import MainWindow
from codelab_pipeline.alignment import chain as alignment
from codelab_pipeline.io import vlinks_store

failures = []


def check(ok, msg):
    print(('  PASS  ' if ok else '  FAIL  ') + msg)
    if not ok:
        failures.append(msg)


dialogs = {'warning': [], 'critical': [], 'information': []}


def rec(kind):
    def _f(parent, title, text, *a, **k):
        dialogs[kind].append(f'{title}: {text}')
        return QtWidgets.QMessageBox.Ok
    return _f


with mock.patch.object(QtWidgets.QMessageBox, 'information', rec('information')), \
        mock.patch.object(QtWidgets.QMessageBox, 'warning', rec('warning')), \
        mock.patch.object(QtWidgets.QMessageBox, 'critical', rec('critical')), \
        mock.patch.object(QtWidgets.QMessageBox, 'question',
                          return_value=QtWidgets.QMessageBox.Yes):
    w = MainWindow(CONFIG)
    w._activate_fov(FOV)
    ctp = w.ui.CelltypeDeterminationPanel

    # -- arm: two celltypes, barcode channels in DIFFERENT modalities,
    #    one of them the bridge hybe --
    rna_recs = {r['folder']: r for r in w._active_hybe_records_for_modality('RNA')}
    dna_recs = {r['folder']: r for r in w._active_hybe_records_for_modality('DNA')}
    ch_130 = alignment.pick_channel_by_type(rna_recs['Hyb_130'], 'readout')
    ch_002 = alignment.pick_channel_by_type(dna_recs['Hyb_002'], 'readout')
    bch_a = ('Hyb_130', ch_130, 'RNA')
    bch_b = ('Hyb_002', ch_002, 'DNA')
    print(f'  info  barcode channels: TypeA={bch_a}  TypeB={bch_b}')

    ctp.CelltypeNamesListWidget.clear()
    ctp.CelltypeNamesListWidget.addItems(['TypeA', 'TypeB'])
    w._barcode_channel_by_celltype = {'TypeA': bch_a, 'TypeB': bch_b}
    for bch in (bch_a, bch_b):
        w._barcode_calibration['scale'].setdefault(bch, {})[FOV] = 1.0
        w._barcode_calibration['lower_bound'].setdefault(bch, {})[FOV] = 300.0
        w._barcode_calibration['upper_bound'].setdefault(bch, {})[FOV] = 1500.0
    ctp.BarcodeModeRadioButton.setChecked(True)
    ctp.IncludeTransientCheckBox.setChecked(True)
    ctp.OverviewFovLineEdit.setText(str(FOV))

    # -- door 1: Show Barcode Channel Overview (raised TypeError before) --
    shown = {}
    with mock.patch.object(w.barcode_overview_displayer, 'set_data',
                           side_effect=lambda imgs, labels: shown.update(imgs=imgs, labels=labels)), \
            mock.patch.object(w.barcode_overview_displayer, 'show'), \
            mock.patch.object(w.barcode_overview_displayer, 'raise_'):
        w._show_barcode_overview()
    check(not dialogs['warning'] and not dialogs['critical'],
          f"overview raised no dialogs (got {dialogs['warning'] + dialogs['critical']})")
    check(len(shown.get('imgs', {})) == 2,
          f"overview shows both barcode channels ({list(shown.get('labels', {}).values())})")

    # -- door 2: Run Celltype Determination (barcode mode) --
    # expected counts are DISTINCT cells / owned spots: the permanent and
    # transient containers hold synced copies of the same cells, and the
    # log once double-counted them (confirmed real: "200 cell(s), 280
    # spot(s)" reported for a 100-cell, 140-owned-spot FOV).
    distinct = {(f, int(c.id)) for cont in w._celltype_cell_containers()
                for f, cells in cont.data.items() for c in cells.values()}
    n_expected_cells = len(distinct)
    n_expected_spots = sum(len(w.spot_container.of_cell(f, cid)) for f, cid in distinct)
    with mock.patch.object(w, '_show_celltype_result') as auto_show:
        w._run_celltype_determination()
    check(not dialogs['critical'], f"run raised no error dialog (got {dialogs['critical']})")
    summary = next((t for t in dialogs['information'] if 'classified' in t), '')
    import re as _re
    m = _re.search(r'(\d+) cell\(s\), (\d+) spot\(s\) classified', summary)
    check(m is not None and (int(m.group(1)), int(m.group(2))) == (n_expected_cells, n_expected_spots),
          f'summary counts DISTINCT cells/spots: expected {n_expected_cells}/{n_expected_spots}, '
          f'got {m.groups() if m else summary!r}')
    after = {c.id: c.celltype for c in w.cell_container_permanent.get_cells(FOV)}
    n_typed = sum(1 for v in after.values() if v in ('TypeA', 'TypeB'))
    check(n_typed > 0, f'{n_typed}/{len(after)} cells classified into TypeA/TypeB')
    check(auto_show.called, 'run auto-opened the result view')

    # spot celltypes persist to disk alongside the cells' (per explicit
    # correction: cells persisted, spots silently did not)
    sp = w._storage_path_for_modality('DNA') or w._storage_path_for_modality('RNA')
    disk_ct = {d['uid']: d.get('celltype', '') for d in vlinks_store.read_spots(sp, FOV)}
    session_ct = {s.uid: s.celltype for s in w.spot_container.all(FOV) if s.uid in disk_ct}
    n_disk_typed = sum(1 for v in disk_ct.values() if v in ('TypeA', 'TypeB'))
    check(n_disk_typed > 0, f'{n_disk_typed} spot celltypes persisted to vlinks.h5')
    check(all(disk_ct[u] == session_ct[u] for u in session_ct),
          'disk spot celltypes match the session state')

    # -- door 3: Show Celltype Result -- orientation ground truth --
    got = {}
    with mock.patch.object(w.celltype_result_displayer, 'set_data',
                           side_effect=lambda img, mask, ct: got.update(img=img, mask=mask, ct=ct)), \
            mock.patch.object(w.celltype_result_displayer, 'show'), \
            mock.patch.object(w.celltype_result_displayer, 'raise_'):
        w._show_celltype_result(FOV)
    check('mask' in got, 'result view received data')
    if 'mask' in got:
        cells = w.cell_container_permanent.get_cells(FOV)
        expected = np.zeros(cells[0].frame_shape, dtype=np.uint8)
        for c in cells:
            y, x = c.area
            expected[y.astype(int), x.astype(int)] = c.id
        check(np.array_equal(got['mask'], expected), 'mask matches the (y, x) ground truth')
        check(not np.array_equal(got['mask'], expected.T),
              'mask is NOT the transpose (the reported bug would fail this)')

print('ALL GOOD' if not failures else f'{len(failures)} FAILURES')
sys.exit(1 if failures else 0)
