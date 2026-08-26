"""
Both Alignment Results lists are scoped to their OWN FOV spinbox, and
show every modality for that one FOV.

They used to list every FOV in the Ingestion tab's FOV list, which on
the real project is ~100 hybes x 40 FOVs x 2 modalities in one widget --
the rows you actually want being the ones for the FOV you are looking
at. This mirrors "Results (per cell, per hybe)", which is scoped to one
cell.

Also pins the panel layout the request specified: Overlay channel sits
between Max shift and Run Current FOV Alignment.

Run: python tests/test_alignment_results_scope.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np                                   # noqa: E402
from unittest import mock                            # noqa: E402
from PyQt5 import QtCore, QtWidgets                  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
for _m in ('critical', 'warning', 'information', 'question'):
    mock.patch.object(QtWidgets.QMessageBox, _m,
                      return_value=QtWidgets.QMessageBox.Yes).start()

from windows.main_window import MainWindow           # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(('  ok ' if cond else 'FAIL ') + name + (f' -- {detail}' if detail and not cond else ''))


def rec(folder):
    return {'folder': folder, 'fiducial_channel': 555, 'channels': [555],
            'datatype': 'H', 'readout_id': 0, 'readout_name': '', 'total_frames': 2}


def main():
    mw = MainWindow()
    ap, ip = mw.ui.AlignmentPanel, mw.ui.IngestionPanel

    # --- layout the request specified -------------------------------
    lay = ap.SameModalityChannelTypeComboBox.parent().layout()
    labels = []
    for i in range(lay.rowCount()):
        lbl = lay.itemAt(i, QtWidgets.QFormLayout.LabelRole)
        fld = lay.itemAt(i, QtWidgets.QFormLayout.FieldRole)
        span = lay.itemAt(i, QtWidgets.QFormLayout.SpanningRole)
        w = (fld or span)
        text = lbl.widget().text() if lbl and lbl.widget() else ''
        if w is not None and w.widget() is not None and isinstance(w.widget(), QtWidgets.QPushButton):
            text = w.widget().text()
        labels.append(text)
    try:
        i_shift = labels.index('Max shift (0 = unbounded):')
        i_chan = labels.index('Overlay channel:')
        i_run = labels.index('Run Current FOV Alignment')
        ordered = i_shift < i_chan < i_run
    except ValueError:
        ordered = False
    check('Overlay channel sits between Max shift and Run Current FOV Alignment',
          ordered, str(labels))
    check('same-modality Results list is tall',
          ap.SameModalityResultsListWidget.minimumHeight() >= 320,
          str(ap.SameModalityResultsListWidget.minimumHeight()))
    check('cross-modal Results list is not capped short',
          ap.CrossModalResultsListWidget.maximumHeight() > 1000
          and ap.CrossModalResultsListWidget.minimumHeight() >= 160,
          f'min={ap.CrossModalResultsListWidget.minimumHeight()} '
          f'max={ap.CrossModalResultsListWidget.maximumHeight()}')

    # --- FOV scoping, same-modality ---------------------------------
    ip.modality_names = ['DNA', 'RNA']
    ip.FovListLineEdit.setText('1,2,3')
    ip.modality_data = {
        'DNA': {'storage_path': '/store/DNA', 'layout_path': '', 'dax_directory': '',
                'active_hybe_list': [rec('D_ref'), rec('D_a')]},
        'RNA': {'storage_path': '/store/RNA', 'layout_path': '', 'dax_directory': '',
                'active_hybe_list': [rec('R_ref'), rec('R_a')]},
    }
    ap.build_same_modality_reference_hybe_fields(['DNA', 'RNA'])
    ap.populate_same_modality_reference_hybe_choices(
        [(r, 'DNA') for r in ip.modality_data['DNA']['active_hybe_list']]
        + [(r, 'RNA') for r in ip.modality_data['RNA']['active_hybe_list']])
    ap.select_same_modality_reference_hybe('DNA', 'D_ref')
    ap.select_same_modality_reference_hybe('RNA', 'R_ref')

    mw._storage_path_for_modality = lambda m: {'DNA': '/store/DNA', 'RNA': '/store/RNA'}[m]
    mw._ingested_hybes_for_fov = lambda sp, fov, recs: ([r['folder'] for r in recs], [], [])
    mw._merge_fov_matrices = lambda *a, **k: None
    mw._fov_matrices_for = lambda sp, fov: None
    mw.fov_matrices = {}
    mw._pending_same_modality_alignment = None

    def fake_read(sp, fov, recs):
        # every FOV has matrices, so any FOV leaking in would be visible
        return {(r['folder'], 'DNA' if 'DNA' in sp else 'RNA'): np.eye(3) * fov for r in recs}

    import windows.main_window as MW
    with mock.patch.object(MW.alignment, 'read_same_modality_matrices', side_effect=fake_read):
        ap.SameModalityFovSpinBox.setValue(2)
        mw._refresh_same_modality_results_list()
        rows = [ap.SameModalityResultsListWidget.item(i).data(QtCore.Qt.UserRole)
                for i in range(ap.SameModalityResultsListWidget.count())]

    fovs = {f for f, _h, _m in rows}
    mods = {m for _f, _h, m in rows}
    check('same-modality list shows ONLY the spinbox FOV', fovs == {2}, str(fovs))
    check('same-modality list shows EVERY modality for it', mods == {'DNA', 'RNA'}, str(mods))
    check('every hybe of that FOV is listed',
          {(h, m) for _f, h, m in rows}
          == {('D_ref', 'DNA'), ('D_a', 'DNA'), ('R_ref', 'RNA'), ('R_a', 'RNA')},
          str(sorted((h, m) for _f, h, m in rows)))

    with mock.patch.object(MW.alignment, 'read_same_modality_matrices', side_effect=fake_read):
        ap.SameModalityFovSpinBox.setValue(3)
        mw._refresh_same_modality_results_list()
        rows3 = [ap.SameModalityResultsListWidget.item(i).data(QtCore.Qt.UserRole)
                 for i in range(ap.SameModalityResultsListWidget.count())]
    check('changing the FOV spinbox re-scopes the list',
          {f for f, _h, _m in rows3} == {3}, str({f for f, _h, _m in rows3}))

    # --- FOV scoping, cross-modal -----------------------------------
    mw._shared_frame_modality = lambda: 'RNA'
    mw._cross_modal_moving_modalities = lambda: ['DNA']
    mw.cross_modal_result = {}
    mw.cross_modal_z = {}
    mw._pending_cross_modal = None
    mw._pending_cross_modal_z = None
    mw._pending_cross_modal_quality = None
    with mock.patch.object(MW.analysis_store, 'read_cross_modal_matrix',
                           side_effect=lambda sp, fov, modality=None: np.eye(3) * fov), \
         mock.patch.object(MW.analysis_store, 'read_cross_modal_z',
                           side_effect=lambda sp, fov, modality=None: 0.0), \
         mock.patch.object(MW.analysis_store, 'read_cross_modal_quality',
                           side_effect=lambda sp, fov, modality=None: {}):
        ap.CrossModalFovSpinBox.setValue(2)
        mw._refresh_cross_modal_results_list()
        xrows = [ap.CrossModalResultsListWidget.item(i).text()
                 for i in range(ap.CrossModalResultsListWidget.count())]
    check('cross-modal list shows ONLY the spinbox FOV',
          len(xrows) == 1 and xrows[0].startswith('FOV002'), str(xrows))

    print()
    print(f'{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        raise SystemExit('FAILURES: ' + ', '.join(FAIL))
    print('ALL GOOD')


if __name__ == '__main__':
    main()
