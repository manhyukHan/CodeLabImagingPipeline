"""
Cell-overlay figure contract:
1. spec/block order follows the SYSTEM modality order (Ingestion tab's
   configured list), never the clicked cell's own modality;
2. the FIGURE callers drop datatype-B hybes (H/R/T only), while the
   data-consumer call (no figure_datatypes) keeps every datatype.
Calls the real MainWindow._cell_overlay_target_specs unbound, against a
stub carrying only the collaborators it touches.
Run: python tests/test_overlay_specs.py
"""
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt5 import QtWidgets                       # noqa: E402
_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
from windows.main_window import MainWindow        # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(('  ok ' if cond else 'FAIL ') + name + (f' -- {detail}' if detail and not cond else ''))


def rec(folder, datatype):
    return {'folder': folder, 'datatype': datatype, 'fiducial_channel': 555,
            'channels': [555, 635]}


DNA_RECS = [rec('D_H', 'H'), rec('D_B', 'B'), rec('D_R', 'R')]
RNA_RECS = [rec('R_T', 'T'), rec('R_B', 'B')]


class Cell:
    id = 1
    reference_hybe = 'D_H'
    reference_modality = 'DNA'
    matrices = {}


class Stub:
    """Only what _cell_overlay_target_specs actually touches."""
    CELL_OVERLAY_FIGURE_DATATYPES = MainWindow.CELL_OVERLAY_FIGURE_DATATYPES

    def __init__(self, modality_names, cell_modality):
        self.ui = types.SimpleNamespace(
            IngestionPanel=types.SimpleNamespace(modality_names=modality_names))
        self._cell_modality = cell_modality

    def _modality_for_storage_path(self, storage_path):
        return self._cell_modality

    def _storage_path_for_modality(self, name):
        return f'/store/{name}'

    def _active_hybe_records_for_modality(self, name):
        return {'DNA': DNA_RECS, 'RNA': RNA_RECS}.get(name, [])

    def _fov_matrices_for_cell_modality(self, modality, cell, fov):
        recs = self._active_hybe_records_for_modality(modality)
        return {(r['folder'], modality): np.eye(3) for r in recs}

    def _matrix_to_shared(self, hybe, modality, cell, fov):
        return np.eye(3)


def specs_for(modality_names, cell_modality, **kw):
    stub = Stub(modality_names, cell_modality)
    home_recs = DNA_RECS if cell_modality == 'DNA' else RNA_RECS
    return MainWindow._cell_overlay_target_specs(
        stub, Cell(), f'/store/{cell_modality}', 1, home_recs, 'readout', **kw)


def block_order(specs):
    """modalities in first-appearance order -- exactly how the canvas
    groups them into blocks."""
    out = []
    for s in specs:
        if s['modality'] not in out:
            out.append(s['modality'])
    return out


def main():
    # 1. order follows the system list, whichever cell was clicked
    dna_click = specs_for(['DNA', 'RNA'], 'DNA', figure_datatypes=MainWindow.CELL_OVERLAY_FIGURE_DATATYPES)
    rna_click = specs_for(['DNA', 'RNA'], 'RNA', figure_datatypes=MainWindow.CELL_OVERLAY_FIGURE_DATATYPES)
    check('DNA-cell click -> DNA block first', block_order(dna_click) == ['DNA', 'RNA'],
          str(block_order(dna_click)))
    check('RNA-cell click -> STILL DNA block first (was RNA-first)',
          block_order(rna_click) == ['DNA', 'RNA'], str(block_order(rna_click)))
    check('block order identical regardless of click',
          block_order(dna_click) == block_order(rna_click))

    # order follows the configured list, so reversing it reverses blocks
    rev = specs_for(['RNA', 'DNA'], 'DNA', figure_datatypes=MainWindow.CELL_OVERLAY_FIGURE_DATATYPES)
    check("system order ['RNA','DNA'] -> RNA block first", block_order(rev) == ['RNA', 'DNA'],
          str(block_order(rev)))

    # 2. figure callers drop B; data callers keep everything
    fig_hybes = sorted(s['hybe'] for s in dna_click)
    check('figure specs omit datatype B', fig_hybes == ['D_H', 'D_R', 'R_T'], str(fig_hybes))
    data_hybes = sorted(s['hybe'] for s in specs_for(['DNA', 'RNA'], 'DNA'))
    check('data specs (no filter) keep B', data_hybes == ['D_B', 'D_H', 'D_R', 'R_B', 'R_T'],
          str(data_hybes))

    # a cell whose modality is not in the configured list still resolves
    orphan = specs_for(['RNA'], 'DNA', figure_datatypes=MainWindow.CELL_OVERLAY_FIGURE_DATATYPES)
    check("unconfigured cell modality still appears (after configured ones)",
          block_order(orphan) == ['RNA', 'DNA'], str(block_order(orphan)))

    # 3. clicking a per-hybe Results row shows the ONE-VS-ONE figure
    from PyQt5 import QtCore
    drawn = {}

    class ClickStub:
        _cell_per_hybe_context = {'fov': 1, 'cell': Cell(), 'storage_path': '/store/DNA',
                                  'hybe_records': DNA_RECS}

        def _show_cell_alignment_preview_for_hybe(self, target_key=None):
            drawn['target_key'] = target_key
            drawn['ctx_target'] = self._cell_preview_context.get('target_key')

    item = QtWidgets.QListWidgetItem('D_R row')
    item.setData(QtCore.Qt.UserRole, (1, 1, 'D_R', 'DNA'))
    stub = ClickStub()
    MainWindow._show_cell_alignment_preview(stub, item)
    check('click routes the clicked row to the one-vs-one preview',
          drawn.get('target_key') == ('D_R', 'DNA'), str(drawn))
    check('click stores the target in the preview context',
          drawn.get('ctx_target') == ('D_R', 'DNA'), str(drawn))

    # the real handler must show/raise the window in CANVAS mode (it draws
    # live) and draw the 3-col figure
    import inspect
    src = inspect.getsource(MainWindow._show_cell_alignment_preview_for_hybe)
    check('one-vs-one handler shows + raises the preview window',
          'alignment_preview_window.show_canvas()' in src
          and 'alignment_preview_window.raise_()' in src)
    check('one-vs-one handler draws the 3-col figure',
          'draw_cell_alignment_preview_3col' in src)
    # and the one-vs-all figure must NOT hijack the window on accept
    accept_src = inspect.getsource(MainWindow._accept_per_cell_alignment)
    check('accept does not pop the one-vs-all window over it',
          'alignment_preview_window.show' not in accept_src)

    # 4. the one-vs-all viewer SHOWS the saved PNG instead of recomputing
    overlay_src = inspect.getsource(MainWindow._show_cell_all_readouts_overlay)
    check('one-vs-all viewer displays the saved PNG when it exists',
          '_cell_overlay_png_path' in overlay_src
          and 'os.path.exists(png)' in overlay_src
          and 'show_image' in overlay_src)
    check('one-vs-all viewer never draws on the GUI thread',
          'preview_canvas.draw_cell_all_readouts_overlay' not in overlay_src)
    check('missing-PNG fallback renders in the background and saves',
          '_start_cell_overlay_save_worker' in overlay_src)

    # 5. the expensive save doors are backgrounded + warned
    saveall_src = inspect.getsource(MainWindow._save_all_cell_overlays)
    check('Save All warns about the cost before starting',
          'QMessageBox.question' in saveall_src and 'Estimated time' in saveall_src)
    check('Save All renders in the background',
          '_start_cell_overlay_save_worker' in saveall_src
          and 'preview_canvas.draw_cell_all_readouts_overlay' not in saveall_src)

    # 6. celltype determination is background + FOV-parallel
    ct_src = inspect.getsource(MainWindow._run_celltype_determination_body)
    check('celltype barcode mode runs in a background worker',
          'FnWorker' in ct_src and '_celltype_worker' in ct_src)
    check('celltype barcode mode fans out per FOV',
          'ThreadPoolExecutor' in ct_src)

    print()
    print(f'{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        raise SystemExit('FAILURES: ' + ', '.join(FAIL))
    print('ALL GOOD')


if __name__ == '__main__':
    main()
