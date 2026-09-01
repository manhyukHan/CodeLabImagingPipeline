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

    # 6. every worker that reads HDF5 must use PROCESSES, not threads.
    #    h5py takes one process-wide lock for every call, held for the
    #    whole read, so a thread doing stack/MIP reads starves the GUI's
    #    own image loads -- measured 16.5 ms -> 2043 ms on the real
    #    store, and 16.8 ms (unaffected) with the same reads in separate
    #    processes. This is the contract that fix has to keep.
    ct_src = inspect.getsource(MainWindow._run_celltype_determination_body)
    check('celltype barcode mode runs in background PROCESSES',
          'ProcWorker' in ct_src and '_celltype_worker' in ct_src
          and 'ThreadPoolExecutor' not in ct_src, 'still thread-backed')
    check('celltype barcode mode fans out per FOV',
          '_classify_celltype_fov' in ct_src and 'max_workers' in ct_src)

    for name in ('_start_cell_overlay_save_worker', '_run_3d_localize',
                 '_view_3d_localize', '_apply_focus_detection'):
        src = inspect.getsource(getattr(MainWindow, name))
        check(f'{name} uses ProcWorker (h5py reads off this process)',
              'ProcWorker' in src and 'FnWorker' not in src, 'still FnWorker')

    # the child entry points must be module-level, or 'spawn' cannot pickle them
    import codelab_pipeline.localization.localization as _loc
    from canvas import pipeline_canvas as _pc
    from codelab_pipeline.segmentation import segment as _seg
    import pickle as _pickle
    for fn in (_loc.refine_spots_batch, _pc.render_cell_overlay_to_png, _seg.focus_profile):
        try:
            _pickle.loads(_pickle.dumps(fn))
            check(f'{fn.__name__} is picklable for spawn', True)
        except Exception as e:
            check(f'{fn.__name__} is picklable for spawn', False, str(e))

    # 7. cell view highlights the selected cell in the LEFT panel
    from canvas.spot_crop_displayer import SpotCropDisplayer
    import numpy as _np
    d = SpotCropDisplayer()
    # This section reads the figure's own artists, so the window has to be
    # VISIBLE: a hidden displayer defers its redraw (rasterizing a whole
    # FOV for nobody was 3.5 s of app startup) and its figure would hold
    # no axes to inspect. Showing it is also the condition being tested --
    # these are the colors a user sees.
    d.show()
    ys_g, xs_g = _np.mgrid[0:64, 0:64]
    masks = []
    for cid, (cx, cy) in enumerate([(15, 15), (45, 20), (30, 45)], start=44):
        m = (xs_g - cx) ** 2 + (ys_g - cy) ** 2 <= 8 ** 2
        yy, xx = _np.where(m)
        masks.append((cid, xx.astype(float), yy.astype(float)))
    img = _np.zeros((64, 64), dtype=_np.float32)

    def ctx_colors(highlight):
        d.set_data(img[10:30, 10:30], [], context_image=img, context_masks=masks,
                   context_title='t', context_highlight=highlight)
        ax = d._context_axes if d._context_axes is not None else d.canvas.figure.axes[0]
        return ({ln.get_color() for ln in ax.get_lines()},
                {t.get_text(): t.get_color() for t in ax.texts})

    lines, labels = ctx_colors(45)
    check('selected cell contour turns red', lines == {'yellow', 'red'}, str(lines))
    check('selected cell label turns red',
          labels.get('45') == 'red' and labels.get('44') == 'yellow'
          and labels.get('46') == 'yellow', str(labels))
    lines, labels = ctx_colors(None)
    check('no highlight -> everything stays yellow',
          lines == {'yellow'} and set(labels.values()) == {'yellow'}, f'{lines} {labels}')
    lines, labels = ctx_colors(999)
    check('highlight of an absent cell changes nothing',
          lines == {'yellow'} and set(labels.values()) == {'yellow'}, f'{lines} {labels}')

    # 8. O/P step the View list with wrap-around
    from PyQt5 import QtCore as _Qt
    mw = MainWindow()
    sp_panel = mw.ui.SpotLocalizationPanel
    mw._on_spot_cell_selected = lambda *a: None      # isolate the stepping

    class _Cell:
        def __init__(self, i):
            self.id = i
            self.fov = 1

    sp_panel.populate_cell_choices([_Cell(44), _Cell(45), _Cell(46)], {})
    lw = sp_panel.CellListWidget
    check('View list is FOV row + one row per cell', lw.count() == 4, str(lw.count()))

    lw.setCurrentRow(0)
    rows = []
    for _ in range(4):
        mw._step_spot_view(+1)
        rows.append(lw.currentRow())
    check('P walks forward and wraps last -> FOV row', rows == [1, 2, 3, 0], str(rows))

    lw.setCurrentRow(0)
    mw._step_spot_view(-1)
    check('O from the FOV row wraps to the LAST cell', lw.currentRow() == 3, str(lw.currentRow()))
    mw._step_spot_view(-1)
    check('O keeps walking backward', lw.currentRow() == 2, str(lw.currentRow()))

    sp_panel.populate_cell_choices([], {})
    lw.setCurrentRow(0)
    mw._step_spot_view(+1)
    p_row = lw.currentRow()
    mw._step_spot_view(-1)
    check('with no cells (FOV row only) both keys do nothing',
          lw.count() == 1 and p_row == 0 and lw.currentRow() == 0,
          f'count={lw.count()} afterP={p_row} afterO={lw.currentRow()}')

    keys = sorted(sc.key().toString() for sc in lw.findChildren(QtWidgets.QShortcut))
    win_keys = sorted(sc.key().toString()
                      for sc in mw.spot_crop_displayer.findChildren(QtWidgets.QShortcut))
    check('O/P bound on the View list and the crop window',
          keys == ['O', 'P'] and win_keys == ['O', 'P'], f'{keys} {win_keys}')
    check('shortcuts are scoped, never application-wide',
          all(sc.context() != _Qt.Qt.ApplicationShortcut
              for sc in lw.findChildren(QtWidgets.QShortcut)
              + mw.spot_crop_displayer.findChildren(QtWidgets.QShortcut)))

    # 9. the Barcode Overview and the Celltype Result must give the SAME
    #    celltype the SAME colour. They used to colour independently --
    #    the overview by its channel dict's insertion order (i.e. the
    #    order celltypes were assigned), the result by sorted name -- so
    #    with WT/4A3/8B1 assigned in that order WT drew red in one window
    #    and green in the other, side by side on screen.
    from canvas.barcode_overview_displayer import BarcodeOverviewDisplayer
    from canvas.celltype_result_displayer import CelltypeResultDisplayer
    from canvas import celltype_colors
    import numpy as _np

    chans = [('Hyb_130', 635, 'DNA'), ('Hyb_132', 635, 'DNA'), ('Hyb_135', 635, 'DNA')]
    ct_names = ['WT', '4A3', '8B1']          # deliberately NOT alphabetical
    ct_by_chan = dict(zip(chans, ct_names))
    ov = BarcodeOverviewDisplayer()
    ov.set_data({c: _np.zeros((16, 16), dtype=_np.uint16) for c in chans},
                {c: f'{n}: {c[0]}' for c, n in zip(chans, ct_names)},
                celltype_by_channel=ct_by_chan)
    overview_colors = {ct_by_chan[c]: ov._color_for(i, c) for i, c in enumerate(chans)}
    result_colors = celltype_colors.colors_for_names(ct_names)
    check('overview and result agree on every celltype colour',
          all(overview_colors[n] == result_colors[n] for n in ct_names),
          str({n: (celltype_colors.hex_of(overview_colors[n]),
                   celltype_colors.hex_of(result_colors[n])) for n in ct_names}))
    check('colour depends on the NAME, not on assignment order',
          celltype_colors.colors_for_names(['WT', '4A3', '8B1'])
          == celltype_colors.colors_for_names(['8B1', 'WT', '4A3']))
    check('each celltype gets a distinct colour',
          len({tuple(v) for v in result_colors.values()}) == len(ct_names))

    # pick_channel_by_type: generalized beyond the two role labels
    # (reported: alignment offered only fiducial/readout, so a hybe's
    # SECOND readout channel was unreachable)
    from codelab_pipeline.alignment import chain
    r3 = {'folder': 'Hyb_001', 'fiducial_channel': 555,
          'channels': [555, 647, 488]}
    check('role labels resolve as before',
          chain.pick_channel_by_type(r3, 'fiducial') == 555
          and chain.pick_channel_by_type(r3, 'readout') == 647)
    check('a concrete channel resolves to itself (the 2nd readout '
          'is now reachable)',
          chain.pick_channel_by_type(r3, '488') == 488
          and chain.pick_channel_by_type(r3, 488) == 488)
    check('a hybe lacking the requested channel falls back to the '
          'readout rule',
          chain.pick_channel_by_type(r3, '405') == 647)

    print()
    print(f'{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        raise SystemExit('FAILURES: ' + ', '.join(FAIL))
    print('ALL GOOD')


if __name__ == '__main__':
    main()
