"""
WHICH FOV does "Save" write, and can a FOV be saved EMPTY?

Both halves were real, reported bugs, and the second survived a first
fix -- hence a suite of its own.

1. Removing every cell in a FOV is a legitimate result to save. The old
   staged test was "has cells", so an emptied FOV dropped out of the
   staged list and Save silently RETARGETED to whichever other FOV still
   had cells. CellContainer pre-creates an empty dict for every FOV, so
   cell count alone cannot distinguish "emptied on purpose" from "never
   touched" -- _commit_cell_edit records deliberate edits instead.

2. The FOV Save targets is the one ON SCREEN. It used to come from
   _last_segment_context, which is only updated when a FOV successfully
   DISPLAYS cells -- _try_show_existing_cells returns early, before
   setting it, for a FOV with none. So switching to a FOV whose cells you
   just deleted left that context naming an EARLIER FOV, and Save
   overwrote that one: reported with the panel on FOV004 and the
   confirmation dialog naming FOV002.

   The spinbox cannot go stale that way -- but a FOV the user has staged
   nothing for must still never be written, or moving the spinbox would
   wipe an untouched FOV.

Run: python tests/test_save_cells_target.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np                                   # noqa: E402
from unittest import mock                            # noqa: E402
from PyQt5 import QtWidgets                          # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
for _m in ('critical', 'warning', 'information', 'question'):
    mock.patch.object(QtWidgets.QMessageBox, _m,
                      return_value=QtWidgets.QMessageBox.Yes).start()

import windows.main_window as MW                     # noqa: E402
from windows.main_window import MainWindow           # noqa: E402
from codelab_pipeline.models.cell import ACell       # noqa: E402
from codelab_pipeline.models.cell_container import CellContainer  # noqa: E402

PASSED = []
FAILED = []


def check(label, cond, detail=''):
    (PASSED if cond else FAILED).append(label)
    print(('  ok   ' if cond else '  FAIL ') + label + ('' if cond else f'   {detail}'))


def cell(cid, fov):
    c = ACell()
    c.set_metadata(id=cid, fov=fov, reference_hybe='BF',
                   reference_modality='RNA')
    c.area = (np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    return c


def save_target(w):
    """(fov, n_cells) actually handed to the store by _save_cells."""
    written = []
    w._all_analysis_storage_paths = lambda: ['/store/RNA']
    w._recast_persisted_spots = lambda fov: None
    with mock.patch.object(MW.analysis_store, 'mirror_write_cells',
                           side_effect=lambda p, f, c: written.append(
                               (f, len(c.get_cells(f))))):
        w._save_cells()
    return written


def test_emptied_fov_on_screen_is_the_one_saved():
    """The reported case: panel on FOV004, dialog said FOV002."""
    w = MainWindow(None)
    cp = w.ui.CellSegmentPanel
    w.cell_container = CellContainer([2, 4])
    w.cell_container.data[2] = {1: cell(1, 2), 2: cell(2, 2)}
    w.cell_container.data[4] = {7: cell(7, 4), 8: cell(8, 4)}
    w.cell_container_permanent = CellContainer([2, 4])
    w.cell_container_permanent.data[2] = {1: cell(1, 2), 2: cell(2, 2)}
    w.cell_container_permanent.data[4] = {7: cell(7, 4), 8: cell(8, 4)}
    w._transient_staged_fovs = set()
    # stale: FOV4 had no cells to display, so this was never updated
    w._last_segment_context = {'fov': 2, 'reference_hybe': 'BF',
                               'modality': 'RNA'}
    cp.FovSpinBox.setValue(4)

    fp = w._begin_cell_edit(4)
    w.cell_container.remove(4, [7, 8])
    w._commit_cell_edit(4, fp)

    check('an emptied FOV still counts as staged',
          4 in w._staged_transient_fovs(), str(w._staged_transient_fovs()))
    written = save_target(w)
    check('Save writes the FOV ON SCREEN, not the stale context FOV',
          written and written[0][0] == 4, f'wrote {written}')
    check('and writes it EMPTY -- zero cells is a result',
          written and written[0][1] == 0, f'wrote {written}')
    check('the stale-context FOV is left alone',
          bool(w.cell_container_permanent.data.get(2)), 'FOV2 was wiped')
    check('the emptied FOV really loses its cells',
          not w.cell_container_permanent.data.get(4),
          str(w.cell_container_permanent.data.get(4)))


def test_untouched_fov_on_screen_is_never_wiped():
    """Moving the spinbox is not a licence to overwrite."""
    w = MainWindow(None)
    cp = w.ui.CellSegmentPanel
    w.cell_container = CellContainer([2, 9])
    w.cell_container.data[2] = {1: cell(1, 2)}
    w.cell_container_permanent = CellContainer([2, 9])
    w.cell_container_permanent.data[9] = {5: cell(5, 9)}
    w._transient_staged_fovs = set()
    w._last_segment_context = {'fov': 2, 'reference_hybe': 'BF',
                               'modality': 'RNA'}
    cp.FovSpinBox.setValue(9)          # nothing was ever staged for FOV9

    written = save_target(w)
    check('an untouched FOV on screen is not written',
          written and written[0][0] == 2, f'wrote {written}')
    check('its saved cells survive',
          bool(w.cell_container_permanent.data.get(9)), 'FOV9 was wiped')


def test_discard_does_not_make_a_fov_saveable():
    """Discard means "throw away my edits", not "delete what is saved"."""
    w = MainWindow(None)
    cp = w.ui.CellSegmentPanel
    w.cell_container = CellContainer([3])
    w.cell_container.data[3] = {1: cell(1, 3)}
    w.cell_container_permanent = CellContainer([3])
    w.cell_container_permanent.data[3] = {1: cell(1, 3)}
    w._transient_staged_fovs = set()
    w._last_segment_context = {'fov': 3, 'reference_hybe': 'BF',
                               'modality': 'RNA'}
    cp.FovSpinBox.setValue(3)
    w.cell_displayer.reference_image = None
    w._discard_cells()
    check('a discarded FOV is not marked staged',
          3 not in w._transient_staged_fovs, str(w._transient_staged_fovs))


def main():
    for fn in (test_emptied_fov_on_screen_is_the_one_saved,
               test_untouched_fov_on_screen_is_never_wiped,
               test_discard_does_not_make_a_fov_saveable):
        print(f'\n{fn.__name__}')
        fn()
    print(f'\n{len(PASSED)} passed, {len(FAILED)} failed')
    if FAILED:
        for f in FAILED:
            print('  FAILED:', f)
        return 1
    print('ALL GOOD')
    return 0


if __name__ == '__main__':
    sys.exit(main())
