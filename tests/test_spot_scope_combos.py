"""
The Spot panel's three scope combos (FOV / Hybe / Channel) must each
actually change the scope.

The hybe combo did not. Every selection change re-derives the choices for
the current FOV, which clears and repopulates the combo, and the restore
used QComboBox.findData with a (hybe, modality) tuple payload. PyQt5
matches a non-QVariant Python payload by OBJECT IDENTITY, so the
freshly-built tuples never matched the captured one, findData returned
-1, no setCurrentIndex ran, and the combo fell back to index 0. Picking
any hybe snapped back to the first one.

FOV (int payload restored via list.index) and Channel (restored via
setCurrentText) were never affected -- which is why only the hybe combo
looked broken.

Run: python tests/test_spot_scope_combos.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt5 import QtWidgets                                   # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

from canvas.cell_spot_status_displayer import (                # noqa: E402
    CellSpotStatusDisplayer, _index_of_data)

PASS, FAIL = [], []
PAIRS = [('Hyb_002', 'DNA'), ('Hyb_016', 'DNA'), ('Hyb_020', 'DNA')]


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(('  ok ' if cond else 'FAIL ') + name + (f' -- {detail}' if detail and not cond else ''))


def main():
    # -- the primitive, in isolation --
    combo = QtWidgets.QComboBox()
    for h, m in PAIRS:
        combo.addItem(h, (h, m))
    check('_index_of_data matches an EQUAL (not identical) tuple',
          _index_of_data(combo, ('Hyb_016', 'DNA')) == 1,
          str(_index_of_data(combo, ('Hyb_016', 'DNA'))))
    check('_index_of_data reports -1 for an absent value',
          _index_of_data(combo, ('Hyb_999', 'DNA')) == -1)
    # the exact PyQt5 behaviour this exists to route around; if a future
    # PyQt5 makes findData value-based this still passes, it just stops
    # being the reason
    check('findData is the thing that could not do it (documented, not asserted)',
          True)

    d = CellSpotStatusDisplayer()

    # -- the real regression: select, then let the scope refresh rerun --
    d.set_spot_hybe_choices(PAIRS)
    d.SpotHybeComboBox.setCurrentIndex(2)
    check('a hybe can be selected at all', d.current_spot_hybe() == ('Hyb_020', 'DNA'),
          str(d.current_spot_hybe()))

    # This is what _on_cell_spot_status_spot_scope_changed does on EVERY
    # combo change: rebuild the choices from freshly-sorted tuples.
    d.set_spot_hybe_choices(sorted({(h, m) for h, m in PAIRS}))
    check('the selection SURVIVES a choices rebuild (was: snapped to index 0)',
          d.current_spot_hybe() == ('Hyb_020', 'DNA'),
          f'fell back to {d.current_spot_hybe()}')

    # every hybe must be reachable, not just the first
    reached = []
    for i in range(d.SpotHybeComboBox.count()):
        d.SpotHybeComboBox.setCurrentIndex(i)
        d.set_spot_hybe_choices(list(PAIRS))          # the refresh again
        reached.append(d.current_spot_hybe())
    check('EVERY hybe is reachable across a refresh', reached == sorted(PAIRS),
          str(reached))

    # a hybe that disappears (FOV genuinely changed) must not be forced
    d.SpotHybeComboBox.setCurrentIndex(1)
    d.set_spot_hybe_choices([('Hyb_777', 'RNA')])
    check('a vanished hybe falls back rather than erroring',
          d.current_spot_hybe() == ('Hyb_777', 'RNA'), str(d.current_spot_hybe()))

    # -- the two combos that always worked must keep working --
    d.set_spot_fov_choices([1, 2, 3])
    d.SpotFovComboBox.setCurrentIndex(2)
    d.set_spot_fov_choices([1, 2, 3])
    check('FOV selection survives a rebuild', d.current_spot_fov() == 3,
          str(d.current_spot_fov()))

    d.set_spot_channel_choices([405, 555, 635])
    d.SpotChannelComboBox.setCurrentText('635')
    d.set_spot_channel_choices([405, 555, 635])
    check('Channel selection survives a rebuild', d.current_spot_channel() == 635,
          str(d.current_spot_channel()))

    # -- the rebuild must stay silent, or the handler would re-enter --
    fired = []
    d.spot_scope_changed.connect(lambda: fired.append(1))
    d.set_spot_hybe_choices(list(PAIRS))
    d.set_spot_fov_choices([1, 2, 3])
    d.set_spot_channel_choices([405, 555, 635])
    check('repopulating choices emits no scope change', not fired, f'{len(fired)} emitted')

    print()
    print(f'{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        raise SystemExit('FAILURES: ' + ', '.join(FAIL))
    print('ALL GOOD')


if __name__ == '__main__':
    main()
