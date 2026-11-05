"""The readout map, settled by hand in the pipeline's tracing step.

from_layout REFUSES the case this exists for -- the layout names one
readout id per round and two channels are activated, so both would claim
it -- and a refusal leaves nothing to edit. seed_rows builds the ROWS
instead, with the unnamed channel's readout number blank, and the editor
will not close until it is filled.

Run: QT_QPA_PLATFORM=offscreen python tests/test_readout_map_editor.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt5 import QtWidgets                                   # noqa: E402

from codelab_pipeline.analysis.readout_map import (           # noqa: E402
    NOT_A_BIN, ReadoutMap)
from ui.readout_map_editor import ReadoutMapEditor            # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(('  ok ' if cond else 'FAIL ') + name
          + (f'   [{detail}]' if detail and not cond else ''))


def rec(folder, rid, dtype='H', readouts=None, channels=(555, 635, 747)):
    return {'folder': folder, 'readout_id': rid, 'datatype': dtype,
            'channels': list(channels), 'fiducial_channel': 555,
            'readouts': readouts if readouts is not None else [(None, rid)]}


def seeding():
    print('the layout seeds the rows, and says where it falls silent')
    recs = [rec('Hyb_001', 1), rec('Hyb_002', 2)]
    rows = ReadoutMap.seed_rows(recs, channels=[635, 747])
    check('one row per (hybe, channel)', len(rows) == 4, str(len(rows)))
    got = {(r['hybe'], r['channel']): r['rindex'] for r in rows}
    check('the LOWEST activated channel inherits the round\'s readout id',
          got[('Hyb_001', 635)] == 1 and got[('Hyb_002', 635)] == 2, str(got))
    check('and the other is blank -- the layout does not say',
          got[('Hyb_001', 747)] is None and got[('Hyb_002', 747)] is None,
          str(got))

    print('  a layout that DOES name them leaves nothing blank')
    named = [rec('Hyb_001', 1, readouts=[(635, 1), (747, 51)])]
    rows2 = ReadoutMap.seed_rows(named, channels=[635, 747])
    check('both carry their own readout id',
          sorted(r['rindex'] for r in rows2) == [1, 51],
          str([(r['channel'], r['rindex']) for r in rows2]))

    print('  a hybe that does not carry the channel gets no row for it')
    rows3 = ReadoutMap.seed_rows([rec('Hyb_001', 1, channels=(555, 635))],
                                 channels=[635, 747])
    check('only the channel it has',
          [r['channel'] for r in rows3] == [635], str(rows3))


def placing():
    print('\npolymer index is derived, never typed')
    rows = [{'hybe': 'H1', 'channel': 635, 'rindex': 1, 'dtype': 'H'},
            {'hybe': 'H2', 'channel': 635, 'rindex': 3, 'dtype': 'H'},
            {'hybe': 'R1', 'channel': 635, 'rindex': 1, 'dtype': 'R'}]
    out, n, bad = ReadoutMap.place_rows(rows)
    check('an H row sits at rindex - lo',
          [r['pindex'] for r in out] == [0, 2, NOT_A_BIN],
          str([r['pindex'] for r in out]))
    check('the axis spans the acquired ids and the gap shows',
          n == 3 and not bad, '%d slots, %s' % (n, bad))
    check('a repeat round is measured but is not a bin',
          out[2]['pindex'] == NOT_A_BIN)

    print('  the DECLARED range decides both ends')
    out2, n2, bad2 = ReadoutMap.place_rows(rows, '1-8')
    check('bin 0 is readout 1 and the axis runs to 8',
          [r['pindex'] for r in out2][:2] == [0, 2] and n2 == 8 and not bad2,
          '%d slots' % n2)
    out3, n3, bad3 = ReadoutMap.place_rows(rows, '2-8')
    check('a readout outside the range is reported, not placed',
          any('outside the declared range' in c for c in bad3), str(bad3))

    print('  and two bins cannot take one slot')
    clash = [{'hybe': 'H1', 'channel': 635, 'rindex': 1, 'dtype': 'H'},
             {'hybe': 'H1', 'channel': 747, 'rindex': 1, 'dtype': 'H'}]
    _o, _n, bad4 = ReadoutMap.place_rows(clash)
    check('both are named', any('both sit at polymer index' in c
                                for c in bad4), str(bad4))


def editor():
    print('\nthe editor will not close on an unsettled map')
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    recs = [rec('Hyb_001', 1), rec('Hyb_002', 2)]
    rows = ReadoutMap.seed_rows(recs, channels=[635, 747])
    ed = ReadoutMapEditor(rows, '')
    ok = ed.Buttons.button(QtWidgets.QDialogButtonBox.Ok)
    check('OK is disabled while a readout number is blank',
          not ok.isEnabled())
    check('and the blanks are named, not merely counted',
          'no readout number' in ed.Complaints.text(), ed.Complaints.text())
    try:
        ed.value()
        check('value() refuses too', False, 'returned')
    except ValueError as e:
        check('value() refuses too', 'no readout number' in str(e), str(e))

    print('  filling them in settles it')
    for i in range(ed.Table.rowCount()):
        if ed.Table.item(i, 1).text() == '747':
            ed.Table.item(i, 2).setText(
                str(50 + int(ed.Table.item(i, 0).text()[-1])))
    check('OK comes back', ok.isEnabled(), ed.Complaints.text())
    text = ed.value()
    m = ReadoutMap.from_text(text)
    check('the settled map has four bins at their own loci',
          len(m.bins()) == 4
          and m.of('Hyb_001', 747).rindex == 51
          and m.of('Hyb_002', 747).rindex == 52,
          str(m.bins()))
    check('the axis spans 1..52 with the unimaged stretch empty',
          m.n_slots() == 52 and len(m.gaps()) == 48,
          '%d slots, %d gaps' % (m.n_slots(), len(m.gaps())))

    print('  a map that is just the layout stores EMPTY, so it keeps '
          'following it')
    plain = ReadoutMapEditor(ReadoutMap.seed_rows(recs, channels=[635]), '')
    check('nothing to settle, nothing stored', plain.value() == '',
          plain.value())

    print('  a declared range travels with it')
    ed2 = ReadoutMapEditor(ReadoutMap.seed_rows(recs, channels=[635]), '1-8')
    t2 = ed2.value()
    check('the axis is 8 long though only 2 bins were imaged',
          t2.startswith('slots=8,') and ReadoutMap.from_text(t2).n_slots() == 8,
          t2)
    check('and re-opening it reads back as itself',
          ReadoutMap.from_text(t2) == ReadoutMap.from_text(
              ReadoutMapEditor(ReadoutMap.from_text(t2).rows(),
                               '1-8').value()))


def main():
    seeding()
    placing()
    editor()
    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
