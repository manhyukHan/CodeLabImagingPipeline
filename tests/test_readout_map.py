"""The (hybe, channel) -> (rindex, dtype, pindex) table.

Run: python tests/test_readout_map.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline.analysis.readout_map import (          # noqa: E402
    ReadoutMap, ReadoutBin, NOT_A_BIN, ANY_CHANNEL)

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(('  ok ' if cond else 'FAIL ') + name
          + (f'   [{detail}]' if detail and not cond else ''))


def rec(folder, rid, dtype='H', channels=(555, 635, 747), fid=555,
        readouts=None):
    return {'folder': folder, 'readout_id': rid, 'datatype': dtype,
            'channels': list(channels), 'fiducial_channel': fid,
            'readouts': readouts if readouts is not None else [(None, rid)]}


def main():
    print('a single-channel layout behaves as it always has')
    recs = [rec('Hyb_003', 3), rec('Hyb_001', 1), rec('Hyb_002', 2),
            rec('Rep_001', 1, 'R'), rec('Toe_050', 50, 'T')]
    m = ReadoutMap.from_layout(recs, channels=[635])
    check('polymer bins are readout-id ordered',
          [b.hybe for b in m.bins()] == ['Hyb_001', 'Hyb_002', 'Hyb_003'],
          str([b.hybe for b in m.bins()]))
    check('pindex is 0-based along that order',
          [b.pindex for b in m.bins()] == [0, 1, 2])
    check('R and T are measured but are not bins',
          [b.hybe for b in m.extras()] == ['Rep_001', 'Toe_050'],
          str([(b.hybe, b.pindex) for b in m.extras()]))
    check('their pindex says so', all(b.pindex == NOT_A_BIN
                                      for b in m.extras()))
    check('lookup works', m.of('Hyb_002', 635).rindex == 2)
    check('an unknown pair is None, not a guess',
          m.of('Hyb_002', 747) is None and m.of('nope', 635) is None)

    print('\ntwo readouts in one hybe')
    recs = [rec('Hyb_001', 1, readouts=[(635, 1), (747, 51)]),
            rec('Hyb_002', 2, readouts=[(635, 2), (747, 52)])]
    m = ReadoutMap.from_layout(recs, channels=[635, 747])
    check('every (hybe, channel) is its own bin', len(m.bins()) == 4,
          str(len(m.bins())))
    check('ordered by readout id, so the two dyes interleave correctly',
          [(b.hybe, b.channel) for b in m.bins()]
          == [('Hyb_001', 635), ('Hyb_002', 635),
              ('Hyb_001', 747), ('Hyb_002', 747)],
          str([(b.hybe, b.channel) for b in m.bins()]))
    check('pindex follows the readout id, so the gap at 3..50 is real',
          [b.pindex for b in m.bins()] == [0, 1, 50, 51],
          str([b.pindex for b in m.bins()]))
    check('the same hybe maps to different readouts per channel',
          m.of('Hyb_001', 635).rindex == 1 and m.of('Hyb_001', 747).rindex == 51)

    print('\nthe ACTIVATION filters, even where the layout names channels')
    recs = [rec('Hyb_001', 1, readouts=[(635, 1), (747, 51)]),
            rec('Hyb_002', 2, readouts=[(635, 2), (747, 52)])]
    m = ReadoutMap.from_layout(recs, channels=[635])
    check('a layout-named channel nobody activated is NOT a bin',
          [(b.hybe, b.channel) for b in m.bins()]
          == [('Hyb_001', 635), ('Hyb_002', 635)],
          str([(b.hybe, b.channel) for b in m.bins()]))
    check('and the axis is only as long as what is activated',
          m.n_slots() == 2 and m.gaps() == [],
          '%d slots, gaps %s' % (m.n_slots(), m.gaps()))
    check('activating both brings both back',
          len(ReadoutMap.from_layout(recs, channels=[635, 747]).bins()) == 4)
    check('channels=None takes whatever the layout names',
          len(ReadoutMap.from_layout(recs).bins()) == 4)
    check('channels=[] activates nothing, so nothing is a bin',
          ReadoutMap.from_layout(recs, channels=[]).bins() == [])

    print('\nan unnamed readout with no activation is channel-agnostic')
    recs = [rec('Hyb_001', 1), rec('Hyb_002', 2)]
    m = ReadoutMap.from_layout(recs)
    check('the channel is ANY_CHANNEL, not a guessed wavelength',
          [b.channel for b in m.bins()] == [ANY_CHANNEL] * 2,
          str([b.channel for b in m.bins()]))
    check('so a lookup for 635 misses rather than answering',
          m.of('Hyb_001', 635) is None
          and m.of('Hyb_001', ANY_CHANNEL) is not None)

    print('\na readout the design skips leaves a NaN stripe')
    recs = ([rec('Hyb_%03d' % i, i) for i in range(1, 52)]
            + [rec('Hyb_%03d' % i, i) for i in range(54, 81)])
    m = ReadoutMap.from_layout(recs, channels=[635])
    check('78 bins were imaged', len(m.bins()) == 78, str(len(m.bins())))
    check('but the axis is 80 long', m.n_slots() == 80, str(m.n_slots()))
    check('and slots 51, 52 are the holes', m.gaps() == [51, 52], str(m.gaps()))
    ax = m.axis()
    check('the axis carries None exactly there',
          ax[50] is not None and ax[51] is None and ax[52] is None
          and ax[53] is not None)
    check('readout 54 sits at slot 53, not 51',
          m.of('Hyb_054', 635).pindex == 53, str(m.of('Hyb_054', 635)))
    dense = ReadoutMap.from_layout(recs, channels=[635], pindex_from='dense')
    check('dense packing closes the gap, and reports none',
          dense.n_slots() == 78 and dense.gaps() == [],
          '%d slots, gaps %s' % (dense.n_slots(), dense.gaps()))
    check('a contiguous design is unaffected by the policy',
          ReadoutMap.from_layout([rec('H%d' % i, i) for i in range(1, 9)],
                                 channels=[635]).gaps() == [])

    print('\nit refuses what it cannot resolve')
    try:
        ReadoutMap.from_layout([rec('Hyb_001', 1)], channels=[635, 747])
        check('two activated channels with an unnamed readout', False,
              'accepted')
    except ValueError as e:
        check('two activated channels with an unnamed readout',
              'readoutChannels' in str(e), str(e))
    try:
        ReadoutMap([ReadoutBin('H1', 635, 1, 'H', 0),
                    ReadoutBin('H1', 635, 2, 'H', 1)])
        check('a duplicated (hybe, channel)', False, 'accepted')
    except ValueError as e:
        check('a duplicated (hybe, channel)', 'twice' in str(e), str(e))
    try:
        ReadoutMap([ReadoutBin('H1', 635, 1, 'H', 0),
                    ReadoutBin('H2', 635, 2, 'H', 0)])
        check('two bins claiming one polymer index', False, 'accepted')
    except ValueError as e:
        check('two bins claiming one polymer index', 'claimed by both' in str(e),
              str(e))

    print('\na hybe without the channel is skipped, not invented')
    recs = [rec('Hyb_001', 1, channels=(555, 635)),
            rec('Hyb_002', 2, channels=(555, 747))]
    m = ReadoutMap.from_layout(recs, channels=[635])
    check('only the hybe that carries it', [b.hybe for b in m.bins()]
          == ['Hyb_001'], str([b.hybe for b in m.bins()]))

    print('\nround trip through one config attribute')
    recs = [rec('Hyb_001', 1, readouts=[(635, 1), (747, 51)]),
            rec('Rep_001', 1, 'R', readouts=[(635, 1)])]
    m = ReadoutMap.from_layout(recs, channels=[635, 747])
    back = ReadoutMap.from_text(m.to_text())
    check('text survives', back == m, m.to_text())
    check('rows survive', ReadoutMap.from_rows(m.rows()) == m)
    check('the text is one flat attribute',
          '\n' not in m.to_text() and m.to_text().count('|') == 4 * len(m),
          m.to_text())

    print('\nthe real MP58 layout')
    real = (r'Q:\Seonghyeok\ORCA3\20251130_MP58_RNAandDNA_\TiftoDaxOutput'
            r'\251208_YSH_MP58_RNADNA-ExperimentLayout.xlsx')
    if os.path.exists(real):
        from codelab_pipeline.io import preprocess as PRE
        m = ReadoutMap.from_layout(PRE.parse_experiment_layout(real),
                                   channels=[635])
        b = m.bins()
        check('62 polymer bins, densely indexed',
              len(b) == 62 and [x.pindex for x in b] == list(range(62)),
              '%d bins' % len(b))
        check('every one is on 635', {x.channel for x in b} == {635})
        check('the QC rounds came along', len(m.extras()) > 0,
              str(len(m.extras())))
        # drop two H rounds from the middle of the real design: the axis
        # must keep its length and hole exactly where they were, which is
        # also the check that pindex comes from the Readouts column and
        # not from the folder number -- MP58's Hyb_052 carries readout 53.
        recs = PRE.parse_experiment_layout(real)
        held = [r for r in recs
                if r['folder'] not in ('Hyb_052', 'Hyb_053')]
        m2 = ReadoutMap.from_layout(held, channels=[635])
        was = [m.of('Hyb_052', 635).pindex, m.of('Hyb_053', 635).pindex]
        check('dropping two H rounds does not shorten the axis',
              m2.n_slots() == m.n_slots()
              and len(m2.bins()) == len(m.bins()) - 2,
              '%d slots, %d bins' % (m2.n_slots(), len(m2.bins())))
        check('two NaN stripes, exactly where they were',
              m2.gaps() == was, '%s vs %s' % (m2.gaps(), was))
        check('the bins on either side keep their place',
              m2.of('Hyb_051', 635).pindex == m.of('Hyb_051', 635).pindex
              and m2.of('Hyb_054', 635).pindex
              == m.of('Hyb_054', 635).pindex,
              '%s, %s' % (m2.of('Hyb_051', 635), m2.of('Hyb_054', 635)))
    else:
        print('  (skipped: layout not reachable)')

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
