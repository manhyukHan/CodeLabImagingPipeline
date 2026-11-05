"""The three ways a readout gets its channel, end to end.

For each, the whole tuple the trace path needs:
(fiducial_channel, hybe, channel, rindex, pindex).

  (a) the ExperimentLayout names the channels   -- readoutChannels
  (b) it does not, and the panel activates ONE
  (c) it does not, and the operator activates TWO

(c) is REFUSED, and the refusal is the point: the layout gives one
readout id per round, so two channels would both claim it. The second
channel's readout id is a design fact -- adapter 1 read by 635 is R1 and
by 747 is R51 -- and neither continuing the numbering nor calling the
second a repeat can recover it. Both alternatives are exercised below so
the reason is on the record.

Run: python tests/test_readout_assignment_cases.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline.analysis import polymer as P            # noqa: E402
from codelab_pipeline.analysis.readout_map import (           # noqa: E402
    ANY_CHANNEL, ReadoutBin, ReadoutMap)
from codelab_pipeline.localization import tracing_v2 as T2    # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(('  ok ' if cond else 'FAIL ') + name
          + (f'   [{detail}]' if detail and not cond else ''))


def rec(folder, rid, readouts=None, channels=(555, 635, 747), fid=555):
    return {'folder': folder, 'readout_id': rid, 'datatype': 'H',
            'channels': list(channels), 'fiducial_channel': fid,
            'readouts': readouts if readouts is not None else [(None, rid)]}


def full(m, recs):
    """[(fiducial_channel, hybe, channel, rindex, pindex)] in axis order."""
    fid_of = {r['folder']: r['fiducial_channel'] for r in recs}
    return [(fid_of.get(b.hybe), b.hybe, b.channel, b.rindex, b.pindex)
            for b in m.bins()]


def case_a():
    print('(a) the layout names the channels')
    recs = [rec('Hyb_001', 1, [(635, 1), (747, 51)]),
            rec('Hyb_002', 2, [(635, 2), (747, 52)])]
    m = ReadoutMap.from_layout(recs, channels=[635, 747])
    rows = full(m, recs)
    check('every bin is complete and the two dyes interleave by readout id',
          rows == [(555, 'Hyb_001', 635, 1, 0),
                   (555, 'Hyb_002', 635, 2, 1),
                   (555, 'Hyb_001', 747, 51, 50),
                   (555, 'Hyb_002', 747, 52, 51)],
          str(rows))
    check('the axis is as long as the design, gaps and all',
          m.n_slots() == 52 and len(m.gaps()) == 48,
          '%d slots, %d gaps' % (m.n_slots(), len(m.gaps())))
    check('the fiducial channel is NOT a readout bin',
          all(r[2] != r[0] for r in rows))

    print('  and the trace path keys it per bin')
    bins = T2.readout_bins_of({'Hyb_001': [635, 747], 'Hyb_002': [635, 747]})
    check('two channels per round', bins == {'Hyb_001': [635, 747],
                                             'Hyb_002': [635, 747]})
    check('so the key carries the channel',
          T2.bin_key('Hyb_001', 747, 2) == ('Hyb_001', 747))
    # modality='DNA' is what makes the readout number the axis; without
    # it the bins pack densely, which is the right reading for an RNA
    # panel whose numbers are gene ids (see polymer.POLYMER_MODALITIES)
    keys = P.bin_keys(recs, channels=[635, 747], modality='DNA')
    check('and the axis reads it under exactly that key',
          keys[50] == ('Hyb_001', 747) and keys[0] == ('Hyb_001', 635),
          str((keys[0], keys[50])))
    check('an unstated modality packs densely instead',
          len(P.bin_keys(recs, channels=[635, 747])) == 4,
          str(len(P.bin_keys(recs, channels=[635, 747]))))

    print('  activating ONE of the two narrows it, it does not merge them')
    m1 = ReadoutMap.from_layout(recs, channels=[747])
    check('only the 747 bins', full(m1, recs)
          == [(555, 'Hyb_001', 747, 51, 0), (555, 'Hyb_002', 747, 52, 1)],
          str(full(m1, recs)))


def case_b():
    print('\n(b) the layout names none, the panel activates ONE')
    recs = [rec('Hyb_001', 1), rec('Hyb_002', 2)]
    m = ReadoutMap.from_layout(recs, channels=[635])
    rows = full(m, recs)
    check('the activated channel is the bin; the other is simply not one',
          rows == [(555, 'Hyb_001', 635, 1, 0),
                   (555, 'Hyb_002', 635, 2, 1)], str(rows))
    check('747 is not in the map at all -- nothing had to choose',
          m.of('Hyb_001', 747) is None and m.of('Hyb_001', 635) is not None)
    check('activating 747 instead gives the same readout ids',
          [b.rindex for b in
           ReadoutMap.from_layout(recs, channels=[747]).bins()] == [1, 2],
          'the layout names one readout per round, and it is the round\'s')

    print('  the single channel is handed DOWN, not left to be guessed')
    check('the trace path takes one channel for the round',
          T2.readout_bins_of({'Hyb_001': 635}) == {'Hyb_001': [635]})
    check('and the key stays bare, as every store so far holds it',
          T2.bin_key('Hyb_001', 635, 1) == 'Hyb_001')
    check('a split puts the channel back for a backend that needs one',
          P.split_key('Hyb_001', channel=635) == ('Hyb_001', 635)
          and P.split_key(('Hyb_001', 747)) == ('Hyb_001', 747))
    check('while a frame asks only for the round',
          P.key_hybe(('Hyb_001', 747)) == 'Hyb_001'
          and P.key_hybe('Hyb_001') == 'Hyb_001')

    print('  the DEFAULT, when the layout names nothing: the lower channel')
    from codelab_pipeline.alignment import chain as A
    check('auto resolves to one channel, and ties take the lower number',
          A.dominant_readout_channel(recs) == 635,
          str(A.dominant_readout_channel(recs)))
    check('even where the numbers are not the same width',
          A.dominant_readout_channel(
              [rec('H1', 1, channels=(555, 999, 1024))]) == 999,
          'a string sort would have taken 1024')
    check('747 never reaches the trace path, so no crop is read for it',
          T2.readout_bins_of({'Hyb_001': [635]}) == {'Hyb_001': [635]}
          and 747 not in T2.readout_bins_of({'Hyb_001': [635]})['Hyb_001'])
    check('and it is not on the axis either',
          747 not in [b.channel for b in
                      ReadoutMap.from_layout(recs, channels=[635]).bins()])

    print('  with no activation at all the channel is unknown, not guessed')
    m0 = ReadoutMap.from_layout(recs)
    check('the bin carries ANY_CHANNEL',
          [b.channel for b in m0.bins()] == [ANY_CHANNEL] * 2)
    check('and a lookup for 635 misses rather than answering',
          m0.of('Hyb_001', 635) is None)


def case_c():
    print('\n(c) the layout names none, the operator activates TWO')
    recs = [rec('Hyb_001', 1), rec('Hyb_002', 2)]
    try:
        ReadoutMap.from_layout(recs, channels=[635, 747])
        check('it is refused, not resolved', False, 'accepted')
    except ValueError as e:
        check('it is refused, not resolved',
              'names no channel' in str(e) and 'readoutChannels' in str(e),
              str(e))
        check('and the message says both ways out',
              'readoutChannels column' in str(e)
              and 'activate one channel' in str(e), str(e))

    print('  because neither way of inventing the second id is sound')
    # 1. the same readout id for both: two bins, one polymer slot
    try:
        ReadoutMap([ReadoutBin('Hyb_001', 635, 1, 'H', 0),
                    ReadoutBin('Hyb_001', 747, 1, 'H', 0)])
        check('the same readout id for both is refused', False, 'accepted')
    except ValueError as e:
        check('the same readout id for both is refused',
              'claimed by both' in str(e), str(e))
    # 2. the second as a REPEAT of the first: it builds, and that is the
    #    danger -- a repeat means the SAME locus imaged twice, and the
    #    repeat distance is a quality number we read. A different barcode
    #    filed there corrupts a measurement instead of failing.
    m = ReadoutMap([ReadoutBin('Hyb_001', 635, 1, 'H', 0),
                    ReadoutBin('Hyb_001', 747, 1, 'R', -1)])
    check('calling the second a repeat BUILDS -- which is why it is not done',
          [b.channel for b in m.bins()] == [635]
          and [b.channel for b in m.extras()] == [747],
          str((m.bins(), m.extras())))
    check('and it would cost the 747 locus its place on the polymer',
          m.n_slots() == 1, '%d slots for two barcodes' % m.n_slots())
    # 3. continuing the numbering: not implemented, and must not be. R51
    #    is a design fact; 1+n is a guess that mislabels every locus.
    check('nothing continues the numbering on its own',
          not any('consecutive' in (d or '').lower()
                  for d in (ReadoutMap.from_layout.__doc__,
                            P.readout_map_for.__doc__)),
          'no auto-continuation anywhere in the contract')

    print('  naming them in the layout is the way through')
    named = [rec('Hyb_001', 1, [(635, 1), (747, 51)]),
             rec('Hyb_002', 2, [(635, 2), (747, 52)])]
    m2 = ReadoutMap.from_layout(named, channels=[635, 747])
    check('the same two channels now resolve', len(m2.bins()) == 4)
    check('each to its own locus',
          m2.of('Hyb_001', 635).rindex == 1
          and m2.of('Hyb_001', 747).rindex == 51)


def main():
    case_a()
    case_b()
    case_c()
    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
