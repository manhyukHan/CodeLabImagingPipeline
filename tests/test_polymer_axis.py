"""The polymer axis comes from the readout map, holes and all.

polymer.genomic_bins used to be "every H record, sorted by readout id",
which silently closed up a readout the design skipped: readouts 1-51 and
54-80 gave a 78-long polymer where bin 51 was readout 54, and every
distance-against-genomic-separation curve was bent by it. The axis is now
per SLOT, so a skipped readout is a column of NaN with its locus named.

Run: python tests/test_polymer_axis.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                          # noqa: E402

from codelab_pipeline.analysis import export as E           # noqa: E402
from codelab_pipeline.analysis import polymer as P          # noqa: E402
from codelab_pipeline.analysis.readout_map import (         # noqa: E402
    ANY_CHANNEL, ReadoutMap)

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(('  ok ' if cond else 'FAIL ') + name
          + (f'   [{detail}]' if detail and not cond else ''))


def rec(folder, rid, dtype='H', channels=(555, 635, 747), fid=555,
        readouts=None):
    return {'folder': folder, 'readout_id': rid, 'datatype': dtype,
            'channels': list(channels), 'fiducial_channel': fid,
            'readouts': readouts if readouts is not None else [(None, rid)],
            'readout_name': f'R{rid}'}


def _raises(fn, needle):
    try:
        fn()
    except (ValueError, KeyError) as e:
        return needle in str(e)
    return False


def allele(**kw):
    d = {'id': 7, 'cell': 3, 'fov': 3,
         'coordinate': (1.0, 2.0, 3.0), 'raw_coordinate': (1.0, 2.0, 3.0),
         'polymer_adj': {}, 'polymer_raw': {}, 'fiducial_trace_adj': {},
         'fiducial_trace_raw': {}, 'rejected_hybes': {}, 'provenance': {}}
    d.update(kw)
    return d


def contiguous():
    print('a contiguous design is exactly what it always was')
    recs = [rec('Hyb_003', 3), rec('Hyb_001', 1), rec('Hyb_002', 2),
            rec('Rep_001', 1, 'R')]
    check('bins are readout-id ordered, R is not a bin',
          P.genomic_bins(recs, modality='DNA')
          == [(1, 'Hyb_001'), (2, 'Hyb_002'), (3, 'Hyb_003')],
          str(P.genomic_bins(recs, modality='DNA')))
    check('bin_hybes is the bare list',
          P.bin_hybes(recs, modality='DNA')
          == ['Hyb_001', 'Hyb_002', 'Hyb_003'])
    check('bin_keys is channel-agnostic without an activation',
          P.bin_keys(recs, modality='DNA')
          == [('Hyb_001', ANY_CHANNEL), ('Hyb_002', ANY_CHANNEL),
              ('Hyb_003', ANY_CHANNEL)],
          str(P.bin_keys(recs, modality='DNA')))
    check('an unstated modality packs densely, as it always did',
          P.genomic_bins(recs) == P.genomic_bins(recs, modality='DNA'),
          'contiguous, so the two agree')


def holes():
    print('\na skipped readout is a SLOT, not a closed-up gap')
    recs = ([rec('Hyb_%03d' % i, i) for i in (1, 2, 3)]
            + [rec('Hyb_006', 6), rec('Hyb_007', 7)])
    bins = P.genomic_bins(recs, modality='DNA')
    check('the axis is as long as the readout range', len(bins) == 7,
          str(len(bins)))
    check('an RNA layout is NOT forced onto that axis -- it packs dense',
          len(P.genomic_bins(recs, modality='RNA')) == 5,
          str(len(P.genomic_bins(recs, modality='RNA'))))
    check('the empty slots name the loci nobody imaged',
          [b for b in bins if not b[1]] == [(4, ''), (5, '')], str(bins))
    check('a DECLARED range sets the axis outright, ends included',
          P.genomic_bins(recs, modality='DNA', declared='1-9')
          == [(1, 'Hyb_001'), (2, 'Hyb_002'), (3, 'Hyb_003'), (4, ''),
              (5, ''), (6, 'Hyb_006'), (7, 'Hyb_007'), (8, ''), (9, '')],
          str(P.genomic_bins(recs, modality='DNA', declared='1-9')))
    check('and a bin outside it is refused, not dropped',
          _raises(lambda: P.genomic_bins(recs, modality='DNA',
                                         declared='1-5'), 'does not contain'))
    check('and the bins after them keep their genomic position',
          bins[5] == (6, 'Hyb_006') and bins[6] == (7, 'Hyb_007'),
          str(bins[5:]))
    keys = P.bin_keys(recs, modality='DNA')
    check('their key is None, which nothing can match',
          keys[3] is None and keys[4] is None)
    check('bin_hybes reports them as empty labels',
          P.bin_hybes(recs, modality='DNA')[3:5] == ['', ''],
          str(P.bin_hybes(recs, modality='DNA')))

    a = allele(polymer_adj={'Hyb_001': [(1., 2., 3., 10.)],
                            'Hyb_006': [(4., 5., 6., 20.)]})
    pos, amp, n = P.collapse_polymer(a, keys)
    check('the collapse leaves the holes NaN without a special case',
          np.isnan(pos[3]).all() and np.isnan(pos[4]).all())
    check('a bin that WAS imaged and found nothing is NaN too',
          np.isnan(pos[1]).all() and n[1] == 0)
    check('the two measured bins land at slots 0 and 5',
          tuple(pos[0]) == (1., 2., 3.) and tuple(pos[5]) == (4., 5., 6.),
          str(pos))
    check('a hole cannot be told from a miss by position alone, which is '
          'why the export carries the empty hybe', amp.shape == (7,))

    print('\n  the dense policy still exists, and still closes the gap')
    dense = ReadoutMap.from_layout(recs, pindex_from='dense')
    check('dense gives 5 slots and no holes',
          len(P.genomic_bins(rmap=dense)) == 5
          and dense.gaps() == [],
          str(P.genomic_bins(rmap=dense)))
    check('and readout 6 then sits where readout 4 belonged',
          P.genomic_bins(rmap=dense)[3] == (6, 'Hyb_006'),
          str(P.genomic_bins(rmap=dense)[3]))


def two_channels():
    print('\ntwo readouts in one hybe occupy two slots')
    recs = [rec('Hyb_001', 1, readouts=[(635, 1), (747, 51)]),
            rec('Hyb_002', 2, readouts=[(635, 2), (747, 52)])]
    keys = P.bin_keys(recs, channels=[635, 747], modality='DNA')
    labels = P.bin_hybes(recs, channels=[635, 747], modality='DNA')
    check('the axis spans readouts 1..52', len(keys) == 52, str(len(keys)))
    check('each hybe appears twice, under its own channel',
          [k for k in keys if k is not None]
          == [('Hyb_001', 635), ('Hyb_002', 635),
              ('Hyb_001', 747), ('Hyb_002', 747)],
          str([k for k in keys if k is not None]))

    a = allele(polymer_adj={('Hyb_001', 635): [(1., 1., 1., 10.)],
                            ('Hyb_001', 747): [(9., 9., 9., 20.)]})
    pos, _amp, _n = P.collapse_polymer(a, keys)
    check('the same hybe gives DIFFERENT positions to its two slots',
          tuple(pos[0]) == (1., 1., 1.) and tuple(pos[50]) == (9., 9., 9.),
          str((pos[0], pos[50])))

    print('\n  and the hybe name alone stops identifying a locus')
    chans = [(k[1] if k else 0) for k in keys]
    try:
        P.slot_index(labels, 'Hyb_001', chans)
        check('an unqualified hybe is refused', False, 'accepted')
    except ValueError as e:
        check('an unqualified hybe is refused', 'name the channel' in str(e),
              str(e))
    check('with the channel it resolves',
          P.slot_index(labels, 'Hyb_001', chans, channel=747) == 50,
          str(P.slot_index(labels, 'Hyb_001', chans, channel=747)))


def old_stores():
    print('\nan old store keyed by bare hybe still reads')
    recs = [rec('Hyb_001', 1), rec('Hyb_002', 2)]
    keys = P.bin_keys(recs, channels=[635])
    check('the axis keys carry the activated channel',
          keys == [('Hyb_001', 635), ('Hyb_002', 635)], str(keys))
    a = allele(polymer_adj={'Hyb_001': [(1., 2., 3., 10.)]})
    pos, _a, n = P.collapse_polymer(a, keys)
    check('a (hybe, channel) slot falls back to the bare hybe',
          tuple(pos[0]) == (1., 2., 3.) and n[0] == 1, str(pos[0]))

    print('\n  but not where the fallback would duplicate a measurement')
    recs2 = [rec('Hyb_001', 1, readouts=[(635, 1), (747, 51)])]
    keys2 = [k for k in P.bin_keys(recs2, channels=[635, 747],
                                   modality='DNA') if k is not None]
    tries = P.bin_lookup_keys(keys2)
    check('a hybe in two slots is NOT offered the bare key',
          all(all(isinstance(t, tuple) for t in tr) for tr in tries),
          str(tries))
    pos2, _a2, _n2 = P.collapse_polymer(
        allele(polymer_adj={'Hyb_001': [(1., 2., 3., 10.)]}), keys2)
    check('so the old bare data reads as absent, not as both loci',
          np.isnan(pos2).all(), str(pos2))

    print('\n  and a channelled store read through a bare axis refuses')
    try:
        P.collapse_polymer(
            allele(polymer_adj={('Hyb_001', 635): [(1., 2., 3., 10.)]}),
            ['Hyb_001', 'Hyb_002'])
        check('it does not report an empty polymer', False, 'accepted')
    except ValueError as e:
        check('it does not report an empty polymer', 'bin_keys' in str(e),
              str(e))


def export_rows():
    print('\nthe export row says which it was')
    recs = [rec('Hyb_001', 1), rec('Hyb_002', 2), rec('Hyb_004', 4),
            rec('Rep_001', 1, 'R')]
    bins = P.genomic_bins(recs, modality='DNA')
    by_hybe = {r['folder']: r for r in recs}
    a = allele(polymer_adj={'Hyb_001': [(1., 2., 3., 10.)],
                            'Rep_001': [(7., 7., 7., 5.)]},
               rejected_hybes={'Hyb_002': 'fiducial refused'})
    rows = E.allele_rows([a], fov=3, bins=bins, modality='DNA',
                         records_by_hybe=by_hybe)
    by_bin = {}
    for r in rows:
        by_bin.setdefault(r['bin_index'], []).append(r)
    check('every slot gets a row, the empty one included',
          sorted(by_bin) == [-1, 0, 1, 2, 3], str(sorted(by_bin)))
    hole = by_bin[2][0]
    check('the empty slot carries no hybe and no coordinate',
          hole['hybe'] == '' and hole['n_candidates'] == 0
          and np.isnan(hole['readout_adj_x']),
          f"{hole['hybe']!r} {hole['n_candidates']}")
    check('but it names the readout the design skipped',
          hole['readout_id'] == 3, str(hole['readout_id']))
    check('a bin that failed names its reason, so the two are distinct',
          by_bin[1][0]['rejected_reason'] == 'fiducial refused'
          and by_bin[2][0]['rejected_reason'] == '',
          str(by_bin[1][0]['rejected_reason']))
    check('the R round is still exported, at bin_index -1',
          by_bin[-1][0]['hybe'] == 'Rep_001'
          and by_bin[-1][0]['datatype'] == 'R',
          str((by_bin[-1][0]['hybe'], by_bin[-1][0]['datatype'])))
    check('the channel column is 0 when nobody named one',
          {r['readout_channel'] for r in rows} == {0},
          str({r['readout_channel'] for r in rows}))

    print('\n  with channels, each slot reports its own')
    recs2 = [rec('Hyb_001', 1, readouts=[(635, 1), (747, 3)])]
    keys2 = P.bin_keys(recs2, channels=[635, 747], modality='DNA')
    bins2 = P.genomic_bins(recs2, channels=[635, 747], modality='DNA')
    chans2 = [(k[1] if k else 0) for k in keys2]
    a2 = allele(polymer_adj={('Hyb_001', 747): [(9., 9., 9., 20.)]})
    rows2 = E.allele_rows([a2], fov=3, bins=bins2, modality='DNA',
                          records_by_hybe={r['folder']: r for r in recs2},
                          bin_channels=chans2)
    got = {(r['bin_index'], r['hybe'], r['readout_channel'],
            r['n_candidates']) for r in rows2}
    check('the 747 slot has the data and the 635 slot does not',
          (2, 'Hyb_001', 747, 1) in got and (0, 'Hyb_001', 635, 0) in got,
          str(sorted(got)))
    check('and the untouched middle slot is the hole',
          (1, '', 0, 0) in got, str(sorted(got)))
    check('the hybe is NOT repeated as a QC round',
          all(r['bin_index'] >= 0 for r in rows2),
          str([r['bin_index'] for r in rows2]))


def main():
    contiguous()
    holes()
    two_channels()
    old_stores()
    export_rows()
    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
