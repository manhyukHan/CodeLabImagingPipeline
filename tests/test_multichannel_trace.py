"""A round that carries two readouts is traced twice and stored twice.

The trace path took one readout channel per hybe all the way down -- a
bin was keyed by its folder name, so a second dye on the same adapter
had nowhere to go. v2 fits each bin in turn now, keyed (hybe, channel),
while the FIDUCIAL stays the round's: it is cut once and both bins carry
it. The bare hybe stays the key wherever the round has one readout,
which is every store written so far.

Run: python tests/test_multichannel_trace.py
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                            # noqa: E402

from codelab_pipeline.localization import tracing_v2 as T2    # noqa: E402
from codelab_pipeline.localization import localization as L   # noqa: E402
from codelab_pipeline.models.allele import AnAllele           # noqa: E402
from codelab_pipeline.analysis import polymer as P            # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(('  ok ' if cond else 'FAIL ') + name
          + (f'   [{detail}]' if detail and not cond else ''))


def normalising():
    print('the channel map takes either shape')
    check('one channel per hybe, as every caller wrote it',
          T2.readout_bins_of({'H1': 635, 'H2': 747}) == {'H1': [635],
                                                         'H2': [747]})
    check('or a list, for a round with two readouts',
          T2.readout_bins_of({'H1': [635, 747]}) == {'H1': [635, 747]})
    check('a repeat is not two bins',
          T2.readout_bins_of({'H1': [635, 635]}) == {'H1': [635]})
    check('a hybe with no channel is left out, not given one',
          T2.readout_bins_of({'H1': None, 'H2': [], 'H3': 635})
          == {'H3': [635]},
          str(T2.readout_bins_of({'H1': None, 'H2': [], 'H3': 635})))


def keys():
    print('\nthe key is bare while the round has one readout')
    check('one bin keeps the folder name -- every store so far',
          T2.bin_key('Hyb_001', 635, 1) == 'Hyb_001')
    check('two bins take the pair, since the folder names two loci',
          T2.bin_key('Hyb_001', 635, 2) == ('Hyb_001', 635)
          and T2.bin_key('Hyb_001', 747, 2) == ('Hyb_001', 747))
    check('and the rule matches the one the axis reads by',
          P.bin_lookup_keys([('Hyb_001', 635)]) == [(('Hyb_001', 635),
                                                     'Hyb_001')],
          str(P.bin_lookup_keys([('Hyb_001', 635)])))
    check('which stops offering the bare key once the hybe names two slots',
          P.bin_lookup_keys([('Hyb_001', 635), ('Hyb_001', 747)])
          == [(('Hyb_001', 635),), (('Hyb_001', 747),)],
          str(P.bin_lookup_keys([('Hyb_001', 635), ('Hyb_001', 747)])))


def storage_key_vs_frame():
    print('\nthe STORAGE key and the FRAME hybe are two different things')
    from test_v3_in_tracing import FakeEngine, spot, _slab_z0
    z_r = 20.0
    z0 = _slab_z0(z_r, T2.V2Params().z_reach())
    seen = []

    def to_shared(h, y, x, z, ymin, xmin):
        seen.append(h)
        return (y + ymin, x + xmin, z)

    a = AnAllele()
    a.polymer_adj, a.polymer_raw = {}, {}
    p = T2.V2Params(min_p_exist=0.5)
    p.readout_engine = FakeEngine([spot(8.0, 8.0, z_r - z0, 0.91)])
    cube = np.random.RandomState(0).normal(300.0, 5.0, (17, 17, 105))
    ok, why = T2._readout_multi(a, 'Hyb_001', cube, z_r, p, 0., 0., 0.,
                                100, 200, to_shared, None, seed_yx=(8.0, 8.0),
                                key=('Hyb_001', 747))
    check('the fit lands', ok, str(why))
    check('it is STORED under the bin key',
          list(a.polymer_adj) == [('Hyb_001', 747)]
          and list(a.polymer_raw) == [('Hyb_001', 747)],
          str(list(a.polymer_adj)))
    check('but MAPPED through the round, which is what has matrices',
          set(seen) == {'Hyb_001'}, str(set(seen)))
    check('and without a key it is the hybe, as every caller had it',
          T2._readout_multi.__defaults__[-1] is None)


def builder_source():
    print('\nthe builder splits the round from the bin')
    src = inspect.getsource(T2.build_chromatin_trace_allele)
    check('the readout stage loops over the bins of the round',
          'for channel in chans:' in src
          and 'bk = bin_key(hybe, channel, len(chans))' in src)
    check('every readout result is keyed by the bin',
          'allele.polymer_adj[bk] = ' in src
          and 'allele.polymer_raw[bk] = ' in src
          and 'allele.rejected_hybes[bk] = ' in src
          and 'allele.polymer_adj[hybe] = ' not in src)
    check('the fiducial is the ROUND\'s, written under every bin of it',
          '_put(allele.fiducial_trace_adj, hybe,' in src
          and '_put(allele.fiducial_drift, hybe,' in src
          and 'allele.fiducial_trace_adj[hybe] = ' not in src)
    check('the frame mapping still goes through the hybe, never the key',
          '_to_shared(hybe, r.y, r.x, r.z, ymin, xmin)' in src
          and '_to_shared(bk' not in src)
    check('the crop is cut for the bin\'s OWN channel',
          '_cut(hybe, channel)' in src)
    check('and the preview crop is not handed to another wavelength',
          "if channel == chans[0] and hybe in precut:" in src)


def v1_refuses():
    print('\nv1 is single-channel and says so rather than tracing half')
    src = inspect.getsource(L.build_chromatin_trace_allele)
    check('it names the hybe and the channels it will not trace',
          'v1 chromatin tracing:' in src
          and 'carries %d readout channels' in src
          and 'Trace it' in src)
    check('one channel in a list is still one channel',
          'readout_channel = chans[0] if chans else None' in src)


def round_trip():
    print('\nand a two-bin round survives the store')
    from codelab_pipeline.io import columnar as C
    import h5py
    import tempfile
    a = AnAllele()
    a.set_metadata(id=1, fov=2, cell=3, anchor_hybe='Hyb_001',
                   anchor_channel=635, coordinate=(0., 0., 0.),
                   raw_coordinate=(0., 0., 0.),
                   polymer_adj={('Hyb_001', 635): [(1., 1., 1., 10., .9)],
                                ('Hyb_001', 747): [(9., 9., 9., 20., .8)]},
                   # the fiducial is the ROUND's: the same value twice
                   fiducial_trace_adj={('Hyb_001', 635): (5., 5., 5., 7., .6),
                                       ('Hyb_001', 747): (5., 5., 5., 7., .6)},
                   fiducial_drift={('Hyb_001', 635): (0.1, 0.2, 0.3),
                                   ('Hyb_001', 747): (0.1, 0.2, 0.3)})
    path = os.path.join(tempfile.mkdtemp(), 'alleles.h5')
    with h5py.File(path, 'w') as f:
        C.pack_alleles(f.create_group('a'), [a.save()])
    with h5py.File(path, 'r') as f:
        out = C.unpack_alleles(f['a'])[0]
    keys2 = [('Hyb_001', 635), ('Hyb_001', 747)]
    pos, _amp, n = P.collapse_polymer(out, keys2)
    check('the two bins read back as two different positions',
          tuple(pos[0]) == (1., 1., 1.) and tuple(pos[1]) == (9., 9., 9.),
          str(pos))
    check('each counted one candidate', list(n) == [1, 1], str(n))
    check('and both carry the round\'s one fiducial',
          out['fiducial_trace_adj'][('Hyb_001', 635)]
          == out['fiducial_trace_adj'][('Hyb_001', 747)],
          str(out['fiducial_trace_adj']))


def stamping():
    print('\nthe settled map is stamped onto what the trace produced')
    from codelab_pipeline.analysis.readout_map import ReadoutMap
    m = ReadoutMap.from_text(
        'slots=52,Hyb_001|635|1|H|0,Hyb_002|635|2|H|1,'
        'Hyb_001|747|51|H|50,Rep_001|635|1|R|-1')
    a = AnAllele()
    a.polymer_adj = {('Hyb_001', 635): [(1., 1., 1., 9.)],
                     ('Hyb_001', 747): [(9., 9., 9., 9.)]}
    a.rejected_hybes = {('Hyb_002', 635): 'fiducial not found'}
    a.fiducial_trace_adj = {'Rep_001': (0., 0., 0., 1., 1.)}
    P.stamp_readout_bins(a, m)
    check('every key the trace made is described',
          sorted(a.readout_bins, key=str)
          == [('Hyb_001', 635), ('Hyb_001', 747), ('Hyb_002', 635),
              'Rep_001'],
          str(sorted(a.readout_bins, key=str)))
    check('each with its own locus and place',
          a.readout_bins[('Hyb_001', 747)]
          == {'rindex': 51, 'dtype': 'H', 'pindex': 50},
          str(a.readout_bins[('Hyb_001', 747)]))
    check('a QC round is described too, and is not a bin',
          a.readout_bins['Rep_001'] == {'rindex': 1, 'dtype': 'R',
                                        'pindex': -1})
    check('a rejected bin still says what it was',
          a.readout_bins[('Hyb_002', 635)]['pindex'] == 1)

    print('  a bare key resolves only where the round has ONE bin')
    b = AnAllele()
    b.polymer_adj = {'Hyb_001': [(1., 1., 1., 9.)],
                     'Hyb_002': [(2., 2., 2., 9.)]}
    P.stamp_readout_bins(b, m)
    check('Hyb_002 has one bin, so its bare key resolves',
          b.readout_bins.get('Hyb_002') == {'rindex': 2, 'dtype': 'H',
                                            'pindex': 1},
          str(b.readout_bins))
    check('Hyb_001 has two, so a bare key is left undescribed, not guessed',
          'Hyb_001' not in b.readout_bins, str(b.readout_bins))

    print('  and no map means no stamp, not an empty one')
    c = AnAllele()
    c.polymer_adj = {'Hyb_001': [(1., 1., 1., 9.)]}
    c.readout_bins = {'Hyb_001': {'rindex': 7, 'dtype': 'H', 'pindex': 6}}
    P.stamp_readout_bins(c, None)
    check('an existing stamp is left alone',
          c.readout_bins == {'Hyb_001': {'rindex': 7, 'dtype': 'H',
                                         'pindex': 6}},
          str(c.readout_bins))


def main():
    normalising()
    keys()
    storage_key_vs_frame()
    builder_source()
    v1_refuses()
    round_trip()
    stamping()
    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
