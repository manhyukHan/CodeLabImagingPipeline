"""An allele's hybe dicts survive a round trip once a round carries two
readouts.

columnar packed every hybe-keyed dict into ONE string column, so a
(hybe, channel) key stringified on the way out and came back as the
text "('Hyb_001', 635)" -- no crash, no warning, and every lookup
afterwards missing. The hybe column stays a string; the channel rides
beside it in its own column, and the key is rebuilt from the pair.

Run: python tests/test_allele_channel_storage.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py                                                   # noqa: E402
import numpy as np                                            # noqa: E402

from codelab_pipeline.io import columnar as C                 # noqa: E402
from codelab_pipeline.models.allele import AnAllele           # noqa: E402
from codelab_pipeline.analysis import polymer as P            # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(('  ok ' if cond else 'FAIL ') + name
          + (f'   [{detail}]' if detail and not cond else ''))


def roundtrip(dicts, drop=()):
    """Pack, optionally delete columns to mimic an older file, unpack."""
    path = os.path.join(tempfile.mkdtemp(), 'alleles.h5')
    with h5py.File(path, 'w') as f:
        C.pack_alleles(f.create_group('a'), dicts)
    if drop:
        with h5py.File(path, 'a') as f:
            for name in drop:
                if name in f['a']:
                    del f['a'][name]
    with h5py.File(path, 'r') as f:
        return C.unpack_alleles(f['a'])


def base(**kw):
    a = AnAllele()
    a.set_metadata(id=7, fov=3, cell=2, anchor_hybe='Hyb_001',
                   anchor_channel=635, coordinate=(1., 2., 3.),
                   raw_coordinate=(1., 2., 3.), **kw)
    return a.save()


def single_channel():
    print('a single-channel trace is byte-for-byte what it always was')
    d = base(polymer_adj={'Hyb_001': [(1., 2., 3., 10., 0.9)]},
             polymer_raw={'Hyb_001': [(1., 2., 3., 10., 0.9)]},
             fiducial_trace_adj={'Hyb_001': (4., 5., 6., 20., 0.8),
                                 'Hyb_002': None},
             fiducial_drift={'Hyb_001': (0.1, 0.2, 0.3)},
             rejected_hybes={'Hyb_003': 'no fiducial'})
    out = roundtrip([d])[0]
    check('polymer_adj keeps its bare hybe key',
          list(out['polymer_adj']) == ['Hyb_001'],
          str(list(out['polymer_adj'])))
    check('so does the fiducial trace, None and all',
          sorted(out['fiducial_trace_adj']) == ['Hyb_001', 'Hyb_002']
          and out['fiducial_trace_adj']['Hyb_002'] is None,
          str(out['fiducial_trace_adj']))
    check('and the drift and the rejections',
          list(out['fiducial_drift']) == ['Hyb_001']
          and out['rejected_hybes'] == {'Hyb_003': 'no fiducial'},
          str((out['fiducial_drift'], out['rejected_hybes'])))
    check('a key is a str, not a 1-tuple',
          all(isinstance(k, str) for k in out['polymer_adj']))


def two_channels():
    print('\ntwo readouts in one round keep their own positions')
    d = base(polymer_adj={('Hyb_001', 635): [(1., 1., 1., 10., 0.9)],
                          ('Hyb_001', 747): [(9., 9., 9., 20., 0.8)]},
             fiducial_trace_adj={('Hyb_001', 635): (4., 5., 6., 20., 0.8),
                                 ('Hyb_001', 747): None},
             fiducial_drift={('Hyb_001', 747): (0.1, 0.2, 0.3)},
             rejected_hybes={('Hyb_002', 747): 'no fiducial'})
    out = roundtrip([d])[0]
    check('both keys come back as (hybe, channel) tuples',
          sorted(out['polymer_adj']) == [('Hyb_001', 635), ('Hyb_001', 747)],
          str(list(out['polymer_adj'])))
    check('and they carry DIFFERENT positions, not one copied twice',
          out['polymer_adj'][('Hyb_001', 635)][0][:3] == (1., 1., 1.)
          and out['polymer_adj'][('Hyb_001', 747)][0][:3] == (9., 9., 9.),
          str(out['polymer_adj']))
    check('a None fiducial stays None under its own channel',
          out['fiducial_trace_adj'][('Hyb_001', 747)] is None
          and out['fiducial_trace_adj'][('Hyb_001', 635)] is not None)
    check('drift and rejections are channelled too',
          list(out['fiducial_drift']) == [('Hyb_001', 747)]
          and list(out['rejected_hybes']) == [('Hyb_002', 747)],
          str((out['fiducial_drift'], out['rejected_hybes'])))
    check('no key is the STRING of a tuple',
          not any(isinstance(k, str) and k.startswith('(')
                  for k in out['polymer_adj']),
          str(list(out['polymer_adj'])))

    print('\n  and the collapse reads them as two bins')
    keys = [('Hyb_001', 635), None, ('Hyb_001', 747)]
    pos, _amp, n = P.collapse_polymer(out, keys)
    check('each slot gets its own channel',
          tuple(pos[0]) == (1., 1., 1.) and tuple(pos[2]) == (9., 9., 9.)
          and np.isnan(pos[1]).all(), str(pos))
    check('and the empty slot counted nothing', list(n) == [1, 0, 1], str(n))


def stored_identity():
    print('\nthe readout map is STORED beside the trace, not re-derived')
    d = base(polymer_adj={('Hyb_001', 635): [(1., 1., 1., 10., 0.9)]},
             readout_bins={('Hyb_001', 635): {'rindex': 15, 'dtype': 'H',
                                              'pindex': 0},
                           ('Hyb_001', 747): {'rindex': 65, 'dtype': 'H',
                                              'pindex': 50},
                           'Rep_001': {'rindex': 15, 'dtype': 'R',
                                       'pindex': -1}})
    out = roundtrip([d])[0]
    check('every bin survives, channelled and bare alike',
          sorted(out['readout_bins'], key=str)
          == [('Hyb_001', 635), ('Hyb_001', 747), 'Rep_001'],
          str(sorted(out['readout_bins'], key=str)))
    check('the identity comes back exactly',
          out['readout_bins'][('Hyb_001', 747)]
          == {'rindex': 65, 'dtype': 'H', 'pindex': 50},
          str(out['readout_bins'][('Hyb_001', 747)]))
    check('a QC round keeps its -1 polymer index',
          out['readout_bins']['Rep_001']['pindex'] == -1)
    check('it is a declared field, not a property',
          'readout_bins' not in (out.get('properties') or {}))

    a = AnAllele()
    a.set_metadata(**out)
    check('the model takes it back without losing a bin',
          len(a.readout_bins) == 3 and a.save()['readout_bins'] == out['readout_bins'],
          str(a.readout_bins))


def old_files():
    print('\nan older alleles.h5 has none of these columns')
    d = base(polymer_adj={'Hyb_001': [(1., 2., 3., 10., 0.9)]},
             fiducial_trace_adj={'Hyb_001': (4., 5., 6., 20., 0.8)},
             fiducial_drift={'Hyb_001': (0.1, 0.2, 0.3)},
             rejected_hybes={'Hyb_002': 'no fiducial'})
    out = roundtrip([d], drop=('pl_channel', 'plr_channel', 'tr_channel',
                               'trr_channel', 'fd_channel', 'rj_channel',
                               'rb_allele', 'rb_hybe', 'rb_channel',
                               'rb_rindex', 'rb_dtype', 'rb_pindex'))[0]
    check('it still reads, keyed by the bare hybe',
          list(out['polymer_adj']) == ['Hyb_001']
          and list(out['fiducial_trace_adj']) == ['Hyb_001']
          and list(out['fiducial_drift']) == ['Hyb_001']
          and list(out['rejected_hybes']) == ['Hyb_002'],
          str(list(out['polymer_adj'])))
    check('and reports no stored readout map rather than inventing one',
          out['readout_bins'] == {}, str(out['readout_bins']))
    check('the positions are unchanged',
          out['polymer_adj']['Hyb_001'][0][:3] == (1., 2., 3.),
          str(out['polymer_adj']))

    print('\n  a channel column that does not index its hybe column refuses')
    path = os.path.join(tempfile.mkdtemp(), 'alleles.h5')
    with h5py.File(path, 'w') as f:
        C.pack_alleles(f.create_group('a'), [d])
    with h5py.File(path, 'a') as f:
        del f['a']['pl_channel']
        f['a'].create_dataset('pl_channel', data=np.zeros(99, dtype=np.int32))
    try:
        with h5py.File(path, 'r') as f:
            C.unpack_alleles(f['a'])
        check('a mismatched channel column is refused', False, 'accepted')
    except ValueError as e:
        check('a mismatched channel column is refused',
              'index the same rows' in str(e), str(e))


def main():
    single_channel()
    two_channels()
    stored_identity()
    old_files()
    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
