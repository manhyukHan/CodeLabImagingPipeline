"""
Provenance: a free-form record of HOW a trace was made.

v1 and v2 differ by 43-68% in localization error, so a store holding both
is holding two accuracy regimes. Nothing recorded which was which, and
file mtime cannot answer it either -- an append run rewrites the whole FOV
capsule, so every allele in it looks equally fresh.

The container is deliberately a free-form dict, not fields: each engine
records its own inputs, and a future engine adds its hyperparameters
without this class or the on-disk schema changing again.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_allele_provenance.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py                                                     # noqa: E402
import numpy as np                                              # noqa: E402

from codelab_pipeline.io import columnar                        # noqa: E402
from codelab_pipeline.models.allele import AnAllele             # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}' + (f'   [{detail}]' if detail else ''))


def make(aid=1, provenance=None):
    a = AnAllele()
    a.set_metadata(id=aid, fov=1, cell=7, anchor_uid=aid, anchor_hybe='Hyb_016',
                   anchor_channel=555, coordinate=(100.0, 200.0, 30.0),
                   raw_coordinate=(101.0, 201.0, 31.0))
    a.polymer = {'Hyb_020': [(1.0, 2.0, 3.0, 500.0)]}
    if provenance is not None:
        a.provenance = provenance
    return a


def roundtrip(alleles):
    tmp = os.path.join(tempfile.mkdtemp(prefix='prov_'), 'a.h5')
    with h5py.File(tmp, 'w') as f:
        columnar.pack_alleles(f, [a.save() for a in alleles])
    with h5py.File(tmp, 'r') as f:
        return columnar.unpack_alleles(f), tmp


def main():
    print('the container is free-form')
    a = AnAllele()
    check('a fresh allele has an empty provenance, not None',
          a.provenance == {}, repr(a.provenance))
    weird = {'engine': 'v3-ml', 'checkpoint': 'run/42/best.pt',
             'hyperparams': {'lr': 0.001, 'layers': [64, 64]},
             'notes': ['retrained', 'on 2026-09-01']}
    a.provenance = weird
    check('it accepts arbitrary nested content, no schema enforced',
          a.save()['provenance'] == weird)

    print('\nround trip through the columnar store')
    got, path = roundtrip([make(1, {'engine': 'v1', 'engine_label': 'v1 (ChrTracer3 port)'}),
                           make(2, {'engine': 'v2', 'psf': 'universal-default',
                                    'voxel_um': [0.208, 0.208, 0.2]}),
                           make(3, weird)])
    check('provenance survives pack/unpack exactly',
          got[0]['provenance']['engine'] == 'v1'
          and got[1]['provenance']['psf'] == 'universal-default'
          and got[2]['provenance'] == weird,
          str(got[1]['provenance']))
    check('and rebuilding an AnAllele from it keeps the dict',
          AnAllele().__class__ and
          (lambda x: x.provenance == weird)(
              (lambda: (lambda o: (o.set_metadata(**got[2]), o)[1])(AnAllele()))()))
    check('two alleles do not share one provenance dict',
          got[0]['provenance'] is not got[1]['provenance'])

    print('\nJSON longer than S64 -- the reason it is a vlen column')
    long_prov = {'engine': 'v2', 'psf': 'universal-default',
                 'voxel_um': [0.208, 0.208, 0.2],
                 'traced_at': '2026-08-28T03:00:00',
                 'fiducial_gates': {'min_occupancy': 0.25, 'reject_at_bound': True},
                 'readout_gates': {'min_occupancy': 0.40, 'reject_at_bound': True}}
    encoded = json.dumps(long_prov, sort_keys=True)
    check('a realistic entry exceeds S64, so a fixed column would truncate it',
          len(encoded) > 64, f'{len(encoded)} chars')
    got2, _ = roundtrip([make(1, long_prov)])
    check('it survives intact anyway', got2[0]['provenance'] == long_prov,
          str(len(json.dumps(got2[0]['provenance']))) + ' chars back')

    print('\nolder stores stay readable')
    # Exactly an alleles.h5 written before provenance existed.
    tmp = os.path.join(tempfile.mkdtemp(prefix='old_'), 'a.h5')
    with h5py.File(tmp, 'w') as f:
        columnar.pack_alleles(f, [make(1).save()])
        del f['provenance']
    with h5py.File(tmp, 'r') as f:
        old = columnar.unpack_alleles(f)
    check('a file with NO provenance column still unpacks',
          len(old) == 1 and old[0]['id'] == 1)
    check('and reports an empty dict rather than raising -- those traces are '
          'good, they just do not know how they were made',
          old[0]['provenance'] == {}, repr(old[0]['provenance']))
    check('its real data is untouched',
          old[0]['polymer']['Hyb_020'] == [(1.0, 2.0, 3.0, 500.0)],
          str(old[0]['polymer']))

    print('\nthe engines stamp it')
    from codelab_pipeline.localization import tracing_v2 as V2
    from unittest import mock
    from codelab_pipeline.localization import localization as L
    a1 = make(1)
    with mock.patch.object(L, 'build_chromatin_trace_allele',
                           return_value=(a1, None)):
        V2.trace_allele('v1 (ChrTracer3 port)', a1, [], 'R', {}, {}, '/x', 1,
                        'DNA', None, {})
    check('a v1 run records engine=v1', a1.provenance.get('engine') == 'v1',
          str(a1.provenance))
    check('and keeps the label it was selected by',
          'ChrTracer3' in a1.provenance.get('engine_label', ''))
    check('and when it was traced', bool(a1.provenance.get('traced_at')))

    a2 = make(2)
    p = V2.V2Params(voxel_um=(0.208, 0.208, 0.2), psf_label='universal-default')
    V2.trace_allele('v2', a2, [], 'R', {}, {}, '/nowhere', 1, 'DNA', None, {},
                    v2_params=p)
    check('a v2 run records the engine, the PSF and the voxel size',
          a2.provenance.get('engine') == 'v2'
          and a2.provenance.get('psf') == 'universal-default'
          and a2.provenance.get('voxel_um') == [0.208, 0.208, 0.2],
          str(a2.provenance))
    check('and its gates, so the run is reconstructable',
          'min_occupancy' in (a2.provenance.get('readout_gates') or {}),
          str(a2.provenance.get('readout_gates')))
    check('the whole stamp is JSON-serialisable -- it has to survive the store',
          isinstance(json.dumps(a2.provenance), str))

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        for f in FAIL:
            print('  FAILED:', f)
        return 1
    print('ALL GOOD')
    return 0


if __name__ == '__main__':
    sys.exit(main())
