# -*- coding: utf-8 -*-
"""apply_z_consensus: a refused cell gets the round's depth, not zero."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, r'D:\Program Files\Manhyuk\Projects\CodeLabImagingPipeline')

import numpy as np                                             # noqa: E402

from codelab_pipeline.alignment import chain                   # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(('  ok ' if cond else 'FAIL ') + name
          + (f'   [{detail}]' if detail and not cond else ''))


def entry(dz, rejected, fitted=0.0):
    return ({'yx': np.eye(3), 'dz': float(dz), 'yx_is_residual': True},
            {'reference_sequence': 'H->R', 'steps': np.eye(3)[None],
             'z_rejected': bool(rejected), 'z_fitted': float(fitted)})


def build(n_ok, n_bad, dz=-45.0, hybe='Hyb_023', mod='DNA'):
    r = {}
    for i in range(n_ok):
        r[(i, mod)] = {hybe: entry(dz + (i % 3) - 1, False)}
    for j in range(n_bad):
        r[(1000 + j, mod)] = {hybe: entry(0.0, True, fitted=-77.0)}
    return r


print('the substitution')
r = build(100, 15)
rep = chain.apply_z_consensus(r)
filled = [v['Hyb_023'][0]['dz'] for k, v in r.items() if k[0] >= 1000]
check('every refused cell was filled', all(d != 0.0 for d in filled),
      str(filled[:3]))
check('with the median of the ACCEPTED fits', all(d == -45.0 for d in filled),
      str(set(filled)))
check('the accepted cells are untouched',
      sorted({v['Hyb_023'][0]['dz'] for k, v in r.items() if k[0] < 1000})
      == [-46.0, -45.0, -44.0])
check('it reports what it did', rep.get(('Hyb_023', 'DNA')) == (-45.0, 100, 15),
      str(rep))
note = r[(1000, 'DNA')]['Hyb_023'][1]['reference_sequence']
check('the substitution is recorded as IMPUTED', 'IMPUTED' in note, note)
check('and names the value and the support',
      '-45.0px' in note and '100 accepted' in note, note)

print('\nwhen it must NOT act')
r = build(5, 3)
chain.apply_z_consensus(r)
check('too few accepted cells: nothing is filled',
      all(v['Hyb_023'][0]['dz'] == 0.0 for k, v in r.items() if k[0] >= 1000))
r = build(100, 5, dz=0.0)
chain.apply_z_consensus(r)
check('a consensus of zero is not written over a zero',
      all(v['Hyb_023'][0]['dz'] == 0.0 for k, v in r.items() if k[0] >= 1000))
r = build(100, 0)
check('no refusals: a no-op', chain.apply_z_consensus(r) == {})

print('\nscope')
r = build(100, 10, dz=-42.0, hybe='Hyb_048')
r2 = build(100, 10, dz=-19.0, hybe='Hyb_048')
for k, v in r2.items():
    r.setdefault((k[0] + 5000, k[1]), {}).update(
        {'Hyb_048_fov2': v['Hyb_048']})
chain.apply_z_consensus(r)
check('a different (hybe, modality) is a different consensus',
      {v['Hyb_048'][0]['dz'] for k, v in r.items()
       if 'Hyb_048' in v and k[0] >= 1000 and k[0] < 5000} == {-42.0})

print('\nentries this run did not produce are left alone')
r = {(1, 'DNA'): {'Hyb_001': ({'yx': np.eye(3), 'dz': 0.0}, {'steps': None})}}
check('no z_rejected key: skipped, no crash',
      chain.apply_z_consensus(r) == {})

print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
sys.exit(1 if FAIL else 0)
