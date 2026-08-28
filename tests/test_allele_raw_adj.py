"""
Per-hybe positions are stored twice: hybe-native raw, and adjusted.

Mirrors ASpot.raw_coordinate/adj_coordinate, with ONE asymmetry that this
file exists to pin down:

    fiducial_trace_adj - fiducial_trace_raw = alignment only
    polymer_adj        - polymer_raw        = alignment + fiducial drift

A fiducial is not corrected by itself; it IS the correction, so the
readout carries one term more. That asymmetry is deliberate, and an
unstated one is what produced the overlay frame bug -- a shared-frame
delta applied to a raw crop -- so it gets a test rather than a comment.

Also pinned: tr_*/pl_* on disk keep meaning _adj, which is what they
always held, so stores written before raw existed still read correctly
with an empty raw rather than an error.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_allele_raw_adj.py
"""
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


def make(aid=1):
    a = AnAllele()
    a.set_metadata(id=aid, fov=1, cell=7, anchor_uid=aid, anchor_hybe='Hyb_016',
                   anchor_channel=555, coordinate=(100.0, 200.0, 30.0),
                   raw_coordinate=(101.0, 201.0, 31.0))
    a.fiducial_trace_adj = {'Hyb_016': (10.0, 20.0, 30.0, 900.0),
                            'Hyb_020': (11.0, 21.0, 31.0, 800.0)}
    a.fiducial_trace_raw = {'Hyb_016': (14.0, 24.0, 30.0, 900.0),
                            'Hyb_020': (17.0, 27.0, 31.0, 800.0)}
    a.polymer_adj = {'Hyb_020': [(1.0, 2.0, 3.0, 500.0)]}
    a.polymer_raw = {'Hyb_020': [(7.0, 8.0, 3.0, 500.0)]}
    return a


def roundtrip(dicts):
    tmp = os.path.join(tempfile.mkdtemp(), 'a.h5')
    with h5py.File(tmp, 'w') as f:
        columnar.pack_alleles(f, dicts)
    with h5py.File(tmp, 'r') as f:
        return columnar.unpack_alleles(f), tmp


print('both frames survive a round trip')
out, _ = roundtrip([make(1).save()])
d = out[0]
check('fiducial_trace_adj kept', d['fiducial_trace_adj']['Hyb_020'] == (11.0, 21.0, 31.0, 800.0),
      str(d['fiducial_trace_adj']))
check('fiducial_trace_raw kept', d['fiducial_trace_raw']['Hyb_020'] == (17.0, 27.0, 31.0, 800.0),
      str(d['fiducial_trace_raw']))
check('polymer_adj kept', d['polymer_adj']['Hyb_020'] == [(1.0, 2.0, 3.0, 500.0)],
      str(d['polymer_adj']))
check('polymer_raw kept', d['polymer_raw']['Hyb_020'] == [(7.0, 8.0, 3.0, 500.0)],
      str(d['polymer_raw']))
check('raw and adj are NOT the same values (the test would be vacuous)',
      d['polymer_adj']['Hyb_020'] != d['polymer_raw']['Hyb_020'])

print('\nthey stay independent -- no parity assumption between the two')
a = make(2)
a.polymer_raw = {}                      # an engine that fills adj and no raw
out, _ = roundtrip([a.save()])
check('adj alone round-trips with raw empty',
      out[0]['polymer_adj']['Hyb_020'] == [(1.0, 2.0, 3.0, 500.0)]
      and out[0]['polymer_raw'] == {}, str(out[0]['polymer_raw']))
a = make(3)
a.fiducial_trace_raw = {'Hyb_099': (1.0, 1.0, 1.0, 1.0)}   # disjoint hybes
out, _ = roundtrip([a.save()])
check('raw may name hybes adj does not, without cross-assignment',
      set(out[0]['fiducial_trace_raw']) == {'Hyb_099'}
      and set(out[0]['fiducial_trace_adj']) == {'Hyb_016', 'Hyb_020'},
      f"raw={sorted(out[0]['fiducial_trace_raw'])} adj={sorted(out[0]['fiducial_trace_adj'])}")

print('\nmultiple alleles keep their own rows')
out, _ = roundtrip([make(1).save(), make(2).save(), make(3).save()])
check('three alleles, each with both frames intact',
      all(o['polymer_raw']['Hyb_020'] == [(7.0, 8.0, 3.0, 500.0)]
          and o['fiducial_trace_raw']['Hyb_016'] == (14.0, 24.0, 30.0, 900.0)
          for o in out) and len(out) == 3)

print('\na store written before raw existed still reads')
out, tmp = roundtrip([make(1).save()])
with h5py.File(tmp, 'a') as f:
    for k in ('trr_allele', 'trr_hybe', 'trr_isnone', 'trr_vals',
              'plr_allele', 'plr_hybe', 'plr_vals'):
        del f[k]
with h5py.File(tmp, 'r') as f:
    old = columnar.unpack_alleles(f)
check('a file with NO raw columns unpacks rather than raising', len(old) == 1)
check('and reports empty raw -- those traces are good, they just did not '
      'record a hybe-native position',
      old[0]['polymer_raw'] == {} and old[0]['fiducial_trace_raw'] == {},
      str(old[0]['polymer_raw']))
check('its adj data is untouched',
      old[0]['polymer_adj']['Hyb_020'] == [(1.0, 2.0, 3.0, 500.0)]
      and old[0]['fiducial_trace_adj']['Hyb_020'] == (11.0, 21.0, 31.0, 800.0))

print('\nthe legacy unsuffixed keys still load (v1 stores outlive the rename)')
a = AnAllele()
a.set_metadata(id=9, fov=1, cell=1, anchor_hybe='Hyb_016', anchor_channel=555,
               coordinate=(1.0, 2.0, 3.0), raw_coordinate=(1.0, 2.0, 3.0),
               fiducial_trace={'Hyb_020': (5.0, 6.0, 7.0, 8.0)},
               polymer={'Hyb_020': [(1.0, 1.0, 1.0, 1.0)]})
check("'fiducial_trace' loads into fiducial_trace_adj",
      a.fiducial_trace_adj == {'Hyb_020': (5.0, 6.0, 7.0, 8.0)}, str(a.fiducial_trace_adj))
check("'polymer' loads into polymer_adj",
      a.polymer_adj == {'Hyb_020': [(1.0, 1.0, 1.0, 1.0)]}, str(a.polymer_adj))
check('and raw stays empty, not invented', a.polymer_raw == {} and a.fiducial_trace_raw == {})
a2 = AnAllele()
a2.set_metadata(polymer={'A': [(0., 0., 0., 0.)]},
                polymer_adj={'B': [(1., 1., 1., 1.)]})
check('the explicit key wins when both are present',
      set(a2.polymer_adj) == {'B'}, str(a2.polymer_adj))

print('\nNone fiducials survive in both frames')
a = make(4)
a.fiducial_trace_adj['Hyb_030'] = None
a.fiducial_trace_raw['Hyb_030'] = None
out, _ = roundtrip([a.save()])
check('a None fiducial stays None, not a zero tuple',
      out[0]['fiducial_trace_adj']['Hyb_030'] is None
      and out[0]['fiducial_trace_raw']['Hyb_030'] is None,
      f"adj={out[0]['fiducial_trace_adj']['Hyb_030']} raw={out[0]['fiducial_trace_raw']['Hyb_030']}")

print('\nthe asymmetry itself: adj-raw differs between the two dicts')
# Construct the relationship the docstring claims, then assert the
# recoverability it promises: the shared-but-UNCORRECTED readout is
# polymer_adj minus (fid_adj[ref] - fid_adj[hybe]).
align = np.array([3.0, 4.0, 0.0])            # alignment, this hybe
ref, hyb = 'Hyb_016', 'Hyb_020'
a = AnAllele()
a.fiducial_trace_adj = {ref: (10.0, 20.0, 30.0, 1.0), hyb: (12.0, 23.0, 30.0, 1.0)}
a.fiducial_trace_raw = {ref: (10.0, 20.0, 30.0, 1.0),
                        hyb: tuple(np.array([12.0, 23.0, 30.0]) + align) + (1.0,)}
corr = np.array(a.fiducial_trace_adj[ref][:3]) - np.array(a.fiducial_trace_adj[hyb][:3])
raw_r = np.array([50.0, 60.0, 30.0])
a.polymer_raw = {hyb: [tuple(raw_r) + (9.0,)]}
a.polymer_adj = {hyb: [tuple(raw_r - align + corr) + (9.0,)]}
fid_delta = (np.array(a.fiducial_trace_adj[hyb][:3])
             - np.array(a.fiducial_trace_raw[hyb][:3]))
pol_delta = (np.array(a.polymer_adj[hyb][0][:3])
             - np.array(a.polymer_raw[hyb][0][:3]))
check('fiducial adj-raw is alignment ALONE',
      np.allclose(fid_delta, -align), f'{fid_delta} vs {-align}')
check('readout adj-raw is alignment PLUS the fiducial correction',
      np.allclose(pol_delta, -align + corr), f'{pol_delta} vs {-align + corr}')
check('they differ by exactly the fiducial correction -- the asymmetry',
      np.allclose(pol_delta - fid_delta, corr), str(pol_delta - fid_delta))
recovered = np.array(a.polymer_adj[hyb][0][:3]) - corr
check('shared-but-uncorrected is recoverable, so nothing is lost by not '
      'storing it', np.allclose(recovered, raw_r - align), str(recovered))

print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
if FAIL:
    for n in FAIL:
        print('  FAIL', n)
    sys.exit(1)
print('ALL GOOD')
