"""p_exist's two factors are kept, and the write-time floor is minimal.

WHAT WENT WRONG. `p_exist` is `p1 * cal(p3)` (psfmatcher._joint) and only
the product was ever stored, so a stored spot could not be re-gated on
either factor. That is worse than it sounds: the multispot calibration
is near a step -- MEASURED on the pinned default model, cal is 0.002 at
p3 = 0.6 and 0.964 at 0.8 -- so cal(p3) is exactly 0 over most of its
range and p1 cannot be recovered from the product by dividing. The
matcher's own NCC was on the LocalizedSpot as `p` and thrown away by
`batch._spot_dict`; p1 was never on the spot at all.

AND THE FLOOR DELETED THE EVIDENCE. `batch.detect_fov_hybe` applies
min_p_exist BEFORE writing, so whatever it drops needs a NAS re-read to
get back -- 64-85 s per (FOV, hybe) measured on chr19 FOV 17. At the old
0.5 that removed the bulk of the distribution, not a dead tail: on
chr19 FOV 17 (Hyb_064/028/014, pinned default model) only 1.6-2.3% of
scored candidates sit at p_exist 0 while 89-96% land in (0, 0.5).

These tests pin the three things that fix has to keep true:
no schema change (both factors ride as scalar PROPERTIES, which
io/columnar already gives their own columns), a store written before the
change still reads, and the floor stays a floor rather than drifting
back up into a review threshold.

Run: python tests/test_spot_probability_factors.py
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np

from codelab_pipeline.io import columnar
from codelab_pipeline.localization import batch
from codelab_pipeline.localization.engine import LocalizedSpot, _spot
from codelab_pipeline.models.spot import ASpot

_n = {'pass': 0, 'fail': 0}


def check(name, ok, extra=''):
    _n['pass' if ok else 'fail'] += 1
    print('  %-4s %s%s' % ('ok' if ok else 'FAIL', name,
                           ('   [%s]' % extra) if extra else ''))


class _Cell:
    id, celltype = 7, 'WT'


def roundtrip(dicts):
    bio = io.BytesIO()
    with h5py.File(bio, 'w') as f:
        columnar.pack_spots(f.create_group('g'), dicts)
    with h5py.File(bio, 'r') as f:
        return columnar.unpack_spots(f['g'])


def test_the_spot_carries_both_factors():
    print('\nthe engine spot carries p1 beside p and p_exist')
    check('LocalizedSpot declares p1', 'p1' in LocalizedSpot._fields)
    s = _spot(1.0, 2.0, 3.0, 0.8, p_exist=0.6, p1=0.75)
    check('_spot passes it through', s.p1 == 0.75, repr(s.p1))
    check('p is still the per-engine quality', s.p == 0.8)
    check('p_exist is still the product', s.p_exist == 0.6)
    # AN ENGINE WITHOUT A CLASSIFIER MUST NOT HAVE TO INVENT ONE, the
    # same contract the Gaussian fields already have.
    d = _spot(1.0, 2.0, 3.0, 0.5)
    check('it defaults to NaN, not 0.0', np.isnan(d.p1), repr(d.p1))
    check('and p_exist still defaults to NaN', np.isnan(d.p_exist))


def test_both_factors_reach_the_store_dict():
    print('\nbatch._spot_dict writes the factors, not only the product')
    s = _spot(5.0, 6.0, 7.0, 0.82, amplitude=400.0, p_exist=0.61, p1=0.74)
    d = batch._spot_dict(s, 10, 20, None, 0.0, 1, 'Hyb_003', 635, 'DNA',
                         _Cell())
    check('p_exist is there', abs(d.get('p_exist', -1) - 0.61) < 1e-6)
    check('p (the matcher NCC) is there', abs(d.get('p', -1) - 0.82) < 1e-6,
          repr(d.get('p')))
    check('p1 (the classifier) is there', abs(d.get('p1', -1) - 0.74) < 1e-6,
          repr(d.get('p1')))
    # NOT IN ASpot._KNOWN ON PURPOSE: an unknown scalar is kept as a
    # property and io/columnar gives it its own column, so neither the
    # model nor the store schema had to change.
    check('they ride as properties, not new model fields',
          'p' not in ASpot._KNOWN and 'p1' not in ASpot._KNOWN)


def test_they_survive_the_store():
    print('\nthey survive pack -> unpack')
    src = [batch._spot_dict(
        _spot(float(i), float(i + 1), 2.0, 0.7 + i / 100.0, amplitude=100.0,
              p_exist=0.3 + i / 100.0, p1=0.5 + i / 100.0),
        0, 0, None, 0.0, 1, 'Hyb_00%d' % i, 635, 'DNA', _Cell())
        for i in range(4)]
    back = roundtrip(src)
    check('same number of spots', len(back) == len(src),
          '%d vs %d' % (len(back), len(src)))
    ok_p = all(abs(float(b['p']) - float(a['p'])) < 1e-6
               for a, b in zip(src, back))
    ok_p1 = all(abs(float(b['p1']) - float(a['p1'])) < 1e-6
                for a, b in zip(src, back))
    check('p round-trips', ok_p)
    check('p1 round-trips', ok_p1)
    check('p_exist still round-trips',
          all(abs(float(b['p_exist']) - float(a['p_exist'])) < 1e-6
              for a, b in zip(src, back)))


def test_nan_stays_nan():
    print('\nan engine with no classifier reports nothing, not zero')
    d = batch._spot_dict(_spot(1.0, 1.0, 1.0, float('nan')),
                         0, 0, None, 0.0, 1, 'Hyb_001', 635, 'DNA', _Cell())
    back = roundtrip([d])[0]
    # ZERO WOULD BE A MEASUREMENT. `is_refined` reads exactly this: a
    # spot model 3 could not place keeps NaN in p, and filling it with
    # something plausible would hide that its position is an integer.
    check('p comes back NaN', np.isnan(float(back.get('p', 0.0))),
          repr(back.get('p')))
    check('p1 comes back NaN', np.isnan(float(back.get('p1', 0.0))),
          repr(back.get('p1')))


def test_an_older_store_still_reads():
    print('\na store written before the factors existed still opens')
    old = batch._spot_dict(_spot(1.0, 2.0, 3.0, 0.9, p_exist=0.9, p1=0.9),
                           0, 0, None, 0.0, 1, 'Hyb_001', 635, 'DNA', _Cell())
    old.pop('p', None)
    old.pop('p1', None)
    back = roundtrip([old])
    check('it reads back', len(back) == 1)
    check('and simply has no p', 'p' not in back[0] or
          back[0].get('p') is None or np.isnan(float(back[0]['p'])))


def test_the_floor_is_a_floor():
    print('\nthe write-time floor is minimal, not a review threshold')
    # The gate is applied BEFORE the store, so it is the one number in
    # this pipeline that destroys evidence. The review threshold belongs
    # to p_gate.py, posterior, off a histogram.
    check('DEFAULT_MIN_P_EXIST is well below 0.5',
          batch.DEFAULT_MIN_P_EXIST < 0.5,
          repr(batch.DEFAULT_MIN_P_EXIST))
    check('but is not zero -- x23 of the store is not a default',
          batch.DEFAULT_MIN_P_EXIST > 0.0,
          repr(batch.DEFAULT_MIN_P_EXIST))
    # the panel must not hard-code its own number beside this one
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'ui', 'pipeline_panel.py')
    src = io.open(p, encoding='utf-8').read()
    check('the panel takes its default from batch, not a literal',
          'float(_batch.DEFAULT_MIN_P_EXIST)' in src)


def main():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    for t in (test_the_spot_carries_both_factors,
              test_both_factors_reach_the_store_dict,
              test_they_survive_the_store,
              test_nan_stays_nan,
              test_an_older_store_still_reads,
              test_the_floor_is_a_floor):
        t()
    print('\n%d passed, %d failed' % (_n['pass'], _n['fail']))
    return 1 if _n['fail'] else 0


if __name__ == '__main__':
    sys.exit(main())
