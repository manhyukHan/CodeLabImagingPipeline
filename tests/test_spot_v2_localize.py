"""
Spot Localization's v2 3D engine, end to end on a fabricated stack.

Pins: v2 is the batch default; the fit finds a synthetic emitter's z
from the spot's own crop (NO consensus leg); the quality gates reject
honestly with the gate's own words while the fitted position survives
for the blue-circle display; the v1 path still runs when asked; and
the 7-tuple batch contract carries the reason.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_spot_v2_localize.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                             # noqa: E402

from codelab_pipeline.io import paths, preprocess              # noqa: E402
from codelab_pipeline.localization import localization as L    # noqa: E402
from codelab_pipeline.models.spot import ASpot                 # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}'
          + (f'   [{detail}]' if detail else ''))


root = tempfile.mkdtemp(prefix='spotv2_')
try:
    dp = os.path.join(root, 'proj')
    paths.write_manifest(dp, ['DNA'])
    sp = os.path.join(dp, 'DNA')
    H, W, Z = 64, 64, 40
    TY, TX, TZ = 30.0, 34.0, 22.0
    yy, xx, zz = np.meshgrid(np.arange(H), np.arange(W), np.arange(Z),
                             indexing='ij')
    blob = 3000.0 * np.exp(-(((yy - TY) / 1.5) ** 2 + ((xx - TX) / 1.5) ** 2
                             + ((zz - TZ) / 2.0) ** 2) / 2)
    stack = (blob + 100
             + np.random.default_rng(0).normal(0, 5, blob.shape)
             ).clip(0, 65535).astype(np.uint16)
    stack_h5 = paths.stack_path(sp, 1, 'Hyb_001')
    os.makedirs(os.path.dirname(stack_h5), exist_ok=True)
    err = preprocess.publish_stack(
        stack_h5, {'hybe': 'Hyb_001', 'fov': 1, 'fiducial_channel': 555},
        {635: stack}, sp, 1, 'Hyb_001', 555)
    assert err is None, err

    def spot_at(y, x):
        s = ASpot()
        s.set_metadata(fov=1, hybe='Hyb_001', channel=635, modality='DNA',
                       raw_coordinate=(y, x, 0.0), adj_coordinate=(y, x, 0.0))
        return s

    print('v2 is the default engine, fits the emitter from its own crop')
    params = {'spad': 8, 'v2_min_occupancy': 0.40,
              'v2_max_uncert_xy_nm': 0, 'v2_max_uncert_z_nm': 150}
    res = L.refine_spots_batch([(spot_at(30.4, 33.6), None)], sp, 1, 635,
                               'Hyb_001', 'DNA', params, None, None,
                               want_grid=True)
    i, coord, raw, mix, cubic, fit_local, reason = res[0]
    check('accepted: reason None, coordinates real', reason is None
          and coord is not None and raw is not None)
    check('z recovered from the crop (no consensus leg)',
          abs(raw[2] - TZ) < 1.0, f'z={raw[2]:.2f} vs {TZ}')
    check('lateral position recovered sub-pixel',
          abs(raw[0] - TY) < 0.5 and abs(raw[1] - TX) < 0.5,
          f'y={raw[0]:.2f} x={raw[1]:.2f}')
    check('grid payload carries crop-local (x, y, z) fit',
          cubic is not None and fit_local is not None
          and abs(fit_local[2] - TZ) < 1.0)

    print('\nthe gates reject honestly, position kept for the blue circle')
    hard = dict(params, v2_min_occupancy=1.01)   # impossible occupancy
    res = L.refine_spots_batch([(spot_at(30.4, 33.6), None)], sp, 1, 635,
                               'Hyb_001', 'DNA', hard, None, None,
                               want_grid=True)
    i, coord, raw, mix, cubic, fit_local, reason = res[0]
    check('gate-rejected: coordinates None, reason in the gate\'s words',
          coord is None and raw is None and reason is not None
          and 'occupancy' in reason, str(reason))
    check('the fitted position SURVIVES for display (blue circle)',
          fit_local is not None and abs(fit_local[2] - TZ) < 1.0)

    print('\nno-signal crop refuses, does not fabricate')
    res = L.refine_spots_batch([(spot_at(5.0, 58.0), None)], sp, 1, 635,
                               'Hyb_001', 'DNA', params, None, None)
    i, coord, raw, mix, _c, _f, reason = res[0]
    check('background-only crop rejected with a reason',
          coord is None and reason is not None, str(reason))

    print('\nPSF choice: free vs auto (no installed PSF -> same fallback)')
    res_free = L.refine_spots_batch([(spot_at(30.4, 33.6), None)], sp, 1, 635,
                                    'Hyb_001', 'DNA',
                                    dict(params, v2_psf='free'), None, None)
    check('v2_psf=free fits free-sigma and accepts',
          res_free[0][1] is not None and abs(res_free[0][2][2] - TZ) < 1.0)

    print('\nView Stored: crops + circles from the persisted trace, no fit')
    from codelab_pipeline.localization import tracing_v2 as V2
    allele_dict = {'fiducial_trace_raw': {'Hyb_001': (TY, TX, TZ)},
                   'polymer_raw': {'Hyb_001': [(TY, TX, TZ, 3000.0)]}}
    debug = V2.stored_allele_debug(
        (allele_dict, ['Hyb_001'], {'Hyb_001': 635}, {'Hyb_001': 635},
         sp, 1, 8))
    d = debug.get('Hyb_001', {})
    check('stored view reads the crop and circles the stored position',
          d.get('fiducial_cubic') is not None
          and d.get('fiducial_centroid') is not None
          and abs(d['fiducial_centroid'][2] - TZ) < 1e-9
          and d.get('readout_centroids'))
    check('no fit properties fabricated (occ/uncert stay absent)',
          'fiducial_occupancy' not in d and 'readout_occupancy' not in d)

    print('\nthe v1 engine stays selectable')
    v1 = {'engine': 'gaussian', 'spad': 8, 'peak_bound': 2.0,
          'max_sigma': 2.5, 'max_uncert': 2.0, 'min_hb_ratio': 1.15,
          'min_ah_ratio': 0.15, 'min_sep': 3.0, 'multi_mode': False,
          'z_window': 15}
    res = L.refine_spots_batch([(spot_at(30.4, 33.6), None)], sp, 1, 635,
                               'Hyb_001', 'DNA', v1, None, None)
    i, coord, raw, mix, _c, _f, reason = res[0]
    check('v1 path accepts the same emitter through the 7-tuple contract',
          coord is not None and abs(raw[2] - TZ) < 1.5 and reason is None)
finally:
    shutil.rmtree(root, ignore_errors=True)

print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
if FAIL:
    for n in FAIL:
        print('  FAILED:', n)
    sys.exit(1)
print('ALL GOOD')
