"""
A degenerate fit is rejected silently, not one RuntimeWarning per fit.

WHY. fit3d_um took sqrt of a covariance diagonal that pinv can make
negative, got NaN, and rejected the fit on the next line -- correct --
while numpy printed a RuntimeWarning for it. A bundle build runs
hundreds of thousands of anchor fits, and every one of those lines went
down the pipe into the Build model window's log.

Run:  python tests/test_fit_warning_silenced.py
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline.localization import fit3d_um as F      # noqa: E402

CHECKS = [0, 0]


def check(name, ok, note=''):
    CHECKS[0] += 1
    CHECKS[1] += 1 if ok else 0
    print(('  ok   ' if ok else '  FAIL ') + name
          + (('  -- ' + note) if note else ''))


def main():
    print('fit3d_um on cubes that cannot support a covariance')
    import inspect
    src = inspect.getsource(F.fit_gaussian_3d_um)
    check('the sqrt is under errstate(invalid=ignore)',
          "np.errstate(invalid='ignore')" in src
          and src.index("np.errstate(invalid='ignore')")
          < src.index('se = np.sqrt(np.diag(cov))'))
    check('and the NaN rejection that follows it is still there',
          'if not np.all(np.isfinite(se)):' in src)

    rng = np.random.RandomState(0)
    cubes = {
        'flat': np.full((15, 15, 21), 300.0),
        'noise only': rng.normal(300.0, 1.0, (15, 15, 21)),
        'one hot voxel': np.where(
            np.arange(15 * 15 * 21).reshape(15, 15, 21) == 7 * 15 * 21 + 7 * 21 + 10,
            5000.0, 300.0),
    }
    for name, cube in cubes.items():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            try:
                out = F.fit_gaussian_3d_um(cube, 7, 7, 10)
                raised = None
            except Exception as exc:                        # noqa: BLE001
                out, raised = None, f'{type(exc).__name__}: {exc}'
        rt = [x for x in w if issubclass(x.category, RuntimeWarning)
              and 'fit3d_um' in str(x.filename)]
        check(f'{name}: no RuntimeWarning from fit3d_um itself',
              not rt, str(rt[0].message) if rt else '')
        check(f'{name}: no exception', raised is None, str(raised))
    print()
    print('%d/%d checks passed' % (CHECKS[1], CHECKS[0]))
    return 0 if CHECKS[1] == CHECKS[0] else 1


if __name__ == '__main__':
    sys.exit(main())
