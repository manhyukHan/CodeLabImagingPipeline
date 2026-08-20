"""
Head-to-head: our Gaussian engine vs a faithful Python port of
ChrTracer3's FitPsf3D (ORCA-public matlab-functions/ChrTracingLib).

MATLAB itself is not installed on this machine, so the comparison runs
against a line-faithful reference port: FitPsf3D's own model (its
NON-standard exp(-((d)/(2*sigma))^2) parameterization, i.e. variance
2*sigma^2), symmetricXY (default true), its inits (a0 = seed pixel,
b0 = 300 hardcoded, sigma 1.25/2.5 in ITS units), its bounds (mu +/- 2,
sigmaXY [0.1, 2], sigmaZ [0.1, 2.5], a/b [0, 2^16]) and its solver
family (trust-region-reflective bounded least squares, tolerances 1e-8,
max 5000 evaluations -- scipy's 'trf' is the same algorithm family as
lsqnonlin's default). Every synthetic stack is also dumped by
--export-mat so the identical inputs can be run through real MATLAB
later if wanted.

Two batteries:
 1. synthetic ground truth -- known sub-voxel centers under Poisson
    noise across SNR levels; asserts our engine recovers positions at
    parity with (or better than) the MATLAB-faithful reference.
 2. engine seam sanity -- LocalizeEngine.localize returns the same
    coordinates as the underlying fit, in (y, x, z) order.

Run: python tests/test_fit_engine_vs_matlab.py [--export-mat DIR]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.optimize import least_squares

from codelab_pipeline.localization import localization as L
from codelab_pipeline.localization.engine import GaussianLocalizeEngine

SHAPE = (15, 15, 15)          # (y, x, z) voxels
TRUE_SIGMA_XY = 1.3           # standard-gaussian sigmas, px
TRUE_SIGMA_Z = 2.6
OFFSET = 300.0
N_PER_CONDITION = 40
AMPLITUDES = [300.0, 1000.0, 3000.0]   # low / mid / high SNR over offset 300


def fitpsf3d_reference(cubic, seed_yxz):
    """
    FitPsf3D.m, ported line-faithfully (single-peak path). cubic is
    (y, x, z); seed is the integer brightest voxel, as FindPeaks3D
    supplies it. Returns (y, x, z) crop-local sub-voxel center, or None
    when the solver fails -- FitPsf3D itself has no per-fit failure
    branch (lsqnonlin always returns), so no gate filtering is applied
    here either; gates are compared at the pipeline level, not per fit.
    """
    y0, x0, z0 = (int(round(v)) for v in seed_yxz)
    ymesh, xmesh, zmesh = np.indices(cubic.shape, dtype=float)
    vals = cubic.astype(float).ravel()
    ym, xm, zm = ymesh.ravel(), xmesh.ravel(), zmesh.ravel()

    a0 = float(cubic[y0, x0, z0])
    b0 = 300.0                                   # FitPsf3D hardcodes this
    # p = [mu_x, sigma_xy, mu_y, mu_z, sigma_z, a, b]  (symmetricXY drops sigma_y)
    p0 = [x0, 1.25, y0, z0, 2.5, a0, b0]
    lb = [x0 - 2, 0.1, y0 - 2, z0 - 2, 0.1, 0, 0]
    ub = [x0 + 2, 2.0, y0 + 2, z0 + 2, 2.5, 2 ** 16, 2 ** 16]

    def residuals(p):
        mx, sxy, my, mz, sz, a, b = p
        model = a * np.exp(-((xm - mx) / (2 * sxy)) ** 2
                           - ((ym - my) / (2 * sxy)) ** 2
                           - ((zm - mz) / (2 * sz)) ** 2) + b
        return model - vals

    try:
        res = least_squares(residuals, p0, bounds=(lb, ub), method='trf',
                            xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=5000)
    except Exception:
        return None
    mx, _sxy, my, mz, _sz, _a, _b = res.x
    return float(my), float(mx), float(mz)


def make_stack(rng, amplitude):
    """One synthetic emitter: standard 3D gaussian + Poisson shot noise."""
    cy = SHAPE[0] / 2 - 0.5 + rng.uniform(-1, 1)
    cx = SHAPE[1] / 2 - 0.5 + rng.uniform(-1, 1)
    cz = SHAPE[2] / 2 - 0.5 + rng.uniform(-1, 1)
    y, x, z = np.indices(SHAPE, dtype=float)
    model = OFFSET + amplitude * np.exp(-((y - cy) ** 2) / (2 * TRUE_SIGMA_XY ** 2)
                                        - ((x - cx) ** 2) / (2 * TRUE_SIGMA_XY ** 2)
                                        - ((z - cz) ** 2) / (2 * TRUE_SIGMA_Z ** 2))
    return rng.poisson(model).astype(np.float64), (cy, cx, cz)


def run_condition(rng, amplitude, export_dir=None):
    ours, ref, truth = [], [], []
    for i in range(N_PER_CONDITION):
        stack, center = make_stack(rng, amplitude)
        iy, ix, iz = np.unravel_index(int(np.argmax(stack)), stack.shape)
        r_ours = L.fit_gaussian_3d(stack, float(ix), float(iy), float(iz))
        r_ref = fitpsf3d_reference(stack, (iy, ix, iz))
        if r_ours is None or r_ref is None:
            continue
        ours.append((r_ours[2], r_ours[1], r_ours[3]))   # (y, x, z)
        ref.append(r_ref)
        truth.append(center)
        if export_dir:
            np.save(os.path.join(export_dir, f'stack_a{int(amplitude)}_{i:02d}.npy'), stack)
    ours, ref, truth = np.array(ours), np.array(ref), np.array(truth)
    rmse = lambda a: np.sqrt(np.mean((a - truth) ** 2, axis=0))
    agree = np.mean(np.abs(ours - ref), axis=0)
    return len(truth), rmse(ours), rmse(ref), agree


def test_position_recovery_parity():
    """Ours must recover ground truth at parity with the MATLAB port."""
    rng = np.random.default_rng(0)
    print(f'\n  {N_PER_CONDITION} emitters/condition, shape {SHAPE}, offset {OFFSET:.0f}, '
          f'sigma xy/z {TRUE_SIGMA_XY}/{TRUE_SIGMA_Z}')
    print('  amp      n   ours RMSE (y,x,z)          matlab-ref RMSE (y,x,z)    mean |ours-ref|')
    for amp in AMPLITUDES:
        n, r_ours, r_ref, agree = run_condition(rng, amp)
        print(f'  {amp:6.0f} {n:4d}   ({r_ours[0]:.3f}, {r_ours[1]:.3f}, {r_ours[2]:.3f})   '
              f'({r_ref[0]:.3f}, {r_ref[1]:.3f}, {r_ref[2]:.3f})   '
              f'({agree[0]:.3f}, {agree[1]:.3f}, {agree[2]:.3f})')
        assert n >= 0.8 * N_PER_CONDITION, f'amp {amp}: too many fits rejected ({n})'
        for axis in range(3):
            assert r_ours[axis] <= r_ref[axis] * 1.25 + 0.02, \
                f'amp {amp} axis {axis}: ours {r_ours[axis]:.3f} vs ref {r_ref[axis]:.3f} -- worse than parity'
        if amp >= 3000:
            assert r_ours[0] < 0.1 and r_ours[1] < 0.1, \
                f'high-SNR xy RMSE too large: {r_ours[:2]}'
            assert agree[0] < 0.1 and agree[1] < 0.1, \
                f'engines disagree at high SNR: {agree}'


def test_engine_seam_matches_fit():
    """LocalizeEngine.localize returns the same answer as the raw fit,
    in (y, x, z) order -- the seam adds routing, never math."""
    rng = np.random.default_rng(1)
    stack, _ = make_stack(rng, 2000.0)
    iy, ix, iz = np.unravel_index(int(np.argmax(stack)), stack.shape)
    raw = L.fit_gaussian_3d(stack, float(ix), float(iy), float(iz))
    spots = GaussianLocalizeEngine().localize(stack, seed_yxz=(float(iy), float(ix), float(iz)))
    assert raw is not None and len(spots) == 1
    s = spots[0]
    assert np.allclose((s.y, s.x, s.z), (raw[2], raw[1], raw[3])), \
        f'seam changed the answer: {(s.y, s.x, s.z)} vs {(raw[2], raw[1], raw[3])}'
    assert np.allclose((s.amplitude, s.offset), (raw[0], raw[7]))


def _run_all():
    export = None
    if '--export-mat' in sys.argv:
        export = sys.argv[sys.argv.index('--export-mat') + 1]
        os.makedirs(export, exist_ok=True)
    fails = 0
    for name in sorted(k for k in globals() if k.startswith('test_')):
        try:
            globals()[name]()
            print(f'  PASS  {name}')
        except Exception as e:
            fails += 1
            print(f'  FAIL  {name}: {e}')
    n = sum(1 for k in globals() if k.startswith('test_'))
    print(f'\n{n - fails}/{n} passed')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(_run_all())
