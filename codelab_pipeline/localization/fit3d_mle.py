"""
v2 estimator: Poisson maximum likelihood, on a calibrated PSF.

TWO CHANGES FROM fit3d_um, EACH FOR ITS OWN REASON
--------------------------------------------------
1. NOISE MODEL. Least squares is the maximum-likelihood estimator when
   the noise is Gaussian with CONSTANT variance. Photon counting is not:
   the variance of a pixel equals its mean, so bright pixels are noisier
   in absolute terms and dim ones are more informative than least
   squares believes. Weighting every voxel equally therefore throws away
   information, and provably fails to reach the Cramer-Rao bound that
   Poisson MLE attains. This matters most exactly where the tracing is
   hardest -- dim readout spots.

   Implemented through POISSON DEVIANCE RESIDUALS rather than a raw
   likelihood call:

       r_i = sign(y_i - mu_i) * sqrt(2 * (y_i*log(y_i/mu_i) - (y_i - mu_i)))

   Sum of r_i^2 IS the deviance, so minimising it maximises the Poisson
   likelihood exactly -- while still being a least_squares problem, which
   keeps the same bounded solver, the same Jacobian, and therefore the
   same confidence intervals the gates are computed from. Switching the
   noise model does not mean rewriting everything downstream.

2. PSF SHAPE. The emitter shape comes from psf.py's calibration for this
   experiment (family + parameters, fitted once on the reference hybe)
   instead of being a Gaussian with two free sigmas per spot. A shape
   that is CALIBRATED rather than re-fitted per spot removes two free
   parameters from every fit, which is what stops a fit from explaining
   neighbouring signal by inflating its own width -- the failure measured
   across this bench, where sigma sat on its bound in ~100% of fits.

   free_shape=True keeps them free, for comparison against the
   calibrated shape rather than instead of it.

CAMERA UNITS
------------
Poisson statistics apply to PHOTOELECTRONS, not to camera ADU. counts =
(adu - offset) * gain. Defaults are gain 1.0 / offset 0.0, i.e. "ADU are
already proportional to counts", which is the honest neutral assumption
when the camera has not been characterised -- it makes the weighting
correct up to a scale factor, which is all the estimator needs.
"""
from collections import namedtuple

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import t as student_t

from . import psf as psf_mod

DEFAULT_VOXEL_UM = psf_mod.DEFAULT_VOXEL_UM

FitMLE = namedtuple('FitMLE', [
    'amplitude', 'y', 'x', 'z',
    'y_um', 'x_um', 'z_um',
    'sigma_xy_um', 'sigma_z_um',
    'background', 'at_bound', 'n_voxels',
    'deviance', 'chi2_per_dof',
    'ci_y_um', 'ci_x_um', 'ci_z_um',
    'peak_bg_ratio', 'amp_h_ratio',
    'family', 'noise'])

_FLOOR = 1e-6      # mu must stay positive for log(); below one photon


def _deviance_residuals(mu, y):
    """
    Signed sqrt of each voxel's contribution to the Poisson deviance.
    y*log(y/mu) is taken as 0 where y == 0, which is its limit and the
    standard convention; without it every empty voxel would be a NaN.
    """
    mu = np.maximum(mu, _FLOOR)
    with np.errstate(divide='ignore', invalid='ignore'):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    d = 2.0 * (term - (y - mu))
    return np.sign(y - mu) * np.sqrt(np.maximum(d, 0.0))


def fit_gaussian_3d_mle(cubic, y0, x0, z0, voxel_um=DEFAULT_VOXEL_UM,
                        family='gaussian', shape_params=(0.15, 0.40),
                        free_shape=False, noise='poisson',
                        peak_bound_um=0.416, peak_bound_z_um=None,
                        fit_radius_um=(1.0, 1.0, 3.0),
                        background='linear',
                        camera_gain=1.0, camera_offset=0.0,
                        min_hb_ratio=None, min_ah_ratio=None,
                        max_uncert_um=None, apply_gates=False):
    """
    One emitter, Poisson-MLE fitted with a calibrated PSF shape.

    cubic: (Y, X, Z) crop. y0/x0/z0: seed in VOXEL INDICES.
    shape_params: the family's parameters, in micrometres, from
    psf.calibrate. free_shape=True re-fits them per spot instead.

    Gate thresholds default to None = do not gate: the v1 constants do
    not transfer to a local background (see tools/fit_gates.py), so this
    estimator declines to inherit them silently. Pass them explicitly,
    or gate downstream on the returned quantities.

    Returns FitMLE or None; never raises.
    """
    dy, dx, dz = (float(v) for v in voxel_um)
    ny, nx, nz = cubic.shape
    iy, ix, iz = np.indices(cubic.shape)
    Y, X, Z = iy * dy, ix * dx, iz * dz
    y0u, x0u, z0u = y0 * dy, x0 * dx, z0 * dz

    counts = (np.asarray(cubic, dtype=float) - camera_offset) * camera_gain
    mask = np.isfinite(counts)
    if fit_radius_um is not None:
        ry, rx, rz = fit_radius_um
        mask = mask & (np.abs(Y - y0u) <= ry) & (np.abs(X - x0u) <= rx) \
            & (np.abs(Z - z0u) <= rz)
    yv, xv, zv, vals = Y[mask], X[mask], Z[mask], counts[mask]
    if vals.size < 40:
        return None

    jy, jx, jz = int(round(y0)), int(round(x0)), int(round(z0))
    if 0 <= jy < ny and 0 <= jx < nx and 0 <= jz < nz and np.isfinite(counts[jy, jx, jz]):
        h = float(counts[jy, jx, jz])
    else:
        h = float(np.nanmax(vals))

    n_shape = len(shape_params) if free_shape else 0
    n_bg = 4 if background == 'linear' else 1

    def unpack(p):
        amp, cy, cx, cz = p[0], p[1], p[2], p[3]
        shape = tuple(p[4:4 + n_shape]) if free_shape else tuple(shape_params)
        bg = p[4 + n_shape:]
        return amp, cy, cx, cz, shape, bg

    def model(p):
        amp, cy, cx, cz, shape, bg = unpack(p)
        s = psf_mod.evaluate(family, shape, yv - cy, xv - cx, zv - cz)
        base = bg[0] + (bg[1] * yv + bg[2] * xv + bg[3] * zv if n_bg == 4 else 0.0)
        return amp * s + base

    def resid(p):
        mu = model(p)
        if noise == 'poisson':
            return _deviance_residuals(mu, vals)
        return mu - vals

    amp0 = float(max(np.nanmax(vals) - np.nanmedian(vals), 1.0))
    # lateral and axial position bounds are separate physical lengths
    pb_z = peak_bound_um if peak_bound_z_um is None else peak_bound_z_um
    p0 = [amp0, y0u, x0u, z0u]
    lb = [0.0, y0u - peak_bound_um, x0u - peak_bound_um, z0u - pb_z]
    ub = [np.inf, y0u + peak_bound_um, x0u + peak_bound_um, z0u + pb_z]
    names = ['amplitude', 'y', 'x', 'z']
    if free_shape:
        for nm, v, (lo, hi) in zip(psf_mod.FAMILIES[family][1], shape_params,
                                   psf_mod.FAMILIES[family][3]):
            p0.append(float(v))
            lb.append(lo)
            ub.append(hi)
            names.append(nm)
    p0.append(float(np.nanmedian(vals)))
    lb.append(-np.inf)
    ub.append(np.inf)
    names.append('bg0')
    for nm in (['bg_y', 'bg_x', 'bg_z'] if n_bg == 4 else []):
        p0.append(0.0)
        lb.append(-np.inf)
        ub.append(np.inf)
        names.append(nm)

    try:
        res = least_squares(resid, p0, bounds=(lb, ub), max_nfev=600)
    except Exception:
        return None
    if not res.success:
        return None

    amp, cy, cx, cz, shape, bg = unpack(res.x)
    tol = 1e-9
    at_bound = tuple(n for n, v, lo, hi in zip(names, res.x, lb, ub)
                     if (np.isfinite(lo) and abs(v - lo) <= tol)
                     or (np.isfinite(hi) and abs(v - hi) <= tol))

    dof = vals.size - len(p0)
    if dof <= 0:
        return None
    deviance = float(np.sum(res.fun ** 2))
    try:
        cov = (deviance / dof) * np.linalg.pinv(res.jac.T @ res.jac)
        se = np.sqrt(np.diag(cov))
    except Exception:
        return None
    if not np.all(np.isfinite(se[:4])):
        return None
    ci = student_t.ppf(0.975, dof) * se

    bg_at_spot = bg[0] + (bg[1] * cy + bg[2] * cx + bg[3] * cz if n_bg == 4 else 0.0)
    peak_bg = (h / bg_at_spot) if bg_at_spot > 0 else 0.0
    amp_h = (amp / h) if h > 0 else 0.0

    if apply_gates:
        if max_uncert_um is not None and (2 * ci[1] >= max_uncert_um
                                          or 2 * ci[2] >= max_uncert_um
                                          or 2 * ci[3] >= 2 * max_uncert_um):
            return None
        if min_hb_ratio is not None and peak_bg < min_hb_ratio:
            return None
        if min_ah_ratio is not None and amp_h < min_ah_ratio:
            return None

    s_xy = shape[0]
    s_z = shape[1] if len(shape) > 1 else shape[0]
    return FitMLE(amplitude=float(amp),
                  y=float(cy / dy), x=float(cx / dx), z=float(cz / dz),
                  y_um=float(cy), x_um=float(cx), z_um=float(cz),
                  sigma_xy_um=float(s_xy), sigma_z_um=float(s_z),
                  background=float(bg_at_spot), at_bound=at_bound,
                  n_voxels=int(vals.size), deviance=deviance,
                  chi2_per_dof=float(deviance / dof),
                  ci_y_um=float(ci[1]), ci_x_um=float(ci[2]), ci_z_um=float(ci[3]),
                  peak_bg_ratio=float(peak_bg), amp_h_ratio=float(amp_h),
                  family=family, noise=noise)
