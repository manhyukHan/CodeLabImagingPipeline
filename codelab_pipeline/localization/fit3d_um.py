"""
Version 2 of the 3D Gaussian fit: the crop as an intensity FIELD over
real space, not an array of pixel indices.

v1 (localization.fit_gaussian_3d) is a direct port of ChrTracer3's
FitPsf3D and stays exactly as it is -- it is the reference
implementation. This module is a second engine, free to differ.

WHAT CHANGES
------------
v1 fits in index units, so every bound means something different along
z than along x/y and nothing at all physically: 'sigma <= 2.5' is 2.5
pixels laterally and 2.5 planes axially, two unrelated distances that
happen to share a number. A real PSF is an ellipsoid with a physical
shape -- roughly 0.25 um laterally and 0.6-0.8 um axially -- so its
bounds belong in micrometres.

Here the crop becomes a point cloud: voxel (iy, ix, iz) sits at
(iy*dy, ix*dx, iz*dz) micrometres and carries an intensity. The
Gaussian is evaluated at those real positions and the fit minimises the
same sum of squared deviations, so this is v1's estimator expressed in
a different coordinate system -- not a different objective.

WHAT THAT DOES AND DOES NOT BUY
-------------------------------
Being explicit, because the sampling here is nearly isotropic already:
at the defaults (xy 0.208 um/px, z 0.2 um/plane) a plane and a pixel
differ by 4%, so the GEOMETRIC effect of the recast is small. What it
actually buys is that the bounds become physical and separable -- a
lateral sigma cap and an axial sigma cap can now be set to what a PSF
really is, instead of being tied together as max_sigma and 2*max_sigma.
It also makes the numbers portable: the same config is correct on a
different objective or z-step, which index units never were.

The defaults below are deliberately v1's defaults CONVERTED, not
retuned, so a comparison between the two isolates the coordinate change:

    peak_bound      2.0 px    -> 0.416 um
    max_sigma xy    2.5 px    -> 0.520 um
    max_sigma z     5.0 planes-> 1.000 um
    max_uncert      2.0 px    -> 0.416 um

fit_radius_um is the one genuinely new knob and defaults to OFF, so it
can be measured on its own rather than confounded with the recast.
"""
from collections import namedtuple

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import t as student_t

# (dy, dx, dz) micrometres per voxel step. The measured defaults for
# this scope; every caller may override, and nothing here assumes them.
DEFAULT_VOXEL_UM = (0.208, 0.208, 0.2)

FitUm = namedtuple('FitUm', [
    'amplitude', 'y', 'x', 'z',                      # planes/pixels, for callers
    'y_um', 'x_um', 'z_um',
    'sigma_y_um', 'sigma_x_um', 'sigma_z_um',
    'offset', 'at_bound', 'n_voxels', 'rss'])


def _model(p, y, x, z):
    amp, y0, x0, z0, sy, sx, sz, off = p
    return off + amp * np.exp(-(((y - y0) ** 2) / (2 * sy ** 2)
                                + ((x - x0) ** 2) / (2 * sx ** 2)
                                + ((z - z0) ** 2) / (2 * sz ** 2)))


def fit_gaussian_3d_um(cubic, y0, x0, z0, voxel_um=DEFAULT_VOXEL_UM,
                       peak_bound_um=0.416, min_sigma_um=0.02,
                       max_sigma_xy_um=0.520, max_sigma_z_um=1.000,
                       min_hb_ratio=1.2, min_ah_ratio=0.25,
                       max_uncert_um=0.416, fit_radius_um=None):
    """
    Fit one 3D Gaussian to `cubic` in micrometres.

    cubic: (height, width, depth) = (Y, X, Z), this project's standard
    crop layout. y0/x0/z0: the seed, in VOXEL INDICES (the units every
    caller already has); converted internally.

    fit_radius_um: when set, only voxels within this distance of the
    seed take part in the fit -- (ry, rx, rz) for an anisotropic
    neighbourhood, or a scalar for a sphere. None (default) fits every
    voxel in the crop, which is what v1 does.

    Returns a FitUm, or None for a rejected/failed fit -- never raises,
    matching v1's "absence is not an error" contract. `at_bound` names
    every parameter that finished ON its constraint instead of at an
    interior optimum; such a value is the bound you supplied, not a
    measurement, and its confidence interval does not describe it.
    """
    dy, dx, dz = (float(v) for v in voxel_um)
    ny, nx, nz = cubic.shape
    iy, ix, iz = np.indices(cubic.shape)
    mask = np.isfinite(cubic)

    # the point cloud: real position + intensity
    Y, X, Z = iy * dy, ix * dx, iz * dz
    y0u, x0u, z0u = y0 * dy, x0 * dx, z0 * dz

    if fit_radius_um is not None:
        r = fit_radius_um
        ry, rx, rz = (r, r, r) if np.isscalar(r) else (float(r[0]), float(r[1]), float(r[2]))
        mask = mask & (np.abs(Y - y0u) <= ry) & (np.abs(X - x0u) <= rx) & (np.abs(Z - z0u) <= rz)

    yv, xv, zv = Y[mask], X[mask], Z[mask]
    values = cubic[mask].astype(float)
    if values.size <= 8:
        return None

    # raw intensity at the seed voxel, for the same two gates v1 applies
    jy, jx, jz = int(round(y0)), int(round(x0)), int(round(z0))
    if 0 <= jy < ny and 0 <= jx < nx and 0 <= jz < nz and np.isfinite(cubic[jy, jx, jz]):
        h = float(cubic[jy, jx, jz])
    else:
        h = float(np.nanmax(values))

    amp0, off0 = float(np.nanmax(values)), float(np.nanmin(values))
    s_xy0 = min(0.25, max_sigma_xy_um * 0.6)
    s_z0 = min(0.50, max_sigma_z_um * 0.6)
    p0 = [amp0, y0u, x0u, z0u, s_xy0, s_xy0, s_z0, off0]
    lb = [0.0, y0u - peak_bound_um, x0u - peak_bound_um, z0u - peak_bound_um,
          min_sigma_um, min_sigma_um, min_sigma_um, 0.0]
    ub = [65535.0, y0u + peak_bound_um, x0u + peak_bound_um, z0u + peak_bound_um,
          max_sigma_xy_um, max_sigma_xy_um, max_sigma_z_um, 65535.0]
    p0 = [min(max(v, lo), hi) for v, lo, hi in zip(p0, lb, ub)]

    try:
        res = least_squares(lambda p: _model(p, yv, xv, zv) - values,
                            p0, bounds=(lb, ub))
    except Exception:
        return None
    if not res.success:
        return None
    amp, fy, fx, fz, sy, sx, sz, off = res.x

    # which parameters finished ON a constraint
    names = ('amplitude', 'y', 'x', 'z', 'sigma_y', 'sigma_x', 'sigma_z', 'offset')
    tol = 1e-9
    at_bound = tuple(n for n, v, lo, hi in zip(names, res.x, lb, ub)
                     if abs(v - lo) <= tol or abs(v - hi) <= tol)

    dof = values.size - 8
    if dof <= 0:
        return None
    rss = float(np.sum(res.fun ** 2))
    try:
        cov = (rss / dof) * np.linalg.pinv(res.jac.T @ res.jac)
        se = np.sqrt(np.diag(cov))
    except Exception:
        return None
    if not np.all(np.isfinite(se)):
        return None
    ci = student_t.ppf(0.975, dof) * se

    # Same three gates as v1, with the position CI now in micrometres so
    # one number bounds all three axes honestly instead of x/y sharing a
    # pixel bound and z silently getting twice it.
    if (2 * ci[1] >= max_uncert_um or 2 * ci[2] >= max_uncert_um
            or 2 * ci[3] >= 2 * max_uncert_um):
        return None
    if off <= 0 or h / off < min_hb_ratio:
        return None
    if h <= 0 or amp / h < min_ah_ratio:
        return None

    return FitUm(amplitude=float(amp),
                 y=float(fy / dy), x=float(fx / dx), z=float(fz / dz),
                 y_um=float(fy), x_um=float(fx), z_um=float(fz),
                 sigma_y_um=float(sy), sigma_x_um=float(sx), sigma_z_um=float(sz),
                 offset=float(off), at_bound=at_bound,
                 n_voxels=int(values.size), rss=rss)
