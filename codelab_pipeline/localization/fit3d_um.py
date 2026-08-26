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
    'offset', 'at_bound', 'n_voxels', 'rss',
    # -- the quantities the acceptance gates are computed FROM, carried
    # on every fit so thresholds can be re-derived post-hoc against real
    # data instead of inherited as constants. See apply_gates. --
    'ci_y_um', 'ci_x_um', 'ci_z_um',   # 95% CI half-widths on position
    'peak_bg_ratio',                   # raw seed voxel / background AT THE SPOT
    'amp_h_ratio',                     # fitted amplitude / raw seed voxel
    'gates_passed'])                   # None when gating was not applied


# How the non-spot signal is modelled. NOT a noise term -- noise is the
# residual and least squares already accounts for it; this is the
# deterministic part of the image that is not the emitter (out-of-focus
# fluorescence from the cell, neighbouring structure, the emitter's own
# axial tail).
#
#   'constant'  one offset. What v1 does. A constant has no degrees of
#               freedom to represent a gradient, so any real trend is
#               left in the residual, where the only parameters able to
#               absorb it are the Gaussian's own centre and sigma.
#   'linear_z'  offset + bz*z. The measured trend here is axial (1.09x
#               to 1.75x end to end); lateral crops are only ~3.5 um
#               wide and show far less.
#   'linear'    offset + by*y + bx*x + bz*z, a tilted plane in 3D.
#
# A low-order background is only honest over a volume where the real
# background IS low-order. Across a whole 110-plane column it is not --
# see fit_radius_um.
BACKGROUNDS = ('constant', 'linear_z', 'linear')
_N_BG = {'constant': 1, 'linear_z': 2, 'linear': 4}


def intensity_centroid(cube, centre_yxz, half_yxz, voxel_um=DEFAULT_VOXEL_UM,
                       floor_quantile=0.5):
    """
    Intensity-weighted centroid within a box, in VOXEL INDICES.

    A better seed than either of the two the pipeline uses today. The
    argmax is one voxel and therefore as noisy as one voxel, and it
    chases whatever is brightest in view; the crop centre is where the
    alignment says the emitter is, which is a prior, not an observation.
    The centroid uses every voxel in the box and is the natural
    first-moment estimate of where the light is.

    Weights are intensity above `floor_quantile` of the box (default the
    median), clipped at zero: without a floor the background -- which
    occupies most of any box -- drags the centroid toward the box centre
    regardless of where the emitter is.

    Returns None when the box holds no signal above the floor.
    """
    cy, cx, cz = centre_yxz
    hy, hx, hz = half_yxz
    ny, nx, nz = cube.shape
    y0, y1 = max(0, int(round(cy - hy))), min(ny, int(round(cy + hy)) + 1)
    x0, x1 = max(0, int(round(cx - hx))), min(nx, int(round(cx + hx)) + 1)
    z0, z1 = max(0, int(round(cz - hz))), min(nz, int(round(cz + hz)) + 1)
    sub = cube[y0:y1, x0:x1, z0:z1]
    if sub.size == 0 or not np.isfinite(sub).any():
        return None
    floor = np.nanquantile(sub, floor_quantile)
    w = np.clip(sub - floor, 0, None)
    w = np.where(np.isfinite(w), w, 0.0)
    total = w.sum()
    if total <= 0:
        return None
    iy, ix, iz = np.indices(sub.shape)
    return (float((w * iy).sum() / total) + y0,
            float((w * ix).sum() / total) + x0,
            float((w * iz).sum() / total) + z0)


def extract_box(cube, centre_yxz, half_yxz):
    """
    A fixed-shape box around `centre_yxz`, NaN-PADDED where it runs off
    the crop rather than clipped.

    Clipping would silently change the box's shape and its centre when
    an emitter sits near the top or bottom of the acquired slab, so two
    hybes' boxes would no longer be the same volume and their fits would
    not be comparable. NaN padding keeps the geometry identical
    everywhere and marks missing data as missing -- which every fit in
    this module already handles, since it masks on isfinite.

    Returns (box, origin_yxz) where origin is the index in `cube` of the
    box's [0,0,0] corner (may be negative, which is the point).
    """
    cy, cx, cz = (int(round(v)) for v in centre_yxz)
    hy, hx, hz = (int(round(v)) for v in half_yxz)
    shape = (2 * hy + 1, 2 * hx + 1, 2 * hz + 1)
    box = np.full(shape, np.nan, dtype=float)
    y0, x0, z0 = cy - hy, cx - hx, cz - hz
    sy0, sy1 = max(0, y0), min(cube.shape[0], y0 + shape[0])
    sx0, sx1 = max(0, x0), min(cube.shape[1], x0 + shape[1])
    sz0, sz1 = max(0, z0), min(cube.shape[2], z0 + shape[2])
    if sy1 > sy0 and sx1 > sx0 and sz1 > sz0:
        box[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0, sz0 - z0:sz1 - z0] = \
            cube[sy0:sy1, sx0:sx1, sz0:sz1]
    return box, (y0, x0, z0)


def _model(p, y, x, z, background='constant'):
    amp, y0, x0, z0, sy, sx, sz = p[:7]
    gauss = amp * np.exp(-(((y - y0) ** 2) / (2 * sy ** 2)
                           + ((x - x0) ** 2) / (2 * sx ** 2)
                           + ((z - z0) ** 2) / (2 * sz ** 2)))
    if background == 'constant':
        return gauss + p[7]
    if background == 'linear_z':
        return gauss + p[7] + p[8] * z
    return gauss + p[7] + p[8] * y + p[9] * x + p[10] * z


def fit_gaussian_3d_um(cubic, y0, x0, z0, voxel_um=DEFAULT_VOXEL_UM,
                       peak_bound_um=0.416, peak_bound_z_um=None,
                       min_sigma_um=0.02,
                       max_sigma_xy_um=0.520, max_sigma_z_um=1.000,
                       min_hb_ratio=1.2, min_ah_ratio=0.25,
                       max_uncert_um=0.416, fit_radius_um=None,
                       background='constant', apply_gates=True):
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

    if background not in BACKGROUNDS:
        raise ValueError(f'background must be one of {BACKGROUNDS}, got {background!r}')
    n_params = 7 + _N_BG[background]
    if values.size <= n_params:
        return None

    amp0, off0 = float(np.nanmax(values)), float(np.nanmin(values))
    s_xy0 = min(0.25, max_sigma_xy_um * 0.6)
    s_z0 = min(0.50, max_sigma_z_um * 0.6)
    p0 = [amp0, y0u, x0u, z0u, s_xy0, s_xy0, s_z0, off0]
    # Lateral and axial position bounds are SEPARATE physical distances.
    # One shared number tied x/y/z together as if a pixel and a plane
    # were the same thing; they are different lengths and the drift
    # along z is far larger, so a bound that is right laterally is wrong
    # axially. Defaults to peak_bound_um when not given, which keeps the
    # previous single-bound behaviour.
    pb_z = peak_bound_um if peak_bound_z_um is None else peak_bound_z_um
    lb = [0.0, y0u - peak_bound_um, x0u - peak_bound_um, z0u - pb_z,
          min_sigma_um, min_sigma_um, min_sigma_um, 0.0]
    ub = [65535.0, y0u + peak_bound_um, x0u + peak_bound_um, z0u + pb_z,
          max_sigma_xy_um, max_sigma_xy_um, max_sigma_z_um, 65535.0]
    names = ['amplitude', 'y', 'x', 'z', 'sigma_y', 'sigma_x', 'sigma_z', 'offset']
    # Background SLOPES are signed and effectively unbounded -- a
    # gradient that had to be positive would be a different assumption
    # about the sample, not a safer one.
    for slope in {'linear_z': ['bg_z'], 'linear': ['bg_y', 'bg_x', 'bg_z']}.get(background, []):
        p0.append(0.0)
        lb.append(-np.inf)
        ub.append(np.inf)
        names.append(slope)
    p0 = [min(max(v, lo), hi) for v, lo, hi in zip(p0, lb, ub)]

    try:
        res = least_squares(lambda p: _model(p, yv, xv, zv, background) - values,
                            p0, bounds=(lb, ub))
    except Exception:
        return None
    if not res.success:
        return None
    amp, fy, fx, fz, sy, sx, sz, off = res.x[:8]

    # which parameters finished ON a constraint
    tol = 1e-9
    at_bound = tuple(n for n, v, lo, hi in zip(names, res.x, lb, ub)
                     if (np.isfinite(lo) and abs(v - lo) <= tol)
                     or (np.isfinite(hi) and abs(v - hi) <= tol))

    dof = values.size - n_params
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
    #
    # apply_gates=False returns the fit WITH its gate quantities instead
    # of rejecting: thresholds inherited from a flat-background,
    # whole-column fit do not carry over to a local background, and
    # re-deriving them means fitting once and sweeping afterwards, not
    # refitting per candidate threshold.
    gate_uncert = not (2 * ci[1] >= max_uncert_um or 2 * ci[2] >= max_uncert_um
                       or 2 * ci[3] >= 2 * max_uncert_um)
    if apply_gates and not gate_uncert:
        return None
    # The peak/background gate must use the background AT THE SPOT, not
    # the intercept: with a slope, `off` is the background extrapolated
    # to (0,0,0) -- a corner of the crop the emitter is nowhere near, and
    # legitimately negative. Using it would change what the gate means
    # the moment the background stops being flat.
    if background == 'constant':
        bg_at_spot = off
    elif background == 'linear_z':
        bg_at_spot = off + res.x[8] * fz
    else:
        bg_at_spot = off + res.x[8] * fy + res.x[9] * fx + res.x[10] * fz
    peak_bg = (h / bg_at_spot) if bg_at_spot > 0 else 0.0
    amp_h = (amp / h) if h > 0 else 0.0
    gate_hb = bg_at_spot > 0 and peak_bg >= min_hb_ratio
    gate_ah = h > 0 and amp_h >= min_ah_ratio
    if apply_gates and not (gate_hb and gate_ah):
        return None

    return FitUm(amplitude=float(amp),
                 y=float(fy / dy), x=float(fx / dx), z=float(fz / dz),
                 y_um=float(fy), x_um=float(fx), z_um=float(fz),
                 sigma_y_um=float(sy), sigma_x_um=float(sx), sigma_z_um=float(sz),
                 offset=float(bg_at_spot), at_bound=at_bound,
                 n_voxels=int(values.size), rss=rss,
                 ci_y_um=float(ci[1]), ci_x_um=float(ci[2]), ci_z_um=float(ci[3]),
                 peak_bg_ratio=float(peak_bg), amp_h_ratio=float(amp_h),
                 gates_passed=(gate_uncert and gate_hb and gate_ah)
                 if apply_gates else None)
