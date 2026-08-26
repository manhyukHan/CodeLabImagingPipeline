"""
The experiment's own PSF: candidate shapes, calibration, persistence.

WHY CANDIDATES AND NOT A MEASURED PSF
-------------------------------------
The usual way to get an experimental PSF is a bead calibration stack.
There isn't one for this scheme, so the PSF has to be recovered from the
data itself. That is possible here because the reference hybe gives many
independent images of point-like emitters in the SAME optical
configuration, and -- being the reference -- they need no alignment
first, which is what makes them usable as calibration input.

So: propose a small family of physically-parameterised shapes, fit each
one jointly across many reference-frame crops, and keep whichever
describes the data best. The winner is a property of the EXPERIMENT (its
objective, immersion, wavelength, z step), not of any one spot, so it is
calibrated once and then reused for every hybe, channel and allele.

THE FAMILY
----------
All are separable in (lateral, axial) and evaluated on the real
micrometre grid, so their parameters are physical lengths rather than
pixel counts:

  gaussian   the classic approximation. Two parameters. Underestimates
             the tails of a real PSF, which is exactly where a fit picks
             up bias from neighbouring signal.
  moffat     Gaussian core with a power-law tail (beta controls how
             heavy). Standard in astronomy for the same reason it
             matters here: real optics put more energy far from the
             centre than a Gaussian allows.
  lorentzian the heaviest-tailed of the three (Moffat at beta=1).

Deliberately NOT a full Gibson-Lanni / Born-Wolf scalar model: those
need the NA, immersion and coverslip parameters as inputs, and guessing
those to justify a more elaborate PSF would be assuming the answer. The
family above is chosen so its parameters can be RECOVERED from the data,
which is the whole point of calibrating rather than assuming.

WHAT IS SAVED
-------------
analysis/psf.json in the project, beside params.json -- the calibrated
shape belongs to the experiment, the same way the layout and the
modality registry do. It records the family, the fitted parameters, the
voxel size they were fitted at, what they were fitted ON (hybe, channel,
how many crops) and the scores of every candidate, so a later reader can
see what was rejected and not just what won.
"""
import json
import os

import numpy as np

# (dy, dx, dz) micrometres per voxel step -- the measured defaults for
# this scope. Every entry point takes an override.
DEFAULT_VOXEL_UM = (0.208, 0.208, 0.2)

PSF_FILENAME = 'psf.json'


# -- the candidate family ------------------------------------------------

def _r2_lateral(dy, dx, s_xy):
    return (dy ** 2 + dx ** 2) / (s_xy ** 2)


def gaussian_psf(dy, dx, dz, s_xy, s_z):
    """Unit-peak 3D Gaussian at physical offsets from the centre (um)."""
    return np.exp(-0.5 * (_r2_lateral(dy, dx, s_xy) + (dz / s_z) ** 2))


def moffat_psf(dy, dx, dz, s_xy, s_z, beta=2.5):
    """
    Unit-peak Moffat: (1 + r^2/s^2)^-beta, separably in lateral and
    axial. beta -> infinity recovers a Gaussian; small beta means heavy
    tails. Axial and lateral share beta -- one shape parameter is enough
    to test 'are the tails heavier than Gaussian', and two would be
    poorly constrained by this data.
    """
    lat = (1.0 + _r2_lateral(dy, dx, s_xy)) ** (-beta)
    ax = (1.0 + (dz / s_z) ** 2) ** (-beta)
    return lat * ax


def lorentzian_psf(dy, dx, dz, s_xy, s_z):
    """Moffat at beta = 1: the heaviest tail in the family."""
    return moffat_psf(dy, dx, dz, s_xy, s_z, beta=1.0)


# name -> (callable, free shape parameter names, initial guess, bounds)
FAMILIES = {
    'gaussian': (gaussian_psf, ('sigma_xy_um', 'sigma_z_um'),
                 (0.15, 0.40), ((0.03, 1.5), (0.05, 3.0))),
    'moffat': (moffat_psf, ('sigma_xy_um', 'sigma_z_um', 'beta'),
               (0.15, 0.40, 2.5), ((0.03, 1.5), (0.05, 3.0), (0.6, 12.0))),
    'lorentzian': (lorentzian_psf, ('sigma_xy_um', 'sigma_z_um'),
                   (0.15, 0.40), ((0.03, 1.5), (0.05, 3.0))),
}


def evaluate(family, shape_params, dy, dx, dz):
    """Unit-peak PSF of `family` at physical offsets (um) from centre."""
    fn = FAMILIES[family][0]
    return fn(dy, dx, dz, *shape_params)


# -- calibration ---------------------------------------------------------

def _fit_one_crop(cube, family, shape, voxel_um, fit_radius_um, seed_yxz):
    """
    Nuisance parameters for ONE crop with the PSF SHAPE HELD FIXED:
    amplitude, centre and a linear background. Returns (rss, n) or None.

    The shape is what calibration is solving for and must be shared
    across crops; everything else is per-spot and is profiled out here.
    """
    from scipy.optimize import least_squares
    dy_um, dx_um, dz_um = voxel_um
    iy, ix, iz = np.indices(cube.shape)
    Y, X, Z = iy * dy_um, ix * dx_um, iz * dz_um
    sy, sx, sz = seed_yxz
    y0u, x0u, z0u = sy * dy_um, sx * dx_um, sz * dz_um

    m = np.isfinite(cube)
    ry, rx, rz = fit_radius_um
    m = m & (np.abs(Y - y0u) <= ry) & (np.abs(X - x0u) <= rx) & (np.abs(Z - z0u) <= rz)
    yv, xv, zv, vals = Y[m], X[m], Z[m], cube[m].astype(float)
    if vals.size < 40:
        return None

    def resid(p):
        amp, cy, cx, cz, b0, by, bx, bz = p
        psf = evaluate(family, shape, yv - cy, xv - cx, zv - cz)
        return amp * psf + b0 + by * yv + bx * xv + bz * zv - vals

    amp0 = float(np.nanmax(vals) - np.nanmedian(vals))
    p0 = [max(amp0, 1.0), y0u, x0u, z0u, float(np.nanmedian(vals)), 0.0, 0.0, 0.0]
    lo = [0, y0u - 0.5, x0u - 0.5, z0u - 1.0, -np.inf, -np.inf, -np.inf, -np.inf]
    hi = [np.inf, y0u + 0.5, x0u + 0.5, z0u + 1.0, np.inf, np.inf, np.inf, np.inf]
    try:
        r = least_squares(resid, p0, bounds=(lo, hi), max_nfev=300)
    except Exception:
        return None
    if not r.success:
        return None
    return float(np.sum(r.fun ** 2)), int(vals.size)


def calibrate(crops, voxel_um=DEFAULT_VOXEL_UM, families=None,
              fit_radius_um=(0.8, 0.8, 2.0), verbose=True):
    """
    Find the PSF shape that best describes `crops` -- a list of 3D
    (Y, X, Z) arrays from the REFERENCE hybe, which need no alignment.

    The shape is shared across every crop; amplitude, centre and
    background are per-crop nuisances profiled out for each candidate.
    Score is total residual sum of squares per voxel, so families with
    different crop counts stay comparable.

    Returns {family: {'params': {...}, 'score': float, 'n_crops': int}}
    plus a 'best' key. Never raises on a bad crop -- it is skipped.
    """
    from scipy.optimize import minimize
    families = families or list(FAMILIES)
    seeds = []
    for c in crops:
        iy, ix, iz = np.unravel_index(int(np.nanargmax(c)), c.shape)
        # lateral seed is the crop CENTRE (where the alignment says the
        # emitter is); axial seed is the brightest plane, since nothing
        # upstream knows the depth
        seeds.append(((c.shape[0] - 1) / 2.0, (c.shape[1] - 1) / 2.0, float(iz)))

    results = {}
    for family in families:
        _fn, names, init, bounds = FAMILIES[family]

        def total(theta):
            rss, n = 0.0, 0
            for cube, seed in zip(crops, seeds):
                got = _fit_one_crop(cube, family, tuple(theta), voxel_um,
                                    fit_radius_um, seed)
                if got is None:
                    continue
                rss += got[0]
                n += got[1]
            return (rss / n) if n else np.inf

        res = minimize(total, np.array(init, dtype=float), method='Nelder-Mead',
                       options={'xatol': 1e-3, 'fatol': 1e-2, 'maxiter': 120})
        params = {k: float(v) for k, v in zip(names, res.x)}
        results[family] = {'params': params, 'score': float(res.fun),
                           'n_crops': len(crops)}
        if verbose:
            pretty = '  '.join(f'{k}={v:.4f}' for k, v in params.items())
            print(f'   {family:<11} rss/voxel {res.fun:10.2f}   {pretty}', flush=True)

    best = min(results, key=lambda f: results[f]['score'])
    results['best'] = best
    return results


# -- persistence ---------------------------------------------------------

def psf_path(storage_path):
    """<project>/analysis/psf.json -- beside params.json, because a
    calibrated PSF describes the EXPERIMENT, not one modality."""
    from codelab_pipeline.io import paths
    return os.path.join(paths.analysis_dir(storage_path), PSF_FILENAME)


def save(storage_path, family, params, voxel_um, source, scores=None):
    """
    Write the calibrated PSF atomically (.part + os.replace, this
    project's one write pattern). `source` records what it was fitted on
    so a later reader can judge whether it still applies; `scores` keeps
    every candidate's result, so what was REJECTED is visible too.
    """
    target = psf_path(storage_path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    doc = {'family': family, 'params': params,
           'voxel_um': list(voxel_um), 'source': source,
           'candidates': scores or {}}
    tmp = target + '.part'
    with open(tmp, 'w') as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, target)
    return target


def load(storage_path):
    """The calibrated PSF, or None when this project has never been
    calibrated -- absence is not an error, the caller falls back to a
    Gaussian."""
    try:
        with open(psf_path(storage_path)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
