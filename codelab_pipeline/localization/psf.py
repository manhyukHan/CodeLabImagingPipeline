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


def gaussian_halo_psf(dy, dx, dz, s_xy, s_z, halo_frac=0.15, halo_scale=3.0):
    """
    A narrow Gaussian CORE plus a wider Gaussian HALO, both unit-peak, so
    the sum is still 1 at the centre.

    Added because the single-shape candidates measurably cannot do both
    ends at once on this data: fitted to real reference crops, a Gaussian
    matches the core and falls to zero by +/-0.5 um lateral while the
    data still carries 6-15% intensity out to +/-1 um, and a Lorentzian
    tracks those tails but is too narrow in the core. Residual per voxel
    is dominated by the bright core, so the Gaussian wins on score while
    being visibly wrong exactly where a neighbouring emitter would leak
    in -- which is the part of the shape that biases a fit.

    Two extra parameters: what fraction of the peak lives in the halo,
    and how much wider it is than the core.

    halo_frac is bounded BELOW 0.5 by definition, not by taste. Let it
    past a half and the wide component becomes the majority of the
    emitter, at which point core and halo swap roles and the pair is
    unidentifiable: the fit can describe one broad blob either as a core
    of width s, or as a vanishing core plus a halo of width s*scale.
    Measured on real fiducial crops, it took the second route --
    sigma_xy 0.0388 um with halo_scale pinned at 8.0, whose product is
    0.310 um, against 0.312 um from a plain Gaussian on the same data.
    Same answer, degenerate parameterisation, and the residual could not
    tell the difference (rss/vox 2986 vs 2931), so no score comparison
    would have caught it. A calibrated PSF reporting a 39 nm core -- far
    below the diffraction limit -- would then be trusted by everything
    downstream.
    """
    core = gaussian_psf(dy, dx, dz, s_xy, s_z)
    halo = gaussian_psf(dy, dx, dz, s_xy * halo_scale, s_z * halo_scale)
    return (1.0 - halo_frac) * core + halo_frac * halo


# name -> (callable, free shape parameter names, initial guess, bounds)
FAMILIES = {
    'gaussian': (gaussian_psf, ('sigma_xy_um', 'sigma_z_um'),
                 (0.15, 0.40), ((0.03, 1.5), (0.05, 3.0))),
    'moffat': (moffat_psf, ('sigma_xy_um', 'sigma_z_um', 'beta'),
               (0.15, 0.40, 2.5), ((0.03, 1.5), (0.05, 3.0), (0.6, 12.0))),
    'lorentzian': (lorentzian_psf, ('sigma_xy_um', 'sigma_z_um'),
                   (0.15, 0.40), ((0.03, 1.5), (0.05, 3.0))),
    'gaussian_halo': (gaussian_halo_psf,
                      ('sigma_xy_um', 'sigma_z_um', 'halo_frac', 'halo_scale'),
                      (0.13, 0.35, 0.15, 3.0),
                      # halo_frac < 0.5 keeps the core the majority
                      # component, which is what makes the split
                      # identifiable at all (see gaussian_halo_psf)
                      ((0.03, 1.5), (0.05, 3.0), (0.0, 0.45), (1.5, 6.0))),
}


def evaluate(family, shape_params, dy, dx, dz):
    """Unit-peak PSF of `family` at physical offsets (um) from centre."""
    fn = FAMILIES[family][0]
    return fn(dy, dx, dz, *shape_params)


# -- plausibility --------------------------------------------------------

# Smallest lateral sigma any objective in this class can produce. For an
# oil objective around NA 1.4 at ~600 nm emission the diffraction-limited
# Gaussian sigma is about 90 nm; 70 nm leaves margin for a genuinely
# sharper setup while still rejecting the degenerate answers measured
# here. A calibration below this is not a narrow PSF, it is a fit that
# has collapsed its core and let some other component carry the width.
MIN_PLAUSIBLE_SIGMA_XY_UM = 0.070
MIN_PLAUSIBLE_SIGMA_Z_UM = 0.150


def plausible(family, params, tol=1e-6):
    """
    (ok, reasons) for a calibrated shape.

    Exists because a bad calibration is SILENT: measured on real readout
    crops, small samples returned lorentzian with sigma_xy pinned at its
    0.030 um lower bound -- a 30 nm core, well below anything the optics
    can produce -- and the residual could not distinguish it from the
    sane answer. Everything downstream trusts this shape, so it has to
    refuse to hand over one that is physically impossible rather than
    letting it propagate into every fit and every gate threshold.
    """
    reasons = []
    names = FAMILIES[family][1]
    bounds = FAMILIES[family][3]
    for n, (lo, hi) in zip(names, bounds):
        v = params[n]
        if abs(v - lo) <= tol:
            reasons.append(f'{n} at its lower bound ({lo})')
        elif abs(v - hi) <= tol:
            reasons.append(f'{n} at its upper bound ({hi})')
    sxy = params.get('sigma_xy_um')
    if sxy is not None and sxy < MIN_PLAUSIBLE_SIGMA_XY_UM:
        reasons.append(f'sigma_xy {sxy * 1000:.0f} nm is below the optical '
                       f'limit ({MIN_PLAUSIBLE_SIGMA_XY_UM * 1000:.0f} nm)')
    sz = params.get('sigma_z_um')
    if sz is not None and sz < MIN_PLAUSIBLE_SIGMA_Z_UM:
        reasons.append(f'sigma_z {sz * 1000:.0f} nm is below the optical '
                       f'limit ({MIN_PLAUSIBLE_SIGMA_Z_UM * 1000:.0f} nm)')
    return (not reasons), reasons


# -- calibration ---------------------------------------------------------

def _prepare_projections(cube, voxel_um, seed_yxz, half_yxz=(5, 5, 15)):
    """
    A crop reduced to its three 2D MAX PROJECTIONS, with coordinates.

    Exact for this PSF family, not an approximation. Every candidate here
    is separable and co-centred, so

        max_z [ g_xy(y,x) * g_z(z) ] = g_xy(y,x) * 1 = g_xy(y,x)

    because g_z peaks at 0 and 0 is inside the box. gaussian_halo
    survives it too: both of its components peak at the same z, so they
    attain their maxima simultaneously and the projection is the lateral
    profile of the sum. So YX carries sigma_xy, and ZX/ZY carry sigma_z
    (and sigma_xy again, which is a free consistency check the 3D fit
    does not provide).

    Where it is NOT exact is background: max(signal + bg) != max(signal)
    + bg. A background flat in z collapses to a constant the per-
    projection plane absorbs; a strong axial gradient leaves the bright
    end. Measured against the 3D fit on real fiducial crops, sigma_xy
    agreed to 1 nm and sigma_z to 39 nm, so the distortion is small here.

    Returns [(image, coord0, coord1, kinds), ...] or None.
    """
    from .fit3d_um import extract_box
    box, _origin = extract_box(cube, seed_yxz, half_yxz)
    if not np.isfinite(box).any():
        return None
    dy, dx, dz = voxel_um
    ny, nx, nz = box.shape
    y = (np.arange(ny) - (ny - 1) / 2.0) * dy
    x = (np.arange(nx) - (nx - 1) / 2.0) * dx
    z = (np.arange(nz) - (nz - 1) / 2.0) * dz
    with np.errstate(all='ignore'):
        yx = np.nanmax(box, axis=2)      # (y, x)
        zx = np.nanmax(box, axis=0)      # (x, z)
        zy = np.nanmax(box, axis=1)      # (y, z)
    out = []
    for img, a0, a1, kinds in ((yx, y, x, ('y', 'x')),
                               (zx, x, z, ('x', 'z')),
                               (zy, y, z, ('y', 'z'))):
        m = np.isfinite(img)
        if m.sum() < 20:
            continue
        A0, A1 = np.meshgrid(a0, a1, indexing='ij')
        out.append((img[m], A0[m], A1[m], kinds))
    return out or None


def _fit_projections(prep, family, shape):
    """
    All three projections in ONE least_squares call, sharing amplitude
    and a single 3D centre.

    The shared parameterisation is the point. Fitting each projection
    separately needs three solver calls per crop, and at ~800 points
    each the per-call overhead dominates -- measured 0.6x, i.e. SLOWER
    than the single 3D fit it was meant to replace. Sharing (amp, cy,
    cx, cz) across the three cuts it to one call and 13 parameters
    instead of 18, and is also the physically correct statement: one
    emitter, seen three ways. Only the background plane is per-
    projection, since each projection's background is a different
    non-linear reduction of the same 3D one.
    """
    from scipy.optimize import least_squares
    if not prep:
        return None
    n_proj = len(prep)
    peak = max(float(np.nanmax(v) - np.nanmedian(v)) for v, _a, _b, _k in prep)

    def split(p):
        amp, cy, cx, cz = p[:4]
        return amp, {'y': cy, 'x': cx, 'z': cz}, p[4:]

    def resid(p):
        amp, c, bg = split(p)
        parts = []
        for i, (v, A0, A1, kinds) in enumerate(prep):
            d = {'y': 0.0, 'x': 0.0, 'z': 0.0}
            d[kinds[0]] = A0 - c[kinds[0]]
            d[kinds[1]] = A1 - c[kinds[1]]
            s = evaluate(family, shape, d['y'], d['x'], d['z'])
            b0, g0, g1 = bg[3 * i:3 * i + 3]
            parts.append(amp * s + b0 + g0 * A0 + g1 * A1 - v)
        return np.concatenate(parts)

    p0 = [max(peak, 1.0), 0.0, 0.0, 0.0]
    lo = [0.0, -0.5, -0.5, -1.0]
    hi = [np.inf, 0.5, 0.5, 1.0]
    for v, _a, _b, _k in prep:
        p0 += [float(np.nanmedian(v)), 0.0, 0.0]
        lo += [-np.inf] * 3
        hi += [np.inf] * 3
    try:
        r = least_squares(resid, p0, bounds=(lo, hi), max_nfev=300)
    except Exception:
        return None
    if not r.success:
        return None
    return float(np.sum(r.fun ** 2)), int(sum(v.size for v, _a, _b, _k in prep))


def _prepare_crop(cube, voxel_um, fit_radius_um, seed_yxz):
    """
    Everything about ONE crop that does NOT depend on the PSF shape:
    the masked coordinate arrays, the values, and the seed in um.

    Split out because calibration evaluates many candidate shapes
    against the same crops -- Nelder-Mead over 4 families is thousands
    of inner fits -- and rebuilding the index mesh and radius mask each
    time recomputed the identical geometry every call. Measured: that
    waste was about half the total runtime.

    Returns None when the crop has too few usable voxels.
    """
    dy_um, dx_um, dz_um = voxel_um
    iy, ix, iz = np.indices(cube.shape)
    Y, X, Z = iy * dy_um, ix * dx_um, iz * dz_um
    sy, sx, sz = seed_yxz
    y0u, x0u, z0u = sy * dy_um, sx * dx_um, sz * dz_um
    ry, rx, rz = fit_radius_um
    m = (np.isfinite(cube) & (np.abs(Y - y0u) <= ry)
         & (np.abs(X - x0u) <= rx) & (np.abs(Z - z0u) <= rz))
    if m.sum() < 40:
        return None
    vals = cube[m].astype(float)
    return (Y[m], X[m], Z[m], vals, (y0u, x0u, z0u),
            float(max(np.nanmax(vals) - np.nanmedian(vals), 1.0)),
            float(np.nanmedian(vals)))


def _fit_prepared(prep, family, shape):
    """
    Nuisance parameters for one prepared crop with the PSF SHAPE HELD
    FIXED: amplitude, centre and a linear background. Returns (rss, n)
    or None. The shape is what calibration solves for and must be shared
    across crops; everything else is per-spot and is profiled out here.
    """
    from scipy.optimize import least_squares
    yv, xv, zv, vals, (y0u, x0u, z0u), amp0, med = prep

    def resid(p):
        amp, cy, cx, cz, b0, by, bx, bz = p
        s = evaluate(family, shape, yv - cy, xv - cx, zv - cz)
        return amp * s + b0 + by * yv + bx * xv + bz * zv - vals

    p0 = [amp0, y0u, x0u, z0u, med, 0.0, 0.0, 0.0]
    lo = [0, y0u - 0.5, x0u - 0.5, z0u - 1.0, -np.inf, -np.inf, -np.inf, -np.inf]
    hi = [np.inf, y0u + 0.5, x0u + 0.5, z0u + 1.0, np.inf, np.inf, np.inf, np.inf]
    try:
        r = least_squares(resid, p0, bounds=(lo, hi), max_nfev=200)
    except Exception:
        return None
    if not r.success:
        return None
    return float(np.sum(r.fun ** 2)), int(vals.size)


def calibrate(crops, voxel_um=DEFAULT_VOXEL_UM, families=None,
              fit_radius_um=(0.8, 0.8, 2.0), verbose=True, z_centres=None,
              mode='box'):
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
    # geometry is prepared ONCE per crop and reused for every candidate
    # shape and every optimiser iteration (see _prepare_crop)
    preps = []
    for i, c in enumerate(crops):
        # Lateral seed is the crop CENTRE -- where the alignment says the
        # emitter is, which for a crop cut by reference_to_raw is a real
        # prior and not a guess.
        #
        # Axial: prefer a z_centre supplied by the caller, which should
        # come from the SAME consensus placement the fit itself uses. The
        # fallback is this crop's own argmax over the full depth, and
        # that fallback is known to misplace the box -- a pillar argmax
        # chases whatever is brightest in the column, which may be a
        # different object entirely. Calibrating on misplaced boxes would
        # bias the very shape being measured, so callers with a baseline
        # should always pass one.
        if z_centres is not None and z_centres[i] is not None:
            zc = float(np.clip(z_centres[i], 0, c.shape[2] - 1))
        else:
            zc = float(np.unravel_index(int(np.nanargmax(c)), c.shape)[2])
        seed = ((c.shape[0] - 1) / 2.0, (c.shape[1] - 1) / 2.0, zc)
        prep = (_prepare_projections(c, voxel_um, seed) if mode == 'projections'
                else _prepare_crop(c, voxel_um, fit_radius_um, seed))
        if prep is not None:
            preps.append(prep)

    results = {}
    for family in families:
        _fn, names, init, bounds = FAMILIES[family]

        def total(theta):
            rss, n = 0.0, 0
            inner = _fit_projections if mode == 'projections' else _fit_prepared
            for prep in preps:
                got = inner(prep, family, tuple(theta))
                if got is None:
                    continue
                rss += got[0]
                n += got[1]
            return (rss / n) if n else np.inf

        # BOUNDS ARE PASSED. FAMILIES declares them, and this call used to
        # ignore them -- Nelder-Mead without bounds wanders wherever the
        # objective leads. Measured consequence on real crops: the
        # fiducial calibration converged to sigma_xy 0.037 um (37 nm, far
        # below the diffraction limit) with halo_scale 8.36, OUTSIDE the
        # declared upper bound of 8.0. That is the model degenerating --
        # a near-delta core with a huge halo doing all the work -- and it
        # scored slightly WORSE than the sane answer (rss/vox 2986 vs
        # 2931), so a plain score comparison would not have caught it
        # either. A calibrated PSF that is physically impossible is worse
        # than no calibration, because everything downstream trusts it.
        res = minimize(total, np.array(init, dtype=float), method='Nelder-Mead',
                       bounds=bounds,
                       options={'xatol': 1e-3, 'fatol': 1e-2, 'maxiter': 120})
        params = {k: float(v) for k, v in zip(names, res.x)}
        results[family] = {'params': params, 'score': float(res.fun),
                           'n_crops': len(preps)}
        if verbose:
            pretty = '  '.join(f'{k}={v:.4f}' for k, v in params.items())
            print(f'   {family:<11} rss/voxel {res.fun:10.2f}   {pretty}', flush=True)

    return select_best(results, verbose=verbose)


def select_best(results, verbose=True):
    """Mark plausibility and pick the winner, in place.

    Split out of `calibrate` so that a driver which fits the families in
    SEPARATE PROCESSES -- the obvious way to parallelise this, since the
    families are independent -- can recombine the parts through exactly
    this code rather than its own copy of it. Two implementations of
    "which PSF do we believe" would eventually disagree, and the one that
    disagreed silently would be the one in the tool.

    Score alone must not choose: a degenerate shape can score as well as
    a sane one (measured: rss/vox 2986 vs 2931 for a 39 nm core against a
    312 nm one). Prefer the best-scoring PLAUSIBLE candidate, and fall
    back to the best overall only when every candidate is implausible --
    which is itself the signal that this data cannot constrain a PSF and
    needs more crops, flagged rather than hidden.
    """
    for family in list(results):
        if not isinstance(results[family], dict) or 'params' not in results[family]:
            continue
        ok, why = plausible(family, results[family]['params'])
        results[family]['plausible'] = ok
        results[family]['warnings'] = why
    usable = [f for f in results
              if isinstance(results[f], dict) and results[f].get('plausible')]
    if usable:
        best = min(usable, key=lambda f: results[f]['score'])
        results['all_implausible'] = False
    else:
        best = min((f for f in results if isinstance(results[f], dict)
                    and 'score' in results[f]),
                   key=lambda f: results[f]['score'])
        results['all_implausible'] = True
        if verbose:
            print('   WARNING: no candidate PSF is physically plausible; '
                  'this calibration should not be used')
    results['best'] = best
    if verbose and results[best].get('warnings'):
        for w in results[best]['warnings']:
            print(f'   WARNING: {w}')
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
