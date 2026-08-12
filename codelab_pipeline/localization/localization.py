import os
import numpy as np
import numpy.linalg as la
from scipy.optimize import minimize, least_squares
from scipy.stats import t as student_t
from skimage.feature import peak_local_max
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py

from ..io import preprocess
import cv2

from ..alignment import chain as alignment
from ..alignment import spot_mapper

def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo, yo = float(xo), float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    return offset + amplitude * np.exp(- (a*(x-xo)**2 + 2*b*(x-xo)*(y-yo) + c*(y-yo)**2))

def cost_function(params, x, y, z):
    amp, xo, yo, sigma_x, sigma_y, theta, offset = params
    model = gaussian_2d((x, y), amp, xo, yo, sigma_x, sigma_y, theta, offset)
    residuals = (z - model)**2
    return np.nansum(residuals) 

def fit_gaussian_2d(img, x0, y0,):
    x, y = np.arange(img.shape[1]), np.arange(img.shape[0])
    x, y = np.meshgrid(x, y)
    mask = ~np.isnan(img)
    x, y, z = x[mask], y[mask], img[mask]
    initial_guess = (img[mask].max(), x0, y0, 1, 1, 0, img[mask].min())
    result = minimize(cost_function, initial_guess, args=(x, y, z), method='Powell')
    if result.success:
        return result.x
    else:
        return None

def gaussian_3d(xyz, amplitude, x0, y0, z0, sigma_x, sigma_y, sigma_z, offset):
    x, y, z = xyz
    return offset + amplitude * np.exp(
        -(((x - x0)**2) / (2 * sigma_x**2)
          + ((y - y0)**2) / (2 * sigma_y**2)
          + ((z - z0)**2) / (2 * sigma_z**2))
    )

def _residuals_3d(params, x, y, z, values):
    amp, x0, y0, z0, sigma_x, sigma_y, sigma_z, offset = params
    return gaussian_3d((x, y, z), amp, x0, y0, z0, sigma_x, sigma_y, sigma_z, offset) - values

def fit_gaussian_3d(cubic, x0, y0, z0, peak_bound=2.0, init_sigma_xy=1.25, init_sigma_z=2.5,
                    min_sigma=0.1, max_sigma=2.5, min_hb_ratio=1.15, min_ah_ratio=0.15, max_uncert=2.0):
    """
    Bounded least-squares 3D Gaussian fit, matching ChrTracer3's FitPsf3D
    (/Users/hanmanhyuk/Downloads/ChrTracingLib/FitPsf3D.m) bounds and
    rejection criteria -- standard Gaussian normalization though (matching
    gaussian_2d/gaussian_3d's own convention elsewhere in this module),
    not FitPsf3D's non-standard 2*sigma-scaled form.

    cubic: (height, width, depth) i.e. (Y, X, Z) axis order -- this
    project's standard layout everywhere a 3D crop is built
    (localization._build_cell_crop's 'stacks', spot_mapper.
    crop_for_localization's use_stack=True return). x0/y0/z0 (the seed)
    are real pixel coordinates in that same frame. This must stay
    consistent: an earlier version built its index mesh assuming a (Z,Y,X)
    layout (a stale "zyx" naming inherited from before this function took
    real crops), which silently mismatched every actual caller's data --
    the bounds (seed +/- a few px) ended up constraining position along
    the WRONG axis (e.g. the x0 bound compared against a mesh that was
    really the depth axis, spanning the crop's full ~100+ planes), making
    the true optimum geometrically unreachable. The fit then just
    returned the initial guess unchanged every time (zero shift, sigma
    frozen at the init values) while independently miscomputing its own
    accept/reject peak height too (its seed-pixel index lookup used the
    same wrong axes, went out of bounds, and silently fell back to the
    crop's global max instead) -- rejecting almost nothing regardless of
    real signal quality. Confirmed on real data (10 real spots, every one
    landing exactly on its seed) before being traced to this.

    Peak position is bounded to the seed +/- peak_bound (default 2px, a
    real fit can't wander far from where a peak was already detected);
    sigma is bounded to [min_sigma, max_sigma] in xy (2x max_sigma for z,
    since z-PSF is wider); amplitude/offset are bounded to [0, 65535]
    (this project's real uint16 camera range). initial_guess seeds
    sigma_x/y/z at init_sigma_xy/init_sigma_z rather than 1px flat --
    matches FitPsf3D's own initSigmaXY/initSigmaZ defaults, a better
    starting point for a real PSF than an arbitrary 1px guess.

    Rejects (returns None) exactly like FitPsf3D's own filterSpot, computed
    from the fit's residual Jacobian -- 95% CI half-width via the
    t-distribution (same construction as MATLAB's nlparci):
    - the CI on x0 or y0 is wider than max_uncert px, or on z0 wider than
      2*max_uncert px (z is coarser-sampled, same 2x factor FitPsf3D uses)
    - peak/background ratio (raw pixel value at the seed / fitted offset)
      is below min_hb_ratio
    - fitted amplitude / that same raw seed pixel value is below min_ah_ratio
    Never raises -- a rejected or failed fit is a graceful "no real spot
    here," per this project's "no no-alignment error" pattern; the caller
    treats it exactly like "no candidate found," not an error.
    """
    y, x, z = np.indices(cubic.shape)  # axis0=Y, axis1=X, axis2=Z -- see cubic's own docstring above
    mask = np.isfinite(cubic)
    x, y, z, values = x[mask].astype(float), y[mask].astype(float), z[mask].astype(float), cubic[mask].astype(float)

    n_params = 8
    if len(values) <= n_params:
        return None  # not enough data to constrain the fit at all

    iy, ix, iz = int(round(y0)), int(round(x0)), int(round(z0))
    if 0 <= iy < cubic.shape[0] and 0 <= ix < cubic.shape[1] and 0 <= iz < cubic.shape[2] and np.isfinite(cubic[iy, ix, iz]):
        h = float(cubic[iy, ix, iz])
    else:
        h = float(np.nanmax(values))

    amp0 = float(np.nanmax(values))
    offset0 = float(np.nanmin(values))
    p0 = [amp0, x0, y0, z0, init_sigma_xy, init_sigma_xy, init_sigma_z, offset0]
    lb = [0, x0 - peak_bound, y0 - peak_bound, z0 - peak_bound, min_sigma, min_sigma, min_sigma, 0]
    ub = [65535, x0 + peak_bound, y0 + peak_bound, z0 + peak_bound, max_sigma, max_sigma, 2 * max_sigma, 65535]

    try:
        result = least_squares(_residuals_3d, p0, bounds=(lb, ub), args=(x, y, z, values))
    except Exception:
        return None
    if not result.success:
        return None

    amp, x0f, y0f, z0f, sigma_x, sigma_y, sigma_z, offset = result.x

    dof = len(values) - n_params
    residual_var = float(np.sum(result.fun ** 2)) / dof
    try:
        cov = residual_var * np.linalg.pinv(result.jac.T @ result.jac)
        se = np.sqrt(np.diag(cov))
    except Exception:
        return None
    if not np.all(np.isfinite(se)):
        return None
    ci_half = student_t.ppf(0.975, dof) * se  # order matches p0: amp,x0,y0,z0,sx,sy,sz,offset

    if (2 * ci_half[1] >= max_uncert) or (2 * ci_half[2] >= max_uncert) or (2 * ci_half[3] >= 2 * max_uncert):
        return None
    if offset <= 0 or h / offset < min_hb_ratio:
        return None
    if h <= 0 or amp / h < min_ah_ratio:
        return None

    return amp, x0f, y0f, z0f, sigma_x, sigma_y, sigma_z, offset

def find_local_peaks_3d(cubic, min_sep=3.0, threshold_rel=0.3, max_peaks=3):
    """
    Finds up to max_peaks local-maxima seeds within an already-cropped spot
    window -- the FindPeaks3D-equivalent piece of ChrTracer3's own pipeline
    (/Users/hanmanhyuk/Downloads/ChrTracingLib/FindPeaks3D.m: imregionalmax
    + bwconncomp + minSep merge), scoped here to a single spot's own crop
    rather than a whole FOV, so refine_spot_z can tell "one real blob" from
    "two blobs a single click landed between" before committing to a
    single-Gaussian fit. min_sep mirrors FindPeaks3D's own default (3px,
    peaks closer than this are suppressed to one); threshold_rel keeps
    only maxima within threshold_rel of the crop's own brightest voxel, so
    background noise never counts as a second component. skimage's
    peak_local_max already performs the regional-max + min-distance
    suppression FindPeaks3D built by hand from imregionalmax/bwconncomp --
    no separate connected-component step needed here.

    Returns [(x, y, z), ...] in this module's (row=Y, col=X, depth=Z)
    convention, brightest first, capped at max_peaks. Empty only when
    cubic has no finite voxel at all -- callers still supply their own
    fallback (x0, y0, z0) for the single-component case.
    """
    finite = np.isfinite(cubic)
    if not finite.any():
        return []
    safe = np.where(finite, cubic, np.nanmin(cubic[finite])).astype(np.float64)
    coords = peak_local_max(safe, min_distance=max(1, int(round(min_sep))),
                            threshold_rel=threshold_rel, exclude_border=False,
                            num_peaks=max_peaks)
    peaks = [(float(x), float(y), float(z)) for y, x, z in coords]
    peaks.sort(key=lambda p: -safe[int(round(p[1])), int(round(p[0])), int(round(p[2]))])
    return peaks

def _residuals_3d_mixture(params, x, y, z, values, n_components):
    offset = params[-1]
    model = np.full(values.shape, offset)
    for i in range(n_components):
        amp, x0, y0, z0, sx, sy, sz = params[i * 7:(i + 1) * 7]
        model = model + amp * np.exp(
            -(((x - x0) ** 2) / (2 * sx ** 2)
              + ((y - y0) ** 2) / (2 * sy ** 2)
              + ((z - z0) ** 2) / (2 * sz ** 2)))
    return model - values

def fit_gaussian_mixture_3d(cubic, seeds, peak_bound=2.0, init_sigma_xy=1.25, init_sigma_z=2.5,
                            min_sigma=0.1, max_sigma=2.5, min_hb_ratio=1.15, min_ah_ratio=0.15,
                            max_uncert=2.0):
    """
    Sum-of-N-Gaussians counterpart to fit_gaussian_3d, one amplitude/
    position/sigma set per entry in `seeds` ((x0,y0,z0), ..., see
    find_local_peaks_3d) sharing ONE background offset -- for a crop that
    genuinely contains more than one real PSF, e.g. two spots a single
    click (or auto-detect) landed between. Same per-axis bounds as
    fit_gaussian_3d, applied per component around ITS OWN seed; accept/
    reject is also evaluated per component (a crowded crop can have one
    solid blob and one marginal one), against that component's own seed
    pixel and the ONE shared fitted offset.

    Returns a list, same order/length as `seeds`, of either a
    fit_gaussian_3d-shaped tuple (amp, x, y, z, sx, sy, sz, offset) for an
    accepted component or None for a rejected one. Never raises; returns
    [None] * len(seeds) if the joint fit itself fails to converge -- every
    component is only as good as the shared fit that found it.
    """
    n = len(seeds)
    y, x, z = np.indices(cubic.shape)
    mask = np.isfinite(cubic)
    x, y, z, values = x[mask].astype(float), y[mask].astype(float), z[mask].astype(float), cubic[mask].astype(float)

    n_params = 7 * n + 1
    if len(values) <= n_params:
        return [None] * n

    # cubic.shape is (Y, X, Z) -- see this function's own docstring/
    # fit_gaussian_3d's. height/width/depth here clip each component's own
    # +/-peak_bound search window to the crop's real extent, per explicit
    # request: a seed near the crop's edge previously let lb/ub run past
    # [0, shape) on that side, so least_squares was free to return a
    # centroid physically outside the cropped box -- not a plausible real
    # position, just an unconstrained edge of the search space. Clipped
    # here (constraining the fit itself), not rejected after the fact,
    # since a bound that already can't be violated is strictly better than
    # catching it post-hoc.
    height, width, depth = cubic.shape
    offset0 = float(np.nanmin(values))
    p0, lb, ub, seed_heights = [], [], [], []
    for (sx0, sy0, sz0) in seeds:
        iy, ix, iz = int(round(sy0)), int(round(sx0)), int(round(sz0))
        if 0 <= iy < cubic.shape[0] and 0 <= ix < cubic.shape[1] and 0 <= iz < cubic.shape[2] and np.isfinite(cubic[iy, ix, iz]):
            h = float(cubic[iy, ix, iz])
        else:
            h = float(np.nanmax(values))
        seed_heights.append(h)
        p0 += [h, sx0, sy0, sz0, init_sigma_xy, init_sigma_xy, init_sigma_z]
        lb += [0, max(sx0 - peak_bound, 0), max(sy0 - peak_bound, 0), max(sz0 - peak_bound, 0),
               min_sigma, min_sigma, min_sigma]
        ub += [65535, min(sx0 + peak_bound, width - 1), min(sy0 + peak_bound, height - 1), min(sz0 + peak_bound, depth - 1),
               max_sigma, max_sigma, 2 * max_sigma]
    p0.append(offset0)
    lb.append(0)
    ub.append(65535)

    try:
        result = least_squares(_residuals_3d_mixture, p0, bounds=(lb, ub), args=(x, y, z, values, n))
    except Exception:
        return [None] * n
    if not result.success:
        return [None] * n

    dof = len(values) - n_params
    if dof <= 0:
        return [None] * n
    residual_var = float(np.sum(result.fun ** 2)) / dof
    try:
        cov = residual_var * np.linalg.pinv(result.jac.T @ result.jac)
        se = np.sqrt(np.diag(cov))
    except Exception:
        return [None] * n
    if not np.all(np.isfinite(se)):
        return [None] * n
    ci_half = student_t.ppf(0.975, dof) * se

    offset = result.x[-1]
    out = []
    for i in range(n):
        amp, x0f, y0f, z0f, sigma_x, sigma_y, sigma_z = result.x[i * 7:(i + 1) * 7]
        cx, cy, cz = ci_half[i * 7 + 1], ci_half[i * 7 + 2], ci_half[i * 7 + 3]
        h = seed_heights[i]
        if (2 * cx >= max_uncert) or (2 * cy >= max_uncert) or (2 * cz >= 2 * max_uncert):
            out.append(None)
            continue
        if offset <= 0 or h / offset < min_hb_ratio:
            out.append(None)
            continue
        if h <= 0 or amp / h < min_ah_ratio:
            out.append(None)
            continue
        out.append((amp, x0f, y0f, z0f, sigma_x, sigma_y, sigma_z, offset))
    return out

def localize_spots_worker(fov, hybe, hybe_list, cell_parameter_dict,
                           max_to_background, max_to_average, frac, absolute_threshold, cpad, spad, min_distance, channel, including_z,
                           max_num_alleles, max_sigma, max_deviation, h5_save_path):
    cid = cell_parameter_dict['cid']
    cb = cell_parameter_dict['cb']
    matrix_yx = cell_parameter_dict['matrix_yx']
    matrix_zx = cell_parameter_dict['matrix_zx']
    cy,cx = cell_parameter_dict['cy'], cell_parameter_dict['cx']
    xyzs = []
    xyzs_crop = []
    with h5py.File(os.path.join(h5_save_path, f'FOV{fov:02d}', f'{hybe}_stack.h5'), 'r') as f:
        depth, height, width = f[f'stack/ch{channel}'].shape
        H1 = f[f'/matrix/{hybe}'][:]
        if including_z:
            H2 = matrix_yx
            H3 = matrix_zx
            H2[0,2] -= H3[0,2]
        else:
            H2 = matrix_yx
        H = H2@H1
        Hinv =  la.inv(H)
        ry,rx = alignment.align_cell((cy, cx), Hinv, (height, width))
        if len(ry) == 0: return np.zeros((0,7), dtype=float), None, hybe

        rymin,rymax,rxmin,rxmax = max(0, ry.min()-cpad), min(height, ry.max()+cpad+1), max(0, rx.min()-cpad), min(width, rx.max()+cpad+1)
        img = np.full((rymax-rymin, rxmax-rxmin), np.nan, dtype=float)
        img[ry-rymin,rx-rxmin] = f[f'/mip/ch{channel}'][:][ry,rx]
        
        stacks = np.full((depth, rymax-rymin, rxmax-rxmin), np.nan, dtype=float)
        stacks_value = f[f'/stack/ch{channel}'][:,rymin:rymax,rxmin:rxmax]
        stacks[:,ry-rymin,rx-rxmin] = stacks_value[:,ry-rymin,rx-rxmin]
        bimg = np.nanmax(stacks,axis=1)

        v,c = np.unique(img[~np.isnan(img)], return_counts=True)
        cutoff = max_to_background * v[c.argmax()]
        yx = peak_local_max(img, min_distance=min_distance,
                            exclude_border=1,threshold_abs=cutoff)
        brightness = img[yx[:,0], yx[:,1]]
        for j in brightness.argsort()[::-1][:max_num_alleles]:
            if (brightness[j] < brightness.max() * frac) or (brightness[j] < absolute_threshold): continue
            y,x = yx[j]
            if including_z:
                zs = peak_local_max(bimg[:,x], exclude_border=1, threshold_abs=max(brightness[j]*.9,cutoff,absolute_threshold))
            else:
                zs = [np.nan]
            if len(zs) == 0: continue
            z = zs[0]
            x1,y1,_ = H@np.array([x+rxmin,y+rymin,1]).reshape(3,1)
            xyzs.append([fov,cid,hybe_list.index(hybe),cb,float(x1),float(y1),float(z+H3[1,2])])
            xyzs_crop.append([fov,cid,hybe_list.index(hybe),cb,float(x),float(y),float(z)])
                    
            """
            symin,symax,sxmin,sxmax = max(0, y-spad), min(img.shape[0], y+spad+1), max(0, x-spad), min(img.shape[1], x+spad+1)
            params2d = fit_gaussian_2d(img[symin:symax,sxmin:sxmax], x-sxmin, y-symin)
            if params2d is not None:
                amp, x0, y0, sigma_x, sigma_y, theta, offset = params2d
                sigma = abs(sigma_x*sigma_y)**0.5
                if (abs(sigma_x) > .5 and abs(sigma_y) > .5 and sigma < max_sigma and 
                        ((x0+sxmin - x)**2 + (y0+symin - y)**2)**.5 < max_deviation and 
                        amp > (max_to_average-1) * offset and brightness[j] > brightness.max() * frac):
                    if including_z:
                        zs = peak_local_max(bimg[:,x], exclude_border=1, threshold_abs=cutoff)
                        if len(zs) == 0: continue
                        zs = zs[np.isclose(bimg[zs,x], brightness[j], atol=10)].flatten()
                    else:
                        zs = [np.nan]

                    for z in zs:
                        x1,y1,_ = H@np.array([x+rxmin,y+rymin,1]).reshape(3,1)
                        xyzs.append([fov,cid,hybe_list.index(hybe),cb,float(x1),float(y1),float(z+H3[1,2])])
                        xyzs_crop.append([fov,cid,hybe_list.index(hybe),cb,float(x),float(y),float(z)])
            
                    szmin,szmax = max(0, z-spad), min(depth, z+spad+1)
                    zyx = stacks[szmin:szmax,symin:symax,sxmin:sxmax]
                    params3d = fit_gaussian_3d(zyx, x-sxmin, y-symin, z-szmin)
                    if params3d is not None:
                        amp, x0, y0, z0, sigma_x, sigma_y, sigma_z, offset = params3d
                        sigma = abs(sigma_x*sigma_y*sigma_z)**(1/3)
                        if abs(sigma_x) > .5 and abs(sigma_y) > .5 and abs(sigma_z) > .5 and sigma < max_sigma:
                            if ((x0+sxmin - x)**2 + (y0+symin - y)**2 + (z0+szmin - z)**2)**.5 < max_deviation:
                                if amp > (max_to_average-1) * offset and brightness[j] > brightness.max() * frac:
                                    x1,y1,_ = H@np.array([x+rxmin,y+rymin,1]).reshape(3,1)
                                    xyzs.append([fov,cid,hybe_list.index(hybe),cb,float(x1),float(y1),float(z+H3[1,2])])
                                    xyzs_crop.append([fov,cid,hybe_list.index(hybe),cb,float(x),float(y),float(z)])"""

    if len(xyzs) > 0:
        return np.array(xyzs, dtype=float), (img,bimg,np.array(xyzs_crop, dtype=float)), hybe
    else:
        return np.zeros((0,7), dtype=float), (img,bimg,np.zeros((0,7), dtype=float)), hybe

def _build_cell_crop(cell, hybe, channel, storage_path, fov, pad, modality=None):
    """
    Shared crop-building logic for localize_cell_2d_worker/3d_worker AND
    the interactive spot localization panel's "Current Cell" scope --
    factored out so the bulk workers and the interactive panel can't drift
    apart. Transforms this cell's own mask area into `hybe`'s native frame
    (cell.get_area_in_readout, itself never resampling raw pixels -- only
    coordinates move), reads a padded bbox crop of both the MIP and full
    Z-stack, and NaNs out every pixel outside the cell's own mask within
    that bbox (so background/neighboring-cell pixels never contaminate a
    per-cell peak search). h5py fancy-indexing requires ascending-order
    indices (unlike numpy); y_area/x_area come from np.where and aren't
    sorted, so the rectangular crop is sliced first (always
    contiguous/ascending) and the cell-mask fancy indexing done on the
    resulting in-memory array instead.

    Returns None only if the transformed area is genuinely empty (e.g.
    the cell has no mask pixels at all) -- get_area_in_readout itself
    never raises: no alignment matrix yet for (hybe, modality) means
    identity (no correction), never an error, per the same "no no-
    alignment case" principle compute_cell_alignment's own fallbacks
    already implement. Otherwise returns a dict: {'img': (h,w) MIP crop
    (NaN outside cell), 'stacks': (h,w,depth) Z-stack crop (NaN outside
    cell), 'bimg': (w,depth) Z-profile per column (nanmax over y),
    'rxmin': int, 'rymin': int, 'H': cell's yx matrix for this hybe
    (identity if none), 'Hz': cell's zx matrix for this hybe (identity
    if none)}.
    """
    modality = modality if modality is not None else cell.modality
    x_area, y_area = cell.get_area_in_readout(hybe, modality)
    if len(x_area) == 0:
        return None
    x_area, y_area = x_area.astype(int), y_area.astype(int)

    height, width = cell.frame_shape
    rymin, rymax = max(0, y_area.min() - pad), min(height, y_area.max() + pad + 1)
    rxmin, rxmax = max(0, x_area.min() - pad), min(width, x_area.max() + pad + 1)

    h5path = os.path.join(storage_path, f'FOV{fov:02d}', f'{hybe}_stack.h5')
    with h5py.File(h5path, 'r') as f:
        mip_crop = f[f'/mip/ch{channel}'][rymin:rymax, rxmin:rxmax]
        stacks_value = f[f'/stack/ch{channel}'][rymin:rymax, rxmin:rxmax, :]
        img = np.full((rymax - rymin, rxmax - rxmin), np.nan, dtype=float)
        img[y_area - rymin, x_area - rxmin] = mip_crop[y_area - rymin, x_area - rxmin]

    depth = stacks_value.shape[-1]
    stacks = np.full((rymax - rymin, rxmax - rxmin, depth), np.nan, dtype=float)
    stacks[y_area - rymin, x_area - rxmin] = stacks_value[y_area - rymin, x_area - rxmin]
    bimg = np.nanmax(stacks, axis=0)  # (width_crop, depth) -- Z-profile per column

    # matrix_to_shared (not a direct cell.matrices lookup): the caller
    # below applies H forward to a raw point to land in the pipeline's
    # ONE shared reference frame (RNA's own same-modality reference hybe
    # -- see ACell.matrix_to_shared's own docstring for why, not
    # cell.reference_hybe's frame), matching spot_mapper.raw_to_
    # reference's own convention, which matrix_to_shared is what
    # actually resolves.
    H = cell.matrix_to_shared(hybe, modality)
    Hz = cell.matrices.get((hybe, modality), {}).get('zx', np.eye(3))

    return {'img': img, 'stacks': stacks, 'bimg': bimg, 'rxmin': rxmin, 'rymin': rymin, 'H': H, 'Hz': Hz}


def refine_spot_z(spot, storage_path, fov, channel, hybe=None, cell=None, modality=None,
                  spad=5, peak_bound=2.0, init_sigma_xy=1.25, init_sigma_z=2.5,
                  min_sigma=0.1, max_sigma=2.5, min_hb_ratio=1.15, min_ah_ratio=0.15, max_uncert=2.0,
                  min_sep=3.0, component_threshold=0.3, max_components=3, claimed_positions=None,
                  use_mixture=True, z_window=15):
    """
    Adds/refines Z on a spot that's ALREADY PLACED (2D auto-detect or a
    manual click, so spot.raw_coordinate's own x,y are already known and
    trusted) -- never a fresh from-scratch detection pass. Per explicit
    design: 3D localization's job is to add Z information onto an
    existing 2D spot, not to find spots itself.

    Crops a small cubic centered on the spot's own raw (x,y) via
    spot_mapper.crop_for_localization(use_stack=True) -- the 3D exception,
    the only reason this touches the raw stack file at all -- finds a
    coarse Z as the crop's own single brightest voxel (no separate 2D
    peak search needed, x,y are already trusted).

    use_mixture (default True, no behavior change for existing callers
    that don't pass it): when False, skips find_local_peaks_3d and the
    multi-Gaussian mixture path ENTIRELY -- always a single fit_gaussian_3d
    at the original (x0,y0,z0) seed, regardless of how many real blobs the
    crop might actually contain. Per explicit request: the mixture fit
    (fit_gaussian_mixture_3d's least_squares over 7*n+1 params) is
    meaningfully slower than the single-Gaussian fit (8 params), and most
    spots are isolated single blobs that never needed it -- this project's
    UI now defaults to single-mode (fast) and treats mixture mode as an
    explicit, opt-in toggle for the crowded-crop case this was built for
    (see the multi-component paragraph below).

    z_window (default 15, only matters when use_mixture=True): candidate
    seeds are only ever searched for within +/-z_window planes of z0 (the
    crop's own coarse brightest-voxel Z, computed above) -- NOT across
    cubic's full native Z depth. cubic always carries the FULL Z-stack
    (frequently 100+ planes, see crop_for_localization/_build_cell_crop),
    so an unrestricted find_local_peaks_3d search over the whole thing
    would treat ANY brightish voxel ANYWHERE in that huge range as a
    candidate "second component" -- confirmed on real data as the actual
    cause of two real, confirmed failure modes: (1) an accepted mixture
    component whose fitted Z ends up dozens+ of planes from this spot's
    real neighborhood (still numerically "inside" cubic, so the earlier
    per-seed peak_bound clamp in fit_gaussian_mixture_3d never catches
    it -- that clamp only bounds the fit AROUND its own seed, and this
    seed was already wrong), rendering as a centroid floating outside the
    crop's real, physically-relevant extent; (2) most/all crops -- not
    just genuinely crowded ones -- registering a spurious "second
    component" once mixture mode is on, since SOME unrelated brightish
    voxel exists somewhere in a 100+-plane stack far more often than a
    real second PSF does. Restricting the seed search itself (not just
    post-filtering or clamping the fit afterward) means a spurious
    far-away voxel never becomes a candidate at all, and also fixes
    find_local_peaks_3d's own threshold_rel being computed against the
    WRONG (full-stack) maximum instead of this spot's own local
    neighborhood.

    When use_mixture=True and the crop plausibly contains more than one
    real PSF within that Z window (find_local_peaks_3d -- ChrTracer3's own
    FindPeaks3D-equivalent step, scoped to this one spot's crop; see that
    function's docstring): real case this handles is two spots close
    enough that a single click/auto-detect pick landed ambiguously between them (e.g.
    chr19_downstream_new/DNA_queue Cell 23, spots 29/30 -- ~2px apart,
    same z-slice, a lone single-Gaussian fit pulled toward the midpoint
    between both blobs). Exactly one detected component (by far the
    common case) fits identically to the use_mixture=False path -- fit_
    gaussian_3d, single Gaussian, using the ORIGINAL (x0,y0,z0) seed
    regardless of what find_local_peaks_3d returned, so ordinary isolated
    spots are unaffected byte-for-byte. More than one component runs
    fit_gaussian_mixture_3d instead and keeps only the BRIGHTEST accepted
    fitted component as THIS spot's own result -- per explicit request,
    the representative position saved for a spot should be whichever real
    blob is actually brightest, not just whichever landed nearest the
    original click. The other accepted component(s) are never spawned as
    separate ASpot records (see mixture_centroids below) -- they belong
    to the SAME physical spot's crop, just a different real blob within
    it, only returned alongside for display/record context.

    claimed_positions (optional): [(abs_x, abs_y), ...] of OTHER spots'
    own already-resolved positions from earlier in the SAME batch (e.g.
    _run_3d_localize/_view_3d_localize's own loop over multiple selected
    spots) -- a candidate component within min_sep of one of these is
    skipped in favor of this spot's next-brightest remaining candidate.
    Fixes a real confirmed failure mode on real data (chr19_downstream_
    new/DNA_queue: 3 of 14 multi-spot cells, including the user-flagged
    Cell 7) where two DIFFERENT already-placed spots, refined
    independently with no memory of each other, both end up nearest to
    the SAME dominant blob and silently collapse onto (near-)identical
    positions -- distinct spot identities should never converge onto one
    physical position when the crop genuinely offers a second, real,
    still-unclaimed candidate. Only ever skips a candidate in favor of
    another REAL accepted component already found in this crop; never
    invents one, and never blocks a spot whose only viable component is
    the contested one (genuine ambiguity, e.g. only 1 real blob for 2
    placed spots, is left to collide rather than forced apart).

    hybe defaults to spot.hybe. cell (optional): when given, the fitted
    raw (x,y,z) is also transformed through cell.matrix_to_shared/
    matrices' zx entry into the pipeline's ONE shared reference frame
    (RNA's own same-modality reference hybe -- matching every other spot
    coordinate in this project, see ACell.matrix_to_shared); omit for a
    cell-less (e.g. FOV unassigned-pool) spot, where coordinate ==
    raw_coordinate, no transform applies.

    Returns (new_coordinate, new_raw_coordinate, cubic, centroid,
    extra_results, mixture_centroids) -- NEVER mutates spot itself, so
    the caller decides whether/how to apply the result. Always this same
    6-tuple shape, even on an early "can't even crop this one" exit
    (cubic/centroid/extra_results/mixture_centroids fall back to
    None/None/[]/() respectively) -- callers can unpack unconditionally.
    cubic (the raw crop) is for fit-status display even when the fit
    failed. centroid, when not None, is a LIST of crop-local (x,y,z)
    tuples: this spot's own accepted (brightest) position first (if
    accepted), followed by any OTHER accepted component found in the same
    crop, for display context only (canvas/spot_fit_status.py draws the
    representative yellow, the rest blue). None only when THIS spot's own
    component was rejected -- new_coordinate/new_raw_coordinate are also
    None in that case, even if another component was accepted.

    extra_results: [(new_coordinate, new_raw_coordinate, amplitude), ...]
    -- one entry per accepted OTHER component (excluding whichever one
    became this spot's own representative result above), each already
    resolved into the same real coordinate frame new_coordinate/new_raw_
    coordinate are in (crop-local -> raw -> cell.reference_hybe's frame,
    same as this spot's own). Never includes a rejected component. Empty
    when the crop only ever had one real accepted component.

    mixture_centroids: () normally; when the mixture path found more than
    one accepted component, a tuple of (x, y, z, amplitude) in that SAME
    real coordinate frame, representative (brightest) first, then every
    other accepted component -- this is the ASpot.mixture_centroids-
    shaped value a caller should persist onto spot itself (see
    MainWindow._run_3d_localize) instead of spawning new ASpot records
    per component, per explicit request.
    """
    hybe = hybe or spot.hybe
    raw_x, raw_y = float(spot.raw_coordinate[0]), float(spot.raw_coordinate[1])
    try:
        cubic, (ymin, xmin) = spot_mapper.crop_for_localization(storage_path, fov, hybe, channel,
                                                                 (raw_x, raw_y), pad=spad, use_stack=True)
    except OSError:
        # This hybe's raw stack file doesn't exist (e.g. a spot recorded
        # against a hybe that was never ingested for this storage_path/
        # modality) -- graceful "can't refine this one," matching the
        # "no no-alignment error" pattern everywhere else in this
        # codebase, not a crash that aborts a whole batch of spots over
        # one bad one.
        return None, None, None, None, [], ()
    if cubic.size == 0:
        return None, None, cubic, None, [], ()

    x0, y0 = raw_x - xmin, raw_y - ymin
    z0 = float(np.unravel_index(np.nanargmax(cubic), cubic.shape)[2])

    if use_mixture:
        # Search for seeds only within +/-z_window of z0, not across
        # cubic's full native Z depth -- see this function's own
        # docstring on z_window for why the unrestricted search is a
        # confirmed, real bug (both a far-away accepted centroid and a
        # spurious "second component" on crops that are really just one
        # real blob). threshold_rel is also computed relative to THIS
        # sub-cube's own max, not the full stack's, so a real local peak
        # that isn't the single brightest voxel anywhere in 100+ planes
        # is no longer overlooked either.
        z0_idx = int(round(z0))
        zwin_min = max(0, z0_idx - z_window)
        zwin_max = min(cubic.shape[2], z0_idx + z_window + 1)
        seeds_local = find_local_peaks_3d(cubic[:, :, zwin_min:zwin_max], min_sep=min_sep,
                                          threshold_rel=component_threshold, max_peaks=max_components)
        seeds = [(sx, sy, sz + zwin_min) for (sx, sy, sz) in seeds_local]
    else:
        seeds = []
    if len(seeds) <= 1:
        results = [fit_gaussian_3d(cubic, x0, y0, z0, peak_bound=peak_bound, init_sigma_xy=init_sigma_xy,
                                   init_sigma_z=init_sigma_z, min_sigma=min_sigma, max_sigma=max_sigma,
                                   min_hb_ratio=min_hb_ratio, min_ah_ratio=min_ah_ratio, max_uncert=max_uncert)]
        primary = 0
    else:
        results = fit_gaussian_mixture_3d(cubic, seeds, peak_bound=peak_bound, init_sigma_xy=init_sigma_xy,
                                          init_sigma_z=init_sigma_z, min_sigma=min_sigma, max_sigma=max_sigma,
                                          min_hb_ratio=min_hb_ratio, min_ah_ratio=min_ah_ratio, max_uncert=max_uncert)
        # Brightness ranking (fitted amplitude, results[i][0]) -- per
        # explicit request, the representative component is whichever
        # real blob is actually brightest, not whichever landed nearest
        # the original click/auto-detect seed. A rejected (None) result
        # sorts last -- it's not a real candidate at all.
        order = sorted(range(len(seeds)), key=lambda i: -results[i][0] if results[i] is not None else float('inf'))
        primary = order[0]
        if claimed_positions:
            def _is_claimed(i):
                if results[i] is None:
                    return False
                ax, ay = results[i][1] + xmin, results[i][2] + ymin
                return any((ax - cx) ** 2 + (ay - cy) ** 2 < min_sep ** 2 for cx, cy in claimed_positions)
            unclaimed = [i for i in order if not _is_claimed(i)]
            if unclaimed:
                primary = unclaimed[0]
            # else: every viable component here is already claimed by a
            # sibling spot -- genuine ambiguity, fall back to brightest
            # rather than inventing a distinction that isn't in the data.

    centroids = [(r[1], r[2], r[3]) for r in results if r is not None]
    if results[primary] is not None:
        own = (results[primary][1], results[primary][2], results[primary][3])
        centroids = [own] + [c for c in centroids if c != own]
    centroids = centroids or None

    def _to_real(r):
        # Same crop-local -> real-frame transform for ANY accepted
        # component, representative or not -- a non-representative
        # component is just as real a fit, it just isn't the brightest
        # one for THIS spot (see this function's own docstring on
        # claimed_positions). Returns (new_coordinate, new_raw_coordinate, amp).
        amp, xf, yf, zf, _, _, _, _ = r
        raw = (float(xf + xmin), float(yf + ymin), float(zf))
        if cell is not None:
            m = modality if modality is not None else cell.modality
            H = cell.matrix_to_shared(hybe, m)
            Hz = cell.matrices.get((hybe, m), {}).get('zx', np.eye(3))
            x1, y1, _ = H @ np.array([raw[0], raw[1], 1]).reshape(3, 1)
            coord = (float(x1), float(y1), float(zf + Hz[1, 2]))
        else:
            coord = raw
        return coord, raw, float(amp)

    # Every OTHER accepted component in this crop, real-frame-resolved --
    # for display, and for a caller that persists the full mixture as
    # mixture_centroids (see MainWindow._run_3d_localize) instead of
    # spawning a separate ASpot per component. Never includes the
    # representative itself, and never includes a rejected component
    # (nothing real here from a fit that failed its own accept criteria).
    extra_results = [_to_real(r) for i, r in enumerate(results) if i != primary and r is not None]

    if results[primary] is None:
        return None, None, cubic, centroids, extra_results, ()

    new_coordinate, new_raw, amp = _to_real(results[primary])
    mixture_centroids = (((new_coordinate[0], new_coordinate[1], new_coordinate[2], amp),)
                         + tuple((c[0][0], c[0][1], c[0][2], c[2]) for c in extra_results)) if extra_results else ()
    return new_coordinate, new_raw, cubic, centroids, extra_results, mixture_centroids


def localize_cell_2d_worker(cell, hybe, channel, storage_path, fov,
                            max_to_background, max_to_average, absolute_threshold,
                            min_distance, frac, max_num_alleles, pad):
    """
    2D localization for one cell in one hybe, with sub-pixel gaussian
    refinement (the fit_gaussian_2d step that exists but is commented out
    in localize_spots_worker above, reactivated here). Ports
    scripts/utils.py's localize_2d_spots_worker, adapted to this project's
    ACell/composed-matrix model instead of the old flat H5 /cells/matrix
    arrays. Returns (cell.id, hybe, [ASpot, ...]) -- run inside a
    ProcessPoolExecutor, so results are returned rather than mutating cell
    in place (a separate-process copy wouldn't be visible to the caller).
    """
    from ..models.spot import ASpot
    spots = []
    crop = _build_cell_crop(cell, hybe, channel, storage_path, fov, pad)
    if crop is None:
        return cell.id, hybe, spots
    img, stacks, bimg = crop['img'], crop['stacks'], crop['bimg']
    rxmin, rymin, H, Hz = crop['rxmin'], crop['rymin'], crop['H'], crop['Hz']

    cutoff = max_to_background * np.nanquantile(img, 0.5)
    yx = peak_local_max(img, min_distance=min_distance, exclude_border=1,
                        threshold_abs=max(cutoff, absolute_threshold))
    if len(yx) == 0:
        return cell.id, hybe, spots
    brightness = img[yx[:, 0], yx[:, 1]]

    for j in brightness.argsort()[::-1][:max_num_alleles]:
        y, x = yx[j]
        z_candidates = peak_local_max(bimg[x], exclude_border=1,
                                      threshold_abs=max(bimg[x].max() * .9, absolute_threshold, cutoff))
        if len(z_candidates) == 0:
            continue
        z = int(z_candidates[bimg[x, z_candidates].argmax()])

        symin, symax = max(0, y - pad), min(img.shape[0], y + pad + 1)
        sxmin, sxmax = max(0, x - pad), min(img.shape[1], x + pad + 1)
        params = fit_gaussian_2d(img[symin:symax, sxmin:sxmax], x - sxmin, y - symin)
        if params is None:
            continue
        amp, xo, yo, sigma_x, sigma_y, theta, offset = params
        if not (abs(sigma_x) > .5 and abs(sigma_y) > .5):
            continue
        if ((xo + sxmin - x) ** 2 + (yo + symin - y) ** 2) ** .5 >= 3:
            continue
        if not (amp + offset > max_to_average * offset and brightness[j] > brightness.max() * frac):
            continue

        raw_x, raw_y = x + rxmin, y + rymin
        x1, y1, _ = H @ np.array([raw_x, raw_y, 1]).reshape(3, 1)
        z1 = z + Hz[1, 2]

        spot = ASpot()
        spot.set_metadata(fov=fov, hybe=hybe, channel=channel, cell=cell.id,
                          coordinate=(float(x1), float(y1), float(z1)),
                          raw_coordinate=(float(raw_x), float(raw_y), float(z)),
                          brightness=float(brightness[j]))
        spots.append(spot)

    return cell.id, hybe, spots

def localize_cells_2d(cell_container, fov, hybe_records, channel,
                      max_to_background=1.25, max_to_average=1.25, absolute_threshold=450.0,
                      min_distance=3, frac=0.8, max_num_alleles=2, pad=5,
                      storage_path=None, n_procs=4):
    """
    Bulk (non-interactive) 2D localization over every cell in
    cell_container.get_cells(fov), across every hybe in hybe_records.
    Parameters are an already-confirmed set -- tune interactively via
    LocalizationWidget first, then run this in bulk; not re-tuned per call.
    Writes results directly into each cell's .spots/.num_spots/.total_num_spots.
    """
    cells = cell_container.get_cells(fov)
    tasks = [(cell, record['folder']) for cell in cells for record in hybe_records]

    with ProcessPoolExecutor(max_workers=n_procs) as executor:
        futures = [executor.submit(localize_cell_2d_worker, cell, hybe, channel, storage_path, fov,
                                   max_to_background, max_to_average, absolute_threshold,
                                   min_distance, frac, max_num_alleles, pad)
                  for cell, hybe in tasks]
        cells_by_id = {c.id: c for c in cells}
        for future in as_completed(futures):
            cell_id, hybe, spots = future.result()
            if len(spots) == 0:
                continue
            cell = cells_by_id[cell_id]
            cell.spots.extend(spots)
            cell.num_spots[hybe] = cell.num_spots.get(hybe, 0) + len(spots)
            cell.total_num_spots += len(spots)

def localize_cell_3d_worker(cell, hybe, channel, storage_path, fov,
                            max_to_background=1.25, max_to_average=1.25, absolute_threshold=200.0,
                            min_distance=3, frac=0.8, max_num_alleles=2, max_sigma=2.5, pad=5, spad=5,
                            peak_bound=2.0, max_uncert=2.0, min_hb_ratio=1.15, min_ah_ratio=0.15):
    """
    3D localization for one cell in one hybe: a 2D peak search + coarse
    z-profile seed (same as localize_cell_2d_worker/localize_spots_worker),
    a quick unconstrained 2D gaussian refinement of that seed (cheap
    pre-filter, saves running the real 3D fit on obviously-bad seeds), then
    the real sub-pixel 3D gaussian fit (fit_gaussian_3d) -- bounded, with
    its own internal accept/reject built in (see that function's own
    docstring; peak_bound/max_uncert/min_hb_ratio/min_ah_ratio are threaded
    straight through to it, real defaults from ChrTracer3's Pars Fit
    Spots.csv). max_sigma is also threaded through as fit_gaussian_3d's
    own sigma upper bound now, not a separate post-hoc check -- the fit
    itself can no longer return an over-wide PSF to reject after the fact.
    """
    from ..models.spot import ASpot
    spots = []
    crop = _build_cell_crop(cell, hybe, channel, storage_path, fov, pad)
    if crop is None:
        return cell.id, hybe, spots
    img, stacks, bimg = crop['img'], crop['stacks'], crop['bimg']
    rxmin, rymin, H, Hz = crop['rxmin'], crop['rymin'], crop['H'], crop['Hz']
    depth = stacks.shape[-1]

    cutoff = max_to_background * np.nanquantile(img, 0.5)
    yx = peak_local_max(img, min_distance=min_distance, exclude_border=1,
                        threshold_abs=max(cutoff, absolute_threshold))
    if len(yx) == 0:
        return cell.id, hybe, spots
    brightness = img[yx[:, 0], yx[:, 1]]

    for j in brightness.argsort()[::-1][:max_num_alleles]:
        y, x = yx[j]
        z_candidates = peak_local_max(bimg[x], exclude_border=1,
                                      threshold_abs=max(bimg[x].max() * .9, absolute_threshold, cutoff))
        if len(z_candidates) == 0:
            continue
        z = int(z_candidates[bimg[x, z_candidates].argmax()])

        symin, symax = max(0, y - pad), min(img.shape[0], y + pad + 1)
        sxmin, sxmax = max(0, x - pad), min(img.shape[1], x + pad + 1)
        params2d = fit_gaussian_2d(img[symin:symax, sxmin:sxmax], x - sxmin, y - symin)
        if params2d is None:
            continue
        amp, xo, yo, sigma_x, sigma_y, theta, offset = params2d
        if not (abs(sigma_x) > .5 and abs(sigma_y) > .5 and (sigma_x * sigma_y) ** .5 < max_sigma):
            continue
        if ((xo + sxmin - x) ** 2 + (yo + symin - y) ** 2) ** .5 >= peak_bound + 1:
            continue  # coarse pre-filter only -- the real bound is enforced inside fit_gaussian_3d below
        if not (amp > (max_to_average - 1) * offset and brightness[j] > brightness.max() * frac):
            continue

        szmin, szmax = max(0, z - spad), min(depth, z + spad + 1)
        zyx = stacks[symin:symax, sxmin:sxmax, szmin:szmax]
        params3d = fit_gaussian_3d(zyx, x - sxmin, y - symin, z - szmin,
                                   peak_bound=peak_bound, max_sigma=max_sigma,
                                   max_uncert=max_uncert, min_hb_ratio=min_hb_ratio, min_ah_ratio=min_ah_ratio)
        if params3d is None:
            continue
        amp3, x0, y0, z0, sigma_x3, sigma_y3, sigma_z3, offset3 = params3d
        if not (brightness[j] > brightness.max() * frac):
            continue  # cross-spot relative-brightness filter -- not something fit_gaussian_3d checks itself

        raw_x, raw_y, raw_z = x + rxmin, y + rymin, z0 + szmin
        x1, y1, _ = H @ np.array([raw_x, raw_y, 1]).reshape(3, 1)
        z1 = raw_z + Hz[1, 2]

        spot = ASpot()
        spot.set_metadata(fov=fov, hybe=hybe, channel=channel, cell=cell.id,
                          coordinate=(float(x1), float(y1), float(z1)),
                          raw_coordinate=(float(raw_x), float(raw_y), float(raw_z)),
                          brightness=float(brightness[j]))
        spots.append(spot)

    return cell.id, hybe, spots

def localize_cells_3d(cell_container, fov, hybe_records, channel,
                      max_to_background=1.25, max_to_average=1.25, absolute_threshold=200.0,
                      min_distance=3, frac=0.8, max_num_alleles=2, max_sigma=2.5,
                      pad=5, spad=5, storage_path=None, n_procs=4,
                      peak_bound=2.0, max_uncert=2.0, min_hb_ratio=1.15, min_ah_ratio=0.15):
    """3D counterpart of localize_cells_2d -- see localize_cell_3d_worker. Defaults
    (max_sigma, peak_bound, max_uncert, min_hb_ratio, min_ah_ratio, absolute_threshold)
    are ChrTracer3's own real values (Pars Fit Spots.csv: datMinPeakHeight,
    fidMaxFitWidth-derived max_sigma, etc.), not arbitrary starting points."""
    cells = cell_container.get_cells(fov)
    tasks = [(cell, record['folder']) for cell in cells for record in hybe_records]

    with ProcessPoolExecutor(max_workers=n_procs) as executor:
        futures = [executor.submit(localize_cell_3d_worker, cell, hybe, channel, storage_path, fov,
                                   max_to_background, max_to_average, absolute_threshold,
                                   min_distance, frac, max_num_alleles, max_sigma, pad, spad,
                                   peak_bound, max_uncert, min_hb_ratio, min_ah_ratio)
                  for cell, hybe in tasks]
        cells_by_id = {c.id: c for c in cells}
        for future in as_completed(futures):
            cell_id, hybe, spots = future.result()
            if len(spots) == 0:
                continue
            cell = cells_by_id[cell_id]
            cell.spots.extend(spots)
            cell.num_spots[hybe] = cell.num_spots.get(hybe, 0) + len(spots)
            cell.total_num_spots += len(spots)