import os
import numpy as np
import numpy.linalg as la
from scipy.optimize import minimize, least_squares
from scipy.stats import t as student_t
from skimage.feature import peak_local_max
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py

from ..io import paths
from ..io import preprocess
from .engine import make_engine
import cv2

from ..alignment import chain as alignment
from ..io import analysis_store
from ..alignment import spot_mapper

def cell_z_offset(cell, hybe, modality, resolver=None):
    """
    Additive z (planes) taking a spot in (hybe, modality) to the shared
    frame.

    THE single place localization reads z from. This replaced five
    independent inline `z + Hz` sites -- that repetition is exactly
    what let a wrong index (`Hz[1, 2]`, structurally always 0) survive
    unnoticed in all five at once, silently discarding every Z correction.

    `resolver` (frames.FrameResolver, from MainWindow._frame_resolver) is
    required to include the FOV-level CROSS-MODAL z drift, which is
    FOV-bounded and therefore applies to unassigned spots too. Without it
    only the cell's own same-modality residual is available; a bare
    residual entry raises rather than being half-applied.
    """
    if resolver is not None:
        return float(resolver.z_to_shared(hybe, modality, cell))
    if cell is None:
        return 0.0
    entry = cell.matrices.get((hybe, modality), {})
    if entry.get('yx_is_residual') is None and 'dz' not in entry and 'zx' not in entry:
        return 0.0
    return alignment.entry_dz(entry)


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
                    min_sigma=0.1, max_sigma=2.5, min_hb_ratio=1.2, min_ah_ratio=0.25, max_uncert=2.0,
                    symmetric_xy=False):
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

    n_params = 7 if symmetric_xy else 8
    if len(values) <= n_params:
        return None  # not enough data to constrain the fit at all

    iy, ix, iz = int(round(y0)), int(round(x0)), int(round(z0))
    if 0 <= iy < cubic.shape[0] and 0 <= ix < cubic.shape[1] and 0 <= iz < cubic.shape[2] and np.isfinite(cubic[iy, ix, iz]):
        h = float(cubic[iy, ix, iz])
    else:
        h = float(np.nanmax(values))

    amp0 = float(np.nanmax(values))
    offset0 = float(np.nanmin(values))
    if symmetric_xy:
        # ChrTracer3 FitPsf3D's own reduced model (its symmetricXY
        # default): ONE shared XY sigma. Fewer degrees of freedom is
        # steadier on dim spots, at the cost of astigmatic PSFs. The
        # first four parameters keep the same layout as the free model,
        # so the CI position gates below index identically.
        def _residuals(p):
            a_, x_, y_, z_, sxy_, sz_, o_ = p
            return _residuals_3d([a_, x_, y_, z_, sxy_, sxy_, sz_, o_], x, y, z, values)
        p0 = [amp0, x0, y0, z0, init_sigma_xy, init_sigma_z, offset0]
        lb = [0, x0 - peak_bound, y0 - peak_bound, z0 - peak_bound, min_sigma, min_sigma, 0]
        ub = [65535, x0 + peak_bound, y0 + peak_bound, z0 + peak_bound, max_sigma, 2 * max_sigma, 65535]
        fit_args = ()
    else:
        _residuals = _residuals_3d
        p0 = [amp0, x0, y0, z0, init_sigma_xy, init_sigma_xy, init_sigma_z, offset0]
        lb = [0, x0 - peak_bound, y0 - peak_bound, z0 - peak_bound, min_sigma, min_sigma, min_sigma, 0]
        ub = [65535, x0 + peak_bound, y0 + peak_bound, z0 + peak_bound, max_sigma, max_sigma, 2 * max_sigma, 65535]
        fit_args = (x, y, z, values)

    try:
        result = least_squares(_residuals, p0, bounds=(lb, ub), args=fit_args)
    except Exception:
        return None
    if not result.success:
        return None

    if symmetric_xy:
        amp, x0f, y0f, z0f, sigma_xy_f, sigma_z, offset = result.x
        sigma_x = sigma_y = sigma_xy_f
    else:
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
    ci_half = student_t.ppf(0.975, dof) * se  # both layouts start amp,x0,y0,z0 -- gates below index those

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
                            min_sigma=0.1, max_sigma=2.5, min_hb_ratio=1.2, min_ah_ratio=0.25,
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

def _build_cell_crop(cell, hybe, channel, storage_path, fov, pad, modality=None,
                     fov_matrices=None, cell_reference_hybe_matrix=None, resolver=None):
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

    fov_matrices/cell_reference_hybe_matrix (both optional, default None):
    live FOV/cross-modal matrices for `hybe`'s own modality, and cell.
    reference_hybe's own transform into that same shared frame -- same
    two parameters compute_cell_alignment itself takes, same convention
    (cell_reference_hybe_matrix defaults to fov_matrices.get(cell.
    reference_hybe, identity) when not given, correct whenever fov_
    matrices is the CELL's own modality; a cross-modal caller must
    resolve and pass it explicitly, since cell.reference_hybe is never a
    real key in the OTHER modality's own fov_matrices). Used ONLY as a
    fallback -- per confirmed real bug, cell.get_area_in_readout/matrix_
    to_shared silently collapse to IDENTITY (not a real FOV-level
    transform) whenever this cell has no cell.matrices/matrix_anchors
    entry for (hybe, modality) at all (never had cell-level alignment run
    for it), mispositioning both this crop's own placement AND any spot
    coordinate a caller derives from it. When real cell-level data IS
    present for both this hybe and cell.reference_hybe (same "have_real"
    condition MainWindow._matrix_to_cellref/_matrix_to_shared already
    use), that real, cell-level-refined result is used unchanged --
    fov_matrices only ever substitutes for a genuinely MISSING cell-level
    entry, never overrides a real one. Omitting fov_matrices preserves
    the old identity-default behavior exactly (existing callers that
    haven't been updated yet still work, just without this fix).

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
    modality = modality if modality is not None else cell.reference_modality
    key = (hybe, modality)
    # reference_MODALITY -- cell.reference_hybe can
    # belong to the other modality once a cytoplasm is attached (see
    # ACell's own docstring), and pairing it with the cell's home
    # modality would key a (hybe, modality) entry that never exists.
    reference_modality = cell.reference_modality
    self_key = (cell.reference_hybe, reference_modality)
    have_real = (key in cell.matrices and self_key in cell.matrices
                 and modality in cell.matrix_anchors and reference_modality in cell.matrix_anchors)
    if resolver is not None:
        # THE composition for session callers -- handles every storage
        # form uniformly (residual entries recomposed with the live FOV
        # layer, legacy composed passed through, missing layers identity),
        # making the have_real distinction below irrelevant. The ACell-
        # method branch RAISES by design on residual-form matrices
        # (confirmed real crash: 3D Localization View/Run on any cell
        # that had run cell alignment), so a session must never reach it.
        H_cellref, _dz, _missing = resolver.transform(
            (hybe, modality), (cell.reference_hybe, cell.reference_modality), cell)
        y_lit, x_lit = cell.area
        cy, cx = alignment.align_cell((y_lit, x_lit), la.inv(H_cellref), cell.frame_shape)
        x_area, y_area = cx, cy
    elif have_real or fov_matrices is None or (hybe, fov_matrices.modality) not in fov_matrices:
        # real cell-level data, or no fallback available -- old behavior
        # (identity-default via cell.get_area_in_readout/matrix_to_shared
        # when neither real data nor a fallback exists). Legacy composed
        # entries only: get_area_in_readout raises on residual matrices.
        y_area, x_area = cell.get_area_in_readout(hybe, modality)
    else:
        # No real cell-level entry for (hybe, modality) -- fall back to
        # the FOV/cross-modal-only transform instead of silently
        # collapsing to identity. Mirrors ACell.get_area_in_readout's own
        # body exactly, substituting the FOV-only H_cellref for cell.
        # matrix_to(hybe, modality).
        if cell_reference_hybe_matrix is None:
            cell_reference_hybe_matrix = fov_matrices.get((cell.reference_hybe, cell.reference_modality), np.eye(3))
        H_cellref = alignment.hybe_to_cellref_matrix(fov_matrices, cell_reference_hybe_matrix, hybe)
        y_lit, x_lit = cell.area
        cy, cx = alignment.align_cell((y_lit, x_lit), la.inv(H_cellref), cell.frame_shape)
        x_area, y_area = cx, cy
    if len(x_area) == 0:
        return None
    x_area, y_area = x_area.astype(int), y_area.astype(int)

    height, width = cell.frame_shape
    rymin, rymax = max(0, y_area.min() - pad), min(height, y_area.max() + pad + 1)
    rxmin, rxmax = max(0, x_area.min() - pad), min(width, x_area.max() + pad + 1)

    h5path = paths.stack_path(storage_path, fov, hybe)
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
    # actually resolves. Same have_real-gated fallback as the area/crop
    # placement above -- fov_matrices[hybe] directly (no cell-reference
    # bridging needed here, unlike the area case, since fov_matrices is
    # already expressed in the shared frame) whenever this cell has no
    # real cell-level entry for (hybe, modality).
    if resolver is not None:
        H = resolver.to_shared(hybe, modality, cell)
    elif not have_real and fov_matrices is not None and (hybe, fov_matrices.modality) in fov_matrices:
        H = fov_matrices[(hybe, fov_matrices.modality)]
    else:
        H = cell.matrix_to_shared(hybe, modality)
    Hz = alignment.entry_dz(cell.matrices.get((hybe, modality)))

    return {'img': img, 'stacks': stacks, 'bimg': bimg, 'rxmin': rxmin, 'rymin': rymin, 'H': H, 'Hz': Hz}


def refine_spot_z(spot, storage_path, fov, channel, hybe=None, cell=None, modality=None,
                  spad=5, peak_bound=2.0, init_sigma_xy=1.25, init_sigma_z=2.5,
                  min_sigma=0.1, max_sigma=2.5, min_hb_ratio=1.2, min_ah_ratio=0.25, max_uncert=2.0,
                  min_sep=3.0, component_threshold=0.3, max_components=3, claimed_positions=None,
                  use_mixture=True, z_window=15, fov_matrices=None, resolver=None):
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
    coordinate in this project, see ACell.matrix_to_shared).

    fov_matrices (optional, only consulted when cell is None): the
    {hybe: 3x3} FOV/cross-modal-level chain for this storage_path/fov
    (see main_window._composed_fov_matrices_for_cell_alignment and
    spot_mapper.raw_to_reference's own cell=None docstring) -- maps a
    cell-less (FOV unassigned-pool) spot's fitted (x,y) into the SAME
    shared frame too, identity only for the missing cell-level residual,
    so an unassigned spot's coordinate stays comparable to a cell-owned
    one's instead of silently staying in its own hybe's raw frame. Falls
    back to raw==coordinate (x,y unchanged) when omitted or this hybe
    has no FOV-level matrix yet -- the graceful "no alignment yet"
    degradation, matching every other cell=None caller in this codebase.

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
    one accepted component, a tuple of (Y, X, z, amplitude) -- Y FIRST,
    built from new_coordinate at line 761, which is adj_coordinate-shaped
    and therefore yx (legacy/migrate_store_to_yx.py). This said
    "(x, y, z, amplitude)" until 2026-08-27 and was stale -- in that SAME
    real coordinate frame, representative (brightest) first, then every
    other accepted component -- this is the ASpot.mixture_centroids-
    shaped value a caller should persist onto spot itself (see
    MainWindow._run_3d_localize) instead of spawning new ASpot records
    per component, per explicit request.
    """
    hybe = hybe or spot.hybe
    raw_y, raw_x = float(spot.raw_coordinate[0]), float(spot.raw_coordinate[1])
    try:
        cubic, (ymin, xmin) = spot_mapper.crop_for_localization(storage_path, fov, hybe, channel,
                                                                 (raw_y, raw_x), pad=spad, use_stack=True)
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

    # THE localizer seam (engine.py): seeding + fitting live behind
    # LocalizeEngine so the gaussian engine and a future ML engine are
    # swappable; the z_window-restricted seed search rationale lives in
    # this function's own docstring.
    engine = make_engine('gaussian', peak_bound=peak_bound, init_sigma_xy=init_sigma_xy,
                                    init_sigma_z=init_sigma_z, min_sigma=min_sigma, max_sigma=max_sigma,
                                    min_hb_ratio=min_hb_ratio, min_ah_ratio=min_ah_ratio, max_uncert=max_uncert,
                                    min_sep=min_sep, component_threshold=component_threshold, z_window=z_window)
    results, seeds = engine.raw_components(cubic, (y0, x0, z0),
                                           n_max=max_components if use_mixture else 1)
    if len(seeds) <= 1:
        primary = 0
    else:
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
                apy, apx = results[i][2] + ymin, results[i][1] + xmin
                return any((apy - py) ** 2 + (apx - px) ** 2 < min_sep ** 2 for py, px in claimed_positions)
            unclaimed = [i for i in order if not _is_claimed(i)]
            if unclaimed:
                primary = unclaimed[0]
            # else: every viable component here is already claimed by a
            # sibling spot -- genuine ambiguity, fall back to brightest
            # rather than inventing a distinction that isn't in the data.

        # Three post-fit QC gates on every OTHER accepted component,
        # evaluated relative to the REPRESENTATIVE (whichever one
        # actually became primary above), not to z0/the click -- a
        # component passing fit_gaussian_mixture_3d's own per-component
        # accept criteria (CI width, h/offset, amp/h) only proves IT is a
        # real, well-fit peak, not that it's actually the SAME physical
        # spot's own second blob rather than some other, unrelated real
        # feature the crop happens to also contain:
        # 1. Z: more than z_window from the representative's own Z.
        #    z_window above only bounds the SEED SEARCH domain around z0
        #    (the crop's own coarse brightest-voxel estimate, NOT
        #    necessarily the representative's own final fitted Z) -- two
        #    seeds can each individually land within z_window of z0 while
        #    still being up to ~2*z_window apart from EACH OTHER (one on
        #    either side of z0). Confirmed on real data (an unassigned
        #    Hyb_010 spot): representative Z=130.6, sibling Z=114.0,
        #    16.6px apart -- each within z_window=15 of a z0 near the
        #    midpoint, but not of each other.
        # 2. X,Y: more than spad (the crop's own half-width) from the
        #    representative's own (x,y) -- per confirmed real bug/
        #    screenshot: only Z was bounded, so a component clear across
        #    the crop (a visibly different blob near the crop's own edge,
        #    nothing to do with the representative) still passed through
        #    as a "sibling." Real "two blobs one click landed between"
        #    pairs are both within the SAME small crop and close to each
        #    other by construction; spad is the same bound already used
        #    to build that crop in the first place.
        # 3. Amplitude: less than half the representative's own fitted
        #    amplitude -- per explicit request, a trivial QC floor. A
        #    real second blob worth reporting as its own mixture
        #    component should be reasonably comparable in brightness to
        #    the representative; something far dimmer passing the OTHER
        #    accept criteria is more likely a marginal noise peak that
        #    happened to also clear those thresholds.
        if results[primary] is not None:
            primary_amp, primary_x, primary_y, primary_z = results[primary][:4]
            for i, r in enumerate(results):
                if i == primary or r is None:
                    continue
                amp, x0f, y0f, z0f = r[:4]
                too_far_z = abs(z0f - primary_z) > z_window
                too_far_xy = np.hypot(x0f - primary_x, y0f - primary_y) > spad
                too_dim = amp < 0.5 * primary_amp
                if too_far_z or too_far_xy or too_dim:
                    results[i] = None

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
        raw = (float(yf + ymin), float(xf + xmin), float(zf))
        if cell is not None:
            m = modality if modality is not None else cell.reference_modality
            # Resolver first -- ACell.matrix_to_shared raises by design on
            # residual-form matrices (post cell alignment).
            H = (resolver.to_shared(hybe, m, cell) if resolver is not None
                 else cell.matrix_to_shared(hybe, m))
            Hz = alignment.entry_dz(cell.matrices.get((hybe, m)))
            y1, x1, _ = H @ np.array([raw[0], raw[1], 1]).reshape(3, 1)
            coord = (float(y1), float(x1), float(zf + cell_z_offset(cell, hybe, m, resolver)))
        elif fov_matrices and (hybe, fov_matrices.modality) in fov_matrices:
            y1, x1 = spot_mapper.raw_to_reference((raw[0], raw[1]), hybe, fov_matrices, modality=modality, cell=None)
            coord = (y1, x1, raw[2])
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
                            min_distance, frac, max_num_alleles, pad, resolver=None):
    """
    2D localization for one cell in one hybe, with sub-pixel gaussian
    refinement (the fit_gaussian_2d step that exists but is commented out
    in legacy/localize_spots_worker.py, reactivated here -- take only the
    fit from there, never its matrix handling, which reads the obsolete
    stack-file /matrix store; see that file's own header). Ports
    scripts/utils.py's localize_2d_spots_worker, adapted to this project's
    ACell/composed-matrix model instead of the old flat H5 /cells/matrix
    arrays. Returns (cell.id, hybe, [ASpot, ...]) -- run inside a
    ProcessPoolExecutor, so results are returned rather than mutating cell
    in place (a separate-process copy wouldn't be visible to the caller).
    """
    from ..models.spot import ASpot
    spots = []
    crop = _build_cell_crop(cell, hybe, channel, storage_path, fov, pad, resolver=resolver)
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
        y1, x1, _ = H @ np.array([raw_y, raw_x, 1]).reshape(3, 1)
        z1 = z + cell_z_offset(cell, hybe, modality, resolver)

        spot = ASpot()
        spot.modality = analysis_store.modality_of(storage_path)
        spot.set_metadata(fov=fov, hybe=hybe, channel=channel, cell=cell.id,
                          adj_coordinate=(float(y1), float(x1), float(z1)),
                          raw_coordinate=(float(raw_y), float(raw_x), float(z)),
                          brightness=float(brightness[j]))
        spots.append(spot)

    return cell.id, hybe, spots

# -- chromatin tracing --
#
# An allele's (x,y) is already known (the seed spot the user selected in
# Spot Localization -- see AnAllele.anchor_hybe/coordinate) -- no fresh
# detection here. Per hybe: crop+fit the fiducial channel (single
# component, this hybe's own local anchor), compute this hybe's own drift
# relative to the reference hybe's fiducial (both already in the shared
# frame -- see spot_mapper.raw_to_reference), reject the hybe outright if
# that drift is too large, otherwise crop+fit the readout channel
# (mixture-capable) and apply the same drift correction to every accepted
# candidate. This is plain 3D localization (fit_gaussian_3d/
# fit_gaussian_mixture_3d, unmodified) plus a shift calculation -- no new
# fitting math, no new registration subsystem. `cell` is passed straight
# through to spot_mapper exactly like refine_spot_z already does (None for
# an allele whose anchor spot has no owning cell, a real ACell otherwise)
# -- chromatin tracing has no mechanistic dependency on cell-based
# alignment; it just uses whichever matrix chain already applies to this
# allele's own anchor spot.

def _z_boundary_offset(depth, z_boundary_trim, min_fit_depth=9):
    """
    How many planes to actually shave off EACH end of a depth-`depth` crop
    for a requested boundary trim -- clamped so the fit always keeps at
    least min_fit_depth planes. On real stacks (120/177 planes) the clamp
    never engages; it exists so a pathological short stack degrades to a
    smaller trim instead of an empty fit domain.
    """
    if z_boundary_trim <= 0:
        return 0
    return max(0, min(int(z_boundary_trim), (depth - min_fit_depth) // 2))


def _localize_fiducial_hybe(shared_xy, hybe, fiducial_channel, storage_path, fov, modality, cell, fov_matrices,
                            spad=8, peak_bound=2.0, init_sigma_xy=1.25, init_sigma_z=2.5,
                            min_sigma=0.1, max_sigma=2.5, min_hb_ratio=1.2, min_ah_ratio=0.25, max_uncert=2.0,
                            z_boundary_trim=0, resolver=None):
    """
    Crops+fits ONE hybe's fiducial channel around an allele's already-known
    shared-frame (x,y). Always single-component (fit_gaussian_3d) -- no
    mixture mode here, per explicit request: a fiducial's whole purpose is
    ONE per-hybe drift-correction anchor, and a fiducial bead field has no
    legitimate multi-locus case the way a real genomic-locus readout does
    (see _localize_readout_hybe's own mixture-capable search).

    Returns (shared_xyz_amp_or_None, cubic_or_None, crop_local_xyz_or_None)
    -- cubic/crop_local_xyz are for display only (canvas.spot_fit_status.
    draw_spot_fit_status's own cubic/centroid params, unmodified), always
    returned regardless of caller the same way refine_spot_z always returns
    its own cubic -- a batch caller (build_chromatin_trace_allele) simply
    discards them; a preview caller keeps them. cubic is None only when
    this hybe's raw stack doesn't exist or the crop itself came back empty;
    crop_local_xyz is None whenever nothing was accepted, even if cubic
    itself is real (matching draw_spot_fit_status's own "circled = good,
    plain = missing" convention).
    """
    try:
        raw_y, raw_x = spot_mapper.reference_to_raw(shared_xy, hybe, fov_matrices, modality=modality, cell=cell, resolver=resolver)
        cubic, (ymin, xmin) = spot_mapper.crop_for_localization(storage_path, fov, hybe, fiducial_channel,
                                                                 (raw_y, raw_x), pad=spad, use_stack=True)
    except OSError:
        return None, None, None, None
    if cubic.size == 0:
        return None, None, None, None
    x0, y0 = raw_x - xmin, raw_y - ymin

    # BOUNDARY trim, not a center window: the fit runs on the stack minus its
    # outermost z_boundary_trim planes each side. Per explicit decision, a
    # center window (+-N around a seed) is dangerous for an allele sitting
    # near the top or bottom of its cell -- the real peak can fall outside a
    # window placed around a noisy seed -- while the stack's outermost planes
    # are out-of-focus junk regardless of where the allele sits. The FIT is
    # restricted; the DISPLAY cubic stays full-depth, and every returned z is
    # mapped back to absolute plane units (zf + z_off), so traces, drift
    # gates and the overlay/grid builders are untouched.
    z_off = _z_boundary_offset(cubic.shape[2], z_boundary_trim)
    fit_cubic = cubic[:, :, z_off:cubic.shape[2] - z_off] if z_off else cubic
    z0 = float(np.unravel_index(np.nanargmax(fit_cubic), fit_cubic.shape)[2])

    engine = make_engine('gaussian', peak_bound=peak_bound, init_sigma_xy=init_sigma_xy,
                                    init_sigma_z=init_sigma_z, min_sigma=min_sigma, max_sigma=max_sigma,
                                    min_hb_ratio=min_hb_ratio, min_ah_ratio=min_ah_ratio, max_uncert=max_uncert)
    result = engine.raw_components(fit_cubic, (y0, x0, z0), n_max=1)[0][0]
    if result is None:
        return None, cubic, None, None
    amp, xf, yf, zf = result[:4]
    zf = zf + z_off   # back to the full stack's own absolute plane index
    raw_fx, raw_fy = xf + xmin, yf + ymin
    sy, sx = spot_mapper.raw_to_reference((raw_fy, raw_fx), hybe, fov_matrices, modality=modality, cell=cell, resolver=resolver)
    sz = zf
    if cell is not None:
        m = modality if modality is not None else cell.reference_modality
        Hz = alignment.entry_dz(cell.matrices.get((hybe, m)))
        sz = zf + cell_z_offset(cell, hybe, m, resolver)
    shared_result = (float(sy), float(sx), float(sz), float(amp))
    # The SAME fit in this hybe's own untransformed frame. raw_fy/raw_fx
    # are already the crop origin plus the crop-local fit, clamp included,
    # so this indexes the full frame directly. Recording it changes no
    # fitted number -- it is bookkeeping, not algorithm, which is why the
    # reference implementation can carry it without ceasing to be one.
    raw_result = (float(raw_fy), float(raw_fx), float(zf), float(amp))
    return shared_result, cubic, (xf, yf, zf), raw_result


def _localize_readout_hybe(shared_xy, hybe, readout_channel, storage_path, fov, modality, cell, fov_matrices, delta,
                           spad=8, use_mixture=True, peak_bound=2.0, init_sigma_xy=1.25, init_sigma_z=2.5,
                           min_sigma=0.1, max_sigma=2.5, min_hb_ratio=1.2, min_ah_ratio=0.25, max_uncert=2.0,
                           min_sep=3.0, component_threshold=0.3, max_components=3, z_window=15,
                           z_boundary_trim=0, resolver=None):
    """
    Crops+fits ONE hybe's readout channel around the same allele anchor,
    mixture-capable (find_local_peaks_3d + fit_gaussian_mixture_3d, same
    z_window-restricted seed search refine_spot_z already uses, for the
    same reason: an unrestricted search over a 100+-plane stack treats any
    brightish voxel anywhere in it as a candidate "second component").
    Every ACCEPTED component is kept -- unlike refine_spot_z's own Z/XY-
    distance-from-representative and amplitude-ratio sibling gates (built
    to decide "is this second blob actually the same physical spot"),
    chromatin-tracing candidates within one hybe are never pruned against
    each other: a real second locus in one round (e.g. sister chromatids)
    is a genuine, independent trace value, not noise to filter out.

    delta: (dx, dy, dz) shared-frame local drift correction for this hybe,
    already computed by the caller from this hybe's own fiducial fit vs.
    the reference hybe's (see build_chromatin_trace_allele) -- added to
    every accepted component after it's mapped to the shared frame, same
    "fiducial(ref) - fiducial(round)" correction ChrTracer3_FitSpots.m
    applies, just expressed in the shared frame instead of raw pixels.

    Returns (candidates, cubic_or_None, crop_local_xyz_list) -- candidates
    is a list of (Y, X, z, amplitude) in the shared frame, already delta-
    corrected (empty if none accepted). Y FIRST: this said "(x, y, z,
    amplitude)" until 2026-08-27 and was stale -- it predates
    legacy/migrate_store_to_yx.py, which swapped the whole store to yx
    order, polymer_adj entries included. The code below has always appended
    (sy + dy, sx + dx, ...). A reimplementation trusted the docstring over
    the code and mirrored every traced position; cubic/crop_local_xyz_list are for
    display only, same "always returned, caller discards if unused"
    convention as _localize_fiducial_hybe above -- crop_local_xyz_list is
    in the SAME order as candidates (never includes a rejected component).
    """
    try:
        raw_y, raw_x = spot_mapper.reference_to_raw(shared_xy, hybe, fov_matrices, modality=modality, cell=cell, resolver=resolver)
        cubic, (ymin, xmin) = spot_mapper.crop_for_localization(storage_path, fov, hybe, readout_channel,
                                                                 (raw_y, raw_x), pad=spad, use_stack=True)
    except OSError:
        return [], None, [], []
    if cubic.size == 0:
        return [], None, [], []
    x0, y0 = raw_x - xmin, raw_y - ymin
    # Same boundary trim as _localize_fiducial_hybe (see the comment there):
    # fit on the stack minus its outermost planes, display the full crop,
    # report absolute plane indices.
    z_off = _z_boundary_offset(cubic.shape[2], z_boundary_trim)
    fit_cubic = cubic[:, :, z_off:cubic.shape[2] - z_off] if z_off else cubic
    z0 = float(np.unravel_index(np.nanargmax(fit_cubic), fit_cubic.shape)[2])

    engine = make_engine('gaussian', peak_bound=peak_bound, init_sigma_xy=init_sigma_xy,
                                    init_sigma_z=init_sigma_z, min_sigma=min_sigma, max_sigma=max_sigma,
                                    min_hb_ratio=min_hb_ratio, min_ah_ratio=min_ah_ratio, max_uncert=max_uncert,
                                    min_sep=min_sep, component_threshold=component_threshold, z_window=z_window)
    results, _seeds = engine.raw_components(fit_cubic, (y0, x0, z0),
                                            n_max=max_components if use_mixture else 1)

    dy, dx, dz = delta   # (y, x, z), rasterized order
    m = modality if modality is not None else (cell.reference_modality if cell is not None else None)
    Hz = alignment.entry_dz(cell.matrices.get((hybe, m))) if cell is not None else 0.0
    candidates, crop_local, raw_candidates = [], [], []
    for r in results:
        if r is None:
            continue
        amp, xf, yf, zf = r[:4]
        zf = zf + z_off   # back to the full stack's own absolute plane index
        raw_rx, raw_ry = xf + xmin, yf + ymin
        sy, sx = spot_mapper.raw_to_reference((raw_ry, raw_rx), hybe, fov_matrices, modality=modality, cell=cell, resolver=resolver)
        sz = zf + cell_z_offset(cell, hybe, m, resolver)
        candidates.append((float(sy + dy), float(sx + dx), float(sz + dz), float(amp)))
        # SAME ORDER as `candidates`, one raw per accepted component, and
        # carrying NO correction at all -- neither the alignment nor the
        # `delta` fiducial drift added above. That is the documented
        # asymmetry: adj-raw is one term larger for a readout than for a
        # fiducial. See AnAllele.
        raw_candidates.append((float(raw_ry), float(raw_rx), float(zf), float(amp)))
        crop_local.append((xf, yf, zf))
    return candidates, cubic, crop_local, raw_candidates


# Fiducial-fit and readout-fit params are independently configurable (see
# build_chromatin_trace_allele's own fiducial_params/readout_params) --
# this is just the shared fallback for whichever keys a caller's own dict
# doesn't set, so a caller only needs to override what it actually wants
# to differ between the two. Fiducial has no mixture mode (see
# _localize_fiducial_hybe) so it only ever uses the common subset; readout
# additionally accepts min_sep/use_mixture. Both channels default to
# ChrTracer3 FitPsf3D's own gate values (minHBratio 1.2, minAHratio
# 0.25), per explicit request (2026-08-20) -- this SUPERSEDES the earlier
# readout relaxation to 1.05 ("a readout spot is legitimately dimmer
# than a fiducial bead"); that remains re-tunable per channel in the
# Chromatin Tracing panel if real traces start losing hybes.
_DEFAULT_FIDUCIAL_FIT_PARAMS = dict(peak_bound=2.0, max_sigma=2.5, max_uncert=2.0,
                                    min_hb_ratio=1.2, min_ah_ratio=0.25)
_DEFAULT_READOUT_FIT_PARAMS = dict(_DEFAULT_FIDUCIAL_FIT_PARAMS,
                                   min_sep=3.0, use_mixture=False)


# -- parallel per-hybe tracing (see build_chromatin_trace_allele) ----------
#
# One hybe's fit is independent of every other's within a phase: phase 1
# (fiducial) needs nothing but the anchor, and phase 2 (readout) needs only
# the per-hybe delta computed between the phases. Measured on real alleles the
# work is 96% Gaussian fitting (369 ms/call) and under 4% I/O, so unlike the
# disk-bound cell-alignment pool this one is pure CPU and scales with cores.
#
# The SAME executor serves both callers: View Crop passes one for a single
# allele (the interactive click was ~85 s of fitting on one core), and the
# Fit-All worker creates one pool for its whole run and passes it into every
# allele's build -- alleles stay a serial outer loop, each fanning its ~111
# hybes across the pool, so the two paths share this code entirely and there
# is no allele-level mutate-and-merge to get wrong: build mutates the real
# allele in the PARENT; only the per-hybe fits travel.
#
# Task functions are module-level (pickling requires it) and Qt-free; every
# payload field is plain data (FrameResolver is constructed from plain data
# by design -- see its own docstring -- and pickles).

def _init_tracing_worker():
    """One cv2 thread per child -- the standard oversubscription guard; the
    fits are scipy, but crop reads go through cv2-adjacent code paths."""
    cv2.setNumThreads(1)


def _fiducial_task(payload):
    (shared_xy, hybe, channel, storage_path, fov, modality, cell, fov_matrices,
     kwargs, resolver, want_debug) = payload
    result, cubic, centroid, raw_result = _localize_fiducial_hybe(
        shared_xy, hybe, channel, storage_path, fov, modality, cell, fov_matrices,
        resolver=resolver, **kwargs)
    # cubic is display-only (~100+ KB per hybe); never ship it back for a
    # batch run that would discard it. The raw tuple is 4 floats and must
    # travel, or every parallel run silently loses the raw frame.
    return hybe, result, (cubic if want_debug else None), centroid, raw_result


def _readout_task(payload):
    (shared_xy, hybe, channel, storage_path, fov, modality, cell, fov_matrices,
     delta, kwargs, resolver, want_debug) = payload
    candidates, cubic, crop_local, raw_candidates = _localize_readout_hybe(
        shared_xy, hybe, channel, storage_path, fov, modality, cell, fov_matrices,
        delta, resolver=resolver, **kwargs)
    return (hybe, candidates, (cubic if want_debug else None), crop_local,
            raw_candidates)


def max_tracing_workers(hard_ceiling=32):
    """CPU-bound (96% Gaussian fitting), so the FOV-alignment pool's own
    ceiling applies rather than the disk-bound cell pool's lower one. Two
    cores stay reserved for the GUI thread and the coordinator."""
    return max(1, min((os.cpu_count() or 4) - 2, hard_ceiling))


def build_chromatin_trace_allele(allele, hybes, reference_hybe, hybe_fiducial_channels, hybe_readout_channels,
                                 storage_path, fov, modality, cell, fov_matrices, max_fiducial_drift=5.0,
                                 max_fiducial_drift_z=10.0,
                                 spad=8, z_window=15, fiducial_params=None, readout_params=None,
                                 collect_debug=False, resolver=None, z_boundary_trim=0, executor=None,
                                 append=False):
    """
    Fills in allele.fiducial_trace_adj/polymer_adj/rejected_hybes for every hybe in
    `hybes` (folder names) -- full replace by default, same "re-run
    overwrites" convention as _replace_cell_spots/_replace_fov_unassigned_
    spots elsewhere in this app.

    append=True (the mid-ingestion delta mode -- see MainWindow's batch
    Fit All FOVs Append option): the three dicts are NOT reset, only the
    requested `hybes` are (re)fitted and merged in, and the reference
    hybe's fiducial baseline is REUSED from the allele's stored
    fiducial_trace_adj when present (a fiducial baseline is a physical fact
    about the reference stack -- new hybes landing on disk do not change
    it; AnAllele.save round-trips fiducial_trace_adj, so the baseline
    survives sessions). When the stored baseline is missing and
    reference_hybe is not among `hybes`, its fiducial alone is fitted as
    an extra so phase 2's delta gate has a real baseline rather than
    rejecting everything with 'reference hybe fiducial not found'.

    Two phases:

    1. Fiducial-only fit for every hybe (_localize_fiducial_hybe) ->
       allele.fiducial_trace_adj[hybe].
    2. baseline = allele.fiducial_trace_adj[reference_hybe]; for every other
       hybe, delta = baseline - fiducial_trace_adj[hybe] (shared frame); reject
       (allele.rejected_hybes[hybe] = reason, no readout fit attempted) when
       either fiducial is missing, or the XY magnitude of delta exceeds
       max_fiducial_drift -- per explicit request, evaluated in the shared
       frame, never against raw_coordinates (a raw-frame distance would be
       comparing two different hybes' own native pixel grids, not the same
       physical space). XY only (not a combined XY+Z magnitude), matching
       this pipeline's own established convention elsewhere (compute_cell_
       alignment / the mixture-sibling QC gates each bound XY and Z
       separately, never combined) -- otherwise fits the readout channel
       (_localize_readout_hybe) and stores its delta-corrected candidates
       in allele.polymer_adj[hybe].

    hybe_fiducial_channels/hybe_readout_channels: {hybe: channel(int)} --
    one entry per hybe each, independently resolved from that hybe's own
    ExperimentLayout record (hybe_record['fiducial_channel'] and whichever
    of hybe_record['channels'] isn't the fiducial one). Deliberately NOT
    derived from allele.anchor_channel (the seed spot's own channel) --
    per confirmed real bug/explicit correction: an allele's seed spot only
    LOCATES the allele-frame (x, y, z); it does not determine which
    channel gets traced. Building the readout channel from anchor_channel
    meant an allele seeded from a fiducial-channel spot (a legitimate,
    explicitly-supported choice -- "no need to be fiducial channel" was
    never "must not be") silently traced the FIDUCIAL channel through
    every hybe instead of the real readout channel, rendering visually
    identical fiducial/readout grids. Both dicts can differ hybe to hybe.

    spad/max_fiducial_drift apply identically to both fiducial and readout
    fitting -- crop placement and the drift-rejection gate are cross-
    cutting, not something fiducial vs. readout fitting would ever want to
    disagree about. z_window only affects readout (its mixture seed-search
    Z-window -- fiducial has no mixture mode, see _localize_fiducial_hybe).
    fiducial_params (subset of peak_bound, max_sigma, max_uncert,
    min_hb_ratio, min_ah_ratio) and readout_params (those five plus
    min_sep, use_mixture) are independently configurable, per explicit
    request -- a fiducial bead and a real genomic-locus probe can have
    genuinely different brightness/PSF characteristics worth tuning
    separately, though only readout ever needs multi-component search: a
    fiducial's whole purpose is ONE per-hybe drift-correction anchor.
    Missing keys in either dict fall back to _DEFAULT_FIDUCIAL_FIT_PARAMS/
    _DEFAULT_READOUT_FIT_PARAMS's own values, so a caller only needs to
    override what it actually wants to differ.

    collect_debug=False (default, the "Fit All FOVs" batch path -- no
    reason to hold every hybe's raw crop in memory for a run that never
    displays them, or to pay for a readout crop+fit on a hybe already
    known to be rejected): returns allele alone, and a rejected hybe's
    readout channel is never even cropped. collect_debug=True (the
    "Preview One Allele" path, see canvas.chromatin_trace_grid_displayer):
    ALWAYS crops+fits the readout channel for every hybe, accepted or
    rejected -- per explicit request, a crop should always be visible
    (map the allele's coordinate into that hybe's own frame and crop
    nearby); only the fitted position marker depends on whether the gate
    passed (delta stays uncorrected, (0,0,0), for a rejected hybe's
    preview crop). Also returns a {hybe: {'fiducial_cubic',
    'fiducial_centroid', 'readout_cubic', 'readout_centroids'}} dict, each
    cubic/centroid already shaped exactly as canvas.spot_fit_status.
    draw_spot_fit_status expects (centroid=None when nothing was accepted
    -- that function's own "plain = missing" convention).

    Mutates allele in place either way.
    """
    # raw/adj is a property of AnAllele, not of whichever engine filled
    # it, so v1 maintains both exactly as v2 does -- no engine-shaped hole
    # in the container's contract. Recording the raw frame changes no
    # fitted number (it is the same fit, expressed before the matrix), so
    # this module remains the reference implementation every v2 claim was
    # measured against.
    #
    # Clearing raw alongside adj, per-hybe in append mode too, is what
    # keeps the pair honest: a hybe whose adj is re-derived must not keep
    # a raw from the previous run.
    if not append:
        allele.fiducial_trace_adj = {}
        allele.polymer_adj = {}
        allele.fiducial_trace_raw = {}
        allele.polymer_raw = {}
        allele.rejected_hybes = {}
    else:
        # merge mode: keep what earlier passes established; a hybe in
        # `hybes` still gets fully re-derived below (its own entries are
        # overwritten), only hybes OUTSIDE the request are left alone.
        for hybe in hybes:
            allele.rejected_hybes.pop(hybe, None)
            allele.polymer_adj.pop(hybe, None)
            allele.polymer_raw.pop(hybe, None)
            allele.fiducial_trace_raw.pop(hybe, None)
    shared_xy = (allele.coordinate[0], allele.coordinate[1])
    debug = {} if collect_debug else None

    fiducial_kwargs = dict(spad=spad, z_boundary_trim=z_boundary_trim,
                           **{**_DEFAULT_FIDUCIAL_FIT_PARAMS, **(fiducial_params or {})})
    readout_kwargs = dict(spad=spad, z_window=z_window, z_boundary_trim=z_boundary_trim,
                          **{**_DEFAULT_READOUT_FIT_PARAMS, **(readout_params or {})})

    # -- phase 1: fiducial fits. Independent per hybe, so pooled whenever an
    # executor is supplied -- byte-identical either way (deterministic scipy
    # least-squares, no RNG anywhere in this path); as_completed only changes
    # the order dict keys are INSERTED, and every consumer looks up by hybe.
    fid_todo = []
    for hybe in hybes:
        if debug is not None:
            debug[hybe] = {'fiducial_cubic': None, 'fiducial_centroid': None,
                           'readout_cubic': None, 'readout_centroids': None}
        fid_channel = hybe_fiducial_channels.get(hybe)
        if fid_channel is None:
            allele.fiducial_trace_adj[hybe] = None
            allele.fiducial_trace_raw[hybe] = None
            allele.rejected_hybes[hybe] = 'no fiducial channel configured'
            continue
        fid_todo.append((hybe, fid_channel))
    if append and reference_hybe not in hybes and allele.fiducial_trace_adj.get(reference_hybe) is None:
        # the delta gate below needs the reference baseline; in append
        # mode it normally comes stored from the earlier pass, but a
        # first-ever append (or a legacy allele saved without it) must
        # fit it once -- fiducial only, never gated/traced itself here.
        ref_channel = hybe_fiducial_channels.get(reference_hybe)
        if ref_channel is not None:
            if debug is not None:
                debug[reference_hybe] = {'fiducial_cubic': None, 'fiducial_centroid': None,
                                         'readout_cubic': None, 'readout_centroids': None}
            fid_todo.append((reference_hybe, ref_channel))

    def _store_fiducial(hybe, fid_result, fid_cubic, fid_centroid, fid_raw=None):
        allele.fiducial_trace_adj[hybe] = fid_result
        allele.fiducial_trace_raw[hybe] = fid_raw
        if debug is not None:
            debug[hybe]['fiducial_cubic'] = fid_cubic
            debug[hybe]['fiducial_centroid'] = fid_centroid

    if executor is not None and len(fid_todo) > 1:
        futures = [executor.submit(_fiducial_task,
                                   (shared_xy, hybe, ch, storage_path, fov, modality, cell,
                                    fov_matrices, fiducial_kwargs, resolver, debug is not None))
                   for hybe, ch in fid_todo]
        for future in as_completed(futures):
            _store_fiducial(*future.result())
    else:
        for hybe, ch in fid_todo:
            fid_result, fid_cubic, fid_centroid, fid_raw = _localize_fiducial_hybe(
                shared_xy, hybe, ch, storage_path, fov, modality, cell, fov_matrices,
                resolver=resolver, **fiducial_kwargs)
            _store_fiducial(hybe, fid_result, fid_cubic, fid_centroid, fid_raw)

    baseline = allele.fiducial_trace_adj.get(reference_hybe)
    # -- phase 2a: the drift gate. Cheap arithmetic, stays serial; produces
    # per hybe (reject_reason, delta, readout_channel) so the fits below can
    # run detached from the gating.
    gate = {}
    for hybe in hybes:
        # reject_reason may already be set from phase 1 ('no fiducial
        # channel configured'); otherwise derive it from this hybe's own
        # fiducial vs. the reference's.
        reject_reason = allele.rejected_hybes.get(hybe)
        fid = allele.fiducial_trace_adj.get(hybe)
        delta = (0.0, 0.0, 0.0)
        if reject_reason is None:
            if baseline is None:
                reject_reason = 'reference hybe fiducial not found'
            elif fid is None:
                reject_reason = 'fiducial not found'
            else:
                dy, dx, dz = baseline[0] - fid[0], baseline[1] - fid[1], baseline[2] - fid[2]
                delta = (dy, dx, dz)
                drift = float(np.hypot(dx, dy))
                if drift > max_fiducial_drift:
                    reject_reason = f'drift {drift:.1f}px > max {max_fiducial_drift}px'
                # Z gated separately, in planes -- per explicit request:
                # a fiducial fit can pass the XY bound while landing on
                # entirely different content in depth (confirmed real
                # case: a barcode round's weak fiducial fit 20 planes
                # from the reference at only 1.4px XY drift), and using
                # such a fit would "correct" every readout in that hybe
                # by a bogus dz.
                elif abs(dz) > max_fiducial_drift_z:
                    reject_reason = f'z drift {abs(dz):.1f} planes > max {max_fiducial_drift_z}'

        readout_channel = hybe_readout_channels.get(hybe)
        if readout_channel is None and reject_reason is None:
            reject_reason = 'no readout channel configured'
        gate[hybe] = (reject_reason, delta, readout_channel)

    # -- phase 2b: the readout fits the gate decided to pay for. Batch mode
    # (collect_debug=False, nothing ever displays this crop) skips a hybe
    # already known to be rejected -- no reason to pay for the readout
    # crop+fit. Preview mode (collect_debug=True) always crops+fits every
    # hybe regardless of accept/reject -- per explicit request, a crop
    # should always be visible; only the FIT/marker depends on whether the
    # gate passed. delta stays (0, 0, 0) (uncorrected) whenever it couldn't
    # be computed. A missing readout channel means there is nothing to crop
    # at all -- always skipped, even in preview mode.
    ro_todo = [hybe for hybe in hybes
               if gate[hybe][2] is not None
               and not (gate[hybe][0] is not None and debug is None)]
    ro_results = {}
    if executor is not None and len(ro_todo) > 1:
        futures = [executor.submit(_readout_task,
                                   (shared_xy, hybe, gate[hybe][2], storage_path, fov, modality,
                                    cell, fov_matrices, gate[hybe][1], readout_kwargs, resolver,
                                    debug is not None))
                   for hybe in ro_todo]
        for future in as_completed(futures):
            hybe, candidates, cubic, crop_local, raw_cands = future.result()
            ro_results[hybe] = (candidates, cubic, crop_local, raw_cands)
    else:
        for hybe in ro_todo:
            candidates, cubic, crop_local, raw_cands = _localize_readout_hybe(
                shared_xy, hybe, gate[hybe][2], storage_path, fov, modality, cell, fov_matrices,
                gate[hybe][1], resolver=resolver, **readout_kwargs)
            ro_results[hybe] = (candidates, cubic, crop_local, raw_cands)

    # -- phase 2c: bookkeeping, in the caller's own hybe order --
    for hybe in hybes:
        reject_reason, _delta, readout_channel = gate[hybe]
        if hybe not in ro_results:
            if readout_channel is None or (reject_reason is not None and debug is None):
                allele.rejected_hybes[hybe] = reject_reason
            continue
        candidates, readout_cubic, readout_centroids, raw_cands = ro_results[hybe]
        if debug is not None:
            debug[hybe]['readout_cubic'] = readout_cubic
            debug[hybe]['readout_centroids'] = readout_centroids or None
        if reject_reason is not None:
            allele.rejected_hybes[hybe] = reject_reason
            continue
        if candidates:
            allele.polymer_adj[hybe] = candidates
            allele.polymer_raw[hybe] = raw_cands
        else:
            allele.rejected_hybes[hybe] = 'no readout peak accepted'
    return (allele, debug) if collect_debug else allele

def refine_spots_batch(targets, storage_path, fov, channel, hybe, modality,
                       params, fov_matrices, resolver, want_grid=False):
    """
    Refine Z for a whole SELECTION in one call, in order -- the
    child-process entry point behind Spot Localization's Run and View.

    targets: [(spot, cell_or_None), ...]. The loop must stay sequential
    and in this order: `claimed_positions` accumulates as it goes, so two
    distinct spots sharing an ambiguous crop cannot collapse onto the
    same blob (see refine_spot_z). Parallelising per spot would silently
    change the result, so the whole batch is one job.

    Returns [(index, new_coordinate, new_raw, mixture_centroids, cubic,
    centroid), ...]; cubic/centroid are None unless want_grid (View needs
    them to draw the fit-status grid, Run does not and they are the bulky
    part of the payload). Index-keyed because the caller's spot objects
    live in ANOTHER process -- it maps results back onto its own objects.

    Runs where it is called from a process pool rather than a thread
    because every crop here reads the raw stack, and h5py serialises
    every HDF5 call in a process behind one lock -- as a thread this
    starves the GUI's own image loads (measured 16.5 ms -> 2043 ms).
    """
    claimed_positions = []
    out = []
    for i, (spot, cell) in enumerate(targets):
        new_coordinate, new_raw, cubic, centroid, _extra, mixture_centroids = refine_spot_z(
            spot, storage_path, fov, channel, hybe=hybe, cell=cell, modality=modality,
            spad=params['spad'], peak_bound=params['peak_bound'],
            max_sigma=params['max_sigma'], max_uncert=params['max_uncert'],
            min_hb_ratio=params['min_hb_ratio'], min_ah_ratio=params['min_ah_ratio'],
            min_sep=params['min_sep'], claimed_positions=claimed_positions,
            use_mixture=params['multi_mode'], z_window=params['z_window'],
            fov_matrices=fov_matrices, resolver=resolver)
        if new_raw is not None:
            claimed_positions.append((new_raw[0], new_raw[1]))
        out.append((i, new_coordinate, new_raw, mixture_centroids,
                    cubic if want_grid else None,
                    centroid if want_grid else None))
    return out
