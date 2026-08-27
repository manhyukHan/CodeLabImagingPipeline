"""
The v2 chromatin-tracing path, whole, in one additive module.

WHY A SEPARATE FILE
-------------------
v1 (localization.build_chromatin_trace_allele) is a direct port of
ChrTracer3's FitPsf3D and is the REFERENCE IMPLEMENTATION: every v2 claim
was established by measuring against it, and that stays possible only
while it can still be run unchanged. So nothing here edits it. This
module adds a second path with the same contract, and the panel chooses
between them.

WHAT v2 CHANGES, AND WHAT EACH CHANGE WAS WORTH
-----------------------------------------------
Measured on 48 real MP58 alleles (tools/v2_variants.py), one change at a
time, scored by occupancy -- intensity at the fitted centroid over
intensity at the argmax, both above local background:

    BOX not pillar, + linear background   occupancy 0.373 -> 0.806,
                                          blank-region fits 31% -> 4%
    box placed at the CONSENSUS depth     fiducial z spread 1.00 -> 0.56 planes
    intensity-centroid seed               v1 0.354 -> 0.597
    loose separate bounds (5 px / 10 pl)  at-bound 75-100% -> 2-13%
    calibrated PSF, sigma fixed           fiducial 0.803 -> 0.838, 37% faster

Final: occupancy 0.354 -> 0.838, blank-region fits 34% -> 0%.

Poisson MLE is deliberately NOT used: synthetic Poisson data predicted
15-27% better axial precision, and on real crops it delivered nothing
measurable while costing 20-40% more time.

THE TWO CHANNELS ARE NOT THE SAME MEASUREMENT
---------------------------------------------
v1 already kept per-channel gate values, and v2 needs the separation
MORE, not less, because the two channels image different kinds of object:

    readout    a single locus: a point source. Its shape is optical and
               near-universal (four experiments over a 64x range in
               genomic scope agreed to 20 nm), so its sigma is FIXED from
               the calibrated PSF and never fitted per spot.
    fiducial   the whole traced region: an extended object. Its width is
               not even well defined -- fit a Gaussian and the recovered
               sigma follows the FIT WINDOW as sigma ~ r^0.5 with no
               plateau. So its sigma is FREE, generously bounded, and
               treated as a QC observation rather than a measurement.

Consequently a readout bound applied to a fiducial rails the fit, and a
railed fit reports the bound rather than a position -- which then becomes
the drift correction for every readout in that round.

WHAT THE FIDUCIAL IS FOR
------------------------
One number per hybe: delta = reference_position - this_position, which
corrects every readout in the round. Image-matching the two crops instead
was tried and is WORSE -- 14.8% on the median of 305 replicate pairs,
closer on only 37% of them. An ill-defined WIDTH does not imply a bad
CENTRE: a symmetric model on a roughly symmetric object still gives an
unbiased centre. Registration is kept as a QC cross-check (`qc_shift`),
where its better tail behaviour is useful, not as the estimator.
"""
import numpy as np

from codelab_pipeline.localization import fit3d_um as U
from codelab_pipeline.localization import fit3d_mle as M

DEFAULT_VOXEL_UM = (0.208, 0.208, 0.2)

# -- gates, per channel, in NANOMETRES ------------------------------------
#
# Nanometres and not pixels, with lateral and axial INDEPENDENT. v1 wrote
# its axial gate as `2 * max_uncert` in pixels, assuming a plane is twice
# a pixel. Here a plane is 0.2 um and a pixel 0.208 -- they differ by 4%,
# not by 2x, and the assumption is wrong on any other microscope too.
#
# Every v1 gate quantity CHANGED MEANING when the background became local,
# so none of the v1 constants transfer (tools/gate_sweep_v2.py). What is
# on by default is only what was measured to be free:
#
#   at_bound        295/311 pairs at 0.218 um against 311 at 0.294
#                   ungated -- keeps 95% of pairs, improves the median
#                   26%, and has NO THRESHOLD TO CHOOSE. A parameter that
#                   stopped on its constraint is the bound you supplied,
#                   not a measurement, and its Jacobian CI (which assumes
#                   an interior optimum) does not describe it either.
#   occupancy       the tunable one, and the best-behaved: it degrades
#                   smoothly instead of falling off a cliff.
#
# Deliberately absent, both v1 heritage:
#   min_hb_ratio    untunable -- 311 pairs at 1.0, 40 at 1.2, ~10 by 1.6.
#                   A 0.1 change swings coverage by an order of magnitude.
#   min_ah_ratio    dominated by occupancy, which measures the same intent
#                   properly.
#
# The uncertainty gates default OFF rather than to an inherited number.
# At v1's own coverage they reach 61-79 nm against v1's 183 nm, but those
# thresholds were derived on ONE dataset and a silently inherited constant
# is how the v1 gates became wrong in the first place.
FIDUCIAL_GATES = {
    'reject_at_bound': True,
    # POSITION only. A fiducial's sigma has no value to converge to -- a
    # single-crop fit walks it to whatever ceiling exists, measured
    # directly: every one of 14 consecutive HoxA rounds came back at
    # exactly 600 nm, the ceiling, while occupancy stayed 0.44-0.69, i.e.
    # the centroid was ON the object the whole time. (The survey got
    # stable widths only because psf.calibrate fits ONE shared shape
    # across many crops with per-crop nuisances; alone, there is nothing
    # to pin it.) Rejecting those fits discards a usable centroid over an
    # undefined width -- and the centroid is the only thing the fiducial
    # is for.
    #
    # Position at a bound is different and stays fatal: it means the fit
    # could not reach the emitter, so the value IS the bound and the drift
    # correction built from it is fiction.
    'at_bound_fatal': ('y', 'x', 'z'),
    'min_occupancy': 0.25,      # looser: an extended object spreads its peak
    'max_uncert_xy_nm': None,
    'max_uncert_z_nm': None,
}
READOUT_GATES = {
    'reject_at_bound': True,
    # None = ANY parameter at a bound is fatal. Safe here because the
    # readout's sigma is FIXED from the calibrated PSF and so cannot be at
    # a bound at all; anything that rails is a position or a background,
    # and both matter.
    'at_bound_fatal': None,
    'min_occupancy': 0.40,
    'max_uncert_xy_nm': None,
    'max_uncert_z_nm': None,
}

# -- fit domains ----------------------------------------------------------
#
# A BOX, not a pillar. The tracing crop is bounded in XY but takes the
# full slab in Z (17x17x110): the emitter is a few hundred voxels against
# ~34k of out-of-focus content, so least squares spends its position and
# sigma parameters describing background. That single change moved
# occupancy 0.373 -> 0.806.
FIDUCIAL_FIT_RADIUS_UM = (1.0, 1.0, 3.0)
# THE SAME domain for both channels, because that is what was measured.
# tools/fiducial_match.py, tools/v2_variants.py and tools/gate_sweep_v2.py
# all use (1.0, 1.0, 3.0) for readouts as well as fiducials, so every
# quoted readout number -- occupancy 0.561 -> 0.677, the at_bound filter's
# 295/311 pairs at 0.218 um -- is a measurement at this radius.
#
# It briefly read (0.8, 0.8, 2.0) here, which is psf.calibrate's default
# CALIBRATION window (psf.py:355), not a tracing domain: 1029 voxels
# instead of 2511, 41% of the data every readout number was measured on,
# while still fitting the same 8-11 parameters. It also broke the position
# bounds, which are 1.04 um lateral and 2.0 um axial and must sit INSIDE
# the domain -- at an axial radius of 2.0 um the bound equalled the domain.
READOUT_FIT_RADIUS_UM = (1.0, 1.0, 3.0)

# Loose, and SEPARATE laterally and axially. Tight bounds put 75-100% of
# fits on a constraint, which is why every dz in a fiducial overlay used
# to print as a whole number: fitted z = integer argmax +/- exactly the
# bound.
FIDUCIAL_PEAK_BOUND_UM = 1.04        # ~5 px
FIDUCIAL_PEAK_BOUND_Z_UM = 2.0       # ~10 planes
READOUT_PEAK_BOUND_UM = 1.04
READOUT_PEAK_BOUND_Z_UM = 2.0

# The fiducial's sigma is FREE and must be allowed to be large: measured
# 257-574 nm depending on the fit window, against v1's 520 nm default
# ceiling. A fiducial fitted against a readout-sized bound rails.
# THE VALUES EVERY v2 MEASUREMENT WAS TAKEN WITH. tools/fiducial_match.py
# used max_sigma_xy_um=3.0 / max_sigma_z_um=6.0, and so did the variants
# ladder and the gate sweep -- so "at-bound fell to 2-13%" was measured
# with sigma effectively UNBOUNDED, where at_bound almost always meant a
# POSITION had railed.
#
# Tightening this is tempting and was tried twice here (1.20, then 0.60 on
# the argument that a Gaussian as wide as its own fit window is
# indistinguishable from the background beside it). The argument is sound
# in principle and wrong in practice: sigma simply pins to whatever
# ceiling exists -- 14 consecutive HoxA rounds returned exactly 600 nm --
# because a single-crop fiducial fit has no width to converge to at all.
# Moving the ceiling only moves the number, and quietly replaces a
# validated configuration with an untested one.
#
# So the ceiling stays where every measurement was taken, and sigma being
# at a bound is handled where it belongs: it is not fatal for a fiducial
# (see FIDUCIAL_GATES 'at_bound_fatal'), because the fiducial's width is
# not a measurement and its centroid is what the round is for.
FIDUCIAL_MAX_SIGMA_XY_UM = 3.00
FIDUCIAL_MAX_SIGMA_Z_UM = 6.00

# A LOWER bound at the optical limit, not at the fitter's default 0.02 um.
# 20 nm is a quarter of the smallest width this microscope can produce, so
# it is not a bound the truth can sit near -- it is only reachable by the
# fit collapsing onto a single bright voxel, which is precisely what a
# noisy box invites. Measured on HoxA before this was set: 33 of 45 hybes
# in one allele rejected as `at bound (sigma_y, sigma_x, offset)`, the
# signature of exactly that collapse. 70 nm is psf.plausible's own lateral
# floor, from the diffraction limit.
FIDUCIAL_MIN_SIGMA_UM = 0.070


class V2Params(object):
    """Everything the v2 path needs that v1 did not have.

    Built once per run from the panel, so a trace and a calibration can
    never disagree about the voxel size or the PSF in play.
    """

    def __init__(self, voxel_um=DEFAULT_VOXEL_UM, psf_family=None,
                 psf_shape=None, psf_label='', fiducial_gates=None,
                 readout_gates=None, qc_shift=True):
        self.voxel_um = tuple(float(v) for v in voxel_um)
        # None = no calibrated PSF, so the readout sigma is fitted per spot
        # like the fiducial's. Supported, but it gives up both the accuracy
        # and the 37% speed the fixed shape buys.
        self.psf_family = psf_family
        self.psf_shape = tuple(psf_shape) if psf_shape else None
        self.psf_label = psf_label
        self.fiducial_gates = dict(FIDUCIAL_GATES, **(fiducial_gates or {}))
        self.readout_gates = dict(READOUT_GATES, **(readout_gates or {}))
        self.qc_shift = bool(qc_shift)

    @property
    def has_psf(self):
        return bool(self.psf_family and self.psf_shape)

    @classmethod
    def from_panel(cls, params, storage_path=None):
        """Build from ChromatinTracingPanel.params(), resolving the PSF.

        The PSF comes from the copy INSTALLED in the store, not from the
        library: a run must be reproducible from its own store after the
        library has moved on.
        """
        from codelab_pipeline.localization import psf_library as LIB
        voxel = params.get('voxel_um', DEFAULT_VOXEL_UM)
        fam = shape = None
        label = params.get('readout_psf', '')
        doc = LIB.installed(storage_path) if storage_path else None
        if doc is None and label:
            doc = LIB.read(label)
        got = LIB.shape_tuple(doc) if doc else None
        if got:
            fam, shape = got
            label = doc.get('installed_from') or doc.get('label') or label
        return cls(voxel_um=voxel, psf_family=fam, psf_shape=shape,
                   psf_label=label)

    def describe(self):
        if not self.has_psf:
            return 'v2, readout sigma FREE (no calibrated PSF installed)'
        sxy = self.psf_shape[0] * 1000.0
        return (f'v2, readout PSF {self.psf_label!r} ({self.psf_family}, '
                f'sigma_xy {sxy:.0f} nm), voxel {self.voxel_um}')


# -- depth placement ------------------------------------------------------

def consensus_native_z(cubes_by_hybe, z_offsets):
    """
    {hybe: expected native z} -- where this allele should sit, in depth,
    in EACH hybe's own stack.

    The alleles here have z = 0: they are detected on MIPs and never
    3D-refined, so there is no anchor depth to place a box with. It is
    derived instead, and the derivation must happen in the SHARED frame:
    crops are XY-transformed but take the full Z slab, so their z is
    NATIVE, and two hybes' fiducials differ by the cell-level z offset --
    up to 21 planes in MP58.

        shared_z(h) = argmax_z(h) + offset(h)
        baseline    = median over h          <- only meaningful HERE
        native_z(h) = baseline - offset(h)

    Fit-free on purpose. Measured 1.05 planes of placement error at
    0.09 s, against 4.65 planes at 106 s for a per-hybe pillar fit: the
    expensive route is WORSE, because a pillar fit is exactly the
    degenerate fit this module exists to avoid.

    A single shared depth for every hybe -- the obvious shortcut -- is
    wrong in a way that still produces numbers: it put a comparison tool
    at a 1.78 um median pair distance where the correct placement gives
    0.41 um, because seeds 21 planes out cannot be reached from within
    the axial bound and the fit rails instead.
    """
    shared = []
    for hybe, cube in cubes_by_hybe.items():
        off = z_offsets.get(hybe)
        if cube is None or off is None or not np.isfinite(cube).any():
            continue
        z = float(np.unravel_index(int(np.nanargmax(cube)), cube.shape)[2])
        shared.append(z + float(off))
    if not shared:
        return {}
    baseline = float(np.median(shared))
    return {h: baseline - float(o) for h, o in z_offsets.items()}


# -- fitting --------------------------------------------------------------

def _seed(cube, z_centre, voxel_um, half=(5, 5, 10)):
    """Intensity-weighted centroid, falling back to the crop centre.

    Both engines start here, per explicit request. It is worth more to v1
    (0.354 -> 0.597) than to v2 (0.799 -> 0.818), because v2's boxed
    domain has already removed most of what a bad seed used to chase.
    """
    cy = (cube.shape[0] - 1) / 2.0
    cx = (cube.shape[1] - 1) / 2.0
    zc = float(np.clip(z_centre, 0, cube.shape[2] - 1))
    got = U.intensity_centroid(cube, (cy, cx, zc), half, voxel_um)
    return got if got is not None else (cy, cx, zc)


def fit_fiducial(cube, z_centre, p):
    """Fiducial: sigma FREE, generous bounds, linear background.

    Free because there is no fiducial PSF to fix it to -- the fiducial is
    an extended object and its Gaussian sigma tracks the fit window rather
    than the object (sigma ~ r^0.5, no plateau, measured on four
    experiments). The number that comes back is a QC observation about
    this hybe, never a shape to reuse.
    """
    if cube is None or not np.isfinite(cube).any():
        return None
    sy, sx, sz = _seed(cube, z_centre, p.voxel_um)
    return U.fit_gaussian_3d_um(
        cube, sy, sx, sz, voxel_um=p.voxel_um,
        peak_bound_um=FIDUCIAL_PEAK_BOUND_UM,
        peak_bound_z_um=FIDUCIAL_PEAK_BOUND_Z_UM,
        min_sigma_um=FIDUCIAL_MIN_SIGMA_UM,
        max_sigma_xy_um=FIDUCIAL_MAX_SIGMA_XY_UM,
        max_sigma_z_um=FIDUCIAL_MAX_SIGMA_Z_UM,
        fit_radius_um=FIDUCIAL_FIT_RADIUS_UM,
        background='linear', apply_gates=False)


def fit_readout(cube, z_centre, p):
    """Readout: sigma FIXED from the calibrated PSF when there is one.

    Fixing it is worth accuracy AND time (occupancy 0.803 -> 0.838, 37%
    faster) because the shape is a property of the microscope and the
    probe design, not of this spot -- so fitting it per spot spends
    parameters on something already known, and lets a noisy crop buy
    residual with an implausible width.

    noise='gaussian' -- least squares, NOT Poisson MLE. The MLE was
    measured to give nothing on real crops for 20-40% more time.
    """
    if cube is None or not np.isfinite(cube).any():
        return None
    sy, sx, sz = _seed(cube, z_centre, p.voxel_um)
    if not p.has_psf:
        return U.fit_gaussian_3d_um(
            cube, sy, sx, sz, voxel_um=p.voxel_um,
            peak_bound_um=READOUT_PEAK_BOUND_UM,
            peak_bound_z_um=READOUT_PEAK_BOUND_Z_UM,
            fit_radius_um=READOUT_FIT_RADIUS_UM,
            background='linear', apply_gates=False)
    return M.fit_gaussian_3d_mle(
        cube, sy, sx, sz, voxel_um=p.voxel_um,
        family=p.psf_family, shape_params=p.psf_shape, free_shape=False,
        noise='gaussian',
        peak_bound_um=READOUT_PEAK_BOUND_UM,
        peak_bound_z_um=READOUT_PEAK_BOUND_Z_UM,
        fit_radius_um=READOUT_FIT_RADIUS_UM,
        background='linear', apply_gates=False)


# -- quality and gating ---------------------------------------------------

def occupancy(cube, fit, voxel_um=DEFAULT_VOXEL_UM):
    """Intensity at the fitted centroid over intensity at the argmax,
    both above local background. 1.0 = the fit is ON the emitter, <= 0 =
    it is in background.

    Needs no ground truth, which is what makes it usable as a gate: the
    replicate score is the SCORE, and gating on it would be circular.
    """
    if fit is None or cube is None:
        return float('nan')
    if not np.isfinite(cube).any():
        return float('nan')
    # A LOCAL PLANE, evaluated separately at the argmax and at the fitted
    # centroid -- NOT one scalar median for the whole crop.
    #
    # This is the definition tools/fit_quality.py used to produce every
    # occupancy number on record (0.354 -> 0.838 fiducial, 0.561 -> 0.677
    # readout), so the 0.25 / 0.40 thresholds are calibrated against it and
    # only against it. A global median measures something else: over a
    # pillar with a real intensity gradient it sits far below the local
    # level near the emitter and far above it in a dim corner, so the ratio
    # is inflated where the gradient is positive and depressed where it is
    # negative. Gating a differently-defined quantity with an inherited
    # threshold is how the v1 gates went wrong in the first place.
    bg = _local_background_plane(cube, voxel_um)
    iy, ix, iz = np.unravel_index(int(np.nanargmax(cube)), cube.shape)
    fy = int(np.clip(round(fit.y), 0, cube.shape[0] - 1))
    fx = int(np.clip(round(fit.x), 0, cube.shape[1] - 1))
    fz = int(np.clip(round(fit.z), 0, cube.shape[2] - 1))
    denom = float(cube[iy, ix, iz]) - float(bg[iy, ix, iz])
    if not np.isfinite(denom) or denom <= 0:
        return float('nan')
    return (float(cube[fy, fx, fz]) - float(bg[fy, fx, fz])) / denom


def _local_background_plane(cube, voxel_um=DEFAULT_VOXEL_UM,
                            radius_um=(1.0, 1.0, 3.0)):
    """b0 + by*y + bx*x + bz*z fitted to the SHELL around the argmax.

    The shell is inside the fit radius but outside a core region, so the
    emitter cannot lift its own background and then be measured against
    it. Falls back to the global median only when the shell is too small
    to constrain a plane -- which is the degenerate case, not the normal
    one. Ported verbatim from tools/fit_quality.local_background so the
    gate and the measurement cannot drift apart.
    """
    dy, dx, dz = voxel_um
    iy, ix, iz = np.unravel_index(int(np.nanargmax(cube)), cube.shape)
    Y, X, Z = np.indices(cube.shape)
    Y, X, Z = Y * dy, X * dx, Z * dz
    cy, cx, cz = iy * dy, ix * dx, iz * dz
    ry, rx, rz = radius_um
    inside = ((np.abs(Y - cy) <= ry) & (np.abs(X - cx) <= rx)
              & (np.abs(Z - cz) <= rz))
    core = ((np.abs(Y - cy) <= 0.45) & (np.abs(X - cx) <= 0.45)
            & (np.abs(Z - cz) <= 1.2))
    shell = inside & ~core & np.isfinite(cube)
    if shell.sum() < 50:
        return np.full(cube.shape, float(np.nanmedian(cube)))
    A = np.column_stack([np.ones(int(shell.sum())), Y[shell], X[shell], Z[shell]])
    try:
        c, *_ = np.linalg.lstsq(A, cube[shell], rcond=None)
    except Exception:
        return np.full(cube.shape, float(np.nanmedian(cube)))
    return c[0] + c[1] * Y + c[2] * X + c[3] * Z


def gate(fit, cube, gates, voxel_um=DEFAULT_VOXEL_UM):
    """(passed, reason) -- reason is None when it passed.

    at_bound is checked FIRST and unconditionally: it is the only free
    one, and a value on its constraint is not a measurement to test the
    other gates against.
    """
    if fit is None:
        return False, 'fit failed'
    railed = getattr(fit, 'at_bound', None) or ()
    if isinstance(railed, str):
        railed = (railed,)
    if gates.get('reject_at_bound', True) and railed:
        fatal = gates.get('at_bound_fatal')       # None = every parameter
        hit = tuple(railed) if fatal is None else tuple(
            n for n in railed if n in fatal)
        if hit:
            return False, f'at bound ({", ".join(hit)})'
    occ = occupancy(cube, fit, voxel_um)
    thr = gates.get('min_occupancy')
    if thr is not None:
        if not np.isfinite(occ):
            return False, 'occupancy undefined (no signal above background)'
        if occ < thr:
            return False, f'occupancy {occ:.2f} < {thr:.2f}'
    lim = gates.get('max_uncert_xy_nm')
    if lim is not None:
        ci = 1000.0 * max(getattr(fit, 'ci_y_um', 0.0) or 0.0,
                          getattr(fit, 'ci_x_um', 0.0) or 0.0)
        if not np.isfinite(ci) or ci > lim:
            return False, f'lateral uncertainty {ci:.0f} nm > {lim:.0f} nm'
    lim = gates.get('max_uncert_z_nm')
    if lim is not None:
        ci = 1000.0 * (getattr(fit, 'ci_z_um', 0.0) or 0.0)
        if not np.isfinite(ci) or ci > lim:
            return False, f'axial uncertainty {ci:.0f} nm > {lim:.0f} nm'
    return True, None


def qc_shift(reference_cube, cube, z_ref, z_here, half=15):
    """Independent estimate of the same displacement, by image matching.

    NOT the estimator -- measured 14.8% worse than the fit on the median
    of 305 replicate pairs, and closer on only 37% of them. It is here
    because it has the better TAIL (p90 1.858 against 1.945) and is
    independent: two estimates of one displacement that DISAGREE is a
    strong outlier signal, and it costs ~20 s per 48 alleles.

    Returns (dy, dx, dz) in voxels, or None.
    """
    from codelab_pipeline.localization import shapefree as SF
    if reference_cube is None or cube is None:
        return None
    a = _slab(reference_cube, z_ref, half)
    b = _slab(cube, z_here, half)
    if a is None or b is None:
        return None
    s, _q = SF.shift_yxz(a, b, upsample=20, min_coverage=0.9)
    if s is None:
        return None
    # each slab is cut around its OWN depth, so the bulk difference is
    # carried by the origins and the correlation finds only the residual
    return (s[0], s[1], s[2] + (float(z_ref) - float(z_here)))


def build_chromatin_trace_allele(allele, hybes, reference_hybe,
                                 hybe_fiducial_channels, hybe_readout_channels,
                                 storage_path, fov, modality, cell, fov_matrices,
                                 params=None, max_fiducial_drift=5.0,
                                 max_fiducial_drift_z=10.0, spad=8,
                                 collect_debug=False, resolver=None,
                                 append=False):
    """
    v2's counterpart to localization.build_chromatin_trace_allele, filling
    the same three fields on `allele` and returning the same
    (allele, debug) pair, so the panel can switch engines without anything
    downstream knowing.

    The COORDINATE work is not reimplemented -- spot_mapper and
    cell_z_offset are v1's and are called here unchanged. Duplicating
    frame algebra is how two paths silently disagree about where a spot
    is, and that algebra is the part of v1 that was never in question.

    What differs is the order of operations. v1 fits each hybe
    independently and only then compares them. v2 must place its boxes
    BEFORE fitting, which needs every hybe's crop first, so the fiducial
    crops are all cut in one pass, the consensus depth derived from them
    together, and only then is anything fitted.
    """
    from codelab_pipeline.alignment import spot_mapper
    from codelab_pipeline.localization import localization as L

    p = params or V2Params()
    if not append:
        allele.fiducial_trace, allele.polymer, allele.rejected_hybes = {}, {}, {}
    debug = {} if collect_debug else None
    # (y, x). NOT (x, y). allele.coordinate is rasterized order (y, x, z)
    # per models/allele.py, and spot_mapper.reference_to_raw unpacks
    # `y, x = coordinate`. v1 passes (coordinate[0], coordinate[1]) at
    # localization.py:1173 and this must match it exactly.
    #
    # It did not. Transposing these cuts every crop at the MIRRORED image
    # location -- a spot at y=300, x=700 was fitted at y=700, x=300 -- and
    # the failure is silent, because a crop taken anywhere still contains
    # pixels and still fits something. It surfaced only as symptoms that
    # each looked like a different problem: readouts "failing", occupancy
    # below threshold, fiducial drift exceeding its gate.
    shared_xy = (float(allele.coordinate[0]), float(allele.coordinate[1]))
    mod = modality if modality is not None else getattr(cell, 'reference_modality', None)

    def _cut(hybe, channel):
        raw_y, raw_x = spot_mapper.reference_to_raw(
            shared_xy, hybe, fov_matrices, modality=modality, cell=cell,
            resolver=resolver)
        cube, (ymin, xmin) = spot_mapper.crop_for_localization(
            storage_path, fov, hybe, channel, (raw_y, raw_x), pad=spad,
            use_stack=True)
        return cube, ymin, xmin

    def _to_shared(hybe, yf, xf, zf, ymin, xmin):
        sy, sx = spot_mapper.raw_to_reference(
            (yf + ymin, xf + xmin), hybe, fov_matrices, modality=modality,
            cell=cell, resolver=resolver)
        sz = zf if cell is None else zf + L.cell_z_offset(cell, hybe, mod, resolver)
        return float(sy), float(sx), float(sz)

    # -- phase 1: cut every fiducial crop, then place the boxes ---------
    fid_cubes, fid_origin, z_offsets = {}, {}, {}
    todo = [h for h in hybes if hybe_fiducial_channels.get(h)]
    if reference_hybe not in todo and hybe_fiducial_channels.get(reference_hybe):
        todo.append(reference_hybe)
    for hybe in todo:
        try:
            cube, ymin, xmin = _cut(hybe, hybe_fiducial_channels[hybe])
        except (OSError, ValueError):
            continue
        if cube is None or cube.size == 0:
            continue
        fid_cubes[hybe] = cube
        fid_origin[hybe] = (ymin, xmin)
        z_offsets[hybe] = (0.0 if cell is None
                           else float(L.cell_z_offset(cell, hybe, mod, resolver)))
    zexp = consensus_native_z(fid_cubes, z_offsets)

    # -- phase 2: fit the fiducials at their own expected depth ---------
    fid_local, ref_note = {}, None
    for hybe, cube in fid_cubes.items():
        if debug is not None:
            debug.setdefault(hybe, {'fiducial_cubic': None, 'fiducial_centroid': None,
                                    'readout_cubic': None, 'readout_centroids': None})
            debug[hybe]['fiducial_cubic'] = cube
        z0 = zexp.get(hybe, cube.shape[2] / 2.0)
        f = fit_fiducial(cube, z0, p)
        ok, why = gate(f, cube, p.fiducial_gates, p.voxel_um)
        # THE REFERENCE IS NOT AN ORDINARY HYBE. Every delta is measured
        # against it, so gating it out does not reject one round -- it
        # rejects the ALLELE, and it does so while reporting one bland
        # 'reference hybe fiducial not found' per round, which hides the
        # single real cause behind N identical symptoms. Measured before
        # this exemption: 2 of 4 HoxA alleles lost all 45 rounds apiece
        # to a gate applied to one fit.
        #
        # So the reference is accepted whenever it FITTED AT ALL. A
        # mediocre baseline still defines a usable frame -- and it is a
        # frame, not a measurement: it cancels out of every pair distance
        # (delta(a) - delta(b) = fid(b) - fid(a)), so its quality bounds
        # precision rather than biasing the result. Its gate verdict is
        # kept and reported instead of discarded.
        if hybe == reference_hybe and not ok and f is not None:
            ref_note = why
            ok = True
        if not ok:
            allele.rejected_hybes[hybe] = f'fiducial {why}'
            continue
        ymin, xmin = fid_origin[hybe]
        allele.fiducial_trace[hybe] = _to_shared(hybe, f.y, f.x, f.z, ymin, xmin) \
            + (float(f.amplitude),)
        fid_local[hybe] = (f.y, f.x, f.z)
        if debug is not None:
            debug[hybe]['fiducial_centroid'] = (f.x, f.y, f.z)
    if ref_note and debug is not None:
        debug.setdefault(reference_hybe, {})['reference_warning'] = ref_note

    baseline = allele.fiducial_trace.get(reference_hybe)

    # -- phase 3: the drift gate, then the readouts ---------------------
    for hybe in hybes:
        if hybe in allele.rejected_hybes:
            continue
        fid = allele.fiducial_trace.get(hybe)
        if baseline is None:
            allele.rejected_hybes[hybe] = 'reference hybe fiducial not found'
            continue
        if fid is None:
            allele.rejected_hybes[hybe] = 'fiducial not found'
            continue
        dy, dx, dz = (baseline[0] - fid[0], baseline[1] - fid[1],
                      baseline[2] - fid[2])
        drift = float(np.hypot(dx, dy))
        if drift > max_fiducial_drift:
            allele.rejected_hybes[hybe] = f'drift {drift:.1f}px > max {max_fiducial_drift}px'
            continue
        # Z gated SEPARATELY, in planes: a fiducial fit can pass the XY
        # bound while landing on entirely different content in depth --
        # a real case had a weak fit 20 planes from the reference at only
        # 1.4 px of lateral drift, and using it would have "corrected"
        # every readout in that round by a bogus dz.
        if abs(dz) > max_fiducial_drift_z:
            allele.rejected_hybes[hybe] = (f'z drift {dz:.1f} planes > max '
                                           f'{max_fiducial_drift_z} planes')
            continue

        # QC only, never the correction. See qc_shift.
        if p.qc_shift and hybe != reference_hybe and debug is not None:
            debug[hybe]['qc_shift'] = qc_shift(
                fid_cubes.get(reference_hybe), fid_cubes.get(hybe),
                zexp.get(reference_hybe, 0), zexp.get(hybe, 0))

        channel = hybe_readout_channels.get(hybe)
        if channel is None:
            allele.rejected_hybes[hybe] = 'no readout channel configured'
            continue
        try:
            cube, ymin, xmin = _cut(hybe, channel)
        except (OSError, ValueError):
            allele.rejected_hybes[hybe] = 'readout crop unreadable'
            continue
        if cube is None or cube.size == 0:
            allele.rejected_hybes[hybe] = 'readout crop empty'
            continue
        if debug is not None:
            debug[hybe]['readout_cubic'] = cube
        r = fit_readout(cube, zexp.get(hybe, cube.shape[2] / 2.0), p)
        ok, why = gate(r, cube, p.readout_gates, p.voxel_um)
        if not ok:
            allele.rejected_hybes[hybe] = f'readout {why}'
            continue
        sy, sx, sz = _to_shared(hybe, r.y, r.x, r.z, ymin, xmin)
        # the same fiducial(ref) - fiducial(round) correction v1 applies,
        # in the shared frame
        allele.polymer[hybe] = [(float(sx + dx), float(sy + dy), float(sz + dz),
                                 float(r.amplitude))]
        if debug is not None:
            debug[hybe]['readout_centroids'] = [(r.x, r.y, r.z)]
    return allele, debug


def is_v2(engine):
    """True for the v2 engine name, whatever decoration the combo carries.

    Matched on a PREFIX rather than on equality so the panel's label can
    gain explanation ('v2 (calibrated PSF)') without silently switching
    every run back to v1 -- a failure that would show up only as slightly
    worse numbers.
    """
    return str(engine or '').strip().lower().startswith('v2')


def trace_allele(engine, allele, hybes, reference_hybe, hybe_fiducial_channels,
                 hybe_readout_channels, storage_path, fov, modality, cell,
                 fov_matrices, v2_params=None, max_fiducial_drift=5.0,
                 max_fiducial_drift_z=10.0, spad=8, z_window=15,
                 fiducial_params=None, readout_params=None, collect_debug=False,
                 resolver=None, z_boundary_trim=0, executor=None, append=False):
    """
    Route one allele to the chosen engine. The ONE place the choice is
    made, so the preview and the batch run cannot diverge -- they took
    different code paths to the same v1 call before, and a switch added to
    only one of them would be invisible until the two disagreed.

    v1 keeps every argument it had. The ones v2 has no use for are dropped
    rather than accepted and ignored: z_window is a mixture-mode seed
    search (v2 fits one emitter), z_boundary_trim shaves the pillar ends
    (v2 fits a box placed at the consensus depth, so the ends are already
    out of the domain), and the per-channel *_params carry v1's gate
    constants, which do not transfer to a local background.
    """
    if not is_v2(engine):
        from codelab_pipeline.localization import localization as L
        return L.build_chromatin_trace_allele(
            allele, hybes, reference_hybe, hybe_fiducial_channels,
            hybe_readout_channels, storage_path, fov, modality, cell,
            fov_matrices, max_fiducial_drift=max_fiducial_drift,
            max_fiducial_drift_z=max_fiducial_drift_z, spad=spad,
            z_window=z_window, fiducial_params=fiducial_params,
            readout_params=readout_params, collect_debug=collect_debug,
            resolver=resolver, z_boundary_trim=z_boundary_trim,
            executor=executor, append=append)
    return build_chromatin_trace_allele(
        allele, hybes, reference_hybe, hybe_fiducial_channels,
        hybe_readout_channels, storage_path, fov, modality, cell, fov_matrices,
        params=v2_params, max_fiducial_drift=max_fiducial_drift,
        max_fiducial_drift_z=max_fiducial_drift_z, spad=spad,
        collect_debug=collect_debug, resolver=resolver, append=append)


def _slab(cube, zc, half):
    """Planes [zc-half, zc+half], NaN-PADDED where they run off the stack.

    Padded, never clipped: clipping changes a slab's extent and therefore
    its centre, and two slabs of different extent are not comparable --
    which is the entire point of correlating them. Boxes are padded for
    the same reason (0.1% of real crops).
    """
    if cube is None:
        return None
    nz = cube.shape[2]
    z0 = int(round(float(zc) - half))
    z1 = z0 + 2 * int(half) + 1
    out = np.full((cube.shape[0], cube.shape[1], z1 - z0), np.nan, dtype=float)
    a0, b0, a1 = max(z0, 0), max(-z0, 0), min(z1, nz)
    if a1 <= a0:
        return None
    out[:, :, b0:b0 + (a1 - a0)] = cube[:, :, a0:a1]
    return out if np.isfinite(out).any() else None
