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

# How each hybe's FIDUCIAL box is placed in depth. Proposed and A/B-tested
# 2026-08-29, both arms through the engine at the shipping gates (fid occ
# 0.25, readout occ 0.40, z-uncert 150, lateral at-bound fatal; MP58 FOV1,
# 127 alleles):
#
#                     traced   fiducials   pairs   med nm   p90 nm
#     consensus         3395        7694     230      129      320
#     self              3111        7288     204      126      399
#
# 'self' (per-hybe pillar intensity centroid) was proposed to stop losing
# fiducials to a mis-placed consensus box -- and measured, it keeps FEWER
# fiducials: over ~110 planes the single-pillar centroid sometimes centres
# the box on the wrong depth structure, that fit scores low occupancy, and
# the hybe dies. The cross-hybe median pools ~70 hybes and is the more
# robust estimator of the same quantity -- the identity-over-greed lesson
# again, one axis down. Kept selectable for re-testing on other data.
#
# The READOUT box is unaffected by this switch: it sits at its own hybe's
# fitted fiducial z in both modes (see the readout phase), which is the
# piece of the same proposal that survives on physical grounds.
Z_PLACEMENT = 'consensus'

# Retry a gate-failed fiducial fit once from the argmax seed. See the
# fallback site in the trace loop for the basin story and the numbers.
FIDUCIAL_SEED_FALLBACK = True

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
#   at_bound        READOUT ONLY. 295/311 pairs at 0.218 um against 311
#                   at 0.294 ungated -- keeps 95% of pairs, improves the
#                   median 26%, and has NO THRESHOLD TO CHOOSE. A
#                   parameter that stopped on its constraint is the bound
#                   you supplied, not a measurement, and its Jacobian CI
#                   (which assumes an interior optimum) does not describe
#                   it either. That measurement is a readout one --
#                   gate_sweep_v2.score filters ra/rb and never ga/gb --
#                   and on the FIDUCIAL the same gate measured WORSE than
#                   a readout uncertainty gate at matched coverage. See
#                   FIDUCIAL_GATES.
#   occupancy       the tunable one, and the best-behaved: it degrades
#                   smoothly instead of falling off a cliff.
#
# Deliberately absent, both v1 heritage:
#   min_hb_ratio    untunable -- 311 pairs at 1.0, 40 at 1.2, ~10 by 1.6.
#                   A 0.1 change swings coverage by an order of magnitude.
#   min_ah_ratio    dominated by occupancy, which measures the same intent
#                   properly.
#
# The uncertainty gates default OFF, with ONE exception: the readout's
# axial gate, set to 150 nm by explicit decision (2026-08-29) after
# inspecting real tiles with both CIs printed on them. The measured
# ladder behind it (127 MP58 FOV1 alleles, post hoc on one permissive
# pass; same-locus repeat distance):
#
#     z-uncert gate   pairs   med nm   p90 nm   readouts kept
#     off               443      194     1605      69%
#     <= 500            413      180     1297      64%
#     <= 300            363      159      811      57%
#     <= 150            269      149      675      44%
#
# 150 trades ~40% of pairs for a 2.4x better p90 -- precision over
# coverage, deliberately. The knee is near 300 if coverage matters more.
# Derived on ONE dataset; the panel knob (0 = off) is the escape hatch,
# and the tiles print the very number the gate tests, so re-deriving it
# on another experiment is a matter of reading its own grid.
FIDUCIAL_GATES = {
    # OFF for the fiducial. at_bound belongs to the READOUT, where the 295/311
    # measurement was actually made; on the fiducial it is a blunt proxy for
    # round quality that the readout's own gates measure better and tunably.
    #
    # Measured post-hoc over 9007 permissively-fitted readouts, MP58 FOV1,
    # 127 alleles (one fitting pass, gates applied afterwards):
    #
    #     configuration                        pairs  med nm  p90 nm
    #     readout gates only                     384     168     900
    #     + fiducial at_bound FATAL              378     165     794
    #     + readout z-uncert <= 600 nm           375     164     735
    #     + readout xy-uncert <= 200 nm          377     164     797
    #     + readout z-uncert <= 500 nm           365     161     622
    #
    # At matched coverage the readout axial-uncertainty gate beats this one:
    # 3 fewer pairs (0.8%) for a 7.4% better p90, and it has a knob.
    #
    # WHY the swap works, and why it is not the same set: hybes whose
    # fiducial railed do have worse readouts -- z-uncert median 299 nm
    # against 171, xy 92 against 53, occupancy 0.5 against 0.7 -- so a
    # railed fiducial really does mark a poor round. But of the 798
    # readouts this gate removed, only 159 (20%) are also removed by
    # z-uncert <= 600. It was discarding 639 readouts that were not the
    # damaging ones.
    #
    # Position-at-bound is still recorded and still shown on the fiducial
    # grid beside occupancy and both CIs; it is a diagnostic here, not a
    # verdict.
    'reject_at_bound': False,
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
    # Kept for when reject_at_bound is switched back on: if it is, only
    # POSITION should be fatal, never sigma.
    'at_bound_fatal': ('y', 'x', 'z'),
    'min_occupancy': 0.25,      # looser: an extended object spreads its peak
    'max_uncert_xy_nm': None,
    'max_uncert_z_nm': None,
}
READOUT_GATES = {
    'reject_at_bound': True,
    # POSITION only, matching the filter that was actually swept:
    # gate_sweep_v2.py:159 is `any(s in ('x', 'y', 'z') for s in
    # f.at_bound)`, and that is the filter behind 295/311 pairs at
    # 0.218 um. (The tuple there is a membership SET over parameter names
    # -- fit3d_um.py:245 -- not a coordinate order, so ('x','y','z') and
    # ('y','x','z') are the same test.)
    #
    # This read None (= any parameter fatal) and would have rejected on a
    # railed background offset, which the measurement never did. It bites
    # only on the no-PSF fallback, where sigma is free and can rail too.
    #
    # 'z' IS DELIBERATELY ABSENT. The readout box is pre-placed at the
    # fiducial-derived CONSENSUS depth -- that placement is what makes the
    # fit fast and well-conditioned -- so the axial bound measures how far
    # this hybe's locus sits from the median fiducial plane, which is a
    # property of the allele's geometry, not of the readout's data
    # quality. A hybe that genuinely sits off the consensus depth rails
    # inevitably and innocently: Hyb_043 on FOV1 railed on z while
    # reporting CIs of xy 18 nm and z 57 nm, i.e. an extremely well
    # determined fit that merely stopped at its leash.
    #
    # Lateral is not analogous and stays fatal. The crop is centred by
    # alignment, not by a consensus taken over other hybes, so a railed
    # y or x really does mean the fit could not reach the emitter.
    'at_bound_fatal': ('y', 'x'),
    'min_occupancy': 0.40,
    'max_uncert_xy_nm': None,
    'max_uncert_z_nm': 150.0,     # see the ladder above
}

# -- fit domains ----------------------------------------------------------
#
# A BOX, not a pillar. The tracing crop is bounded in XY but takes the
# full slab in Z (17x17x110): the emitter is a few hundred voxels against
# ~34k of out-of-focus content, so least squares spends its position and
# sigma parameters describing background. That single change moved
# occupancy 0.373 -> 0.806.
# Widened with the bounds above, and for their sake: a position bound must
# sit INSIDE the fitted box or the fit is allowed to walk somewhere it has
# no data (this module already carries that lesson at an axial radius of
# 2.0 um, where the bound equalled the domain). At 7 px the bound is
# 1.456 um and at 15 planes it is 3.000 um, so both old radii were too
# small to hold them. READOUT_FIT_RADIUS_UM is deliberately NOT changed:
# every readout number on record was measured at (1.0, 1.0, 3.0), and the
# readout is not what is railing.
FIDUCIAL_FIT_RADIUS_UM = (1.7, 1.7, 3.6)
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
# 7 px / 15 planes, raised from 5/10. THE BOUND AND THE DRIFT GATE ARE THE
# SAME NUMBER and have to move together: the fit cannot travel further than
# its bound, so a drift gate at 7 px could never observe a value above the
# 5 px bound -- it would gate on a quantity the fit was incapable of
# producing, and every genuinely-larger drift would arrive pre-labelled
# 'at bound' instead. Measured on FOV1 allele 15 at 5/10: 33 of 73 hybes
# rejected 'fiducial at bound', 29 of them on y alone, while the readout
# crops in those same hybes are clean, bright and obviously fittable.
FIDUCIAL_PEAK_BOUND_UM = 1.456       # 7 px
FIDUCIAL_PEAK_BOUND_Z_UM = 3.0       # 15 planes
# BOTH ENGINES ANCHOR THESE BOUNDS AT THE SEED (fit3d_um.py builds lb/ub
# from y0/x0/z0; fit3d_mle.py identically), not at the crop centre. That
# is what closed the railed-by-drift failure: while the seed was trapped
# within +/-5 px of the crop centre, the bound was effectively anchored
# to the alignment prior, and crop-placement error (drift) turned into
# at-bound verdicts on clean spots. With the crop-spanning seed, the
# bound travels to wherever the centroid found the emitter -- railing
# fell 4.4% -> 1.4% of readouts from the seed change alone.
READOUT_PEAK_BOUND_UM = 1.04         # 5 px
# 14 planes, up from 10. Neutralising the z at_bound gate (see
# READOUT_GATES) removes the REJECTION but not the truncation: a fit that
# stops on its axial leash still reports the leash as its z, and that
# value would now flow into the polymer instead of being thrown away --
# a visible rejection traded for an invisible error. Widening the bound
# is what makes the neutralisation safe rather than merely quieter.
#
# 2.8 um sits INSIDE the unchanged 3.0 um axial fit radius, so the fit
# DOMAIN -- the thing every recorded readout number was measured at
# (1.0, 1.0, 3.0) -- is untouched; only how far the centre may travel
# within it changes.
READOUT_PEAK_BOUND_Z_UM = 2.8

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
# Same ceilings for the readout's FREE-sigma fallback. Only reached when no
# plausible calibrated PSF is installed; with one, sigma is fixed and these
# are unused.
READOUT_MAX_SIGMA_XY_UM = 3.00
READOUT_MAX_SIGMA_Z_UM = 6.00

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
        # The panel's v2 page, when it is there. Absent (a config written
        # before the page existed, or a caller that is not the panel) means
        # the MEASURED defaults stand -- V2Params merges over
        # FIDUCIAL_GATES / READOUT_GATES rather than replacing them, so a
        # missing key can never silently disable a gate.
        v2 = params.get('v2') or {}
        fam = shape = None
        label = params.get('readout_psf', '')
        doc = LIB.installed(storage_path) if storage_path else None
        if doc is None and label:
            doc = LIB.read(label)
        got = LIB.shape_tuple(doc) if doc else None
        if got:
            # REFUSE a degenerate shape rather than fixing every readout to
            # it. Score cannot arbitrate this -- a 39 nm core scored rss/vox
            # 2986 against 2931 for a 312 nm one on the same real data -- so
            # psf.plausible() checks the physics instead: nothing on a
            # declared bound, nothing below the optical limit. A calibrated
            # PSF that is physically impossible is worse than no calibration,
            # because free sigma can still recover while a fixed wrong shape
            # cannot. Falling back to free sigma is the safe direction.
            from codelab_pipeline.localization import psf as P
            ok, why = P.plausible(got[0], doc.get('params') or {})
            if ok:
                fam, shape = got
                label = doc.get('installed_from') or doc.get('label') or label
            else:
                label = f'{label} [REJECTED: {"; ".join(why)}]'
        return cls(voxel_um=voxel, psf_family=fam, psf_shape=shape,
                   psf_label=label,
                   fiducial_gates=v2.get('fiducial'),
                   readout_gates=v2.get('readout'),
                   qc_shift=v2.get('qc_shift', True))

    def describe(self):
        """One line that reconstructs the run.

        BOTH branches name the voxel size and whatever is known about the
        PSF. The fallback branch used to say only "no calibrated PSF
        installed" -- which is wrong when one IS installed and was
        REJECTED, discards the reason plausible() computed, and omits the
        voxel size entirely. A run that silently becomes a different run
        (readout sigma free instead of fixed) has to leave a record saying
        which run it became.
        """
        if not self.has_psf:
            why = f' ({self.psf_label})' if self.psf_label else ''
            return (f'v2, readout sigma FREE -- no usable calibrated PSF{why}; '
                    f'voxel {self.voxel_um}')
        sxy = self.psf_shape[0] * 1000.0
        return (f'v2, readout PSF {self.psf_label!r} ({self.psf_family}, '
                f'sigma_xy {sxy:.0f} nm), voxel {self.voxel_um}')


# -- depth placement ------------------------------------------------------

def consensus_native_z(cubes_by_hybe, z_offsets):
    """
    {hybe: expected native z} -- where this allele should sit, in depth,
    in EACH hybe's own stack.

    Still the FIDUCIAL placement (Z_PLACEMENT = 'consensus'): the
    per-hybe alternative (own_native_z) was proposed and A/B-tested at
    the shipping gates and kept fewer fiducials and fewer pairs at a
    worse p90 -- see the table at Z_PLACEMENT. The READOUT box no longer
    uses this: it sits at its own hybe's fitted fiducial z.

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


def own_native_z(cube, voxel_um=DEFAULT_VOXEL_UM):
    """This hybe's OWN depth: intensity-weighted z centroid of the pillar.

    The proposed replacement for the cross-hybe consensus (2026-08-29):
    the consensus is an external prior, and a hybe genuinely off the
    median depth got its box mis-placed, railing innocently. MEASURED at
    the shipping gates it is NOT the default: the single-pillar centroid
    is noisier than the pooled median and lost more fiducials than the
    mis-placement did (see Z_PLACEMENT). Selectable for re-testing; also
    the readout-phase fallback when a hybe has no fiducial fit.

    The centroid, NOT the argmax: the argmax is one voxel and as noisy as
    one voxel (consensus_native_z's own measurement), and NOT a pillar
    fit: measured at 4.65 planes of placement error against 1.05 for the
    fit-free routes -- a pillar fit is the degenerate fit this module
    exists to avoid. The centroid is floor-clipped at the median so the
    ~110 planes of out-of-focus background do not drag it to mid-stack.

    Boundary needs no NaN padding: the fit DOMAIN (fit_radius_um around
    the seed) clips at the stack edge by construction, and display crops
    stay full-depth.
    """
    ny, nx, nz = cube.shape
    if not np.isfinite(cube).any():
        # nanargmax on an all-NaN cube raises; mid-stack is the only
        # honest answer when the pillar holds nothing.
        return (nz - 1) / 2.0
    got = U.intensity_centroid(
        cube, ((ny - 1) / 2.0, (nx - 1) / 2.0, (nz - 1) / 2.0),
        (max(1, ny // 2 - 1), max(1, nx // 2 - 1), nz), voxel_um)
    if got is not None:
        return float(got[2])
    return float(np.unravel_index(int(np.nanargmax(cube)), cube.shape)[2])


# -- fitting --------------------------------------------------------------

def _seed(cube, z_centre, voxel_um, z_half):
    """Intensity-weighted centroid, falling back to the crop centre.

    Both engines start here, per explicit request. It is worth more to v1
    (0.354 -> 0.597) than to v2 (0.799 -> 0.818), because v2's boxed
    domain has already removed most of what a bad seed used to chase.

    THE LATERAL WINDOW SPANS THE CROP, less one voxel at each edge. It was
    a fixed +/-5 px centred on the crop CENTRE, while the crop at pad=8 is
    17x17: a spot more than 5 px off-centre fell entirely outside the
    window, so the centroid stayed near the middle, the fit box got placed
    around a point that was not the emitter, and the fit converged into
    background. That is reported as low occupancy on a crop that plainly
    holds one clean PSF -- the number looks like a bad spot when it is
    really a mis-aimed search. The crop is already the statement of where
    the emitter might be; the seed should search all of it.

    One voxel is left at each edge deliberately. A centroid computed hard
    against a boundary is one-sided, and a seed on the rim gives the fit
    box nothing to work with on that side.

    z_half is EXPLICIT and comes from the caller's own fit radius, so the
    seed searches exactly as far as the fit can subsequently reach -- no
    more (the crop spans the whole slab, and a centroid over ~110 planes
    is dragged by out-of-focus content) and no less (a seed window
    narrower than the fit domain hides emitters the fit could have found).
    """
    ny, nx, nz = cube.shape
    cy = (ny - 1) / 2.0
    cx = (nx - 1) / 2.0
    zc = float(np.clip(z_centre, 0, nz - 1))
    half = (max(1, ny // 2 - 1), max(1, nx // 2 - 1), max(1, int(z_half)))
    got = U.intensity_centroid(cube, (cy, cx, zc), half, voxel_um)
    return got if got is not None else (cy, cx, zc)


def _seed_z_half(fit_radius_um, voxel_um):
    """Axial seed half-width in PLANES, from a fit radius in micrometres."""
    return max(1, int(round(fit_radius_um[2] / float(voxel_um[2]))) - 1)


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
    seed = _seed(cube, z_centre, p.voxel_um,
                 _seed_z_half(FIDUCIAL_FIT_RADIUS_UM, p.voxel_um))
    return fit_fiducial_from(cube, seed, p)


def fit_fiducial_from(cube, seed, p):
    """fit_fiducial's core with the SEED chosen by the caller.

    Exists for the second-start fallback: on faint extended fiducials the
    fit landscape has two basins (emitter vs background-soak) and the
    seed decides which one least squares falls into, so a failed
    centroid-seeded fit is retried from the argmax.
    """
    if cube is None or not np.isfinite(cube).any():
        return None
    sy, sx, sz = seed
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
    sy, sx, sz = _seed(cube, z_centre, p.voxel_um,
                       _seed_z_half(READOUT_FIT_RADIUS_UM, p.voxel_um))
    if not p.has_psf:
        # The FREE-sigma fallback must carry the validated ceilings too.
        # Passing neither leaves fit3d_um's own defaults (0.520 / 1.000 um)
        # in force -- a readout-sized ceiling that the measured ladder
        # never used, and tight enough that a slightly broad spot rails
        # instead of fitting. Every v2 number was taken at 3.0 / 6.0.
        return U.fit_gaussian_3d_um(
            cube, sy, sx, sz, voxel_um=p.voxel_um,
            peak_bound_um=READOUT_PEAK_BOUND_UM,
            peak_bound_z_um=READOUT_PEAK_BOUND_Z_UM,
            min_sigma_um=FIDUCIAL_MIN_SIGMA_UM,
            max_sigma_xy_um=READOUT_MAX_SIGMA_XY_UM,
            max_sigma_z_um=READOUT_MAX_SIGMA_Z_UM,
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


def uncertainty_nm(fit):
    """(lateral, axial) FULL 95% interval in nanometres, or (nan, nan).

    THE SAME EXPRESSION THE GATE TESTS, deliberately shared rather than
    re-derived at the display: `2000 * max(ci_y, ci_x)` and `2000 * ci_z`.
    FitUm.ci_*_um are HALF-widths, so a display that used 1000x would show
    a number half the size of the threshold it is meant to help choose --
    the exact off-by-2x this module already carries a comment about.
    """
    if fit is None:
        return float('nan'), float('nan')
    xy = 2000.0 * max(getattr(fit, 'ci_y_um', 0.0) or 0.0,
                      getattr(fit, 'ci_x_um', 0.0) or 0.0)
    z = 2000.0 * (getattr(fit, 'ci_z_um', 0.0) or 0.0)
    return float(xy), float(z)


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
    # 2000x, not 1000x. FitUm.ci_*_um are HALF-widths, and the sweep that
    # produced the recorded thresholds converted with
    # `2000 * max(f.ci_y_um, f.ci_x_um)` (gate_sweep_v2.py:157-158) -- i.e.
    # the FULL 95% interval in nanometres. Using the half-width here would
    # compare against a threshold calibrated on twice the quantity, and
    # every uncertainty gate would be exactly 2x looser than the number it
    # was set from. Latent today only because these default to None.
    lim = gates.get('max_uncert_xy_nm')
    if lim is not None:
        ci = 2000.0 * max(getattr(fit, 'ci_y_um', 0.0) or 0.0,
                          getattr(fit, 'ci_x_um', 0.0) or 0.0)
        if not np.isfinite(ci) or ci > lim:
            return False, f'lateral uncertainty {ci:.0f} nm > {lim:.0f} nm'
    lim = gates.get('max_uncert_z_nm')
    if lim is not None:
        ci = 2000.0 * (getattr(fit, 'ci_z_um', 0.0) or 0.0)
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
                                 collect_debug=False, resolver=None):
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
    # ALWAYS a full re-derivation. v2 has no merge mode, deliberately.
    #
    # It briefly had one, mirroring v1's, and it is now unreachable: an
    # allele only reaches this function because append-mode membership
    # said it has no committed trace, so there is nothing to merge into.
    # Leaving the branch in place would advertise a mode nothing selects
    # and invite a future caller to switch it on, which is exactly how the
    # per-hybe append rule survived long enough to mix two engines'
    # estimates inside one polymer_adj.
    allele.fiducial_trace_adj, allele.polymer_adj = {}, {}
    allele.fiducial_trace_raw, allele.polymer_raw = {}, {}
    allele.rejected_hybes = {}
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
        # UNCONDITIONAL, unlike v1's `if cell is not None` guard. With a
        # resolver, cell_z_offset returns resolver.z_to_shared(...) which
        # carries the FOV-level CROSS-MODAL z drift -- and its own
        # docstring says that drift "is FOV-bounded and therefore applies
        # to unassigned spots too". v1's guard skips the resolver entirely
        # for a cell-less allele and silently drops that correction. The
        # function already returns 0.0 for the genuinely-nothing-known
        # case (cell None, no resolver), so calling it always is both safe
        # and more correct.
        sz = zf + L.cell_z_offset(cell, hybe, mod, resolver)
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
        z_offsets[hybe] = float(L.cell_z_offset(cell, hybe, mod, resolver))
    # Each hybe's box at its OWN depth -- no cross-hybe consensus, no
    # external prior for the placement. z_offsets stay: the drift gate and
    # the shared-frame conversion still need raw->shared per hybe.
    # Z_PLACEMENT is module state so the two schemes stay A/B-able; the
    # consensus is the measured reference this change is judged against.
    if Z_PLACEMENT == 'consensus':
        zexp = consensus_native_z(fid_cubes, z_offsets)
    else:
        zexp = {h: own_native_z(c, p.voxel_um) for h, c in fid_cubes.items()}

    # -- phase 2: fit the fiducials at their own expected depth ---------
    fid_local, ref_note = {}, None
    precut = {}     # hybe -> (cube, ymin, xmin) already read for the preview
    for hybe, cube in fid_cubes.items():
        if debug is not None:
            debug.setdefault(hybe, {'fiducial_cubic': None, 'fiducial_centroid': None,
                                    'readout_cubic': None, 'readout_centroids': None,
                                    'fiducial_occupancy': float('nan'),
                                    'readout_occupancy': float('nan'),
                                    'fiducial_uncert_nm': (float('nan'), float('nan')),
                                    'readout_uncert_nm': (float('nan'), float('nan')),
                                    'fiducial_at_bound': (),
                                    'readout_at_bound': (),
                                    'fiducial_rejected_centroid': None,
                                    'readout_rejected_centroids': None,
                                    'fiducial_seed': None,
                                    'readout_seed': None,
                                    'readout_zexp': float('nan')})
            debug[hybe]['fiducial_cubic'] = cube
        z0 = zexp.get(hybe, cube.shape[2] / 2.0)
        if debug is not None:
            debug[hybe]['fiducial_seed'] = _seed(
                cube, z0, p.voxel_um,
                _seed_z_half(FIDUCIAL_FIT_RADIUS_UM, p.voxel_um))
        f = fit_fiducial(cube, z0, p)
        ok, why = gate(f, cube, p.fiducial_gates, p.voxel_um)
        if not ok and FIDUCIAL_SEED_FALLBACK:
            # SECOND START FROM THE ARGMAX, only when the first fit failed
            # its gate. Faint extended fiducials (contrast ~1.5x, which
            # per-tile display normalization renders indistinguishable
            # from a bright one) give the enlarged fit domain TWO basins:
            # the emitter, and a background-soak solution with a huge free
            # sigma. The centroid seed sometimes starts in the soak basin;
            # the brightest voxel is in the emitter basin by construction.
            #
            # Measured on the five reported allele-15 failures, same
            # bytes: centroid-seeded occupancy -0.36..0.22, argmax-seeded
            # 0.37..0.75 -- all five recover at the 0.25 gate. Passing
            # hybes never pay (no retry), and greedy-seed identity risk is
            # bounded: this fires only where the alternative was LOSING
            # the hybe, and the drift gate against the reference still
            # applies to whatever the retry returns.
            ay, ax, az = np.unravel_index(int(np.nanargmax(cube)), cube.shape)
            f2 = fit_fiducial_from(cube, (float(ay), float(ax), float(az)), p)
            ok2, why2 = gate(f2, cube, p.fiducial_gates, p.voxel_um)
            if ok2:
                f, ok, why = f2, ok2, why2
                if debug is not None:
                    debug[hybe]['fiducial_seed_fallback'] = True
        # Recorded BEFORE the reject below, on purpose: the occupancy that
        # FAILED is the one worth seeing when deciding where the threshold
        # belongs, and a rejected hybe still draws a tile. gate() computes
        # this internally but does not return it; recomputing here keeps
        # gate()'s signature (and its many callers) untouched, and costs
        # nothing in a batch run, where debug is None.
        if debug is not None:
            debug[hybe]['fiducial_occupancy'] = occupancy(cube, f, p.voxel_um)
            debug[hybe]['fiducial_uncert_nm'] = uncertainty_nm(f)
            debug[hybe]['fiducial_at_bound'] = tuple(
                getattr(f, 'at_bound', None) or ())
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
            # The fit EXISTS; the gate refused it. Its position is what the
            # blue circle draws -- yellow = traced, blue = fitted but
            # gated, no circle = no fit at all.
            if debug is not None and f is not None:
                debug[hybe]['fiducial_rejected_centroid'] = (f.x, f.y, f.z)
            allele.rejected_hybes[hybe] = f'fiducial {why}'
            continue
        ymin, xmin = fid_origin[hybe]
        allele.fiducial_trace_adj[hybe] = _to_shared(hybe, f.y, f.x, f.z, ymin, xmin) \
            + (float(f.amplitude),)
        # The SAME fit, before any matrix: crop-local plus the crop's own
        # origin. ymin/xmin are the actual origin, clamp included, so this
        # indexes that hybe's full frame directly and the image can be
        # re-reached without inverting anything.
        allele.fiducial_trace_raw[hybe] = (float(f.y + ymin), float(f.x + xmin),
                                           float(f.z), float(f.amplitude))
        fid_local[hybe] = (f.y, f.x, f.z)
        if debug is not None:
            debug[hybe]['fiducial_centroid'] = (f.x, f.y, f.z)
    if ref_note:
        # On the ALLELE, not only in debug. debug is None in every batch
        # run (collect_debug=False), so a reference that only just scraped
        # through would have been invisible exactly when it matters most --
        # a whole FOV traced against a doubtful frame, with nothing saying
        # so. The frame cancels out of pair distances, but it still bounds
        # precision and the operator deserves to know.
        allele.reference_warning = ref_note
        # AND into provenance, which is persisted and readable. The
        # attribute alone was written by v2 and read by NOTHING -- not
        # saved (AnAllele.save never carried it), not displayed, not
        # logged -- so a whole FOV could be traced against a doubtful
        # reference frame with the only record living in an object that
        # is discarded at the end of the run.
        allele.provenance = dict(allele.provenance or {})
        allele.provenance['reference_warning'] = ref_note
        if debug is not None:
            debug.setdefault(reference_hybe, {})['reference_warning'] = ref_note

    baseline = allele.fiducial_trace_adj.get(reference_hybe)

    # -- phase 3: the drift gate, then the readouts ---------------------
    for hybe in hybes:
        # PREVIEW FIRST, VERDICT SECOND. A rejected round still has to show
        # its crop: View Crop exists to let a person see WHY a round was
        # rejected, and a grid that silently omits the failures shows only
        # the rounds that already worked. Only the fitted-position marker
        # depends on the gate. Costs one crop read per rejected hybe and
        # only when debug is being collected, i.e. never in a batch run.
        if debug is not None and hybe not in (debug or {}):
            debug.setdefault(hybe, {'fiducial_cubic': None, 'fiducial_centroid': None,
                                    'readout_cubic': None, 'readout_centroids': None,
                                    'fiducial_occupancy': float('nan'),
                                    'readout_occupancy': float('nan'),
                                    'fiducial_uncert_nm': (float('nan'), float('nan')),
                                    'readout_uncert_nm': (float('nan'), float('nan')),
                                    'fiducial_at_bound': (),
                                    'readout_at_bound': (),
                                    'fiducial_rejected_centroid': None,
                                    'readout_rejected_centroids': None,
                                    'fiducial_seed': None,
                                    'readout_seed': None,
                                    'readout_zexp': float('nan')})
        if debug is not None and debug[hybe].get('readout_cubic') is None:
            ch0 = hybe_readout_channels.get(hybe)
            if ch0 is not None:
                try:
                    c0, y0, x0 = _cut(hybe, ch0)
                    if c0 is not None and c0.size:
                        debug[hybe]['readout_cubic'] = c0
                        # REUSE IT below. Cutting the preview tile here and
                        # then cutting the same crop again for the fit read
                        # every accepted hybe's readout stack TWICE from the
                        # NAS -- ~111 extra stack opens per previewed allele,
                        # for bytes already in hand.
                        precut[hybe] = (c0, y0, x0)
                except (OSError, ValueError):
                    pass
        if hybe in allele.rejected_hybes:
            continue
        fid = allele.fiducial_trace_adj.get(hybe)
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
        if hybe in precut:
            cube, ymin, xmin = precut.pop(hybe)
        else:
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
        # The readout box sits at this hybe's OWN fiducial depth. The
        # fiducial images the whole traced region and the readout is one
        # locus inside it, so the region's fitted z is the right prior --
        # and it carries this hybe's own drift, unlike the old cross-hybe
        # consensus. With the z bound (2.8 um) anchored at the seed inside
        # this box, the readout's axial reach is now measured FROM ITS OWN
        # FIDUCIAL, which is the physically meaningful anchor.
        z_r = (float(fid_local[hybe][2]) if hybe in fid_local
               else own_native_z(cube, p.voxel_um))
        if debug is not None:
            debug[hybe]['readout_zexp'] = float(z_r)
            debug[hybe]['readout_seed'] = _seed(
                cube, z_r, p.voxel_um,
                _seed_z_half(READOUT_FIT_RADIUS_UM, p.voxel_um))
        r = fit_readout(cube, z_r, p)
        ok, why = gate(r, cube, p.readout_gates, p.voxel_um)
        # Same as the fiducial: before the reject, so a gated-out readout
        # still reports the number it was gated on.
        if debug is not None:
            debug[hybe]['readout_occupancy'] = occupancy(cube, r, p.voxel_um)
            debug[hybe]['readout_uncert_nm'] = uncertainty_nm(r)
            debug[hybe]['readout_at_bound'] = tuple(
                getattr(r, 'at_bound', None) or ())
        if not ok:
            if debug is not None and r is not None:
                debug[hybe]['readout_rejected_centroids'] = [(r.x, r.y, r.z)]
            allele.rejected_hybes[hybe] = f'readout {why}'
            continue
        sy, sx, sz = _to_shared(hybe, r.y, r.x, r.z, ymin, xmin)
        # the same fiducial(ref) - fiducial(round) correction v1 applies,
        # in the shared frame
        # (y, x, z, amplitude) -- y FIRST, matching v1's actual code at
        # localization.py:995 and the whole store's yx convention (see
        # legacy/migrate_store_to_yx.py, which exists purely to enforce it
        # and swaps polymer_adj entries along with coordinate/fiducial_trace_adj).
        #
        # v1's DOCSTRING for _localize_readout_hybe says "(x, y, z,
        # amplitude)" and is stale -- it predates that migration. Writing
        # this tuple x-first, as the docstring implies, mirrors every
        # traced position relative to v1 while remaining a perfectly
        # well-formed 4-tuple that nothing downstream can detect.
        allele.polymer_adj[hybe] = [(float(sy + dy), float(sx + dx), float(sz + dz),
                                     float(r.amplitude))]
        # Raw carries NO correction of any kind -- neither the alignment
        # nor the fiducial drift. adj - raw for a READOUT is therefore
        # alignment PLUS fiducial correction, one term more than the same
        # difference on a fiducial. See AnAllele's docstring.
        allele.polymer_raw[hybe] = [(float(r.y + ymin), float(r.x + xmin),
                                     float(r.z), float(r.amplitude))]
        if debug is not None:
            debug[hybe]['readout_centroids'] = [(r.x, r.y, r.z)]
    return allele, debug


def allele_task(payload):
    """Run ONE allele end to end in a child process.

    THE UNIT OF PARALLELISM FOR v2, and it has to be the allele rather
    than the hybe. (Historically because consensus_native_z needed every
    hybe's fiducial argmax; placement now self-centres per hybe, but the
    drift gate still measures every hybe against the reference fiducial,
    so the allele remains the natural unit,
    mapped into the shared frame, before ANY fit can be seeded -- a
    barrier that v1 does not have, because v1 fits each hybe independently
    end to end.

    Splitting across that barrier per hybe would mean either shipping the
    crops back to the parent and out again (~12 MB per allele each way) or
    reading every crop from the NAS twice. Keeping the whole allele in one
    child keeps the barrier inside one process, and the only thing that
    crosses a process boundary is the finished trace.

    The median itself is not the cost: 0.09 s, measured, against 106 s for
    the fit-based placement it replaced. What it forces is that all the
    crop reads complete before any fitting starts.

    Returns plain dicts, not the AnAllele: the parent owns the object the
    rest of the app holds references to, and merging three dicts into it
    is unambiguous where returning a rebuilt object would quietly replace
    identity.
    """
    (meta, hybes, reference_hybe, fid_ch, read_ch, storage_path, fov, modality,
     cell, fov_matrices, params, max_drift, max_drift_z, spad, resolver) = payload
    from codelab_pipeline.models.allele import AnAllele
    allele = AnAllele()
    allele.set_metadata(**meta)
    # The child bypasses trace_allele, so it stamps its own provenance --
    # otherwise every parallel v2 run would produce unstamped traces while
    # the serial path stamped them, which is worse than not stamping at all.
    import time as _time
    allele.provenance = {
        'engine': 'v2', 'engine_label': 'v2',
        'traced_at': _time.strftime('%Y-%m-%dT%H:%M:%S'),
        'voxel_um': list(params.voxel_um) if params else None,
        'psf': (params.psf_label or None) if params else None,
        'psf_family': params.psf_family if params else None,
    }
    build_chromatin_trace_allele(
        allele, hybes, reference_hybe, fid_ch, read_ch, storage_path, fov,
        modality, cell, fov_matrices, params=params,
        max_fiducial_drift=max_drift, max_fiducial_drift_z=max_drift_z,
        spad=spad, collect_debug=False, resolver=resolver)
    return (int(meta['id']), allele.fiducial_trace_adj, allele.polymer_adj,
            allele.fiducial_trace_raw, allele.polymer_raw,
            allele.rejected_hybes, getattr(allele, 'reference_warning', None),
            dict(getattr(allele, 'provenance', {}) or {}))


def allele_task_with_debug(payload):
    """allele_task, but returning the debug crops the preview renders.

    Separate from allele_task because the crops are ~12 MB per allele and
    a BATCH run must never ship them back -- it renders nothing. The
    preview does, once, for one allele, and paying that transfer is what
    buys keeping every HDF5 read out of the GUI thread.
    """
    (meta, hybes, reference_hybe, fid_ch, read_ch, storage_path, fov, modality,
     cell, fov_matrices, params, max_drift, max_drift_z, spad, resolver) = payload
    from codelab_pipeline.models.allele import AnAllele
    import time as _time
    allele = AnAllele()
    allele.set_metadata(**meta)
    allele.provenance = {
        'engine': 'v2', 'engine_label': 'v2',
        'traced_at': _time.strftime('%Y-%m-%dT%H:%M:%S'),
        'voxel_um': list(params.voxel_um) if params else None,
        'psf': (params.psf_label or None) if params else None,
        'psf_family': params.psf_family if params else None,
    }
    _a, debug = build_chromatin_trace_allele(
        allele, hybes, reference_hybe, fid_ch, read_ch, storage_path, fov,
        modality, cell, fov_matrices, params=params,
        max_fiducial_drift=max_drift, max_fiducial_drift_z=max_drift_z,
        spad=spad, collect_debug=True, resolver=resolver)
    return ((int(meta['id']), allele.fiducial_trace_adj, allele.polymer_adj,
             allele.fiducial_trace_raw, allele.polymer_raw,
             allele.rejected_hybes, getattr(allele, 'reference_warning', None),
             dict(allele.provenance or {})), debug)


def apply_allele_result(allele, result):
    """Merge a child's result into the parent's own AnAllele, in place."""
    (_aid, fiducial_trace_adj, polymer_adj, fiducial_trace_raw, polymer_raw,
     rejected, warning, provenance) = result
    allele.fiducial_trace_adj = fiducial_trace_adj
    allele.polymer_adj = polymer_adj
    allele.fiducial_trace_raw = fiducial_trace_raw
    allele.polymer_raw = polymer_raw
    allele.rejected_hybes = rejected
    if warning:
        allele.reference_warning = warning
    if provenance:
        allele.provenance = provenance
    return allele


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
                 resolver=None, z_boundary_trim=0, executor=None):
    """
    Route one allele to the chosen engine. The ONE place the choice is
    made, so the preview and the batch run cannot diverge -- they took
    different code paths to the same v1 call before, and a switch added to
    only one of them would be invisible until the two disagreed.

    NEITHER engine is offered an `append` mode here. Which alleles run is
    decided by membership before the worker starts, so an allele arriving
    here never has a committed trace to merge into. v1 still HAS the
    parameter for direct callers (it is the reference implementation and
    stays unchanged); the dispatcher simply never asks for it.

    v1 keeps every other argument it had. The ones v2 has no use for are dropped
    rather than accepted and ignored: z_window is a mixture-mode seed
    search (v2 fits one emitter), z_boundary_trim shaves the pillar ends
    (v2 fits a box placed at the consensus depth, so the ends are already
    out of the domain), and the per-channel *_params carry v1's gate
    constants, which do not transfer to a local background.
    """
    # STAMP HOW THIS TRACE WAS MADE, whichever engine runs. Free-form on
    # purpose: each engine records its own inputs, so a future engine adds
    # its hyperparameters here without anything else changing. The base is
    # only what is true of every engine.
    if allele is not None and hasattr(allele, 'provenance'):
        import time as _time
        stamp = {'engine': 'v2' if is_v2(engine) else 'v1',
                 'engine_label': str(engine or ''),
                 'traced_at': _time.strftime('%Y-%m-%dT%H:%M:%S')}
        if is_v2(engine) and v2_params is not None:
            stamp.update({
                'voxel_um': list(v2_params.voxel_um),
                'psf': v2_params.psf_label or None,
                'psf_family': v2_params.psf_family,
                'fiducial_gates': {k: v for k, v in v2_params.fiducial_gates.items()
                                   if not isinstance(v, tuple)},
                'readout_gates': {k: v for k, v in v2_params.readout_gates.items()
                                  if not isinstance(v, tuple)},
            })
        allele.provenance = stamp
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
            executor=executor)
    return build_chromatin_trace_allele(
        allele, hybes, reference_hybe, hybe_fiducial_channels,
        hybe_readout_channels, storage_path, fov, modality, cell, fov_matrices,
        params=v2_params, max_fiducial_drift=max_fiducial_drift,
        max_fiducial_drift_z=max_fiducial_drift_z, spad=spad,
        collect_debug=collect_debug, resolver=resolver)


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
