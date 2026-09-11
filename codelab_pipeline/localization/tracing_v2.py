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
import os
from collections import namedtuple

import numpy as np

from codelab_pipeline.localization import fit3d_um as U
from codelab_pipeline.localization import fiducial_xcorr as FX
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
# fallback site in the trace loop for the basin story and the accuracy
# numbers. TIME cost, measured on MP58 FOV1 (127 alleles x 73 hybes,
# 32-worker pool): 97.5 -> 113.9 s wall, +16.8%. The delta closes
# arithmetically as pure retries -- ~12.4 gate-failed fiducials per
# allele x ~0.33 s per fit x 32 workers -- so the fallback costs exactly
# what it fits and nothing else. For scale, allele-level parallelism
# itself is ~19x on this machine (36.4 min serial -> 1.9 min), so the
# retry costs about 16 s per FOV against the 34 min the pool saves.
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
                 readout_gates=None, qc_shift=True,
                 readout_engine=None, min_p_exist=None, min_p3=None,
                 readout_model_dir=None, engine_label='',
                 z_window=None, z_boundary_trim=10,
                 fiducial_model_dir=None, lateral_reach_px=None,
                 fiducial_z_window=None, min_p_exist_fiducial=None,
                 genomic_resolution_kb=None, template_mode='select',
                 fiducial_method=None, fiducial_refine='gauss',
                 fiducial_min_ncc=None, xcorr_template_half=None):
        self.voxel_um = tuple(float(v) for v in voxel_um)
        # HOW THE FIDUCIAL IS FOUND on every hybe but the reference:
        # 'xcorr' -- ChrTracer3's way, the hybe's crop REGISTERED to the
        # reference's by 3D normalized cross-correlation
        # (fiducial_xcorr.register), the reference itself still placed
        # by the Gaussian or the learned engine -- or 'gaussian' (v2's
        # fit per hybe, or the learned engine when a fiducial model is
        # set, subtracted). THE DEFAULT IS THE CORRELATION, by
        # measurement (tools/fiducial_ab.py, replicate distance on the
        # common pairs, 48 alleles per experiment): MP58 0.086 um vs
        # the Gaussian's 0.166 (-47%) and the learned fiducial's 0.101;
        # JP chr19 -45% with 97 pairs against 78; HoxA the Gaussian's
        # coverage. With the learned readout (v3) it kept the most pairs
        # of any arm (104 vs 88 for the learned fiducial) at -43%. The
        # Gaussian stays selectable on the v3 page and in the A/B.
        # And how a learned candidate WITHOUT a sub-voxel position is
        # refined: 'gauss' (v2's fit seeded there) or 'xcorr'
        # (correlation with the reference around it).
        self.fiducial_method = str(fiducial_method or DEFAULT_FIDUCIAL_METHOD)
        self.fiducial_refine = str(fiducial_refine or 'gauss')
        if self.fiducial_method not in ('gaussian', 'xcorr'):
            raise ValueError(f'fiducial_method {self.fiducial_method!r}')
        if self.fiducial_refine not in ('gauss', 'xcorr'):
            raise ValueError(f'fiducial_refine {self.fiducial_refine!r}')
        self.fiducial_min_ncc = (None if fiducial_min_ncc is None
                                 else float(fiducial_min_ncc))
        self.xcorr_template_half = (None if xcorr_template_half is None
                                    else tuple(int(v) for v in xcorr_template_half))
        if self.xcorr_template_half is not None and (
                len(self.xcorr_template_half) != 3
                or any(v < 1 for v in self.xcorr_template_half)):
            raise ValueError(f'xcorr_template_half {xcorr_template_half!r}: '
                             f'three half-sizes (y, x, z) of at least 1')
        # Which PSF template a learned engine matches with when its bank
        # carries one per experiment: 'select' by this experiment's
        # resolution (the default), 'pooled', or 'all' (A/B only).
        self.template_mode = str(template_mode or 'select')
        # kb per readout step, from the Ingestion tab; a learned head
        # trained with it as a feature needs it to score. None = unknown.
        self.genomic_resolution_kb = (float(genomic_resolution_kb)
                                      if genomic_resolution_kb else None)
        # THREE SETTINGS THE CODE HAD AND THE PANEL DID NOT. Each is None
        # for the measured value the code used before it was a setting:
        # the readout's lateral reach (5 px, the 1 um v2's readout fit
        # may move), the learned fiducial's depth window (17 planes, the
        # Gaussian's seed window), and the fiducial's own p_exist
        # threshold (the readout's, when unset -- one number gated both
        # models, and they are different models).
        self.lateral_reach_px = (None if lateral_reach_px is None
                                 else int(lateral_reach_px))
        self.fiducial_z_window = (None if fiducial_z_window is None
                                  else int(fiducial_z_window))
        self.min_p_exist_fiducial = (None if min_p_exist_fiducial is None
                                     else float(min_p_exist_fiducial))
        # A SECOND MODEL FOR THE FIDUCIAL, optional. Trained on the
        # fiducial channel, it calls fiducial spots the way the readout
        # model calls readouts -- but the fiducial phase takes BEST OF
        # ONE: the highest-p_exist candidate is the alignment, because a
        # hybe's frame has one answer, never a list. None keeps v2's
        # Gaussian fiducial, which is the default and stays the
        # reference the learned one is measured against.
        self.fiducial_model_dir = (str(fiducial_model_dir)
                                   if fiducial_model_dir else None)
        self._fiducial_engine = None
        # THE PANEL'S TWO Z CONTROLS, honoured by v2 and v3 alike. They
        # were v1's -- accepted by the v2 dispatcher and ignored, and
        # invisible once v3 had its own page -- so a person turning them
        # changed nothing for two of the three engines. z_window: how far
        # in planes from the fiducial's z a readout is looked for (None =
        # the measured fit radius); z_boundary_trim: planes shaved off
        # each stack end, where everything is out of focus.
        self.z_window = None if z_window is None else int(z_window)
        self.z_boundary_trim = int(z_boundary_trim or 0)
        # THE MODEL DIRECTORY, NOT THE ENGINE, is what crosses a process
        # boundary. An engine holds torch weights and an HDF5 bank; a
        # V2Params is pickled into every allele_task payload. The engine
        # is built on first use in whichever process asks, and dropped
        # from the pickle (__getstate__).
        self.readout_model_dir = (str(readout_model_dir)
                                  if readout_model_dir else None)
        self.engine_label = str(engine_label or '')
        self._readout_engine = readout_engine
        # None = no calibrated PSF, so the readout sigma is fitted per spot
        # like the fiducial's. Supported, but it gives up both the accuracy
        # and the 37% speed the fixed shape buys.
        self.psf_family = psf_family
        self.psf_shape = tuple(psf_shape) if psf_shape else None
        self.psf_label = psf_label
        self.fiducial_gates = dict(FIDUCIAL_GATES, **(fiducial_gates or {}))
        self.readout_gates = dict(READOUT_GATES, **(readout_gates or {}))
        self.qc_shift = bool(qc_shift)
        # THE READOUT MAY BE LOCALIZED BY A LEARNED ENGINE, and only the
        # readout. The fiducial's job is to place this hybe's frame
        # against the reference, which is a question with ONE answer per
        # hybe -- a second fiducial candidate is not a second alignment,
        # it is an ambiguity, and fiducial_trace_adj carries one tuple
        # per hybe precisely because that is the contract. So the
        # fiducial phase is untouched by this and always fits one.
        #
        # The readout is the opposite: two real loci in one hybe (sister
        # chromatids) are a fact this pipeline already models --
        # polymer_adj holds a LIST per hybe and AnAllele's own docstring
        # says they are kept side by side and never pruned against each
        # other. v2 has only ever written lists of length one because
        # fit_readout returns one fit. A multispot engine here fills the
        # shape that was always declared.
        # p_exist GATES HERE, exactly the way max_uncert does: cut before
        # the write. It is ALSO written, as the fifth slot of every stored
        # spot (AnAllele: (y, x, z, amplitude, quality)) -- not for the
        # tracer, which never reads it back, but for a judgement made
        # later (review, a threshold chosen after the fact) that would
        # otherwise have to re-trace to get the number the engine had.
        self.min_p_exist = (None if min_p_exist is None
                            else float(min_p_exist))
        # AND THE MATCHER'S OWN GATE, AS A CALIBRATED PROBABILITY. The
        # engine's shipped Platt pair turns its raw NCC remap into one,
        # so this is 0.5 by construction rather than 0.732 by
        # measurement -- and it is applied HERE rather than written back
        # into the spot, because p and p_exist each mean exactly one
        # thing and a gate is where a decision belongs.
        #
        # THE TWO GATES ARE NOT THE SAME QUESTION and neither replaces
        # the other. The classifier answers P(a reviewer keeps this |
        # it is a candidate), from pass/fail labels, per BOX -- so
        # siblings out of one box share it. The calibrated matcher score
        # answers P(a reviewer calls this real | it is a match beside a
        # confirmed spot), from multispot labels, per MATCH -- which is
        # the only one of the two that can tell siblings apart.
        self.min_p3 = None if min_p3 is None else float(min_p3)

    def z_reach(self):
        """Readout axial reach in planes: the panel's, else the measured."""
        return (self.z_window if self.z_window is not None
                else _seed_z_half(READOUT_FIT_RADIUS_UM, self.voxel_um))

    def lateral_reach(self):
        """Readout lateral reach in px from the fiducial: the panel's,
        else the measured 1 um (5 px at 0.208 um)."""
        return (self.lateral_reach_px if self.lateral_reach_px is not None
                else _lateral_reach_px(READOUT_FIT_RADIUS_UM, self.voxel_um))

    def fiducial_window(self):
        """The learned fiducial's depth window in planes: the panel's,
        else the Gaussian's own seed window (17 at 0.2 um)."""
        return (self.fiducial_z_window if self.fiducial_z_window is not None
                else fiducial_window_planes(self.voxel_um))

    def fiducial_min_p(self):
        """The learned fiducial's p1 threshold: its own, else the measured
        default FIDUCIAL_MIN_P1 (0.3) -- not the readout's."""
        return (self.min_p_exist_fiducial
                if self.min_p_exist_fiducial is not None else FIDUCIAL_MIN_P1)

    def template_half(self):
        """The correlation template's (y, x, z) half-sizes in voxels."""
        return (self.xcorr_template_half
                if self.xcorr_template_half is not None
                else FX.DEFAULT_TEMPLATE_HALF)

    def min_ncc(self):
        """The correlation fiducial's acceptance: peak NCC at least this."""
        return (self.fiducial_min_ncc if self.fiducial_min_ncc is not None
                else FIDUCIAL_MIN_NCC)

    @property
    def readout_engine(self):
        """The learned readout engine, built on first use, or None."""
        if self._readout_engine is None and self.readout_model_dir:
            from .engine import make_engine
            self._readout_engine = make_engine(ROUTE_V3,
                                               model_dir=self.readout_model_dir)
            self._give_context(self._readout_engine)
        return self._readout_engine

    def context(self):
        """Experiment-level facts a learned head may score with."""
        return {'genomic_resolution_kb': self.genomic_resolution_kb}

    def _give_context(self, engine):
        if engine is not None and hasattr(engine, 'context'):
            engine.context = dict(getattr(engine, 'context', None) or {},
                                  **self.context())
            if hasattr(engine, 'template_mode'):
                engine.template_mode = self.template_mode

    @readout_engine.setter
    def readout_engine(self, value):
        self._readout_engine = value

    @property
    def is_learned(self):
        return bool(self.readout_model_dir or self._readout_engine is not None)

    @property
    def fiducial_engine(self):
        """The learned fiducial engine, built on first use, or None."""
        if self._fiducial_engine is None and self.fiducial_model_dir:
            from .engine import make_engine
            self._fiducial_engine = make_engine(
                ROUTE_V3, model_dir=self.fiducial_model_dir)
            self._give_context(self._fiducial_engine)
        return self._fiducial_engine

    @fiducial_engine.setter
    def fiducial_engine(self, value):
        self._fiducial_engine = value

    def __getstate__(self):
        d = dict(self.__dict__)
        d['_readout_engine'] = None      # rebuilt in the child from the dir
        d['_fiducial_engine'] = None
        return d

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
        v3 = params.get('v3') or {}
        learned = route(params.get('engine', '')) == ROUTE_V3
        trim = int(params.get('z_boundary_trim', 10) or 0)
        fg = dict(v2.get('fiducial') or {}, z_boundary_trim=trim)
        rg = dict(v2.get('readout') or {}, z_boundary_trim=trim)
        return cls(voxel_um=voxel, psf_family=fam, psf_shape=shape,
                   psf_label=label,
                   fiducial_gates=fg,
                   readout_gates=rg,
                   qc_shift=v2.get('qc_shift', True),
                   z_window=params.get('z_window'),
                   z_boundary_trim=trim,
                   readout_model_dir=(v3.get('model_dir') if learned else None),
                   fiducial_model_dir=(v3.get('fiducial_model_dir')
                                       if learned else None),
                   min_p_exist=(v3.get('min_p_exist') if learned else None),
                   min_p_exist_fiducial=(v3.get('min_p_exist_fiducial')
                                         if learned else None),
                   lateral_reach_px=(v3.get('lateral_reach_px')
                                     if learned else None),
                   fiducial_z_window=(v3.get('fiducial_z_window')
                                      if learned else None),
                   # the v3 page's fiducial method for v3; v2 has no
                   # control for it and takes the measured default
                   fiducial_method=((v3.get('fiducial_method')
                                     or DEFAULT_FIDUCIAL_METHOD)
                                    if learned else DEFAULT_FIDUCIAL_METHOD),
                   genomic_resolution_kb=params.get('genomic_resolution_kb'),
                   engine_label=str(params.get('engine_label') or
                                    params.get('engine') or ''))

    def describe(self):
        """One line that reconstructs the run, the fiducial's method
        included when it is not the engine's own."""
        line = self._describe_engine()
        if self.fiducial_method == 'xcorr':
            line += (f'; fiducial by 3D correlation with the reference '
                     f'(ncc >= {self.min_ncc():g}, template half '
                     f'{self.template_half()})')
        elif self.fiducial_refine == 'xcorr' and self.fiducial_model_dir:
            line += '; unrefined learned fiducials refined by correlation'
        return line

    def _describe_engine(self):
        """One line that reconstructs the run.

        BOTH branches name the voxel size and whatever is known about the
        PSF. The fallback branch used to say only "no calibrated PSF
        installed" -- which is wrong when one IS installed and was
        REJECTED, discards the reason plausible() computed, and omits the
        voxel size entirely. A run that silently becomes a different run
        (readout sigma free instead of fixed) has to leave a record saying
        which run it became.
        """
        if self.is_learned:
            t = ('none' if self.min_p_exist is None
                 else f'{self.min_p_exist:g}')
            fp = self.fiducial_min_p()
            fid = (f'fiducial best-of-one from '
                   f'{os.path.basename(str(self.fiducial_model_dir))} '
                   f'(p_exist >= {"none" if fp is None else f"{fp:g}"}, '
                   f'window {self.fiducial_window()} planes)'
                   if self.fiducial_model_dir else 'fiducial by v2 Gaussian')
            return (f'v3, readout by the learned engine from '
                    f'{os.path.basename(str(self.readout_model_dir or "?"))} '
                    f'(p_exist >= {t}); {fid}; voxel {self.voxel_um}')
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


def _z_trim_offset(depth, z_boundary_trim, min_fit_depth=9):
    """How many planes to shave off EACH end -- v1's own rule, kept so
    the two engines agree: the outermost planes of a stack are out of
    focus wherever the allele sits, and the trim is clamped so at least
    min_fit_depth planes remain."""
    if not z_boundary_trim or z_boundary_trim <= 0:
        return 0
    return max(0, min(int(z_boundary_trim), (int(depth) - min_fit_depth) // 2))


def _lateral_reach_px(fit_radius_um, voxel_um):
    """Lateral reach in PIXELS, from the same radius the axial one uses."""
    return max(1, int(round(fit_radius_um[0] / float(voxel_um[0]))))


LearnedFit = namedtuple('LearnedFit',
                        ['y', 'x', 'z', 'amplitude', 'p_exist', 'at_bound'])

# THE FIDUCIAL'S GATE ON p1, BY DEFAULT 0.3 -- not the readout's 0.5.
# MEASURED on MP58 (48 alleles, out-of-sample FOVs, pooled ch555 model):
#
#     fiducial p1 >=   coverage   common-pair median   vs v2 Gaussian
#     0.5              88.2%      0.098 um             -46% (52 pairs)
#     0.3              93.2%      0.101 um             -40% (62 pairs)
#     (v2 Gaussian)    96.4%      0.166 um
#
# The fiducials the classifier is only 0.3-0.5 sure of align as well as
# the ones it is sure of; refusing them cost a twentieth of the hybes
# for nothing. The readout keeps 0.5: a readout candidate is a trace
# position and a false one is a wrong distance, while a fiducial is one
# alignment per hybe that the drift gate still checks.
FIDUCIAL_MIN_P1 = 0.3

# THE CORRELATION FIDUCIAL'S GATE: peak NCC between the reference's
# template and this hybe's crop. A number to MEASURE, not to reason
# about: MATLAB's Register3D accepts its SSD score at 0.99 and otherwise
# falls back to 2D projections. Reported per hybe by tools/fiducial_ab.py
# (fiducial_ncc) so the threshold is set from a real distribution.
FIDUCIAL_MIN_NCC = 0.5
# The fiducial method every engine uses unless told otherwise. See the
# measurement at V2Params.__init__.
DEFAULT_FIDUCIAL_METHOD = 'xcorr'
# How far a learned candidate may move when the correlation refines it:
# it named the voxel, the correlation names the sub-voxel position.
XCORR_REFINE_REACH = (2, 2, 3)
# The correlation fiducial searches this much PAST the drift gate, so a
# drift right at the gate is bracketed (a peak on the search edge is
# refused as unmeasured) and the gate in phase 3 makes the decision.
XCORR_SEARCH_MARGIN = 2


def alt_marker(why):
    """A short tag beside a rejected candidate's p_exist: WHY it lost.

        z    outside the depth window / z reach
        xy   outside the fiducial's lateral reach
        e    within the trimmed planes at a stack end
        ~    no sub-voxel position (model 3 placed no peak)
        <    below min p_exist
        g    the engine's own candidate, which the Gaussian fit refined
        (none)  'not the best': a lower p_exist inside the window -- and
                ties lose on decimals the two-place label does not show

    Two grids asked the same question -- 'why was this p 1.00 candidate
    not taken?' -- and a label that is only a p cannot answer it.
    """
    w = str(why or '')
    if 'px from the fiducial' in w:
        return ' xy'
    if 'stack end' in w:
        return ' e'
    if 'beyond the fiducial window' in w or 'planes from' in w:
        return ' z'
    if 'no sub-voxel' in w:
        return ' ~'
    if 'refined by the Gaussian' in w:
        return ' g'
    if 'p_exist' in w or w.startswith('p1 '):
        return ' <'
    return ''


def _learned_slab(cube, z_c, p, reach=None):
    """The z-slab a learned engine searches around z_c: the reach (the
    panel's, unless the caller has its own) padded by the engine's own
    box, never into the trimmed ends. Returns (slab, z0, z1, reach,
    z_off)."""
    from .psfmatcher import BOX_RZ
    reach = p.z_reach() if reach is None else int(reach)
    depth = int(np.asarray(cube).shape[2])
    z_off = _z_trim_offset(depth, p.z_boundary_trim)
    if z_c is not None and np.isfinite(z_c):
        z0 = max(z_off, int(round(float(z_c))) - reach - BOX_RZ)
        z1 = min(depth - z_off, int(round(float(z_c))) + reach + BOX_RZ + 1)
    else:
        z0, z1 = z_off, depth - z_off
    if z1 <= z0:
        return None, z0, z1, reach, z_off
    return np.asarray(cube)[:, :, z0:z1], z0, z1, reach, z_off


def fiducial_window_planes(voxel_um):
    """How far from the expected depth a learned fiducial is looked for:
    the SAME window v2's Gaussian fiducial seeds in (its fit radius,
    3.6 um -> 17 planes), not the readout's reach (14).

    MEASURED, MP58/DNA, 48 alleles x 73 hybes, learned fiducial from the
    ch555 model gated at the readout reach: 656 of 994 refusals were
    'every candidate out of z reach', the refused candidate a median 17
    planes from the expected depth, and the Gaussian arm had found a
    fiducial in 96% of exactly those hybes -- the same ten hybes in
    nearly every allele. That is a per-hybe depth offset the consensus
    placement does not carry, which the Gaussian tolerates because its
    seed window is 17 planes and its fit may move further, with the
    drift gate against the reference (phase 3) as the real acceptance.
    So the learned fiducial gets the Gaussian's window, and past it the
    drift gate decides, as it does for the Gaussian.
    """
    return _seed_z_half(FIDUCIAL_FIT_RADIUS_UM, voxel_um)


def fiducial_quality(f):
    """The stored quality of a fiducial: the number its method gates on.
    p_exist for a learned candidate (also one the Gaussian or the
    correlation refined), the peak NCC for a correlation fiducial, NaN
    for a Gaussian fit, which has no single such number."""
    for name in ('p_exist', 'ncc'):
        v = getattr(f, name, None)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                return float('nan')
    return float('nan')


def _xcorr_gate(f, why, cube, p):
    """(passed, reason) for a correlation fiducial: it must exist, its
    peak must be bracketed (not on the search bound), its NCC at least
    the threshold, and its depth clear of the trimmed stack ends. No
    occupancy, no CI: a registration has neither."""
    if f is None:
        return False, str(why or 'registration failed')
    if f.at_bound:
        return False, f'shift at the search bound ({", ".join(f.at_bound)})'
    thr = p.min_ncc()
    if not np.isfinite(f.ncc) or f.ncc < thr:
        return False, f'ncc {f.ncc:.2f} < {thr:.2f}'
    return gate(f, cube, {'z_boundary_trim': p.fiducial_gates.get('z_boundary_trim'),
                          'reject_at_bound': False}, p.voxel_um)


def _fiducial_learned(cube, z0, p, ref=None):
    """The fiducial by the learned engine, BEST OF ONE.

    Returns (fit, why, alternatives, how): `fit` is a LearnedFit -- or a
    Gaussian fit when the learned candidate was refined by one -- or
    None, `why` the reason when None, `alternatives` every other
    candidate in the crop as (LearnedFit, why) for the grid, and `how`
    says which way the answer came:

        'v3'               the best p_exist inside the fiducial window
        'v3 beyond window' nothing inside it; the best in the slab, and
                           the drift gate against the reference decides
        'v3+gauss'         the engine believed a spot but placed no
                           sub-voxel peak; v2's Gaussian fit, seeded
                           there and passed through v2's own gates,
                           gives it the position
        'v3+xc'            the same case with p.fiducial_refine ==
                           'xcorr' and `ref` = (reference crop, its
                           fiducial): correlation with the reference's
                           template, searched XCORR_REFINE_REACH around
                           the candidate, gives it the position

    One answer, not a list: the fiducial's job is to place this hybe's
    frame, and a second candidate is an ambiguity rather than a second
    alignment. Within a tier the choice is the highest p_exist above
    the threshold.

    WHY THE TWO FALLBACKS. Measured on MP58/DNA (see
    fiducial_window_planes): with the readout reach and no refinement,
    the learned fiducial covered 71.6% of hybes against the Gaussian's
    96.4%, and where both existed they agreed to 0.4 px -- so the loss
    was almost entirely hybes the Gaussian handled and this path refused
    on its own stricter, unmeasured rules. 15% of refusals were
    'unrefined': model 1 said spot, model 3 found no peak, and
    psfmatcher.is_refined records that 3 of 4 such spots were real by
    human judgement. Dropping the hybe over a missing sub-voxel step
    when a Gaussian fit can supply it was the wrong trade.
    """
    window = p.fiducial_window()
    slab, s0, s1, reach, z_off = _learned_slab(cube, z0, p, reach=window)
    if slab is None:
        return None, 'no planes left between the trimmed ends', [], None
    try:
        cands = p.fiducial_engine.localize(slab, seed_yxz=None, n_max=None)
    except Exception as exc:                                # noqa: BLE001
        return None, f'engine failed: {type(exc).__name__}: {exc}', [], None
    cands = [c._replace(z=float(c.z) + s0) for c in cands]
    h, w = np.asarray(cube).shape[:2]
    cands = [c for c in cands
             if -0.5 <= float(c.y) < h - 0.5 and -0.5 <= float(c.x) < w - 0.5]
    if not cands:
        return None, 'engine found nothing', [], None
    from . import psfmatcher as PSFM
    t = p.fiducial_min_p()
    # THE GATE IS p1, THE RANK IS p_exist. p_exist = p1 x cal(p3): the
    # classifier's belief that a spot is here, times the multispot
    # calibration's belief that THIS match is a real emitter among
    # several in a pillar. The second factor answers a different
    # question from 'is there a fiducial here', and a steep calibration
    # (fitted on a reviewer's multispot standard) refused real
    # fiducials wholesale -- MEASURED on three experiments: coverage
    # 84.6 / 49.5 / 66.8% against the Gaussian's 96 / 84 / 98%, nearly
    # every refusal 'below min p_exist', while the fiducials that passed
    # aligned better than the Gaussian's (-50% repeat distance on MP58).
    # A fiducial at NCC 0.7 is still a sub-voxel position. So the
    # threshold applies to p1, recovered exactly as p_exist / cal(p3)
    # when the engine has a calibration, and the ranking keeps p_exist.
    cal = getattr(p.fiducial_engine, 'multispot_cal', None)
    kb = (getattr(p.fiducial_engine, 'context', None) or {}).get(
        'genomic_resolution_kb')

    def p1_of(c):
        pe = float(c.p_exist)
        if cal is None or not np.isfinite(c.p):
            return pe
        q = cal.score(float(c.p)) if kb is None else cal.score(
            float(c.p), resolution_kb=kb)
        q = float(np.asarray(q).ravel()[0])
        return float(min(1.0, pe / q)) if q > 0 else pe

    inside, beyond, unrefined, alts = [], [], [], []
    for c in cands:
        dzr = (abs(float(c.z) - float(z0))
               if z0 is not None and np.isfinite(z0) else 0.0)
        lf = LearnedFit(float(c.y), float(c.x), float(c.z),
                        float(c.amplitude) if np.isfinite(c.amplitude) else 0.0,
                        float(c.p_exist), ())
        p1 = p1_of(c)
        if t is not None and not (p1 >= t):
            alts.append((lf, f'p1 {p1:.2f} < {t:g} (p_exist {float(c.p_exist):.2f})'))
        elif not PSFM.is_refined(c):
            unrefined.append((lf, dzr))
        elif dzr > window:
            beyond.append((lf, dzr))
        else:
            inside.append(lf)
    if inside:
        inside.sort(key=lambda f: -f.p_exist)
        best = inside[0]
        rest = ([(f, 'not the best') for f in inside[1:]]
                + [(f, f'beyond the fiducial window ({d:.1f} > {window} '
                       f'planes from expected)') for f, d in beyond]
                + [(f, 'no sub-voxel position') for f, _d in unrefined]
                + alts)
        return best, None, rest, 'v3'
    if beyond:
        beyond.sort(key=lambda fd: -fd[0].p_exist)
        best, d = beyond[0]
        rest = ([(f, f'not the best ({dd:.1f} planes from expected)')
                 for f, dd in beyond[1:]]
                + [(f, 'no sub-voxel position') for f, _d in unrefined]
                + alts)
        return best, None, rest, 'v3 beyond window'
    if unrefined:
        # THE ENGINE BELIEVES IT, MODEL 3 COULD NOT PLACE IT. Give the
        # integer position to v2's Gaussian fit as its seed and hold the
        # result to v2's own gates -- the same path the Gaussian fiducial
        # takes, only seeded by the learned engine instead of the
        # centroid.
        unrefined.sort(key=lambda fd: -fd[0].p_exist)
        cand, _d = unrefined[0]
        rest = ([(f, 'no sub-voxel position') for f, _dd in unrefined[1:]]
                + alts)
        if p.fiducial_refine == 'xcorr' and ref is not None:
            ref_cube, ref_pos = ref
            xf, xwhy = FX.register(ref_cube, ref_pos, cube,
                                   (cand.y, cand.x, cand.z),
                                   max_shift=XCORR_REFINE_REACH,
                                   half=p.template_half())
            xok, xwhy = _xcorr_gate(xf, xwhy, cube, p)
            if xok:
                return xf, None, \
                    [(cand, 'refined by correlation with the reference')] \
                    + rest, 'v3+xc'
            return None, f'unrefined; xcorr {xwhy}', \
                [(cand, 'no sub-voxel position; correlation refinement '
                        'failed')] + rest, None
        g = fit_fiducial_from(cube, (cand.y, cand.x, cand.z), p)
        ok, gwhy = gate(g, cube, p.fiducial_gates, p.voxel_um)
        if ok and g is not None:
            return g, None, [(cand, 'refined by the Gaussian fit')] + rest, \
                'v3+gauss'
        # A TILE TITLE, so it is short: the gate that refused is the part
        # a person needs, and the long form clipped it off the tile.
        # Reads 'fiducial unrefined; refit occupancy 0.16 < 0.25' there.
        return None, f'unrefined; refit {gwhy}', \
            [(cand, 'no sub-voxel position; Gaussian refinement failed')] \
            + rest, None
    return None, (f'no candidate at p1 >= {t:g}'
                  if t is not None else 'no candidate'), alts, None


def _readout_multi(allele, hybe, cube, z_r, p, dy, dx, dz, ymin, xmin,
                   to_shared, debug=None, seed_yx=None, display_offset=(0, 0),
                   display_shape=None):
    """Every readout candidate in one crop, via a learned engine.

    Returns (wrote_anything, reason). Fills polymer_adj / polymer_raw with
    a LIST -- the shape AnAllele has always declared and v2 has never
    used, because fit_readout returns one fit.

    THE p_exist GATE IS HERE AND ONLY HERE. A candidate below the
    threshold is dropped before anything is written, so the stored tuple
    stays (y, x, z, amplitude) and no persisted contract moves for a
    number whose whole job is finished at this line. That is how
    max_uncert already works one function up.

    amplitude is the engine's own -- for v3 a MEASURED PEAK above the
    region background, not a fitted Gaussian amplitude (psfmatcher.
    _peak_above says why). analysis.polymer.max_brightness compares it,
    so it has to be finite and comparable within this crop; it is not a
    quantity to compare against a v1/v2 amplitude from another run.
    """
    # THE SEARCH IS THE FIDUCIAL'S SLAB, not the whole column. The crop
    # carries every plane of the stack, and a learned engine handed all
    # of it found real emitters forty planes from this locus -- other
    # cells' spots in the same column -- which the reach gate then
    # rejected, and the grid drew them off the ZX image because the
    # window it shows is the reach. Cutting the slab first means such a
    # candidate never forms, the engine does a third of the work, and
    # everything it returns is inside the picture a person sees. Lateral
    # extent stays the crop's: the engine's own boxes are 15 px wide and
    # need room; reach is gated below instead.
    # PADDED BY THE ENGINE'S OWN BOX, then gated to the reach. A slab cut
    # to exactly the reach hands the engine a candidate at its edge with
    # no room for the 25-plane feature box or the 11-plane template --
    # the same boundary pinning the matcher had laterally before its
    # window was widened. So the search runs on reach + BOX_RZ each way,
    # and the reach is applied to what comes back. The trim is applied
    # to the slab itself: shaved planes are never searched.
    from .psfmatcher import BOX_RZ
    z_half = p.z_reach()
    depth = int(np.asarray(cube).shape[2])
    z_off = _z_trim_offset(depth, p.z_boundary_trim)
    if z_r is not None and np.isfinite(z_r):
        z0 = max(z_off, int(round(float(z_r))) - z_half - BOX_RZ)
        z1 = min(depth - z_off, int(round(float(z_r))) + z_half + BOX_RZ + 1)
    else:
        z0, z1 = z_off, depth - z_off
    if z1 <= z0:
        return False, (f'readout: the stack has no planes left between the '
                       f'trimmed ends ({z_off} each side)')
    slab = np.asarray(cube)[:, :, z0:z1]
    try:
        cands = p.readout_engine.localize(slab, seed_yxz=None, n_max=None)
        cands = [c._replace(z=float(c.z) + z0) for c in cands]
    except Exception as exc:                                # noqa: BLE001
        # A tracing run must not die on one crop. Same rule the rest of
        # this module keeps: a hybe that cannot be fitted is a rejected
        # hybe with a reason, never an exception out of a worker.
        return False, f'readout engine failed: {type(exc).__name__}: {exc}'
    # THE DISPLAY BOX IS A HARD BOUNDARY, not a viewing choice. The search
    # crop is wider than the fiducial's box by the engine's own margin so
    # a candidate near the box edge gets full template room; a candidate
    # whose LOCALIZED position lies outside that box is a neighbour the
    # margin caught, and it is removed here, silently -- never counted,
    # never listed, never drawn. What survives is what is in the box.
    if display_shape is not None:
        oy, ox = (float(display_offset[0]), float(display_offset[1]))
        h_d, w_d = int(display_shape[0]), int(display_shape[1])
        cands = [c for c in cands
                 if -0.5 <= float(c.y) - oy < h_d - 0.5
                 and -0.5 <= float(c.x) - ox < w_d - 0.5]
    if debug is not None:
        debug[hybe]['readout_engine'] = 'v3'
        debug[hybe]['readout_p_exist'] = [float(c.p_exist) for c in cands]
        debug[hybe]['readout_n_before_p_gate'] = len(cands)
    if not cands:
        return False, 'readout found nothing'

    from . import p_gate as PG
    from . import psfmatcher as PSFM
    # THE FIDUCIAL'S OWN AXIAL REACH. The crop is the box the fiducial
    # anchors, and z_r is where this hybe's fiducial says the locus sits;
    # a candidate further from that in z than v2's own readout search
    # would have looked is a spot with a different z-drift from the
    # fiducial's, which is exactly the spot this trace must not carry.
    # Lateral reach needs no gate: the crop's own edges are it.
    r_lat = p.lateral_reach()
    if seed_yx is None:
        h, w = np.asarray(cube).shape[:2]
        seed_yx = ((h - 1) / 2.0, (w - 1) / 2.0)
    dropped = []                    # (cand, why) for the grid
    in_reach = []
    for c in cands:
        dzr = (abs(float(c.z) - float(z_r))
               if z_r is not None and np.isfinite(z_r) else 0.0)
        dlat = max(abs(float(c.y) - float(seed_yx[0])),
                   abs(float(c.x) - float(seed_yx[1])))
        if dzr > z_half:
            dropped.append((c, f'z {dzr:.1f} planes from the fiducial '
                               f'> {z_half}'))
        elif z_off and (float(c.z) < z_off or float(c.z) > depth - 1 - z_off):
            dropped.append((c, f'z {float(c.z):.1f} within {z_off} planes '
                               f'of the stack end'))
        elif dlat > r_lat:
            # THE FIDUCIAL'S OWN LATERAL REACH, the same +/-1 um v2's
            # readout fit is allowed to move from its seed. A candidate
            # further out is a neighbour the crop happened to include.
            dropped.append((c, f'{dlat:.1f} px from the fiducial > {r_lat}'))
        else:
            in_reach.append(c)
    t = p.min_p_exist
    kept = (in_reach if t is None
            else PG.apply(in_reach, t, which=PG.CALIBRATED))
    for c in in_reach:
        if c not in kept:
            dropped.append((c, f'p_exist {float(c.p_exist):.2f} < {t:g}'))
    # AND ONLY CANDIDATES MODEL 3 ACTUALLY PLACED. polymer_adj is a list
    # of POSITIONS, and a candidate the matched filter could not place
    # keeps its anchor's INTEGER coordinate -- 208 nm here against a
    # localization precision near 30 nm. Writing that into a trace would
    # be an order-of-magnitude worse position wearing the same shape as
    # every other one, and nothing downstream could tell.
    #
    # THE SPOT IS NOT DELETED, it is only not TRACED. Detection keeps it
    # (psfmatcher.is_refined says why: MEASURED, 3 of the 4 such spots in
    # 613 labels were real), and it stays an ASpot with its p_exist. What
    # it does not get is a position it does not have.
    # AN EXTRA FLOOR ON THE MATCH ALONE, and off by default. p_exist
    # already carries cal(p3): psfmatcher._joint multiplies the pillar's
    # p1 by it per hit, so min_p_exist above is the joint cut and this
    # one exists only to say 'and the match itself must be at least this
    # good regardless of how much the classifier believed the pillar'.
    # Setting both means cutting one factor twice; that is a choice, not
    # a mistake, which is why neither has a default.
    cal = getattr(p.readout_engine, 'multispot_cal', None)
    if p.min_p3 is not None and cal is not None:
        before = len(kept)
        kept = [c for c in kept
                if np.isfinite(c.p) and float(cal.score(c.p)) >= p.min_p3]
        if debug is not None:
            debug[hybe]['readout_n_below_p3'] = before - len(kept)
    n_pre = len(kept)
    refined = [c for c in kept if PSFM.is_refined(c)]
    for c in kept:
        if c not in refined:
            dropped.append((c, 'no sub-voxel position'))
    kept = refined
    oy, ox = (float(display_offset[0]), float(display_offset[1]))
    if debug is not None:
        debug[hybe]['readout_n_unrefined'] = n_pre - len(kept)
        debug[hybe]['readout_n_out_of_reach'] = len(cands) - len(in_reach)
        debug[hybe]['readout_search'] = {'z0': int(z0), 'z1': int(z1),
                                         'reach': int(z_half),
                                         'lateral_px': int(r_lat),
                                         'trim': int(z_off)}
        # EVERY CANDIDATE REACHES THE GRID, kept or not, each with its
        # own p_exist: the whole point of looking at one allele is to
        # see the p_exist distribution and decide where to cut. Drawn in
        # the DISPLAY crop's coordinates: the search crop is wider by the
        # engine's box, and the grid shows the crop the fiducial shows.
        debug[hybe]['readout_rejected_centroids'] = (
            [(float(c.x) - ox, float(c.y) - oy, float(c.z))
             for c, _w in dropped] or None)
        debug[hybe]['readout_rejected_labels'] = (
            [f'{float(c.p_exist):.2f}{alt_marker(w)}' for c, w in dropped]
            or None)
        debug[hybe]['readout_dropped_why'] = [w for _c, w in dropped]
    if not kept:
        if not dropped:
            return False, 'readout found nothing'
        # THE BREAKDOWN, NOT THE LAST REASON. 'every candidate below
        # p_exist 0.5' on a tile whose 0.82 candidate had been cut by the
        # lateral reach read as a contradiction. Every dropped candidate
        # has its own reason; the title counts them by kind, in the same
        # tags the labels carry (alt_marker), short enough for a tile.
        names = {'xy': 'xy', 'z': 'z', 'e': 'end', '~': '~',
                 '<': (f'p<{t:g}' if t is not None else 'p')}
        counts = {}
        for _c, w in dropped:
            k = alt_marker(w).strip() or '?'
            counts[k] = counts.get(k, 0) + 1
        parts = [f'{counts[k]} {names.get(k, k)}'
                 for k in ('xy', 'z', 'e', '<', '~', '?') if k in counts]
        return False, 'readout: none kept -- ' + ', '.join(parts)

    adj, raw = [], []
    for c in kept:
        sy, sx, sz = to_shared(hybe, c.y, c.x, c.z, ymin, xmin)
        # THE FIFTH SLOT IS THE ENGINE'S OWN GATE NUMBER -- p_exist here
        # -- so a judgement made later has what the engine had.
        q = float(c.p_exist) if np.isfinite(c.p_exist) else float('nan')
        adj.append((float(sy + dy), float(sx + dx), float(sz + dz),
                    float(c.amplitude), q))
        raw.append((float(c.y + ymin), float(c.x + xmin), float(c.z),
                    float(c.amplitude), q))
    allele.polymer_adj[hybe] = adj
    allele.polymer_raw[hybe] = raw
    if debug is not None:
        debug[hybe]['readout_centroids'] = [(float(c.x) - ox, float(c.y) - oy,
                                             float(c.z)) for c in kept]
        debug[hybe]['readout_labels'] = [f'{float(c.p_exist):.2f}'
                                         for c in kept]
    return True, ''


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
    sy, sx, sz = _seed(cube, z_centre, p.voxel_um, p.z_reach())
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
    # THE STACK'S OUTERMOST PLANES ARE OUT OF FOCUS wherever the allele
    # sits (v1's boundary trim, its reasoning). v2's fit domain can reach
    # into them from a seed near an end; a fit that lands there is the
    # junk v1 refuses to look at, and is rejected with the plane named.
    trim = gates.get('z_boundary_trim')
    if trim:
        depth = int(np.asarray(cube).shape[2])
        off = _z_trim_offset(depth, trim)
        z = float(getattr(fit, 'z', float('nan')))
        if off and np.isfinite(z) and (z < off or z > depth - 1 - off):
            return False, f'z {z:.1f} within {off} planes of the stack end'
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
                                 params=None, max_fiducial_drift=7.0,
                                 max_fiducial_drift_z=22.0, spad=8,
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
    allele.fiducial_drift = {}
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

    def _cut(hybe, channel, pad=None):
        raw_y, raw_x = spot_mapper.reference_to_raw(
            shared_xy, hybe, fov_matrices, modality=modality, cell=cell,
            resolver=resolver)
        cube, (ymin, xmin) = spot_mapper.crop_for_localization(
            storage_path, fov, hybe, channel, (raw_y, raw_x),
            pad=(spad if pad is None else int(pad)), use_stack=True)
        return cube, ymin, xmin

    def _raw_centre(hybe):
        """The allele's shared coordinate in `hybe`'s raw frame -- the
        point every crop of this allele is cut around."""
        return spot_mapper.reference_to_raw(
            shared_xy, hybe, fov_matrices, modality=modality, cell=cell,
            resolver=resolver)

    def _template_crop(hybe, f, origin, cube):
        """The reference's crop RE-CUT around its fitted fiducial, for the
        correlation template: (cube, (y, x, z) in it, (ymin, xmin)).

        The display crop is cut around the allele's anchor, and the
        reference fiducial's fit may land at its edge and still be the
        reference (it is accepted whenever it fitted at all). A template
        centred there would hold the crop's corner. MEASURED, smoke run:
        one of four alleles had its reference at x = -1.5 in a 17 px
        crop, and every one of its 72 hybes was refused for want of a
        template. Register3D's box is around the reference PEAK for the
        same reason. Falls back to the display crop when the re-cut
        fails."""
        ymin, xmin = origin
        raw = (float(f.y) + ymin, float(f.x) + xmin)
        try:
            c2, (ymin2, xmin2) = spot_mapper.crop_for_localization(
                storage_path, fov, hybe, hybe_fiducial_channels[hybe], raw,
                pad=spad, use_stack=True)
        except (OSError, ValueError):
            c2 = None
        if c2 is None or c2.size == 0:
            return cube, (float(f.y), float(f.x), float(f.z)), (ymin, xmin)
        return c2, (raw[0] - ymin2, raw[1] - xmin2, float(f.z)), (ymin2, xmin2)

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
    # THE REFERENCE FIRST. The correlation fiducial measures every other
    # hybe AGAINST the reference's crop and position, and so does the
    # learned fiducial's correlation refinement, so the reference's own
    # fiducial has to exist before theirs are looked for. ref_ctx is
    # (template crop, its fiducial in that crop, the crop's origin) once
    # it does -- the crop re-cut around the fiducial, see _template_crop.
    ref_ctx = None
    wants_ref = p.fiducial_method == 'xcorr' or (
        p.fiducial_engine is not None and p.fiducial_refine == 'xcorr')
    order = ([reference_hybe] if reference_hybe in fid_cubes else []) + \
        [h for h in fid_cubes if h != reference_hybe]
    for hybe in order:
        cube = fid_cubes[hybe]
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
            # THE DEPTH WINDOW, for the XZ panel to draw: the expected
            # depth and how far either engine looks from it (the
            # Gaussian's seed window and the learned fiducial's window
            # are the same 17 planes). A candidate tagged 'z' is outside
            # these lines, and a person can now see that.
            debug[hybe]['fiducial_zexp'] = float(z0)
            debug[hybe]['fiducial_z_window'] = int(p.fiducial_window())
        used_xcorr = False
        if p.fiducial_method == 'xcorr' and hybe != reference_hybe:
            # THE FIDUCIAL BY IMAGE CORRELATION WITH THE REFERENCE --
            # ChrTracer3's Register3D. The reference's crop around its
            # own fiducial is the template; this hybe's crop is cut
            # WIDER by the drift gate's reach, so a shift as large as
            # the gate allows still has the whole template inside it;
            # the search is centred where the reference's fiducial
            # lands in this crop by the two crops' origins (and the
            # hybes' z offsets), and bounded by the drift gate, because
            # a shift beyond it would only be refused in phase 3. The
            # position comes back in this hybe's DISPLAY crop
            # coordinates so everything downstream -- the origins, the
            # shared-frame conversion, the tiles -- is the Gaussian's.
            used_xcorr = True
            f, how, ok, why = None, None, False, None
            if ref_ctx is None:
                why = 'reference fiducial not found (no template)'
            else:
                ms_xy = int(np.ceil(float(max_fiducial_drift))) + XCORR_SEARCH_MARGIN
                ms_z = int(np.ceil(float(max_fiducial_drift_z))) + XCORR_SEARCH_MARGIN
                ref_cube, (ry, rx, rz), (ymin_r, xmin_r) = ref_ctx
                ymin, xmin = fid_origin[hybe]
                half = p.template_half()
                # THE SEARCH CROP IS CUT AROUND THE EXPECTED FIDUCIAL, not
                # around the allele's anchor: the reference fiducial may
                # sit several px from the anchor (measured -9.5 px on one
                # smoke allele), and a crop centred on the anchor then
                # ran out of room on that side, so the template shrank
                # (9 -> 7 -> 5 px) or the range was clipped, silently.
                # The crop has the whole template plus the whole range on
                # every side of the expected position.
                cy_r, cx_r = _raw_centre(reference_hybe)
                cy_h, cx_h = _raw_centre(hybe)
                exp_y = float(cy_h) + (float(ry) + ymin_r - float(cy_r))
                exp_x = float(cx_h) + (float(rx) + xmin_r - float(cx_r))
                try:
                    wide, (ymin_w, xmin_w) = spot_mapper.crop_for_localization(
                        storage_path, fov, hybe, hybe_fiducial_channels[hybe],
                        (exp_y, exp_x), pad=int(max(half[0], half[1])) + ms_xy + 1,
                        use_stack=True)
                except (OSError, ValueError):
                    wide = None
                if wide is None or wide.size == 0:
                    why = 'fiducial crop unreadable'
                else:
                    # WHERE THE REFERENCE'S FIDUCIAL IS EXPECTED IN THIS
                    # CROP. The display crops of this allele are all cut
                    # around the same shared coordinate mapped into each
                    # hybe's raw frame, so the reference fiducial's offset
                    # from ITS crop centre carries over to this hybe's
                    # mapped centre -- that is exp_y/exp_x above. The
                    # crop ORIGINS alone would not do: they sit in two raw
                    # frames that differ by the inter-hybe alignment (5-7
                    # px on MP58), and using them put that translation
                    # into the shift, which then ran into the search
                    # bound on a third of the hybes -- measured on the
                    # smoke run: 'shift at the search bound (y)' at
                    # exactly 7.0 px, NCC 0.85.
                    # depth: native_h = native_ref + off_ref - off_h,
                    # since shared = native + off (consensus_native_z)
                    centre = (exp_y - ymin_w, exp_x - xmin_w,
                              rz + z_offsets.get(reference_hybe, 0.0)
                              - z_offsets.get(hybe, 0.0))
                    xf, xwhy = FX.register(
                        ref_cube, (ry, rx, rz), wide, centre,
                        max_shift=(ms_xy, ms_xy, ms_z), half=half)
                    if xf is not None:
                        xf = xf._replace(y=xf.y + ymin_w - ymin,
                                         x=xf.x + xmin_w - xmin)
                    ok, why = _xcorr_gate(xf, xwhy, cube, p)
                    f, how = xf, ('xcorr' if ok else None)
            if debug is not None:
                debug[hybe]['fiducial_engine'] = 'xc'
                debug[hybe]['fiducial_how'] = how
                debug[hybe]['fiducial_ncc'] = (float(f.ncc) if f is not None
                                               else float('nan'))
                debug[hybe]['fiducial_shift'] = (tuple(f.shift) if f is not None
                                                 else None)
        elif p.fiducial_engine is not None:
            # THE LEARNED FIDUCIAL, best of one. No Gaussian gate applies
            # -- a matched filter has no occupancy or CI -- the p_exist
            # threshold and the fiducial's own reach are the gate, and
            # every other candidate reaches the grid, labelled, as what
            # was NOT chosen. The Gaussian seed fallback below is v2's
            # and does not run for it.
            f, why, alts, how = _fiducial_learned(
                cube, z0, p,
                ref=(ref_ctx[:2] if ref_ctx is not None
                     and p.fiducial_refine == 'xcorr' else None))
            ok = f is not None
            if debug is not None:
                debug[hybe]['fiducial_engine'] = 'v3'
                debug[hybe]['fiducial_how'] = how
                # the correlation's own facts, when it did the refining
                # (an XcorrFit carries them; a LearnedFit does not)
                if getattr(f, 'ncc', None) is not None:
                    debug[hybe]['fiducial_ncc'] = float(f.ncc)
                    debug[hybe]['fiducial_shift'] = tuple(f.shift)
                debug[hybe]['fiducial_p_exist'] = (
                    float(getattr(f, 'p_exist', float('nan')))
                    if f is not None else float('nan'))
                debug[hybe]['fiducial_n_candidates'] = (
                    len(alts) + (1 if f is not None else 0))
                debug[hybe]['fiducial_rejected_centroids'] = (
                    [(a.x, a.y, a.z) for a, _w in alts] or None)
                debug[hybe]['fiducial_rejected_labels'] = (
                    [f'{a.p_exist:.2f}{alt_marker(w)}' for a, w in alts]
                    or None)
                debug[hybe]['fiducial_dropped_why'] = [w for _a, w in alts]
        else:
            f = fit_fiducial(cube, z0, p)
            ok, why = gate(f, cube, p.fiducial_gates, p.voxel_um)
        if (not ok and FIDUCIAL_SEED_FALLBACK and p.fiducial_engine is None
                and not used_xcorr):
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
            # a correlation position may lie outside the display crop
            # (the search ran on a wider one), where occupancy would be
            # measured at a clipped voxel; the tile does not show it
            debug[hybe]['fiducial_occupancy'] = (
                float('nan') if used_xcorr else occupancy(cube, f, p.voxel_um))
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
            # THE KEY STAYS, WITH None: looked for and not accepted, which
            # v1 has always written and readers (export's fiducial_found)
            # already distinguish from 'never tried'.
            allele.fiducial_trace_adj[hybe] = None
            allele.fiducial_trace_raw[hybe] = None
            allele.fiducial_drift[hybe] = None
            continue
        ymin, xmin = fid_origin[hybe]
        q = fiducial_quality(f)
        allele.fiducial_trace_adj[hybe] = _to_shared(hybe, f.y, f.x, f.z, ymin, xmin) \
            + (float(f.amplitude), q)
        # The SAME fit, before any matrix: crop-local plus the crop's own
        # origin. ymin/xmin are the actual origin, clamp included, so this
        # indexes that hybe's full frame directly and the image can be
        # re-reached without inverting anything.
        allele.fiducial_trace_raw[hybe] = (float(f.y + ymin), float(f.x + xmin),
                                           float(f.z), float(f.amplitude), q)
        fid_local[hybe] = (f.y, f.x, f.z)
        if hybe == reference_hybe and wants_ref:
            ref_ctx = _template_crop(hybe, f, fid_origin[hybe], cube)
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
            allele.fiducial_drift[hybe] = None
            continue
        if fid is None:
            allele.rejected_hybes[hybe] = 'fiducial not found'
            allele.fiducial_drift[hybe] = None
            continue
        dy, dx, dz = (baseline[0] - fid[0], baseline[1] - fid[1],
                      baseline[2] - fid[2])
        # Recorded BEFORE the gates: a drift past the gate is a fact
        # about this hybe worth keeping, and rejected_hybes says it was
        # refused. See AnAllele.fiducial_drift.
        allele.fiducial_drift[hybe] = (float(dy), float(dx), float(dz))
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
            debug[hybe]['readout_z_reach'] = int(p.z_reach())
            debug[hybe]['readout_seed'] = _seed(
                cube, z_r, p.voxel_um,
                _seed_z_half(READOUT_FIT_RADIUS_UM, p.voxel_um))
        if p.readout_engine is not None:
            # THE LEARNED PATH. One call, every readout candidate in this
            # crop, then a posterior p_exist cut -- the same shape as the
            # max_uncert gate below, applied to a different number.
            # `_to_shared` is the builder's own closure -- it carries this
            # run's resolver, fov matrices, modality and cell -- so it is
            # passed in rather than reached for. A module-level helper
            # that rebuilt that mapping would be a second implementation
            # of the frame conversion, which is the divergence this
            # codebase keeps having to hunt down.
            # THE SEED IS THE FIDUCIAL'S OWN FITTED POSITION in this crop
            # -- both channels are cut by _cut around the same mapped
            # centre with the same pad, so fid_local's (y, x) is valid
            # here -- and that, not the crop centre, is what a readout
            # candidate has to be near.
            # THE SEARCH CROP IS WIDER THAN THE DISPLAY CROP by the
            # engine's own lateral box: a candidate within reach of the
            # fiducial but near the display crop's edge would otherwise
            # get a padded feature box and no template room (the
            # boundary pinning the matcher had before). Same centre,
            # same channel; the hits are reported in the display crop's
            # coordinates so the grid keeps showing the fiducial's box.
            from .psfmatcher import BOX_R
            try:
                cube_s, ymin_s, xmin_s = _cut(hybe, channel, pad=spad + BOX_R)
            except (OSError, ValueError):
                cube_s, ymin_s, xmin_s = cube, ymin, xmin
            oy, ox = ymin - ymin_s, xmin - xmin_s
            seed_yx = ((float(fid_local[hybe][0]) + oy,
                        float(fid_local[hybe][1]) + ox)
                       if hybe in fid_local else None)
            done, why_multi = _readout_multi(allele, hybe, cube_s, z_r, p,
                                             dy, dx, dz, ymin_s, xmin_s,
                                             _to_shared, debug,
                                             seed_yx=seed_yx,
                                             display_offset=(oy, ox),
                                             display_shape=cube.shape[:2])
            if not done:
                allele.rejected_hybes[hybe] = why_multi
            continue
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
                                     float(r.amplitude), float('nan'))]
        # Raw carries NO correction of any kind -- neither the alignment
        # nor the fiducial drift. adj - raw for a READOUT is therefore
        # alignment PLUS fiducial correction, one term more than the same
        # difference on a fiducial. See AnAllele's docstring.
        allele.polymer_raw[hybe] = [(float(r.y + ymin), float(r.x + xmin),
                                     float(r.z), float(r.amplitude), float('nan'))]
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
        'engine': ('v3' if params is not None and params.is_learned else 'v2'),
        'engine_label': (params.engine_label or 'v2') if params else 'v2',
        'readout_model_dir': (params.readout_model_dir if params else None),
        'fiducial_model_dir': (params.fiducial_model_dir if params else None),
        'min_p_exist': (params.min_p_exist if params else None),
        'min_p_exist_fiducial': (params.min_p_exist_fiducial if params else None),
        'lateral_reach_px': (params.lateral_reach_px if params else None),
        'fiducial_z_window': (params.fiducial_z_window if params else None),
        'genomic_resolution_kb': (params.genomic_resolution_kb
                                  if params else None),
        'traced_at': _time.strftime('%Y-%m-%dT%H:%M:%S'),
        'voxel_um': list(params.voxel_um) if params else None,
        'psf': (params.psf_label or None) if params else None,
        'psf_family': params.psf_family if params else None,
        # reconcile (analysis/reconcile.py) re-derives polymer_adj from
        # polymer_raw under CURRENT matrices; the fiducial correction it
        # re-applies is ref-relative, so the trace must say which hybe
        # was its reference -- and which modality frame its hybes name.
        'reference_hybe': reference_hybe,
        'modality': modality,
        # WHICH CHANNELS this trace was actually fitted on, per hybe.
        # Nothing recorded it before, and "readout" does not name one
        # wavelength: hybes in an experiment carry different channel sets,
        # so the auto rule resolves to 635 for a two-channel hybe and 475
        # for a three-channel one. Without this a stored trace cannot be
        # told from one fitted on a different channel. Sorted for a stable
        # diff between runs; old alleles simply lack the keys, the same way
        # the provenance column itself was introduced.
        'fiducial_channels': {h: fid_ch[h] for h in sorted(fid_ch or {})},
        'readout_channels': {h: read_ch[h] for h in sorted(read_ch or {})},
    }
    build_chromatin_trace_allele(
        allele, hybes, reference_hybe, fid_ch, read_ch, storage_path, fov,
        modality, cell, fov_matrices, params=params,
        max_fiducial_drift=max_drift, max_fiducial_drift_z=max_drift_z,
        spad=spad, collect_debug=False, resolver=resolver)
    return (int(meta['id']), allele.fiducial_trace_adj, allele.polymer_adj,
            allele.fiducial_trace_raw, allele.polymer_raw,
            allele.rejected_hybes, getattr(allele, 'reference_warning', None),
            dict(getattr(allele, 'provenance', {}) or {}),
            dict(getattr(allele, 'fiducial_drift', {}) or {}))


def stored_allele_debug(payload):
    """The View-Stored path: the SAME per-hybe grid payload the fit path
    produces, built from an allele's ALREADY-PERSISTED trace -- crops
    read around the stored raw positions, circles AT those positions,
    and NO fitting at all (per request: re-running ~200 Gaussian fits
    just to LOOK at a finished allele was pure waste; the reads are the
    image, the fits were the cost). occupancy/uncertainty stay absent --
    they are properties of a fit, and this view shows the record, not a
    new measurement (the grid's own 'occ n/a' + stored-reason titling
    already renders that honestly).

    payload: (allele_dict, hybes, hybe_fiducial_channels,
    hybe_readout_channels, storage_path, fov, spad). Runs in a child
    process (stack reads; the h5py one-lock-per-process rule).
    """
    from codelab_pipeline.alignment import spot_mapper
    (allele_dict, hybes, hybe_fiducial_channels, hybe_readout_channels,
     storage_path, fov, spad) = payload
    fid = allele_dict.get('fiducial_trace_raw') or {}
    pol = allele_dict.get('polymer_raw') or {}
    debug = {}
    for hybe in hybes:
        fpos = fid.get(hybe)
        cands = pol.get(hybe) or []
        centre = fpos or (cands[0] if cands else None)
        if centre is None:
            continue                    # nothing stored -> nothing to show
        cy, cx = float(centre[0]), float(centre[1])
        d = {}
        fch = hybe_fiducial_channels.get(hybe)
        if fch is not None:
            try:
                cube, (ymin, xmin) = spot_mapper.crop_for_localization(
                    storage_path, fov, hybe, fch, (cy, cx), pad=spad,
                    use_stack=True)
                d['fiducial_cubic'] = cube
                # crop-local (x, y, z), the grids' centroid contract
                d['fiducial_centroid'] = (
                    (float(fpos[1] - xmin), float(fpos[0] - ymin),
                     float(fpos[2])) if fpos else None)
            except OSError:
                pass
        rch = hybe_readout_channels.get(hybe)
        if rch is not None:
            try:
                cube, (ymin, xmin) = spot_mapper.crop_for_localization(
                    storage_path, fov, hybe, rch, (cy, cx), pad=spad,
                    use_stack=True)
                d['readout_cubic'] = cube
                d['readout_centroids'] = (
                    [(float(x - xmin), float(y - ymin), float(z))
                     for y, x, z, *_ in cands] or None)
            except OSError:
                pass
        if d:
            debug[hybe] = d
    return debug


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
        'engine': ('v3' if params is not None and params.is_learned else 'v2'),
        'engine_label': (params.engine_label or 'v2') if params else 'v2',
        'readout_model_dir': (params.readout_model_dir if params else None),
        'fiducial_model_dir': (params.fiducial_model_dir if params else None),
        'min_p_exist': (params.min_p_exist if params else None),
        'min_p_exist_fiducial': (params.min_p_exist_fiducial if params else None),
        'lateral_reach_px': (params.lateral_reach_px if params else None),
        'fiducial_z_window': (params.fiducial_z_window if params else None),
        'genomic_resolution_kb': (params.genomic_resolution_kb
                                  if params else None),
        'traced_at': _time.strftime('%Y-%m-%dT%H:%M:%S'),
        'voxel_um': list(params.voxel_um) if params else None,
        'psf': (params.psf_label or None) if params else None,
        'psf_family': params.psf_family if params else None,
        # reconcile (analysis/reconcile.py) re-derives polymer_adj from
        # polymer_raw under CURRENT matrices; the fiducial correction it
        # re-applies is ref-relative, so the trace must say which hybe
        # was its reference -- and which modality frame its hybes name.
        'reference_hybe': reference_hybe,
        'modality': modality,
        # WHICH CHANNELS this trace was actually fitted on, per hybe.
        # Nothing recorded it before, and "readout" does not name one
        # wavelength: hybes in an experiment carry different channel sets,
        # so the auto rule resolves to 635 for a two-channel hybe and 475
        # for a three-channel one. Without this a stored trace cannot be
        # told from one fitted on a different channel. Sorted for a stable
        # diff between runs; old alleles simply lack the keys, the same way
        # the provenance column itself was introduced.
        'fiducial_channels': {h: fid_ch[h] for h in sorted(fid_ch or {})},
        'readout_channels': {h: read_ch[h] for h in sorted(read_ch or {})},
    }
    _a, debug = build_chromatin_trace_allele(
        allele, hybes, reference_hybe, fid_ch, read_ch, storage_path, fov,
        modality, cell, fov_matrices, params=params,
        max_fiducial_drift=max_drift, max_fiducial_drift_z=max_drift_z,
        spad=spad, collect_debug=True, resolver=resolver)
    return ((int(meta['id']), allele.fiducial_trace_adj, allele.polymer_adj,
             allele.fiducial_trace_raw, allele.polymer_raw,
             allele.rejected_hybes, getattr(allele, 'reference_warning', None),
             dict(allele.provenance or {}),
             dict(getattr(allele, 'fiducial_drift', {}) or {})), debug)


def apply_allele_result(allele, result):
    """Merge a child's result into the parent's own AnAllele, in place."""
    (_aid, fiducial_trace_adj, polymer_adj, fiducial_trace_raw, polymer_raw,
     rejected, warning, provenance, fiducial_drift) = result
    allele.fiducial_trace_adj = fiducial_trace_adj
    allele.polymer_adj = polymer_adj
    allele.fiducial_trace_raw = fiducial_trace_raw
    allele.polymer_raw = polymer_raw
    allele.fiducial_drift = dict(fiducial_drift or {})
    allele.rejected_hybes = rejected
    if warning:
        allele.reference_warning = warning
    if provenance:
        allele.provenance = provenance
    return allele


# THE ONE VOCABULARY. Two selectors used to offer the same idea in
# incompatible words -- the tracing panel said 'v1'/'v2' and the
# 3D-localization popup said 'gaussian'/'v2' -- so a value copied from
# one to the other did not route, and 'gaussian' collided by name with a
# real make_engine key while meaning something else entirely. It was
# never a third engine: the popup's own combo reads
# addItem('v1 gaussian', 'gaussian'), so the LABEL already said v1 and
# only the stored value disagreed.
ROUTE_V1 = 'v1'
ROUTE_V2 = 'v2-anchor-fit'
ROUTE_V3 = 'v3-psfmatcher'
ROUTES = (ROUTE_V1, ROUTE_V2, ROUTE_V3)


def route(engine):
    """Which fit path a stored or typed engine value names.

    PREFIX MATCHED, not equality, so a label may gain explanation
    ('v2-anchor-fit (calibrated PSF)') without silently switching every
    run back to v1 -- a failure that would show up only as slightly
    worse numbers. Equality is what the popup's consumer used, and it is
    why a decorated v2 label would have routed correctly in tracing and
    fallen back to v1 there.

    EVERY OLD SPELLING STILL LANDS: 'v2' is the value in five checked-in
    configs and prefixes ROUTE_V2; 'gaussian' was the popup's not-v2
    sentinel and is the v1 route. Data on disk outlives a rename, the
    same rule ASpot.coordinate -> adj_coordinate follows.
    """
    s = str(engine or '').strip().lower()
    if s.startswith('v3'):
        return ROUTE_V3
    if s.startswith('v2'):
        return ROUTE_V2
    return ROUTE_V1


def is_v2(engine):
    """True for the v2 anchor-fit route, whatever spelling reached it."""
    return route(engine) == ROUTE_V2


def uses_v2_tracer(engine):
    """v2 AND v3: the learned engine replaces only the READOUT fit.

    v3 keeps everything else of the v2 path -- the fiducial fitted by
    v2's own Gaussian (one major spot, no multispot search), the drift
    and z-drift gates against the reference fiducial, and the readout
    crop cut around the fiducial-mapped seed so a readout candidate is
    judged inside the SAME box the fiducial anchors. Every dispatch that
    asked is_v2() sent v3 down the v1 tracer, which has no readout
    engine at all: v1's gates fired, no multispot ever formed, and the
    grid showed v1's numbers. The panel's own docstring predicted it.
    """
    return route(engine) in (ROUTE_V2, ROUTE_V3)


def is_v3(engine):
    """True for the learned route."""
    return route(engine) == ROUTE_V3


def trace_allele(engine, allele, hybes, reference_hybe, hybe_fiducial_channels,
                 hybe_readout_channels, storage_path, fov, modality, cell,
                 fov_matrices, v2_params=None, max_fiducial_drift=7.0,
                 max_fiducial_drift_z=22.0, spad=8, z_window=15,
                 fiducial_params=None, readout_params=None, collect_debug=False,
                 resolver=None, z_boundary_trim=10, executor=None):
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

    v1 keeps every other argument it had. z_window and z_boundary_trim
    are HONOURED by v2 and v3 too, carried on V2Params (from_panel reads
    them): the readout's axial reach around the fiducial, and the planes
    shaved off each stack end. They used to be dropped here as v1-only,
    which left two of the three engines deaf to the two z controls on
    the panel. The per-channel *_params stay v1's: they carry v1's gate
    constants, which do not transfer to a local background.
    """
    # STAMP HOW THIS TRACE WAS MADE, whichever engine runs. Free-form on
    # purpose: each engine records its own inputs, so a future engine adds
    # its hyperparameters here without anything else changing. The base is
    # only what is true of every engine.
    if allele is not None and hasattr(allele, 'provenance'):
        import time as _time
        stamp = {'engine': ('v3' if is_v3(engine) else
                            'v2' if is_v2(engine) else 'v1'),
                 'engine_label': str(engine or ''),
                 'traced_at': _time.strftime('%Y-%m-%dT%H:%M:%S')}
        if uses_v2_tracer(engine) and v2_params is not None:
            stamp.update({
                'voxel_um': list(v2_params.voxel_um),
                'psf': v2_params.psf_label or None,
                'psf_family': v2_params.psf_family,
                'fiducial_gates': {k: v for k, v in v2_params.fiducial_gates.items()
                                   if not isinstance(v, tuple)},
                'readout_gates': {k: v for k, v in v2_params.readout_gates.items()
                                  if not isinstance(v, tuple)},
            })
            if v2_params.is_learned:
                stamp.update({'readout_model_dir': v2_params.readout_model_dir,
                              'fiducial_model_dir': v2_params.fiducial_model_dir,
                              'min_p_exist': v2_params.min_p_exist,
                              'min_p_exist_fiducial': v2_params.min_p_exist_fiducial,
                              'lateral_reach_px': v2_params.lateral_reach_px,
                              'fiducial_z_window': v2_params.fiducial_z_window})
        allele.provenance = stamp
    if not uses_v2_tracer(engine):
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
