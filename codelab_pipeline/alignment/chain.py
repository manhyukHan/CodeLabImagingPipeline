import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial, reduce
import numpy as np

from .frames import FrameMatrices  # re-exported: callers reach it as alignment.FrameMatrices
from .convention import as_cv2, to_yx  # THE y-major<->cv2 adapter (see convention.py)
import numpy.linalg as la
import h5py
from skimage import filters as skimage_filters
from skimage.feature import peak_local_max

from ..io import paths
from ..io import preprocess
from .. import process_guard
from .. import tuning
from ..io import analysis_store
import cv2

# Hard, non-configurable engine-level bounds on any fitted alignment matrix
# -- FOV-level (same-modality, cross-modal, via align_readout_to_reference)
# and cell-level (the compute_cell_alignment residual) alike -- per explicit
# request, deliberately NOT exposed as a tunable UI setting: anything a fit
# claims beyond this is treated as an optimizer artifact, not a real result,
# regardless of what any particular experiment's own drift tolerance might
# be. Confirmed via a deliberate bad-value test (dx=1071, dy=42, angle=100
# deg on a real DNA-RNA cross-modal pair) that fits well outside real
# biological drift can otherwise be silently accepted and propagated.
MAX_ALIGNMENT_TRANSLATION_PX = 30.0
MAX_ALIGNMENT_ROTATION_DEG = 10.0

# Rotation is quantized to this step before it is used (see
# align_readout_to_reference). Fine-scale rotation carries no information for
# this pipeline: ORB resolves rotation to well under a degree, real inter-hybe
# rotational drift here is ~0.1 deg (measured across 77 real pairs: |angle| max
# 0.1007 deg), and a sub-quantum angle changes a 1024x1024 frame's corners by
# less than the alignment's own translational precision. Quantizing also makes
# the common "no real rotation" case exactly representable as 0.0, so it can be
# tested for rather than approximated with a threshold.
ANGLE_QUANTUM_DEG = 0.5

# How far the cell-level Z refinement may move a hybe, in PLANES.
#
# Was pad/2 -- i.e. 5 planes at the default pad of 10 -- which derived a DEPTH
# bound from an XY search radius. The two are not related: pad bounds how far
# the YX crop could contain matching content, while Z drift between
# hybridization rounds is a property of the focus/stage, and a 120-plane stack
# has far more room in Z than a +-10 px crop has in XY.
#
# Measured on a real run (110 hybe entries, one FOV): the Z leg APPLIED a shift
# on 75.5% of entries with |z| median 3.0 and max exactly 5.0 -- pinned at the
# old cap, which is the signature of a distribution truncated by its limit
# rather than by physics. Another 16.4% were rejected on magnitude alone, with
# |z| median 7.5 and max 11.0: real, plausible drift that the bound discarded,
# leaving those hybes with no Z correction at all.
#
# 15 covers the observed maximum with headroom while staying far below the
# noise-locking artifacts this gate exists to catch (a -35 plane "fit" on a
# low-signal crop is documented in compute_cell_alignment). It remains a
# backstop, not the primary defence -- the reconstruction-residual quality gate
# beside it is what actually rejects bad fits, and caught 8.2% here on its own.
MAX_CELL_Z_SHIFT_PLANES = 15.0


def _center_displacement(H, shape):
    """
    How far H actually moves the image centre, in px: (dy, dx).

    This, not H's raw translation column, is what "how big is this
    correction" means. A rotation about the image centre, written as a
    matrix about the ORIGIN, carries a large translation column that is
    pure bookkeeping -- an 8 deg rotation of a 1024x1024 frame has
    t=(70.8, -64.1) while moving the centre not at all. Reading that
    column as displacement makes rotation magnitude scale with frame size
    and with distance from the origin, which is meaningless.
    """
    cx, cy = shape[1] / 2.0, shape[0] / 2.0
    nx, ny = (np.asarray(H, dtype=float)[:2] @ np.array([cx, cy, 1.0]))
    return ny - cy, nx - cx


def _within_hard_alignment_bounds(H, shape=None, max_translation=MAX_ALIGNMENT_TRANSLATION_PX,
                                  max_rotation=MAX_ALIGNMENT_ROTATION_DEG):
    """
    True iff H's translation (dy, dx independently) and rotation stay
    within the hard engine-level bounds above. Shared by every fitted-
    matrix gate in this module so they can never independently drift apart
    on what counts as "plausible."

    `shape` (height, width) makes the translation test measure displacement
    AT THE IMAGE CENTRE via _center_displacement. Without it the test falls
    back to H's raw translation column, which is correct only for a pure
    translation: for any real rotation that column is dominated by the
    rotation's own origin offset, so the gate rejected correct fits purely
    for containing rotation. Confirmed on synthetic ground truth -- ORB
    recovered 8 deg exactly, and this gate threw the result away for
    "70.8 px of translation" that did not move the centre at all. Pass
    shape wherever it is available.
    """
    if shape is not None:
        dy, dx = _center_displacement(H, shape)
    else:
        dx, dy = H[0, 2], H[1, 2]
    if abs(dx) > max_translation or abs(dy) > max_translation:
        return False
    return abs(_h_rotation_angle_degrees(H)) <= max_rotation


def align_cell(yx, H, shape):
    """
    Transforms mask coordinates yx=(y,x) into H's target frame. Checks
    the MATRIX, not any caller's notion of "is this a special hybe" --
    every caller (get_area_in_readout, compute_cell_alignment, etc.)
    should always just call this with whatever H applies, never branch
    on hybe/modality identity first: H itself already carries the
    information needed to decide whether any real transform is
    happening (H == identity, checked here, once, for ANY caller) or
    not, and hybe-name-based special-casing scattered across callers is
    exactly the pattern that kept reintroducing real bugs this session
    (the cross-modal bridge hybe collision, the double-counted FOV
    correction) by comparing names instead of the actual math.

    When H is (numerically) identity, skips the morphological closing
    below entirely -- confirmed on real data that closing is NOT a true
    no-op even at H=identity (up to +15px silently ADDED to a cell's own
    mask across 12/66 real cells, from the closing kernel alone, no real
    transform involved). Closing exists to bridge small rounding gaps a
    REAL (non-identity) transform can leave between adjacent points;
    applying it when there's provably no transform at all just corrupts
    the mask for no reason.
    """
    y,x = yx
    height,width = shape
    if np.allclose(H, np.eye(3)):
        cx,cy = x.astype(int), y.astype(int)
        bad = (cx < 0) | (cy < 0) | (cx >= width) | (cy >= height)
        return cy[~bad], cx[~bad]
    # H is y-major (convention.py): row 0 -> y, row 1 -> x.
    cy,cx = (H[:2]@np.array([y,x,np.ones_like(x)])).astype(int)
    bad = (cx < 0) | ( cy < 0) | (cx >= width) | (cy >= height)
    cx,cy = cx[~bad],cy[~bad]
    if cx.size == 0:
        return cy, cx
    # Rasterize + close in a LOCAL bbox, not the full frame: closing with
    # a 3x3 kernel is a 1px-radius local operation, so a bbox with a 2px
    # margin of REAL zeros (1 for the dilation reach + 1 for the erosion
    # neighborhood -- cv2 morphology treats the array border itself with
    # special +/-inf values, NOT zeros, so a 1px margin left border-
    # touching pixels uneroded: confirmed 5-extra-point mismatch) yields
    # the byte-identical point set at a fraction of the cost. Where the
    # bbox is clipped by the actual frame edge the local border COINCIDES
    # with the frame border, reproducing the full-frame border behavior
    # exactly. The full-frame version allocated and morphed height*width
    # pixels PER CELL, which made the FOV-view "cell masks" overlay the
    # dominant cost of every hybe/channel switch (101 cells: ~0.9s of a
    # 1.8s switch, measured).
    pad = 2
    x0, y0 = max(0, int(cx.min()) - pad), max(0, int(cy.min()) - pad)
    x1, y1 = min(width, int(cx.max()) + pad + 1), min(height, int(cy.max()) + pad + 1)
    local = np.zeros((y1 - y0, x1 - x0))
    local[cy - y0, cx - x0] = 1
    closed = cv2.morphologyEx(local, cv2.MORPH_CLOSE, kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    ly, lx = np.where(closed > 0)
    return ly + y0, lx + x0

def compose_chain(matrices):
    """
    Compose an ordered list of 3x3 affine-like matrices into one final
    matrix. matrices[0] is applied to a raw point first (innermost),
    matrices[-1] last (outermost) -- e.g. compose_chain([H_within, H_across])
    for the within/across-experiment case, or
    compose_chain([H_within, H_across, H_fine]) once a per-cell/per-spot
    fine-alignment step (applied after segmentation/localization) exists.
    """
    return reduce(np.matmul, reversed(matrices))

def hybe_to_cellref_matrix(fov_matrices, H_cellref_to_within, hybe):
    """
    hybe's own native frame -> cell.reference_hybe's own native frame.
    fov_matrices: {hybe: 3x3}, hybe's own frame -> whatever within-
    experiment shared frame align_same_modality used -- see that
    function. H_cellref_to_within: cell.reference_hybe's OWN transform
    into that SAME shared frame -- an EXPLICIT parameter, deliberately
    never looked up from fov_matrices[cell.reference_hybe] internally,
    because a cross-modal caller's fov_matrices (the OTHER modality's own
    hybes) never contains cell.reference_hybe as a key at all (it's a
    same-modality-only hybe name) -- callers must resolve
    H_cellref_to_within from whichever fov_matrices dict cell.reference_
    hybe ACTUALLY lives in (almost always the cell's own, same-modality
    one) and pass it through explicitly, never assume it's a key in the
    SAME dict `hybe` itself is being looked up in.

    This is the ONE place this conversion should ever be computed --
    every caller that needs to relate cell.area's coordinates (always
    native to cell.reference_hybe) to some OTHER hybe's own native frame
    must go through this, rather than using fov_matrices[hybe] directly.
    fov_matrices' own shared frame is an internal implementation detail of
    align_same_modality (whichever hybe the FOV-alignment step used as
    ITS reference), which has no reason to coincide with
    cell.reference_hybe (the segmentation hybe) -- and in general does
    not. Using fov_matrices[hybe] directly wrongly assumes those two
    frames are the same thing; real data confirmed this produced both a
    genuine double-counted correction in compute_cell_alignment's own
    fit AND a matching-but-independently-wrong "FOV/cross-modal" preview
    column, whenever the two reference hybes actually differed.

    Composing this consistently everywhere is also what makes the whole
    alignment process invariant to which hybe happens to be chosen as
    reference_hybe for FOV-alignment or for cell-alignment's own fitting
    anchor: those choices only ever affect where phase correlation
    anchors ITS OWN comparison, never what coordinate frame a result is
    allowed to end up expressed in, and never make the process require
    prior alignment to have "succeeded" -- fov_matrices.get(..., identity)
    means a hybe/layer with no alignment computed yet contributes nothing
    (identity), not a missing-data error.
    """
    H_hybe_to_within = fov_matrices.get((hybe, fov_matrices.modality), np.eye(3))
    return compose_chain([H_hybe_to_within, la.inv(H_cellref_to_within)])

def h_angle_degrees(H_yx):
    """Rotation angle (degrees) of a Y-MAJOR matrix -- the display/report
    read for every stored/composed H in this pipeline. Numerically equal
    to the engine-internal x-major read of the same physical rotation
    (proven in tests/test_convention.py), so reported angles are
    continuous across the convention flip. _h_rotation_angle_degrees
    below stays the x-major read and is ENGINE-INTERNAL only."""
    H_yx = np.asarray(H_yx, dtype=float)
    return float(np.degrees(np.arctan2(H_yx[0, 1], H_yx[0, 0])))


def _reconstruction_residual(moving_norm, reference_norm, H, min_overlap_frac=0.5, signal_threshold=10):
    """
    Mean squared pixel error after warping moving_norm by H, over the
    region where the REFERENCE has real signal. Lower is a better fit;
    used to pick between candidate alignment methods below.

    signal_threshold (default 10, on the 0-255 normalized scale): a pixel
    only counts toward the residual if reference_norm > signal_threshold
    there -- not "both images are nonzero" (the previous criterion,
    `(warped>0)&(reference>0)`). That older criterion had two compounding
    problems on a sparse/mostly-empty image (e.g. a diffuse fiducial hybe
    with lots of true background): (1) background-to-background pixels,
    which are the majority of a real microscopy frame and trivially
    near-zero error under ANY transform, padded the mean and diluted the
    handful of pixels that actually carry discriminating signal; (2) worse,
    it silently EXCLUDED the most informative pixels for detecting a wrong
    transform -- wherever the reference has real content but a bad
    transform sampled background into that spot (warped~0), those pixels
    failed `warped>0` and got dropped from the average instead of
    penalizing the bad fit. Together this let a confident-but-wrong
    correspondence (e.g. ORB finding a large spurious rotation on a sparse
    image) score deceptively well. Gating on the reference alone makes the
    metric mean what it should: how much of the reference's actual content
    did this transform actually reconstruct.

    The overlap-fraction guard is now a separate, purely geometric check
    (does the transform place min_overlap_frac of the frame within
    bounds at all -- via warping an all-ones mask, same technique
    msd_cost_function already uses), decoupled from signal content: a
    degenerate transform that rotates+translates the moving image into a
    small, coincidentally-similar corner can still score a low RAW MSD by
    averaging over far fewer, cherry-picked pixels -- observed on a real
    cross-modal pair, where a bad transform (8.6% overlap) scored better
    than the correct one (62.2% overlap). Two hybridization rounds of the
    same FOV should overlap by a large majority of the frame; a tiny
    overlap is itself evidence of a bad fit, independent of what the
    signal-gated residual says.
    """
    h, w = reference_norm.shape[:2]
    warped = cv2.warpAffine(moving_norm.astype(np.float32), H[:2], (w, h),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    coverage = cv2.warpAffine(np.ones_like(moving_norm, dtype=np.uint8), H[:2], (w, h),
                              flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if (coverage > 0).sum() < min_overlap_frac * h * w:
        return np.inf
    valid = reference_norm > signal_threshold
    if valid.sum() == 0:
        return np.inf  # no real reference signal at all -- nothing meaningful to score
    return float(((warped[valid] - reference_norm[valid].astype(np.float32)) ** 2).mean())

def _clip_background(crop, method='yen'):
    """
    Soft background suppression for a cell-level crop, BEFORE it's handed
    to phase correlation and the reconstruction-residual quality gate --
    subtracts this crop's own Otsu/Yen threshold and clips below zero,
    rather than binarizing to 0/1. A hard binary mask would discard the
    intensity gradient sub-pixel phase correlation actually relies on for
    its own accuracy; this keeps that gradient intact above threshold
    while suppressing the diffuse background that can otherwise pull
    cv2.phaseCorrelate toward a confident-but-wrong correspondence (the
    same "background dilutes/hides the discriminating signal" failure
    mode _reconstruction_residual's own docstring identifies for MSD
    scoring -- addressed here one step earlier, in the FIT itself, not
    just at scoring time).

    method: 'yen' (default) or 'otsu' -- same two skimage.filters
    threshold functions already used for cell segmentation (segment.py),
    kept consistent rather than introducing a third convention.

    Falls back to the crop unchanged (float32) if thresholding fails
    (skimage raises on a degenerate, e.g. all-constant, crop) -- a
    pathological crop is exactly the case with no meaningful background
    to separate out, not a reason to error the whole alignment.
    """
    crop_f = crop.astype(np.float32)
    try:
        threshold = float(skimage_filters.threshold_yen(crop) if method == 'yen'
                          else skimage_filters.threshold_otsu(crop))
    except ValueError:
        return crop_f
    return np.clip(crop_f - threshold, 0, None)

def _multi_peak_translation(target_crop, reference_crop, half=12, min_distance=15,
                            num_peaks=4, threshold_rel=0.3):
    """
    Alternative to a single whole-crop phase-correlation fit: finds
    several DISTINCT bright peaks in reference_crop (mild Gaussian blur
    first, so a single hot pixel can't register as its own peak), fits
    translation independently at each via phase correlation on a small
    window around it, then takes the MEDIAN shift across peaks as the
    consensus H2.

    Motivation, confirmed on real data (cell 3, Hyb_105 vs Hyb_101): a
    whole-crop fit pools every pixel together, so it can get pulled
    toward whichever region is brightest -- but "brightest" isn't the
    same as "representative." On that real cell, 3 of 4 distinct peaks
    independently agreed on dy=0, while the single brightest, most
    isolated peak (a lone fiducial bead, visually distinct from the
    diffuse chromatin/cell body signal) wanted dy=-2 -- and the whole-
    crop fit landed near that ONE outlier peak, not the 3-peak majority.
    Taking the median across several independently-fit peaks is robust
    to exactly this: one outlier peak can't drag the consensus the way
    it drags a single pooled fit, without at least half the peaks
    agreeing with it.

    Falls back to a single whole-crop phase-correlation fit when fewer
    than 2 peaks are found -- a median of 0 or 1 values has no
    consensus/robustness benefit over just fitting the whole crop
    directly, and a low-signal crop legitimately might not have that
    many distinct bright features to key off of.

    Each peak's OWN fit is gated exactly like the whole-crop fit is
    gated one level up in compute_cell_alignment -- per explicit
    feedback, a peak here is just as capable of a bad phase-correlation
    lock as the whole crop is, and un-gated peaks can drag the median
    toward a wrong answer as surely as a single un-gated whole-crop fit
    can. TWO gates per peak: (1) magnitude > half of the peak's own
    small window -- a shift that size couldn't be real content within
    that window, only noise/padding (confirmed on real data: cell 49,
    Hyb_131 vs Hyb_101 -- one peak's fit hit 15px in a half=12 window,
    clearly a lock onto nothing); (2) _reconstruction_residual doesn't
    improve on that peak's own small crop -- catches the more common
    case where the shift is small enough to pass (1) but still makes the
    local match WORSE, not better (confirmed on the same real cell: the
    other 3 of 4 peaks all passed the magnitude gate at ~11px < half=12,
    but EVERY one of the 4 failed the quality gate -- residual got worse
    at every single peak, correctly signaling this hybe pair has no real
    correspondence in this crop at all, not just one bad peak among
    several good ones). Rejected peaks are dropped entirely, not zeroed
    (a zero would itself bias the median toward "no shift", a claim this
    function has no basis to make about a peak whose own fit failed).
    Falls back to identity (dx=dy=0) if EVERY peak is rejected -- same
    "no no-alignment" principle as the whole-crop reject bound: reject
    means "no correction found", never a value derived from data the
    gates themselves flagged as untrustworthy.
    """
    height, width = reference_crop.shape
    ref_smooth = cv2.GaussianBlur(reference_crop.astype(np.float32), (5, 5), 0)
    peaks = peak_local_max(ref_smooth, min_distance=min_distance, num_peaks=num_peaks,
                           threshold_rel=threshold_rel)
    if len(peaks) < 2:
        return np.vstack([preprocess.find_translation_via_phase_correlation(target_crop, reference_crop),
                          np.array([0, 0, 1])])
    shifts = []
    for py, px in peaks:
        ymin, ymax = max(0, py - half), min(height, py + half + 1)
        xmin, xmax = max(0, px - half), min(width, px + half + 1)
        small_ref = reference_crop[ymin:ymax, xmin:xmax]
        small_tgt = target_crop[ymin:ymax, xmin:xmax]
        H_peak = preprocess.find_translation_via_phase_correlation(small_tgt, small_ref)
        dx, dy = H_peak[0, 2], H_peak[1, 2]
        H_peak3 = np.vstack([H_peak, np.array([0, 0, 1])])
        magnitude_rejected = np.hypot(dx, dy) > half
        residual_before = _reconstruction_residual(small_tgt, small_ref, np.eye(3))
        residual_after = _reconstruction_residual(small_tgt, small_ref, H_peak3)
        quality_rejected = not (residual_after < residual_before)
        if not (magnitude_rejected or quality_rejected):
            shifts.append((dx, dy))
    if not shifts:
        return np.eye(3)
    shifts = np.array(shifts)
    dx, dy = float(np.median(shifts[:, 0])), float(np.median(shifts[:, 1]))
    return np.array([[1., 0., dx], [0., 1., dy], [0., 0., 1.]])

def _find_z_shift(target_profile, ref_profile):
    """
    1D cross-correlation shift estimate between two depth profiles
    (already collapsed down to a single Z-axis signal each -- see
    compute_cell_alignment's own Z-alignment leg). NOT cv2.phaseCorrelate:
    that's built for 2D images and its internal FFT/Hann-window machinery
    errors on a genuinely 1D (width-1) array (confirmed on real data --
    cv2.error out of cv2.phaseCorrelate on a (depth,1)-shaped input). A
    plain 1D cross-correlation is both the simpler and the more correct
    tool once X has already been collapsed out of the problem entirely.

    Returns the shift such that target_profile, moved by this amount,
    best matches ref_profile (same sign convention as
    find_translation_via_phase_correlation's own translation output).
    """
    t = target_profile.astype(np.float64).ravel()
    r = ref_profile.astype(np.float64).ravel()
    t = t - t.mean()
    r = r - r.mean()
    corr = np.correlate(r, t, mode='full')
    best_k = int(np.argmax(corr))
    return float(best_k - (len(t) - 1))

def _h_rotation_angle_degrees(H):
    """
    Degrees such that cv2.getRotationMatrix2D(center, angle, 1.0)'s
    rotation block matches H[:2, :2] -- i.e. the angle to hand back into
    compute_msd_homography_matrix's fixed_angle to reproduce the same
    rotation ORB found. Confirmed empirically, not just derived: a
    synthetic known-rotation test (apply +5/+8/-8 deg to a real MIP,
    recover via compute_features_affinelike_matrix, extract with this
    formula, feed the result back into a Powell translation-only fit
    under that fixed angle) gave reconstruction residual ~17 (near-
    perfect) for this exact sign convention; the other three sign/axis
    combinations all gave residual 1500-1950 (visibly wrong direction).
    """
    return float(np.degrees(np.arctan2(H[0, 1], H[0, 0])))


def entry_dz(entry):
    """
    The Z shift out of an ACell.matrices entry, as a plain float.

    'dz' is the current shape. A legacy 'zx' 3x3 is read at [0, 2] -- the
    only element it ever carried: Z alignment is a 1D correlation, that
    matrix was never composed, inverted or multiplied anywhere, and every
    reader read exactly that one element. The conversion is therefore
    exact and unambiguous, so legacy data is accepted rather than rejected.
    """
    if not entry:
        return 0.0
    if 'dz' in entry:
        return float(entry['dz'])
    zx = entry.get('zx')
    return 0.0 if zx is None else float(np.asarray(zx)[0, 2])


def align_readout_to_reference(moving_mip, reference_mip, lb=0.3, ub=0.9999, border_trim=0, max_shift=None):
    """
    Compute the affine-like matrix aligning moving_mip onto reference_mip.
    Takes plain MIP arrays -- usable for both within-experiment (fiducial
    MIP vs. fiducial MIP) and cross-experiment (readout MIP vs. readout
    MIP) alignment; the caller is responsible for always passing
    same-channel-type inputs on both sides, never mixed.

    ORB DETECTS, POWELL REFINES -- they are not two competing estimators
    picked between, and Powell no longer searches rotation at all.

    1. Run ORB+RANSAC (preprocess.compute_features_affinelike_matrix).
       ORB is the only one of the two methods that can find real rotation
       at all -- MSD/Powell's optimizer converges to angle~0 regardless of
       true rotation, even at 8 degrees (synthetic ground-truth tested) --
       so ORB's angle is the only signal available for "is there real
       rotational drift," and Powell has nothing to add to it.
    2. Admit ORB's result as a SEED, all three parameters or none
       (_within_hard_alignment_bounds, which fails on dx OR dy OR angle).
       One-out, because ORB's three outputs come from a single
       correspondence set: a wrong match corrupts all of them together, so
       trusting the translation of a fit whose angle is implausible would
       be trusting the same bad correspondence twice. A rejected seed
       falls back to the cold (0, 0, 0) start.
       Measured justification, 77 real hybe pairs in one FOV: ORB's centre
       displacement sat a median 0.165 px from Powell's own final answer
       (max 1.089 px; 76/77 within 1 px) at 1/127th of Powell's cost. The
       previous code computed exactly this and discarded it, then let
       Powell rediscover it from scratch.
       Note ORB is a seed and NOT the answer: on those same 77 pairs
       Powell's refined fit still won on reconstruction residual 76/77
       times (mean 84.4 vs 87.0). ORB gets close; Powell finishes.
    3. Quantize the seed angle to ANGLE_QUANTUM_DEG (0.5) -- see that
       constant -- and seed Powell's angle parameter with it. The angle
       stays FREE, not fixed. Fixing it was tried and measured and loses on
       both axes (24 real pairs, vs the cold-start baseline: seed+free
       1.28x faster and mean residual -0.217; cold+fixed 0.93x SLOWER and
       +0.569; seed+fixed 1.22x and -0.015). That third parameter is not
       acting as a rotation estimate -- it cannot, Powell converges to ~0
       regardless of true rotation -- it is acting as a SEARCH DIRECTION
       that lets the line searches escape shallow valleys in the
       translation landscape. Removing it costs evaluations rather than
       saving them. See the inline comment for the full table.
    4. If the quantized angle is 0 (the overwhelmingly common case -- real
       drift here is ~0.1 deg), one seeded Powell fit IS the answer. If it
       is non-zero, keep the three-candidate bake-off: ORB's own transform,
       the seeded free-angle Powell fit, and a pinned zero-rotation Powell
       fit, scored by reconstruction residual. The zero-rotation baseline
       stays because bounds cannot replace it -- an ORB angle can be
       spurious yet perfectly in bounds (observed on real data: a
       readout-channel cross-modal pair reporting ~162 degrees is caught by
       bounds, but a wrong 3 degrees would not be), and without a
       no-rotation candidate there is nothing for a wrong-but-plausible
       angle to lose to.

    border_trim (default 0, no behavior change): crop this many pixels off
    every edge of BOTH images before running either method -- vignetting,
    scan-seam artifacts, or partial-FOV edge content can pull ORB's feature
    matcher toward a confident-but-wrong correspondence near the boundary;
    trimming removes that region from consideration without resampling the
    interior. The returned translation is in the TRIMMED images' own
    coordinate frame, which is identical to the original frame's (a crop
    only shifts the origin, translations measured within it are unaffected)
    -- no offset correction needed by the caller.

    max_shift (default None, no behavior change): the winning candidate's
    translation is clamped to +/-max_shift px (rotation/scale left alone)
    if it exceeds that bound -- a hard cap, not just a tiebreaker, per
    explicit request for a tunable bound on how much real physical drift
    is plausible between two hybridization rounds/imaging sessions.

    Independently of max_shift (which is opt-in, off by default): every
    candidate is also checked against the hard, non-configurable engine-
    level bounds (MAX_ALIGNMENT_TRANSLATION_PX/MAX_ALIGNMENT_ROTATION_DEG,
    see their own module-level docstring) BEFORE candidate selection --
    whichever candidate has the lowest reconstruction residual AMONG the
    ones actually within bounds wins, same "best of the genuinely
    plausible options" principle used everywhere else in this pipeline
    (e.g. mixture-fit sibling selection). If NO candidate qualifies (a
    single-candidate free-angle fit that itself is out of bounds, or --
    in the ORB/rotation branch -- all three of ORB/fixed-angle-confirm/
    zero-angle are out of bounds), returns identity: an out-of-bound fit
    is treated as "no real correction found," never applied partially.
    """
    if border_trim > 0:
        moving_mip = moving_mip[border_trim:-border_trim, border_trim:-border_trim]
        reference_mip = reference_mip[border_trim:-border_trim, border_trim:-border_trim]

    moving_norm = preprocess.normalize_to_uint8(moving_mip, lb, ub)
    reference_norm = preprocess.normalize_to_uint8(reference_mip, lb, ub)

    # Native optimizer bounds for the Powell/MSD candidates below -- unlike
    # ORB+RANSAC (a closed-form/RANSAC estimator with no bounds mechanism
    # to hook into, see compute_features_affinelike_matrix), Powell's own
    # scipy implementation supports bounds= directly, constraining the
    # SEARCH itself rather than only checking its result after the fact.
    powell_bounds = [(-MAX_ALIGNMENT_TRANSLATION_PX, MAX_ALIGNMENT_TRANSLATION_PX),
                     (-MAX_ALIGNMENT_TRANSLATION_PX, MAX_ALIGNMENT_TRANSLATION_PX),
                     (-MAX_ALIGNMENT_ROTATION_DEG, MAX_ALIGNMENT_ROTATION_DEG)]

    H_orb = preprocess.compute_features_affinelike_matrix(moving_norm, reference_norm)
    angle_orb = _h_rotation_angle_degrees(H_orb)

    # ORB SEEDS POWELL instead of being discarded (see this function's own
    # docstring for the measurement that motivated it: on 77 real hybe pairs
    # ORB's centre displacement sat a median 0.165 px from Powell's final
    # answer, max 1.089 px, at 1/127th of Powell's cost -- and the old code
    # threw it away on every one of them, leaving Powell to start cold at
    # (0, 0) and rediscover it).
    #
    # ONE-OUT ADMISSION. _within_hard_alignment_bounds is reused rather than
    # open-coded precisely because it already implements the required rule:
    # it fails if dx OR dy OR the angle is out of bounds, so a single bad
    # parameter voids the whole seed. A rejected seed falls back to the
    # original cold start -- never a partially-trusted one, since ORB's three
    # outputs come from ONE correspondence set and a wrong correspondence
    # corrupts all three together.
    seed_admitted = _within_hard_alignment_bounds(H_orb, moving_norm.shape)
    if seed_admitted:
        seed_dy, seed_dx = _center_displacement(H_orb, moving_norm.shape)
        # dx/dy in msd_cost_function's parameter space ARE the centre
        # displacement: it rotates about the centre (which maps the centre to
        # itself) and only then adds dx/dy. So ORB's centre displacement is
        # directly usable as the seed, and being <= MAX_ALIGNMENT_TRANSLATION_PX
        # it is inside powell_bounds by construction.
        angle_seed = round(angle_orb / ANGLE_QUANTUM_DEG) * ANGLE_QUANTUM_DEG
    else:
        seed_dy = seed_dx = 0.0
        angle_seed = 0.0
    seed = [seed_dx, seed_dy, angle_seed]

    # The angle stays a FREE Powell parameter, seeded from ORB rather than
    # fixed to it. Holding it fixed was tried and measured, and it loses on
    # both axes -- 24 real hybe pairs, versus the old cold-start baseline:
    #
    #     seed + free angle    1.28x faster    mean residual -0.217  (22/24 better)
    #     cold + fixed angle   0.93x SLOWER    mean residual +0.569  ( 3/24 better)
    #     seed + fixed angle   1.22x faster    mean residual -0.015  ( 3/24 better)
    #
    # The reason the a-priori argument for dropping it fails: that parameter is
    # not functioning as a rotation ESTIMATE (it cannot -- Powell converges to
    # ~0 regardless of true rotation, and the half-degree gate in
    # find_best_alignment discards whatever it lands on). It functions as a
    # SEARCH DIRECTION. Powell's line searches use it to leave shallow valleys
    # in the translation landscape, and the dx/dy reached that way reconstruct
    # measurably better. Removing it removes an escape route, so the optimizer
    # needs MORE evaluations to meet the same tolerance, not fewer.
    #
    # Seeding it with ORB's quantized angle is what makes this the best of both:
    # when rotation is real, Powell starts already holding it instead of having
    # to discover a rotation it provably cannot discover.
    if angle_seed == 0.0:
        candidates = [preprocess.compute_msd_homography_matrix(moving_norm, reference_norm,
                                                                fixed_scale=1.0, fixed_angle=False,
                                                                initial_guess=seed, bounds=powell_bounds)]
    else:
        # Real, in-bounds rotation: keep the three-way bake-off. The
        # zero-rotation baseline is the load-bearing one -- an ORB angle can be
        # spurious yet perfectly in bounds (bounds catch 162 deg, not a wrong
        # 3 deg), and without a candidate that assumes no rotation there is
        # nothing for a wrong-but-plausible angle to lose to. It is pinned with
        # fixed_angle=True rather than merely seeded at zero, so it stays a
        # genuine no-rotation hypothesis that the free-angle candidate cannot
        # drift into agreeing with.
        H_confirm = preprocess.compute_msd_homography_matrix(moving_norm, reference_norm, fixed_scale=1.0,
                                                              fixed_angle=False, initial_guess=seed,
                                                              bounds=powell_bounds)
        H_zero = preprocess.compute_msd_homography_matrix(moving_norm, reference_norm, fixed_scale=1.0,
                                                           fixed_angle=True,
                                                           initial_guess=[seed_dx, seed_dy, 0.0],
                                                           bounds=powell_bounds)
        candidates = [H_orb, H_confirm, H_zero]

    # ORB is still checked post-hoc here (no native bounds available for
    # it); the Powell candidates above are now bounded at the SEARCH level
    # too, so this is a genuine belt-and-suspenders double-check for them,
    # not their only safeguard.
    in_bounds = [H for H in candidates if _within_hard_alignment_bounds(H, moving_norm.shape)]
    if not in_bounds:
        return np.eye(3)
    residuals = [_reconstruction_residual(moving_norm, reference_norm, H) for H in in_bounds]
    H_final = in_bounds[int(np.argmin(residuals))]

    if max_shift is not None:
        # Clip the CENTRE displacement, not the raw translation column, and
        # adjust the column by the difference so the rotation block survives
        # untouched. Clipping the column directly would corrupt any rotation
        # (whose column is mostly origin offset, not displacement) into a
        # different, arbitrary transform rather than a smaller one.
        dy, dx = _center_displacement(H_final, moving_norm.shape)
        if abs(dx) > max_shift or abs(dy) > max_shift:
            H_final = H_final.copy()
            H_final[0, 2] += float(np.clip(dx, -max_shift, max_shift)) - dx
            H_final[1, 2] += float(np.clip(dy, -max_shift, max_shift)) - dy

    # THE engine boundary (convention.py): everything above -- ORB, Powell,
    # residual scoring, bounds -- is cv2-native x-major; everything outside
    # this function speaks y-major. One conversion, here, nowhere else.
    return to_yx(H_final)

def write_same_modality_matrices(storage_path, fov, matrices, reference_hybe):
    """
    Persists an already-computed {hybe: matrix} dict (from
    align_same_modality(..., write=False), or a manual-mode staged
    result the user just accepted) into vlinks.h5's /FOV##/matrix/{hybe},
    plus reference_sequence/steps provenance attrs -- split out from
    align_same_modality so the write step can be deferred (manual
    review mode) or run standalone.

    Delegates to analysis_store rather than writing into each hybe's own raw
    {hybe}_stack.h5 (the previous behavior) -- per explicit principle,
    vlinks.h5 must be the pipeline's authoritative store for this, not N
    scattered raw per-hybe files that require heavy I/O (opening every
    stack file) just to answer "has this FOV been aligned." Outside
    ingestion and 3D localization, the raw stack files should not need to
    be touched at all.
    """
    analysis_store.write_same_modality_matrices(storage_path, fov, matrices, reference_hybe)


def read_same_modality_matrices(storage_path, fov, hybe_records):
    """
    Reads back whatever's already in vlinks.h5's /FOV##/matrix/{hybe} (see
    write_same_modality_matrices) without requiring align_same_modality to
    be re-run. A hybe already ingested (real MIP present in vlinks.h5) but
    with no matrix entry yet still legitimately gets an identity default.
    A hybe never ingested at all is silently SKIPPED (no entry in the
    returned dict), never given a fake identity default. This matters:
    hybe_records passed in here can be the full parsed ExperimentLayout
    (declaring far more hybes than were ever actually ingested), and a
    fake identity entry for a non-ingested hybe used to leak into
    self.fov_matrices, making it look like a real, processable hybe to
    downstream code (e.g. cell-based alignment), which then crashed trying
    to open a stack file that genuinely doesn't exist. This is the
    read-back half of "activation": self.fov_matrices in the GUI should
    reflect whatever alignment has already been computed and written,
    without the user needing to re-run alignment just to see it again.

    Delegates to analysis_store.read_same_modality_matrices instead of
    opening each hybe's own raw
    {hybe}_stack.h5 -- this is now a single vlinks.h5 open, not N raw file
    opens, so callers can refresh this freely.
    """
    hybe_list = [record['folder'] for record in hybe_records]
    return analysis_store.read_same_modality_matrices(storage_path, fov, hybe_list)


# -- parallel same-modality alignment (see align_same_modality) -----------
#
# One hybe's fit is completely independent of every other's: it reads its own
# MIP and compares it against the one shared reference. The loop was serial on
# a single QThread, so on a 64-core machine it used exactly one of them -- and
# the fit is pure CPU (measured on real 1024x1024 MIPs: ~3.5 s per hybe, 93%
# of it inside Powell, which converges in 100-260 objective evaluations of
# ~20 ms each). A 78-hybe FOV therefore took ~4.5 minutes on one core while
# 63 sat idle. Nothing about the fit is I/O-bound: a 1024x1024 MIP is ~2 MB.
#
# spawn context, matching IngestionWorker's own pool. The children must never
# import Qt, so the worker function lives here in a Qt-free module and takes
# only picklable arguments.

_ALIGN_WORKER_REFERENCE = {}


def _init_align_worker(storage_path, fov, reference_hybe, ref_channel):
    """
    Read the shared reference MIP ONCE per worker process, not once per task.

    Also pins OpenCV to one thread inside the child. Without that, each of N
    workers spins up its own cv2 pool sized to the whole machine and they
    fight over the same cores -- the classic oversubscription that can make a
    pool slower than the serial loop it replaced.
    """
    cv2.setNumThreads(1)
    _ALIGN_WORKER_REFERENCE['mip'] = analysis_store.read_hybe_mip(
        storage_path, fov, reference_hybe, ref_channel)


def _align_one_hybe(task):
    """One (hybe -> reference) fit, run in a worker process."""
    storage_path, fov, hybe, channel, lb, ub, border_trim, max_shift = task
    reference_mip = _ALIGN_WORKER_REFERENCE.get('mip')
    if reference_mip is None:
        return hybe, None, 'reference MIP unavailable in worker'
    moving_mip = analysis_store.read_hybe_mip(storage_path, fov, hybe, channel)
    if moving_mip is None:
        return hybe, None, 'not ingested'
    H = align_readout_to_reference(moving_mip, reference_mip, lb, ub,
                                   border_trim=border_trim, max_shift=max_shift)
    return hybe, H, None


# -- FOV-major, all-modalities alignment ----------------------------------
#
# One pool per FOV carrying every (modality, hybe) that FOV needs, rather
# than one pool per (modality, FOV) run back to back. Same rule ingestion
# follows: finish a FOV completely across every modality, then move to
# the next one, so a FOV becomes fully usable as early as possible
# instead of every modality advancing in lockstep.
#
# The single-value reference cache above cannot serve this -- two
# modalities in one pool means two different (storage_path, reference
# hybe) pairs -- so the multi-modality worker keys its cache and fills it
# lazily. Each reference MIP is still read at most once per child.

_ALIGN_REFERENCE_CACHE = {}


def _init_align_worker_multi():
    """Pin OpenCV to one thread per child (see _init_align_worker); the
    reference MIPs are cached lazily per key instead of up front."""
    cv2.setNumThreads(1)


def _align_one_hybe_multi(task):
    """One (modality, hybe -> that modality's reference) fit."""
    (modality, storage_path, fov, hybe, channel, reference_hybe,
     ref_channel, lb, ub, border_trim, max_shift) = task
    key = (storage_path, fov, reference_hybe, ref_channel)
    if key not in _ALIGN_REFERENCE_CACHE:
        _ALIGN_REFERENCE_CACHE[key] = analysis_store.read_hybe_mip(
            storage_path, fov, reference_hybe, ref_channel)
    reference_mip = _ALIGN_REFERENCE_CACHE[key]
    if reference_mip is None:
        return modality, hybe, None, f'reference {reference_hybe} not ingested'
    moving_mip = analysis_store.read_hybe_mip(storage_path, fov, hybe, channel)
    if moving_mip is None:
        return modality, hybe, None, 'not ingested'
    H = align_readout_to_reference(moving_mip, reference_mip, lb, ub,
                                   border_trim=border_trim, max_shift=max_shift)
    return modality, hybe, H, None


def align_fov_all_modalities(fov, specs, lb=0.3, ub=0.9999, write=True,
                             border_trim=0, max_shift=None, progress=None,
                             workers=None):
    """
    Align EVERY configured modality's hybes for ONE FOV, in one pool.

    specs: [{'modality', 'storage_path', 'hybe_records', 'reference_hybe'},
    ...] -- one entry per modality. Each modality aligns to ITS OWN
    reference hybe: alignment is per-modality maths (a hybe is comparable
    only to another hybe of the same modality, fiducial to fiducial), and
    only the SCHEDULING is shared.

    Returns {modality: {hybe: 3x3}}. The reference hybe of each modality
    maps to identity by construction.

    progress: callable(done, total, fov, label) after every hybe, where
    total counts hybes across ALL modalities -- this is one FOV's work,
    so it reports as one unit.

    write=True persists each modality's matrices under its own storage
    path, exactly as align_same_modality does.
    """
    per_modality = {}
    tasks = []
    total = 0
    for spec in specs:
        modality = spec['modality']
        storage_path = spec['storage_path']
        reference_hybe = spec['reference_hybe']
        records = list(spec['hybe_records'])
        by_folder = {r['folder']: r for r in records}
        ref_record = by_folder.get(reference_hybe)
        if ref_record is None:
            raise ValueError(f"reference hybe {reference_hybe} is not in {modality}'s "
                             f"hybe list for FOV{fov:03d}")
        per_modality[modality] = {reference_hybe: np.eye(3)}
        total += len(records)
        for r in records:
            if r['folder'] == reference_hybe:
                continue
            tasks.append((modality, storage_path, fov, r['folder'], r['fiducial_channel'],
                          reference_hybe, ref_record['fiducial_channel'],
                          lb, ub, border_trim, max_shift))

    done = [0]

    def _report(label):
        done[0] += 1
        if progress is not None:
            progress(done[0], max(total, 1), fov, label)

    for spec in specs:                       # the identity references
        _report(f"{spec['reference_hybe']} ({spec['modality']})")

    n_workers = max_alignment_workers() if workers is None else max(1, int(workers))
    executor = None
    if n_workers > 1 and len(tasks) > 1:
        try:
            executor = ProcessPoolExecutor(
                max_workers=min(n_workers, len(tasks)),
                mp_context=multiprocessing.get_context('spawn'),
                initializer=partial(process_guard.child_initializer,
                                    _init_align_worker_multi))
        except Exception:
            executor = None                  # degrade to serial, never fail here

    if executor is not None:
        with executor:
            futures = [executor.submit(_align_one_hybe_multi, t) for t in tasks]
            for future in as_completed(futures):
                modality, hybe, H, err = future.result()
                if err is not None:
                    raise ValueError(f'FOV{fov:03d} {hybe} ({modality}): {err}')
                per_modality[modality][hybe] = H
                _report(f'{hybe} ({modality})')
    else:
        for task in tasks:
            modality, hybe, H, err = _align_one_hybe_multi(task)
            if err is not None:
                raise ValueError(f'FOV{fov:03d} {hybe} ({modality}): {err}')
            per_modality[modality][hybe] = H
            _report(f'{hybe} ({modality})')

    if write:
        for spec in specs:
            modality = spec['modality']
            write_same_modality_matrices(spec['storage_path'], fov,
                                         per_modality[modality], spec['reference_hybe'])
    return per_modality


def max_alignment_workers(hard_ceiling=32):
    """
    How many hybe fits to run at once.

    A core count, not a memory budget: each worker holds two 1024x1024 uint8
    crops plus a few float32 temporaries, tens of MB at most. That is the
    opposite of preprocess.max_ingestion_workers, which is bounded by whole
    DAX files sitting in RAM. Two cores are left for the GUI thread and the
    coordinator so the app stays responsive while a FOV aligns.
    """
    return max(1, min((os.cpu_count() or 4) - 2, hard_ceiling))


# -- parallel cell-level residual alignment (see CellAlignmentWorker) ------
#
# One cell's residual fit is independent of every other cell's: it reads its
# own crops out of shared read-only files and writes only its own matrices.
# The loop was serial on a single QThread. Unlike the per-hybe FOV pool above
# (pure CPU, near-linear), this workload is ~74% disk reads after the ref_zx
# hoist and windowed-MIP fixes, so the pool's ceiling is the DRIVE, not the
# core count: measured on the real E: store, concurrent windowed stack reads
# scale 2.87x at 8 workers / 3.74x at 16 / 4.98x at 24 -- sublinear the whole
# way. Expect roughly 4x, not core-count.
#
# The genuinely risky part is not memory (a worker holds a few ~82x75 crops
# and ZX projections -- tens of MB) but the MUTATION CONTRACT:
# compute_cell_alignment commits by mutating the cell in place, and in
# automatic mode those are the real ACell objects. A child process mutates a
# PICKLED COPY, so the parent must replace the real cell's three mutated
# attributes with the child's end state. The child started from the same
# pickled state the serial code would have mutated, so its end state IS the
# serial end state -- wholesale replacement of exactly those three dicts
# (matrices, matrix_anchors, matrix_provenance -- see compute_cell_alignment's
# docstring for why it is these three and nothing else) is faithful, not
# approximate.

class CellOffFrameError(ValueError):
    """The cell's mask, projected into a pass's reference-hybe frame,
    lands entirely outside the image: the cell is simply not visible in
    that modality's view (the two cameras' frames overlap imperfectly --
    measured up to ~28 px of bridge shift on real data, enough to carry
    a thin edge cell wholly off-frame). Callers SKIP that pass: the cell
    keeps FOV/bridge-level transforms for that modality, per the
    absence-of-alignment-is-identity principle. It must never abort a
    batch -- one real store held exactly 2 such slivers among 1091
    cells, and aborting cost the other 1089 their run."""


def _init_cell_align_worker():
    """One cv2 thread per child -- same oversubscription guard as the pools
    above; the fit maths here is small, the reads are the real cost."""
    cv2.setNumThreads(1)


def _align_one_cell(task):
    """
    All passes for ONE cell, in a worker process.

    `passes` arrive with hybe_records already filtered against fov_matrices
    membership and cellref_matrix already resolved per cell -- that logic
    stays in the parent (CellAlignmentWorker), which is the code that has
    always owned it; this function is deliberately just the loop body.

    Returns the cell's three mutated dicts rather than the whole cell:
    the parent already holds the real object, and shipping ~44 KB of mask
    arrays back per cell would be pure overhead.
    """
    cell, fov, passes, channel_type, pad, z_max_shift = task
    skipped = []
    for p in passes:
        try:
            compute_cell_alignment(
                cell, p['storage_path'], fov, p['hybe_records'], p['fov_matrices'],
                reference_hybe=p['reference_hybe'], channel_type=channel_type,
                pad=pad, modality=p['modality'],
                cell_reference_hybe_matrix=p['cellref_matrix'],
                z_max_shift=z_max_shift)
        except CellOffFrameError as e:
            # skip THIS pass, keep the others (see the class docstring)
            skipped.append(f"{p['modality']}: {e}")
    return (fov, cell.id, cell.matrices, cell.matrix_anchors,
            cell.matrix_provenance, skipped)


def max_cell_alignment_workers(hard_ceiling=16):
    """
    How many cells to fit at once. Capped well below max_alignment_workers'
    32 because this pool is disk-bound (see the block comment above): the
    measured read-throughput curve has visibly flattened by 16 workers, and
    workers beyond the knee add spawn cost and seek pressure for little
    return. Two cores stay reserved for the GUI thread and the coordinator.

    "Flattened" was as far as that measurement went, and flat is not the
    same as best: ingestion, which reads the same storage the same way,
    measured 117.6 MB/s at 12 workers against 66 MB/s at 36 -- so past the
    knee this store gets actively SLOWER, and nobody has yet found the cell
    pool's equivalent of that number. Until someone does, tuning.py can
    override this from a file between runs so the curve can be walked on
    the real store rather than argued about; an unset knob leaves the
    measured behaviour exactly as it was.
    """
    override, _source = tuning.cell_alignment_workers()
    if override is not None:
        # Deliberately NOT floored by hard_ceiling: the whole point of the
        # override is to test values the current default cannot express,
        # and the interesting direction is downward.
        return max(1, int(override))
    return measured_cell_alignment_workers(hard_ceiling)


def measured_cell_alignment_workers(hard_ceiling=16):
    """The default above with the tuning override deliberately NOT applied.

    For the per-hybe pool, which is a different pool answering a different
    question. The override exists to throttle a BATCH run that is starving
    the GUI of disk; the per-hybe pool is what the GUI itself uses to draw
    a single-cell preview, so throttling it slows down the very thing the
    throttle is meant to protect.

    That distinction is not hypothetical. An earlier change made the batch
    path hybe-major by editing the projection helper the preview shares,
    and the preview -- which calls it 2-3 times per hybe, against a
    measured break-even of ~55 cells per stack -- became slow enough that
    the app had to be reverted mid-session. One knob reaching two pools
    with opposite interests is how that happens.
    """
    return max(1, min((os.cpu_count() or 4) - 2, hard_ceiling))


def align_same_modality(storage_path, fov, hybe_records, reference_hybe, lb=0.3, ub=0.9999, write=True,
                            border_trim=0, max_shift=None, progress=None, workers=None):
    """
    Align every hybe's fiducial-channel MIP to reference_hybe's fiducial-
    channel MIP -- always fiducial-to-fiducial, never the readout channel,
    since fiducial images the same physical object (beads/chromatin) across
    every readout in one experiment and is what's directly comparable.
    reference_hybe can be any hybe in hybe_records, not just the first, so
    the mechanism is exercised generally rather than defaulting trivially.

    write=True (default, preserves prior behavior/all existing call sites):
    writes each result into that hybe's own H5 /matrix/{hybe} immediately,
    via write_same_modality_matrices. write=False computes and returns
    the matrices without touching H5 -- for manual-review GUI mode, where
    the write only happens if/when the user accepts the staged result
    (call write_same_modality_matrices explicitly at that point).
    Returns {hybe: matrix} either way.

    border_trim/max_shift: passed straight through to
    align_readout_to_reference for every non-reference hybe -- see that
    function's docstring. Both default to no-op (0 / None), so existing
    callers see no behavior change unless they opt in.

    progress: optional callable(done, total, fov, hybe), invoked after
    EVERY hybe. Exists because this loop is slow enough to look hung: on
    real 1024x1024 MIPs each align_readout_to_reference is an ORB fit
    plus one to three Powell/MSD optimizations, ~3.5 s per hybe (Powell
    converges in ~155 objective evaluations at ~20 ms each), so a
    78-hybe FOV runs ~4.5 minutes. AlignmentWorker used to report only
    once the whole FOV returned, which read as a hang. Default None
    leaves every existing call site byte-identical.
    """
    record_by_folder = {r['folder']: r for r in hybe_records}
    ref_record = record_by_folder[reference_hybe]

    # FOV-level (same-modality) alignment is not one of the 3D exceptions
    # (3D spot localization, 3D cell-based alignment, 3D spot-based
    # alignment) -- reads vlinks.h5's real MIP copy, never the raw stack
    # file.
    reference_mip = analysis_store.read_hybe_mip(storage_path, fov, reference_hybe, ref_record['fiducial_channel'])
    if reference_mip is None:
        raise ValueError(f'FOV{fov:03d} {reference_hybe} not ingested -- ingest it first.')

    matrices = {reference_hybe: np.eye(3)}
    todo = [r for r in hybe_records if r['folder'] != reference_hybe]
    n_workers = max_alignment_workers() if workers is None else max(1, int(workers))
    total = len(hybe_records)
    done = [0]

    def _report(hybe):
        # Per-HYBE progress, not per-FOV -- this loop is the app's longest
        # silent stretch, and a caller must not have to wait for the whole
        # FOV to learn anything at all.
        done[0] += 1
        if progress is not None:
            progress(done[0], total, fov, hybe)

    _report(reference_hybe)

    executor = None
    if n_workers > 1 and len(todo) > 1:
        try:
            executor = ProcessPoolExecutor(
                max_workers=min(n_workers, len(todo)),
                mp_context=multiprocessing.get_context('spawn'),
                initializer=partial(process_guard.child_initializer, _init_align_worker),
                initargs=(storage_path, fov, reference_hybe, ref_record['fiducial_channel']))
        except Exception:
            executor = None      # degrade to the serial loop below, never fail here

    if executor is not None:
        with executor:
            tasks = {executor.submit(_align_one_hybe,
                                     (storage_path, fov, r['folder'], r['fiducial_channel'],
                                      lb, ub, border_trim, max_shift)): r['folder']
                     for r in todo}
            for future in as_completed(tasks):
                hybe, H, err = future.result()
                if err == 'not ingested':
                    raise ValueError(f'FOV{fov:03d} {hybe} not ingested -- ingest it first.')
                if err is not None:
                    raise ValueError(f'FOV{fov:03d} {hybe}: {err}')
                matrices[hybe] = H
                _report(hybe)
    else:
        for record in todo:
            hybe = record['folder']
            moving_mip = analysis_store.read_hybe_mip(storage_path, fov, hybe, record['fiducial_channel'])
            if moving_mip is None:
                raise ValueError(f'FOV{fov:03d} {hybe} not ingested -- ingest it first.')
            matrices[hybe] = align_readout_to_reference(moving_mip, reference_mip, lb, ub,
                                                        border_trim=border_trim, max_shift=max_shift)
            _report(hybe)

    # Restore hybe_records order: as_completed yields whatever finishes first,
    # and no caller should be able to tell from this dict whether it was
    # computed serially or in a pool.
    matrices = {r['folder']: matrices[r['folder']] for r in hybe_records if r['folder'] in matrices}

    if write:
        write_same_modality_matrices(storage_path, fov, matrices, reference_hybe)

    return matrices

def link_cross_modal(rna_storage_path, dna_storage_path, fov,
                      rna_fov_matrices, dna_fov_matrices,
                      rna_reference_hybe='Hyb_500', dna_reference_hybe='Hyb_400',
                      channel_type='readout', lb=0.3, ub=0.9999,
                      border_trim=0, max_shift=None, with_residuals=False):
    """
    Align the two experiments using the specified channel of each reference
    hybe -- 'readout' (default) uses each hybe's non-fiducial channel, e.g.
    DAPI via Hyb_500 (RNA) / Hyb_400 (DNA), since DNA_Expt/RNA_Expt are
    different imaging sessions with no generally-shared fiducial signal.
    'fiducial' is also valid when the chosen reference_hybe is itself a
    readout physically shared between both experiments (e.g. the barcode
    round Hyb_130, imaged in both DNA_Expt and RNA_Expt) -- in that specific
    case the fiducial channel *is* comparable across experiments too, same
    as within one experiment. The reference readout for each modality
    (rna_reference_hybe/dna_reference_hybe) is always an explicit input,
    never inferred from datatype -- barcode readouts exist in both
    experiments for celltype classification, not as a hardcoded alignment
    default.

    rna_fov_matrices/dna_fov_matrices: {hybe: H_within} from each modality's
    own align_same_modality. Each reference-hybe MIP is warped by its
    own H_within[reference_hybe] *before* the cross-modal comparison --
    this chain is strictly hierarchical (FOV -> cross-experiment -> cell ->
    spot), so cross-modal alignment must build on top of within-experiment
    alignment, not recompute from the raw/uncorrected MIP. Comparing raw
    MIPs directly silently drops that hybe's own FOV-level correction
    whenever the cross-modal reference hybe isn't also the FOV-level
    reference hybe -- the common case. This warp is the same
    derive-a-correction-matrix-only exception already used throughout this
    module (e.g. compute_cell_alignment's crops) -- it is never used to
    produce stored data, only to compute H_across.

    Returns H_across: maps DNA's within-experiment reference frame
    (dna_reference_hybe) onto RNA's within-experiment reference frame
    (rna_reference_hybe). RNA's reference readout is the shared global
    frame by convention (there's no third, independent frame without extra
    information), so every RNA readout's final matrix is
    compose_chain([H_within_RNA[readout], np.eye(3)]) -- H_across is
    identity for RNA, appended for symmetry with the design rather than
    treating RNA as a hardcoded special case -- while every DNA readout's
    final matrix is compose_chain([H_within_DNA[readout], H_across]).

    border_trim/max_shift: passed straight through to
    align_readout_to_reference for the actual DNA->RNA correlation step --
    see that function's docstring. Both default to no-op (0 / None).

    with_residuals (default False, no behavior change): return
    (H_across, {'residual_before', 'residual_after'}) instead of bare
    H_across. Both numbers are the signal-gated reconstruction MSD
    (_reconstruction_residual) measured on the SAME trimmed+normalized
    pair the fit itself was scored on -- 'before' under identity (no
    correction), 'after' under the returned H_across -- so they are a
    measured quality of THIS result, comparable across FOVs of one run.
    Lower is better; after >= before means the fit did not actually
    improve the match and the result deserves a human look. Persisted
    beside the matrix at accept time (analysis_store.write_cross_modal_
    quality), per explicit request: the status viewer must be able to
    report the bridge's fit quality, not just its dx/dy/angle.
    """
    # Cross-modality alignment is not one of the 3D exceptions -- reads
    # vlinks.h5's real MIP copies, never the raw stack file. channel_mip
    # resolves 'fiducial'/'readout'/a CONCRETE channel value alike (the
    # old either/or branch silently read the readout MIP for a concrete
    # choice -- exactly the second-readout channel it was picked FOR).
    rna_mip = analysis_store.channel_mip(rna_storage_path, fov,
                                         rna_reference_hybe, channel_type)
    dna_mip = analysis_store.channel_mip(dna_storage_path, fov,
                                         dna_reference_hybe, channel_type)
    if rna_mip is None or dna_mip is None:
        # real modality names, not the historical RNA/DNA slot names --
        # the positional slots carry shared/moving semantics since the
        # hub+bridges refactor, and the FrameMatrices already know whose
        # they are (zero extra I/O)
        shared_name = getattr(rna_fov_matrices, 'modality', None) or 'shared'
        moving_name = getattr(dna_fov_matrices, 'modality', None) or 'moving'
        raise ValueError(f'FOV{fov:03d}: {rna_reference_hybe} ({shared_name}) and/or '
                         f'{dna_reference_hybe} ({moving_name}) not in vlinks.h5 -- ingest them first.')

    h, w = rna_mip.shape
    H_rna_within = rna_fov_matrices.get((rna_reference_hybe, rna_fov_matrices.modality), np.eye(3))
    H_dna_within = dna_fov_matrices.get((dna_reference_hybe, dna_fov_matrices.modality), np.eye(3))
    rna_mip_aligned = cv2.warpAffine(rna_mip.astype(np.float32), as_cv2(H_rna_within)[:2], (w, h))
    dna_mip_aligned = cv2.warpAffine(dna_mip.astype(np.float32), as_cv2(H_dna_within)[:2], (w, h))

    H_across = align_readout_to_reference(dna_mip_aligned, rna_mip_aligned, lb, ub,
                                          border_trim=border_trim, max_shift=max_shift)
    if not with_residuals:
        return H_across
    # Score on exactly what the fit saw: same trim, same normalization.
    # Measured on real 1024x1024 MIPs from the NAS store: 97 ms for this
    # whole block vs 8.0 s for the fit itself (~1.2%) -- not worth a
    # separate opt-out.
    moving, reference = dna_mip_aligned, rna_mip_aligned
    if border_trim > 0:
        moving = moving[border_trim:-border_trim, border_trim:-border_trim]
        reference = reference[border_trim:-border_trim, border_trim:-border_trim]
    moving_norm = preprocess.normalize_to_uint8(moving, lb, ub)
    reference_norm = preprocess.normalize_to_uint8(reference, lb, ub)
    residuals = {'residual_before': _reconstruction_residual(moving_norm, reference_norm, np.eye(3)),
                 'residual_after': _reconstruction_residual(moving_norm, reference_norm, as_cv2(H_across))}
    return H_across, residuals


MAX_CROSS_MODAL_Z_PLANES = 80.0


def estimate_cross_modal_z(rna_storage_path, dna_storage_path, fov, rna_hybe, dna_hybe,
                           rna_channel, dna_channel, y0=192, y1=832, x0=192, x1=832,
                           with_focus_diagnostics=False):
    """
    FOV-level cross-modal Z drift, in PLANES, from DNA's frame into RNA's --
    the same direction the 2D H_across is stored in, so both legs of the
    cross-modal bridge share one convention.

    METHOD: 1D cross-correlation of the two references' whole-FOV depth
    profiles, computed from BOTH the ZX and the ZY projection and summed
    before taking the peak. Summing the two correlation curves (rather
    than averaging two separately-chosen peaks) lets a weak axis be
    outvoted instead of contributing its own spurious argmax.

    Focal-plane matching was tried first and is NOT used: measured on real
    data, the variance-of-Laplacian peak of these fiducial stacks
    frequently lands on a stack ENDPOINT (RNA Hyb_130 FOV02 z=0, DNA
    Hyb_400 FOV01 z=161/177, FOV02 z=171/177, prominence 0.20-0.27),
    because the focus curve has no interior maximum. That produced drift
    estimates of -51 and -171 planes against a per-cell truth near -12.
    The focal planes are still REPORTED in `diagnostics` as a cross-check.

    Returns (dz, quality, diagnostics). quality is the peak normalized
    correlation (1.0 = perfect); gate on it and keep the manual override,
    since on equal-depth stacks this returns ~0 where the truth was
    +12/+4 -- a graceful failure (degrades toward no correction) but a
    real one.

    with_focus_diagnostics (default False): also probe each stack's
    focal plane (segment.focus_profile, a per-plane scan of the whole
    stack) and report it in `diagnostics`. Off by default per direct
    measurement: the two probes cost ~43 s of a 57 s call on a LOCAL
    SSD -- 75% of cross-modal alignment's entire per-FOV runtime -- and
    the one production caller (CrossModalAlignmentWorker) discarded the
    diagnostics unread. Focal planes are a cross-check for a human
    investigating a suspect dz, not part of the measurement (the method
    docstring above already rules them out as an estimator), so they are
    computed only when someone asks to see them.
    """
    def depth_profiles(storage_path, hybe, channel):
        # ONE windowed read per hybe -- the ZX and ZY projections are two
        # reductions of the SAME crop, so reading it twice through
        # hybe_zx_projection/hybe_zy_projection (each opens the file and
        # decompresses every covering chunk independently) paid the whole
        # I/O cost twice for identical bytes. Measured: 1.9 s of the old
        # 3.7 s of crop reads per FOV, gone.
        h5path = paths.stack_path(storage_path, fov, hybe)
        with h5py.File(h5path, 'r') as f:
            crop = f[f'/stack/ch{channel}'][y0:y1, x0:x1, :]
        zx = crop.max(axis=0).astype(np.float32)   # (width, depth)
        zy = crop.max(axis=1).astype(np.float32)   # (height, depth)
        return zx.mean(axis=0).astype(np.float64), zy.mean(axis=0).astype(np.float64)

    rna_zx, rna_zy = depth_profiles(rna_storage_path, rna_hybe, rna_channel)
    dna_zx, dna_zy = depth_profiles(dna_storage_path, dna_hybe, dna_channel)

    def z(v):
        v = np.asarray(v, dtype=np.float64)
        return (v - v.mean()) / (v.std() + 1e-9)

    span = int(MAX_CROSS_MODAL_Z_PLANES)
    lags = list(range(-span, span + 1))
    totals, per_axis = [], {}
    for name, (pa, pb) in (('zx', (rna_zx, dna_zx)), ('zy', (rna_zy, dna_zy))):
        a, b = z(pa), z(pb)
        curve = []
        for lag in lags:
            ia = np.arange(len(a)); ib = ia + lag
            m = (ib >= 0) & (ib < len(b))
            curve.append(float(np.mean(a[ia[m]] * b[ib[m]])) if m.sum() >= 50 else -np.inf)
        curve = np.array(curve)
        per_axis[name] = -float(lags[int(np.argmax(curve))])
        totals.append(curve)
    combined = np.where(np.isfinite(totals[0]) & np.isfinite(totals[1]),
                        totals[0] + totals[1], -np.inf)
    best_i = int(np.argmax(combined))
    dz = -float(lags[best_i])
    quality = float(combined[best_i] / 2.0)

    diagnostics = {'zx_dz': per_axis['zx'], 'zy_dz': per_axis['zy']}
    if with_focus_diagnostics:
        from ..segmentation import segment as _segment
        zs_r, v_r = _segment.focus_profile(rna_storage_path, fov, rna_hybe, rna_channel)
        zs_d, v_d = _segment.focus_profile(dna_storage_path, fov, dna_hybe, dna_channel)
        focal_rna, focal_dna = int(zs_r[int(np.argmax(v_r))]), int(zs_d[int(np.argmax(v_d))])
        diagnostics.update({'focal_rna': focal_rna, 'focal_dna': focal_dna,
                            'focal_dz': -float(focal_dna - focal_rna)})
    return dz, quality, diagnostics


# One reusable destination for the Z-stack window read, per THREAD.
#
# The read at the bottom of hybe_zx_projection is a cell-sized crop across
# the full depth: at a measured median crop of 79x79 px that is 79x79x110x2
# = 1.31 MB, and 2.11 MB at 177 planes. Both are over a boundary that was
# measured on this machine, sharply:
#
#     1016 KB  ->  40/40 allocations came back DIRTY  (recycled in-process)
#     1020 KB  ->   0/40 came back dirty              (fresh from the kernel)
#
# Windows' heap hands anything past ~1 MB straight to VirtualAlloc and
# gives it back on free, so every such read takes freshly zeroed pages from
# the kernel and returns them to be zeroed again. This leg runs once per
# (cell, hybe) -- of order 390,000 times in a whole-project run -- so it is
# the single largest generator of demand-zero traffic on the path.
#
# Thread-local, not module-global: pipeline_canvas draws the ZX preview
# from the GUI's own threads while a pool child may be fitting, and one
# shared buffer would have them overwrite each other's pixels.
_ZX_BUFFERS = threading.local()

# The crop is bounded by the cell bbox plus pad. Measured over 227 real
# cells in two FOVs of a real store: median bbox 58x58, p90 66x67, max
# 76x118 -- so 160 leaves better than 2x headroom on the largest observed
# cell. A crop past it still works; it just takes the old fresh-allocation
# path rather than silently reading into a buffer too small for it.
_ZX_MAX_SIDE = 160


def _read_zx_window(ds, ymin, ymax, xmin, xmax):
    """The (h, w, depth) window of `ds`, read into a reused buffer.

    Returns a VIEW of that buffer, valid until this thread's next call --
    every caller here consumes it immediately (a max-projection) and keeps
    nothing. Falls back to a plain fresh read for a crop larger than the
    buffer, or if read_direct is unavailable for this dataset.
    """
    h, w = ymax - ymin, xmax - xmin
    depth = ds.shape[2]
    if h > _ZX_MAX_SIDE or w > _ZX_MAX_SIDE:
        return ds[ymin:ymax, xmin:xmax, :]
    key = (depth, ds.dtype.str)
    buf = getattr(_ZX_BUFFERS, 'buf', None)
    if buf is None or getattr(_ZX_BUFFERS, 'key', None) != key:
        buf = np.empty((_ZX_MAX_SIDE, _ZX_MAX_SIDE, depth), dtype=ds.dtype)
        _ZX_BUFFERS.buf, _ZX_BUFFERS.key = buf, key
    try:
        ds.read_direct(buf, np.s_[ymin:ymax, xmin:xmax, :], np.s_[0:h, 0:w, :])
    except (TypeError, ValueError, OSError):
        return ds[ymin:ymax, xmin:xmax, :]
    return buf[:h, :w, :]


def hybe_zx_projection(storage_path, fov, hybe, channel, ymin, ymax, xmin, xmax, lb, ub, normalize=True):
    """
    A cell-region Z-stack crop, max-projected along the height (Y) axis to
    give an (width, depth) "X-by-Z" image usable for a 1D phase-correlation
    Z-offset estimate. Stack datasets here are (height, width, depth) --
    the DAX-sourced convention established in Phase 1 -- so this projects
    axis 0, not the differently-shaped legacy virtual-link indexing
    AlignmentWidget.cell_based_align uses; conceptually the same idea
    (project out one in-plane axis, compare what's left via phase
    correlation), just adapted to this project's own stack shape.

    ymin/ymax/xmin/xmax must already be valid (real, non-negative,
    in-frame) slice bounds -- this never clamps them itself. A caller
    building a true (possibly out-of-frame) window -- e.g. pipeline_
    canvas.py's true-extent cell crops -- must clip to the real frame
    before calling this, then place the result into its own larger
    canvas; passing a negative index straight through would silently
    wrap via Python/h5py's own negative-index convention instead of
    raising, corrupting the read.

    normalize=True (default): returns a uint8 image, unchanged behavior
    for every existing caller. normalize=False returns the raw float32
    projection instead -- for a caller (pipeline_canvas.py) that needs
    to place this into a NaN-padded true-extent canvas and defer
    normalization until final compositing, matching how the YX crops
    it's paired with are handled (normalizing a uint8-already image a
    second time, or baking in a fixed 0-255 range before it's known
    which pixels end up NaN-masked, would double-stretch contrast).

    Public (not module-private) since canvas/pipeline_canvas.py's
    draw_cell_alignment_preview_3col reuses it directly to show the ZX
    plane in the cell-alignment preview, at the same 3 correction stages
    as the YX plane -- the same projection this function's own caller
    below (compute_cell_alignment) uses for its z-depth refinement.
    """
    h5path = paths.stack_path(storage_path, fov, hybe)
    # Opened per call, deliberately. A handle cache was written and MEASURED
    # here, and it bought nothing: holding the file removes only the open,
    # and warm that is 0.07 ms of a 1.04 ms call (6.4%; the rest is 0.49 ms
    # reading the window and 0.40 ms normalizing). End to end on the real
    # store -- 6 workers, 15 hybes, 12 cell-sized windows each, ABAB over
    # cold FOVs 8-11 -- reopening ran 20.20 ms/call against 21.57 ms/call
    # holding the handles: 0.94x, i.e. no gain.
    #
    # An earlier probe claimed 15x for the same cache. It was wrong: its
    # held arm re-read ONE window 12 times from a single dataset object, so
    # HDF5's chunk cache served reads 2..12. Real alignment reads twelve
    # DIFFERENT cell windows per hybe, which share no chunks, so that reuse
    # never happens. Data-level reuse would need cells grouped by region --
    # a separate question from holding the file.
    #
    # And it is not free even at 1.0x. Measured here: os.replace succeeds
    # with no handle open and fails with PermissionError [WinError 5] while
    # a read handle is open. Ingestion writes every stack as a .part it
    # then replaces, so a held handle can make that swap fail outright.
    # The overwrite path now stops readers first, which is why the cache
    # was safe enough to test -- but nothing bought it a reason to exist.
    with h5py.File(h5path, 'r') as f:
        ds = f[f'/stack/ch{channel}']
        stack = _read_zx_window(ds, ymin, ymax, xmin, xmax)
        projection = stack.max(axis=0)      # (width, depth); a small array
    return preprocess.normalize_to_uint8(projection, lb, ub) if normalize else projection.astype(np.float32)

def pick_channel_by_type(record, channel_type):
    """Resolve a channel CHOICE to this hybe's actual channel.

    'fiducial' -> always the fiducial; 'readout' -> the FIRST
    non-fiducial in the layout's channel order (falls back to fiducial
    if a hybe genuinely has none). Any other value is a CONCRETE
    channel (e.g. '488' -- the generalization the two role labels
    could not express once hybes carry more than one readout channel,
    per report): used when this hybe has it, else the readout rule --
    per-hybe channel lists differ, and a hybe lacking the requested
    wavelength still needs SOME same-role crop to align."""
    fiducial = record['fiducial_channel']
    if channel_type == 'fiducial':
        return fiducial
    if channel_type != 'readout':
        for c in record['channels']:
            if str(c) == str(channel_type):
                return c
    readout = [c for c in record['channels'] if c != fiducial]
    return readout[0] if readout else fiducial


def _cell_native_crop(ctx, hybe, channel):
    """
    The crop resolver formerly closed over compute_cell_alignment's
    locals -- top-level now so _cell_hybe_task can run in a spawn-pool
    child. `ctx` is the plain-dict loop-invariant context that function
    builds (see its executor parameter); everything read here is
    picklable by construction.
    """
    height, width = ctx['frame_shape']
    # cell.reference_hybe's frame -> hybe's own native frame, via
    # hybe_to_cellref_matrix (see its own docstring) -- always the
    # SAME general formula, no "is this hybe special" branch; see
    # compute_cell_alignment's original inline comment for the
    # mispositioned-crop bug this formula fixed.
    H_to_hybe = la.inv(hybe_to_cellref_matrix(ctx['fov_matrices'], ctx['cell_reference_hybe_matrix'], hybe))
    cy, cx = align_cell((ctx['y_ref'], ctx['x_ref']), H_to_hybe, (height, width))
    x, y = cx, cy
    if len(x) == 0:
        return None  # cell doesn't overlap this hybe's frame at all
    pad = ctx['pad']
    ymin, ymax = max(0, int(y.min()) - pad), min(height, int(y.max()) + pad + 1)
    xmin, xmax = max(0, int(x.min()) - pad), min(width, int(x.max()) + pad + 1)
    # YX (2D) residual fit -- not one of the 3D exceptions (only the
    # ZX/depth leg, via hybe_zx_projection, genuinely needs the raw
    # Z-stack) -- reads vlinks.h5's real MIP copy, window-only (the
    # bounds are already known; measured 58x cheaper than a full read).
    mip = analysis_store.read_hybe_mip(ctx['storage_path'], ctx['fov'], hybe, channel,
                                     window=(ymin, ymax, xmin, xmax))
    if mip is None:
        return None  # not ingested -- same graceful "no crop" path as no-overlap
    crop = preprocess.normalize_to_uint8(mip, ctx['lb'], ctx['ub'])
    return crop, (ymin, ymax, xmin, xmax)


def compute_cell_alignment(cell, storage_path, fov, hybe_records, fov_matrices,
                           reference_hybe=None, channel_type='readout',
                           pad=10, lb=0.3, ub=0.9999, including_z=True,
                           cell_reference_hybe_matrix=None, modality=None,
                           background_clip=None, fit_method='phase_correlation',
                           integer_shift=False, z_max_shift=None, executor=None):
    """
    Compute this cell's own per-hybe alignment correction (matrices['yx']
    and matrices['zx']), refining the already-established FOV-level matrix
    with a small residual derived from RAW, native-frame crops -- ports
    AlignmentWidget.cell_based_align's algorithm (codelab_pipeline/alignment.py)
    as a standalone, non-widget function. Unlike SG_analysis.ipynb's
    version, this never warps a whole image -- for each hybe, the cell mask
    coordinates are inverse-warped via align_cell to find that hybe's own
    native-frame crop, which is compared directly (via phase correlation)
    against the reference hybe's native-frame crop at the cell's own bbox.
    Only crops, never full images, ever get resampled.

    reference_hybe/channel_type: previously this always silently reused
    cell.reference_hybe (the segmentation hybe) and the fiducial channel for
    every crop -- conflating "which hybe the cell mask was drawn on" with
    "which hybe/channel this refinement should anchor to," which is not
    always the same choice (per explicit request: cell-based alignment
    needs its own reference hybe + channel, defaulting to the readout
    channel -- actual signal correlates with real per-cell content better
    than fiducial does for this residual). reference_hybe=None keeps the
    old default (cell.reference_hybe); channel_type='readout' is the new
    default, 'fiducial' restores the old channel choice.

    When reference_hybe differs from cell.reference_hybe, that hybe's own
    native-frame crop is derived the exact same way every OTHER hybe's is
    (via fov_matrices, not treated as the special "already in the right
    frame" case cell.reference_hybe gets) -- fov_matrices always has a
    real entry for every hybe (identity for whichever hybe FOV-alignment
    itself used as reference), so this generalizes cleanly rather than
    requiring reference_hybe to be that same FOV-alignment reference.

    fov_matrices: {hybe: 3x3} -- the already-established FOV-level matrices
    for this FOV/modality (H_within, or H_within composed with H_across for
    a cross-modal cell) -- always an explicit input; this function never
    re-derives or infers them.

    cell_reference_hybe_matrix: cell.reference_hybe's own transform into
    fov_matrices' shared within-experiment frame, i.e. what
    fov_matrices.get(cell.reference_hybe, identity) would give IF
    cell.reference_hybe were a key in fov_matrices. Defaults to exactly
    that lookup, which is correct whenever fov_matrices is the SAME
    modality cell.reference_hybe belongs to (the common, same-modality
    call). Pass this EXPLICITLY for a cross-modal call (fov_matrices =
    the OTHER modality's own matrices): cell.reference_hybe is a same-
    modality-only hybe name and is never a real key in the other
    modality's own fov_matrices dict, so the default lookup would
    silently resolve to identity there regardless of the real value --
    the caller must resolve it from the SAME-MODALITY fov_matrices it
    already has in scope and pass it through explicitly instead.

    background_clip: None (default, no behavior change), 'yen', or 'otsu'
    -- soft-subtracts each crop's own background threshold (see
    _clip_background) before the phase-correlation fit AND the
    reconstruction-residual quality gate, on EVERY hybe's target crop and
    once on the shared reference crop. Never affects `reference_crop`/
    `target_crop` themselves (matrix_provenance, callers) -- only the
    fitting/scoring inputs. Opt-in: still experimental, not yet the
    pipeline's own default.

    fit_method: 'phase_correlation' (default, no behavior change -- one
    whole-crop cv2.phaseCorrelate fit, as before) or 'multi_peak' (see
    _multi_peak_translation) -- fits several distinct bright peaks
    independently and takes their median as a consensus H2, robust to a
    single unusually bright/isolated feature dominating a whole-crop fit.
    Opt-in: still experimental, not yet the pipeline's own default.

    integer_shift: False (default, no behavior change) or True -- rounds
    H2 to the nearest whole pixel immediately after fitting (either
    method), BEFORE both reject gates score it. Per explicit request,
    this cell-level step is meant to be a small translation-only
    REFINEMENT, not a free continuous optimization -- rounding keeps it
    "no sub-pixel" the same way it's already "no rotation". Opt-in: still
    experimental, not yet the pipeline's own default.

    NOTE: this option's ORIGINAL second justification is now obsolete and
    deliberately no longer claimed here. It used to argue that the
    preview could only reposition an integer-rounded crop window, so a
    sub-pixel H2 would render as anywhere from invisible to a blunt 1px
    snap, and rounding made the gate score what the preview could
    actually draw. draw_cell_alignment_preview_3col now applies the
    residual to the IMAGE via warpAffine (float, both the YX and the ZX
    row) instead of moving an int() window, so it displays a sub-pixel
    residual at full precision -- the display no longer constrains the
    fit, and rounding is now purely a modelling choice about what a
    cell-level refinement should be allowed to express.

    modality: which modality hybe_records/fov_matrices/storage_path belong
    to for THIS call -- cell.matrices/cell.matrix_provenance are keyed by
    (hybe, modality), never bare hybe, because the cross-modal "bridge"
    hybe (e.g. Hyb_130) is a real, distinct file in BOTH modalities; a
    bare-string key would collide the two calls (same-modality and
    cross-modal) this function is meant to be invoked with once each per
    cell, silently dropping whichever one runs second. Defaults to
    cell.reference_modality (the common, same-modality call); the
    cross-modal call MUST pass the other modality's own name explicitly.

    Writes cell.matrices[(hybe, modality)] = {'yx': H_within_or_across @
    ... composed with the cell's own residual, 'zx': depth correction}
    and cell.matrix_provenance[(hybe, modality)] for traceability,
    mirroring the FOV-level /matrix/{hybe} provenance from Phase 1.

    cell.matrices[(hybe, modality)]['yx'] maps hybe's own native frame
    into THIS RUN's own reference_hybe's frame -- not fov_matrices' own
    shared frame, and not cell.reference_hybe's frame either. This means
    cell.reference_hybe is not privileged here: it goes through this
    exact per-hybe loop like any other hybe (no identity shortcut),
    because it genuinely has its own real, uncorrected FOV drift
    relative to reference_hybe, same as any other hybe -- forcing it
    to identity would silently discard that real correction rather
    than apply it.

    The cell-level residual (H2, fitted via phase correlation between the
    target and reference crops) is rejected -- falls back to identity,
    keeping the FOV/cross-modal-only alignment -- under THREE independent
    gates: (1) its magnitude exceeds `pad` (the crop couldn't have
    contained real content that far out), (2) its magnitude exceeds the
    hard, non-configurable engine-level bound (see this module's own
    MAX_ALIGNMENT_TRANSLATION_PX -- independent of `pad`, which is user-
    tunable and sometimes intentionally larger), and (3) it doesn't
    actually improve the crop match (_reconstruction_residual after
    applying H2 isn't strictly better than before, both measured on the
    same target/reference crops this residual was fitted from) -- phase
    correlation can converge to a local minimum that's a worse match than
    no correction at all, which neither magnitude bound alone catches
    since the bad shift can still be small.

    cell.matrix_anchors[modality] is ALSO written here (once per call) --
    reference_hybe's own transform into fov_matrices' shared frame
    (what used to be folded into every cell.matrices entry). A consumer
    that needs a hybe's position relative to THIS cell's own native
    frame (where cell.area actually lives), or relative to some OTHER
    hybe/modality entirely, goes through ACell.matrix_to/matrix_between,
    which bridge through matrix_anchors only when the two entries being
    related come from different modalities/runs -- same-modality
    composition never needs the shared frame at all, since both sides
    already share this run's own reference_hybe by construction.

    executor: None (default -- serial per-hybe loop, byte-identical to
    the pre-refactor behavior) or a spawn-context ProcessPoolExecutor to
    fan the per-hybe work (_cell_hybe_task: crop reads, fit, gates, Z
    leg) across children. Measured on the real store, a cell's cost is
    99.5% NAS I/O (2 file opens + 2 reads per hybe), so overlapping the
    hybes is the whole win. THE ONE-AXIS RULE (per explicit design):
    only the SINGLE-task path (Preview This Cell / a one-cell FOV) may
    pass an executor -- CellAlignmentWorker's multi-cell batch already
    parallelizes across cells and its pool children call this with
    executor=None, so per-hybe and per-cell pooling can never stack.
    """
    height, width = cell.frame_shape
    reference_hybe = reference_hybe or cell.reference_hybe
    modality = modality if modality is not None else cell.reference_modality
    y_ref, x_ref = cell.area  # (y, x), native to cell.reference_hybe's own frame
    if cell_reference_hybe_matrix is None:
        cell_reference_hybe_matrix = fov_matrices.get((cell.reference_hybe, cell.reference_modality), np.eye(3))

    # The loop-invariant context every per-hybe task reads -- a PLAIN,
    # picklable dict, because _cell_hybe_task must be able to run in a
    # spawn-pool child (see `executor` above). The reference-derived
    # entries (ref_channel/crop/window, H_ref_to_shared) are filled in
    # just below, once the reference crop exists; ref_zx stays None until
    # the first task that needs it fills it (serial path -- same lazy
    # semantics as before: a call with no processable hybes does no Z
    # read at all) or the pooled dispatch fills it eagerly (children get
    # pickled COPIES of this dict, so a child's lazy fill could never
    # propagate back and each child would pay its own reference read).
    ctx = {
        'storage_path': storage_path, 'fov': fov, 'frame_shape': (height, width),
        'y_ref': y_ref, 'x_ref': x_ref, 'pad': pad, 'lb': lb, 'ub': ub,
        'fov_matrices': fov_matrices, 'cell_reference_hybe_matrix': cell_reference_hybe_matrix,
        'reference_hybe': reference_hybe, 'channel_type': channel_type,
        'background_clip': background_clip, 'fit_method': fit_method,
        'integer_shift': integer_shift, 'including_z': including_z,
        'z_max_shift': z_max_shift, 'cell_id': cell.id,
        'ref_channel': None, 'reference_crop_for_fit': None, 'ref_window': None,
        'ref_zx': None, 'H_ref_to_shared': None,
    }

    record_by_folder = {r['folder']: r for r in hybe_records}
    if reference_hybe not in record_by_folder:
        # A bare dict lookup here raised an opaque KeyError -- e.g. when a
        # caller's reference_hybe belongs to a different modality than the
        # hybe_records passed in (this project's own same-modality vs.
        # cross-modal hybe_records are always modality-specific, never
        # mixed). Surface the actual mismatch instead.
        raise ValueError(f"reference_hybe '{reference_hybe}' is not in the {len(hybe_records)} "
                         f"hybe_records passed to compute_cell_alignment (modality={modality}) -- "
                         f"it likely belongs to a different modality.")
    ref_record = record_by_folder[reference_hybe]
    ref_channel = pick_channel_by_type(ref_record, channel_type)
    ctx['ref_channel'] = ref_channel
    ref_result = _cell_native_crop(ctx, reference_hybe, ref_channel)
    if ref_result is None:
        # No reference crop exists to fit ANY of this pass's hybes
        # against -- the cell is wholly outside this modality's view.
        # Per the same principle as the per-hybe no-overlap branch in
        # _cell_hybe_task (omitting entries was that code's own
        # confirmed-wrong earlier behavior): write explicit IDENTITY
        # residuals with honest provenance for the whole pass, so reads
        # compose the FOV/cross-modal matrices alone and append mode
        # sees the pass as done -- then raise so callers log the skip.
        H_ref_to_shared = fov_matrices.get(
            (reference_hybe, fov_matrices.modality), np.eye(3))
        cell.matrix_anchors[modality] = H_ref_to_shared
        for record in hybe_records:
            key = (record['folder'], modality)
            cell.matrices[key] = {'yx': np.eye(3), 'dz': 0.0,
                                  'yx_is_residual': True}
            if record['folder'] == reference_hybe:
                # the reference carries no provenance entry -- and a
                # stale one from a run where it wasn't reference must go
                cell.matrix_provenance.pop(key, None)
                continue
            H1 = compose_chain([
                fov_matrices.get((record['folder'], fov_matrices.modality),
                                 np.eye(3)),
                la.inv(H_ref_to_shared)])
            cell.matrix_provenance[key] = {
                'reference_sequence':
                    f'{record["folder"]}(cell {cell.id})->{reference_hybe} '
                    f'[cell-level residual SKIPPED: cell does not overlap '
                    f'reference hybe {reference_hybe}\'s frame, fell back '
                    f'to FOV/cross-modal only]',
                'steps': np.stack([H1, np.eye(3)]),
            }
        raise CellOffFrameError(
            f"Cell {cell.id} doesn't overlap reference hybe "
            f"{reference_hybe}'s frame")
    reference_crop, ref_window = ref_result
    ctx['ref_window'] = ref_window
    # Optional background suppression, computed ONCE for reference_crop
    # (fixed for the whole call) -- see _clip_background's own docstring.
    # Only affects the phase-correlation fit and the quality gate in the
    # per-hybe task; never applied to `reference_crop`/`target_crop`
    # themselves, which stay the plain quantile-normalized crops
    # everywhere else (matrix_provenance, callers) that might reasonably
    # expect them.
    ctx['reference_crop_for_fit'] = (_clip_background(reference_crop, background_clip)
                                     if background_clip else reference_crop)

    # The Z leg's REFERENCE projection (ctx['ref_zx']) is a loop
    # invariant: it depends only on reference_hybe, ref_channel and
    # ref_window, all fixed above, plus lb/ub. It used to be re-read from
    # disk once per hybe -- 110 identical windowed stack reads per cell
    # on a real FOV, measured at 14.6 ms each (20% of a cell). Filled
    # lazily by the first task that needs it (serial), or eagerly by the
    # pooled dispatch below -- see the ctx comment above.

    # relative transform FROM reference_hybe's native frame TO the shared
    # FOV frame -- identity when reference_hybe is also the FOV-alignment's
    # own reference hybe (fov_matrices always carries a real entry per
    # hybe, see read_same_modality_matrices). Stashed on the cell (not
    # composed into every entry below) -- see this function's own
    # docstring for why, and ACell.matrix_to/matrix_between for how a
    # consumer uses it to bridge across modalities when needed.
    H_ref_to_shared = fov_matrices.get((reference_hybe, fov_matrices.modality), np.eye(3))
    ctx['H_ref_to_shared'] = H_ref_to_shared
    cell.matrix_anchors[modality] = H_ref_to_shared

    # This run is the sole source of truth for every hybe in
    # hybe_records -- clear any stale entry up front so a hybe that fails
    # in its task (out-of-frame or rejected), or that changed roles
    # between runs (e.g. was a regular target hybe with a real residual
    # in a previous run but IS this run's reference_hybe), ends up with
    # exactly what this run computed, never a leftover from a previous
    # call on this same (real, persisted-to-disk) cell object with
    # different params or before the reject bound existed.
    for record in hybe_records:
        stale_key = (record['folder'], modality)
        cell.matrices.pop(stale_key, None)
        cell.matrix_provenance.pop(stale_key, None)

    targets = [r for r in hybe_records if r['folder'] != reference_hybe]
    results_by_hybe = {}
    if executor is not None and len(targets) > 1:
        # Per-hybe fan-out across the caller's spawn pool -- the
        # single-task path's ONE axis of parallelism (see `executor` in
        # this function's docstring: the across-cells batch pool never
        # passes one, so the two axes can never stack). Children receive
        # pickled COPIES of ctx, so the reference Z projection is read
        # once HERE rather than lazily (a lazy fill inside a child could
        # never propagate back, and every child would pay its own read).
        if including_z and ctx['ref_zx'] is None:
            ctx['ref_zx'] = hybe_zx_projection(storage_path, fov, reference_hybe, ref_channel,
                                               *ref_window, lb, ub)
        futures = [executor.submit(_cell_hybe_task, ctx, r) for r in targets]
        for future in as_completed(futures):
            hybe_result, entry, provenance = future.result()
            results_by_hybe[hybe_result] = (entry, provenance)
    else:
        for r in targets:
            hybe_result, entry, provenance = _cell_hybe_task(ctx, r)
            results_by_hybe[hybe_result] = (entry, provenance)

    # Merge in hybe_records order -- identical insertion order to the old
    # inline loop, pooled or not, so nothing downstream can observe which
    # dispatch ran.
    for record in hybe_records:
        hybe = record['folder']
        key = (hybe, modality)
        if hybe == reference_hybe:
            # hybe's own transform into ITSELF (this run's reference_hybe)
            # is trivially identity -- no fitting to do (a frame's position
            # relative to itself isn't a fitting question; the phase-
            # correlation "residual" that would come out of comparing
            # reference_hybe's crop against itself is pure noise, observed
            # on real data at ~0.9px). This covers cell.reference_hybe too
            # when it happens to also be this run's reference_hybe -- no
            # separate identity shortcut is needed for cell.reference_hybe
            # on its own, since it has real, uncorrected FOV drift like any
            # other hybe whenever it ISN'T this run's reference_hybe, and
            # forcing it to identity in that case would silently discard
            # that real correction rather than compute it.
            cell.matrices[key] = {'yx': np.eye(3), 'dz': 0.0, 'yx_is_residual': True}
            continue
        entry, provenance = results_by_hybe[hybe]
        cell.matrices[key] = entry
        cell.matrix_provenance[key] = provenance
    return


def _cell_hybe_task(ctx, record):
        """
        The per-hybe body of compute_cell_alignment -- native crop,
        residual fit, three reject gates, Z leg -- returning (hybe,
        matrices entry, provenance entry) instead of writing to the cell,
        so the single-task path can run it in spawn-pool children (see
        compute_cell_alignment's `executor` parameter). The body below is
        the old inline loop body moved VERBATIM (the deep indent is the
        price of that byte-level fidelity), equivalence verified against
        a captured pre-refactor baseline on the real store, serial AND
        pooled.
        """
        hybe = record['folder']
        storage_path, fov = ctx['storage_path'], ctx['fov']
        reference_hybe, ref_channel = ctx['reference_hybe'], ctx['ref_channel']
        fov_matrices, H_ref_to_shared = ctx['fov_matrices'], ctx['H_ref_to_shared']
        reference_crop_for_fit = ctx['reference_crop_for_fit']
        (rymin, rymax, rxmin, rxmax) = ctx['ref_window']
        pad, lb, ub = ctx['pad'], ctx['lb'], ctx['ub']
        channel_type, background_clip = ctx['channel_type'], ctx['background_clip']
        fit_method, integer_shift = ctx['fit_method'], ctx['integer_shift']
        including_z, z_max_shift = ctx['including_z'], ctx['z_max_shift']

        # H1 (hybe's native frame -> shared frame -> reference_hybe's
        # native frame -> cell.reference_hybe's frame) is pure matrix
        # algebra from fov_matrices -- it never needs any image data, so
        # it's always computable regardless of whether a crop can be
        # built below.
        H1 = compose_chain([fov_matrices[(hybe, fov_matrices.modality)], la.inv(H_ref_to_shared)])

        target_channel = pick_channel_by_type(record, channel_type)
        result = _cell_native_crop(ctx, hybe, target_channel)
        if result is None:
            # Cell doesn't overlap this hybe's frame at all -- no crop
            # exists to fit a cell-level residual against. Per the same
            # "no no-alignment" principle as the reject bound below: this
            # is NOT a reason to omit the hybe from cell.matrices (that
            # was this code's own earlier, WRONG behavior) -- H1 above is
            # still fully valid, so fall back to H1-only (H2=identity)
            # exactly like a rejected residual does.
            # BARE residual (identity here -- no crop, so nothing fitted).
            # H1 is deliberately NOT baked in: it is recomposed at read
            # time from the CURRENT FOV matrices, so re-running FOV
            # alignment updates this cell without a re-fit.
            return hybe, {'yx': np.eye(3), 'dz': 0.0, 'yx_is_residual': True}, {
                'reference_sequence': f'{hybe}(cell {ctx["cell_id"]})->{reference_hybe} '
                                      f'[cell-level residual SKIPPED: cell does not overlap this hybe\'s frame, '
                                      f'fell back to FOV/cross-modal only]',
                'steps': np.stack([H1, np.eye(3)]),
            }
        target_crop, (cymin, cymax, cxmin, cxmax) = result
        target_crop_for_fit = (_clip_background(target_crop, background_clip)
                               if background_clip else target_crop)

        if fit_method == 'multi_peak':
            H2_fitted = _multi_peak_translation(target_crop_for_fit, reference_crop_for_fit)
        else:
            H2_fitted = np.vstack([preprocess.find_translation_via_phase_correlation(
                                       target_crop_for_fit, reference_crop_for_fit),
                                   np.array([0, 0, 1])])
        if integer_shift:
            # Snap to a WHOLE pixel, before gating -- per explicit
            # request: this cell-level step is meant as a small
            # translation-only REFINEMENT (no rotation already; this
            # keeps it "no sub-pixel" too), not a free continuous
            # optimization. Also closes a real, confirmed gap between
            # what the quality gate scores and what the preview can
            # actually show: cv2.phaseCorrelate returns a sub-pixel
            # shift, _reconstruction_residual evaluates it via a genuine
            # bilinear-interpolated warpAffine, but the preview crops
            # never resample pixel content -- they only reposition an
            # INTEGER-rounded crop window. A sub-pixel H2 the gate scores
            # as an improvement can render as anywhere from no visible
            # change to a blunt, un-interpolated 1px snap once displayed.
            #
            # NOT plain round() -- confirmed on real data (cell 3, Hyb_105
            # vs Hyb_101) that nearest-integer rounding can land on the
            # WRONG side of a close call: the continuous fit found
            # dx=0.91, which rounds to 1, but an exhaustive integer sweep
            # found dx=0 scores meaningfully better (reconstruction
            # residual 310 vs 357) -- round() only ever asks "which whole
            # pixel is numerically closest," never "which whole pixel
            # actually reconstructs the reference better," and those two
            # questions can have different answers near a 0.5 boundary.
            # Instead, evaluate all 4 combinations of floor/ceil per axis
            # (the immediate integer neighborhood around the continuous
            # estimate -- exactly where a close call like this lives) via
            # the SAME _reconstruction_residual the outer gate already
            # trusts, and keep whichever actually wins. This is a local
            # search around the continuous fit's own estimate, not a
            # blind wide grid search -- the continuous fit still supplies
            # the neighborhood to search, just not the final answer
            # within it.
            fx, fy = H2_fitted[0, 2], H2_fitted[1, 2]
            candidates = {(dx, dy) for dx in (np.floor(fx), np.ceil(fx))
                                   for dy in (np.floor(fy), np.ceil(fy))}
            best_dx, best_dy = min(
                candidates,
                key=lambda dxdy: _reconstruction_residual(
                    target_crop_for_fit, reference_crop_for_fit,
                    np.array([[1., 0., dxdy[0]], [0., 1., dxdy[1]], [0., 0., 1.]])))
            H2_fitted = np.array([[1., 0., best_dx], [0., 1., best_dy], [0., 0., 1.]])
        # This residual is a FINE-TUNING correction on top of an already-
        # good FOV/cross-modal alignment -- the crop itself only extends
        # `pad` px beyond the cell's expected position, so the true
        # matching content for any real correction can only ever lie
        # within `pad` px of center. A returned shift bigger than that
        # isn't a real correction (the crop couldn't contain the content
        # needed to justify it) -- it's cv2.phaseCorrelate locking onto
        # noise/padding on a low-signal crop (observed on real data: up to
        # several thousand px on crops only ~80px wide).
        #
        # Reject rather than clamp -- but per explicit principle, "reject"
        # NEVER means "no alignment data at all" for this hybe (that was
        # this code's own earlier, WRONG behavior: a `continue` here left
        # hybe out of cell.matrices entirely, indistinguishable from a
        # hybe that never overlapped this frame at all -- silently
        # dropping a hybe from the Results/Overlay cell counts, and
        # discarding the perfectly good FOV/cross-modal alignment along
        # with the untrustworthy refinement). Rejecting the cell-level
        # residual means falling back to identity for H2 specifically --
        # the FOV/cross-modal layer this refinement was built on top of
        # is still valid and must still be written.
        magnitude_rejected = np.hypot(H2_fitted[0, 2], H2_fitted[1, 2]) > pad
        # Independent of pad (which bounds how far the crop itself could
        # even show real content, and is user-tunable, sometimes larger
        # than the hard cap below for legitimate reasons): the hard,
        # non-configurable engine-level translation bound (see this
        # module's own MAX_ALIGNMENT_TRANSLATION_PX) applies here too --
        # a cell-level residual is meant to be a SMALL correction on top
        # of already-good FOV/cross-modal alignment, so anything beyond
        # this bound is an optimizer artifact regardless of what pad
        # happens to be configured to.
        # No `shape` argument here, deliberately: a cell-level residual is
        # translation-only by construction -- every producer of H2_fitted
        # above (_multi_peak_translation, find_translation_via_phase_
        # correlation, and the brute-force dx/dy search) returns a matrix
        # whose 2x2 block is exactly identity. Centre displacement and the
        # raw translation column are therefore identically equal, so the
        # column read is correct and passing shape would be a no-op. Rotation
        # is not expected at this layer at all; it belongs to the FOV/cross-
        # modal fits, which is where _center_displacement matters.
        hard_bound_rejected = not _within_hard_alignment_bounds(H2_fitted)
        # Second, independent gate: phase correlation can lock onto a
        # local minimum that's a worse match than doing nothing at all
        # (a real cv2.phaseCorrelate failure mode -- it doesn't return
        # 0,0 when the target crop is already well-aligned, it can return
        # a small, pad-bound-passing shift that still makes the overlap
        # WORSE). The pad-magnitude bound above only catches wildly wrong
        # shifts; it says nothing about whether a small, plausible-looking
        # shift actually helped. Directly compare image match quality
        # before (H2=identity, i.e. the FOV/cross-modal crop as-is) vs
        # after (H2_fitted applied) on the SAME target_crop/reference_crop
        # this residual was fitted from -- via _reconstruction_residual
        # (the same signal-gated, overlap-guarded metric align_readout_
        # to_reference already uses to pick between candidate transforms,
        # see its own docstring), not a naive nonzero-background MSD --
        # background-to-background pixels are the majority of a real
        # crop and dilute exactly the handful of pixels that would show a
        # bad shift. Reject (fall back to H2=identity) whenever it didn't
        # strictly improve the reconstruction of the reference's real
        # content.
        residual_before = _reconstruction_residual(target_crop_for_fit, reference_crop_for_fit, np.eye(3))
        residual_after = _reconstruction_residual(target_crop_for_fit, reference_crop_for_fit, H2_fitted)
        quality_rejected = not (residual_after < residual_before)
        rejected = magnitude_rejected or hard_bound_rejected or quality_rejected
        H2 = np.eye(3) if rejected else H2_fitted
        # Stored BARE, per explicit decision. H2 is the residual measured
        # against reference_hybe's own crop, in that reference frame.
        #
        # This used to store compose_chain([H1, H2]) -- H1 baked in. That
        # made a stored cell entry a COMPLETE hybe->anchor transform, and
        # therefore froze the FOV matrix as it stood at fit time: re-running
        # FOV alignment changed nothing for any already-fitted cell until it
        # was individually re-fit, silently. Storing only what this step
        # actually measured keeps the three layers independently updatable,
        # which is the whole point of having layers. Consumers recompose H1
        # from CURRENT matrices -- see frames.FrameResolver.to_shared.
        # H2 was fitted x-major (cv2.phaseCorrelate territory); stored
        # y-major like every persisted matrix. The ZX leg below reads
        # H2's own x-major dx BEFORE this conversion, deliberately.
        H_yx = to_yx(H2)

        z_shift = 0.0
        if including_z:
            # z-depth correction uses the same channel_type-resolved
            # channel as the YX fit (ref_channel/target_channel) -- a
            # readout-channel Z crop should show the readout signal, not
            # silently fall back to fiducial regardless of channel_type.
            if ctx['ref_zx'] is None:
                # the serial path's lazy fill -- the SAME ctx dict is
                # reused across hybes there, so this reads at most once;
                # pooled children always receive it pre-filled (see the
                # dispatch in compute_cell_alignment)
                ctx['ref_zx'] = hybe_zx_projection(storage_path, fov, reference_hybe, ref_channel,
                                                   rymin, rymax, rxmin, rxmax, lb, ub)
            ref_zx = ctx['ref_zx']
            target_zx = hybe_zx_projection(storage_path, fov, hybe, target_channel,
                                           cymin, cymax, cxmin, cxmax, lb, ub)
            # hybe_zx_projection returns (width, depth), not (height,
            # width) like every other crop in this module -- H2 (a YX-
            # plane matrix) can't be applied to it directly via
            # cv2.warpAffine: cv2's own (col,row) convention would put
            # H2's X/width component on this array's DEPTH axis and its
            # Y/height component on its WIDTH axis, backwards on both
            # counts (verified via a synthetic single-pixel test). Only
            # the X/width component is meaningful here (pre-aligning this
            # projection's width position with the reference's, matching
            # the same window the YX fit used) -- place H2[0,2] on axis0
            # (width, this array's M[1]/"row" slot) and leave axis1
            # (depth) untouched.
            H2_to_zx = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, H2[0, 2]]])
            target_zx_aligned = cv2.warpAffine(target_zx, H2_to_zx, (target_zx.shape[1], target_zx.shape[0]))
            # Per explicit request: restrict this leg to Z ONLY,
            # algorithmically -- not by fitting a full 2D (x,z) shift and
            # discarding/gating an unwanted x-component after the fact
            # (that was this code's own earlier design: a real x-shift
            # could still slip through whenever it stayed under the 3px
            # reject bound, silently overriding a correctly-chosen H2 --
            # confirmed on real data, cell 3 Hyb_105 vs Hyb_101, where a
            # gate-verified best H2 of (0,-1) still ended up displaced to
            # (1,-1) by this leg's own independent x measurement). X
            # alignment is already H2's own job; collapsing the WIDTH
            # axis away entirely (max-project) before correlating leaves
            # no axis left for an x-shift to exist on, not just one that
            # gets checked and possibly rejected afterward.
            ref_z_profile = ref_zx.max(axis=0)
            target_z_profile = target_zx_aligned.max(axis=0)
            z_shift_fitted = _find_z_shift(target_z_profile, ref_z_profile)
            if integer_shift:
                # Same "no sub-pixel" constraint as H2 -- this leg is a
                # small, whole-pixel-only depth refinement, not a free
                # continuous fit.
                z_shift_fitted = round(z_shift_fitted)
            # Same two-gate treatment as H2 just above, ported to the Z
            # leg -- _find_z_shift is a raw np.correlate(mode='full')
            # argmax with no rejection path of its own, so it was free to
            # lock onto noise exactly the way phase correlation on a
            # low-signal H2 crop was (see that gate's own comment above).
            # Confirmed on real data: cell 16, Hyb_130 vs Hyb_101, where
            # H2 was correctly quality-rejected (residual 1840.3 >= 1840.3,
            # no improvement) on this SAME crop, yet the ungated Z leg
            # still applied z=-35.0px -- a shift nowhere near physically
            # plausible for one hybridization round's worth of Z drift.
            #
            # Magnitude gate. Its purpose is unchanged -- a 'full'-mode
            # correlation can return anything up to the whole profile length
            # as its "best" lag, which is the noise-locking failure mode, not
            # a real registration -- but the BOUND is no longer pad/2.
            #
            # Deriving a depth limit from the XY search radius conflated two
            # unrelated quantities, and measured against a real run it was
            # cutting into genuine drift: applied shifts piled up exactly at
            # the old 5-plane cap while 16.4% of entries were rejected on
            # magnitude alone with |z| up to 11. See
            # MAX_CELL_Z_SHIFT_PLANES for the full measurement.
            z_bound = MAX_CELL_Z_SHIFT_PLANES if z_max_shift is None else float(z_max_shift)
            z_magnitude_rejected = abs(z_shift_fitted) > z_bound
            # Quality gate: mirrors H2's own reconstruction-residual check,
            # applied on the SAME (width, depth) ZX crops the shift was
            # fitted from -- reject unless the shift strictly improves the
            # reconstruction of the reference's real content over doing
            # nothing.
            H_zx_fitted = np.array([[1., 0., z_shift_fitted], [0., 1., 0.], [0., 0., 1.]])
            z_residual_before = _reconstruction_residual(target_zx_aligned, ref_zx, np.eye(3))
            z_residual_after = _reconstruction_residual(target_zx_aligned, ref_zx, H_zx_fitted)
            z_quality_rejected = not (z_residual_after < z_residual_before)
            z_rejected = z_magnitude_rejected or z_quality_rejected
            # Reject rather than clamp, same as H2 -- "reject" here just
            # means z=0 (identity), never dropping the hybe.
            z_shift = 0.0 if z_rejected else z_shift_fitted

        if not including_z:
            zx_note = ''
        elif z_rejected:
            if z_magnitude_rejected:
                z_reject_reason = f'{abs(z_shift_fitted):.1f}px > z_max={z_bound:g}'
            else:
                z_reject_reason = (f'reconstruction residual {z_residual_after:.1f} >= {z_residual_before:.1f} '
                                   f'(no improvement over FOV/cross-modal)')
            zx_note = f' [z-alignment REJECTED: {z_reject_reason}, fell back to z=0.0px]'
        else:
            zx_note = f' [z-alignment applied: z={z_shift:.1f}px]'

        if rejected:
            if magnitude_rejected:
                reject_reason = (f'{np.hypot(H2_fitted[0, 2], H2_fitted[1, 2]):.1f}px > pad={pad}')
            elif hard_bound_rejected:
                reject_reason = (f'{np.hypot(H2_fitted[0, 2], H2_fitted[1, 2]):.1f}px > hard cap='
                                 f'{MAX_ALIGNMENT_TRANSLATION_PX}px')
            else:
                reject_reason = (f'reconstruction residual {residual_after:.1f} >= {residual_before:.1f} '
                                 f'(no improvement over FOV/cross-modal)')
            reject_note = f' [cell-level residual REJECTED: {reject_reason}, fell back to FOV/cross-modal only]'
        else:
            reject_note = ''
        return hybe, {'yx': H_yx, 'dz': float(z_shift), 'yx_is_residual': True}, {
            'reference_sequence': f'{hybe}(cell {ctx["cell_id"]})->{reference_hybe}' + reject_note + zx_note,
            'steps': np.stack([H1, H2]),
        }
