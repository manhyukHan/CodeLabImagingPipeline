import os
from functools import reduce
import numpy as np
import numpy.linalg as la
import h5py

from ..io import preprocess
from ..io import vlinks_store
import cv2

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
    cx,cy = (H[:2]@np.array([x,y,np.ones_like(x)])).astype(int)
    bad = (cx < 0) | ( cy < 0) | (cx >= width) | (cy >= height)
    cx,cy = cx[~bad],cy[~bad]
    adjusted_mask = np.zeros((height,width))
    adjusted_mask[cy,cx] = 1
    closed = cv2.morphologyEx(adjusted_mask, cv2.MORPH_CLOSE, kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    cy,cx = np.where(closed > 0)
    return cy,cx

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
    H_hybe_to_within = fov_matrices.get(hybe, np.eye(3))
    return compose_chain([H_hybe_to_within, la.inv(H_cellref_to_within)])

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


def align_readout_to_reference(moving_mip, reference_mip, lb=0.3, ub=0.9999, border_trim=0, max_shift=None,
                               angle_threshold=0.5):
    """
    Compute the affine-like matrix aligning moving_mip onto reference_mip.
    Takes plain MIP arrays -- usable for both within-experiment (fiducial
    MIP vs. fiducial MIP) and cross-experiment (readout MIP vs. readout
    MIP) alignment; the caller is responsible for always passing
    same-channel-type inputs on both sides, never mixed.

    Two-stage, ORB-first strategy (replaces the earlier "run both, pick
    whichever has lower residual" approach -- that let ORB win purely on
    residual even when its correspondence was spurious, since a wrong
    match can still reconstruct deceptively well):

    1. Run ORB+RANSAC (preprocess.compute_features_affinelike_matrix) and
       read off its rotation angle. ORB is the only one of the two
       methods that can find real rotation at all -- MSD/Powell's
       optimizer converges to angle~0 regardless of true rotation, even
       at 8 degrees (synthetic ground-truth tested) -- so ORB's angle is
       the only signal available for "is there real rotational drift."
    2. If |ORB's angle| < angle_threshold degrees (default 0.5 -- no real
       rotation found): don't trust ORB's translation either. On real
       data (Hyb_130 barcode round vs. a regular hybe) ORB has been
       observed to lock onto a confident-but-wrong correspondence while
       still reporting ~0 rotation -- reporting no rotation doesn't mean
       the match itself was good. Use a fresh, independent free-angle
       MSD/Powell fit instead (effectively translation-only, since
       Powell won't find real rotation either way).
    3. If ORB DID find real rotation: confirm it three ways, not two.
       Re-run MSD/Powell with ORB's exact angle held fixed
       (fixed_angle=<ORB's angle>) to get an independently-optimized
       translation under the SAME rotation ORB claimed, ALSO run a plain
       zero-angle Powell fit as a sanity-check baseline, then keep
       whichever of the three (ORB's own transform, the fixed-angle
       Powell fit, the zero-angle Powell fit) actually reconstructs
       reference_mip better. The zero-angle baseline matters even here:
       without it, a spurious ORB angle (observed on real data: a
       readout-channel cross-modal pair reporting ~162 degrees, plainly
       wrong) and its fixed-angle "confirmation" both inherit the same
       bad rotation and can both fail _reconstruction_residual's overlap
       guard (return inf) -- inf <= inf still picks ORB, so a genuinely
       sane zero-rotation candidate has to be in the running to win.

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
    """
    if border_trim > 0:
        moving_mip = moving_mip[border_trim:-border_trim, border_trim:-border_trim]
        reference_mip = reference_mip[border_trim:-border_trim, border_trim:-border_trim]

    moving_norm = preprocess.normalize_to_uint8(moving_mip, lb, ub)
    reference_norm = preprocess.normalize_to_uint8(reference_mip, lb, ub)

    H_orb = preprocess.compute_features_affinelike_matrix(moving_norm, reference_norm)
    angle_orb = _h_rotation_angle_degrees(H_orb)

    if abs(angle_orb) < angle_threshold:
        H_final = preprocess.compute_msd_homography_matrix(moving_norm, reference_norm, fixed_scale=1.0, fixed_angle=False)
    else:
        H_confirm = preprocess.compute_msd_homography_matrix(moving_norm, reference_norm, fixed_scale=1.0, fixed_angle=angle_orb)
        H_zero = preprocess.compute_msd_homography_matrix(moving_norm, reference_norm, fixed_scale=1.0, fixed_angle=True)
        candidates = [H_orb, H_confirm, H_zero]
        residuals = [_reconstruction_residual(moving_norm, reference_norm, H) for H in candidates]
        H_final = candidates[int(np.argmin(residuals))]

    if max_shift is not None:
        dx, dy = H_final[0, 2], H_final[1, 2]
        if abs(dx) > max_shift or abs(dy) > max_shift:
            H_final = H_final.copy()
            H_final[0, 2] = np.clip(dx, -max_shift, max_shift)
            H_final[1, 2] = np.clip(dy, -max_shift, max_shift)

    return H_final

def write_same_modality_matrices(storage_path, fov, matrices, reference_hybe):
    """
    Persists an already-computed {hybe: matrix} dict (from
    align_same_modality(..., write=False), or a manual-mode staged
    result the user just accepted) into vlinks.h5's /FOV##/matrix/{hybe},
    plus reference_sequence/steps provenance attrs -- split out from
    align_same_modality so the write step can be deferred (manual
    review mode) or run standalone.

    Delegates to vlinks_store rather than writing into each hybe's own raw
    {hybe}_stack.h5 (the previous behavior) -- per explicit principle,
    vlinks.h5 must be the pipeline's authoritative store for this, not N
    scattered raw per-hybe files that require heavy I/O (opening every
    stack file) just to answer "has this FOV been aligned." Outside
    ingestion and 3D localization, the raw stack files should not need to
    be touched at all.
    """
    vlinks_store.write_same_modality_matrices(storage_path, fov, matrices, reference_hybe)


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

    Delegates to vlinks_store (see that module's ingested_hybes_for_fov /
    read_same_modality_matrices) instead of opening each hybe's own raw
    {hybe}_stack.h5 -- this is now a single vlinks.h5 open, not N raw file
    opens, so callers can refresh this freely.
    """
    hybe_list = [record['folder'] for record in hybe_records]
    return vlinks_store.read_same_modality_matrices(storage_path, fov, hybe_list)


def align_same_modality(storage_path, fov, hybe_records, reference_hybe, lb=0.3, ub=0.9999, write=True,
                            border_trim=0, max_shift=None):
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
    """
    record_by_folder = {r['folder']: r for r in hybe_records}
    ref_record = record_by_folder[reference_hybe]

    # FOV-level (same-modality) alignment is not one of the 3D exceptions
    # (3D spot localization, 3D cell-based alignment, 3D spot-based
    # alignment) -- reads vlinks.h5's real MIP copy, never the raw stack
    # file.
    reference_mip = vlinks_store.read_hybe_mip(storage_path, fov, reference_hybe, ref_record['fiducial_channel'])
    if reference_mip is None:
        raise ValueError(f'FOV{fov:02d} {reference_hybe} not in vlinks.h5 -- ingest it first.')

    matrices = {}
    for record in hybe_records:
        hybe = record['folder']
        if hybe == reference_hybe:
            H = np.eye(3)
        else:
            moving_mip = vlinks_store.read_hybe_mip(storage_path, fov, hybe, record['fiducial_channel'])
            if moving_mip is None:
                raise ValueError(f'FOV{fov:02d} {hybe} not in vlinks.h5 -- ingest it first.')
            H = align_readout_to_reference(moving_mip, reference_mip, lb, ub, border_trim=border_trim, max_shift=max_shift)
        matrices[hybe] = H

    if write:
        write_same_modality_matrices(storage_path, fov, matrices, reference_hybe)

    return matrices

def link_cross_modal(rna_storage_path, dna_storage_path, fov,
                      rna_fov_matrices, dna_fov_matrices,
                      rna_reference_hybe='Hyb_500', dna_reference_hybe='Hyb_400',
                      channel_type='readout', lb=0.3, ub=0.9999,
                      border_trim=0, max_shift=None):
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
    """
    # Cross-modality alignment is not one of the 3D exceptions -- reads
    # vlinks.h5's real MIP copies, never the raw stack file.
    mip_fn = vlinks_store.fiducial_channel_mip if channel_type == 'fiducial' else vlinks_store.readout_channel_mip
    rna_mip = mip_fn(rna_storage_path, fov, rna_reference_hybe)
    dna_mip = mip_fn(dna_storage_path, fov, dna_reference_hybe)
    if rna_mip is None or dna_mip is None:
        raise ValueError(f'FOV{fov:02d}: {rna_reference_hybe} (RNA) and/or {dna_reference_hybe} (DNA) '
                         f'not in vlinks.h5 -- ingest them first.')

    h, w = rna_mip.shape
    H_rna_within = rna_fov_matrices.get(rna_reference_hybe, np.eye(3))
    H_dna_within = dna_fov_matrices.get(dna_reference_hybe, np.eye(3))
    rna_mip_aligned = cv2.warpAffine(rna_mip.astype(np.float32), H_rna_within[:2], (w, h))
    dna_mip_aligned = cv2.warpAffine(dna_mip.astype(np.float32), H_dna_within[:2], (w, h))

    return align_readout_to_reference(dna_mip_aligned, rna_mip_aligned, lb, ub, border_trim=border_trim, max_shift=max_shift)


def write_cross_modal_matrix(dna_storage_path, fov, H, rna_reference_hybe, dna_reference_hybe, channel_type):
    """
    Persists an already-computed H_across (from link_cross_modal) into a new
    /matrix_across dataset in the DNA reference hybe's own H5 file --
    link_cross_modal itself never wrote anywhere (unlike
    align_same_modality, which always did), so this closes that gap.
    Written into the DNA side specifically since H_across is, by
    link_cross_modal's own convention, the matrix that maps DNA's
    within-experiment frame onto RNA's -- RNA's own frame is the shared
    global frame and never needs an /matrix_across entry of its own.
    Mirrors /matrix/{hybe}'s existing provenance-attrs convention
    (reference_sequence/steps) so both are discoverable the same way.
    """
    h5path = os.path.join(dna_storage_path, f'FOV{fov:02d}', f'{dna_reference_hybe}_stack.h5')
    with h5py.File(h5path, 'r+') as f:
        if '/matrix_across' not in f:
            f.create_dataset('/matrix_across', shape=(3, 3), dtype='float64')
        f['/matrix_across'][:] = H
        f['/matrix_across'].attrs['rna_reference_hybe'] = rna_reference_hybe
        f['/matrix_across'].attrs['dna_reference_hybe'] = dna_reference_hybe
        f['/matrix_across'].attrs['channel_type'] = channel_type


def read_cross_modal_matrix(dna_storage_path, fov, dna_reference_hybe):
    """
    Reads back an already-computed/written H_across from the DNA reference
    hybe's own H5 (see write_cross_modal_matrix) -- returns None if nothing
    has been written yet for this FOV (unlike within-experiment matrices,
    there's no identity default seeded at ingestion time here: a FOV
    genuinely hasn't been cross-modal-aligned until this has been run at
    least once, so None is the honest answer, not a fabricated identity).
    This is the read-back half needed so a cross-modal overlay can be
    re-shown without requiring the result to still be in the in-memory
    cross_modal_result dict from the current session.
    """
    h5path = os.path.join(dna_storage_path, f'FOV{fov:02d}', f'{dna_reference_hybe}_stack.h5')
    try:
        with h5py.File(h5path, 'r') as f:
            if '/matrix_across' not in f:
                return None
            return f['/matrix_across'][:]
    except OSError:
        return None


def hybe_zx_projection(storage_path, fov, hybe, channel, ymin, ymax, xmin, xmax, lb, ub):
    """
    A cell-region Z-stack crop, max-projected along the height (Y) axis to
    give an (width, depth) "X-by-Z" image usable for a 1D phase-correlation
    Z-offset estimate. Stack datasets here are (height, width, depth) --
    the DAX-sourced convention established in Phase 1 -- so this projects
    axis 0, not the differently-shaped legacy virtual-link indexing
    AlignmentWidget.cell_based_align uses; conceptually the same idea
    (project out one in-plane axis, compare what's left via phase
    correlation), just adapted to this project's own stack shape.

    Public (not module-private) since canvas/pipeline_canvas.py's
    draw_cell_alignment_preview_3col reuses it directly to show the ZX
    plane in the cell-alignment preview, at the same 3 correction stages
    as the YX plane -- the same projection this function's own caller
    below (compute_cell_alignment) uses for its z-depth refinement.
    """
    h5path = os.path.join(storage_path, f'FOV{fov:02d}', f'{hybe}_stack.h5')
    with h5py.File(h5path, 'r') as f:
        stack = f[f'/stack/ch{channel}'][ymin:ymax, xmin:xmax, :]
    projection = stack.max(axis=0)  # (width, depth)
    return preprocess.normalize_to_uint8(projection, lb, ub)

def pick_channel_by_type(record, channel_type):
    """'readout' -> that hybe's one non-fiducial channel (falls back to
    fiducial if a hybe genuinely has none); 'fiducial' -> always fiducial."""
    fiducial = record['fiducial_channel']
    if channel_type == 'fiducial':
        return fiducial
    readout = [c for c in record['channels'] if c != fiducial]
    return readout[0] if readout else fiducial


def compute_cell_alignment(cell, storage_path, fov, hybe_records, fov_matrices,
                           reference_hybe=None, channel_type='readout',
                           pad=10, lb=0.3, ub=0.9999, including_z=True,
                           cell_reference_hybe_matrix=None, modality=None):
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

    modality: which modality hybe_records/fov_matrices/storage_path belong
    to for THIS call -- cell.matrices/cell.matrix_provenance are keyed by
    (hybe, modality), never bare hybe, because the cross-modal "bridge"
    hybe (e.g. Hyb_130) is a real, distinct file in BOTH modalities; a
    bare-string key would collide the two calls (same-modality and
    cross-modal) this function is meant to be invoked with once each per
    cell, silently dropping whichever one runs second. Defaults to
    cell.modality (the common, same-modality call); the cross-modal call
    MUST pass the other modality's own name explicitly.

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
    """
    height, width = cell.frame_shape
    reference_hybe = reference_hybe or cell.reference_hybe
    modality = modality if modality is not None else cell.modality
    x_ref, y_ref = cell.area  # always native to cell.reference_hybe's own frame
    if cell_reference_hybe_matrix is None:
        cell_reference_hybe_matrix = fov_matrices.get(cell.reference_hybe, np.eye(3))

    def _native_crop(hybe, channel):
        # cell.reference_hybe's frame -> hybe's own native frame, via
        # hybe_to_cellref_matrix (see its own docstring) -- always the
        # SAME general formula, no "is this hybe special" branch. When
        # hybe IS cell.reference_hybe (same modality too), this
        # mathematically reduces to exactly identity on its own (both
        # legs of the composition become the same fov_matrices lookup
        # and cancel) -- align_cell itself checks the MATRIX for that,
        # once, and skips its own resampling machinery accordingly, so
        # there's no need (and no precision cost) to special-case it
        # here too. Previously this used fov_matrices[hybe] directly,
        # silently assuming cell.reference_hybe's frame WAS fov_matrices'
        # own shared frame -- true only when segmentation and FOV/cell
        # alignment share the same reference hybe. When they don't,
        # every non-cell.reference_hybe crop -- including THIS RUN'S OWN
        # reference_hybe crop -- was mispositioned by cell.reference_hybe's
        # own real FOV correction, so phase correlation rediscovered that
        # same correction as a "residual" and it got applied AGAIN on top
        # of the already-correct FOV matrix (observed on real data: a
        # real -3.2px FOV correction plus a "residual" phase correlation
        # found by comparing crops that never had that correction applied
        # -> -6.2px total applied, when ~0px residual was correct).
        H_to_hybe = la.inv(hybe_to_cellref_matrix(fov_matrices, cell_reference_hybe_matrix, hybe))
        cy, cx = align_cell((y_ref, x_ref), H_to_hybe, (height, width))
        x, y = cx, cy
        if len(x) == 0:
            return None  # cell doesn't overlap this hybe's frame at all
        ymin, ymax = max(0, int(y.min()) - pad), min(height, int(y.max()) + pad + 1)
        xmin, xmax = max(0, int(x.min()) - pad), min(width, int(x.max()) + pad + 1)
        # YX (2D) residual fit -- not one of the 3D exceptions (only the
        # ZX/depth leg below, via hybe_zx_projection, genuinely needs the
        # raw Z-stack) -- reads vlinks.h5's real MIP copy.
        mip = vlinks_store.read_hybe_mip(storage_path, fov, hybe, channel)
        if mip is None:
            return None  # not ingested -- same graceful "no crop" path as no-overlap
        crop = preprocess.normalize_to_uint8(mip[ymin:ymax, xmin:xmax], lb, ub)
        return crop, (ymin, ymax, xmin, xmax)

    record_by_folder = {r['folder']: r for r in hybe_records}
    ref_record = record_by_folder[reference_hybe]
    ref_channel = pick_channel_by_type(ref_record, channel_type)
    ref_result = _native_crop(reference_hybe, ref_channel)
    if ref_result is None:
        raise ValueError(f"Cell {cell.id} doesn't overlap reference hybe {reference_hybe}'s frame")
    reference_crop, (rymin, rymax, rxmin, rxmax) = ref_result
    # relative transform FROM reference_hybe's native frame TO the shared
    # FOV frame -- identity when reference_hybe is also the FOV-alignment's
    # own reference hybe (fov_matrices always carries a real entry per
    # hybe, see read_same_modality_matrices). Stashed on the cell (not
    # composed into every entry below) -- see this function's own
    # docstring for why, and ACell.matrix_to/matrix_between for how a
    # consumer uses it to bridge across modalities when needed.
    H_ref_to_shared = fov_matrices.get(reference_hybe, np.eye(3))
    cell.matrix_anchors[modality] = H_ref_to_shared

    for record in hybe_records:
        hybe = record['folder']
        # This run is the sole source of truth for every hybe in
        # hybe_records -- clear any stale entry up front (both branches
        # below) so a hybe that fails further down (out-of-frame or
        # rejected), or that changed roles between runs (e.g. was a
        # regular target hybe with a real residual in a previous run but
        # IS this run's reference_hybe), ends up with exactly what this
        # run computed, never a leftover from a previous call on this
        # same (real, persisted-to-disk) cell object with different
        # params or before the reject bound existed. Re-added below only
        # on success.
        key = (hybe, modality)
        cell.matrices.pop(key, None)
        cell.matrix_provenance.pop(key, None)

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
            cell.matrices[key] = {'yx': np.eye(3), 'zx': np.eye(3)}
            continue

        # H1 (hybe's native frame -> shared frame -> reference_hybe's
        # native frame -> cell.reference_hybe's frame) is pure matrix
        # algebra from fov_matrices -- it never needs any image data, so
        # it's always computable regardless of whether a crop can be
        # built below.
        H1 = compose_chain([fov_matrices[hybe], la.inv(H_ref_to_shared)])

        target_channel = pick_channel_by_type(record, channel_type)
        result = _native_crop(hybe, target_channel)
        if result is None:
            # Cell doesn't overlap this hybe's frame at all -- no crop
            # exists to fit a cell-level residual against. Per the same
            # "no no-alignment" principle as the reject bound below: this
            # is NOT a reason to omit the hybe from cell.matrices (that
            # was this code's own earlier, WRONG behavior) -- H1 above is
            # still fully valid, so fall back to H1-only (H2=identity)
            # exactly like a rejected residual does.
            H_yx = H1
            cell.matrices[key] = {'yx': H_yx, 'zx': np.eye(3)}
            cell.matrix_provenance[key] = {
                'reference_sequence': f'{hybe}(cell {cell.id})->{reference_hybe} '
                                      f'[cell-level residual SKIPPED: cell does not overlap this hybe\'s frame, '
                                      f'fell back to FOV/cross-modal only]',
                'steps': np.stack([H1, np.eye(3)]),
            }
            continue
        target_crop, (cymin, cymax, cxmin, cxmax) = result

        H2_fitted = np.vstack([preprocess.find_translation_via_phase_correlation(target_crop, reference_crop),
                               np.array([0, 0, 1])])
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
        rejected = np.hypot(H2_fitted[0, 2], H2_fitted[1, 2]) > pad
        H2 = np.eye(3) if rejected else H2_fitted
        # H2 outermost (this cell's own residual refinement measured
        # against reference_hybe's own crop) -- H1@H2 lands directly in
        # reference_hybe's own frame, this run's natural resting place;
        # no further push to a shared frame (see this function's own
        # docstring for why that's a consumer-side concern now, via
        # matrix_anchors).
        H_yx = compose_chain([H1, H2])

        H_zx = np.eye(3)
        zx_rejected = False
        secondary_x_shift = 0.0
        z_shift = 0.0
        if including_z:
            # z-depth correction stays fiducial-based regardless of
            # channel_type -- structural (bead/fiducial) depth alignment,
            # not signal-dependent, so the yx channel choice doesn't apply here
            ref_zx = hybe_zx_projection(storage_path, fov, reference_hybe, ref_record['fiducial_channel'],
                                        rymin, rymax, rxmin, rxmax, lb, ub)
            target_zx = hybe_zx_projection(storage_path, fov, hybe, record['fiducial_channel'],
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
            A3 = preprocess.find_translation_via_phase_correlation(target_zx_aligned, ref_zx)
            # Same (width, depth) convention as above: A3[0,2] is this
            # array's axis1 (depth) shift -- the real z-correction.
            # A3[1,2] is axis0 (width) -- a SECOND, independent
            # measurement of the x-shift left over after H2's own x-shift
            # was already applied above, not a depth value. A confidently-
            # reported large leftover x-shift here is the same "locked
            # onto noise, not real content" signal H2's own reject bound
            # guards against -- distrust BOTH this z-correction and the
            # x-refinement together when that happens, rather than
            # keeping one and discarding the other.
            secondary_x_shift = float(A3[1, 2])
            z_shift = float(A3[0, 2])
            zx_rejected = abs(secondary_x_shift) > 3
            if not zx_rejected:
                # This leftover x-shift is a second, independent
                # measurement of the SAME physical x-drift H2 already
                # estimated -- fold it into H_yx as an additive
                # refinement, making this the cell's real, final x-
                # position. H_zx carries the z-correction alone; its own
                # x-component is intentionally zeroed here, not double-
                # applied on top of the x already folded into H_yx.
                H_yx[0, 2] += secondary_x_shift
                H_zx[0, 2] = z_shift

        zx_note = ''
        if including_z:
            if zx_rejected:
                zx_note = (f' [z-alignment REJECTED: secondary x-shift {secondary_x_shift:.1f}px > 3px, '
                          f'z left uncorrected]')
            else:
                zx_note = f' [z-alignment applied: z={z_shift:.1f}px, yx x refined by {secondary_x_shift:.1f}px]'

        cell.matrices[key] = {'yx': H_yx, 'zx': H_zx}
        cell.matrix_provenance[key] = {
            'reference_sequence': f'{hybe}(cell {cell.id})->{reference_hybe}'
                                  + (f' [cell-level residual REJECTED: '
                                     f'{np.hypot(H2_fitted[0, 2], H2_fitted[1, 2]):.1f}px > pad={pad}, '
                                     f'fell back to FOV/cross-modal only]' if rejected else '')
                                  + zx_note,
            'steps': np.stack([H1, H2]),
        }
