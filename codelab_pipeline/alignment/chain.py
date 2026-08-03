import os
from functools import reduce
import numpy as np
import numpy.linalg as la
import ipywidgets as widgets
import time
import h5py

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from ..io import preprocess
import cv2

from IPython.display import display, clear_output
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
          '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
          '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
          '#000075', '#808080', '#000000']

plt.rcParams['font.family'] = 'arial'
plt.rcParams['font.size'] = 14
plt.rcParams['pdf.fonttype'] = 42

fontdict_label = {'fontname': 'arial',
                  'fontsize':16,
                  }
fontdict_ticks = {'fontname': 'arial',
                  'fontsize': 14,
                  }
fontdict_title = {'fontname': 'arial',
                  'fontsize': 24,
                  'fontweight': 'bold',
                  }

red_to_cyan_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['#FF0000', '#FFFFFF', '#00FFFF'], N=256)

def align_cell(yx, H, shape):
    y,x = yx
    height,width = shape
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
    result the user just accepted) into each hybe's own H5 /matrix/{hybe},
    plus reference_sequence/steps provenance attrs -- split out from
    align_same_modality so the write step can be deferred (manual
    review mode) or run standalone.
    """
    fov_path = os.path.join(storage_path, f'FOV{fov:02d}')
    for hybe, H in matrices.items():
        h5path = os.path.join(fov_path, f'{hybe}_stack.h5')
        with h5py.File(h5path, 'r+') as f:
            f['/matrix'][hybe][:] = H
            f['/matrix'][hybe].attrs['reference_sequence'] = np.array([f'{hybe}->{reference_hybe}'], dtype='S')
            f['/matrix'][hybe].attrs['steps'] = H[None, ...].astype('float32')


def read_same_modality_matrices(storage_path, fov, hybe_records):
    """
    Reads back whatever's already on disk in each hybe's own H5
    /matrix/{hybe} (self-keyed dataset -- see write_same_modality_matrices)
    without requiring align_same_modality to be re-run. preprocess.py's
    ingestion always seeds this dataset to identity for every hybe, so in
    the normal case this never needs the identity fallback below -- it's
    only there for a hybe H5 that predates that seeding. This is the
    read-back half of "activation": self.fov_matrices in the GUI should
    reflect whatever alignment has already been computed and written,
    without the user needing to re-run alignment just to see it again.
    """
    fov_path = os.path.join(storage_path, f'FOV{fov:02d}')
    matrices = {}
    for record in hybe_records:
        hybe = record['folder']
        h5path = os.path.join(fov_path, f'{hybe}_stack.h5')
        try:
            with h5py.File(h5path, 'r') as f:
                if f'/matrix/{hybe}' in f:
                    matrices[hybe] = f[f'/matrix/{hybe}'][:]
                else:
                    matrices[hybe] = np.eye(3)
        except OSError:
            matrices[hybe] = np.eye(3)
    return matrices


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
    fov_path = os.path.join(storage_path, f'FOV{fov:02d}')
    record_by_folder = {r['folder']: r for r in hybe_records}
    ref_record = record_by_folder[reference_hybe]

    with h5py.File(os.path.join(fov_path, f'{reference_hybe}_stack.h5'), 'r') as f:
        reference_mip = f[f'/mip/ch{ref_record["fiducial_channel"]}'][:]

    matrices = {}
    for record in hybe_records:
        hybe = record['folder']
        h5path = os.path.join(fov_path, f'{hybe}_stack.h5')
        if hybe == reference_hybe:
            H = np.eye(3)
        else:
            with h5py.File(h5path, 'r') as f:
                moving_mip = f[f'/mip/ch{record["fiducial_channel"]}'][:]
            H = align_readout_to_reference(moving_mip, reference_mip, lb, ub, border_trim=border_trim, max_shift=max_shift)
        matrices[hybe] = H

    if write:
        write_same_modality_matrices(storage_path, fov, matrices, reference_hybe)

    return matrices

def _readout_channel_mip(h5path):
    """The one non-fiducial channel's MIP for a hybe H5 file, read from its own attrs."""
    with h5py.File(h5path, 'r') as f:
        channels = [int(c.decode()) for c in f.attrs['channel_list']]
        fiducial_ch = int(f.attrs['fiducial_channel'])
        readout_ch = [c for c in channels if c != fiducial_ch][0]
        return f[f'/mip/ch{readout_ch}'][:]

def _fiducial_channel_mip(h5path):
    """The fiducial channel's MIP for a hybe H5 file, read from its own attrs."""
    with h5py.File(h5path, 'r') as f:
        fiducial_ch = int(f.attrs['fiducial_channel'])
        return f[f'/mip/ch{fiducial_ch}'][:]

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
    mip_fn = _fiducial_channel_mip if channel_type == 'fiducial' else _readout_channel_mip
    rna_h5path = os.path.join(rna_storage_path, f'FOV{fov:02d}', f'{rna_reference_hybe}_stack.h5')
    dna_h5path = os.path.join(dna_storage_path, f'FOV{fov:02d}', f'{dna_reference_hybe}_stack.h5')
    rna_mip = mip_fn(rna_h5path)
    dna_mip = mip_fn(dna_h5path)

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
                           pad=10, lb=0.3, ub=0.9999, including_z=True):
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

    Writes cell.matrices[hybe] = {'yx': H_within_or_across @ ... composed
    with the cell's own residual, 'zx': depth correction} and
    cell.matrix_provenance[hybe] for traceability, mirroring the FOV-level
    /matrix/{hybe} provenance from Phase 1.
    """
    height, width = cell.frame_shape
    reference_hybe = reference_hybe or cell.reference_hybe
    x_ref, y_ref = cell.area  # always native to cell.reference_hybe's own frame

    def _native_crop(hybe, channel):
        if hybe == cell.reference_hybe:
            x, y = x_ref, y_ref
        else:
            H_to_shared = fov_matrices[hybe]
            cy, cx = align_cell((y_ref, x_ref), la.inv(H_to_shared), (height, width))
            x, y = cx, cy
        if len(x) == 0:
            return None  # cell doesn't overlap this hybe's frame at all
        ymin, ymax = max(0, int(y.min()) - pad), min(height, int(y.max()) + pad + 1)
        xmin, xmax = max(0, int(x.min()) - pad), min(width, int(x.max()) + pad + 1)
        with h5py.File(os.path.join(storage_path, f'FOV{fov:02d}', f'{hybe}_stack.h5'), 'r') as f:
            mip = f[f'/mip/ch{channel}'][:]
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
    # own reference hybe (the previously-only-supported case; fov_matrices
    # always carries a real entry per hybe, see read_same_modality_matrices)
    H_ref_to_shared = fov_matrices.get(reference_hybe, np.eye(3))

    for record in hybe_records:
        hybe = record['folder']
        if hybe == reference_hybe:
            # reference_hybe's own transform back into cell.reference_hybe's
            # frame (everything in cell.matrices is always expressed
            # relative to cell.reference_hybe -- get_area_in_readout/get_mip
            # depend on that unconditionally) -- identity only in the
            # original, common case where reference_hybe IS cell.reference_hybe.
            cell.matrices[hybe] = {'yx': H_ref_to_shared, 'zx': np.eye(3)}
            continue

        target_channel = pick_channel_by_type(record, channel_type)
        result = _native_crop(hybe, target_channel)
        if result is None:
            continue
        target_crop, (cymin, cymax, cxmin, cxmax) = result

        H2 = np.vstack([preprocess.find_translation_via_phase_correlation(target_crop, reference_crop),
                        np.array([0, 0, 1])])
        # hybe's native frame -> shared frame -> reference_hybe's native
        # frame (H_ref_to_shared is identity in the original, common case,
        # so this reduces to plain fov_matrices[hybe] exactly as before),
        # H2 outermost (this cell's own residual refinement measured
        # against reference_hybe's own crop) -- then back OUT to
        # cell.reference_hybe's frame via H_ref_to_shared once more, since
        # cell.matrices must stay expressed relative to cell.reference_hybe
        # regardless of which hybe this refinement anchored to.
        H1 = compose_chain([fov_matrices[hybe], la.inv(H_ref_to_shared)])
        H_yx = compose_chain([H1, H2, H_ref_to_shared])

        H_zx = np.eye(3)
        if including_z:
            # z-depth correction stays fiducial-based regardless of
            # channel_type -- structural (bead/fiducial) depth alignment,
            # not signal-dependent, so the yx channel choice doesn't apply here
            ref_zx = hybe_zx_projection(storage_path, fov, reference_hybe, ref_record['fiducial_channel'],
                                        rymin, rymax, rxmin, rxmax, lb, ub)
            target_zx = hybe_zx_projection(storage_path, fov, hybe, record['fiducial_channel'],
                                           cymin, cymax, cxmin, cxmax, lb, ub)
            # apply the just-computed yx residual to the target's projection so it's
            # in the same orientation as the reference's before comparing depth
            target_zx_aligned = cv2.warpAffine(target_zx, H2[:2], (target_zx.shape[1], target_zx.shape[0]))
            A3 = preprocess.find_translation_via_phase_correlation(target_zx_aligned, ref_zx)
            H_zx = np.vstack([A3[:2], np.array([0, 0, 1])])

        cell.matrices[hybe] = {'yx': H_yx, 'zx': H_zx}
        cell.matrix_provenance[hybe] = {
            'reference_sequence': f'{hybe}(cell {cell.id})->{reference_hybe}',
            'steps': np.stack([H1, H2]),
        }

def crop_or_pad_to_shape(img, target_shape, pad_value=0):
    h, w = img.shape[:2]
    H, W = target_shape
    dh, dw = H - h, W - w

    # Compute crop or pad for height
    if dh < 0:
        top = (-dh) // 2
        bottom = top + H
        img = img[top:bottom, :]
    else:
        top_pad = dh // 2
        bottom_pad = dh - top_pad
        img = np.pad(img, ((top_pad, bottom_pad), (0, 0)), constant_values=pad_value)

    # Compute crop or pad for width
    if dw < 0:
        left = (-dw) // 2
        right = left + W
        img = img[:, left:right]
    else:
        left_pad = dw // 2
        right_pad = dw - left_pad
        img = np.pad(img, ((0, 0), (left_pad, right_pad)), constant_values=pad_value)

    return img

class AlignmentWidget(object):
    """
    A widget for aligning multispot microscopy images using Cellpose segmentation.
    """
    def __init__(self, reference_hybe, h5_save_path, filename='vlinks.h5'):
        self.h5_save_path = h5_save_path
        self.filename = filename
        with h5py.File(os.path.join(h5_save_path,self.filename), 'r') as f:
            self.fov_list = f.attrs['fov_list'][:].tolist()
            self.channels_list = f.attrs['channels_list'][:].tolist()
            self.hybe_list = [str(h.decode()) for h in f.attrs['hybe_list']]

        self.total_channel_list = np.unique(np.ravel(self.channels_list)).tolist()
        self.reference_hybe = reference_hybe


        self.create_widgets()

    def generate_colormap(self, sub_hybe_list):
        colormaps = {
            hybe: mcolors.LinearSegmentedColormap.from_list(
                f'{hybe}',
                [(0, 0, 0, 1), red_to_cyan_cmap(int(hid / len(sub_hybe_list) * 256))],
                256,
                gamma=1.8
            )
            for hid, hybe in enumerate(sub_hybe_list)
        }
        return colormaps

    def create_widgets(self):
        self.widgets = {}
        self.widgets['including_z'] = widgets.Checkbox(value=True, description='Including Z:',)
        self.widgets['reference_hybe'] = widgets.Dropdown(value=self.reference_hybe, options=self.hybe_list, description='Reference Hybe:',)
        self.widgets['sub_hybe_list'] = widgets.SelectMultiple(options=self.hybe_list, value=tuple(self.hybe_list), description='Hybes to Align:',)

        self.widgets['reference_channel'] = widgets.Dropdown(value=self.total_channel_list[0], options=self.total_channel_list, description='reference Channel:',)
        self.widgets['fov_list'] = widgets.BoundedIntText(value=self.fov_list[0], min=1, max=max(self.fov_list)+1, step=1, description='FOV:',)
        self.widgets['pad'] = widgets.BoundedIntText(value=0, min=0, max=100, step=5, description='Pad:',)
        self.widgets['id_spinbox'] = widgets.BoundedIntText(value=1, min=1, max=10000, step=1, description='Cell ID:',)
        self.widgets['lb'] = widgets.BoundedFloatText(value=85, min=0, max=90, step=0.01, description='Lower Bound:',)
        self.widgets['ub'] = widgets.BoundedFloatText(value=99.99, min=90, max=100, step=0.01, description='Upper Bound:',)
        
        self.widgets['align_button'] = widgets.Button(description='Align', button_style='success',)
        self.widgets['save_and_pass_button'] = widgets.Button(description='Save and Pass', button_style='success',)
        self.widgets['discard_button'] = widgets.Button(description='Discard', button_style='danger',)
        self.widgets['spinbox_align_buttons'] = widgets.HBox([self.widgets['id_spinbox'], self.widgets['align_button']])
        
        self.widgets['loading_label'] = widgets.Label(value='Push "Align" to start aligning cell... please wait.',)
        
        self.output = widgets.Output()

        self.mips_by_hybe = {hybe:None for hybe in self.hybe_list}

        self.figure = None
        self.axes = None
        self.current_image_parameters = {'lb':0, 'ub':0, 'fov':0, 'reference_channel':0}

        self.ui = widgets.VBox([
            widgets.Text(value=self.h5_save_path, description='Save Directory:',),
            self.widgets['including_z'],
            self.widgets['reference_hybe'],
            self.widgets['sub_hybe_list'],
            self.widgets['fov_list'],
            self.widgets['reference_channel'],
            self.widgets['pad'],
            self.widgets['lb'],
            self.widgets['ub'],
            self.widgets['spinbox_align_buttons'],
            widgets.HBox([self.widgets['save_and_pass_button'], self.widgets['discard_button']]),
            self.widgets['loading_label'],
            self.output
        ])

        self.widgets['save_and_pass_button'].on_click(lambda b: self.run_save_and_pass())
        self.widgets['discard_button'].on_click(lambda b: self.run_discard())
        self.widgets['align_button'].on_click(lambda b: self.run_cell_based_align())

    def show(self):
        display(self.ui)

    def run_cell_based_align(self,):
        with self.output:
            clear_output(wait=True)
            self.widgets['loading_label'].value = "🔄 Aligning cell... please wait."
            start = time.time()
            try:
                lb = self.widgets['lb'].value/100
                ub = self.widgets['ub'].value/100
                
                fov = self.widgets['fov_list'].value
                reference_channel = self.widgets['reference_channel'].value
                id = self.widgets['id_spinbox'].value

                cell_align_dir = os.path.join(self.h5_save_path, f'FOV{fov:02d}/Aligned_Cells')
                os.makedirs(cell_align_dir, exist_ok=True)
                os.makedirs(os.path.join(cell_align_dir, 'Pass'), exist_ok=True)
                os.makedirs(os.path.join(cell_align_dir, 'Discard'), exist_ok=True)
                
                t0 = time.time()
                with h5py.File(os.path.join(self.h5_save_path, self.filename), 'r') as f:
                    mask = f[f'FOV{fov:02d}/cells/mask'][:]
                    id_list = np.unique(mask[mask > 0]).tolist()
                    if (id == 1) and (id not in id_list):
                        id = id_list[0]
                        self.widgets['id_spinbox'].value = id
                    elif id not in id_list:
                        id = np.array(id_list)[np.array(id_list) > id][0]
                        self.widgets['loading_label'].value = f'⚠️ Cell ID {self.widgets["id_spinbox"].value} not found in FOV {fov}. Loading next available ID {id}.'
                        self.widgets['id_spinbox'].value = id
                    condition = f[f'FOV{fov:02d}/cells/cellbarcodes'][id_list.index(id)].decode('utf-8')

                    matrices_fov = np.zeros((len(self.hybe_list),3,3))
                    for hid,hybe in enumerate(self.hybe_list): matrices_fov[hid] = f[f'FOV{fov:02d}/matrix/{hybe}'][:]
                        
                    if (fov != self.current_image_parameters['fov']) or (reference_channel != self.current_image_parameters['reference_channel']):  
                        self.current_image_parameters['lb'] = lb
                        self.current_image_parameters['ub'] = ub
                        self.current_image_parameters['fov'] = fov
                        self.current_image_parameters['reference_channel'] = reference_channel
                        
                        for hid,hybe in enumerate(self.hybe_list): self.mips_by_hybe[hybe] = preprocess.normalize_to_uint8(f[f'FOV{fov:02d}/mip/ch{reference_channel}'][hid,...],lb,ub)
                    print(f'Loading time: {time.time()-t0:.2f} seconds')
                        
                if id == 1 and id not in id_list:
                    id = id_list[0]
                    self.widgets['id_spinbox'].value = id
                if np.sum(mask == id) == 0:
                    self.widgets['loading_label'].value = f'❌ No cells found with ID {id} in FOV {fov}.'
                    return
                self.cell_based_align(id, condition, mask, matrices_fov, )
            finally:
                self.widgets['loading_label'].value = f"✅ Cell aligned successfully. {time.time()-start:.2f} seconds."
                display(self.figure)

    def cell_based_align(self, id, condition, mask, matrices_fov, ):
        # parameters
        lb,ub = self.current_image_parameters['lb'], self.current_image_parameters['ub']
        pad = self.widgets['pad'].value
        including_z = self.widgets['including_z'].value
        reference_hybe = str(self.widgets['reference_hybe'].value)
        id_list = np.unique(mask[mask > 0])
        id_of_mask = np.where(id_list == id)[0][0]
        height,width = mask.shape
        reference_channel = self.widgets['reference_channel'].value
        fov = self.widgets['fov_list'].value
        sub_hybe_list = list(self.widgets['sub_hybe_list'].value)
        colormaps = self.generate_colormap(sub_hybe_list)

        H1 = matrices_fov[self.hybe_list.index(reference_hybe)]
        y,x = np.where(mask == id)
        ry,rx = align_cell((y,x), la.inv(H1), (height,width))
        rymin,rymax,rxmin,rxmax = max(0,ry.min()-pad), min(height,ry.max()+pad+1), max(0,rx.min()-pad), min(width,rx.max()+pad+1)
        reference_image_norm = self.mips_by_hybe[reference_hybe]
        reference_image_crop = reference_image_norm[rymin:rymax,rxmin:rxmax]

        ycomp1,ycomp2,ycomp3 = [np.zeros((rymax-rymin,rxmax-rxmin,4,len(sub_hybe_list)),dtype=float) for _ in range(3)]

        with h5py.File(os.path.join(self.h5_save_path,self.filename ), 'r') as f:
            matrices_yx = f[f'FOV{fov:02d}/cells/matrix/yx'][id_of_mask,:]
            matrices_zx = f[f'FOV{fov:02d}/cells/matrix/zx'][id_of_mask,:]

        for shid,hybe in enumerate(sub_hybe_list):
            hid = self.hybe_list.index(hybe)
            cmap = colormaps[hybe]
            if hybe != reference_hybe:
                target_image_norm = self.mips_by_hybe[hybe]
                H1 = matrices_fov[hid]
                cy,cx = align_cell((ry,rx), la.inv(H1), (height,width))
                cymin,cymax,cxmin,cxmax = max(0,cy.min()-pad), min(height,cy.max()+pad+1), max(0,cx.min()-pad), min(width,cx.max()+pad+1)
                target_image_crop = target_image_norm[rymin:rymax,rxmin:rxmax]
                target_image_crop_first = target_image_norm[cymin:cymax,cxmin:cxmax]

                H2 = np.vstack([preprocess.find_translation_via_phase_correlation(target_image_crop_first,
                                                                                  reference_image_crop),
                                                                                  np.array([0,0,1])])
                matrices_yx[hid] = H2
                target_image_crop_final = cv2.warpAffine(target_image_crop_first, H2[:2], (cxmax-cxmin,cymax-cymin), )

                target_image_crop_ = target_image_crop.astype(float) / target_image_crop.max() * 255
                target_image_crop_first_ = target_image_crop_first.astype(float) / target_image_crop_first.max() * 255
                target_image_crop_final_ = target_image_crop_final.astype(float) / target_image_crop_final.max() * 255
                ycomp1[...,shid] = cmap(target_image_crop_.astype(np.uint8))
                ycomp2[...,shid] = cmap(crop_or_pad_to_shape(target_image_crop_first_.astype(np.uint8),(rymax-rymin,rxmax-rxmin)))
                ycomp3[...,shid] = cmap(crop_or_pad_to_shape(target_image_crop_final_.astype(np.uint8),(rymax-rymin,rxmax-rxmin)))
            else:
                reference_image_crop_ = reference_image_crop.astype(float) / reference_image_crop.max() * 255
                ycomp1[...,shid] = cmap(reference_image_crop_.astype(np.uint8))
                ycomp2[...,shid] = ycomp1[...,shid]
                ycomp3[...,shid] = ycomp1[...,shid]
            
        if including_z:
            with h5py.File(os.path.join(self.h5_save_path,self.filename), 'r') as f:
                depth = f.attrs['stack_shape'][0]
                hid = self.hybe_list.index(reference_hybe)
                stacks = f[f'FOV{fov:02d}/stack/ch{reference_channel}'][hid,:,rymin:rymax,rxmin:rxmax].squeeze()
                zcomp1,zcomp2,zcomp3 = [np.zeros((depth,rxmax-rxmin,4,len(sub_hybe_list)),dtype=float) for _ in range(3)]
                stack_ref_zx = preprocess.normalize_to_uint8(stacks.max(1),lb,ub)
                
                for shid,hybe in enumerate(sub_hybe_list):
                    hid = self.hybe_list.index(hybe)
                    cmap = colormaps[hybe]
                    
                    if hybe != reference_hybe:
                        stacks = f[f'FOV{fov:02d}/stack/ch{reference_channel}'][hid,:,rymin:rymax,rxmin:rxmax].squeeze()
                        stack_hyb_zx = preprocess.normalize_to_uint8(stacks.max(1),lb,ub)
                        H1 = matrices_fov[hid]
                        cy,cx = align_cell((ry,rx), la.inv(H1), (height,width))
                        cymin,cymax,cxmin,cxmax = max(0,cy.min()-pad), min(height,cy.max()+pad+1), max(0,cx.min()-pad), min(width,cx.max()+pad+1)

                        stacks = f[f'FOV{fov:02d}/stack/ch{reference_channel}'][hid,:,cymin:cymax,cxmin:cxmax].squeeze()
                        stack_fov_zx = preprocess.normalize_to_uint8(stacks.max(1),lb,ub)

                        H2 = matrices_yx[hid]
                        stack_mip_zx = preprocess.normalize_to_uint8(np.concatenate([cv2.warpAffine(stacks[j], H2[:2],
                                                                                                    (cxmax-cxmin,cymax-cymin),)[None,...]
                                                                                                      for j in range(depth)], axis=0).max(1),lb,ub)

                        A3 = preprocess.find_translation_via_phase_correlation(stack_mip_zx, stack_ref_zx)
                        zcomp1[...,shid] = cmap(stack_hyb_zx)
                        zcomp2[...,shid] = cmap(crop_or_pad_to_shape(stack_fov_zx, (depth,rxmax-rxmin)))
                        zcomp3[...,shid] = cmap(crop_or_pad_to_shape(cv2.warpAffine(stack_mip_zx, A3, (cxmax-cxmin,depth)),(depth,rxmax-rxmin)))
                        H3 = np.vstack([A3[:2],np.array([0,0,1])])
                        matrices_zx[hid] = H3
                    else:
                        zcomp1[...,shid] = cmap(stack_ref_zx)
                        zcomp2[...,shid] = zcomp1[...,shid]
                        zcomp3[...,shid] = zcomp1[...,shid]
                
        with h5py.File(os.path.join(self.h5_save_path,self.filename), 'r+') as f:
            f[f'/FOV{fov:02d}/cells/matrix/yx'][id_of_mask,:] = matrices_yx
            f[f'/FOV{fov:02d}/cells/matrix/zx'][id_of_mask,:] = matrices_zx

        if not including_z:
            fig,ax = plt.subplots(1,3,figsize=(18,6))
            ax[0].imshow(ycomp1.max(-1),)
            ax[1].imshow(ycomp2.max(-1),)
            ax[2].imshow(ycomp3.max(-1),)
            cell = np.zeros((rymax-rymin,rxmax-rxmin),dtype=np.uint8)
            cell[ry-rymin,rx-rxmin] = 1
            boundaries = (cell).astype(np.uint8) - cv2.erode((cell).astype(np.uint8), np.ones((3,3),np.uint8), iterations=1)
            y,x = np.where(boundaries > 0)
            ax[0].scatter(x,y, color='yellow', s=10, marker='s', alpha=.5)
            ax[1].scatter(x,y, color='yellow', s=10, marker='s', alpha=.5)
            ax[2].scatter(x,y, color='yellow', s=10, marker='s', alpha=.5)
            ax[0].axis('off')
            ax[1].axis('off')
            ax[2].axis('off')
            ax[0].set_title(f'Original Image\nRED: {sub_hybe_list[0]} CYAN: {sub_hybe_list[-1]}\nFOV: {fov} Cell ID: {id} Barcode {condition}', fontdict=fontdict_label)
            ax[1].set_title(f'FOV Aligned Image\nRED: {sub_hybe_list[0]} CYAN: {sub_hybe_list[-1]}\nFOV: {fov} Cell ID: {id} Barcode {condition}', fontdict=fontdict_label)
            ax[2].set_title(f'Final Image\nRED: {sub_hybe_list[0]} CYAN: {sub_hybe_list[-1]}\nFOV: {fov} Cell ID: {id} Barcode {condition}', fontdict=fontdict_label)
        else:
            fig,ax = plt.subplots(2,3,figsize=(18,12))
            ax[0,0].imshow(ycomp1.max(-1),aspect='auto')
            ax[0,1].imshow(ycomp2.max(-1),aspect='auto')
            ax[0,2].imshow(ycomp3.max(-1),aspect='auto')
            cell = np.zeros((rymax-rymin,rxmax-rxmin),dtype=np.uint8)
            cell[ry-rymin,rx-rxmin] = 1
            boundaries = (cell).astype(np.uint8) - cv2.erode((cell).astype(np.uint8), np.ones((3,3),np.uint8), iterations=1)
            y,x = np.where(boundaries > 0)
            ax[0,0].scatter(x,y, color='yellow', s=10, marker='s', alpha=.5)
            ax[0,1].scatter(x,y, color='yellow', s=10, marker='s', alpha=.5)
            ax[0,2].scatter(x,y, color='yellow', s=10, marker='s', alpha=.5)
            ax[0,0].set_title(f'Original Image\nRED: {sub_hybe_list[0]} CYAN: {sub_hybe_list[-1]}\nFOV: {fov} Cell ID: {id} Barcode {condition}', fontdict=fontdict_label)
            ax[0,1].set_title(f'FOV Aligned Image\nRED: {sub_hybe_list[0]} CYAN: {sub_hybe_list[-1]}\nFOV: {fov} Cell ID: {id} Barcode {condition}', fontdict=fontdict_label)
            ax[0,2].set_title(f'Final Image\nRED: {sub_hybe_list[0]} CYAN: {sub_hybe_list[-1]}\nFOV: {fov} Cell ID: {id} Barcode {condition}', fontdict=fontdict_label)
            ax[1,0].imshow(zcomp1.max(-1),aspect='auto')
            ax[1,1].imshow(zcomp2.max(-1),aspect='auto')
            ax[1,2].imshow(zcomp3.max(-1),aspect='auto')
        
            ax[0,0].set_xticks([])
            ax[0,1].set_xticks([])
            ax[0,2].set_xticks([])
            ax[0,0].set_yticks([])
            ax[0,1].set_yticks([])
            ax[0,2].set_yticks([])
            ax[1,0].set_xticks([])
            ax[1,1].set_xticks([])
            ax[1,2].set_xticks([])
            ax[1,0].set_yticks([])
            ax[1,1].set_yticks([])
            ax[1,2].set_yticks([])
            ax[0,0].set_ylabel('Y',rotation=0,ha='right',**fontdict_label)
            ax[0,1].set_ylabel('Y',rotation=0,ha='right',**fontdict_label)
            ax[0,2].set_ylabel('Y',rotation=0,ha='right',**fontdict_label)
            ax[1,0].set_ylabel('Z',rotation=0,ha='right',**fontdict_label)
            ax[1,1].set_ylabel('Z',rotation=0,ha='right',**fontdict_label)
            ax[1,2].set_ylabel('Z',rotation=0,ha='right',**fontdict_label)
            ax[1,0].set_xlabel('X',**fontdict_label)
            ax[1,1].set_xlabel('X',**fontdict_label)
            ax[1,2].set_xlabel('X',**fontdict_label)

        plt.tight_layout()
        self.figure = fig
        self.axes = ax
        plt.close(fig)

    def run_save_and_pass(self):
        with self.output:
            clear_output(wait=True)
            start = time.time()
            fov = self.widgets['fov_list'].value
            id = self.widgets['id_spinbox'].value
            cell_align_dir = os.path.join(self.h5_save_path, f'FOV{fov:02d}/Aligned_Cells')

            with h5py.File(os.path.join(self.h5_save_path,self.filename), 'r+') as f:
                mask = f[f'FOV{fov:02d}/cells/mask'][:]
                id_list = np.unique(mask)[1:].tolist()
                if id not in f[f'FOV{fov:02d}/cells'].attrs['good'] and id in id_list:
                    f[f'FOV{fov:02d}/cells'].attrs['good'] = np.unique(np.concatenate((f[f'FOV{fov:02d}/cells'].attrs['good'], [id])))
                
                self.figure.savefig(os.path.join(cell_align_dir,'Pass',
                                                 f'aligned_reference_{self.widgets["reference_hybe"].value}_FOV{fov:02d}_ID{id}.png'))
            if id_list.index(id)+1 < len(id_list):
                new_id = id_list[id_list.index(id)+1]
                self.widgets['id_spinbox'].value = new_id
                self.run_cell_based_align()
            else:
                self.widgets['loading_label'].value = f'✅ All cells aligned in FOV {fov}. {time.time()-start:.2f} seconds.'
                return
            
    def run_discard(self):
        with self.output:
            clear_output(wait=True)
            start = time.time()
            fov = self.widgets['fov_list'].value
            id = self.widgets['id_spinbox'].value
            cell_align_dir = os.path.join(self.h5_save_path, f'FOV{fov:02d}/Aligned_Cells')

            with h5py.File(os.path.join(self.h5_save_path,self.filename), 'r+') as f:
                mask = f[f'FOV{fov:02d}/cells/mask'][:]
                id_list = np.unique(mask)[1:].tolist()
                if id in id_list:
                    f[f'FOV{fov:02d}/cells'].attrs['good'] = np.unique(np.array([i for i in f[f'FOV{fov:02d}/cells'].attrs['good'] if i != id]))
                self.figure.savefig(os.path.join(cell_align_dir,'Discard',
                                                 f'aligned_reference_{self.widgets["reference_hybe"].value}_FOV{fov:02d}_ID{id}.png'))
            if id_list.index(id)+1 < len(id_list):
                new_id = id_list[id_list.index(id)+1]
                self.widgets['id_spinbox'].value = new_id
                self.run_cell_based_align()
            else:
                print(f'No more cells to align in FOV {fov}.')
                return
