import numpy as np
import cv2

from codelab_pipeline.io import preprocess
from codelab_pipeline.io import vlinks_store
from codelab_pipeline.alignment import chain as alignment

def _sequential_color(i, n):
    """
    Linear interpolation from red (1,0,0) at i=0 to cyan (0,1,1) at i=n-1 --
    a categorical palette (distinct hues per readout) reads as arbitrary and
    isn't intuitive; a sequential red-cyan gradient keeps the same color
    language as the existing pairwise preview (red=reference, cyan=moving),
    and since before_images/after_images always insert the reference hybe
    first, it lands at i=0 (pure red) automatically.
    """
    t = i / max(n - 1, 1)
    return (1 - t, t, t)


def _composite_multi(images, lb=0.3, ub=0.9999):
    """
    images: {label: 2D array}. Normalizes each (NaN-aware: percentiles and
    the final normalized value are computed only over each image's own
    real, non-NaN pixels -- normalize_to_uint8's plain uint8 cast turns a
    NaN into undefined garbage otherwise), maps it through a sequential
    red-cyan color (reference hybe = red, last readout = cyan, everything
    else interpolated between), pixelwise-max across all of them -- same
    composites.max(-1) pattern already used in
    legacy/segment_widgets.py's CellbarcodeWidget.
    Gives one combined image showing every readout simultaneously, instead
    of the 2-color pairwise composite -- the pairwise view can't show
    whether alignment succeeded overall across many hybes at once.
    A pixel where NO hybe has real data (every one NaN there -- true-
    extent crops of different true sizes, see _true_bounds) renders as
    flat neutral gray rather than black, so "nothing here" reads as
    distinct from "real signal, just dark".
    """
    labels = list(images.keys())
    shape = next(iter(images.values())).shape
    composite = np.zeros((*shape, 3), dtype=float)
    valid_any = np.zeros(shape, dtype=bool)
    n = len(labels)
    for i, label in enumerate(labels):
        color = _sequential_color(i, n)
        img = images[label]
        valid = np.isfinite(img)
        norm = np.zeros(shape, dtype=float)
        if valid.any():
            norm[valid] = preprocess.normalize_to_uint8(img[valid], lb, ub).astype(float)
        norm /= 255
        layer = np.stack([norm * c for c in color], axis=-1)
        composite = np.maximum(composite, layer)
        valid_any |= valid
    composite[~valid_any] = 90 / 255
    return composite


def _true_bounds(H_eff, x, y, pad):
    """
    Unclamped mask bounding box (+ pad margin) in H_eff's target frame --
    NOT run through align_cell's own drop-out-of-frame-points step, so a
    cell near a real frame edge gets its TRUE extent (possibly negative
    ymin/xmin, or ymax/xmax beyond the real frame) instead of a silently
    truncated one.

    Why this matters: two hybes' own crops, built from their own
    independently-fitted H_eff, can end up different sizes purely because
    one hybe's own warped mask sits closer to the real frame edge than
    the other's -- clamping each independently BEFORE comparing them (the
    old approach) baked that disagreement into an arbitrary crop-window
    offset with nothing to do with real image content. That's what made a
    real, already-fitted cell-level correction ("cyan should move up by
    1px") invisible for any cell whose mask reaches a real frame edge:
    the correction shifted the true window, but clamping silently
    absorbed the shift into the window's size instead of its position.
    Keeping the window at its true (possibly out-of-frame) extent and
    NaN-filling the missing part (see _nan_native_crop) instead preserves
    the shift as an actual, visible change in where the real data sits
    within the crop.
    """
    cx, cy = (H_eff[:2] @ np.array([x, y, np.ones_like(x)])).astype(int)
    ymin, ymax = int(cy.min()) - pad, int(cy.max()) + pad + 1
    xmin, xmax = int(cx.min()) - pad, int(cx.max()) + pad + 1
    return ymin, ymax, xmin, xmax


def _nan_native_crop(mip, bounds, height, width):
    """
    mip sliced to `bounds` (ymin,ymax,xmin,xmax from _true_bounds, may
    extend past the real [0,height)x[0,width) frame on any side) -- NaN
    fills whatever part of that window has no real pixel data, real
    pixel data everywhere else. Two hybes' own crops, each built this way
    from their own (possibly differently-clamped) true bounds, are then
    directly comparable position-for-position: overlaying them at a
    shared top-left origin (see _draw_three_way's own composite)
    reproduces exactly the correspondence the alignment fit itself
    assumes (target_crop[i,j] <-> reference_crop[i,j]), whether or not
    either side got real-frame-edge-clipped -- NOT an absolute-frame-
    position alignment, which would reintroduce the two hybes' own
    (real, but visualization-irrelevant) FOV-level offset as a spurious
    extra shift in the overlay.
    """
    ymin, ymax, xmin, xmax = bounds
    out = np.full((ymax - ymin, xmax - xmin), np.nan, dtype=np.float32)
    ry0, ry1 = max(ymin, 0), min(ymax, height)
    rx0, rx1 = max(xmin, 0), min(xmax, width)
    if ry1 > ry0 and rx1 > rx0:
        out[ry0 - ymin:ry1 - ymin, rx0 - xmin:rx1 - xmin] = mip[ry0:ry1, rx0:rx1]
    return out


def _nan_mask_crop(cy, cx, bounds, height, width):
    """
    Same true-extent placement as _nan_native_crop, for the boolean
    cell-mask overlay. cy/cx are align_cell's own already-frame-clamped
    mask points (a mask point that doesn't exist in the real frame has
    no meaningful position to plot), so this never needs NaN -- just
    False outside both the mask and the real frame.
    """
    ymin, ymax, xmin, xmax = bounds
    mask_full = np.zeros((height, width), dtype=bool)
    mask_full[cy, cx] = True
    out = np.zeros((ymax - ymin, xmax - xmin), dtype=bool)
    ry0, ry1 = max(ymin, 0), min(ymax, height)
    rx0, rx1 = max(xmin, 0), min(xmax, width)
    if ry1 > ry0 and rx1 > rx0:
        out[ry0 - ymin:ry1 - ymin, rx0 - xmin:rx1 - xmin] = mask_full[ry0:ry1, rx0:rx1]
    return out


def _nan_zx_crop(storage_path, fov, hybe, channel, bounds, height, width, lb, ub):
    """
    ZX (depth) projection at the SAME true x-extent as the YX row's own
    crop from the same `bounds`, so the two rows stay column-aligned
    (see _draw_three_way's sharex='col'). Only x needs true-extent/NaN
    handling here -- y disappears entirely in hybe_zx_projection's own
    max-projection, so it only needs clamping to stay a valid slice,
    never NaN-tracking; y and x are both clamped before the read since
    a negative index would otherwise silently wrap (h5py/numpy's own
    negative-index convention) instead of raising.
    """
    ymin, ymax, xmin, xmax = bounds
    ry0, ry1 = max(ymin, 0), min(ymax, height)
    rx0, rx1 = max(xmin, 0), min(xmax, width)
    if ry1 <= ry0 or rx1 <= rx0:
        return None
    projection = alignment.hybe_zx_projection(storage_path, fov, hybe, channel,
                                              ry0, ry1, rx0, rx1, lb, ub, normalize=False)
    out = np.full((xmax - xmin, projection.shape[1]), np.nan, dtype=np.float32)
    out[rx0 - xmin:rx1 - xmin, :] = projection
    return out


class PipelineCanvas():
    """
    Visualization + orchestration for the GUI, growing across milestones
    (this pass: alignment before/after previews; later: segmentation,
    localization previews). Draws into an already-created
    FigureCanvasQTAgg (from AlignmentPanelUI.PreviewCanvas) rather than
    creating new figures each time, so redraws don't leak figures.
    """
    def __init__(self, preview_canvas):
        self.preview_canvas = preview_canvas

    def _draw_before_after(self, reference_img, before_img, after_img, title, lb=0.3, ub=0.9999, save_path=None):
        """
        Shared red/cyan composite plotting for both draw_alignment_preview
        (full-MIP) and draw_cell_alignment_preview (cell-crop) -- keeps
        subplot titles short and fixed ('before'/'after alignment') with the
        dynamic, potentially-long title text on one figure-wide suptitle
        instead of duplicated into each half-width subplot title, which was
        overflowing each subplot's title into the other's on a narrow panel.
        """
        def norm(a):
            return preprocess.normalize_to_uint8(a, lb, ub)

        def composite(a, b):
            c = np.zeros((*a.shape, 3), dtype=float)
            c[..., 0] = a
            c[..., 1] = b
            c[..., 2] = b
            return c / 255

        ref_n = norm(reference_img)
        fig = self.preview_canvas.figure
        fig.clear()
        ax = fig.subplots(1, 2)
        ax[0].imshow(composite(ref_n, norm(before_img)))
        ax[0].set_title('before', fontsize=10)
        ax[0].axis('off')
        ax[1].imshow(composite(ref_n, norm(after_img)))
        ax[1].set_title('after alignment', fontsize=10)
        ax[1].axis('off')
        fig.suptitle(f'{title} (red=ref, cyan=moving)', fontsize=10, wrap=True)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        self.preview_canvas.draw()
        if save_path:
            fig.savefig(save_path, dpi=150)

    def draw_alignment_preview(self, reference_mip, moving_mip, H, title_prefix, lb=0.3, ub=0.9999, save_path=None):
        """
        Red/cyan composite overlay, before (raw moving_mip) and after
        (moving_mip warped by H) -- same visualization already validated
        throughout this session's scripted Phase 1/2 verification.
        """
        h, w = reference_mip.shape
        moving_aligned = cv2.warpAffine(moving_mip.astype(np.float32), H[:2], (w, h))
        self._draw_before_after(reference_mip, moving_mip, moving_aligned, title_prefix, lb, ub, save_path)

    def _draw_three_way(self, rows, title, lb=0.3, ub=0.9999, save_path=None):
        """
        Shared multi-row, 3-panel red/cyan composite for
        draw_cell_alignment_preview_3col.

        rows: [(row_label, reference_panels, labeled_images), ...] -- one
        row per alignment plane (YX, ZX) sharing one figure/pop-up, per
        explicit request to see the ZX plane alongside YX rather than as
        a separate preview.
        reference_panels: [(image, mask), ...] -- one PER COLUMN, matching
        labeled_images' own length/order, not one shared across every
        column. Each of the 3 correction stages needs its own reference
        crop: raw must be H=identity (cell.area's coordinates taken
        literally, matching the target's own raw column, so a same-hybe
        reference-vs-target comparison is trivially a perfect overlay);
        FOV/cross-modal and final both reuse the SAME FOV-only crop
        (reference never gets a cell-level residual of its own -- only
        the target side does). mask is a boolean array (same shape as
        image) marking where the cell's own mask lands in the reference
        crop's frame, or None to skip drawing a boundary (the ZX-plane
        row has no such mask -- it's a depth projection, not a spatial
        arrangement the mask lands in).
        labeled_images: [(label, comparison_img), ...] -- each panel pairs
        reference_panel's image against its own comparison_img, with the
        SAME reference boundary contour drawn on every panel in that row
        (a fixed visual anchor -- per explicit request, "specify the cell
        boundary of the reference hybe"). Unlike _draw_before_after, the
        images here are independent, natively-cropped hybe images (not
        both warped into one common frame), so they aren't guaranteed to
        be the same shape -- pad each pair (and the mask) to their shared
        max shape before compositing, purely for display.
        """
        def pad_to_nan(a, h, w):
            a2 = np.full((h, w), np.nan, dtype=np.float32)
            a2[:a.shape[0], :a.shape[1]] = a
            return a2

        def pad_to_zero(a, h, w):
            a2 = np.zeros((h, w), dtype=a.dtype)
            a2[:a.shape[0], :a.shape[1]] = a
            return a2

        def composite(a, b):
            # a/b are true-extent crops (see _true_bounds/_nan_native_crop)
            # -- NaN wherever that hybe's own true window has no real
            # pixel data. A composited pixel is only "real" if BOTH sides
            # have data there; otherwise it renders as one flat neutral
            # marker, never a blend against a fake fallback value in the
            # missing channel (that blend previously created false-
            # colored lines exactly at each hybe's own frame-edge
            # boundary, misreadable as real shifted content).
            h, w = max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])
            pa, pb = pad_to_nan(a, h, w), pad_to_nan(b, h, w)
            both_valid = np.isfinite(pa) & np.isfinite(pb)
            def norm(x):
                out = np.zeros((h, w), dtype=float)
                if both_valid.any():
                    out[both_valid] = preprocess.normalize_to_uint8(x[both_valid], lb, ub)
                return out
            c = np.full((h, w, 3), 90, dtype=float)
            na, nb = norm(pa), norm(pb)
            c[both_valid, 0] = na[both_valid]
            c[both_valid, 1] = nb[both_valid]
            c[both_valid, 2] = nb[both_valid]
            return c / 255, h, w

        ncols = max(len(labeled_images) for _, _, labeled_images in rows)
        fig = self.preview_canvas.figure
        fig.clear()
        # sharex='col' locks each column's rows (e.g. a YX panel and the
        # ZX panel directly below it) to the SAME horizontal data extent
        # -- both are built from the same xmin:xmax crop window, so this
        # keeps them left/right-aligned pixel-for-pixel, letting the two
        # planes read together as one 3D shape.
        ax = fig.subplots(len(rows), ncols, squeeze=False, sharex='col')
        multi_row = len(rows) > 1
        for row_idx, (row_label, reference_panels, labeled_images) in enumerate(rows):
            for col_idx in range(ncols):
                a = ax[row_idx][col_idx]
                if col_idx >= len(labeled_images):
                    a.axis('off')
                    continue
                label, img = labeled_images[col_idx]
                ref_img, ref_mask = reference_panels[col_idx]
                rgb, h, w = composite(ref_img, img)
                # aspect='auto': imshow's default ('equal', pixel-for-pixel)
                # shrinks each Axes' own drawn box to preserve pixel aspect,
                # independent per-Axes -- sharex='col' only locks the DATA
                # x-range, not the rendered box size, so a row of short/wide
                # crops and a row of tall/narrow ones (e.g. YX vs ZX) end up
                # rendered at very different physical sizes even though
                # their columns are "aligned". 'auto' fills the grid cell
                # uniformly instead, which is what actually makes the grid
                # read as one consistent 3D view.
                a.imshow(rgb, aspect='auto')
                if ref_mask is not None and ref_mask.any():
                    a.contour(pad_to_zero(ref_mask.astype(np.uint8), h, w), levels=[0.5], colors='yellow', linewidths=1)
                a.set_title(f'{row_label}: {label}' if multi_row else label, fontsize=10)
                a.axis('off')
        fig.suptitle(f'{title} (red=ref, cyan=moving, yellow=cell boundary)', fontsize=10, wrap=True)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        self.preview_canvas.draw()
        if save_path:
            fig.savefig(save_path, dpi=150)

    def draw_cell_alignment_preview_3col(self, cell, fov,
                                         reference_storage_path, reference_hybe, reference_channel,
                                         reference_fov_matrix,
                                         target_storage_path, target_hybe, target_channel,
                                         fov_only_matrix, final_matrix,
                                         pad=30, lb=0.3, ub=0.9999, save_path=None, target_modality=None,
                                         reference_final_matrix=None, mask_anchor_fov_matrix=None):
        """
        3-way comparison for one target hybe against this cell, per
        explicit request, drawn as TWO rows -- YX plane and ZX plane
        (cell alignment is a real 3D, yx+z correction via
        cell.matrices[hybe]['zx'], so a YX-only preview can't show
        whether the z-depth refinement actually helped):
        Each column pairs a TARGET crop against its OWN reference crop --
        the reference is not one shared image across all three columns,
        per explicit request:
        1. 'raw': BOTH target and reference cropped using the cell mask's
           coordinates taken literally at face value, H=identity on both
           sides -- no correction applied at all, not even FOV-level.
           What you'd see with zero alignment. When target_hybe ==
           reference_hybe, this column is trivially a perfect overlay --
           same crop, same transform, same everything.
        2. 'FOV/cross-modal': target cropped via the cell mask inverse-
           warped through fov_only_matrix; reference cropped via the SAME
           kind of FOV-only matrix (reference_fov_matrix) -- the already-
           established FOV-level (same-modality, or same-modality +
           cross-modal composed for a hybe from the OTHER modality)
           matrix on each side. Neither side's cell-level residual is
           applied here. When target_hybe == reference_hybe, this column
           is also trivially a perfect overlay.
        3. 'final': target cropped via the cell mask inverse-warped
           through final_matrix -- fov_only_matrix refined by this cell's
           own residual (cell.matrices[target_hybe]['yx']). Reference
           cropped via reference_final_matrix -- the SAME KIND of matrix
           (_matrix_to_shared, not reference_fov_matrix's plain FOV
           lookup) applied to reference_hybe, computed independently
           rather than reusing column 2's crop. This is NOT a cosmetic
           choice: reference_hybe generally has its OWN nonzero cell-
           level residual (see ACell.matrix_to_shared) -- reusing column
           2's (purely FOV-only, residual-blind) crop for the reference's
           'final' column would silently disagree with the target's own
           'final' column by exactly that residual, even when target_hybe
           == reference_hybe (confirmed on real data: differed by ~0.9px
           purely from the reference hybe's own residual, breaking the
           same-hybe-perfect-overlay invariant in this column only).
           Defaults to reference_fov_matrix (the old behavior) when not
           given.

        The ZX row uses the SAME crop windows (in ymin/ymax/xmin/xmax) as
        the YX row, per column -- via hybe_zx_projection, on the SAME
        channel_type-resolved channel as the YX row (reference_channel/
        target_channel), matching compute_cell_alignment's own z-depth
        refinement -- so the ZX 'raw'/'FOV/
        cross-modal' columns show exactly the state compute_cell_
        alignment measured its Z-offset against, and 'final' additionally
        applies cell.matrices[target_hybe]['zx'] (a translation on top of
        the 'FOV/cross-modal'-window projection, matching how that matrix
        was computed) so the actual z-registration result is visible, not
        just the yx one -- the reference side's 'final' ZX row uses its
        own reference_final_matrix-derived window (no additional z-warp of
        its own -- there is no per-reference z-residual concept here, only
        the target's own cell.matrices[target_hybe]['zx']). The reference
        row has no cell-mask contour in the ZX row (a depth projection has
        no matching spatial mask).

        reference_fov_matrix: the matrix mapping reference_hybe's own
        native frame to the pipeline's ONE shared reference frame (RNA's
        own same-modality reference hybe -- fov_matrices.get(reference_
        hybe, identity) at compute time). cell.area's coordinates are
        native to cell.reference_hybe (the segmentation hybe), moved once
        into that same shared frame (see this method's own mask_to_shared
        step) -- positioning the reference crop correctly needs the same
        inverse-warp treatment as every other hybe, not a direct,
        untransformed use of cell.area (which was this function's bug in
        its first version -- silently mispositioning the reference crop,
        and therefore both other crops relative to it).

        mask_anchor_fov_matrix: cell.reference_hybe's own native frame ->
        the shared frame, via ONLY the FOV/cross-modal matrix (fov_
        matrices.get(cell.reference_hybe, identity)) -- NO cell-level
        residual. Used as THE anchor for every non-literal column ('FOV/
        cross-modal' AND 'final' alike), never cell.matrix_to_shared
        (cell.reference_hybe, ...) (which includes cell.reference_hybe's
        own fitted residual). Per explicit request: the reference side
        should never move between columns ("red not moved, only cyan
        moved") -- since the reference hybe's OWN matrix is always
        identity (it's this run's own anchor), using the residual-
        bearing anchor made the reference crop's window silently drift
        with cell.reference_hybe's unrelated fit result, confirmed on
        real data both as a visible bug (switching only the cell-level
        fit method moved the reference crop) and as NOT a real spot-
        mapping dependency (perturbing cell.reference_hybe's own fitted
        residual leaves matrix_to_shared(any other hybe, ...) completely
        unchanged -- this was always a display-only artifact). Falls back
        to cell.matrix_to_shared(cell.reference_hybe, ...) (the OLD, pre-
        fix behavior) when not given, rather than crashing -- a caller
        that hasn't been updated yet just doesn't get this fix.

        pad (default 30, user-configurable): pixels of margin included
        around the cell's own bounding box in every crop, so there's
        actual surrounding (non-cell) context to visually judge alignment
        against -- a tight pad shows nothing but undifferentiated signal
        inside the mask, with no way to tell whether nearby structures
        (e.g. neighboring nuclei) also line up.

        Every YX crop is a genuine, unresampled crop of that hybe's own
        raw data -- via align_cell's inverse-warp, the SAME technique
        compute_cell_alignment's own _native_crop uses internally --
        never a warped/resampled whole image. reference_storage_path/
        target_storage_path can differ (a target hybe from the OTHER
        modality lives under its own storage path), which is exactly why
        every input here is explicit rather than assumed to share one
        storage_path/hybe_records the way the older 2-column version did.
        """
        height, width = cell.frame_shape
        x_lit, y_lit = cell.area
        # ONE anchor for every non-literal column -- FOV-only, never the
        # cell-residual-bearing version. Per explicit request: the
        # reference side should never move at all between columns ("red
        # not moved, only cyan moved" -- the most intuitive registration
        # convention, and also the one a spot's own coordinate transform
        # actually uses: matrix_to_shared(hybe, modality) never routes
        # through cell.reference_hybe for any hybe other than the
        # segmentation hybe itself, confirmed by directly perturbing
        # cell.reference_hybe's own fitted residual on real data and
        # finding matrix_to_shared(target_hybe, ...) completely
        # unchanged -- so this was always a display-only artifact, not a
        # real spot-mapping dependency). Previously this function used
        # mask_to_shared_final (cell.matrix_to_shared(cell.reference_hybe,
        # ...), residual-included) as the anchor for 'final' -- since the
        # reference hybe's OWN matrix is always identity (it's this run's
        # own anchor), the reference crop's apparent movement between
        # 'FOV/cross-modal' and 'final' came ENTIRELY from cell.reference_
        # hybe's own residual leaking in through the anchor, not from any
        # real correction to the reference hybe itself. Fixed by using
        # the FOV-only anchor unconditionally: the reference side's crop
        # window is now identical across every non-literal column,
        # exactly matching what a spot's own transform already does.
        mask_anchor = (mask_anchor_fov_matrix if mask_anchor_fov_matrix is not None
                      else cell.matrix_to_shared(cell.reference_hybe, cell.modality))

        def bounds_via(H, basis='final'):
            # Compose the anchor and this column's own H into ONE matrix
            # and apply it to cell.area (x_lit,y_lit) in a SINGLE
            # align_cell call, rather than pre-transforming the anchor
            # via its own align_cell call and feeding that (already
            # int-truncated) result into a second one -- align_cell
            # truncates to int internally, so doing this in two hops
            # accumulates a different (and confirmed, on real data, a
            # real ~1px different) truncation than compute_cell_
            # alignment's own _native_crop, which always composes its
            # two matrices algebraically first and truncates once. Two
            # code paths computing "the same" crop window should not be
            # able to disagree by a pixel merely from HOW MANY steps the
            # composition took.
            if basis == 'literal':
                H_eff = np.linalg.inv(H)
            else:
                H_eff = np.linalg.inv(H) @ mask_anchor
            cy, cx = alignment.align_cell((y_lit, x_lit), H_eff, (height, width))
            if len(cx) == 0:
                return None
            return cy, cx, _true_bounds(H_eff, x_lit, y_lit, pad)

        def crop_via(mip, H, basis='final'):
            b = bounds_via(H, basis=basis)
            if b is None:
                return np.full((1, 1), np.nan, dtype=np.float32), np.zeros((1, 1), dtype=bool), None
            cy, cx, true_bounds = b
            return (_nan_native_crop(mip, true_bounds, height, width),
                    _nan_mask_crop(cy, cx, true_bounds, height, width), true_bounds)

        def zx_via(storage_path, hybe, channel, bounds):
            if bounds is None:
                return np.full((1, 1), np.nan, dtype=np.float32)
            crop = _nan_zx_crop(storage_path, fov, hybe, channel, bounds, height, width, lb, ub)
            return crop if crop is not None else np.full((1, 1), np.nan, dtype=np.float32)

        if reference_final_matrix is None:
            reference_final_matrix = reference_fov_matrix

        ref_mip = _read_mip(reference_storage_path, fov, reference_hybe, reference_channel)
        # Per explicit request: the reference/red side needs its OWN
        # matrix per column, not one shared across raw/FOV/final -- raw is
        # H=identity (cell.area taken literally, matching the target's own
        # raw column); FOV uses the FOV-only matrix; final uses reference_
        # final_matrix, computed INDEPENDENTLY (not reused from FOV) --
        # see this function's own docstring for why final needs its own
        # crop: cell.reference_hybe generally has its own nonzero cell-
        # level residual, which _matrix_to_cellref's composition folds
        # into ANY hybe expressed in its frame, target included, so
        # reusing the residual-blind FOV crop for reference's own final
        # column would silently disagree with the target's.
        reference_raw_crop, reference_raw_mask, reference_raw_bounds = crop_via(ref_mip, np.eye(3), basis='literal')
        reference_fov_crop, reference_fov_mask, reference_fov_bounds = crop_via(ref_mip, reference_fov_matrix, basis='fov')
        reference_final_crop, reference_final_mask, reference_final_bounds = crop_via(
            ref_mip, reference_final_matrix)
        reference_raw_zx = zx_via(reference_storage_path, reference_hybe, reference_channel, reference_raw_bounds)
        reference_fov_zx = zx_via(reference_storage_path, reference_hybe, reference_channel, reference_fov_bounds)
        reference_final_zx = zx_via(reference_storage_path, reference_hybe, reference_channel, reference_final_bounds)

        target_mip = _read_mip(target_storage_path, fov, target_hybe, target_channel)
        # cell mask's OWN coords, no transform at all
        raw_crop, _, raw_bounds = crop_via(target_mip, np.eye(3), basis='literal')
        fov_crop, _, fov_bounds = crop_via(target_mip, fov_only_matrix, basis='fov')
        final_crop, _, final_bounds = crop_via(target_mip, final_matrix)

        raw_zx = zx_via(target_storage_path, target_hybe, target_channel, raw_bounds)
        fov_zx = zx_via(target_storage_path, target_hybe, target_channel, fov_bounds)
        final_zx_precorrection = zx_via(target_storage_path, target_hybe, target_channel, final_bounds)
        H_zx = cell.matrices.get((target_hybe, target_modality if target_modality is not None else cell.modality), {}).get('zx', np.eye(3))
        # H_zx was computed (compute_cell_alignment) against hybe_zx_projection's
        # native (width, depth) orientation -- warp BEFORE transposing for display.
        # borderValue=NaN (not the default 0): a pixel warped in from
        # outside this crop's own true extent has no real data either,
        # same as everywhere else NaN marks "no data" in this preview.
        final_zx = cv2.warpAffine(final_zx_precorrection, H_zx[:2],
                                  (final_zx_precorrection.shape[1], final_zx_precorrection.shape[0]),
                                  borderValue=float('nan'))

        # hybe_zx_projection returns (width, depth) -- transpose here, at
        # display time only, so X (shared with the YX row above, same
        # xmin:xmax window) reads left-to-right and Z reads top-to-bottom,
        # matching the YX row's own orientation and letting the two
        # combine into one readable 3D shape (per explicit request).
        reference_raw_zx, reference_fov_zx, reference_final_zx, raw_zx, fov_zx, final_zx = (
            a.T for a in (reference_raw_zx, reference_fov_zx, reference_final_zx, raw_zx, fov_zx, final_zx))

        title = f'cell {cell.id}: {target_hybe} vs {reference_hybe}'
        self._draw_three_way([
            ('YX', [(reference_raw_crop, reference_raw_mask), (reference_fov_crop, reference_fov_mask),
                    (reference_final_crop, reference_final_mask)],
             [('raw', raw_crop), ('FOV/cross-modal', fov_crop), ('final', final_crop)]),
            ('ZX', [(reference_raw_zx, None), (reference_fov_zx, None), (reference_final_zx, None)],
             [('raw', raw_zx), ('FOV/cross-modal', fov_zx), ('final', final_zx)]),
        ], title, lb, ub, save_path)

    def draw_all_readouts_overlay(self, before_images, after_images, title, save_path=None):
        """
        before_images/after_images: {label: 2D array} (already cropped by
        the caller for the cell-level variant). One combined image per
        side, every readout composited simultaneously in its own color --
        the "did this whole FOV/cell's alignment succeed overall" summary
        view, vs. the pairwise preview which only ever shows one comparison
        at a time.
        """
        fig = self.preview_canvas.figure
        fig.clear()
        ax = fig.subplots(1, 2)
        ax[0].imshow(_composite_multi(before_images))
        ax[0].set_title('before', fontsize=10)
        ax[0].axis('off')
        ax[1].imshow(_composite_multi(after_images))
        ax[1].set_title('after alignment', fontsize=10)
        ax[1].axis('off')
        fig.suptitle(title, fontsize=10, wrap=True)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        self.preview_canvas.draw()
        if save_path:
            fig.savefig(save_path, dpi=150)

    def draw_fov_all_readouts_overlay(self, storage_path, fov, hybe_records, reference_hybe, matrices,
                                      save_path=None, channel_type='readout'):
        """
        Reads every hybe's raw MIP, warps each by its own matrix, builds
        one before/after all-readouts composite for the whole FOV.

        channel_type only affects which channel is READ/DISPLAYED here --
        matrices are always computed fiducial-to-fiducial by
        align_same_modality regardless of this argument (that
        computation choice is deliberate: fiducial images the same
        physical object across every readout hybe, unlike each hybe's own
        readout signal, which shows completely different real content
        hybe-to-hybe and wouldn't correlate meaningfully). A geometric
        affine transform applies just as validly to any channel's pixels,
        so warping the readout channel for viewing/saving is the same
        warp-for-visualization-only pattern used throughout this module --
        readout is the default here because that's the channel with actual
        biological content worth looking at, not fiducial's plain
        bead/chromatin staining.
        """
        record_by_folder = {r['folder']: r for r in hybe_records}
        ref_channel = alignment.pick_channel_by_type(record_by_folder[reference_hybe], channel_type)
        ref_mip = _read_mip(storage_path, fov, reference_hybe, ref_channel)
        height, width = ref_mip.shape

        before_images = {reference_hybe: ref_mip}
        after_images = {reference_hybe: ref_mip}
        for record in hybe_records:
            hybe = record['folder']
            if hybe == reference_hybe or hybe not in matrices:
                continue
            channel = alignment.pick_channel_by_type(record, channel_type)
            mip = _read_mip(storage_path, fov, hybe, channel)
            before_images[hybe] = mip
            after_images[hybe] = cv2.warpAffine(mip.astype(np.float32), matrices[hybe][:2], (width, height))

        title = f'FOV{fov:02d}: all readouts vs {reference_hybe} (before/after)'
        self.draw_all_readouts_overlay(before_images, after_images, title, save_path)

    def draw_cell_all_readouts_overlay(self, cell, fov, reference_hybe, reference_storage_path, reference_channel,
                                       target_specs, pad=10, lb=0.3, ub=0.9999,
                                       save_path=None, reference_matrix=None, reference_final_matrix=None,
                                       mask_anchor_fov_matrix=None):
        """
        Multi-hybe "did this cell's alignment succeed overall" composite,
        rebuilt to match everything the single-hybe preview
        (draw_cell_alignment_preview_3col) already established: real
        per-hybe native crops (never a whole-image warp-then-crop), the 3
        correction stages (raw/FOV-cross-modal/final) as columns, a YX
        row AND a ZX row, a cell-boundary contour, and full cross-modal
        support -- the old version predated all of that and only ever
        showed 2 fiducial-channel panels (before/after) for one modality.

        target_specs: [{'hybe', 'storage_path', 'channel',
        'fov_only_matrix', 'final_matrix',
        'zx_matrix'}, ...] -- one entry per hybe in cell.matrices (both
        modalities), pre-resolved by the caller
        (MainWindow._cell_overlay_target_specs) since resolving which
        modality each hybe's own data lives in needs state (RNA/DNA
        storage paths, cross-modal matrices) this module has no access
        to. Every target here shares ONE fixed reference -- unlike the
        single-hybe preview, there's no per-target anchor to resolve.

        reference_matrix: reference_hybe's own native frame -> the
        pipeline's ONE shared reference frame (RNA's own same-modality
        reference hybe), FOV-LEVEL ONLY -- no cell-level residual (the
        same shape/meaning as each target spec's own 'fov_only_matrix',
        resolved by the caller as a plain fov_matrices lookup -- see
        MainWindow._matrix_to_shared's own docstring for the fallback
        this mirrors). Used for the FOV/cross-modal column; the raw
        column always uses true H=identity regardless of this argument.
        Defaults to identity.

        reference_final_matrix: reference_hybe's own native frame -> the
        same shared frame, the SAME KIND of matrix each target spec's own
        'final_matrix' is (resolved by the caller via MainWindow.
        _matrix_to_shared) -- used for the final column, computed
        INDEPENDENTLY from reference_matrix, NOT reused from the FOV/
        cross-modal column. This matters because the reference hybe
        generally has its own nonzero cell-level residual from this same
        alignment run (see ACell.matrix_to_shared) -- reusing the
        residual-blind reference_matrix crop for the reference's own
        final column would silently disagree with the target's final
        column by exactly that residual, even when reference_hybe == a
        target hybe. Defaults to reference_matrix when not given.

        Unlike _composite_multi's other caller (draw_fov_all_readouts_overlay,
        which warps whole same-shape images), each hybe's own native crop
        here can be a different pixel size (different H per hybe/stage) --
        pad every image in a given panel to that panel's own shared max
        shape before compositing, same technique _draw_three_way uses.

        mask_anchor_fov_matrix: same meaning as draw_cell_alignment_
        preview_3col's own parameter of the same name -- cell.reference_
        hybe's own native frame -> shared frame via ONLY the FOV/cross-
        modal matrix, no cell-level residual. Needed so the 'FOV/cross-
        modal' column's own crop window doesn't silently depend on
        cell.reference_hybe's cell-level fit result (confirmed as a real
        bug on real data -- see that function's own docstring). Falls
        back to the 'final' column's own (residual-bearing) anchor when
        not given, matching the pre-fix behavior rather than crashing.
        """
        height, width = cell.frame_shape
        # ONE anchor for every non-literal column, always FOV-only -- see
        # draw_cell_alignment_preview_3col's own bounds_via for the full
        # rationale (same fix, same function pair, verified on real data
        # both ways: perturbing cell.reference_hybe's own fitted residual
        # has zero effect on any spot's actual matrix_to_shared coordinate,
        # so this was always display-only). 'raw' (Q1): cell.area
        # literally, no transform at all. Every other column: the SAME
        # FOV-only anchor -- the reference panel's own crop window is now
        # identical across every column (its own matrix is always
        # identity, being this run's anchor), matching "red never moves,
        # only the target moves" exactly.
        x_lit, y_lit = cell.area
        mask_anchor = (mask_anchor_fov_matrix if mask_anchor_fov_matrix is not None
                      else cell.matrix_to_shared(cell.reference_hybe, cell.modality))
        if reference_matrix is None:
            reference_matrix = np.eye(3)
        if reference_final_matrix is None:
            reference_final_matrix = reference_matrix

        def bounds_via(H, basis='final'):
            # Single composed-matrix, single-truncation approach -- see
            # draw_cell_alignment_preview_3col's own bounds_via for the
            # full rationale (same fix, same function pair): two
            # sequential align_cell calls (anchor, then H) truncate
            # twice and can disagree by a real pixel with
            # compute_cell_alignment's own single-truncation _native_crop.
            if basis == 'literal':
                H_eff = np.linalg.inv(H)
            else:
                H_eff = np.linalg.inv(H) @ mask_anchor
            cy, cx = alignment.align_cell((y_lit, x_lit), H_eff, (height, width))
            if len(cx) == 0:
                return None
            return cy, cx, _true_bounds(H_eff, x_lit, y_lit, pad)

        def crop_via(mip, H, basis='final'):
            b = bounds_via(H, basis=basis)
            if b is None:
                return None, None, None
            cy, cx, true_bounds = b
            return (_nan_native_crop(mip, true_bounds, height, width),
                    _nan_mask_crop(cy, cx, true_bounds, height, width), true_bounds)

        def zx_via(storage_path, hybe, channel, bounds):
            if bounds is None:
                return None
            return _nan_zx_crop(storage_path, fov, hybe, channel, bounds, height, width, lb, ub)

        def pad_group(d):
            # NaN-fill (not zero-fill): see _nan_native_crop -- a hybe
            # with no real data at a given pixel (either its own true
            # extent doesn't reach there, or it's shorter than another
            # hybe in this group) must stay distinguishable from real,
            # dark signal. _composite_multi treats NaN as "this hybe
            # doesn't contribute here" per-pixel, not "contributes zero".
            imgs = {k: v for k, v in d.items() if v is not None}
            if not imgs:
                return {}, 1, 1
            h = max(a.shape[0] for a in imgs.values())
            w = max(a.shape[1] for a in imgs.values())
            padded = {}
            for k, a in imgs.items():
                a2 = np.full((h, w), np.nan, dtype=np.float32)
                a2[:a.shape[0], :a.shape[1]] = a
                padded[k] = a2
            return padded, h, w

        ref_mip = _read_mip(reference_storage_path, fov, reference_hybe, reference_channel)
        # Per explicit request: raw is ALWAYS true H=identity (the shared-
        # frame mask taken literally, matching every target's own raw
        # column), regardless of reference_matrix. FOV/cross-modal uses
        # reference_matrix; final uses reference_final_matrix, computed
        # INDEPENDENTLY (not reused from FOV) -- the reference hybe
        # generally has its own nonzero cell-level residual (see
        # ACell.matrix_to_shared), so reusing the residual-blind FOV crop
        # for reference's own final column would silently disagree with
        # the target's. The reference panel is otherwise no longer
        # special-cased: it uses the exact same H = matrix_to_shared(...)
        # formula every target hybe does (see MainWindow._matrix_to_
        # shared), just resolved for reference_hybe/reference_modality
        # by the caller instead of a loop.
        ref_raw_crop, ref_raw_mask, ref_raw_bounds = crop_via(ref_mip, np.eye(3), basis='literal')
        ref_fov_crop, ref_fov_mask, ref_fov_bounds = crop_via(ref_mip, reference_matrix, basis='fov')
        ref_final_crop, ref_final_mask, ref_final_bounds = crop_via(ref_mip, reference_final_matrix)
        ref_raw_zx_raw = zx_via(reference_storage_path, reference_hybe, reference_channel, ref_raw_bounds)
        ref_fov_zx_raw = zx_via(reference_storage_path, reference_hybe, reference_channel, ref_fov_bounds)
        ref_final_zx_raw = zx_via(reference_storage_path, reference_hybe, reference_channel, ref_final_bounds)
        ref_raw_zx = ref_raw_zx_raw.T if ref_raw_zx_raw is not None else None
        ref_fov_zx = ref_fov_zx_raw.T if ref_fov_zx_raw is not None else None
        ref_final_zx = ref_final_zx_raw.T if ref_final_zx_raw is not None else None

        raw_yx, fov_yx, final_yx = {reference_hybe: ref_raw_crop}, {reference_hybe: ref_fov_crop}, {reference_hybe: ref_final_crop}
        raw_zx, fov_zx, final_zx = {reference_hybe: ref_raw_zx}, {reference_hybe: ref_fov_zx}, {reference_hybe: ref_final_zx}
        ref_masks = [ref_raw_mask, ref_fov_mask, ref_final_mask]

        for spec in target_specs:
            hybe, sp = spec['hybe'], spec['storage_path']
            # label, not bare hybe, keys every compositing dict below --
            # the same hybe NAME can legitimately appear once per
            # modality (the cross-modal bridge hybe, e.g. Hyb_130, is a
            # real, distinct file in both RNA and DNA); a bare-hybe key
            # would silently let one overwrite the other in these dicts,
            # dropping it from the overlay even though both are correctly
            # present in cell.matrices.
            modality = spec.get('modality')
            label = f'{hybe} ({modality})' if modality else hybe
            mip = _read_mip(sp, fov, hybe, spec['channel'])
            raw_crop, _, raw_b = crop_via(mip, np.eye(3), basis='literal')
            fov_crop, _, fov_b = crop_via(mip, spec['fov_only_matrix'], basis='fov')
            if fov_crop is None:
                continue
            final_crop, _, final_b = crop_via(mip, spec['final_matrix'])
            if raw_crop is None or final_crop is None:
                continue
            raw_yx[label], fov_yx[label], final_yx[label] = raw_crop, fov_crop, final_crop

            raw_zx_raw = zx_via(sp, hybe, spec['channel'], raw_b)
            fov_zx_raw = zx_via(sp, hybe, spec['channel'], fov_b)
            final_zx_precorrection = zx_via(sp, hybe, spec['channel'], final_b)
            if raw_zx_raw is None or fov_zx_raw is None or final_zx_precorrection is None:
                continue
            H_zx = spec.get('zx_matrix', np.eye(3))
            # borderValue=NaN -- see draw_cell_alignment_preview_3col's
            # own identical warpAffine call for the rationale.
            final_zx_raw = cv2.warpAffine(final_zx_precorrection, H_zx[:2],
                                          (final_zx_precorrection.shape[1], final_zx_precorrection.shape[0]),
                                          borderValue=float('nan'))
            raw_zx[label] = raw_zx_raw.T
            fov_zx[label] = fov_zx_raw.T
            final_zx[label] = final_zx_raw.T

        yx_panels = [pad_group(d) for d in (raw_yx, fov_yx, final_yx)]
        zx_panels = [pad_group(d) for d in (raw_zx, fov_zx, final_zx)]
        col_titles = ['raw', 'FOV/cross-modal', 'final']

        fig = self.preview_canvas.figure
        fig.clear()
        ax = fig.subplots(2, 3, squeeze=False, sharex='col')
        # aspect='auto' -- see _draw_three_way's own comment on why 'equal'
        # (imshow's default) breaks a shared grid layout across rows of
        # differently-proportioned crops.
        for col, (imgs, h, w) in enumerate(yx_panels):
            a = ax[0][col]
            a.imshow(_composite_multi(imgs, lb, ub) if imgs else np.zeros((1, 1, 3)), aspect='auto')
            ref_mask = ref_masks[col]
            if ref_mask is not None and ref_mask.any():
                # ref_mask is the reference hybe's own crop, sized independently
                # of (h, w) (the target hybes' shared max crop size for this
                # column) -- affine distortion between hybes means the
                # reference crop can end up LARGER than every target crop in
                # some cells, so clip rather than assume it always fits (a
                # real crash seen on real data: mask (96,71) vs panel (96,70)).
                mh, mw = ref_mask.shape
                ch, cw = min(mh, h), min(mw, w)
                mask_pad = np.zeros((h, w), dtype=np.uint8)
                mask_pad[:ch, :cw] = ref_mask[:ch, :cw].astype(np.uint8)
                a.contour(mask_pad, levels=[0.5], colors='yellow', linewidths=1)
            a.set_title(f'YX: {col_titles[col]}', fontsize=10)
            a.axis('off')
        for col, (imgs, h, w) in enumerate(zx_panels):
            a = ax[1][col]
            a.imshow(_composite_multi(imgs, lb, ub) if imgs else np.zeros((1, 1, 3)), aspect='auto')
            a.set_title(f'ZX: {col_titles[col]}', fontsize=10)
            a.axis('off')

        n_readouts = 1 + len(target_specs)
        # reference_hybe here is just which hybe is shown as the
        # left-most/"reference" panel (the alignment run's own anchor,
        # now always that modality's own same-modality reference hybe --
        # see MainWindow._run_cell_alignment's own comment) -- NOT the
        # coordinate frame itself. Every column, reference panel included,
        # is warped into the pipeline's ONE shared frame (RNA's own same-
        # modality reference hybe, see ACell.matrix_to_shared), regardless
        # of which hybe reference_hybe happens to be.
        fig.suptitle(f'cell {cell.id}: {n_readouts} readout(s), reference panel={reference_hybe} '
                    f'(sequential color, red=frame, yellow=cell boundary)', fontsize=10, wrap=True)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        self.preview_canvas.draw()
        if save_path:
            fig.savefig(save_path, dpi=150)

    def draw_same_modality_preview(self, storage_path, fov, reference_hybe, target_hybe, fiducial_channels, matrices):
        """
        fiducial_channels: {hybe: channel} -- each hybe's own fiducial
        channel (from ExperimentLayout, never assumed uniform).
        matrices: {hybe: 3x3} -- from alignment.align_same_modality.
        """
        reference_mip = _read_mip(storage_path, fov, reference_hybe, fiducial_channels[reference_hybe])
        moving_mip = _read_mip(storage_path, fov, target_hybe, fiducial_channels[target_hybe])
        self.draw_alignment_preview(reference_mip, moving_mip, matrices[target_hybe],
                                    f'{target_hybe} -> {reference_hybe} (fiducial)')

    def draw_cross_modal_preview(self, rna_storage_path, dna_storage_path, fov,
                                 rna_reference_hybe, dna_reference_hybe, channel_type, H_across,
                                 rna_fov_matrices=None, dna_fov_matrices=None, save_path=None):
        """
        rna_fov_matrices/dna_fov_matrices: {hybe: 3x3} -- each modality's own
        already-established within-experiment matrices (same input
        alignment.link_cross_modal itself takes). 'before' shows each raw
        MIP already warped by its own within-experiment correction (which
        is what H_across was actually computed FROM, per link_cross_modal's
        own docstring), not the fully raw, uncorrected MIP -- same-modality
        alignment is a prior, independent layer this one builds on top of,
        not something the cross-modal comparison should re-litigate.
        Omitting either dict (back-compat) falls back to identity, same as
        link_cross_modal's own .get(hybe, identity) default.
        """
        mip_fn = vlinks_store.fiducial_channel_mip if channel_type == 'fiducial' else vlinks_store.readout_channel_mip
        rna_mip = mip_fn(rna_storage_path, fov, rna_reference_hybe)
        dna_mip = mip_fn(dna_storage_path, fov, dna_reference_hybe)

        h, w = rna_mip.shape
        H_rna_within = (rna_fov_matrices or {}).get(rna_reference_hybe, np.eye(3))
        H_dna_within = (dna_fov_matrices or {}).get(dna_reference_hybe, np.eye(3))
        rna_mip_corrected = cv2.warpAffine(rna_mip.astype(np.float32), H_rna_within[:2], (w, h))
        dna_mip_corrected = cv2.warpAffine(dna_mip.astype(np.float32), H_dna_within[:2], (w, h))

        self.draw_alignment_preview(rna_mip_corrected, dna_mip_corrected, H_across,
                                    f'{dna_reference_hybe} (DNA) -> {rna_reference_hybe} (RNA), {channel_type}',
                                    save_path=save_path)


def _read_mip(storage_path, fov, hybe, channel):
    """
    This hybe's MIP straight from vlinks.h5 (see
    vlinks_store.write_hybe_mip) -- previously opened the raw
    {hybe}_stack.h5 directly. Per explicit principle, display/preview code
    should never need the raw stack file; only ingestion (which just wrote
    it) and 3D localization (which needs the full Z-stack, not just the
    MIP) are allowed to touch it.
    """
    return vlinks_store.read_hybe_mip(storage_path, fov, hybe, channel)
