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
                                         reference_specs,
                                         target_storage_path, target_hybe, target_channel,
                                         fov_only_matrix, final_matrix,
                                         pad=30, lb=0.3, ub=0.9999, save_path=None, target_modality=None,
                                         mask_anchor_fov_matrix=None):
        """
        3-way comparison for one target hybe against this cell, drawn as
        one 2-row (YX plane, ZX plane) block PER entry in reference_specs
        -- per explicit request, one block per configured modality's own
        reference hybe, all comparing against the SAME target hybe, so a
        single-hybe preview gives a multi-modality context view instead
        of picking just one reference (cell alignment is a real 3D, yx+z
        correction via cell.matrices[hybe]['zx'], so a YX-only preview
        can't show whether the z-depth refinement actually helped):

        reference_specs: [{'modality', 'storage_path', 'hybe', 'channel',
        'fov_matrix', 'final_matrix'}, ...] -- one entry per configured
        modality's own reference hybe (typically from AlignmentPanel.
        cell_align_references()). 'fov_matrix'/'final_matrix' are the
        SAME KIND of matrix reference_fov_matrix/reference_final_matrix
        used to be for a single reference (a plain FOV-only lookup / the
        real cell-level-refined _matrix_to_shared result) -- 'final_matrix'
        defaults to 'fov_matrix' when omitted from a given spec.

        The target crop (this preview's own single target_hybe) is built
        ONCE and reused across every block -- only the reference side
        differs per modality.
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
           warped through fov_only_matrix; reference cropped via that
           block's own spec['fov_matrix'] -- the already-established
           FOV-level (same-modality, or same-modality + cross-modal
           composed for a hybe from a different modality) matrix on each
           side. Neither side's cell-level residual is applied here. When
           target_hybe == that block's own reference hybe, this column is
           trivially a perfect overlay.
        3. 'final': the cell-level residual (final_matrix minus
           fov_only_matrix) applied to the TARGET IMAGE as a real,
           full-precision float translation, cropped at column 2's OWN
           window -- the window never moves. So the ONLY difference
           between columns 2 and 3 is cyan's own sub-pixel correction,
           and the on-screen red/cyan shift between them IS the residual,
           directly readable. The ZX row gets the same residual's own dx
           applied alongside its depth correction in one warp, so the two
           rows' X-local maxima stay parallel. Reference
           cropped via spec['fov_matrix'] again -- the SAME matrix as
           column 2, never independently recomputed -- per explicit,
           deliberately-enforced principle: red (reference) never moves
           between columns 2 and 3, only cyan (target) does; column 3's
           whole purpose is showing whether target's own correction
           actually lands it on a FIXED reference, not whether reference
           also moved. spec['final_matrix'] is still accepted (defaults
           to spec['fov_matrix'] when omitted) for any caller that has a
           real reason to give the reference its own distinct final
           matrix, but callers building reference_specs from ap.
           cell_align_references() (the normal case -- see _show_cell_
           alignment_preview_for_hybe) omit it entirely on purpose: that
           reference_hybe is ALWAYS this modality's own cell_align_
           reference, i.e. the residual-fit ANCHOR itself, whose own
           residual against itself is identity by construction -- so
           pinning to spec['fov_matrix'] loses nothing real while making
           "red stays fixed" a structural guarantee instead of an
           incidental numerical coincidence (confirmed real regression,
           since fixed: an earlier version called _matrix_to_shared
           independently here, which happened to still equal spec
           ['fov_matrix'] today only because reference_hybe is always the
           anchor -- nothing enforced it). This also does NOT reintroduce
           the older, now-obsolete same-hybe-mismatch bug from a PRIOR
           version (when reference_hybe could be freely chosen as ANY
           hybe, not necessarily the anchor): when target_hybe equals
           that block's own reference hybe, TARGET's own final_matrix
           also reduces to its FOV-only value in that case (the anchor's
           own residual against itself is identity too), so both sides
           still collapse to the same FOV-only crop and the same-hybe-
           perfect-overlay invariant holds either way.

        The ZX row uses the SAME crop windows (in ymin/ymax/xmin/xmax) as
        the YX row, per column -- via hybe_zx_projection, on the SAME
        channel_type-resolved channel as the YX row (that block's own
        spec['channel']/target_channel), matching compute_cell_
        alignment's own z-depth refinement -- so the ZX 'raw'/'FOV/
        cross-modal' columns show exactly the state compute_cell_
        alignment measured its Z-offset against, and 'final' additionally
        applies cell.matrices[target_hybe]['zx'] (a translation on top of
        the 'FOV/cross-modal'-window projection, matching how that matrix
        was computed) so the actual z-registration result is visible, not
        just the yx one -- each block's own reference-side 'final' ZX row
        uses its own spec['final_matrix']-derived window (no additional
        z-warp of its own -- there is no per-reference z-residual concept
        here, only the target's own cell.matrices[target_hybe]['zx']).
        The reference row has no cell-mask contour in the ZX row (a depth
        projection has no matching spatial mask).

        Each spec's own 'fov_matrix': the matrix mapping that spec's
        reference hybe's own native frame to the pipeline's ONE shared
        reference frame (RNA's own same-modality reference hybe -- fov_
        matrices.get(hybe, identity) at compute time). cell.area's
        coordinates are native to cell.reference_hybe (the segmentation
        hybe), moved once into that same shared frame (see this method's
        own mask_to_shared step) -- positioning each reference crop
        correctly needs the same inverse-warp treatment as every other
        hybe, not a direct, untransformed use of cell.area (which was
        this function's bug in its first version -- silently
        mispositioning the reference crop, and therefore both other
        crops relative to it).

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
            cy, cx, bounds = b
            return (_nan_native_crop(mip, bounds, height, width),
                    _nan_mask_crop(cy, cx, bounds, height, width), bounds)

        def zx_via(storage_path, hybe, channel, bounds):
            # Read straight from HDF5 at this column's OWN window -- the
            # SAME (ymin,ymax,xmin,xmax) the YX row above used, which is
            # what keeps the two rows' X in lockstep (see _draw_three_way's
            # sharex='col'). Nothing is ever re-windowed afterward.
            if bounds is None:
                return np.full((1, 1), np.nan, dtype=np.float32)
            crop = _nan_zx_crop(storage_path, fov, hybe, channel, bounds, height, width, lb, ub)
            return crop if crop is not None else np.full((1, 1), np.nan, dtype=np.float32)

        # Target crop built ONCE -- this preview's own single target_hybe,
        # the same across every modality's own reference block below.
        target_mip = _read_mip(target_storage_path, fov, target_hybe, target_channel)
        # cell mask's OWN coords, no transform at all
        raw_crop, _, raw_bounds = crop_via(target_mip, np.eye(3), basis='literal')
        fov_crop, _, fov_bounds = crop_via(target_mip, fov_only_matrix, basis='fov')

        # --- column 3: the cell-level residual, applied to the IMAGE ---
        # Per explicit design decision, column 3 reuses column 2's OWN
        # window unchanged and moves the cyan/target CONTENT instead. The
        # residual is a plain translation, so this is one warpAffine of
        # the target's own native mip -- no re-windowing anywhere.
        #
        # This REPLACES the old "recompute the window from final_matrix"
        # approach, whose window came from _true_bounds' own int() -- so a
        # sub-pixel residual truncated to a 0px window shift and rendered
        # as no change at all (confirmed on real data: a real dx=-0.506
        # displayed as 0), while the ZX row's own z-correction went
        # through warpAffine and DID move sub-pixel. YX and ZX therefore
        # disagreed sub-pixel-wise and their X-local maxima stopped being
        # parallel. Warping both keeps the residual at full float
        # precision and keeps the two rows' X in lockstep.
        residual_dx = float(final_matrix[0, 2] - fov_only_matrix[0, 2])
        residual_dy = float(final_matrix[1, 2] - fov_only_matrix[1, 2])
        T_residual = np.array([[1., 0., residual_dx], [0., 1., residual_dy]])
        col3_source = cv2.warpAffine(target_mip.astype(np.float32), T_residual,
                                     (width, height), borderValue=float('nan'))
        final_crop = _nan_native_crop(col3_source, fov_bounds, height, width)

        raw_zx = zx_via(target_storage_path, target_hybe, target_channel, raw_bounds)
        fov_zx = zx_via(target_storage_path, target_hybe, target_channel, fov_bounds)
        H_zx = cell.matrices.get((target_hybe, target_modality if target_modality is not None else cell.modality), {}).get('zx', np.eye(3))
        # ZX column 3 reads at column 2's OWN native window (never a
        # residual-moved one), then applies BOTH corrections in one warp:
        # the depth shift (H_zx[0,2]) and the SAME float residual_dx the
        # YX row above just used. hybe_zx_projection's own array is
        # (width, depth), so in cv2's (x=column, y=row) convention on THAT
        # array the depth shift lands in [0,2] and the x shift in [1,2] --
        # which is also why compute_cell_alignment writes z at [0,2].
        # Warp BEFORE transposing for display. borderValue=NaN (not the
        # default 0): a pixel warped in from outside this crop's own true
        # extent has no real data either, same as everywhere else NaN
        # marks "no data" in this preview.
        # Read a WIDER x-window than needed, warp, then trim back to the
        # exact column-2 window. Warping the tight crop directly would
        # fabricate a NaN (-> flat gray "no data") sliver along its own
        # edge, even though real pixels exist just outside it -- gray must
        # only ever mean genuinely-absent data (a window running off the
        # real frame), never an artifact of how far we shifted. The YX row
        # above has no such problem because it warps the FULL mip and only
        # then crops, so its shift pulls in real neighbouring pixels.
        # Depth needs no such margin: a shift along z legitimately runs
        # out of stack, and that thin band IS real missing data.
        # +2, not +1: warpAffine's bilinear kernel makes a pixel NaN if
        # ANY of its source neighbours is NaN, so NaN bleeds one row
        # further inward than the pure shift distance.
        x_margin = int(np.ceil(abs(residual_dx))) + 2
        ymin_b, ymax_b, xmin_b, xmax_b = fov_bounds
        wide_bounds = (ymin_b, ymax_b, xmin_b - x_margin, xmax_b + x_margin)
        zx_wide = _nan_zx_crop(target_storage_path, fov, target_hybe, target_channel,
                               wide_bounds, height, width, lb, ub)
        if zx_wide is None:
            final_zx = np.full((1, 1), np.nan, dtype=np.float32)
        else:
            H_zx_display = np.array([[1., 0., float(H_zx[0, 2])], [0., 1., residual_dx]])
            if np.allclose(H_zx_display[:, 2], 0.0):
                # Nothing to apply -- skip warpAffine entirely rather than
                # round-tripping through it. Confirmed real artifact: even
                # an exactly-identity warp NaNs the final DEPTH row, because
                # bilinear sampling at the last row reaches one row past the
                # array into borderValue. That rendered as a flat gray line
                # across a mid-frame cell that has no missing data at all.
                # align_cell checks its own matrix for identity for the same
                # reason -- check the math, not the caller's expectations.
                warped_wide = zx_wide
            else:
                warped_wide = cv2.warpAffine(zx_wide, H_zx_display,
                                             (zx_wide.shape[1], zx_wide.shape[0]),
                                             borderValue=float('nan'))
            final_zx = warped_wide[x_margin:x_margin + (xmax_b - xmin_b)]
        # hybe_zx_projection returns (width, depth) -- transpose here, at
        # display time only, so X (shared with the YX row above, same
        # xmin:xmax window) reads left-to-right and Z reads top-to-bottom,
        # matching the YX row's own orientation and letting the two
        # combine into one readable 3D shape (per explicit request).
        raw_zx, fov_zx, final_zx = raw_zx.T, fov_zx.T, final_zx.T

        # One 2-row (YX, ZX) block per entry in reference_specs -- per
        # explicit request, one block per configured modality's own
        # reference hybe, all comparing against the SAME target crop
        # built above.
        rows = []
        for spec in reference_specs:
            modality = spec.get('modality')
            reference_storage_path, reference_hybe = spec['storage_path'], spec['hybe']
            reference_channel = spec['channel']
            reference_fov_matrix = spec['fov_matrix']

            ref_mip = _read_mip(reference_storage_path, fov, reference_hybe, reference_channel)
            # raw is H=identity (cell.area taken literally, matching the
            # target's own raw column); columns 2 and 3 BOTH use this
            # spec's own reference_fov_matrix. spec['final_matrix'] is
            # deliberately not read at all any more -- red never moves
            # between those columns, so there is no second reference
            # matrix to honour (see this function's own "3. 'final'"
            # docstring paragraph).
            reference_raw_crop, reference_raw_mask, reference_raw_bounds = crop_via(
                ref_mip, np.eye(3), basis='literal')
            reference_fov_crop, reference_fov_mask, reference_fov_bounds = crop_via(
                ref_mip, reference_fov_matrix, basis='fov')
            reference_raw_zx = zx_via(reference_storage_path, reference_hybe, reference_channel,
                                      reference_raw_bounds)
            reference_fov_zx = zx_via(reference_storage_path, reference_hybe, reference_channel,
                                      reference_fov_bounds)
            # Columns 2 and 3 share the SAME red arrays by construction --
            # the same objects, not merely equal ones. Red must never move
            # between those columns (column 3's whole role is showing cyan
            # corrected onto a FIXED red), and reusing the arrays makes
            # that a structural guarantee rather than something that holds
            # only while reference_final_matrix happens to equal
            # reference_fov_matrix. Applies to BOTH the YX and the ZX row.
            reference_final_crop, reference_final_mask = reference_fov_crop, reference_fov_mask
            reference_final_zx = reference_fov_zx
            reference_raw_zx, reference_fov_zx, reference_final_zx = (
                reference_raw_zx.T, reference_fov_zx.T, reference_final_zx.T)

            yx_label = f'YX ({modality})' if modality else 'YX'
            zx_label = f'ZX ({modality})' if modality else 'ZX'
            rows.append((yx_label, [(reference_raw_crop, reference_raw_mask), (reference_fov_crop, reference_fov_mask),
                                    (reference_final_crop, reference_final_mask)],
                        [('raw', raw_crop), ('FOV/cross-modal', fov_crop), ('final', final_crop)]))
            rows.append((zx_label, [(reference_raw_zx, None), (reference_fov_zx, None), (reference_final_zx, None)],
                        [('raw', raw_zx), ('FOV/cross-modal', fov_zx), ('final', final_zx)]))

        reference_summary = ', '.join(f"{spec['hybe']} ({spec.get('modality')})" for spec in reference_specs)
        title = f'cell {cell.id}: {target_hybe} vs {reference_summary}'
        self._draw_three_way(rows, title, lb, ub, save_path)

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

        Per explicit request, target_specs is grouped by modality into
        one 2x3 (YX/ZX x raw/FOV/final) BLOCK per configured modality,
        stacked in one figure -- previously every hybe (potentially from
        two different modalities/stains) was thrown into ONE shared
        composite, confusing when hybes from different modalities blended
        into the same sequential-color image. The SAME reference panel
        (this run's own anchor) is included in every block; only the
        TARGET hybes are split.

        target_specs: [{'hybe', 'storage_path', 'channel', 'modality',
        'fov_only_matrix', 'final_matrix', 'zx_matrix'}, ...] -- one entry
        per active hybe in EVERY configured modality (not just cell.
        matrices' own keys -- a modality this cell has no cell-level
        residual for still gets a spec, via the FOV-only fallback), pre-
        resolved by the caller (MainWindow._cell_overlay_target_specs)
        since resolving which modality each hybe's own data lives in
        needs state (RNA/DNA storage paths, cross-modal matrices) this
        module has no access to. Every target shares ONE fixed reference
        -- unlike the single-hybe preview, there's no per-target anchor
        to resolve.

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

        ref_masks = [ref_raw_mask, ref_fov_mask, ref_final_mask]

        # Group target_specs by modality -- per explicit request, each
        # configured modality gets its OWN 2x3 (YX/ZX x raw/FOV/final)
        # block instead of every hybe (potentially from two different
        # modalities/stains) being thrown into ONE shared composite,
        # which was confusing when hybes from different modalities
        # blended into the same sequential-color image. The SAME
        # reference panel is included in every modality's own block (the
        # one shared anchor the whole run compares against) -- only the
        # TARGET hybes are split by modality. Dict order follows
        # target_specs' own order (MainWindow._cell_overlay_target_specs
        # emits this cell's own modality first, then every other
        # configured modality), so this cell's own modality's block
        # renders first without needing an explicit sort.
        specs_by_modality = {}
        for spec in target_specs:
            specs_by_modality.setdefault(spec.get('modality'), []).append(spec)

        blocks = []  # [(modality, yx_panels, zx_panels), ...]
        for modality, specs in specs_by_modality.items():
            raw_yx = {reference_hybe: ref_raw_crop}
            fov_yx = {reference_hybe: ref_fov_crop}
            final_yx = {reference_hybe: ref_final_crop}
            raw_zx = {reference_hybe: ref_raw_zx}
            fov_zx = {reference_hybe: ref_fov_zx}
            final_zx = {reference_hybe: ref_final_zx}
            for spec in specs:
                hybe, sp = spec['hybe'], spec['storage_path']
                # bare hybe as the label is now safe -- each block is
                # already scoped to ONE modality, so the cross-modal
                # bridge hybe (e.g. Hyb_130, a real distinct file in both
                # RNA and DNA) can no longer collide within a single block
                # the way it could when every modality shared one dict.
                mip = _read_mip(sp, fov, hybe, spec['channel'])
                raw_crop, _, raw_b = crop_via(mip, np.eye(3), basis='literal')
                fov_crop, _, fov_b = crop_via(mip, spec['fov_only_matrix'], basis='fov')
                if fov_crop is None:
                    continue
                final_crop, _, final_b = crop_via(mip, spec['final_matrix'])
                if raw_crop is None or final_crop is None:
                    continue
                raw_yx[hybe], fov_yx[hybe], final_yx[hybe] = raw_crop, fov_crop, final_crop

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
                raw_zx[hybe] = raw_zx_raw.T
                fov_zx[hybe] = fov_zx_raw.T
                final_zx[hybe] = final_zx_raw.T

            yx_panels = [pad_group(d) for d in (raw_yx, fov_yx, final_yx)]
            zx_panels = [pad_group(d) for d in (raw_zx, fov_zx, final_zx)]
            blocks.append((modality, yx_panels, zx_panels))

        col_titles = ['raw', 'FOV/cross-modal', 'final']
        num_blocks = max(len(blocks), 1)

        fig = self.preview_canvas.figure
        fig.clear()
        ax = fig.subplots(2 * num_blocks, 3, squeeze=False, sharex='col')
        # aspect='auto' -- see _draw_three_way's own comment on why 'equal'
        # (imshow's default) breaks a shared grid layout across rows of
        # differently-proportioned crops.
        for block_idx, (modality, yx_panels, zx_panels) in enumerate(blocks):
            yx_row, zx_row = 2 * block_idx, 2 * block_idx + 1
            modality_label = f' ({modality})' if modality else ''
            for col, (imgs, h, w) in enumerate(yx_panels):
                a = ax[yx_row][col]
                a.imshow(_composite_multi(imgs, lb, ub) if imgs else np.zeros((1, 1, 3)), aspect='auto')
                ref_mask = ref_masks[col]
                if ref_mask is not None and ref_mask.any():
                    # ref_mask is the reference hybe's own crop, sized
                    # independently of (h, w) (this block's own target
                    # hybes' shared max crop size for this column) --
                    # affine distortion between hybes means the reference
                    # crop can end up LARGER than every target crop in
                    # some cells, so clip rather than assume it always
                    # fits (a real crash seen on real data: mask (96,71)
                    # vs panel (96,70)).
                    mh, mw = ref_mask.shape
                    ch, cw = min(mh, h), min(mw, w)
                    mask_pad = np.zeros((h, w), dtype=np.uint8)
                    mask_pad[:ch, :cw] = ref_mask[:ch, :cw].astype(np.uint8)
                    a.contour(mask_pad, levels=[0.5], colors='yellow', linewidths=1)
                a.set_title(f'YX{modality_label}: {col_titles[col]}', fontsize=10)
                a.axis('off')
            for col, (imgs, h, w) in enumerate(zx_panels):
                a = ax[zx_row][col]
                a.imshow(_composite_multi(imgs, lb, ub) if imgs else np.zeros((1, 1, 3)), aspect='auto')
                a.set_title(f'ZX{modality_label}: {col_titles[col]}', fontsize=10)
                a.axis('off')

        n_readouts = 1 + len(target_specs)
        # reference_hybe here is just which hybe is shown as the
        # reference panel in EVERY block (the alignment run's own anchor,
        # now always that modality's own same-modality reference hybe --
        # see MainWindow._run_cell_alignment's own comment) -- NOT the
        # coordinate frame itself. Every column, reference panel included,
        # is warped into the pipeline's ONE shared frame (RNA's own same-
        # modality reference hybe, see ACell.matrix_to_shared), regardless
        # of which hybe reference_hybe happens to be.
        fig.suptitle(f'cell {cell.id}: {n_readouts} readout(s) across {num_blocks} modality block(s), '
                    f'reference panel={reference_hybe} '
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
