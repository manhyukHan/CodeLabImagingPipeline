import os
import numpy as np
import h5py
import cv2

from codelab_pipeline.io import preprocess
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
    images: {label: 2D array}. Normalizes each, maps it through a
    sequential red-cyan color (reference hybe = red, last readout = cyan,
    everything else interpolated between), pixelwise-max across all of them
    -- same composites.max(-1) pattern already used in
    codelab_pipeline/segmentation/segment.py's legacy CellbarcodeWidget.
    Gives one combined image showing every readout simultaneously, instead
    of the 2-color pairwise composite -- the pairwise view can't show
    whether alignment succeeded overall across many hybes at once.
    """
    labels = list(images.keys())
    shape = next(iter(images.values())).shape
    composite = np.zeros((*shape, 3), dtype=float)
    n = len(labels)
    for i, label in enumerate(labels):
        color = _sequential_color(i, n)
        norm = preprocess.normalize_to_uint8(images[label], lb, ub).astype(float) / 255
        layer = np.stack([norm * c for c in color], axis=-1)
        composite = np.maximum(composite, layer)
    return composite


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

        rows: [(row_label, reference_panel, labeled_images), ...] -- one
        row per alignment plane (YX, ZX) sharing one figure/pop-up, per
        explicit request to see the ZX plane alongside YX rather than as
        a separate preview.
        reference_panel: (image, mask) -- mask is a boolean array (same
        shape as image) marking where the cell's own mask lands in the
        reference crop's frame, or None to skip drawing a boundary (the
        ZX-plane row has no such mask -- it's a depth projection, not a
        spatial arrangement the mask lands in).
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
        def norm(a):
            return preprocess.normalize_to_uint8(a, lb, ub)

        def pad_to(a, h, w):
            a2 = np.zeros((h, w), dtype=a.dtype)
            a2[:a.shape[0], :a.shape[1]] = a
            return a2

        def composite(a, b):
            h, w = max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])
            c = np.zeros((h, w, 3), dtype=float)
            c[..., 0] = norm(pad_to(a, h, w))
            c[..., 1] = norm(pad_to(b, h, w))
            c[..., 2] = norm(pad_to(b, h, w))
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
        for row_idx, (row_label, reference_panel, labeled_images) in enumerate(rows):
            ref_img, ref_mask = reference_panel
            for col_idx in range(ncols):
                a = ax[row_idx][col_idx]
                if col_idx >= len(labeled_images):
                    a.axis('off')
                    continue
                label, img = labeled_images[col_idx]
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
                    a.contour(pad_to(ref_mask.astype(np.uint8), h, w), levels=[0.5], colors='yellow', linewidths=1)
                a.set_title(f'{row_label}: {label}' if multi_row else label, fontsize=10)
                a.axis('off')
        fig.suptitle(f'{title} (red=ref, cyan=moving, yellow=cell boundary)', fontsize=10, wrap=True)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        self.preview_canvas.draw()
        if save_path:
            fig.savefig(save_path, dpi=150)

    def draw_cell_alignment_preview_3col(self, cell, fov,
                                         reference_storage_path, reference_hybe, reference_channel,
                                         reference_fiducial_channel, reference_fov_matrix,
                                         target_storage_path, target_hybe, target_channel, target_fiducial_channel,
                                         fov_only_matrix, final_matrix,
                                         pad=30, lb=0.3, ub=0.9999, save_path=None):
        """
        3-way comparison for one target hybe against this cell, per
        explicit request, drawn as TWO rows -- YX plane and ZX plane
        (cell alignment is a real 3D, yx+z correction via
        cell.matrices[hybe]['zx'], so a YX-only preview can't show
        whether the z-depth refinement actually helped):
        1. 'raw': target hybe's own image, cropped using the cell mask's
           coordinates taken literally at face value -- no correction
           applied at all, not even FOV-level. What you'd see with zero
           alignment.
        2. 'FOV/cross-modal': target hybe cropped via the cell mask
           inverse-warped through fov_only_matrix -- the already-
           established FOV-level (same-modality, or same-modality +
           cross-modal composed for a hybe from the OTHER modality)
           matrix. This cell's own residual refinement is NOT applied here.
        3. 'final': target hybe cropped via the cell mask inverse-warped
           through final_matrix -- fov_only_matrix refined by this cell's
           own residual (cell.matrices[target_hybe]['yx']).

        The ZX row uses the SAME 3 crop windows (in ymin/ymax/xmin/xmax)
        as the YX row -- via hybe_zx_projection, the same fiducial-channel
        Z-stack max-projection compute_cell_alignment's own z-depth
        refinement is built from -- so the ZX 'raw'/'FOV/cross-modal'
        columns show exactly the state compute_cell_alignment measured
        its Z-offset against, and 'final' additionally applies
        cell.matrices[target_hybe]['zx'] (a translation on top of the
        'FOV/cross-modal'-window projection, matching how that matrix was
        computed) so the actual z-registration result is visible, not
        just the yx one. The reference row has no cell-mask contour in
        the ZX row (a depth projection has no matching spatial mask).

        reference_fov_matrix: the matrix mapping reference_hybe's own
        native frame to cell.reference_hybe's frame (fov_matrices.get
        (reference_hybe, identity) at compute time) -- reference_hybe is
        whatever ACTUALLY anchored this target_hybe's phase-correlation,
        which is only cell.reference_hybe itself in the common case;
        cell.area's coordinates are native to cell.reference_hybe, so
        positioning the reference crop correctly needs the same inverse-
        warp treatment as every other hybe, not a direct, untransformed
        use of cell.area (which was this function's bug in its first
        version -- silently mispositioning the reference crop, and
        therefore both other crops relative to it, whenever the compute-
        time anchor differed from the segmentation hybe).

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
        x_ref, y_ref = cell.area

        def bounds_via(H):
            cy, cx = alignment.align_cell((y_ref, x_ref), np.linalg.inv(H), (height, width))
            if len(cx) == 0:
                return None
            ymin, ymax = max(0, int(cy.min()) - pad), min(height, int(cy.max()) + pad + 1)
            xmin, xmax = max(0, int(cx.min()) - pad), min(width, int(cx.max()) + pad + 1)
            return cy, cx, ymin, ymax, xmin, xmax

        def crop_via(mip, H):
            b = bounds_via(H)
            if b is None:
                return np.zeros((1, 1), dtype=mip.dtype), np.zeros((1, 1), dtype=bool), None
            cy, cx, ymin, ymax, xmin, xmax = b
            mask_full = np.zeros((height, width), dtype=bool)
            mask_full[cy, cx] = True
            return mip[ymin:ymax, xmin:xmax], mask_full[ymin:ymax, xmin:xmax], (ymin, ymax, xmin, xmax)

        def zx_via(storage_path, hybe, fiducial_channel, bounds):
            if bounds is None:
                return np.zeros((1, 1), dtype=np.uint8)
            ymin, ymax, xmin, xmax = bounds
            return alignment.hybe_zx_projection(storage_path, fov, hybe, fiducial_channel,
                                                ymin, ymax, xmin, xmax, lb, ub)

        ref_mip = _read_mip(_h5_path(reference_storage_path, fov, reference_hybe), reference_channel)
        reference_crop, reference_mask, reference_bounds = crop_via(ref_mip, reference_fov_matrix)
        reference_zx = zx_via(reference_storage_path, reference_hybe, reference_fiducial_channel, reference_bounds)

        target_mip = _read_mip(_h5_path(target_storage_path, fov, target_hybe), target_channel)
        # cell mask's OWN coords, no transform at all
        raw_crop, _, raw_bounds = crop_via(target_mip, np.eye(3))
        fov_crop, _, fov_bounds = crop_via(target_mip, fov_only_matrix)
        final_crop, _, final_bounds = crop_via(target_mip, final_matrix)

        raw_zx = zx_via(target_storage_path, target_hybe, target_fiducial_channel, raw_bounds)
        fov_zx = zx_via(target_storage_path, target_hybe, target_fiducial_channel, fov_bounds)
        final_zx_precorrection = zx_via(target_storage_path, target_hybe, target_fiducial_channel, final_bounds)
        H_zx = cell.matrices.get(target_hybe, {}).get('zx', np.eye(3))
        # H_zx was computed (compute_cell_alignment) against hybe_zx_projection's
        # native (width, depth) orientation -- warp BEFORE transposing for display.
        final_zx = cv2.warpAffine(final_zx_precorrection, H_zx[:2],
                                  (final_zx_precorrection.shape[1], final_zx_precorrection.shape[0]))

        # hybe_zx_projection returns (width, depth) -- transpose here, at
        # display time only, so X (shared with the YX row above, same
        # xmin:xmax window) reads left-to-right and Z reads top-to-bottom,
        # matching the YX row's own orientation and letting the two
        # combine into one readable 3D shape (per explicit request).
        reference_zx, raw_zx, fov_zx, final_zx = (a.T for a in (reference_zx, raw_zx, fov_zx, final_zx))

        title = f'cell {cell.id}: {target_hybe} vs {reference_hybe}'
        self._draw_three_way([
            ('YX', (reference_crop, reference_mask),
             [('raw', raw_crop), ('FOV/cross-modal', fov_crop), ('final', final_crop)]),
            ('ZX', (reference_zx, None),
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
        ref_mip = _read_mip(_h5_path(storage_path, fov, reference_hybe), ref_channel)
        height, width = ref_mip.shape

        before_images = {reference_hybe: ref_mip}
        after_images = {reference_hybe: ref_mip}
        for record in hybe_records:
            hybe = record['folder']
            if hybe == reference_hybe or hybe not in matrices:
                continue
            channel = alignment.pick_channel_by_type(record, channel_type)
            mip = _read_mip(_h5_path(storage_path, fov, hybe), channel)
            before_images[hybe] = mip
            after_images[hybe] = cv2.warpAffine(mip.astype(np.float32), matrices[hybe][:2], (width, height))

        title = f'FOV{fov:02d}: all readouts vs {reference_hybe} (before/after)'
        self.draw_all_readouts_overlay(before_images, after_images, title, save_path)

    def draw_cell_all_readouts_overlay(self, cell, fov, reference_hybe, reference_storage_path, reference_channel,
                                       reference_fiducial_channel, target_specs, pad=10, lb=0.3, ub=0.9999, save_path=None):
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
        'fiducial_channel', 'fov_only_matrix', 'final_matrix',
        'zx_matrix'}, ...] -- one entry per hybe in cell.matrices (both
        modalities), pre-resolved by the caller
        (MainWindow._cell_overlay_target_specs) since resolving which
        modality each hybe's own data lives in needs state (RNA/DNA
        storage paths, cross-modal matrices) this module has no access
        to. Every target here shares ONE fixed reference -- unlike the
        single-hybe preview, there's no per-target anchor to resolve:
        cell.reference_hybe is always read at H=eye (cell.area is already
        native to its own frame).

        Unlike _composite_multi's other caller (draw_fov_all_readouts_overlay,
        which warps whole same-shape images), each hybe's own native crop
        here can be a different pixel size (different H per hybe/stage) --
        pad every image in a given panel to that panel's own shared max
        shape before compositing, same technique _draw_three_way uses.
        """
        height, width = cell.frame_shape
        x_ref, y_ref = cell.area

        def bounds_via(H):
            cy, cx = alignment.align_cell((y_ref, x_ref), np.linalg.inv(H), (height, width))
            if len(cx) == 0:
                return None
            ymin, ymax = max(0, int(cy.min()) - pad), min(height, int(cy.max()) + pad + 1)
            xmin, xmax = max(0, int(cx.min()) - pad), min(width, int(cx.max()) + pad + 1)
            return cy, cx, ymin, ymax, xmin, xmax

        def crop_via(mip, H):
            b = bounds_via(H)
            if b is None:
                return None, None, None
            cy, cx, ymin, ymax, xmin, xmax = b
            mask_full = np.zeros((height, width), dtype=bool)
            mask_full[cy, cx] = True
            return mip[ymin:ymax, xmin:xmax], mask_full[ymin:ymax, xmin:xmax], (ymin, ymax, xmin, xmax)

        def zx_via(storage_path, hybe, fiducial_channel, bounds):
            if bounds is None:
                return None
            ymin, ymax, xmin, xmax = bounds
            return alignment.hybe_zx_projection(storage_path, fov, hybe, fiducial_channel,
                                                ymin, ymax, xmin, xmax, lb, ub)

        def pad_group(d):
            imgs = {k: v for k, v in d.items() if v is not None}
            if not imgs:
                return {}, 1, 1
            h = max(a.shape[0] for a in imgs.values())
            w = max(a.shape[1] for a in imgs.values())
            padded = {}
            for k, a in imgs.items():
                a2 = np.zeros((h, w), dtype=a.dtype)
                a2[:a.shape[0], :a.shape[1]] = a
                padded[k] = a2
            return padded, h, w

        ref_mip = _read_mip(_h5_path(reference_storage_path, fov, reference_hybe), reference_channel)
        ref_crop, ref_mask, ref_bounds = crop_via(ref_mip, np.eye(3))
        ref_zx_raw = zx_via(reference_storage_path, reference_hybe, reference_fiducial_channel, ref_bounds)
        ref_zx = ref_zx_raw.T if ref_zx_raw is not None else None

        raw_yx, fov_yx, final_yx = {reference_hybe: ref_crop}, {reference_hybe: ref_crop}, {reference_hybe: ref_crop}
        raw_zx, fov_zx, final_zx = {reference_hybe: ref_zx}, {reference_hybe: ref_zx}, {reference_hybe: ref_zx}

        for spec in target_specs:
            hybe, sp = spec['hybe'], spec['storage_path']
            mip = _read_mip(_h5_path(sp, fov, hybe), spec['channel'])
            raw_crop, _, raw_b = crop_via(mip, np.eye(3))
            fov_crop, _, fov_b = crop_via(mip, spec['fov_only_matrix'])
            final_crop, _, final_b = crop_via(mip, spec['final_matrix'])
            if raw_crop is None or fov_crop is None or final_crop is None:
                continue
            raw_yx[hybe], fov_yx[hybe], final_yx[hybe] = raw_crop, fov_crop, final_crop

            raw_zx_raw = zx_via(sp, hybe, spec['fiducial_channel'], raw_b)
            fov_zx_raw = zx_via(sp, hybe, spec['fiducial_channel'], fov_b)
            final_zx_precorrection = zx_via(sp, hybe, spec['fiducial_channel'], final_b)
            if raw_zx_raw is None or fov_zx_raw is None or final_zx_precorrection is None:
                continue
            H_zx = spec.get('zx_matrix', np.eye(3))
            final_zx_raw = cv2.warpAffine(final_zx_precorrection, H_zx[:2],
                                          (final_zx_precorrection.shape[1], final_zx_precorrection.shape[0]))
            raw_zx[hybe] = raw_zx_raw.T
            fov_zx[hybe] = fov_zx_raw.T
            final_zx[hybe] = final_zx_raw.T

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
        # "vs {reference_hybe}" would misleadingly read as "the cell-
        # alignment reference hybe" -- reference_hybe here is
        # cell.reference_hybe, the SEGMENTATION hybe (a different concept,
        # see _cell_overlay_target_specs' own docstring): the shared
        # coordinate frame every hybe is warped into, not necessarily
        # what any hybe's own residual was computed against.
        fig.suptitle(f'cell {cell.id}: {n_readouts} readout(s), frame={reference_hybe} (segmentation hybe) '
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
        ref_path = _h5_path(storage_path, fov, reference_hybe)
        target_path = _h5_path(storage_path, fov, target_hybe)
        reference_mip = _read_mip(ref_path, fiducial_channels[reference_hybe])
        moving_mip = _read_mip(target_path, fiducial_channels[target_hybe])
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
        mip_fn = alignment._fiducial_channel_mip if channel_type == 'fiducial' else alignment._readout_channel_mip
        rna_path = _h5_path(rna_storage_path, fov, rna_reference_hybe)
        dna_path = _h5_path(dna_storage_path, fov, dna_reference_hybe)
        rna_mip = mip_fn(rna_path)
        dna_mip = mip_fn(dna_path)

        h, w = rna_mip.shape
        H_rna_within = (rna_fov_matrices or {}).get(rna_reference_hybe, np.eye(3))
        H_dna_within = (dna_fov_matrices or {}).get(dna_reference_hybe, np.eye(3))
        rna_mip_corrected = cv2.warpAffine(rna_mip.astype(np.float32), H_rna_within[:2], (w, h))
        dna_mip_corrected = cv2.warpAffine(dna_mip.astype(np.float32), H_dna_within[:2], (w, h))

        self.draw_alignment_preview(rna_mip_corrected, dna_mip_corrected, H_across,
                                    f'{dna_reference_hybe} (DNA) -> {rna_reference_hybe} (RNA), {channel_type}',
                                    save_path=save_path)


def _h5_path(storage_path, fov, hybe):
    return os.path.join(storage_path, f'FOV{fov:02d}', f'{hybe}_stack.h5')


def _read_mip(h5path, channel):
    with h5py.File(h5path, 'r') as f:
        return f[f'/mip/ch{channel}'][:]
