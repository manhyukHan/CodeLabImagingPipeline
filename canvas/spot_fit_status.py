import numpy as np

from codelab_pipeline.io import preprocess


def draw_spot_fit_status(ax_yx, ax_xz, cubic, centroid=None, lb=0.3, ub=0.9999, title='',
                         marker_size=130, z_display_pad=15, title_fontsize=9,
                         rejected=None, scale_half=2, scale_half_z=5,
                         lateral=None):
    """
    Renders one spot's fit-status: a YX max-projection (over Z) and an XZ
    max-projection (over Y -- X horizontal, Z vertical, same display
    convention codelab_pipeline.alignment.chain.hybe_zx_projection's own
    callers already use), with the fitted centroid circled in yellow when
    a fit succeeded.

    Deliberately a plain function over given axes, not a QWidget -- a
    caller arranges N of these into either a single pop-up (one spot, one
    hybe -- interactive 3D localization, see
    canvas/localize_3d_displayer.py) or a multi-hybe grid (one spot/
    allele, every hybe -- fiducial+readout spot-based alignment QC, the
    ChrTracer3 FitSpots reference grid this mirrors) without this function
    caring which; it only ever draws into whatever axes pair it's given.

    Expects ax_xz to be placed BELOW ax_yx by the caller, and BOTH created
    with sharex=True (e.g. fig.subplots(2, N, sharex='col') for an N-column
    grid) -- same locking canvas/pipeline_canvas.py's own _draw_three_way
    already uses for this exact YX/ZX-row-pair problem. sharex keeps the X
    range/column alignment locked between the two rows; aspect='auto' on
    BOTH (set here, not left to imshow's 'equal' default) is what actually
    makes the two rendered boxes the same width -- sharex alone only syncs
    the data range, not the box geometry, when aspect could otherwise
    differ per-axes (see that function's own comment on this).

    cubic: (height, width, depth) raw crop, this project's standard y,x,z
    layout (matches localization.cell_crop's own 'stacks' field and
    refine_spot_z's own cubic, which always carries the FULL Z depth --
    fitting deliberately searches the whole depth for robustness, see
    fit_gaussian_3d).
    centroid: a LIST of (x, y, z) crop-local coordinates, or None if this
    spot's own fit was rejected -- drawn as circles when given, omitted
    entirely when not (never a red X or other "failed" marker), matching
    ChrTracer3's own "circled = good, plain = missing" convention so a
    grid of these reads the same way by eye. FOUR STATES NOW, not
    three: `lateral` below added a WHITE DASHED ring for a spot whose z
    was never fitted, so "no circle" means no marker was ASKED FOR
    rather than "no fit" -- fitted against unfitted is carried by the
    ring's colour and dash, and a SOLID ring still means a fit. The FIRST entry is always
    this spot's own representative position -- the BRIGHTEST accepted
    component when the crop's fit found more than one real blob (yellow);
    any FURTHER entries are the other accepted components detected in the
    same crop (localization.refine_spot_z's mixture-fit path -- e.g. two
    real blobs a single click landed between), drawn blue for context
    only, since they're saved as mixture_centroids on the same spot
    rather than as separate spots. A bare (x, y, z) 3-tuple is also
    accepted, treated as a single-entry list.
    marker_size: scatter-marker size in POINTS^2 (screen space), not data
    units -- deliberately NOT a Circle patch (data-space radius), which
    renders as a squashed ellipse once aspect='auto' makes x/z pixels
    non-square; an unfilled scatter marker stays visually circular
    regardless of the axes' own data aspect.
    scale_half / scale_half_z: the grey scale's white point is taken
    from a box around THE MARKED SPOT -- +/-scale_half pixels laterally,
    +/-scale_half_z planes axially -- not from the whole crop. The box is
    ANISOTROPIC because the PSF is: this project measures sigma_xy 0.66
    px against sigma_z 2.35 planes, so a box that is square in pixels
    would be 3 sigma wide and under 1 sigma deep, and would keep letting
    a lateral neighbour's tail in while cutting off the spot's own
    z-wings.
    DISPLAY-ONLY, and the reason is that a 15x15 crop's ub=0.9999
    quantile is its own maximum for all practical purposes (225 pixels),
    so ONE bright neighbour inside the crop sets white and crushes the
    circled spot to near-black -- confirmed on a real grid, where the
    marked spot was invisible in several tiles while an off-centre blob
    was saturated. Scaling to the marked spot lets neighbours clip to
    white instead, which says "there is something brighter nearby"
    rather than "there is nothing here". The black point stays the
    crop's own lb quantile, so the noise floor still reads as noise.
    Falls back to whole-crop quantiles when there is no marker, or when
    the marked box is not brighter than that floor.
    lateral: (x, y) crop-local, or a list of them, for a spot whose Z
    was NEVER FITTED. Drawn WHITE and DASHED, and only on the YX panel,
    because "circled = fitted" is a convention the rest of this grid
    depends on -- the docstring for `centroid` above says no circle
    means no fit, and a yellow ring on an unfitted spot would quietly
    claim a measurement nobody made. Its y and x ARE measured, so the
    YX panel can say which blob is under discussion; its z is not, so
    the XZ panel gets no ring at all and is merely CENTRED on the
    brightest plane of that spot's own column -- which is an
    observation about the pixels, not a fit.
    z_display_pad: the XZ panel only shows +/-z_display_pad z-planes
    around the centroid (or the cubic's own brightest voxel if no
    centroid) -- DISPLAY-ONLY, never affects what fit_gaussian_3d actually
    searched. Confirmed on real data: an unrestricted full-depth XZ (e.g.
    11px wide x 177 z-planes for a real DNA hybe) renders as a near-1D
    sliver, unreadable even though the fit itself lands correctly.
    """
    centroids = None
    if centroid is not None:
        centroids = list(centroid) if isinstance(centroid, list) else [centroid]
    # rejected: fits that EXIST but were gate-rejected, drawn blue -- the
    # same colour as mixture-context components, and the same meaning: a
    # fit that is not a traced position. FOUR states per tile: yellow
    # solid = traced, blue solid = fitted but gated, white dashed (see
    # `lateral`) = a real spot whose z was never fitted, and no ring at
    # all = the caller named no position.
    rejected_list = None
    if rejected is not None:
        rejected_list = list(rejected) if isinstance(rejected, list) else [rejected]
    lateral_list = None
    if lateral is not None:
        lateral_list = list(lateral) if isinstance(lateral, list) else [lateral]

    depth = cubic.shape[2]
    if centroids:
        z_center = centroids[0][2]
    elif rejected_list:
        # centre the depth window on the rejected fit, or a rejected tile
        # shows a slab picked by the brightest voxel and the blue circle
        # can fall outside its own display window
        z_center = rejected_list[0][2]
    elif lateral_list:
        # THE SPOT'S OWN COLUMN, not the crop's. A spot with no fitted z
        # still has a measured y and x, and the brightest plane at THAT
        # (y, x) is where its own signal is -- centring on the crop's
        # global maximum would frame a neighbour instead and leave the
        # spot outside the window entirely.
        z_center = _column_peak_z(cubic, lateral_list[0][1],
                                  lateral_list[0][0])
    else:
        z_center = float(np.unravel_index(np.nanargmax(cubic), cubic.shape)[2])
    # CLAMPED INTO THE CROP, because z_center is not always inside it.
    # A stored z outlives a re-ingestion that produced FEWER planes, so
    # a spot fitted at plane 100 of a 120-plane stack lands past the end
    # of a 60-plane one. Unclamped that gives zmin >= zmax, an EMPTY
    # slab, and projecting an empty axis raises -- out of a dialog's
    # __init__, where it takes the whole dialog with it instead of
    # drawing a worse picture. A non-finite z has no plane to refer to
    # at all, so it goes to the middle.
    if not np.isfinite(z_center):
        z_center = (depth - 1) / 2.0
    z_center = min(max(float(z_center), 0.0), float(max(depth - 1, 0)))
    zmin = max(0, int(round(z_center)) - z_display_pad)
    zmax = min(depth, int(round(z_center)) + z_display_pad + 1)
    if zmax <= zmin:                       # a zero-depth cube, nothing else
        zmin, zmax = 0, depth

    # THE SCALE FOLLOWS THE MARKED SPOT. Which spot the picture is about
    # is known here -- it is the one that gets the yellow circle -- so
    # the white point comes from its own neighbourhood and a brighter
    # neighbour saturates instead of deciding the whole tile's exposure.
    if centroids:
        mark = centroids[0]
    elif rejected_list:
        mark = rejected_list[0]
    elif lateral_list:
        mark = (lateral_list[0][0], lateral_list[0][1], z_center)
    else:
        my, mx, mz = np.unravel_index(np.nanargmax(cubic), cubic.shape)
        mark = (float(mx), float(my), float(mz))
    yx_img = _project(cubic, 2)
    xz_img = _project(cubic[:, :, zmin:zmax], 0).T         # (z window, width)
    # EACH PANEL'S OWN BLACK POINT. The two are different pictures: a
    # max over 120 planes and a max over 15 rows do not share a noise
    # floor, and taking one quantile of the raw cube for both left the
    # YX panel's background sitting a sixth of the way up the grey
    # scale (MEASURED median 43/255 against 12/255 per-panel).
    yx = _scaled(yx_img, _floor(yx_img, lb),
                 _peak(yx_img, mark[1], mark[0], scale_half, scale_half),
                 lb, ub)
    xz = _scaled(xz_img, _floor(xz_img, lb),
                 _peak(xz_img, mark[2] - zmin, mark[0],
                       scale_half_z, scale_half), lb, ub)

    ax_yx.imshow(yx, cmap='gray', aspect='auto')
    ax_xz.imshow(xz, cmap='gray', aspect='auto')
    ax_yx.set_xticks([])
    ax_yx.set_yticks([])
    ax_xz.set_xticks([])
    ax_xz.set_yticks([])
    if title:
        ax_yx.set_title(title, fontsize=title_fontsize)

    if centroids:
        for i, (cx, cy, cz) in enumerate(centroids):
            color = 'yellow' if i == 0 else 'blue'
            marker_kwargs = dict(s=marker_size, marker='o', facecolors='none', edgecolors=color, linewidths=1.2)
            ax_yx.scatter([cx], [cy], **marker_kwargs)
            ax_xz.scatter([cx], [cz - zmin], **marker_kwargs)
    if rejected_list:
        for cx, cy, cz in rejected_list:
            marker_kwargs = dict(s=marker_size, marker='o', facecolors='none',
                                 edgecolors='deepskyblue', linewidths=1.2)
            ax_yx.scatter([cx], [cy], **marker_kwargs)
            ax_xz.scatter([cx], [cz - zmin], **marker_kwargs)
    if lateral_list:
        # YX ONLY, dashed. See the `lateral` paragraph above: this marks
        # WHICH blob is under discussion without claiming a z nobody
        # fitted, so the XZ panel deliberately gets no ring.
        for cx, cy in lateral_list:
            ax_yx.scatter([cx], [cy], s=marker_size, marker='o',
                          facecolors='none', edgecolors='white',
                          linewidths=1.2, linestyle='--')


def _column_peak_z(cube, y, x):
    """The brightest plane in one (y, x) column, or the crop's own peak.

    Falls back to the whole-crop argmax when the column is off the crop
    or holds nothing finite -- there is then no column to speak of.
    """
    a = np.asarray(cube, float)
    try:
        fy, fx = float(y), float(x)
        r, c = ((int(round(fy)), int(round(fx)))
                if np.isfinite(fy) and np.isfinite(fx) else (-1, -1))
    except (TypeError, ValueError):
        r = c = -1
    if 0 <= r < a.shape[0] and 0 <= c < a.shape[1]:
        col = a[r, c, :]
        if np.isfinite(col).any():
            return float(np.nanargmax(col))
    return float(np.unravel_index(np.nanargmax(a), a.shape)[2])


def _project(cube, axis):
    """A max projection that ignores a cell mask's NaN holes.

    Plain max propagates NaN, so a single masked row blanks an entire
    column of the projection -- and then there is no scale to take and
    nothing to look at.
    """
    a = np.asarray(cube, float)
    if a.dtype.kind == 'f' and np.isnan(a).any():
        return np.nanmax(a, axis=axis)
    return a.max(axis=axis)


def _floor(a, lb):
    """The crop's own black point: its lb quantile, NaN-tolerant."""
    a = np.asarray(a, float)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return None
    return float(np.quantile(finite, lb)) if lb < 1 else float(lb)


def _peak(img, row, col, half_row, half_col):
    """The brightest value in the box around (row, col), or None.

    None when the box lands entirely off the image or holds nothing
    finite -- the caller then falls back to whole-image quantiles rather
    than inventing a scale.

    A neighbour close enough to put its own tail inside this box DOES
    raise the white point, and that is not a defect: at 4 px the two
    PSFs overlap, and a scale that pretended otherwise would be drawing
    a spot the crop does not contain.
    """
    a = np.asarray(img, float)
    try:
        fr, fc = float(row), float(col)
    except (TypeError, ValueError):
        return None
    # FINITE FIRST. int(round(inf)) raises OverflowError, which this
    # caught neither as TypeError nor ValueError -- and an infinite
    # coordinate reaches here from a stored z the display clamps but
    # the MARKER keeps, so the guard has to live on both paths.
    if not (np.isfinite(fr) and np.isfinite(fc)):
        return None
    r, c = int(round(fr)), int(round(fc))
    r0, r1 = max(0, r - half_row), min(a.shape[0], r + half_row + 1)
    c0, c1 = max(0, c - half_col), min(a.shape[1], c + half_col + 1)
    if r1 <= r0 or c1 <= c0:
        return None
    box = a[r0:r1, c0:c1]
    box = box[np.isfinite(box)]
    return float(box.max()) if box.size else None


def _scaled(img, lo, hi, lb, ub):
    """img as uint8 on [lo, hi], or on its own quantiles if that fails.

    Values above hi CLIP to white on purpose: a neighbour brighter than
    the marked spot should read as present and saturated, not set the
    exposure for the tile.
    """
    if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi) \
            or hi <= lo:
        return preprocess.normalize_to_uint8(img, lb, ub)
    a = np.clip(np.asarray(img, float), lo, hi)
    a = np.where(np.isfinite(a), a, lo)
    return ((a - lo) / (hi - lo) * 255).astype(np.uint8)
