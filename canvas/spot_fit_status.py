import numpy as np

from codelab_pipeline.io import preprocess


def draw_spot_fit_status(ax_yx, ax_xz, cubic, centroid=None, lb=0.3, ub=0.9999, title='',
                         marker_size=130, z_display_pad=15, title_fontsize=9):
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
    layout (matches localization._build_cell_crop's own 'stacks' field and
    refine_spot_z's own cubic, which always carries the FULL Z depth --
    fitting deliberately searches the whole depth for robustness, see
    fit_gaussian_3d).
    centroid: a LIST of (x, y, z) crop-local coordinates, or None if this
    spot's own fit was rejected -- drawn as circles when given, omitted
    entirely when not (never a red X or other "failed" marker), matching
    ChrTracer3's own "circled = good, plain = missing" convention so a
    grid of these reads the same way by eye. The FIRST entry is always
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
    z_display_pad: the XZ panel only shows +/-z_display_pad z-planes
    around the centroid (or the cubic's own brightest voxel if no
    centroid) -- DISPLAY-ONLY, never affects what fit_gaussian_3d actually
    searched. Confirmed on real data: an unrestricted full-depth XZ (e.g.
    11px wide x 177 z-planes for a real DNA hybe) renders as a near-1D
    sliver, unreadable even though the fit itself lands correctly.
    """
    yx = preprocess.normalize_to_uint8(cubic.max(axis=2), lb, ub)

    centroids = None
    if centroid is not None:
        centroids = list(centroid) if isinstance(centroid, list) else [centroid]

    depth = cubic.shape[2]
    z_center = centroids[0][2] if centroids else float(np.unravel_index(np.nanargmax(cubic), cubic.shape)[2])
    zmin = max(0, int(round(z_center)) - z_display_pad)
    zmax = min(depth, int(round(z_center)) + z_display_pad + 1)
    xz = preprocess.normalize_to_uint8(cubic[:, :, zmin:zmax].max(axis=0).T, lb, ub)  # (z window, width)

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
