import os
import h5py
import numpy as np
from skimage import filters as skimage_filters, morphology as skimage_morphology, segmentation as skimage_segmentation
from skimage.feature import peak_local_max
from scipy import ndimage as scind
import warnings

from ..io import vlinks_store

warnings.filterwarnings("ignore", category=UserWarning, module="cellpose")

_model_cyto = None

def get_model_cyto():
    """Lazily load the Cellpose cyto3 model on first use, not at import time."""
    global _model_cyto
    if _model_cyto is None:
        import cellpose.models
        _model_cyto = cellpose.models.Cellpose(gpu=True, model_type='cyto3')
    return _model_cyto

def segment_fov(storage_path, fov, reference_hybe, channel, diameter=40, min_size=1000, max_size=10000,
                projection_mode='MIP (stored)', z_plane=None, z_range=None):
    """
    Bulk (non-interactive) cell segmentation for one FOV -- reads the
    reference MIP from vlinks.h5 (vlinks_store.read_hybe_mip), a real copy
    written by ingestion, not the raw per-hybe {hybe}_stack.h5 -- per
    explicit principle, segmentation is display/2D-analysis, not ingestion
    or 3D localization, so it should never need the raw stack file.
    Returns (mask, reference_image); doesn't display or save anything itself
    -- matches localize_cells_2d's separation of computation from I/O, so the
    GUI can run this off the main thread and review the result before saving.

    Core Cellpose-call + size-filter + relabel logic mirrors
    legacy/segment_widgets.py's SegmentWidget.create_mask_in_reference_hybe,
    minus its Jupyter/plotting/H5-write scaffolding.
    """
    # projection_mode defaults to the stored MIP, so existing behaviour is
    # untouched. The depth-resolved modes are here for the same reason
    # cytoplasm search needed them: on a brightfield-like stack a full-depth
    # MIP piles every plane's halo on top of the boundaries you are trying
    # to segment. See read_projection.
    reference_image = read_projection(storage_path, fov, reference_hybe, channel,
                                      mode=projection_mode, z_plane=z_plane, z_range=z_range)
    if reference_image is None:
        raise ValueError(f'FOV{fov:02d} {reference_hybe} ch{channel} not in vlinks.h5 -- ingest it first.')

    # eval()'s return tuple length varies by cellpose version/model class
    # (3-tuple for CellposeModel/cpsam, 4-tuple for the classical
    # Cellpose/cyto3 class used here) -- take masks by position only,
    # matching CellClassifier/utils/cellpose_segmentation.py's approach.
    try:
        result = get_model_cyto().eval([reference_image], diameter=diameter, channels=[0, 0], do_3D=False)
    except Exception:
        # masks_to_flows_gpu's boundary IndexError (MouseLand/cellpose#1004
        # and others) -- MPS in particular is a much less mature PyTorch
        # backend than CUDA, already hit here and independently in
        # CellClassifier/utils/cellpose_segmentation.py. Retry on CPU with a
        # fresh model instance rather than propagating; doesn't touch the
        # cached GPU singleton, so later calls still try GPU first.
        import cellpose.models
        cpu_model = cellpose.models.Cellpose(gpu=False, model_type='cyto3')
        result = cpu_model.eval([reference_image], diameter=diameter, channels=[0, 0], do_3D=False)
    masks = result[0]
    mask = masks[0].astype(float)
    mask = _filter_and_relabel(mask, min_size, max_size)

    return mask, reference_image


def _filter_and_relabel(mask, min_size, max_size):
    """
    Shared by every segmentation method (Cellpose, classical, and manual's
    own additive commits use the same convention): drop labels outside
    [min_size, max_size] by area, then relabel sequentially starting at 1
    (0 stays background). Starting enumerate() at 0 (as the legacy
    SegmentWidget.create_mask_in_reference_hybe below does) maps the first
    valid cell to -0 == 0 and silently drops it into the background --
    fixed here via start=1, not propagated.
    """
    v, c = np.unique(mask, return_counts=True)
    mask = mask.copy()
    mask[np.isin(mask, v[c < min_size])] = 0
    mask[np.isin(mask, v[c > max_size])] = 0
    for i, id in enumerate(np.unique(mask)[1:], start=1):
        mask[mask == id] = -i
    return (-1 * mask).astype(np.uint8)


def segment_fov_classical(storage_path, fov, reference_hybe, channel, method='otsu',
                          absolute_cutoff=None, min_distance=7, min_size=1000, max_size=10000,
                          projection_mode='MIP (stored)', z_plane=None, z_range=None):
    """
    Bulk (non-interactive) classical threshold+watershed cell segmentation
    for one FOV -- same I/O contract as segment_fov (reads reference_hybe's
    MIP from vlinks.h5, returns (mask, reference_image)). Ports
    CellClassifier/canvas/main_image_canvas.py::_runCellSegment's classical
    branch (method in {'otsu','yen','li','triangle','manual'}), minus its
    pyqtgraph/live-canvas coupling. 'manual' there meant "type an absolute
    intensity cutoff by hand" (unrelated to this project's own polygon-draw
    "Manual" segmentation method) -- renamed here to method='absolute' with
    an explicit absolute_cutoff value, to avoid that name collision.

    CellClassifier's original call used
    peak_local_max(distance_transformed, indices=False, min_distance=...),
    an indices=False boolean-mask return mode removed from the skimage
    version already used elsewhere in this project (localization.py never
    passes indices=). The boolean marker mask is rebuilt manually from the
    returned peak coordinates instead.
    """
    reference_image = read_projection(storage_path, fov, reference_hybe, channel,
                                      mode=projection_mode, z_plane=z_plane, z_range=z_range)
    if reference_image is None:
        raise ValueError(f'FOV{fov:02d} {reference_hybe} ch{channel} not in vlinks.h5 -- ingest it first.')

    if method == 'absolute':
        cutoff = float(absolute_cutoff)
    elif method == 'yen':
        cutoff = skimage_filters.threshold_yen(reference_image)
    elif method == 'otsu':
        cutoff = skimage_filters.threshold_otsu(reference_image)
    elif method == 'triangle':
        cutoff = skimage_filters.threshold_triangle(reference_image)
    elif method == 'li':
        cutoff = skimage_filters.threshold_li(reference_image)
    else:
        raise ValueError(f"Unknown classical segmentation method '{method}'")

    binary = (reference_image > cutoff).astype(np.uint8)
    opened_binary = skimage_morphology.opening(binary)
    distance_transformed = scind.distance_transform_edt(opened_binary)

    peak_coords = peak_local_max(distance_transformed, min_distance=min_distance, labels=opened_binary)
    local_max = np.zeros_like(distance_transformed, dtype=bool)
    local_max[tuple(peak_coords.T)] = True
    markers, _ = scind.label(local_max, structure=np.ones((3, 3)))
    mask = skimage_segmentation.watershed(-distance_transformed, markers, mask=opened_binary).astype(float)
    mask = _filter_and_relabel(mask, min_size, max_size)

    return mask, reference_image


PROJECTION_MODES = ('MIP (stored)', 'single plane', 'range MIP', 'range mean')


def describe_projection(mode, z_plane=None, z_range=None):
    """
    Short human-readable form of a projection choice, e.g.
    'MIP (stored)' / 'single plane z=76' / 'range MIP z=69-80'.

    One formatter so a Run button's label, its confirmation dialog and the
    log line can never disagree about what is about to happen -- which
    matters here because the choice is not cosmetic: on real data the same
    FOV/hybe/channel/parameters yielded 33 cells from the stored MIP versus
    91 from single plane z=76.
    """
    if mode == 'single plane':
        return f'{mode} z={z_plane}'
    if mode in ('range MIP', 'range mean'):
        z0, z1 = z_range if z_range else (None, None)
        return f'{mode} z={z0}-{z1}'
    return mode


def focus_profile(storage_path, fov, hybe, channel, step=1, crop=512):
    """
    (z_indices, sharpness) for every step-th plane of this hybe's raw
    Z-stack -- variance of the Laplacian on a central `crop`-square window,
    the standard passive autofocus metric.

    Exists because "the middle plane is the focal plane" is an ASSUMPTION
    that measurably fails: on FOV01/Hyb_500/ch635 the stack is 130 planes,
    so the middle is z=65, but the sharpest plane is z=76 -- off by 11.
    Detecting the peak removes the assumption instead of replacing it with
    a manual guess.

    Reads {hybe}_stack.h5 directly. That is deliberate and is the same
    documented exception hybe_zx_projection already takes: the
    standing principle is that
    MIP-ONLY reads never need the raw stack, and a depth-resolved
    projection is by definition not a MIP-only read. Central crop + `step`
    keep it cheap (h5py slices per plane, never materializing the stack).
    """
    h5path = os.path.join(storage_path, f'FOV{fov:02d}', f'{hybe}_stack.h5')
    with h5py.File(h5path, 'r') as f:
        ds = f[f'/stack/ch{channel}']
        height, width, depth = ds.shape
        half = min(crop, height, width) // 2
        y0, y1 = height // 2 - half, height // 2 + half
        x0, x1 = width // 2 - half, width // 2 + half
        zs = list(range(0, depth, max(step, 1)))
        vals = [float(scind.laplace(ds[y0:y1, x0:x1, z].astype(np.float32)).var()) for z in zs]
    return np.array(zs), np.array(vals)


def read_projection(storage_path, fov, hybe, channel, mode='MIP (stored)', z_plane=None, z_range=None):
    """
    The 2D image to run cytoplasm segmentation on, per PROJECTION_MODES.

    'MIP (stored)' returns vlinks.h5's own MIP -- unchanged default, and
    the only mode that touches no raw stack. The others read
    {hybe}_stack.h5 (see focus_profile on why that exception applies).

    A max projection over the FULL depth accumulates the brightest halo
    from every plane, which on real brightfield data visibly destroys the
    phase contrast that defines a cell boundary -- which is the whole
    reason the depth-resolved modes exist. 'range MIP'/'range mean' over
    the focus plateau are usually more robust than a single plane, because
    focus varies ACROSS the field (curvature/tilt) and focus_profile's own
    metric only samples the centre.
    """
    if mode == 'MIP (stored)':
        return vlinks_store.read_hybe_mip(storage_path, fov, hybe, channel)
    h5path = os.path.join(storage_path, f'FOV{fov:02d}', f'{hybe}_stack.h5')
    with h5py.File(h5path, 'r') as f:
        ds = f[f'/stack/ch{channel}']
        depth = ds.shape[2]
        if mode == 'single plane':
            z = int(np.clip(z_plane if z_plane is not None else depth // 2, 0, depth - 1))
            return ds[:, :, z].astype(np.float32)
        z0, z1 = z_range if z_range else (0, depth - 1)
        z0, z1 = int(np.clip(z0, 0, depth - 1)), int(np.clip(z1, 0, depth - 1))
        if z1 < z0:
            z0, z1 = z1, z0
        sub = ds[:, :, z0:z1 + 1].astype(np.float32)
    return sub.max(axis=2) if mode == 'range MIP' else sub.mean(axis=2)


def stack_depth(storage_path, fov, hybe, channel):
    """Number of z planes, so the UI can bound its own spinboxes to reality."""
    h5path = os.path.join(storage_path, f'FOV{fov:02d}', f'{hybe}_stack.h5')
    if not os.path.exists(h5path):
        return 0
    with h5py.File(h5path, 'r') as f:
        key = f'/stack/ch{channel}'
        return f[key].shape[2] if key in f else 0


SEED_MODES = ('rim emphasis', 'seam-separated', 'binary')


def render_nucleus_seed(label_mask, mode='rim emphasis'):
    """
    Turn a nucleus LABEL mask into the synthetic nuclear channel cellpose
    is seeded with (see segment_cytoplasm on why a real image, not labels).

    Never encodes cell ids as intensity: cellpose normalizes this channel
    and reads it as a stain, so id-valued pixels would become a meaningless
    brightness ramp (cell 67 sixty-seven times brighter than cell 1) and the
    low-id nuclei would normalize into the background. Instance separation
    comes from cellpose's own predicted flows, i.e. from SHAPE and gaps --
    which is what every mode below manipulates.

    Modes, measured on real data (FOV01, 2x50 random nuclei, Hyb_500 BF):
      'rim emphasis'   -- dim body + bright 1px outline. 100/100 nuclei
        recovered, the only mode that lost none; median cytoplasm/nucleus
        area 1.50. The bright rim reads as an edge and pulls the predicted
        boundary slightly inward, which is the price for that recovery.
      'seam-separated' -- plain binary, except the 1px seam where two
        DIFFERENT labels touch is cut to background. 99/100 recovered with
        the largest cytoplasm extent (1.57). Most conservative: isolated
        nuclei are untouched.
      'binary'         -- the naive fill. 97/100, extent 1.56. Kept for
        comparison; its weakness is real and measured -- projecting 50
        nuclei fused them into 45 and 40 connected blobs in two trials, and
        a fused nucleus loses its cytoplasm to whichever neighbour claims
        the merged blob.

    Plain erosion was also tested and is WORSE than doing nothing (1px:
    94/100, 2px: 96/100) -- it shrinks every nucleus including the isolated
    ones that had no contact, while often not breaking a real contact.

    Returns a float32 image in [0, 1]; the caller scales it to the
    cytoplasm image's own dynamic range.
    """
    body = label_mask > 0
    if mode == 'binary':
        return body.astype(np.float32)

    # Where a pixel's 3x3 neighbourhood holds two DIFFERENT nonzero labels,
    # i.e. exactly the inter-nucleus contacts (never a nucleus/background
    # edge, which min/max agree on).
    big = np.where(body, label_mask, np.iinfo(np.int32).max)
    contact = body & (scind.maximum_filter(label_mask, size=3) != scind.minimum_filter(big, size=3))

    if mode == 'seam-separated':
        return (body & ~contact).astype(np.float32)

    out = body.astype(np.float32) * 0.45
    outer = body & (scind.minimum_filter(body.astype(np.uint8), size=3) == 0)
    out[outer | contact] = 1.0
    return out


_model_cyto_nuc = None


def get_model_cyto_nuclear():
    """
    Lazily load a Cellpose cyto3 model for NUCLEUS-SEEDED cytoplasm
    segmentation. Separate singleton from get_model_cyto() only so the two
    call paths can't fight over one instance's internal state; the weights
    are the same cyto3 model.
    """
    global _model_cyto_nuc
    if _model_cyto_nuc is None:
        import cellpose.models
        _model_cyto_nuc = cellpose.models.Cellpose(gpu=True, model_type='cyto3')
    return _model_cyto_nuc


def segment_cytoplasm(cyto_image, nucleus_seed_image, diameter=60, min_size=1000, max_size=100000):
    """
    Cellpose cytoplasm segmentation SEEDED by a nuclear channel.

    Cellpose has no API that takes a label mask as a seed -- its nucleus-
    assisted mode takes a second IMAGE channel (channels=[cyto, nuc]). So
    `nucleus_seed_image` is a SYNTHETIC nuclear image the caller renders
    from the real, already-segmented nuclei (see MainWindow._build_nucleus_
    seed_image), projected into cyto_image's own frame. That synthetic
    channel is genuinely stitched: each cell's nucleus is projected from
    ITS OWN nucleus_hybe, which can differ cell to cell.

    Stacked into an explicit 3-channel RGB-like array (R=cytoplasm,
    G=nucleus, B=0) with channels=[1, 2] rather than a bare 2-channel
    array -- cellpose's channel indices are 1-based into RGB, and the
    (H,W,2) form is ambiguous across versions.

    Returns cellpose's own raw labels, deliberately NOT relabeled: the
    caller has to match them back to real nucleus ids (see
    incorporate_cytoplasm), and _filter_and_relabel's renumbering would
    destroy exactly the correspondence that matching depends on. Size
    filtering here therefore drops labels in place, keeping ids intact.
    """
    cyto = np.asarray(cyto_image, dtype=np.float32)
    nuc = np.asarray(nucleus_seed_image, dtype=np.float32)
    if cyto.shape != nuc.shape:
        raise ValueError(f'cyto image {cyto.shape} and nucleus seed {nuc.shape} must share a frame')
    rgb = np.zeros((*cyto.shape, 3), dtype=np.float32)
    rgb[..., 0] = cyto
    rgb[..., 1] = nuc

    try:
        result = get_model_cyto_nuclear().eval([rgb], diameter=diameter, channels=[1, 2], do_3D=False)
    except Exception:
        # Same GPU-backend fallback rationale as segment_fov's own retry.
        import cellpose.models
        cpu_model = cellpose.models.Cellpose(gpu=False, model_type='cyto3')
        result = cpu_model.eval([rgb], diameter=diameter, channels=[1, 2], do_3D=False)

    labels = np.asarray(result[0][0]).astype(np.int32)
    return _drop_labels_by_size(labels, min_size, max_size)


def _drop_labels_by_size(labels, min_size, max_size):
    """
    Size filter that PRESERVES label values -- unlike _filter_and_relabel,
    which renumbers. Cytoplasm labels have to keep their identity until
    they've been matched to nuclei.
    """
    out = labels.copy()
    values, counts = np.unique(out, return_counts=True)
    bad = values[(counts < min_size) | (counts > max_size)]
    out[np.isin(out, bad[bad != 0])] = 0
    return out


def incorporate_cytoplasm(cyto_labels, nucleus_label_mask, eligible_ids=None):
    """
    Merge a raw cytoplasm label image into the cell-id label space.

    cyto_labels: segment_cytoplasm's own output (arbitrary cellpose ids).
    nucleus_label_mask: EVERY cell's nucleus painted with its real cell id,
    already projected into cyto_labels' frame -- including cells that opted
    OUT of the cytoplasm search, because those still have to win overlaps.

    eligible_ids: the cells that were actually SELECTED as seeds. Only
    these may claim a cytoplasm; None means "all of them". This is a
    genuinely separate role from the mask above, and conflating the two
    was a real bug caught on live data: matching against every nucleus let
    unselected cells claim cytoplasms cellpose happened to grow around
    them (66 of 67 cells came back with a cytoplasm from a 50-cell
    selection). Overlap authority is global; claiming authority is not.

    Rules, per explicit spec:
      * a cytoplasm inherits the id of the nucleus inside it -- so no cell
        id is created, renamed, or renumbered by this step;
      * a cytoplasm containing NO nucleus is discarded (cellpose readily
        invents these);
      * if several nuclei fall inside one cytoplasm, the largest-overlap
        nucleus claims it and the others keep only their own nucleus --
        never a split, which would have to invent an id;
      * if one nucleus is claimed by several cytoplasms, it keeps the
        largest-overlap one;
      * nuclei always win overlaps: every nucleus pixel is painted last,
        so a cytoplasm can never eat into another cell's nucleus.

    Returns (merged_mask, claimed) -- claimed is {cell_id: n_pixels} for the
    cells that actually ended up with a real cytoplasm, so the caller can
    report/skip the rest without re-deriving it.
    """
    merged = np.zeros_like(nucleus_label_mask, dtype=np.int32)
    eligible = None if eligible_ids is None else {int(i) for i in eligible_ids}
    best_for_nucleus = {}  # cell id -> (overlap, cyto label)
    for cyto_id in np.unique(cyto_labels):
        if cyto_id == 0:
            continue
        inside = nucleus_label_mask[cyto_labels == cyto_id]
        inside = inside[inside != 0]
        if eligible is not None:
            inside = inside[np.isin(inside, list(eligible))] if inside.size else inside
        if inside.size == 0:
            continue  # no ELIGIBLE nucleus inside -- discard outright
        ids, counts = np.unique(inside, return_counts=True)
        winner, overlap = int(ids[np.argmax(counts)]), int(counts.max())
        prior = best_for_nucleus.get(winner)
        if prior is None or overlap > prior[0]:
            best_for_nucleus[winner] = (overlap, cyto_id)

    claimed = {}
    for cell_id, (_, cyto_id) in best_for_nucleus.items():
        region = cyto_labels == cyto_id
        merged[region] = cell_id
        claimed[cell_id] = int(region.sum())

    # Nuclei last: unconditional, so an unselected cell's nucleus also
    # carves itself back out of any cytoplasm that overlapped it.
    nucleus_pixels = nucleus_label_mask != 0
    merged[nucleus_pixels] = nucleus_label_mask[nucleus_pixels]
    return merged, claimed
