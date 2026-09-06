import os
import h5py
import numpy as np
from skimage import filters as skimage_filters, morphology as skimage_morphology, segmentation as skimage_segmentation
from skimage.feature import peak_local_max
from scipy import ndimage as scind
import warnings

from ..io import paths
from ..io import analysis_store
from ..io import stack_cache

warnings.filterwarnings("ignore", category=UserWarning, module="cellpose")

# Cellpose 4 (Cellpose-SAM) deleted the `models.Cellpose` class and the
# cyto3 weights along with it; 4.x offers `models.CellposeModel` and a
# SAM-backed checkpoint instead. Rather than pin this project to the
# superseded line, both are supported and the installed version decides.
# MEASURED against cellpose 4.2.1.1 + torch 2.7.1+cu118 on an RTX 3070
# (scratchpad probe, 2026-09-06): models.Cellpose is absent, CellposeModel
# loads in 16.3 s, uses 0.57 GiB of VRAM, and segments a 512x512 uint16
# field 25/25 correctly.
#
# 'cpsam' rather than the 4.2 default 'cpsam_v2': cpsam is the one name
# present in MODEL_NAMES across the whole 4.x line, and it is the
# checkpoint the behaviour above was actually verified on. Change it here,
# in one place, if a newer default is adopted -- and re-measure when you do,
# because it is a different model, not a faster one.
CELLPOSE4_MODEL = 'cpsam'

_cellpose_major_cached = None


def cellpose_major():
    """Major version of the installed Cellpose. Cached; the import is lazy.

    Cellpose costs seconds to import and pulls torch with it, so nothing
    here runs at module import time -- same reason the models themselves
    are lazy singletons.
    """
    global _cellpose_major_cached
    if _cellpose_major_cached is None:
        import cellpose
        _cellpose_major_cached = int(str(cellpose.version).split('.')[0])
    return _cellpose_major_cached


def make_cellpose_model(gpu=True):
    """Build a segmentation model from whichever Cellpose generation is installed.

    3.x -> models.Cellpose(model_type='cyto3'), the model every segmentation
           number recorded in this repository was measured with.
    4.x -> models.CellposeModel(pretrained_model=CELLPOSE4_MODEL).

    These are DIFFERENT MODELS, not two spellings of one. Masks, cell counts
    and therefore every downstream trace will differ between them. Which one
    ran is worth recording beside any result you intend to publish.
    """
    import cellpose.models
    if cellpose_major() >= 4:
        return cellpose.models.CellposeModel(gpu=gpu, pretrained_model=CELLPOSE4_MODEL)
    return cellpose.models.Cellpose(gpu=gpu, model_type='cyto3')


def cellpose_eval(model, images, diameter, channels):
    """eval() across both generations, returning the raw result tuple.

    `channels` selects the cytoplasm/nucleus planes in 3.x. Cellpose 4 takes
    images with arbitrary channel order and IGNORES the argument -- passing
    it earns one deprecation warning per call and nothing else, so it is
    dropped there rather than passed and swallowed. MEASURED on 4.2.1.1: the
    masks are bit-identical with and without it.

    The caller reads masks by POSITION (`result[0]`) because the tuple
    length differs -- 4 elements from 3.x's Cellpose, 3 from CellposeModel,
    the fourth being the estimated diameters 4.x no longer produces.
    """
    if cellpose_major() >= 4:
        return model.eval(images, diameter=diameter, do_3D=False)
    return model.eval(images, diameter=diameter, channels=channels, do_3D=False)


_model_cyto = None

def get_model_cyto():
    """Lazily load the segmentation model on first use, not at import time."""
    global _model_cyto
    if _model_cyto is None:
        _model_cyto = make_cellpose_model(gpu=True)
    return _model_cyto

def segment_fov(storage_path, fov, reference_hybe, channel, diameter=40, min_size=1000, max_size=10000,
                projection_mode='MIP (stored)', z_plane=None, z_range=None):
    """
    Bulk (non-interactive) cell segmentation for one FOV -- reads the
    reference MIP from vlinks.h5 (analysis_store.read_hybe_mip), a real copy
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
        raise ValueError(f'FOV{fov:03d} {reference_hybe} ch{channel} not ingested -- ingest it first.')

    # eval()'s return tuple length varies by cellpose version/model class
    # (3-tuple for CellposeModel/cpsam, 4-tuple for the classical
    # Cellpose/cyto3 class) -- take masks by position only, matching
    # CellClassifier/utils/cellpose_segmentation.py's approach.
    try:
        result = cellpose_eval(get_model_cyto(), [reference_image], diameter, [0, 0])
    except Exception:
        # masks_to_flows_gpu's boundary IndexError (MouseLand/cellpose#1004
        # and others) -- MPS in particular is a much less mature PyTorch
        # backend than CUDA, already hit here and independently in
        # CellClassifier/utils/cellpose_segmentation.py. Retry on CPU with a
        # fresh model instance rather than propagating; doesn't touch the
        # cached GPU singleton, so later calls still try GPU first.
        cpu_model = make_cellpose_model(gpu=False)
        result = cellpose_eval(cpu_model, [reference_image], diameter, [0, 0])
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
                          absolute_cutoff=None, min_distance=7, min_size=500, max_size=10000,
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
        raise ValueError(f'FOV{fov:03d} {reference_hybe} ch{channel} not ingested -- ingest it first.')

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
    h5path = paths.stack_path(storage_path, fov, hybe)
    shape = stack_cache.stack_shape(h5path, channel)
    if shape is None:
        return np.array([]), np.array([])
    height, width, depth, _slab = shape
    half = min(crop, height, width) // 2
    y0, y1 = height // 2 - half, height // 2 + half
    x0, x1 = width // 2 - half, width // 2 + half
    zs = list(range(0, depth, max(step, 1)))
    if not stack_cache.enabled():
        # Cache off: read only the central crop each plane needs. Going
        # through slab() here would inflate the FULL frame's chunks once
        # per plane with nothing retained -- measured 4x SLOWER than this
        # direct read (113 s vs 24.7 s per hybe on the real store).
        with h5py.File(h5path, 'r') as f:
            ds = f[f'/stack/ch{channel}']
            vals = [float(scind.laplace(ds[y0:y1, x0:x1, z].astype(np.float32)).var()) for z in zs]
        return np.array(zs), np.array(vals)
    # Through the slab cache: reading one plane inflates its whole chunk
    # slab anyway, so profiling every plane costs ceil(depth/slab) slab
    # reads instead of `depth` of them -- measured 24.7 s -> ~3 s per
    # hybe on the real store, and the slabs stay cached for the
    # plane-scrolling that usually follows. See io/stack_cache.py.
    vals = []
    for z in zs:
        got = stack_cache.slab(h5path, channel, z)
        if got is None:
            return np.array([]), np.array([])
        arr, s0 = got
        window = arr[y0:y1, x0:x1, z - s0].astype(np.float32)
        vals.append(float(scind.laplace(window).var()))
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
        return analysis_store.read_hybe_mip(storage_path, fov, hybe, channel)
    # Depth-resolved modes go through the slab cache: a single-plane read
    # measured 2.3-4.6 s on the real store (1024 gzip chunks inflated to
    # hand back one 2 MB plane), which is a per-tick freeze when the
    # Z-plane spinbox drives this. Cached, every other plane in the same
    # 64-plane slab is free. See io/stack_cache.py.
    h5path = paths.stack_path(storage_path, fov, hybe)
    shape = stack_cache.stack_shape(h5path, channel)
    if shape is None:
        return None
    depth = shape[2]
    if mode == 'single plane':
        z = int(np.clip(z_plane if z_plane is not None else depth // 2, 0, depth - 1))
        return stack_cache.plane(h5path, channel, z)
    z0, z1 = z_range if z_range else (0, depth - 1)
    sub = stack_cache.planes(h5path, channel, z0, z1)
    if sub is None:
        return None
    return sub.max(axis=2) if mode == 'range MIP' else sub.mean(axis=2)


def stack_depth(storage_path, fov, hybe, channel):
    """Number of z planes, so the UI can bound its own spinboxes to reality."""
    h5path = paths.stack_path(storage_path, fov, hybe)
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
    Lazily load a model for NUCLEUS-SEEDED cytoplasm segmentation. Separate
    singleton from get_model_cyto() only so the two call paths can't fight
    over one instance's internal state; the weights are the same.
    """
    global _model_cyto_nuc
    if _model_cyto_nuc is None:
        _model_cyto_nuc = make_cellpose_model(gpu=True)
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

    CELLPOSE 4 IGNORES `channels` BUT STILL USES THE NUCLEAR PLANE. What
    4.x drops is the ROLE DECLARATION, not the information: cyto3 has a
    dedicated nuclear input slot that `channels` fills, while cpsam takes
    up to three channels in arbitrary order and infers what they are. The
    array below is therefore built identically for both, and the seeding
    survives the version change -- do not "simplify" it away on 4.x.

    MEASURED, and by mask overlap rather than label count, because a label
    count cannot tell 40 correct cells from 40 wrong ones. Synthetic field
    at this function's own scale: cell d=60 px bodies overlapping so
    heavily that the cytoplasm channel alone cannot separate them, nuclei
    d=26 px clearly apart, 40 nuclei in 20 fused pairs. Scored as nuclei
    straddling two predicted masks / masks holding two nuclei:

                             split  merged  purity   median mask area
      cpsam, no nucleus        17      19    0.948       2590 px
      cpsam, WITH nucleus       0       0    1.000       2584 px
      cyto3, no nucleus         7      20    0.963       3137 px
      cyto3, WITH nucleus       0       0    1.000       2766 px

    (a whole cell is ~2827 px, a bare nucleus ~531 px, so the perfect arms
    are returning cells, not nuclei.) Both versions go from broken to exact
    when handed the nuclear plane.

    The "no nucleus" rows are a CONTROL, not a scenario. They exist to show
    the plane is what makes the difference; this function cannot reach that
    state, because `nucleus_seed_image` is required and the array is always
    built with it. Do not read them as a risk of running on 4.x.

    The real-world analogue of a bad seed is not an absent nuclear plane
    but a MISREGISTERED one -- the nucleus is projected in from its own
    hybe, which is often a different modality, so drift between that hybe
    and the cytoplasm hybe puts the seed in the wrong place. That is a
    property of the projection, not of the Cellpose version, and it is
    unmeasured.

    All of the above is synthetic besides. This route has never run on
    persisted production data under EITHER version, so nothing here is
    validated against a real cytoplasm.

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
        result = cellpose_eval(get_model_cyto_nuclear(), [rgb], diameter, [1, 2])
    except Exception:
        # Same GPU-backend fallback rationale as segment_fov's own retry.
        cpu_model = make_cellpose_model(gpu=False)
        result = cellpose_eval(cpu_model, [rgb], diameter, [1, 2])

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
