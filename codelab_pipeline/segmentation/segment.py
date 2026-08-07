import os
import numpy as np
import h5py
from skimage import filters as skimage_filters, morphology as skimage_morphology, segmentation as skimage_segmentation
from skimage.feature import peak_local_max
from scipy import ndimage as scind
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="cellpose")

_model_cyto = None

def get_model_cyto():
    """Lazily load the Cellpose cyto3 model on first use, not at import time."""
    global _model_cyto
    if _model_cyto is None:
        import cellpose.models
        _model_cyto = cellpose.models.Cellpose(gpu=True, model_type='cyto3')
    return _model_cyto

def segment_fov(storage_path, fov, reference_hybe, channel, diameter=40, min_size=1000, max_size=10000):
    """
    Bulk (non-interactive) cell segmentation for one FOV -- reads the
    reference MIP straight from that hybe's per-hybe H5 file (the current
    ingestion convention: {storage_path}/FOV{fov:02d}/{hybe}_stack.h5,
    /mip/ch{channel}), not the old vlinks.h5-based layout SegmentWidget used.
    Returns (mask, reference_image); doesn't display or save anything itself
    -- matches localize_cells_2d's separation of computation from I/O, so the
    GUI can run this off the main thread and review the result before saving.

    Core Cellpose-call + size-filter + relabel logic mirrors
    SegmentWidget.create_mask_in_reference_hybe below, minus its
    Jupyter/plotting/H5-write scaffolding.
    """
    h5path = os.path.join(storage_path, f'FOV{fov:02d}', f'{reference_hybe}_stack.h5')
    with h5py.File(h5path, 'r') as f:
        reference_image = f[f'/mip/ch{channel}'][:]

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
                          absolute_cutoff=None, min_distance=7, min_size=1000, max_size=10000):
    """
    Bulk (non-interactive) classical threshold+watershed cell segmentation
    for one FOV -- same I/O contract as segment_fov (reads
    {storage_path}/FOV{fov:02d}/{reference_hybe}_stack.h5's /mip/ch{channel},
    returns (mask, reference_image)). Ports
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
    h5path = os.path.join(storage_path, f'FOV{fov:02d}', f'{reference_hybe}_stack.h5')
    with h5py.File(h5path, 'r') as f:
        reference_image = f[f'/mip/ch{channel}'][:]

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
