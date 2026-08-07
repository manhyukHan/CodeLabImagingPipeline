import numpy as np

"""
Celltype determination: two modes, ported from CellClassifier
(models/spot_metadata_analyzer.py) as plain functions -- no I/O, no Qt,
matches codelab_pipeline/alignment/chain.py's convention of pure
computation functions with the display/plotting scaffolding left out.

FOV mode: a static {fov: celltype} lookup built once from user-entered
FOV-range strings (e.g. "1-46, 50-93" -> FOVs 1..46 and 50..93 inclusive).
CellClassifier's own range parser used `range(st, end)`, excluding the
range's own right endpoint -- a real off-by-one that silently drops the
last FOV of every range (e.g. "1-46" never assigned FOV 46). Fixed here
via range(st, end + 1), inclusive on both ends, matching what a human
typing "1-46" actually means.

Barcode mode: classify_cell_barcode/classify_spot_barcode port
CellClassifier's 200-random-pixel majority-vote algorithm, with one
REQUIRED (not cosmetic) correctness fix, not just a naming adaptation:
CellClassifier assumes every barcode channel is read from one shared,
already-co-registered mosaic frame, so one shared (x, y) sample index is
valid across every channel. This project has no such assumption -- a
"barcode channel" here is a specific readout hybe on a specific physical
channel, and different hybes can carry independent stage drift (that's
the entire reason the alignment subsystem exists). Reusing one shared
raw-frame (x, y) across channels from different hybes would silently mix
coordinate frames -- exactly the class of bug the FOV/cross-modal/cell
alignment-chain fixes earlier in this project were about. So both
functions take PER-CHANNEL, already-frame-correct coordinates (the caller
transforms once via ACell.matrix_to / spot_mapper, which relate a hybe's
own entry back to cell.reference_hybe's frame -- see compute_cell_
alignment's docstring for why that's not a direct cell.matrices lookup)
instead of one shared x, y reused blindly across channels.
"""


def build_celltype_from_fov_ranges(range_strings_by_celltype):
    """
    {celltype(str): "1-46, 50-93"} -> ({fov(int): celltype(str)}, {celltype(str): [fov(int), ...]}).
    Each range string is a comma-separated list of "start-end" (inclusive
    both ends) or single FOV numbers.
    """
    celltype_from_fov = {}
    fov_from_celltype = {}
    for celltype, range_string in range_strings_by_celltype.items():
        fovs = []
        for chunk in range_string.split(','):
            chunk = chunk.strip()
            if not chunk:
                continue
            if '-' in chunk:
                st, end = chunk.split('-')
                fovs.extend(range(int(st.strip()), int(end.strip()) + 1))
            else:
                fovs.append(int(chunk))
        fov_from_celltype[celltype] = fovs
        for fov in fovs:
            celltype_from_fov[fov] = celltype
    return celltype_from_fov, fov_from_celltype


def classify_fov(fov, celltype_from_fov):
    """FOV-mode celltype lookup for one cell/spot -- '' if this FOV isn't covered by any defined range."""
    return celltype_from_fov.get(int(fov), '')


def classify_cell_barcode(area_by_channel, fov, images, celltype_determination, n_samples=200, method='vote'):
    """
    method='vote' (default): sample n_samples random point-indices (with
    replacement), normalize each barcode (hybe, channel)'s intensity at
    that sample via that channel's per-FOV (scale, lower_bound,
    upper_bound), argmax across barcode channels per sample (or '' if
    every normalized value is <=0), then majority-vote the per-sample
    winners. Fully vectorized with numpy fancy indexing -- an earlier
    version of this function did the exact same math inside a pure-Python
    loop over all n_samples * n_channels points, which was the actual,
    confirmed cause of "Run Celltype Determination takes too long" on a
    real dataset (O(n_cells * 200 * n_channels) Python-level iterations).
    Vectorizing preserves identical statistics -- same formula, same
    random sampling, same majority vote -- just without the interpreter
    overhead.

    method='median': normalize the MEDIAN intensity over the cell's
    entire area per channel, then argmax once (no sampling, no vote) --
    a simpler, even faster alternative per explicit request, trading the
    vote's robustness-to-a-few-noisy-pixels for a single deterministic
    per-cell value.

    area_by_channel: {(hybe, channel): (x_array, y_array)} -- every array
    the SAME length and SAME point order (index i is the same physical
    point in every channel's array, just expressed in that channel's own
    hybe-native pixel frame). The caller builds this once per cell, e.g.
    by applying cell.matrix_to(hybe, modality) (inverted) to cell.area for
    each distinct hybe among the barcode channels -- a plain per-point
    matrix multiply, deliberately NOT cell.get_area_in_readout's masking/
    morphological-closing machinery, which can change point count/order
    and would break the "same index i across channels" guarantee this
    function depends on.
    images: {fov(str): {(hybe,channel): ndarray}}.
    celltype_determination: {'barcode': {'barcode_channel': [(hybe,channel),...],
    'barcode_celltype': [celltype(str),...], 'scale'/'lower_bound'/'upper_bound':
    {(hybe,channel): {fov(int): float}}}}. barcode_channel and barcode_celltype
    are index-paired -- argmax position i means barcode_celltype[i].
    """
    barcode = celltype_determination['barcode']
    barcode_channel = barcode['barcode_channel']
    barcode_celltype = barcode['barcode_celltype']
    if not barcode_channel:
        return ''
    n_points = len(area_by_channel[barcode_channel[0]][0])
    if n_points == 0:
        return ''

    if method == 'median':
        values = np.empty(len(barcode_channel))
        for j, bch in enumerate(barcode_channel):
            x_arr, y_arr = area_by_channel[bch]
            xi = np.round(x_arr).astype(int)
            yi = np.round(y_arr).astype(int)
            raw = np.median(images[str(fov)][bch][yi, xi])
            scaled = barcode['scale'][bch][int(fov)] * raw
            lo, hi = barcode['lower_bound'][bch][int(fov)], barcode['upper_bound'][bch][int(fov)]
            values[j] = (scaled - lo) / (hi - lo)
        return barcode_celltype[int(np.argmax(values))] if values.max() > 0 else ''

    rand_ids = np.random.choice(n_points, n_samples)
    values = np.empty((n_samples, len(barcode_channel)))
    for j, bch in enumerate(barcode_channel):
        x_arr, y_arr = area_by_channel[bch]
        xi = np.round(x_arr[rand_ids]).astype(int)
        yi = np.round(y_arr[rand_ids]).astype(int)
        raw = images[str(fov)][bch][yi, xi]
        scaled = barcode['scale'][bch][int(fov)] * raw
        lo, hi = barcode['lower_bound'][bch][int(fov)], barcode['upper_bound'][bch][int(fov)]
        values[:, j] = (scaled - lo) / (hi - lo)

    winners = np.argmax(values, axis=1)
    cts = np.array([barcode_celltype[w] for w in winners])
    cts[values.max(axis=1) <= 0] = ''

    vals, counts = np.unique(cts, return_counts=True)
    return vals[np.argmax(counts)]


def classify_spot_barcode(xy_by_channel, fov, images, celltype_determination):
    """
    Single-point variant of classify_cell_barcode's inner per-sample
    formula -- one physical point, no majority vote needed.

    xy_by_channel: {(hybe, channel): (x, y)} -- this spot's position,
    already transformed into EACH barcode channel's own hybe-native frame
    (via spot_mapper.reference_to_raw(spot.coordinate[:2], hybe,
    fov_matrices, cell=owning_cell) per channel -- the caller's job, same
    reasoning as classify_cell_barcode above).
    """
    barcode = celltype_determination['barcode']
    barcode_channel = barcode['barcode_channel']
    barcode_celltype = barcode['barcode_celltype']

    values = []
    for bch in barcode_channel:
        x, y = xy_by_channel[bch]
        raw = images[str(fov)][bch][int(round(y)), int(round(x))]
        scaled = barcode['scale'][bch][int(fov)] * raw
        lo, hi = barcode['lower_bound'][bch][int(fov)], barcode['upper_bound'][bch][int(fov)]
        values.append((scaled - lo) / (hi - lo))
    return barcode_celltype[int(np.argmax(values))] if np.max(values) > 0 else ''
