"""
Per-cell expression estimation, spot-based AND mask-based, with the
normalizations the condition system gates on.

A "source" everywhere in this package is the triple (modality, hybe,
channel) -- the same key the spot store slices by, and the only
unambiguous name for a measured signal (the bridge hybe exists in two
modalities; a bare hybe name is not an identity).

Spot-based metrics come from the stored detections: count, median and
total of ASpot.brightness (raw MIP counts at detection; ASpot.size is a
dead field and is deliberately not offered). Mask-based intensity is the
median MIP value over the CELL'S OWN mask -- defined for every cell,
including cells with zero detected spots, which is exactly the case
spot-based metrics cannot see.

Mask frame honesty: the exact projection of a cell mask into the source
hybe's raw frame needs the alignment chain. When the cell's stored
matrices can compose it (matrix_to), it is used; when they are residual-
form (post cell-alignment), the reference-frame mask is used unprojected
and the row says so in mask_frame -- 'native' vs 'reference'. Callers
with a FrameResolver (the app) can pass it for exact projection in every
case. Alignment offsets are a few px against ~50 px cell masks, so the
fallback is a small, FLAGGED approximation, never a silent one.
"""
import os

import numpy as np
import pandas as pd

from codelab_pipeline.io import analysis_store
from codelab_pipeline.models.cell import ACell

DEFAULT_VOXEL_UM = (0.208, 0.208, 0.2)


def fov_expression_table(storage_path, fov, sources, mask_intensity=False,
                         resolver=None):
    """Tidy long expression rows for one FOV.

    sources: [(modality, hybe, channel), ...]. One row per (cell x
    source): fov, cell, celltype, modality, hybe, channel, n_spots,
    brightness_median, brightness_total, and -- when mask_intensity --
    mask_median, mask_frame. Homeless spots (cell == -1) are excluded
    from per-cell rows; their count is returned per source in the
    companion dict so nothing disappears silently.

    Returns (DataFrame, {'homeless': {source: n}}).
    """
    cells, _ = analysis_store.read_cells(storage_path, fov)
    cells = cells or []
    rows, homeless = [], {}
    cell_objs = {}
    if mask_intensity:
        for c in cells:
            obj = ACell()
            obj.set_metadata(**c)
            cell_objs[int(c['id'])] = obj
    for src in sources:
        modality, hybe, channel = src
        spots = analysis_store.read_spots(storage_path, fov, modality=modality,
                                          hybe=hybe, channel=channel)
        by_cell = {}
        n_homeless = 0
        for s in spots:
            cid = int(s.get('cell', -1))
            if cid < 0:
                n_homeless += 1
                continue
            by_cell.setdefault(cid, []).append(float(s.get('brightness', np.nan)))
        homeless[src] = n_homeless
        mip = None
        if mask_intensity:
            # The MIP must come from the SOURCE's modality tree -- a bare
            # (storage_path, hybe) read would resolve an RNA source
            # against the DNA tree whenever the population was built from
            # the DNA storage_path (the bridge hybe exists in both).
            mip_sp = os.path.join(os.path.dirname(
                os.path.normpath(storage_path)), modality)
            mip = analysis_store.read_hybe_mip(mip_sp, fov, hybe,
                                               int(channel))
        for c in cells:
            cid = int(c['id'])
            b = np.array(by_cell.get(cid, []), float)
            b = b[np.isfinite(b)]
            row = {'fov': int(fov), 'cell': cid,
                   'celltype': str(c.get('celltype') or ''),
                   'modality': modality, 'hybe': hybe, 'channel': int(channel),
                   'n_spots': int(len(b)),
                   'brightness_median': float(np.median(b)) if len(b) else np.nan,
                   'brightness_total': float(b.sum()) if len(b) else 0.0}
            if mask_intensity:
                med, frame = _mask_median(cell_objs.get(cid), mip, hybe,
                                          modality, resolver)
                row['mask_median'] = med
                row['mask_frame'] = frame
            rows.append(row)
    return pd.DataFrame(rows), {'homeless': homeless}


def _mask_median(cell, mip, hybe, modality, resolver):
    """Median MIP intensity over the cell mask, with frame provenance."""
    if cell is None or mip is None:
        return np.nan, 'missing'
    ys = xs = None
    frame = 'native'
    if resolver is not None:
        try:
            # transform(src, dst) ALREADY maps src-frame points into
            # dst's frame (frames.py contract) -- the mask lives in the
            # cell's reference frame, so H applies DIRECTLY. The first
            # version inverted it, moving the mask by minus the
            # alignment offset while labeling the result exact; caught
            # by adversarial review with a stub resolver before any
            # caller shipped.
            H, _dz, _missing = resolver.transform(
                (cell.reference_hybe, cell.reference_modality),
                (hybe, modality), cell=cell)
            ay, ax = cell.area
            pts = np.stack([ay, ax, np.ones(len(ay))])
            moved = H @ pts
            ys, xs = moved[0], moved[1]
        except Exception:
            ys = xs = None
    if ys is None:
        try:
            ys, xs = cell.get_area_in_readout(hybe, modality)
        except Exception:
            ys, xs = cell.area          # residual matrices: honest fallback
            frame = 'reference'
    ys = np.clip(np.round(ys).astype(int), 0, mip.shape[0] - 1)
    xs = np.clip(np.round(xs).astype(int), 0, mip.shape[1] - 1)
    if len(ys) == 0:
        return np.nan, frame
    return float(np.median(mip[ys, xs])), frame


def normalize(table, metric, mode, ref_source=None):
    """Add a normalized column to an expression table.

    metric: 'n_spots' | 'brightness_median' | 'brightness_total' |
    'mask_median'. mode:
      'by_source'      value / the SAME cell's value of `ref_source`
                       (modality, hybe, channel) -- expression relative
                       to a reference gene/round;
      'by_total_count' value / the cell's total n_spots across every
                       source in the table -- a per-cell detection-load
                       normalization.
    Returns a COPY with '<metric>_norm'; division by zero or a missing
    reference yields NaN, never a fabricated value.
    """
    t = table.copy()
    if mode == 'by_source':
        if ref_source is None:
            raise ValueError("mode='by_source' needs ref_source=(modality, hybe, channel)")
        m, h, ch = ref_source
        ref = t[(t['modality'] == m) & (t['hybe'] == h) & (t['channel'] == int(ch))]
        ref_by_cell = ref.set_index(['fov', 'cell'])[metric]
        idx = pd.MultiIndex.from_frame(t[['fov', 'cell']])
        denom = ref_by_cell.reindex(idx).to_numpy(dtype=float)
    elif mode == 'by_total_count':
        totals = t.groupby(['fov', 'cell'])['n_spots'].transform('sum').to_numpy(dtype=float)
        denom = totals
    else:
        raise ValueError(f'unknown mode {mode!r}')
    with np.errstate(all='ignore'):
        vals = t[metric].to_numpy(dtype=float) / denom
    vals[~np.isfinite(vals)] = np.nan
    t[f'{metric}_norm'] = vals
    return t
