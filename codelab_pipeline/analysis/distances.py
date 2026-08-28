"""
Generalized pairwise distances between any two (modality, hybe, channel)
spot sets -- per cell, in micrometres, celltype-decomposable.

Distances use adj coordinates, which the store keeps in ONE shared frame
across hybes and modalities -- no matrix work here, by construction.
Center-to-center only: ASpot.size is a dead field, so the CellClassifier
edge-to-edge variant has nothing real to subtract.
"""
import numpy as np
import pandas as pd


def _spots_of(pop, source):
    m, h, ch = source
    t = pop.spots
    if t is None or len(t) == 0:
        raise ValueError('population carries no spot table; build with '
                         'spot_sources=[...] first')
    rows = t[(t['modality'] == m) & (t['hybe'] == h) & (t['channel'] == int(ch))]
    if len(rows) == 0:
        raise ValueError(f'no spots for source {source!r} in the population '
                         f'(available: {sorted(set(map(tuple, t[["modality", "hybe", "channel"]].itertuples(index=False))))[:8]} ...)')
    return rows


def pair_distances(pop, source_a, source_b):
    """Tidy per-pair rows: fov, cell, celltype, d_um.

    Every cross-set pair WITHIN a cell (homeless spots excluded --
    distance across cells is not a cellular quantity). When the two
    sources are identical, self-pairs and double counting are excluded
    (i < j).
    """
    a = _spots_of(pop, source_a)
    b = _spots_of(pop, source_b)
    same = tuple(source_a) == tuple(source_b)
    out = []
    bg = {k: v for k, v in b.groupby(['fov', 'cell'])}
    for key, ga in a.groupby(['fov', 'cell']):
        fov, cell = key
        if cell < 0:
            continue
        gb = bg.get(key)
        if gb is None:
            continue
        pa = ga[['y_um', 'x_um', 'z_um']].to_numpy(float)
        pb = gb[['y_um', 'x_um', 'z_um']].to_numpy(float)
        d = np.sqrt(((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1))
        if same:
            iu = np.triu_indices(len(pa), k=1)
            vals = d[iu]
        else:
            vals = d.ravel()
        ct = str(ga['celltype'].iloc[0])
        for v in vals:
            out.append({'fov': int(fov), 'cell': int(cell),
                        'celltype': ct, 'd_um': float(v)})
    return pd.DataFrame(out, columns=['fov', 'cell', 'celltype', 'd_um'])


def pair_distance_per_cell(pop, source_a, source_b, collapse='median'):
    """Series indexed (fov, cell): the collapsed per-cell distance.

    MEDIAN by default and deliberately: min rides the zero-bounded noise
    floor (the SG scripts' documented rule). 'min'/'mean' available for
    callers who state their reasons.
    """
    pairs = pair_distances(pop, source_a, source_b)
    if len(pairs) == 0:
        return pd.Series(dtype=float)
    agg = {'median': 'median', 'mean': 'mean', 'min': 'min'}[collapse]
    return pairs.groupby(['fov', 'cell'])['d_um'].agg(agg)


def distance_histogram(pop, source_a, source_b, mask=None, bins=100,
                       range_um=None, per_celltype=False):
    """Histogram(s) of per-pair distances over gated cells.

    Gating is by CELL mask (Population.cells order); the flags axis:
    per_celltype=True returns {celltype: (counts, edges)} with the ''
    bucket labeled 'Unassigned' -- shown grey, never dropped.
    """
    pairs = pair_distances(pop, source_a, source_b)
    if mask is not None:
        keep = pop.cells.loc[np.asarray(mask, bool), ['fov', 'cell']]
        keys = set(map(tuple, keep.itertuples(index=False)))
        pairs = pairs[[(f, c) in keys for f, c in
                       zip(pairs['fov'], pairs['cell'])]]
    def hist(vals):
        return np.histogram(vals, bins=bins, range=range_um)
    if not per_celltype:
        return hist(pairs['d_um'].to_numpy())
    out = {}
    for ct, g in pairs.groupby('celltype'):
        out[ct if ct else 'Unassigned'] = hist(g['d_um'].to_numpy())
    return out
