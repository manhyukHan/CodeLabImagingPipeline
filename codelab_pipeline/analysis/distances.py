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

    VECTORIZED as one within-cell cross join (merge on (fov, cell)) plus
    column arithmetic. The per-cell Python loop this replaces measured
    8.7 s at the design point (12,000 cells, ~772k pairs); cells can be
    tens of thousands, so per-cell Python is banned from every gate
    path.
    """
    cols = ['fov', 'cell', 'celltype', 'y_um', 'x_um', 'z_um']
    a = _spots_of(pop, source_a)[cols]
    a = a[a['cell'] >= 0].reset_index(drop=True)
    same = tuple(source_a) == tuple(source_b)
    b = a if same else _spots_of(pop, source_b)[cols]
    if not same:
        b = b[b['cell'] >= 0].reset_index(drop=True)
    a = a.assign(_ia=np.arange(len(a)))
    b = b.assign(_ib=np.arange(len(b)))
    m = a.merge(b.drop(columns=['celltype']), on=['fov', 'cell'],
                suffixes=('', '_b'))
    if same:
        m = m[m['_ia'] < m['_ib']]
    d = np.sqrt((m['y_um'].to_numpy() - m['y_um_b'].to_numpy()) ** 2
                + (m['x_um'].to_numpy() - m['x_um_b'].to_numpy()) ** 2
                + (m['z_um'].to_numpy() - m['z_um_b'].to_numpy()) ** 2)
    return pd.DataFrame({'fov': m['fov'].to_numpy(),
                         'cell': m['cell'].to_numpy(),
                         'celltype': m['celltype'].to_numpy(),
                         'd_um': d})


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
        # an inner merge, not a Python membership loop: it is a real
        # semi-join at any scale, and on ZERO pairs it stays a 0-row
        # frame WITH its columns -- the list-comprehension form fed
        # pandas an empty list, which selects COLUMNS, and the next
        # access raised KeyError instead of returning an empty histogram.
        keep = pop.cells.loc[np.asarray(mask, bool), ['fov', 'cell']]
        pairs = pairs.merge(keep.drop_duplicates(), on=['fov', 'cell'],
                            how='inner')
    def hist(vals):
        return np.histogram(vals, bins=bins, range=range_um)
    if not per_celltype:
        return hist(pairs['d_um'].to_numpy())
    out = {}
    for ct, g in pairs.groupby('celltype'):
        out[ct if ct else 'Unassigned'] = hist(g['d_um'].to_numpy())
    return out
