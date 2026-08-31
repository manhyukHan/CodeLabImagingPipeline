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


TRACED = 'traced'


def _traced_of(pop, source, alleles=None):
    """Per-allele TRACED positions for one genomic bin, in the same tidy
    shape as spots: fov, cell, celltype, y_um, x_um, z_um.

    A source (modality, hybe, TRACED) means "the polymer position this
    allele's trace assigns to `hybe`" -- one point per ALLELE, not per
    spot. That is the quantity chromatin tracing produces, and without
    it a distance could only ever relate raw detections, never a locus
    to an RNA spot (report). Untraced bins are NaN and simply absent
    here; homeless alleles (cell -1) are excluded, as for spots.

    alleles: a QC-filtered allele dict when the caller has one
    (pop.alleles otherwise) -- so a gate and a view can agree on which
    alleles count.
    """
    m, h, _ch = source
    al = alleles if alleles is not None else pop.alleles
    if al is None:
        raise ValueError('population carries no alleles; build with '
                         'records=/hybes= first')
    bins = list(al.get('bin_hybes') or [])
    if h not in bins:
        raise ValueError(f'{h} is not a traced genomic bin '
                         f'(bins: {bins[:6]}{" ..." if len(bins) > 6 else ""})')
    j = bins.index(h)
    pos = np.asarray(al['pos_um'])[:, j, :]
    cell = np.asarray(al['cell'])
    keep = np.isfinite(pos[:, 0]) & (cell >= 0)
    if not keep.any():
        raise ValueError(f'no traced positions for {m}/{h} in this '
                         f'population')
    ct = np.asarray(al.get('celltype') if al.get('celltype') is not None
                    else [''] * len(cell), dtype=object)
    return pd.DataFrame({'fov': np.asarray(al['fov'])[keep],
                         'cell': cell[keep],
                         'celltype': ct[keep],
                         'y_um': pos[keep, 0],
                         'x_um': pos[keep, 1],
                         'z_um': pos[keep, 2]})


def is_traced_source(source):
    """(modality, hybe, 'traced') -- a polymer bin, not a spot slice."""
    return len(source) > 2 and str(source[2]) == TRACED


def points_of(pop, source, alleles=None):
    """The tidy points for ANY source: a spot slice, or a traced bin."""
    if is_traced_source(source):
        return _traced_of(pop, source, alleles)
    return _spots_of(pop, source)


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


def pair_distances(pop, source_a, source_b, alleles=None):
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
    a = points_of(pop, source_a, alleles)[cols]
    a = a[a['cell'] >= 0].reset_index(drop=True)
    same = tuple(source_a) == tuple(source_b)
    b = a if same else points_of(pop, source_b, alleles)[cols]
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


def pair_distance_per_cell(pop, source_a, source_b, collapse='median',
                           alleles=None):
    """Series indexed (fov, cell): the collapsed per-cell distance.

    MEDIAN by default and deliberately: min rides the zero-bounded noise
    floor (the SG scripts' documented rule). 'min'/'mean' available for
    callers who state their reasons.
    """
    pairs = pair_distances(pop, source_a, source_b, alleles)
    if len(pairs) == 0:
        return pd.Series(dtype=float)
    agg = {'median': 'median', 'mean': 'mean', 'min': 'min'}[collapse]
    return pairs.groupby(['fov', 'cell'])['d_um'].agg(agg)


def distance_histogram(pop, source_a, source_b, mask=None, bins=100,
                       range_um=None, per_celltype=False, alleles=None):
    """Histogram(s) of per-pair distances over gated cells.

    Gating is by CELL mask (Population.cells order); the flags axis:
    per_celltype=True returns {celltype: (counts, edges)} with the ''
    bucket labeled 'Unassigned' -- shown grey, never dropped.
    """
    pairs = pair_distances(pop, source_a, source_b, alleles)
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
