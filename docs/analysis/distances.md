# Pair distances

`codelab_pipeline/analysis/distances.py` computes pairwise 3D distances between any two
spot sets — per cell, in micrometres, ready for celltype decomposition. It is part of the
headless analysis toolbox: importing it loads no Qt and no app module, so everything on
this page runs in a plain script or notebook against a store path.

Two sources go in, a tidy table of distances comes out. A **source** is the triple
`(modality, hybe, channel)` — for example `('RNA', 'Hyb_103', 635)` — and the spots for
each source must already be in the population: build it with `spot_sources=[...]` so
`Population.spots` carries the tables these functions read. Positions in that table
(`y_um`, `x_um`, `z_um`) were scaled from `(y, x, z)` pixel/plane coordinates by
`voxel_um=(0.208, 0.208, 0.2)` exactly once, at extraction, so all distances here are in
micrometres with no further conversion.

Two design decisions from the module itself are worth restating:

- Distances use **adj coordinates**, which the store keeps in one shared reference frame
  across hybes and modalities. No alignment matrices are applied here — cross-modal
  distances are correct by construction of the frame, not by per-call transforms.
- Distances are **center-to-center only**. `ASpot.size` is a dead field, so there is no
  real radius to subtract for an edge-to-edge variant.

## pair_distances: every cross-set pair within a cell

```python
def pair_distances(pop, source_a, source_b):
```

Returns a tidy `pandas.DataFrame` with one row per spot pair and exactly four columns:

| column | meaning |
|---|---|
| `fov` | field of view of the cell |
| `cell` | cell id within that FOV — always read together with `fov` |
| `celltype` | the cell's type string (taken from the `source_a` spot rows) |
| `d_um` | 3D Euclidean distance between the two spots, micrometres |

Pairs are formed **within cells only**: every spot of `source_a` is paired with every
spot of `source_b` that lives in the same `(fov, cell)`. Homeless spots (`cell < 0`) are
excluded before pairing — distance across cells is not a cellular quantity. When
`source_a` and `source_b` are the same triple, self-pairs and double counting are
excluded by an `i < j` filter, so a cell with *n* spots contributes *n(n−1)/2* rows, not
*n²*.

The implementation is a single vectorized within-cell cross join — a `merge` of the two
spot tables on `['fov', 'cell']` followed by column arithmetic — rather than a per-cell
Python loop. This matters at real population sizes: the per-cell loop it replaced
measured 8.7 s at the design point of 12,000 cells (~772k pairs); the vectorized join
does the same work in roughly 0.2 s. Because cell counts here run to tens of thousands,
per-cell Python is banned from every gate path, and this function is what the
`PairDistanceRange` gate predicate calls under the hood.

If the population carries no spot table at all, or carries no rows for a requested
source, you get a `ValueError` with an actionable message — the second case lists the
first few `(modality, hybe, channel)` triples that *are* available, which is the fastest
way to catch a typo in a hybe name or a channel passed with the wrong value.

## pair_distance_per_cell: one number per cell

```python
def pair_distance_per_cell(pop, source_a, source_b, collapse='median'):
```

Runs `pair_distances` and collapses the per-pair rows to a `pandas.Series` of `d_um`
values indexed by the `(fov, cell)` MultiIndex — one value per cell that has at least one
pair.

`collapse` accepts `'median'`, `'mean'`, or `'min'`. The default is **median, and
deliberately so**: `min` rides the zero-bounded noise floor (two dense spot sets almost
always have *some* close pair, so per-cell minima cluster near zero regardless of
biology). This is the documented rule from the SG scripts; use `'min'` or `'mean'` only
when you can state why.

If there are no pairs at all, the function returns an empty `pd.Series(dtype=float)`.

A cell that is missing spots from either source simply does not appear in the Series.
To line the result up with the population's cell axis — for gating, or for joining onto
other per-cell quantities — reindex it against `Population.cells`, which turns absent
cells into honest `NaN` rather than fabricated zeros:

```python
import pandas as pd

per_cell = distances.pair_distance_per_cell(pop, src_rna, src_dna)
idx = pd.MultiIndex.from_frame(pop.cells[['fov', 'cell']])
values = per_cell.reindex(idx).to_numpy(dtype=float)   # (n_cells,), NaN = no value
```

This is exactly what `gate.PairDistanceRange` does internally, and its range test uses
`np.isfinite`, so a cell missing either spot set *fails* the gate — it is not known to be
in range.

## distance_histogram: gated distributions

```python
def distance_histogram(pop, source_a, source_b, mask=None, bins=100,
                       range_um=None, per_celltype=False):
```

Histograms the per-pair distances over gated cells.

- `mask` is a boolean **cell** mask in `Population.cells` row order — typically the
  output of `gate.Condition(...).mask(pop)`. Gating is implemented as a real semi-join:
  the surviving `(fov, cell)` pairs are inner-merged onto the pair table. This is not a
  style preference — an earlier list-comprehension form fed pandas an empty list when
  zero pairs survived, which pandas interpreted as selecting *columns*, and the next
  column access raised `KeyError`. The merge form stays a 0-row frame **with** its
  columns, so an empty gate yields an empty histogram instead of a crash.
- `bins` and `range_um` pass through to `np.histogram` (`range_um` is its `range`
  argument, in micrometres).
- With `per_celltype=False` (default) the return value is the `np.histogram` tuple
  `(counts, edges)`.
- With `per_celltype=True` the return value is a dict `{celltype: (counts, edges)}`.
  The empty-string celltype bucket is relabeled `'Unassigned'` — by convention it is
  shown grey and never dropped. Note that with zero surviving pairs this dict is empty.

## Worked example: RNA-to-DNA distance in gated cells

The script below builds a population carrying two spot sources, gates to one celltype,
and reports the per-cell median RNA–DNA distance plus a gated histogram. Because
`Population.build` with `jobs != 1` uses multiprocessing, and Windows starts child
processes by *spawn* (which re-imports your script), the whole thing **must** sit under
an `if __name__ == '__main__':` guard.

```python
import pandas as pd
from codelab_pipeline.analysis import population, gate, distances

STORE = r'E:/Students/2026-08-07-SG-test/DNA'
src_rna = ('RNA', 'Hyb_103', 635)
src_dna = ('DNA', 'Hyb_012', 750)

if __name__ == '__main__':
    pop = population.Population.build(
        STORE, fovs=range(34),
        spot_sources=[src_rna, src_dna],
        jobs=12)
    print(pop.summary())

    # gate: one celltype, and cells actually holding both spot sets
    cond = gate.Condition([
        gate.CelltypeIn(['WT']),
        gate.PairDistanceRange(src_rna, src_dna, lo=0.0),  # finite => both sets present
    ])
    mask = cond.mask(pop)
    print(f'{mask.sum()} of {len(pop.cells)} cells pass the gate')

    # one median distance per gated cell, in micrometres
    per_cell = distances.pair_distance_per_cell(pop, src_rna, src_dna,
                                                collapse='median')
    idx = pd.MultiIndex.from_frame(pop.cells.loc[mask, ['fov', 'cell']])
    gated = per_cell.reindex(idx).dropna()
    print(gated.describe())

    # distribution of ALL pair distances inside the gated cells
    counts, edges = distances.distance_histogram(
        pop, src_rna, src_dna, mask=mask, bins=80, range_um=(0.0, 8.0))
```

`gated` is a float Series indexed by `(fov, cell)`; `counts` and `edges` are the usual
`np.histogram` arrays. Swap `mask=mask` for `per_celltype=True` (with or without the
mask) to get one `(counts, edges)` per celltype instead.

## Pitfalls

- **Build with `spot_sources`, or nothing works.** All three functions read
  `Population.spots`; a population built without `spot_sources=[...]` raises
  `ValueError('population carries no spot table; ...')` rather than a bare `KeyError`.
  A source with zero matching rows also raises, listing what is available.
- **Cells are `(fov, cell)` pairs, always.** The per-cell Series is indexed by the pair,
  the histogram gate joins on the pair, and bare cell ids do not exist in this package —
  cell ids repeat across FOVs, and keying on them alone silently merges unrelated cells.
- **Homeless spots never contribute.** Rows with `cell < 0` are dropped before pairing;
  if a detection run left many spots unassigned, your pair counts shrink accordingly and
  no warning is printed.
- **Channel matching casts to `int`.** The source's channel is compared as
  `int(channel)`, so `635` and `'635'` both match — but a wavelength that does not
  exist in the spot table fails with the available-sources `ValueError`.
- **`celltype` in the pair table comes from `source_a`'s rows.** The merge drops
  `source_b`'s celltype column. Within one cell they are the same string, so this is
  only visible if you expected a second column.
- **Median is the default collapse for a reason.** `'min'` is available but sits on the
  zero-bounded noise floor; prefer `'median'` unless you have an explicit argument.
  An unknown `collapse` string raises `KeyError`.
- **Empty inputs are honest, and differently shaped.** No pairs at all:
  `pair_distance_per_cell` returns an empty float Series with a plain (not MultiIndex)
  index — reindexing against `pop.cells` still works and yields all-NaN.
  `distance_histogram` returns a zero-count histogram, except with
  `per_celltype=True`, where it returns `{}`.
- **NaN means "no value", never zero.** After reindexing per-cell distances onto
  `Population.cells`, treat NaN as missing; gates do (`np.isfinite` first), and so
  should any statistic you compute.
- **The cross join is fast but its output is quadratic per cell.** Memory and row count
  scale with the summed product of per-cell spot counts. Two sources with dozens of
  spots per cell across tens of thousands of cells can produce many millions of rows;
  gate first (via `mask` in `distance_histogram`, or by restricting `fovs` at build
  time) if that becomes a concern.
- **Windows + `jobs != 1` needs the main guard.** `Population.build` fans out over a
  spawn-based process pool; without `if __name__ == '__main__':` your script re-executes
  itself in every child.
