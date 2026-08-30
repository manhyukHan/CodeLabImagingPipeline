# Expression: spot- and mask-based, honestly framed

`codelab_pipeline/analysis/expression.py` estimates per-cell expression two ways — from stored spot detections and from mask intensity — and provides the normalizations the condition system gates on. Like everything in the analysis toolbox it is headless: it imports numpy, pandas, and the store layer, never Qt or any app module, so you can run it from a script or notebook pointed at a store.

Two conventions to keep in mind throughout:

- A **source** is always the triple `(modality, hybe, channel)`, e.g. `('RNA', 'Hyb_103', 635)`. A bare hybe name is not an identity — the cross-modal bridge hybe exists as a real, distinct file in *both* modalities.
- Cells are keyed by the **pair** `(fov, cell)`. Bare cell ids do not exist in this package.

Spot-based metrics come from the stored detections: the count, median, and total of `ASpot.brightness` (raw MIP counts at detection time; `ASpot.size` is a dead field and is deliberately not offered). Mask-based intensity is the median MIP value over the cell's own mask — defined for every cell, including cells with zero detected spots, which is exactly the case spot-based metrics cannot see.

## `fov_expression_table`

```python
def fov_expression_table(storage_path, fov, sources, mask_intensity=False,
                         resolver=None):
    ...
    # returns (DataFrame, {'homeless': {source: n}})
```

Builds a tidy long table for one FOV: one row per (cell × source). `sources` is a list of `(modality, hybe, channel)` triples. Every cell in the FOV gets a row for every source, even with zero spots.

Columns, always present:

| column | type | meaning |
|---|---|---|
| `fov` | int | the FOV |
| `cell` | int | cell id within the FOV (key with `fov`) |
| `celltype` | str | assigned cell type, `''` when unassigned |
| `modality`, `hybe`, `channel` | str, str, int | the source triple |
| `n_spots` | int | spots of this source assigned to this cell (finite brightness only) |
| `brightness_median` | float | median spot brightness; **NaN** when the cell has no spots |
| `brightness_total` | float | summed spot brightness; **0.0** when the cell has no spots |

Note the deliberate asymmetry at zero spots: a median of nothing is unknowable (NaN), but a total of nothing is genuinely zero.

With `mask_intensity=True` two more columns appear:

| column | type | meaning |
|---|---|---|
| `mask_median` | float | median MIP intensity over the cell mask; NaN when the cell or MIP is missing, or the mask is empty |
| `mask_frame` | str | provenance of the frame the mask was evaluated in (next section) |

The second return value is a companion dict, `{'homeless': {source: n}}`, counting spots with `cell == -1` per source. Homeless spots are excluded from every per-cell row, and this dict exists so they do not disappear silently — if the homeless count is large, your spot-to-cell assignment deserves a look before you trust the table.

The MIP for `mask_median` is read from the **source's own modality tree** (`<store root>/<modality>`), not from whichever `storage_path` you passed — otherwise an RNA source would resolve against the DNA tree whenever the table was built from the DNA storage path, and the bridge hybe would silently collide.

## `mask_frame`: where the mask actually was

Projecting a cell mask exactly into a source hybe's raw frame needs the alignment chain. The function is honest about how much of that chain it could apply, per row:

- **`'native'`** — the mask was projected exactly into the source hybe's own raw frame, either by a complete resolver transform or by the cell's own composable matrices (`matrix_to` via `get_area_in_readout`).
- **`'native-partial:<layers>'`** — a resolver was used, but some alignment layers had not been computed and were defaulted to identity. The suffix names them, comma-joined and sorted, e.g. `native-partial:cross-modal:DNA->RNA` or `native-partial:same-modality:RNA/Hyb_103`. Calling this `'native'` would overclaim exactness, so the row names the gap instead.
- **`'reference'`** — no resolver, and the cell's stored matrices are residual-form (post cell-alignment) so they cannot compose the projection; the reference-frame mask was used unprojected.
- **`'missing'`** — the cell object or the MIP could not be loaded at all; `mask_median` is NaN.

Why the `'reference'` fallback matters: for a **same-modality** source the reference-frame mask is off by a few pixels of inter-round drift — usually tolerable against ~50 px masks. For a **cross-modal** source it is off by the entire cross-modal bridge, measured at **~13 px** on the real store. Against ~50 px masks that is background dilution, not a rounding error: the median samples a substantially wrong region. Either way the approximation is flagged in `mask_frame`, never silent — filter on it before comparing mask medians across modalities.

## The `resolver` argument

`resolver` is a `FrameResolver` (see `codelab_pipeline/alignment/frames.py`): its `transform(src, dst, cell=...)` maps points from `src`'s frame into `dst`'s and reports which layers were defaulted. You do not need the app to get one — `codelab_pipeline/analysis/resolvers.py` assembles it purely from the store:

```python
from codelab_pipeline.analysis import resolvers, expression

r = resolvers.resolver_for(storage_path, fov)   # infers the hub modality
table, extra = expression.fov_expression_table(
    storage_path, fov,
    sources=[('RNA', 'Hyb_103', 635), ('DNA', 'Hyb_130', 748)],
    mask_intensity=True, resolver=r)
```

With a resolver, every row gets exact (or explicitly `native-partial`) projection, cross-modal sources included. Without one, `fov_expression_table` uses only the cell's own stored matrices and falls back to `'reference'` when they cannot compose. For cross-modal sources you should always pass one, for the ~13 px reason above.

`Population.build(..., mask_intensity=True)` builds this resolver for you per FOV when you do not pass `resolvers=`, so the exact projection is the headless default there. Only a direct `fov_expression_table` call leaves it entirely up to you.

## `normalize`

```python
def normalize(table, metric, mode, ref_source=None):
    ...
    # returns a copy of table with a '<metric>_norm' column added
```

`metric` is one of `'n_spots'`, `'brightness_median'`, `'brightness_total'`, `'mask_median'`. The input table is not mutated; a copy with the extra column comes back.

**`mode='by_modality'`** divides each row's metric by the *same kind of quantity computed over the whole modality within that cell*. The denominators are the `mod_*` columns that `Population.build` attaches:

| metric | denominator column | denominator meaning (per cell) |
|---|---|---|
| `n_spots` | `mod_n_spots` | count of **all** the modality's spots in the cell |
| `brightness_median` | `mod_brightness_median` | median brightness over all the modality's spots in the cell |
| `brightness_total` | `mod_brightness_total` | total brightness over all the modality's spots in the cell |
| `mask_median` | `mod_mask_median` | median of `mask_median` over that modality's **built** mask sources in the cell |

The first three are computed at `Population.build` from one whole-FOV spot read per modality — over **all** of the modality's spots in the store, not just the sources you requested. `mod_mask_median`, by contrast, is the median over the mask sources the population actually built (a `groupby(['modality', 'cell'])` over the table's own `mask_median`): add or remove built sources and this denominator changes. `'by_total_count'` is accepted as a legacy alias for `'by_modality'`.

If the `mod_*` column for your metric is absent — you built the table with a bare `fov_expression_table` call, or with an older cache — `normalize` **refuses** with a `ValueError` telling you to rebuild the population, rather than silently normalizing by nothing.

**`mode='by_source'`** divides by the *same metric* of one reference source in the *same cell*. `ref_source=(modality, hybe, channel)` is required; omitting it raises. The reference rows are matched per `(fov, cell)` pair, so a cell with no reference row gets NaN. This mode needs no `mod_*` columns and works on a raw `fov_expression_table` output.

In every mode, division by zero or by a missing denominator yields **NaN, never a fabricated value** — non-finite results are set to NaN wholesale. A cell with zero spots of a modality has no `mod_*` entry at all (NaN denominator), and a `brightness_total` of 0.0 over a `mod_brightness_total` of 0.0 is NaN, not 1.

## A complete headless run

```python
from codelab_pipeline.analysis.population import Population
from codelab_pipeline.analysis import expression

STORE = 'E:/Students/2026-08-07-SG-test/DNA'
SOURCES = [('RNA', 'Hyb_103', 635), ('RNA', 'Hyb_105', 561)]

if __name__ == '__main__':                     # REQUIRED with jobs != 1
    pop = Population.build(STORE, fovs=range(34),
                           hybes=[], sources=SOURCES,
                           mask_intensity=True, jobs=8)
    t = pop.expression                          # has the mod_* columns
    t = expression.normalize(t, 'n_spots', 'by_modality')
    t = expression.normalize(t, 'brightness_total', 'by_source',
                             ref_source=('RNA', 'Hyb_103', 635))
    exact = t[t['mask_frame'] == 'native']
    print(exact.groupby('celltype')['n_spots_norm'].median())
```

## Pitfalls

- **`n_spots` counts finite brightnesses only.** A stored spot whose brightness is NaN is dropped from all three spot metrics of its row. The `mod_n_spots` denominator, computed separately at build, counts every assigned spot of the modality — the two counts can differ when NaN brightnesses exist.
- **Zero-spot asymmetry.** `brightness_median` is NaN with no spots; `brightness_total` is 0.0. Filtering on one and aggregating the other will surprise you.
- **`mask_frame == 'reference'` on a cross-modal source is a ~13 px error.** Pass a resolver (`resolvers.resolver_for`) or use `Population.build`, which does. Never average `'reference'` rows together with `'native'` rows across modalities.
- **`native-partial` is not `native`.** The resolver always returns a usable transform, with uncomputed layers defaulted to identity — the suffix names exactly which layers are provisional. Decide per analysis whether that is acceptable.
- **`normalize(..., 'by_modality')` refuses on tables without `mod_*` columns.** Only `Population.build` attaches them. On a bare `fov_expression_table` output, use `by_source` or rebuild through the population.
- **`mod_mask_median` depends on which sources you built.** It is the median over the population's built mask sources of the modality, not over everything in the store.
- **The build cache cannot see changed inputs.** Expression rows persist per FOV and per source (append mode); after re-detection or re-alignment pass `overwrite_cache=True` to `Population.build`, or you will normalize fresh sources against stale ones.
- **An FOV with no cells yields an empty DataFrame with no columns.** Check `len(table)` before indexing columns; `Population.build` already guards its concatenation against this.
- **Windows multiprocessing.** Any script calling `Population.build` with `jobs != 1` must keep that call under `if __name__ == '__main__':` — Windows spawns fresh interpreters that re-import your script.
- **Homeless spots are excluded, not lost.** They are counted per source in the companion dict; a per-cell sum over the table will not equal the store's total spot count whenever homeless spots exist.
