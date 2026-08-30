# Gates: conditions that narrow cells

The gate system in `codelab_pipeline/analysis/gate.py` answers one question: *which cells (or alleles) does this analysis run on?* A gate is a `Condition` — an OR of AND-clauses of `Predicate` objects — that evaluates to a boolean mask over `Population.cells`. Gates **narrow** the cell set; they never fan out into subgroups (that is the job of flags in the figure layer). Everything here is headless: no Qt, no app imports, just numpy/pandas over the tidy tables a `Population` carries.

Three design rules run through the whole file:

- **Every predicate carries its own source scope.** An `ExpressionRange` names its `(modality, hybe, channel)` triple itself; two gates in one condition can look at two different channels without coupling.
- **Evaluation is vectorized** over `Population.cells` — masks are numpy boolean arrays in exactly the row order of `pop.cells`, never a per-cell Python loop.
- **A gate is data.** `to_dict()`/`from_dict()` round-trip losslessly, so a saved figure can record the exact gate that produced it, the app config can store your gate list, and a saved gate re-runs headless in a script.

Missingness stays honest throughout: a cell with no value for a gated quantity **fails** a range predicate (it is not known to be in range), but it is visible in `report()` counts rather than silently vanishing.

## The Predicate base class

Every predicate implements `mask(pop) -> np.ndarray[bool]` of length `len(pop.cells)`, aligned to that table's row order. Predicates serialize through a registry keyed by a `kind` string:

```python
d = pred.to_dict()             # {'kind': 'celltype_in', 'names': [...]}
same = Predicate.from_dict(d)  # dispatches on d['kind'], rebuilds the object
```

Two predicates additionally declare `level = 'allele'` and implement `allele_mask(pop)`; everything else defaults to cell level (`getattr(p, 'level', 'cell')`).

## Cell-level predicates

### `CelltypeIn(names)` — kind `'celltype_in'`

Passes cells whose `celltype` column is in `names`. Unassigned cells carry the empty string `''`, so `CelltypeIn([''])` is a valid gate for the unassigned bucket.

### `FovIn(fovs)` — kind `'fov_in'`

Passes cells whose `fov` is in the given list. FOV ids are coerced to `int` at construction.

### `ExpressionRange(source, metric, lo=None, hi=None, normalize=None)` — kind `'expression_range'`

Gates a per-cell expression metric of one **source** — the triple `(modality, hybe, channel)`, e.g. `('RNA', 'Hyb_103', 635)` — within `[lo, hi]`. `metric` is one of `'n_spots'`, `'brightness_median'`, `'brightness_total'`, or `'mask_median'`. Either bound may be `None` for an open interval.

`values(pop)` returns the `(n_cells,)` float vector the gate actually inspects, `NaN` where a cell has no value. It is public on purpose: the GUI's histogram picker and the gate share this one definition, so the threshold you pick on the histogram is the threshold the gate applies. It raises `ValueError` if the population carries no expression table (build with `sources=[...]` first).

**Normalization.** `normalize` is `None`, `('by_modality',)`, or `('by_source', ref_source)`; the gate calls `expression.normalize(...)` and then reads the `'<metric>_norm'` column instead of the raw one:

- `('by_modality',)` divides by the *same kind* of quantity aggregated over the **whole modality within the cell** — `n_spots` over all spots of that modality the cell holds, `brightness_total` over the modality-wide total, and so on. The denominators are the `mod_*` columns computed at `Population.build` from **all** the modality's spots in the store, not just the sources you requested. `'by_total_count'` is accepted as the legacy alias of `'by_modality'`.
- `('by_source', (modality, hybe, channel))` divides by the *same metric* of the reference source in the same cell — e.g. spot count of your gene relative to spot count of a housekeeping hybe.

Division by zero or a missing denominator yields `NaN`, never a fabricated value — and `NaN` fails the range (below).

The mask is `np.isfinite(v)`, AND `v >= lo` if `lo` is given, AND `v <= hi` if `hi` is given. **NaN always fails**, even a fully open range with `lo=None, hi=None` — which makes bound-free `ExpressionRange` a usable "has a measured value here" gate.

### `PairDistanceRange(source_a, source_b, lo=None, hi=None, collapse='median')` — kind `'pair_distance_range'`

Per cell, every cross-set spot pair between `source_a` and `source_b` yields a 3D distance in micrometres; those are collapsed to one number per cell and the collapsed value is gated in um. The collapse is `'median'` by default and deliberately — `'min'` rides the zero-bounded noise floor — with `'mean'` and `'min'` available to callers who state their reasons. `values(pop)` again returns the aligned `(n_cells,)` float vector; cells missing either spot set are `NaN` and therefore fail.

### `AlleleCount(lo=1, hi=None, min_bins=2)` — kind `'allele_count'`

Passes cells holding between `lo` and `hi` **traced** alleles, where "traced" means `n_traced >= min_bins`. This is a cell-level predicate — it counts alleles but gates cells. Homeless alleles (`cell == -1`) are excluded from the count. With no alleles in the population, every cell counts zero (so the default `lo=1` fails everything rather than erroring).

## Allele-level predicates

Two predicates carry `level = 'allele'`, because the quantity they gate is genuinely a property of one homolog, not of the cell. The genetic modification can be **heterogeneous**: within a single cell, one homolog may carry the barcode and the other not. A cell-level verdict would erase exactly the distinction the experiment is designed to see, so these predicates' primary product is `allele_mask(pop)` — a boolean over allele rows. Their `mask(pop)` still exists and means "the cell holds **at least one** qualifying allele", so they compose with cell-level predicates in an ordinary cell gate.

### `BarcodePresence(hybes, absent=False)` — kind `'barcode_presence'`

`hybes` lists genomic bins that must **all** be present in the allele — present meaning a finite traced position in `alleles['pos_um']` at that bin. With `absent=True` the test inverts: all listed bins must be **missing**. A hybe name that is not a genomic bin raises `ValueError` immediately (with the first few valid bin names in the message) rather than silently matching nothing. Both `allele_mask` and `mask` raise `ValueError` if the population carries no alleles.

### `CompletenessRange(lo=None, hi=None)` — kind `'completeness_range'`

The polymer-QC completeness as a gate: the allele's traced-bin count `n_traced` within `[lo, hi]`. Bounds are integers (coerced at construction); either may be `None`.

## Condition: OR of AND-clauses

```python
Condition([p1, p2])                    # one AND clause -- the common case
Condition(clauses=[[p1, p2], [p3]])    # (p1 AND p2) OR (p3)
```

- **`mask(pop)`** gates cells: within each clause the predicate masks are ANDed, across clauses ORed. An empty condition (no clauses) is all-`True` — no gate means every cell passes.
- **`allele_mask(pop)`** gates alleles. Per clause, cell-level predicates are ANDed into a cell mask and projected onto allele rows via `pop.allele_mask_from_cells` (an allele survives iff its `(fov, cell)` survives; homeless alleles never survive a cell gate), allele-level predicates are ANDed directly on allele rows, and the two are combined with AND; clauses are then ORed. This is what lets a heterozygous cell pass the gate while only its modified allele feeds a distance map.
- **`values(...)`** lives on the range predicates themselves (`ExpressionRange`, `PairDistanceRange`), not on `Condition` — use it to look at the distribution before choosing `lo`/`hi`.
- **`report(pop)`** returns `[(repr(predicate), n_cells_after)]` applied *sequentially within each clause*, with clause boundaries marked by a `('-- OR --', running_total)` entry. The intermediate counts therefore depend on predicate order; the final mask does not.
- **`predicates`** is a flat read-only view over all clauses, for listing.

## Serialization

`Condition.to_dict()` returns `{'clauses': [[predicate_dict, ...], ...]}` of plain JSON-safe values. `Condition.from_dict` accepts that form, plus the legacy single-clause form `{'predicates': [...]}`. The round-trip is stable — the app's analysis panel stores each predicate's *normalized* `to_dict()` in the condition list and snapshots it into the saved config, and re-running `from_dict` on that config reproduces the gate exactly:

```python
import json
d = cond.to_dict()
cond2 = Condition.from_dict(json.loads(json.dumps(d)))   # identical behavior
```

## Worked example

Select cells that are either (classified `'Positive'` **and** in a normalized expression window) **or** (hold a well-traced allele that **lacks** the barcode bin), then print the funnel. `jobs != 1` forks worker processes, so on Windows the script body must sit under a `__main__` guard.

```python
from codelab_pipeline.analysis.population import Population
from codelab_pipeline.analysis.gate import (
    Condition, CelltypeIn, ExpressionRange, CompletenessRange, BarcodePresence)

STORE = 'E:/Students/2026-08-07-SG-test/DNA'
BIN_HYBES = [...]                      # ordered genomic-bin hybe list
SRC = ('RNA', 'Hyb_103', 635)          # a source is (modality, hybe, channel)

def main():
    pop = Population.build(STORE, fovs=range(34),
                           hybes=BIN_HYBES,     # -> pop.alleles
                           sources=[SRC],       # -> pop.expression
                           jobs=8)
    print(pop.summary())

    cond = Condition(clauses=[
        [CelltypeIn(['Positive']),
         ExpressionRange(SRC, 'n_spots', lo=0.05, hi=0.60,
                         normalize=('by_modality',))],
        [CompletenessRange(lo=80),
         BarcodePresence(['Hyb_017'], absent=True)],
    ])

    cell_mask = cond.mask(pop)          # bool over pop.cells rows
    allele_mask = cond.allele_mask(pop) # bool over allele rows
    print(f'{int(cell_mask.sum())} / {len(pop.cells)} cells pass; '
          f'{int(allele_mask.sum())} alleles feed the maps')

    for label, n in cond.report(pop):
        print(f'{n:>7}  {label}')

if __name__ == '__main__':             # REQUIRED on Windows with jobs != 1
    main()
```

The report reads as a funnel, one line per predicate with the cell count surviving so far, `-- OR --` lines carrying the running union:

```
   4210  CelltypeIn(names=['Positive'])
   1873  ExpressionRange(source=('RNA', 'Hyb_103', 635), metric='n_spots', ...)
-- OR --  1873
   2954  CompletenessRange(lo=80, hi=None)
    311  BarcodePresence(hybes=['Hyb_017'], absent=True)
```

## Pitfalls

- **NaN fails ranges — always.** A cell absent from the expression table, a zero/missing normalization denominator, a cell missing a spot set for `PairDistanceRange`: all evaluate to `NaN` and fail. This is the honest reading ("not known to be in range"), but it means a gate can shrink your population for reasons of missingness, not biology. `report()` and the predicate's `values()` are how you tell the two apart.
- **Build what you gate.** `ExpressionRange` raises `ValueError` unless the population was built with `sources=[...]` that include your source; the allele predicates raise unless it was built with `hybes=`/`records=`. The errors are actionable, not `KeyError`s — but they arrive at evaluation time, not construction time.
- **`('by_modality',)` needs build-time denominators.** The `mod_*` columns are computed during `Population.build` from *all* of the modality's spots in the store. `normalize` on a hand-assembled table without them refuses with an explicit message; the fix is to rebuild the population, not to fake a column.
- **Sources match exactly.** `modality` and `hybe` are compared as strings, `channel` as `int`. `('RNA', 'Hyb_103', 635)` and `('RNA', 'hyb_103', 635)` are different sources; the second silently matches zero rows and every cell goes `NaN`.
- **`BarcodePresence` validates its bins.** A hybe that is not a genomic bin raises `ValueError` up front — a readout or bridge hybe name here is a bug in the gate, not an empty result.
- **Cells are `(fov, cell)` pairs, alleles can be homeless.** All joins key on the pair — bare cell ids do not exist in this package. Alleles with `cell == -1` never pass any cell projection, and both allele-level predicates' `mask()` drop them when lifting allele verdicts to cells.
- **An empty `Condition` passes everything.** `Condition().mask(pop)` is all-`True`. Convenient as a default; surprising if you expected "no gate configured" to mean "nothing selected".
- **Windows + `jobs != 1` needs the `__main__` guard.** `Population.build` uses a multiprocessing pool with spawn semantics; an unguarded script re-imports itself in every worker.
- **`'by_total_count'` is legacy.** It behaves exactly like `'by_modality'` and exists so old saved gates keep loading; write `('by_modality',)` in new code.
