# Population: one build, plain tables

`codelab_pipeline.analysis.population` assembles everything the gates and analyzers
consume into one object, in one pass over the store. It is headless by construction:
importing it loads no Qt and no app module, so you can build a `Population` from a
script or a notebook pointed at a storage path. The build fans out FOV-major through
the repo's one process pool (`codelab_pipeline.parallel.pmap`), and what comes back
is plain data — pandas DataFrames and stacked numpy arrays, nothing that needs the
app to interpret.

Throughout this package, a cell is keyed by the *pair* `(fov, cell)`; bare cell ids
do not exist here (keying by basename once silently merged 2172 cells into 198
groups). A "source" is the triple `(modality, hybe, channel)`, e.g.
`('RNA', 'Hyb_103', 635)`. Positions in analysis tables are in **micrometres**,
scaled exactly once at extraction from `(y, x, z)` pixel/plane coordinates by
`voxel_um` (default `(0.208, 0.208, 0.2)`). NaN always means "not measured", never
a fabricated value.

## What a Population holds

A built `Population` has these attributes:

- **`cells`** — DataFrame with columns `[fov, cell, celltype]`, one row per stored
  cell. This is the *mask axis*: every boolean cell gate you write is over this
  row order, and always present (possibly empty, but with its columns intact).
  `celltype` is `''` where unassigned.
- **`alleles`** — a dict of arrays stacked across FOVs, or `None` unless the build
  was given `records=` or `hybes=`. Keys (from `polymer.fov_polymer_table`):
  - `pos_um` — `(n_alleles, n_bins, 3)` float, `(y, x, z)` in µm; NaN where a bin
    was untraced. All-NaN alleles are kept — dropping is a QC decision, not an
    extraction side effect.
  - `amp` — `(n_alleles, n_bins)` selected-candidate amplitude (NaN where no candidate).
  - `n_cand` — `(n_alleles, n_bins)` int, candidates per bin (sister chromatids show up here).
  - `allele_id`, `cell`, `fov` — `(n_alleles,)` int; `cell` is `-1` for homeless alleles.
  - `celltype` — `list[str]`, `''` where unknown or homeless.
  - `n_traced` — `(n_alleles,)` int, finite bins per allele.
  - `bin_hybes` — the ordered hybe-folder list that *is* the bin axis.
  - `bin_ids` — the matching `readout_id` list (empty unless `records=` was given).
  - `repeat_pos_um` / `toe_pos_um` — same-shaped position stacks for the R (repeat)
    and T (toe) QC rounds, read in the same allele order; present only when
    `records=` was given, the layout has such rounds, and every FOV produced them.
    `repeat_ids` / `toe_ids` carry their readout ids.
- **`expression`** — long-form DataFrame, or `None` unless `sources=` was given.
  One row per (cell × source): `fov, cell, celltype, modality, hybe, channel,
  n_spots, brightness_median, brightness_total`, plus per-modality denominators
  `mod_n_spots, mod_brightness_median, mod_brightness_total` (aggregated over *all*
  of that modality's spots in the store, not just the requested sources), plus
  `mask_median, mask_frame, mod_mask_median` when `mask_intensity=True`.
- **`spots`** — DataFrame, or `None` unless `spot_sources=` was given. Columns
  `fov, cell, celltype, modality, hybe, channel, y_um, x_um, z_um, brightness`,
  with numeric dtypes enforced. Positions come from each spot's `adj_coordinate`
  (the shared reference frame) scaled to µm. Homeless spots appear with `cell == -1`.
- **`failures`** — list of `(fov, message)` for FOVs whose extraction raised. A
  failing FOV does **not** abort the build; check this list yourself.
- **`cache_stats`** — `{'cached': n, 'computed': n}` source counts summed over
  FOVs, or `None` when no expression sources were requested.

Also stored: `storage_path`, `fovs` (the requested list), and `voxel_um`.

## Population.build

```python
@classmethod
def build(cls, storage_path, fovs, records=None, hybes=None,
          sources=None, spot_sources=None, voxel_um=DEFAULT_VOXEL_UM,
          mask_intensity=False, resolvers=None, jobs=None,
          on_done=None, overwrite_cache=False):
```

- **`storage_path`** — the modality directory of a v2 store, e.g. `<root>/DNA`.
- **`fovs`** — iterable of FOV integers. Duplicates raise `ValueError` immediately:
  a duplicated FOV would duplicate every `(fov, cell)` key and every downstream
  join would fabricate signal.
- **`records`** — parsed layout records (`preprocess.parse_experiment_layout`, or
  headless via `polymer.records_for(storage_path)`, which reads the layout path the
  store's own manifest names). When given, `hybes` defaults to the datatype-`H`
  rounds in `readout_id` order, `bin_ids` is filled, and the repeat/toe QC rounds
  are extracted alongside.
- **`hybes`** — alternatively, pass the ordered genomic-bin hybe list directly.
  With neither `records` nor `hybes`, `alleles` is `None`.
- **`sources`** / **`spot_sources`** — lists of `(modality, hybe, channel)` triples
  for the expression and spots tables respectively. Omit either and the matching
  attribute is `None`.
- **`voxel_um`** — per-axis `(y, x, z)` µm scaling; default `(0.208, 0.208, 0.2)`.
  y/x are pixels, z is plane index — the axes scale independently.
- **`mask_intensity`** — adds `mask_median`/`mask_frame` columns (median MIP
  intensity inside each cell's mask, per source).
- **`resolvers`** — `{fov: FrameResolver}` for *exact* projection of the mask into
  each source hybe's own raw frame (`mask_frame == 'native'`). The resolver is
  plain data and pickles into the child processes. If `mask_intensity=True` and no
  resolver is supplied for a FOV, the worker builds one from the store
  (`analysis.resolvers.resolver_for`) — exact by default, headless. Only a store
  with no alignment yet falls back to the flagged reference-frame mask
  (`mask_frame == 'reference'`), which for a cross-modal source is off by the whole
  cross-modal bridge (~13 px measured).
- **`jobs`** — worker count. `None` uses the measured I/O default (12 workers —
  more was *slower* against the NAS), capped at the number of FOVs. `jobs=1` runs
  inline in this process with no pool at all, so it is breakpoint-debuggable and
  byte-identical in behavior to the parallel path.
- **`on_done`** — progress callback `on_done(n_done, n_total, index, result)`,
  called on the parent as each FOV finishes, in completion order.
- **`overwrite_cache`** — recompute and rewrite every requested cache entry
  (see below).

Each FOV is extracted by a module-level function through one `pmap(kind='io')`
pool: HDF5 files are opened in the *child* processes only, per the measured h5py
process-lock rule, and everything crossing the process boundary is picklable plain
data. Results come back in input order; a FOV that raises comes back as a
`parallel.Failure` and lands in `population.failures` instead of killing the build.

## The append-mode expression cache

Computed cell attributes are the NAS-expensive part of a build (per-source spot
slices, mask medians over MIPs). They persist per FOV in `expression.json` beside
the FOV's other capsules, read and written by
`analysis_store.read_fov_expression(storage_path, fov)` (returns the payload dict
or `None`) and `write_fov_expression(storage_path, fov, payload)` (a full-replace
atomic write; the *merging* lives in the population build, not in the store layer).

The capsule keys each source as `'MOD|HYBE|CH'` and stores, per source: the base
per-cell rows (`cell, celltype, n_spots, brightness_median, brightness_total`, plus
`mask_median`/`mask_frame` when computed with masks), a `mask` flag, and the FOV's
homeless-spot count for that source. Separately, `agg_by_mod` caches the per-cell
whole-modality aggregates (`mod_n_spots`, `mod_brightness_median`,
`mod_brightness_total`) — the denominators that by-modality normalization divides by.

Rebuild semantics:

- A source already in the capsule is **reused, not recomputed** — so adding one
  source to a later build computes only that source (append mode).
- A cached entry written without mask columns cannot serve a
  `mask_intensity=True` build; that source is recomputed.
- **Celltype is never trusted from the cache.** Assignments change after
  re-classification, so every load refreshes `celltype` from the FOV's current
  cells.
- The `mod_*` columns are never baked into the per-source rows; they are joined at
  load time, so a later build with more sources never carries a stale denominator.
- A FOV with no segmented cells caches an *empty* row list — still a valid result
  (its spot slices were read, homeless spots were counted).
- `overwrite_cache=True` recomputes and rewrites everything requested. Use it
  after re-detection or re-alignment: the cache cannot see that its inputs changed.

`population.cache_stats` reports how much of the build the cache served, and
`summary()` includes it.

## dmaps, cell-to-allele masks, summary

**`pop.dmaps()`** returns `(n_alleles, n_bins, n_bins)` pairwise Euclidean
distances in µm (float32, computed once and memoized on the instance). NaN
positions propagate to NaN rows/columns, exactly the ORCA behavior. It raises
`ValueError` if the population carries no alleles.

**`pop.allele_mask_from_cells(cell_mask)`** projects a *cell* gate onto allele
rows: given a boolean array over `pop.cells` rows, it returns a boolean array over
allele rows in which an allele survives iff its `(fov, cell)` pair survives.
Homeless alleles (`cell == -1`) never survive a cell gate — they belong to no
gated cell.

**`pop.summary()`** returns a one-line human-readable string: cell/FOV counts,
allele and traced-allele counts, expression row count with cache stats, spot
count, failed FOVs (names them all, quotes only the first message), and the
celltype tally.

## Worked example

```python
"""Build a population over a store, headless -- no GUI, no config file."""
from codelab_pipeline.analysis import polymer as P
from codelab_pipeline.analysis.population import Population

STORE = 'E:/Students/2026-08-07-SG-test/DNA'   # the modality directory

def main():
    # The store's manifest names its own layout file; records_for parses it.
    records = P.records_for(STORE)

    pop = Population.build(
        STORE,
        fovs=range(34),
        records=records,                          # bins + repeat/toe rounds
        sources=[('RNA', 'Hyb_103', 635),         # expression table
                 ('RNA', 'Hyb_104', 635)],
        spot_sources=[('RNA', 'Hyb_103', 635)],   # per-spot table, um
        on_done=lambda done, total, i, r: print(f'{done}/{total} FOVs'),
    )
    print(pop.summary())
    if pop.failures:
        for fov, msg in pop.failures:
            print('FAILED', fov, msg)

    # Gate cells, project the gate onto alleles, slice the distance maps.
    gate = pop.cells['celltype'].eq('K562').to_numpy()
    dmaps = pop.dmaps()[pop.allele_mask_from_cells(gate)]

    # Expression is a plain tidy DataFrame -- ordinary pandas from here on.
    expr = pop.expression
    frac = expr['brightness_total'] / expr['mod_brightness_total']

if __name__ == '__main__':   # REQUIRED on Windows for the parallel build
    main()
```

## Pitfalls

- **The `__main__` guard is mandatory on Windows** for any script calling
  `Population.build` with `jobs != 1`. Multiprocessing spawn re-imports your
  script in every child; without the guard each child starts its own build.
  `jobs=1` is the escape hatch — fully serial, in-process, breakpointable.
- **A build does not raise on a failed FOV.** Check `pop.failures`; `summary()`
  names failed FOVs but quotes only the first message.
- **`expression` and `spots` are `None`, not empty frames, when not requested.**
  Test for `None` before touching them.
- **NaN conventions differ between median and total.** A cell with zero spots for
  a source has `n_spots == 0`, `brightness_median == NaN`, but
  `brightness_total == 0.0` (an honest sum of nothing). A cell with no spots in a
  whole modality has NaN for all three `mod_*` columns.
- **The cache cannot detect stale inputs.** After re-running spot detection,
  alignment, or mask edits, rebuild with `overwrite_cache=True`; a plain rebuild
  will happily serve the old numbers. Celltype is the one exception — it is
  refreshed from the store on every load.
- **`mask_frame` tells you which mask you got.** `'native'` means the mask was
  projected exactly into the source hybe's own frame; `'reference'` means the
  fallback reference-frame mask, which is ~13 px off for cross-modal sources.
- **Homeless spots never appear in expression rows.** They are counted per source
  (and cached), but excluded from per-cell aggregates; in the `spots` table they
  appear with `cell == -1`. Homeless alleles likewise never pass a cell gate.
- **Do not hand-build empty DataFrames into these tables.** The build itself
  refuses to concatenate zero-row frames because one all-object empty frame
  downgrades the numeric dtypes of the entire population table (`np.sqrt` on an
  object column then fails); `spots` dtypes are additionally enforced after concat.
