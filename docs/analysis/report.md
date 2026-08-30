# Reports and provenance

A map without its population is not a result. The reporting layer in
`codelab_pipeline/analysis/report.py` exists so that every figure that leaves the
pipeline carries its own evidence: the exact gate that selected the cells, how many
survived each predicate, how many per celltype, and the parameters the caller used.
Like everything in the analysis toolbox, it is headless — importing it loads no Qt,
and the app's "Save Result..." button calls the very same function you call from a
script.

Two public functions do all the work: `gate_summary` builds the provenance block,
and `save_result` writes it to disk next to the figure and its tables.

## `gate_summary(pop, condition)` — the provenance block

```python
def gate_summary(pop, condition):
```

`pop` is a `Population`; `condition` is a `gate.Condition` or `None`. The return
value is a plain dict, safe to `json.dump`:

```python
{'gate':          {'clauses': [[{'kind': 'celltype_in', 'names': ['WT']}, ...]]},
 'sequential':    [['CelltypeIn(names=[\'WT\'])', 812],
                   ['AlleleCount(lo=1, hi=None, min_bins=2)', 640]],
 'n_cells_total': 2172,
 'n_cells_gated': 640,
 'by_celltype':   {'WT': [640, 900], 'Unassigned': [0, 130]}}
```

The pieces, exactly as computed:

- **`gate`** is the *whole* serialized condition, `condition.to_dict()` — an
  OR-of-AND structure keyed `'clauses'`, where each clause is a list of predicate
  dicts. It round-trips: `gate.Condition.from_dict(block['gate'])` rebuilds the
  gate, so a saved sidecar can be re-run headless. When `condition` is `None` or
  has no predicates, the block records `{'clauses': []}` and the mask is all-True.
- **`sequential`** comes from `Condition.report(pop)`: within a clause, each entry
  is `[repr(predicate), n_cells_surviving_the_AND_so_far]`; between clauses a
  marker row `['-- OR --', running_union_count]` appears, and the next clause's
  counts start fresh. This is the survivor waterfall the GUI shows.
- **`n_cells_total` / `n_cells_gated`** are the length of `pop.cells` and the sum
  of the final cell mask. Cells are always keyed by the pair `(fov, cell)`;
  the counts are over that row order.
- **`by_celltype`** maps each celltype name to a two-element list
  `[gated, total]`. The empty celltype `''` is reported as `'Unassigned'`, never
  dropped — a figure that quietly excluded the unclassified cells would be lying
  about its denominator.

Because range predicates treat NaN as honest missingness (a cell with no value for
a gated quantity *fails* the gate — it is not known to be in range), the sequential
counts already reflect missing data; nothing needs to be re-derived later.

## `save_result(...)` — PNG + one CSV per table + JSON sidecar

```python
def save_result(out_dir, name, fig=None, tables=None, pop=None,
                condition=None, params=None, dpi=150):
```

It creates `out_dir` if needed, then writes up to three kinds of file and returns
the list of written paths (the JSON sidecar is always last):

- **`<name>.png`** — only if `fig` is given; saved with
  `fig.savefig(path, dpi=dpi, facecolor='white', bbox_inches='tight')`.
- **`<name>__<table>.csv`** — one per entry in the `tables` dict
  (`{table_name: object}`). A `DataFrame` is written with `to_csv(index=False)`;
  a `Series` via `to_frame().to_csv()` (its index is kept); a numpy array of
  ndim ≤ 2 via `np.savetxt` with format `%.6g`, comma-delimited; an array of
  higher rank is flattened one slab per block, each block preceded by a
  `# slab i` comment line (read back with `pd.read_csv(..., comment='#')`).
- **`<name>.json`** — always written, even with no figure and no tables. It
  contains `name` (the base actually used, see below), `saved_at` (ISO timestamp
  to the second), and `params` (your dict, or `{}`). If `pop` is given it adds a
  `population` block — `storage_path`, `fovs`, `voxel_um`, and the human-readable
  `pop.summary()` string — and a `gate` block, which is exactly
  `gate_summary(pop, condition)`.

**Names never collide silently.** Before writing, if either `<name>.png` or
`<name>.json` already exists in `out_dir`, the base becomes `<name>_1`, then
`<name>_2`, and so on until both are free. All files of one save share the chosen
base, and the sidecar's `name` field records it — so when reading back, trust
`side['name']`, not the name you asked for. There is deliberately no atomic-write
machinery here: these are exports, not the store.

The sidecar is what makes a figure self-describing. A PNG found on disk months
later sits next to a JSON that answers, by itself: which store and FOVs it came
from, what the voxel size was, the exact gate as re-runnable data, how many cells
passed each predicate in order, how many per celltype, and every parameter the
analysis was called with.

## How the app saves the same thing

Every analysis view window in the GUI (`canvas/analysis_figure_displayer.py`)
carries a **Save Result... (PNG + CSV + provenance JSON)** button. The panel that
produced the figure stashes the figure, its tables, the `Population`, the active
`Condition`, and the parameters when it calls `set_figure(...)`; the button opens a
directory picker and hands that payload straight to `report.save_result`. Nothing
is recomputed at save time, and there is no app-only format: a save from the GUI
and a save from your script are byte-for-byte the same function.

## Worked example: an ensemble map, saved headless

This builds a population from the store, gates it, saves the ensemble distance map
with full provenance, and reads the sidecar back. Positions in the population are
in micrometres (scaled once at extraction from `voxel_um=(0.208, 0.208, 0.2)`);
the maps are therefore in µm too.

```python
import json

from codelab_pipeline.io import preprocess
from codelab_pipeline.analysis import ensemble, figures, gate, population, report

STORE = r'E:/Students/2026-08-07-SG-test/DNA'

def main():
    records = preprocess.parse_experiment_layout(r'E:/.../layout.xlsx')
    pop = population.Population.build(STORE, fovs=range(34),
                                      records=records, jobs=4)

    cond = gate.Condition([gate.CelltypeIn(['WT']),
                           gate.AlleleCount(lo=1, min_bins=2)])
    amask = cond.allele_mask(pop)          # allele-level projection of the gate
    dmaps = pop.dmaps()                    # (n_alleles, n_bins, n_bins), um

    m, counts = ensemble.ensemble_map(dmaps, amask, reducer='median', min_n=5)
    fig = figures.fig_ensemble(dmaps, mask=amask, title='WT ensemble',
                               reducer='median', min_n=5,
                               bin_ids=pop.alleles['bin_ids'])

    written = report.save_result(
        r'D:/exports/wt_run', 'wt_ensemble',
        fig=fig,
        tables={'map': m, 'counts': counts},
        pop=pop, condition=cond,
        params={'reducer': 'median', 'min_n': 5})

    with open(written[-1]) as f:           # the sidecar is always last
        side = json.load(f)
    g = side['gate']
    print(side['name'])                                    # wt_ensemble (or _1, _2...)
    print(g['n_cells_gated'], '/', g['n_cells_total'])
    for pred_repr, n_after in g['sequential']:
        print(f'  {n_after:6d}  after {pred_repr}')
    rerun = gate.Condition.from_dict(g['gate'])            # the gate, re-runnable

if __name__ == '__main__':
    main()
```

The `if __name__ == '__main__':` guard is mandatory on Windows whenever
`Population.build` runs with `jobs != 1` — the workers start via multiprocessing
spawn, which re-imports the script.

## Pitfalls

- **The gate block requires `pop`.** Passing `condition` without `pop` writes a
  sidecar with *no* `gate` and *no* `population` block — the condition is silently
  ignored. Always pass both when you want provenance.
- **Read `side['gate']['gate']['clauses']`, never a `predicates` key.** Conditions
  serialize as OR-of-AND under `'clauses'`; the source records that reaching for
  `['predicates']` was a real KeyError that once escalated to a no-traceback abort
  inside the GUI. Use `gate.Condition.from_dict(...)` rather than parsing by hand.
- **The name you asked for may not be the name on disk.** No-overwrite suffixing
  can turn `wt_ensemble` into `wt_ensemble_1`; the authoritative base is
  `side['name']`, and `save_result`'s return value lists the actual paths.
- **`params` must be JSON-serializable.** `json.dump` raises `TypeError` on numpy
  scalars and arrays; convert with `int()`/`float()`/`.tolist()` before passing.
  The counts inside `gate_summary` are already plain `int`.
- **CSVs are exports, not the record.** Arrays are written at `%.6g` precision,
  NaN pixels (honest missingness — e.g. ensemble pixels observed in fewer than
  `min_n` alleles) appear as literal `nan` text, and 3-D stacks become `# slab i`
  blocks. Save the `counts` array alongside the map so the NaNs stay explainable.
- **A `Series` keeps its index in the CSV; a `DataFrame` does not.** If the index
  is meaningful (e.g. a `(fov, cell)` MultiIndex), reset it into columns first so
  the pair key survives as data.
- **An empty gate is recorded as `{'clauses': []}`** with an empty `sequential`
  list and `n_cells_gated == n_cells_total` — distinguish "ungated" from "gated
  and everything passed" by looking at the clauses, not the counts.
