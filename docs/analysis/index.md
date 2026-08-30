# The analysis toolbox

`codelab_pipeline.analysis` is the pipeline's analysis layer, and it is
**independent by design**: importing it loads no Qt and no app module (a
promise pinned by its own test suite), every function takes plain data
or a storage path, and everything the app's Analysis tab shows can be
produced from a script or notebook. The app *uses* this package; it
never owns it.

## The shape of an analysis

```python
import sys
sys.path.insert(0, r'path/to/CodeLabImagingPipeline')

from codelab_pipeline.analysis import population, polymer, gate, ensemble, figures

STORE = r'\\nas\experiments\2026-08-07-SG-test\DNA'

def main():
    records = polymer.records_for(STORE)          # the layout, from the manifest
    pop = population.Population.build(
        STORE, fovs=[1, 2, 4, 5], records=records,
        sources=[('RNA', 'Hyb_103', 635), ('DNA', 'Hyb_016', 555)])

    cond = gate.Condition(clauses=[[
        gate.CelltypeIn(['4A3', '8B1']),
        gate.CompletenessRange(lo=10),
    ]])
    amask = cond.allele_mask(pop)

    dmaps = pop.dmaps()[amask]
    fig = figures.fig_ensemble(dmaps, title='gated ensemble',
                               bin_ids=pop.alleles.get('bin_ids'))
    fig.savefig('ensemble.png', dpi=150)

if __name__ == '__main__':     # REQUIRED: build() spawns worker processes
    main()
```

Universal conventions, restated once here and honored everywhere:

- Coordinates are `(y, x, z)` — y/x in pixels, z in plane index — and
  analysis tables carry positions in **micrometres**, scaled exactly
  once at extraction (`voxel_um`, default `(0.208, 0.208, 0.2)`).
- A cell is keyed by the pair `(fov, cell)`. Bare cell ids do not exist.
- A source is the triple `(modality, hybe, channel)`.
- `adj` coordinates live in the shared reference frame, `raw` in the
  hybe's own frame.
- NaN is honest missingness, never a fabricated value.
- Scripts calling `Population.build` with more than one worker must sit
  under `if __name__ == '__main__':` on Windows.

## Module tour

| Page | What it covers |
|---|---|
| {doc}`population` | one build over the store → plain tables; the append-mode attribute cache |
| {doc}`polymer` | traced alleles → distance maps; bins, selectors, QC, efficacy, repeat/toe rounds |
| {doc}`gates` | predicates and OR-of-AND conditions; cell-level vs allele-level |
| {doc}`expression` | spot- and mask-based per-cell expression; normalization semantics |
| {doc}`distances` | pair distances within cells; per-cell collapse; histograms |
| {doc}`ensemble` | ensemble maps, FOV-consistency MSD test, allele differences |
| {doc}`detection` | is barcode co-occurrence real? two calibrated nulls |
| {doc}`resolvers_reconcile` | frame transforms from the store alone; re-deriving adj from raw |
| {doc}`report` | gate summaries and self-describing saved results |

```{toctree}
:maxdepth: 1
:hidden:

population
polymer
gates
expression
distances
ensemble
detection
resolvers_reconcile
report
```
