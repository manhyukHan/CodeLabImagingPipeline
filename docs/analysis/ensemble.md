# Ensemble maps and FOV consistency

Once a `Population` has produced its per-allele distance maps, the questions become
statistical: what does the *typical* map look like, do two groups of cells differ, do
the fields of view agree with each other, and do a cell's two alleles behave like
independent draws? The module `codelab_pipeline/analysis/ensemble.py` answers those
questions on plain numpy arrays, and `codelab_pipeline/analysis/figures.py` renders the
answers as matplotlib (Agg) figures. Both are headless by construction: no Qt, no app
modules — you can drive everything from a script or a notebook against the store.

The inputs everywhere are the same three arrays, one row per allele:

- `dmaps` — `(n_alleles, n_bins, n_bins)` pairwise distance maps in **micrometres**
  (positions were scaled once at extraction with `voxel_um = (0.208, 0.208, 0.2)`).
  Obtain them from `Population.dmaps()`.
- `fovs` — `(n_alleles,)` integer FOV of each allele.
- `cells` — `(n_alleles,)` integer cell id *within its FOV*. A cell is identified only
  by the pair `(fov, cell)`; bare cell ids do not exist in this package. `cell < 0`
  marks a homeless allele (one that sits in no segmented cell).

An unobserved bin pair is `NaN` in an allele's map — honest missingness, never a
fabricated value — and every function below propagates that honestly.

One rule about masks, stated once because every function follows it: **flags are
groupings, not filters**. A boolean `mask` over alleles is the *gate*; splits by FOV,
celltype, or group happen *after* the gate, on the same gated stack. Asking to "show
the FOV differences" can never change which alleles are in the analysis.

```python
from codelab_pipeline.analysis.population import Population
from codelab_pipeline.analysis import ensemble as ens
from codelab_pipeline.analysis import figures as figs

if __name__ == '__main__':          # REQUIRED on Windows when jobs != 1
    pop = Population.build('E:/Students/2026-08-07-SG-test/DNA',
                           fovs=range(34), records=records, jobs=8)
    dmaps = pop.dmaps()                       # (n_alleles, n_bins, n_bins), um
    fovs = pop.alleles['fov']
    cells = pop.alleles['cell']
    bin_ids = pop.alleles['bin_ids']          # readout id per bin, for tick labels
```

## The ensemble map: `ensemble_map`

```python
def ensemble_map(dmaps, mask=None, reducer='median', min_n=1)
```

Collapses the (optionally gated) allele stack to one map. Returns a tuple
`(map, counts)`, both `(n_bins, n_bins)`; `counts` is an integer array reporting, for
every pixel, how many gated alleles actually observed that bin pair.

- `mask` is a boolean `(n_alleles,)` gate; `None` means all alleles.
- `reducer` is `'median'` (the default — `np.nanmedian`) or anything else for
  `np.nanmean`. Median is the house default, carried from the reference work.
- `min_n` masks under-observed pixels: any pixel with `counts < min_n` is set to `NaN`
  in the returned map. The counts are *not* masked — they always report the truth, so
  you can see exactly how thin the evidence was.

An empty selection (mask keeps nothing) returns an all-`NaN` map and an all-zero
counts array rather than raising.

```python
m, counts = ens.ensemble_map(dmaps, mask=my_gate, min_n=5)
```

## Group differences: `subtraction_map`

```python
def subtraction_map(dmaps, mask_a, mask_b, reducer='median', min_n=1)
```

Computes `ensemble_map(A) - ensemble_map(B)` and returns `(diff, counts_a, counts_b)`.
A pixel is `NaN` in the difference whenever **either** group observed it fewer than
`min_n` times — a comparison is only shown where both sides have evidence.

## Map similarity: `scc`

```python
def scc(map_a, map_b, h=1, max_stratum=None)
```

Stratum-adjusted correlation between two maps (the HiCRep statistic, ported from the
ORCA work). Each off-diagonal `|i - j| = d` is a stratum; per stratum it takes the
Pearson correlation over entries finite in **both** maps, then combines strata with
weights `n_d * std_a * std_b`. `h` applies a NaN-aware `(2h+1)^2` box smoothing first
(`h=0` disables). Returns a float in `[-1, 1]`, or `NaN` when no stratum has at least
3 shared finite entries with nonzero spread. Kept for completeness — the QC figure
below deliberately no longer uses it (see `fov_msd_test`).

## Per-FOV ensembles: `fov_consistency`

```python
def fov_consistency(dmaps, fovs, mask=None, reducer='median', min_n=1, h=1)
```

Splits the *gated* stack by FOV — a flag, not a filter — builds one ensemble map per
FOV, and correlates every pair with `scc`. Returns a dict:

```python
{'fovs':   [int, ...],            # sorted FOVs present after the gate
 'maps':   {fov: (n_bins, n_bins)},
 'counts': {fov: (n_bins, n_bins) int},
 'scc':    (n_fov, n_fov) float}  # symmetric, 1s on the diagonal
```

## The tested version: `fov_msd_test`

```python
def fov_msd_test(dmaps, fovs, mask=None, max_pairs_per_class=20000,
                 min_shared=10, seed=0)
```

The SCC matrix is one correlation per FOV pair, and at ~10^2 alleles per FOV the two
ensemble maps being correlated are grainy — the matrix is weak evidence either way.
`fov_msd_test` instead treats FOV deviation as a **tested distribution**. For a random
sample of allele *pairs* it computes the MSD between the two alleles' maps — the mean
squared difference over the pair's **shared finite** upper-triangle entries, discarded
as `NaN` when fewer than `min_shared` entries are shared — and builds that distribution
separately for **in-FOV** pairs (both alleles from the same FOV) and **cross-FOV**
pairs. Up to `max_pairs_per_class` pairs are drawn per class (seeded, `seed=0`), and a
Welch t-test compares the two distributions. Consistent FOVs give in ≈ cross (large
p); a batch effect separates them.

The return value:

```python
{'msd_in':    (n_in,)  float,     # in-FOV pairwise MSDs, um^2
 'msd_cross': (n_cx,)  float,
 't': float, 'p': float, 'neglog10p': float,
 'per_fov': [{'fov': int, 'n_with': int, 'n_without': int,
              't': float, 'p': float, 'signed_neglog10p': float}, ...]}
```

Fewer than 3 gated alleles returns empty arrays and `NaN` statistics.

### The per-FOV signed verdict

Naming *which* FOV deviates needs the right comparison, and the obvious one fails.
Testing each FOV's in-pairs against its cross-pairs cannot isolate a deviant: one
shifted FOV inflates **every** other FOV's cross-class too, so all FOVs saturate
together (measured in this codebase: three FOVs, one planted deviant, all three at
`-log10 p = 300`).

The discriminating split is over cross-pairs only: cross-pairs **involving** FOV `f`
against cross-pairs **not involving** `f`, Welch-tested and **signed** —
`signed_neglog10p = sign(t) * -log10(p)`. Only the deviant FOV's own cross-pairs are
systematically larger than the rest, so it alone scores strongly *positive*; its
neighbours score *negative*, because their with-`f` mixture sits below the without-`f`
pool. Read the verdict as: large positive = the deviant, mildly negative = its
victims.

One honesty note, recorded in the source so no reader rediscovers it: sampled pairs
share alleles, so observations are not independent. The p-values are honest **ranking**
scores and anticonservative in absolute terms. Use them to order and flag FOVs, not to
quote calibrated error rates.

## Allele differences: `allele_difference`

```python
def allele_difference(dmaps, fovs, cells, mask=None, rng=None, n_null=1000)
```

There is deliberately **no stable allele indexing** within a cell — nothing makes one
allele "allele 1" — so only the *difference* between a cell's alleles is meaningful,
and this function computes exactly that. For every gated cell (keyed by
`(fov, cell)`) holding at least 2 alleles, every allele pair contributes
`mean |d1 - d2|` over the bin-pair entries finite in **both** maps (`NaN` if fewer than
3 are shared). The null is the same statistic over `n_null` randomly drawn cross-cell
pairs (default RNG is `np.random.default_rng(0)`; pass `rng` to override). Homeless
alleles (`cell < 0`) can never form a within-cell pair.

```python
{'within':          (n_pairs,) float,    # um
 'within_cells':    [(fov, cell), ...],  # one key per within pair
 'null':            (<= n_null,) float,
 'n_multi_allelic': int}
```

## The figures

All figure helpers live in `figures.py`, force the Agg backend at import, and return a
`matplotlib.figure.Figure` — you decide whether it becomes a PNG or lands in a canvas.
The distance-map conventions are uniform and worth internalizing once:

- colormap is `seismic_r`, so **red means close** and blue means far;
- color limits default to the **2% / 98% quantiles** of the map's finite values;
- `NaN` pixels are drawn **dark** (`set_bad('0.15')`) — a pixel with too few
  observations *looks* different, it is never silently averaged;
- every figure states its gate and its N in the title, because a map without its
  population is not a result.

`style_ax(ax)` applies the house style to every non-image axes — top and right spines
removed — and returns the axes. `dmap_ticks(ax, n, ids=None)` sets distance-map ticks
at `[0] + every 10th bin + [n-1]`; with `ids` (the layout's readout id per bin, in bin
order, length exactly `n`) the labels are the **real readout indices** the experiment
layout gives each round — gaps and all — and the axis is labelled `readout id`. Without
`ids` the fallback labels are the 1-based bin position under the label `barcode`. Pass
`bin_ids=pop.alleles['bin_ids']` to get real readout labels.

```python
fig = figs.fig_ensemble(dmaps, mask=gate, title='chr2 locus',
                        min_n=5, bin_ids=bin_ids)
fig.savefig('ensemble.png', dpi=150)
```

`fig_ensemble(dmaps, mask=None, title='ensemble', reducer='median', min_n=1,
group_masks=None, bin_ids=None)` draws one ensemble map, or — with
`group_masks={'name': bool_mask, ...}` — one panel per named group. Groups split the
gated stack, they never re-gate. All panels are forced onto a **shared** color scale
(the widest of the individual 2/98% limits), with one colorbar labelled
`distance (um)` and the reducer and `min_n` stated in the suptitle.

`fig_fov_consistency(dmaps, fovs, mask=None, min_n=1, group_masks=None,
show_maps=True, bin_ids=None)` is the FOV-level QC view. SCC is gone from this figure
per explicit decision — one correlation between two grainy ensembles is meaningless
next to the MSD distributions, which carry thousands of allele pairs. Each group row
shows the per-FOV ensemble maps (skipped entirely with `show_maps=False`) and the
`fov_msd_test` histogram: cross-FOV pairs in grey, in-FOV pairs in crimson, on shared
bin edges, with the overall `-log10 p` and the wrapped per-FOV signed verdicts
(`+ = deviant`) in the panel title. Groups emptied by the gate get no row — their names
are listed in the suptitle instead — and a group with too few pairs gets a labelled
blank panel rather than phantom axes.

`fig_allele_difference(dmaps, fovs, cells, mask=None)` runs `allele_difference` and
overlays the within-cell histogram (crimson) on the cross-cell null (grey), both
density-normalized, titled with the multi-allelic cell count.

## Pitfalls

- **`counts` are never masked.** `ensemble_map` masks the *map* below `min_n` but the
  counts array always reports true observation numbers. Do not treat a finite count as
  "this pixel is shown".
- **`fov_msd_test` p-values rank, they do not calibrate.** Sampled pairs share
  alleles; the Welch p is anticonservative. A permutation upgrade is planned behind
  the same return shape, but today treat `neglog10p` as a score.
- **Per-FOV in-vs-cross is the wrong test** — one deviant FOV saturates every FOV's
  verdict. Use the shipped signed with-`f` / without-`f` split and read the sign.
- **The per-FOV dict keys are** `n_with` / `n_without` / `signed_neglog10p` (what the
  code returns), not the `n_in` / `n_cross` a quick docstring skim might suggest.
- **Empty inputs degrade, they do not raise**: an empty gate gives an all-NaN map;
  fewer than 3 alleles gives an empty `fov_msd_test`; `scc` returns `NaN` when no
  stratum has 3 shared finite entries. Check for `NaN` before formatting results.
- **`allele_difference` nulls can be short**: within-cell draws are skipped, and pairs
  sharing fewer than 3 finite bin entries yield `NaN`, so filter with
  `res['null'][np.isfinite(res['null'])]` (the figure helper already does).
- **Windows multiprocessing**: any script calling `Population.build` with `jobs != 1`
  must live under an `if __name__ == '__main__':` guard, or the spawn-based workers
  will re-execute your module. The ensemble functions themselves are single-process
  numpy and safe anywhere.
- **`dmap_ticks` silently falls back** to 1-based bin positions if `ids` is `None` *or
  the wrong length* — if your axis says `barcode` when you expected `readout id`,
  check `len(bin_ids) == n_bins`.
