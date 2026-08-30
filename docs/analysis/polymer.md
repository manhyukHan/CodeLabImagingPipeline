# Polymer: from traced alleles to distance maps

`codelab_pipeline.analysis.polymer` turns the alleles persisted in a store into
polymer position tables and pairwise distance maps, and applies ORCA-style
quality control to them. It is headless: it imports nothing from the app and no
Qt, so you can run everything below from a plain script or notebook pointed at
a storage path such as `E:/Students/2026-08-07-SG-test/DNA`.

Two conventions run through the whole module. Coordinates are ordered
`(y, x, z)` — y and x in pixels, z as a plane index — until they are scaled to
micrometres exactly once, per axis, by `voxel_um` (default
`DEFAULT_VOXEL_UM = (0.208, 0.208, 0.2)`; note z planes are *not* the same size
as pixels, so the axes must scale independently). And NaN always means honest
missingness — a rejected or absent bin — never a fabricated value.

An end-to-end run looks like this:

```python
import numpy as np
from codelab_pipeline.analysis import polymer

store = 'E:/Students/2026-08-07-SG-test/DNA'

records = polymer.records_for(store)          # layout, via the store's manifest
hybes = polymer.bin_hybes(records)            # ordered genomic bins (H rounds only)

table = polymer.fov_polymer_table(store, 7, hybes)     # one FOV, um-scaled
dmaps = polymer.polymer_distmaps(table['pos_um'])      # pre-QC distance maps
thr = polymer.qc_thresholds(table, dmaps)              # data-derived thresholds
qc = polymer.apply_qc(table, dmaps, thr, min_traced=2)

median_map = np.nanmedian(qc['dmaps'], axis=0)         # (n_bins, n_bins) summary
```

## Finding the layout and the genomic bins

`records_for(storage_path, modality=None)` is the headless entry point for the
experiment layout. It reads the store manifest in the *parent* of
`storage_path`, looks up the modality (defaulting to the basename of
`storage_path`, e.g. `'DNA'`), and parses the `layout_path` the manifest
records. It raises `ValueError` (including the manifest content) when the
manifest records no layout for that modality, and `FileNotFoundError` when the
layout file is unreachable — the layout lives on the acquisition share, which
is not always mounted, and the module says so rather than guessing. The return
value is the list of per-hybe record dicts from
`preprocess.parse_experiment_layout`: each has `folder`, `readout_id`,
`datatype`, `hybe_num`, `channels`, `fiducial_channel`, `channel_layout`,
`total_frames`, `readout_name`.

`genomic_bins(records)` selects only the records whose `datatype` is `'H'` and
returns `[(readout_id, hybe_folder)]` sorted by `readout_id`. The layout is
authoritative for bin identity and order: `readout_id` is the genomic-locus
axis every distance map shares, and it is *not* the acquisition order
(`hybe_num`) or the folder-name order. `bin_hybes(records)` is the same list
reduced to the bare hybe folder names — this is the `hybes` argument every
downstream function takes.

R and T rounds are **never** polymer bins. `qc_rounds(records)` returns them
separately as `{'repeats': [(readout_id, hybe)], 'toes': [...]}`, each list
sorted by `readout_id`: an R round re-images an H bin (same `readout_id`) as a
repeatability check, and T rounds carry identity markers. Keep them out of the
`hybes` list you pass to the table builders.

## Selectors: collapsing multi-candidate bins

A traced allele stores, per hybe, a *list* of candidate spots in
`polymer_adj` — the `'adj'` suffix meaning the shared reference frame, as
opposed to `'raw'` coordinates in each hybe's own frame. Sister chromatids and
ambiguous fits mean a bin can carry more than one candidate, and a polymer
needs exactly one position per bin. A **Selector** is any callable with this
contract:

```python
selector([(y, x, z, amplitude), ...]) -> (y, x, z, amplitude)
```

It is called only on non-empty candidate lists. The default,
`max_brightness(candidates)`, returns the brightest candidate whole
(`max(candidates, key=lambda c: c[3])` — amplitude is index 3 of every
candidate tuple). This is the same rule `AnAllele.final_polymer` has always
cited (`QualityControlORCA.combineFOV`, `'maxBrightness'`), but here it is an
ordinary function argument rather than a string switch, so swapping in a
nearest-to-anchor or amplitude-weighted-centroid rule is a one-line change at
the call site:

```python
def nearest_to(anchor):
    def sel(cands):
        return min(cands, key=lambda c: sum((a - b) ** 2
                                            for a, b in zip(c[:3], anchor)))
    return sel

table = polymer.fov_polymer_table(store, 7, hybes, selector=nearest_to((512, 512, 60)))
```

A composite selector may return a synthetic amplitude (e.g. the sum over
candidates); the QC brightness gates then act on whatever the selector
reported.

`collapse_polymer(allele_dict, hybes, selector=max_brightness)` applies the
selector to one allele and returns `(pos, amp, n_cand)`: `pos` is
`(n_bins, 3)` float in `(y, x, z)` order, `amp` is `(n_bins,)`, and `n_cand`
is `(n_bins,)` int32 counting candidates per bin. Bins with no candidate stay
NaN in `pos`/`amp` and 0 in `n_cand`. Positions are still in pixels/planes at
this level — the caller scales to micrometres once, which keeps
`collapse_polymer` unit-agnostic and testable.

## The per-FOV table

`fov_polymer_table(storage_path, fov, hybes, voxel_um=DEFAULT_VOXEL_UM,
selector=max_brightness)` reads every stored allele of one FOV and stacks the
collapsed polymers into a plain dict (picklable, so it maps cleanly over FOVs
in a worker pool). The keys:

| key | shape / type | meaning |
|---|---|---|
| `pos_um` | `(n_alleles, n_bins, 3)` float | `(y, x, z)` in **micrometres** |
| `amp` | `(n_alleles, n_bins)` float | selector-reported amplitude, NaN where absent |
| `n_cand` | `(n_alleles, n_bins)` int32 | candidates per bin (sister chromatids show up here) |
| `allele_id` | `(n_alleles,)` int64 | stored allele id |
| `cell` | `(n_alleles,)` int64 | owning cell id, `-1` when the allele is homeless |
| `fov` | `(n_alleles,)` int64 | the FOV, repeated |
| `celltype` | `list[str]` | joined via this FOV's cells; `''` when unknown or homeless |
| `n_traced` | `(n_alleles,)` int32 | finite bins per allele |

Scaling happens here and only here: each axis of `pos_um` is multiplied by the
matching component of `voxel_um`. Cell identity is per-FOV — a cell is always
the pair `(fov, cell)`, which is why the table carries both columns; a bare
cell id means nothing across FOVs.

Alleles with zero traced bins are **kept** as all-NaN rows. Dropping them is a
QC decision (`apply_qc` with `min_traced`), not an extraction side effect.

## Distance maps

`polymer_distmaps(pos_um)` returns `(n_alleles, n_bins, n_bins)` pairwise
Euclidean distances in micrometres, as float32. NaN positions propagate to NaN
rows and columns of the corresponding map — exactly the ORCA behaviour, and
exactly what "honest missingness" requires. The computation is chunked over
alleles so peak scratch memory stays around 50 MB even at tens of thousands of
alleles (the naive float64 broadcast peaked at ~1.1 GB for a 738 MB result at
24k alleles x 62 bins); micrometre distances do not need float64 precision.

## Quality control

`qc_thresholds(table, dmaps, brightness_q=(0.05, 0.95), jump_q=0.75,
max_dist_q=0.95)` derives thresholds from *this* dataset's quantiles, in the
style of `polymeric_qc`, and returns
`{'min_brightness', 'max_brightness', 'max_jump_um', 'max_dist_um'}`. The
brightness bounds are quantiles of the finite amplitudes; `max_jump_um` is the
`jump_q` quantile of finite genomically-adjacent distances (the first
superdiagonal of each map); `max_dist_um` is the `max_dist_q` quantile of
per-bin median distances. When an input is empty, the fallbacks are permissive
(`0.0` / `inf`). Thresholds are derived, never inherited constants — every
constant this repo ported across datasets has been wrong in practice.

`apply_qc(table, dmaps, thresholds, min_traced=2)` marks a bin bad when any of
three gates fires:

1. **Amplitude window** — the amplitude is finite and lies outside
   `[min_brightness, max_brightness]`. Only the amplitude is tested; the
   original ORCA code compared all four tuple components elementwise, so any
   *coordinate* smaller than the 5th-percentile brightness was falsely
   flagged. That bug is fixed here.
2. **Both-neighbor jump** — for interior bins (applied only when
   `n_bins >= 3`), both genomic neighbors are farther than `max_jump_um`.
   One long edge is tolerated; two brands the bin an outlier.
3. **Median distance** — the bin's median distance to all bins (nanmedian over
   each map's rows) is finite and exceeds `max_dist_um`.

Bad bins become NaN in copies of `pos_um` and `amp`; the inputs are never
mutated. An allele is kept only with at least `min_traced` finite bins
remaining (ORCA used `> 1`, i.e. the same as the default `min_traced=2`). The
returned dict:

- `pos_um`, `amp` — filtered copies, **kept rows only**;
- `bads` — `(n_alleles, n_bins)` bool over the **input** rows;
- `kept` — `(n_alleles,)` bool over the **input** rows;
- `index` — `np.flatnonzero(kept)`, mapping each output row back to its input
  row (and hence to `table['allele_id']`, `table['cell']`, …);
- `dmaps` — distance maps **recomputed** from the filtered positions of the
  kept alleles (`(n_kept, n_bins, n_bins)` float32).

Zero alleles in is a legitimate outcome, not an error: the empty-input guard
returns correctly shaped empty arrays instead of the `IndexError` the jump
gate's diagonal stack used to raise.

Two small summaries close the loop: `efficacy(pos_um)` returns `(n_bins,)`,
the fraction of alleles with a finite position per bin (zeros when there are
no alleles), and `completeness(pos_um)` returns `(n_alleles,)`, the count of
finite bins per allele. Run them before and after `apply_qc` to see what the
gates cost you.

## Scaling beyond one FOV

The table dicts are designed to be concatenated:

```python
tables = [polymer.fov_polymer_table(store, f, hybes) for f in range(34)]
pos = np.concatenate([t['pos_um'] for t in tables])
```

For parallel multi-FOV assembly, `Population.build` in
`codelab_pipeline.analysis.population` wraps this loop (pass it `records` or
`hybes` directly). On Windows, any script that calls `Population.build` with
`jobs != 1` **must** live under an `if __name__ == '__main__':` guard —
multiprocessing uses spawn, which re-imports the script in every worker.

## Pitfalls

- **Pass `bin_hybes(records)`, never a raw folder list.** R and T rounds can
  never be polymer bins; if their folders leak into `hybes`, whatever
  `polymer_adj` holds for them is silently promoted to a genomic bin and every
  map's locus axis is wrong.
- **Bin order is `readout_id` order**, not acquisition order and not
  alphabetical folder order. Two stores with the same loci but different
  imaging schedules still produce comparable maps.
- **`qc_thresholds` and `apply_qc` want the *pre-QC* maps** from
  `polymer_distmaps(table['pos_um'])`. Post-QC maps come out of `apply_qc`
  itself (`qc['dmaps']`), already recomputed from the filtered positions.
- **Row bookkeeping after QC:** `pos_um`/`amp`/`dmaps` in the result contain
  kept rows only, while `bads` and `kept` index the input. Use
  `qc['index']` to pull matching metadata, e.g.
  `table['allele_id'][qc['index']]`.
- **The amplitude column is whatever the selector returned.** A custom
  selector reporting a synthetic amplitude changes what the brightness gates
  measure; derive thresholds from the same table you filter.
- **Do not rescale.** `pos_um` is already micrometres, scaled once at
  extraction; z was scaled by 0.2 um/plane, not the 0.208 um pixel pitch, so
  a second uniform scaling would distort z anisotropically.
- **NaN is meaningful everywhere.** All-NaN alleles are kept by
  `fov_polymer_table` on purpose; expect NumPy `All-NaN slice` warnings from
  the nanmedian steps when they are present — they are benign, and the gates
  themselves ignore non-finite medians.
- **Empty inputs are shaped, not fatal:** `apply_qc` on a zero-allele table
  and `efficacy` on an empty stack both return empty/zero arrays of the right
  shape, so per-FOV loops need no special-casing for FOVs that have no
  alleles yet.
- **The layout may be unreachable.** `records_for` reads the layout path from
  the store manifest; if the acquisition share is not mounted it raises
  `FileNotFoundError` naming the path rather than guessing — mount the share
  or copy the layout and update the manifest, but do not substitute a
  different layout silently.
