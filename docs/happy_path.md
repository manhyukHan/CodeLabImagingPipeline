# The standard path: raw images to gated figures

This is the happy path through the app, one tab at a time, in the order
the tabs are laid out. Screenshots come from a real experiment (MP58: a
DNA chromatin-tracing store with an RNA modality, 111 rounds per FOV,
1024×1024 MIPs) rendered by the app itself.

Every stage writes into the same **v2 project store**: one project root
holding `manifest.json` plus one directory per modality, with per-FOV
capsules for cells, alleles, spots, matrices, and computed attributes.
Every write goes through an atomic door (`.part` file + rename), so an
interrupted run never leaves a half-written capsule that later reads as
truth — re-launch and re-run; append modes skip what is already complete.

## 1. Ingestion

```{image} _static/screenshots/tab_0_ingestion.png
:alt: Ingestion tab
:width: 100%
```

Name the **project root** once — the manifest does the rest. Register
each modality (e.g. `DNA`, `RNA`) with its experiment-layout spreadsheet
and raw DAX directory, list the FOVs, and ingest. Raw camera stacks
become per-hybe HDF5 z-stacks plus per-channel MIPs, written FOV-major
through one worker pool (worker count is measured, not guessed: on the
lab NAS 12 workers beat 36 by a factor of two).

Append mode tests **completeness, not existence**: a truncated HDF5 file
opens happily and lies about its shape, so only reading its first and
last element distinguishes complete from cut short. Re-running ingestion
after an interruption fills exactly the holes.
`tools/verify_store.py <storage_path>` read-probes a whole store and
reports broken files, orphaned MIPs, and leftover `.part` files.

## 2. Cell Segmentation

```{image} _static/screenshots/tab_1_cell_segmentation.png
:alt: Cell Segmentation tab
:width: 100%
```

Pick the reference hybe/channel to segment on, the method (Cellpose on
GPU when available, or the classical fallback), and the size bounds,
then segment the FOVs. Cells land in each FOV's `cells.h5` capsule with
their masks; everything downstream keys them as `(fov, cell)` pairs.
Append mode skips FOVs that already carry cells.

## 3. Alignment

```{image} _static/screenshots/tab_2_alignment.png
:alt: Alignment tab
:width: 100%
```

Three layers, all stored as per-FOV matrices:

- **FOV alignment (within modality)** — every hybe registered to the
  modality's reference hybe on the fiducial channel.
- **Cross-modal alignment** — modalities meet through a hub-and-bridge
  design: the first-activated modality is the hub, and every other
  modality stores one bridge to it through its own bridge hybe.
- **Cell alignment** — per-cell residual refinement on top of the FOV
  transforms.

The stored matrices are the single source of frame truth: the analysis
toolbox rebuilds the full transform chain from the store alone (see
{doc}`analysis/resolvers_reconcile`), and `reconcile` can re-derive every
saved shared-frame coordinate from raw ones after any re-alignment.

## 4. Spot Localization

```{image} _static/screenshots/tab_3_spot_localization.png
:alt: Spot Localization tab
:width: 100%
```

Detect spots per source — a source is always the triple
`(modality, hybe, channel)`, the only unambiguous name for a measured
signal. Each source's spots are stored in their own slice file per FOV,
carrying both `raw` (own-hybe frame) and `adj` (shared reference frame)
coordinates plus brightness, and are assigned to cells where masks say
so. A hundred slices can be written by a hundred workers with zero
contention, and deleting one slice can never touch another.

## 5. Celltype Determination

```{image} _static/screenshots/tab_4_celltype_determination.png
:alt: Celltype Determination tab
:width: 100%
```

Assign each cell a celltype from barcode readouts and/or FOV ranges,
with per-channel calibration. Celltype is a **cell attribute**, and the
analysis layer treats the unassigned `''` as a first-class group named
`Unassigned` — never silently dropped from a figure.

## 6. Chromatin Tracing

```{image} _static/screenshots/tab_5_chromatin_tracing.png
:alt: Chromatin Tracing tab
:width: 100%
```

Check the rounds to trace and fit each allele's polymer: a fiducial spot
per round anchors local drift correction, and readout spots are fitted
in a crop box around the expected position with quality gates
(brightness, lateral/axial uncertainty, at-bound rejection on readouts).
Failed fits retry with a fallback seed. Traces store `raw` and `adj`
coordinate pairs — the adj polymer is the final, shared-frame object the
analysis consumes. Repeat (R) and toe (T) rounds are traced alongside
but are never polymer bins; they exist so the Analysis tab can measure
replication error and marker efficacy.

## 7. Analysis

```{image} _static/screenshots/analysis_tab_populated.png
:alt: Analysis tab with a built population
:width: 100%
```

The last tab is a thin GUI over the headless toolbox documented in
{doc}`analysis/index`. The flow inside the tab, top to bottom:

1. **Population** — pick FOVs and sources (or one-push defaults:
   everything with spots, or a whole modality+channel excluding
   fiducials) and build. Computed per-cell attributes persist per FOV
   and rebuilds reuse them; only new sources are computed.
2. **Polymer QC** — derive thresholds from the data as quantiles,
   inspect them on the distributions they gate, edit them, apply. The
   Repeat/Toe QC button lives here too.
3. **Conditions** — compose the gate: OR-of-AND clauses over celltype,
   FOV, expression ranges (optionally normalized), pair distances,
   allele counts, barcode presence/absence, completeness. Preview any
   range on its real distribution before adding it. The gate is saved
   in the app config and restored on load.
4. **Views** — ensemble distance maps, FOV consistency, allele
   differences, expression and distance histograms, brightness-vs-count,
   each opening in its own window.
5. **Save Result…** — PNG + one CSV per table + a JSON sidecar carrying
   the exact gate, sequential survivor counts, and per-celltype counts,
   so a figure on disk answers "what was gated, and how many" by itself.

```{image} _static/screenshots/analysis_ensemble_window.png
:alt: An ensemble-map result window
:width: 85%
```

A taste of what the views produce on MP58:

```{image} _static/figures/gui_ensemble_map.png
:alt: Celltype-decomposed ensemble distance maps
:width: 100%
```

```{image} _static/figures/gui_repeat_toe.png
:alt: Repeat/toe QC
:width: 100%
```
