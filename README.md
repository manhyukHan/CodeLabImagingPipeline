# CodeLabImagingPipeline

CODE Lab imaging pipeline for multi-modal (RNA + DNA) ORCA / chromatin
tracing: raw DAX/TIFF ingestion, FOV / cross-modal / cell-level
alignment, spot localization, chromatin tracing, and a headless analysis
toolbox — built for huge experiments on network storage, with
interruption-safe stores and append-everywhere semantics.

**Full documentation: <https://codelabimagingpipeline.readthedocs.io>**
(sources under [`docs/`](docs/) — installation, the standard
ingestion→analysis walkthrough with screenshots, the design principles,
and a module-by-module reference for the analysis toolbox).

## Quick start

```console
git clone https://github.com/manhyukHan/CodeLabImagingPipeline.git
cd CodeLabImagingPipeline
```

Then double-click the launcher for your platform — `launch_codelab.bat`
(Windows), `launch_codelab.command` (macOS) — and let it create the
conda environment from `requirements.txt` on first run. Details and the
manual route: [docs/installation.md](docs/installation.md).

## Analysis without the app

`codelab_pipeline.analysis` imports no Qt and no app module — build
populations, compose gates, and render ensemble maps from any script or
notebook pointed at a store. Start at
[docs/analysis/index.md](docs/analysis/index.md).

### Flat tables

```
python tools/export_tables.py <project-root> [--fovs 1-9] [--combined]
```

Writes one row per cell, per spot and per (allele, bin, candidate),
plus `cells_arrays.npz` holding the masks and per-hybe matrices the
cell table only summarises. That is the export: three tables, every
field, as sheets of one .xlsx (`--format csv` writes a file each --
CSV has no sheets).

`--combined` adds a fourth table for analysis: one row per spot with
its cell joined on, and COMPACT. It keeps the keys, the position, and
every field the pipeline's own schema does not declare -- which is to
say the measurements somebody attached -- and drops the geometry,
frames, alignment and link state that are already in the other sheets.
A reader handed 84 columns cannot tell which carry the result.
`--combined-full` keeps them all.

**The columns are discovered, not declared.** A field a project writes
onto its cells or spots appears as a column without anything in the
exporter naming it. The alignment is exported AS STORED -- the cell's
own matrix plus `home_hybe` and `home_yx_is_residual` -- never
pre-composed: a resolved transform baked into a file goes stale the
moment FOV alignment is re-run, and the file cannot say so. Compose it
on read with `alignment.frames.FrameResolver`
(`analysis.resolvers.resolver_for(store, fov)` builds one from the
store alone).

The same export is in the app: **Export Cells / Spots / Alleles...**
on the cell/spot status panel, which writes the same three sheets
and the same `.npz` for the FOVs the Ingestion panel lists (every
FOV in the store when that box is empty). Only `--combined` is
CLI-only -- it is an analysis convenience, not the export.

### Attaching your own fields

`ACell`, `ASpot` and `AnAllele` keep any keyword they are not declared
to understand on `.properties`, `save()` writes it back, and the store
gives it its own typed column. So an analysis can measure something
per cell or per spot, attach it, and have it survive a save and come
back on the next read -- and be exported -- without a line changing in
the model, the store or the exporter.

```python
cell.set_metadata(n_speckle=23, speckle_frac=0.064)
analysis_store.write_cells(store, fov, container)   # kept
```

A property is one SCALAR per item (number, string, bool). Anything
array-shaped is refused with its name rather than dropped: the schema
was closed before this, and a field that vanishes on save is
indistinguishable from an analysis that never ran.
