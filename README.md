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
