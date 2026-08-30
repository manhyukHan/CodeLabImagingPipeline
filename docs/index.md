# CodeLabImagingPipeline

A desktop pipeline for multi-modal (RNA + DNA) ORCA / chromatin-tracing
experiments: raw DAX/TIFF ingestion, FOV / cross-modal / cell-level
alignment, spot localization, chromatin tracing, and a headless analysis
toolbox that runs with or without the app.

The pipeline is built for the real case: huge experiments on network
storage, driven from Windows, with interruptions, re-runs, and appends as
normal events rather than exceptions. Every store write goes through an
atomic door; every derived quantity can be re-derived; and the analysis
layer is a plain-Python package the GUI *uses* but never owns.

```{image} _static/screenshots/analysis_tab_populated.png
:alt: The Analysis tab on a real experiment
:width: 100%
```

## Where to start

- New machine? {doc}`installation` — clone, bootstrap the environment,
  launch.
- New to the pipeline? {doc}`happy_path` — the standard route from raw
  images to gated figures, one tab at a time, with screenshots from a
  real experiment (MP58).
- Want to script analyses without the GUI? {doc}`analysis/index` — the
  headless toolbox, documented module by module.
- Want to understand *why* things are shaped this way? {doc}`principles`.

```{toctree}
:maxdepth: 2
:hidden:

installation
happy_path
analysis/index
principles
```
