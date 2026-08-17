# Legacy code

Everything in this folder is dead code, kept only for reference: ipywidgets/Jupyter-notebook-era
UI classes and CLI/batch-script entry points from before this project had a PyQt5 GUI. None of
it is imported by `main.py` / `windows/main_window.py`, or by anything else outside this folder.
Each real, currently-used equivalent lives in the module named in the file's header comment.

Moved out of the active tree on 2026-08-07 as part of a codebase cleanup pass (see
`codelab_pipeline/io/vlinks_store.py`'s own comment, which already called these out as
"dead code, never instantiated by this GUI" before the move).

- `segment_widgets.py` -- `SegmentWidget`/`CellbarcodeWidget`, from `codelab_pipeline/segmentation/segment.py`.
  Superseded by `segment_fov`/`segment_fov_classical`, which the GUI's `ui/cell_segment_panel.py` drives.
- `chain_widget.py` -- `AlignmentWidget`, from `codelab_pipeline/alignment/chain.py`.
  Superseded by `compute_cell_alignment`/`align_same_modality`, which `ui/alignment_panel.py` drives.
- `localization_widget.py` -- `LocalizationWidget`, from `codelab_pipeline/localization/localization.py`.
  Superseded by `localize_cells_2d`/`localize_cells_3d`, which `ui/spot_localization_panel.py` drives.
- `preprocess_legacy.py` -- from `codelab_pipeline/io/preprocess.py`:
  - The whole TIFF-ingestion path (`convert_tiff_to_h5_worker`/`_main`) -- this pipeline only
    ever ingests DAX (see `convert_dax_to_h5_worker`, still live).
  - `convert_dax_to_h5_main`/`align_mips_main` -- batch-loop wrappers superseded by the GUI's
    own per-item `QThread` worker loops in `windows/main_window.py`.
  - `dax_vlinks_h5`/`vlinks_h5` -- an aggregate single-vlinks.h5 builder never called by the GUI
    (the real pipeline reads each hybe's own `{hybe}_stack.h5` directly at ingestion time only,
    and cell/spot/param/MIP/matrix persistence into vlinks.h5 goes through
    `codelab_pipeline/io/vlinks_store.py` instead -- see 2026-08-08's vlinks-MIP-storage pass).

Second cleanup pass, 2026-08-17, after the `FrameResolver` alignment refactor. A dead-code
scan (docstring/comment-stripped, so prose mentions do not count as callers) found these two
modules had no importer anywhere in the tree -- not just no callers, no `import` of the module
itself. Unlike the 2026-08-07 pass, this was checked at module level, which is what the
`find_best_alignment` correction below taught: a function-name grep is not enough.

- `fiducial_spot_mapper.py` -- from `codelab_pipeline/matlab/fiducial_spot_mapper.py` (that
  package is now gone; it held nothing else). A ChrTracer3 (MATLAB) `AllFits.csv` /
  `selectSpots.csv` import bridge. Superseded by this pipeline's own 3D localization:
  `localize_chromatin_trace_hybe` in `codelab_pipeline/localization/localization.py`, which
  `ui/chromatin_tracing_panel.py` drives, builds `AnAllele` objects directly.
  Its `from ..models.allele import AnAllele` was rewritten absolute so it still imports here.
- `cellclassifier_compat.py` -- from `codelab_pipeline/io/cellclassifier_compat.py`. Exports
  `CellContainer`/`ACell`/`ASpot` into CellClassifier's `.smeta` pickle shape. No live export
  path in the GUI calls it. Kept because its header documents CellClassifier's three pickle
  formats and which of them round-trip.

Third cleanup pass, 2026-08-17, removing obsolete metadata from the raw stack files. Per the
standing principle that `vlinks.h5` is the authoritative store for metadata/parameters and
`{hybe}_stack.h5` holds raw data only, `/matrix` and `/matrix_across` were removed from both
the code and the data. `/mip` stays: it is derived from the raw stack in the same ingestion
pass, is never recomputed, and its live reader (`_build_cell_crop`) takes it from the same
open file handle it must use for `/stack` anyway. The hazard being removed is specifically
*mutable* metadata duplicated across two stores -- matrices are refit per alignment run and
keyed by reference hybe, so they rot; a MIP is a pure function of raw data written once.

- `localize_spots_worker.py` -- from `codelab_pipeline/localization/localization.py`. Zero code
  references (AST check, docstrings stripped -- the check `find_best_alignment` below taught).
  Moved rather than deleted because localization.py's own docstrings cite it as a porting
  reference, and it needs to carry a warning: it reads `/matrix/{hybe}` out of the raw stack
  file, a store that no longer has any writer and was measured to disagree with vlinks on
  16 of 18 hybes. See that file's header. Live equivalents: `localize_cell_2d_worker`/
  `localize_cell_3d_worker`.

  Correction (2026-08-08): `find_best_alignment`/`msd_cost_function` were *not* dead -- they were
  moved here in the original 2026-08-07 pass on a same-file-reference-only grep that missed
  `compute_msd_homography_matrix` (still live in `preprocess.py`) calling `find_best_alignment`
  internally. Caught when `align_same_modality` broke on real data (`NameError:
  find_best_alignment`). Moved back to `codelab_pipeline/io/preprocess.py`.
