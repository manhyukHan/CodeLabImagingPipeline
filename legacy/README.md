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

  Correction (2026-08-08): `find_best_alignment`/`msd_cost_function` were *not* dead -- they were
  moved here in the original 2026-08-07 pass on a same-file-reference-only grep that missed
  `compute_msd_homography_matrix` (still live in `preprocess.py`) calling `find_best_alignment`
  internally. Caught when `align_same_modality` broke on real data (`NameError:
  find_best_alignment`). Moved back to `codelab_pipeline/io/preprocess.py`.
