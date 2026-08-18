"""
Dead code -- see legacy/README.md. Moved out of
codelab_pipeline/localization/localization.py on 2026-08-19.

Bulk (batch/CLI-era) localization drivers with ZERO GUI callers: the app
drives localize_cell_2d_worker/_3d_worker per cell interactively and feeds
results into the session's SpotContainer through MainWindow's own doors
(_replace_cell_spots, which allocates uids at entry). After cells stopped
holding spot lists, these were rewritten to RETURN their spots -- and
nothing ever consumed the return.

If a batch mode returns, port the SHAPE, not the code: run workers, collect
spots (each already carrying its owner in ASpot.cell), allocate uids at the
container door, add to a SpotContainer, persist per slice.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed

from codelab_pipeline.localization.localization import (
    localize_cell_2d_worker, localize_cell_3d_worker)


def localize_cells_2d(cell_container, fov, hybe_records, channel,
                      max_to_background=1.25, max_to_average=1.25, absolute_threshold=450.0,
                      min_distance=3, frac=0.8, max_num_alleles=2, pad=5,
                      storage_path=None, n_procs=4):
    """
    Bulk (non-interactive) 2D localization over every cell in
    cell_container.get_cells(fov), across every hybe in hybe_records.
    Parameters are an already-confirmed set -- tune interactively via
    LocalizationWidget first, then run this in bulk; not re-tuned per call.
    Returns the localized spots (each carrying its owner in ASpot.cell).
    """
    cells = cell_container.get_cells(fov)
    tasks = [(cell, record['folder']) for cell in cells for record in hybe_records]

    with ProcessPoolExecutor(max_workers=n_procs) as executor:
        futures = [executor.submit(localize_cell_2d_worker, cell, hybe, channel, storage_path, fov,
                                   max_to_background, max_to_average, absolute_threshold,
                                   min_distance, frac, max_num_alleles, pad)
                  for cell, hybe in tasks]
        out = []
        for future in as_completed(futures):
            cell_id, hybe, spots = future.result()
            # Cells hold no spot lists: each returned spot already carries
            # its owning cell in ASpot.cell, and the caller feeds them into
            # the session's SpotContainer (with uids allocated at that
            # door). This function computes; it does not own storage.
            out.extend(spots)
        return out


def localize_cells_3d(cell_container, fov, hybe_records, channel,
                      max_to_background=1.25, max_to_average=1.25, absolute_threshold=200.0,
                      min_distance=3, frac=0.8, max_num_alleles=2, max_sigma=2.5,
                      pad=5, spad=5, storage_path=None, n_procs=4,
                      peak_bound=2.0, max_uncert=2.0, min_hb_ratio=1.15, min_ah_ratio=0.15):
    """3D counterpart of localize_cells_2d -- see localize_cell_3d_worker. Defaults
    (max_sigma, peak_bound, max_uncert, min_hb_ratio, min_ah_ratio, absolute_threshold)
    are ChrTracer3's own real values (Pars Fit Spots.csv: datMinPeakHeight,
    fidMaxFitWidth-derived max_sigma, etc.), not arbitrary starting points."""
    cells = cell_container.get_cells(fov)
    tasks = [(cell, record['folder']) for cell in cells for record in hybe_records]

    with ProcessPoolExecutor(max_workers=n_procs) as executor:
        futures = [executor.submit(localize_cell_3d_worker, cell, hybe, channel, storage_path, fov,
                                   max_to_background, max_to_average, absolute_threshold,
                                   min_distance, frac, max_num_alleles, max_sigma, pad, spad,
                                   peak_bound, max_uncert, min_hb_ratio, min_ah_ratio)
                  for cell, hybe in tasks]
        out = []
        for future in as_completed(futures):
            cell_id, hybe, spots = future.result()
            # Cells hold no spot lists: each returned spot already carries
            # its owning cell in ASpot.cell, and the caller feeds them into
            # the session's SpotContainer (with uids allocated at that
            # door). This function computes; it does not own storage.
            out.extend(spots)
        return out


