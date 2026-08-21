import os
import re
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime

import numpy as np
import numpy.linalg as la
import h5py
import cv2
from PyQt5 import QtWidgets, QtCore

from config import path as repo_path, config_name
from ui.main_window_ui import MainWindowUI
from canvas.pipeline_canvas import PipelineCanvas
from canvas.cell_displayer import CellDisplayer
from ui.cytoplasm_panel import CytoplasmSegmentationWindow
from canvas.spot_crop_displayer import SpotCropDisplayer
from canvas.localize_3d_displayer import Localize3DDisplayer, Localize3DGridDisplayer
from canvas.barcode_overview_displayer import BarcodeOverviewDisplayer
from canvas.celltype_result_displayer import CelltypeResultDisplayer
from canvas.mip_viewer import MipViewerDisplayer
from canvas.cell_spot_status_displayer import CellSpotStatusDisplayer
from canvas.alignment_preview_window import AlignmentPreviewWindow
from canvas.chromatin_trace_grid_displayer import ChromatinTraceGridDisplayer
from codelab_pipeline.io import paths
from codelab_pipeline.io import preprocess
from codelab_pipeline.io import vlinks_store
from codelab_pipeline.alignment import chain as alignment
from codelab_pipeline.alignment import frames
from codelab_pipeline.alignment import spot_mapper
from codelab_pipeline.alignment.convention import as_cv2
from codelab_pipeline.segmentation import segment
from codelab_pipeline.localization import assignment
from codelab_pipeline.localization import localization
from codelab_pipeline.models.cell_container import CellContainer
from codelab_pipeline.models.spot import ASpot
from codelab_pipeline.models.spot_container import DiffUndo, SpotContainer
from codelab_pipeline.models.allele import AnAllele
from codelab_pipeline.models import celltype
from skimage.feature import peak_local_max


class IngestionWorker(QtCore.QThread):
    """
    Converts every (FOV, hybe) task through a spawn-context
    ProcessPoolExecutor (max_workers from the Ingestion tab's Parallel
    workers spinbox; 1 = plain sequential). The work is I/O-bound --
    each task is one huge sequential DAX read plus an uncompressed H5
    write -- so a few workers overlapping read latency is where the
    speedup lives, especially off a network share.

    Two invariants:
    - vlinks.h5 stays SINGLE-WRITER: children only ever write their own
      independent {hybe}_stack.h5; the MIP copy into the one shared
      vlinks.h5 happens HERE, in this coordinator thread, as each future
      completes (HDF5 forbids concurrent writers as a format matter,
      independent of file size).
    - spawn context requires a __main__-guarded entry (main.py is; any
      script that triggers ingestion must be). convert_dax_to_h5_worker
      is a top-level function in a Qt-free module, so children never
      import the GUI. If the pool cannot be created at all, the run
      degrades to the old sequential loop rather than failing.
    """
    progress = QtCore.pyqtSignal(int, int, str)
    # (n_errors, n_total, error_lines) -- per-task errors from
    # convert_dax_to_h5_worker never raise (each returns (fov, hybe, err)
    # and the loop always continues), so "the worker didn't raise" is NOT
    # the same as "ingestion actually succeeded" -- every task could have
    # individually failed (e.g. a bad/empty DAX directory) while this
    # signal still fires. The caller must check n_errors, not just which
    # signal fired.
    finished_ok = QtCore.pyqtSignal(int, int, list)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, fov_list, hybe_records, dax_directory, storage_path, modality, overwrite=True,
                 max_workers=4):
        super().__init__()
        self.max_workers = max(1, int(max_workers))
        self.fov_list = fov_list
        self.hybe_records = hybe_records
        self.dax_directory = dax_directory
        self.storage_path = storage_path
        self.modality = modality
        # True = "Overwrite All" (re-convert everything, even already-done
        # targets); False = "Append" (convert_dax_to_h5_worker's own
        # overwrite=False path returns early with err=None for a target
        # that already exists, so already-ingested hybes are left alone
        # and only the missing ones actually get converted).
        self.overwrite = overwrite

    def run(self):
        try:
            tasks = [(fov, record) for fov in self.fov_list for record in self.hybe_records]
            error_lines = []
            done = [0]

            def finish_task(fov_r, hybe_r, err, record):
                # Coordinator-side tail of every task, pooled or not:
                # ingestion is one of the two places this app is allowed
                # to touch the raw per-hybe stack file directly (see
                # vlinks_store.write_hybe_mip's own docstring) -- the
                # file's already just been written, so this MIP copy is
                # cheap, and doing it HERE keeps vlinks.h5 single-writer.
                if err is None and not paths.is_v2(self.storage_path):
                    # v1 only: copy the MIP into the shared vlinks.h5 from
                    # THIS coordinator thread (single-writer). In a v2
                    # store the worker already wrote the standalone
                    # per-hybe MIP file itself -- nothing to do here, and
                    # vlinks.h5 sees zero ingestion traffic.
                    try:
                        h5path = paths.stack_path(self.storage_path, fov_r, hybe_r)
                        with h5py.File(h5path, 'r') as f:
                            channel_mips = {ch: f[f'/mip/ch{ch}'][:] for ch in record['channels']}
                        vlinks_store.write_hybe_mip(self.storage_path, fov_r, hybe_r, channel_mips,
                                                    fiducial_channel=record['fiducial_channel'])
                    except Exception as e:
                        err = f'ingested but failed to write vlinks.h5 MIP: {e}'
                status = 'OK' if err is None else f'ERROR: {err}'
                if err is not None:
                    error_lines.append(f'FOV{fov_r:02d} {hybe_r}: {err}')
                done[0] += 1
                self.progress.emit(done[0], len(tasks), f'FOV{fov_r:02d} {hybe_r}: {status}')

            executor = None
            if self.max_workers > 1 and len(tasks) > 1:
                try:
                    executor = ProcessPoolExecutor(max_workers=min(self.max_workers, len(tasks)),
                                                   mp_context=multiprocessing.get_context('spawn'))
                except Exception as e:
                    self.progress.emit(0, len(tasks), f'process pool unavailable ({e}) -- running sequentially')

            if executor is not None:
                with executor:
                    task_by_future = {executor.submit(preprocess.convert_dax_to_h5_worker,
                                                      fov, record, self.dax_directory, self.storage_path,
                                                      self.modality, overwrite=self.overwrite): (fov, record)
                                      for fov, record in tasks}
                    for future in as_completed(task_by_future):
                        fov, record = task_by_future[future]
                        try:
                            fov_r, hybe_r, err = future.result()
                        except Exception as e:
                            # a task must never kill the run -- same
                            # per-task error contract as the worker's own
                            fov_r, hybe_r, err = fov, record['folder'], f'worker process failed: {e}'
                        finish_task(fov_r, hybe_r, err, record)
            else:
                for fov, record in tasks:
                    fov_r, hybe_r, err = preprocess.convert_dax_to_h5_worker(
                        fov, record, self.dax_directory, self.storage_path, self.modality, overwrite=self.overwrite)
                    finish_task(fov_r, hybe_r, err, record)
            # dax_vlinks_h5 (a single aggregate vlinks.h5 across every hybe)
            # is deliberately NOT called here -- it's only ever read by the
            # legacy Jupyter-widget classes now in legacy/segment_widgets.py,
            # legacy/chain_widget.py, legacy/localization_widget.py
            # (SegmentWidget, AlignmentWidget, LocalizationWidget,
            # CellbarcodeWidget), none of which this GUI ever instantiates.
            # The real pipeline (segment_fov, align_same_modality,
            # localize_cells_2d, ...) reads directly from each hybe's own
            # {hybe}_stack.h5, never vlinks.h5. Calling it here used to mean
            # one single missing/bad hybe (out of possibly many requested)
            # would raise and report the ENTIRE ingestion run as failed --
            # even though every other hybe's per-hybe H5 had already been
            # written successfully and was already fully usable.
            self.finished_ok.emit(len(error_lines), len(tasks), error_lines)
        except Exception as e:
            self.failed.emit(str(e))


class CellSegmentWorker(QtCore.QThread):
    finished_ok = QtCore.pyqtSignal(object, object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, storage_path, fov, reference_hybe, channel, diameter, min_size, max_size,
                 projection=('MIP (stored)', None, None)):
        super().__init__()
        self.storage_path = storage_path
        self.fov = fov
        self.reference_hybe = reference_hybe
        self.channel = channel
        self.diameter = diameter
        self.min_size = min_size
        self.max_size = max_size
        self.projection = projection

    def run(self):
        try:
            mode, z_plane, z_range = self.projection
            mask, reference_image = segment.segment_fov(
                self.storage_path, self.fov, self.reference_hybe, self.channel,
                diameter=self.diameter, min_size=self.min_size, max_size=self.max_size,
                projection_mode=mode, z_plane=z_plane, z_range=z_range)
            self.finished_ok.emit(mask, reference_image)
        except Exception as e:
            self.failed.emit(str(e))


class ClassicalSegmentWorker(QtCore.QThread):
    """Mirrors CellSegmentWorker's exact signal shape so _on_cell_segment_finished/_failed are reused unchanged."""
    finished_ok = QtCore.pyqtSignal(object, object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, storage_path, fov, reference_hybe, channel, method, absolute_cutoff, min_distance,
                 min_size, max_size, projection=('MIP (stored)', None, None)):
        super().__init__()
        self.projection = projection
        self.storage_path = storage_path
        self.fov = fov
        self.reference_hybe = reference_hybe
        self.channel = channel
        self.method = method
        self.absolute_cutoff = absolute_cutoff
        self.min_distance = min_distance
        self.min_size = min_size
        self.max_size = max_size

    def run(self):
        try:
            mode, z_plane, z_range = self.projection
            mask, reference_image = segment.segment_fov_classical(
                self.storage_path, self.fov, self.reference_hybe, self.channel,
                method=self.method, absolute_cutoff=self.absolute_cutoff, min_distance=self.min_distance,
                min_size=self.min_size, max_size=self.max_size,
                projection_mode=mode, z_plane=z_plane, z_range=z_range)
            self.finished_ok.emit(mask, reference_image)
        except Exception as e:
            self.failed.emit(str(e))


class AlignmentWorker(QtCore.QThread):
    """
    fov_list: one entry in manual mode (today's single-FOV behavior,
    unchanged), every FOV in the ingestion panel's FOV list in automatic
    mode -- "automatic" now means batch, not single-FOV-as-a-demo.
    """
    progress = QtCore.pyqtSignal(int, int, str)
    finished_ok = QtCore.pyqtSignal(dict)  # {fov: {hybe: H}}
    failed = QtCore.pyqtSignal(str)

    def __init__(self, storage_path, fov_list, hybe_records, reference_hybe, write=True, border_trim=0, max_shift=None):
        super().__init__()
        self.storage_path = storage_path
        self.fov_list = fov_list
        self.hybe_records = hybe_records
        self.reference_hybe = reference_hybe
        self.write = write
        self.border_trim = border_trim
        self.max_shift = max_shift

    def run(self):
        try:
            results = {}
            z_results = {}
            for i, fov in enumerate(self.fov_list):
                matrices = alignment.align_same_modality(self.storage_path, fov, self.hybe_records,
                                                              self.reference_hybe, write=self.write,
                                                              border_trim=self.border_trim, max_shift=self.max_shift)
                results[fov] = matrices
                self.progress.emit(i + 1, len(self.fov_list), f'FOV{fov:02d}: {len(matrices)} hybe(s) aligned')
            self.finished_ok.emit(results)
        except Exception as e:
            self.failed.emit(str(e))


class CellAlignmentWorker(QtCore.QThread):
    """
    jobs: list of (fov, cells, passes). cells are the real ACell objects
    (automatic mode -- compute_cell_alignment mutates cell.matrices in
    place, so this IS the commit) or deepcopies of them (manual mode --
    staged, only merged into the real cells on Accept). Manual mode
    passes one job (today's single-FOV behavior); automatic mode passes
    one job per FOV that has permanent segmented cells.

    passes: one dict PER MODALITY (see MainWindow._cell_alignment_passes,
    which builds them), each with 'modality', 'storage_path',
    'hybe_records', 'fov_matrices', 'reference_hybe' and
    'cellref_fov_matrices'. Every cell gets one compute_cell_alignment
    call per pass, so a cell ends up carrying a real, fitted residual for
    EVERY configured modality's hybes -- not just its own segmentation
    modality's.

    Per explicit correction, what was dropped is narrower than an earlier
    version of this code assumed: only the genuinely CROSS-MODAL FIT (a
    DNA hybe's crop phase-correlated against an RNA anchor's crop, i.e.
    comparing images across the modality boundary) is gone. Each
    modality's own hybes are still fit against THAT MODALITY'S OWN cell
    alignment reference (pass['reference_hybe'] = ap.current_cell_
    reference_hybe(that modality)) -- DNA hybes vs DNA's anchor, RNA
    hybes vs RNA's anchor -- so both crops in every fit are always the
    same modality, which is the property that made same-modality fitting
    work well in the first place. An earlier version instead skipped the
    other modality's hybes ENTIRELY, which silently left every DNA spot
    in an RNA cell with no cell-level residual at all (its coordinate
    fell back to the plain FOV/cross-modal matrix), losing a real,
    computable, purely same-modality correction: the drift between that
    DNA hybe and DNA's own cell alignment reference.

    The cell's mask still has to be projected across the modality
    boundary to define the other modality's crop windows (unavoidable --
    cell.area is native to the segmentation hybe's frame), via each
    pass's own 'cellref_fov_matrices'; only the residual FIT itself is
    kept strictly within one modality.
    """
    progress = QtCore.pyqtSignal(int, int, str)
    finished_ok = QtCore.pyqtSignal(list)  # [(fov, cells), ...]
    failed = QtCore.pyqtSignal(str)

    def __init__(self, jobs, channel_type='readout', pad=10):
        super().__init__()
        self.jobs = jobs
        self.channel_type = channel_type
        self.pad = pad

    def run(self):
        try:
            results = []
            total = sum(len(cells) * max(len(passes), 1) for _, cells, passes in self.jobs)
            done = 0
            for fov, cells, passes in self.jobs:
                for cell in cells:
                    for p in passes:
                        fov_matrices = p['fov_matrices']
                        # only the hybes actually present in this FOV's own
                        # fov_matrices are valid -- hybe_records can hold
                        # more (e.g. every hybe in the parsed layout) than
                        # what FOV alignment was actually run/accepted for.
                        hybe_records = [r for r in p['hybe_records']
                                if (r['folder'], fov_matrices.modality) in fov_matrices]
                        # Resolved PER CELL, not once per pass:
                        # cell.reference_hybe genuinely varies cell-to-cell
                        # under append-mode segmentation. For the cell's own
                        # modality this reproduces compute_cell_alignment's
                        # own default lookup exactly (same dict); for any
                        # other modality it supplies the value that
                        # function's docstring explicitly requires the
                        # caller to pass, since cell.reference_hybe is never
                        # a key in another modality's own fov_matrices.
                        # Resolved from the CELL's own reference_modality, not
                        # the pass's -- after cytoplasmic segmentation a cell's
                        # reference_hybe can belong to the other modality, and
                        # looking it up in the wrong modality's dict would miss
                        # and silently fall back to identity.
                        frame_modality = cell.reference_modality
                        cellref_matrix = p['cellref_fov_matrices'].get(
                            frame_modality, {}).get((cell.reference_hybe, frame_modality), np.eye(3))
                        alignment.compute_cell_alignment(
                            cell, p['storage_path'], fov, hybe_records, fov_matrices,
                            reference_hybe=p['reference_hybe'], channel_type=self.channel_type,
                            pad=self.pad, modality=p['modality'],
                            cell_reference_hybe_matrix=cellref_matrix)
                        done += 1
                        self.progress.emit(done, total,
                                           f"FOV{fov:02d} cell {cell.id} ({p['modality']}): aligned")
                results.append((fov, cells))
            self.finished_ok.emit(results)
        except Exception as e:
            self.failed.emit(str(e))


class CrossModalAlignmentWorker(QtCore.QThread):
    """fov_list: one entry (manual) or every FOV in the FOV list (automatic)."""
    progress = QtCore.pyqtSignal(int, int, str)
    finished_ok = QtCore.pyqtSignal(dict)  # {fov: H}
    failed = QtCore.pyqtSignal(str)

    def __init__(self, rna_storage_path, dna_storage_path, fov_list, all_fov_matrices,
                 rna_reference_hybe, dna_reference_hybe, channel_type, border_trim=0, max_shift=None,
                 rna_fiducial_channel=None, dna_fiducial_channel=None):
        super().__init__()
        self.rna_storage_path = rna_storage_path
        self.dna_storage_path = dna_storage_path
        self.fov_list = fov_list
        self.all_fov_matrices = all_fov_matrices  # {(storage_path, fov): {hybe: H}}
        self.rna_reference_hybe = rna_reference_hybe
        self.dna_reference_hybe = dna_reference_hybe
        self.channel_type = channel_type
        self.border_trim = border_trim
        self.max_shift = max_shift
        # Z drift is measured in the SAME run as dx/dy -- it is a component
        # of the cross-modal result, not a separate parameter (per explicit
        # correction: a drift is a measurement like X and Y, so it belongs
        # in Results and in Accept, never in a spinbox).
        self.rna_fiducial_channel = rna_fiducial_channel
        self.dna_fiducial_channel = dna_fiducial_channel

    def run(self):
        try:
            results = {}
            z_results = {}
            for i, fov in enumerate(self.fov_list):
                rna_fov_matrices = self.all_fov_matrices.get(fov, alignment.FrameMatrices()).for_modality(vlinks_store.modality_of(self.rna_storage_path))
                dna_fov_matrices = self.all_fov_matrices.get(fov, alignment.FrameMatrices()).for_modality(vlinks_store.modality_of(self.dna_storage_path))
                H = alignment.link_cross_modal(self.rna_storage_path, self.dna_storage_path, fov,
                                               rna_fov_matrices, dna_fov_matrices,
                                               self.rna_reference_hybe, self.dna_reference_hybe, self.channel_type,
                                               border_trim=self.border_trim, max_shift=self.max_shift)
                results[fov] = H
                dz = 0.0
                if self.rna_fiducial_channel is not None and self.dna_fiducial_channel is not None:
                    try:
                        dz, _q, _diag = alignment.estimate_cross_modal_z(
                            self.rna_storage_path, self.dna_storage_path, fov,
                            self.rna_reference_hybe, self.dna_reference_hybe,
                            self.rna_fiducial_channel, self.dna_fiducial_channel)
                    except (OSError, KeyError):
                        dz = 0.0   # raw stack unavailable -- no z layer, not a failed run
                z_results[fov] = float(dz)
                self.progress.emit(i + 1, len(self.fov_list),
                                   f'FOV{fov:02d}: cross-modal computed (dz={dz:+.1f})')
            self.finished_ok.emit({'H': results, 'z': z_results})
        except Exception as e:
            self.failed.emit(str(e))


class ChromatinTracingWorker(QtCore.QThread):
    """
    jobs: [(storage_path, fov, [AnAllele, ...]), ...] -- every FOV that
    already has alleles built (see MainWindow._build_chromatin_alleles_
    from_selection; this worker never builds alleles itself, only fits
    whatever's already there -- same "preview/build first, batch commits
    second" split every other Run-All-style action in this app follows).
    fov_matrices_by_fov: {fov: {hybe: H}}, precomputed on the main thread
    before this worker starts (MainWindow._composed_fov_matrices_for_cell_
    alignment is a plain dict read, but keeping session-state access on
    the main thread is the safer convention already used elsewhere).
    cell_lookup(fov, cell_id) -> ACell-or-None resolves each allele's
    owning cell, if any (MainWindow._find_cell_by_id).
    resolver_by_fov: {fov: FrameResolver}, likewise precomputed on the
    main thread -- MainWindow._frame_resolver READS QT WIDGETS (the
    storage-path line edits), so it must never be called from here.
    Cell-independent by design: the z path uses only bridge_z_between
    plus the cell's own zx, never the resolver's per-cell anchors.
    """
    progress = QtCore.pyqtSignal(int, int, str)
    finished_ok = QtCore.pyqtSignal(dict)  # {(storage_path, fov): [AnAllele, ...]}
    failed = QtCore.pyqtSignal(str)

    def __init__(self, jobs, hybes, reference_hybe, hybe_fiducial_channels, hybe_readout_channels, modality,
                fov_matrices_by_fov, cell_lookup, max_fiducial_drift, spad, z_window,
                fiducial_params, readout_params, resolver_by_fov=None,
                max_fiducial_drift_z=10.0):
        super().__init__()
        self.resolver_by_fov = resolver_by_fov or {}
        self.jobs = jobs
        self.hybes = hybes
        self.reference_hybe = reference_hybe
        self.hybe_fiducial_channels = hybe_fiducial_channels
        self.hybe_readout_channels = hybe_readout_channels
        self.modality = modality
        self.fov_matrices_by_fov = fov_matrices_by_fov
        self.cell_lookup = cell_lookup
        self.max_fiducial_drift = max_fiducial_drift
        self.max_fiducial_drift_z = max_fiducial_drift_z
        self.spad = spad
        self.z_window = z_window
        self.fiducial_params = fiducial_params
        self.readout_params = readout_params

    def run(self):
        try:
            results = {}
            total = sum(len(alleles) for _, _, alleles in self.jobs)
            done = 0
            for storage_path, fov, alleles in self.jobs:
                fov_matrices = self.fov_matrices_by_fov.get(fov, {})
                for allele in alleles:
                    cell = self.cell_lookup(fov, allele.cell) if allele.cell != -1 else None
                    localization.build_chromatin_trace_allele(
                        allele, self.hybes, self.reference_hybe, self.hybe_fiducial_channels,
                        self.hybe_readout_channels, storage_path, fov, self.modality, cell, fov_matrices,
                        max_fiducial_drift=self.max_fiducial_drift,
                        max_fiducial_drift_z=self.max_fiducial_drift_z,
                        spad=self.spad, z_window=self.z_window,
                        fiducial_params=self.fiducial_params, readout_params=self.readout_params,
                        resolver=self.resolver_by_fov.get(fov))
                    done += 1
                    self.progress.emit(done, total, f'FOV{fov:02d} allele {allele.id}: '
                                       f'{len(allele.polymer)}/{len(self.hybes)} hybe(s) traced')
                results[(storage_path, fov)] = alleles
            self.finished_ok.emit(results)
        except Exception as e:
            self.failed.emit(str(e))


def _matrix_dxdy_angle(H):
    """'dx=, dy=, angle= deg' with no hybe prefix -- same numbers _matrix_
    summary reports, for a caller (Cell/Spot Status Detail's matrix panel)
    that already shows the hybe name in its own tree column."""
    # y-major H (convention.py): ty = H[0,2], tx = H[1,2]. Labels keep
    # meaning dx=x / dy=y for the user; only the reads move.
    angle = alignment.h_angle_degrees(H)
    return f'dx={H[1,2]:.2f}, dy={H[0,2]:.2f}, angle={angle:.3f} deg'


def _matrix_summary(hybe, H):
    return f'{hybe}: {_matrix_dxdy_angle(H)}'


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config_file=None):
        super().__init__()
        self.ui = MainWindowUI()
        self.ui.setupUi(self)

        self.hybe_records = []
        self.fov_matrices = {}   # {fov: FrameMatrices{(hybe, modality): H}}
        # (storage_path, fov) -> {(hybe, channel): [ASpot, ...]} -- Whole
        # The transient spot tier: ONE container per session holding every
        # in-memory spot, keyed {fov: {uid: ASpot}}. Replaces the old
        # fov_unassigned_spots pool outright (no parallel state -- per
        # explicit decision, keeping the old structure in sync would hand
        # back false green signs). During the transition to the full
        # two-tier design, ASSIGNED spots still live on cell.spots; only
        # the unassigned view of this container is authoritative, via
        # spot_container.unassigned(fov).
        self.spot_container = SpotContainer()
        # The permanent tier: an in-memory mirror of vlinks' spot store,
        # loaded per FOV alongside the transient tier and updated by every
        # persist. Revert reads THIS, not disk -- one source of "what was
        # saved" that cannot drift from what persist just wrote.
        self.spot_container_permanent = SpotContainer()
        self._spot_loaded_fovs = set()   # {fov}: disk spots staged once per session
        # Shared celltype identity list (see ui/celltype_determination_
        # panel.py's own docstring) -- default empty, seeded from a loaded
        # config's celltype_names and/or any real classified celltype
        # already found in vlinks.h5 (see _refresh_celltype_names_from_
        # vlinks), so the Celltype Determination tab's own listview is
        # already usable without the user re-typing every name back in.
        # Names only ever get ADDED here automatically, never removed --
        # Remove Selected in the panel stays a manual, explicit action.
        self.current_celltype_list = []
        # modality_names / modality_data / current_modality live on the
        # IngestionPanel (see its own comment) -- MainWindow deliberately
        # has no such attributes, so any code still expecting a
        # system-level modality fails loudly here instead of silently
        # reading a stale mode.
        self.ui.IngestionPanel.modality_data = {
            name: self._blank_modality_state()
            for name in self.ui.IngestionPanel.modality_names}
        self.total_active_hybe_list = []  # [(hybe_record, modality_name), ...] -- see _refresh_active_hybe_lists
        self._job_queue = []
        self._job_queue_index = 0
        self._job_queue_overwrite = True
        self.save_path = repo_path

        self.cell_container = None
        self.cell_container_permanent = None
        self._last_segment_context = None  # {'fov': .., 'reference_hybe': ..}
        self.cell_displayer = CellDisplayer()
        self.cell_displayer.mask_edited.connect(self._on_displayer_mask_edited)
        self.cell_displayer.ids_removed.connect(self._on_displayer_ids_removed)
        self.cell_displayer.undo_requested.connect(self._undo_cell_action)
        self.cell_displayer.redo_requested.connect(self._redo_cell_action)

        # ONE displayer for both nucleus segmentation and cytoplasm review
        # -- per explicit request, now that primary segmentation can itself
        # run on a single/range projection there is nothing left that
        # distinguished the two views. self._cell_displayer_mode routes
        # mask edits to the right handler ('segmentation' -> the transient
        # cell container, 'cytoplasm' -> the staged cytoplasm result), so a
        # remove-by-ID in either context still lands where it belongs.
        self.cytoplasm_window = CytoplasmSegmentationWindow()
        self._cell_displayer_mode = 'segmentation'
        self._cytoplasm_result = None  # {'labels','fov','hybe','modality','channel','image',...}

        self.spot_crop_displayer = SpotCropDisplayer()
        self.spot_crop_displayer.spots_edited.connect(self._on_spot_crop_edited)
        self.spot_crop_displayer.readonly_point_removed.connect(self._on_readonly_spot_removed)
        self._spot_crop_context = None  # {'cell': ACell, 'hybe': str, 'channel': int, 'rxmin': int, 'rymin': int}
        # Diff-based undo over the spot container: two streaks deep,
        # storing invertible {added, removed, changed} fingerprint deltas.
        self.spot_undo = DiffUndo(self.spot_container)
        # Cell undo: same DiffUndo, duck-typed onto CellContainer's own
        # fingerprint/apply_inverse/apply_forward (canonical-bytes entries).
        # Rebound on use because self.cell_container is created lazily and
        # occasionally replaced wholesale.
        self.cell_undo = DiffUndo(None)
        self._current_view_spot_refs = []  # [(ASpot, ACell-or-None), ...] -- see _refresh_localize_3d_spot_choices

        self.localize_3d_displayer = Localize3DDisplayer()
        self.localize_3d_displayer.run_requested.connect(self._run_3d_localize)
        self.localize_3d_displayer.view_requested.connect(self._view_3d_localize)
        self.localize_3d_displayer.show_crop_requested.connect(self._show_3d_crop_only)
        # Detached fit-status grid, per explicit request -- a separate
        # pop-up MainWindow shows/raises whenever View or Show Crop
        # populates it, rather than an embedded 3rd column on
        # localize_3d_displayer itself. Run never touches it (see
        # Localize3DDisplayer's own docstring on why).
        self.localize_3d_grid_displayer = Localize3DGridDisplayer()

        self.mip_viewer = MipViewerDisplayer()

        self.cell_spot_status_displayer = CellSpotStatusDisplayer()
        cssd = self.cell_spot_status_displayer
        cssd.refresh_requested.connect(self._refresh_cell_spot_status_full)
        cssd.cell_fov_changed.connect(self._refresh_cell_spot_status_cell_panel)
        cssd.spot_scope_changed.connect(self._on_cell_spot_status_spot_scope_changed)
        cssd.allele_fov_changed.connect(self._refresh_cell_spot_status_allele_panel)

        self.barcode_overview_displayer = BarcodeOverviewDisplayer()
        self.celltype_result_displayer = CelltypeResultDisplayer()
        self._fov_ranges_by_celltype = {}       # {celltype(str): range_string}
        self._barcode_channel_by_celltype = {}  # {celltype(str): (hybe,channel,modality)}
        self._barcode_calibration = {'scale': {}, 'lower_bound': {}, 'upper_bound': {}}  # each {(hybe,channel,modality): {fov(int): float}}

        # single shared pop-up + canvas for every alignment preview (FOV,
        # cross-modal, cell) -- nothing is embedded in the docked panel, see
        # ui/alignment_panel.py's docstring for why.
        self.alignment_preview_window = AlignmentPreviewWindow()
        self.preview_canvas = PipelineCanvas(self.alignment_preview_window.canvas)
        self.cross_modal_result = {}  # {(dna_storage_path, fov): H}, committed
        self.cross_modal_z = {}       # {(dna_storage_path, fov): planes}, DNA frame -> RNA frame
        self._same_modality_context = None
        self._pending_same_modality_alignment = None  # {fov: {hybe: H}} awaiting Accept/Reject
        self._cross_modal_context = None
        self._pending_cross_modal = None    # {fov: H} awaiting Accept/Reject
        self._pending_cross_modal_z = {}    # {fov: planes} staged alongside
        # Align All Cells in FOV always computes AND saves immediately (no
        # staging) -- only the per-cell tuning tool below stages a result.
        self._pending_per_cell_alignment = None  # (real_cell, staged_cell) awaiting Accept/Reject, or None
        self._pending_per_cell_alignment_fov = None
        self._pending_per_cell_alignment_params = None
        self._cell_alignment_display_cells = []  # [(fov, cell), ...] -- every cell in the Overlay FOV (tier 3); Save All Cell Overlays batches over this
        # "Results (per cell, per hybe)" is scoped to ONE cell (tier 1's
        # own FOV/Cell ID spinboxes), separate from the above -- narrowing
        # it down must never also narrow what Save All Cell Overlays sees.
        self._cell_per_hybe_context = None  # {'fov':, 'cell':, 'storage_path':, 'hybe_records':}, or None
        self._vlinks_refreshed_paths = set()  # storage_paths already reconciled from vlinks this session (see _refresh_params_from_vlinks)
        self._activated_fovs = set()  # FOVs _try_show_existing_cells has already staged into self.cell_container this session (see _activate_fov)

        # (storage_path, fov) -> [AnAllele, ...] -- chromatin tracing's own
        # session-transient allele list, built from whatever's currently
        # selected in Spot Localization (see _build_chromatin_alleles_
        # from_selection), same shape/rationale as the old unassigned pool
        # above. Only Fit All FOVs persists these (mirror_write_fov_
        # alleles) -- building/previewing stays in-memory only, same
        # "explicit Save step" convention Spot Localization's own Save
        # Current Spots already follows.
        self.chromatin_alleles = {}
        self.chromatin_fiducial_grid_displayer = ChromatinTraceGridDisplayer('Fiducial')
        self.chromatin_readout_grid_displayer = ChromatinTraceGridDisplayer('Readout')
        self.chromatin_fiducial_overlay_displayer = ChromatinTraceGridDisplayer('Fiducial Overlay')

        self._connect_signals()
        self._switch_current_modality(self.ui.IngestionPanel.ModalityComboBox.currentText())

        if config_file is not None:
            self._load_config(config_file)

    def _connect_signals(self):
        ip = self.ui.IngestionPanel
        ap = self.ui.AlignmentPanel
        cp = self.ui.CellSegmentPanel
        sp = self.ui.SpotLocalizationPanel
        ctp = self.ui.CelltypeDeterminationPanel

        ip.SetNumModalitiesPushButton.clicked.connect(self._on_set_num_modalities)
        ip.ActivateModalitiesPushButton.clicked.connect(self._on_activate_modalities)
        ip.ModalityComboBox.currentTextChanged.connect(self._switch_current_modality)

        ip.ParseLayoutPushButton.clicked.connect(self._parse_layout)
        ip.DaxDirectoryLineEdit.editingFinished.connect(
            lambda: ip.refresh_hybe_checks(ip.DaxDirectoryLineEdit.text().strip()) if self.hybe_records else None)
        ip.FovListLineEdit.editingFinished.connect(self._refresh_fov_spinbox_bounds)
        ip.RunIngestionPushButton.clicked.connect(self._run_ingestion)
        ip.ShowMipViewerPushButton.clicked.connect(self._show_mip_viewer)
        # keep an already-open viewer live as the user flips through FOV/hybe/channel
        ip.ViewerFovSpinBox.valueChanged.connect(lambda: self._show_mip_viewer(silent=True) if self.mip_viewer.isVisible() else None)
        ip.ViewerHybeComboBox.currentIndexChanged.connect(lambda: self._show_mip_viewer(silent=True) if self.mip_viewer.isVisible() else None)
        ip.ViewerChannelComboBox.currentIndexChanged.connect(lambda: self._show_mip_viewer(silent=True) if self.mip_viewer.isVisible() else None)
        ip.CheckIngestionStatusPushButton.clicked.connect(lambda: self._check_ingestion_status(silent=False))
        ip.ShowCellSpotStatusDisplayerPushButton.clicked.connect(self._show_cell_spot_status_displayer)
        ip.AddJobPushButton.clicked.connect(self._add_job_to_queue)
        ip.RemoveJobPushButton.clicked.connect(self._remove_selected_jobs)
        ip.RunQueuePushButton.clicked.connect(self._run_job_queue)
        ip.JobQueueListWidget.itemClicked.connect(self._load_job_into_form)

        cp.RunSegmentationPushButton.clicked.connect(self._run_cell_segmentation)
        cp.ShowDisplayerPushButton.toggled.connect(self._toggle_cell_displayer)
        cp.ShowCytoplasmPushButton.clicked.connect(self._show_cytoplasm_window)
        cp.AutoFocusPushButton.clicked.connect(self._autodetect_segmentation_focus)
        for _w in (cp.ZStartSpinBox, cp.ZEndSpinBox):
            _w.valueChanged.connect(lambda _=None: cp.refresh_run_label())
        cp.ZPlaneSpinBox.valueChanged.connect(lambda _=None: self._on_segmentation_plane_changed())
        cp.ProjectionModeComboBox.currentIndexChanged.connect(
            lambda _=None: (cp.refresh_run_label(), self._refresh_cell_displayer_view()))
        cp.ViewRangePushButton.clicked.connect(self._refresh_cell_displayer_view)
        cp.refresh_run_label()
        cp.FovSpinBox.valueChanged.connect(lambda _: self._refresh_segmentation_depth())
        cp.ReferenceHybeComboBox.currentIndexChanged.connect(lambda _: self._refresh_segmentation_depth())
        cp.ChannelComboBox.currentIndexChanged.connect(lambda _: self._refresh_segmentation_depth())
        # FOV/reference-hybe/channel now drive the Cell Displayer's own view
        # directly. These used to require the displayer's 'Update' button
        # (since removed): that button is a remnant of when the panel also
        # carried a modality selector and a change here could not be acted
        # on unambiguously. It can now, so a stale view after changing any
        # of the three is just a bug.
        cp.FovSpinBox.valueChanged.connect(lambda _: self._refresh_cell_displayer_view())
        cp.ReferenceHybeComboBox.currentIndexChanged.connect(
            lambda _: self._refresh_cell_displayer_view())
        cp.ChannelComboBox.currentIndexChanged.connect(
            lambda _: self._refresh_cell_displayer_view())
        cw = self.cytoplasm_window
        cw.RefreshCellsPushButton.clicked.connect(self._refresh_cytoplasm_cell_list)
        cw.HybeComboBox.currentIndexChanged.connect(self._on_cytoplasm_hybe_changed)
        cw.FovSpinBox.valueChanged.connect(lambda _: self._refresh_cytoplasm_cell_list())
        cw.ChannelComboBox.currentIndexChanged.connect(lambda _: self._refresh_cytoplasm_depth())
        cw.AutoFocusPushButton.clicked.connect(self._autodetect_cytoplasm_focus)
        for _w in (cw.SeedModeComboBox,):
            _w.currentIndexChanged.connect(lambda _=None: cw.refresh_run_label())
        for _w in (cw.ZStartSpinBox, cw.ZEndSpinBox):
            _w.valueChanged.connect(lambda _=None: cw.refresh_run_label())
        cw.ZPlaneSpinBox.valueChanged.connect(lambda _=None: self._on_cytoplasm_plane_changed())
        cw.ProjectionModeComboBox.currentIndexChanged.connect(
            lambda _=None: (cw.refresh_run_label(), self._refresh_cytoplasm_view()))
        cw.ViewRangePushButton.clicked.connect(self._refresh_cytoplasm_view)
        cw.refresh_run_label()
        cw.PreviewNucleiPushButton.clicked.connect(self._preview_cytoplasm_nuclei)
        cw.RunPushButton.clicked.connect(self._run_cytoplasm_segmentation)
        cw.IncorporatePushButton.clicked.connect(self._incorporate_cytoplasm)
        cp.SaveCellsPushButton.clicked.connect(self._save_cells)
        cp.DiscardCellsPushButton.clicked.connect(self._discard_cells)
        cp.SendPermanentPushButton.clicked.connect(self._send_permanent_cells_to_transient)
        cp.FovSpinBox.valueChanged.connect(self._activate_fov)

        ap.RunFovAlignmentPushButton.clicked.connect(self._run_fov_alignment)
        ap.RunAllFovAlignmentPushButton.clicked.connect(self._run_fov_alignment_all)
        # Results list refreshes live as either changes -- since
        # _refresh_same_modality_results_list is itself scoped to
        # whichever reference hybe (and FOV list) is currently picked.
        ap.SameModalityFovSpinBox.valueChanged.connect(lambda _: self._refresh_same_modality_results_list())
        ap.ReferenceHybeComboBox.currentIndexChanged.connect(lambda _: self._refresh_same_modality_results_list())
        ap.SameModalityResultsListWidget.itemClicked.connect(self._show_same_modality_preview)
        ap.SameModalityShowOverlayPushButton.clicked.connect(self._show_same_modality_all_readouts_overlay)
        ap.SameModalityAcceptPushButton.clicked.connect(self._accept_same_modality_alignment)
        ap.SameModalityRejectPushButton.clicked.connect(self._reject_same_modality_alignment)
        ap.RunCrossModalPushButton.clicked.connect(self._run_cross_modal_alignment)
        ap.RunAllCrossModalPushButton.clicked.connect(self._run_cross_modal_alignment_all)
        ap.CrossModalResultsListWidget.itemClicked.connect(self._show_cross_modal_result_preview)
        ap.CrossModalShowOverlayPushButton.clicked.connect(self._show_cross_modal_overlay)
        ap.CrossModalAcceptPushButton.clicked.connect(self._accept_cross_modal)
        ap.CrossModalRejectPushButton.clicked.connect(self._reject_cross_modal)
        ap.RunCellAlignmentPushButton.clicked.connect(self._run_cell_alignment)
        # "Results (per cell, per hybe)" is scoped to the single cell/FOV/
        # reference hybe picked here (tier 1) -- refreshes live as any of
        # the three changes, per explicit request, independent of tier
        # 3's own Overlay FOV / Preview reference hybe pairing below.
        ap.CellFovSpinBox.valueChanged.connect(lambda _: self._refresh_cell_per_hybe_results_from_spinboxes())
        ap.CellIdSpinBox.valueChanged.connect(lambda _: self._refresh_cell_per_hybe_results_from_spinboxes())
        ap.CellResultsListWidget.itemClicked.connect(self._show_cell_alignment_preview)
        ap.PreviewThisCellPushButton.clicked.connect(self._run_cell_alignment_for_selected_cell)
        ap.CellPadSpinBox.valueChanged.connect(lambda _: self._show_cell_alignment_preview_for_hybe())
        # Tier 3 (visualization-only): selecting a FOV here only READS
        # whatever's already saved/staged for it -- never computes or writes.
        ap.CellOverlayFovSpinBox.valueChanged.connect(lambda _: self._refresh_cell_fov_panels_from_combo())
        ap.CellOverlayCellListWidget.itemClicked.connect(self._show_cell_all_readouts_overlay)
        ap.SaveAllCellOverlaysPushButton.clicked.connect(self._save_all_cell_overlays)
        ap.PerCellAcceptPushButton.clicked.connect(self._accept_per_cell_alignment)
        ap.PerCellRejectPushButton.clicked.connect(self._reject_per_cell_alignment)

        sp.RefreshCellListPushButton.clicked.connect(self._refresh_spot_cell_list)
        sp.FovSpinBox.valueChanged.connect(lambda _: self._refresh_spot_cell_list())
        sp.CellListWidget.itemClicked.connect(self._on_spot_cell_selected)
        sp.FovListWidget.itemClicked.connect(self._on_fov_list_item_clicked)
        sp.HybeComboBox.currentIndexChanged.connect(self._show_spot_displayer)
        sp.ChannelComboBox.currentIndexChanged.connect(self._show_spot_displayer)
        sp.AutoDetectPushButton.clicked.connect(self._run_spot_auto_detect)
        sp.ShowDisplayerPushButton.toggled.connect(self._toggle_spot_crop_displayer)
        sp.Show3DLocalizationPushButton.toggled.connect(self._toggle_localize_3d_displayer)
        sp.RemoveTransientSpotsPushButton.clicked.connect(self._remove_transient_spots)
        sp.ClearHybeChannelPushButton.clicked.connect(self._clear_current_hybe_channel)
        sp.UndoPushButton.clicked.connect(self._undo_spot_action)
        sp.RedoPushButton.clicked.connect(self._redo_spot_action)
        sp.SaveCurrentSpotsPushButton.clicked.connect(self._save_current_spots)
        sp.SaveAllFovSpotsPushButton.clicked.connect(self._save_all_fov_spots)
        sp.ThresholdPercentLineEdit.editingFinished.connect(self._sync_threshold_from_percent)
        sp.ThresholdAbsoluteLineEdit.editingFinished.connect(self._sync_threshold_from_absolute)

        ctp.SetFovRangesPushButton.clicked.connect(self._set_celltype_fov_ranges)
        ctp.AssignBarcodeChannelPushButton.clicked.connect(self._assign_barcode_channel)
        ctp.ApplyCalibrationPushButton.clicked.connect(self._apply_barcode_calibration)
        ctp.ShowBarcodeOverviewPushButton.clicked.connect(self._show_barcode_overview)
        ctp.RunCelltypeDeterminationPushButton.clicked.connect(self._run_celltype_determination)
        ctp.ShowCelltypeResultPushButton.clicked.connect(lambda: self._show_celltype_result())

        chp = self.ui.ChromatinTracingPanel
        # Check/Uncheck Selected are already self-wired inside
        # ChromatinTracingPanelUI.setupUi itself (same pattern ingestion_
        # panel.py's own HybeListWidget buttons already use) -- nothing to
        # connect here for those two.
        chp.HybeListWidget.itemChanged.connect(lambda _: self._refresh_chromatin_allele_hybe_choices())
        chp.AlleleFovSpinBox.valueChanged.connect(lambda _: self._on_chromatin_allele_fov_changed())
        chp.AlleleHybeComboBox.currentIndexChanged.connect(lambda _: self._on_chromatin_allele_hybe_changed())
        chp.AlleleChannelComboBox.currentIndexChanged.connect(lambda _: self._refresh_chromatin_allele_spot_choices())
        chp.BuildAllelesPushButton.clicked.connect(self._build_chromatin_alleles_from_selection)
        chp.ViewCropPushButton.clicked.connect(self._view_chromatin_trace_crop)
        chp.FitAllFovsPushButton.clicked.connect(self._run_chromatin_tracing_fit_all)

        self.ui.actionLoad_Config.triggered.connect(self._load_config_dialog)
        self.ui.actionSave_Config.triggered.connect(self._save_config_dialog)

    # -- modality setup / switching --

    @staticmethod
    def _blank_modality_state():
        return {'layout_path': '', 'dax_directory': '', 'storage_path': '', 'active_hybe_list': []}

    def _save_current_modality_fields(self):
        ip = self.ui.IngestionPanel
        # update() in place, not a wholesale dict replacement -- this
        # modality's own active_hybe_list (and anything else computed
        # separately, e.g. by _refresh_active_hybe_lists) must survive a
        # switch-away/switch-back, not get silently wiped back to the
        # blank-state default just because the user changed tabs.
        data = self.ui.IngestionPanel.modality_data.setdefault(self.ui.IngestionPanel.current_modality, self._blank_modality_state())
        data.update({
            'layout_path': ip.LayoutPathLineEdit.text().strip(),
            'dax_directory': ip.DaxDirectoryLineEdit.text().strip(),
            'storage_path': ip.StoragePathLineEdit.text().strip(),
        })
        # every stash is also the freshest storage_path -> modality fact
        # (see vlinks_store.declare_modality: fresh-store bootstrap)
        vlinks_store.declare_modality(data['storage_path'], self.ui.IngestionPanel.current_modality)

    def _switch_current_modality(self, name):
        """
        The single "current modality" -- Ingestion tab's own selector,
        and NOTHING else. Affects exactly four things: LayoutPathLineEdit,
        DaxDirectoryLineEdit, StoragePathLineEdit, and the Hybes-to-Ingest
        checkbox list (HybeListWidget, rebuilt via _parse_layout below).
        fov_list and every other global config field are untouched, same
        as before.

        Every OTHER panel (Cell Segmentation, Same-Modality Alignment,
        Spot Localization, Celltype Determination) used to mirror this
        same selector via its own ModalityComboBox, resetting whatever
        hybe/reference-hybe combo it owned on every switch -- that's what
        caused a real bug (in-progress Spot Localization picks silently
        disappearing from view on a modality switch, nothing was actually
        deleted). Those panels now offer choices from
        self.total_active_hybe_list (every modality's hybes at once, each
        tagged with its own modality) instead, refreshed by Parse Layout/
        ingestion completing -- see _refresh_active_hybe_lists -- not by
        this switch, so they're untouched here.
        """
        if not name:
            return
        ip = self.ui.IngestionPanel
        if self.ui.IngestionPanel.current_modality is not None and self.ui.IngestionPanel.current_modality != name:
            self._save_current_modality_fields()
        self.ui.IngestionPanel.current_modality = name
        data = self.ui.IngestionPanel.modality_data.setdefault(name, self._blank_modality_state())
        ip.LayoutPathLineEdit.setText(data['layout_path'])
        ip.DaxDirectoryLineEdit.setText(data['dax_directory'])
        ip.StoragePathLineEdit.setText(data['storage_path'])
        if data['layout_path']:
            self._parse_layout()
        else:
            self.hybe_records = []
            ip.HybeListWidget.clear()
            ip.IngestionStatusTextEdit.clear()
        if data['storage_path'] and data['storage_path'] not in self._vlinks_refreshed_paths:
            # vlinks-actual values (whatever was really computed and
            # accepted) always win over a stale config default -- runs
            # LAST, after layout parsing above, so every choice-dependent
            # combo it touches (cell-alignment anchor, etc.) already has
            # real items to match against. May itself trigger a first-
            # time _parse_layout() if layout_path was blank above and
            # only vlinks has it.
            #
            # Only ever done ONCE per storage_path per session (tracked via
            # _vlinks_refreshed_paths): this reconciles a stale config
            # default on first load, but must never re-fire on a later
            # switch-back within the same session, or it would clobber a
            # live in-session combo edit the user hasn't run/accepted yet
            # with whatever vlinks still has on disk from a prior run.
            self._refresh_params_from_vlinks(data['storage_path'])
            self._vlinks_refreshed_paths.add(data['storage_path'])

    def _refresh_params_from_vlinks(self, storage_path):
        """
        "Parse every current metadata from the storage path" -- reads
        storage_path's vlinks.h5 /params (see vlinks_store.read_global_params)
        and applies it to live session state, ALWAYS winning over
        whatever's currently showing (a stale config default, or nothing
        at all), since vlinks only ever holds what was actually computed
        and accepted (see write_global_params's callers -- same-modality
        alignment accept, cross-modal accept, cell-alignment run/accept).
        Toggling a combobox never writes here, only a real accepted
        calculation does, so this is genuinely "what was last run", not
        "whatever's currently selected."

        Runs late in _switch_current_modality (after layout parsing/combo
        population), so every choice-dependent combo it touches already
        has real items to match against -- may itself trigger a one-time
        _parse_layout() if layout_path was blank until just now and only
        vlinks had it (self.hybe_records/choices don't exist yet in that
        specific case, so nothing else to wait for).

        For cross-modal params specifically, the two sides are named by
        modality (cross_modal_rna_modality / cross_modal_dna_modality) and
        their storage paths come from self.ui.IngestionPanel.modality_data -- so
        _other_modality_cell_alignment_inputs's other_data lookup and
        H_across itself stay available. There is no paired-storage-path
        param any more: one unified vlinks holds both modalities.
        """
        ip, ap = self.ui.IngestionPanel, self.ui.AlignmentPanel
        params = vlinks_store.read_global_params(storage_path)
        if not params:
            return
        data = self.ui.IngestionPanel.modality_data.get(self.ui.IngestionPanel.current_modality)

        layout_path = params.get('layout_path')
        if layout_path and data is not None and not data['layout_path']:
            data['layout_path'] = layout_path
            ip.LayoutPathLineEdit.setText(layout_path)
            self._parse_layout()

        # this storage_path's own persisted reference hybe -- select it in
        # the combo (now shared across every modality, tagged by
        # itemData) via (folder, modality), not a bare findText, so it
        # resolves to the right item even if another modality happens to
        # have a same-named hybe folder.
        reference_hybe = params.get('same_modality_reference_hybe')
        if reference_hybe:
            ap.select_reference_hybe(reference_hybe, self.ui.IngestionPanel.current_modality)
        channel_type = params.get('same_modality_channel_type')
        if channel_type:
            ap.SameModalityChannelTypeComboBox.setCurrentText(channel_type)

        # cell_alignment_reference_hybe is now per-modality-suffixed --
        # per explicit decision, each modality has its own independent
        # cell-alignment reference hybe (no more cross-modal residual fit
        # that needed a single, ambiguous combo -- see CellAlignmentWorker's
        # own docstring). storage_path here is always ONE specific
        # modality's own file, so only that modality's own key/combo apply.
        this_cell_modality = self._modality_for_storage_path(storage_path)
        cell_reference_hybe = params.get(f'cell_alignment_reference_hybe_{this_cell_modality}')
        if this_cell_modality and cell_reference_hybe and not ap.current_cell_reference_hybe(this_cell_modality):
            ap.select_cell_reference_hybe(this_cell_modality, cell_reference_hybe)
        cell_channel_type = params.get('cell_alignment_channel_type')
        if cell_channel_type:
            ap.CellChannelTypeComboBox.setCurrentText(cell_channel_type)
        cell_pad = params.get('cell_alignment_pad')
        if cell_pad is not None:
            ap.CellPadSpinBox.setValue(int(cell_pad))

        rna_modality = params.get('cross_modal_rna_modality')
        dna_modality = params.get('cross_modal_dna_modality')
        if rna_modality and dna_modality:
            rna_path = self.ui.IngestionPanel.modality_data.get(rna_modality, {}).get('storage_path', '')
            dna_path = self.ui.IngestionPanel.modality_data.get(dna_modality, {}).get('storage_path', '')
            if rna_path and dna_path:
                ap.RnaStoragePathLineEdit.setText(rna_path)
                ap.DnaStoragePathLineEdit.setText(dna_path)
            rna_ref = params.get('cross_modal_rna_reference_hybe')
            dna_ref = params.get('cross_modal_dna_reference_hybe')
            cross_channel_type = params.get('cross_modal_channel_type')
            if rna_ref:
                ap.RnaReferenceHybeComboBox.setCurrentText(rna_ref)
            if dna_ref:
                ap.DnaReferenceHybeComboBox.setCurrentText(dna_ref)
            if cross_channel_type:
                ap.ChannelTypeComboBox.setCurrentText(cross_channel_type)

            # Backfill self.fov_matrices for the PAIRED (non-current)
            # modality too -- normally only populated by _activate_fov for
            # whichever modality is "current" (never the paired one, since
            # that requires an actual modality switch) -- read it directly
            # here, same read_same_modality_matrices call _activate_fov
            # itself uses, for the CellSegmentPanel's current FOV plus
            # anything in the Ingestion tab's own FOV list. This tail used
            # to also DISCOVER the paired modality from the retired
            # cross_modal_role/cross_modal_paired_storage_path params --
            # obsolete (and, left half-deleted, a NameError that blocked
            # config load) now that _activate_modalities seeds every
            # configured modality's own state before this ever runs.
            for paired_modality in {rna_modality, dna_modality} - {self._modality_for_storage_path(storage_path)}:
                paired_data = self.ui.IngestionPanel.modality_data.get(paired_modality, {})
                paired_path = paired_data.get('storage_path', '')
                paired_layout = paired_data.get('layout_path', '') or (
                    vlinks_store.read_global_params(paired_path).get('layout_path', '') if paired_path else '')
                if not paired_path or not paired_layout:
                    continue
                try:
                    paired_hybe_records = preprocess.parse_experiment_layout(paired_layout)
                except Exception:
                    continue
                fovs_to_populate = set(self._parse_fov_list(ip.FovListLineEdit.text()))
                fovs_to_populate.add(self.ui.CellSegmentPanel.FovSpinBox.value())
                for fov_to_populate in fovs_to_populate:
                    if not self._fov_matrices_for(paired_path, fov_to_populate):
                        try:
                            self._merge_fov_matrices(
                                fov_to_populate,
                                alignment.read_same_modality_matrices(
                                    paired_path, fov_to_populate,
                                    paired_hybe_records))
                        except Exception:
                            pass

        # backfill self.cross_modal_result from vlinks so
        # _other_modality_cell_alignment_inputs finds a real H_across
        # without a fresh Run Cross-Modal Alignment this session.
        dna_storage_path = ap.DnaStoragePathLineEdit.text().strip()
        if dna_storage_path:
            for fov in self._parse_fov_list(ip.FovListLineEdit.text()):
                if (dna_storage_path, fov) in self.cross_modal_result:
                    continue
                H = vlinks_store.read_cross_modal_matrix(dna_storage_path, fov)
                if H is not None:
                    self.cross_modal_result[(dna_storage_path, fov)] = H

    def _activate_modalities(self, names, modality_fields=None):
        """
        names: ordered list[str] becoming the live modality set. modality_
        fields: optional {name: {layout_path/dax_directory/storage_path}}
        to pre-seed from (e.g. a loaded config's 'modalities' dict) --
        omitted fields default blank. A config saved before this app's
        modality decoupling may still carry old reference_hybe/
        same_modality_channel_type keys here too -- the state.update
        filter below (`if k in state`) already drops anything that isn't
        one of _blank_modality_state's current keys, so those are
        silently ignored rather than erroring.
        """
        modality_fields = modality_fields or {}
        self.ui.IngestionPanel.modality_names = names
        self.ui.IngestionPanel.modality_data = {}
        for name in names:
            state = self._blank_modality_state()
            state.update({k: v for k, v in modality_fields.get(name, {}).items() if k in state})
            self.ui.IngestionPanel.modality_data[name] = state
            # fresh-store bootstrap: the store can't know its modality
            # before ingestion, but the UI does -- see declare_modality
            vlinks_store.declare_modality(state.get('storage_path'), name)
        self.ui.IngestionPanel.current_modality = None
        # the Ingestion tab's own combo is the real modality SWITCHER (see
        # _switch_current_modality's own docstring). Cell-Based Alignment
        # has no modality selector of its own at all -- which modality a
        # cell belongs to is read directly off the cell itself (cell.
        # reference_modality) wherever a (FOV, Cell ID) resolves to a cell.
        ip = self.ui.IngestionPanel
        ip.ModalityComboBox.blockSignals(True)
        ip.ModalityComboBox.clear()
        ip.ModalityComboBox.addItems(names)
        ip.ModalityComboBox.blockSignals(False)
        ap = self.ui.AlignmentPanel
        ap.build_cell_reference_hybe_fields(names)
        # Each combo is a brand-new QComboBox object every rebuild (unlike
        # CellFovSpinBox/CellIdSpinBox, which persist and are wired once
        # in __init__) -- reconnect here every time.
        for combo in ap.CellReferenceHybeComboBoxes.values():
            combo.currentIndexChanged.connect(lambda _: self._refresh_cell_per_hybe_results_from_spinboxes())
        self._switch_current_modality(names[0])

    def _active_hybe_records_for_modality(self, name):
        """
        hybe_records (real folder dicts) for modality `name`, filtered
        down to hybes actually INGESTED (real {hybe}_stack.h5 on disk
        with real /mip data, see _ingested_hybes_for_fov) for at least
        one FOV in the Ingestion tab's FOV list. A parsed ExperimentLayout
        can declare far more hybes than were ever actually converted to
        H5 (e.g. a 103-hybe DNA layout with only a handful ingested for
        this local dataset) -- this is the ONE canonical "what's actually
        usable right now" source; every hybe-choosing field must be
        populated from this (directly, or via _refresh_active_hybe_lists'
        combined total_active_hybe_list below), never from the raw parsed
        ExperimentLayout and never from the Ingestion tab's hybe-to-
        ingest checkboxes (those reflect intent-to-ingest, not ingestion-
        complete -- see _run_ingestion/_add_job_to_queue for the one
        legitimate use of checkbox state). Offering an unfiltered/wrong-
        signal list here let a never-ingested hybe be picked as an
        alignment reference/anchor, which then crashed deep inside
        align_same_modality/compute_cell_alignment trying to open a
        stack file that doesn't exist.
        """
        ip = self.ui.IngestionPanel
        fov_list = self._parse_fov_list(ip.FovListLineEdit.text())
        data = self.ui.IngestionPanel.modality_data.get(name)
        if not data or not data.get('storage_path') or not fov_list:
            return []
        if name == self.ui.IngestionPanel.current_modality and self.hybe_records:
            records = self.hybe_records
        elif data.get('layout_path'):
            try:
                records = preprocess.parse_experiment_layout(data['layout_path'])
            except Exception:
                return []
        else:
            return []
        ingested_folders = set()
        for fov in fov_list:
            ready, _, _ = self._ingested_hybes_for_fov(data['storage_path'], fov, records)
            ingested_folders.update(ready)
        return [r for r in records if r['folder'] in ingested_folders]

    def _refresh_active_hybe_lists(self):
        """
        Recomputes active_hybe_list for every configured modality (real,
        disk-ingested hybes only) and total_active_hybe_list (every
        modality's active hybes combined, each entry tagged with its own
        modality name via (record, modality_name) tuples, so a folder
        name that happens to collide across modalities can't be confused
        for the other one's), then repopulates every hybe-choosing field
        that must track them.

        Called at exactly the two moments active_hybe_list is allowed to
        change (per explicit design): whenever the current modality is
        (re)activated -- _switch_current_modality, standing in for "the
        vlink was just parsed" for that modality -- and right after
        ingestion completes (_on_ingestion_finished). Deliberately NOT
        gated by _vlinks_refreshed_paths (that gate exists specifically
        to protect a live, in-session PARAMS edit from being clobbered by
        a stale vlinks default -- active_hybe_list isn't a user choice at
        all, just a disk-truth readout, so re-running this is always
        safe and should always reflect disk's current real state).
        """
        # Flush the LIVE Ingestion line-edits into modality_data FIRST.
        # They were only ever stashed on modality SWITCH-AWAY -- which
        # never happens in single-modality mode, so a one-modality session
        # kept modality_data's storage_path empty forever and every hybe
        # combobox in the app stayed blank (confirmed real, 2026-08-20).
        # Config-save already knew to flush before reading; this is the
        # same rule applied at the other read site.
        if self.ui.IngestionPanel.current_modality is not None:
            self._save_current_modality_fields()
        for name in self.ui.IngestionPanel.modality_names:
            self.ui.IngestionPanel.modality_data.setdefault(name, self._blank_modality_state())['active_hybe_list'] = \
                self._active_hybe_records_for_modality(name)

        total = []
        seen = set()
        for name in self.ui.IngestionPanel.modality_names:
            for r in self.ui.IngestionPanel.modality_data[name]['active_hybe_list']:
                key = (r['folder'], name)
                if key not in seen:
                    seen.add(key)
                    total.append((r, name))
        self.total_active_hybe_list = total  # [(hybe_record, modality_name), ...]

        ip, ap = self.ui.IngestionPanel, self.ui.AlignmentPanel
        # every one of these now gets the FULL cross-modality union, not
        # just self.ui.IngestionPanel.current_modality's own slice -- none of them have
        # their own modality selector any more (see _switch_current_
        # modality's own docstring); each combo item carries its own
        # modality tag instead.
        ap.populate_reference_hybe_choices(self.total_active_hybe_list)
        self.ui.CellSegmentPanel.populate_reference_hybe_choices(self.total_active_hybe_list)
        self.ui.SpotLocalizationPanel.populate_hybe_choices(self.total_active_hybe_list)
        self.ui.CelltypeDeterminationPanel.populate_hybe_choices(self.total_active_hybe_list)
        ip.populate_viewer_hybe_choices(self.total_active_hybe_list)
        ap.populate_cell_reference_hybe_choices(self.total_active_hybe_list)
        for name, populate in (('RNA', ap.populate_rna_reference_hybe_choices),
                               ('DNA', ap.populate_dna_reference_hybe_choices)):
            populate(self.ui.IngestionPanel.modality_data.get(name, {}).get('active_hybe_list', []))

        chp = self.ui.ChromatinTracingPanel
        chp.populate_hybe_list(self.total_active_hybe_list, default_checked=self._default_chromatin_tracing_hybes)
        chp.populate_reference_hybe_choices(self.total_active_hybe_list)
        self._refresh_chromatin_allele_hybe_choices()

    def _on_set_num_modalities(self):
        ip = self.ui.IngestionPanel
        try:
            n = int(ip.NumModalitiesLineEdit.text().strip())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, 'Modality Setup', 'Enter a valid integer number of modalities.')
            return
        if n < 1:
            QtWidgets.QMessageBox.warning(self, 'Modality Setup', 'Number of modalities must be at least 1.')
            return
        ip.build_modality_name_fields(n)
        ip.NumModalitiesLineEdit.setEnabled(False)
        ip.SetNumModalitiesPushButton.setEnabled(False)

    def _on_activate_modalities(self):
        ip = self.ui.IngestionPanel
        names = ip.modality_name_values()
        if any(not name for name in names):
            QtWidgets.QMessageBox.warning(self, 'Modality Setup', 'Every modality needs a name.')
            return
        if len(set(names)) != len(names):
            QtWidgets.QMessageBox.warning(self, 'Modality Setup', 'Modality names must be unique.')
            return
        ip.lock_modality_setup()
        self._activate_modalities(names)
        ip.LogTextEdit.append(f"Activated modalities: {', '.join(names)}")

    # -- ingestion --

    def _parse_layout(self):
        ip = self.ui.IngestionPanel
        layout_path = ip.LayoutPathLineEdit.text().strip()
        if not layout_path:
            QtWidgets.QMessageBox.warning(self, 'Parse Layout', 'Select an ExperimentLayout.xlsx first.')
            return
        try:
            self.hybe_records = preprocess.parse_experiment_layout(layout_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Parse Layout', f'{type(e).__name__}: {e}')
            return
        storage_path = ip.StoragePathLineEdit.text().strip()
        if storage_path:
            # the typed path may be newer than modality_data's stashed
            # copy -- declare it NOW so a completely fresh store's
            # modality-scoped params can be written (confirmed real boot
            # gate: first Parse Layout of a new dataset raised from
            # modality_of before anything could be ingested)
            if ip.current_modality:
                vlinks_store.declare_modality(storage_path, ip.current_modality)
                # v2 project bootstrap: a storage path named after its own
                # modality (<dp>/{modality}) declares the v2 layout -- write/
                # refresh the manifest so paths.py resolves the tree and
                # modality_of needs no HDF5 at all. v1 queue dirs
                # ('RNA_queue', 'data', ...) never match and stay v1.
                base = os.path.basename(os.path.abspath(storage_path).rstrip(os.sep))
                if base == ip.current_modality or paths.is_v2(storage_path):
                    dp = paths.project_root(storage_path)
                    m = paths.read_manifest(dp) or {'modalities': {}}
                    layouts = {n: v.get('layout_path', '') for n, v in m.get('modalities', {}).items()}
                    daxes = {n: v.get('dax_directory', '') for n, v in m.get('modalities', {}).items()}
                    layouts[ip.current_modality] = layout_path
                    daxes[ip.current_modality] = ip.DaxDirectoryLineEdit.text().strip()
                    names = sorted(set(list(m.get('modalities', {}).keys()) + list(ip.modality_names)))
                    paths.write_manifest(dp, names, layouts, daxes)
            # a real, confirmed fact about this storage path (which
            # ExperimentLayout it uses) -- lets a completely fresh session
            # reconstruct hybe_records/combo choices from vlinks.h5 alone
            # (see _refresh_params_from_vlinks), without ever loading a
            # config file.
            vlinks_store.write_global_params(storage_path, layout_path=layout_path)
        ip.populate_hybe_list(self.hybe_records, dax_directory=ip.DaxDirectoryLineEdit.text().strip())
        ip.RunIngestionPushButton.setEnabled(True)
        ip.LogTextEdit.append(f'Parsed {len(self.hybe_records)} hybe(s) from {layout_path}')
        self._check_ingestion_status(silent=True)
        # active_hybe_list/total_active_hybe_list refresh -- covers every
        # hybe-choosing combo (reference hybes, cell-alignment anchor,
        # spot localization/celltype hybe pickers, RNA/DNA cross-modal
        # reference hybes) in one place; see _refresh_active_hybe_lists.
        self._refresh_active_hybe_lists()
        self._activate_fov(self.ui.CellSegmentPanel.FovSpinBox.value())
        self._refresh_same_modality_results_list()
        self._refresh_cross_modal_results_list()
        self._refresh_fov_spinbox_bounds()
        self._refresh_celltype_names_from_vlinks()
        self._refresh_celltype_config_from_vlinks()

    def _refresh_celltype_config_from_vlinks(self):
        """
        Restores Celltype Determination's setup work -- FOV ranges,
        barcode channel assignments, and computed calibration -- from
        every storage path this session knows about (see write side,
        _persist_celltype_config). Only fills in entries not already
        present this session (never overwrites a live in-session edit,
        or a value from a storage path already read earlier in this same
        loop, with a stale/older one) -- same non-destructive-merge
        principle as _refresh_celltype_names_from_vlinks. Per explicit
        request: re-opening an already-calibrated experiment shouldn't
        require re-running Set FOV Ranges / Assign / Apply Calibration.
        """
        ctp = self.ui.CelltypeDeterminationPanel
        storage_paths = self._all_vlinks_storage_paths()
        if not storage_paths:
            return
        n_dropped_legacy = 0
        for storage_path in vlinks_store.distinct_stores(storage_paths):
            fov_ranges, barcode_channels, calibration, barcode_method = vlinks_store.read_celltype_config(storage_path)
            for name, range_string in fov_ranges.items():
                if name not in self._fov_ranges_by_celltype:
                    self._fov_ranges_by_celltype[name] = range_string
            for name, bch in barcode_channels.items():
                bch = tuple(bch)
                # a config saved before barcode channels carried their own
                # modality tag has bare (hybe, channel) 2-tuples -- there's
                # no way to safely recover which modality one of those
                # belonged to (a wrong guess would silently misclassify),
                # same "drop, don't guess" policy
                # _drop_legacy_matrix_keys already uses for cell.matrices'
                # own pre-tuple-key format. Dropped means this celltype's
                # barcode channel needs re-assigning, not that it crashes.
                if len(bch) != 3:
                    n_dropped_legacy += 1
                    continue
                if name not in self._barcode_channel_by_celltype:
                    self._barcode_channel_by_celltype[name] = bch
            for key in ('scale', 'lower_bound', 'upper_bound'):
                for bch, per_fov in calibration.get(key, {}).items():
                    bch = tuple(bch)
                    if len(bch) != 3:
                        n_dropped_legacy += 1
                        continue
                    dest = self._barcode_calibration[key].setdefault(bch, {})
                    for fov, val in per_fov.items():
                        dest.setdefault(int(fov), val)
            if barcode_method:
                ctp.BarcodeMethodComboBox.setCurrentText(
                    'Median' if barcode_method == 'median' else 'Vote (200-sample)')
        if n_dropped_legacy:
            ctp.LogTextEdit.append(f'{n_dropped_legacy} barcode-channel/calibration entr(ies) from an older '
                                   f'saved config had no modality tag -- dropped rather than guessed. '
                                   f'Re-assign/re-calibrate the affected celltype(s) if needed.')
        # any celltype named only via a barcode assignment/FOV range
        # (not yet reflected in any real classified cell) should still
        # show up in the shared identity list.
        for name in list(self._fov_ranges_by_celltype) + list(self._barcode_channel_by_celltype):
            if name not in self.current_celltype_list:
                self.current_celltype_list.append(name)
        ctp.ensure_celltype_names(self.current_celltype_list)
        self._refresh_celltype_summaries()

    def _refresh_celltype_names_from_vlinks(self):
        """
        Scans every FOV in the Ingestion tab's FOV list, across every
        storage path this session knows about, for cells that already
        carry a real classified celltype (from an earlier run/session,
        already saved to vlinks.h5) -- merges any such names into
        self.current_celltype_list (only ever adding, never removing --
        a name already there from a loaded config, or typed manually,
        always survives) and pushes the merged result into the Celltype
        Determination panel's shared identity list. Per explicit request:
        the listview should already be usable the moment vlinks has real
        classified data, without the user first manually re-typing every
        name back in.
        """
        ip = self.ui.IngestionPanel
        ctp = self.ui.CelltypeDeterminationPanel
        storage_paths = self._all_vlinks_storage_paths()
        fov_list = self._parse_fov_list(ip.FovListLineEdit.text())
        if not storage_paths or not fov_list:
            return
        for storage_path in vlinks_store.distinct_stores(storage_paths):
            for fov in fov_list:
                cell_dicts, _ = vlinks_store.read_cells(storage_path, fov)
                if not cell_dicts:
                    continue
                for d in cell_dicts:
                    name = d.get('celltype')
                    if name and name not in self.current_celltype_list:
                        self.current_celltype_list.append(name)
        ctp.ensure_celltype_names(self.current_celltype_list)

    def _refresh_fov_spinbox_bounds(self):
        """
        Every FOV-picking spinbox in the Alignment panel should only
        accept values from the Ingestion tab's own FOV list -- typing/
        scrolling to an FOV that was never parsed/ingested just produces
        a downstream warning today; bounding the spinbox's range instead
        catches it at the input itself. Uses [min(fov_list), max(fov_list)]
        since QSpinBox only supports one continuous range, not an
        arbitrary set -- a non-contiguous list (e.g. "1,2,5") still lets
        3/4 through, a real, pre-existing limitation of QSpinBox itself,
        not something worth a custom widget for. Falls back to the wide-
        open default range when the FOV list is empty/unparsed, so typing
        still works before a layout has been parsed at all.
        """
        ip, ap, sp = self.ui.IngestionPanel, self.ui.AlignmentPanel, self.ui.SpotLocalizationPanel
        fov_list = self._parse_fov_list(ip.FovListLineEdit.text())
        spinboxes = [ap.SameModalityFovSpinBox, ap.SameModalityOverlayFovSpinBox,
                     ap.CrossModalFovSpinBox, ap.CrossModalOverlayFovSpinBox,
                     ap.CellFovSpinBox, ap.CellOverlayFovSpinBox, sp.FovSpinBox]
        if not fov_list:
            for sb in spinboxes:
                sb.setRange(1, 100000)
        else:
            lo, hi = min(fov_list), max(fov_list)
            for sb in spinboxes:
                sb.setRange(lo, hi)
        # Per confirmed real bug: editing the FOV list (ip.FovListLineEdit.
        # editingFinished, this method's own main trigger) only ever
        # updated the spinboxes' RANGE -- Same-Modality Alignment's own
        # Results list (which iterates the FULL fov_list, see its own
        # docstring) never refreshed to match, so a newly-added/removed
        # FOV silently didn't show up (or a removed one lingered) until
        # something ELSE happened to trigger a Results refresh. Self-
        # guarding (early-returns if reference_hybe/storage_path aren't
        # set yet), safe to call unconditionally here.
        self._refresh_same_modality_results_list()

    def _refresh_same_modality_results_list(self):
        """
        Populates the FULL Results list -- every FOV in the FOV list with
        EITHER a disk-persisted same-modality alignment, an in-memory
        fov_matrices entry (already accepted/auto-saved this session), OR
        a not-yet-accepted staged current-FOV result -- never just
        whichever FOV happened to be aligned most recently. Rows for a
        staged (unsaved) result are marked "[pending]" so it's clear
        Accept hasn't been clicked yet. Called after every run/accept/
        reject, and whenever a layout is (re)parsed (directly, or via a
        modality switch) so results already on disk from an earlier
        session aren't left out just because this session hasn't (re-)run
        alignment yet.
        """
        ip, ap = self.ui.IngestionPanel, self.ui.AlignmentPanel
        reference_hybe = ap.current_reference_hybe()
        modality = ap.current_reference_modality()
        storage_path = self._storage_path_for_modality(modality)
        hybe_records = self.ui.IngestionPanel.modality_data.get(modality, {}).get('active_hybe_list', []) if modality else []
        fov_list = self._parse_fov_list(ip.FovListLineEdit.text())
        ap.SameModalityResultsListWidget.clear()
        if not storage_path or not reference_hybe or not fov_list or not hybe_records:
            return
        disk_results = {}
        for fov in fov_list:
            ready, _, _ = self._ingested_hybes_for_fov(storage_path, fov, hybe_records)
            if not ready:
                continue
            ready_records = [r for r in hybe_records if r['folder'] in ready]
            matrices = alignment.read_same_modality_matrices(storage_path, fov, ready_records)
            if matrices:
                disk_results[fov] = matrices
        if disk_results:
            for _fov, _m in disk_results.items():
                self._merge_fov_matrices(_fov, _m)

        display_results = dict(disk_results)
        # Keyed by FOV now, spanning both modalities -- narrow to the one
        # this list is showing rather than filtering on an outer path key.
        for fov in self.fov_matrices:
            matrices = self._fov_matrices_for(storage_path, fov)
            if matrices:
                display_results[fov] = matrices
        pending_fovs = set()
        if self._pending_same_modality_alignment:
            for fov, matrices in self._pending_same_modality_alignment.items():
                display_results[fov] = matrices
                pending_fovs.add(fov)
        if not display_results:
            return
        self._same_modality_context = {'storage_path': storage_path, 'hybe_records': hybe_records, 'reference_hybe': reference_hybe}
        for fov in sorted(display_results.keys()):
            matrices = display_results[fov]
            suffix = ' [pending]' if fov in pending_fovs else ''
            for key, H in matrices.items():
                hybe = key[0] if isinstance(key, tuple) else key
                item = QtWidgets.QListWidgetItem(f'FOV{fov:02d} {_matrix_summary(hybe, H)}{suffix}')
                item.setData(QtCore.Qt.UserRole, (fov, hybe))
                ap.SameModalityResultsListWidget.addItem(item)

    def _run_ingestion(self):
        ip = self.ui.IngestionPanel
        selected_folders = ip.hybe_checkbox_items()
        if not selected_folders:
            QtWidgets.QMessageBox.warning(self, 'Run Ingestion', 'Check at least one hybe to ingest.')
            return
        selected_records = [r for r in self.hybe_records if r['folder'] in selected_folders]
        fov_list = self._parse_fov_list(ip.FovListLineEdit.text())
        if not fov_list:
            QtWidgets.QMessageBox.warning(self, 'Run Ingestion', 'Enter at least one FOV (e.g. 1,2).')
            return
        dax_directory = ip.DaxDirectoryLineEdit.text().strip()
        storage_path = ip.StoragePathLineEdit.text().strip()
        modality = ip.ModalityComboBox.currentText()
        if not dax_directory or not storage_path:
            QtWidgets.QMessageBox.warning(self, 'Run Ingestion', 'Select a DAX directory and storage path first.')
            return
        overwrite_mode = self._confirm_overwrite([(storage_path, fov_list, selected_records)])
        if overwrite_mode is None:
            return

        ip.RunIngestionPushButton.setEnabled(False)
        ip.ProgressBar.setValue(0)
        ip.ProgressBar.setMaximum(len(selected_records) * len(fov_list))
        ip.LogTextEdit.append(f'Starting ingestion ({overwrite_mode}): {len(fov_list)} FOV(s) x {len(selected_records)} hybe(s)...')
        self.statusBar().showMessage('Ingesting...')

        self._ingestion_worker = IngestionWorker(fov_list, selected_records, dax_directory, storage_path, modality,
                                                  overwrite=(overwrite_mode == 'overwrite'),
                                                  max_workers=ip.IngestWorkersSpinBox.value())
        self._wire_ingestion_ui_guard(self._ingestion_worker)
        self._ingestion_worker.progress.connect(self._on_ingestion_progress)
        self._ingestion_worker.finished_ok.connect(self._on_ingestion_finished)
        self._ingestion_worker.failed.connect(self._on_ingestion_failed)
        self._ingestion_worker.start()

    def _existing_h5_targets(self, storage_path, fov_list, selected_records):
        existing = []
        for fov in fov_list:
            for record in selected_records:
                path = os.path.join(storage_path, f'FOV{fov:02d}', f"{record['folder']}_stack.h5")
                if os.path.exists(path):
                    existing.append(path)
        return existing

    def _confirm_overwrite(self, jobs):
        """
        jobs: list of (storage_path, fov_list, selected_records) tuples.
        Returns 'overwrite' (re-convert everything, including targets that
        already exist), 'append' (skip already-existing targets, only
        convert what's actually missing), or None (cancel) -- the
        third option matters a lot for partially-done ingestion runs
        (e.g. adding one more hybe to an experiment that's already mostly
        ingested): forcing an all-or-nothing overwrite/cancel choice would
        either needlessly re-convert everything that already succeeded, or
        block adding the missing piece at all.
        """
        existing = []
        total = 0
        for storage_path, fov_list, selected_records in jobs:
            existing += self._existing_h5_targets(storage_path, fov_list, selected_records)
            total += len(fov_list) * len(selected_records)
        if not existing:
            return 'overwrite'  # nothing to conflict with -- overwrite vs append makes no difference

        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle('Files already exist')
        box.setText(f'{len(existing)} of {total} target H5 file(s) already exist under the selected storage path(s).\n'
                    'How should this ingestion run handle them?')
        overwrite_button = box.addButton('Overwrite All', QtWidgets.QMessageBox.AcceptRole)
        append_button = box.addButton('Append (skip existing)', QtWidgets.QMessageBox.AcceptRole)
        box.addButton('Cancel', QtWidgets.QMessageBox.RejectRole)
        box.setDefaultButton(append_button)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is overwrite_button:
            return 'overwrite'
        if clicked is append_button:
            return 'append'
        return None

    def _on_ingestion_progress(self, done, total, message):
        ip = self.ui.IngestionPanel
        ip.ProgressBar.setValue(done)
        ip.LogTextEdit.append(message)

    def _on_ingestion_finished(self, n_errors, n_total, error_lines):
        ip = self.ui.IngestionPanel
        ip.RunIngestionPushButton.setEnabled(True)
        self._check_ingestion_status(silent=True)
        # a freshly-ingested hybe must become choosable immediately, not
        # only after the user happens to switch modality -- ingestion
        # completion is the other of the two moments active_hybe_list is
        # allowed to change (per explicit design; see
        # _refresh_active_hybe_lists).
        self._refresh_active_hybe_lists()
        self._activate_fov(self.ui.CellSegmentPanel.FovSpinBox.value())
        if n_errors > 0:
            # the worker never raised (each task's error is caught+
            # returned, not thrown), so this branch is reachable even
            # though finished_ok fired -- don't call it "successful" just
            # because nothing raised.
            ip.LogTextEdit.append(f'Ingestion finished with {n_errors}/{n_total} task(s) FAILED:')
            for line in error_lines:
                ip.LogTextEdit.append(f'  {line}')
            self.statusBar().showMessage(f'Ingestion finished with {n_errors}/{n_total} failure(s).', 5000)
            QtWidgets.QMessageBox.warning(self, 'Ingestion finished with errors',
                                          f'{n_errors} of {n_total} FOV/hybe task(s) failed -- see the log for details.\n'
                                          f'First error: {error_lines[0]}')
        else:
            ip.LogTextEdit.append('Ingestion complete.')
            self.statusBar().showMessage('Ingestion complete.', 5000)
            QtWidgets.QMessageBox.information(self, 'Ingestion complete', f'Ingestion finished successfully ({n_total} task(s)).')

    def _on_ingestion_failed(self, message):
        ip = self.ui.IngestionPanel
        ip.LogTextEdit.append(f'Ingestion FAILED: {message}')
        ip.RunIngestionPushButton.setEnabled(True)
        self.statusBar().clearMessage()
        self._check_ingestion_status(silent=True)
        QtWidgets.QMessageBox.critical(self, 'Ingestion error', message)

    @staticmethod
    def _ingested_hybes_for_fov(storage_path, fov, hybe_records):
        """
        Per-hybe readiness for one FOV, read from vlinks.h5 alone (never
        opens a raw {hybe}_stack.h5) -- a single cheap file open instead of
        N heavy ones, since IngestionWorker now writes a real MIP copy into
        vlinks.h5 (vlinks_store.write_hybe_mip) right after each hybe's raw
        conversion succeeds. "Ready" means that hybe's vlinks.h5 MIP has
        every channel its own ExperimentLayout record declares -- a hybe
        with a MIP group but a missing channel (e.g. write_hybe_mip
        interrupted mid-loop) is INCOMPLETE, not silently trusted, same
        distinction the old raw-file-based version made. Returns (ready,
        missing, invalid) hybe-folder lists.
        """
        ready, missing, invalid = [], [], []
        if paths.is_v2(storage_path):
            # v2: MIP files are written atomically, so existence IS
            # completeness -- the whole check is ONE directory listing
            # instead of a vlinks open per hybe (which at 100 FOVs x 100
            # hybes was ~10,000 network opens per refresh).
            present = paths.mips_present(storage_path, fov)
            for record in hybe_records:
                (ready if record['folder'] in present else missing).append(record['folder'])
            return ready, missing, invalid
        for record in hybe_records:
            hybe = record['folder']
            channels_present = vlinks_store.mip_channels_present(storage_path, fov, hybe)
            if channels_present is None:
                missing.append(hybe)
            elif all(str(c) in channels_present for c in record['channels']):
                ready.append(hybe)
            else:
                invalid.append(hybe)
        return ready, missing, invalid

    def _check_ingestion_status(self, silent=False):
        ip = self.ui.IngestionPanel
        storage_path = ip.StoragePathLineEdit.text().strip()
        fov_list = self._parse_fov_list(ip.FovListLineEdit.text())
        if not storage_path or not self.hybe_records or not fov_list:
            if not silent:
                QtWidgets.QMessageBox.warning(self, 'Check Ingestion Status',
                                              'Parse a layout, and set a storage path + FOV list, first.')
            return

        lines = []
        any_missing_or_invalid = False
        for fov in fov_list:
            ready, missing, invalid = self._ingested_hybes_for_fov(storage_path, fov, self.hybe_records)
            status = f'FOV{fov:02d}: {len(ready)}/{len(self.hybe_records)} ready'
            if missing:
                status += f' | MISSING: {", ".join(missing)}'
            if invalid:
                status += f' | INCOMPLETE/UNREADABLE: {", ".join(invalid)}'
            if missing or invalid:
                any_missing_or_invalid = True
            lines.append(status)

        ip.IngestionStatusTextEdit.setPlainText('\n'.join(lines))
        self.statusBar().showMessage(
            'Ingestion status: some FOV/hybe combinations need ingestion.' if any_missing_or_invalid
            else 'Ingestion status: all checked FOV/hybe combinations ready.', 5000)

    def _hybe_records_for_storage_path(self, storage_path):
        """
        The hybe list belonging to storage_path's OWN modality, not
        whichever modality happens to be currently active (self.hybe_
        records) -- a caller iterating multiple storage paths at once
        must resolve each one's own hybe records independently, since
        reusing the active modality's list for a DIFFERENT modality's
        path silently corrupts it the moment its hybe folder names differ
        from the active one's (the normal case, e.g. Hyb_101.. vs
        Hyb_002..).
        """
        ip = self.ui.IngestionPanel
        if storage_path == ip.StoragePathLineEdit.text().strip():
            return self.hybe_records
        for data in self.ui.IngestionPanel.modality_data.values():
            if data['storage_path'] == storage_path and data['layout_path']:
                try:
                    return preprocess.parse_experiment_layout(data['layout_path'])
                except Exception:
                    return []
        return []

    def _reference_hybe_for_storage_path(self, storage_path):
        """
        The real, persisted same-modality reference hybe for this
        storage_path -- read straight from that path's own vlinks.h5
        global params (written whenever a same-modality alignment run is
        accepted), not from live UI state. A caller iterating multiple
        storage paths at once genuinely needs each path's own
        independently-persisted fact, not "whatever the single Reference
        Hybe combo happens to show right now" -- that combo no longer
        even has a notion of "per modality" since it stopped being reset
        by a modality switch.
        """
        params = vlinks_store.read_global_params(storage_path)
        return (params or {}).get('same_modality_reference_hybe', '')

    def _ingestion_is_running(self):
        """True while an IngestionWorker is live -- single run or queued."""
        worker = getattr(self, '_ingestion_worker', None)
        return worker is not None and worker.isRunning()

    def _show_cell_spot_status_displayer(self):
        # Refused outright while an ingestion is live, for two reasons that
        # both trace back to this refresh running on the GUI thread:
        # _refresh_cell_spot_status_full holds the vlinks.h5 lock for its
        # whole duration, so the coordinator's MIP writes would stall behind
        # it (they queue and catch up -- nothing is lost, but the run stops
        # advancing), and the window goes unresponsive meanwhile. Before the
        # lock existed this same overlap did not merely stall, it FAILED:
        # "ingested but failed to write vlinks.h5 MIP: Unable to open file
        # (file is already open for read-only)", losing that hybe's MIP.
        # This guard goes away once the refresh moves off the GUI thread.
        if self._ingestion_is_running():
            box = QtWidgets.QMessageBox(self)
            box.setIcon(QtWidgets.QMessageBox.Information)
            box.setWindowTitle('Cell/Spot status')
            box.setText('An ingestion is running.')
            box.setInformativeText(
                'This view reads the whole store, which would freeze the '
                'window and stall the ingestion for as long as it took. It '
                'becomes available again the moment the run finishes.')
            box.exec_()
            return
        d = self.cell_spot_status_displayer
        self._refresh_cell_spot_status_full()
        d.show()
        d.raise_()

    def _wire_ingestion_ui_guard(self, worker):
        """
        Grey the Cell/Spot status button out for the life of one ingestion.

        Driven off QThread's own started/finished rather than the six places
        that already toggle the Run buttons: finished fires whether run()
        returned or raised, so no failure path can leave the button stuck
        disabled.
        """
        button = self.ui.IngestionPanel.ShowCellSpotStatusDisplayerPushButton
        worker.started.connect(lambda: button.setEnabled(False))
        worker.finished.connect(lambda: button.setEnabled(True))

    def _status_storage_path(self):
        """
        Any configured storage path, for the status viewer's reads.

        With one unified vlinks every storage_path resolves to the same
        file, and the reads this serves -- cells, spots, alleles -- are
        FOV-scoped rather than modality-scoped, so they already span both
        modalities. Which path is passed only decides which directory the
        resolver walks up from, never what comes back.
        """
        for name in self.ui.IngestionPanel.modality_names:
            path = self._storage_path_for_modality(name)
            if path:
                return path
        return None

    def _refresh_cell_spot_status_full(self):
        """
        Re-derives the FOV choices for ALL THREE panels from the Ingestion
        tab's own FOV list (Refresh's own job -- the CHOICES themselves,
        not just the currently-selected scope's data, might be stale;
        see the class docstring on why this is separate from a plain
        combo-change refresh), then refreshes each panel for whatever FOV
        ends up selected.
        """
        d = self.cell_spot_status_displayer
        storage_path = self._status_storage_path()
        fov_list = self._parse_fov_list(self.ui.IngestionPanel.FovListLineEdit.text())
        d.set_cell_fov_choices(fov_list)
        d.set_spot_fov_choices(fov_list)
        d.set_allele_fov_choices(fov_list)
        if not storage_path or not fov_list:
            d.set_cell_data([], 0, 0)
            d.set_spot_hybe_choices([])
            d.set_spot_channel_choices([])
            d.set_spot_data([], 0)
            d.set_allele_data([], 0, 0)
            return
        # ONE vlinks.h5 open for the whole refresh instead of one per read.
        # Between them the four panels below issue several hundred reads
        # (per-FOV totals in the cell and allele panels, per-(FOV, modality)
        # matrices), and each one used to open and close the file for
        # itself. Those opens were the dominant cost of this refresh AND the
        # window in which ingestion's per-task MIP write collided with it --
        # see vlinks_store._open_vlinks for the failure that caused.
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            with vlinks_store.vlinks_session(storage_path):
                self._refresh_cell_spot_status_matrix_panel()
                self._refresh_cell_spot_status_cell_panel()
                self._on_cell_spot_status_spot_scope_changed()
                self._refresh_cell_spot_status_allele_panel()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _refresh_cell_spot_status_matrix_panel(self):
        """
        Ground-truth matrix check -- reads DIRECTLY off vlinks.h5 via
        vlinks_store (never self.fov_matrices/self.cross_modal_result,
        the in-memory caches a stale reference-hybe combo can contaminate
        -- see _refresh_cross_modal_results_list's own comment on the
        confirmed real bug this sidesteps), for EVERY FOV in the
        Ingestion tab's FOV list at once (same "one row per FOV" shape
        _refresh_same_modality_results_list/_refresh_cross_modal_results_
        list already use), scoped only by the shared Modality combo.
        """
        d = self.cell_spot_status_displayer
        storage_path = self._status_storage_path()
        fov_list = self._parse_fov_list(self.ui.IngestionPanel.FovListLineEdit.text())
        if not storage_path or not fov_list:
            d.set_matrix_data([])
            return
        global_params = vlinks_store.read_global_params(storage_path) or {}
        rows = []
        # One row per (FOV, modality). Matrices are the one thing here that
        # IS modality-scoped -- /FOV##/matrix/{modality}/{hybe} -- and the
        # cross-modal bridge hybe appears under both, meaning different
        # things, so showing a single modality would hide half the store and
        # make the bridge look unambiguous when it is not.
        for fov in fov_list:
            for modality in self.ui.IngestionPanel.modality_names:
                path = self._storage_path_for_modality(modality)
                if not path:
                    continue
                hybes = [r['folder'] for r in self._active_hybe_records_for_modality(modality)]
                matrices = vlinks_store.read_same_modality_matrices(path, fov, hybes)
                # keys are (hybe, modality); the row already names the
                # modality, so the child rows show the hybe alone.
                same_modality = [(key[0], _matrix_dxdy_angle(H))
                                 for key, H in sorted(matrices.items())]
                cross_modal = None
                # dz is a measured component of the cross-modal result
                # (stored right beside H_across) -- shown here per explicit
                # request: the viewer reports the data structure honestly,
                # never a convenient subset of it.
                if modality == global_params.get('cross_modal_dna_modality'):
                    H_across = vlinks_store.read_cross_modal_matrix(path, fov)
                    if H_across is not None:
                        dz_across = vlinks_store.read_cross_modal_z(path, fov)
                        cross_modal = f'{_matrix_dxdy_angle(H_across)}, dz={dz_across:.2f}'
                elif modality == global_params.get('cross_modal_rna_modality'):
                    # The RNA side is the cross-modal TARGET frame, never
                    # itself shifted -- shown as an explicit identity row
                    # rather than omitted, so both sides of a link read
                    # consistently.
                    cross_modal = f'{_matrix_dxdy_angle(np.eye(3))}, dz=0.00'
                rows.append({'fov': fov, 'modality': modality,
                             'same_modality': same_modality, 'cross_modal': cross_modal})
        d.set_matrix_data(rows)

    def _refresh_cell_spot_status_allele_panel(self):
        d = self.cell_spot_status_displayer
        storage_path = self._status_storage_path()
        fov = d.current_allele_fov()
        fov_list = self._parse_fov_list(self.ui.IngestionPanel.FovListLineEdit.text())
        if not storage_path or fov is None:
            d.set_allele_data([], 0, len(fov_list))
            return
        allele_dicts = vlinks_store.read_fov_alleles(storage_path, fov)
        n_total = sum(c['alleles'] for c in vlinks_store.fov_counts(storage_path, fov_list).values())
        d.set_allele_data(allele_dicts, n_total, len(fov_list))

    def _refresh_cell_spot_status_cell_panel(self):
        d = self.cell_spot_status_displayer
        storage_path = self._status_storage_path()
        fov = d.current_cell_fov()
        fov_list = self._parse_fov_list(self.ui.IngestionPanel.FovListLineEdit.text())
        if not storage_path or fov is None:
            d.set_cell_data([], 0, len(fov_list))
            return
        cell_dicts, _ = vlinks_store.read_cells(storage_path, fov)
        cell_dicts = cell_dicts or []
        # distmap ON DEMAND only (checkbox, default off): derived fresh
        # from the store's current spots for display, never read from a
        # persisted copy (the saved slot is a legacy field that stays
        # empty by design) -- and never computed on a routine refresh,
        # per explicit decision: it is an O(n_spots^2) analysis product,
        # not status.
        if d.DistmapCheckBox.isChecked():
            import scipy.spatial.distance as _ssd
            spots_by_cell = {}
            for sd in vlinks_store.read_spots(storage_path, fov):
                if int(sd.get('cell', -1)) != -1:
                    spots_by_cell.setdefault(int(sd['cell']), []).append(sd['coordinate'])
            for cell_dict in cell_dicts:
                pos = spots_by_cell.get(int(cell_dict.get('id', -1)))
                if pos and len(pos) > 1:
                    cell_dict['distmap'] = _ssd.squareform(_ssd.pdist(np.array(pos)))
        n_total = sum(c['cells'] for c in vlinks_store.fov_counts(storage_path, fov_list).values())
        # Spots are no longer nested in the cell dict; count them per cell
        # from the FOV's own spot store for the tree's "(N spots)" label.
        spots_by_cell_id = {}
        for s in vlinks_store.read_spots(storage_path, fov):
            cid = int(s.get('cell', -1))
            if cid != -1:
                spots_by_cell_id[cid] = spots_by_cell_id.get(cid, 0) + 1
        d.set_cell_data(cell_dicts, n_total, len(fov_list), spots_by_cell_id)

    def _on_cell_spot_status_spot_scope_changed(self):
        """
        Single handler for all three Spot-panel combos (FOV/hybe/channel)
        -- re-derives hybe/channel CHOICES for the current FOV first
        (cheap, and a no-op on the user's own current selection whenever
        the FOV didn't actually change, since set_spot_hybe_choices/
        set_spot_channel_choices preserve it if still present), then
        refreshes the panel itself.
        """
        self._refresh_cell_spot_status_spot_choices()
        self._refresh_cell_spot_status_spot_panel()

    def _refresh_cell_spot_status_spot_choices(self):
        d = self.cell_spot_status_displayer
        storage_path = self._status_storage_path()
        fov = d.current_spot_fov()
        if not storage_path or fov is None:
            d.set_spot_hybe_choices([])
            d.set_spot_channel_choices([])
            return
        # slice NAMES only -- never the spot payload (see
        # vlinks_store.spot_slices: full-parse for combo choices is
        # seconds of waste per refresh at real scale)
        slices = vlinks_store.spot_slices(storage_path, fov)
        d.set_spot_hybe_choices(sorted({h for _m, h, _c in slices}))
        hybe = d.current_spot_hybe()
        channels = sorted({c for _m, h, c in slices if hybe is None or h == hybe})
        d.set_spot_channel_choices(channels)

    def _ordered_spot_dicts_for_scope(self, storage_path, fov, hybe, channel):
        """
        [(global_index, spot_dict), ...], 1-based -- SAME ordering/
        numbering scheme as _global_spot_order/_global_spot_index_map
        (unassigned pool first in its own on-disk list order, then cells
        sorted by cell id, each contributing its own spots in on-disk
        list order), just rebuilt from persisted dicts (read_cells/
        read_fov_spots) instead of live ASpot objects -- so a spot's
        number in CellSpotStatusDisplayer's tree matches its number in
        the crop displayer / 3D-localization popup, per confirmed real
        bug: labeling from spot_dict['id'] instead showed EVERY spot as
        "Spot 0", since ASpot.id defaults to 0 and nothing that creates a
        spot in this app ever sets it to a real per-spot value.
        """
        cell_dicts, _ = vlinks_store.read_cells(storage_path, fov)
        cell_dicts = cell_dicts or []
        # Spots live in the FOV's own store now (assigned and unassigned
        # together, differing only in `cell`), not nested inside each cell's
        # own dict -- group by owning cell here rather than reading a
        # 'spots' key that no longer exists on a persisted cell.
        # ONLY the selected slice is read (lazy detail, per explicit
        # request): the whole-FOV read this replaces unpacked every
        # slice's payload just to filter one out.
        slice_spots = []
        for mod, h, c in vlinks_store.spot_slices(storage_path, fov):
            if h == hybe and c == channel:
                slice_spots.extend(vlinks_store.read_spots(storage_path, fov, mod, hybe, channel))
        by_cell = {}
        for s in slice_spots:
            by_cell.setdefault(int(s.get('cell', -1)), []).append(s)
        ordered = list(by_cell.get(-1, []))
        for c in sorted(cell_dicts, key=lambda c: c['id']):
            ordered.extend(by_cell.get(c['id'], []))
        # Spots whose owner was PURGED still exist and must stay visible --
        # hiding them made a stale link invisible instead of inspectable.
        # They keep their stored cell id in the label; the next save's
        # reassignment resolves them against the current cells.
        live_ids = {c['id'] for c in cell_dicts}
        for cid in sorted(k for k in by_cell if k != -1 and k not in live_ids):
            for s in by_cell[cid]:
                if s['hybe'] == hybe and s['channel'] == channel:
                    ordered.append(s)
        return list(enumerate(ordered, start=1))

    def _refresh_cell_spot_status_spot_panel(self):
        d = self.cell_spot_status_displayer
        storage_path = self._status_storage_path()
        fov = d.current_spot_fov()
        hybe = d.current_spot_hybe()
        channel = d.current_spot_channel()
        fov_list = self._parse_fov_list(self.ui.IngestionPanel.FovListLineEdit.text())
        if not storage_path or fov is None or hybe is None or channel is None:
            d.set_spot_data([], 0)
            return
        indexed = self._ordered_spot_dicts_for_scope(storage_path, fov, hybe, channel)
        n_total = sum(c['spots'] for c in vlinks_store.fov_counts(storage_path, fov_list).values())
        d.set_spot_data(indexed, n_total)

    def _show_mip_viewer(self, silent=False):
        """
        silent=True is used by the "keep an already-open viewer live as
        combos change" wiring -- best-effort, no error dialogs, since a
        combo box can be in a legitimate (if now further guarded against
        via blockSignals) transient state mid-update that isn't a real
        user-facing error.
        """
        ip = self.ui.IngestionPanel
        hybe = ip.current_viewer_hybe()
        # storage_path comes from the SELECTED hybe's own modality, not
        # from whatever the Ingestion tab's own combo happens to be
        # showing -- ViewerHybeComboBox can now offer any parsed
        # modality's hybes at once (see populate_viewer_hybe_choices), so
        # those two can genuinely differ.
        storage_path = self._storage_path_for_modality(ip.current_viewer_modality())
        channel_text = ip.ViewerChannelComboBox.currentText()
        if not storage_path or not hybe or not channel_text:
            if not silent:
                QtWidgets.QMessageBox.warning(self, 'Show MIP Viewer',
                                              'Set storage path, and parse a layout to pick a hybe/channel, first.')
            return
        fov = ip.ViewerFovSpinBox.value()
        channel = int(channel_text)
        # Reads vlinks.h5, not the raw stack file -- per explicit principle,
        # display code should rely on vlinks, not raw/stacked data. This
        # does mean a bug in write_hybe_mip's own copy step wouldn't be
        # caught by this viewer anymore; Check Ingestion Status (which
        # verifies vlinks.h5 has every declared channel, see
        # _ingested_hybes_for_fov) is the tool for that now.
        try:
            mip = vlinks_store.read_hybe_mip(storage_path, fov, hybe, channel)
            if mip is None:
                raise ValueError(f'FOV{fov:02d} {hybe} ch{channel} not in vlinks.h5 -- ingest it first.')
        except Exception as e:
            if not silent:
                QtWidgets.QMessageBox.critical(self, 'Show MIP Viewer', f'{type(e).__name__}: {e}')
            return
        self.mip_viewer.set_data(mip, title=f'FOV{fov:02d} {hybe} ch{channel}')
        self.mip_viewer.show()
        self.mip_viewer.raise_()

    # -- job queue (batch ingestion across experiments/modalities) --

    def _add_job_to_queue(self):
        ip = self.ui.IngestionPanel
        if not self.hybe_records:
            QtWidgets.QMessageBox.warning(self, 'Add to Queue', 'Parse a layout first.')
            return
        selected_folders = ip.hybe_checkbox_items()
        if not selected_folders:
            QtWidgets.QMessageBox.warning(self, 'Add to Queue', 'Check at least one hybe to ingest.')
            return
        fov_list = self._parse_fov_list(ip.FovListLineEdit.text())
        layout_path = ip.LayoutPathLineEdit.text().strip()
        dax_directory = ip.DaxDirectoryLineEdit.text().strip()
        storage_path = ip.StoragePathLineEdit.text().strip()
        if not fov_list or not layout_path or not dax_directory or not storage_path:
            QtWidgets.QMessageBox.warning(self, 'Add to Queue',
                                          'Fill in the layout, DAX directory, storage path, and FOV list first.')
            return
        job = {
            'layout_path': layout_path,
            'dax_directory': dax_directory,
            'storage_path': storage_path,
            'modality': ip.ModalityComboBox.currentText(),
            'fov_list': fov_list,
            'selected_records': [r for r in self.hybe_records if r['folder'] in selected_folders],
            'hybe_records': list(self.hybe_records),
        }
        ip.add_job_item(job)

    def _remove_selected_jobs(self):
        self.ui.IngestionPanel.remove_selected_jobs()

    def _load_job_into_form(self, item):
        job = item.data(QtCore.Qt.UserRole)
        ip = self.ui.IngestionPanel
        ip.LayoutPathLineEdit.setText(job['layout_path'])
        ip.DaxDirectoryLineEdit.setText(job['dax_directory'])
        ip.StoragePathLineEdit.setText(job['storage_path'])
        ip.ModalityComboBox.setCurrentText(job['modality'])
        ip.FovListLineEdit.setText(','.join(str(f) for f in job['fov_list']))
        self._refresh_fov_spinbox_bounds()
        self.hybe_records = job['hybe_records']
        ip.populate_hybe_list(self.hybe_records)
        self.ui.AlignmentPanel.populate_reference_hybe_choices([(r, job['modality']) for r in self.hybe_records])
        selected_folders = {r['folder'] for r in job['selected_records']}
        for i in range(ip.HybeListWidget.count()):
            it = ip.HybeListWidget.item(i)
            it.setCheckState(QtCore.Qt.Checked if it.data(QtCore.Qt.UserRole) in selected_folders else QtCore.Qt.Unchecked)
        ip.RunIngestionPushButton.setEnabled(True)

    def _run_job_queue(self):
        ip = self.ui.IngestionPanel
        jobs = ip.queued_jobs()
        if not jobs:
            QtWidgets.QMessageBox.warning(self, 'Run Queued Jobs', 'Add at least one job to the queue first.')
            return
        overwrite_mode = self._confirm_overwrite([(job['storage_path'], job['fov_list'], job['selected_records']) for job in jobs])
        if overwrite_mode is None:
            return
        self._job_queue_overwrite = (overwrite_mode == 'overwrite')
        self._job_queue = jobs
        self._job_queue_index = 0
        ip.RunQueuePushButton.setEnabled(False)
        ip.RunIngestionPushButton.setEnabled(False)
        self.statusBar().showMessage('Ingesting...')
        self._run_next_queued_job()

    def _run_next_queued_job(self):
        ip = self.ui.IngestionPanel
        if self._job_queue_index >= len(self._job_queue):
            ip.LogTextEdit.append(f'Job queue complete ({len(self._job_queue)} job(s)).')
            ip.RunQueuePushButton.setEnabled(True)
            ip.RunIngestionPushButton.setEnabled(True)
            self.statusBar().showMessage('Job queue complete.', 5000)
            QtWidgets.QMessageBox.information(self, 'Job queue complete',
                                              f'All {len(self._job_queue)} queued job(s) finished successfully.')
            return
        job = self._job_queue[self._job_queue_index]
        ip.ProgressBar.setValue(0)
        ip.ProgressBar.setMaximum(len(job['selected_records']) * len(job['fov_list']))
        ip.LogTextEdit.append(f"Starting queued job {self._job_queue_index + 1}/{len(self._job_queue)}: "
                              f"{job['modality']}, FOV {job['fov_list']}...")
        self._ingestion_worker = IngestionWorker(job['fov_list'], job['selected_records'],
                                                  job['dax_directory'], job['storage_path'], job['modality'],
                                                  overwrite=self._job_queue_overwrite,
                                                  max_workers=self.ui.IngestionPanel.IngestWorkersSpinBox.value())
        self._wire_ingestion_ui_guard(self._ingestion_worker)
        self._ingestion_worker.progress.connect(self._on_ingestion_progress)
        self._ingestion_worker.finished_ok.connect(self._on_queued_job_finished)
        self._ingestion_worker.failed.connect(self._on_queued_job_failed)
        self._ingestion_worker.start()

    def _on_queued_job_finished(self, n_errors, n_total, error_lines):
        ip = self.ui.IngestionPanel
        self._check_ingestion_status(silent=True)
        self._activate_fov(self.ui.CellSegmentPanel.FovSpinBox.value())
        if n_errors > 0:
            ip.LogTextEdit.append(f'Job {self._job_queue_index + 1}/{len(self._job_queue)}: '
                                  f'{n_errors}/{n_total} task(s) FAILED:')
            for line in error_lines:
                ip.LogTextEdit.append(f'  {line}')
        else:
            ip.LogTextEdit.append(f'Job {self._job_queue_index + 1}/{len(self._job_queue)} complete.')
        self._job_queue_index += 1
        self._run_next_queued_job()

    def _on_queued_job_failed(self, message):
        ip = self.ui.IngestionPanel
        ip.LogTextEdit.append(f'Job {self._job_queue_index + 1}/{len(self._job_queue)} FAILED: {message}')
        ip.RunQueuePushButton.setEnabled(True)
        ip.RunIngestionPushButton.setEnabled(True)
        self.statusBar().clearMessage()
        QtWidgets.QMessageBox.critical(self, 'Queued ingestion error', message)

    # -- cell segmentation --

    def _run_cell_segmentation(self):
        cp = self.ui.CellSegmentPanel
        reference_hybe = cp.current_reference_hybe()
        modality = cp.current_reference_modality()
        if not reference_hybe or not modality:
            QtWidgets.QMessageBox.warning(self, 'Run Segmentation', 'Parse a layout and select a reference hybe/channel first.')
            return
        storage_path = self._storage_path_for_modality(modality)
        if not storage_path:
            QtWidgets.QMessageBox.warning(self, 'Run Segmentation', 'Set storage path in the Ingestion tab first.')
            return
        fov = cp.FovSpinBox.value()
        channel_text = cp.ChannelComboBox.currentText()
        if not channel_text:
            QtWidgets.QMessageBox.warning(self, 'Run Segmentation', 'Parse a layout and select a reference hybe/channel first.')
            return
        channel = int(channel_text)
        method = cp.current_method()
        append = cp.AppendModeCheckBox.isChecked()

        if method == 'manual':
            # nothing to compute -- a single small vlinks.h5 read on the GUI
            # thread, not the multi-second Cellpose/watershed compute the
            # other two methods hide behind a QThread for.
            reference_image = vlinks_store.read_hybe_mip(storage_path, fov, reference_hybe, channel)
            if reference_image is None:
                self._on_cell_segment_failed(f'FOV{fov:02d} {reference_hybe} ch{channel} not in vlinks.h5 -- ingest it first.')
                return
            mask = np.zeros(reference_image.shape, dtype=np.uint8)
            cp.LogTextEdit.append(f'Manual mode: opened empty mask for FOV{fov:02d} ({reference_hybe}, ch{channel}).')
            self._on_cell_segment_finished(mask, reference_image, fov, reference_hybe, modality, append=append)
            self.cell_displayer.ManualAddModeCheckBox.setChecked(True)
            return

        cp.RunSegmentationPushButton.setEnabled(False)
        if not self._confirm_projection_choice(cp, 'Run Segmentation'):
            return
        proj_desc = segment.describe_projection(*cp.current_projection())
        cp.LogTextEdit.append(
            f'Segmenting FOV{fov:02d} ({reference_hybe}, ch{channel}, method={method}, {proj_desc})...')
        self.statusBar().showMessage('Segmenting...')

        if method == 'cellpose':
            diameter = cp.DiameterSpinBox.value()
            min_size = cp.MinSizeSpinBox.value()
            max_size = cp.MaxSizeSpinBox.value()
            self._segment_worker = CellSegmentWorker(storage_path, fov, reference_hybe, channel,
                                                     diameter, min_size, max_size,
                                                     projection=cp.current_projection())
        else:
            classical_method = cp.ClassicalAlgorithmComboBox.currentText().lower()
            absolute_cutoff = cp.ClassicalAbsoluteCutoffSpinBox.value()
            min_distance = cp.ClassicalMinDistanceSpinBox.value()
            min_size = cp.ClassicalMinSizeSpinBox.value()
            max_size = cp.ClassicalMaxSizeSpinBox.value()
            self._segment_worker = ClassicalSegmentWorker(storage_path, fov, reference_hybe, channel,
                                                           classical_method, absolute_cutoff, min_distance,
                                                           min_size, max_size,
                                                           projection=cp.current_projection())
        self._segment_worker.finished_ok.connect(
            lambda mask, ref_img: self._on_cell_segment_finished(mask, ref_img, fov, reference_hybe, modality, append=append))
        self._segment_worker.failed.connect(self._on_cell_segment_failed)
        self._segment_worker.start()

    @staticmethod
    def _merge_append_mask(old_mask, new_mask):
        """
        Combine two label masks for cell segmentation's Append Mode -- the
        new segmentation wins in any overlapping region (per explicit
        spec): compute the overlap (new_mask>0), wipe that region out of
        old_mask first, offset every new_mask label by old_mask's own max
        id so the two label sets can't collide, then a plain elementwise
        add merges them (no per-pixel resampling of anything, just integer
        label bookkeeping). Returns uint16, not uint8 -- accumulating
        cells across multiple rounds can easily exceed uint8's 255-label
        cap that fresh (non-append) segmentation stays within; every
        downstream consumer (np.unique/np.where/int(cell_id) label lookup)
        works the same regardless of the mask's integer width.
        """
        old_max = int(old_mask.max())
        wiped_old = old_mask.astype(np.int32).copy()
        wiped_old[new_mask > 0] = 0
        offset_new = np.where(new_mask > 0, new_mask.astype(np.int32) + old_max, 0)
        return (wiped_old + offset_new).astype(np.uint16)

    @staticmethod
    def _filter_small_labels(mask, min_size):
        """
        Drops any label (old or new) whose total pixel count is below
        min_size -- specifically for Append Mode: wiping the overlap
        region out of the old mask (_merge_append_mask) can leave a cell
        that only partially overlapped as a small leftover fragment still
        carrying its old id, a remnant rather than a real cell. Applied to
        the WHOLE combined mask, not just the fresh segmentation's own
        output, since a remnant is a property of the merge, not of either
        original mask alone.
        """
        if min_size <= 0:
            return mask
        ids, counts = np.unique(mask, return_counts=True)
        keep = ids[(ids > 0) & (counts >= min_size)]
        return np.where(np.isin(mask, keep), mask, 0).astype(mask.dtype)

    def _on_cell_segment_finished(self, mask, reference_image, fov, reference_hybe, modality, append=False):
        cp = self.ui.CellSegmentPanel
        if self.cell_container is None:
            self.cell_container = CellContainer([fov])
        self.cell_container.data.setdefault(fov, {})

        # merged (not just `append`) tracks whether `mask`'s own ids are
        # actually trustworthy as "same physical cell as before" -- only
        # true once _merge_append_mask has run: it explicitly keeps an old
        # cell's own id wherever the new segmentation didn't paint over it.
        # Append checked with NOTHING to append to (no prior mask this
        # session) falls through to a bare fresh mask, same as non-append --
        # its ids are just as untrustworthy as a genuinely fresh run's.
        merged = False
        if append and self._last_segment_context is not None and self._last_segment_context['fov'] == fov \
                and self.cell_displayer.mask is not None and self.cell_displayer.mask.shape == mask.shape:
            n_before = int(self.cell_displayer.mask.max())
            mask = self._merge_append_mask(self.cell_displayer.mask, mask)
            merged = True
            method = cp.current_method()
            min_size = cp.MinSizeSpinBox.value() if method == 'cellpose' \
                else cp.ClassicalMinSizeSpinBox.value() if method == 'classical' else 0
            if min_size > 0:
                n_before_filter = int(np.count_nonzero(np.unique(mask)))
                mask = self._filter_small_labels(mask, min_size)
                n_removed = n_before_filter - int(np.count_nonzero(np.unique(mask)))
                if n_removed > 0:
                    cp.LogTextEdit.append(f'Append mode: removed {n_removed} remnant fragment(s) below {min_size}px after merging.')
            cp.LogTextEdit.append(f'Append mode: merged with the existing {n_before} cell(s) already in this FOV.')
        elif append:
            cp.LogTextEdit.append('Append mode: no existing mask for this FOV to append to -- starting fresh.')

        # min/max size already filtered inside segment_fov/segment_fov_classical -- don't re-filter here
        # preserve_existing=merged: a surviving cell id's own reference_hybe/
        # matrices/spots stay exactly as they were (see CellContainer.
        # load_new_cells's own docstring on why this is only safe once a
        # real pixel-preserving merge guarantees id continuity) -- per
        # explicit request, a cell's reference_hybe is fixed at its own
        # definition time, like its own mask coordinates, and never
        # retroactively touched by a later save regardless of mode.
        fp = self._begin_cell_edit(fov)
        self.cell_container.load_new_cells(fov, mask, reference_hybe, preserve_existing=merged,
                                           reference_modality=modality)
        self._commit_cell_edit(fov, fp)
        self._last_segment_context = {'fov': fov, 'reference_hybe': reference_hybe, 'modality': modality}
        n_cells = len(self.cell_container.get_cells(fov))
        cp.LogTextEdit.append(f'Segmentation complete: {n_cells} cell(s) found.')
        cp.RunSegmentationPushButton.setEnabled(True)
        self.statusBar().showMessage('Segmentation complete.', 5000)
        self.cell_displayer.set_data(reference_image, mask)
        QtWidgets.QMessageBox.information(self, 'Segmentation complete', f'{n_cells} cell(s) found for FOV{fov:02d}.')

    def _on_cell_segment_failed(self, message):
        cp = self.ui.CellSegmentPanel
        cp.LogTextEdit.append(f'Segmentation FAILED: {message}')
        cp.RunSegmentationPushButton.setEnabled(True)
        self.statusBar().clearMessage()
        QtWidgets.QMessageBox.critical(self, 'Segmentation error', message)

    def _refresh_cell_displayer_view(self):
        """
        Re-reads whatever FOV/reference hybe/channel Cell Segmentation now
        points at and repaints the Cell Displayer, but ONLY while it is
        actually showing a segmentation view -- a cytoplasm result staged
        in the same window must not be silently replaced just because the
        user scrolled the FOV spinbox.
        """
        if self._cell_displayer_mode != 'segmentation':
            return
        if not self.cell_displayer.isVisible():
            return
        self._ensure_cell_displayer_initialized()

    def _toggle_cell_displayer(self, checked):
        if checked:
            self._cell_displayer_mode = 'segmentation'
            self._ensure_cell_displayer_initialized()
            self.cell_displayer.show()
            self.cell_displayer.raise_()
        else:
            self.cell_displayer.hide()

    # -- cytoplasmic segmentation --

    def _cytoplasm_cells(self, fov):
        """
        The TRANSIENT container's cells for this FOV -- the ones every
        displayer shows and the ones Save promotes to permanent.

        Transient first, per explicit correction and confirmed real bug:
        this used to prefer the PERMANENT container, so Incorporate wrote
        cytoplasms onto permanent while _save_cells copies transient ->
        permanent and writes THAT -- overwriting the cytoplasms with the
        untouched cells, which is exactly why Save appeared to do nothing.
        Falling back to permanent only when nothing is staged keeps the
        read-only paths (preview, cell list) working before segmentation.
        """
        for container in (self.cell_container, self.cell_container_permanent):
            if container is not None and container.data.get(fov):
                return container, container.get_cells(fov)
        return None, []

    def _cells_for_fov(self, fov):
        """Just the cells -- see _cytoplasm_cells for which container wins."""
        return self._cytoplasm_cells(fov)[1]

    def _show_cytoplasm_window(self):
        cw = self.cytoplasm_window
        cw.FovSpinBox.setValue(self.ui.CellSegmentPanel.FovSpinBox.value())
        cw.set_hybe_choices(self.total_active_hybe_list)
        self._on_cytoplasm_hybe_changed()
        self._refresh_cytoplasm_cell_list()
        cw.show()
        cw.raise_()

    def _on_cytoplasm_hybe_changed(self):
        cw = self.cytoplasm_window
        record = cw.current_hybe_record()
        cw.set_channel_choices(record['channels'] if record else [])
        self._refresh_cytoplasm_depth()

    def _refresh_cytoplasm_depth(self):
        cw = self.cytoplasm_window
        hybe, modality = cw.current_hybe_key()
        channel = cw.current_channel()
        storage_path = self._storage_path_for_modality(modality) if modality else None
        depth = 0
        if storage_path and hybe and channel is not None:
            try:
                depth = segment.stack_depth(storage_path, cw.FovSpinBox.value(), hybe, channel)
            except OSError:
                depth = 0
        cw.set_depth(depth)

    def _autodetect_segmentation_focus(self):
        """Detect Focal Plane for the PRIMARY segmentation panel -- same metric and
        same button-only policy as the cytoplasm one, just a different panel's
        hybe/channel/FOV and z controls."""
        cp = self.ui.CellSegmentPanel
        hybe, modality = cp.current_reference_hybe(), cp.current_reference_modality()
        channel = int(cp.ChannelComboBox.currentText()) if cp.ChannelComboBox.currentText() else None
        storage_path = self._storage_path_for_modality(modality) if modality else None
        if not (storage_path and hybe and channel is not None):
            QtWidgets.QMessageBox.warning(self, 'Cell Segmentation',
                                          'Pick a reference hybe and channel first.')
            return
        self._apply_focus_detection(cp, storage_path, cp.FovSpinBox.value(), hybe, channel,
                                    cp.LogTextEdit.append)

    def _apply_focus_detection(self, panel, storage_path, fov, hybe, channel, log):
        """Shared by both panels' Detect Focal Plane buttons -- one metric, one
        plateau rule, one place to fix. Seeds the RANGE with the >=90%-of-peak
        plateau as well as the single peak, since focus varies across the field
        while the metric only samples the centre."""
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            zs, values = segment.focus_profile(storage_path, fov, hybe, channel)
        except (OSError, KeyError) as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.warning(self, 'Detect Focal Plane',
                                          f"Can't read the raw stack for {hybe} ch{channel}: {exc}")
            return None
        QtWidgets.QApplication.restoreOverrideCursor()
        peak = int(zs[int(np.argmax(values))])
        plateau = zs[values >= 0.9 * values.max()]
        panel.set_depth(int(zs.max()) + 1)
        panel.ZPlaneSpinBox.setValue(peak)
        panel.ZStartSpinBox.setValue(int(plateau.min()))
        panel.ZEndSpinBox.setValue(int(plateau.max()))
        panel.show_focus_profile(zs, values, peak)
        # Detecting a focal plane and then not using it was the trap the Run
        # confirmation existed to catch; switching here removes the trap at
        # its source. The confirmation stays for the case where the mode is
        # later put BACK to the stored MIP by hand.
        panel.ProjectionModeComboBox.setCurrentText('single plane')
        middle = int(zs.max()) // 2
        log(f'Focal plane z={peak} (middle would have been z={middle}, '
            f'off by {peak - middle:+d}); >=90% plateau z={int(plateau.min())}-{int(plateau.max())}. '
            f'Projection switched to single plane z={peak}.')
        return peak

    def _on_segmentation_plane_changed(self):
        """Live view while scrolling Plane z -- single-plane only. A single
        slice is one cheap h5py read, unlike a range (see ViewRangePushButton)."""
        cp = self.ui.CellSegmentPanel
        cp.refresh_run_label()
        if cp.ProjectionModeComboBox.currentText() == 'single plane':
            self._refresh_cell_displayer_view()

    def _on_cytoplasm_plane_changed(self):
        cw = self.cytoplasm_window
        cw.refresh_run_label()
        if cw.ProjectionModeComboBox.currentText() == 'single plane':
            self._refresh_cytoplasm_view()

    def _refresh_cytoplasm_view(self):
        """Repaint the shared displayer with the cytoplasm panel's CURRENT
        projection, keeping the selected-nuclei overlay if there is one."""
        ctx = self._cytoplasm_context()
        if ctx is None:
            return
        fov, hybe, modality, channel, _, image = ctx
        cw = self.cytoplasm_window
        chosen = set(cw.selected_cell_ids())
        cells = [c for c in self._cells_for_fov(fov) if c.id in chosen]
        label_mask = (self._build_nucleus_label_mask(cells, hybe, modality, fov, image.shape)
                      if cells else np.zeros(image.shape, dtype=np.int32))
        self._cell_displayer_mode = 'cytoplasm'
        self.cell_displayer.setWindowTitle(
            f'Cell Displayer -- FOV{fov:02d} {hybe} ch{channel} ({modality}) '
            f'[{segment.describe_projection(*cw.current_projection())}] -- selected nuclei')
        self.cell_displayer.set_data(image, label_mask.astype(float))
        self.cell_displayer.show()
        self.cell_displayer.raise_()

    def _confirm_projection_choice(self, panel, title):
        """
        True to proceed. Blocks ONLY the genuinely ambiguous case: a focal
        plane was detected this session, yet the run is still about to use
        the stored MIP -- so every z control on screen is populated and
        ignored.

        Not cosmetic caution. Measured on real data, same FOV/hybe/channel/
        parameters: stored MIP found 33 cells, single plane z=76 found 91.
        Silently doing the 33-cell thing because a combobox was left alone
        is the failure this exists to catch, and it fires only in that exact
        combination so it cannot become click-through noise.
        """
        mode, z_plane, z_range = panel.current_projection()
        if not (getattr(panel, 'focus_detected', False) and mode == 'MIP (stored)'):
            return True
        reply = QtWidgets.QMessageBox.question(
            self, title,
            f'A focal plane was detected (z={z_plane}, plateau z={z_range[0]}-{z_range[1]}), '
            f'but Projection is still "MIP (stored)" -- the run will use the stored '
            f'full-depth MIP and ignore those values.\n\n'
            f'On real data the stored MIP found substantially fewer cells than the '
            f'focal plane did.\n\nRun with the stored MIP anyway?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        return reply == QtWidgets.QMessageBox.Yes

    def _refresh_segmentation_depth(self):
        cp = self.ui.CellSegmentPanel
        hybe, modality = cp.current_reference_hybe(), cp.current_reference_modality()
        channel = int(cp.ChannelComboBox.currentText()) if cp.ChannelComboBox.currentText() else None
        storage_path = self._storage_path_for_modality(modality) if modality else None
        depth = 0
        if storage_path and hybe and channel is not None:
            try:
                depth = segment.stack_depth(storage_path, cp.FovSpinBox.value(), hybe, channel)
            except OSError:
                depth = 0
        cp.set_depth(depth)

    def _autodetect_cytoplasm_focus(self):
        """
        Finds the sharpest plane and points every z control at it. Explicitly
        button-driven (per explicit preference): the metric reads every plane,
        so it should never fire as a side effect of switching FOV.

        Also seeds the RANGE with the plateau at >=90% of peak sharpness, not
        just the single peak -- focus varies across the field while this
        metric only samples the centre, so a range projection over the
        plateau is generally the safer default.
        """
        cw = self.cytoplasm_window
        hybe, modality = cw.current_hybe_key()
        channel = cw.current_channel()
        storage_path = self._storage_path_for_modality(modality) if modality else None
        if not (storage_path and hybe and channel is not None):
            QtWidgets.QMessageBox.warning(self, 'Cytoplasmic Segmentation',
                                          'Pick a cytoplasm hybe and channel first.')
            return
        self._apply_focus_detection(cw, storage_path, cw.FovSpinBox.value(), hybe, channel, cw.log)

    def _refresh_cytoplasm_cell_list(self):
        cw = self.cytoplasm_window
        fov = cw.FovSpinBox.value()
        _, cells = self._cytoplasm_cells(fov)
        rows = []
        for cell in sorted(cells, key=lambda c: c.id):
            n_nucleus = len(cell.nucleus[0])
            suffix = ' [has cytoplasm]' if cell.has_cytoplasm() else ''
            rows.append((cell.id,
                         f'Cell {cell.id:03d} -- nucleus {n_nucleus}px @ '
                         f'{cell.nucleus_hybe} ({cell.nucleus_modality}){suffix}'))
        cw.set_cell_choices(rows)
        if not rows:
            cw.log(f'FOV{fov:02d}: no cells loaded -- segment nuclei first.')

    def _cell_nucleus_in_readout(self, cell, hybe, modality, fov):
        """
        (y, x) -- this cell's NUCLEUS projected into `hybe`'s own native
        frame, using the live FOV/cross-modal matrices rather than a bare
        cell-level lookup, for exactly the reason _cell_area_in_readout
        exists: the bare model path collapses to identity whenever this
        cell has no cell-level entry for the pair, which silently
        mispositions the projection.

        Anchors on the cell's OWN nucleus_hybe/nucleus_modality, which can
        differ cell to cell within one FOV -- that is what makes the
        rendered seed image a genuine stitch rather than one shared warp.
        """
        H = self._matrix_to_frame(hybe, modality, cell, fov,
                                  cell.nucleus_hybe, cell.nucleus_modality)
        if H is None:
            return np.array([]), np.array([])
        y_lit, x_lit = cell.nucleus
        cy, cx = alignment.align_cell((y_lit, x_lit), la.inv(H), cell.frame_shape)
        return cy, cx

    def _build_nucleus_label_mask(self, cells, hybe, modality, fov, shape):
        """
        Every given cell's nucleus painted with its own REAL cell id, all
        projected into (hybe, modality)'s frame. Used both for the seed
        image and, with the full cell list, as incorporate_cytoplasm's own
        overlap authority -- so an unselected cell can still win its pixels
        back from a cytoplasm that overlapped it.

        Cells are painted largest-first so that where two projected nuclei
        collide, the SMALLER one still ends up visible rather than being
        buried -- a collision here is a projection artifact, and silently
        losing a whole small cell is the worse failure.
        """
        label_mask = np.zeros(shape, dtype=np.int32)
        painted = []
        for cell in cells:
            y, x = self._cell_nucleus_in_readout(cell, hybe, modality, fov)
            if len(x) == 0:
                continue
            painted.append((len(x), cell.id, x, y))
        for _, cell_id, x, y in sorted(painted, reverse=True):
            ix, iy = x.astype(int), y.astype(int)
            valid = (iy >= 0) & (iy < shape[0]) & (ix >= 0) & (ix < shape[1])
            label_mask[iy[valid], ix[valid]] = cell_id
        return label_mask

    def _cytoplasm_context(self):
        """(fov, hybe, modality, channel, storage_path, image) or None, with the user told why."""
        cw = self.cytoplasm_window
        fov = cw.FovSpinBox.value()
        hybe, modality = cw.current_hybe_key()
        channel = cw.current_channel()
        if not hybe or modality is None or channel is None:
            QtWidgets.QMessageBox.warning(self, 'Cytoplasmic Segmentation',
                                          'Pick a cytoplasm hybe and channel first.')
            return None
        storage_path = self._storage_path_for_modality(modality)
        mode, z_plane, z_range = cw.current_projection()
        try:
            image = (segment.read_projection(storage_path, fov, hybe, channel, mode=mode,
                                             z_plane=z_plane, z_range=z_range)
                     if storage_path else None)
        except (OSError, KeyError) as exc:
            QtWidgets.QMessageBox.warning(
                self, 'Cytoplasmic Segmentation',
                f"Can't read {hybe} ch{channel} ({modality}) for FOV{fov:02d} at '{mode}': {exc}")
            return None
        if image is None:
            QtWidgets.QMessageBox.warning(
                self, 'Cytoplasmic Segmentation',
                f'FOV{fov:02d} {hybe} ch{channel} ({modality}) is not in vlinks.h5 -- ingest it first.')
            return None
        return fov, hybe, modality, channel, storage_path, image

    def _preview_cytoplasm_nuclei(self):
        ctx = self._cytoplasm_context()
        if ctx is None:
            return
        fov, hybe, modality, channel, _, image = ctx
        cw = self.cytoplasm_window
        _, cells = self._cytoplasm_cells(fov)
        chosen = set(cw.selected_cell_ids())
        selected = [c for c in cells if c.id in chosen]
        if not selected:
            QtWidgets.QMessageBox.warning(self, 'Cytoplasmic Segmentation',
                                          'No cells selected as nuclei.')
            return
        label_mask = self._build_nucleus_label_mask(selected, hybe, modality, fov, image.shape)
        n_drawn = len(np.unique(label_mask)) - 1
        self._cell_displayer_mode = 'cytoplasm'
        self.cell_displayer.setWindowTitle(
            f'Cytoplasm Displayer -- FOV{fov:02d} {hybe} ch{channel} ({modality}) -- selected nuclei')
        self.cell_displayer.set_data(image, label_mask.astype(float))
        self.cell_displayer.show()
        self.cell_displayer.raise_()
        cw.log(f'Preview: {n_drawn}/{len(selected)} selected nucleus/nuclei project into '
               f'{hybe} ({modality}).')
        if n_drawn < len(selected):
            cw.log('  (some nuclei fall outside this hybe\'s frame entirely -- check alignment)')

    def _run_cytoplasm_segmentation(self):
        ctx = self._cytoplasm_context()
        if ctx is None:
            return
        fov, hybe, modality, channel, _, image = ctx
        cw = self.cytoplasm_window
        _, cells = self._cytoplasm_cells(fov)
        chosen = set(cw.selected_cell_ids())
        selected = [c for c in cells if c.id in chosen]
        if not selected:
            QtWidgets.QMessageBox.warning(self, 'Cytoplasmic Segmentation',
                                          'No cells selected as nuclei.')
            return

        if not self._confirm_projection_choice(cw, 'Run Cytoplasmic Search'):
            return
        seed_labels = self._build_nucleus_label_mask(selected, hybe, modality, fov, image.shape)
        seed_image = segment.render_nucleus_seed(seed_labels, mode=cw.current_seed_mode())
        dilation = cw.NucleusDilationSpinBox.value()
        if dilation:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation + 1, 2 * dilation + 1))
            seed_image = cv2.dilate(seed_image, kernel)
        # Scaled to the cytoplasm image's own dynamic range: cellpose
        # normalizes per channel, and a 0/1 channel beside a real 16-bit
        # one is not something its own preprocessing expects.
        seed_image = seed_image * float(np.percentile(image, 99.9))

        diameter = cw.DiameterSpinBox.value() or None
        proj_desc = segment.describe_projection(*cw.current_projection())
        cw.log(f'Running cellpose on {hybe} ch{channel} ({modality}) [{proj_desc}], '
               f'{len(selected)} nucleus seed(s), seed style={cw.current_seed_mode()}, '
               f'diameter={diameter or "auto"}...')
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            cyto_labels = segment.segment_cytoplasm(
                image, seed_image, diameter=diameter,
                min_size=cw.MinSizeSpinBox.value(), max_size=cw.MaxSizeSpinBox.value())
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, 'Cytoplasmic Segmentation', str(exc))
            cw.log(f'FAILED: {exc}')
            return
        QtWidgets.QApplication.restoreOverrideCursor()

        # Selection captured HERE, at run time -- incorporating later must
        # honour what was actually seeded, not whatever the checkboxes say
        # by then.
        self._cytoplasm_result = {'labels': cyto_labels, 'fov': fov, 'hybe': hybe,
                                  'modality': modality, 'channel': channel, 'image': image,
                                  'selected_ids': sorted(chosen)}
        cw.IncorporatePushButton.setEnabled(True)
        n = len(np.unique(cyto_labels)) - 1
        cw.log(f'Cellpose returned {n} cytoplasm label(s). Review/remove in the displayer, '
               f'then Incorporate.')
        self._cell_displayer_mode = 'cytoplasm'
        self.cell_displayer.setWindowTitle(
            f'Cytoplasm Displayer -- FOV{fov:02d} {hybe} ch{channel} ({modality}) -- raw cytoplasm')
        self.cell_displayer.set_data(image, cyto_labels.astype(float))
        self.cell_displayer.show()
        self.cell_displayer.raise_()

    def _incorporate_cytoplasm(self):
        result = self._cytoplasm_result
        if result is None:
            return
        fov, hybe, modality = result['fov'], result['hybe'], result['modality']
        cw = self.cytoplasm_window
        container, cells = self._cytoplasm_cells(fov)
        if not cells:
            QtWidgets.QMessageBox.warning(self, 'Cytoplasmic Segmentation',
                                          'No cells loaded for this FOV any more.')
            return
        # Cytoplasms are a STAGED edit: they belong in the transient
        # container so Save promotes them, and so every displayer shows
        # them before saving. If only permanent cells exist, stage a copy
        # rather than mutating what is already on disk.
        if container is not self.cell_container:
            if self.cell_container is None:
                self.cell_container = CellContainer([fov])
            self.cell_container.data[fov] = {int(c.id): c for c in deepcopy(cells)}
            cells = self.cell_container.get_cells(fov)

        # EVERY cell's nucleus, not just the selected ones -- an unselected
        # cell still has to win its own pixels back from any cytoplasm that
        # overlapped it (see incorporate_cytoplasm's own docstring).
        nucleus_labels = self._build_nucleus_label_mask(cells, hybe, modality, fov,
                                                        result['image'].shape)
        merged, claimed = segment.incorporate_cytoplasm(
            result['labels'], nucleus_labels, eligible_ids=result.get('selected_ids'))

        n_updated = 0
        for cell in cells:
            if cell.id not in claimed:
                continue  # keeps its own nucleus, reference hybe and frame untouched
            y, x = np.where(merged == cell.id)
            if len(x) == 0:
                continue
            cell.set_metadata(area=(y, x), reference_hybe=hybe, reference_modality=modality,
                              frame_shape=merged.shape)
            n_updated += 1

        skipped = [c.id for c in cells if c.id not in claimed]
        cw.log(f'Incorporated {n_updated} cytoplasm(s) at {hybe} ({modality}); '
               f'{len(skipped)} cell(s) left nucleus-only.')
        if skipped:
            cw.log(f'  nucleus-only: {", ".join(str(i) for i in sorted(skipped)[:20])}'
                   + (' ...' if len(skipped) > 20 else ''))
        cw.log('Cell ids and count are unchanged. Alignment matrices were NOT recomputed -- '
               're-run Cell-Based Alignment if you want the residual refit against the cytoplasm.')
        # Review is OVER: the merge just landed in the transient container,
        # which is the one authority from here on (per explicit principle:
        # the raw cytoplasm labels are super-temporal, never stored;
        # post-integration state lives in the container). Consume the
        # staged result, return the displayer to segmentation mode, and
        # render FROM the container via THE one renderer. The old tail
        # kept showing the merged raster in 'cytoplasm' mode -- the
        # display stayed sourced from a dead intermediate, and a
        # subsequent Remove was routed into the cytoplasm-labels branch
        # where the container never heard of it (confirmed real bug:
        # cell removed from the store, contour still on screen).
        self._cytoplasm_result = None
        self._cell_displayer_mode = 'segmentation'
        self.cell_displayer.setWindowTitle(
            f'Cell Displayer -- FOV{fov:02d} {hybe} ({modality}) -- cytoplasm incorporated')
        self._render_cell_displayer(fov, self.cell_container.get_cells(fov), hybe, modality, result['image'])
        self._refresh_cytoplasm_cell_list()
        QtWidgets.QMessageBox.information(
            self, 'Cytoplasmic Segmentation',
            f'{n_updated} cell(s) now carry a cytoplasm; {len(skipped)} kept nucleus-only.\n\n'
            f'Not yet written to vlinks.h5 -- use Cell Segmentation\'s Save.')

    def _ensure_cell_displayer_initialized(self):
        """
        Every toggle-on re-reads FOV/reference hybe/channel LIVE from the
        Cell Segmentation panel and reloads the displayer to match --
        CellDisplayer has no FOV/hybe/channel controls of its own, so per
        explicit request the panel always drives what's shown, not
        whatever the displayer happened to be showing before ("toggle
        just hides/shows the image we already have" was exactly the
        complaint). This replaced an earlier version that delegated to
        _try_show_existing_cells(fov), which picks the displayed hybe from
        whatever hybe the PERSISTED cells happen to be segmented in --
        completely ignoring the panel's current hybe selection whenever
        they differ, which was the real bug.

        Existing cells always show on the right panel, whatever hybe/
        channel the panel is currently displaying -- see the mask-
        building block below for how positions get transformed (or, at
        worst, shown raw/approximate) so the mask is never just hidden.
        """
        cp = self.ui.CellSegmentPanel
        fov = cp.FovSpinBox.value()
        reference_hybe = cp.current_reference_hybe()
        modality = cp.current_reference_modality()
        storage_path = self._storage_path_for_modality(modality)
        channel_text = cp.ChannelComboBox.currentText()
        if not storage_path or not reference_hybe or not channel_text:
            return
        channel = int(channel_text)
        # Whatever projection the panel is currently set to -- so what you
        # review is what Run will actually segment, rather than the stored
        # MIP always standing in for it.
        proj_mode, proj_plane, proj_range = cp.current_projection()
        try:
            reference_image = segment.read_projection(storage_path, fov, reference_hybe, channel,
                                                      mode=proj_mode, z_plane=proj_plane, z_range=proj_range)
        except (OSError, KeyError) as exc:
            cp.LogTextEdit.append(f"Can't read {reference_hybe} ch{channel} for FOV{fov:02d}: {exc}")
            return
        if reference_image is None:
            cp.LogTextEdit.append(f'{reference_hybe} ch{channel} not in vlinks.h5 for FOV{fov:02d} -- ingest it first.')
            return

        # activate whatever's persisted for this FOV (disk or already in
        # memory), same source _try_show_existing_cells uses -- but only
        # to CHECK whether it matches the panel's current hybe, never to
        # override the panel's own fov/hybe/channel choice.
        have_in_memory = (self.cell_container_permanent is not None and fov in self.cell_container_permanent.data
                          and self.cell_container_permanent.data[fov])
        if not have_in_memory:
            cell_dicts, modality = vlinks_store.read_cells(storage_path, fov)
            modality = modality or self._modality_for_storage_path(storage_path)
            if cell_dicts:
                loaded = CellContainer.load({fov: cell_dicts}, modality=modality)
                if self.cell_container_permanent is None:
                    self.cell_container_permanent = loaded
                else:
                    self.cell_container_permanent.data[fov] = loaded.data[fov]
                    if fov not in self.cell_container_permanent.fov_list:
                        self.cell_container_permanent.fov_list.append(fov)

        cells = None
        if self.cell_container_permanent is not None and self.cell_container_permanent.data.get(fov):
            cells = self.cell_container_permanent.get_cells(fov)

        if self.cell_container is None:
            self.cell_container = CellContainer([fov])
        self.cell_container.data.setdefault(fov, {})

        # The TRANSIENT tier is authoritative in-session: seed it from the
        # permanent cells only when it holds NOTHING for this FOV yet.
        # The old unconditional overwrite here (transient = deepcopy of
        # permanent on every displayer reload) silently discarded every
        # in-session container edit -- confirmed real bug: remove a cell,
        # toggle the displayer, and the cell was resurrected in BOTH the
        # view and the transient container. Same hazard _try_show_
        # existing_cells' own once-per-FOV gate already documents.
        if cells and not self.cell_container.data.get(fov):
            self.cell_container.data[fov] = {int(c.id): c for c in deepcopy(cells)}

        display_cells = self.cell_container.get_cells(fov)
        self._render_cell_displayer(fov, display_cells, reference_hybe, modality, reference_image)
        cp.LogTextEdit.append(f'Displayer showing FOV{fov:02d} ({reference_hybe}, ch{channel}) -- {len(display_cells)} cell(s).')

    def _render_cell_displayer(self, fov, cells, reference_hybe, modality, reference_image):
        """
        THE one renderer for the segmentation displayer -- every path that
        needs the mask redrawn (initial load, post-removal refresh, undo/
        redo) builds it here, from the TRANSIENT container's cells, via
        _cell_area_in_readout (the resolver-backed projection, NEVER
        ACell.get_area_in_readout directly: that raises by design on
        residual-form matrices once cell alignment has run -- confirmed
        real crash). A second, divergent renderer is exactly how a stale
        contour survives one path and not another. An empty `cells` list
        renders an empty mask (removing the last cell must clear the
        view, never skip the redraw). int32 mask: ids above 255 silently
        wrapped in the old uint8 raster.
        """
        cp = self.ui.CellSegmentPanel
        mask = np.zeros(reference_image.shape, dtype=np.int32)
        approximate = False
        height, width = mask.shape
        for cell in cells:
            # reference_hybe's own modality (from the panel selection, see
            # current_reference_modality) -- NOT necessarily this cell's
            # own segmentation modality; cells can come from either.
            if (reference_hybe, modality) not in cell.matrices:
                approximate = True
            y, x = self._cell_area_in_readout(cell, reference_hybe, modality, fov)
            if len(x) == 0:
                continue
            xi, yi = x.astype(int), y.astype(int)
            valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
            mask[yi[valid], xi[valid]] = cell.id
        if cells and approximate:
            cp.LogTextEdit.append(f'Note: no alignment matrix for {reference_hybe} yet -- cell positions shown here '
                                  f'are raw/untransformed (approximate) until cell-based alignment is run for this hybe.')
        self._last_segment_context = {'fov': fov, 'reference_hybe': reference_hybe, 'modality': modality}
        self.cell_displayer.set_data(reference_image, mask)

    def _on_displayer_ids_removed(self, ids):
        """
        Remove-by-ID from the CellDisplayer, as the ID-LIST operation it
        is -- straight into the container (the authority), then a fresh
        re-render FROM the container, which is what makes a stale contour
        structurally impossible (see _refresh_cell_displayer_from_
        container). Never routed through the raster: rebuilding cells
        from the DISPLAYED mask redefines their geometry from a
        projection whenever the display frame differs from a cell's
        native frame (exactly the post-cytoplasm case), and the old
        raster path also silently no-oped without a fresh segmentation
        context, leaving the removed cell alive to resurrect (confirmed
        real bug: 'contour remains after removing a cell right after
        incorporating cytoplasmic masks').
        """
        if self._cell_displayer_mode == 'cytoplasm':
            # Cytoplasm review: the staged label raster IS the authority
            # there; the displayer already zeroed its copy -- mirror that
            # into the staged result, exactly as the old path did.
            if self._cytoplasm_result is not None:
                labels = np.asarray(self._cytoplasm_result['labels']).astype(np.int32).copy()
                labels[np.isin(labels, ids)] = 0
                self._cytoplasm_result['labels'] = labels
            return
        if self.cell_container is None:
            return
        fov = (self._last_segment_context or {}).get('fov',
                                                     self.ui.CellSegmentPanel.FovSpinBox.value())
        fp = self._begin_cell_edit(fov)
        removed = self.cell_container.remove(fov, ids)
        self._commit_cell_edit(fov, fp)
        self._refresh_cell_displayer_from_container(fov)
        self.ui.CellSegmentPanel.LogTextEdit.append(
            f'Removed {len(removed)} cell(s) ({", ".join(str(c.id) for c in removed) or "none found"}) -- '
            f'{len(self.cell_container.get_cells(fov))} remain. Undo restores.')

    def _on_displayer_mask_edited(self, mask):
        """
        Routed by self._cell_displayer_mode, since one CellDisplayer now
        serves both flows: a cytoplasm-review edit must NOT be pushed
        through load_new_cells (that would rewrite the cell container from
        a cytoplasm label image), and a segmentation edit must not be
        mistaken for a staged cytoplasm result.

        A manual add/remove in CellDisplayer -- `mask` is always an edit of
        the mask the user is already looking at (one label added or one
        removed), never a fresh independent clustering, so every id it
        still carries genuinely is the same physical cell as before --
        preserve_existing=True keeps that cell's own reference_hybe/
        matrices/spots intact (see CellContainer.load_new_cells's own
        docstring); only a newly-clicked cell (a genuinely new id) starts
        blank, at this call's own reference_hybe.
        """
        if self._cell_displayer_mode == 'cytoplasm':
            if self._cytoplasm_result is not None:
                self._cytoplasm_result['labels'] = np.asarray(mask).astype(np.int32)
            return
        if self._last_segment_context is None or self.cell_container is None:
            return
        fov = self._last_segment_context['fov']
        reference_hybe = self._last_segment_context['reference_hybe']
        fp = self._begin_cell_edit(fov)
        self.cell_container.load_new_cells(
            fov, mask, reference_hybe, preserve_existing=True,
            reference_modality=self._last_segment_context.get('modality'))
        self._commit_cell_edit(fov, fp)
        n_cells = len(self.cell_container.get_cells(fov))
        self.ui.CellSegmentPanel.LogTextEdit.append(f'Mask edited in displayer: {n_cells} cell(s) remain.')

    def _all_vlinks_storage_paths(self):
        """
        Every experiment storage path this session currently knows about --
        the Ingestion tab's own path plus, if set, the Cross-Modal
        Alignment layer's RNA/DNA pair. A cell's spots span whichever
        hybes actually got localized for it, which can straddle both an
        RNA and a DNA experiment (that's the whole point of cross-modal
        alignment), so a saved cell's full state -- cells, spots, and
        per-cell alignment matrices, all riding along inside
        CellContainer.save() -- belongs in every one of these vlinks.h5
        files, not just whichever one happens to be the "current" tab.
        """
        ip = self.ui.IngestionPanel
        ap = self.ui.AlignmentPanel
        paths = []
        for p in (ip.StoragePathLineEdit.text().strip(),
                  ap.RnaStoragePathLineEdit.text().strip(),
                  ap.DnaStoragePathLineEdit.text().strip()):
            if p and p not in paths:
                paths.append(p)
        return paths

    def _mirror_cross_modal_params_to_vlinks(self, rna_storage_path, dna_storage_path, fov, H,
                                             rna_reference_hybe, dna_reference_hybe, channel_type):
        """
        Mirrors an accepted cross-modal alignment result into BOTH
        modalities' own vlinks.h5 -- global params (both reference hybes,
        channel type, and each side's OWN paired storage path + role) plus
        the per-FOV H_across matrix -- so either modality's own FOV
        activation can reconstruct the full cross-modal picture without
        ever needing the OTHER side's config loaded first (see
        _other_modality_cell_alignment_inputs, which today can only find
        any of this via manually-populated UI fields).
        """
        # One write, not one per side. These describe the RELATIONSHIP
        # between the two modalities, so a single copy is the correct
        # representation -- and with a unified vlinks both storage paths
        # resolve to the same /params group, so writing twice with opposite
        # cross_modal_role values would leave only whichever went last.
        # The sides are named by modality rather than by a per-file role,
        # and the paired storage path is gone: there is one file now.
        vlinks_store.write_global_params(
            rna_storage_path,
            cross_modal_rna_modality=self._modality_for_storage_path(rna_storage_path),
            cross_modal_dna_modality=self._modality_for_storage_path(dna_storage_path),
            cross_modal_rna_reference_hybe=rna_reference_hybe,
            cross_modal_dna_reference_hybe=dna_reference_hybe,
            cross_modal_channel_type=channel_type)
        vlinks_store.write_cross_modal_matrix(rna_storage_path, fov, H)
        vlinks_store.write_cross_modal_matrix(dna_storage_path, fov, H)

    def _save_cells(self):
        cp = self.ui.CellSegmentPanel
        if self.cell_container is None or self._last_segment_context is None:
            QtWidgets.QMessageBox.warning(self, 'Save Cells', 'Run segmentation first.')
            return
        # The FOV actually staged in the transient container -- not
        # _last_segment_context's, which only tracks the last SEGMENTATION
        # run and goes stale after a cytoplasm incorporate on another FOV.
        fov = self._last_segment_context['fov']
        staged = [f for f, cells in self.cell_container.data.items() if cells]
        if staged and fov not in staged:
            fov = staged[0]
        already_saved = (self.cell_container_permanent is not None
                          and fov in self.cell_container_permanent.data
                          and len(self.cell_container_permanent.data[fov]) > 0)
        if already_saved:
            reply = QtWidgets.QMessageBox.question(
                self, 'Save cells', f'This will overwrite the saved cells for FOV{fov:02d}.\nAre you sure?',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if reply != QtWidgets.QMessageBox.Yes:
                return
        if self.cell_container_permanent is None:
            self.cell_container_permanent = CellContainer([fov])
        self.cell_container_permanent.sync_from(self.cell_container, fov)
        storage_paths = self._all_vlinks_storage_paths()
        if storage_paths:
            # Cells changed -> spot ownership may have changed everywhere in
            # this FOV. Reassign every spot and persist all slices now, so no
            # spot is left pointing at a removed or reshaped cell.
            self._recast_persisted_spots(fov)
            vlinks_store.mirror_write_cells(storage_paths, fov, self.cell_container_permanent)
            # No segmentation_reference_hybe param write: each cell already
            # carries its own reference_hybe/nucleus_hybe (with modalities),
            # cells in one FOV can legitimately disagree, and nothing ever
            # read the FOV-level copy back.
            where = ', '.join(storage_paths)
            cp.LogTextEdit.append(f'Saved {len(self.cell_container.get_cells(fov))} cell(s) for FOV{fov:02d} to permanent '
                                  f'container and vlinks.h5 ({where}).')
        else:
            cp.LogTextEdit.append(f'Saved {len(self.cell_container.get_cells(fov))} cell(s) for FOV{fov:02d} to permanent '
                                  f'container (no storage path set -- not written to vlinks.h5).')

    def _discard_cells(self):
        cp = self.ui.CellSegmentPanel
        if self.cell_container is None or self._last_segment_context is None:
            return
        fov = self._last_segment_context['fov']
        self.cell_container.data[fov] = {}
        if self.cell_displayer.reference_image is not None:
            empty_mask = np.zeros(self.cell_displayer.reference_image.shape, dtype=np.uint8)
            self.cell_displayer.set_data(self.cell_displayer.reference_image, empty_mask)
        cp.LogTextEdit.append(f'Discarded transient cells for FOV{fov:02d}.')

    def _send_permanent_cells_to_transient(self):
        cp = self.ui.CellSegmentPanel
        if self.cell_container_permanent is None or self._last_segment_context is None:
            QtWidgets.QMessageBox.warning(self, 'Send Permanent to Transient', 'No saved cells yet.')
            return
        fov = self._last_segment_context['fov']
        if fov not in self.cell_container_permanent.data or len(self.cell_container_permanent.data[fov]) == 0:
            QtWidgets.QMessageBox.warning(self, 'Send Permanent to Transient', f'No saved cells for FOV{fov:02d}.')
            return
        if self.cell_container is None:
            self.cell_container = CellContainer([fov])
        self.cell_container.sync_from(self.cell_container_permanent, fov)
        cp.LogTextEdit.append(f'Pulled {len(self.cell_container.get_cells(fov))} cell(s) from permanent for FOV{fov:02d}.')

    def _activate_fov(self, fov):
        """
        "Activation": whatever's already been computed and persisted for
        this FOV -- segmented cells (with their spots and per-cell
        alignment matrices, all riding along inside CellContainer.save()),
        plus FOV-level alignment matrices -- gets pulled into the running
        transient/runtime state automatically the moment this FOV becomes
        current, with no button and no re-computation required. Fires on
        every Cell Segmentation FOV switch (cp.FovSpinBox.valueChanged),
        right after ingestion/parsing complete, AND from Spot
        Localization's own _refresh_spot_cell_list (which calls this
        directly rather than relying on Cell Segmentation's spinbox
        having already been touched for the same FOV -- cp.FovSpinBox and
        sp.FovSpinBox are two separate widgets, only their range is kept
        in sync, not their value; see that method's own comment), per the
        explicit principle that already-persisted state should never
        require redoing the work that produced it.

        Loops over EVERY configured modality's own storage_path (not just
        ip.StoragePathLineEdit's CURRENT one) -- per confirmed real bug: a
        hybe-choosing combo (Spot Localization's own HybeComboBox
        included) can freely offer hybes from either modality at once
        (see _storage_path_for_modality's own docstring on this), but this
        method used to only ever activate whichever ONE modality the
        Ingestion tab happened to be showing. Selecting a hybe from the
        OTHER modality then read an fov_matrices/unassigned-pool key
        that was never loaded at all -- e.g. the FOV-view crop displayer
        showing 0 unassigned spots for a DNA hybe while the Ingestion tab
        was still on RNA, even though real unassigned spots existed on
        disk for that exact (storage_path, fov).
        """
        for modality_name in self.ui.IngestionPanel.modality_names:
            storage_path = self._storage_path_for_modality(modality_name)
            if not storage_path:
                continue
            hybe_records = self._hybe_records_for_storage_path(storage_path)
            if not hybe_records:
                continue
            if not self._fov_matrices_for(storage_path, fov):
                try:
                    self._merge_fov_matrices(
                        fov, alignment.read_same_modality_matrices(storage_path, fov, hybe_records))
                except Exception:
                    pass
        # ONE fov-level spot load, outside the per-modality loop: the
        # container is modality-agnostic (each spot carries its own
        # modality), so the old per-storage-path staging -- which caused
        # the "0 unassigned spots for the other modality" bug -- has
        # nothing left to key on. Once per session, so a slice the user
        # cleared in memory is not resurrected by a later activation.
        if fov not in self._spot_loaded_fovs:
            self._spot_loaded_fovs.add(fov)
            any_path = next(iter(self._all_vlinks_storage_paths()), None)
            if any_path:
                # Heal legacy modality='' spots at the door where possible:
                # an empty modality resolves NO frame, so every later
                # recast silently degrades to identity for that spot. A
                # hybe present in exactly ONE modality's own records is
                # unambiguous; the cross-modal bridge hybe (present in
                # both) genuinely cannot be resolved and is left as-is.
                hybe_owners = {}
                for m in self.ui.IngestionPanel.modality_names:
                    for r in self._active_hybe_records_for_modality(m):
                        hybe_owners.setdefault(r['folder'], set()).add(m)
                unambiguous = {h: next(iter(ms)) for h, ms in hybe_owners.items()
                               if len(ms) == 1}
                try:
                    for d in vlinks_store.read_spots(any_path, fov):
                        if not d.get('modality') and d.get('hybe') in unambiguous:
                            d = dict(d, modality=unambiguous[d['hybe']])
                        spot = ASpot()
                        spot.set_metadata(**d)
                        # Both tiers stage EVERY spot, assigned and
                        # unassigned alike -- cells hold no spot lists, so
                        # the container is the one transient home and
                        # ASpot.cell is the only link to a cell.
                        self.spot_container_permanent.add(fov, spot)
                        twin = ASpot()
                        twin.set_metadata(**d)
                        self.spot_container.add(fov, twin)
                except Exception:
                    pass
        # Only once per FOV per session (per confirmed real regression):
        # _try_show_existing_cells unconditionally OVERWRITES self.
        # cell_container.data[fov] with a fresh deepcopy of cell_container_
        # permanent's own (possibly stale) snapshot -- correct the FIRST
        # time (populating the transient container from what's persisted),
        # but calling it again after the transient container has since been
        # mutated in-session (e.g. Save Current Spots identifying spots
        # into cells) would silently discard that mutation, since cell_
        # container_permanent is a SEPARATE container _save_current_spots
        # never writes back into. _activate_fov now runs on every Spot
        # Localization FOV/hybe/channel change (see _refresh_spot_cell_
        # list's own comment), so without this guard, EVERY refresh after
        # a save would revert cell-owned spot counts back to their pre-
        # save state even though the disk write itself succeeded.
        if fov not in self._activated_fovs:
            if self._try_show_existing_cells(fov):
                self._activated_fovs.add(fov)

    def _try_show_existing_cells(self, fov):
        """
        If this FOV already has persisted cells -- either already staged in
        cell_container_permanent this session, or sitting on disk in
        vlinks.h5 from a previous session -- show them immediately.
        Switching to (or restarting the app on) a FOV that was already
        segmented shouldn't require re-running segmentation just to see
        what's there. Reconstructs a label mask straight from the saved
        ACell.area coordinates (no re-computation), reads a fresh reference
        MIP for display, and stages the same cells into the transient
        container so Save/Discard/Send behave exactly as if they'd just
        been segmented interactively.

        Only ever safe to call ONCE per FOV per session past this point --
        see _activate_fov's own guard (self._activated_fovs) on why a
        second call would clobber real, later, in-session mutations to
        self.cell_container. Fresh-segmentation call sites (which set
        self._last_segment_context directly, not through this function)
        are unaffected by that guard -- they never call this at all.
        """
        ip = self.ui.IngestionPanel
        cp = self.ui.CellSegmentPanel
        storage_path = ip.StoragePathLineEdit.text().strip()

        have_in_memory = (self.cell_container_permanent is not None and fov in self.cell_container_permanent.data
                          and self.cell_container_permanent.data[fov])
        if not have_in_memory and storage_path:
            cell_dicts, modality = vlinks_store.read_cells(storage_path, fov)
            modality = modality or self._modality_for_storage_path(storage_path)
            if cell_dicts:
                loaded = CellContainer.load({fov: cell_dicts}, modality=modality)
                if self.cell_container_permanent is None:
                    self.cell_container_permanent = loaded
                else:
                    self.cell_container_permanent.data[fov] = loaded.data[fov]
                    if fov not in self.cell_container_permanent.fov_list:
                        self.cell_container_permanent.fov_list.append(fov)
                cp.LogTextEdit.append(f'Activated {len(cell_dicts)} cell(s) for FOV{fov:02d} from vlinks.h5 ({storage_path}).')

        if self.cell_container_permanent is None or fov not in self.cell_container_permanent.data \
                or not self.cell_container_permanent.data[fov]:
            return False
        if not storage_path:
            return False
        cells = self.cell_container_permanent.get_cells(fov)
        reference_hybe = cells[0].reference_hybe
        frame_shape = cells[0].frame_shape
        # same cross-modality mismatch _show_celltype_result guards against
        # -- these cells may have been segmented under a DIFFERENT modality
        # than the one currently active (they're mirrored into every
        # configured modality's own vlinks.h5), so reference_hybe might not
        # exist under the CURRENT storage_path at all. Resolve from the
        # cell's own segmentation modality first.
        # (reference_hybe, reference_MODALITY) is the pair: after cytoplasm
        # incorporation a cell's reference is e.g. (Hyb_500, RNA) while its
        # modality can still read 'DNA'; pairing the hybe with the wrong
        # modality asked DNA_queue for an RNA hybe -> None -> the loader
        # silently bailed and the FOV showed no cells at all.
        cell_storage_path = (self.ui.IngestionPanel.modality_data
                             .get(cells[0].reference_modality or '', {})
                             .get('storage_path') or storage_path)
        reference_image = vlinks_store.fiducial_channel_mip(cell_storage_path, fov, reference_hybe)
        if reference_image is None:
            cp.LogTextEdit.append(f'{reference_hybe} not in vlinks.h5 for FOV{fov:02d} -- cannot show existing cells.')
            return False

        # same uint8 label convention segment_fov/segment_fov_classical
        # already use elsewhere in this app (max 255 cells/FOV) -- not a
        # new limitation introduced here
        mask = np.zeros(frame_shape, dtype=np.uint8)
        for cell in cells:
            y, x = cell.area
            mask[y.astype(int), x.astype(int)] = cell.id

        if self.cell_container is None:
            self.cell_container = CellContainer([fov])
        self.cell_container.sync_from(self.cell_container_permanent, fov)
        # the CELLS' own segmentation modality, not the app's current one
        # (see cell_container.load_new_cells on why these can differ)
        self._last_segment_context = {'fov': fov, 'reference_hybe': reference_hybe,
                                      'modality': cells[0].reference_modality}
        self.cell_displayer.set_data(reference_image, mask)
        ap = self.ui.AlignmentPanel
        ap.CellOverlayFovSpinBox.blockSignals(True)
        ap.CellOverlayFovSpinBox.setValue(fov)
        ap.CellOverlayFovSpinBox.blockSignals(False)
        self._refresh_cell_fov_panels(fov)
        cp.LogTextEdit.append(f'Showing {len(cells)} already-saved cell(s) for FOV{fov:02d} (from permanent container).')
        return True

    @staticmethod
    def _cell_hybe_result_label(cell, fov, spec, reference_hybe, dx, dy, dz):
        """
        Row text for one (cell, hybe) spec (see _cell_overlay_target_
        specs) in "Results (per cell, per hybe)", per explicit request:
        'FOV{fov:03d} Cell{cell:03d}: {hybe} ({modality}) | {reference}
        ({modality}): dx=, dy=, dz='. Always tags BOTH sides with their
        own modality -- unlike the old 2-column version, every row here
        can independently be this cell's own modality or a different
        configured one (_cell_overlay_target_specs now enumerates every
        modality's hybes, not just cell.matrices' own keys), and hybe
        names aren't guaranteed unique across modalities, so a bare name
        alone would be ambiguous.
        """
        modality = spec['modality']
        # Two distinct groups, both shown -- per explicit correction ("be
        # honest for viewer"): the first dx/dy/dz is the TOTAL shift --
        # xy relative to this modality's own reference hybe (dominated by
        # the FOV-level inter-hybe shift, often several px) and z from
        # the cross-modal layer (dz, the modality->shared drift; within
        # one modality there is no FOV-level z term, so the reference-
        # relative z IS the cross-modal one). The trailing group is this
        # hybe's own CELL-LEVEL residual in full (dx, dy, AND its own dz
        # from cell.matrices -- the only part the 3-col preview's column
        # 2 -> 3 cyan movement shows, usually subpixel). Showing only the
        # total made a ~7px row look like a correction the preview then
        # visibly "failed" to apply; parking the cell-level dz beside the
        # total's dx/dy mislabeled which layer it belongs to.
        res_dx = spec['final_matrix'][1, 2] - spec['fov_only_matrix'][1, 2]
        res_dy = spec['final_matrix'][0, 2] - spec['fov_only_matrix'][0, 2]
        res_dz = spec.get('dz', 0.0)
        return (f"FOV{fov:03d} Cell{cell.id:03d}: {spec['hybe']} ({modality}) | "
               f"{reference_hybe} ({modality}): dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f} "
               f"| cell residual dx={res_dx:.2f}, dy={res_dy:.2f}, dz={res_dz:.2f}")

    def _refresh_cell_preview_reference_choices(self, cells_with_matrices, storage_path):
        """
        Populates Preview reference hybe FOV-wide -- the union of every
        hybe ANY cell in the current Overlay FOV has data for. This combo
        is paired with Overlay FOV and "Results (per cell, overlay)" (see
        _refresh_cell_overlay_list/_show_cell_all_readouts_overlay) -- it
        supplies the anchor for that all-readouts overlay, NOT "Results
        (per cell, per hybe)" (tier 1, which uses its own Reference hybe
        combo instead -- see _refresh_cell_per_hybe_results). Preserves
        the prior selection across a refresh, by key.
        """
        ap = self.ui.AlignmentPanel
        this_modality = self._modality_for_storage_path(storage_path)
        all_keys = set()
        for cell in cells_with_matrices:
            all_keys.update(cell.matrices.keys())

        def _label(hybe, modality):
            return hybe if modality == this_modality else f'{hybe} ({modality})'

        ap.CellPreviewReferenceHybeComboBox.blockSignals(True)
        last_reference_key = ap.CellPreviewReferenceHybeComboBox.currentData()
        ap.CellPreviewReferenceHybeComboBox.clear()
        for hybe, modality in sorted(all_keys):
            ap.CellPreviewReferenceHybeComboBox.addItem(_label(hybe, modality), (hybe, modality))
        if ap.CellPreviewReferenceHybeComboBox.count():
            restore_index = next((i for i in range(ap.CellPreviewReferenceHybeComboBox.count())
                                  if ap.CellPreviewReferenceHybeComboBox.itemData(i) == last_reference_key), 0)
            ap.CellPreviewReferenceHybeComboBox.setCurrentIndex(restore_index)
        ap.CellPreviewReferenceHybeComboBox.blockSignals(False)

    def _refresh_cell_overlay_list(self, fov):
        """
        Populates "Results (per cell, overlay)" -- every cell already
        saved/staged for `fov` (tier 3, driven by Overlay FOV) -- a pure
        read, never computes or writes anything. Also the source Save All
        Cell Overlays batches over (_cell_alignment_display_cells) --
        "every cell currently browsed in this FOV" is exactly the right
        scope for a batch save, and must stay that way regardless of
        whatever single cell "Results (per cell, per hybe)" happens to be
        showing (see _refresh_cell_per_hybe_results).
        """
        ap = self.ui.AlignmentPanel
        ip = self.ui.IngestionPanel
        storage_path = ip.StoragePathLineEdit.text().strip()
        cells = []
        if self.cell_container_permanent is not None:
            cells = self.cell_container_permanent.get_cells(fov)
        if not cells and self.cell_container is not None:
            cells = self.cell_container.get_cells(fov)
        cells_with_matrices = [c for c in cells if c.matrices]

        self._cell_alignment_display_cells = [(fov, c) for c in cells_with_matrices]
        self._refresh_cell_preview_reference_choices(cells_with_matrices, storage_path)

        ap.CellOverlayCellListWidget.clear()
        for cell in cells:
            item = QtWidgets.QListWidgetItem(f'Cell {cell.id}: {len(cell.matrices)} hybe(s) aligned')
            item.setData(QtCore.Qt.UserRole, cell.id)
            ap.CellOverlayCellListWidget.addItem(item)

    def _refresh_cell_fov_panels_from_combo(self):
        """Reads the FOV picked in CellOverlayFovSpinBox and refreshes
        "Results (per cell, overlay)" for it -- see
        _refresh_cell_overlay_list. Tier 3's own control; never touches
        "Results (per cell, per hybe)" (tier 1, scoped by its own FOV/
        Cell ID spinboxes instead -- see _refresh_cell_per_hybe_results)."""
        ap = self.ui.AlignmentPanel
        self._refresh_cell_overlay_list(ap.CellOverlayFovSpinBox.value())

    def _refresh_cell_per_hybe_results(self, fov, cell_id):
        """
        Populates "Results (per cell, per hybe)" for exactly the cell
        identified by the tier-1 FOV/Cell ID spinboxes -- one row per
        hybe in EVERY configured modality (via _cell_overlay_target_specs,
        the same enumeration the one-to-all overlay uses), each row
        resolving ITS OWN modality's configured reference hybe (ap.
        cell_align_references()) independently -- per explicit request,
        no modality picker needed: this app only ever holds one modality's
        cells resident in memory at a time, so a cell's reference pair alone (once
        the cell itself is found) is unambiguous, and showing every
        modality's hybes at once (not just the cell's home one) needs no
        further per-row choice either, since each row's own reference is
        already fully determined by that row's own modality.
        That's a SEPARATE pairing: Overlay FOV + Preview reference hybe +
        "Results (per cell, overlay)" (tier 3 -- see
        _refresh_cell_overlay_list/_show_cell_all_readouts_overlay).
        Refreshes live whenever FOV or Cell ID changes (see
        _refresh_cell_per_hybe_results_from_spinboxes). Pure read of
        already-saved/staged real cell data -- never computes or writes.
        Uses its own _cell_per_hybe_context, separate from
        _cell_alignment_display_cells (tier 3's "every cell in this FOV"
        state, which Save All Cell Overlays batches over) -- this list's
        single-cell scope must never narrow that batch down to one cell.
        """
        ap = self.ui.AlignmentPanel
        cell = None
        if self.cell_container_permanent is not None:
            cell = next((c for c in self.cell_container_permanent.get_cells(fov) if c.id == cell_id), None)
        if cell is None and self.cell_container is not None:
            cell = next((c for c in self.cell_container.get_cells(fov) if c.id == cell_id), None)

        ap.CellResultsListWidget.clear()
        if cell is None:
            self._cell_per_hybe_context = None
            return

        # storage_path/hybe_records represent the CELL's OWN (home)
        # modality -- the anchor _resolve_preview_hybe_context/_cell_
        # overlay_target_specs need, resolving every OTHER configured
        # modality's own hybes independently from there.
        storage_path = self._storage_path_for_modality(cell.reference_modality)
        hybe_records = self._active_hybe_records_for_modality(cell.reference_modality)
        self._cell_per_hybe_context = {'fov': fov, 'cell': cell, 'storage_path': storage_path,
                                       'hybe_records': hybe_records}
        if not storage_path:
            return

        channel_type = ap.CellChannelTypeComboBox.currentText()
        specs = self._cell_overlay_target_specs(cell, storage_path, fov, hybe_records, channel_type)
        for spec in sorted(specs, key=lambda s: (s['modality'], s['hybe'])):
            modality = spec['modality']
            reference_hybe = ap.current_cell_reference_hybe(modality)
            if not reference_hybe:
                continue
            reference_final_matrix = self._matrix_to_shared(reference_hybe, modality, cell, fov)
            if reference_final_matrix is None:
                continue
            dx = spec['final_matrix'][1, 2] - reference_final_matrix[1, 2]
            dy = spec['final_matrix'][0, 2] - reference_final_matrix[0, 2]
            # The TOTAL group's dz is the cross-modal layer's own z drift
            # for this row's modality into the shared frame (0 for the
            # shared side). The cell-level per-hybe z (spec['dz']) belongs
            # to the trailing "cell residual" group -- the label reads it
            # off spec itself.
            dz = self._cross_modal_z(
                modality, self._shared_frame_modality() or cell.reference_modality, fov)
            item = QtWidgets.QListWidgetItem(self._cell_hybe_result_label(cell, fov, spec, reference_hybe, dx, dy, dz))
            item.setData(QtCore.Qt.UserRole, (fov, cell.id, spec['hybe'], modality))
            ap.CellResultsListWidget.addItem(item)

    def _refresh_cell_per_hybe_results_from_spinboxes(self):
        """Reads FOV/Cell ID straight from tier 1's own spinboxes and
        refreshes "Results (per cell, per hybe)" for that cell -- see
        _refresh_cell_per_hybe_results."""
        ap = self.ui.AlignmentPanel
        self._refresh_cell_per_hybe_results(ap.CellFovSpinBox.value(), ap.CellIdSpinBox.value())

    def _refresh_cell_fov_panels(self, fov):
        """
        Convenience wrapper for call sites that want both lists
        refreshed: "Results (per cell, overlay)" for `fov` (tier 3,
        Overlay-FOV-driven) and "Results (per cell, per hybe)" for
        whatever's currently in the tier-1 FOV/Cell ID spinboxes -- per
        explicit request, that list is scoped ONLY by tier-1's own
        spinboxes, never by this function's own `fov` argument.
        """
        self._refresh_cell_overlay_list(fov)
        self._refresh_cell_per_hybe_results_from_spinboxes()

    # -- spot localization --

    def _refresh_spot_cell_list(self):
        """
        Self-activating -- calls _activate_fov(fov) first, so this never
        depends on the user having separately touched Cell Segmentation's
        OWN FovSpinBox first. Per confirmed real bug: cp.FovSpinBox and
        sp.FovSpinBox are two genuinely SEPARATE spinbox widgets (only
        their RANGE is kept in sync, see _refresh_fov_spinbox_bounds, not
        their value), but _activate_fov -- which lazily loads this FOV's
        cells/fov_matrices/spot_container from vlinks.h5 into the
        running session -- was only ever wired to cp.FovSpinBox.
        valueChanged. A session that only ever touches Spot Localization
        (never visits Cell Segmentation, or visits a DIFFERENT FOV there)
        left self.cell_container/self.spot_container never
        populated for Spot Localization's own current FOV -- Refresh Cell
        List showed nothing, and the FOV-view crop displayer showed 0
        unassigned spots, until SOME unrelated action elsewhere happened
        to trigger _activate_fov for the right FOV.
        """
        sp = self.ui.SpotLocalizationPanel
        fov = self._current_spot_fov()
        if fov is not None:
            self._activate_fov(fov)
        if self.cell_container is None or fov is None:
            sp.populate_cell_choices([])
        else:
            cells = self.cell_container.get_cells(fov)
            n_by_cell = {}
            for spot in self.spot_container.all(fov):
                if int(spot.cell) != -1:
                    n_by_cell[int(spot.cell)] = n_by_cell.get(int(spot.cell), 0) + 1
            sp.populate_cell_choices(cells, n_by_cell)
            sp.LogTextEdit.append(f'Cell list refreshed: {len(cells)} cell(s) for FOV{fov:02d}.')
        self._refresh_spot_fov_summary()
        self._refresh_spot_breakdown()

    def _refresh_spot_fov_summary(self):
        """
        Populates "FOV (all spots, this FOV)" -- per-(hybe, channel) spot
        COUNTS aggregated across every cell currently in this FOV, per
        explicit request ("to see all spots in FOV"), plus a second row
        per (hybe, channel) for spots that don't belong to any cell (see
        spot_container.unassigned/_replace_fov_unassigned_spots) -- these are
        real, kept spots too, just without a cell link, so they get
        their own visible count rather than being folded silently into
        the cell-owned total. Pure read -- never computes or writes.
        """
        sp = self.ui.SpotLocalizationPanel
        fov = self._current_spot_fov()
        sp.FovListWidget.clear()
        if self.cell_container is None or fov is None:
            return
        counts = {}  # (hybe, channel) -> [n_spots, {cell_id, ...}]
        for s in self.spot_container.all(fov):
            if int(s.cell) == -1:
                continue
            entry = counts.setdefault((s.hybe, s.channel), [0, set()])
            entry[0] += 1
            entry[1].add(int(s.cell))
        for hybe, channel in sorted(counts.keys()):
            n_spots, cell_ids = counts[(hybe, channel)]
            item = QtWidgets.QListWidgetItem(f'{hybe} ch{channel}: {n_spots} spot(s) across {len(cell_ids)} cell(s)')
            item.setData(QtCore.Qt.UserRole, (hybe, channel))
            sp.FovListWidget.addItem(item)
        # Every configured modality's own unassigned pool (per confirmed
        # real bug: this used to only ever read ip.StoragePathLineEdit's
        # CURRENT modality, silently omitting the other modality's own
        # unassigned spots from the count whenever the Ingestion tab
        # wasn't pointed at the same modality the FOV-view crop displayer
        # was actually showing -- e.g. saving DNA-hybe unassigned spots
        # while Ingestion was still on RNA left this list looking
        # unchanged even though real spots were identified/persisted).
        unassigned = self.spot_container.unassigned(fov)
        unassigned_counts = {}
        for s in unassigned:
            unassigned_counts[(s.hybe, s.channel)] = unassigned_counts.get((s.hybe, s.channel), 0) + 1
        for hybe, channel in sorted(unassigned_counts.keys()):
            item = QtWidgets.QListWidgetItem(
                f'{hybe} ch{channel}: {unassigned_counts[(hybe, channel)]} spot(s) unassigned (no cell)')
            item.setData(QtCore.Qt.UserRole, (hybe, channel))
            sp.FovListWidget.addItem(item)

    def _modality_owning_hybe(self, hybe):
        """Which configured modality's own active_hybe_list actually
        contains this hybe folder, or None if it's not in any of them."""
        for name in self.ui.IngestionPanel.modality_names:
            records = self.ui.IngestionPanel.modality_data.get(name, {}).get('active_hybe_list', [])
            if any(r['folder'] == hybe for r in records):
                return name
        return None

    def _on_fov_list_item_clicked(self, item):
        """
        Jumps HybeComboBox/ChannelComboBox to this row's (hybe, channel) --
        per explicit request, a shortcut from "see all spots in FOV" straight
        to that (hybe, channel)'s own view, without touching CellListWidget's
        current cell/FOV-view selection (that's a SEPARATE axis: which cell
        vs which hybe/channel).

        This list aggregates spots across every cell in the FOV, and a
        cell's own spots can span BOTH modalities (cell-based alignment
        processes both, see _other_modality_cell_alignment_inputs) -- a
        row's hybe is therefore not guaranteed to belong to whichever
        modality this row's data happens to resolve to. sp.HybeComboBox
        now offers every modality's hybes at once (see
        SpotLocalizationPanel.populate_hybe_choices), so jumping to it is
        a direct select_hybe(hybe, modality) call -- no need to switch
        any modality selector first the way this used to (there's no
        longer a modality selector on this panel at all).
        """
        data = item.data(QtCore.Qt.UserRole)
        if data is None:
            return
        hybe, channel = data
        sp = self.ui.SpotLocalizationPanel
        owning_modality = self._modality_owning_hybe(hybe)
        sp.select_hybe(hybe, owning_modality)
        # _on_hybe_changed (triggered above if the hybe actually changed)
        # already repopulates ChannelComboBox from that hybe's own channel
        # list -- only set the channel index after, and only if needed.
        cidx = sp.ChannelComboBox.findText(str(channel))
        if cidx >= 0 and sp.ChannelComboBox.currentIndex() != cidx:
            sp.ChannelComboBox.setCurrentIndex(cidx)

    def _refresh_spot_breakdown(self):
        """
        Populates "Spot (transient, this view)" -- per-(hybe, channel)
        spot counts for whichever view is currently selected: a real
        cell's own cell.spots, or (when the FOV pseudo-row is selected)
        this FOV's own unassigned-spot pool, per explicit request that
        this list not go blank just because the FOV row is picked. Pure
        read.
        """
        sp = self.ui.SpotLocalizationPanel
        sp.SpotListWidget.clear()
        counts = {}
        if sp.current_view() == 'cell':
            cell = self._selected_spot_cell()
            if cell is None:
                return
            for s in self.spot_container.of_cell(self._current_spot_fov(), cell.id):
                counts[(s.hybe, s.channel)] = counts.get((s.hybe, s.channel), 0) + 1
            suffix = 'spot(s)'
        else:
            # Every configured modality's own unassigned pool -- same fix
            # as _refresh_spot_fov_summary's own (see its comment): this
            # used to only ever read ip.StoragePathLineEdit's CURRENT
            # modality, silently omitting the other modality's spots.
            fov = self._current_spot_fov()
            for s in self.spot_container.unassigned(fov):
                counts[(s.hybe, s.channel)] = counts.get((s.hybe, s.channel), 0) + 1
            suffix = 'unassigned spot(s)'
        for hybe, channel in sorted(counts.keys()):
            item = QtWidgets.QListWidgetItem(f'{hybe} ch{channel}: {counts[(hybe, channel)]} {suffix}')
            sp.SpotListWidget.addItem(item)

    def _current_spot_fov(self):
        # Dedicated FOV spinbox (per explicit request) -- this panel now
        # owns its own FOV scope entirely, no longer borrowing Cell
        # Segmentation's own selector.
        return self.ui.SpotLocalizationPanel.FovSpinBox.value()

    def _selected_spot_cell(self):
        sp = self.ui.SpotLocalizationPanel
        fov = self._current_spot_fov()
        if fov is None or self.cell_container is None:
            return None
        cell_id = sp.selected_cell_id()
        if cell_id is None:
            return None
        for cell in self.cell_container.get_cells(fov):
            if cell.id == cell_id:
                return cell
        return None

    def _cell_area_in_readout(self, cell, hybe, modality, fov):
        """
        (y_area, x_area) -- this cell's own mask projected into hybe's
        own native frame, via _matrix_to_cellref (not cell.get_area_in_
        readout directly) -- per confirmed real bug, cell.matrix_to/get_
        area_in_readout silently collapse to IDENTITY (not a real FOV-
        level transform) whenever this cell has no cell.matrices/matrix_
        anchors entry for (hybe, modality) at all (cell-level alignment
        never run for it), mispositioning any mask overlay/crop built
        from it. _matrix_to_cellref already implements the correct have_
        real-gated fallback to the live FOV/cross-modal matrix. Shared by
        every MainWindow-level caller that needs this projection for
        real, currently-displayed positioning (not internal-only use,
        where ACell's own bare get_area_in_readout -- with no access to
        live session state -- is the correct/only option, e.g. inside
        localization._build_cell_crop's own have_real branch).

        Confirmed real bug this fixes: _load_fov_spot_display's own
        "cell masks" FOV-view overlay called cell.get_area_in_readout
        directly and never got this fallback, so it silently disagreed
        with _build_cell_display_crop's own (already-fixed) Cell-view
        projection for the exact same cell/hybe whenever no real cell-
        level data existed -- the FOV-view overview (which a user
        naturally trusts as ground truth) was the one still wrong, not
        the Cell-view crop being compared against it.
        """
        H_cellref = self._matrix_to_cellref(hybe, modality, cell, fov)
        if H_cellref is None:
            return np.array([]), np.array([])
        y_lit, x_lit = cell.area
        cy, cx = alignment.align_cell((y_lit, x_lit), la.inv(H_cellref), cell.frame_shape)
        return cy, cx

    def _build_cell_display_crop(self, cell, hybe, channel, storage_path, fov, pad, modality):
        """
        Raw (unmasked) crop + cell-boundary mask for the interactive
        displayer's Current Cell scope -- per explicit request, the
        interactive view shows real surrounding context (not neutralized
        to NaN/background the way localization._build_cell_crop's own
        peak-search crop legitimately is, to keep neighboring-cell/
        background pixels out of auto-detect) and draws the cell's own
        boundary as a contour instead. Same bbox math as _build_cell_crop
        (padding, clamping); returns None if the cell doesn't overlap
        this hybe's frame at all (same "no no-alignment" graceful case).
        Uses _cell_area_in_readout (see its own docstring for the
        fallback this relies on).
        """
        y_area, x_area = self._cell_area_in_readout(cell, hybe, modality, fov)
        if len(x_area) == 0:
            return None
        x_area, y_area = x_area.astype(int), y_area.astype(int)
        height, width = cell.frame_shape
        rymin, rymax = max(0, y_area.min() - pad), min(height, y_area.max() + pad + 1)
        rxmin, rxmax = max(0, x_area.min() - pad), min(width, x_area.max() + pad + 1)
        mip = vlinks_store.read_hybe_mip(storage_path, fov, hybe, channel)
        if mip is None:
            return None
        mip_crop = mip[rymin:rymax, rxmin:rxmax]
        mask = np.zeros((rymax - rymin, rxmax - rxmin), dtype=bool)
        local_y, local_x = y_area - rymin, x_area - rxmin
        valid = (local_y >= 0) & (local_y < mask.shape[0]) & (local_x >= 0) & (local_x < mask.shape[1])
        mask[local_y[valid], local_x[valid]] = True
        # full_mip/mask_x/mask_y (whole-FOV frame, pre-crop) are for the
        # displayer's LEFT context panel (see _load_spot_crop_for_display)
        # -- returned here rather than re-read/re-computed by the caller,
        # since both are already in hand from the work just done above.
        return {'img': mip_crop, 'mask': mask, 'rxmin': rxmin, 'rymin': rymin,
                'full_mip': mip, 'mask_x': x_area, 'mask_y': y_area}

    def _global_spot_order(self, storage_path, fov, hybe, channel):
        """
        Canonical, FOV-WIDE ordering for one (hybe, channel): the
        unassigned pool first (in its own list order), then every cell's
        own spots, cells visited in ascending cell.id order -- same
        "editable list first, then cell-owned" convention the FOV view
        already uses (see _load_fov_spot_display's docstring), just
        extended to span every cell in the FOV rather than one view's
        worth. Position i (0-based) in the returned list is global index
        i+1 -- see _global_spot_index_map.

        Recomputed fresh on every call, not a persisted per-spot id: a
        spot's global index is its CURRENT position in this ordering, so
        it can shift if spots are added/removed elsewhere in the FOV
        between calls. What's guaranteed is that opening any ONE view
        (a single cell, or the unassigned pool) never renumbers starting
        from 1 just because that view happens to be the one open --
        selecting cell 65 shows its spots at whatever numbers they
        already hold in the full-FOV count (e.g. 145,146,147...).
        """
        ordered = [(s, None) for s in self.spot_container.unassigned(fov)
                  if s.hybe == hybe and s.channel == channel]
        cells = self.cell_container.get_cells(fov) if self.cell_container is not None else []
        cells_by_id = {c.id: c for c in cells}
        by_cell = {}
        for s in self.spot_container.all(fov):
            if int(s.cell) != -1 and s.hybe == hybe and s.channel == channel:
                by_cell.setdefault(int(s.cell), []).append(s)
        for cell in sorted(cells, key=lambda c: c.id):
            for s in by_cell.get(cell.id, []):
                ordered.append((s, cell))
        return ordered

    def _global_spot_index_map(self, storage_path, fov, hybe, channel):
        """{id(spot): global_index(1-based)} -- see _global_spot_order."""
        return {id(s): i + 1 for i, (s, _) in enumerate(self._global_spot_order(storage_path, fov, hybe, channel))}

    def _load_spot_crop_for_display(self, *_args, keep_view=False):
        sp = self.ui.SpotLocalizationPanel
        fov_for_activation = self._current_spot_fov()
        if fov_for_activation is not None:
            self._activate_fov(fov_for_activation)  # see _refresh_spot_cell_list's own comment on why
        cell = self._selected_spot_cell()
        hybe = sp.current_hybe_folder()
        channel_text = sp.ChannelComboBox.currentText()
        if cell is None or not hybe or not channel_text:
            return
        channel = int(channel_text)
        storage_path = self._storage_path_for_modality(sp.current_hybe_modality())
        fov = self._current_spot_fov()
        pad = sp.PadSpinBox.value()
        crop = self._build_cell_display_crop(cell, hybe, channel, storage_path, fov, pad, modality=sp.current_hybe_modality())
        if crop is None:
            sp.LogTextEdit.append(f'Cell {cell.id}: no crop for {hybe} -- the hybe has no image data '
                                  f'for this FOV, or the cell mask projects outside its frame. '
                                  f'(Alignment is NOT required -- missing layers default to identity.)')
            return
        rxmin, rymin = crop['rxmin'], crop['rymin']
        self._spot_crop_context = {'kind': 'cell', 'cell': cell, 'hybe': hybe, 'channel': channel,
                                   'modality': sp.current_hybe_modality(), 'rxmin': rxmin, 'rymin': rymin}
        scoped_spots = [s for s in self.spot_container.of_cell(fov, cell.id)
                        if s.hybe == hybe and s.channel == channel]
        existing_points = [(s.raw_coordinate[1] - rxmin, s.raw_coordinate[0] - rymin) for s in scoped_spots]
        # EVERY cell in this FOV, not just the selected one -- the left
        # panel exists to orient you, and a single lone contour among a
        # field of unmarked cells does not (confirmed: this used to pass
        # only [(cell.id, ...)], so every neighbour was invisible).
        # The selected cell is still the one the RIGHT panel crops to.
        context_masks = []
        for other in self._cells_for_fov(fov):
            oy, ox = self._cell_area_in_readout(other, hybe, sp.current_hybe_modality(), fov)
            if len(ox):
                context_masks.append((other.id, ox, oy))
        if not context_masks:
            context_masks = [(cell.id, crop['mask_x'], crop['mask_y'])]
        gmap = self._global_spot_index_map(storage_path, fov, hybe, channel)
        spot_indices = [gmap[id(s)] for s in scoped_spots]
        self.spot_crop_displayer.set_data(
            crop['img'], existing_points, mask=crop['mask'],
            context_image=crop['full_mip'], context_masks=context_masks,
            context_title=f'FOV{fov:02d} {hybe} ch{channel} (full)',
            spot_indices=spot_indices, keep_view=keep_view)
        preserve_ids = self._selected_3d_spot_ids()
        self._current_view_spot_refs = [(s, cell) for s in scoped_spots]
        self._refresh_localize_3d_spot_choices(preserve_selected_ids=preserve_ids)

    def _load_fov_spot_display(self, keep_view=False):
        """
        FOV view -- the full raw hybe/channel MIP with BOTH the current
        FOV-level unassigned spots (spot_container.unassigned, yellow --
        the EDITABLE list) and already-identified cell-owned spots for
        this FOV (red, read-only context) marked. Only the unassigned
        pool is actually editable here: manual click add/remove and
        spots_edited both only ever touch the yellow list (see
        SpotCropDisplayer.set_data's readonly_points param) -- mixing
        cell-owned points INTO that editable list would make a manual
        edit's "hand back the whole current point list" full-replace
        silently turn already-identified spots back into unassigned
        ones. To edit a specific cell's own spots, select that cell
        instead.

        rxmin/rymin are 0 here (no crop offset -- this is the raw,
        un-cropped FOV MIP), and _spot_crop_context IS set (kind='fov'),
        unlike the old Whole FOV scope -- manual clicks/removals here now
        genuinely edit the unassigned tier via _on_spot_crop_edited.
        """
        sp = self.ui.SpotLocalizationPanel
        modality = sp.current_hybe_modality()
        storage_path = self._storage_path_for_modality(modality)
        fov = self._current_spot_fov()
        if fov is not None:
            self._activate_fov(fov)  # see _refresh_spot_cell_list's own comment on why -- confirmed real
                                     # bug: without this, the spot container could still be empty here
                                     # (0 unassigned spots shown) whenever this ran before ANY FOV-spinbox
                                     # valueChanged had ever fired for this exact FOV this session.
        hybe = sp.current_hybe_folder()
        channel_text = sp.ChannelComboBox.currentText()
        if not storage_path or fov is None or not hybe or not channel_text:
            return
        channel = int(channel_text)
        mip = vlinks_store.read_hybe_mip(storage_path, fov, hybe, channel)
        if mip is None:
            sp.LogTextEdit.append(f'{hybe} ch{channel} not in vlinks.h5 for FOV{fov:02d} -- ingest it first.')
            return
        unassigned_spots = [s for s in self.spot_container.unassigned(fov)
                           if s.hybe == hybe and s.channel == channel]
        unassigned_points = [(float(s.raw_coordinate[1]), float(s.raw_coordinate[0])) for s in unassigned_spots]
        cell_owned_points = []
        cell_owned_refs = []
        context_masks = []
        if self.cell_container is not None:
            cells = self.cell_container.get_cells(fov)
            cells_by_id = {c.id: c for c in cells}
            # Two independent enumerations: boundaries are per CELL (every
            # cell draws, spots or not), points are per assigned SPOT.
            for cell in cells:
                # _cell_area_in_readout (not cell.get_area_in_readout
                # directly) -- per confirmed real bug, see that method's
                # own docstring: this FOV-view "cell masks" overview was
                # the one call site that never got the live FOV/cross-
                # modal fallback _build_cell_display_crop's own Cell-view
                # crop already has, so the two silently disagreed for any
                # cell without real cell-level alignment for this hybe.
                y_area, x_area = self._cell_area_in_readout(cell, hybe, modality, fov)
                if len(x_area):
                    context_masks.append((cell.id, x_area, y_area))
            for s in self.spot_container.all(fov):
                cell = cells_by_id.get(int(s.cell)) if int(s.cell) != -1 else None
                if cell is not None and s.hybe == hybe and s.channel == channel:
                    cell_owned_points.append((float(s.raw_coordinate[1]), float(s.raw_coordinate[0]), cell.id))
                    cell_owned_refs.append((s, cell))
        # The RIGHT (working) panel gets the cell boundaries too, as one
        # combined raster -- per confirmed real bug it was passed mask=None
        # here, so the panel a user actually clicks in showed no cell
        # outlines at all, while the LEFT context panel drew all 67. Cell
        # view never had the problem because its crop always carries a real
        # mask. rxmin/rymin are 0 in this view, so the FOV-frame mask
        # coordinates are already the panel's own coordinates.
        # LABEL raster, not boolean: the displayer contours each label
        # separately so touching cells stay distinguishable (a boolean
        # would trace only their merged outer hull).
        fov_mask = np.zeros(mip.shape, dtype=np.int32)
        for cell_id, xs, ys in context_masks:
            ix, iy = np.asarray(xs).astype(int), np.asarray(ys).astype(int)
            valid = (iy >= 0) & (iy < mip.shape[0]) & (ix >= 0) & (ix < mip.shape[1])
            fov_mask[iy[valid], ix[valid]] = int(cell_id)
        self._spot_crop_context = {'kind': 'fov', 'storage_path': storage_path, 'fov': fov,
                                   'hybe': hybe, 'channel': channel, 'modality': modality, 'rxmin': 0, 'rymin': 0}
        gmap = self._global_spot_index_map(storage_path, fov, hybe, channel)
        spot_indices = [gmap[id(s)] for s in unassigned_spots]
        readonly_indices = [gmap[id(s)] for s, _ in cell_owned_refs]
        self.spot_crop_displayer.set_data(
            mip, unassigned_points, color='yellow', readonly_points=cell_owned_points,
            mask=fov_mask if fov_mask.any() else None,
            context_image=mip, context_masks=context_masks,
            context_title=f'FOV{fov:02d} {hybe} ch{channel} -- cell masks',
            spot_indices=spot_indices, readonly_indices=readonly_indices, keep_view=keep_view)
        # same "editable list numbered first, then readonly" order
        # SpotCropDisplayer itself already draws (see its class docstring)
        # -- keeps the 3D-localization popup's row numbering identical to
        # what's actually on screen.
        preserve_ids = self._selected_3d_spot_ids()
        self._current_view_spot_refs = [(s, None) for s in unassigned_spots] + cell_owned_refs
        self._refresh_localize_3d_spot_choices(preserve_selected_ids=preserve_ids)

    def _show_spot_displayer(self, *_args):
        """
        Populates the pop-up crop displayer from whatever's currently set
        (modality/FOV/hybe/channel/view), independent of Run Auto-Detect
        -- per explicit request, viewing shouldn't require running
        detection first. Cell view needs a selected cell too (there's no
        "current cell" without one); FOV view doesn't.
        """
        sp = self.ui.SpotLocalizationPanel
        if sp.current_view() == 'cell':
            self._load_spot_crop_for_display()
        else:
            self._load_fov_spot_display()

    def _on_spot_cell_selected(self, *_args):
        """
        View selection changed (a real cell, or the FOV pseudo-row) --
        refresh the per-cell spot breakdown, keep Remove All Spots
        enabled only for the FOV view (it has no per-cell equivalent),
        and if the crop displayer is already open, refresh it to follow
        the newly selected view.
        """
        self._refresh_spot_breakdown()
        sp = self.ui.SpotLocalizationPanel
        if sp.ShowDisplayerPushButton.isChecked():
            self._show_spot_displayer()

    def _toggle_spot_crop_displayer(self, checked):
        if checked:
            self._show_spot_displayer()
            self.spot_crop_displayer.show()
            self.spot_crop_displayer.raise_()
        else:
            self.spot_crop_displayer.hide()

    def _toggle_localize_3d_displayer(self, checked):
        if checked:
            self._refresh_localize_3d_spot_choices()
            self.localize_3d_displayer.show()
            self.localize_3d_displayer.raise_()
        else:
            self.localize_3d_displayer.hide()

    @staticmethod
    def _z_status_text(spot):
        """
        'Z-accepted'/'Z-rejected'/'Z-not run' -- spot._z_status is a
        plain, SESSION-transient Python attribute (not part of ASpot's
        persisted schema/set_metadata/save()), set directly by
        _run_3d_localize the moment a spot goes through refine_spot_z,
        whichever way the fit came out. Deliberately not persisted: it's
        a "have I looked at this spot's Z this session" note, not real
        localization data (that's coordinate/raw_coordinate themselves).
        """
        status = getattr(spot, '_z_status', None)
        return {'accepted': 'Z-accepted', 'rejected': 'Z-rejected'}.get(status, 'Z-not run')

    def _refresh_localize_3d_spot_choices(self, preserve_selected_ids=None):
        """
        Repopulates the 3D-localization popup's spot list from
        self._current_view_spot_refs -- the SAME ordered (ASpot,
        ACell-or-None) list _load_spot_crop_for_display/_load_fov_spot_
        display just built the crop displayer's own points from, so row
        N here always corresponds to the crop displayer's own point #N.
        Cheap enough to call unconditionally every time either of those
        two run, whether or not the popup is currently visible.

        Labels use the SAME global, FOV-wide index _load_spot_crop_for_
        display/_load_fov_spot_display pass to the crop displayer (see
        _global_spot_index_map) -- not a 1..N recount over just this
        view -- so a spot's number matches between the two windows and
        stays put across a cell/view switch.

        preserve_selected_ids (optional): a set of id(spot) values --
        whichever of these are still present in self._current_view_
        spot_refs get RE-selected instead of the default select-all.
        Callers pass this whenever this refresh is happening for a
        reason OTHER than a genuine view switch (e.g. redrawing after a
        manual spot add/remove within the SAME cell/hybe/channel) --
        per confirmed real bug, defaulting to select-all on every single
        redraw silently discarded a user's deliberate selection before
        they'd had a chance to click Run/View, so Run ended up acting on
        every spot in view instead of just the selected ones. Falls back
        to select-all when none of these ids are found in the new view
        (a genuine view switch has nothing old to preserve).
        """
        sp = self.ui.SpotLocalizationPanel
        storage_path = self._storage_path_for_modality(sp.current_hybe_modality())
        fov = self._current_spot_fov()
        hybe = sp.current_hybe_folder()
        channel_text = sp.ChannelComboBox.currentText()
        if not storage_path or fov is None or not hybe or not channel_text:
            self.localize_3d_displayer.set_spot_choices([])
            return
        gmap = self._global_spot_index_map(storage_path, fov, hybe, int(channel_text))
        labels = [f'Spot {gmap.get(id(spot), "?")} | Cell {"unassigned" if cell is None else cell.id} '
                 f'| {self._z_status_text(spot)}'
                 for spot, cell in self._current_view_spot_refs]
        keep_selected = None
        if preserve_selected_ids:
            keep_selected = [i for i, (spot, _cell) in enumerate(self._current_view_spot_refs)
                             if id(spot) in preserve_selected_ids]
        self.localize_3d_displayer.set_spot_choices(labels, keep_selected=keep_selected)

    def _selected_3d_spot_ids(self):
        """
        id(spot) for whichever rows are currently selected in the
        3D-localization popup's own list, resolved against the CURRENT
        (about-to-be-replaced) self._current_view_spot_refs -- call this
        BEFORE overwriting that list/calling _refresh_localize_3d_spot_
        choices, so the selection can be carried over by spot identity
        rather than being silently reset (see that method's own
        preserve_selected_ids docstring).
        """
        return {id(self._current_view_spot_refs[i][0])
                for i in self.localize_3d_displayer.selected_indices()
                if i < len(self._current_view_spot_refs)}

    def _resolve_selected_3d_targets(self):
        """
        Shared by Run and View: current storage_path/fov/hybe/modality/
        channel + whichever (ASpot, ACell-or-None) pairs are SELECTED in
        the popup's list, resolved back from row indices via
        self._current_view_spot_refs. Returns (storage_path, fov, hybe,
        modality, channel, targets) or None (after showing the relevant
        warning dialog itself) if anything's missing/empty/stale.
        """
        sp = self.ui.SpotLocalizationPanel
        modality = sp.current_hybe_modality()
        storage_path = self._storage_path_for_modality(modality)
        fov = self._current_spot_fov()
        hybe = sp.current_hybe_folder()
        channel_text = sp.ChannelComboBox.currentText()
        if not storage_path or fov is None or not hybe or not channel_text:
            QtWidgets.QMessageBox.warning(self, '3D Localization', 'Set storage path/FOV/hybe/channel first.')
            return None
        channel = int(channel_text)

        indices = self.localize_3d_displayer.selected_indices()
        if not indices:
            QtWidgets.QMessageBox.warning(self, '3D Localization', 'Select at least one spot from the list first.')
            return None
        targets = [self._current_view_spot_refs[i] for i in indices if i < len(self._current_view_spot_refs)]
        if not targets:
            QtWidgets.QMessageBox.warning(self, '3D Localization',
                                          'Selected spot(s) are no longer in the current view -- refresh and try again.')
            return None
        return storage_path, fov, hybe, modality, channel, targets

    def _spot_grid_title(self, storage_path, fov, hybe, channel, spot, cell):
        gidx = self._global_spot_index_map(storage_path, fov, hybe, channel).get(id(spot), '?')
        tag = 'unassigned' if cell is None else cell.id
        return f'Spot {gidx} | Cell {tag}'

    def _run_3d_localize(self):
        """
        Adds/refines Z on whichever spots are SELECTED in the popup's own
        list -- never a fresh detection pass (2D auto-detect finds spots;
        this only refines spots that already exist), and never silently
        every spot in view (per explicit request: Run only touches what's
        selected).

        Works from either view -- refine_spot_z itself already accepts
        cell=None for an unassigned FOV-pool spot (mapped into the shared
        frame via fov_matrices instead of a cell residual -- see refine_
        spot_z's own fov_matrices docstring), so a selection can freely
        mix cell-owned and unassigned rows; each is refined with the
        right cell= for its own row.

        Pure action, no visualization -- per explicit request to keep
        Run and View fully separate concerns, this never touches the
        fit-status grid at all (that's View's job now, not an optional
        side effect gated by a checkbox).

        A crop that resolves to more than one real component (mixture
        mode only -- see localize_3d_displayer's MultiModeCheckBox and
        localization.refine_spot_z's use_mixture) is saved onto THIS ONE
        spot, never spawned as separate ASpot records -- per explicit
        request: spot.coordinate/raw_coordinate become the BRIGHTEST
        accepted component's position, and spot.mixture_centroids records
        every accepted component's own (x, y, z, amplitude), representative
        first, for later reference/display (canvas/spot_fit_status.py
        draws the representative yellow, the rest blue). Cleared back to
        () on a spot whose crop only ever had one component this run.
        """
        sp = self.ui.SpotLocalizationPanel
        resolved = self._resolve_selected_3d_targets()
        if resolved is None:
            return
        storage_path, fov, hybe, modality, channel, targets = resolved
        params = self.localize_3d_displayer.params()

        fp_undo = self._begin_spot_edit(fov)

        n_refined = 0
        n_mixture = 0
        claimed_positions = []  # (abs_x, abs_y) of already-refined spots in THIS batch, so
                                # two distinct spots sharing an ambiguous crop don't collapse
                                # onto the same blob -- see refine_spot_z's own docstring
        fov_matrices = self._composed_fov_matrices_for_cell_alignment(storage_path, fov)
        # ONE resolver for the whole batch, complete for XY and Z alike:
        # anchors are modality-level and _frame_resolver now carries them
        # even with cell=None, so to_shared composes each row's own cell
        # residual from the `cell` argument per spot. (The old claim that
        # a cell-free resolver was complete held only for Z -- its empty
        # anchors silently dropped every XY residual to the FOV route.)
        resolver = self._frame_resolver(None, fov)
        t0 = time.perf_counter()
        for spot, cell in targets:
            new_coordinate, new_raw, _cubic, _centroid, _extra_results, mixture_centroids = localization.refine_spot_z(
                spot, storage_path, fov, channel, hybe=hybe, cell=cell, modality=modality,
                spad=params['spad'], peak_bound=params['peak_bound'], max_sigma=params['max_sigma'],
                max_uncert=params['max_uncert'], min_hb_ratio=params['min_hb_ratio'], min_ah_ratio=params['min_ah_ratio'],
                min_sep=params['min_sep'], claimed_positions=claimed_positions, use_mixture=params['multi_mode'],
                z_window=params['z_window'], fov_matrices=fov_matrices, resolver=resolver)
            if new_coordinate is not None:
                spot.coordinate = new_coordinate
                spot.raw_coordinate = new_raw
                spot._z_status = 'accepted'
                spot.mixture_centroids = mixture_centroids
                n_refined += 1
                if mixture_centroids:
                    n_mixture += 1
                claimed_positions.append((new_raw[0], new_raw[1]))
            else:
                spot._z_status = 'rejected'
        elapsed = time.perf_counter() - t0
        # The whole batch ran synchronously above; one undo step covers it.
        self._commit_spot_edit(fov, fp_undo)

        mode_label = 'mixture' if params['multi_mode'] else 'single'
        mixture_msg = f', {n_mixture} with >1 real component saved as mixture_centroids' if n_mixture else ''
        self.localize_3d_displayer.StatusLabel.setText(
            f'{hybe} ch{channel}: {n_refined}/{len(targets)} selected spot(s) refined with real Z '
            f'({len(targets) - n_refined} rejected -- z left as-is){mixture_msg}. '
            f'[{mode_label} mode, {elapsed:.2f}s for {len(targets)} spot(s)]')
        self._refresh_spot_cell_list()
        if sp.current_view() == 'cell':
            self._load_spot_crop_for_display()
        else:
            self._load_fov_spot_display()

    def _view_3d_localize(self):
        """
        Preview-only counterpart to Run -- fits the SAME selection but
        never writes spot.coordinate/raw_coordinate, never pushes undo,
        never marks _z_status. This is the ONLY action that renders the
        fit-status grid -- always, unconditionally (no checkbox): viewing
        the crop/fit IS the whole point of this button.
        """
        resolved = self._resolve_selected_3d_targets()
        if resolved is None:
            return
        storage_path, fov, hybe, modality, channel, targets = resolved
        params = self.localize_3d_displayer.params()

        n_would_accept = 0
        n_would_be_mixture = 0
        grid_results = []
        claimed_positions = []  # see _run_3d_localize's own comment on this
        fov_matrices = self._composed_fov_matrices_for_cell_alignment(storage_path, fov)
        resolver = self._frame_resolver(None, fov)  # same resolver _run_3d_localize uses, so
                                                    # preview and run report the same z
        t0 = time.perf_counter()
        for spot, cell in targets:
            title = self._spot_grid_title(storage_path, fov, hybe, channel, spot, cell)
            _, new_raw, cubic, centroid, _extra_results, mixture_centroids = localization.refine_spot_z(
                spot, storage_path, fov, channel, hybe=hybe, cell=cell, modality=modality,
                spad=params['spad'], peak_bound=params['peak_bound'], max_sigma=params['max_sigma'],
                max_uncert=params['max_uncert'], min_hb_ratio=params['min_hb_ratio'], min_ah_ratio=params['min_ah_ratio'],
                min_sep=params['min_sep'], claimed_positions=claimed_positions, use_mixture=params['multi_mode'],
                z_window=params['z_window'], fov_matrices=fov_matrices, resolver=resolver)
            if cubic is not None:
                grid_results.append((cubic, centroid, title))
            if new_raw is not None:
                n_would_accept += 1
                claimed_positions.append((new_raw[0], new_raw[1]))
            if mixture_centroids:
                n_would_be_mixture += 1
        elapsed = time.perf_counter() - t0

        self.localize_3d_grid_displayer.show_fit_status_grid(grid_results)
        self.localize_3d_grid_displayer.show()
        self.localize_3d_grid_displayer.raise_()
        mode_label = 'mixture' if params['multi_mode'] else 'single'
        mixture_msg = f', {n_would_be_mixture} would save >1 component as mixture_centroids' if n_would_be_mixture else ''
        self.localize_3d_displayer.StatusLabel.setText(
            f'{hybe} ch{channel}: PREVIEW ONLY, nothing saved -- {n_would_accept}/{len(targets)} '
            f'selected spot(s) would be accepted{mixture_msg}. '
            f'[{mode_label} mode, {elapsed:.2f}s for {len(targets)} spot(s)]')

    def _mixture_centroid_to_raw(self, coord_xyz, cell, hybe, modality, fov_matrices=None):
        """
        Inverse of localization.refine_spot_z's own _to_real closure --
        maps a shared-frame (y, x, z) (the frame spot.coordinate AND
        every entry of spot.mixture_centroids beyond the first already
        live in, see ASpot.mixture_centroids' own docstring) back to
        hybe's raw pixel frame, so an ALREADY-persisted mixture sibling
        can be drawn on a raw crop without ever re-running a fit.

        cell is None (unassigned spot): mirrors _to_real's own cell=None
        branch -- inverts through fov_matrices (see main_window._
        composed_fov_matrices_for_cell_alignment) when this hybe has a
        real FOV-level matrix, identity otherwise. Returns (y, x, z) --
        rasterized order, same as every coordinate in this pipeline.
        """
        if cell is None:
            if fov_matrices and (hybe, fov_matrices.modality) in fov_matrices:
                y, x, z = coord_xyz
                ry, rx = spot_mapper.reference_to_raw((y, x), hybe, fov_matrices, modality=modality, cell=None)
                return ry, rx, z
            return coord_xyz
        y, x, z = coord_xyz
        # Resolver-backed, never ACell.matrix_to_shared directly -- that
        # raises by design once this cell carries residual-form matrices
        # (post cell alignment); the resolver composes the FOV leg live.
        H = self._matrix_to_shared(hybe, modality, cell, cell.fov)
        if H is None:
            H = np.eye(3)
        Hinv = la.inv(H)
        ry, rx, _ = Hinv @ np.array([y, x, 1.0])
        # entry_dz returns a plain float now -- the old Hz[1, 2] indexing
        # here was a latent TypeError (floats don't index) that #62's
        # sweep missed because this path only runs for persisted mixture
        # siblings.
        dz_cell = alignment.entry_dz(cell.matrices.get((hybe, modality)))
        return float(ry), float(rx), float(z - dz_cell)

    def _show_3d_crop_only(self):
        """
        Fit-free counterpart to View -- crops the raw Z-stack around each
        selected spot's own raw (x,y) via spot_mapper.crop_for_localization
        directly (the exact same crop step localization.refine_spot_z's
        own first few lines perform), never calling refine_spot_z itself
        so no fit (single or mixture) ever runs. Per explicit request:
        View always runs the real fit to preview whether it would
        succeed, which is exactly the expensive part to skip when someone
        just wants a quick look at the raw crop before deciding whether
        fitting it is even worth doing.

        Still draws a circle for a spot that ALREADY has a real,
        persisted Z (coordinate[2] != 0.0 -- the same proxy the Memory
        Status/Cell-Spot-Status-Detail "Spot Z" column already uses,
        since _z_status itself is session-transient) -- per explicit
        request: "show yellow/blue circles if exist, draw only if
        exists." This reconstructs crop-local positions from spot.raw_
        coordinate (already raw-frame, no transform needed) for the
        representative, and spot.mixture_centroids[1:] (shared-frame,
        inverse-transformed via _mixture_centroid_to_raw above) for any
        other accepted component -- it never runs a NEW fit, only places
        markers for whatever was already computed and saved by an
        earlier Run. A spot with no real Z yet gets centroid=None, same
        as before -- no circle, nothing to show.
        """
        resolved = self._resolve_selected_3d_targets()
        if resolved is None:
            return
        storage_path, fov, hybe, modality, channel, targets = resolved
        params = self.localize_3d_displayer.params()
        fov_matrices = self._composed_fov_matrices_for_cell_alignment(storage_path, fov)

        grid_results = []
        for spot, cell in targets:
            title = self._spot_grid_title(storage_path, fov, hybe, channel, spot, cell)
            raw_y, raw_x = float(spot.raw_coordinate[0]), float(spot.raw_coordinate[1])
            try:
                cubic, (ymin, xmin) = spot_mapper.crop_for_localization(storage_path, fov, hybe, channel,
                                                                        (raw_y, raw_x), pad=params['spad'], use_stack=True)
            except OSError:
                continue
            if cubic.size == 0:
                continue
            centroid = None
            if spot.coordinate[2] != 0.0:
                # raw_coordinate is (y, x, z); the displayer's centroid
                # contract is crop-local (x, y, z) -- the old line
                # subtracted xmin from Y (transposed circles, confirmed
                # real on the fit-status grid)
                own_raw = spot.raw_coordinate
                centroid = [(own_raw[1] - xmin, own_raw[0] - ymin, own_raw[2])]
                for c in spot.mixture_centroids[1:]:
                    ry, rx, rz = self._mixture_centroid_to_raw(c[:3], cell, hybe, modality, fov_matrices=fov_matrices)
                    centroid.append((rx - xmin, ry - ymin, rz))
            grid_results.append((cubic, centroid, title))

        self.localize_3d_grid_displayer.show_fit_status_grid(grid_results)
        self.localize_3d_grid_displayer.show()
        self.localize_3d_grid_displayer.raise_()
        self.localize_3d_displayer.StatusLabel.setText(
            f'{hybe} ch{channel}: showing raw crop for {len(grid_results)}/{len(targets)} selected spot(s) -- no fit computed '
            f'(existing Z shown where already saved).')

    def _current_spot_scope_max(self):
        """Best-effort max intensity of whatever's currently shown in the
        crop displayer -- the same 'scope max' Run Auto-Detect itself
        would use -- for live-syncing the two threshold fields. None if
        nothing's loaded yet (fields are left alone in that case)."""
        img = self.spot_crop_displayer.crop_image
        if img is None:
            return None
        val = float(np.nanmax(img))
        return val if val > 0 else None

    def _sync_threshold_from_percent(self):
        """Typing/confirming a % recomputes the Absolute field to match,
        against whatever's currently displayed (see _current_spot_scope_max)
        -- per explicit request, the two fields stay linked in both
        directions rather than the Absolute field being a rarely-touched
        override. No-ops silently if no scope max is known yet or the %
        text isn't a number (the user is mid-edit; threshold_abs() still
        raises a real error at Run Auto-Detect time if it's left invalid)."""
        sp = self.ui.SpotLocalizationPanel
        scope_max = self._current_spot_scope_max()
        if scope_max is None:
            return
        try:
            pct = float(sp.ThresholdPercentLineEdit.text().strip())
        except ValueError:
            return
        sp.ThresholdAbsoluteLineEdit.setText(f'{(pct / 100.0) * scope_max:.1f}')

    def _sync_threshold_from_absolute(self):
        """Inverse of _sync_threshold_from_percent."""
        sp = self.ui.SpotLocalizationPanel
        scope_max = self._current_spot_scope_max()
        abs_text = sp.ThresholdAbsoluteLineEdit.text().strip()
        if scope_max is None or not abs_text:
            return
        try:
            abs_val = float(abs_text)
        except ValueError:
            return
        sp.ThresholdPercentLineEdit.setText(f'{(abs_val / scope_max) * 100.0:.1f}')

    def _remove_transient_spots(self):
        """
        Reverts the current view's current (hybe, channel) back to
        exactly what's on disk -- re-reads vlinks.h5 fresh (the real
        "permanent" source of truth) rather than tracking a separate
        in-memory flag per spot, since manual edits already rebuild the
        whole point list from scratch on every add/remove (see
        _on_spot_crop_edited), which would make a per-object "is this
        spot transient" flag unreliable to maintain anyway. Spots
        already saved survive untouched; anything added/edited since the
        last save is discarded.
        """
        sp = self.ui.SpotLocalizationPanel
        storage_path = self._storage_path_for_modality(sp.current_hybe_modality())
        fov = self._current_spot_fov()
        hybe = sp.current_hybe_folder()
        channel_text = sp.ChannelComboBox.currentText()
        if not storage_path or fov is None or not hybe or not channel_text:
            return
        channel = int(channel_text)
        if sp.current_view() == 'cell':
            cell = self._selected_spot_cell()
            if cell is None:
                QtWidgets.QMessageBox.warning(self, 'Remove Transient Spots', 'Select a cell first.')
                return
            # "What was saved" is the permanent tier, in memory -- kept
            # exact by the loader and by every persist -- so Revert never
            # re-reads disk. Deep copies: the tiers must not share objects.
            permanent = [deepcopy(sp) for sp in self.spot_container_permanent.of_cell(fov, cell.id)
                         if sp.hybe == hybe and int(sp.channel) == channel]
            fp = self._begin_spot_edit(fov)
            self._replace_cell_spots(cell, hybe, channel, permanent)
            self._commit_spot_edit(fov, fp)
            sp.LogTextEdit.append(f'Cell {cell.id}, {hybe} ch{channel}: reverted to {len(permanent)} '
                                  f'permanent spot(s) (transient discarded).')
        else:
            permanent = [deepcopy(sp) for sp in self.spot_container_permanent.unassigned(fov)
                         if sp.hybe == hybe and int(sp.channel) == channel]
            fp = self._begin_spot_edit(fov)
            self._replace_fov_unassigned_spots(storage_path, fov, hybe, channel, permanent)
            self._commit_spot_edit(fov, fp)
            sp.LogTextEdit.append(f'FOV{fov:02d}, {hybe} ch{channel}: reverted to {len(permanent)} '
                                  f'permanent unassigned spot(s) (transient discarded).')
        self._refresh_spot_cell_list()
        if sp.ShowDisplayerPushButton.isChecked():
            self._show_spot_displayer()

    def _save_all_fov_spots(self):
        """
        Whole-FOV spot save, per explicit request: reassigns every spot
        in this FOV against the current cells (the same recomputation
        every save runs) and then persists EVERY (modality, hybe,
        channel) slice the session holds -- not just the currently
        viewed one -- clearing stale slices, so nothing is left behind.
        Deliberately slower than the slice-scoped Save Current Spots
        (seconds: full reassignment plus every slice write) in exchange
        for needing to be re-run far less often. Writes SPOTS only,
        never cells (same rule as every spot door).
        """
        sp = self.ui.SpotLocalizationPanel
        fov = self._current_spot_fov()
        if fov is None or not self._all_vlinks_storage_paths():
            QtWidgets.QMessageBox.warning(self, 'Save ALL FOV Spots', 'Set a storage path and FOV first.')
            return
        t0 = time.perf_counter()
        n_identified, n_identified_cells = self._reassign_fov_spots(fov)
        n_written = self._persist_fov_spots(fov)
        elapsed = time.perf_counter() - t0
        sp.LogTextEdit.append(
            f'FOV{fov:02d}: ALL slices saved -- {n_written} spot(s) across every hybe/channel '
            f'({n_identified} unassigned spot(s) newly identified into {n_identified_cells} '
            f'cell(s); {elapsed:.1f}s). Cells are never written here.')
        QtWidgets.QMessageBox.information(
            self, 'Save ALL FOV Spots',
            f'{n_written} spot(s) saved across all slices of FOV{fov:02d} ({elapsed:.1f}s).')
        self._refresh_spot_cell_list()

    def _save_current_spots(self):
        """
        Saves EVERY cell's current in-memory spots for this FOV in one
        pass -- not scoped to whichever cell/view happens to be open --
        plus identifies+saves the FOV-level unassigned pool against the
        current cell mask. Replaces the old view-scoped "Save View"
        (Cell view: only the selected cell; FOV view: only the
        unassigned pool), which had a real, confirmed bug: pick spots
        across many cells in Cell view, switch to FOV view, click Save --
        nothing for any of those cells ever reached disk, because the
        FOV-view branch only ever looked at the unassigned pool.

        Writes SPOTS ONLY -- never cells (see the inline note below).
        """
        sp = self.ui.SpotLocalizationPanel
        fov = self._current_spot_fov()
        storage_paths = self._all_vlinks_storage_paths()
        if not storage_paths:
            QtWidgets.QMessageBox.warning(self, 'Save Current Spots', 'No storage path available.')
            return

        # No cells loaded in memory is only a wipe HAZARD when real cells
        # already exist on disk for this FOV AND we're about to write_
        # cells below (a bulk write would then silently drop them) -- so
        # this check only runs at all when there's something to write.
        # An empty cell_container is a perfectly legitimate state
        # otherwise (e.g. just locating some FOV-level unassigned spots
        # for a modality that's never had cell segmentation run at all),
        # and must never block the unassigned-pool save below, which
        # needs no cells at all -- per confirmed real bug (twice now):
        # this used to compare against self.ui.IngestionPanel.
        # StoragePathLineEdit's CURRENT storage path regardless of which
        # modality was actually being worked on, but write_cells mirrors
        # every cell across EVERY storage_paths entry on every save
        # (confirmed on real data: a DNA-only FOV's cells appear
        # identically under RNA's own vlinks.h5 too) -- so on-disk count
        # is never actually 0 once ANY modality has cells anywhere,
        # meaning this guard fired (and, being unconditional, blocked
        # the ENTIRE function including the unrelated unassigned-pool
        # save) even for a session with zero DNA cells loaded that was
        # only ever trying to save RNA's own unassigned spots. Scoping
        # to self.cell_container's own modality/storage path, and only
        # skipping write_cells (never returning early), fixes both.
        # No in-memory-vs-disk count guard: the transient container is
        # AUTHORITATIVE at save time. Fewer cells in memory than on disk is
        # the normal result of deliberately removing a cell, and save means
        # "replace" -- blocking it forced a Refresh that resurrected the
        # removed cell. Spots that pointed at a removed cell are handled by
        # the reassignment below, which recomputes every spot's owner from
        # the current cells (~1.4us/spot measured: cheap enough for EVERY
        # save -- cell, matrix, and spot alike, per explicit decision).
        # SPOT SAVE WRITES SPOTS -- NEVER CELLS. Per explicit correction
        # (confirmed real data loss): this door used to bulk-write the
        # TRANSIENT cell container to disk, while cell alignment mutates
        # the PERMANENT tier's copies -- so saving spots after a cell
        # alignment run silently wiped every cell's matrices from
        # vlinks.h5. Nothing in a spot operation mutates a cell (ASpot.
        # cell is the only link, and it lives on the spot), so there is
        # nothing here for cells to persist. Cell persistence belongs
        # exclusively to the cell doors (segmentation save, cytoplasm
        # incorporation, cell alignment, celltype persist).
        n_identified, n_identified_cells = self._reassign_fov_spots(fov)
        n_spots = len(self.spot_container.all(fov))

        # The FOV-level unassigned pool needs no cells at all -- always
        # attempted regardless of the above. _identify_fov_unassigned_
        # spots already wrote the leftover pool once (with whatever stale
        # .id each spot had coming in); re-write it now that _assign_spot_
        # indices has given every spot its real, current index, so the
        # ids that actually land on disk match write_cells' own cell-owned
        # spots above, not a snapshot from before assignment.
        #
        # write_fov_spots to JUST this modality's own storage_path here --
        # NOT mirror_write_fov_spots(storage_paths, ...), which writes the
        # SAME payload to every storage path this session knows about.
        # Confirmed real bug: fov_unassigned_spots is keyed by (storage_
        # path, fov), i.e. genuinely per-modality with no owning cell to
        # justify cross-modality mirroring (unlike write_cells, where the
        # SAME cell legitimately has spots spanning both modalities) --
        # mirroring RNA's own pool onto DNA's storage path (and vice
        # versa) means whichever modality is processed LAST in self.
        # modality_names permanently overwrites every OTHER modality's own
        # unassigned pool with its own spots. Reproduced on a scratch
        # copy: with self.ui.IngestionPanel.modality_names == ['RNA', 'DNA'], an RNA Hyb_105
        # spot and a DNA Hyb_010 spot both pending at once resulted in
        # BOTH storage paths ending up with only the DNA spot -- RNA's own
        # spot silently vanished. This is the root cause of "RNA unassigned
        # spots don't save" whenever DNA also has a pending pool in memory.
        # Per explicit decision, Save is scoped to the CURRENT (hybe,
        # channel, modality) -- the same scope as every removal control --
        # so it can never touch a hybe the user never opened. The old
        # "view-scoped save loses other cells' spots" bug cannot recur
        # under this scoping: a slice save persists that hybe/channel for
        # EVERY cell and the unassigned pool together, because they all
        # live in the one slice.
        current_slice = None
        hybe = sp.current_hybe_folder()
        modality = sp.current_hybe_modality()
        channel_text = sp.ChannelComboBox.currentText().strip()
        if hybe and modality and channel_text:
            current_slice = (modality, hybe, int(channel_text))
        n_written = self._persist_fov_spots(fov, only_slice=current_slice)
        sp.LogTextEdit.append(
            f'FOV{fov:02d} {hybe} ch{channel_text}: {n_written} spot(s) persisted, '
            f'{assignment.count_unassigned(self._all_transient_spots(fov))} unassigned in FOV.')

        sp.LogTextEdit.append(f'FOV{fov:02d}: {n_spots} total spot(s) saved to vlinks.h5 '
                              f'({n_identified} unassigned spot(s) newly identified into '
                              f'{n_identified_cells} cell(s) first). Cells are never written here.')
        QtWidgets.QMessageBox.information(
            self, 'Save Current Spots', f'{n_spots} total spot(s) saved. (Spot saves never touch cells.)')
        self._refresh_spot_cell_list()
        if sp.ShowDisplayerPushButton.isChecked():
            self._show_spot_displayer()

    def _all_transient_spots(self, fov):
        """Every live in-memory spot for this FOV -- pools + cell-owned."""
        return list(self.spot_container.all(fov))

    def _clear_current_hybe_channel(self):
        """
        Remove EVERY spot of the current (hybe, channel, modality) from the
        transient state -- assigned and unassigned alike, one operation that
        never asks which kind a spot is. In-memory only, like every other
        edit here: the emptied slice reaches vlinks.h5 on the next Save,
        which writes the slice empty. Restores the capability deleted with
        the old "Remove Spots in View" button, which could only ever reach
        the unassigned pool because the split store hid assigned spots
        from it.
        """
        sp = self.ui.SpotLocalizationPanel
        fov = self._current_spot_fov()
        hybe = sp.current_hybe_folder()
        modality = sp.current_hybe_modality()
        channel_text = sp.ChannelComboBox.currentText().strip()
        if fov is None or not hybe or not modality or not channel_text:
            return
        channel = int(channel_text)
        fp = self._begin_spot_edit(fov)
        doomed = [s.uid for s in self.spot_container.all(fov)
                  if s.hybe == hybe and int(s.channel) == channel
                  and (s.modality or modality) == modality]
        n = len(self.spot_container.remove(fov, doomed))
        self._commit_spot_edit(fov, fp)
        sp.LogTextEdit.append(f'FOV{fov:02d} {hybe} ch{channel}: {n} spot(s) cleared '
                              f'(in memory -- Save persists the empty slice).')
        self._refresh_spot_cell_list()
        if sp.ShowDisplayerPushButton.isChecked():
            self._show_spot_displayer()

    def _persist_fov_spots(self, fov, only_slice=None):
        """
        Write spots into the unified slice store -- one write per
        (modality, hybe, channel), cell-owned and unassigned together.

        only_slice: optional (modality, hybe, channel). When given, ONLY
        that slice is written; every other slice on disk is left exactly
        as it is. This is what makes the Save button (hybe, channel)-
        scoped, per explicit decision: saving can then never touch a hybe
        the user never opened, and its scope matches every removal
        control. When None, every slice in memory is written and stale
        on-disk slices are cleared -- the full-FOV path used by bulk
        operations.

        Assigned and unassigned spots share a slice and differ only in
        ASpot.cell, so they MUST be written together: a slice write is a
        full replace, so writing only one kind would delete the other.
        That is also the point -- assignment stops being a move between two
        stores and becomes a field on a spot that never leaves its slice.

        Slices that used to hold spots but no longer do are written empty
        rather than skipped, so removing the last spot from a hybe actually
        removes it from disk instead of leaving the previous contents
        stranded.
        """
        # Refuse to persist a FOV whose spots were never staged: with no
        # transient state, the stale-slice pass below would enumerate every
        # slice on disk and write it EMPTY -- the accept loops call this for
        # every FOV that has matrices loaded, which is a superset of the
        # FOVs whose spots were ever activated. "Not loaded" means "nothing
        # to say", never "delete everything".
        if fov not in self._spot_loaded_fovs:
            return 0
        # Keyed by (modality, hybe, channel) -- the SLICE -- never by
        # storage_path. Since vlinks was unified every storage_path resolves
        # to the same file, so keying by path yields two different keys for
        # one physical slice, and whichever wrote last won. That is exactly
        # how a populated slice got overwritten by the empty entry the
        # stale-clearing pass below created for it.
        by_slice = {}
        seen_uids = {}

        # One home per spot now, so no cross-structure dedup is needed:
        # the container's own uid uniqueness is the guarantee.
        for spot in self.spot_container.all(fov):
            by_slice.setdefault((spot.modality, spot.hybe, int(spot.channel)), []).append(spot)

        # Slices on disk with nothing in memory get written empty, so
        # removing a hybe's last spot really removes it. One read is enough:
        # every storage_path resolves to the same file, and a FOV-wide read
        # already spans both modalities.
        any_path = next(iter(self._all_vlinks_storage_paths()), None)
        if any_path:
            for d in vlinks_store.read_spots(any_path, fov):
                by_slice.setdefault((d.get('modality', ''), d.get('hybe', ''),
                                     int(d.get('channel', 0))), [])

        if only_slice is not None:
            want = (only_slice[0], only_slice[1], int(only_slice[2]))
            by_slice = {k: v for k, v in by_slice.items() if k == want}
            # The slice must exist even if empty, so clearing the last spot
            # of the CURRENT hybe/channel still reaches disk.
            by_slice.setdefault(want, by_slice.get(want, []))

        for (modality, hybe, channel), spots in by_slice.items():
            path = self._storage_path_for_modality(modality)
            if not path or not modality or not hybe:
                continue
            vlinks_store.write_spots(path, fov, modality, hybe, channel, spots)
            # Mirror the exact written slice into the permanent tier, so
            # "what was saved" is answerable in memory and Revert never
            # re-reads disk.
            self.spot_container_permanent.replace_slice(
                fov, modality, hybe, channel, [deepcopy(sp) for sp in spots])
        return sum(len(v) for v in by_slice.values())

    def _ensure_spot_uids(self, fov, spots):
        """
        Batch-allocate real uids for any spot still carrying uid=0.

        Identity is allocated the moment a spot enters the transient
        container -- never retrofitted at save -- because diff/undo and the
        container's own duplicate check key on uid. One allocator call for
        the whole batch (an h5 attr bump), not one per spot. Any configured
        storage path works: the unified vlinks keeps one per-FOV counter.
        """
        missing = [sp for sp in spots if not int(getattr(sp, 'uid', 0))]
        if not missing:
            return
        any_path = next(iter(self._all_vlinks_storage_paths()), None)
        if any_path is None:
            raise RuntimeError('no storage path configured -- cannot allocate spot uids')
        for sp, uid in zip(missing, vlinks_store.allocate_spot_uids(any_path, fov, len(missing))):
            sp.uid = int(uid)

    def _recast_persisted_spots(self, fov):
        """
        THE post-matrix-write spot recast: coordinates are DERIVED data
        (coordinate = f(raw, current matrices)), so every door that
        changes matrices for a FOV must recast that FOV's spots -- all of
        them, on disk, not just whatever happens to be staged in this
        session. Per confirmed real bug: accepting alignment (including
        auto-save-all across every FOV) used _reassign+_persist, both of
        which no-op for a FOV whose spots were never staged, so persisted
        spots kept their pre-alignment coordinates until a manual spot
        save happened to touch that FOV.

        Staged FOV: the session containers are authoritative -- the
        existing in-session reassign+persist runs unchanged. Unstaged
        FOV: a DISK-SCOPED pass -- cells and spots are hydrated straight
        from vlinks.h5, ownership+coordinates recomputed through the SAME
        resolver-backed transform, and spots written back per slice.
        Deliberately never staged into the session containers (this is a
        persistence refresh, not a view activation). Writes SPOTS only.
        """
        if fov in self._spot_loaded_fovs:
            self._reassign_fov_spots(fov)
            self._persist_fov_spots(fov)
            return
        any_path = next(iter(self._all_vlinks_storage_paths()), None)
        if not any_path:
            return
        dicts = vlinks_store.read_spots(any_path, fov)
        if not dicts:
            return
        spots = []
        for d in dicts:
            spot = ASpot()
            spot.set_metadata(**d)
            spots.append(spot)
        cell_dicts, cells_modality = vlinks_store.read_cells(any_path, fov)
        cells = (CellContainer.load({fov: cell_dicts}, modality=cells_modality).get_cells(fov)
                 if cell_dicts else [])
        cells_by_id = {c.id: c for c in cells}

        def to_shared(hybe, modality, owner):
            # FULL 3D: (H, dz) -- the z leg (cell dz + cross-modal z
            # drift) was silently dropped before, so coordinate[2] never
            # left raw[2] (confirmed real: cell dz=-2 with spot z
            # unchanged through every save).
            fallback = owner.reference_modality if owner is not None else None
            m = modality or fallback
            resolver = self._frame_resolver(owner, fov)
            if resolver.shared is None:
                return None
            return resolver.to_shared(hybe, m, owner), resolver.z_to_shared(hybe, m, owner)

        if cells:
            assignment.assign_spots(spots, cells, cells[0].frame_shape, cells_by_id,
                                    matrix_to_shared=to_shared)
        else:
            # no cells on disk: ownership stays as stored (-1 by
            # definition), coordinates still recast through the FOV/
            # cross-modal legs
            assignment.recast_spots_to_shared(spots, to_shared, cells_by_id)

        by_slice = {}
        for spot in spots:
            by_slice.setdefault((spot.modality, spot.hybe, int(spot.channel)), []).append(spot)
        for (modality, hybe, channel), slice_spots in by_slice.items():
            path = self._storage_path_for_modality(modality) or any_path
            vlinks_store.write_spots(path, fov, modality, hybe, channel, slice_spots)

    def _reassign_fov_spots(self, fov):
        """
        Recompute ownership for EVERY spot in this FOV, not just the
        unassigned ones.

        Assignment has to consider spots that already have a cell: cells get
        deleted, resegmented, or replaced by differently-shaped ones, and a
        spot whose owner no longer contains it must lose that owner. Walking
        only the unassigned pool -- what this replaces -- could never notice,
        so a stale owner survived indefinitely.

        The mask is built from the cells themselves, in each SPOT's own
        (hybe, modality) frame, rather than read from
        self.cell_displayer.mask. That widget only held a mask for the FOV
        just segmented this session, so the old path silently assigned
        nothing whenever you had not just re-segmented.
        """
        cells = self.cell_container.get_cells(fov) if self.cell_container else []
        spots = list(self.spot_container.all(fov))
        if not spots:
            return 0, 0

        def to_shared(hybe, modality, owner):
            # owner=None is a real case (unassigned spots): the resolver's
            # cell leg defaults to identity, FOV/cross-modal still compose.
            # FULL 3D: (H, dz) -- see _recast_persisted_spots' twin above.
            fallback = owner.reference_modality if owner is not None else None
            m = modality or fallback
            resolver = self._frame_resolver(owner, fov)
            if resolver.shared is None:
                return None
            return resolver.to_shared(hybe, m, owner), resolver.z_to_shared(hybe, m, owner)

        # ONE uniform flow, cell existence never required (per explicit
        # spec): (1) label mask in each spot's own frame -- ALL-ZEROS when
        # no cells are loaded; (2) mask[int(y), int(x)] sets cell (and
        # celltype from the owner) -- so with zero cells every spot
        # becomes -1, the transient container being AUTHORITATIVE at save
        # exactly as it is for cell writes; (3) coordinates recast to the
        # shared frame regardless -- through the owner's cell leg when
        # assigned, through modality alone when not. The earlier version
        # returned early with no cells (confirmed real bug: a save in
        # such a session wrote coordinate == raw_coordinate for every
        # spot and touched no ownership).
        cells_by_id = {c.id: c for c in cells}
        # frame_shape only sizes the label mask; with zero cells the mask
        # is all zeros and every lookup lands on "no owner" regardless of
        # shape, so the placeholder is exact, not approximate.
        frame_shape = cells[0].frame_shape if cells else (1, 1)

        def area_in_frame(cell, hybe, modality):
            # Resolver-backed projection -- the library default
            # (cell.get_area_in_readout) raises on residual-form matrices
            # and would silently DROP every aligned cell from the mask.
            return self._cell_area_in_readout(cell, hybe, modality, fov)

        n_assigned, n_unassigned = assignment.assign_spots(
            spots, cells, frame_shape, cells_by_id, matrix_to_shared=to_shared,
            area_in_frame=area_in_frame)

        # No redistribution: assignment wrote each spot's `cell` in place
        # and the spot never leaves the container -- that field IS the
        # whole ownership model. uids only need ensuring for any spot that
        # entered by a door that predates allocation (none should remain).
        self._ensure_spot_uids(fov, spots)
        return n_assigned, n_unassigned

    def _begin_cell_edit(self, fov):
        if self.cell_container is None:
            return None
        self.cell_undo.container = self.cell_container
        return self.cell_container.fingerprint(fov)

    def _commit_cell_edit(self, fov, fp):
        if fp is None or self.cell_container is None:
            return
        self.cell_undo.container = self.cell_container
        self.cell_undo.push(fov, fp)
        self._update_cell_undo_buttons()

    def _update_cell_undo_buttons(self):
        d = self.cell_displayer
        d.UndoPushButton.setEnabled(self.cell_undo.can_undo())
        d.RedoPushButton.setEnabled(self.cell_undo.can_redo())

    def _refresh_cell_displayer_from_container(self, fov):
        """
        Re-render the segmentation displayer FROM the transient container --
        never by mutating the displayer's own mask array. The displayed
        raster is derived state; deriving it fresh after every container
        change is what makes a stale contour (a removed cell's outline
        surviving on screen) structurally impossible rather than a bug to
        chase per edit path.
        """
        cells = self.cell_container.get_cells(fov) if self.cell_container else []
        # Same display context the view was loaded with (falling back to
        # the panel's live selection), same already-displayed reference
        # image, and THE same renderer as the initial load -- this used to
        # rasterize each cell's RAW area (unprojected -- subtly wrong
        # whenever the display frame differs from a cell's native frame),
        # silently skip set_data when its own image re-read failed, and
        # skip entirely on an empty container (removing the LAST cell left
        # its contour on screen).
        ctx = self._last_segment_context or {}
        cp = self.ui.CellSegmentPanel
        reference_hybe = ctx.get('reference_hybe') or cp.current_reference_hybe()
        modality = ctx.get('modality') or cp.current_reference_modality()
        reference_image = self.cell_displayer.reference_image
        if reference_image is None or not reference_hybe:
            return
        self._render_cell_displayer(fov, cells, reference_hybe, modality, reference_image)
        self._refresh_spot_cell_list()

    def _undo_cell_action(self):
        self.cell_undo.container = self.cell_container
        fov = self.cell_undo.undo()
        if fov is not None:
            self._refresh_cell_displayer_from_container(fov)
            self.ui.CellSegmentPanel.LogTextEdit.append('Undo: cell state restored.')
        self._update_cell_undo_buttons()

    def _redo_cell_action(self):
        self.cell_undo.container = self.cell_container
        fov = self.cell_undo.redo()
        if fov is not None:
            self._refresh_cell_displayer_from_container(fov)
            self.ui.CellSegmentPanel.LogTextEdit.append('Redo: cell state restored.')
        self._update_cell_undo_buttons()

    def _find_cell_by_id(self, fov, cell_id):
        if self.cell_container is None:
            return None
        return next((c for c in self.cell_container.get_cells(fov) if c.id == cell_id), None)

    def _begin_spot_edit(self, fov):
        """Fingerprint the FOV's transient spots BEFORE an edit; pair with
        _commit_spot_edit after it. The pair replaces snapshot-taking: the
        undo stack stores only the invertible diff between the two."""
        return self.spot_container.fingerprint(fov)

    def _commit_spot_edit(self, fov, fp):
        self.spot_undo.push(fov, fp)
        self._update_undo_redo_buttons()

    def _undo_spot_action(self):
        if self.spot_undo.undo() is not None:
            self._after_spot_undo_redo('Undo')

    def _redo_spot_action(self):
        if self.spot_undo.redo() is not None:
            self._after_spot_undo_redo('Redo')

    def _after_spot_undo_redo(self, label, missing=None):
        sp = self.ui.SpotLocalizationPanel
        self._update_undo_redo_buttons()
        self._refresh_spot_cell_list()
        if missing:
            sp.LogTextEdit.append(f'{label}: cell(s) {missing} no longer exist -- their spot state could not be restored.')
        if sp.current_view() == 'cell':
            self._load_spot_crop_for_display()
        else:
            self._load_fov_spot_display()
        sp.LogTextEdit.append(f'{label}: spot state restored.')

    def _update_undo_redo_buttons(self):
        sp = self.ui.SpotLocalizationPanel
        sp.UndoPushButton.setEnabled(self.spot_undo.can_undo())
        sp.RedoPushButton.setEnabled(self.spot_undo.can_redo())

    def _replace_cell_spots(self, cell, hybe, channel, new_spots, append=False):
        """
        Full-replace this cell's spots for exactly (hybe, channel) -- mirrors
        CellClassifier's own "rerunning localization on a cell/FOV only
        replaces that scope's spots" semantics, and keeps manual-edit
        reconciliation simple (spots_edited always hands back the FULL
        current crop-local point list, so a plain replace can't double-count).
        append=True (Append Mode) skips the clear step, so new_spots joins
        whatever's already there for this (hybe, channel) instead of
        replacing it -- manual-edit callers never pass append=True, since
        their point list already IS the full current state.
        """
        fov = cell.fov
        if not append:
            doomed = [s.uid for s in self.spot_container.of_cell(fov, cell.id)
                      if s.hybe == hybe and s.channel == channel]
            self.spot_container.remove(fov, doomed)
        self._ensure_spot_uids(fov, new_spots)
        for spot in new_spots:
            spot.cell = int(cell.id)
            self.spot_container.add(fov, spot)

    def _replace_fov_unassigned_spots(self, storage_path, fov, hybe, channel, new_spots, append=False):
        """
        Full-replace, for exactly (hybe, channel), the FOV-level
        unassigned-spot pool -- same replace-not-append semantics (and
        the same append=True escape hatch) as _replace_cell_spots, just
        keyed by (storage_path, fov) instead of living on a cell (there
        is no owning cell for these by definition -- see
        _reassign_fov_spots for when that gets decided).
        """
        if not append:
            doomed = [s.uid for s in self.spot_container.unassigned(fov)
                      if s.hybe == hybe and s.channel == channel]
            self.spot_container.remove(fov, doomed)
        # Identity at the entry boundary: manual/auto FOV-view spots arrive
        # with uid=0 and get real uids HERE, not at save.
        self._ensure_spot_uids(fov, new_spots)
        for spot in new_spots:
            spot.modality = spot.modality or vlinks_store.modality_of(storage_path)
            self.spot_container.add(fov, spot)

    def _on_spot_crop_edited(self, points):
        """
        Handles both views' manual clicks/removals -- the crop
        displayer's spots_edited signal always hands back the FULL
        current crop-local point list, regardless of which view
        (_spot_crop_context['kind']) is currently open. Cell view builds
        real cell-owned ASpots (cell=cell.id, coordinate via cell.matrix_to);
        FOV view builds unassigned ASpots (cell stays at its -1 default,
        coordinate mapped via the FOV/cross-modal chain alone, same as
        FOV-view auto-detect -- see spot_mapper.raw_to_reference's own
        cell=None docstring) --
        identification is deferred to Save Current Spots either way,
        matching FOV auto-detect's own deferred design.
        """
        ctx = self._spot_crop_context
        if ctx is None:
            return
        hybe, channel = ctx['hybe'], ctx['channel']
        rxmin, rymin = ctx['rxmin'], ctx['rymin']
        img = self.spot_crop_displayer.crop_image
        if ctx['kind'] != 'cell':
            fov_matrices = self._composed_fov_matrices_for_cell_alignment(ctx['storage_path'], ctx['fov'])
        new_spots = []
        for x, y in points:
            raw_x, raw_y = x + rxmin, y + rymin
            iy, ix = int(round(y)), int(round(x))
            brightness = 0.0
            if img is not None and 0 <= iy < img.shape[0] and 0 <= ix < img.shape[1]:
                val = img[iy, ix]
                brightness = float(val) if np.isfinite(val) else 0.0
            spot = ASpot()
            # Never '' -- an empty modality resolves NO frame, so every
            # later recast silently degrades to identity for this spot.
            spot.modality = ctx.get('modality') or vlinks_store.modality_of(ctx['storage_path'])
            if ctx['kind'] == 'cell':
                cell = ctx['cell']
                # _matrix_to_shared (not spot_mapper.raw_to_reference's own
                # cell branch) -- per confirmed real bug, cell.matrix_to_
                # shared silently collapses to identity when this cell has
                # no real cell-level alignment for this hybe/modality yet;
                # _build_cell_display_crop now falls back to the live
                # FOV/cross-modal matrix to build this crop in the first
                # place (see its own docstring), so a crop existing no
                # longer guarantees a real cell.matrices entry the way it
                # used to -- resolve the same way here instead of assuming.
                H_shared = self._matrix_to_shared(hybe, ctx['modality'], cell, cell.fov)
                if H_shared is not None:
                    cy, cx, _ = H_shared @ np.array([raw_y, raw_x, 1.0])
                else:
                    cy, cx = float(raw_y), float(raw_x)
                spot.set_metadata(fov=cell.fov, hybe=hybe, channel=channel, cell=cell.id,
                                  coordinate=(cy, cx, 0.0), raw_coordinate=(raw_y, raw_x, 0.0),
                                  size=0.0, brightness=brightness)
            else:
                # No owning cell yet, but still mapped into the SHARED
                # frame via the FOV/cross-modal chain alone (identity only
                # for the missing cell-level residual) -- see spot_mapper.
                # raw_to_reference's own cell=None docstring. Falls back to
                # raw==coordinate only when this hybe truly has no FOV-level
                # matrix yet (Same-Modality Alignment never run for it).
                if (hybe, fov_matrices.modality) in fov_matrices:
                    cy, cx = spot_mapper.raw_to_reference((raw_y, raw_x), hybe, fov_matrices,
                                                          modality=ctx['modality'], cell=None)
                else:
                    cy, cx = float(raw_y), float(raw_x)
                spot.set_metadata(fov=ctx['fov'], hybe=hybe, channel=channel,
                                  coordinate=(cy, cx, 0.0), raw_coordinate=(raw_y, raw_x, 0.0),
                                  size=0.0, brightness=brightness)
            new_spots.append(spot)

        sp = self.ui.SpotLocalizationPanel
        if ctx['kind'] == 'cell':
            cell = ctx['cell']
            fp = self._begin_spot_edit(cell.fov)
            self._replace_cell_spots(cell, hybe, channel, new_spots)
            self._commit_spot_edit(cell.fov, fp)
            sp.LogTextEdit.append(f'Cell {cell.id}, {hybe} ch{channel}: {len(new_spots)} spot(s) after manual edit.')
        else:
            fp = self._begin_spot_edit(ctx['fov'])
            self._replace_fov_unassigned_spots(ctx['storage_path'], ctx['fov'], hybe, channel, new_spots)
            self._commit_spot_edit(ctx['fov'], fp)
            sp.LogTextEdit.append(f'FOV{ctx["fov"]:02d}, {hybe} ch{channel}: {len(new_spots)} unassigned '
                                  f'spot(s) after manual edit.')
        self._refresh_spot_cell_list()
        # re-derives global display indices and the 3D-localization
        # popup's spot list from the just-mutated state -- without this,
        # a freshly manually-added point would show a local fallback
        # number instead of its real global one until some LATER action
        # happened to trigger a refresh (see SpotCropDisplayer's own
        # spot_indices fallback). keep_view=True: this is a redraw of the
        # SAME view after an in-place edit, not a view switch -- per
        # confirmed real bug, the previous unconditional reset snapped
        # the zoom/pan back to full-frame after every single manual
        # click, making it impossible to place several spots precisely
        # while zoomed in.
        if ctx['kind'] == 'cell':
            self._load_spot_crop_for_display(keep_view=True)
        else:
            self._load_fov_spot_display(keep_view=True)

    def _on_readonly_spot_removed(self, cell_id, x, y):
        """
        A cell-owned spot was removed via the FOV-view crop displayer's
        readonly (red) list -- by index or right-click, see
        SpotCropDisplayer.readonly_point_removed. tag is that spot's
        owning cell.id (see _load_fov_spot_display, which is the only
        place readonly_points is built); x/y are its raw coordinates
        (FOV view has no crop offset, so crop-local == raw here). Finds
        the exact matching spot on that cell (hybe/channel/raw_coordinate)
        and removes it -- same in-memory-only, Save-Current-Spots-to-
        persist convention every other edit in this panel already
        follows; this does NOT touch vlinks.h5 by itself.
        """
        ctx = self._spot_crop_context
        if ctx is None or ctx['kind'] != 'fov' or self.cell_container is None:
            return
        fov = ctx['fov']
        hybe, channel = ctx['hybe'], ctx['channel']
        cell = next((c for c in self.cell_container.get_cells(fov) if c.id == cell_id), None)
        if cell is None:
            return
        match = next((s for s in self.spot_container.of_cell(fov, cell.id)
                     if s.hybe == hybe and s.channel == channel
                     and abs(s.raw_coordinate[0] - y) < 0.5 and abs(s.raw_coordinate[1] - x) < 0.5), None)
        if match is None:
            return
        fp = self.spot_container.fingerprint(fov)
        self.spot_container.remove(fov, [match.uid])
        self.spot_undo.push(fov, fp)
        self._update_undo_redo_buttons()
        sp = self.ui.SpotLocalizationPanel
        # in-memory only, same "Save Current Spots to persist" convention
        # as every other edit here -- Save Current Spots now covers this
        # cell too regardless of which view is open (see its own
        # docstring), no need to switch back to Cell {cell.id}'s own view
        # first the way the old view-scoped Save View required.
        sp.LogTextEdit.append(f'Cell {cell.id}, {hybe} ch{channel}: removed 1 spot from FOV view '
                              f'(not yet saved -- click Save Current Spots to persist).')
        self._refresh_spot_cell_list()
        self._load_fov_spot_display(keep_view=True)  # re-derives global indices + 3D-localization popup list

    def _run_spot_auto_detect(self):
        sp = self.ui.SpotLocalizationPanel
        modality = sp.current_hybe_modality()
        storage_path = self._storage_path_for_modality(modality)
        fov = self._current_spot_fov()
        hybe = sp.current_hybe_folder()
        channel_text = sp.ChannelComboBox.currentText()
        if not storage_path or fov is None or not hybe or not channel_text:
            QtWidgets.QMessageBox.warning(self, 'Run Auto-Detect',
                                          'Set storage path/FOV in Ingestion, and pick a hybe/channel here.')
            return
        channel = int(channel_text)
        min_distance = sp.MinDistanceSpinBox.value()
        pad = sp.PadSpinBox.value()
        try:
            self._run_spot_auto_detect_body(sp, storage_path, fov, hybe, modality, channel, min_distance, pad)
        except ValueError as e:
            # threshold_abs()'s own parse errors -- bad text in the
            # threshold fields, a real user-input mistake, not a bug.
            QtWidgets.QMessageBox.warning(self, 'Run Auto-Detect', str(e))
        except Exception as e:
            # this method had no error handling at all -- given the same
            # silent-failure pattern was confirmed real elsewhere (a
            # reproducible KeyError in celltype determination that crashed
            # with zero dialog/log/feedback), an unhandled exception here
            # would look identical to "nothing happened" -- exactly what
            # was reported ("still didn't appear"). Surface it for real.
            QtWidgets.QMessageBox.critical(self, 'Run Auto-Detect error', f'{type(e).__name__}: {e}')

    def _run_spot_auto_detect_body(self, sp, storage_path, fov, hybe, modality, channel, min_distance, pad):
        append = sp.AppendModeCheckBox.isChecked()
        if sp.current_view() == 'cell':
            cell = self._selected_spot_cell()
            if cell is None:
                QtWidgets.QMessageBox.warning(self, 'Run Auto-Detect', 'Select a cell first (Cell view).')
                return
            fov_matrices = self._fov_matrices_for_cell_modality(modality, cell, fov)
            crop = localization._build_cell_crop(cell, hybe, channel, storage_path, fov, pad, modality=modality,
                                                 fov_matrices=fov_matrices, resolver=self._frame_resolver(cell, fov))
            if crop is None:
                QtWidgets.QMessageBox.warning(self, 'Run Auto-Detect',
                                              f'Cell {cell.id}: no crop for {hybe} -- the hybe has no image '
                                              f'data for this FOV, or the cell mask projects outside its '
                                              f'frame. (Alignment is NOT required.)')
                return
            img, rxmin, rymin = crop['img'], crop['rxmin'], crop['rymin']
            threshold_abs = sp.threshold_abs(np.nanmax(img))
            coords = peak_local_max(img, min_distance=min_distance, exclude_border=1, threshold_abs=threshold_abs)
            # H resolved once, outside the loop -- per confirmed real bug,
            # spot_mapper.raw_to_reference(..., cell=cell) has no fallback
            # of its own (cell.matrix_to_shared silently collapses to
            # identity when this cell has no real cell-level alignment for
            # this hybe/modality); _matrix_to_shared already has the
            # correct FOV/cross-modal fallback (same one crop above just
            # used), so resolve through it directly instead.
            H = self._matrix_to_shared(hybe, modality, cell, fov)
            new_spots = []
            for y, x in coords:
                raw_x, raw_y = int(x) + rxmin, int(y) + rymin
                if H is not None:
                    cy, cx, _ = H @ np.array([raw_y, raw_x, 1.0])
                else:
                    cy, cx = float(raw_y), float(raw_x)
                spot = ASpot()
                # Never '' -- see the manual-click door's own comment.
                spot.modality = modality or vlinks_store.modality_of(storage_path)
                spot.set_metadata(fov=fov, hybe=hybe, channel=channel, cell=cell.id,
                                  coordinate=(cy, cx, 0.0), raw_coordinate=(raw_y, raw_x, 0.0),
                                  size=0.0, brightness=float(img[y, x]))
                new_spots.append(spot)
            fp = self._begin_spot_edit(cell.fov)
            self._replace_cell_spots(cell, hybe, channel, new_spots, append=append)
            self._commit_spot_edit(cell.fov, fp)
            sp.LogTextEdit.append(f'Cell {cell.id}, {hybe} ch{channel}: {len(new_spots)} spot(s) detected '
                                  f'(Cell view{", appended" if append else ""}).')
            self._load_spot_crop_for_display()
            self._refresh_spot_cell_list()
        else:
            # FOV view: detection here no longer needs cell segmentation,
            # a matching mask, or even FOV alignment for this hybe -- it's
            # a plain peak search over the raw MIP, full stop. Every
            # detected peak becomes an unassigned spot (ASpot.cell stays
            # at its model default, -1) in fov_unassigned_spots; cell
            # ownership is only decided later, at save time (see
            # _reassign_fov_spots) -- per explicit request,
            # identification is deferred to save, not done eagerly at
            # detect time.
            mip = vlinks_store.read_hybe_mip(storage_path, fov, hybe, channel)
            if mip is None:
                sp.LogTextEdit.append(f'{hybe} ch{channel} not in vlinks.h5 for FOV{fov:02d} -- ingest it first.')
                return
            threshold_abs = sp.threshold_abs(mip.max())
            coords = peak_local_max(mip, min_distance=min_distance, exclude_border=1, threshold_abs=threshold_abs)
            # No owning cell yet -- see _on_spot_crop_edited's matching FOV
            # branch: still mapped into the shared frame via the FOV/cross-
            # modal chain, identity only for the missing cell-level residual.
            fov_matrices = self._composed_fov_matrices_for_cell_alignment(storage_path, fov)
            new_spots = []
            for y, x in coords:
                raw_x, raw_y = int(x), int(y)
                if (hybe, fov_matrices.modality) in fov_matrices:
                    cy, cx = spot_mapper.raw_to_reference((raw_y, raw_x), hybe, fov_matrices, modality=modality, cell=None)
                else:
                    cy, cx = float(raw_y), float(raw_x)
                spot = ASpot()
                # Never '' -- see the manual-click door's own comment.
                spot.modality = modality or vlinks_store.modality_of(storage_path)
                spot.set_metadata(fov=fov, hybe=hybe, channel=channel,
                                  coordinate=(cy, cx, 0.0), raw_coordinate=(raw_y, raw_x, 0.0),
                                  size=0.0, brightness=float(mip[y, x]))
                new_spots.append(spot)
            fp = self._begin_spot_edit(fov)
            self._replace_fov_unassigned_spots(storage_path, fov, hybe, channel, new_spots, append=append)
            self._commit_spot_edit(fov, fp)
            sp.LogTextEdit.append(f'FOV{fov:02d} {hybe} ch{channel}: {len(new_spots)} peak(s) detected '
                                  f'(unassigned{", appended" if append else ""} -- run Save Current Spots to identify cell ownership).')
            self._refresh_spot_cell_list()
            self._load_fov_spot_display()
            self.spot_crop_displayer.show()
            self.spot_crop_displayer.raise_()

    # -- chromatin tracing --
    #
    # An allele's (x,y) is already known -- built from whatever's currently
    # SELECTED in Spot Localization (_build_chromatin_alleles_from_
    # selection reuses _resolve_selected_3d_targets exactly as-is, the same
    # selection 3D Localization's own Run/View already act on). Building/
    # previewing (View Crop) stay in-memory only; Fit All FOVs is the one
    # action that persists (mirror_write_fov_alleles), same "explicit Save
    # step" convention Spot Localization's own Save Current Spots follows.

    @staticmethod
    def _default_chromatin_tracing_hybes(record, modality):
        """
        Default-checked state for the Hybes Involved list: modality=='DNA'
        and datatype in ('H','R','T') -- H is the main genomic-locus
        hybridization round, R/T are repeat/toehold QC rounds; B (barcode/
        cell-identity rounds) is deliberately excluded, per explicit
        request. Confirmed against this repo's own real ExperimentLayout.
        xlsx files (DNA: 90xH/8xR/4xB/1xT; RNA also carries H-type rows, so
        the modality check matters -- datatype alone isn't enough to keep
        RNA's own H rounds out).
        """
        return modality == 'DNA' and record.get('datatype') in ('H', 'R', 'T')

    @staticmethod
    def _chromatin_channel_params(full_params):
        """
        (fiducial_params, readout_params) ready for localization.
        build_chromatin_trace_allele's own fiducial_params/readout_params
        kwargs, from ChromatinTracingPanel.params()'s own {'fiducial': {...
        }, 'readout': {..., 'multi_mode':}} shape. fiducial_params passes
        straight through unchanged (no mixture mode -- _localize_fiducial_
        hybe has no use_mixture parameter at all, per explicit request);
        readout_params renames its own 'multi_mode' key to 'use_mixture'
        (localization.py's own name for it, matching _localize_readout_
        hybe's own parameter).
        """
        readout_params = dict(full_params['readout'])
        readout_params['use_mixture'] = readout_params.pop('multi_mode')
        return dict(full_params['fiducial']), readout_params

    def _chromatin_tracing_context(self):
        """
        Validates + resolves the Hybes Involved checklist into (hybes,
        hybe_fiducial_channels, hybe_readout_channels, modality,
        storage_path, reference_hybe) -- shared by Build Alleles/View Crop/
        Fit All FOVs so the three can never silently disagree about which
        hybes/modality/reference/channels are in play. Returns None (after
        a warning dialog) if anything required is missing.
        """
        chp = self.ui.ChromatinTracingPanel
        checked = chp.checked_hybes()
        if not checked:
            QtWidgets.QMessageBox.warning(self, 'Chromatin Tracing', 'Check at least one hybe first.')
            return None
        modalities = {m for _, m in checked}
        if len(modalities) > 1:
            QtWidgets.QMessageBox.warning(self, 'Chromatin Tracing',
                                          'Checked hybes span more than one modality -- pick hybes from a single modality.')
            return None
        modality = modalities.pop()
        storage_path = self._storage_path_for_modality(modality)
        if not storage_path:
            QtWidgets.QMessageBox.warning(self, 'Chromatin Tracing', f'No storage path configured for {modality}.')
            return None
        reference_hybe = chp.current_reference_hybe()
        if not reference_hybe:
            QtWidgets.QMessageBox.warning(self, 'Chromatin Tracing', 'Pick a reference hybe first.')
            return None
        hybes = [folder for folder, m in checked]
        if reference_hybe not in hybes:
            QtWidgets.QMessageBox.warning(self, 'Chromatin Tracing',
                                          'Reference hybe must also be checked in Hybes Involved.')
            return None
        records_by_folder = {r['folder']: r for r in self._hybe_records_for_storage_path(storage_path)}
        hybe_fiducial_channels = {folder: records_by_folder[folder]['fiducial_channel']
                                  for folder in hybes if folder in records_by_folder}
        # The seed spot's own channel only LOCATES the allele-frame (x, y,
        # z) -- it never determines which channel gets traced (per
        # explicit correction: seeding from a fiducial-channel spot is
        # legitimate and explicitly supported, "no need to be fiducial
        # channel" was never "must not be"). Each hybe's own readout
        # channel is instead whichever of its real channels ISN'T the
        # fiducial one -- independent of allele.anchor_channel entirely.
        hybe_readout_channels = {}
        for folder in hybes:
            record = records_by_folder.get(folder)
            if record is None:
                continue
            fiducial_channel = record['fiducial_channel']
            readout = next((c for c in record.get('channels', []) if c != fiducial_channel), None)
            if readout is not None:
                hybe_readout_channels[folder] = readout
        return hybes, hybe_fiducial_channels, hybe_readout_channels, modality, storage_path, reference_hybe

    def _chromatin_storage_path_and_modality(self):
        """
        (modality, storage_path) resolved from the Hybes Involved
        checklist alone -- lighter than _chromatin_tracing_context (no
        reference-hybe requirement, no warning dialogs), used just to
        scope the Alleles section's own Hybe/Channel/spot choices. Returns
        (None, None) if the checked hybes don't yet resolve to exactly one
        modality/storage_path.
        """
        checked = self.ui.ChromatinTracingPanel.checked_hybes()
        modalities = {m for _, m in checked}
        if len(modalities) != 1:
            return None, None
        modality = modalities.pop()
        return modality, self._storage_path_for_modality(modality)

    def _refresh_chromatin_allele_hybe_choices(self):
        """
        Populates the Alleles section's own Hybe combobox from every
        active hybe in the SAME modality/storage_path as the Hybes
        Involved checklist -- not restricted to the checked subset itself
        (a seed spot's own anchor hybe doesn't need to be one of the
        traced rounds). Cascades into Channel + spot choices.
        """
        chp = self.ui.ChromatinTracingPanel
        modality, storage_path = self._chromatin_storage_path_and_modality()
        if not storage_path:
            chp.populate_allele_hybe_choices([])
            self._on_chromatin_allele_hybe_changed()
            return
        hybe_records = self._hybe_records_for_storage_path(storage_path)
        chp.populate_allele_hybe_choices([(r, modality) for r in hybe_records])
        self._on_chromatin_allele_hybe_changed()

    def _on_chromatin_allele_hybe_changed(self):
        chp = self.ui.ChromatinTracingPanel
        record, _modality = chp.current_allele_hybe_record_and_modality()
        chp.populate_allele_channel_choices(record)
        self._refresh_chromatin_allele_spot_choices()

    def _refresh_chromatin_allele_spot_choices(self):
        """Populates the Alleles section's spot-selection list for the
        currently picked (FOV, hybe, channel), read straight from
        vlinks.h5 (_ordered_spot_dicts_for_scope) -- same real-data source
        CellSpotStatusDisplayer already uses, independent of whatever
        Spot Localization's own live session state currently shows."""
        chp = self.ui.ChromatinTracingPanel
        _modality, storage_path = self._chromatin_storage_path_and_modality()
        fov = chp.AlleleFovSpinBox.value()
        hybe, _ = chp.current_allele_hybe_key()
        channel = chp.current_allele_channel()
        if not storage_path or not hybe or channel is None:
            chp.populate_spot_choices([])
            return
        chp.populate_spot_choices(self._ordered_spot_dicts_for_scope(storage_path, fov, hybe, channel))

    def _build_chromatin_alleles_from_selection(self):
        """
        Turns whatever's currently SELECTED in this panel's OWN spot list
        (Alleles section -- scoped by its own FOV/Hybe/Channel pickers,
        never Spot Localization's live session state, see class docstring
        on ui/chromatin_tracing_panel.py) into this FOV's allele list --
        full replace (same "re-run overwrites" convention as _replace_
        cell_spots/_replace_fov_unassigned_spots elsewhere in this app),
        never an incremental merge, so clicking this again after changing
        the selection can't leave stale alleles from a previous click
        mixed in. Builds directly from the persisted spot dicts (id/cell/
        hybe/channel/coordinate/raw_coordinate) -- no live ASpot/ACell
        needed at build time; the owning cell is resolved later, lazily,
        wherever a real fit actually needs it (View Crop/Fit All FOVs).
        """
        chp = self.ui.ChromatinTracingPanel
        _modality, storage_path = self._chromatin_storage_path_and_modality()
        if not storage_path:
            QtWidgets.QMessageBox.warning(self, 'Chromatin Tracing', 'Check at least one hybe (Hybes Involved) first.')
            return
        fov = chp.AlleleFovSpinBox.value()
        selected = chp.selected_spot_dicts()
        if not selected:
            QtWidgets.QMessageBox.warning(self, 'Chromatin Tracing', 'Select at least one spot first.')
            return
        alleles = []
        for i, d in enumerate(selected, start=1):
            allele = AnAllele()
            allele.set_metadata(id=i, fov=fov, cell=d['cell'], anchor_uid=d.get('uid', 0),
                                anchor_hybe=d['hybe'], anchor_channel=d['channel'],
                                coordinate=d['coordinate'], raw_coordinate=d['raw_coordinate'])
            alleles.append(allele)
        self.chromatin_alleles[(storage_path, fov)] = alleles
        self._refresh_chromatin_allele_lists(storage_path, fov)
        chp.StatusLabel.setText(f'Built {len(alleles)} allele(s) for FOV{fov:02d} from {len(selected)} selected spot(s).')

    def _refresh_chromatin_allele_lists(self, storage_path, fov):
        chp = self.ui.ChromatinTracingPanel
        alleles = self.chromatin_alleles.get((storage_path, fov))
        if alleles is None:
            # Stage persisted alleles once per (storage_path, fov) --
            # vlinks is the authoritative container (rule 4), and cells/
            # spots already stage this way at _activate_fov. Without this,
            # a fresh session's tracing panel showed an empty list even
            # though the store held real alleles, and only a re-Build
            # could bring them back.
            alleles = []
            for d in vlinks_store.read_fov_alleles(storage_path, fov) or []:
                a = AnAllele()
                a.set_metadata(**d)
                alleles.append(a)
            self.chromatin_alleles[(storage_path, fov)] = alleles
        rows = [(a.id, f"Allele {a.id}: cell={'unassigned' if a.cell == -1 else a.cell} "
                       f"anchor={a.anchor_hybe}/{a.anchor_channel} @ "
                       f"({a.coordinate[0]:.1f}, {a.coordinate[1]:.1f}, {a.coordinate[2]:.1f}) "
                       f"[{len(a.polymer)} hybe(s) traced]")
                for a in alleles]
        chp.populate_allele_list(rows)
        chp.populate_preview_allele_choices(rows)

    def _on_chromatin_allele_fov_changed(self):
        chp = self.ui.ChromatinTracingPanel
        self._refresh_chromatin_allele_spot_choices()
        _modality, storage_path = self._chromatin_storage_path_and_modality()
        if not storage_path:
            chp.populate_allele_list([])
            chp.populate_preview_allele_choices([])
            return
        self._refresh_chromatin_allele_lists(storage_path, chp.AlleleFovSpinBox.value())

    def _view_chromatin_trace_crop(self):
        """
        Runs build_chromatin_trace_allele(collect_debug=True) for the ONE
        allele currently picked in Preview One Allele, with the CURRENT fit
        parameters -- a real fit, same computation Fit All FOVs would do
        for this allele, just scoped to one allele and not yet persisted
        to disk (Fit All FOVs is the one action that writes to vlinks.h5).
        Populates both grid pop-ups, one tile per active hybe.
        """
        chp = self.ui.ChromatinTracingPanel
        ctx = self._chromatin_tracing_context()
        if ctx is None:
            return
        hybes, hybe_fiducial_channels, hybe_readout_channels, modality, storage_path, reference_hybe = ctx
        fov = chp.AlleleFovSpinBox.value()
        allele_id = chp.current_preview_allele_id()
        if allele_id is None:
            QtWidgets.QMessageBox.warning(self, 'Chromatin Tracing', 'Build alleles for this FOV first.')
            return
        alleles = self.chromatin_alleles.get((storage_path, fov), [])
        allele = next((a for a in alleles if a.id == allele_id), None)
        if allele is None:
            return
        # Alleles are built straight from persisted spot dicts (see
        # _build_chromatin_alleles_from_selection), never from a live
        # ASpot -- self.cell_container may not have this FOV loaded yet
        # this session, so _find_cell_by_id would silently resolve to
        # None even for a real cell-owned allele. Self-activating here
        # matches every other spot-related method's own guard against
        # exactly that (see _refresh_spot_cell_list).
        self._activate_fov(fov)
        # Anchor from the source spot's CURRENT position (by anchor_uid),
        # not the Build-time snapshot -- a 3D refinement moves the spot.
        self._refresh_allele_anchor(allele, fov)
        cell = self._find_cell_by_id(fov, allele.cell) if allele.cell != -1 else None
        fov_matrices = self._composed_fov_matrices_for_cell_alignment(storage_path, fov)

        full_params = chp.params()
        fiducial_params, readout_params = self._chromatin_channel_params(full_params)
        _, debug = localization.build_chromatin_trace_allele(
            allele, hybes, reference_hybe, hybe_fiducial_channels, hybe_readout_channels,
            storage_path, fov, modality, cell, fov_matrices, max_fiducial_drift=full_params['max_fiducial_drift'],
            max_fiducial_drift_z=full_params['max_fiducial_drift_z'],
            spad=full_params['spad'], z_window=full_params['z_window'],
            fiducial_params=fiducial_params, readout_params=readout_params, collect_debug=True,
            resolver=self._frame_resolver(None, fov))

        fid_results, readout_results = [], []
        for hybe in hybes:
            d = debug.get(hybe, {})
            if d.get('fiducial_cubic') is not None:
                centroid = [d['fiducial_centroid']] if d['fiducial_centroid'] is not None else None
                fid_results.append((d['fiducial_cubic'], centroid, hybe))
            if d.get('readout_cubic') is not None:
                readout_results.append((d['readout_cubic'], d['readout_centroids'], hybe))

        allele_label = f'FOV{fov:02d}_allele{allele.id}'
        self.chromatin_fiducial_grid_displayer.show_fit_status_grid(fid_results, allele_label=allele_label, params=full_params)
        self.chromatin_readout_grid_displayer.show_fit_status_grid(readout_results, allele_label=allele_label, params=full_params)
        overlay_entries = self._build_fiducial_overlay_entries(allele, reference_hybe, debug)
        self.chromatin_fiducial_overlay_displayer.show_overlay_grid(
            overlay_entries, allele_label=allele_label, params=full_params)
        self.chromatin_fiducial_grid_displayer.show()
        self.chromatin_fiducial_grid_displayer.raise_()
        self.chromatin_readout_grid_displayer.show()
        self.chromatin_readout_grid_displayer.raise_()
        self.chromatin_fiducial_overlay_displayer.show()
        self.chromatin_fiducial_overlay_displayer.raise_()
        self._refresh_chromatin_allele_lists(storage_path, fov)
        chp.StatusLabel.setText(f'Allele {allele.id}: {len(allele.polymer)}/{len(hybes)} hybe(s) traced '
                                f'({len(allele.rejected_hybes)} rejected).')

    @staticmethod
    def _build_fiducial_overlay_entries(allele, reference_hybe, debug):
        """
        Red-cyan overlay tiles for the Fiducial Overlay popup: per hybe,
        red = reference hybe's fiducial crop (max-projected), cyan = this
        hybe's -- BEFORE as the trace actually cut them (centers already
        carry the modality/cell-residual correction via reference_to_raw)
        and AFTER with the moving crop shifted by the GAUSSIAN-CENTROID
        drift (allele.fiducial_trace[h] - fiducial_trace[reference]) --
        fiducial alignment is centroid matching, never image matching, so
        the applied shift IS the fit result. A hybe whose Gaussian fit
        failed (fiducial_trace None / missing) is omitted, per explicit
        spec; so is the whole overlay when the reference itself has no
        fit, since every drift is measured against it.
        """
        def norm2d(img):
            img = np.asarray(img, dtype=float)
            lo, hi = np.nanquantile(img, 0.3), np.nanquantile(img, 0.999)
            return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)

        def rgb(ref_img, mov_img):
            h = min(ref_img.shape[0], mov_img.shape[0])
            w = min(ref_img.shape[1], mov_img.shape[1])
            out = np.zeros((h, w, 3))
            out[..., 0] = ref_img[:h, :w]
            out[..., 1] = mov_img[:h, :w]
            out[..., 2] = mov_img[:h, :w]
            return out

        def shifted(img, dx_h, dy_v):
            M = np.array([[1.0, 0.0, -dx_h], [0.0, 1.0, -dy_v]], dtype=np.float64)
            return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        Z_PAD = 15   # same z display window the fit-status grids use
        fid = allele.fiducial_trace or {}
        ref_fit = fid.get(reference_hybe)
        ref_cubic = (debug.get(reference_hybe) or {}).get('fiducial_cubic')
        if ref_fit is None or ref_cubic is None:
            return []
        ref_cubic = np.asarray(ref_cubic, dtype=float)
        # ONE absolute z-window (the reference fit's) applied to BOTH
        # sides of every pair: windowing each hybe around its OWN fit
        # would silently re-center the depth axis and hide the very dz
        # drift the ZX row exists to show.
        z0 = max(0, int(round(ref_fit[2])) - Z_PAD)
        z1 = min(ref_cubic.shape[2], int(round(ref_fit[2])) + Z_PAD + 1)
        ref_yx = norm2d(np.nanmax(ref_cubic, axis=2))
        ref_zx = norm2d(np.nanmax(ref_cubic[:, :, z0:z1], axis=0).T)
        entries = []
        for hybe in sorted(fid):
            if hybe == reference_hybe:
                continue
            fit = fid[hybe]
            cubic = (debug.get(hybe) or {}).get('fiducial_cubic')
            if fit is None or cubic is None:
                continue
            cubic = np.asarray(cubic, dtype=float)
            mz1 = min(cubic.shape[2], z1)
            mov_yx = norm2d(np.nanmax(cubic, axis=2))
            mov_zx = norm2d(np.nanmax(cubic[:, :, z0:mz1], axis=0).T)
            dy, dx, dz = fit[0] - ref_fit[0], fit[1] - ref_fit[1], fit[2] - ref_fit[2]
            entries.append((rgb(ref_yx, mov_yx), rgb(ref_yx, shifted(mov_yx, dx, dy)),
                            rgb(ref_zx, mov_zx), rgb(ref_zx, shifted(mov_zx, dx, dz)),
                            f'{hybe}  d=({dx:+.2f},{dy:+.2f},{dz:+.2f})'))
        return entries

    def _refresh_allele_anchor(self, allele, fov):
        """
        Re-derive this allele's anchor coordinates from its source spot's
        CURRENT position (by anchor_uid) -- the anchor fields are a Build-
        time snapshot, and a later 3D refinement legitimately moves the
        spot (confirmed real divergence between an allele and its own
        spot). uid=0 (legacy allele, built before the link existed) or a
        since-removed spot leaves the snapshot as-is.
        """
        uid = int(getattr(allele, 'anchor_uid', 0))
        if not uid or self.spot_container is None:
            return False
        spot = self.spot_container.data.get(fov, {}).get(uid)
        if spot is None or spot.hybe != allele.anchor_hybe:
            return False
        allele.coordinate = tuple(spot.coordinate)
        allele.raw_coordinate = tuple(spot.raw_coordinate)
        return True

    def _run_chromatin_tracing_fit_all(self):
        """
        Batch-fits every FOV in the Ingestion tab's own FOV list that
        already has alleles built (Build/Refresh Alleles per FOV, or a
        previous View Crop -- this never builds alleles itself). Background
        QThread + progress bar, same pattern as AlignmentWorker/
        CellAlignmentWorker/CrossModalAlignmentWorker; persists via
        mirror_write_fov_alleles as soon as the batch completes.
        """
        chp = self.ui.ChromatinTracingPanel
        ctx = self._chromatin_tracing_context()
        if ctx is None:
            return
        hybes, hybe_fiducial_channels, hybe_readout_channels, modality, storage_path, reference_hybe = ctx
        ip = self.ui.IngestionPanel
        fov_list = self._parse_fov_list(ip.FovListLineEdit.text())

        jobs, fov_matrices_by_fov, resolver_by_fov = [], {}, {}
        for fov in fov_list:
            alleles = self.chromatin_alleles.get((storage_path, fov), [])
            if not alleles:
                continue
            # Alleles are built from persisted spot dicts, not a live
            # ASpot -- self.cell_container may not have this FOV loaded
            # yet this session. Activate on the MAIN thread, before the
            # worker starts, so ChromatinTracingWorker's own cell_lookup
            # (_find_cell_by_id) has real data to read from the background
            # thread without touching session state itself.
            self._activate_fov(fov)
            for a in alleles:
                # anchor from the source spot's CURRENT position -- see
                # _refresh_allele_anchor.
                self._refresh_allele_anchor(a, fov)
            jobs.append((storage_path, fov, alleles))
            fov_matrices_by_fov[fov] = self._composed_fov_matrices_for_cell_alignment(storage_path, fov)
            # Built here, on the main thread, for the same reason as
            # fov_matrices_by_fov: _frame_resolver reads Qt line edits.
            resolver_by_fov[fov] = self._frame_resolver(None, fov)
        if not jobs:
            QtWidgets.QMessageBox.warning(self, 'Fit All FOVs',
                                          "No alleles built yet for any FOV in the Ingestion tab's FOV list -- "
                                          "use Build/Refresh Alleles per FOV first.")
            return

        full_params = chp.params()
        fiducial_params, readout_params = self._chromatin_channel_params(full_params)

        chp.ProgressBar.setValue(0)
        chp.FitAllFovsPushButton.setEnabled(False)
        self._chromatin_worker = ChromatinTracingWorker(jobs, hybes, reference_hybe, hybe_fiducial_channels,
                                                         hybe_readout_channels, modality,
                                                         fov_matrices_by_fov, self._find_cell_by_id,
                                                         full_params['max_fiducial_drift'], full_params['spad'],
                                                         full_params['z_window'], fiducial_params, readout_params,
                                                         resolver_by_fov=resolver_by_fov,
                                                         max_fiducial_drift_z=full_params['max_fiducial_drift_z'])
        self._chromatin_worker.progress.connect(self._on_chromatin_fit_progress)
        self._chromatin_worker.finished_ok.connect(self._on_chromatin_fit_finished)
        self._chromatin_worker.failed.connect(self._on_chromatin_fit_failed)
        self._chromatin_worker.start()

    def _on_chromatin_fit_progress(self, done, total, msg):
        chp = self.ui.ChromatinTracingPanel
        chp.ProgressBar.setMaximum(total)
        chp.ProgressBar.setValue(done)
        chp.StatusLabel.setText(msg)

    def _on_chromatin_fit_finished(self, results):
        chp = self.ui.ChromatinTracingPanel
        chp.FitAllFovsPushButton.setEnabled(True)
        storage_paths = self._all_vlinks_storage_paths()
        for (storage_path, fov), alleles in results.items():
            vlinks_store.mirror_write_fov_alleles(storage_paths, fov, alleles)
        n_alleles = sum(len(alleles) for alleles in results.values())
        chp.StatusLabel.setText(f'Fit All FOVs done -- {n_alleles} allele(s) across {len(results)} FOV(s), saved to vlinks.h5.')
        current_fov = chp.AlleleFovSpinBox.value()
        for (storage_path, fov) in results:
            if fov == current_fov:
                self._refresh_chromatin_allele_lists(storage_path, fov)
                break

    def _on_chromatin_fit_failed(self, message):
        chp = self.ui.ChromatinTracingPanel
        chp.FitAllFovsPushButton.setEnabled(True)
        QtWidgets.QMessageBox.critical(self, 'Fit All FOVs error', message)

    # -- celltype determination --

    def _set_celltype_fov_ranges(self):
        ctp = self.ui.CelltypeDeterminationPanel
        celltype_name = ctp.selected_celltype()
        if celltype_name is None:
            QtWidgets.QMessageBox.warning(self, 'Set FOV Ranges', 'Select a celltype first.')
            return
        self._fov_ranges_by_celltype[celltype_name] = ctp.FovRangesLineEdit.text().strip()
        ctp.LogTextEdit.append(f"{celltype_name} FOV ranges set to '{self._fov_ranges_by_celltype[celltype_name]}'")
        self._refresh_celltype_summaries()
        self._persist_celltype_config()

    def _assign_barcode_channel(self):
        """
        Assigning a barcode channel also immediately runs calibration for
        it (using whatever's currently set in the Classification method/
        Calibrate for FOV(s)/Scale/Lower/Upper bound fields) -- per
        explicit correction, calibration was previously a fully separate
        step (Apply Calibration), which meant a freshly-assigned channel
        had no calibration at all until that second click. _apply_
        barcode_calibration is still its own standalone action too (e.g.
        to recalibrate for more FOVs, or after changing the bound
        fields), it's just no longer the ONLY way calibration happens.
        """
        ctp = self.ui.CelltypeDeterminationPanel
        celltype_name = ctp.selected_celltype()
        if celltype_name is None:
            QtWidgets.QMessageBox.warning(self, 'Assign Barcode Channel', 'Select a celltype first.')
            return
        inputs = ctp.get_barcode_calibration_inputs()
        hybe, channel, modality = inputs['hybe'], inputs['channel'], inputs['modality']
        if not hybe or channel is None or not modality:
            QtWidgets.QMessageBox.warning(self, 'Assign Barcode Channel', 'Pick a barcode hybe/channel first.')
            return
        self._barcode_channel_by_celltype[celltype_name] = (hybe, channel, modality)
        ctp.LogTextEdit.append(f'{celltype_name} <- {hybe} ch{channel} ({modality})')
        self._refresh_celltype_summaries()
        self._apply_barcode_calibration()

    def _apply_barcode_calibration(self):
        ctp = self.ui.CelltypeDeterminationPanel
        ip = self.ui.IngestionPanel
        celltype_name = ctp.selected_celltype()
        if celltype_name is None or celltype_name not in self._barcode_channel_by_celltype:
            QtWidgets.QMessageBox.warning(self, 'Apply Calibration',
                                          'Select a celltype with an assigned barcode channel first (Assign to Selected Celltype).')
            return
        bch = self._barcode_channel_by_celltype[celltype_name]
        hybe, channel, modality = bch
        storage_path = self._storage_path_for_modality(modality)
        if not storage_path:
            QtWidgets.QMessageBox.warning(self, 'Apply Calibration', 'Set storage path in the Ingestion tab first.')
            return
        inputs = ctp.get_barcode_calibration_inputs()
        fov_text = inputs['fov_scope_text']
        fovs = self._parse_fov_list(fov_text) if fov_text else self._parse_fov_list(ip.FovListLineEdit.text())
        if not fovs:
            QtWidgets.QMessageBox.warning(self, 'Apply Calibration', 'No FOV(s) to calibrate for (blank uses the Ingestion tab\'s FOV list).')
            return

        self._barcode_calibration['scale'].setdefault(bch, {})
        self._barcode_calibration['lower_bound'].setdefault(bch, {})
        self._barcode_calibration['upper_bound'].setdefault(bch, {})
        for fov in fovs:
            img = vlinks_store.read_hybe_mip(storage_path, fov, hybe, channel)
            if img is None:
                ctp.LogTextEdit.append(f'FOV{fov:02d}: {hybe} ch{channel} not in vlinks.h5 -- ingest it first.')
                continue
            lower = np.quantile(img, inputs['lower_value']) if inputs['lower_is_quantile'] else inputs['lower_value']
            upper = np.quantile(img, inputs['upper_value']) if inputs['upper_is_quantile'] else inputs['upper_value']
            # asymmetric clamp kept for fidelity with CellClassifier's own _setupCellBarcode --
            # it clamps the upper bound down to the image's own max but never clamps the
            # lower bound up to the image's own min; ported as-is, not silently "fixed".
            upper = min(upper, float(img.max()))
            self._barcode_calibration['scale'][bch][fov] = inputs['scale']
            self._barcode_calibration['lower_bound'][bch][fov] = float(lower)
            self._barcode_calibration['upper_bound'][bch][fov] = float(upper)
        ctp.LogTextEdit.append(f'Calibrated {celltype_name} ({hybe} ch{channel}) for FOV(s) {fovs}: '
                               f"scale={inputs['scale']}, lower={lower:.2f}, upper={upper:.2f} (last FOV shown)")
        self._refresh_celltype_summaries()
        self._persist_celltype_config()

    def _persist_celltype_config(self):
        """
        Auto-persists Celltype Determination's ENTIRE setup work (FOV
        ranges, barcode channel assignments, the actual computed per-FOV
        calibration bounds, and the classification method) to vlinks.h5
        immediately -- per explicit request, this should reach the same
        "just usable, no additional move" standard cells/spots/alignment
        matrices already meet elsewhere in this app, not just the
        celltype names themselves. Mirrored to every storage path this
        session knows about (see _all_vlinks_storage_paths), same
        reasoning as celltype names/cell data -- a celltype's barcode
        channel can belong to either configured modality.
        """
        storage_paths = self._all_vlinks_storage_paths()
        if not storage_paths:
            return
        ctp = self.ui.CelltypeDeterminationPanel
        vlinks_store.mirror_write_celltype_config(
            storage_paths, self._fov_ranges_by_celltype, self._barcode_channel_by_celltype,
            self._barcode_calibration, barcode_method=ctp.barcode_method())

    def _refresh_celltype_summaries(self):
        ctp = self.ui.CelltypeDeterminationPanel
        fov_lines = [f'{name}: {rng}' for name, rng in self._fov_ranges_by_celltype.items() if rng]
        ctp.FovRangesSummaryTextEdit.setPlainText('\n'.join(fov_lines))
        cal_lines = []
        for name, bch in self._barcode_channel_by_celltype.items():
            hybe, channel, modality = bch
            n_fovs = len(self._barcode_calibration['scale'].get(bch, {}))
            cal_lines.append(f'{name}: {hybe} ch{channel} ({modality}) (calibrated for {n_fovs} FOV(s))')
        ctp.CalibrationSummaryTextEdit.setPlainText('\n'.join(cal_lines))

    def _show_barcode_overview(self):
        ctp = self.ui.CelltypeDeterminationPanel
        fov_text = ctp.OverviewFovLineEdit.text().strip()
        fov = int(fov_text) if fov_text else self._current_spot_fov()
        if fov is None:
            QtWidgets.QMessageBox.warning(self, 'Show Barcode Overview', 'Set an FOV.')
            return
        images_by_channel, labels_by_channel = {}, {}
        for name in ctp.celltype_names():
            bch = self._barcode_channel_by_celltype.get(name)
            if bch is None:
                continue
            hybe, channel, modality = bch
            storage_path = self._storage_path_for_modality(modality)
            if not storage_path:
                continue
            img = vlinks_store.read_hybe_mip(storage_path, fov, hybe, channel)
            if img is None:
                ctp.LogTextEdit.append(f'Overview: {hybe} ch{channel} not in vlinks.h5 -- ingest it first.')
                continue
            # warp each channel into the pipeline's ONE shared frame for
            # visualization ONLY (never for stored/analyzed data) -- same
            # established exception used by every alignment preview in
            # canvas/pipeline_canvas.py. Through the resolver, because the
            # barcode channels of different celltypes can live in DIFFERENT
            # modalities: only (hybe, modality) names a frame, and only the
            # shared frame is common to all of them.
            H = self._matrix_to_shared(hybe, modality, None, fov)
            if H is not None:
                height, width = img.shape
                img = cv2.warpAffine(img.astype(np.float32), as_cv2(H)[:2], (width, height))
            images_by_channel[bch] = img
            labels_by_channel[bch] = f'{name}: {hybe} ch{channel} ({modality})'
        if not images_by_channel:
            QtWidgets.QMessageBox.warning(self, 'Show Barcode Overview', 'Assign at least one celltype to a barcode channel first.')
            return
        self.barcode_overview_displayer.set_data(images_by_channel, labels_by_channel)
        self.barcode_overview_displayer.show()
        self.barcode_overview_displayer.raise_()

    def _celltype_cell_containers(self):
        ctp = self.ui.CelltypeDeterminationPanel
        containers = []
        if self.cell_container_permanent is not None:
            containers.append(self.cell_container_permanent)
        if ctp.IncludeTransientCheckBox.isChecked() and self.cell_container is not None:
            containers.append(self.cell_container)
        return containers

    def _celltype_distinct_cells(self, containers):
        """
        {fov: {cell_id: [every container's copy of that cell]}} --
        permanent's copy first when both tiers hold it.

        The permanent and transient containers hold synced COPIES of the
        same cell, so iterating "every cell in every container" classified
        -- and counted, and logged -- each cell twice (confirmed real:
        "200 cell(s), 280 spot(s)" for a 100-cell, 140-assigned-spot FOV).
        Classification happens once per DISTINCT cell; the result is then
        applied to every copy so the tiers stay in sync.
        """
        by_fov = {}
        for container in containers:
            for fov, cells in container.data.items():
                for cell in cells.values():
                    by_fov.setdefault(fov, {}).setdefault(int(cell.id), []).append(cell)
        return by_fov

    def _cell_is_permanent(self, fov, cell_id):
        return (self.cell_container_permanent is not None
                and self.cell_container_permanent.data.get(fov, {}).get(int(cell_id)) is not None)

    def _run_celltype_determination(self):
        ctp = self.ui.CelltypeDeterminationPanel
        containers = self._celltype_cell_containers()
        if not containers:
            QtWidgets.QMessageBox.warning(self, 'Run Celltype Determination',
                                          'No saved (or transient) cells to classify -- check "Also classify unsaved '
                                          '(transient) cells" above if you haven\'t clicked Save in Cell Segmentation yet.')
            return

        names = ctp.celltype_names()
        now = datetime.now().isoformat()

        # this loop is synchronous (real per-FOV H5 reads in barcode mode)
        # -- disable the button and force the "Running..." status to paint
        # immediately, so a real multi-second run doesn't look like a dead
        # click with no feedback at all.
        ctp.RunCelltypeDeterminationPushButton.setEnabled(False)
        self.statusBar().showMessage('Running celltype determination...')
        QtWidgets.QApplication.processEvents()
        try:
            self._run_celltype_determination_body(ctp, containers, names, now)
        except Exception as e:
            # this loop used to have NO error handling at all -- a real,
            # reproducible KeyError (calibration missing for one FOV) was
            # confirmed crashing it silently: no dialog, nothing in the
            # log, and the button left permanently disabled, three
            # separate times across sessions with zero visible feedback
            # each time. Any future unexpected failure now surfaces for
            # real instead of vanishing into stderr.
            ctp.RunCelltypeDeterminationPushButton.setEnabled(True)
            self.statusBar().clearMessage()
            QtWidgets.QMessageBox.critical(self, 'Celltype determination error', f'{type(e).__name__}: {e}')

    def _run_celltype_determination_body(self, ctp, containers, names, now):
        if ctp.mode() == 'fov':
            range_strings = {n: self._fov_ranges_by_celltype[n] for n in names
                             if self._fov_ranges_by_celltype.get(n)}
            if not range_strings:
                ctp.RunCelltypeDeterminationPushButton.setEnabled(True)
                self.statusBar().clearMessage()
                QtWidgets.QMessageBox.warning(self, 'Run Celltype Determination', 'No FOV ranges set for any celltype.')
                return
            celltype_from_fov, _ = celltype.build_celltype_from_fov_ranges(range_strings)
            n_cells = 0
            last_fov = None
            permanent_fovs_touched = set()
            for fov, cellmap in self._celltype_distinct_cells(containers).items():
                ct = celltype.classify_fov(fov, celltype_from_fov)
                for cell_id, copies in cellmap.items():
                    for cell in copies:
                        cell.celltype = ct
                        cell.linked, cell.linked_at = True, now
                    for spot in self.spot_container.of_cell(fov, cell_id):
                        spot.celltype = ct
                        spot.linked, spot.linked_at = True, now
                    n_cells += 1
                    last_fov = fov
                    if self._cell_is_permanent(fov, cell_id):
                        permanent_fovs_touched.add(fov)
            self._persist_celltype_results(permanent_fovs_touched)
            ctp.RunCelltypeDeterminationPushButton.setEnabled(True)
            ctp.LogTextEdit.append(f'FOV-mode: {n_cells} cell(s) classified'
                                   f'{f", saved to vlinks.h5 for FOV(s) {sorted(permanent_fovs_touched)}" if permanent_fovs_touched else ""}.')
            self.statusBar().showMessage('Celltype determination complete.', 5000)
            QtWidgets.QMessageBox.information(self, 'Celltype determination complete', f'{n_cells} cell(s) classified.')
            if last_fov is not None:
                self._show_celltype_result(last_fov)
            return

        barcode_channel = [self._barcode_channel_by_celltype[n] for n in names if n in self._barcode_channel_by_celltype]
        barcode_celltype = [n for n in names if n in self._barcode_channel_by_celltype]
        if not barcode_channel:
            ctp.RunCelltypeDeterminationPushButton.setEnabled(True)
            self.statusBar().clearMessage()
            QtWidgets.QMessageBox.warning(self, 'Run Celltype Determination',
                                          'No celltype has an assigned+calibrated barcode channel.')
            return
        celltype_determination = {'barcode': {
            'barcode_channel': barcode_channel,
            'barcode_celltype': barcode_celltype,
            'scale': self._barcode_calibration['scale'],
            'lower_bound': self._barcode_calibration['lower_bound'],
            'upper_bound': self._barcode_calibration['upper_bound'],
        }}

        image_cache = {}  # {fov(str): {(hybe,channel,modality): ndarray or None}}
        n_cells, n_spots, n_cells_skipped = 0, 0, 0
        skip_reasons = set()   # REAL causes, named -- never a blanket blame
        last_fov = None
        permanent_fovs_touched = set()
        for fov, cellmap in self._celltype_distinct_cells(containers).items():
                if cellmap:
                    last_fov = fov
                fov_key = str(fov)
                if fov_key not in image_cache:
                    image_cache[fov_key] = {}
                    for bch in barcode_channel:
                        hybe, channel_id, bch_modality = bch
                        # each celltype's own barcode channel can belong
                        # to a DIFFERENT modality than the others --
                        # resolve each one's real storage path
                        # independently rather than assuming they all
                        # share one.
                        bch_storage_path = self._storage_path_for_modality(bch_modality)
                        image_cache[fov_key][bch] = (vlinks_store.read_hybe_mip(bch_storage_path, fov, hybe, channel_id)
                                                     if bch_storage_path else None)

                # calibration is per-(hybe,channel)-per-FOV (Apply Calibration
                # only covers whichever FOV(s) were explicitly calibrated) --
                # missing it for even one barcode channel used to reach
                # classify_cell_barcode's unconditional barcode['scale'][bch][fov]
                # lookup and crash with an uncaught KeyError, silently aborting
                # the whole run with no dialog (confirmed via a real crash:
                # KeyError('Hyb_130', 635), repeated across 3 separate clicks
                # with zero visible feedback each time). Checked once per FOV,
                # not per cell -- calibration completeness doesn't vary by cell.
                # a barcode channel with no MIP in vlinks.h5 is missing
                # DATA -- the only honest skip. Alignment is NEVER a skip
                # cause: an uncomputed layer is identity by the pipeline's
                # own rule, and the resolver always answers.
                missing_mips = [bch for bch in barcode_channel
                                if image_cache[fov_key].get(bch) is None]
                if missing_mips:
                    labels = ', '.join(f'{h}/ch{c} ({m})' for h, c, m in missing_mips)
                    ctp.LogTextEdit.append(f'FOV{fov:02d}: skipped ({len(cellmap)} cell(s)) -- no MIP in '
                                           f'vlinks.h5 for barcode channel(s) {labels}. Ingest those '
                                           f'(hybe, channel) pairs; alignment is NOT required.')
                    n_cells_skipped += len(cellmap)
                    skip_reasons.add(f'no MIP for {labels}')
                    continue

                missing_calibration = [bch for bch in barcode_channel
                                       if int(fov) not in self._barcode_calibration['scale'].get(bch, {})
                                       or int(fov) not in self._barcode_calibration['lower_bound'].get(bch, {})
                                       or int(fov) not in self._barcode_calibration['upper_bound'].get(bch, {})]
                if missing_calibration:
                    ctp.LogTextEdit.append(f'FOV{fov:02d}: skipped ({len(cellmap)} cell(s)) -- no calibration for '
                                           f'{missing_calibration} at this FOV (Apply Calibration first).')
                    n_cells_skipped += len(cellmap)
                    skip_reasons.add('no calibration at this FOV (Apply Calibration first)')
                    continue

                for cell_id, copies in cellmap.items():
                    cell = copies[0]         # permanent's copy when both tiers hold it
                    y_ref, x_ref = cell.area
                    area_by_channel = {}
                    for bch in barcode_channel:
                        hybe, channel_id, modality = bch
                        img = image_cache[fov_key].get(bch)
                        if img is None:
                            continue
                        # modality comes straight from the stored (hybe,
                        # channel, modality) triple this celltype was
                        # assigned with (see _assign_barcode_channel) --
                        # not a live "current modality" guess, so a
                        # selection mixing RNA- and DNA-owned barcode
                        # channels across different celltypes resolves
                        # each one correctly regardless of what's
                        # currently showing anywhere in the UI.
                        # _matrix_to_cellref prefers a real, cell-level-
                        # refined entry when compute_cell_alignment has
                        # run for (hybe, modality) -- including cell.
                        # reference_hybe's own entry, which is no longer
                        # forced to identity -- falling back to the FOV/
                        # cross-modal-only matrix otherwise, exactly like
                        # compute_cell_alignment's own reject/out-of-frame
                        # fallbacks do. A plain per-point inverse
                        # transform, not cell.get_area_in_readout's
                        # masking/closing machinery, so point order/count
                        # stays identical across every channel.
                        H = self._matrix_to_cellref(hybe, modality, cell, cell.fov)
                        if H is None:
                            continue
                        pts = la.inv(H) @ np.vstack([y_ref, x_ref, np.ones_like(x_ref, dtype=float)])
                        y_h, x_h = pts[0], pts[1]
                        height, width = img.shape
                        area_by_channel[bch] = (np.clip(x_h, 0, width - 1), np.clip(y_h, 0, height - 1))
                    if len(area_by_channel) < len(barcode_channel):
                        n_cells_skipped += 1
                        skip_reasons.add('a barcode frame could not be resolved (unexpected -- report this)')
                        continue
                    ct = celltype.classify_cell_barcode(area_by_channel, cell.fov, image_cache,
                                                        celltype_determination, method=ctp.barcode_method())
                    for c in copies:         # every tier's copy stays in sync
                        c.celltype = ct
                        c.linked, c.linked_at = True, now
                    n_cells += 1
                    if self._cell_is_permanent(fov, cell_id):
                        permanent_fovs_touched.add(fov)

                    for spot in self.spot_container.of_cell(fov, cell.id):
                        xy_by_channel = {}
                        for bch in barcode_channel:
                            hybe, channel_id, modality = bch
                            # spot.coordinate lives in the pipeline's shared
                            # frame (ACell.matrix_to_shared), NOT cell.
                            # reference_hybe's frame like cell.area above --
                            # _matrix_to_shared is the matching "best
                            # available, no no-alignment" resolver for it.
                            H = self._matrix_to_shared(hybe, modality, cell, cell.fov)
                            if H is None:
                                continue
                            sy, sx, _ = la.inv(H) @ np.array([spot.coordinate[0], spot.coordinate[1], 1.0])
                            # clip exactly like the cell-area path above:
                            # an aligned position can land a pixel outside
                            # this channel's own frame, and an unclipped
                            # index crashed the WHOLE run (IndexError 1024)
                            height, width = image_cache[fov_key][bch].shape
                            xy_by_channel[bch] = (float(np.clip(sx, 0, width - 1)),
                                                  float(np.clip(sy, 0, height - 1)))
                        if len(xy_by_channel) < len(barcode_channel):
                            continue
                        spot.celltype = celltype.classify_spot_barcode(xy_by_channel, cell.fov, image_cache, celltype_determination)
                        spot.linked, spot.linked_at = True, now
                        n_spots += 1

        self._persist_celltype_results(permanent_fovs_touched)
        ctp.RunCelltypeDeterminationPushButton.setEnabled(True)
        skipped_note = (f'{n_cells_skipped} cell(s) skipped: {"; ".join(sorted(skip_reasons))}'
                        if n_cells_skipped else '0 skipped')
        ctp.LogTextEdit.append(f'Barcode-mode: {n_cells} cell(s), {n_spots} spot(s) classified ({skipped_note})'
                               f'{f", saved to vlinks.h5 for FOV(s) {sorted(permanent_fovs_touched)}" if permanent_fovs_touched else ""}.')
        self.statusBar().showMessage('Celltype determination complete.', 5000)
        QtWidgets.QMessageBox.information(self, 'Celltype determination complete',
                                          f'{n_cells} cell(s), {n_spots} spot(s) classified ({skipped_note}).')
        if last_fov is not None:
            self._show_celltype_result(last_fov)

    def _persist_celltype_results(self, fovs):
        """
        Auto-persists celltype/linked updates for already-saved (permanent)
        cells to vlinks.h5 immediately -- per explicit correction, the
        Memory Viewer's Cell celltype/Spot celltype columns read disk
        state only (same "what's actually persisted" semantics as every
        other column there), and without this, a real, successful
        classification run left that status looking stuck at 0/N until an
        unrelated future Save. Scoped to cell_container_permanent only,
        never the transient container -- classifying transient cells
        still requires an explicit Save in Cell Segmentation before
        they're promoted/persisted at all; celltype determination isn't
        meant to be a second, implicit way to do that.
        """
        if not fovs or self.cell_container_permanent is None:
            return
        storage_paths = self._all_vlinks_storage_paths()
        if not storage_paths:
            return
        for fov in fovs:
            if self.cell_container_permanent.data.get(fov):
                vlinks_store.mirror_write_cells(storage_paths, fov, self.cell_container_permanent)
            # spot celltypes persist through the SAME door every accept
            # loop uses (per explicit correction: cells persisted their
            # celltype here, spots silently did not). _persist_fov_spots
            # refuses FOVs whose spots were never staged, so a fov with
            # classified cells but untouched spots stays untouched.
            self._persist_fov_spots(fov)

    def _show_celltype_result(self, fov=None):
        """
        Reconstructs the same Reference | Celltype two-panel view as
        _try_show_existing_cells does for Reference | Mask -- straight from
        whichever container (permanent, or transient if included) actually
        holds this FOV's cells right now, no re-computation. fov=None
        (button-triggered) falls back to the Barcode page's own Overview
        FOV field, same resolution order _show_barcode_overview already
        uses.
        """
        ctp = self.ui.CelltypeDeterminationPanel
        ip = self.ui.IngestionPanel
        storage_path = ip.StoragePathLineEdit.text().strip()
        if fov is None:
            fov_text = ctp.OverviewFovLineEdit.text().strip()
            fov = int(fov_text) if fov_text else self._current_spot_fov()
        if not storage_path or fov is None:
            QtWidgets.QMessageBox.warning(self, 'Show Celltype Result', 'Set storage path (Ingestion tab) and an FOV.')
            return
        cells = None
        for container in self._celltype_cell_containers():
            if container.data.get(fov):
                cells = container.get_cells(fov)
                break
        if not cells:
            QtWidgets.QMessageBox.warning(self, 'Show Celltype Result', f'No cells for FOV{fov:02d}.')
            return

        reference_hybe = cells[0].reference_hybe
        frame_shape = cells[0].frame_shape
        # cells are mirrored into every configured modality's own vlinks.h5
        # (see _all_vlinks_storage_paths), so the CURRENT Ingestion tab's
        # storage_path is only guaranteed to be right when this FOV's cells
        # were segmented in the currently-active modality -- classifying
        # RNA-segmented cells against a DNA barcode channel (this app's own
        # real dual-modality workflow) leaves storage_path pointing at DNA
        # while reference_hybe is an RNA-only hybe (e.g. Hyb_500), which a
        # DNA_queue lookup can't find. Resolve the storage path from the
        # cell's OWN segmentation modality instead, falling back to the
        # current one only if that modality's own path isn't configured.
        # (reference_hybe, reference_MODALITY) is the pair: after cytoplasm
        # incorporation a cell's reference is e.g. (Hyb_500, RNA) while its
        # modality can still read 'DNA'; pairing the hybe with the wrong
        # modality asked DNA_queue for an RNA hybe -> None -> the loader
        # silently bailed and the FOV showed no cells at all.
        cell_storage_path = (self.ui.IngestionPanel.modality_data
                             .get(cells[0].reference_modality or '', {})
                             .get('storage_path') or storage_path)
        reference_image = vlinks_store.fiducial_channel_mip(cell_storage_path, fov, reference_hybe)
        if reference_image is None:
            QtWidgets.QMessageBox.critical(self, 'Show Celltype Result',
                                           f'{reference_hybe} not in vlinks.h5 for FOV{fov:02d}.')
            return

        mask = np.zeros(frame_shape, dtype=np.uint8)
        for cell in cells:
            y, x = cell.area
            mask[y.astype(int), x.astype(int)] = cell.id
        celltype_by_id = {cell.id: cell.celltype for cell in cells}

        self.celltype_result_displayer.set_data(reference_image, mask, celltype_by_id)
        self.celltype_result_displayer.show()
        self.celltype_result_displayer.raise_()

    # -- within-experiment alignment --

    def _run_fov_alignment(self):
        """Current-FOV alignment (ap.SameModalityFovSpinBox) -- computes into
        a staged result with its own Accept/Reject, nothing persists until
        Accept. Run All FOV Alignment below (_run_fov_alignment_all) is the
        always-auto-save, no-staging counterpart for every FOV at once."""
        ap = self.ui.AlignmentPanel
        reference_hybe = ap.current_reference_hybe()
        modality = ap.current_reference_modality()
        if not reference_hybe or not modality:
            QtWidgets.QMessageBox.warning(self, 'Run FOV Alignment', 'Parse a layout first, then pick a reference hybe.')
            return
        storage_path = self._storage_path_for_modality(modality)
        fov = ap.SameModalityFovSpinBox.value()
        if not storage_path:
            QtWidgets.QMessageBox.warning(self, 'Run FOV Alignment', 'Set storage path in the Ingestion tab first.')
            return

        # active_hybe_list, never the Ingestion tab's hybe-to-ingest
        # checkboxes -- a checked hybe only means the user WANTS it
        # ingested (see _run_ingestion), not that it actually has a real
        # {hybe}_stack.h5 on disk yet. Using checkbox state here let a
        # checked-but-never-ingested hybe reach align_same_modality's
        # bare h5py.File() open and crash.
        hybe_records = self.ui.IngestionPanel.modality_data.get(modality, {}).get('active_hybe_list', [])
        if not hybe_records:
            QtWidgets.QMessageBox.warning(self, 'Run FOV Alignment',
                                          'No ingested hybes found for this modality/FOV list yet -- run ingestion first.')
            return
        if reference_hybe not in {r['folder'] for r in hybe_records}:
            QtWidgets.QMessageBox.warning(self, 'Run FOV Alignment', 'Reference hybe must be an ingested hybe.')
            return

        border_trim = ap.SameModalityBorderTrimSpinBox.value()
        max_shift = ap.SameModalityMaxShiftSpinBox.value() or None

        ap.RunFovAlignmentPushButton.setEnabled(False)
        self.statusBar().showMessage('Running FOV alignment...')
        self._alignment_worker = AlignmentWorker(storage_path, [fov], hybe_records, reference_hybe, write=False,
                                                  border_trim=border_trim, max_shift=max_shift)
        self._alignment_worker.progress.connect(lambda done, total, msg: self.statusBar().showMessage(msg))
        self._alignment_worker.finished_ok.connect(
            lambda results: self._on_fov_alignment_finished(results, storage_path, hybe_records, reference_hybe))
        self._alignment_worker.failed.connect(self._on_fov_alignment_failed)
        self._alignment_worker.start()

    def _on_fov_alignment_finished(self, results, storage_path, hybe_records, reference_hybe):
        ap = self.ui.AlignmentPanel
        self._same_modality_context = {'storage_path': storage_path, 'hybe_records': hybe_records, 'reference_hybe': reference_hybe}
        self._pending_same_modality_alignment = results
        self._refresh_same_modality_results_list()
        ap.RunFovAlignmentPushButton.setEnabled(True)
        self.statusBar().showMessage('FOV alignment computed.', 5000)

        ap.SameModalityAcceptPushButton.setEnabled(True)
        ap.SameModalityRejectPushButton.setEnabled(True)

        # Per explicit request: the all-readouts overlay for the just-
        # aligned FOV shows immediately, same as cross-modal's own
        # preview-on-compute behavior, rather than requiring a separate
        # Show click. Only the current-FOV run does this (one FOV, one
        # popup) -- Run All FOV Alignment below stays silent, since N
        # auto-popups for N FOVs would just be noise.
        fov = list(results.keys())[0]
        channel_type = ap.SameModalityChannelTypeComboBox.currentText()
        self.alignment_preview_window.show()
        self.alignment_preview_window.raise_()
        self.preview_canvas.draw_fov_all_readouts_overlay(storage_path, fov, hybe_records, reference_hybe, results[fov],
                                                  channel_type=channel_type)

    def _on_fov_alignment_failed(self, message):
        self.ui.AlignmentPanel.RunFovAlignmentPushButton.setEnabled(True)
        self.statusBar().clearMessage()
        QtWidgets.QMessageBox.critical(self, 'Alignment error', message)

    def _run_fov_alignment_all(self):
        """Every FOV in the Ingestion tab's FOV list, computed AND saved
        immediately -- no staging, no Accept step. Use Run Current FOV
        Alignment above first to confirm the parameters on one real FOV."""
        ap = self.ui.AlignmentPanel
        ip = self.ui.IngestionPanel
        reference_hybe = ap.current_reference_hybe()
        modality = ap.current_reference_modality()
        if not reference_hybe or not modality:
            QtWidgets.QMessageBox.warning(self, 'Run FOV Alignment', 'Parse a layout first, then pick a reference hybe.')
            return
        storage_path = self._storage_path_for_modality(modality)
        fov_list = self._parse_fov_list(ip.FovListLineEdit.text())
        if not storage_path or not fov_list:
            QtWidgets.QMessageBox.warning(self, 'Run FOV Alignment', 'Set storage path and FOV list in the Ingestion tab first.')
            return

        hybe_records = self.ui.IngestionPanel.modality_data.get(modality, {}).get('active_hybe_list', [])
        if not hybe_records:
            QtWidgets.QMessageBox.warning(self, 'Run FOV Alignment',
                                          'No ingested hybes found for this modality/FOV list yet -- run ingestion first.')
            return
        if reference_hybe not in {r['folder'] for r in hybe_records}:
            QtWidgets.QMessageBox.warning(self, 'Run FOV Alignment', 'Reference hybe must be an ingested hybe.')
            return

        border_trim = ap.SameModalityBorderTrimSpinBox.value()
        max_shift = ap.SameModalityMaxShiftSpinBox.value() or None

        ap.RunAllFovAlignmentPushButton.setEnabled(False)
        self.statusBar().showMessage('Running FOV alignment for all FOVs...')
        self._alignment_worker = AlignmentWorker(storage_path, fov_list, hybe_records, reference_hybe, write=True,
                                                  border_trim=border_trim, max_shift=max_shift)
        self._alignment_worker.progress.connect(lambda done, total, msg: self.statusBar().showMessage(msg))
        self._alignment_worker.finished_ok.connect(
            lambda results: self._on_fov_alignment_all_finished(results, storage_path, hybe_records, reference_hybe))
        self._alignment_worker.failed.connect(self._on_fov_alignment_all_failed)
        self._alignment_worker.start()

    def _on_fov_alignment_all_finished(self, results, storage_path, hybe_records, reference_hybe):
        ap = self.ui.AlignmentPanel
        self._same_modality_context = {'storage_path': storage_path, 'hybe_records': hybe_records, 'reference_hybe': reference_hybe}
        for _fov, _m in results.items():
            self._merge_fov_matrices(_fov, _m)
        self._pending_same_modality_alignment = None
        self._refresh_same_modality_results_list()
        ap.RunAllFovAlignmentPushButton.setEnabled(True)
        self.statusBar().showMessage('FOV alignment computed.', 5000)

        channel_type = ap.SameModalityChannelTypeComboBox.currentText()
        vlinks_store.write_global_params(storage_path, same_modality_reference_hybe=reference_hybe,
                                         same_modality_channel_type=channel_type)
        for fov, matrices in results.items():
            save_path = paths.figure_path(storage_path, 'alignment', fov, 'alignment_overlay.png')
            self.preview_canvas.draw_fov_all_readouts_overlay(storage_path, fov, hybe_records, reference_hybe, matrices,
                                                      save_path=save_path, channel_type=channel_type)
        QtWidgets.QMessageBox.information(self, 'FOV alignment complete',
                                          f'{len(results)} FOV(s) aligned and saved; overlay image(s) written.')

    def _on_fov_alignment_all_failed(self, message):
        self.ui.AlignmentPanel.RunAllFovAlignmentPushButton.setEnabled(True)
        self.statusBar().clearMessage()
        QtWidgets.QMessageBox.critical(self, 'Alignment error', message)

    def _accept_same_modality_alignment(self):
        ap = self.ui.AlignmentPanel
        ctx = self._same_modality_context
        if not self._pending_same_modality_alignment or ctx is None:
            return
        channel_type = ap.SameModalityChannelTypeComboBox.currentText()
        vlinks_store.write_global_params(ctx['storage_path'], same_modality_reference_hybe=ctx['reference_hybe'],
                                         same_modality_channel_type=channel_type)
        for fov, matrices in self._pending_same_modality_alignment.items():
            alignment.write_same_modality_matrices(ctx['storage_path'], fov, matrices, ctx['reference_hybe'])
            save_path = paths.figure_path(ctx['storage_path'], 'alignment', fov, 'alignment_overlay.png')
            self.preview_canvas.draw_fov_all_readouts_overlay(ctx['storage_path'], fov, ctx['hybe_records'],
                                                      ctx['reference_hybe'], matrices, save_path=save_path,
                                                      channel_type=channel_type)
        for _fov, _m in self._pending_same_modality_alignment.items():
            self._merge_fov_matrices(_fov, _m)
        self._pending_same_modality_alignment = None
        self._refresh_same_modality_results_list()
        ap.SameModalityAcceptPushButton.setEnabled(False)
        ap.SameModalityRejectPushButton.setEnabled(False)
        QtWidgets.QMessageBox.information(self, 'FOV alignment accepted', 'Matrices written to H5; overlay image(s) saved.')

        # Matrices changed -> every spot's shared-frame coordinate (and
        # possibly its owner) is stale. Assignment is cheap; recompute for
        # every loaded FOV and persist all slices, per explicit decision
        # that saving cells, matrices, or spots all re-run assignment.
        for _fov in list(self.fov_matrices.keys()):
            try:
                self._recast_persisted_spots(_fov)
            except Exception as e:
                self.statusBar().showMessage(f'spot reassignment after accept failed for FOV{_fov}: {e}')

    def _reject_same_modality_alignment(self):
        ap = self.ui.AlignmentPanel
        self._pending_same_modality_alignment = None
        self._refresh_same_modality_results_list()
        ap.SameModalityAcceptPushButton.setEnabled(False)
        ap.SameModalityRejectPushButton.setEnabled(False)

    def _show_same_modality_preview(self, item):
        fov, hybe = item.data(QtCore.Qt.UserRole)
        ctx = self._same_modality_context
        if ctx is None:
            return
        # the reference hybe's own row is a legitimate result too (its
        # matrix is the identity, by construction -- see
        # align_same_modality) -- show it like any other, not a
        # silent no-op, so before/after correctly display as identical.
        fiducial_channels = {r['folder']: r['fiducial_channel'] for r in ctx['hybe_records']}
        # a staged (not-yet-accepted) manual result takes priority so the
        # preview shows what the user is actually about to accept/reject,
        # not a stale prior result
        matrices = ((self._pending_same_modality_alignment or {}).get(fov)
                    or self._fov_matrices_for(ctx['storage_path'], fov))
        if matrices is None:
            return
        self.alignment_preview_window.show()
        self.alignment_preview_window.raise_()
        self.preview_canvas.draw_same_modality_preview(ctx['storage_path'], fov, ctx['reference_hybe'], hybe,
                                                    fiducial_channels, matrices)

    def _show_same_modality_all_readouts_overlay(self):
        """
        FOV is an explicit field (SameModalityOverlayFovSpinBox). Matrices
        come from whichever source actually has them: a staged (not-yet-
        accepted) result takes priority, then the in-memory fov_matrices
        cache (already populated by _activate_fov for any FOV that's been
        visited), then a direct disk read as a last resort -- so this never
        requires re-running alignment in this session just to view an
        already-aligned FOV.
        """
        ap = self.ui.AlignmentPanel
        fov = ap.SameModalityOverlayFovSpinBox.value()
        reference_hybe = ap.current_reference_hybe()
        modality = ap.current_reference_modality()
        storage_path = self._storage_path_for_modality(modality)
        hybe_records = self.ui.IngestionPanel.modality_data.get(modality, {}).get('active_hybe_list', []) if modality else []
        if not storage_path or not reference_hybe or not hybe_records:
            QtWidgets.QMessageBox.warning(self, 'Show All-Readouts Overlay',
                                          'Set storage path (Ingestion tab) and reference hybe first.')
            return

        ctx = self._same_modality_context
        matrices = None
        if ctx is not None and ctx.get('storage_path') == storage_path:
            matrices = (self._pending_same_modality_alignment or {}).get(fov)
        if matrices is None:
            matrices = self._fov_matrices_for(storage_path, fov)
        if matrices is None:
            matrices = alignment.read_same_modality_matrices(storage_path, fov, hybe_records)

        channel_type = ap.SameModalityChannelTypeComboBox.currentText()
        self.alignment_preview_window.show()
        self.alignment_preview_window.raise_()
        self.preview_canvas.draw_fov_all_readouts_overlay(storage_path, fov, hybe_records, reference_hybe, matrices,
                                                  channel_type=channel_type)

    # -- cross-modal alignment --

    def _refresh_cross_modal_results_list(self):
        """
        Populates the FULL Results list -- every FOV in the FOV list with
        EITHER a disk-persisted cross-modal result, an in-memory
        cross_modal_result entry (already accepted/auto-saved this
        session), OR a not-yet-accepted staged current-FOV result --
        never just whichever FOV happened to be aligned most recently
        (mirrors _refresh_same_modality_results_list's own pattern). Rows
        for a staged (unsaved) result are marked "[pending]".
        """
        ip, ap = self.ui.IngestionPanel, self.ui.AlignmentPanel
        rna_storage_path = ap.RnaStoragePathLineEdit.text().strip()
        dna_storage_path = ap.DnaStoragePathLineEdit.text().strip()
        dna_reference_hybe = ap.DnaReferenceHybeComboBox.currentText().strip()
        fov_list = self._parse_fov_list(ip.FovListLineEdit.text())
        ap.CrossModalResultsListWidget.clear()
        if not rna_storage_path or not dna_storage_path or not dna_reference_hybe or not fov_list:
            return
        disk_results = {}
        for fov in fov_list:
            # vlinks_store's own mirror first -- keyed only by (storage_path,
            # fov), never by dna_reference_hybe's own name, so it can't be
            # read from the wrong hybe's stack file the way chain.py's own
            # read can when the reference-hybe combo hasn't been reconciled
            # against vlinks yet (confirmed real bug: during config load,
            # ap.DnaReferenceHybeComboBox briefly holds the config file's
            # own text before _refresh_params_from_vlinks corrects it,
            # so a read here with the wrong reference hybe silently pulled
            # /matrix_across out of an unrelated hybe's own file and
            # permanently contaminated self.cross_modal_result for the rest
            # of the session -- vlinks_store's copy is immune to this since
            # it never depends on which hybe is currently selected anywhere).
            # There is deliberately NO fallback to chain.py's reference-hybe-
            # keyed read. An empty mirror means "no cross-modal alignment has
            # ever been accepted", and the correct answer to that is no bridge
            # at all -- identity, which the FrameResolver already supplies as
            # its documented default for an uncomputed layer. Reading
            # {dna_reference_hybe}_stack.h5's own /matrix_across instead
            # cannot distinguish an accepted matrix from whatever stale bytes
            # happen to sit in an unrelated hybe's file, and a fallback that
            # only holds when the mirror is ALREADY populated guards nothing:
            # on a freshly-rebuilt vlinks it fires every time and permanently
            # caches garbage here (confirmed -- this is what rotated every
            # projected cell mask ~100 degrees in the RNA hybes while the DNA
            # hybes, needing no bridge, stayed correct).
            H = vlinks_store.read_cross_modal_matrix(dna_storage_path, fov)
            if H is not None:
                disk_results[fov] = H
        if disk_results:
            self.cross_modal_result.update({(dna_storage_path, fov): H for fov, H in disk_results.items()})

        display_results = dict(disk_results)
        for (sp, fov), H in self.cross_modal_result.items():
            if sp == dna_storage_path:
                display_results[fov] = H
        pending_fovs = set()
        if self._pending_cross_modal:
            for fov, H in self._pending_cross_modal.items():
                display_results[fov] = H
                pending_fovs.add(fov)
        for fov in sorted(display_results.keys()):
            H = display_results[fov]
            suffix = ' [pending]' if fov in pending_fovs else ''
            dz = self._pending_cross_modal_z.get(fov)
            if dz is None:
                dz = self.cross_modal_z.get((dna_storage_path, fov))
            if dz is None:
                dz = vlinks_store.read_cross_modal_z(dna_storage_path, fov)
            item = QtWidgets.QListWidgetItem(
                f'FOV{fov:02d} {_matrix_summary("DNA->RNA", H)}, dz={float(dz or 0.0):+.1f}{suffix}')
            item.setData(QtCore.Qt.UserRole, fov)
            ap.CrossModalResultsListWidget.addItem(item)

    def _show_cross_modal_result_preview(self, item):
        """Clicking a Results row shows that FOV's overlay -- sets the
        Overlay FOV spinbox to match (so Show Overlay stays consistent
        with whatever was just clicked) and reuses _show_cross_modal_
        overlay's own matrix-source priority (staged/cache/disk)."""
        ap = self.ui.AlignmentPanel
        fov = item.data(QtCore.Qt.UserRole)
        ap.CrossModalOverlayFovSpinBox.blockSignals(True)
        ap.CrossModalOverlayFovSpinBox.setValue(fov)
        ap.CrossModalOverlayFovSpinBox.blockSignals(False)
        self._show_cross_modal_overlay()

    def _run_cross_modal_alignment(self):
        """Current-FOV alignment (ap.CrossModalFovSpinBox) -- computes into a
        staged result with its own Accept/Reject. Run Cross-Modal Alignment
        for All FOVs below (_run_cross_modal_alignment_all) is the always-
        auto-save, no-staging counterpart for every FOV at once."""
        ap = self.ui.AlignmentPanel
        rna_storage_path = ap.RnaStoragePathLineEdit.text().strip()
        dna_storage_path = ap.DnaStoragePathLineEdit.text().strip()
        rna_reference_hybe = ap.RnaReferenceHybeComboBox.currentText().strip()
        dna_reference_hybe = ap.DnaReferenceHybeComboBox.currentText().strip()
        channel_type = ap.ChannelTypeComboBox.currentText()
        fov = ap.CrossModalFovSpinBox.value()
        if not all([rna_storage_path, dna_storage_path, rna_reference_hybe, dna_reference_hybe]):
            QtWidgets.QMessageBox.warning(self, 'Run Cross-Modal Alignment',
                                          'Fill in both storage paths and reference hybes first.')
            return

        border_trim = ap.CrossModalBorderTrimSpinBox.value()
        max_shift = ap.CrossModalMaxShiftSpinBox.value() or None

        # Fiducial channels for the Z leg -- z drift is measured on the
        # same reference pair as dx/dy, in the same run.
        rna_fid = dna_fid = None
        try:
            rna_rec = {r['folder']: r for r in self._active_hybe_records_for_modality(
                self._modality_for_storage_path(rna_storage_path))}[rna_reference_hybe]
            dna_rec = {r['folder']: r for r in self._active_hybe_records_for_modality(
                self._modality_for_storage_path(dna_storage_path))}[dna_reference_hybe]
            rna_fid = alignment.pick_channel_by_type(rna_rec, 'fiducial')
            dna_fid = alignment.pick_channel_by_type(dna_rec, 'fiducial')
        except (KeyError, TypeError):
            pass
        ap.RunCrossModalPushButton.setEnabled(False)
        self.statusBar().showMessage('Running cross-modal alignment...')
        self._cross_modal_worker = CrossModalAlignmentWorker(rna_storage_path, dna_storage_path, [fov], self.fov_matrices,
                                                              rna_reference_hybe, dna_reference_hybe, channel_type,
                                                              border_trim=border_trim, max_shift=max_shift,
                                                              rna_fiducial_channel=rna_fid, dna_fiducial_channel=dna_fid)
        self._cross_modal_worker.progress.connect(lambda done, total, msg: self.statusBar().showMessage(msg))
        self._cross_modal_worker.finished_ok.connect(
            lambda results: self._on_cross_modal_finished(results, rna_storage_path, dna_storage_path,
                                                           rna_reference_hybe, dna_reference_hybe, channel_type))
        self._cross_modal_worker.failed.connect(self._on_cross_modal_failed)
        self._cross_modal_worker.start()

    def _on_cross_modal_finished(self, results, rna_storage_path, dna_storage_path, rna_reference_hybe, dna_reference_hybe, channel_type):
        # worker now returns {'H': {fov: H}, 'z': {fov: planes}} -- the z
        # drift is part of the cross-modal RESULT, staged and accepted with it.
        if isinstance(results, dict) and 'H' in results and 'z' in results:
            self._pending_cross_modal_z = dict(results['z'])
            results = results['H']
        ap = self.ui.AlignmentPanel
        ap.RunCrossModalPushButton.setEnabled(True)
        self.statusBar().showMessage('Cross-modal alignment computed.', 5000)
        self._cross_modal_context = {'rna_storage_path': rna_storage_path, 'dna_storage_path': dna_storage_path,
                                      'rna_reference_hybe': rna_reference_hybe, 'dna_reference_hybe': dna_reference_hybe,
                                      'channel_type': channel_type}
        self._pending_cross_modal = results
        self._refresh_cross_modal_results_list()
        last_fov = list(results.keys())[-1]
        # this is the only place the current-FOV preview auto-shows (the
        # Results list below still lets any OTHER already-computed FOV be
        # previewed on demand via a click) -- pop it up here rather than
        # waiting for a separate interactive trigger
        self.alignment_preview_window.show()
        self.alignment_preview_window.raise_()
        self.preview_canvas.draw_cross_modal_preview(rna_storage_path, dna_storage_path, last_fov,
                                             rna_reference_hybe, dna_reference_hybe, channel_type, results[last_fov],
                                             rna_fov_matrices=self._fov_matrices_for(rna_storage_path, last_fov),
                                             dna_fov_matrices=self._fov_matrices_for(dna_storage_path, last_fov))

        ap.CrossModalAcceptPushButton.setEnabled(True)
        ap.CrossModalRejectPushButton.setEnabled(True)

    def _on_cross_modal_failed(self, message):
        self.ui.AlignmentPanel.RunCrossModalPushButton.setEnabled(True)
        self.statusBar().clearMessage()
        QtWidgets.QMessageBox.critical(self, 'Cross-modal alignment error', message)

    def _run_cross_modal_alignment_all(self):
        """Every FOV in the Ingestion tab's FOV list, computed AND saved
        immediately -- no staging, no Accept step."""
        ap = self.ui.AlignmentPanel
        ip = self.ui.IngestionPanel
        rna_storage_path = ap.RnaStoragePathLineEdit.text().strip()
        dna_storage_path = ap.DnaStoragePathLineEdit.text().strip()
        rna_reference_hybe = ap.RnaReferenceHybeComboBox.currentText().strip()
        dna_reference_hybe = ap.DnaReferenceHybeComboBox.currentText().strip()
        channel_type = ap.ChannelTypeComboBox.currentText()
        fov_list = self._parse_fov_list(ip.FovListLineEdit.text())
        if not all([rna_storage_path, dna_storage_path, rna_reference_hybe, dna_reference_hybe]) or not fov_list:
            QtWidgets.QMessageBox.warning(self, 'Run Cross-Modal Alignment',
                                          'Fill in both storage paths and reference hybes, and set a FOV list in the Ingestion tab.')
            return

        border_trim = ap.CrossModalBorderTrimSpinBox.value()
        max_shift = ap.CrossModalMaxShiftSpinBox.value() or None

        # Fiducial channels for the Z leg -- z drift is measured on the
        # same reference pair as dx/dy, in the same run.
        rna_fid = dna_fid = None
        try:
            rna_rec = {r['folder']: r for r in self._active_hybe_records_for_modality(
                self._modality_for_storage_path(rna_storage_path))}[rna_reference_hybe]
            dna_rec = {r['folder']: r for r in self._active_hybe_records_for_modality(
                self._modality_for_storage_path(dna_storage_path))}[dna_reference_hybe]
            rna_fid = alignment.pick_channel_by_type(rna_rec, 'fiducial')
            dna_fid = alignment.pick_channel_by_type(dna_rec, 'fiducial')
        except (KeyError, TypeError):
            pass
        ap.RunAllCrossModalPushButton.setEnabled(False)
        self.statusBar().showMessage('Running cross-modal alignment for all FOVs...')
        self._cross_modal_worker = CrossModalAlignmentWorker(rna_storage_path, dna_storage_path, fov_list, self.fov_matrices,
                                                              rna_reference_hybe, dna_reference_hybe, channel_type,
                                                              border_trim=border_trim, max_shift=max_shift,
                                                              rna_fiducial_channel=rna_fid, dna_fiducial_channel=dna_fid)
        self._cross_modal_worker.progress.connect(lambda done, total, msg: self.statusBar().showMessage(msg))
        self._cross_modal_worker.finished_ok.connect(
            lambda results: self._on_cross_modal_all_finished(results, rna_storage_path, dna_storage_path,
                                                               rna_reference_hybe, dna_reference_hybe, channel_type))
        self._cross_modal_worker.failed.connect(self._on_cross_modal_all_failed)
        self._cross_modal_worker.start()

    def _on_cross_modal_all_finished(self, results, rna_storage_path, dna_storage_path, rna_reference_hybe, dna_reference_hybe, channel_type):
        # worker now returns {'H': {fov: H}, 'z': {fov: planes}} -- the z
        # drift is part of the cross-modal RESULT, staged and accepted with it.
        if isinstance(results, dict) and 'H' in results and 'z' in results:
            self._pending_cross_modal_z = dict(results['z'])
            results = results['H']
        ap = self.ui.AlignmentPanel
        ap.RunAllCrossModalPushButton.setEnabled(True)
        self.statusBar().showMessage('Cross-modal alignment computed.', 5000)
        self._cross_modal_context = {'rna_storage_path': rna_storage_path, 'dna_storage_path': dna_storage_path,
                                      'rna_reference_hybe': rna_reference_hybe, 'dna_reference_hybe': dna_reference_hybe,
                                      'channel_type': channel_type}
        self.cross_modal_result.update({(dna_storage_path, fov): H for fov, H in results.items()})
        self._pending_cross_modal = None
        self._refresh_cross_modal_results_list()
        for fov, H in results.items():
            save_path = paths.figure_path(dna_storage_path, 'alignment', fov, 'cross_modal_overlay.png')
            self.preview_canvas.draw_cross_modal_preview(rna_storage_path, dna_storage_path, fov,
                                                 rna_reference_hybe, dna_reference_hybe, channel_type, H, save_path=save_path,
                                                 rna_fov_matrices=self._fov_matrices_for(rna_storage_path, fov),
                                                 dna_fov_matrices=self._fov_matrices_for(dna_storage_path, fov))
            self._commit_cross_modal_z(fov, self._pending_cross_modal_z.get(fov))
            self._mirror_cross_modal_params_to_vlinks(rna_storage_path, dna_storage_path, fov, H,
                                                      rna_reference_hybe, dna_reference_hybe, channel_type)
        QtWidgets.QMessageBox.information(self, 'Cross-modal alignment complete',
                                          f'{len(results)} FOV(s) aligned and saved; overlay image(s) written.')

    def _on_cross_modal_all_failed(self, message):
        self.ui.AlignmentPanel.RunAllCrossModalPushButton.setEnabled(True)
        self.statusBar().clearMessage()
        QtWidgets.QMessageBox.critical(self, 'Cross-modal alignment error', message)

    def _accept_cross_modal(self):
        ap = self.ui.AlignmentPanel
        ctx = self._cross_modal_context
        if not self._pending_cross_modal or ctx is None:
            return
        for fov, H in self._pending_cross_modal.items():
            save_path = paths.figure_path(ctx['dna_storage_path'], 'alignment', fov, 'cross_modal_overlay.png')
            self.preview_canvas.draw_cross_modal_preview(ctx['rna_storage_path'], ctx['dna_storage_path'], fov,
                                                 ctx['rna_reference_hybe'], ctx['dna_reference_hybe'], ctx['channel_type'],
                                                 H, save_path=save_path,
                                                 rna_fov_matrices=self._fov_matrices_for(ctx['rna_storage_path'], fov),
                                                 dna_fov_matrices=self._fov_matrices_for(ctx['dna_storage_path'], fov))
            self._commit_cross_modal_z(fov, self._pending_cross_modal_z.get(fov))
            self._mirror_cross_modal_params_to_vlinks(ctx['rna_storage_path'], ctx['dna_storage_path'], fov, H,
                                                      ctx['rna_reference_hybe'], ctx['dna_reference_hybe'], ctx['channel_type'])
        self.cross_modal_result.update({(ctx['dna_storage_path'], fov): H for fov, H in self._pending_cross_modal.items()})
        self._pending_cross_modal = None
        self._refresh_cross_modal_results_list()
        ap.CrossModalAcceptPushButton.setEnabled(False)
        ap.CrossModalRejectPushButton.setEnabled(False)
        QtWidgets.QMessageBox.information(self, 'Cross-modal alignment accepted', 'Result written to H5; overlay image(s) saved.')

        # Matrices changed -> every spot's shared-frame coordinate (and
        # possibly its owner) is stale. Assignment is cheap; recompute for
        # every loaded FOV and persist all slices, per explicit decision
        # that saving cells, matrices, or spots all re-run assignment.
        for _fov in list(self.fov_matrices.keys()):
            try:
                self._recast_persisted_spots(_fov)
            except Exception as e:
                self.statusBar().showMessage(f'spot reassignment after accept failed for FOV{_fov}: {e}')

    def _reject_cross_modal(self):
        ap = self.ui.AlignmentPanel
        self._pending_cross_modal = None
        self._refresh_cross_modal_results_list()
        ap.CrossModalAcceptPushButton.setEnabled(False)
        ap.CrossModalRejectPushButton.setEnabled(False)

    def _show_cross_modal_overlay(self):
        """
        Cross-modal alignment used to have no dedicated overlay viewer at
        all -- the only preview was the one auto-shown right after Run
        Cross-Modal Alignment, using whatever storage paths/reference
        hybes/channel_type were selected AT THAT TIME. This reads all of
        those LIVE from the panel every click instead, so re-viewing an
        already-computed result with a freshly-changed channel_type (or
        just checking a different FOV) doesn't require re-running the
        alignment. Matrix source, in priority order: a staged (not-yet-
        accepted) result, the in-memory cross_modal_result cache, then
        vlinks_store.read_cross_modal_matrix as a last resort -- same
        "never require re-computation just to look" pattern used by the
        within-experiment/cell overlay viewers. vlinks is the only disk
        source; nothing here reads a raw stack file.
        """
        ap = self.ui.AlignmentPanel
        rna_storage_path = ap.RnaStoragePathLineEdit.text().strip()
        dna_storage_path = ap.DnaStoragePathLineEdit.text().strip()
        rna_reference_hybe = ap.RnaReferenceHybeComboBox.currentText().strip()
        dna_reference_hybe = ap.DnaReferenceHybeComboBox.currentText().strip()
        channel_type = ap.ChannelTypeComboBox.currentText()
        if not all([rna_storage_path, dna_storage_path, rna_reference_hybe, dna_reference_hybe]):
            QtWidgets.QMessageBox.warning(self, 'Show Cross-Modal Overlay',
                                          'Fill in both storage paths and reference hybes first.')
            return

        fov = ap.CrossModalOverlayFovSpinBox.value()

        H = None
        if self._pending_cross_modal is not None:
            H = self._pending_cross_modal.get(fov)
        if H is None:
            H = self.cross_modal_result.get((dna_storage_path, fov))
        if H is None:
            # vlinks_store's own mirror is the ONLY accepted disk source --
            # reference-hybe-independent, immune to a not-yet-reconciled combo
            # (see _refresh_cross_modal_results_list's own comment for the full
            # reasoning). Nothing here is a real answer -- "not accepted yet",
            # reported below -- not a reason to scavenge
            # {dna_reference_hybe}_stack.h5's own /matrix_across.
            H = vlinks_store.read_cross_modal_matrix(dna_storage_path, fov)
        if H is None:
            QtWidgets.QMessageBox.warning(self, 'Show Cross-Modal Overlay',
                                          f'No cross-modal result for FOV{fov:02d} yet -- run alignment first.')
            return

        self.alignment_preview_window.show()
        self.alignment_preview_window.raise_()
        self.preview_canvas.draw_cross_modal_preview(rna_storage_path, dna_storage_path, fov,
                                             rna_reference_hybe, dna_reference_hybe, channel_type, H,
                                             rna_fov_matrices=self._fov_matrices_for(rna_storage_path, fov),
                                             dna_fov_matrices=self._fov_matrices_for(dna_storage_path, fov))

    # -- cell-based alignment --

    def _storage_path_for_modality(self, modality):
        """
        Forward lookup, mirroring _modality_for_storage_path's reverse
        one: which storage_path is configured for this modality, or ''
        if none/not yet set. Needed once a hybe-choosing combo (Spot
        Localization, Same-Modality Alignment, Cell Segmentation, MIP
        Viewer) can offer hybes from ANY modality at once -- the storage
        path to actually read from vlinks.h5 must come from the SELECTED
        hybe's own modality, not from whatever the Ingestion tab's own
        combo happens to be showing right now (those two can now genuinely
        differ, unlike before this modality decoupling).
        """
        return self.ui.IngestionPanel.modality_data.get(modality, {}).get('storage_path', '') if modality else ''

    def _modality_for_storage_path(self, storage_path):
        """
        Reverse-lookup: which configured modality name owns this storage
        path, or None if it doesn't match any. Used wherever cell-level
        alignment matrices need to be tagged/read by (hybe, modality) key
        but only a storage_path is in scope -- a cell's own reference pair is NOT a
        substitute for this (a cell's own segmentation modality doesn't
        tell you which modality a given storage_path/hybe_records
        argument belongs to; those can be the SAME cell's own modality or
        the OTHER one, depending on which call site this is).
        """
        for name, data in self.ui.IngestionPanel.modality_data.items():
            if data.get('storage_path') == storage_path:
                return name
        return None

    def _shared_frame_modality(self):
        """
        Which modality's own within-experiment frame IS the pipeline's ONE
        shared frame -- always the RNA side of the cross-modal pair (the
        side whose own H_across is identity by design; see ACell.
        matrix_to_shared: "there is no separate DNA's own shared frame").
        None when no cross-modal pair is configured at all, in which case
        each caller falls back to its own modality (nothing to bridge).
        """
        rna_storage_path = self.ui.AlignmentPanel.RnaStoragePathLineEdit.text().strip()
        return self._modality_for_storage_path(rna_storage_path) if rna_storage_path else None

    def _frame_resolver(self, cell, fov):
        """
        Build the ONE resolver (alignment/frames.py) from live session
        state. Everything frame-related should route through this; the
        older per-question resolvers remain only as thin wrappers.

        `within` is deliberately the RAW per-modality matrices straight
        out of self.fov_matrices -- NOT _composed_fov_matrices_for_cell_
        alignment's output, which already has the cross-modal bridge
        folded in. FrameResolver applies the bridge itself, so handing it
        pre-bridged matrices would count that leg twice.
        """
        ap = self.ui.AlignmentPanel
        rna_sp = ap.RnaStoragePathLineEdit.text().strip()
        dna_sp = ap.DnaStoragePathLineEdit.text().strip()
        rna_modality = self._modality_for_storage_path(rna_sp) if rna_sp else None
        dna_modality = self._modality_for_storage_path(dna_sp) if dna_sp else None
        shared = rna_modality or (cell.reference_modality if cell is not None else None)
        if shared is None:
            # single-modality mode has no cross-modal RNA path to name the
            # shared frame, but the shared frame plainly IS the one
            # configured modality -- without this, every cell-less
            # transform (unassigned-spot recast, FOV-level mapping)
            # resolved to None and callers skipped work that identity
            # answers exactly (the alignment-requirement violation again).
            configured = [n for n in self.ui.IngestionPanel.modality_names
                          if self._storage_path_for_modality(n)]
            if len(configured) == 1:
                shared = configured[0]

        within = {}
        for modality in self.ui.IngestionPanel.modality_names:
            sp = self._storage_path_for_modality(modality)
            if sp:
                # FrameResolver's `within` is {modality: {hybe: 3x3}} with BARE
                # hybe keys -- the outer key already names the modality, so it
                # is unambiguous there. Strip the pair rather than handing it
                # tuple keys, which would miss every lookup and silently
                # resolve to identity.
                within[modality] = {h: H for (h, _m), H in self._fov_matrices_for(sp, fov).items()}

        bridge_xy = None
        bridge_z = 0.0
        if rna_modality and dna_modality and rna_modality != dna_modality:
            bridge_xy = self.cross_modal_result.get((dna_sp, fov))
            if bridge_xy is None:
                bridge_xy = vlinks_store.read_cross_modal_matrix(dna_sp, fov)
            bridge_z = self.cross_modal_z.get((dna_sp, fov))
            if bridge_z is None:
                bridge_z = vlinks_store.read_cross_modal_z(dna_sp, fov)

        # Anchors are MODALITY-level facts (each modality's own cell-
        # alignment reference hybe's live FOV matrix, in the shared
        # frame), not per-cell state -- populate them for every configured
        # modality unconditionally. The old cell-gated version left
        # anchors EMPTY whenever the resolver was built with cell=None
        # (the 3D localization and chromatin-tracing paths), which
        # silently dropped every residual-bearing cell's own correction
        # down to the bare FOV route. Derived from THIS resolver's own
        # within+bridge -- never via _fov_matrices_for_cell_modality,
        # whose bridge step builds another resolver (confirmed real
        # RecursionError). The cell's stored snapshot remains the
        # fallback when the live combo has no pick.
        resolver = frames.FrameResolver(within, shared, bridge_xy=bridge_xy,
                                        bridge_z=float(bridge_z or 0.0),
                                        bridge_from=dna_modality if bridge_xy is not None else None)
        anchors = {}
        for modality in self.ui.IngestionPanel.modality_names:
            anchor_hybe = ap.current_cell_reference_hybe(modality)
            H_a = within.get(modality, {}).get(anchor_hybe) if anchor_hybe else None
            if H_a is not None:
                b = resolver.bridge(modality, shared) if shared else None
                anchors[modality] = (b if b is not None else np.eye(3)) @ np.asarray(H_a, dtype=float)
            elif cell is not None and modality in cell.matrix_anchors:
                anchors[modality] = cell.matrix_anchors[modality]
        resolver.anchors = anchors
        return resolver

    def _cross_modal_bridge(self, from_modality, to_modality, fov):
        """
        from_modality's frame -> to_modality's. Thin wrapper over
        frames.FrameResolver.bridge -- the direction rule lives there, in
        one place, shared with the Z leg.
        """
        return self._frame_resolver(None, fov).bridge(from_modality, to_modality)
    def _commit_cross_modal_z(self, fov, dz=None):
        """
        Persist this FOV's measured z drift alongside its H_across. `dz`
        comes from the SAME alignment run that produced H -- it is a
        measured component of the result, never a user-set parameter.
        """
        dna_sp = self.ui.AlignmentPanel.DnaStoragePathLineEdit.text().strip()
        if not dna_sp or dz is None:
            return
        self.cross_modal_z[(dna_sp, fov)] = float(dz)
        vlinks_store.write_cross_modal_z(dna_sp, fov, float(dz))

    def _cross_modal_z(self, from_modality, to_modality, fov):
        """
        Planes to ADD moving z between modality frames. Wrapper over
        frames.FrameResolver.bridge_z_between, so the XY and Z legs can no
        longer disagree about direction -- they read the same rule.
        """
        return self._frame_resolver(None, fov).bridge_z_between(from_modality, to_modality)
    def _fov_matrices_for(self, storage_path, fov):
        """
        One modality's FOV matrices, named by the storage_path that selects
        it. Empty (not None) when nothing is cached for that FOV.

        self.fov_matrices is keyed by FOV alone and holds every modality's
        entries under (hybe, modality) keys. storage_path is no longer part
        of the key -- it used to smuggle modality into an outer tuple, which
        is what forced the inner dict to use a bare hybe and made the bridge
        hybe ambiguous. It now does one job: naming which modality this
        caller means, at the boundary where it already knows.
        """
        if not storage_path:
            return alignment.FrameMatrices()
        return self.fov_matrices.get(fov, alignment.FrameMatrices()).for_modality(
            vlinks_store.modality_of(storage_path))

    def _merge_fov_matrices(self, fov, matrices):
        """
        Fold one modality's matrices into the FOV's shared store.

        Merge, never replace: the store spans both modalities now, so
        assigning a single modality's dict over it would silently drop the
        other one -- the same shape of bug as the spot-slice overwrite.
        """
        current = self.fov_matrices.get(fov)
        if current is None:
            current = alignment.FrameMatrices()
            self.fov_matrices[fov] = current
        current.update(matrices)
        current.modality = None      # spans modalities; callers must narrow
        return current

    def _fov_matrices_in_frame(self, source_modality, frame_modality, fov):
        """
        {hybe: 3x3} for every one of source_modality's OWN hybes, each
        mapping that hybe's own native frame into frame_modality's own
        within-experiment frame -- the one generalized primitive both
        same-modality and cross-modal callers now use, differing only in
        what they pass as frame_modality.

        Composition is always [within, bridge]: the same-modality layer
        first, then _cross_modal_bridge's own (from -> to) factor, which
        is identity whenever the two frames coincide -- so RNA->RNA and
        DNA->DNA fall out of the SAME expression as RNA->DNA/DNA->RNA
        rather than needing their own branch.

        Returns None when the needed layer genuinely isn't available (no
        within-experiment matrices for this FOV, or a required cross-modal
        link that hasn't been accepted) -- a missing hybe WITHIN an
        otherwise-available dict still degrades to identity at lookup time
        (see hybe_to_cellref_matrix's own fov_matrices.get(hybe, eye)),
        which is the "works even without each layer" property.
        """
        source_storage_path = self._storage_path_for_modality(source_modality)
        if not source_storage_path:
            return None
        within = self._fov_matrices_for(source_storage_path, fov)
        if not within:
            return None
        bridge = self._cross_modal_bridge(source_modality, frame_modality, fov)
        if bridge is None:
            return None
        if np.allclose(bridge, np.eye(3)):
            return alignment.FrameMatrices(within, modality=within.modality)
        return alignment.FrameMatrices(
            {key: alignment.compose_chain([H, bridge]) for key, H in within.items()},
            modality=within.modality)

    def _composed_fov_matrices_for_cell_alignment(self, storage_path, fov):
        """
        The FOV-level matrices to use as compute_cell_alignment's "H1"
        input for this (storage_path, fov), expressed in the pipeline's
        ONE shared frame (see _shared_frame_modality). Thin wrapper over
        _fov_matrices_in_frame now -- kept as its own name because
        several call sites have only a storage_path in scope. Returns a
        NEW dict -- never mutates self.fov_matrices, so /matrix/{hybe}'s
        on-disk meaning stays strictly within-experiment for any other
        reader. Falls back to the raw within-experiment matrices when no
        shared frame is resolvable (no cross-modal pair configured),
        matching this function's own long-standing "nothing to compose
        in yet" behavior rather than returning None.
        """
        modality = self._modality_for_storage_path(storage_path)
        frame_modality = self._shared_frame_modality() or modality
        composed = self._fov_matrices_in_frame(modality, frame_modality, fov) if modality else None
        return composed if composed is not None else self._fov_matrices_for(storage_path, fov)

    def _other_modality_cell_alignment_inputs(self, storage_path, fov):
        """
        For genuinely cross-modal cell alignment (per explicit request:
        once same-modality AND cross-modality layers are both
        established, a cell should get residual-refined against the
        OTHER modality's hybes too, not just its own) -- returns
        (other_storage_path, other_hybe_records, other_fov_matrices,
        other_reference_hybe, other_modality) for the OTHER modality, or
        None if storage_path isn't one of the two configured Cross-
        Modality Alignment paths, or no cross-modal result exists yet
        for this FOV.

        other_fov_matrices: {hybe: 3x3} -- each of the OTHER modality's
        own within-experiment matrices, bridged into the pipeline's ONE
        shared frame (_shared_frame_modality), exactly like
        _composed_fov_matrices_for_cell_alignment's own output for the
        same-modality case -- both now go through the SAME
        _fov_matrices_in_frame primitive, so the two legs are guaranteed
        to land in one frame rather than agreeing only by construction.
        Per confirmed real bug, this used to bridge into storage_path's
        (the CELL's) own frame instead, inverting H_across whenever the
        cell sat on the DNA side -- which made the destination frame
        depend on the loaded cell's modality and left a DNA cell's two
        legs disagreeing by 2x H_across (see _cross_modal_bridge's own
        docstring for the measured numbers). Previously excluded any hybe name already
        present in self.hybe_records (the SAME modality's own set) --
        the shared cross-modal bridge hybe (e.g. Hyb_130) is a real,
        DISTINCT file in BOTH modalities, and that exclusion silently
        dropped the other modality's own copy of it entirely (its own
        real, computable alignment just never existed). Now that
        cell.matrices/matrix_provenance are keyed by (hybe, modality),
        the same-modality entry and this modality's entry for a same-
        named hybe coexist without colliding, so the exclusion is gone.

        other_reference_hybe: that modality's own WITHIN-EXPERIMENT
        reference hybe (e.g. DNA's own Hyb_002) -- used as
        compute_cell_alignment's own phase-correlation anchor for this
        second call, since the same-modality run's own reference hybe
        (e.g. Hyb_101) doesn't exist in the other modality's hybe_records
        at all. Can now legitimately be the cross-modal bridge hybe too
        (e.g. Hyb_130) -- no longer needs to avoid it, for the same
        reason as above.

        other_modality: the OTHER modality's own configured name (e.g.
        'DNA') -- the caller must pass this through to
        compute_cell_alignment's own modality= parameter so every write
        it makes is correctly tagged.
        """
        ap = self.ui.AlignmentPanel
        rna_storage_path = ap.RnaStoragePathLineEdit.text().strip()
        dna_storage_path = ap.DnaStoragePathLineEdit.text().strip()
        if not rna_storage_path or not dna_storage_path or rna_storage_path == dna_storage_path:
            return None
        if storage_path == rna_storage_path:
            other_storage_path = dna_storage_path
        elif storage_path == dna_storage_path:
            other_storage_path = rna_storage_path
        else:
            return None

        other_modality = self._modality_for_storage_path(other_storage_path)
        other_data = self.ui.IngestionPanel.modality_data.get(other_modality) if other_modality else None
        if not other_data or not other_data['layout_path']:
            return None
        # the real, persisted same-modality reference hybe for this
        # storage_path -- read from vlinks.h5 global params (see
        # _reference_hybe_for_storage_path), not modality_data, since
        # that no longer tracks a per-modality reference hybe at all
        # (Same-Modality Alignment's own reference-hybe combo isn't
        # modality-switch-scoped any more).
        other_reference_hybe = self._reference_hybe_for_storage_path(other_storage_path)
        if not other_reference_hybe:
            return None
        try:
            other_hybe_records = preprocess.parse_experiment_layout(other_data['layout_path'])
        except Exception:
            return None

        # ONE shared frame for both legs, never storage_path's own -- see
        # this function's own other_fov_matrices docstring and
        # _cross_modal_bridge for the measured bug that motivated it.
        frame_modality = self._shared_frame_modality() or other_modality
        other_fov_matrices = self._fov_matrices_in_frame(other_modality, frame_modality, fov)
        if not other_fov_matrices:
            return None
        return other_storage_path, other_hybe_records, other_fov_matrices, other_reference_hybe, other_modality

    def _cell_alignment_passes(self, cell_modality, storage_path, fov):
        """
        One compute_cell_alignment "pass" dict per configured modality --
        what CellAlignmentWorker consumes (see its own docstring). The
        cell's OWN modality always comes first; the other modality is
        appended only when a real cross-modal link exists for this FOV
        (_other_modality_cell_alignment_inputs), so a single-modality or
        not-yet-cross-modally-linked session degrades to exactly the old
        one-pass behavior.

        Every pass anchors on THAT MODALITY'S OWN cell alignment
        reference (ap.current_cell_reference_hybe(modality)) -- NOT
        _other_modality_cell_alignment_inputs' own 4th return value,
        which is that modality's SAME-MODALITY FOV-ALIGNMENT reference
        (a different, independently-configured setting; see
        _reference_hybe_for_storage_path). Using the cell-alignment one
        is what keeps every residual fit strictly within one modality
        while still being anchored where the rest of the cell-alignment
        layer expects.

        'cellref_fov_matrices' is {modality: composed matrices} for EVERY
        configured modality, identical across passes -- the frame cell.area
        is native to is a PER-CELL fact (cell.reference_modality), not a
        per-pass one, once cytoplasmic segmentation can move a cell's
        reference_hybe into the other modality. The worker picks the right
        entry per cell. Every modality's matrices are already composed into
        the one shared frame (see _fov_matrices_in_frame), so they are
        directly comparable.
        """
        ap = self.ui.AlignmentPanel
        own_fov_matrices = self._composed_fov_matrices_for_cell_alignment(storage_path, fov)
        # Keyed by modality: a cell's own reference_hybe may live in EITHER
        # modality once a cytoplasm is attached, and that is a per-cell fact
        # the worker has to resolve, not something a per-FOV pass can bake in.
        frames_by_modality = {cell_modality: own_fov_matrices}
        passes = [{
            'modality': cell_modality,
            'storage_path': storage_path,
            'hybe_records': self._active_hybe_records_for_modality(cell_modality),
            'fov_matrices': own_fov_matrices,
            'reference_hybe': ap.current_cell_reference_hybe(cell_modality) or None,
            'cellref_fov_matrices': frames_by_modality,
        }]
        other = self._other_modality_cell_alignment_inputs(storage_path, fov)
        if other is not None:
            other_storage_path, _, other_fov_matrices, _, other_modality = other
            frames_by_modality[other_modality] = other_fov_matrices
            other_reference_hybe = ap.current_cell_reference_hybe(other_modality)
            other_records = self._active_hybe_records_for_modality(other_modality)
            # No configured cell-alignment reference for that modality (or
            # nothing ingested there) means there's nothing well-defined to
            # anchor its own fit against -- skip that pass rather than
            # silently falling back to some other hybe, leaving those hybes
            # on the FOV/cross-modal matrix alone exactly as before.
            if other_reference_hybe and other_records:
                passes.append({
                    'modality': other_modality,
                    'storage_path': other_storage_path,
                    'hybe_records': other_records,
                    'fov_matrices': other_fov_matrices,
                    'reference_hybe': other_reference_hybe,
                    'cellref_fov_matrices': frames_by_modality,
                })
        return passes

    def _fov_matrices_for_cell_modality(self, modality, cell, fov):
        """
        The already-composed {hybe: H_to_shared} dict for `modality`,
        every entry expressed in the pipeline's ONE shared frame (see
        _shared_frame_modality) regardless of which modality `cell` itself
        was segmented in. Returns None if that FOV-level layer isn't
        available at all yet (FOV alignment, or the cross-modal link this
        modality would need, hasn't been run/accepted). Every caller
        sources this lookup here, the one place, rather than re-deriving.

        `cell` no longer selects the destination frame -- it only supplies
        the fallback when no cross-modal pair is configured at all (then
        there is nothing to bridge and each modality is its own frame).
        Per confirmed real bug, this used to route the cell's OWN modality
        through _composed_fov_matrices_for_cell_alignment (-> the shared/
        RNA frame) but any OTHER modality through _other_modality_cell_
        alignment_inputs' own cell-relative composition (-> the CELL's
        frame): for a DNA cell those two destinations differ by 2x
        H_across, so a cell's two legs silently disagreed by ~27px on real
        data. Both now go through the same _fov_matrices_in_frame
        primitive with the same frame_modality.
        """
        # cell=None with no shared frame: each modality is its own frame
        # (nothing to bridge), same fallback _shared_frame_modality's own
        # docstring names for every caller.
        frame_modality = self._shared_frame_modality() or (
            cell.reference_modality if cell is not None else modality)
        return self._fov_matrices_in_frame(modality, frame_modality, fov)

    # (_fov_only_matrix_for_hybe was deleted: it had no callers left and
    # carried its OWN composition (hybe_to_cellref_matrix over two dict
    # lookups) -- a second resolver in waiting, which is exactly the
    # divergence class every alignment bug here has been. The FOV-only
    # "before" view is a plain _fov_matrices_for_cell_modality lookup;
    # anything needing a frame-to-frame transform goes through
    # frames.FrameResolver.transform. tests/test_transform_single_source
    # pins the remaining independent compositions to the resolver
    # numerically.)

    # (_live_cell_matrix_anchor was folded into _frame_resolver's own
    # anchor derivation -- anchors are computed from the resolver's own
    # within+bridge there, the ONE place, because routing the lookup
    # through _fov_matrices_for_cell_modality builds another resolver and
    # recursed. The stored cell.matrix_anchors snapshot remains the
    # fallback for a cell whose modality has no live combo pick; see
    # _frame_resolver's anchor block for the staleness rationale.)

    def _matrix_to_cellref(self, hybe, modality, cell, fov):
        """hybe's raw frame -> the frame cell.area lives in. Wrapper over _matrix_to_frame."""
        return self._matrix_to_frame(hybe, modality, cell, fov,
                                     cell.reference_hybe, cell.reference_modality)

    def _matrix_to_frame(self, hybe, modality, cell, fov, frame_hybe, frame_modality):
        """
        hybe's raw frame -> (frame_hybe, frame_modality)'s raw frame.

        Thin wrapper over frames.FrameResolver.transform. The target frame
        stays an explicit argument because a cell defines masks in two of
        them -- reference_hybe for `area`, nucleus_hybe for `nucleus` --
        and after cytoplasmic segmentation those can sit in different
        modalities.
        """
        resolver = self._frame_resolver(cell, fov)
        H, _dz, _missing = resolver.transform((hybe, modality),
                                              (frame_hybe, frame_modality or cell.reference_modality), cell)
        return H

    def _matrix_to_shared(self, hybe, modality, cell, fov):
        """
        hybe's own raw frame -> the pipeline's ONE shared frame.

        Thin wrapper over frames.FrameResolver (see that module) -- kept
        because many call sites read better as "to shared" than as a
        transform to an explicit key, but it owns no composition logic of
        its own any more. Returns None only when the shared frame itself
        can't be identified.
        """
        resolver = self._frame_resolver(cell, fov)
        if resolver.shared is None:
            return None
        return resolver.to_shared(hybe, modality, cell)

    def _cell_overlay_target_specs(self, cell, storage_path, fov, hybe_records, channel_type):
        """
        Resolves EVERY active hybe in EVERY configured modality (not just
        cell.matrices' own keys) into what draw_cell_all_readouts_overlay
        needs to read/crop/warp it: storage path, channel (also reused for
        the ZX row -- z-alignment respects channel_type, same as the yx
        fit), the FOV-level matrix (the 'FOV/cross-modal' stage), and this
        cell's own final yx/zx matrices -- both resolved into the
        pipeline's ONE shared frame (RNA's own same-modality reference
        hybe; see ACell.matrix_to_shared's own docstring), which is what
        draw_cell_all_readouts_overlay's crop_via expects. fov_only_matrix
        is a plain fov_matrices lookup (no cell involvement at all -- no
        bridge step left to compute for the FOV/cross-modal stage);
        final_matrix goes through _matrix_to_shared, which already falls
        back to fov_only_matrix's own value whenever this cell has no
        real cell-level residual for that hybe/modality (per confirmed
        real bug fix -- see ACell.matrix_to_shared's own docstring) --
        this cell's own modality naturally gets a real residual for every
        hybe it was aligned against; the OTHER modality (no cell-level
        residual fit is attempted across modality boundaries at all, per
        explicit decision -- see CellAlignmentWorker's own docstring)
        always resolves to fov_only_matrix here, showing 'FOV/cross-modal'
        and 'final' as identical for those hybes, which is the honest
        picture (no per-cell refinement was ever computed for them).

        Iterating every configured modality's own active hybe_records
        directly (not cell.matrices.items()) is what makes other-modality
        hybes show up here at all now that CellAlignmentWorker no longer
        writes ANY cell.matrices entry for them -- previously this
        iterated cell.matrices as its only source, which silently dropped
        a hybe the moment there was no residual computed for it.
        """
        this_modality = self._modality_for_storage_path(storage_path)
        record_by_folder_by_modality = {this_modality: {r['folder']: r for r in hybe_records}}
        storage_path_by_modality = {this_modality: storage_path}
        for name in self.ui.IngestionPanel.modality_names:
            if name == this_modality:
                continue
            other_storage_path = self._storage_path_for_modality(name)
            other_hybe_records = self._active_hybe_records_for_modality(name)
            if not other_storage_path or not other_hybe_records:
                continue
            record_by_folder_by_modality[name] = {r['folder']: r for r in other_hybe_records}
            storage_path_by_modality[name] = other_storage_path

        specs = []
        for modality, record_by_folder in record_by_folder_by_modality.items():
            fov_matrices_for_hybe = self._fov_matrices_for_cell_modality(modality, cell, fov)
            if not fov_matrices_for_hybe:
                continue
            for hybe, record in record_by_folder.items():
                if (hybe, modality) not in fov_matrices_for_hybe:
                    continue
                fov_only_matrix = fov_matrices_for_hybe[(hybe, modality)]
                final_matrix = self._matrix_to_shared(hybe, modality, cell, fov)
                if final_matrix is None:
                    continue
                specs.append({
                    'hybe': hybe, 'modality': modality, 'storage_path': storage_path_by_modality[modality],
                    'channel': alignment.pick_channel_by_type(record, channel_type),
                    'fov_only_matrix': fov_only_matrix,
                    'final_matrix': final_matrix,
                    'dz': alignment.entry_dz(cell.matrices.get((hybe, modality))),
                })
        return specs

    def _run_cell_alignment(self):
        """
        Aligns every cell in the single FOV picked by the tier-1 CellFovSpinBox
        (same field the per-cell tuning tool above uses). Always computes AND
        saves immediately -- no staging/Accept step, since the per-cell tool
        already gives a cheap, reviewable way to validate parameters on one
        real cell before committing to the whole FOV.
        """
        ap = self.ui.AlignmentPanel
        ip = self.ui.IngestionPanel
        fov_list_all = self._parse_fov_list(ip.FovListLineEdit.text())
        fov = ap.CellFovSpinBox.value()
        if fov not in fov_list_all:
            QtWidgets.QMessageBox.warning(self, 'Run Cell Alignment',
                                          f'FOV{fov} is not in the Ingestion tab\'s FOV list.')
            return

        # Cells here are the REAL objects (whichever container has them for
        # this FOV, permanent/saved preferred) -- CellAlignmentWorker mutates
        # them in place, so no staging/deepcopy is needed for an always-save
        # operation. Modality is read directly off the cells themselves
        # (this app only ever holds ONE modality's cells resident in memory
        # at a time, so every cell in the container shares one
        # reference_modality), never a separate picker -- and that
        # modality's own configured reference hybe
        # (ap.current_cell_reference_hybe) is what compute_cell_alignment
        # anchors against.
        container = None
        if self.cell_container_permanent is not None and self.cell_container_permanent.data.get(fov):
            container = self.cell_container_permanent
        elif self.cell_container is not None and self.cell_container.data.get(fov):
            container = self.cell_container
        if container is None:
            QtWidgets.QMessageBox.warning(self, 'Run Cell Alignment',
                                          'No cells segmented for this FOV yet -- run Cell Segmentation first.')
            return
        real_cells = container.get_cells(fov)

        cell_modality = real_cells[0].reference_modality
        cell_reference_hybe = ap.current_cell_reference_hybe(cell_modality) or None
        storage_path = self._storage_path_for_modality(cell_modality)
        hybe_records = self._active_hybe_records_for_modality(cell_modality)
        if not storage_path:
            QtWidgets.QMessageBox.warning(self, 'Run Cell Alignment',
                                          f'No storage path configured for {cell_modality}.')
            return
        # No FOV-alignment prerequisite -- per core principle, absence of
        # alignment at any layer is IDENTITY, never a blocker: every layer
        # must run standalone. (The old gate also read the pre-unification
        # (storage_path, fov) key shape, so it blocked unconditionally.)
        if not self._fov_matrices_for(storage_path, fov):
            self.statusBar().showMessage(
                'No FOV alignment for this FOV yet -- proceeding with identity.', 8000)

        channel_type = ap.CellChannelTypeComboBox.currentText()
        pad = ap.CellPadSpinBox.value()

        ap.RunCellAlignmentPushButton.setEnabled(False)
        self.statusBar().showMessage('Computing cell alignment...')
        worker_jobs = [(fov, real_cells, self._cell_alignment_passes(cell_modality, storage_path, fov))]
        self._cell_alignment_worker = CellAlignmentWorker(worker_jobs, channel_type=channel_type, pad=pad)
        self._cell_alignment_worker.progress.connect(lambda done, total, msg: self.statusBar().showMessage(msg))
        self._cell_alignment_worker.finished_ok.connect(
            lambda results: self._on_cell_alignment_finished(results, fov, container, storage_path,
                                                              cell_reference_hybe, cell_modality, channel_type, pad))
        self._cell_alignment_worker.failed.connect(self._on_cell_alignment_failed)
        self._cell_alignment_worker.start()

    def _on_cell_alignment_finished(self, results, fov, container, storage_path, cell_reference_hybe, cell_modality, channel_type, pad):
        """results: [(fov, cells)] -- cells are the real objects, mutated in
        place by compute_cell_alignment. Always writes to vlinks.h5
        immediately -- no staging/Accept step."""
        ap = self.ui.AlignmentPanel
        ap.RunCellAlignmentPushButton.setEnabled(True)
        self.statusBar().showMessage('Cell alignment computed.', 5000)

        storage_paths = self._all_vlinks_storage_paths()
        # per-modality-suffixed key -- see _refresh_params_from_vlinks'
        # own read side and CellAlignmentWorker's own docstring for why
        # (each modality now has its own independent reference hybe, no
        # more single shared/ambiguous value).
        cell_alignment_kwargs = {f'cell_alignment_reference_hybe_{cell_modality}': cell_reference_hybe} if cell_modality else {}
        for path in storage_paths:
            vlinks_store.write_global_params(path, cell_alignment_channel_type=channel_type,
                                             cell_alignment_pad=pad, **cell_alignment_kwargs)
        # Alignment mutated `container`'s cells in place -- the OTHER
        # tier holds independent copies of the same cells, and any later
        # legitimate cell write from that tier (celltype persist,
        # segmentation save) would silently WIPE these matrices from disk
        # (confirmed real). Propagate the alignment-owned fields to the
        # twin copies before persisting.
        other = (self.cell_container if container is self.cell_container_permanent
                 else self.cell_container_permanent)
        if other is not None and other.data.get(fov):
            src_by_id = {c.id: c for c in container.get_cells(fov)}
            for twin in other.get_cells(fov):
                src = src_by_id.get(twin.id)
                if src is not None:
                    twin.matrices = deepcopy(src.matrices)
                    twin.matrix_anchors = deepcopy(src.matrix_anchors)
                    twin.matrix_provenance = deepcopy(src.matrix_provenance)
        if storage_paths:
            vlinks_store.mirror_write_cells(storage_paths, fov, container)

        total_cells = sum(len(cells) for _, cells in results)
        auto_save_threshold = ap.CellOverlayAutoSaveThresholdSpinBox.value()
        n_auto_saved = 0
        for _, cells in results:
            for cell in cells:
                # Drawing+saving one overlay costs ~9x the cell's own
                # alignment fit (real profiling: ~1.6s vs ~0.18s/cell) --
                # doing this unconditionally for every cell is exactly what
                # made this take "a minute" for a FOV's worth of cells. Only
                # auto-save the ones whose own residual shift is large
                # enough to be worth a human look; Save All Cell Overlays
                # covers the rest on demand.
                if self._cell_max_residual_shift(cell) > auto_save_threshold:
                    if self._save_cell_overlay(cell, fov, storage_path, channel_type, pad,
                                               overlay_reference_hybe=cell_reference_hybe, modality=cell_modality):
                        n_auto_saved += 1

        self._refresh_cell_fov_panels(fov)
        QtWidgets.QMessageBox.information(self, 'Cell alignment complete',
                                          f'{total_cells} cell(s) in FOV{fov:02d} aligned, saved to vlinks.h5; '
                                          f'{n_auto_saved} overlay image(s) auto-saved (shift > {auto_save_threshold}px). '
                                          f'Use Save All Cell Overlays to generate the rest on demand.')

    @staticmethod
    def _cell_max_residual_shift(cell):
        """
        The largest cell-level residual correction (compute_cell_alignment's
        own H2, before composition with the FOV/cross-modal matrices)
        across this cell's hybes, in px -- cell.matrix_provenance[hybe]
        ['steps'] is [H1, H2] (see compute_cell_alignment), so steps[1] is
        always this cell's own fine-tuning fit, not the FOV-level shift
        that H1 already carries. Used to flag cells worth a human look
        (see CellOverlayAutoSaveThresholdSpinBox) -- the residual is the
        right quantity here, not the final composed matrix's translation,
        since a legitimately large FOV-level drift shouldn't itself
        trigger a "was this cell's OWN fit unusual" flag.
        """
        if not cell.matrix_provenance:
            return 0.0
        return max(
            float(np.hypot(prov['steps'][1][0, 2], prov['steps'][1][1, 2]))
            for prov in cell.matrix_provenance.values()
        )

    def _save_cell_overlay(self, cell, fov, storage_path, channel_type, pad, overlay_reference_hybe=None, modality=None):
        """Draws + saves one cell's all-readouts overlay PNG. overlay_
        reference_hybe/modality should be the SAME alignment run's own
        anchor hybe/modality this cell was actually aligned against
        (falls back to cell.reference_hybe/reference_modality -- the
        segmentation hybe -- only when not given, matching compute_cell_
        alignment's own reference_hybe=None default) -- using cell.
        reference_hybe unconditionally here previously redrew the overlay
        against a DIFFERENT hybe than the one actually used to compute
        it, which crashed _read_mip whenever that hybe wasn't ingested
        under storage_path's own modality. Returns False (no-op) if the
        resolved reference hybe's record can't be found in that
        modality's own hybe list, matching the automatic-mode skip that
        already existed before this was factored out."""
        modality = modality or cell.reference_modality
        overlay_reference_hybe = overlay_reference_hybe or cell.reference_hybe
        hybe_records = self._active_hybe_records_for_modality(modality) if modality else self.hybe_records
        record_by_folder = {r['folder']: r for r in hybe_records}
        reference_record = record_by_folder.get(overlay_reference_hybe)
        if reference_record is None:
            return False
        save_path = paths.figure_path(storage_path, 'cells', fov, f'cell{cell.id}_alignment_overlay.png')
        reference_channel = alignment.pick_channel_by_type(reference_record, channel_type)
        target_specs = self._cell_overlay_target_specs(cell, storage_path, fov, hybe_records, channel_type)
        mask_anchor_fov_matrix = (self._fov_matrices_for_cell_modality(cell.reference_modality, cell, fov) or {}).get(
            (cell.reference_hybe, cell.reference_modality), np.eye(3))
        self.preview_canvas.draw_cell_all_readouts_overlay(
            cell, fov, overlay_reference_hybe, storage_path, reference_channel,
            target_specs, pad=pad, save_path=save_path, mask_anchor_fov_matrix=mask_anchor_fov_matrix)
        return True

    def _save_all_cell_overlays(self):
        """On-demand batch save of every currently-computed cell's overlay
        PNG (self._cell_alignment_display_cells, populated by the last Run
        Cell Alignment call, manual or automatic) -- lets a user skim the
        whole run's alignment quality by eye without needing every cell to
        have tripped the auto-save-on-large-shift threshold. Reads the
        SAME live tier-1 controls (reference hybe/channel/pad) Run Cell
        Alignment itself reads -- matches this button's own "reuse
        whatever the panel currently shows" contract. Modality is read
        directly off the first cell in the list -- every cell here comes
        from the SAME single-modality container (self.cell_container/
        cell_container_permanent), so they all share one modality anyway,
        no separate picker needed."""
        ap = self.ui.AlignmentPanel
        if not self._cell_alignment_display_cells:
            QtWidgets.QMessageBox.warning(self, 'Save All Cell Overlays', 'Run Cell Alignment first.')
            return
        cell_modality = self._cell_alignment_display_cells[0][1].reference_modality
        storage_path = self._storage_path_for_modality(cell_modality)
        if not storage_path:
            QtWidgets.QMessageBox.warning(self, 'Save All Cell Overlays', f'No storage path configured for {cell_modality}.')
            return
        cell_reference_hybe = ap.current_cell_reference_hybe(cell_modality) or None
        channel_type = ap.CellChannelTypeComboBox.currentText()
        pad = ap.CellPadSpinBox.value()
        n_saved = 0
        for fov, cell in self._cell_alignment_display_cells:
            if self._save_cell_overlay(cell, fov, storage_path, channel_type, pad,
                                       overlay_reference_hybe=cell_reference_hybe, modality=cell_modality):
                n_saved += 1
        QtWidgets.QMessageBox.information(self, 'Save All Cell Overlays', f'{n_saved} overlay image(s) saved.')

    def _on_cell_alignment_failed(self, message):
        ap = self.ui.AlignmentPanel
        ap.RunCellAlignmentPushButton.setEnabled(True)
        self.statusBar().clearMessage()
        QtWidgets.QMessageBox.critical(self, 'Cell alignment error', message)

    def _show_cell_alignment_preview(self, item):
        """
        Every row in "Results (per cell, per hybe)" belongs to the SAME
        single cell -- the row itself IS the target-hybe choice, so
        clicking it previews that hybe directly. The reference side is no
        longer a single fixed choice (see _show_cell_alignment_preview_
        for_hybe -- one block per configured modality's own reference
        hybe now), so only target_key needs to be remembered here.
        """
        pctx = self._cell_per_hybe_context
        if pctx is None:
            return
        fov, cell_id, hybe, modality = item.data(QtCore.Qt.UserRole)
        cell = pctx['cell']
        target_key = (hybe, modality)
        self._cell_preview_context = {'fov': fov, 'cell': cell, 'storage_path': pctx['storage_path'],
                                      'hybe_records': pctx['hybe_records'], 'target_key': target_key}
        self._show_cell_alignment_preview_for_hybe(target_key=target_key)

    def _resolve_preview_hybe_context(self, hybe, modality, storage_path, hybe_records, fov):
        """
        (record, resolved_storage_path) for `hybe`/`modality`, whether
        that's this cell-alignment context's own modality (storage_path/
        hybe_records passed straight through) or the OTHER one
        (re-derived fresh via _other_modality_cell_alignment_inputs, same
        as at Run Cell Alignment time) -- resolved directly from the
        given modality, not inferred from name membership (the old,
        collision-prone heuristic). Returns (None, None, error_message)
        on failure so the caller can report it via the status bar instead
        of silently doing nothing. Shared by target and reference
        resolution in _show_cell_alignment_preview_for_hybe, since either
        one can now independently be either modality (free choice, not
        tied to whatever the target's own provenance recorded).
        """
        this_modality = self._modality_for_storage_path(storage_path)
        if modality == this_modality:
            record_by_folder = {r['folder']: r for r in hybe_records}
            if hybe not in record_by_folder:
                return None, None, f"{hybe} ({modality}) isn't in this modality's parsed layout."
            return record_by_folder[hybe], storage_path, None
        # Another modality's hybe: resolve it from the Ingestion tab's own
        # configuration DIRECTLY -- per the identity-default rule, an
        # unaccepted cross-modal bridge is identity (provisional), never a
        # refusal, so this deliberately has NO alignment requirement (the
        # old version demanded an accepted cross-modal result and refused
        # otherwise -- the exact violated principle, third recurrence).
        data = self.ui.IngestionPanel.modality_data.get(modality)
        if not data or not data.get('storage_path') or not data.get('layout_path'):
            return None, None, f"{modality} isn't a configured modality (Ingestion tab)."
        try:
            other_records = preprocess.parse_experiment_layout(data['layout_path'])
        except Exception as e:
            return None, None, f"couldn't parse {modality}'s layout: {e}"
        other_record_by_folder = {r['folder']: r for r in other_records}
        if hybe not in other_record_by_folder:
            return None, None, f"{hybe} ({modality}) isn't in that modality's parsed layout."
        return other_record_by_folder[hybe], data['storage_path'], None

    def _show_cell_alignment_preview_for_hybe(self, target_key=None):
        """
        target_key: (hybe, modality) tuple -- matches cell.matrices' own
        key shape. Defaults to whatever's stored in _cell_preview_context
        (set by _show_cell_alignment_preview from the last-clicked
        Results-list row) -- no separate target combo needed here.

        The reference side is no longer a single fixed choice -- per
        explicit request, one 2x3 block is drawn per configured
        modality's own reference hybe (ap.cell_align_references()), all
        compared against the SAME target hybe. Modalities with no real
        pick, or whose reference hybe fails to resolve, are silently
        skipped (status bar shows target-level errors only; a genuinely
        empty reference_specs list still calls through and shows
        whatever target-only info _draw_three_way can given zero rows,
        rather than silently doing nothing).

        Every reference/target matrix is resolved the exact same way --
        a plain fov_matrices lookup for fov_only_matrix/spec['fov_matrix'],
        _matrix_to_shared for final_matrix/spec['final_matrix'] -- since
        draw_cell_alignment_preview_3col only ever needs each hybe's OWN
        matrix independently expressed relative to the pipeline's ONE
        shared reference frame (each crops the shared-frame mask via its
        own inverse-warp, then the crops are composited for visual
        comparison), never a direct target-vs-reference matrix. Target
        and any reference don't need to share a modality -- any hybe can
        be compared against any other hybe directly, RNA against DNA
        included (see ACell.matrix_between).
        """
        pctx = getattr(self, '_cell_preview_context', None)
        if pctx is None:
            return
        ap = self.ui.AlignmentPanel
        if target_key is None:
            target_key = pctx.get('target_key')
        if not target_key:
            return
        target_hybe, target_modality = target_key
        cell, fov = pctx['cell'], pctx['fov']
        storage_path, hybe_records = pctx['storage_path'], pctx['hybe_records']
        channel_type = ap.CellChannelTypeComboBox.currentText()
        pad = ap.CellPadSpinBox.value()

        target_record, target_storage_path, err = self._resolve_preview_hybe_context(
            target_hybe, target_modality, storage_path, hybe_records, fov)
        if err:
            self.statusBar().showMessage(f"Can't preview {target_hybe} ({target_modality}): {err}", 8000)
            return

        # fov_only_matrix is a plain fov_matrices lookup (no cell
        # involvement, no bridge step left to compute); final_matrix goes
        # through _matrix_to_shared for the pipeline's ONE shared
        # reference frame (what draw_cell_alignment_preview_3col's
        # bounds_via now expects -- see ACell.matrix_to_shared).
        target_fov_matrices = self._fov_matrices_for_cell_modality(target_modality, cell, fov)
        fov_only_matrix = (target_fov_matrices or {}).get((target_hybe, target_modality))
        final_matrix = self._matrix_to_shared(target_hybe, target_modality, cell, fov)
        # Missing FOV-level alignment is IDENTITY, never a blocker (core
        # principle: every layer runs standalone) -- say so and proceed.
        if fov_only_matrix is None or final_matrix is None:
            self.statusBar().showMessage(
                f'No FOV-level alignment for {target_hybe} ({target_modality}) yet -- previewing with identity.', 8000)
        if fov_only_matrix is None:
            fov_only_matrix = np.eye(3)
        if final_matrix is None:
            final_matrix = np.eye(3)
        target_channel = alignment.pick_channel_by_type(target_record, channel_type)

        reference_specs = []
        for modality, reference_hybe in ap.cell_align_references().items():
            record, reference_storage_path, err = self._resolve_preview_hybe_context(
                reference_hybe, modality, storage_path, hybe_records, fov)
            if err:
                continue
            reference_fov_matrices = self._fov_matrices_for_cell_modality(modality, cell, fov)
            # Missing matrix = identity (never a skip) -- only a genuinely
            # unresolvable record/storage path (err above) drops a block.
            reference_fov_matrix = (reference_fov_matrices or {}).get((reference_hybe, modality))
            if reference_fov_matrix is None:
                reference_fov_matrix = np.eye(3)
            # 'final_matrix' deliberately OMITTED -- draw_cell_alignment_
            # preview_3col's own default (spec.get('final_matrix',
            # reference_fov_matrix)) then pins the reference/red side to
            # the SAME matrix on 'FOV/cross-modal' AND 'final', per
            # explicit correction of a confirmed real regression: this
            # used to call _matrix_to_shared(reference_hybe, ...)
            # independently here, which, now that reference_hybe is
            # ALWAYS this modality's own cell_align_reference (the
            # residual-fit ANCHOR itself -- see ap.cell_align_references),
            # only ever happened to numerically coincide with reference_
            # fov_matrix (the anchor's own residual against itself is
            # ~identity by construction) rather than being a STRUCTURAL
            # guarantee -- silently reintroducing "red moves between
            # columns" as a live possibility, breaking the established
            # principle that column 3's whole role is showing whether
            # cyan (target) was corrected onto a FIXED red (reference),
            # never the reverse. The Results list's own dx/dy math is a
            # legitimately different use of the TRUE final matrix and
            # still calls _matrix_to_shared directly (see
            # _refresh_cell_per_hybe_results) -- only this preview's own
            # reference crop is pinned.
            reference_specs.append({
                'modality': modality, 'storage_path': reference_storage_path, 'hybe': reference_hybe,
                'channel': alignment.pick_channel_by_type(record, channel_type),
                'fov_matrix': reference_fov_matrix,
                # This reference's OWN cross-modal z, so the preview can use
                # the target's drift RELATIVE to it rather than an absolute
                # value the pinned reference never receives.
                'cross_modal_z': self._cross_modal_z(
                    modality, self._shared_frame_modality() or cell.reference_modality, fov),
            })

        mask_anchor_fov_matrix = (self._fov_matrices_for_cell_modality(cell.reference_modality, cell, fov) or {}).get(
            (cell.reference_hybe, cell.reference_modality), np.eye(3))

        self.alignment_preview_window.show()
        self.alignment_preview_window.raise_()
        self.preview_canvas.draw_cell_alignment_preview_3col(
            cell, fov, reference_specs,
            target_storage_path, target_hybe, target_channel,
            fov_only_matrix, final_matrix, pad=pad, target_modality=target_modality,
            mask_anchor_fov_matrix=mask_anchor_fov_matrix,
            cross_modal_z=self._cross_modal_z(
                target_modality, self._shared_frame_modality() or cell.reference_modality, fov))

    def _show_cell_all_readouts_overlay(self, item=None):
        """
        Draws the one-vs-all sequential overlay for whichever cell was
        just clicked in "Results (per cell, overlay)", anchored at
        whatever's picked in Preview reference hybe (paired with Overlay
        FOV/this list -- see _refresh_cell_preview_reference_choices),
        falling back to cell.reference_hybe when nothing's selected --
        itemClicked supplies `item` directly; the currentItem() fallback
        covers a direct/programmatic call. Pure read of already-saved/
        staged data (tier 3 -- visualization only), never computes or
        writes.
        """
        ap = self.ui.AlignmentPanel
        ip = self.ui.IngestionPanel
        storage_path = ip.StoragePathLineEdit.text().strip()
        fov = ap.CellOverlayFovSpinBox.value()
        item = item or ap.CellOverlayCellListWidget.currentItem()
        if not storage_path or item is None or not self.hybe_records:
            QtWidgets.QMessageBox.warning(self, 'Show All-Readouts Overlay',
                                          'Set storage path (Ingestion tab) and select a cell first.')
            return
        cell_id = item.data(QtCore.Qt.UserRole)
        cell = None
        if self.cell_container_permanent is not None:
            cell = next((c for c in self.cell_container_permanent.get_cells(fov) if c.id == cell_id), None)
        if cell is None and self.cell_container is not None:
            cell = next((c for c in self.cell_container.get_cells(fov) if c.id == cell_id), None)
        if cell is None:
            return
        channel_type = ap.CellChannelTypeComboBox.currentText()
        pad = ap.CellPadSpinBox.value()

        reference_key = ap.CellPreviewReferenceHybeComboBox.currentData()
        reference_hybe, reference_modality = (reference_key if reference_key
                                              else (cell.reference_hybe, cell.reference_modality))
        reference_record, reference_storage_path, err = self._resolve_preview_hybe_context(
            reference_hybe, reference_modality, storage_path, self.hybe_records, fov)
        if err:
            QtWidgets.QMessageBox.warning(self, 'Show All-Readouts Overlay',
                                          f"Can't use {reference_hybe} ({reference_modality}) as reference: {err}")
            return
        reference_channel = alignment.pick_channel_by_type(reference_record, channel_type)
        # Plain FOV-only lookup for the FOV/cross-modal column's own crop
        # -- no cell involvement, no bridge step left to compute.
        reference_fov_matrices = self._fov_matrices_for_cell_modality(reference_modality, cell, fov)
        reference_matrix = (reference_fov_matrices or {}).get((reference_hybe, reference_modality), np.eye(3))
        # Computed INDEPENDENTLY for the final column -- see draw_cell_
        # all_readouts_overlay's own docstring for why reference_hybe's
        # final crop needs the SAME KIND of matrix (_matrix_to_shared,
        # folding in reference_hybe's own real residual) the target's own
        # final column uses, not the residual-blind FOV-only one reused.
        reference_final_matrix = self._matrix_to_shared(reference_hybe, reference_modality, cell, fov)
        if reference_final_matrix is None:
            reference_final_matrix = np.eye(3)
        target_specs = self._cell_overlay_target_specs(cell, storage_path, fov, self.hybe_records, channel_type)
        mask_anchor_fov_matrix = (self._fov_matrices_for_cell_modality(cell.reference_modality, cell, fov) or {}).get(
            (cell.reference_hybe, cell.reference_modality), np.eye(3))
        self.alignment_preview_window.show()
        self.alignment_preview_window.raise_()
        self.preview_canvas.draw_cell_all_readouts_overlay(
            cell, fov, reference_hybe, reference_storage_path, reference_channel,
            target_specs, pad=pad, reference_matrix=reference_matrix, reference_final_matrix=reference_final_matrix,
            mask_anchor_fov_matrix=mask_anchor_fov_matrix)

    def _run_cell_alignment_for_selected_cell(self):
        """
        Computes cell-based alignment for exactly the cell identified by
        (FOV, Cell ID) in the top (tier-1) section, using the live
        Reference hybe/Channel/Pad values from that same section -- lets
        a parameter tweak be checked against one real cell in well under
        a second, instead of paying Align All Cells in FOV's whole-FOV
        cost just to see whether it helped. Entirely self-contained in
        tier 1 -- doesn't read or write anything in tier 3
        (visualization), and tier 3's own state (which FOV/reference hybe
        is being BROWSED) is never touched by running this.

        Reuses CellAlignmentWorker.run() directly, called synchronously
        (not started as a background QThread -- one cell's real cost is
        ~0.2s, not worth a background thread) so the EXACT SAME tested
        computation path runs here as in the real batch run -- same
        same-modality + cross-modal calls, same reject bounds, same "no
        no-alignment" fallbacks -- zero duplicated logic that could drift
        out of sync with it.

        Stages into _pending_per_cell_alignment -- so nothing reaches
        vlinks.h5 or a PNG, and the real cell object is untouched, until
        PerCellAcceptPushButton is clicked; PerCellRejectPushButton
        discards the staged result outright. Align All Cells in FOV below
        has no staging of its own -- it always computes and saves directly.
        """
        ap = self.ui.AlignmentPanel
        fov = ap.CellFovSpinBox.value()
        cell_id = ap.CellIdSpinBox.value()
        real_cell = None
        if self.cell_container_permanent is not None:
            real_cell = next((c for c in self.cell_container_permanent.get_cells(fov) if c.id == cell_id), None)
        if real_cell is None and self.cell_container is not None:
            real_cell = next((c for c in self.cell_container.get_cells(fov) if c.id == cell_id), None)
        if real_cell is None:
            QtWidgets.QMessageBox.warning(self, 'Preview This Cell',
                                          f'No segmented cell with ID {cell_id} found in FOV{fov:02d}.')
            return

        # Modality read directly off real_cell -- no separate picker (see
        # ui/alignment_panel.py's own comment on why: this app only ever
        # holds one modality's cells resident in memory at a time, so the
        # found cell's own .modality is already unambiguous). Reference
        # hybe still comes from that modality's own configured combo,
        # never from IngestionPanel's current selection (see populate_
        # cell_reference_hybe_choices' own docstring for the crash that
        # used to cause).
        cell_modality = real_cell.reference_modality
        cell_reference_hybe = ap.current_cell_reference_hybe(cell_modality) or None
        storage_path = self._storage_path_for_modality(cell_modality)
        hybe_records = self._active_hybe_records_for_modality(cell_modality)
        if not storage_path or not hybe_records:
            QtWidgets.QMessageBox.warning(self, 'Preview This Cell',
                                          'Pick a reference hybe (Cell-Based Alignment) and parse that modality\'s layout first.')
            return
        # No FOV-alignment prerequisite -- per core principle, absence of
        # alignment at any layer is IDENTITY, never a blocker: every layer
        # must run standalone. (The old gate also read the pre-unification
        # (storage_path, fov) key shape, so it blocked unconditionally.)
        if not self._fov_matrices_for(storage_path, fov):
            self.statusBar().showMessage(
                'No FOV alignment for this FOV yet -- proceeding with identity.', 8000)

        channel_type = ap.CellChannelTypeComboBox.currentText()
        pad = ap.CellPadSpinBox.value()

        staged_cell = deepcopy(real_cell)

        worker = CellAlignmentWorker(
            [(fov, [staged_cell], self._cell_alignment_passes(cell_modality, storage_path, fov))],
            channel_type=channel_type, pad=pad)
        result_holder = {}
        worker.finished_ok.connect(lambda results: result_holder.__setitem__('results', results))
        worker.failed.connect(lambda message: result_holder.__setitem__('error', message))
        worker.run()  # synchronous -- one cell, no background thread needed
        if 'error' in result_holder:
            QtWidgets.QMessageBox.critical(self, 'Preview This Cell', result_holder['error'])
            return
        # staged_cell was mutated in place by compute_cell_alignment inside
        # worker.run() -- result_holder['results'] carries the same object,
        # not a copy, so staged_cell already reflects the computed result.

        self._pending_per_cell_alignment = (real_cell, staged_cell)
        self._pending_per_cell_alignment_fov = fov
        self._pending_per_cell_alignment_params = {'reference_hybe': cell_reference_hybe, 'modality': cell_modality,
                                                    'storage_path': storage_path, 'channel_type': channel_type, 'pad': pad}
        ap.PerCellAcceptPushButton.setEnabled(True)
        ap.PerCellRejectPushButton.setEnabled(True)

        # Per explicit request: the preview shown here is the ref-hybe-vs-
        # all-hybe overlay anchored at THIS RUN's own reference_hybe (the
        # alignment anchor picked above) -- not cell.reference_hybe (the
        # segmentation hybe), a different concept (see
        # _cell_overlay_target_specs' own docstring). draw_cell_all_
        # readouts_overlay's FOV/final columns default to assuming H=eye
        # between the reference hybe and the shared mask's own frame,
        # which is only equivalent to the real transform when they're the
        # same hybe -- reference_matrix (a plain fov_matrices lookup, the
        # same resolver target_specs itself uses) supplies the real one
        # whenever they differ. Tier 3's "Results (per cell, per hybe)"
        # list -- one row per (cell, hybe) -- gives a per-hybe 2-column
        # comparison on demand instead.
        overlay_reference_hybe = cell_reference_hybe or staged_cell.reference_hybe
        record_by_folder = {r['folder']: r for r in hybe_records}
        reference_record = record_by_folder.get(overlay_reference_hybe)
        if reference_record is not None:
            reference_channel = alignment.pick_channel_by_type(reference_record, channel_type)
            reference_fov_matrices = self._fov_matrices_for_cell_modality(cell_modality, staged_cell, fov)
            reference_matrix = (reference_fov_matrices or {}).get((overlay_reference_hybe, cell_modality), np.eye(3))
            # Computed INDEPENDENTLY for the final column -- see draw_cell_
            # all_readouts_overlay's own docstring for why.
            reference_final_matrix = self._matrix_to_shared(overlay_reference_hybe, cell_modality, staged_cell, fov)
            if reference_final_matrix is None:
                reference_final_matrix = np.eye(3)
            target_specs = self._cell_overlay_target_specs(staged_cell, storage_path, fov, hybe_records, channel_type)
            mask_anchor_fov_matrix = (self._fov_matrices_for_cell_modality(staged_cell.reference_modality, staged_cell, fov) or {}).get(
                (staged_cell.reference_hybe, staged_cell.reference_modality), np.eye(3))
            self.alignment_preview_window.show()
            self.alignment_preview_window.raise_()
            self.preview_canvas.draw_cell_all_readouts_overlay(
                staged_cell, fov, overlay_reference_hybe, storage_path, reference_channel,
                target_specs, pad=pad, reference_matrix=reference_matrix,
                reference_final_matrix=reference_final_matrix, mask_anchor_fov_matrix=mask_anchor_fov_matrix)
        self.statusBar().showMessage(f'Cell {cell_id} (FOV{fov:02d}) alignment previewed -- Accept to save.', 8000)

    def _accept_per_cell_alignment(self):
        """
        Applies the staged per-cell result to the real cell object,
        persists it (matrices to vlinks.h5, overlay PNG), and refreshes
        tier 3's lists for that FOV so the newly-saved cell shows up
        there immediately.
        """
        ap = self.ui.AlignmentPanel
        if not self._pending_per_cell_alignment:
            return
        real_cell, staged_cell = self._pending_per_cell_alignment
        fov = self._pending_per_cell_alignment_fov
        run_params = self._pending_per_cell_alignment_params or {}
        channel_type = run_params.get('channel_type') or ap.CellChannelTypeComboBox.currentText()
        pad = run_params.get('pad', ap.CellPadSpinBox.value())
        fp_cells = self._begin_cell_edit(fov)
        real_cell.matrices = staged_cell.matrices
        real_cell.matrix_anchors = staged_cell.matrix_anchors
        real_cell.matrix_provenance = staged_cell.matrix_provenance
        # real_cell lives in ONE tier; the other tier holds an independent
        # twin of the same cell. Propagate the alignment-owned fields so
        # the tiers can never disagree about this cell's matrices -- the
        # same sync the batch door does (per confirmed real wipe: any
        # later legitimate cell write from the stale tier would destroy
        # this accept's result on disk).
        for other in (self.cell_container_permanent, self.cell_container):
            twin = other.by_id(fov, real_cell.id) if other is not None else None
            if twin is not None and twin is not real_cell:
                twin.matrices = deepcopy(real_cell.matrices)
                twin.matrix_anchors = deepcopy(real_cell.matrix_anchors)
                twin.matrix_provenance = deepcopy(real_cell.matrix_provenance)

        # Re-derive the overlay from the SAME (reference_hybe, modality,
        # storage_path) the Preview step actually ran with (stashed in
        # run_params) -- NOT real_cell.reference_hybe (the segmentation
        # hybe) and NOT IngestionPanel's current selection, either of
        # which can legitimately differ from what this run used and used
        # to crash _read_mip with a hybe that was never ingested under
        # the wrong modality's storage_path. Falls back to the cell's own
        # segmentation hybe/modality only if run_params is somehow empty
        # (matches compute_cell_alignment's own reference_hybe=None
        # default), not as the primary source.
        overlay_modality = run_params.get('modality') or real_cell.reference_modality
        overlay_reference_hybe = run_params.get('reference_hybe') or real_cell.reference_hybe
        storage_path = run_params.get('storage_path') or self._storage_path_for_modality(overlay_modality)
        wrote = False
        self._commit_cell_edit(fov, fp_cells)
        if storage_path and fov is not None:
            storage_paths = self._all_vlinks_storage_paths()
            container = None
            if self.cell_container_permanent is not None and real_cell in self.cell_container_permanent.get_cells(fov):
                container = self.cell_container_permanent
            elif self.cell_container is not None and real_cell in self.cell_container.get_cells(fov):
                container = self.cell_container
            if storage_paths and container is not None:
                vlinks_store.mirror_write_cells(storage_paths, fov, container)
                wrote = True
            hybe_records = self._active_hybe_records_for_modality(overlay_modality) if overlay_modality else []
            reference_record = {r['folder']: r for r in hybe_records}.get(overlay_reference_hybe)
            if reference_record is not None:
                save_path = paths.figure_path(storage_path, 'cells', fov, f'cell{real_cell.id}_alignment_overlay.png')
                reference_channel = alignment.pick_channel_by_type(reference_record, channel_type)
                target_specs = self._cell_overlay_target_specs(real_cell, storage_path, fov, hybe_records, channel_type)
                mask_anchor_fov_matrix = (self._fov_matrices_for_cell_modality(real_cell.reference_modality, real_cell, fov) or {}).get(
                    (real_cell.reference_hybe, real_cell.reference_modality), np.eye(3))
                self.preview_canvas.draw_cell_all_readouts_overlay(
                    real_cell, fov, overlay_reference_hybe, storage_path, reference_channel,
                    target_specs, pad=pad, save_path=save_path, mask_anchor_fov_matrix=mask_anchor_fov_matrix)

        self._pending_per_cell_alignment = None
        self._pending_per_cell_alignment_fov = None
        self._pending_per_cell_alignment_params = None
        ap.PerCellAcceptPushButton.setEnabled(False)
        ap.PerCellRejectPushButton.setEnabled(False)
        if fov is not None:
            self._refresh_cell_fov_panels(fov)
        saved_msg = 'saved to vlinks.h5; ' if wrote else ''
        QtWidgets.QMessageBox.information(self, 'Cell alignment accepted',
                                          f'Cell {real_cell.id} (FOV{fov:02d}) alignment applied, {saved_msg}overlay image saved.')

        # Matrices changed -> every spot's shared-frame coordinate (and
        # possibly its owner) is stale. Assignment is cheap; recompute for
        # every loaded FOV and persist all slices, per explicit decision
        # that saving cells, matrices, or spots all re-run assignment.
        for _fov in list(self.fov_matrices.keys()):
            try:
                self._recast_persisted_spots(_fov)
            except Exception as e:
                self.statusBar().showMessage(f'spot reassignment after accept failed for FOV{_fov}: {e}')

    def _reject_per_cell_alignment(self):
        ap = self.ui.AlignmentPanel
        self._pending_per_cell_alignment = None
        self._pending_per_cell_alignment_fov = None
        self._pending_per_cell_alignment_params = None
        ap.PerCellAcceptPushButton.setEnabled(False)
        ap.PerCellRejectPushButton.setEnabled(False)
        self.statusBar().showMessage('Per-cell alignment preview discarded.', 5000)

    # -- config --

    def _load_config_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load configuration file', self.save_path, 'configuration file (*.xml)')
        if path:
            self._load_config(path)

    def _load_config(self, path):
        """
        Config is modality-nested (see preprocess.make_xml_file): a
        {name: fields} 'modalities' dict for anything genuinely per-
        modality (layout_path/dax_directory/storage_path, that modality's
        own within-experiment reference_hybe + same_modality_channel_type, and
        its own cross_modality_reference_hybe -- the hybe THAT modality
        uses as the cross-modal bridge), plus a flat 'global' dict for
        settings that aren't modality-specific (fov_list, cross_channel_
        type, cell_align_reference_hybe, cell_align_channel_type,
        cell_seg_fov). Deliberately does NOT restore cell_seg_reference_
        hybe/channel/method -- those describe whatever a real segmentation
        run actually did (truth lives in vlinks.h5), not something an
        external config should dictate.

        The cross-modal section (Rna/DnaStoragePathLineEdit +
        Rna/DnaReferenceHybeComboBox) gets populated directly from
        modalities['RNA']/['DNA'] -- no separate rna_storage_path/
        dna_storage_path keys needed, the modality entries ARE that
        information. The Ingestion tab + within-experiment section are
        single-context (one active modality at a time in this app's
        current UI), so they're populated from whichever modality appears
        FIRST in the file.
        """
        try:
            cfg = preprocess.load_xml_file(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Load Config', f'{type(e).__name__}: {e}')
            return
        ip, ap, cp = self.ui.IngestionPanel, self.ui.AlignmentPanel, self.ui.CellSegmentPanel
        glob = cfg.get('global', {})
        modalities = cfg.get('modalities', {})
        self._config_modalities = modalities  # kept around for reference even beyond what's live-populated below

        ip.FovListLineEdit.setText(','.join(str(f) for f in glob.get('fov_list', [])))
        self._refresh_fov_spinbox_bounds()
        # Seed current_celltype_list from the file's own default names
        # BEFORE _activate_modalities below (which triggers _parse_layout
        # -> _refresh_celltype_names_from_vlinks) -- that call only ever
        # ADDS names on top of whatever's already here, so config-provided
        # defaults must land first to end up merged with, not overwritten
        # by, whatever real classified celltypes vlinks.h5 turns out to have.
        celltype_names_text = glob.get('celltype_names', '').strip()
        if celltype_names_text:
            for name in celltype_names_text.split(','):
                name = name.strip()
                if name and name not in self.current_celltype_list:
                    self.current_celltype_list.append(name)
            self.ui.CelltypeDeterminationPanel.ensure_celltype_names(self.current_celltype_list)
        if glob.get('cross_modality_channel_type'):
            ap.ChannelTypeComboBox.setCurrentText(glob['cross_modality_channel_type'])
        if glob.get('cell_align_channel_type'):
            ap.CellChannelTypeComboBox.setCurrentText(glob['cell_align_channel_type'])
        if glob.get('cell_seg_fov'):
            cp.FovSpinBox.setValue(int(glob['cell_seg_fov']))

        rna_fields = modalities.get('RNA', {})
        ap.RnaStoragePathLineEdit.setText(rna_fields.get('storage_path', ''))
        ap.RnaReferenceHybeComboBox.setCurrentText(rna_fields.get('cross_modality_reference_hybe', ''))
        dna_fields = modalities.get('DNA', {})
        ap.DnaStoragePathLineEdit.setText(dna_fields.get('storage_path', ''))
        ap.DnaReferenceHybeComboBox.setCurrentText(dna_fields.get('cross_modality_reference_hybe', ''))

        # Modality Setup: rebuild the name-entry fields to match the file's
        # modality count/names, lock them (mirrors what clicking Activate
        # does), then hand every modality's full field set to
        # _activate_modalities -- this both establishes the live modality
        # list AND (via _switch_current_modality -> _parse_layout)
        # repopulates every hybe-choice combo box + applies the first
        # modality's own reference_hybe/same_modality_channel_type, in the right
        # order (selection after repopulation, not before).
        names = list(modalities.keys())
        if not names:
            names = [f'Modality {i + 1}' for i in range(int(glob.get('num_modalities', 2)))]
        ip.build_modality_name_fields(len(names))
        for combo, name in zip(ip.ModalityNameComboBoxes, names):
            combo.setCurrentText(name)
        ip.NumModalitiesLineEdit.setText(str(len(names)))
        ip.lock_modality_setup()
        self._activate_modalities(names, modality_fields=modalities)

        # cell_align_reference_hybe is now per-modality-suffixed (see
        # _save_config_dialog's own write side) -- one independent XML
        # key per configured modality, matching cell_align_references'
        # own {modality: hybe} shape. Only a fallback for when vlinks
        # (already applied by _activate_modalities -> _refresh_params_
        # from_vlinks, which runs before this line) had nothing real to
        # say -- vlinks-actual values must always win over this stale
        # config default, never get overwritten by it after the fact.
        for name in self.ui.IngestionPanel.modality_names:
            config_value = glob.get(f'cell_align_reference_hybe_{name}')
            if config_value and not ap.current_cell_reference_hybe(name):
                ap.select_cell_reference_hybe(name, config_value)

    def _save_config_dialog(self):
        """
        Builds the modality-nested config from whatever this session's UI
        currently has -- the Ingestion tab's active modality becomes one
        <modality> entry (with its full layout_path/dax_directory/
        storage_path), and the cross-modal section's OTHER storage path
        (if filled in) becomes a second <modality> entry (storage_path +
        cross_modality_reference_hybe only -- this app's current UI has
        no live layout_path/dax_directory for a modality that isn't the
        Ingestion tab's active one). reference_hybe/same_modality_
        channel_type are no longer per-modality state at all (Same-
        Modality Alignment's own reference-hybe combo isn't modality-
        switch-scoped any more), so they're never written here -- a
        config saved by an OLDER version of this app may still have them
        under an old <modality> entry, which _activate_modalities' own
        state.update filter silently drops on load. A future N-modality
        UI would just add more entries to the same 'modalities' dict; the
        file format itself already supports it.
        """
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save configuration file', self.save_path, 'configuration file (*.xml)')
        if not path:
            return
        ip, ap, cp = self.ui.IngestionPanel, self.ui.AlignmentPanel, self.ui.CellSegmentPanel
        if self.ui.IngestionPanel.current_modality is not None:
            self._save_current_modality_fields()  # flush whatever's live in the fields right now

        cross_hybe_fields = {'RNA': ap.RnaReferenceHybeComboBox, 'DNA': ap.DnaReferenceHybeComboBox}
        modalities = {}
        for name in self.ui.IngestionPanel.modality_names:
            state = self.ui.IngestionPanel.modality_data.get(name, self._blank_modality_state())
            if not state['layout_path'] and not state['storage_path']:
                continue  # never configured -- omit rather than writing an empty placeholder
            entry = dict(state)
            if name in cross_hybe_fields:
                entry['cross_modality_reference_hybe'] = cross_hybe_fields[name].currentText().strip()
            modalities[name] = entry

        cfg = {
            'global': {
                'num_modalities': len(self.ui.IngestionPanel.modality_names),
                'fov_list': self._parse_fov_list(ip.FovListLineEdit.text()),
                'cross_modality_channel_type': ap.ChannelTypeComboBox.currentText(),
                # per-modality-suffixed -- see cell_align_references' own
                # {modality: hybe} shape (CellAlignmentWorker no longer
                # attempts a cross-modal residual fit, so each modality
                # needs its own independent, explicit reference hybe --
                # see that class's own docstring).
                **{f'cell_align_reference_hybe_{name}': hybe
                  for name, hybe in ap.cell_align_references().items()},
                'cell_align_channel_type': ap.CellChannelTypeComboBox.currentText(),
                'cell_seg_fov': cp.FovSpinBox.value(),
                'celltype_names': self.ui.CelltypeDeterminationPanel.celltype_names(),
            },
            'modalities': modalities,
        }
        preprocess.make_xml_file(cfg, path)

    @staticmethod
    def _parse_fov_list(text):
        """
        Comma- and/or whitespace-separated FOV numbers and/or inclusive
        ranges, e.g. '1-10,15,20-25' or '1-10 15 20-25' or a mix ->
        [1..10, 15, 20..25]. Order is preserved and duplicates are dropped
        (a range overlapping an explicit number shouldn't double-queue it
        for batch operations like ingestion/alignment).
        """
        fovs = []
        seen = set()
        for chunk in re.split(r'[,\s]+', text.strip()):
            chunk = chunk.strip()
            if not chunk:
                continue
            if '-' in chunk:
                start_str, end_str = chunk.split('-', 1)
                for fov in range(int(start_str.strip()), int(end_str.strip()) + 1):
                    if fov not in seen:
                        seen.add(fov)
                        fovs.append(fov)
            else:
                fov = int(chunk)
                if fov not in seen:
                    seen.add(fov)
                    fovs.append(fov)
        return fovs
