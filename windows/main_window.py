import os
import re
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
from canvas.spot_crop_displayer import SpotCropDisplayer
from canvas.barcode_overview_displayer import BarcodeOverviewDisplayer
from canvas.celltype_result_displayer import CelltypeResultDisplayer
from canvas.mip_viewer import MipViewerDisplayer
from canvas.memory_viewer import MemoryViewerDisplayer
from canvas.alignment_preview_window import AlignmentPreviewWindow
from codelab_pipeline.io import preprocess
from codelab_pipeline.io import vlinks_store
from codelab_pipeline.alignment import chain as alignment
from codelab_pipeline.alignment import spot_mapper
from codelab_pipeline.segmentation import segment
from codelab_pipeline.localization import localization
from codelab_pipeline.models.cell_container import CellContainer
from codelab_pipeline.models.spot import ASpot
from codelab_pipeline.models import celltype
from skimage.feature import peak_local_max


class IngestionWorker(QtCore.QThread):
    """
    Runs convert_dax_to_h5_worker sequentially (not via ProcessPoolExecutor)
    -- deliberately: macOS's spawn-based multiprocessing needs the entry
    script properly __main__-guarded to submit work from a ProcessPoolExecutor,
    which doesn't hold reliably when called from inside a QThread in a GUI
    app. A QThread already keeps the ingestion off the GUI thread, which is
    what actually matters for responsiveness here; throughput can be
    revisited later if ingesting large hybe/FOV counts through the GUI
    turns out too slow in practice.
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

    def __init__(self, fov_list, hybe_records, dax_directory, storage_path, modality, overwrite=True):
        super().__init__()
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
            for i, (fov, record) in enumerate(tasks):
                fov_r, hybe_r, err = preprocess.convert_dax_to_h5_worker(
                    fov, record, self.dax_directory, self.storage_path, self.modality, overwrite=self.overwrite)
                if err is None:
                    # Ingestion is one of the two places this app is allowed
                    # to touch the raw per-hybe stack file directly (see
                    # vlinks_store.write_hybe_mip's own docstring) -- the
                    # file's already just been written, so this MIP copy is
                    # cheap. Everything downstream (ingestion-status checks,
                    # displayers, "has this FOV been aligned") should read
                    # vlinks.h5 from here on, never re-open this stack file
                    # just to check whether it's usable.
                    try:
                        h5path = os.path.join(self.storage_path, f'FOV{fov_r:02d}', f'{hybe_r}_stack.h5')
                        with h5py.File(h5path, 'r') as f:
                            channel_mips = {ch: f[f'/mip/ch{ch}'][:] for ch in record['channels']}
                        vlinks_store.write_hybe_mip(self.storage_path, fov_r, hybe_r, channel_mips,
                                                    fiducial_channel=record['fiducial_channel'])
                    except Exception as e:
                        err = f'ingested but failed to write vlinks.h5 MIP: {e}'
                status = 'OK' if err is None else f'ERROR: {err}'
                if err is not None:
                    error_lines.append(f'FOV{fov_r:02d} {hybe_r}: {err}')
                self.progress.emit(i + 1, len(tasks), f'FOV{fov_r:02d} {hybe_r}: {status}')
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

    def __init__(self, storage_path, fov, reference_hybe, channel, diameter, min_size, max_size):
        super().__init__()
        self.storage_path = storage_path
        self.fov = fov
        self.reference_hybe = reference_hybe
        self.channel = channel
        self.diameter = diameter
        self.min_size = min_size
        self.max_size = max_size

    def run(self):
        try:
            mask, reference_image = segment.segment_fov(
                self.storage_path, self.fov, self.reference_hybe, self.channel,
                diameter=self.diameter, min_size=self.min_size, max_size=self.max_size)
            self.finished_ok.emit(mask, reference_image)
        except Exception as e:
            self.failed.emit(str(e))


class ClassicalSegmentWorker(QtCore.QThread):
    """Mirrors CellSegmentWorker's exact signal shape so _on_cell_segment_finished/_failed are reused unchanged."""
    finished_ok = QtCore.pyqtSignal(object, object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, storage_path, fov, reference_hybe, channel, method, absolute_cutoff, min_distance, min_size, max_size):
        super().__init__()
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
            mask, reference_image = segment.segment_fov_classical(
                self.storage_path, self.fov, self.reference_hybe, self.channel,
                method=self.method, absolute_cutoff=self.absolute_cutoff, min_distance=self.min_distance,
                min_size=self.min_size, max_size=self.max_size)
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
    jobs: list of (fov, cells, fov_matrices_for_that_fov, other_ctx).
    cells are the real ACell objects (automatic mode --
    compute_cell_alignment mutates cell.matrices in place, so this IS the
    commit) or deepcopies of them (manual mode -- staged, only merged
    into the real cells on Accept). Manual mode passes one job (today's
    single-FOV behavior); automatic mode passes one job per FOV that has
    permanent segmented cells.

    other_ctx: (other_storage_path, other_hybe_records, other_fov_matrices,
    other_reference_hybe, other_modality) or None -- see
    MainWindow._other_modality_cell_alignment_inputs. When present, every
    cell also gets a SECOND, independent compute_cell_alignment call
    against the OTHER modality's own hybes (e.g. DNA hybes for an RNA-
    segmented cell), composed into the cell's own frame via the cross-
    modal correction -- per explicit request: once both the same-
    modality and cross-modality layers are established, cell-based
    alignment shouldn't be limited to a single modality's hybes.
    """
    progress = QtCore.pyqtSignal(int, int, str)
    finished_ok = QtCore.pyqtSignal(list)  # [(fov, cells), ...]
    failed = QtCore.pyqtSignal(str)

    def __init__(self, jobs, storage_path, hybe_records, modality, reference_hybe=None, channel_type='readout', pad=10):
        super().__init__()
        self.jobs = jobs
        self.storage_path = storage_path
        self.hybe_records = hybe_records
        self.modality = modality
        self.reference_hybe = reference_hybe
        self.channel_type = channel_type
        self.pad = pad

    def run(self):
        try:
            results = []
            total = sum(len(cells) for _, cells, _, _ in self.jobs)
            done = 0
            for fov, cells, fov_matrices, other_ctx in self.jobs:
                # only the hybes actually present in this FOV's fov_matrices
                # are valid -- self.hybe_records can hold more (e.g. every
                # hybe in the parsed layout) than what FOV alignment was
                # actually run/accepted for.
                hybe_records = [r for r in self.hybe_records if r['folder'] in fov_matrices]
                if other_ctx is not None:
                    other_storage_path, other_records_full, other_fov_matrices, other_reference_hybe, other_modality = other_ctx
                    other_hybe_records = [r for r in other_records_full if r['folder'] in other_fov_matrices]
                    if other_reference_hybe not in other_fov_matrices:
                        # other_reference_hybe (that modality's own within-
                        # experiment reference hybe, as configured on its
                        # own Alignment tab) isn't itself in the ingested/
                        # aligned set -- compute_cell_alignment's own
                        # reference_hybe lookup (record_by_folder[reference_
                        # _hybe]) would raise a bare KeyError for every cell
                        # in this FOV, aborting the WHOLE batch run rather
                        # than just skipping this optional extra layer.
                        # Same "best-effort, not fatal to the already-done
                        # same-modality result" treatment as the ValueError
                        # catch below -- just caught here, before it can
                        # happen, instead of after.
                        other_hybe_records = None
                else:
                    other_storage_path = other_hybe_records = other_fov_matrices = other_reference_hybe = other_modality = None
                for cell in cells:
                    alignment.compute_cell_alignment(cell, self.storage_path, fov, hybe_records, fov_matrices,
                                                     reference_hybe=self.reference_hybe, channel_type=self.channel_type,
                                                     pad=self.pad, modality=self.modality)
                    if other_ctx is not None and other_hybe_records:
                        try:
                            # cell.reference_hybe is a same-modality-only
                            # hybe name -- it's never a real key in
                            # other_fov_matrices (the OTHER modality's own
                            # hybes), so compute_cell_alignment's default
                            # lookup would silently treat it as identity
                            # there. Resolve it from fov_matrices (the
                            # SAME modality this cell/its reference hybe
                            # actually belongs to, already in scope here)
                            # and pass it through explicitly instead.
                            alignment.compute_cell_alignment(cell, other_storage_path, fov, other_hybe_records,
                                                             other_fov_matrices, reference_hybe=other_reference_hybe,
                                                             channel_type=self.channel_type, pad=self.pad,
                                                             cell_reference_hybe_matrix=fov_matrices.get(cell.reference_hybe, np.eye(3)),
                                                             modality=other_modality)
                        except ValueError:
                            # cell doesn't overlap the other modality's own
                            # reference-hybe frame at all -- a best-effort
                            # extra layer, not fatal to the (already-done)
                            # same-modality result.
                            pass
                    done += 1
                    self.progress.emit(done, total, f'FOV{fov:02d} cell {cell.id}: aligned')
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
                 rna_reference_hybe, dna_reference_hybe, channel_type, write=True, border_trim=0, max_shift=None):
        super().__init__()
        self.rna_storage_path = rna_storage_path
        self.dna_storage_path = dna_storage_path
        self.fov_list = fov_list
        self.all_fov_matrices = all_fov_matrices  # {(storage_path, fov): {hybe: H}}
        self.rna_reference_hybe = rna_reference_hybe
        self.dna_reference_hybe = dna_reference_hybe
        self.channel_type = channel_type
        self.write = write
        self.border_trim = border_trim
        self.max_shift = max_shift

    def run(self):
        try:
            results = {}
            for i, fov in enumerate(self.fov_list):
                rna_fov_matrices = self.all_fov_matrices.get((self.rna_storage_path, fov), {})
                dna_fov_matrices = self.all_fov_matrices.get((self.dna_storage_path, fov), {})
                H = alignment.link_cross_modal(self.rna_storage_path, self.dna_storage_path, fov,
                                               rna_fov_matrices, dna_fov_matrices,
                                               self.rna_reference_hybe, self.dna_reference_hybe, self.channel_type,
                                               border_trim=self.border_trim, max_shift=self.max_shift)
                if self.write:
                    alignment.write_cross_modal_matrix(self.dna_storage_path, fov, H,
                                                        self.rna_reference_hybe, self.dna_reference_hybe, self.channel_type)
                results[fov] = H
                self.progress.emit(i + 1, len(self.fov_list), f'FOV{fov:02d}: cross-modal computed')
            self.finished_ok.emit(results)
        except Exception as e:
            self.failed.emit(str(e))


def _matrix_summary(hybe, H):
    angle = np.degrees(np.arctan2(H[1, 0], H[0, 0]))
    return f'{hybe}: dx={H[0,2]:.2f}, dy={H[1,2]:.2f}, angle={angle:.3f} deg'


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config_file=None):
        super().__init__()
        self.ui = MainWindowUI()
        self.ui.setupUi(self)

        self.hybe_records = []
        self.fov_matrices = {}
        # (storage_path, fov) -> {(hybe, channel): [ASpot, ...]} -- Whole
        # FOV spot auto-detect peaks that don't land inside any cell's
        # mask. Not stored on any ACell (there's nothing to own them),
        # kept per-FOV instead, same shape as fov_matrices itself; each
        # ASpot here keeps ASpot.cell at its model default (-1, "no
        # link") since a proper cell link genuinely doesn't apply.
        self.fov_unassigned_spots = {}
        # Shared celltype identity list (see ui/celltype_determination_
        # panel.py's own docstring) -- default empty, seeded from a loaded
        # config's celltype_names and/or any real classified celltype
        # already found in vlinks.h5 (see _refresh_celltype_names_from_
        # vlinks), so the Celltype Determination tab's own listview is
        # already usable without the user re-typing every name back in.
        # Names only ever get ADDED here automatically, never removed --
        # Remove Selected in the panel stays a manual, explicit action.
        self.current_celltype_list = []
        self.modality_names = ['DNA', 'RNA']
        self.modality_data = {name: self._blank_modality_state() for name in self.modality_names}
        self.total_active_hybe_list = []  # [(hybe_record, modality_name), ...] -- see _refresh_active_hybe_lists
        self.current_modality = None
        self._job_queue = []
        self._job_queue_index = 0
        self._job_queue_overwrite = True
        self.save_path = repo_path

        self.cell_container = None
        self.cell_container_permanent = None
        self._last_segment_context = None  # {'fov': .., 'reference_hybe': ..}
        self.cell_displayer = CellDisplayer()
        self.cell_displayer.mask_edited.connect(self._on_displayer_mask_edited)

        self.spot_crop_displayer = SpotCropDisplayer()
        self.spot_crop_displayer.spots_edited.connect(self._on_spot_crop_edited)
        self._spot_crop_context = None  # {'cell': ACell, 'hybe': str, 'channel': int, 'rxmin': int, 'rymin': int}

        self.mip_viewer = MipViewerDisplayer()
        self.memory_viewer = MemoryViewerDisplayer()
        self.memory_viewer.RefreshPushButton.clicked.connect(self._refresh_memory_viewer)

        self.barcode_overview_displayer = BarcodeOverviewDisplayer()
        self.celltype_result_displayer = CelltypeResultDisplayer()
        self._fov_ranges_by_celltype = {}       # {celltype(str): range_string}
        self._barcode_channel_by_celltype = {}  # {celltype(str): (hybe,channel)}
        self._barcode_calibration = {'scale': {}, 'lower_bound': {}, 'upper_bound': {}}  # each {(hybe,channel): {fov(int): float}}

        # single shared pop-up + canvas for every alignment preview (FOV,
        # cross-modal, cell) -- nothing is embedded in the docked panel, see
        # ui/alignment_panel.py's docstring for why.
        self.alignment_preview_window = AlignmentPreviewWindow()
        self.preview_canvas = PipelineCanvas(self.alignment_preview_window.canvas)
        self.cross_modal_result = {}  # {(dna_storage_path, fov): H}, committed
        self._same_modality_context = None
        self._pending_same_modality_alignment = None  # {fov: {hybe: H}} awaiting Accept/Reject
        self._cross_modal_context = None
        self._pending_cross_modal = None  # {fov: H} awaiting Accept/Reject
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
        cp.ModalityComboBox.currentTextChanged.connect(self._switch_current_modality)
        ap.ModalityComboBox.currentTextChanged.connect(self._switch_current_modality)
        sp.ModalityComboBox.currentTextChanged.connect(self._switch_current_modality)
        ctp.ModalityComboBox.currentTextChanged.connect(self._switch_current_modality)

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
        ip.ShowMemoryViewerPushButton.clicked.connect(self._show_memory_viewer)
        ip.AddJobPushButton.clicked.connect(self._add_job_to_queue)
        ip.RemoveJobPushButton.clicked.connect(self._remove_selected_jobs)
        ip.RunQueuePushButton.clicked.connect(self._run_job_queue)
        ip.JobQueueListWidget.itemClicked.connect(self._load_job_into_form)

        cp.RunSegmentationPushButton.clicked.connect(self._run_cell_segmentation)
        cp.ShowDisplayerPushButton.toggled.connect(self._toggle_cell_displayer)
        self.cell_displayer.UpdatePushButton.clicked.connect(self._ensure_cell_displayer_initialized)
        cp.SaveCellsPushButton.clicked.connect(self._save_cells)
        cp.DiscardCellsPushButton.clicked.connect(self._discard_cells)
        cp.SendPermanentPushButton.clicked.connect(self._send_permanent_cells_to_transient)
        cp.FovSpinBox.valueChanged.connect(self._activate_fov)

        ap.RunFovAlignmentPushButton.clicked.connect(self._run_fov_alignment)
        ap.RunAllFovAlignmentPushButton.clicked.connect(self._run_fov_alignment_all)
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
        ap.CellReferenceHybeComboBox.currentIndexChanged.connect(lambda _: self._refresh_cell_per_hybe_results_from_spinboxes())
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
        sp.HybeComboBox.currentIndexChanged.connect(self._show_spot_displayer)
        sp.ChannelComboBox.currentIndexChanged.connect(self._show_spot_displayer)
        sp.AutoDetectPushButton.clicked.connect(self._run_spot_auto_detect)
        sp.ShowDisplayerPushButton.toggled.connect(self._toggle_spot_crop_displayer)
        sp.RemoveTransientSpotsPushButton.clicked.connect(self._remove_transient_spots)
        sp.RemoveSpotsInViewPushButton.clicked.connect(self._remove_all_spots_in_view)
        sp.RemoveAllSpotsPushButton.clicked.connect(self._remove_all_spots_in_fov)
        sp.SaveViewPushButton.clicked.connect(self._save_view)
        sp.ThresholdPercentLineEdit.editingFinished.connect(self._sync_threshold_from_percent)
        sp.ThresholdAbsoluteLineEdit.editingFinished.connect(self._sync_threshold_from_absolute)

        ctp.SetFovRangesPushButton.clicked.connect(self._set_celltype_fov_ranges)
        ctp.AssignBarcodeChannelPushButton.clicked.connect(self._assign_barcode_channel)
        ctp.ApplyCalibrationPushButton.clicked.connect(self._apply_barcode_calibration)
        ctp.ShowBarcodeOverviewPushButton.clicked.connect(self._show_barcode_overview)
        ctp.RunCelltypeDeterminationPushButton.clicked.connect(self._run_celltype_determination)
        ctp.ShowCelltypeResultPushButton.clicked.connect(lambda: self._show_celltype_result())

        self.ui.actionLoad_Config.triggered.connect(self._load_config_dialog)
        self.ui.actionSave_Config.triggered.connect(self._save_config_dialog)

    # -- modality setup / switching --

    @staticmethod
    def _blank_modality_state():
        return {'layout_path': '', 'dax_directory': '', 'storage_path': '', 'reference_hybe': '', 'same_modality_channel_type': '',
                'active_hybe_list': []}

    def _all_modality_combo_boxes(self):
        boxes = [self.ui.IngestionPanel.ModalityComboBox]
        for attr in ('CellSegmentPanel', 'AlignmentPanel', 'SpotLocalizationPanel', 'CelltypeDeterminationPanel'):
            boxes.append(getattr(self.ui, attr).ModalityComboBox)
        return boxes

    def _sync_modality_combo_items(self, names):
        for combo in self._all_modality_combo_boxes():
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            combo.blockSignals(False)

    def _sync_modality_combo_text(self, name):
        for combo in self._all_modality_combo_boxes():
            if combo.currentText() != name:
                combo.blockSignals(True)
                combo.setCurrentText(name)
                combo.blockSignals(False)

    def _save_current_modality_fields(self):
        ip, ap = self.ui.IngestionPanel, self.ui.AlignmentPanel
        # update() in place, not a wholesale dict replacement -- this
        # modality's own active_hybe_list (and anything else computed
        # separately, e.g. by _refresh_active_hybe_lists) must survive a
        # switch-away/switch-back, not get silently wiped back to the
        # blank-state default just because the user changed tabs.
        data = self.modality_data.setdefault(self.current_modality, self._blank_modality_state())
        data.update({
            'layout_path': ip.LayoutPathLineEdit.text().strip(),
            'dax_directory': ip.DaxDirectoryLineEdit.text().strip(),
            'storage_path': ip.StoragePathLineEdit.text().strip(),
            'reference_hybe': ap.ReferenceHybeComboBox.currentText(),
            'same_modality_channel_type': ap.SameModalityChannelTypeComboBox.currentText(),
        })

    def _switch_current_modality(self, name):
        """
        The single "current modality" driving the Ingestion tab's per-
        modality fields (layout/dax/storage path, hybes-to-ingest,
        ingestion status) plus every downstream panel's own mirrored
        ModalityComboBox (Cell Segmentation, within-experiment Alignment,
        Spot Localization, Celltype Determination) -- changing any one of
        those combos calls this and every other combo follows, since
        there's exactly one active modality app-wide at a time. fov_list
        and the other global config fields are untouched by this switch.
        """
        if not name:
            return
        ip, ap = self.ui.IngestionPanel, self.ui.AlignmentPanel
        if self.current_modality is not None and self.current_modality != name:
            self._save_current_modality_fields()
        self.current_modality = name
        data = self.modality_data.setdefault(name, self._blank_modality_state())
        ip.LayoutPathLineEdit.setText(data['layout_path'])
        ip.DaxDirectoryLineEdit.setText(data['dax_directory'])
        ip.StoragePathLineEdit.setText(data['storage_path'])
        if data['layout_path']:
            self._parse_layout()
        else:
            self.hybe_records = []
            ip.HybeListWidget.clear()
            ip.populate_viewer_hybe_choices([])
            ap.populate_reference_hybe_choices([])
            ap.populate_cell_reference_hybe_choices(self._all_modality_hybe_records())
            self.ui.CellSegmentPanel.populate_reference_hybe_choices([])
            self.ui.SpotLocalizationPanel.populate_hybe_choices([])
            self.ui.CelltypeDeterminationPanel.populate_hybe_choices([])
            ip.IngestionStatusTextEdit.clear()
        if data['reference_hybe']:
            idx = ap.ReferenceHybeComboBox.findText(data['reference_hybe'])
            if idx >= 0:
                ap.ReferenceHybeComboBox.setCurrentIndex(idx)
        if data['same_modality_channel_type']:
            ap.SameModalityChannelTypeComboBox.setCurrentText(data['same_modality_channel_type'])
        if data['storage_path'] and data['storage_path'] not in self._vlinks_refreshed_paths:
            # vlinks-actual values (whatever was really computed and
            # accepted) always win over a stale config default -- runs
            # LAST, after layout parsing/combo population above, so every
            # choice-dependent combo it touches (reference hybes, cell-
            # alignment anchor) already has real items to match against.
            # May itself trigger a first-time _parse_layout() if layout_path
            # was blank above and only vlinks has it.
            #
            # Only ever done ONCE per storage_path per session (tracked via
            # _vlinks_refreshed_paths): this reconciles a stale config
            # default on first load, but must never re-fire on a later
            # switch-back within the same session, or it would clobber a
            # live in-session combo edit the user hasn't run/accepted yet
            # with whatever vlinks still has on disk from a prior run.
            self._refresh_params_from_vlinks(data['storage_path'])
            self._vlinks_refreshed_paths.add(data['storage_path'])
        # active_hybe_list/total_active_hybe_list refresh -- unlike the
        # params backfill above, this is never gated by
        # _vlinks_refreshed_paths (see _refresh_active_hybe_lists' own
        # docstring for why): every modality switch stands in for "the
        # vlink was just parsed" for that modality, per explicit design.
        self._refresh_active_hybe_lists()
        self._sync_modality_combo_text(name)
        self._refresh_same_modality_results_list()

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

        For cross-modal params specifically, also reconstructs
        self.modality_data for the OTHER (paired) modality via
        cross_modal_paired_storage_path -- so
        _other_modality_cell_alignment_inputs's other_data lookup and
        H_across itself are both available even when that OTHER
        modality's own config was never loaded this session at all.
        """
        ip, ap = self.ui.IngestionPanel, self.ui.AlignmentPanel
        params = vlinks_store.read_global_params(storage_path)
        if not params:
            return
        data = self.modality_data.get(self.current_modality)

        layout_path = params.get('layout_path')
        if layout_path and data is not None and not data['layout_path']:
            data['layout_path'] = layout_path
            ip.LayoutPathLineEdit.setText(layout_path)
            self._parse_layout()

        reference_hybe = params.get('same_modality_reference_hybe')
        if reference_hybe:
            if data is not None:
                data['reference_hybe'] = reference_hybe
            idx = ap.ReferenceHybeComboBox.findText(reference_hybe)
            if idx >= 0:
                ap.ReferenceHybeComboBox.setCurrentIndex(idx)
        channel_type = params.get('same_modality_channel_type')
        if channel_type:
            if data is not None:
                data['same_modality_channel_type'] = channel_type
            ap.SameModalityChannelTypeComboBox.setCurrentText(channel_type)

        cell_reference_hybe = params.get('cell_alignment_reference_hybe')
        if cell_reference_hybe:
            ap.CellReferenceHybeComboBox.setCurrentText(cell_reference_hybe)
        cell_channel_type = params.get('cell_alignment_channel_type')
        if cell_channel_type:
            ap.CellChannelTypeComboBox.setCurrentText(cell_channel_type)
        cell_pad = params.get('cell_alignment_pad')
        if cell_pad is not None:
            ap.CellPadSpinBox.setValue(int(cell_pad))

        role = params.get('cross_modal_role')
        paired_path = params.get('cross_modal_paired_storage_path')
        if role and paired_path:
            rna_path, dna_path = (storage_path, paired_path) if role == 'RNA' else (paired_path, storage_path)
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

            other_name = 'DNA' if role == 'RNA' else 'RNA'
            other_data = self.modality_data.setdefault(other_name, self._blank_modality_state())
            other_data['storage_path'] = paired_path
            other_params = vlinks_store.read_global_params(paired_path)
            if not other_data['layout_path'] and other_params.get('layout_path'):
                other_data['layout_path'] = other_params['layout_path']
            if not other_data['reference_hybe'] and other_params.get('same_modality_reference_hybe'):
                other_data['reference_hybe'] = other_params['same_modality_reference_hybe']
            if not other_data['same_modality_channel_type'] and other_params.get('same_modality_channel_type'):
                other_data['same_modality_channel_type'] = other_params['same_modality_channel_type']
            if other_name not in self.modality_names:
                self.modality_names = list(self.modality_names) + [other_name]

            # _other_modality_cell_alignment_inputs also needs
            # self.fov_matrices[(paired_path, fov)] -- normally only
            # populated by _activate_fov for whichever modality is
            # "current" (never the paired one, since that requires an
            # actual modality switch) -- read it directly here, same
            # read_same_modality_matrices call _activate_fov itself uses,
            # for the CellSegmentPanel's current FOV plus anything in the
            # Ingestion tab's own FOV list.
            if other_data['layout_path']:
                try:
                    other_hybe_records_for_matrices = preprocess.parse_experiment_layout(other_data['layout_path'])
                except Exception:
                    other_hybe_records_for_matrices = None
                if other_hybe_records_for_matrices:
                    fovs_to_populate = set(self._parse_fov_list(ip.FovListLineEdit.text()))
                    fovs_to_populate.add(self.ui.CellSegmentPanel.FovSpinBox.value())
                    for fov_to_populate in fovs_to_populate:
                        key = (paired_path, fov_to_populate)
                        if key not in self.fov_matrices:
                            try:
                                self.fov_matrices[key] = alignment.read_same_modality_matrices(
                                    paired_path, fov_to_populate, other_hybe_records_for_matrices)
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
        fields: optional {name: {layout_path/dax_directory/storage_path/
        reference_hybe/same_modality_channel_type}} to pre-seed from (e.g. a
        loaded config's 'modalities' dict) -- omitted fields default blank.
        """
        modality_fields = modality_fields or {}
        self.modality_names = names
        self.modality_data = {}
        for name in names:
            state = self._blank_modality_state()
            state.update({k: v for k, v in modality_fields.get(name, {}).items() if k in state})
            self.modality_data[name] = state
        self.current_modality = None
        self._sync_modality_combo_items(names)
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
        data = self.modality_data.get(name)
        if not data or not data.get('storage_path') or not fov_list:
            return []
        if name == self.current_modality and self.hybe_records:
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

    def _all_modality_hybe_records(self):
        """
        Every configured modality's own active_hybe_list, deduped by
        folder name -- for combos that need choices spanning BOTH
        modalities regardless of which one is "current" (e.g.
        CellReferenceHybeComboBox, now that cell-based alignment
        processes both). See _active_hybe_records_for_modality for the
        actual ingestion-filtering.
        """
        merged = []
        seen = set()
        for name in self.modality_names:
            for r in self._active_hybe_records_for_modality(name):
                if r['folder'] not in seen:
                    seen.add(r['folder'])
                    merged.append(r)
        return merged

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
        for name in self.modality_names:
            self.modality_data.setdefault(name, self._blank_modality_state())['active_hybe_list'] = \
                self._active_hybe_records_for_modality(name)

        total = []
        seen = set()
        for name in self.modality_names:
            for r in self.modality_data[name]['active_hybe_list']:
                key = (r['folder'], name)
                if key not in seen:
                    seen.add(key)
                    total.append((r, name))
        self.total_active_hybe_list = total  # [(hybe_record, modality_name), ...]

        ap = self.ui.AlignmentPanel
        current_active = self.modality_data.get(self.current_modality, {}).get('active_hybe_list', [])
        ap.populate_reference_hybe_choices(current_active)
        self.ui.CellSegmentPanel.populate_reference_hybe_choices(current_active)
        self.ui.SpotLocalizationPanel.populate_hybe_choices(current_active)
        self.ui.CelltypeDeterminationPanel.populate_hybe_choices(current_active)
        ap.populate_cell_reference_hybe_choices([r for r, _ in self.total_active_hybe_list])
        for name, populate in (('RNA', ap.populate_rna_reference_hybe_choices),
                               ('DNA', ap.populate_dna_reference_hybe_choices)):
            populate(self.modality_data.get(name, {}).get('active_hybe_list', []))

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
            # a real, confirmed fact about this storage path (which
            # ExperimentLayout it uses) -- lets a completely fresh session
            # reconstruct hybe_records/combo choices from vlinks.h5 alone
            # (see _refresh_params_from_vlinks), without ever loading a
            # config file.
            vlinks_store.write_global_params(storage_path, layout_path=layout_path)
        ip.populate_hybe_list(self.hybe_records, dax_directory=ip.DaxDirectoryLineEdit.text().strip())
        ip.populate_viewer_hybe_choices(self.hybe_records)
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
        for storage_path in storage_paths:
            fov_ranges, barcode_channels, calibration, barcode_method = vlinks_store.read_celltype_config(storage_path)
            for name, range_string in fov_ranges.items():
                if name not in self._fov_ranges_by_celltype:
                    self._fov_ranges_by_celltype[name] = range_string
            for name, bch in barcode_channels.items():
                if name not in self._barcode_channel_by_celltype:
                    self._barcode_channel_by_celltype[name] = tuple(bch)
            for key in ('scale', 'lower_bound', 'upper_bound'):
                for bch, per_fov in calibration.get(key, {}).items():
                    bch = tuple(bch)
                    dest = self._barcode_calibration[key].setdefault(bch, {})
                    for fov, val in per_fov.items():
                        dest.setdefault(int(fov), val)
            if barcode_method:
                ctp.BarcodeMethodComboBox.setCurrentText(
                    'Median' if barcode_method == 'median' else 'Vote (200-sample)')
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
        for storage_path in storage_paths:
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
            return
        lo, hi = min(fov_list), max(fov_list)
        for sb in spinboxes:
            sb.setRange(lo, hi)

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
        storage_path = ip.StoragePathLineEdit.text().strip()
        reference_hybe = ap.ReferenceHybeComboBox.currentText()
        fov_list = self._parse_fov_list(ip.FovListLineEdit.text())
        ap.SameModalityResultsListWidget.clear()
        if not storage_path or not reference_hybe or not fov_list or not self.hybe_records:
            return
        disk_results = {}
        for fov in fov_list:
            ready, _, _ = self._ingested_hybes_for_fov(storage_path, fov, self.hybe_records)
            if not ready:
                continue
            ready_records = [r for r in self.hybe_records if r['folder'] in ready]
            matrices = alignment.read_same_modality_matrices(storage_path, fov, ready_records)
            if matrices:
                disk_results[fov] = matrices
        if disk_results:
            self.fov_matrices.update({(storage_path, fov): m for fov, m in disk_results.items()})

        display_results = dict(disk_results)
        for (sp, fov), matrices in self.fov_matrices.items():
            if sp == storage_path and matrices:
                display_results[fov] = matrices
        pending_fovs = set()
        if self._pending_same_modality_alignment:
            for fov, matrices in self._pending_same_modality_alignment.items():
                display_results[fov] = matrices
                pending_fovs.add(fov)
        if not display_results:
            return
        self._same_modality_context = {'storage_path': storage_path, 'hybe_records': self.hybe_records, 'reference_hybe': reference_hybe}
        for fov in sorted(display_results.keys()):
            matrices = display_results[fov]
            suffix = ' [pending]' if fov in pending_fovs else ''
            for hybe, H in matrices.items():
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
                                                  overwrite=(overwrite_mode == 'overwrite'))
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

    def _show_memory_viewer(self):
        self._refresh_memory_viewer()
        self.memory_viewer.show()
        self.memory_viewer.raise_()

    def _hybe_records_for_storage_path(self, storage_path):
        """
        Each Memory Status row needs the hybe list belonging to ITS OWN
        modality, not whichever modality happens to be currently active
        (self.hybe_records) -- reusing the active modality's list for
        every row silently corrupts a different modality's row the
        moment its hybe folder names differ from the active one's (the
        normal case, e.g. Hyb_101.. vs Hyb_002..), which is exactly what
        made a DNA row's FOV-align count collapse after switching to RNA
        and refreshing.
        """
        ip = self.ui.IngestionPanel
        if storage_path == ip.StoragePathLineEdit.text().strip():
            return self.hybe_records
        for data in self.modality_data.values():
            if data['storage_path'] == storage_path and data['layout_path']:
                try:
                    return preprocess.parse_experiment_layout(data['layout_path'])
                except Exception:
                    return []
        return []

    def _reference_hybe_for_storage_path(self, storage_path):
        ip, ap = self.ui.IngestionPanel, self.ui.AlignmentPanel
        if storage_path == ip.StoragePathLineEdit.text().strip():
            return ap.ReferenceHybeComboBox.currentText()
        for data in self.modality_data.values():
            if data['storage_path'] == storage_path:
                return data['reference_hybe']
        return ''

    def _refresh_memory_viewer(self):
        """
        Builds the Cell/Spot Memory Status table: for every storage path
        this session knows about (RNA and DNA both, if the Alignment tab's
        Cross-Modal fields are set -- see _all_vlinks_storage_paths) and
        every FOV in the Ingestion tab's FOV list, what's actually
        persisted in that experiment's vlinks.h5 right now, i.e. exactly
        what _activate_fov would load with no further computation.

        Matrix status is 4 separate layers, each with its own intuitive
        denominator (per explicit request -- the old single "computed/
        total" column mixed FOV-level per-hybe counts into one unexplained
        number, e.g. "4/16"):
        - FOV: (hybes with a real, computed matrix) / (hybes actually
          INGESTED for this FOV -- not len(self.hybe_records), which can
          include hybes from the parsed layout that were never ingested
          for this specific FOV).
        - Cross-modal: 0/1 or 1/1 (n/a if this storage path isn't paired
          with another modality via the Alignment tab's fields).
        - Cell: cells with cell-based alignment computed / total cells.
        - Spot: spots with linked=True (celltype determination has run) /
          total spots.

        Two more columns track celltype ASSIGNMENT specifically (linked=
        True, same flag, but distinct from "Cell align" above which
        tracks cell-based alignment, not celltype): cells with a celltype
        assigned / total cells, and the same for spots.
        """
        ip = self.ui.IngestionPanel
        ap = self.ui.AlignmentPanel
        storage_paths = self._all_vlinks_storage_paths()
        fov_list = self._parse_fov_list(ip.FovListLineEdit.text())
        if not storage_paths or not fov_list:
            QtWidgets.QMessageBox.warning(self, 'Cell/Spot Memory Status',
                                          'Set a storage path (Ingestion tab, and/or the Alignment tab\'s '
                                          'Cross-Modal RNA/DNA fields) and a FOV list first.')
            return

        rna_path = ap.RnaStoragePathLineEdit.text().strip()
        dna_path = ap.DnaStoragePathLineEdit.text().strip()
        cross_paired = bool(rna_path and dna_path)

        rows = []
        for storage_path in storage_paths:
            for fov in fov_list:
                cell_dicts, _ = vlinks_store.read_cells(storage_path, fov)
                cell_dicts = cell_dicts or []
                saved_at = (vlinks_store.summarize_fov(storage_path, fov) or {}).get('saved_at', '')

                fov_total = 0
                fov_computed = 0
                row_hybe_records = self._hybe_records_for_storage_path(storage_path)
                if row_hybe_records:
                    ingested, _, _ = self._ingested_hybes_for_fov(storage_path, fov, row_hybe_records)
                    fov_total = len(ingested)
                    if ingested:
                        ingested_records = [r for r in row_hybe_records if r['folder'] in ingested]
                        matrices = alignment.read_same_modality_matrices(storage_path, fov, ingested_records)
                        reference_hybe = self._reference_hybe_for_storage_path(storage_path)
                        # identity is BOTH the correct, real matrix for the
                        # reference hybe itself (aligned to itself) AND the
                        # not-yet-aligned seed default for every other hybe
                        # -- only the actual reference hybe should count as
                        # "computed" when its matrix happens to be identity.
                        fov_computed = sum(1 for hybe, H in matrices.items()
                                          if hybe == reference_hybe or not np.allclose(H, np.eye(3)))

                cross_total = 1 if cross_paired else 0
                cross_computed = 0
                if cross_total:
                    if (storage_path, fov) in self.cross_modal_result:
                        cross_computed = 1
                    else:
                        # in-memory cache only reflects THIS session's runs --
                        # a matrix computed in an earlier session is still
                        # real and persisted (write_cross_modal_matrix writes
                        # into the DNA reference hybe's own H5), so fall back
                        # to reading it straight off disk before calling this
                        # FOV "not yet cross-modal aligned".
                        dna_reference_hybe = ap.DnaReferenceHybeComboBox.currentText().strip()
                        if dna_reference_hybe:
                            try:
                                H_disk = alignment.read_cross_modal_matrix(dna_path, fov, dna_reference_hybe)
                            except Exception:
                                H_disk = None
                            cross_computed = 1 if H_disk is not None else 0

                cell_total = len(cell_dicts)
                cell_computed = sum(1 for c in cell_dicts if c.get('matrices'))
                cell_celltype_computed = sum(1 for c in cell_dicts if c.get('linked'))

                all_spots = [s for c in cell_dicts for s in c.get('spots', [])]
                spot_total = len(all_spots)
                spot_computed = sum(1 for s in all_spots if s.get('linked'))

                rows.append({'storage_path': storage_path, 'fov': fov, 'saved_at': saved_at, 'n_spots': spot_total,
                            'fov_computed': fov_computed, 'fov_total': fov_total,
                            'cross_computed': cross_computed, 'cross_total': cross_total,
                            'cell_computed': cell_computed, 'cell_total': cell_total,
                            'cell_celltype_computed': cell_celltype_computed, 'cell_celltype_total': cell_total,
                            'spot_computed': spot_computed, 'spot_total': spot_total})
        self.memory_viewer.set_data(rows)

    def _show_mip_viewer(self, silent=False):
        """
        silent=True is used by the "keep an already-open viewer live as
        combos change" wiring -- best-effort, no error dialogs, since a
        combo box can be in a legitimate (if now further guarded against
        via blockSignals) transient state mid-update that isn't a real
        user-facing error.
        """
        ip = self.ui.IngestionPanel
        storage_path = ip.StoragePathLineEdit.text().strip()
        hybe = ip.ViewerHybeComboBox.currentText()
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
        self.ui.AlignmentPanel.populate_reference_hybe_choices(self.hybe_records)
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
                                                  overwrite=self._job_queue_overwrite)
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
        ip = self.ui.IngestionPanel
        cp = self.ui.CellSegmentPanel
        storage_path = ip.StoragePathLineEdit.text().strip()
        if not storage_path:
            QtWidgets.QMessageBox.warning(self, 'Run Segmentation', 'Set storage path in the Ingestion tab first.')
            return
        fov = cp.FovSpinBox.value()
        reference_hybe = cp.ReferenceHybeComboBox.currentText()
        channel_text = cp.ChannelComboBox.currentText()
        if not reference_hybe or not channel_text:
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
            self._on_cell_segment_finished(mask, reference_image, fov, reference_hybe, append=append)
            self.cell_displayer.ManualAddModeCheckBox.setChecked(True)
            return

        cp.RunSegmentationPushButton.setEnabled(False)
        cp.LogTextEdit.append(f'Segmenting FOV{fov:02d} ({reference_hybe}, ch{channel}, method={method})...')
        self.statusBar().showMessage('Segmenting...')

        if method == 'cellpose':
            diameter = cp.DiameterSpinBox.value()
            min_size = cp.MinSizeSpinBox.value()
            max_size = cp.MaxSizeSpinBox.value()
            self._segment_worker = CellSegmentWorker(storage_path, fov, reference_hybe, channel, diameter, min_size, max_size)
        else:
            classical_method = cp.ClassicalAlgorithmComboBox.currentText().lower()
            absolute_cutoff = cp.ClassicalAbsoluteCutoffSpinBox.value()
            min_distance = cp.ClassicalMinDistanceSpinBox.value()
            min_size = cp.ClassicalMinSizeSpinBox.value()
            max_size = cp.ClassicalMaxSizeSpinBox.value()
            self._segment_worker = ClassicalSegmentWorker(storage_path, fov, reference_hybe, channel,
                                                           classical_method, absolute_cutoff, min_distance, min_size, max_size)
        self._segment_worker.finished_ok.connect(
            lambda mask, ref_img: self._on_cell_segment_finished(mask, ref_img, fov, reference_hybe, append=append))
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

    def _on_cell_segment_finished(self, mask, reference_image, fov, reference_hybe, append=False):
        cp = self.ui.CellSegmentPanel
        if self.cell_container is None:
            self.cell_container = CellContainer([fov], modality=self.current_modality)
        self.cell_container.data.setdefault(fov, [])

        if append and self._last_segment_context is not None and self._last_segment_context['fov'] == fov \
                and self.cell_displayer.mask is not None and self.cell_displayer.mask.shape == mask.shape:
            n_before = int(self.cell_displayer.mask.max())
            mask = self._merge_append_mask(self.cell_displayer.mask, mask)
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
        self.cell_container.load_new_cells(fov, mask, reference_hybe)
        self._last_segment_context = {'fov': fov, 'reference_hybe': reference_hybe}
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

    def _toggle_cell_displayer(self, checked):
        if checked:
            self._ensure_cell_displayer_initialized()
            self.cell_displayer.show()
            self.cell_displayer.raise_()
        else:
            self.cell_displayer.hide()

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
        ip = self.ui.IngestionPanel
        fov = cp.FovSpinBox.value()
        storage_path = ip.StoragePathLineEdit.text().strip()
        reference_hybe = cp.ReferenceHybeComboBox.currentText()
        channel_text = cp.ChannelComboBox.currentText()
        if not storage_path or not reference_hybe or not channel_text:
            return
        channel = int(channel_text)
        reference_image = vlinks_store.read_hybe_mip(storage_path, fov, reference_hybe, channel)
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
            cells = self.cell_container_permanent.data[fov]

        if self.cell_container is None:
            self.cell_container = CellContainer([fov], modality=self.current_modality)
        self.cell_container.data.setdefault(fov, [])

        # cells always show, regardless of which hybe/channel the panel is
        # currently displaying -- get_area_in_readout(reference_hybe,
        # modality) transforms each cell's mask into this hybe's own frame
        # via whatever matrix is known, defaulting to identity (the cell's
        # own raw/untransformed position) when cell-based alignment hasn't
        # been run for it yet, rather than hiding the mask entirely -- an
        # earlier version suppressed the whole overlay on a hybe mismatch,
        # which is exactly the behavior this replaces per explicit
        # correction. get_area_in_readout itself never raises (identity is
        # a real, valid answer, not an error) -- the "approximate" note
        # below is a direct membership check, not exception-based, purely
        # to tell the user WHY a shown position might be less precise.
        mask = np.zeros(reference_image.shape, dtype=np.uint8)
        if cells:
            approximate = False
            height, width = mask.shape
            for cell in cells:
                # reference_hybe belongs to whichever modality is current
                # (it's drawn from the Cell Segmentation panel's own live
                # hybe context) -- NOT necessarily this cell's own
                # segmentation modality, since cells here can come from
                # either modality.
                if (reference_hybe, self.current_modality) not in cell.matrices:
                    approximate = True
                x, y = cell.get_area_in_readout(reference_hybe, self.current_modality)
                xi, yi = x.astype(int), y.astype(int)
                valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
                mask[yi[valid], xi[valid]] = cell.id
            self.cell_container.data[fov] = deepcopy(cells)
            if approximate:
                cp.LogTextEdit.append(f'Note: no alignment matrix for {reference_hybe} yet -- cell positions shown here '
                                      f'are raw/untransformed (approximate) until cell-based alignment is run for this hybe.')

        self._last_segment_context = {'fov': fov, 'reference_hybe': reference_hybe}
        self.cell_displayer.set_data(reference_image, mask)
        cp.LogTextEdit.append(f'Displayer showing FOV{fov:02d} ({reference_hybe}, ch{channel}) -- {len(cells) if cells else 0} cell(s).')

    def _on_displayer_mask_edited(self, mask):
        if self._last_segment_context is None or self.cell_container is None:
            return
        fov = self._last_segment_context['fov']
        reference_hybe = self._last_segment_context['reference_hybe']
        self.cell_container.load_new_cells(fov, mask, reference_hybe)
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
        vlinks_store.write_global_params(rna_storage_path,
                                         cross_modal_role='RNA',
                                         cross_modal_paired_storage_path=dna_storage_path,
                                         cross_modal_rna_reference_hybe=rna_reference_hybe,
                                         cross_modal_dna_reference_hybe=dna_reference_hybe,
                                         cross_modal_channel_type=channel_type)
        vlinks_store.write_global_params(dna_storage_path,
                                         cross_modal_role='DNA',
                                         cross_modal_paired_storage_path=rna_storage_path,
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
        fov = self._last_segment_context['fov']
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
            self.cell_container_permanent = CellContainer([fov], modality=self.cell_container.modality)
        self.cell_container_permanent.data[fov] = deepcopy(self.cell_container.data[fov])
        storage_paths = self._all_vlinks_storage_paths()
        if storage_paths:
            vlinks_store.mirror_write_cells(storage_paths, fov, self.cell_container_permanent)
            segmentation_reference_hybe = self._last_segment_context['reference_hybe']
            for path in storage_paths:
                vlinks_store.write_fov_params(path, fov, segmentation_reference_hybe=segmentation_reference_hybe)
            where = ', '.join(storage_paths)
            cp.LogTextEdit.append(f'Saved {len(self.cell_container.data[fov])} cell(s) for FOV{fov:02d} to permanent '
                                  f'container and vlinks.h5 ({where}).')
        else:
            cp.LogTextEdit.append(f'Saved {len(self.cell_container.data[fov])} cell(s) for FOV{fov:02d} to permanent '
                                  f'container (no storage path set -- not written to vlinks.h5).')

    def _discard_cells(self):
        cp = self.ui.CellSegmentPanel
        if self.cell_container is None or self._last_segment_context is None:
            return
        fov = self._last_segment_context['fov']
        self.cell_container.data[fov] = []
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
            self.cell_container = CellContainer([fov], modality=self.cell_container_permanent.modality)
        self.cell_container.data[fov] = deepcopy(self.cell_container_permanent.data[fov])
        cp.LogTextEdit.append(f'Pulled {len(self.cell_container.data[fov])} cell(s) from permanent for FOV{fov:02d}.')

    def _activate_fov(self, fov):
        """
        "Activation": whatever's already been computed and persisted for
        this FOV -- segmented cells (with their spots and per-cell
        alignment matrices, all riding along inside CellContainer.save()),
        plus FOV-level alignment matrices -- gets pulled into the running
        transient/runtime state automatically the moment this FOV becomes
        current, with no button and no re-computation required. Fires on
        every FOV switch (Cell Segmentation and Spot Localization share
        the same FovSpinBox) and right after ingestion/parsing complete,
        per the explicit principle that already-persisted state should
        never require redoing the work that produced it.
        """
        ip = self.ui.IngestionPanel
        storage_path = ip.StoragePathLineEdit.text().strip()
        if storage_path and self.hybe_records:
            key = (storage_path, fov)
            if key not in self.fov_matrices:
                try:
                    self.fov_matrices[key] = alignment.read_same_modality_matrices(storage_path, fov, self.hybe_records)
                except Exception:
                    pass
            if key not in self.fov_unassigned_spots:
                try:
                    spot_dicts = vlinks_store.read_fov_spots(storage_path, fov)
                    spots = []
                    for d in spot_dicts:
                        spot = ASpot()
                        spot.set_metadata(**d)
                        spots.append(spot)
                    if spots:
                        self.fov_unassigned_spots[key] = spots
                except Exception:
                    pass
        self._try_show_existing_cells(fov)

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
        cells = self.cell_container_permanent.data[fov]
        reference_hybe = cells[0].reference_hybe
        frame_shape = cells[0].frame_shape
        # same cross-modality mismatch _show_celltype_result guards against
        # -- these cells may have been segmented under a DIFFERENT modality
        # than the one currently active (they're mirrored into every
        # configured modality's own vlinks.h5), so reference_hybe might not
        # exist under the CURRENT storage_path at all. Resolve from the
        # cell's own segmentation modality first.
        cell_storage_path = self.modality_data.get(cells[0].modality, {}).get('storage_path') or storage_path
        reference_image = vlinks_store.fiducial_channel_mip(cell_storage_path, fov, reference_hybe)
        if reference_image is None:
            cp.LogTextEdit.append(f'{reference_hybe} not in vlinks.h5 for FOV{fov:02d} -- cannot show existing cells.')
            return False

        # same uint8 label convention segment_fov/segment_fov_classical
        # already use elsewhere in this app (max 255 cells/FOV) -- not a
        # new limitation introduced here
        mask = np.zeros(frame_shape, dtype=np.uint8)
        for cell in cells:
            x, y = cell.area
            mask[y.astype(int), x.astype(int)] = cell.id

        if self.cell_container is None:
            self.cell_container = CellContainer([fov], modality=self.cell_container_permanent.modality)
        self.cell_container.data[fov] = deepcopy(cells)
        self._last_segment_context = {'fov': fov, 'reference_hybe': reference_hybe}
        self.cell_displayer.set_data(reference_image, mask)
        ap = self.ui.AlignmentPanel
        ap.CellOverlayFovSpinBox.blockSignals(True)
        ap.CellOverlayFovSpinBox.setValue(fov)
        ap.CellOverlayFovSpinBox.blockSignals(False)
        self._refresh_cell_fov_panels(fov)
        cp.LogTextEdit.append(f'Showing {len(cells)} already-saved cell(s) for FOV{fov:02d} (from permanent container).')
        return True

    @staticmethod
    def _cell_hybe_result_label(cell, fov, hybe, modality, reference_key):
        """
        Row text for one (cell, hybe) pair in "Results (per cell, per
        hybe)" -- 'FOV{fov:03d} Cell {cell:03d}: {hybe} | {reference}',
        per explicit request. The "(modality)" suffix (on either side)
        only appears when that hybe belongs to the OTHER (not this cell's
        own) modality -- hybe names aren't guaranteed unique across
        modalities, so the bare name alone would be ambiguous there;
        cell.matrices' own (hybe, modality) key always disambiguates it
        correctly regardless.
        """
        hybe_label = hybe if modality == cell.modality else f'{hybe} ({modality})'
        if reference_key:
            reference_hybe, reference_modality = reference_key
            reference_label = (reference_hybe if reference_modality == cell.modality
                               else f'{reference_hybe} ({reference_modality})')
        else:
            reference_label = cell.reference_hybe
        return f'FOV{fov:03d} Cell {cell.id:03d}: {hybe_label} | {reference_label}'

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
            cells = self.cell_container_permanent.data.get(fov, [])
        if not cells and self.cell_container is not None:
            cells = self.cell_container.data.get(fov, [])
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
        identified by the tier-1 FOV/Cell ID spinboxes, using tier-1's
        own Reference hybe combo (ap.CellReferenceHybeComboBox -- the
        actual anchor a per-cell-alignment run uses/used) as the
        reference -- per explicit request, this list shows the result of
        per-cell alignment for THIS run's configuration (FOV, Cell ID,
        Reference hybe all from tier 1), not a free-choice browsing tool.
        That's a SEPARATE pairing: Overlay FOV + Preview reference hybe +
        "Results (per cell, overlay)" (tier 3 -- see
        _refresh_cell_overlay_list/_show_cell_all_readouts_overlay).
        Refreshes live whenever FOV, Cell ID, or Reference hybe changes
        (see _refresh_cell_per_hybe_results_from_spinboxes). Pure read of
        already-saved/staged real cell data -- never computes or writes.
        Uses its own _cell_per_hybe_context, separate from
        _cell_alignment_display_cells (tier 3's "every cell in this FOV"
        state, which Save All Cell Overlays batches over) -- this list's
        single-cell scope must never narrow that batch down to one cell.
        """
        ap = self.ui.AlignmentPanel
        ip = self.ui.IngestionPanel
        storage_path = ip.StoragePathLineEdit.text().strip()
        cell = None
        if self.cell_container_permanent is not None:
            cell = next((c for c in self.cell_container_permanent.data.get(fov, []) if c.id == cell_id), None)
        if cell is None and self.cell_container is not None:
            cell = next((c for c in self.cell_container.data.get(fov, []) if c.id == cell_id), None)

        ap.CellResultsListWidget.clear()
        if cell is None or not cell.matrices:
            self._cell_per_hybe_context = None
            return

        this_modality = self._modality_for_storage_path(storage_path)
        # Same fallback the actual alignment run itself uses (see
        # _run_cell_alignment_for_selected_cell/compute_cell_alignment's
        # own reference_hybe=None default): an empty combo means "anchor
        # to the cell's own segmentation hybe."
        cell_reference_hybe = ap.CellReferenceHybeComboBox.currentText().strip() or cell.reference_hybe
        reference_key = (cell_reference_hybe, this_modality)
        self._cell_per_hybe_context = {'fov': fov, 'cell': cell, 'storage_path': storage_path,
                                       'hybe_records': self.hybe_records, 'reference_key': reference_key}
        for hybe, modality in sorted(cell.matrices.keys()):
            item = QtWidgets.QListWidgetItem(self._cell_hybe_result_label(cell, fov, hybe, modality, reference_key))
            item.setData(QtCore.Qt.UserRole, (fov, cell.id, hybe, modality))
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
        sp = self.ui.SpotLocalizationPanel
        fov = self._current_spot_fov()
        if self.cell_container is None or fov is None:
            sp.populate_cell_choices([])
        else:
            cells = self.cell_container.data.get(fov, [])
            sp.populate_cell_choices(cells)
            sp.LogTextEdit.append(f'Cell list refreshed: {len(cells)} cell(s) for FOV{fov:02d}.')
        self._refresh_spot_fov_summary()
        self._refresh_spot_breakdown()
        sp.RemoveAllSpotsPushButton.setEnabled(sp.current_view() == 'fov')

    def _refresh_spot_fov_summary(self):
        """
        Populates "FOV (all spots, this FOV)" -- per-(hybe, channel) spot
        COUNTS aggregated across every cell currently in this FOV, per
        explicit request ("to see all spots in FOV"), plus a second row
        per (hybe, channel) for spots that don't belong to any cell (see
        fov_unassigned_spots/_replace_fov_unassigned_spots) -- these are
        real, kept spots too, just without a cell link, so they get
        their own visible count rather than being folded silently into
        the cell-owned total. Pure read -- never computes or writes.
        """
        sp = self.ui.SpotLocalizationPanel
        ip = self.ui.IngestionPanel
        fov = self._current_spot_fov()
        sp.FovListWidget.clear()
        if self.cell_container is None or fov is None:
            return
        counts = {}  # (hybe, channel) -> [n_spots, {cell_id, ...}]
        for cell in self.cell_container.data.get(fov, []):
            for s in cell.spots:
                entry = counts.setdefault((s.hybe, s.channel), [0, set()])
                entry[0] += 1
                entry[1].add(cell.id)
        for hybe, channel in sorted(counts.keys()):
            n_spots, cell_ids = counts[(hybe, channel)]
            item = QtWidgets.QListWidgetItem(f'{hybe} ch{channel}: {n_spots} spot(s) across {len(cell_ids)} cell(s)')
            sp.FovListWidget.addItem(item)
        storage_path = ip.StoragePathLineEdit.text().strip()
        unassigned = self.fov_unassigned_spots.get((storage_path, fov), [])
        unassigned_counts = {}
        for s in unassigned:
            unassigned_counts[(s.hybe, s.channel)] = unassigned_counts.get((s.hybe, s.channel), 0) + 1
        for hybe, channel in sorted(unassigned_counts.keys()):
            item = QtWidgets.QListWidgetItem(
                f'{hybe} ch{channel}: {unassigned_counts[(hybe, channel)]} spot(s) unassigned (no cell)')
            sp.FovListWidget.addItem(item)

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
            for s in cell.spots:
                counts[(s.hybe, s.channel)] = counts.get((s.hybe, s.channel), 0) + 1
            suffix = 'spot(s)'
        else:
            storage_path = self.ui.IngestionPanel.StoragePathLineEdit.text().strip()
            fov = self._current_spot_fov()
            for s in self.fov_unassigned_spots.get((storage_path, fov), []):
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
        for cell in self.cell_container.data.get(fov, []):
            if cell.id == cell_id:
                return cell
        return None

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
        """
        x_area, y_area = cell.get_area_in_readout(hybe, modality)
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
        return {'img': mip_crop, 'mask': mask, 'rxmin': rxmin, 'rymin': rymin}

    def _load_spot_crop_for_display(self, *_args):
        sp = self.ui.SpotLocalizationPanel
        ip = self.ui.IngestionPanel
        cell = self._selected_spot_cell()
        hybe = sp.HybeComboBox.currentText()
        channel_text = sp.ChannelComboBox.currentText()
        if cell is None or not hybe or not channel_text:
            return
        channel = int(channel_text)
        storage_path = ip.StoragePathLineEdit.text().strip()
        fov = self._current_spot_fov()
        pad = sp.PadSpinBox.value()
        crop = self._build_cell_display_crop(cell, hybe, channel, storage_path, fov, pad, modality=self.current_modality)
        if crop is None:
            sp.LogTextEdit.append(f'Cell {cell.id} has no alignment/overlap for {hybe} yet -- '
                                  f'run cell-based alignment for this hybe first.')
            return
        rxmin, rymin = crop['rxmin'], crop['rymin']
        self._spot_crop_context = {'kind': 'cell', 'cell': cell, 'hybe': hybe, 'channel': channel,
                                   'rxmin': rxmin, 'rymin': rymin}
        existing_points = [(s.raw_coordinate[0] - rxmin, s.raw_coordinate[1] - rymin)
                           for s in cell.spots if s.hybe == hybe and s.channel == channel]
        self.spot_crop_displayer.set_data(crop['img'], existing_points, mask=crop['mask'])

    def _load_fov_spot_display(self):
        """
        FOV view -- the full raw hybe/channel MIP with BOTH the current
        FOV-level unassigned-spot pool (fov_unassigned_spots, yellow --
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
        genuinely edit fov_unassigned_spots via _on_spot_crop_edited.
        """
        sp = self.ui.SpotLocalizationPanel
        ip = self.ui.IngestionPanel
        storage_path = ip.StoragePathLineEdit.text().strip()
        fov = self._current_spot_fov()
        hybe = sp.HybeComboBox.currentText()
        channel_text = sp.ChannelComboBox.currentText()
        if not storage_path or fov is None or not hybe or not channel_text:
            return
        channel = int(channel_text)
        mip = vlinks_store.read_hybe_mip(storage_path, fov, hybe, channel)
        if mip is None:
            sp.LogTextEdit.append(f'{hybe} ch{channel} not in vlinks.h5 for FOV{fov:02d} -- ingest it first.')
            return
        unassigned_points = [(float(s.raw_coordinate[0]), float(s.raw_coordinate[1]))
                             for s in self.fov_unassigned_spots.get((storage_path, fov), [])
                             if s.hybe == hybe and s.channel == channel]
        cell_owned_points = []
        if self.cell_container is not None:
            for cell in self.cell_container.data.get(fov, []):
                for s in cell.spots:
                    if s.hybe == hybe and s.channel == channel:
                        cell_owned_points.append((float(s.raw_coordinate[0]), float(s.raw_coordinate[1])))
        self._spot_crop_context = {'kind': 'fov', 'storage_path': storage_path, 'fov': fov,
                                   'hybe': hybe, 'channel': channel, 'rxmin': 0, 'rymin': 0}
        self.spot_crop_displayer.set_data(mip, unassigned_points, color='yellow', readonly_points=cell_owned_points)

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
        sp.RemoveAllSpotsPushButton.setEnabled(sp.current_view() == 'fov')
        if sp.ShowDisplayerPushButton.isChecked():
            self._show_spot_displayer()

    def _toggle_spot_crop_displayer(self, checked):
        if checked:
            self._show_spot_displayer()
            self.spot_crop_displayer.show()
            self.spot_crop_displayer.raise_()
        else:
            self.spot_crop_displayer.hide()

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
        storage_path = self.ui.IngestionPanel.StoragePathLineEdit.text().strip()
        fov = self._current_spot_fov()
        hybe = sp.HybeComboBox.currentText()
        channel_text = sp.ChannelComboBox.currentText()
        if not storage_path or fov is None or not hybe or not channel_text:
            return
        channel = int(channel_text)
        if sp.current_view() == 'cell':
            cell = self._selected_spot_cell()
            if cell is None:
                QtWidgets.QMessageBox.warning(self, 'Remove Transient Spots', 'Select a cell first.')
                return
            cell_dicts, _ = vlinks_store.read_cells(storage_path, fov)
            on_disk_cell = next((d for d in (cell_dicts or []) if d['id'] == cell.id), None)
            permanent = []
            for d in (on_disk_cell or {}).get('spots', []):
                if d.get('hybe') == hybe and d.get('channel') == channel:
                    spot = ASpot()
                    spot.set_metadata(**d)
                    permanent.append(spot)
            self._replace_cell_spots(cell, hybe, channel, permanent)
            sp.LogTextEdit.append(f'Cell {cell.id}, {hybe} ch{channel}: reverted to {len(permanent)} '
                                  f'permanent spot(s) (transient discarded).')
        else:
            on_disk = vlinks_store.read_fov_spots(storage_path, fov)
            permanent = []
            for d in on_disk:
                if d.get('hybe') == hybe and d.get('channel') == channel:
                    spot = ASpot()
                    spot.set_metadata(**d)
                    permanent.append(spot)
            self._replace_fov_unassigned_spots(storage_path, fov, hybe, channel, permanent)
            sp.LogTextEdit.append(f'FOV{fov:02d}, {hybe} ch{channel}: reverted to {len(permanent)} '
                                  f'permanent unassigned spot(s) (transient discarded).')
        self._refresh_spot_cell_list()
        if sp.ShowDisplayerPushButton.isChecked():
            self._show_spot_displayer()

    def _remove_all_spots_in_view(self):
        """
        Clears the current view's current (hybe, channel) outright --
        both permanent and transient. Nothing is deleted on disk by this
        alone; the emptied state only reaches vlinks.h5 once Save View is
        clicked afterward, same as any other in-memory edit here. Not to
        be confused with _remove_all_spots_in_fov (the FOV-view-only,
        immediately-persisted, whole-FOV wipe behind the separate
        "Remove All Spots" button at the very bottom of the panel).
        """
        sp = self.ui.SpotLocalizationPanel
        storage_path = self.ui.IngestionPanel.StoragePathLineEdit.text().strip()
        fov = self._current_spot_fov()
        hybe = sp.HybeComboBox.currentText()
        channel_text = sp.ChannelComboBox.currentText()
        if not storage_path or fov is None or not hybe or not channel_text:
            return
        channel = int(channel_text)
        if sp.current_view() == 'cell':
            cell = self._selected_spot_cell()
            if cell is None:
                QtWidgets.QMessageBox.warning(self, 'Remove spots in view', 'Select a cell first.')
                return
            self._replace_cell_spots(cell, hybe, channel, [])
            sp.LogTextEdit.append(f'Cell {cell.id}, {hybe} ch{channel}: all spots removed from view '
                                  f'(not yet saved).')
        else:
            self._replace_fov_unassigned_spots(storage_path, fov, hybe, channel, [])
            sp.LogTextEdit.append(f'FOV{fov:02d}, {hybe} ch{channel}: all unassigned spots removed from '
                                  f'view (not yet saved).')
        self._refresh_spot_cell_list()
        if sp.ShowDisplayerPushButton.isChecked():
            self._show_spot_displayer()

    def _remove_all_spots_in_fov(self):
        """
        FOV view only (guarded here too, not just via the button's
        setEnabled state) -- a COMPLETE wipe of every spot in this FOV:
        the unassigned pool AND every cell's own spots, across every
        hybe/channel (not just whatever's currently selected in the
        Hybe/Channel comboboxes). Confirmed via a warning dialog first.

        Unlike every other edit in this panel, this is immediately
        persisted as part of the same click, not staged for a later Save
        View -- Save View's own FOV-view branch (_save_fov_view) only
        ever writes cells it just identified NEW spots into, never a
        cell that's merely being emptied out, so a plain in-memory clear
        here wouldn't actually reach already-saved cells on disk the way
        every other "edit then Save View" flow does.
        """
        sp = self.ui.SpotLocalizationPanel
        if sp.current_view() != 'fov':
            return
        fov = self._current_spot_fov()
        storage_path = self.ui.IngestionPanel.StoragePathLineEdit.text().strip()
        storage_paths = self._all_vlinks_storage_paths()
        if not storage_path or fov is None or not storage_paths:
            return
        key = (storage_path, fov)
        n_unassigned = len(self.fov_unassigned_spots.get(key, []))
        cells = self.cell_container.data.get(fov, []) if self.cell_container else []
        n_cell_spots = sum(c.total_num_spots for c in cells)
        if n_unassigned == 0 and n_cell_spots == 0:
            QtWidgets.QMessageBox.information(self, 'Remove All Spots', f'FOV{fov:02d} has no spots to remove.')
            return
        reply = QtWidgets.QMessageBox.warning(
            self, 'Remove All Spots',
            f'This will permanently clear ALL spots for FOV{fov:02d} -- {n_unassigned} unassigned '
            f'spot(s) and {n_cell_spots} spot(s) across {len(cells)} cell(s) -- and save the '
            f'emptied state to vlinks.h5 immediately. This cannot be undone. Continue?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if reply != QtWidgets.QMessageBox.Yes:
            return
        self.fov_unassigned_spots[key] = []
        for cell in cells:
            cell.spots = []
            cell.num_spots = {}
            cell.total_num_spots = 0
            vlinks_store.mirror_write_single_cell(storage_paths, fov, cell)
        vlinks_store.mirror_write_fov_spots(storage_paths, fov, [])
        sp.LogTextEdit.append(f'FOV{fov:02d}: ALL spots removed ({n_unassigned} unassigned + '
                              f'{n_cell_spots} across {len(cells)} cell(s)) and saved to vlinks.h5.')
        self._refresh_spot_cell_list()
        if sp.ShowDisplayerPushButton.isChecked():
            self._show_spot_displayer()

    def _save_view(self):
        """
        Cell view: identical to the old narrow single-cell save -- writes
        ONLY the currently selected cell to vlinks.h5 (vlinks_store.
        write_single_cell reads the existing on-disk cell list, replaces
        just this cell's own entry by id, and writes that back -- every
        other cell's persisted data is untouched).

        FOV view: there's no single cell to scope to, so this instead
        triages the FOV's pending unassigned-spot pool (see
        _save_fov_view) -- identification against the current cell mask
        happens HERE, not at detect/click time (per explicit request).
        """
        sp = self.ui.SpotLocalizationPanel
        fov = self._current_spot_fov()
        storage_paths = self._all_vlinks_storage_paths()
        if not storage_paths:
            QtWidgets.QMessageBox.warning(self, 'Save View', 'No storage path available.')
            return
        if sp.current_view() == 'cell':
            cell = self._selected_spot_cell()
            if cell is None or self.cell_container is None:
                QtWidgets.QMessageBox.warning(self, 'Save View', 'Select a cell first.')
                return
            vlinks_store.mirror_write_single_cell(storage_paths, fov, cell)
            sp.LogTextEdit.append(f'Cell {cell.id} (FOV{fov:02d}): {cell.total_num_spots} spot(s) saved to vlinks.h5 '
                                  f'(this cell only, other cells on disk untouched).')
            QtWidgets.QMessageBox.information(self, 'Save View', f'Cell {cell.id} saved to vlinks.h5.')
        else:
            self._save_fov_view(fov, storage_paths)

    def _save_fov_view(self, fov, storage_paths):
        """
        Identifies each pending FOV-level unassigned spot (ACROSS every
        hybe/channel ever accumulated in fov_unassigned_spots for this
        FOV, not just the one currently selected in the Hybe/Channel
        comboboxes -- Save View commits the whole pending pool, matching
        "Save View... saves spots for the view") against the current
        cell mask. Newly-identified spots are APPENDED onto their owning
        cell's own spots (not a full replace -- this is one spot at a
        time joining whatever that cell already has), and that cell is
        written to disk narrowly (write_single_cell, same as a Cell-view
        save). Anything that still can't be identified (no segmented
        mask for this FOV, or genuinely outside every cell) stays in
        fov_unassigned_spots and is explicitly saved too, via
        write_fov_spots, per explicit request ("remember to save the
        unassigned spots too").

        The mask lives in the SEGMENTATION's own reference hybe frame
        (cell.reference_hybe), which is NOT necessarily the same hybe as
        the Same-Modality FOV alignment's own reference hybe that plain
        fov_matrices targets -- on this real dataset they're genuinely
        different (Hyb_500 vs Hyb_101). So the lookup point must be
        transformed via _fov_only_matrix_for_hybe(spot.hybe, modality,
        ref_cell, fov) (FOV/cross-modal composition into cell.reference_
        hybe's frame, no cell-level residual -- there's no owning cell
        yet to have a residual for), NOT the bare fov_matrices spot_
        mapper.raw_to_reference(..., cell=None) used to compute a
        REFERENCE-frame coordinate elsewhere in this app; using the
        latter here would silently look up the mask at the wrong pixel
        whenever those two reference hybes differ.
        """
        sp = self.ui.SpotLocalizationPanel
        storage_path = self.ui.IngestionPanel.StoragePathLineEdit.text().strip()
        key = (storage_path, fov)
        pending = self.fov_unassigned_spots.get(key, [])
        mask = self.cell_displayer.mask
        mask_matches_fov = bool(self._last_segment_context is not None and self._last_segment_context['fov'] == fov)
        cells = self.cell_container.data.get(fov, []) if self.cell_container else []
        cells_by_id = {c.id: c for c in cells}
        can_identify = mask is not None and mask_matches_fov and bool(cells_by_id)
        ref_cell = cells[0] if can_identify else None
        spot_modality = self._modality_for_storage_path(storage_path) if can_identify else None
        mask_lookup_matrix_cache = {}

        still_unassigned = []
        newly_identified_cells = {}
        for spot in pending:
            owning_cell = None
            if can_identify:
                H = mask_lookup_matrix_cache.get(spot.hybe)
                if H is None and spot.hybe not in mask_lookup_matrix_cache:
                    H = self._fov_only_matrix_for_hybe(spot.hybe, spot_modality, ref_cell, fov)
                    mask_lookup_matrix_cache[spot.hybe] = H
                if H is not None:
                    rx, ry, _ = H @ np.array([spot.raw_coordinate[0], spot.raw_coordinate[1], 1.0])
                    iry, irx = int(round(ry)), int(round(rx))
                    if 0 <= iry < mask.shape[0] and 0 <= irx < mask.shape[1]:
                        label = int(mask[iry, irx])
                        owning_cell = cells_by_id.get(label) if label != 0 else None
            if owning_cell is not None:
                cx, cy = spot_mapper.raw_to_reference(
                    (spot.raw_coordinate[0], spot.raw_coordinate[1]), spot.hybe, {},
                    modality=spot_modality, cell=owning_cell)
                spot.set_metadata(cell=owning_cell.id, coordinate=(cx, cy, 0.0))
                owning_cell.spots.append(spot)
                owning_cell.num_spots[spot.hybe] = sum(1 for s in owning_cell.spots if s.hybe == spot.hybe)
                owning_cell.total_num_spots = len(owning_cell.spots)
                newly_identified_cells[owning_cell.id] = owning_cell
            else:
                still_unassigned.append(spot)

        self.fov_unassigned_spots[key] = still_unassigned
        for cell in newly_identified_cells.values():
            vlinks_store.mirror_write_single_cell(storage_paths, fov, cell)
        vlinks_store.mirror_write_fov_spots(storage_paths, fov, still_unassigned)

        n_identified = len(pending) - len(still_unassigned)
        note = '' if can_identify else ' (no matching cell mask for this FOV -- nothing could be identified)'
        sp.LogTextEdit.append(f'FOV{fov:02d}: {n_identified} spot(s) identified into '
                              f'{len(newly_identified_cells)} cell(s), {len(still_unassigned)} remain '
                              f'unassigned{note}. All saved to vlinks.h5.')
        QtWidgets.QMessageBox.information(self, 'Save View',
                                          f'{n_identified} spot(s) assigned to cells, '
                                          f'{len(still_unassigned)} still unassigned.')
        self._refresh_spot_cell_list()
        if sp.ShowDisplayerPushButton.isChecked():
            self._show_spot_displayer()

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
        if not append:
            cell.spots = [s for s in cell.spots if not (s.hybe == hybe and s.channel == channel)]
        cell.spots.extend(new_spots)
        cell.num_spots[hybe] = sum(1 for s in cell.spots if s.hybe == hybe)
        cell.total_num_spots = len(cell.spots)

    def _replace_fov_unassigned_spots(self, storage_path, fov, hybe, channel, new_spots, append=False):
        """
        Full-replace, for exactly (hybe, channel), the FOV-level
        unassigned-spot pool -- same replace-not-append semantics (and
        the same append=True escape hatch) as _replace_cell_spots, just
        keyed by (storage_path, fov) instead of living on a cell (there
        is no owning cell for these by definition -- see _save_fov_view
        for when that gets decided).
        """
        key = (storage_path, fov)
        existing = self.fov_unassigned_spots.get(key, [])
        if not append:
            existing = [s for s in existing if not (s.hybe == hybe and s.channel == channel)]
        existing.extend(new_spots)
        self.fov_unassigned_spots[key] = existing

    def _on_spot_crop_edited(self, points):
        """
        Handles both views' manual clicks/removals -- the crop
        displayer's spots_edited signal always hands back the FULL
        current crop-local point list, regardless of which view
        (_spot_crop_context['kind']) is currently open. Cell view builds
        real cell-owned ASpots (cell=cell.id, coordinate via cell.matrix_to);
        FOV view builds unassigned ASpots (cell stays at its -1 default,
        coordinate == raw_coordinate, same as FOV-view auto-detect) --
        identification is deferred to Save View either way, matching FOV
        auto-detect's own deferred design.
        """
        ctx = self._spot_crop_context
        if ctx is None:
            return
        hybe, channel = ctx['hybe'], ctx['channel']
        rxmin, rymin = ctx['rxmin'], ctx['rymin']
        img = self.spot_crop_displayer.crop_image
        new_spots = []
        for x, y in points:
            raw_x, raw_y = x + rxmin, y + rymin
            iy, ix = int(round(y)), int(round(x))
            brightness = 0.0
            if img is not None and 0 <= iy < img.shape[0] and 0 <= ix < img.shape[1]:
                val = img[iy, ix]
                brightness = float(val) if np.isfinite(val) else 0.0
            spot = ASpot()
            if ctx['kind'] == 'cell':
                cell = ctx['cell']
                # empty fov_matrices: cell.matrices[(hybe, modality)] must
                # already exist to have gotten a crop at all (_build_cell_crop's
                # own precondition), and spot_mapper._resolve_matrix always
                # prefers cell.matrices over fov_matrices when a cell is given.
                cx, cy = spot_mapper.raw_to_reference((raw_x, raw_y), hybe, {}, modality=self.current_modality, cell=cell)
                spot.set_metadata(fov=cell.fov, hybe=hybe, channel=channel, cell=cell.id,
                                  coordinate=(cx, cy, 0.0), raw_coordinate=(raw_x, raw_y, 0.0),
                                  size=0.0, brightness=brightness)
            else:
                spot.set_metadata(fov=ctx['fov'], hybe=hybe, channel=channel,
                                  coordinate=(float(raw_x), float(raw_y), 0.0), raw_coordinate=(raw_x, raw_y, 0.0),
                                  size=0.0, brightness=brightness)
            new_spots.append(spot)

        sp = self.ui.SpotLocalizationPanel
        if ctx['kind'] == 'cell':
            cell = ctx['cell']
            self._replace_cell_spots(cell, hybe, channel, new_spots)
            sp.LogTextEdit.append(f'Cell {cell.id}, {hybe} ch{channel}: {len(new_spots)} spot(s) after manual edit.')
        else:
            self._replace_fov_unassigned_spots(ctx['storage_path'], ctx['fov'], hybe, channel, new_spots)
            sp.LogTextEdit.append(f'FOV{ctx["fov"]:02d}, {hybe} ch{channel}: {len(new_spots)} unassigned '
                                  f'spot(s) after manual edit.')
        self._refresh_spot_cell_list()

    def _run_spot_auto_detect(self):
        sp = self.ui.SpotLocalizationPanel
        ip = self.ui.IngestionPanel
        storage_path = ip.StoragePathLineEdit.text().strip()
        fov = self._current_spot_fov()
        hybe = sp.HybeComboBox.currentText()
        channel_text = sp.ChannelComboBox.currentText()
        if not storage_path or fov is None or not hybe or not channel_text:
            QtWidgets.QMessageBox.warning(self, 'Run Auto-Detect',
                                          'Set storage path/FOV in Ingestion, and pick a hybe/channel here.')
            return
        channel = int(channel_text)
        min_distance = sp.MinDistanceSpinBox.value()
        pad = sp.PadSpinBox.value()
        try:
            self._run_spot_auto_detect_body(sp, storage_path, fov, hybe, channel, min_distance, pad)
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

    def _run_spot_auto_detect_body(self, sp, storage_path, fov, hybe, channel, min_distance, pad):
        append = sp.AppendModeCheckBox.isChecked()
        if sp.current_view() == 'cell':
            cell = self._selected_spot_cell()
            if cell is None:
                QtWidgets.QMessageBox.warning(self, 'Run Auto-Detect', 'Select a cell first (Cell view).')
                return
            crop = localization._build_cell_crop(cell, hybe, channel, storage_path, fov, pad, modality=self.current_modality)
            if crop is None:
                QtWidgets.QMessageBox.warning(self, 'Run Auto-Detect', f'Cell {cell.id} has no alignment/overlap for {hybe} yet.')
                return
            img, rxmin, rymin = crop['img'], crop['rxmin'], crop['rymin']
            threshold_abs = sp.threshold_abs(np.nanmax(img))
            coords = peak_local_max(img, min_distance=min_distance, exclude_border=1, threshold_abs=threshold_abs)
            new_spots = []
            for y, x in coords:
                raw_x, raw_y = int(x) + rxmin, int(y) + rymin
                cx, cy = spot_mapper.raw_to_reference((raw_x, raw_y), hybe, {}, modality=self.current_modality, cell=cell)
                spot = ASpot()
                spot.set_metadata(fov=fov, hybe=hybe, channel=channel, cell=cell.id,
                                  coordinate=(cx, cy, 0.0), raw_coordinate=(raw_x, raw_y, 0.0),
                                  size=0.0, brightness=float(img[y, x]))
                new_spots.append(spot)
            self._replace_cell_spots(cell, hybe, channel, new_spots, append=append)
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
            # ownership is only decided later, at Save View time (see
            # _save_fov_view) -- per explicit request, identification is
            # deferred to save, not done eagerly at detect time.
            mip = vlinks_store.read_hybe_mip(storage_path, fov, hybe, channel)
            if mip is None:
                sp.LogTextEdit.append(f'{hybe} ch{channel} not in vlinks.h5 for FOV{fov:02d} -- ingest it first.')
                return
            threshold_abs = sp.threshold_abs(mip.max())
            coords = peak_local_max(mip, min_distance=min_distance, exclude_border=1, threshold_abs=threshold_abs)
            new_spots = []
            for y, x in coords:
                raw_x, raw_y = int(x), int(y)
                spot = ASpot()
                spot.set_metadata(fov=fov, hybe=hybe, channel=channel,
                                  coordinate=(float(raw_x), float(raw_y), 0.0), raw_coordinate=(raw_x, raw_y, 0.0),
                                  size=0.0, brightness=float(mip[y, x]))
                new_spots.append(spot)
            self._replace_fov_unassigned_spots(storage_path, fov, hybe, channel, new_spots, append=append)
            sp.LogTextEdit.append(f'FOV{fov:02d} {hybe} ch{channel}: {len(new_spots)} peak(s) detected '
                                  f'(unassigned{", appended" if append else ""} -- run Save View to identify cell ownership).')
            self._refresh_spot_cell_list()
            self._load_fov_spot_display()
            self.spot_crop_displayer.show()
            self.spot_crop_displayer.raise_()

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
        hybe, channel = inputs['hybe'], inputs['channel']
        if not hybe or channel is None:
            QtWidgets.QMessageBox.warning(self, 'Assign Barcode Channel', 'Pick a barcode hybe/channel first.')
            return
        self._barcode_channel_by_celltype[celltype_name] = (hybe, channel)
        ctp.LogTextEdit.append(f'{celltype_name} <- {hybe} ch{channel}')
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
        hybe, channel = bch
        storage_path = ip.StoragePathLineEdit.text().strip()
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
            hybe, channel = bch
            n_fovs = len(self._barcode_calibration['scale'].get(bch, {}))
            cal_lines.append(f'{name}: {hybe} ch{channel} (calibrated for {n_fovs} FOV(s))')
        ctp.CalibrationSummaryTextEdit.setPlainText('\n'.join(cal_lines))

    def _show_barcode_overview(self):
        ctp = self.ui.CelltypeDeterminationPanel
        ip = self.ui.IngestionPanel
        storage_path = ip.StoragePathLineEdit.text().strip()
        fov_text = ctp.OverviewFovLineEdit.text().strip()
        fov = int(fov_text) if fov_text else self._current_spot_fov()
        if not storage_path or fov is None:
            QtWidgets.QMessageBox.warning(self, 'Show Barcode Overview', 'Set storage path (Ingestion tab) and an FOV.')
            return
        # warp each channel into a common frame for visualization ONLY
        # (never for stored/analyzed data) -- same established exception
        # used by every alignment preview in canvas/pipeline_canvas.py
        fmats = self.fov_matrices.get((storage_path, fov), {})
        images_by_channel, labels_by_channel = {}, {}
        for name in ctp.celltype_names():
            bch = self._barcode_channel_by_celltype.get(name)
            if bch is None:
                continue
            hybe, channel = bch
            img = vlinks_store.read_hybe_mip(storage_path, fov, hybe, channel)
            if img is None:
                ctp.LogTextEdit.append(f'Overview: {hybe} ch{channel} not in vlinks.h5 -- ingest it first.')
                continue
            if hybe in fmats:
                height, width = img.shape
                img = cv2.warpAffine(img.astype(np.float32), fmats[hybe][:2], (width, height))
            images_by_channel[bch] = img
            labels_by_channel[bch] = f'{name}: {hybe} ch{channel}'
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

    def _run_celltype_determination(self):
        ctp = self.ui.CelltypeDeterminationPanel
        ip = self.ui.IngestionPanel
        storage_path = ip.StoragePathLineEdit.text().strip()
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
            self._run_celltype_determination_body(ctp, storage_path, containers, names, now)
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

    def _run_celltype_determination_body(self, ctp, storage_path, containers, names, now):
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
            for container in containers:
                is_permanent = container is self.cell_container_permanent
                for cells in container.data.values():
                    for cell in cells:
                        cell.celltype = celltype.classify_fov(cell.fov, celltype_from_fov)
                        cell.linked, cell.linked_at = True, now
                        for spot in cell.spots:
                            spot.celltype = cell.celltype
                            spot.linked, spot.linked_at = True, now
                        n_cells += 1
                        last_fov = cell.fov
                        if is_permanent:
                            permanent_fovs_touched.add(cell.fov)
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

        image_cache = {}  # {fov(str): {(hybe,channel): ndarray or None}}
        n_cells, n_spots, n_cells_skipped = 0, 0, 0
        last_fov = None
        permanent_fovs_touched = set()
        for container in containers:
            is_permanent = container is self.cell_container_permanent
            for fov, cells in container.data.items():
                if cells:
                    last_fov = fov
                fov_key = str(fov)
                if fov_key not in image_cache:
                    image_cache[fov_key] = {}
                    for bch in barcode_channel:
                        hybe, channel_id = bch
                        image_cache[fov_key][bch] = vlinks_store.read_hybe_mip(storage_path, fov, hybe, channel_id)

                # calibration is per-(hybe,channel)-per-FOV (Apply Calibration
                # only covers whichever FOV(s) were explicitly calibrated) --
                # missing it for even one barcode channel used to reach
                # classify_cell_barcode's unconditional barcode['scale'][bch][fov]
                # lookup and crash with an uncaught KeyError, silently aborting
                # the whole run with no dialog (confirmed via a real crash:
                # KeyError('Hyb_130', 635), repeated across 3 separate clicks
                # with zero visible feedback each time). Checked once per FOV,
                # not per cell -- calibration completeness doesn't vary by cell.
                missing_calibration = [bch for bch in barcode_channel
                                       if int(fov) not in self._barcode_calibration['scale'].get(bch, {})
                                       or int(fov) not in self._barcode_calibration['lower_bound'].get(bch, {})
                                       or int(fov) not in self._barcode_calibration['upper_bound'].get(bch, {})]
                if missing_calibration:
                    ctp.LogTextEdit.append(f'FOV{fov:02d}: skipped ({len(cells)} cell(s)) -- no calibration for '
                                           f'{missing_calibration} at this FOV (Apply Calibration first).')
                    n_cells_skipped += len(cells)
                    continue

                for cell in cells:
                    x_ref, y_ref = cell.area
                    area_by_channel = {}
                    for bch in barcode_channel:
                        hybe, channel_id = bch
                        img = image_cache[fov_key].get(bch)
                        if img is None:
                            continue
                        # barcode hybes always belong to self.current_modality
                        # (BarcodeHybeComboBox's own scope). _matrix_to_cellref
                        # prefers a real, cell-level-refined entry when
                        # compute_cell_alignment has run for (hybe, self.
                        # current_modality) -- including cell.reference_hybe's
                        # own entry, which is no longer forced to identity --
                        # falling back to the FOV/cross-modal-only matrix
                        # otherwise, exactly like compute_cell_alignment's own
                        # reject/out-of-frame fallbacks do. A plain per-point
                        # inverse transform, not cell.get_area_in_readout's
                        # masking/closing machinery, so point order/count
                        # stays identical across every channel.
                        H = self._matrix_to_cellref(hybe, self.current_modality, cell, cell.fov)
                        if H is None:
                            continue
                        pts = la.inv(H) @ np.vstack([x_ref, y_ref, np.ones_like(x_ref, dtype=float)])
                        x_h, y_h = pts[0], pts[1]
                        height, width = img.shape
                        area_by_channel[bch] = (np.clip(x_h, 0, width - 1), np.clip(y_h, 0, height - 1))
                    if len(area_by_channel) < len(barcode_channel):
                        n_cells_skipped += 1
                        continue
                    cell.celltype = celltype.classify_cell_barcode(area_by_channel, cell.fov, image_cache,
                                                                   celltype_determination, method=ctp.barcode_method())
                    cell.linked, cell.linked_at = True, now
                    n_cells += 1
                    if is_permanent:
                        permanent_fovs_touched.add(fov)

                    for spot in cell.spots:
                        xy_by_channel = {}
                        for bch in barcode_channel:
                            hybe, channel_id = bch
                            # same "best available, no no-alignment" resolution
                            # as the cell-area loop above.
                            H = self._matrix_to_cellref(hybe, self.current_modality, cell, cell.fov)
                            if H is None:
                                continue
                            sx, sy, _ = la.inv(H) @ np.array([spot.coordinate[0], spot.coordinate[1], 1.0])
                            xy_by_channel[bch] = (sx, sy)
                        if len(xy_by_channel) < len(barcode_channel):
                            continue
                        spot.celltype = celltype.classify_spot_barcode(xy_by_channel, cell.fov, image_cache, celltype_determination)
                        spot.linked, spot.linked_at = True, now
                        n_spots += 1

        self._persist_celltype_results(permanent_fovs_touched)
        ctp.RunCelltypeDeterminationPushButton.setEnabled(True)
        ctp.LogTextEdit.append(f'Barcode-mode: {n_cells} cell(s), {n_spots} spot(s) classified '
                               f'({n_cells_skipped} cell(s) skipped -- missing alignment for a barcode hybe)'
                               f'{f", saved to vlinks.h5 for FOV(s) {sorted(permanent_fovs_touched)}" if permanent_fovs_touched else ""}.')
        self.statusBar().showMessage('Celltype determination complete.', 5000)
        QtWidgets.QMessageBox.information(self, 'Celltype determination complete',
                                          f'{n_cells} cell(s), {n_spots} spot(s) classified '
                                          f'({n_cells_skipped} cell(s) skipped -- missing alignment for a barcode hybe).')
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
                cells = container.data[fov]
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
        cell_storage_path = self.modality_data.get(cells[0].modality, {}).get('storage_path') or storage_path
        reference_image = vlinks_store.fiducial_channel_mip(cell_storage_path, fov, reference_hybe)
        if reference_image is None:
            QtWidgets.QMessageBox.critical(self, 'Show Celltype Result',
                                           f'{reference_hybe} not in vlinks.h5 for FOV{fov:02d}.')
            return

        mask = np.zeros(frame_shape, dtype=np.uint8)
        for cell in cells:
            x, y = cell.area
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
        ip = self.ui.IngestionPanel
        if not self.hybe_records:
            QtWidgets.QMessageBox.warning(self, 'Run FOV Alignment', 'Parse a layout first.')
            return
        reference_hybe = ap.ReferenceHybeComboBox.currentText()
        storage_path = ip.StoragePathLineEdit.text().strip()
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
        hybe_records = self.modality_data.get(self.current_modality, {}).get('active_hybe_list', [])
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
        if not self.hybe_records:
            QtWidgets.QMessageBox.warning(self, 'Run FOV Alignment', 'Parse a layout first.')
            return
        reference_hybe = ap.ReferenceHybeComboBox.currentText()
        storage_path = ip.StoragePathLineEdit.text().strip()
        fov_list = self._parse_fov_list(ip.FovListLineEdit.text())
        if not storage_path or not fov_list:
            QtWidgets.QMessageBox.warning(self, 'Run FOV Alignment', 'Set storage path and FOV list in the Ingestion tab first.')
            return

        hybe_records = self.modality_data.get(self.current_modality, {}).get('active_hybe_list', [])
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
        self.fov_matrices.update({(storage_path, fov): m for fov, m in results.items()})
        self._pending_same_modality_alignment = None
        self._refresh_same_modality_results_list()
        ap.RunAllFovAlignmentPushButton.setEnabled(True)
        self.statusBar().showMessage('FOV alignment computed.', 5000)

        channel_type = ap.SameModalityChannelTypeComboBox.currentText()
        vlinks_store.write_global_params(storage_path, same_modality_reference_hybe=reference_hybe,
                                         same_modality_channel_type=channel_type)
        for fov, matrices in results.items():
            save_path = os.path.join(storage_path, f'FOV{fov:02d}', 'alignment_overlay.png')
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
            save_path = os.path.join(ctx['storage_path'], f'FOV{fov:02d}', 'alignment_overlay.png')
            self.preview_canvas.draw_fov_all_readouts_overlay(ctx['storage_path'], fov, ctx['hybe_records'],
                                                      ctx['reference_hybe'], matrices, save_path=save_path,
                                                      channel_type=channel_type)
        self.fov_matrices.update({(ctx['storage_path'], fov): m for fov, m in self._pending_same_modality_alignment.items()})
        self._pending_same_modality_alignment = None
        self._refresh_same_modality_results_list()
        ap.SameModalityAcceptPushButton.setEnabled(False)
        ap.SameModalityRejectPushButton.setEnabled(False)
        QtWidgets.QMessageBox.information(self, 'FOV alignment accepted', 'Matrices written to H5; overlay image(s) saved.')

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
        matrices = (self._pending_same_modality_alignment or {}).get(fov) or self.fov_matrices.get((ctx['storage_path'], fov))
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
        ip = self.ui.IngestionPanel
        storage_path = ip.StoragePathLineEdit.text().strip()
        fov = ap.SameModalityOverlayFovSpinBox.value()
        reference_hybe = ap.ReferenceHybeComboBox.currentText()
        if not storage_path or not reference_hybe or not self.hybe_records:
            QtWidgets.QMessageBox.warning(self, 'Show All-Readouts Overlay',
                                          'Set storage path (Ingestion tab) and reference hybe first.')
            return

        ctx = self._same_modality_context
        matrices = None
        if ctx is not None and ctx.get('storage_path') == storage_path:
            matrices = (self._pending_same_modality_alignment or {}).get(fov)
        if matrices is None:
            matrices = self.fov_matrices.get((storage_path, fov))
        if matrices is None:
            matrices = alignment.read_same_modality_matrices(storage_path, fov, self.hybe_records)

        channel_type = ap.SameModalityChannelTypeComboBox.currentText()
        self.alignment_preview_window.show()
        self.alignment_preview_window.raise_()
        self.preview_canvas.draw_fov_all_readouts_overlay(storage_path, fov, self.hybe_records, reference_hybe, matrices,
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
            H = alignment.read_cross_modal_matrix(dna_storage_path, fov, dna_reference_hybe)
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
            item = QtWidgets.QListWidgetItem(f'FOV{fov:02d} {_matrix_summary("DNA->RNA", H)}{suffix}')
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

        ap.RunCrossModalPushButton.setEnabled(False)
        self.statusBar().showMessage('Running cross-modal alignment...')
        self._cross_modal_worker = CrossModalAlignmentWorker(rna_storage_path, dna_storage_path, [fov], self.fov_matrices,
                                                              rna_reference_hybe, dna_reference_hybe, channel_type, write=False,
                                                              border_trim=border_trim, max_shift=max_shift)
        self._cross_modal_worker.progress.connect(lambda done, total, msg: self.statusBar().showMessage(msg))
        self._cross_modal_worker.finished_ok.connect(
            lambda results: self._on_cross_modal_finished(results, rna_storage_path, dna_storage_path,
                                                           rna_reference_hybe, dna_reference_hybe, channel_type))
        self._cross_modal_worker.failed.connect(self._on_cross_modal_failed)
        self._cross_modal_worker.start()

    def _on_cross_modal_finished(self, results, rna_storage_path, dna_storage_path, rna_reference_hybe, dna_reference_hybe, channel_type):
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
                                             rna_fov_matrices=self.fov_matrices.get((rna_storage_path, last_fov), {}),
                                             dna_fov_matrices=self.fov_matrices.get((dna_storage_path, last_fov), {}))

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

        ap.RunAllCrossModalPushButton.setEnabled(False)
        self.statusBar().showMessage('Running cross-modal alignment for all FOVs...')
        self._cross_modal_worker = CrossModalAlignmentWorker(rna_storage_path, dna_storage_path, fov_list, self.fov_matrices,
                                                              rna_reference_hybe, dna_reference_hybe, channel_type, write=True,
                                                              border_trim=border_trim, max_shift=max_shift)
        self._cross_modal_worker.progress.connect(lambda done, total, msg: self.statusBar().showMessage(msg))
        self._cross_modal_worker.finished_ok.connect(
            lambda results: self._on_cross_modal_all_finished(results, rna_storage_path, dna_storage_path,
                                                               rna_reference_hybe, dna_reference_hybe, channel_type))
        self._cross_modal_worker.failed.connect(self._on_cross_modal_all_failed)
        self._cross_modal_worker.start()

    def _on_cross_modal_all_finished(self, results, rna_storage_path, dna_storage_path, rna_reference_hybe, dna_reference_hybe, channel_type):
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
            save_path = os.path.join(dna_storage_path, f'FOV{fov:02d}', 'cross_modal_overlay.png')
            self.preview_canvas.draw_cross_modal_preview(rna_storage_path, dna_storage_path, fov,
                                                 rna_reference_hybe, dna_reference_hybe, channel_type, H, save_path=save_path,
                                                 rna_fov_matrices=self.fov_matrices.get((rna_storage_path, fov), {}),
                                                 dna_fov_matrices=self.fov_matrices.get((dna_storage_path, fov), {}))
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
            alignment.write_cross_modal_matrix(ctx['dna_storage_path'], fov, H, ctx['rna_reference_hybe'],
                                               ctx['dna_reference_hybe'], ctx['channel_type'])
            save_path = os.path.join(ctx['dna_storage_path'], f'FOV{fov:02d}', 'cross_modal_overlay.png')
            self.preview_canvas.draw_cross_modal_preview(ctx['rna_storage_path'], ctx['dna_storage_path'], fov,
                                                 ctx['rna_reference_hybe'], ctx['dna_reference_hybe'], ctx['channel_type'],
                                                 H, save_path=save_path,
                                                 rna_fov_matrices=self.fov_matrices.get((ctx['rna_storage_path'], fov), {}),
                                                 dna_fov_matrices=self.fov_matrices.get((ctx['dna_storage_path'], fov), {}))
            self._mirror_cross_modal_params_to_vlinks(ctx['rna_storage_path'], ctx['dna_storage_path'], fov, H,
                                                      ctx['rna_reference_hybe'], ctx['dna_reference_hybe'], ctx['channel_type'])
        self.cross_modal_result.update({(ctx['dna_storage_path'], fov): H for fov, H in self._pending_cross_modal.items()})
        self._pending_cross_modal = None
        self._refresh_cross_modal_results_list()
        ap.CrossModalAcceptPushButton.setEnabled(False)
        ap.CrossModalRejectPushButton.setEnabled(False)
        QtWidgets.QMessageBox.information(self, 'Cross-modal alignment accepted', 'Result written to H5; overlay image(s) saved.')

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
        accepted) result, the in-memory cross_modal_result cache,
        then a direct disk read (read_cross_modal_matrix) as a last
        resort -- same "never require re-computation just to look" pattern
        used by the within-experiment/cell overlay viewers.
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
            H = alignment.read_cross_modal_matrix(dna_storage_path, fov, dna_reference_hybe)
        if H is None:
            QtWidgets.QMessageBox.warning(self, 'Show Cross-Modal Overlay',
                                          f'No cross-modal result for FOV{fov:02d} yet -- run alignment first.')
            return

        self.alignment_preview_window.show()
        self.alignment_preview_window.raise_()
        self.preview_canvas.draw_cross_modal_preview(rna_storage_path, dna_storage_path, fov,
                                             rna_reference_hybe, dna_reference_hybe, channel_type, H,
                                             rna_fov_matrices=self.fov_matrices.get((rna_storage_path, fov), {}),
                                             dna_fov_matrices=self.fov_matrices.get((dna_storage_path, fov), {}))

    # -- cell-based alignment --

    def _modality_for_storage_path(self, storage_path):
        """
        Reverse-lookup: which configured modality name owns this storage
        path, or None if it doesn't match any. Used wherever cell-level
        alignment matrices need to be tagged/read by (hybe, modality) key
        but only a storage_path is in scope -- cell.modality is NOT a
        substitute for this (a cell's own segmentation modality doesn't
        tell you which modality a given storage_path/hybe_records
        argument belongs to; those can be the SAME cell's own modality or
        the OTHER one, depending on which call site this is).
        """
        for name, data in self.modality_data.items():
            if data.get('storage_path') == storage_path:
                return name
        return None

    def _composed_fov_matrices_for_cell_alignment(self, storage_path, fov):
        """
        The FOV-level matrices to use as compute_cell_alignment's "H1"
        input for this (storage_path, fov). If an accepted cross-modal
        result exists here (storage_path is a DNA modality already linked
        to some RNA experiment), every hybe's matrix gets H_across composed
        on top via compose_chain, so DNA cell-alignment builds on the
        cross-modal correction too, not just the within-experiment one --
        RNA's H_across is always identity by design, so this is a no-op for
        RNA; DNA before any cross-modal run also falls through unchanged
        (nothing to compose in yet). Returns a NEW dict -- never mutates
        self.fov_matrices, so /matrix/{hybe}'s on-disk meaning stays
        strictly within-experiment for any other reader.
        """
        within = self.fov_matrices.get((storage_path, fov), {})
        H_across = self.cross_modal_result.get((storage_path, fov))
        if H_across is None:
            return within
        return {hybe: alignment.compose_chain([H, H_across]) for hybe, H in within.items()}

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
        own within-experiment matrices, composed with the cross-modal
        correction (or its inverse, if storage_path is the DNA side and
        H_across therefore needs to be undone to land in DNA's own frame)
        so it lands directly in storage_path's (the cell's) own frame --
        ready to hand to compute_cell_alignment as a second, independent
        fov_matrices input, exactly like
        _composed_fov_matrices_for_cell_alignment's own output for the
        same-modality case. Previously excluded any hybe name already
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
            dna_reference_hybe = ap.DnaReferenceHybeComboBox.currentText().strip()
            invert = False
        elif storage_path == dna_storage_path:
            other_storage_path = rna_storage_path
            dna_reference_hybe = ap.DnaReferenceHybeComboBox.currentText().strip()
            invert = True
        else:
            return None
        if not dna_reference_hybe:
            return None

        H_across = self.cross_modal_result.get((dna_storage_path, fov))
        if H_across is None:
            H_across = alignment.read_cross_modal_matrix(dna_storage_path, fov, dna_reference_hybe)
        if H_across is None:
            return None
        H_compose = la.inv(H_across) if invert else H_across

        other_modality = self._modality_for_storage_path(other_storage_path)
        other_data = self.modality_data.get(other_modality) if other_modality else None
        if not other_data or not other_data['layout_path']:
            return None
        other_reference_hybe = other_data['reference_hybe']
        if not other_reference_hybe:
            return None
        try:
            other_hybe_records = preprocess.parse_experiment_layout(other_data['layout_path'])
        except Exception:
            return None

        other_within = self.fov_matrices.get((other_storage_path, fov), {})
        if not other_within:
            return None

        other_fov_matrices = {hybe: alignment.compose_chain([H, H_compose]) for hybe, H in other_within.items()}
        if not other_fov_matrices:
            return None
        return other_storage_path, other_hybe_records, other_fov_matrices, other_reference_hybe, other_modality

    def _fov_matrices_for_cell_modality(self, modality, cell, fov):
        """
        The already-composed {hybe: H_to_shared} dict for `modality`,
        resolved relative to `cell`'s own storage path -- same modality as
        the cell -> its own within-experiment matrices; any other
        (currently configured) modality -> the cross-modal-composed
        matrices for that other modality. Returns None if that FOV-level
        layer isn't available at all yet (FOV alignment or cross-modal
        alignment for that pairing hasn't been run/accepted). Shared by
        _fov_only_matrix_for_hybe so every caller sources this lookup the
        same way, rather than re-deriving it.
        """
        cell_storage_path = self.modality_data.get(cell.modality, {}).get('storage_path')
        if not cell_storage_path:
            return None
        if modality == cell.modality:
            return self._composed_fov_matrices_for_cell_alignment(cell_storage_path, fov)
        other = self._other_modality_cell_alignment_inputs(cell_storage_path, fov)
        if other is None:
            return None
        _, _, fov_matrices_for_hybe, _, other_modality = other
        if other_modality != modality:
            return None
        return fov_matrices_for_hybe

    def _fov_only_matrix_for_hybe(self, hybe, modality, cell, fov):
        """
        hybe's own FOV/cross-modal-level matrix (no cell-level residual,
        even if a real one now exists) into cell.reference_hybe's own
        frame -- used specifically where the pre-cell-alignment state is
        wanted on purpose (the overlay/preview's 'FOV/cross-modal' column,
        a deliberate "before" comparison against 'final'), and as
        _matrix_to_cellref's fallback when real cell-level data isn't
        (fully) available. hybe/modality can belong to EITHER of the two
        configured modalities; cell may
        belong to either too (cell_container_permanent isn't filtered by
        modality). Returns None if the needed FOV-level layer isn't
        available at all yet for hybe's own leg -- there's no correction
        to fall back to in that case, a genuinely different situation
        from "cell-level just wasn't run" (cell.reference_hybe's own leg
        stays lenient, defaulting to identity, matching the "before" of
        an unaligned experiment rather than failing the whole lookup).
        """
        fov_matrices_for_hybe = self._fov_matrices_for_cell_modality(modality, cell, fov)
        if fov_matrices_for_hybe is None or hybe not in fov_matrices_for_hybe:
            return None
        cell_fov_matrices = self._fov_matrices_for_cell_modality(cell.modality, cell, fov)
        cell_reference_hybe_matrix = (cell_fov_matrices or {}).get(cell.reference_hybe, np.eye(3))
        return alignment.hybe_to_cellref_matrix(fov_matrices_for_hybe, cell_reference_hybe_matrix, hybe)

    def _matrix_to_cellref(self, hybe, modality, cell, fov):
        """
        hybe's own raw frame -> cell.reference_hybe's own frame -- the
        "best available" transform get_area_in_readout-style consumers
        need. cell.matrices entries now target THAT compute_cell_
        alignment call's own reference_hybe, not a shared frame (see its
        docstring) -- a same-modality entry and a cross-modal entry can
        rest in genuinely different frames, bridged only via
        cell.matrix_anchors (see ACell.matrix_between). So this prefers
        cell.matrix_to (the real, cell-level-refined composition) only
        when BOTH hybe's own and cell.reference_hybe's own entries AND
        both modalities' anchors are actually present -- mixing a real
        entry for one leg with a raw FOV value for the other would
        silently combine two different frames, exactly the class of bug
        this whole redesign exists to avoid. Otherwise falls back to
        _fov_only_matrix_for_hybe's pure FOV-level computation (no
        cell.matrices/matrix_anchors dependency at all), so a cell that's
        never had cell-level alignment run for a given hybe/modality
        still gets a real, non-identity answer instead of ACell.matrix_
        to's bare identity default. Returns None only when even the
        FOV-only fallback has nothing to work with.
        """
        key = (hybe, modality)
        self_key = (cell.reference_hybe, cell.modality)
        have_real = (key in cell.matrices and self_key in cell.matrices
                     and modality in cell.matrix_anchors and cell.modality in cell.matrix_anchors)
        if have_real:
            return cell.matrix_to(hybe, modality)
        return self._fov_only_matrix_for_hybe(hybe, modality, cell, fov)

    def _cell_overlay_target_specs(self, cell, storage_path, fov, hybe_records, channel_type):
        """
        Resolves every hybe in cell.matrices (both modalities) into what
        draw_cell_all_readouts_overlay needs to read/crop/warp it: storage
        path, channel, fiducial channel, the FOV-level matrix (the
        'FOV/cross-modal' stage), and this cell's own final yx/zx
        matrices -- both resolved into cell.reference_hybe's own frame
        (via _fov_only_matrix_for_hybe / _matrix_to_cellref respectively),
        which is what draw_cell_all_readouts_overlay's crop_via expects,
        not a direct cell.matrices lookup (that targets the shared FOV
        frame instead -- see compute_cell_alignment's docstring). Unlike
        the single-hybe preview, every target here is warped into ONE
        fixed shared coordinate FRAME (cell.reference_hybe, always read at
        H=eye since cell.area is already native to it) -- that part has no
        per-target ambiguity.

        Excludes hybes with NO cell.matrix_provenance entry -- these are
        the cell-alignment ANCHOR(s) (same-modality and/or other-modality,
        see compute_cell_alignment: a hybe skips provenance only when it
        IS that call's own reference_hybe param), which never got a
        residual computed against anything, so there's no meaningful raw/
        FOV/final comparison to show for them. Deliberately NOT "hybe ==
        cell.reference_hybe" (the segmentation hybe) -- those are two
        different concepts that only coincide when the user happens to
        pick the segmentation hybe as the alignment anchor too. When they
        don't coincide (e.g. segmented on Hyb_500, aligned against
        Hyb_101), cell.reference_hybe legitimately HAS its own provenance
        entry and real matrices, and belongs in the target list like any
        other hybe -- excluding it by the old, wrong criterion silently
        dropped its own before/after comparison from the overlay.
        """
        this_modality = self._modality_for_storage_path(storage_path)
        same_record_by_folder = {r['folder']: r for r in hybe_records}
        other = self._other_modality_cell_alignment_inputs(storage_path, fov)
        other_record_by_folder, other_storage_path, other_modality = {}, None, None
        if other is not None:
            other_storage_path, other_hybe_records, _, _, other_modality = other
            other_record_by_folder = {r['folder']: r for r in other_hybe_records}

        specs = []
        for (hybe, modality), mats in cell.matrices.items():
            # modality (not bare-name dict membership) resolves which
            # side each entry belongs to -- necessary now that the same
            # hybe NAME can legitimately appear once per modality (the
            # cross-modal bridge hybe, e.g. Hyb_130, is a real, distinct
            # file in both).
            if (hybe, modality) not in cell.matrix_provenance:
                continue
            if modality == this_modality and hybe in same_record_by_folder:
                record = same_record_by_folder[hybe]
                target_storage_path = storage_path
            elif modality == other_modality and hybe in other_record_by_folder:
                record = other_record_by_folder[hybe]
                target_storage_path = other_storage_path
            else:
                continue
            fov_only_matrix = self._fov_only_matrix_for_hybe(hybe, modality, cell, fov)
            if fov_only_matrix is None:
                continue
            final_matrix = self._matrix_to_cellref(hybe, modality, cell, fov)
            if final_matrix is None:
                continue
            specs.append({
                'hybe': hybe, 'modality': modality, 'storage_path': target_storage_path,
                'channel': alignment.pick_channel_by_type(record, channel_type),
                'fiducial_channel': record['fiducial_channel'],
                'fov_only_matrix': fov_only_matrix,
                'final_matrix': final_matrix,
                'zx_matrix': mats.get('zx', np.eye(3)),
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
        storage_path = ip.StoragePathLineEdit.text().strip()
        fov_list_all = self._parse_fov_list(ip.FovListLineEdit.text())
        if not storage_path or not fov_list_all:
            QtWidgets.QMessageBox.warning(self, 'Run Cell Alignment', 'Set storage path and FOV list in the Ingestion tab first.')
            return

        fov = ap.CellFovSpinBox.value()
        if fov not in fov_list_all:
            QtWidgets.QMessageBox.warning(self, 'Run Cell Alignment',
                                          f'FOV{fov} is not in the Ingestion tab\'s FOV list.')
            return
        if (storage_path, fov) not in self.fov_matrices:
            QtWidgets.QMessageBox.warning(self, 'Run Cell Alignment', 'Run (and accept) FOV alignment for this FOV first.')
            return

        # Cells here are the REAL objects (whichever container has them for
        # this FOV, permanent/saved preferred) -- CellAlignmentWorker mutates
        # them in place, so no staging/deepcopy is needed for an always-save
        # operation.
        container = None
        if self.cell_container_permanent is not None and self.cell_container_permanent.data.get(fov):
            container = self.cell_container_permanent
        elif self.cell_container is not None and self.cell_container.data.get(fov):
            container = self.cell_container
        if container is None:
            QtWidgets.QMessageBox.warning(self, 'Run Cell Alignment',
                                          'No cells segmented for this FOV yet -- run Cell Segmentation first.')
            return
        real_cells = container.data[fov]

        cell_reference_hybe = ap.CellReferenceHybeComboBox.currentText() or None
        channel_type = ap.CellChannelTypeComboBox.currentText()
        pad = ap.CellPadSpinBox.value()

        ap.RunCellAlignmentPushButton.setEnabled(False)
        self.statusBar().showMessage('Computing cell alignment...')
        worker_jobs = [(fov, real_cells, self._composed_fov_matrices_for_cell_alignment(storage_path, fov),
                        self._other_modality_cell_alignment_inputs(storage_path, fov))]
        self._cell_alignment_worker = CellAlignmentWorker(worker_jobs, storage_path, self.hybe_records,
                                                           self._modality_for_storage_path(storage_path),
                                                           reference_hybe=cell_reference_hybe, channel_type=channel_type,
                                                           pad=pad)
        self._cell_alignment_worker.progress.connect(lambda done, total, msg: self.statusBar().showMessage(msg))
        self._cell_alignment_worker.finished_ok.connect(
            lambda results: self._on_cell_alignment_finished(results, fov, container, storage_path,
                                                              cell_reference_hybe, channel_type, pad))
        self._cell_alignment_worker.failed.connect(self._on_cell_alignment_failed)
        self._cell_alignment_worker.start()

    def _on_cell_alignment_finished(self, results, fov, container, storage_path, cell_reference_hybe, channel_type, pad):
        """results: [(fov, cells)] -- cells are the real objects, mutated in
        place by compute_cell_alignment. Always writes to vlinks.h5
        immediately -- no staging/Accept step."""
        ap = self.ui.AlignmentPanel
        ap.RunCellAlignmentPushButton.setEnabled(True)
        self.statusBar().showMessage('Cell alignment computed.', 5000)

        storage_paths = self._all_vlinks_storage_paths()
        for path in storage_paths:
            vlinks_store.write_global_params(path, cell_alignment_reference_hybe=cell_reference_hybe,
                                             cell_alignment_channel_type=channel_type, cell_alignment_pad=pad)
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
                    if self._save_cell_overlay(cell, fov, storage_path, channel_type, pad):
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

    def _save_cell_overlay(self, cell, fov, storage_path, channel_type, pad):
        """Draws + saves one cell's all-readouts overlay PNG. Returns False
        (no-op) if this cell's own segmentation hybe record can't be
        resolved from the current hybe list, matching the automatic-mode
        skip that already existed before this was factored out."""
        record_by_folder = {r['folder']: r for r in self.hybe_records}
        reference_record = record_by_folder.get(cell.reference_hybe)
        if reference_record is None:
            return False
        save_path = os.path.join(storage_path, f'FOV{fov:02d}', f'cell{cell.id}_alignment_overlay.png')
        reference_channel = alignment.pick_channel_by_type(reference_record, channel_type)
        target_specs = self._cell_overlay_target_specs(cell, storage_path, fov, self.hybe_records, channel_type)
        self.preview_canvas.draw_cell_all_readouts_overlay(
            cell, fov, cell.reference_hybe, storage_path, reference_channel,
            reference_record['fiducial_channel'], target_specs, pad=pad, save_path=save_path)
        return True

    def _save_all_cell_overlays(self):
        """On-demand batch save of every currently-computed cell's overlay
        PNG (self._cell_alignment_display_cells, populated by the last Run
        Cell Alignment call, manual or automatic) -- lets a user skim the
        whole run's alignment quality by eye without needing every cell to
        have tripped the auto-save-on-large-shift threshold."""
        ip = self.ui.IngestionPanel
        storage_path = ip.StoragePathLineEdit.text().strip()
        if not storage_path or not self._cell_alignment_display_cells:
            QtWidgets.QMessageBox.warning(self, 'Save All Cell Overlays', 'Run Cell Alignment first.')
            return
        ap = self.ui.AlignmentPanel
        channel_type = ap.CellChannelTypeComboBox.currentText()
        pad = ap.CellPadSpinBox.value()
        n_saved = 0
        for fov, cell in self._cell_alignment_display_cells:
            if self._save_cell_overlay(cell, fov, storage_path, channel_type, pad):
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
        single cell, and the reference is tier 1's own Reference hybe
        (both already resolved into _cell_per_hybe_context by
        _refresh_cell_per_hybe_results) -- the row itself IS the target-
        hybe choice, so clicking it previews that hybe directly against
        that reference, no separate target combo needed.
        """
        pctx = self._cell_per_hybe_context
        if pctx is None:
            return
        fov, cell_id, hybe, modality = item.data(QtCore.Qt.UserRole)
        cell = pctx['cell']
        target_key = (hybe, modality)
        reference_key = pctx['reference_key']
        self._cell_preview_context = {'fov': fov, 'cell': cell, 'storage_path': pctx['storage_path'],
                                      'hybe_records': pctx['hybe_records'], 'target_key': target_key,
                                      'reference_key': reference_key}
        self._show_cell_alignment_preview_for_hybe(target_key=target_key, reference_key=reference_key)

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
        other = self._other_modality_cell_alignment_inputs(storage_path, fov)
        if other is None:
            return None, None, (f"no cross-modal alignment result found for FOV{fov:02d} yet "
                                f"(run and accept Cross-Modality Alignment first).")
        other_storage_path, other_hybe_records, _, _, other_modality = other
        if modality != other_modality:
            return None, None, f"{modality} isn't configured as either Cross-Modality Alignment path right now."
        other_record_by_folder = {r['folder']: r for r in other_hybe_records}
        if hybe not in other_record_by_folder:
            return None, None, f"{hybe} ({modality}) isn't in that modality's parsed layout."
        return other_record_by_folder[hybe], other_storage_path, None

    def _show_cell_alignment_preview_for_hybe(self, target_key=None, reference_key=None):
        """
        target_key/reference_key: (hybe, modality) tuples -- matches
        cell.matrices' own key shape. Both default to whatever's stored
        in _cell_preview_context (set by _show_cell_alignment_preview
        from the last-clicked Results-list row and tier 1's own Reference
        hybe combo respectively) -- neither needs a separate combo of its
        own here. Target and reference don't need to share a modality --
        any hybe can be compared against any other hybe directly, RNA
        against DNA included (see ACell.matrix_between).

        This works because draw_cell_alignment_preview_3col only ever
        needs target's and reference's OWN matrices independently, both
        expressed relative to cell.reference_hybe's frame (each crops
        cell.area via its own inverse-warp, then the two crops are
        composited for visual comparison) -- it never needs a direct
        target-vs-reference matrix. So both are resolved the exact same
        way, via _fov_only_matrix_for_hybe/_matrix_to_cellref (same
        resolvers _cell_overlay_target_specs uses), completely
        independently of one another.
        """
        pctx = getattr(self, '_cell_preview_context', None)
        if pctx is None:
            return
        ap = self.ui.AlignmentPanel
        if target_key is None:
            target_key = pctx.get('target_key')
        if reference_key is None:
            reference_key = pctx.get('reference_key')
        if not target_key or not reference_key:
            return
        target_hybe, target_modality = target_key
        reference_hybe, reference_modality = reference_key
        cell, fov = pctx['cell'], pctx['fov']
        storage_path, hybe_records = pctx['storage_path'], pctx['hybe_records']
        channel_type = ap.CellChannelTypeComboBox.currentText()
        pad = ap.CellPadSpinBox.value()

        target_record, target_storage_path, err = self._resolve_preview_hybe_context(
            target_hybe, target_modality, storage_path, hybe_records, fov)
        if err:
            self.statusBar().showMessage(f"Can't preview {target_hybe} ({target_modality}): {err}", 8000)
            return
        reference_record, reference_storage_path, err = self._resolve_preview_hybe_context(
            reference_hybe, reference_modality, storage_path, hybe_records, fov)
        if err:
            self.statusBar().showMessage(f"Can't use {reference_hybe} ({reference_modality}) as reference: {err}", 8000)
            return

        # fov_only_matrix/reference_fov_matrix/final_matrix all need to be
        # expressed in cell.reference_hybe's own frame (what draw_cell_
        # alignment_preview_3col's bounds_via expects) -- neither a raw
        # fov_matrices lookup nor a raw cell.matrices lookup gives that
        # directly (cell.matrices now targets each call's own
        # reference_hybe, never a shared frame -- see compute_cell_
        # alignment's docstring), so both go through the same resolvers
        # _cell_overlay_target_specs uses.
        fov_only_matrix = self._fov_only_matrix_for_hybe(target_hybe, target_modality, cell, fov)
        reference_fov_matrix = self._fov_only_matrix_for_hybe(reference_hybe, reference_modality, cell, fov)
        final_matrix = self._matrix_to_cellref(target_hybe, target_modality, cell, fov)
        # Computed INDEPENDENTLY from reference_fov_matrix, not reused --
        # see draw_cell_alignment_preview_3col's own docstring for why
        # reference_hybe's own final column needs the SAME KIND of matrix
        # (_matrix_to_cellref, folding in cell.reference_hybe's own real
        # residual) the target's final column uses, not the residual-
        # blind FOV-only one.
        reference_final_matrix = self._matrix_to_cellref(reference_hybe, reference_modality, cell, fov)
        if fov_only_matrix is None or reference_fov_matrix is None or final_matrix is None or reference_final_matrix is None:
            self.statusBar().showMessage(
                f"Can't preview {target_hybe} ({target_modality}): no FOV-level alignment available for it "
                "or its reference hybe yet.", 8000)
            return

        reference_channel = alignment.pick_channel_by_type(reference_record, channel_type)
        target_channel = alignment.pick_channel_by_type(target_record, channel_type)
        reference_fiducial_channel = reference_record['fiducial_channel']
        target_fiducial_channel = target_record['fiducial_channel']

        self.alignment_preview_window.show()
        self.alignment_preview_window.raise_()
        self.preview_canvas.draw_cell_alignment_preview_3col(
            cell, fov, reference_storage_path, reference_hybe, reference_channel, reference_fiducial_channel,
            reference_fov_matrix,
            target_storage_path, target_hybe, target_channel, target_fiducial_channel,
            fov_only_matrix, final_matrix, pad=pad, target_modality=target_modality,
            reference_final_matrix=reference_final_matrix)

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
            cell = next((c for c in self.cell_container_permanent.data.get(fov, []) if c.id == cell_id), None)
        if cell is None and self.cell_container is not None:
            cell = next((c for c in self.cell_container.data.get(fov, []) if c.id == cell_id), None)
        if cell is None:
            return
        channel_type = ap.CellChannelTypeComboBox.currentText()
        pad = ap.CellPadSpinBox.value()

        reference_key = ap.CellPreviewReferenceHybeComboBox.currentData()
        reference_hybe, reference_modality = reference_key if reference_key else (cell.reference_hybe, cell.modality)
        reference_record, reference_storage_path, err = self._resolve_preview_hybe_context(
            reference_hybe, reference_modality, storage_path, self.hybe_records, fov)
        if err:
            QtWidgets.QMessageBox.warning(self, 'Show All-Readouts Overlay',
                                          f"Can't use {reference_hybe} ({reference_modality}) as reference: {err}")
            return
        reference_channel = alignment.pick_channel_by_type(reference_record, channel_type)
        # FOV-only for the FOV/cross-modal column's own crop.
        reference_matrix = self._fov_only_matrix_for_hybe(reference_hybe, reference_modality, cell, fov)
        if reference_matrix is None:
            reference_matrix = np.eye(3)
        # Computed INDEPENDENTLY for the final column -- see draw_cell_
        # all_readouts_overlay's own docstring for why reference_hybe's
        # final crop needs the SAME KIND of matrix (_matrix_to_cellref,
        # folding in cell.reference_hybe's own real residual) the
        # target's own final column uses, not the residual-blind FOV-only
        # one reused.
        reference_final_matrix = self._matrix_to_cellref(reference_hybe, reference_modality, cell, fov)
        if reference_final_matrix is None:
            reference_final_matrix = np.eye(3)
        target_specs = self._cell_overlay_target_specs(cell, storage_path, fov, self.hybe_records, channel_type)
        self.alignment_preview_window.show()
        self.alignment_preview_window.raise_()
        self.preview_canvas.draw_cell_all_readouts_overlay(
            cell, fov, reference_hybe, reference_storage_path, reference_channel, reference_record['fiducial_channel'],
            target_specs, pad=pad, reference_matrix=reference_matrix, reference_final_matrix=reference_final_matrix)

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
        ip = self.ui.IngestionPanel
        storage_path = ip.StoragePathLineEdit.text().strip()
        fov = ap.CellFovSpinBox.value()
        cell_id = ap.CellIdSpinBox.value()
        if not storage_path or not self.hybe_records:
            QtWidgets.QMessageBox.warning(self, 'Preview This Cell',
                                          'Set storage path (Ingestion tab) and parse a layout first.')
            return
        real_cell = None
        if self.cell_container_permanent is not None:
            real_cell = next((c for c in self.cell_container_permanent.data.get(fov, []) if c.id == cell_id), None)
        if real_cell is None and self.cell_container is not None:
            real_cell = next((c for c in self.cell_container.data.get(fov, []) if c.id == cell_id), None)
        if real_cell is None:
            QtWidgets.QMessageBox.warning(self, 'Preview This Cell',
                                          f'No segmented cell with ID {cell_id} found in FOV{fov:02d}.')
            return
        if (storage_path, fov) not in self.fov_matrices:
            QtWidgets.QMessageBox.warning(self, 'Preview This Cell', 'Run (and accept) FOV alignment for this FOV first.')
            return

        cell_reference_hybe = ap.CellReferenceHybeComboBox.currentText() or None
        channel_type = ap.CellChannelTypeComboBox.currentText()
        pad = ap.CellPadSpinBox.value()

        fov_matrices = self._composed_fov_matrices_for_cell_alignment(storage_path, fov)
        other_ctx = self._other_modality_cell_alignment_inputs(storage_path, fov)
        staged_cell = deepcopy(real_cell)

        worker = CellAlignmentWorker([(fov, [staged_cell], fov_matrices, other_ctx)], storage_path, self.hybe_records,
                                     self._modality_for_storage_path(storage_path), reference_hybe=cell_reference_hybe,
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
        self._pending_per_cell_alignment_params = {'reference_hybe': cell_reference_hybe,
                                                    'channel_type': channel_type, 'pad': pad}
        ap.PerCellAcceptPushButton.setEnabled(True)
        ap.PerCellRejectPushButton.setEnabled(True)

        # Per explicit request: the preview shown here is the ref-hybe-vs-
        # all-hybe overlay anchored at THIS RUN's own reference_hybe (the
        # alignment anchor picked above) -- not cell.reference_hybe (the
        # segmentation hybe, e.g. Hyb_500), a different concept (see
        # _cell_overlay_target_specs' own docstring). draw_cell_all_
        # readouts_overlay's FOV/final columns default to assuming H=eye
        # between the reference hybe and cell.area's own frame, which is
        # only equivalent to the real transform when they're the same
        # hybe -- reference_matrix (FOV-only, via _fov_only_matrix_for_
        # hybe, the same resolver target_specs itself uses) supplies the
        # real one whenever they differ. Tier 3's "Results (per cell, per
        # hybe)" list -- one row per (cell, hybe) -- gives a per-hybe
        # 2-column comparison on demand instead.
        this_modality = self._modality_for_storage_path(storage_path)
        overlay_reference_hybe = cell_reference_hybe or staged_cell.reference_hybe
        record_by_folder = {r['folder']: r for r in self.hybe_records}
        reference_record = record_by_folder.get(overlay_reference_hybe)
        if reference_record is not None:
            reference_channel = alignment.pick_channel_by_type(reference_record, channel_type)
            reference_matrix = self._fov_only_matrix_for_hybe(overlay_reference_hybe, this_modality, staged_cell, fov)
            if reference_matrix is None:
                reference_matrix = np.eye(3)
            # Computed INDEPENDENTLY for the final column -- see draw_cell_
            # all_readouts_overlay's own docstring for why.
            reference_final_matrix = self._matrix_to_cellref(overlay_reference_hybe, this_modality, staged_cell, fov)
            if reference_final_matrix is None:
                reference_final_matrix = np.eye(3)
            target_specs = self._cell_overlay_target_specs(staged_cell, storage_path, fov, self.hybe_records, channel_type)
            self.alignment_preview_window.show()
            self.alignment_preview_window.raise_()
            self.preview_canvas.draw_cell_all_readouts_overlay(
                staged_cell, fov, overlay_reference_hybe, storage_path, reference_channel,
                reference_record['fiducial_channel'], target_specs, pad=pad, reference_matrix=reference_matrix,
                reference_final_matrix=reference_final_matrix)
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
        real_cell.matrices = staged_cell.matrices
        real_cell.matrix_anchors = staged_cell.matrix_anchors
        real_cell.matrix_provenance = staged_cell.matrix_provenance

        ip = self.ui.IngestionPanel
        storage_path = ip.StoragePathLineEdit.text().strip()
        wrote = False
        if storage_path and fov is not None:
            storage_paths = self._all_vlinks_storage_paths()
            container = None
            if self.cell_container_permanent is not None and real_cell in self.cell_container_permanent.data.get(fov, []):
                container = self.cell_container_permanent
            elif self.cell_container is not None and real_cell in self.cell_container.data.get(fov, []):
                container = self.cell_container
            if storage_paths and container is not None:
                vlinks_store.mirror_write_cells(storage_paths, fov, container)
                wrote = True
            reference_record = {r['folder']: r for r in self.hybe_records}.get(real_cell.reference_hybe)
            if reference_record is not None:
                save_path = os.path.join(storage_path, f'FOV{fov:02d}', f'cell{real_cell.id}_alignment_overlay.png')
                reference_channel = alignment.pick_channel_by_type(reference_record, channel_type)
                target_specs = self._cell_overlay_target_specs(real_cell, storage_path, fov, self.hybe_records, channel_type)
                self.preview_canvas.draw_cell_all_readouts_overlay(
                    real_cell, fov, real_cell.reference_hybe, storage_path, reference_channel,
                    reference_record['fiducial_channel'], target_specs, pad=pad, save_path=save_path)

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

        if glob.get('cell_align_reference_hybe') and not ap.CellReferenceHybeComboBox.currentText():
            # Only a fallback for when vlinks (already applied by
            # _activate_modalities -> _refresh_params_from_vlinks, which
            # runs before this line) had nothing real to say -- vlinks-
            # actual values must always win over this stale config
            # default, never get overwritten by it after the fact.
            idx = ap.CellReferenceHybeComboBox.findText(glob['cell_align_reference_hybe'])
            if idx >= 0:
                ap.CellReferenceHybeComboBox.setCurrentIndex(idx)

    def _save_config_dialog(self):
        """
        Builds the modality-nested config from whatever this session's UI
        currently has -- the Ingestion tab's active modality becomes one
        <modality> entry (with its full layout_path/dax_directory/
        storage_path/reference_hybe/same_modality_channel_type), and the cross-
        modal section's OTHER storage path (if filled in) becomes a
        second <modality> entry (storage_path + cross_modality_reference_
        hybe only -- this app's current UI has no live layout_path/
        dax_directory/reference_hybe/same_modality_channel_type for a modality
        that isn't the Ingestion tab's active one). A future N-modality UI
        would just add more entries to the same 'modalities' dict; the
        file format itself already supports it.
        """
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save configuration file', self.save_path, 'configuration file (*.xml)')
        if not path:
            return
        ip, ap, cp = self.ui.IngestionPanel, self.ui.AlignmentPanel, self.ui.CellSegmentPanel
        if self.current_modality is not None:
            self._save_current_modality_fields()  # flush whatever's live in the fields right now

        cross_hybe_fields = {'RNA': ap.RnaReferenceHybeComboBox, 'DNA': ap.DnaReferenceHybeComboBox}
        modalities = {}
        for name in self.modality_names:
            state = self.modality_data.get(name, self._blank_modality_state())
            if not state['layout_path'] and not state['storage_path']:
                continue  # never configured -- omit rather than writing an empty placeholder
            entry = dict(state)
            if name in cross_hybe_fields:
                entry['cross_modality_reference_hybe'] = cross_hybe_fields[name].currentText().strip()
            modalities[name] = entry

        cfg = {
            'global': {
                'num_modalities': len(self.modality_names),
                'fov_list': self._parse_fov_list(ip.FovListLineEdit.text()),
                'cross_modality_channel_type': ap.ChannelTypeComboBox.currentText(),
                'cell_align_reference_hybe': ap.CellReferenceHybeComboBox.currentText(),
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
