import contextlib
import io
import os
import re
import sys
import traceback
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
                status = 'OK' if err is None else f'ERROR: {err}'
                if err is not None:
                    error_lines.append(f'FOV{fov_r:02d} {hybe_r}: {err}')
                self.progress.emit(i + 1, len(tasks), f'FOV{fov_r:02d} {hybe_r}: {status}')
            # dax_vlinks_h5 (a single aggregate vlinks.h5 across every hybe)
            # is deliberately NOT called here -- it's only ever read by the
            # legacy Jupyter-widget classes in segmentation/segment.py,
            # alignment/chain.py, localization/localization.py
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
    other_reference_hybe) or None -- see
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

    def __init__(self, jobs, storage_path, hybe_records, reference_hybe=None, channel_type='readout', pad=10):
        super().__init__()
        self.jobs = jobs
        self.storage_path = storage_path
        self.hybe_records = hybe_records
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
                    other_storage_path, other_records_full, other_fov_matrices, other_reference_hybe = other_ctx
                    other_hybe_records = [r for r in other_records_full if r['folder'] in other_fov_matrices]
                else:
                    other_storage_path = other_hybe_records = other_fov_matrices = other_reference_hybe = None
                for cell in cells:
                    alignment.compute_cell_alignment(cell, self.storage_path, fov, hybe_records, fov_matrices,
                                                     reference_hybe=self.reference_hybe, channel_type=self.channel_type,
                                                     pad=self.pad)
                    if other_ctx is not None and other_hybe_records:
                        try:
                            alignment.compute_cell_alignment(cell, other_storage_path, fov, other_hybe_records,
                                                             other_fov_matrices, reference_hybe=other_reference_hybe,
                                                             channel_type=self.channel_type, pad=self.pad)
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
        self.modality_names = ['DNA', 'RNA']
        self.modality_data = {name: self._blank_modality_state() for name in self.modality_names}
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
        self._pending_cell_alignment = None  # list of (real_cell, staged_cell) awaiting Accept/Reject
        self._pending_cell_alignment_fov = None
        self._pending_cell_alignment_params = None  # {'reference_hybe':, 'channel_type':, 'pad':} for the pending run
        self._cell_alignment_context = None
        self._cell_alignment_display_cells = []  # [(fov, cell), ...]
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
        ap.SameModalityResultsListWidget.itemClicked.connect(self._show_same_modality_preview)
        ap.SameModalityShowOverlayPushButton.clicked.connect(self._show_same_modality_all_readouts_overlay)
        ap.SameModalityAcceptPushButton.clicked.connect(self._accept_same_modality_alignment)
        ap.SameModalityRejectPushButton.clicked.connect(self._reject_same_modality_alignment)
        ap.RunCrossModalPushButton.clicked.connect(self._run_cross_modal_alignment)
        ap.CrossModalShowOverlayPushButton.clicked.connect(self._show_cross_modal_overlay)
        ap.CrossModalAcceptPushButton.clicked.connect(self._accept_cross_modal)
        ap.CrossModalRejectPushButton.clicked.connect(self._reject_cross_modal)
        ap.RunCellAlignmentPushButton.clicked.connect(self._run_cell_alignment)
        ap.CellResultsListWidget.itemClicked.connect(self._show_cell_alignment_preview)
        ap.CellPreviewHybeComboBox.currentIndexChanged.connect(lambda _: self._show_cell_alignment_preview_for_hybe())
        ap.CellPadSpinBox.valueChanged.connect(lambda _: self._show_cell_alignment_preview_for_hybe())
        ap.CellOverlayFovLineEdit.editingFinished.connect(self._refresh_cell_overlay_list)
        ap.CellShowOverlayPushButton.clicked.connect(self._show_cell_all_readouts_overlay)
        ap.CellAcceptPushButton.clicked.connect(self._accept_cell_alignment)
        ap.CellRejectPushButton.clicked.connect(self._reject_cell_alignment)

        sp.RefreshCellListPushButton.clicked.connect(self._refresh_spot_cell_list)
        sp.CellListWidget.itemClicked.connect(self._load_spot_crop_for_display)
        sp.HybeComboBox.currentIndexChanged.connect(self._load_spot_crop_for_display)
        sp.ChannelComboBox.currentIndexChanged.connect(self._load_spot_crop_for_display)
        sp.AutoDetectPushButton.clicked.connect(self._run_spot_auto_detect)
        sp.ShowDisplayerPushButton.toggled.connect(self._toggle_spot_crop_displayer)

        ctp.SetFovRangesPushButton.clicked.connect(self._set_celltype_fov_ranges)
        ctp.AssignBarcodeChannelPushButton.clicked.connect(self._assign_barcode_channel)
        ctp.ApplyCalibrationPushButton.clicked.connect(self._apply_barcode_calibration)
        ctp.ShowBarcodeOverviewPushButton.clicked.connect(self._show_barcode_overview)
        ctp.RunCelltypeDeterminationPushButton.clicked.connect(self._run_celltype_determination)
        ctp.ShowCelltypeResultPushButton.clicked.connect(lambda: self._show_celltype_result())

        self.ui.actionLoad_Config.triggered.connect(self._load_config_dialog)
        self.ui.actionSave_Config.triggered.connect(self._save_config_dialog)

        anp = self.ui.AnalysisPanel
        anp.RunPushButton.clicked.connect(self._run_analysis_code)
        anp.ClearOutputPushButton.clicked.connect(anp.OutputDisplay.clear)
        anp.ClearCodePushButton.clicked.connect(anp.CodeEditor.clear)

    # -- modality setup / switching --

    @staticmethod
    def _blank_modality_state():
        return {'layout_path': '', 'dax_directory': '', 'storage_path': '', 'reference_hybe': '', 'same_modality_channel_type': ''}

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
        self.modality_data[self.current_modality] = {
            'layout_path': ip.LayoutPathLineEdit.text().strip(),
            'dax_directory': ip.DaxDirectoryLineEdit.text().strip(),
            'storage_path': ip.StoragePathLineEdit.text().strip(),
            'reference_hybe': ap.ReferenceHybeComboBox.currentText(),
            'same_modality_channel_type': ap.SameModalityChannelTypeComboBox.currentText(),
        }

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
        self._sync_modality_combo_text(name)
        self._refresh_same_modality_results_from_disk()

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
        self._refresh_cross_modal_hybe_choices()

    def _all_modality_hybe_records(self):
        """
        self.hybe_records (current modality) plus every OTHER configured
        modality's own full parsed hybe list, deduped by folder name --
        for combos that need choices spanning BOTH modalities regardless
        of which one is "current" (e.g. CellReferenceHybeComboBox, now
        that cell-based alignment processes both).
        """
        seen = {r['folder'] for r in self.hybe_records}
        merged = list(self.hybe_records)
        for name, data in self.modality_data.items():
            if name == self.current_modality or not data.get('layout_path'):
                continue
            try:
                records = preprocess.parse_experiment_layout(data['layout_path'])
            except Exception:
                continue
            for r in records:
                if r['folder'] not in seen:
                    seen.add(r['folder'])
                    merged.append(r)
        return merged

    def _refresh_cross_modal_hybe_choices(self):
        """
        The Cross-Modality section's RNA/DNA reference hybe combos need
        BOTH modalities' hybe lists at once -- unlike the single-"current
        modality" combos elsewhere, they aren't populated as a side effect
        of _parse_layout, since _parse_layout only ever runs for whichever
        modality happens to be current. Without this, a modality that was
        never made "current" during the session (the common case right
        after loading a config -- only the first modality in the file
        becomes current) would leave its own combo empty forever, even
        though its layout_path is already known.
        """
        ap = self.ui.AlignmentPanel
        for name, populate in (('RNA', ap.populate_rna_reference_hybe_choices),
                               ('DNA', ap.populate_dna_reference_hybe_choices)):
            data = self.modality_data.get(name)
            if not data or not data['layout_path']:
                continue
            try:
                records = preprocess.parse_experiment_layout(data['layout_path'])
            except Exception:
                continue
            populate(records)

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
        ap = self.ui.AlignmentPanel
        ap.populate_reference_hybe_choices(self.hybe_records)
        ap.populate_cell_reference_hybe_choices(self._all_modality_hybe_records())
        self.ui.CellSegmentPanel.populate_reference_hybe_choices(self.hybe_records)
        self.ui.SpotLocalizationPanel.populate_hybe_choices(self.hybe_records)
        self.ui.CelltypeDeterminationPanel.populate_hybe_choices(self.hybe_records)
        # the cross-modal RNA/DNA reference hybe combos are keyed by literal
        # modality name (that section is still hardcoded to exactly RNA/DNA,
        # see _load_config's docstring) -- only refresh whichever one this
        # parse actually belongs to.
        if self.current_modality == 'RNA':
            ap.populate_rna_reference_hybe_choices(self.hybe_records)
        elif self.current_modality == 'DNA':
            ap.populate_dna_reference_hybe_choices(self.hybe_records)
        ap.populate_overlay_fov_choices(self._parse_fov_list(ip.FovListLineEdit.text()))
        ip.LogTextEdit.append(f'Parsed {len(self.hybe_records)} hybe(s) from {layout_path}')
        self._check_ingestion_status(silent=True)
        self._activate_fov(self.ui.CellSegmentPanel.FovSpinBox.value())
        self._refresh_same_modality_results_from_disk()

    def _refresh_same_modality_results_from_disk(self):
        """
        FOV-alignment matrices already persisted from an earlier run/
        session are real, activation-worthy state -- don't leave the
        Results list empty just because THIS session hasn't (re-)run
        alignment yet. Whenever a layout is (re)parsed (directly, or via
        a modality switch), read back whatever's already on disk for
        every FOV in the FOV list and populate the Results list +
        self.fov_matrices cache exactly as a fresh run would, using the
        panel's own current reference hybe.
        """
        ip, ap = self.ui.IngestionPanel, self.ui.AlignmentPanel
        storage_path = ip.StoragePathLineEdit.text().strip()
        reference_hybe = ap.ReferenceHybeComboBox.currentText()
        fov_list = self._parse_fov_list(ip.FovListLineEdit.text())
        ap.SameModalityResultsListWidget.clear()
        if not storage_path or not reference_hybe or not fov_list or not self.hybe_records:
            return
        results = {}
        for fov in fov_list:
            ready, _, _ = self._ingested_hybes_for_fov(storage_path, fov, self.hybe_records)
            if not ready:
                continue
            ready_records = [r for r in self.hybe_records if r['folder'] in ready]
            results[fov] = alignment.read_same_modality_matrices(storage_path, fov, ready_records)
        if not results:
            return
        self._same_modality_context = {'storage_path': storage_path, 'hybe_records': self.hybe_records, 'reference_hybe': reference_hybe}
        for fov, matrices in results.items():
            for hybe, H in matrices.items():
                item = QtWidgets.QListWidgetItem(f'FOV{fov:02d} {_matrix_summary(hybe, H)}')
                item.setData(QtCore.Qt.UserRole, (fov, hybe))
                ap.SameModalityResultsListWidget.addItem(item)
        self.fov_matrices.update({(storage_path, fov): m for fov, m in results.items()})

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
        Per-hybe readiness for one FOV: does {storage_path}/FOV{fov:02d}/
        {hybe}_stack.h5 exist, and does it actually contain /mip/ch{c} and
        /stack/ch{c} for every channel that hybe's own ExperimentLayout
        record declares? File-exists alone isn't enough to trust -- a
        partially-written or corrupted H5 (e.g. an interrupted ingestion
        run) would otherwise look "ready" and fail confusingly later, deep
        inside segmentation/alignment instead of here where it's obvious
        what's missing. Returns (ready, missing, invalid) hybe-folder lists.
        """
        ready, missing, invalid = [], [], []
        for record in hybe_records:
            hybe = record['folder']
            h5path = os.path.join(storage_path, f'FOV{fov:02d}', f'{hybe}_stack.h5')
            if not os.path.exists(h5path):
                missing.append(hybe)
                continue
            try:
                with h5py.File(h5path, 'r') as f:
                    complete = all(f'/mip/ch{c}' in f and f'/stack/ch{c}' in f for c in record['channels'])
                (ready if complete else invalid).append(hybe)
            except Exception:
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

                all_spots = [s for c in cell_dicts for s in c.get('spots', [])]
                spot_total = len(all_spots)
                spot_computed = sum(1 for s in all_spots if s.get('linked'))

                rows.append({'storage_path': storage_path, 'fov': fov, 'saved_at': saved_at, 'n_spots': spot_total,
                            'fov_computed': fov_computed, 'fov_total': fov_total,
                            'cross_computed': cross_computed, 'cross_total': cross_total,
                            'cell_computed': cell_computed, 'cell_total': cell_total,
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
        h5path = os.path.join(storage_path, f'FOV{fov:02d}', f'{hybe}_stack.h5')
        try:
            with h5py.File(h5path, 'r') as f:
                mip = f[f'/mip/ch{channel}'][:]
        except Exception as e:
            if not silent:
                QtWidgets.QMessageBox.critical(self, 'Show MIP Viewer', f'Could not read {h5path}:\n{type(e).__name__}: {e}')
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
            # nothing to compute -- a single small H5 read on the GUI
            # thread, not the multi-second Cellpose/watershed compute the
            # other two methods hide behind a QThread for.
            h5path = os.path.join(storage_path, f'FOV{fov:02d}', f'{reference_hybe}_stack.h5')
            try:
                with h5py.File(h5path, 'r') as f:
                    reference_image = f[f'/mip/ch{channel}'][:]
            except Exception as e:
                self._on_cell_segment_failed(str(e))
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
            self.cell_container = CellContainer([fov], modality='')
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
        h5path = os.path.join(storage_path, f'FOV{fov:02d}', f'{reference_hybe}_stack.h5')
        try:
            with h5py.File(h5path, 'r') as f:
                reference_image = f[f'/mip/ch{channel}'][:]
        except Exception as e:
            cp.LogTextEdit.append(f'Could not load {reference_hybe} ch{channel} for FOV{fov:02d}: {type(e).__name__}: {e}')
            return

        # activate whatever's persisted for this FOV (disk or already in
        # memory), same source _try_show_existing_cells uses -- but only
        # to CHECK whether it matches the panel's current hybe, never to
        # override the panel's own fov/hybe/channel choice.
        have_in_memory = (self.cell_container_permanent is not None and fov in self.cell_container_permanent.data
                          and self.cell_container_permanent.data[fov])
        if not have_in_memory:
            cell_dicts, modality = vlinks_store.read_cells(storage_path, fov)
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
            self.cell_container = CellContainer([fov], modality='')
        self.cell_container.data.setdefault(fov, [])

        # cells always show, regardless of which hybe/channel the panel is
        # currently displaying -- get_area_in_readout(reference_hybe)
        # transforms each cell's mask into this hybe's own frame when a
        # matrix is known (exact if this IS the cell's own reference_hybe,
        # aligned via cell.matrices[reference_hybe] otherwise); if no
        # matrix exists yet for this hybe (cell-based alignment hasn't
        # been run for it), falls back to the cell's raw/untransformed
        # position rather than hiding the mask entirely -- an earlier
        # version suppressed the whole overlay on a hybe mismatch, which
        # is exactly the behavior this replaces per explicit correction.
        mask = np.zeros(reference_image.shape, dtype=np.uint8)
        if cells:
            approximate = False
            height, width = mask.shape
            for cell in cells:
                try:
                    x, y = cell.get_area_in_readout(reference_hybe)
                except KeyError:
                    x, y = cell.area
                    approximate = True
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
        h5path = os.path.join(storage_path, f'FOV{fov:02d}', f'{reference_hybe}_stack.h5')
        try:
            with h5py.File(h5path, 'r') as f:
                channel = int(f.attrs['fiducial_channel'])
                reference_image = f[f'/mip/ch{channel}'][:]
        except Exception as e:
            cp.LogTextEdit.append(f'Could not load a reference image for existing FOV{fov:02d} cells: {type(e).__name__}: {e}')
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
        self._refresh_cell_results_from_disk(fov)
        cp.LogTextEdit.append(f'Showing {len(cells)} already-saved cell(s) for FOV{fov:02d} (from permanent container).')
        return True

    def _refresh_cell_results_from_disk(self, fov):
        """
        Cell-alignment matrices already persisted (vlinks.h5, from an
        earlier session) or already sitting in cell_container_permanent
        this session are real, activation-worthy state -- same principle
        as _refresh_same_modality_results_from_disk (FOV-level layer), applied to
        the per-cell layer: don't leave the Results list, Preview, and
        Show All-Readouts Overlay empty/unusable just because THIS session
        hasn't (re-)run cell alignment for this FOV yet. Populates
        CellResultsListWidget + the context _show_cell_alignment_preview
        needs, and the Overlay FOV/cell list, straight from whatever's
        already on the activated cells.
        """
        ap = self.ui.AlignmentPanel
        ip = self.ui.IngestionPanel
        storage_path = ip.StoragePathLineEdit.text().strip()
        if self.cell_container_permanent is None or fov not in self.cell_container_permanent.data or not storage_path:
            return
        cells = [c for c in self.cell_container_permanent.data[fov] if c.matrices]
        if not cells:
            return
        self._cell_alignment_context = {'storage_path': storage_path, 'hybe_records': self.hybe_records}
        self._cell_alignment_display_cells = [(fov, c) for c in cells]
        ap.CellResultsListWidget.clear()
        for cell in cells:
            item = QtWidgets.QListWidgetItem(f'FOV{fov:02d} Cell {cell.id}: {len(cell.matrices)} hybe(s) aligned')
            item.setData(QtCore.Qt.UserRole, (fov, cell.id))
            ap.CellResultsListWidget.addItem(item)
        ap.CellOverlayFovLineEdit.setText(str(fov))
        self._refresh_cell_overlay_list()

    # -- spot localization --

    def _refresh_spot_cell_list(self):
        sp = self.ui.SpotLocalizationPanel
        fov = self._current_spot_fov()
        if self.cell_container is None or fov is None:
            sp.populate_cell_choices([])
            return
        cells = self.cell_container.data.get(fov, [])
        sp.populate_cell_choices(cells)
        sp.LogTextEdit.append(f'Cell list refreshed: {len(cells)} cell(s) for FOV{fov:02d}.')

    def _current_spot_fov(self):
        # spot localization always operates on already-segmented cells, so
        # it shares Cell Segmentation's own FOV selector rather than the
        # Ingestion tab's FOV list (which is scoped to batch operations --
        # ingestion, batch alignment -- a different, decoupled concept).
        return self.ui.CellSegmentPanel.FovSpinBox.value()

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
        crop = localization._build_cell_crop(cell, hybe, channel, storage_path, fov, pad)
        if crop is None:
            sp.LogTextEdit.append(f'Cell {cell.id} has no alignment/overlap for {hybe} yet -- '
                                  f'run cell-based alignment for this hybe first.')
            return
        rxmin, rymin = crop['rxmin'], crop['rymin']
        self._spot_crop_context = {'cell': cell, 'hybe': hybe, 'channel': channel, 'rxmin': rxmin, 'rymin': rymin}
        existing_points = [(s.raw_coordinate[0] - rxmin, s.raw_coordinate[1] - rymin)
                           for s in cell.spots if s.hybe == hybe and s.channel == channel]
        self.spot_crop_displayer.set_data(crop['img'], existing_points)

    def _toggle_spot_crop_displayer(self, checked):
        if checked:
            self._load_spot_crop_for_display()
            self.spot_crop_displayer.show()
            self.spot_crop_displayer.raise_()
        else:
            self.spot_crop_displayer.hide()

    def _replace_cell_spots(self, cell, hybe, channel, new_spots):
        """
        Full-replace this cell's spots for exactly (hybe, channel) -- mirrors
        CellClassifier's own "rerunning localization on a cell/FOV only
        replaces that scope's spots" semantics, and keeps manual-edit
        reconciliation simple (spots_edited always hands back the FULL
        current crop-local point list, so a plain replace can't double-count).
        """
        cell.spots = [s for s in cell.spots if not (s.hybe == hybe and s.channel == channel)]
        cell.spots.extend(new_spots)
        cell.num_spots[hybe] = sum(1 for s in cell.spots if s.hybe == hybe)
        cell.total_num_spots = len(cell.spots)

    def _on_spot_crop_edited(self, points):
        ctx = self._spot_crop_context
        if ctx is None:
            return
        cell, hybe, channel = ctx['cell'], ctx['hybe'], ctx['channel']
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
            # empty fov_matrices: cell.matrices[hybe] must already exist to
            # have gotten a crop at all (_build_cell_crop's own precondition),
            # and spot_mapper._resolve_matrix always prefers cell.matrices
            # over fov_matrices when a cell is given.
            cx, cy = spot_mapper.raw_to_reference((raw_x, raw_y), hybe, {}, cell=cell)
            spot = ASpot()
            spot.set_metadata(fov=cell.fov, hybe=hybe, channel=channel, cell=cell.id,
                              coordinate=(cx, cy, 0.0), raw_coordinate=(raw_x, raw_y, 0.0),
                              size=0.0, brightness=brightness)
            new_spots.append(spot)
        self._replace_cell_spots(cell, hybe, channel, new_spots)
        self.ui.SpotLocalizationPanel.LogTextEdit.append(
            f'Cell {cell.id}, {hybe} ch{channel}: {len(new_spots)} spot(s) after manual edit.')
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
        pct = sp.ThresholdPercentSpinBox.value()
        min_distance = sp.MinDistanceSpinBox.value()
        pad = sp.PadSpinBox.value()
        try:
            self._run_spot_auto_detect_body(sp, storage_path, fov, hybe, channel, pct, min_distance, pad)
        except Exception as e:
            # this method had no error handling at all -- given the same
            # silent-failure pattern was confirmed real elsewhere (a
            # reproducible KeyError in celltype determination that crashed
            # with zero dialog/log/feedback), an unhandled exception here
            # would look identical to "nothing happened" -- exactly what
            # was reported ("still didn't appear"). Surface it for real.
            QtWidgets.QMessageBox.critical(self, 'Run Auto-Detect error', f'{type(e).__name__}: {e}')

    def _run_spot_auto_detect_body(self, sp, storage_path, fov, hybe, channel, pct, min_distance, pad):
        if sp.scope() == 'current_cell':
            cell = self._selected_spot_cell()
            if cell is None:
                QtWidgets.QMessageBox.warning(self, 'Run Auto-Detect', 'Select a cell first (Current Cell scope).')
                return
            crop = localization._build_cell_crop(cell, hybe, channel, storage_path, fov, pad)
            if crop is None:
                QtWidgets.QMessageBox.warning(self, 'Run Auto-Detect', f'Cell {cell.id} has no alignment/overlap for {hybe} yet.')
                return
            img, rxmin, rymin = crop['img'], crop['rxmin'], crop['rymin']
            threshold_abs = (pct / 100.0) * np.nanmax(img)
            coords = peak_local_max(img, min_distance=min_distance, exclude_border=1, threshold_abs=threshold_abs)
            new_spots = []
            for y, x in coords:
                raw_x, raw_y = int(x) + rxmin, int(y) + rymin
                cx, cy = spot_mapper.raw_to_reference((raw_x, raw_y), hybe, {}, cell=cell)
                spot = ASpot()
                spot.set_metadata(fov=fov, hybe=hybe, channel=channel, cell=cell.id,
                                  coordinate=(cx, cy, 0.0), raw_coordinate=(raw_x, raw_y, 0.0),
                                  size=0.0, brightness=float(img[y, x]))
                new_spots.append(spot)
            self._replace_cell_spots(cell, hybe, channel, new_spots)
            sp.LogTextEdit.append(f'Cell {cell.id}, {hybe} ch{channel}: {len(new_spots)} spot(s) detected (Current Cell scope).')
            self._load_spot_crop_for_display()
            self._refresh_spot_cell_list()
        else:
            # Whole-FOV scope still requires cell segmentation to have run
            # for this FOV -- every ASpot belongs to a cell in this app
            # (no independent "unassigned spot" pool), so there's nowhere
            # sound to put a detected peak without one. This used to be
            # loosened into a "preview only, no cells required" fallback,
            # but that was solving the wrong problem: the actual gap was
            # that already-segmented cells weren't being activated from
            # vlinks.h5 when this FOV became current (see _activate_fov),
            # not that the requirement itself was wrong. With activation
            # wired, a legitimately-segmented FOV always has its cells
            # available here without needing to re-run segmentation.
            has_cells = bool(self.cell_container is not None and self.cell_container.data.get(fov))
            mask_matches_fov = bool(self._last_segment_context is not None and self._last_segment_context['fov'] == fov)
            if not (has_cells and mask_matches_fov):
                reason = 'no cells segmented yet for this FOV' if not has_cells else \
                         'the displayed cell mask is from a different FOV -- reselect this FOV in Cell Segmentation first'
                QtWidgets.QMessageBox.warning(self, 'Run Auto-Detect',
                                              f'Whole FOV scope requires cell segmentation first ({reason}).')
                return

            fmats = self.fov_matrices.get((storage_path, fov), {})
            if hybe not in fmats:
                QtWidgets.QMessageBox.warning(self, 'Run Auto-Detect',
                                              f"Run FOV alignment for '{hybe}' first (needed to place whole-FOV peaks "
                                              f"in the shared reference frame -- fov_matrices already stores an "
                                              f"identity entry for the reference hybe itself, so this covers every hybe).")
                return
            h5path = os.path.join(storage_path, f'FOV{fov:02d}', f'{hybe}_stack.h5')
            with h5py.File(h5path, 'r') as f:
                mip = f[f'/mip/ch{channel}'][:]
            threshold_abs = (pct / 100.0) * mip.max()
            coords = peak_local_max(mip, min_distance=min_distance, exclude_border=1, threshold_abs=threshold_abs)

            cells = self.cell_container.data[fov]
            mask = self.cell_displayer.mask
            cells_by_id = {c.id: c for c in cells}
            spots_by_cell = {c.id: [] for c in cells}
            n_discarded = 0
            for y, x in coords:
                raw_x, raw_y = int(x), int(y)
                ref_x, ref_y = spot_mapper.raw_to_reference((raw_x, raw_y), hybe, fmats)
                iry, irx = int(round(ref_y)), int(round(ref_x))
                if mask is None or not (0 <= iry < mask.shape[0] and 0 <= irx < mask.shape[1]):
                    n_discarded += 1
                    continue
                label = int(mask[iry, irx])
                owning_cell = cells_by_id.get(label)
                if label == 0 or owning_cell is None:
                    n_discarded += 1
                    continue
                cx, cy = spot_mapper.raw_to_reference((raw_x, raw_y), hybe, fmats, cell=owning_cell)
                spot = ASpot()
                spot.set_metadata(fov=fov, hybe=hybe, channel=channel, cell=owning_cell.id,
                                  coordinate=(cx, cy, 0.0), raw_coordinate=(raw_x, raw_y, 0.0),
                                  size=0.0, brightness=float(mip[y, x]))
                spots_by_cell[owning_cell.id].append(spot)

            for cell in cells:
                self._replace_cell_spots(cell, hybe, channel, spots_by_cell[cell.id])
            n_total = sum(len(v) for v in spots_by_cell.values())
            sp.LogTextEdit.append(f'Whole FOV {hybe} ch{channel}: {len(coords)} peak(s) detected, '
                                  f'{n_total} assigned to cells, {n_discarded} outside any cell and discarded.')
            self._refresh_spot_cell_list()

            # real visual confirmation -- Whole FOV scope has no single
            # "current cell" crop to show, so the crop displayer shows the
            # whole raw hybe MIP with every detected peak marked (raw pixel
            # coordinates, same frame as mip itself). Manual click editing
            # doesn't map cleanly onto a whole-FOV view (a click here would
            # need its own cell-ownership lookup to know which cell.spots
            # to edit), so _spot_crop_context stays None -- same
            # already-guarded no-op as every other non-editable view.
            self._spot_crop_context = None
            all_points = [(float(x), float(y)) for y, x in coords]
            self.spot_crop_displayer.set_data(mip, all_points)
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

    def _assign_barcode_channel(self):
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
            h5path = os.path.join(storage_path, f'FOV{fov:02d}', f'{hybe}_stack.h5')
            try:
                with h5py.File(h5path, 'r') as f:
                    img = f[f'/mip/ch{channel}'][:]
            except Exception as e:
                ctp.LogTextEdit.append(f'FOV{fov:02d}: could not read {hybe} ch{channel} -- {e}')
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
            h5path = os.path.join(storage_path, f'FOV{fov:02d}', f'{hybe}_stack.h5')
            try:
                with h5py.File(h5path, 'r') as f:
                    img = f[f'/mip/ch{channel}'][:]
            except Exception as e:
                ctp.LogTextEdit.append(f'Overview: could not read {hybe} ch{channel} -- {e}')
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
            for container in containers:
                for cells in container.data.values():
                    for cell in cells:
                        cell.celltype = celltype.classify_fov(cell.fov, celltype_from_fov)
                        cell.linked, cell.linked_at = True, now
                        for spot in cell.spots:
                            spot.celltype = cell.celltype
                            spot.linked, spot.linked_at = True, now
                        n_cells += 1
                        last_fov = cell.fov
            ctp.RunCelltypeDeterminationPushButton.setEnabled(True)
            ctp.LogTextEdit.append(f'FOV-mode: {n_cells} cell(s) classified.')
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
        for container in containers:
            for fov, cells in container.data.items():
                if cells:
                    last_fov = fov
                fov_key = str(fov)
                if fov_key not in image_cache:
                    image_cache[fov_key] = {}
                    for bch in barcode_channel:
                        hybe, channel_id = bch
                        h5path = os.path.join(storage_path, f'FOV{fov:02d}', f'{hybe}_stack.h5')
                        try:
                            with h5py.File(h5path, 'r') as f:
                                image_cache[fov_key][bch] = f[f'/mip/ch{channel_id}'][:]
                        except Exception:
                            image_cache[fov_key][bch] = None

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
                    area_by_channel, ok = {}, True
                    for bch in barcode_channel:
                        hybe, channel_id = bch
                        img = image_cache[fov_key].get(bch)
                        if img is None:
                            ok = False
                            break
                        if hybe == cell.reference_hybe:
                            x_h, y_h = x_ref, y_ref
                        elif hybe in cell.matrices:
                            # cell.matrices[hybe] is already the final, fully-composed
                            # matrix once cell-based alignment has run (FOV + cross-modal
                            # were already folded in when this cell's own alignment was
                            # computed) -- a plain per-point inverse transform, not
                            # cell.get_area_in_readout's masking/closing machinery, so
                            # point order/count stays identical across every channel.
                            Hinv = la.inv(cell.matrices[hybe]['yx'])
                            pts = Hinv @ np.vstack([x_ref, y_ref, np.ones_like(x_ref, dtype=float)])
                            x_h, y_h = pts[0], pts[1]
                        else:
                            ok = False
                            break
                        height, width = img.shape
                        area_by_channel[bch] = (np.clip(x_h, 0, width - 1), np.clip(y_h, 0, height - 1))
                    if not ok:
                        n_cells_skipped += 1
                        continue
                    cell.celltype = celltype.classify_cell_barcode(area_by_channel, cell.fov, image_cache,
                                                                   celltype_determination, method=ctp.barcode_method())
                    cell.linked, cell.linked_at = True, now
                    n_cells += 1

                    for spot in cell.spots:
                        xy_by_channel, spot_ok = {}, True
                        for bch in barcode_channel:
                            hybe, channel_id = bch
                            if hybe not in cell.matrices and hybe != cell.reference_hybe:
                                spot_ok = False
                                break
                            sx, sy = spot_mapper.reference_to_raw(spot.coordinate[:2], hybe, {}, cell=cell)
                            xy_by_channel[bch] = (sx, sy)
                        if not spot_ok:
                            continue
                        spot.celltype = celltype.classify_spot_barcode(xy_by_channel, cell.fov, image_cache, celltype_determination)
                        spot.linked, spot.linked_at = True, now
                        n_spots += 1

        ctp.RunCelltypeDeterminationPushButton.setEnabled(True)
        ctp.LogTextEdit.append(f'Barcode-mode: {n_cells} cell(s), {n_spots} spot(s) classified '
                               f'({n_cells_skipped} cell(s) skipped -- missing alignment for a barcode hybe).')
        self.statusBar().showMessage('Celltype determination complete.', 5000)
        QtWidgets.QMessageBox.information(self, 'Celltype determination complete',
                                          f'{n_cells} cell(s), {n_spots} spot(s) classified '
                                          f'({n_cells_skipped} cell(s) skipped -- missing alignment for a barcode hybe).')
        if last_fov is not None:
            self._show_celltype_result(last_fov)

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
        h5path = os.path.join(storage_path, f'FOV{fov:02d}', f'{reference_hybe}_stack.h5')
        try:
            with h5py.File(h5path, 'r') as f:
                channel = int(f.attrs['fiducial_channel'])
                reference_image = f[f'/mip/ch{channel}'][:]
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Show Celltype Result', f'{type(e).__name__}: {e}')
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
        ap = self.ui.AlignmentPanel
        ip = self.ui.IngestionPanel
        if not self.hybe_records:
            QtWidgets.QMessageBox.warning(self, 'Run FOV Alignment', 'Parse a layout first.')
            return
        reference_hybe = ap.ReferenceHybeComboBox.currentText()
        storage_path = ip.StoragePathLineEdit.text().strip()
        fov_list_all = self._parse_fov_list(ip.FovListLineEdit.text())
        if not storage_path or not fov_list_all:
            QtWidgets.QMessageBox.warning(self, 'Run FOV Alignment', 'Set storage path and FOV list in the Ingestion tab first.')
            return

        selected_folders = set(ip.hybe_checkbox_items())
        hybe_records = [r for r in self.hybe_records if r['folder'] in selected_folders] if selected_folders else self.hybe_records
        if reference_hybe not in {r['folder'] for r in hybe_records}:
            QtWidgets.QMessageBox.warning(self, 'Run FOV Alignment', 'Reference hybe must be checked (and ingested) in the Ingestion tab.')
            return

        manual = ap.is_manual_mode()
        # manual mode stays single-FOV (today's behavior, human-paced
        # review doesn't scale to many FOVs); automatic batches every FOV.
        fov_list = [fov_list_all[0]] if manual else fov_list_all

        border_trim = ap.SameModalityBorderTrimSpinBox.value()
        max_shift = ap.SameModalityMaxShiftSpinBox.value() or None

        ap.RunFovAlignmentPushButton.setEnabled(False)
        self.statusBar().showMessage('Running FOV alignment...')
        self._alignment_worker = AlignmentWorker(storage_path, fov_list, hybe_records, reference_hybe, write=not manual,
                                                  border_trim=border_trim, max_shift=max_shift)
        self._alignment_worker.progress.connect(lambda done, total, msg: self.statusBar().showMessage(msg))
        self._alignment_worker.finished_ok.connect(
            lambda results: self._on_fov_alignment_finished(results, storage_path, hybe_records, reference_hybe, manual))
        self._alignment_worker.failed.connect(self._on_fov_alignment_failed)
        self._alignment_worker.start()

    def _on_fov_alignment_finished(self, results, storage_path, hybe_records, reference_hybe, manual):
        ap = self.ui.AlignmentPanel
        self._same_modality_context = {'storage_path': storage_path, 'hybe_records': hybe_records, 'reference_hybe': reference_hybe}
        ap.SameModalityResultsListWidget.clear()
        for fov, matrices in results.items():
            for hybe, H in matrices.items():
                item = QtWidgets.QListWidgetItem(f'FOV{fov:02d} {_matrix_summary(hybe, H)}')
                item.setData(QtCore.Qt.UserRole, (fov, hybe))
                ap.SameModalityResultsListWidget.addItem(item)
        ap.RunFovAlignmentPushButton.setEnabled(True)
        self.statusBar().showMessage('FOV alignment computed.', 5000)

        if manual:
            self._pending_same_modality_alignment = results
            ap.SameModalityAcceptPushButton.setEnabled(True)
            ap.SameModalityRejectPushButton.setEnabled(True)
        else:
            self.fov_matrices.update({(storage_path, fov): m for fov, m in results.items()})
            self._pending_same_modality_alignment = None
            channel_type = ap.SameModalityChannelTypeComboBox.currentText()
            vlinks_store.write_global_params(storage_path, same_modality_reference_hybe=reference_hybe,
                                             same_modality_channel_type=channel_type)
            for fov, matrices in results.items():
                save_path = os.path.join(storage_path, f'FOV{fov:02d}', 'alignment_overlay.png')
                self.preview_canvas.draw_fov_all_readouts_overlay(storage_path, fov, hybe_records, reference_hybe, matrices,
                                                          save_path=save_path, channel_type=channel_type)
            QtWidgets.QMessageBox.information(self, 'FOV alignment complete',
                                              f'{len(results)} FOV(s) aligned and saved; overlay image(s) written.')

    def _on_fov_alignment_failed(self, message):
        self.ui.AlignmentPanel.RunFovAlignmentPushButton.setEnabled(True)
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
        ap.SameModalityAcceptPushButton.setEnabled(False)
        ap.SameModalityRejectPushButton.setEnabled(False)
        QtWidgets.QMessageBox.information(self, 'FOV alignment accepted', 'Matrices written to H5; overlay image(s) saved.')

    def _reject_same_modality_alignment(self):
        ap = self.ui.AlignmentPanel
        self._pending_same_modality_alignment = None
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
        FOV is an explicit field, not implicitly "whatever was just
        aligned" -- an empty field falls back to the last-run FOV (the old
        behavior, still convenient right after Run FOV Alignment), but any
        FOV can be typed in directly. Matrices come from whichever source
        actually has them: a staged (not-yet-accepted) manual result takes
        priority, then the in-memory fov_matrices cache (already populated
        by _activate_fov for any FOV that's been visited), then a direct
        disk read as a last resort -- so this never requires re-running
        alignment in this session just to view an already-aligned FOV.
        """
        ap = self.ui.AlignmentPanel
        ip = self.ui.IngestionPanel
        storage_path = ip.StoragePathLineEdit.text().strip()
        fov_text = ap.SameModalityOverlayFovLineEdit.text().strip()
        if fov_text:
            fov = int(fov_text)
        elif ap.SameModalityResultsListWidget.count() > 0:
            fov = ap.SameModalityResultsListWidget.item(0).data(QtCore.Qt.UserRole)[0]
        else:
            fov = None
        reference_hybe = ap.ReferenceHybeComboBox.currentText()
        if not storage_path or fov is None or not reference_hybe or not self.hybe_records:
            QtWidgets.QMessageBox.warning(self, 'Show All-Readouts Overlay',
                                          'Set storage path (Ingestion tab), reference hybe, and an FOV first.')
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

    def _run_cross_modal_alignment(self):
        ap = self.ui.AlignmentPanel
        ip = self.ui.IngestionPanel
        rna_storage_path = ap.RnaStoragePathLineEdit.text().strip()
        dna_storage_path = ap.DnaStoragePathLineEdit.text().strip()
        rna_reference_hybe = ap.RnaReferenceHybeComboBox.currentText().strip()
        dna_reference_hybe = ap.DnaReferenceHybeComboBox.currentText().strip()
        channel_type = ap.ChannelTypeComboBox.currentText()
        fov_list_all = self._parse_fov_list(ip.FovListLineEdit.text())
        if not all([rna_storage_path, dna_storage_path, rna_reference_hybe, dna_reference_hybe]) or not fov_list_all:
            QtWidgets.QMessageBox.warning(self, 'Run Cross-Modal Alignment',
                                          'Fill in both storage paths and reference hybes, and set a FOV list in the Ingestion tab.')
            return

        manual = ap.is_manual_mode()
        fov_list = [fov_list_all[0]] if manual else fov_list_all

        border_trim = ap.CrossModalBorderTrimSpinBox.value()
        max_shift = ap.CrossModalMaxShiftSpinBox.value() or None

        ap.RunCrossModalPushButton.setEnabled(False)
        self.statusBar().showMessage('Running cross-modal alignment...')
        self._cross_modal_worker = CrossModalAlignmentWorker(rna_storage_path, dna_storage_path, fov_list, self.fov_matrices,
                                                              rna_reference_hybe, dna_reference_hybe, channel_type, write=not manual,
                                                              border_trim=border_trim, max_shift=max_shift)
        self._cross_modal_worker.progress.connect(lambda done, total, msg: self.statusBar().showMessage(msg))
        self._cross_modal_worker.finished_ok.connect(
            lambda results: self._on_cross_modal_finished(results, rna_storage_path, dna_storage_path,
                                                           rna_reference_hybe, dna_reference_hybe, channel_type, manual))
        self._cross_modal_worker.failed.connect(self._on_cross_modal_failed)
        self._cross_modal_worker.start()

    def _on_cross_modal_finished(self, results, rna_storage_path, dna_storage_path, rna_reference_hybe, dna_reference_hybe, channel_type, manual):
        ap = self.ui.AlignmentPanel
        ap.RunCrossModalPushButton.setEnabled(True)
        self.statusBar().showMessage('Cross-modal alignment computed.', 5000)
        self._cross_modal_context = {'rna_storage_path': rna_storage_path, 'dna_storage_path': dna_storage_path,
                                      'rna_reference_hybe': rna_reference_hybe, 'dna_reference_hybe': dna_reference_hybe,
                                      'channel_type': channel_type}
        ap.CrossModalResultLabel.setText(' | '.join(f'FOV{fov:02d} {_matrix_summary("DNA->RNA", H)}' for fov, H in results.items()))
        last_fov = list(results.keys())[-1]
        # cross-modal has no results-list to click through (just one label)
        # -- this is the only place its preview ever gets shown, so show the
        # pop-up here rather than waiting for a separate interactive trigger
        self.alignment_preview_window.show()
        self.alignment_preview_window.raise_()
        self.preview_canvas.draw_cross_modal_preview(rna_storage_path, dna_storage_path, last_fov,
                                             rna_reference_hybe, dna_reference_hybe, channel_type, results[last_fov],
                                             rna_fov_matrices=self.fov_matrices.get((rna_storage_path, last_fov), {}),
                                             dna_fov_matrices=self.fov_matrices.get((dna_storage_path, last_fov), {}))

        if manual:
            self._pending_cross_modal = results
            ap.CrossModalAcceptPushButton.setEnabled(True)
            ap.CrossModalRejectPushButton.setEnabled(True)
        else:
            self.cross_modal_result.update({(dna_storage_path, fov): H for fov, H in results.items()})
            self._pending_cross_modal = None
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

    def _on_cross_modal_failed(self, message):
        self.ui.AlignmentPanel.RunCrossModalPushButton.setEnabled(True)
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
        ap.CrossModalAcceptPushButton.setEnabled(False)
        ap.CrossModalRejectPushButton.setEnabled(False)
        QtWidgets.QMessageBox.information(self, 'Cross-modal alignment accepted', 'Result written to H5; overlay image(s) saved.')

    def _reject_cross_modal(self):
        ap = self.ui.AlignmentPanel
        self._pending_cross_modal = None
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
        accepted) manual result, the in-memory cross_modal_result cache,
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

        fov_text = ap.CrossModalOverlayFovComboBox.currentText().strip()
        if fov_text:
            fov = int(fov_text)
        elif self._pending_cross_modal:
            fov = list(self._pending_cross_modal.keys())[-1]
        elif self.cross_modal_result:
            matching = [f for (sp, f) in self.cross_modal_result if sp == dna_storage_path]
            fov = matching[-1] if matching else None
        else:
            fov = None
        if fov is None:
            QtWidgets.QMessageBox.warning(self, 'Show Cross-Modal Overlay', 'Enter an Overlay FOV first.')
            return

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
        other_reference_hybe) for the OTHER modality, or None if
        storage_path isn't one of the two configured Cross-Modality
        Alignment paths, or no cross-modal result exists yet for this FOV.

        other_fov_matrices: {hybe: 3x3} -- each of the OTHER modality's
        own within-experiment matrices, composed with the cross-modal
        correction (or its inverse, if storage_path is the DNA side and
        H_across therefore needs to be undone to land in DNA's own frame)
        so it lands directly in storage_path's (the cell's) own frame --
        ready to hand to compute_cell_alignment as a second, independent
        fov_matrices input, exactly like
        _composed_fov_matrices_for_cell_alignment's own output for the
        same-modality case. Any hybe name already present in
        self.hybe_records (the SAME modality's own set, which
        storage_path always corresponds to when called from
        _run_cell_alignment) is skipped -- the shared cross-modal bridge
        hybe (e.g. Hyb_130) is a real file in BOTH modalities, and the
        same-modality version is used directly rather than overwritten
        by a same-named cross-modal one.

        other_reference_hybe: that modality's own WITHIN-EXPERIMENT
        reference hybe (e.g. DNA's own Hyb_002) -- used as
        compute_cell_alignment's own phase-correlation anchor for this
        second call, since the same-modality run's own reference hybe
        (e.g. Hyb_101) doesn't exist in the other modality's hybe_records
        at all. Deliberately NOT the cross-modal bridge hybe (e.g.
        Hyb_130): that hybe is excluded from other_fov_matrices below (see
        above) precisely to avoid colliding with the same-modality set's
        own entry for it, so it can't double as the anchor here either --
        compute_cell_alignment always writes cell.matrices[reference_hybe]
        too, which would silently re-introduce the exact collision the
        exclusion was meant to prevent.
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

        other_data = next((d for d in self.modality_data.values() if d['storage_path'] == other_storage_path), None)
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

        same_hybe_names = {r['folder'] for r in self.hybe_records}
        other_fov_matrices = {hybe: alignment.compose_chain([H, H_compose]) for hybe, H in other_within.items()
                              if hybe not in same_hybe_names}
        if not other_fov_matrices:
            return None
        return other_storage_path, other_hybe_records, other_fov_matrices, other_reference_hybe

    def _cell_overlay_target_specs(self, cell, storage_path, fov, hybe_records, channel_type):
        """
        Resolves every hybe in cell.matrices (both modalities) into what
        draw_cell_all_readouts_overlay needs to read/crop/warp it: storage
        path, channel, fiducial channel, the FOV-level matrix (the
        'FOV/cross-modal' stage), and this cell's own final yx/zx
        matrices. Unlike the single-hybe preview, every target here is
        warped into ONE fixed shared coordinate FRAME (cell.reference_hybe,
        always read at H=eye since cell.area is already native to it) --
        that part has no per-target ambiguity.

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
        same_record_by_folder = {r['folder']: r for r in hybe_records}
        same_fov_matrices = self._composed_fov_matrices_for_cell_alignment(storage_path, fov)
        other = self._other_modality_cell_alignment_inputs(storage_path, fov)
        other_record_by_folder, other_fov_matrices, other_storage_path = {}, {}, None
        if other is not None:
            other_storage_path, other_hybe_records, other_fov_matrices, _ = other
            other_record_by_folder = {r['folder']: r for r in other_hybe_records}

        specs = []
        for hybe, mats in cell.matrices.items():
            if hybe not in cell.matrix_provenance:
                continue
            if hybe in same_record_by_folder:
                record = same_record_by_folder[hybe]
                target_storage_path = storage_path
                fov_only_matrix = same_fov_matrices.get(hybe, np.eye(3))
            elif hybe in other_record_by_folder:
                record = other_record_by_folder[hybe]
                target_storage_path = other_storage_path
                fov_only_matrix = other_fov_matrices.get(hybe, np.eye(3))
            else:
                continue
            specs.append({
                'hybe': hybe, 'storage_path': target_storage_path,
                'channel': alignment.pick_channel_by_type(record, channel_type),
                'fiducial_channel': record['fiducial_channel'],
                'fov_only_matrix': fov_only_matrix,
                'final_matrix': mats.get('yx', np.eye(3)),
                'zx_matrix': mats.get('zx', np.eye(3)),
            })
        return specs

    def _run_cell_alignment(self):
        ap = self.ui.AlignmentPanel
        ip = self.ui.IngestionPanel
        storage_path = ip.StoragePathLineEdit.text().strip()
        fov_list_all = self._parse_fov_list(ip.FovListLineEdit.text())
        if not storage_path or not fov_list_all:
            QtWidgets.QMessageBox.warning(self, 'Run Cell Alignment', 'Set storage path and FOV list in the Ingestion tab first.')
            return

        manual = ap.is_manual_mode()
        jobs = []  # (fov, cells_to_compute, real_cells_or_None, fov_matrices_for_fov)

        if manual:
            fov = ap.CellFovSpinBox.value()
            if fov not in fov_list_all:
                QtWidgets.QMessageBox.warning(self, 'Run Cell Alignment',
                                              f'FOV{fov} is not in the Ingestion tab\'s FOV list.')
                return
            if (storage_path, fov) not in self.fov_matrices:
                QtWidgets.QMessageBox.warning(self, 'Run Cell Alignment', 'Run (and accept) FOV alignment for this FOV first.')
                return
            if (self.cell_container is None or fov not in self.cell_container.data
                    or len(self.cell_container.data[fov]) == 0):
                QtWidgets.QMessageBox.warning(self, 'Run Cell Alignment',
                                              'No cells segmented for this FOV yet -- run Cell Segmentation first.')
                return
            real_cells = self.cell_container.data[fov]
            staged_cells = [deepcopy(c) for c in real_cells]
            jobs.append((fov, staged_cells, real_cells, self._composed_fov_matrices_for_cell_alignment(storage_path, fov),
                        self._other_modality_cell_alignment_inputs(storage_path, fov)))
            self._pending_cell_alignment_fov = fov
        else:
            # automatic mode only processes FOVs with *saved* (permanent)
            # segmentation -- Cellpose parameters are per-FOV-tuned, so this
            # never auto-segments on the user's behalf, only auto-aligns
            # what's already been confirmed.
            if self.cell_container_permanent is None:
                QtWidgets.QMessageBox.warning(self, 'Run Cell Alignment', 'No saved (permanent) cell segmentation for any FOV yet.')
                return
            skipped = []
            for fov, cells in self.cell_container_permanent.data.items():
                if not cells:
                    continue
                if (storage_path, fov) not in self.fov_matrices:
                    skipped.append(fov)
                    continue
                jobs.append((fov, cells, None, self._composed_fov_matrices_for_cell_alignment(storage_path, fov),
                            self._other_modality_cell_alignment_inputs(storage_path, fov)))
            if not jobs:
                msg = 'No FOV has both saved cells and FOV-level alignment yet.'
                if skipped:
                    msg += f' Skipped (no FOV alignment yet): {sorted(skipped)}'
                QtWidgets.QMessageBox.warning(self, 'Run Cell Alignment', msg)
                return
            if skipped:
                QtWidgets.QMessageBox.information(self, 'Run Cell Alignment',
                                                  f'Skipping FOV(s) without FOV-level alignment yet: {sorted(skipped)}')

        cell_reference_hybe = ap.CellReferenceHybeComboBox.currentText() or None
        channel_type = ap.CellChannelTypeComboBox.currentText()
        pad = ap.CellPadSpinBox.value()

        ap.RunCellAlignmentPushButton.setEnabled(False)
        self.statusBar().showMessage('Computing cell alignment...')
        worker_jobs = [(fov, cells, fov_matrices, other_ctx) for fov, cells, _, fov_matrices, other_ctx in jobs]
        real_cells_by_fov = {fov: real for fov, _, real, _, _ in jobs if real is not None}
        self._cell_alignment_worker = CellAlignmentWorker(worker_jobs, storage_path, self.hybe_records,
                                                           reference_hybe=cell_reference_hybe, channel_type=channel_type,
                                                           pad=pad)
        self._cell_alignment_worker.progress.connect(lambda done, total, msg: self.statusBar().showMessage(msg))
        self._cell_alignment_worker.finished_ok.connect(
            lambda results: self._on_cell_alignment_finished(results, real_cells_by_fov, storage_path, manual,
                                                              cell_reference_hybe, channel_type, pad))
        self._cell_alignment_worker.failed.connect(self._on_cell_alignment_failed)
        self._cell_alignment_worker.start()

    def _on_cell_alignment_finished(self, results, real_cells_by_fov, storage_path, manual,
                                    cell_reference_hybe, channel_type, pad):
        """results: [(fov, cells), ...] -- cells are staged (manual) or real (automatic, mutated in place)."""
        ap = self.ui.AlignmentPanel
        ap.RunCellAlignmentPushButton.setEnabled(True)
        self.statusBar().showMessage('Cell alignment computed.', 5000)

        self._cell_alignment_context = {'storage_path': storage_path, 'hybe_records': self.hybe_records}
        self._cell_alignment_display_cells = []
        ap.CellResultsListWidget.clear()
        for fov, cells in results:
            self._cell_alignment_display_cells.extend((fov, cell) for cell in cells)
            for cell in cells:
                item = QtWidgets.QListWidgetItem(f'FOV{fov:02d} Cell {cell.id}: {len(cell.matrices)} hybe(s) aligned')
                item.setData(QtCore.Qt.UserRole, (fov, cell.id))
                ap.CellResultsListWidget.addItem(item)

        total_cells = sum(len(cells) for _, cells in results)
        if manual:
            fov, cells = results[0]
            real_cells = real_cells_by_fov[fov]
            self._pending_cell_alignment = list(zip(real_cells, cells))
            self._pending_cell_alignment_params = {'reference_hybe': cell_reference_hybe,
                                                    'channel_type': channel_type, 'pad': pad}
            ap.CellAcceptPushButton.setEnabled(True)
            ap.CellRejectPushButton.setEnabled(True)
            QtWidgets.QMessageBox.information(self, 'Cell alignment computed',
                                              f'{total_cells} cell(s) computed -- review then Accept or Reject.')
        else:
            self._pending_cell_alignment = None
            storage_paths = self._all_vlinks_storage_paths()
            for path in storage_paths:
                vlinks_store.write_global_params(path, cell_alignment_reference_hybe=cell_reference_hybe,
                                                 cell_alignment_channel_type=channel_type, cell_alignment_pad=pad)
            record_by_folder = {r['folder']: r for r in self.hybe_records}
            for fov, cells in results:
                for cell in cells:
                    reference_record = record_by_folder.get(cell.reference_hybe)
                    if reference_record is None:
                        continue
                    save_path = os.path.join(storage_path, f'FOV{fov:02d}', f'cell{cell.id}_alignment_overlay.png')
                    reference_channel = alignment.pick_channel_by_type(reference_record, channel_type)
                    target_specs = self._cell_overlay_target_specs(cell, storage_path, fov, self.hybe_records, channel_type)
                    self.preview_canvas.draw_cell_all_readouts_overlay(
                        cell, fov, cell.reference_hybe, storage_path, reference_channel,
                        reference_record['fiducial_channel'], target_specs, pad=pad, save_path=save_path)
                # automatic mode's own docstring: cells here ARE the real
                # objects inside cell_container_permanent.data[fov], mutated
                # in place -- but that in-memory mutation alone never
                # reached vlinks.h5. "Automatic (compute + save)" is the
                # documented contract for this mode (same as within-
                # experiment alignment's automatic path, which does write
                # immediately) -- cell-based alignment silently didn't honor
                # it until now.
                if storage_paths and self.cell_container_permanent is not None:
                    vlinks_store.mirror_write_cells(storage_paths, fov, self.cell_container_permanent)
            QtWidgets.QMessageBox.information(self, 'Cell alignment complete',
                                              f'{total_cells} cell(s) across {len(results)} FOV(s) aligned, saved to vlinks.h5; '
                                              f'overlay image(s) written.')

    def _on_cell_alignment_failed(self, message):
        ap = self.ui.AlignmentPanel
        ap.RunCellAlignmentPushButton.setEnabled(True)
        self.statusBar().clearMessage()
        QtWidgets.QMessageBox.critical(self, 'Cell alignment error', message)

    def _show_cell_alignment_preview(self, item):
        """
        Populates CellPreviewHybeComboBox with every hybe this cell has ANY
        alignment data for -- cell.matrices (not cell.matrix_provenance,
        which excludes whichever hybe was used as a call's own phase-
        correlation anchor, e.g. DNA's own within-experiment reference
        hybe -- that hybe still legitimately has a cell.matrices entry
        (the FOV-level matrix, with no cell-level residual), and per
        explicit request should still be selectable/previewable, not
        invisible). Each combo item's visible label carries a "(RNA)"/
        "(DNA)" suffix for whichever hybe DIDN'T come from this cell's own
        (same-modality) hybe_records, since hybe names aren't guaranteed
        unique across modalities; the real (bare) hybe name is stored as
        the item's data (Qt.UserRole via addItem's userData) so resolution
        never depends on parsing the label back apart.

        Preserves whatever hybe was already selected (by bare name, via
        itemData) across a Results-list cell switch -- previously this
        always reset to the first hybe in the new cell's list, discarding
        the user's in-progress comparison every time they clicked a
        different cell.
        """
        ctx = self._cell_alignment_context
        if ctx is None:
            return
        fov, cell_id = item.data(QtCore.Qt.UserRole)
        cell = next((c for f, c in self._cell_alignment_display_cells if f == fov and c.id == cell_id), None)
        if cell is None or not cell.matrices:
            return
        self._cell_preview_context = {'fov': fov, 'cell': cell, 'storage_path': ctx['storage_path'],
                                      'hybe_records': ctx['hybe_records']}
        ap = self.ui.AlignmentPanel
        last_hybe = ap.CellPreviewHybeComboBox.currentData()
        target_hybes = list(cell.matrices.keys())
        same_names = {r['folder'] for r in ctx['hybe_records']}
        other_modality_name = next((n for n in self.modality_names if n != self.current_modality), None)

        ap.CellPreviewHybeComboBox.blockSignals(True)
        ap.CellPreviewHybeComboBox.clear()
        for hybe in target_hybes:
            label = hybe if (hybe in same_names or not other_modality_name) else f'{hybe} ({other_modality_name})'
            ap.CellPreviewHybeComboBox.addItem(label, hybe)
        restore_index = next((i for i in range(ap.CellPreviewHybeComboBox.count())
                              if ap.CellPreviewHybeComboBox.itemData(i) == last_hybe), 0 if target_hybes else -1)
        if restore_index >= 0:
            ap.CellPreviewHybeComboBox.setCurrentIndex(restore_index)
        ap.CellPreviewHybeComboBox.blockSignals(False)
        if restore_index >= 0:
            self._show_cell_alignment_preview_for_hybe(ap.CellPreviewHybeComboBox.itemData(restore_index))

    def _show_cell_alignment_preview_for_hybe(self, target_hybe=None):
        """
        The reference hybe shown is whatever ACTUALLY anchored this
        specific target_hybe's phase-correlation at compute time --
        parsed straight out of
        cell.matrix_provenance[target_hybe]['reference_sequence']
        (format 'hybe(cell N)->reference_hybe'), NOT cell.reference_hybe
        (the segmentation hybe -- a different concept entirely, see
        compute_cell_alignment's own docstring). Earlier versions of this
        preview hardcoded cell.reference_hybe here, which silently showed
        the wrong reference crop (and therefore a meaningless comparison)
        whenever CellReferenceHybeComboBox's value at Run time differed
        from the segmentation hybe -- exactly the common case once
        cell-based alignment gained its own independent reference hybe.

        target_hybe with NO provenance entry (an anchor hybe itself, e.g.
        DNA's own within-experiment reference) has no "real anchor" to
        parse -- shown against itself (reference_hybe = target_hybe),
        which is exactly what it IS: the FOV-level matrix with no cell-
        level residual applied, so 'FOV/cross-modal' and 'final' will
        legitimately be identical.

        target_hybe (and its real anchor) may belong to EITHER the same
        modality (storage_path/hybe_records straight from
        _cell_preview_context) or the OTHER one (re-derived fresh via
        _other_modality_cell_alignment_inputs, same as at Run Cell
        Alignment time) -- always as a matching PAIR, since a
        compute_cell_alignment call only ever anchors against a hybe from
        its OWN modality's hybe_records. Every failure to resolve reports
        a real status-bar message instead of silently doing nothing.
        """
        pctx = getattr(self, '_cell_preview_context', None)
        if pctx is None:
            return
        ap = self.ui.AlignmentPanel
        if target_hybe is None:
            target_hybe = ap.CellPreviewHybeComboBox.currentData()
        if not target_hybe:
            return
        cell, fov = pctx['cell'], pctx['fov']
        storage_path, hybe_records = pctx['storage_path'], pctx['hybe_records']
        provenance = cell.matrix_provenance.get(target_hybe)
        reference_hybe = provenance['reference_sequence'].split('->')[-1] if provenance else target_hybe
        channel_type = ap.CellChannelTypeComboBox.currentText()
        pad = ap.CellPadSpinBox.value()

        same_record_by_folder = {r['folder']: r for r in hybe_records}
        same_fov_matrices = self._composed_fov_matrices_for_cell_alignment(storage_path, fov)

        if target_hybe in same_record_by_folder and reference_hybe in same_record_by_folder:
            target_storage_path = reference_storage_path = storage_path
            target_record = same_record_by_folder[target_hybe]
            reference_record = same_record_by_folder[reference_hybe]
            fov_only_matrix = same_fov_matrices.get(target_hybe, np.eye(3))
            reference_fov_matrix = same_fov_matrices.get(reference_hybe, np.eye(3))
        else:
            other = self._other_modality_cell_alignment_inputs(storage_path, fov)
            if other is None:
                self.statusBar().showMessage(
                    f"Can't preview {target_hybe}: no cross-modal alignment result found for FOV{fov:02d} yet "
                    "(run and accept Cross-Modality Alignment first).", 8000)
                return
            other_storage_path, other_hybe_records, other_fov_matrices, _ = other
            other_record_by_folder = {r['folder']: r for r in other_hybe_records}
            if target_hybe not in other_record_by_folder or reference_hybe not in other_record_by_folder:
                self.statusBar().showMessage(
                    f"Can't preview {target_hybe}: it or its reference {reference_hybe!r} isn't in either "
                    "modality's parsed layout.", 8000)
                return
            target_storage_path = reference_storage_path = other_storage_path
            target_record = other_record_by_folder[target_hybe]
            reference_record = other_record_by_folder[reference_hybe]
            fov_only_matrix = other_fov_matrices.get(target_hybe, np.eye(3))
            reference_fov_matrix = other_fov_matrices.get(reference_hybe, np.eye(3))

        reference_channel = alignment.pick_channel_by_type(reference_record, channel_type)
        target_channel = alignment.pick_channel_by_type(target_record, channel_type)
        reference_fiducial_channel = reference_record['fiducial_channel']
        target_fiducial_channel = target_record['fiducial_channel']
        final_matrix = cell.matrices.get(target_hybe, {}).get('yx', np.eye(3))

        self.alignment_preview_window.show()
        self.alignment_preview_window.raise_()
        self.preview_canvas.draw_cell_alignment_preview_3col(
            cell, fov, reference_storage_path, reference_hybe, reference_channel, reference_fiducial_channel,
            reference_fov_matrix,
            target_storage_path, target_hybe, target_channel, target_fiducial_channel,
            fov_only_matrix, final_matrix, pad=pad)

    def _refresh_cell_overlay_list(self):
        """
        Populates CellOverlayCellListWidget for whatever FOV was just typed
        into CellOverlayFovLineEdit -- reads from cell_container_permanent
        first, falling back to the transient container, exactly the same
        "already-activated cells" sources _selected_spot_cell uses. Not
        tied to the last Run Cell Alignment result -- a cell whose matrices
        were activated from vlinks.h5 (never computed interactively this
        session) shows up here too, since that's exactly the "current
        status" this whole panel is supposed to reflect.
        """
        ap = self.ui.AlignmentPanel
        ap.CellOverlayCellListWidget.clear()
        fov_text = ap.CellOverlayFovLineEdit.text().strip()
        if not fov_text:
            return
        fov = int(fov_text)
        cells = []
        if self.cell_container_permanent is not None:
            cells = self.cell_container_permanent.data.get(fov, [])
        if not cells and self.cell_container is not None:
            cells = self.cell_container.data.get(fov, [])
        for cell in cells:
            item = QtWidgets.QListWidgetItem(f'Cell {cell.id}: {len(cell.matrices)} hybe(s) aligned')
            item.setData(QtCore.Qt.UserRole, cell.id)
            ap.CellOverlayCellListWidget.addItem(item)

    def _show_cell_all_readouts_overlay(self):
        ap = self.ui.AlignmentPanel
        ip = self.ui.IngestionPanel
        storage_path = ip.StoragePathLineEdit.text().strip()
        fov_text = ap.CellOverlayFovLineEdit.text().strip()
        item = ap.CellOverlayCellListWidget.currentItem()
        if not storage_path or not fov_text or item is None or not self.hybe_records:
            QtWidgets.QMessageBox.warning(self, 'Show All-Readouts Overlay',
                                          'Set storage path (Ingestion tab), an Overlay FOV, and select a cell first.')
            return
        fov = int(fov_text)
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
        reference_record = {r['folder']: r for r in self.hybe_records}.get(cell.reference_hybe)
        if reference_record is None:
            QtWidgets.QMessageBox.warning(self, 'Show All-Readouts Overlay',
                                          f"Cell's segmentation hybe {cell.reference_hybe!r} isn't in the "
                                          "current layout.")
            return
        reference_channel = alignment.pick_channel_by_type(reference_record, channel_type)
        target_specs = self._cell_overlay_target_specs(cell, storage_path, fov, self.hybe_records, channel_type)
        self.alignment_preview_window.show()
        self.alignment_preview_window.raise_()
        self.preview_canvas.draw_cell_all_readouts_overlay(
            cell, fov, cell.reference_hybe, storage_path, reference_channel, reference_record['fiducial_channel'],
            target_specs, pad=pad)

    def _accept_cell_alignment(self):
        ap = self.ui.AlignmentPanel
        if not self._pending_cell_alignment:
            return
        ctx = self._cell_alignment_context
        fov = self._pending_cell_alignment_fov
        # the params this pending result was actually COMPUTED with, not
        # whatever the live combos happen to show now -- the user may have
        # kept reviewing (and touching channel/pad) after Run but before
        # Accept, and staged_cell.matrices below always reflects the
        # original run, never a live re-compute.
        run_params = self._pending_cell_alignment_params or {}
        channel_type = run_params.get('channel_type') or ap.CellChannelTypeComboBox.currentText()
        pad = run_params.get('pad', ap.CellPadSpinBox.value())
        for real_cell, staged_cell in self._pending_cell_alignment:
            real_cell.matrices = staged_cell.matrices
            if ctx is not None and fov is not None:
                reference_record = {r['folder']: r for r in ctx['hybe_records']}.get(real_cell.reference_hybe)
                if reference_record is None:
                    continue
                save_path = os.path.join(ctx['storage_path'], f'FOV{fov:02d}', f'cell{real_cell.id}_alignment_overlay.png')
                reference_channel = alignment.pick_channel_by_type(reference_record, channel_type)
                target_specs = self._cell_overlay_target_specs(real_cell, ctx['storage_path'], fov,
                                                                ctx['hybe_records'], channel_type)
                self.preview_canvas.draw_cell_all_readouts_overlay(
                    real_cell, fov, real_cell.reference_hybe, ctx['storage_path'], reference_channel,
                    reference_record['fiducial_channel'], target_specs, pad=pad, save_path=save_path)
        # _run_cell_alignment's manual branch sources real_cells from
        # self.cell_container (transient) for this FOV -- that's the
        # object graph real_cell.matrices just got mutated on, so that's
        # what needs to reach vlinks.h5. Same gap as automatic mode: Accept
        # updated matrices in memory but never wrote them to disk.
        storage_paths = self._all_vlinks_storage_paths()
        wrote = False
        if storage_paths and fov is not None and self.cell_container is not None and self.cell_container.data.get(fov):
            vlinks_store.mirror_write_cells(storage_paths, fov, self.cell_container)
            for path in storage_paths:
                vlinks_store.write_global_params(path, cell_alignment_reference_hybe=run_params.get('reference_hybe'),
                                                 cell_alignment_channel_type=channel_type, cell_alignment_pad=pad)
            wrote = True
        self._pending_cell_alignment = None
        self._pending_cell_alignment_fov = None
        self._pending_cell_alignment_params = None
        ap.CellAcceptPushButton.setEnabled(False)
        ap.CellRejectPushButton.setEnabled(False)
        saved_msg = 'saved to vlinks.h5; ' if wrote else ''
        QtWidgets.QMessageBox.information(self, 'Cell alignment accepted',
                                          f'Cell alignment matrices applied, {saved_msg}overlay image(s) saved.')

    def _reject_cell_alignment(self):
        ap = self.ui.AlignmentPanel
        self._pending_cell_alignment = None
        self._pending_cell_alignment_fov = None
        self._pending_cell_alignment_params = None
        ap.CellAcceptPushButton.setEnabled(False)
        ap.CellRejectPushButton.setEnabled(False)
        ap.CellResultsListWidget.clear()

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

    # -- interactive analysis console --

    def _build_analysis_namespace(self):
        """
        Return a dict that is used as the global/local namespace when the
        user runs code in the Analysis tab.  It exposes the most-used
        pipeline objects plus the standard scientific libraries so that
        typical one-liners (print, plt.show, pd.DataFrame, …) work without
        any imports.
        """
        import numpy as np   # noqa: F401 – re-imported here so the name
        import pandas as pd  # noqa: F401   is always in the returned dict
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except Exception:
            plt = None

        ns = {
            '__builtins__': __builtins__,
            'window': self,
            'cell_container': self.cell_container,
            'cell_container_permanent': self.cell_container_permanent,
            'fov_matrices': self.fov_matrices,
            'modality_data': self.modality_data,
            'modality_names': self.modality_names,
            'hybe_records': self.hybe_records,
            'cross_modal_result': self.cross_modal_result,
            'np': np,
            'pd': pd,
            'plt': plt,
        }
        return ns

    def _run_analysis_code(self):
        """
        Execute whatever is in the Analysis tab's code editor.
        If text is selected, only the selection is run.
        stdout/stderr are captured and appended to the output display.
        """
        anp = self.ui.AnalysisPanel
        cursor = anp.CodeEditor.textCursor()
        code = cursor.selectedText() if cursor.hasSelection() else anp.CodeEditor.toPlainText()
        # QPlainTextEdit uses the Unicode paragraph separator (U+2029) as a
        # line break inside selected text – normalise it to '\n' so exec()
        # sees a syntactically valid multi-line block.
        code = code.replace('\u2029', '\n').strip()
        if not code:
            return

        ns = self._build_analysis_namespace()
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), \
             contextlib.redirect_stderr(stderr_capture):
            try:
                exec(compile(code, '<analysis>', 'exec'), ns)  # noqa: S102 – intentional interactive console
            except Exception:
                traceback.print_exc(file=stderr_capture)

        output = stdout_capture.getvalue()
        errors = stderr_capture.getvalue()
        combined = (output + errors).rstrip('\n')
        if combined:
            anp.OutputDisplay.appendPlainText(combined)
        else:
            anp.OutputDisplay.appendPlainText('(no output)')

