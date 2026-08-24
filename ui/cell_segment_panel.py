from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from codelab_pipeline.segmentation.segment import PROJECTION_MODES, describe_projection


class CellSegmentPanelUI(object):
    """
    Cell segmentation: pick a reference hybe/channel + a method (Cellpose /
    Classical threshold+watershed / Manual polygon draw, each with its own
    parameter page), run in bulk (off the main thread, except Manual which
    has nothing to compute), review/hand-correct the mask in the pop-up
    displayer (canvas/cell_displayer.py -- not embedded here, see that
    module's docstring for why), then stage/save/discard through a
    transient/permanent CellContainer pair -- mirrors CellClassifier's own
    proven cell-segmentation temp/approve pattern. All three methods funnel
    into the same mask -> CellContainer.load_new_cells path; see
    cell_displayer.py's docstring for how Manual-mode drawing composes
    additively with whatever a Cellpose/Classical run already produced.
    """
    def setupUi(self, Widget):
        Widget.setObjectName('CellSegmentPanel')
        layout = QtWidgets.QVBoxLayout(Widget)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.FovSpinBox = QtWidgets.QSpinBox()
        self.FovSpinBox.setRange(1, 100000)
        self.FovSpinBox.setValue(1)
        form.addRow('FOV:', self.FovSpinBox)

        self.ReferenceHybeComboBox = QtWidgets.QComboBox()
        form.addRow('Reference hybe:', self.ReferenceHybeComboBox)

        self.ChannelComboBox = QtWidgets.QComboBox()
        form.addRow('Channel:', self.ChannelComboBox)

        self.MethodComboBox = QtWidgets.QComboBox()
        self.MethodComboBox.addItems(['Cellpose', 'Classical', 'Manual'])
        form.addRow('Method:', self.MethodComboBox)

        self.MethodStackedWidget = QtWidgets.QStackedWidget()
        layout.addWidget(self.MethodStackedWidget)
        self.MethodStackedWidget.addWidget(self._build_cellpose_page())
        self.MethodStackedWidget.addWidget(self._build_classical_page())
        self.MethodStackedWidget.addWidget(self._build_manual_page())
        self.MethodComboBox.currentIndexChanged.connect(self.MethodStackedWidget.setCurrentIndex)

        self.AppendModeCheckBox = QtWidgets.QCheckBox('Append Mode')
        layout.addWidget(self.AppendModeCheckBox)
        appendModeLabel = QtWidgets.QLabel(
            'Add to the current mask instead of replacing it -- e.g. segment one hybe, then a '
            'different cell-barcode hybe, and accumulate cells from both.')
        appendModeLabel.setWordWrap(True)
        layout.addWidget(appendModeLabel)

        projGroup = QtWidgets.QGroupBox('Projection (source image)')
        projLayout = QtWidgets.QFormLayout(projGroup)
        self.ProjectionModeComboBox = QtWidgets.QComboBox()
        self.ProjectionModeComboBox.addItems(list(PROJECTION_MODES))
        # 'MIP (stored)' first, so nothing existing changes. The
        # depth-resolved modes are available here -- not only in cytoplasmic
        # search -- per explicit request: a single/range projection is often
        # the better image to find cells in at all, whether or not any
        # nucleus has been determined yet.
        projLayout.addRow('Projection:', self.ProjectionModeComboBox)
        self.ZPlaneSpinBox = QtWidgets.QSpinBox(); self.ZPlaneSpinBox.setRange(0, 100000)
        projLayout.addRow('Plane z:', self.ZPlaneSpinBox)
        zRow = QtWidgets.QWidget(); zLayout = QtWidgets.QHBoxLayout(zRow)
        zLayout.setContentsMargins(0, 0, 0, 0)
        self.ZStartSpinBox = QtWidgets.QSpinBox(); self.ZStartSpinBox.setRange(0, 100000)
        self.ZEndSpinBox = QtWidgets.QSpinBox(); self.ZEndSpinBox.setRange(0, 100000)
        self.ViewRangePushButton = QtWidgets.QPushButton('View')
        self.ViewRangePushButton.setMaximumWidth(60)
        # Range projections are the one mode with no natural "live" trigger:
        # editing z-start or z-end mid-edit would recompute a range that the
        # user has not finished specifying, and a range read is a real
        # multi-plane stack read, not a single slice. So the range is applied
        # on demand, explicitly, unlike single-plane which updates as you
        # scroll its own spinbox.
        zLayout.addWidget(self.ZStartSpinBox); zLayout.addWidget(QtWidgets.QLabel('to'))
        zLayout.addWidget(self.ZEndSpinBox); zLayout.addWidget(self.ViewRangePushButton)
        projLayout.addRow('Range z:', zRow)
        self.AutoFocusPushButton = QtWidgets.QPushButton('Detect Focal Plane')
        projLayout.addRow(self.AutoFocusPushButton)
        self.FocusCanvas = FigureCanvasQTAgg(Figure(figsize=(4, 1.35)))
        self.FocusCanvas.setMinimumHeight(115)
        projLayout.addRow(self.FocusCanvas)
        layout.addWidget(projGroup)

        self.RunSegmentationPushButton = QtWidgets.QPushButton('Run Segmentation')
        # Label restated from the live projection on every change -- see
        # refresh_run_label. The button you are about to press should say
        # what it will actually do; a log line after the fact does not help
        # when the wrong choice costs two thirds of the cells.
        self.focus_detected = False
        layout.addWidget(self.RunSegmentationPushButton)

        self.ShowDisplayerPushButton = QtWidgets.QPushButton('Show Cell Displayer')
        self.ShowDisplayerPushButton.setCheckable(True)
        layout.addWidget(self.ShowDisplayerPushButton)

        self.ShowCytoplasmPushButton = QtWidgets.QPushButton('Cytoplasmic Segmentation...')
        # Opens its own pop-up (ui/cytoplasm_panel.py) rather than adding a
        # mode to this panel: cytoplasmic search runs AFTER nuclei already
        # exist, against a different hybe (often the other modality's), with
        # its own cell selection and parameters -- none of which the fields
        # above can express without silently changing what they mean for the
        # ordinary nucleus-segmentation flow.
        layout.addWidget(self.ShowCytoplasmPushButton)

        transientGroup = QtWidgets.QGroupBox('Transient Cell Container (this FOV, staged)')
        transientLayout = QtWidgets.QVBoxLayout(transientGroup)
        buttonsRow = QtWidgets.QWidget()
        buttonsLayout = QtWidgets.QHBoxLayout(buttonsRow)
        buttonsLayout.setContentsMargins(0, 0, 0, 0)
        self.SaveCellsPushButton = QtWidgets.QPushButton('Save')
        self.DiscardCellsPushButton = QtWidgets.QPushButton('Discard')
        self.SendPermanentPushButton = QtWidgets.QPushButton('Send Permanent to Transient')
        buttonsLayout.addWidget(self.SaveCellsPushButton)
        buttonsLayout.addWidget(self.DiscardCellsPushButton)
        buttonsLayout.addWidget(self.SendPermanentPushButton)
        transientLayout.addWidget(buttonsRow)

        transientInfoLabel = QtWidgets.QLabel(
            'Saved cells are written to vlinks.h5 automatically (both RNA and DNA storage paths, '
            'if the Alignment tab\'s Cross-Modal fields are set) and activate automatically when you '
            'return to this FOV -- no separate file save/load needed.')
        transientInfoLabel.setWordWrap(True)
        transientLayout.addWidget(transientInfoLabel)
        layout.addWidget(transientGroup)

        self.ProgressBar = QtWidgets.QProgressBar()
        layout.addWidget(self.ProgressBar)

        # The panel log boxes moved into the one combined log window (see
        # ui/log_window.py) -- the stretch keeps the controls top-anchored
        # where the log box used to soak up the leftover height.
        layout.addStretch(1)

        self.ReferenceHybeComboBox.currentIndexChanged.connect(self._on_reference_hybe_changed)

    def _build_cellpose_page(self):
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        self.DiameterSpinBox = QtWidgets.QSpinBox()
        self.DiameterSpinBox.setRange(1, 500)
        self.DiameterSpinBox.setValue(40)
        form.addRow('Diameter (px):', self.DiameterSpinBox)
        self.MinSizeSpinBox = QtWidgets.QSpinBox()
        self.MinSizeSpinBox.setRange(0, 1000000)
        self.MinSizeSpinBox.setValue(1000)
        form.addRow('Min size (px):', self.MinSizeSpinBox)
        self.MaxSizeSpinBox = QtWidgets.QSpinBox()
        self.MaxSizeSpinBox.setRange(1, 1000000)
        self.MaxSizeSpinBox.setValue(10000)
        form.addRow('Max size (px):', self.MaxSizeSpinBox)
        return page

    def _build_classical_page(self):
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        self.ClassicalAlgorithmComboBox = QtWidgets.QComboBox()
        self.ClassicalAlgorithmComboBox.addItems(['Otsu', 'Yen', 'Li', 'Triangle', 'Absolute'])
        form.addRow('Algorithm:', self.ClassicalAlgorithmComboBox)
        self.ClassicalAbsoluteCutoffSpinBox = QtWidgets.QDoubleSpinBox()
        self.ClassicalAbsoluteCutoffSpinBox.setRange(0, 1e9)
        self.ClassicalAbsoluteCutoffSpinBox.setValue(1000)
        self.ClassicalAbsoluteCutoffSpinBox.setEnabled(False)
        form.addRow('Absolute cutoff:', self.ClassicalAbsoluteCutoffSpinBox)
        self.ClassicalAlgorithmComboBox.currentTextChanged.connect(
            lambda text: self.ClassicalAbsoluteCutoffSpinBox.setEnabled(text == 'Absolute'))
        self.ClassicalMinDistanceSpinBox = QtWidgets.QSpinBox()
        self.ClassicalMinDistanceSpinBox.setRange(1, 1000)
        self.ClassicalMinDistanceSpinBox.setValue(7)
        form.addRow('Min distance between cells (px):', self.ClassicalMinDistanceSpinBox)
        self.ClassicalMinSizeSpinBox = QtWidgets.QSpinBox()
        self.ClassicalMinSizeSpinBox.setRange(0, 1000000)
        self.ClassicalMinSizeSpinBox.setValue(500)
        form.addRow('Min size (px):', self.ClassicalMinSizeSpinBox)
        self.ClassicalMaxSizeSpinBox = QtWidgets.QSpinBox()
        self.ClassicalMaxSizeSpinBox.setRange(1, 1000000)
        self.ClassicalMaxSizeSpinBox.setValue(10000)
        form.addRow('Max size (px):', self.ClassicalMaxSizeSpinBox)
        return page

    def _build_manual_page(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        label = QtWidgets.QLabel(
            'Opens the displayer with an empty mask, Manual Add Mode pre-enabled.\n'
            'In the displayer: left-click to place polygon vertices, "A" to commit\n'
            'the current polygon as a new cell, "D" to undo the last vertex.')
        label.setWordWrap(True)
        layout.addWidget(label)
        layout.addStretch()
        return page

    def populate_reference_hybe_choices(self, total_active_hybe_list):
        """
        total_active_hybe_list: [(hybe_record, modality_name), ...] --
        the union of every configured modality's active hybes, NOT one
        modality's own list -- no Modality selector on this panel any
        more. Same itemData-tagged pattern as SpotLocalizationPanel.
        populate_hybe_choices/AlignmentPanel.populate_reference_hybe_
        choices; see either for the full rationale. Preserves the
        current selection across a refresh by (folder, modality).
        """
        current = self.current_reference_hybe_key()
        self.ReferenceHybeComboBox.blockSignals(True)
        self.ReferenceHybeComboBox.clear()
        for record, modality in total_active_hybe_list:
            self.ReferenceHybeComboBox.addItem(f"{record['folder']} ({modality})", (record, modality))
        if self.ReferenceHybeComboBox.count():
            restore_index = next((i for i in range(self.ReferenceHybeComboBox.count())
                                  if self._reference_hybe_item_key(i) == current), 0)
            self.ReferenceHybeComboBox.setCurrentIndex(restore_index)
        self.ReferenceHybeComboBox.blockSignals(False)
        self._on_reference_hybe_changed()

    def _reference_hybe_item_key(self, index):
        data = self.ReferenceHybeComboBox.itemData(index)
        return (data[0]['folder'], data[1]) if data is not None else (None, None)

    def current_reference_hybe_key(self):
        data = self.ReferenceHybeComboBox.currentData()
        return (data[0]['folder'], data[1]) if data is not None else (None, None)

    def current_reference_hybe(self):
        """Real hybe folder name for whatever's currently selected, or '' if nothing is."""
        data = self.ReferenceHybeComboBox.currentData()
        return data[0]['folder'] if data is not None else ''

    def current_reference_modality(self):
        """Owning modality name for whatever's currently selected, or None if nothing is."""
        data = self.ReferenceHybeComboBox.currentData()
        return data[1] if data is not None else None

    def refresh_run_label(self):
        self.RunSegmentationPushButton.setText(
            f'Run Segmentation  [{describe_projection(*self.current_projection())}]')

    def current_projection(self):
        """(mode, z_plane, (z0, z1)) -- ready to splat into segment.read_projection."""
        return (self.ProjectionModeComboBox.currentText(),
                self.ZPlaneSpinBox.value(),
                (self.ZStartSpinBox.value(), self.ZEndSpinBox.value()))

    def set_depth(self, depth):
        top = max(depth - 1, 0)
        for box in (self.ZPlaneSpinBox, self.ZStartSpinBox, self.ZEndSpinBox):
            box.setMaximum(top)
        if depth and self.ZEndSpinBox.value() == 0:
            self.ZEndSpinBox.setValue(top)

    def show_focus_profile(self, zs, values, peak):
        self.focus_detected = True
        fig = self.FocusCanvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(zs, values, lw=1.0)
        ax.axvline(peak, color='red', ls='--', lw=1.0)
        ax.set_xlabel('z', fontsize=7); ax.set_ylabel('focus', fontsize=7)
        ax.tick_params(labelsize=6); ax.set_title(f'sharpest z={peak}', fontsize=7)
        fig.tight_layout(); self.FocusCanvas.draw()

    def current_method(self):
        """'cellpose' / 'classical' / 'manual' -- lowercase, matches segment.py's own method kwarg casing."""
        return self.MethodComboBox.currentText().lower()

    def _on_reference_hybe_changed(self):
        data = self.ReferenceHybeComboBox.currentData()
        record = data[0] if data is not None else None
        # blockSignals so clear()+addItems() reads as one atomic "channel
        # list changed" update, not a transient empty-then-refilled state
        # any downstream listener could observe mid-update
        self.ChannelComboBox.blockSignals(True)
        self.ChannelComboBox.clear()
        if record is not None:
            self.ChannelComboBox.addItems([str(c) for c in record['channels']])
            # default to the readout channel, not fiducial -- fiducial is
            # only ever the default for FOV/within-experiment alignment's
            # own computation, everywhere else (segmentation included)
            # readout is the default per explicit principle
            fiducial = record.get('fiducial_channel')
            readout_channels = [c for c in record['channels'] if c != fiducial]
            if readout_channels:
                idx = self.ChannelComboBox.findText(str(readout_channels[0]))
                if idx >= 0:
                    self.ChannelComboBox.setCurrentIndex(idx)
        self.ChannelComboBox.blockSignals(False)
        self.ChannelComboBox.currentIndexChanged.emit(self.ChannelComboBox.currentIndex())
