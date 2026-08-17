from PyQt5 import QtWidgets, QtCore

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from codelab_pipeline.segmentation.segment import SEED_MODES, PROJECTION_MODES, describe_projection


class CytoplasmSegmentationWindow(QtWidgets.QMainWindow):
    """
    Pop-up control window for nucleus-seeded cytoplasmic segmentation --
    opened from Cell Segmentation, never embedded (same convention as every
    other dynamic-content view in this app, see CellDisplayer's own
    docstring).

    Drives the four-step flow:
      1. pick which of this FOV's existing cells take part as nuclei
         (all checked by default -- opting a cell OUT is the exception),
      2. set the cytoplasm hybe/channel + Cellpose parameters,
      3. Preview: draw the selected nuclei, projected into the cytoplasm
         hybe's own frame, over that hybe's image,
      4. Run: Cellpose with a synthetic nuclear channel, then Incorporate
         the result back onto the cells.

    The hybe/channel choice is deliberately its own control rather than
    inherited from Cell Segmentation: the cytoplasm hybe is generally NOT
    the hybe the nuclei were segmented in, and can belong to the other
    modality entirely (e.g. a DNA brightfield stack for RNA-segmented
    nuclei) -- which is exactly why every nucleus has to be projected
    per-cell rather than drawn at its stored coordinates.

    Pure UI: owns no data and performs no projection, segmentation, or
    merging itself. MainWindow wires every button.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Cytoplasmic Segmentation')
        self.resize(420, 620)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)

        # -- target frame --
        targetGroup = QtWidgets.QGroupBox('1. Cytoplasm image')
        targetLayout = QtWidgets.QFormLayout(targetGroup)
        self.FovSpinBox = QtWidgets.QSpinBox()
        self.FovSpinBox.setRange(1, 100000)
        self.FovSpinBox.setValue(1)
        targetLayout.addRow('FOV:', self.FovSpinBox)
        self.HybeComboBox = QtWidgets.QComboBox()
        # Holds (record, modality) in itemData, same shape Spot
        # Localization's own hybe combo uses -- the modality has to travel
        # with the pick, since a bare folder name is ambiguous for a hybe
        # that exists in both modalities (e.g. the cross-modal bridge hybe).
        targetLayout.addRow('Cytoplasm hybe:', self.HybeComboBox)
        self.ChannelComboBox = QtWidgets.QComboBox()
        targetLayout.addRow('Channel:', self.ChannelComboBox)

        self.ProjectionModeComboBox = QtWidgets.QComboBox()
        self.ProjectionModeComboBox.addItems(list(PROJECTION_MODES))
        # 'MIP (stored)' stays first/default so nothing existing changes
        # behaviour. The depth-resolved modes exist because a full-depth MIP
        # of a brightfield stack accumulates every plane's halo and visibly
        # destroys the phase contrast a cell boundary is made of.
        targetLayout.addRow('Projection:', self.ProjectionModeComboBox)
        self.ZPlaneSpinBox = QtWidgets.QSpinBox()
        self.ZPlaneSpinBox.setRange(0, 100000)
        targetLayout.addRow('Plane z:', self.ZPlaneSpinBox)
        zRangeRow = QtWidgets.QWidget()
        zRangeLayout = QtWidgets.QHBoxLayout(zRangeRow)
        zRangeLayout.setContentsMargins(0, 0, 0, 0)
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
        zRangeLayout.addWidget(self.ZStartSpinBox)
        zRangeLayout.addWidget(QtWidgets.QLabel('to'))
        zRangeLayout.addWidget(self.ZEndSpinBox)
        zRangeLayout.addWidget(self.ViewRangePushButton)
        targetLayout.addRow('Range z:', zRangeRow)
        self.AutoFocusPushButton = QtWidgets.QPushButton('Detect Focal Plane')
        # Button, not automatic on FOV change, per explicit preference --
        # the metric reads every plane, so it should be an explicit act.
        # There is deliberately no "use the middle plane" option: the middle
        # is an assumption that measurably fails (z=65 vs a real peak at
        # z=76 on FOV01/Hyb_500), and replacing it with a manual guess would
        # not be better. Detect, then adjust.
        targetLayout.addRow(self.AutoFocusPushButton)
        self.FocusCanvas = FigureCanvasQTAgg(Figure(figsize=(4, 1.35)))
        self.FocusCanvas.setMinimumHeight(115)
        targetLayout.addRow(self.FocusCanvas)
        outer.addWidget(targetGroup)

        # -- nucleus participation --
        nucleusGroup = QtWidgets.QGroupBox('2. Nuclei to seed with')
        nucleusLayout = QtWidgets.QVBoxLayout(nucleusGroup)
        selectRow = QtWidgets.QHBoxLayout()
        self.SelectAllPushButton = QtWidgets.QPushButton('Select all')
        self.SelectNonePushButton = QtWidgets.QPushButton('Select none')
        self.RefreshCellsPushButton = QtWidgets.QPushButton('Refresh')
        selectRow.addWidget(self.SelectAllPushButton)
        selectRow.addWidget(self.SelectNonePushButton)
        selectRow.addWidget(self.RefreshCellsPushButton)
        nucleusLayout.addLayout(selectRow)
        self.CellListWidget = QtWidgets.QListWidget()
        # Checkable per-cell rows, default checked -- mirrors how alleles
        # are picked in Chromatin Tracing. A cell left unchecked keeps its
        # own nucleus and reference hybe untouched, and still wins any
        # overlap against another cell's cytoplasm.
        self.CellListWidget.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        nucleusLayout.addWidget(self.CellListWidget)
        self.CellCountLabel = QtWidgets.QLabel('no cells loaded')
        nucleusLayout.addWidget(self.CellCountLabel)
        outer.addWidget(nucleusGroup)

        # -- parameters --
        paramGroup = QtWidgets.QGroupBox('3. Cellpose parameters')
        paramLayout = QtWidgets.QFormLayout(paramGroup)
        self.DiameterSpinBox = QtWidgets.QSpinBox()
        self.DiameterSpinBox.setRange(0, 1000)
        self.DiameterSpinBox.setValue(60)
        self.DiameterSpinBox.setSpecialValueText('auto')
        self.DiameterSpinBox.setSuffix(' px')
        # Cytoplasm diameter, not nucleus diameter -- deliberately its own
        # field, defaulting larger than Cell Segmentation's nucleus default.
        paramLayout.addRow('Cytoplasm diameter:', self.DiameterSpinBox)
        self.MinSizeSpinBox = QtWidgets.QSpinBox()
        self.MinSizeSpinBox.setRange(0, 10_000_000)
        self.MinSizeSpinBox.setValue(1000)
        self.MinSizeSpinBox.setSuffix(' px')
        paramLayout.addRow('Min size:', self.MinSizeSpinBox)
        self.MaxSizeSpinBox = QtWidgets.QSpinBox()
        self.MaxSizeSpinBox.setRange(0, 10_000_000)
        self.MaxSizeSpinBox.setValue(100000)
        self.MaxSizeSpinBox.setSuffix(' px')
        paramLayout.addRow('Max size:', self.MaxSizeSpinBox)
        self.SeedModeComboBox = QtWidgets.QComboBox()
        # segment.SEED_MODES order puts the measured best default first --
        # see render_nucleus_seed for the per-mode numbers this ranking
        # came from. Exposed rather than hard-coded because the margin
        # between the top two is one cell out of 100 on a single FOV,
        # which is not enough to decide for every dataset.
        self.SeedModeComboBox.addItems(list(SEED_MODES))
        paramLayout.addRow('Nucleus seed style:', self.SeedModeComboBox)
        self.NucleusDilationSpinBox = QtWidgets.QSpinBox()
        self.NucleusDilationSpinBox.setRange(0, 50)
        self.NucleusDilationSpinBox.setValue(0)
        self.NucleusDilationSpinBox.setSuffix(' px')
        # The synthetic nuclear channel is a hard-edged rendering of real
        # masks, unlike a true nuclear stain. A small dilation softens that
        # for Cellpose without changing the stored nucleus itself.
        paramLayout.addRow('Seed dilation:', self.NucleusDilationSpinBox)
        outer.addWidget(paramGroup)

        # -- actions --
        actionGroup = QtWidgets.QGroupBox('4. Run')
        actionLayout = QtWidgets.QVBoxLayout(actionGroup)
        self.PreviewNucleiPushButton = QtWidgets.QPushButton('Preview Selected Nuclei')
        actionLayout.addWidget(self.PreviewNucleiPushButton)
        self.RunPushButton = QtWidgets.QPushButton('Run Cytoplasmic Search')
        self.focus_detected = False
        actionLayout.addWidget(self.RunPushButton)
        self.IncorporatePushButton = QtWidgets.QPushButton('Incorporate Into Cells')
        # Disabled until a real run has produced a mask -- incorporating
        # nothing would silently rewrite every cell's frame fields for no
        # gain.
        self.IncorporatePushButton.setEnabled(False)
        actionLayout.addWidget(self.IncorporatePushButton)
        outer.addWidget(actionGroup)

        self.LogTextEdit = QtWidgets.QPlainTextEdit()
        self.LogTextEdit.setReadOnly(True)
        self.LogTextEdit.setMaximumHeight(110)
        outer.addWidget(self.LogTextEdit)

        self.SelectAllPushButton.clicked.connect(lambda: self.set_all_checked(True))
        self.SelectNonePushButton.clicked.connect(lambda: self.set_all_checked(False))

    # -- hybe/channel --

    def set_hybe_choices(self, entries):
        """
        entries: [(record, modality), ...] -- every active hybe across
        every configured modality, exactly what Spot Localization's own
        combo is populated from. Keeps the current pick when it survives
        the refresh.
        """
        current = self.current_hybe_key()
        self.HybeComboBox.blockSignals(True)
        self.HybeComboBox.clear()
        for record, modality in entries:
            self.HybeComboBox.addItem(f"{record['folder']} ({modality})", (record, modality))
        if current != (None, None):
            for i in range(self.HybeComboBox.count()):
                data = self.HybeComboBox.itemData(i)
                if data is not None and (data[0]['folder'], data[1]) == current:
                    self.HybeComboBox.setCurrentIndex(i)
                    break
        self.HybeComboBox.blockSignals(False)

    def current_hybe_key(self):
        data = self.HybeComboBox.currentData()
        return (data[0]['folder'], data[1]) if data is not None else (None, None)

    def current_hybe_record(self):
        data = self.HybeComboBox.currentData()
        return data[0] if data is not None else None

    def current_hybe_folder(self):
        return self.current_hybe_key()[0] or ''

    def current_hybe_modality(self):
        return self.current_hybe_key()[1]

    def set_channel_choices(self, channels):
        current = self.ChannelComboBox.currentText()
        self.ChannelComboBox.blockSignals(True)
        self.ChannelComboBox.clear()
        self.ChannelComboBox.addItems([str(c) for c in channels])
        if current and current in [str(c) for c in channels]:
            self.ChannelComboBox.setCurrentText(current)
        self.ChannelComboBox.blockSignals(False)

    def refresh_run_label(self):
        """Projection AND seed style, since both silently change the result."""
        self.RunPushButton.setText(
            f'Run Cytoplasmic Search  [{describe_projection(*self.current_projection())}, '
            f'{self.current_seed_mode()}]')

    def current_projection(self):
        """(mode, z_plane, (z0, z1)) -- ready to splat into segment.read_projection."""
        return (self.ProjectionModeComboBox.currentText(),
                self.ZPlaneSpinBox.value(),
                (self.ZStartSpinBox.value(), self.ZEndSpinBox.value()))

    def set_depth(self, depth):
        """Bound every z control to the real stack depth, keeping current values sane."""
        top = max(depth - 1, 0)
        for box in (self.ZPlaneSpinBox, self.ZStartSpinBox, self.ZEndSpinBox):
            box.setMaximum(top)
        if depth and self.ZEndSpinBox.value() == 0:
            self.ZEndSpinBox.setValue(top)

    def show_focus_profile(self, zs, values, peak):
        """Sets focus_detected (see MainWindow's own Run guard) and draws the focus-vs-depth curve so the PLATEAU is visible, not just the peak --
        a range over the plateau is generally more robust than one plane, since focus
        varies across the field and the metric only samples the centre."""
        self.focus_detected = True
        fig = self.FocusCanvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(zs, values, lw=1.0)
        ax.axvline(peak, color='red', ls='--', lw=1.0)
        ax.set_xlabel('z', fontsize=7); ax.set_ylabel('focus', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_title(f'sharpest z={peak}', fontsize=7)
        fig.tight_layout()
        self.FocusCanvas.draw()

    def current_seed_mode(self):
        return self.SeedModeComboBox.currentText()

    def current_channel(self):
        text = self.ChannelComboBox.currentText().strip()
        return int(text) if text else None

    # -- cell selection --

    def set_cell_choices(self, cells):
        """
        cells: [(cell_id, label_text), ...]. Every row starts CHECKED --
        taking part is the default, opting out is the exception. Preserves
        any existing unchecked state across a refresh so re-listing doesn't
        silently undo the user's own de-selections.
        """
        previously_unchecked = {cid for cid in self.unchecked_cell_ids()}
        self.CellListWidget.clear()
        for cell_id, label in cells:
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.UserRole, int(cell_id))
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked if cell_id in previously_unchecked
                               else QtCore.Qt.Checked)
            self.CellListWidget.addItem(item)
        self._refresh_count_label()

    def selected_cell_ids(self):
        return [self.CellListWidget.item(i).data(QtCore.Qt.UserRole)
                for i in range(self.CellListWidget.count())
                if self.CellListWidget.item(i).checkState() == QtCore.Qt.Checked]

    def unchecked_cell_ids(self):
        return [self.CellListWidget.item(i).data(QtCore.Qt.UserRole)
                for i in range(self.CellListWidget.count())
                if self.CellListWidget.item(i).checkState() != QtCore.Qt.Checked]

    def set_all_checked(self, checked):
        state = QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked
        for i in range(self.CellListWidget.count()):
            self.CellListWidget.item(i).setCheckState(state)
        self._refresh_count_label()

    def _refresh_count_label(self):
        total = self.CellListWidget.count()
        chosen = len(self.selected_cell_ids())
        self.CellCountLabel.setText(f'{chosen} of {total} cell(s) selected as nuclei')

    def log(self, message):
        self.LogTextEdit.appendPlainText(message)
