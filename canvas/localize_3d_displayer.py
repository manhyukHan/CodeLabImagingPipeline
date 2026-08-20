from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from canvas import spot_fit_status

MAX_GRID_COLUMNS = 8

# ChrTracer3's own real values (Pars Fit Spots.csv / FitPsf3D.m), not
# arbitrary starting points -- see fit_gaussian_3d's own docstring. Single
# source of truth for both the spinboxes' initial values (__init__) and
# what ResetDefaultsPushButton restores them to -- keeps the two from
# silently drifting apart if one is ever tuned without the other.
DEFAULT_PARAMS = {'spad': 5, 'peak_bound': 2.0, 'max_sigma': 2.5,
                  'max_uncert': 2.0, 'min_hb_ratio': 1.15, 'min_ah_ratio': 0.15,
                  'min_sep': 3.0, 'multi_mode': False, 'z_window': 15}


class Localize3DDisplayer(QtWidgets.QMainWindow):
    """
    2-panel pop-up for 3D localization -- adding Z (and refining
    sub-pixel x,y) onto spots that are ALREADY PLACED by 2D auto-detect
    or a manual click, never a fresh from-scratch detection pass (see
    codelab_pipeline.localization.localization.refine_spot_z). Toggled
    open/closed via SpotLocalizationPanel's own button, same freely-
    resizable pop-up convention as every other interactive window in
    this app (CellDisplayer, SpotCropDisplayer, ...).

    Per explicit request, the YX/XZ fit-status grid this class used to
    render as a 3rd column now lives in its own SEPARATE pop-up
    (Localize3DGridDisplayer, below) -- MainWindow owns one instance of
    each and wires this class's Run/View/Show Crop actions to populate
    the grid window directly, rather than this window drawing it itself.
    This class now only ever holds the two panels below, each in its own
    QGroupBox inside a QSplitter so either can be resized independently.

    Params panel (QGroupBox): the 7 fit parameters (bounds/rejection
    thresholds, plus min_sep -- how close two detected peaks in one
    spot's crop must be before they're treated as separate components
    and routed to the mixture fit instead of a single Gaussian; see
    localization.find_local_peaks_3d/refine_spot_z).

    Spots panel (QGroupBox): every spot currently shown in the crop
    displayer's CURRENT view (Cell view: that cell's own spots; FOV
    view: BOTH the unassigned pool AND every cell's readonly spots), as
    a multi-select list: "Spot {global_index} | Cell {unassigned|
    cell_id} | {Z-accepted|Z-rejected|Z-not run}". global_index is
    GLOBAL/FOV-wide, not a local 1..N recount of just this view (see
    MainWindow._global_spot_order) -- selecting a different cell shows
    its spots at whatever numbers they already hold in the full-FOV
    count, never renumbered from 1. The Z-status suffix is a plain,
    session-transient note (ASpot._z_status, not part of the persisted
    schema) set the moment a spot goes through Run or View. Run only
    ever processes the SELECTED rows, never silently every spot in view
    -- per explicit request, since a cell/FOV can carry far more spots
    than a user wants Z-refined in one pass. refine_spot_z itself
    already accepts cell=None for an unassigned spot (coordinate stays
    == raw_coordinate, no transform), so FOV-view rows work exactly like
    Cell-view ones from this class's own point of view -- it just hands
    back the SELECTED indices, MainWindow._run_3d_localize resolves
    each one back to the real (ASpot, ACell-or-None) pair and picks the
    right cell= to pass through.

    Run and View act on the SAME selection but are otherwise fully
    separate concerns, per explicit request -- no checkbox governs
    either: Run actually refines and SAVES the Z (mutating, undoable)
    and never touches the grid pop-up at all; View (ViewPushButton/
    view_requested) runs the identical fit but only ever DISPLAYS the
    result there -- never touches spot.coordinate/raw_coordinate, never
    pushes undo -- a pure preview for "would this spot's Z fit succeed
    and where would the peak land" before committing to Run. A spot with
    no accepted fit still renders its crop in the grid, just without the
    yellow centroid circle (draw_spot_fit_status's own "circled = good,
    plain = missing/rejected" convention, unchanged).

    Show Crop (ShowCropPushButton/show_crop_requested) is a THIRD, even
    cheaper action on the same selection -- per explicit request, View
    still always runs the real fit (single or mixture) to produce its
    preview, which is exactly the expensive part someone wants to skip
    when all they want is a quick look at the raw crop before deciding
    whether fitting it is even worth doing. It draws whatever Z/mixture
    result a spot ALREADY has saved (no new fit), and nothing at all for
    a spot that's never been refined -- see MainWindow._show_3d_crop_
    only's own docstring.
    """
    run_requested = QtCore.pyqtSignal()
    view_requested = QtCore.pyqtSignal()
    show_crop_requested = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle('3D Localization')
        self.resize(660, 560)

        central = QtWidgets.QWidget()
        outer_layout = QtWidgets.QVBoxLayout(central)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(central)
        # QSplitter (not a plain QHBoxLayout) holding two QGroupBox
        # panels -- per confirmed real request, each panel needs a clear
        # visual boundary/label AND a drag handle so either can be
        # widened/narrowed independently (e.g. the Spots panel's own
        # list needs more width to read long "Spot N | Cell M | Z-..."
        # labels without truncation). Same splitter pattern already used
        # in canvas/cell_spot_status_displayer.py's own Cells/Spots
        # panels.
        outer = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        outer_layout.addWidget(outer)

        # -- params panel --
        paramsGroup = QtWidgets.QGroupBox('Fit Parameters')
        left_layout = QtWidgets.QVBoxLayout(paramsGroup)
        form = QtWidgets.QFormLayout()
        left_layout.addLayout(form)

        def double_spin(default, minv, maxv, step=0.1, decimals=2):
            sb = QtWidgets.QDoubleSpinBox()
            sb.setDecimals(decimals)
            sb.setRange(minv, maxv)
            sb.setSingleStep(step)
            sb.setValue(default)
            return sb

        # Initial values come from DEFAULT_PARAMS (module-level, see its
        # own comment) -- ResetDefaultsPushButton below restores the same
        # dict, so the two can never silently drift apart.
        self.SpadSpinBox = QtWidgets.QSpinBox()
        self.SpadSpinBox.setRange(1, 50)
        self.SpadSpinBox.setValue(DEFAULT_PARAMS['spad'])
        form.addRow('Crop half-width (px):', self.SpadSpinBox)

        self.PeakBoundSpinBox = double_spin(DEFAULT_PARAMS['peak_bound'], 0.5, 10.0)
        form.addRow('Peak bound (px from seed):', self.PeakBoundSpinBox)

        self.MaxSigmaSpinBox = double_spin(DEFAULT_PARAMS['max_sigma'], 0.5, 10.0)
        form.addRow('Max sigma (px, xy):', self.MaxSigmaSpinBox)

        self.MaxUncertSpinBox = double_spin(DEFAULT_PARAMS['max_uncert'], 0.1, 10.0)
        form.addRow('Max uncertainty (95% CI, px):', self.MaxUncertSpinBox)

        self.MinHBRatioSpinBox = double_spin(DEFAULT_PARAMS['min_hb_ratio'], 1.0, 10.0)
        form.addRow('Min peak/background ratio:', self.MinHBRatioSpinBox)

        self.MinAHRatioSpinBox = double_spin(DEFAULT_PARAMS['min_ah_ratio'], 0.0, 1.0)
        form.addRow('Min amplitude/peak ratio:', self.MinAHRatioSpinBox)

        self.MinSepSpinBox = double_spin(DEFAULT_PARAMS['min_sep'], 0.5, 20.0)
        form.addRow('Min separation (px, merge closer peaks):', self.MinSepSpinBox)

        # Default OFF (single-Gaussian fit only) -- per explicit request:
        # the multi-Gaussian mixture fit (fit_gaussian_mixture_3d) is
        # meaningfully slower than a single fit_gaussian_3d, and most
        # spots are isolated single blobs that never needed it. Mixture
        # mode (find_local_peaks_3d + fit_gaussian_mixture_3d when a
        # crop's own crowded, see localization.refine_spot_z) is now an
        # explicit opt-in for the crowded-crop case it was built for.
        self.MultiModeCheckBox = QtWidgets.QCheckBox('Multi-Gaussian mixture mode (slower -- crowded crops only)')
        self.MultiModeCheckBox.setChecked(DEFAULT_PARAMS['multi_mode'])
        form.addRow(self.MultiModeCheckBox)

        # How far in Z a second mixture component may be found from this
        # spot's own coarse Z, before the search even runs (see
        # localization.refine_spot_z's own z_window docstring) -- a real
        # crop's raw Z-stack is typically 100+ planes, so an unrestricted
        # search treats any brightish voxel anywhere in that range as a
        # candidate "second component."
        self.ZWindowSpinBox = QtWidgets.QSpinBox()
        self.ZWindowSpinBox.setRange(1, 200)
        self.ZWindowSpinBox.setValue(DEFAULT_PARAMS['z_window'])
        form.addRow('Z search window (+/-px, mixture mode only):', self.ZWindowSpinBox)

        self.ResetDefaultsPushButton = QtWidgets.QPushButton('Reset to Defaults')
        self.ResetDefaultsPushButton.clicked.connect(self.reset_defaults)
        left_layout.addWidget(self.ResetDefaultsPushButton)

        left_layout.addStretch(1)
        outer.addWidget(paramsGroup)

        # -- spots panel --
        spotsGroup = QtWidgets.QGroupBox('Spots')
        middle_layout = QtWidgets.QVBoxLayout(spotsGroup)
        middle_layout.addWidget(QtWidgets.QLabel('Spots in current view (select which to refine):'))

        self.SpotListWidget = QtWidgets.QListWidget()
        self.SpotListWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        middle_layout.addWidget(self.SpotListWidget, stretch=1)

        selectRow = QtWidgets.QWidget()
        selectRowLayout = QtWidgets.QHBoxLayout(selectRow)
        selectRowLayout.setContentsMargins(0, 0, 0, 0)
        self.SelectAllPushButton = QtWidgets.QPushButton('Select All')
        self.SelectAllPushButton.clicked.connect(self.SpotListWidget.selectAll)
        self.SelectNonePushButton = QtWidgets.QPushButton('Select None')
        self.SelectNonePushButton.clicked.connect(self.SpotListWidget.clearSelection)
        selectRowLayout.addWidget(self.SelectAllPushButton)
        selectRowLayout.addWidget(self.SelectNonePushButton)
        middle_layout.addWidget(selectRow)

        actionRow = QtWidgets.QWidget()
        actionRowLayout = QtWidgets.QHBoxLayout(actionRow)
        actionRowLayout.setContentsMargins(0, 0, 0, 0)
        self.RunPushButton = QtWidgets.QPushButton('Run (refine + save Z)')
        self.RunPushButton.clicked.connect(self.run_requested.emit)
        self.ViewPushButton = QtWidgets.QPushButton('View (crop + fit status only, nothing saved)')
        self.ViewPushButton.clicked.connect(self.view_requested.emit)
        self.ShowCropPushButton = QtWidgets.QPushButton('Show Crop (no fit, instant)')
        self.ShowCropPushButton.clicked.connect(self.show_crop_requested.emit)
        actionRowLayout.addWidget(self.RunPushButton)
        actionRowLayout.addWidget(self.ViewPushButton)
        actionRowLayout.addWidget(self.ShowCropPushButton)
        middle_layout.addWidget(actionRow)

        self.StatusLabel = QtWidgets.QLabel('')
        self.StatusLabel.setWordWrap(True)
        middle_layout.addWidget(self.StatusLabel)

        outer.addWidget(spotsGroup)
        outer.setStretchFactor(0, 0)
        outer.setStretchFactor(1, 1)
        outer.setSizes([300, 360])

    def params(self):
        return {'spad': self.SpadSpinBox.value(),
                'peak_bound': self.PeakBoundSpinBox.value(),
                'max_sigma': self.MaxSigmaSpinBox.value(),
                'max_uncert': self.MaxUncertSpinBox.value(),
                'min_hb_ratio': self.MinHBRatioSpinBox.value(),
                'min_ah_ratio': self.MinAHRatioSpinBox.value(),
                'min_sep': self.MinSepSpinBox.value(),
                'multi_mode': self.MultiModeCheckBox.isChecked(),
                'z_window': self.ZWindowSpinBox.value()}

    def reset_defaults(self):
        """Restores all fields to DEFAULT_PARAMS -- undoes any manual
        tuning in this session, same values the fields start at on open."""
        self.SpadSpinBox.setValue(DEFAULT_PARAMS['spad'])
        self.PeakBoundSpinBox.setValue(DEFAULT_PARAMS['peak_bound'])
        self.MaxSigmaSpinBox.setValue(DEFAULT_PARAMS['max_sigma'])
        self.MaxUncertSpinBox.setValue(DEFAULT_PARAMS['max_uncert'])
        self.MinHBRatioSpinBox.setValue(DEFAULT_PARAMS['min_hb_ratio'])
        self.MinAHRatioSpinBox.setValue(DEFAULT_PARAMS['min_ah_ratio'])
        self.MinSepSpinBox.setValue(DEFAULT_PARAMS['min_sep'])
        self.MultiModeCheckBox.setChecked(DEFAULT_PARAMS['multi_mode'])
        self.ZWindowSpinBox.setValue(DEFAULT_PARAMS['z_window'])

    def set_spot_choices(self, labels, keep_selected=None):
        """
        Repopulates the spots panel's list from scratch -- called by
        MainWindow whenever the crop displayer redraws (new cell/hybe/
        channel selected, OR spots added/removed/undone/refined within
        the SAME view). Without keep_selected, everything starts selected
        (convenient default matching the old "process every spot in
        view" behavior) -- used for a genuine view switch, where an old
        selection has nothing meaningful to map onto.

        keep_selected (optional): 0-based row indices, in THIS call's own
        `labels` numbering, to select instead of defaulting to all --
        the caller (MainWindow._refresh_localize_3d_spot_choices) resolves
        these from the PREVIOUS selection by spot identity, not row
        position (row numbers shift when spots are added/removed/
        reordered). Per confirmed real bug: this repopulation used to
        unconditionally selectAll() every single time, including redraws
        triggered by something else entirely (e.g. adding a manual spot,
        or Run's own end-of-batch refresh) -- silently discarding
        whatever subset the user had deliberately selected moments
        earlier, so a SUBSEQUENT Run/View would act on every spot in
        view again, not just the ones the user actually picked.
        """
        self.SpotListWidget.clear()
        self.SpotListWidget.addItems(labels)
        if keep_selected:
            for i in keep_selected:
                if 0 <= i < self.SpotListWidget.count():
                    self.SpotListWidget.item(i).setSelected(True)
        else:
            self.SpotListWidget.selectAll()

    def selected_indices(self):
        """0-based row indices, matching set_spot_choices' own enumeration order."""
        return sorted(self.SpotListWidget.row(item) for item in self.SpotListWidget.selectedItems())


class Localize3DGridDisplayer(QtWidgets.QMainWindow):
    """
    Detached pop-up (per explicit request -- this used to be
    Localize3DDisplayer's own 3rd column) holding the FIXED, scrollable
    grid of YX+XZ pairs -- one PER spot most recently processed by View
    or Show Crop (up to MAX_GRID_COLUMNS=8 columns; more than 8 wraps to
    additional row-PAIRS, not more columns), built around spot_fit_
    status.draw_spot_fit_status unmodified, called once per spot into
    its own nested-GridSpec-carved axes pair (an OUTER GridSpec of one
    cell per spot, each cell holding its own INNER 2-row GridSpec for
    that spot's YX/XZ pair) -- sharex only ever locks WITHIN one spot's
    own pair, never across different spots that land in the same column
    position after wrapping, and outer/inner spacing can differ (tight
    within a pair, roomier between spots) since they're two separate
    GridSpecs.

    MainWindow owns one instance of this alongside Localize3DDisplayer
    and shows/raises it whenever View or Show Crop populates it -- Run
    never touches it at all (see Localize3DDisplayer's own docstring on
    why). Deliberately knows nothing about spots/cells/fitting itself,
    same "pure display, caller supplies already-resolved data" separation
    every other displayer in this app follows.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle('3D Localization -- Fit Status Grid')
        self.resize(700, 560)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)

        self.canvas = FigureCanvasQTAgg()
        self.canvas.setMinimumSize(320, 240)
        # setWidgetResizable(False) + explicit canvas.resize() in
        # show_fit_status_grid -- a many-spot grid needs real per-crop
        # pixel size to stay readable, so it's allowed to grow past the
        # window's own size and scroll, rather than being squeezed to fit.
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setWidget(self.canvas)
        layout.addWidget(self.scroll, stretch=1)

    def show_fit_status_grid(self, results):
        """
        results: list of (cubic, centroid, title) -- one entry per spot
        just previewed (View) or shown (Show Crop), in the SAME order as
        the selection. Rejected/no-fit spots still get an entry (cubic
        given, centroid=None) so their crop still renders, just without
        the yellow centroid circle -- draw_spot_fit_status already treats
        centroid=None that way; only spots where the raw stack couldn't
        even be cropped (cubic itself is None, e.g. hybe never ingested)
        should be left out of `results` entirely by the caller.

        Max MAX_GRID_COLUMNS columns -- more than that many results wrap
        to additional YX/XZ row-PAIRS, not more columns. Each spot gets
        its own nested GridSpec (an outer per-spot cell, an inner 2-row
        split for that spot's own YX/XZ pair) so sharex only ever locks
        within one spot's pair, and outer (between-spot) spacing can be
        generous while inner (within-pair) spacing stays tight.
        """
        fig = self.canvas.figure
        fig.clear()
        if not results:
            self._resize_canvas(self.canvas.minimumWidth(), self.canvas.minimumHeight())
            self._force_repaint()
            return
        ncols = min(len(results), MAX_GRID_COLUMNS)
        nrows_pairs = -(-len(results) // ncols)  # ceil division
        # Real per-crop pixel budget: a title line + two roughly-square
        # small images (YX, XZ) needs real room, not a sliver -- the
        # earlier flat single-hspace GridSpec packed titles and images
        # from adjacent row-pairs into each other once more than one
        # pair-row was needed. outer spacing (between different spots)
        # is generous; inner spacing (within one spot's own YX/XZ pair)
        # stays tight so the pair still reads as one connected crop.
        col_px, pair_px = 190, 300
        # top=0.90 (not 0.97) -- the top row's own titles sit just above
        # ax_yx, inside that same top margin; 0.97 left too little room and
        # clipped them off the very top of the canvas (confirmed real bug).
        outer = fig.add_gridspec(nrows_pairs, ncols, hspace=0.55, wspace=0.4,
                                 left=0.03, right=0.98, top=0.90, bottom=0.03)
        for i, (cubic, centroid, title) in enumerate(results):
            row_pair, col = divmod(i, ncols)
            inner = outer[row_pair, col].subgridspec(2, 1, hspace=0.08)
            ax_yx = fig.add_subplot(inner[0])
            ax_xz = fig.add_subplot(inner[1], sharex=ax_yx)
            spot_fit_status.draw_spot_fit_status(ax_yx, ax_xz, cubic, centroid=centroid,
                                                 title=title, title_fontsize=8)
        width_px = max(self.canvas.minimumWidth(), ncols * col_px)
        height_px = max(self.canvas.minimumHeight(), nrows_pairs * pair_px)
        self._resize_canvas(width_px, height_px)
        self._force_repaint()

    def _resize_canvas(self, width_px, height_px):
        """
        Explicit pixel sizing (not fig.tight_layout(), which fights a
        QScrollArea in setWidgetResizable(False) mode) -- each crop gets
        a fixed on-screen size regardless of grid extent, so a many-spot
        grid scrolls instead of squeezing every crop unreadably small to
        fit the window.
        """
        dpi = self.canvas.figure.get_dpi()
        self.canvas.figure.set_size_inches(width_px / dpi, height_px / dpi)
        self.canvas.resize(width_px, height_px)

    def _force_repaint(self):
        """
        canvas.draw() alone (a synchronous Agg re-render into the
        canvas's OWN backing buffer) doesn't reliably force Qt to
        repaint the QScrollArea's viewport when the canvas just shrank
        (setWidgetResizable(False) mode) -- observed as the PREVIOUS
        grid's content/titles still visible, overlapping the new one,
        until some unrelated repaint happened to occur. repaint() forces
        an immediate, synchronous full repaint instead of a merely
        scheduled one; resetting scroll position also means a fresh
        selection's grid always starts visible from its own top-left,
        not wherever the previous (possibly much larger) grid had been
        scrolled to. viewport().repaint() (not just canvas.repaint()) --
        a canvas.repaint() alone only guarantees the canvas's OWN new
        (shrunk) rect gets redrawn; the surrounding viewport area the
        previous, larger canvas used to cover needs its own forced,
        synchronous repaint too, or stale pixels from that larger
        previous grid can persist there. self.repaint() covers the same
        risk one level up, at the whole pop-up window.
        """
        self.canvas.draw()
        self.canvas.repaint()
        self.scroll.horizontalScrollBar().setValue(0)
        self.scroll.verticalScrollBar().setValue(0)
        self.scroll.viewport().repaint()
        self.repaint()
