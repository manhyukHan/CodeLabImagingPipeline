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
                  'min_sep': 3.0, 'multi_mode': False}


class Localize3DDisplayer(QtWidgets.QMainWindow):
    """
    3-column pop-up for 3D localization -- adding Z (and refining
    sub-pixel x,y) onto spots that are ALREADY PLACED by 2D auto-detect
    or a manual click, never a fresh from-scratch detection pass (see
    codelab_pipeline.localization.localization.refine_spot_z). Toggled
    open/closed via SpotLocalizationPanel's own button, same freely-
    resizable pop-up convention as every other interactive window in
    this app (CellDisplayer, SpotCropDisplayer, ...).

    Left column: the 7 fit parameters (bounds/rejection thresholds, plus
    min_sep -- how close two detected peaks in one spot's crop must be
    before they're treated as separate components and routed to the
    mixture fit instead of a single Gaussian; see localization.
    find_local_peaks_3d/refine_spot_z).

    Middle column: every spot currently shown in the crop displayer's
    CURRENT view (Cell view: that cell's own spots; FOV view: BOTH the
    unassigned pool AND every cell's readonly spots), as a multi-select
    list: "Spot {global_index} | Cell {unassigned|cell_id} | {Z-accepted|
    Z-rejected|Z-not run}". global_index is GLOBAL/FOV-wide, not a local
    1..N recount of just this view (see MainWindow._global_spot_order) --
    selecting a different cell shows its spots at whatever numbers they
    already hold in the full-FOV count, never renumbered from 1. The
    Z-status suffix is a plain, session-transient note (ASpot._z_status,
    not part of the persisted schema) set the moment a spot goes through
    Run or View. Run only ever processes the SELECTED rows, never silently every spot
    in view -- per explicit request, since a cell/FOV can carry far more
    spots than a user wants Z-refined in one pass. refine_spot_z itself
    already accepts cell=None for an unassigned spot (coordinate stays
    == raw_coordinate, no transform), so FOV-view rows work exactly like
    Cell-view ones from this class's own point of view -- it just hands
    back the SELECTED indices, MainWindow.-_run_3d_localize resolves
    each one back to the real (ASpot, ACell-or-None) pair and picks the
    right cell= to pass through.

    Right column: a FIXED, scrollable grid of YX+XZ pairs -- one PER
    selected spot (up to MAX_GRID_COLUMNS=8 columns; more than 8 wraps to
    additional row-PAIRS, not more columns), built around spot_fit_status.
    draw_spot_fit_status unmodified, called once per spot into its own
    nested-GridSpec-carved axes pair (an OUTER GridSpec of one cell per
    spot, each cell holding its own INNER 2-row GridSpec for that spot's
    YX/XZ pair) -- sharex only ever locks WITHIN one spot's own pair,
    never across different spots that land in the same column position
    after wrapping, and outer/inner spacing can differ (tight within a
    pair, roomier between spots) since they're two separate GridSpecs.

    Run and View act on the SAME selection but are otherwise fully
    separate concerns, per explicit request -- no checkbox governs
    either: Run actually refines and SAVES the Z (mutating, undoable)
    and never touches this grid at all; View (ViewPushButton/
    view_requested) runs the identical fit but only ever DISPLAYS the
    result here -- never touches spot.coordinate/raw_coordinate, never
    pushes undo -- a pure preview for "would this spot's Z fit succeed
    and where would the peak land" before committing to Run. A spot with
    no accepted fit still renders its crop in the grid, just without the
    yellow centroid circle (draw_spot_fit_status's own "circled = good,
    plain = missing/rejected" convention, unchanged).
    """
    run_requested = QtCore.pyqtSignal()
    view_requested = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle('3D Localization')
        self.resize(980, 560)

        central = QtWidgets.QWidget()
        outer = QtWidgets.QHBoxLayout(central)
        self.setCentralWidget(central)

        # -- left column: fit parameters --
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
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

        self.ResetDefaultsPushButton = QtWidgets.QPushButton('Reset to Defaults')
        self.ResetDefaultsPushButton.clicked.connect(self.reset_defaults)
        left_layout.addWidget(self.ResetDefaultsPushButton)

        left_layout.addStretch(1)
        outer.addWidget(left, stretch=0)

        # -- middle column: spot list + actions --
        middle = QtWidgets.QWidget()
        middle_layout = QtWidgets.QVBoxLayout(middle)
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
        actionRowLayout.addWidget(self.RunPushButton)
        actionRowLayout.addWidget(self.ViewPushButton)
        middle_layout.addWidget(actionRow)

        self.StatusLabel = QtWidgets.QLabel('')
        self.StatusLabel.setWordWrap(True)
        middle_layout.addWidget(self.StatusLabel)

        outer.addWidget(middle, stretch=1)

        # -- right column: fixed, scrollable YX/XZ grid --
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        self.canvas = FigureCanvasQTAgg()
        self.canvas.setMinimumSize(320, 240)
        # setWidgetResizable(False) + explicit canvas.resize() in
        # show_fit_status_grid -- a many-spot grid needs real per-crop
        # pixel size to stay readable, so it's allowed to grow past the
        # window's own size and scroll, rather than being squeezed to fit.
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setWidget(self.canvas)
        right_layout.addWidget(self.scroll, stretch=1)
        outer.addWidget(right, stretch=1)

    def params(self):
        return {'spad': self.SpadSpinBox.value(),
                'peak_bound': self.PeakBoundSpinBox.value(),
                'max_sigma': self.MaxSigmaSpinBox.value(),
                'max_uncert': self.MaxUncertSpinBox.value(),
                'min_hb_ratio': self.MinHBRatioSpinBox.value(),
                'min_ah_ratio': self.MinAHRatioSpinBox.value(),
                'min_sep': self.MinSepSpinBox.value(),
                'multi_mode': self.MultiModeCheckBox.isChecked()}

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

    def set_spot_choices(self, labels):
        """
        Repopulates the middle column's list from scratch -- called by
        MainWindow whenever the crop displayer's own current view changes
        (new cell/hybe/channel selected, spots added/removed/undone).
        Everything starts selected (convenient default matching the old
        "process every spot in view" behavior); the user deselects rows
        they don't want touched, rather than needing to reselect the same
        set after every single refresh.
        """
        self.SpotListWidget.clear()
        self.SpotListWidget.addItems(labels)
        self.SpotListWidget.selectAll()

    def selected_indices(self):
        """0-based row indices, matching set_spot_choices' own enumeration order."""
        return sorted(self.SpotListWidget.row(item) for item in self.SpotListWidget.selectedItems())

    def show_fit_status_grid(self, results):
        """
        results: list of (cubic, centroid, title) -- one entry per spot
        just processed (Run) or previewed (View), in the SAME order as
        the selection. Rejected/no-fit spots still get an entry (cubic
        given, centroid=None) so their crop still renders, just without
        the yellow centroid circle -- draw_spot_fit_status already treats
        centroid=None that way; only spots where the raw stack couldn't
        even be cropped (cubic itself is None, e.g. hybe never ingested)
        should be left out of `results` entirely by the caller. Always
        renders when called -- no checkbox gate (see class docstring);
        MainWindow only ever calls this from View.

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
        outer = fig.add_gridspec(nrows_pairs, ncols, hspace=0.55, wspace=0.4,
                                 left=0.03, right=0.98, top=0.97, bottom=0.03)
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
        scrolled to.
        """
        self.canvas.draw()
        self.canvas.repaint()
        self.scroll.horizontalScrollBar().setValue(0)
        self.scroll.verticalScrollBar().setValue(0)
        self.scroll.viewport().update()
