from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from canvas import spot_fit_status
from canvas.localize_3d_displayer import MAX_GRID_COLUMNS


class ChromatinTraceGridDisplayer(QtWidgets.QMainWindow):
    """
    Pop-up grid for ONE allele's chromatin trace, one tile (YX/XZ pair) per
    active hybe -- same fixed MAX_GRID_COLUMNS=8 columns, nested-GridSpec-
    per-tile approach, and canvas/spot_fit_status.draw_spot_fit_status
    reuse as canvas/localize_3d_displayer.py's Localize3DGridDisplayer
    (that class grids one spot per tile across a SELECTION; this one grids
    one HYBE per tile for a SINGLE allele -- otherwise the same "pure
    display, caller supplies already-resolved data" pop-up). MainWindow
    owns two instances, one per channel kind ('Fiducial'/'Readout'), both
    populated together by View Crop (see localization.
    build_chromatin_trace_allele's collect_debug=True path).

    Adds a Save Figure button (which Localize3DGridDisplayer doesn't have)
    -- per explicit request, the saved filename embeds the live fit-
    parameter values so a figure on disk is self-describing about which
    parameters produced it.
    """
    def __init__(self, kind):
        super().__init__()
        self.kind = kind  # 'Fiducial' or 'Readout' -- label only, no behavior difference
        self.setWindowTitle(f'Chromatin Tracing -- {kind} Grid')
        self.resize(760, 600)
        self._current_allele_label = ''
        self._current_params = {}

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)

        self.canvas = FigureCanvasQTAgg()
        # Minimum matches roughly the window's own default resize(760, 600)
        # (minus title/button-row/scrollbar chrome) -- per explicit request,
        # a SMALL allele (few hybes) should still fill the pop-up's own
        # default size instead of rendering as a tiny image in a mostly
        # empty window; a bigger allele still grows past this and scrolls.
        self.canvas.setMinimumSize(720, 540)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setWidget(self.canvas)
        layout.addWidget(self.scroll, stretch=1)

        self.SaveFigurePushButton = QtWidgets.QPushButton('Save Figure...')
        self.SaveFigurePushButton.clicked.connect(self.save_figure)
        layout.addWidget(self.SaveFigurePushButton)

    def show_fit_status_grid(self, results, allele_label='', params=None):
        """
        results: list of (cubic, centroid, title) -- one entry per active
        hybe, same shape/semantics Localize3DGridDisplayer.
        show_fit_status_grid already uses (cubic=None entries should
        already be filtered out by the caller; centroid=None still renders
        the crop, just without a circled position).

        allele_label/params: remembered for Save Figure's own default
        filename only -- no effect on what's drawn.
        """
        self._current_allele_label = allele_label
        self._current_params = dict(params) if params else {}

        fig = self.canvas.figure
        fig.clear()
        if not results:
            self._resize_canvas(self.canvas.minimumWidth(), self.canvas.minimumHeight())
            self._force_repaint()
            return
        ncols = min(len(results), MAX_GRID_COLUMNS)
        nrows_pairs = -(-len(results) // ncols)  # ceil division
        col_px, pair_px = 190, 300
        # top=0.90 (not 0.97) -- see Localize3DGridDisplayer's own comment
        # on this same value: the top row's titles sit inside this margin
        # and were getting clipped off the canvas at 0.97.
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

    def show_overlay_grid(self, entries, allele_label='', params=None):
        """
        Red-cyan fiducial overlay tiles, one per hybe: TOP = before
        fiducial alignment (each crop cut at its modality/cell-residual-
        corrected center, exactly as the trace took it), BOTTOM = after
        (the moving crop shifted by the GAUSSIAN-CENTROID drift measured
        against the reference hybe -- fiducial alignment is centroid
        matching, never image matching, so the shift IS the fit result,
        not a correlation). red = reference hybe's crop, cyan = this
        hybe's; grey/white = coincident. Hybes whose Gaussian fit failed
        are omitted by the caller per explicit spec.

        entries: [(yx_before, yx_after, zx_before, zx_after, title), ...]
        -- each an (h,w,3) RGB; the tile is a 2x2 with ZX UNDER YX (the
        fit-status grids' own YX-above-depth convention) and before |
        after as LEFT vs RIGHT columns, so the comparison reads
        side-by-side per plane.
        """
        self._current_allele_label = allele_label
        self._current_params = dict(params) if params else {}
        fig = self.canvas.figure
        fig.clear()
        if not entries:
            self._resize_canvas(self.canvas.minimumWidth(), self.canvas.minimumHeight())
            self._force_repaint()
            return
        ncols = min(len(entries), MAX_GRID_COLUMNS)
        nrows_pairs = -(-len(entries) // ncols)
        col_px, pair_px = 320, 320
        outer = fig.add_gridspec(nrows_pairs, ncols, hspace=0.55, wspace=0.4,
                                 left=0.05, right=0.98, top=0.90, bottom=0.03)
        for i, (yx_b, yx_a, zx_b, zx_a, title) in enumerate(entries):
            row_pair, col = divmod(i, ncols)
            inner = outer[row_pair, col].subgridspec(2, 2, hspace=0.08, wspace=0.08)
            axes = [[fig.add_subplot(inner[r, c]) for c in range(2)] for r in range(2)]
            # rows: YX above, ZX below (grid convention); columns:
            # before left, after right.
            for ax, img in ((axes[0][0], yx_b), (axes[0][1], yx_a),
                            (axes[1][0], zx_b), (axes[1][1], zx_a)):
                ax.imshow(img, aspect='auto')
                ax.set_xticks([])
                ax.set_yticks([])
            axes[0][0].set_title(f'{title}\nbefore', fontsize=8)
            axes[0][1].set_title('after', fontsize=8)
            if col == 0:
                axes[0][0].set_ylabel('YX', fontsize=7)
                axes[1][0].set_ylabel('ZX', fontsize=7)
        width_px = max(self.canvas.minimumWidth(), ncols * col_px)
        height_px = max(self.canvas.minimumHeight(), nrows_pairs * pair_px)
        self._resize_canvas(width_px, height_px)
        self._force_repaint()

    def save_figure(self):
        """
        Default filename embeds the allele + the live fit-parameter values
        (params passed to the most recent show_fit_status_grid call) so a
        saved figure is self-describing -- per explicit request.
        """
        p = self._current_params
        # p is MainWindow.ChromatinTracingPanel.params()'s own shape:
        # {'spad':, 'z_window':, 'max_fiducial_drift':, 'fiducial': {...},
        # 'readout': {...}} -- flatten to the cross-mode keys PLUS just
        # THIS grid's own channel ('fiducial' or 'readout'), so the
        # filename reflects the values this specific grid was actually
        # fit with, not the other channel's independently-tuned ones.
        flat = {k: v for k, v in p.items() if not isinstance(v, dict)}
        flat.update(p.get(self.kind.lower(), {}))
        param_bits = '_'.join(f'{k}{v}' for k, v in flat.items()) if flat else 'params'
        safe_allele = (self._current_allele_label or 'allele').replace(' ', '_').replace('/', '-')
        default_name = f'{safe_allele}_{self.kind.lower()}_{param_bits}.png'
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, f'Save {self.kind} Grid Figure', default_name,
                                                         'PNG image (*.png)')
        if not path:
            return
        self.canvas.figure.savefig(path)

    def _resize_canvas(self, width_px, height_px):
        dpi = self.canvas.figure.get_dpi()
        self.canvas.figure.set_size_inches(width_px / dpi, height_px / dpi)
        self.canvas.resize(width_px, height_px)

    def _force_repaint(self):
        """
        Every level forced with repaint() (synchronous), not update()
        (queued) -- per confirmed real bug: previewing a smaller allele
        (fewer active hybes) right after a bigger one left stale pixels
        from the old, larger canvas visible around the new, smaller grid.
        A queued update() on just the viewport isn't reliably enough to
        invalidate a shrunk child widget's old backing-store region on
        every platform/compositor -- repaint()ing the canvas, the
        viewport, AND this window itself immediately, in that order,
        forces the whole chain to redraw against the new (already-resized)
        geometry before anything is shown to the user.
        """
        self.canvas.draw()
        self.canvas.repaint()
        self.scroll.horizontalScrollBar().setValue(0)
        self.scroll.verticalScrollBar().setValue(0)
        self.scroll.viewport().repaint()
        self.repaint()
