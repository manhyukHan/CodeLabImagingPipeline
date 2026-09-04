import re

import numpy as np
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.cm as cm
import matplotlib.patheffects as path_effects
from mpl_toolkits.axes_grid1 import make_axes_locatable

from canvas.scale_control import ScaleControlWidget
from canvas import zoom_pan


def _parse_index_list(text):
    """'1-10', '1,2,3', '1 2 4 5', or any mix -> sorted unique ints.
    Same grammar as the FOV list field, so every multi-index box in the
    app parses identically."""
    out = []
    for chunk in re.split(r'[,\s]+', text.strip()):
        if not chunk:
            continue
        if '-' in chunk:
            a, b = chunk.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(chunk))
    return sorted(set(out))


class SpotCropDisplayer(QtWidgets.QMainWindow):
    """
    Pop-up (not embedded) crop viewer for interactive spot localization --
    same shape as canvas/cell_displayer.py (pop-up QMainWindow +
    FigureCanvasQTAgg + mpl_connect interactivity, matplotlib event-based
    per this session's decision not to port CellClassifier's pyqtgraph
    widgets), scoped to one cell's own crop (see
    ui/spot_localization_panel.py's docstring for why the interactive
    manual-click path is always single-cell, never whole-FOV).

    Deliberately knows nothing about ACell/ASpot/hybe/alignment matrices --
    it only ever deals in crop-local pixel coordinates, exactly mirroring
    how CellDisplayer only ever deals in a plain mask array. The caller
    (windows/main_window.py) is responsible for converting spot_points'
    crop-local (x, y) into raw_coordinate/coordinate (via
    codelab_pipeline.alignment.spot_mapper) and building/removing the
    actual ASpot objects on cell.spots -- this class only visualizes and
    collects raw clicks, same separation of concerns as CellDisplayer
    owning a raw label mask instead of ACell objects.

    Manual Click Mode: left-click adds a point immediately (each click IS
    a spot -- no separate commit step, matching CellClassifier's own
    manual-spot semantics exactly, unlike cell polygon drawing which needs
    an explicit commit); right-click removes the nearest point within a
    small pixel radius. spots_edited fires with the full current
    crop-local point list after every add/remove so the caller can rebuild
    cell.spots from scratch each time (simplest way to stay in sync,
    mirrors mask_edited's "hand back the whole current state" pattern).

    Every point -- both the editable spot_points AND the read-only
    readonly_points -- is labeled with a DISPLAY index supplied by the
    caller (spot_indices/readonly_indices in set_data), not one computed
    here: the caller (MainWindow) assigns GLOBAL, FOV-wide index numbers
    (stable across which cell/view happens to be open -- selecting cell
    65 shows that cell's own spots at whatever numbers they already have
    in the full-FOV count, e.g. 145,146,147..., not renumbered 1,2,3
    every time you switch views), this class just draws whatever numbers
    it's given. Falls back to local continuous numbering (1..N over
    spot_points, then continuing over readonly_points) when the caller
    doesn't pass them, so it still works standalone. Only color
    distinguishes editable vs read-only. readonly_points can be removed
    too (by typed index or right-click, same as spot_points) --
    "read-only" only ever meant "not part of spots_edited's full-replace
    payload" (so a manual FOV-view edit can't accidentally turn an
    already-identified cell spot back into an unassigned one), never
    "cannot be deleted." Removing one emits readonly_point_removed with
    whatever opaque tag the caller attached to it (see set_data), so the
    caller can find and remove the matching real spot on its own side;
    this class still never touches ACell/ASpot itself.
    """
    spots_edited = QtCore.pyqtSignal(object)  # list of (x, y) crop-local coordinates
    readonly_point_removed = QtCore.pyqtSignal(object, float, float)  # (tag, x, y)
    NEAREST_REMOVE_RADIUS = 4.0  # crop-local pixels

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Spot Crop Displayer')
        # Bigger default than the other pop-ups -- spot crops are typically
        # much smaller in native pixels than a cell mask or FOV MIP, so
        # this one benefits most from starting already-enlarged rather
        # than needing a manual resize every time it opens. Free resize
        # (both bigger and smaller) still works via the normal window
        # edges -- nothing here constrains it in either direction.
        self.resize(760, 760)
        self.crop_image = None
        # set when data arrived while hidden; paid off by showEvent
        self._redraw_deferred = False
        # the marker artists from the last draw, so update_spots can
        # replace exactly those and leave the rest of the figure alone
        self._spot_artists = []
        self.spot_points = []
        self.spot_indices = []
        self.crop_mask = None
        self.spot_color = 'red'
        self.readonly_points = []
        self.readonly_indices = []
        self.context_image = None
        self.context_masks = None
        self.context_highlight = None
        self.context_title = ''
        self._axes = None
        self._context_axes = None
        self._manual_mode = False
        self._mpl_cids = []

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)

        self.canvas = FigureCanvasQTAgg()
        self.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        layout.addWidget(self.canvas)
        zoom_pan.install_scroll_zoom(self.canvas)
        zoom_pan.install_keyboard_zoom(self.canvas)
        zoom_pan.install_drag_pan(self.canvas)

        scaleRow = QtWidgets.QWidget()
        scaleRowLayout = QtWidgets.QHBoxLayout(scaleRow)
        scaleRowLayout.setContentsMargins(0, 0, 0, 0)
        self.ScaleControl = ScaleControlWidget()
        self.ScaleControl.changed.connect(self._redraw)
        scaleRowLayout.addWidget(self.ScaleControl)
        self.ResetViewPushButton = QtWidgets.QPushButton('Reset View')
        self.ResetViewPushButton.clicked.connect(lambda: zoom_pan.reset_view(self.canvas))
        scaleRowLayout.addWidget(self.ResetViewPushButton)
        layout.addWidget(scaleRow)

        self.ManualClickModeCheckBox = QtWidgets.QCheckBox(
            'Manual Click Mode (click canvas to focus, then: left-click=add spot, right-click=remove nearest)')
        layout.addWidget(self.ManualClickModeCheckBox)
        self.ManualClickModeCheckBox.toggled.connect(self._set_manual_mode)

        removeRow = QtWidgets.QWidget()
        removeLayout = QtWidgets.QHBoxLayout(removeRow)
        removeLayout.setContentsMargins(0, 0, 0, 0)
        self.RemoveSpotIndexLineEdit = QtWidgets.QLineEdit()
        self.RemoveSpotIndexLineEdit.setPlaceholderText('spot indices to remove, e.g. 146 or 1-10 or 3 7 12')
        self.RemoveSpotPushButton = QtWidgets.QPushButton('Remove')
        removeLayout.addWidget(self.RemoveSpotIndexLineEdit)
        removeLayout.addWidget(self.RemoveSpotPushButton)
        layout.addWidget(removeRow)

        self.RemoveSpotPushButton.clicked.connect(self._remove_by_index)

    def set_data(self, crop_image, spot_points, mask=None, color='red', readonly_points=None,
                 context_image=None, context_masks=None, context_title='',
                 context_highlight=None,
                 spot_indices=None, readonly_indices=None, keep_view=False):
        """
        spot_indices/readonly_indices: optional lists of DISPLAY index
        numbers, parallel (same length/order) to spot_points/
        readonly_points -- the caller's GLOBAL, FOV-wide numbering (see
        class docstring). Falls back to local continuous 1..N numbering
        (spot_points first, then readonly_points) when omitted.

        mask: optional boolean cell-boundary array, same shape as
        crop_image -- drawn as a yellow contour (same convention as
        pipeline_canvas.py's cell-boundary overlays), for the Cell view's
        "unmasked crop + boundary line" display. None (the FOV view's raw
        MIP, which has no single cell boundary) leaves the image exactly
        as before.

        color: marker color for spot_points -- the EDITABLE list (manual
        click add/remove and spots_edited both only ever touch
        spot_points). Cell view passes the default 'red'; FOV view passes
        'yellow' for its unassigned-spot pool, so the two are visually
        distinct wherever they might appear side by side.

        readonly_points: optional list of (x, y, tag) -- tag is whatever
        opaque identifier the caller wants echoed back via
        readonly_point_removed (e.g. a cell id) AND shown next to that
        point's index as "{index} | {tag}" (e.g. "152 | 4"). NOT excluded
        from spots_edited's full-replace payload (that's still
        spot_points-only, see class docstring), but IS removable, by
        index or right-click, same as spot_points.

        context_image: optional 2D array for a LEFT-hand panel -- the
        broader raw hybe image (Cell view: that cell's own hybe MIP, wider
        than the tight spot crop; FOV view: the whole FOV's hybe MIP)
        giving spatial context for where the crop/cell actually sits.
        None (the default) skips the left panel entirely and renders the
        single-panel layout exactly as before -- this class still never
        computes anything cell/alignment-related itself, the caller
        (windows/main_window.py) is responsible for building this array
        via the appropriate vlinks MIP read.

        context_masks: optional list of (label, x_array, y_array) --
        one entry per cell to outline on the LEFT panel, already
        transformed into the CURRENT hybe's own frame by the caller (via
        ACell.get_area_in_readout), same "this class only draws plain
        arrays" separation as mask/crop_image above. Both views pass
        EVERY cell in the FOV (the panel exists to orient you, and one
        lone contour in a field of unmarked cells does not).
        Ignored when context_image is None.

        context_highlight: optional label to draw in red instead of
        yellow -- in cell view this is the cell the RIGHT panel is
        cropped to, so the left panel says WHERE you are. Purely a
        per-label colour choice at draw time: same contours, same
        labels, no extra work. None (default) draws everything yellow.

        context_title: optional title string for the LEFT panel.

        keep_view (default False, no behavior change): True preserves
        the current zoom/pan instead of resetting to the full crop --
        per confirmed real bug, every manual click (add or remove)
        round-trips through MainWindow back to this same set_data call
        to reflect the now-real ASpot-backed point, and the previous
        unconditional reset meant a user zoomed in to place several
        spots precisely had the view snap back to full-frame after
        EVERY single click. Callers doing a genuine view switch (a
        different cell/hybe/channel) should still leave this False.
        """
        self.crop_image = crop_image
        self.spot_points = list(spot_points)
        self.spot_indices = list(spot_indices) if spot_indices is not None else list(range(1, len(self.spot_points) + 1))
        self.crop_mask = mask
        self.spot_color = color
        self.readonly_points = list(readonly_points) if readonly_points else []
        start = len(self.spot_points) + 1
        self.readonly_indices = (list(readonly_indices) if readonly_indices is not None
                                 else list(range(start, start + len(self.readonly_points))))
        self.context_image = context_image
        self.context_masks = list(context_masks) if context_masks else []
        self.context_highlight = context_highlight
        self.context_title = context_title
        self._redraw(keep_view=keep_view)

    def showEvent(self, event):
        """Draw what arrived while this pop-up was hidden -- see _redraw."""
        super().showEvent(event)
        if self._redraw_deferred:
            self._redraw_deferred = False
            self._redraw(keep_view=False)

    def update_spots(self, spot_points, spot_indices=None,
                     readonly_points=None, readonly_indices=None):
        """Redraw ONLY the spot markers, onto the axes already on screen.

        A manual click changes the spot lists and NOTHING else, yet the
        full _redraw clears the figure and rebuilds everything on it --
        including the left context panel's 1024x1024 image, one contour
        per cell and one text label per cell. On a 73-cell FOV that is
        438 ms of CPU, and while a cell-alignment run had the machine
        paging at ~5000 pages/s it was 15 s of WALL time. Per click.
        None of that content changed, so none of it is touched here: the
        previous marker artists are removed and the new ones drawn on the
        same axes.

        Returns False when there is no live figure to patch -- never
        drawn yet, or hidden, or the axes were torn down -- so the caller
        can fall back to a full set_data rather than show a stale view.
        """
        if not self.isVisible() or self._axes is None or self.crop_image is None:
            return False
        ax = self._axes
        if ax.figure is not self.canvas.figure:
            return False        # figure was rebuilt under us
        self.spot_points = list(spot_points)
        self.spot_indices = (list(spot_indices) if spot_indices is not None
                             else list(range(1, len(self.spot_points) + 1)))
        if readonly_points is not None:
            self.readonly_points = list(readonly_points)
            self.readonly_indices = (list(readonly_indices)
                                     if readonly_indices is not None
                                     else list(range(len(self.readonly_points))))
        for artist in self._spot_artists:
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                pass            # already gone with a cleared figure
        self._spot_artists = []
        self._draw_spot_markers(ax)
        self.canvas.draw_idle()
        return True

    def _redraw(self, keep_view=True):
        if self.crop_image is None:
            return
        if not self.isVisible():
            # A whole-FOV render costs 3.5 s (measured, real store) and
            # set_data is reached during MainWindow construction, before
            # this pop-up has ever been shown -- rasterizing a figure
            # nobody can see was the single largest item in app startup.
            # showEvent pays this off the moment there is a viewer.
            self._redraw_deferred = True
            return
        fig = self.canvas.figure
        saved_view = zoom_pan.capture_view(fig) if keep_view else None
        fig.clear()
        if self.context_image is not None:
            ax_ctx, ax = fig.subplots(1, 2)
            self._draw_context(ax_ctx)
            self._context_axes = ax_ctx
        else:
            ax = fig.subplots(1, 1)
            self._context_axes = None
        vmin, vmax = self.ScaleControl.vmin_vmax(self.crop_image)
        im = ax.imshow(self.crop_image, cmap=cm.gray, vmin=vmin, vmax=vmax)
        # make_axes_locatable + append_axes, not fig.colorbar(im, ax=ax,
        # fraction=..., pad=...) -- the fraction/pad form only ever
        # shrinks THIS axes to make room for the colorbar, so with a
        # left context panel present (same fig.subplots(1,2) split as
        # this one) the two panels silently render at different widths
        # once only the right one loses space to its own colorbar. A
        # same-size blank axes appended to the LEFT panel via the same
        # divider mechanism (see below) keeps both panels' actual image
        # area equal instead.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)
        if self._context_axes is not None:
            left_divider = make_axes_locatable(self._context_axes)
            blank_cax = left_divider.append_axes('right', size='5%', pad=0.05)
            blank_cax.axis('off')
        if self.crop_mask is not None:
            # An integer LABEL mask (FOV view passes one) gets a contour per
            # label, so touching cells stay visually separate -- a single
            # boolean contour would trace only their merged outer hull. A
            # plain boolean mask (Cell view's single-cell crop) still takes
            # the cheap one-call path.
            arr = np.asarray(self.crop_mask)
            if arr.dtype != bool and arr.max() > 1:
                for value in np.unique(arr):
                    if value == 0:
                        continue
                    ys, xs = np.where(arr == value)
                    self._contour_one(ax, xs, ys)
            else:
                ax.contour(arr.astype(np.uint8), levels=[0.5], colors='yellow', linewidths=1)
        ax.axis('off')
        # sets the spot-count title too, so it stays correct on both the
        # full redraw and the incremental update
        self._draw_spot_markers(ax)
        fig.tight_layout()
        zoom_pan.restore_view(fig, saved_view)
        self._axes = ax
        # draw_idle, not draw -- the same choice update_spots already makes
        # a few lines up, and for a reason that was measured here.
        #
        # This redraw is expensive and it is expensive because of TEXT.
        # Sampling the live app at 100 Hz for 3 minutes while the window
        # was unresponsive: 67.7% of the GUI thread was real work, and
        # 55.6% of the whole window -- 82% of that work -- was matplotlib
        # text layout and glyph rendering, against 5.2% for every h5py
        # call combined. Benchmarked in isolation, one redraw costs 145 ms
        # with no spot labels and 1177 ms with the 300 that LABEL_LIMIT
        # allows: 8.1x.
        #
        # draw() forces a synchronous render EVERY time. Clicking through
        # hybes therefore paid for every intermediate view nobody saw:
        # measured over five rapid redraws, draw() blocked the GUI thread
        # for 5315 ms and draw_idle() for 758 ms, because Qt collapses the
        # queued repaints into one. Same picture, 7x less blocking, and it
        # is the last view -- the one actually on screen -- that gets
        # rendered.
        self.canvas.draw_idle()

    def _draw_spot_markers(self, ax):
        """The spot markers and their index labels, and nothing else.

        Factored out of _redraw so update_spots draws them the SAME way
        on an existing figure -- one implementation, so an incremental
        update can never disagree with a full redraw. Every artist it
        creates is recorded in _spot_artists so update_spots can take
        them back off again.

        A single batched scatter call regardless of point count -- one
        ax.scatter([x],[y]) + one ax.text() PER POINT was fine for a
        handful of manually-clicked spots, but is ruinously slow (each
        Text artist does real font-layout work) once a view can
        legitimately carry thousands of real detected spots (see
        fov_unassigned_spots). Per-point index labels are only useful
        at a glance for small counts anyway, so they're skipped above
        LABEL_LIMIT rather than rendered unreadably on top of each other.
        """
        LABEL_LIMIT = 300
        made = []
        total = len(self.spot_points) + len(self.readonly_points)
        if self.readonly_points:
            xs = [p[0] for p in self.readonly_points]
            ys = [p[1] for p in self.readonly_points]
            made.append(ax.scatter(xs, ys, edgecolor='red', facecolor='none',
                                   s=60, linewidth=1.2))
            if total <= LABEL_LIMIT:
                for i, p in enumerate(self.readonly_points):
                    x, y = p[0], p[1]
                    # "{index} | {tag}" (e.g. "146 | 4") -- compact, per
                    # explicit request; "152 (cell 4)" took too much space.
                    # index is the caller's GLOBAL number, not a local i+1.
                    disp = self.readonly_indices[i] if i < len(self.readonly_indices) else i + 1
                    tag_text = f' | {p[2]}' if len(p) > 2 and p[2] is not None else ''
                    made.append(ax.text(x + 2, y - 2, f'{disp}{tag_text}',
                                        color='red', fontsize=8))
        if self.spot_points:
            xs, ys = zip(*self.spot_points)
            made.append(ax.scatter(xs, ys, edgecolor=self.spot_color,
                                   facecolor='none', s=60, linewidth=1.2))
            if total <= LABEL_LIMIT:
                for i, (x, y) in enumerate(self.spot_points):
                    disp = self.spot_indices[i] if i < len(self.spot_indices) else i + 1
                    made.append(ax.text(x + 2, y - 2, str(disp),
                                        color=self.spot_color, fontsize=8))
        # the title counts spots, so it moves with them
        if self.readonly_points:
            ax.set_title(f'{len(self.spot_points)} spot(s) + '
                         f'{len(self.readonly_points)} other', fontsize=10)
        else:
            ax.set_title(f'{len(self.spot_points)} spot(s)', fontsize=10)
        self._spot_artists = made

    def _draw_context(self, ax):
        """
        LEFT panel: the broader raw hybe image with cell mask boundary(ies)
        overlaid, for orienting where the crop/cell actually sits (see
        set_data's context_image/context_masks docstring). Plain arrays
        only -- no ACell/ASpot/alignment knowledge here, matching the rest
        of this class.
        """
        vmin, vmax = self.ScaleControl.vmin_vmax(self.context_image)
        ax.imshow(self.context_image, cmap=cm.gray, vmin=vmin, vmax=vmax)
        # ONE CONTOUR PER CELL, each on its own local bbox raster.
        #
        # This used to rasterize every cell into a single shared boolean
        # array and make one contour call, on the stated assumption that
        # "real segmentation masks don't touch/overlap". That assumption is
        # false for this data: touching cells merge into one blob and the
        # contour then traces their OUTER hull only, so a cluster like
        # cells 4/5/6/7 came out as one shape with no divisions -- exactly
        # the cells you most need told apart.
        #
        # Per-cell contouring keeps every boundary, including the shared
        # ones. Cost is kept off the full frame by rasterizing each cell
        # into its own padded bbox and passing explicit X/Y coordinates, so
        # this is ~N small arrays rather than N full-frame ones.
        if self.context_masks:
            highlight = self.context_highlight
            def _is_selected(label):
                return highlight is not None and label is not None and label == highlight
            # The selected cell is drawn LAST so its boundary sits on top
            # where it touches a neighbour -- a shared edge otherwise
            # takes whichever colour happened to be drawn second.
            ordered = sorted(self.context_masks, key=lambda m: _is_selected(m[0]))
            for label, xs, ys in ordered:
                selected = _is_selected(label)
                self._contour_one(ax, xs, ys,
                                  color='red' if selected else 'yellow',
                                  linewidth=2 if selected else 1)
            for label, xs, ys in ordered:
                if label is None or len(xs) == 0:
                    continue
                # a thin black stroke keeps the label legible over both
                # the bright cell interior and the dark background now
                # that the mask itself is just a boundary line, not a
                # solid fill to sit on top of.
                txt = ax.text(float(np.mean(xs)), float(np.mean(ys)), str(label),
                              color='red' if _is_selected(label) else 'yellow',
                              fontsize=8, fontweight='bold', ha='center', va='center')
                txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
        if self.context_title:
            ax.set_title(self.context_title, fontsize=10)
        ax.axis('off')

    @staticmethod
    def _contour_one(ax, xs, ys, color='yellow', linewidth=1):
        """
        One cell's own closed boundary, rasterized only over its own
        bounding box (+1px pad so the contour closes even when the cell
        runs to the box edge) and drawn with explicit X/Y coordinates so it
        lands in true image pixel coordinates regardless of the axes'
        origin convention.
        """
        if len(xs) == 0:
            return
        ix, iy = np.asarray(xs).astype(int), np.asarray(ys).astype(int)
        x0, x1 = int(ix.min()) - 1, int(ix.max()) + 1
        y0, y1 = int(iy.min()) - 1, int(iy.max()) + 1
        local = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
        local[iy - y0, ix - x0] = 1
        # skimage.find_contours + plot, NOT ax.contour: the extraction is the
        # same marching squares, but ax.contour builds a full ContourSet
        # artist per call, and with one call per cell per panel that
        # machinery alone was ~1.2s per redraw on 102 real cells (profiled)
        # -- the dominant cost of every manual-click redraw. Plain Line2D
        # artists draw the identical boundary for ~1% of that.
        from skimage import measure
        for poly in measure.find_contours(local.astype(float), 0.5):
            ax.plot(poly[:, 1] + x0, poly[:, 0] + y0,
                    color=color, linewidth=linewidth)

    def _set_manual_mode(self, on):
        self._manual_mode = on
        for cid in self._mpl_cids:
            self.canvas.mpl_disconnect(cid)
        self._mpl_cids = []
        if on:
            self.canvas.setFocus()
            self._mpl_cids = [self.canvas.mpl_connect('button_press_event', self._on_manual_click)]

    def _on_manual_click(self, event):
        if event.inaxes is None or event.inaxes is not self._axes or event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            self.spot_points.append((float(event.xdata), float(event.ydata)))
            # No _redraw() here: spots_edited's handler (MainWindow) always
            # rebuilds this displayer with fresh GLOBAL indices, so drawing
            # first paid the full canvas twice per click -- half of the
            # measured manual-mode slowness. Same in _remove_at_local.
            self.spots_edited.emit(list(self.spot_points))
        elif event.button == 3:
            self._remove_nearest(event.xdata, event.ydata)

    def _remove_nearest(self, x, y):
        """Nearest point across BOTH spot_points and readonly_points --
        removes from whichever list it's actually in (see _remove_at_local)."""
        candidates = [(False, i, px, py) for i, (px, py) in enumerate(self.spot_points)]
        candidates += [(True, i, p[0], p[1]) for i, p in enumerate(self.readonly_points)]
        if not candidates:
            return
        dists = [((px - x) ** 2 + (py - y) ** 2) ** 0.5 for _, _, px, py in candidates]
        i = int(np.argmin(dists))
        if dists[i] <= self.NEAREST_REMOVE_RADIUS:
            in_readonly, local_pos, _, _ = candidates[i]
            self._remove_at_local(in_readonly, local_pos)

    def _remove_by_index(self):
        """
        text is a DISPLAY index -- whatever's actually printed on screen
        (the caller's global numbering, see class docstring), not a local
        list position -- resolved back to a local position via
        spot_indices/readonly_indices before removing.
        """
        text = self.RemoveSpotIndexLineEdit.text().strip()
        if not text:
            return
        try:
            wanted = _parse_index_list(text)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, 'Remove spot',
                                          'Enter spot indices as shown on screen: e.g. 146, 1-10, or 3 7 12.')
            return
        missing = [i for i in wanted
                   if i not in self.spot_indices and i not in self.readonly_indices]
        # Resolve every display index BEFORE removing anything: each removal
        # shifts local positions, so removing one-at-a-time by re-lookup
        # would delete the wrong points for any multi-index request.
        keep = [k for k, di in enumerate(self.spot_indices) if di not in wanted]
        removed_ro = [(p, di) for p, di in zip(self.readonly_points, self.readonly_indices)
                      if di in wanted]
        self.spot_points = [self.spot_points[k] for k in keep]
        self.spot_indices = [self.spot_indices[k] for k in keep]
        keep_ro = [k for k, di in enumerate(self.readonly_indices) if di not in wanted]
        self.readonly_points = [self.readonly_points[k] for k in keep_ro]
        self.readonly_indices = [self.readonly_indices[k] for k in keep_ro]
        self.RemoveSpotIndexLineEdit.clear()
        for p, _di in removed_ro:
            tag = p[2] if len(p) > 2 else None
            self.readonly_point_removed.emit(tag, p[0], p[1])
        self.spots_edited.emit(list(self.spot_points))
        if missing:
            QtWidgets.QMessageBox.warning(self, 'Remove spot',
                                          f'Not in the current view (skipped): {missing}')

    def _remove_at_local(self, in_readonly, local_pos):
        """
        local_pos: 0-based position within spot_points (in_readonly=False)
        or readonly_points (in_readonly=True) -- NOT a display index (see
        _remove_by_index/_remove_nearest, the only two callers, for how
        each resolves a click/typed display index down to this). Routes
        to the right list and fires the matching signal -- spots_edited
        (full spot_points payload) for an editable point,
        readonly_point_removed (tag, x, y) for a read-only one, so the
        caller can find and remove the corresponding real spot on its own
        side (this class still never touches ACell/ASpot).
        """
        if not in_readonly:
            if 0 <= local_pos < len(self.spot_points):
                self.spot_points.pop(local_pos)
                if local_pos < len(self.spot_indices):
                    self.spot_indices.pop(local_pos)
                    self.spots_edited.emit(list(self.spot_points))
            return
        if 0 <= local_pos < len(self.readonly_points):
            p = self.readonly_points.pop(local_pos)
            if local_pos < len(self.readonly_indices):
                self.readonly_indices.pop(local_pos)
            x, y = p[0], p[1]
            tag = p[2] if len(p) > 2 else None
            self.readonly_point_removed.emit(tag, x, y)
