import numpy as np
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.cm as cm

from canvas.scale_control import ScaleControlWidget
from canvas import zoom_pan


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
    """
    spots_edited = QtCore.pyqtSignal(object)  # list of (x, y) crop-local coordinates
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
        self.spot_points = []
        self.crop_mask = None
        self.spot_color = 'red'
        self.readonly_points = []
        self._axes = None
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
        self.RemoveSpotIndexLineEdit.setPlaceholderText('spot index to remove (1-based), e.g. 3')
        self.RemoveSpotPushButton = QtWidgets.QPushButton('Remove')
        removeLayout.addWidget(self.RemoveSpotIndexLineEdit)
        removeLayout.addWidget(self.RemoveSpotPushButton)
        layout.addWidget(removeRow)

        self.RemoveSpotPushButton.clicked.connect(self._remove_by_index)

    def set_data(self, crop_image, spot_points, mask=None, color='red', readonly_points=None):
        """
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

        readonly_points: optional second list of crop-local (x, y),
        drawn for CONTEXT only -- always red, never affected by clicks/
        removal/spots_edited. FOV view uses this for already-identified,
        cell-owned spots: showing them without making them part of the
        editable pool avoids a manual edit's full-replace semantics
        (spots_edited hands back the whole current spot_points list)
        silently turning already-identified spots back into unassigned
        ones.
        """
        self.crop_image = crop_image
        self.spot_points = list(spot_points)
        self.crop_mask = mask
        self.spot_color = color
        self.readonly_points = list(readonly_points) if readonly_points else []
        self._redraw(keep_view=False)

    def _redraw(self, keep_view=True):
        if self.crop_image is None:
            return
        fig = self.canvas.figure
        saved_view = zoom_pan.capture_view(fig) if keep_view else None
        fig.clear()
        ax = fig.subplots(1, 1)
        vmin, vmax = self.ScaleControl.vmin_vmax(self.crop_image)
        im = ax.imshow(self.crop_image, cmap=cm.gray, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        if self.crop_mask is not None:
            ax.contour(self.crop_mask.astype(np.uint8), levels=[0.5], colors='yellow', linewidths=1)
        if self.readonly_points:
            title = f'{len(self.spot_points)} spot(s) + {len(self.readonly_points)} other'
        else:
            title = f'{len(self.spot_points)} spot(s)'
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        # A single batched scatter call regardless of point count -- one
        # ax.scatter([x],[y]) + one ax.text() PER POINT was fine for a
        # handful of manually-clicked spots, but is ruinously slow (each
        # Text artist does real font-layout work) once a view can
        # legitimately carry thousands of real detected spots (see
        # fov_unassigned_spots). Per-point index labels are only useful
        # at a glance for small counts anyway, so they're skipped above
        # LABEL_LIMIT rather than rendered unreadably on top of each other.
        LABEL_LIMIT = 300
        if self.readonly_points:
            xs, ys = zip(*self.readonly_points)
            ax.scatter(xs, ys, edgecolor='red', facecolor='none', s=60, linewidth=1.2)
        if self.spot_points:
            xs, ys = zip(*self.spot_points)
            ax.scatter(xs, ys, edgecolor=self.spot_color, facecolor='none', s=60, linewidth=1.2)
            if len(self.spot_points) <= LABEL_LIMIT:
                for i, (x, y) in enumerate(self.spot_points, start=1):
                    ax.text(x + 2, y - 2, str(i), color=self.spot_color, fontsize=8)
        fig.tight_layout()
        zoom_pan.restore_view(fig, saved_view)
        self._axes = ax
        self.canvas.draw()

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
            self._redraw()
            self.spots_edited.emit(list(self.spot_points))
        elif event.button == 3:
            self._remove_nearest(event.xdata, event.ydata)

    def _remove_nearest(self, x, y):
        if not self.spot_points:
            return
        dists = [((px - x) ** 2 + (py - y) ** 2) ** 0.5 for px, py in self.spot_points]
        i = int(np.argmin(dists))
        if dists[i] <= self.NEAREST_REMOVE_RADIUS:
            self.spot_points.pop(i)
            self._redraw()
            self.spots_edited.emit(list(self.spot_points))

    def _remove_by_index(self):
        text = self.RemoveSpotIndexLineEdit.text().strip()
        if not text:
            return
        try:
            index = int(text)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, 'Remove spot', 'Enter a single 1-based integer spot index.')
            return
        if not (1 <= index <= len(self.spot_points)):
            QtWidgets.QMessageBox.warning(self, 'Remove spot', f'Index out of range (1-{len(self.spot_points)}).')
            return
        self.spot_points.pop(index - 1)
        self.RemoveSpotIndexLineEdit.clear()
        self._redraw()
        self.spots_edited.emit(list(self.spot_points))
