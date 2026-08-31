import re

import numpy as np
import cv2
from skimage.draw import polygon as sk_polygon
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.cm as cm

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


class CellDisplayer(QtWidgets.QMainWindow):
    """
    Pop-up (not embedded) mask reviewer for cell segmentation. Freely
    resizable rather than docked into the main panel -- per this session's
    lesson on matplotlib title/font rendering not universalizing cleanly
    across platforms (macOS/Windows Server/Linux), dynamic-content displays
    in this app get their own window, not a fixed-size panel region.

    Shows the reference image beside the labeled mask overlay (boundary
    scatter + ID text at each cell's centroid -- ports the plotting from
    legacy/segment_widgets.py's SegmentWidget, cleaned up).
    A "remove cell ID(s)" field lets the user null out bad detections from
    the mask before it's staged/saved.

    Manual Add Mode lets the user hand-draw additional cells on top of
    whatever mask is currently shown (from Cellpose, Classical, or a prior
    manual commit) -- always additive, never replaces the mask the way
    set_data does. Left-click (while enabled) places a polygon vertex on
    the mask subplot; "A" commits the accumulated vertices as a new cell
    (rasterized via skimage.draw.polygon -- a real scanline fill, not
    CellClassifier's brute-force per-pixel shapely.Point.within() scan,
    which is an O(bbox-area) perf concern this port doesn't reproduce);
    "D" undoes the last vertex. New label id = mask.max()+1; if the new
    polygon overlaps an existing label, the manual polygon wins (the user
    is explicitly hand-correcting, so their intent takes priority). These
    are wired via matplotlib's own mpl_connect (button_press_event /
    key_press_event), not QShortcut -- QShortcut would fire "A"/"D" even
    while typing into RemoveIdsLineEdit in this same window; mpl_connect
    only fires while the canvas itself has keyboard focus. CellClassifier's
    own manual mode also needs an "F" shortcut to toggle pyqtgraph's
    pan/zoom capture off so it doesn't eat clicks meant for vertex
    placement -- not needed here, since this canvas has no
    NavigationToolbar2QT to fight with in the first place.

    mask_edited fires with the updated mask (from either remove-by-ID or a
    manual commit) so the caller can keep its transient CellContainer in
    sync -- one signal, one downstream handler, for every kind of edit.
    """
    mask_edited = QtCore.pyqtSignal(object)
    # Remove-by-ID is an ID-LIST operation, not a raster edit: routing it
    # through mask_edited made MainWindow rebuild the container FROM the
    # displayed raster -- which is a PROJECTION whenever the display frame
    # differs from a cell's native frame (exactly the post-cytoplasm
    # case), and which silently no-oped whenever the segmentation context
    # wasn't the last thing loaded, leaving the removed cell alive in the
    # container to resurrect its contour on the next re-render (confirmed
    # real bug). The ids go out as-is; the container is the authority.
    ids_removed = QtCore.pyqtSignal(list)
    undo_requested = QtCore.pyqtSignal()
    redo_requested = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Cell Displayer')
        self.resize(760, 480)
        self.reference_image = None
        self.mask = None
        self._mask_axes = None
        self._manual_mode = False
        self._pending_vertices = []
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
        # main_window.py wires this to _ensure_cell_displayer_initialized --
        # the same FOV/hybe/channel-from-the-panel refresh that already runs
        # on toggle-open, but callable directly so picking up a panel change
        # (new FOV, different hybe) doesn't require closing and reopening
        # this window.
        layout.addWidget(scaleRow)

        self.ManualAddModeCheckBox = QtWidgets.QCheckBox(
            'Manual Add Mode (click canvas to focus, then: left-click=vertex, A=commit, D=undo last)')
        layout.addWidget(self.ManualAddModeCheckBox)
        self.ManualAddModeCheckBox.toggled.connect(self._set_manual_mode)

        removeRow = QtWidgets.QWidget()
        removeLayout = QtWidgets.QHBoxLayout(removeRow)
        removeLayout.setContentsMargins(0, 0, 0, 0)
        self.RemoveIdsLineEdit = QtWidgets.QLineEdit()
        self.RemoveIdsLineEdit.setPlaceholderText('cell IDs to remove, e.g. 3,7,12 or 1-10 or 3 7 12')
        self.RemoveIdsPushButton = QtWidgets.QPushButton('Remove')
        removeLayout.addWidget(self.RemoveIdsLineEdit)
        removeLayout.addWidget(self.RemoveIdsPushButton)
        self.RemoveEdgeCellsPushButton = QtWidgets.QPushButton(
            'Remove Edge Cells')
        self.RemoveEdgeCellsPushButton.setToolTip(
            'Remove every cell touching the frame boundary: they are '
            'clipped by the field of view, so their masks are partial '
            'and they align and analyse poorly.')
        removeLayout.addWidget(self.RemoveEdgeCellsPushButton)
        # Undo/redo over the CELL CONTAINER (two diff streaks, wired by
        # MainWindow) -- this widget only emits; it owns no cell state.
        self.UndoPushButton = QtWidgets.QPushButton('Undo')
        self.UndoPushButton.setEnabled(False)
        removeLayout.addWidget(self.UndoPushButton)
        self.RedoPushButton = QtWidgets.QPushButton('Redo')
        self.RedoPushButton.setEnabled(False)
        removeLayout.addWidget(self.RedoPushButton)
        self.UndoPushButton.clicked.connect(self.undo_requested.emit)
        self.RedoPushButton.clicked.connect(self.redo_requested.emit)
        layout.addWidget(removeRow)

        self.RemoveIdsPushButton.clicked.connect(self._remove_ids)
        self.RemoveEdgeCellsPushButton.clicked.connect(self._remove_edge_cells)

    def set_data(self, reference_image, mask):
        self.reference_image = reference_image
        self.mask = mask
        self._pending_vertices = []
        self._redraw(keep_view=False)

    def _redraw(self, keep_view=True):
        if self.reference_image is None or self.mask is None:
            return
        fig = self.canvas.figure
        # capture BEFORE fig.clear() -- every interaction (add/remove a
        # cell, change scale) rebuilds the axes from scratch, which would
        # otherwise silently snap back to full-extent zoom on every single
        # click. keep_view=False (only from set_data, i.e. genuinely new
        # data -- a different FOV/mask) skips this, since an old zoom
        # region is meaningless for unrelated new content.
        saved_view = zoom_pan.capture_view(fig) if keep_view else None
        fig.clear()
        ax = fig.subplots(1, 2)
        vmin, vmax = self.ScaleControl.vmin_vmax(self.reference_image)
        ax[0].imshow(self.reference_image, cmap=cm.gray, vmin=vmin, vmax=vmax)
        ax[0].set_title('Reference', fontsize=10)
        ax[0].axis('off')
        ax[1].imshow(self.reference_image, cmap=cm.gray, vmin=vmin, vmax=vmax)
        ax[1].set_title('Cell Mask', fontsize=10)
        ax[1].axis('off')
        for cell_id in np.unique(self.mask)[1:]:
            cell_mask = (self.mask == cell_id).astype(np.uint8)
            boundary = cell_mask - cv2.erode(cell_mask, np.ones((3, 3), np.uint8), iterations=1)
            y, x = np.where(boundary > 0)
            ax[1].scatter(x, y, color='yellow', s=0.5, alpha=0.5)
            ax[1].text(x.mean(), y.mean(), str(int(cell_id)), color='red', fontsize=9, ha='center', va='center')
        if self._pending_vertices:
            xs, ys = zip(*self._pending_vertices)
            ax[1].plot(list(xs) + [xs[0]], list(ys) + [ys[0]], 'c--', linewidth=1)
            ax[1].scatter(xs, ys, color='cyan', s=15, zorder=5)
        fig.tight_layout()
        zoom_pan.restore_view(fig, saved_view)
        self._mask_axes = ax[1]
        self.canvas.draw()

    def _remove_ids(self):
        text = self.RemoveIdsLineEdit.text().strip()
        if not text or self.mask is None:
            return
        try:
            ids = _parse_index_list(text)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, 'Remove cell IDs', 'Enter a comma-separated list of integer cell IDs.')
            return
        self.mask = self.mask.copy()
        self.mask[np.isin(self.mask, ids)] = 0
        self.RemoveIdsLineEdit.clear()
        self._redraw()
        # ids, not the raster -- see ids_removed's own comment. The local
        # zeroing above is only the immediate visual echo; the authoritative
        # state change (container removal, or cytoplasm-label update) is
        # the receiver's job, and a container-backed receiver re-renders
        # this displayer from the container afterwards anyway.
        self.ids_removed.emit([int(i) for i in ids])

    def _remove_edge_cells(self):
        """Remove every cell whose mask touches the frame boundary.

        A cell clipped by the field of view carries a partial mask: its
        centroid is biased inward, its area is arbitrary, and both
        alignment and analysis inherit that -- so they are removed as a
        class rather than hunted for by eye. Emits ids_removed like the
        manual path, so the container (not this widget) owns the change.
        """
        if self.mask is None:
            return
        edge = np.concatenate([self.mask[0, :], self.mask[-1, :],
                               self.mask[:, 0], self.mask[:, -1]])
        ids = sorted({int(i) for i in np.unique(edge) if int(i) != 0})
        if not ids:
            QtWidgets.QMessageBox.information(
                self, 'Remove Edge Cells',
                'No cell touches the frame boundary.')
            return
        n_total = len({int(i) for i in np.unique(self.mask) if int(i) != 0})
        if QtWidgets.QMessageBox.question(
                self, 'Remove Edge Cells',
                f'Remove {len(ids)} of {n_total} cell(s) touching the frame '
                f'boundary?\n\n{", ".join(str(i) for i in ids[:20])}'
                + (' ...' if len(ids) > 20 else ''),
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
                QtWidgets.QMessageBox.Yes) != QtWidgets.QMessageBox.Yes:
            return
        self.mask = self.mask.copy()
        self.mask[np.isin(self.mask, ids)] = 0
        self._redraw()
        self.ids_removed.emit(ids)

    def _set_manual_mode(self, on):
        self._manual_mode = on
        self._pending_vertices = []
        for cid in self._mpl_cids:
            self.canvas.mpl_disconnect(cid)
        self._mpl_cids = []
        if on:
            self.canvas.setFocus()
            self._mpl_cids = [
                self.canvas.mpl_connect('button_press_event', self._on_manual_click),
                self.canvas.mpl_connect('key_press_event', self._on_manual_key),
            ]
        self._redraw()

    def _on_manual_click(self, event):
        if event.button != 1 or event.inaxes is None or event.inaxes is not self._mask_axes:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._pending_vertices.append((event.xdata, event.ydata))
        self._redraw()

    def _on_manual_key(self, event):
        if event.key == 'd':
            if self._pending_vertices:
                self._pending_vertices.pop()
                self._redraw()
        elif event.key == 'a':
            if len(self._pending_vertices) >= 3:
                self._commit_manual_polygon()

    def _commit_manual_polygon(self):
        xs, ys = zip(*self._pending_vertices)
        rr, cc = sk_polygon(np.array(ys), np.array(xs), shape=self.mask.shape)
        new_id = int(self.mask.max()) + 1
        self.mask = self.mask.copy()
        self.mask[rr, cc] = new_id
        self._pending_vertices = []
        self._redraw()
        self.mask_edited.emit(self.mask)
