import numpy as np
import cv2
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.cm as cm

from canvas.scale_control import ScaleControlWidget
from canvas import zoom_pan

# same categorical palette as canvas/barcode_overview_displayer.py -- one
# distinct color per celltype name, not a gradient, since celltypes are
# genuinely different unordered categories
_CATEGORICAL_COLORS = [
    (0.90, 0.10, 0.10), (0.10, 0.65, 0.90), (0.15, 0.80, 0.15), (0.95, 0.75, 0.10),
    (0.70, 0.20, 0.90), (0.95, 0.45, 0.10), (0.10, 0.85, 0.75), (0.85, 0.10, 0.55),
]


class CelltypeResultDisplayer(QtWidgets.QMainWindow):
    """
    Pop-up two-panel comparison for celltype determination results --
    mirrors canvas/cell_displayer.py's Reference | Mask layout (per
    explicit request: "we also need to see the two figure comparison like
    cell segmentation, in celltype determination too"), except the right
    panel colors each cell's boundary by its ASSIGNED CELLTYPE (with a
    legend) instead of by numeric cell ID -- the actual classification
    result is visible at a glance, not just a one-line log message.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Celltype Determination Result')
        self.resize(820, 480)
        self.reference_image = None
        self.mask = None
        self.celltype_by_id = {}

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)

        self.canvas = FigureCanvasQTAgg()
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

        self._legend_label = QtWidgets.QLabel('')
        self._legend_label.setWordWrap(True)
        layout.addWidget(self._legend_label)

    def set_data(self, reference_image, mask, celltype_by_id):
        """
        reference_image: 2D ndarray. mask: integer-labeled cell mask (same
        convention as CellDisplayer/CellContainer.load_new_cells, 0 =
        background). celltype_by_id: {cell_id(int): celltype_name(str)} --
        cell IDs missing from this dict (not yet classified) are drawn in
        gray, called out in the legend as "(unclassified)".
        """
        self.reference_image = reference_image
        self.mask = mask
        self.celltype_by_id = celltype_by_id
        self._redraw(keep_view=False)

    def _redraw(self, keep_view=True):
        if self.reference_image is None or self.mask is None:
            return
        fig = self.canvas.figure
        saved_view = zoom_pan.capture_view(fig) if keep_view else None
        fig.clear()
        ax = fig.subplots(1, 2)
        vmin, vmax = self.ScaleControl.vmin_vmax(self.reference_image)
        ax[0].imshow(self.reference_image, cmap=cm.gray, vmin=vmin, vmax=vmax)
        ax[0].set_title('Reference', fontsize=10)
        ax[0].axis('off')
        ax[1].imshow(self.reference_image, cmap=cm.gray, vmin=vmin, vmax=vmax)
        ax[1].set_title('Celltype', fontsize=10)
        ax[1].axis('off')

        names = sorted(set(v for v in self.celltype_by_id.values() if v))
        color_by_name = {name: _CATEGORICAL_COLORS[i % len(_CATEGORICAL_COLORS)] for i, name in enumerate(names)}
        counts = {name: 0 for name in names}
        counts['(unclassified)'] = 0
        for cell_id in np.unique(self.mask)[1:]:
            cell_id = int(cell_id)
            name = self.celltype_by_id.get(cell_id)
            color = color_by_name.get(name, (0.55, 0.55, 0.55))
            counts[name if name in color_by_name else '(unclassified)'] += 1
            cell_mask = (self.mask == cell_id).astype(np.uint8)
            boundary = cell_mask - cv2.erode(cell_mask, np.ones((3, 3), np.uint8), iterations=1)
            y, x = np.where(boundary > 0)
            ax[1].scatter(x, y, color=color, s=0.5, alpha=0.7)

        legend_parts = []
        for name in names:
            rgb_hex = '#%02x%02x%02x' % tuple(int(c * 255) for c in color_by_name[name])
            legend_parts.append(f'<span style="color:{rgb_hex}">&#9632;</span> {name} ({counts[name]})')
        if counts['(unclassified)']:
            legend_parts.append(f'<span style="color:#8c8c8c">&#9632;</span> (unclassified) ({counts["(unclassified)"]})')
        self._legend_label.setText('  '.join(legend_parts))

        fig.tight_layout()
        zoom_pan.restore_view(fig, saved_view)
        self.canvas.draw()
