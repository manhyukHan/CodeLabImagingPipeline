import os

from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg


class AlignmentPreviewWindow(QtWidgets.QMainWindow):
    """
    Pop-up (not embedded) host for a cell-based alignment before/after
    preview -- same reasoning as canvas/cell_displayer.py: freely resizable
    rather than squeezed into the docked alignment panel, since this app
    must render correctly on macOS/Windows Server/Linux and fixed-size
    embedded plot regions are exactly where font/DPI rendering has broken
    before.

    Two display modes share one window, swapped by a QStackedWidget:

    - `canvas` (index 0): the live matplotlib canvas PipelineCanvas draws
      into -- the interactive previews that must be computed now.
    - `image` (index 1): a plain scaled PNG view, for showing an overlay
      figure that was ALREADY rendered and saved. Rebuilding a saved
      cell overlay costs ~35 s of raw-stack reads to produce a
      byte-identical picture, so the viewer displays the file instead
      (see MainWindow._show_cell_all_readouts_overlay).
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Cell Alignment Preview')
        self.resize(900, 550)
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)

        self.stack = QtWidgets.QStackedWidget()
        self.canvas = FigureCanvasQTAgg()
        self.stack.addWidget(self.canvas)

        self.imageLabel = QtWidgets.QLabel()
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imageLabel.setMinimumSize(1, 1)
        self.imageScroll = QtWidgets.QScrollArea()
        self.imageScroll.setWidgetResizable(True)
        self.imageScroll.setWidget(self.imageLabel)
        self.stack.addWidget(self.imageScroll)

        layout.addWidget(self.stack)
        self._pixmap = None

    # -- mode switching ---------------------------------------------------

    def show_canvas(self):
        """Switch to the live matplotlib canvas (drawing path)."""
        self.stack.setCurrentWidget(self.canvas)
        self.show()

    def show_image(self, path, title=None):
        """
        Display an already-saved PNG. Returns False (and leaves the
        window untouched) if the file can't be loaded, so the caller can
        fall back to rendering.
        """
        if not path or not os.path.exists(path):
            return False
        pixmap = QtGui.QPixmap(path)
        if pixmap.isNull():
            return False
        self._pixmap = pixmap
        self.setWindowTitle(title or 'Cell Alignment Preview -- saved overlay')
        self._rescale()
        self.stack.setCurrentWidget(self.imageScroll)
        self.show()
        return True

    # -- scaling ----------------------------------------------------------

    def _rescale(self):
        if self._pixmap is None:
            return
        area = self.imageScroll.viewport().size()
        if area.width() < 2 or area.height() < 2:
            self.imageLabel.setPixmap(self._pixmap)
            return
        # KeepAspectRatio + SmoothTransformation: the saved figure is a
        # dense multi-panel composite, so downscaling without smoothing
        # aliases the per-hybe crops into noise.
        self.imageLabel.setPixmap(self._pixmap.scaled(
            area, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.stack.currentWidget() is self.imageScroll:
            self._rescale()
