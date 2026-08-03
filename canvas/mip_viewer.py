import numpy as np
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.cm as cm

from canvas.scale_control import ScaleControlWidget
from canvas import zoom_pan


class MipViewerDisplayer(QtWidgets.QMainWindow):
    """
    General-purpose pop-up for visually spot-checking a single FOV/hybe/
    channel MIP straight out of ingestion -- "did this actually convert to
    something sane" is otherwise only answerable indirectly (via Check
    Ingestion Status's exists-and-has-datasets check, which can't catch a
    garbled or blank image). Same pop-up/FigureCanvasQTAgg/ScaleControl
    shape as every other interactive displayer in this app, freely
    resizable (no size constraints set anywhere here or in ScaleControlWidget).
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle('MIP Viewer')
        self.resize(700, 650)
        self.image = None

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

    def set_data(self, image, title=''):
        self.image = image
        self.setWindowTitle(f'MIP Viewer -- {title}' if title else 'MIP Viewer')
        self._title = title
        self._redraw(keep_view=False)

    def _redraw(self, keep_view=True):
        if self.image is None:
            return
        fig = self.canvas.figure
        saved_view = zoom_pan.capture_view(fig) if keep_view else None
        fig.clear()
        ax = fig.subplots(1, 1)
        vmin, vmax = self.ScaleControl.vmin_vmax(self.image)
        ax.imshow(self.image, cmap=cm.gray, vmin=vmin, vmax=vmax)
        ax.set_title(self._title, fontsize=10)
        ax.axis('off')
        fig.tight_layout()
        zoom_pan.restore_view(fig, saved_view)
        self.canvas.draw()
