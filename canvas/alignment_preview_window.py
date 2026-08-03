from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg


class AlignmentPreviewWindow(QtWidgets.QMainWindow):
    """
    Pop-up (not embedded) host for a cell-based alignment before/after
    preview -- same reasoning as canvas/cell_displayer.py: freely resizable
    rather than squeezed into the docked alignment panel, since this app
    must render correctly on macOS/Windows Server/Linux and fixed-size
    embedded plot regions are exactly where font/DPI rendering has broken
    before. Just a canvas host; PipelineCanvas does the actual drawing,
    pointed at this window's `canvas` instead of the docked panel's.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Cell Alignment Preview')
        self.resize(900, 550)
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)
        self.canvas = FigureCanvasQTAgg()
        layout.addWidget(self.canvas)
