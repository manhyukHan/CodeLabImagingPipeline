from PyQt5 import QtWidgets

from ui.ingestion_panel import IngestionPanelUI
from ui.alignment_panel import AlignmentPanelUI
from ui.cell_segment_panel import CellSegmentPanelUI
from ui.spot_localization_panel import SpotLocalizationPanelUI
from ui.celltype_determination_panel import CelltypeDeterminationPanelUI
from ui.chromatin_tracing_panel import ChromatinTracingPanelUI


class MainWindowUI(object):
    """
    App shell: a menu bar (load/save config) and a tabbed layout hosting
    the ingestion and alignment panels. Later milestones (segmentation,
    cell-based alignment, localization) add more tabs here without
    restructuring this shell.
    """
    def setupUi(self, MainWindow):
        MainWindow.setObjectName('MainWindow')
        # Smaller default so it fits comfortably on a laptop screen
        # alongside other windows (a pop-up displayer, an editor, etc.) --
        # every panel below is wrapped in a QScrollArea so shrinking the
        # window scrolls its controls instead of clipping them.
        MainWindow.resize(820, 620)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName('centralwidget')
        layout = QtWidgets.QVBoxLayout(self.centralwidget)

        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        layout.addWidget(self.tabWidget)

        self.IngestionPanelWidget = QtWidgets.QWidget()
        self.IngestionPanel = IngestionPanelUI()
        self.IngestionPanel.setupUi(self.IngestionPanelWidget)
        self.tabWidget.addTab(self._scrollable(self.IngestionPanelWidget), 'Ingestion')

        self.CellSegmentPanelWidget = QtWidgets.QWidget()
        self.CellSegmentPanel = CellSegmentPanelUI()
        self.CellSegmentPanel.setupUi(self.CellSegmentPanelWidget)
        self.tabWidget.addTab(self._scrollable(self.CellSegmentPanelWidget), 'Cell Segmentation')

        self.AlignmentPanelWidget = QtWidgets.QWidget()
        self.AlignmentPanel = AlignmentPanelUI()
        self.AlignmentPanel.setupUi(self.AlignmentPanelWidget)
        self.tabWidget.addTab(self._scrollable(self.AlignmentPanelWidget), 'Alignment')

        self.SpotLocalizationPanelWidget = QtWidgets.QWidget()
        self.SpotLocalizationPanel = SpotLocalizationPanelUI()
        self.SpotLocalizationPanel.setupUi(self.SpotLocalizationPanelWidget)
        self.tabWidget.addTab(self._scrollable(self.SpotLocalizationPanelWidget), 'Spot Localization')

        self.CelltypeDeterminationPanelWidget = QtWidgets.QWidget()
        self.CelltypeDeterminationPanel = CelltypeDeterminationPanelUI()
        self.CelltypeDeterminationPanel.setupUi(self.CelltypeDeterminationPanelWidget)
        self.tabWidget.addTab(self._scrollable(self.CelltypeDeterminationPanelWidget), 'Celltype Determination')

        self.ChromatinTracingPanelWidget = QtWidgets.QWidget()
        self.ChromatinTracingPanel = ChromatinTracingPanelUI()
        self.ChromatinTracingPanel.setupUi(self.ChromatinTracingPanelWidget)
        self.tabWidget.addTab(self._scrollable(self.ChromatinTracingPanelWidget), 'Chromatin Tracing')

        # The combined log window replaced every panel's own log box -- this
        # corner button (visible from every tab) re-opens it after the user
        # closes it. MainWindow wires the click.
        self.ShowLogPushButton = QtWidgets.QPushButton('Show Log')
        self.tabWidget.setCornerWidget(self.ShowLogPushButton)

        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menuFile = self.menubar.addMenu('File')
        self.actionLoad_Config = QtWidgets.QAction('Load Config...', MainWindow)
        self.actionSave_Config = QtWidgets.QAction('Save Config...', MainWindow)
        self.menuFile.addAction(self.actionLoad_Config)
        self.menuFile.addAction(self.actionSave_Config)
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

    @staticmethod
    def _scrollable(widget):
        """Wraps a panel widget in a QScrollArea so a smaller main window scrolls its controls instead of clipping them."""
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        return scroll

    def retranslateUi(self, MainWindow):
        _translate = QtWidgets.QApplication.translate
        MainWindow.setWindowTitle(_translate('MainWindow', 'CODE Lab Imaging Pipeline'))
