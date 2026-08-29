from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from canvas import zoom_pan
from codelab_pipeline.analysis import report


class AnalysisFigureDisplayer(QtWidgets.QMainWindow):
    """One window per analysis view, hosting whatever Figure the toolbox
    returned, with the PNG+CSV+provenance saver attached.

    Save Result... writes <name>.png beside one CSV per underlying table
    and a JSON sidecar carrying the exact gate (predicates, sequential
    survivor counts, per-celltype gated/total) via report.save_result --
    a figure on disk answers "what was gated, and how many" by itself.
    """

    def __init__(self, title='Analysis', parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1100, 560)
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        # ONE canvas for the window's whole life; set_figure swaps the
        # FIGURE onto it. The first version built a new FigureCanvasQTAgg
        # per view and deleteLater'd the old one -- seven views in quick
        # succession died 0xC0000409 in offscreen Qt (widget destruction
        # racing the paint), the same crash class this repo has hit
        # before. Swapping figures does no widget churn at all.
        from matplotlib.figure import Figure
        self.canvas = FigureCanvasQTAgg(Figure())
        # SCROLLABLE: multi-group QC figures grow tall/wide (measured: a
        # celltype-decomposed FOV view is 4 rows x 6 panels); the scroll
        # area shows them at natural size instead of crushing them.
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(False)
        layout.insertWidget(0, scroll, stretch=1)
        zoom_pan.install_scroll_zoom(self.canvas)
        zoom_pan.install_drag_pan(self.canvas)
        row = QtWidgets.QHBoxLayout()
        self.SaveResultPushButton = QtWidgets.QPushButton(
            'Save Result... (PNG + CSV + provenance JSON)')
        self.SaveResultPushButton.clicked.connect(self._save_result)
        row.addStretch(1)
        row.addWidget(self.SaveResultPushButton)
        layout.addLayout(row)
        self.setCentralWidget(central)
        self._payload = None

    def set_figure(self, fig, name='analysis', tables=None, pop=None,
                   condition=None, params=None, default_dir=''):
        import matplotlib.pyplot as plt
        old = self.canvas.figure
        self.canvas.figure = fig
        fig.set_canvas(self.canvas)
        # natural pixel size inside the scroll area
        w, h = fig.get_size_inches() * fig.dpi
        self.canvas.resize(int(w), int(h))
        if old is not None and old is not fig:
            plt.close(old)      # pyplot keeps registry references otherwise
        self._payload = {'fig': fig, 'name': name, 'tables': tables or {},
                         'pop': pop, 'condition': condition,
                         'params': params or {}, 'default_dir': default_dir}
        # SYNCHRONOUS draw, deliberately: draw_idle defers rendering into
        # a later paint event, and a figure swapped mid-schedule died
        # 0xC0000409 (fail-fast -- faulthandler prints nothing for those).
        # A sync draw renders here, now, on this figure, and turns any
        # Agg error into an ordinary exception.
        self.canvas.draw()

    def _save_result(self):
        if not self._payload:
            return
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Save result into...',
            self._payload.get('default_dir') or '')
        if not d:
            return
        p = self._payload
        written = report.save_result(d, p['name'], fig=p['fig'],
                                     tables=p['tables'], pop=p['pop'],
                                     condition=p['condition'],
                                     params=p['params'])
        QtWidgets.QMessageBox.information(
            self, 'Save Result',
            'Written:\n' + '\n'.join(written))
