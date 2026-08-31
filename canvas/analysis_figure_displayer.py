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
        # VIEW RANGE -- image-level only, per explicit request: these
        # never re-gate and never touch the prepared data. x/y limits are
        # applied to the axes; the bin count asks the view to re-render
        # the SAME rows at a different binning (a rebuild callback the
        # caller supplies), which is still presentation, not selection.
        rangeRow = QtWidgets.QHBoxLayout()
        self.XMinLineEdit = QtWidgets.QLineEdit()
        self.XMaxLineEdit = QtWidgets.QLineEdit()
        self.YMinLineEdit = QtWidgets.QLineEdit()
        self.YMaxLineEdit = QtWidgets.QLineEdit()
        self.BinsLineEdit = QtWidgets.QLineEdit()
        for label, w, tip in (('X', self.XMinLineEdit, 'x min'),
                              ('-', self.XMaxLineEdit, 'x max'),
                              ('Y', self.YMinLineEdit, 'y min'),
                              ('-', self.YMaxLineEdit, 'y max'),
                              ('bins', self.BinsLineEdit, 'x bins')):
            w.setPlaceholderText(tip + ' (auto)')
            w.setMaximumWidth(90)
            rangeRow.addWidget(QtWidgets.QLabel(label + ':'))
            rangeRow.addWidget(w)
            w.returnPressed.connect(self._apply_view_range)
        self.ApplyRangePushButton = QtWidgets.QPushButton('Apply')
        self.ApplyRangePushButton.clicked.connect(self._apply_view_range)
        rangeRow.addWidget(self.ApplyRangePushButton)
        self.AutoRangePushButton = QtWidgets.QPushButton('Auto')
        self.AutoRangePushButton.clicked.connect(self._auto_view_range)
        rangeRow.addWidget(self.AutoRangePushButton)
        self.RangeHintLabel = QtWidgets.QLabel('view range (display only)')
        rangeRow.addWidget(self.RangeHintLabel)
        rangeRow.addStretch(1)
        layout.addLayout(rangeRow)

        row = QtWidgets.QHBoxLayout()
        self.SaveResultPushButton = QtWidgets.QPushButton(
            'Save Result... (PNG + CSV + provenance JSON)')
        self.SaveResultPushButton.clicked.connect(self._save_result)
        row.addStretch(1)
        row.addWidget(self.SaveResultPushButton)
        layout.addLayout(row)
        self.setCentralWidget(central)
        self._payload = None

    def _range_values(self):
        def num(w, cast=float):
            t = w.text().strip()
            return cast(t) if t else None
        return {'xmin': num(self.XMinLineEdit), 'xmax': num(self.XMaxLineEdit),
                'ymin': num(self.YMinLineEdit), 'ymax': num(self.YMaxLineEdit),
                'bins': num(self.BinsLineEdit, int)}

    def _apply_view_range(self):
        try:
            v = self._range_values()
        except ValueError:
            QtWidgets.QMessageBox.warning(self, 'View range',
                                          'Limits must be numbers (bins an '
                                          'integer); leave blank for auto.')
            return
        rebuild = (self._payload or {}).get('rebuild')
        if v['bins'] and rebuild is not None:
            # a different binning of the SAME rows -- the caller re-renders
            try:
                fig = rebuild(v['bins'])
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, 'View range', str(e))
                return
            if fig is not None:
                payload = dict(self._payload)
                self.set_figure(fig, name=payload['name'],
                                tables=payload['tables'], pop=payload['pop'],
                                condition=payload['condition'],
                                params=payload['params'],
                                default_dir=payload['default_dir'],
                                rebuild=rebuild, keep_range=True)
        for ax in self.canvas.figure.axes:
            if v['xmin'] is not None or v['xmax'] is not None:
                ax.set_xlim(left=v['xmin'], right=v['xmax'])
            if v['ymin'] is not None or v['ymax'] is not None:
                ax.set_ylim(bottom=v['ymin'], top=v['ymax'])
        self.canvas.draw()

    def _auto_view_range(self):
        for w in (self.XMinLineEdit, self.XMaxLineEdit, self.YMinLineEdit,
                  self.YMaxLineEdit, self.BinsLineEdit):
            w.clear()
        rebuild = (self._payload or {}).get('rebuild')
        if rebuild is not None:
            fig = rebuild(None)
            if fig is not None:
                payload = dict(self._payload)
                self.set_figure(fig, name=payload['name'],
                                tables=payload['tables'], pop=payload['pop'],
                                condition=payload['condition'],
                                params=payload['params'],
                                default_dir=payload['default_dir'],
                                rebuild=rebuild, keep_range=True)
                return
        for ax in self.canvas.figure.axes:
            ax.autoscale()
        self.canvas.draw()

    def set_figure(self, fig, name='analysis', tables=None, pop=None,
                   condition=None, params=None, default_dir='',
                   rebuild=None, keep_range=False):
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
                         'params': params or {}, 'default_dir': default_dir,
                         # rebuild(bins) -> a new figure over the SAME
                         # rows; None for views with no binning to change
                         'rebuild': rebuild}
        has = rebuild is not None
        self.BinsLineEdit.setEnabled(has)
        self.RangeHintLabel.setText(
            'view range (display only)' if has else
            'view range (display only; bins n/a for this view)')
        if not keep_range:
            for w in (self.XMinLineEdit, self.XMaxLineEdit, self.YMinLineEdit,
                      self.YMaxLineEdit, self.BinsLineEdit):
                w.clear()
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
