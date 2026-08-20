import numpy as np
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

from canvas.scale_control import ScaleControlWidget
from canvas import zoom_pan

# distinct categorical colors, one per barcode channel -- unlike the
# alignment all-readouts overlay (canvas/pipeline_canvas.py), which
# deliberately switched to a sequential red-cyan gradient for a
# before/after single-transform comparison, here every channel is a
# genuinely different population/marker the user needs to tell apart at a
# glance, so a categorical palette is the right choice, not a gradient.
_CATEGORICAL_COLORS = [
    (0.90, 0.10, 0.10), (0.10, 0.65, 0.90), (0.15, 0.80, 0.15), (0.95, 0.75, 0.10),
    (0.70, 0.20, 0.90), (0.95, 0.45, 0.10), (0.10, 0.85, 0.75), (0.85, 0.10, 0.55),
]


class BarcodeOverviewDisplayer(QtWidgets.QMainWindow):
    """
    Pop-up composite overview for celltype barcode-mode calibration -- one
    color per assigned barcode (hybe, channel), each with its OWN
    interactive scale control (canvas/scale_control.py), max-composited
    together so the user can visually judge separation between populations
    and pick sensible lower/upper calibration bounds by eye before
    committing them (per explicit request: "it's very important to
    interactively modify the image dynamic range... we can modify each
    channel's scale and see the overview to determine celltype").

    Images passed to set_data are expected to already be warped into a
    common (reference) frame by the caller via cv2.warpAffine + that
    hybe's own alignment matrix -- warping for VISUALIZATION ONLY is the
    same established exception used throughout canvas/pipeline_canvas.py's
    overlay previews (never for stored/analyzed data, only so genuinely
    different-frame images can be compared by eye in one picture).
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Barcode Channel Overview')
        self.resize(680, 560)
        self.images_by_channel = {}   # {(hybe,channel): warped ndarray}
        self.labels_by_channel = {}   # {(hybe,channel): display label str}
        self._scale_controls = {}     # {(hybe,channel): ScaleControlWidget}

        central = QtWidgets.QWidget()
        self._layout = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)

        self.canvas = FigureCanvasQTAgg()
        self._layout.addWidget(self.canvas)
        zoom_pan.install_scroll_zoom(self.canvas)
        zoom_pan.install_keyboard_zoom(self.canvas)
        zoom_pan.install_drag_pan(self.canvas)

        self._legend_label = QtWidgets.QLabel('')
        self._legend_label.setWordWrap(True)
        self._layout.addWidget(self._legend_label)

        self.ResetViewPushButton = QtWidgets.QPushButton('Reset View')
        self.ResetViewPushButton.clicked.connect(lambda: zoom_pan.reset_view(self.canvas))
        self._layout.addWidget(self.ResetViewPushButton)

        self._scale_controls_container = QtWidgets.QWidget()
        self._scale_controls_layout = QtWidgets.QVBoxLayout(self._scale_controls_container)
        self._scale_controls_layout.setContentsMargins(0, 0, 0, 0)
        self._layout.addWidget(self._scale_controls_container)

    def set_data(self, images_by_channel, labels_by_channel):
        """
        images_by_channel: {(hybe,channel): ndarray}, already warped into
        one common frame by the caller. labels_by_channel: {(hybe,channel):
        str} -- e.g. "TypeA: Hyb_105 ch555".
        """
        self.images_by_channel = images_by_channel
        self.labels_by_channel = labels_by_channel

        # rebuild one ScaleControlWidget per channel (channel set can
        # change between calibration sessions as celltypes get assigned).
        # Must remove the ROW widgets from the layout, not just detach the
        # inner ScaleControlWidget -- setParent(None)'ing only the control
        # left its row (a QWidget holding the label + control) still sitting
        # in _scale_controls_layout, so every re-calibration accumulated
        # another stale label row underneath the real ones.
        while self._scale_controls_layout.count():
            item = self._scale_controls_layout.takeAt(0)
            row_widget = item.widget()
            if row_widget is not None:
                row_widget.setParent(None)
                row_widget.deleteLater()
        self._scale_controls = {}
        for bch in images_by_channel:
            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(QtWidgets.QLabel(labels_by_channel.get(bch, str(bch))))
            control = ScaleControlWidget()
            control.changed.connect(self._redraw)
            row_layout.addWidget(control)
            self._scale_controls_layout.addWidget(row)
            self._scale_controls[bch] = control
        self._redraw(keep_view=False)

    def _redraw(self, keep_view=True):
        if not self.images_by_channel:
            return
        shape = next(iter(self.images_by_channel.values())).shape
        composite = np.zeros((*shape, 3), dtype=float)
        legend_parts = []
        for i, (bch, img) in enumerate(self.images_by_channel.items()):
            color = _CATEGORICAL_COLORS[i % len(_CATEGORICAL_COLORS)]
            control = self._scale_controls.get(bch)
            vmin, vmax = control.vmin_vmax(img) if control is not None else (np.nanmin(img), np.nanmax(img))
            norm = np.clip((img.astype(float) - vmin) / max(vmax - vmin, 1e-9), 0, 1)
            layer = np.stack([norm * c for c in color], axis=-1)
            composite = np.maximum(composite, layer)
            rgb_hex = '#%02x%02x%02x' % tuple(int(c * 255) for c in color)
            legend_parts.append(f'<span style="color:{rgb_hex}">&#9632;</span> {self.labels_by_channel.get(bch, str(bch))}')

        fig = self.canvas.figure
        saved_view = zoom_pan.capture_view(fig) if keep_view else None
        fig.clear()
        # Explicit axes rectangles (figure-fraction coords), not fig.
        # subplots()+fig.colorbar(ax=...) -- chaining colorbar() calls on
        # the same ax repeatedly shrinks that ax and appends each new
        # colorbar right next to whatever was added last, so its own
        # `pad` ends up relative to the PREVIOUS colorbar's edge, not a
        # fixed reference -- there is no separate knob for "gap between
        # the image and the first colorbar" vs "gap BETWEEN colorbars".
        # Placing every axes explicitly gives two independent gaps.
        n = len(self.images_by_channel)
        main_left, main_width = 0.05, 0.55
        img_to_cb_gap = 0.05      # image -> first colorbar
        cb_gap = 0.14             # BETWEEN colorbars (bigger, per explicit request)
        cb_width = 0.035
        ax = fig.add_axes([main_left, 0.05, main_width, 0.9])
        ax.imshow(composite)
        ax.axis('off')
        # One colorbar PER channel, not a single shared one -- the main
        # image is an RGB max-composite (no single scalar mappable exists
        # for it), and each channel genuinely has its own color AND its
        # own independently-adjustable [vmin, vmax] (see the per-channel
        # ScaleControlWidget rows above). Each colorbar goes from black to
        # that channel's own color over ITS OWN range, so it reads
        # correctly as "how bright does this channel have to be to show
        # up," not a shared/misleading scale.
        x = main_left + main_width + img_to_cb_gap
        for i, (bch, img) in enumerate(self.images_by_channel.items()):
            color = _CATEGORICAL_COLORS[i % len(_CATEGORICAL_COLORS)]
            control = self._scale_controls.get(bch)
            vmin, vmax = control.vmin_vmax(img) if control is not None else (float(np.nanmin(img)), float(np.nanmax(img)))
            cmap = LinearSegmentedColormap.from_list(f'barcode_ch{i}', [(0, 0, 0), color])
            sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
            cax = fig.add_axes([x, 0.15, cb_width, 0.7])
            cb = fig.colorbar(sm, cax=cax)
            # title (top), not set_label (which puts a rotated label along
            # the side and reads poorly at this width) -- per explicit
            # request to move the label to the top of each colorbar.
            cb.ax.set_title(self.labels_by_channel.get(bch, str(bch)), fontsize=7, pad=6)
            cb.ax.tick_params(labelsize=6)
            x += cb_width + cb_gap
        zoom_pan.restore_view(fig, saved_view)
        self.canvas.draw()
        self._legend_label.setText('  '.join(legend_parts))
