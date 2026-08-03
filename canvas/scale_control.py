import numpy as np
from PyQt5 import QtWidgets, QtCore


class ScaleControlWidget(QtWidgets.QWidget):
    """
    Small reusable dynamic-range (contrast) control, embedded in every
    interactive pop-up displayer (CellDisplayer, SpotCropDisplayer,
    BarcodeOverviewDisplayer) so the user can interactively adjust vmin/
    vmax rather than being stuck with a fixed quantile -- important for
    picking barcode calibration bounds by eye, and generally for images
    whose useful dynamic range isn't well captured by one fixed default.

    Quantile mode (default) reads vmin/vmax as fractions (0-1) of the
    image's own intensity distribution; Absolute mode reads them as raw
    intensity values directly. changed fires on every edit so the owning
    displayer can just connect it straight to its own _redraw.
    """
    changed = QtCore.pyqtSignal()

    def __init__(self, lower_default=0.3, upper_default=0.999):
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel('Scale:'))

        self.ModeComboBox = QtWidgets.QComboBox()
        self.ModeComboBox.addItems(['Quantile', 'Absolute'])
        layout.addWidget(self.ModeComboBox)

        layout.addWidget(QtWidgets.QLabel('lower'))
        self.LowerSpinBox = QtWidgets.QDoubleSpinBox()
        self.LowerSpinBox.setRange(0, 1)
        self.LowerSpinBox.setDecimals(4)
        self.LowerSpinBox.setSingleStep(0.01)
        self.LowerSpinBox.setValue(lower_default)
        layout.addWidget(self.LowerSpinBox)

        layout.addWidget(QtWidgets.QLabel('upper'))
        self.UpperSpinBox = QtWidgets.QDoubleSpinBox()
        self.UpperSpinBox.setRange(0, 1)
        self.UpperSpinBox.setDecimals(4)
        self.UpperSpinBox.setSingleStep(0.01)
        self.UpperSpinBox.setValue(upper_default)
        layout.addWidget(self.UpperSpinBox)

        self.ModeComboBox.currentIndexChanged.connect(self._on_mode_changed)
        self.LowerSpinBox.valueChanged.connect(self.changed.emit)
        self.UpperSpinBox.valueChanged.connect(self.changed.emit)

    def _on_mode_changed(self):
        # sensible default ranges/values per mode -- quantile is always
        # 0-1, absolute needs headroom for real intensity values
        if self.ModeComboBox.currentText() == 'Quantile':
            self.LowerSpinBox.setRange(0, 1)
            self.UpperSpinBox.setRange(0, 1)
            self.LowerSpinBox.setValue(0.3)
            self.UpperSpinBox.setValue(0.999)
        else:
            self.LowerSpinBox.setRange(0, 1e9)
            self.UpperSpinBox.setRange(0, 1e9)
        self.changed.emit()

    def vmin_vmax(self, image):
        """
        (vmin, vmax) for imshow, given the image currently being displayed.
        Always returns vmax > vmin -- each spinbox emits `changed`
        independently, so live-editing (e.g. typing a new lower bound
        before the upper one catches up) routinely passes through a
        transient state where the two spinboxes disagree; matplotlib's
        Normalize raises ValueError on vmin >= vmax, which would otherwise
        crash mid-keystroke on a completely ordinary interactive edit.
        """
        finite = image[np.isfinite(image)]
        if self.ModeComboBox.currentText() == 'Quantile':
            lo, hi = self.LowerSpinBox.value(), self.UpperSpinBox.value()
            if finite.size == 0:
                vmin, vmax = 0, 1
            else:
                vmin, vmax = np.quantile(finite, lo), np.quantile(finite, hi)
        else:
            vmin, vmax = self.LowerSpinBox.value(), self.UpperSpinBox.value()
        if vmax <= vmin:
            vmax = vmin + max(abs(vmin) * 1e-6, 1e-6)
        return vmin, vmax
