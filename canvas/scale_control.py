import numpy as np
from PyQt5 import QtWidgets, QtCore


class ScaleControlWidget(QtWidgets.QWidget):
    """
    Small reusable dynamic-range (contrast) control, embedded in every
    interactive pop-up displayer (CellDisplayer, SpotCropDisplayer,
    MipViewerDisplayer, CelltypeResultDisplayer, BarcodeOverviewDisplayer)
    so the user can interactively adjust vmin/vmax rather than being stuck
    with a fixed quantile.

    Two always-visible rows -- Quantile (0-1 fractions of the current
    image's own intensity distribution) and Absolute (raw intensity
    values) -- kept automatically in sync against whatever image was last
    shown, rather than a single field whose meaning flips depending on a
    Mode combobox. Editing either row recomputes the other (via the last-
    seen image); Absolute is the actual value handed to matplotlib, so
    reading vmin_vmax never needs to re-derive anything or care which row
    the user touched most recently.

    Per confirmed real bug in the old Mode-combobox design: switching
    modes left the OTHER interpretation's stale numbers sitting in the
    same field (e.g. absolute values like 219/501 still in the box after
    flipping to Quantile), and the combobox's own re-range call clamped
    one spinbox before the other had caught up -- a live vmin_vmax() call
    during that transient, half-updated state fed an out-of-[0,1] value
    straight into np.quantile, raising ValueError. Two independent rows
    with no shared field/mode to desync eliminates that whole class of
    bug -- there's no clamp-then-reassign step at all any more.
    """
    changed = QtCore.pyqtSignal()

    def __init__(self, lower_default=0.3, upper_default=0.999):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._last_image = None
        # Re-entrancy guard: writing into one row FROM the other's own
        # valueChanged handler must not bounce back and re-trigger a sync
        # in the opposite direction (or double-emit `changed`).
        self._syncing = False

        self.QuantileLowerSpinBox, self.QuantileUpperSpinBox = self._add_row(
            layout, 'Quantile:', lower_default, upper_default, decimals=4, minv=0, maxv=1, step=0.01)
        self.AbsoluteLowerSpinBox, self.AbsoluteUpperSpinBox = self._add_row(
            layout, 'Absolute:', 0.0, 0.0, decimals=2, minv=0, maxv=1e9, step=1.0)

        self.QuantileLowerSpinBox.valueChanged.connect(self._on_quantile_changed)
        self.QuantileUpperSpinBox.valueChanged.connect(self._on_quantile_changed)
        self.AbsoluteLowerSpinBox.valueChanged.connect(self._on_absolute_changed)
        self.AbsoluteUpperSpinBox.valueChanged.connect(self._on_absolute_changed)

    @staticmethod
    def _add_row(layout, label, lower_default, upper_default, decimals, minv, maxv, step):
        row = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(QtWidgets.QLabel(label))

        def spin(default):
            sb = QtWidgets.QDoubleSpinBox()
            sb.setDecimals(decimals)
            sb.setRange(minv, maxv)
            sb.setSingleStep(step)
            sb.setValue(default)
            return sb

        lower = spin(lower_default)
        upper = spin(upper_default)
        row_layout.addWidget(lower)
        row_layout.addWidget(QtWidgets.QLabel('-'))
        row_layout.addWidget(upper)
        layout.addWidget(row)
        return lower, upper

    @staticmethod
    def _finite(image):
        if image is None:
            return None
        finite = image[np.isfinite(image)]
        return finite if finite.size else None

    def _on_quantile_changed(self):
        if self._syncing:
            return
        finite = self._finite(self._last_image)
        if finite is not None:
            self._syncing = True
            self.AbsoluteLowerSpinBox.setValue(float(np.quantile(finite, self.QuantileLowerSpinBox.value())))
            self.AbsoluteUpperSpinBox.setValue(float(np.quantile(finite, self.QuantileUpperSpinBox.value())))
            self._syncing = False
        self.changed.emit()

    def _on_absolute_changed(self):
        if self._syncing:
            return
        finite = self._finite(self._last_image)
        if finite is not None:
            lo, hi = self.AbsoluteLowerSpinBox.value(), self.AbsoluteUpperSpinBox.value()
            self._syncing = True
            self.QuantileLowerSpinBox.setValue(float(np.clip(np.mean(finite <= lo), 0.0, 1.0)))
            self.QuantileUpperSpinBox.setValue(float(np.clip(np.mean(finite <= hi), 0.0, 1.0)))
            self._syncing = False
        self.changed.emit()

    def vmin_vmax(self, image):
        """
        (vmin, vmax) for imshow -- reads the Absolute row (the two rows are
        kept in sync, see _on_quantile_changed/_on_absolute_changed above,
        so Absolute is always a ready-to-use answer).

        Quantile is the default whenever the VIEW changes -- every call
        where `image` is a different array than the last call saw (a new
        cell/spot/hybe/channel selected, i.e. the caller's own set_data
        reassigning its stored image attribute) re-derives Absolute from
        the Quantile row's CURRENT values against this new image, before
        reading it back out. Per confirmed real bug/explicit request:
        previously this only primed Absolute from Quantile on the very
        first-ever call, so switching views kept reusing whatever Absolute
        values were last dialed in for a COMPLETELY DIFFERENT image's own
        intensity range -- since the two rows are now always kept in sync,
        Quantile is the one meant to carry across views (an intensity
        fraction stays meaningful image to image; a raw absolute value
        doesn't). A caller that redraws the SAME image again (e.g. this
        widget's own `changed` firing from a manual Absolute edit) passes
        the identical array object back -- `image is not self._last_image`
        only trips on a genuine view switch, never on a scale-only redraw,
        so a manual Absolute edit for the CURRENT image is never clobbered
        by this.

        Always returns vmax > vmin -- each spinbox emits `changed`
        independently, so live-editing (e.g. typing a new lower bound
        before the upper one catches up) routinely passes through a
        transient state where the two disagree; matplotlib's Normalize
        raises ValueError on vmin >= vmax, which would otherwise crash
        mid-keystroke on a completely ordinary interactive edit.
        """
        finite = self._finite(image)
        image_changed = image is not self._last_image
        self._last_image = image
        if image_changed and finite is not None:
            self._syncing = True
            self.AbsoluteLowerSpinBox.setValue(float(np.quantile(finite, self.QuantileLowerSpinBox.value())))
            self.AbsoluteUpperSpinBox.setValue(float(np.quantile(finite, self.QuantileUpperSpinBox.value())))
            self._syncing = False

        vmin, vmax = self.AbsoluteLowerSpinBox.value(), self.AbsoluteUpperSpinBox.value()
        if vmax <= vmin:
            vmax = vmin + max(abs(vmin) * 1e-6, 1e-6)
        return vmin, vmax
