"""
Pick a p threshold by looking at what it keeps and what it throws away.

A NUMBER TYPED WITHOUT A PICTURE IS A GUESS. The classifier's p is
calibrated, so 0.7 means something -- but what it means for THIS run is
a question about this run's distribution, and the two things a person
needs in front of them are the histogram (where is the mass, is it
bimodal, is there anything in the middle at all) and some actual spots
from either side of the line they are drawing.

WHY THE EXAMPLES ARE RANDOM AND RE-DRAWN. Ranked examples show the
easiest cases: the top of the kept pile is obviously a spot and the
bottom of the denied pile is obviously nothing, and neither tells you
whether the threshold is in the right place. The draw is random, it is
restricted to a band around the threshold by default, and Refresh draws
again -- because one fixed sample of four invites reading four spots as
the whole distribution.

The dialog owns no pixels. A caller hands it `crop_of(spot) -> 3D array
or None`, so this works the same on a live detection run, on spots read
back from a store, and in a test with synthetic arrays.
"""
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from codelab_pipeline.localization import p_gate as G

# Half-width of the YX thumbnail cut around an example spot, in pixels.
# Wide enough to show a spot's surroundings -- whether it sits in a
# bright smear or on clean background is most of the judgement -- and
# small enough that eight of them fit on one row.
THUMB_R = 7

# How far either side of the threshold an example may be drawn from.
# The cases AT the line are the ones that decide whether the line is in
# the right place; 1.0 means "anywhere", offered as a checkbox.
DEFAULT_BAND = 0.15


class PGateDialog(QtWidgets.QDialog):
    """Preview a p threshold. Returns the chosen value, or None on cancel.

        dlg = PGateDialog(spots, crop_of=..., threshold=0.5, parent=w)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            keep = p_gate.apply(spots, dlg.threshold())
    """

    def __init__(self, spots, crop_of=None, threshold=G.DEFAULT_THRESHOLD,
                 parent=None, seed=None):
        super().__init__(parent)
        self.setWindowTitle('Spot probability gate')
        self._spots = list(spots or ())
        self._crop_of = crop_of
        self._rng = np.random.default_rng(seed)
        self.resize(980, 660)

        lay = QtWidgets.QVBoxLayout(self)
        self.head = QtWidgets.QLabel()
        self.head.setWordWrap(True)
        lay.addWidget(self.head)

        self.fig = Figure(figsize=(9.4, 2.4), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        lay.addWidget(self.canvas, 2)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Keep spots with p ≥'))
        self.edit = QtWidgets.QLineEdit(f'{float(threshold):.3f}')
        self.edit.setObjectName('threshold')
        self.edit.setMaximumWidth(90)
        self.edit.setValidator(QtGui.QDoubleValidator(0.0, 1.0, 4, self))
        row.addWidget(self.edit)
        self.apply_btn = QtWidgets.QPushButton('Preview histogram')
        self.apply_btn.setObjectName('preview')
        row.addWidget(self.apply_btn)
        self.refresh_btn = QtWidgets.QPushButton('Refresh examples')
        self.refresh_btn.setObjectName('refresh')
        row.addWidget(self.refresh_btn)
        self.band_box = QtWidgets.QCheckBox('only near the threshold')
        self.band_box.setObjectName('band')
        self.band_box.setChecked(True)
        self.band_box.setToolTip(
            'Draw examples from within %.2f of the threshold. The cases AT '
            'the line decide whether the line is in the right place; the '
            'extremes are easy either way.' % DEFAULT_BAND)
        row.addWidget(self.band_box)
        row.addStretch(1)
        lay.addLayout(row)

        self.efig = Figure(figsize=(9.4, 2.6), dpi=100)
        self.ecanvas = FigureCanvasQTAgg(self.efig)
        lay.addWidget(self.ecanvas, 3)

        self.note = QtWidgets.QLabel()
        self.note.setWordWrap(True)
        self.note.setStyleSheet('color:#555;')
        lay.addWidget(self.note)

        btn = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok
                                         | QtWidgets.QDialogButtonBox.Cancel)
        btn.accepted.connect(self.accept)
        btn.rejected.connect(self.reject)
        lay.addWidget(btn)

        self.apply_btn.clicked.connect(self._redraw)
        self.refresh_btn.clicked.connect(self._draw_examples)
        self.edit.returnPressed.connect(self._redraw)
        self.band_box.toggled.connect(self._draw_examples)
        self._redraw()

    # -- state -----------------------------------------------------------

    def threshold(self):
        """The number in the box, clamped to [0, 1]. Never raises: a
        half-typed value must not be able to take the dialog down."""
        try:
            v = float(self.edit.text())
        except (TypeError, ValueError):
            return G.DEFAULT_THRESHOLD
        return min(max(v, 0.0), 1.0)

    # -- drawing ---------------------------------------------------------

    def _redraw(self):
        t = self.threshold()
        s = G.summary(self._spots, t)
        if not G.available(self._spots):
            self.head.setText(
                'THIS RESULT HAS NO CALIBRATED PROBABILITY. Only the learned '
                'engine (v3-psfmatcher) produces one; v1, v2 and psf-match '
                'report a per-engine quality score instead, and a threshold '
                'on those is not a threshold on a probability. Nothing here '
                'would mean anything.')
        else:
            self.head.setText(
                f"{s['n_kept']} of {s['n_scored']} spots kept at p ≥ "
                f"{t:.3f}   ({100 * s['kept_frac']:.1f}%)"
                + (f"   ·   {s['n_unscored']} carry no probability and are "
                   f"NOT denied" if s['n_unscored'] else '')
                + f"   ·   p median {s['p_median']:.3f}")
        self._draw_hist(t)
        self._draw_examples()

    def _draw_hist(self, t):
        counts, edges = G.histogram(self._spots)
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        centres = 0.5 * (edges[:-1] + edges[1:])
        below = centres < t
        ax.bar(centres[below], counts[below], width=edges[1] - edges[0],
               color='#c9ccd1', label='denied')
        ax.bar(centres[~below], counts[~below], width=edges[1] - edges[0],
               color='#00a05a', label='kept')
        ax.axvline(t, color='#d33', lw=1.6)
        ax.set_xlim(0, 1)
        ax.set_xlabel('p — probability a spot is here (calibrated)',
                      fontsize=8)
        ax.set_ylabel('spots', fontsize=8)
        ax.tick_params(labelsize=7)
        # A LOG COUNT AXIS, because the distribution is strongly bimodal
        # in practice and a linear axis hides the middle -- which is the
        # only part of it a threshold is chosen in.
        if counts.max() > 0:
            ax.set_yscale('symlog', linthresh=1)
        ax.legend(fontsize=7, loc='upper center', frameon=False, ncol=2)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _draw_examples(self):
        t = self.threshold()
        band = DEFAULT_BAND if self.band_box.isChecked() else None
        kept, denied = G.examples(self._spots, t, n=4, rng=self._rng,
                                  band=band)
        self.efig.clear()
        rows = [('kept  (p ≥ %.3f)' % t, kept, '#00a05a'),
                ('denied  (p < %.3f)' % t, denied, '#8a8f96')]
        gs = self.efig.add_gridspec(2, 4, hspace=0.55, wspace=0.12)
        shown = 0
        for r, (label, pool, col) in enumerate(rows):
            for c in range(4):
                ax = self.efig.add_subplot(gs[r, c])
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color(col)
                if c >= len(pool):
                    ax.set_facecolor('#f4f4f4')
                    for sp in ax.spines.values():
                        sp.set_alpha(0.25)
                    # AN EMPTY ROW IS A FINDING, NOT A GLITCH. With the
                    # band on, no example here means the distribution has
                    # a clean gap at the threshold -- nothing this run
                    # produced is a borderline case, which is the best
                    # possible news about a threshold and looks exactly
                    # like a broken panel if nobody says so.
                    if c == 0 and not pool:
                        ax.text(0.5, 0.5,
                                'nothing within %.2f of the line'
                                % DEFAULT_BAND if band is not None
                                else 'none',
                                ha='center', va='center', fontsize=7.5,
                                color='#999', transform=ax.transAxes)
                        ax.set_title(label.split('(')[0].strip(),
                                     fontsize=7.5, color=col, pad=2)
                    continue
                img = self._thumb(pool[c])
                if img is None:
                    ax.text(0.5, 0.5, 'no pixels', ha='center', va='center',
                            fontsize=7, color='#999', transform=ax.transAxes)
                else:
                    ax.imshow(img, cmap='gray', interpolation='nearest')
                    shown += 1
                ax.set_title(f'{label.split("(")[0].strip() if c == 0 else ""}'
                             f'  p={G.p_of(pool[c]):.3f}',
                             fontsize=7.5, color=col, pad=2)
        gap = [label.split('(')[0].strip()
               for label, pool, _c in rows if not pool]
        if not shown and self._crop_of is None:
            self.note.setText(
                'No example images: this dialog was given no way to read '
                'pixels, so it can show the distribution but not the spots. '
                'The threshold is still applied.')
        elif gap and band is not None:
            self.note.setText(
                'Nothing is drawn for: ' + ', '.join(gap)
                + f' — this run produced no such spot within {DEFAULT_BAND:.2f} '
                  'of the threshold. That is a GAP in the distribution, not a '
                  'missing picture: the line sits where nothing is borderline. '
                  'Untick the box to draw from the whole range.')
        else:
            self.note.setText(
                'Examples are drawn at RANDOM'
                + (f' from within {DEFAULT_BAND:.2f} of the threshold — the '
                   'cases at the line are the ones that decide whether the '
                   'line is right' if band is not None
                   else ' from the whole range')
                + '. Refresh draws again.')
        self.ecanvas.draw_idle()

    def _thumb(self, spot):
        """A YX maximum projection around one spot, or None."""
        if self._crop_of is None:
            return None
        try:
            cube = self._crop_of(spot)
        except Exception:                                   # noqa: BLE001
            return None
        if cube is None:
            return None
        a = np.asarray(cube, float)
        if a.ndim == 3:
            a = np.nanmax(a, axis=2)
        if a.size == 0 or not np.isfinite(a).any():
            return None
        return a


def choose_threshold(spots, crop_of=None, threshold=G.DEFAULT_THRESHOLD,
                     parent=None):
    """Run the dialog. Returns the chosen threshold, or None if cancelled."""
    dlg = PGateDialog(spots, crop_of=crop_of, threshold=threshold,
                      parent=parent)
    if dlg.exec_() != QtWidgets.QDialog.Accepted:
        return None
    return dlg.threshold()
