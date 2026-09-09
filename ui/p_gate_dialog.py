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
again -- because one fixed sample invites reading a handful of spots as
the whole distribution.

The dialog owns no pixels. A caller hands it `crop_of(spot)`, which
returns either a bare (y, x, z) cube -- pixels and nothing else -- or a
GateCrop, which adds where in that cube the spot actually is. Both are
first-class: a test with a synthetic array has no spot position to give,
and a live store does. With a GateCrop the tile is RINGED and gets a ZX
panel centred on the spot; with a bare cube it is drawn plain, because a
ring drawn at a guessed centre would be a claim about pixels nobody
measured. `None` means this spot's pixels could not be read.
"""
from collections import namedtuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from canvas import spot_fit_status
from codelab_pipeline.localization import p_gate as G

# What a crop_of may return instead of a bare cube.
#   cube      (y, x, z), this project's standard order
#   x, y      the spot's CROP-LOCAL lateral position. Not the centre:
#             a window clipped at a frame edge puts the spot off-centre,
#             and that is exactly when a centred ring lies.
#   z         crop-local z, or None when this spot's z was never fitted
#             -- models/spot.py is explicit that z == 0.0 cannot be read
#             as 'unfitted', so the provider decides from z_status and
#             says None rather than letting the dialog guess.
#   rejected  the z EXISTS but its fit was gate-rejected; drawn blue,
#             the convention canvas/spot_fit_status.py already uses.
GateCrop = namedtuple('GateCrop', 'cube x y z rejected')
GateCrop.__new__.__defaults__ = (None, False)

# Half-width of the crop cut around an example spot, in pixels. Wide
# enough to show a spot's surroundings -- whether it sits in a bright
# smear or on clean background is most of the judgement -- and small
# enough that a row of them fits.
#
# THE PROVIDER READS THIS. It used to be dead: this module named a
# half-width nothing used while windows/main_window's own pad=7 default
# was what actually cut the crop -- two numbers that had to agree, with
# nothing making them.
THUMB_R = 7

# How far either side of the threshold an example may be drawn from.
# The cases AT the line are the ones that decide whether the line is in
# the right place; 1.0 means "anywhere", offered as a checkbox.
DEFAULT_BAND = 0.15

# ONE TILE'S PIXEL BUDGET, TAKEN FROM THE 3D VIEWER so the two grids
# render a crop identically -- canvas/localize_3d_displayer.py's own
# col_px/pair_px. Its reason applies here word for word: "each crop gets
# a fixed on-screen size regardless of grid extent, so a many-spot grid
# scrolls instead of squeezing every crop unreadably small to fit the
# window". Squeezing is what went wrong: stretched to the dialog's width
# a 15x15 crop rendered at 105x70, and a ring sized for a real tile then
# covered the very spot it was pointing at.
COL_PX, PAIR_PX = 190, 300

# Examples drawn per side of the threshold, and so columns in the grid.
# FIVE at COL_PX each is 950 px, which is what a default-width dialog
# shows without scrolling sideways.
N_EXAMPLES = 5


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
        self.resize(1000, 900)

        lay = QtWidgets.QVBoxLayout(self)
        self.head = QtWidgets.QLabel()
        self.head.setWordWrap(True)
        lay.addWidget(self.head)
        self.warn = QtWidgets.QLabel()
        self.warn.setWordWrap(True)
        self.warn.setObjectName('quantity_note')
        self.warn.setStyleSheet('color:#7a5200; background:#fff6e0;'
                                ' padding:4px; border:1px solid #e8d9a8;')
        lay.addWidget(self.warn)

        self.fig = Figure(figsize=(9.4, 2.4), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        lay.addWidget(self.canvas, 2)

        row = QtWidgets.QHBoxLayout()
        self.klabel = QtWidgets.QLabel('Keep spots with p ≥')
        row.addWidget(self.klabel)
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

        # SIZED IN PIXELS AND SCROLLED, never stretched -- see COL_PX.
        # setWidgetResizable(False) is the part that matters: with it
        # True the scroll area would resize the canvas to itself and
        # squeeze the tiles again, which is the whole thing this avoids.
        self.efig = Figure(figsize=(N_EXAMPLES * COL_PX / 100.0,
                                    2 * PAIR_PX / 100.0), dpi=100)
        self.ecanvas = FigureCanvasQTAgg(self.efig)
        self.escroll = QtWidgets.QScrollArea()
        self.escroll.setWidgetResizable(False)
        self.escroll.setWidget(self.ecanvas)
        self.escroll.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        lay.addWidget(self.escroll, 3)

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
            self.head.setText('This result has no spots to gate.')
        else:
            q = s['quantity']
            self.head.setText(
                f"{s['n_kept']} of {s['n_scored']} spots kept at {q} ≥ "
                f"{t:.3f}   ({100 * s['kept_frac']:.1f}%)"
                + (f"   ·   {s['n_unscored']} have no {q} and are NOT denied"
                   if s['n_unscored'] else '')
                + f"   ·   median {s['p_median']:.3f}")
        # NAME THE QUANTITY, DO NOT REFUSE IT. Thresholding a v1 or v2
        # result is mechanically fine; what would be wrong is letting
        # someone read a per-engine ranking as a probability, or carry a
        # threshold chosen on one across to the other. So the warning is
        # about MEANING, and the picture is drawn either way.
        if not G.available(self._spots):
            self.warn.setText('')
        elif s['calibrated']:
            self.warn.setText(
                'p_exist — a CALIBRATED probability that a spot is here, from '
                'a classifier trained on human verdicts and Platt-scaled on '
                'held-out cells. Its ranking is sound (93.5% at 0.5 on 2,068 '
                'labels); its absolute scale is optimistic, because early '
                'stopping and the calibration share one validation split. '
                'Read the picture, not the number 0.5.')
        else:
            extra = (' Every value here is identical, so no threshold '
                     'separates anything — this engine ranks nothing.'
                     if s['degenerate'] else '')
            self.warn.setText(
                f"p — a per-ENGINE quality score, NOT a probability. It ranks "
                f"candidates within this one engine and this one "
                f"parameterisation; a threshold chosen here does not carry to "
                f"another engine or another run.{extra}")
        q = G.quantity(self._spots)
        self.klabel.setText(f'Keep spots with {q or "p"} ≥')
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
        q = G.quantity(self._spots) or G.CALIBRATED
        ax.set_xlabel(
            'p_exist — calibrated probability a spot is here' if q == G.CALIBRATED
            else "p — this engine's own quality ranking (not a probability)",
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
        kept, denied = G.examples(self._spots, t, n=N_EXAMPLES,
                                  rng=self._rng, band=band)
        self.efig.clear()
        rows = [('kept  (p ≥ %.3f)' % t, kept, '#00a05a'),
                ('denied  (p < %.3f)' % t, denied, '#8a8f96')]
        # NESTED, like canvas/localize_3d_displayer's own grid and for
        # its reason: an outer cell per example, split into a YX panel
        # over a ZX one, so the tight pairing stays tight while the gap
        # BETWEEN examples stays generous enough for a title. The
        # spacings are that grid's too, so a crop drawn here and the same
        # crop drawn in the 3D viewer come out the same size.
        quantity = G.quantity(self._spots)
        self.efig.set_size_inches(N_EXAMPLES * COL_PX / 100.0,
                                  2 * PAIR_PX / 100.0)
        self.ecanvas.setFixedSize(N_EXAMPLES * COL_PX, 2 * PAIR_PX)
        outer = self.efig.add_gridspec(2, N_EXAMPLES, hspace=0.55,
                                       wspace=0.4, left=0.055, right=0.98,
                                       top=0.90, bottom=0.03)
        # THE HANDLE IS RELEASED WHATEVER HAPPENS. Outside a finally
        # this leaked permanently the first time any tile raised: the
        # provider holds an open HDF5 handle on a NAS file, and the
        # next Refresh would simply open another.
        try:
            shown = self._draw_rows(outer, rows, quantity, band)
        finally:
            closer = getattr(self._crop_of, 'close', None)
            if closer is not None:
                try:
                    closer()
                except Exception:                       # noqa: BLE001
                    # A close that fails must not take down a dialog
                    # that has already drawn everything it can.
                    pass
        gap = [label.split('(')[0].strip()
               for label, pool, _c in rows if not pool]
        if not shown and self._crop_of is None:
            self.note.setText(
                'No example images: this dialog was given no way to read '
                'pixels, so it can show the distribution but not the spots. '
                'The threshold is still applied.')
        elif not shown and any(pool for _l, pool, _c in rows):
            # THERE WERE SPOTS TO DRAW AND NONE COULD BE READ. Falling
            # through to 'drawn at RANDOM' here would describe a picture
            # that is not on screen, and the reader would blame the
            # threshold for an unreadable store.
            self.note.setText(
                'There are spots on both sides of the line, but NONE of '
                'their pixels could be read -- the Z-STACK for this hybe '
                'and channel is missing or unreadable, or the spots carry '
                'no raw coordinate to centre a crop on. (The stack, not '
                'the MIP: a ZX panel needs the z axis, so an intact MIP '
                'beside a missing stack still lands here.) The '
                'distribution and the threshold are unaffected.')
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

    def _draw_rows(self, outer, rows, quantity, band):
        """Every tile. Split out ONLY so the handle's finally in
        _draw_examples wraps all of them -- the note branches stay
        in _draw_examples, where they are read from."""
        shown = 0
        for r, (label, pool, col) in enumerate(rows):
            for c in range(N_EXAMPLES):
                inner = outer[r, c].subgridspec(2, 1, hspace=0.08)
                ax_yx = self.efig.add_subplot(inner[0])
                ax_zx = self.efig.add_subplot(inner[1], sharex=ax_yx)
                for ax in (ax_yx, ax_zx):
                    ax.set_xticks([]); ax.set_yticks([])
                    for sp in ax.spines.values():
                        sp.set_color(col)
                if c >= len(pool):
                    for ax in (ax_yx, ax_zx):
                        ax.set_facecolor('#f4f4f4')
                        for sp in ax.spines.values():
                            sp.set_alpha(0.25)
                    # AN EMPTY ROW IS A FINDING, NOT A GLITCH. With the
                    # band on, no example here means the distribution has
                    # a clean gap at the threshold -- nothing this run
                    # produced is a borderline case, which is the best
                    # possible news about a threshold and looks exactly
                    # like a broken panel if nobody says so.
                    if c == 0:
                        self._row_label(ax_yx, label, col)
                        if not pool:
                            ax_yx.text(0.5, 0.5,
                                       'nothing within %.2f of the line'
                                       % DEFAULT_BAND if band is not None
                                       else 'none',
                                       ha='center', va='center',
                                       fontsize=7.5, color='#999',
                                       wrap=True,
                                       transform=ax_yx.transAxes)
                    continue
                if c == 0:
                    self._row_label(ax_yx, label, col)
                title = f'{quantity}={G.p_of(pool[c]):.3f}'
                if self._draw_one(ax_yx, ax_zx, pool[c], title, col):
                    shown += 1
                else:
                    ax_yx.text(0.5, 0.5, 'no pixels', ha='center',
                               va='center', fontsize=7, color='#999',
                               transform=ax_yx.transAxes)
                    ax_yx.set_title(title, fontsize=7.5, color=col, pad=2)
        return shown

    @staticmethod
    def _row_label(ax_yx, label, colour):
        """Which side of the line this row is, on the axis rather than
        in a title -- at eight columns a title holding both the label
        and the number does not fit, and the label describes the ROW."""
        ax_yx.set_ylabel(label.split('(')[0].strip(), fontsize=7.5,
                         color=colour, rotation=90, labelpad=2)

    def _draw_one(self, ax_yx, ax_zx, spot, title, colour):
        """One example as a YX/ZX pair with the spot ringed. True if drawn.

        The ring is the point of the pair. A reviewer looking at a
        borderline p has to know WHICH blob the number is about, and a
        15x15 crop near a bright neighbour routinely contains two.
        """
        got = self._crop(spot)
        if got is None:
            return False
        cube, x, y, z, rejected = got
        if cube.ndim == 2:
            # A FLAT WINDOW STILL DRAWS. One plane is a legal cube of
            # depth 1; the ZX panel is then a single row, which is
            # honest -- there is no depth to show.
            cube = cube[:, :, None]
        centroid = rejected_at = lateral = None
        if x is not None and y is not None:
            if z is None:
                lateral = (float(x), float(y))
            elif rejected:
                rejected_at = (float(x), float(y), float(z))
            else:
                centroid = (float(x), float(y), float(z))
        # EVERY DRAWING ARGUMENT IS THE 3D VIEWER'S. marker_size and
        # z_display_pad were overridden here (70 and 8) to fit tiles that
        # were being squeezed; with the tile budget fixed there is
        # nothing to compensate for, and overriding them made the same
        # crop look different in the two places -- a smaller ring over a
        # shallower z window, which is a different picture, not a
        # smaller one.
        spot_fit_status.draw_spot_fit_status(
            ax_yx, ax_zx, cube, centroid=centroid, rejected=rejected_at,
            lateral=lateral, title=title, title_fontsize=8)
        ax_yx.title.set_color(colour)
        return True

    def _crop(self, spot):
        """crop_of's answer, normalised. (cube, x, y, z, rejected) or None.

        Accepts both forms the module docstring names: a GateCrop, or a
        bare array with no position -- which becomes (cube, None, None,
        None, False) and so draws no ring at all.
        """
        if self._crop_of is None:
            return None
        try:
            got = self._crop_of(spot)
        except Exception:                                   # noqa: BLE001
            return None
        if got is None:
            return None
        if isinstance(got, GateCrop):
            cube, x, y, z, rejected = got
        elif isinstance(got, tuple):
            # A plain tuple is accepted in GateCrop's field order, so a
            # caller need not import the type to say where the spot is.
            cube, x, y, z, rejected = (list(got) + [None, None, None,
                                                    False])[:5]
        else:
            cube, x, y, z, rejected = got, None, None, None, False
        a = np.asarray(cube, float)
        if a.size == 0 or not np.isfinite(a).any():
            return None
        return a, x, y, z, bool(rejected)


def choose_threshold(spots, crop_of=None, threshold=G.DEFAULT_THRESHOLD,
                     parent=None):
    """Run the dialog. Returns the chosen threshold, or None if cancelled."""
    dlg = PGateDialog(spots, crop_of=crop_of, threshold=threshold,
                      parent=parent)
    accepted = dlg.exec_() == QtWidgets.QDialog.Accepted
    out = dlg.threshold() if accepted else None
    # A PARENTED DIALOG OUTLIVES exec_() -- Qt holds it as a child until
    # the parent dies, along with its spot list and two figures.
    dlg.deleteLater()
    return out
