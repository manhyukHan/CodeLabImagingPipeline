"""
Spot Check: the review app lab members run. One bundle in, verdicts out.

WHAT IT IS NOT. It is not the pipeline. It opens one bundle directory and
nothing else -- no store, no NAS, no config, no alignment. That is
deliberate: ten people reviewing at once must not each be pulling
cell-sized windows out of 266 MB gzipped stacks over a share (measured
673 ms a crop, against 12 ms from a bundle), and they must not need the
analysis environment installed to help.

THE DEFAULT IS REJECT. A reviewer presses a number only for a spot that
is REAL; everything else on the page is a negative by simply not being
pressed. Most pages end empty -- 11 candidates with 4 real is a good
cell, and a blank cell yields 24 pieces of noise -- so making rejection
the silent case is what keeps 100k judgements from being 100k clicks.

The risk of that default is a reviewer holding Space and labelling
nothing. Two things push back: the page's dwell time is recorded with
the verdict, so a pass done in 0.3 s is visible afterwards, and an EMPTY
page still writes a record, so "reviewed and found nothing" is a real
labelled result rather than an absence.

KEYS
  1 2 3 4     toggle that card (keep / drop). Default is drop.
  Space       commit the page and go to the next
  Backspace   previous page (its verdict can be re-committed; the later
              line wins and both survive)
  A           add a spot the candidates missed -- then click it on the
              cell image at the left
  U           undo the last added spot
  S           skip this page WITHOUT recording a verdict (use for a crop
              you cannot judge; it stays unlabelled rather than becoming
              a false negative)
  Q           quit -- nothing is lost, the log is flushed per page

Run:
  python -m spotcheck.app <bundle_dir> --reviewer <name>
"""
import argparse
import os
import sys
import time

import numpy as np
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline.training import bundle as B          # noqa: E402
from codelab_pipeline.training import verdicts as V        # noqa: E402
from codelab_pipeline.training import view as VIEW         # noqa: E402


class Queue:
    """Every (shard, crop, page) in the bundle, minus what is already done.

    Built once from the shard INDEXES only -- never from the pixels --
    so opening a 30k-crop bundle costs a few index reads rather than a
    few gigabytes.
    """

    def __init__(self, bundle_dir, log, per_page=VIEW.PER_PAGE):
        self.dir = str(bundle_dir)
        self.log = log
        self.items = []
        done = log.done_pages()
        for shard in B.shard_paths(self.dir):
            for row in B.read_index(shard):
                n = int(row['n_candidates'])
                if n == 0:
                    continue          # nothing to judge, not a skipped page
                for pi, ix in enumerate(VIEW.pages_of(n, per_page)):
                    if (row['key'], pi) in done:
                        continue
                    self.items.append((shard, row, pi, ix))
        self.i = 0

    def __len__(self):
        return len(self.items)

    def current(self):
        return self.items[self.i] if 0 <= self.i < len(self.items) else None

    def advance(self, step=1):
        self.i = max(0, min(len(self.items) - 1, self.i + step))
        return self.current()


class SpotCheck(QtWidgets.QMainWindow):

    def __init__(self, bundle_dir, reviewer, per_page=VIEW.PER_PAGE):
        super().__init__()
        self.bundle_dir = str(bundle_dir)
        self.per_page = int(per_page)
        self.log = V.VerdictLog(self.bundle_dir, reviewer)
        self.queue = Queue(self.bundle_dir, self.log, self.per_page)

        self.setWindowTitle(f'Spot Check — {reviewer} — {self.bundle_dir}')
        self.resize(1750, 780)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        lay = QtWidgets.QVBoxLayout(central)

        self.fig = Figure(figsize=(15.0, 5.6), dpi=110)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        lay.addWidget(self.canvas, 1)

        self.status = QtWidgets.QLabel()
        self.status.setStyleSheet('font-family: monospace; padding: 3px;')
        lay.addWidget(self.status)

        self.help = QtWidgets.QLabel(
            '1-4 keep/drop   Space commit+next   Backspace back   '
            'A add missed (then click the cell)   U undo add   '
            'S skip unlabelled   Q quit          '
            'DEFAULT IS DROP — press a number only for a real spot')
        self.help.setStyleSheet('color:#555; padding: 2px;')
        lay.addWidget(self.help)

        self.canvas.mpl_connect('button_press_event', self._on_click)
        self._state = None
        self._adding = False
        self._t0 = time.time()
        self._load()

    # -- data ------------------------------------------------------------

    def _load(self):
        item = self.queue.current()
        if item is None:
            self._finish()
            return
        shard, row, page, ix = item
        stack, mask, cands, words = B.read_crop(shard, row['key'])
        rows = [(float(c['y']), float(c['x']), float(c['z']), float(c['p']),
                 int(c['fit_ok']), int(c['gate_pass']), w)
                for c, w in zip(cands, words)]
        npage = len(VIEW.pages_of(len(rows), self.per_page))
        self._state = dict(shard=shard, row=row, page=page, ix=ix,
                           stack=stack, mask=mask, cands=rows, npage=npage,
                           accepted=set(), added=[])
        self._adding = False
        self._t0 = time.time()
        self._draw()

    def _draw(self):
        s = self._state
        row = s['row']
        header = (f"FOV{int(row['fov']):03d}   {row['hybe']}   "
                  f"ch{int(row['channel'])}   cell {int(row['cell'])}")
        VIEW.draw_page(self.fig, s['stack'], s['mask'], s['cands'], s['ix'],
                       header=header, accepted=s['accepted'],
                       added=s['added'], page=s['page'], npage=s['npage'],
                       per_page=self.per_page)
        self.canvas.draw_idle()
        kept = ', '.join(f'#{i + 1}' for i in sorted(s['accepted'])) or 'none'
        self.status.setText(
            f'queue {self.queue.i + 1}/{len(self.queue)}   '
            f'|  keeping: {kept}   '
            f'|  added: {len(s["added"])}   '
            + ('|  CLICK THE CELL TO ADD A SPOT (Esc cancels)'
               if self._adding else ''))

    def _finish(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111); ax.axis('off')
        ax.text(0.5, 0.5, 'Nothing left to review in this bundle.\n'
                          'Thank you — your verdicts are saved.',
                ha='center', va='center', fontsize=14)
        self.canvas.draw_idle()
        self.status.setText(f'log: {self.log.path}')

    # -- input -----------------------------------------------------------

    def _on_click(self, ev):
        if not (self._adding and self._state and ev.inaxes is not None
                and ev.xdata is not None):
            return
        # Only the cell overview accepts an added spot: its axes is the
        # one whose coordinates are crop-local (y, x), which is the frame
        # the verdict is recorded in.
        if ev.inaxes is not self.fig.axes[0]:
            return
        self._state['added'].append((float(ev.ydata), float(ev.xdata)))
        self._adding = False
        self._draw()

    def keyPressEvent(self, e):
        s = self._state
        k = e.key()
        if k in (QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape) and not self._adding:
            self.close(); return
        if k == QtCore.Qt.Key_Escape:
            self._adding = False; self._draw(); return
        if s is None:
            return
        if QtCore.Qt.Key_1 <= k <= QtCore.Qt.Key_9:
            slot = k - QtCore.Qt.Key_1
            if slot < len(s['ix']):
                i = s['ix'][slot]
                s['accepted'].symmetric_difference_update({i})
                self._draw()
            return
        if k == QtCore.Qt.Key_A:
            self._adding = True; self._draw(); return
        if k == QtCore.Qt.Key_U:
            if s['added']:
                s['added'].pop(); self._draw()
            return
        if k == QtCore.Qt.Key_S:
            # No record at all. A crop the reviewer cannot judge must stay
            # UNLABELLED -- committing it empty would file it as four
            # confirmed negatives, which is a lie the model would learn.
            self.queue.advance(1); self._load(); return
        if k == QtCore.Qt.Key_Space:
            self.log.commit(s['row']['key'], s['page'], s['ix'],
                            s['accepted'], added=s['added'],
                            seconds=time.time() - self._t0,
                            bundle=os.path.basename(s['shard']))
            self.queue.advance(1); self._load(); return
        if k == QtCore.Qt.Key_Backspace:
            self.queue.advance(-1); self._load(); return
        super().keyPressEvent(e)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('bundle_dir')
    ap.add_argument('--reviewer', required=True,
                    help='your name/initials -- names the verdict file, so '
                         'ten people can share one bundle directory')
    ap.add_argument('--per-page', type=int, default=VIEW.PER_PAGE)
    a = ap.parse_args(argv)
    if not os.path.isdir(a.bundle_dir):
        raise SystemExit(f'no such bundle directory: {a.bundle_dir}')
    app = QtWidgets.QApplication(sys.argv[:1])
    w = SpotCheck(a.bundle_dir, a.reviewer, a.per_page)
    if len(w.queue) == 0:
        print('nothing left to review in', a.bundle_dir)
    w.show()
    w.canvas.setFocus()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
