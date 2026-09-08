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
import matplotlib
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# Glyph rasterization is the single largest cost in this window --
# MEASURED 0.48 ms per glyph, 65% of a full page draw. Unhinted glyphs
# cost 0.32 ms. This is an application, so setting it process-wide is
# ours to do; the library modules leave rcParams alone.
matplotlib.rcParams['text.hinting'] = 'none'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline.training import bundle as B          # noqa: E402
from codelab_pipeline.training import verdicts as V        # noqa: E402
from codelab_pipeline.training import view as VIEW         # noqa: E402


# How close a click has to land to count as "that candidate" rather than
# a new spot. A real emitter is ~1.3 px wide and the drawn circle has
# radius 3.8, so anything inside the circle a person was aiming at snaps.
ADD_SNAP_PX = 4.0

# How many candidates per cell a reviewer is asked to judge, by default.
#
# EIGHT, BECAUSE CONFIRMED SPOTS PER PAGE IS WHAT LIMITS EVERYTHING.
# MEASURED on 117 pages a reviewer actually judged, the keep rate falls
# steeply with the bundle's own ordering (gate-pass, then fitted, then
# by quality):
#
#     rank    1    2    3    4    5    6    7    8  | 9-16
#     kept   97%  91%  83%  62%  69%  55%  45%  31% |  21%
#
# so ranks 1-8 keep 67% and ranks 9-16 keep 21%. A page is four cards,
# which makes it
#
#     confirmed spots per page   cap 8: 2.69     cap 16: 1.80
#
# -- the same number of true positives for a THIRD fewer pages turned.
# It is not free: those 222 shown ranks 9-16 still held 47 real spots.
#
# WHAT THE CAP USED TO BUY WAS THE GRAY AREA, and the data supplies that
# on its own. Hybes differ enormously in how many candidates clear the
# gate -- MEASURED over 455 crops of MP58/RNA, per crop:
#
#     Hyb_101   0.7 gate-pass   ( 0.9% of its candidates)
#     Hyb_105   2.3             ( 2.2%)
#     Hyb_103   6.8             ( 6.6%)
#     Hyb_107  31.0             (28.2%)
#
# a factor of forty. The quiet hybes are all boundary; the loud ones are
# all confident. A fixed cap therefore means something different on each,
# and 33% of crops now carry more than 8 gate-passes (median 4, p90 34,
# max 54), so on those the reviewer sees confident spots only. That is
# the trade taken deliberately: volume of confirmed spots first, and the
# gray area from the hybes that are made of it.
#
# (Hyb_107 clearing 28% of its own candidates is itself worth a look --
# either it is genuinely much brighter, or the gate does not hold there.
# Unmeasured either way.)
#
# The bundle keeps every candidate regardless. This bounds the view, so
# raising it later is a flag rather than a re-extraction -- candidate
# rows are 19 bytes against 1.58 GiB of pixels, and truncating at write
# time would buy 0.6% and cost the option.
DEFAULT_MAX_PER_CROP = 8

# How much of each reviewer's stream is drawn from the shared order that
# EVERY reviewer walks, rather than from their own shuffle.
#
# The rest spreads out so that stopping early still leaves an unbiased
# sample of the whole bundle. This slice is what stays comparable: two
# people who each judge 200 pages of an 18,000-page bundle would
# otherwise share about two, and an inter-rater number needs more than
# that. At 0.1, every tenth page is one everybody sees.
OVERLAP_FRAC = 0.1


def ask_session(bundle_dir=None, reviewer=None, parent=None):
    """(bundle_dir, reviewer) -- prompting for whatever was not given.

    The reviewer NAMES THE VERDICT FILE, so the same program writes to a
    different place per person and ten of them share one bundle folder
    without colliding. Getting it as a Qt dialog rather than only as a
    command-line flag is what makes the app double-clickable, and it
    removes the shell prompts a .bat was doing badly.

    A name typed differently on the second day starts a FRESH queue and
    re-reviews everything already done, so the field is pre-filled with
    the last one used and the answer is remembered.
    """
    from PyQt5 import QtWidgets as W
    remembered = _remembered()
    if not bundle_dir:
        bundle_dir = W.QFileDialog.getExistingDirectory(
            parent, 'Choose the review bundle folder',
            remembered.get('bundle', ''))
        if not bundle_dir:
            return None, None
    if not reviewer:
        reviewer, ok = W.QInputDialog.getText(
            parent, 'Spot Check',
            'Your name or initials.\n\n'
            'It names your verdict file, so several people can share one\n'
            'bundle folder. Use the SAME spelling every session -- a new\n'
            'name starts a fresh queue and re-reviews what you have done.',
            text=remembered.get('reviewer', ''))
        if not ok or not str(reviewer).strip():
            return None, None
        reviewer = str(reviewer).strip()
    _remember(bundle_dir, reviewer)
    return bundle_dir, reviewer


def _remember_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'last_session.json')


def _remembered():
    import json
    try:
        with open(_remember_path(), encoding='utf-8') as f:
            return json.load(f)
    except Exception:                                       # noqa: BLE001
        return {}


def _remember(bundle_dir, reviewer):
    import json
    try:
        with open(_remember_path(), 'w', encoding='utf-8') as f:
            json.dump({'bundle': str(bundle_dir),
                       'reviewer': str(reviewer)}, f)
    except Exception:                                       # noqa: BLE001
        pass          # remembering is a convenience, never a requirement


class KeyPassingCanvas(FigureCanvasQTAgg):
    """A matplotlib canvas that lets keystrokes reach the window.

    WITHOUT THIS THE APP IS ENTIRELY DEAD TO THE KEYBOARD, and it fails
    silently: the window draws, the mouse works, and every key does
    nothing. matplotlib's FigureCanvasQT.keyPressEvent (3.11.1,
    backends/backend_qt.py) neither calls super() nor ignores the event,
    so Qt treats it as handled and never propagates it up to the
    QMainWindow. The canvas holds focus because it is the only focusable
    widget here, so nothing else can ever see a key.

    Ignoring the event hands it back to Qt's propagation, which walks up
    to SpotCheck.keyPressEvent. Mouse handling is untouched -- it goes
    through mpl_connect and never needed focus.

    This was invisible to a test that called keyPressEvent directly, and
    the repo's first version of that test did exactly that. Anything
    claiming the keyboard works has to go through QTest.keyClick into the
    real focus widget, inside a running event loop.
    """

    def keyPressEvent(self, event):
        event.ignore()


class Queue:
    """Every (shard, crop, page) in the bundle, minus what is already done.

    Built once from the shard INDEXES only -- never from the pixels --
    so opening a 30k-crop bundle costs a few index reads rather than a
    few gigabytes.
    """

    def __init__(self, bundle_dir, log, per_page=VIEW.PER_PAGE,
                 max_per_crop=None, reviewer=None, shuffle=True,
                 overlap_frac=OVERLAP_FRAC, seed=None):
        self.dir = str(bundle_dir)
        self.log = log
        every = []
        for shard in B.shard_paths(self.dir):
            for row in B.read_index(shard):
                n = int(row['n_candidates'])
                if n == 0:
                    continue          # nothing to judge, not a skipped page
                # THE REVIEW BUDGET, and it lives here rather than in the
                # bundle. The extractor keeps every candidate the data
                # produced; how many of them a person is asked to judge is
                # a per-session choice, and applying it here costs nothing
                # and destroys nothing. Candidates are stored gate-pass
                # first, so a cap always keeps the informative ones.
                if max_per_crop:
                    n = min(n, int(max_per_crop))
                for pi, ix in enumerate(VIEW.pages_of(n, per_page)):
                    every.append((shard, row, pi, ix))
        order = (self._mixed(every, reviewer, overlap_frac, seed)
                 if shuffle else every)
        # DONE PAGES ARE DROPPED AFTER THE ORDER IS FIXED, so a reviewer
        # who stops and comes back gets the same sequence minus what they
        # finished, rather than a resequenced queue.
        done = log.done_pages()
        self.items = [it for it in order if (it[1]['key'], it[2]) not in done]
        self.i = 0

    @staticmethod
    def _mixed(every, reviewer, overlap_frac, seed):
        """A per-reviewer order, with a shared slice everyone sees.

        THE SERIAL ORDER WAS A SAMPLING BUG, not an inconvenience. The
        bundle is written FOV by FOV, hybe by hybe, cell by cell, and
        nobody reviews 18,000 pages -- so wherever a reviewer stopped,
        the labels covered the first few FOVs of the first few hybes and
        nothing else. A model trained on that has seen one corner of the
        experiment. Worse, done_pages() is per reviewer, so ten people
        starting together all judged the SAME first pages: ten times the
        duplication and a tenth of the coverage.

        Shuffling per reviewer fixes coverage but destroys the other
        thing overlapping assignments are for -- with 18,000 pages and
        200 judged each, two people would share about two pages, which
        measures no agreement at all. So a fraction of every stream comes
        from ONE bundle-wide shuffle that every reviewer walks in the
        same order: those pages get judged by everybody, and the rest
        spreads out.

        Both streams are seeded, so a reviewer who resumes tomorrow --
        or a second window today -- continues the same sequence.
        """
        import hashlib
        import random
        base = (str(seed) if seed is not None
                else hashlib.sha256(str(len(every)).encode()).hexdigest())
        shared = list(every)
        random.Random('shared:' + base).shuffle(shared)
        if not reviewer:
            return shared
        mine = list(every)
        random.Random('own:' + base + ':' + str(reviewer)).shuffle(mine)

        frac = min(max(float(overlap_frac), 0.0), 1.0)
        if frac <= 0:
            return mine
        step = max(2, int(round(1.0 / frac)))
        out, seen = [], set()
        si = mi = 0
        while si < len(shared) or mi < len(mine):
            take_shared = (len(out) % step == 0) and si < len(shared)
            src, idx = ((shared, si) if take_shared
                        else (mine, mi) if mi < len(mine) else (shared, si))
            while idx < len(src):
                it = src[idx]
                idx += 1
                k = (it[1]['key'], it[2])
                if k not in seen:
                    seen.add(k)
                    out.append(it)
                    break
            if src is shared:
                si = idx
            else:
                mi = idx
        return out

    def __len__(self):
        return len(self.items)

    def current(self):
        return self.items[self.i] if 0 <= self.i < len(self.items) else None

    def advance(self, step=1):
        """Move, and allow running one PAST the end.

        It used to clamp to the last index, so the final page never
        advanced: it redisplayed itself with the reviewer's keeps cleared
        from the screen, the completion screen was unreachable, and the
        natural response -- press Space again -- filed an empty verdict
        that superseded the real one and turned that page's kept spots
        into confirmed negatives. Clamping only at the bottom.
        """
        self.i = max(0, min(len(self.items), self.i + step))
        return self.current()


class SpotCheck(QtWidgets.QMainWindow):

    def __init__(self, bundle_dir, reviewer, per_page=VIEW.PER_PAGE,
                 max_per_crop=None, shuffle=True,
                 overlap_frac=OVERLAP_FRAC):
        super().__init__()
        self.bundle_dir = str(bundle_dir)
        self.per_page = int(per_page)
        self.log = V.VerdictLog(self.bundle_dir, reviewer)
        self.max_per_crop = max_per_crop
        self.queue = Queue(self.bundle_dir, self.log, self.per_page,
                           max_per_crop=max_per_crop, reviewer=reviewer,
                           shuffle=shuffle, overlap_frac=overlap_frac)

        self.setWindowTitle(f'Spot Check — {reviewer} — {self.bundle_dir}')
        self.resize(1750, 780)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        lay = QtWidgets.QVBoxLayout(central)

        # WHAT THE PAGE IS, in Qt rather than in the figure. Matplotlib
        # charges ~0.48 ms per glyph to rasterize text here, so this line
        # cost ~65 ms of every repaint while it was an axes title.
        self.header = QtWidgets.QLabel()
        self.header.setStyleSheet(
            'font-family: monospace; font-size: 12px; font-weight: bold;'
            ' padding: 4px 3px 1px 3px;')
        lay.addWidget(self.header)

        self.fig = Figure(figsize=(15.0, 5.6), dpi=110)
        self.canvas = KeyPassingCanvas(self.fig)
        self.canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        lay.addWidget(self.canvas, 1)

        self.status = QtWidgets.QLabel()
        self.status.setStyleSheet('font-family: monospace; padding: 3px;')
        lay.addWidget(self.status)

        self.help = QtWidgets.QLabel(
            '1-4 keep/drop   Space commit+next   Backspace back   '
            'A add missed (then click the cell)   U undo add   '
            'S skip unlabelled   Q quit          '
            'DEFAULT IS DROP — press a number only for a real spot\n'
            + VIEW.PANEL_LEGEND)
        self.help.setStyleSheet('color:#555; padding: 2px;')
        lay.addWidget(self.help)

        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.canvas.mpl_connect('draw_event', self._on_draw)
        self._state = None
        self._adding = False
        self._snapped = None
        self._outside = False
        self._offpage = None
        self._committed = set()
        # UNCOMMITTED WORK, PER PAGE. Backspace moves the queue and _load
        # rebuilds the page from the FILE, so keeps not yet committed were
        # simply gone -- and at queue position 0 Backspace did not even
        # move, it just silently wiped the page the reviewer was working
        # on. Worse than losing them: the reviewer comes forward again,
        # sees an empty page, presses Space, and the spots they had marked
        # are filed as confirmed negatives.
        self._draft = {}
        self._art = None
        self._bg = None
        self._t0 = time.time()
        self._load()

    # -- data ------------------------------------------------------------

    def _stash(self):
        """Remember this page's uncommitted keeps before leaving it."""
        s = self._state
        if s is None:
            return
        self._draft[(str(s['row']['key']), int(s['page']))] = (
            set(s['accepted']), list(s['added']))

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
        shown_n = (min(len(rows), int(self.max_per_crop))
                   if self.max_per_crop else len(rows))
        # THE OVERVIEW SHOWS WHAT CAN BE JUDGED, and nothing else. It used
        # to draw every candidate in the crop, so a busy cell put 338
        # circles and 338 numbers on the image when only the top 16 were
        # ever reachable by a keystroke -- 322 marks the reviewer cannot
        # act on, and MEASURED 1730 ms a page against 640 ms for a cell of
        # the same size with a normal candidate count.
        n_total = len(rows)
        rows = rows[:shown_n]
        npage = len(VIEW.pages_of(shown_n, self.per_page))
        # What this reviewer already recorded for THIS page, if anything.
        # Without it, stepping back with Backspace showed a committed page
        # with every keep wiped from the screen -- the display contradicting
        # the file -- and one more Space then overwrote the real verdict
        # with an empty one.
        prior = self.log.page_verdict(row['key'], page)
        draft = self._draft.get((str(row['key']), int(page)))
        # The store path comes from the shard's own meta, so a verdict can
        # say where its pixels came from and be re-cut after the bundle
        # is deleted (verdicts.recut).
        meta, _n, _v = B.read_meta(shard)
        self._state = dict(shard=shard, row=row, page=page, ix=ix,
                           stack=stack, mask=mask, cands=rows, npage=npage,
                           store=meta.get('storage_path'),
                           n_total=n_total,
                           accepted=(set(draft[0]) if draft else
                                     set(prior['accepted']) if prior else set()),
                           added=(list(draft[1]) if draft else
                                  list(prior['added']) if prior else []),
                           revisited=bool(prior))
        self._adding = False
        self._outside = False
        self._offpage = None
        # AND THE SNAP BANNER. It was set on a click and never cleared, so
        # "clicked ON candidate #7 -- kept it" stayed under every later
        # page for the rest of the session, claiming a keep on crops the
        # reviewer had not touched.
        self._snapped = None
        self._art = None
        self._bg = None
        self._t0 = time.time()
        self._draw()

    def _draw(self, restyle_only=False):
        """Repaint the page.

        A KEEP-TOGGLE DOES NOT REDRAW THE IMAGES. Pressing 1-4 changes
        circle colours and nothing else, yet matplotlib rasterizes the
        whole figure for any change -- MEASURED 542 ms, against 0.1 ms to
        actually restyle the artists. Over an 870-page assignment that is
        twelve minutes spent watching nine identical images be painted
        again.

        So the toggle path BLITS. Everything restyle() can touch is
        marked animated, which takes it out of the normal draw; the
        rasterized page underneath is cached on the draw_event that
        follows, and a toggle restores that cache and redraws only the
        ~50 changed artists. Any failure falls back to a real draw,
        because a stale blit is a screen that lies about what is
        recorded.
        """
        s = dict(self._state)
        s['snapped'] = self._snapped
        row = s['row']
        if restyle_only and self._art is not None:
            VIEW.restyle(self._art, s['accepted'])
            if not self._blit():
                self.canvas.draw_idle()
        else:
            header = (f"FOV{int(row['fov']):03d}   {row['hybe']}   "
                      f"ch{int(row['channel'])}   cell {int(row['cell'])}")
            self._art = VIEW.draw_page(
                self.fig, s['stack'], s['mask'], s['cands'], s['ix'],
                header=header, accepted=s['accepted'], added=s['added'],
                page=s['page'], npage=s['npage'], per_page=self.per_page,
                n_total=s['n_total'])
            for a in VIEW.mutable_artists(self._art):
                a.set_animated(True)
            self._bg = None              # invalid until the next draw lands
            self.canvas.draw_idle()
        self._set_status(s)

    def _on_draw(self, _event):
        """Cache the rasterized page as the blit background.

        Fires at the end of every real draw, including the ones Qt does
        on its own for a resize or an expose -- which is exactly when the
        old cache stopped matching the screen.
        """
        if self._art is None:
            return
        try:
            self._bg = self.canvas.copy_from_bbox(self.fig.bbox)
            self._draw_mutable()
        except Exception:                                   # noqa: BLE001
            self._bg = None

    def _draw_mutable(self):
        for a in VIEW.mutable_artists(self._art):
            self.fig.draw_artist(a)

    def _blit(self):
        """Restore the cached page and repaint only the changed artists.

        Returns False -- meaning "do a real draw instead" -- whenever the
        cache is missing or anything goes wrong.
        """
        if self._bg is None or self._art is None:
            return False
        try:
            self.canvas.restore_region(self._bg)
            self._draw_mutable()
            self.canvas.blit(self.fig.bbox)
            return True
        except Exception:                                   # noqa: BLE001
            self._bg = None
            return False

    def _status_only(self):
        """Repaint the words, not the picture.

        Add mode, its cancel, and the two refusal messages change a Qt
        label and nothing in the figure, but each went through _draw() --
        MEASURED 366-434 ms to rebuild nine imshows for a frame in which
        no pixel of the plot differs.
        """
        if self._state is None:
            return
        s = dict(self._state)
        s['snapped'] = self._snapped
        self._set_status(s)

    def _set_status(self, s):
        kept = ', '.join(f'#{i + 1}' for i in sorted(s['accepted'])) or 'none'
        if self._art is not None:
            self.header.setText(
                self._art['header_text']
                + f'   |   accepted so far: {len(s["accepted"])}'
                + (f' (+{len(s["added"])} added)' if s['added'] else ''))
        self.status.setText(
            ('RE-VISITING a page you already judged -- Space re-commits it   |  '
             if s.get('revisited') else '')
            + (f'clicked ON candidate #{s["snapped"] + 1} -- kept it '
               f'instead of adding a duplicate   |  '
               if s.get('snapped') is not None else '')
            + ('THAT CLICK WAS OUTSIDE THE CELL IMAGE -- nothing recorded. '
               'Click on the picture itself.   |  ' if self._outside else '')
            + (f'THAT IS CANDIDATE #{self._offpage[0] + 1}, JUDGED ON PAGE '
               f'{self._offpage[1]} -- nothing recorded here. Keep it there, '
               f'where its YX and ZX panels are.   |  '
               if self._offpage else '')
            + f'queue {self.queue.i + 1}/{len(self.queue)}   '
            f'|  keeping: {kept}   '
            f'|  added: {len(s["added"])}   '
            + ('|  CLICK THE CELL TO ADD A SPOT (Esc cancels)'
               if self._adding else ''))

    def _finish(self):
        # CLEAR THE STATE. Leaving the last page in _state meant a Space
        # pressed on the completion screen re-committed it -- a duplicate
        # record for a page already judged, and since merge() takes the
        # later line, an empty one would have superseded the real verdict.
        # Every key handler below returns early on _state is None.
        self._state = None
        self._adding = False
        self._outside = False
        self._offpage = None
        self._art = None
        self._bg = None
        self.header.setText('')
        self.fig.clear()
        # SAY WHICH ENDING THIS IS. One message covered three situations
        # and was wrong in two of them: a bundle with nothing in it, and a
        # bundle this reviewer had already finished on an earlier day,
        # both opened straight to "thank you, your verdicts are saved" --
        # thanking someone for work they had not done, and giving a person
        # handed an empty or mis-built bundle no hint that anything was
        # wrong with it.
        if self._committed:
            # PAGES, not commits. Re-checking a page with Backspace and
            # committing it again is one page judged; counting the writes
            # let the tally run past the size of the bundle.
            msg = ('That was the last page.\n'
                   f'{len(self._committed)} judged this session — '
                   'your verdicts are saved.')
        elif len(self.queue) == 0 and self.log.done_pages():
            msg = ('You have already reviewed every page of this bundle.\n'
                   'Nothing further to do here.')
        elif len(self.queue) == 0:
            msg = ('This bundle has no pages to review.\n'
                   'Either it holds no candidates, or it is not a bundle.')
        else:
            msg = 'Nothing left to review in this bundle.'
        ax = self.fig.add_subplot(111); ax.set_axis_off()
        ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=14)
        self.canvas.draw_idle()
        self.status.setText(f'log: {self.log.path}')

    # -- input -----------------------------------------------------------

    def _on_click(self, ev):
        if not (self._adding and self._state and ev.inaxes is not None
                and ev.xdata is not None):
            return
        # LEFT BUTTON ONLY. Any button reached this, so a right-click --
        # which on a plot is a reflex, not a decision -- filed a spot the
        # reviewer never claimed to see.
        if ev.button != 1:
            return
        # Only the cell overview accepts an added spot: its axes is the
        # one whose coordinates are crop-local (y, x), which is the frame
        # the verdict is recorded in.
        if ev.inaxes is not self.fig.axes[0]:
            return
        y, x = float(ev.ydata), float(ev.xdata)
        # AND INSIDE THE IMAGE. This is an invariant, not a fix for an
        # observed bug: imshow's equal aspect with adjustable='box' shrinks
        # the axes to hug the image, so today a click in the blank band
        # beside a tall crop reports inaxes=None and never gets here
        # (VERIFIED on a 90x30 crop -- the axes bbox and the image bbox are
        # the same rectangle). It is cheap insurance for the day someone
        # sets an explicit xlim or switches to adjustable='datalim', when a
        # click outside the pixels would otherwise be filed as a label
        # pointing at voxels that do not exist and recut() would fetch from
        # the wrong place. Keep add mode on and say so, rather than
        # swallowing the click.
        h, w = self._state['stack'].shape[0], self._state['stack'].shape[1]
        if not (-0.5 <= y <= h - 0.5 and -0.5 <= x <= w - 0.5):
            self._outside = True
            self._status_only()
            return
        self._outside = False
        # A CLICK ON AN EXISTING CANDIDATE ACCEPTS IT, never adds a
        # duplicate. This is not a nicety: a reviewer did exactly this --
        # hand-added two spots that were already candidates, because the
        # off-page circles were drawn too faint to see. That produces a
        # label the training set counts twice and a judgement nobody
        # needed to make. Snapping means the mistake is impossible rather
        # than merely less likely.
        near, best = None, ADD_SNAP_PX ** 2
        for i, c in enumerate(self._state['cands']):
            d2 = (float(c[0]) - y) ** 2 + (float(c[1]) - x) ** 2
            if d2 < best:
                near, best = i, d2
        if near is not None and near not in self._state['ix']:
            # AN OFF-PAGE CANDIDATE CANNOT BE KEPT FROM HERE, and pretending
            # otherwise turned a confirmed spot into a confirmed negative.
            # The whole crop's circles are on the overview and all of them
            # are clickable, but commit() writes only this page's indices
            # (`for i in page_ix`), so a keep on any other index vanished
            # from the record -- while the screen gave three confirmations
            # that it had landed: the circle went green, the status said
            # "kept it instead of adding a duplicate", and the header
            # counted it. The page that owns that candidate then loaded
            # with nothing accepted, and one Space filed the reviewer's
            # explicit yes as a hard negative.
            #
            # Refusing is not a limitation to route around: that candidate
            # gets judged on its own page, with its own YX and ZX panels,
            # which is a better look at it than a click on the overview.
            self._offpage = (near, self._page_of(near))
            self._snapped = None
            self._status_only()
            return
        if near is not None:
            self._state['accepted'].add(near)
            self._snapped = near
            self._offpage = None
            self._adding = False
        else:
            self._state['added'].append((y, x))
            self._snapped = None
            self._offpage = None
            self._adding = False
        self._draw()

    def _page_of(self, i):
        """Which page of this crop judges candidate `i` (1-based)."""
        for pi, ix in enumerate(VIEW.pages_of(len(self._state['cands']),
                                              self.per_page)):
            if i in ix:
                return pi + 1
        return None

    def keyPressEvent(self, e):
        s = self._state
        k = e.key()
        # Q QUITS, ALWAYS -- including in add mode, where it used to do
        # nothing at all and left the only way out as a click.
        if k == QtCore.Qt.Key_Q:
            self.close(); return
        # ESCAPE CANCELS, AND NEVER QUITS. It closed the window and threw
        # away the page's uncommitted keeps, which is the opposite of what
        # Escape means everywhere else and is one fumbled keystroke away
        # from the reviewer's last few minutes of work.
        if k == QtCore.Qt.Key_Escape:
            if self._adding:
                self._adding = False
                self._outside = False
                self._offpage = None
                self._status_only()
            return
        # BACKSPACE WORKS ON THE COMPLETION SCREEN. It has no _state, and
        # the old guard returned before reaching this, so a reviewer who
        # pressed Space once too often was stranded on "thank you" with no
        # way back to the page they had just committed.
        if k == QtCore.Qt.Key_Backspace:
            if self.queue.i == 0 and s is not None:
                # NOWHERE TO GO. This used to reload page 0, which threw
                # away every uncommitted keep on it -- Backspace is the
                # universal undo reflex and the app's own U undoes only
                # adds, so it was the likeliest key to press here.
                self._outside = False
                self._offpage = None
                self.status.setText(
                    'ALREADY AT THE FIRST PAGE -- nothing behind it.   |  '
                    + self.status.text())
                return
            self._stash()
            self.queue.advance(-1); self._load(); return
        if s is None:
            return
        if QtCore.Qt.Key_1 <= k <= QtCore.Qt.Key_9:
            slot = k - QtCore.Qt.Key_1
            if slot < len(s['ix']):
                i = s['ix'][slot]
                s['accepted'].symmetric_difference_update({i})
                # AND DROP THE SNAP BANNER. "clicked ON candidate #3 --
                # kept it" was rebuilt from _snapped on every restyle, so
                # pressing 3 to drop that very candidate left one status
                # line asserting both the keep and "keeping: none". The
                # cross-page case was fixed in _load; this is the same
                # sentence going stale within the page.
                self._snapped = None
                self._draw(restyle_only=True)
            return
        if k == QtCore.Qt.Key_A:
            self._adding = True; self._status_only(); return
        if k == QtCore.Qt.Key_U:
            if s['added']:
                s['added'].pop(); self._draw()
            return
        if k == QtCore.Qt.Key_S:
            # No record at all. A crop the reviewer cannot judge must stay
            # UNLABELLED -- committing it empty would file it as four
            # confirmed negatives, which is a lie the model would learn.
            self._stash()
            self.queue.advance(1); self._load(); return
        if k == QtCore.Qt.Key_Space:
            # The whole index row and the whole candidate list go in, so
            # the record can carry coordinates and crop geometry rather
            # than positions in a list this bundle happens to have.
            # A WRITE THAT FAILS MUST NOT TAKE THE WINDOW WITH IT. An
            # unhandled exception inside a Qt slot aborts the process --
            # MEASURED exit 127, no traceback, no message -- so a share
            # going read-only mid-session looked like the app vanishing,
            # and the reviewer had no way to know which page was the last
            # one saved. Stay on the page, say what happened, and let them
            # retry once the disk is back.
            try:
                self.log.commit(s['row'], s['page'], s['ix'], s['cands'],
                                s['accepted'], added=s['added'],
                                seconds=time.time() - self._t0,
                                bundle=os.path.basename(s['shard']),
                                store=s['store'])
            except Exception as exc:                        # noqa: BLE001
                self._stash()
                QtWidgets.QMessageBox.critical(
                    self, 'Spot Check -- verdict NOT saved',
                    'This page could not be written:\n\n'
                    f'{type(exc).__name__}: {exc}\n\n'
                    f'{self.log.path}\n\n'
                    'Your keeps are still on screen. Fix the disk or the '
                    'share and press Space again. Nothing already saved '
                    'is affected.')
                self.status.setText('VERDICT NOT SAVED -- see the message; '
                                    'press Space to retry.   |  '
                                    + self.status.text())
                return
            self._committed.add((s['row']['key'], s['page']))
            self._draft.pop((str(s['row']['key']), int(s['page'])), None)
            self.queue.advance(1); self._load(); return
        if k == QtCore.Qt.Key_Backspace:
            self.queue.advance(-1); self._load(); return
        super().keyPressEvent(e)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('bundle_dir', nargs='?', default=None)
    ap.add_argument('--reviewer', default=None,
                    help='your name/initials -- names the verdict file, so '
                         'ten people can share one bundle directory')
    ap.add_argument('--per-page', type=int, default=VIEW.PER_PAGE)
    ap.add_argument('--no-shuffle', action='store_true',
                    help='walk the bundle in file order. Only for looking at '
                         'one FOV deliberately -- a review stopped partway '
                         'through file order labels the first few FOVs and '
                         'nothing else.')
    ap.add_argument('--overlap-frac', type=float, default=OVERLAP_FRAC,
                    help=f'fraction of pages drawn from the order EVERY '
                         f'reviewer walks, so agreement between people is '
                         f'measurable (default {OVERLAP_FRAC})')
    ap.add_argument('--max-per-crop', type=int, default=DEFAULT_MAX_PER_CROP,
                    help='judge at most this many candidates per crop '
                         f'(default {DEFAULT_MAX_PER_CROP}). The bundle keeps '
                         'every candidate the data produced; this bounds only '
                         'what a person is shown, so raising it later needs no '
                         're-extraction. Candidates are ordered gate-pass, then '
                         'fitted, then by quality, so a cap keeps the confident '
                         'spots AND the boundary cases either side of the gate '
                         '-- which is where a detector learns. Pass 0 for no '
                         'limit.')
    a = ap.parse_args(argv)
    app = QtWidgets.QApplication(sys.argv[:1])
    # Whatever was not given on the command line is ASKED FOR, so the
    # file can simply be double-clicked.
    bundle_dir, reviewer = ask_session(a.bundle_dir, a.reviewer)
    if not bundle_dir or not reviewer:
        return 0
    if not os.path.isdir(bundle_dir):
        QtWidgets.QMessageBox.critical(
            None, 'Spot Check', 'No such bundle folder:\n' + str(bundle_dir))
        return 1
    w = SpotCheck(bundle_dir, reviewer, a.per_page,
                  max_per_crop=(a.max_per_crop or None),
                  shuffle=not a.no_shuffle, overlap_frac=a.overlap_frac)
    if len(w.queue) == 0:
        print('nothing left to review in', a.bundle_dir)
    w.show()
    w.canvas.setFocus()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
