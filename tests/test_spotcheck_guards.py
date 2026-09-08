"""
The ways a reviewer can lose work by accident, and the ones they cannot.

Every case here is a gesture a tired person makes at 11pm: a fumbled
Escape, a right-click on a plot, one Space too many, a second window.
None of them may cost a verdict, and none may file one nobody made.

The keyboard goes through QTest.keyClick into the real focus widget, for
the reason test_spotcheck_keys.py explains at length -- calling the
handler directly proves nothing about whether the key arrives. Mouse
events go through canvas.callbacks.process, which is the same dispatch
mpl_connect uses, with the event built from DATA coordinates so the test
says where on the picture the person clicked rather than where on the
screen.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_spotcheck_guards.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# THE CONSOLE HERE IS cp949, AND THIS FILE PRINTS THE APP'S OWN WORDS.
# Without this the suite dies on the first em dash it echoes back -- at
# check 18 of 41, on a PASSING check, with no summary line -- so a green
# run and a broken run look the same at a glance and the later test
# groups never execute at all. It was invisible while every run happened
# to carry PYTHONIOENCODING=utf-8.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


import numpy as np                                          # noqa: E402
from PyQt5 import QtCore, QtWidgets                          # noqa: E402
from PyQt5.QtTest import QTest                               # noqa: E402
from matplotlib.backend_bases import MouseEvent              # noqa: E402
from matplotlib.figure import Figure                         # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402

from codelab_pipeline.training import bundle as B            # noqa: E402
from codelab_pipeline.training import verdicts as V          # noqa: E402
from codelab_pipeline.training import view as VIEW           # noqa: E402
import spotcheck.app as A                                    # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}'
          + (f'   [{detail}]' if detail else ''))


def make_bundle(d, n_crops=2, h=90, w=30, n_cands=6):
    """A TALL, NARROW crop on purpose: the shape that would leave blank
    bands beside the image if the axes did not hug it."""
    rng = np.random.default_rng(3)
    with B.BundleWriter(os.path.join(d, 'fov001__Hyb_001__c000.h5'),
                        meta={'storage_path': os.path.join(d, 'nostore'),
                              'hybe': 'Hyb_001', 'channel': 555,
                              'pad': 14}) as bw:
        for c in range(n_crops):
            stack = rng.integers(200, 400, (h, w, 25)).astype(np.uint16)
            mask = np.zeros((h, w), np.uint8)
            mask[5:h - 5, 5:w - 5] = 1
            cands = [(10.0 + 6 * k, 8.0 + k, 8.0 + k, 0.9 - 0.05 * k,
                      1, 1 if k < 2 else 0, '' if k < 2 else 'occupancy low')
                     for k in range(n_cands)]
            bw.add(1, 'Hyb_001', 555, c + 1, stack, mask, 0, 0, cands)


def app_on(d, reviewer='guard'):
    app = (QtWidgets.QApplication.instance()
           or QtWidgets.QApplication(sys.argv[:1]))
    w = A.SpotCheck(d, reviewer, max_per_crop=A.DEFAULT_MAX_PER_CROP)
    w.show()
    w.canvas.setFocus()
    for _ in range(3):
        app.processEvents()
    return app, w


def key(app, w, k):
    QTest.keyClick(QtWidgets.QApplication.focusWidget() or w, k)
    app.processEvents()


def click(app, w, ydata, xdata, button=1):
    """A click at a place on the cell overview, in image coordinates."""
    ax = w.fig.axes[0]
    px, py = ax.transData.transform((xdata, ydata))
    ev = MouseEvent('button_press_event', w.canvas, px, py, button=button)
    w.canvas.callbacks.process('button_press_event', ev)
    app.processEvents()
    return ev


# -- Escape and Q ---------------------------------------------------------

def test_escape_and_q():
    print('\n-- Escape cancels, Q quits, and neither is the other --')
    d = tempfile.mkdtemp()
    make_bundle(d)
    app, w = app_on(d, 'esc')

    key(app, w, QtCore.Qt.Key_1)
    kept = set(w._state['accepted'])
    key(app, w, QtCore.Qt.Key_Escape)
    check('Escape does not close the window', w.isVisible())
    check('Escape does not discard the page', w._state is not None)
    check('Escape does not discard what was already kept',
          w._state is not None and set(w._state['accepted']) == kept,
          str(sorted(kept)))

    key(app, w, QtCore.Qt.Key_A)
    check('A turns add mode on', w._adding)
    key(app, w, QtCore.Qt.Key_Escape)
    check('Escape turns add mode off', not w._adding)
    check('and still does not close the window', w.isVisible())

    key(app, w, QtCore.Qt.Key_A)
    check('A turns add mode on again', w._adding)
    key(app, w, QtCore.Qt.Key_Q)
    check('Q quits even from add mode, where it used to do nothing',
          not w.isVisible())
    w.close()


# -- the mouse ------------------------------------------------------------

def test_click_guards():
    print('\n-- only a left click on the picture adds a spot --')
    d = tempfile.mkdtemp()
    make_bundle(d)
    app, w = app_on(d, 'click')
    h, wd = w._state['stack'].shape[:2]

    key(app, w, QtCore.Qt.Key_A)
    n0 = len(w._state['added'])
    click(app, w, h / 2, wd / 2, button=3)
    check('a RIGHT click adds nothing', len(w._state['added']) == n0,
          f'{n0} -> {len(w._state["added"])}')
    check('and leaves add mode on, so the intended click still works',
          w._adding)
    click(app, w, h / 2, wd / 2, button=2)
    check('a MIDDLE click adds nothing', len(w._state['added']) == n0)

    # Where a click outside the pixels would land, if it could.
    ax = w.fig.axes[0]
    ev = click(app, w, h / 2, -20.0, button=1)
    check('a click in the blank band never reaches the axes at all '
          '(imshow hugs the image)', ev.inaxes is not ax,
          f'inaxes={ev.inaxes}')
    check('so nothing is added by it', len(w._state['added']) == n0)

    # The invariant itself, exercised directly: a synthetic event that
    # claims to be inside the axes but outside the pixels.
    ev = MouseEvent('button_press_event', w.canvas,
                    *ax.transData.transform((wd / 2, h / 2)), button=1)
    ev.inaxes = ax
    ev.xdata, ev.ydata = float(wd) + 5.0, float(h) / 2
    w._on_click(ev)
    app.processEvents()
    check('an out-of-crop coordinate is refused, not recorded',
          len(w._state['added']) == n0, str(w._state['added']))
    check('and the reviewer is told, with add mode still on',
          w._outside and w._adding)

    key(app, w, QtCore.Qt.Key_A)
    click(app, w, h - 8.0, wd / 2, button=1)
    check('a left click on the picture DOES add a spot',
          len(w._state['added']) == n0 + 1, str(w._state['added']))
    w.close()


def test_offpage_snap_is_refused():
    """A KEEP THAT CANNOT BE RECORDED MUST NOT BE CLAIMED.

    Every candidate in the crop is drawn on the overview and every circle
    is clickable, but commit() writes only the current page's indices. A
    click that snapped to an off-page candidate therefore turned green,
    lit the status bar and bumped the header count, and then vanished:
    the page that owned it loaded with nothing accepted and one Space
    filed the reviewer's explicit yes as a confirmed hard negative.
    """
    print('\n-- a click on a candidate belonging to another page --')
    d = tempfile.mkdtemp()
    # 8 candidates, 4 a page: #5-#8 are off-page while page 1 is shown.
    make_bundle(d, n_crops=1, n_cands=8)
    app, w = app_on(d, 'offpage')
    s = w._state
    check('page 1 judges the first four', list(s['ix']) == [0, 1, 2, 3],
          str(list(s['ix'])))
    off = 6                                   # candidate #7, on page 2
    oy, ox = float(s['cands'][off][0]), float(s['cands'][off][1])

    key(app, w, QtCore.Qt.Key_A)
    click(app, w, oy, ox, button=1)
    check('it is NOT accepted, because this page cannot record it',
          off not in w._state['accepted'], str(sorted(w._state['accepted'])))
    check('and it is NOT filed as a new hand-added spot either',
          not w._state['added'], str(w._state['added']))
    check('the reviewer is told which page judges it',
          'PAGE 2' in w.status.text(), w.status.text()[:90])

    # The whole failure, end to end: commit both pages and read the labels.
    key(app, w, QtCore.Qt.Key_Space)          # page 1, nothing kept
    key(app, w, QtCore.Qt.Key_2)              # page 2: keep its 2nd card
    kept_ix = sorted(w._state['accepted'])
    key(app, w, QtCore.Qt.Key_Space)
    e = V.labels(d)[str(B.read_index(B.shard_paths(d)[0])[0]['key'])]
    neg = {(round(p[0], 1), round(p[1], 1)) for p in e['negative']}
    pos = {(round(p[0], 1), round(p[1], 1)) for p in e['positive']}
    here = (round(oy, 1), round(ox, 1))
    check('candidate #7 was judged on its own page, not lost',
          here in pos or here in neg, f'{here} pos={len(pos)} neg={len(neg)}')
    check('and page 2 recorded the keep that was actually made there',
          kept_ix == [5], str(kept_ix))
    w.close()


# -- the end of the queue -------------------------------------------------

def test_completion_screen():
    print('\n-- the completion screen is not a dead end --')
    d = tempfile.mkdtemp()
    make_bundle(d, n_crops=1)
    app, w = app_on(d, 'fin')
    n = len(w.queue)
    check('the queue has pages', n > 0, str(n))
    for _ in range(n + 1):
        key(app, w, QtCore.Qt.Key_Space)
    check('running past the last page reaches the completion screen',
          w._state is None)
    txt = ' '.join(t.get_text() for ax in w.fig.axes for t in ax.texts)
    check('which says the work is done, and how much of it',
          'last page' in txt and str(n) in txt, txt.replace('\n', ' | '))
    key(app, w, QtCore.Qt.Key_Backspace)
    check('Backspace from the completion screen goes back to a page',
          w._state is not None,
          'stranded' if w._state is None else str(w._state['page']))
    w.close()


def test_empty_and_already_done():
    print('\n-- an empty bundle is not thanked for work it has not done --')
    d = tempfile.mkdtemp()          # no shards at all
    app, w = app_on(d, 'empty')
    check('an empty bundle opens on the completion screen', w._state is None)
    txt = ' '.join(t.get_text() for ax in w.fig.axes for t in ax.texts)
    check('and says the bundle is empty, not "thank you"',
          'no pages to review' in txt and 'saved' not in txt,
          txt.replace('\n', ' | '))
    w.close()

    d2 = tempfile.mkdtemp()
    make_bundle(d2, n_crops=1)
    app, w = app_on(d2, 'twice')
    for _ in range(len(w.queue) + 1):
        key(app, w, QtCore.Qt.Key_Space)
    w.close()
    _app, w2 = app_on(d2, 'twice')          # same reviewer, second session
    check('re-opening a finished bundle goes straight to the end',
          w2._state is None)
    txt = ' '.join(t.get_text() for ax in w2.fig.axes for t in ax.texts)
    check('and says it was already reviewed, not that work was just saved',
          'already reviewed' in txt, txt.replace('\n', ' | '))
    w2.close()


# -- the log --------------------------------------------------------------

def test_page_verdict_cache():
    print('\n-- page_verdict serves the latest verdict without re-reading --')
    d = tempfile.mkdtemp()
    make_bundle(d, n_crops=1)
    log = V.VerdictLog(d, 'cache', session='fixed-1')
    shard = B.shard_paths(d)[0]
    row = B.read_index(shard)[0]
    cands = [(10.0, 8.0, 8.0, 0.9, 1, 1, ''), (16.0, 9.0, 9.0, 0.8, 1, 1, '')]

    check('an unjudged page has no verdict',
          log.page_verdict(str(row['key']), 0) is None)
    log.commit(row, 0, [0, 1], cands, {0})
    got = log.page_verdict(str(row['key']), 0)
    check('after a commit it is served', got is not None and got['accepted'] == [0],
          str(got))
    log.commit(row, 0, [0, 1], cands, {1})
    got = log.page_verdict(str(row['key']), 0)
    check('a re-commit supersedes it, matching merge()',
          got is not None and got['accepted'] == [1], str(got))

    # A fresh log over the same file must see what is on disk, since the
    # cache starts empty.
    log2 = V.VerdictLog(d, 'cache', session='fixed-1')
    got2 = log2.page_verdict(str(row['key']), 0)
    check('a new session reading the same file seeds from disk',
          got2 is not None and got2['accepted'] == [1], str(got2))

    # It must not read the file once per page.
    reads = {'n': 0}
    real = V.read_log

    def counted(p):
        reads['n'] += 1
        return real(p)
    V.read_log = counted
    try:
        for _ in range(50):
            log.page_verdict(str(row['key']), 0)
    finally:
        V.read_log = real
    check('50 lookups do not touch the file 50 times', reads['n'] == 0,
          f'{reads["n"]} reads')


def test_added_spots_are_voted_on():
    print('\n-- an added spot is one label, however many people marked it --')
    d = tempfile.mkdtemp()
    make_bundle(d, n_crops=1)
    shard = B.shard_paths(d)[0]
    row = B.read_index(shard)[0]
    cands = [(10.0, 8.0, 8.0, 0.9, 1, 1, '')]
    for who in ('ann', 'bob', 'cat'):
        log = V.VerdictLog(d, who, session='s1')
        log.commit(row, 0, [0], cands, set(), added=[(40.0, 12.0)])
    # cat also marks one nobody else did
    V.VerdictLog(d, 'cat', session='s2').commit(
        row, 1, [0], cands, set(), added=[(40.0, 12.0), (60.0, 20.0)])

    lab = V.labels(d)
    key0 = str(row['key'])
    e = lab.get(key0)
    check('the crop has labels', e is not None)
    if e is None:
        return
    check('the spot three reviewers added appears ONCE, not three times',
          len(e['added']) == 2, str(e['added']))
    votes = e.get('added_votes') or {}
    by_y = {round(k[0]): v for k, v in votes.items()}
    check('and carries how many of the reviewers marked it',
          by_y.get(40) == (3, 3), str(votes))
    check('while one person\'s lone addition is visibly a minority',
          by_y.get(60) == (1, 3), str(votes))
    check('the denominator is reviewers, not records',
          e['reviewers'] == 3, str(e['reviewers']))


def test_added_spots_merge_real_clicks():
    """THE CASE THE APP CAN ACTUALLY PRODUCE.

    Two people never click the same float. The first version of this
    merge keyed on round(v/0.5) -- a GRID -- so three clicks agreeing to
    0.4 px landed in three bins and one emitter became three labels, each
    reported as a lone 1-of-3 minority: agreement recorded as
    disagreement. A test with byte-identical coordinates passes on any
    implementation at all, including no merge whatsoever, so it proved
    nothing about the only input that occurs.
    """
    print('\n-- clicks that merely AGREE, as human clicks do --')
    d = tempfile.mkdtemp()
    make_bundle(d, n_crops=1)
    shard = B.shard_paths(d)[0]
    row = B.read_index(shard)[0]
    cands = [(10.0, 8.0, 8.0, 0.9, 1, 1, '')]
    # Three reviewers at one emitter, scattered as a hand scatters, and
    # deliberately straddling the 0.5 px bin edges at 40.25 and 12.25.
    marks = {'ann': (40.24, 12.10), 'bob': (40.26, 12.09),
             'cat': (40.31, 12.11)}
    for who, pt in marks.items():
        V.VerdictLog(d, who, session='s1').commit(
            row, 0, [0], cands, set(), added=[pt])
    e = V.labels(d)[str(row['key'])]
    check('three near-identical clicks become ONE added spot',
          len(e['added']) == 1, str(e['added']))
    v = list((e.get('added_votes') or {}).values())
    check('reported as unanimous, not as three lone opinions',
          v == [(3, 3)], str(e.get('added_votes')))

    # And it must not merge things that are genuinely apart.
    d2 = tempfile.mkdtemp()
    make_bundle(d2, n_crops=1)
    row2 = B.read_index(B.shard_paths(d2)[0])[0]
    V.VerdictLog(d2, 'ann', session='s1').commit(
        row2, 0, [0], cands, set(),
        added=[(20.0, 10.0), (34.0, 10.0), (48.0, 10.0)])
    e2 = V.labels(d2)[str(row2['key'])]
    check('spots 14 px apart stay three spots',
          len(e2['added']) == 3, str(e2['added']))

    # A chain of clicks 3 px apart must not smear into one spot.
    d3 = tempfile.mkdtemp()
    make_bundle(d3, n_crops=1)
    row3 = B.read_index(B.shard_paths(d3)[0])[0]
    V.VerdictLog(d3, 'ann', session='s1').commit(
        row3, 0, [0], cands, set(),
        added=[(20.0 + 3.0 * k, 10.0) for k in range(6)])
    e3 = V.labels(d3)[str(row3['key'])]
    check('a 15 px line of clicks does not chain into one spot',
          len(e3['added']) >= 2, str(e3['added']))


# -- the figure -----------------------------------------------------------

def test_figure_has_no_axis_furniture():
    print('\n-- the panels are pictures: no ticks, no labels, no boxes --')
    d = tempfile.mkdtemp()
    make_bundle(d, n_crops=1)
    shard = B.shard_paths(d)[0]
    row = B.read_index(shard)[0]
    stack, mask, cands, words = B.read_crop(shard, row['key'])
    rows = [(float(c['y']), float(c['x']), float(c['z']), float(c['p']),
             int(c['fit_ok']), int(c['gate_pass']), w)
            for c, w in zip(cands, words)]
    fig = Figure(figsize=(15, 5.6), dpi=110)
    FigureCanvasAgg(fig)
    art = VIEW.draw_page(fig, stack, mask, rows, VIEW.pages_of(len(rows), 4)[0],
                         header='h', page=0, npage=2, n_total=len(rows))
    fig.canvas.draw()

    img_axes = [ax for ax in fig.axes if ax.images and ax is not None]
    check('there are image panels to check', len(img_axes) >= 2,
          str(len(img_axes)))
    labelled = [ax for ax in img_axes
                if ax.get_xlabel() or ax.get_ylabel()]
    check('no image panel carries an axis label', not labelled,
          str([(a.get_xlabel(), a.get_ylabel()) for a in labelled]))
    # `axison` is the thing to assert, and the only thing. set_axis_off()
    # neither clears the tick locations nor unsets visibility on the axis
    # artists -- it sets this flag, and Axes.draw skips both axes when it
    # is False, which takes the ticks, the tick labels and the spines with
    # it. Probing xaxis.get_visible() reports True on a panel that draws
    # nothing.
    boxed = [ax for ax in img_axes if ax.axison]
    check('no image panel draws its axis -- ticks, labels or spines',
          not boxed, str(len(boxed)))

    # The card panels legitimately carry a title -- "[1] #1 YX @ z=64" is
    # per-spot and changes with the page. What moved to Qt is the PAGE
    # header, which lived on the overview.
    check('the page header is returned rather than drawn on the overview',
          bool(art['header_text']) and not art['axm'].get_title(),
          art['header_text'][:40])
    # A SEPARATE FIGURE. Drawing into `fig` again clears it, and every
    # Axes captured above becomes a stale object that is no longer in
    # fig.axes -- which is exactly how the colourbar check below came to
    # be pointing at the overview and passing on nothing.
    fig2 = Figure(figsize=(15, 5.6), dpi=110)
    FigureCanvasAgg(fig2)
    check('and it says a cap is a cap',
          'top ' in VIEW.draw_page(
              fig2, stack, mask, rows[:2], [0, 1], header='h',
              n_total=99)['header_text'])

    # THE COLOURBAR IS NOT IN fig.axes AT ALL -- inset_axes makes it a
    # child of the overview -- so draw_page names it on the art dict.
    # Scanning the figure for it found the overview and passed on
    # nothing, which is how a restored cb.set_label() went unnoticed.
    cax = art.get('cax')
    check('draw_page names the colourbar axes', cax is not None)
    if cax is not None:
        check('the colourbar carries no caption of its own',
              not cax.get_ylabel() and not cax.get_xlabel(),
              repr(cax.get_ylabel() or cax.get_xlabel()))
        check('while keeping its tick numbers, which are the data',
              len(cax.get_yticklabels()) > 0,
              str(len(cax.get_yticklabels())))

    check('the panel legend exists for the window to show',
          bool(VIEW.PANEL_LEGEND) and 'YX' in VIEW.PANEL_LEGEND)
    d2 = tempfile.mkdtemp()
    make_bundle(d2, n_crops=1)
    _app, w = app_on(d2, 'legend')
    check('and the window shows it',
          VIEW.PANEL_LEGEND in w.help.text())
    w.close()


def test_blit_shows_what_it_records():
    """THE SCREEN AND THE STATE MUST NOT DISAGREE.

    A keep-toggle no longer repaints the figure; it restores a cached
    background and redraws ~25 artists. That is the largest change in
    this app and the one that fails most quietly: if the blit does not
    land, the reviewer sees an unchanged page while `accepted` has
    changed underneath, and commits a verdict that does not match what
    is on screen. Nothing else in the suite would notice.

    So this asserts on PIXELS, not on flags.
    """
    print('\n-- a toggle actually repaints, without a full draw --')
    d = tempfile.mkdtemp()
    make_bundle(d, n_crops=1)
    app, w = app_on(d, 'blit')
    for _ in range(3):
        app.processEvents()
    check('a background was cached for blitting', w._bg is not None)

    before = np.asarray(w.canvas.buffer_rgba()).copy()
    full = {'n': 0}
    real_draw = type(w.canvas).draw

    def counted(self, *a, **k):
        full['n'] += 1
        return real_draw(self, *a, **k)
    type(w.canvas).draw = counted
    try:
        key(app, w, QtCore.Qt.Key_1)
        after = np.asarray(w.canvas.buffer_rgba()).copy()
    finally:
        type(w.canvas).draw = real_draw

    check('the toggle changed the state', 0 in w._state['accepted'],
          str(sorted(w._state['accepted'])))
    changed = int((before != after).any(axis=2).sum())
    check('and the pixels on screen changed with it', changed > 0,
          f'{changed} px')
    check('without a full figure redraw', full['n'] == 0, str(full['n']))

    # Toggling back must restore the pixels exactly -- a blit that leaks
    # would accumulate.
    key(app, w, QtCore.Qt.Key_1)
    back = np.asarray(w.canvas.buffer_rgba()).copy()
    check('toggling back restores the page exactly',
          bool((back == before).all()),
          f'{int((back != before).any(axis=2).sum())} px differ')

    # A page turn invalidates the cache and must rebuild it.
    key(app, w, QtCore.Qt.Key_Space)
    for _ in range(3):
        app.processEvents()
    if w._state is not None:
        check('a new page caches a fresh background', w._bg is not None)
    w.close()


def test_status_claims_expire_with_the_page():
    """A BANNER MUST NOT OUTLIVE WHAT IT DESCRIBES.

    `_snapped` was set by a click and never cleared, so "clicked ON
    candidate #7 -- kept it" sat under every later page for the rest of
    the session, claiming a keep on crops the reviewer had not touched.
    """
    print('\n-- what the status bar says stops being true elsewhere --')
    d = tempfile.mkdtemp()
    make_bundle(d, n_crops=2, n_cands=8)
    app, w = app_on(d, 'banner')
    on_page = w._state['ix'][1]
    cy, cx = float(w._state['cands'][on_page][0]), \
        float(w._state['cands'][on_page][1])
    key(app, w, QtCore.Qt.Key_A)
    click(app, w, cy, cx, button=1)
    check('a click on an ON-page candidate keeps it',
          on_page in w._state['accepted'], str(sorted(w._state['accepted'])))
    check('and the status says so', 'clicked ON' in w.status.text(),
          w.status.text()[:70])
    key(app, w, QtCore.Qt.Key_Space)
    check('the claim does not follow the reviewer to the next page',
          'clicked ON' not in w.status.text(), w.status.text()[:70])
    w.close()


def test_status_only_changes_do_not_redraw():
    """Add mode changes a Qt label, not a pixel of the plot."""
    print('\n-- turning add mode on does not rebuild the figure --')
    d = tempfile.mkdtemp()
    make_bundle(d, n_crops=1)
    app, w = app_on(d, 'statusonly')
    for _ in range(3):
        app.processEvents()
    before = np.asarray(w.canvas.buffer_rgba()).copy()
    full = {'n': 0}
    real = type(w.canvas).draw

    def counted(self, *a, **k):
        full['n'] += 1
        return real(self, *a, **k)
    type(w.canvas).draw = counted
    try:
        key(app, w, QtCore.Qt.Key_A)
        key(app, w, QtCore.Qt.Key_Escape)
    finally:
        type(w.canvas).draw = real
    after = np.asarray(w.canvas.buffer_rgba()).copy()
    check('A then Escape redraws the figure zero times', full['n'] == 0,
          str(full['n']))
    check('and leaves the plot pixel-identical',
          bool((before == after).all()),
          f'{int((before != after).any(axis=2).sum())} px differ')
    check('while the words did change', not w._adding)
    w.close()


def test_draw_page_survives_an_empty_page():
    """draw_page is public; an empty page must not be a crash."""
    print('\n-- a page with nothing on it --')
    fig = Figure(figsize=(15, 5.6), dpi=110)
    FigureCanvasAgg(fig)
    st = np.random.default_rng(0).integers(200, 400, (40, 40, 20)).astype(float)
    mk = np.ones((40, 40), np.uint8)
    try:
        art = VIEW.draw_page(fig, st, mk, [], [], header='h', page=0, npage=1)
        fig.canvas.draw()
        ok, why = True, art['header_text'][-28:]
    except Exception as exc:                                # noqa: BLE001
        ok, why = False, f'{type(exc).__name__}: {exc}'
    check('an empty page_ix draws instead of raising', ok, why)


def test_completion_counts_pages_not_commits():
    print('\n-- the session tally counts pages --')
    d = tempfile.mkdtemp()
    make_bundle(d, n_crops=1, n_cands=8)
    app, w = app_on(d, 'tally')
    n = len(w.queue)
    key(app, w, QtCore.Qt.Key_Space)
    key(app, w, QtCore.Qt.Key_Backspace)      # back to page 1
    key(app, w, QtCore.Qt.Key_Space)          # commit it a second time
    for _ in range(n + 1):
        key(app, w, QtCore.Qt.Key_Space)
    txt = ' '.join(t.get_text() for ax in w.fig.axes for t in ax.texts)
    check('a page committed twice counts once',
          f'{n} judged' in txt, txt.replace('\n', ' | ')[:80])
    w.close()


def main():
    test_escape_and_q()
    test_click_guards()
    test_offpage_snap_is_refused()
    test_status_claims_expire_with_the_page()
    test_status_only_changes_do_not_redraw()
    test_completion_screen()
    test_completion_counts_pages_not_commits()
    test_empty_and_already_done()
    test_page_verdict_cache()
    test_added_spots_are_voted_on()
    test_added_spots_merge_real_clicks()
    test_figure_has_no_axis_furniture()
    test_draw_page_survives_an_empty_page()
    test_blit_shows_what_it_records()
    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        for f in FAIL:
            print('  FAILED:', f)
        return 1
    print('ALL GOOD')
    return 0


if __name__ == '__main__':
    sys.exit(main())
