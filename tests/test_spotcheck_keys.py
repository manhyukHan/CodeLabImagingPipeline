"""
The review app answers a REAL keystroke, delivered the way Qt delivers one.

WHY THIS EXISTS AS ITS OWN TEST, and why the obvious version of it is
worthless: the app was completely dead to the keyboard and no test
noticed, because the test called SpotCheck.keyPressEvent directly.

That call bypasses everything that actually matters. matplotlib's
FigureCanvasQT.keyPressEvent neither calls super() nor ignores the event,
so Qt considers it handled and never propagates it to the QMainWindow --
and the canvas is the only focusable widget in the window, so it holds
focus and every key died there. The window drew, the mouse worked, and
1/2/3/4, Space, S and Q all did nothing. Ten people would have opened the
app, seen a page, and been unable to record a single verdict.

So every key here goes through QTest.keyClick into the widget that
actually has focus, inside a real event loop, with the window shown --
the same path a person's keypress takes. A test that pokes the handler
directly can only ever prove the handler is correct, which was never the
part that was broken.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_spotcheck_keys.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                          # noqa: E402
from PyQt5 import QtCore, QtWidgets                          # noqa: E402
from PyQt5.QtTest import QTest                               # noqa: E402

from codelab_pipeline.training import bundle as B            # noqa: E402
from codelab_pipeline.training import verdicts as V          # noqa: E402
from spotcheck.app import SpotCheck                          # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}' + (f'   [{detail}]' if detail else ''))


def make_bundle(d, n_crops=3, n_cands=9):
    rng = np.random.default_rng(1)
    with B.BundleWriter(os.path.join(d, 'fov001__Hyb_001__c000.h5'),
                        meta={'storage_path': os.path.join(d, 'nostore'),
                              'hybe': 'Hyb_001', 'channel': 555,
                              'pad': 14}) as w:
        for c in range(n_crops):
            h, wd, depth = 36, 40, 25
            stack = rng.integers(200, 400, (h, wd, depth)).astype(np.uint16)
            mask = np.zeros((h, wd), np.uint8)
            mask[4:32, 4:36] = 1
            cands = [(6.0 + k, 7.0 + 2 * k, 8.0 + k,
                      0.9 - 0.08 * k, 1 if k < n_cands - 2 else 0,
                      1 if k < 2 else 0, '' if k < 2 else 'occupancy low')
                     for k in range(n_cands)]
            w.add(1, 'Hyb_001', 555, c + 1, stack, mask, 0, 0, cands)


def type_key(app, w, key):
    """A keystroke the way Qt delivers one: into the focus widget."""
    target = QtWidgets.QApplication.focusWidget() or w
    QTest.keyClick(target, key)
    app.processEvents()


def main():
    d = tempfile.mkdtemp()
    make_bundle(d)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    w = SpotCheck(d, 'keytest')
    w.show()
    w.canvas.setFocus()
    app.processEvents()

    focus = QtWidgets.QApplication.focusWidget()
    print(f'\n-- focus is on {type(focus).__name__} --')
    check('the focused widget lets keys through to the window',
          focus is None or focus.__class__.__name__ == 'KeyPassingCanvas'
          or focus is w, type(focus).__name__)

    print('\n-- a real keystroke toggles a card --')
    first = w.queue.current()
    check('there is a page to judge', first is not None)
    type_key(app, w, QtCore.Qt.Key_1)
    check('pressing 1 accepts the first card on the page',
          w._state and 0 in w._state['accepted'],
          str(sorted(w._state['accepted']) if w._state else None))
    type_key(app, w, QtCore.Qt.Key_3)
    check('pressing 3 accepts the third as well',
          w._state and {0, 2} <= set(w._state['accepted']),
          str(sorted(w._state['accepted'])))
    type_key(app, w, QtCore.Qt.Key_1)
    check('pressing 1 again un-accepts it (it is a toggle)',
          w._state and 0 not in w._state['accepted'],
          str(sorted(w._state['accepted'])))

    print('\n-- Space commits what was actually pressed --')
    key0, page0 = first[1]['key'], first[2]
    type_key(app, w, QtCore.Qt.Key_Space)
    recs = V.read_log(w.log.path)
    check('a real Space wrote a record', len(recs) == 1, str(len(recs)))
    kept = [e['i'] for e in recs[0]['shown'] if e['keep']] if recs else []
    check('and it kept exactly the card still toggled on', kept == [2], str(kept))
    check('the app advanced to another page',
          w._state and (w._state['row']['key'], w._state['page']) != (key0, page0))

    print('\n-- S skips without recording --')
    before = len(V.read_log(w.log.path))
    here = (w._state['row']['key'], w._state['page'])
    type_key(app, w, QtCore.Qt.Key_S)
    check('S wrote nothing', len(V.read_log(w.log.path)) == before)
    check('and moved on', (w._state['row']['key'], w._state['page']) != here)

    print('\n-- Backspace shows a judged page WITH its keeps --')
    # The screen must not contradict the file. It used to: a committed
    # page came back with every keep cleared, and one more Space then
    # superseded the real verdict with an empty one.
    while (w._state and (w._state['row']['key'], w._state['page']) != (key0, page0)):
        if w.queue.i == 0:
            break
        type_key(app, w, QtCore.Qt.Key_Backspace)
    on_page = w._state and (w._state['row']['key'], w._state['page']) == (key0, page0)
    check('stepped back to the committed page', bool(on_page))
    if on_page:
        check('its keeps are restored on screen, not blank',
              w._state['accepted'] == {2}, str(sorted(w._state['accepted'])))
        type_key(app, w, QtCore.Qt.Key_Space)
        recs = V.read_log(w.log.path)
        last = [r for r in recs if r['key'] == key0 and r['page'] == page0][-1]
        again = [e['i'] for e in last['shown'] if e['keep']]
        check('re-committing it preserves the verdict instead of blanking it',
              again == [2], str(again))

    print('\n-- the last page ends the queue --')
    # It used to clamp to the final index, so the last page redisplayed
    # itself with keeps cleared and the natural extra Space erased it.
    guard = 0
    while w.queue.current() is not None and guard < 400:
        type_key(app, w, QtCore.Qt.Key_Space)
        guard += 1
    check('the queue really runs out', w.queue.current() is None,
          f'stopped after {guard} commits')
    n_before = len(V.read_log(w.log.path))
    type_key(app, w, QtCore.Qt.Key_Space)
    check('a stray Space at the end writes no duplicate record',
          len(V.read_log(w.log.path)) == n_before,
          f'{len(V.read_log(w.log.path))} vs {n_before}')

    print('\n-- every page committed exactly once, none blanked --')
    recs = V.read_log(w.log.path)
    pages = [(r['key'], r['page']) for r in recs]
    check('no page was committed twice with a later empty line',
          all(sum(1 for p in pages if p == pg) == 1
              or any(e['keep'] for r in recs if (r['key'], r['page']) == pg
                     for e in r['shown'])
              for pg in set(pages)))

    w.close()
    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        for f in FAIL:
            print('  FAILED:', f)
        return 1
    print('ALL GOOD')
    return 0


if __name__ == '__main__':
    sys.exit(main())
