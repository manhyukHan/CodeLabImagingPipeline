"""
The second posterior gate, and the one thing it must refuse to do.

edge_gate asks "can this position be believed"; this one asks "is there a
spot here at all". Both run AFTER everything, on finished answers, and
neither belongs inside a model -- an engine that applied its own p
threshold would be choosing for the reviewer and hiding the very numbers
they choose from.

THE REFUSAL IS THE POINT. LocalizedSpot.p means four different things in
four engines -- a constant 1.0, a contrast, an occupancy-times-CI
product, an affine NCC -- and none of them is a probability. Only
p_exist is. So `available()` is False for v1, v2 and psf-match, and a
spot with no p_exist is NEVER denied: treating "this engine does not
answer that question" as "the answer is no" would silently delete every
spot from every other engine the first time a caller forgot to check.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_p_gate.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np                                          # noqa: E402
from PyQt5 import QtCore, QtWidgets                          # noqa: E402

from codelab_pipeline.localization import engine as E       # noqa: E402
from codelab_pipeline.localization import p_gate as G       # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}'
          + (f'   [{detail}]' if detail else ''))


def spots(ps):
    return [E._spot(i, i, i, 0.5, p_exist=p) for i, p in enumerate(ps)]


def test_it_is_unavailable_where_p_is_not_a_probability():
    print('\navailable(): the gate refuses to look applicable to v1 and v2')
    check('a v3 result offers it', G.available(spots([0.1, 0.9])))
    # Every other engine leaves p_exist NaN, which is not a threshold.
    quality_only = [E._spot(0, 0, 0, 1.0), E._spot(1, 1, 1, 0.42)]
    check('a v1/v2/psf-match result does NOT',
          not G.available(quality_only))
    check('nor does an empty result', not G.available([]))
    check('nor does None', not G.available(None))
    # AND THE UNSCORED ARE NEVER DELETED.
    mixed = spots([0.9, 0.1]) + quality_only
    kept = G.apply(mixed, 0.5)
    check('a spot with no p_exist is never denied',
          len(kept) == 3 and all(np.isnan(G.p_of(s)) or s.p_exist >= 0.5
                                 for s in kept),
          f'{len(kept)} of {len(mixed)} kept')
    s = G.summary(mixed, 0.5)
    check('and the summary counts it as unscored, not as denied',
          s['n_unscored'] == 2 and s['n_denied'] == 1,
          f"unscored {s['n_unscored']}, denied {s['n_denied']}")


def test_the_threshold_does_what_it_says():
    print('\napply / deny: the arithmetic, including its edges')
    got = spots([0.0, 0.2, 0.4999, 0.5, 0.5001, 1.0])
    kept = [s.p_exist for s in G.apply(got, 0.5)]
    check('the threshold is INCLUSIVE -- p == t is kept',
          0.5 in kept, str(kept))
    check('and everything under it goes', all(p >= 0.5 for p in kept),
          str(kept))
    check('order is preserved, not re-sorted',
          kept == sorted(kept), str(kept))
    check('t=0 keeps everything', len(G.apply(got, 0.0)) == len(got))
    check('t=1 keeps only a certain spot',
          [s.p_exist for s in G.apply(got, 1.0)] == [1.0])
    check('annotate returns every spot with its verdict',
          len(G.annotate(got, 0.5)) == len(got)
          and sum(1 for _s, _p, d in G.annotate(got, 0.5) if d) == 3)
    check('a dict works as well as a LocalizedSpot -- these cross a store',
          G.p_of({'p_exist': 0.7}) == 0.7 and np.isnan(G.p_of({})))
    check('and junk in the field reads as absent, never as 0.0',
          np.isnan(G.p_of({'p_exist': 'yes'})))


def test_the_histogram_is_the_evidence():
    print('\nhistogram: fixed to [0, 1], because a probability lives there')
    counts, edges = G.histogram(spots([0.05, 0.06, 0.95]), bins=10)
    check('the axis is the unit interval whatever the data',
          edges[0] == 0.0 and edges[-1] == 1.0, f'{edges[0]}..{edges[-1]}')
    check('every scored spot lands in a bin', counts.sum() == 3, str(counts))
    check('the mass is where the data is',
          counts[0] == 2 and counts[-1] == 1, str(counts))
    c2, e2 = G.histogram([], bins=10)
    check('an empty set draws an empty histogram, not an exception',
          c2.sum() == 0 and len(e2) == 11)
    # AN AUTO-RANGE WOULD LIE. A run with no confident spot must still
    # show its mass at the bottom of a full axis, not filling one.
    c3, e3 = G.histogram(spots([0.01, 0.02, 0.03]), bins=10)
    check('a run with nothing confident still shows a full axis',
          e3[-1] == 1.0 and c3[0] == 3, str(c3))


def test_examples_are_drawn_where_a_threshold_is_decided():
    print('\nexamples: at the line, at random, re-drawable')
    rng = np.random.default_rng(0)
    got = spots([0.01, 0.02, 0.03, 0.45, 0.48, 0.52, 0.55, 0.98, 0.99])
    kept, denied = G.examples(got, 0.5, n=4, rng=rng, band=0.15)
    check('kept examples come from ABOVE the line',
          all(s.p_exist >= 0.5 for s in kept), str([s.p_exist for s in kept]))
    check('denied examples from below',
          all(s.p_exist < 0.5 for s in denied),
          str([s.p_exist for s in denied]))
    check('and the band excludes the easy extremes',
          all(abs(s.p_exist - 0.5) <= 0.15 for s in kept + denied),
          str(sorted(s.p_exist for s in kept + denied)))
    wide_k, wide_d = G.examples(got, 0.5, n=4, rng=rng, band=None)
    check('without a band the whole range is eligible',
          len(wide_k) + len(wide_d) > len(kept) + len(denied),
          f'{len(wide_k)+len(wide_d)} vs {len(kept)+len(denied)}')
    # A DISTRIBUTION WITH A GAP RETURNS AN EMPTY SIDE, and that is a
    # finding the dialog says out loud rather than a blank panel.
    gapped = spots([0.01, 0.02, 0.97, 0.98])
    k2, d2 = G.examples(gapped, 0.5, n=4, rng=rng, band=0.15)
    check('a clean gap at the line yields no borderline examples at all',
          k2 == [] and d2 == [], f'{len(k2)} kept, {len(d2)} denied')
    check('drawing twice can give a different sample',
          len({tuple(sorted(s.p_exist for s in
                            G.examples(got, 0.5, n=2, rng=rng, band=None)[1]))
               for _ in range(12)}) > 1)


def test_the_dialog():
    print('\nthe dialog: a number, a picture, and what it costs')
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    app = (QtWidgets.QApplication.instance()
           or QtWidgets.QApplication(sys.argv[:1]))
    from ui.p_gate_dialog import PGateDialog
    rng = np.random.default_rng(1)
    got = spots(list(rng.uniform(0, 0.2, 40)) + list(rng.uniform(0.8, 1, 12))
                + [0.48, 0.52])
    cube = np.zeros((15, 15, 5)) + 1.0

    d = PGateDialog(got, crop_of=lambda s: cube, threshold=0.5, seed=0)
    d.show()
    for _ in range(3):
        app.processEvents()
    check('it opens on the given threshold', d.threshold() == 0.5)
    check('and says what that keeps',
          '13 of 54' in d.head.text(), d.head.text()[:60])
    d.edit.setText('0.90')
    d.apply_btn.click()
    for _ in range(3):
        app.processEvents()
    check('changing the number changes the count',
          '0.900' in d.head.text() and '13 of 54' not in d.head.text(),
          d.head.text()[:60])
    # A HALF-TYPED NUMBER MUST NOT TAKE THE DIALOG DOWN.
    d.edit.setText('')
    check('an empty box falls back rather than raising',
          d.threshold() == G.DEFAULT_THRESHOLD)
    # '0.' is a VALID float and means 0.0 -- keep everything, which is a
    # sane thing for a half-typed box to mean. '.' and '-' are not.
    d.edit.setText('0.')
    check("'0.' parses as 0.0 rather than falling back",
          d.threshold() == 0.0, str(d.threshold()))
    for junk in ('.', '-', 'nan'):
        d.edit.setText(junk)
        ok = (d.threshold() == G.DEFAULT_THRESHOLD
              if junk != 'nan' else not np.isfinite(d.threshold()) is False)
        check(f'{junk!r} does not raise', isinstance(d.threshold(), float),
              str(d.threshold()))
    d.edit.setText('7')
    check('out of range is clamped, never used raw', d.threshold() == 1.0)
    d.edit.setText('0.5')
    d.apply_btn.click()
    for _ in range(3):
        app.processEvents()
    n_axes = len(d.efig.axes)
    d.refresh_btn.click()
    for _ in range(3):
        app.processEvents()
    check('refresh redraws the examples', len(d.efig.axes) == n_axes,
          str(n_axes))
    check('the band checkbox is on by default', d.band_box.isChecked())
    d.band_box.setChecked(False)
    for _ in range(3):
        app.processEvents()
    check('and unticking it widens the draw',
          'whole range' in d.note.text(), d.note.text()[:60])
    d.close()

    # THE UNAVAILABLE CASE SAYS SO IN WORDS.
    d2 = PGateDialog([E._spot(0, 0, 0, 1.0)], threshold=0.5)
    d2.show()
    for _ in range(2):
        app.processEvents()
    check('a v1/v2 result gets an explanation, not an empty histogram',
          'NO CALIBRATED PROBABILITY' in d2.head.text(),
          d2.head.text()[:60])
    d2.close()

    # And with no pixel source it still works.
    d3 = PGateDialog(got, crop_of=None, threshold=0.5)
    d3.show()
    for _ in range(2):
        app.processEvents()
    check('no crop_of: the distribution still draws, and it says why '
          'there are no pictures',
          'no way to read' in d3.note.text() or 'GAP' in d3.note.text(),
          d3.note.text()[:70])
    d3.close()


def main():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    for t in (test_it_is_unavailable_where_p_is_not_a_probability,
              test_the_threshold_does_what_it_says,
              test_the_histogram_is_the_evidence,
              test_examples_are_drawn_where_a_threshold_is_decided,
              test_the_dialog):
        t()
    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    for f in FAIL:
        print('  FAILED:', f)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
