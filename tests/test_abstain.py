"""
"I cannot call this one" -- and the three ways it used to be lost.

The app's default is DROP: a reviewer presses a number only for a real
spot, and that is what keeps 100k judgements from being 100k clicks. The
cost of it is that a card nobody pressed and a card someone REFUSED to
call are the same record. A reviewer sure about three cards and unsure
about the fourth had two moves and both threw information away -- Space
filed the unsure one as a confident negative, S discarded the three they
were sure about along with it.

MEASURED on the 2,068 labels of MP58/RNA before this existed: 41% of
pages are mixed, so per-card discrimination is what reviewers actually
do, and 25% of all negatives sit inside the p band where kept and
dropped spots overlap ([0.382, 0.614], 227 negatives against 376
positives). `contested` was 0 and could never be anything else -- one
reviewer, and that bucket only fills on DISAGREEMENT -- so there was no
channel for uncertainty at all.

Shift+N is that channel, and keep == -1 is what it writes.

THE TRAP THIS SUITE EXISTS FOR IS THAT -1 IS TRUTHY. Three readers
tested a keep with `if e.get('keep')`, and every one of them would have
counted an abstention as an enthusiastic yes: labels() would have made
it a positive, page_verdict() would have put it back on screen as an
accepted card, and Multispot.load would have done the same on a pillar.
None of that raises.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_abstain.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np                                          # noqa: E402
from PyQt5 import QtCore, QtWidgets                          # noqa: E402
from PyQt5.QtTest import QTest                               # noqa: E402
from matplotlib.figure import Figure                         # noqa: E402

from codelab_pipeline.training import bundle as B            # noqa: E402
from codelab_pipeline.training import verdicts as V          # noqa: E402
from codelab_pipeline.training import view as VIEW           # noqa: E402
from codelab_pipeline.training import multispot_view as MV   # noqa: E402
from codelab_pipeline.training import dataset as D           # noqa: E402
import spotcheck.app as A                                    # noqa: E402
from spotcheck import modes as M                             # noqa: E402

# The fixtures live in one place; this suite asks different questions of
# the same bundle shape.
from test_spotcheck_modes import (make_bundle, make_bank,    # noqa: E402
                                  seed_passfail, gauss)
# A crop with SIX candidates, so a page has four cards to press. The
# multispot fixture plants two emitters and gives two candidates, which
# is the right shape for that question and the wrong one for a keyboard.
from test_spotcheck_guards import make_bundle as make_card_bundle  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}'
          + (f'   [{detail}]' if detail else ''))


def cands_for(d, shard, key):
    _st, _m, cands, words = B.read_crop(shard, key)
    return [(float(c['y']), float(c['x']), float(c['z']), float(c['p']),
             int(c['fit_ok']), int(c['gate_pass']), w)
            for c, w in zip(cands, words)]


def one_row(d):
    shard = B.shard_paths(d)[0]
    return shard, B.read_index(shard)[0]


# -- the record -----------------------------------------------------------

def test_keep_has_three_values():
    print('\nthe record: keep is 1, 0 or -1')
    with tempfile.TemporaryDirectory() as d:
        make_bundle(d)
        shard, row = one_row(d)
        rows = cands_for(d, shard, row['key'])
        log = V.VerdictLog(d, 'ann', session='s1')
        rec = log.commit(row, 0, list(range(len(rows))), rows,
                         accepted={0}, unsure={1})
        got = {e['i']: e['keep'] for e in rec['shown']}
        check('kept is 1', got.get(0) == 1, str(got))
        check('abstained is -1', got.get(1) == -1, str(got))
        check('everything else is still 0',
              all(v == 0 for i, v in got.items() if i not in (0, 1)), str(got))

        # A card cannot be both. `unsure` loses, because pressing the
        # plain number is the deliberate later act in the app and a
        # record that says both is a record no reader can resolve.
        rec2 = log.commit(row, 0, list(range(len(rows))), rows,
                          accepted={0}, unsure={0, 1})
        got2 = {e['i']: e['keep'] for e in rec2['shown']}
        check('a card marked both is a KEEP, never both', got2.get(0) == 1,
              str(got2))


def test_page_verdict_does_not_turn_an_abstention_into_a_keep():
    print('\npage_verdict: -1 is truthy, and that was the bug')
    with tempfile.TemporaryDirectory() as d:
        make_bundle(d)
        shard, row = one_row(d)
        rows = cands_for(d, shard, row['key'])
        log = V.VerdictLog(d, 'ann', session='s1')
        log.commit(row, 0, list(range(len(rows))), rows,
                   accepted={0}, unsure={1})
        pv = log.page_verdict(row['key'], 0)
        check('the keep comes back as accepted', pv['accepted'] == [0],
              str(pv['accepted']))
        check('THE ABSTENTION DOES NOT', 1 not in pv['accepted'],
              str(pv['accepted']))
        check('it comes back as unsure', pv['unsure'] == [1],
              str(pv['unsure']))


# -- the tally ------------------------------------------------------------

def test_an_abstention_is_not_a_vote():
    print('\nlabels(): abstaining is not saying no, and not saying yes')
    with tempfile.TemporaryDirectory() as d:
        make_bundle(d, n_crops=1)
        shard, row = one_row(d)
        rows = cands_for(d, shard, row['key'])
        ix = list(range(len(rows)))
        # ann abstains on card 0 and keeps nothing; bob keeps card 0.
        V.VerdictLog(d, 'ann', session='s1').commit(
            row, 0, ix, rows, accepted=set(), unsure={0})
        V.VerdictLog(d, 'bob', session='s1').commit(
            row, 0, ix, rows, accepted={0}, unsure=set())
        e = V.labels(d)[str(row['key'])]
        xyz = (rows[0][0], rows[0][1], rows[0][2])
        kept, seen = e['votes'][xyz]
        check('the abstention is not counted in `seen`', seen == 1,
              f'{kept} of {seen}')
        check('so one keep and one abstention is a clean POSITIVE',
              any(abs(p[0] - xyz[0]) < 1e-6 for p in e['positive']),
              f"+{len(e['positive'])} -{len(e['negative'])} "
              f"?{len(e['contested'])} u{len(e['undetermined'])}")
        check('NOT contested -- nobody disagreed', not e['contested'])
        check('and NOT a positive by truthiness of -1 alone',
              kept == 1, f'kept={kept}')


def test_a_spot_everyone_skipped_is_undetermined():
    print('\nlabels(): the fourth bucket, and why it is not a negative')
    with tempfile.TemporaryDirectory() as d:
        make_bundle(d, n_crops=1)
        shard, row = one_row(d)
        rows = cands_for(d, shard, row['key'])
        ix = list(range(len(rows)))
        for who in ('ann', 'bob'):
            V.VerdictLog(d, who, session='s1').commit(
                row, 0, ix, rows, accepted={1}, unsure={0})
        e = V.labels(d)[str(row['key'])]
        xyz = (rows[0][0], rows[0][1], rows[0][2])
        check('a spot everyone abstained on is UNDETERMINED',
              any(abs(u[0] - xyz[0]) < 1e-6 for u in e['undetermined']),
              f"u={len(e['undetermined'])}")
        check('it is not a negative -- nobody said no',
              not any(abs(n[0] - xyz[0]) < 1e-6 for n in e['negative']))
        check('it is not contested -- nobody disagreed',
              not any(abs(c[0] - xyz[0]) < 1e-6 for c in e['contested']))
        check('the card they did judge is still a positive',
              len(e['positive']) == 1, str(len(e['positive'])))
        check('and its votes read 0 of 0', e['votes'][xyz] == (0, 0),
              str(e['votes'][xyz]))


def test_training_never_sees_an_abstention():
    print('\ndataset.rows(): -1 is the filter every caller already has')
    with tempfile.TemporaryDirectory() as d:
        make_bundle(d, n_crops=1)
        shard, row = one_row(d)
        rows = cands_for(d, shard, row['key'])
        V.VerdictLog(d, 'ann', session='s1').commit(
            row, 0, list(range(len(rows))), rows, accepted={1}, unsure={0})
        got = D.rows(d)
        by_origin = {}
        for r in got:
            by_origin.setdefault(r['origin'], []).append(r)
        check('an undetermined spot is carried, not dropped',
              'undetermined' in by_origin, str(sorted(by_origin)))
        check('at label -1, which is what training filters on',
              all(r['label'] == -1 for r in by_origin.get('undetermined', [])),
              str([r['label'] for r in by_origin.get('undetermined', [])]))
        check('so it is never a 0 the model learns from',
              not any(r['label'] == 0 and r['origin'] == 'undetermined'
                      for r in got))
        check('and origin tells it apart from contested',
              'contested' not in by_origin
              or by_origin['contested'] is not by_origin['undetermined'])
        without = D.rows(d, include_contested=False)
        check('include_contested=False drops it too',
              not any(r['origin'] == 'undetermined' for r in without),
              f'{len(without)} rows')
        summ = D.summary(got)
        check('summary counts it separately from contested',
              summ['undetermined'] == 1 and summ['contested'] == 0,
              f"u={summ['undetermined']} c={summ['contested']}")


def test_multispot_tally_has_the_same_rule():
    print('\nmultispot_labels(): the same trap, the same fix')
    with tempfile.TemporaryDirectory() as d:
        make_bundle(d, n_crops=1)
        seed_passfail(d)
        bank = make_bank(os.path.join(d, 'psf_bank.h5'))
        seeds = []
        for who, acc, uns in (('ann', {0}, {1}), ('bob', {0}, {1})):
            mode = M.Multispot(bank=bank, bundle_dir=d)
            log = V.VerdictLog(d, who, session='s1', kind=mode.log_kind)
            q = A.Queue(d, log, mode=mode, reviewer=who, shuffle=False)
            s = mode.load(q.items[0], log, 4, 8)
            s['accepted'] = set(acc) & set(s['ix'])
            s['unsure'] = set(uns) & set(s['ix'])
            s['added'] = []
            seeds.append((s, mode.commit(log, s, 1.0)))
        s0, rec = seeds[0]
        vals = {e['i']: e['keep'] for e in rec['shown']}
        check('a multispot record writes -1 too', -1 in vals.values()
              if len(s0['ix']) > 1 else True, str(vals))
        lab = V.multispot_labels(d)
        pil = list(list(lab.values())[0]['pillars'].values())[0]
        check('the kept match is a positive', len(pil['positive']) >= 1,
              f"+{len(pil['positive'])} u{len(pil['undetermined'])}")
        if len(s0['ix']) > 1:
            check('the abstained one is UNDETERMINED, not negative',
                  len(pil['undetermined']) == 1 and not pil['negative'],
                  f"u{len(pil['undetermined'])} -{len(pil['negative'])}")
        # And it must come back on screen as unsure, not as a keep.
        mode2 = M.Multispot(bank=bank, bundle_dir=d)
        log2 = V.VerdictLog(d, 'ann', session='s2', kind=V.MULTISPOT_KIND)
        q2 = A.Queue(d, log2, mode=mode2, reviewer='ann', shuffle=False)
        again = mode2.load(q2.items[0] if q2.items else
                           A.Queue(d, V.VerdictLog(d, 'zz', session='s1',
                                                   kind=V.MULTISPOT_KIND),
                                   mode=M.Multispot(bank=bank, bundle_dir=d),
                                   reviewer='zz', shuffle=False).items[0],
                           log2, 4, 8)
        check('a revisited pillar does not resurrect it as a keep',
              not (again['prior_accepted'] & again['prior_unsure']),
              f"acc={again['prior_accepted']} uns={again['prior_unsure']}")


# -- the key --------------------------------------------------------------

def app_on(d, mode, reviewer='ab'):
    app = (QtWidgets.QApplication.instance()
           or QtWidgets.QApplication(sys.argv[:1]))
    w = A.SpotCheck(d, reviewer, max_per_crop=A.DEFAULT_MAX_PER_CROP,
                    mode=mode)
    w.show(); w.canvas.setFocus()
    for _ in range(3):
        app.processEvents()
    return app, w


def press(app, w, key, mod=QtCore.Qt.NoModifier):
    QTest.keyClick(QtWidgets.QApplication.focusWidget() or w, key, mod)
    app.processEvents()


def test_every_way_shift_one_can_arrive():
    print('\nthe key: Shift+1 is not Key_1 on the keyboards this runs on')
    with tempfile.TemporaryDirectory() as d:
        make_card_bundle(d, n_cands=6)
        app, w = app_on(d, M.PassFail())
        # THE REAL-HARDWARE PATH FIRST. A US or Korean layout sends
        # Key_Exclam with no modifier flag on the event Qt delivers; a
        # check for "Key_1 and ShiftModifier" would do nothing at all
        # here, silently, on the machines reviewers actually use.
        press(app, w, QtCore.Qt.Key_Exclam)
        check('bare Key_Exclam abstains on card 1',
              w._state['unsure'] == {w._state['ix'][0]},
              str(w._state['unsure']))
        check('and does NOT keep it', not w._state['accepted'],
              str(w._state['accepted']))
        press(app, w, QtCore.Qt.Key_Exclam)
        check('pressing it again clears the abstention',
              not w._state['unsure'], str(w._state['unsure']))

        press(app, w, QtCore.Qt.Key_1, QtCore.Qt.ShiftModifier)
        check('Shift+Key_1 abstains too',
              w._state['unsure'] == {w._state['ix'][0]},
              str(w._state['unsure']))
        press(app, w, QtCore.Qt.Key_1, QtCore.Qt.ShiftModifier)
        press(app, w, QtCore.Qt.Key_NumberSign, QtCore.Qt.ShiftModifier)
        check('Shift+3 lands on card 3, not card 1',
              w._state['unsure'] == {w._state['ix'][2]},
              str(w._state['unsure']))

        # MUTUALLY EXCLUSIVE, both directions.
        press(app, w, QtCore.Qt.Key_3)
        check('the plain number turns an abstention into a keep',
              w._state['accepted'] == {w._state['ix'][2]}
              and not w._state['unsure'],
              f"acc={w._state['accepted']} uns={w._state['unsure']}")
        press(app, w, QtCore.Qt.Key_NumberSign)
        check('and Shift turns a keep into an abstention',
              w._state['unsure'] == {w._state['ix'][2]}
              and not w._state['accepted'],
              f"acc={w._state['accepted']} uns={w._state['unsure']}")

        check('the status line names which cards carry it',
              '#3' in w.status.text() and 'UNSURE' in w.status.text(),
              w.status.text()[:110])

        # Nothing else may have become a digit on the way past.
        press(app, w, QtCore.Qt.Key_A)
        check('A is still add mode, not a card', w._adding)
        press(app, w, QtCore.Qt.Key_Escape)
        w.close()


def test_an_abstention_survives_backspace_and_reaches_the_file():
    print('\nthe page: uncommitted, then committed, then revisited')
    with tempfile.TemporaryDirectory() as d:
        make_card_bundle(d, n_crops=3, n_cands=6)
        app, w = app_on(d, M.PassFail())
        press(app, w, QtCore.Qt.Key_1)
        press(app, w, QtCore.Qt.Key_At)              # Shift+2
        want_acc = set(w._state['accepted'])
        want_uns = set(w._state['unsure'])
        check('a page can hold both at once',
              want_acc and want_uns and not (want_acc & want_uns),
              f'acc={want_acc} uns={want_uns}')
        press(app, w, QtCore.Qt.Key_S)               # leave without saving
        press(app, w, QtCore.Qt.Key_Backspace)
        check('the abstention survives leaving the page uncommitted',
              w._state['unsure'] == want_uns, str(w._state['unsure']))
        check('and so does the keep', w._state['accepted'] == want_acc)

        press(app, w, QtCore.Qt.Key_Space)
        recs = V.read_log(w.log.path)
        vals = {e['i']: e['keep'] for e in recs[-1]['shown']}
        check('the file records -1 for it',
              sorted(v for v in vals.values() if v == -1) == [-1] * len(want_uns),
              str(vals))
        press(app, w, QtCore.Qt.Key_Backspace)
        check('and a revisited page shows it as unsure, not as a keep',
              w._state['unsure'] == want_uns
              and w._state['accepted'] == want_acc,
              f"acc={w._state['accepted']} uns={w._state['unsure']}")
        check('the page knows it is a revisit', w._state['revisited'])
        w.close()


# -- the picture ----------------------------------------------------------

def test_the_third_state_is_visible():
    print('\nthe picture: grey, dashed, and the word')
    with tempfile.TemporaryDirectory() as d:
        make_card_bundle(d, n_cands=6)
        shard, row = one_row(d)
        st, mask, cands, words = B.read_crop(shard, row['key'])
        rows = cands_for(d, shard, row['key'])
        fig = Figure(figsize=(15, 5.6), dpi=100)
        ix = VIEW.pages_of(len(rows), 4)[0]
        art = VIEW.draw_page(fig, st, mask, rows, ix, accepted={ix[0]},
                             unsure={ix[1]}, page=0, npage=1, per_page=4,
                             n_total=len(rows))
        imgs = {id(im): im.get_array().tobytes()
                for ax in fig.axes for im in ax.images}
        c_keep = art['cards'][ix[0]][0]
        c_uns = art['cards'][ix[1]][0]
        check('an abstained card is not the kept colour',
              c_uns.get_edgecolor() != c_keep.get_edgecolor())
        check('and it is dashed, so it reads on a grayscale page',
              c_uns.get_linestyle() != c_keep.get_linestyle(),
              str(c_uns.get_linestyle()))
        check('its title says UNSURE in words',
              '? UNSURE' in art['cards'][ix[1]][1].get_text(),
              art['cards'][ix[1]][1].get_text())
        check('the kept one still says KEEP',
              '✓ KEEP' in art['cards'][ix[0]][1].get_text())

        # restyle must move it without touching a pixel.
        VIEW.restyle(art, {ix[1]}, unsure={ix[0]})
        check('restyle can swap the two states',
              '✓ KEEP' in art['cards'][ix[1]][1].get_text()
              and '? UNSURE' in art['cards'][ix[0]][1].get_text())
        check('and redraws no image data',
              all(im.get_array().tobytes() == imgs[id(im)]
                  for ax in fig.axes for im in ax.images if id(im) in imgs))
        VIEW.restyle(art, set())
        check('a card with neither state loses both words',
              'KEEP' not in art['cards'][ix[0]][1].get_text()
              and 'UNSURE' not in art['cards'][ix[0]][1].get_text(),
              art['cards'][ix[0]][1].get_text())


def test_the_pillar_page_shows_it_too():
    print('\nthe pillar page: same three states')
    fig = Figure(figsize=(15, 5.6), dpi=100)
    p = 3 * gauss((15, 15, 90), centre=(7, 7, 45))
    hits = [(7.0, 7.0, 45.0, 0.9), (7.0, 11.0, 45.0, 0.7),
            (3.0, 4.0, 60.0, 0.6)]
    art = MV.draw_pillar(fig, p, hits, accepted={0}, unsure={1})
    check('the abstained match is a different colour from both',
          len({art['cards'][i]['mark'].get_edgecolor()
               for i in (0, 1, 2)}) == 3)
    check('its card says UNSURE',
          '? UNSURE' in art['cards'][1]['title'].get_text(),
          art['cards'][1]['title'].get_text().replace('\n', ' | '))
    check('and it keeps its own y line',
          'own y' in art['cards'][1]['title'].get_text())
    MV.restyle(art, {1}, unsure={2})
    check('restyle moves the word off the old card',
          '? UNSURE' not in art['cards'][1]['title'].get_text()
          and '? UNSURE' in art['cards'][2]['title'].get_text(),
          art['cards'][2]['title'].get_text().replace('\n', ' | '))
    check('and does not duplicate it on repeat calls',
          MV.restyle(art, {1}, unsure={2}) is not None
          and art['cards'][2]['title'].get_text().count('UNSURE') == 1,
          art['cards'][2]['title'].get_text().replace('\n', ' | '))
    check('the blit set still covers what changed',
          art['cards'][2]['mark'] in MV.mutable_artists(art))


def main():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    for t in (test_keep_has_three_values,
              test_page_verdict_does_not_turn_an_abstention_into_a_keep,
              test_an_abstention_is_not_a_vote,
              test_a_spot_everyone_skipped_is_undetermined,
              test_training_never_sees_an_abstention,
              test_multispot_tally_has_the_same_rule,
              test_every_way_shift_one_can_arrive,
              test_an_abstention_survives_backspace_and_reaches_the_file,
              test_the_third_state_is_visible,
              test_the_pillar_page_shows_it_too):
        t()
    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    for f in FAIL:
        print('  FAILED:', f)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
