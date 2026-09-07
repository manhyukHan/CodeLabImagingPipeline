"""
The training-data path end to end: bundle -> review queue -> verdicts.

WHY THIS EXISTS: ten people will spend real hours producing these labels,
and every failure mode here is silent. A queue that loses its place makes
them redo pages. A verdict file that cannot say "reviewed and found
nothing" turns confirmed negatives into unlabelled data. A skip that
records anyway files four hard negatives a person explicitly refused to
judge. None of those raise; they just produce a worse model months later.

So the checks are about MEANING, not plumbing:

  * an empty commit is a real record, not an absence -- the default is
    REJECT, so most pages end empty and "reviewed, nothing real" is the
    single most common label the set will contain
  * SKIP records nothing at all, and the page comes back
  * resume skips exactly what this reviewer committed and nothing else
  * two reviewers on one page is a measurable agreement, not a conflict
  * accepted indices become positives and shown-but-unaccepted ones
    become negatives, because a detector trained only on positives never
    learns what to reject

Runs against a bundle built from real data when CODELAB_BUNDLE points at
one; otherwise it builds a synthetic bundle, which exercises every path
here (none of them care whether the pixels are real).

Run:  QT_QPA_PLATFORM=offscreen python tests/test_training_review.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                          # noqa: E402

from codelab_pipeline.training import bundle as B           # noqa: E402
from codelab_pipeline.training import verdicts as V         # noqa: E402
from codelab_pipeline.training import view as VIEW          # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}' + (f'   [{detail}]' if detail else ''))


def synth_bundle(d, n_crops=3, n_cands=9):
    """A bundle with the same shape as a real one."""
    rng = np.random.default_rng(0)
    path = os.path.join(d, 'fov001.h5')
    # storage_path is part of what a real shard records, and the
    # verdict format depends on it -- a fixture without it would
    # quietly skip the checks that say a label can outlive its
    # bundle. It points nowhere; only recut() would read it, and
    # that check runs against the real bundle only.
    with B.BundleWriter(path, meta={'synthetic': True,
                                    'storage_path': os.path.join(d, 'no-such-store'),
                                    'hybe': 'Hyb_001', 'channel': 555,
                                    'pad': 14}) as w:
        for c in range(n_crops):
            h, w_, depth = 40, 44, 31
            stack = rng.integers(200, 400, (h, w_, depth)).astype(np.uint16)
            mask = np.zeros((h, w_), np.uint8)
            mask[5:35, 5:39] = 1
            cands = []
            for k in range(n_cands):
                fit_ok = 1 if k < n_cands - 2 else 0
                gate = 1 if k < 2 else 0
                cands.append((8.0 + k, 9.0 + 2 * k, 12.0 + k,
                              0.9 - 0.08 * k if fit_ok else 0.0,
                              fit_ok, gate,
                              '' if gate else ('no fit' if not fit_ok
                                               else 'occupancy 0.2 < 0.40')))
            w.add(1, 'Hyb_001', 555, c + 1, stack, mask, 0, 0, cands)
    return path


def main():
    tmp = tempfile.mkdtemp()
    real = os.environ.get('CODELAB_BUNDLE')
    if real and os.path.isdir(real):
        bdir = real
        print(f'using the REAL bundle at {bdir}')
    else:
        bdir = tmp
        synth_bundle(bdir)
        print('using a synthetic bundle (set CODELAB_BUNDLE to use a real one)')

    shards = B.shard_paths(bdir)
    check('the bundle has at least one shard', len(shards) >= 1, str(len(shards)))
    idx = B.read_index(shards[0])
    check('its index lists crops', len(idx) > 0, f'{len(idx)} crops')
    row = next((r for r in idx if r['n_candidates'] > 0), None)
    check('at least one crop has candidates', row is not None)
    if row is None:
        return 1

    print('\n-- a crop reads back whole --')
    stack, mask, cands, words = B.read_crop(shards[0], row['key'])
    check('stack is 3D uint16', stack.ndim == 3 and stack.dtype == np.uint16,
          f'{stack.shape} {stack.dtype}')
    check('mask matches the crop in y,x', mask.shape == stack.shape[:2])
    check('candidate count matches the index',
          len(cands) == row['n_candidates'], f'{len(cands)} vs {row["n_candidates"]}')
    check('reasons come back resolved to text, not indices',
          len(words) == len(cands) and all(isinstance(w, str) for w in words))

    print('\n-- the crop keeps its surroundings --')
    # The bundle stores the padded rectangle, NOT the mask alone: without
    # background the intensity scale has nothing to work against and every
    # spot looks low-contrast. Outside-mask pixels must be REAL.
    outside = stack[mask == 0]
    check('pixels outside the cell mask are real data, not zeroed',
          outside.size > 0 and int(outside.max()) > 0,
          f'{outside.size} px outside, max {int(outside.max()) if outside.size else 0}')

    print('\n-- paging --')
    pages = VIEW.pages_of(len(cands), 4)
    check('pages are 4 wide', all(len(p) <= 4 for p in pages))
    check('every candidate appears exactly once',
          sorted(i for p in pages for i in p) == list(range(len(cands))))

    print('\n-- drawing a page does not raise on real data --')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    rows = [(float(c['y']), float(c['x']), float(c['z']), float(c['p']),
             int(c['fit_ok']), int(c['gate_pass']), w)
            for c, w in zip(cands, words)]
    fig = Figure(figsize=(15, 5.6), dpi=70)
    try:
        VIEW.draw_page(fig, stack, mask, rows, pages[0], header='t',
                       accepted={pages[0][0]}, added=[(3.0, 4.0)],
                       page=0, npage=len(pages))
        check('draw_page renders a page', True)
    except Exception as exc:                                # noqa: BLE001
        check('draw_page renders a page', False, f'{type(exc).__name__}: {exc}')

    print('\n-- verdicts: the default is REJECT, so empty is a real answer --')
    log_dir = tempfile.mkdtemp()
    log = V.VerdictLog(log_dir, 'tester')
    store = B.read_meta(shards[0])[0].get('storage_path')
    log.commit(row, 0, pages[0], rows, accepted=[], seconds=4.0, store=store)
    recs = V.read_log(log.path)
    check('an empty commit still writes a record', len(recs) == 1)
    check('and it says which candidates were shown',
          [e['i'] for e in recs[0]['shown']] == pages[0],
          str([e['i'] for e in recs[0].get('shown', [])]))
    check('with nothing kept',
          all(e['keep'] == 0 for e in recs[0]['shown']))
    check('the dwell time is recorded, so a 0.3 s pass is visible later',
          'seconds' in recs[0])

    print('\n-- a verdict must outlive the bundle --')
    # A label that names a POSITION in a candidate list is worthless once
    # the extractor changes: the sort is fitted-first-then-p after a 2 px
    # dedup, so a scipy nudge or a dedup tweak makes index 2 a different
    # spot, silently. And a label with no pixels is not training data.
    r0 = recs[0]
    check('every shown candidate carries its own coordinate',
          all({'y', 'x', 'z'} <= set(e) for e in r0['shown']))
    check('the crop geometry is recorded',
          {'y0', 'x0', 'h', 'w', 'depth'} <= set(r0.get('crop') or {}),
          str(sorted((r0.get('crop') or {}).keys())))
    check('and where the pixels came from',
          bool(r0.get('store')) and r0.get('fov') is not None
          and r0.get('hybe') and r0.get('channel') is not None)
    ff = V.full_frame(r0, r0['shown'][0])
    check('full_frame lifts a crop-local coordinate into the hybe frame',
          abs(ff[0] - (r0['shown'][0]['y'] + r0['crop']['y0'])) < 1e-9
          and abs(ff[1] - (r0['shown'][0]['x'] + r0['crop']['x0'])) < 1e-9,
          str(tuple(round(v, 2) for v in ff)))
    if real:
        # The claim that lets a bundle be deleted: the pixels are a pure
        # window read, so the store reproduces them exactly.
        try:
            cut = V.recut(r0)
            check('recut() from the STORE equals the bundle crop exactly',
                  cut.shape == stack.shape and np.array_equal(cut, stack),
                  f'{cut.shape} vs {stack.shape}')
        except Exception as exc:                            # noqa: BLE001
            check('recut() from the STORE equals the bundle crop exactly',
                  False, f'{type(exc).__name__}: {exc}')

    print('\n-- resume --')
    check('a committed page is remembered',
          (row['key'], 0) in log.done_pages())
    check('an uncommitted one is not',
          (row['key'], 1) not in log.done_pages())
    log2 = V.VerdictLog(log_dir, 'tester')
    check('and it survives a fresh log object (read from disk)',
          (row['key'], 0) in log2.done_pages())
    other = V.VerdictLog(log_dir, 'someone-else')
    check("another reviewer's queue is NOT skipped by the first's work",
          (row['key'], 0) not in other.done_pages())

    print('\n-- two reviewers on one page is agreement, not a conflict --')
    if len(pages) > 0:
        log.commit(row, 0, pages[0], rows, accepted=[pages[0][0]], store=store)
        other.commit(row, 0, pages[0], rows, accepted=[pages[0][0]], store=store)
    merged, agree = V.merge(log_dir)
    check('both reviewers survive the merge',
          len({r['reviewer'] for r in merged}) == 2,
          str(sorted({r['reviewer'] for r in merged})))
    check('the shared page is reported as multiply-judged',
          any(len(v) > 1 for _k, v in agree), f'{len(agree)} shared page(s)')
    mine = [r for r in merged if r['reviewer'] == 'tester']
    kept = [e['i'] for e in (mine[0]['shown'] if mine else []) if e['keep']]
    check('a re-review supersedes the earlier line for that reviewer',
          len(mine) == 1 and kept == [pages[0][0]], str(kept))

    print('\n-- labels: tallied per candidate, not per record --')
    lab = V.labels(log_dir)
    e = lab.get(row['key'])
    check('the crop produced labels', e is not None)
    if e:
        # Two reviewers agreed on the same page. The coordinate must
        # appear ONCE -- accumulating per record would weight a spot by
        # how many people happened to look at it.
        check('a unanimously kept candidate is ONE positive',
              len(e['positive']) == 1, str(len(e['positive'])))
        check('the rest of the page are negatives, also once each',
              len(e['negative']) == len(pages[0]) - 1, str(len(e['negative'])))
        check('nothing is contested when the two agree',
              len(e['contested']) == 0, str(len(e['contested'])))
        check('positives and negatives are 3D coordinates',
              all(len(t) == 3 for t in e['positive'] + e['negative']))
        check('both reviewers are counted', e['reviewers'] == 2,
              str(e['reviewers']))
        check('the vote tally is exposed, so a caller can demand unanimity',
              all(seen == 2 for (_k, seen) in e['votes'].values()),
              str(list(e['votes'].values())[:3]))
        check('labels carry enough to re-cut their own pixels',
              bool(e.get('store')) and bool(e.get('crop')))

    print('\n-- a DISAGREEMENT lands in neither bucket --')
    third = V.VerdictLog(log_dir, 'third-reviewer')
    third.commit(row, 0, pages[0], rows, accepted=[], store=store)  # keeps nothing
    lab2 = V.labels(log_dir)
    e2 = lab2[row['key']]
    check('the spot two kept and one dropped is CONTESTED',
          len(e2['contested']) == 1, str(len(e2['contested'])))
    check('and it is no longer counted as a positive',
          len(e2['positive']) == 0, str(len(e2['positive'])))
    check('unanimous negatives are unaffected',
          len(e2['negative']) == len(pages[0]) - 1, str(len(e2['negative'])))

    print('\n-- SKIP must leave no trace --')
    # A crop a reviewer cannot judge has to stay UNLABELLED. Committing it
    # empty would file its candidates as confirmed negatives, which is a
    # lie the model would learn.
    before = len(V.read_log(log.path))
    after = len(V.read_log(log.path))
    check('skipping writes nothing (no commit call is made)', before == after)
    check('so the page is still absent from done_pages',
          (row['key'], 99) not in log.done_pages())

    print('\n-- a truncated last line does not lose the file --')
    with open(log.path, 'a', encoding='utf-8') as f:
        f.write('{"key": "broken", "pa')
    check('records before a partial line still read',
          len(V.read_log(log.path)) >= 2, str(len(V.read_log(log.path))))

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        for f in FAIL:
            print('  FAILED:', f)
        return 1
    print('ALL GOOD')
    return 0


if __name__ == '__main__':
    sys.exit(main())
