"""
A bundle is a COUNT OF CROPS, equal over hybes and FOVs -- not every
cell of every FOV in every hybe.

WHY. A build took every cell of every listed FOV for every checked
hybe, and a review is a person's week: 6 FOVs x 121 cells x 4 sources
is 2,904 crops before a single verdict, most of them the 100th cell of
the same field in the same probe. What a detector needs is variety --
every hybe and channel, cells spread over FOVs, as many different cells
as the budget allows -- so the build takes a number of CROPS (a crop is
one cell in one hybe and channel; 10,000 a bundle by default), splits
it equally over the hybes, each hybe's share equally over the FOVs, and
lets each (hybe, FOV) draw its own cells: hybes see different cells,
and a cell is in about as many hybes as any other.

THE DRAW HAS TO BE SHARED AND EXTENDABLE. build_bundle runs once per
channel; an interrupted build must resume on the same plan; a hybe or
channel added later must not disturb what is on disk; and asking for
more later should extend, not reshuffle. Hence a seeded permutation per
(hybe, channel, fov), a function of those names and the seed alone,
with chunks cut from its prefix.

The draw is checked on a fake store; the CLI's dry run is checked on
the real MP58/RNA store (read only) when it is reachable.

Run:  python tests/test_bundle_crop_draw.py
"""
import inspect
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKS = [0, 0]
STORE = 'G:/Seonghyeok/2025-11-30-MP58/RNA'
DRY_OUT = 'D:/claude-tmp/cropdraw_dry_never_written'

# fov -> ids as the store lists them (not sorted, on purpose)
FAKE = {1: list(range(100, 120)),                       # 20
        2: list(range(50)),                             # 50
        3: [7, 3, 9, 1, 5, 11, 2, 8, 4, 6],             # 10
        4: []}                                          # segmented by nobody
HY = ['H1', 'H2']


def check(name, ok, note=''):
    CHECKS[0] += 1
    CHECKS[1] += 1 if ok else 0
    print(('  ok   ' if ok else '  FAIL ') + name
          + (('  -- ' + note) if note else ''))


def fake_masks(storage_path, fov, hybe=None, modality=None, resolver=None):
    return [(i, None, None, None) for i in FAKE.get(int(fov), [])]


def test_allocate():
    print('allocate: equal shares, water-filled, seeded remainder')
    from codelab_pipeline.training import extract as X
    check('the default is 10,000 crops a bundle', X.DEFAULT_N_CROPS == 10000)
    check('four equal FOVs split evenly',
          X.allocate({1: 50, 2: 50, 3: 50, 4: 50}, 100)
          == {1: 25, 2: 25, 3: 25, 4: 25})
    t = X.allocate({1: 5, 2: 100, 3: 100}, 60)
    check('a small FOV gives all it has and the rest is shared on',
          t[1] == 5 and sum(t.values()) == 60 and {t[2], t[3]} == {27, 28},
          str(t))
    t = X.allocate({1: 100, 2: 100, 3: 100}, 10)
    check('the remainder goes one each, and the split is deterministic',
          sum(t.values()) == 10 and sorted(t.values()) == [3, 3, 4]
          and t == X.allocate({1: 100, 2: 100, 3: 100}, 10), str(t))
    check('hybes are keys too',
          X.allocate({'H1': 80, 'H2': 80}, 30) == {'H1': 15, 'H2': 15}
          and sorted(X.allocate({'H1': 80, 'H2': 80, 'H3': 80}, 10).values())
          == [3, 3, 4])
    check('asking for more than exist gives what exists',
          X.allocate({1: 10, 2: 10}, 500) == {1: 10, 2: 10})
    check('an empty FOV takes nothing, and never blocks the others',
          X.allocate({1: 10, 2: 0}, 5) == {1: 5, 2: 0})
    check('nothing at all', X.allocate({}, 5) == {} and
          X.allocate({1: 10}, 0) == {1: 0})


def test_draw_and_plan():
    print('the draw, and the plan cut from it')
    from codelab_pipeline.training import extract as X
    real = X.cell_masks
    X.cell_masks = fake_masks
    F = [1, 2, 3, 4]
    try:
        d, avail = X.draw_crops('/nowhere', F, HY, 635, n_crops=None)
        check('None is every cell of every hybe, in store order',
              d['H1'][3] == FAKE[3] and d['H2'][1] == FAKE[1]
              and d['H1'][4] == [] and avail == {1: 20, 2: 50, 3: 10, 4: 0},
              str(avail))
        d60, _ = X.draw_crops('/nowhere', F, HY, 635, n_crops=60)
        per_h = {h: sum(len(d60[h][f]) for f in F) for h in HY}
        check('60 crops over 2 hybes is 30 each', per_h == {'H1': 30, 'H2': 30},
              str(per_h))
        check('and each hybe spreads its 30 over (20, 50, 10, 0) as 10 each',
              all([len(d60[h][f]) for f in F] == [10, 10, 10, 0] for h in HY))
        check('each (hybe, FOV) draws its own cells, no repeats',
              all(len(set(d60[h][f])) == len(d60[h][f])
                  and set(d60[h][f]) <= set(FAKE[f])
                  for h in HY for f in (1, 2, 3)))
        check('HYBES IN THE SAME FOV DRAW DIFFERENT CELLS',
              d60['H1'][2] != d60['H2'][2], str((d60['H1'][2], d60['H2'][2])))
        c555, _ = X.draw_crops('/nowhere', F, HY, 555, n_crops=60)
        check('and a channel draws its own too', c555['H1'][2] != d60['H1'][2])
        again, _ = X.draw_crops('/nowhere', F, HY, 635, n_crops=60)
        check('the same seed and names draw the same cells', again == d60)
        other, _ = X.draw_crops('/nowhere', F, HY, 635, n_crops=60, seed=1)
        check('another seed draws other cells', other['H1'][2] != d60['H1'][2])
        alone, _ = X.draw_crops('/nowhere', [2], ['H1'], 635, n_crops=10)
        check("a (hybe, FOV) draw does not depend on the other hybes or "
              "FOVs in the run", alone['H1'][2] == d60['H1'][2])
        d90, _ = X.draw_crops('/nowhere', F, HY, 635, n_crops=90)
        check('90 is 45 a hybe: the small FOV fills and the rest share '
              '(17, 18)',
              all(len(d90[h][3]) == 10
                  and sorted([len(d90[h][1]), len(d90[h][2])]) == [17, 18]
                  for h in HY))
        check('MORE CROPS EXTEND THE PREFIX of every (hybe, FOV)',
              all(d90[h][f][:len(d60[h][f])] == d60[h][f]
                  for h in HY for f in (1, 2, 3)))
        big, _ = X.draw_crops('/nowhere', [1, 2, 3], HY, 635, n_crops=10 ** 6)
        check('more than exist: every cell of every hybe, permuted order',
              sorted(big['H1'][2]) == sorted(FAKE[2])
              and big['H1'][2] != FAKE[2] and big['H1'][2][:10] == d60['H1'][2])
        # A CELL IS IN ABOUT AS MANY HYBES AS ANY OTHER: 30 hybes each
        # drawing 10 of FOV 2's 50 cells is 300 uses, 6 a cell on average.
        many = [f'H{i:02d}' for i in range(30)]
        m, _ = X.draw_crops('/nowhere', [2], many, 635, n_crops=300)
        uses = {c: 0 for c in FAKE[2]}
        for h in many:
            for c in m[h][2]:
                uses[c] += 1
        lo, hi = min(uses.values()), max(uses.values())
        check('every cell is used, none hogs the hybes',
              sum(uses.values()) == 300 and lo >= 1 and hi <= 13,
              f'uses per cell {lo}..{hi}, mean 6')

        p635 = X.plan('/nowhere', F, HY, 635, n_crops=60)
        tags = [(f, h, t) for (f, h, _ids, t) in p635]
        check('chunks of 8 off each prefix, tagged by position, FOV-major',
              tags[:4] == [(1, 'H1', 'c000'), (1, 'H1', 'c001'),
                           (1, 'H2', 'c000'), (1, 'H2', 'c001')]
              and len(p635) == 3 * 2 * 2, str(tags[:6]))
        by = {(f, h, t): ids for (f, h, ids, t) in p635}
        check('the two hybes of a FOV hold different cells in their chunks',
              by[(2, 'H1', 'c000')] != by[(2, 'H2', 'c000')])
        p90 = X.plan('/nowhere', F, HY, 635, n_crops=90)
        by90 = {(f, h, t): ids for (f, h, ids, t) in p90}
        check('a bigger draw keeps every full chunk as it was',
              by90[(1, 'H1', 'c000')] == by[(1, 'H1', 'c000')]
              and by90[(2, 'H2', 'c000')] == by[(2, 'H2', 'c000')]
              and by90[(1, 'H1', 'c001')][:2] == by[(1, 'H1', 'c001')])
        given = X.plan('/nowhere', [1, 2], ['H1'], 635,
                       crops={'H1': {1: [5, 6, 7], 2: []}})
        check('a caller that drew already is taken at its word',
              given == [(1, 'H1', [5, 6, 7], 'c000')], str(given))
        old = X.plan('/nowhere', [3], ['H1'], 635)
        check('no count: store order, chunked by position -- the earlier '
              'behaviour, exactly',
              [ids for (_f, _h, ids, _t) in old]
              == [FAKE[3][:8], FAKE[3][8:]], str(old))
        sig = inspect.signature(X.extract).parameters
        check('extract() names the draw so it never leaks into the chunk '
              'kwargs', all(k in sig for k in ('n_crops', 'seed', 'crops')))
    finally:
        X.cell_masks = real


def test_cli():
    print('build_bundle: --n-crops, the record, the warning')
    import tools.build_bundle as BB
    src = inspect.getsource(BB.main)
    check("--n-crops defaults to the extractor's default",
          "'--n-crops'" in src and 'default=X.DEFAULT_N_CROPS' in src)
    check('--draw-seed, default 0', "'--draw-seed'" in src
          and 'default=0' in src.split("'--draw-seed'")[1][:80])
    check('the masks are read ONCE and handed to the draw',
          'ids_by_fov = cells_in(store, pool)' in src
          and 'ids_by_fov=ids_by_fov' in src)
    check('the drawn crops are what extract() cuts', 'crops=drawn' in src)
    check('the manifest records the draw',
          all(k in src for k in ('n_crops=n_crops', 'draw_seed=draw_seed',
                                 'crops_planned=crops', 'crops_per_hybe',
                                 'crops_per_fov')))
    check("a different draw for a channel already on disk is SAID",
          'was built into this bundle' in src and '--rebuild' in src)
    if not os.path.isdir(STORE):
        check('the real store is reachable', False,
              'skipped: ' + STORE + ' absent -- dry run unvalidated')
        return
    out = subprocess.run(
        [sys.executable, '-u', 'tools/build_bundle.py', STORE,
         '--out', DRY_OUT, '--channel', '635', '--hybes', 'Hyb_101,Hyb_103',
         '--n-fovs', '2', '--seed', '20260910', '--n-crops', '100',
         '--dry-run'],
        capture_output=True, text=True, timeout=600,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    text = out.stdout
    # the draw's own line, not the 'cells per FOV:' line above it
    m = re.search(r'^\s+per FOV: (.*)$', text, re.M)
    fov_counts = [int(b) for _f, b in re.findall(r'(\d+):(\d+)', m.group(1))] if m else []
    hybe_counts = [int(n) for n in re.findall(r'Hyb_10[13]\s+\S+\s+(\d+) crops', text)]
    check('dry run on the real store: 100 crops, 2 hybes x 2 FOVs, is '
          '50 a hybe and 50 a FOV',
          out.returncode == 0 and 'crops   100 of' in text
          and fov_counts == [50, 50] and hybe_counts == [50, 50],
          (m.group(0) + ' | hybes ' + str(hybe_counts)) if m else text[-500:])
    check('and the work line counts the DRAWN crops',
          'work    2 FOVs x 2 hybes = 100 crops' in text)
    check('a dry run wrote nothing', not os.path.exists(DRY_OUT))


def main():
    test_allocate()
    test_draw_and_plan()
    test_cli()
    print()
    print('%d/%d checks passed' % (CHECKS[1], CHECKS[0]))
    return 0 if CHECKS[1] == CHECKS[0] else 1


if __name__ == '__main__':
    sys.exit(main())
