"""
A bundle cuts a COUNT of cells spread over its FOVs -- not every cell.

WHY. A build took every cell of every listed FOV, and a review is a
person's week: 6 FOVs x 121 cells x 4 sources is 2,904 crops before a
single verdict, most of them the 100th cell of the same field. What a
detector needs is variety -- cells across FOVs, every hybe and channel
of each -- so the build takes a number of CELLS (default 10,000),
spreads it evenly over the FOVs, and cuts every checked source for
each drawn cell: crops = cells x sources, four sources, four crops a
cell.

THE DRAW HAS TO BE SHARED AND EXTENDABLE. build_bundle runs once per
channel, so two channels must draw the SAME cells; an interrupted build
must resume on the same plan; and asking for more later should extend,
not reshuffle. Hence a seeded permutation per FOV, a function of (seed,
fov) alone, with chunks cut from its prefix.

The draw is checked on a fake store; the CLI's dry run is checked on
the real MP58/RNA store (read only) when it is reachable.

Run:  python tests/test_bundle_cell_draw.py
"""
import inspect
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKS = [0, 0]
STORE = 'G:/Seonghyeok/2025-11-30-MP58/RNA'
DRY_OUT = 'D:/claude-tmp/celldraw_dry_never_written'

# fov -> ids as the store lists them (not sorted, on purpose)
FAKE = {1: list(range(100, 120)),                       # 20
        2: list(range(50)),                             # 50
        3: [7, 3, 9, 1, 5, 11, 2, 8, 4, 6],             # 10
        4: []}                                          # segmented by nobody


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
    check('the default is 10,000 cells', X.DEFAULT_N_CELLS == 10000)
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
    try:
        d, avail = X.draw_cells('/nowhere', [1, 2, 3, 4], n_cells=None)
        check('None is every cell, in store order',
              d[3] == FAKE[3] and d[1] == FAKE[1] and d[4] == []
              and avail == {1: 20, 2: 50, 3: 10, 4: 0}, str(avail))
        d30, _ = X.draw_cells('/nowhere', [1, 2, 3, 4], n_cells=30)
        check('30 over (20, 50, 10, 0) is 10 each',
              [len(d30[f]) for f in (1, 2, 3, 4)] == [10, 10, 10, 0])
        check('each FOV draws its own cells, no repeats',
              all(len(set(d30[f])) == len(d30[f])
                  and set(d30[f]) <= set(FAKE[f]) for f in (1, 2, 3)))
        check('and not simply the first ten in store order',
              d30[2] != FAKE[2][:10], str(d30[2]))
        d45, _ = X.draw_cells('/nowhere', [1, 2, 3, 4], n_cells=45)
        check('45 fills the small FOV and shares the rest: 10 + (17, 18)',
              len(d45[3]) == 10 and sorted([len(d45[1]), len(d45[2])])
              == [17, 18], str([len(d45[f]) for f in (1, 2, 3)]))
        check('MORE CELLS EXTEND THE PREFIX: the 30-draw is a prefix of '
              'the 45-draw in every FOV',
              all(d45[f][:len(d30[f])] == d30[f] for f in (1, 2, 3)))
        again, _ = X.draw_cells('/nowhere', [1, 2, 3, 4], n_cells=30)
        check('the same seed draws the same cells', again == d30)
        other, _ = X.draw_cells('/nowhere', [1, 2, 3, 4], n_cells=30, seed=1)
        check('another seed draws other cells', other[2] != d30[2])
        alone, _ = X.draw_cells('/nowhere', [2], n_cells=10)
        check("a FOV's order does not depend on which other FOVs are in "
              "the run", alone[2] == d30[2])
        big, _ = X.draw_cells('/nowhere', [1, 2, 3], n_cells=10000)
        check('more than exist: every cell, still in the permuted order',
              sorted(big[2]) == sorted(FAKE[2]) and big[2] != FAKE[2]
              and big[2][:10] == d30[2])

        p635 = X.plan('/nowhere', [1, 2, 3, 4], ['H1', 'H2'], 635, n_cells=30)
        p555 = X.plan('/nowhere', [1, 2, 3, 4], ['H1', 'H2'], 555, n_cells=30)
        check('EVERY CHANNEL PLANS THE SAME CELLS', p635 == p555 and p635)
        tags = [(f, h, t) for (f, h, _ids, t) in p635]
        check('chunks of 8 off the prefix, tagged by position, FOV-major',
              tags[:4] == [(1, 'H1', 'c000'), (1, 'H1', 'c001'),
                           (1, 'H2', 'c000'), (1, 'H2', 'c001')]
              and len(p635) == 3 * 2 * 2, str(tags[:6]))
        p45 = X.plan('/nowhere', [1, 2, 3, 4], ['H1'], 635, n_cells=45)
        c0 = {(f, t): ids for (f, _h, ids, t) in p635 if _h == 'H1'}
        c1 = {(f, t): ids for (f, _h, ids, t) in p45}
        check('a bigger draw keeps every full chunk as it was',
              c1[(1, 'c000')] == c0[(1, 'c000')]
              and c1[(2, 'c000')] == c0[(2, 'c000')]
              and c1[(1, 'c001')][:2] == c0[(1, 'c001')])
        given = X.plan('/nowhere', [1, 2], ['H1'], 635,
                       cells={1: [5, 6, 7], 2: []})
        check('a caller that drew already is taken at its word',
              given == [(1, 'H1', [5, 6, 7], 'c000')], str(given))
        old = X.plan('/nowhere', [3], ['H1'], 635)
        check('no count: store order, chunked by position -- the earlier '
              'behaviour, exactly',
              [ids for (_f, _h, ids, _t) in old]
              == [FAKE[3][:8], FAKE[3][8:]], str(old))
        sig = inspect.signature(X.extract).parameters
        check('extract() names the draw so it never leaks into the chunk '
              'kwargs', all(k in sig for k in ('n_cells', 'seed', 'cells')))
    finally:
        X.cell_masks = real


def test_cli():
    print('build_bundle: --n-cells, the record, the warning')
    import tools.build_bundle as BB
    src = inspect.getsource(BB.main)
    check('--n-cells defaults to the extractor\'s default',
          "'--n-cells'" in src and 'default=X.DEFAULT_N_CELLS' in src)
    check('--cell-seed, default 0', "'--cell-seed'" in src
          and 'default=0' in src.split("'--cell-seed'")[1][:80])
    check('the masks are read ONCE and handed to the draw',
          'ids_by_fov = cells_in(store, pool)' in src
          and 'ids_by_fov=ids_by_fov' in src)
    check('the drawn cells are what extract() cuts', 'cells=drawn' in src)
    check('the manifest records the draw',
          all(k in src for k in ('n_cells=n_cells', 'cell_seed=cell_seed',
                                 'cells_drawn=cells', 'cells_drawn_per_fov')))
    check('a different draw against shards on disk is SAID',
          'drew their' in src and 'cells differently' in src
          and '--rebuild' in src)
    if not os.path.isdir(STORE):
        check('the real store is reachable', False,
              'skipped: ' + STORE + ' absent -- dry run unvalidated')
        return
    out = subprocess.run(
        [sys.executable, '-u', 'tools/build_bundle.py', STORE,
         '--out', DRY_OUT, '--channel', '635', '--hybes', 'Hyb_101',
         '--n-fovs', '2', '--seed', '20260910', '--n-cells', '50',
         '--dry-run'],
        capture_output=True, text=True, timeout=600,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    text = out.stdout
    m = re.search(r'per FOV drawn/have: (.*)', text)
    pairs = re.findall(r'(\d+):(\d+)/(\d+)', m.group(1)) if m else []
    check('dry run on the real store: 50 cells over 2 FOVs is 25 + 25',
          out.returncode == 0 and 'cells   50 of' in text
          and [int(b) for _f, b, _c in pairs] == [25, 25]
          and all(int(c) >= 25 for _f, _b, c in pairs),
          (m.group(0) if m else text[-400:]))
    check('and the work line counts crops from the DRAWN cells',
          'work    2 FOVs x 1 hybes = 50 crops' in text)
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
