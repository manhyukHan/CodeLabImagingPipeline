"""Paths found as chains, scored by the measured chain law.

WHAT THESE PIN. Not the physics -- that was measured once and lives in
physics/MEASUREMENT.md -- but the four properties the route depends on,
each of which was broken at some point while this was being written:

  1. the scale FOLLOWS THE GENOME. A step's acceptance distance moves
     with its genomic separation, which is the whole difference from the
     1.0 um radius it replaces.
  2. a path is a CHAIN: strictly increasing slots, so it can never take
     two spots of one round, and that is the graph's shape rather than a
     repair afterwards.
  3. clutter makes paths HARDER to justify, not easier. The first
     version charged a flat log V per unclaimed spot and built whole
     paths out of noise.
  4. two paths that CROSS get uncrossed. Extracting the best path first
     leaves it woven through both alleles when they overlap, and
     re-deriving it cannot undo that.

THE FIXTURE IS A REAL CHAIN. Positions are drawn so that
Var(x_i - x_j) = sigma_poly(s_ij)^2 exactly -- the fBm construction --
because an independent-increment walk is a DIFFERENT polymer (alpha
0.5, not 0.267) and a compact Gaussian blob is not a polymer at all.
Testing a chain loss against either would be testing the wrong thing,
and a blob does in fact come apart into three paths, correctly.

Run: python tests/test_pathfind.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from codelab_pipeline.localization import pathfind as PF
from codelab_pipeline.physics import chain as PH
from codelab_pipeline.physics import panel as PANEL

_n = {'pass': 0, 'fail': 0}
CAL = PH.Calibration()


def check(name, ok, extra=''):
    _n['pass' if ok else 'fail'] += 1
    print('  %-4s %s%s' % ('ok' if ok else 'FAIL', name,
                           ('   [%s]' % extra) if extra else ''))


def chain_positions(n_slots, step_kb, rs):
    """A chain obeying sigma_poly(s) at EVERY separation, not just one."""
    s = np.arange(n_slots) * float(step_kb)
    v = CAL.sigma_poly(np.maximum(np.abs(s[:, None] - s[None, :]), 1e-9)) ** 2
    vi = CAL.sigma_poly(np.maximum(s, 1e-9)) ** 2
    cov = 0.5 * (vi[:, None] + vi[None, :] - v)
    w, q = np.linalg.eigh(cov)
    return (q @ np.diag(np.sqrt(np.maximum(w, 0)))) @ rs.normal(size=(n_slots, 3))


def cell(k, n_slots, step_kb, eff=0.9, seed=0, sep_um=2.0, clutter=0,
         box=8.0):
    """(bins, pos, truth) -- k chains, optional uniform clutter."""
    rs = np.random.RandomState(seed)
    bins, pos, truth = [], [], []
    for j in range(k):
        walk = chain_positions(n_slots, step_kb, rs) + np.array([0., j * sep_um, 0.])
        for b in range(n_slots):
            if rs.rand() > eff:
                continue
            bins.append(b)
            truth.append(j)
            pos.append(walk[b] + rs.normal(0, CAL.sigma_loc_xy, 3))
    for _ in range(clutter):
        bins.append(rs.randint(n_slots))
        truth.append(-1)
        pos.append(rs.uniform(-box / 2, box / 2, 3))
    return np.array(bins), np.array(pos), np.array(truth)


def purity(labels, truth, j):
    m = labels == j
    if not m.any():
        return 0.0
    _v, c = np.unique(truth[m], return_counts=True)
    return float(c.max()) / float(m.sum())


def mixture_costs(pos, bins, sep, cal):
    """step_costs scored by the MIXTURE instead of the chain term.

    Kept here, not shipped, so the alternative this module rejects stays
    runnable: a claim that one scoring beats another is only as good as
    the ability to re-run the other.
    """
    p = np.asarray(pos, float)
    b = np.asarray(bins, np.int64)
    n = len(p)
    out = np.full((n, n), np.inf)
    s = sep.of(b)
    ok = (b[None, :] > b[:, None]) & np.isfinite(s)
    d = p[None, :, :] - p[:, None, :]
    cost = PH.pair_cost(d, np.where(ok, s, 1.0), cal)
    out[ok] = cost[ok]
    return out


def test_the_scale_follows_the_genome():
    print('\nthe acceptance distance moves with genomic separation')
    steps = (25, 100, 1000)
    got = [PF.acceptance_um(s, 100, CAL) for s in steps]
    check('it rises with s over the range steps actually span',
          all(a < b for a, b in zip(got, got[1:])),
          ' '.join('%.2f' % v for v in got))
    check('by half again, not by a rounding error',
          got[-1] / got[0] > 1.5, 'x%.2f' % (got[-1] / got[0]))

    print('  and it TURNS OVER past about a megabase, which is real')
    # a step across a huge genomic gap is broad, so its density is low
    # everywhere: even a perfect step carries little, and the normalizing
    # term eats the budget the widening variance gave. The same
    # degeneracy MEASUREMENT.md reports for the mixture weight at large
    # s, and the reason `saturation_um` returns infinity there.
    far = PF.acceptance_um(4000, 100, CAL)
    check('4 Mb accepts no further than 1 Mb does', far < got[-1],
          '%.2f vs %.2f um' % (far, got[-1]))

    print('  it also tightens as the cell gets more crowded')
    few = PF.acceptance_um(100, 20, CAL)
    many = PF.acceptance_um(100, 400, CAL)
    check('a crowded cell demands a closer step', many < few,
          '%.2f um at 400 spots vs %.2f at 20' % (many, few))


def test_a_path_is_a_chain():
    print('\na path walks the slots forward, one spot each')
    bins, pos, _t = cell(1, 30, 50.0, eff=1.0, seed=1)
    # give every slot a second, far-away spot: a cloud-shaped objective
    # would be free to take both
    bins = np.concatenate([bins, bins])
    pos = np.concatenate([pos, pos + np.array([0., 3.0, 0.])])
    sep = PF.Separation.uniform(30, 50.0)
    w = PF.step_weights(pos, bins, sep, CAL)
    b = np.asarray(bins)
    back = np.isfinite(w) & (b[None, :] <= b[:, None])
    check('no step goes to the same slot or backwards', not back.any(),
          '%d such steps' % int(back.sum()))
    idx, weight = PF.best_path(w, bins)
    slots = [int(bins[i]) for i in idx]
    check('so the best path has strictly increasing slots',
          all(a < c for a, c in zip(slots, slots[1:])), str(slots[:8]))
    check('and it is worth something', weight < 0, '%.1f nats' % weight)


def test_it_finds_the_alleles():
    print('\none, two and three alleles, 60 slots at 50 kb')
    sep = PF.Separation.uniform(60, 50.0)
    for truth_k in (1, 2, 3):
        ks, purs = [], []
        for seed in range(4):
            bins, pos, truth = cell(truth_k, 60, 50.0, seed=seed)
            r = PF.choose_paths(pos, bins, sep, cal=CAL)
            ks.append(r.k)
            purs += [purity(r.labels, truth, j) for j in range(r.k)]
        check('k = %d is recovered in every seed' % truth_k,
              ks == [truth_k] * 4, str(ks))
        check('  and every path is one allele',
              min(purs) > 0.95, 'worst purity %.2f' % min(purs))

    print('  two alleles 1 um apart still come apart')
    pur = []
    for seed in range(4):
        bins, pos, truth = cell(2, 60, 50.0, seed=seed, sep_um=1.0)
        r = PF.choose_paths(pos, bins, sep, cal=CAL)
        pur += [purity(r.labels, truth, j) for j in range(r.k)]
    check('purity stays high at one micron', min(pur) > 0.9,
          'worst %.2f' % min(pur))

    print('  and OVERLAPPING alleles are where the chain term earns its place')
    # the mixture's second component IS "the other allele", so scoring a
    # step with it makes stepping across cheap exactly here
    scores = {}
    for tag in ('chain', 'mixture'):
        pur = []
        for seed in range(8):
            bins, pos, truth = cell(2, 60, 50.0, seed=seed, sep_um=0.4)
            c = (PF.step_costs(pos, bins, sep, CAL) if tag == 'chain'
                 else mixture_costs(pos, bins, sep, CAL))
            labels, weights = PF.find_paths(pos, bins, 2, sep, CAL, costs=c)
            pur += [purity(labels, truth, j) for j in range(len(weights))]
        scores[tag] = float(np.mean(pur))
    check('the chain term is the purer of the two at 0.4 um',
          scores['chain'] > scores['mixture'],
          'chain %.3f vs mixture %.3f' % (scores['chain'], scores['mixture']))


def test_crossing_paths_are_uncrossed():
    print('\ntwo paths that cross get their tails exchanged')
    # built so the greedy extraction MUST cross: the two chains swap
    # which is cheaper half way along
    bins = np.repeat(np.arange(8), 2)
    a = np.array([[0., 0., 0.]] * 8)
    b = np.array([[0., 0.9, 0.]] * 8)
    a[:, 1] += np.linspace(0, 0.9, 8)      # A drifts toward B
    b[:, 1] -= np.linspace(0, 0.9, 8)      # B drifts toward A
    pos = np.empty((16, 3))
    pos[0::2], pos[1::2] = a, b
    sep = PF.Separation.uniform(8, 50.0)
    w = PF.step_weights(pos, bins, sep, CAL)
    paths = [list(range(0, 16, 2)), list(range(1, 16, 2))]
    before = PF._path_weight(w, paths[0]) + PF._path_weight(w, paths[1])
    moved = PF._uncross(w, bins, paths)
    after = PF._path_weight(w, paths[0]) + PF._path_weight(w, paths[1])
    check('a crossing is found', moved)
    check('and exchanging the tails is cheaper', after < before,
          '%.2f -> %.2f nats' % (before, after))
    check('both halves are still legal paths',
          all(all(bins[i] < bins[j] for i, j in zip(p, p[1:]))
              for p in paths))
    check('and no spot was lost or duplicated',
          sorted(paths[0] + paths[1]) == list(range(16)))


def test_clutter_does_not_make_paths():
    print('\nclutter makes a path harder to justify, not easier')
    sep = PF.Separation.uniform(60, 50.0)
    out = []
    for c in (0, 60, 240):
        ks = []
        for seed in range(3):
            bins, pos, truth = cell(1, 60, 50.0, seed=seed, clutter=c)
            r = PF.choose_paths(pos, bins, sep, cal=CAL)
            ks.append(r.k)
            out.append(int(((r.labels >= 0) & (truth == -1)).sum()))
        check('%3d clutter spots still give k = 1' % c, ks == [1, 1, 1],
              str(ks))
    check('and almost no clutter spot lands on a path', max(out) <= 2,
          'worst %d taken' % max(out))

    print('  which is the rate doing it, not the volume')
    check('a crowded cell prices an unclaimed spot lower',
          PF.clutter_nats(300) < PF.clutter_nats(50),
          '%.2f vs %.2f nats' % (PF.clutter_nats(300), PF.clutter_nats(50)))


def test_the_clutter_rate_is_re_estimated():
    print('\nthe clutter rate comes from what the last pass left over')
    # THE FIRST PASS HAS TO ASSUME EVERY SPOT MIGHT BE CLUTTER, which
    # over-charges every step in a clean cell and makes the loss decline
    # real loci -- the same complaint `reach` earned, by another road.
    for step, n_slots, allowed in ((50.0, 60, 0.02), (200.0, 91, 0.02)):
        sep = PF.Separation.uniform(n_slots, step)
        once = dropped = total = 0
        for seed in range(6):
            bins, pos, truth = cell(2, n_slots, step, seed=seed)
            costs = PF.step_costs(pos, bins, sep, CAL)
            for rounds, bucket in ((1, 'once'), (4, 'settled')):
                labels, weights = PF.find_paths(pos, bins, 2, sep, CAL,
                                                costs=costs,
                                                clutter_rounds=rounds)
                lost = 0
                for j in range(len(weights)):
                    m = labels == j
                    v, c = np.unique(truth[m], return_counts=True)
                    own = v[np.argmax(c)]
                    lost += int(((truth == own) & (labels < 0)).sum())
                    if bucket == 'settled':
                        total += int((truth == own).sum())
                if bucket == 'once':
                    once += lost
                else:
                    dropped += lost
        print('   %2d slots at %5.0f kb: %d real loci declined on the first '
              'pass, %d once the rate settles (of %d)'
              % (n_slots, step, once, dropped, total))
        check('a settled rate keeps nearly every real locus',
              dropped <= allowed * total,
              '%.1f%% lost' % (100.0 * dropped / max(total, 1)))
        check('  and it is the re-estimation doing it', dropped < once,
              '%d vs %d' % (dropped, once))


def test_the_prior_separates_real_from_spurious():
    print('\nthe prior sits between two populations, it is not a threshold')
    sep = PF.Separation.uniform(60, 50.0)

    def gain_of_second(bins, pos):
        c = PF.step_costs(pos, bins, sep, CAL)
        _l1, w1 = PF.find_paths(pos, bins, 1, sep, CAL, costs=c)
        _l2, w2 = PF.find_paths(pos, bins, 2, sep, CAL, costs=c)
        return sum(w2) - sum(w1)

    spurious = [gain_of_second(*cell(1, 60, 50.0, seed=s, clutter=60)[:2])
                for s in range(8)]
    real = [gain_of_second(*cell(2, 60, 50.0, seed=s)[:2]) for s in range(8)]
    check('a spurious second path gains little',
          min(spurious) > -PF.DEFAULT_PATH_PRIOR_NATS,
          'worst %.1f nats vs prior %.1f' % (min(spurious),
                                             PF.DEFAULT_PATH_PRIOR_NATS))
    check('a real one gains a great deal',
          max(real) < -5 * PF.DEFAULT_PATH_PRIOR_NATS,
          'weakest %.1f nats' % max(real))


def test_the_separation_comes_from_the_design():
    print('\ngenomic separation comes from the panel, not from the index')
    fd, path = tempfile.mkstemp(suffix='_bed.txt')
    with os.fdopen(fd, 'w') as f:
        f.write('track name="toy"\n')
        # slots 0..2 are 30 kb apart; slot 3 sits 530 kb later; slot 4
        # is on another chromosome entirely
        for c, a, b, nm in (('chr20', 100000, 130000, 'TOY_001'),
                            ('chr20', 130001, 160001, 'TOY_002'),
                            ('chr20', 160002, 190002, 'TOY_003'),
                            ('chr20', 690003, 720003, 'TOY_004'),
                            ('chr7', 10000, 40000, 'TOY_005')):
            f.write('%s\t%d\t%d\t%s\t1\t+\n' % (c, a, b, nm))
    try:
        design = PANEL.read_probe_design(path)
        sep = PF.Separation.from_design(design, [1, 2, 3, 4, 5])
        m = sep.matrix
        check('neighbouring slots are one step apart',
              abs(m[0, 1] - 30.0) < 0.1, '%.1f kb' % m[0, 1])
        check('a gap in the design is NOT one step',
              abs(m[2, 3] - 530.0) < 0.1, '%.1f kb' % m[2, 3])
        check('and a cross-chromosome pair has no separation at all',
              np.isnan(m[0, 4]))
        # the step that matters: a pair with no separation can never be
        # a step, however close the two spots happen to be
        pos = np.array([[0., 0., 0.], [0., 0.05, 0.]])
        w = PF.step_weights(pos, [0, 4], sep, CAL)
        check('so no step is ever scored between them',
              not np.isfinite(w).any(), str(w))
        check('the source says which design it was',
              'probe design' in sep.source, sep.source)
    finally:
        os.unlink(path)

    print('  and the fallback says it is assuming')
    u = PF.Separation.uniform(10, 50.0)
    check('a uniform separation labels itself ASSUMED',
          'ASSUMED' in u.source, u.source)


def test_it_is_deterministic():
    print('\nthe same cell gives the same answer')
    sep = PF.Separation.uniform(60, 50.0)
    bins, pos, _t = cell(2, 60, 50.0, seed=7)
    a = PF.choose_paths(pos, bins, sep, cal=CAL)
    b = PF.choose_paths(pos, bins, sep, cal=CAL)
    check('labels are identical', (a.labels == b.labels).all())
    check('and so is the cost', a.cost == b.cost)


def test_rows_point_back_at_the_callers_spots():
    print('\nthe answer points back at the rows it came from')
    bins = np.array([0, 0, 1, 2, 2])
    pos = np.array([[0., 0., 0.], [0., 3., 0.],
                    [0., 0.1, 0.],
                    [0., 0.2, 0.], [0., 3.1, 0.]])
    rows = np.array([100, 101, 102, 103, 104])
    labels = np.array([0, PF.UNASSIGNED, 0, 0, PF.UNASSIGNED])
    out = PF.path_rows(labels, bins, rows, 1, 3)
    check('one row id per slot, the caller\'s own',
          out.tolist() == [[100, 102, 103]], str(out))
    check('and an unclaimed slot reads NO_SPOT',
          PF.path_rows(np.full(5, PF.UNASSIGNED), bins, rows, 1, 3).tolist()
          == [[PF.NO_SPOT] * 3])


def main():
    for t in (test_the_scale_follows_the_genome,
              test_the_clutter_rate_is_re_estimated,
              test_a_path_is_a_chain,
              test_it_finds_the_alleles,
              test_crossing_paths_are_uncrossed,
              test_clutter_does_not_make_paths,
              test_the_prior_separates_real_from_spurious,
              test_the_separation_comes_from_the_design,
              test_it_is_deterministic,
              test_rows_point_back_at_the_callers_spots):
        t()
    print('\n%d passed, %d failed' % (_n['pass'], _n['fail']))
    return 1 if _n['fail'] else 0


if __name__ == '__main__':
    sys.exit(main())
