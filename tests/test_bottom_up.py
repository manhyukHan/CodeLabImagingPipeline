"""Traces built from the spots up.

Three questions, three pure functions: how many alleles, which spots are
whose, and which ONE spot is each allele's bin. The constraint doing the
work is that a readout round images one locus, so it gives one signal
per allele -- which is both what pins the allele count down and what
fixes the pairs position alone gets wrong.

Run: python tests/test_bottom_up.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                            # noqa: E402

from codelab_pipeline.localization import bottom_up as BU     # noqa: E402
from codelab_pipeline.localization import pathfind as PF      # noqa: E402
from codelab_pipeline.physics import chain as PH              # noqa: E402

PASS, FAIL = [], []
CAL = PH.Calibration()

#: every cell here is 30 or 40 slots of a regular panel; the separation
#: is stated rather than assumed anywhere downstream
SEP30 = PF.Separation.uniform(30, 50.0)
SEP40 = PF.Separation.uniform(40, 50.0)


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(('  ok ' if cond else 'FAIL ') + name
          + (f'   [{detail}]' if detail and not cond else ''))


def make_cell(k, n_slots, efficiency=1.0, sep_um=2.0, rg_um=0.25, seed=0):
    """k alleles, each a compact cloud, one spot per (allele, bin) kept
    with probability `efficiency`. Returns (CellSpots, truth)."""
    rs = np.random.RandomState(seed)
    centres = np.array([[0.0, i * sep_um, 0.0] for i in range(k)])
    bins, pos, truth = [], [], []
    for j in range(k):
        for b in range(n_slots):
            if rs.rand() > efficiency:
                continue
            bins.append(b)
            pos.append(centres[j] + rs.normal(0, rg_um, 3))
            truth.append(j)
    return (BU.cell_spots(bins, np.array(pos)), np.array(truth), centres)


def chain_cell(k, n_slots, step_kb=50.0, efficiency=1.0, sep_um=2.0, seed=0):
    """k alleles that are actually CHAINS, not blobs. (CellSpots, truth).

    Positions obey Var(x_i - x_j) = sigma_poly(s_ij)^2 at every pair, so
    the fixture is drawn from the same law the loss scores with. A
    compact Gaussian cloud -- which is what make_cell gives, and what
    the k-means pieces above are rightly tested on -- is NOT a chain,
    and the chain loss correctly takes one apart into three paths.
    """
    rs = np.random.RandomState(seed)
    g = np.arange(n_slots) * float(step_kb)
    v = CAL.sigma_poly(np.maximum(np.abs(g[:, None] - g[None, :]), 1e-9)) ** 2
    vi = CAL.sigma_poly(np.maximum(g, 1e-9)) ** 2
    w, q = np.linalg.eigh(0.5 * (vi[:, None] + vi[None, :] - v))
    root = q @ np.diag(np.sqrt(np.maximum(w, 0)))
    bins, pos, truth = [], [], []
    for j in range(k):
        walk = root @ rs.normal(size=(n_slots, 3)) + np.array([0., j * sep_um, 0.])
        for b in range(n_slots):
            if rs.rand() > efficiency:
                continue
            bins.append(b)
            truth.append(j)
            pos.append(walk[b] + rs.normal(0, CAL.sigma_loc_xy, 3))
    return BU.cell_spots(bins, np.array(pos)), np.array(truth)


def inputs():
    print('the input shape is checked once, not trusted everywhere')
    try:
        BU.cell_spots([0, 1], np.zeros((3, 2)))
        check('a transposed position array is refused', False, 'accepted')
    except ValueError as e:
        check('a transposed position array is refused', '(n, 3)' in str(e),
              str(e))
    try:
        BU.cell_spots([0, 1, 2], np.zeros((2, 3)))
        check('a length mismatch is refused', False, 'accepted')
    except ValueError as e:
        check('a length mismatch is refused', 'but' in str(e), str(e))
    s = BU.cell_spots([0, 1], np.zeros((2, 3)))
    check('row ids default to the row order', list(s.row) == [0, 1])
    check('and amplitude/quality default to NaN, not to zero',
          np.isnan(s.amp).all() and np.isnan(s.quality).all())

    print('  an empty bin is counted, because it is the evidence')
    s2 = BU.cell_spots([0, 0, 2], np.zeros((3, 3)))
    counts, _ = BU.bin_counts(s2, n_slots=4)
    check('zeros are kept', list(counts) == [2, 0, 1, 0], str(counts))


def multiplicity():
    print('\nhow many alleles, from the per-bin counts alone')
    for k in (1, 2, 3):
        spots, _t, _c = make_cell(k, 60, efficiency=0.9, seed=k)
        m = BU.multiplicity_gate(spots, 60)
        check('k=%d at 90%% efficiency is read back' % k, m.k == k,
              '%s (e=%.2f)' % (m.k, m.efficiency))

    print('  and the efficiency comes with it, because they trade off')
    spots, _t, _c = make_cell(2, 120, efficiency=0.6, seed=7)
    m = BU.multiplicity_gate(spots, 120)
    check('k=2 survives a 60% efficiency', m.k == 2,
          '%s (e=%.2f)' % (m.k, m.efficiency))
    check('and the efficiency is recovered to within 0.1',
          abs(m.efficiency - 0.6) < 0.1, '%.2f' % m.efficiency)

    print('  the obvious estimators are the ones that fail')
    counts, _ = BU.bin_counts(spots, 120)
    check('the MODE reads this dim cell as fewer alleles than it has',
          np.bincount(counts).argmax() < 2,
          'mode %d, truth 2' % np.bincount(counts).argmax())
    check('while the fit does not', BU.multiplicity_gate(spots, 120).k == 2)

    print('  the COUNTS engine is kept, and is not the default')
    thin = BU.cell_spots([0, 1], np.zeros((2, 3)))
    m2 = BU.multiplicity_gate(thin, 2,
                              engine=BU.counts_engine(BU.binomial_multiplicity))
    check('two bins is not a count', m2.k is None and 'needed' in m2.reason,
          m2.reason)
    empty = BU.cell_spots([], np.zeros((0, 3)))
    m3 = BU.multiplicity_gate(empty, 40)
    check('a cell with no spot is 0 alleles, not None',
          m3.k == 0 and 'no spot' in m3.reason, m3.reason)

    print('  AND NO SINGLE BARCODE DECIDES IT')
    # one clean allele over 40 bins, then ONE barcode picking up two
    # spurious extras. Before the contamination term this forced k >= 3:
    # a whole cell's allele count set by one round.
    rs = np.random.RandomState(4)
    pos40 = rs.normal(0, 0.2, (42, 3))
    counts = BU.counts_engine(BU.binomial_multiplicity)
    for where, label in ((0, 'bin 0'), (37, 'a LATE barcode')):
        cs = BU.cell_spots(list(range(40)) + [where, where], pos40)
        check('spatially, one barcode\'s extra spots are not an allele '
              '(%s)' % label, BU.multiplicity_gate(cs, 40).k == 1,
              BU.multiplicity_gate(cs, 40).reason)
        m = BU.multiplicity_gate(cs, 40, engine=counts)
        check('and the counts engine no longer forces k=3 either (%s)'
              % label, m.k == 1 and 'above k' in m.reason, m.reason)


def clustering():
    print('\nwhich spots are whose')
    spots, truth, centres = make_cell(2, 60, efficiency=1.0, sep_um=2.0,
                                      seed=3)
    clu = BU.cluster_cell(spots, 2)
    check('both alleles are found', clu.k == 2)
    agree = max((clu.labels == truth).mean(), (clu.labels != truth).mean())
    check('every spot lands on its own allele', agree == 1.0,
          '%.2f' % agree)
    check('the centres are where they were put',
          min(np.linalg.norm(clu.centres - c, axis=1).min()
              for c in centres) < 0.1,
          str(clu.centres))

    print('  seeding is deterministic, so a trace does not move on its own')
    a = BU.cluster_cell(spots, 2).labels
    b = BU.cluster_cell(spots, 2).labels
    check('the same cell clusters the same way twice',
          np.array_equal(a, b))

    print('  k=1 is every spot, not a degenerate fit')
    one = BU.cluster_cell(spots, 1)
    check('one centre, every label 0',
          one.k == 1 and set(one.labels) == {0} and one.centres.shape == (1, 3))


def paths():
    print('\none spot per bin per allele')
    spots, truth, _c = make_cell(2, 40, efficiency=1.0, sep_um=2.0, seed=5)
    clu = BU.cluster_cell(spots, 2)
    p = BU.resolve_paths(spots, clu, 40)
    check('each allele gets one spot in every bin',
          p.rows.shape == (2, 40) and (p.rows >= 0).all(),
          str(p.rows.shape))
    check('and no spot is taken twice',
          len(set(p.rows.ravel().tolist())) == p.rows.size,
          '%d unique of %d' % (len(set(p.rows.ravel().tolist())),
                               p.rows.size))

    print('  a bin with fewer spots than alleles leaves one empty')
    bins = [0, 0, 1, 2, 2]
    pos = np.array([[0., 0., 0.], [0., 2., 0.],
                    [0., 0., 0.],
                    [0., 0., 0.], [0., 2., 0.]])
    s = BU.cell_spots(bins, pos)
    clu2 = BU.cluster_cell(s, 2, radius_um=0.5, min_bins=2)
    p2 = BU.resolve_paths(s, clu2, 3)
    check('bin 1 is taken by one allele only',
          sorted(p2.rows[:, 1].tolist()) == [BU.NO_SPOT, 2],
          str(p2.rows[:, 1]))
    check('and it goes to the allele it is NEAR, not to allele 0',
          p2.rows[int(np.argmin(np.linalg.norm(clu2.centres
                                               - pos[2], axis=1))), 1] == 2,
          str(p2.rows))

    print('  the constraint fixes what position alone cannot')
    # two alleles 2 um apart, but in bin 1 BOTH spots sit near allele 0:
    # position says allele 0 twice, which one round cannot give
    bins = [0, 0, 1, 1, 2, 2]
    pos = np.array([[0., 0., 0.], [0., 2., 0.],
                    [0., 0.0, 0.], [0., 0.6, 0.],
                    [0., 0., 0.], [0., 2., 0.]])
    s = BU.cell_spots(bins, pos)
    clu3 = BU.cluster_cell(s, 2, radius_um=0.8, min_bins=2)
    p3 = BU.resolve_paths(s, clu3, 3)
    check('bin 1 still gives each allele exactly one spot',
          sorted(p3.rows[:, 1].tolist()) == [2, 3], str(p3.rows[:, 1]))
    check('and the move is reported, not silent', p3.moved >= 1,
          'moved %d' % p3.moved)

    print('  a spot beyond an allele\'s reach is refused, not taken')
    # two real alleles, each with two bins, plus one spot 40 um away
    bins = [0, 0, 1, 1, 2]
    pos = np.array([[0., 0., 0.], [0., 2., 0.],
                    [0., 0., 0.], [0., 2., 0.],
                    [0., 40., 0.]])
    s = BU.cell_spots(bins, pos)
    clu4 = BU.cluster_cell(s, 2, radius_um=0.5, min_bins=2)
    check('the outlier does not become an allele of its own', clu4.k == 2,
          '%s -- %s' % (clu4.k, clu4.reason))
    p4 = BU.resolve_paths(s, clu4, 3, max_um=1.0)
    check('and the far spot is nobody\'s',
          (p4.rows[:, 2] == BU.NO_SPOT).all(), str(p4.rows[:, 2]))
    p5 = BU.resolve_paths(s, clu4, 3)
    check('without a bound it is taken, so the bound is what refuses it',
          (p5.rows[:, 2] >= 0).any(), str(p5.rows[:, 2]))


def seams():
    print('\nthe three are separable, which is what the seam is for')
    spots, _t, _c = make_cell(2, 40, efficiency=0.9, seed=11)

    called = []

    def fake_mult(s, n_slots, **kw):
        called.append('multiplicity')
        return BU.Multiplicity(3, 0.5, 0.0, n_slots, np.zeros(0, dtype=int),
                               'stub')

    def fake_cluster(s, k, **kw):
        called.append('cluster')
        return BU.kmeans_cluster(s, k)

    def fake_paths(s, c, n, **kw):
        called.append('paths')
        return BU.nearest_path(s, c, n)

    t = BU.build_cell_traces(spots, 40, SEP40, engines={
        'multiplicity': fake_mult, 'cluster': fake_cluster,
        'paths': fake_paths})
    # THE SPLIT AND THE ASSIGNMENT ARE ONE STAGE NOW, so build_cell_traces
    # no longer calls a cluster engine and a paths engine in turn --
    # separating them was the defect, not the design. cluster_cell and
    # resolve_paths remain as the separable pieces and keep their own
    # seams; this route has the multiplicity seam and the joint fit.
    check('the multiplicity engine is still asked', called == ['multiplicity'],
          str(called))
    check('and the cluster/paths engines are not, because they merged',
          'cluster' not in called and 'paths' not in called, str(called))
    check('the joint fit decides the count, not the stubbed proposal',
          t.multiplicity.k == 3 and t.rows.shape[0] <= 3,
          '%s proposed, %d fitted' % (t.multiplicity.k, t.rows.shape[0]))
    check('and the trace reports both', t.paths is t.clustering)

    print('  a known ploidy beats an estimated one')
    t3 = BU.build_cell_traces(spots, 40, SEP40, k=2)
    check('k given is k used, with no sweep', t3.rows.shape[0] == 2,
          str(t3.rows.shape))
    check('and the estimate is still reported beside it',
          t3.multiplicity.k is not None, str(t3.multiplicity.k))


def joint():
    print('\nthe split and the assignment are ONE objective')
    cs, truth = chain_cell(2, 30, seed=0)
    f = BU.choose_paths(cs, 30, SEP30)
    check('a clean two-allele cell costs least at k=2', f.k == 2,
          '%s -- %s' % (f.k, f.reason))
    check('every spot is paid for', f.unassigned == 0, str(f.unassigned))
    check('and each path fills its own 30 bins',
          (f.rows >= 0).sum(axis=1).tolist() == [30, 30],
          str((f.rows >= 0).sum(axis=1)))
    check('each allele gets its own evidence, not a pooled number',
          len(f.weights) == 2 and max(f.weights) < 0,
          '%s nats' % [round(w) for w in f.weights])
    check('the cost of every k is reported, so a close call is visible',
          len(f.per_k) >= 2 and f.per_k[0][1] > f.per_k[1][1], str(f.per_k))

    print('  BIN IDENTITY DECIDES IT NOW -- the thing that was missing')
    rs = np.random.RandomState(0)
    perm = rs.permutation(len(cs.bin))
    f2 = BU.choose_paths(BU.cell_spots(np.asarray(cs.bin)[perm], cs.pos),
                         30, SEP30)
    check('the same points with shuffled labels cost MORE',
          f2.cost > f.cost, '%.1f vs %.1f nats' % (f2.cost, f.cost))
    check('and leave spots no path will pay for',
          f2.unassigned > f.unassigned,
          '%d vs %d' % (f2.unassigned, f.unassigned))

    print("  a bin's SECOND spot costs the path that cannot take it")
    # one path, two spots in bin 0: a chain walks the slots forward, so
    # taking both is not expensive -- it is not expressible
    bins3 = [0, 0] + list(range(1, 12))
    pos3 = np.array([[0., 0., 0.], [0., 0.3, 0.]]
                    + [[0., 0., 0.]] * 11, dtype=float)
    f3 = BU.fit_paths(BU.cell_spots(bins3, pos3), 12, 1,
                      PF.Separation.uniform(12, 50.0))
    check('one path takes one of them', (f3.rows[0] >= 0).sum() == 12,
          str((f3.rows[0] >= 0).sum()))
    check('and the other is unassigned, not quietly added',
          f3.unassigned == 1, str(f3.unassigned))

    print('  a path must pay for itself')
    one, _t = chain_cell(1, 30, seed=4)
    r1 = BU.choose_paths(one, 30, SEP30)
    check('a single allele stays k=1', r1.k == 1,
          '%s -- %s' % (r1.k, r1.reason))

    print('  and build_cell_traces now routes through it')
    t = BU.build_cell_traces(cs, 30, SEP30)
    check('the trace carries the fit, not a separate clustering',
          t.rows.shape[0] == 2 and t.paths is t.clustering,
          str(t.rows.shape))
    check('and the count proposal is still reported beside it',
          t.multiplicity is not None and t.multiplicity.k is not None,
          str(t.multiplicity.k))

    print('  the genomic separation is an INPUT, and it changes the answer')
    # the same cell told its panel is 50 kb per slot and 5 Mb per slot:
    # at 5 Mb every step spans so much genome that the law expects the
    # loci anywhere, and the chain stops being evidence
    wide = BU.choose_paths(cs, 30, PF.Separation.uniform(30, 5000.0))
    check('a panel claimed 100x coarser buys far less evidence',
          wide.cost > f.cost, '%.0f vs %.0f nats' % (wide.cost, f.cost))
    check('which is why Separation has no default',
          'ASSUMED' in f.reason, f.reason)


def cap():
    print('\nthe path count is CAPPED, because the cost cannot cap it')
    # eight real alleles, each a full 30-slot chain. Every one of them
    # pays for itself many times over, so only the cap refuses them.
    cs, _t = chain_cell(8, 30, sep_um=3.0, seed=1)
    f = BU.choose_paths(cs, 30, SEP30)
    check('eight real sites are still capped at the default',
          f.k <= pathfind_k_max(), '%s -- %s' % (f.k, f.reason))
    check('and the cap says so in the reason', 'capped at' in f.reason,
          f.reason)
    check('the cap is a default, not a law',
          BU.choose_paths(cs, 30, SEP30, k_max=8).k > pathfind_k_max(),
          str(BU.choose_paths(cs, 30, SEP30, k_max=8).k))
    two, _t2 = chain_cell(2, 30, sep_um=3.0, seed=1)
    check('a cell within the cap is untouched by it',
          BU.choose_paths(two, 30, SEP30).k == 2,
          str(BU.choose_paths(two, 30, SEP30).k))

    print('  and the count proposals are capped at the same number')
    m = BU.spatial_multiplicity(cs, 30, radius_um=1.0, min_bins=5)
    check('spatial_multiplicity stops at the cap', m.k <= BU.DEFAULT_K_MAX,
          '%s -- %s' % (m.k, m.reason))
    check('raised per call, it counts further',
          BU.spatial_multiplicity(cs, 30, radius_um=1.0, min_bins=5,
                                  k_max=8).k > BU.DEFAULT_K_MAX,
          str(BU.spatial_multiplicity(cs, 30, radius_um=1.0, min_bins=5,
                                      k_max=8).k))
    counts = np.array([BU.DEFAULT_K_MAX + 3] * 30)
    check('binomial_multiplicity cannot exceed it either',
          BU.binomial_multiplicity(counts).k <= BU.DEFAULT_K_MAX,
          str(BU.binomial_multiplicity(counts).k))


def pathfind_k_max():
    """The path cap lives with the loss now, beside the prior it works
    with; BU.DEFAULT_K_MAX still caps the COUNT PROPOSALS, which are a
    different question asked of the per-bin counts."""
    return PF.DEFAULT_K_MAX


def main():
    inputs()
    multiplicity()
    clustering()
    paths()
    seams()
    joint()
    cap()
    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
