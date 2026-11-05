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

PASS, FAIL = [], []


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

    t = BU.build_cell_traces(spots, 40, engines={
        'multiplicity': fake_mult, 'cluster': fake_cluster,
        'paths': fake_paths})
    check('each engine is asked exactly once',
          called == ['multiplicity', 'cluster', 'paths'], str(called))
    check('the stub\'s k is the one asked for', t.multiplicity.k == 3)
    # ... and the clustering still REFUSES to invent a third allele this
    # cell cannot support, which is the outlier guard doing its job
    check('but a cell that supports 2 gives 2, not a padded 3',
          t.rows.shape[0] == 2 and t.clustering.k == 2,
          '%s -- %s' % (t.rows.shape, t.clustering.reason))

    print('  replacing one leaves the other two alone')
    t2 = BU.build_cell_traces(spots, 40, engines={'multiplicity': fake_mult})
    check('the defaults still run under a stubbed count',
          t2.rows.shape[0] == 2 and t2.clustering.k == 2,
          str(t2.rows.shape))

    print('  a known ploidy beats an estimated one')
    t3 = BU.build_cell_traces(spots, 40, k=2)
    check('k given is k used', t3.rows.shape[0] == 2, str(t3.rows.shape))
    check('and the estimate is still reported beside it',
          t3.multiplicity.k == 2, str(t3.multiplicity))


def main():
    inputs()
    multiplicity()
    clustering()
    paths()
    seams()
    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
