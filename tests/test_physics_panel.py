"""A probe design is read the same from BED, LibRegions FASTA or probe FASTA.

WHY IT MATTERS. Genomic separation is the one input the physics term
cannot do without, and nothing in the pipeline carried it: the layout
spreadsheet has FolderName, Readouts, DataType, HybNum and channels, and
`genomic_resolution_kb` is a single per-experiment number that is often
blank. So `s` has to come from the probe design, and a design that is
misread is worse than one that is missing -- it yields a plausible wrong
number rather than an error.

THE TWO MISREADINGS THIS GUARDS. Multiplying an index by a resolution
(chr19's panel covers two regions 148 kb apart, so every pair spanning
the gap would be wrong, and a design may skip a readout that failed
specificity), and taking the readout id from the probe FASTA's own
`read001` token (it counts probes within the library, and on a panel
whose readouts start at 2 the two disagree).

Synthetic fixtures, so this runs anywhere. The real designs live on
`E:/Oligos` and are checked opportunistically at the end.

Run: python tests/test_physics_panel.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline.physics import panel as P

_n = {'pass': 0, 'fail': 0}

#: Regular 30 kb steps with ONE 530 kb jump in the middle -- the shape
#: of a panel covering two regions, which is what breaks an
#: index-times-resolution reading. Ids start at 002, since a design is
#: under no obligation to start at 1. The last entry is on another
#: chromosome.
_ROWS = [('chr20', 100000, 130000, 'TOY_002'),
         ('chr20', 130001, 160001, 'TOY_003'),
         ('chr20', 160002, 190002, 'TOY_004'),
         ('chr20', 690003, 720003, 'TOY_005'),      # the jump
         ('chr20', 720004, 750004, 'TOY_006'),
         ('chr7', 10000, 40000, 'TOY_007')]
BED = ('browser position:  chr20:100000-200000\n'
       'track name="toy" description="toy" visibility=1 itemRgb="On"\n'
       + ''.join('%s\t%d\t%d\t%s\t1\t+\t%d\t%d\t255,0,0\n'
                 % (c, a, b, n, a, b) for c, a, b, n in _ROWS))
LIBFA = ''.join('>%s:%d-%d gene=%s\nACGT\n' % (c, a, b, n)
                for c, a, b, n in _ROWS)
#: `read###` deliberately disagrees with the region number: it counts
#: probes within the library, so a reader that took it would renumber
#: the whole panel and every separation with it
PROBEFA = ''.join(
    '>A02F__read%03d__%s_%s:%d-%d_p%04d__A10R\nACGT\n'
    % (i + 1, n, c, a, b, p)
    for i, (c, a, b, n) in enumerate(_ROWS) for p in (1, 2))


def check(name, ok, extra=''):
    _n['pass' if ok else 'fail'] += 1
    print('  %-4s %s%s' % ('ok' if ok else 'FAIL', name,
                           ('   [%s]' % extra) if extra else ''))


def written(text, suffix):
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(text)
    return path


def test_every_format_reads_the_same():
    print('\nBED, LibRegions FASTA and probe FASTA agree')
    paths = {'bed': written(BED, '_bed.txt'),
             'libfa': written(LIBFA, '_LibRegions.fasta'),
             'probefa': written(PROBEFA, '_probe.fasta')}
    try:
        got = {k: P.read_probe_design(v) for k, v in paths.items()}
        for k, d in got.items():
            check('%-8s finds 6 readouts' % k, len(d) == 6, str(len(d)))
            check('%-8s keeps the ids the design uses (2..7)' % k,
                  sorted(d.by_id) == [2, 3, 4, 5, 6, 7],
                  str(sorted(d.by_id)))
        a, b = got['bed'].by_id, got['probefa'].by_id
        check('BED and probe FASTA give identical regions', a == b)
        check('LibRegions FASTA too', a == got['libfa'].by_id)
    finally:
        for v in paths.values():
            os.unlink(v)


def test_separation_uses_coordinates_not_indices():
    print('\nseparation comes from the coordinates')
    p = written(BED, '_bed.txt')
    try:
        d = P.read_probe_design(p)
        # consecutive ids, but the design jumps: 3 -> 4 is not one step
        s23 = d.separation_kb(2, 3)
        s34 = d.separation_kb(4, 5)
        check('adjacent regions are one step apart', abs(s23 - 30.0) < 0.1,
              '%.1f kb' % s23)
        check('a jump in the design is NOT one step',
              abs(s34 - 530.0) < 0.1, '%.1f kb' % s34)
        check('which an index-times-step reading would have missed',
              abs(s34 - s23) > 100.0)
        check('the median step is reported, not assumed',
              abs(d.step_kb() - 30.0) < 0.1, '%.1f kb' % d.step_kb())
    finally:
        os.unlink(p)


def test_a_cross_chromosome_pair_is_refused():
    print('\na separation that is not a contour length is refused')
    p = written(BED, '_bed.txt')
    try:
        d = P.read_probe_design(p)
        # TOY_005 is on chr7: no contour runs between the two, so a
        # chain model must never be handed a number for it
        check('cross-chromosome separation is None',
              d.separation_kb(2, 7) is None)
        check('a missing readout is None too',
              d.separation_kb(2, 99) is None)
        check('both chromosomes are reported',
              d.chromosomes == ['chr20', 'chr7'], str(d.chromosomes))
    finally:
        os.unlink(p)


def test_it_refuses_what_it_cannot_read():
    print('\nan unreadable design raises rather than returning nothing')
    empty = written('browser position: chr1:1-2\ntrack name="x"\n', '.txt')
    probebed = written('chr20\t100000\t100042\n' * 5, '_probe_bed.txt')
    try:
        for tag, pth in (('a header-only file', empty),
                         ('a per-probe BED with no names', probebed)):
            try:
                P.read_probe_design(pth)
                check('%s raises' % tag, False)
            except ValueError as e:
                check('%s raises, and says what a design is' % tag,
                      'REGION' in str(e) or 'region' in str(e))
        try:
            P.read_probe_design('nowhere.bed')
            check('a missing file raises', False)
        except FileNotFoundError:
            check('a missing file raises', True)
    finally:
        os.unlink(empty)
        os.unlink(probebed)


def test_the_real_designs_if_present():
    print('\nthe real designs on E:/Oligos, if this machine has them')
    real = [
        ('chr19', 'E:/Oligos/Library007/Chr19_downstream/trial7/'
                  'Chr19_downtrial7_bed.txt', 91, 200.0),
        ('RSF1', 'E:/Oligos/Library008/YSH_Cowork_RSF1/'
                 'RSF1_FOXA2_2.09Mb_r30_v5/'
                 'RSF1RSF1_FOXA2_2.09Mb_r30_v5_bed.txt', 69, 30.0),
        ('MP58', 'E:/Oligos/Library004/CriGri-PICRH_MP58/MP58_4M_v7/'
                 'AllRegions_bed.txt', 80, 50.0),
    ]
    seen = 0
    for tag, pth, n, step in real:
        if not os.path.exists(pth):
            continue
        seen += 1
        d = P.read_probe_design(pth)
        check('%-6s %s' % (tag, d.summary()),
              len(d) == n and abs(d.step_kb() - step) < 1.5)
    if not seen:
        print('  --   none present, skipped')


def main():
    for t in (test_every_format_reads_the_same,
              test_separation_uses_coordinates_not_indices,
              test_a_cross_chromosome_pair_is_refused,
              test_it_refuses_what_it_cannot_read,
              test_the_real_designs_if_present):
        t()
    print('\n%d passed, %d failed' % (_n['pass'], _n['fail']))
    return 1 if _n['fail'] else 0


if __name__ == '__main__':
    sys.exit(main())
