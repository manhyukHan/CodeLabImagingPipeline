"""The probe design: which readout sits where on the genome.

NOTHING IN THE PIPELINE KNEW THIS. The experiment layout carries
FolderName, Readouts, DataType, HybNum and the channels, and that is
all; `genomic_resolution_kb` is one number per experiment, often blank,
and it is used as classifier context rather than as a coordinate. So a
readout's genomic position lived only in the probe design on `E:\\Oligos`
and never reached an analysis.

A SINGLE RESOLUTION IS NOT ENOUGH, which is why this reads the design
rather than multiplying an index. chr19's downstream panel covers TWO
regions -- chr19:40000000-54716295 and chr19:54864547-58617616 -- so
every pair spanning that 148 kb gap would be misstated by
|delta readout| * 200 kb. Designs also skip readouts that failed
specificity, and a skipped one is a gap of two steps, not one.

THREE FORMATS, ALL CARRYING THE SAME THING, because a design folder
holds whichever the person kept:

    BED             chr20  21532000  21562000  RSF1_001  1  +  ...
    LibRegions fa   >chr20:21532000-21562000 gene=RSF1_001
    probe fa        >A02F__read001__RSF1_001_chr20:21532000-21562000_p0001__A10R

The format is sniffed from the content, not the extension: the same
folder names a `_bed.txt` and a `_probe_bed.txt` that differ in meaning
(regions against individual probes), and an extension cannot tell them
apart.

THE READOUT ID IS THE NAME'S TRAILING NUMBER (`RSF1_001` -> 1,
`CHO_initial_MP58_017` -> 17), which is the `rindex` the layout records.
The probe FASTA's own `read###` token is NOT used: it counts probes
within the library, so a panel that skips or renumbers readouts has the
two disagree, and nothing in the header says which one a reader took.

A DESIGN NEED NOT START AT 1 AND NEED NOT BE DENSE. Ids are read as
they are written rather than renumbered, because a design that dropped
a readout for specificity leaves a gap that IS two steps wide.

A DESIGN IS A REGION TABLE, AND A REGION HAS WIDTH. The separation
between two readouts is taken between MIDPOINTS, so a 200 kb step reads
200 kb rather than 200 kb plus or minus a region width depending on
which ends are compared.
"""
import os
import re

#: the coordinate part alone -- `:21532000-21562000`. The chromosome is
#: read BACKWARDS from it rather than matched in one pattern: a probe
#: header is `A02F__read001__RSF1_001_chr20:21532000-21562000_p0001`,
#: and any name-shaped alternation greedy enough to allow `NC_048598.1`
#: also swallows `A02F__read001__RSF1_001` and leaves no name behind.
_SPAN = re.compile(r':(\d+)-(\d+)')
#: a chromosome token at the END of whatever precedes the coordinate
_CHROM = re.compile(r'((?:chr|Chr|CHR|NC_|NW_|NT_|scaffold|contig)'
                    r'[0-9A-Za-z_.]*)$')
#: the trailing number of a region name, which is the readout id
_TAIL = re.compile(r'(\d+)\s*$')


class ProbeDesign:
    """readout id -> (chromosome, start, end), and separations in kb.

    `by_id` is the whole table; `missing` names ids the design does not
    carry, which a caller should report rather than silently treat as a
    zero separation.
    """

    def __init__(self, by_id, source=''):
        self.by_id = dict(by_id)
        self.source = str(source)

    def __len__(self):
        return len(self.by_id)

    def __contains__(self, rid):
        return int(rid) in self.by_id

    @property
    def chromosomes(self):
        return sorted({c for c, _s, _e in self.by_id.values()})

    def midpoint_kb(self, rid):
        """Midpoint in kb, or None when the design does not carry it."""
        v = self.by_id.get(int(rid))
        return None if v is None else (v[1] + v[2]) / 2000.0

    def separation_kb(self, a, b):
        """|midpoint(a) - midpoint(b)| in kb.

        None when either readout is absent OR the two sit on different
        chromosomes: a cross-chromosome separation is not a contour
        length and must not be fed to a chain model as though it were.
        """
        va, vb = self.by_id.get(int(a)), self.by_id.get(int(b))
        if va is None or vb is None or va[0] != vb[0]:
            return None
        return abs((va[1] + va[2]) - (vb[1] + vb[2])) / 2000.0

    def step_kb(self):
        """The design's own median step between CONSECUTIVE ids.

        Reported rather than assumed: it is the number a config's
        `genomic_resolution_kb` claims, and comparing the two is how a
        design pointed at the wrong panel announces itself.

        THE MEDIAN, NOT A MIDDLE INDEX. A panel that spans two regions
        contributes one enormous gap among its regular steps, and
        `d[len(d) // 2]` on an even-length list returns the upper of the
        two middle values -- which on a short design is that very gap.
        Cross-chromosome neighbours drop out, since their separation is
        not a contour length.
        """
        ids = sorted(self.by_id)
        d = sorted(v for v in (self.separation_kb(x, y)
                               for x, y in zip(ids, ids[1:])) if v)
        if not d:
            return None
        n = len(d)
        return d[n // 2] if n % 2 else 0.5 * (d[n // 2 - 1] + d[n // 2])

    def summary(self):
        s = self.step_kb()
        return ('%d readout(s), %s, step %s kb, ids %s-%s'
                % (len(self.by_id), '/'.join(self.chromosomes[:3]),
                   ('%.0f' % s) if s else '?',
                   min(self.by_id) if self.by_id else '-',
                   max(self.by_id) if self.by_id else '-'))


def _from_bed(lines):
    out = {}
    for line in lines:
        p = line.rstrip('\n').split('\t')
        if len(p) < 4 or not p[1].isdigit() or not p[2].isdigit():
            continue
        m = _TAIL.search(p[3])
        if m:
            out[int(m.group(1))] = (p[0], int(p[1]), int(p[2]))
    return out


def _from_fasta(lines):
    out = {}
    for line in lines:
        if not line.startswith('>'):
            continue
        h = line[1:].strip()
        m = _SPAN.search(h)
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        pre = h[:m.start()]
        cm = _CHROM.search(pre)
        chrom = cm.group(1) if cm else pre.rsplit('_', 1)[-1]
        # `gene=RSF1_001` when the header names one, otherwise whatever
        # sits before the chromosome token (`..._RSF1_001_chr20:...`)
        g = re.search(r'gene=([^\s]+)', h)
        name = (g.group(1) if g
                else (pre[:cm.start()] if cm else pre).rstrip('_'))
        t = _TAIL.search(name)
        if t:
            out.setdefault(int(t.group(1)), (chrom, a, b))
    return out


def read_probe_design(path):
    """Read a BED or FASTA probe design into a ProbeDesign.

    Sniffs the content: a line starting with '>' means FASTA, otherwise
    the tab-separated BED reading is tried. Raises ValueError when
    neither yields a readout, because an empty design silently turns
    every genomic separation into None and would look like a store with
    no usable pairs.
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError('no probe design at %r' % (path,))
    with open(path, encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    head = ''.join(lines[:200])
    by_id = _from_fasta(lines) if '>' in head else _from_bed(lines)
    if not by_id:
        # a probe BED (one row per probe) has no region names, and a
        # browser/track header alone parses to nothing -- say which
        raise ValueError(
            '%s carries no readout regions. A design is the REGION table '
            '(`*_bed.txt`, `*_LibRegions.fasta`), not the per-probe one '
            '(`*_probe_bed.txt`).' % path)
    return ProbeDesign(by_id, path)
