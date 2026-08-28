"""
The generalized Condition: a multi-input cell gate.

Semantics taken from CellClassifier's analysis panel, with its measured
defects designed out:
  - a Condition is an ordered AND of predicates, each evaluating to a
    boolean mask over Population.cells -- gates NARROW the cell set,
    figures multiply through FLAGS elsewhere, never here;
  - every predicate carries its OWN source scope (the reference coupled
    three different gates to one channel-list intersection);
  - evaluation is vectorized over the population's tidy tables, never a
    per-cell Python loop over objects;
  - a gate is data: to_dict()/from_dict() round-trip, so every figure
    can record exactly the gate that produced it, and a saved gate can
    be re-run headless.

Missing stays honest: a cell with no value for a gated quantity FAILS a
range predicate (it is not known to be in range) but is reported in the
per-predicate summary, never silently dropped from counts.
"""
import numpy as np
import pandas as pd

_REGISTRY = {}


def _register(cls):
    _REGISTRY[cls.kind] = cls
    return cls


class Predicate:
    kind = ''

    def mask(self, pop):
        raise NotImplementedError

    def to_dict(self):
        d = {'kind': self.kind}
        d.update(self._params())
        return d

    def _params(self):
        return {}

    @staticmethod
    def from_dict(d):
        cls = _REGISTRY[d['kind']]
        p = dict(d)
        p.pop('kind')
        return cls(**p)

    def __repr__(self):
        inner = ', '.join(f'{k}={v!r}' for k, v in self._params().items())
        return f'{type(self).__name__}({inner})'


@_register
class CelltypeIn(Predicate):
    kind = 'celltype_in'

    def __init__(self, names):
        self.names = list(names)

    def _params(self):
        return {'names': self.names}

    def mask(self, pop):
        return pop.cells['celltype'].isin(self.names).to_numpy()


@_register
class FovIn(Predicate):
    kind = 'fov_in'

    def __init__(self, fovs):
        self.fovs = [int(f) for f in fovs]

    def _params(self):
        return {'fovs': self.fovs}

    def mask(self, pop):
        return pop.cells['fov'].isin(self.fovs).to_numpy()


@_register
class ExpressionRange(Predicate):
    """metric of `source` within [lo, hi], optionally normalized.

    source: (modality, hybe, channel). metric: 'n_spots' |
    'brightness_median' | 'brightness_total' | 'mask_median'.
    normalize: None | ('by_source', ref_source) | ('by_total_count',).
    Open bounds via lo=None / hi=None.
    """
    kind = 'expression_range'

    def __init__(self, source, metric, lo=None, hi=None, normalize=None):
        self.source = tuple(source)
        self.metric = str(metric)
        self.lo, self.hi = lo, hi
        self.normalize = tuple(normalize) if normalize else None

    def _params(self):
        return {'source': list(self.source), 'metric': self.metric,
                'lo': self.lo, 'hi': self.hi,
                'normalize': list(self.normalize) if self.normalize else None}

    def values(self, pop):
        """(n_cells,) float, NaN where the cell has no value -- exposed
        so the histogram picker and the gate share ONE definition."""
        from codelab_pipeline.analysis import expression as E
        t = pop.expression
        if t is None or len(t) == 0:
            raise ValueError('population carries no expression table; build '
                             'with sources=[...] first')
        col = self.metric
        if self.normalize:
            mode = self.normalize[0]
            ref = tuple(self.normalize[1]) if len(self.normalize) > 1 else None
            t = E.normalize(t, self.metric, mode, ref_source=ref)
            col = f'{self.metric}_norm'
        m, h, ch = self.source
        rows = t[(t['modality'] == m) & (t['hybe'] == h)
                 & (t['channel'] == int(ch))]
        by_cell = rows.set_index(['fov', 'cell'])[col]
        idx = pd.MultiIndex.from_frame(pop.cells[['fov', 'cell']])
        return by_cell.reindex(idx).to_numpy(dtype=float)

    def mask(self, pop):
        v = self.values(pop)
        ok = np.isfinite(v)
        if self.lo is not None:
            ok &= v >= float(self.lo)
        if self.hi is not None:
            ok &= v <= float(self.hi)
        return ok


@_register
class PairDistanceRange(Predicate):
    """Per-cell collapsed distance between two spot sets within [lo, hi] um.

    source_a / source_b: (modality, hybe, channel). Per cell, every
    cross-set spot pair's 3D um distance is collapsed by MEDIAN (never
    min -- the zero-bounded noise floor), and the collapsed value is
    gated. Cells missing either set fail.
    """
    kind = 'pair_distance_range'

    def __init__(self, source_a, source_b, lo=None, hi=None,
                 collapse='median'):
        self.source_a, self.source_b = tuple(source_a), tuple(source_b)
        self.lo, self.hi = lo, hi
        self.collapse = collapse

    def _params(self):
        return {'source_a': list(self.source_a),
                'source_b': list(self.source_b),
                'lo': self.lo, 'hi': self.hi, 'collapse': self.collapse}

    def values(self, pop):
        from codelab_pipeline.analysis import distances as D
        per_cell = D.pair_distance_per_cell(pop, self.source_a, self.source_b,
                                            collapse=self.collapse)
        idx = pd.MultiIndex.from_frame(pop.cells[['fov', 'cell']])
        return per_cell.reindex(idx).to_numpy(dtype=float)

    def mask(self, pop):
        v = self.values(pop)
        ok = np.isfinite(v)
        if self.lo is not None:
            ok &= v >= float(self.lo)
        if self.hi is not None:
            ok &= v <= float(self.hi)
        return ok


@_register
class AlleleCount(Predicate):
    """Cells holding between lo and hi TRACED alleles (n_traced >= min_bins)."""
    kind = 'allele_count'

    def __init__(self, lo=1, hi=None, min_bins=2):
        self.lo, self.hi, self.min_bins = lo, hi, int(min_bins)

    def _params(self):
        return {'lo': self.lo, 'hi': self.hi, 'min_bins': self.min_bins}

    def mask(self, pop):
        al = pop.alleles
        counts = np.zeros(len(pop.cells), dtype=int)
        if al is not None and len(al['cell']):
            traced = al['n_traced'] >= self.min_bins
            df = pd.DataFrame({'fov': al['fov'][traced],
                               'cell': al['cell'][traced]})
            df = df[df['cell'] >= 0]
            got = df.value_counts(['fov', 'cell'])
            idx = pd.MultiIndex.from_frame(pop.cells[['fov', 'cell']])
            counts = got.reindex(idx, fill_value=0).to_numpy()
        ok = counts >= int(self.lo)
        if self.hi is not None:
            ok &= counts <= int(self.hi)
        return ok


@_register
class BarcodePresence(Predicate):
    """ALLELE-LEVEL, per explicit decision: the genetic modification can
    be heterogeneous, so presence/absence of a barcode distinguishes the
    modified from the unmodified HOMOLOG within one cell -- a cell-level
    verdict would erase exactly that distinction.

    hybes: bins that must ALL be present (finite traced position) in the
    allele; absent=True inverts (all must be MISSING). allele_mask() is
    the primary product; the cell-level mask() says "the cell holds at
    least one qualifying allele", so cell gating still composes.
    """
    kind = 'barcode_presence'
    level = 'allele'

    def __init__(self, hybes, absent=False):
        self.hybes = list(hybes)
        self.absent = bool(absent)

    def _params(self):
        return {'hybes': self.hybes, 'absent': self.absent}

    def allele_mask(self, pop):
        al = pop.alleles
        if al is None:
            raise ValueError('population carries no alleles')
        bins = list(al['bin_hybes'])
        idxs = []
        for h in self.hybes:
            if h not in bins:
                raise ValueError(f'{h!r} is not a genomic bin '
                                 f'(bins: {bins[:5]}...)')
            idxs.append(bins.index(h))
        present = np.isfinite(al['pos_um'][:, idxs, 0])
        return (~present).all(1) if self.absent else present.all(1)

    def mask(self, pop):
        amask = self.allele_mask(pop)
        al = pop.alleles
        keep = pd.DataFrame({'fov': al['fov'][amask],
                             'cell': al['cell'][amask]})
        keep = keep[keep['cell'] >= 0].drop_duplicates()
        idx = pd.MultiIndex.from_frame(pop.cells[['fov', 'cell']])
        got = pd.Series(True, index=pd.MultiIndex.from_frame(keep))
        return got.reindex(idx, fill_value=False).to_numpy()


class Condition:
    """OR of AND-clauses over predicates -> boolean masks.

    Condition([p1, p2]) is one AND clause (the common case);
    Condition(clauses=[[p1, p2], [p3]]) is (p1 AND p2) OR (p3) -- the
    composition the founding spec names: additive within a category AND
    across categories, with OR between clause groups.

    Two mask levels, per explicit decision. mask() gates CELLS.
    allele_mask() gates ALLELES: cell-level predicates project through
    the cell (an allele survives iff its cell does), allele-level
    predicates (BarcodePresence) apply directly -- so a heterozygous
    cell can pass while only its modified allele feeds the map.
    """

    def __init__(self, predicates=(), clauses=None):
        if clauses is not None:
            self.clauses = [list(c) for c in clauses]
        else:
            self.clauses = [list(predicates)] if predicates else []

    @property
    def predicates(self):
        """Flat view over all clauses -- for reporting and GUI listing."""
        return [p for c in self.clauses for p in c]

    def mask(self, pop):
        if not self.clauses:
            return np.ones(len(pop.cells), dtype=bool)
        out = np.zeros(len(pop.cells), dtype=bool)
        for clause in self.clauses:
            m = np.ones(len(pop.cells), dtype=bool)
            for p in clause:
                m &= np.asarray(p.mask(pop), bool)
            out |= m
        return out

    def allele_mask(self, pop):
        """Allele-level gate: OR over clauses of (cell-projection AND
        the clause's own allele-level predicates)."""
        al = pop.alleles
        if al is None:
            raise ValueError('population carries no alleles')
        n_al = len(al['cell'])
        if not self.clauses:
            return np.ones(n_al, dtype=bool)
        out = np.zeros(n_al, dtype=bool)
        for clause in self.clauses:
            cm = np.ones(len(pop.cells), dtype=bool)
            am = np.ones(n_al, dtype=bool)
            for p in clause:
                if getattr(p, 'level', 'cell') == 'allele':
                    am &= np.asarray(p.allele_mask(pop), bool)
                else:
                    cm &= np.asarray(p.mask(pop), bool)
            out |= am & pop.allele_mask_from_cells(cm)
        return out

    def report(self, pop):
        """[(repr, n_after)] per clause sequentially, clause boundaries
        marked with ('-- OR --', running total)."""
        out = []
        total = np.zeros(len(pop.cells), dtype=bool)
        for ci, clause in enumerate(self.clauses):
            if ci:
                out.append(('-- OR --', int(total.sum())))
            m = np.ones(len(pop.cells), dtype=bool)
            for p in clause:
                m &= np.asarray(p.mask(pop), bool)
                out.append((repr(p), int(m.sum())))
            total |= m
        return out

    def to_dict(self):
        return {'clauses': [[p.to_dict() for p in c] for c in self.clauses]}

    @staticmethod
    def from_dict(d):
        if 'clauses' in d:
            return Condition(clauses=[[Predicate.from_dict(x) for x in c]
                                      for c in d['clauses']])
        return Condition([Predicate.from_dict(x)
                          for x in d.get('predicates', [])])
