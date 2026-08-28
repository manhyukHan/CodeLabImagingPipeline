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


class Condition:
    """Ordered AND of predicates -> one boolean cell mask."""

    def __init__(self, predicates=()):
        self.predicates = list(predicates)

    def mask(self, pop):
        m = np.ones(len(pop.cells), dtype=bool)
        for p in self.predicates:
            m &= np.asarray(p.mask(pop), bool)
        return m

    def report(self, pop):
        """[(repr, n_surviving_after)] -- the CellClassifier summary line,
        as data. The sequential counts show which predicate narrowed."""
        out = []
        m = np.ones(len(pop.cells), dtype=bool)
        for p in self.predicates:
            m &= np.asarray(p.mask(pop), bool)
            out.append((repr(p), int(m.sum())))
        return out

    def to_dict(self):
        return {'predicates': [p.to_dict() for p in self.predicates]}

    @staticmethod
    def from_dict(d):
        return Condition([Predicate.from_dict(x)
                          for x in d.get('predicates', [])])
