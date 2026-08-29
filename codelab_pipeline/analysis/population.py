"""
Population: everything the gates and analyzers consume, assembled once.

Built FOV-major through ONE pmap pool (codelab_pipeline/parallel.py) --
the extractors are module-level picklable functions reading HDF5 in
CHILD processes, per the measured h5py process-lock rule. The result is
plain tables:

  cells       DataFrame [fov, cell, celltype]      the mask axis: every
                                                   boolean cell gate is
                                                   over THIS row order
  alleles     dict of stacked arrays               fov_polymer_table
              (+ 'bin_hybes')                      concatenated
  expression  DataFrame (see expression.py)        None unless sources
  spots       DataFrame [fov, cell, celltype,      None unless
              modality, hybe, channel,             spot_sources
              y_um, x_um, z_um, brightness]

Keys are ALWAYS (fov, cell) pairs. The SG scripts carry a measured scar
here: cells keyed by basename collided across FOVs and silently merged
2172 cells into 198 groups. Bare cell ids do not exist in this package.
"""
import numpy as np
import pandas as pd

from codelab_pipeline import parallel
from codelab_pipeline.io import analysis_store
from codelab_pipeline.analysis import polymer as P
from codelab_pipeline.analysis import expression as E

DEFAULT_VOXEL_UM = (0.208, 0.208, 0.2)


def _fov_bundle(item):
    """One FOV's full extraction -- module-level for pmap."""
    (storage_path, fov, hybes, sources, spot_sources, voxel_um,
     mask_intensity, resolver) = item
    out = {'fov': int(fov)}
    cells, _ = analysis_store.read_cells(storage_path, fov)
    out['cells'] = [{'fov': int(fov), 'cell': int(c['id']),
                     'celltype': str(c.get('celltype') or '')}
                    for c in (cells or [])]
    if hybes:
        out['alleles'] = P.fov_polymer_table(storage_path, fov, hybes,
                                             voxel_um=voxel_um)
    if sources:
        table, extra = E.fov_expression_table(storage_path, fov, sources,
                                              mask_intensity=mask_intensity,
                                              resolver=resolver)
        out['expression'] = table
        out['homeless'] = extra['homeless']
    if spot_sources:
        dy, dx, dz = (float(v) for v in voxel_um)
        rows = []
        for src in spot_sources:
            m, h, ch = src
            for s in analysis_store.read_spots(storage_path, fov, modality=m,
                                               hybe=h, channel=ch):
                y, x, z = s['adj_coordinate']
                rows.append({'fov': int(fov), 'cell': int(s.get('cell', -1)),
                             'celltype': str(s.get('celltype') or ''),
                             'modality': m, 'hybe': h, 'channel': int(ch),
                             'y_um': float(y) * dy, 'x_um': float(x) * dx,
                             'z_um': float(z) * dz,
                             'brightness': float(s.get('brightness', np.nan))})
        out['spots'] = pd.DataFrame(
            rows, columns=['fov', 'cell', 'celltype', 'modality', 'hybe',
                           'channel', 'y_um', 'x_um', 'z_um', 'brightness'])
    return out


class Population:
    def __init__(self, storage_path, fovs, voxel_um, cells, alleles,
                 expression, spots, failures):
        self.storage_path = storage_path
        self.fovs = list(fovs)
        self.voxel_um = tuple(voxel_um)
        self.cells = cells
        self.alleles = alleles
        self.expression = expression
        self.spots = spots
        self.failures = failures
        self._dmaps = None

    @classmethod
    def build(cls, storage_path, fovs, records=None, hybes=None,
              sources=None, spot_sources=None, voxel_um=DEFAULT_VOXEL_UM,
              mask_intensity=False, resolvers=None, jobs=None,
              on_done=None):
        """Assemble a Population from the store, headless.

        records: parsed layout records (preprocess.parse_experiment_layout)
        -- or pass hybes directly as the ordered genomic-bin list. sources
        / spot_sources: [(modality, hybe, channel)] for expression / spot
        tables; either may be omitted and the matching predicates then
        refuse with an actionable error rather than a KeyError.

        resolvers: {fov: FrameResolver} for EXACT mask projection into
        each source hybe's own raw frame (mask_frame='native'). The
        resolver is plain-data by design and pickles into the child
        processes; without one, post-cell-alignment cells fall to the
        FLAGGED reference-frame mask -- which, for a cross-modal source,
        is off by the whole cross-modal bridge (~13 px measured), not a
        rounding error. The app passes MainWindow._frame_resolver(None,
        fov) per FOV; cell-level residuals still apply because
        transform(..., cell=cell) takes the cell at call time.
        """
        fovs = [int(f) for f in fovs]
        if len(set(fovs)) != len(fovs):
            # a duplicated FOV silently duplicates every (fov, cell) key
            # and every downstream join then fabricates signal
            raise ValueError(f'duplicate FOVs in {fovs}')
        if hybes is None and records is not None:
            hybes = P.bin_hybes(records)
        items = [(storage_path, int(f), hybes, sources, spot_sources,
                  tuple(voxel_um), bool(mask_intensity),
                  (resolvers or {}).get(int(f))) for f in fovs]
        results = parallel.pmap(_fov_bundle, items, kind='io', jobs=jobs,
                                on_done=on_done)
        bundles, fails = [], []
        for f, r in zip(fovs, results):
            if isinstance(r, parallel.Failure):
                fails.append((int(f), str(r)))
            else:
                bundles.append(r)
        cells = pd.DataFrame(
            [row for b in bundles for row in b['cells']],
            columns=['fov', 'cell', 'celltype'])
        alleles = None
        parts = [b['alleles'] for b in bundles if 'alleles' in b]
        if parts:
            alleles = {
                'pos_um': np.concatenate([p['pos_um'] for p in parts]),
                'amp': np.concatenate([p['amp'] for p in parts]),
                'n_cand': np.concatenate([p['n_cand'] for p in parts]),
                'allele_id': np.concatenate([p['allele_id'] for p in parts]),
                'cell': np.concatenate([p['cell'] for p in parts]),
                'fov': np.concatenate([p['fov'] for p in parts]),
                'celltype': sum((p['celltype'] for p in parts), []),
                'n_traced': np.concatenate([p['n_traced'] for p in parts]),
                'bin_hybes': list(hybes or []),
            }
        expr = [b['expression'] for b in bundles if 'expression' in b]
        expression = pd.concat(expr, ignore_index=True) if expr else None
        sp = [b['spots'] for b in bundles if 'spots' in b]
        spots = pd.concat(sp, ignore_index=True) if sp else None
        return cls(storage_path, fovs, voxel_um, cells, alleles,
                   expression, spots, fails)

    def dmaps(self):
        """(n_alleles, n_bins, n_bins) um distance maps, computed once."""
        if self.alleles is None:
            raise ValueError('population carries no alleles; build with '
                             'records=/hybes= first')
        if self._dmaps is None:
            self._dmaps = P.polymer_distmaps(self.alleles['pos_um'])
        return self._dmaps

    def allele_mask_from_cells(self, cell_mask):
        """Project a CELL gate onto allele rows: an allele survives iff
        its (fov, cell) survives. Homeless alleles never survive a cell
        gate -- they belong to no gated cell."""
        keep = self.cells.loc[np.asarray(cell_mask, bool), ['fov', 'cell']]
        keys = set(map(tuple, keep.itertuples(index=False)))
        al = self.alleles
        return np.array([(int(f), int(c)) in keys and c >= 0
                         for f, c in zip(al['fov'], al['cell'])])

    def summary(self):
        n_ct = self.cells['celltype'].replace('', 'Unassigned').value_counts()
        parts = [f'{len(self.cells)} cells over {len(self.fovs)} FOVs']
        if self.alleles is not None:
            parts.append(f"{len(self.alleles['cell'])} alleles "
                         f"({int((self.alleles['n_traced'] >= 2).sum())} traced)")
        if self.expression is not None:
            parts.append(f'{len(self.expression)} expression rows')
        if self.spots is not None:
            parts.append(f'{len(self.spots)} spots')
        if self.failures:
            parts.append(f'FAILED FOVs: {self.failures}')
        return ' | '.join(parts) + f' | celltypes: {dict(n_ct)}'
