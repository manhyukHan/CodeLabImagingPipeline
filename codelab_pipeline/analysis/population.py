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

_EXPR_BASE_COLS = ('cell', 'celltype', 'n_spots', 'brightness_median',
                   'brightness_total')
_EXPR_MASK_COLS = ('mask_median', 'mask_frame')


def _skey(src):
    return f'{src[0]}|{src[1]}|{int(src[2])}'


def _fov_bundle(item):
    """One FOV's full extraction -- module-level for pmap."""
    (storage_path, fov, hybes, sources, spot_sources, voxel_um,
     mask_intensity, resolver, qc_round_hybes, overwrite) = item
    out = {'fov': int(fov)}
    cells, _ = analysis_store.read_cells(storage_path, fov)
    out['cells'] = [{'fov': int(fov), 'cell': int(c['id']),
                     'celltype': str(c.get('celltype') or '')}
                    for c in (cells or [])]
    if hybes:
        out['alleles'] = P.fov_polymer_table(storage_path, fov, hybes,
                                             voxel_um=voxel_um)
        # repeat/toe rounds: same alleles, same read order, other hybes
        for key, rounds in (qc_round_hybes or {}).items():
            if rounds:
                aux = P.fov_polymer_table(
                    storage_path, fov, [h for _rid, h in rounds],
                    voxel_um=voxel_um)
                out['alleles'][f'{key}_pos_um'] = aux['pos_um']
    if sources:
        # THE APPEND-MODE CACHE (item 3): computed cell attributes are
        # the NAS-expensive part of a build (spot-slice reads, mask
        # medians over MIPs); once computed for a source they persist in
        # the FOV's expression.json capsule and a rebuild reuses them --
        # skip-already-built unless overwrite, per explicit request.
        # Celltype is NOT trusted from the cache (assignments change
        # after re-classification); it refreshes from the cells read
        # above on every load.
        cache = {} if overwrite else \
            (analysis_store.read_fov_expression(storage_path, fov) or {})
        csources = dict(cache.get('sources') or {})
        cagg = dict(cache.get('agg_by_mod') or {})
        celltype_of = {r['cell']: r['celltype'] for r in out['cells']}
        cached_rows, fresh_srcs, homeless = [], [], {}
        for src in sources:
            ent = csources.get(_skey(src))
            # a cached entry without mask columns cannot serve a
            # mask_intensity build -- recompute that source
            if ent is not None and (not mask_intensity or ent.get('mask')):
                m, h, ch = src
                for r in ent['rows']:
                    row = dict(r)
                    row.update({'fov': int(fov), 'modality': m,
                                'hybe': h, 'channel': int(ch),
                                'celltype': celltype_of.get(
                                    int(r['cell']), r.get('celltype', ''))})
                    cached_rows.append(row)
                homeless[src] = int(ent.get('homeless', 0))
            else:
                fresh_srcs.append(src)
        # item 11's denominators: per-cell aggregates over ALL of a
        # modality's spots in the STORE (not just the checked sources) --
        # what by_modality normalization divides by. One whole-FOV read
        # per modality when not cached; joined onto every expression row
        # of that modality below.
        agg_by_mod, fresh_mods = {}, []
        for m in sorted({src[0] for src in sources}):
            if m in cagg:
                agg_by_mod[m] = {int(k): v for k, v in cagg[m].items()}
                continue
            fresh_mods.append(m)
            allspots = analysis_store.read_spots(storage_path, fov, modality=m)
            per = {}
            for s_ in allspots:
                cid = int(s_.get('cell', -1))
                if cid < 0:
                    continue
                per.setdefault(cid, []).append(
                    float(s_.get('brightness', np.nan)))
            agg_by_mod[m] = {
                cid: {'mod_n_spots': len(v),
                      'mod_brightness_median': float(np.nanmedian(v)) if v else np.nan,
                      'mod_brightness_total': float(np.nansum(v)) if v else 0.0}
                for cid, v in per.items()}
        table_new = None
        if fresh_srcs:
            if mask_intensity and resolver is None:
                # exact projection BY DEFAULT, headless: the store
                # carries everything the resolver needs
                # (analysis/resolvers.py); a store with no alignment yet
                # falls back to the flagged reference-frame mask
                from codelab_pipeline.analysis import resolvers as R
                try:
                    resolver = R.resolver_for(storage_path, fov)
                except Exception:
                    resolver = None
            table_new, extra = E.fov_expression_table(
                storage_path, fov, fresh_srcs,
                mask_intensity=mask_intensity, resolver=resolver)
            homeless.update(extra['homeless'])
        frames = [f for f in (
            pd.DataFrame(cached_rows) if cached_rows else None,
            table_new if table_new is not None and len(table_new) else None)
            if f is not None]
        # NO empty frame (same trap as spots below): a rows-less
        # DataFrame is all-object and one such FOV poisons the numeric
        # dtypes of the whole concatenated population table
        if frames:
            table = pd.concat(frames, ignore_index=True)
            for col in ('mod_n_spots', 'mod_brightness_median',
                        'mod_brightness_total'):
                table[col] = [
                    agg_by_mod.get(r_mod, {}).get(int(r_cell), {}).get(col, np.nan)
                    for r_mod, r_cell in zip(table['modality'], table['cell'])]
            if mask_intensity and 'mask_median' in table.columns:
                # median over the population's mask sources of that modality
                mm = table.groupby(['modality', 'cell'])['mask_median']                     .median().rename('mod_mask_median')
                table = table.merge(mm, on=['modality', 'cell'], how='left')
            out['expression'] = table
        out['homeless'] = homeless
        out['expr_cache'] = {'cached': len(sources) - len(fresh_srcs),
                             'computed': len(fresh_srcs)}
        if fresh_srcs or fresh_mods:
            # merge the fresh work into the capsule and full-replace it
            # through the atomic door; mod_* stays OUT of the per-source
            # rows (it is joined at load, so a later build with more
            # sources never bakes in a stale denominator)
            if table_new is not None:
                keep = [c for c in _EXPR_BASE_COLS + _EXPR_MASK_COLS
                        if c in table_new.columns]
                for src in fresh_srcs:
                    m, h, ch = src
                    if len(table_new):
                        sub = table_new[(table_new['modality'] == m)
                                        & (table_new['hybe'] == h)
                                        & (table_new['channel'] == int(ch))]
                        rows = sub[keep].to_dict('records')
                    else:
                        # a FOV with no segmented cells computes to zero
                        # rows -- still a result worth caching (its spot
                        # slices were read; homeless got counted). The
                        # empty frame has NO columns, so slicing it
                        # KeyErrors -- the bug that failed 37/41 FOVs on
                        # the first store-wide build.
                        rows = []
                    csources[_skey(src)] = {
                        'mask': bool(mask_intensity),
                        'homeless': int(homeless.get(src, 0)),
                        'rows': rows}
            for m in fresh_mods:
                cagg[m] = {str(k): v for k, v in agg_by_mod[m].items()}
            analysis_store.write_fov_expression(
                storage_path, fov,
                {'version': 1, 'sources': csources, 'agg_by_mod': cagg})
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
        if rows:
            out['spots'] = pd.DataFrame(rows)
        # NO empty frame: a rows-less DataFrame has all-object columns,
        # and one such FOV in the concat downgrades y_um/x_um/z_um to
        # object for the WHOLE population -- np.sqrt on an object array
        # then dies with the no-callable-sqrt ufunc error (reported from
        # the GUI, reproduced, and pinned below by dtype enforcement).
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
        self.cache_stats = None
        self._dmaps = None

    @classmethod
    def build(cls, storage_path, fovs, records=None, hybes=None,
              sources=None, spot_sources=None, voxel_um=DEFAULT_VOXEL_UM,
              mask_intensity=False, resolvers=None, jobs=None,
              on_done=None, overwrite_cache=False):
        """Assemble a Population from the store, headless.

        records: parsed layout records (preprocess.parse_experiment_layout)
        -- or pass hybes directly as the ordered genomic-bin list. sources
        / spot_sources: [(modality, hybe, channel)] for expression / spot
        tables; either may be omitted and the matching predicates then
        refuse with an actionable error rather than a KeyError.

        Computed cell attributes persist per FOV (expression.json beside
        the other capsules) and a rebuild REUSES them per source --
        append mode, so adding a source computes only that source.
        overwrite_cache=True recomputes and rewrites everything
        requested (use after re-detection or re-alignment: the cache
        cannot see that its inputs changed).

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
        qc_round_hybes = None
        bin_ids = None
        if records is not None:
            if hybes is None:
                hybes = P.bin_hybes(records)
            bin_ids = [rid for rid, _h in P.genomic_bins(records)]
            qc_round_hybes = {'repeat': P.qc_rounds(records)['repeats'],
                              'toe': P.qc_rounds(records)['toes']}
        items = [(storage_path, int(f), hybes, sources, spot_sources,
                  tuple(voxel_um), bool(mask_intensity),
                  (resolvers or {}).get(int(f)), qc_round_hybes,
                  bool(overwrite_cache))
                 for f in fovs]
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
                'bin_ids': list(bin_ids or []),
            }
            for key, rounds in (qc_round_hybes or {}).items():
                if rounds and all(f'{key}_pos_um' in p for p in parts):
                    alleles[f'{key}_pos_um'] = np.concatenate(
                        [p[f'{key}_pos_um'] for p in parts])
                    alleles[f'{key}_ids'] = [rid for rid, _h in rounds]
        expr = [b['expression'] for b in bundles if 'expression' in b]
        expression = pd.concat(expr, ignore_index=True) if expr else None
        cache_stats = None
        cs = [b['expr_cache'] for b in bundles if 'expr_cache' in b]
        if cs:
            cache_stats = {'cached': sum(c['cached'] for c in cs),
                           'computed': sum(c['computed'] for c in cs)}
        sp = [b['spots'] for b in bundles if 'spots' in b]
        spots = pd.concat(sp, ignore_index=True) if sp else None
        if spots is not None:
            # belt for the empty-frame suspenders: numeric stays numeric
            spots = spots.astype({'fov': 'int64', 'cell': 'int64',
                                  'channel': 'int64', 'y_um': 'float64',
                                  'x_um': 'float64', 'z_um': 'float64',
                                  'brightness': 'float64'})
        pop = cls(storage_path, fovs, voxel_um, cells, alleles,
                  expression, spots, fails)
        pop.cache_stats = cache_stats
        return pop

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
            note = ''
            if self.cache_stats:
                note = (f" ({self.cache_stats['cached']} src cached, "
                        f"{self.cache_stats['computed']} computed)")
            parts.append(f'{len(self.expression)} expression rows{note}')
        if self.spots is not None:
            parts.append(f'{len(self.spots)} spots')
        if self.failures:
            # readable, not exhaustive: 37 stacked tracebacks once made
            # the status label a wall -- name the FOVs, show one message
            fovs = [f for f, _m in self.failures]
            parts.append(f'FAILED FOVs {fovs}: {self.failures[0][1]}'
                         + (' (first of '
                            f'{len(self.failures)} messages)'
                            if len(self.failures) > 1 else ''))
        return ' | '.join(parts) + f' | celltypes: {dict(n_ct)}'
