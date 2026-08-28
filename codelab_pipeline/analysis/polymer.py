"""
Polymer construction from stored alleles, and the ORCA-style QC on it.

The collapse recipe is the one AnAllele.final_polymer has cited since the
field existed (QualityControlORCA.combineFOV, 'maxBrightness'): per
genomic bin keep the BRIGHTEST candidate among polymer_adj[hybe] -- the
amplitude is index 3 of every candidate tuple -- one (y, x, z) per bin,
NaN where the hybe was rejected or absent. Bin order comes from the
layout's readout_id over datatype 'H' records only; R/T/B rounds are QC
and identity markers, never polymer bins.

Units: positions leave this module in MICROMETRES, scaled per axis by
voxel_um exactly once. z is stored as plane index and pixels are not
planes (0.208 vs 0.2 um) -- the axes scale independently or the maps are
wrong anisotropically.

QC follows QualityControlORCA with its measured bugs fixed:
  - brightness bounds test the AMPLITUDE column only (the original
    compared all four components elementwise, so any coordinate below
    the 5th-percentile brightness was falsely flagged);
  - thresholds are quantile-derived from the data at hand
    (polymeric_qc), never inherited constants -- the repo rule.
"""
import os

import numpy as np

from codelab_pipeline.io import analysis_store
from codelab_pipeline.io import paths as store_paths
from codelab_pipeline.io import preprocess

DEFAULT_VOXEL_UM = (0.208, 0.208, 0.2)


def records_for(storage_path, modality=None):
    """Parse the experiment layout the STORE's own manifest names.

    The headless entry point: give the storage_path (the modality
    directory, e.g. <root>/DNA), get the parsed layout records -- no app,
    no config file. modality defaults to the storage_path's basename.
    Raises with the manifest content when the layout is not recorded or
    the file has moved (the layout lives on the acquisition share, which
    is not always mounted -- say so rather than guessing).
    """
    root = os.path.dirname(os.path.abspath(storage_path))
    manifest = store_paths.read_manifest(root)
    name = modality or os.path.basename(os.path.normpath(storage_path))
    entry = (manifest.get('modalities') or {}).get(name) or {}
    layout = entry.get('layout_path')
    if not layout:
        raise ValueError(f'manifest at {root} records no layout_path for '
                         f'modality {name!r}: {manifest!r}')
    if not os.path.exists(layout):
        raise FileNotFoundError(
            f'layout {layout} (from the manifest) is not reachable -- the '
            f'acquisition share may not be mounted on this machine')
    return preprocess.parse_experiment_layout(layout)


def genomic_bins(records):
    """[(readout_id, hybe_folder)] for datatype 'H', sorted by readout_id.

    The layout is authoritative for bin identity and order; readout_id is
    the genomic-locus axis every distance map shares. Returns a list of
    (readout_id, hybe) pairs; use bin_hybes() for the bare hybe list.
    """
    rows = [(int(r['readout_id']), str(r['folder'])) for r in records
            if str(r.get('datatype', '')).upper() == 'H']
    rows.sort()
    return rows


def bin_hybes(records):
    return [h for _rid, h in genomic_bins(records)]


def max_brightness(candidates):
    """The default Selector: the brightest candidate, whole.

    A Selector is any callable
        selector([(y, x, z, amplitude), ...]) -> (y, x, z, amplitude)
    called only on NON-EMPTY candidate lists. ORCA gated this choice
    behind a 'barcode_selection' string and never implemented a second
    rule; here the rule is an ordinary function argument, so swapping it
    (nearest-to-anchor, amplitude-weighted centroid, ...) is a one-line
    call-site change and never a code fork. A composite selector may
    return a synthetic amplitude (e.g. the sum); the QC brightness gates
    then act on whatever the selector reported.
    """
    return max(candidates, key=lambda c: c[3])


def collapse_polymer(allele_dict, hybes, selector=max_brightness):
    """One allele's polymer_adj -> (pos (n_bins, 3) yxz, amp (n_bins,),
    n_cand (n_bins,) int). NaN / 0 where the bin has no candidate.

    `selector` picks ONE position per multi-candidate bin (sister
    chromatids); see max_brightness for the contract. Positions stay in
    px/planes here; the caller scales to um once, so this stays
    unit-agnostic and testable.
    """
    n = len(hybes)
    pos = np.full((n, 3), np.nan)
    amp = np.full(n, np.nan)
    n_cand = np.zeros(n, dtype=np.int32)
    poly = allele_dict.get('polymer_adj') or {}
    for j, h in enumerate(hybes):
        cands = poly.get(h)
        if not cands:
            continue
        n_cand[j] = len(cands)
        y, x, z, a = selector(cands)
        pos[j] = (y, x, z)
        amp[j] = a
    return pos, amp, n_cand


def fov_polymer_table(storage_path, fov, hybes, voxel_um=DEFAULT_VOXEL_UM,
                      selector=max_brightness):
    """Every stored allele of one FOV as stacked arrays, um-scaled.

    Returns a plain dict (picklable, pmap-friendly):
      pos_um    (n_alleles, n_bins, 3) float  y, x, z in MICROMETRES
      amp       (n_alleles, n_bins)           brightest-candidate amplitude
      n_cand    (n_alleles, n_bins) int       candidates per bin (sister
                                              chromatids show up here)
      allele_id, cell, fov (n_alleles,) int
      celltype  list[str]                     joined via this FOV's cells;
                                              '' where the cell is unknown
                                              or the allele is homeless
      n_traced  (n_alleles,) int              finite bins per allele

    Alleles with zero traced bins are KEPT as all-NaN rows -- dropping is
    a QC decision (apply_qc), not an extraction side effect.
    """
    dicts = analysis_store.read_fov_alleles(storage_path, fov)
    cells, _ = analysis_store.read_cells(storage_path, fov)
    ct_by_id = {int(c['id']): str(c.get('celltype') or '')
                for c in (cells or [])}
    n_a, n_b = len(dicts), len(hybes)
    pos = np.full((n_a, n_b, 3), np.nan)
    amp = np.full((n_a, n_b), np.nan)
    n_cand = np.zeros((n_a, n_b), dtype=np.int32)
    aid = np.zeros(n_a, dtype=np.int64)
    cid = np.full(n_a, -1, dtype=np.int64)
    celltype = []
    for i, d in enumerate(dicts):
        p, a, k = collapse_polymer(d, hybes, selector=selector)
        pos[i], amp[i], n_cand[i] = p, a, k
        aid[i] = int(d['id'])
        cid[i] = int(d.get('cell', -1))
        celltype.append(ct_by_id.get(cid[i], ''))
    dy, dx, dz = (float(v) for v in voxel_um)
    pos *= np.array([dy, dx, dz])[None, None, :]
    return {'pos_um': pos, 'amp': amp, 'n_cand': n_cand,
            'allele_id': aid, 'cell': cid,
            'fov': np.full(n_a, int(fov), dtype=np.int64),
            'celltype': celltype,
            'n_traced': np.isfinite(pos[:, :, 0]).sum(1).astype(np.int32)}


def polymer_distmaps(pos_um):
    """(n_alleles, n_bins, n_bins) pairwise Euclidean um distances.

    NaN positions propagate to NaN rows/columns, exactly the ORCA kappa
    behaviour. Vectorized over alleles; ~25 MB for 127 alleles x 90 bins.
    """
    pos = np.asarray(pos_um, np.float32)
    n_a, n_b = pos.shape[0], pos.shape[1]
    out = np.empty((n_a, n_b, n_b), np.float32)
    # float32 and CHUNKED: cells run to tens of thousands, so alleles do
    # too. At 24k alleles x 62 bins the float64 all-at-once broadcast
    # peaked at ~1.1 GB of scratch for a 738 MB result; float32 halves
    # both and um distances do not need 15 significant digits. Measured:
    # 5.3 s / 738 MB -> chunked float32 keeps peak scratch ~50 MB.
    step = max(1, int(2e7 // max(n_b * n_b * 3, 1)))
    for i0 in range(0, n_a, step):
        blk = pos[i0:i0 + step]
        diff = blk[:, :, None, :] - blk[:, None, :, :]
        out[i0:i0 + step] = np.sqrt((diff ** 2).sum(-1))
    return out


# -- ORCA QC ---------------------------------------------------------------

def qc_thresholds(table, dmaps,
                  brightness_q=(0.05, 0.95), jump_q=0.75, max_dist_q=0.95):
    """Quantile-derived thresholds, polymeric_qc-style, from THIS data.

    Returns {'min_brightness', 'max_brightness', 'max_jump_um',
    'max_dist_um'}. The jump quantile runs over finite genomically-
    adjacent distances; max_dist over per-bin median distances. Derived,
    not inherited: the repo has been burned by every constant it ever
    ported across datasets.
    """
    amp = table['amp']
    finite_amp = amp[np.isfinite(amp)]
    neighbors = np.concatenate([np.diagonal(d, 1) for d in dmaps]) \
        if len(dmaps) else np.array([])
    neighbors = neighbors[np.isfinite(neighbors)]
    med_per_bin = np.nanmedian(dmaps, axis=1) if len(dmaps) else np.empty((0, 0))
    med = med_per_bin[np.isfinite(med_per_bin)]
    q = lambda v, p, fallback: float(np.quantile(v, p)) if len(v) else fallback
    return {
        'min_brightness': q(finite_amp, brightness_q[0], 0.0),
        'max_brightness': q(finite_amp, brightness_q[1], np.inf),
        'max_jump_um': q(neighbors, jump_q, np.inf),
        'max_dist_um': q(med, max_dist_q, np.inf),
    }


def apply_qc(table, dmaps, thresholds, min_traced=2):
    """ORCA filtering with the elementwise-brightness bug fixed.

    Marks a bin bad when: amplitude outside [min, max] brightness (the
    AMPLITUDE only); both genomic neighbors further than max_jump_um
    (interior bins); its median distance to all bins exceeds max_dist_um.
    Bad bins become NaN; alleles keep membership only with >= min_traced
    finite bins (ORCA used > 1).

    Returns dict: pos_um/amp (filtered COPIES), bads (n_a, n_bins) bool,
    kept (n_a,) bool over the INPUT rows, plus the recomputed dmaps of
    the kept alleles and their index into the input.
    """
    pos = table['pos_um'].copy()
    amp = table['amp'].copy()
    n_a, n_b = amp.shape
    if n_a == 0:
        # zero alleles is a legitimate gate outcome, not an error: the
        # jump gate's diagonal stack collapses to 1-D on empty input and
        # crashed here (IndexError) before this guard.
        return {'pos_um': pos, 'amp': amp,
                'bads': np.zeros((0, n_b), bool), 'kept': np.zeros(0, bool),
                'index': np.zeros(0, np.int64),
                'dmaps': np.empty((0, n_b, n_b), np.float32)}
    bads = np.zeros((n_a, n_b), dtype=bool)
    bads |= np.isfinite(amp) & (amp > thresholds['max_brightness'])
    bads |= np.isfinite(amp) & (amp < thresholds['min_brightness'])
    if n_b >= 3:
        d1 = np.array([np.diagonal(d, 1) for d in dmaps])   # (n_a, n_b-1)
        far = d1 > thresholds['max_jump_um']
        bads[:, 1:n_b - 1] |= far[:, 1:] & far[:, :-1]
    med = np.nanmedian(dmaps, axis=1)                       # (n_a, n_b)
    with np.errstate(invalid='ignore'):
        bads |= np.isfinite(med) & (med > thresholds['max_dist_um'])
    pos[bads] = np.nan
    amp[bads] = np.nan
    kept = np.isfinite(pos[:, :, 0]).sum(1) >= int(min_traced)
    out_pos = pos[kept]
    return {'pos_um': out_pos, 'amp': amp[kept], 'bads': bads,
            'kept': kept, 'index': np.flatnonzero(kept),
            'dmaps': polymer_distmaps(out_pos)}


def efficacy(pos_um):
    """(n_bins,) fraction of alleles with a finite position per bin."""
    if len(pos_um) == 0:
        return np.zeros(pos_um.shape[1])
    return np.isfinite(pos_um[:, :, 0]).mean(0)


def completeness(pos_um):
    """(n_alleles,) count of finite bins per allele."""
    return np.isfinite(pos_um[:, :, 0]).sum(1)
