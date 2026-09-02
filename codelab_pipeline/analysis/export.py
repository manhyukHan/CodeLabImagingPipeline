"""
Flat, analysis-ready tables of every spot and every allele.

The point is a file someone who has never seen this codebase can open and
work from: one row per thing, every coordinate spelled out, no nested
dicts, no knowledge of the store layout required.

TWO SHAPES, because the data has two shapes:

  spots   -- one row per spot. A spot IS a single position.

  alleles -- LONG format. An allele is a chromatin trace: one anchor plus
             a position per genomic bin, and a bin can hold SEVERAL
             accepted candidates (sister chromatids are kept side by side,
             never pruned -- see AnAllele.polymer_adj). So one row per
             (allele, bin, candidate), all sharing one allele_id.
             Every bin of every allele appears, including bins that were
             rejected (coordinates blank, rejected_reason filled) -- a
             missing bin is a result, and silently dropping it would make
             traces look complete when they are not.

COORDINATE ORDER IS THE TRAP. Everything inside this pipeline is
(y, x, z). Everything in these files is named _x/_y/_z, so index 1 is x,
index 0 is y, index 2 is z. That mapping lives in _xyz() alone; nothing
else here indexes a coordinate.

UNITS. y/x are pixels and z is a plane INDEX, exactly as stored. The _um
columns beside them are that value times voxel_um per axis -- the same
scaling analysis/polymer.py applies, applied here exactly once, so the
two agree. Pixels are not square with z, which is why the scaling is
per-axis and not a single number.

FIDUCIAL vs READOUT. Both are recorded per hybe:
  fiducial_* is the drift anchor fitted in that hybe (NOT the trace),
  readout_*  is the chromatin position itself.
Each exists in a raw (that hybe's own frame) and an adj (the shared
reference frame, fiducial-drift corrected) form, and both are exported --
adj is what analysis should use; raw is what makes the correction
auditable.
"""
import os

import numpy as np

from . import polymer

DEFAULT_VOXEL_UM = polymer.DEFAULT_VOXEL_UM


def _xyz(value, n=3):
    """(y, x, z[, amplitude]) -> (x, y, z) plus amplitude when present.

    THE one place a coordinate is unpacked. (y, x, z) is the pipeline's
    order; _x/_y/_z in the output are the reader's. Returns NaNs for a
    missing value so a row is never silently short.
    """
    if value is None:
        return (float('nan'),) * n
    v = [float(x) for x in value]
    while len(v) < 3:
        v.append(float('nan'))
    out = (v[1], v[0], v[2])            # x, y, z
    if n == 4:
        out = out + (v[3] if len(v) > 3 else float('nan'),)
    return out


def _um(xyz, voxel_um):
    """An (x, y, z) pixel/plane triple in micrometres.

    voxel_um is given per PIPELINE axis (y, x, z), so x takes voxel_um[1]
    and y takes voxel_um[0] -- the swap _xyz already made.
    """
    dy, dx, dz = (float(v) for v in voxel_um)
    return (xyz[0] * dx, xyz[1] * dy, xyz[2] * dz)


SPOT_COLUMNS = (
    'spot_key', 'fov', 'modality', 'hybe', 'readout_name', 'datatype',
    'readout_id', 'channel', 'channel_role',
    'cell', 'cell_key', 'celltype', 'uid',
    'raw_x', 'raw_y', 'raw_z',
    'adj_x', 'adj_y', 'adj_z',
    'raw_x_um', 'raw_y_um', 'raw_z_um',
    'adj_x_um', 'adj_y_um', 'adj_z_um',
    'size', 'brightness', 'n_mixture_candidates', 'linked', 'linked_at',
)


def _layout_of(layout_by_modality, modality, hybe):
    return ((layout_by_modality or {}).get(modality) or {}).get(hybe) or {}


def spot_rows(spot_dicts, fov=None, celltype_of=None, voxel_um=DEFAULT_VOXEL_UM,
              layout_by_modality=None):
    """One row per spot, as plain dicts keyed by SPOT_COLUMNS.

    fov is read from the SPOT, not from the caller: the value is stored on
    every spot and a caller passing the wrong one would mislabel every row
    while the truth sat unused. `fov` here is only a fallback for a spot
    that somehow lacks it.

    celltype_of: {cell_id: celltype} from the FOV's cells -- a spot carries
    its own copy, but the cell is authoritative and a spot saved before the
    last celltype run would disagree with it.

    layout_by_modality: {modality: {hybe: layout_record}}. The store knows
    hybe FOLDER names only; the ExperimentLayout is where a bin's identity
    lives. Joining it turns 'Hyb_105' into 'Fmo4_exon' and -- more
    importantly -- lets channel_role say whether a detection is in the
    FIDUCIAL channel or a readout one. On the real store most spots are
    fiducial-channel detections, and a reader who assumes otherwise is
    reading alignment beads as signal.

    cell == -1 means unassigned/homeless. Exported as -1, never dropped:
    it is most of the real data, and dropping it would silently change
    every count a reader makes.
    """
    celltype_of = celltype_of or {}
    rows = []
    for s in spot_dicts or []:
        f = int(s.get('fov', fov if fov is not None else -1))
        modality = str(s.get('modality', ''))
        hybe = str(s.get('hybe', ''))
        channel = int(s.get('channel', 0))
        rec = _layout_of(layout_by_modality, modality, hybe)
        fid = rec.get('fiducial_channel')
        raw = _xyz(s.get('raw_coordinate'))
        adj = _xyz(s.get('adj_coordinate'))
        raw_um, adj_um = _um(raw, voxel_um), _um(adj, voxel_um)
        cell = int(s.get('cell', -1))
        rows.append({
            # uid is unique per FOV, NOT per project, and is shared across
            # modalities within one FOV -- so a reader who concatenates
            # FOVs needs a key that survives that. This is that key.
            'spot_key': f'F{f:03d}-U{int(s.get("uid", 0))}',
            'fov': f,
            'modality': modality,
            'hybe': hybe,
            'readout_name': str(rec.get('readout_name') or ''),
            'datatype': str(rec.get('datatype') or ''),
            'readout_id': int(rec['readout_id']) if rec.get('readout_id') is not None else -1,
            'channel': channel,
            'channel_role': ('' if fid is None else
                             ('fiducial' if int(fid) == channel else 'readout')),
            'cell': cell,
            'cell_key': '' if cell < 0 else f'F{f:03d}-C{cell}',
            'celltype': celltype_of.get(cell, str(s.get('celltype', '') or '')),
            'uid': int(s.get('uid', 0)),
            'raw_x': raw[0], 'raw_y': raw[1], 'raw_z': raw[2],
            'adj_x': adj[0], 'adj_y': adj[1], 'adj_z': adj[2],
            'raw_x_um': raw_um[0], 'raw_y_um': raw_um[1], 'raw_z_um': raw_um[2],
            'adj_x_um': adj_um[0], 'adj_y_um': adj_um[1], 'adj_z_um': adj_um[2],
            'size': float(s.get('size', float('nan'))),
            'brightness': float(s.get('brightness', float('nan'))),
            'n_mixture_candidates': len(s.get('mixture_centroids') or ()),
            'linked': bool(s.get('linked', False)),
            'linked_at': s.get('linked_at') or '',
        })
    return rows


ALLELE_COLUMNS = (
    'allele_key', 'allele_id', 'fov', 'cell', 'cell_key', 'celltype',
    'modality', 'modality_source',
    'bin_index', 'readout_id', 'hybe', 'readout_name', 'datatype',
    'candidate_index', 'n_candidates', 'is_selected', 'selection_rule',
    'fiducial_found', 'rejected_reason',
    'readout_adj_x', 'readout_adj_y', 'readout_adj_z', 'readout_adj_amplitude',
    'readout_raw_x', 'readout_raw_y', 'readout_raw_z', 'readout_raw_amplitude',
    'readout_adj_x_um', 'readout_adj_y_um', 'readout_adj_z_um',
    'final_x', 'final_y', 'final_z',
    'final_x_um', 'final_y_um', 'final_z_um',
    'fiducial_adj_x', 'fiducial_adj_y', 'fiducial_adj_z', 'fiducial_adj_amplitude',
    'fiducial_raw_x', 'fiducial_raw_y', 'fiducial_raw_z', 'fiducial_raw_amplitude',
    'anchor_uid', 'anchor_hybe', 'anchor_channel',
    'anchor_adj_x', 'anchor_adj_y', 'anchor_adj_z',
    'anchor_raw_x', 'anchor_raw_y', 'anchor_raw_z',
)

_NAN3 = (float('nan'),) * 3


def _committed_positions(allele_dict, hybes):
    """((n_bins, 3) y/x/z in bin order, source label).

    AnAllele.final_polymer is the committed position per bin -- but it is
    EMPTY until the collapse has been run, and on a store where tracing
    has not been finalised that is every allele. Exporting a column of
    blanks there would be useless, so the same collapse the analysis
    layer uses (polymer.collapse_polymer, brightest candidate) is applied
    here instead, and final_source records which it was: a reader can
    tell a stored decision from one this export derived.
    """
    fp = np.asarray(allele_dict.get('final_polymer')
                    if allele_dict.get('final_polymer') is not None
                    else np.empty((0, 3)), dtype=np.float64).reshape(-1, 3)
    if len(fp) >= len(hybes) and len(fp):
        return fp, 'stored'
    pos, _amp, _n = polymer.collapse_polymer(allele_dict, hybes)
    return pos, 'computed'


def _final_for_bin(positions, j):
    """One bin's committed position as (x, y, z), or NaNs."""
    if positions is None or j >= len(positions):
        return _NAN3
    return _xyz(tuple(positions[j]))


def _allele_axis(bins, allele, records_by_hybe):
    """[(bin_index, readout_id, hybe, datatype), ...] -- the rows this
    allele gets.

    Every genomic bin of the layout comes first, in genomic order, whether
    or not this allele saw it. Then any OTHER hybe the allele actually
    carries data for: polymer_adj/fiducial_trace_adj also hold the R
    (repeat) and T (toe) QC rounds, and a bin-only export would throw
    those traced positions away. They get bin_index -1 -- they are real
    measurements but not genomic-locus bins, and the datatype column is
    how a reader filters them.
    """
    axis = [(j, int(rid), h, 'H') for j, (rid, h) in enumerate(bins)]
    in_bins = {h for _r, h in bins}
    have = (set(allele.get('polymer_adj') or {})
            | set(allele.get('fiducial_trace_adj') or {})
            | set(allele.get('rejected_hybes') or {}))
    for h in sorted(have - in_bins):
        rec = records_by_hybe.get(h) or {}
        rid = rec.get('readout_id')
        axis.append((-1, int(rid) if rid is not None else -1, h,
                     str(rec.get('datatype') or '')))
    return axis


def allele_rows(allele_dicts, fov=None, bins=(), modality='',
                celltype_of=None, voxel_um=DEFAULT_VOXEL_UM,
                records_by_hybe=None):
    """Long-format rows: one per (allele, genomic bin, candidate).

    bins: [(readout_id, hybe), ...] in genomic order, from
    analysis.polymer.genomic_bins(layout_records) -- the LAYOUT is
    authoritative for which bins exist and in what order, so a bin the
    allele never saw still gets a row.

    modality: an allele does not store one directly -- it is recorded in
    provenance by whatever traced it. That is preferred, and the caller's
    value is only a fallback; modality_source says which was used, because
    a label that was assumed should not read like one that was recorded.

    Candidate pairing: polymer_adj[hybe] and polymer_raw[hybe] are built
    together from one result tuple and are index-aligned by construction
    (localization.py assigns both from ro_results[hybe]). If they ever
    disagree in length the raw side is left blank rather than paired by
    position, because a wrong pairing is worse than a missing value.
    """
    celltype_of = celltype_of or {}
    records_by_hybe = records_by_hybe or {}
    rows = []
    for a in allele_dicts or []:
        f = int(a.get('fov', fov if fov is not None else -1))
        cell = int(a.get('cell', -1))
        anchor_adj = _xyz(a.get('coordinate'))
        anchor_raw = _xyz(a.get('raw_coordinate'))
        adj_by_hybe = a.get('polymer_adj') or {}
        raw_by_hybe = a.get('polymer_raw') or {}
        fid_adj = a.get('fiducial_trace_adj') or {}
        fid_raw = a.get('fiducial_trace_raw') or {}
        rejected = a.get('rejected_hybes') or {}
        prov = a.get('provenance') or {}
        prov_modality = str(prov.get('modality') or '')
        committed, final_source = _committed_positions(a, [h for _r, h in bins])
        base = {
            'allele_key': f'F{f:03d}-A{int(a.get("id", 0))}',
            'allele_id': int(a.get('id', 0)),
            'fov': f,
            'cell': cell,
            'cell_key': '' if cell < 0 else f'F{f:03d}-C{cell}',
            'celltype': celltype_of.get(cell, ''),
            'modality': prov_modality or str(modality or ''),
            'modality_source': 'provenance' if prov_modality else 'assumed',
            'anchor_uid': int(a.get('anchor_uid', 0)),
            'anchor_hybe': str(a.get('anchor_hybe', '')),
            'anchor_channel': int(a.get('anchor_channel', 0)),
            'anchor_adj_x': anchor_adj[0], 'anchor_adj_y': anchor_adj[1],
            'anchor_adj_z': anchor_adj[2],
            'anchor_raw_x': anchor_raw[0], 'anchor_raw_y': anchor_raw[1],
            'anchor_raw_z': anchor_raw[2],
        }
        for j, readout_id, hybe, datatype in _allele_axis(bins, a, records_by_hybe):
            cands_adj = list(adj_by_hybe.get(hybe) or [])
            cands_raw = list(raw_by_hybe.get(hybe) or [])
            paired = len(cands_raw) == len(cands_adj)
            fa = _xyz(fid_adj.get(hybe), n=4)
            fr = _xyz(fid_raw.get(hybe), n=4)
            # committed positions are indexed by GENOMIC bin; a QC round
            # (bin_index -1) has no committed position by construction
            fin = _final_for_bin(committed, j) if j >= 0 else _NAN3
            fin_um = _um(fin, voxel_um)
            rec = records_by_hybe.get(hybe) or {}
            per_bin = dict(base)
            per_bin.update({
                'bin_index': j, 'readout_id': int(readout_id), 'hybe': str(hybe),
                'readout_name': str(rec.get('readout_name') or ''),
                'datatype': datatype,
                'n_candidates': len(cands_adj),
                'selection_rule': final_source,
                # a hybe KEY that is present but None means the fiducial
                # fit failed there; a missing key means it was never tried
                'fiducial_found': (hybe in fid_adj and fid_adj.get(hybe) is not None),
                'rejected_reason': str(rejected.get(hybe, '') or ''),
                'final_x': fin[0], 'final_y': fin[1], 'final_z': fin[2],
                'final_x_um': fin_um[0], 'final_y_um': fin_um[1],
                'final_z_um': fin_um[2],
                'fiducial_adj_x': fa[0], 'fiducial_adj_y': fa[1],
                'fiducial_adj_z': fa[2], 'fiducial_adj_amplitude': fa[3],
                'fiducial_raw_x': fr[0], 'fiducial_raw_y': fr[1],
                'fiducial_raw_z': fr[2], 'fiducial_raw_amplitude': fr[3],
            })
            if not cands_adj:
                # the bin still gets a row -- see the module docstring
                empty = dict(per_bin)
                empty.update({
                    'candidate_index': -1, 'is_selected': False,
                    'readout_adj_x': float('nan'), 'readout_adj_y': float('nan'),
                    'readout_adj_z': float('nan'),
                    'readout_adj_amplitude': float('nan'),
                    'readout_raw_x': float('nan'), 'readout_raw_y': float('nan'),
                    'readout_raw_z': float('nan'),
                    'readout_raw_amplitude': float('nan'),
                    'readout_adj_x_um': float('nan'),
                    'readout_adj_y_um': float('nan'),
                    'readout_adj_z_um': float('nan'),
                })
                rows.append(empty)
                continue
            for k, cand in enumerate(cands_adj):
                ca = _xyz(cand, n=4)
                cr = _xyz(cands_raw[k], n=4) if paired else _xyz(None, n=4)
                ca_um = _um(ca[:3], voxel_um)
                row = dict(per_bin)
                row.update({
                    'candidate_index': k,
                    # is_selected marks the candidate the collapse chose,
                    # so `is_selected == True` alone is the simple
                    # one-row-per-bin view of the same file.
                    'is_selected': bool(np.isfinite(fin[0])
                                     and np.allclose(ca[:3], fin, equal_nan=False,
                                                     atol=0.01, rtol=0.0)),
                    'readout_adj_x': ca[0], 'readout_adj_y': ca[1],
                    'readout_adj_z': ca[2], 'readout_adj_amplitude': ca[3],
                    'readout_raw_x': cr[0], 'readout_raw_y': cr[1],
                    'readout_raw_z': cr[2], 'readout_raw_amplitude': cr[3],
                    'readout_adj_x_um': ca_um[0], 'readout_adj_y_um': ca_um[1],
                    'readout_adj_z_um': ca_um[2],
                })
                rows.append(row)
    return rows


# -- collecting a whole project ------------------------------------------

def _layouts(storage_paths_by_modality, errors):
    """({modality: {hybe: record}}, {modality: [(readout_id, hybe)]}).

    The ExperimentLayout is per MODALITY and lives outside the store, so a
    layout that cannot be reached (the acquisition share is not mounted)
    costs the readable NAMES, not the export: coordinates do not need it.
    Every such failure is reported rather than silently blanking columns.
    """
    by_hybe, bins = {}, {}
    for modality, sp in (storage_paths_by_modality or {}).items():
        if not sp:
            continue
        try:
            records = polymer.records_for(sp, modality)
            by_hybe[modality] = {str(r['folder']): r for r in records}
            bins[modality] = polymer.genomic_bins(records)
        except Exception as e:
            by_hybe[modality], bins[modality] = {}, []
            errors.append(f'{modality}: layout unreadable, so readout names '
                          f'and genomic bins are missing ({e})')
    return by_hybe, bins


def collect(storage_paths_by_modality, fovs=None, voxel_um=DEFAULT_VOXEL_UM,
            progress=None, should_stop=None):
    """Build both tables for a whole project.

    storage_paths_by_modality: {modality: storage_path}.

    ONE PASS PER FOV, NOT ONE PER MODALITY. analysis/ is shared by every
    modality of a project -- paths.analysis_dir('<proj>/DNA') and
    ('<proj>/RNA') are the same directory -- so reading spots and alleles
    once per modality returns the identical rows twice and doubles every
    count in the file. (Verified on the real store: read_spots via the DNA
    and RNA paths returns the same 153 uids.) Modality is a property of
    each SPOT and of each allele's provenance, not of the path it was read
    through.

    The layout is still resolved per modality -- that is genuinely
    per-modality data -- and each allele is binned against ITS OWN
    modality's genomic bins.

    fovs: None means every FOV the project holds. progress(done, total,
    message) is called per FOV; should_stop() is polled so a long export
    can be cancelled. A FOV that fails is recorded in summary['errors']
    and the rest still export -- a partial table beats no table.
    """
    from ..io import analysis_store, paths

    errors = []
    layout_by_modality, bins_by_modality = _layouts(storage_paths_by_modality,
                                                    errors)
    # one representative path per ANALYSIS dir -- the shared capsule store
    by_analysis = {}
    for modality, sp in (storage_paths_by_modality or {}).items():
        if not sp:
            continue
        try:
            by_analysis.setdefault(os.path.normcase(paths.analysis_dir(sp)), sp)
        except Exception as e:
            errors.append(f'{modality}: {e}')
    default_modality = next(iter(storage_paths_by_modality or {}), '')

    todo = []
    for sp in by_analysis.values():
        for fov in (fovs if fovs is not None else _fovs_in_store(paths, sp)):
            todo.append((sp, int(fov)))
    total = len(todo)

    spots_out, alleles_out, seen = [], [], []
    for i, (sp, fov) in enumerate(todo, start=1):
        if should_stop is not None and should_stop():
            errors.append('cancelled by the user')
            break
        if progress is not None:
            progress(i, total, f'FOV{fov:03d}')
        try:
            cells, _ = analysis_store.read_cells(sp, fov)
            ct = {int(c['id']): str(c.get('celltype') or '')
                  for c in (cells or [])}
            spots_out.extend(spot_rows(
                analysis_store.read_spots(sp, fov), fov, celltype_of=ct,
                voxel_um=voxel_um, layout_by_modality=layout_by_modality))
            for a in analysis_store.read_fov_alleles(sp, fov) or []:
                m = str((a.get('provenance') or {}).get('modality')
                        or default_modality)
                alleles_out.extend(allele_rows(
                    [a], fov, bins_by_modality.get(m, []), modality=m,
                    celltype_of=ct, voxel_um=voxel_um,
                    records_by_hybe=layout_by_modality.get(m, {})))
            seen.append(fov)
        except Exception as e:
            errors.append(f'FOV{fov:03d}: {e}')
    return spots_out, alleles_out, {
        'fovs': seen, 'n_spots': len(spots_out),
        'n_allele_rows': len(alleles_out), 'errors': errors}


def _fovs_in_store(paths, storage_path):
    """Every FOV with an analysis capsule, ascending."""
    out = []
    try:
        root = paths.analysis_dir(storage_path)
        for name in sorted(os.listdir(root)):
            if name.startswith('fov') and name[3:].isdigit():
                out.append(int(name[3:]))
    except OSError:
        pass
    return out


# -- writing ------------------------------------------------------------

EXCEL_MAX_ROWS = 1_048_576 - 1          # minus the header row


def _replace(tmp, target):
    os.replace(tmp, target)


def write_csv(rows, columns, path):
    """One table, written atomically (.part + os.replace, the convention
    every other writer here follows) so an interrupted export leaves the
    previous file intact instead of a half one."""
    import csv
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    tmp = path + '.part'
    with open(tmp, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    _replace(tmp, path)
    return path


def write_excel(sheets, path):
    """sheets: [(name, rows, columns), ...] -> one .xlsx workbook.

    Raises ValueError if a sheet exceeds what Excel can hold, rather than
    writing a file that silently loses rows -- the caller should fall back
    to CSV, which has no such limit.
    """
    import pandas as pd
    for name, rows, _cols in sheets:
        if len(rows) > EXCEL_MAX_ROWS:
            raise ValueError(
                f'sheet {name!r} has {len(rows)} rows, more than Excel can '
                f'hold ({EXCEL_MAX_ROWS}) -- export as CSV instead')
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    # the temp name KEEPS the .xlsx extension: pandas picks/validates its
    # engine from the extension and refuses a bare '.part'
    tmp = path + '.part.xlsx'
    with pd.ExcelWriter(tmp, engine='openpyxl') as xw:
        for name, rows, columns in sheets:
            df = pd.DataFrame(rows, columns=list(columns))
            df.to_excel(xw, sheet_name=name, index=False)
    _replace(tmp, path)
    return path
