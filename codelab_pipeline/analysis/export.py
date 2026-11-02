"""
Flat, analysis-ready tables of every spot and every allele.

The point is a file someone who has never seen this codebase can open and
work from: one row per thing, every coordinate spelled out, no nested
dicts, no knowledge of the store layout required.

THREE SHAPES, because the data has three shapes:

  cells   -- one row per segmented cell. Geometry, provenance and the
             counts that let a reader check the other two tables add up.

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

from codelab_pipeline.models import spot as spot_model  # noqa: E402

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
    # then amplitude, quality, ... as far as `n` asks; NaN where absent
    extra = tuple(v[3:n]) + (float('nan'),) * max(0, n - max(3, len(v)))
    return out + extra[:max(0, n - 3)]


def _um(xyz, voxel_um):
    """An (x, y, z) pixel/plane triple in micrometres.

    voxel_um is given per PIPELINE axis (y, x, z), so x takes voxel_um[1]
    and y takes voxel_um[0] -- the swap _xyz already made.
    """
    dy, dx, dz = (float(v) for v in voxel_um)
    return (xyz[0] * dx, xyz[1] * dy, xyz[2] * dz)


def _affine_parts(H, dz=0.0):
    """A 3x3 yx affine plus its z shift, as (theta_deg, dx, dy, dz).

    The stored transform is a general affine, but what it carries here
    is a rigid move: a rotation in the plane and a translation, with z
    separable and additive (alignment/frames.py, "Z is separable"). So
    four numbers say it, and they say it in the reader's units --
    degrees and pixels -- instead of nine matrix entries nobody can
    read.

    theta comes from the linear block, dx/dy from the translation
    column, and both are reported in the OUTPUT order (x, y) like every
    other coordinate in this file, while the matrix is y-major.

    A scale or a shear would make four numbers a lossy summary. There
    is no silent rounding: `scale` is returned beside them, and a
    reader who sees it away from 1 knows the summary is not the whole
    transform. Returns NaNs for a missing matrix rather than identity,
    because "no alignment ran" and "alignment found nothing to do" are
    different facts.
    """
    import math
    if H is None:
        return (float('nan'),) * 5
    A = np.asarray(H, float)
    if A.shape != (3, 3):
        return (float('nan'),) * 5
    # y-major: row 0 is y, row 1 is x
    theta = math.degrees(math.atan2(A[1, 0], A[0, 0]))
    scale = math.hypot(A[0, 0], A[1, 0])
    return (theta, float(A[1, 2]), float(A[0, 2]),
            float(dz if dz is not None else 0.0), scale)


#: the keys spot_rows reads itself. Everything else a stored spot
#: carries is passed through as its own column -- the same rule
#: cell_rows follows, and the reason an analysis can attach a
#: measurement to a spot and find it in the file.
_SPOT_STRUCTURAL = frozenset((
    'uid', 'fov', 'modality', 'hybe', 'channel', 'cell', 'celltype',
    'raw_coordinate', 'adj_coordinate', 'coordinate', 'size',
    'brightness', 'p_exist', 'z_status', 'mixture_centroids', 'linked',
    'linked_at', 'properties',
))

SPOT_COLUMNS = (
    'spot_key', 'fov', 'modality', 'hybe', 'readout_name', 'datatype',
    'readout_id', 'channel', 'channel_role',
    'cell', 'cell_key', 'celltype', 'uid',
    'raw_x', 'raw_y', 'raw_z',
    'adj_x', 'adj_y', 'adj_z',
    'raw_x_um', 'raw_y_um', 'raw_z_um',
    'adj_x_um', 'adj_y_um', 'adj_z_um',
    'size', 'brightness', 'p_exist', 'z_status', 'engine',
    'n_mixture_candidates', 'linked', 'linked_at',
)


class _Attr:
    """A dict seen as an object, for the spot helpers that take either."""

    def __init__(self, d):
        self._d = d

    def __getattr__(self, name):
        return self._d.get(name)


def _p_exist(s):
    v = s.get('p_exist')
    try:
        return float('nan') if v is None else float(v)
    except (TypeError, ValueError):
        return float('nan')


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
            # The per-spot quality, and the two fields that say where it
            # came from. p_exist is the learned engine's calibrated
            # probability that the spot exists, and it is what every count
            # in this project is gated on -- 'count@0.5' means p_exist >=
            # 0.5. Exporting spots without it leaves a reader unable to
            # reproduce a single published number. NaN where no learned
            # engine ran; z_status reads 'not_fit' on anything that
            # predates the field rather than raising.
            'p_exist': _p_exist(s),
            'z_status': spot_model.z_status_of(_Attr(s)),
            'engine': 'learned' if spot_model.from_learned_engine(s) else '',
            'n_mixture_candidates': len(s.get('mixture_centroids') or ()),
            'linked': bool(s.get('linked', False)),
            'linked_at': s.get('linked_at') or '',
        })
        for k, v in s.items():
            # `k not in rows[-1]` is the guard that matters: a
            # stored field named like a column this function
            # already filled would otherwise replace it silently,
            # and the file would say something the exporter never
            # computed under a name that promises it did.
            if (k not in _SPOT_STRUCTURAL and k not in rows[-1]
                    and _is_scalar(v)):
                rows[-1][k] = v
    return rows


#: structural fields the cell row spells out itself. Everything else a
#: cell carries is passed through untouched -- see cell_rows.
_CELL_STRUCTURAL = frozenset((
    'id', 'fov', 'celltype', 'nucleus', 'area', 'frame_shape',
    'reference_hybe', 'reference_modality', 'nucleus_hybe',
    'nucleus_modality', 'matrices', 'matrix_anchors',
    'matrix_provenance', 'linked', 'linked_at',
))

CELL_COLUMNS = (
    'cell_key', 'fov', 'cell', 'celltype',
    'home_hybe', 'home_modality',
    'nucleus_hybe', 'nucleus_modality',
    'nucleus_px', 'nucleus_um2',
    'nucleus_centroid_x', 'nucleus_centroid_y',
    'nucleus_centroid_x_um', 'nucleus_centroid_y_um',
    'nucleus_min_x', 'nucleus_max_x', 'nucleus_min_y', 'nucleus_max_y',
    'area_px', 'area_um2', 'area_centroid_x', 'area_centroid_y',
    'touches_frame_edge', 'frame_width', 'frame_height',
    'n_spots', 'n_spots_readout', 'n_alleles',
    # The cell's OWN stored alignment, not a resolved one. home_yx_* is
    # the matrix it keeps for (home_hybe, home_modality) and
    # home_yx_is_residual says whether it is a bare residual -- which
    # it is for everything written since the residual change, meaning
    # the FOV leg is NOT in it. A reader composes the two with
    # alignment.frames.FrameResolver, exactly as the app does. Nothing
    # here is pre-composed on purpose: a composition baked into a file
    # goes stale the moment FOV alignment is re-run, and the file
    # cannot say so.
    'home_yx_theta_deg', 'home_yx_dx', 'home_yx_dy', 'home_dz',
    'home_yx_scale', 'home_yx_is_residual',
    'n_hybes_aligned', 'arrays',
    'linked', 'linked_at',
)


def _is_scalar(v):
    """True when a value belongs in a table cell as it stands."""
    if v is None or isinstance(v, (bool, int, float, str, bytes)):
        return True
    if isinstance(v, np.generic):
        return True
    return False


def _mask_stats(mask, voxel_um):
    """(count, um2, centroid_x, centroid_y, min_x, max_x, min_y, max_y).

    A stored mask is (ys, xs) -- pixel INDICES, not an image -- so the
    area is how many there are and the extent is their range. Empty
    gives NaNs rather than zeros: a cell with no mask and a cell whose
    mask sits at the origin must not read the same.
    """
    nan = float('nan')
    if mask is None or len(mask) < 2 or not len(mask[0]):
        return (0, nan, nan, nan, nan, nan, nan, nan)
    ys = np.asarray(mask[0], float)
    xs = np.asarray(mask[1], float)
    dy, dx, _dz = (float(v) for v in voxel_um)
    return (int(len(ys)), float(len(ys)) * dy * dx,
            float(xs.mean()), float(ys.mean()),
            float(xs.min()), float(xs.max()),
            float(ys.min()), float(ys.max()))


def cell_rows(cell_dicts, fov=None, voxel_um=DEFAULT_VOXEL_UM,
              n_spots_of=None, n_readout_spots_of=None,
              n_alleles_of=None, arrays=None):
    """One row per segmented cell, as plain dicts.

    NOTHING IS HARDCODED AWAY. The columns above are the ones this
    function spells out -- geometry, provenance, counts, the stored
    alignment -- and every OTHER key a cell carries becomes a column of
    its own, named exactly as the store names it. A project that writes
    `n_speckle` or `d_surface_um` onto its cells gets those columns
    without a line changing here, which is the point: the exporter must
    not be a list of one project's fields.

    TWO MASKS, and they are not the same thing. `nucleus` is the
    nuclear segmentation and `area` the whole-cell one; both get their
    own count, area and centroid, because which one a number was
    computed over changes what it means. Neither is overwritten by an
    adjusted copy: an `adj_nucleus`, if a project ever writes one,
    arrives as its own field beside the original.

    `arrays` (a dict, optional) collects everything that is not a
    scalar -- the masks themselves, the per-hybe matrices, distmap --
    under '<cell_key>/<field>' for write_npz. The table then carries
    only what is readable in a cell, and the column `arrays` lists the
    field names stored beside it, so a reader can see what was kept
    without opening the .npz. Pass None to drop them.

    The counts come from the caller, not the cell, which carries none:
    they are what lets a reader check the three tables agree. n_spots
    counts every spot assigned to the cell, fiducial detections
    included; n_spots_readout counts only readout-channel ones, which
    is the far smaller number most analyses mean.

    touches_frame_edge is computed here rather than trusted from a
    stored flag -- it is a property of the mask and the frame, both in
    hand, and a stale flag is worse than none.
    """
    rows = []
    for c in cell_dicts or []:
        f = int(c.get('fov', fov if fov is not None else -1))
        cid = int(c.get('id', -1))
        key = 'F%03d-C%d' % (f, cid)
        frame = c.get('frame_shape') or ()
        fh = float(frame[0]) if len(frame) > 0 else float('nan')
        fw = float(frame[1]) if len(frame) > 1 else float('nan')
        nuc = _mask_stats(c.get('nucleus'), voxel_um)
        ar = _mask_stats(c.get('area'), voxel_um)
        edge = ''
        if nuc[0]:
            edge = bool(nuc[4] <= 0 or nuc[6] <= 0
                        or (fw == fw and nuc[5] >= fw - 1)
                        or (fh == fh and nuc[7] >= fh - 1))
        home_h = str(c.get('reference_hybe') or '')
        home_m = str(c.get('reference_modality') or '')
        mats = c.get('matrices') or {}
        entry = mats.get((home_h, home_m)) or {}
        th, dx_, dy_, dz_, sc = _affine_parts(entry.get('yx'),
                                              entry.get('dz', 0.0))
        row = {
            'cell_key': key, 'fov': f, 'cell': cid,
            'celltype': str(c.get('celltype') or ''),
            'home_hybe': home_h, 'home_modality': home_m,
            'nucleus_hybe': str(c.get('nucleus_hybe') or ''),
            'nucleus_modality': str(c.get('nucleus_modality') or ''),
            'nucleus_px': nuc[0], 'nucleus_um2': nuc[1],
            'nucleus_centroid_x': nuc[2], 'nucleus_centroid_y': nuc[3],
            'nucleus_centroid_x_um': nuc[2] * float(voxel_um[1]),
            'nucleus_centroid_y_um': nuc[3] * float(voxel_um[0]),
            'nucleus_min_x': nuc[4], 'nucleus_max_x': nuc[5],
            'nucleus_min_y': nuc[6], 'nucleus_max_y': nuc[7],
            'area_px': ar[0], 'area_um2': ar[1],
            'area_centroid_x': ar[2], 'area_centroid_y': ar[3],
            'touches_frame_edge': edge,
            'frame_width': fw, 'frame_height': fh,
            'n_spots': int((n_spots_of or {}).get(cid, 0)),
            'n_spots_readout': int((n_readout_spots_of or {}).get(cid, 0)),
            'n_alleles': int((n_alleles_of or {}).get(cid, 0)),
            'home_yx_theta_deg': th, 'home_yx_dx': dx_,
            'home_yx_dy': dy_, 'home_dz': dz_, 'home_yx_scale': sc,
            'home_yx_is_residual': bool(entry.get('yx_is_residual', False)),
            'n_hybes_aligned': len(mats),
            'linked': bool(c.get('linked', False)),
            'linked_at': c.get('linked_at') or '',
        }
        kept = []
        for k, v in c.items():
            if k in _CELL_STRUCTURAL:
                continue
            if _is_scalar(v) and k not in row:
                row[k] = v
            elif arrays is not None:
                arrays['%s/%s' % (key, k)] = v
                kept.append(k)
        if arrays is not None:
            for k in ('nucleus', 'area'):
                if c.get(k) is not None and len(c.get(k) or ()) >= 2:
                    arrays['%s/%s_y' % (key, k)] = np.asarray(c[k][0])
                    arrays['%s/%s_x' % (key, k)] = np.asarray(c[k][1])
                    kept.append(k)
            for (hy, mo), e in mats.items():
                arrays['%s/matrix/%s/%s/yx' % (key, mo, hy)] = np.asarray(
                    e.get('yx'), float)
            if mats:
                kept.append('matrices')
            for mo, H in (c.get('matrix_anchors') or {}).items():
                arrays['%s/anchor/%s' % (key, mo)] = np.asarray(H, float)
                kept.append('matrix_anchors')
        row['arrays'] = ' '.join(sorted(set(kept)))
        rows.append(row)
    return rows


def columns_for(rows, known):
    """`known` first, in order, then every other key the rows carry.

    The curated order is what makes a table readable; discovery is what
    keeps it complete. A field a project invented appears at the end
    rather than not at all, and the writers take this instead of a
    fixed tuple so nothing is silently dropped on the way out.
    """
    seen, extra = set(known), []
    for r in rows or []:
        for k in r:
            if k not in seen:
                seen.add(k)
                extra.append(k)
    return tuple(known) + tuple(sorted(extra))


def write_npz(arrays, path):
    """The non-scalar fields, beside their table.

    Keys are '<cell_key>/<field>', so np.load(path)['F001-C5/nucleus_y']
    is the mask the row F001-C5 summarises. Written atomically like
    every other writer here.
    """
    if not arrays:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    tmp = path + '.part.npz'
    np.savez_compressed(tmp, **{k: np.asarray(v, dtype=object)
                                if np.asarray(v).dtype == object
                                else np.asarray(v)
                                for k, v in arrays.items()})
    _replace(tmp, path)
    return path


#: the keys allele_rows reads itself; see _SPOT_STRUCTURAL. An
#: allele's own extra field repeats on each of its rows, which is what
#: a long table does with a per-item value.
_ALLELE_STRUCTURAL = frozenset((
    'id', 'fov', 'cell', 'celltype', 'anchor_uid', 'anchor_hybe',
    'anchor_channel', 'provenance', 'coordinate', 'raw_coordinate',
    'fiducial_trace_adj', 'fiducial_trace_raw', 'fiducial_trace',
    'fiducial_drift', 'polymer_adj', 'polymer_raw', 'polymer',
    'rejected_hybes', 'final_polymer', 'linked', 'linked_at',
    'properties',
))

ALLELE_COLUMNS = (
    'allele_key', 'allele_id', 'fov', 'cell', 'cell_key', 'celltype',
    'modality', 'modality_source',
    'bin_index', 'readout_id', 'hybe', 'readout_name', 'datatype',
    'candidate_index', 'n_candidates', 'is_selected', 'selection_rule',
    'fiducial_found', 'rejected_reason',
    'readout_adj_x', 'readout_adj_y', 'readout_adj_z', 'readout_adj_amplitude',
    'readout_quality',
    'readout_raw_x', 'readout_raw_y', 'readout_raw_z', 'readout_raw_amplitude',
    'readout_adj_x_um', 'readout_adj_y_um', 'readout_adj_z_um',
    'final_x', 'final_y', 'final_z',
    'final_x_um', 'final_y_um', 'final_z_um',
    'fiducial_adj_x', 'fiducial_adj_y', 'fiducial_adj_z', 'fiducial_adj_amplitude',
    'fiducial_quality',
    'fiducial_raw_x', 'fiducial_raw_y', 'fiducial_raw_z', 'fiducial_raw_amplitude',
    'fiducial_drift_x', 'fiducial_drift_y', 'fiducial_drift_z',
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
            fa = _xyz(fid_adj.get(hybe), n=5)
            fr = _xyz(fid_raw.get(hybe), n=5)
            fdr = (a.get('fiducial_drift') or {}).get(hybe)
            fdr = ((float(fdr[1]), float(fdr[0]), float(fdr[2]))   # x, y, z
                   if fdr is not None else _NAN3)
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
                'fiducial_quality': fa[4],
                'fiducial_raw_x': fr[0], 'fiducial_raw_y': fr[1],
                'fiducial_raw_z': fr[2], 'fiducial_raw_amplitude': fr[3],
                'fiducial_drift_x': fdr[0], 'fiducial_drift_y': fdr[1],
                'fiducial_drift_z': fdr[2],
            })
            if not cands_adj:
                # the bin still gets a row -- see the module docstring
                empty = dict(per_bin)
                empty.update({
                    'candidate_index': -1, 'is_selected': False,
                    'readout_adj_x': float('nan'), 'readout_adj_y': float('nan'),
                    'readout_adj_z': float('nan'),
                    'readout_adj_amplitude': float('nan'),
                    'readout_quality': float('nan'),
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
                ca = _xyz(cand, n=5)
                cr = _xyz(cands_raw[k], n=5) if paired else _xyz(None, n=5)
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
                    'readout_quality': ca[4],
                    'readout_raw_x': cr[0], 'readout_raw_y': cr[1],
                    'readout_raw_z': cr[2], 'readout_raw_amplitude': cr[3],
                    'readout_adj_x_um': ca_um[0], 'readout_adj_y_um': ca_um[1],
                    'readout_adj_z_um': ca_um[2],
                })
                for _k, _v in a.items():
                    if (_k not in _ALLELE_STRUCTURAL
                            and _k not in row and _is_scalar(_v)):
                        row[_k] = _v
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


class Tables(object):
    """What one export pass produced.

    A class and not a tuple because the pass now yields four things
    plus a summary, and `cells, spots, alleles, arrays, summary = ...`
    is a line nobody can read or safely extend.
    """

    __slots__ = ('cells', 'spots', 'alleles', 'arrays', 'summary')

    def __init__(self, cells, spots, alleles, arrays, summary):
        self.cells = cells
        self.spots = spots
        self.alleles = alleles
        self.arrays = arrays
        self.summary = summary

    def sheets(self):
        """[(name, rows, columns)] for write_excel / write_csv.

        Columns are DISCOVERED per table (columns_for), so a field a
        project wrote onto its cells or spots reaches the file without
        anything here naming it.
        """
        return [('cells', self.cells, columns_for(self.cells, CELL_COLUMNS)),
                ('spots', self.spots, columns_for(self.spots, SPOT_COLUMNS)),
                ('alleles', self.alleles,
                 columns_for(self.alleles, ALLELE_COLUMNS))]


def collect(storage_paths_by_modality, fovs=None, voxel_um=DEFAULT_VOXEL_UM,
            progress=None, should_stop=None, want_arrays=True):
    """Build every table for a whole project.

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

    The cells of each FOV were already being read here for celltype and
    then thrown away; they now become the third table at no extra I/O,
    carrying the per-cell counts computed from the two tables beside
    them -- which is what lets a reader check that the three agree.

    want_arrays=False skips collecting the masks and matrices, for a
    caller who only wants the tables (they are ~10 KB per cell).

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

    cells_out, spots_out, alleles_out, seen = [], [], [], []
    arrays = {} if want_arrays else None
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
            srows = spot_rows(
                analysis_store.read_spots(sp, fov), fov, celltype_of=ct,
                voxel_um=voxel_um, layout_by_modality=layout_by_modality)
            arows = []
            for a in analysis_store.read_fov_alleles(sp, fov) or []:
                m = str((a.get('provenance') or {}).get('modality')
                        or default_modality)
                arows.extend(allele_rows(
                    [a], fov, bins_by_modality.get(m, []), modality=m,
                    celltype_of=ct, voxel_um=voxel_um,
                    records_by_hybe=layout_by_modality.get(m, {})))
            n_spots, n_read, n_all = {}, {}, {}
            for r in srows:
                c = r['cell']
                n_spots[c] = n_spots.get(c, 0) + 1
                if r['channel_role'] == 'readout':
                    n_read[c] = n_read.get(c, 0) + 1
            for a in {(r['cell'], r['allele_id']) for r in arows}:
                n_all[a[0]] = n_all.get(a[0], 0) + 1
            cells_out.extend(cell_rows(
                cells, fov, voxel_um=voxel_um, n_spots_of=n_spots,
                n_readout_spots_of=n_read, n_alleles_of=n_all,
                arrays=arrays))
            spots_out.extend(srows)
            alleles_out.extend(arows)
            seen.append(fov)
        except Exception as e:
            errors.append(f'FOV{fov:03d}: {e}')
    return Tables(cells_out, spots_out, alleles_out, arrays, {
        'fovs': seen, 'n_cells': len(cells_out), 'n_spots': len(spots_out),
        'n_allele_rows': len(alleles_out), 'errors': errors})


#: fields a spot row already carries under the same name -- taken from
#: the SPOT row, not duplicated from the cell, so the combined table
#: never holds two columns that should be one.
_COMBINED_SKIP = frozenset(('fov', 'cell', 'celltype', 'cell_key'))

#: what a COMPACT combined row keeps from the spot: what identifies the
#: measurement, and where it is. Everything dropped here is still in
#: the spots sheet -- this table is not the record, it is the thing you
#: group by.
COMBINED_SPOT = (
    'spot_key', 'fov', 'modality', 'hybe', 'readout_name', 'readout_id',
    'channel_role', 'cell', 'cell_key', 'celltype', 'uid',
    'adj_x_um', 'adj_y_um', 'adj_z_um', 'p_exist',
)
#: and from the cell. Two summaries earn their place because analyses
#: routinely divide by one and filter on the other; the geometry, the
#: alignment and the bookkeeping do not.
COMBINED_CELL = ('nucleus_um2', 'n_spots_readout', 'touches_frame_edge')


def combined_rows(tables, prefix='cell_', compact=True):
    """One row per spot, with its cell joined on. THE ANALYSIS CACHE.

    Every number in it is already in `spots` or `cells`; the join is
    cell_key. It exists because the question an analysis actually asks
    -- this locus, in this cell, of this genotype, at this distance --
    spans two tables, and joining them by hand is where the first
    mistake goes.

    COMPACT BY DEFAULT, and that is the point of the table. A reader
    handed 84 columns cannot tell which ones carry the result, so this
    keeps three things and drops the rest:

      1. what identifies the measurement and where it is (COMBINED_SPOT)
      2. two cell summaries analyses divide by or filter on
         (COMBINED_CELL)
      3. EVERY FIELD THAT IS NOT PART OF THE DECLARED SCHEMA

    (3) is the rule that makes this generic rather than a list of one
    project's columns. The declared schema is the pipeline's own
    bookkeeping -- geometry, frames, alignment, link state, all of it
    already in the other sheets. A field outside it got there because
    somebody measured it and attached it, which is exactly what an
    analysis cache should carry. So a project's own fields arrive
    automatically and the pipeline's plumbing does not.

    compact=False joins the whole cell row instead, for a caller who
    wants one flat file with everything in it.

    Cell fields arrive under `prefix` so a reader can always tell which
    table a column came from. The four the spot row already carries
    under the same name (fov, cell, celltype, cell_key) are not copied:
    one column, from the spot.

    A spot with no cell (cell == -1, most of the real data) keeps its
    row with the cell columns blank -- dropping it would silently
    change every count, the same rule spot_rows follows.
    """
    by_cell = {r['cell_key']: r for r in (tables.cells or [])}
    keep_s = set(COMBINED_SPOT)
    keep_c = set(COMBINED_CELL)
    declared_s, declared_c = set(SPOT_COLUMNS), set(CELL_COLUMNS)
    out = []
    for s in (tables.spots or []):
        if compact:
            row = {k: v for k, v in s.items()
                   if k in keep_s or k not in declared_s}
        else:
            row = dict(s)
        c = by_cell.get(s.get('cell_key') or '')
        if c:
            for k, v in c.items():
                if k in _COMBINED_SKIP:
                    continue
                if compact and k not in keep_c and k in declared_c:
                    continue
                row[prefix + k] = v
        out.append(row)
    return out


def combined_columns(rows, tables, prefix='cell_', compact=True):
    """Spot columns first, then the joined cell ones, then discoveries."""
    spot_known = COMBINED_SPOT if compact else SPOT_COLUMNS
    cell_known = (COMBINED_CELL if compact
                  else [k for k in CELL_COLUMNS if k not in _COMBINED_SKIP])
    known = tuple(spot_known) + tuple(prefix + k for k in cell_known)
    present = [k for k in known if any(k in r for r in rows)] if rows         else list(known)
    return columns_for(rows, present)


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
