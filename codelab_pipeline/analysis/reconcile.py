"""
reconcile: bring every stored adj_coordinate back into agreement with
the CURRENT alignment.

adj fields are SAVE-TIME SNAPSHOTS -- each was computed with whatever
matrices existed at its save. Re-run any alignment layer later (FOV
refit, cross-modal re-bridge, cell residuals) and every stored adj is
stale while raw stays ground truth. This module re-derives them:

  spot.adj_coordinate        <- raw_coordinate through the resolver
  allele.coordinate          <- raw_coordinate (anchor_hybe frame)
  fiducial_trace_adj[h]      <- fiducial_trace_raw[h]
  polymer_adj[h]             <- polymer_raw[h] projected, PLUS the
                                fiducial drift correction re-derived
                                from the freshly projected fiducials
                                (delta = fid_adj[ref] - fid_adj[h])

using the store-built FrameResolver (analysis/resolvers.py), so it runs
headless. Everything it cannot re-derive it COUNTS and names instead of
guessing: alleles saved before the raw fields existed, traces whose
provenance carries no reference hybe and none was passed, spots with no
raw. Nothing is ever interpolated into existence.

The 2D-placeholder rule: a spot with raw z == 0 and adj z == 0 is a
2D detection whose depth was never measured; projecting 0 through the
z-chain would MINT a depth out of a placeholder, so it stays 0 and is
counted as z_placeholder.

Dry-run by default: write=False reports the drift (how stale the store
is) without touching a byte. write=True persists through the store's
own atomic doors (write_spots / write_allele_dicts).
"""
import numpy as np

from codelab_pipeline.io import analysis_store
from codelab_pipeline.models.cell import ACell
from codelab_pipeline.models.spot import ASpot


def _project_yx(resolver, hybe, modality, cell, ry, rx):
    H = resolver.to_shared(hybe, modality, cell)
    y, x, _ = H @ np.array([float(ry), float(rx), 1.0])
    return float(y), float(x)


def _shift_stats(deltas):
    if not deltas:
        return {'n': 0}
    d = np.array(deltas)
    return {'n': int(len(d)), 'median': float(np.median(d)),
            'p90': float(np.percentile(d, 90)), 'max': float(d.max())}


def _anchor_hybe_of(cell_dict, modality):
    """The hybe this modality's cell residuals were measured against,
    parsed from matrix_provenance's reference_sequence ('Hyb_103(cell
    3)->Hyb_101 [...]' -> 'Hyb_101'). None when nothing testifies."""
    for key, prov in (cell_dict.get('matrix_provenance') or {}).items():
        k_hybe, k_mod = (key if isinstance(key, tuple) else (key, modality))
        if k_mod != modality:
            continue
        seq = str((prov or {}).get('reference_sequence') or '')
        if '->' in seq:
            target = seq.split('->', 1)[1]
            target = target.split(' [', 1)[0].split('(', 1)[0].strip()
            if target:
                return target
    return None


def reconcile_cell_dicts(cells, resolver):
    """Refresh every cell's matrix_anchors under the CURRENT matrices.

    The mask needs nothing -- it lives in the cell's own reference
    frame, raw-like. But matrix_anchors are alignment-time SNAPSHOTS
    (bridge @ within[anchor_hybe] as of that run), sitting inside every
    cell-route transform; a later FOV or cross-modal refit strands them
    exactly like an adj coordinate. The anchor hybe per modality is
    parsed from the residuals' own provenance; a modality nothing
    testifies for is skipped and counted, never guessed.
    """
    shifts = []
    skipped = {'anchor_hybe_unknown': 0, 'no_within': 0}
    updated = 0
    for d in cells:
        anchors = dict(d.get('matrix_anchors') or {})
        touched = False
        for modality in list(anchors.keys()):
            hybe = _anchor_hybe_of(d, modality)
            if hybe is None:
                skipped['anchor_hybe_unknown'] += 1
                continue
            H_w = (resolver.within.get(modality) or {}).get(hybe)
            if H_w is None:
                skipped['no_within'] += 1
                continue
            bridge = resolver.bridge(modality, resolver.shared)
            if bridge is None:
                bridge = np.eye(3)
            new = np.asarray(bridge, float) @ np.asarray(H_w, float)
            old = np.asarray(anchors[modality], float)
            shifts.append(float(np.hypot(new[0, 2] - old[0, 2],
                                         new[1, 2] - old[1, 2])))
            anchors[modality] = new
            touched = True
        if touched:
            d['matrix_anchors'] = anchors
            updated += 1
    return {'updated': updated, 'skipped': skipped,
            'shift_px': _shift_stats(shifts)}


def reconcile_spot_dicts(spots, resolver, cells_by_id):
    """Recompute adj for spot dicts IN PLACE; returns the stats dict.

    Homeless spots (cell == -1) take the FOV route -- the cross-modal
    term is FOV-bounded and still applies.
    """
    shifts = []
    n_placeholder = 0
    for s in spots:
        ry, rx, rz = s['raw_coordinate']
        cell = cells_by_id.get(int(s.get('cell', -1)))
        y, x = _project_yx(resolver, s['hybe'], s['modality'], cell, ry, rx)
        oy, ox, oz = s['adj_coordinate']
        if float(rz) == 0.0 and float(oz) == 0.0:
            z = 0.0                     # 2D placeholder, never minted into depth
            n_placeholder += 1
        else:
            z = float(rz) + resolver.z_to_shared(s['hybe'], s['modality'], cell)
        shifts.append(float(np.hypot(y - float(oy), x - float(ox))))
        s['adj_coordinate'] = (y, x, z)
    return {'updated': len(spots), 'z_placeholder': n_placeholder,
            'shift_px': _shift_stats(shifts)}


def reconcile_allele_dicts(alleles, resolver, cells_by_id, modality=None,
                           reference_hybe=None):
    """Recompute every adj field of allele dicts IN PLACE; returns stats.

    Per allele, modality and reference hybe come from its PROVENANCE
    when stamped (v2 stamps both since 2026-08-30), else from the
    arguments; an allele that names neither is SKIPPED and counted --
    re-applying a fiducial correction relative to a guessed reference
    would corrupt the polymer, not reconcile it.
    """
    shifts_fid, shifts_poly, shifts_anchor = [], [], []
    skipped = {'no_raw': 0, 'no_reference': 0, 'ref_not_traced': 0}
    updated = 0
    for d in alleles:
        prov = d.get('provenance') or {}
        mod = prov.get('modality') or modality
        ref = prov.get('reference_hybe') or reference_hybe
        fid_raw = d.get('fiducial_trace_raw') or {}
        if not fid_raw and not (d.get('polymer_raw') or {}):
            skipped['no_raw'] += 1
            continue
        if mod is None or ref is None:
            skipped['no_reference'] += 1
            continue
        cell = cells_by_id.get(int(d.get('cell', -1)))

        # anchor
        ry, rx, rz = d['raw_coordinate']
        ah = d.get('anchor_hybe') or ref
        y, x = _project_yx(resolver, ah, mod, cell, ry, rx)
        z = float(rz) + resolver.z_to_shared(ah, mod, cell) \
            if float(rz) != 0.0 else float(d['coordinate'][2])
        oy, ox, _oz = d['coordinate']
        shifts_anchor.append(float(np.hypot(y - float(oy), x - float(ox))))
        d['coordinate'] = (y, x, z)

        # fiducials, raw -> adj under current matrices
        new_fid = dict(d.get('fiducial_trace_adj') or {})
        for h, v in fid_raw.items():
            if v is None:
                new_fid[h] = None
                continue
            fy, fx = _project_yx(resolver, h, mod, cell, v[0], v[1])
            fz = float(v[2]) + resolver.z_to_shared(h, mod, cell)
            old = (d.get('fiducial_trace_adj') or {}).get(h)
            if old is not None:
                shifts_fid.append(float(np.hypot(fy - old[0], fx - old[1])))
            new_fid[h] = (fy, fx, fz, float(v[3]))
        d['fiducial_trace_adj'] = new_fid

        # polymer: project raw, re-apply the ref-relative drift correction
        poly_raw = d.get('polymer_raw') or {}
        if poly_raw:
            base = new_fid.get(ref)
            if base is None:
                skipped['ref_not_traced'] += 1
            else:
                new_poly = dict(d.get('polymer_adj') or {})
                for h, cands in poly_raw.items():
                    fid_h = new_fid.get(h)
                    if fid_h is None:
                        continue    # no fiducial, no correction -- leave as-is
                    dy = base[0] - fid_h[0]
                    dx = base[1] - fid_h[1]
                    dz = base[2] - fid_h[2]
                    out = []
                    for c in cands:
                        cy, cx = _project_yx(resolver, h, mod, cell,
                                             c[0], c[1])
                        cz = float(c[2]) + resolver.z_to_shared(h, mod, cell)
                        old = (d.get('polymer_adj') or {}).get(h)
                        if old:
                            shifts_poly.append(float(np.hypot(
                                cy + dy - old[0][0], cx + dx - old[0][1])))
                        out.append((cy + dy, cx + dx, cz + dz, float(c[3])))
                    new_poly[h] = out
                d['polymer_adj'] = new_poly
        updated += 1
    return {'updated': updated, 'skipped': skipped,
            'shift_px': {'anchor': _shift_stats(shifts_anchor),
                         'fiducial': _shift_stats(shifts_fid),
                         'polymer': _shift_stats(shifts_poly)}}


def reconcile_fov(storage_path, fov, resolver=None, modality=None,
                  reference_hybe=None, write=False):
    """One FOV brought into agreement with the current matrices.

    write=False (default) is a DRY RUN: the report says how far every
    category has drifted, and the store is untouched. write=True
    persists through the atomic store doors.
    """
    if resolver is None:
        from codelab_pipeline.analysis import resolvers as R
        resolver = R.resolver_for(storage_path, fov)
    cells, _ = analysis_store.read_cells(storage_path, fov)
    cells = cells or []

    # CELLS FIRST. Their matrix_anchors are the stalest layer -- every
    # cell-route projection below composes through them, so spots and
    # alleles must see the REFRESHED anchors, not the harvested
    # snapshots (the factory resolver was built from the store's own
    # possibly-stale cells).
    cell_stats = reconcile_cell_dicts(cells, resolver)
    fresh_anchors = {}
    for c in cells:
        for m, H in (c.get('matrix_anchors') or {}).items():
            fresh_anchors.setdefault(m, np.asarray(H, float))
    if fresh_anchors:
        resolver.anchors = fresh_anchors

    cells_by_id = {}
    for c in cells:
        obj = ACell()
        obj.set_metadata(**c)
        cells_by_id[int(c['id'])] = obj

    report = {'fov': int(fov), 'written': bool(write), 'cells': cell_stats}
    if write and cell_stats['updated']:
        analysis_store.write_cell_dicts(storage_path, fov, cells)
    slice_stats, all_shifts = [], []
    for (mod_s, hybe, channel) in analysis_store.spot_slices(storage_path, fov):
        spots = analysis_store.read_spots(storage_path, fov, modality=mod_s,
                                          hybe=hybe, channel=channel)
        if not spots:
            continue
        st = reconcile_spot_dicts(spots, resolver, cells_by_id)
        st['slice'] = (mod_s, hybe, int(channel))
        slice_stats.append(st)
        if write:
            objs = []
            for sd in spots:
                o = ASpot()
                o.set_metadata(**sd)
                o.modality = sd['modality']
                objs.append(o)
            analysis_store.write_spots(storage_path, fov, mod_s, hybe,
                                       int(channel), objs)
        if st['shift_px']['n']:
            all_shifts.extend([st['shift_px']['median']] * st['shift_px']['n'])
    report['spots'] = {
        'slices': len(slice_stats),
        'updated': int(sum(s['updated'] for s in slice_stats)),
        'z_placeholder': int(sum(s['z_placeholder'] for s in slice_stats)),
        'per_slice': slice_stats}

    alleles = analysis_store.read_fov_alleles(storage_path, fov)
    if alleles:
        st = reconcile_allele_dicts(alleles, resolver, cells_by_id,
                                    modality=modality,
                                    reference_hybe=reference_hybe)
        if write:
            analysis_store.write_allele_dicts(storage_path, fov, alleles)
        report['alleles'] = st
    else:
        report['alleles'] = {'updated': 0, 'skipped': {}, 'shift_px': {}}
    return report


def reconcile(storage_path, fovs, modality=None, reference_hybe=None,
              write=False, jobs=None, on_done=None):
    """Every FOV, FOV-major through one pool. Returns per-FOV reports.

    The name means what it says: after alignment moved, the stored adj
    coordinates disagree with the matrices; this makes the store agree
    with itself again -- from raw, which never lies.
    """
    from codelab_pipeline import parallel
    items = [(storage_path, int(f), modality, reference_hybe, bool(write))
             for f in fovs]
    results = parallel.pmap(_reconcile_item, items, kind='io', jobs=jobs,
                            on_done=on_done)
    out = []
    for f, r in zip(fovs, results):
        if isinstance(r, parallel.Failure):
            out.append({'fov': int(f), 'failed': str(r)})
        else:
            out.append(r)
    return out


def _reconcile_item(item):
    storage_path, fov, modality, reference_hybe, write = item
    return reconcile_fov(storage_path, fov, modality=modality,
                         reference_hybe=reference_hybe, write=write)
