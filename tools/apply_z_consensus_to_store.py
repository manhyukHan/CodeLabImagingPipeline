# -*- coding: utf-8 -*-
"""Fill a stored cell alignment's refused z fits with the consensus depth.

`chain.apply_z_consensus` does this inline for new runs. A store aligned
before it existed holds `dz = 0.0` wherever the per-cell z fit was
refused -- which asserts "no depth correction needed" in rounds where
every other cell was moved tens of planes. On MP58 FOV 1 that leaves
12-18 % of the cells of Hyb_021-025 and Hyb_041-049 between 38 and 48
planes (7.6-9.6 um) out of place.

RE-FITTING IS NOT NEEDED. The consensus is the median of the ACCEPTED
depths, and those are already in the store; the refused entries are
already identifiable, because `_cell_hybe_task` wrote its reason into
`matrix_provenance[...]['reference_sequence']` as
`[z-alignment REJECTED: ...]`. So this reads what is there and fills in
the fallback, leaving every measured value untouched.

The z bound itself is NOT changed. A fit refused at 55 planes stays
refused; only the value written in its place changes, from a zero that
is known to be wrong to an estimate that is known to be close.

Writes `cells.h5` per FOV, after copying it to
`cells_before_z_consensus_<YYYYMMDD>.h5` beside it.

    python tools/apply_z_consensus_to_store.py <storage_path> [--fovs 1,2]
                                               [--apply] [--min-cells 8]

Without --apply it reports and writes nothing.
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline.io import analysis_store as AS          # noqa: E402

MARK = '[z-alignment REJECTED'
DONE = '[z CONSENSUS applied'


def fov_list(storage_path, spec):
    if spec:
        return [int(v) for v in spec.replace(',', ' ').split()]
    import glob
    from codelab_pipeline.io import paths
    out = []
    for d in glob.glob(os.path.join(paths.analysis_dir(storage_path), 'fov*')):
        tail = os.path.basename(d)[3:]
        if tail.isdigit():
            out.append(int(tail))
    return sorted(out)


def plan_fov(cells, min_cells):
    """-> {(hybe, modality): (dz, n_accepted, [(cell_index, key)])}"""
    accepted, refused = {}, {}
    for i, c in enumerate(cells):
        prov = c.get('matrix_provenance') or {}
        for key, entry in (c.get('matrices') or {}).items():
            note = str((prov.get(key) or {}).get('reference_sequence') or '')
            if DONE in note:
                continue
            if MARK in note:
                refused.setdefault(key, []).append((i, key))
            else:
                dz = float(entry.get('dz', 0.0))
                if np.isfinite(dz):
                    accepted.setdefault(key, []).append(dz)
    out = {}
    for key, rows in refused.items():
        ok = accepted.get(key) or []
        if len(ok) < min_cells:
            continue
        dz = float(np.median(ok))
        if dz == 0.0:
            continue
        out[key] = (dz, len(ok), rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('storage_path')
    ap.add_argument('--fovs', default='')
    ap.add_argument('--min-cells', type=int, default=8)
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()

    stamp = time.strftime('%Y%m%d')
    total_filled = total_cells = 0
    for fov in fov_list(a.storage_path, a.fovs):
        cells, _ = AS.read_cells(a.storage_path, fov)
        if not cells:
            continue
        plan = plan_fov(cells, a.min_cells)
        n = sum(len(v[2]) for v in plan.values())
        total_filled += n
        total_cells += len(cells)
        if not plan:
            print('fov %-3d  %3d cells  nothing to fill' % (fov, len(cells)))
            continue
        worst = sorted(plan.items(), key=lambda kv: -abs(kv[1][0]))[:3]
        print('fov %-3d  %3d cells  %4d entries over %2d (hybe, modality); '
              'largest: %s' % (fov, len(cells), n, len(plan),
                               ', '.join('%s %+.0f (%d ok)'
                                         % (k[0], v[0], v[1])
                                         for k, v in worst)))
        if not a.apply:
            continue
        for key, (dz, n_ok, rows) in plan.items():
            for i, k in rows:
                cells[i]['matrices'][k]['dz'] = dz
                prov = cells[i].setdefault('matrix_provenance', {}).setdefault(
                    k, {})
                prov['reference_sequence'] = str(
                    prov.get('reference_sequence', '')) + (
                    f' {DONE}: {dz:+.1f}px from {n_ok} accepted cells of this '
                    f'hybe in this FOV, replacing the 0.0px fallback -- '
                    f'IMPUTED, not measured]')
        src = AS.cells_path(a.storage_path, fov)
        bak = os.path.join(os.path.dirname(src),
                           'cells_before_z_consensus_%s.h5' % stamp)
        if not os.path.exists(bak):
            shutil.copy2(src, bak)
        AS.write_cell_dicts(a.storage_path, fov, cells)

    print('\n%d (cell, hybe) entries over %d cells'
          % (total_filled, total_cells))
    print('APPLIED, originals copied to cells_before_z_consensus_%s.h5'
          % stamp if a.apply else 'DRY RUN -- nothing written; pass --apply')


if __name__ == '__main__':
    main()
