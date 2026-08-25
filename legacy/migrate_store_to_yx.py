"""
One-shot migration of a vlinks.h5 store to the rasterized (y, x)
convention (codelab_pipeline/alignment/convention.py).

    python tools/migrate_store_to_yx.py <vlinks.h5 | storage_path>

What it does, in place (after copying <file>.bak alongside):
  - every (3, 3) float dataset (same-modality /matrix entries, the
    cross-modal matrix_across) is conjugated H -> P @ H @ P;
  - cell blobs: area/nucleus tuples swap to (y, x); matrices entries'
    'yx' conjugated (legacy 3x3 'zx' collapsed to the scalar 'dz' it
    always was); matrix_anchors conjugated; provenance step stacks
    conjugated;
  - spot blobs: coordinate/raw_coordinate -> (y, x, z);
    mixture_centroids entries swap their first two components;
  - allele blobs: coordinate/raw_coordinate, fiducial_trace, polymer
    candidates swap their first two components; final_polymer columns
    reordered (y, x, z);
  - the store is stamped coordinate_order='yx'. Already-stamped stores
    are refused (never double-swap).
"""
import os
import pickle
import shutil
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codelab_pipeline.alignment.convention import to_yx  # noqa: E402


def swap2(t):
    t = tuple(t)
    return (t[1], t[0]) + t[2:]


def migrate_cell(d):
    for key in ('area', 'nucleus'):
        if d.get(key) is not None:
            a, b = d[key]
            d[key] = (np.asarray(b), np.asarray(a))
    mats = {}
    for k, entry in (d.get('matrices') or {}).items():
        entry = dict(entry)
        if entry.get('yx') is not None:
            entry['yx'] = to_yx(entry['yx'])
        if 'zx' in entry and 'dz' not in entry:
            zx = np.asarray(entry.pop('zx'))
            entry['dz'] = float(zx[0, 2])
        mats[k] = entry
    if mats:
        d['matrices'] = mats
    if d.get('matrix_anchors'):
        d['matrix_anchors'] = {m: to_yx(H) for m, H in d['matrix_anchors'].items()}
    prov = {}
    for k, p in (d.get('matrix_provenance') or {}).items():
        p = dict(p)
        if p.get('steps') is not None:
            p['steps'] = np.stack([to_yx(H) for H in np.asarray(p['steps'])])
        prov[k] = p
    if prov:
        d['matrix_provenance'] = prov
    return d


def migrate_spot(d):
    for key in ('coordinate', 'raw_coordinate'):
        if d.get(key) is not None:
            d[key] = swap2(d[key])
    if d.get('mixture_centroids'):
        d['mixture_centroids'] = tuple(swap2(c) for c in d['mixture_centroids'])
    return d


def migrate_allele(d):
    for key in ('coordinate', 'raw_coordinate'):
        if d.get(key) is not None:
            d[key] = swap2(d[key])
    if d.get('fiducial_trace'):
        d['fiducial_trace'] = {h: (swap2(v) if v is not None else None)
                               for h, v in d['fiducial_trace'].items()}
    if d.get('polymer'):
        d['polymer'] = {h: [swap2(c) for c in v] for h, v in d['polymer'].items()}
    fp = np.asarray(d.get('final_polymer', []))
    if fp.size:
        d['final_polymer'] = fp[:, [1, 0, 2]]
    return d


def _rewrite_blob(f, path, per_item, wrapper_key=None):
    """Blobs come in two shapes: a plain list of dicts (spots, alleles)
    or a {wrapper_key: [dicts], ...} envelope (cells)."""
    raw = bytes(f[path]['blob'][()])
    payload = pickle.loads(raw)
    if isinstance(payload, dict) and wrapper_key is not None:
        payload = dict(payload)
        payload[wrapper_key] = [per_item(dict(i)) for i in payload.get(wrapper_key, [])]
        n = len(payload[wrapper_key])
    else:
        payload = [per_item(dict(i)) for i in payload]
        n = len(payload)
    del f[path]['blob']
    f[path].create_dataset('blob', data=np.void(pickle.dumps(payload)))
    return n


def main(target):
    vlinks = target if target.endswith('.h5') else os.path.join(
        os.path.dirname(os.path.abspath(target).rstrip(os.sep)), 'vlinks.h5')
    if not os.path.exists(vlinks):
        raise SystemExit(f'not found: {vlinks}')
    with h5py.File(vlinks, 'r') as f:
        if f.attrs.get('coordinate_order') == 'yx':
            raise SystemExit(f'{vlinks} is already coordinate_order=yx -- refusing to double-swap.')
    bak = vlinks + '.bak'
    shutil.copy2(vlinks, bak)
    print(f'backup: {bak}')

    n_mats = n_cells = n_spots = n_alleles = 0
    with h5py.File(vlinks, 'a') as f:
        mat_paths, cell_paths, spot_paths, allele_paths = [], [], [], []

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                if obj.shape == (3, 3) and np.issubdtype(obj.dtype, np.floating):
                    mat_paths.append(name)
                elif name.endswith('/blob'):
                    parent = name.rsplit('/', 1)[0]
                    if '/spots/' in f'/{parent}/':
                        spot_paths.append(parent)
                    elif parent.endswith('/cells'):
                        cell_paths.append(parent)
                    elif parent.endswith('/alleles'):
                        allele_paths.append(parent)
        f.visititems(visit)

        for name in mat_paths:
            f[name][...] = to_yx(f[name][()])
            n_mats += 1
        for parent in cell_paths:
            n_cells += _rewrite_blob(f, parent, migrate_cell, wrapper_key='cells')
        for parent in spot_paths:
            n_spots += _rewrite_blob(f, parent, migrate_spot)
        for parent in allele_paths:
            n_alleles += _rewrite_blob(f, parent, migrate_allele)
        f.attrs['coordinate_order'] = 'yx'

    print(f'{vlinks}: {n_mats} matrices conjugated, {n_cells} cells, '
          f'{n_spots} spots, {n_alleles} alleles swapped; stamped yx.')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
