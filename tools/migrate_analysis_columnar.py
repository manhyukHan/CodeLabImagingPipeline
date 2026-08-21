"""
One-shot migration of an analysis store's pickled blobs to the columnar
schema (codelab_pipeline/io/columnar.py).

    python tools/migrate_analysis_columnar.py <vlinks.h5 | storage_path>

Copies <file>.bak alongside first; refuses stores already stamped
columnar. Cells, spots (every slice), and alleles are re-read through
the ordinary doors (so any legacy quirks the readers already normalize
stay normalized) and repacked as typed datasets; the store is stamped
analysis_schema='columnar', after which every future write stays
columnar. The tiny /params celltype-config blob stays pickle by design.
Works on v1 and v2 layouts alike (it only touches the analysis file).
"""
import os
import shutil
import sys

import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codelab_pipeline.io import columnar, paths  # noqa: E402


def main(target):
    vlinks = target if target.endswith('.h5') else paths.vlinks_path(target)
    if not os.path.exists(vlinks):
        raise SystemExit(f'not found: {vlinks}')
    with h5py.File(vlinks, 'r') as f:
        schema = f.attrs.get('analysis_schema', b'')
        schema = schema.decode() if isinstance(schema, bytes) else str(schema)
        if schema == 'columnar':
            raise SystemExit(f'{vlinks} is already analysis_schema=columnar.')
    bak = vlinks + '.bak-columnar'
    shutil.copy2(vlinks, bak)
    print(f'backup: {bak}')

    import pickle
    import numpy as np
    n_c = n_s = n_a = 0
    with h5py.File(vlinks, 'a') as f:
        for fov_name in sorted(k for k in f.keys() if k.startswith('FOV')):
            g = f[fov_name]
            if 'cells' in g and 'blob' in g['cells']:
                dicts = pickle.loads(bytes(g['cells']['blob'][()])).get('cells', [])
                attrs = dict(g['cells'].attrs)
                del f[f'{fov_name}/cells']
                grp = f.require_group(f'{fov_name}/cells')
                columnar.pack_cells(grp, dicts)
                grp.attrs.update(attrs)
                n_c += len(dicts)
            if 'spots' in g:
                for mod in list(g['spots']):
                    for hy in list(g['spots'][mod]):
                        for ch in list(g['spots'][mod][hy]):
                            sg = g['spots'][mod][hy][ch]
                            if 'blob' not in sg:
                                continue
                            dicts = pickle.loads(bytes(sg['blob'][()]))
                            attrs = dict(sg.attrs)
                            path = f'{fov_name}/spots/{mod}/{hy}/{ch}'
                            del f[path]
                            grp = f.require_group(path)
                            columnar.pack_spots(grp, dicts)
                            grp.attrs.update(attrs)
                            n_s += len(dicts)
            if 'alleles' in g and 'blob' in g['alleles']:
                dicts = pickle.loads(bytes(g['alleles']['blob'][()]))
                dicts = [{'anchor_uid': 0, **d} for d in dicts]
                attrs = dict(g['alleles'].attrs)
                del f[f'{fov_name}/alleles']
                grp = f.require_group(f'{fov_name}/alleles')
                columnar.pack_alleles(grp, dicts)
                grp.attrs.update(attrs)
                n_a += len(dicts)
        f.attrs['analysis_schema'] = 'columnar'
    before, after = os.path.getsize(bak), os.path.getsize(vlinks)
    print(f'{vlinks}: {n_c} cells, {n_s} spots, {n_a} alleles repacked columnar; '
          f'{before/1e6:.1f} MB -> {after/1e6:.1f} MB (h5repack recommended to reclaim '
          f'freed space fully).')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
