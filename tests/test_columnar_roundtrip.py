"""
Columnar serialization fidelity, on REAL data: every cells/spots/alleles
dict from an existing store must survive pack->unpack field-for-field
(the two documented exceptions: distmap reads back empty; geometry
arrays come back float64).

CODELAB_COL_STORE selects the source store (default: the real one,
read-only -- this test never writes to any store; packing happens in an
in-memory HDF5 file).

Run: python tests/test_columnar_roundtrip.py
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np

from codelab_pipeline.io import analysis_store as V
from codelab_pipeline.io import columnar

STORE = os.environ.get('CODELAB_COL_STORE', 'data/chr19_downstream_new/RNA_queue')
FOV = 1


def roundtrip(pack, unpack, dicts):
    bio = io.BytesIO()
    with h5py.File(bio, 'w') as f:
        pack(f.create_group('g'), dicts)
    with h5py.File(bio, 'r') as f:
        back = unpack(f['g'])
    return back, bio.getbuffer().nbytes


def compare(a, b, path=''):
    """Recursive equality with numeric tolerance; returns list of diffs."""
    diffs = []
    if isinstance(a, dict):
        if set(a) != set(b):
            return [f'{path}: keys {sorted(map(str, set(a) ^ set(b)))}']
        for k in a:
            if k == 'distmap':
                continue                      # documented drop
            diffs += compare(a[k], b[k], f'{path}.{k}')
    elif isinstance(a, (list, tuple)) and not isinstance(a, str):
        if len(a) != len(b):
            return [f'{path}: len {len(a)} vs {len(b)}']
        for i, (x, y) in enumerate(zip(a, b)):
            diffs += compare(x, y, f'{path}[{i}]')
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if np.asarray(a).shape != np.asarray(b).shape:
            diffs.append(f'{path}: shape {np.asarray(a).shape} vs {np.asarray(b).shape}')
        elif np.asarray(a).size and not np.allclose(np.asarray(a, dtype=float),
                                                    np.asarray(b, dtype=float), atol=1e-9):
            diffs.append(f'{path}: values differ')
    elif isinstance(a, float) or isinstance(b, float):
        if abs(float(a) - float(b)) > 1e-9:
            diffs.append(f'{path}: {a} vs {b}')
    elif a != b:
        diffs.append(f'{path}: {a!r} vs {b!r}')
    return diffs


def _run(kind, dicts, pack, unpack, blob_hint=''):
    back, packed_bytes = roundtrip(pack, unpack, dicts)
    diffs = compare(dicts, back, kind)
    for d in diffs[:6]:
        print(f'  DIFF {d}')
    status = 'PASS' if not diffs else 'FAIL'
    print(f'  {status}  {kind}: {len(dicts)} items round-trip exactly '
          f'(packed {packed_bytes/1e3:.0f} KB{blob_hint})')
    return not diffs


def main():
    ok = True
    cells, _ = V.read_cells(STORE, FOV)
    if cells:
        ok &= _run('cells', cells, columnar.pack_cells, columnar.unpack_cells)
    spots = V.read_spots(STORE, FOV)
    if spots:
        ok &= _run('spots', spots, columnar.pack_spots, columnar.unpack_spots)
    alleles = V.read_fov_alleles(STORE, FOV)
    if alleles:
        # legacy allele dicts predate anchor_uid; the canonical shape
        # always carries it (default 0), so normalize before comparing
        alleles = [{'anchor_uid': 0, **d} for d in alleles]
        ok &= _run('alleles', alleles, columnar.pack_alleles, columnar.unpack_alleles)
    print('ALL GOOD' if ok else 'FAILURES')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
