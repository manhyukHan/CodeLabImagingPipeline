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
import shutil
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


def test_properties_round_trip():
    """A field the packer does not know must survive the store.

    This is the whole point of the props columns: the schema used to be
    closed, so an analysis could measure something, attach it to a cell,
    save, and find it gone on the next read -- which looks exactly like
    the analysis never having run. Every scalar kind is checked because
    bool IS an int in Python and would come back as 0/1 if the column
    type were picked per value instead of per column.
    """
    import tempfile

    import h5py

    ok = True
    extras = {'n_speckle': 23, 'speckle_frac': 0.064, 'genotype': 'C',
              'is_target': True, 'never_measured': None}
    cases = (
        ('cells', columnar.pack_cells, columnar.unpack_cells,
         {'id': 7, 'fov': 3, 'reference_hybe': 'Hyb_IFdapi',
          'reference_modality': 'DNA', 'nucleus_hybe': 'Hyb_IFdapi',
          'nucleus_modality': 'DNA', 'celltype': '',
          'nucleus': (np.array([1.0]), np.array([2.0])),
          'area': (np.array([1.0]), np.array([2.0])),
          'frame_shape': (16, 16), 'matrices': {}, 'matrix_anchors': {},
          'matrix_provenance': {}, 'distmap': np.array([]),
          'linked': False, 'linked_at': None}),
        ('spots', columnar.pack_spots, columnar.unpack_spots,
         {'uid': 1, 'fov': 3, 'modality': 'DNA', 'hybe': 'Hyb_015',
          'channel': 635, 'cell': 7, 'celltype': '',
          'adj_coordinate': (1.0, 2.0, 3.0),
          'raw_coordinate': (1.0, 2.0, 3.0), 'size': 1.0,
          'brightness': 1.0, 'linked': False, 'linked_at': None,
          'mixture_centroids': ()}),
    )
    d = tempfile.mkdtemp(prefix='props_')
    for name, pack, unpack, base in cases:
        rows = [dict(base, **extras),
                dict(base)]                      # the second has NONE
        rows[1]['uid' if name == 'spots' else 'id'] = 99
        p = os.path.join(d, name + '.h5')
        with h5py.File(p, 'w') as f:
            pack(f.require_group('g'), rows)
        with h5py.File(p, 'r') as f:
            back = unpack(f['g'])
        for k, v in extras.items():
            got = back[0].get(k, '<missing>')
            if v is None:
                good = k not in back[0]
                why = 'a None value must not come back as a value'
            else:
                good = got == v and type(got) is type(v)
                why = 'got %r (%s)' % (got, type(got).__name__)
            print(('  ok   ' if good else '  FAIL ')
                  + '%s: %s survives the store' % (name, k)
                  + ('' if good else '   ' + why))
            ok &= good
        absent = all(k not in back[1] for k in extras if k != 'never_measured')
        print(('  ok   ' if absent else '  FAIL ')
              + '%s: an item without the property does not get one' % name)
        ok &= absent
    shutil.rmtree(d, ignore_errors=True)
    return ok


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
    print('\nproperties (the schema is open now)')
    ok &= test_properties_round_trip()
    print('ALL GOOD' if ok else 'FAILURES')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
