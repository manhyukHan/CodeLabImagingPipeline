"""
Columnar (typed-dataset) serialization for the analysis store -- the
phase-2 replacement for the pickled per-FOV blobs.

Why: a pickled blob must be read, unpickled, and rewritten WHOLE to
touch anything inside it (the real FOV cells blob measured 12.8 MB --
float64 coordinate arrays -- rewritten on every save and re-read on
every status refresh, over NAS); nothing inside is partially readable;
and pickle couples the store to Python class details. Here every field
is a plain typed HDF5 dataset: geometry lands as compressed uint16
columns (~25x smaller), scalars as one structured table, and ragged
per-item data (mixture centroids, traces, provenance steps) as
flat-array + offsets.

The model classes are UNTOUCHED: pack_* consume exactly the
ACell/ASpot/AnAllele.save() dicts, unpack_* return dicts of the same
shape, so CellContainer.load / the spot hydrators / allele loaders work
unchanged. vlinks_store picks pickle vs columnar per store (see its
analysis_schema stamp); this module never opens files itself -- it
packs into / unpacks from an already-open h5py group.

Fidelity contract, pinned by tests/test_columnar_roundtrip.py on REAL
data: unpack(pack(dicts)) equals the original dicts field-for-field
(coordinates as tuples, geometry as (y_array, x_array) pairs, matrices
keyed by (hybe, modality) tuples). Two deliberate exceptions:
'distmap' is dropped on write and read back as an empty array (it is a
legacy slot that stays empty by design), and geometry arrays come back
float64 regardless of stored width.
"""
import json

import h5py
import numpy as np

_STR = 'S64'          # fixed small strings (hybe/modality/celltype names)
# Provenance is JSON of arbitrary length, so it CANNOT use _STR: a real
# entry is ~113 chars and S64 would truncate it into invalid JSON that
# only fails at read time. Variable-length, measured at 10k alleles x 100
# hybes: pack -0.0%, unpack +4.7%, file size +0.8%.
_JSON = h5py.string_dtype(encoding='utf-8')


def _s(v):
    return str(v).encode() if v is not None else b''


def _rd(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def _write(grp, name, data, compress=False):
    if name in grp:
        del grp[name]
    kw = dict(compression='gzip', compression_opts=1, shuffle=True) if compress and np.asarray(data).size else {}
    grp.create_dataset(name, data=data, **kw)


def _coords_pack(grp, name, pairs):
    """pairs: [(a_array, b_array), ...] per item -> <name> (N,2) + offsets.
    uint16 when integral and in range (real masks always are -- pixel
    indices from np.where), float32 otherwise."""
    flat, off = [], [0]
    for a, b in pairs:
        a = np.asarray(a, dtype=float).ravel()
        b = np.asarray(b, dtype=float).ravel()
        flat.append(np.stack([a, b], axis=1) if a.size else np.empty((0, 2)))
        off.append(off[-1] + a.size)
    flat = np.concatenate(flat, axis=0) if flat else np.empty((0, 2))
    integral = flat.size == 0 or (np.all(flat >= 0) and np.all(flat < 65536)
                                  and np.allclose(flat, np.round(flat)))
    _write(grp, name, flat.astype(np.uint16 if integral else np.float32), compress=True)
    _write(grp, name + '_off', np.asarray(off, dtype=np.int64))


def _coords_load(grp, name):
    """Bulk-load a coords dataset ONCE -- per-item h5py slicing costs one
    HDF5 call each and measured ~1s for a 101-cell FOV; in-memory numpy
    slices are microseconds."""
    return grp[name][()].astype(np.float64), grp[name + '_off'][()]


def _coords_at(flat_off, i):
    flat, off = flat_off
    seg = flat[off[i]:off[i + 1]]
    return (seg[:, 0].copy(), seg[:, 1].copy())


# -- cells ---------------------------------------------------------------

def pack_cells(grp, dicts):
    n = len(dicts)
    tab = np.zeros(n, dtype=[('id', 'i4'), ('fov', 'i4'), ('linked', 'u1'),
                             ('fs0', 'i4'), ('fs1', 'i4')])
    strs = {k: [] for k in ('reference_hybe', 'reference_modality', 'nucleus_hybe',
                            'nucleus_modality', 'celltype', 'linked_at')}
    mat = {'cell': [], 'hybe': [], 'modality': [], 'H': [], 'dz': [], 'residual': []}
    anc = {'cell': [], 'modality': [], 'H': []}
    prov = {'cell': [], 'hybe': [], 'modality': [], 'refseq': [], 'steps': [], 'n_steps': []}
    for i, d in enumerate(dicts):
        tab[i] = (d['id'], d['fov'], bool(d.get('linked', False)),
                  d['frame_shape'][0], d['frame_shape'][1])
        for k in strs:
            strs[k].append(_s(d.get(k)))
        for (hybe, modality), m in (d.get('matrices') or {}).items():
            mat['cell'].append(i); mat['hybe'].append(_s(hybe)); mat['modality'].append(_s(modality))
            mat['H'].append(np.asarray(m['yx'], dtype=np.float64))
            mat['dz'].append(float(m.get('dz', 0.0))); mat['residual'].append(bool(m.get('yx_is_residual', False)))
        for modality, H in (d.get('matrix_anchors') or {}).items():
            anc['cell'].append(i); anc['modality'].append(_s(modality))
            anc['H'].append(np.asarray(H, dtype=np.float64))
        for (hybe, modality), p in (d.get('matrix_provenance') or {}).items():
            steps = np.asarray(p.get('steps', np.empty((0, 3, 3))), dtype=np.float64)
            prov['cell'].append(i); prov['hybe'].append(_s(hybe)); prov['modality'].append(_s(modality))
            prov['refseq'].append(_s(p.get('reference_sequence')))
            prov['steps'].append(steps.reshape(-1, 3, 3)); prov['n_steps'].append(len(steps))
    _write(grp, 'table', tab)
    for k, v in strs.items():
        _write(grp, k, np.asarray(v, dtype=_STR))
    _coords_pack(grp, 'area', [d['area'] for d in dicts])
    _coords_pack(grp, 'nucleus', [d['nucleus'] for d in dicts])
    _write(grp, 'mat_cell', np.asarray(mat['cell'], dtype=np.int32))
    _write(grp, 'mat_hybe', np.asarray(mat['hybe'], dtype=_STR))
    _write(grp, 'mat_modality', np.asarray(mat['modality'], dtype=_STR))
    _write(grp, 'mat_H', np.asarray(mat['H']).reshape(-1, 3, 3))
    _write(grp, 'mat_dz', np.asarray(mat['dz'], dtype=np.float64))
    _write(grp, 'mat_residual', np.asarray(mat['residual'], dtype=np.uint8))
    _write(grp, 'anc_cell', np.asarray(anc['cell'], dtype=np.int32))
    _write(grp, 'anc_modality', np.asarray(anc['modality'], dtype=_STR))
    _write(grp, 'anc_H', np.asarray(anc['H']).reshape(-1, 3, 3))
    _write(grp, 'prov_cell', np.asarray(prov['cell'], dtype=np.int32))
    _write(grp, 'prov_hybe', np.asarray(prov['hybe'], dtype=_STR))
    _write(grp, 'prov_modality', np.asarray(prov['modality'], dtype=_STR))
    _write(grp, 'prov_refseq', np.asarray(prov['refseq'], dtype='S1024'))
    _write(grp, 'prov_steps',
           np.concatenate(prov['steps'], axis=0) if prov['steps'] else np.empty((0, 3, 3)))
    _write(grp, 'prov_nsteps', np.asarray(prov['n_steps'], dtype=np.int32))


def unpack_cells(grp):
    # every dataset loaded ONCE, sliced in memory (see _coords_load)
    tab = grp['table'][()]
    n = len(tab)
    strs = {k: grp[k][()] for k in ('reference_hybe', 'reference_modality', 'nucleus_hybe',
                                    'nucleus_modality', 'celltype', 'linked_at')}
    area, nucleus = _coords_load(grp, 'area'), _coords_load(grp, 'nucleus')
    dicts = []
    for i in range(n):
        la = _rd(strs['linked_at'][i])
        dicts.append({
            'id': int(tab['id'][i]), 'fov': int(tab['fov'][i]),
            'reference_hybe': _rd(strs['reference_hybe'][i]),
            'reference_modality': _rd(strs['reference_modality'][i]),
            'nucleus': _coords_at(nucleus, i),
            'nucleus_hybe': _rd(strs['nucleus_hybe'][i]),
            'nucleus_modality': _rd(strs['nucleus_modality'][i]),
            'celltype': _rd(strs['celltype'][i]),
            'area': _coords_at(area, i),
            'frame_shape': (int(tab['fs0'][i]), int(tab['fs1'][i])),
            'matrices': {}, 'matrix_anchors': {}, 'matrix_provenance': {},
            'distmap': np.array([]),
            'linked': bool(tab['linked'][i]),
            'linked_at': la if la else None})
    mat_cell, mat_hybe, mat_mod = grp['mat_cell'][()], grp['mat_hybe'][()], grp['mat_modality'][()]
    mat_H, mat_dz, mat_res = grp['mat_H'][()], grp['mat_dz'][()], grp['mat_residual'][()]
    for j, ci in enumerate(mat_cell):
        dicts[ci]['matrices'][(_rd(mat_hybe[j]), _rd(mat_mod[j]))] = {
            'yx': mat_H[j], 'dz': float(mat_dz[j]), 'yx_is_residual': bool(mat_res[j])}
    anc_cell, anc_mod, anc_H = grp['anc_cell'][()], grp['anc_modality'][()], grp['anc_H'][()]
    for j, ci in enumerate(anc_cell):
        dicts[ci]['matrix_anchors'][_rd(anc_mod[j])] = anc_H[j]
    prov_cell, prov_hybe, prov_mod = grp['prov_cell'][()], grp['prov_hybe'][()], grp['prov_modality'][()]
    prov_ref, prov_ns, steps_all = grp['prov_refseq'][()], grp['prov_nsteps'][()], grp['prov_steps'][()]
    k = 0
    for j, ci in enumerate(prov_cell):
        ns = int(prov_ns[j])
        dicts[ci]['matrix_provenance'][(_rd(prov_hybe[j]), _rd(prov_mod[j]))] = {
            'reference_sequence': _rd(prov_ref[j]), 'steps': steps_all[k:k + ns]}
        k += ns
    return dicts


# -- spots (one slice group) --------------------------------------------

def pack_spots(grp, dicts):
    n = len(dicts)
    tab = np.zeros(n, dtype=[('uid', 'i8'), ('fov', 'i4'), ('channel', 'i4'), ('cell', 'i4'),
                             ('y', 'f8'), ('x', 'f8'), ('z', 'f8'),
                             ('ry', 'f8'), ('rx', 'f8'), ('rz', 'f8'),
                             ('size', 'f8'), ('brightness', 'f8'), ('linked', 'u1')])
    strs = {k: [] for k in ('modality', 'hybe', 'celltype', 'linked_at')}
    mix, mix_off = [], [0]
    for i, d in enumerate(dicts):
        c, r = d['adj_coordinate'], d['raw_coordinate']
        tab[i] = (d['uid'], d['fov'], d['channel'], d['cell'],
                  c[0], c[1], c[2], r[0], r[1], r[2],
                  d.get('size', 0.0), d.get('brightness', 0.0), bool(d.get('linked', False)))
        for k in strs:
            strs[k].append(_s(d.get(k)))
        # centroids are variable-width on real data (legacy 3-tuples
        # without amplitude, current 4-tuples) -- widths are preserved
        cents = d.get('mixture_centroids') or ()
        for c in cents:
            mix.append(np.asarray(c, dtype=np.float64))
        mix_off.append(mix_off[-1] + len(cents))
    _write(grp, 'table', tab, compress=True)
    for k, v in strs.items():
        _write(grp, k, np.asarray(v, dtype=_STR))
    widths = np.asarray([len(c) for c in mix], dtype=np.int8)
    _write(grp, 'mix_vals', np.concatenate(mix) if mix else np.empty(0))
    _write(grp, 'mix_width', widths)
    _write(grp, 'mix_off', np.asarray(mix_off, dtype=np.int64))


def unpack_spots(grp):
    tab = grp['table'][()]
    strs = {k: grp[k][()] for k in ('modality', 'hybe', 'celltype', 'linked_at')}
    mvals, mwidth, off = grp['mix_vals'][()], grp['mix_width'][()], grp['mix_off'][()]
    mstart = np.concatenate([[0], np.cumsum(mwidth)])
    out = []
    for i in range(len(tab)):
        la = _rd(strs['linked_at'][i])
        out.append({
            'uid': int(tab['uid'][i]), 'fov': int(tab['fov'][i]),
            'modality': _rd(strs['modality'][i]), 'hybe': _rd(strs['hybe'][i]),
            'channel': int(tab['channel'][i]), 'cell': int(tab['cell'][i]),
            'celltype': _rd(strs['celltype'][i]),
            'adj_coordinate': (float(tab['y'][i]), float(tab['x'][i]), float(tab['z'][i])),
            'raw_coordinate': (float(tab['ry'][i]), float(tab['rx'][i]), float(tab['rz'][i])),
            'size': float(tab['size'][i]), 'brightness': float(tab['brightness'][i]),
            'linked': bool(tab['linked'][i]),
            'linked_at': la if la else None,
            'mixture_centroids': tuple(tuple(mvals[mstart[j]:mstart[j + 1]])
                                       for j in range(off[i], off[i + 1]))})
    return out


# -- alleles -------------------------------------------------------------

def pack_alleles(grp, dicts):
    n = len(dicts)
    tab = np.zeros(n, dtype=[('id', 'i4'), ('fov', 'i4'), ('cell', 'i4'),
                             ('anchor_uid', 'i8'), ('anchor_channel', 'i4'),
                             ('y', 'f8'), ('x', 'f8'), ('z', 'f8'),
                             ('ry', 'f8'), ('rx', 'f8'), ('rz', 'f8'), ('linked', 'u1')])
    strs = {k: [] for k in ('anchor_hybe', 'linked_at')}
    prov = []
    # tr_*/pl_* KEEP THEIR MEANING: in every file ever written they held
    # the shared-frame values, which is exactly what _adj now names. So
    # old stores stay readable with no translation, and the hybe-native
    # values get NEW columns (trr_*/plr_*) whose absence means "written
    # before raw was recorded", not "empty".
    #
    # The raw columns carry their OWN allele/hybe index rather than
    # riding on the adj rows. Parity between the two dicts is not
    # guaranteed -- v1 fills adj and no raw at all -- and an index that
    # silently assumes it would mis-assign every hybe the moment one
    # engine wrote a different set.
    tr = {'allele': [], 'hybe': [], 'isnone': [], 'vals': []}
    trr = {'allele': [], 'hybe': [], 'isnone': [], 'vals': []}
    pl = {'allele': [], 'hybe': [], 'vals': []}
    plr = {'allele': [], 'hybe': [], 'vals': []}
    rj = {'allele': [], 'hybe': [], 'reason': []}
    fp, fp_off = [], [0]
    for i, d in enumerate(dicts):
        c, r = d['coordinate'], d['raw_coordinate']
        tab[i] = (d['id'], d['fov'], d['cell'], d.get('anchor_uid', 0), d['anchor_channel'],
                  c[0], c[1], c[2], r[0], r[1], r[2], bool(d.get('linked', False)))
        for k in strs:
            strs[k].append(_s(d.get(k)))
        prov.append(json.dumps(d.get('provenance') or {}, sort_keys=True))
        for col, key in ((tr, 'fiducial_trace_adj'), (trr, 'fiducial_trace_raw')):
            for hybe, v in (d.get(key) or {}).items():
                col['allele'].append(i); col['hybe'].append(_s(hybe))
                col['isnone'].append(v is None)
                col['vals'].append(np.zeros(4) if v is None
                                   else np.asarray(v, dtype=np.float64))
        for col, key in ((pl, 'polymer_adj'), (plr, 'polymer_raw')):
            for hybe, cands in (d.get(key) or {}).items():
                for cand in cands:
                    col['allele'].append(i); col['hybe'].append(_s(hybe))
                    col['vals'].append(np.asarray(cand, dtype=np.float64))
        for hybe, reason in (d.get('rejected_hybes') or {}).items():
            rj['allele'].append(i); rj['hybe'].append(_s(hybe)); rj['reason'].append(_s(reason))
        f = np.asarray(d.get('final_polymer', np.empty((0, 3))), dtype=np.float64).reshape(-1, 3)
        fp.append(f); fp_off.append(fp_off[-1] + len(f))
    _write(grp, 'table', tab)
    for k, v in strs.items():
        _write(grp, k, np.asarray(v, dtype=_STR))
    grp.create_dataset('provenance', data=np.asarray(prov, dtype=object), dtype=_JSON)
    for pre, col in (('tr', tr), ('trr', trr)):
        _write(grp, pre + '_allele', np.asarray(col['allele'], dtype=np.int32))
        _write(grp, pre + '_hybe', np.asarray(col['hybe'], dtype=_STR))
        _write(grp, pre + '_isnone', np.asarray(col['isnone'], dtype=np.uint8))
        _write(grp, pre + '_vals', np.asarray(col['vals']).reshape(-1, 4)
               if col['vals'] else np.empty((0, 4)))
    for pre, col in (('pl', pl), ('plr', plr)):
        _write(grp, pre + '_allele', np.asarray(col['allele'], dtype=np.int32))
        _write(grp, pre + '_hybe', np.asarray(col['hybe'], dtype=_STR))
        _write(grp, pre + '_vals', np.asarray(col['vals']).reshape(-1, 4)
               if col['vals'] else np.empty((0, 4)))
    _write(grp, 'rj_allele', np.asarray(rj['allele'], dtype=np.int32))
    _write(grp, 'rj_hybe', np.asarray(rj['hybe'], dtype=_STR))
    _write(grp, 'rj_reason', np.asarray(rj['reason'], dtype='S256'))
    _write(grp, 'fp', np.concatenate(fp, axis=0) if fp else np.empty((0, 3)))
    _write(grp, 'fp_off', np.asarray(fp_off, dtype=np.int64))


def unpack_alleles(grp):
    tab = grp['table'][()]
    strs = {k: grp[k][()] for k in ('anchor_hybe', 'linked_at')}
    # TOLERATE ITS ABSENCE. Every alleles.h5 written before provenance
    # existed has no such column, and those files are still perfectly good
    # traces -- they simply do not know how they were made. Empty dict is
    # the honest answer for them, not an error.
    prov = (grp['provenance'].asstr()[:] if 'provenance' in grp
            else [''] * len(tab))
    fp, fpo = grp['fp'][()], grp['fp_off'][()]
    out = []
    for i in range(len(tab)):
        la = _rd(strs['linked_at'][i])
        out.append({
            'id': int(tab['id'][i]), 'fov': int(tab['fov'][i]), 'cell': int(tab['cell'][i]),
            'anchor_uid': int(tab['anchor_uid'][i]),
            'anchor_hybe': _rd(strs['anchor_hybe'][i]),
            'anchor_channel': int(tab['anchor_channel'][i]),
            'coordinate': (float(tab['y'][i]), float(tab['x'][i]), float(tab['z'][i])),
            'raw_coordinate': (float(tab['ry'][i]), float(tab['rx'][i]), float(tab['rz'][i])),
            'fiducial_trace_adj': {}, 'fiducial_trace_raw': {},
            'polymer_adj': {}, 'polymer_raw': {}, 'rejected_hybes': {},
            # empty comes back shape-(0,) exactly as AnAllele.save()
            # produces it (np.array([]) of an empty polymer)
            'final_polymer': (fp[fpo[i]:fpo[i + 1]] if fpo[i + 1] > fpo[i]
                              else np.array([])),
            'provenance': (json.loads(prov[i]) if prov[i] else {}),
            'linked': bool(tab['linked'][i]),
            'linked_at': la if la else None})
    # trr_*/plr_* ABSENT means the file predates the raw fields, exactly
    # as a missing provenance column does -- an honest empty, not an error.
    for pre, key in (('tr', 'fiducial_trace_adj'), ('trr', 'fiducial_trace_raw')):
        if pre + '_allele' not in grp:
            continue
        vals = grp[pre + '_vals'][()]
        hybe = grp[pre + '_hybe'][()]
        none = grp[pre + '_isnone'][()]
        for j, ai in enumerate(grp[pre + '_allele'][()]):
            out[ai][key][_rd(hybe[j])] = None if none[j] else tuple(vals[j])
    for pre, key in (('pl', 'polymer_adj'), ('plr', 'polymer_raw')):
        if pre + '_allele' not in grp:
            continue
        vals = grp[pre + '_vals'][()]
        hybe = grp[pre + '_hybe'][()]
        for j, ai in enumerate(grp[pre + '_allele'][()]):
            out[ai][key].setdefault(_rd(hybe[j]), []).append(tuple(vals[j]))
    rj_hybe, rj_reason = grp['rj_hybe'][()], grp['rj_reason'][()]
    for j, ai in enumerate(grp['rj_allele'][()]):
        out[ai]['rejected_hybes'][_rd(rj_hybe[j])] = _rd(rj_reason[j])
    return out
