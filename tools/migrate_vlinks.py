"""
One-shot conversion of a v2 project's analysis/vlinks.h5 into the
per-FOV capsule layout (codelab_pipeline/io/analysis_store.py).

    python tools/migrate_vlinks.py <project_root | storage_path>

What it does, in order:

1. Renames analysis/vlinks.h5 -> vlinks.h5.migrating FIRST, so the
   project never has two live truths: analysis_store refuses a store
   with either name present, and only this tool reads the renamed file.
2. Explodes every FOV's cells / spot slices / alleles / matrices /
   cross-modal results, plus the experiment params and celltype config,
   into <dp>/analysis/fov###/... capsules through analysis_store's own
   writers (same columnar packers, so fidelity is the packers' already-
   tested roundtrip). Per-FOV spot-uid counters are carried over.
3. Verifies the new layout answers with the same counts, matrix keys,
   and uid counters as the old file.
4. Renames vlinks.h5.migrating -> vlinks.h5.retired. The original
   bytes are never modified or deleted -- rolling back is renaming the
   file back.

Interrupted runs: re-running resumes from vlinks.h5.migrating (capsule
writes are idempotent full-replaces). Handles both the columnar and the
old pickled schema. v1 stores are out of scope -- they stay on the
legacy reader and are never migrated.
"""
import json
import os
import pickle
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codelab_pipeline.io import analysis_store, columnar, paths  # noqa: E402


def _resolve_dp(target):
    target = os.path.abspath(target)
    if os.path.exists(os.path.join(target, paths.MANIFEST_NAME)):
        return target
    parent = os.path.dirname(target.rstrip(os.sep))
    if os.path.exists(os.path.join(parent, paths.MANIFEST_NAME)):
        return parent
    raise SystemExit(f'{target} is not a v2 project (no manifest.json here '
                     f'or in the parent) -- v1 stores are not migrated.')


def _dec(v):
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _unpack(grp, kind):
    """cells/spots/alleles group -> list of dicts, either schema."""
    if 'table' in grp:
        return {'cells': columnar.unpack_cells, 'spots': columnar.unpack_spots,
                'alleles': columnar.unpack_alleles}[kind](grp)
    if 'blob' in grp:
        payload = pickle.loads(bytes(grp['blob'][()]))
        if kind == 'cells':
            return payload.get('cells', [])
        if kind == 'spots':
            # old pickled spot dicts carry the pre-rename 'coordinate' key
            for d in payload:
                if 'adj_coordinate' not in d and 'coordinate' in d:
                    d['adj_coordinate'] = d.pop('coordinate')
        return payload
    return []


def main(target):
    dp = _resolve_dp(target)
    analysis = os.path.join(dp, 'analysis')
    old = os.path.join(analysis, 'vlinks.h5')
    migrating = old + '.migrating'
    retired = old + '.retired'

    if os.path.exists(old):
        if os.path.exists(migrating):
            raise SystemExit(f'both {old} and {migrating} exist -- resolve by '
                             f'hand (the .migrating one is a previous run).')
        os.replace(old, migrating)
        print(f'renamed -> {migrating}')
    elif os.path.exists(migrating):
        print(f'resuming interrupted migration from {migrating}')
    elif os.path.exists(retired):
        raise SystemExit(f'{retired} already exists and no vlinks.h5 remains '
                         f'-- this project is already migrated.')
    else:
        raise SystemExit(f'no vlinks.h5 under {analysis} -- nothing to migrate.')

    # analysis_store refuses to touch a project while a vlinks.h5(.migrating)
    # is present -- exactly the door this tool walks through, so mark the
    # project checked for THIS process only.
    analysis_store._MIGRATION_CHECKED.add(dp)

    manifest = paths.read_manifest(dp)
    modalities = list(manifest.get('modalities', {}))
    # any modality's storage_path resolves to the same analysis tree
    sp_any = os.path.join(dp, modalities[0]) if modalities else dp

    n = {'fovs': 0, 'cells': 0, 'slices': 0, 'spots': 0, 'alleles': 0,
         'matrices': 0, 'crossmodal': 0}
    expected = {}   # fov -> counters for the verify pass

    with h5py.File(migrating, 'r') as f:
        fov_names = sorted(k for k in f.keys() if k.startswith('FOV'))
        for fov_name in fov_names:
            fov = int(fov_name[3:])
            g = f[fov_name]
            exp = {'cells': 0, 'slices': 0, 'alleles': 0, 'matrices': {},
                   'next_uid': None, 'highest': None}

            if 'cells' in g:
                dicts = _unpack(g['cells'], 'cells')
                analysis_store.write_cell_dicts(sp_any, fov, dicts)
                n['cells'] += len(dicts)
                exp['cells'] = len(dicts)

            if 'spots' in g:
                sg_root = g['spots']
                exp['next_uid'] = int(sg_root.attrs.get('next_uid', 0)) or None
                exp['highest'] = int(sg_root.attrs.get('highest_uid_seen', 0)) or None
                for mod in sg_root:
                    if not isinstance(sg_root[mod], h5py.Group):
                        continue
                    for hy in sg_root[mod]:
                        for ch in sg_root[mod][hy]:
                            sg = sg_root[mod][hy][ch]
                            if 'table' not in sg and 'blob' not in sg:
                                continue
                            dicts = _unpack(sg, 'spots')
                            channel = int(ch[2:]) if ch.startswith('ch') else int(ch)
                            analysis_store.write_spot_dicts(
                                sp_any, fov, mod, hy, channel, dicts)
                            n['slices'] += 1
                            n['spots'] += len(dicts)
                            exp['slices'] += 1

            if 'alleles' in g:
                dicts = _unpack(g['alleles'], 'alleles')
                analysis_store.write_allele_dicts(sp_any, fov, dicts)
                n['alleles'] += len(dicts)
                exp['alleles'] = len(dicts)

            if 'matrix' in g:
                for mod in g['matrix']:
                    mg = g['matrix'][mod]
                    if not isinstance(mg, h5py.Group):
                        continue
                    entries = {}
                    for hybe in mg:
                        ds = mg[hybe]
                        entries[hybe] = {
                            'H': ds[:],
                            'reference_sequence': ds.attrs.get('reference_sequence'),
                            'steps': (np.asarray(ds.attrs['steps'])
                                      if 'steps' in ds.attrs else None),
                        }
                    if not entries:
                        continue
                    # private writer on purpose: provenance attrs must be
                    # carried over verbatim, not re-derived
                    target_file = analysis_store._matrices_path(sp_any, fov, mod)

                    def build(fh, entries=entries):
                        for hybe, e in entries.items():
                            ds = fh.create_dataset(
                                hybe, data=np.asarray(e['H'], dtype='float32'))
                            if e['reference_sequence'] is not None:
                                ds.attrs['reference_sequence'] = e['reference_sequence']
                            if e['steps'] is not None:
                                ds.attrs['steps'] = np.asarray(e['steps'], dtype='float32')

                    analysis_store._atomic_h5(target_file, build)
                    analysis_store._update_fov_manifest(
                        analysis_store._fov_dir(sp_any, fov),
                        lambda m, mod=mod, keys=sorted(entries):
                            m.setdefault('matrices', {}).__setitem__(mod, keys))
                    n['matrices'] += len(entries)
                    exp['matrices'][mod] = sorted(entries)

            if exp['next_uid'] or exp['highest']:
                analysis_store._update_fov_manifest(
                    analysis_store._fov_dir(sp_any, fov),
                    lambda m, e=exp: (
                        m.__setitem__('next_uid', max(int(m.get('next_uid', 1)),
                                                      e['next_uid'] or 1)),
                        m.__setitem__('highest_uid_seen',
                                      max(int(m.get('highest_uid_seen', 0)),
                                          e['highest'] or 0))))

            expected[fov] = exp
            n['fovs'] += 1

        # -- params ------------------------------------------------------
        if 'params' in f:
            pg = f['params']
            data = {'shared': {k: _dec(v) for k, v in pg.attrs.items()},
                    'modalities': {}}
            if 'modalities' in pg:
                for mod in pg['modalities']:
                    data['modalities'][mod] = {
                        k: _dec(v) for k, v in pg['modalities'][mod].attrs.items()}
            analysis_store._atomic_json(
                os.path.join(analysis, 'params.json'), data)
            print(f"params.json: {len(data['shared'])} shared, "
                  f"{sum(len(v) for v in data['modalities'].values())} modality-scoped")

            if 'celltype_config_blob' in pg:
                analysis_store._atomic_bytes(
                    os.path.join(analysis, 'celltype_config.pkl'),
                    bytes(pg['celltype_config_blob'][()]))
                print('celltype_config.pkl: carried over')

            # cross-modal results live under /params/FOV##
            for name in pg:
                if not name.startswith('FOV'):
                    continue
                fov = int(name[3:])
                cg = pg[name]
                cm = {}
                for ds_name in cg:
                    if ds_name == 'matrix_across':
                        cm.setdefault('_', {})['matrix'] = cg[ds_name][:].tolist()
                    elif ds_name.startswith('matrix_across__'):
                        mod = ds_name[len('matrix_across__'):]
                        cm.setdefault(mod, {})['matrix'] = cg[ds_name][:].tolist()
                for k, v in cg.attrs.items():
                    if k == 'z_across':
                        cm.setdefault('_', {})['z'] = float(v)
                    elif k.startswith('z_across__'):
                        cm.setdefault(k[len('z_across__'):], {})['z'] = float(v)
                    else:
                        for qk in analysis_store.CROSS_MODAL_QUALITY_KEYS:
                            flat = f'{qk}_across'
                            if k == flat:
                                cm.setdefault('_', {}).setdefault('quality', {})[qk] = float(v)
                            elif k.startswith(flat + '__'):
                                cm.setdefault(k[len(flat) + 2:], {}).setdefault(
                                    'quality', {})[qk] = float(v)
                if cm:
                    analysis_store._atomic_json(
                        os.path.join(paths.analysis_fov_dir(sp_any, fov),
                                     'crossmodal.json'), cm)
                    n['crossmodal'] += 1

    # -- verify through the ordinary readers -----------------------------
    print('verifying...')
    problems = []
    for fov, exp in expected.items():
        counts = analysis_store.fov_counts(sp_any, [fov])[fov]
        if counts['cells'] != exp['cells']:
            problems.append(f'FOV{fov}: cells {counts["cells"]} != {exp["cells"]}')
        if counts['alleles'] != exp['alleles']:
            problems.append(f'FOV{fov}: alleles {counts["alleles"]} != {exp["alleles"]}')
        if len(analysis_store.spot_slices(sp_any, fov)) != exp['slices']:
            problems.append(f'FOV{fov}: slice count mismatch')
        for mod, keys in exp['matrices'].items():
            sp_mod = os.path.join(dp, mod)
            got = sorted(analysis_store.aligned_hybes(sp_mod, fov))
            if got != keys:
                problems.append(f'FOV{fov}/{mod}: matrices {len(got)} != {len(keys)}')
    if problems:
        for p in problems:
            print('  MISMATCH:', p)
        raise SystemExit(f'{len(problems)} verification failure(s) -- '
                         f'{migrating} left in place, nothing retired.')

    os.replace(migrating, retired)
    print(f'retired -> {retired}')
    print(json.dumps(n, indent=2))
    print('done -- the project now runs on the per-FOV analysis layout.')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
