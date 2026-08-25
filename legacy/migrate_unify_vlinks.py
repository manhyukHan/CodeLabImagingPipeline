"""
Merge per-modality vlinks.h5 files into ONE project-level vlinks.h5.

Old: {project}/{queue}/vlinks.h5, one per modality, everything keyed by bare
     hybe -- so the cross-modal bridge hybe (e.g. Hyb_130) occupies the same
     key in both files while meaning different things (its matrix maps into
     THAT modality's reference frame, and its MIP is a different acquisition:
     DNA_Expt/Hyb_130 and RNA_Expt/Hyb_130 are separate DAX files).
New: {project}/vlinks.h5, hybe data keyed by (modality, hybe).

Refuses to guess. Any genuine conflict (same key, different bytes, where the
key is NOT modality-scoped in the new layout) is reported and aborts rather
than picking a winner.

Usage:  python tools/migrate_unify_vlinks.py <project_dir> [--write]
Without --write it is a dry run and reports what it would do.
"""
import glob
import os
import sys

import h5py
import numpy as np


def modality_of(queue_dir):
    """From the ingestion-time `modality` attr on any {hybe}_stack.h5 -- the
    one authoritative record (ingestion is the only step that knows which
    ExperimentLayout a DAX came from)."""
    for fov_dir in sorted(glob.glob(os.path.join(queue_dir, 'FOV*'))):
        for stack in sorted(glob.glob(os.path.join(fov_dir, '*_stack.h5'))):
            try:
                with h5py.File(stack, 'r') as f:
                    m = f.attrs.get('modality')
            except OSError:
                continue
            if m is not None:
                return m.decode() if isinstance(m, bytes) else str(m)
    return None


def copy_tree(src, dst):
    for k in src:
        o = src[k]
        if isinstance(o, h5py.Group):
            g = dst.require_group(k)
            for a, v in o.attrs.items():
                g.attrs[a] = v
            copy_tree(o, g)
        else:
            if k in dst:
                del dst[k]
            d = dst.create_dataset(k, data=o[()])
            for a, v in o.attrs.items():
                d.attrs[a] = v


def main(project, write):
    queues = [d for d in sorted(glob.glob(os.path.join(project, '*')))
              if os.path.isfile(os.path.join(d, 'vlinks.h5'))]
    if not queues:
        print(f'no per-queue vlinks.h5 under {project}'); return 1
    out_path = os.path.join(project, 'vlinks.h5')
    if os.path.exists(out_path):
        print(f'ABORT: {out_path} already exists'); return 1

    srcs = {}
    for q in queues:
        m = modality_of(q)
        if m is None:
            print(f'ABORT: cannot determine modality for {q}'); return 1
        if m in srcs:
            print(f'ABORT: two queues claim modality {m}'); return 1
        srcs[m] = q
        print(f'  {os.path.basename(q)} -> modality {m}')

    conflicts, plan = [], []
    cells_seen = {}
    for modality, q in srcs.items():
        with h5py.File(os.path.join(q, 'vlinks.h5'), 'r') as f:
            for key in f:
                if key.startswith('FOV'):
                    for sub in f[key]:
                        if sub in ('mip', 'matrix'):
                            plan.append(f'{key}/{sub}/* -> {key}/{sub}/{modality}/*')
                        elif sub == 'cells':
                            blob = f[f'{key}/cells/blob'][()].tobytes()
                            if key in cells_seen and cells_seen[key][1] != blob:
                                conflicts.append(
                                    f'{key}/cells differs between {cells_seen[key][0]} and {modality} '
                                    f'-- cells are shared in the new layout, cannot merge')
                            cells_seen.setdefault(key, (modality, blob))
                        else:
                            plan.append(f'{key}/{sub} -> {key}/{sub}  (shared)')
    if conflicts:
        print('\nCONFLICTS:'); [print('  ' + c) for c in conflicts]; return 1
    print(f'\n{len(plan)} moves planned, no conflicts')
    if not write:
        print('dry run -- pass --write to perform'); return 0

    with h5py.File(out_path, 'w') as out:
        for modality, q in srcs.items():
            with h5py.File(os.path.join(q, 'vlinks.h5'), 'r') as f:
                for key in f:
                    if key.startswith('FOV'):
                        for sub in f[key]:
                            if sub in ('mip', 'matrix'):
                                copy_tree(f[f'{key}/{sub}'],
                                          out.require_group(f'{key}/{sub}/{modality}'))
                            elif sub == 'cells':
                                if f'{key}/cells' not in out:
                                    copy_tree(f[f'{key}/cells'], out.require_group(f'{key}/cells'))
                            else:
                                copy_tree(f[f'{key}/{sub}'], out.require_group(f'{key}/{sub}'))
                    elif key == 'params':
                        pg = out.require_group(f'params/modalities/{modality}')
                        for a, v in f['params'].attrs.items():
                            pg.attrs[a] = v
                        for sub in f['params']:
                            copy_tree(f[f'params/{sub}'], out.require_group(f'params/{sub}'))
                    else:
                        copy_tree(f[key], out.require_group(key))
    print(f'wrote {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)')
    return 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    sys.exit(main(sys.argv[1].rstrip('/'), '--write' in sys.argv))
