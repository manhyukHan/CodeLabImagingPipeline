"""
One-shot migration of a v1 project to the v2 storage layout
(codelab_pipeline/io/paths.py).

    python tools/migrate_store_v2.py <dp_out> <modality>=<v1_queue_dir> [...]

e.g.
    python tools/migrate_store_v2.py /data/proj_v2 RNA=/data/old/RNA_queue DNA=/data/old/DNA_queue

Builds, without touching the v1 source:
  <dp_out>/manifest.json
  <dp_out>/{modality}/stacks/FOV##/{hybe}.h5    stacks REWRITTEN chunked
                                               (32, 32, z-slab) + gzip-1,
                                               the layout the pipeline's
                                               small-XY/deep-Z crops need
  <dp_out>/{modality}/mips/FOV##/{hybe}.h5     extracted from the v1
                                               vlinks.h5 /mip groups
                                               (falling back to the stack
                                               file's own /mip), atomic
  <dp_out>/analysis/vlinks.h5                  the v1 vlinks.h5 minus its
                                               /FOV##/mip groups
  <dp_out>/figures/...                          PNGs found beside v1 data

The v1 store is read-only throughout. Layout files/dax dirs are read
from the v1 stack attrs where available; refresh them by Parse Layout in
the app afterwards.
"""
import os
import shutil
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codelab_pipeline.io import paths, preprocess  # noqa: E402


def rewrite_stack_chunked(src, dst):
    with h5py.File(src, 'r') as fi, h5py.File(dst + '.part', 'w') as fo:
        fo.attrs.update(dict(fi.attrs))
        for group in fi.keys():
            gi, go = fi[group], fo.create_group(group)
            go.attrs.update(dict(gi.attrs))
            for name in gi.keys():
                d = gi[name]
                if group == 'stack' and d.ndim == 3:
                    go.create_dataset(name, data=d[:], dtype=d.dtype,
                                      chunks=preprocess.stack_chunks(d.shape),
                                      compression='gzip', compression_opts=1, shuffle=True)
                else:
                    go.create_dataset(name, data=d[:], dtype=d.dtype)
    os.replace(dst + '.part', dst)


def main(dp_out, modality_dirs):
    os.makedirs(dp_out, exist_ok=True)
    paths.write_manifest(dp_out, sorted(modality_dirs))
    v1_vlinks = None

    for modality, src_root in modality_dirs.items():
        v1_vlinks = v1_vlinks or os.path.join(os.path.dirname(os.path.abspath(src_root).rstrip(os.sep)), 'vlinks.h5')
        dst_root = os.path.join(dp_out, modality)
        n_stacks = 0
        for fov_dir in sorted(d for d in os.listdir(src_root) if d.startswith('FOV')):
            src_fov = os.path.join(src_root, fov_dir)
            for f in sorted(os.listdir(src_fov)):
                if f.endswith('_stack.h5'):
                    hybe = f[:-len('_stack.h5')]
                    fov = int(fov_dir[3:])
                    dst = paths.stack_path(dst_root, fov, hybe)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    rewrite_stack_chunked(os.path.join(src_fov, f), dst)
                    n_stacks += 1
                elif f.endswith('.png'):
                    category = 'cells' if f.startswith('cell') else 'alignment'
                    dst = paths.figure_path(dst_root, category, int(fov_dir[3:]), f)
                    shutil.copy2(os.path.join(src_fov, f), dst)
        print(f'{modality}: {n_stacks} stacks rewritten chunked')

    # analysis vlinks: v1 copy minus /mip groups; MIPs become per-hybe files
    dst_vlinks = os.path.join(dp_out, 'analysis', 'vlinks.h5')
    os.makedirs(os.path.dirname(dst_vlinks), exist_ok=True)
    n_mips = 0
    with h5py.File(v1_vlinks, 'r') as fi, h5py.File(dst_vlinks, 'w') as fo:
        fo.attrs.update(dict(fi.attrs))

        def copy(name, obj):
            parts = name.split('/')
            if len(parts) >= 2 and parts[1] == 'mip':
                return                      # extracted below, not copied
            if isinstance(obj, h5py.Group):
                fo.require_group(name).attrs.update(dict(obj.attrs))
            else:
                fo.require_group(os.path.dirname(name) or '/')
                fi.copy(name, fo, name=name)
        fi.visititems(copy)

        for fov_name in [k for k in fi.keys() if k.startswith('FOV')]:
            if 'mip' not in fi[fov_name]:
                continue
            fov = int(fov_name[3:])
            for modality in fi[fov_name]['mip']:
                if modality not in modality_dirs:
                    continue
                for hybe in fi[fov_name]['mip'][modality]:
                    grp = fi[fov_name]['mip'][modality][hybe]
                    target = paths.mip_path(os.path.join(dp_out, modality), fov, hybe)
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with h5py.File(target + '.part', 'w') as fm:
                        fm.attrs['coordinate_order'] = 'yx'
                        if 'fiducial_channel' in grp.attrs:
                            fm.attrs['fiducial_channel'] = int(grp.attrs['fiducial_channel'])
                        for ch in grp:
                            fm.create_dataset(ch, data=grp[ch][:], chunks=True,
                                              compression='gzip', compression_opts=1)
                    os.replace(target + '.part', target)
                    n_mips += 1
    print(f'analysis/vlinks.h5 written (mips stripped); {n_mips} per-hybe MIP files extracted')
    print(f'v2 project ready at {dp_out} -- storage paths are ' +
          ', '.join(os.path.join(dp_out, m) for m in sorted(modality_dirs)))


if __name__ == '__main__':
    if len(sys.argv) < 3 or any('=' not in a for a in sys.argv[2:]):
        raise SystemExit(__doc__)
    main(sys.argv[1], dict(a.split('=', 1) for a in sys.argv[2:]))
