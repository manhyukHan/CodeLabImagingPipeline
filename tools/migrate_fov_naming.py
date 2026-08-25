"""
One-shot rename of a v2 project's FOV directories to the unified fov###
naming (paths.fov_dir_name) -- stacks and mips for every modality in the
manifest. FOV is a 3-digit object everywhere now; the 2-digit FOV##
naming survives only in frozen v1 stores, which this tool refuses to
touch.

    python tools/migrate_fov_naming.py <project_root | storage_path>

Directory renames only -- no file is opened, copied, or rewritten, so
the operation is metadata-only and near-instant even on NAS. Idempotent:
already-renamed directories are skipped. A name collision (both FOV07
and fov007 present) aborts loudly with nothing half-done for that pair.
Old figures under the pre-modality figures/{category}/FOV##/ layout are
regenerable outputs and are left untouched, same policy as the figures
relayout itself.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codelab_pipeline.io import paths  # noqa: E402

_LEGACY = re.compile(r'^fov(\d+)$', re.IGNORECASE)


def _resolve_dp(target):
    target = os.path.abspath(target)
    if os.path.exists(os.path.join(target, paths.MANIFEST_NAME)):
        return target
    parent = os.path.dirname(target.rstrip(os.sep))
    if os.path.exists(os.path.join(parent, paths.MANIFEST_NAME)):
        return parent
    raise SystemExit(f'{target} is not a v2 project (no manifest.json here '
                     f'or in the parent) -- v1 stores keep their naming.')


def main(target):
    dp = _resolve_dp(target)
    modalities = list(paths.read_manifest(dp).get('modalities', {}))
    renamed = skipped = 0
    for modality in modalities:
        for sub in ('stacks', 'mips'):
            root = os.path.join(dp, modality, sub)
            try:
                entries = sorted(os.listdir(root))
            except OSError:
                continue
            for name in entries:
                m = _LEGACY.match(name)
                if not m or not os.path.isdir(os.path.join(root, name)):
                    continue
                new = paths.fov_dir_name(int(m.group(1)))
                if name == new:
                    skipped += 1
                    continue
                src, dst = os.path.join(root, name), os.path.join(root, new)
                if os.path.exists(dst) and name.lower() != new.lower():
                    raise SystemExit(f'both {src} and {dst} exist -- resolve '
                                     f'by hand before re-running.')
                if name.lower() == new.lower():
                    # case-only rename: two-step for filesystems that
                    # consider the names identical
                    tmp = os.path.join(root, new + '.renaming')
                    os.rename(src, tmp)
                    os.rename(tmp, dst)
                else:
                    os.rename(src, dst)
                renamed += 1
                print(f'  {modality}/{sub}/{name} -> {new}')
    print(f'done: {renamed} renamed, {skipped} already fov###.')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
