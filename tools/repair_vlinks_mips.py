"""
Re-write vlinks.h5 MIPs for hybes whose stack file exists but whose MIP
copy is missing -- the exact damage left by

    FOV06 Hyb_073: ERROR: ingested but failed to write vlinks.h5 MIP:
    Unable to open file (file is already open for read-only)

which happened whenever the Cell/Spot status viewer's reads overlapped
IngestionWorker's per-task MIP write. vlinks_store._open_vlinks stops that
class of failure happening again; this repairs the runs that already hit it.

Why a tool and not just a re-ingest: the conversion itself SUCCEEDED in
these cases. {hybe}_stack.h5 is complete and already holds the MIP under
its own /mip/ch{c} (written in the same pass), so the repair is a local
file copy taking a second or so per hybe. Re-ingesting would drag the whole
DAX back over the network for nothing -- and Append mode would not even do
that, because convert_dax_to_h5_worker returns early on the stack file
merely EXISTING, so it skips these hybes and never notices the missing MIP.
That is precisely why this damage is easy to carry forward unnoticed.

Run with the app CLOSED. The lock inside vlinks_store is per-process, so it
cannot protect this tool from a running GUI that has vlinks.h5 open; HDF5
would refuse the write the same way it refused it during ingestion.

    python tools/repair_vlinks_mips.py <storage_path>            # report
    python tools/repair_vlinks_mips.py <storage_path> --apply    # fix
    python tools/repair_vlinks_mips.py <storage_path> --apply --fov 6 --hybe Hyb_073
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py

from codelab_pipeline.io import vlinks_store


def stack_files(storage_path):
    """[(fov, hybe, path), ...] for every {hybe}_stack.h5 under storage_path."""
    out = []
    pattern = os.path.join(storage_path, 'FOV*', '*_stack.h5')
    for path in sorted(glob.glob(pattern)):
        fov_dir = os.path.basename(os.path.dirname(path))
        m = re.fullmatch(r'FOV(\d+)', fov_dir)
        if not m:
            continue
        hybe = os.path.basename(path)[:-len('_stack.h5')]
        out.append((int(m.group(1)), hybe, path))
    return out


def stack_mips(path):
    """
    (channel_mips, fiducial_channel) read from a stack file's own /mip, or
    (None, None) if the file is unreadable or has no MIP group -- an
    INCOMPLETE/UNREADABLE stack is a different problem (re-ingest it) and
    must not be silently half-repaired here.
    """
    try:
        with h5py.File(path, 'r') as f:
            if '/mip' not in f:
                return None, None
            mips = {}
            for name in f['/mip']:
                if name.startswith('ch'):
                    mips[name[2:]] = f['/mip'][name][:]
            if not mips:
                return None, None
            fid = f.attrs.get('fiducial_channel')
            return mips, (int(fid) if fid is not None else None)
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('storage_path', help="a modality's stack directory (the one holding FOV##/)")
    ap.add_argument('--apply', action='store_true', help='write the repairs (default: report only)')
    ap.add_argument('--fov', type=int, action='append', help='limit to this FOV (repeatable)')
    ap.add_argument('--hybe', action='append', help='limit to this hybe folder (repeatable)')
    args = ap.parse_args()

    storage_path = os.path.abspath(args.storage_path)
    if not os.path.isdir(storage_path):
        ap.error(f'not a directory: {storage_path}')
    vlinks_path = vlinks_store._vlinks_path(storage_path)
    if not os.path.exists(vlinks_path):
        ap.error(f'no vlinks.h5 beside {storage_path} (looked at {vlinks_path})')

    stacks = stack_files(storage_path)
    if args.fov:
        stacks = [s for s in stacks if s[0] in set(args.fov)]
    if args.hybe:
        stacks = [s for s in stacks if s[1] in set(args.hybe)]
    if not stacks:
        print('no stack files matched.')
        return 0

    modality = vlinks_store.modality_of(storage_path)
    print(f'store    : {vlinks_path}')
    print(f'modality : {modality}')
    print(f'checking : {len(stacks)} stack file(s) under {storage_path}')
    print()

    # Scan under ONE open handle -- this is hundreds of reads otherwise.
    todo, unreadable = [], []
    with vlinks_store.vlinks_session(storage_path):
        for fov, hybe, path in stacks:
            mips, _ = stack_mips(path)
            if mips is None:
                unreadable.append((fov, hybe, path))
                continue
            missing = [ch for ch in mips
                       if vlinks_store.read_hybe_mip(storage_path, fov, hybe, ch) is None]
            if missing:
                todo.append((fov, hybe, path, sorted(missing)))

    for fov, hybe, _path, missing in todo:
        print(f'  MISSING MIP  FOV{fov:02d} {hybe}  ch{", ch".join(missing)}')
    for fov, hybe, path in unreadable:
        print(f'  UNREADABLE   FOV{fov:02d} {hybe}  -- re-ingest this one ({path})')

    if not todo:
        print('  every stack file already has its vlinks.h5 MIP.')
        return 0 if not unreadable else 1

    print()
    if not args.apply:
        print(f'{len(todo)} hybe(s) would be repaired. Re-run with --apply to write them.')
        return 0

    # Writes happen OUTSIDE the read session above: a write nested inside one
    # is refused by design (see vlinks_store._open_vlinks).
    repaired, failed = 0, []
    for fov, hybe, path, _missing in todo:
        mips, fiducial = stack_mips(path)
        if mips is None:
            failed.append((fov, hybe, 'stack became unreadable'))
            continue
        try:
            vlinks_store.write_hybe_mip(storage_path, fov, hybe, mips,
                                        fiducial_channel=fiducial)
            repaired += 1
            print(f'  repaired  FOV{fov:02d} {hybe}')
        except Exception as e:
            failed.append((fov, hybe, str(e)))
            print(f'  FAILED    FOV{fov:02d} {hybe}: {e}')

    print()
    print(f'repaired {repaired}/{len(todo)}')
    if failed:
        print(f'{len(failed)} failed -- if these say "already open", close the app and re-run.')
    return 1 if (failed or unreadable) else 0


if __name__ == '__main__':
    sys.exit(main())
