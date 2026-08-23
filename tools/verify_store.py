"""
Integrity checkup for an ingested store: does every stack/MIP file on disk
actually open and read end to end?

Written for the specific damage an interrupted OVERWRITE run leaves behind.
convert_dax_to_h5_worker in overwrite mode does os.remove(stack) and THEN
rebuilds it, and the rebuild is a plain h5py.File(..., 'w') -- not atomic.
Kill the run inside that window and the stack is either gone or truncated,
while its MIP (written atomically, .part + os.replace, and only AFTER the
stack) survives from the earlier run. So on a v2 store:

  MIP present, stack MISSING    -> overwrite deleted it, never rebuilt it
  MIP present, stack UNREADABLE -> killed mid-rebuild
  stack present, MIP missing    -> killed between the two writes

None of this is caught by the app's own checkup, which on a v2 store only
lists the MIP directory. Existence really is completeness for MIPs, since
they are written atomically -- but that says nothing about the stack file
sitting beside them, which is the one that actually holds the pixels.

Every check here is READ-ONLY.

Truncation is found by touching the FIRST and LAST element of each dataset:
HDF5 opens a truncated file quite happily and reports the declared shape,
and only fails once the missing bytes are actually read.

    python tools/verify_store.py <storage_path>
    python tools/verify_store.py <storage_path> --workers 8
    python tools/verify_store.py <storage_path> --list-bad
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py

from codelab_pipeline.io import paths


def _touch_ends(dset):
    """Read the first and last element -- the cheap truncation probe."""
    if dset.size == 0:
        return 'empty dataset'
    dset[tuple(0 for _ in dset.shape)]
    dset[tuple(n - 1 for n in dset.shape)]
    return None


def check_stack(path):
    """(path, kind, problem or None)."""
    try:
        if os.path.getsize(path) == 0:
            return (path, 'stack', 'zero bytes')
    except OSError as e:
        return (path, 'stack', 'stat failed: %s' % e)
    try:
        with h5py.File(path, 'r') as f:
            if '/stack' not in f:
                return (path, 'stack', 'no /stack group')
            raw = f.attrs.get('channel_list')
            if raw is None:
                channels = [n[2:] for n in f['/stack'] if n.startswith('ch')]
            else:
                channels = [c.decode() if isinstance(c, bytes) else str(c) for c in raw]
            if not channels:
                return (path, 'stack', 'no channels')
            depth = f.attrs.get('expected_depth')
            for ch in channels:
                name = 'ch%s' % ch
                if name not in f['/stack']:
                    return (path, 'stack', '/stack/%s missing' % name)
                d = f['/stack'][name]
                if d.ndim != 3:
                    return (path, 'stack', '/stack/%s ndim=%d, expected 3' % (name, d.ndim))
                if depth is not None and d.shape[-1] != int(depth):
                    return (path, 'stack', '/stack/%s depth %d != expected %d'
                            % (name, d.shape[-1], int(depth)))
                bad = _touch_ends(d)
                if bad:
                    return (path, 'stack', '/stack/%s: %s' % (name, bad))
                if '/mip' not in f or name not in f['/mip']:
                    return (path, 'stack', '/mip/%s missing (in-stack MIP)' % name)
                m = f['/mip'][name]
                if m.ndim != 2:
                    return (path, 'stack', '/mip/%s ndim=%d, expected 2' % (name, m.ndim))
                bad = _touch_ends(m)
                if bad:
                    return (path, 'stack', '/mip/%s: %s' % (name, bad))
    except Exception as e:
        return (path, 'stack', '%s: %s' % (type(e).__name__, e))
    return (path, 'stack', None)


def check_mip(path):
    try:
        if os.path.getsize(path) == 0:
            return (path, 'mip', 'zero bytes')
    except OSError as e:
        return (path, 'mip', 'stat failed: %s' % e)
    try:
        with h5py.File(path, 'r') as f:
            names = [n for n in f if n.startswith('ch')]
            if not names:
                return (path, 'mip', 'no ch* datasets')
            for n in names:
                d = f[n]
                if d.ndim != 2:
                    return (path, 'mip', '%s ndim=%d, expected 2' % (n, d.ndim))
                bad = _touch_ends(d)
                if bad:
                    return (path, 'mip', '%s: %s' % (n, bad))
    except Exception as e:
        return (path, 'mip', '%s: %s' % (type(e).__name__, e))
    return (path, 'mip', None)


def check_one(job):
    kind, path = job
    return check_stack(path) if kind == 'stack' else check_mip(path)


def enumerate_files(storage_path):
    """{(fov, hybe): {'stack': path or None, 'mip': path or None}}"""
    v2 = paths.is_v2(storage_path)
    out = {}
    stacks_root = os.path.join(storage_path, 'stacks') if v2 else storage_path
    if os.path.isdir(stacks_root):
        for fov_dir in sorted(d for d in os.listdir(stacks_root) if d.startswith('FOV')):
            full = os.path.join(stacks_root, fov_dir)
            if not os.path.isdir(full):
                continue
            fov = int(fov_dir[3:])
            for name in sorted(os.listdir(full)):
                if v2 and name.endswith('.h5'):
                    hybe = name[:-3]
                elif not v2 and name.endswith('_stack.h5'):
                    hybe = name[:-len('_stack.h5')]
                else:
                    continue
                entry = out.setdefault((fov, hybe), {'stack': None, 'mip': None})
                entry['stack'] = os.path.join(full, name)
    if v2:
        mips_root = os.path.join(storage_path, 'mips')
        if os.path.isdir(mips_root):
            for fov_dir in sorted(d for d in os.listdir(mips_root) if d.startswith('FOV')):
                full = os.path.join(mips_root, fov_dir)
                if not os.path.isdir(full):
                    continue
                fov = int(fov_dir[3:])
                for name in sorted(os.listdir(full)):
                    if not name.endswith('.h5'):
                        continue
                    entry = out.setdefault((fov, name[:-3]), {'stack': None, 'mip': None})
                    entry['mip'] = os.path.join(full, name)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('storage_path')
    ap.add_argument('--workers', type=int,
                    default=max(1, min(8, (os.cpu_count() or 4) // 2)))
    ap.add_argument('--list-bad', action='store_true',
                    help='print every affected file, not just the first 40 of each kind')
    args = ap.parse_args()

    sp = os.path.abspath(args.storage_path)
    if not os.path.isdir(sp):
        ap.error('not a directory: %s' % sp)

    v2 = paths.is_v2(sp)
    pairs = enumerate_files(sp)
    strays = [os.path.join(dp, f)
              for dp, _, fs in os.walk(sp) for f in fs if f.endswith('.part')]

    print('store   : %s' % sp)
    print('layout  : %s' % ('v2' if v2 else 'v1'))
    print('pairs   : %d (fov, hybe) entries' % len(pairs))
    print('.part   : %d interrupted atomic write(s)' % len(strays))

    missing_stack = sorted(k for k, v in pairs.items() if v['stack'] is None)
    missing_mip = sorted(k for k, v in pairs.items() if v2 and v['mip'] is None)

    jobs = []
    for entry in pairs.values():
        if entry['stack']:
            jobs.append(('stack', entry['stack']))
        if entry['mip']:
            jobs.append(('mip', entry['mip']))
    print('files   : %d to open and read-probe, %d worker(s)' % (len(jobs), args.workers))
    print()

    bad, done = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(check_one, job) for job in jobs]
        for fut in as_completed(futures):
            path, kind, problem = fut.result()
            done += 1
            if problem:
                bad.append((kind, path, problem))
            if done % 500 == 0:
                print('  ... %d/%d checked, %d bad so far' % (done, len(jobs), len(bad)))

    print()
    print('=' * 72)
    print('checked            : %d' % len(jobs))
    print('UNREADABLE/BROKEN  : %d' % len(bad))
    print('MIP but NO STACK   : %d   <- interrupted overwrite' % len(missing_stack))
    print('STACK but no MIP   : %d' % len(missing_mip))
    print('.part leftovers    : %d' % len(strays))
    print('=' * 72)

    def show(title, items, fmt):
        if not items:
            return
        print()
        print(title)
        limit = len(items) if args.list_bad else 40
        for item in items[:limit]:
            print('    %s' % fmt(item))
        if len(items) > limit:
            print('    ... and %d more (--list-bad for all)' % (len(items) - limit))

    show('BROKEN FILES:', bad, lambda b: '%-5s %s -- %s' % (b[0], b[1], b[2]))
    show('MIP present but STACK missing:', missing_stack,
         lambda k: 'FOV%02d %s' % (k[0], k[1]))
    show('STACK present but MIP missing:', missing_mip,
         lambda k: 'FOV%02d %s' % (k[0], k[1]))
    show('.part leftovers:', strays, lambda p: p)

    return 1 if (bad or missing_stack or missing_mip or strays) else 0


if __name__ == '__main__':
    sys.exit(main())
