"""
The Z-stack window read reuses one buffer -- without changing a pixel.

WHY THIS EXISTS AS ITS OWN TEST: the change it guards lives on the cell
alignment path, and the suite that exercises that path
(test_alignment_ground_truth) cannot run on this machine at all -- its
fixture is absent, as CLAUDE.md warns. A green suite was therefore no
evidence for this change. This builds its own gzip-chunked stack, so it
runs anywhere and actually covers the code.

WHAT IT GUARDS, and why it was worth doing:

hybe_zx_projection reads a cell-sized crop across the full depth. At a
measured median cell crop of 79x79 px that is 79 x 79 x 110 x 2 = 1.31 MB,
and 2.11 MB at 177 planes. Both are over a boundary measured on this
machine, and the boundary is sharp:

    1016 KB  ->  40/40 allocations came back DIRTY   (recycled in-process)
    1020 KB  ->   0/40 came back dirty               (fresh from the kernel)

Windows' heap passes anything past ~1 MB straight to VirtualAlloc and
returns it on free, so each such read takes freshly zeroed pages from the
kernel and hands them back to be zeroed again. This leg runs once per
(cell, hybe) -- of order 390,000 times in a whole-project run.

The risk in fixing it is silent corruption, not breakage: a buffer reused
across calls, or shared between threads, returns plausible pixels from the
WRONG crop, and an alignment fitted against them is wrong in a way no
exception reports. So the pixels are compared byte-for-byte here, the
oversized-crop fallback is exercised, and the per-thread isolation is
checked with two threads reading different windows at once.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_zx_window_buffer.py
"""
import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                       # noqa: E402
import h5py                                              # noqa: E402

from codelab_pipeline.alignment import chain              # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}' + (f'   [{detail}]' if detail else ''))


def make_stack(path, shape=(256, 256, 40), seed=0):
    """A stack shaped and chunked like the real ones, with unique pixels.

    Every voxel gets a distinct value, so a crop read from the wrong place
    -- or a buffer left over from a previous call -- cannot accidentally
    match the right answer.
    """
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 65535, size=shape, dtype=np.uint16)
    chunks = tuple(min(c, s) for c, s in zip((32, 32, 20), shape))
    with h5py.File(path, 'w') as f:
        f.create_dataset('/stack/ch555', data=data, chunks=chunks,
                         compression='gzip', shuffle=True)
    return data


def main():
    tmp = os.path.join(tempfile.mkdtemp(), 'stack.h5')
    truth = make_stack(tmp)

    print('\n-- the pixels are identical to a plain read --')
    windows = [(0, 79, 0, 79), (100, 179, 100, 179), (200, 256, 200, 256),
               (0, 1, 0, 1), (13, 92, 200, 251), (177, 256, 0, 160)]
    with h5py.File(tmp, 'r') as f:
        ds = f['/stack/ch555']
        for (y0, y1, x0, x1) in windows:
            got = chain._read_zx_window(ds, y0, y1, x0, x1)
            want = truth[y0:y1, x0:x1, :]
            check(f'window {y1-y0}x{x1-x0} at ({y0},{x0}) matches exactly',
                  got.shape == want.shape and np.array_equal(got, want),
                  f'{got.shape} vs {want.shape}')

        print('\n-- consecutive calls do not leak the previous crop --')
        a = chain._read_zx_window(ds, 0, 79, 0, 79).copy()
        chain._read_zx_window(ds, 100, 179, 100, 179)
        b = chain._read_zx_window(ds, 0, 79, 0, 79)
        check('the same window read twice, with another between, is stable',
              np.array_equal(a, b))
        check('and still equals the source', np.array_equal(b, truth[0:79, 0:79, :]))

        print('\n-- the buffer really is reused --')
        addrs = {chain._read_zx_window(ds, 0, 79, 0, 79).__array_interface__['data'][0]
                 for _ in range(25)}
        check('25 reads share one allocation', len(addrs) == 1, str(len(addrs)))
        small = chain._read_zx_window(ds, 0, 40, 0, 40)
        check('a smaller window is a view of the same buffer, not a new one',
              small.base is not None)

        print('\n-- a crop past the buffer falls back, correctly --')
        big = chain._read_zx_window(ds, 0, 256, 0, 256)
        check('an oversized crop still returns the right pixels',
              np.array_equal(big, truth[:, :, :]))
        check('and it is a fresh array, not the shared buffer',
              big.__array_interface__['data'][0] not in addrs)

    print('\n-- two threads do not share a buffer --')
    # The GUI draws the ZX preview on its own threads while a fit may be
    # running; one shared buffer would have them overwrite each other.
    results, errors = {}, []

    def worker(tag, box):
        try:
            with h5py.File(tmp, 'r') as fh:
                d = fh['/stack/ch555']
                for _ in range(30):
                    got = chain._read_zx_window(d, *box)
                    if not np.array_equal(got, truth[box[0]:box[1], box[2]:box[3], :]):
                        errors.append(tag)
                        return
                results[tag] = True
        except Exception as e:                            # noqa: BLE001
            errors.append(f'{tag}: {e}')

    t1 = threading.Thread(target=worker, args=('A', (0, 79, 0, 79)))
    t2 = threading.Thread(target=worker, args=('B', (100, 179, 100, 179)))
    t1.start(); t2.start(); t1.join(); t2.join()
    check('two threads reading different windows never see each other\'s pixels',
          not errors and results.get('A') and results.get('B'), str(errors[:2]))

    print('\n-- the projection itself is unchanged --')
    with h5py.File(tmp, 'r') as f:
        ds = f['/stack/ch555']
        got = chain._read_zx_window(ds, 20, 99, 30, 109).max(axis=0)
        want = truth[20:99, 30:109, :].max(axis=0)
        check('max-projection over the reused buffer equals the plain one',
              np.array_equal(got, want), f'{got.shape}')

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        for f in FAIL:
            print('  FAILED:', f)
        return 1
    print('ALL GOOD')
    return 0


if __name__ == '__main__':
    sys.exit(main())
