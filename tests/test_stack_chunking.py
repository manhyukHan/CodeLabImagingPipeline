"""
Stack z-slabs are sized to DIVIDE the depth, not fixed at 64.

WHY THIS EXISTS. Storage v2 (65a562f) chose chunks=(32, 32, z-slab) so a
small-XY/deep-Z crop costs one network read per z-chunk -- "a 17x17xZ crop
drops from 289 scattered network reads to 3" -- and slabbed rather than
whole-depth chunking so the plane viewer keeps affordable partial-Z access.
Both are right. What was wrong is that the slab was the CONSTANT 64,
clamped to the depth, so whenever depth was not a multiple of 64 the last
chunk covered planes that do not exist and every full-depth read inflated
them anyway.

The waste is worst just past a boundary, which is exactly where the real
data sits. Measured on the real MAZ store (1024x1024x129):

    3 chunks of 64  ->  cover 192 planes to hold 129   (1.49x inflated)
    3 chunks of 43  ->  cover 129 planes exactly       (1.00x)

and on one real 129-plane stack, same file size (138 vs 139 MB):

    58 real cell windows, full depth    0.841 s -> 0.670 s   (1.26x)
    one full-frame z-plane            553.7 ms -> 399.7 ms   (1.39x)

BOTH directions improve, and the reason is arithmetic rather than luck:
with n = ceil(d/64) slabs, ceil(d/n) is never larger than 64. That is the
property this test pins -- the change can only shrink a chunk, never grow
one, so it cannot be worse than the fixed 64 it replaces on ANY access
pattern. The chunk COUNT is unchanged, so v2's own "3 network reads"
design survives untouched.

The risk in getting this wrong is not a crash. A chunk larger than the
dataset makes h5py raise (loudly, fine), but a chunk that merely covers
too much just quietly costs time forever. So the properties are asserted
across a sweep of depths, not on one example, and the pixels are compared
byte-for-byte after a real round trip.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_stack_chunking.py
"""
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                       # noqa: E402
import h5py                                              # noqa: E402

from codelab_pipeline.io import preprocess, stack_cache   # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}' + (f'   [{detail}]' if detail else ''))


# Real depths this pipeline has seen, plus the boundaries that break naive
# arithmetic. 129 is the real MAZ store; 120 the real E: store; 177 the
# real cross-modal store stack_cache.py was measured on.
DEPTHS = [1, 2, 7, 8, 31, 32, 63, 64, 65, 100, 110, 118, 120, 125, 127, 128,
          129, 130, 135, 150, 177, 191, 192, 193, 200, 256, 300]


def main():
    print('\n-- the slab covers the depth, and wastes nothing it need not --')
    # Equal slabs cannot always land exactly on the depth: 130 planes are
    # forced into 3 slabs (the count is ceil(d/64)) and 43x3 = 129 is one
    # short, so 44 is optimal and covers 132. What IS guaranteed is that
    # the total slack stays under one plane per slab -- n*ceil(d/n) - d <=
    # n-1 -- which, since n <= d/64 + 1, bounds the waste at about 1/64
    # (~1.6%) for every depth. That is the ceiling the old fixed 64 blew
    # through at 49%.
    worst = (0, None)
    bad = []
    for d in DEPTHS:
        _, _, z = preprocess.stack_chunks((1024, 1024, d))
        n = math.ceil(d / z)
        waste = n * z / d
        if waste > worst[0]:
            worst = (waste, d)
        if not (n * z >= d and (n - 1) * z < d and n * z - d <= n - 1):
            bad.append((d, z, n, n * z))
    check('every depth is covered, with no empty slab and under one plane '
          'of slack per slab', not bad, str(bad[:3]))
    check('so the waste is bounded near 1/64 at every depth, not 1.49x',
          worst[0] <= 1.016, f'worst {worst[0]:.4f}x at depth {worst[1]}')

    print('\n-- it can never be worse than the fixed 64 it replaces --')
    # The whole safety argument. A bigger chunk would inflate more on a
    # single-plane read; a different count would change the network-read
    # profile v2 was designed around.
    grew, count_changed = [], []
    for d in DEPTHS:
        _, _, z = preprocess.stack_chunks((1024, 1024, d))
        old = min(d, 64)
        if z > old:
            grew.append((d, z, old))
        if math.ceil(d / z) != math.ceil(d / old):
            count_changed.append((d, math.ceil(d / z), math.ceil(d / old)))
    check('no depth gets a LARGER slab than the old fixed 64',
          not grew, str(grew[:3]))
    check('the slab COUNT is identical to the old scheme, so v2\'s '
          'reads-per-crop is unchanged', not count_changed, str(count_changed[:3]))

    print('\n-- the real stores get the shapes that were measured --')
    check('129 planes (real MAZ) -> 43, not 64',
          preprocess.stack_chunks((1024, 1024, 129))[2] == 43,
          str(preprocess.stack_chunks((1024, 1024, 129))))
    check('120 planes (real E:) -> 60', preprocess.stack_chunks((1024, 1024, 120))[2] == 60)
    check('177 planes (real cross-modal) -> 59',
          preprocess.stack_chunks((1024, 1024, 177))[2] == 59)
    check('a depth that already divides evenly is left alone',
          preprocess.stack_chunks((1024, 1024, 128))[2] == 64)

    print('\n-- never larger than the data, on any axis --')
    # h5py RAISES on a chunk bigger than the dataset, so this is the
    # difference between ingesting a small test stack and crashing on it.
    for shape in [(1024, 1024, 129), (32, 32, 8), (17, 9, 3), (1, 1, 1),
                  (8, 1024, 200), (2000, 3, 64)]:
        c = preprocess.stack_chunks(shape)
        ok = all(1 <= c[i] <= shape[i if i < 2 else -1] for i in range(3))
        check(f'chunks{c} fit shape{shape}', ok)

    print('\n-- h5py accepts them, and the pixels survive --')
    tmp = tempfile.mkdtemp()
    for shape in [(64, 48, 129), (32, 32, 8), (40, 40, 65)]:
        rng = np.random.default_rng(shape[2])
        data = rng.integers(0, 65535, size=shape, dtype=np.uint16)
        p = os.path.join(tmp, f'stack_{shape[2]}.h5')
        chunks = preprocess.stack_chunks(shape)
        try:
            with h5py.File(p, 'w') as f:
                f.create_dataset('/stack/ch555', data=data, chunks=chunks,
                                 compression='gzip', compression_opts=1, shuffle=True)
            wrote, err = True, ''
        except Exception as e:                            # noqa: BLE001
            wrote, err = False, f'{type(e).__name__}: {e}'
        check(f'h5py writes shape{shape} with chunks{chunks}', wrote, err)
        if not wrote:
            continue
        with h5py.File(p, 'r') as f:
            ds = f['/stack/ch555']
            check(f'  depth {shape[2]}: the stored chunking is what we asked for',
                  tuple(ds.chunks) == chunks, f'{ds.chunks} vs {chunks}')
            check(f'  depth {shape[2]}: a full-depth cell window round-trips exactly',
                  np.array_equal(ds[3:20, 4:22, :], data[3:20, 4:22, :]))
            check(f'  depth {shape[2]}: a single z-plane round-trips exactly',
                  np.array_equal(ds[:, :, shape[2] // 2], data[:, :, shape[2] // 2]))

    print('\n-- the plane cache follows the file, it does not assume 64 --')
    # stack_cache slabs on ds.chunks[2]; if it ever hardcoded 64 again it
    # would cache a span HDF5 does not inflate as a unit, and quietly
    # re-inflate the same chunks it just paid for.
    p = os.path.join(tmp, 'stack_129.h5')
    rng = np.random.default_rng(7)
    data = np.ascontiguousarray(rng.integers(0, 65535, size=(64, 48, 129), dtype=np.uint16))
    with h5py.File(p, 'w') as f:
        f.create_dataset('/stack/ch555', data=data,
                         chunks=preprocess.stack_chunks(data.shape),
                         compression='gzip', compression_opts=1, shuffle=True)
    stack_cache.clear()
    shape = stack_cache.stack_shape(p, 555)
    check('stack_shape reports the file\'s OWN slab',
          shape is not None and shape[3] == 43, str(shape))
    check('and the depth it reports is the real one',
          shape is not None and shape[2] == 129, str(shape))
    got = stack_cache.plane(p, 555, 100, dtype=np.uint16)
    check('a plane served through the cache equals the raw pixels',
          got is not None and np.array_equal(got, data[:, :, 100]))
    got2 = stack_cache.plane(p, 555, 128, dtype=np.uint16)
    check('and so does the LAST plane, the one a 64-slab would over-cover',
          got2 is not None and np.array_equal(got2, data[:, :, 128]))

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        for f in FAIL:
            print('  FAILED:', f)
        return 1
    print('ALL GOOD')
    return 0


if __name__ == '__main__':
    sys.exit(main())
