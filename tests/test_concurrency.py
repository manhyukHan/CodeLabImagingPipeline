"""
Concurrency stress for the per-FOV analysis store -- the "possible
concurrent writing/reading trials in theory" cases, exercised for real
with threads on a temp project:

1. two threads merge-writing DISJOINT hybe sets into the SAME
   (FOV, modality) matrices file -- no lost update (the RMW lock);
2. four threads allocating spot uids on the same FOV -- all unique;
3. a writer full-replacing a spot slice while a reader hammers
   whole-FOV reads -- reads never error and never see a torn file;
4. mixed readers (cells/counts/matrices) + writers across kinds at
   once -- cache and manifest survive.
Run: python tests/test_concurrency.py
"""
import os
import shutil
import sys
import tempfile
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline.io import analysis_store as A   # noqa: E402
from codelab_pipeline.io import paths                 # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(('  ok ' if cond else 'FAIL ') + name + (f' -- {detail}' if detail and not cond else ''))


def spot_dict(uid, fov):
    return {'uid': uid, 'fov': fov, 'modality': 'DNA', 'hybe': 'H1', 'channel': 1,
            'cell': -1, 'celltype': '', 'adj_coordinate': (1., 2., 3.),
            'raw_coordinate': (1., 2., 3.), 'size': 1.0, 'brightness': 1.0,
            'linked': False, 'linked_at': None, 'mixture_centroids': ()}


def run_threads(fns):
    errors = []

    def wrap(fn):
        try:
            fn()
        except Exception as e:
            errors.append(f'{type(e).__name__}: {e}')
    ts = [threading.Thread(target=wrap, args=(fn,)) for fn in fns]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return errors


def main():
    root = tempfile.mkdtemp(prefix='astore_conc_')
    try:
        dp = os.path.join(root, 'proj')
        paths.write_manifest(dp, ['DNA', 'RNA'])
        sp = os.path.join(dp, 'DNA')
        fov = 1

        # 1. concurrent matrices merges, disjoint hybe sets ---------------
        def merge(prefix):
            def go():
                for i in range(30):
                    A.write_same_modality_matrices(
                        sp, fov, {f'{prefix}{i:03d}': np.eye(3) * (i + 1)}, 'REF')
            return go
        errs = run_threads([merge('A'), merge('B')])
        ah = A.aligned_hybes(sp, fov)
        check('matrices: no exceptions', not errs, '; '.join(errs[:3]))
        check('matrices: no lost update under concurrent merge',
              len(ah) == 60, f'{len(ah)}/60 survived')

        # 2. concurrent uid allocation ------------------------------------
        got = []
        lock = threading.Lock()

        def alloc():
            for _ in range(25):
                uids = A.allocate_spot_uids(sp, fov, 4)
                with lock:
                    got.extend(uids)
        errs = run_threads([alloc] * 4)
        check('uids: no exceptions', not errs, '; '.join(errs[:3]))
        check('uids: all unique under 4-thread allocation',
              len(got) == 400 and len(set(got)) == 400,
              f'{len(got)} issued, {len(set(got))} unique')

        # 3. writer replaces a slice while a reader hammers the FOV -------
        stop = threading.Event()
        bad_reads = []

        def writer():
            try:
                for n in range(50):
                    A.write_spot_dicts(sp, fov, 'DNA', 'H1', 1,
                                       [spot_dict(1000 + i, fov) for i in range(n % 7)])
            finally:
                stop.set()   # a dead writer must not leave readers spinning

        def reader():
            while not stop.is_set():
                spots = A.read_spots(sp, fov)
                if not isinstance(spots, list) or any('uid' not in s for s in spots):
                    bad_reads.append('malformed read')
                    return
        errs = run_threads([writer, reader, reader])
        check('slice write vs whole-FOV reads: no exceptions', not errs, '; '.join(errs[:3]))
        check('reads never torn', not bad_reads)

        # 4. mixed kinds all at once --------------------------------------
        def w_cells():
            for i in range(20):
                A.write_cell_dicts(sp, fov, [])
        def w_alleles():
            for i in range(20):
                A.write_allele_dicts(sp, fov, [])
        def r_mixed():
            for i in range(60):
                A.read_cells(sp, fov)
                A.fov_counts(sp, [fov])
                A.read_same_modality_matrices(sp, fov, ['A000', 'B000'])
                A.spot_slices(sp, fov)
        errs = run_threads([w_cells, w_alleles, r_mixed, r_mixed, r_mixed])
        check('mixed writers+readers: no exceptions', not errs, '; '.join(errs[:3]))
        counts = A.fov_counts(sp, [fov])[fov]
        check('final state consistent', counts['cells'] == 0 and counts['alleles'] == 0)
        strays = [f for _, _, fs in os.walk(dp) for f in fs if f.endswith('.part')]
        check('no leftover .part files', not strays, str(strays[:3]))

        # 5. stack slab cache: FIFO capacity law -------------------------
        from codelab_pipeline.io import stack_cache as SC
        import h5py
        SLAB = 4
        stacks = []
        for i in range(5):
            p = os.path.join(root, f'stk{i}.h5')
            with h5py.File(p, 'w') as f:
                f.create_dataset('/stack/ch1', data=np.full((32, 32, 16), i, dtype='uint16'),
                                 chunks=(16, 16, SLAB))
            stacks.append(p)
        one = 32 * 32 * SLAB * 2
        os.environ['CODELAB_STACK_CACHE_GB'] = str(one * 3 / 2 ** 30)   # room for 3
        SC.clear()
        for p in stacks:
            SC.plane(p, 1, 0)
        held = [k[0] for k in SC._CACHE]
        check('slab cache holds exactly the budgeted count', len(held) == 3, str(len(held)))
        check('eviction is FIFO (oldest dropped, newest kept)',
              held == [os.path.abspath(p) for p in stacks[2:]], str([os.path.basename(h) for h in held]))

        SC.clear()
        SC.STATS['hits'] = SC.STATS['misses'] = 0
        for z in range(SLAB):
            SC.plane(stacks[0], 1, z)
        check('planes within one slab cost ONE read',
              (SC.STATS['misses'], SC.STATS['hits']) == (1, SLAB - 1),
              f"misses={SC.STATS['misses']} hits={SC.STATS['hits']}")
        SC.plane(stacks[0], 1, SLAB)
        check('crossing into the next slab adds an entry', len(SC._CACHE) == 2, str(len(SC._CACHE)))

        os.environ['CODELAB_STACK_CACHE_GB'] = '0'
        SC.clear()
        got = SC.plane(stacks[0], 1, 0)
        check('budget 0 disables retention but still returns data',
              got is not None and len(SC._CACHE) == 0, str(len(SC._CACHE)))
        os.environ.pop('CODELAB_STACK_CACHE_GB', None)
        SC.clear()

        print()
        print(f'{len(PASS)} passed, {len(FAIL)} failed')
        if FAIL:
            raise SystemExit('FAILURES: ' + ', '.join(FAIL))
        print('ALL GOOD')
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == '__main__':
    main()
