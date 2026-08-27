"""
Freeze real crops for any registered experiment, in parallel.

WHY THIS EXISTS SEPARATELY FROM fit_testbox.harvest
---------------------------------------------------
That one is serial and pinned to MP58 by module constants. Four
experiments at MP58's measured 65 s/allele is about four and a half hours
of wall clock, nearly all of it a single process waiting on NAS reads
while 31 cores do nothing.

The bench FORMAT is unchanged, deliberately -- `fit_testbox score`,
`gate_sweep_v2` and `calibrate_psf` all read these files and none of them
needed a line changed.

TWO SCOPES
----------
    --rounds all         every traced round; what the replicate score needs
    --rounds reference   the reference hybe only

The second exists because the PSF question does not need the other
hundred rounds. Calibration only ever looks at reference-hybe crops, so
harvesting the rest to answer it is 40-100x the I/O for nothing.

PARALLELISM
-----------
One allele is one work item. Items are sorted by FOV so a worker walking
consecutive items stays on one FOV -- `_activate_fov` reloads FOV-level
state, and paying that per allele instead of per block is pure waste.

Worker count is NOT cpu_count. This is NAS-bound work, and this repo has
already measured what that does: ingestion got 117.6 MB/s at 12 workers
and 66 MB/s at 36. `parallel.pmap(kind='io')` defaults accordingly, and
`--jobs` overrides it for the sweep that checks the default still holds
on this shape of read.

Usage:
    python tools/harvest.py --exp Chr19 --alleles 68 --out bench.npz
    python tools/harvest.py --exp HoxA --rounds reference --alleles 8 --jobs 1
"""
import argparse
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline import parallel as PL                     # noqa: E402
from tools.experiments import EXPERIMENTS, open_session, config_fovs   # noqa: E402
from tools.fit_testbox import anchor_label, replicate_pairs     # noqa: E402


# -- worker ---------------------------------------------------------------

def _init(exp_name, rounds):
    """Build ONE session per worker and keep it. Constructing a MainWindow
    and reading the layout is the fixed cost of a worker; doing it per
    allele would dominate everything else."""
    from codelab_pipeline.localization import localization as L   # noqa: F401
    exp = EXPERIMENTS[exp_name]
    mw, sp = open_session(exp)
    records = mw._hybe_records_for_storage_path(sp)
    by_folder = {r['folder']: r for r in records}
    traced = [r['folder'] for r in records
              if str(r['datatype']).upper() in exp.datatypes]
    hybes = [exp.reference_hybe] if rounds == 'reference' else traced
    fid_ch = {h: by_folder[h]['fiducial_channel'] for h in hybes}
    read_ch = {}
    for h in hybes:
        others = [c for c in by_folder[h]['channels']
                  if c != by_folder[h]['fiducial_channel']]
        read_ch[h] = others[0] if others else by_folder[h]['fiducial_channel']
    return {'exp': exp, 'mw': mw, 'sp': sp, 'hybes': hybes,
            'fid_ch': fid_ch, 'read_ch': read_ch, 'fov': None}


def _one(item, st):
    """One allele, through the production path. Returns only picklable data."""
    from codelab_pipeline.localization import localization as L
    from codelab_pipeline.models.allele import AnAllele
    i, fov, d = item
    mw, exp = st['mw'], st['exp']

    # Only when it CHANGES. Sorting items by FOV upstream makes this rare.
    if st['fov'] != fov:
        mw._activate_fov(fov)
        st['fov'] = fov

    allele = AnAllele()
    allele.set_metadata(id=i, fov=fov, cell=d['cell'], anchor_uid=d.get('uid', 0),
                        anchor_hybe=d['hybe'], anchor_channel=d['channel'],
                        coordinate=d['adj_coordinate'],
                        raw_coordinate=d['raw_coordinate'])
    cell = mw._find_cell_by_id(fov, allele.cell) if allele.cell != -1 else None
    resolver = mw._frame_resolver(cell, fov)
    fov_matrices = mw._composed_fov_matrices_for_cell_alignment(st['sp'], fov)

    t0 = time.perf_counter()
    _a, debug = L.build_chromatin_trace_allele(
        allele, st['hybes'], exp.reference_hybe, st['fid_ch'], st['read_ch'],
        st['sp'], fov, exp.modality, cell, fov_matrices, spad=8, z_window=15,
        z_boundary_trim=0, collect_debug=True, resolver=resolver)
    dt = time.perf_counter() - t0

    frozen, zpred = {}, {}
    for hybe, dbg in (debug or {}).items():
        off = L.cell_z_offset(cell, hybe, exp.modality, resolver)
        zpred[f'a{i}|{hybe}'] = (float(allele.coordinate[2]) - float(off),
                                 float(off))
        for kind in ('fiducial', 'readout'):
            cube = dbg.get(f'{kind}_cubic')
            if cube is not None:
                frozen[f'a{i}|{hybe}|{kind}'] = np.asarray(cube, dtype=np.float32)
    return {'i': i, 'fov': fov, 'uid': d.get('uid', 0), 'cell': d['cell'],
            'kept': len(frozen), 'label': f'a{i}|{anchor_label(fov, d)}',
            'frozen': frozen, 'zpred': zpred, 'seconds': dt}


# -- driver ---------------------------------------------------------------

def choose(exp, n_alleles, fovs):
    """Stratified per FOV, never a flat draw.

    A flat sample of 12 from 439 once put both of a probe's alleles in one
    FOV, and a bench that silently covers one FOV cannot show a per-FOV
    effect -- segmentation, drift and focus all vary FOV to FOV.
    """
    from codelab_pipeline.io import analysis_store as V
    mw, sp = open_session(exp)
    rng = random.Random(exp.seed)
    per = max(1, n_alleles // len(fovs))
    chosen, counts = [], {}
    for fov in fovs:
        pool = V.read_spots(sp, fov, exp.modality, exp.anchor_hybe,
                            exp.anchor_channel)
        counts[fov] = len(pool)
        for d in rng.sample(pool, min(per, len(pool))):
            chosen.append((fov, d))
    return chosen, counts, sp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exp', required=True, choices=sorted(EXPERIMENTS))
    ap.add_argument('--out', default=None)
    ap.add_argument('--alleles', type=int, default=48)
    ap.add_argument('--rounds', choices=['all', 'reference'], default='all')
    ap.add_argument('--jobs', type=int, default=None,
                    help='workers; default is the measured io default')
    a = ap.parse_args()

    exp = EXPERIMENTS[a.exp]
    fovs = list(exp.fovs) if exp.fovs else list(config_fovs(exp.config))
    chosen, counts, sp = choose(exp, a.alleles, fovs)
    # FOV-major: consecutive items share a FOV, so _activate_fov is paid
    # per block rather than per allele.
    chosen.sort(key=lambda fd: fd[0])
    items = [(i, fov, d) for i, (fov, d) in enumerate(chosen, start=1)]

    out = a.out or os.path.join('bench', f'{a.exp.lower()}_{len(items)}.npz')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)

    jobs = PL.cpu_budget('io', a.jobs, len(items))
    print(f'experiment : {exp.name}  ({exp.scope_mb} Mb scope)')
    print(f'store      : {sp}')
    print(f'candidates : ' + '  '.join(f'FOV{f:03d}:{c}' for f, c in sorted(counts.items())))
    print(f'selected   : {len(items)} alleles over {len(fovs)} FOV (seed {exp.seed})')
    print(f'rounds     : {a.rounds}')
    print(f'workers    : {jobs}')
    print(f'out        : {out}\n', flush=True)

    t0 = time.perf_counter()
    done_t = []

    def progress(n, total, _idx, r):
        if isinstance(r, PL.Failure):
            print(f'  [{n:>4}/{total}] FAILED {r.error}', flush=True)
            return
        done_t.append(r['seconds'])
        if n % 5 == 0 or n == total:
            el = time.perf_counter() - t0
            rate = n / el
            print(f'  [{n:>4}/{total}] {el:6.0f}s elapsed  '
                  f'{rate * 60:5.1f} alleles/min  '
                  f'eta {(total - n) / rate / 60:5.1f} min', flush=True)

    results = PL.pmap(_one, items, kind='io', jobs=jobs,
                      initializer=_init, initargs=(a.exp, a.rounds),
                      on_done=progress, chunksize=4)
    wall = time.perf_counter() - t0

    bad = PL.failures(results)
    good = PL.ok(results)
    frozen, zpred, meta, labels = {}, {}, [], []
    for r in sorted(good, key=lambda r: r['i']):
        frozen.update(r['frozen'])
        zpred.update(r['zpred'])
        meta.append((r['i'], r['fov'], r['uid'], r['cell'], r['kept']))
        labels.append(r['label'])

    mw, sp2 = open_session(exp)
    records = mw._hybe_records_for_storage_path(sp2)
    hybes = ([exp.reference_hybe] if a.rounds == 'reference'
             else [r['folder'] for r in records
                   if str(r['datatype']).upper() in exp.datatypes])

    np.savez_compressed(
        out,
        __meta__=np.array(meta, dtype=np.int64),
        __hybes__=np.array(hybes),
        __pairs__=np.array([f'{x}|{y}|{r}' for x, y, r in replicate_pairs(records)]),
        __zpred__=np.array([f'{k}|{v[0]:.4f}|{v[1]:.4f}' for k, v in zpred.items()]),
        __labels__=np.array(labels),
        **frozen)

    cpu = sum(done_t)
    print(f'\nfroze {len(frozen)} crops from {len(meta)} alleles -> {out} '
          f'({os.path.getsize(out) / 1e6:.1f} MB)')
    print(f'wall {wall:.0f}s   summed per-allele build time {cpu:.0f}s   '
          f'speedup {cpu / wall:.1f}x on {jobs} workers')
    if bad:
        print(f'{len(bad)} alleles FAILED:')
        for f in bad[:5]:
            print(f'   {f.error}')


if __name__ == '__main__':
    main()
