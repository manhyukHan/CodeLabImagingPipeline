"""
Build a review bundle from a store, and write down exactly what was built.

The bundle is what ten people spend a week judging, so which FOVs and
which hybes went into it is not a detail to reconstruct later from shard
filenames. This writes bundle_manifest.json beside the shards: the store,
the hybes, the channel, the FOVs AND THE SEED that chose them, the engine
and its anchor settings, the wall clock, and what came out.

WHY A SEED RATHER THAN A LIST. Picking FOVs by hand biases the set toward
whatever was on screen. Picking them randomly and recording the seed
gives an unbiased draw that anyone can reproduce exactly -- including
"extend it to eight FOVs" without re-drawing the four already reviewed,
since the same seed prefixes the same sequence.

FEWER FOVs, MORE HYBES is usually the right shape. A detector learns
from the variety of SIGNAL, and a hybe is a different probe on the same
cells while a FOV is the same probes on different cells; 4 FOVs x 12
hybes covers twelve experiments at ~480 cells each, where 12 FOVs x 4
hybes covers four.

Reads the store. Writes ONLY inside --out.

  python tools/build_bundle.py G:/Seonghyeok/2025-11-30-MP58/RNA \\
      --hybes all --channel 635 --n-fovs 4 --seed 20260907 \\
      --out D:/bundles/MP58_RNA_all_4fov
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline.io import paths                        # noqa: E402
from codelab_pipeline.localization import engine as E        # noqa: E402
from codelab_pipeline.training import extract as X           # noqa: E402


# Which acquisitions are IMAGING ROUNDS, as the store's own stacks say.
#
#   H  hyb        the readout itself
#   R  replicate  a hybe imaged a second time -- the same signal again
#   T  toehold    after strand displacement, where the signal should be
#                 GONE
#   B  barcode    celltype calling, not a spot readout
#
# Default H,R. The toehold is the interesting exclusion. A generous
# anchor CANNOT TELL IT APART: measured over the 420 MP58/RNA Toe_133
# crops of this draw, it anchors a median of 64 candidates a crop against
# 103 for the eleven real hybes -- the same order, not the empty frame
# one might expect. So its whole 420 crops would enter the review queue
# looking exactly like work, and be work, for a round whose answer is
# already known. A toehold is worth having as a NEGATIVE CONTROL rather
# than as a readout, and the ones to use for that are on the DNA side.
DEFAULT_DATATYPES = 'H,R'


def hybe_datatype(storage_path, fov, hybe):
    """What the stack file says it is -- 'H', 'R', 'T', 'B', or ''."""
    import h5py
    try:
        with h5py.File(paths.stack_path(storage_path, fov, hybe), 'r') as f:
            dt = f.attrs.get('datatype', '')
    except Exception:                                        # noqa: BLE001
        return ''
    return dt.decode() if isinstance(dt, bytes) else str(dt)


def hybes_in(storage_path, fov, datatypes=None):
    """Every hybe this FOV has a stack for, filtered by datatype.

    Filtering on the stack's own attribute rather than on the name: a
    prefix convention is a habit, and 'Toe_133' being a toehold is a fact
    the file records.
    """
    d = os.path.dirname(paths.stack_path(storage_path, fov, 'x'))
    names = sorted(n[:-3] for n in os.listdir(d) if n.endswith('.h5'))
    if not datatypes:
        return names
    want = {s.strip().upper() for s in str(datatypes).split(',') if s.strip()}
    return [h for h in names
            if hybe_datatype(storage_path, fov, h).upper() in want]


def fovs_with_cells(storage_path, candidates):
    """(fov, n_cells) for the FOVs that carry a segmentation.

    A FOV with no masks is not an error and not a gap in the data -- it
    is one nobody has segmented yet -- but drawing it into the sample
    would silently shrink the sample.
    """
    out = []
    for fov in candidates:
        try:
            ids = [c[0] for c in X.cell_masks(storage_path, fov)]
        except Exception:                                    # noqa: BLE001
            continue
        if ids:
            out.append((int(fov), len(ids)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('storage_path')
    ap.add_argument('--out', required=True)
    ap.add_argument('--channel', type=int, required=True)
    ap.add_argument('--hybes', default='all',
                    help='"all" for every hybe of the wanted datatypes, or '
                         'a comma list (a named hybe is taken as asked for, '
                         'whatever its datatype)')
    ap.add_argument('--datatypes', default=DEFAULT_DATATYPES,
                    help='which acquisitions count as readouts, by the '
                         "stack's own datatype attribute: H hyb, R "
                         'replicate, T toehold, B barcode. '
                         f'Default {DEFAULT_DATATYPES}. Pass "" for all.')
    ap.add_argument('--n-fovs', type=int, default=None,
                    help='draw this many FOVs at random from those that '
                         'carry cell masks')
    ap.add_argument('--fovs', default=None,
                    help='comma list, instead of a random draw')
    ap.add_argument('--fov-pool', default='1-41',
                    help='which FOVs to consider, e.g. "1-41" or "1,2,7"')
    ap.add_argument('--seed', type=int, default=None,
                    help='the draw is recorded with this; default is a '
                         'timestamp, which is still written down')
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--chunk', type=int, default=X.DEFAULT_CHUNK)
    ap.add_argument('--pad', type=int, default=X.DEFAULT_PAD)
    ap.add_argument('--dry-run', action='store_true',
                    help='say what would be built, touch nothing')
    a = ap.parse_args(argv)

    store = str(a.storage_path)
    if not os.path.isdir(store):
        print('no such store:', store)
        return 1

    pool = []
    for part in str(a.fov_pool).split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-')
            pool.extend(range(int(lo), int(hi) + 1))
        elif part:
            pool.append(int(part))

    print(f'store   {store}')
    print('scanning for cell masks ...', flush=True)
    t = time.time()
    have = fovs_with_cells(store, pool)
    print(f'   {len(have)} of {len(pool)} FOVs carry masks, '
          f'{sum(n for _, n in have)} cells   ({time.time() - t:.1f}s)')
    if not have:
        print('nothing to extract')
        return 1

    if a.fovs:
        want = [int(x) for x in a.fovs.split(',') if x.strip()]
        seed = None
        missing = [f for f in want if f not in {g for g, _ in have}]
        if missing:
            print('these FOVs have no cell masks:', missing)
            return 1
        fovs = want
    else:
        n = int(a.n_fovs or 4)
        seed = int(a.seed) if a.seed is not None else int(time.time())
        # Sample from the FOVs that HAVE masks, sorted, so the same seed
        # gives the same draw on any machine.
        fovs = sorted(random.Random(seed).sample(
            sorted(g for g, _ in have), min(n, len(have))))

    ncell = {g: c for g, c in have}
    hybes = (hybes_in(store, fovs[0], a.datatypes) if a.hybes == 'all'
             else [h.strip() for h in a.hybes.split(',') if h.strip()])
    dts = {h: hybe_datatype(store, fovs[0], h) for h in hybes}
    # A hybe missing from one of the drawn FOVs would fail mid-run.
    for f in fovs[1:]:
        here = set(hybes_in(store, f, a.datatypes))
        gone = [h for h in hybes if h not in here]
        if gone:
            print(f'   fov{f:03d} is missing {gone} -- dropping them')
            hybes = [h for h in hybes if h in here]

    cells = sum(ncell[f] for f in fovs)
    print(f'\nFOVs    {fovs}' + (f'   (random, seed {seed})' if seed else
                                 '   (given)'))
    print(f'        cells per FOV: '
          + ', '.join(f'{f}:{ncell[f]}' for f in fovs) + f'   total {cells}')
    print(f'hybes   {len(hybes)}  (datatypes {a.datatypes or "all"})')
    for h in hybes:
        print(f'          {h:10s} {dts.get(h, "?")}')
    print(f'channel {a.channel}')
    print(f'out     {a.out}')
    print(f'\nwork    {len(fovs)} FOVs x {len(hybes)} hybes = '
          f'{cells * len(hybes):,} crops')
    if a.dry_run:
        print('\n(dry run -- nothing written)')
        return 0

    os.makedirs(str(a.out), exist_ok=True)
    run = dict(
        storage_path=store, out=str(a.out), channel=int(a.channel),
        hybes=hybes, hybe_datatypes=dts, datatypes=str(a.datatypes),
        fovs=fovs, fov_seed=seed, fov_pool=str(a.fov_pool),
        cells_per_fov={str(f): ncell[f] for f in fovs},
        pad=int(a.pad), chunk=int(a.chunk), workers=a.workers,
        engine='anchor-v2', anchor=dict(E.GENEROUS_ANCHOR),
        started=time.strftime('%Y-%m-%dT%H:%M:%S'))
    mpath = os.path.join(str(a.out), 'bundle_manifest.json')
    # ONE BUNDLE, ONE MANIFEST, ONE ENTRY PER RUN. A bundle may now hold
    # several channels (each build adds shards named for its own), and a
    # manifest that was simply rewritten described only the last one --
    # so a two-channel bundle claimed to be a one-channel bundle, and
    # the reviewer's own record of what they were reviewing was wrong.
    # The newest run stays at the top level so every existing reader
    # keeps working unchanged; `runs` is what makes the file true.
    manifest = dict(run)
    prior = []
    try:
        with open(mpath, encoding='utf-8') as f:
            was = json.load(f)
        prior = list(was.get('runs') or [])
        if not prior and was.get('channel') is not None:
            prior = [{k: v for k, v in was.items() if k != 'runs'}]
    except Exception:                                        # noqa: BLE001
        prior = []
    # Same channel built again REPLACES its own entry rather than
    # stacking duplicates: the shards were just overwritten too.
    prior = [r for r in prior if int(r.get('channel', -1)) != int(a.channel)]
    manifest['runs'] = prior + [run]
    manifest['channels'] = sorted({int(r['channel']) for r in manifest['runs']})
    # Written BEFORE the run, so an interrupted bundle still says what it
    # was trying to be. Completion is stamped on at the end.
    with open(mpath, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    t0 = time.time()
    state = {'crops': 0, 'cands': 0}

    def on_task(done, total, fov, hybe, path, n_crops, n_cands):
        state['crops'] += n_crops
        state['cands'] += n_cands
        el = time.time() - t0
        rate = done / max(el, 1e-9)
        eta = (total - done) / rate if rate else 0
        print(f'  [{done:4d}/{total}] fov{fov:03d} {hybe:9s} '
              f'{n_crops:3d} crops {n_cands:6d} cand   '
              f'{el / 60:5.1f} min elapsed, ~{eta / 60:5.1f} left',
              flush=True)

    print()
    X.extract(store, fovs, hybes, int(a.channel), str(a.out),
              workers=a.workers, chunk=int(a.chunk), pad=int(a.pad),
              on_task=on_task)
    wall = time.time() - t0

    summary = X.summarize(str(a.out))
    manifest.update(finished=time.strftime('%Y-%m-%dT%H:%M:%S'),
                    wall_seconds=round(wall, 1), summary=summary,
                    complete=True)
    with open(mpath, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f'\ndone in {wall / 60:.1f} min')
    print(f"   {summary['crops']:,} crops   {summary['candidates']:,} candidates"
          f"   {summary['bytes'] / 2**30:.2f} GiB")
    print(f"   candidates per crop: median "
          f"{summary['candidates_per_crop_median']:.0f}, "
          f"p90 {summary['candidates_per_crop_p90']:.0f}")
    print(f'   manifest: {mpath}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
