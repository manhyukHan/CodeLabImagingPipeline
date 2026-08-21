"""
Performance harness: times the REAL operations the pipeline actually
performs, against any store, so every storage-refactor phase lands with
before/after numbers on record.

    python tools/perf_harness.py <storage_path> [--fov N] [--json out.json]

<storage_path> is a queue/modality dir exactly as the app takes it (the
harness resolves vlinks/stacks through the same vlinks_store/spot_mapper
doors the app uses, so a storage-layout change is measured through the
code that serves it, not through hand-rolled paths).

READ-ONLY by default. --allow-writes additionally times the write path
(full cells-blob write) and REFUSES unless the store path contains
'veri_data', 'clone', 'scratch', or 'bench' -- never point writes at
real data.

Operations timed (median of N):
  open_vlinks      one guarded vlinks open+attr read
  read_cells       whole-FOV cells read (blob decode included)
  read_spots       whole-FOV spots read (all slices)
  crop3d           one 17x17xZ stack crop (spot_mapper, use_stack=True)
  trace20          20 crops across alternating hybes -- the per-allele
                   chromatin-trace access shape
  checkup          the ingestion-status pattern for ONE fov: per-hybe
                   MIP-presence checks (mip_channels_present x hybes)
  layout_parse     ExperimentLayout parse (when a layout is findable)
  write_cells      (--allow-writes only) one full cells-blob write
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np

from codelab_pipeline.io import vlinks_store, preprocess
from codelab_pipeline.alignment import spot_mapper


def timed(fn, repeat=5):
    out = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return statistics.median(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('storage_path')
    ap.add_argument('--fov', type=int, default=1)
    ap.add_argument('--json', default=None)
    ap.add_argument('--allow-writes', action='store_true')
    args = ap.parse_args()
    sp, fov = args.storage_path, args.fov

    results = {}

    def rec(name, seconds, note=''):
        results[name] = {'ms': seconds * 1000.0, 'note': note}
        print(f'  {name:<14} {seconds*1000:9.1f} ms  {note}')

    print(f'perf harness on {sp} (FOV{fov:02d})')

    vp = vlinks_store._vlinks_path(sp)
    rec('open_vlinks', timed(lambda: h5py.File(vp, 'r').close(), 10))

    cells_holder = {}
    def read_cells():
        cells_holder['dicts'], _ = vlinks_store.read_cells(sp, fov)
    rec('read_cells', timed(read_cells),
        f"{len(cells_holder['dicts'] or [])} cells, blob {os.path.getsize(vp)//1_000_000} MB store")

    spots_holder = {}
    def read_spots():
        spots_holder['d'] = vlinks_store.read_spots(sp, fov)
    rec('read_spots', timed(read_spots), f"{len(spots_holder['d'])} spots")

    # a real (hybe, channel) with a stack file on disk, from the store
    # itself -- layout-aware (io/paths.py): v2 keeps stacks under
    # sp/stacks/FOV##/{hybe}.h5, v1 under sp/FOV##/{hybe}_stack.h5
    from codelab_pipeline.io import paths
    if paths.is_v2(sp):
        fov_dir = os.path.join(sp, 'stacks', f'FOV{fov:02d}')
        stacks = sorted(f for f in os.listdir(fov_dir) if f.endswith('.h5')) if os.path.isdir(fov_dir) else []
        hybes = [s[:-3] for s in stacks]
    else:
        fov_dir = os.path.join(sp, f'FOV{fov:02d}')
        stacks = sorted(f for f in os.listdir(fov_dir) if f.endswith('_stack.h5')) if os.path.isdir(fov_dir) else []
        hybes = [s[:-len('_stack.h5')] for s in stacks]
    if stacks:
        with h5py.File(os.path.join(fov_dir, stacks[0]), 'r') as f:
            channels = [int(k[2:]) for k in f['stack'].keys()]
            shape = f['stack'][f'ch{channels[0]}'].shape
            chunks = f['stack'][f'ch{channels[0]}'].chunks
        ch = channels[0]
        rng = np.random.default_rng(0)

        def one_crop():
            y, x = float(rng.uniform(100, shape[0] - 100)), float(rng.uniform(100, shape[1] - 100))
            spot_mapper.crop_for_localization(sp, fov, hybes[0], ch, (y, x), pad=8, use_stack=True)
        rec('crop3d', timed(one_crop), f'stack {shape} chunks={chunks}')

        def trace20():
            for i in range(20):
                h = hybes[i % len(hybes)]
                y, x = float(rng.uniform(100, shape[0] - 100)), float(rng.uniform(100, shape[1] - 100))
                try:
                    spot_mapper.crop_for_localization(sp, fov, h, ch, (y, x), pad=8, use_stack=True)
                except (OSError, KeyError):
                    pass
        rec('trace20', timed(trace20, 3), f'20 crops over {min(20, len(hybes))} hybe file(s)')

        def checkup():
            for h in hybes:
                vlinks_store.mip_channels_present(sp, fov, h)
        rec('checkup', timed(checkup), f'{len(stacks)} per-hybe MIP-presence checks (1 FOV)')

    layout = None
    for cand in ('data/raw_subset/RNA_Expt/Results/ExperimentLayout.xlsx',):
        if os.path.exists(cand):
            layout = cand
    if layout:
        rec('layout_parse', timed(lambda: preprocess.parse_experiment_layout(layout), 3), layout)

    if args.allow_writes:
        assert any(k in sp for k in ('veri_data', 'clone', 'scratch', 'bench')), \
            '--allow-writes refused: store path does not look like a clone'
        from codelab_pipeline.models.cell_container import CellContainer
        dicts, modality = vlinks_store.read_cells(sp, fov)
        cont = CellContainer.load({fov: dicts}, modality=modality)
        rec('write_cells', timed(lambda: vlinks_store.write_cells(sp, fov, cont), 3),
            f'full blob rewrite, {len(dicts or [])} cells')

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'storage_path': sp, 'fov': fov, 'results': results}, f, indent=2)
        print(f'wrote {args.json}')


if __name__ == '__main__':
    main()
