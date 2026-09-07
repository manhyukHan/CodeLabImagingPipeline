"""
Build review bundles from a store: cell crops + generous anchor-fit.

THE SHAPE OF THE WORK. One task is one (fov, hybe, chunk of ~8 cells),
dispatched FOV-major through ONE shared pool -- never a pool per
modality, never a pass per hybe -- the same rule ingestion follows.

The task used to be a whole FOV, sized against the assumption that
reading dominates. It does not. MEASURED per crop, MP58/RNA Hyb_109
ch635, on an open stack handle:

    read    0.03 s
    anchor  0.07 s
    FIT     4.54 s      99.8% of the work, at 0.127 s a fit

That 673 ms-a-crop figure from the bundle-vs-store comparison is real but
it is the cost of OPENING a 266 MB gzipped stack and pulling one window
cold -- not of reading the next 120 windows from the handle already open.
So batching whole FOVs bought almost nothing and cost a great deal: one
FOV is 121 cells at ~6 s, a 12-minute serial chain, and a pool over 20 of
those spends most of its life waiting on the slowest. Chunked, the pool
stays full and the tail is one chunk long.

This workload is CPU bound, which is the opposite of the alignment path
-- that one is bandwidth bound and measured FASTER at 3 workers than at
6. Do not carry that worker count over here.

GENEROSITY IS THE POINT AND IT IS COSTLY. At mode+1sigma the median real
cell yields 21 candidates and the worst yields 156, and each v2 fit is
~110 ms. Two caps keep that bounded without raising the threshold (which
would drop the dim spots this whole exercise exists to learn): `max_fits`
bounds how many anchors are fitted at all, brightest first, and
`keep_top` bounds how many reach the bundle. Neither is a quality filter
-- both are effort limits, and both are recorded in the bundle's meta so
a later reader knows the list was truncated rather than exhausted.

`keep_top` is applied to the COMBINED list, not inside the engine. The
engine caps what it fitted; the unfitted anchors are appended after, and
capping inside let them past the limit -- which is how the first real run
asked for 12 candidates and produced a median of 33.

WHAT IT WILL NOT DO. It never writes to the store. It reads cells,
matrices and stacks; the only thing it creates is the bundle directory.
"""
import os
import time

import numpy as np

from ..io import analysis_store, paths
from ..localization import engine as E
from . import bundle as B

# The cell bbox plus this much on every side. Widened from 10 on
# 2026-09-07 together with dropping the mask from candidate generation:
# both exist so a spot sitting on or just past the segmented boundary --
# where a segmentation slip or a small inter-hybe alignment residual puts
# it -- is still in frame and still proposed.
DEFAULT_PAD = 14

# How many anchors get fitted at all, brightest first. NOT a quality
# filter -- an effort limit -- and it was set to 40 against the MAZ
# store, where a crop yields ~21 anchors and every gate-passing spot came
# from brightness rank <= 21. Applying that number to MP58/RNA, which is
# roughly four times denser, was the mistake CLAUDE.md warns about:
#
#   anchors per crop, MP58, all 2266 crops:
#       p25 35   MEDIAN 90   p75 174   p90 239   max 464
#   crops where max_fits=40 binds:            1625 / 2266  (71.7%)
#   anchors never fitted because of it:     173076 / 250481 (69.1%)
#   production-passing spots NEVER PROPOSED:  31 / 534  (5.8%)
#       their anchor brightness rank: median 88, p90 147, max 215
#       their p: 0.023 .. 0.785
#
# A spot never proposed is a label that can never be collected, and the
# ones lost are systematically the dim ones -- exactly what a learned
# detector is being built to find. 250 covers the observed max rank with
# headroom. Re-measure on any store whose density is unknown; do not
# assume this number transfers either.
DEFAULT_MAX_FITS = 250
DEFAULT_KEEP_TOP = 24


def cell_masks(storage_path, fov):
    """[(cell_id, y_area, x_area, frame_shape), ...] for one FOV.

    Reads the persisted cell dicts rather than building ACell objects:
    the extractor needs a mask and a bbox, and nothing else a cell knows.
    The mask is stored and drawn but NOT used to filter candidates -- see
    extract_fov.
    Going through CellContainer would drag in matrix resolution that this
    path has no use for -- crops here are in the HYBE'S OWN native frame,
    deliberately, because that is the frame a learned detector will be
    handed at inference time. Alignment is applied to the coordinates
    afterwards, by whoever needs them in a shared frame.
    """
    dicts, _ = analysis_store.read_cells(storage_path, fov)
    out = []
    for c in dicts or []:
        ya, xa = c['area']
        if len(ya) == 0:
            continue
        out.append((int(c['id']), np.asarray(ya).astype(int),
                    np.asarray(xa).astype(int), tuple(c['frame_shape'])))
    return out


def _read_crops(stack_path, channel, cells, pad):
    """Every cell's crop out of ONE open stack file.

    h5py fancy-indexing needs ascending indices, which np.where output is
    not, so the rectangle is sliced first (always ascending) and the mask
    applied to the in-memory array -- the same order cell_crop uses, for
    the same reason.
    """
    import h5py
    out = []
    with h5py.File(stack_path, 'r') as f:
        name = f'/stack/ch{channel}'
        if name not in f:
            return out
        ds = f[name]
        H, W = ds.shape[0], ds.shape[1]
        for (cid, ya, xa, _shape) in cells:
            y0, y1 = max(0, ya.min() - pad), min(H, ya.max() + pad + 1)
            x0, x1 = max(0, xa.min() - pad), min(W, xa.max() + pad + 1)
            if y1 <= y0 or x1 <= x0:
                continue
            block = ds[y0:y1, x0:x1, :]
            mask = np.zeros(block.shape[:2], np.uint8)
            mask[ya - y0, xa - x0] = 1
            out.append((cid, y0, x0, block, mask))
    return out


def candidates_for(stack, engine, max_fits, anchor, keep_top=None):
    """[(y, x, z, p, fit_ok, gate_pass, reason), ...] for one crop.

    `stack` is the WHOLE padded rectangle, mask NOT applied. The
    parameter was called stack_masked until 2026-09-07 and that is no
    longer what it is: the mask cuts FOV-scale background down to a
    region worth looking at, and does not decide whose spot this is.

    Anchors are found ONCE here and handed to the engine, not found here
    and then found again inside it -- the earlier version did exactly
    that, paying for the anchor pass twice per crop.

    They are capped before fitting, brightest first, because fitting is
    the expensive half: at ~110 ms a fit, the worst real cell measured
    (156 anchors) would cost 17 s on its own.

    Anchors the fit could not use are still reported, at the anchor's own
    position with fit_ok=0. A person can look at one and say whether
    there was a spot there, and if there was, that is precisely the case
    the current pipeline gets wrong -- the fit is bounded to the anchor,
    so nothing downstream can recover a spot the fit gave up on.

    keep_top caps the COMBINED list. It has to be applied here rather
    than inside the engine: the engine caps only what it fitted, and the
    unfitted anchors appended afterwards would otherwise sail past it --
    which is why an earlier run asked for 12 and produced a median of 33.
    """
    anchors = E.anchor_candidates(stack, n_max=max_fits, **anchor)
    if not anchors:
        return []
    detailed = engine.localize_detailed(stack, seeds=anchors,
                                        n_max=max_fits)
    out = [(s.y, s.x, s.z, s.p, 1, 1 if ok else 0, why)
           for (s, ok, why) in detailed]
    taken = [(s.y, s.x) for (s, _o, _w) in detailed]
    for (ay, ax, az) in anchors:
        if any((ay - ty) ** 2 + (ax - tx) ** 2 < 9 for (ty, tx) in taken):
            continue
        out.append((ay, ax, az, 0.0, 0, 0, 'no fit'))
    if keep_top:
        # GATE-PASS FIRST, then fitted, then by p.
        #
        # Sorting on (fit_ok, p) alone dropped production-passing spots:
        # p and the gate disagree, and MEASURED on a real bundle the
        # gate-pass candidates a wider search recovered had p from 0.023
        # to 0.785, median 0.061 -- far below the junk that outranks them.
        # A spot the production gate would accept must never be evicted
        # by an effort cap, because a candidate never shown is a label
        # that can never be collected.
        out.sort(key=lambda c: (-c[5], -c[4], -c[3]))
        out = out[:int(keep_top)]
    return out


DEFAULT_CHUNK = 8


def extract_chunk(storage_path, fov, hybe, channel, cell_ids, out_dir, tag,
                  pad=DEFAULT_PAD, max_fits=DEFAULT_MAX_FITS,
                  keep_top=DEFAULT_KEEP_TOP, anchor=None,
                  engine_name='anchor-v2', meta=None):
    """One (fov, hybe, few cells) -> one shard. THE unit of parallel work.

    It used to be one whole FOV, and that was sized against the wrong
    cost. MEASURED per crop on MP58/RNA Hyb_109 ch635:

        read    0.03 s      one open stack, a cell-sized window
        anchor  0.07 s
        FIT     4.54 s      <- 99.8% of it, at 0.127 s a fit

    Reading is not the cost, so there is nothing to be gained by keeping
    a whole FOV in one task -- and a great deal lost: a FOV is 121 cells
    at ~6 s, so one task was a 12-minute serial chain and 20 FOVs across
    a pool left most workers idle waiting on the slowest. In chunks the
    same pool stays full and the tail is one chunk long, not one FOV.

    Each chunk reopens the stack file. That costs a file open per chunk
    instead of per FOV, which against 0.03 s of reading and minutes of
    fitting is not a trade worth thinking about.
    """
    anchor = dict(anchor if anchor is not None else E.GENEROUS_ANCHOR)
    engine = E.make_engine(engine_name, anchor=anchor)
    wanted = set(int(c) for c in cell_ids)
    cells = [c for c in cell_masks(storage_path, fov) if c[0] in wanted]
    path = os.path.join(str(out_dir), f'fov{int(fov):03d}__{hybe}__{tag}.h5')
    info = dict(meta or {})
    info.update(fov=int(fov), hybe=str(hybe), channel=int(channel),
                pad=int(pad), max_fits=int(max_fits), keep_top=int(keep_top),
                engine=engine_name, anchor=anchor,
                cells=sorted(wanted), storage_path=str(storage_path))
    sp = paths.stack_path(storage_path, fov, hybe)
    if not cells or not os.path.exists(sp):
        return path, 0, 0
    try:
        crops = _read_crops(sp, channel, cells, pad)
    except OSError:
        return path, 0, 0          # a broken stack is a skip, not a crash
    if not crops:
        return path, 0, 0
    n_crop = n_cand = 0
    with B.BundleWriter(path, meta=info) as w:
        for (cid, y0, x0, block, mask) in crops:
            cands = candidates_for(block.astype(float), engine, max_fits,
                                   anchor, keep_top=keep_top)
            w.add(fov, hybe, channel, cid, block, mask, y0, x0, cands)
            n_crop += 1
            n_cand += len(cands)
    return path, n_crop, n_cand


def plan(storage_path, fovs, hybes, channel, chunk=DEFAULT_CHUNK):
    """[(fov, hybe, cell_ids, tag), ...] -- every task, FOV-major.

    FOV-major ordering, one shared pool: the same rule ingestion follows.
    Ordering matters even with balanced chunks, because a run stopped
    half way then has whole FOVs finished rather than a scatter of
    fragments across all of them.
    """
    tasks = []
    for fov in [int(f) for f in fovs]:
        ids = [c[0] for c in cell_masks(storage_path, fov)]
        if not ids:
            continue
        for hybe in hybes:
            for k in range(0, len(ids), int(chunk)):
                part = ids[k:k + int(chunk)]
                tasks.append((fov, hybe, part, f'c{k // int(chunk):03d}'))
    return tasks


def extract_fov(storage_path, fov, hybes, channel, out_dir, pad=DEFAULT_PAD,
                max_fits=DEFAULT_MAX_FITS, keep_top=DEFAULT_KEEP_TOP,
                anchor=None, engine_name='anchor-v2', meta=None):
    """One FOV -> one shard. Returns (path, n_crops, n_candidates).

    Runs in a worker process; everything it needs is picklable and it
    returns counts, not pixels.
    """
    anchor = dict(anchor if anchor is not None else E.GENEROUS_ANCHOR)
    # keep_top is deliberately NOT handed to the engine. The cap has to
    # see the unfitted anchors as well, and the engine never sees those --
    # capping inside it let them append past the limit, which is how a run
    # asking for 12 produced a median of 33.
    engine = E.make_engine(engine_name, anchor=anchor)
    cells = cell_masks(storage_path, fov)
    path = os.path.join(str(out_dir), f'fov{int(fov):03d}.h5')
    n_crop = n_cand = 0
    info = dict(meta or {})
    info.update(fov=int(fov), channel=int(channel), pad=int(pad),
                max_fits=int(max_fits), keep_top=int(keep_top),
                engine=engine_name, anchor=anchor,
                storage_path=str(storage_path))
    if not cells:
        return path, 0, 0
    with B.BundleWriter(path, meta=info) as w:
        for hybe in hybes:
            sp = paths.stack_path(storage_path, fov, hybe)
            if not os.path.exists(sp):
                continue
            try:
                crops = _read_crops(sp, channel, cells, pad)
            except OSError:
                continue          # a broken stack is a skip, not a crash
            for (cid, y0, x0, block, mask) in crops:
                # CANDIDATES COME FROM THE WHOLE PADDED RECTANGLE, mask
                # not applied. The cell mask is here to cut FOV-scale
                # background down to a region worth looking at, not to
                # decide which cell a spot belongs to -- that is a
                # question for the analysis path, and this is a training
                # set of biological spots.
                #
                # Masking cost real spots. A boundary a couple of pixels
                # off, or a small alignment residual between this hybe
                # and the one the cell was segmented in, clips emitters
                # sitting near the edge, and those are precisely the ones
                # a detector most needs examples of. A neighbouring
                # cell's spot intruding into the pad is not a problem
                # either: it is a real spot in a real crop, and a person
                # judging "is this a spot" does not need to know whose.
                #
                # The mask is still stored and still drawn, as an
                # outline, so a reviewer sees where the cell is.
                cands = candidates_for(block.astype(float), engine,
                                       max_fits, anchor, keep_top=keep_top)
                w.add(fov, hybe, channel, cid, block, mask, y0, x0, cands)
                n_crop += 1
                n_cand += len(cands)
    return path, n_crop, n_cand


def extract(storage_path, fovs, hybes, channel, out_dir, workers=None,
            chunk=DEFAULT_CHUNK, on_task=None, **kw):
    """Build a whole bundle. (fov, hybe, cell-chunk) through ONE pool.

    on_task(done, total, fov, hybe, path, n_crops, n_candidates) fires as
    each chunk lands, so progress is visible without waiting on the
    slowest anything.

    `workers` defaults to cpu_count-2 capped at 16. This workload is CPU
    bound -- 99.8% of a crop is the fit -- so unlike the alignment path,
    which is bandwidth bound and measured FASTER at 3 workers, here more
    readers do not contend and the count should track cores.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    os.makedirs(str(out_dir), exist_ok=True)
    tasks = plan(storage_path, fovs, hybes, channel, chunk=chunk)
    if workers is None:
        workers = max(1, min(16, (multiprocessing.cpu_count() or 4) - 2))
    workers = int(workers)
    results, done = [], 0
    if workers <= 1:
        for (fov, hybe, ids, tag) in tasks:
            r = extract_chunk(storage_path, fov, hybe, channel, ids,
                              out_dir, tag, **kw)
            done += 1
            if on_task:
                on_task(done, len(tasks), fov, hybe, *r)
            results.append((fov, hybe, tag) + r)
        return results
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(extract_chunk, storage_path, fov, hybe, channel,
                          ids, out_dir, tag, **kw): (fov, hybe, tag)
                for (fov, hybe, ids, tag) in tasks}
        for fut in as_completed(futs):
            fov, hybe, tag = futs[fut]
            r = fut.result()
            done += 1
            if on_task:
                on_task(done, len(tasks), fov, hybe, *r)
            results.append((fov, hybe, tag) + r)
    return sorted(results)


def summarize(out_dir):
    """What a finished bundle contains -- crops, candidates, bytes."""
    n_crop = n_cand = 0
    size = 0
    per_crop = []
    for p in B.shard_paths(out_dir):
        size += os.path.getsize(p)
        for row in B.read_index(p):
            n_crop += 1
            n_cand += int(row['n_candidates'])
            per_crop.append(int(row['n_candidates']))
    a = np.array(per_crop) if per_crop else np.array([0])
    return dict(crops=n_crop, candidates=n_cand, bytes=size,
                candidates_per_crop_median=float(np.median(a)),
                candidates_per_crop_p90=float(np.percentile(a, 90)),
                bytes_per_crop=float(size / max(1, n_crop)))
