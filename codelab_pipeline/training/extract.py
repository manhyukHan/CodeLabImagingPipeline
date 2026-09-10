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

GENEROSITY IS THE POINT, AND NOTHING CAPS IT. How many candidates a crop
has is a property of that crop, and the anchor threshold already decides
it from the data -- mode + k*sigma of the crop's own background. Both
per-crop ceilings are gone; see DEFAULT_MAX_FITS below for the measured
reason. The bundle is complete, its meta says so, and the review budget
lives in the viewer where it can be changed without destroying anything.

Candidates are ordered gate-pass, then fitted, then by p -- so a reviewer
meets the informative ones first and stopping early costs the least.

WHAT IT WILL NOT DO. It never writes to the store. It reads cells,
matrices and stacks; the only thing it creates is the bundle directory.
"""
import os
import time

import numpy as np

import numpy.linalg as la

from ..alignment.chain import align_cell
from ..io import analysis_store, paths
from ..localization import engine as E
from . import bundle as B

# The cell bbox plus this much on every side. Widened from 10 on
# 2026-09-07 together with dropping the mask from candidate generation:
# both exist so a spot sitting on or just past the segmented boundary --
# where a segmentation slip or a small inter-hybe alignment residual puts
# it -- is still in frame and still proposed.
DEFAULT_PAD = 14

# NO PER-CROP CANDIDATE LIMIT. Both of these are None on purpose.
#
# How many spots a crop has is a property of the crop, and the anchor
# threshold already answers it from the data: mode + k*sigma of that
# crop's own background. Putting a fixed ceiling on top substitutes an
# arbitrary number for a measured one, and it does not survive a change
# of store.
#
# It was 40, chosen on the MAZ store where a crop yields ~21 anchors and
# every gate-passing spot came from brightness rank <= 21. MP58/RNA is
# about four times denser -- median 90 anchors a crop, max 464 -- and
# that same 40 then:
#
#   bound on            1625 / 2266 crops   (71.7%)
#   left unfitted     173076 / 250481 anchors (69.1%)
#   NEVER PROPOSED        31 / 534 production-passing spots (5.8%),
#                         at brightness rank up to 215, p up to 0.785
#
# Raising it to 250 would only have moved the same mistake to the next
# store. A spot never proposed is a label that can never be collected,
# and the ones lost are systematically the dim ones a learned detector
# exists to find.
#
# The two caps existed for genuinely different reasons, and only one of
# them belongs to the data at all:
#
#   max_fits  was a COMPUTE budget. Measured, it is not needed: fitting
#             every anchor of this whole bundle is 250,481 fits at
#             0.127 s = 8.8 h serial, 17 min across 32 workers, against
#             6.1 min capped. Left as an optional safety valve for a
#             pathological crop, off by default.
#   keep_top  was a REVIEW budget, and a review budget has no business
#             being baked into the data. Truncating at write time
#             destroys candidates permanently; the same limit applied in
#             the viewer costs nothing and can be changed per session.
#             Keeping everything grows the bundle by 0.23% -- candidate
#             rows are ~19 bytes against 1.57 GiB of pixels.
#
# So: the bundle is COMPLETE, and spotcheck's --max-per-crop bounds what
# a person is shown.
DEFAULT_MAX_FITS = None
DEFAULT_KEEP_TOP = None


def cell_masks(storage_path, fov, hybe=None, modality=None, resolver=None):
    """[(cell_id, y_area, x_area, frame_shape), ...] IN `hybe`'S OWN FRAME.

    THE MASK HAS TO BE MOVED, and an earlier version of this function did
    not move it. Its docstring argued that "crops here are in the hybe's
    own native frame, deliberately, so this path has no use for matrix
    resolution" -- which inverts the actual requirement. The pixels do
    stay in the hybe's frame; the cell was segmented in a DIFFERENT hybe
    (cell.reference_hybe), so it is the mask that has to travel. "The
    crop is in the native frame" is the reason the transform is needed,
    not a reason to skip it.

    It is not cosmetic. The bbox is computed FROM the mask, so an
    untransformed mask also places the crop WINDOW wrong, and the pad is
    only 14 px: a drift larger than that clips the cell out of its own
    crop. MEASURED on MP58/RNA Hyb_109, all 2266 cells, |shift| median
    1.60 px, p90 2.38, max 4.63, systematic per FOV (fov003 dx -2.36,
    fov009 dy +2.15) -- comfortably inside the pad there, and no reason
    at all to assume the next experiment, FOV or hybe is as kind.

    resolver: a frames.FrameResolver for this FOV. Build it once per FOV
    and pass it in -- it reads matrices off disk. With none given, or no
    hybe, the raw stored area is returned unchanged, which is correct
    only when the crop comes from the cell's own reference hybe.
    """
    dicts, _ = analysis_store.read_cells(storage_path, fov)
    dicts = dicts or []
    cells = None
    if resolver is not None and hybe:
        from ..models.cell_container import CellContainer
        cells = CellContainer.load({int(fov): dicts}).data[int(fov)]
    out = []
    for c in dicts:
        ya, xa = c['area']
        if len(ya) == 0:
            continue
        ya = np.asarray(ya).astype(int)
        xa = np.asarray(xa).astype(int)
        if cells is not None:
            cell = cells.get(int(c['id']))
            if cell is not None:
                m = modality or c.get('reference_modality') or ''
                # transform(src, dst) maps src -> dst, so this is
                # hybe -> reference; the mask lives in the reference
                # frame, so it moves under the INVERSE. Same composition
                # localization.cell_crop uses for the production crop.
                H, _dz, _missing = resolver.transform(
                    (hybe, m),
                    (cell.reference_hybe, cell.reference_modality), cell)
                cy, cx = align_cell((cell.area[0], cell.area[1]), la.inv(H),
                                    cell.frame_shape)
                if len(cy):
                    ya, xa = cy.astype(int), cx.astype(int)
        out.append((int(c['id']), ya, xa, tuple(c['frame_shape'])))
    return out


def resolver_for_fov(storage_path, fov):
    """A FrameResolver for one FOV, or None if the store cannot make one.

    None is not an error: a store with no matrices yet still has cells,
    and returning the untransformed mask is the honest answer there --
    the same "an uncomputed layer is identity" rule the resolver itself
    follows. What must never happen is silently skipping the transform on
    a store that COULD have done it.
    """
    try:
        from ..analysis import resolvers
        return resolvers.resolver_for(storage_path, int(fov))
    except Exception:                                       # noqa: BLE001
        return None


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
    detailed = engine.localize_detailed(stack, seeds=anchors, n_max=None)
    out = [(s.y, s.x, s.z, s.p, 1, 1 if ok else 0, why)
           for (s, ok, why) in detailed]
    taken = [(s.y, s.x) for (s, _o, _w) in detailed]
    for (ay, ax, az) in anchors:
        if any((ay - ty) ** 2 + (ax - tx) ** 2 < 9 for (ty, tx) in taken):
            continue
        out.append((ay, ax, az, 0.0, 0, 0, 'no fit'))
    # GATE-PASS FIRST, then fitted, then by p -- ALWAYS, cap or no cap.
    #
    # This is the review order, and it matters even when nothing is
    # truncated: the reviewer meets the informative candidates first, so
    # stopping early costs the least. Sorting on (fit_ok, p) alone was
    # wrong because p and the gate disagree -- MEASURED, the gate-pass
    # candidates a wider search recovered had p from 0.023 to 0.785,
    # median 0.061, far below the junk outranking them.
    out.sort(key=lambda c: (-c[5], -c[4], -c[3]))
    if keep_top:
        # Only ever a caller's explicit ask. The default is None: a
        # truncation at write time is permanent, and the same limit in
        # the viewer costs nothing and is a per-session choice.
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
    modality = analysis_store.modality_of(storage_path)
    resolver = resolver_for_fov(storage_path, fov)
    cells = [c for c in cell_masks(storage_path, fov, hybe=hybe,
                                   modality=modality,
                                   resolver=resolver)
             if c[0] in wanted]
    # THE CHANNEL IS PART OF THE NAME, and leaving it out silently
    # destroyed data. Everything INSIDE a shard already separates
    # channels -- bundle.crop_key writes 'fov007|Hyb_101|ch635|cell13',
    # the index carries a 'channel' field, verdicts record it and
    # dataset.rows() rides it -- so two channels can live in one bundle
    # directory and every reader already handles them. The one thing
    # that could not was this filename: a second build into the same
    # --out produced the SAME name, and BundleWriter's os.replace
    # overwrote the first channel's shard entirely. Not a mix-up of
    # pixels -- the earlier channel simply vanished, and the bundle
    # still looked complete.
    path = os.path.join(str(out_dir),
                        f'fov{int(fov):03d}__{hybe}__ch{int(channel)}'
                        f'__{tag}.h5')
    info = dict(meta or {})
    info.update(fov=int(fov), hybe=str(hybe), channel=int(channel),
                pad=int(pad),
                # None means "nothing was truncated" -- a reader has to be
                # able to tell a complete candidate list from a capped one.
                max_fits=None if max_fits is None else int(max_fits),
                keep_top=None if keep_top is None else int(keep_top),
                complete=(max_fits is None and keep_top is None),
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
        # ids only -- the frame does not change which cells exist
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
    modality = analysis_store.modality_of(storage_path)
    resolver = resolver_for_fov(storage_path, fov)
    path = os.path.join(str(out_dir), f'fov{int(fov):03d}.h5')
    n_crop = n_cand = 0
    info = dict(meta or {})
    info.update(fov=int(fov), channel=int(channel), pad=int(pad),
                max_fits=None if max_fits is None else int(max_fits),
                keep_top=None if keep_top is None else int(keep_top),
                complete=(max_fits is None and keep_top is None),
                engine=engine_name, anchor=anchor,
                storage_path=str(storage_path))
    with B.BundleWriter(path, meta=info) as w:
        for hybe in hybes:
            sp = paths.stack_path(storage_path, fov, hybe)
            if not os.path.exists(sp):
                continue
            cells = cell_masks(storage_path, fov, hybe=hybe,
                               modality=modality, resolver=resolver)
            if not cells:
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


def default_workers():
    """cpu_count - 2, capped at 32.

    The cap was 16. The measurement this module already cites -- 250,481
    anchor fits in 17 min across 32 workers against 8.8 h serial -- was
    made at 32, and the docstring below says the count should track
    cores because the work is 99.8% fit. On a 64-core machine a cap of
    16 was leaving that measured speed on the table. 16 against 32 has
    NOT been compared head to head on one workload; --workers overrides.
    """
    import multiprocessing
    return max(1, min(32, (multiprocessing.cpu_count() or 4) - 2))


def extract(storage_path, fovs, hybes, channel, out_dir, workers=None,
            chunk=DEFAULT_CHUNK, on_task=None, **kw):
    """Build a whole bundle. (fov, hybe, cell-chunk) through ONE pool.

    on_task(done, total, fov, hybe, path, n_crops, n_candidates) fires as
    each chunk lands, so progress is visible without waiting on the
    slowest anything.

    `workers` defaults to default_workers(). This workload is CPU
    bound -- 99.8% of a crop is the fit -- so unlike the alignment path,
    which is bandwidth bound and measured FASTER at 3 workers, here more
    readers do not contend and the count should track cores.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    os.makedirs(str(out_dir), exist_ok=True)
    tasks = plan(storage_path, fovs, hybes, channel, chunk=chunk)
    if workers is None:
        workers = default_workers()
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
