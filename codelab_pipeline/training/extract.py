"""
Build review bundles from a store: cell crops + generous anchor-fit.

THE SHAPE OF THE WORK. One task is one (fov, hybe) pair, and it reads
that hybe's stack file ONCE and crops every cell out of the open handle.
That is not an optimisation detail, it is the whole cost model: a stack
file is ~266 MB and opening it is what the 673 ms per-crop measurement
was mostly measuring, so a task that reopened per cell would pay it 73
times per FOV instead of once. Tasks are dispatched FOV-major through ONE
shared pool -- never a pool per modality, never a pass per hybe -- for
the same reason ingestion is.

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

DEFAULT_PAD = 10
DEFAULT_MAX_FITS = 40
DEFAULT_KEEP_TOP = 12


def cell_masks(storage_path, fov):
    """[(cell_id, y_area, x_area, frame_shape), ...] for one FOV.

    Reads the persisted cell dicts rather than building ACell objects:
    the extractor needs a mask and a bbox, and nothing else a cell knows.
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


def candidates_for(stack_masked, engine, max_fits, anchor, keep_top=None):
    """[(y, x, z, p, fit_ok, gate_pass, reason), ...] for one crop.

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
    anchors = E.anchor_candidates(stack_masked, n_max=max_fits, **anchor)
    if not anchors:
        return []
    detailed = engine.localize_detailed(stack_masked, seeds=anchors,
                                        n_max=max_fits)
    out = [(s.y, s.x, s.z, s.p, 1, 1 if ok else 0, why)
           for (s, ok, why) in detailed]
    taken = [(s.y, s.x) for (s, _o, _w) in detailed]
    for (ay, ax, az) in anchors:
        if any((ay - ty) ** 2 + (ax - tx) ** 2 < 9 for (ty, tx) in taken):
            continue
        out.append((ay, ax, az, 0.0, 0, 0, 'no fit'))
    if keep_top:
        # fitted candidates first (p > 0), then unfitted anchors, each
        # group already best-first -- an unfitted anchor is worth showing
        # but never at the expense of a fit that succeeded.
        out.sort(key=lambda c: (-c[4], -c[3]))
        out = out[:int(keep_top)]
    return out


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
                st = np.where(mask[:, :, None].astype(bool),
                              block.astype(float), np.nan)
                cands = candidates_for(st, engine, max_fits, anchor,
                                       keep_top=keep_top)
                w.add(fov, hybe, channel, cid, block, mask, y0, x0, cands)
                n_crop += 1
                n_cand += len(cands)
    return path, n_crop, n_cand


def extract(storage_path, fovs, hybes, channel, out_dir, workers=None,
            on_fov=None, **kw):
    """Build a whole bundle. FOV-major through one pool.

    on_fov(fov, path, n_crops, n_candidates) is called as each shard
    lands, so a GUI or a CLI can report progress without waiting for the
    slowest FOV.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    os.makedirs(str(out_dir), exist_ok=True)
    fovs = [int(f) for f in fovs]
    if workers is None or int(workers) <= 1:
        results = []
        for f in fovs:
            r = extract_fov(storage_path, f, hybes, channel, out_dir, **kw)
            if on_fov:
                on_fov(f, *r)
            results.append((f,) + r)
        return results
    results = []
    with ProcessPoolExecutor(max_workers=int(workers)) as ex:
        futs = {ex.submit(extract_fov, storage_path, f, hybes, channel,
                          out_dir, **kw): f for f in fovs}
        for fut in as_completed(futs):
            f = futs[fut]
            r = fut.result()
            if on_fov:
                on_fov(f, *r)
            results.append((f,) + r)
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
