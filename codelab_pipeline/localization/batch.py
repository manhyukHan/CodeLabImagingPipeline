"""
Batch spot localization with the learned engine: every cell of every
(FOV, hybe) asked for, headless, straight into the spot store.

The per-cell geometry is the interactive panel's own (MainWindow.
_v3_one_cell / _v3_spot, localization.cell_crop's resolver branch):
the cell's mask is projected into the hybe's raw frame through the
store-built FrameResolver, a padded rectangle around it is the engine's
input with the mask as its label, and each spot lands in both frames --
raw (the hybe's own) and adj (the pipeline's one shared frame, through
resolver.to_shared, plus the cell's own z shift). What differs from
the panel is only the I/O shape: one (FOV, hybe) reads its stack ONCE
and cuts every cell's rectangle from memory (a per-cell hyperslab read
of a ~200 MB stack, 300 times per hybe, is what made the interactive
path unusable as a batch), and the pool's workers never touch the
store -- every write happens in the parent, in one process, because
the FOV manifest's read-modify-write (spot uids, slice counts) is
guarded by a thread lock, not a cross-process one.

Append mode: a (FOV, hybe, channel) slice already in the store is
skipped unless overwrite is asked for, so an interrupted run resumes
where it stopped; each slice is written atomically (analysis_store).

The p_exist gate is applied HERE, before the store, at the threshold
the caller names (the classifier's own 0.5 by default): what the store
holds is what the count paths downstream see, and p_min per slice
records the gate for them (analysis/cellcycle.count_table).
"""
import os
import time

import h5py
import numpy as np
import numpy.linalg as la

from codelab_pipeline import parallel
from codelab_pipeline.alignment import chain as alignment
from codelab_pipeline.analysis import resolvers as R
from codelab_pipeline.io import analysis_store, paths
from codelab_pipeline.models.cell import ACell
from codelab_pipeline.models.spot import ASpot, Z_ACCEPTED as SPOT_Z_ACCEPTED

DEFAULT_PAD = 10
DEFAULT_MIN_P_EXIST = 0.5


def make_engine_state(model_dir):
    """pmap initializer: the engine, with its lazy parts loaded once per
    worker rather than on the first cell."""
    from codelab_pipeline.localization.engine import make_engine
    eng = make_engine('v3-psfmatcher', model_dir=model_dir)
    eng.classifier
    eng.refines
    return eng


def _spot_dict(s, rymin, rxmin, H, Hz, fov, hybe, channel, modality, cell):
    """One LocalizedSpot -> ASpot.save() dict, in both frames -- the
    panel's _v3_spot, with the cell's celltype carried."""
    raw_y, raw_x = float(s.y) + rymin, float(s.x) + rxmin
    if H is not None:
        cy, cx, _ = H @ np.array([raw_y, raw_x, 1.0])
    else:
        cy, cx = raw_y, raw_x
    spot = ASpot()
    spot.modality = modality
    spot.set_metadata(
        fov=int(fov), hybe=hybe, channel=int(channel), cell=int(cell.id),
        celltype=str(cell.celltype or ''),
        adj_coordinate=(float(cy), float(cx), float(s.z) + float(Hz)),
        raw_coordinate=(raw_y, raw_x, float(s.z)),
        size=0.0,
        brightness=(float(s.amplitude) if np.isfinite(s.amplitude) else 0.0),
        z_status=SPOT_Z_ACCEPTED,
        p_exist=float(s.p_exist))
    return spot.save()


def detect_fov_hybe(item, engine):
    """One (FOV, hybe): every cell localized, gated, returned as spot
    dicts with uid 0 (the parent allocates). Module-level for pmap.

    item: (storage_path, fov, hybe, channel, pad, min_p_exist).
    Returns {'fov', 'hybe', 'channel', 'spots': [dict], 'n_cells',
             'n_candidates', 'n_kept', 'missing': [layer names],
             't_read', 't_total'}.
    """
    storage_path, fov, hybe, channel, pad, min_p = item
    t0 = time.time()
    modality = analysis_store.modality_of(storage_path)
    resolver = R.resolver_for(storage_path, fov)
    dicts, _ = analysis_store.read_cells(storage_path, fov)
    cells = []
    for d in (dicts or []):
        c = ACell()
        c.set_metadata(**d)
        cells.append(c)
    with h5py.File(paths.stack_path(storage_path, fov, hybe), 'r') as f:
        stack = f[f'/stack/ch{int(channel)}'][...]
    t_read = time.time() - t0
    out, missing_any = [], set()
    n_cand = 0
    for cell in cells:
        # the resolver branch of localization.cell_crop, verbatim in
        # geometry: the mask into the hybe's raw frame, then a padded
        # rectangle cut from the in-memory stack
        H_cellref, _dz, missing = resolver.transform(
            (hybe, modality), (cell.reference_hybe, cell.reference_modality), cell)
        missing_any.update(missing or ())
        y_lit, x_lit = cell.area
        cy, cx = alignment.align_cell((y_lit, x_lit), la.inv(H_cellref), cell.frame_shape)
        if len(cy) == 0:
            continue
        ya, xa = cy.astype(int), cx.astype(int)
        height, width = cell.frame_shape
        rymin, rymax = max(0, ya.min() - pad), min(height, ya.max() + pad + 1)
        rxmin, rxmax = max(0, xa.min() - pad), min(width, xa.max() + pad + 1)
        sub = np.asarray(stack[rymin:rymax, rxmin:rxmax, :], float)
        labels = np.zeros(sub.shape[:2], int)
        inside = (ya >= rymin) & (ya < rymax) & (xa >= rxmin) & (xa < rxmax)
        labels[ya[inside] - rymin, xa[inside] - rxmin] = 1
        spots = engine.localize(sub, labels=labels, n_max=None)
        n_cand += len(spots)
        H = resolver.to_shared(hybe, modality, cell)
        Hz = alignment.entry_dz(cell.matrices.get((hybe, modality)))
        for s in spots:
            if np.isfinite(s.p_exist) and s.p_exist < min_p:
                continue
            out.append(_spot_dict(s, rymin, rxmin, H, Hz, fov, hybe, channel, modality, cell))
    return {'fov': int(fov), 'hybe': hybe, 'channel': int(channel), 'spots': out,
            'n_cells': len(cells), 'n_candidates': n_cand, 'n_kept': len(out),
            'missing': sorted(missing_any), 't_read': t_read, 't_total': time.time() - t0}


def write_result(storage_path, res):
    """The parent's half: uids from the FOV manifest, then the slice,
    atomically. Full replace of that (modality, hybe, channel) slice."""
    modality = analysis_store.modality_of(storage_path)
    spots = res['spots']
    if spots:
        uids = analysis_store.allocate_spot_uids(storage_path, res['fov'], len(spots))
        for d, uid in zip(spots, uids):
            d['uid'] = int(uid)
    analysis_store.write_spot_dicts(storage_path, res['fov'], modality, res['hybe'],
                                    res['channel'], spots)


def pending(storage_path, fovs, hybes, channel, overwrite=False):
    """[(fov, hybe)] still to run, FOV-major (one FOV's hybes together,
    so the file cache serves the second hybe of a FOV warm)."""
    modality = analysis_store.modality_of(storage_path)
    items = []
    for fov in fovs:
        have = set() if overwrite else set(analysis_store.spot_slices(storage_path, fov))
        for hybe in hybes:
            if (modality, hybe, int(channel)) in have:
                continue
            items.append((int(fov), hybe))
    return items


def run(storage_path, fovs, hybes, channel, model_dir, min_p_exist=DEFAULT_MIN_P_EXIST,
        pad=DEFAULT_PAD, jobs=None, overwrite=False, log=print, on_done=None):
    """Localize every pending (FOV, hybe) at `channel` and write it.

    Returns {'written': n, 'failed': [((fov, hybe), message)],
             'skipped': n_already_in_store, 'minutes': elapsed}.
    """
    t0 = time.time()
    todo = pending(storage_path, fovs, hybes, channel, overwrite=overwrite)
    n_all = len(fovs) * len(hybes)
    log(f'{len(todo)} of {n_all} (FOV, hybe) slices to localize, '
        f'{n_all - len(todo)} already in the store; model {os.path.basename(model_dir)}, '
        f'p_exist >= {min_p_exist}, pad {pad}')
    items = [(storage_path, f, h, int(channel), int(pad), float(min_p_exist)) for f, h in todo]
    failed, written = [], []

    def _done(n_done, n_total, index, result):
        if isinstance(result, parallel.Failure):
            failed.append((todo[index], str(result)))
            log(f'[{n_done}/{n_total}] FAILED FOV{todo[index][0]:03d} {todo[index][1]}: {result}')
        else:
            try:
                write_result(storage_path, result)
                written.append(todo[index])
            except Exception as exc:                        # noqa: BLE001
                failed.append((todo[index], f'write: {type(exc).__name__}: {exc}'))
                log(f'[{n_done}/{n_total}] WRITE FAILED FOV{result["fov"]:03d} {result["hybe"]}: {exc}')
                return
            el = time.time() - t0
            if n_done % 10 == 0 or n_done == n_total:
                log(f'[{n_done}/{n_total}] FOV{result["fov"]:03d} {result["hybe"]}: '
                    f'{result["n_kept"]}/{result["n_candidates"]} spots kept over '
                    f'{result["n_cells"]} cells (read {result["t_read"]:.0f}s, '
                    f'{result["t_total"]:.0f}s) | {el / 60:.0f} min, '
                    f'~{(n_total - n_done) * el / max(n_done, 1) / 60:.0f} min left')
        if on_done is not None:
            on_done(n_done, n_total, index, result)

    if items:
        parallel.pmap(detect_fov_hybe, items, kind='cpu', jobs=jobs,
                      initializer=make_engine_state, initargs=(model_dir,), on_done=_done)
    return {'written': len(written), 'failed': failed,
            'skipped': n_all - len(todo), 'minutes': (time.time() - t0) / 60}
