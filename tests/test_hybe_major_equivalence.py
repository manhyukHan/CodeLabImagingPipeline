"""
Hybe-major fitting produces byte-identical results to the serial loop.

WHY THIS EXISTS. Cell alignment was restructured from cell-major (a pool
child owns a cell and walks its 135 hybes) to hybe-major (a task owns a
hybe and walks the FOV's cells). The reason is memory and reads: cell-major
makes every worker read every hybe's MIP, so six children hold six copies
of all 135 -- ~1.6 GB retained, which is what drained this machine's free
page list. Hybe-major reads each MIP once and holds only what is in flight.

The restructure is only allowed to change WHEN work happens, never WHAT
comes out. Three independent reviewers agreed the transposition is
result-preserving -- _cell_hybe_task reads a fixed ctx and returns a value,
with no loop-carried state but a pure memo -- and all three also pointed
out that NO checked-in test would notice if that were wrong:
compute_cell_alignment appeared in tests only as a mock. This is that test.

WHAT IT PINS. Every fitted number and every provenance string, compared
element-by-element across three paths that must agree exactly:

    serial      compute_cell_alignment, one cell at a time (the reference)
    hybe-major  prepare -> run_hybe_tasks -> commit, in this process
    pooled      the same, across a real spawn pool

It also pins the properties the reviewers flagged as load-bearing and easy
to lose in a restructure: cell.matrices insertion order (it follows
hybe_records, and ACell.save comprehends over it), the whole-pass stale-key
clear, the identity entry for the reference hybe, and CellOffFrameError's
identity residuals for a cell that does not overlap the reference frame.

It builds its own store, so it runs on this machine -- the real one it was
measured against is 400 GB on a NAS, and the repo's data/ fixtures are
absent here (see CLAUDE.md).

Run:  QT_QPA_PLATFORM=offscreen python tests/test_hybe_major_equivalence.py
"""
import copy
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                       # noqa: E402

from codelab_pipeline.alignment import chain              # noqa: E402
from codelab_pipeline.alignment.frames import FrameMatrices   # noqa: E402
from codelab_pipeline.io import preprocess, analysis_store, paths  # noqa: E402
from codelab_pipeline.models.cell import ACell            # noqa: E402

PASS, FAIL = [], []
MODALITY = 'DNA'
FOV = 1
FRAME = (256, 256)
DEPTH = 24
FIDUCIAL = 635
READOUT = 555
HYBES = [f'Hyb_{i:03d}' for i in range(1, 7)]
REFERENCE = HYBES[0]


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}' + (f'   [{detail}]' if detail else ''))


def build_store(root):
    """A v2 modality store with real stacks and MIPs, one FOV, six hybes.

    Each hybe is the same scene SHIFTED by a known amount, so the fit has
    something real to find rather than noise -- a run that silently
    produced identity everywhere would otherwise still "agree" across the
    three paths and prove nothing.
    """
    storage_path = os.path.join(root, MODALITY)
    os.makedirs(os.path.join(storage_path, 'stacks', paths.fov_dir_name(FOV)),
                exist_ok=True)
    rng = np.random.default_rng(11)
    h, w = FRAME
    base = np.zeros((h, w), dtype=np.float64)
    for cy, cx in rng.integers(20, min(h, w) - 20, size=(90, 2)):
        yy, xx = np.ogrid[:h, :w]
        base += 900.0 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / 12.0))
    shifts = {}
    for i, hybe in enumerate(HYBES):
        dy, dx = (0, 0) if i == 0 else (int(rng.integers(-4, 5)), int(rng.integers(-4, 5)))
        shifts[hybe] = (dy, dx)
        frame = np.roll(np.roll(base, dy, axis=0), dx, axis=1)
        stack = np.zeros((h, w, DEPTH), dtype=np.uint16)
        centre = 8 + (i % 5)          # a real, hybe-dependent z position
        for z in range(DEPTH):
            weight = np.exp(-((z - centre) ** 2) / 6.0)
            stack[:, :, z] = np.clip(frame * weight + rng.normal(40, 6, (h, w)),
                                     0, 65535).astype(np.uint16)
        preprocess.publish_stack(
            paths.stack_path(storage_path, FOV, hybe),
            {'depth': DEPTH}, {READOUT: stack, FIDUCIAL: stack},
            storage_path, FOV, hybe, FIDUCIAL)
    return storage_path, shifts


def make_cells():
    """Four cells with real, distinct masks -- plus one wholly off-frame.

    The off-frame cell is not decoration: CellOffFrameError writes identity
    residuals for a whole pass and then raises, and that write has to
    happen in the same place under both dispatches.
    """
    cells = []
    for i, (cy, cx) in enumerate([(60, 60), (60, 180), (180, 70), (180, 190)]):
        yy, xx = np.mgrid[cy - 22:cy + 22, cx - 22:cx + 22]
        keep = (yy - cy) ** 2 + (xx - cx) ** 2 <= 22 ** 2
        cell = ACell()
        cell.id = i + 1
        cell.fov = FOV
        cell.reference_hybe = REFERENCE
        cell.reference_modality = MODALITY
        cell.frame_shape = FRAME
        cell.area = (yy[keep].astype(np.int64), xx[keep].astype(np.int64))
        cells.append(cell)
    return cells


def make_matrices():
    fm = FrameMatrices(modality=MODALITY)
    rng = np.random.default_rng(5)
    for hybe in HYBES:
        H = np.eye(3)
        if hybe != REFERENCE:
            H[0, 2] = float(rng.integers(-3, 4))
            H[1, 2] = float(rng.integers(-3, 4))
        fm[(hybe, MODALITY)] = H
    return fm


RECORDS = [{'folder': h, 'channels': [READOUT, FIDUCIAL],
            'fiducial_channel': FIDUCIAL} for h in HYBES]


def snapshot(cells):
    """Everything a run is allowed to write, in a comparable form."""
    out = {}
    for cell in cells:
        out[cell.id] = {
            'matrices': [(k, {kk: np.asarray(vv).copy() if not isinstance(vv, bool) else vv
                              for kk, vv in v.items()})
                         for k, v in cell.matrices.items()],
            'anchors': {k: np.asarray(v).copy() for k, v in cell.matrix_anchors.items()},
            'provenance': [(k, str(v['reference_sequence']), np.asarray(v['steps']).copy())
                           for k, v in cell.matrix_provenance.items()],
        }
    return out


def same(a, b, label):
    """Element-by-element, including key ORDER -- not just equal contents."""
    problems = []
    for cell_id in sorted(set(a) | set(b)):
        ea, eb = a.get(cell_id), b.get(cell_id)
        if ea is None or eb is None:
            problems.append(f'cell {cell_id} missing from one side')
            continue
        ka = [k for k, _ in ea['matrices']]
        kb = [k for k, _ in eb['matrices']]
        if ka != kb:
            problems.append(f'cell {cell_id} matrices key ORDER differs')
            continue
        for (k, va), (_k, vb) in zip(ea['matrices'], eb['matrices']):
            for field in set(va) | set(vb):
                x, y = va.get(field), vb.get(field)
                if not np.array_equal(np.asarray(x), np.asarray(y)):
                    problems.append(f'cell {cell_id} {k} {field}: {x!r} vs {y!r}')
        if [k for k, _s, _st in ea['provenance']] != [k for k, _s, _st in eb['provenance']]:
            problems.append(f'cell {cell_id} provenance key order differs')
        for (k, sa, sta), (_k, sb, stb) in zip(ea['provenance'], eb['provenance']):
            if sa != sb:
                problems.append(f'cell {cell_id} {k} provenance text differs')
            if not np.array_equal(sta, stb):
                problems.append(f'cell {cell_id} {k} provenance steps differ')
        for k in set(ea['anchors']) | set(eb['anchors']):
            if not np.array_equal(ea['anchors'].get(k), eb['anchors'].get(k)):
                problems.append(f'cell {cell_id} anchor {k} differs')
    check(label, not problems, '; '.join(problems[:2]))
    return not problems


def run_serial(cells, storage_path, fm):
    for cell in cells:
        chain.compute_cell_alignment(
            cell, storage_path, FOV, RECORDS, fm, reference_hybe=REFERENCE,
            channel_type='readout', pad=10, modality=MODALITY,
            cell_reference_hybe_matrix=fm[(REFERENCE, MODALITY)])


def run_hybe_major(cells, storage_path, fm, workers=0):
    passes_by_cell = {c.id: [{'storage_path': storage_path, 'hybe_records': RECORDS,
                              'fov_matrices': fm, 'reference_hybe': REFERENCE,
                              'modality': MODALITY,
                              'cellref_matrix': fm[(REFERENCE, MODALITY)]}]
                      for c in cells}
    work, plans, skipped = chain.prepare_cell_passes(
        cells, FOV, passes_by_cell, 'readout', 10, None)
    pool = (chain.make_cell_hybe_pool(work, workers, chain._init_cell_align_worker)
            if workers > 1 else None)
    try:
        if pool is None:
            chain.fill_reference_zx(work)
        results = chain.run_hybe_tasks(work, executor=pool)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    for key, (cell, plan) in plans.items():
        chain.commit_cell_alignment(cell, plan, results[key])
    return skipped, pool is not None


def main():
    root = tempfile.mkdtemp()
    try:
        print('\n-- building a real store (6 hybes, 256x256x24) --')
        storage_path, shifts = build_store(root)
        ok = all(os.path.exists(paths.stack_path(storage_path, FOV, h)) and
                 os.path.exists(paths.mip_path(storage_path, FOV, h)) for h in HYBES)
        check('every hybe has a stack and a MIP', ok)
        fm = make_matrices()

        print('\n-- the serial path fits something real --')
        base_cells = make_cells()
        run_serial(base_cells, storage_path, fm)
        reference = snapshot(base_cells)
        n_entries = sum(len(v['matrices']) for v in reference.values())
        check('every cell got an entry for every hybe',
              n_entries == len(base_cells) * len(HYBES), str(n_entries))
        dz = [float(dict(v['matrices'])[(h, MODALITY)]['dz'])
              for v in reference.values() for h in HYBES if h != REFERENCE]
        check('the Z leg produced non-zero corrections (so the fit ran)',
              any(abs(z) > 0 for z in dz), f'max |dz| {max(abs(z) for z in dz):.2f}')
        yx = [dict(v['matrices'])[(h, MODALITY)]['yx']
              for v in reference.values() for h in HYBES if h != REFERENCE]
        check('the YX leg produced non-identity residuals',
              any(not np.array_equal(m, np.eye(3)) for m in yx))

        print('\n-- hybe-major, in this process --')
        cells2 = make_cells()
        analysis_store.clear_caches() if hasattr(analysis_store, 'clear_caches') else None
        skipped2, _ = run_hybe_major(cells2, storage_path, fm, workers=0)
        same(reference, snapshot(cells2), 'byte-identical to the serial loop')
        check('and nothing was skipped', not skipped2, str(skipped2))

        print('\n-- hybe-major, across a real spawn pool --')
        cells3 = make_cells()
        _skipped3, pooled = run_hybe_major(cells3, storage_path, fm, workers=4)
        check('a pool was actually used (not a silent degrade)', pooled)
        same(reference, snapshot(cells3), 'byte-identical across the pool')

        print('\n-- the dispatch plan is per hybe --')
        cells4 = make_cells()
        passes = {c.id: [{'storage_path': storage_path, 'hybe_records': RECORDS,
                          'fov_matrices': fm, 'reference_hybe': REFERENCE,
                          'modality': MODALITY,
                          'cellref_matrix': fm[(REFERENCE, MODALITY)]}] for c in cells4}
        work, _plans, _sk = chain.prepare_cell_passes(cells4, FOV, passes, 'readout', 10, None)
        groups = chain.hybe_groups(work)
        check('one group per TARGET hybe, reference excluded',
              len(groups) == len(HYBES) - 1, str(len(groups)))
        check('and each group carries every cell that needs it',
              all(len(keys) == len(cells4) for _f, _r, keys in groups),
              str([len(k) for _f, _r, k in groups]))
        check('so a hybe\'s stack is opened once, not once per cell',
              sum(len(k) for _f, _r, k in groups) == (len(HYBES) - 1) * len(cells4))

        print('\n-- the reference hybe keeps its identity entry --')
        for cell in cells3:
            entry = cell.matrices[(REFERENCE, MODALITY)]
            check(f'cell {cell.id}: reference is identity with no provenance',
                  np.array_equal(entry['yx'], np.eye(3)) and entry['dz'] == 0.0
                  and (REFERENCE, MODALITY) not in cell.matrix_provenance)

        print('\n-- a stale key from a previous run is cleared, not kept --')
        cells5 = make_cells()
        ghost = ('Hyb_099', MODALITY)
        for cell in cells5:
            cell.matrices[(HYBES[1], MODALITY)] = {'yx': np.full((3, 3), 7.0),
                                                   'dz': 99.0, 'yx_is_residual': True}
            cell.matrices[ghost] = {'yx': np.eye(3), 'dz': 1.0, 'yx_is_residual': True}
        run_hybe_major(cells5, storage_path, fm, workers=0)
        rewritten = all(not np.array_equal(c.matrices[(HYBES[1], MODALITY)]['yx'],
                                           np.full((3, 3), 7.0)) for c in cells5)
        check('this pass\'s stale entry is overwritten by what this run fitted',
              rewritten)
        check('a key OUTSIDE this pass is left alone',
              all(ghost in c.matrices for c in cells5))

        print('\n-- an off-frame cell writes identity residuals and is skipped --')
        cells6 = make_cells()
        far = ACell()
        far.id = 99
        far.fov = FOV
        far.reference_hybe = REFERENCE
        far.reference_modality = MODALITY
        far.frame_shape = FRAME
        # A mask whose inverse-warp lands wholly outside the frame.
        yy, xx = np.mgrid[100000:100010, 100000:100010]
        far.area = (yy.ravel().astype(np.int64), xx.ravel().astype(np.int64))
        skipped6, _ = run_hybe_major(cells6 + [far], FOV and storage_path, fm, workers=0)
        check('the off-frame cell is reported as skipped',
              any(cid == 99 for cid, _why in skipped6), str(skipped6))
        check('and it still carries an entry for every hybe (identity, not absent)',
              len(far.matrices) == len(HYBES), str(len(far.matrices)))
        check('the other cells were unaffected by it',
              same(reference, snapshot(cells6), '  their results still match'))

        print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
        if FAIL:
            for f in FAIL:
                print('  FAILED:', f)
            return 1
        print('ALL GOOD')
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
