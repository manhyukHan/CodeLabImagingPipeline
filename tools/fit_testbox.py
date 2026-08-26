"""
A frozen, real-data bench for 3D localization fitting.

WHY IT EXISTS
-------------
The Gaussian fit engine is a direct port of ChrTracer3's FitPsf3D and must
stay that way -- it is the reference implementation. Changing how spots
are localized therefore means writing a SECOND engine and PROVING it is
better, on real NAS-shaped data, rather than asserting it.

THE GROUND TRUTH
----------------
This experiment images 10 genomic loci MORE THAN ONCE, in independent
hybridization rounds:

    readout  32: Hyb_032 (H), Toe_032 (T), Rep_032 (R)
    readout  23: Hyb_023 (H), Rep_023 (R)
    ... 10 loci, 12 pairs in all

Two rounds probing the SAME locus in the SAME allele should localize to
the SAME physical point, so the distance between them measures
localization error directly -- no simulation, no hand-labelling. That is
why the traced set is H/T/R. Lower median pair distance = better.

The metric cannot be gamed by accepting more fits: a pair is only scored
when BOTH rounds produced one, and the fit counts are reported next to
it, so an engine that buys precision by discarding hard spots shows up as
such.

TWO STAGES
----------
harvest  Drives a REAL session (MainWindow + the production
         build_chromatin_trace_allele) over N randomly chosen alleles and
         freezes every crop, plus the PREDICTED z for each hybe, to one
         .npz. This is the slow part (NAS reads of 73 hybes x 2 channels
         per allele) and is done once.

score    Replays frozen crops with zero NAS access, so a change is
         measured in seconds against byte-identical input.

The predicted z is stored because it is the pipeline's own answer to
"where should this allele be, in depth, in this hybe's raw stack" --
allele z (shared) minus that hybe's cell_z_offset. The current engine
does not use it: it seeds z at the global argmax over the full ~120-plane
column and then bounds the fit to seed +/- peak_bound (2 planes), so the
seed alone decides depth. `score --z-source` measures what changes when
the prediction is used instead.

Usage:
    python tools/fit_testbox.py harvest --out bench.npz [--alleles 12]
    python tools/fit_testbox.py score bench.npz
"""
import argparse
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# -- the testbox, per explicit decision ----------------------------------
#
# Its OWN config, deliberately not the working configs/2026-08-25-SH.xml
# for the same project: the bench is pinned to Hyb_016 as both the allele
# source and the drift baseline, while the working config traces from
# Hyb_031 and is edited as the experiment proceeds. A benchmark whose
# inputs move is not a benchmark.
CONFIG = 'configs/2025-11-30-MP58-testbox.xml'
MODALITY = 'DNA'
FOVS = (1, 2, 4, 5)
ANCHOR_HYBE = 'Hyb_016'      # allele candidates: its 555 spots (~440 over 4 FOVs)
ANCHOR_CHANNEL = 555
REFERENCE_HYBE = 'Hyb_016'   # drift baseline
DATATYPES = ('H', 'T', 'R')
SEED = 20260826


# -- stage 1: harvest ----------------------------------------------------

DEFAULT_FIGURES_DIR = os.path.join('notes', 'chromatin_tracing_optimization')


def anchor_label(fov, spot_dict):
    """
    A name that identifies the SOURCE SPOT, not the allele's position in
    a selection.

    An allele's id is just its index in whatever was selected, so
    "allele 1" means nothing once the selection changes and cannot be
    traced back to anything on disk. The anchor can: uid is the spot's
    unique key within its FOV, and the cell / hybe / channel /
    coordinate make it findable by eye in the Spot panel. Filename-safe,
    so the same string works as figure title and file name.
    """
    y, x, z = spot_dict['adj_coordinate']
    cell = int(spot_dict['cell'])
    # -1 is the real "no owning cell" value; printing it as cell-01 reads
    # like a cell id, which it is not
    cell_bit = 'cellNONE' if cell == -1 else f'cell{cell:03d}'
    return (f'FOV{fov:03d}_uid{int(spot_dict.get("uid", 0)):05d}_{cell_bit}'
            f'_{spot_dict["hybe"]}-ch{int(spot_dict["channel"])}'
            f'_y{y:.0f}-x{x:.0f}-z{z:.0f}')


def _save_allele_figures(mw, allele, debug, out_dir, label):
    """
    The four pop-ups View Crop opens, rendered straight to PNG.

    Driven through the REAL displayers (show_overlay_grid /
    show_fit_status_grid) and saved with canvas.figure.savefig, which is
    what the Save Figure... button does once the dialog returns -- so a
    figure on disk is the same picture the app shows, not a second
    plotting path that could drift from it.
    """
    import os as _os
    _os.makedirs(out_dir, exist_ok=True)
    chp = mw.ui.ChromatinTracingPanel
    params = chp.params()
    written = []

    entries, total = mw._build_fiducial_overlay_entries(allele, REFERENCE_HYBE, debug)
    if entries:
        d = mw.chromatin_fiducial_overlay_displayer
        d.show_overlay_grid(entries, allele_label=label, params=params)
        p = _os.path.join(out_dir, f'{label}_fiducial_overlay_one_vs_one.png')
        d.canvas.figure.savefig(p, dpi=140, bbox_inches='tight')
        written.append(p)
    if total is not None:
        d = mw.chromatin_fiducial_total_overlay_displayer
        d.show_total_overlay(total, allele_label=label, params=params)
        p = _os.path.join(out_dir, f'{label}_fiducial_overlay_one_vs_all.png')
        d.canvas.figure.savefig(p, dpi=140, bbox_inches='tight')
        written.append(p)

    for kind, disp in (('fiducial', mw.chromatin_fiducial_grid_displayer),
                       ('readout', mw.chromatin_readout_grid_displayer)):
        results = []
        for hybe in sorted(debug or {}):
            dbg = debug[hybe]
            cube = dbg.get(f'{kind}_cubic')
            if cube is None:
                continue
            c = dbg.get(f'{kind}_centroid' if kind == 'fiducial' else 'readout_centroids')
            if kind == 'fiducial':
                c = [c] if c is not None else None
            results.append((cube, c, hybe))
        if not results:
            continue
        disp.show_fit_status_grid(results, allele_label=label, params=params)
        p = _os.path.join(out_dir, f'{label}_{kind}_grid.png')
        disp.canvas.figure.savefig(p, dpi=140, bbox_inches='tight')
        written.append(p)
    return written


def harvest(out_path, n_alleles, figures_dir=None):
    """Freeze real crops + predicted z, through the real session path."""
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from unittest import mock
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    for m in ('critical', 'warning', 'information', 'question'):
        mock.patch.object(QtWidgets.QMessageBox, m,
                          return_value=QtWidgets.QMessageBox.Yes).start()

    from windows.main_window import MainWindow
    from codelab_pipeline.io import analysis_store as V
    from codelab_pipeline.localization import localization as L
    from codelab_pipeline.models.allele import AnAllele

    mw = MainWindow(CONFIG)
    sp = mw._storage_path_for_modality(MODALITY)
    records = mw._hybe_records_for_storage_path(sp)
    by_folder = {r['folder']: r for r in records}
    hybes = [r['folder'] for r in records
             if str(r['datatype']).upper() in DATATYPES]
    fid_ch = {h: by_folder[h]['fiducial_channel'] for h in hybes}
    read_ch = {}
    for h in hybes:
        others = [c for c in by_folder[h]['channels']
                  if c != by_folder[h]['fiducial_channel']]
        read_ch[h] = others[0] if others else by_folder[h]['fiducial_channel']

    print(f'store      : {sp}')
    print(f'hybes      : {len(hybes)} of datatype {"/".join(DATATYPES)} '
          f'(reference {REFERENCE_HYBE})')
    print(f'replicates : {len(replicate_pairs(records))} same-locus pairs')

    # STRATIFIED by FOV, not a flat random draw: a flat sample of 12 from
    # 439 legitimately put both of the first probe's alleles in one FOV,
    # and a bench that silently covers one FOV cannot show a per-FOV
    # effect (segmentation, drift and focus all vary FOV to FOV).
    rng = random.Random(SEED)
    by_fov, chosen = {}, []
    for fov in FOVS:
        by_fov[fov] = V.read_spots(sp, fov, MODALITY, ANCHOR_HYBE, ANCHOR_CHANNEL)
    total = sum(len(v) for v in by_fov.values())
    per_fov = max(1, n_alleles // len(FOVS))
    for fov in FOVS:
        pool = by_fov[fov]
        for d in rng.sample(pool, min(per_fov, len(pool))):
            chosen.append((fov, d))
    print(f'candidates : {total} {ANCHOR_HYBE}/ch{ANCHOR_CHANNEL} spots '
          f'over FOV {list(FOVS)} '
          f'({", ".join(f"FOV{f:03d}:{len(by_fov[f])}" for f in FOVS)})')
    print(f'selected   : {len(chosen)} alleles, {per_fov} per FOV (seed {SEED})\n')

    frozen, meta, zpred, labels = {}, [], {}, []
    t_start = time.perf_counter()
    for i, (fov, d) in enumerate(chosen, start=1):
        mw._activate_fov(fov)
        allele = AnAllele()
        allele.set_metadata(id=i, fov=fov, cell=d['cell'], anchor_uid=d.get('uid', 0),
                            anchor_hybe=d['hybe'], anchor_channel=d['channel'],
                            coordinate=d['adj_coordinate'],
                            raw_coordinate=d['raw_coordinate'])
        # EXACTLY what _view_chromatin_trace_crop resolves. Without the
        # cell and the resolver the crop loses the cell-level residual --
        # in this dataset that is up to 21 planes of z and several px of
        # xy, so a bench built without them would not be measuring the
        # pipeline at all.
        cell = mw._find_cell_by_id(fov, allele.cell) if allele.cell != -1 else None
        resolver = mw._frame_resolver(cell, fov)
        fov_matrices = mw._composed_fov_matrices_for_cell_alignment(sp, fov)

        t0 = time.perf_counter()
        _a, debug = L.build_chromatin_trace_allele(
            allele, hybes, REFERENCE_HYBE, fid_ch, read_ch, sp, fov, MODALITY,
            cell, fov_matrices, spad=8, z_window=15, z_boundary_trim=0,
            collect_debug=True, resolver=resolver)
        dt = time.perf_counter() - t0

        kept = 0
        for hybe, dbg in (debug or {}).items():
            # the pipeline's OWN prediction of this allele's depth in this
            # hybe's raw stack: allele z (shared) minus the correction
            # that maps this hybe's raw z into the shared frame
            off = L.cell_z_offset(cell, hybe, MODALITY, resolver)
            zpred[f'a{i}|{hybe}'] = (float(allele.coordinate[2]) - float(off),
                                     float(off))
            for kind in ('fiducial', 'readout'):
                cube = dbg.get(f'{kind}_cubic')
                if cube is None:
                    continue
                frozen[f'a{i}|{hybe}|{kind}'] = np.asarray(cube, dtype=np.float32)
                kept += 1
        meta.append((i, fov, d.get('uid', 0), d['cell'], kept))
        label = anchor_label(fov, d)
        labels.append(f'a{i}|{label}')
        note = ''
        if figures_dir:
            written = _save_allele_figures(mw, allele, debug, figures_dir, label)
            note = f'  +{len(written)} figures'
        print(f'  [{i:>3}/{len(chosen)}] {label}  {kept:>3} crops  {dt:6.1f}s{note}')

    np.savez_compressed(
        out_path,
        __meta__=np.array(meta, dtype=np.int64),
        __hybes__=np.array(hybes),
        __pairs__=np.array([f'{a}|{b}|{r}' for a, b, r in replicate_pairs(records)]),
        __zpred__=np.array([f'{k}|{v[0]:.4f}|{v[1]:.4f}' for k, v in zpred.items()]),
        # a{i} -> the SOURCE SPOT, so a frozen bench stays traceable back
        # to the store after the selection that produced it is gone
        __labels__=np.array(labels),
        **frozen)
    print(f'\nfroze {len(frozen)} crops from {len(meta)} alleles -> {out_path} '
          f'({os.path.getsize(out_path) / 1e6:.1f} MB) '
          f'in {time.perf_counter() - t_start:.0f}s')


def replicate_pairs(records):
    """[(folder_a, folder_b, readout_id), ...] -- hybes probing one locus."""
    from collections import defaultdict
    by_id = defaultdict(list)
    for r in records:
        if str(r['datatype']).upper() in DATATYPES:
            by_id[r['readout_id']].append(r['folder'])
    out = []
    for rid, folders in sorted(by_id.items()):
        folders = sorted(folders)
        for i in range(len(folders)):
            for j in range(i + 1, len(folders)):
                out.append((folders[i], folders[j], rid))
    return out


# -- stage 2: score ------------------------------------------------------

def _load(bench_path):
    b = np.load(bench_path, allow_pickle=False)
    zpred = {}
    for s in b['__zpred__']:
        key, zp, off = s.rsplit('|', 2)
        zpred[key] = (float(zp), float(off))
    pairs = [tuple(s.split('|')) for s in b['__pairs__']]
    return b, zpred, pairs


def _seed(cube, key, zpred, source, zhalf=15):
    """
    Where the fit starts, as (y, x, z).

    'argmax' is what ships today: the brightest voxel in the whole
    ~110-plane column, XY and Z together.

    'predicted' uses the pipeline's own answer for the depth -- allele z
    (shared) minus this hybe's cell_z_offset -- and then takes XY from
    the brightest voxel WITHIN that depth neighbourhood. Taking XY from
    the global argmax instead would be a rigged comparison: that voxel
    can sit at a completely different depth, so the seed would be a point
    no emitter occupies.
    """
    if source == 'argmax':
        iy, ix, iz = np.unravel_index(int(np.nanargmax(cube)), cube.shape)
        return float(iy), float(ix), float(iz)
    zp = zpred.get(key.rsplit('|', 1)[0], (None, None))[0]
    if zp is None:
        return _seed(cube, key, zpred, 'argmax')
    zc = int(np.clip(round(zp), 0, cube.shape[2] - 1))
    lo, hi = max(0, zc - zhalf), min(cube.shape[2], zc + zhalf + 1)
    window = cube[:, :, lo:hi]
    if not np.isfinite(window).any():
        return _seed(cube, key, zpred, 'argmax')
    iy, ix, iz = np.unravel_index(int(np.nanargmax(window)), window.shape)
    return float(iy), float(ix), float(iz + lo)


def _fit_one(cube, zhalf, peak_bound, max_sigma):
    """
    One crop, fitted the way the pipeline does it, optionally with the
    fit restricted to +/-zhalf planes around the seed.

    Returns (result_in_full_crop_coords, seed, railed_flags) or None.
    zhalf=None is what ships today: the whole ~110-plane column.

    SEEDING MATCHES PRODUCTION EXACTLY, and the two axes differ:
    _localize_fiducial_hybe seeds XY at the ANCHOR's mapped position
    (x0 = raw_x - xmin, i.e. the crop centre -- the allele is where the
    alignment says it is) but seeds Z at the crop's global ARGMAX,
    because the anchor carries no usable depth. Seeding XY at the argmax
    instead measures a fit the pipeline never performs.
    """
    from codelab_pipeline.localization import localization as L
    iz = int(np.unravel_index(int(np.nanargmax(cube)), cube.shape)[2])
    iy, ix = (cube.shape[0] - 1) / 2.0, (cube.shape[1] - 1) / 2.0
    if zhalf is None:
        sub, z_off = cube, 0
    else:
        lo, hi = max(0, iz - zhalf), min(cube.shape[2], iz + zhalf + 1)
        sub, z_off = cube[:, :, lo:hi], lo
    r = L.fit_gaussian_3d(sub, float(ix), float(iy), float(iz - z_off),
                          peak_bound=peak_bound, max_sigma=max_sigma)
    if r is None:
        return None
    amp, x0, y0, z0, sx, sy, sz, off = r
    railed_pos = abs(abs(z0 - (iz - z_off)) - peak_bound) < 1e-6
    railed_sig = (abs(sx - max_sigma) < 1e-6 or abs(sy - max_sigma) < 1e-6
                  or abs(sz - 2 * max_sigma) < 1e-6)
    return (y0, x0, z0 + z_off), railed_pos, railed_sig


def volumes(bench_path, zhalves, peak_bound, max_sigma):
    """
    Does restricting the FIT VOLUME change the answer? Reported against
    the replicate-pair ground truth, per channel, with railing counts so
    a change in accuracy can be read together with a change in how often
    the fit hit a constraint instead of an optimum.
    """
    b, zpred, pairs = _load(bench_path)
    meta = b['__meta__']
    print(f'bench      : {bench_path}')
    print(f'alleles    : {len(meta)}   crops: {len(b.files) - 5}')
    print(f'bounds     : peak_bound={peak_bound}  max_sigma={max_sigma}\n')

    for kind in ('fiducial', 'readout'):
        keys = [k for k in b.files if k.endswith('|' + kind)]
        depth = b[keys[0]].shape[2]
        print(f'--- {kind}: {len(keys)} crops, full depth {depth} planes ---')
        hdr = (f'{"fit volume":<22}{"accepted":>11}{"pos railed":>12}'
               f'{"sig railed":>12}{"pairs":>10}{"median XY":>12}{"p90":>8}')
        print(hdr)
        print('-' * len(hdr))
        for zhalf in zhalves:
            found, rp, rs, n = {}, 0, 0, 0
            for k in keys:
                out = _fit_one(b[k].astype(float), zhalf, peak_bound, max_sigma)
                if out is None:
                    continue
                pos, railed_pos, railed_sig = out
                n += 1
                rp += railed_pos
                rs += railed_sig
                found[k] = pos
            dists, both, either = [], 0, 0
            for aid, *_r in meta:
                for a, bb, _rid in pairs:
                    ka, kb = f'a{aid}|{a}|{kind}', f'a{aid}|{bb}|{kind}'
                    if ka not in b.files or kb not in b.files:
                        continue
                    either += 1
                    pa, pb2 = found.get(ka), found.get(kb)
                    if pa is None or pb2 is None:
                        continue
                    both += 1
                    dists.append(float(np.linalg.norm(
                        np.array(pa[:2]) - np.array(pb2[:2]))))
            d = np.array(dists) if dists else np.array([np.nan])
            label = 'full column (SHIPS)' if zhalf is None else f'+/-{zhalf} planes'
            med, p90 = np.nanmedian(d), np.nanpercentile(d, 90)
            print(f'{label:<22}{n:>5}/{len(keys):<5}{rp:>7}/{n:<4}{rs:>7}/{n:<4}'
                  f'{both:>6}/{either:<3}{med:>11.3f}px{p90:>7.3f}')
        print()
    print('median XY = distance between two rounds of the SAME locus in the SAME')
    print('allele; lower is better. "railed" = the fit stopped on a constraint')
    print('rather than at an optimum, so its value and its CI are not measurements.')


def score(bench_path, sources, n_max):
    from codelab_pipeline.localization.engine import make_engine
    b, zpred, pairs = _load(bench_path)
    meta = b['__meta__']
    crops = [k for k in b.files if not k.startswith('__')]
    labels = dict(s.split('|', 1) for s in b['__labels__']) if '__labels__' in b.files else {}
    print(f'bench      : {bench_path}')
    print(f'alleles    : {len(meta)}   crops: {len(crops)}')
    print(f'replicates : {len(pairs)} same-locus pairs per allele')
    if labels:
        print('anchors    :')
        for aid, *_rest in meta:
            print(f'   a{aid:<4} {labels.get(f"a{aid}", "?")}')
    print()

    # -- how far apart are the two candidate seeds, before any fitting? --
    gaps = []
    for k in crops:
        if not k.endswith('|readout'):
            continue
        cube = b[k]
        gaps.append(abs(_seed(cube, k, zpred, 'argmax')[2]
                        - _seed(cube, k, zpred, 'predicted')[2]))
    gaps = np.array(gaps)
    print('SEED DISAGREEMENT (readout crops), |argmax z - predicted z|, planes')
    print(f'  median {np.median(gaps):6.1f}   mean {gaps.mean():6.1f}   '
          f'max {gaps.max():6.1f}')
    for thr in (2, 5, 10, 20):
        print(f'  > {thr:>2} planes apart: {100 * (gaps > thr).mean():5.1f}% '
              f'of {len(gaps)} crops')
    print('  (the fit bounds z to seed +/- peak_bound = 2 planes, so any gap')
    print('   above 2 means the two seeds cannot reach the same answer)\n')

    rows = []
    for source in sources:
        engine = make_engine('gaussian')
        found = {}
        t0 = time.perf_counter()
        for k in crops:
            cube = b[k]
            spots = engine.localize(cube, seed_yxz=_seed(cube, k, zpred, source),
                                    n_max=n_max)
            if spots:
                s = spots[0]
                found[k] = (s.y, s.x, s.z)
        dt = time.perf_counter() - t0

        dists, both, either = [], 0, 0
        for aid, _fov, _uid, _cell, _n in meta:
            for a, bb, _rid in pairs:
                ka, kb = f'a{aid}|{a}|readout', f'a{aid}|{bb}|readout'
                if ka not in b.files or kb not in b.files:
                    continue
                either += 1
                pa, pb = found.get(ka), found.get(kb)
                if pa is None or pb is None:
                    continue
                both += 1
                dists.append(float(np.linalg.norm(
                    np.array(pa[:2]) - np.array(pb[:2]))))
        d = np.array(dists) if dists else np.array([np.nan])
        rows.append((source, len(crops), len(found), both, either,
                     np.nanmedian(d), np.nanpercentile(d, 90), dt))

    hdr = f'{"z seed":<12}{"fits":>13}{"pairs":>12}{"median XY":>12}{"p90":>8}{"time":>8}'
    print(hdr)
    print('-' * len(hdr))
    for src, ncrop, nf, both, either, med, p90, dt in rows:
        print(f'{src:<12}{nf:>6}/{ncrop:<6}{both:>6}/{either:<5}'
              f'{med:>11.3f}px{p90:>7.3f}{dt:>7.1f}s')
    print('\nmedian XY = same-locus pair distance; lower is better.')
    print('fits/pairs are shown so an engine cannot look good by rejecting spots.')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    h = sub.add_parser('harvest', help='freeze real crops to an .npz (slow, NAS)')
    h.add_argument('--out', required=True)
    h.add_argument('--alleles', type=int, default=12)
    h.add_argument('--figures', nargs='?', const=DEFAULT_FIGURES_DIR, default=None,
                   help=f'also render View Crop\'s four figures per allele; '
                        f'bare --figures uses {DEFAULT_FIGURES_DIR}')
    s = sub.add_parser('score', help='replay frozen crops (fast, local)')
    s.add_argument('bench')
    s.add_argument('--z-source', action='append', default=None,
                   choices=['argmax', 'predicted'])
    s.add_argument('--n-max', type=int, default=1)
    v = sub.add_parser('volumes', help='does restricting the FIT VOLUME help?')
    v.add_argument('bench')
    v.add_argument('--peak-bound', type=float, default=2.0)
    v.add_argument('--max-sigma', type=float, default=2.5)
    v.add_argument('--z-half', action='append', type=int, default=None,
                   help='half-window in planes; repeatable. Full column always included.')
    a = ap.parse_args()
    if a.cmd == 'harvest':
        harvest(a.out, a.alleles, a.figures)
    elif a.cmd == 'volumes':
        volumes(a.bench, [None] + (a.z_half or [25, 15, 8]),
                a.peak_bound, a.max_sigma)
    else:
        score(a.bench, a.z_source or ['argmax', 'predicted'], a.n_max)


if __name__ == '__main__':
    main()
