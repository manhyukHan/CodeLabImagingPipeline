"""
v1 against v2, end to end, on real stores, scored by replicate distance.

NOT a frozen-bench replay. Both engines run through the production path --
real crops read from the NAS, real frame resolution, real gates -- because
the wiring is part of what is being tested and a bench replay would skip
it entirely.

WHAT IS HELD FIXED
------------------
The SAME alleles, in the same order, from the same seed, for both arms.
Each arm gets its own AnAllele instance because tracing mutates it in
place, but the metadata is identical, so the only difference between the
two runs is the engine.

THE SCORE
---------
Same-locus repeat distance. H and R rounds probe the same physical locus,
so after each round's own fiducial correction they must coincide, and
their separation IS localization error -- no simulation, no labelling.
Toehold rounds are excluded: they are displacement controls, and scoring
them books designed displacement as error.

It cannot be gamed by rejecting spots. A pair counts only when BOTH
rounds produced a fit in that arm, and the coverage is printed beside
every median. The like-for-like comparison is the pairs BOTH arms
scored -- an engine that drops the hard rounds and then shows a better
median has not improved anything.

Usage:
    python tools/engine_ab.py --exp MP58 --alleles 24
    python tools/engine_ab.py --all --jobs 8
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline import parallel as PL                     # noqa: E402
from codelab_pipeline.localization import tracing_v2 as V2      # noqa: E402
from tools.experiments import EXPERIMENTS, open_session, config_fovs  # noqa: E402
from tools.fit_testbox import replicate_pairs                   # noqa: E402

VOXEL = (0.208, 0.208, 0.2)
DEFAULT_OUT = os.path.join('notes', 'chromatin_tracing_optimization')


def _init(exp_name, psf_label):
    from codelab_pipeline.localization import psf_library as LIB
    exp = EXPERIMENTS[exp_name]
    mw, sp = open_session(exp)
    records = mw._hybe_records_for_storage_path(sp)
    by_folder = {r['folder']: r for r in records}
    hybes = [r['folder'] for r in records
             if str(r['datatype']).upper() in exp.datatypes]
    fid_ch = {h: by_folder[h]['fiducial_channel'] for h in hybes}
    read_ch = {}
    for h in hybes:
        others = [c for c in by_folder[h]['channels']
                  if c != by_folder[h]['fiducial_channel']]
        read_ch[h] = others[0] if others else by_folder[h]['fiducial_channel']
    doc = LIB.read(psf_label)
    got = LIB.shape_tuple(doc) if doc else None
    v2p = V2.V2Params(voxel_um=VOXEL,
                      psf_family=got[0] if got else None,
                      psf_shape=got[1] if got else None,
                      psf_label=psf_label if got else '',
                      qc_shift=False)     # QC costs time and is not scored here
    return {'exp': exp, 'mw': mw, 'sp': sp, 'hybes': hybes, 'fid_ch': fid_ch,
            'read_ch': read_ch, 'v2p': v2p, 'fov': None}


def _one(item, st):
    """One allele through BOTH engines. Returns polymers, not fits."""
    from codelab_pipeline.models.allele import AnAllele
    i, fov, d = item
    mw, exp = st['mw'], st['exp']
    if st['fov'] != fov:
        mw._activate_fov(fov)
        st['fov'] = fov
    cell = mw._find_cell_by_id(fov, d['cell']) if d['cell'] != -1 else None
    resolver = mw._frame_resolver(cell, fov)
    fov_matrices = mw._composed_fov_matrices_for_cell_alignment(st['sp'], fov)

    out = {'i': i, 'fov': fov, 'cell': d['cell'], 'uid': d.get('uid', 0)}
    for engine, key in ((None, 'v1'), ('v2', 'v2')):
        # A FRESH allele per arm: tracing fills fiducial_trace_adj/polymer_adj/
        # rejected_hybes in place, so reusing one would let the first arm
        # decide what the second even attempts.
        allele = AnAllele()
        allele.set_metadata(id=i, fov=fov, cell=d['cell'],
                            anchor_uid=d.get('uid', 0), anchor_hybe=d['hybe'],
                            anchor_channel=d['channel'],
                            coordinate=d['adj_coordinate'],
                            raw_coordinate=d['raw_coordinate'])
        t0 = time.perf_counter()
        V2.trace_allele(engine, allele, st['hybes'], exp.reference_hybe,
                        st['fid_ch'], st['read_ch'], st['sp'], fov,
                        exp.modality, cell, fov_matrices,
                        v2_params=st['v2p'], spad=8, resolver=resolver)
        # keep only the brightest component per hybe; v1 can return several
        poly = {}
        for h, comps in (allele.polymer_adj or {}).items():
            if comps:
                best = max(comps, key=lambda c: c[3])
                poly[h] = (float(best[0]), float(best[1]), float(best[2]))
        out[key] = {'polymer_adj': poly,
                    'n_rejected': len(allele.rejected_hybes or {}),
                    'seconds': time.perf_counter() - t0}
    return out


def _pairs(results, pairs, key):
    """{(allele, a, b): (d3_um, dxy_um)} for one engine."""
    dy, dx, dz = VOXEL
    out, either = {}, 0
    for r in results:
        poly = r[key]['polymer_adj']
        for a, b, _rid in pairs:
            either += 1
            if a not in poly or b not in poly:
                continue
            # polymer_adj entries are (x, y, z) in the shared frame
            pa, pb = np.array(poly[a]), np.array(poly[b])
            d = (pa - pb) * np.array([dx, dy, dz])
            out[(r['i'], a, b)] = (float(np.linalg.norm(d)),
                                   float(np.linalg.norm(d[:2])))
    return out, either


def run_one(name, n_alleles, jobs, psf_label):
    exp = EXPERIMENTS[name]
    mw, sp = open_session(exp)
    records = mw._hybe_records_for_storage_path(sp)
    pairs = [(a, b, r) for a, b, r in replicate_pairs(records)]
    fovs = list(exp.fovs) if exp.fovs else list(config_fovs(exp.config))

    from codelab_pipeline.io import analysis_store as V
    rng = random.Random(exp.seed)
    per = max(1, n_alleles // len(fovs))
    chosen = []
    for fov in fovs:
        pool = V.read_spots(sp, fov, exp.modality, exp.anchor_hybe,
                            exp.anchor_channel)
        for d in rng.sample(pool, min(per, len(pool))):
            chosen.append((fov, d))
    chosen.sort(key=lambda t: t[0])          # FOV-major: _activate_fov is dear
    items = [(i, fov, d) for i, (fov, d) in enumerate(chosen, start=1)]

    print(f'=== {name} ({exp.scope_mb} Mb) ===')
    print(f'  store    : {sp}')
    print(f'  alleles  : {len(items)} over FOV {fovs}   '
          f'pairs/allele: {len(pairs)}   possible: {len(pairs) * len(items)}',
          flush=True)

    t0 = time.perf_counter()

    def prog(n, total, _i, r):
        if n % 5 == 0 or n == total:
            el = time.perf_counter() - t0
            print(f'    [{n:>3}/{total}] {el:6.0f}s  '
                  f'eta {(total - n) * el / max(n, 1) / 60:5.1f} min', flush=True)

    res = PL.pmap(_one, items, kind='io', jobs=jobs, initializer=_init,
                  initargs=(name, psf_label), on_done=prog, chunksize=2)
    good = PL.ok(res)
    wall = time.perf_counter() - t0
    for f in PL.failures(res)[:3]:
        print('    FAILED:', f.error)

    A, either = _pairs(good, pairs, 'v1')
    B, _ = _pairs(good, pairs, 'v2')
    common = sorted(set(A) & set(B))
    row = {'experiment': name, 'scope_mb': exp.scope_mb,
           'alleles': len(good), 'possible_pairs': either // 2,
           'wall_s': round(wall, 1), 'psf': psf_label,
           'common_pairs': len(common)}
    for key, D, arm in (('v1', A, 'v1'), ('v2', B, 'v2')):
        secs = sum(r[arm]['seconds'] for r in good)
        rej = sum(r[arm]['n_rejected'] for r in good)
        row[f'{key}_pairs'] = len(D)
        row[f'{key}_seconds'] = round(secs, 1)
        row[f'{key}_rejected'] = rej
        if D:
            a3 = np.array([v[0] for v in D.values()])
            axy = np.array([v[1] for v in D.values()])
            row[f'{key}_median_3d_um'] = round(float(np.median(a3)), 4)
            row[f'{key}_median_xy_um'] = round(float(np.median(axy)), 4)
            row[f'{key}_p90_3d_um'] = round(float(np.percentile(a3, 90)), 4)
    if common:
        m1 = np.median([A[k][0] for k in common])
        m2 = np.median([B[k][0] for k in common])
        row['common_v1_median_3d_um'] = round(float(m1), 4)
        row['common_v2_median_3d_um'] = round(float(m2), 4)
        row['v2_change_pct'] = round(float(100 * (m2 - m1) / m1), 1)
        row['v2_closer_pairs'] = sum(1 for k in common if B[k][0] < A[k][0])
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exp', action='append', default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--alleles', type=int, default=None)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--psf', default='universal-default')
    ap.add_argument('--out', default=DEFAULT_OUT)
    a = ap.parse_args()

    names = list(EXPERIMENTS) if a.all else (a.exp or ['MP58'])
    # Sized so every experiment contributes a COMPARABLE number of pairs,
    # not a comparable number of alleles: HoxA has 2 replicate pairs per
    # allele against CrossMod's 8, so equal allele counts would give it a
    # quarter of the evidence.
    per_pairs = {'MP58': 24, 'Chr19': 32, 'CrossMod': 20, 'HoxA': 60}
    rows = []
    for name in names:
        n = a.alleles or per_pairs.get(name, 24)
        try:
            rows.append(run_one(name, n, a.jobs, a.psf))
        except Exception as e:
            import traceback
            print(f'  {name} FAILED: {type(e).__name__}: {e}')
            traceback.print_exc()
        print(flush=True)

    if not rows:
        return
    print('=' * 100)
    print(f'{"experiment":<11}{"Mb":>6}  {"pairs":>11}  {"v1 median":>11}'
          f'{"v2 median":>11}{"change":>9}  {"v2 closer":>11}  {"v1 s":>7}{"v2 s":>7}')
    print('=' * 100)
    for r in rows:
        c = r.get('common_pairs', 0)
        print(f'{r["experiment"]:<11}{r["scope_mb"]:>6}  '
              f'{c:>5}/{r.get("possible_pairs", 0):<5}  '
              f'{r.get("common_v1_median_3d_um", float("nan")):>10.4f}u'
              f'{r.get("common_v2_median_3d_um", float("nan")):>10.4f}u'
              f'{r.get("v2_change_pct", float("nan")):>8.1f}%  '
              f'{r.get("v2_closer_pairs", 0):>4}/{c:<5}  '
              f'{r.get("v1_seconds", 0):>6.0f}{r.get("v2_seconds", 0):>7.0f}')
    print('\nmedian = same-locus repeat distance on the pairs BOTH engines '
          'scored.\nnegative change = v2 better. coverage is shown so neither '
          'engine can\nlook good by rejecting the hard rounds.')

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, 'engine_ab.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
