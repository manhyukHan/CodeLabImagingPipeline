"""
The learned fiducial against v2's Gaussian fiducial, end to end, on a real
store, scored by replicate distance.

WHAT IS COMPARED. Three arms on the SAME alleles, each through the
production tracer (tracing_v2.build_chromatin_trace_allele):

    v2      Gaussian fiducial, Gaussian readout          -- the reference
    v2+lf   Gaussian readout, LEARNED fiducial (best of one)
    v3+lf   learned readout, learned fiducial            -- the app's v3 page

v2 and v2+lf share the readout engine, so any difference between them
in the same-locus repeat distance is the fiducial alignment alone. v3+lf
is what a person gets by choosing v3 with both models; its readout model
may be another modality's (the readout PSF is near-universal), and the
arm says which one it used.

THE SCORE is engine_ab's: H and R rounds probe the same locus, so after
each round's own fiducial correction they must coincide, and their
separation IS the error. It cannot be gamed by rejecting hybes -- a pair
counts only when both rounds fitted, and the like-for-like number is on
the pairs every arm scored.

WHAT ELSE IS MEASURED, because the fiducial is the thing under test:
how many hybes got a fiducial at all (coverage), the learned p_exist
distribution, how far the learned fiducial sits from the Gaussian one
where both exist (a shift of the whole frame, not of one spot), and
why the learned one was refused when it was.

Usage:
    python tools/fiducial_ab.py --exp MP58 --fid-model models/Manhyuk_DNA555_default
    python tools/fiducial_ab.py --exp MP58 --fid-model ... --read-model D:/models/mp58_rna --alleles 24 --jobs 4
"""
import argparse
import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline import parallel as PL                     # noqa: E402
from codelab_pipeline.localization import tracing_v2 as V2      # noqa: E402
from tools.experiments import EXPERIMENTS, open_session, config_fovs  # noqa: E402
from tools.fit_testbox import replicate_pairs                   # noqa: E402

VOXEL = (0.208, 0.208, 0.2)
DEFAULT_OUT = os.path.join('notes', 'chromatin_tracing_optimization')


def _params(psf_label, fid_model=None, read_model=None, min_p=0.5,
            resolution_kb=None, template_mode='select', fid_min_p=None):
    from codelab_pipeline.localization import psf_library as LIB
    doc = LIB.read(psf_label)
    got = LIB.shape_tuple(doc) if doc else None
    return V2.V2Params(voxel_um=VOXEL,
                       psf_family=got[0] if got else None,
                       psf_shape=got[1] if got else None,
                       psf_label=psf_label if got else '',
                       qc_shift=False,
                       fiducial_model_dir=fid_model,
                       readout_model_dir=read_model,
                       min_p_exist=min_p,
                       min_p_exist_fiducial=fid_min_p,
                       genomic_resolution_kb=resolution_kb,
                       template_mode=template_mode)


def _init(exp_name, psf_label, fid_model, read_model, min_p, arms_wanted=None):
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
    # THE TEMPLATE ARMS. The registry's step_kb is the experiment's
    # genomic resolution; 'select' matches with the bank's template for
    # it, 'pooled' with the pooled mean, 'all' with every candidate.
    kb = exp.step_kb
    arms = [('v2', _params(psf_label)),
            ('v2+lf', _params(psf_label, fid_model=fid_model, min_p=min_p,
                              resolution_kb=kb, template_mode='select')),
            ('v2+lf/pooled', _params(psf_label, fid_model=fid_model, min_p=min_p,
                                     resolution_kb=kb, template_mode='pooled')),
            ('v2+lf/all', _params(psf_label, fid_model=fid_model, min_p=min_p,
                                  resolution_kb=kb, template_mode='all'))]
    if read_model:
        arms.append(('v3+lf', _params(psf_label, fid_model=fid_model,
                                      read_model=read_model, min_p=min_p,
                                      resolution_kb=kb)))
    if arms_wanted:
        arms = [a for a in arms if a[0] in arms_wanted]
    return {'exp': exp, 'mw': mw, 'sp': sp, 'hybes': hybes, 'fid_ch': fid_ch,
            'read_ch': read_ch, 'arms': arms, 'fov': None}


def _one(item, st):
    """One allele through every arm. Returns positions and fiducial facts."""
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
    for key, p in st['arms']:
        allele = AnAllele()
        allele.set_metadata(id=i, fov=fov, cell=d['cell'],
                            anchor_uid=d.get('uid', 0), anchor_hybe=d['hybe'],
                            anchor_channel=d['channel'],
                            coordinate=d['adj_coordinate'],
                            raw_coordinate=d['raw_coordinate'])
        t0 = time.perf_counter()
        _a, debug = V2.build_chromatin_trace_allele(
            allele, st['hybes'], exp.reference_hybe, st['fid_ch'],
            st['read_ch'], st['sp'], fov, exp.modality, cell, fov_matrices,
            params=p, spad=8, collect_debug=True, resolver=resolver)
        secs = time.perf_counter() - t0
        poly = {}
        for h, comps in (allele.polymer_adj or {}).items():
            if comps:
                best = max(comps, key=lambda c: c[3])
                poly[h] = (float(best[0]), float(best[1]), float(best[2]))
        fid = {h: tuple(float(v) for v in t[:3])
               for h, t in (allele.fiducial_trace_adj or {}).items()
               if t is not None}
        facts = {}
        for h in st['hybes']:
            dh = (debug or {}).get(h) or {}
            facts[h] = {'engine': dh.get('fiducial_engine'),
                        'how': dh.get('fiducial_how'),
                        'p': dh.get('fiducial_p_exist'),
                        'n_cand': dh.get('fiducial_n_candidates'),
                        'why': dh.get('fiducial_dropped_why')
                        or (allele.rejected_hybes or {}).get(h)}
        out[key] = {'polymer_adj': poly, 'fiducial_adj': fid,
                    'n_rejected': len(allele.rejected_hybes or {}),
                    'rejected': {h: str(w) for h, w in
                                 (allele.rejected_hybes or {}).items()},
                    'facts': facts, 'seconds': secs}
    return out


def _pairs(results, pairs, key):
    """{(allele, a, b): (d3_um, dxy_um)} for one arm."""
    dy, dx, dz = VOXEL
    out, either = {}, 0
    for r in results:
        poly = r[key]['polymer_adj']
        for a, b, _rid in pairs:
            either += 1
            if a not in poly or b not in poly:
                continue
            pa, pb = np.array(poly[a]), np.array(poly[b])
            d = (pa - pb) * np.array([dx, dy, dz])
            out[(r['i'], a, b)] = (float(np.linalg.norm(d)),
                                   float(np.linalg.norm(d[:2])))
    return out, either


def _kind(why):
    """One refusal -> its KIND, so 'z 17.6 planes from expected' and
    'z 20.5 planes ...' count together. A learned refusal lists every
    candidate's reason; the kind names the most permissive one present,
    since a candidate refused only for being unrefined was closer to
    acceptance than one at p 0.00."""
    import re
    text = ' '.join(why) if isinstance(why, (list, tuple)) else str(why)
    if 'found nothing' in text:
        return 'learned: engine found nothing'
    kinds = []
    if 'no sub-voxel' in text:
        kinds.append('unrefined')
    if 'from expected' in text:
        kinds.append('out of z reach')
    if 'p_exist' in text:
        kinds.append('below min p_exist')
    if kinds:
        return ('learned: every candidate ' + kinds[0] if len(kinds) == 1
                else 'learned: mixed (' + '; '.join(kinds) + ')')
    if 'occupancy' in text:
        return 'gaussian: occupancy below 0.25'
    return re.sub(r'[-+]?\d+(\.\d+)?', 'N', text)[:70]


def _fiducial_stats(good, key, hybes, ref_key='v2'):
    """Coverage, p_exist, shift against the Gaussian fiducial, refusals."""
    n_hybe = len(hybes) * len(good)
    cov = sum(len(r[key]['fiducial_adj']) for r in good)
    ps = [f['p'] for r in good for f in r[key]['facts'].values()
          if f.get('p') is not None and np.isfinite(f['p'])]
    shifts_xy, shifts_z = [], []
    for r in good:
        a, b = r[key]['fiducial_adj'], r[ref_key]['fiducial_adj']
        for h in a:
            if h in b:
                d = np.array(a[h]) - np.array(b[h])
                shifts_xy.append(float(np.hypot(d[0], d[1])))
                shifts_z.append(float(abs(d[2])))
    whys = Counter()
    for r in good:
        for h in hybes:
            if h not in r[key]['fiducial_adj']:
                w = r[key]['facts'][h].get('why') or r[key]['rejected'].get(h) \
                    or 'no fiducial (unsaid)'
                whys[_kind(w)] += 1
    hows = Counter(str(r[key]['facts'][h].get('how'))
                   for r in good for h in r[key]['fiducial_adj']
                   if r[key]['facts'][h].get('how'))
    st = {'hybes_with_fiducial': cov, 'hybes_possible': n_hybe,
          'coverage_pct': round(100.0 * cov / max(1, n_hybe), 1),
          'how': dict(hows.most_common()),
          'refusals': dict(whys.most_common(8))}
    if ps:
        ps = np.array(ps, float)
        st['p_exist'] = {'n': int(len(ps)), 'median': round(float(np.median(ps)), 3),
                         'p10': round(float(np.percentile(ps, 10)), 3),
                         'frac_ge_0.5': round(float((ps >= 0.5).mean()), 3),
                         'frac_ge_0.9': round(float((ps >= 0.9).mean()), 3)}
    if shifts_xy and key != ref_key:
        st['shift_vs_gaussian'] = {
            'n': len(shifts_xy),
            'xy_px_median': round(float(np.median(shifts_xy)), 3),
            'xy_px_p90': round(float(np.percentile(shifts_xy, 90)), 3),
            'z_planes_median': round(float(np.median(shifts_z)), 3),
            'z_planes_p90': round(float(np.percentile(shifts_z, 90)), 3),
            'xy_over_2px_pct': round(100.0 * float(np.mean(np.array(shifts_xy) > 2.0)), 1)}
    return st


def run_one(name, n_alleles, jobs, psf_label, fid_model, read_model, min_p,
            arms_wanted=None):
    exp = EXPERIMENTS[name]
    mw, sp = open_session(exp)
    records = mw._hybe_records_for_storage_path(sp)
    pairs = [(a, b, r) for a, b, r in replicate_pairs(records)]
    fovs = list(exp.fovs) if exp.fovs else list(config_fovs(exp.config))
    by_folder = {r['folder']: r for r in records}
    hybes = [r['folder'] for r in records
             if str(r['datatype']).upper() in exp.datatypes]

    from codelab_pipeline.io import analysis_store as V
    rng = random.Random(exp.seed)
    per = max(1, n_alleles // len(fovs))
    chosen = []
    for fov in fovs:
        pool = V.read_spots(sp, fov, exp.modality, exp.anchor_hybe,
                            exp.anchor_channel)
        for d in rng.sample(pool, min(per, len(pool))):
            chosen.append((fov, d))
    chosen.sort(key=lambda t: t[0])
    items = [(i, fov, d) for i, (fov, d) in enumerate(chosen, start=1)]

    print(f'=== {name} ({exp.scope_mb} Mb) ===')
    print(f'  store      : {sp}')
    print(f'  fiducial   : {fid_model}   (min p_exist {min_p}, resolution '
          f'{exp.step_kb} kb)')
    print(f'  readout v3 : {read_model or "-- (no v3 arm)"}')
    print(f'  alleles    : {len(items)} over FOV {fovs}   hybes {len(hybes)}   '
          f'pairs/allele: {len(pairs)}   possible: {len(pairs) * len(items)}',
          flush=True)

    t0 = time.perf_counter()

    def prog(n, total, _i, r):
        if n % 4 == 0 or n == total:
            el = time.perf_counter() - t0
            print(f'    [{n:>3}/{total}] {el:6.0f}s  '
                  f'eta {(total - n) * el / max(n, 1) / 60:5.1f} min', flush=True)

    res = PL.pmap(_one, items, kind='io', jobs=jobs, initializer=_init,
                  initargs=(name, psf_label, fid_model, read_model, min_p,
                            arms_wanted),
                  on_done=prog, chunksize=1)
    good = PL.ok(res)
    wall = time.perf_counter() - t0
    for f in PL.failures(res)[:3]:
        print('    FAILED:', f.error)
    if not good:
        raise RuntimeError('no allele traced')

    arms = [k for k in ('v2', 'v2+lf', 'v2+lf/pooled', 'v2+lf/all', 'v3+lf')
            if k in good[0]]
    D = {k: _pairs(good, pairs, k)[0] for k in arms}
    common = sorted(set.intersection(*[set(D[k]) for k in arms]))
    row = {'experiment': name, 'alleles': len(good), 'hybes': len(hybes),
           'fovs': fovs, 'wall_s': round(wall, 1), 'psf': psf_label,
           'fiducial_model': fid_model, 'readout_model': read_model,
           'min_p_exist': min_p, 'common_pairs': len(common),
           'possible_pairs': len(pairs) * len(good), 'arms': {}}
    for k in arms:
        a = {'seconds': round(sum(r[k]['seconds'] for r in good), 1),
             'pairs': len(D[k]),
             'hybes_rejected': sum(r[k]['n_rejected'] for r in good)}
        if D[k]:
            a3 = np.array([v[0] for v in D[k].values()])
            axy = np.array([v[1] for v in D[k].values()])
            a['median_3d_um'] = round(float(np.median(a3)), 4)
            a['median_xy_um'] = round(float(np.median(axy)), 4)
            a['p90_3d_um'] = round(float(np.percentile(a3, 90)), 4)
        if common:
            a['common_median_3d_um'] = round(
                float(np.median([D[k][c][0] for c in common])), 4)
        a['fiducial'] = _fiducial_stats(good, k, hybes)
        row['arms'][k] = a
    # EVERY HYBE OF EVERY ALLELE, for analysis after the fact: the
    # fiducial position per arm and the learned engine's facts.
    row['_details'] = [{'i': r['i'], 'fov': r['fov'], 'cell': r['cell'],
                        'arms': {k: {'fiducial_adj': r[k]['fiducial_adj'],
                                     'facts': r[k]['facts'],
                                     'rejected': r[k]['rejected']}
                                 for k in arms}} for r in good]
    if common and 'v2+lf' in D:
        m1 = np.median([D['v2'][c][0] for c in common])
        m2 = np.median([D['v2+lf'][c][0] for c in common])
        row['lf_change_pct'] = round(float(100 * (m2 - m1) / m1), 1)
        row['lf_closer_pairs'] = sum(1 for c in common
                                     if D['v2+lf'][c][0] < D['v2'][c][0])
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exp', default='MP58')
    ap.add_argument('--fid-model', required=True)
    ap.add_argument('--read-model', default=None)
    ap.add_argument('--alleles', type=int, default=24)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--psf', default='universal-default')
    ap.add_argument('--min-p', type=float, default=0.5)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--arms', default=None,
                    help='comma list of arms to run, e.g. v2,v2+lf,v2+lf/pooled')
    a = ap.parse_args()

    row = run_one(a.exp, a.alleles, a.jobs, a.psf, os.path.abspath(a.fid_model),
                  os.path.abspath(a.read_model) if a.read_model else None,
                  a.min_p, [x.strip() for x in a.arms.split(',')] if a.arms else None)
    print()
    print('=' * 96)
    print(f'{"arm":<8}{"pairs":>12}  {"median 3D":>10}{"common 3D":>11}'
          f'{"p90 3D":>9}  {"fid cov":>8}  {"rejected":>9}  {"s":>7}')
    print('=' * 96)
    for k, r in row['arms'].items():
        print(f'{k:<8}{r["pairs"]:>5}/{row["possible_pairs"]:<6}  '
              f'{r.get("median_3d_um", float("nan")):>9.4f}u'
              f'{r.get("common_median_3d_um", float("nan")):>10.4f}u'
              f'{r.get("p90_3d_um", float("nan")):>8.4f}u  '
              f'{r["fiducial"]["coverage_pct"]:>7.1f}%  '
              f'{r["hybes_rejected"]:>9}  {r["seconds"]:>7.0f}')
    print(f'\ncommon pairs: {row["common_pairs"]}   '
          + (f'learned fiducial change {row["lf_change_pct"]:+.1f}%  '
             f'(closer on {row["lf_closer_pairs"]}/{row["common_pairs"]})'
             if 'lf_change_pct' in row else ''))
    for k, r in row['arms'].items():
        f = r['fiducial']
        print(f'\n{k}: fiducial on {f["hybes_with_fiducial"]}/{f["hybes_possible"]} hybes')
        if f.get('how'):
            print(f'    how      {f["how"]}')
        if 'p_exist' in f:
            print(f'    p_exist  {f["p_exist"]}')
        if 'shift_vs_gaussian' in f:
            print(f'    shift vs Gaussian  {f["shift_vs_gaussian"]}')
        for w, n in f['refusals'].items():
            print(f'    {n:>5}  {w}')
    os.makedirs(a.out, exist_ok=True)
    if row.get('_details'):
        dpath = os.path.join(a.out, f'fiducial_ab_details_{a.exp}.json')
        with open(dpath, 'w', encoding='utf-8') as fh:
            json.dump(row.pop('_details'), fh)
        print(f'per-hybe details: {dpath}')
    path = os.path.join(a.out, f'fiducial_ab_{a.exp}.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(row, fh, indent=2)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
