"""
Calibrate every experiment's PSF, in parallel, and compare them.

THE QUESTION
------------
One microscope, one optical configuration -- so can ONE default PSF serve
every experiment? MP58 alone cannot answer it: a shape derived and tested
on the same dataset is not evidence about any other.

It is really two questions with OPPOSITE expected answers, because the
two channels image different kinds of object:

    readout    a single locus -- a point source. Optical. Should be the
               SAME everywhere if the microscope is the only thing
               setting it.
    fiducial   the whole traced region -- an extended object whose size
               is PHYSICAL. Should scale with genomic scope, and these
               four span 2 Mb to ~20 Mb.

So the readout shapes agreeing across experiments would justify a
shipped default; the fiducial shapes tracking scope would confirm what
the 1.7x fiducial/readout gap in MP58 was telling us. Either can fail.

THE SCALE TO BEAT
-----------------
Not zero. MP58's own readout calibration moved ~40 nm when the crop
SELECTION changed (188 nm brightest-first vs 146 nm first-40), and the
family flipped with it. Two experiments agreeing to better than that
scatter means something; agreeing to worse than it means nothing.

PARALLELISM
-----------
A (bench, kind, family) triple is an independent fit -- families share no
state, and the winner is chosen afterwards. Four experiments x two kinds
x four families is 32 jobs against 32 physical cores.

Crops are gathered in the PARENT and shipped with the job (~5 MB), so no
worker ever loads a 330 MB bench. Recombination goes through
psf.select_best, the same code `calibrate` uses, so a parallel result and
a serial one cannot disagree about which family won.

Usage:
    python tools/psf_survey.py --bench MP58=bench/mp58_48.npz \\
                               --bench HoxA=bench/hoxa_160.npz
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline import parallel as PL                     # noqa: E402
from codelab_pipeline.localization import psf as P              # noqa: E402
from tools.experiments import EXPERIMENTS                       # noqa: E402
from tools.calibrate_psf import gather, render                  # noqa: E402

DEFAULT_OUT = os.path.join('notes', 'chromatin_tracing_optimization')


def _fit_family(item):
    """One family against one set of crops. The whole parallel unit."""
    exp_name, kind, family, n, crops, voxel, radius = item
    t0 = time.perf_counter()
    # families=[family] makes calibrate fit exactly this one; verbose off
    # because 32 workers interleaving progress lines is unreadable.
    r = P.calibrate(crops, voxel_um=voxel, families=[family],
                    fit_radius_um=radius, verbose=False)
    return {'exp': exp_name, 'kind': kind, 'family': family, 'n': n,
            'fit_radius_um': list(radius),
            'params': r[family]['params'], 'score': r[family]['score'],
            'n_crops': r[family]['n_crops'], 'seconds': time.perf_counter() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bench', action='append', required=True,
                    metavar='NAME=PATH', help='repeatable, e.g. MP58=bench/mp58_48.npz')
    # A LIST, not a number. The readout calibration is known to be badly
    # constrained at 40 crops -- the winning family flipped and sigma moved
    # ~40 nm when the crop SELECTION changed, which is the signature of a
    # fit that has not converged in the amount of data. One crop count
    # cannot show that; a series can, and a series that stops moving is the
    # evidence that the count is finally enough.
    ap.add_argument('--crops', default='40',
                    help='comma-separated counts, e.g. 40,80,160')
    # Defaults to the two families with a MECHANISM behind them. moffat
    # models atmospheric seeing, which a microscope does not have, and a
    # Lorentzian has no diffraction basis and undefined variance -- so its
    # "sigma" is not the width the rest of the code assumes. Measured over
    # 4 experiments x 2 channels: moffat won 0 of 8, lorentzian won 0 of 8
    # plausibly (it took MP58's best readout score by going degenerate,
    # sigma on its lower bound), and the two were the slowest jobs in every
    # experiment. Still selectable, so the claim stays falsifiable.
    ap.add_argument('--families', default='gaussian,gaussian_halo',
                    help='comma-separated; default is the two with a '
                         'physical mechanism. Pass all four to re-test.')
    # The fit window, in micrometres. NOT cosmetic: at the default 0.8 um
    # the large-scope fiducials (sigma ~400 nm) are fitted over only +/-2
    # sigma, so the fit never sees the tail whose width it is measuring and
    # sigma comes back biased LOW -- worst for exactly the largest regions,
    # which is where any scaling with genomic scope would show. A crop is
    # 17 px at 0.208 um, so 1.6 um is about the largest radius that still
    # fits inside it.
    ap.add_argument('--fit-radius', default='0.8,0.8,2.0',
                    help='lateral,lateral,axial in um (default 0.8,0.8,2.0)')
    ap.add_argument('--kinds', default='fiducial,readout')
    ap.add_argument('--jobs', type=int, default=None)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--voxel', default='0.208,0.208,0.2')
    ap.add_argument('--no-figures', action='store_true')
    a = ap.parse_args()

    voxel = tuple(float(v) for v in a.voxel.split(','))
    crop_counts = sorted({int(c) for c in a.crops.split(',') if c.strip()})
    radius = tuple(float(v) for v in a.fit_radius.split(','))
    kinds = tuple(k.strip() for k in a.kinds.split(',') if k.strip())
    families = [f.strip() for f in a.families.split(',') if f.strip()]
    unknown = [f for f in families if f not in P.FAMILIES]
    if unknown:
        sys.exit(f'unknown families {unknown}; known: {sorted(P.FAMILIES)}')
    benches = {}
    for spec in a.bench:
        name, _, path = spec.partition('=')
        if name not in EXPERIMENTS:
            sys.exit(f'unknown experiment {name!r}; known: {sorted(EXPERIMENTS)}')
        benches[name] = path

    # -- gather in the parent: each bench is opened exactly once ---------
    items, crops_by = [], {}
    for name, path in benches.items():
        exp = EXPERIMENTS[name]
        t0 = time.perf_counter()
        bench = np.load(path, allow_pickle=False)
        got = []
        for kind in kinds:
            # gather is brightest-first, so the n=40 set is a PREFIX of the
            # n=160 set. That makes the series nested: a shape that moves
            # between counts moved because of the extra crops, not because
            # it is looking at a different sample.
            biggest, _keys = gather(bench, exp.reference_hybe, kind,
                                    max(crop_counts))
            if not biggest:
                print(f'{name}/{kind}: no {exp.reference_hybe} crops in {path}')
                continue
            for n in crop_counts:
                if n > len(biggest):
                    continue
                crops = biggest[:n]
                crops_by[(name, kind, n)] = crops
                for family in families:
                    items.append((name, kind, family, n, crops, voxel, radius))
            got.append(f'{kind}:{len(biggest)}')
        print(f'{name:<10} {os.path.basename(path):<22} loaded in '
              f'{time.perf_counter() - t0:5.1f}s   ' + '  '.join(got), flush=True)

    jobs = PL.cpu_budget('cpu', a.jobs, len(items))
    print(f'\n{len(items)} independent fits on {jobs} workers\n', flush=True)

    t0 = time.perf_counter()

    def progress(n, total, _i, r):
        if isinstance(r, PL.Failure):
            print(f'  [{n:>2}/{total}] FAILED {r.error}', flush=True)
        else:
            print(f'  [{n:>3}/{total}] {r["exp"]:<10} {r["kind"]:<9} '
                  f'{r["family"]:<14} n={r["n"]:<4} {r["seconds"]:6.1f}s  '
                  f'rss/vox {r["score"]:.0f}', flush=True)

    results = PL.pmap(_fit_family, items, kind='cpu', jobs=jobs, on_done=progress)
    wall = time.perf_counter() - t0
    good = PL.ok(results)
    cpu = sum(r['seconds'] for r in good)
    print(f'\nwall {wall:.0f}s   summed fit time {cpu:.0f}s   '
          f'speedup {cpu / wall:.1f}x on {jobs} workers')
    if PL.failures(results):
        for f in PL.failures(results):
            print('  FAILED:', f.error)

    # -- recombine through psf's OWN selection --------------------------
    summary = {}
    biggest_n = max(crop_counts)
    for (name, kind, n), crops in crops_by.items():
        per = {r['family']: {'params': r['params'], 'score': r['score'],
                             'n_crops': r['n_crops']}
               for r in good
               if r['exp'] == name and r['kind'] == kind and r['n'] == n}
        if not per:
            continue
        P.select_best(per, verbose=False)
        best = per['best']
        summary[f'{name}|{kind}|{n}'] = {
            'experiment': name, 'kind': kind, 'n': n,
            'fit_radius_um': list(radius),
            'scope_mb': EXPERIMENTS[name].scope_mb,
            'reference_hybe': EXPERIMENTS[name].reference_hybe,
            'best': best, 'params': per[best]['params'],
            'score': per[best]['score'], 'n_crops': per[best]['n_crops'],
            'all_implausible': per.get('all_implausible', False),
            'per_family': {f: per[f] for f in per
                           if isinstance(per[f], dict) and 'params' in per[f]},
        }
        # Only the largest count gets a figure -- the smaller ones exist to
        # show convergence, not to be inspected one by one.
        if not a.no_figures and n == biggest_n:
            try:
                fig = render(crops, per, voxel, kind,
                             f'{name}_{EXPERIMENTS[name].reference_hybe}_n{n}', a.out)
                summary[f'{name}|{kind}|{n}']['figure'] = fig
            except Exception as e:
                print(f'  figure for {name}/{kind} failed: {type(e).__name__}: {e}')

    print('\n' + '=' * 92)
    print(f'{"experiment":<11}{"Mb":>4}  {"kind":<9}{"n":>5}  {"best family":<15}'
          f'{"sxy nm":>8}{"sz nm":>8}   plausible')
    print('=' * 92)
    for key in sorted(summary, key=lambda k: (summary[k]['kind'],
                                              summary[k]['scope_mb'] or 0,
                                              summary[k]['experiment'],
                                              summary[k]['n'])):
        s = summary[key]
        pr = s['params']
        sxy = 1000 * pr.get('sigma_xy_um', float('nan'))
        sz = 1000 * pr.get('sigma_z_um', float('nan'))
        print(f'{s["experiment"]:<11}{s["scope_mb"] or 0:>4}  {s["kind"]:<9}'
              f'{s["n"]:>5}  {s["best"]:<15}{sxy:>8.0f}{sz:>8.0f}   '
              f'{"NO -- unusable" if s["all_implausible"] else "yes"}')

    # -- has the shape stopped moving? ----------------------------------
    #
    # The number that matters is not sigma at the largest count, it is how
    # much sigma CHANGED on the way there. A calibration still drifting at
    # the last step has not converged, and quoting its final value as the
    # answer would be reporting a waypoint as a destination.
    if len(crop_counts) > 1:
        print('\nconvergence -- change in sigma_xy over the last step '
              '(the scale to beat is the ~40 nm this moves under a '
              'different crop SELECTION):')
        lo, hi = crop_counts[-2], crop_counts[-1]
        for name in sorted(benches):
            for kind in ('fiducial', 'readout'):
                a_ = summary.get(f'{name}|{kind}|{lo}')
                b_ = summary.get(f'{name}|{kind}|{hi}')
                if not a_ or not b_:
                    continue
                da = 1000 * (b_['params'].get('sigma_xy_um', float('nan'))
                             - a_['params'].get('sigma_xy_um', float('nan')))
                flip = '' if a_['best'] == b_['best'] else \
                    f'   FAMILY FLIPPED {a_["best"]} -> {b_["best"]}'
                print(f'  {name:<10} {kind:<9} n={lo}->{hi}: '
                      f'{da:+7.1f} nm{flip}')

    os.makedirs(a.out, exist_ok=True)
    # The filename carries the RUN's shape. A fixed name meant a two-family
    # convergence run silently destroyed the four-family comparison that
    # justified dropping two of them -- the evidence for a decision, wiped
    # by the run the decision enabled.
    tag = (f'{len(families)}fam_n{"-".join(str(c) for c in crop_counts)}'
           f'_r{radius[0]:g}')
    path = os.path.join(a.out, f'psf_survey_{tag}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
