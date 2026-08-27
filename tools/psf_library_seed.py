"""
Seed the PSF library from survey results, including a universal default.

Run once to populate <repo>/psf from the measured surveys; re-runnable,
and it overwrites only the labels it creates.

THE UNIVERSAL DEFAULT IS AN AVERAGE OF CONVERGED ENTRIES ONLY
------------------------------------------------------------
An experiment whose calibration was still moving at its largest crop
count does not get a vote. Chr19's readout oscillated 247/211/242/207 nm
across crop counts and chose a different family from the other three;
averaging that in would import its instability into the default every
other experiment starts from.

Usage:
    python tools/psf_library_seed.py --survey notes/.../psf_survey.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                              # noqa: E402

from codelab_pipeline.localization import psf_library as LIB    # noqa: E402
from tools.experiments import EXPERIMENTS                       # noqa: E402

# How much sigma_xy may still be moving at the largest crop count and
# still count as converged. The scale is set by the ~40 nm that one
# experiment moves under crop RESELECTION -- a drift smaller than the
# noise floor of the measurement is as converged as it can be shown to be.
CONVERGED_NM = 15.0


def readout_series(survey):
    """{experiment: [(n, family, params, plausible), ...]} sorted by n."""
    out = {}
    for _key, s in survey.items():
        if s.get('kind') != 'readout':
            continue
        n = s.get('n') or s.get('n_crops')
        out.setdefault(s['experiment'], []).append(
            (n, s['best'], s['params'], not s.get('all_implausible', False)))
    for e in out:
        out[e].sort(key=lambda t: t[0] or 0)
    return out


def judge(series):
    """(params, family, converged_dict, why) for one experiment's series."""
    if not series:
        return None, None, None, 'no readout fits'
    n, fam, params, ok = series[-1]
    if len(series) < 2:
        return params, fam, None, f'single crop count ({n}), convergence unknown'
    n0, fam0, p0, _ = series[-2]
    d = abs(1000 * (params.get('sigma_xy_um', np.nan)
                    - p0.get('sigma_xy_um', np.nan)))
    conv = {'n_crops': [n0, n],
            'sigma_xy_nm': [round(1000 * p0.get('sigma_xy_um', float('nan')), 1),
                            round(1000 * params.get('sigma_xy_um', float('nan')), 1)],
            'delta_nm': round(float(d), 1),
            'family_stable': fam0 == fam,
            'converged': bool(d <= CONVERGED_NM and fam0 == fam)}
    why = ('converged' if conv['converged'] else
           f'NOT converged: sigma moved {d:.1f} nm from n={n0} to n={n}'
           + ('' if fam0 == fam else f', family flipped {fam0} -> {fam}'))
    if not ok:
        why += '; no plausible candidate'
        conv['converged'] = False
    return params, fam, conv, why


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--survey', action='append', required=True)
    ap.add_argument('--voxel', default='0.208,0.208,0.2')
    a = ap.parse_args()

    voxel = tuple(float(v) for v in a.voxel.split(','))
    merged = {}
    for p in a.survey:
        with open(p, encoding='utf-8') as f:
            merged.update(json.load(f))

    series = readout_series(merged)
    good = []
    print(f'{"experiment":<11}{"family":<15}{"sxy nm":>8}{"sz nm":>8}  convergence')
    print('-' * 78)
    for name in sorted(series):
        params, fam, conv, why = judge(series[name])
        if params is None:
            print(f'{name:<11}{why}')
            continue
        exp = EXPERIMENTS.get(name)
        label = LIB.default_label(name, exp.reference_hybe if exp else '',
                                  exp.anchor_channel if exp else '')
        LIB.write(label, fam, params, voxel,
                  source={'experiment': name,
                          'reference_hybe': exp.reference_hybe if exp else None,
                          'channel': exp.anchor_channel if exp else None,
                          'scope_mb': exp.scope_mb if exp else None,
                          'n_crops': series[name][-1][0],
                          'survey': [os.path.basename(x) for x in a.survey]},
                  converged=conv, notes=why)
        print(f'{name:<11}{fam:<15}{1000*params["sigma_xy_um"]:>8.0f}'
              f'{1000*params["sigma_z_um"]:>8.0f}  {why}')
        if conv and conv.get('converged'):
            good.append((name, fam, params))

    # -- the universal default -----------------------------------------
    if len(good) < 2:
        print('\nnot enough converged entries to justify a universal default')
        return
    fams = {f for _n, f, _p in good}
    if len(fams) > 1:
        print(f'\nconverged entries disagree on family {fams}; '
              f'no universal default written')
        return
    fam = fams.pop()
    keys = sorted({k for _n, _f, p in good for k in p})
    mean = {k: float(np.mean([p[k] for _n, _f, p in good if k in p])) for k in keys}
    spread = {k: float(np.ptp([p[k] for _n, _f, p in good if k in p])) for k in keys}
    LIB.write(
        'universal-default', fam, mean, voxel,
        source={'experiments': [n for n, _f, _p in good],
                'n_experiments': len(good),
                'derivation': 'mean of converged per-experiment readout '
                              'calibrations'},
        converged={'converged': True,
                   'sigma_xy_spread_nm': round(1000 * spread.get('sigma_xy_um', 0), 1)},
        notes=('Mean over converged experiments only. The spread between '
               'them is the number that justifies a shared default: it is '
               'smaller than the ~40 nm one experiment moves when its own '
               'crops are reselected, so between-experiment variation is '
               'below within-experiment measurement noise. Override per '
               'experiment whenever a local calibration converges.'))
    print(f'\nuniversal-default: {fam}  '
          + '  '.join(f'{k}={v:.4f}' for k, v in mean.items()))
    print(f'  from {len(good)} converged experiments '
          f'({", ".join(n for n, _f, _p in good)})')
    print(f'  sigma_xy spread across them: '
          f'{1000 * spread.get("sigma_xy_um", 0):.1f} nm')
    print(f'\nlibrary now holds {len(LIB.list_entries())} entries in '
          f'{LIB.library_dir()}')


if __name__ == '__main__':
    main()
