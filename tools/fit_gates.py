"""
Re-derive the acceptance gates for the v2 fit, against real data.

WHY
---
v1's three gates -- position CI < max_uncert, raw-peak/background >=
min_hb_ratio, amplitude/raw-peak >= min_ah_ratio -- are inherited from
ChrTracer3's filterSpot with the constants that came with it. Every one
of them is computed from a quantity whose MEANING changed when the
background stopped being a single constant fitted over a whole
110-plane column:

  * `offset` used to converge to roughly the global dark floor, so
    h/offset was comfortably large. With a local background over a
    restricted volume it becomes the real background AT the spot, and
    the same photons now score far lower against the same 1.2.
  * `amp` now means height above the LOCAL background, not above the
    global floor, so amp/h measures something different too.
  * the CI comes from a Jacobian covariance that assumes an interior
    optimum -- for a parameter sitting on a bound it does not describe
    the estimator at all, and most v1 fits sit on a bound.

So the constants cannot simply be carried over, and "acceptance fell"
is not evidence of anything until they are re-derived.

HOW
---
The replicate pairs are the criterion. A locus imaged twice in one
allele must land in the same place, so for any candidate gate we can
measure BOTH:

    coverage  -- how many same-locus pairs survive the gate
    accuracy  -- the median distance between the two rounds of a pair

A gate that is too loose admits junk and accuracy degrades; too tight
and coverage collapses. That is a trade-off curve measured on real
spots, not a constant.

Fitting happens ONCE, ungated (apply_gates=False), and every candidate
threshold is applied to the cached results afterwards -- otherwise a
sweep of N thresholds costs N full passes over the bench.

Usage:
    python tools/fit_gates.py bench.npz [--kind readout] [--out DIR]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline.localization import fit3d_um as U      # noqa: E402

DEFAULT_OUT = os.path.join('notes', 'chromatin_tracing_optimization')
FIT_RADIUS = (1.0, 1.0, 3.0)     # um; the volume the background term is honest over
BACKGROUND = 'linear'


def fit_all(bench, kind, limit=None):
    """Every crop of one kind, fitted ONCE with gating switched off."""
    keys = [k for k in bench.files if k.endswith('|' + kind)]
    if limit:
        keys = keys[:limit]
    out, t0 = {}, time.perf_counter()
    for i, k in enumerate(keys):
        c = bench[k].astype(float)
        cy, cx = (c.shape[0] - 1) / 2.0, (c.shape[1] - 1) / 2.0
        iz = float(np.unravel_index(int(np.nanargmax(c)), c.shape)[2])
        f = U.fit_gaussian_3d_um(c, cy, cx, iz, max_sigma_xy_um=3.0,
                                 max_sigma_z_um=6.0, fit_radius_um=FIT_RADIUS,
                                 background=BACKGROUND, apply_gates=False)
        if f is not None:
            out[k] = f
        if (i + 1) % 200 == 0:
            print(f'   fitted {i + 1}/{len(keys)}...', flush=True)
    print(f'   {len(out)}/{len(keys)} converged in {time.perf_counter() - t0:.0f}s')
    return out, keys


def pair_stats(fits, bench, kind, keep):
    """(n_pairs, median_xy_um, p90) over same-locus pairs both of whose
    rounds passed `keep`."""
    meta = bench['__meta__']
    pairs = [tuple(s.split('|')) for s in bench['__pairs__']]
    d = []
    for aid, *_r in meta:
        for a, b, _rid in pairs:
            ka, kb = f'a{aid}|{a}|{kind}', f'a{aid}|{b}|{kind}'
            fa, fb = fits.get(ka), fits.get(kb)
            if fa is None or fb is None or not keep(fa) or not keep(fb):
                continue
            d.append(float(np.hypot(fa.y_um - fb.y_um, fa.x_um - fb.x_um)))
    if not d:
        return 0, float('nan'), float('nan')
    d = np.array(d)
    return len(d), float(np.median(d)), float(np.percentile(d, 90))


def sweep(fits, bench, kind):
    """One gate at a time, the other two wide open, so each curve shows
    that gate's own effect rather than the product of three."""
    grids = {
        'min_hb_ratio': np.round(np.arange(1.00, 2.61, 0.10), 2),
        'min_ah_ratio': np.round(np.arange(0.00, 1.01, 0.05), 2),
        'max_uncert_um': np.round(np.arange(0.02, 0.62, 0.02), 3),
    }
    keeps = {
        'min_hb_ratio': lambda f, t: f.peak_bg_ratio >= t,
        'min_ah_ratio': lambda f, t: f.amp_h_ratio >= t,
        'max_uncert_um': lambda f, t: (2 * f.ci_y_um < t and 2 * f.ci_x_um < t
                                       and 2 * f.ci_z_um < 2 * t),
    }
    curves = {}
    for name, grid in grids.items():
        rows = []
        for t in grid:
            n, med, p90 = pair_stats(fits, bench, kind,
                                     lambda f, t=t: keeps[name](f, t))
            kept = sum(1 for f in fits.values() if keeps[name](f, t))
            rows.append((float(t), kept, n, med, p90))
        curves[name] = rows
    return curves


def render(curves, kind, fits, out_dir, v1_defaults):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    names = list(curves)
    fig, axes = plt.subplots(2, len(names), figsize=(5.2 * len(names), 7.4),
                             constrained_layout=True)
    fig.suptitle(f'Acceptance gates re-derived on real spots -- {kind} channel\n'
                 f'background={BACKGROUND}, fit radius={FIT_RADIUS} um, '
                 f'{len(fits)} converged fits',
                 fontsize=12)
    for j, name in enumerate(names):
        rows = np.array([(t, kept, n, med, p90) for t, kept, n, med, p90 in curves[name]],
                        dtype=float)
        t, kept, npair, med, p90 = rows.T
        ax = axes[0][j]
        ax.plot(t, npair, 'o-', color='#1f77b4', label='same-locus pairs scored')
        ax.set_ylabel('pairs scored (coverage)', color='#1f77b4')
        ax.tick_params(axis='y', labelcolor='#1f77b4')
        ax2 = ax.twinx()
        ax2.plot(t, kept, 's--', color='#999999', ms=3, label='fits kept')
        ax2.set_ylabel('fits kept', color='#999999')
        ax.set_title(name)
        if name in v1_defaults:
            ax.axvline(v1_defaults[name], color='crimson', ls=':', lw=2)
            ax.annotate(f"v1 default\n{v1_defaults[name]}",
                        xy=(v1_defaults[name], ax.get_ylim()[1]),
                        xytext=(4, -28), textcoords='offset points',
                        color='crimson', fontsize=8)
        ax.grid(alpha=.3)

        ax = axes[1][j]
        ax.plot(t, med, 'o-', color='#d62728', label='median')
        ax.plot(t, p90, '^--', color='#ff9896', ms=4, label='p90')
        ax.set_xlabel(name)
        ax.set_ylabel('same-locus pair distance (um)')
        if name in v1_defaults:
            ax.axvline(v1_defaults[name], color='crimson', ls=':', lw=2)
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    path = os.path.join(out_dir, f'gate_sweep_{kind}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def render_distributions(fits, kind, out_dir, v1_defaults):
    """What the gate quantities actually look like -- a threshold can
    only be judged against the distribution it is cutting."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    series = {
        'min_hb_ratio': [f.peak_bg_ratio for f in fits.values()],
        'min_ah_ratio': [f.amp_h_ratio for f in fits.values()],
        'max_uncert_um': [2 * max(f.ci_y_um, f.ci_x_um) for f in fits.values()],
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    fig.suptitle(f'Distribution of each gate quantity -- {kind} channel '
                 f'({len(fits)} fits, background={BACKGROUND})')
    for ax, (name, vals) in zip(axes, series.items()):
        vals = np.array([v for v in vals if np.isfinite(v)])
        hi = np.percentile(vals, 99) if vals.size else 1
        ax.hist(vals, bins=60, range=(0, hi), color='#4c72b0')
        if name in v1_defaults:
            ax.axvline(v1_defaults[name], color='crimson', ls=':', lw=2,
                       label=f'v1 default {v1_defaults[name]}')
            frac = 100.0 * np.mean(vals < v1_defaults[name]) if name != 'max_uncert_um' \
                else 100.0 * np.mean(vals >= v1_defaults[name])
            ax.set_title(f'{name}\nv1 default rejects {frac:.0f}%')
            ax.legend(fontsize=8)
        ax.set_xlabel(name)
        ax.grid(alpha=.3)
    path = os.path.join(out_dir, f'gate_distributions_{kind}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bench')
    ap.add_argument('--kind', action='append', default=None,
                    choices=['fiducial', 'readout'])
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()

    v1_defaults = {'min_hb_ratio': 1.2, 'min_ah_ratio': 0.25, 'max_uncert_um': 0.416}
    bench = np.load(a.bench, allow_pickle=False)
    summary = {}
    for kind in (a.kind or ['fiducial', 'readout']):
        print(f'--- {kind} ---')
        fits, keys = fit_all(bench, kind, a.limit)
        if not fits:
            print('   nothing converged; skipping')
            continue
        curves = sweep(fits, bench, kind)
        fig1 = render(curves, kind, fits, a.out, v1_defaults)
        fig2 = render_distributions(fits, kind, a.out, v1_defaults)
        print(f'   wrote {fig1}')
        print(f'   wrote {fig2}')

        print(f'\n   {"gate":<16}{"threshold":>10}{"fits kept":>11}'
              f'{"pairs":>8}{"median um":>11}{"p90":>8}')
        print('   ' + '-' * 64)
        for name, rows in curves.items():
            for t, kept, n, med, p90 in rows:
                near_default = abs(t - v1_defaults.get(name, -99)) < 1e-9
                if near_default or (n and n == max(r[2] for r in rows)):
                    tag = '  <- v1 default' if near_default else '  <- max coverage'
                    print(f'   {name:<16}{t:>10.3f}{kept:>11}{n:>8}'
                          f'{med:>11.4f}{p90:>8.4f}{tag}')
        summary[kind] = {'n_fits': len(fits), 'curves': curves}
        print()

    path = os.path.join(a.out, 'gate_sweep.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=1, default=float)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
