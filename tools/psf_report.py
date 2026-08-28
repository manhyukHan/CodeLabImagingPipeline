"""
Turn psf_survey JSON into a table and a picture.

A survey answers three questions and each wants a different view:

  which FAMILY describes this channel   -> per-family scores, side by side
  is the shape UNIVERSAL                -> sigma across experiments, one
                                           family throughout
  is the shape CONVERGED                -> sigma against crop count

The long-format table carries every row so nothing has to be re-run to
ask a new question of it, and the figure shows the two comparisons that
are hard to read as numbers.

ONE FAMILY AT A TIME, ALWAYS
----------------------------
Comparing sigma across experiments while letting each pick its own
best-scoring family is not a comparison. It produced a spurious
"fiducial follows N^(1/3) polymer_adj scaling" here: HoxA contributed its
gaussian_halo core (194 nm) while the others contributed their gaussian
(332-399 nm), and HoxA's own gaussian is 273 nm. With one family
throughout the exponent fell from +0.30 to +0.16 and the interpretation
went with it. Every cross-experiment plot in this file therefore fixes
the family and says which one.

Usage:
    python tools/psf_report.py notes/.../psf_survey_4fam_n40.json
    python tools/psf_report.py A.json B.json --out notes/...
"""
import argparse
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_OUT = os.path.join('notes', 'chromatin_tracing_optimization')
FAMILY_COLOUR = {'gaussian': '#1f77b4', 'gaussian_halo': '#9467bd',
                 'lorentzian': '#d62728', 'moffat': '#2ca02c'}


def rows_from(paths):
    """Long format: one row per (experiment, kind, n_crops, family)."""
    out = []
    for p in paths:
        with open(p, encoding='utf-8') as f:
            d = json.load(f)
        for key, s in d.items():
            n = s.get('n', s.get('n_crops'))
            for family, pf in s.get('per_family', {}).items():
                pr = pf.get('params', {})
                out.append({
                    'source': os.path.basename(p),
                    'experiment': s['experiment'],
                    'scope_mb': s.get('scope_mb'),
                    'kind': s['kind'],
                    'reference_hybe': s.get('reference_hybe'),
                    'n_crops': n,
                    # The lateral fit window. Belongs in the table because
                    # the fiducial's sigma is not independent of it --
                    # measured sigma ~ r^0.5, so a fiducial sigma quoted
                    # without its window is not a number about the object.
                    'fit_radius_um': (s.get('fit_radius_um') or [None])[0],
                    'family': family,
                    'rss_per_voxel': round(pf['score'], 2),
                    'sigma_xy_nm': round(1000 * pr['sigma_xy_um'], 1)
                        if 'sigma_xy_um' in pr else None,
                    'sigma_z_nm': round(1000 * pr['sigma_z_um'], 1)
                        if 'sigma_z_um' in pr else None,
                    'halo_frac': round(pr['halo_frac'], 3)
                        if 'halo_frac' in pr else None,
                    'halo_scale': round(pr['halo_scale'], 3)
                        if 'halo_scale' in pr else None,
                    'beta': round(pr['beta'], 3) if 'beta' in pr else None,
                    'plausible': bool(pf.get('plausible')),
                    'is_best': family == s.get('best'),
                    'warnings': '; '.join(pf.get('warnings') or ()),
                })
    out.sort(key=lambda r: (r['kind'], r['scope_mb'] or 0, r['experiment'],
                            r['n_crops'] or 0, r['family']))
    return out


def scaling_exponent(points):
    """Least-squares slope of log(sigma) on log(scope). None if degenerate."""
    pts = [(m, s) for m, s in points if m and s and m > 0 and s > 0]
    if len(pts) < 3 or len({m for m, _ in pts}) < 2:
        return None
    xs = [math.log(m) for m, _ in pts]
    ys = [math.log(s) for _, s in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    return None if denom == 0 else sum((x - mx) * (y - my)
                                       for x, y in zip(xs, ys)) / denom


def write_table(rows, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    fields = list(rows[0])
    csv_path = os.path.join(out_dir, f'{stem}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    written = [csv_path]

    # utf-8-sig above is for Excel: without the BOM it guesses the codepage
    # and mangles anything non-ASCII on a Korean-locale Windows.
    try:
        import pandas as pd
        xl = os.path.join(out_dir, f'{stem}.xlsx')
        df = pd.DataFrame(rows)
        with pd.ExcelWriter(xl, engine='openpyxl') as xw:
            df.to_excel(xw, sheet_name='all_fits', index=False)
            best = df[df['is_best']]
            if not best.empty:
                best.to_excel(xw, sheet_name='winners', index=False)
            for kind in sorted(df['kind'].unique()):
                sub = df[df['kind'] == kind]
                sub.to_excel(xw, sheet_name=str(kind)[:28], index=False)
        written.append(xl)
    except Exception as e:
        print(f'  (no .xlsx: {type(e).__name__}: {e})')
    return written


def render(rows, out_dir, stem):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Only the channels actually present. A fiducial-only sweep (the fit
    # window test) would otherwise build an empty readout row and die in
    # the legend, losing the figure the run existed to produce.
    kinds = [k for k in ('fiducial', 'readout')
             if any(r['kind'] == k for r in rows)]
    counts = sorted({r['n_crops'] for r in rows if r['n_crops']})
    biggest = max(counts) if counts else None
    has_series = len(counts) > 1

    ncols = 3 if has_series else 2
    nrows = len(kinds)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6.0 * ncols, 4.75 * nrows),
                             squeeze=False, constrained_layout=True)
    fig.suptitle('PSF calibration across experiments\n'
                 'hollow marker = fit rejected as physically implausible '
                 '(parameter on a bound, or sigma below the optical limit)',
                 fontsize=12)

    for row, kind in enumerate(kinds):
        # -- column 0: score per family, grouped by experiment -----------
        ax = axes[row][0]
        sel = [r for r in rows if r['kind'] == kind and r['n_crops'] == biggest]
        exps = sorted({r['experiment'] for r in sel},
                      key=lambda e: next(x['scope_mb'] or 0
                                         for x in sel if x['experiment'] == e))
        fams = sorted({r['family'] for r in sel})
        width = 0.8 / max(len(fams), 1)
        for j, fam in enumerate(fams):
            xs, ys, edge = [], [], []
            for i, e in enumerate(exps):
                m = [r for r in sel if r['experiment'] == e and r['family'] == fam]
                if not m:
                    continue
                xs.append(i + j * width - 0.4 + width / 2)
                ys.append(m[0]['rss_per_voxel'])
                edge.append(m[0]['plausible'])
            ax.bar(xs, ys, width * 0.92, label=fam,
                   color=[FAMILY_COLOUR.get(fam, '#888') if ok else 'white'
                          for ok in edge],
                   edgecolor=FAMILY_COLOUR.get(fam, '#888'), linewidth=1.4)
        ax.set_xticks(range(len(exps)))
        ax.set_xticklabels([f'{e}\n{next(x["scope_mb"] for x in sel if x["experiment"] == e)} Mb'
                            for e in exps], fontsize=8)
        ax.set_ylabel('rss / voxel   (lower is better)')
        ax.set_title(f'{kind}: which family fits  (n={biggest})', fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=.3, axis='y')
        # rss/voxel scales with absolute intensity, so bars are only
        # comparable WITHIN an experiment -- a log axis stops one bright
        # experiment flattening every other group into the floor.
        ax.set_yscale('log')

        # -- column 1: sigma vs scope, ONE family at a time ---------------
        ax = axes[row][1]
        for fam in fams:
            pts = [(r['scope_mb'], r['sigma_xy_nm'], r['plausible'])
                   for r in sel if r['family'] == fam and r['sigma_xy_nm']]
            if not pts:
                continue
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], '-',
                    color=FAMILY_COLOUR.get(fam, '#888'), lw=1.5, alpha=.8)
            for mb, s, ok in pts:
                ax.plot(mb, s, 'o', ms=8, color=FAMILY_COLOUR.get(fam, '#888')
                        if ok else 'white', mec=FAMILY_COLOUR.get(fam, '#888'),
                        mew=1.6)
            expo = scaling_exponent([(p[0], p[1]) for p in pts if p[2]])
            lbl = f'{fam}' + (f'  (exponent {expo:+.2f})' if expo is not None else '')
            ax.plot([], [], 'o-', color=FAMILY_COLOUR.get(fam, '#888'), label=lbl)
        ax.set_xscale('log')
        ax.set_xlabel('genomic scope (Mb, approximate)')
        ax.set_ylabel('sigma_xy (nm)')
        ax.set_title(f'{kind}: does the shape track scope?', fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=.3, which='both')

        # -- column 2: convergence in crop count -------------------------
        if has_series:
            ax = axes[row][2]
            for e in exps:
                for fam in fams:
                    pts = [(r['n_crops'], r['sigma_xy_nm'], r['plausible'])
                           for r in rows
                           if r['kind'] == kind and r['experiment'] == e
                           and r['family'] == fam and r['sigma_xy_nm']]
                    if len(pts) < 2:
                        continue
                    pts.sort()
                    style = '-' if fam == 'gaussian_halo' else '--'
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], style,
                            marker='o', ms=4, lw=1.3,
                            color=FAMILY_COLOUR.get(fam, '#888'),
                            alpha=.85, label=f'{e} {fam}')
            ax.set_xscale('log')
            ax.set_xlabel('crops used in the calibration')
            ax.set_ylabel('sigma_xy (nm)')
            ax.set_title(f'{kind}: has it converged?', fontsize=10)
            ax.legend(fontsize=6, ncol=2)
            ax.grid(alpha=.3, which='both')

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{stem}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def render_truncation(rows, out_dir, stem):
    """sigma against the FIT WINDOW -- the plot that says whether a width
    is a property of the object or of the measurement.

    A genuine Gaussian gives a flat line: widening the window adds only
    baseline and the fitted sigma does not move. A line that keeps rising
    says the profile has tails the model cannot represent, and the fit is
    reporting the window instead. Measured here: sigma ~ r^0.5 on every
    fiducial, so no fiducial sigma means anything without its window.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    radii = sorted({r['fit_radius_um'] for r in rows if r['fit_radius_um']})
    if len(radii) < 2:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    fig.suptitle('Is the fitted width a property of the OBJECT or of the WINDOW?\n'
                 'flat = a real Gaussian width;  rising = the model cannot '
                 'describe the tails, so the fit reports the window',
                 fontsize=12)
    exps = sorted({r['experiment'] for r in rows},
                  key=lambda e: next(x['scope_mb'] or 0
                                     for x in rows if x['experiment'] == e))
    for ax, logy in ((axes[0], False), (axes[1], True)):
        for e in exps:
            pts = sorted((r['fit_radius_um'], r['sigma_xy_nm'])
                         for r in rows if r['experiment'] == e
                         and r['fit_radius_um'] and r['sigma_xy_nm'])
            if len(pts) < 2:
                continue
            mb = next(x['scope_mb'] for x in rows if x['experiment'] == e)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], 'o-',
                    lw=1.8, ms=6, label=f'{e}  ({mb} Mb)')
        if logy:
            ax.set_xscale('log'); ax.set_yscale('log')
            # slope 0.5 reference, anchored low so it does not overlap data
            xs = [min(radii), max(radii)]
            y0 = 150.0
            ax.plot(xs, [y0 * (x / xs[0]) ** 0.5 for x in xs], 'k--', lw=1.2,
                    label='slope 0.5 (window-limited)')
            ax.plot(xs, [y0, y0], 'k:', lw=1.2, label='slope 0 (true Gaussian)')
        ax.set_xlabel('lateral fit radius (um)')
        ax.set_ylabel('fitted sigma_xy (nm)')
        ax.grid(alpha=.3, which='both')
        ax.legend(fontsize=8)
    path = os.path.join(out_dir, f'{stem}_truncation.png')
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('json', nargs='+')
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--stem', default=None)
    a = ap.parse_args()

    rows = rows_from(a.json)
    if not rows:
        sys.exit('no rows in those files')
    stem = a.stem or os.path.splitext(os.path.basename(a.json[0]))[0] + '_report'

    for p in write_table(rows, a.out, stem):
        print(f'wrote {p}   ({len(rows)} rows)')
    print(f'wrote {render(rows, a.out, stem)}')
    trunc = render_truncation(rows, a.out, stem)
    if trunc:
        print(f'wrote {trunc}')

    print('\nscaling exponent of sigma_xy on genomic scope, ONE family throughout')
    print('(only plausible fits contribute; mixing families invents trends):')
    counts = sorted({r['n_crops'] for r in rows if r['n_crops']})
    for kind in ('fiducial', 'readout'):
        for fam in sorted({r['family'] for r in rows}):
            sel = [r for r in rows if r['kind'] == kind and r['family'] == fam
                   and r['n_crops'] == max(counts) and r['plausible']]
            e = scaling_exponent([(r['scope_mb'], r['sigma_xy_nm']) for r in sel])
            if e is None:
                print(f'  {kind:<9} {fam:<15} -- too few plausible fits '
                      f'({len(sel)}) to fit an exponent')
            else:
                print(f'  {kind:<9} {fam:<15} {e:+.2f}   from {len(sel)} experiments')


if __name__ == '__main__':
    main()
