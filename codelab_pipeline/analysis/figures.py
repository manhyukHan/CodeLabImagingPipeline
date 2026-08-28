"""
The visualizer pad: matplotlib figures over toolbox outputs.

Still headless -- Agg figures returned to the caller, who decides
whether they land in a Qt canvas (the app) or a PNG (a notebook, these
validation runs). Conventions carried from the reference work: distance
maps in seismic_r (RED = CLOSE), color limits at the 2/98% quantiles of
finite values, NaN drawn dark -- a pixel with too few observations
LOOKS different, never silently averaged; celltype '' plots as
'Unassigned' in grey and is never dropped; every figure states its gate
and its N in the title, because a map without its population is not a
result.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib import cm                                      # noqa: E402

from codelab_pipeline.analysis import ensemble as ens          # noqa: E402

UNASSIGNED = 'Unassigned'


def _dmap_ax(ax, m, title, vmin=None, vmax=None):
    finite = m[np.isfinite(m)]
    if vmin is None:
        vmin = np.quantile(finite, 0.02) if len(finite) else 0
    if vmax is None:
        vmax = np.quantile(finite, 0.98) if len(finite) else 1
    cmap = cm.seismic_r.copy()
    cmap.set_bad('0.15')
    im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def fig_ensemble(dmaps, mask=None, title='ensemble', reducer='median',
                 min_n=1, group_masks=None):
    """One ensemble map, optionally decomposed by named group masks
    (the FLAG axis: groups split the gated stack, they never re-gate)."""
    groups = [('all', mask)] if not group_masks else list(group_masks.items())
    n = len(groups)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.4), squeeze=False)
    ims = []
    for ax, (name, gmask) in zip(axes[0], groups):
        m, counts = ens.ensemble_map(dmaps, gmask, reducer, min_n)
        n_al = int(gmask.sum()) if gmask is not None else len(dmaps)
        ims.append(_dmap_ax(ax, m, f'{name}  (n={n_al})'))
    vmin = min(i.get_clim()[0] for i in ims)
    vmax = max(i.get_clim()[1] for i in ims)
    for i in ims:
        i.set_clim(vmin, vmax)
    fig.colorbar(ims[-1], ax=axes[0], shrink=0.8, label='distance (um)')
    fig.suptitle(f'{title}  [{reducer}, min_n={min_n}]', fontsize=11)
    return fig


def fig_fov_consistency(dmaps, fovs, mask=None, min_n=1, group_masks=None):
    """FOV-level QC: per-FOV ensembles, SCC matrix, and the MSD TEST.

    group_masks decomposes the whole view -- celltype-decompose means
    THREE FOV matrices, not one, per the flag principle: flags multiply
    figures. One ROW per group: its per-FOV maps, its SCC matrix, and
    the in-FOV vs cross-FOV allele-map MSD distributions with the Welch
    verdict. The MSD test is the quantitative instrument: SCC compares
    two grainy ensembles (one number per FOV pair) while the MSD
    histogram carries thousands of allele pairs per class -- p-values
    are ranking scores (pairs share alleles; see fov_msd_test).
    """
    groups = [('all', mask)] if not group_masks else list(group_masks.items())
    all_fovs = sorted(set(int(f) for f in np.asarray(fovs)))
    k = len(all_fovs)
    ncols = k + 2
    fig, axes = plt.subplots(len(groups), ncols,
                             figsize=(3.1 * ncols, 3.5 * len(groups)),
                             squeeze=False)
    for row, (name, gmask) in enumerate(groups):
        res = ens.fov_consistency(dmaps, fovs, gmask, min_n=min_n)
        for col, f in enumerate(all_fovs):
            ax = axes[row][col]
            if f in res['maps']:
                _dmap_ax(ax, res['maps'][f], f'{name}  FOV{f:03d}')
            else:
                ax.set_axis_off()
        ax = axes[row][k]
        kk = len(res['fovs'])
        im = ax.imshow(res['scc'], cmap='viridis', vmin=0, vmax=1)
        ax.set_xticks(range(kk))
        ax.set_yticks(range(kk))
        ax.set_xticklabels([str(f) for f in res['fovs']], fontsize=7)
        ax.set_yticklabels([str(f) for f in res['fovs']], fontsize=7)
        for i in range(kk):
            for j in range(kk):
                ax.text(j, i, f'{res["scc"][i, j]:.2f}', ha='center',
                        va='center', fontsize=6,
                        color='white' if res['scc'][i, j] < 0.6 else 'black')
        ax.set_title(f'{name}: SCC', fontsize=9)

        ax = axes[row][k + 1]
        t = ens.fov_msd_test(dmaps, fovs, gmask)
        if len(t['msd_in']) and len(t['msd_cross']):
            edges = np.histogram_bin_edges(
                np.concatenate([t['msd_in'], t['msd_cross']]), bins=40)
            ax.hist(t['msd_cross'], bins=edges, density=True, alpha=0.5,
                    color='0.6', label=f"cross-FOV (n={len(t['msd_cross'])})")
            ax.hist(t['msd_in'], bins=edges, density=True, alpha=0.6,
                    color='crimson', label=f"in-FOV (n={len(t['msd_in'])})")
            ax.legend(fontsize=7)
            per = '  '.join(f"F{d['fov']}:{d['signed_neglog10p']:+.1f}"
                            for d in t['per_fov'])
            ax.set_title(f"{name}: MSD  -log10 p = {t['neglog10p']:.1f}\n"
                         f'per-FOV signed (+ = deviant)  {per}', fontsize=8)
        else:
            ax.set_title(f'{name}: too few pairs', fontsize=9)
        ax.set_xlabel('pairwise map MSD (um^2)', fontsize=8)
    fig.suptitle('FOV-level consistency', fontsize=12)
    return fig


def fig_allele_difference(dmaps, fovs, cells, mask=None):
    """Within-cell allele-pair dissimilarity against the cross-cell null."""
    res = ens.allele_difference(dmaps, fovs, cells, mask)
    fig, ax = plt.subplots(figsize=(5.2, 4))
    w = res['within'][np.isfinite(res['within'])]
    nl = res['null'][np.isfinite(res['null'])]
    if len(nl):
        ax.hist(nl, bins=40, density=True, alpha=0.5, color='0.6',
                label=f'cross-cell null (n={len(nl)})')
    if len(w):
        ax.hist(w, bins=40, density=True, alpha=0.7, color='crimson',
                label=f'within-cell pairs (n={len(w)})')
    ax.set_xlabel('mean |d1 - d2| over shared bins (um)')
    ax.set_ylabel('density')
    ax.legend(fontsize=8)
    ax.set_title(f'allele differences -- {res["n_multi_allelic"]} '
                 f'multi-allelic cells', fontsize=10)
    return fig


def fig_polymer_qc(table, dmaps, thresholds, qc_result=None):
    """The polymeric-QC view: every gated quantity as a histogram with
    its threshold drawn ON the distribution it was derived from, plus
    efficacy per bin and completeness per allele -- the tracked, visible
    QC stage the gate builder sits behind.
    """
    fig, axes = plt.subplots(1, 5, figsize=(21, 3.6))
    amp = table['amp']
    finite_amp = amp[np.isfinite(amp)]
    ax = axes[0]
    if len(finite_amp):
        ax.hist(finite_amp, bins=60, color='0.5')
    ax.axvline(thresholds['min_brightness'], color='crimson', ls='--')
    ax.axvline(thresholds['max_brightness'], color='crimson', ls='--')
    ax.set_title('bin amplitude + brightness gates', fontsize=9)
    ax = axes[1]
    if len(dmaps):
        nb = np.concatenate([np.diagonal(d, 1) for d in dmaps])
        nb = nb[np.isfinite(nb)]
        if len(nb):
            ax.hist(nb, bins=60, color='0.5')
    ax.axvline(thresholds['max_jump_um'], color='crimson', ls='--')
    ax.set_title('neighbor distance (um) + max jump', fontsize=9)
    ax = axes[2]
    if len(dmaps):
        med = np.nanmedian(dmaps, axis=1)
        med = med[np.isfinite(med)]
        if len(med):
            ax.hist(med, bins=60, color='0.5')
    ax.axvline(thresholds['max_dist_um'], color='crimson', ls='--')
    ax.set_title('per-bin median distance (um) + max', fontsize=9)
    ax = axes[3]
    from codelab_pipeline.analysis import polymer as _P
    pos = qc_result['pos_um'] if qc_result is not None else table['pos_um']
    eff = _P.efficacy(pos)
    ax.bar(np.arange(len(eff)), eff, color='steelblue')
    ax.set_ylim(0, 1)
    ax.set_title('efficacy per bin (after QC)' if qc_result is not None
                 else 'efficacy per bin', fontsize=9)
    ax = axes[4]
    comp = _P.completeness(pos)
    if len(comp):
        ax.hist(comp, bins=min(40, max(5, int(comp.max()) + 1)),
                color='steelblue')
    ax.set_title('completeness per allele', fontsize=9)
    if qc_result is not None:
        kept = int(qc_result['kept'].sum())
        fig.suptitle(f'polymer QC -- {kept}/{len(table["amp"])} alleles kept, '
                     f'{int(qc_result["bads"].sum())} bins removed',
                     fontsize=11)
    else:
        fig.suptitle('polymer QC (preview -- not applied)', fontsize=11)
    return fig


def fig_expression_hist(table, source, metric='n_spots', bins=30,
                        per_celltype=True, picked_range=None):
    """Expression histogram, celltype-decomposed; the range picker's view.

    picked_range=(lo, hi) draws the chosen gate band -- what the panel
    shows while the user drags the ExpressionRange condition."""
    m, h, ch = source
    rows = table[(table['modality'] == m) & (table['hybe'] == h)
                 & (table['channel'] == int(ch))]
    fig, ax = plt.subplots(figsize=(5.6, 4))
    vals_all = rows[metric].to_numpy(dtype=float)
    finite = vals_all[np.isfinite(vals_all)]
    if len(finite) == 0:
        ax.set_title(f'{source}: no finite {metric}')
        return fig
    edges = np.histogram_bin_edges(finite, bins=bins)
    if per_celltype:
        for ct, g in rows.groupby('celltype'):
            name = ct if ct else UNASSIGNED
            color = '0.6' if not ct else None
            v = g[metric].to_numpy(dtype=float)
            ax.hist(v[np.isfinite(v)], bins=edges, alpha=0.55,
                    label=f'{name} (n={len(g)})', color=color)
        ax.legend(fontsize=8)
    else:
        ax.hist(finite, bins=edges, alpha=0.8)
    if picked_range is not None:
        ax.axvspan(picked_range[0], picked_range[1], color='gold', alpha=0.25,
                   label='selected range')
    ax.set_xlabel(metric)
    ax.set_ylabel('cells')
    ax.set_title(f'{m}/{h}/ch{ch} -- {metric} per cell '
                 f'({len(rows)} cells)', fontsize=10)
    return fig


def fig_brightness_vs_count(table, source):
    """QC view 4b: per-cell brightness vs spot count."""
    m, h, ch = source
    rows = table[(table['modality'] == m) & (table['hybe'] == h)
                 & (table['channel'] == int(ch))]
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for ct, g in rows.groupby('celltype'):
        name = ct if ct else UNASSIGNED
        ax.scatter(g['n_spots'], g['brightness_median'], s=14, alpha=0.6,
                   label=f'{name}', color='0.6' if not ct else None)
    ax.set_xlabel('spots per cell')
    ax.set_ylabel('median spot brightness (counts)')
    ax.legend(fontsize=8)
    ax.set_title(f'{m}/{h}/ch{ch} -- brightness vs count', fontsize=10)
    return fig


def fig_distance_hist(hists, title, range_um=None):
    """distances.distance_histogram output -> figure. hists is either
    (counts, edges) or {celltype: (counts, edges)}."""
    fig, ax = plt.subplots(figsize=(5.6, 4))
    if isinstance(hists, dict):
        for name, (counts, edges) in hists.items():
            ax.stairs(counts, edges, alpha=0.7, fill=False,
                      label=f'{name} ({int(counts.sum())} pairs)',
                      color='0.6' if name == UNASSIGNED else None)
        ax.legend(fontsize=8)
    else:
        counts, edges = hists
        ax.stairs(counts, edges, fill=True, alpha=0.7)
    ax.set_xlabel('pair distance (um)')
    ax.set_ylabel('pairs')
    ax.set_title(title, fontsize=10)
    return fig
