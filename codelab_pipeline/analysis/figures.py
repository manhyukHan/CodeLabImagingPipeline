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


def fig_fov_consistency(dmaps, fovs, mask=None, min_n=1):
    """Per-FOV ensembles + their pairwise SCC matrix -- QC view 2."""
    res = ens.fov_consistency(dmaps, fovs, mask, min_n=min_n)
    k = len(res['fovs'])
    fig, axes = plt.subplots(1, k + 1, figsize=(3.4 * (k + 1), 3.8),
                             squeeze=False)
    for ax, f in zip(axes[0], res['fovs']):
        _dmap_ax(ax, res['maps'][f], f'FOV{f:03d}')
    ax = axes[0][-1]
    im = ax.imshow(res['scc'], cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels([str(f) for f in res['fovs']], fontsize=8)
    ax.set_yticklabels([str(f) for f in res['fovs']], fontsize=8)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f'{res["scc"][i, j]:.2f}', ha='center',
                    va='center', fontsize=7,
                    color='white' if res['scc'][i, j] < 0.6 else 'black')
    ax.set_title('pairwise SCC', fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle('FOV-level consistency', fontsize=11)
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
