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


def style_ax(ax):
    """The house style, applied to every non-image axes: no top/right
    spines (universal view convention, per explicit request)."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax


def _tick_positions(n):
    return sorted(set([0] + list(range(9, n - 1, 10)) + [n - 1]))


def dmap_ticks(ax, n, ids=None):
    """Ticks for an n-bin distance map: [0] + every 10th + [n-1].

    With ids (the layout's readout_id per bin, in bin order) the labels
    are the REAL readout indices -- the name the experiment layout gives
    each round, gaps and all -- per explicit request. Without them the
    fallback is the 1-based bin position."""
    pos = _tick_positions(n)
    have_ids = ids is not None and len(ids) == n
    labels = [str(int(ids[p])) if have_ids else str(p + 1) for p in pos]
    ax.set_xticks(pos)
    ax.set_yticks(pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('readout id' if have_ids else 'barcode', fontsize=10)


def _dmap_ax(ax, m, title, vmin=None, vmax=None, ids=None):
    finite = m[np.isfinite(m)]
    if vmin is None:
        vmin = np.quantile(finite, 0.02) if len(finite) else 0
    if vmax is None:
        vmax = np.quantile(finite, 0.98) if len(finite) else 1
    cmap = cm.seismic_r.copy()
    cmap.set_bad('0.15')
    im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title(title, fontsize=10)
    dmap_ticks(ax, m.shape[0], ids=ids)
    return im


def fig_ensemble(dmaps, mask=None, title='ensemble', reducer='median',
                 min_n=1, group_masks=None, bin_ids=None):
    """One ensemble map, optionally decomposed by named group masks
    (the FLAG axis: groups split the gated stack, they never re-gate)."""
    groups = [('all', mask)] if not group_masks else list(group_masks.items())
    n = len(groups)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.4), squeeze=False)
    ims = []
    for ax, (name, gmask) in zip(axes[0], groups):
        m, counts = ens.ensemble_map(dmaps, gmask, reducer, min_n)
        n_al = int(gmask.sum()) if gmask is not None else len(dmaps)
        ims.append(_dmap_ax(ax, m, f'{name}  (n={n_al})', ids=bin_ids))
    vmin = min(i.get_clim()[0] for i in ims)
    vmax = max(i.get_clim()[1] for i in ims)
    for i in ims:
        i.set_clim(vmin, vmax)
    fig.colorbar(ims[-1], ax=axes[0], shrink=0.8, label='distance (um)')
    fig.suptitle(f'{title}  [{reducer}, min_n={min_n}]', fontsize=11)
    return fig


def fig_fov_consistency(dmaps, fovs, mask=None, min_n=1, group_masks=None,
                        show_maps=True, bin_ids=None):
    """FOV-level QC: the MSD test, optionally with the per-FOV maps.

    SCC is GONE, per explicit decision -- one correlation between two
    grainy ensembles is meaningless next to the MSD distributions, which
    carry thousands of allele pairs. show_maps=False collapses each row
    to just the tested histogram; every panel names its n.
    """
    groups = [('all', mask)] if not group_masks else list(group_masks.items())
    fov_n = len(np.asarray(fovs))
    # a group emptied by the gate gets NO row -- the gate summary
    # already reports per-celltype counts, and a full-size blank panel
    # per excluded celltype is dead space (reported from the GUI)
    empty = [name for name, g in groups
             if int((np.ones(fov_n, bool) if g is None
                     else np.asarray(g, bool)).sum()) == 0]
    groups = [(name, g) for name, g in groups if name not in empty]
    if not groups:
        fig, ax = plt.subplots(figsize=(6.4, 3))
        ax.set_axis_off()
        ax.set_title('every group is empty after the gate', fontsize=10)
        return fig
    all_fovs = sorted(set(int(f) for f in np.asarray(fovs)))
    k = len(all_fovs) if show_maps else 0
    ncols = k + 1
    # histogram-only mode keeps a readable width -- 3.1 in is the MAP
    # column's size, and a lone 3.1-in histogram under a two-line title
    # is what made the stacked view collide (reported from the GUI)
    width = 3.1 * ncols if show_maps else 6.4
    fig, axes = plt.subplots(len(groups), ncols,
                             figsize=(width, 3.5 * len(groups)),
                             squeeze=False)
    fov_arr = np.asarray(fovs)
    for row, (name, gmask) in enumerate(groups):
        base = np.ones(len(fov_arr), bool) if gmask is None \
            else np.asarray(gmask, bool)
        if show_maps:
            res = ens.fov_consistency(dmaps, fovs, gmask, min_n=min_n)
            for col, f in enumerate(all_fovs):
                ax = axes[row][col]
                n_f = int((base & (fov_arr == f)).sum())
                if f in res['maps']:
                    _dmap_ax(ax, res['maps'][f],
                             f'{name} FOV{f:03d} (n={n_f})', ids=bin_ids)
                else:
                    ax.set_axis_off()
        ax = axes[row][-1]
        style_ax(ax)
        t = ens.fov_msd_test(dmaps, fovs, gmask)
        if len(t['msd_in']) and len(t['msd_cross']):
            edges = np.histogram_bin_edges(
                np.concatenate([t['msd_in'], t['msd_cross']]), bins=40)
            ax.hist(t['msd_cross'], bins=edges, density=True, alpha=0.5,
                    color='0.6', label=f"cross-FOV (n={len(t['msd_cross'])})")
            ax.hist(t['msd_in'], bins=edges, density=True, alpha=0.6,
                    color='crimson', label=f"in-FOV (n={len(t['msd_in'])})")
            ax.legend(fontsize=7)
            # wrapped, not one endless line: 41 FOVs of verdicts in a
            # single title row blows past any figure width
            items = [f"F{d['fov']}:{d['signed_neglog10p']:+.1f}"
                     for d in t['per_fov']]
            per = '\n'.join('  '.join(items[i:i + 8])
                            for i in range(0, len(items), 8))
            ax.set_title(f"{name} (n={int(base.sum())}): "
                         f"MSD -log10 p = {t['neglog10p']:.1f}\n"
                         f'per-FOV signed (+ = deviant)\n{per}', fontsize=8)
            ax.set_xlabel('pairwise map MSD (um^2)', fontsize=8)
            ax.set_ylabel('density', fontsize=8)
        else:
            # a degenerate group (e.g. 2 Unassigned alleles) gets a
            # labeled BLANK, not an empty axes with phantom 0..1 ticks
            ax.set_title(f'{name} (n={int(base.sum())}): too few pairs',
                         fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    sup = 'FOV-level consistency'
    if empty:
        sup += f"  (empty after gate: {', '.join(empty)})"
    fig.suptitle(sup, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def fig_allele_difference(dmaps, fovs, cells, mask=None):
    """Within-cell allele-pair dissimilarity against the cross-cell null."""
    res = ens.allele_difference(dmaps, fovs, cells, mask)
    fig, ax = plt.subplots(figsize=(5.2, 4))
    style_ax(ax)
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


def fig_polymer_qc(table, dmaps, thresholds, qc_result=None, bin_ids=None):
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
    # red-to-purple across the bins (reversed rainbow, per request)
    ax.bar(np.arange(len(eff)), eff,
           color=cm.rainbow.reversed()(np.linspace(0, 1, max(len(eff), 1))))
    ax.set_ylim(0, 1)
    if bin_ids is not None and len(bin_ids) == len(eff):
        # tpos, NOT pos: reusing `pos` here shadowed the position array
        # the completeness panel below still reads (hard-crashed the
        # preview through the Qt slot before the rename)
        tpos = _tick_positions(len(eff))
        ax.set_xticks(tpos)
        ax.set_xticklabels([str(int(bin_ids[p])) for p in tpos], fontsize=8)
        ax.set_xlabel('readout id', fontsize=10)
    else:
        ax.set_xlabel('barcode', fontsize=10)
    ax.set_ylabel('efficacy', fontsize=10)
    ax.set_title('efficacy per bin (after QC)' if qc_result is not None
                 else 'efficacy per bin', fontsize=9)
    ax = axes[4]
    comp = _P.completeness(pos)
    if len(comp):
        bins = np.arange(comp.max() + 2) - 0.5
        ax.hist(comp, bins=bins, color='steelblue', density=True,
                label=f'observed (n={len(comp)})')
        # the QUALITY MODEL's fit, drawn over the data (per request):
        # completeness K under u_i ~ N(0, tau^2) with per-bin b_j --
        # detection.fit_quality_model on the binary matrix, then the
        # model's K distribution by GHQ mixture of Poisson-binomials
        # (normal approximation per node, adequate at these n_bins).
        try:
            from codelab_pipeline.analysis import detection as _D
            from scipy import stats as _st
            X = _D.detection_matrix(pos)
            keep = X.sum(1) > 0
            if keep.sum() >= 20:
                fit = _D.fit_quality_model(X[keep], n_nodes=15, maxiter=200)
                nodes, w = np.polynomial.hermite_e.hermegauss(15)
                w = w / w.sum()
                u = nodes * fit['tau']
                S = 1 / (1 + np.exp(-(fit['b'][None, :] + u[:, None])))
                ks = np.arange(0, X.shape[1] + 1)
                pdf = np.zeros_like(ks, dtype=float)
                for i in range(len(u)):
                    mu = S[i].sum()
                    var = (S[i] * (1 - S[i])).sum()
                    pdf += w[i] * _st.norm.pdf(ks, mu, np.sqrt(max(var, 1e-9)))
                ax.plot(ks, pdf, color='crimson', lw=2,
                        label=f"quality model (tau={fit['tau']:.2f})")
        except Exception:
            pass
        ax.legend(fontsize=7)
    ax.set_xlabel('traced bins per allele', fontsize=8)
    ax.set_ylabel('density', fontsize=8)
    ax.set_title('completeness per allele + model fit', fontsize=9)
    for a in axes:
        style_ax(a)
    if qc_result is not None:
        kept = int(qc_result['kept'].sum())
        fig.suptitle(f'polymer QC -- {kept}/{len(table["amp"])} alleles kept, '
                     f'{int(qc_result["bads"].sum())} bins removed',
                     fontsize=11)
    else:
        fig.suptitle('polymer QC (preview -- not applied)', fontsize=11)
    return fig


def fig_expression_hist(table, source, metric='n_spots', bins=30,
                        per_celltype=True, picked_range=None,
                        source_label=None, norm_label=None):
    """Expression histogram, celltype-decomposed; the range picker's view.

    picked_range=(lo, hi) draws the chosen gate band -- what the panel
    shows while the user drags the ExpressionRange condition. norm_label
    (e.g. 'normalized: by_source, Hyb_133 (SO57_exon)') is appended to
    the title on its own line, so the figure SAYS which normalization
    produced its axis."""
    m, h, ch = source
    source_label = source_label or f'{m}/{h}/ch{ch}'
    tail = f'\n{norm_label}' if norm_label else ''
    rows = table[(table['modality'] == m) & (table['hybe'] == h)
                 & (table['channel'] == int(ch))]
    fig, ax = plt.subplots(figsize=(5.6, 4))
    style_ax(ax)
    vals_all = rows[metric].to_numpy(dtype=float)
    finite = vals_all[np.isfinite(vals_all)]
    if len(finite) == 0:
        ax.set_title(f'{source_label}: no finite {metric}{tail}')
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
    ax.set_title(f'{source_label} -- {metric} per cell '
                 f'({len(rows)} cells){tail}', fontsize=10)
    return fig


def fig_brightness_vs_count(table, source, source_label=None):
    """QC view 4b: per-cell brightness vs spot count."""
    m, h, ch = source
    source_label = source_label or f'{m}/{h}/ch{ch}'
    rows = table[(table['modality'] == m) & (table['hybe'] == h)
                 & (table['channel'] == int(ch))]
    fig, ax = plt.subplots(figsize=(5, 4.2))
    style_ax(ax)
    for ct, g in rows.groupby('celltype'):
        name = ct if ct else UNASSIGNED
        ax.scatter(g['n_spots'], g['brightness_median'], s=14, alpha=0.6,
                   label=f'{name}', color='0.6' if not ct else None)
    ax.set_xlabel('spots per cell')
    ax.set_ylabel('median spot brightness (counts)')
    ax.legend(fontsize=8)
    ax.set_title(f'{source_label} -- brightness vs count', fontsize=10)
    return fig


def fig_distance_hist(hists, title, range_um=None):
    """distances.distance_histogram output -> figure. hists is either
    (counts, edges) or {celltype: (counts, edges)}."""
    fig, ax = plt.subplots(figsize=(5.6, 4))
    style_ax(ax)
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


def expression_hist_ax(ax, table, source, metric='n_spots', bins=30,
                       per_celltype=True, picked_range=None, label=None):
    """One expression histogram INTO a given axes -- the composable form
    the allele-mode group panels build on. label: display name for the
    source (e.g. 'RNA/Hyb_111 (Gorab_exon)/ch635')."""
    style_ax(ax)
    m, h, ch = source
    label = label or f'{m}/{h}/ch{ch}'
    rows = table[(table['modality'] == m) & (table['hybe'] == h)
                 & (table['channel'] == int(ch))]
    vals_all = rows[metric].to_numpy(dtype=float)
    finite = vals_all[np.isfinite(vals_all)]
    if len(finite) == 0:
        ax.set_title(f'{label}: no finite {metric}', fontsize=9)
        return ax
    edges = np.histogram_bin_edges(finite, bins=bins)
    if per_celltype:
        for ct, g in rows.groupby('celltype'):
            name = ct if ct else UNASSIGNED
            v = g[metric].to_numpy(dtype=float)
            ax.hist(v[np.isfinite(v)], bins=edges, alpha=0.55,
                    label=f'{name} (n={len(g)})',
                    color='0.6' if not ct else None)
        ax.legend(fontsize=7)
    else:
        ax.hist(finite, bins=edges, alpha=0.8)
    if picked_range is not None:
        ax.axvspan(picked_range[0], picked_range[1], color='gold', alpha=0.25)
    ax.set_xlabel(metric, fontsize=8)
    ax.set_ylabel('cells', fontsize=8)
    ax.set_title(label, fontsize=9)
    return ax


def distance_hist_ax(ax, pairs, bins=40, per_celltype=True):
    """One distance histogram INTO a given axes, from tidy pair/cell rows
    carrying d_um and celltype."""
    style_ax(ax)
    vals = pairs['d_um'].to_numpy(dtype=float) if len(pairs) else np.array([])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        ax.set_title('no pairs', fontsize=9)
        return ax
    edges = np.histogram_bin_edges(vals, bins=bins)
    if per_celltype:
        for ct, g in pairs.groupby('celltype'):
            name = ct if ct else UNASSIGNED
            ax.hist(g['d_um'].to_numpy(dtype=float), bins=edges, alpha=0.55,
                    label=f'{name} (n={len(g)})',
                    color='0.6' if not ct else None)
        ax.legend(fontsize=7)
    else:
        ax.hist(vals, bins=edges, alpha=0.8)
    ax.set_xlabel('distance (um)', fontsize=8)
    ax.set_ylabel('count', fontsize=8)
    return ax


def fig_repeat_toe_qc(pop):
    """The missed QC pair, per request: hybe-repeat distance and toe
    efficacy.

    Repeat rounds (datatype R) re-image the SAME genomic bin as their H
    round; |pos_H - pos_R| per allele is the whole-pipeline replication
    error in um. Toe rounds (datatype T) carry identity markers whose
    per-round efficacy is a hybridization health bar. Both extracted at
    Population.build (repeat_pos_um / toe_pos_um beside the H bins).
    """
    al = pop.alleles or {}
    rep = al.get('repeat_pos_um')
    rep_ids = al.get('repeat_ids') or []
    toe = al.get('toe_pos_um')
    toe_ids = al.get('toe_ids') or []
    bin_ids = list(al.get('bin_ids') or [])

    # ONE PANEL PER PAIR, not one axis carrying all of them. Overlaid step
    # lines answered "how wide is the spread" but not the question a
    # replication check is actually asking, which is per pair: is THIS
    # bin's repeat tight? Seven lines sharing one axis also hid a single
    # bad pair inside the pooled envelope.
    #
    # Each panel carries a GRAY reference: the same allele's distance to
    # every OTHER genomic bin -- non-identical, non-repeat. That is the
    # null this measurement has to beat. A repeat histogram sitting on top
    # of the gray one means the "replicate" is no closer than an unrelated
    # locus, which no summary statistic states as plainly.
    dists, refs = {}, {}
    if rep is not None and len(rep_ids) and bin_ids:
        pos = al['pos_um']
        for j, rid in enumerate(rep_ids):
            if rid not in bin_ids:
                continue
            k = bin_ids.index(rid)
            hpos = pos[:, k, :]
            d = np.sqrt(((hpos - rep[:, j, :]) ** 2).sum(1))
            d = d[np.isfinite(d)]
            if not len(d):
                continue
            dists[j] = d
            # distance from THIS bin to every other bin, same alleles
            other = np.delete(np.arange(pos.shape[1]), k)
            if len(other):
                dd = np.sqrt(((hpos[:, None, :] - pos[:, other, :]) ** 2).sum(-1))
                dd = dd[np.isfinite(dd)]
                refs[j] = dd

    n_pairs = len(dists)
    ncols = min(4, max(1, n_pairs))
    nrows = int(np.ceil(n_pairs / ncols)) if n_pairs else 1
    # constrained_layout: a per-pair grid stacked above a full-width toe
    # panel collides otherwise -- row 2's titles land on row 1's x labels
    fig = plt.figure(figsize=(3.3 * ncols, 2.9 * nrows + 3.8),
                     constrained_layout=True)
    gs = fig.add_gridspec(nrows + 1, ncols,
                          height_ratios=[1] * nrows + [1.3])
    if n_pairs:
        colors = cm.rainbow(np.linspace(0, 1, len(rep_ids)))
        # one shared x-range across every panel, so the panels can be
        # compared to each other and not just to their own gray
        top = float(max(d.max() for d in dists.values()))
        edges = np.linspace(0.0, top if top > 0 else 1.0, 41)
        for i, (j, d) in enumerate(sorted(dists.items())):
            ax = fig.add_subplot(gs[i // ncols, i % ncols])
            style_ax(ax)
            ref = refs.get(j)
            if ref is not None and len(ref):
                ax.hist(ref, bins=edges, density=True, color='0.85',
                        label=f'other bins (n={len(ref)})')
            ax.hist(d, bins=edges, density=True, histtype='step',
                    linewidth=1.6, color=colors[j])
            ax.axvline(float(np.median(d)), color=colors[j], linestyle='--',
                       linewidth=1.0)
            ax.set_title(f'H{rep_ids[j]} vs R{rep_ids[j]}\n'
                         f'n={len(d)}, med {np.median(d):.2f} um', fontsize=8)
            ax.set_xlim(edges[0], edges[-1])
            if i % ncols == 0:
                ax.set_ylabel('density (1/um)', fontsize=8)
            ax.set_xlabel('distance (um)', fontsize=8)
            ax.tick_params(labelsize=7)
        # the gray legend once, on the first panel only
        first = fig.axes[0]
        first.legend(fontsize=6, loc='upper right')
    else:
        ax = fig.add_subplot(gs[0, :])
        style_ax(ax)
        ax.set_title('no repeat rounds in the layout / population',
                     fontsize=10)

    ax = fig.add_subplot(gs[nrows, :])
    style_ax(ax)
    if toe is not None and len(toe_ids) and len(toe):
        # toes are IDENTITY markers -- a toe absent outside its own
        # celltype is working, not failing; pooled efficacy buries that
        # (measured on MP58: pooled reads ~0 while the signal is
        # celltype-structured), so the bars decompose by celltype
        cts = np.array([c if c else UNASSIGNED
                        for c in al.get('celltype', [''] * len(toe))])
        names = sorted(set(cts))
        width = 0.8 / max(len(names), 1)
        hit = np.isfinite(toe[:, :, 0])
        top = 0.0
        for k, name in enumerate(names):
            sel = cts == name
            eff = hit[sel].mean(0) if sel.any() else np.zeros(len(toe_ids))
            top = max(top, float(eff.max()) if len(eff) else 0.0)
            ax.bar(np.arange(len(toe_ids)) + k * width, eff, width=width,
                   label=f'{name} (n={int(sel.sum())})',
                   color='0.6' if name == UNASSIGNED else None)
        ax.set_xticks(np.arange(len(toe_ids)) + 0.4 - width / 2)
        ax.set_xticklabels([f'T{t}' for t in toe_ids], fontsize=8)
        ax.set_ylim(0, max(0.05, top * 1.25))
        ax.legend(fontsize=7)
        ax.set_xlabel('toe round', fontsize=8)
        ax.set_ylabel('traced fraction (alleles)', fontsize=8)
        ax.set_title(f'toe efficacy by celltype (n={len(toe)} alleles)',
                     fontsize=10)
    else:
        ax.set_title('no toe rounds in the layout / population', fontsize=10)
    fig.suptitle('repeat / toe QC', fontsize=12)
    return fig
