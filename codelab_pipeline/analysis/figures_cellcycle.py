"""
Figures for the Cell Cycle stage: the ring seen four ways, the gene
profiles by biological role, the spectrum and the panel's total count
along the cycle, the anchors, and the agreement of disjoint sub-panels.

Conventions fixed with the user (2026-09-13):
  - phase runs 0..360 deg with 0 = S-phase, 180 = G2/M, 360 = G1/S;
    every phase axis and the one horizontal colour bar carry those
    three labels; the phase colour map is a rainbow;
  - the ring in the CLR plane is drawn as contours of the centre ratio
    rho (cell radius / ring radius): solid at the expected rho = 1, the
    others dashed;
  - gene profiles are lines only (no binned dots), one figure per role
    (S indicators, G2/M indicators, unassigned), gene colours from a
    categorical palette, never the phase rainbow.

matplotlib only at import; scikit-learn and umap-learn are imported
inside the functions that need them and reported as missing rather
than crashing the stage.
"""
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib import cm, colors                                # noqa: E402

from codelab_pipeline.analysis import cellcycle as CC            # noqa: E402

PHASE_TICKS = ((0, 'S-phase'), (180, 'G2/M'), (360, 'G1/S'))
PHASE_CMAP = 'rainbow'
PHASE_NORM = colors.Normalize(vmin=0.0, vmax=360.0)
# Okabe-Ito, then tab20 for panels beyond eight genes
OKABE_ITO = ('#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2',
             '#D55E00', '#CC79A7', '#000000')


def gene_palette(n):
    if n <= len(OKABE_ITO):
        return list(OKABE_ITO[:n])
    return [plt.get_cmap('tab20')(i % 20) for i in range(n)]


def phase_axis(ax, which='x'):
    """Ticks and labels of a 0..360 phase axis."""
    pos = [p for p, _ in PHASE_TICKS]
    lab = [l for _, l in PHASE_TICKS]
    if which == 'x':
        ax.set_xlim(0, 360)
        ax.set_xticks(pos)
        ax.set_xticklabels(lab)
    else:
        ax.set_ylim(0, 360)
        ax.set_yticks(pos)
        ax.set_yticklabels(lab)


def phase_colorbar(fig, axes, label='cell-cycle phase'):
    """ONE horizontal colour bar under the given axes."""
    sm = cm.ScalarMappable(norm=PHASE_NORM, cmap=PHASE_CMAP)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.05,
                      pad=0.10, aspect=45)
    cb.set_ticks([p for p, _ in PHASE_TICKS])
    cb.set_ticklabels([l for _, l in PHASE_TICKS])
    cb.set_label(label)
    return cb


# -- the ring in the CLR plane and three other embeddings ----------------------

def ring_plane(model, name):
    """(plane (2, G), centre (G,), ring2d (T, 2)): the ring's PCA plane in
    the CLR space of `name`'s panel -- the same plane radius() uses."""
    ring = CC.clr(np.exp(model.log_pi(name)))
    cen = ring.mean(0)
    _U, _S, Vt = np.linalg.svd(ring - cen, full_matrices=False)
    plane = Vt[:2]
    return plane, cen, (ring - cen) @ plane.T


def embed_pca(model, name, X, pseudo=0.5):
    """Cells in the ring's plane: (n, 2), plus the ring's own curve."""
    plane, cen, ring2d = ring_plane(model, name)
    F = (X + pseudo) / (X + pseudo).sum(1, keepdims=True)
    return (CC.clr(F) - cen) @ plane.T, ring2d


def clr_z(X, pseudo=0.5):
    """CLR then per-gene z-score: the input the notebook-style
    embeddings (tSNE, UMAP) and the Mapper see."""
    F = (X + pseudo) / (X + pseudo).sum(1, keepdims=True)
    Z = CC.clr(F)
    return (Z - Z.mean(0)) / (Z.std(0) + 1e-9)


def mapper_graph(Z, lens, n_intervals=8, overlap=0.3, eps=None, min_samples=3):
    """A minimal Mapper (Singh, Memoli, Carlsson 2007): cover the 2-D
    lens with an overlapping grid, cluster the rows of Z inside each
    cover cell (DBSCAN), one node per cluster, an edge where two nodes
    share a cell. A loop in the graph is the ring seen topologically,
    with no dimension reduction of the data itself.

    Returns (nodes: [index arrays], edges: [(i, j)])."""
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
    if eps is None:
        nn = NearestNeighbors(n_neighbors=6).fit(Z)
        d, _ = nn.kneighbors(Z)
        eps = 1.5 * float(np.median(d[:, -1]))
    nodes = []
    lo, hi = lens.min(0), lens.max(0)
    width = (hi - lo) / n_intervals
    step = width * (1.0 - overlap)
    for i in range(n_intervals):
        for j in range(n_intervals):
            a = lo + np.array([i, j]) * step
            b = a + width
            if i == n_intervals - 1:
                b[0] = hi[0] + 1e-9
            if j == n_intervals - 1:
                b[1] = hi[1] + 1e-9
            inside = np.where((lens[:, 0] >= a[0]) & (lens[:, 0] < b[0])
                              & (lens[:, 1] >= a[1]) & (lens[:, 1] < b[1]))[0]
            if len(inside) < min_samples:
                continue
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit(Z[inside]).labels_
            for k in np.unique(labels):
                if k < 0:
                    continue
                nodes.append(inside[labels == k])
    sets = [set(n.tolist()) for n in nodes]
    edges = [(i, j) for i in range(len(nodes)) for j in range(i + 1, len(nodes))
             if sets[i] & sets[j]]
    return nodes, edges


def _circ_mean_deg(theta_deg):
    z = np.exp(1j * np.radians(theta_deg)).mean()
    return float(np.degrees(np.angle(z)) % 360.0)


def fig_embeddings(model, name, X, theta_deg, seed=0, max_cells=3000, title=None):
    """The same cells in four pictures, coloured by the model's phase:
    the ring's CLR plane (PCA) with the rho contours, tSNE, UMAP, and the
    Mapper graph. Returns (fig, notes) -- notes names any embedding that
    could not be drawn (missing library) instead of failing."""
    from matplotlib.collections import LineCollection
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = np.arange(n) if n <= max_cells else rng.choice(n, max_cells, replace=False)
    Xs, th = np.asarray(X, float)[idx], np.asarray(theta_deg, float)[idx]
    Z = clr_z(Xs)
    P, ring2d = embed_pca(model, name, Xs)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    notes = []
    kw = dict(c=th, cmap=PHASE_CMAP, norm=PHASE_NORM, s=6, alpha=0.75, linewidths=0)

    ax = axes[0]
    ax.scatter(P[:, 0], P[:, 1], **kw)
    closed = np.vstack([ring2d, ring2d[:1]])
    for rho, style, lw in ((0.5, '--', 0.8), (1.0, '-', 1.6), (1.5, '--', 0.8)):
        ax.plot(rho * closed[:, 0], rho * closed[:, 1], style, color='k', lw=lw, alpha=0.9)
    ax.plot(0, 0, 'k+', ms=8)
    ax.set_title('CLR plane of the ring (PCA)\ncontours: centre ratio 0.5, 1 (solid), 1.5')
    ax.set_xlabel('PC1 (CLR)')
    ax.set_ylabel('PC2 (CLR)')
    ax.set_aspect('equal', adjustable='datalim')

    ax = axes[1]
    try:
        from sklearn.manifold import TSNE
        E = TSNE(n_components=2, perplexity=30, init='pca', random_state=seed).fit_transform(Z)
        ax.scatter(E[:, 0], E[:, 1], **kw)
        ax.set_title('tSNE (CLR, z-scored genes)')
    except Exception as exc:                                    # noqa: BLE001
        notes.append(f'tSNE: {type(exc).__name__}: {exc}')
        ax.set_title('tSNE unavailable')
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[2]
    try:
        import umap
        E = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=seed).fit_transform(Z)
        ax.scatter(E[:, 0], E[:, 1], **kw)
        ax.set_title('UMAP (CLR, z-scored genes)')
    except Exception as exc:                                    # noqa: BLE001
        notes.append(f'UMAP: {type(exc).__name__}: {exc}')
        ax.set_title('UMAP unavailable')
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[3]
    try:
        nodes, edges = mapper_graph(Z, P)
        pos = np.array([P[m].mean(0) for m in nodes])
        col = np.array([_circ_mean_deg(th[m]) for m in nodes])
        size = np.array([len(m) for m in nodes], float)
        if edges:
            segs = [(pos[i], pos[j]) for i, j in edges]
            ax.add_collection(LineCollection(segs, colors='0.6', linewidths=0.7, zorder=1))
        ax.scatter(pos[:, 0], pos[:, 1], c=col, cmap=PHASE_CMAP, norm=PHASE_NORM,
                   s=12 + 3.0 * np.sqrt(size), edgecolors='k', linewidths=0.4, zorder=2)
        ax.set_title(f'TDA Mapper graph ({len(nodes)} nodes, {len(edges)} edges)\n'
                     'lens: the two PCs; clusters: DBSCAN in CLR')
        ax.set_aspect('equal', adjustable='datalim')
        ax.autoscale_view()
    except Exception as exc:                                    # noqa: BLE001
        notes.append(f'Mapper: {type(exc).__name__}: {exc}')
        ax.set_title('Mapper unavailable')
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(title or f'{name}: {len(idx)} cells, phase from the model', y=1.02)
    phase_colorbar(fig, list(axes))
    return fig, notes


# -- gene profiles by biological role -----------------------------------------

def fold_profiles(model):
    """(T, G) fold change of each gene's expected fraction against its
    own cycle mean, on the grid, from the SHARED profiles."""
    f = model.B @ model.coef.T
    return np.exp(f - f.mean(0, keepdims=True))


def fig_profiles_by_role(model, roles, ncols=1, title=None):
    """roles: ordered {role: [gene, ...]} -- one axes per role, one line
    per gene, categorical colours, the shared phase axis."""
    deg = np.degrees(model.grid)
    order = np.argsort(deg)
    fc = fold_profiles(model)
    roles = {r: [g for g in gs if g in model.gi] for r, gs in roles.items()}
    roles = {r: gs for r, gs in roles.items() if gs}
    nrow = int(np.ceil(len(roles) / ncols))
    fig, axes = plt.subplots(nrow, ncols, figsize=(6.4 * ncols, 3.2 * nrow), squeeze=False)
    axes = axes.ravel()
    for ax, (role, genes) in zip(axes, roles.items()):
        pal = gene_palette(len(genes))
        for g, c in zip(genes, pal):
            ax.plot(deg[order], fc[order, model.gi[g]], color=c, lw=1.6, label=g)
        ax.axhline(1.0, color='0.7', lw=0.8, ls=':')
        ax.set_ylabel('fraction / cycle mean')
        ax.set_title(f'{role} ({len(genes)} genes)')
        phase_axis(ax)
        ax.legend(fontsize=8, ncol=2 if len(genes) > 6 else 1, frameon=False)
    for ax in axes[len(roles):]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


# -- spectrum and total count along the cycle ---------------------------------

def binned_stat(theta_deg, values, n_bins=24, stat='median'):
    """(centres, stat per bin, q25, q75) over phase bins."""
    edges = np.linspace(0, 360, n_bins + 1)
    b = np.clip(np.digitize(theta_deg % 360.0, edges) - 1, 0, n_bins - 1)
    c = 0.5 * (edges[:-1] + edges[1:])
    m = np.full(n_bins, np.nan); lo = np.full(n_bins, np.nan); hi = np.full(n_bins, np.nan)
    for k in range(n_bins):
        v = np.asarray(values)[b == k]
        if len(v) >= 5:
            m[k] = np.median(v) if stat == 'median' else np.mean(v)
            lo[k], hi[k] = np.percentile(v, [25, 75])
    return c, m, lo, hi


def fig_spectrum_and_totals(model, groups, q, totals, title=None):
    """Two axes: the spectrum per condition (posterior-mean density,
    smoothed as the model's prior is) and the panel's total count per
    cell along the cycle (median line, quartile band) per condition."""
    deg = np.degrees(model.grid)
    order = np.argsort(deg)
    theta = np.degrees(np.angle(q @ np.exp(1j * model.grid))) % 360.0
    names = [g for g in dict.fromkeys(groups)]
    pal = gene_palette(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 3.8))
    for g, c in zip(names, pal):
        k = np.asarray(groups) == g
        if k.sum() < 5:
            continue
        w = model.spectrum(q[k])
        ax1.plot(deg[order], w[order] * len(w) / 360.0, color=c, lw=1.6, label=f'{g} (n={int(k.sum())})')
        cen, m, lo, hi = binned_stat(theta[k], np.asarray(totals)[k])
        ax2.plot(cen, m, color=c, lw=1.6, label=g)
        ax2.fill_between(cen, lo, hi, color=c, alpha=0.12, linewidth=0)
    ax1.set_ylabel('cell density (per degree x 360)')
    ax1.set_title('spectrum per condition')
    ax1.legend(fontsize=8, frameon=False)
    ax2.set_ylabel('panel total count per cell')
    ax2.set_title('total count along the cycle (median, quartiles)')
    for ax in (ax1, ax2):
        phase_axis(ax)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


# -- anchors across experiments ------------------------------------------------

def anchor_table(entries, n_boot=500, seed=0):
    """entries: [(experiment, condition, theta_deg, R, bf_ring, radius)].
    Per row: n, circular mean with a bootstrap 95% CI half-width of the
    mean direction, concentration, median R, fraction log bf_ring > 0,
    median radius."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for exp, cond, th, R, bf, rad in entries:
        th = np.asarray(th, float)
        z = np.exp(1j * np.radians(th))
        mean = float(np.degrees(np.angle(z.mean())) % 360.0)
        boots = []
        for _ in range(n_boot):
            zz = z[rng.integers(0, len(z), len(z))].mean()
            boots.append(np.degrees(np.angle(zz * np.conj(z.mean()))))
        rows.append({'experiment': exp, 'condition': cond, 'n': int(len(th)),
                     'mean_deg': round(mean, 1),
                     'ci95_halfwidth_deg': round(float(np.percentile(np.abs(boots), 95)), 1),
                     'concentration': round(float(np.abs(z.mean())), 2),
                     'median_R': round(float(np.median(R)), 2),
                     'frac_bf_ring_pos': round(float(np.mean(np.asarray(bf) > 0)), 2),
                     'median_radius': round(float(np.median(rad)), 2)})
    return pd.DataFrame(rows)


def fig_anchor_spectra(model, entries, conditions, title=None):
    """The arrested conditions of every experiment on ONE phase axis:
    entries [(experiment, condition, q)], one line per (experiment,
    condition) for the named conditions, mean direction marked."""
    deg = np.degrees(model.grid)
    order = np.argsort(deg)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    styles = ['-', '--', ':', '-.']
    exps = list(dict.fromkeys(e for e, _c, _q in entries))
    pal = gene_palette(len(conditions))
    for (exp, cond, q) in entries:
        if cond not in conditions:
            continue
        w = model.spectrum(q)
        c = pal[conditions.index(cond)]
        ls = styles[exps.index(exp) % len(styles)]
        ax.plot(deg[order], w[order] * len(w) / 360.0, color=c, ls=ls, lw=1.7, label=f'{exp} {cond} (n={len(q)})')
        mean = np.degrees(np.angle((q @ np.exp(1j * model.grid)).mean())) % 360.0
        ax.axvline(mean, color=c, ls=ls, lw=0.9, alpha=0.7)
    ax.set_ylabel('cell density (per degree x 360)')
    ax.set_title(title or 'anchors: arrested conditions across experiments')
    ax.legend(fontsize=8, frameon=False)
    phase_axis(ax)
    fig.tight_layout()
    return fig


# -- disjoint sub-panels ---------------------------------------------------------

def subpanel_agreement(model, name, X, subsets, prior='uniform'):
    """Phases of the same cells from disjoint gene subsets of one panel.
    subsets: {label: [genes]}. Returns (thetas {label: deg, 'full': deg},
    stats DataFrame over every pair incl. each subset vs full)."""
    import pandas as pd
    thetas = {'full': np.degrees(model.phase(name, X, prior=prior)[0]) % 360.0}
    for lab, genes in subsets.items():
        genes = [g for g in genes if g in model.panels[name]]
        sub = X[:, [model.panels[name].index(g) for g in genes]]
        thetas[lab] = np.degrees(model.phase(name, sub, prior=prior, subset=genes)[0]) % 360.0
    labs = list(thetas)
    rows = []
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            d = np.abs((thetas[labs[i]] - thetas[labs[j]] + 180.0) % 360.0 - 180.0)
            rows.append({'a': labs[i], 'b': labs[j], 'n': int(len(d)),
                         'median_abs_diff_deg': round(float(np.median(d)), 1),
                         'frac_within_30': round(float(np.mean(d <= 30)), 2),
                         'frac_within_45': round(float(np.mean(d <= 45)), 2),
                         'frac_within_90': round(float(np.mean(d <= 90)), 2)})
    return thetas, pd.DataFrame(rows)


def fig_subpanel(thetas, a, b, title=None):
    """theta_a vs theta_b as a 2-D histogram (both 0..360) and the
    histogram of their circular difference."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={'width_ratios': [1.15, 1]})
    ax1.hist2d(thetas[a], thetas[b], bins=36, range=[[0, 360], [0, 360]], cmap='Greys')
    ax1.plot([0, 360], [0, 360], 'r-', lw=0.8, alpha=0.6)
    ax1.set_xlabel(f'phase from {a}')
    ax1.set_ylabel(f'phase from {b}')
    phase_axis(ax1, 'x'); phase_axis(ax1, 'y')
    ax1.set_aspect('equal')
    d = (thetas[a] - thetas[b] + 180.0) % 360.0 - 180.0
    ax2.hist(d, bins=36, range=(-180, 180), color='0.4')
    ax2.axvline(0, color='r', lw=0.8)
    ax2.set_xlabel(f'phase difference {a} - {b} (deg)')
    ax2.set_ylabel('cells')
    ax2.set_title(f'median |diff| {np.median(np.abs(d)):.0f} deg, {np.mean(np.abs(d) <= 45) * 100:.0f}% within 45')
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
