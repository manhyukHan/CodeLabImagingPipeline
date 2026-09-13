"""
Figures for the Cell Cycle stage: the ring seen three ways, the gene
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


def _point_in_polygon(x, y, poly):
    """Even-odd test: is (x, y) inside the closed polygon `poly` (n, 2)?"""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xc:
                inside = not inside
    return inside


def _circ_mean_deg(theta_deg):
    z = np.exp(1j * np.radians(theta_deg)).mean()
    return float(np.degrees(np.angle(z)) % 360.0)


def fig_embeddings(model, name, X, theta_deg, seed=0, max_cells=3000, title=None):
    """The same cells in three pictures, coloured by the model's phase:
    the ring's CLR plane (PCA) with the rho contours, tSNE and UMAP.
    Returns (fig, notes) -- notes names any embedding that could not be
    drawn (missing library) instead of failing. (A TDA Mapper panel was
    tried and dropped with the user, 2026-09-13: on this data the CLR
    cloud is a filled disc with a phase gradient, and the Mapper on a
    PCA lens returned the cover grid, not a loop.)"""
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = np.arange(n) if n <= max_cells else rng.choice(n, max_cells, replace=False)
    Xs, th = np.asarray(X, float)[idx], np.asarray(theta_deg, float)[idx]
    Z = clr_z(Xs)
    P, ring2d = embed_pca(model, name, Xs)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    notes = []
    kw = dict(c=th, cmap=PHASE_CMAP, norm=PHASE_NORM, s=6, alpha=0.75, linewidths=0)

    ax = axes[0]
    ax.scatter(P[:, 0], P[:, 1], **kw)
    # the fitted ring as the 72-gon of its grid points (solid), and the
    # same polygon with every vertex's distance from the centre scaled
    # by 0.5 and 1.5 (dashed) -- the user's construction (2026-09-13):
    # the ring's radius differs per grid theta and the contours keep
    # that. The scaled copies nest only when the centre lies inside the
    # polygon; when it does not (JP_001's crescent-shaped ring), they
    # cross, and the title says so: there the scalar centre ratio is a
    # coarse picture and bf_ring is the inside/outside criterion.
    closed = np.vstack([ring2d, ring2d[:1]])
    for rho, style, lw in ((0.5, '--', 0.8), (1.0, '-', 1.6), (1.5, '--', 0.8)):
        ax.plot(rho * closed[:, 0], rho * closed[:, 1], style, color='k', lw=lw, alpha=0.9)
    ax.plot(0, 0, 'k+', ms=8)
    inside = _point_in_polygon(0.0, 0.0, ring2d)
    ax.set_title('CLR plane of the ring (PCA)\nsolid: fitted ring; dashed: centre ratio 0.5 and 1.5'
                 + ('' if inside else '\n(centre outside the ring polygon: contours cross)'))
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


# -- the angle as cycle time -------------------------------------------------

def cycle_time_map(w, grid, birth_deg=0.0, growth='uniform'):
    """The phase axis re-scaled by where the CYCLING cells are.

    The fitted angle is a distance in composition, not time: the ring
    covers a stretch of the cycle where composition barely changes
    (G1) with the same number of degrees as one where it changes fast.
    Under the ergodic assumption -- an asynchronous population spends
    cells on each part of the cycle in proportion to the time spent
    there -- the spectrum of the unsynchronized cells IS the clock.

    w: the training population's spectrum on `grid` (T,), summing to 1.
    birth_deg: where the cycle starts (age 0) -- cell division. With
    growth='uniform' the returned tau is the plain cumulative fraction
    of cycling cells from birth_deg forward; with growth='exponential'
    the age distribution of an exponentially growing population,
    p(age) = 2 ln2 / T * 2^(-age/T), is inverted (age/T = -log2(1 -
    F/2)), which stretches the late cycle: young cells are twice as
    numerous as cells about to divide, so equal cell counts late in
    the cycle stand for more time.

    Returns a function deg -> tau in [0, 1) (vectorised), and the
    per-grid tau.
    """
    deg = np.degrees(np.asarray(grid)) % 360.0
    order = np.argsort(deg)
    d, ww = deg[order], np.asarray(w, float)[order]
    ww = ww / ww.sum()
    # rotate so the cycle starts at birth_deg
    start = np.searchsorted(d, birth_deg % 360.0)
    d2 = np.concatenate([d[start:], d[:start] + 360.0])
    w2 = np.concatenate([ww[start:], ww[:start]])
    edges = np.concatenate([[d2[0]], 0.5 * (d2[1:] + d2[:-1]), [d2[0] + 360.0]])
    F_edges = np.concatenate([[0.0], np.cumsum(w2)])
    if growth == 'exponential':
        F_edges = -np.log2(1.0 - np.clip(F_edges, 0, 1) / 2.0)
    F_edges = F_edges / F_edges[-1]

    def tau(theta_deg):
        x = (np.asarray(theta_deg, float) - birth_deg) % 360.0 + d2[0]
        return np.interp(x, edges, F_edges)
    return tau, tau(deg)


def fig_cycle_time(model, spectra, birth_deg=0.0, anchors=None, title=None):
    """spectra: {experiment: training spectrum (T,)}. Left: the spectra
    over the angle; middle: tau(angle) for each experiment, uniform and
    exponential-growth versions; right: the same spectra over tau (flat
    by construction for the training population). anchors:
    [(label, experiment, deg)] marked on the middle axes."""
    deg = np.degrees(model.grid)
    order = np.argsort(deg)
    names = list(spectra)
    pal = gene_palette(len(names))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
    for name, c in zip(names, pal):
        w = np.asarray(spectra[name], float)
        ax1.plot(deg[order], w[order] * len(w) / 360.0, color=c, lw=1.6, label=name)
        tau_u, tu = cycle_time_map(w, model.grid, birth_deg, 'uniform')
        tau_e, te = cycle_time_map(w, model.grid, birth_deg, 'exponential')
        xs = np.linspace(0, 360, 361)
        ax2.plot(xs, tau_u(xs), color=c, lw=1.6, label=f'{name} uniform')
        ax2.plot(xs, tau_e(xs), color=c, lw=1.0, ls='--', label=f'{name} exponential growth')
        # the ring's speed in time: degrees of angle per 1% of cycle time
        # -- high where the composition changes a lot in little time
        # (few cycling cells per degree), low on a plateau. This is
        # 1 / (dtau/dtheta) per grid bin; the bin width is 360/T.
        speed = (360.0 / len(w)) / np.maximum(w[order] / w.sum(), 1e-9) / 100.0
        ax3.plot(deg[order], speed, color=c, lw=1.6, label=name)
    ax1.set_ylabel('cell density (per degree x 360)')
    ax1.set_title('training spectrum over the angle')
    ax1.legend(fontsize=8, frameon=False)
    phase_axis(ax1)
    ax2.set_ylabel('cycle time fraction tau')
    ax2.set_title(f'angle -> time (birth at {birth_deg:.0f} deg)')
    ax2.axhline(0.5, color='0.8', lw=0.8, ls=':')
    for lab, exp_, dg in (anchors or []):
        c = pal[names.index(exp_)] if exp_ in names else 'k'
        tau_u, _ = cycle_time_map(spectra[exp_], model.grid, birth_deg, 'uniform')
        ax2.plot([dg], [tau_u(dg)], 'o', color=c, ms=6)
        ax2.annotate(lab, (dg, tau_u(dg)), textcoords='offset points', xytext=(5, -10), fontsize=8, color=c)
    ax2.legend(fontsize=7, frameon=False)
    phase_axis(ax2)
    ax3.set_ylabel('degrees of angle per 1% of cycle time')
    ax3.set_title('how fast the composition moves along the cycle')
    ax3.set_yscale('log')
    ax3.legend(fontsize=8, frameon=False)
    phase_axis(ax3)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
