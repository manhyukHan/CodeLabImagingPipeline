"""
Figures for the Cell Cycle stage: the ring in its CLR plane (and tSNE),
the gene profiles by biological role, the spectrum and the panel's
total count along the cycle, the anchors, the agreement of disjoint
sub-panels, the angle as cycle time, category arcs proposed from cycle
time, the FOV overlay, and the phase / category histograms the
Analysis tab shows under a gate.

Conventions fixed with the user (2026-09-13):
  - phase axes carry DEGREE ticks (0..360 by 60) with the roles as a
    second row under the axis ('S' at the S genes' mean peak, 'G2/M'
    at the G2/M genes' mean peak -- role_marks); the one horizontal
    rainbow colour bar sits under the figure, clear of the axis labels;
  - the ring in the CLR plane is the fitted 72-gon (solid) with the
    same polygon scaled per vertex to centre ratio 0.5 and 1.5
    (dashed); the plane axes carry no ticks, only labels with the
    variance each direction explains; UMAP was dropped (slow, and the
    PCA plane already shows the gradient); tSNE stays as an option on
    at most 1500 cells;
  - histograms are centre-connected lines, never step outlines;
  - gene profiles are lines only, one figure per role, categorical
    gene colours, never the phase rainbow;
  - every axes loses its top and right spines (style_ax), and every
    title stays INSIDE the figure: the displayer shows a figure at its
    own pixel size, so a suptitle placed above y = 1 or a title wider
    than the figure is simply cut off (seen on the sub-panel and the
    embeddings figures). Long titles wrap (wrap_title) and suptitles
    get their own band (suptitle).

matplotlib only at import; scikit-learn is imported inside the function
that needs it and reported as missing rather than crashing the stage.
"""
import textwrap

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib import cm, colors, transforms                    # noqa: E402

from codelab_pipeline.analysis import cellcycle as CC            # noqa: E402

PHASE_CMAP = 'rainbow_r'      # 0 deg red -> 360 deg violet (user: the forward rainbow read backwards)
PHASE_NORM = colors.Normalize(vmin=0.0, vmax=360.0)
PHASE_TICKS = (0, 60, 120, 180, 240, 300, 360)
# Okabe-Ito, then tab20 for panels beyond eight genes
OKABE_ITO = ('#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2',
             '#D55E00', '#CC79A7', '#000000')
# The educated guess for cultured mammalian cells (~24 h cycle: G1 ~11 h,
# S ~8 h, G2+M ~5 h), as fractions of the cycle from birth. A starting
# point for the category arcs when no arrested population anchors them;
# the panel lets the user edit the shares.
DEFAULT_PHASE_SHARES = (('G1', 0.45), ('S', 0.33), ('G2/M', 0.22))


def gene_palette(n):
    if n <= len(OKABE_ITO):
        return list(OKABE_ITO[:n])
    return [plt.get_cmap('tab20')(i % 20) for i in range(n)]


def style_ax(ax):
    """The universal convention: no top or right spine."""
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    return ax


def wrap_title(text, width=80):
    """A title that fits the figure: wrapped at `width` characters."""
    return '\n'.join(textwrap.wrap(str(text), width=width)) if text else ''


def suptitle(fig, text, width=None):
    """A figure title INSIDE the figure, with its own band on top,
    wrapped to the figure's width (about 11 characters per inch at
    11 pt -- a 6-inch figure takes 66 characters a line)."""
    if not text:
        return
    if width is None:
        width = max(30, int(fig.get_figwidth() * 11))
    t = wrap_title(text, width)
    n = t.count('\n') + 1
    fig.suptitle(t, y=0.995, va='top', fontsize=11)
    fig.subplots_adjust(top=1.0 - 0.05 * n - 0.06)


def finish(fig, title=None, width=None):
    for ax in fig.axes:
        style_ax(ax)
    if title:
        suptitle(fig, title, width)
    return fig


def role_marks(model, roles):
    """{'S': deg, 'G2/M': deg}: the mean peak of each role's genes in the
    fitted model -- the second row of a phase axis. Roles: {gene: role}."""
    pk, _ = model.peak_phase()
    out = {}
    for role in ('S', 'G2/M'):
        idx = [model.gi[g] for g in model.genes if (roles or {}).get(g) == role]
        if idx:
            out[role] = float(np.degrees(np.angle(np.mean(np.exp(1j * pk[idx])))) % 360.0)
    return out


def phase_axis(ax, which='x', marks=None):
    """Degree ticks on a 0..360 phase axis, and the role marks as a
    second row of labels under (or beside) the ticks."""
    ticks = list(PHASE_TICKS)
    # the role marks go RIGHT UNDER the tick labels and the axis label
    # under them (the first draft parked the marks below the label, in
    # a band of their own -- user screenshot)
    if which == 'x':
        ax.set_xlim(0, 360)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{t:d}' for t in ticks])
        ax.set_xlabel('phase (deg)', labelpad=(14 if marks else 4))
        if marks:
            tr = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            for name, deg in marks.items():
                ax.annotate(name, (float(deg) % 360.0, 0), xycoords=tr, xytext=(0, -17), textcoords='offset points',
                            ha='center', va='top', fontsize=8, color='0.35', annotation_clip=False)
    else:
        ax.set_ylim(0, 360)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{t:d}' for t in ticks])
        ax.set_ylabel('phase (deg)', labelpad=(28 if marks else 4))
        if marks:
            tr = transforms.blended_transform_factory(ax.transAxes, ax.transData)
            for name, deg in marks.items():
                ax.annotate(name, (0, float(deg) % 360.0), xycoords=tr, xytext=(-36, 0), textcoords='offset points',
                            ha='right', va='center', fontsize=8, color='0.35', annotation_clip=False)


def phase_colorbar(fig, axes, label='cell-cycle phase (deg)', marks=None, host=None):
    """ONE horizontal colour bar in its OWN axes under the host (the
    first of `axes` unless given), placed by make_axes_locatable: 5% of
    the host's height, 0.6 in below it, so it never rides over the
    host's x label (it did, with fig.colorbar's pad on a one-panel
    figure). The role marks sit under the tick labels, the bar's own
    label under those."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    host = host if host is not None else (axes[0] if isinstance(axes, (list, tuple)) else axes)
    sm = cm.ScalarMappable(norm=PHASE_NORM, cmap=PHASE_CMAP)
    sm.set_array([])
    cax = make_axes_locatable(host).append_axes('bottom', size='5%', pad=0.6)
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_ticks(list(PHASE_TICKS))
    cb.set_ticklabels([f'{t:d}' for t in PHASE_TICKS])
    cb.set_label(label, labelpad=(22 if marks else 4))
    if marks:
        tr = transforms.blended_transform_factory(cax.transData, cax.transAxes)
        for name, deg in marks.items():
            cax.annotate(name, (float(deg) % 360.0, 0), xycoords=tr, xytext=(0, -17), textcoords='offset points',
                         ha='center', va='top', fontsize=8, color='0.35', annotation_clip=False)
    return cb


def line_hist(ax, values, bins=40, range=None, density=True, **kw):
    """A histogram drawn as a line through the bin centres (the user's
    convention: no step outlines). Returns the line or None."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return None
    h, edges = np.histogram(v, bins=bins, range=range, density=density)
    c = 0.5 * (edges[:-1] + edges[1:])
    return ax.plot(c, h, '-', **kw)


# -- the ring in the CLR plane ---------------------------------------------------

def ring_plane(model, name):
    """(plane (2, G), centre (G,), ring2d (T, 2)): the ring's PCA plane in
    the CLR space of `name`'s panel -- the same plane radius() uses."""
    ring = CC.clr(np.exp(model.log_pi(name)))
    cen = ring.mean(0)
    _U, _S, Vt = np.linalg.svd(ring - cen, full_matrices=False)
    plane = Vt[:2]
    return plane, cen, (ring - cen) @ plane.T


def embed_pca(model, name, X, pseudo=0.5):
    """Cells in the ring's plane: (n, 2), the ring's own curve, and the
    fraction of the cells' CLR variance each plane axis explains."""
    plane, cen, ring2d = ring_plane(model, name)
    F = (X + pseudo) / (X + pseudo).sum(1, keepdims=True)
    c = CC.clr(F) - cen
    c2 = c @ plane.T
    total = float(c.var(0).sum()) + 1e-12
    frac = c2.var(0) / total
    return c2, ring2d, frac


def clr_z(X, pseudo=0.5):
    """CLR then per-gene z-score: the input the notebook-style
    embedding (tSNE) sees."""
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


def fig_embeddings(model, name, X, theta_deg, seed=0, max_cells=1500, tsne=True, title=None, marks=None):
    """The cells in the ring's CLR plane (and tSNE when asked), coloured
    by the model's phase. Returns (fig, notes)."""
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = np.arange(n) if n <= max_cells else rng.choice(n, max_cells, replace=False)
    Xs, th = np.asarray(X, float)[idx], np.asarray(theta_deg, float)[idx]
    P, ring2d, frac = embed_pca(model, name, Xs)
    ncol = 2 if tsne else 1
    fig, axes = plt.subplots(1, ncol, figsize=(5.4 * ncol + 0.6, 5.6), squeeze=False)
    axes = axes[0]
    notes = []
    kw = dict(c=th, cmap=PHASE_CMAP, norm=PHASE_NORM, s=6, alpha=0.75, linewidths=0)

    ax = axes[0]
    ax.scatter(P[:, 0], P[:, 1], **kw)
    # the fitted ring as the 72-gon of its grid points (solid), and the
    # same polygon with every vertex's distance from the centre scaled
    # by 0.5 and 1.5 (dashed) -- the user's construction: the ring's
    # radius differs per grid theta and the contours keep that. The
    # scaled copies nest only when the centre lies inside the polygon.
    closed = np.vstack([ring2d, ring2d[:1]])
    for rho, style, lw in ((0.5, '--', 0.8), (1.0, '-', 1.6), (1.5, '--', 0.8)):
        ax.plot(rho * closed[:, 0], rho * closed[:, 1], style, color='k', lw=lw, alpha=0.9)
    ax.plot(0, 0, 'k+', ms=8)
    inside = _point_in_polygon(0.0, 0.0, ring2d)
    ax.set_title('CLR plane of the ring\nsolid: fitted ring; dashed: centre ratio 0.5 and 1.5'
                 + ('' if inside else '\n(centre outside the ring polygon: contours cross)'), fontsize=9)
    ax.set_xlabel(f'PC1 (CLR, {100 * frac[0]:.0f}%)')
    ax.set_ylabel(f'PC2 (CLR, {100 * frac[1]:.0f}%)')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='datalim')

    if tsne:
        ax = axes[1]
        try:
            from sklearn.manifold import TSNE
            Z = clr_z(Xs)
            E = TSNE(n_components=2, perplexity=30, init='pca', random_state=seed).fit_transform(Z)
            ax.scatter(E[:, 0], E[:, 1], **kw)
            ax.set_title(f'tSNE (CLR, z-scored genes; {len(idx)} cells)', fontsize=9)
        except Exception as exc:                                # noqa: BLE001
            notes.append(f'tSNE: {type(exc).__name__}: {exc}')
            ax.set_title('tSNE unavailable', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.27, top=0.82, wspace=0.12)
    phase_colorbar(fig, list(axes), marks=marks)
    finish(fig)
    if title:
        fig.suptitle(wrap_title(title, max(30, int(fig.get_figwidth() * 11))), y=0.995, va='top', fontsize=11)
    return fig, notes


# -- gene profiles by biological role -----------------------------------------

def fold_profiles(model):
    """(T, G) fold change of each gene's expected fraction against its
    own cycle mean, on the grid, from the SHARED profiles."""
    f = model.B @ model.coef.T
    return np.exp(f - f.mean(0, keepdims=True))


def fig_profiles_by_role(model, roles, ncols=1, title=None, marks=None):
    """roles: ordered {role: [gene, ...]} -- one axes per role, one line
    per gene, categorical colours, the shared phase axis."""
    deg = np.degrees(model.grid)
    order = np.argsort(deg)
    fc = fold_profiles(model)
    roles = {r: [g for g in gs if g in model.gi] for r, gs in roles.items()}
    roles = {r: gs for r, gs in roles.items() if gs}
    nrow = max(1, int(np.ceil(len(roles) / ncols)))
    fig, axes = plt.subplots(nrow, ncols, figsize=(6.4 * ncols, 3.6 * nrow), squeeze=False)
    axes = axes.ravel()
    for ax, (role, genes) in zip(axes, roles.items()):
        pal = gene_palette(len(genes))
        for g, c in zip(genes, pal):
            ax.plot(deg[order], fc[order, model.gi[g]], color=c, lw=1.6, label=g)
        ax.axhline(1.0, color='0.7', lw=0.8, ls=':')
        ax.set_ylabel('fraction / cycle mean')
        ax.set_title(f'{role} ({len(genes)} genes)', fontsize=10)
        phase_axis(ax, marks=marks)
        ax.legend(fontsize=8, ncol=2 if len(genes) > 6 else 1, frameon=False)
    for ax in axes[len(roles):]:
        ax.set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.95 if title else 1))
    return finish(fig, title)


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


def fig_spectrum_and_totals(model, groups, q, totals, title=None, marks=None):
    """Two axes: the spectrum per condition (posterior-mean density,
    smoothed as the model's prior is) and the panel's total count per
    cell along the cycle (median line, quartile band) per condition."""
    deg = np.degrees(model.grid)
    order = np.argsort(deg)
    theta = np.degrees(np.angle(q @ np.exp(1j * model.grid))) % 360.0
    names = [g for g in dict.fromkeys(groups)]
    pal = gene_palette(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for g, c in zip(names, pal):
        k = np.asarray(groups) == g
        if k.sum() < 5:
            continue
        w = model.spectrum(q[k])
        ax1.plot(deg[order], w[order] * len(w) / 360.0, color=c, lw=1.6, label=f'{g or "Unassigned"} (n={int(k.sum())})')
        cen, m, lo, hi = binned_stat(theta[k], np.asarray(totals)[k])
        ax2.plot(cen, m, color=c, lw=1.6, label=g or 'Unassigned')
        ax2.fill_between(cen, lo, hi, color=c, alpha=0.12, linewidth=0)
    ax1.set_ylabel('cell density (per degree x 360)')
    ax1.set_title('spectrum per condition', fontsize=10)
    ax1.legend(fontsize=8, frameon=False)
    ax2.set_ylabel('panel total count per cell')
    ax2.set_title('total count along the cycle (median, quartiles)', fontsize=10)
    for ax in (ax1, ax2):
        phase_axis(ax, marks=marks)
    fig.tight_layout(rect=(0, 0, 1, 0.93 if title else 1))
    return finish(fig, title)


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


def fig_anchor_spectra(model, entries, conditions, title=None, marks=None):
    """The named conditions of every experiment on ONE phase axis:
    entries [(experiment, condition, q)], one line per (experiment,
    condition), mean direction marked."""
    deg = np.degrees(model.grid)
    order = np.argsort(deg)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    styles = ['-', '--', ':', '-.']
    exps = list(dict.fromkeys(e for e, _c, _q in entries))
    pal = gene_palette(len(conditions))
    for (exp, cond, q) in entries:
        if cond not in conditions or len(q) < 5:
            continue
        w = model.spectrum(q)
        c = pal[conditions.index(cond)]
        ls = styles[exps.index(exp) % len(styles)]
        lab = f'{cond} (n={len(q)})' if len(exps) == 1 else f'{exp} {cond} (n={len(q)})'
        ax.plot(deg[order], w[order] * len(w) / 360.0, color=c, ls=ls, lw=1.7, label=lab)
        mean = np.degrees(np.angle((q @ np.exp(1j * model.grid)).mean())) % 360.0
        ax.axvline(mean, color=c, ls=ls, lw=0.9, alpha=0.7)
    ax.set_ylabel('cell density (per degree x 360)')
    ax.legend(fontsize=8, frameon=False)
    phase_axis(ax, marks=marks)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return finish(fig, title or 'conditions: spectrum and mean direction')


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


def fig_subpanel(thetas, a, b, title=None, marks=None):
    """theta_a vs theta_b as a 2-D histogram (both 0..360) and the
    distribution of their circular difference as a line."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.0), gridspec_kw={'width_ratios': [1.15, 1]})
    ax1.hist2d(thetas[a], thetas[b], bins=36, range=[[0, 360], [0, 360]], cmap='Greys')
    ax1.plot([0, 360], [0, 360], 'r-', lw=0.8, alpha=0.6)
    phase_axis(ax1, 'x', marks=marks)
    phase_axis(ax1, 'y', marks=marks)
    ax1.set_xlabel(f'phase from {a} (deg)')
    ax1.set_ylabel(f'phase from {b} (deg)')
    ax1.set_aspect('equal')
    d = (thetas[a] - thetas[b] + 180.0) % 360.0 - 180.0
    line_hist(ax2, d, bins=36, range=(-180, 180), color='0.3', lw=1.6)
    ax2.axvline(0, color='r', lw=0.8)
    ax2.set_xlabel(f'phase difference {a} - {b} (deg)')
    ax2.set_ylabel('density')
    ax2.set_title(f'median |diff| {np.median(np.abs(d)):.0f} deg, {np.mean(np.abs(d) <= 45) * 100:.0f}% within 45', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90 if title else 1))
    return finish(fig, title)


# -- the angle as cycle time -------------------------------------------------

def cycle_time_map(w, grid, birth_deg=0.0, growth='uniform'):
    """The phase axis re-scaled by where the CYCLING cells are.

    The fitted angle is a distance in composition, not time: the ring
    covers a stretch of the cycle where composition barely changes
    (G1) with the same number of degrees as one where it changes fast.
    Under the ergodic assumption -- an asynchronous population spends
    cells on each part of the cycle in proportion to the time spent
    there -- the spectrum of the unsynchronized cells IS the clock.

    Both readings of the clock come from the SAME measured spectrum;
    they differ in what they assume about the population:
      'uniform'      every cell counts once: tau is the plain
                     cumulative fraction of cycling cells from birth --
                     a steady-state population that is not growing
                     exponentially;
      'exponential'  an exponentially growing population holds twice
                     as many newborn cells as cells about to divide
                     (age density 2 ln2 / T * 2^(-age/T)); inverting it
                     (age/T = -log2(1 - F/2)) stretches the late cycle.
    Returns (tau(deg) -> [0, 1), tau per grid point, theta(tau) -> deg).
    """
    deg = np.degrees(np.asarray(grid)) % 360.0
    order = np.argsort(deg)
    d, ww = deg[order], np.asarray(w, float)[order]
    ww = ww / ww.sum()
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

    def theta(tau_val):
        x = np.interp(np.asarray(tau_val, float), F_edges, edges)
        return (x - d2[0] + birth_deg) % 360.0
    return tau, tau(deg), theta


def propose_arcs_from_time(w, grid, birth_deg, shares=DEFAULT_PHASE_SHARES, growth='uniform'):
    """Category arcs from cycle-time shares: the phases in order from
    birth with their fractions of the cycle (default G1 45%, S 33%,
    G2/M 22%), mapped back to angles through the clock. Returns
    [{'name', 'start_deg', 'end_deg'}] covering the whole circle."""
    _tau, _tg, theta = cycle_time_map(w, grid, birth_deg, growth)
    names = [n for n, _s in shares]
    vals = np.array([float(s) for _n, s in shares], float)
    vals = vals / vals.sum()
    bounds = np.concatenate([[0.0], np.cumsum(vals)])
    arcs = []
    for i, n in enumerate(names):
        a = float(birth_deg % 360.0) if i == 0 else float(theta(bounds[i]))
        b = float(birth_deg % 360.0) if i + 1 == len(names) else float(theta(bounds[i + 1]))
        arcs.append({'name': n, 'start_deg': a % 360.0, 'end_deg': b % 360.0})
    return arcs


def fig_cycle_time(model, spectra, birth_deg=0.0, training=None, groups_theta=None,
                   anchors=None, title=None, marks=None):
    """spectra: {label: spectrum (T,)} for every condition; training:
    the label whose spectrum is the clock (the cycling population).
    Left: the spectra over the angle; second: tau(angle) for the clock,
    uniform and exponential-growth; third: the ring's speed; right:
    the conditions over tau (groups_theta: {label: theta_deg})."""
    deg = np.degrees(model.grid)
    order = np.argsort(deg)
    names = list(spectra)
    training = training if training in spectra else names[0]
    pal = gene_palette(len(names))
    ncol = 4 if groups_theta else 3
    fig, axes = plt.subplots(1, ncol, figsize=(5.0 * ncol, 4.4))
    ax1, ax2, ax3 = axes[0], axes[1], axes[2]
    for name, c in zip(names, pal):
        w = np.asarray(spectra[name], float)
        ax1.plot(deg[order], w[order] * len(w) / 360.0, color=c, lw=1.8 if name == training else 1.3,
                 label=f'{name}{" (clock)" if name == training else ""}')
    ax1.set_ylabel('cell density (per degree x 360)')
    ax1.set_title('spectrum per condition over the angle', fontsize=10)
    ax1.legend(fontsize=8, frameon=False)
    phase_axis(ax1, marks=marks)

    w = np.asarray(spectra[training], float)
    xs = np.linspace(0, 360, 361)
    tau_u, _tu, _ = cycle_time_map(w, model.grid, birth_deg, 'uniform')
    tau_e, _te, _ = cycle_time_map(w, model.grid, birth_deg, 'exponential')
    ax2.plot(xs, tau_u(xs), color='k', lw=1.6, label='uniform (every cell counts once)')
    ax2.plot(xs, tau_e(xs), color='k', lw=1.0, ls='--', label='exponential growth (age-corrected)')
    ax2.axhline(0.5, color='0.8', lw=0.8, ls=':')
    for lab, dg in (anchors or []):
        ax2.plot([dg], [tau_u(dg)], 'o', color='#D55E00', ms=6)
        ax2.annotate(lab, (dg, tau_u(dg)), textcoords='offset points', xytext=(5, -10), fontsize=8, color='#D55E00')
    ax2.set_ylabel('cycle time fraction tau')
    ax2.set_title(f'angle -> time (birth at {birth_deg:.0f} deg)', fontsize=10)
    ax2.legend(fontsize=8, frameon=False)
    phase_axis(ax2, marks=marks)

    speed = (360.0 / len(w)) / np.maximum(w[order] / w.sum(), 1e-9) / 100.0
    ax3.plot(deg[order], speed, color='k', lw=1.6)
    ax3.set_ylabel('degrees of angle per 1% of cycle time')
    ax3.set_title('how fast the composition moves along the cycle', fontsize=10)
    ax3.set_yscale('log')
    phase_axis(ax3, marks=marks)

    if groups_theta:
        ax4 = axes[3]
        for (name, th), c in zip(groups_theta.items(), gene_palette(len(groups_theta))):
            line_hist(ax4, tau_u(np.asarray(th, float)), bins=40, range=(0, 1), color=c, lw=1.6,
                      label=f'{name} (n={len(th)})')
        ax4.set_xlabel(f'cycle time fraction tau ({training} clock)')
        ax4.set_ylabel('density')
        ax4.set_title('conditions over cycle time', fontsize=10)
        ax4.set_xlim(0, 1)
        ax4.legend(fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9 if title else 1))
    return finish(fig, (title + f' (clock: the {training} spectrum)') if title else None)


# -- the FOV overlay ----------------------------------------------------------------

def fig_fov_overlay(mip, cells, values, mode='phase', categories=None, title=None, marks=None):
    """Cells painted on the FOV's reference MIP: mode 'phase' colours
    each cell by its angle (rainbow), 'category' by its category
    (categorical palette; '' = unassigned in grey). cells: [{'id',
    'area': (ys, xs)}]; values: {cell id: angle or category}; a cell
    with no value is shown in a faint grey."""
    mip = np.asarray(mip, float)
    finite = mip[np.isfinite(mip)]
    lo, hi = (np.percentile(finite, [1, 99.5]) if finite.size else (0.0, 1.0))
    H, W = mip.shape
    rgba = np.zeros((H, W, 4), float)
    cats = list(categories or [])
    lut = {}
    if mode == 'category':
        cats = cats or sorted({v for v in values.values() if v})
        lut = {c: colors.to_rgba(p) for c, p in zip(cats, gene_palette(len(cats)))}
    cmap = plt.get_cmap(PHASE_CMAP)
    n_drawn = 0
    for c in cells:
        cid = int(c['id'])
        ys, xs = c['area']
        ys = np.clip(np.asarray(ys).astype(int), 0, H - 1)
        xs = np.clip(np.asarray(xs).astype(int), 0, W - 1)
        v = values.get(cid)
        if v is None or (mode == 'phase' and not np.isfinite(float(v))):
            col = (0.6, 0.6, 0.6, 0.25)
        elif mode == 'phase':
            r, g, b, _a = cmap(PHASE_NORM(float(v)))
            col = (r, g, b, 0.55)
            n_drawn += 1
        else:
            r, g, b, _a = lut.get(v, (0.5, 0.5, 0.5, 1.0))
            col = (r, g, b, 0.55) if v else (0.5, 0.5, 0.5, 0.3)
            n_drawn += bool(v)
        rgba[ys, xs] = col
    fig, ax = plt.subplots(figsize=(8.5, 9.4))
    ax.imshow(mip, cmap='gray', vmin=lo, vmax=hi, interpolation='nearest')
    ax.imshow(rgba, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.14 if mode == 'phase' else 0.03, top=0.93)
    if mode == 'phase':
        phase_colorbar(fig, [ax], marks=marks)
    else:
        from matplotlib.patches import Patch
        handles = [Patch(color=lut[c], label=c) for c in cats] + [Patch(color=(0.5, 0.5, 0.5), label='Unassigned')]
        ax.legend(handles=handles, loc='upper right', fontsize=8, frameon=True)
    finish(fig)
    t = title or f'{n_drawn} cells coloured by {mode}'
    fig.suptitle(wrap_title(t, max(30, int(fig.get_figwidth() * 11))), y=0.995, va='top', fontsize=11)
    return fig


# -- the Analysis tab's views under a gate -------------------------------------------

def fig_phase_hist(theta_deg, groups=None, mask=None, title=None, marks=None, bins=36):
    """The angle distribution of the gated cells as centre-connected
    lines, one per group (celltype) plus all together; cells outside
    the mask or without a phase are left out."""
    th = np.asarray(theta_deg, float)
    keep = np.isfinite(th) & (np.ones(len(th), bool) if mask is None else np.asarray(mask, bool))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    line_hist(ax, th[keep], bins=bins, range=(0, 360), color='k', lw=1.8, label=f'all gated (n={int(keep.sum())})')
    if groups is not None:
        g = np.asarray(groups, dtype=object)
        names = [x for x in dict.fromkeys(g[keep])]
        for name, c in zip(names, gene_palette(len(names))):
            k = keep & (g == name)
            if k.sum() >= 5:
                line_hist(ax, th[k], bins=bins, range=(0, 360), color=c, lw=1.4, label=f'{name or "Unassigned"} (n={int(k.sum())})')
    ax.set_ylabel('density')
    ax.legend(fontsize=8, frameon=False)
    phase_axis(ax, marks=marks)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return finish(fig, title or 'cell-cycle phase of the gated cells')


def fig_category_hist(categories, groups=None, mask=None, order=None, title=None):
    """Cells per category (bars) for the gated cells, side by side per
    group (celltype) when groups are given; '' reads Unassigned."""
    cat = np.asarray(['Unassigned' if not c else str(c) for c in categories], dtype=object)
    keep = np.ones(len(cat), bool) if mask is None else np.asarray(mask, bool)
    names = list(order) if order else [x for x in dict.fromkeys(cat[keep]) if x != 'Unassigned']
    if 'Unassigned' in set(cat[keep]) and 'Unassigned' not in names:
        names.append('Unassigned')
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(names))
    if groups is None:
        counts = [int(((cat == n) & keep).sum()) for n in names]
        ax.bar(x, counts, color='0.4')
        for xi, n in zip(x, counts):
            ax.text(xi, n, str(n), ha='center', va='bottom', fontsize=8)
    else:
        g = np.asarray(groups, dtype=object)
        gnames = [v for v in dict.fromkeys(g[keep])]
        width = 0.8 / max(1, len(gnames))
        for j, (gn, c) in enumerate(zip(gnames, gene_palette(len(gnames)))):
            counts = [int(((cat == n) & keep & (g == gn)).sum()) for n in names]
            ax.bar(x + (j - (len(gnames) - 1) / 2) * width, counts, width=width, color=c, label=f'{gn or "Unassigned"}')
        ax.legend(fontsize=8, frameon=False)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('cells')
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return finish(fig, title or 'cell-cycle category of the gated cells')


# -- DAPI as the routine verification --------------------------------------------

def fig_dapi_vs_phase(theta_deg, dapi, groups=None, categories=None, order=None, title=None, marks=None,
                      fov=None, training=None, area=None, area_unit='px'):
    """DNA content (a DAPI sum inside the mask) along the phase, per
    condition (binned median, quartiles), and per category as centre-
    connected lines. Each cell's DAPI is first divided by the median of
    its own FOV (fov given): staining and illumination differ per FOV,
    and in an FOV-mode experiment that difference IS the condition
    difference (measured on JP_002: FOV medians span 1.6e7..3.1e7). Then
    the scale is the median of the first category (G1 when the arcs
    start at birth). The title carries the G2/M-over-G1 ratio (about 2
    is the routine pass) and the angle where the DNA content halves --
    division, found by the same step detector as the panel total's drop,
    on the training cells when given."""
    th = np.asarray(theta_deg, float)
    d = np.asarray(dapi, float)
    if fov is not None:
        f = np.asarray(fov)
        d = d.copy()
        for fv in np.unique(f):
            k = (f == fv) & np.isfinite(d)
            med = np.median(d[k]) if k.sum() >= 5 else np.nan
            d[f == fv] = d[f == fv] / med if np.isfinite(med) and med > 0 else np.nan
    ok = np.isfinite(th) & np.isfinite(d)
    cat = None if categories is None else np.asarray(['Unassigned' if not c else str(c) for c in categories], dtype=object)
    names = list(order) if order else ([x for x in dict.fromkeys(cat[ok]) if x != 'Unassigned'] if cat is not None else [])
    ref = d[ok & (cat == names[0])] if (cat is not None and names) else d[ok]
    scale = float(np.median(ref)) if ref.size else 1.0
    dn = d / (scale if scale > 0 else 1.0)
    ncol = 1 + (cat is not None) + (area is not None)
    fig, axes = plt.subplots(1, ncol, figsize=(6.4 * ncol + 0.5, 4.4), squeeze=False)
    ax = axes[0][0]
    g = np.asarray(groups, dtype=object) if groups is not None else np.array(['all'] * len(th), dtype=object)
    conds = [x for x in dict.fromkeys(g[ok])]
    for cond, c in zip(conds, gene_palette(len(conds))):
        k = ok & (g == cond)
        if k.sum() < 10:
            continue
        cen, m, lo, hi = binned_stat(th[k], dn[k], n_bins=24)
        ax.plot(cen, m, color=c, lw=1.6, label=f'{cond or "Unassigned"} (n={int(k.sum())})')
        ax.fill_between(cen, lo, hi, color=c, alpha=0.12, linewidth=0)
    ax.axhline(1.0, color='0.7', lw=0.8, ls=':')
    ax.axhline(2.0, color='0.7', lw=0.8, ls=':')
    ax.set_ylabel(f'DAPI sum above background / median of {names[0] if names else "all"}')
    ax.set_title('DNA content along the cycle (median, quartiles)', fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    phase_axis(ax, marks=marks)
    ratio_txt = ''
    if area is not None:
        # the mask area beside the DNA content (user request): the cell
        # grows through the cycle and halves at division -- a second,
        # DAPI-free reading of the same event, from the segmentation
        ar = np.asarray(area, float)
        oka = np.isfinite(th) & np.isfinite(ar)
        ax3 = axes[0][ncol - 1]
        for cond, c in zip(conds, gene_palette(len(conds))):
            k = oka & (g == cond)
            if k.sum() < 10:
                continue
            cen, m_, lo_, hi_ = binned_stat(th[k], ar[k], n_bins=24)
            ax3.plot(cen, m_, color=c, lw=1.6, label=f'{cond or "Unassigned"}')
            ax3.fill_between(cen, lo_, hi_, color=c, alpha=0.12, linewidth=0)
        ax3.set_ylabel(f'cell mask area ({area_unit})')
        ax3.set_title('mask area along the cycle (median, quartiles)', fontsize=10)
        ax3.legend(fontsize=8, frameon=False)
        phase_axis(ax3, marks=marks)
        tra = oka & (np.asarray(training, bool) if training is not None else np.ones(len(th), bool))
        if tra.sum() >= 200:
            a_, fac_ = CC.total_drop_angle(th[tra], ar[tra], min_cells=200)
            if a_ is not None and fac_ >= 1.2:
                ratio_txt += f'; mask area falls x{fac_:.2f} at {a_:.0f} deg'
                ax3.axvline(a_, color='k', lw=0.9, ls='--')
    if cat is not None:
        ax2 = axes[0][1]
        meds = {}
        allnames = names + (['Unassigned'] if 'Unassigned' in set(cat[ok]) else [])
        hi_x = float(np.nanpercentile(dn[ok], 99.5)) if ok.any() else 3.0
        for name, c in zip(allnames, gene_palette(len(allnames))):
            k = ok & (cat == name)
            if k.sum() < 10:
                continue
            meds[name] = float(np.median(dn[k]))
            line_hist(ax2, dn[k], bins=40, range=(0, max(hi_x, 1.0)), color=c, lw=1.6,
                      label=f'{name} (n={int(k.sum())}, median {meds[name]:.2f})')
        ax2.set_xlabel('DNA content (normalised)')
        ax2.set_ylabel('density')
        ax2.set_title('per category', fontsize=10)
        ax2.legend(fontsize=8, frameon=False)
        g2m = next((n for n in names if 'G2' in n.upper() or n.upper() == 'M'), None)
        if g2m in meds and names[0] in meds and meds[names[0]] > 0:
            ratio_txt += f'; {g2m} / {names[0]} median ratio {meds[g2m] / meds[names[0]]:.2f} (about 2 expected)'
    drop_txt = ''
    tr = ok & (np.asarray(training, bool) if training is not None else np.ones(len(th), bool))
    if tr.sum() >= 200:
        # the DNA content halves at division: the panel total's step
        # detector on the DAPI curve (the inverse of the content, so the
        # 'drop' is the halving)
        a, fac = CC.total_drop_angle(th[tr], dn[tr], min_cells=200)
        if a is not None and fac >= 1.3:
            drop_txt = f'; DNA halves at {a:.0f} deg (x{fac:.2f}) = division'
            ax.axvline(a, color='k', lw=0.9, ls='--')
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return finish(fig, (title or 'DAPI content as the routine verification') + ratio_txt + drop_txt)


def fig_gallery(rows, size=64, title=None, outline='#E69F00'):
    """Example images per phase bin: rows = [(label, [(crop, mask, lo,
    hi), ...])], one tile per example, each on its own FOV scale, the
    cell mask outlined. The user's routine verification by eye -- a
    G2/M row should show larger, brighter nuclei and mitotic figures."""
    ncol = max(len(ex) for _l, ex in rows) if rows else 1
    fig, axes = plt.subplots(len(rows), ncol, figsize=(1.35 * ncol + 1.2, 1.35 * len(rows) + 0.8), squeeze=False)
    for i, (label, examples) in enumerate(rows):
        for j in range(ncol):
            ax = axes[i][j]
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ('top', 'right', 'bottom', 'left'):
                ax.spines[side].set_visible(False)
            if j >= len(examples):
                ax.set_visible(False)
                continue
            crop, mask, lo, hi = examples[j]
            ax.imshow(crop, cmap='gray', vmin=lo, vmax=hi, interpolation='nearest')
            if mask is not None and mask.any():
                ax.contour(mask.astype(float), levels=[0.5], colors=[outline], linewidths=0.6)
        axes[i][0].set_ylabel(label, fontsize=8, rotation=0, ha='right', va='center', labelpad=4)
    if title:
        fig.suptitle(wrap_title(title, 110), y=0.995, va='top', fontsize=10)
    # a tall figure: the title band is a fixed few percent, not the
    # generic suptitle band (which left a blank strip above the tiles)
    fig.subplots_adjust(left=0.16, right=0.995, bottom=0.01, top=0.965 if title else 0.995, wspace=0.04, hspace=0.06)
    return fig


def propose_arcs_from_profiles(model, roles, birth_deg=0.0):
    """Category arcs from the role profiles: a phase begins where the
    mean fold-profile of its indicator genes rises through its cycle
    mean (1.0) on the way to its peak -- S at the S genes' rise, G2/M at
    the G2/M genes' rise -- and G1 begins at birth (the origin when it
    is division). Returns [{'name', 'start_deg', 'end_deg'}] in the
    order G1, S, G2/M around the circle from birth."""
    deg = np.degrees(model.grid) % 360.0
    order = np.argsort(deg)
    d = deg[order]
    fc = fold_profiles(model)[order]

    def rise_before_peak(genes):
        idx = [model.gi[g] for g in genes if g in model.gi]
        if not idx:
            return None
        prof = fc[:, idx].mean(1)
        k = int(np.argmax(prof))
        # walk back from the peak to the last grid point at or below 1
        j = k
        for _ in range(len(d)):
            jj = (j - 1) % len(d)
            if prof[jj] <= 1.0:
                return float(d[j])
            j = jj
        return float(d[k])
    s_start = rise_before_peak([g for g, r in (roles or {}).items() if r == 'S'])
    g_start = rise_before_peak([g for g, r in (roles or {}).items() if r == 'G2/M'])
    if s_start is None or g_start is None:
        raise ValueError('propose from roles needs at least one S and one G2/M gene in the model')
    b = float(birth_deg) % 360.0
    return [{'name': 'G1', 'start_deg': b, 'end_deg': s_start % 360.0},
            {'name': 'S', 'start_deg': s_start % 360.0, 'end_deg': g_start % 360.0},
            {'name': 'G2/M', 'start_deg': g_start % 360.0, 'end_deg': b}]


def dna_curve(theta_deg, dna_norm, n_bins=36, smooth=1):
    """(bin centres, circularly smoothed median DNA content per bin)."""
    th = np.asarray(theta_deg, float) % 360.0
    d = np.asarray(dna_norm, float)
    ok = np.isfinite(th) & np.isfinite(d)
    edges = np.linspace(0.0, 360.0, n_bins + 1)
    b = np.clip(np.digitize(th[ok], edges) - 1, 0, n_bins - 1)
    med = np.array([np.median(d[ok][b == k]) if (b == k).sum() >= 5 else np.nan for k in range(n_bins)])
    if np.isnan(med).any():
        good = np.where(np.isfinite(med))[0]
        x = np.concatenate([good - n_bins, good, good + n_bins])
        med = np.interp(np.arange(n_bins), x, np.tile(med[good], 3))
    if smooth > 0:
        k = np.ones(2 * smooth + 1) / (2 * smooth + 1)
        med = np.convolve(np.concatenate([med[-smooth:], med, med[:smooth]]), k, mode='valid')
    return 0.5 * (edges[:-1] + edges[1:]), med


def propose_arcs_from_dapi(theta_deg, dna, birth_deg=0.0, fov=None, rise=1.15, plateau=0.85):
    """Category arcs from the measured DNA content of the training cells:
    from birth the content is flat (G1, 2N) until it starts to RISE
    (S begins where the smoothed median first exceeds `rise` x the G1
    level), S ends where it reaches `plateau` of the way from the G1
    level to the maximum (G2/M), and G2/M ends at birth. dna: the DAPI
    sum per cell, divided per FOV when fov is given. Returns (arcs,
    info) with the levels used; raises ValueError when no rise is
    found (a flat curve says nothing)."""
    th = np.asarray(theta_deg, float) % 360.0
    d = np.asarray(dna, float).copy()
    if fov is not None:
        f = np.asarray(fov)
        for fv in np.unique(f):
            k = (f == fv) & np.isfinite(d)
            med = np.median(d[k]) if k.sum() >= 5 else np.nan
            d[f == fv] = d[f == fv] / med if np.isfinite(med) and med > 0 else np.nan
    # unsmoothed bins: a box across the division discontinuity smears the
    # 4N plateau into the first 2N bins and reads as a rise
    cen, med = dna_curve(th, d, smooth=0)
    n = len(cen)
    # walk from the first bin that lies wholly AFTER birth: the bin
    # straddling birth mixes the plateau before division with the 2N
    # level after it and read as a rise on the synthetic check
    width = 360.0 / n
    start = int(np.searchsorted(cen - width / 2, birth_deg % 360.0)) % n
    order = [(start + i) % n for i in range(n)]
    curve = med[order]
    g1_level = float(np.median(curve[:max(3, n // 6)]))          # the first sixth of the cycle after birth
    top = float(np.nanmax(curve))
    if top < g1_level * rise:
        raise ValueError(f'the DNA content never rises above {rise:.2f} x its post-birth level ({g1_level:.2f} -> {top:.2f}); '
                         'no S boundary can be read from DAPI')
    # a SUSTAINED rise: two consecutive bins above the threshold
    i_s = next(i for i in range(n - 1) if curve[i] >= g1_level * rise and curve[i + 1] >= g1_level * rise)
    target = g1_level + plateau * (top - g1_level)
    i_g = next((i for i in range(i_s, n) if curve[i] >= target), n - 1)
    half = 180.0 / n
    s_start = (cen[order[i_s]] - half) % 360.0
    g_start = (cen[order[i_g]] - half) % 360.0
    b = float(birth_deg) % 360.0
    arcs = [{'name': 'G1', 'start_deg': b, 'end_deg': s_start},
            {'name': 'S', 'start_deg': s_start, 'end_deg': g_start},
            {'name': 'G2/M', 'start_deg': g_start, 'end_deg': b}]
    return arcs, {'g1_level': g1_level, 'top': top, 'ratio': top / g1_level if g1_level > 0 else float('nan')}


def post_division_end(w, grid, birth_deg=0.0, level=0.5, smooth=5):
    """The angle after birth where the cycling cells' density first
    recovers to `level` x its median: the end of the sparse, fast
    stretch right after division (measured on JP_002: 2% of the cells
    in the first 30 deg, density back to half the median at 30 deg and
    to the median at 40). Returns the angle, or None when the density
    never falls below the level after birth (no such stretch)."""
    deg = np.degrees(np.asarray(grid)) % 360.0
    order = np.argsort(deg)
    d, ww = deg[order], np.asarray(w, float)[order]
    ww = CC._circ_smooth(ww / ww.sum(), smooth)
    med = float(np.median(ww))
    start = int(np.searchsorted(d, birth_deg % 360.0)) % len(d)
    if ww[start] >= level * med:
        return None
    for i in range(1, len(d)):
        j = (start + i) % len(d)
        if ww[j] >= level * med:
            width = 360.0 / len(d)
            return float((d[j] - width / 2) % 360.0)
    return None


def with_post_m(arcs, w, grid, birth_deg=0.0, level=0.5, name='post-M'):
    """Split the arc that starts at birth into 'post-M' (birth -> the
    density recovery) and the rest; the arcs come back unchanged when
    no sparse stretch follows birth or the recovery lies beyond the
    first arc's end."""
    end = post_division_end(w, grid, birth_deg, level)
    if end is None or not arcs:
        return arcs
    first = arcs[0]
    span = (first['end_deg'] - first['start_deg']) % 360.0
    cut = (end - first['start_deg']) % 360.0
    if cut <= 0 or cut >= span:
        return arcs
    return ([{'name': name, 'start_deg': first['start_deg'], 'end_deg': end},
             {'name': first['name'], 'start_deg': end, 'end_deg': first['end_deg']}] + list(arcs[1:]))
