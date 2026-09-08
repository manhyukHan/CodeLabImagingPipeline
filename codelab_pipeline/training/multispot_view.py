"""
One pillar, every candidate in it, for a person to judge.

THE SECOND MODEL PROPOSES, A PERSON DISPOSES. psf-match returns whatever
is in a 15 x 15 x full-depth pillar, and 38% of confirmed-spot pillars
hold more than one match while 13% hold three or more. Whether those
extra matches are real loci, out-of-focus neighbours or non-specific
binding is not a question the images answer on their own -- it is the
same question the first review answered for single spots, asked again.

THREE THINGS THIS PAGE MUST DO, and they were learned the hard way:

  MARK THE POSITIONS ON THE YX MIP. Without it a reviewer is judging
  axial slices with no idea where in the cell they came from, and two
  candidates four pixels apart look like two unrelated pictures.

  CUT EACH ZX AT ITS OWN y. A neighbour a few pixels off the pillar's
  centre row does not appear in the centre row's slice at all -- a real
  second object read as flat background in the first figure drawn from
  this, and only showed itself once each match got its own cut.

  ONE SCALE, WITH A BAR BESIDE IT. The scale is taken from the CENTRE ZX
  slice: 15 x depth, mostly background, so its percentiles are set by
  background and an emitter stands out of them. A scale taken from the
  15 x 15 lateral plane is set by the spot's own neighbourhood and
  flattens everything. The bar is there so a reviewer can read INTENSITY
  and not only shape -- two candidates that look alike at a glance may
  be an order of magnitude apart, and that is a thing worth seeing.

Units are sigma above the crop's own background (mode plus the left
half-width), the same units the classifier reads, so a number here means
the same thing as a number there.
"""
import numpy as np
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

CMAP = 'gray'
MARK_C = '#00c0ff'          # a match psf-match proposes
CHOSEN_C = '#00e676'        # one the reviewer has kept
SEED_C = '#ffd400'          # the candidate the pillar was centred on
ZHALF = 16                  # planes either side of a match in its close-up
# Radius on the YX MIP, IN IMAGE PIXELS. The pillar is 15 px across and
# can hold seven candidates; a ring sized to be a comfortable click
# target covers the picture it is pointing at. At 1.4 the ring is about
# three pixels across, which traces an emitter (~1.3 px sigma) instead
# of enclosing its whole neighbourhood.
MARK_R = 1.4


def pillar_at(crop, y, x, half=7, background=0.0, sigma=1.0):
    """A (2*half+1, 2*half+1, depth) pillar, PADDED rather than skipped.

    Returns (pillar, (y_origin, x_origin), pad_fraction). Clamping the
    slice instead would drop every candidate within `half` of the crop's
    edge -- 24.6% of the labelled spots in the MP58 RNA bundle, which is
    a quarter of the data leaving without being counted.
    """
    st = np.asarray(crop, float)
    h, w, d = st.shape
    n = 2 * int(half) + 1
    iy, ix = int(round(float(y))), int(round(float(x)))
    out = np.full((n, n, d), float(background))
    y0, y1 = max(0, iy - half), min(h, iy + half + 1)
    x0, x1 = max(0, ix - half), min(w, ix + half + 1)
    if y1 > y0 and x1 > x0:
        out[y0 - (iy - half):y1 - (iy - half),
            x0 - (ix - half):x1 - (ix - half), :] = st[y0:y1, x0:x1, :]
    inside = max(0, y1 - y0) * max(0, x1 - x0)
    pad = 1.0 - inside / float(n * n)
    return (out - background) / max(float(sigma), 1e-9), (iy - half, ix - half), pad


def scale_of(pillar, lo_pct=1.0, hi_pct=99.7):
    """The shared display range, taken from the CENTRE ZX slice."""
    p = np.asarray(pillar, float)
    zx = p[p.shape[0] // 2, :, :].T
    if not np.isfinite(zx).any():
        return 0.0, 1.0
    return (float(np.nanpercentile(zx, lo_pct)),
            float(np.nanpercentile(zx, hi_pct)))


def draw_pillar(fig, pillar, hits, header='', accepted=(), seed_yx=None,
                zhalf=ZHALF, pad_frac=0.0):
    """Render one pillar and all its candidates. Returns an `art` dict.

    `hits` are (y, x, z, p) in PILLAR coordinates -- psf-match searches
    the pillar, so its answers are pillar-local and a caller adds the
    pillar's own origin to reach the crop. `accepted` are indices into
    `hits` the reviewer has kept.
    """
    fig.clear()
    p = np.asarray(pillar, float)
    ny, nx, nz = p.shape
    lo, hi = scale_of(p)
    hits = list(hits)
    accepted = set(accepted)
    ncol = max(1, len(hits))

    # THE MIP KEEPS ITS SIZE whatever the candidate count. Scaling it
    # against ncol made it a postage stamp on the pillar that most needed
    # reading -- seven candidates is exactly when a reviewer has to see
    # where they sit relative to one another.
    outer = fig.add_gridspec(1, 2, width_ratios=[2.4, 1.0 * ncol],
                             wspace=0.16)
    art = {'fig': fig, 'axm': None, 'cax': None, 'cards': {},
           'scale': (lo, hi), 'hits': hits}

    axm = fig.add_subplot(outer[0, 0])
    art['axm'] = axm
    im = axm.imshow(p.max(axis=2), cmap=CMAP, vmin=lo, vmax=hi,
                    interpolation='nearest')
    axm.set_xticks([]); axm.set_yticks([])
    for sp in axm.spines.values():
        sp.set_visible(False)
    if seed_yx is not None:
        axm.plot(seed_yx[1], seed_yx[0], '+', color=SEED_C, ms=13, mew=1.8,
                 zorder=4)
    for i, (hy, hx, hz, hp) in enumerate(hits):
        col = CHOSEN_C if i in accepted else MARK_C
        c = Circle((hx, hy), MARK_R, fill=False, ec=col, lw=1.6, zorder=6)
        axm.add_patch(c)
        # A white stroke under the number: on a grayscale stack a bare
        # glyph disappears into whatever it lands on.
        import matplotlib.patheffects as _pe
        t = axm.annotate(str(i + 1), (hx, hy), color=col, fontsize=9,
                         weight='bold', xytext=(5, 4),
                         textcoords='offset points', zorder=7,
                         path_effects=[_pe.withStroke(linewidth=2.2,
                                                      foreground='black',
                                                      alpha=0.85)])
        art['cards'][i] = {'mark': c, 'num': t}
    art['header'] = header + (f'   ·   {100 * pad_frac:.0f}% of the pillar '
                              f'ran off the crop and is padded'
                              if pad_frac > 0.01 else '')
    fig.suptitle(art['header'], fontsize=10, y=0.985, color='#333')

    # THE BAR, so intensity is readable and not only shape.
    cax = make_axes_locatable(axm).append_axes('right', size='5%', pad=0.06)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7, length=2, pad=1.5)
    cb.set_label('σ above the crop background', fontsize=7.5, labelpad=3)
    art['cax'] = cb.ax

    grid = outer[0, 1].subgridspec(1, ncol, wspace=0.22)
    for i, (hy, hx, hz, hp) in enumerate(hits):
        iy = int(np.clip(round(hy), 0, ny - 1))
        z0 = int(np.clip(round(hz) - zhalf, 0, max(nz - 2 * zhalf - 1, 0)))
        z1 = min(nz, z0 + 2 * zhalf + 1)
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(p[iy, :, z0:z1].T, cmap=CMAP, vmin=lo, vmax=hi,
                  aspect='auto', interpolation='nearest')
        m, = ax.plot(hx, hz - z0, 'o', mfc='none', ms=13, mew=1.9,
                     mec=CHOSEN_C if i in accepted else MARK_C)
        ax.set_xticks([]); ax.set_yticks([])
        col = CHOSEN_C if i in accepted else MARK_C
        for sp in ax.spines.values():
            sp.set_color(col); sp.set_linewidth(1.4)
        ttl = ax.set_title(f'{i + 1}   z={hz:.1f}   p={hp:.2f}\n'
                           f'ZX at its own y={hy:.1f}',
                           fontsize=8.5, pad=4, color=col)
        art['cards'][i].update(ax=ax, marker=m, title=ttl, z0=z0)
    return art


def restyle(art, accepted):
    """Recolour for a changed set of keeps. No pixel is redrawn."""
    accepted = set(accepted)
    for i, card in art['cards'].items():
        col = CHOSEN_C if i in accepted else MARK_C
        card['mark'].set_edgecolor(col)
        card['num'].set_color(col)
        if 'marker' in card:
            card['marker'].set_markeredgecolor(col)
            card['title'].set_color(col)
            for sp in card['ax'].spines.values():
                sp.set_color(col)
    return art['fig']


def mutable_artists(art):
    """The blit set: everything restyle() can touch."""
    out = []
    for card in art['cards'].values():
        out.extend([card['mark'], card['num']])
        if 'marker' in card:
            out.extend([card['marker'], card['title']])
            out.extend(card['ax'].spines.values())
    return out
