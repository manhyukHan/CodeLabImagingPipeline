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

ONE PICTURE PER QUESTION, AND NEITHER OF THEM IS A NUMBER. A reviewer
turning 1,159 pillars judges by looking, and a page that has to be parsed
is a page that costs ten seconds instead of two. So the two things being
asked are split across the two panels, each on the scale that answers it:

  BRIGHTNESS LIVES ON THE MIP, on ONE scale with a bar, so every match is
  a ring on the same picture and "that one is faint" is a glance. Its
  ceiling is the BRIGHTEST MATCH -- see mip_scale, and the 56% of rings
  that used to sit on identical white.

  SHAPE LIVES ON THE ZX CARDS, each on ITS OWN scale. This is where a
  shared scale was actively wrong: MEASURED on 487 cards from 250
  confirmed pillars, the peak of 38% of them fell BELOW the shared
  ceiling, so they rendered grey on grey with no white anywhere, and the
  question a card exists to answer -- does this thing have an emitter's
  axial profile, or is it a smear -- cannot be answered in the third of
  the ramp they were given.

      rank        n     own hi (median)     peak / shared ceiling
      seed      237        8.30 sigma            1.40x
      2         152        4.36                  0.77x
      3          64        5.00                  0.75x
      4          24        3.22                  0.64x
      5           7        3.04                  0.50x

  Nothing is lost by splitting them, because a grey level was never how a
  person compared two brightnesses anyway -- the MIP with its rings is.

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
UNSURE_C = '#9e9e9e'        # one the reviewer declined to judge
UNSURE_LS = (0, (3, 2))
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


def mip_scale(mip, values=(), lo_pct=1.0, headroom=1.05, hi_pct=99.7):
    """The MIP's display range: background floor, brightest-MATCH ceiling.

    THE CEILING IS A MATCH, not a percentile, and that is the whole
    reason this exists. The MIP is a max projection, so its bright pixels
    run far above any percentile of a single slice -- MEASURED on 487
    matches, a ceiling taken from the centre ZX slice saturated 56% of
    them to pure white and 100% of the seeds. Every one of those rings
    sat on the same white, so the picture that is supposed to say which
    match is bright said nothing at all.

        ceiling rule                       saturated   faintest match
        centre ZX p99.7                        56%          --
        the MIP's own p99.9                    40%         0.30
        the MIP's own max                      40%         0.29
        brightest match x1.05                   0%         0.29 (p10 0.14)

    Anchoring the top at the brightest match puts every match inside the
    ramp by construction: on a pillar with more than one, the faintest
    sits at 0.29 of it against the brightest near 0.95. A non-match pixel
    brighter than any match does clip, and that is worth seeing -- it is
    something psf-match did not propose and the A key exists for.
    """
    m = np.asarray(mip, float)
    if not np.isfinite(m).any():
        return 0.0, 1.0
    lo = float(np.nanpercentile(m, lo_pct))
    vals = [float(v) for v in values if np.isfinite(v)]
    hi = (max(vals) * float(headroom) if vals
          else float(np.nanpercentile(m, hi_pct)))
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def card_scale(window, lo_pct=1.0, hi_pct=99.7):
    """One ZX card's own range, plus its peak. (lo, hi, peak).

    NO FLOOR ON THE SPAN, deliberately. Stretching a faint card to full
    contrast does make its noise grainier, and that is the honest
    picture: MEASURED, the faintest cards still span 2.2 sigma at the
    5th percentile, so nothing here is a pure-background window being
    magnified into apparent structure. The peak rides along and gets
    printed, which is what tells a reviewer that grainy card is grainy
    because it is faint.
    """
    w = np.asarray(window, float)
    if not np.isfinite(w).any():
        return 0.0, 1.0, 0.0
    return (float(np.nanpercentile(w, lo_pct)),
            float(np.nanpercentile(w, hi_pct)),
            float(np.nanmax(w)))


def draw_pillar(fig, pillar, hits, header='', accepted=(), seed_yx=None,
                zhalf=ZHALF, pad_frac=0.0, added=(), unsure=()):
    """Render one pillar and all its candidates. Returns an `art` dict.

    `hits` are (y, x, z, p) in PILLAR coordinates -- psf-match searches
    the pillar, so its answers are pillar-local and a caller adds the
    pillar's own origin to reach the crop. `accepted` are indices into
    `hits` the reviewer has kept. `added` are (y, x) the reviewer marked
    where psf-match found nothing.

    THE HEADER IS RETURNED, NOT DRAWN. Glyph rasterization is the single
    largest cost in this window -- MEASURED 0.48 ms per glyph, 65% of a
    page draw -- and a header is the one piece of text that changes every
    page while none of the pixels under it do. The app puts it in a Qt
    label; view.py does the same for the same reason.
    """
    fig.clear()
    p = np.asarray(pillar, float)
    ny, nx, nz = p.shape
    hits = list(hits)
    flat = p.max(axis=2)
    lo, hi = mip_scale(flat, [flat[int(np.clip(round(h[0]), 0, ny - 1)),
                              int(np.clip(round(h[1]), 0, nx - 1))]
                              for h in hits])
    accepted = set(accepted)
    unsure = set(unsure) - accepted
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
    im = axm.imshow(flat, cmap=CMAP, vmin=lo, vmax=hi,
                    interpolation='nearest')
    axm.set_xticks([]); axm.set_yticks([])
    for sp in axm.spines.values():
        sp.set_visible(False)
    if seed_yx is not None:
        axm.plot(seed_yx[1], seed_yx[0], '+', color=SEED_C, ms=13, mew=1.8,
                 zorder=4)
    for i, (hy, hx, hz, hp) in enumerate(hits):
        col = _colour(i, accepted, unsure)
        c = Circle((hx, hy), MARK_R, fill=False, ec=col, lw=1.6, zorder=6,
                   ls=UNSURE_LS if i in unsure else 'solid')
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
    # A SPOT THE REVIEWER MARKED WHERE PSF-MATCH FOUND NOTHING. Drawn as
    # a cross rather than a ring so it cannot be mistaken for a match
    # that happens to be kept -- the two mean different things to
    # whatever reads the labels, and a page that draws them alike invites
    # a reviewer to add a duplicate of something already on screen.
    art['added'] = [axm.plot(ax_, ay_, 'x', color=CHOSEN_C, ms=9, mew=1.8,
                             zorder=5)[0]
                    for (ay_, ax_) in added]
    art['header_text'] = header + (
        f'   ·   {100 * pad_frac:.0f}% of the pillar ran off the crop '
        f'and is padded' if pad_frac > 0.01 else '')

    # THE BAR, so intensity is readable and not only shape.
    cax = make_axes_locatable(axm).append_axes('right', size='5%', pad=0.06)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7, length=2, pad=1.5)
    # NAMED AS THE MIP'S, because it is only the MIP's. Every ZX card is
    # on its own range and prints its own peak; a bar that looked like it
    # described them would be worse than no bar.
    cb.set_label('σ above the crop background — MIP only', fontsize=7.5,
                 labelpad=3)
    art['cax'] = cb.ax

    grid = outer[0, 1].subgridspec(1, ncol, wspace=0.22)
    for i, (hy, hx, hz, hp) in enumerate(hits):
        iy = int(np.clip(round(hy), 0, ny - 1))
        z0 = int(np.clip(round(hz) - zhalf, 0, max(nz - 2 * zhalf - 1, 0)))
        z1 = min(nz, z0 + 2 * zhalf + 1)
        ax = fig.add_subplot(grid[0, i])
        win = p[iy, :, z0:z1].T
        clo, chi, cpeak = card_scale(win)
        ax.imshow(win, cmap=CMAP, vmin=clo, vmax=chi,
                  aspect='auto', interpolation='nearest')
        col = _colour(i, accepted, unsure)
        m, = ax.plot(hx, hz - z0, 'o', mfc='none', ms=13, mew=1.9, mec=col)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(col); sp.set_linewidth(1.4)
            sp.set_linestyle(UNSURE_LS if i in unsure else 'solid')
        # NO NUMBER FOR THE BRIGHTNESS. It was printed here for one
        # revision and it was the wrong answer to the right worry: a
        # reviewer turning 1,159 pillars reads pictures, not a table, and
        # a page that has to be parsed is a page that takes ten seconds.
        # The MIP carries brightness -- every match is a ring on the one
        # scaled picture -- and these cards carry shape. Each panel
        # answers one question and neither asks to be read.
        ttl = ax.set_title(f'{i + 1}   z={hz:.1f}   p={hp:.2f}'
                           + ('   ? UNSURE' if i in unsure else '')
                           + f'\nZX at its own y={hy:.1f}',
                           fontsize=8.5, pad=4, color=col)
        art['cards'][i].update(ax=ax, marker=m, title=ttl, z0=z0,
                               scale=(clo, chi), peak=cpeak)
    return art


def _colour(i, accepted, unsure):
    return (CHOSEN_C if i in accepted else
            UNSURE_C if i in unsure else MARK_C)


def restyle(art, accepted, unsure=()):
    """Recolour for a changed set of keeps. No pixel is redrawn."""
    accepted = set(accepted)
    unsure = set(unsure) - accepted
    for i, card in art['cards'].items():
        col = _colour(i, accepted, unsure)
        ls = UNSURE_LS if i in unsure else 'solid'
        card['mark'].set_edgecolor(col)
        card['mark'].set_linestyle(ls)
        card['num'].set_color(col)
        if 'marker' in card:
            card['marker'].set_markeredgecolor(col)
            card['title'].set_color(col)
            # THE WORD, NOT ONLY THE COLOUR. Grey against cyan is a fine
            # distinction on a grayscale page, and a reviewer who has
            # just pressed Shift+3 needs to see that it landed.
            head, _, tail = card['title'].get_text().partition('\n')
            head = head.split('   ? UNSURE')[0]
            card['title'].set_text(
                head + ('   ? UNSURE' if i in unsure else '')
                + ('\n' + tail if tail else ''))
            for sp in card['ax'].spines.values():
                sp.set_color(col); sp.set_linestyle(ls)
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
