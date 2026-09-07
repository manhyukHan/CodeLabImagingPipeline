"""
Drawing one review page. Pure matplotlib -- no Qt, no file access.

The same function backs the review GUI and the offline PNG dump, so what
a reviewer judges and what anyone else looks at later cannot drift apart.

THE LAYOUT IS NOT DECORATION. Each choice below was made against real
crops from the MAZ store, and the wrong version of each was tried first:

  4 candidates a page, not 6 or 24. At six the axial panel is too small
  to show the PSF's double cone, which is the single feature that
  separates a real emitter from a bright nothing. At 24 the cell overview
  became a postage stamp.

  ZX pinned to the YX width with make_axes_locatable, the way the app's
  own crop viewer pins its colourbar. sharex aligns the DATA limits but
  not the rendered boxes -- YX keeps an equal aspect and shrinks inside
  its cell while ZX fills the width -- so a column in one was not a
  column in the other.

  x runs ACROSS in both panels. It used to run down in ZX, which is the
  app's `bimg` convention, and it read as an inversion every time.

  ZX shows a WINDOW in z, not all ~129 planes. Locked to the YX width,
  the full depth is either seven times taller than it is wide or, once
  squashed, one row of peak.

  The crop is the whole padded rectangle with the mask as an outline, NOT
  the mask alone. With only the cell in frame there is no background for
  the intensity scale to work against, so the cell filled the range and
  every spot looked low-contrast. Candidates are generated from that same
  padded rectangle: the mask is here to cut FOV-scale background down to
  a region worth looking at, not to decide whose spot this is. Masking
  clipped real emitters near a boundary that a segmentation slip or a
  small alignment residual had moved, and those are the examples a
  detector most needs.

  One intensity scale for every panel in the figure. Per-panel scaling
  makes a dim candidate look exactly like a bright one, which is the one
  judgement the reviewer is there to make.

  Colours are the app's own: yellow accepted, blue fitted-but-rejected,
  red anchored-with-no-fit. Someone who has used Spot Localization
  already reads them.
"""
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

HALF = 9           # half-width of a candidate's YX/ZX window, px
ZHALF = 20         # planes shown either side of a candidate
CMAP = 'gray'
# Cyan, not blue, for the gate-rejected ring: on a grayscale stack a
# mid-blue sits close to the dark end of the image and the reviewer has
# to hunt for it, while cyan is a hue the data never has.
PASS_C, REJ_C, NOFIT_C = '#ffd400', '#00c0ff', '#ff3b30'
CHOSEN_C = '#00e676'      # what the REVIEWER accepted, over the engine's colour
PER_PAGE = 4

# Draw order on the cell overview. Candidates are drawn in index order and
# the off-page ones are dense, so without these an off-page circle painted
# over the very candidates being judged.
Z_OFF, Z_ON, Z_ADDED = 3, 6, 8

# Radius of the marker on the cell overview, IN IMAGE PIXELS. A crop is
# only ~80 px across and a busy one carries hundreds of candidates, so a
# ring big enough to be a comfortable click target on screen is a mess on
# the image. At 1.6 the ring traces the emitter -- a real spot is ~1.3 px
# sigma -- rather than enclosing a whole neighbourhood of it.
#
# It is NOT the click tolerance. A click snaps to a candidate within
# spotcheck.app's ADD_SNAP_PX, which stays generous because nobody can
# click to 1.6 px and should not have to.
MARK_R = 1.6


def candidate_colour(cand):
    """The engine's own verdict, in the app's colours."""
    _y, _x, _z, _p, fit_ok, gate, _why = cand
    return PASS_C if (fit_ok and gate) else (REJ_C if fit_ok else NOFIT_C)


def pages_of(n, per_page=PER_PAGE):
    return [list(range(i, min(i + per_page, n))) for i in range(0, n, per_page)]


def _frame(ax, xlabel, ylabel):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=5.6, labelpad=1.0, color='#555')
    ax.set_ylabel(ylabel, fontsize=5.6, labelpad=1.0, color='#555')
    for sp in ax.spines.values():
        sp.set_linewidth(0.4); sp.set_color('#888')


def draw_page(fig, stack, mask, cands, page_ix, header='', accepted=(),
              added=(), page=0, npage=1, per_page=PER_PAGE, n_total=None):
    """Render one page into `fig` (which is cleared first).

    stack     (h, w, depth) raw crop, mask NOT applied
    mask      (h, w) bool, drawn as an outline
    cands     [(y, x, z, p, fit_ok, gate, reason), ...] crop-local
    page_ix   which candidate indices this page shows
    accepted  indices the REVIEWER has accepted so far (any page)
    added     [(y, x), ...] spots the reviewer marked that no candidate covers
    """
    fig.clear()
    st = np.asarray(stack, dtype=float)
    mip = np.nanmax(st, axis=2)
    finite = mip[np.isfinite(mip)]
    vlo, vhi = ((np.percentile(finite, 2.0), np.percentile(finite, 99.9))
                if finite.size else (0.0, 1.0))
    page_ix = list(page_ix)
    accepted = set(accepted)
    ncol = max(1, per_page)

    # THE ARTISTS A TOGGLE TOUCHES, captured as they are made.
    #
    # Pressing 1-4 changes colours and nothing else, but redrawing the
    # page to show that meant clearing the figure and rebuilding nine
    # imshows -- MEASURED 756 ms a keystroke, and a reviewer presses one
    # about four times a page. Over an 870-page assignment that alone was
    # 44 minutes of waiting. restyle() updates these in place instead.
    art = {'fig': fig, 'axm': None, 'overview': {}, 'cards': {},
           'cands': cands, 'page_ix': page_ix, 'header_text': ''}

    outer = fig.add_gridspec(1, 2, width_ratios=[5.4, 1.95 * ncol], wspace=0.16)
    axm = fig.add_subplot(outer[0, 0])
    art['axm'] = axm
    im = axm.imshow(mip, cmap=CMAP, vmin=vlo, vmax=vhi, interpolation='nearest')
    axm.contour(np.asarray(mask, float), levels=[0.5], colors='#00d0a0',
                linewidths=0.9, alpha=0.9, zorder=2)

    # Every candidate is drawn; the ones on THIS page are solid and
    # labelled. Judging four of eleven still needs to show where those
    # four sit among the rest, or "is this the same spot as two pages
    # ago" has no answer.
    for i, c in enumerate(cands):
        y, x = float(c[0]), float(c[1])
        here = i in page_ix
        col = CHOSEN_C if i in accepted else candidate_colour(c)
        # ONE MEANING PER CHANNEL.
        #   colour     what the engine said -- yellow gate-pass, blue
        #              fitted-but-rejected, red anchored-with-no-fit,
        #              green the reviewer kept it
        #   line style ON THIS PAGE or not. Nothing else.
        #
        # The dash used to ALSO mean "gate-rejected", so a blue candidate
        # was dashed even while it was one of the four being judged, and
        # the reader had to work out which of the two things a dash meant
        # at each circle. Colour already says the verdict.
        #
        # And the circle stays visible off-page. It was lw=0.7,
        # alpha=0.30, and a reviewer who could not see it hand-added two
        # spots that were ALREADY candidates -- a duplicate label and a
        # wasted judgement.
        #
        # ON TOP, too. Candidates are drawn in index order, so without an
        # explicit zorder a later off-page circle paints over an on-page
        # one -- and off-page circles are dense. The four being judged
        # must never be the ones underneath.
        circ = Circle((x, y), MARK_R, fill=False, ec=col,
                      lw=1.8 if here else 1.1,
                      alpha=1.0 if here else 0.85,
                      zorder=Z_ON if here else Z_OFF,
                      ls='-' if here else (0, (1.6, 1.0)))
        axm.add_patch(circ)
        # THE NUMBER carries "this one is on your page", the circle
        # carries "there is a candidate here". They are different jobs
        # and want opposite treatments.
        #
        # The circle has to be visible off-page, because a reviewer who
        # cannot see it hand-adds a duplicate of a candidate that already
        # exists -- that happened. The NUMBER does not: making it bold
        # off-page too made every candidate shout equally and it stopped
        # being obvious which four were actually up for judgement. So the
        # number fades while the circle stays, and the dashes still say
        # "not this page" on both.
        #
        # Stroked either way: a coloured glyph sits ON the image, right
        # where the bright pixels are, and plain colour vanished against
        # them.
        num = axm.text(x + MARK_R + 1.2, y - MARK_R - 0.8, str(i + 1), color=col,
                 fontsize=9.5 if here else 7.0,
                 alpha=1.0 if here else 0.40, ha='left', va='top',
                 weight='bold' if here else 'normal',
                 zorder=(Z_ON if here else Z_OFF) + 1,
                 path_effects=[pe.withStroke(
                     linewidth=2.4 if here else 1.6, foreground='black',
                     alpha=0.85 if here else 0.35)])
        art['overview'][i] = (circ, num)
    for (ay, ax_) in added:
        axm.plot(ax_, ay, marker='P', color=CHOSEN_C, ms=7, mew=1.6,
                 mfc='none', ls='none', zorder=Z_ADDED,
                 path_effects=[pe.withStroke(linewidth=3.2,
                                             foreground='black', alpha=0.8)])

    # The page header is RETURNED, not drawn. Matplotlib rasterizes text at
    # ~0.48 ms per glyph on this machine -- MEASURED, and true of a bare
    # figure too, so it is not something this module causes. Those four
    # lines of metadata are ~135 glyphs, which is ~65 ms of every repaint
    # and, because the accepted count sat in them, of every keep-toggle as
    # well. A Qt label above the canvas says the same words for free.
    # Say when a cap is hiding candidates. `cands` is what the reviewer can
    # reach; a busy cell can carry hundreds more that the viewer's per-crop
    # limit dropped, and a header reading "16 candidates" would state the
    # limit as if it were the data.
    n_seen = (f'{len(cands)} candidates' if not n_total or n_total <= len(cands)
              else f'top {len(cands)} of {n_total} candidates')
    header_text = (f'{header}   |   {mip.shape[0]}(y) x {mip.shape[1]}(x) px, '
                   f'{st.shape[2]} planes   |   {n_seen}'
                   f'   |   page {page + 1}/{npage}, showing '
                   f'#{page_ix[0] + 1}-#{page_ix[-1] + 1}')
    art['header_text'] = header_text
    _frame(axm, 'x  →', 'y  ↓')

    cax = axm.inset_axes([1.035, 0.0, 0.030, 1.0])
    cb = fig.colorbar(im, cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=6.0, length=2, pad=1.5)
    cb.set_label('counts — same scale in every panel', fontsize=6.2,
                 labelpad=3)

    grid = outer[0, 1].subgridspec(1, ncol, wspace=0.46)
    for slot, i in enumerate(page_ix):
        c = cands[i]
        y, x, z, p, fit_ok, gate, why = c
        iy, ix, iz = int(round(y)), int(round(x)), int(round(z))
        ya0, ya1 = max(0, iy - HALF), min(st.shape[0], iy + HALF + 1)
        xa0, xa1 = max(0, ix - HALF), min(st.shape[1], ix + HALF + 1)
        chosen = i in accepted
        col = CHOSEN_C if chosen else candidate_colour(c)

        card = grid[0, slot].subgridspec(2, 1, height_ratios=[1.0, 0.34],
                                         hspace=0.72)
        ax1 = fig.add_subplot(card[0])
        ax1.imshow(st[ya0:ya1, xa0:xa1, max(0, min(iz, st.shape[2] - 1))],
                   cmap=CMAP, vmin=vlo, vmax=vhi, interpolation='nearest')
        ccirc = Circle((x - xa0, y - ya0), 2.9, fill=False, ec=col,
                       lw=2.2 if chosen else 1.3)
        ax1.add_patch(ccirc)
        ctitle = ax1.set_title(f'[{slot + 1}]  #{i + 1}  YX @ z={iz}'
                               + ('   ✓ KEEP' if chosen else ''),
                               fontsize=6.6, pad=1.8, color=col, weight='bold')
        _frame(ax1, '', 'y ↓')

        z0 = max(0, iz - ZHALF)
        z1 = min(st.shape[2], iz + ZHALF + 1)
        zx = st[iy, xa0:xa1, z0:z1].T if 0 <= iy < st.shape[0] else np.zeros((1, 1))
        ax2 = make_axes_locatable(ax1).append_axes(
            'bottom',
            size=f'{100.0 * zx.shape[0] / max(1, zx.shape[1]):.0f}%',
            pad=0.10, sharex=ax1)
        ax2.imshow(zx, cmap=CMAP, vmin=vlo, vmax=vhi, interpolation='nearest')
        # A marker ring, not a Circle patch: a patch lives in data
        # coordinates and would swallow the very spot it points at
        # whenever the two axes scale differently.
        cmark, = ax2.plot(x - xa0, z - z0, 'o', mfc='none', mec=col, ms=8,
                          mew=2.2 if chosen else 1.3)
        _frame(ax2, 'x →', 'z ↓')

        kind = ('PASS' if gate else 'reject') if fit_ok else 'NO FIT'
        axt = fig.add_subplot(card[1]); axt.axis('off')
        ctext = axt.text(0, 1.0, f'p={p:.3f}  {kind}', fontsize=6.5, color=col,
                         va='top', ha='left', family='monospace', weight='bold')
        body = (f'y{y:5.1f} x{x:5.1f} z{z:5.1f}\n'
                f'z shown {z0}-{z1 - 1}')
        if why:
            body += '\n' + str(why)[:26]
        axt.text(0, 0.60, body, fontsize=5.6, color='#555', va='top',
                 ha='left', family='monospace', linespacing=1.5)
        art['cards'][i] = (ccirc, ctitle, cmark, ctext, slot, iz)
    return art


def mutable_artists(art):
    """The artists a keep-toggle can actually change -- the blit set.

    NOT every artist restyle() touches. restyle() walks the whole
    overview because that is cheaper than working out which entries
    moved, but a toggle only ever changes the four candidates ON the
    page: an off-page circle keeps the engine's colour whatever the
    reviewer presses. Blitting is paid per artist, and a real cell
    carries a few hundred candidates -- MEASURED 208 ms per toggle
    blitting all of them, against a 520 ms full draw. Restricting the
    set to the page is what makes blitting worth doing at all.
    """
    out = []
    for i in art['page_ix']:
        pair = art['overview'].get(i)
        if pair is not None:
            out.extend(pair)
    for ccirc, ctitle, cmark, ctext, _slot, _iz in art['cards'].values():
        out.extend((ccirc, ctitle, cmark, ctext))
    return out


def restyle(art, accepted, added=()):
    """Update only what a keep-toggle changes: colours, widths, the tick.
    Returns the figure, already re-styled.

    The accepted count is NOT here. It used to live in the axes title,
    which made every keystroke redraw 135 glyphs at ~0.48 ms each; it now
    lives in a Qt label the app updates for free.

    THE POINT IS WHAT IT DOES NOT DO. draw_page clears the figure and
    builds nine imshows plus a colourbar; a toggle changes no pixel of
    any of them. MEASURED before this existed: 756 ms per keystroke, four
    keystrokes to a page, 44 minutes of an 870-page assignment spent
    watching a figure redraw itself identically.

    Called with an `art` from draw_page for the page currently shown. A
    newly ADDED spot needs a new artist, so that still goes through
    draw_page -- it is rare, and a click is already a slow gesture.
    """
    accepted = set(accepted)
    for i, (circ, num) in art['overview'].items():
        here = i in art['page_ix']
        col = CHOSEN_C if i in accepted else candidate_colour(art['cands'][i])
        circ.set_edgecolor(col)
        num.set_color(col)
    for i, (ccirc, ctitle, cmark, ctext, slot, iz) in art['cards'].items():
        chosen = i in accepted
        col = CHOSEN_C if chosen else candidate_colour(art['cands'][i])
        ccirc.set_edgecolor(col)
        ccirc.set_linewidth(2.2 if chosen else 1.3)
        cmark.set_markeredgecolor(col)
        cmark.set_markeredgewidth(2.2 if chosen else 1.3)
        ctitle.set_color(col)
        ctitle.set_text(f'[{slot + 1}]  #{i + 1}  YX @ z={iz}'
                        + ('   ✓ KEEP' if chosen else ''))
        ctext.set_color(col)
    return art['fig']
