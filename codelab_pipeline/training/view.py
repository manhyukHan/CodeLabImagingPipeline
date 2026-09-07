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
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

HALF = 9           # half-width of a candidate's YX/ZX window, px
ZHALF = 20         # planes shown either side of a candidate
CMAP = 'gray'
PASS_C, REJ_C, NOFIT_C = '#ffd400', '#2f6dff', '#ff3b30'
CHOSEN_C = '#00e676'      # what the REVIEWER accepted, over the engine's colour
PER_PAGE = 4


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
              added=(), page=0, npage=1, per_page=PER_PAGE):
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

    outer = fig.add_gridspec(1, 2, width_ratios=[5.4, 1.95 * ncol], wspace=0.16)
    axm = fig.add_subplot(outer[0, 0])
    im = axm.imshow(mip, cmap=CMAP, vmin=vlo, vmax=vhi, interpolation='nearest')
    axm.contour(np.asarray(mask, float), levels=[0.5], colors='#00d0a0',
                linewidths=0.9, alpha=0.9)

    # Every candidate is drawn; the ones on THIS page are solid and
    # labelled. Judging four of eleven still needs to show where those
    # four sit among the rest, or "is this the same spot as two pages
    # ago" has no answer.
    for i, c in enumerate(cands):
        y, x = float(c[0]), float(c[1])
        here = i in page_ix
        col = CHOSEN_C if i in accepted else candidate_colour(c)
        solid = (i in accepted) or (c[4] and c[5])
        axm.add_patch(Circle((x, y), 3.8, fill=False, ec=col,
                             lw=2.0 if here else 0.7,
                             alpha=1.0 if here else 0.30,
                             ls='-' if solid else (0, (2.2, 1.3))))
        axm.text(x + 4.8, y - 4.2, str(i + 1), color=col,
                 fontsize=8.0 if here else 6.0,
                 alpha=1.0 if here else 0.35, ha='left', va='top',
                 weight='bold')
    for (ay, ax_) in added:
        axm.plot(ax_, ay, marker='P', color=CHOSEN_C, ms=9, mew=1.4,
                 mfc='none', ls='none')

    n_acc = len(accepted)
    axm.set_title(f'{header}\n{mip.shape[0]}(y) x {mip.shape[1]}(x) px, '
                  f'{st.shape[2]} planes   |   {len(cands)} candidates\n'
                  f'page {page + 1}/{npage}   —   '
                  f'showing #{page_ix[0] + 1}–#{page_ix[-1] + 1}   '
                  f'|   accepted so far: {n_acc}'
                  + (f' (+{len(added)} added)' if added else ''),
                  fontsize=8.4)
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
        ax1.add_patch(Circle((x - xa0, y - ya0), 2.9, fill=False, ec=col,
                             lw=2.2 if chosen else 1.3))
        ax1.set_title(f'[{slot + 1}]  #{i + 1}  YX @ z={iz}'
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
        ax2.plot(x - xa0, z - z0, 'o', mfc='none', mec=col, ms=8,
                 mew=2.2 if chosen else 1.3)
        _frame(ax2, 'x →', 'z ↓')

        kind = ('PASS' if gate else 'reject') if fit_ok else 'NO FIT'
        axt = fig.add_subplot(card[1]); axt.axis('off')
        axt.text(0, 1.0, f'p={p:.3f}  {kind}', fontsize=6.5, color=col,
                 va='top', ha='left', family='monospace', weight='bold')
        body = (f'y{y:5.1f} x{x:5.1f} z{z:5.1f}\n'
                f'z shown {z0}-{z1 - 1}')
        if why:
            body += '\n' + str(why)[:26]
        axt.text(0, 0.60, body, fontsize=5.6, color='#555', va='top',
                 ha='left', family='monospace', linespacing=1.5)
    return fig
