# -*- coding: utf-8 -*-
"""Intensity and morphology flags for segmented nuclei, and the drop set.

Segmentation finds objects; it does not decide which of them are cells
worth keeping. That decision was being made by hand in the GUI, once per
experiment, with the rules living in whoever's head ran it -- so a batch
run could not reproduce it and two people could not agree on it.

THE FIVE QUANTITIES, all PER FOV:

    area_rel  = area / this FOV's median nucleus area
    solidity  = area / convex-hull area
    dapi_z    = z-score of the mean stain inside the mask, within the FOV
    border    = the mask reaches within `edge_px` of the frame
    (small / large / irregular / dim / bright follow from the first three)

Per FOV, not pooled, and that is not a detail: staining level and focus
differ between fields, so a pooled threshold flags whole fields rather
than cells. Measured on the Lamin set (5443 nuclei, 39 FOVs): pooled,
the flagged and clean ranges OVERLAP on every driver; per FOV the cut is
clean in 37/38, 37/37, 18/18 and 37/37 fields respectively.

WHICH FLAGS DROP IS NOT DECIDED HERE. `flag()` reports; `to_drop()`
applies a caller's choice. The Lamin set dropped dim + irregular +
border; the first MAZ/SRRM1 set dropped border + irregular + large and
KEPT dim, having looked at both overlays. A default that suited one
would have silently mis-gated the other.

THE FLAGS MOVE WHEN CELLS ARE REMOVED. Every driver is relative to the
FOV's own population, so re-flagging after a removal gives different
numbers -- correctly. Flag once, drop once; do not iterate to a fixed
point.
"""
import numpy as np

#: thresholds measured off the Lamin set's own flags table, where each
#: separates cleanly inside a field
SMALL_REL, LARGE_REL = 0.5, 2.0
IRREGULAR_SOLIDITY = 0.9
DIM_Z, BRIGHT_Z = -1.5, 3.0
EDGE_PX = 5

FLAGS = ('small', 'large', 'irregular', 'dim', 'bright', 'border')


def _mask_metrics(ys, xs, frame_shape, edge_px):
    from skimage import measure
    ys = np.asarray(ys, int)
    xs = np.asarray(xs, int)
    h, w = frame_shape
    m = np.zeros((int(ys.max() - ys.min()) + 3,
                  int(xs.max() - xs.min()) + 3), np.uint8)
    m[ys - ys.min() + 1, xs - xs.min() + 1] = 1
    p = measure.regionprops(m)[0]
    return dict(n_px=int(len(ys)), solidity=float(p.solidity),
                centroid_y=float(ys.mean()), centroid_x=float(xs.mean()),
                border=bool(ys.min() < edge_px or xs.min() < edge_px
                            or ys.max() >= h - edge_px
                            or xs.max() >= w - edge_px))


def flag(cells, image, px_um, frame_shape=None, edge_px=EDGE_PX):
    """[{id, area_um2, solidity, dapi_z, area_rel, ...flags}] for ONE FOV.

    `image` is the 2D stain the cells were found in, read in the same
    frame as the masks -- the mean inside each mask is what dapi_z
    scores. `cells` are the container's own dicts (area = (ys, xs)).
    """
    if not len(cells):
        return []
    shape = frame_shape or getattr(image, 'shape', (1024, 1024))
    rows = []
    for c in cells:
        ys = np.asarray(c['area'][0], int)
        xs = np.asarray(c['area'][1], int)
        r = _mask_metrics(ys, xs, shape, edge_px)
        r['id'] = int(c['id'])
        r['area_um2'] = r['n_px'] * float(px_um) ** 2
        r['mean_stain'] = (float(np.asarray(image)[ys, xs].mean())
                           if image is not None else float('nan'))
        rows.append(r)
    area = np.array([r['area_um2'] for r in rows])
    stain = np.array([r['mean_stain'] for r in rows])
    med = np.median(area) or 1.0
    sd = np.nanstd(stain)
    for r in rows:
        r['area_rel'] = r['area_um2'] / med
        r['dapi_z'] = ((r['mean_stain'] - np.nanmean(stain)) / sd
                       if sd > 0 else 0.0)
        r['small'] = bool(r['area_rel'] < SMALL_REL)
        r['large'] = bool(r['area_rel'] > LARGE_REL)
        r['irregular'] = bool(r['solidity'] < IRREGULAR_SOLIDITY)
        r['dim'] = bool(r['dapi_z'] < DIM_Z)
        r['bright'] = bool(r['dapi_z'] > BRIGHT_Z)
    return rows


def parse_drop(text):
    """'border,irregular,large' -> ('border', 'irregular', 'large').

    An unknown name is an error, not a silent no-op: a typo that drops
    nothing looks exactly like a gate the user chose to leave open.
    """
    names = tuple(t.strip() for t in str(text or '').split(',') if t.strip())
    bad = [n for n in names if n not in FLAGS]
    if bad:
        raise ValueError('unknown cell flag(s) %s; known: %s'
                         % (', '.join(bad), ', '.join(FLAGS)))
    return names


def to_drop(rows, drop):
    """Ids to remove, given the flags to honour. Empty `drop` removes
    nothing -- flagging without dropping is a real, useful state."""
    names = parse_drop(drop) if isinstance(drop, str) else tuple(drop or ())
    return [r['id'] for r in rows if any(r.get(n) for n in names)]


def summarise(rows, drop=()):
    names = parse_drop(drop) if isinstance(drop, str) else tuple(drop or ())
    counts = {f: int(sum(bool(r.get(f)) for r in rows)) for f in FLAGS}
    n_drop = len(to_drop(rows, names))
    return '%d nuclei; %s; dropping %s -> %d removed, %d kept' % (
        len(rows), ', '.join('%s %d' % (f, counts[f]) for f in FLAGS
                             if counts[f]) or 'no flags',
        '+'.join(names) if names else 'nothing', n_drop, len(rows) - n_drop)
