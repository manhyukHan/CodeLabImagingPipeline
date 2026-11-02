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

AND THE NUMBERS ARE STORED, ALL OF THEM. annotate() writes every
quantity flag() measured onto the cells themselves, prefixed `gate_`.
They used to be computed, used to drop, logged as counts and thrown
away -- so "why was this cell kept" could only be answered by
re-reading the stain and re-flagging a population that no longer
exists. A cell carries arbitrary properties through save() and the
columnar store, and the exporter turns every undeclared field into a
column of its own, so storing them costs one float per cell per field
and nothing else.

Every field, not the drivers in use: which flags an experiment drops is
its own decision (the Lamin set dropped dim+irregular+border, the first
MAZ/SRRM1 set border+irregular+large), and a store that kept only the
drivers somebody chose today answers the next question with a re-run
instead of a read. The per-cell provenance says what the numbers are
relative to, because `area_rel` and `dapi_z` are relative to the FOV's
population AT FLAG TIME and that population is about to shrink.
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
    scores. `cells` are the container's own ACell objects, which is what
    CellContainer.get_cells hands out and what annotate() has to mutate;
    this read them as dicts and raised TypeError on every real call.
    """
    if not len(cells):
        return []
    shape = frame_shape or getattr(image, 'shape', (1024, 1024))
    rows = []
    for c in cells:
        ys = np.asarray(c.area[0], int)
        xs = np.asarray(c.area[1], int)
        r = _mask_metrics(ys, xs, shape, edge_px)
        r['id'] = int(c.id)
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


#: every stored field carries this prefix. The exporter spells out
#: `area_um2` and `solidity` of its own (from the mask and the voxel
#: size), so an unprefixed field would quietly overwrite a column that
#: means something else.
PREFIX = 'gate_'

#: what flag() measures, stored in this order. All of it -- see the
#: module docstring for why not just the drivers in use.
MEASURED = ('n_px', 'area_um2', 'area_rel', 'solidity', 'dapi_z',
            'mean_stain', 'centroid_y', 'centroid_x')


def annotate(cells, rows, drop=(), stain=''):
    """Write what the gate measured onto the cells, and return how many
    cells were written.

    CALL THIS BEFORE THE REMOVAL. `rows` covers every nucleus the FOV
    had, the dropped ones included, and those cells are about to be
    deleted -- their numbers go with them, which is why the overlay PNG
    is drawn from the same rows and is the only record of what left.

    Three provenance fields go on every cell beside the measurements:

        gate_drop            the flags honoured, '' when the gate was
                             left open (flagging without dropping is a
                             real state, and then these numbers are a
                             pure description, not a reason)
        gate_stain           where dapi_z and mean_stain were measured
        gate_n_population    how many nuclei the FOV held when flagged

    The last one is not decoration. `area_rel` is this cell's area over
    the population's MEDIAN area and `dapi_z` is its z-score within the
    population, so both are statements about a population -- the one
    about to shrink by `drop`. Without its size the stored numbers
    cannot be compared with a later re-flagging of the same field.
    """
    names = parse_drop(drop) if isinstance(drop, str) else tuple(drop or ())
    by_id = {int(r['id']): r for r in rows}
    label = '+'.join(names)
    n = 0
    for c in cells:
        r = by_id.get(int(c.id))
        if r is None:
            continue
        props = {PREFIX + k: _scalar(r.get(k)) for k in MEASURED}
        props.update({PREFIX + f: bool(r.get(f)) for f in FLAGS})
        props[PREFIX + 'drop'] = label
        props[PREFIX + 'stain'] = str(stain or '')
        props[PREFIX + 'n_population'] = int(len(rows))
        c.set_metadata(**props)
        n += 1
    return n


def _scalar(v):
    """A plain Python number, never a numpy one: the columnar store types
    a property column off the first value it sees."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    return float(v)


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
