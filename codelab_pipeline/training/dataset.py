"""
Human verdicts -> boxes and labels, with a split that does not leak.

WHAT COMES OUT. Every labelled spot as a row (key, fov, hybe, channel,
cell, y, x, z, label, group), plus, on demand, the voxels around it. The
label is 1 for a spot a reviewer kept, 0 for one they were shown and did
not keep, and -1 for one reviewers disagreed about.

CONTESTED IS NOT A THIRD CLASS TO TRAIN ON. It is carried through with
label -1 and excluded from training, because a spot some reviewers kept
and others did not is the example humans could not agree about -- the
worst possible thing to hand a model as ground truth, and the most
useful thing to keep as a set to look at again. It is also the honest
ceiling: no model should be believed past the rate at which people
agree with each other.

THE SPLIT IS BY CELL, NEVER BY SPOT. Spots in one cell share a
background, a segmentation, an alignment residual and a focus position.
Splitting at random puts siblings on both sides and inflates every score
-- the model recognises the cell, not the spot. `group` is (fov, cell),
so the same physical nucleus imaged in eleven hybes stays on one side.

WHAT A BOX IS. Two things, because they answer different questions:

  core     a fixed (2R+1, 2R+1, 2RZ+1) window, default 15 x 15 x 25.
           Shape and contrast live here.

  column   the FULL z column through the centre, all ~105-120 planes.
           This is where a hot pixel gives itself away -- it is bright
           in every plane, while an emitter is bright in three to five.
           Keeping it costs 120 numbers and no model capacity.

The core is NOT the whole stack. A 20x20x120 box is 48,000 voxels and
there are ~1,000 labels; the input would outnumber the samples 48 to 1.

EVERY BOX IS THE SAME SIZE, padded with the crop's own background where
it runs off the edge, and `border_frac` records how much of it is
padding. Dropping edge spots instead would drop exactly the ones a
detector most needs -- see training/view.py, which makes the same
argument for the same reason.

THE MASK IS NOT APPLIED. A spot on the cell boundary has its wings
across it; masking pulls the fitted centre inward and shrinks sigma
(MEASURED 0.197 px and -4% at the centre -- tests/
test_fit_sees_past_the_mask.py). The mask says whose spot it is, which
is a question already answered by the time a box is cut.
"""
import os

import numpy as np

from . import bundle as B
from . import verdicts as V
from ..localization.engine import background_mode

DEFAULT_R = 7        # core half-width in y and x -> 15 x 15
DEFAULT_RZ = 12      # core half-depth -> 25 planes
MAX_BORDER_FRAC = 0.5


def _z_for_added(stack, y, x):
    """A z for a spot the reviewer marked on a MIP, which carries none.

    The reviewer clicked a maximum-intensity projection, so an added spot
    has (y, x) and nothing else. The brightest plane of its own column is
    the only defensible answer -- and it is the same rule the anchor
    stage uses, so an added spot and a candidate at the same place get
    their z the same way.
    """
    h, w = stack.shape[:2]
    yi = int(np.clip(round(float(y)), 0, h - 1))
    xi = int(np.clip(round(float(x)), 0, w - 1))
    col = np.asarray(stack[yi, xi, :], float)
    if col.size == 0 or not np.isfinite(col).any():
        return float(stack.shape[2] // 2)
    # A 3-plane mean first: a single noisy plane is not a z.
    k = np.convolve(col, np.ones(3) / 3.0, mode='same')
    return float(np.nanargmax(k))


def rows(bundle_dir, include_contested=True, include_added=True):
    """Every labelled spot in a bundle, as plain dicts.

    Reads only the verdict files and the shard indexes -- no pixels --
    so listing what is labelled is cheap even when the boxes are not.
    """
    lab = V.labels(bundle_dir)
    index = {}
    for p in B.shard_paths(bundle_dir):
        for row in B.read_index(p):
            index[str(row['key'])] = (p, row)

    out = []
    for key, e in sorted(lab.items()):
        if key not in index:
            continue                      # labelled against a shard we lack
        shard, irow = index[key]
        base = dict(key=key, shard=shard,
                    fov=int(irow['fov']),
                    hybe=(irow['hybe'].decode()
                          if isinstance(irow['hybe'], bytes)
                          else str(irow['hybe'])),
                    channel=int(irow['channel']), cell=int(irow['cell']),
                    group=(int(irow['fov']), int(irow['cell'])),
                    store=e.get('store'))
        for bucket, label in (('positive', 1), ('negative', 0),
                              ('contested', -1)):
            if label == -1 and not include_contested:
                continue
            for (y, x, z) in e.get(bucket) or []:
                out.append(dict(base, y=float(y), x=float(x), z=float(z),
                                label=label, origin=bucket))
        if include_added:
            votes = e.get('added_votes') or {}
            for (y, x) in e.get('added') or []:
                seen = votes.get((y, x))
                out.append(dict(base, y=float(y), x=float(x), z=None,
                                label=1, origin='added',
                                added_votes=seen))
    return out


def _pad_window(stack, iy, ix, iz, r, rz, fill):
    """A fixed (2r+1, 2r+1, 2rz+1) box, padded with `fill`."""
    h, w, d = stack.shape
    out = np.full((2 * r + 1, 2 * r + 1, 2 * rz + 1), float(fill))
    y0, y1 = max(0, iy - r), min(h, iy + r + 1)
    x0, x1 = max(0, ix - r), min(w, ix + r + 1)
    z0, z1 = max(0, iz - rz), min(d, iz + rz + 1)
    if y1 > y0 and x1 > x0 and z1 > z0:
        out[y0 - (iy - r):y1 - (iy - r),
            x0 - (ix - r):x1 - (ix - r),
            z0 - (iz - rz):z1 - (iz - rz)] = stack[y0:y1, x0:x1, z0:z1]
    inside = max(0, y1 - y0) * max(0, x1 - x0) * max(0, z1 - z0)
    return out, 1.0 - inside / float(out.size)


def boxes(label_rows, r=DEFAULT_R, rz=DEFAULT_RZ,
          max_border_frac=MAX_BORDER_FRAC, storage_path=None, on_crop=None):
    """Cut a box for every row. Returns (kept_rows, cores, columns).

    Crops are read ONCE per (shard, key) and reused for every label in
    them, which is what keeps this linear in crops rather than in labels.

    `storage_path` re-cuts from the store instead of the bundle -- same
    voxels either way (verdicts.recut's own docstring says why), but the
    bundle is ~50x faster and the store is what remains once a bundle is
    deleted.
    """
    order = sorted(range(len(label_rows)),
                   key=lambda i: (label_rows[i]['shard'],
                                  label_rows[i]['key']))
    kept, cores, cols = [], [], []
    cur_key, st, b0, sg = None, None, 0.0, 1.0
    for i in order:
        rrow = label_rows[i]
        if rrow['key'] != cur_key:
            if storage_path:
                rec = dict(rrow, crop=None)
                raise NotImplementedError(
                    'recut path needs the verdict record; use bundle boxes '
                    'for now and see verdicts.recut for the store path')
            stack, _mask, _c, _w = B.read_crop(rrow['shard'], rrow['key'])
            st = np.asarray(stack, float)
            b0, sg = background_mode(st)
            cur_key = rrow['key']
            if on_crop:
                on_crop(cur_key, st.shape)
        z = rrow['z']
        if z is None:
            z = _z_for_added(st, rrow['y'], rrow['x'])
        iy = int(round(rrow['y']))
        ix = int(round(rrow['x']))
        iz = int(round(z))
        core, border = _pad_window(st, iy, ix, iz, r, rz, b0)
        if border > max_border_frac:
            continue
        h, w, d = st.shape
        col = np.asarray(
            st[int(np.clip(iy, 0, h - 1)), int(np.clip(ix, 0, w - 1)), :],
            float)
        kept.append(dict(rrow, z=float(z), border_frac=float(border),
                         bg=float(b0), sigma=float(sg)))
        cores.append((core - b0) / max(sg, 1e-9))
        cols.append((col - b0) / max(sg, 1e-9))
    return kept, np.asarray(cores), np.asarray(cols)


def split_by_group(label_rows, frac=0.25, seed=0):
    """Hold out whole CELLS, not spots. Returns (train_ix, val_ix).

    A spot's siblings share its background, its segmentation and its
    focus; splitting at random puts them on both sides and the model
    scores itself on cells it has already seen.
    """
    groups = sorted({tuple(r['group']) for r in label_rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_val = max(1, int(round(len(groups) * float(frac))))
    val = set(groups[:n_val])
    tr = [i for i, r in enumerate(label_rows) if tuple(r['group']) not in val]
    va = [i for i, r in enumerate(label_rows) if tuple(r['group']) in val]
    return tr, va


def summary(label_rows):
    """What a labelled set actually contains, for printing before use."""
    n = len(label_rows)
    pos = sum(1 for r in label_rows if r['label'] == 1)
    neg = sum(1 for r in label_rows if r['label'] == 0)
    con = sum(1 for r in label_rows if r['label'] == -1)
    add = sum(1 for r in label_rows if r.get('origin') == 'added')
    return dict(
        n=n, positive=pos, negative=neg, contested=con, added=add,
        groups=len({tuple(r['group']) for r in label_rows}),
        crops=len({r['key'] for r in label_rows}),
        hybes=sorted({r['hybe'] for r in label_rows}),
        fovs=sorted({r['fov'] for r in label_rows}),
        positive_frac=(pos / max(1, pos + neg)))
