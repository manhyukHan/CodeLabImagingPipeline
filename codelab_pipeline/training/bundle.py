"""
The review bundle: cell crops plus anchor-fit candidates, self-contained.

WHY A BUNDLE AND NOT THE STORE ITSELF. The pixels are already in the
store, so duplicating them needs a reason, and the reason is the SHAPE of
the read, not where the file sits. Pulling one cell-sized window out of a
266 MB gzip-chunked stack costs a measured 673 ms median (40 crops over
40 FOVs and 36 hybes, on a FIXED drive -- the real NAS is worse). The
same pixels as a contiguous dataset here: 12 ms. That is 56x, and it is
the difference between 18.7 hours of pure I/O for 100k reviewed
candidates and 20 minutes.

The duplication is smaller than it sounds. A median crop is 74x75x129
uint16 = 1.37 MB raw; masked outside the cell and gzipped it is 0.34 MB,
because the mask fills only 40% of the bbox. 100k crops is ~34 GiB --
less than ONE FOV's stack files (42.5 GiB) and 1.6% of that experiment's
2.07 TiB. And 34 GiB copies to a reviewer's local disk, which is the
whole point; 2 TiB does not.

WHAT IS AND IS NOT IN HERE. Pixels, the cell mask, and the candidates
anchor-fit proposed, with the numbers needed to rank and trace them.
NOT human verdicts: a bundle is READ-ONLY once written, so ten people can
review the same file at once and their answers are merged from separate
verdict files afterwards. Nothing here is ever rewritten.

CANDIDATES ARE NOT SPOTS. They are deliberately generous -- most of them
are junk, and the junk is the point, because a candidate a person
rejects is a labelled hard negative and that is the more informative half
of a training set. `gate_pass` records what the PRODUCTION gate would
have said, which is information, not a filter: nothing is dropped for
failing it.

WHY NOT ASpot. ASpot has no field for "has this been 3D-localized, and
did it pass" -- main_window._z_status_text carries it as a
session-transient Python attribute that is explicitly not persisted, so
it cannot cross a process boundary, let alone a machine. The extractor
therefore never builds an ASpot at all: it goes crop -> engine ->
candidates, and the four states live here as real columns.

LAYOUT (one file per shard, sharded by FOV so extraction is FOV-major
through one pool and a reviewer can be handed whole FOVs):

    /index                 one row per crop: key, fov, hybe, channel,
                           cell, y0, x0, h, w, depth, n_candidates
    /crops/<key>           uint16 (h, w, depth), 0 outside the cell mask
    /masks/<key>           uint8  (h, w), 1 inside the cell
    /cands/<key>           CANDIDATE_DTYPE, best-p first
    /reasons               the gate's own words; candidates hold an index
    attrs on the file      store, modality, channel, engine + its params,
                           when it was built, and the commit it was built
                           from
"""
import json
import os
import time

import numpy as np
import h5py

FORMAT_VERSION = 1

CANDIDATE_DTYPE = np.dtype([
    ('y', '<f4'), ('x', '<f4'), ('z', '<f4'),   # crop-local, sub-pixel
    ('p', '<f4'),                               # engine ranking, (0, 1]
    ('fit_ok', 'u1'),                           # 0 = anchor only, the fit failed
    ('gate_pass', 'u1'),                        # what PRODUCTION would have said
    ('reason', '<u2'),                          # index into /reasons, 0 = none
])

INDEX_DTYPE = np.dtype([
    ('key', h5py.string_dtype()),
    ('fov', '<i4'), ('hybe', h5py.string_dtype()), ('channel', '<i4'),
    ('cell', '<i4'),
    ('y0', '<i4'), ('x0', '<i4'),               # crop origin in the hybe's frame
    ('h', '<i4'), ('w', '<i4'), ('depth', '<i4'),
    ('n_candidates', '<i4'),
])


def crop_key(fov, hybe, channel, cell):
    """The one name a crop has, everywhere. Flat, not nested: a reviewer
    addresses crops by key and never lists an HDF5 group, which is what
    keeps opening a 30k-crop shard fast."""
    return f'fov{int(fov):03d}|{hybe}|ch{int(channel)}|cell{int(cell)}'


class BundleWriter:
    """Write one shard. Not reopenable -- a bundle is written once.

    `.part` + os.replace, and never delete-then-write, for the same
    reason ingestion does it: an interrupted extraction that left a
    half-file in place would be indistinguishable from a complete one at
    every later step. See CLAUDE.md.
    """

    def __init__(self, path, meta=None, compression=4):
        self.path = str(path)
        self.tmp = self.path + '.part'
        self.compression = int(compression)
        self.meta = dict(meta or {})
        self._rows = []
        self._reasons = ['']            # index 0 is "no reason"
        self._reason_ix = {'': 0}
        self._f = None

    def __enter__(self):
        d = os.path.dirname(os.path.abspath(self.tmp))
        if d:
            os.makedirs(d, exist_ok=True)
        self._f = h5py.File(self.tmp, 'w')
        self._f.create_group('crops')
        self._f.create_group('masks')
        self._f.create_group('cands')
        return self

    def _reason_index(self, reason):
        r = str(reason or '')
        if r not in self._reason_ix:
            self._reason_ix[r] = len(self._reasons)
            self._reasons.append(r)
        return self._reason_ix[r]

    def add(self, fov, hybe, channel, cell, stack, mask, y0, x0, candidates):
        """One crop and its candidates.

        stack       (h, w, depth); NaN or masked-out pixels are stored as 0
                    -- uint16 with a separate mask is half the size of
                    float32 with NaN, and the mask is what a viewer needs
                    to draw the boundary anyway.
        candidates  [(y, x, z, p, fit_ok, gate_pass, reason), ...] in
                    crop-local coordinates, best first.
        """
        key = crop_key(fov, hybe, channel, cell)
        arr = np.asarray(stack)
        arr = np.where(np.isfinite(arr), arr, 0)
        arr = np.clip(arr, 0, 65535).astype(np.uint16)
        m = np.asarray(mask).astype(np.uint8)

        kw = dict(compression='gzip', compression_opts=self.compression,
                  shuffle=True)
        self._f['crops'].create_dataset(key, data=arr, **kw)
        self._f['masks'].create_dataset(key, data=m, **kw)

        rows = np.zeros(len(candidates), dtype=CANDIDATE_DTYPE)
        for i, c in enumerate(candidates):
            rows[i] = (float(c[0]), float(c[1]), float(c[2]), float(c[3]),
                       1 if c[4] else 0, 1 if c[5] else 0,
                       self._reason_index(c[6]))
        self._f['cands'].create_dataset(key, data=rows, **kw) if len(rows) \
            else self._f['cands'].create_dataset(key, shape=(0,), dtype=CANDIDATE_DTYPE)

        self._rows.append((key, int(fov), str(hybe), int(channel), int(cell),
                           int(y0), int(x0), int(arr.shape[0]), int(arr.shape[1]),
                           int(arr.shape[2]), int(len(rows))))
        return key

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                idx = np.array(self._rows, dtype=INDEX_DTYPE)
                self._f.create_dataset('index', data=idx, compression='gzip')
                self._f.create_dataset(
                    'reasons', data=np.array(self._reasons, dtype=object),
                    dtype=h5py.string_dtype())
                self._f.attrs['format_version'] = FORMAT_VERSION
                self._f.attrs['written_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
                self._f.attrs['n_crops'] = len(self._rows)
                self._f.attrs['meta'] = json.dumps(self.meta, default=str)
        finally:
            self._f.close()
            self._f = None
        if exc_type is None:
            os.replace(self.tmp, self.path)
        elif os.path.exists(self.tmp):
            os.remove(self.tmp)          # a partial shard is worse than none
        return False


def read_meta(path):
    """(meta dict, n_crops, format_version) without touching the pixels."""
    with h5py.File(path, 'r') as f:
        return (json.loads(f.attrs.get('meta', '{}')),
                int(f.attrs.get('n_crops', 0)),
                int(f.attrs.get('format_version', 0)))


def read_index(path):
    """Every crop's row, as plain dicts. This is what a reviewer's queue
    is built from -- it is small, so it loads whole, and no group listing
    is ever needed."""
    with h5py.File(path, 'r') as f:
        idx = f['index'][...]
        out = []
        for r in idx:
            d = {n: r[n] for n in idx.dtype.names}
            for k, v in list(d.items()):
                if isinstance(v, bytes):
                    d[k] = v.decode()
                elif isinstance(v, np.integer):
                    d[k] = int(v)
            out.append(d)
        return out


def read_crop(path, key):
    """(stack uint16 (h,w,depth), mask uint8 (h,w), candidates recarray).

    Candidates come back with `reason` already resolved to text in a
    parallel list, because a caller that has to re-open the file to find
    out why something was rejected will simply not bother.
    """
    with h5py.File(path, 'r') as f:
        stack = f['crops'][key][...]
        mask = f['masks'][key][...]
        cands = f['cands'][key][...]
        reasons = [r.decode() if isinstance(r, bytes) else str(r)
                   for r in f['reasons'][...]]
    words = [reasons[int(c['reason'])] if int(c['reason']) < len(reasons) else ''
             for c in cands]
    return stack, mask, cands, words


def shard_paths(bundle_dir):
    """Every shard in a bundle directory, in FOV order."""
    d = str(bundle_dir)
    return sorted(os.path.join(d, n) for n in os.listdir(d)
                  if n.endswith('.h5') and not n.endswith('.part.h5'))
