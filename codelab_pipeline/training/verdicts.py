"""
Human verdicts: append-only, one file per reviewer, never inside the bundle.

WHY NOT IN THE BUNDLE. Ten people review the same shards. A bundle that
were also the answer sheet would have to be locked, copied per person, or
merged under contention -- so it is READ-ONLY, and every reviewer writes
their own JSONL beside it. Merging is then a concatenation, and two
people reviewing the same page is a feature (it measures agreement)
rather than a write conflict.

A VERDICT MUST OUTLIVE THE BUNDLE, AND MUST NOT REFER TO IT BY INDEX.
This recorded only `accepted: [0, 2]` -- positions in a candidate list --
until 2026-09-07. That was wrong in two ways, both of which fail
silently.

  The pixels. A label is training data only if the image it labels can be
  produced, and a verdict file holds no pixels. It does not need to: a
  crop is a plain window read, so (store, fov, hybe, channel, y0, x0, h,
  w) re-cuts exactly the same voxels with no algorithm in the way. Every
  record carries that, which is what makes the bundle a cache rather than
  the archive. See recut().

  The identity. A candidate's INDEX is an artefact of the extractor: the
  list is sorted fitted-first then by p, after a 2 px dedup. Change the
  dedup, change the engine, or let a scipy version shift one fit by an
  epsilon, and index 2 is a different spot. Nothing raises -- the labels
  simply point at the wrong places from then on. So a record stores each
  shown candidate's own COORDINATE, and keeps the index only as
  provenance.

  Coordinates are crop-local (y, x, z), the frame the reviewer actually
  saw. Add (y0, x0) for the hybe's own full frame -- see full_frame().

APPEND-ONLY, one JSON object per line, flushed and fsynced per commit. A
reviewer's machine dying mid-session costs the page in progress and
nothing else. Re-reviewing a page appends a second line; the later one
wins, and both survive so a disagreement with an earlier pass stays
visible.

WHY EVERY REVIEWED PAGE IS RECORDED, EVEN AN EMPTY ONE. The default is
REJECT: a reviewer clicks only the real spots, so most pages end with
nothing accepted. If silence meant rejection, "reviewed and found
nothing" and "never opened" would be the same record, and the difference
is the whole value of a negative -- an unreviewed page is unlabelled
data, a reviewed-and-empty page is four confirmed hard negatives.

A line:
  {"key": "fov001|Hyb_109|ch635|cell3",
   "fov": 1, "hybe": "Hyb_109", "channel": 635, "cell": 3,
   "store": "G:/Seonghyeok/2025-11-30-MP58/RNA",
   "crop": {"y0": 0, "x0": 241, "h": 50, "w": 78, "depth": 105},
   "page": 0,
   "shown": [{"i": 0, "y": 12.4, "x": 30.1, "z": 61.7,
              "p": 0.85, "fit_ok": 1, "gate": 1, "keep": 1}, ...],
   "added": [{"y": 41.2, "x": 17.8}],
   "reviewer": "shj", "at": "2026-09-07T14:03:11", "seconds": 6.2,
   "bundle": "fov001__Hyb_109__c000.h5"}
"""
import json
import os
import time

FILENAME_FMT = 'verdicts_{reviewer}.jsonl'

# Two coordinates within this many pixels are the same spot when tallying
# votes. Well under a real emitter's ~1.3 px sigma, and loose enough that
# a bundle regenerated after a library upgrade still aggregates with the
# labels made against the old one.
SAME_SPOT_PX = 0.5


def path_for(bundle_dir, reviewer):
    return os.path.join(str(bundle_dir), FILENAME_FMT.format(reviewer=reviewer))


class VerdictLog:
    """Append-only writer + the resume index for one reviewer."""

    def __init__(self, bundle_dir, reviewer):
        self.path = path_for(bundle_dir, reviewer)
        self.reviewer = str(reviewer)
        self._done = None

    def done_pages(self):
        """{(key, page)} already committed BY THIS REVIEWER.

        Read once and cached: it is what lets a session resume where it
        stopped, and re-reading per page would make the queue O(n^2) in a
        file that grows all session.
        """
        if self._done is None:
            self._done = set()
            for rec in read_log(self.path):
                self._done.add((rec.get('key'), int(rec.get('page', 0))))
        return self._done

    def page_verdict(self, key, page):
        """This reviewer's latest verdict for one page, as
        {'accepted': [i, ...], 'added': [(y, x), ...]} -- or None.

        The review app needs this to REDISPLAY a page it already judged.
        Without it, stepping back showed the page with every keep wiped
        from the screen while the file still held them, and one more
        Space then superseded the real verdict with an empty one, turning
        the reviewer's positives into confirmed negatives.

        Latest wins, matching merge(): a page re-judged is a correction.
        """
        found = None
        for rec in read_log(self.path):
            if rec.get('key') == key and int(rec.get('page', 0)) == int(page):
                found = rec
        if found is None:
            return None
        return {'accepted': [int(e['i']) for e in (found.get('shown') or [])
                             if e.get('keep')],
                'added': [(float(a['y']), float(a['x']))
                          for a in (found.get('added') or [])]}

    def commit(self, row, page, page_ix, cands, accepted, added=(),
               seconds=None, bundle=None, store=None):
        """Record one judged page.

        row       the bundle index row for this crop (key, fov, hybe,
                  channel, cell, y0, x0, h, w, depth)
        page_ix   the candidate indices shown on this page
        cands     the crop's FULL candidate list, so each shown one is
                  stored by COORDINATE rather than by position
        accepted  the indices the reviewer kept
        added     [(y, x), ...] crop-local, spots no candidate covered

        Everything needed to re-cut the pixels and to identify each label
        goes into the line. Nothing refers to the bundle except as
        provenance.
        """
        keep = set(int(i) for i in accepted)
        shown = []
        for i in page_ix:
            i = int(i)
            if i >= len(cands):
                continue
            y, x, z, p, fit_ok, gate = cands[i][:6]
            shown.append({'i': i,
                          'y': round(float(y), 3), 'x': round(float(x), 3),
                          'z': round(float(z), 3), 'p': round(float(p), 4),
                          'fit_ok': int(fit_ok), 'gate': int(gate),
                          'keep': 1 if i in keep else 0})
        rec = {'key': str(row['key']),
               'fov': int(row['fov']), 'hybe': str(row['hybe']),
               'channel': int(row['channel']), 'cell': int(row['cell']),
               'crop': {'y0': int(row['y0']), 'x0': int(row['x0']),
                        'h': int(row['h']), 'w': int(row['w']),
                        'depth': int(row['depth'])},
               'page': int(page),
               'shown': shown,
               'added': [{'y': round(float(a), 3), 'x': round(float(b), 3)}
                         for (a, b) in added],
               'reviewer': self.reviewer,
               'at': time.strftime('%Y-%m-%dT%H:%M:%S')}
        if store:
            rec['store'] = str(store)
        if seconds is not None:
            rec['seconds'] = round(float(seconds), 2)
        if bundle:
            rec['bundle'] = str(bundle)
        d = os.path.dirname(os.path.abspath(self.path))
        if d:
            os.makedirs(d, exist_ok=True)
        # newline='\n' explicitly: this file is read by whoever collects
        # the labels, possibly not on Windows, and one JSON object per
        # line is the whole contract.
        with open(self.path, 'a', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(rec) + '\n')
            f.flush()
            os.fsync(f.fileno())        # a page committed is a page kept
        self.done_pages().add((rec['key'], rec['page']))
        return rec


def full_frame(rec, entry):
    """One shown candidate's (y, x, z) in the hybe's own full frame.

    Crop-local plus the crop origin. z takes no offset: a crop carries
    the entire depth.
    """
    c = rec['crop']
    return (float(entry['y']) + c['y0'], float(entry['x']) + c['x0'],
            float(entry['z']))


def recut(rec, storage_path=None):
    """The exact voxels this verdict was made against, from the store.

    This is what lets a bundle be deleted once its labels are collected:
    a crop is a pure window read -- no anchoring, no engine, no fit -- so
    given the same store this returns the same array the reviewer saw,
    element for element.
    """
    import h5py
    from ..io import paths
    sp = storage_path or rec.get('store')
    if not sp:
        raise ValueError('this verdict records no store; pass storage_path')
    c = rec['crop']
    p = paths.stack_path(sp, int(rec['fov']), str(rec['hybe']))
    with h5py.File(p, 'r') as f:
        ds = f[f"/stack/ch{int(rec['channel'])}"]
        return ds[c['y0']:c['y0'] + c['h'], c['x0']:c['x0'] + c['w'], :]


def read_log(path):
    """Every record in one file.

    A truncated last line -- the session was killed mid-write -- is
    skipped rather than raised: the rest of the file is perfectly good
    data, and refusing to read it would lose a day's work over one
    partial line.
    """
    out = []
    if not os.path.exists(path):
        return out
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def merge(bundle_dir):
    """Every reviewer's log, latest verdict per (reviewer, key, page).

    Returns (records, agreement) where agreement lists the (key, page)
    more than one reviewer judged, with what each said -- the raw
    material for an inter-rater number. Nothing is resolved here: which
    of two disagreeing reviewers is right is not a decision this file
    gets to make.
    """
    latest, by_page = {}, {}
    for name in sorted(os.listdir(str(bundle_dir))):
        if not (name.startswith('verdicts_') and name.endswith('.jsonl')):
            continue
        for rec in read_log(os.path.join(str(bundle_dir), name)):
            k = (rec.get('reviewer'), rec.get('key'), int(rec.get('page', 0)))
            latest[k] = rec            # a later line supersedes an earlier one
    for (reviewer, key, page), rec in latest.items():
        by_page.setdefault((key, page), []).append(rec)
    agreement = [(k, v) for k, v in sorted(by_page.items()) if len(v) > 1]
    return list(latest.values()), agreement


def _spot_key(y, x, z):
    q = SAME_SPOT_PX
    return (round(float(y) / q), round(float(x) / q), round(float(z) / q))


def labels(bundle_dir):
    """Verdicts turned into per-crop training labels, tallied by VOTE.

    Takes no bundle: a record already carries the coordinates and the
    crop geometry, so labels can be built after the bundle is gone. Use
    recut() to fetch the pixels for any of them.

    {key: {'crop': {...}, 'fov':, 'hybe':, 'channel':, 'cell':, 'store':,
           'positive': [(y,x,z), ...], 'negative': [...],
           'contested': [...], 'added': [(y,x), ...],
           'reviewers': n, 'votes': {(y,x,z): (kept, seen)}}}
    all crop-local; full_frame() converts.

    TALLIED PER SPOT, not per record. Two reviewers judging one page is
    the point of overlapping assignments -- it is how agreement gets
    measured -- but accumulating their records would put the same
    coordinate in the training set twice, silently weighting a spot by
    how many people happened to look at it.

    THREE buckets, not two. A spot every reviewer who saw it kept is a
    positive; one nobody kept is a negative; one kept by some and not
    others is CONTESTED and belongs to neither. Forcing those into a
    bucket by majority would train the model on exactly the examples
    humans could not agree about, which is the worst possible use of
    them: they are far more valuable as a list to look at again.

    Negatives are the shown-but-not-kept spots, and they are the useful
    half -- a detector trained only on positives learns nothing about
    what to reject.
    """
    recs, _agree = merge(bundle_dir)
    tally, coord, meta, added, who = {}, {}, {}, {}, {}
    for rec in recs:
        key = rec.get('key')
        if key is None:
            continue
        meta.setdefault(key, {k: rec.get(k) for k in
                              ('crop', 'fov', 'hybe', 'channel', 'cell',
                               'store')})
        t = tally.setdefault(key, {})
        c = coord.setdefault(key, {})
        for e in rec.get('shown') or []:
            sk = _spot_key(e['y'], e['x'], e['z'])
            kept, seen = t.get(sk, (0, 0))
            t[sk] = (kept + (1 if e.get('keep') else 0), seen + 1)
            c[sk] = (float(e['y']), float(e['x']), float(e['z']))
        added.setdefault(key, []).extend(
            (float(a['y']), float(a['x'])) for a in (rec.get('added') or []))
        who.setdefault(key, set()).add(rec.get('reviewer'))

    out = {}
    for key, t in tally.items():
        e = dict(meta.get(key) or {})
        e.update(positive=[], negative=[], contested=[],
                 added=added.get(key, []), reviewers=len(who.get(key, ())),
                 votes={})
        for sk, (kept, seen) in t.items():
            xyz = coord[key][sk]
            e['votes'][xyz] = (kept, seen)
            if kept == seen:
                e['positive'].append(xyz)
            elif kept == 0:
                e['negative'].append(xyz)
            else:
                e['contested'].append(xyz)
        for b in ('positive', 'negative', 'contested'):
            e[b].sort()
        out[key] = e
    return out
