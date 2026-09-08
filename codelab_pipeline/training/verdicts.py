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
SESSION_FMT = 'verdicts_{reviewer}__{session}.jsonl'


def _safe(name):
    """A reviewer name that is also a filename. Anything outside
    [A-Za-z0-9._-] becomes '-', so a name with a space or a slash cannot
    put the log somewhere unexpected -- or nowhere."""
    s = ''.join(c if (c.isalnum() or c in '._-') else '-' for c in str(name))
    return s.strip('-') or 'anon'


def _session_tag():
    """Unique per RUNNING PROCESS: timestamp plus pid.

    The pid matters. Two windows opened in the same second would
    otherwise share a file and hit the very append race this is here to
    avoid.
    """
    return f'{time.strftime("%Y%m%d-%H%M%S")}-{os.getpid()}'

# Two coordinates within this many pixels are the same spot when tallying
# votes. Well under a real emitter's ~1.3 px sigma, and loose enough that
# a bundle regenerated after a library upgrade still aggregates with the
# labels made against the old one.
SAME_SPOT_PX = 0.5


def path_for(bundle_dir, reviewer):
    return os.path.join(str(bundle_dir), FILENAME_FMT.format(reviewer=reviewer))


class VerdictLog:
    """Append-only writer + the resume index for one reviewer.

    ONE FILE PER SESSION, not per reviewer, and that is a correctness
    decision rather than tidiness. `open(path, 'a')` is NOT an atomic
    append on Windows -- the CRT seeks to end and then writes, so two
    processes appending to one file can compute the same offset and
    overwrite each other. Measured: two instances open under the same
    reviewer name lose verdicts at the seam.

    Locking would fix it and bring its own failure -- a crashed process
    leaving a lock behind, on a program whose whole point is surviving a
    crash. Giving each session its own file removes the shared resource
    instead: two windows, two files, no seam. merge() already globs, and
    done_pages() reads every file this reviewer owns, so resume is
    unaffected.
    """

    def __init__(self, bundle_dir, reviewer, session=None):
        self.dir = str(bundle_dir)
        self.reviewer = str(reviewer)
        self.session = str(session) if session else _session_tag()
        self.path = os.path.join(self.dir, SESSION_FMT.format(
            reviewer=_safe(self.reviewer), session=self.session))
        self._done = None
        self._latest = None

    def my_logs(self):
        """Every file this REVIEWER has written to this bundle, any
        session. Resume has to see all of them or a second window would
        re-serve pages the first one already judged."""
        import glob
        pat = os.path.join(self.dir, SESSION_FMT.format(
            reviewer=_safe(self.reviewer), session='*'))
        found = sorted(glob.glob(pat))
        legacy = path_for(self.dir, self.reviewer)   # pre-session layout
        if os.path.exists(legacy) and legacy not in found:
            found.append(legacy)
        return found

    def done_pages(self):
        """{(key, page)} already committed BY THIS REVIEWER, any session.

        Read once and cached: it is what lets a session resume where it
        stopped, and re-reading per page would make the queue O(n^2) in a
        file that grows all session.
        """
        if self._done is None:
            self._done = set()
            for p in self.my_logs():
                for rec in read_log(p):
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

        Served from an in-memory index, not by re-reading the file. The
        app calls this on EVERY page load and the file grows all session,
        so re-reading made a session quadratic in its own length -- and it
        is called on the page-turn path, where the reviewer is waiting.
        The index is seeded once from whatever this session's file already
        holds (normally nothing) and updated by commit().
        """
        if self._latest is None:
            self._latest = {}
            for rec in read_log(self.path):
                self._latest[(rec.get('key'), int(rec.get('page', 0)))] = rec
        found = self._latest.get((key, int(page)))
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
            # HEAL A TORN TAIL BEFORE WRITING. A process killed mid-write
            # leaves a fragment with no newline; appending straight onto
            # it fuses the fragment and the next record into one
            # unparseable line, and read_log then discards BOTH -- so a
            # perfectly good verdict disappears, the page is missing from
            # done_pages, and it is silently re-served. MEASURED: two
            # commits either side of a torn line, one readable.
            #
            # One newline separates them, and the fragment stays its own
            # (discarded) line, which is the only part that was ever lost.
            if f.tell() > 0:
                f.flush()
                with open(self.path, 'rb') as probe:
                    probe.seek(-1, os.SEEK_END)
                    if probe.read(1) != b'\n':
                        f.write('\n')
            f.write(json.dumps(rec) + '\n')
            f.flush()
            os.fsync(f.fileno())        # a page committed is a page kept
        self.done_pages().add((rec['key'], rec['page']))
        if self._latest is None:
            self.page_verdict(rec['key'], rec['page'])   # seed from the file
        self._latest[(rec['key'], rec['page'])] = rec
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
    """Quantise a FITTED coordinate. Not for anything a hand produced.

    Two records name the same shown candidate with the same float -- both
    read it out of the same read-only bundle -- so this is effectively an
    identity function with a little slack for a re-extraction. It is a
    GRID, not a radius: two values 0.02 px apart either side of a bin edge
    get different keys, which is harmless for coordinates that agree
    exactly and useless for coordinates that only agree closely.
    """
    q = SAME_SPOT_PX
    return (round(float(y) / q), round(float(x) / q), round(float(z) / q))


# How far apart two people's clicks can be and still be one spot.
#
# NOT SAME_SPOT_PX. That is 0.5 px, chosen for fitted coordinates that
# arrive bit-identical, and it was briefly used for added spots too --
# which failed in both possible ways. Measured on this figure geometry
# (Figure(15.0, 5.6) at dpi 110), the overview renders at 4-8 device px
# per image px, so 0.5 px is about three screen pixels: nobody aims that
# well. Three reviewers marking one emitter within 0.4 px of each other
# still produced three separate labels, each reported as a lone 1-of-3
# minority -- agreement recorded as disagreement, which is worse than the
# duplication it replaced.
#
# 4.0 px is what the app already uses to decide "this click IS that
# candidate" (spotcheck.app.ADD_SNAP_PX), for the same reason and against
# the same hand.
SAME_ADDED_PX = 4.0


def _cluster_added(points, radius=SAME_ADDED_PX):
    """[(y, x, reviewer), ...] -> [((y, x), {reviewers}), ...].

    Proximity, not quantisation: a bin edge does not decide whether two
    clicks at the same emitter are the same spot. Each point joins the
    first cluster whose CENTRE it falls within, which bounds the chaining
    that single-link would allow -- a line of clicks 3 px apart stays
    several spots rather than becoming one long smear.

    Deterministic: points are sorted first, so the same logs give the same
    clusters whatever order the files were read in.
    """
    r2 = float(radius) ** 2
    out = []                       # [[sum_y, sum_x, n, {reviewers}], ...]
    for y, x, who in sorted(points, key=lambda p: (p[0], p[1], str(p[2]))):
        for c in out:
            cy, cx = c[0] / c[2], c[1] / c[2]
            if (cy - y) ** 2 + (cx - x) ** 2 <= r2:
                c[0] += y
                c[1] += x
                c[2] += 1
                c[3].add(who)
                break
        else:
            out.append([y, x, 1, {who}])
    return [((round(c[0] / c[2], 3), round(c[1] / c[2], 3)), c[3])
            for c in out]


def labels(bundle_dir):
    """Verdicts turned into per-crop training labels, tallied by VOTE.

    Takes no bundle: a record already carries the coordinates and the
    crop geometry, so labels can be built after the bundle is gone. Use
    recut() to fetch the pixels for any of them.

    {key: {'crop': {...}, 'fov':, 'hybe':, 'channel':, 'cell':, 'store':,
           'positive': [(y,x,z), ...], 'negative': [...],
           'contested': [...], 'added': [(y,x), ...],
           'added_votes': {(y,x): (added_by, reviewers)},
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
    tally, coord, meta, who = {}, {}, {}, {}
    add_who = {}                 # key -> [(y, x, reviewer), ...]
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
        # ADDED SPOTS ARE VOTED ON TOO. They used to be concatenated
        # straight from every record, so one missed emitter that three
        # reviewers all marked entered the training set three times --
        # exactly the silent per-spot weighting the shown candidates are
        # tallied to avoid -- and there was no way to tell a spot everyone
        # who looked at the cell saw from one that a single person marked
        # and nobody else did.
        #
        # THEY DO NOT MERGE WITH `shown`, and an earlier version of this
        # comment claimed they did. They cannot: an added spot has no z --
        # the reviewer clicked a MIP -- so there is no common key, and
        # the two are tallied separately by design. A reviewer who adds a
        # spot that another reviewer's page already showed and rejected
        # therefore produces both an `added` entry and a `negative` one
        # for the same place. That is the honest record of what happened;
        # resolving it is a question about those two people, not about
        # this function.
        for a in rec.get('added') or []:
            add_who.setdefault(key, []).append(
                (float(a['y']), float(a['x']), rec.get('reviewer')))
        who.setdefault(key, set()).add(rec.get('reviewer'))

    out = {}
    for key, t in tally.items():
        e = dict(meta.get(key) or {})
        clusters = _cluster_added(add_who.get(key, []))
        nrev = len(who.get(key, ()))
        e.update(positive=[], negative=[], contested=[],
                 added=sorted(c for c, _w in clusters),
                 added_votes={c: (len(w), nrev) for c, w in clusters},
                 reviewers=nrev, votes={})
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
