"""
Human verdicts: append-only, one file per reviewer, never inside the bundle.

WHY NOT IN THE BUNDLE. Ten people review the same shards. A bundle that
were also the answer sheet would have to be locked, copied per person, or
merged under contention -- so it is READ-ONLY, and every reviewer writes
their own JSONL beside it. Merging is then a concatenation, and two
people reviewing the same page is a feature (it measures agreement)
rather than a write conflict.

WHY EVERY REVIEWED PAGE IS RECORDED, EVEN AN EMPTY ONE. The default is
REJECT: a reviewer clicks only the real spots, so most pages end with
nothing accepted. If silence meant rejection, "reviewed and found
nothing" and "never opened" would be the same record, and the difference
is the whole value of a negative -- an unreviewed page is unlabelled
data, a reviewed-and-empty page is 4 confirmed hard negatives. So a page
commit ALWAYS appends a line, and `accepted` is simply empty.

APPEND-ONLY, one JSON object per line, flushed per commit. A reviewer's
machine dying mid-session costs the page in progress and nothing else.
Re-reviewing a page appends a second line; the later one wins, and both
survive so a disagreement with an earlier pass stays visible.

A line:
  {"key": "fov003|Hyb_107|ch555|cell57",   the crop, as bundle.crop_key
   "page": 0, "page_ix": [0,1,2,3],        which candidates were shown
   "accepted": [0, 2],                     indices the reviewer kept
   "added": [[41.2, 17.8], ...],           spots NO candidate covered
   "reviewer": "shj", "at": "2026-09-07T14:03:11",
   "seconds": 6.2,                         how long the page took
   "bundle": "fov003.h5"}
"""
import json
import os
import time

FILENAME_FMT = 'verdicts_{reviewer}.jsonl'


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
        stopped, and re-reading it per page would make the queue O(n^2)
        in a file that grows all session.
        """
        if self._done is None:
            self._done = set()
            for rec in read_log(self.path):
                self._done.add((rec.get('key'), int(rec.get('page', 0))))
        return self._done

    def commit(self, key, page, page_ix, accepted, added=(), seconds=None,
               bundle=None):
        rec = {'key': str(key), 'page': int(page),
               'page_ix': [int(i) for i in page_ix],
               'accepted': sorted(int(i) for i in accepted),
               'added': [[float(a), float(b)] for (a, b) in added],
               'reviewer': self.reviewer,
               'at': time.strftime('%Y-%m-%dT%H:%M:%S')}
        if seconds is not None:
            rec['seconds'] = round(float(seconds), 2)
        if bundle:
            rec['bundle'] = str(bundle)
        d = os.path.dirname(os.path.abspath(self.path))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec) + '\n')
            f.flush()
            os.fsync(f.fileno())      # a page committed is a page kept
        self.done_pages().add((rec['key'], rec['page']))
        return rec


def read_log(path):
    """Every record in one file. A truncated last line -- the session was
    killed mid-write -- is skipped, not raised: the rest of the file is
    perfectly good data and refusing to read it would lose a day's work
    over one partial line."""
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
    that more than one reviewer judged, with what each of them said --
    the raw material for an inter-rater number. Nothing is resolved here:
    which of two disagreeing reviewers is right is not a decision this
    file gets to make.
    """
    latest, by_page = {}, {}
    for name in sorted(os.listdir(str(bundle_dir))):
        if not (name.startswith('verdicts_') and name.endswith('.jsonl')):
            continue
        for rec in read_log(os.path.join(str(bundle_dir), name)):
            k = (rec.get('reviewer'), rec.get('key'), int(rec.get('page', 0)))
            latest[k] = rec           # a later line supersedes an earlier one
    for (reviewer, key, page), rec in latest.items():
        by_page.setdefault((key, page), []).append(rec)
    agreement = [(k, v) for k, v in sorted(by_page.items()) if len(v) > 1]
    return list(latest.values()), agreement


def labels(bundle_dir, bundle_index_by_key):
    """Verdicts turned into per-crop training labels, by VOTE.

    bundle_index_by_key: {key: candidate list} from the bundle, so an
    accepted INDEX becomes a coordinate. Returns
    {key: {'positive': [...], 'negative': [...], 'contested': [...],
           'added': [(y,x), ...], 'reviewers': n, 'votes': {i: (kept, seen)}}}
    with each coordinate as (y, x, z).

    TALLIED PER CANDIDATE, not per record. Two reviewers judging the same
    page is the point of overlapping assignments -- it is how agreement
    gets measured -- but accumulating their records would put the same
    coordinate in the training set twice, silently weighting a spot by
    how many people happened to look at it.

    THREE buckets, not two. A candidate every reviewer who saw it kept is
    a positive; one nobody kept is a negative; one kept by some and not
    others is CONTESTED and belongs to neither. Forcing a contested spot
    into a bucket by majority would train the model on the examples
    humans could not agree about, which is the worst possible use of
    them -- they are far more valuable as a list of what to look at
    again, or to send to a third reviewer.

    Negatives are the shown-but-not-kept candidates, and they are the
    useful half: a detector trained only on positives learns nothing
    about what to reject.
    """
    recs, _agree = merge(bundle_dir)
    tally, added, who = {}, {}, {}
    for rec in recs:
        key = rec.get('key')
        if key not in bundle_index_by_key:
            continue
        acc = set(rec.get('accepted') or [])
        t = tally.setdefault(key, {})
        for i in rec.get('page_ix') or []:
            kept, seen = t.get(i, (0, 0))
            t[i] = (kept + (1 if i in acc else 0), seen + 1)
        added.setdefault(key, []).extend(
            tuple(a) for a in (rec.get('added') or []))
        who.setdefault(key, set()).add(rec.get('reviewer'))

    out = {}
    for key, t in tally.items():
        cands = bundle_index_by_key[key]
        e = {'positive': [], 'negative': [], 'contested': [],
             'added': added.get(key, []), 'reviewers': len(who.get(key, ())),
             'votes': dict(t)}
        for i, (kept, seen) in sorted(t.items()):
            if i >= len(cands):
                continue
            c = cands[i]
            xyz = (float(c[0]), float(c[1]), float(c[2]))
            if kept == seen:
                e['positive'].append(xyz)
            elif kept == 0:
                e['negative'].append(xyz)
            else:
                e['contested'].append(xyz)
        out[key] = e
    return out
