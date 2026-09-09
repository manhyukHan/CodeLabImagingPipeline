"""
The two questions Spot Check asks, behind one seam.

PASS/FAIL asks "is there a spot here" -- four candidate cards from a
crop, judged against the cell they sit in. MULTISPOT asks "how many are
in this pillar, and where" -- one CONFIRMED spot's 15 x 15 x full-depth
column, with every match psf-match found in it.

WHY A SEAM RATHER THAN A FLAG. Everything that made the review app
survivable -- the per-reviewer shuffle with its shared slice, drafts that
outlive a Backspace, blitting a keep-toggle instead of redrawing nine
imshows, a write failure that does not take the window with it, a
completion screen that says which of three endings this is -- is the same
in both modes and none of it should be written twice or grown an `if`.
What actually differs is five things, and they are the five methods here:

    all_items    what a queue is made of
    item_key     what "already done" is keyed on
    load         crop -> the page's state
    draw         state -> figure
    record       state -> the line that gets appended

WHAT MULTISPOT REVIEWS IS A CONFIRMED SPOT, not a candidate. Its queue is
built from the pass/fail labels of the same bundle -- the positives every
reviewer who saw them kept -- so a pillar always contains at least one
emitter nobody disputes. That makes its negatives the useful class:
matches found BESIDE a real spot, which is exactly what a demultiplexer
must not be fooled by and exactly what pass/fail never shows.

DONE-TRACKING IS BY COORDINATE, NOT BY PAGE NUMBER. A crop's pillars are
ordinals over its confirmed spots, so one more pass/fail positive landing
in that crop renumbers every pillar after it and a reviewer who had
finished them would be handed them all again. See verdicts.multispot_key.
"""
import os
import time

import numpy as np

from codelab_pipeline.training import bundle as B
from codelab_pipeline.training import verdicts as V
from codelab_pipeline.training import view as VIEW
from codelab_pipeline.training import multispot_view as MVIEW

# THE PILLAR IS 15 x 15 x THE WHOLE STACK. Half-width 7 in y and x, and
# never a cut in z: the axial extent is the evidence a reviewer is being
# asked for, and a window in z would decide the question by cropping it.
PILLAR_HALF = 7

# HOW MANY MATCHES ONE PAGE CAN HOLD. The number keys are 1-9 and a
# tenth card could be drawn but never toggled -- a candidate on screen
# that no keystroke reaches is the defect the pass/fail overview already
# had once. MEASURED on 400 confirmed pillars of MP58/RNA at k=4.0, the
# most psf-match returned was six and the distribution was
#
#     0: 3.8%   1: 42.2%   2: 32.5%   3: 14.8%   4: 5.0%   5: 1.0%   6: 0.8%
#
# so this cuts nothing today. `n_found` records what was actually there,
# so a page that ever does hit the cap says so in its header rather than
# quietly showing nine of twelve.
MAX_HITS = 9

# The seed z for a spot a reviewer ADDED by hand. It has (y, x) and
# nothing else -- it was marked on a MIP -- and the pillar does not need
# a z, since it spans the whole stack. This value only has to make the
# identity key stable and never collide with a fitted z, which is >= 0.
ADDED_SEED_Z = -1.0

BANK_NAME = 'psf_bank.h5'

# HOW MANY SIGMA A MATCH MUST CLEAR TO BE PUT IN FRONT OF A PERSON.
#
# LOOSER THAN DETECTION, DELIBERATELY, because the costs are not the same
# ones. In the pipeline a false peak becomes a spot; here it becomes a
# card the reviewer does not press, and the default is drop -- so a
# proposal nobody keeps costs zero keystrokes. A match never proposed
# costs a label, and the reviewer can only get it back by noticing the
# omission and adding it by hand.
#
# MEASURED on 200 confirmed-spot pillars of MP58/RNA, 15 x 15 x 105, with
# the 7 x 7 x 11 template cut from D:/models/mp58_rna/psf_bank.h5. "seed
# found" is a match within 2 px of the spot pass/fail already confirmed
# -- a pillar showing NOTHING there is a page that cannot be judged:
#
#     k     seed found      0      1      2      3     4+   mean
#    3.0        98.0%     2.0%  10.5%  20.0%  24.5%  43.0%  2.96
#    3.5        96.5%     3.5%  22.5%  25.5%  25.0%  23.5%  2.42
#    4.0        95.5%     4.5%  32.0%  32.0%  18.0%  13.5%  2.04
#    4.5        93.5%     6.5%  44.0%  28.0%  13.0%   8.5%  1.73
#    5.0        92.5%     7.5%  52.5%  25.0%  10.5%   4.5%  1.52
#
# 4.0 is the trade taken: 95.5% of pillars propose the spot that is known
# to be there, two cards on an average page, and 4+ cards on one page in
# seven. At the detection default of 5.0 one pillar in thirteen comes up
# blank on a spot everybody already confirmed. Below 4.0 the page fills
# with cards faster than the recovery improves.
#
# THIS IS NOT psf_bank.K_SIGMA AND MUST NOT BECOME IT. That default
# governs what the pipeline calls a spot with nobody watching.
REVIEW_K_SIGMA = 4.0


class PassFail:
    """What this app has always done. Four cards from one crop."""

    name = 'passfail'
    label = 'Pass / fail — is each candidate a real spot?'
    log_kind = V.DEFAULT_KIND
    help_text = ('1-4 keep/drop   Space commit+next   Backspace back   '
                 'A add missed (then click the cell)   U undo add   '
                 'S skip unlabelled   Q quit          '
                 'DEFAULT IS DROP — press a number only for a real spot\n'
                 + VIEW.PANEL_LEGEND)

    def __init__(self, log=None):
        self.log = log

    def all_items(self, bundle_dir, per_page, max_per_crop):
        every = []
        for shard in B.shard_paths(str(bundle_dir)):
            for row in B.read_index(shard):
                n = int(row['n_candidates'])
                if n == 0:
                    continue      # nothing to judge, not a skipped page
                # THE REVIEW BUDGET, and it lives here rather than in the
                # bundle. The extractor keeps every candidate the data
                # produced; how many of them a person is asked to judge
                # is a per-session choice, and applying it here costs
                # nothing and destroys nothing. Candidates are stored
                # gate-pass first, so a cap always keeps the informative
                # ones.
                if max_per_crop:
                    n = min(n, int(max_per_crop))
                for pi, ix in enumerate(VIEW.pages_of(n, per_page)):
                    every.append((shard, row, pi, ix))
        return every

    @staticmethod
    def item_key(item):
        return (item[1]['key'], item[2])

    def done_keys(self, log):
        return log.done_pages()

    def load(self, item, log, per_page, max_per_crop):
        shard, row, page, ix = item
        stack, mask, cands, words = B.read_crop(shard, row['key'])
        rows = [(float(c['y']), float(c['x']), float(c['z']), float(c['p']),
                 int(c['fit_ok']), int(c['gate_pass']), w)
                for c, w in zip(cands, words)]
        shown_n = (min(len(rows), int(max_per_crop))
                   if max_per_crop else len(rows))
        n_total = len(rows)
        rows = rows[:shown_n]
        prior = log.page_verdict(row['key'], page)
        meta, _n, _v = B.read_meta(shard)
        return dict(shard=shard, row=row, page=page, ix=ix,
                    stack=stack, mask=mask, cands=rows,
                    npage=len(VIEW.pages_of(shown_n, per_page)),
                    store=meta.get('storage_path'), n_total=n_total,
                    prior_accepted=set(prior['accepted']) if prior else set(),
                    prior_added=list(prior['added']) if prior else [],
                    revisited=bool(prior))

    def draw(self, fig, s, per_page):
        row = s['row']
        header = (f"FOV{int(row['fov']):03d}   {row['hybe']}   "
                  f"ch{int(row['channel'])}   cell {int(row['cell'])}")
        return VIEW.draw_page(fig, s['stack'], s['mask'], s['cands'], s['ix'],
                              header=header, accepted=s['accepted'],
                              added=s['added'], page=s['page'],
                              npage=s['npage'], per_page=per_page,
                              n_total=s['n_total'])

    restyle = staticmethod(VIEW.restyle)
    mutable_artists = staticmethod(VIEW.mutable_artists)

    def page_of(self, s, i, per_page):
        for pi, ix in enumerate(VIEW.pages_of(len(s['cands']), per_page)):
            if i in ix:
                return pi + 1
        return None

    def commit(self, log, s, seconds):
        return log.commit(s['row'], s['page'], s['ix'], s['cands'],
                          s['accepted'], added=s['added'], seconds=seconds,
                          bundle=os.path.basename(s['shard']),
                          store=s['store'])


class Multispot:
    """One confirmed spot's pillar, and every match psf-match found in it.

    Needs a PSF bank -- an average of boxes people confirmed, written by
    tools/train_spotmodel.py. It is looked for beside the bundle so that
    copying a folder to a reviewer copies everything the mode needs; a
    path can be given explicitly instead.
    """

    name = 'multispot'
    label = 'Multispot — how many spots are in this pillar?'
    log_kind = V.MULTISPOT_KIND
    # EXPLICIT LINE BREAKS, not wrapping. A QLabel reports the height of
    # the text it was GIVEN, so a line the widget wraps for itself is a
    # line the layout never reserved room for -- MEASURED, the last line
    # of this was cut off mid-sentence at 1750 px, and it is the line that
    # says what the reviewer is looking at.
    help_text = (
        '1-9 keep/drop a match   Space commit+next   Backspace back   '
        'A add one psf-match missed (then click the MIP)   U undo add   '
        'S skip unlabelled   Q quit\n'
        'DEFAULT IS DROP — press a number only for a match that is a REAL, '
        'SEPARATE emitter. The pillar is 15 × 15 × the whole stack, centred '
        'on a spot everybody already confirmed,\n'
        'so at least one of these is real. Brightness is σ above the crop '
        'background and the bar reads the same on every card, so two cards '
        'that look alike can still be an order of magnitude apart.')

    def __init__(self, bank=None, bundle_dir=None, threshold=None,
                 k_sigma=REVIEW_K_SIGMA):
        self.bank_path = self._find_bank(bank, bundle_dir)
        self.threshold = threshold
        self.k_sigma = k_sigma
        self._engine = None
        self._prior = None

    @staticmethod
    def _find_bank(bank, bundle_dir):
        if bank:
            return str(bank)
        if bundle_dir:
            here = os.path.join(str(bundle_dir), BANK_NAME)
            if os.path.exists(here):
                return here
        return None

    @property
    def engine(self):
        """Built once, lazily -- so constructing the mode to read its
        label or its help text does not need a bank on disk."""
        if self._engine is None:
            if not self.bank_path or not os.path.exists(self.bank_path):
                raise FileNotFoundError(
                    'Multispot review needs a PSF bank and none was found.\n\n'
                    f'Put {BANK_NAME} in the bundle folder, or pass '
                    '--psf-bank PATH.\n\n'
                    'One is written by:\n'
                    '  python tools/train_spotmodel.py <bundle> --out <dir>')
            from codelab_pipeline.localization.engine import make_engine
            self._engine = make_engine('psf-match', bank=self.bank_path,
                                       threshold=self.threshold,
                                       k_sigma=self.k_sigma)
        return self._engine

    # -- the queue -------------------------------------------------------

    def all_items(self, bundle_dir, per_page, max_per_crop):
        """One item per CONFIRMED spot in this bundle's pass/fail labels.

        `max_per_crop` is not applied. It bounds how many candidates a
        person is ASKED ABOUT; here every item is a spot a person already
        said yes to, and dropping some of those would throw away
        finished work rather than bound unfinished work.
        """
        index = {}
        for shard in B.shard_paths(str(bundle_dir)):
            for row in B.read_index(shard):
                index[str(row['key'])] = (shard, row)
        out = []
        for key, e in sorted(V.labels(str(bundle_dir)).items()):
            if key not in index:
                continue              # labelled against a shard we lack
            shard, row = index[key]
            seeds = [(float(y), float(x), float(z))
                     for (y, x, z) in (e.get('positive') or [])]
            # A HAND-ADDED SPOT IS A CONFIRMED SPOT TOO. It carries no z
            # -- it was marked on a MIP -- and the pillar does not need
            # one, spanning the whole stack.
            seeds += [(float(y), float(x), ADDED_SEED_Z)
                      for (y, x) in (e.get('added') or [])]
            for pi, seed in enumerate(sorted(seeds)):
                out.append((shard, row, pi, seed))
        return out

    @staticmethod
    def item_key(item):
        y, x, z = item[3]
        return (item[1]['key'], V.multispot_key(y, x, z))

    def done_keys(self, log):
        return set(self._priors(log))

    def _priors(self, log):
        """{(key, seed_key): rec} for everything this reviewer has filed.

        Read once per session from every file this reviewer owns, the
        same way done_pages() is, and updated by commit(). It is what
        redisplays a pillar reached with Backspace -- without it, a page
        already judged comes back with every keep wiped from the screen
        while the file still holds them, and one more Space supersedes
        the real verdict with an empty one.
        """
        if self._prior is None:
            self._prior = {}
            for p in log.my_logs():
                for rec in V.read_log(p):
                    seed = rec.get('seed')
                    if not seed:
                        continue
                    self._prior[(rec.get('key'),
                                 V.multispot_key(seed['y'], seed['x'],
                                                 seed['z']))] = rec
        return self._prior

    # -- one page --------------------------------------------------------

    def load(self, item, log, per_page, max_per_crop):
        from codelab_pipeline.localization.engine import background_mode
        shard, row, page, seed = item
        sy, sx, sz = seed
        stack, mask, _cands, _words = B.read_crop(shard, row['key'])
        st = np.asarray(stack, float)
        bg, sigma = background_mode(st)
        pillar, origin, pad = MVIEW.pillar_at(st, sy, sx, half=PILLAR_HALF,
                                              background=bg, sigma=sigma)
        hits = self.engine.localize(pillar, n_max=None)
        hits.sort(key=lambda h: -float(h.p))
        n_found = len(hits)
        hits = hits[:MAX_HITS]
        # Candidate rows in the SAME shape pass/fail uses, so the click
        # snap, the keep toggle and the status line need no special case.
        # fit_ok is 0 and gate is 1: there is no Gaussian fit here, and
        # the pillar's seed already cleared whatever gate the bundle ran.
        rows = [(float(h.y), float(h.x), float(h.z), float(h.p), 0, 1, '')
                for h in hits]
        prior = self._priors(log).get(
            (str(row['key']), V.multispot_key(sy, sx, sz)))
        acc, add = set(), []
        if prior:
            # Priors are stored CROP-local; the page works in pillar
            # coordinates, so they come back through the same origin.
            y0, x0 = origin
            for e in prior.get('shown') or []:
                py, px = float(e['y']) - y0, float(e['x']) - x0
                near = _nearest(rows, py, px)
                if e.get('keep') and near is not None:
                    acc.add(near)
            add = [(float(a['y']) - y0, float(a['x']) - x0)
                   for a in prior.get('added') or []]
        meta, _n, _v = B.read_meta(shard)
        return dict(shard=shard, row=row, page=page, ix=list(range(len(rows))),
                    stack=pillar, mask=None, cands=rows, npage=1,
                    store=meta.get('storage_path'), n_total=n_found,
                    seed=seed, origin=origin, pad_frac=float(pad),
                    bg=float(bg), sigma=float(sigma),
                    depth=int(st.shape[2]),
                    prior_accepted=acc, prior_added=add,
                    revisited=bool(prior))

    def draw(self, fig, s, per_page):
        row = s['row']
        sy, sx, _sz = s['seed']
        y0, x0 = s['origin']
        header = (f"FOV{int(row['fov']):03d}   {row['hybe']}   "
                  f"ch{int(row['channel'])}   cell {int(row['cell'])}   "
                  f"pillar at y={sy:.1f} x={sx:.1f}   "
                  f"{s['n_total']} match{'' if s['n_total'] == 1 else 'es'}"
                  + (f' (showing {MAX_HITS})' if s['n_total'] > MAX_HITS
                     else ''))
        hits = [(c[0], c[1], c[2], c[3]) for c in s['cands']]
        return MVIEW.draw_pillar(fig, s['stack'], hits, header=header,
                                 accepted=s['accepted'],
                                 seed_yx=(sy - y0, sx - x0),
                                 pad_frac=s['pad_frac'], added=s['added'])

    restyle = staticmethod(MVIEW.restyle)
    mutable_artists = staticmethod(MVIEW.mutable_artists)

    def page_of(self, s, i, per_page):
        return 1              # one pillar is one page; nothing is off it

    def commit(self, log, s, seconds):
        row, y0, x0 = s['row'], s['origin'][0], s['origin'][1]
        sy, sx, sz = s['seed']
        keep = set(int(i) for i in s['accepted'])
        # STORED CROP-LOCAL, like every other verdict in this repo, so
        # full_frame() and recut() work on these records unchanged and a
        # coordinate means the same thing in both files.
        shown = [{'i': i, 'y': round(c[0] + y0, 3), 'x': round(c[1] + x0, 3),
                  'z': round(c[2], 3), 'p': round(c[3], 4),
                  'keep': 1 if i in keep else 0}
                 for i, c in enumerate(s['cands'])]
        rec = {'key': str(row['key']),
               'fov': int(row['fov']), 'hybe': str(row['hybe']),
               'channel': int(row['channel']), 'cell': int(row['cell']),
               'crop': {'y0': int(row['y0']), 'x0': int(row['x0']),
                        'h': int(row['h']), 'w': int(row['w']),
                        'depth': int(row['depth'])},
               'page': int(s['page']),
               'seed': {'y': round(float(sy), 3), 'x': round(float(sx), 3),
                        'z': round(float(sz), 3)},
               'pillar': {'y0': int(y0), 'x0': int(x0),
                          'half': int(PILLAR_HALF), 'depth': int(s['depth']),
                          'pad_frac': round(float(s['pad_frac']), 4),
                          'bg': round(float(s['bg']), 3),
                          'sigma': round(float(s['sigma']), 4)},
               'n_found': int(s['n_total']),
               'bank': os.path.basename(self.bank_path or ''),
               'shown': shown,
               'added': [{'y': round(float(a) + y0, 3),
                          'x': round(float(b) + x0, 3)}
                         for (a, b) in s['added']],
               'reviewer': log.reviewer,
               'at': time.strftime('%Y-%m-%dT%H:%M:%S'),
               'bundle': os.path.basename(s['shard'])}
        if s.get('store'):
            rec['store'] = str(s['store'])
        if seconds is not None:
            rec['seconds'] = round(float(seconds), 2)
        out = log._append(rec)
        self._priors(log)[(rec['key'],
                           V.multispot_key(sy, sx, sz))] = rec
        return out


def _nearest(rows, y, x, within=1.0):
    """Which of `rows` is at (y, x), or None.

    A prior verdict names a match by coordinate, not by position -- the
    same reason the pass/fail record does. psf-match re-run on the same
    pillar with the same bank returns the same sub-voxel answer, so this
    is effectively an identity lookup with slack for a re-trained bank.
    """
    best, bd = None, float(within) ** 2
    for i, c in enumerate(rows):
        d = (float(c[0]) - y) ** 2 + (float(c[1]) - x) ** 2
        if d < bd:
            best, bd = i, d
    return best


MODES = {PassFail.name: PassFail, Multispot.name: Multispot}


def make_mode(name, bundle_dir=None, bank=None, k_sigma=REVIEW_K_SIGMA):
    try:
        cls = MODES[str(name)]
    except KeyError:
        raise ValueError(f'unknown review mode {name!r} -- '
                         f'known: {sorted(MODES)}')
    if cls is Multispot:
        return cls(bank=bank, bundle_dir=bundle_dir, k_sigma=k_sigma)
    return cls()
