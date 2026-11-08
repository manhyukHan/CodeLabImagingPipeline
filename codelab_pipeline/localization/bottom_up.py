"""Building traces from the spots up, instead of from an anchor down.

THE CURRENT ROUTE IS TOP-DOWN. An operator picks an anchor spot, that
anchor becomes an allele, and every round is then searched in a box
around it. It works, and it costs: the anchor is chosen by hand, a cell
with two alleles needs two choices, and a round whose readout sits just
outside the box is lost without anything noticing.

THIS ROUTE STARTS FROM WHAT WAS DETECTED. Every readout round of a cell
is localized first -- which the pipeline already does, through the
learned engine -- and the traces are then the structure IN those spots.
Three questions, three functions, each pure and each with an `engine`
seam so a learned model can take it over without the others moving:

  multiplicity_gate  how many alleles does this cell have?
  cluster_cell       which spots belong to which allele?
  resolve_paths      which ONE spot is each allele's bin?

THE CONSTRAINT THAT MAKES IT WORK. A readout round images one locus,
so it gives ONE signal per allele. A cell with two alleles should show
two spots in most rounds; a round showing five is telling us something
is wrong with that round, not that the cell has five alleles. That
turns the per-bin spot counts into a measurement of the allele count --
and, in the same breath, of the detection efficiency, because the
counts below the allele count are the misses.

UNITS. Positions are MICROMETRES in the shared frame, as
polymer.fov_polymer_table already produces. Working in pixels here
would make the lateral and axial scales incomparable, and every
distance in this module is a real one.

WHAT THIS MODULE DOES NOT DO. It does not read a store, open an image,
or re-fit anything: it takes detections and returns assignments. The
re-fitting that an allele's own shape makes possible -- the correction
step -- needs crops, so it belongs above this, with these answers in
hand.
"""
import collections
import itertools

import numpy as np

from codelab_pipeline.localization import pathfind

#: a spot that belongs to no allele
UNASSIGNED = -1

#: a bin with no spot for this allele
NO_SPOT = -1


# ---------------------------------------------------------------- input

CellSpots = collections.namedtuple('CellSpots', 'bin pos amp quality row')
CellSpots.__doc__ = """One cell's detections, as parallel arrays.

bin      (n,) int    polymer slot of each spot; UNASSIGNED where the
                     readout map does not place it (a QC round, or a
                     round nobody named)
pos      (n, 3)      y, x, z in MICROMETRES, shared frame
amp      (n,)        brightness
quality  (n,)        p_exist, or NaN where the engine reported none
row      (n,) int    the caller's own row ids, so an answer points back
                     at the spot it came from rather than at an index
                     into a filtered copy
"""


def cell_spots(bins, pos, amp=None, quality=None, row=None):
    """A CellSpots from loose arrays, with the shapes checked once.

    Checked here and not at every use: an (n, 3) that is really (3, n)
    reads as three spots and answers confidently, which is the failure
    this costs one function call to make impossible.
    """
    b = np.asarray(bins, dtype=np.int64).ravel()
    p = np.asarray(pos, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError('pos must be (n, 3) y/x/z in um, got %r'
                         % (p.shape,))
    if len(b) != len(p):
        raise ValueError('%d bins but %d positions' % (len(b), len(p)))
    n = len(b)
    a = (np.full(n, np.nan) if amp is None
         else np.asarray(amp, dtype=np.float64).ravel())
    q = (np.full(n, np.nan) if quality is None
         else np.asarray(quality, dtype=np.float64).ravel())
    r = (np.arange(n, dtype=np.int64) if row is None
         else np.asarray(row, dtype=np.int64).ravel())
    for name, arr in (('amp', a), ('quality', q), ('row', r)):
        if len(arr) != n:
            raise ValueError('%s has %d entries for %d spots'
                             % (name, len(arr), n))
    return CellSpots(b, p, a, q, r)


def bin_counts(spots, n_slots=None):
    """(counts (n_slots,) int, slots (n_slots,) int) -- spots per bin.

    Bins with no spot are counted as 0 and KEPT: a round that detected
    nothing is evidence about the detection efficiency, and dropping it
    is how an efficiency gets overestimated.
    """
    b = spots.bin[spots.bin >= 0]
    n = int(n_slots if n_slots is not None
            else (b.max() + 1 if len(b) else 0))
    counts = np.zeros(n, dtype=np.int64)
    for v in b:
        if 0 <= v < n:
            counts[v] += 1
    return counts, np.arange(n, dtype=np.int64)


# ------------------------------------------------- 1. how many alleles

Multiplicity = collections.namedtuple(
    'Multiplicity', 'k efficiency loglik support counts reason')
Multiplicity.__doc__ = """How many alleles the per-bin counts imply.

k           the allele count
efficiency  the detection efficiency that goes with it: the fraction of
            (allele, bin) pairs that produced a spot. It is not a
            by-product -- k and efficiency trade off against each other
            exactly, and a k reported without one is half an answer
loglik      the fitted log-likelihood, for comparing k against k
support     how many bins the fit saw
counts      the per-bin counts it was fitted to
reason      why this k, in words, or why no k could be chosen
"""


#: THE HARD CAP ON HOW MANY ALLELES ANY ROUTE MAY REPORT.
#:
#: A cap is not a cosmetic bound here, it is the only thing standing
#: between the cost sweep and an answer the cell cannot hold. Measured
#: on JP chr19 FOVs 7-10, choose_paths returned a median k of 8 against
#: a store count of 2, and it was not a bug in the search: the objective
#: charges a CONSTANT per path while a path's saving grows with the
#: number of spots it collects, so the rule it actually implements is
#: "believe a path once it collects more than path_odds x min_bins = 9
#: spots" -- nine, whatever the panel's size, which on a 90-bin panel is
#: a 10% trace. Nothing in the objective knows how many alleles a
#: nucleus can have, so that knowledge goes here instead, as a number
#: and not as a tuned threshold.
#:
#: 3, not 2: two homologues is the diploid expectation, but these lines
#: are not reliably diploid, and the store's own merged counts run to 4
#: in a minority of cells. 3 admits the common aneuploid case while
#: still refusing the sweep's 8. Raise it per call, deliberately, for an
#: experiment that needs more -- every entry point takes k_max.
DEFAULT_K_MAX = 3


#: how often a bin is expected to hold a count no allele count can
#: explain -- a spurious detection, two spots the localizer split, a
#: neighbouring cell's spot inside this mask. Small and FIXED rather
#: than fitted: fitting it would let it absorb a real allele.
DEFAULT_CONTAMINATION = 0.05


def binomial_multiplicity(counts, k_max=None, min_support=8,
                          contamination=DEFAULT_CONTAMINATION):
    """THE DEFAULT ENGINE: fit Binomial(k, e) to the per-bin counts.

    NO SINGLE BIN DECIDES THIS. The count is fitted over every bin at
    once, and a bin holding more spots than k alleles could give is
    possible-but-unlikely rather than impossible: without that, ONE
    barcode picking up two spurious detections made k at least 3 in an
    otherwise single-allele cell, which is the failure this mixture
    exists to stop. The efficiency is estimated from the counts clipped
    at k for the same reason -- the raw mean is inflated by exactly the
    contamination being modelled.

    A round images one locus, so each of the cell's k alleles either
    gives it a spot or does not. The number of spots in a bin is then
    Binomial(k, e) with e the detection efficiency, and k and e are
    fitted together over every bin -- including the empty ones, which
    are the misses and are exactly what pins e down.

    WHY NOT THE OBVIOUS ANSWERS. The maximum over bins is an upper
    bound, not an estimate: one bin with a spurious pair sets it. The
    mode is the count that happens most, which at e = 0.5 is k - 1 and
    at lower e is lower still -- it reads a dim experiment as a cell
    with fewer alleles, which is the error that would quietly halve a
    dataset. The mean divided by e would do, if e were known; it is not,
    and that is the point of fitting them together.

    Parsimony on a tie: the smallest k within 1 nat of the best, because
    a larger k can always absorb a smaller e and the likelihood barely
    moves.
    """
    k_max = DEFAULT_K_MAX if k_max is None else k_max
    c = np.asarray(counts, dtype=np.int64).ravel()
    if len(c) < int(min_support):
        return Multiplicity(None, float('nan'), float('nan'), len(c), c,
                            'only %d bin(s); %d needed to fit a count'
                            % (len(c), int(min_support)))
    if not c.any():
        return Multiplicity(0, 0.0, float('nan'), len(c), c,
                            'no spot in any bin')
    contam = min(max(float(contamination), 1e-9), 0.5)
    span = float(np.log(max(int(c.max()), 1) + 1))
    best = []
    for k in range(1, int(k_max) + 1):
        # clipped at k: a bin above k is contamination, and letting it
        # raise the mean would raise the efficiency it is evidence
        # against
        e = float(np.minimum(c, k).mean()) / k
        e = min(max(e, 1e-6), 1 - 1e-6)
        ll = float(np.sum(np.logaddexp(
            np.log1p(-contam) + _log_binom_pmf(c, k, e),
            np.log(contam) - span)))
        best.append((k, e, ll))
    if not best:
        return Multiplicity(None, float('nan'), float('nan'), len(c), c,
                            'k_max=%d is below 1' % int(k_max))
    top = max(ll for _k, _e, ll in best)
    k, e, ll = min(((k, e, ll) for k, e, ll in best if ll >= top - 1.0),
                   key=lambda t: t[0])
    return Multiplicity(k, e, ll, len(c), c,
                        'binomial fit over %d bin(s): k=%d at efficiency '
                        '%.2f, %d bin(s) above k' % (len(c), k, e,
                                                     int((c > k).sum())))


def _log_binom_pmf(c, k, e):
    """log P(count | k, e), elementwise. Small k, so exact and cheap."""
    from math import lgamma
    c = np.asarray(c, dtype=np.int64)
    out = np.empty(len(c), dtype=np.float64)
    lk = lgamma(k + 1)
    for i, v in enumerate(c):
        if v < 0 or v > k:
            out[i] = -np.inf
            continue
        out[i] = (lk - lgamma(v + 1) - lgamma(k - v + 1)
                  + v * np.log(e) + (k - v) * np.log1p(-e))
    return out


#: how many of the axis's bins must be near a site before it counts as
#: an allele, as a FRACTION of the axis. A fraction and not a number so
#: it travels between experiments: 62 bins at 0.24 is 15, which is what
#: the MP58 sweep below was run at, and an 80-bin design asks for 19.
DEFAULT_SITE_FRACTION = 0.24


def spatial_multiplicity(spots, n_slots, radius_um=1.5,
                         site_fraction=DEFAULT_SITE_FRACTION,
                         min_bins=None, k_max=None):
    """THE DEFAULT ENGINE: count the places that look like an allele.

    A SITE is a position with at least `min_bins` DISTINCT bins within
    `radius_um`, and no two sites may sit inside one radius of each
    other. It is the allele count per CELL, and nothing about it depends
    on how many spots any one barcode produced.

    WHY NOT THE COUNTS. binomial_multiplicity, below, reads the allele
    count off the per-bin spot counts, and measured on MP58 FOV 1 it is
    wrong: median k=4 against a stored count of 2, agreeing on 12% of
    125 cells, biased +1.64. The model is the problem, not its
    constants -- the pooled per-bin counts have var/mean = 1.25, and a
    Binomial(k, e) must have var/mean < 1, so no (k, e) can fit them.
    The overdispersion is per-round: bin 2 averages 5.20 spots per cell
    and bin 37 averages 0.49, a tenfold spread that a constant
    efficiency cannot carry, and 49 of the 62 bins have at least ten
    cells holding more than two spots. It is not one bad barcode.

    SISTER CHROMATIDS ARE NOT WHAT THIS COUNTS, and on this data they
    are not visible at all: of 9291 within-bin spot pairs, 0% are closer
    than 1 um and 82% are farther than 2 um (median 3.75). Two spots in
    one bin here are two ALLELES, or a spurious detection -- not two
    sisters. Separating sisters is a different question at a different
    scale, and it is not this function's.

    THE CHECK THAT THIS IS RIGHT: once the cell is split spatially, each
    cluster holds a median of 1.05 spots per bin (p95 1.27) over 239
    clusters. One round gives one signal per allele, and that is what a
    correctly separated allele looks like.

    MEASURED, NOT TUNED. The sweep on MP58 FOV 1, against the stored
    top-down count (which is itself an operator's hand-picked anchors,
    not ground truth):

        radius  min_bins  median  agree  bias
         1.0      15        2      44%   +0.27
         1.5      15        2      50%   +0.45
         1.5      20        2      42%   -0.12
         2.0      20        2      42%   +0.20
         3.0      10        4      14%   +1.57

    The defaults are the best row of one FOV of one experiment, which is
    not enough to call them settled -- hence the fraction, and hence
    this table, so the next person sees what they are inheriting.
    """
    n = int(n_slots or 0)
    kcap = int(DEFAULT_K_MAX if k_max is None else k_max)
    mb = int(min_bins if min_bins is not None
             else max(2, round(float(site_fraction) * n)))
    p = spots.pos
    if not len(p):
        return Multiplicity(0, 0.0, float('nan'), n, np.zeros(0, dtype=int),
                            'no spots')
    sup = bin_support(p, spots.bin, radius_um)
    order = sorted(range(len(p)), key=lambda i: (-int(sup[i]), int(i)))
    taken = []
    for i in order:
        if sup[i] < mb:
            break
        if any(np.linalg.norm(p[i] - p[j]) <= float(radius_um)
               for j in taken):
            continue
        taken.append(i)
        # the sites are visited in order of decreasing support, so the
        # cap keeps the best-supported ones rather than the first found
        if len(taken) >= kcap:
            break
    counts, _ = bin_counts(spots, n)
    k = len(taken)
    e = (float(np.minimum(counts, k).mean()) / k) if k else 0.0
    return Multiplicity(k, e, float('nan'), n, counts,
                        '%d site(s) with >=%d distinct bin(s) within '
                        '%.2f um, capped at %d'
                        % (k, mb, radius_um, kcap))


def multiplicity_gate(spots, n_slots=None, engine=None, **kw):
    """How many alleles this cell has.

    engine: a callable (spots, n_slots, **kw) -> Multiplicity, defaulting
    to spatial_multiplicity. THE SEAM: a learned model goes here, and
    nothing else in this module has to know. binomial_multiplicity is
    the other engine in this module and takes counts rather than spots,
    so it is wrapped rather than passed directly -- see
    counts_engine().
    """
    return (engine or spatial_multiplicity)(spots, n_slots, **kw)


def counts_engine(fn):
    """Adapt a counts-based estimator to the multiplicity_gate seam."""
    def _engine(spots, n_slots, **kw):
        counts, _slots = bin_counts(spots, n_slots)
        return fn(counts, **kw)
    _engine.__name__ = 'counts_engine(%s)' % getattr(fn, '__name__', 'fn')
    return _engine


# ------------------------------------------------ 2. which spots group

Clustering = collections.namedtuple(
    'Clustering', 'labels centres k inertia reason')
Clustering.__doc__ = """Which allele each spot was assigned to.

labels    (n,) int   0..k-1, or UNASSIGNED
centres   (k, 3)     each allele's centre in um
k         how many
inertia   summed squared distance to the assigned centre, um^2
reason    in words
"""


#: how far an allele reaches, for deciding what is near it. Roughly a
#: chromatin domain's own size.
#:
#: IT IS A SEEDING RADIUS AND NOTHING ELSE NOW. It used to be four
#: things at once -- support radius, seed spacing, a hard cutoff on what
#: a path could take, and the price of leaving a spot unassigned -- and
#: the last two are gone: the loss is a likelihood over chain steps
#: (localization/pathfind.py), whose scale comes from each step's own
#: genomic separation. Nothing here gates a measurement.
SEED_RADIUS_UM = 1.0

#: how many DISTINCT bins must be near a spot before it can start an
#: allele. Distinct is the whole point: three spots of one barcode is
#: one locus detected three times, which says something is wrong with
#: that round, not that there are three alleles here.
DEFAULT_MIN_BINS = 3


def bin_support(pos, bins, radius_um=SEED_RADIUS_UM):
    """(n,) how many DISTINCT bins sit within radius_um of each spot.

    The measure of "something is really here". A spurious detection is
    alone; an allele is surrounded by its own other rounds. Counting
    distinct BINS rather than spots is what keeps one barcode that
    detected three times from looking like a dense neighbourhood.
    """
    p = np.asarray(pos, dtype=np.float64)
    b = np.asarray(bins, dtype=np.int64)
    out = np.zeros(len(p), dtype=np.int64)
    if not len(p):
        return out
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    near = d <= float(radius_um)
    for i in range(len(p)):
        out[i] = len(set(b[near[i]].tolist()))
    return out


def supported_seeds(pos, bins, k, radius_um=SEED_RADIUS_UM,
                    min_bins=DEFAULT_MIN_BINS):
    """k starting centres, each with real support, and deterministic.

    FARTHEST-POINT SEEDING WAS WRONG HERE. It takes the extremes, so a
    single spurious spot 25 um away became a seed and then an entire
    allele of one spot -- the seeding error. A seed has to be somewhere
    an allele could be, which means somewhere several DIFFERENT rounds
    agree, so seeds are taken in order of bin support and no seed may
    sit inside another's reach.

    Deterministic by construction: no random start, and ties break on
    support then on distance from the cloud's centre, so the same cell
    seeds the same way every run. A trace that moved when nothing
    changed would be one nobody can check.

    Returns fewer than k centres when the cell cannot support k. That is
    an answer, not a failure: it says this cell does not look like k
    alleles, and the caller decides what to do about it.
    """
    p = np.asarray(pos, dtype=np.float64)
    if len(p) == 0 or k <= 0:
        return np.empty((0, 3))
    sup = bin_support(p, bins, radius_um)
    centre = p.mean(0)
    far = np.linalg.norm(p - centre, axis=1)
    order = sorted(range(len(p)), key=lambda i: (-int(sup[i]), -float(far[i])))
    taken = []
    for i in order:
        if sup[i] < int(min_bins):
            break
        if any(np.linalg.norm(p[i] - p[j]) <= float(radius_um) for j in taken):
            continue
        taken.append(i)
        if len(taken) == k:
            break
    return p[taken] if taken else np.empty((0, 3))


def farthest_point_seeds(pos, k):
    """k centres by farthest-point. KEPT, and not the default.

    It is the right seeding when every point is real and the question is
    coverage. It is the wrong one here, because the farthest point from
    everything is what a spurious detection looks like. supported_seeds
    says why.
    """
    p = np.asarray(pos, dtype=np.float64)
    if len(p) == 0 or k <= 0:
        return np.empty((0, 3))
    first = int(np.argmax(np.linalg.norm(p - p.mean(0), axis=1)))
    idx = [first]
    while len(idx) < k:
        d = np.min(np.linalg.norm(p[:, None, :] - p[idx][None, :, :],
                                  axis=2), axis=1)
        nxt = int(np.argmax(d))
        if nxt in idx:
            break
        idx.append(nxt)
    return p[idx]


def kmeans_cluster(spots, k, iters=50, engine=None, **kw):
    """THE DEFAULT ENGINE: Lloyd's algorithm on the positions, seeded
    farthest-point.

    An allele is compact -- its bins sit within its own radius of
    gyration, a few hundred nm -- while two alleles of one cell are
    usually microns apart, so position alone separates them most of the
    time. Most of the time is not always, and this is why resolve_paths
    exists: the ONE-spot-per-bin constraint is the information position
    does not have, and applying it afterwards fixes the pairs this step
    gets wrong without making this step pretend to know about bins.
    """
    p = spots.pos
    if k <= 0 or len(p) == 0:
        return Clustering(np.full(len(p), UNASSIGNED, dtype=np.int64),
                          np.empty((0, 3)), 0, 0.0, 'nothing to cluster')
    if k == 1:
        c = p.mean(0, keepdims=True)
        d = np.sum((p - c) ** 2)
        return Clustering(np.zeros(len(p), dtype=np.int64), c, 1, float(d),
                          'one allele: every spot is its own')
    # A SPOT WITH NO SUPPORT IS NOT CLUSTERED AT ALL. Guarding only the
    # seed was not enough: Lloyd re-centres freely, and on a cell with
    # two alleles 2 um apart and one spot 40 um away it dragged a centre
    # onto that spot and merged the two real alleles into the other
    # cluster -- the outlier did not just become an allele, it cost one.
    # Excluding it up front leaves the geometry to the spots that have
    # company, which is the only geometry an allele is in.
    radius = float(kw.get('radius_um', SEED_RADIUS_UM))
    min_bins = int(kw.get('min_bins', DEFAULT_MIN_BINS))
    sup = bin_support(p, spots.bin, radius)
    live = sup >= min_bins
    n_out = int((~live).sum())
    if not live.any():
        return Clustering(np.full(len(p), UNASSIGNED, dtype=np.int64),
                          np.empty((0, 3)), 0, 0.0,
                          'no spot has %d distinct bin(s) within %.2f um, '
                          'so nothing here looks like an allele'
                          % (min_bins, radius))
    idx = np.flatnonzero(live)
    p_all, bins_all = p, spots.bin
    p = p_all[idx]
    centres = supported_seeds(p, bins_all[idx], k, radius_um=radius,
                              min_bins=min_bins)
    if len(centres) < k:
        # THE CELL DOES NOT SUPPORT k. Said, not padded: a centre
        # invented to reach k would be an allele made of whatever was
        # left over, which is the outlier becoming a trace.
        k = len(centres)
    if k == 0:
        return Clustering(np.full(len(p_all), UNASSIGNED, dtype=np.int64),
                          np.empty((0, 3)), 0, 0.0,
                          'no position has %d distinct bin(s) within '
                          '%.2f um, so nothing here looks like an allele'
                          % (min_bins, radius))
    labels = np.zeros(len(p), dtype=np.int64)
    for _ in range(int(iters)):
        d = np.linalg.norm(p[:, None, :] - centres[None, :, :], axis=2)
        new = np.argmin(d, axis=1)
        if np.array_equal(new, labels) and _ > 0:
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                centres[j] = p[m].mean(0)
    # THE SEED GUARD IS NOT ENOUGH. Lloyd re-centres freely, so a
    # cluster seeded on a real allele can walk onto an outlier and end
    # up centred on it with one spot -- measured on a cell with two
    # alleles and one spot 40 um away, which converged to a centre
    # sitting exactly on that spot. So the same support the seed had to
    # show is required of the RESULT: a cluster whose spots come from
    # fewer than min_bins DISTINCT bins is not an allele, and saying so
    # is the whole "second trace or outlier" decision. Its spots go back
    # to UNASSIGNED rather than to the nearest survivor: attaching a
    # spot nothing supports is the other half of the same error.
    keep, dropped = [], 0
    for j in range(k):
        m = labels == j
        if m.any() and len(set(bins_all[idx][m].tolist())) >= min_bins:
            keep.append(j)
        else:
            dropped += int(m.sum())
    if len(keep) < k:
        remap = {j: i for i, j in enumerate(keep)}
        labels = np.array([remap.get(int(v), UNASSIGNED) for v in labels],
                          dtype=np.int64)
        centres = centres[keep] if keep else np.empty((0, 3))
        k = len(keep)
    if k == 0:
        return Clustering(np.full(len(p_all), UNASSIGNED, dtype=np.int64),
                          np.empty((0, 3)), 0, 0.0,
                          'no cluster drew on %d distinct bin(s)' % min_bins)
    full = np.full(len(p_all), UNASSIGNED, dtype=np.int64)
    full[idx] = labels
    on = full >= 0
    d = np.linalg.norm(p_all[on] - centres[full[on]], axis=1)
    return Clustering(full, centres, k, float(np.sum(d ** 2)),
                      'k-means over %d supported spot(s) of %d into %d '
                      'allele(s)' % (len(p), len(p_all), k)
                      + ('; %d unsupported spot(s) left out' % n_out
                         if n_out else '')
                      + ('; %d spot(s) left unassigned, their cluster '
                         'drew on too few bins to be an allele' % dropped
                         if dropped else ''))


def cluster_cell(spots, k, engine=None, **kw):
    """Which allele each spot belongs to.

    engine: a callable (spots, k, **kw) -> Clustering, defaulting to
    kmeans_cluster. THE SEAM: a learned model that reads the image, or
    one that clusters on shape rather than position, goes here.
    """
    return (engine or kmeans_cluster)(spots, k, **kw)


# ------------------------------------------- 3. one spot per bin, each

Paths = collections.namedtuple('Paths', 'rows cost moved reason')
Paths.__doc__ = """The spot each allele takes in each bin.

rows   (k, n_slots) int   the caller's row id, NO_SPOT where the allele
                          has none in that bin
cost   total distance, um, from each taken spot to its allele's centre
moved  how many spots ended up under a different allele than the
       clustering gave them -- the constraint doing work
reason in words
"""


def nearest_path(spots, clustering, n_slots, max_um=None, engine=None):
    """THE DEFAULT ENGINE: per bin, the assignment of its spots to the
    alleles that costs least.

    A bin is one round and a round images one locus, so it gives at most
    ONE spot per allele. That makes each bin a small assignment problem
    on its own -- its spots against the k allele centres -- and with k
    at 2 or 3 the exact answer is a handful of permutations, so nothing
    here is a heuristic standing in for one.

    The clustering's label is NOT taken as given. Two alleles that pass
    close in one round get both of their spots nearer to one centre,
    and position alone cannot undo that; the one-per-bin constraint can,
    and `moved` counts the times it did.

    max_um, when given, refuses a spot farther than that from its
    allele's centre rather than taking it because nothing else wanted
    it: an allele is compact, and a spot beyond its reach is more likely
    another allele's miss or a spurious detection than a real bin.
    """
    k = int(clustering.k)
    n = int(n_slots)
    rows = np.full((k, n), NO_SPOT, dtype=np.int64)
    if k <= 0 or n <= 0 or len(spots.bin) == 0:
        return Paths(rows, 0.0, 0, 'nothing to resolve')
    centres = np.asarray(clustering.centres, dtype=np.float64)
    by_bin = collections.defaultdict(list)
    for i, b in enumerate(spots.bin):
        if 0 <= b < n:
            by_bin[int(b)].append(i)
    total, moved = 0.0, 0
    for b, idxs in by_bin.items():
        d = np.linalg.norm(spots.pos[idxs][:, None, :] - centres[None, :, :],
                           axis=2)
        if max_um is not None:
            d = np.where(d <= float(max_um), d, np.inf)
        take, cost = _best_assignment(d)
        for j, i_local in enumerate(take):
            if i_local is None:
                continue
            i = idxs[i_local]
            rows[j, b] = int(spots.row[i])
            total += float(d[i_local, j])
            if clustering.labels[i] != j:
                moved += 1
        del cost
    return Paths(rows, float(total), int(moved),
                 'one spot per bin per allele over %d bin(s); %d spot(s) '
                 'moved off their cluster' % (len(by_bin), moved))


def _best_assignment(d):
    """(take, cost) minimising total distance, one spot per allele.

    d is (n_spots_in_bin, k). take[j] is the local spot index allele j
    gets, or None. Exact: a bin holds a handful of spots and k is 2 or
    3, so every assignment is enumerated rather than approximated.
    """
    n, k = d.shape
    best, best_cost = [None] * k, float('inf')
    spots = list(range(n))
    for chosen in itertools.permutations(spots, min(n, k)):
        for picks in _pad_to(chosen, k):
            c = 0.0
            ok = True
            for j, i in enumerate(picks):
                if i is None:
                    continue
                if not np.isfinite(d[i, j]):
                    ok = False
                    break
                c += float(d[i, j])
            if ok and c < best_cost:
                best, best_cost = list(picks), c
    if best_cost == float('inf'):
        return [None] * k, 0.0
    return best, best_cost


def _pad_to(chosen, k):
    """Every way of placing `chosen` into k allele slots, Nones elsewhere.

    A bin with fewer spots than alleles leaves some alleles empty, and
    WHICH of them is empty is part of the choice -- handing the spots to
    the first alleles and calling the rest missing would assign by index
    rather than by distance.
    """
    m = len(chosen)
    if m >= k:
        yield tuple(chosen[:k])
        return
    for slots in itertools.combinations(range(k), m):
        picks = [None] * k
        for s, i in zip(slots, chosen):
            picks[s] = i
        yield tuple(picks)


SplitVerdict = collections.namedtuple('SplitVerdict', 'split reason')


def never_split(spots, rows, n_slots, **kw):
    """THE DEFAULT: one allele is one path. NOT IMPLEMENTED, on purpose.

    THE QUESTION THIS STANDS IN FOR. An allele that has replicated holds
    two sister chromatids, and they are close -- far closer than two
    alleles. Position cannot separate them, and clustering should not
    try: the shape of the question is a model comparison, two paths
    against one path with several outliers, and a split that is merely
    allowed will take every noisy trace.

    WHY IT IS NOT GUESSED AT HERE. Measured on MP58 FOV 1: of 9291 spot
    pairs sharing a bin, NONE are closer than 1 um and 82% are farther
    than 2 um, median 3.75. There is no sub-micron population in this
    data at all -- two spots in one bin are two alleles or a spurious
    detection, and splitting on them would manufacture sisters that the
    images do not show. And once the cell is split spatially, a cluster
    holds a median of 1.05 spots per bin (p95 1.27, 239 clusters),
    which is one round giving one signal per allele -- the thing a
    sister split would have to argue against.

    So the default refuses, says it refused, and leaves the seam. A
    model comparison that earns a split needs data where sisters are
    resolved, and this is not it.
    """
    return SplitVerdict(False, 'sister splitting is not attempted: no '
                               'sub-micron within-bin population was '
                               'measured in the data this was built on')


def path_split(spots, rows, n_slots, engine=None, **kw):
    """Should this allele's path be two paths?

    engine: a callable (spots, rows, n_slots, **kw) -> SplitVerdict,
    defaulting to never_split. THE SEAM, and the one most worth filling:
    it is the sister-chromatid question, and it is a model comparison
    rather than a clustering.
    """
    return (engine or never_split)(spots, rows, n_slots, **kw)


def resolve_paths(spots, clustering, n_slots, engine=None, **kw):
    """Which ONE spot each allele takes in each bin.

    engine: a callable (spots, clustering, n_slots, **kw) -> Paths,
    defaulting to nearest_path. THE SEAM: a model that scores a whole
    path rather than each bin on its own -- one that knows a polymer is
    continuous, which this default does not -- goes here.
    """
    return (engine or nearest_path)(spots, clustering, n_slots, **kw)


# --------------------------------------------------------- composition


# --------------------------------------- the two questions, asked once


#: how far apart two sites must be, laterally, before their separation
#: is taken as settled. Measured on MP58 FOVs 1-4 against the stored
#: counts: two sites >= 2.0 um apart agree 83 % of the time (n=155) and
#: >= 3.0 um 86 % (n=120), consistent across all four FOVs. Spots within
#: one allele sit a few hundred nm apart, so a micron-scale gap is not a
#: close call.
CLEAR_SEPARATION_UM = 2.0


#: how many spots of one site may repeat a bin it already holds,
#: as a FRACTION of that site's spots. Not zero: 62 bins carrying ~100
#: spots will repeat a few by chance, and measured on MP58 FOVs 1-4
#: only 19 % of cells have none at all while the median is 3. Zero
#: tolerance rejected four cells in five for something normal.
DEFAULT_DUP_FRACTION = 0.15


def clearly_separated(spots, k, radius_um=None, min_bins=None,
                      dup_fraction=DEFAULT_DUP_FRACTION):
    """(verdict, labels, centres, reason) -- is this cell simply split?

    THE EXPENSIVE OBJECTIVE IS FOR THE HARD CASES. A cell whose sites
    sit microns apart with no bin claimed twice is not a close call, and
    sweeping k under an exact per-bin assignment to re-derive what the
    geometry already says costs time and changes nothing. This answers
    cheaply, and says no whenever anything is ambiguous.

    No is the safe answer and it is the default: an unclear verdict
    sends the cell to fit_paths, which is where the judgement lives.
    """
    reach = float(SEED_RADIUS_UM if radius_um is None else radius_um)
    mb = int(DEFAULT_MIN_BINS if min_bins is None else min_bins)
    if k < 1:
        return False, None, None, 'no sites proposed'
    clu = kmeans_cluster(spots, k, radius_um=reach, min_bins=mb)
    if clu.k != k or clu.k < 1:
        return False, None, None, 'the cell does not support %d site(s)' % k
    allow = float(dup_fraction)
    if clu.k == 1:
        dup = _duplicate_bins(spots.bin, clu.labels, 0)
        n_here = int((clu.labels == 0).sum())
        if dup > allow * max(n_here, 1):
            return (False, None, None,
                    'one site, but %d of its %d spot(s) repeat a bin'
                    % (dup, n_here))
        return True, clu.labels, clu.centres, 'a single compact site'
    c = np.asarray(clu.centres, dtype=np.float64)
    lat = np.linalg.norm(c[:, None, :2] - c[None, :, :2], axis=2)
    lat += np.eye(len(c)) * 1e9
    gap = float(lat.min())
    if gap < CLEAR_SEPARATION_UM:
        return (False, None, None,
                'two sites only %.2f um apart laterally' % gap)
    for j in range(clu.k):
        dup = _duplicate_bins(spots.bin, clu.labels, j)
        n_here = int((clu.labels == j).sum())
        if dup > allow * max(n_here, 1):
            return (False, None, None,
                    'site %d repeats a bin in %d of its %d spot(s)'
                    % (j, dup, n_here))
    return (True, clu.labels, c,
            '%d site(s), %.2f um apart, no bin claimed twice'
            % (clu.k, gap))


def _duplicate_bins(bins, labels, j):
    """How many spots of one site repeat a bin that site already has.

    A round images one locus, so a site holding two spots of one bin is
    either two alleles merged or a spurious detection -- either way, not
    a cell to wave through.
    """
    b = np.asarray(bins)[labels == j]
    return int(len(b) - len(set(b.tolist())))


def rows_from_labels(spots, labels, k, n_slots):
    """(k, n_slots) rows from a settled partition, one spot per bin.

    The brightest candidate where a site does hold two of a bin -- the
    same rule collapse_polymer has always used. Only reached on the
    clear path, where duplicates have already been ruled out.
    """
    rows = np.full((int(k), int(n_slots)), NO_SPOT, dtype=np.int64)
    best = {}
    for i, (b, j) in enumerate(zip(spots.bin, labels)):
        if j < 0 or not (0 <= b < n_slots):
            continue
        key = (int(j), int(b))
        a = spots.amp[i]
        if key not in best or (np.isfinite(a) and a > best[key][1]):
            best[key] = (i, a if np.isfinite(a) else -np.inf)
    for (j, b), (i, _a) in best.items():
        rows[j, b] = int(spots.row[i])
    return rows


Fit = collections.namedtuple(
    'Fit', 'k labels rows cost weights unassigned per_k reason')
Fit.__doc__ = """One cell's paths, chosen by what they cost.

k           how many paths the cost chose
labels      (n,) which path each spot went to, UNASSIGNED for none
rows        (k, n_slots) the spot row id each path takes in each bin
cost        the total, in NATS and relative to calling every spot
            clutter -- negative is better than nothing, and the scale is
            a log-likelihood rather than micrometres
weights     per path, in nats: the evidence for THAT allele, separately,
            so a cell with one strong and one marginal site reads as
            that rather than as a number
unassigned  how many spots no path would pay for
per_k       [(k, priced total), ...] -- what every k was worth, so a
            close call is visible rather than hidden behind the winner
reason      in words, including which calibration and which separation
"""


def fit_paths(spots, n_slots, k, sep, cal=None, **kw):
    """The k cheapest chains through this cell. See localization/pathfind.

    THE OBJECTIVE IS A CHAIN, NOT A CLOUD, and that is the whole of what
    changed here. A path used to be a centroid with a radius: every bin
    went to the nearest centre, a spot beyond `reach` was refused
    outright, and genomic separation never entered, so bin 1 and bin 90
    were asked the same question. Now a path is an ordered walk through
    the slots scored by its own steps, each step priced by the measured
    chain law at the separation that step actually spans
    (physics/MEASUREMENT.md), against the alternative that the spot is
    clutter.

    `sep` says what the genomic separation between two slots IS -- from
    the panel's probe design, or `pathfind.Separation.uniform` which
    assumes a regular step and says so in its source string. There is no
    default: a chain loss with no genomic axis is the thing this
    replaced.
    """
    labels, weights = pathfind.find_paths(
        spots.pos, spots.bin, int(k), sep, cal=cal, **kw)
    kk = len(weights)
    rows = pathfind.path_rows(labels, spots.bin, spots.row, kk, n_slots)
    return Fit(kk, labels, rows, float(sum(weights)), weights,
               int((labels == UNASSIGNED).sum()), [],
               '%d chain(s) over %d slot(s)' % (kk, int(n_slots)))


def choose_paths(spots, n_slots, sep, cal=None, **kw):
    """How many chains, decided by whether another one pays for itself.

    Every extra path lowers the cost -- its weight is <= 0 by
    construction -- so this is settled by a prior and a cap, not by the
    sweep finding a minimum. pathfind.DEFAULT_PATH_PRIOR_NATS carries
    what the prior is worth and what it was measured against.
    """
    res = pathfind.choose_paths(spots.pos, spots.bin, sep, cal=cal, **kw)
    rows = pathfind.path_rows(res.labels, spots.bin, spots.row, res.k, n_slots)
    return Fit(res.k, res.labels, rows, res.cost, res.weights,
               int((res.labels == UNASSIGNED).sum()), res.per_k, res.reason)



Trace = collections.namedtuple('Trace', 'rows multiplicity clustering paths')


def build_cell_traces(spots, n_slots, sep, k=None, engines=None, **kw):
    """One cell, end to end.

    The count is no longer decided before the split. choose_paths sweeps
    k and keeps the one that pays for itself, with the chain loss
    deciding the partition rather than tidying it afterwards.
    multiplicity_gate still runs and is reported beside it, because a
    proposal that disagrees with the cost is worth seeing.

    `sep` is where every step's genomic separation comes from and has no
    default; `pathfind.Separation` builds one from a probe design, or
    from an assumed uniform step that labels itself as assumed.

    k: given, it is used as-is and no sweep happens. engines: as before,
    for the pieces that are still separable.

    THE CLEAR-CASE SHORTCUT IS GONE. It answered a cell whose sites sit
    microns apart from a DIFFERENT objective -- k-means plus a
    one-per-bin repair -- which was defensible while the real objective
    was a slow enumeration over every bin. It is not any more: the chain
    loss is a DP over the slot order, measured at 0.11 s for a 282-spot
    cell, and two objectives answering the same question is how two
    answers start disagreeing.
    """
    engines = engines or {}
    mult = multiplicity_gate(spots, n_slots,
                             engine=engines.get('multiplicity'),
                             **(kw.get('multiplicity') or {}))
    opts = dict(kw.get('paths') or {})
    if k is not None:
        fit = fit_paths(spots, n_slots, int(k), sep, **opts)
        return Trace(fit.rows, mult, fit, fit)
    fit = choose_paths(spots, n_slots, sep, **opts)
    return Trace(fit.rows, mult, fit, fit)
