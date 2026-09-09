"""
The threshold a person picks off a histogram, applied after everything.

THIS IS A POSTERIOR GATE, the second one, and it sits beside edge_gate
for the same reason that one does: "is a spot here?" is what the model
answers, and "now that we have every answer, which do we keep?" is a
different question asked at a different time -- after detection, after
sub-voxel refinement, with the whole field's answers in hand and a
person looking at their distribution.

Keeping it out of the engine is the whole point. An engine that applied
its own threshold would be choosing for the reviewer AND hiding the
evidence they would choose from: the histogram this module draws is the
histogram of the very numbers a pre-filtered engine would have thrown
away.

IT IS AVAILABLE ONLY WHERE p_exist IS. LocalizedSpot.p is a per-engine
quality scalar -- a constant 1.0 for the Gaussian engine, a contrast for
anchor-v1, an occupancy-times-CI product for anchor-v2, an affine NCC for
psf-match -- and a threshold set on one of those means nothing on
another and is not a probability on any of them. p_exist is the
classifier's calibrated existence probability and NaN everywhere else,
so `available()` answers honestly and a caller that respects it cannot
offer a gate that would be meaningless. THIS IS WHY p_exist IS ITS OWN
FIELD: in a shared `p` slot the gate would look applicable to v1 and v2
and would silently be nonsense there.

WHAT THE DEFAULT IS, AND WHY IT IS NOT 0.5 BY FIAT. The classifier saves
its own operating point (SpotClassifier.threshold, 0.5 today) and that
is the number the training run's own report was written against, so it
is the honest starting position for the slider -- not a constant chosen
here. `DEFAULT_THRESHOLD` is only the fallback for a caller with no
classifier in hand.

NOTHING IS DELETED. apply() returns the survivors and annotate() returns
every spot with its verdict, so a caller can show what a threshold
removes as easily as what it keeps -- which is the entire purpose of
showing a person the histogram before they commit to a number.
"""
import numpy as np

DEFAULT_THRESHOLD = 0.5

# Bin count for the preview histogram. Enough to show the shape of a
# distribution that is strongly bimodal in practice -- MEASURED on 2,068
# human-labelled MP58/RNA spots, the classifier puts positives at a
# median 0.967 and negatives at 0.018 -- without turning the tail into
# single-count noise a person would try to read.
DEFAULT_BINS = 50


def p_of(spot):
    """A spot's calibrated existence probability, or NaN.

    Tolerant of a plain tuple or dict as well as a LocalizedSpot, because
    this runs on things that have crossed a store boundary as often as on
    live objects.
    """
    if spot is None:
        return float('nan')
    if isinstance(spot, dict):
        v = spot.get('p_exist')
    else:
        v = getattr(spot, 'p_exist', None)
    try:
        return float('nan') if v is None else float(v)
    except (TypeError, ValueError):
        return float('nan')


def values(spots):
    """Every finite p_exist in `spots`, as an array. May be empty."""
    v = np.asarray([p_of(s) for s in (spots or ())], float)
    return v[np.isfinite(v)] if v.size else v


def available(spots):
    """Whether a p-gate means anything for this result set.

    False for v1, v2 and psf-match, whose p is a quality score and whose
    p_exist is NaN -- a caller should not offer the dialog at all rather
    than offer a threshold on numbers that are not probabilities.
    """
    return bool(values(spots).size)


def deny(spot, threshold=DEFAULT_THRESHOLD):
    """True when this spot falls below the threshold.

    A spot with NO p_exist is NEVER denied. The gate is not applicable to
    it, and treating "this engine does not answer that question" as "the
    answer is no" would silently delete every spot from every other
    engine the moment a caller forgot to check available().
    """
    p = p_of(spot)
    return bool(np.isfinite(p) and p < float(threshold))


def apply(spots, threshold=DEFAULT_THRESHOLD):
    """The survivors, in the order they came in."""
    return [s for s in (spots or ()) if not deny(s, threshold)]


def annotate(spots, threshold=DEFAULT_THRESHOLD):
    """[(spot, p_exist, denied), ...] -- what the gate did, not only what
    it removed. The dialog draws its examples from this."""
    t = float(threshold)
    out = []
    for s in (spots or ()):
        p = p_of(s)
        out.append((s, p, bool(np.isfinite(p) and p < t)))
    return out


def histogram(spots, bins=DEFAULT_BINS, lo=0.0, hi=1.0):
    """(counts, edges) over [0, 1] -- the picture the threshold is picked from.

    The range is FIXED at [0, 1] rather than taken from the data. A
    probability's axis is the unit interval, and an auto-range would
    redraw the same distribution differently every time a run happened to
    contain no confident spots, which is exactly when a person most needs
    to see that the mass is all at the bottom.
    """
    v = values(spots)
    if not v.size:
        return np.zeros(int(bins), dtype=int), np.linspace(lo, hi, int(bins) + 1)
    counts, edges = np.histogram(v, bins=int(bins), range=(float(lo), float(hi)))
    return counts, edges


def summary(spots, threshold=DEFAULT_THRESHOLD):
    """What a threshold costs, in the numbers a person is deciding with."""
    all_v = np.asarray([p_of(s) for s in (spots or ())], float)
    v = all_v[np.isfinite(all_v)]
    t = float(threshold)
    kept = int((v >= t).sum())
    return dict(
        threshold=t,
        n_total=int(all_v.size),
        n_scored=int(v.size),
        n_unscored=int(all_v.size - v.size),
        n_kept=kept,
        n_denied=int(v.size - kept),
        kept_frac=(kept / v.size) if v.size else float('nan'),
        p_median=float(np.median(v)) if v.size else float('nan'),
        p_min=float(v.min()) if v.size else float('nan'),
        p_max=float(v.max()) if v.size else float('nan'))


def examples(spots, threshold=DEFAULT_THRESHOLD, n=4, rng=None, band=None):
    """A random sample of spots to LOOK at for a chosen threshold.

    Returns (kept, denied): `n` of each, drawn at random rather than by
    rank. THE RANK ORDER IS THE WRONG SAMPLE: the top of the kept pile
    and the bottom of the denied pile are the easy cases, and a person
    checking a threshold needs to see what sits AT it -- so `band`, when
    given, restricts the draw to spots within that distance of the
    threshold, which is where a threshold is actually right or wrong.

    Random, and re-drawn on request, because one fixed sample invites
    reading four particular spots as the whole distribution.
    """
    rng = np.random.default_rng() if rng is None else rng
    t = float(threshold)
    kept, denied = [], []
    for s, p, d in annotate(spots, t):
        if not np.isfinite(p):
            continue
        if band is not None and abs(p - t) > float(band):
            continue
        (denied if d else kept).append(s)

    def draw(pool):
        if not pool:
            return []
        k = min(int(n), len(pool))
        idx = rng.choice(len(pool), size=k, replace=False)
        return [pool[int(i)] for i in idx]
    return draw(kept), draw(denied)
