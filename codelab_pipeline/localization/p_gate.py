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

IT GATES ON WHATEVER NUMBER THE RESULT HAS, AND SAYS WHICH ONE. Every
engine reports something in (0, 1] that ranks its own candidates, so
thresholding is mechanically fine everywhere -- and refusing to open on
v1 and v2 would be this module deciding for a person what they are
allowed to look at. What it must never do is let the two be CONFUSED:

    p_exist   the classifier's calibrated existence probability. A
              threshold on it means the same thing on any run, any
              bundle, any engine that produces one. v3 only.

    p         a per-engine quality scalar -- a constant 1.0 for the
              Gaussian engine, a contrast for anchor-v1, an
              occupancy-times-CI product for anchor-v2, an affine NCC
              for psf-match. Ranks within one engine and one
              parameterisation, and means nothing across them.

`quantity()` names which of the two a result set carries and every other
function here takes that answer, so a threshold is always attached to a
named quantity rather than floating free. The dialog prints it on the
axis. In practice v1's p is 1.0 for everything -- its histogram is one
bar and a threshold does nothing, which the picture shows better than a
refusal would -- and v2's is a ranking nobody has calibrated; that is a
reason not to USE the gate there, not a reason to hide it.

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


CALIBRATED = 'p_exist'
QUALITY = 'p'


def _field(spot, name):
    """One numeric field off a LocalizedSpot, a dict or anything else.

    Tolerant because this runs on things that have crossed a store
    boundary as often as on live objects.
    """
    if spot is None:
        return float('nan')
    v = spot.get(name) if isinstance(spot, dict) else getattr(spot, name, None)
    try:
        return float('nan') if v is None else float(v)
    except (TypeError, ValueError):
        return float('nan')


def quantity(spots):
    """Which number this result set can be gated on, or None.

    ONE QUANTITY FOR THE WHOLE SET, never a per-spot fallback. Mixing a
    calibrated probability and a quality score into one histogram would
    put two different scales on one axis and let a threshold mean
    different things for different spots in the same run -- the exact
    confusion this module exists to prevent. p_exist wins when ANY spot
    has one; otherwise p.
    """
    for s in (spots or ()):
        if np.isfinite(_field(s, CALIBRATED)):
            return CALIBRATED
    for s in (spots or ()):
        if np.isfinite(_field(s, QUALITY)):
            return QUALITY
    return None


def is_calibrated(spots):
    """Whether the number being gated is a probability or a ranking."""
    return quantity(spots) == CALIBRATED


def p_of(spot, which=None):
    """The gated value of one spot, or NaN.

    `which` names the quantity; without it, p_exist if the spot has one
    and p otherwise -- which is right for a single spot and wrong for a
    set, so set-level callers resolve it once with quantity().
    """
    if which is None:
        v = _field(spot, CALIBRATED)
        return v if np.isfinite(v) else _field(spot, QUALITY)
    return _field(spot, which)


def values(spots, which=None):
    """Every finite gated value in `spots`, as an array. May be empty."""
    which = which or quantity(spots)
    if which is None:
        return np.zeros(0)
    v = np.asarray([_field(s, which) for s in (spots or ())], float)
    return v[np.isfinite(v)] if v.size else v


def available(spots):
    """Whether ANY number here can be thresholded at all.

    True for every engine that reported something finite. False only for
    an empty result -- see quantity() for which number it would be, and
    is_calibrated() for whether that number is a probability.
    """
    return bool(values(spots).size)


def deny(spot, threshold=DEFAULT_THRESHOLD, which=None):
    """True when this spot falls below the threshold.

    A spot with NO value for the gated quantity is NEVER denied. Whether
    it is a v1 spot in a v3 result or a v3 candidate the matcher never
    scored, "this number is absent" is not "this number is low", and
    treating it as low would silently delete spots nobody judged.
    """
    p = p_of(spot, which)
    return bool(np.isfinite(p) and p < float(threshold))


def apply(spots, threshold=DEFAULT_THRESHOLD, which=None):
    """The survivors, in the order they came in."""
    which = which or quantity(spots)
    return [s for s in (spots or ()) if not deny(s, threshold, which)]


def annotate(spots, threshold=DEFAULT_THRESHOLD, which=None):
    """[(spot, value, denied), ...] -- what the gate did, not only what
    it removed. The dialog draws its examples from this."""
    which = which or quantity(spots)
    t = float(threshold)
    out = []
    for s in (spots or ()):
        p = p_of(s, which)
        out.append((s, p, bool(np.isfinite(p) and p < t)))
    return out


def histogram(spots, bins=DEFAULT_BINS, lo=0.0, hi=1.0, which=None):
    """(counts, edges) over [0, 1] -- the picture a threshold is picked from.

    The range is FIXED at [0, 1] rather than taken from the data. Both
    quantities live in (0, 1] by contract, and an auto-range would redraw
    the same distribution differently every time a run happened to
    contain nothing confident -- which is exactly when a person most
    needs to see that the mass is all at the bottom. It is also what
    makes v1 legible: a single bar at 1.0 says "this engine ranks
    nothing" far more plainly than an axis rescaled around it.
    """
    v = values(spots, which)
    if not v.size:
        return np.zeros(int(bins), dtype=int), np.linspace(lo, hi, int(bins) + 1)
    counts, edges = np.histogram(v, bins=int(bins), range=(float(lo), float(hi)))
    return counts, edges


def summary(spots, threshold=DEFAULT_THRESHOLD, which=None):
    """What a threshold costs, in the numbers a person is deciding with."""
    which = which or quantity(spots)
    all_v = np.asarray([p_of(s, which) for s in (spots or ())], float)
    v = all_v[np.isfinite(all_v)]
    t = float(threshold)
    kept = int((v >= t).sum())
    return dict(
        quantity=which,
        calibrated=(which == CALIBRATED),
        threshold=t,
        n_total=int(all_v.size),
        n_scored=int(v.size),
        n_unscored=int(all_v.size - v.size),
        n_kept=kept,
        n_denied=int(v.size - kept),
        kept_frac=(kept / v.size) if v.size else float('nan'),
        # A DEGENERATE COLUMN IS WORTH SAYING OUT LOUD. v1 reports a
        # constant 1.0, so every threshold at or below it keeps
        # everything and every threshold above it keeps nothing -- there
        # is no operating point in between, and a person staring at one
        # bar deserves to be told that rather than left to infer it.
        degenerate=bool(v.size and float(v.min()) == float(v.max())),
        p_median=float(np.median(v)) if v.size else float('nan'),
        p_min=float(v.min()) if v.size else float('nan'),
        p_max=float(v.max()) if v.size else float('nan'))


def examples(spots, threshold=DEFAULT_THRESHOLD, n=4, rng=None, band=None,
             which=None):
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
    which = which or quantity(spots)
    t = float(threshold)
    kept, denied = [], []
    for s, p, d in annotate(spots, t, which):
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
