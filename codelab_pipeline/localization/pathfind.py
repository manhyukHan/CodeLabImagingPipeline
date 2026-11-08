"""Path optimization: a chain through the loci, not a cloud around a centre.

WHAT WAS WRONG WITH THE OLD OBJECTIVE, and it was the objective rather
than the cost function. Each bin was assigned to the nearest path
CENTROID, so a path was a ball: it had a centre, a radius
(`DEFAULT_REACH_UM`, 1.0 um at every genomic separation), and no notion
of order. Two consequences, both measured:

  - an allele larger than the radius had its outer half structurally
    excluded (`d = where(d <= reach, d, inf)`), and on chr19, whose
    alleles run to Rg 1.50 um, 52% of the bins bottom-up dropped had a
    spot present, unassigned, ALL of them beyond reach;
  - genomic separation did not enter at all, so readout 1 and readout 90
    were asked the same question.

A CHROMATIN FIBRE IS A CHAIN, so the objective here is a chain. A path
is an ordered walk through the polymer slots -- at most one spot per
slot, slots strictly increasing -- and it is scored by its own steps:

    cost(path) = sum over consecutive assigned loci of
                 -log N(x_next - x_prev; 0, C_chain(s))

`s` is the real genomic separation of the two slots the step connects,
from the panel's probe design, so a step over a gap is correctly asked
about the longer separation it actually spans. There is no centre and no
radius.

WHAT A SPOT COSTS IF NOBODY TAKES IT. The objective is a likelihood, so
an unclaimed spot is not free -- it is clutter, drawn from a Poisson
field of rate `rho` per um^3, and costs `-log rho` nats. Writing both
terms out, the whole cell's negative log-likelihood is

    -N log rho  +  sum over paths, over steps of
                   [-log N_chain(step) + log rho]

because a path of m loci removes m clutter terms and pays one back for
its own first locus, which carries no information. The first term is a
constant of the cell, so MINIMIZING THE TOTAL IS MINIMIZING A SUM OF
STEP WEIGHTS, each

    w(j -> i) = -log N(x_i - x_j; 0, C_chain(s(b_j, b_i)))  +  log rho

A negative weight is a step worth taking. A one-locus path weighs zero,
which is right: a lone spot cannot be told from clutter.

A RATE, NOT A VOLUME, and this was the second thing wrong here. Charging
a flat `log V` for an unclaimed spot says clutter is rare wherever you
look, so in a cell full of spurious detections every chance pair of
neighbours looked like evidence: measured on the synthetic cell below,
one true allele plus 120 uniform clutter spots returned k = 3 in six
seeds out of six, built out of clutter. The rate fixes it, because it is
the cell's OWN spot density: more spots means a chance neighbour is
cheaper to explain as clutter, so a step has to be that much better to
be worth taking. Same cell, same seeds, with the rate: k = 1 throughout.

rho is taken as (all spots) / (nuclear volume), which over-states it --
some of those spots are the real loci -- and over-stating it is the
conservative direction: it makes paths shorter and fewer, never longer.

THE CHAIN COMPONENT ALONE, NOT `physics.pair_cost`. `pair_cost` is the
MIXTURE -- chain plus the measured contaminant (tau_xy 0.685, tau_z
1.177), which is "a second allele, or a wrong pick". Inside a per-step
term that contaminant is exactly the alternative this module already
has two better names for: a different path, or clutter. Using it anyway
double-counts the escape hatch, and makes a step ONTO the other allele
cheap in precisely the case where the two have to be told apart.

Measured head to head, 8 seeds per case, everything else held fixed:

    two alleles      2.0 um   1.0 um   0.6 um   0.4 um
    chain term        1.000    0.996    0.933    0.934   mean purity
    mixture           1.000    0.996    0.933    0.885

So it changes nothing until the alleles overlap, and then it helps.
Both give k correctly in every case, clutter included. The honest
summary is that this is a modelling argument with a small measured
confirmation, not a large effect.

The mixture is not retracted -- it is right where it was measured, as
the description of a pair in a STORED trace, and `p_same_chain` scores
one. It is the wrong term for a model that represents the second allele
explicitly.

THAT MAKES IT A MINIMUM-COST PATH COVER of a DAG (slots give the
ordering), the standard shape of multi-object tracking, and it is solved
the standard way: the single best path by dynamic programming over the
slot order, its spots removed, repeat -- then a refinement sweep that
lifts each path out in turn and re-derives it against everything now
free. Re-deriving minimizes over a superset of what that path already
was, so the sweep can only lower the total.

THE NUCLEAR VOLUME IS THE ONE KNOB, and it is visible rather than buried:
`acceptance_um(s)` reports the separation at which a step stops paying
for itself, which is what `reach` was trying to be and could not --
because it MOVES WITH s, and because it is a consequence of the model
rather than a bound bolted onto it. Nothing is refused outright either:
past that distance the locus is called clutter at a FIXED price, so one
bad locus cannot dominate a path the way an `inf` wall did.

MEASURED ON THE REAL STORE, not only on the fixture: 7 FOVs over three
experiments -- chr19 downstream (FOV 17/27/28/31), MP58 on G: (10/11)
and MP58 on Q: (1) -- every FOV that carries both cell-wide readout
spots and existing traces AND agreed A/B/C labels. 1066 alleles, 0.1 s
per cell, separations from each panel's own design on `E:\\Oligos`.

THE CHAIN TERM IS WHAT BINDS, and it is one Gaussian where the data is
not. The decisive run offers an A allele NOTHING but its own
fiducial-corrected loci and forces k = 1, so neither clutter nor the
path count can be blamed:

    80 A alleles, own corrected loci, k = 1   recovery median 0.77
                                              recovery == 1.00 in 4 (5%)

and the reason is visible step by step. Of 2427 consecutive real steps,
31% lie beyond `acceptance_um` for their own genomic separation, and the
distance distribution is far heavier-tailed than the single 3D Gaussian
`-log N_chain` assumes -- p90/p50 measured 2.7-3.1 against the
Gaussian's 1.63, in EVERY slot-gap bucket, so it is the shape and not a
mis-set scale:

    slot gap   steps   s median   p50     p90    p90/p50   beyond acc.
    1           1344      50 kb   0.39    1.21      3.10         30%
    2            542     100 kb   0.44    1.20      2.75         28%
    3-4          367     200 kb   0.55    1.50      2.74         35%
    5-9          132     300 kb   0.53    1.58      3.00         29%

This is the same conclusion the R(s) measurement reached from the other
side (MEASUREMENT.md): P(r|s) has to be a MIXTURE, not one Gaussian.
Until it is, a third of real steps cost more than calling the locus
clutter, and a true chain breaks there.

WHAT A FIDUCIAL CORRECTION IS AND IS NOT WORTH. `adj_coordinate` is
alignment-only, so a step between two rounds inherits the difference of
their fiducial drifts -- on chr19 a lateral sd of 0.086 um (negligible)
and an AXIAL sd of 1.646 um, a median 0.334 um per step against a real
step of 0.44 um. Correcting it helps and does not rescue: holding the
spot set fixed and moving only the frame,

    pool          frame                      loci on a path (chr19)
    every spot    alignment only                            15%
    every spot    + the cell's median drift                 16%
    every spot    + that allele's own drift                 16%
    the trace's   alignment only                            71%
    the trace's   + that allele's own drift                 79%

so the frame is worth 8 points and WHICH POINTS ARE OFFERED is worth 56.
An earlier version of this docstring said the pathfinder needs corrected
input and called that the finding; that compared two arms differing in
both the frame and the spot set, and the spot set was doing the work.

THE TAIL IS NOT A MATCHING ARTEFACT. A stored locus is tied to a spot by
nearest neighbour within the same round, and that cut could manufacture
the tail by pairing a locus with the wrong spot. It does not: tightening
it from 350 nm to 100 nm -- where the two independent fits agree to a
tenth of a micron and are certainly the same object -- leaves every
number standing.

    cut     A alleles   k=1 recovery   steps beyond acc.   p90/p50 gap 1
    100 nm         36           0.82                 26%            3.00
    200 nm         64           0.80                 30%            3.12
    350 nm         80           0.77                 31%            3.10

Of an A allele's 3356 stored loci, 81% have a spot of their round within
350 nm, 17% have one further away (the nearest-spot distance is bimodal:
p50 162 nm, p90 980 nm, p99 7.5 um -- widening to 1 um buys 8 points and
to 2 um only 2 more), and 2% have no spot of that round in the cell at
all. So the unmatched loci are not missing detections -- the permissive
route offers a median 171 spots per cell against a trace of 30 -- they
are places where the two routes disagree by microns about where the
locus is.

WHAT NONE OF IT COVERS: the target is the current anchor route's own
answer, not truth, and that route picks the same allele several times.
The heavy tail could still be real polymer heterogeneity or wrong loci
in the answer sheet; separating those needs loci confirmed independently
of either route.

UNITS. Positions in micrometres, genomic separation in kb, every cost in
nats.
"""
import collections
import math

import numpy as np

from codelab_pipeline.physics import chain as physics

#: a spot that belongs to no path
UNASSIGNED = -1

#: a slot with no spot for this path
NO_SPOT = -1

#: The volume an unclaimed spot could be anywhere in, um^3 -- a nucleus
#: roughly 10 um across. With the cell's own spot count it gives the
#: clutter rate rho = n / V, and `-log rho` is what every step has to
#: beat. It is the only free number in the loss. Raising it makes paths
#: greedier (clutter becomes a worse explanation, so more is worth
#: attaching); lowering it makes them stricter. `acceptance_um` turns it
#: into a distance, so the choice can be read rather than guessed at.
CLUTTER_VOLUME_UM3 = 500.0

#: What a new path must beat before it is believed, in nats. Every extra
#: path lowers the cost -- its weight is <= 0 by construction, exactly as
#: the old loss's k never rose -- so this and the cap are the whole of
#: what stops k from running away.
#:
#: MEASURED, on synthetic cells drawn from the chain law itself (the two
#: populations are two orders of magnitude apart, so this is a choice
#: between them and not a tuned threshold):
#:
#:   a SPURIOUS second path, over 480 one-allele cells at 0 to 240
#:   clutter spots, two panel shapes:   worst gain 12.2 nats
#:   a REAL second allele, by how many rounds detected it:
#:       4.5 loci   -2.8      17 loci   -29
#:       6.9 loci   -7.7      31 loci   -60
#:      10.5 loci  -17.3      53 loci  -118
#:
#: 15 nats clears every spurious path measured (0.0% of 480) and costs a
#: second allele seen in fewer than about 9 rounds, which is the trade
#: being made and the direction to make it in: a spurious path puts real
#: loci on an allele that does not exist, while a missing one leaves a
#: trace that stops short and says so.
DEFAULT_PATH_PRIOR_NATS = 15.0

#: Fewer loci than this is not a chain, whatever it costs. Two spots
#: near each other are a coincidence a nucleus full of spots produces
#: readily; the step weight alone cannot express that.
DEFAULT_MIN_PATH_LOCI = 3

#: How many paths may be found at all. The cost cannot say "a nucleus
#: does not hold eight copies of this chromosome", so the bound is
#: applied as a bound.
DEFAULT_K_MAX = 3


# ----------------------------------------------------- genomic distance

class Separation:
    """Slot to slot genomic distance in kb -- NaN where no chain runs.

    NaN is not missing data, it is a refusal: two slots on different
    chromosomes have no contour length between them, and a step over
    that pair must never be scored as though they did. A slot the design
    does not carry is NaN for the same reason.
    """

    def __init__(self, matrix, source='unknown'):
        self.matrix = np.asarray(matrix, dtype=np.float64)
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError('separation must be square, got %r'
                             % (self.matrix.shape,))
        self.source = str(source)

    def __len__(self):
        return len(self.matrix)

    def __repr__(self):
        return 'Separation(%d slots, %s)' % (len(self), self.source)

    @classmethod
    def from_design(cls, design, readout_ids):
        """The panel's own coordinates: slot -> readout id -> midpoint.

        readout_ids is one id per polymer slot, which is what
        analysis.polymer.genomic_bins returns. Never
        |delta slot| x resolution: a panel covering two regions has a gap
        far wider than its step, and a design may skip a readout that
        failed specificity, which is a gap of two steps and not one.
        """
        ids = [int(r) for r in readout_ids]
        n = len(ids)
        m = np.full((n, n), np.nan)
        for a in range(n):
            for b in range(a + 1, n):
                s = design.separation_kb(ids[a], ids[b])
                if s is not None:
                    m[a, b] = m[b, a] = float(s)
        return cls(m, 'probe design %s' % getattr(design, 'source', '?'))

    @classmethod
    def uniform(cls, n_slots, step_kb):
        """|delta slot| x step. A FALLBACK, and it says so.

        Right only where the panel is one contiguous region with no
        skipped readout. It exists because a store with no design
        recorded still has to be analysable, and because stating the
        assumption in `source` is better than an analysis quietly
        inventing a coordinate.
        """
        n = int(n_slots)
        idx = np.arange(n, dtype=np.float64)
        m = np.abs(idx[:, None] - idx[None, :]) * float(step_kb)
        np.fill_diagonal(m, np.nan)
        return cls(m, 'uniform %.3g kb/slot (ASSUMED, no probe design)'
                   % float(step_kb))

    def of(self, bins):
        """(n, n) kb between the slots of these spots; NaN where refused."""
        b = np.asarray(bins, dtype=np.int64)
        ok = (b >= 0) & (b < len(self.matrix))
        safe = np.where(ok, b, 0)
        out = self.matrix[np.ix_(safe, safe)]
        bad = ~(ok[:, None] & ok[None, :])
        return np.where(bad, np.nan, out)


# ------------------------------------------------------------ the loss

def clutter_nats(n_spots, volume_um3=None):
    """-log rho -- what one unclaimed spot costs, rho = n / V.

    The cell's own density, so a crowded cell is harder to build a path
    in than an empty one. n is every spot, which over-states the clutter
    rate and therefore under-states what a step is worth: the
    conservative direction.
    """
    v = float(CLUTTER_VOLUME_UM3 if volume_um3 is None else volume_um3)
    return math.log(v / max(float(n_spots), 1.0))


def step_costs(pos, bins, sep, cal=None):
    """(n, n) -log N_chain of stepping from spot j to spot i, +inf if barred.

    Barred means the step is not a step: same slot or backwards (a path
    walks the polymer in one direction and takes at most one spot per
    slot), or a pair the design gives no separation for.

    Everything else is finite. The old loss wrote `inf` beyond a radius;
    here a distant locus simply stops paying for itself and is called
    clutter instead.

    The clutter price is NOT in here, so it can be re-estimated (see
    find_paths) without recomputing the physics.
    """
    p = np.asarray(pos, dtype=np.float64)
    b = np.asarray(bins, dtype=np.int64)
    n = len(p)
    out = np.full((n, n), np.inf)
    if n < 2:
        return out
    s = sep.of(b)
    forward = b[None, :] > b[:, None]          # [j, i]: slot of i after j
    ok = forward & np.isfinite(s)
    if not ok.any():
        return out
    d = p[None, :, :] - p[:, None, :]          # [j, i] = x_i - x_j
    s_safe = np.where(ok, s, 1.0)
    cost = -physics.log_p_chain(d, s_safe, cal or physics.Calibration())
    out[ok] = cost[ok]
    return out


def step_weights(pos, bins, sep, cal=None, n_clutter=None, volume_um3=None):
    """step_costs minus what the spot would cost as clutter.

    n_clutter defaults to every spot, which is the first pass's
    assumption and the conservative one.
    """
    costs = step_costs(pos, bins, sep, cal)
    n = len(costs) if n_clutter is None else n_clutter
    return costs - clutter_nats(n, volume_um3)


def acceptance_um(s_kb, n_spots, cal=None, volume_um3=None, axis='xy',
                  hi=20.0):
    """Where a step stops paying for itself, um. The honest `reach`.

    Beyond this a locus is better called clutter than attached, so it is
    the distance the loss actually works at -- but it is derived from the
    model rather than imposed on it, and it MOVES WITH s. Infinity means
    the clutter density is so low that no separation is ever worse than
    calling the spot clutter; that is a statement about `volume_um3`,
    not about the chain.

    IT TURNS OVER past roughly a megabase, and that is not a defect. A
    step across a huge genomic gap has a broad distribution, so its
    density is low even at zero separation, and the normalizing term
    eats the budget the widening variance just handed it. The step stops
    being worth much at ANY distance -- the same statement MEASUREMENT.md
    makes about the mixture weight being unidentifiable at large s, and
    the reason `saturation_um` returns infinity there. A route that
    leans on long-range steps to hold a path together is leaning on
    nothing; what holds a path together is its short steps.

    At a 100-spot cell with the module defaults:

        s            25 kb   100 kb   1 Mb   4 Mb
        acceptance   0.43    0.54     0.70   0.65 um
    """
    cal = cal or physics.Calibration()
    lam = clutter_nats(n_spots, volume_um3)

    def cost(r):
        d = np.zeros(3)
        d[2 if axis == 'z' else 0] = r
        return float(-physics.log_p_chain(d, s_kb, cal))

    if cost(hi) < lam:
        return float('inf')
    lo = 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if cost(mid) < lam:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# -------------------------------------------------------- the algorithm

def best_path(w, bins, usable=None):
    """The single cheapest chain through `usable` spots. (indices, weight).

    Dynamic programming over slot order, which is a topological order of
    the step graph because a step always goes to a later slot. For each
    spot, the best chain ending there either starts there (weight 0) or
    extends the best chain ending at some spot in an earlier slot:

        dp[i] = min(0, min over j in earlier slots of dp[j] + w[j, i])

    Spots of the SAME slot are finalized together, which is what keeps a
    path from taking two spots of one round -- the one-locus-per-round
    constraint, expressed as the graph's shape rather than as a repair
    afterwards.

    O(n^2) with the inner loop vectorized. Returns ([], 0.0) when no
    chain is worth anything, which is an answer: this cell has no second
    allele in it.
    """
    n = len(bins)
    live = (np.ones(n, dtype=bool) if usable is None
            else np.asarray(usable, dtype=bool).copy())
    b = np.asarray(bins, dtype=np.int64)
    dp = np.zeros(n, dtype=np.float64)
    back = np.full(n, -1, dtype=np.int64)
    done = []
    for slot in sorted({int(v) for v in b[live]}):
        grp = np.flatnonzero(live & (b == slot))
        if done:
            prev = np.asarray(done, dtype=np.int64)
            cand = dp[prev][:, None] + w[np.ix_(prev, grp)]
            j = np.argmin(cand, axis=0)
            best = cand[j, np.arange(len(grp))]
            take = best < 0.0
            dp[grp] = np.where(take, best, 0.0)
            back[grp] = np.where(take, prev[j], -1)
        done.extend(grp.tolist())
    if not done:
        return [], 0.0
    end = int(min(done, key=lambda i: dp[i]))
    if dp[end] >= 0.0:
        return [], 0.0
    out = [end]
    while back[out[-1]] >= 0:
        out.append(int(back[out[-1]]))
    return out[::-1], float(dp[end])


def find_paths(pos, bins, k, sep, cal=None, volume_um3=None, iters=4,
               min_loci=None, costs=None, clutter_rounds=4):
    """Up to k vertex-disjoint chains, cheapest first. (labels, weights).

    Extract the best path, take its spots out, extract again. Then sweep:
    lift each path out in turn and re-derive it against everything free
    including its own spots. The re-derivation minimizes over a superset
    of what the path already was, so the sweep is monotone -- it cannot
    make the total worse, and it is what undoes the greediness of taking
    the first path first. `_uncross` then makes the one move neither can.

    A path shorter than `min_loci` is dropped and its spots freed: the
    step weight cannot express "two nearby spots are a coincidence", and
    in a nucleus holding a hundred spots they constantly are.

    THE CLUTTER RATE IS RE-ESTIMATED FROM WHAT IS LEFT OVER, which is
    `clutter_rounds`, and it matters more than anything else here. The
    first pass has to assume every spot might be clutter, and that
    over-charges every step in a clean cell: measured on two clean
    alleles, the loss then declined 3.5% of the REAL loci on a 60-slot
    50 kb panel and 11.5% on a 91-slot 200 kb one -- the same complaint
    the old `reach` earned, arriving by a different road. Taking the
    rate from the spots the previous pass left unclaimed gives 0.2% and
    0.1%, and changes nothing in a dirty cell: one allele among 240
    clutter spots still returns k = 1 with no clutter spot on the path,
    before and after.

    It is a fixed point, not a fit: the rate stops moving after two or
    three rounds, and the floor of one unclaimed spot keeps a perfectly
    clean cell from driving the price to infinity.
    """
    p = np.asarray(pos, dtype=np.float64)
    n = len(p)
    ml = int(DEFAULT_MIN_PATH_LOCI if min_loci is None else min_loci)
    labels = np.full(n, UNASSIGNED, dtype=np.int64)
    if n == 0 or k <= 0:
        return labels, []
    if costs is None:
        costs = step_costs(p, bins, sep, cal)
    n_clutter = n
    best = ([], [])
    for _ in range(max(1, int(clutter_rounds))):
        w = costs - clutter_nats(n_clutter, volume_um3)
        paths = _extract(w, bins, n, int(k), ml, int(iters))
        taken = sum(len(x) for x in paths)
        best = (paths, w)
        nxt = max(n - taken, 1)
        if nxt == n_clutter:
            break
        n_clutter = nxt
    paths, w = best
    weights = []
    for j, idx in enumerate(paths):
        labels[idx] = j
        weights.append(_path_weight(w, idx))
    return labels, weights


def _extract(w, bins, n, k, min_loci, iters):
    """k disjoint chains under a FIXED set of step weights."""
    free = np.ones(n, dtype=bool)
    paths = []
    for _ in range(k):
        idx, _weight = best_path(w, bins, free)
        if len(idx) < min_loci:
            break
        paths.append(idx)
        free[idx] = False
    for _ in range(iters):
        moved = False
        for j in range(len(paths)):
            pool = free.copy()
            pool[paths[j]] = True
            idx, _weight = best_path(w, bins, pool)
            if len(idx) < min_loci:
                continue
            if idx != paths[j]:
                moved = True
                free[paths[j]] = True
                free[idx] = False
                paths[j] = idx
        moved = _uncross(w, bins, paths) or moved
        if not moved:
            break
    return paths


def _path_weight(w, idx):
    return float(sum(w[a, b] for a, b in zip(idx, idx[1:]))) if len(idx) > 1 else 0.0


def _uncross(w, bins, paths):
    """Swap two paths' tails wherever that is cheaper. True if anything moved.

    THE MOVE NEITHER EXTRACTION NOR RE-DERIVATION CAN MAKE. Taking the
    best path first means that when two alleles overlap, path 1 takes
    the better locus at EVERY slot and ends up woven through both, with
    path 2 left the remainder. Re-deriving a path against the free spots
    cannot undo that -- what it holds really is the cheapest chain
    through what it is allowed to see.

    Two paths that cross are repaired by exchanging everything after the
    crossing, which is the classic 2-opt move and is valid here for
    free: both tails are still slot-increasing, so the result is still
    two legal paths. Only the two junction steps change, so a cut costs
    four lookups to evaluate.

    Measured on two synthetic alleles 1.0 um apart: mean purity 0.69
    without this, 0.98 with it.
    """
    b = np.asarray(bins, dtype=np.int64)
    moved = False
    for x in range(len(paths)):
        for y in range(x + 1, len(paths)):
            pa = np.asarray(paths[x], dtype=np.int64)
            pb = np.asarray(paths[y], dtype=np.int64)
            if len(pa) < 2 or len(pb) < 2:
                continue
            ah, at = pa[:-1], pa[1:]        # head end / tail start, per cut
            bh, bt = pb[:-1], pb[1:]
            now = w[ah, at][:, None] + w[bh, bt][None, :]
            then = w[np.ix_(ah, bt)] + w[np.ix_(bh, at)].T
            # both tails must still start after both heads end, or the
            # swap would make a path that walks backwards
            ok = ((b[bt][None, :] > b[ah][:, None])
                  & (b[at][:, None] > b[bh][None, :]))
            delta = np.where(ok, then - now, np.inf)
            flat = int(np.argmin(delta))
            if delta.flat[flat] < -1e-9:
                i, j = np.unravel_index(flat, delta.shape)
                i, j = int(i) + 1, int(j) + 1
                paths[x] = pa[:i].tolist() + pb[j:].tolist()
                paths[y] = pb[:j].tolist() + pa[i:].tolist()
                moved = True
    return moved


Paths = collections.namedtuple(
    'Paths', 'k labels weights cost per_k reason')
Paths.__doc__ = """One cell's chains and what they were worth.

k         how many chains the cost kept
labels    (n,) which chain each spot joined, UNASSIGNED for none
weights   per chain, in nats -- how much better than calling its spots
          clutter. Negative is good; this is the evidence FOR that
          allele and it is per-allele rather than pooled.
cost      the total, the thing minimized
per_k     [(k, total + k x prior), ...] so a close call is visible
reason    in words, including which calibration and which separation
"""


def choose_paths(pos, bins, sep, k_max=None, cal=None, volume_um3=None,
                 prior_nats=None, min_loci=None, iters=4):
    """How many chains, decided by whether another one pays for itself.

    EVERY EXTRA PATH LOWERS THE COST -- a path's weight is <= 0 by
    construction, so the sweep over k is monotone and a bare "pick the
    minimum" would always pick k_max. That is the same degeneracy the
    old loss had (cost(k) never rose, measured on 175 cells), and it is
    not a defect of this loss either: it is what a likelihood does when
    the model gains a component. The prior is what answers it, and the
    cap is what bounds it.
    """
    cal = cal or physics.Calibration()
    prior = float(DEFAULT_PATH_PRIOR_NATS if prior_nats is None else prior_nats)
    kmax = int(DEFAULT_K_MAX if k_max is None else k_max)
    p = np.asarray(pos, dtype=np.float64)
    if not len(p):
        return Paths(0, np.zeros(0, dtype=np.int64), [], 0.0, [],
                     'no spots')
    costs = step_costs(p, bins, sep, cal)
    best, table = None, []
    for k in range(1, kmax + 1):
        labels, weights = find_paths(p, bins, k, sep, cal, volume_um3,
                                     iters=iters, min_loci=min_loci,
                                     costs=costs)
        total = float(sum(weights))
        scored = total + prior * len(weights)
        table.append((len(weights), round(scored, 3)))
        if best is None or scored < best[0] - 1e-9:
            best = (scored, labels, weights, total)
        if len(weights) < k:
            break                     # the cell cannot support another
    k = len(best[2])
    return Paths(k, best[1], best[2], best[3], table,
                 'chain loss over %d spot(s): %s, %s, clutter %.2f nats/spot '
                 '(%.0f um^3), prior %.2f nats/path, k capped at %d'
                 % (len(p), cal.source, sep.source,
                    clutter_nats(len(p), volume_um3),
                    float(CLUTTER_VOLUME_UM3 if volume_um3 is None
                          else volume_um3), prior, kmax))


def path_rows(labels, bins, rows, k, n_slots):
    """(k, n_slots) the caller's row id each chain takes in each slot."""
    out = np.full((int(k), int(n_slots)), NO_SPOT, dtype=np.int64)
    b = np.asarray(bins, dtype=np.int64)
    r = np.asarray(rows, dtype=np.int64)
    for i, j in enumerate(np.asarray(labels, dtype=np.int64)):
        if j >= 0 and 0 <= b[i] < int(n_slots):
            out[j, b[i]] = int(r[i])
    return out
