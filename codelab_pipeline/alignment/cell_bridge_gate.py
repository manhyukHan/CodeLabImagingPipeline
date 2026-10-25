# -*- coding: utf-8 -*-
"""Whether a store may have a per-cell cross-modal term at all.

The cell-level cross-modal layer sits between the per-FOV bridge and the
per-cell same-modality residual, and it is not universally applicable:
it works where a nucleus has ENDS in z and fails where the optics smear
it into a column. Measured on four stores, the same estimator recovers a
known 1 um displacement to 87-93% on one and errs by 1.2-1.6 um per cell
on another. A layer that cannot tell those apart would faithfully fit
noise on the second and apply it.

So the layer is gated, in two tiers.

TIER 1, `admission` -- free, from a volume already in hand. How much
lateral gradient energy survives `offset_um` from the plane where it
peaks. A bounded nucleus keeps little; a column keeps most:

    Lamin  0.02-0.10      MP58 0.44      HoxA 0.46      chr19 0.65-0.81

It is a PRE-FILTER, not the decision. Its threshold sits in an empty gap
between two stores and is not calibrated; and it is indirect, answering
"does this object have ends" rather than "can the estimator find a
shift". It is also invalid on brightfield, where a phase object is
faintest AT focus so the plane of maximum gradient is not the focal
plane at all -- hence `admission` refuses rather than guesses when told
the signal is a phase image.

TIER 2, `recovery` -- the decision. Impose a KNOWN displacement on the
store's own data and see how much of it comes back. Nothing is borrowed
from another store: the number that decides is measured here. Two
failure modes are checked for explicitly rather than trusted, because
both produced confident wrong answers during this layer's development:

    the FFT wraps     `register3d` correlates circularly, so once the
                      search bound approaches half the kept depth a
                      shift of +s and one of -(n-s) are one peak. A run
                      that searched +-28 planes of a 25-plane window
                      reported a clean, plausible, entirely false table.
    the sign flips    the convention was assumed once and was backwards,
                      which turned a correction into a doubling of the
                      error and read, in pictures, as "the fit makes it
                      worse".

Both are verified here, and a failure of either is a REFUSAL rather than
a number, because a gate that quietly mis-measures is worse than no gate.

Hibernation needs no mechanism of its own: a refused store simply gets
no per-cell entries, and the resolver's existing rule -- a layer that was
never computed contributes identity -- is exactly the desired behaviour.
"""
import numpy as np

#: fraction of peak lateral gradient still present at +-OFFSET_UM.
#: Provisional: fitted to the gap between two stores, not calibrated.
ADMISSION_MAX = 0.25
ADMISSION_OFFSET_UM = 5.0

#: Tier 2. Lamin recovered 87-93% with a per-cell error IQR of 0.09-0.20
#: um; chr19 errs by 1.2-1.6 um. These sit between those, and like
#: ADMISSION_MAX they rest on two stores -- but unlike it, what they are
#: compared against is measured on the store being judged.
MIN_RECOVERED_FRACTION = 0.70
MAX_ERROR_IQR_UM = 0.30
IMPOSE_UM = 1.0
MIN_CELLS = 12


class Verdict(object):
    """admitted / refused, and why -- never a bare boolean.

    A caller that only gets False cannot tell "this store's optics
    cannot support the layer" from "the test could not be run here",
    and those call for different responses.
    """

    __slots__ = ('ok', 'reason', 'stats')

    def __init__(self, ok, reason, **stats):
        self.ok = bool(ok)
        self.reason = reason
        self.stats = stats

    def __bool__(self):
        return self.ok

    __nonzero__ = __bool__

    def __repr__(self):
        s = '  '.join('%s=%s' % (k, _fmt(v))
                      for k, v in sorted(self.stats.items()))
        return '<%s: %s%s>' % ('ADMIT' if self.ok else 'REFUSE',
                               self.reason, ('   ' + s) if s else '')


def _fmt(v):
    return ('%.3f' % v) if isinstance(v, float) else str(v)


def lateral_gradient_profile(vol):
    """per-plane lateral gradient energy -- how much structure a plane has"""
    v = np.asarray(vol, dtype=float)
    return ((np.diff(v, axis=0) ** 2).mean(axis=(0, 1))
            + (np.diff(v, axis=1) ** 2).mean(axis=(0, 1)))


def admission(vol, voxel_z_um, offset_um=ADMISSION_OFFSET_UM,
              phase_contrast=False):
    """Tier 1. Verdict on whether this volume's object has ends in z.

    `phase_contrast=True` refuses outright. A brightfield phase object is
    faintest AT focus and shows opposite-signed contrast on either side,
    so "the plane of greatest lateral structure" is a side lobe and every
    number built on it means something other than it claims. Refusing is
    honest; the alternative is a figure that looks fine and is not.
    """
    if phase_contrast:
        return Verdict(False, 'phase-contrast: this statistic locates focus '
                              'as the plane of MAXIMUM structure, and a phase '
                              'object is faintest there')
    g = lateral_gradient_profile(vol)
    base, peak = np.percentile(g, 5), g.max()
    if not np.isfinite(peak) or peak <= base:
        return Verdict(False, 'no lateral structure anywhere in this volume')
    gn = (g - base) / (peak - base)
    k = int(np.argmax(gn))
    step = int(round(offset_um / float(voxel_z_um)))
    sides = [gn[k + s * step] for s in (-1, 1) if 0 <= k + s * step < len(gn)]
    if not sides:
        return Verdict(False, 'stack too shallow to sample +-%.1f um from '
                              'focus (%d planes)' % (offset_um, len(gn)),
                       planes=len(gn))
    score = float(np.mean(sides))
    ok = score < ADMISSION_MAX
    return Verdict(ok,
                   'contrast %.2f at +-%.0f um is %s %.2f'
                   % (score, offset_um, 'under' if ok else 'over',
                      ADMISSION_MAX),
                   contrast=score, focal_plane=k, sides=len(sides),
                   one_sided=(len(sides) == 1))


def transit_planes(vol, probe=6):
    """(n_leading, n_trailing) end planes that are not where they claim.

    A z-scan returns the stage to its home position, and the frame that
    catches that transit is written at one end of the stack: the LAST
    plane of a fiducial channel, the FIRST of a readout channel, in the
    stores seen here. It holds an image from the MIDDLE of the stack, so
    when two stacks are correlated the artefact sits at the same index
    in both and contributes a large term at exactly zero lag. Measured
    on the full Lamin store by another session, that pull read the
    cross-modal z at about a third of its value.

    This DETECTS rather than assumes. An end plane is a transit if it
    resembles a MIDDLE plane AS WELL AS this stack's interior
    planes resemble their own neighbours -- which is the signature and
    is self-validating, so a store that writes its transit somewhere
    else, or not at all, is handled without anyone editing a
    convention into this file. On a store whose transits have already
    been trimmed away it finds nothing, which is the point.

    Measured on Lamin: fiducial ch408 last plane correlates 0.98 with
    plane 22 against 0.93 with its neighbour; readout ch635 first plane
    0.99 with plane 20 against 0.84 with its neighbour.
    """
    v = np.asarray(vol, float)
    n = v.shape[2]
    if n < 3 * probe:
        return 0, 0

    def corr(a, b):
        a = a - a.mean()
        b = b - b.mean()
        return float((a * b).mean() / (a.std() * b.std() + 1e-12))

    # What adjacent planes normally look like in THIS stack's interior.
    # Without it, a noisy channel trips the test on noise alone: Lamin
    # DNA ch750 has an interior neighbour correlation of about 0.6, so
    # "some far plane matches better" was true for its first five
    # planes and the detector wanted to cut all of them. An end plane
    # has to be anomalous FOR THIS STACK, not merely imperfect.
    typical = np.median([corr(v[:, :, j], v[:, :, j + 1])
                         for j in range(probe, n - probe - 1)])

    def is_transit(k, neighbour, far):
        c_near = corr(v[:, :, k], v[:, :, neighbour])
        if c_near >= typical - 0.02:
            return False
        c_far = max(corr(v[:, :, k], v[:, :, j]) for j in far)
        # ABSOLUTE, not merely better than c_near, and to the SAME
        # tolerance used just above. The two conditions together say
        # the plane's resemblance has MOVED: away from its neighbour,
        # onto a distant plane, at full interior strength -- which is
        # what a frame holding a middle image looks like.
        #
        # Asking only that c_far beat c_near fired at GENUINE scan
        # ends, where defocus drives c_near down and anything
        # smoother clears it by 0.02: on the re-trimmed Lamin store
        # that cut planes from 28 of 39 FOVs with no transit left to
        # find. Asking for c_far >= typical outright was too strict
        # the other way and lost the real one: on Lamin_IF ch408
        # rebuilt at 41 planes the true transit matches plane 23 at
        # 0.9900 while that stack's interior adjacency is 0.9956.
        return c_far >= typical - 0.02

    # WHERE the match sits, not just how good it is. The transit
    # frame was taken with the stage back at its home z, and home is
    # mid-scan, so a transit resembles a MIDDLE plane. Measured on
    # the Lamin store: true transits matched at fractional depth 0.50
    # (p10 0.50, p90 0.57, 32 of 33), while the false positives that
    # survived the correlation test matched at 0.22 (p90 0.31) --
    # hard against the probe boundary, which is to say they matched
    # the nearest plane they were allowed to and nothing more.
    #
    # This is a SECOND, independent condition. It is deliberately
    # not a tighter threshold: the margin above already carries one
    # tuned number and a second would be fitting this experiment.
    far = range(max(probe, n // 3), min(n - probe, n - n // 3))
    lead = 0
    while lead < probe and is_transit(lead, lead + 1,
                                      [j for j in far if abs(j - lead) > probe]):
        lead += 1
    trail = 0
    while trail < probe and is_transit(n - 1 - trail, n - 2 - trail,
                                       [j for j in far
                                        if abs(j - (n - 1 - trail)) > probe]):
        trail += 1
    return lead, trail


#: the stage returns home ONCE, so the artefact is one frame. A detector
#: asking for more than this is mis-firing on a noisy channel rather
#: than finding more transits: on Lamin DNA ch750 it wanted five planes
#: of forty-one, and on one FOV's crop of the real fit pair it wanted
#: six. Capping costs nothing when the detector is right and bounds the
#: damage when it is not.
MAX_TRIM_PER_END = 2


def trim_transit(*vols, **kw):
    """the same volumes with every stack's transit planes removed.

    Trimmed TOGETHER and by the union, so the pair stays plane-aligned:
    dropping a plane from one stack and not the other would shift them
    against each other by exactly the amount being measured.

    Returns ((trimmed, ...), (lead, trail)) so a caller can record what
    was cut. Over-trimming is capped: see MAX_TRIM_PER_END.
    """
    probe = kw.get('probe', 6)
    cap = kw.get('cap', MAX_TRIM_PER_END)
    lead = trail = 0
    for v in vols:
        a, b = transit_planes(v, probe)
        lead, trail = max(lead, a), max(trail, b)
    lead, trail = min(lead, cap), min(trail, cap)
    n = min(v.shape[2] for v in vols)
    return tuple(v[:, :, lead:n - trail] for v in vols), (lead, trail)


def check_search_bound(dz_max_planes, n_planes):
    """REFUSE when an FFT correlation would wrap.

    Circular correlation makes +s and -(n-s) the same peak, so a search
    bound near half the depth cannot be read at all. This is a hard
    refusal and not a warning: the wrapped run that prompted it printed
    a table no one could have told was wrong from its face.
    """
    limit = n_planes // 2
    if dz_max_planes >= limit:
        return Verdict(False,
                       'search bound %d planes against %d kept would wrap '
                       '(needs < %d)' % (dz_max_planes, n_planes, limit),
                       dz_max=dz_max_planes, n_planes=n_planes)
    return Verdict(True, 'search bound %d planes clears the %d-plane wrap '
                         'limit' % (dz_max_planes, limit))


def shift_sign(register, vol, shift_planes, dz_max, dxy_max):
    """+1 or -1: which way `register` reports a known displacement.

    Correlating a volume against itself is useless as a noise floor and
    exactly right for reading off a convention. The convention was
    assumed once, was backwards, and inverted the meaning of every
    before/after picture built on it; it is never assumed again.
    """
    n = vol.shape[2] - abs(int(shift_planes))
    s = int(shift_planes)
    got = register(vol[:, :, :n], vol[:, :, s:s + n], dz_max, dxy_max)[2]
    if abs(abs(got) - abs(s)) > 0.5:
        return None, Verdict(False, 'a known %d-plane shift came back as '
                                    '%+.2f -- the registration is not '
                                    'measuring what it claims' % (s, got),
                             imposed=s, got=float(got))
    return (1 if got * s > 0 else -1), Verdict(
        True, 'a known %+d-plane shift returns %+.2f' % (s, got))


def recovery(fractions, errors_um, n_cells=None):
    """Tier 2 verdict, from the measurements a caller has already made.

    Deliberately takes NUMBERS, not volumes: the fitting and the judging
    stay separable, so the thresholds can be re-read against a stored
    table without re-running anything, and so this is testable without a
    store.
    """
    f = np.asarray(fractions, dtype=float)
    e = np.asarray(errors_um, dtype=float)
    f, e = f[np.isfinite(f)], e[np.isfinite(e)]
    n = int(n_cells if n_cells is not None else len(f))
    if n < MIN_CELLS or not len(f):
        return Verdict(False, 'only %d usable cells (needs %d) -- the test '
                              'did not run, which is not the same as the '
                              'store failing it' % (n, MIN_CELLS), cells=n)
    frac = float(np.median(f))
    iqr = float(np.percentile(e, 75) - np.percentile(e, 25))
    why = []
    if frac < MIN_RECOVERED_FRACTION:
        why.append('recovered %.0f%% of the imposed shift (needs %.0f%%)'
                   % (100 * frac, 100 * MIN_RECOVERED_FRACTION))
    if iqr > MAX_ERROR_IQR_UM:
        why.append('per-cell error IQR %.2f um (needs under %.2f)'
                   % (iqr, MAX_ERROR_IQR_UM))
    if why:
        return Verdict(False, '; '.join(why), recovered=frac, err_iqr=iqr,
                       cells=n)
    return Verdict(True, 'recovered %.0f%% with a per-cell error IQR of '
                         '%.2f um over %d cells' % (100 * frac, iqr, n),
                   recovered=frac, err_iqr=iqr, cells=n)


def decide(admission_verdict, recovery_verdict):
    """The layer runs only if both tiers say so.

    Tier 1 alone is not enough -- it is indirect and its threshold is
    uncalibrated -- and tier 2 alone would spend the expensive test on
    stores tier 1 can reject for free.
    """
    for name, v in (('admission', admission_verdict),
                    ('recovery', recovery_verdict)):
        if v is None:
            return Verdict(False, '%s was never run' % name)
        if not v.ok:
            return Verdict(False, '%s: %s' % (name, v.reason), **v.stats)
    return Verdict(True, 'admission and recovery both pass',
                   **dict(recovery_verdict.stats))


def _selftest():
    rng = np.random.default_rng(0)

    # a bounded object: structure dies away from focus -> admitted
    z = np.arange(60)
    env = np.exp(-((z - 30) ** 2) / (2 * 4.0 ** 2))
    bounded = (rng.random((40, 40, 60)) * env[None, None, :])
    v = admission(bounded, 0.25, offset_um=3.0)
    assert v.ok, v

    # a column: the same texture at every depth -> refused
    column = np.repeat(rng.random((40, 40, 1)), 60, axis=2)
    column = column + 0.01 * rng.random((40, 40, 60))
    v = admission(column, 0.25, offset_um=3.0)
    assert not v.ok, v

    # brightfield is refused by kind, before any number is computed
    v = admission(bounded, 0.25, phase_contrast=True)
    assert not v.ok and 'phase' in v.reason, v

    # a stack too shallow to reach the offset says so, rather than
    # silently scoring one side and presenting it as both
    v = admission(bounded[:, :, :8], 0.25, offset_um=5.0)
    assert not v.ok and 'shallow' in v.reason, v

    assert not check_search_bound(28, 25).ok
    assert not check_search_bound(13, 26).ok      # exactly at the limit
    assert check_search_bound(12, 26).ok

    good = recovery([0.90] * 20, list(np.linspace(-0.08, 0.08, 20)))
    assert good.ok, good
    assert not recovery([0.40] * 20, [0.0] * 20).ok
    assert not recovery([0.95] * 20, list(np.linspace(-1.0, 1.0, 20))).ok
    thin = recovery([0.95] * 5, [0.0] * 5)
    assert not thin.ok and 'did not run' in thin.reason, thin

    # a sign convention is read, never assumed
    def fake_register(a, b, dz_max, dxy_max, flip=False):
        return (0.0, 0.0, -4.0 if flip else 4.0, 0.9, False)
    s, v = shift_sign(fake_register, bounded, 4, 10, 5)
    assert v.ok and s == 1, (s, v)
    s, v = shift_sign(lambda *a: fake_register(*a, flip=True), bounded, 4,
                      10, 5)
    assert v.ok and s == -1, (s, v)
    s, v = shift_sign(lambda *a: (0, 0, 0.1, 0.9, False), bounded, 4, 10, 5)
    assert s is None and not v.ok, v

    assert decide(admission(bounded, 0.25, offset_um=3.0), good).ok
    assert not decide(admission(column, 0.25, offset_um=3.0), good).ok
    assert not decide(None, good).ok
    print('cell_bridge_gate: all checks pass')


if __name__ == '__main__':
    _selftest()
