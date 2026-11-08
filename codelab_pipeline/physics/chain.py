"""P(separation | genomic distance): the physics term of the path loss.

WHAT IT IS FOR. Bottom-up path optimization asks, of two loci, whether
they belong to the same chromatin fibre. The loss it had could not ask
that: it scored a spot by its distance to a cluster centroid and knew
NOTHING of genomic separation, so bin 1 and bin 90 were treated alike,
and the only thing keeping an allele together was a hand-set radius
(`bottom_up.DEFAULT_REACH_UM`) that declared anything beyond 1.0 um "not
this path's at any price". An allele larger than that radius had its
outer half structurally excluded.

THE MODEL. A pair's separation vector d, given their genomic separation
s in kb, under a two-component mixture:

    P(d | s) = (1 - e) N(d; 0, C_chain(s)) + e N(d; 0, C_other)

    C_chain(s) = diag(sp^2 + 2 Lxy^2,  sp^2 + 2 Lxy^2,  sp^2 + 2 Lz^2)
    C_other    = diag(Txy^2, Txy^2, Tz^2)
    sp         = sigma_poly(s) = A s^alpha

THE SECOND COMPONENT IS THE POINT. When d is large for its s the chain
term collapses and the cost saturates at -log(e N(d; 0, C_other))
instead of growing as |d|^2 / 2 sigma^2. That saturation is the whole of
the robustness: one wrongly assigned locus cannot dominate a path, and a
far locus costs a FIXED amount rather than being refused outright. When
enough loci pay it, two paths explain the cell more cheaply than one --
which is how sister chromatids come apart, and it is the job
`DEFAULT_REACH_UM` was doing badly.

MEASURED, NOT ASSUMED -- see MEASUREMENT.md beside this file for the
whole derivation, the data, and the two fits that failed first. In
brief: separation vectors from every stored trace of five ORCA panels,
genomic separation from each panel's own probe design (midpoint to
midpoint, never |delta readout|), a two-stage EM with closed-form M
steps, and the instrument removed using repeat rounds (s = 0).

THE DEFAULTS BELOW COME FROM THREE PANELS, chr19 EXCLUDED. On chr19 the
fitted sigma_poly,z / sigma_poly,xy is 1.56 where the other panels read
1.00-1.12, and chr19 is also the panel whose z precision is worst by far
(sigma_loc,z/xy = 4.6 against 2.3-2.7). The two move together, so the
likely reading is an incomplete z instrument model on that store rather
than an anisotropic polymer -- but it is not settled, so chr19 does not
set a default. HoxA is excluded too, for having 53 usable alleles.

    panel          alpha   A      Lxy    Lz     Txy    Tz     z/xy   e
    RSF1 30kb      0.222  0.0669  0.018  0.043  0.638  1.076  1.11  0.270
    MP58 Q 50kb    0.280  0.0539  0.031  0.076  0.737  1.257  1.03  0.347
    MP58 G 50kb    0.267  0.0584  0.033  0.087  0.685  1.177  1.00  0.320

A AND alpha TRADE OFF, so the default pair is not the median of each
column. They are anchored at s = 100 kb, where the three panels give
sigma_poly 0.186, 0.196 and 0.200 um; taking the median there with the
median alpha fixes A. Over 30-1000 kb the default stays within about
10% of all three.

sigma_loc IS PER STORE AND MUST BE MEASURED, not taken from here: the
three panels span 0.018-0.033 um laterally, a factor of 1.8, because it
is a property of the imaging and not of the polymer. `Calibration`
exists so a caller carries its own; the module default is a starting
point for a store that has not been measured, and a caller that uses it
should say so.
"""
import math

import numpy as np

#: sigma_poly(s) = SIGMA_POLY_A * s**SIGMA_POLY_ALPHA, s in kb, um.
#: Anchored at s = 100 kb across RSF1, MP58 Q and MP58 G (isolated
#: alleles); alpha is their median, A follows from the anchor.
SIGMA_POLY_A = 0.0572
SIGMA_POLY_ALPHA = 0.267

#: Localization spread per axis, um. A STARTING POINT ONLY -- measure it
#: per store from the repeat rounds (`calibration_from_repeats`).
SIGMA_LOC_XY = 0.031
SIGMA_LOC_Z = 0.076

#: The contaminant: a second allele, or a wrong pick. Stretched in z in
#: EVERY panel measured (Tz/Txy 1.69-2.47), which is the "one MIP
#: anchor, two alleles stacked in z" case as a number.
TAU_XY = 0.685
TAU_Z = 1.177

#: Prior weight on the contaminant, per PAIR. Per LOCUS the rate is
#: f = 1 - sqrt(1 - e), about 0.18 here, because a pair is clean only
#: when both of its loci are.
E_MIX = 0.32

#: Genomic separations outside the fitted range are extrapolation. The
#: fit covered roughly 25 kb to 4 Mb; below SMALL_S_KB the curve is
#: below the localization floor anyway and the chain term is dominated
#: by the instrument.
FIT_S_MIN_KB, FIT_S_MAX_KB = 25.0, 4000.0

_LOG2PI = math.log(2.0 * math.pi)


class Calibration:
    """One store's instrument and mixture, with the panel's chain law.

    Everything here is a number somebody measured; nothing is tuned to
    make an answer come out. `source` records where it came from so a
    result can say whether it used a store's own calibration or the
    module defaults.
    """

    def __init__(self, sigma_loc_xy=SIGMA_LOC_XY, sigma_loc_z=SIGMA_LOC_Z,
                 a=SIGMA_POLY_A, alpha=SIGMA_POLY_ALPHA,
                 tau_xy=TAU_XY, tau_z=TAU_Z, e_mix=E_MIX, source='defaults'):
        self.sigma_loc_xy = float(sigma_loc_xy)
        self.sigma_loc_z = float(sigma_loc_z)
        self.a = float(a)
        self.alpha = float(alpha)
        self.tau_xy = float(tau_xy)
        self.tau_z = float(tau_z)
        self.e_mix = float(e_mix)
        self.source = str(source)

    def __repr__(self):
        return ('Calibration(sigma_loc=(%.3f, %.3f), sigma_poly=%.4f*s^%.3f, '
                'tau=(%.3f, %.3f), e=%.3f, source=%r)'
                % (self.sigma_loc_xy, self.sigma_loc_z, self.a, self.alpha,
                   self.tau_xy, self.tau_z, self.e_mix, self.source))

    def sigma_poly(self, s_kb):
        """The chain's own spread per axis, um, at genomic separation s."""
        s = np.maximum(np.asarray(s_kb, float), 1e-6)
        return self.a * s ** self.alpha

    def chain_var(self, s_kb):
        """(lateral, axial) variance of the OBSERVED chain separation.

        The instrument rides on the polymer: two loci are each measured
        once, so each axis carries 2 sigma_loc^2 on top of the chain.
        """
        sp2 = self.sigma_poly(s_kb) ** 2
        return (sp2 + 2.0 * self.sigma_loc_xy ** 2,
                sp2 + 2.0 * self.sigma_loc_z ** 2)

    def other_var(self):
        return self.tau_xy ** 2, self.tau_z ** 2


def _log_gauss(d, vxy, vz):
    """log N(d; 0, diag(vxy, vxy, vz)) for d of shape (..., 3)."""
    d = np.asarray(d, float)
    q = (d[..., 0] ** 2 + d[..., 1] ** 2) / vxy + d[..., 2] ** 2 / vz
    return -0.5 * q - np.log(vxy) - 0.5 * np.log(vz) - 1.5 * _LOG2PI


def log_p_chain(d, s_kb, cal=None):
    """log P(d | s) under the chain component alone."""
    cal = cal or Calibration()
    vxy, vz = cal.chain_var(s_kb)
    return _log_gauss(d, vxy, vz)


def log_p_other(d, cal=None):
    """log P(d) under the contaminant alone -- no s, by measurement.

    rms/median fell with s in every panel, which is what an
    s-independent absolute-distance contaminant must look like, so this
    deliberately takes no genomic separation.
    """
    cal = cal or Calibration()
    vxy, vz = cal.other_var()
    return _log_gauss(d, vxy, vz)


def pair_cost(d, s_kb, cal=None):
    """-log P(d | s) under the mixture: the loss's per-pair term.

    This is the whole physics contribution. It grows like |d|^2 while
    the pair is plausibly one chain and then FLATTENS, so a far locus is
    expensive rather than impossible -- the behaviour `reach` could not
    express.
    """
    cal = cal or Calibration()
    lc = math.log(max(1.0 - cal.e_mix, 1e-12)) + log_p_chain(d, s_kb, cal)
    lo = math.log(max(cal.e_mix, 1e-12)) + log_p_other(d, cal)
    m = np.maximum(lc, lo)
    return -(m + np.log(np.exp(lc - m) + np.exp(lo - m)))


def p_same_chain(d, s_kb, cal=None):
    """Posterior that this pair is one chain -- the mixture's own weight.

    The quantity a path can be scored by and a reviewer can look at,
    rather than a cost in arbitrary units.
    """
    cal = cal or Calibration()
    lc = math.log(max(1.0 - cal.e_mix, 1e-12)) + log_p_chain(d, s_kb, cal)
    lo = math.log(max(cal.e_mix, 1e-12)) + log_p_other(d, cal)
    m = np.maximum(lc, lo)
    a, b = np.exp(lc - m), np.exp(lo - m)
    return a / (a + b)


def saturation_um(s_kb, cal=None, axis='xy'):
    """Where the cost stops growing: the two components cross.

    Returned so a caller can SEE the scale the loss actually works at
    instead of inferring it. Beyond this separation a pair is better
    explained by the contaminant, which is the honest replacement for
    `reach`: not a wall, but the point past which paying the fixed price
    is cheaper than stretching the chain. Note it MOVES WITH s -- that
    is the entire difference from a constant radius.

    INFINITY MEANS THE CHAIN ALWAYS WINS, and it is a real answer rather
    than an error: past roughly 500 kb here the chain is broad enough
    that no separation is better explained by the contaminant, so the
    pair carries little information about which path it belongs to. It
    is the same degeneracy that makes the mixture weight unidentifiable
    at large s, and a caller that leans on long-range pairs to split
    paths should know it is leaning on nothing.
    """
    cal = cal or Calibration()
    vxy, vz = cal.chain_var(s_kb)
    v_c = vxy if axis == 'xy' else vz
    v_o = (cal.tau_xy ** 2) if axis == 'xy' else (cal.tau_z ** 2)
    if v_o <= v_c:
        return float('inf')
    # one axis, equal mixture-weighted densities
    k = (math.log(max(cal.e_mix, 1e-12)) - math.log(max(1 - cal.e_mix, 1e-12))
         + 0.5 * math.log(v_o / v_c))
    val = 2.0 * k / (1.0 / v_c - 1.0 / v_o)
    # k <= 0 is no crossing at all, not a crossing at zero: the chain's
    # weighted density is above the contaminant's everywhere.
    return float(math.sqrt(val)) if val > 0 else float('inf')


def calibration_from_repeats(rep_vectors, iters=200, **kw):
    """sigma_loc for THIS store, from its repeat rounds.

    A Rep_ round re-images the same readout, so the separation of a
    repeat pair is two measurements of one point and has per-axis
    variance 2 sigma_loc^2 -- no polymer in it at all.

    THE SAME TWO-COMPONENT FIT THE CONSTANTS CAME FROM, not a moment.
    Repeats are contaminated too (a repeat that landed on the other
    allele), measured at rms/median 2.9-8.1, and the obvious estimators
    both fail in opposite directions:

      mean square   the tail dominates outright -- 0.36 um against a
                    true 0.02 um on the synthetic case in the tests
      1.4826 * MAD  biased HIGH by a knowable factor, because a
                    symmetric contaminant pushes the central 50%
                    outward: at 20% contamination the clean component
                    must supply 0.5/0.8 of its own mass, so the
                    deviation lands at 0.887 sigma instead of 0.674 and
                    the estimate comes out 1.31x too large. Observed
                    1.305x.

    So MAD only INITIALISES here, and a diagonal two-component EM --
    closed form, no optimizer -- separates the clean repeats from the
    mis-assigned ones. That is exactly stage 1 of the measurement in
    MEASUREMENT.md, so a store calibrated here is calibrated the way the
    defaults were.
    """
    v = np.asarray(rep_vectors, float).reshape(-1, 3)
    v = v[np.isfinite(v).all(axis=1)]
    if len(v) < 30:
        raise ValueError('need at least 30 repeat pairs, got %d' % len(v))

    def rob2(x):
        return (1.4826 * float(np.median(np.abs(x - np.median(x))))) ** 2

    sxy = max(0.5 * (rob2(v[:, 0]) + rob2(v[:, 1])), 1e-10)
    sz = max(rob2(v[:, 2]), 1e-10)
    txy = max(float(np.percentile(v[:, 0] ** 2 + v[:, 1] ** 2, 99)) / 2.0,
              4.0 * sxy)
    tz = max(float(np.percentile(v[:, 2] ** 2, 99)), 4.0 * sz)
    e = 0.15
    for _ in range(int(iters)):
        lc = math.log(max(1.0 - e, 1e-12)) + _log_gauss(v, sxy, sz)
        lo = math.log(max(e, 1e-12)) + _log_gauss(v, txy, tz)
        m = np.maximum(lc, lo)
        w = np.exp(lc - m) / (np.exp(lc - m) + np.exp(lo - m))
        sw = w.sum()
        u = 1.0 - w
        su = u.sum()
        n_sxy = (float((w * (v[:, 0] ** 2 + v[:, 1] ** 2)).sum() / (2.0 * sw))
                 if sw > 10 else sxy)
        n_sz = float((w * v[:, 2] ** 2).sum() / sw) if sw > 10 else sz
        if su > 10:
            txy = float((u * (v[:, 0] ** 2 + v[:, 1] ** 2)).sum() / (2.0 * su))
            tz = float((u * v[:, 2] ** 2).sum() / su)
        n_e = float(1.0 - w.mean())
        done = (abs(n_sxy - sxy) < 1e-12 and abs(n_sz - sz) < 1e-12
                and abs(n_e - e) < 1e-10)
        sxy, sz, e = max(n_sxy, 1e-12), max(n_sz, 1e-12), n_e
        if done:
            break
    # a repeat separation is a DIFFERENCE of two measurements
    return Calibration(sigma_loc_xy=math.sqrt(sxy / 2.0),
                       sigma_loc_z=math.sqrt(sz / 2.0),
                       source='repeats (n=%d, mis-assigned %.0f%%)'
                              % (len(v), 100.0 * e), **kw)
