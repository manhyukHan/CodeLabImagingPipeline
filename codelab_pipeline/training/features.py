"""
A box -> a fixed vector, in units the classifier can use.

WHY FEATURES AND NOT THE RAW BOX. The core is 15 x 15 x 25 = 5,625
voxels and the labelled set is around a thousand spots. A model that
reads the box directly has to learn what a point emitter looks like from
fewer examples than it has inputs; a model that reads thirty numbers
computed from the box starts with that knowledge built in. The raw box
is still available -- classify.py's conv variant takes it -- so the two
can be compared on the same split rather than argued about.

NO PSF-DICTIONARY FEATURES HERE, deliberately. Correlating the box
against a stored template would couple this model to a calibration:
every PSF recalibration would silently move the classifier's inputs and
its threshold with them, and the two artifacts would have to be versioned
together forever. They are kept independent so either can be rebuilt
alone. The template's job is DETECTION (psf_bank's matched filter finds
the spots, including several in one box); this model's job is judging one
box that already has a centre.

INPUTS ARE IN SIGMA ABOVE BACKGROUND. dataset.boxes divides by the
crop's own mode-and-left-sigma, so a feature means the same thing in a
dim cell and a bright one. Nothing here re-estimates the background.

WHAT ACTUALLY CARRIES THE MODEL, measured on 2,057 human-gated boxes by
shuffling each feature and watching held-out ROC fall -- LEAVE ONE HYBE
OUT, because what matters is what it leans on when it meets a round it
has never seen. Baseline ROC 0.943.

    group                 shuffled together
    contrast                    +0.277
    axial (full column)         +0.170
    lateral shape               +0.076
    context (border_frac)       -0.001

    feature              alone   shuffled alone
    log_peak             0.653       +0.104
    annulus_med          0.563       +0.082
    core3                0.616       +0.077
    core_over_annulus    0.816       +0.076
    ring4                0.652       +0.054
    sigma_z              0.649       +0.046
    col_n_runs           0.834       +0.035
    ...
    border_frac, z_from_edge                                     ~0

INERT IS A STATEMENT ABOUT THE LINEAR HEAD, and the mlp is the better
model (held-out ROC 0.961 against 0.943). Shuffled against the mlp, four
of the features that do nothing linearly come alive:

    feature            linear      mlp
    z_fwhm            -0.0014   +0.0233
    centroid_offset   +0.0017   +0.0124
    col_skew          -0.0001   +0.0037
    eccentricity      +0.0018   +0.0036
    border_frac       +0.0000   +0.0022
    z_from_edge       -0.0004   -0.0006

centroid_offset is the clearest case and the reason it was worth keeping:
the box is centred on a local maximum BY CONSTRUCTION, so a clean single
emitter has an offset near zero and anything else in the window drags the
intensity centroid off it. It is a merged-spot detector, and only a
non-linear head can use it that way.

Dropping the six costs the mlp 0.9613 -> 0.9571. border_frac and
z_from_edge are the two with no case in either head; they stay because
border_frac fires on 20% of boxes here (10% are more than a quarter
padding) and describes a real condition even though it does not predict
this label, and because dropping features on one experiment's labels is
how a feature list gets fitted to a bundle.

READ THE TWO COLUMNS AGAINST EACH OTHER. `col_n_runs` knows the most by
itself (ROC 0.834 alone -- a hot pixel is bright in every one of ~105
planes and an emitter in three to five) and contributes little uniquely,
because the rest of the axial group covers for it. `log_peak` knows
least of the leaders alone and contributes most, because nothing else
says how bright. Single-feature AUC ranks what a feature KNOWS;
permutation ranks what it ALONE brings, and correlated features share
credit -- which is why the group rows are the ones to trust.

The five inert features are kept anyway. Dropping them would be fitting
the feature list to one experiment's labels, and border_frac earns its
place on a bundle whose crops clip more spots than this one does.

THE FULL COLUMN IS WORTH ITS 105 NUMBERS: the axial group is the second
largest contributor and it is the only one that can see a hot pixel for
what it is.
"""
import numpy as np

# The order is the contract: a saved model stores this list, and
# refuses to score a vector built by a different version of this file.
# NAMES SAY WHERE THEY LOOK AND WHAT THEY COUNT, at whatever length that
# takes. The first version used `col_*` for a line one voxel wide through
# the whole stack and `run` for a stretch of adjacent planes above half
# maximum, and both had to be explained every time somebody read them --
# `column` was read as the box's 25 planes, `run` as anything at all.
# Nobody types these; a name that needs a footnote is a worse trade than
# a name that is long.
#
#   box_*    the 15 x 15 x 25 window around the candidate
#   stack_*  the 1 x 1 x 105 line through its centre, WHOLE depth
NAMES = (
    # contrast
    'box_peak_log_sigma', 'box_centre_sum_sigma',
    'box_surround_median_sigma', 'box_centre_over_surround',
    # lateral shape
    'box_lateral_spread_px', 'box_lateral_asymmetry',
    'box_centroid_offset_px', 'box_radial_at_2px', 'box_radial_at_4px',
    # axial shape, read from the whole stack
    'stack_axial_spread_planes', 'stack_longest_bright_stretch_planes',
    'stack_frac_above_half_max', 'stack_n_bright_stretches',
    'stack_axial_skew', 'axial_over_lateral_spread',
    # context
    'box_frac_padded',
)

# NOT A FEATURE: how close the brightest plane sits to the end of the
# stack. localization/edge_gate.py computes it, and computes it AFTER
# everything -- after pass/fail, after the matched filter has placed the
# spot to sub-voxel precision -- because it gates a finished answer
# rather than informing one.
#
# Given to the classifier it becomes "spots near the end of a stack are
# less likely to be real", which is false. The reviewer refuses almost
# everything down there (3.4% confirmed within 11 planes of an end,
# against 57.5% beyond 20), and a model that learns that refusal will
# apply it to the real emitters too.
#
# It has nothing to do with z_status either. That field says whether a
# spot's Z has been FITTED -- accepted / rejected / not_fit -- and is
# where THIS MODEL's verdict is stored. A spot the edge gate denies has
# a perfectly good fit; what it lacks is stack either side of it.

# NOT HERE, AND NOT BY ACCIDENT: engine_p, fit_ok, gate_pass.
#
# They were declared, never populated -- labels() returns coordinates,
# not the engine's verdict -- so they sat at a constant zero and the
# model was reading pixels alone. Removing them rather than wiring them
# up, because gate_pass would smuggle the ROUND in through the back
# door: the gate passes 0.9% of Hyb_101's candidates and 28.2% of
# Hyb_107's, a factor of forty, so a model given it can score well by
# recognising which hybe it is looking at. That is the one thing this
# model must not learn, and holding out a whole hybe is the only test
# that would catch it (ROC 0.943 +- 0.033 across eleven, against 0.958
# on a cell-level split -- close, because there is nothing to catch).
#
# Nothing identifying the hybe, FOV, cell or crop is a feature either.
# The box and the full z column through it are the whole input.


def names_sha():
    """Short hash of NAMES in order -- stamped on every stored vector, so
    a verdict featurised under one feature set is never fed to a model
    trained on another."""
    import hashlib
    return hashlib.sha256(','.join(NAMES).encode()).hexdigest()[:8]


def _bright_stretches(v):
    """(count, longest) of the maximal stretches where v is True.

    ADJACENT INDICES, nothing else. Walking planes 0..104, one plane
    dipping below the threshold ends a stretch and the next plane above
    it starts a new one -- no smoothing, no gap tolerance. The
    strictness is the signal: a confirmed spot has a median of ONE
    stretch and 69% have exactly one, while a rejected candidate has a
    median of six and only 18% have one.
    """
    runs, cur = [], 0
    for t in v:
        if t:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return len(runs), (max(runs) if runs else 0)


def one(box, stack_line, frac_padded=0.0):
    """One box -> one vector, in NAMES order.

    `core` and `col` are already background-subtracted and in sigma
    units (dataset.boxes).
    """
    core = np.asarray(box, float)
    col = np.asarray(stack_line, float)
    ny, nx, nz = core.shape
    cy, cx, cz = ny // 2, nx // 2, nz // 2
    f = {}

    # `pk` is used below as the scale for the radial profile; only its
    # LOGARITHM is a feature. The raw maximum was one too, until its rank
    # correlation with the log came back at exactly 1.000000 -- one is a
    # monotone function of the other, so they cannot differ on any
    # ranking measure, and removing it moved held-out ROC by +0.0001
    # (linear) and +0.0008 (mlp).
    pk = float(np.nanmax(core)) if core.size else 0.0
    f['box_peak_log_sigma'] = float(np.log1p(max(pk, 0.0)))

    pl = core[:, :, cz]
    f['box_centre_sum_sigma'] = float(np.nansum(pl[max(cy - 1, 0):cy + 2,
                                    max(cx - 1, 0):cx + 2]))
    ann = pl.astype(float).copy()
    ann[max(cy - 2, 0):cy + 3, max(cx - 2, 0):cx + 3] = np.nan
    f['box_surround_median_sigma'] = (float(np.nanmedian(ann))
                        if np.isfinite(ann).any() else 0.0)
    f['box_centre_over_surround'] = f['box_centre_sum_sigma'] / (abs(f['box_surround_median_sigma']) + 1.0)

    P = np.clip(np.nan_to_num(pl), 0, None)
    s = P.sum() + 1e-9
    yy, xx = np.mgrid[0:ny, 0:nx]
    my = float((P * yy).sum() / s)
    mx = float((P * xx).sum() / s)
    vy = float((P * (yy - my) ** 2).sum() / s)
    vx = float((P * (xx - mx) ** 2).sum() / s)
    f['box_lateral_spread_px'] = float(np.sqrt(max(vy, 0.0) + max(vx, 0.0)))
    f['box_lateral_asymmetry'] = abs(vy - vx) / (vy + vx + 1e-9)
    f['box_centroid_offset_px'] = float(np.hypot(my - cy, mx - cx))
    # How fast it falls off: a real emitter is ~1.3 px sigma, a hot pixel
    # is gone by r=2, a blob of background is still there at r=4.
    rr = np.hypot(yy - cy, xx - cx)
    c0 = max(pk, 1e-9)
    for rad, name in ((2.0, 'box_radial_at_2px'),
                  (4.0, 'box_radial_at_4px')):
        m = (rr >= rad - 0.5) & (rr < rad + 0.5)
        f[name] = float(np.nanmean(pl[m]) / c0) if m.any() else 0.0

    c = np.clip(np.nan_to_num(col), 0, None)
    cmax = float(c.max()) if c.size else 0.0
    above = c > (cmax / 2.0 if cmax > 0 else 1.0)
    n_stretches, longest = _bright_stretches(above)
    f['stack_frac_above_half_max'] = float(above.mean()) if c.size else 0.0
    f['stack_n_bright_stretches'] = float(n_stretches)
    f['stack_longest_bright_stretch_planes'] = float(longest)
    zs = np.arange(c.size, dtype=float)
    ss = c.sum() + 1e-9
    mz = float((c * zs).sum() / ss)
    var_z = float((c * (zs - mz) ** 2).sum() / ss)
    f['stack_axial_spread_planes'] = float(np.sqrt(max(var_z, 0.0)))
    sd = np.sqrt(max(var_z, 1e-9))
    f['stack_axial_skew'] = float((c * ((zs - mz) / sd) ** 3).sum() / ss)
    f['axial_over_lateral_spread'] = f['stack_axial_spread_planes'] / (f['box_lateral_spread_px'] + 1e-9)

    f['box_frac_padded'] = float(frac_padded)

    v = np.array([f[n] for n in NAMES], dtype=np.float64)
    # A non-finite feature is a bug upstream, not a value to propagate
    # into a model that will then produce a non-finite p.
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


def many(cores, cols, rows=None):
    """(n, len(NAMES)) for a stack of boxes."""
    out = np.zeros((len(cores), len(NAMES)))
    for i in range(len(cores)):
        r = (rows[i] if rows is not None else {}) or {}
        out[i] = one(cores[i], cols[i],
                     frac_padded=r.get('border_frac', 0.0))
    return out


class Standardiser:
    """Mean/sd from the TRAINING rows only, applied to everything.

    Fitting it on all the data leaks the validation set's distribution
    into training -- a small leak next to splitting by spot, but free to
    avoid.
    """

    def __init__(self, mean=None, sd=None):
        self.mean = None if mean is None else np.asarray(mean, float)
        self.sd = None if sd is None else np.asarray(sd, float)

    def fit(self, X):
        X = np.asarray(X, float)
        self.mean = X.mean(0)
        sd = X.std(0)
        sd[sd < 1e-9] = 1.0          # a constant feature contributes nothing
        self.sd = sd
        return self

    def __call__(self, X):
        return (np.asarray(X, float) - self.mean) / self.sd

    def to_dict(self):
        return {'mean': self.mean.tolist(), 'sd': self.sd.tolist(),
                'names': list(NAMES)}

    @staticmethod
    def from_dict(d):
        if list(d.get('names', NAMES)) != list(NAMES):
            raise ValueError(
                'this model was trained on a different feature list; '
                'features.NAMES has changed since it was saved')
        return Standardiser(d['mean'], d['sd'])
