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

THE AXIAL GROUP IS THE POINT. A hot camera pixel is bright in EVERY
plane of the stack; a real emitter is bright in three to five. That is
what `col_frac_above_half` and `col_n_runs` measure, and on the only
labelled set that exists they were the strongest single features by a
distance -- which is also why the full column is carried alongside the
core rather than thrown away.
"""
import numpy as np

# The order is the contract: a saved model stores this list, and
# refuses to score a vector built by a different version of this file.
NAMES = (
    # contrast
    'peak', 'log_peak', 'core3', 'annulus_med', 'core_over_annulus',
    # lateral shape
    'sigma_xy', 'eccentricity', 'centroid_offset', 'ring2', 'ring4',
    # axial shape, from the FULL column
    'sigma_z', 'z_fwhm', 'col_frac_above_half', 'col_n_runs',
    'col_skew', 'z_from_edge', 'aspect_z_over_xy',
    # context
    'border_frac', 'engine_p', 'fit_ok', 'gate_pass',
)


def _runs_above(v):
    """(count, longest) of the runs where v is True."""
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


def one(core, col, border_frac=0.0, engine_p=np.nan,
        fit_ok=np.nan, gate_pass=np.nan):
    """One box -> one vector, in NAMES order.

    `core` and `col` are already background-subtracted and in sigma
    units (dataset.boxes). `engine_p`, `fit_ok`, `gate_pass` are what the
    classical engine said about this candidate -- carried so the model
    can be compared against it and, if it helps, use it.
    """
    core = np.asarray(core, float)
    col = np.asarray(col, float)
    ny, nx, nz = core.shape
    cy, cx, cz = ny // 2, nx // 2, nz // 2
    f = {}

    pk = float(np.nanmax(core)) if core.size else 0.0
    f['peak'] = pk
    f['log_peak'] = float(np.log1p(max(pk, 0.0)))

    pl = core[:, :, cz]
    f['core3'] = float(np.nansum(pl[max(cy - 1, 0):cy + 2,
                                    max(cx - 1, 0):cx + 2]))
    ann = pl.astype(float).copy()
    ann[max(cy - 2, 0):cy + 3, max(cx - 2, 0):cx + 3] = np.nan
    f['annulus_med'] = (float(np.nanmedian(ann))
                        if np.isfinite(ann).any() else 0.0)
    f['core_over_annulus'] = f['core3'] / (abs(f['annulus_med']) + 1.0)

    P = np.clip(np.nan_to_num(pl), 0, None)
    s = P.sum() + 1e-9
    yy, xx = np.mgrid[0:ny, 0:nx]
    my = float((P * yy).sum() / s)
    mx = float((P * xx).sum() / s)
    vy = float((P * (yy - my) ** 2).sum() / s)
    vx = float((P * (xx - mx) ** 2).sum() / s)
    f['sigma_xy'] = float(np.sqrt(max(vy, 0.0) + max(vx, 0.0)))
    f['eccentricity'] = abs(vy - vx) / (vy + vx + 1e-9)
    f['centroid_offset'] = float(np.hypot(my - cy, mx - cx))
    # How fast it falls off: a real emitter is ~1.3 px sigma, a hot pixel
    # is gone by r=2, a blob of background is still there at r=4.
    rr = np.hypot(yy - cy, xx - cx)
    c0 = max(pk, 1e-9)
    for rad, name in ((2.0, 'ring2'), (4.0, 'ring4')):
        m = (rr >= rad - 0.5) & (rr < rad + 0.5)
        f[name] = float(np.nanmean(pl[m]) / c0) if m.any() else 0.0

    c = np.clip(np.nan_to_num(col), 0, None)
    cmax = float(c.max()) if c.size else 0.0
    above = c > (cmax / 2.0 if cmax > 0 else 1.0)
    n_runs, longest = _runs_above(above)
    f['col_frac_above_half'] = float(above.mean()) if c.size else 0.0
    f['col_n_runs'] = float(n_runs)
    f['z_fwhm'] = float(longest)
    zs = np.arange(c.size, dtype=float)
    ss = c.sum() + 1e-9
    mz = float((c * zs).sum() / ss)
    var_z = float((c * (zs - mz) ** 2).sum() / ss)
    f['sigma_z'] = float(np.sqrt(max(var_z, 0.0)))
    sd = np.sqrt(max(var_z, 1e-9))
    f['col_skew'] = float((c * ((zs - mz) / sd) ** 3).sum() / ss)
    zi = int(np.clip(np.argmax(c) if c.size else 0, 0, max(c.size - 1, 0)))
    f['z_from_edge'] = float(min(zi, max(c.size - 1 - zi, 0)))
    f['aspect_z_over_xy'] = f['sigma_z'] / (f['sigma_xy'] + 1e-9)

    f['border_frac'] = float(border_frac)
    f['engine_p'] = float(engine_p) if np.isfinite(engine_p) else 0.0
    f['fit_ok'] = float(fit_ok) if np.isfinite(fit_ok) else 0.0
    f['gate_pass'] = float(gate_pass) if np.isfinite(gate_pass) else 0.0

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
                     border_frac=r.get('border_frac', 0.0),
                     engine_p=r.get('engine_p', np.nan),
                     fit_ok=r.get('fit_ok', np.nan),
                     gate_pass=r.get('gate_pass', np.nan))
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
