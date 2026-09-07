"""
THE localizer seam: LocalizeEngine(stack) -> [(y, x, z), ...].

Every 3D-coordinate locator in this pipeline is the same question --
given a (height, width, depth) crop and (optionally) a seed, where are
the emitters? -- so it is answered behind ONE interface, per explicit
decision, so the current Gaussian engine and a future machine-learning
engine are swappable without touching a single caller:

    engine = make_engine('gaussian', peak_bound=2.0, ...)
    spots = engine.localize(stack, seed_yxz=(y, x, z), n_max=3)

Coordinates are (y, x, z) -- the pipeline's rasterized order
(alignment/convention.py), crop-local; sub-pixel; z in planes. Callers
add their own crop origin back.

The Gaussian engine wraps localization.py's fit_gaussian_3d /
find_local_peaks_3d / fit_gaussian_mixture_3d -- it owns NO math of its
own (a second implementation of the fit is exactly the divergence this
codebase keeps having to hunt down). The existing localization workers
additionally use `raw_components`, the gaussian-specific aligned form
their sibling-gating logic needs; new engines only ever need
`localize`.
"""
from collections import namedtuple

import numpy as np

LocalizedSpot = namedtuple(
    'LocalizedSpot',
    ['y', 'x', 'z', 'p', 'amplitude', 'sigma_y', 'sigma_x', 'sigma_z', 'offset'])
LocalizedSpot.__doc__ = """One emitter: (y, x, z) crop-local sub-pixel, plus p.

`p` IS NOT A PROBABILITY. It is a per-engine quality scalar in (0, 1],
higher is better, comparable only WITHIN one engine and one
parameterisation. Its job is to rank candidates -- to order a review
queue and to cut an obviously-dead tail -- not to be believed as a
calibrated confidence. Each engine documents how it computes p; a
learned engine may return a real calibrated probability, in which case
say so there rather than assuming it here.

The remaining fields are Gaussian-shaped and may be NaN for an engine
that does not fit a Gaussian (a learned detector returns positions and
p, and has no sigma to report). Never require them.
"""


def _spot(y, x, z, p, amplitude=float('nan'), sigma_y=float('nan'),
          sigma_x=float('nan'), sigma_z=float('nan'), offset=float('nan')):
    """LocalizedSpot with the shape fields defaulted -- so an engine that
    has no Gaussian to report does not have to invent one."""
    return LocalizedSpot(y=float(y), x=float(x), z=float(z), p=float(p),
                         amplitude=float(amplitude), sigma_y=float(sigma_y),
                         sigma_x=float(sigma_x), sigma_z=float(sigma_z),
                         offset=float(offset))


class LocalizeEngine:
    """
    Contract: localize(stack, seed_yxz=None, n_max=1) returns a
    best-first list of LocalizedSpot (possibly empty, never None).

    stack     (height, width, depth) ndarray, y-major, NaN = masked.
    seed_yxz  optional (y, x, z) crop-local starting point; None means
              the engine finds its own candidates -- THE form a learned
              detector takes, and the form the training-data helper
              drives (codelab_pipeline/training/).
    n_max     maximum number of emitters to return.

    Engines must never raise on bad data -- "no spot here" is an empty
    list, matching the pipeline-wide "absence is not an error" rule.

    The unseeded call is the whole contract: stack -> [(y, x, z, p), ...].
    Anything an engine needs beyond the stack is constructor state, so
    that a Gaussian anchor-fit and a trained network are interchangeable
    at the call site without the caller knowing which it holds.
    """

    name = 'abstract'

    def localize(self, stack, seed_yxz=None, n_max=1):
        raise NotImplementedError


class GaussianLocalizeEngine(LocalizeEngine):
    """
    Bounded least-squares 3D Gaussian fitting (fit_gaussian_3d), with
    z-window-restricted multi-component seeding and mixture fitting for
    n_max > 1 -- the exact behavior the localization workers had inline
    before this seam existed. Fit/gate parameters are engine STATE
    (constructor), not per-call arguments: an engine instance IS one
    parameterization.
    """

    name = 'gaussian'

    def __init__(self, peak_bound=2.0, init_sigma_xy=1.25, init_sigma_z=2.5,
                 min_sigma=0.1, max_sigma=2.5, min_hb_ratio=1.2,
                 min_ah_ratio=0.25, max_uncert=2.0, min_sep=3.0,
                 component_threshold=0.3, z_window=15, symmetric_xy=False):
        self.fit_kwargs = dict(peak_bound=peak_bound, init_sigma_xy=init_sigma_xy,
                               init_sigma_z=init_sigma_z, min_sigma=min_sigma,
                               max_sigma=max_sigma, min_hb_ratio=min_hb_ratio,
                               min_ah_ratio=min_ah_ratio, max_uncert=max_uncert)
        self.min_sep = min_sep
        self.component_threshold = component_threshold
        self.z_window = z_window
        # single-emitter fits only -- the mixture keeps free XY (its
        # components exist precisely because the window is not one
        # clean symmetric emitter)
        self.symmetric_xy = symmetric_xy

    def raw_components(self, stack, seed_yxz, n_max=1):
        """
        (results, seeds) in the fit functions' own raw form: results[i]
        is (amp, x, y, z, sx, sy, sz, offset) or None, index-aligned
        with seeds (crop-local (x, y, z) tuples) when a mixture ran --
        the alignment refine_spot_z's sibling gates depend on. seeds is
        [] when a single fit ran.
        """
        from . import localization as L
        y0, x0, z0 = seed_yxz
        if n_max > 1:
            z0_idx = int(round(z0))
            zwin_min = max(0, z0_idx - self.z_window)
            zwin_max = min(stack.shape[2], z0_idx + self.z_window + 1)
            seeds_local = L.find_local_peaks_3d(stack[:, :, zwin_min:zwin_max],
                                                min_sep=self.min_sep,
                                                threshold_rel=self.component_threshold,
                                                max_peaks=n_max)
            seeds = [(sx, sy, sz + zwin_min) for (sx, sy, sz) in seeds_local]
        else:
            seeds = []
        if len(seeds) <= 1:
            results = [L.fit_gaussian_3d(stack, x0, y0, z0, symmetric_xy=self.symmetric_xy,
                                         **self.fit_kwargs)]
            seeds = []
        else:
            results = L.fit_gaussian_mixture_3d(stack, seeds, **self.fit_kwargs)
        return results, seeds

    def localize(self, stack, seed_yxz=None, n_max=1):
        if stack is None or stack.size == 0 or not np.isfinite(stack).any():
            return []
        if seed_yxz is None:
            iy, ix, iz = np.unravel_index(int(np.nanargmax(stack)), stack.shape)
            seed_yxz = (float(iy), float(ix), float(iz))
        results, _ = self.raw_components(stack, seed_yxz, n_max=n_max)
        # p is left at 1.0 here on purpose: this engine reports a fit, and
        # every fit it returns already passed fit_gaussian_3d's internal
        # gates (a rejection arrives as None, not as a low score). There is
        # no surviving quantity to rank the survivors by. AnchorFitEngine
        # fills p in with contrast; AnchorFitV2Engine computes a real one.
        spots = [_spot(y=r[2], x=r[1], z=r[3], p=1.0, amplitude=r[0],
                       sigma_y=r[5], sigma_x=r[4], sigma_z=r[6], offset=r[7])
                 for r in results if r is not None]
        spots.sort(key=lambda s: -s.amplitude)
        return spots[:n_max]


GENEROUS_ANCHOR = dict(min_distance=2, threshold_rel=0.12,
                       max_to_background=1.25, background_quantile=0.5)
"""Anchor settings for BUILDING A TRAINING SET, not for production.

Production anchoring is tuned so that what it returns is mostly real; a
training run wants the opposite, because a person is going to look at
every candidate anyway and a rejected one is a labelled hard negative --
the more informative half of the set. A candidate never proposed is a
label that can never be collected, so recall here is worth far more than
precision.

MEASURED on a synthetic field of three emitters at amplitude 9000/5000/
2500 over background 300: the production defaults (threshold_rel 0.5,
max_to_background 2.0) find TWO -- the 2500 spot sits under 0.5x the
brightest peak and is never proposed. These settings find all three.
Do not "clean up" a training run by tightening them.
"""


def anchor_candidates(stack, n_max=8, min_distance=3, threshold_rel=0.5,
                      absolute_threshold=0.0, background_quantile=0.5,
                      max_to_background=2.0):
    """THE anchor step, alone: (h, w, depth) stack -> [(y, x, z), ...].

    This is the auto half of a slot with exactly two occupants. The other
    is a person: clicking a spot on the MIP in the crop displayer answers
    the SAME question in the SAME coordinates, and enters the engine
    through the same door (`localize(stack, seed_yxz=...)`). That is why
    the human step does not break the stack -> list encapsulation --
    a person is an alternative anchor SOURCE at the very front, not a
    stage wedged between anchoring and fitting. Nothing downstream can
    tell which one produced a seed.

    Ports localize_cell_2d_worker's detection unchanged in spirit: peak
    picking on the MIP (NaN outside the cell mask is simply never a
    peak), then a per-column z from the (x, depth) profile. Brightest
    first. No fit here -- fitting is the caller's next step, and keeping
    them apart is what lets the same anchors feed a Gaussian fit, a PSF
    fit, or nothing at all.
    """
    import warnings
    from skimage.feature import peak_local_max
    if stack is None or stack.size == 0 or not np.isfinite(stack).any():
        return []
    # A cell crop is mostly mask: whole columns and planes ARE all-NaN by
    # construction, and nanmax says so once per column. Silenced here
    # rather than at the caller -- this is the expected shape of the
    # input, not a condition anyone can act on.
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mip = np.nanmax(stack, axis=2)
        bimg = np.nanmax(stack, axis=0)           # (width, depth)
        floor = np.nanquantile(mip, background_quantile)
        peak = np.nanmax(mip)
    if not np.isfinite(peak):
        return []
    cutoff = max(max_to_background * (floor if np.isfinite(floor) else 0.0),
                 absolute_threshold, threshold_rel * peak)
    flat = np.where(np.isfinite(mip), mip, -np.inf)
    yx = peak_local_max(flat, min_distance=min_distance, exclude_border=1,
                        threshold_abs=cutoff)
    if len(yx) == 0:
        return []
    bright = mip[yx[:, 0], yx[:, 1]]
    out = []
    for j in np.argsort(bright)[::-1][:n_max]:
        y, x = int(yx[j][0]), int(yx[j][1])
        col = bimg[x]
        if not np.isfinite(col).any():
            continue
        out.append((float(y), float(x), float(np.nanargmax(col))))
    return out


class AnchorFitEngine(LocalizeEngine):
    """Anchor on the MIP, then fit each anchor in 3D -- v1 Gaussian.

    The unseeded contract, made real: localize(stack) alone returns every
    candidate in the crop, which is what a training-data run and a
    learned detector both need. Seeded, it degenerates to one fit at the
    given point, which is what an interactive click needs. Same engine.

    p is CONTRAST ONLY -- amplitude / (amplitude + offset), in (0, 1).
    Deliberately coarse, and worth being explicit about: fit_gaussian_3d
    applies its own CI and peak/background gates INTERNALLY and reports a
    rejection by returning None, so the quantities a richer score would
    use are already spent and unavailable here. Use the v2 engine when p
    has to carry more than "is there contrast".
    """

    name = 'anchor-v1'

    def __init__(self, anchor=None, **fit_params):
        self.anchor = dict(anchor or {})
        self.fit = GaussianLocalizeEngine(**fit_params)

    def localize(self, stack, seed_yxz=None, n_max=1):
        seeds = ([seed_yxz] if seed_yxz is not None
                 else anchor_candidates(stack, n_max=n_max, **self.anchor))
        out = []
        for s in seeds:
            for sp in self.fit.localize(stack, seed_yxz=s, n_max=1):
                amp, off = sp.amplitude, sp.offset
                denom = amp + off
                p = float(amp / denom) if denom > 0 else 0.0
                out.append(sp._replace(p=max(min(p, 1.0), 1e-6)))
        out.sort(key=lambda s: -s.p)
        return out[:n_max]


class AnchorFitV2Engine(LocalizeEngine):
    """Anchor on the MIP, then fit each anchor with the PSF-aware v2 fit.

    The engine to prefer for producing training candidates, because it is
    the one that can say HOW GOOD each candidate is. p combines the three
    quantities the v2 gates already compute per fit -- occupancy, and the
    95% CI on position laterally and axially -- into one monotone score:

        p = occupancy * ref_xy/(ref_xy + ci_xy) * ref_z/(ref_z + ci_z)

    quartered when the fit sits on a position bound. Higher is better in
    every term. It is a RANKING, not a calibrated probability: no
    threshold on it means anything until it has been compared against
    human verdicts, which is precisely the dataset this feeds.

    Deliberately does NOT gate. A generous candidate list is the point --
    a rejected candidate a person can look at is a hard negative, and
    throwing it away here is throwing away the more informative half of
    the training set. Callers that want the production gate can still
    call tracing_v2.gate themselves.
    """

    name = 'anchor-v2'
    REF_CI_XY_NM = 100.0
    REF_CI_Z_NM = 300.0

    def __init__(self, anchor=None, params=None, fit_radius=8):
        self.anchor = dict(anchor or {})
        self.params = params
        self.fit_radius = int(fit_radius)

    def _params(self):
        from . import tracing_v2 as V2
        return self.params if self.params is not None else V2.V2Params()

    def localize(self, stack, seed_yxz=None, n_max=1):
        from . import tracing_v2 as V2
        if stack is None or stack.size == 0 or not np.isfinite(stack).any():
            return []
        p = self._params()
        seeds = ([seed_yxz] if seed_yxz is not None
                 else anchor_candidates(stack, n_max=n_max, **self.anchor))
        r = self.fit_radius
        out = []
        for (sy, sx, sz) in seeds:
            y0, y1 = max(0, int(sy) - r), min(stack.shape[0], int(sy) + r + 1)
            x0, x1 = max(0, int(sx) - r), min(stack.shape[1], int(sx) + r + 1)
            cube = stack[y0:y1, x0:x1, :]
            if cube.size == 0 or not np.isfinite(cube).any():
                continue
            fit = V2.fit_readout(cube, float(sz), p)
            if fit is None:
                continue
            occ = V2.occupancy(cube, fit, p.voxel_um)
            ci_xy, ci_z = V2.uncertainty_nm(fit)
            score = (max(0.0, occ if np.isfinite(occ) else 0.0)
                     * self.REF_CI_XY_NM / (self.REF_CI_XY_NM + (ci_xy if np.isfinite(ci_xy) else 1e6))
                     * self.REF_CI_Z_NM / (self.REF_CI_Z_NM + (ci_z if np.isfinite(ci_z) else 1e6)))
            railed = getattr(fit, 'at_bound', None) or ()
            if isinstance(railed, str):
                railed = (railed,)
            if any(n in ('y', 'x', 'z') for n in railed):
                score *= 0.25
            out.append(_spot(y=fit.y + y0, x=fit.x + x0, z=fit.z,
                             p=max(min(score, 1.0), 1e-6),
                             amplitude=fit.amplitude, offset=fit.offset,
                             sigma_y=fit.sigma_y_um, sigma_x=fit.sigma_x_um,
                             sigma_z=fit.sigma_z_um))
        out.sort(key=lambda s: -s.p)
        return out[:n_max]


ENGINES = {
    GaussianLocalizeEngine.name: GaussianLocalizeEngine,
    AnchorFitEngine.name: AnchorFitEngine,
    AnchorFitV2Engine.name: AnchorFitV2Engine,
}


def make_engine(name, **params):
    """The one factory a config/UI names an engine through."""
    try:
        cls = ENGINES[name]
    except KeyError:
        raise ValueError(f'unknown localize engine {name!r} -- known: {sorted(ENGINES)}')
    return cls(**params)
