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
    ['y', 'x', 'z', 'amplitude', 'sigma_y', 'sigma_x', 'sigma_z', 'offset'])


class LocalizeEngine:
    """
    Contract: localize(stack, seed_yxz=None, n_max=1) returns a
    brightest-first list of LocalizedSpot (possibly empty, never None).

    stack     (height, width, depth) ndarray, y-major, NaN = masked.
    seed_yxz  optional (y, x, z) crop-local starting point; None means
              the engine finds its own candidates.
    n_max     maximum number of emitters to return.

    Engines must never raise on bad data -- "no spot here" is an empty
    list, matching the pipeline-wide "absence is not an error" rule.
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
        spots = [LocalizedSpot(y=float(r[2]), x=float(r[1]), z=float(r[3]),
                               amplitude=float(r[0]), sigma_y=float(r[5]),
                               sigma_x=float(r[4]), sigma_z=float(r[6]),
                               offset=float(r[7]))
                 for r in results if r is not None]
        spots.sort(key=lambda s: -s.amplitude)
        return spots[:n_max]


ENGINES = {GaussianLocalizeEngine.name: GaussianLocalizeEngine}


def make_engine(name, **params):
    """The one factory a config/UI names an engine through."""
    try:
        cls = ENGINES[name]
    except KeyError:
        raise ValueError(f'unknown localize engine {name!r} -- known: {sorted(ENGINES)}')
    return cls(**params)
