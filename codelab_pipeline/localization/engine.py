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
    ['y', 'x', 'z', 'p', 'amplitude', 'sigma_y', 'sigma_x', 'sigma_z',
     'offset', 'p_exist'],
    defaults=(float('nan'),))
LocalizedSpot.__doc__ = """One emitter: (y, x, z) crop-local sub-pixel, plus p.

`p` IS NOT A PROBABILITY. It is a per-engine quality scalar in (0, 1],
higher is better, comparable only WITHIN one engine and one
parameterisation. Its job is to rank candidates -- to order a review
queue and to cut an obviously-dead tail -- not to be believed as a
calibrated confidence. Each engine documents how it computes p.

`p_exist` IS ONE, and it is a SEPARATE FIELD for exactly that reason. It
is the probability that a spot is here at all, from a classifier trained
on human verdicts with a proper scoring rule and Platt-scaled on
held-out cells -- so it is comparable across engines, across bundles and
against a threshold a person picks off a histogram. NaN means this
engine does not produce one.

PUTTING IT IN `p` WOULD HAVE BEEN THE CHEAP MOVE AND IT IS THE WRONG
ONE. Four engines already write four different meanings into `p` --
a constant 1.0, a contrast, an occupancy-times-CI product, an affine
NCC -- and they are already stored under one column name in the review
bundle and the verdict log, where nothing downstream can tell them
apart. A fifth meaning in the same slot would make a posterior p-gate
LOOK applicable to v1 and v2, where the number it reads is not a
probability and the threshold a person chose on v3 means nothing. With a
separate field the gate is simply unavailable there: NaN is not a
threshold you can set.

The remaining fields are Gaussian-shaped and may be NaN for an engine
that does not fit a Gaussian (a learned detector returns positions and
p, and has no sigma to report). Never require them.
"""


def _spot(y, x, z, p, amplitude=float('nan'), sigma_y=float('nan'),
          sigma_x=float('nan'), sigma_z=float('nan'), offset=float('nan'),
          p_exist=float('nan')):
    """LocalizedSpot with the shape fields defaulted -- so an engine that
    has no Gaussian to report does not have to invent one."""
    return LocalizedSpot(y=float(y), x=float(x), z=float(z), p=float(p),
                         amplitude=float(amplitude), sigma_y=float(sigma_y),
                         sigma_x=float(sigma_x), sigma_z=float(sigma_z),
                         offset=float(offset), p_exist=float(p_exist))


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
        return spots if n_max is None else spots[:int(n_max)]


def background_mode(img):
    """(mode, sigma) of the background in a cell projection.

    On a cell crop the background IS the mode: spots are a thin bright
    tail over a broad flat floor, so the most common intensity bin is the
    floor. The median is not -- MEASURED over 40 real (fov, hybe, cell)
    crops from the MAZ store, `2.0 * median` sits at 947 counts where
    `mode + 1 sigma` sits at 545, and the median-based threshold finds
    NOTHING on 39 of 40 cells.

    Binning to 100 counts (`round(-2)`) is what makes the mode findable at
    all on 16-bit data -- unbinned, every value is nearly unique and the
    argmax of the histogram is noise.

    sigma comes from the BELOW-mode half only. For a background that is
    locally normal, E[(x - mode)^2 | x <= mode] = sigma^2, and taking only
    that half means the spots -- which are all on the bright side -- cannot
    inflate the very width used to decide what counts as a spot.
    """
    f = img[np.isfinite(img)]
    if f.size == 0:
        return float('nan'), float('nan')
    v, c = np.unique(f.round(-2), return_counts=True)
    mode = float(v[c.argmax()])
    below = f[f <= mode]
    sigma = float(np.sqrt(np.mean((below - mode) ** 2))) if below.size else float('nan')
    return mode, sigma


GENEROUS_ANCHOR = dict(min_distance=2, mode_k=1.0, threshold_rel=0.0,
                       max_to_background=0.0)
"""Anchor settings for BUILDING A TRAINING SET, not for production.

Production anchoring is tuned so that what it returns is mostly real. A
training run wants the opposite: a person looks at every candidate
anyway, and a rejected one is a labelled hard negative -- the more
informative half of the set. A candidate never proposed is a label that
can never be collected, so recall here is worth more than precision.

MEASURED, 40 real cell crops, MAZ store, candidates per cell:

    production defaults      median  0.0   39/40 cells yield NOTHING
    mode + 3.0 sigma         median  0.0   26/40 empty
    mode + 2.0 sigma         median  1.5   17/40 empty
    mode + 1.5 sigma         median  6.5   10/40 empty
    mode + 1.0 sigma         median 21.0    3/40 empty   <- this

THRESHOLD ALONE CANNOT CONTROL REVIEW EFFORT, and that is why there is a
cap elsewhere rather than a higher k here. At k=1.0 the median cell gives
21 candidates but the worst gives 156; the spread across cells is larger
than the spread across k. So this stays low for recall, `n_max` bounds
the fitting, and the engine keeps only its best `keep_top` by p for a
person to look at. Raising k to make the list shorter throws away dim
real spots -- exactly the ones a learned detector is being built to find.
"""


def anchor_candidates(stack, n_max=None, min_distance=3, mode_k=None,
                      threshold_rel=0.5, absolute_threshold=0.0,
                      background_quantile=0.5, max_to_background=2.0):
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
    n_max = None if n_max is None else int(n_max)
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
    # Every enabled policy is a floor, and the highest wins. A term is off
    # at 0, so a caller picks its policy by which numbers it passes rather
    # than by a mode flag -- GENEROUS_ANCHOR zeroes the two production
    # terms and sets mode_k, production leaves mode_k None.
    cutoff = max(max_to_background * (floor if np.isfinite(floor) else 0.0),
                 absolute_threshold, threshold_rel * peak)
    if mode_k is not None:
        mode, sigma = background_mode(mip)
        if np.isfinite(mode) and np.isfinite(sigma):
            cutoff = max(cutoff, mode + mode_k * sigma)
    flat = np.where(np.isfinite(mip), mip, -np.inf)
    yx = peak_local_max(flat, min_distance=min_distance, exclude_border=1,
                        threshold_abs=cutoff)
    if len(yx) == 0:
        return []
    bright = mip[yx[:, 0], yx[:, 1]]
    out = []
    # n_max=None means every anchor the threshold proposed. The threshold
    # is mode + k*sigma of THIS crop, so how many there are is already the
    # data's answer; a fixed ceiling on top of it is a second, arbitrary
    # one that does not transfer between stores.
    order = np.argsort(bright)[::-1]
    for j in (order if n_max is None else order[:int(n_max)]):
        y, x = int(yx[j][0]), int(yx[j][1])
        # THIS ANCHOR'S OWN COLUMN, not the y-collapsed profile.
        #
        # It used to read `bimg[x]`, the max over ALL y at that x, and
        # take that profile's argmax -- the brightest plane anywhere in
        # the column, which for a crop with more than one emitter is
        # routinely some other spot's plane. MEASURED over the 1213
        # anchor-only candidates of a real bundle: the stored z matched
        # the y-collapsed argmax 420/420 of the time and the anchor's own
        # 267/420, and 33% of them sat more than 3 planes from the
        # emitter -- far enough that the reviewer's YX panel, drawn at
        # exactly that plane, showed background. Every anchor is above
        # threshold at its OWN plane by construction, so a z that puts it
        # below background can only be the wrong z.
        #
        # bimg survives as the fallback for a column that is entirely
        # masked out, which the mask-free extractor no longer produces
        # but an in-cell caller still can.
        col = stack[y, x, :]
        if not np.isfinite(col).any():
            col = bimg[x]
        if not np.isfinite(col).any():
            continue
        out.append((float(y), float(x), float(np.nanargmax(col))))
    return out


def dedupe(items, key=lambda s: s, min_sep=2.0):
    """Drop fits that landed on the same emitter, keeping the best p.

    A generous anchor uses min_distance=2, so two anchors 2 px apart are
    both proposed -- and a real emitter is ~1.3 px wide, so both fits
    converge on it and return nearly the same point. MEASURED on real
    data: candidates like (50.6, 65.0, p=0.450) and (50.6, 64.9, p=0.449)
    in the same crop, which are one spot reported twice.

    That matters more here than it would in production. A duplicate is a
    review a person has to make twice, and if the two copies are judged
    differently the training set contains a direct contradiction about
    one piece of pixel data.

    Lateral only. Two emitters genuinely stacked in z within 2 px
    laterally cannot be told apart by this fit anyway, so pretending the
    z separation resolves them would be inventing a distinction.
    """
    kept = []
    for it in sorted(items, key=lambda t: -key(t).p):
        s = key(it)
        if any((s.y - key(k).y) ** 2 + (s.x - key(k).x) ** 2 < min_sep ** 2
               for k in kept):
            continue
        kept.append(it)
    return kept


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

    def __init__(self, anchor=None, dedup_px=2.0, **fit_params):
        self.anchor = dict(anchor or {})
        self.dedup_px = float(dedup_px)
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
        out = dedupe(out, min_sep=self.dedup_px)
        out.sort(key=lambda s: -s.p)
        return out if n_max is None else out[:int(n_max)]


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

    The two reference constants are MEASURED, not chosen. They are the
    MEDIAN CI of a real generous-candidate population -- 101 fits over 40
    real cell crops from the MAZ store -- so each term is 0.5 at the
    median candidate and the score spreads over the range that actually
    occurs:

        ci_xy nm   p25  130   MEDIAN  355   p75  859   p90 2402
        ci_z  nm   p25  280   MEDIAN  453   p75  872   p90 1675
        occupancy  p25  0.1   MEDIAN  0.3   p75  0.5   p90  0.9
        at a position bound: 36 of 101

    An earlier version used 100 and 300 nm, which were invented. At 100 nm
    the lateral term is 0.22 for a MEDIAN candidate and the whole score
    compresses into a band near zero, which ranks nothing. Note also that
    the median generous candidate has occupancy 0.3 against a production
    gate of 0.40 -- i.e. most of this population would be rejected in
    production. That is the intended shape of a training set, not a fault.
    """

    name = 'anchor-v2'
    REF_CI_XY_NM = 355.0
    REF_CI_Z_NM = 453.0

    def __init__(self, anchor=None, params=None, fit_radius=8, keep_top=None,
                 ref_ci_xy_nm=None, ref_ci_z_nm=None, dedup_px=2.0):
        self.anchor = dict(anchor or {})
        self.params = params
        self.fit_radius = int(fit_radius)
        # THE review-effort control, and the reason the threshold can stay
        # low. Candidate counts per cell are heavy-tailed -- MEASURED
        # median 21, max 156 at mode+1sigma -- so no threshold both keeps
        # the dim spots and bounds what a person is shown. Anchor
        # generously, fit, then hand over only the best keep_top by p.
        self.keep_top = keep_top
        self.ref_ci_xy_nm = float(ref_ci_xy_nm if ref_ci_xy_nm is not None
                                  else self.REF_CI_XY_NM)
        self.ref_ci_z_nm = float(ref_ci_z_nm if ref_ci_z_nm is not None
                                 else self.REF_CI_Z_NM)
        self.dedup_px = float(dedup_px)

    def _params(self):
        from . import tracing_v2 as V2
        return self.params if self.params is not None else V2.V2Params()

    def localize_detailed(self, stack, seed_yxz=None, n_max=1, seeds=None):
        """[(LocalizedSpot, gate_pass, reason), ...], best-p first.

        The richer return `localize` throws away. gate_pass is the REAL
        production verdict -- tracing_v2.gate against SPOT_V2_GATES on
        this fit and this cube -- not a threshold on p. p is a composite
        ranking and comparing it to an occupancy threshold, as an earlier
        version did, produced sentences like "p 0.39 below 0.40" that
        named a gate the number had nothing to do with.

        `seeds` lets a caller supply anchors it has already computed, so
        the anchor step runs ONCE per crop rather than once here and once
        in the caller. reason is '' when the gate passed.
        """
        from . import tracing_v2 as V2
        from .localization import SPOT_V2_GATES
        if stack is None or stack.size == 0 or not np.isfinite(stack).any():
            return []
        p = self._params()
        if seeds is None:
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
            rxy, rz = self.ref_ci_xy_nm, self.ref_ci_z_nm
            score = (max(0.0, occ if np.isfinite(occ) else 0.0)
                     * rxy / (rxy + (ci_xy if np.isfinite(ci_xy) else 1e6))
                     * rz / (rz + (ci_z if np.isfinite(ci_z) else 1e6)))
            railed = getattr(fit, 'at_bound', None) or ()
            if isinstance(railed, str):
                railed = (railed,)
            if any(n in ('y', 'x', 'z') for n in railed):
                score *= 0.25
            ok, why = V2.gate(fit, cube, SPOT_V2_GATES, p.voxel_um)
            out.append((_spot(y=fit.y + y0, x=fit.x + x0, z=fit.z,
                              p=max(min(score, 1.0), 1e-6),
                              amplitude=fit.amplitude, offset=fit.offset,
                              sigma_y=fit.sigma_y_um, sigma_x=fit.sigma_x_um,
                              sigma_z=fit.sigma_z_um),
                        bool(ok), '' if ok else str(why or 'gate rejected')))
        out = dedupe(out, key=lambda t: t[0], min_sep=self.dedup_px)
        out.sort(key=lambda t: -t[0].p)
        cap = self.keep_top if n_max is None else (
            min(n_max, self.keep_top) if self.keep_top else n_max)
        return out if cap is None else out[:int(cap)]

    def localize(self, stack, seed_yxz=None, n_max=1):
        return [s for (s, _ok, _why)
                in self.localize_detailed(stack, seed_yxz=seed_yxz, n_max=n_max)]


class PsfMatchEngine(LocalizeEngine):
    """Find spots by matching a MEASURED PSF, not by fitting a Gaussian.

    The template comes from psf_bank -- an average of boxes humans
    confirmed -- and detection is a normalised cross-correlation whose
    local maxima are the emitters. That is the whole point of it: a box
    with three spots in it yields three peaks, with no decision about how
    many components to fit and no mixture model to fail.

    p IS THE NCC SCORE mapped onto (0, 1], so it is a SHAPE agreement and
    not a probability -- LocalizedSpot's own docstring asks each engine
    to say which it is. The Gaussian fields are NaN: there is no fit here
    and inventing a sigma to fill them would be a lie a caller could not
    detect.

    Give it either `templates` (arrays) or `bank` (a path), plus the
    `voxel_um` the data is on -- psf_bank.load refuses a grid the
    template was not measured on.
    """

    name = 'psf-match'

    def __init__(self, templates=None, bank=None, storage_path=None,
                 voxel_um=None, r=None, rz=None,
                 min_distance=3, threshold=None, k_sigma=None, **_ignored):
        from . import psf_bank as PB
        r = PB.DEFAULT_R if r is None else int(r)
        rz = PB.DEFAULT_RZ if rz is None else int(rz)
        self.meta = {}
        if templates is not None:
            templates = ([templates] if np.asarray(templates).ndim == 3
                         else list(templates))
        elif storage_path:
            # THE ORDINARY WAY TO BUILD ONE: render the store's own
            # calibrated shape at the matching size. A formula can be
            # rendered at whatever size the search needs, which is the
            # whole advantage of keeping the PSF as parameters -- and it
            # is what makes psf_bank.DEFAULT_R/RZ mean something. They
            # were dead constants until this path existed: nothing read
            # them, so setting them changed nothing.
            from . import psf as P
            doc = P.load(storage_path)
            if not doc:
                raise ValueError(f'{storage_path} has no calibrated PSF; '
                                 'pass templates= or bank=')
            vx = voxel_um or doc.get('voxel_um') or P.DEFAULT_VOXEL_UM
            templates = [PB.normalise(
                PB.render(doc['family'], doc['params'], r, rz, tuple(vx)))]
            self.meta = {'family': doc['family'], 'params': doc['params'],
                         'voxel_um': list(vx), 'r': r, 'rz': rz,
                         'source': 'store calibration'}
        elif bank:
            mean, comps, meta = PB.load(bank, voxel_um=voxel_um)
            # CUT IT TO THE SEARCH SIZE. A bank is stored at the size it
            # was measured at -- train_spotmodel averages the
            # classifier's 15 x 15 x 25 boxes -- and ncc() scores only
            # where the template fits, so handing that straight to a
            # 15 x 15 pillar leaves ONE lateral position to search and
            # every neighbour is unfindable at any threshold. See
            # psf_bank.centre_crop, which measured it.
            meta = dict(meta)
            meta['stored_shape'] = list(np.asarray(mean).shape)
            if tuple(np.asarray(mean).shape) != (2 * r + 1, 2 * r + 1,
                                                 2 * rz + 1):
                mean = PB.normalise(PB.centre_crop(mean, r, rz))
                meta['cropped_to'] = [2 * r + 1, 2 * r + 1, 2 * rz + 1]
            templates = [mean]
            self.meta = meta
        else:
            raise ValueError('psf-match needs templates=, bank= or '
                             'storage_path=')
        self.templates = [np.asarray(t, float) for t in templates]
        self.min_distance = int(min_distance)
        # None means "this template's own 4.5 sigma" -- see
        # psf_bank.K_SIGMA for why a raw NCC number cannot be a
        # default when the template size can change.
        self.threshold = None if threshold is None else float(threshold)
        # HOW MANY SIGMA A PEAK MUST CLEAR, when no absolute threshold is
        # given. Separate from `threshold` because it is the parameter
        # that survives a change of template size: a raw NCC number does
        # not (psf_bank.K_SIGMA's own header says why), and a caller who
        # wants a looser or stricter search wants it in these units.
        self.k_sigma = (PB.K_SIGMA if k_sigma is None else float(k_sigma))

    def localize(self, stack, seed_yxz=None, n_max=1):
        from . import psf_bank as PB
        # WITH A SEED, TAKE EVERYTHING AND THEN CHOOSE. Cutting to n_max
        # inside match() ranks by score, so the seed arrived after the
        # only candidate it could have picked had already been thrown
        # away -- asking for the spot nearest a click returned the
        # brightest spot in the box instead, silently.
        hits = PB.match(np.asarray(stack, float), self.templates,
                        min_distance=self.min_distance,
                        threshold=self.threshold, k_sigma=self.k_sigma,
                        n_max=None if seed_yxz is not None else n_max)
        if seed_yxz is not None and hits:
            sy, sx, sz = (float(v) for v in seed_yxz)
            hits.sort(key=lambda r: (r[0] - sy) ** 2 + (r[1] - sx) ** 2
                      + (r[2] - sz) ** 2)
            if n_max is not None:
                hits = hits[:int(n_max)]
        # NCC runs to -1; only agreement is evidence, so the negative
        # half collapses to the floor rather than wrapping around.
        return [_spot(y, x, z, max(min((s + 1.0) / 2.0, 1.0), 1e-6))
                for (y, x, z, s) in hits]


def _v3():
    """The learned engine, imported on demand.

    It reads the training package (features, the box window, the
    classifier) and training.dataset reads THIS module for
    background_mode, so a module-level import here is a cycle. Deferring
    it to first use costs one dict lookup and keeps both directions
    legal.
    """
    from .psfmatcher import PsfMatcherV3Engine
    return PsfMatcherV3Engine


ENGINES = {
    GaussianLocalizeEngine.name: GaussianLocalizeEngine,
    AnchorFitEngine.name: AnchorFitEngine,
    AnchorFitV2Engine.name: AnchorFitV2Engine,
    PsfMatchEngine.name: PsfMatchEngine,
    'v3-psfmatcher': _v3,
    # THE TWO NAMESPACES MEET HERE. What a person picks is a ROUTE
    # (tracing_v2.ROUTE_*, one vocabulary across both panels and the
    # config); what this factory builds is an ENGINE. They were entirely
    # unconnected -- no config key, GUI widget or CLI flag ever supplied
    # this factory's `name`, and every in-app callsite passed the literal
    # 'gaussian' -- so make_engine's own promise that "a config/UI names
    # an engine through" it was wired to nothing. These aliases let a
    # stored route name an engine directly, with no translation table to
    # drift.
    'v1': GaussianLocalizeEngine,
    'v2-anchor-fit': AnchorFitV2Engine,
}


def make_engine(name, **params):
    """The one factory a config/UI names an engine through."""
    try:
        cls = ENGINES[name]
    except KeyError:
        raise ValueError(f'unknown localize engine {name!r} -- known: {sorted(ENGINES)}')
    # A value may be a class or a thunk that imports one; see _v3.
    if not isinstance(cls, type):
        cls = cls()
    return cls(**params)
