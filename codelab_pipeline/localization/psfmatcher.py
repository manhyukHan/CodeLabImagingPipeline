"""
v3-psfmatcher: stack in, [(y, x, z, p_exist), ...] out, nobody in between.

WHAT WAS MISSING. The training stack produced a calibrated existence
probability and nothing loaded it -- SpotClassifier.load had no caller
outside a test, so the one number in this repo that is a probability
reached no engine, no gate and no z_status. This is the engine that
reads it.

TWO LEARNED ARTEFACTS AND ONE ALGORITHM, not three models:

    candidates       the wrapper below. No parameters, not a model.
    the classifier   SpotClassifier: weights + a Platt pair, trained on
                     human keep/drop           -> p_exist
    the template     psf_bank.h5: the MEAN of 1,157 human-confirmed
                     spots. Learned, but it is an array.
    the matcher      NCC + peak_local_max over that template. ZERO
                     learned parameters        -> p = (NCC + 1) / 2

Calling the template and the search "model 2" and "model 3" split one
thing -- a matched filter with a data-derived kernel -- into two, and
this module said so for a while. There is no third model, and `p` is not
the output of a final layer: it is an affine remap of a correlation
coefficient, so p = 0.74 means NCC = 0.48 and nothing about probability.

AND THEY ARE NOT INDEPENDENT. The classifier and the template come from
the SAME 1,157 human positives -- train_spotmodel builds one from the
feature vectors and the other from the voxels of those very spots. When
they agree, that is two readings of one label set and not a second
opinion, and any number quoting their agreement has to say so.

p_exist IS THE CALIBRATED ANSWER AND p IS NOT. The classifier is trained
on "did a person keep this", with a proper scoring rule and Platt-scaled
on held-out cells. The NCC scores how well a thing matches a PSF, which
is a different question -- a bright blob of non-specific stain matches a
PSF beautifully, and MEASURED on 613 labels the matcher found a peak on
172 spots both the classifier and the reviewer called nothing.

BUT ON ITS OWN QUESTION THE NCC IS EXCELLENT, and that is worth saying
because this module used to claim otherwise. Asked "is this EXTRA match,
beside a spot already confirmed, a real emitter?", MEASURED on 387
matches from 150 human-judged pillars: PR-AUC 0.993, best F1 at p = 0.74
(precision 0.958, recall 0.975). The caveat is the selection -- every one
of those 387 was proposed BY the matcher, so this says its ranking is
good among what it offers, not that it offers everything. And 0.74 is
tied to the 7 x 7 x 11 template: psf_bank.K_SIGMA's own header explains
why a raw NCC number moves when the template size does.

THE CANDIDATE LAYER IS INSIDE THIS MODULE ON PURPOSE. It looks like
pre-processing a caller could own, and it must not be: v4 is meant to
absorb candidate-finding into its own network, and if the app calls a
candidate finder and then an engine, v4 cannot replace v3 without the
app changing too. Everything from raw stack to final coordinates lives
behind one localize() call, so a v4 that finds candidates in a
convolution is a drop-in.

VIEW: 'cell' OR 'fov', AND IT ONLY AFFECTS THE FIRST LAYER. Background
here is the mode plus its left-hand sigma (engine.background_mode) --
a per-region statistic. Estimated over a whole field it is one number
for cells that differ in stain uptake, focus and thickness, so a dim
cell's spots fall under a threshold set by a bright cell's background
and a bright cell's noise clears it. In 'cell' view every labelled
region gets its own background, its own candidates and its own boxes;
'fov' takes one background for the field, and is there for a field with
no segmentation rather than as an equal option.

Given a stack that is ALREADY one cell's crop, both views agree: one
region, one background. That is the ordinary call and it needs no mask.
"""
import numpy as np

from .engine import (LocalizeEngine, _spot, anchor_candidates,
                     background_mode, dedupe)

# The box the classifier was trained on. NOT a free parameter: features
# are computed on a fixed-size window and a Standardiser refuses a
# changed NAMES, so a box of another size would produce numbers that
# mean something else under the same names.
BOX_R = 7
BOX_RZ = 12

# How generous the candidate layer is. A candidate is not a detection --
# model 1 judges every one of them and the p-gate cuts the tail -- so
# this errs toward offering, the same way the training extractor does.
DEFAULT_ANCHOR = dict(min_distance=3, mode_k=1.0, threshold_rel=0.0,
                      max_to_background=0.0)

# Candidates closer than this are the same emitter. Lateral only, like
# engine.dedupe: this layer has no z model to justify separating them.
DEDUP_PX = 2.0

VIEWS = ('cell', 'fov')


def is_refined(spot):
    """Whether model 3 placed this spot, or only model 1 believes in it.

    A spot model 3 could not place keeps its ANCHOR's integer position
    and NaN in `p`. It is not deleted -- MEASURED on 613 human-labelled
    spots, the case is rare (4, 0.7%) and 3 of those 4 were REAL by human
    judgement, against 94% for spots both models agree on. Model 3 scores
    SHAPE, and its silence is weak evidence at best: on the same set it
    found a peak on 172 spots that both the classifier and the reviewer
    called nothing.

    But an integer position is not a measurement. A pixel here is 208 nm
    against a localization precision of ~30 nm, so a caller that needs a
    POSITION -- a chromatin trace does -- must be able to tell these
    apart rather than silently write a coordinate an order of magnitude
    worse than the rest. That is what this is for, and it is why NaN is
    left in `p` rather than filled with something plausible.
    """
    return bool(np.isfinite(getattr(spot, 'p', float('nan'))))


def _joint(p1, p3, cal):
    """P(a real emitter is HERE) = p1 * P(this match is real | pillar real).

    THIS IS THE CHAIN RULE, NOT AN INDEPENDENCE ASSUMPTION, and that is
    what makes the product a probability rather than a score:

      p1        the classifier's calibrated answer to "does this pillar
                hold a real spot at all", trained on pass/fail verdicts
                over crop-pillars.
      cal(p3)   the multispot Platt's answer to "is THIS match a real
                emitter", trained on verdicts inside pillars that
                already contain one spot everyone agreed on (see
                verdicts.multispot_labels: 'every pillar in this file
                already contains one spot everybody agreed on -- the
                seed'). It is therefore conditional on p1's event.

    P(A and B) = P(A) * P(B | A) exactly. No rescaling is needed or
    wanted: both factors are already in (0, 1), so the product is, and
    renormalising a chain rule would destroy the calibration it has.

    NO CALIBRATION SHIPPED -> p1 UNCHANGED. Multiplying by a raw NCC
    would look like a probability and not be one; a model directory
    without psf_multispot.json simply keeps the pillar-level answer.
    """
    import numpy as _np
    p1 = float(p1)
    if cal is None or not _np.isfinite(p3):
        return p1
    return p1 * float(cal.score(float(p3)))


def _peak_above(stack, y, x, z, background):
    """The image value at a found position, above this region's background.

    A MEASURED PEAK, NOT A FITTED AMPLITUDE. psf-match runs no Gaussian,
    so there is no fitted amplitude and LocalizedSpot.amplitude is NaN by
    design -- but AnAllele.polymer_adj carries a 4-tuple whose last
    element is what analysis.polymer.max_brightness compares, and NaN
    there does not raise. It makes the comparison undefined and the
    selector silently returns whichever candidate came first: MEASURED on
    real v3 output, exactly that happened, and the collapsed position was
    "the first one" dressed up as "the brightest one".

    Raw counts above this region's background, which is comparable
    between candidates of the same region and is at least the right KIND
    of quantity for that slot.
    """
    st = np.asarray(stack, float)
    h, w, d = st.shape
    iy = int(np.clip(round(float(y)), 0, h - 1))
    ix = int(np.clip(round(float(x)), 0, w - 1))
    iz = int(np.clip(round(float(z)), 0, d - 1))
    v = float(st[iy, ix, iz])
    return v - float(background) if np.isfinite(v) else float('nan')


class PsfMatcherV3Engine(LocalizeEngine):
    """The learned engine. One call: pixels in, coordinates and p out.

        engine = make_engine('v3-psfmatcher', model_dir='D:/models/mp58_rna')
        spots = engine.localize(cell_crop)
        spots = engine.localize(field, labels=cellmask)     # 'cell' view

    Needs a MODEL DIRECTORY -- what tools/train_spotmodel.py --out
    writes: spot_classifier_<head>.json, psf_bank.h5 and report.json.
    The head is not named here; classify.load_best takes the one the
    run's own held-out PR-AUC chose, so a re-train moves it without an
    edit. `head=` pins one for a comparison.
    """

    name = 'v3-psfmatcher'

    def __init__(self, model_dir=None, classifier=None, bank=None,
                 view='cell', head=None, anchor=None, dedup_px=DEDUP_PX,
                 k_sigma=None, min_distance=3, r=None, rz=None,
                 voxel_um=None, **_ignored):
        if str(view) not in VIEWS:
            raise ValueError(f'view must be one of {VIEWS}, not {view!r}')
        self.view = str(view)
        self.model_dir = str(model_dir) if model_dir else None
        self.head = head
        self.anchor = dict(DEFAULT_ANCHOR, **(anchor or {}))
        self.dedup_px = float(dedup_px)
        self.bank_path = bank
        self._k_sigma = k_sigma
        self._min_distance = int(min_distance)
        # THE SIZE THE RUN WAS TRAINED AT, unless a caller overrides it.
        # report.json records template_r/template_rz, so a model trained
        # at another size is SEARCHED at that size without anyone
        # remembering to pass a flag -- and the multispot calibration's
        # own check then agrees instead of refusing.
        if (r is None or rz is None) and model_dir:
            import json as _json
            import os as _os
            try:
                with open(_os.path.join(str(model_dir), 'report.json'),
                          encoding='utf-8') as _f:
                    _rep = _json.load(_f)
                r = _rep.get('template_r') if r is None else r
                rz = _rep.get('template_rz') if rz is None else rz
            except (OSError, ValueError):
                pass
        self._r, self._rz, self._voxel_um = r, rz, voxel_um
        self._clf = classifier
        self._why = None
        self._match = None
        self._refines = None
        self._no_refine = None
        self._cal = False              # False = not looked for yet

    # -- the models ------------------------------------------------------

    @property
    def classifier(self):
        """Model 1, loaded once. Lazy so that constructing this engine to
        read its name or its view does not need a trained model on disk."""
        if self._clf is None:
            from ..training import classify as C
            if not self.model_dir:
                raise ValueError(
                    'v3-psfmatcher needs model_dir= (what '
                    'tools/train_spotmodel.py --out wrote) or classifier=')
            self._clf, self._why = C.load_best(self.model_dir,
                                               prefer=self.head)
        return self._clf

    @property
    def matcher(self):
        """Models 2 and 3: the measured template, and the search over it.

        Raises if there is no bank. Callers that can live without
        sub-voxel refinement should ask `refines` first -- see
        _refine for why running without it is a legitimate state and
        not a silent one.
        """
        if self._match is None:
            import os
            from .engine import make_engine
            bank = self.bank_path
            if not bank and self.model_dir:
                bank = os.path.join(self.model_dir, 'psf_bank.h5')
            self._match = make_engine(
                'psf-match', bank=bank, voxel_um=self._voxel_um,
                r=self._r, rz=self._rz, k_sigma=self._k_sigma,
                min_distance=self._min_distance)
        return self._match

    @property
    def refines(self):
        """Whether model 3 is available. False means INTEGER positions.

        A missing PSF bank is not a reason to answer nothing -- model 1
        still says whether a spot is there, and an anchor still says
        roughly where. But it is a reason to be able to TELL, so this is
        a property a caller can read and `no_refine_why` says what went
        wrong, rather than positions quietly losing their sub-voxel
        precision and every p going NaN with no explanation.
        """
        if self._refines is None:
            try:
                self.matcher
                self._refines, self._no_refine = True, None
            except Exception as exc:                        # noqa: BLE001
                self._refines = False
                self._no_refine = f'{type(exc).__name__}: {exc}'
        return self._refines

    @property
    def no_refine_why(self):
        self.refines
        return self._no_refine

    @property
    def multispot_cal(self):
        """The matcher's shipped Platt pair, or None.

        Loaded from the model directory like everything else, so a new
        experiment needs NO multispot labels to get a calibrated matcher
        score -- and anyone with their own labels re-fits it the same way
        they would re-fit the classifier. Absent, the matcher keeps its
        sigma threshold and `p` stays a raw NCC remap.
        """
        if self._cal is False:
            import os
            from ..training import classify as C
            self._cal = None
            path = (os.path.join(self.model_dir, C.MULTISPOT_NAME)
                    if self.model_dir else None)
            if path and os.path.exists(path):
                cal = C.MultispotCalibration.load(path)
                if self.refines:
                    # REFUSE ONE FITTED ON ANOTHER TEMPLATE. The score it
                    # calibrates is an affine remap of a raw NCC, which
                    # means something different at another template size.
                    cal.check(tuple(np.asarray(
                        self.matcher.templates[0]).shape))
                self._cal = cal
        return self._cal

    @property
    def chosen(self):
        """Which head is in use and why. None until the classifier loads."""
        self.classifier
        return self._why

    # -- layer 1: candidates ---------------------------------------------

    def regions(self, stack, labels=None):
        """[(mask_or_None, background, sigma), ...] -- one per background.

        This is the whole of what `view` decides. 'cell' with labels
        gives one entry per region; anything else gives one entry for the
        field, which is also what a stack that is already a single cell's
        crop gets.
        """
        st = np.asarray(stack, float)
        if self.view != 'cell' or labels is None:
            bg, sg = background_mode(st)
            return [(None, bg, sg)]
        lab = np.asarray(labels)
        out = []
        for v in np.unique(lab):
            if v == 0:
                continue                  # 0 is background, not a cell
            m = (lab == v)
            if not m.any():
                continue
            # THE BACKGROUND IS THIS CELL'S, from this cell's own pixels.
            here = np.where(m[:, :, None], st, np.nan)
            bg, sg = background_mode(here)
            if not np.isfinite(bg) or not np.isfinite(sg) or sg <= 0:
                continue                  # nothing to measure against
            out.append((m, float(bg), float(sg)))
        return out

    def candidates(self, stack, mask=None, bg=None, sigma=None, n_max=None):
        """Where to look, in this region, at this region's background."""
        st = np.asarray(stack, float)
        if mask is not None:
            st = np.where(mask[:, :, None], st, np.nan)
        if bg is None or sigma is None:
            bg, sigma = background_mode(st)
        if not np.isfinite(bg) or not np.isfinite(sigma) or sigma <= 0:
            return []
        seeds = anchor_candidates(st, n_max=n_max, **self.anchor)
        return [(float(y), float(x), float(z)) for (y, x, z) in seeds]

    # -- layer 2: boxes, and model 1 on them ------------------------------

    def _boxes(self, stack, seeds, bg, sigma):
        """(features, cores) for the classifier, cut the way it was trained.

        Delegates the window to training.dataset so the box a spot is
        scored in is the same object the model was fitted on, padding
        rule included. A second implementation of that window is the
        divergence this codebase keeps having to hunt down.
        """
        from ..training import dataset as D
        from ..training import features as F
        st = np.asarray(stack, float)
        h, w, d = st.shape
        feats, cores = [], []
        for (y, x, z) in seeds:
            iy, ix, iz = int(round(y)), int(round(x)), int(round(z))
            core, border = D._pad_window(st, iy, ix, iz, BOX_R, BOX_RZ, bg)
            core = (core - bg) / max(sigma, 1e-9)
            col = np.asarray(st[int(np.clip(iy, 0, h - 1)),
                                int(np.clip(ix, 0, w - 1)), :], float)
            col = (col - bg) / max(sigma, 1e-9)
            feats.append(F.one(core, col, frac_padded=float(border)))
            cores.append(core)
        return np.asarray(feats, float), np.asarray(cores, float)

    # -- the seam --------------------------------------------------------

    def localize(self, stack, seed_yxz=None, n_max=1, labels=None):
        """Every emitter this engine believes is in `stack`.

        Returns LocalizedSpot with `p_exist` from the classifier and
        `p` from the matcher's NCC. NOTHING IS DROPPED ON p_exist HERE -- the
        p-gate is posterior and a person chooses its threshold off a
        histogram of these very numbers, so an engine that pre-filtered
        would be choosing it for them and hiding the evidence.

        `seed_yxz`, when given, means "the one nearest here": the same
        contract every other engine keeps.
        """
        st = np.asarray(stack, float)
        out = []
        for mask, bg, sigma in self.regions(st, labels):
            seeds = self.candidates(st, mask, bg, sigma)
            if not seeds:
                continue
            feats, cores = self._boxes(st, seeds, bg, sigma)
            p_exist = self.classifier.score(X=feats, boxes=cores)
            for (y, x, z), pe in zip(seeds, p_exist):
                out.extend(self._refine(st, y, x, z, float(pe), bg, sigma))
        # THE RESOLUTION BOUND, NOT A LATERAL DEDUP. engine.dedupe merges
        # anything within 2 px laterally whatever its z, and says so:
        # "two emitters genuinely stacked in z ... cannot be told apart by
        # this fit anyway". That is true of a Gaussian anchor fit and
        # FALSE of a matched filter -- MEASURED over 2,639 pairs, 130 of
        # the 135 within 2 px laterally are more than 5 planes apart,
        # median 17.3, which is 7 sigma_z and comfortably resolved. So the
        # bound is the PSF's own anisotropy (psf_bank.resolution_bound),
        # and what survives a collision is the BRIGHTER one rather than
        # the better-correlated one.
        from . import psf_bank as PB
        lat, ax = PB.resolution_bound(getattr(self.matcher, 'meta', None)
                                      if self.refines else None,
                                      lateral_px=self.dedup_px)
        out = PB.merge_unresolvable(
            out, lambda h: (h.amplitude if np.isfinite(h.amplitude)
                            else -np.inf), lat, ax)
        if seed_yxz is not None:
            sy, sx, sz = (float(v) for v in seed_yxz)
            out.sort(key=lambda s: ((s.y - sy) ** 2 + (s.x - sx) ** 2
                                    + (s.z - sz) ** 2))
        else:
            out.sort(key=lambda s: -s.p_exist)
        return out if n_max is None else out[:int(n_max)]

    def _joint_p(self, p1, p3):
        """Public spelling of the joint probability, for callers and tests."""
        return _joint(p1, p3, self.multispot_cal)

    def _refine(self, stack, y, x, z, p1, bg, sigma):
        """The matched filter on one candidate. Returns a LIST.

        The template is searched in a window around the candidate, and a
        candidate model 3 finds nothing at keeps the anchor's own integer
        position -- a spot the classifier believes in does not stop
        existing because the matched filter had no peak over threshold,
        it just has no sub-voxel refinement.

        EVERY HIT FROM ONE CANDIDATE USED TO SHARE ONE p_exist, which
        made the second and third emitter in a pillar ungateable: they
        carried the pillar's number, not their own, so no threshold could
        separate a real second locus from a spurious one. Each hit now
        gets p1 * P(this match is real | the pillar is real) -- see
        _joint below for why that is a chain rule and not an assumption.
        """
        if not self.refines:
            return [_spot(y, x, z, float('nan'), p_exist=p1,
                          amplitude=_peak_above(stack, y, x, z, bg))]
        st = np.asarray(stack, float)
        h, w, d = st.shape
        tpl = np.asarray(self.matcher.templates[0])
        ry, rx = tpl.shape[0] // 2, tpl.shape[1] // 2
        # Searched wider than the box it reports in, for psf_bank's own
        # reason: ncc scores only where the template FITS, so a window
        # cut to the candidate would only report its central few voxels
        # and push everything else onto the boundary.
        iy, ix = int(round(y)), int(round(x))
        pad = BOX_R + max(ry, rx)
        y0, y1 = max(0, iy - pad), min(h, iy + pad + 1)
        x0, x1 = max(0, ix - pad), min(w, ix + pad + 1)
        win = (st[y0:y1, x0:x1, :] - bg) / max(sigma, 1e-9)
        hits = []
        if win.shape[0] > tpl.shape[0] and win.shape[1] > tpl.shape[1]:
            # n_max=None, NOT 1. Returning one position per candidate is
            # what stopped this engine from demultiplexing at all: two
            # emitters in one candidate's neighbourhood came back as one,
            # and the only reason two ever appeared was that the CANDIDATE
            # layer happened to propose two seeds. MEASURED on 150 human
            # multispot verdicts, 43% of confirmed pillars hold more than
            # one real spot -- so one-per-candidate discards the answer to
            # the question this engine exists to ask.
            hits = self.matcher.localize(win, seed_yxz=None, n_max=None)
        cal = self.multispot_cal
        out = []
        for hh in hits:
            fy, fx, fz = hh.y + y0, hh.x + x0, hh.z
            out.append(_spot(fy, fx, fz, hh.p,
                             p_exist=_joint(p1, hh.p, cal),
                             amplitude=_peak_above(st, fy, fx, fz, bg)))
        if out:
            return out
        # NO MATCH: p3 does not exist, so neither does the product. The
        # anchor keeps the PILLAR's number and `p` stays NaN, which is
        # what is_refined() reads. Pushing these to ~0 instead would be
        # inventing a measurement, and would contradict one: MEASURED,
        # 3 of the 4 unrefined spots in 613 labels were real.
        return [_spot(y, x, z, float('nan'), p_exist=p1,
                      amplitude=_peak_above(st, y, x, z, bg))]
