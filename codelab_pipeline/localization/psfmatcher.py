"""
v3-psfmatcher: stack in, [(y, x, z, p_exist), ...] out, nobody in between.

WHAT WAS MISSING. The training stack produced a calibrated existence
probability and nothing loaded it -- SpotClassifier.load had no caller
outside a test, so the one number in this repo that is a probability
reached no engine, no gate and no z_status. This is the engine that
reads it.

THE THREE MODELS RUN IN ORDER, with no human in between:

    candidates   the wrapper below, NOT a model
    model 1      SpotClassifier   -> p_exist, "is a spot here at all"
    model 2      psf_bank.h5      -> the measured template
    model 3      psf-match        -> sub-voxel (y, x, z) + an NCC quality

p_exist IS THE ANSWER AND THE OTHERS ARE NOT. Model 1 is trained on
"did a person keep this", which is the question. Models 2 and 3 score
how well a thing matches a PSF, which is a statement about quality --
a bright blob of non-specific stain matches a PSF beautifully. So
`p_exist` carries the classifier and `p` keeps the NCC, in two fields,
for the reason LocalizedSpot's own docstring now gives at length.

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
        self._r, self._rz, self._voxel_um = r, rz, voxel_um
        self._clf = classifier
        self._why = None
        self._match = None
        self._refines = None
        self._no_refine = None

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

        Returns LocalizedSpot with `p_exist` set from model 1 and `p`
        from model 3's NCC. NOTHING IS DROPPED ON p_exist HERE -- the
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
                out.append(self._refine(st, y, x, z, float(pe), bg, sigma))
        # Lateral dedup on the FINAL positions: two regions can propose
        # the same emitter on a boundary, and model 3 can walk two seeds
        # onto one peak.
        out = dedupe(out, min_sep=self.dedup_px)
        if seed_yxz is not None:
            sy, sx, sz = (float(v) for v in seed_yxz)
            out.sort(key=lambda s: ((s.y - sy) ** 2 + (s.x - sx) ** 2
                                    + (s.z - sz) ** 2))
        else:
            out.sort(key=lambda s: -s.p_exist)
        return out if n_max is None else out[:int(n_max)]

    def _refine(self, stack, y, x, z, p_exist, bg, sigma):
        """Model 3 on one candidate: sub-voxel position and an NCC quality.

        The template is searched in a window around the candidate, and a
        candidate model 3 finds nothing at keeps the anchor's own integer
        position -- a spot the classifier believes in does not stop
        existing because the matched filter had no peak over threshold,
        it just has no sub-voxel refinement.
        """
        if not self.refines:
            return _spot(y, x, z, float('nan'), p_exist=p_exist)
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
            hits = self.matcher.localize(
                win, seed_yxz=(y - y0, x - x0, z), n_max=1)
        if hits:
            hh = hits[0]
            return _spot(hh.y + y0, hh.x + x0, hh.z, hh.p, p_exist=p_exist)
        return _spot(y, x, z, float('nan'), p_exist=p_exist)
