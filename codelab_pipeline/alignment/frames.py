"""
ONE place that composes the alignment chain.

Every coordinate question in this pipeline is the same question: take a
point (or mask) expressed in ONE (hybe, modality) frame and express it in
ANOTHER. Before this module that composition was written out longhand at
~21 call sites -- MainWindow had 11 resolvers, ACell 5, chain.py 3,
spot_mapper 2 -- each assembling its own subset of the same layers. Every
alignment bug this codebase has had was two of those sites disagreeing:
a stale anchor, a missing layer, a matrix read in the wrong source frame,
a drift applied to one side of a comparison but not the other. None of
them were arithmetic errors; they were all "wrong set of matrices".

FrameResolver makes the set a FUNCTION OF THE TWO ENDPOINTS instead of
something a caller assembles by hand.

The model it implements
-----------------------
Three XY layers and two Z layers:

  same-modality   XY   within[modality][hybe]      hybe -> that modality's own frame
  cross-modality  XY,Z bridge / z_bridge           one modality's frame -> the other's
  cell-level      XY,Z cell.matrices[(hybe,mod)]   fitted per cell, same-modality only

There is ONE shared frame -- the RNA-side modality's own reference frame.
Every route to it is via `to_shared`, and any frame-to-frame transform is
`inv(to_shared(dst)) @ to_shared(src)`. That single definition is what
makes "same hybe -> identity" and "A->B is the inverse of B->A" true by
construction rather than by inspection.

Two structural facts this relies on
-----------------------------------
1. Z is separable. With no XZ/YZ rotation the transform is block-diagonal,
   so z is an additive scalar carried alongside the 2D affine, never
   inside it. This is true by construction, not by data.

2. The cell-level 'yx' is a BARE residual (H2 only). It is fitted on
   crops that were already FOV-corrected, so it stacks ON TOP of the FOV
   route at read time -- that recomposition happens here, from CURRENT
   matrices, so re-running FOV alignment updates every fitted cell without
   a re-fit. Entries written before this change have the FOV leg baked in
   (compose_chain([H1, H2])) and are marked by the ABSENCE of
   'yx_is_residual'; those are consumed whole, since recomposing would
   double-count.

Deliberately NOT relied upon: commutativity. All 22 stored matrices are
currently pure translation, which would make order free -- but the engine
permits rotation up to MAX_ALIGNMENT_ROTATION_DEG and the ORB/homography
path can produce it. So composition here is strictly ordered and correct
for any affine, and nothing anywhere may assume translation-only.
"""

import numpy as np
import numpy.linalg as la

# NO module-level `from . import chain`: chain.py re-exports FrameMatrices
# from THIS module at ITS OWN top, so whichever of the two imports first
# must not need the other at import time. When frames loaded first, its
# eager chain import re-entered chain, whose `from .frames import
# FrameMatrices` then hit this half-initialized module -- confirmed real
# ImportError, order-dependent so most sessions never saw it. chain is
# used in exactly one function (z_to_shared's entry_dz call); imported
# there, at call time, when both modules are fully initialized.


IDENTITY = np.eye(3)



class FrameMatrices(dict):
    """
    {(hybe, modality): 3x3} -- the FOV-level alignment matrices for one FOV.

    Keyed by the PAIR, never by hybe alone. A bare hybe name is not a key in
    this pipeline: the cross-modal bridge hybe (e.g. Hyb_130) is a real,
    distinct acquisition in both modalities with a different matrix in each,
    so `matrices['Hyb_130']` has no single correct answer. cell.matrices
    learned this already; this is the same lesson applied to the FOV layer,
    which had been carrying modality implicitly in an outer storage_path key.

    Bare-string lookups raise rather than returning None. A silent miss here
    is indistinguishable from "no alignment", which downstream code treats as
    identity -- that is how a wrong-frame lookup becomes a wrong picture
    instead of an error. Failing loudly at the call site is the whole point
    of the type.
    """

    def __init__(self, *args, modality=None, **kwargs):
        super().__init__(*args, **kwargs)
        # The modality these matrices were read for. Carried on the dict so a
        # call site holding only `fov_matrices` and a hybe can still name the
        # PAIR explicitly -- (hybe, fm.modality) -- instead of every function
        # in the chain growing a modality parameter it only forwards. None
        # once a dict spans more than one modality, at which point a caller
        # must say which it means.
        self.modality = modality

    @staticmethod
    def _check(key):
        if not (isinstance(key, tuple) and len(key) == 2):
            raise TypeError(
                f'FrameMatrices is keyed by (hybe, modality), got {key!r}. '
                f'A bare hybe is ambiguous -- the cross-modal bridge hybe '
                f'exists in both modalities with different matrices.')

    def __getitem__(self, key):
        self._check(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._check(key)
        return super().get(key, default)

    def __contains__(self, key):
        self._check(key)
        return super().__contains__(key)

    def for_modality(self, modality):
        """
        Just this modality's entries, as a FrameMatrices that knows its own
        modality.

        The store itself is keyed by FOV and spans every modality -- one
        source of truth, with storage_path no longer smuggling modality in
        an outer key. Callers that genuinely work in one frame ask for that
        view HERE, at the boundary where they already know which modality
        they mean, instead of every function downstream growing a modality
        argument it only forwards.
        """
        return FrameMatrices({k: v for k, v in self.items() if k[1] == modality},
                             modality=modality)


class FrameResolver:
    """
    Resolves (hybe, modality) -> (hybe, modality) for one FOV.

    Constructed with plain data, never with session objects, so it is
    unit-testable and cannot silently pick up UI state:

      within        {modality: {hybe: 3x3}}  each modality's own FOV layer,
                    every matrix mapping that hybe into ITS OWN modality's
                    frame (NOT pre-composed with any cross-modal term --
                    passing pre-bridged matrices here double-counts).
      shared        name of the modality whose frame IS the shared frame.
      bridge_xy     3x3, `bridge_from` modality's frame -> shared frame.
      bridge_z      float, planes, same direction as bridge_xy.
      bridge_from   which modality bridge_xy/bridge_z start in (the DNA
                    side). None disables cross-modal entirely.
      anchors       {modality: 3x3} each modality's cell-alignment anchor
                    -> shared frame. Pass LIVE values; a stored snapshot
                    goes stale whenever FOV alignment is re-run.
    """

    def __init__(self, within, shared, bridge_xy=None, bridge_z=0.0,
                 bridge_from=None, anchors=None):
        self.within = within or {}
        self.shared = shared
        self.bridge_xy = np.asarray(bridge_xy, dtype=float) if bridge_xy is not None else None
        self.bridge_z = float(bridge_z)
        self.bridge_from = bridge_from
        self.anchors = anchors or {}

    # -- cross-modal leg -------------------------------------------------

    def bridge(self, from_modality, to_modality):
        """
        3x3 taking from_modality's own frame into to_modality's.

        Takes FRAMES, never roles ("is this the cell's own modality?").
        The role-based version of this is what let a cell's two legs land
        in different frames depending on which modality happened to be
        loaded. Identity within one modality; the stored matrix in the
        stored direction; its inverse in the other.
        """
        if from_modality == to_modality:
            return IDENTITY
        if self.bridge_xy is None or self.bridge_from is None:
            # No cross-modal alignment accepted yet. That is a real,
            # answerable state -- "no correction known", i.e. no shift --
            # not a failure. Per the pipeline's identity-default rule,
            # absence of alignment at ANY layer is identity, and every
            # downstream step must still run: returning None here made
            # callers treat the whole FOV matrix layer as unavailable, so
            # spot assignment silently assigned nothing at all until a
            # cross-modal link happened to be accepted.
            return IDENTITY
        if {from_modality, to_modality} != {self.bridge_from, self.shared}:
            return None
        return self.bridge_xy if from_modality == self.bridge_from else la.inv(self.bridge_xy)

    def bridge_z_between(self, from_modality, to_modality):
        """Planes to ADD moving z from from_modality's frame to to_modality's."""
        if from_modality == to_modality:
            return 0.0
        if self.bridge_from is None:
            return 0.0
        if {from_modality, to_modality} != {self.bridge_from, self.shared}:
            return 0.0
        return self.bridge_z if from_modality == self.bridge_from else -self.bridge_z

    # -- to the shared frame ---------------------------------------------

    def to_shared(self, hybe, modality, cell=None, missing=None):
        """
        3x3 taking `hybe`'s own native frame into the shared frame.

        A layer that has not been COMPUTED YET contributes IDENTITY, never
        a failure: identity means "no correction known for this layer",
        which is the only honest default and lets every mapping work
        before the alignment chain is complete. Same principle
        hybe_to_cellref_matrix already applies with
        fov_matrices.get(hybe, eye).

        Missing layer names are appended to `missing` (a set, if given) so
        a caller can SAY the mapping is provisional rather than silently
        presenting it as aligned. That distinction matters most for the
        cross-modal bridge: defaulting it to identity asserts the two
        modalities coincide, which on this data is a real ~13px claim.

        EITHER the cell-level route OR the FOV route -- never their
        product (a cell entry already contains the FOV leg).
        """
        key = (hybe, modality)
        entry = getattr(cell, 'matrices', {}).get(key) if cell is not None else None
        anchor = self.anchors.get(modality) if entry is not None else None
        if entry is not None and anchor is None and missing is not None:
            missing.add(f'cell-anchor:{modality}')
        if entry is not None and anchor is not None and not entry.get('yx_is_residual'):
            # LEGACY entry: 'yx' is the complete hybe->anchor transform with
            # the FOV leg already baked in (compute_cell_alignment used to
            # store compose_chain([H1, H2])). Consume it whole -- recomposing
            # H1 on top would apply the FOV leg twice.
            return np.asarray(anchor, dtype=float) @ np.asarray(entry['yx'], dtype=float)

        H_within = self.within.get(modality, {}).get(hybe)
        if H_within is None:
            H_within = IDENTITY
            if missing is not None:
                missing.add(f'same-modality:{modality}/{hybe}')
        bridge = self.bridge(modality, self.shared)
        if bridge is None:
            bridge = IDENTITY
            if missing is not None and modality != self.shared:
                missing.add(f'cross-modal:{modality}->{self.shared}')
        fov_route = bridge @ np.asarray(H_within, dtype=float)
        if entry is None or anchor is None:
            return fov_route
        # Cell-level route: the BARE residual re-composed onto the CURRENT
        # FOV route. H2 was fitted in the anchor's frame, so it is
        # conjugated into the shared frame before being applied -- for the
        # translation-only matrices this pipeline actually produces the
        # conjugation is a no-op, but it is written generally so a rotated
        # residual would still land correctly.
        A = np.asarray(anchor, dtype=float)
        H2 = np.asarray(entry['yx'], dtype=float)
        return A @ H2 @ la.inv(A) @ fov_route

    def z_to_shared(self, hybe, modality, cell=None):
        """
        Additive z (planes) taking `hybe`'s own frame to the shared frame:
        the cell's own same-modality residual plus the cross-modal drift.

        FOV-bounded, so the cross-modal term applies even with cell=None --
        an unassigned spot still sits in a modality.
        """
        z_cell = 0.0
        if cell is not None:
            from . import chain   # call-time import -- see module header
            z_cell = chain.entry_dz(getattr(cell, 'matrices', {}).get((hybe, modality)))
        return z_cell + self.bridge_z_between(modality, self.shared)

    # -- the primitive ---------------------------------------------------

    def transform(self, src, dst, cell=None):
        """
        ((hybe, modality), (hybe, modality)) -> (H_yx 3x3, dz, missing).

        H_yx maps a point from src's native frame into dst's; dz is added
        to src's z to get dst's; `missing` is the set of layer names that
        were defaulted to identity because they have not been computed
        yet (empty when the chain is complete).

        ALWAYS returns a usable transform -- an uncomputed layer is
        identity, so mapping works from the first segmentation onward and
        improves as each layer lands. Check `missing` when you need to
        tell the user the result is provisional; ignore it when you just
        need the best available answer.

        Guarantees, by construction:
          src == dst              -> identity, dz 0, no missing
          transform(a,b) inverse  == transform(b,a)
          a->b->c                 == a->c
        """
        if tuple(src) == tuple(dst):
            return IDENTITY.copy(), 0.0, set()
        missing = set()
        H_src = self.to_shared(src[0], src[1], cell, missing)
        H_dst = self.to_shared(dst[0], dst[1], cell, missing)
        dz = (self.z_to_shared(src[0], src[1], cell)
              - self.z_to_shared(dst[0], dst[1], cell))
        return la.inv(H_dst) @ H_src, dz, missing
