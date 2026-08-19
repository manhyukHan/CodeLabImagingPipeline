import os
import numpy as np
import numpy.linalg as la
import h5py
import scipy.spatial.distance as ssd

from ..alignment import chain as alignment
from ..io import vlinks_store


def _frozen_xy(xy):
    """
    (y, x) coordinate arrays, WRITE-LOCKED. Per explicit decision: mask
    coordinates are never partially edited after definition (cytoplasm
    incorporation REPLACES area, demoting the old one to the nucleus
    fields -- it does not edit in place), so the arrays are stored
    non-editable. That is what lets tier transfer share them by reference
    instead of deep-copying kilobytes per cell: an in-place write anywhere
    now raises instead of silently corrupting the other tier.
    """
    a, b = xy                     # (y, x) -- pass-through, order untouched
    a = np.asarray(a)
    b = np.asarray(b)
    try:
        a.flags.writeable = False
        b.flags.writeable = False
    except ValueError:
        # a view of a still-writeable base cannot be locked; copy then lock
        a = a.copy(); b = b.copy()
        a.flags.writeable = False; b.flags.writeable = False
    return (a, b)


class ACell():
    """
    a cell class

    each cell has attributes:
     id: int
     fov: int
     (there is deliberately NO cell.modality: modality only exists coupled
       to a hybe. The pairs below -- (reference_hybe, reference_modality)
       and (nucleus_hybe, nucleus_modality) -- carry every frame identity a
       cell has. A standalone modality field went stale the moment a
       cytoplasm from the other modality was incorporated, and caused two
       real bugs before removal.)
     _removed_modality_doc: str -- this cell's OWN, original modality
       (the one its container was built under). Deliberately NOT retargeted
       when a cytoplasm from the other modality is attached: it stays the
       cell's stable identity, so cells that never take part in cytoplasmic
       segmentation and cells that do are handled by the exact same code.
       The frame a given mask actually lives in is carried by that mask's
       OWN hybe+modality pair below, never by this field.
     reference_hybe: str -- the hybe `area` is native to. Before any
       cytoplasmic segmentation this is the nucleus/segmentation hybe; after
       it, the hybe the cytoplasm search ran in (which can belong to the
       OTHER modality -- see reference_modality).
     reference_modality: str -- the modality reference_hybe belongs to.
       MUST be used wherever reference_hybe is looked up
       in a (hybe, modality)-keyed structure: a cytoplasm hybe from the other
       modality would otherwise resolve to a key that never exists and
       silently fall back to identity. Which is
       exactly right for every pre-cytoplasm cell.
     nucleus: tuple (ndarray y, ndarray x) -- the NUCLEUS mask, native to
       nucleus_hybe's frame. Before cytoplasmic segmentation this is the
       same mask as `area`; after it, `area` becomes the cytoplasm and this
       keeps the original nucleus. Every cell always has one.
     nucleus_hybe / nucleus_modality: the hybe+modality `nucleus` is native
       to. Cells in one FOV can legitimately disagree here -- a cell that
       opted out of cytoplasmic segmentation keeps its own original hybe --
       which is why any nucleus rendered across cells must be projected
       per-cell rather than assumed to share one frame.
     celltype: str
     area: tuple (ndarray y, ndarray x) -- 2D mask coordinates, rasterized order (convention.py), in
       reference_hybe's frame. Cytoplasm once a cytoplasmic search has been
       incorporated, otherwise the nucleus.
     frame_shape: tuple (height, width) -- full-frame size, needed to bound-check
       coordinates transformed into another hybe's frame (align_cell)
     matrices: dict {(hybe, modality): {'yx': ndarray(3,3), 'dz': float}}
       -- this cell's own composed alignment matrix per (hybe, modality)
       (mirrors how a FOV has /matrix/{hybe}); 'yx' is in-plane, 'dz' is
       the depth correction as a plain scalar (Z alignment is a 1D
       correlation; the old 'zx' 3x3 carried one DOF at [0,2], was never
       composed or inverted, and every reader read that one element).
       Read via chain.entry_dz, which also accepts the legacy 'zx'. Keyed by (hybe, modality), NOT bare hybe --
       the cross-modal "bridge" hybe (e.g. Hyb_130) is a real, distinct
       file in BOTH modalities, so a bare-string key would collide the
       two and silently lose one of them. 'yx' maps that (hybe, modality)
       into THAT COMPUTE_CELL_ALIGNMENT CALL's own reference_hybe frame
       (see compute_cell_alignment's docstring) -- a same-modality entry
       and a cross-modal entry generally rest in two DIFFERENT frames
       (each run's own reference_hybe), never directly comparable without
       bridging through matrix_anchors below. self.area is always native
       to self.reference_hybe's own frame regardless of any of this, so
       relating any hybe's entry back to it (or to any OTHER hybe/
       modality) goes through matrix_to/matrix_between below, never a
       direct dict lookup.
     matrix_anchors: dict {modality: ndarray(3,3)} -- for each modality
       this cell has been aligned in, that call's own reference_hybe's
       transform into fov_matrices' shared frame (what
       compute_cell_alignment computed internally as H_ref_to_shared).
       The bridge matrix_between uses to relate two entries from
       DIFFERENT modalities/runs; same-modality composition never needs
       it (both sides already share the same reference_hybe by
       construction, see matrix_between's own docstring).
     matrix_provenance: dict {(hybe, modality): {'reference_sequence':..., 'steps':...}}
     (there is deliberately NO spots/num_spots/total_num_spots on a cell:
       a cell's spots are a QUERY over the FOV's one spot container --
       SpotContainer.of_cell(fov, cell.id) -- with ASpot.cell as the link.
       A second, cell-held copy is exactly the split store that made
       assignment a cross-structure move and hid assigned spots from
       every control that only knew one home.)
     distmap: ndarray (n_spots x n_spots), see calculate_distmap
    """
    def __init__(self):
        self.id = 0
        self.fov = 0
        self.reference_hybe = ''
        self.reference_modality = ''
        self.nucleus = (np.array([]).reshape(-1, 1), np.array([]).reshape(-1, 1))
        self.nucleus_hybe = ''
        self.nucleus_modality = ''
        self.celltype = ''
        self.area = (np.array([]).reshape(-1, 1), np.array([]).reshape(-1, 1))
        self.frame_shape = (0, 0)
        self.matrices = {}
        self.matrix_anchors = {}
        self.matrix_provenance = {}
        self.distmap = np.array([])
        self.linked = False
        self.linked_at = None

    def set_metadata(self, **kwargs):
        if 'id' in kwargs: self.id = int(kwargs['id'])
        if 'fov' in kwargs: self.fov = int(kwargs['fov'])
        if 'reference_hybe' in kwargs: self.reference_hybe = str(kwargs['reference_hybe'])
        if 'reference_modality' in kwargs: self.reference_modality = str(kwargs['reference_modality'])
        if 'nucleus' in kwargs: self.nucleus = _frozen_xy(kwargs['nucleus'])
        if 'nucleus_hybe' in kwargs: self.nucleus_hybe = str(kwargs['nucleus_hybe'])
        if 'nucleus_modality' in kwargs: self.nucleus_modality = str(kwargs['nucleus_modality'])
        if 'celltype' in kwargs: self.celltype = str(kwargs['celltype'])
        if 'area' in kwargs: self.area = _frozen_xy(kwargs['area'])
        if 'frame_shape' in kwargs: self.frame_shape = tuple(kwargs['frame_shape'])
        if 'matrices' in kwargs: self.matrices = dict(kwargs['matrices'])
        if 'matrix_anchors' in kwargs: self.matrix_anchors = dict(kwargs['matrix_anchors'])
        if 'matrix_provenance' in kwargs: self.matrix_provenance = dict(kwargs['matrix_provenance'])
        if 'distmap' in kwargs: self.distmap = np.array(kwargs['distmap'])
        if 'linked' in kwargs: self.linked = bool(kwargs['linked'])
        if 'linked_at' in kwargs: self.linked_at = kwargs['linked_at']
        self._backfill_frame_defaults()

    def _backfill_frame_defaults(self):
        """
        Fills the frame-identity fields any cell saved BEFORE cytoplasmic
        segmentation existed has no value for. Such a cell has `area` =
        its nucleus, native to reference_hybe, in its own modality -- so
        nucleus mirrors area and both modality fields default to
        reference_modality. Idempotent, and never overwrites a real value, so
        it is safe to run on every set_metadata (which is how a partial
        update -- e.g. set_metadata(celltype=...) -- keeps these
        consistent instead of silently leaving them blank).
        """
        if not self.nucleus_hybe:
            self.nucleus_hybe = self.reference_hybe
        if not self.nucleus_modality:
            self.nucleus_modality = self.reference_modality
        if self.nucleus is None or len(self.nucleus[0]) == 0:
            self.nucleus = self.area

    def has_cytoplasm(self):
        """
        True once a real cytoplasm has been incorporated -- i.e. `area` is
        no longer just a mirror of `nucleus`. Checked via the FRAME
        (reference_hybe/modality vs nucleus_hybe/modality) plus size, never
        by a separate boolean flag that could drift out of sync with the
        masks themselves.
        """
        if (self.reference_hybe, self.reference_modality) != (self.nucleus_hybe, self.nucleus_modality):
            return True
        return len(self.area[0]) != len(self.nucleus[0])

    def light_copy(self):
        """
        Tier-transfer copy: fresh object, scalars copied, coordinate arrays
        SHARED (write-locked, see _frozen_xy), matrix dicts shallow-copied
        (entries are replaced wholesale by the alignment code, never
        mutated in place). Replaces deepcopy in container syncing -- the
        arrays are the bulk of a cell, and they are immutable by
        construction now.
        """
        c = ACell()
        c.id, c.fov = self.id, self.fov
        c.reference_hybe, c.reference_modality = self.reference_hybe, self.reference_modality
        c.nucleus_hybe, c.nucleus_modality = self.nucleus_hybe, self.nucleus_modality
        c.celltype, c.linked, c.linked_at = self.celltype, self.linked, self.linked_at
        c.area, c.nucleus = self.area, self.nucleus          # shared, frozen
        c.frame_shape = self.frame_shape
        c.matrices = {k: dict(v) for k, v in self.matrices.items()}
        c.matrix_anchors = dict(self.matrix_anchors)
        c.matrix_provenance = dict(self.matrix_provenance)
        c.distmap = self.distmap
        return c

    def calculate_distmap(self, spots):
        """
        THE analysis-time computation, per explicit decision: distmap is
        derived data, computed fresh from the current spots
        (SpotContainer.of_cell) wherever analysis or inspection needs it --
        never maintained incrementally, never trusted from an earlier
        save. The persisted field is a legacy slot that stays empty.
        """
        if spots:
            pos = np.array([spot.coordinate for spot in spots])
            self.distmap = ssd.squareform(ssd.pdist(pos))

    def _require_composable(self, key):
        """
        Raise if `key`'s stored 'yx' is a BARE residual.

        A bare residual is only half the transform -- the FOV leg is
        recomposed at read time by frames.FrameResolver, which is the only
        thing holding the current FOV matrices. Composing it here, with no
        access to that layer, would silently drop the FOV correction and
        produce coordinates that look plausible and are wrong.

        Failing loudly instead is deliberate: this exact class of silent
        divergence between two resolvers is what every alignment bug in
        this codebase has been. Callers reaching this error should take a
        resolver (MainWindow._frame_resolver) and use resolver.transform.
        """
        entry = self.matrices.get(key)
        if entry is not None and entry.get('yx_is_residual'):
            raise ValueError(
                f'cell {self.id}: matrices[{key}] is a bare cell-level residual; it cannot be '
                f'composed without the FOV layer. Use frames.FrameResolver.transform '
                f'(MainWindow._frame_resolver) instead of ACell.matrix_* here.')

    def matrix_between(self, hybe_a, modality_a, hybe_b, modality_b):
        """
        hybe_a's native frame -> hybe_b's native frame, using ONLY what this
        cell carries. Valid only for LEGACY entries whose 'yx' already
        contains the FOV leg; raises on bare residuals (see
        _require_composable). New code should go through
        frames.FrameResolver, which composes all three layers.
        """
        # Self-map short-circuit: for a == b the composition cancels
        # EXACTLY -- inv(A @ H) @ (A @ H) -- whatever A and H hold, so the
        # answer is identity for residual-form and legacy entries alike.
        # Short-circuiting BEFORE the composability guard keeps the one
        # projection that never needs the FOV layer (a cell's mask in its
        # own reference frame, e.g. label_mask_for_frame's session-free
        # default) working on residual cells instead of raising.
        if (hybe_a, modality_a) == (hybe_b, modality_b):
            return np.eye(3)
        self._require_composable((hybe_a, modality_a))
        self._require_composable((hybe_b, modality_b))
        H_a_to_anchor = self.matrices.get((hybe_a, modality_a), {}).get('yx', np.eye(3))
        H_b_to_anchor = self.matrices.get((hybe_b, modality_b), {}).get('yx', np.eye(3))
        A_a = self.matrix_anchors.get(modality_a, np.eye(3))
        A_b = self.matrix_anchors.get(modality_b, np.eye(3))
        return la.inv(A_b @ H_b_to_anchor) @ (A_a @ H_a_to_anchor)

    def matrix_to(self, hybe, modality):
        """
        The 3x3 matrix mapping `hybe`'s own native (raw) frame into
        self.reference_hybe's own frame -- the frame self.area is always
        native to. Thin wrapper over matrix_between; see its docstring
        for the general bridging logic and the self-cancellation
        guarantee that keeps get_area_in_readout(self.reference_hybe,
        reference_modality) byte-exact regardless of matrix_anchors' actual
        values.
        """
        return self.matrix_between(hybe, modality, self.reference_hybe, self.reference_modality)

    def matrix_to_shared(self, hybe, modality):
        """
        The 3x3 matrix mapping `hybe`'s own native (raw) frame into the
        pipeline's ONE shared reference frame -- RNA's own same-modality
        reference hybe (e.g. Hyb_101). NOT self.reference_hybe's frame
        (contrast with matrix_to above): self.reference_hybe is an
        arbitrary, per-segmentation-run choice (can even vary cell-group
        to cell-group under append-mode segmentation), whereas the shared
        frame here is the one stable, already-well-defined anchor every
        OTHER layer of this pipeline (FOV alignment, cross-modal
        alignment) is already built around.

        matrix_anchors[modality] already IS that modality's own
        reference_hybe's transform into this shared frame -- for a DNA
        hybe, compute_cell_alignment computed it against fov_matrices
        that was already H_across-composed (see _composed_fov_matrices_
        for_cell_alignment / _other_modality_cell_alignment_inputs), so
        matrix_anchors['DNA'] already lands in RNA's own frame. There is
        no separate "DNA's own shared frame" -- any hybe from either
        modality resolved through this method lands in the exact same
        frame once a cross-modal link exists, no further bridging needed
        (unlike matrix_to/matrix_between, which additionally re-anchor to
        self.reference_hybe).
        """
        self._require_composable((hybe, modality))
        H_to_anchor = self.matrices.get((hybe, modality), {}).get('yx', np.eye(3))
        A = self.matrix_anchors.get(modality, np.eye(3))
        return A @ H_to_anchor

    def get_area_in_readout(self, hybe, modality):
        """
        This cell's mask coordinates (y, x) -- rasterized order, see
        convention.py -- transformed into `hybe`'s own native (raw,
        untransformed) frame via matrix_to(hybe, modality).
        Only the coordinates move -- raw pixel data is never resampled.
        align_cell itself special-cases the MATRIX (skips its own
        resampling machinery when H is genuinely identity, checked once
        there) rather than any caller special-casing the hybe name --
        comparing hybe names instead of the actual math is exactly the
        pattern that kept reintroducing real bugs this session.
        """
        H = self.matrix_to(hybe, modality)
        Hinv = la.inv(H)
        y, x = self.area
        cy, cx = alignment.align_cell((y, x), Hinv, self.frame_shape)
        return (cy, cx)

    def get_nucleus_in_readout(self, hybe, modality):
        """
        This cell's NUCLEUS coordinates (y, x) transformed into `hybe`'s own
        native frame -- the nucleus counterpart of get_area_in_readout, and
        the projection cytoplasmic segmentation needs to render a nucleus
        seed into the cytoplasm hybe's frame.

        Anchors on nucleus_hybe/nucleus_modality, NOT reference_hybe/
        reference_modality: once a cytoplasm is attached those two pairs
        genuinely differ (and can even sit in different modalities), so
        reusing the area path here would project the nucleus out of the
        wrong frame. Same identity-default-on-missing-data contract as
        matrix_between.
        """
        H = self.matrix_between(hybe, modality, self.nucleus_hybe, self.nucleus_modality)
        Hinv = la.inv(H)
        y, x = self.nucleus
        cy, cx = alignment.align_cell((y, x), Hinv, self.frame_shape)
        return (cy, cx)

    def get_mip(self, hybe, storage_path, fov, modality, channel=None, pad=5, use_stack=False):
        """
        Crop this cell's region out of `hybe`'s data -- vlinks.h5's real MIP
        copy by default, or the raw stack file's full Z-stack
        (height,width,depth) if use_stack=True (the 3D exception: MIP-only
        reads never need the raw stack file, per explicit principle).
        channel defaults to that hybe's own fiducial channel if not given.
        """
        y, x = self.get_area_in_readout(hybe, modality)
        if len(x) == 0:
            raise ValueError(f'Cell {self.id} has no area overlapping hybe {hybe}')
        height, width = self.frame_shape
        xmin, xmax = max(0, int(x.min()) - pad), min(width, int(x.max()) + pad + 1)
        ymin, ymax = max(0, int(y.min()) - pad), min(height, int(y.max()) + pad + 1)

        if not use_stack:
            mip = (vlinks_store.fiducial_channel_mip(storage_path, fov, hybe) if channel is None
                   else vlinks_store.read_hybe_mip(storage_path, fov, hybe, channel))
            if mip is None:
                raise ValueError(f'FOV{fov:02d} {hybe} not in vlinks.h5 -- ingest it first.')
            return mip[ymin:ymax, xmin:xmax]

        h5path = os.path.join(storage_path, f'FOV{fov:02d}', f'{hybe}_stack.h5')
        with h5py.File(h5path, 'r') as f:
            if channel is None:
                channel = int(f.attrs['fiducial_channel'])
            return f[f'/stack/ch{channel}'][ymin:ymax, xmin:xmax]

    def save(self):
        return {'id': int(self.id),
                'fov': int(self.fov),
                'reference_hybe': str(self.reference_hybe),
                'reference_modality': str(self.reference_modality),
                # fresh copies, never the live (possibly shared) arrays:
                # nucleus and area can reference the SAME objects (backfill
                # sets nucleus = area), and pickle memoizes shared objects
                # into back-references -- so byte-identical VALUES would
                # fingerprint differently depending on sharing history.
                # Decoupling here makes save() bytes a pure function of
                # values, which the diff-undo convergence depends on.
                'nucleus': (np.array(self.nucleus[0]), np.array(self.nucleus[1])),
                'nucleus_hybe': str(self.nucleus_hybe),
                'nucleus_modality': str(self.nucleus_modality),
                'celltype': str(self.celltype),
                'area': (np.array(self.area[0]), np.array(self.area[1])),
                'frame_shape': tuple(self.frame_shape),
                'matrices': {key: {'yx': np.array(m['yx']), 'dz': alignment.entry_dz(m),
                                   'yx_is_residual': bool(m.get('yx_is_residual', False))}
                            for key, m in self.matrices.items()},
                'matrix_anchors': {modality: np.array(H) for modality, H in self.matrix_anchors.items()},
                'matrix_provenance': dict(self.matrix_provenance),
                # Spots are NOT written here and the cell holds no copy at
                # all: they live in the FOV's spot store on disk and in the
                # session's SpotContainer in memory, linked by ASpot.cell.
                'distmap': np.array(self.distmap),
                'linked': bool(self.linked),
                'linked_at': self.linked_at}
