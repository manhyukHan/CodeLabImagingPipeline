import os
import numpy as np
import numpy.linalg as la
import h5py
import scipy.spatial.distance as ssd

from ..alignment import chain as alignment


class ACell():
    """
    a cell class

    each cell has attributes:
     id: int
     fov: int
     modality: str ('RNA'/'DNA')
     reference_hybe: str -- the hybe the cell mask was segmented in; area is
       always in this hybe's frame
     celltype: str
     area: tuple (ndarray x, ndarray y) -- 2D mask coordinates, in reference_hybe's frame
     frame_shape: tuple (height, width) -- full-frame size, needed to bound-check
       coordinates transformed into another hybe's frame (align_cell)
     matrices: dict {(hybe, modality): {'yx': ndarray(3,3), 'zx': ndarray(3,3)}}
       -- this cell's own composed alignment matrix per (hybe, modality)
       (mirrors how a FOV has /matrix/{hybe}); 'yx' is in-plane, 'zx' is
       the depth correction. Keyed by (hybe, modality), NOT bare hybe --
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
     spots: list [ASpot, ...]
     total_num_spots: int
     num_spots: dict {hybe: int}
     distmap: ndarray (num_spots x num_spots)
    """
    def __init__(self):
        self.id = 0
        self.fov = 0
        self.modality = ''
        self.reference_hybe = ''
        self.celltype = ''
        self.area = (np.array([]).reshape(-1, 1), np.array([]).reshape(-1, 1))
        self.frame_shape = (0, 0)
        self.matrices = {}
        self.matrix_anchors = {}
        self.matrix_provenance = {}
        self.spots = []
        self.total_num_spots = 0
        self.num_spots = {}
        self.distmap = np.array([])
        self.linked = False
        self.linked_at = None

    def set_metadata(self, **kwargs):
        if 'id' in kwargs: self.id = int(kwargs['id'])
        if 'fov' in kwargs: self.fov = int(kwargs['fov'])
        if 'modality' in kwargs: self.modality = str(kwargs['modality'])
        if 'reference_hybe' in kwargs: self.reference_hybe = str(kwargs['reference_hybe'])
        if 'celltype' in kwargs: self.celltype = str(kwargs['celltype'])
        if 'area' in kwargs: self.area = tuple(kwargs['area'])
        if 'frame_shape' in kwargs: self.frame_shape = tuple(kwargs['frame_shape'])
        if 'matrices' in kwargs: self.matrices = dict(kwargs['matrices'])
        if 'matrix_anchors' in kwargs: self.matrix_anchors = dict(kwargs['matrix_anchors'])
        if 'matrix_provenance' in kwargs: self.matrix_provenance = dict(kwargs['matrix_provenance'])
        if 'spots' in kwargs: self.spots = list(kwargs['spots'])
        if 'total_num_spots' in kwargs: self.total_num_spots = int(kwargs['total_num_spots'])
        if 'num_spots' in kwargs: self.num_spots = dict(kwargs['num_spots'])
        if 'distmap' in kwargs: self.distmap = np.array(kwargs['distmap'])
        if 'linked' in kwargs: self.linked = bool(kwargs['linked'])
        if 'linked_at' in kwargs: self.linked_at = kwargs['linked_at']

    def calculate_distmap(self):
        if self.total_num_spots > 0:
            pos = np.array([spot.coordinate for spot in self.spots])
            self.distmap = ssd.squareform(ssd.pdist(pos))

    def matrix_between(self, hybe_a, modality_a, hybe_b, modality_b):
        """
        The 3x3 matrix mapping hybe_a's own native (raw) frame into
        hybe_b's own native (raw) frame -- the general primitive every
        other cross-hybe/cross-modality lookup on this cell reduces to
        (matrix_to(hybe, modality) is just
        matrix_between(hybe, modality, self.reference_hybe, self.modality)).

        cell.matrices[(h, m)] targets THAT compute_cell_alignment call's
        own reference_hybe, not a shared frame (see that function's
        docstring) -- so entries from the SAME modality/run already share
        a frame and compose directly, while entries from DIFFERENT
        modalities/runs need bridging through matrix_anchors (each
        modality's own reference_hybe -> the shared FOV frame both
        anchors target) to relate at all. Written as one formula rather
        than an if/else on modality_a == modality_b: when they're equal,
        matrix_anchors[modality_a] and matrix_anchors[modality_b] are the
        SAME lookup and cancel out algebraically on their own, giving
        exactly the direct (bridge-free) composition -- no special case
        needed, matching the "generalize the matrix application" principle
        (check what the matrices actually are, don't branch on names).

        Missing data (this cell's alignment was never run for a given
        hybe/modality, or matrix_anchors lacks an entry) defaults to
        identity for that piece specifically -- "no correction known yet",
        never an error, per the same "no no-alignment" principle used
        throughout compute_cell_alignment itself.
        """
        H_a_to_anchor = self.matrices.get((hybe_a, modality_a), {}).get('yx', np.eye(3))
        H_b_to_anchor = self.matrices.get((hybe_b, modality_b), {}).get('yx', np.eye(3))
        A_a = self.matrix_anchors.get(modality_a, np.eye(3))
        A_b = self.matrix_anchors.get(modality_b, np.eye(3))
        H_a_to_shared = A_a @ H_a_to_anchor
        H_b_to_shared = A_b @ H_b_to_anchor
        return la.inv(H_b_to_shared) @ H_a_to_shared

    def matrix_to(self, hybe, modality):
        """
        The 3x3 matrix mapping `hybe`'s own native (raw) frame into
        self.reference_hybe's own frame -- the frame self.area is always
        native to. Thin wrapper over matrix_between; see its docstring
        for the general bridging logic and the self-cancellation
        guarantee that keeps get_area_in_readout(self.reference_hybe,
        self.modality) byte-exact regardless of matrix_anchors' actual
        values.
        """
        return self.matrix_between(hybe, modality, self.reference_hybe, self.modality)

    def get_area_in_readout(self, hybe, modality):
        """
        This cell's mask coordinates (x, y), transformed into `hybe`'s own
        native (raw, untransformed) frame via matrix_to(hybe, modality).
        Only the coordinates move -- raw pixel data is never resampled.
        align_cell itself special-cases the MATRIX (skips its own
        resampling machinery when H is genuinely identity, checked once
        there) rather than any caller special-casing the hybe name --
        comparing hybe names instead of the actual math is exactly the
        pattern that kept reintroducing real bugs this session.
        """
        H = self.matrix_to(hybe, modality)
        Hinv = la.inv(H)
        x, y = self.area
        cy, cx = alignment.align_cell((y, x), Hinv, self.frame_shape)
        return (cx, cy)

    def get_mip(self, hybe, storage_path, fov, modality, channel=None, pad=5, use_stack=False):
        """
        Crop this cell's region directly out of `hybe`'s raw H5 data --
        /mip/ch{channel} by default, or /stack/ch{channel} (full Z-stack,
        (height,width,depth)) if use_stack=True. channel defaults to that
        hybe's own fiducial channel if not given.
        """
        x, y = self.get_area_in_readout(hybe, modality)
        if len(x) == 0:
            raise ValueError(f'Cell {self.id} has no area overlapping hybe {hybe}')
        height, width = self.frame_shape
        xmin, xmax = max(0, int(x.min()) - pad), min(width, int(x.max()) + pad + 1)
        ymin, ymax = max(0, int(y.min()) - pad), min(height, int(y.max()) + pad + 1)

        h5path = os.path.join(storage_path, f'FOV{fov:02d}', f'{hybe}_stack.h5')
        with h5py.File(h5path, 'r') as f:
            if channel is None:
                channel = int(f.attrs['fiducial_channel'])
            dataset = f'/stack/ch{channel}' if use_stack else f'/mip/ch{channel}'
            return f[dataset][ymin:ymax, xmin:xmax]

    def save(self):
        return {'id': int(self.id),
                'fov': int(self.fov),
                'modality': str(self.modality),
                'reference_hybe': str(self.reference_hybe),
                'celltype': str(self.celltype),
                'area': tuple(self.area),
                'frame_shape': tuple(self.frame_shape),
                'matrices': {key: {'yx': np.array(m['yx']), 'zx': np.array(m['zx'])}
                            for key, m in self.matrices.items()},
                'matrix_anchors': {modality: np.array(H) for modality, H in self.matrix_anchors.items()},
                'matrix_provenance': dict(self.matrix_provenance),
                'spots': [spot.save() for spot in self.spots],
                'total_num_spots': int(self.total_num_spots),
                'num_spots': dict(self.num_spots),
                'distmap': np.array(self.distmap),
                'linked': bool(self.linked),
                'linked_at': self.linked_at}
