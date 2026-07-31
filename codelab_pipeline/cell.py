import os
import numpy as np
import numpy.linalg as la
import h5py
import scipy.spatial.distance as ssd

from . import alignment


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
     matrices: dict {hybe: {'yx': ndarray(3,3), 'zx': ndarray(3,3)}} -- this
       cell's own composed alignment matrix per hybe (mirrors how a FOV has
       /matrix/{hybe}); 'yx' is in-plane, 'zx' is the depth correction
     matrix_provenance: dict {hybe: {'reference_sequence':..., 'steps':...}}
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

    def get_area_in_readout(self, hybe):
        """
        This cell's mask coordinates (x, y), transformed into `hybe`'s own
        native (raw, untransformed) frame via this cell's composed 'yx'
        matrix for that hybe. Only the coordinates move -- raw pixel data
        is never resampled. Returns self.area unchanged if hybe is this
        cell's own reference_hybe.
        """
        if hybe == self.reference_hybe:
            return self.area
        if hybe not in self.matrices:
            raise KeyError(f'No alignment matrix for hybe {hybe} on cell {self.id} -- '
                           f'run compute_cell_alignment for this hybe first')
        H = self.matrices[hybe]['yx']
        Hinv = la.inv(H)
        x, y = self.area
        cy, cx = alignment.align_cell((y, x), Hinv, self.frame_shape)
        return (cx, cy)

    def get_mip(self, hybe, storage_path, fov, channel=None, pad=5, use_stack=False):
        """
        Crop this cell's region directly out of `hybe`'s raw H5 data --
        /mip/ch{channel} by default, or /stack/ch{channel} (full Z-stack,
        (height,width,depth)) if use_stack=True. channel defaults to that
        hybe's own fiducial channel if not given.
        """
        x, y = self.get_area_in_readout(hybe)
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
                'matrices': {hybe: {'yx': np.array(m['yx']), 'zx': np.array(m['zx'])}
                            for hybe, m in self.matrices.items()},
                'matrix_provenance': dict(self.matrix_provenance),
                'spots': [spot.save() for spot in self.spots],
                'total_num_spots': int(self.total_num_spots),
                'num_spots': dict(self.num_spots),
                'distmap': np.array(self.distmap),
                'linked': bool(self.linked),
                'linked_at': self.linked_at}
