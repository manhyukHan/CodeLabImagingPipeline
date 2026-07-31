import numpy as np

from .cell import ACell
from .spot import ASpot


class CellContainer():
    """
    Container for ACell objects, keyed by fov. One container per modality --
    segmentation happens per-modality today, each in its own reference_hybe
    (mirrors CellClassifier's CellContainer, but object-based from the start
    rather than array-backed: this is a batch/backend pipeline, not an
    interactive-redraw GUI, so there's no equivalent hot-path performance
    requirement to design around).
    """
    def __init__(self, fov_list, modality=''):
        if len(fov_list) == 0:
            raise ValueError('Make container with positive-length fovs')
        self.fov_list = fov_list
        self.modality = modality
        self.data = {f: [] for f in fov_list}

    def load_new_cells(self, fov, mask, reference_hybe, min_size=0, max_size=np.inf):
        """
        Build ACell objects directly from a Cellpose (or any integer-labeled)
        mask -- reuses whatever produced the mask (segment.py's Cellpose
        call), doesn't reimplement segmentation itself.
        """
        self.data[fov] = []
        ids = np.unique(mask)
        ids = ids[ids > 0]
        for cell_id in ids:
            y, x = np.where(mask == cell_id)
            if len(x) < min_size or len(x) > max_size:
                continue
            cell = ACell()
            cell.set_metadata(id=int(cell_id), fov=int(fov), modality=self.modality,
                              reference_hybe=reference_hybe, area=(x, y), frame_shape=mask.shape)
            self.data[fov].append(cell)

    def get_cell(self, fov, index):
        return self.data[fov][index]

    def get_cells(self, fov):
        return self.data[fov]

    def save(self):
        return {fov: [cell.save() for cell in cells] for fov, cells in self.data.items()}

    @classmethod
    def load(cls, saved, modality=''):
        fov_list = list(saved.keys())
        container = cls(fov_list, modality=modality)
        for fov, cell_dicts in saved.items():
            container.data[fov] = [_cell_from_dict(d) for d in cell_dicts]
        return container


def _spot_from_dict(d):
    spot = ASpot()
    spot.set_metadata(**d)
    return spot


def _cell_from_dict(d):
    cell = ACell()
    kwargs = dict(d)
    kwargs['spots'] = [_spot_from_dict(sd) for sd in d['spots']]
    cell.set_metadata(**kwargs)
    return cell
