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
            container.data[fov] = [_cell_from_dict(d, modality) for d in cell_dicts]
        return container


def _spot_from_dict(d):
    spot = ASpot()
    spot.set_metadata(**d)
    return spot


def _drop_legacy_matrix_keys(mapping):
    """
    Cells saved before cell.matrices/matrix_provenance switched from
    bare-hybe-string keys to (hybe, modality) tuples still have their
    old bare-string keys as-saved. Reloading them as-is leaves the stale
    bare-string entry sitting forever alongside whatever fresh
    tuple-keyed entry a real compute_cell_alignment run later writes for
    the same hybe (nothing ever clears a key format compute_cell_alignment
    itself doesn't write) -- silently inflating any "how many hybes does
    this cell have aligned" count and, worse, breaking any code that
    sorts or compares these keys (tuple vs str -- a real TypeError,
    caught while diagnosing exactly this).

    Dropped, not migrated to (hybe, cell's own modality): a bare legacy
    key's real modality can't be recovered reliably. It was written under
    the OLD scheme, which is exactly the scheme with the now-fixed
    same-name bridge-hybe collision (e.g. Hyb_130 real in both RNA and
    DNA) -- for a bridge hybe there is no way to know, from the key
    alone, which modality's alignment actually survived being overwritten
    by the other, and for a hybe name unique to the OTHER modality,
    guessing "cell's own modality" would be flatly wrong, silently
    mislabeling real cross-modal data as same-modality. Dropping is the
    honest choice: matrices.get(key, identity) already treats a missing
    key as "no correction yet" (the established default everywhere else
    in this codebase), not an error -- and a real re-run of cell-based
    alignment (same-modality + cross-modal) fully repopulates every
    (hybe, modality) entry from scratch anyway, per the "no no-alignment"
    guarantee compute_cell_alignment itself provides.
    """
    return {key: value for key, value in mapping.items() if isinstance(key, tuple)}


def _cell_from_dict(d, container_modality=''):
    """
    container_modality backfills a cell whose own saved 'modality' is
    empty -- older saves persisted '' for every cell (the transient
    container that produced them was itself built with modality='',
    since fixed), so trusting the per-cell field alone would silently
    keep resurrecting that bug for already-saved data.
    """
    cell = ACell()
    kwargs = dict(d)
    if not kwargs.get('modality'):
        kwargs['modality'] = container_modality
    if 'matrices' in kwargs:
        kwargs['matrices'] = _drop_legacy_matrix_keys(kwargs['matrices'])
    if 'matrix_provenance' in kwargs:
        kwargs['matrix_provenance'] = _drop_legacy_matrix_keys(kwargs['matrix_provenance'])
    kwargs['spots'] = [_spot_from_dict(sd) for sd in d['spots']]
    cell.set_metadata(**kwargs)
    return cell
