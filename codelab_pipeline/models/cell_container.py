from copy import deepcopy

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

    def load_new_cells(self, fov, mask, reference_hybe, min_size=0, max_size=np.inf, preserve_existing=False,
                       reference_modality=None):
        """
        Build ACell objects directly from a Cellpose (or any integer-labeled)
        mask -- reuses whatever produced the mask (segment.py's Cellpose
        call), doesn't reimplement segmentation itself.

        preserve_existing=False (default -- a genuinely fresh/"start over"
        segmentation, mask is a brand-new clustering with NO reliable
        relationship to whatever ids self.data[fov] held before this call:
        Cellpose/watershed always numbers a fresh run starting from 1, so
        matching by id here would risk silently attaching an OLD cell's
        real reference_hybe/matrices/spots to a completely different
        physical cell that just happens to get the same number this run):
        every resulting cell is brand-new, reference_hybe=(this call's own
        parameter), no history -- today's original, unchanged behavior.

        preserve_existing=True (Append Mode's own merged mask, or a manual
        add/remove edit in the displayer -- both cases where mask's own
        existing ids ARE guaranteed to be the SAME physical cells as
        before: _merge_append_mask explicitly keeps an old cell's id
        wherever its own pixels weren't repainted, and a manual edit only
        ever adds one new label or removes one existing label from the
        mask the user is already looking at): per explicit request,
        reference_hybe is fixed at a cell's own DEFINITION time, exactly
        like its own mask coordinates, and never retroactively touched by
        a LATER call regardless of manual/auto/removal mode -- so an id
        that already existed in self.data[fov] keeps everything about
        itself (reference_hybe, celltype, matrices, matrix_anchors,
        matrix_provenance, spots, num_spots, total_num_spots, distmap,
        linked, linked_at). Only a genuinely new id (not already present)
        gets `reference_hybe` (this call's own parameter) and starts blank.

        area/frame_shape refresh from `mask` ONLY when `reference_hybe`
        (this call's own parameter -- whatever hybe the mask being edited/
        segmented is actually expressed in) matches the existing cell's
        OWN reference_hybe. Confirmed real bug otherwise: the mask a user
        edits/appends onto is frequently a DIFFERENT hybe's own displayed
        frame than an existing cell's reference_hybe (ACell.
        get_area_in_readout explicitly projects a cell's area into
        whatever hybe the panel currently shows, precisely so cells stay
        visible while browsing a different hybe -- see MainWindow.
        _ensure_cell_displayer_initialized) -- area is documented as
        always native to reference_hybe's OWN frame (see ACell's own
        docstring), so overwriting it from a mask expressed in some OTHER
        hybe's frame while keeping the OLD reference_hybe label would
        silently mismatch the two. When the hybes differ, area/frame_shape
        stay exactly as they were too -- this call's own mask carries no
        real information about that cell's true boundary in its OWN
        frame without a genuine cross-hybe transform, which a raw pixel
        mask can't provide without resampling.
        """
        existing = {c.id: c for c in self.data.get(fov, [])} if preserve_existing else {}
        self.data[fov] = []
        ids = np.unique(mask)
        ids = ids[ids > 0]
        cell_modality = reference_modality or self.modality
        for cell_id in ids:
            y, x = np.where(mask == cell_id)
            if len(x) < min_size or len(x) > max_size:
                continue
            old = existing.get(int(cell_id))
            cell = ACell()
            if old is not None:
                same_frame = (old.reference_hybe == reference_hybe)
                # Nucleus bookkeeping is carried over ONLY for a cell that
                # actually has a cytoplasm. For a plain nucleus-only cell,
                # `area` IS the nucleus, so the fields are deliberately left
                # unset and ACell._backfill_frame_defaults mirrors them onto
                # whatever area this run just produced -- passing old.nucleus
                # there would freeze the nucleus at the PREVIOUS mask while
                # area moved to the new one, silently desyncing the two.
                nucleus_kwargs = ({'nucleus': old.nucleus, 'nucleus_hybe': old.nucleus_hybe,
                                   'nucleus_modality': old.nucleus_modality}
                                  if old.has_cytoplasm() else {})
                cell.set_metadata(id=int(cell_id), fov=int(fov), modality=old.modality or cell_modality,
                                  reference_hybe=old.reference_hybe, celltype=old.celltype,
                                  reference_modality=old.reference_modality,
                                  **nucleus_kwargs,
                                  area=(x, y) if same_frame else old.area,
                                  frame_shape=mask.shape if same_frame else old.frame_shape,
                                  matrices=deepcopy(old.matrices), matrix_anchors=deepcopy(old.matrix_anchors),
                                  matrix_provenance=deepcopy(old.matrix_provenance), spots=deepcopy(old.spots),
                                  total_num_spots=old.total_num_spots, num_spots=deepcopy(old.num_spots),
                                  distmap=deepcopy(old.distmap), linked=old.linked, linked_at=old.linked_at)
            else:
                # modality comes from the REFERENCE HYBE, not from the
                # container's own label. Confirmed real bug: the container
                # carries whatever modality the program happened to be on,
                # so segmenting on a DNA hybe while the app sat on RNA
                # produced cells whose modality contradicted their own
                # reference_hybe -- and cell alignment then built passes
                # for the wrong modality and wrote nothing at all.
                cell.set_metadata(id=int(cell_id), fov=int(fov), modality=cell_modality,
                                  reference_hybe=reference_hybe, reference_modality=cell_modality,
                                  nucleus_hybe=reference_hybe, nucleus_modality=cell_modality,
                                  area=(x, y), frame_shape=mask.shape)
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
        # The cells are authoritative, not the caller's label: write_cells
        # stores container.modality in the payload, so a container loaded
        # under the program's CURRENT modality would otherwise re-save
        # cells under a modality that contradicts their own.
        loaded_modalities = {c.modality for cells in container.data.values() for c in cells if c.modality}
        if len(loaded_modalities) == 1:
            container.modality = loaded_modalities.pop()
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
