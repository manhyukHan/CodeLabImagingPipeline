import numpy as np


class AnAllele():
    """
    a chromatin-tracing allele (polymer) class

    each allele has attributes:
     id: int
     fov: int
     cell: int (owning cell id, -1 if homeless -- mirrors ASpot.cell)
     anchor_hybe: str -- the hybe folder the seed spot (the one the user
       selected in Spot Localization to define this allele) was localized
       on. raw_coordinate below is expressed in THIS hybe's own raw frame --
       mirrors ASpot.hybe.
     anchor_channel: int -- the channel the seed spot was localized/selected
       on (mirrors ASpot.channel) -- also the one readout channel traced
       through every other hybe for this allele (see
       localization.localize_chromatin_trace_hybe).
     coordinate: tuple (x, y, z) -- this allele's anchor position in the
       pipeline's ONE shared reference frame (same frame ASpot.coordinate
       lives in) -- the point every hybe's own crop is centered around
       (via spot_mapper.reference_to_raw), never itself re-fit.
     raw_coordinate: tuple (x, y, z) -- that same anchor position in
       anchor_hybe's own native (raw, untransformed) frame, before any
       alignment matrix is applied -- mirrors ASpot.raw_coordinate
     fiducial_trace: dict[hybe (str) -> (x, y, z, amplitude) or None] --
       one shared-frame fiducial-channel fit per hybe this allele was
       processed against (None = no real fiducial peak found in that
       hybe's crop). Used to compute each hybe's own local drift
       correction relative to the configured reference hybe (see
       localization.localize_chromatin_trace_hybe) -- not itself the
       chromatin trace, just the per-hybe anchor the trace is corrected by.
     polymer: dict[hybe (str) -> list of (x, y, z, amplitude) candidate
       tuples] -- every ACCEPTED readout-channel detection near this
       allele's anchor for that hybe, already drift-corrected into the
       shared frame, not just the brightest one (multiple real candidates
       in one hybe -- e.g. sister chromatids -- are kept side by side, never
       pruned against each other; see localization.
       localize_chromatin_trace_hybe's own docstring on why the mixture-
       sibling gates used elsewhere in this pipeline don't apply here)
     rejected_hybes: dict[hybe (str) -> str] -- reason a hybe's readout was
       excluded entirely for this allele (no fiducial peak found in that
       hybe, or its local drift vs. the reference hybe exceeded the
       configured bound) -- same visibility every other rejection gate in
       this pipeline already gives (see chain.compute_cell_alignment).
     final_polymer: ndarray (n_bins x 3) -- one committed (x, y, z) position
       per genomic-locus bin, collapsed from `polymer` by brightest-candidate
       selection (mirrors QualityControlORCA.combineFOV's 'maxBrightness'
       selection in /Users/hanmanhyuk/Remote Server/utils.py); empty until
       computed, matching ACell.distmap's placeholder-until-calculated pattern
    """
    def __init__(self):
        self.id = 0
        self.fov = 0
        self.cell = -1
        self.anchor_hybe = ''
        self.anchor_channel = 0
        self.coordinate = (0.0, 0.0, 0.0)
        self.raw_coordinate = (0.0, 0.0, 0.0)
        self.fiducial_trace = {}
        self.polymer = {}
        self.rejected_hybes = {}
        self.final_polymer = np.array([])
        self.linked = False
        self.linked_at = None

    def set_metadata(self, **kwargs):
        if 'id' in kwargs: self.id = int(kwargs['id'])
        if 'fov' in kwargs: self.fov = int(kwargs['fov'])
        if 'cell' in kwargs: self.cell = int(kwargs['cell'])
        if 'anchor_hybe' in kwargs: self.anchor_hybe = str(kwargs['anchor_hybe'])
        if 'anchor_channel' in kwargs: self.anchor_channel = int(kwargs['anchor_channel'])
        if 'coordinate' in kwargs: self.coordinate = tuple(kwargs['coordinate'])
        if 'raw_coordinate' in kwargs: self.raw_coordinate = tuple(kwargs['raw_coordinate'])
        if 'fiducial_trace' in kwargs:
            self.fiducial_trace = {k: (tuple(v) if v is not None else None)
                                   for k, v in dict(kwargs['fiducial_trace']).items()}
        if 'polymer' in kwargs: self.polymer = {k: list(v) for k, v in dict(kwargs['polymer']).items()}
        if 'rejected_hybes' in kwargs: self.rejected_hybes = dict(kwargs['rejected_hybes'])
        if 'final_polymer' in kwargs: self.final_polymer = np.array(kwargs['final_polymer'])
        if 'linked' in kwargs: self.linked = bool(kwargs['linked'])
        if 'linked_at' in kwargs: self.linked_at = kwargs['linked_at']

    def save(self):
        """
        fiducial_trace/polymer values are rounded to 2 decimal places here
        -- ON THE WAY OUT to disk only, never mutating self's own in-memory
        value (mirrors ASpot.save()'s own r2() convention) -- per explicit
        request, sub-hundredth-pixel precision was never meaningful for a
        real detector pixel anyway.
        """
        def r2(v):
            return round(float(v), 2)
        return {'id': int(self.id),
                'fov': int(self.fov),
                'cell': int(self.cell),
                'anchor_hybe': str(self.anchor_hybe),
                'anchor_channel': int(self.anchor_channel),
                'coordinate': tuple(self.coordinate),
                'raw_coordinate': tuple(self.raw_coordinate),
                'fiducial_trace': {k: (tuple(r2(x) for x in v) if v is not None else None)
                                   for k, v in self.fiducial_trace.items()},
                'polymer': {k: [tuple(r2(x) for x in candidate) for candidate in v]
                           for k, v in self.polymer.items()},
                'rejected_hybes': dict(self.rejected_hybes),
                'final_polymer': np.array(self.final_polymer),
                'linked': bool(self.linked),
                'linked_at': self.linked_at}


def _allele_from_dict(d):
    allele = AnAllele()
    allele.set_metadata(**d)
    return allele
