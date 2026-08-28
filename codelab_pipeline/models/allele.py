import numpy as np


def _tuples(d):
    """{hybe -> (y,x,z,amp) or None}, copied and normalised."""
    return {k: (tuple(v) if v is not None else None)
            for k, v in dict(d or {}).items()}


def _lists(d):
    """{hybe -> [candidate tuples]}, copied and normalised."""
    return {k: list(v) for k, v in dict(d or {}).items()}


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
     coordinate: tuple (y, x, z) -- rasterized order (convention.py); anchor position in the
       pipeline's ONE shared reference frame (same frame ASpot.adj_coordinate
       lives in) -- the point every hybe's own crop is centered around
       (via spot_mapper.reference_to_raw), never itself re-fit.
     raw_coordinate: tuple (y, x, z) -- that same anchor position in
       anchor_hybe's own native (raw, untransformed) frame, before any
       alignment matrix is applied -- mirrors ASpot.raw_coordinate
     THE raw/adj PAIRING, AND THE ONE ASYMMETRY IN IT.

       Every per-hybe position is stored twice, mirroring
       ASpot.raw_coordinate/adj_coordinate: once in that hybe's own
       native frame, once adjusted. `_raw` always means the same thing --
       the fit as measured, in that hybe's untransformed full-frame
       pixels, so the image data can be re-reached without inverting a
       matrix. `_adj` does NOT mean the same thing in both dicts, and the
       difference is the whole point:

         fiducial_trace_adj - fiducial_trace_raw
             = alignment only (FOV + cross-modal + cell residual)
         polymer_adj        - polymer_raw
             = alignment  PLUS  the fiducial drift correction

       A fiducial is not corrected by itself; it IS the correction. So
       the readout carries one more term than the fiducial does. Nothing
       is lost by not storing the intermediate: the shared-but-
       uncorrected readout is recoverable as polymer_adj minus
       (fiducial_trace_adj[reference] - fiducial_trace_adj[hybe]), and
       both fiducials are kept.

     fiducial_trace_raw: dict[hybe (str) -> (y, x, z, amplitude) or None]
       -- the fiducial fit in THAT hybe's own raw frame.
     fiducial_trace_adj: dict[hybe (str) -> (y, x, z, amplitude) or None]
       -- the same fit in the pipeline's ONE shared reference frame
       (None = no real fiducial peak found in that hybe's crop). Used to
       compute each hybe's own local drift correction relative to the
       configured reference hybe -- not itself the chromatin trace, just
       the per-hybe anchor the trace is corrected by. Carries NO fiducial
       correction of its own.
     polymer_raw: dict[hybe (str) -> list of (y, x, z, amplitude)] -- the
       accepted readout detections in that hybe's own raw frame,
       uncorrected.
     polymer_adj: dict[hybe (str) -> list of (y, x, z, amplitude)] -- THE
       FINAL per-hybe positions: shared frame AND fiducial-drift
       corrected. Every ACCEPTED readout-channel detection near this
       allele's anchor for that hybe, not just the brightest one
       (multiple real candidates in one hybe -- e.g. sister chromatids --
       are kept side by side, never pruned against each other; see
       localization.localize_chromatin_trace_hybe's own docstring on why
       the mixture-sibling gates used elsewhere in this pipeline don't
       apply here)
     rejected_hybes: dict[hybe (str) -> str] -- reason a hybe's readout was
       excluded entirely for this allele (no fiducial peak found in that
       hybe, or its local drift vs. the reference hybe exceeded the
       configured bound) -- same visibility every other rejection gate in
       this pipeline already gives (see chain.compute_cell_alignment).
     final_polymer: ndarray (n_bins x 3) -- one committed (y, x, z) position
       per genomic-locus bin, collapsed from `polymer_adj` by brightest-candidate
       selection (mirrors QualityControlORCA.combineFOV's 'maxBrightness'
       selection in /Users/hanmanhyuk/Remote Server/utils.py); empty until
       computed, matching ACell.distmap's placeholder-until-calculated pattern
    """
    def __init__(self):
        self.id = 0
        self.fov = 0
        self.cell = -1
        # uid of the ASpot this allele was built from (0 = unknown/legacy).
        # The anchor coordinate fields below are a snapshot taken at Build
        # time; the uid is what lets the session REFRESH them from the
        # spot's current position (a later 3D refinement moves the spot --
        # confirmed real divergence) instead of drifting apart silently.
        self.anchor_uid = 0
        self.anchor_hybe = ''
        self.anchor_channel = 0
        # HOW this allele's trace was produced. A FREE-FORM dict, on
        # purpose: the engine that fills it decides what belongs in it, so
        # v1, v2 and whatever comes next can each record their own inputs
        # without this class growing a field per engine. Empty means "no
        # trace, or a trace made before provenance was recorded".
        #
        # It exists because nothing on disk said which engine produced a
        # polymer. v1 and v2 differ by 43-68% in localization error, so a
        # store mixing them is mixing two accuracy regimes with no way to
        # tell which alleles are which, or to re-fit only the old ones.
        # File mtime cannot answer it either: an append run rewrites the
        # whole FOV capsule, so every allele in it looks equally fresh.
        self.provenance = {}
        self.coordinate = (0.0, 0.0, 0.0)
        self.raw_coordinate = (0.0, 0.0, 0.0)
        self.fiducial_trace_adj = {}
        self.fiducial_trace_raw = {}
        self.polymer_adj = {}
        self.polymer_raw = {}
        self.rejected_hybes = {}
        self.final_polymer = np.array([])
        self.linked = False
        self.linked_at = None

    def set_metadata(self, **kwargs):
        if 'id' in kwargs: self.id = int(kwargs['id'])
        if 'fov' in kwargs: self.fov = int(kwargs['fov'])
        if 'cell' in kwargs: self.cell = int(kwargs['cell'])
        if 'anchor_uid' in kwargs: self.anchor_uid = int(kwargs['anchor_uid'])
        if 'anchor_hybe' in kwargs: self.anchor_hybe = str(kwargs['anchor_hybe'])
        if 'anchor_channel' in kwargs: self.anchor_channel = int(kwargs['anchor_channel'])
        if 'provenance' in kwargs:
            self.provenance = dict(kwargs['provenance'] or {})
        if 'coordinate' in kwargs: self.coordinate = tuple(kwargs['coordinate'])
        if 'raw_coordinate' in kwargs: self.raw_coordinate = tuple(kwargs['raw_coordinate'])
        # _adj first, then the legacy unsuffixed key as a READ-ONLY alias:
        # v1 stores on disk carry 'fiducial_trace'/'polymer' and outlive
        # the API rename, exactly as ASpot.adj_coordinate keeps reading
        # 'coordinate'. A store written before the raw fields existed
        # simply has no _raw, which stays {} and is not an error.
        if 'fiducial_trace_adj' in kwargs:
            self.fiducial_trace_adj = _tuples(kwargs['fiducial_trace_adj'])
        elif 'fiducial_trace' in kwargs:
            self.fiducial_trace_adj = _tuples(kwargs['fiducial_trace'])
        if 'fiducial_trace_raw' in kwargs:
            self.fiducial_trace_raw = _tuples(kwargs['fiducial_trace_raw'])
        if 'polymer_adj' in kwargs:
            self.polymer_adj = _lists(kwargs['polymer_adj'])
        elif 'polymer' in kwargs:
            self.polymer_adj = _lists(kwargs['polymer'])
        if 'polymer_raw' in kwargs:
            self.polymer_raw = _lists(kwargs['polymer_raw'])
        if 'rejected_hybes' in kwargs: self.rejected_hybes = dict(kwargs['rejected_hybes'])
        if 'final_polymer' in kwargs: self.final_polymer = np.array(kwargs['final_polymer'])
        if 'linked' in kwargs: self.linked = bool(kwargs['linked'])
        if 'linked_at' in kwargs: self.linked_at = kwargs['linked_at']

    def save(self):
        """
        fiducial_trace_*/polymer_* values are rounded to 2 decimal places here
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
                'anchor_uid': int(self.anchor_uid),
                'anchor_hybe': str(self.anchor_hybe),
                'anchor_channel': int(self.anchor_channel),
                'coordinate': tuple(self.coordinate),
                'raw_coordinate': tuple(self.raw_coordinate),
                'fiducial_trace_adj': {k: (tuple(r2(x) for x in v) if v is not None else None)
                                       for k, v in self.fiducial_trace_adj.items()},
                'fiducial_trace_raw': {k: (tuple(r2(x) for x in v) if v is not None else None)
                                       for k, v in self.fiducial_trace_raw.items()},
                'polymer_adj': {k: [tuple(r2(x) for x in candidate) for candidate in v]
                                for k, v in self.polymer_adj.items()},
                'polymer_raw': {k: [tuple(r2(x) for x in candidate) for candidate in v]
                                for k, v in self.polymer_raw.items()},
                'rejected_hybes': dict(self.rejected_hybes),
                'final_polymer': np.array(self.final_polymer),
                'provenance': dict(self.provenance or {}),
                'linked': bool(self.linked),
                'linked_at': self.linked_at}


