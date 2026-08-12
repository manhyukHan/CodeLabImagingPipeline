class ASpot():
    """
    a spot class

    each spot has attributes:
     id: int
     fov: int
     hybe: str (readout identity, e.g. 'Hyb_105' -- distinct from channel)
     channel: int (physical imaging channel, e.g. 555/635)
     cell: int (owning cell id, -1 if homeless)
     celltype: str
     coordinate: tuple (x, y, z) -- final position, after H_within/H_across/H_cell
       composition; z is 0.0 for a 2D-only pipeline, but the field always exists
     raw_coordinate: tuple (x, y, z) -- position in this spot's own hybe's native
       (raw, untransformed) full-frame pixel coordinates, before any alignment
       matrix is applied. Kept alongside coordinate so raw data can be
       re-accessed by locality later without inverting any matrix.
     size: float
     brightness: float
     neighbors: tuple (variable size)
     mixture_centroids: tuple of (x, y, z, amplitude) -- set only when this
       spot's own Z was refined via the multi-Gaussian mixture path (see
       localization.refine_spot_z) and the crop held more than one real
       component. Holds EVERY accepted component's own real/shared-frame
       coordinate (same frame as `coordinate`), including this spot's own
       (its entry is whichever one is brightest -- see refine_spot_z), so
       the sibling positions survive even though they no longer spawn
       separate ASpot records. Empty tuple otherwise (single-component fit,
       or Z never refined).
    """
    def __init__(self):
        self.id = 0
        self.fov = 0
        self.hybe = ''
        self.channel = 0
        self.cell = -1
        self.celltype = ''
        self.coordinate = (0.0, 0.0, 0.0)
        self.raw_coordinate = (0.0, 0.0, 0.0)
        self.size = 0.0
        self.brightness = 0.0
        self.neighbors = ()
        self.linked = False
        self.linked_at = None
        self.mixture_centroids = ()

    def set_metadata(self, **kwargs):
        if 'id' in kwargs: self.id = int(kwargs['id'])
        if 'fov' in kwargs: self.fov = int(kwargs['fov'])
        if 'hybe' in kwargs: self.hybe = str(kwargs['hybe'])
        if 'channel' in kwargs: self.channel = int(kwargs['channel'])
        if 'cell' in kwargs: self.cell = int(kwargs['cell'])
        if 'celltype' in kwargs: self.celltype = str(kwargs['celltype'])
        if 'coordinate' in kwargs: self.coordinate = tuple(kwargs['coordinate'])
        if 'raw_coordinate' in kwargs: self.raw_coordinate = tuple(kwargs['raw_coordinate'])
        if 'size' in kwargs: self.size = float(kwargs['size'])
        if 'brightness' in kwargs: self.brightness = float(kwargs['brightness'])
        if 'neighbors' in kwargs: self.neighbors = tuple(kwargs['neighbors'])
        if 'linked' in kwargs: self.linked = bool(kwargs['linked'])
        if 'linked_at' in kwargs: self.linked_at = kwargs['linked_at']
        if 'mixture_centroids' in kwargs: self.mixture_centroids = tuple(kwargs['mixture_centroids'])

    def save(self):
        """
        Per explicit request, every float field is rounded to 2 decimal
        places here -- ON THE WAY OUT to disk only, never mutating self's
        own in-memory value (a caller doing further coordinate math this
        same session, e.g. another localization.refine_spot_z pass,
        still works from full precision; only the PERSISTED copy is
        rounded). Sub-hundredth-pixel precision was never meaningful for
        a real detector pixel anyway -- this just stops it from bloating
        every displayed/persisted coordinate with 15+ meaningless digits
        (e.g. 701.0033910046826).
        """
        def r2(v):
            return round(float(v), 2)
        return {'id': int(self.id),
                'fov': int(self.fov),
                'hybe': str(self.hybe),
                'channel': int(self.channel),
                'cell': int(self.cell),
                'celltype': str(self.celltype),
                'coordinate': tuple(r2(v) for v in self.coordinate),
                'raw_coordinate': tuple(r2(v) for v in self.raw_coordinate),
                'size': r2(self.size),
                'brightness': r2(self.brightness),
                'neighbors': tuple(self.neighbors),
                'linked': bool(self.linked),
                'linked_at': self.linked_at,
                'mixture_centroids': tuple(tuple(r2(v) for v in c) for c in self.mixture_centroids)}
