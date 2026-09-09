Z_ACCEPTED = 'accepted'
Z_REJECTED = 'rejected'
Z_NOT_FIT = 'not_fit'
Z_STATUSES = (Z_ACCEPTED, Z_REJECTED, Z_NOT_FIT)


def z_status_of(spot):
    """A spot's z_status, tolerant of anything that predates the field.

    Old stores have no column and old in-memory objects have no
    attribute; both mean the same thing -- nobody has fitted this spot's
    Z -- so both read as Z_NOT_FIT rather than raising. Use this rather
    than getattr at each site: "no answer" and "not yet fitted" must not
    be allowed to drift apart the way `_z_status` and the persisted
    coordinate already did.
    """
    v = getattr(spot, 'z_status', None)
    return v if v in Z_STATUSES else Z_NOT_FIT


class ASpot():
    """
    a spot class

    each spot has attributes:
     uid: int -- STABLE identity, assigned once at creation from a per-FOV
       counter and never rewritten or reused while the spot lives. Nothing
       else on this object is usable as identity: `cell` changes on
       assignment, `raw_coordinate` changes when 3D localization refines
       the fit, and display numbering is recomputed per view. uid is what
       merge-on-save, undo diffs and staleness marks key on, so it must
       never be derived from any mutable field. 0 means "not yet allocated"
       -- see vlinks_store.allocate_spot_uids.

       There is deliberately no `id`: the old one was a display index that
       got rewritten on every removal, so it could not serve as identity,
       and nothing ever looked a spot up by it. Display numbering lives in
       MainWindow._global_spot_index_map, computed per view.
     fov: int
     modality: str -- which modality this spot's hybe belongs to. Required
       alongside `hybe` because a hybe name alone is ambiguous: the
       cross-modal bridge hybe (e.g. Hyb_130) is a real, distinct
       acquisition in BOTH modalities, so (hybe, modality) is the key
       everywhere else in the pipeline and must be here too. One cell's
       spots can legitimately span modalities, so this cannot be inferred
       from the owning cell either.
     hybe: str (readout identity, e.g. 'Hyb_105' -- distinct from channel)
     channel: int (physical imaging channel, e.g. 555/635)
     cell: int (owning cell id, -1 if homeless)
     celltype: str
     adj_coordinate: tuple (y, x, z) -- rasterized order (convention.py);
       final ADJUSTED position, after H_within/H_across/H_cell composition;
       z is 0.0 for a 2D-only pipeline, but the field always exists.
       (Renamed from `coordinate` -- the adj_ prefix names what it is:
       the alignment-adjusted counterpart of raw_coordinate.)
     raw_coordinate: tuple (y, x, z) -- position in this spot's own hybe's native
       (raw, untransformed) full-frame pixel coordinates, before any alignment
       matrix is applied. Kept alongside adj_coordinate so raw data can be
       re-accessed by locality later without inverting any matrix.
     size: float
     brightness: float
     mixture_centroids: tuple of (Y, X, z, amplitude) -- Y FIRST, matching
      adj_coordinate and the rest of the store (legacy/migrate_store_to_yx.py);
      said "(x, y, z, amplitude)" until 2026-08-27. Set only when this
       spot's own Z was refined via the multi-Gaussian mixture path (see
       localization.refine_spot_z) and the crop held more than one real
       component. Holds EVERY accepted component's own real/shared-frame
       coordinate (same frame as `adj_coordinate`), including this spot's own
       (its entry is whichever one is brightest -- see refine_spot_z), so
       the sibling positions survive even though they no longer spawn
       separate ASpot records. Empty tuple otherwise (single-component fit,
       or Z never refined).
     z_status: str -- 'accepted' | 'rejected' | 'not_fit'. What happened
       when 3D localization last ran on this spot, and PERSISTED, unlike
       the session-only `_z_status` note this replaces. Three states, not
       two, because "the fit rejected this" and "nobody has fitted it"
       are different facts with opposite consequences: a rejected spot is
       a measured negative and should be removable in bulk, while an
       unfitted one is simply unknown and must survive every such sweep.
       Collapsing them -- e.g. inferring "unfitted" from z == 0.0 -- was
       never safe: a real emitter on plane 0 is indistinguishable from a
       placeholder, and manual anchoring writes exactly 0.0.

       Defaults to 'not_fit', which is also what a store written before
       this field existed reads back as (see z_status_of and
       columnar.unpack_spots). Nothing infers it from coordinates.
    """
    def __init__(self):
        self.uid = 0
        self.fov = 0
        self.modality = ''
        self.hybe = ''
        self.channel = 0
        self.cell = -1
        self.celltype = ''
        self.adj_coordinate = (0.0, 0.0, 0.0)
        self.raw_coordinate = (0.0, 0.0, 0.0)
        self.size = 0.0
        self.brightness = 0.0
        self.linked = False
        self.linked_at = None
        self.mixture_centroids = ()
        self.z_status = Z_NOT_FIT
        # NaN = no engine here answered 'is a spot really here'. Only
        # the learned engine (v3-psfmatcher) produces this; v1, v2 and
        # psf-match report a per-engine QUALITY score under a
        # different name and must not be read as a probability. See
        # localization/engine.py's LocalizedSpot docstring.
        self.p_exist = float('nan')

    def set_metadata(self, **kwargs):
        if 'uid' in kwargs: self.uid = int(kwargs['uid'])
        if 'fov' in kwargs: self.fov = int(kwargs['fov'])
        if 'modality' in kwargs: self.modality = str(kwargs['modality'])
        if 'hybe' in kwargs: self.hybe = str(kwargs['hybe'])
        if 'channel' in kwargs: self.channel = int(kwargs['channel'])
        if 'cell' in kwargs: self.cell = int(kwargs['cell'])
        if 'celltype' in kwargs: self.celltype = str(kwargs['celltype'])
        if 'adj_coordinate' in kwargs: self.adj_coordinate = tuple(kwargs['adj_coordinate'])
        elif 'coordinate' in kwargs:
            # v1 stores persisted pickled dicts under the old key; data on
            # disk outlives the API rename, so the alias stays read-only here
            self.adj_coordinate = tuple(kwargs['coordinate'])
        if 'raw_coordinate' in kwargs: self.raw_coordinate = tuple(kwargs['raw_coordinate'])
        if 'size' in kwargs: self.size = float(kwargs['size'])
        if 'brightness' in kwargs: self.brightness = float(kwargs['brightness'])
        if 'linked' in kwargs: self.linked = bool(kwargs['linked'])
        if 'linked_at' in kwargs: self.linked_at = kwargs['linked_at']
        if 'mixture_centroids' in kwargs: self.mixture_centroids = tuple(kwargs['mixture_centroids'])
        if 'p_exist' in kwargs:
            try:
                v = kwargs['p_exist']
                self.p_exist = float('nan') if v is None else float(v)
            except (TypeError, ValueError):
                # Crosses a store boundary, so anything unreadable is
                # 'nobody answered' rather than a crash -- the same
                # rule z_status follows two lines below.
                self.p_exist = float('nan')
        if 'z_status' in kwargs:
            v = str(kwargs['z_status'])
            # An unrecognised value is 'nobody has fitted it', never a
            # crash: this field crosses a store boundary, and a store
            # written by a newer build must stay readable by an older one.
            self.z_status = v if v in Z_STATUSES else Z_NOT_FIT

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
        return {'uid': int(self.uid),
                'fov': int(self.fov),
                'modality': str(self.modality),
                'hybe': str(self.hybe),
                'channel': int(self.channel),
                'cell': int(self.cell),
                'celltype': str(self.celltype),
                'adj_coordinate': tuple(r2(v) for v in self.adj_coordinate),
                'raw_coordinate': tuple(r2(v) for v in self.raw_coordinate),
                'size': r2(self.size),
                'brightness': r2(self.brightness),
                'linked': bool(self.linked),
                'linked_at': self.linked_at,
                'mixture_centroids': tuple(tuple(r2(v) for v in c) for c in self.mixture_centroids),
                # SIX DECIMALS, NOT TWO. Every other float here is a
                # pixel coordinate where a hundredth is already past
                # the detector's precision; this one spans 1e-12 to
                # 1-1e-12 and its mass sits at the BOTTOM -- rounding
                # to 2dp would collapse every spot under 0.005 onto
                # exactly 0.0 and erase the tail the p-gate histogram
                # is drawn from.
                'p_exist': round(float(self.p_exist), 6),
                'z_status': z_status_of(self)}
