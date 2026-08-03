import numpy as np


class AnAllele():
    """
    a chromatin-tracing allele (polymer) class

    each allele has attributes:
     id: int
     fov: int
     cell: int (owning cell id, -1 if homeless -- mirrors ASpot.cell)
     coordinate: tuple (x, y, z) -- final, image-referenced fiducial position
       for this allele (the anchor every readout's polymer candidates are
       gathered around)
     raw_coordinate: tuple (x, y, z) -- this allele's fiducial position in its
       own native (raw, untransformed) reference frame, before any alignment
       matrix is applied -- mirrors ASpot.raw_coordinate
     polymer: dict[readout (int) -> list of (x, y, z, h) candidate tuples] --
       every candidate detection near this allele's fiducial for that readout,
       not just the brightest one (multi-candidate loci are real, not an edge
       case -- keep all of them so downstream code can choose how to collapse)
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
        self.coordinate = (0.0, 0.0, 0.0)
        self.raw_coordinate = (0.0, 0.0, 0.0)
        self.polymer = {}
        self.final_polymer = np.array([])
        self.linked = False
        self.linked_at = None

    def set_metadata(self, **kwargs):
        if 'id' in kwargs: self.id = int(kwargs['id'])
        if 'fov' in kwargs: self.fov = int(kwargs['fov'])
        if 'cell' in kwargs: self.cell = int(kwargs['cell'])
        if 'coordinate' in kwargs: self.coordinate = tuple(kwargs['coordinate'])
        if 'raw_coordinate' in kwargs: self.raw_coordinate = tuple(kwargs['raw_coordinate'])
        if 'polymer' in kwargs: self.polymer = {k: list(v) for k, v in dict(kwargs['polymer']).items()}
        if 'final_polymer' in kwargs: self.final_polymer = np.array(kwargs['final_polymer'])
        if 'linked' in kwargs: self.linked = bool(kwargs['linked'])
        if 'linked_at' in kwargs: self.linked_at = kwargs['linked_at']

    def save(self):
        return {'id': int(self.id),
                'fov': int(self.fov),
                'cell': int(self.cell),
                'coordinate': tuple(self.coordinate),
                'raw_coordinate': tuple(self.raw_coordinate),
                'polymer': {k: list(v) for k, v in self.polymer.items()},
                'final_polymer': np.array(self.final_polymer),
                'linked': bool(self.linked),
                'linked_at': self.linked_at}


def _allele_from_dict(d):
    allele = AnAllele()
    allele.set_metadata(**d)
    return allele
