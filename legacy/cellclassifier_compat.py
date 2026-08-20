"""
Dead code -- see legacy/README.md. A CellClassifier ".smeta" export bridge, never
imported by anything in this project (the module itself has no importer anywhere;
its functions only ever call each other). Kept for reference because it documents
CellClassifier's three pickle formats. Moved here on 2026-08-17, verbatim apart
from promoting the explanatory string below to a real module docstring.
"""
import pickle

import numpy as np

"""
Export CellContainer/ACell/ASpot data into CellClassifier's exact
".smeta" pickle shape, for interop with CellClassifier's own downstream
analysis tooling (SpotMetaDataAnalyzer, AnalysisPanel, etc.).

CellClassifier has THREE pickle formats, not one -- only .smeta actually
round-trips fully. Its ".cells"/".spots" loaders (_load_cells/_load_spots)
are lossy by construction: they only reconstruct area/coordinate/
brightness/size and silently discard celltype/spots/image/linked/etc.
Only .smeta (SpotMetaDataAnalyzer.save()'s shape -- a bare 3-tuple
(cells_dict, spots_dict, channel_list), each dict {fov(int): [dict,...]})
reconstructs full ACell/ASpot objects (see CellClassifier's own
windows/main_window.py::_load_spotmeta). Since every ACell/ASpot in this
project is "linked" from birth -- cells own their spots directly, no
separate link step exists here -- .smeta is the only target that actually
preserves information; this module does not attempt .cells/.spots export.

Two REAL, PERMANENT information-loss points on export (not oversights):
- ASpot.coordinate here is (x, y, z); CellClassifier's is (x, y) -- a
  2-tuple. Its own code does `x, y = spot.coordinate` in several places,
  so exporting the 3-tuple as-is would hand it a ValueError on unpack.
  z is dropped.
- spot.hybe and spot.raw_coordinate have no CellClassifier equivalent at
  all (that project's data model predates the hybe/channel distinction
  and never separates "native" from "aligned" spot position) -- dropped.
"""


def _aspot_to_cellclassifier_dict(spot):
    return {
        'id': int(spot.id),
        'fov': int(spot.fov),
        'celltype': str(spot.celltype),
        'coordinate': (float(spot.coordinate[0]), float(spot.coordinate[1])),
        'cell': int(spot.cell),
        'size': float(spot.size),
        'brightness': float(spot.brightness),
        'neighbors': tuple(spot.neighbors),
        'channel': str(spot.channel),
        'linked': bool(spot.linked),
        'linked_at': spot.linked_at,
    }


def _acell_to_cellclassifier_dict(cell, channel_list, image_source=None):
    """
    image_source: optional callable(cell) -> {channel_key(str): ndarray},
    used to populate CellClassifier's ACell.image/.channels -- this
    project's ACell has no cached per-channel image of its own, so without
    image_source the export just carries image={}/channels=[], which
    CellClassifier's own set_metadata tolerates (each field individually
    `if 'key' in kwargs`-gated).
    """
    image = image_source(cell) if image_source is not None else {}
    # CellClassifier's num_spots is keyed by channel (str); this project's
    # cell.num_spots is keyed by hybe -- cannot pass through directly,
    # must be recomputed by grouping this cell's actual spots by channel.
    num_spots_by_channel = {}
    for spot in cell.spots:
        key = str(spot.channel)
        num_spots_by_channel[key] = num_spots_by_channel.get(key, 0) + 1

    return {
        'id': int(cell.id),
        'fov': int(cell.fov),
        'celltype': str(cell.celltype),
        'area': tuple(cell.area),
        'spots': [_aspot_to_cellclassifier_dict(s) for s in cell.spots],
        'total_num_spots': int(cell.total_num_spots),
        'distmap': np.array(cell.distmap),
        'image': dict(image),
        'channels': [str(c) for c in channel_list] if image else [],
        'num_spots': num_spots_by_channel,
        'linked': bool(cell.linked),
        'linked_at': cell.linked_at,
    }


def export_smeta(cell_container, path, channel_list=None, image_source=None):
    """
    Write a CellClassifier-compatible .smeta pickle from a CellContainer:
    a bare 3-tuple (cells_dict, spots_dict, channel_list), cells_dict/
    spots_dict each {fov(int): [dict, ...]} -- matches
    SpotMetaDataAnalyzer.save()'s exact shape. spots_dict is a flat
    per-FOV list of every spot across every cell in that FOV (this
    project has no separately-tracked "unlinked"/homeless spot pool the
    way CellClassifier's array-backed SpotContainer does -- every spot
    here already belongs to a cell, so the flat list is just cell.spots
    flattened, not a distinct data source).
    """
    channel_list = list(channel_list) if channel_list else []
    cells_dict = {}
    spots_dict = {}
    for fov, cells in cell_container.data.items():
        fov = int(fov)
        cells_dict[fov] = [_acell_to_cellclassifier_dict(c, channel_list, image_source) for c in cells]
        spots_dict[fov] = [spot for cell_dict in cells_dict[fov] for spot in cell_dict['spots']]

    with open(path, 'wb') as f:
        pickle.dump((cells_dict, spots_dict, channel_list), f)
