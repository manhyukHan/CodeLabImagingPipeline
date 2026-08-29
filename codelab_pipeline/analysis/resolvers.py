"""
FrameResolver from the STORE alone -- no Qt, no app, no session.

FrameResolver itself imports nothing but numpy; what was app-bound was
its ASSEMBLY (MainWindow._frame_resolver reads panel combos and session
caches). Everything it needs is in the store:

  within    read_same_modality_matrices per modality, hybe lists from
            the manifest-recorded layouts
  bridges   read_cross_modal_matrix / _z per modality -- and their
            PRESENCE names the moving modalities, so the hub (`shared`)
            is the one modality without a stored bridge: the hub's own
            H_across is identity by design and never persisted, which
            makes absence an identifying fact rather than a gap
  anchors   modality-level facts snapshotted onto every cell at
            alignment time (ACell.matrix_anchors) -- harvested from any
            cell of the FOV that carries them

So `resolver_for(storage_path, fov)` gives the exact same projection
capability in a notebook that the app builds live, and the cell-level
property assigners (mask intensity today, anything tomorrow) stay
inside the toolbox contract: the app USES them, it is not required by
them.
"""
import os

import numpy as np

from codelab_pipeline.alignment import frames
from codelab_pipeline.io import analysis_store
from codelab_pipeline.io import paths as store_paths
from codelab_pipeline.io import preprocess


def _modalities(storage_path):
    root = os.path.dirname(os.path.abspath(os.path.normpath(storage_path)))
    manifest = store_paths.read_manifest(root) or {}
    return root, dict(manifest.get('modalities') or {})


def resolver_for(storage_path, fov, shared=None):
    """A FrameResolver for one FOV, assembled purely from the store.

    shared: the hub modality name; inferred when omitted (the unique
    modality with no stored cross-modal bridge). Raises with the facts
    when the inference is ambiguous -- guessing a hub silently would be
    a ~13 px cross-modal claim.
    """
    root, mods = _modalities(storage_path)
    if not mods:
        raise ValueError(f'no modalities in the manifest at {root}')
    within, bridges = {}, {}
    for name, entry in mods.items():
        sp_m = os.path.join(root, name)
        layout = entry.get('layout_path')
        hybes = []
        if layout and os.path.exists(layout):
            hybes = [r['folder']
                     for r in preprocess.parse_experiment_layout(layout)]
        fm = analysis_store.read_same_modality_matrices(sp_m, int(fov), hybes)
        within[name] = {h: H for (h, _m), H in fm.items()}
        H = analysis_store.read_cross_modal_matrix(sp_m, int(fov),
                                                   modality=name)
        if H is not None:
            z = analysis_store.read_cross_modal_z(sp_m, int(fov),
                                                  modality=name)
            bridges[name] = (np.asarray(H, float), float(z or 0.0))
    if shared is None:
        hubs = [m for m in mods if m not in bridges]
        if len(hubs) != 1:
            raise ValueError(
                f'cannot infer the hub modality: {sorted(mods)} with '
                f'bridges for {sorted(bridges)} leaves {hubs} -- pass '
                f'shared= explicitly')
        shared = hubs[0]
    resolver = frames.FrameResolver(within, shared, bridges=bridges or None)
    # anchors: modality-level, snapshotted per cell at alignment time;
    # any carrying cell of the FOV testifies for the whole FOV
    anchors = {}
    cells, _ = analysis_store.read_cells(storage_path, int(fov))
    for c in (cells or []):
        for m, H in (c.get('matrix_anchors') or {}).items():
            anchors.setdefault(m, np.asarray(H, float))
        if len(anchors) == len(mods):
            break
    resolver.anchors = anchors
    return resolver


def resolvers_for(storage_path, fovs, shared=None):
    """{fov: FrameResolver} -- the shape Population.build takes."""
    return {int(f): resolver_for(storage_path, f, shared=shared)
            for f in fovs}
