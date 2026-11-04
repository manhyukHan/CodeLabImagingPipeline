"""
FrameResolver from the STORE alone -- no Qt, no app, no session.

FrameResolver itself imports nothing but numpy; what was app-bound was
its ASSEMBLY (MainWindow._frame_resolver reads panel combos and session
caches). Everything it needs is in the store:

  within    read_same_modality_matrices per modality, hybe lists from
            aligned_hybes -- the FOV's OWN manifest record of what it
            aligned, so nothing outside the project root is consulted
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
    for name in mods:
        sp_m = os.path.join(root, name)
        # The FOV layer's hybe list comes from the STORE'S OWN record of
        # what it aligned, never from the manifest's layout_path. That
        # path points into the acquisition share -- outside the project
        # root, and carrying whichever drive letter the last session to
        # open the project had the NAS mounted on (Z: for one user, Q:
        # for the next, the same file either way). Unreachable, it left
        # the hybe list empty, hence the FOV layer empty, and
        # to_shared's per-hybe identity fallback then dropped the entire
        # FOV translation without a word: on the MP58 store that is 3116
        # stored matrices ignored across 41 FOVs, up to 13.7 px of real
        # translation per FOV arriving as fiducial drift, which the gate
        # then refuses. aligned_hybes answers from the FOV's own
        # manifest, so the whole assembly stays inside the root.
        hybes = sorted(analysis_store.aligned_hybes(sp_m, int(fov)))
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
    anchors, residual_mods = {}, set()
    cells, _ = analysis_store.read_cells(storage_path, int(fov))
    for c in (cells or []):
        for m, H in (c.get('matrix_anchors') or {}).items():
            anchors.setdefault(m, np.asarray(H, float))
        for (_h, m), e in (c.get('matrices') or {}).items():
            if (e or {}).get('yx_is_residual'):
                residual_mods.add(m)
    # A cell residual is defined RELATIVE to its modality's FOV layer --
    # it is precisely the part the FOV matrix did not account for. So an
    # empty FOV layer does not mean "nothing to compose with": it means
    # the composition cannot be formed at all, and consuming the
    # residual alone returns a position short by the whole FOV
    # translation. Refuse with the facts, the way
    # polymer.layout_records refuses an unreachable layout, rather than
    # hand back a resolver that answers confidently and wrongly.
    broken = sorted(m for m in residual_mods if not within.get(m))
    if broken:
        raise ValueError(
            f'FOV {int(fov)}: cells carry per-cell residual matrices for '
            f'modality {broken}, but the store holds no FOV-level '
            f'matrices for it, so the two legs cannot be composed and '
            f'every mapped position would be short by the FOV '
            f'translation. Re-run FOV alignment for this FOV, or check '
            f'that {store_paths.analysis_fov_dir(storage_path, int(fov))} '
            f'still holds its matrices__<modality>.h5.')
    resolver.anchors = anchors
    return resolver


def resolvers_for(storage_path, fovs, shared=None):
    """{fov: FrameResolver} -- the shape Population.build takes."""
    return {int(f): resolver_for(storage_path, f, shared=shared)
            for f in fovs}
