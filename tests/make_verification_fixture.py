"""
Build the all-layer verification fixture in a CLONE of the real dataset.

Run ONCE against a fresh clone (rerunning adds duplicate spots -- uids
are never reused, so nothing corrupts, but counts double):

    cp -Rc data/chr19_downstream_new /path/to/clone     # APFS CoW clone
    # write a config whose storage_paths point at the clone, then:
    CODELAB_FIXTURE_CONFIG=/path/to/clone_config.xml \
        python tests/make_verification_fixture.py

Everything is written through the app's own paths (session containers ->
mirror_write_cells, the real batch cell-alignment worker, spot_container
-> reassign -> _persist_fov_spots, vlinks_store's allele writer) so the
on-disk shapes are exactly what the app produces. Nothing hand-writes H5.

Layers armed afterwards (verified inventory printed at the end):
  - cells with VARIED reference pairs in FOV2 (RNA-ref, DNA-ref, and one
    cross-modal-decoupled nucleus) beside FOV1's real segmented cells;
    celltypes on a subset in both FOVs.
  - real cell-level alignment (residual-form matrices + anchors +
    provenance) over the new cells.
  - spots across many (hybe, modality, channel) slices in both FOVs,
    including the Hyb_130 bridge hybe in BOTH modalities, a fiducial-
    channel slice, in-cell spots assigned by the real assignment pass
    (inheriting celltype), background spots, and mixture_centroids
    carriers.
  - chromatin-tracing alleles anchored on real assigned DNA spots, with
    fiducial_trace_adj / polymer_adj (incl. a sister-chromatid candidate pair) /
    rejected_hybes / final_polymer.
"""
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from unittest import mock
from PyQt5 import QtWidgets

CONFIG = os.environ.get('CODELAB_FIXTURE_CONFIG')
assert CONFIG and os.path.exists(CONFIG), \
    'set CODELAB_FIXTURE_CONFIG to a config whose storage_paths point at a CLONE'
rng = np.random.default_rng(7)

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
from windows.main_window import MainWindow
from codelab_pipeline.io import analysis_store as vlinks_store
from codelab_pipeline.models.cell import ACell
from codelab_pipeline.models.spot import ASpot
from codelab_pipeline.models.allele import AnAllele
from codelab_pipeline.alignment import chain as alignment


def disk(cx, cy, r):
    """(y, x) pixel arrays of a filled disk -- rasterized order."""
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    keep = (xx * xx + yy * yy) <= r * r
    return yy[keep] + cy, xx[keep] + cx


def build_alleles(w, fov, dna_hybes):
    anchors = [s for s in w.spot_container.all(fov)
               if s.modality == 'DNA' and int(s.cell) != -1][:2]
    alleles = []
    for aid, s in enumerate(anchors, start=1):
        a = AnAllele()
        ax, ay, az = s.adj_coordinate
        fiducial, polymer_adj, rejected = {}, {}, {}
        for j, h in enumerate(dna_hybes):
            if j == len(dna_hybes) - 1 and aid == 1:
                rejected[h] = 'fiducial drift above bound (fixture)'
                continue
            d = rng.normal(0, 0.6, 3)
            fiducial[h] = (ax + d[0], ay + d[1], az + d[2], float(rng.uniform(800, 3000)))
            cands = [(ax + rng.normal(0, 1.2), ay + rng.normal(0, 1.2),
                      az + rng.normal(0, 0.8), float(rng.uniform(500, 2500)))]
            if j == 1:                              # sister-chromatid pair
                cands.append((cands[0][0] + 2.1, cands[0][1] - 1.7,
                              cands[0][2] + 0.5, float(rng.uniform(500, 2500))))
            polymer_adj[h] = cands
        a.set_metadata(id=aid, fov=fov, cell=int(s.cell), anchor_hybe=s.hybe,
                       anchor_channel=int(s.channel),
                       adj_coordinate=tuple(map(float, s.adj_coordinate)),
                       raw_coordinate=tuple(map(float, s.raw_coordinate)))
        a.fiducial_trace_adj, a.polymer_adj, a.rejected_hybes = fiducial, polymer_adj, rejected
        a.final_polymer = np.array([max(v, key=lambda t: t[3])[:3] for v in polymer_adj.values()])
        alleles.append(a)
    return alleles


def main():
    with mock.patch.object(QtWidgets.QMessageBox, 'information'), \
            mock.patch.object(QtWidgets.QMessageBox, 'warning'), \
            mock.patch.object(QtWidgets.QMessageBox, 'critical'), \
            mock.patch.object(QtWidgets.QMessageBox, 'question',
                              return_value=QtWidgets.QMessageBox.Yes):
        w = MainWindow(CONFIG)
        ap = w.ui.AlignmentPanel
        rna_sp = w._storage_path_for_modality('RNA')
        dna_sp = w._storage_path_for_modality('DNA')
        rna_recs = {r['folder']: r for r in w._active_hybe_records_for_modality('RNA')}
        dna_recs = {r['folder']: r for r in w._active_hybe_records_for_modality('DNA')}
        ch = {('RNA', h): alignment.pick_channel_by_type(r, 'readout') for h, r in rna_recs.items()}
        ch.update({('DNA', h): alignment.pick_channel_by_type(r, 'readout') for h, r in dna_recs.items()})
        fid_ch = alignment.pick_channel_by_type(rna_recs['Hyb_101'], 'fiducial')

        # ---- FOV2: six cells, varied reference pairs ----
        w._activate_fov(2)
        mip = vlinks_store.read_hybe_mip(rna_sp, 2, 'Hyb_101', ch[('RNA', 'Hyb_101')])
        frame_shape = tuple(mip.shape)
        height, width = frame_shape
        cell_defs = [
            (1, ('Hyb_101', 'RNA'), ('Hyb_101', 'RNA'), (0.25, 0.30), 16, 'TypeA'),
            (2, ('Hyb_101', 'RNA'), ('Hyb_101', 'RNA'), (0.55, 0.25), 20, 'TypeB'),
            (3, ('Hyb_002', 'DNA'), ('Hyb_002', 'DNA'), (0.35, 0.60), 18, 'TypeA'),
            (4, ('Hyb_002', 'DNA'), ('Hyb_002', 'DNA'), (0.70, 0.55), 15, ''),
            (5, ('Hyb_500', 'RNA'), ('Hyb_002', 'DNA'), (0.50, 0.75), 22, 'TypeB'),
            (6, ('Hyb_400', 'DNA'), ('Hyb_400', 'DNA'), (0.20, 0.80), 14, ''),
        ]
        w.cell_container_permanent.data.setdefault(2, {})
        for cid, (rh, rm), (nh, nm), (fx, fy), r, ct in cell_defs:
            cx, cy = int(fx * width), int(fy * height)
            ay, ax = disk(cx, cy, r)
            ny, nx = disk(cx, cy, max(4, r // 2))
            c = ACell()
            c.set_metadata(id=cid, fov=2, reference_hybe=rh, reference_modality=rm,
                           nucleus=(ny.astype(float), nx.astype(float)),
                           nucleus_hybe=nh, nucleus_modality=nm, celltype=ct,
                           area=(ay.astype(float), ax.astype(float)), frame_shape=frame_shape)
            w.cell_container_permanent.data[2][cid] = c
        w.cell_container.sync_from(w.cell_container_permanent, 2)
        vlinks_store.mirror_write_cells(w._all_vlinks_storage_paths(), 2, w.cell_container_permanent)
        print(f'FOV2: wrote {len(cell_defs)} cells, reference pairs '
              f'{sorted({d[1] for d in cell_defs})}')

        # ---- real batch cell alignment over FOV2 ----
        ap.CellFovSpinBox.setValue(2)
        w._run_cell_alignment()
        worker = getattr(w, '_cell_alignment_worker', None)
        if worker is not None:
            worker.wait(600000)
            for _ in range(50):
                app.processEvents()
        aligned = [c for c in w.cell_container_permanent.get_cells(2) if c.matrices]
        print(f'FOV2 alignment: {len(aligned)}/{len(cell_defs)} cells carry matrices')

        # ---- celltypes on FOV1 cells BEFORE spots, so assignment inherits ----
        w._activate_fov(1)
        for c in sorted(w.cell_container_permanent.get_cells(1), key=lambda c: c.id)[:6]:
            c.celltype = 'TypeA' if c.id % 2 else 'TypeB'
        vlinks_store.mirror_write_cells(w._all_vlinks_storage_paths(), 1, w.cell_container_permanent)

        # ---- spots: several slices per FOV, both modalities + bridge ----
        SLICES = [('RNA', 'Hyb_105'), ('RNA', 'Hyb_500'), ('RNA', 'Hyb_130'),
                  ('DNA', 'Hyb_010'), ('DNA', 'Hyb_400'), ('DNA', 'Hyb_130'),
                  ('RNA', 'Hyb_101')]
        for fov in (1, 2):
            w._activate_fov(fov)
            cells = sorted(w.cell_container_permanent.get_cells(fov), key=lambda c: c.id)
            hosts, fshape = cells[:4], cells[0].frame_shape
            new_spots = []
            for modality, hybe in SLICES:
                channel = fid_ch if hybe == 'Hyb_101' else ch[(modality, hybe)]
                for cell in hosts:
                    y, x = w._cell_area_in_readout(cell, hybe, modality, fov)
                    if len(x) == 0:
                        continue
                    for k in rng.choice(len(x), size=min(2, len(x)), replace=False):
                        z = float(rng.uniform(6, 22))
                        s = ASpot()
                        s.set_metadata(fov=fov, modality=modality, hybe=hybe, channel=channel,
                                       raw_adj_coordinate=(float(y[k]), float(x[k]), z),
                                       adj_coordinate=(float(y[k]), float(x[k]), z),
                                       size=float(rng.uniform(1.1, 2.4)),
                                       brightness=float(rng.uniform(300, 4000)))
                        new_spots.append(s)
                for _ in range(3):
                    bx = float(rng.uniform(20, fshape[1] - 20))
                    by = float(rng.uniform(20, fshape[0] - 20))
                    z = float(rng.uniform(6, 22))
                    s = ASpot()
                    s.set_metadata(fov=fov, modality=modality, hybe=hybe, channel=channel,
                                   raw_coordinate=(by, bx, z), coordinate=(by, bx, z),
                                   size=float(rng.uniform(1.1, 2.4)),
                                   brightness=float(rng.uniform(200, 1500)))
                    new_spots.append(s)
            for s in new_spots[:2]:
                x0, y0, z0 = s.raw_coordinate
                s.set_metadata(mixture_centroids=((x0, y0, z0), (x0 + 1.4, y0 - 0.9, z0 + 0.7)))
            for uid, s in zip(vlinks_store.allocate_spot_uids(rna_sp, fov, len(new_spots)), new_spots):
                s.set_metadata(uid=uid)
                w.spot_container.add(fov, s)
            # The Save button's own order: reassign, THEN persist -- persist
            # alone writes whatever assignment state the spots already carry.
            n_a, n_u = w._reassign_fov_spots(fov)
            w._persist_fov_spots(fov)
            print(f'FOV{fov}: +{len(new_spots)} spots, {n_a} assigned / {n_u} unassigned after save')

        # ---- alleles per FOV, anchored on real assigned DNA spots ----
        dna_hybes = list(dna_recs.keys())
        for fov in (1, 2):
            alleles = build_alleles(w, fov, dna_hybes)
            vlinks_store.write_fov_alleles(dna_sp, fov, alleles)
            print(f'FOV{fov}: wrote {len(alleles)} alleles')

    # ---- inventory, fresh from the store ----
    print('\n===== FIXTURE INVENTORY =====')
    for fov in (1, 2):
        cell_dicts, _ = vlinks_store.read_cells(dna_sp, fov)
        refs = {}
        for c in cell_dicts or []:
            key = (c['reference_hybe'], c['reference_modality'])
            refs[key] = refs.get(key, 0) + 1
        spots = vlinks_store.read_spots(dna_sp, fov) or []
        slices = sorted({(d['modality'], d['hybe'], d['channel']) for d in spots})
        n_alleles = len(vlinks_store.read_fov_alleles(dna_sp, fov) or [])
        H = vlinks_store.read_cross_modal_matrix(dna_sp, fov)
        print(f'FOV{fov}: {len(cell_dicts or [])} cells {dict(sorted(refs.items()))} | '
              f'{len(spots)} spots over {len(slices)} slices | {n_alleles} alleles | '
              f'cross-modal {"H+dz" if H is not None else "none"}')
    print('FIXTURE DONE')
    return 0


if __name__ == '__main__':
    sys.exit(main())
