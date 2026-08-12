from PyQt5 import QtWidgets


class MemoryViewerDisplayer(QtWidgets.QMainWindow):
    """
    Pop-up summarizing what's currently persisted (and would auto-activate,
    see windows/main_window.py's _activate_fov) in every experiment's
    vlinks.h5 this session knows about, per FOV. Spans every storage path
    in _all_vlinks_storage_paths (RNA and DNA both, whenever the Alignment
    tab's Cross-Modal fields are set), since a saved cell's spots/matrices
    get mirrored into all of them -- this view answers "if I open this FOV
    right now, what would show up automatically, with no re-computation."

    Matrix status is reported as 4 separate layers rather than one merged
    number (per explicit request -- a single "computed/total" mixing FOV-
    level per-hybe counts with everything else read as an unexplained
    "4/16"). Each layer's own denominator is the intuitive one for that
    layer, not a shared/confusing one:
    - FOV: 0/1 or 1/1 -- has within-experiment alignment been run for this
      FOV at all (collapsed across hybes; the per-hybe detail lives in the
      Alignment tab's own Results list, not here).
    - Cross-modal: 0/1 or 1/1 -- is this FOV's storage path paired with
      another modality (via the Alignment tab's Cross-Modal RNA/DNA
      fields) and, if so, has that cross-modal matrix been computed.
    - Cell align: (#cells with cell-based alignment computed) / (#cells
      total) -- alignment, NOT celltype.
    - Cell celltype / Spot celltype: (#cells or #spots with linked=True,
      i.e. celltype determination has actually run and set .celltype) /
      total. Two separate columns since celltype determination can (and
      often does) classify cells and spots to different completion
      levels -- FOV mode always keeps them in lockstep (every spot
      inherits its cell's celltype), but Barcode mode classifies each
      independently, so a cell being classified doesn't guarantee every
      one of its spots also got classified (e.g. if a spot's own barcode-
      channel position couldn't be resolved for one of the required
      hybes).

    Freely resizable, matching this app's own convention that dynamic
    content gets a real pop-up window rather than a fixed-size embedded
    panel (matplotlib/Qt font & DPI rendering doesn't universalize cleanly
    across macOS/Windows Server/Linux, so nothing here is size-constrained).
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Cell / Spot Memory Status')
        self.resize(900, 480)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)

        self.RefreshPushButton = QtWidgets.QPushButton('Refresh')
        layout.addWidget(self.RefreshPushButton)

        self.Table = QtWidgets.QTableWidget()
        self.Table.setColumnCount(10)
        self.Table.setHorizontalHeaderLabels(
            ['Storage Path', 'FOV', 'Saved At', 'FOV align', 'Cross-modal', 'Cell align',
             'Cell celltype', 'Spot celltype', 'Spot Z', 'Spots'])
        self.Table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.Table.verticalHeader().setVisible(False)
        layout.addWidget(self.Table)

    def set_data(self, rows):
        """
        rows: list of dicts, one per (storage_path, fov) -- keys
        storage_path, fov, saved_at, n_spots,
        fov_computed, fov_total, cross_computed, cross_total,
        cell_computed, cell_total, cell_celltype_computed, cell_celltype_total,
        spot_computed, spot_total, z_computed, z_total.

        n_spots (the 'Spots' column) is every real persisted spot for
        this FOV -- cell-owned AND the FOV-level unassigned pool, per
        confirmed bug (this used to only count cell-owned spots). 'Spot
        Z' is a SEPARATE count over that same total: how many have a
        real, non-default Z (see MainWindow._refresh_memory_viewer's own
        comment on why coordinate[2] != 0.0 is used as the persisted
        proxy for "Z actually localized", since _z_status itself is
        session-transient and never persisted).
        """
        self.Table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            values = [
                row['storage_path'],
                f"FOV{row['fov']:02d}",
                row['saved_at'] or '(never saved)',
                f"{row['fov_computed']}/{row['fov_total']}",
                f"{row['cross_computed']}/{row['cross_total']}" if row['cross_total'] else 'n/a',
                f"{row['cell_computed']}/{row['cell_total']}",
                f"{row['cell_celltype_computed']}/{row['cell_celltype_total']}",
                f"{row['spot_computed']}/{row['spot_total']}",
                f"{row['z_computed']}/{row['z_total']}",
                str(row['n_spots']),
            ]
            for j, val in enumerate(values):
                self.Table.setItem(i, j, QtWidgets.QTableWidgetItem(val))
        self.Table.resizeColumnsToContents()
        header = self.Table.horizontalHeader()
        header.setStretchLastSection(True)
