from PyQt5 import QtWidgets, QtCore
import numpy as np


def _format_value(value):
    """
    One-line display string for a leaf value -- numpy arrays get a
    compact, bounded repr (never the full array dumped as text, which
    for something like ACell.area's x/y point lists or distmap can run
    to thousands of entries and make the tree unreadable/slow to
    render); everything else uses its own natural str().
    """
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 'array([]) shape=' + str(value.shape)
        if value.ndim == 1 and value.size > 12:
            return (f'array shape={value.shape} dtype={value.dtype} '
                    f'min={np.nanmin(value):.3g} max={np.nanmax(value):.3g} mean={np.nanmean(value):.3g}')
        return np.array2string(value, precision=4, suppress_small=True, max_line_width=120)
    return str(value)


def _add_tree_children(parent_item, value, key_prefix=''):
    """
    Recursively expands a save()-shaped dict/list/tuple into QTreeWidgetItem
    children of parent_item -- one child per key (dict), per index (list/
    tuple of non-trivial items), or a single leaf row (anything else,
    via _format_value). Mirrors exactly what ACell.save()/ASpot.save()
    return, so every real attribute is reachable by expanding the tree,
    never summarized away or hidden -- per explicit request ("tree view
    to show entire attributes").
    """
    if isinstance(value, dict):
        for k, v in value.items():
            child = QtWidgets.QTreeWidgetItem([str(k), ''])
            parent_item.addChild(child)
            _add_tree_children(child, v, key_prefix=str(k))
    elif isinstance(value, (list, tuple)) and value and any(isinstance(v, (dict, list, tuple, np.ndarray)) for v in value):
        for i, v in enumerate(value):
            label = f'{key_prefix}[{i}]' if key_prefix else f'[{i}]'
            child = QtWidgets.QTreeWidgetItem([label, ''])
            parent_item.addChild(child)
            _add_tree_children(child, v, key_prefix=label)
    else:
        parent_item.setText(1, _format_value(value))


def populate_cell_tree(tree, cell_dicts, spots_by_cell_id):
    """
    One top-level item per cell (label: 'Cell {id} ({modality}, N spots)'),
    every ACell.save() key as an expandable child. Cells no longer carry a
    'spots' key -- spots live in the FOV's own store -- so n_spots is passed
    in per cell rather than read off the dict; see spots_by_cell_id.
    """
    tree.clear()
    for c in cell_dicts:
        n_spots = spots_by_cell_id.get(c.get('id'), 0)
        top = QtWidgets.QTreeWidgetItem(
            [f"Cell {c.get('id')} ({c.get('reference_modality', '?')}, {n_spots} spot(s))", ''])
        tree.addTopLevelItem(top)
        _add_tree_children(top, c)


def populate_allele_tree(tree, allele_dicts):
    """
    One top-level item per allele (label: 'Allele {id} (cell={id|
    unassigned}, N hybe(s) traced)'), every AnAllele.save() key as an
    expandable child -- same "show entire attributes" convention as
    populate_cell_tree/populate_spot_tree.
    """
    tree.clear()
    for a in allele_dicts:
        cell_label = 'unassigned' if a.get('cell', -1) == -1 else f"cell {a.get('cell')}"
        n_traced = len(a.get('polymer', {}))
        top = QtWidgets.QTreeWidgetItem([f"Allele {a.get('id')} ({cell_label}, {n_traced} hybe(s) traced)", ''])
        tree.addTopLevelItem(top)
        _add_tree_children(top, a)


def populate_matrix_tree(tree, rows):
    """
    rows: [{'fov':, 'modality':, 'same_modality': [(hybe, summary_str), ...],
    'cross_modal': summary_str or None}, ...] -- one top-level item per
    (FOV, modality), children = one per same-modality hybe plus, when that
    modality has an accepted cross-modal link, one more for it. Keyed by
    modality as well as FOV because matrices are the one modality-scoped
    thing in vlinks: the bridge hybe (e.g. Hyb_130) appears under BOTH
    modalities meaning different things. Pure ground-truth
    display of whatever's persisted (see MainWindow._refresh_cell_spot_
    status_matrix_panel's own docstring for exactly what's read and why)
    -- summary_str is already-formatted text (dx/dy/angle), no further
    nesting needed the way _add_tree_children's own recursive dict/list
    handling is for (this data isn't a save()-shaped dict).
    """
    tree.clear()
    for row in rows:
        n_hybes = len(row['same_modality'])
        top = QtWidgets.QTreeWidgetItem(
            [f"FOV{row['fov']:02d} -- {row.get('modality', '?')} ({n_hybes} hybe(s))", ''])
        tree.addTopLevelItem(top)
        for hybe, summary in row['same_modality']:
            top.addChild(QtWidgets.QTreeWidgetItem([hybe, summary]))
        if row['cross_modal'] is not None:
            top.addChild(QtWidgets.QTreeWidgetItem(['DNA->RNA (cross-modal)', row['cross_modal']]))


def populate_spot_tree(tree, indexed_spot_dicts):
    """
    One top-level item per spot (label: 'Spot {global_index} (hybe, cell)'),
    every ASpot.save() key as a child. indexed_spot_dicts: [(global_index,
    spot_dict), ...] -- the DISPLAY index is supplied by the caller, NOT
    read from spot_dict['id'] (ASpot.id defaults to 0 and is never
    actually set to a real per-spot value anywhere spots get created in
    this app -- confirmed real bug, every spot showed as "Spot 0"). The
    caller resolves the real, FOV-based global index the same way every
    other spot-numbered view in this app already does (see MainWindow.
    _global_spot_order/_global_spot_index_map), so a spot's number here
    matches its number in the crop displayer / 3D-localization popup.
    """
    tree.clear()
    for global_index, s in indexed_spot_dicts:
        cell_label = 'unassigned' if s.get('cell', -1) == -1 else f"cell {s.get('cell')}"
        top = QtWidgets.QTreeWidgetItem([f"Spot {global_index} ({s.get('hybe', '?')} ch{s.get('channel', '?')}, {cell_label})", ''])
        tree.addTopLevelItem(top)
        _add_tree_children(top, s)


class CellSpotStatusDisplayer(QtWidgets.QMainWindow):
    """
    Pop-up detail browser reading DIRECTLY off vlinks.h5 (vlinks_store.
    read_cells/read_fov_spots) every time it refreshes -- never the
    session's own transient self.cell_container/fov_unassigned_spots --
    per explicit request: the existing in-app status views (Memory
    Status included) reflect whatever's currently staged in memory, which
    can drift from what's actually on disk (e.g. edits made but not yet
    Saved); this window exists specifically to answer "what is REALLY
    persisted right now," at full per-attribute detail rather than the
    Memory Status viewer's own summary counts.

    Four independent panels: Matrix (ground-truth FOV-level alignment
    matrices, scoped by modality alone -- see set_matrix_data), and the
    three things vlinks.h5 stores per-cell/spot/allele: Cells (scoped by
    FOV alone -- a cell IS FOV-scoped), Spots (scoped by FOV + hybe +
    channel, since a real spot always belongs to exactly one of each),
    and Alleles (scoped by FOV alone, same as Cells -- an allele's own
    polymer dict already spans every traced hybe internally, no further
    sub-scoping needed). Each panel's own combo selection re-reads and
    re-populates that panel's own tree live; the top-level Refresh button
    additionally re-reads storage_path's own FOV list from scratch and
    the "total" counts, for the case where vlinks.h5 changed via some
    OTHER path (e.g. a batch job, or another session) since this window
    was last opened -- the two are complementary, not redundant: a combo
    change always re-fetches fresh data for its OWN scope already,
    Refresh is for when the CHOICES themselves (which FOVs/hybes/channels
    even exist) might be stale.
    """
    refresh_requested = QtCore.pyqtSignal()
    cell_fov_changed = QtCore.pyqtSignal()
    spot_scope_changed = QtCore.pyqtSignal()
    allele_fov_changed = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Cell / Spot Status Detail')
        self.resize(1100, 700)

        central = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)

        topRow = QtWidgets.QWidget()
        topRowLayout = QtWidgets.QHBoxLayout(topRow)
        topRowLayout.setContentsMargins(0, 0, 0, 0)
        # No modality control: vlinks is one file keyed by (modality, hybe),
        # so every panel shows every modality at once. Toggling a modality
        # here would hide half the store for no reason -- the split it
        # mirrored no longer exists.
        self.RefreshPushButton = QtWidgets.QPushButton('Refresh')
        topRowLayout.addWidget(self.RefreshPushButton)
        topRowLayout.addStretch()
        outer.addWidget(topRow)
        self.RefreshPushButton.clicked.connect(self.refresh_requested.emit)

        # -- Matrix panel: ground-truth check, reads directly off vlinks.h5
        # (bypassing MainWindow's own self.fov_matrices/cross_modal_result
        # in-memory caches entirely) -- per explicit request, "we also need
        # to know the current matrix status stored in vlinks" to
        # troubleshoot independently of any session staleness. No scope
        # combo of its own: shows every FOV at once (like the Alignment
        # tab's own Results lists), scoped only by the shared Modality
        # combo above and refreshed by the shared Refresh button -- sits
        # above the Cells/Spots/Alleles splitter, not a 4th pane inside it.
        matrixGroup = QtWidgets.QGroupBox('Matrix (persisted, all FOVs)')
        matrixLayout = QtWidgets.QVBoxLayout(matrixGroup)
        self.MatrixCountLabel = QtWidgets.QLabel('')
        matrixLayout.addWidget(self.MatrixCountLabel)
        self.MatrixTree = QtWidgets.QTreeWidget()
        self.MatrixTree.setColumnCount(2)
        self.MatrixTree.setHeaderLabels(['Hybe', 'dx, dy, angle'])
        self.MatrixTree.header().setStretchLastSection(False)  # see CellTree's own comment on this
        self.MatrixTree.setMaximumHeight(220)
        matrixLayout.addWidget(self.MatrixTree, stretch=1)
        outer.addWidget(matrixGroup)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        outer.addWidget(splitter, stretch=1)

        # -- Cells panel --
        cellGroup = QtWidgets.QGroupBox('Cells')
        cellLayout = QtWidgets.QVBoxLayout(cellGroup)
        cellForm = QtWidgets.QFormLayout()
        self.CellFovComboBox = QtWidgets.QComboBox()
        cellForm.addRow('FOV:', self.CellFovComboBox)
        cellLayout.addLayout(cellForm)
        self.CellCountLabel = QtWidgets.QLabel('')
        cellLayout.addWidget(self.CellCountLabel)
        # default OFF: the distance map is an O(n_spots^2) pdist PER CELL,
        # recomputed on every refresh -- routine refreshes shouldn't pay
        # for an analysis product nobody asked for. Ticking it refreshes.
        self.DistmapCheckBox = QtWidgets.QCheckBox('Compute spot distance maps (on demand)')
        self.DistmapCheckBox.setChecked(False)
        cellLayout.addWidget(self.DistmapCheckBox)
        self.CellTree = QtWidgets.QTreeWidget()
        self.CellTree.setColumnCount(2)
        self.CellTree.setHeaderLabels(['Attribute', 'Value'])
        # stretchLastSection defaults to True on QTreeWidget's header,
        # which forces the 'Value' column to squeeze into whatever width
        # is left instead of ever exceeding the viewport -- with that on,
        # a long formatted value (a 3x3 matrix, a long tuple, ...) just
        # gets visually clipped with NO way to reach the rest of it, no
        # horizontal scrollbar ever appears since Qt sees no overflow to
        # scroll to. Turning it off lets the real (content-sized) column
        # widths exceed the viewport, which is what actually makes the
        # horizontal scrollbar show up on demand (setHorizontalScroll
        # BarPolicy(AsNeeded) is already QTreeWidget's own default, no
        # need to set it explicitly -- this is the actual missing piece).
        self.CellTree.header().setStretchLastSection(False)
        cellLayout.addWidget(self.CellTree, stretch=1)
        splitter.addWidget(cellGroup)
        self.CellFovComboBox.currentIndexChanged.connect(self.cell_fov_changed.emit)
        self.DistmapCheckBox.toggled.connect(lambda _on: self.cell_fov_changed.emit())

        # -- Spots panel --
        spotGroup = QtWidgets.QGroupBox('Spots')
        spotLayout = QtWidgets.QVBoxLayout(spotGroup)
        spotForm = QtWidgets.QFormLayout()
        self.SpotFovComboBox = QtWidgets.QComboBox()
        spotForm.addRow('FOV:', self.SpotFovComboBox)
        self.SpotHybeComboBox = QtWidgets.QComboBox()
        spotForm.addRow('Hybe:', self.SpotHybeComboBox)
        self.SpotChannelComboBox = QtWidgets.QComboBox()
        spotForm.addRow('Channel:', self.SpotChannelComboBox)
        spotLayout.addLayout(spotForm)
        self.SpotCountLabel = QtWidgets.QLabel('')
        spotLayout.addWidget(self.SpotCountLabel)
        self.SpotTree = QtWidgets.QTreeWidget()
        self.SpotTree.setColumnCount(2)
        self.SpotTree.setHeaderLabels(['Attribute', 'Value'])
        self.SpotTree.header().setStretchLastSection(False)  # see CellTree's own comment on this
        spotLayout.addWidget(self.SpotTree, stretch=1)
        splitter.addWidget(spotGroup)
        self.SpotFovComboBox.currentIndexChanged.connect(self.spot_scope_changed.emit)
        self.SpotHybeComboBox.currentIndexChanged.connect(self.spot_scope_changed.emit)
        self.SpotChannelComboBox.currentIndexChanged.connect(self.spot_scope_changed.emit)

        # -- Alleles panel --
        alleleGroup = QtWidgets.QGroupBox('Alleles')
        alleleLayout = QtWidgets.QVBoxLayout(alleleGroup)
        alleleForm = QtWidgets.QFormLayout()
        self.AlleleFovComboBox = QtWidgets.QComboBox()
        alleleForm.addRow('FOV:', self.AlleleFovComboBox)
        alleleLayout.addLayout(alleleForm)
        self.AlleleCountLabel = QtWidgets.QLabel('')
        alleleLayout.addWidget(self.AlleleCountLabel)
        self.AlleleTree = QtWidgets.QTreeWidget()
        self.AlleleTree.setColumnCount(2)
        self.AlleleTree.setHeaderLabels(['Attribute', 'Value'])
        self.AlleleTree.header().setStretchLastSection(False)  # see CellTree's own comment on this
        alleleLayout.addWidget(self.AlleleTree, stretch=1)
        splitter.addWidget(alleleGroup)
        self.AlleleFovComboBox.currentIndexChanged.connect(self.allele_fov_changed.emit)

    # -- population helpers (pure UI-state setters, no vlinks/file access here --
    # MainWindow reads vlinks_store and hands the results in) --


    def _set_fov_choices(self, combo, fov_list):
        current = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for fov in fov_list:
            combo.addItem(f'FOV{fov:02d}', fov)
        if current in fov_list:
            combo.setCurrentIndex(fov_list.index(current))
        combo.blockSignals(False)

    def set_cell_fov_choices(self, fov_list):
        self._set_fov_choices(self.CellFovComboBox, fov_list)

    def set_spot_fov_choices(self, fov_list):
        self._set_fov_choices(self.SpotFovComboBox, fov_list)

    def set_allele_fov_choices(self, fov_list):
        self._set_fov_choices(self.AlleleFovComboBox, fov_list)

    def current_cell_fov(self):
        return self.CellFovComboBox.currentData()

    def current_spot_fov(self):
        return self.SpotFovComboBox.currentData()

    def current_allele_fov(self):
        return self.AlleleFovComboBox.currentData()

    def set_spot_hybe_choices(self, hybes):
        current = self.SpotHybeComboBox.currentText()
        self.SpotHybeComboBox.blockSignals(True)
        self.SpotHybeComboBox.clear()
        self.SpotHybeComboBox.addItems(hybes)
        if current in hybes:
            self.SpotHybeComboBox.setCurrentText(current)
        self.SpotHybeComboBox.blockSignals(False)

    def set_spot_channel_choices(self, channels):
        current = self.SpotChannelComboBox.currentText()
        self.SpotChannelComboBox.blockSignals(True)
        self.SpotChannelComboBox.clear()
        self.SpotChannelComboBox.addItems([str(c) for c in channels])
        if current in [str(c) for c in channels]:
            self.SpotChannelComboBox.setCurrentText(current)
        self.SpotChannelComboBox.blockSignals(False)

    def current_spot_hybe(self):
        return self.SpotHybeComboBox.currentText() or None

    def current_spot_channel(self):
        text = self.SpotChannelComboBox.currentText()
        return int(text) if text else None

    def set_cell_data(self, cell_dicts, n_total_cells, n_total_fovs, spots_by_cell_id=None):
        populate_cell_tree(self.CellTree, cell_dicts, spots_by_cell_id or {})
        self.CellTree.resizeColumnToContents(0)
        self.CellTree.resizeColumnToContents(1)
        self.CellCountLabel.setText(f'{len(cell_dicts)} cell(s) in this FOV  |  {n_total_cells} cell(s) total across {n_total_fovs} FOV(s)')

    def set_spot_data(self, indexed_spot_dicts, n_total_spots):
        """indexed_spot_dicts: [(global_index, spot_dict), ...] -- see populate_spot_tree's own docstring."""
        populate_spot_tree(self.SpotTree, indexed_spot_dicts)
        self.SpotTree.resizeColumnToContents(0)
        self.SpotTree.resizeColumnToContents(1)
        self.SpotCountLabel.setText(f'{len(indexed_spot_dicts)} spot(s) in this FOV/hybe/channel  |  {n_total_spots} spot(s) total')

    def set_allele_data(self, allele_dicts, n_total_alleles, n_total_fovs):
        populate_allele_tree(self.AlleleTree, allele_dicts)
        self.AlleleTree.resizeColumnToContents(0)
        self.AlleleTree.resizeColumnToContents(1)
        self.AlleleCountLabel.setText(f'{len(allele_dicts)} allele(s) in this FOV  |  {n_total_alleles} allele(s) total across {n_total_fovs} FOV(s)')

    def set_matrix_data(self, rows):
        """rows: see populate_matrix_tree's own docstring."""
        populate_matrix_tree(self.MatrixTree, rows)
        self.MatrixTree.resizeColumnToContents(0)
        self.MatrixTree.resizeColumnToContents(1)
        self.MatrixTree.expandAll()
        n_fovs = len({r['fov'] for r in rows})
        n_cross = sum(1 for r in rows if r['cross_modal'] is not None)
        self.MatrixCountLabel.setText(
            f'{n_fovs} FOV(s) x {len(rows)} modality row(s)'
            f'{f", {n_cross} with a cross-modal matrix" if n_cross else ""}')
