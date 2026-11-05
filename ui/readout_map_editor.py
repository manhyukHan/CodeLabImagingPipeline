"""The readout map, settled by hand.

WHY A TABLE AND NOT A RULE. The ExperimentLayout gives one readout id
per round, which is enough while a round reads one locus. A second dye
on the same adapter reads a DIFFERENT barcode in the same round --
635/adapter1 is R1 and 747/adapter1 is R51 -- and that second number is
a design fact. Continuing the numbering invents it; calling the second
a repeat files a different locus under "the same locus imaged twice",
which is a quality number we read. Neither is recoverable from the
acquisition, and the operator is the one who knows.

So the layout SEEDS the table and the operator settles it. What is
settled is stamped into every trace it produces (AnAllele.readout_bins),
so a trace says what it is without its map being resolved again, and
editing the map later cannot silently re-label measurements already
taken.

The DESIGNED RANGE is the table's other question. An acquired set of
readouts cannot say whether a number it is missing was skipped by
design, not acquired yet, or never contiguous -- so the range is
declared, and it decides both where bin 0 sits and how many slots trail
the last acquired bin.
"""
from PyQt5 import QtCore, QtGui, QtWidgets

from codelab_pipeline.analysis.readout_map import NOT_A_BIN, ReadoutMap

GREY = '#777'
WARN = '#b00'
DTYPES = ('H', 'R', 'T', 'B')

COLS = ('hybe', 'channel', 'readout #', 'datatype', 'polymer index')


class ReadoutMapEditor(QtWidgets.QDialog):
    """hybe | channel | rindex | dtype | pindex, plus the designed range.

    rindex and dtype are editable; pindex is DERIVED and read-only --
    it is `rindex - lo`, and letting it be typed independently would
    make two fields able to disagree about the same thing.
    """

    def __init__(self, rows, declared='', title='Readout map', parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(640, 520)
        self._rows = [dict(r) for r in rows or []]
        outer = QtWidgets.QVBoxLayout(self)

        note = QtWidgets.QLabel(
            'The layout seeds this; you settle it. A readout number the '
            'layout does not name is blank — a second dye on the same '
            'adapter reads a different barcode, and only the design says '
            'which. The polymer index follows from the readout number and '
            'the designed range.')
        note.setWordWrap(True)
        note.setStyleSheet(f'color: {GREY};')
        outer.addWidget(note)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Designed readout range'))
        self.RangeLineEdit = QtWidgets.QLineEdit(str(declared or ''))
        self.RangeLineEdit.setPlaceholderText(
            'e.g. 1-80 — blank means the acquired span')
        self.RangeLineEdit.setToolTip(
            'Which loci the experiment set out to read, as against which '
            'it got. A declared readout nothing imaged is an empty slot '
            'and the NaN stripe of every distance map. Blank takes the '
            'span of what is here, which cannot express a design whose '
            'last readouts are not in yet.')
        row.addWidget(self.RangeLineEdit, stretch=1)
        outer.addLayout(row)

        # FILLING A WHOLE CHANNEL AT ONCE. 62 bins is 62 cells, and
        # typing them one at a time is how a transcription error gets
        # in. This is NOT the auto-continuation the map refuses: nothing
        # derives the start or the step, the operator states them, and
        # every number it writes is visible in the table and editable
        # afterwards. Declaring "747 reads R51 onward" is a design fact;
        # inferring it would have been a guess.
        fill = QtWidgets.QHBoxLayout()
        fill.addWidget(QtWidgets.QLabel('Fill channel'))
        self.FillChannelComboBox = QtWidgets.QComboBox()
        fill.addWidget(self.FillChannelComboBox)
        fill.addWidget(QtWidgets.QLabel('from readout'))
        self.FillStartSpinBox = QtWidgets.QSpinBox()
        self.FillStartSpinBox.setRange(0, 100000)
        self.FillStartSpinBox.setValue(1)
        fill.addWidget(self.FillStartSpinBox)
        fill.addWidget(QtWidgets.QLabel('step'))
        self.FillStepSpinBox = QtWidgets.QSpinBox()
        self.FillStepSpinBox.setRange(-1000, 1000)
        self.FillStepSpinBox.setValue(1)
        fill.addWidget(self.FillStepSpinBox)
        self.FillOnlyBlankCheckBox = QtWidgets.QCheckBox('blank rows only')
        self.FillOnlyBlankCheckBox.setChecked(True)
        self.FillOnlyBlankCheckBox.setToolTip(
            'Leaves the readout numbers the layout DID name alone. '
            'Unticked, it overwrites them too.')
        fill.addWidget(self.FillOnlyBlankCheckBox)
        self.FillPushButton = QtWidgets.QPushButton('Fill')
        self.FillPushButton.setToolTip(
            'Numbers that channel\'s rows in hybe order, from the start '
            'you give. Nothing is inferred -- the start and the step are '
            'yours, and every value lands in the table to be checked.')
        fill.addWidget(self.FillPushButton)
        fill.addStretch(1)
        outer.addLayout(fill)

        self.Table = QtWidgets.QTableWidget(0, len(COLS))
        self.Table.setHorizontalHeaderLabels(COLS)
        self.Table.verticalHeader().setVisible(False)
        self.Table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch)
        self.Table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        outer.addWidget(self.Table, stretch=1)

        self.Complaints = QtWidgets.QLabel('')
        self.Complaints.setWordWrap(True)
        outer.addWidget(self.Complaints)

        bar = QtWidgets.QHBoxLayout()
        self.ResetPushButton = QtWidgets.QPushButton('Reset to the layout')
        bar.addWidget(self.ResetPushButton)
        bar.addStretch(1)
        self.Buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bar.addWidget(self.Buttons)
        outer.addLayout(bar)

        self._seed = [dict(r) for r in self._rows]
        self.FillChannelComboBox.addItems(
            [str(c) for c in sorted({int(r['channel'])
                                     for r in self._rows})])
        # start on the channel that still has blanks, which is the one
        # this control exists for
        blanks = sorted({int(r['channel']) for r in self._rows
                         if r.get('rindex') is None})
        if blanks:
            self.FillChannelComboBox.setCurrentText(str(blanks[0]))
        self._fill()
        self.Buttons.accepted.connect(self.accept)
        self.Buttons.rejected.connect(self.reject)
        self.ResetPushButton.clicked.connect(self._reset)
        self.FillPushButton.clicked.connect(self._fill_channel)
        self.Table.itemChanged.connect(self._changed)
        self.RangeLineEdit.textChanged.connect(lambda _t: self._recompute())

    # -- table -----------------------------------------------------------

    def _fill(self):
        self.Table.blockSignals(True)
        self.Table.setRowCount(len(self._rows))
        for i, r in enumerate(self._rows):
            for j, v in enumerate((r['hybe'], r['channel'])):
                it = QtWidgets.QTableWidgetItem(str(v))
                it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                it.setForeground(QtGui.QBrush(QtGui.QColor(GREY)))
                self.Table.setItem(i, j, it)
            rid = r.get('rindex')
            it = QtWidgets.QTableWidgetItem('' if rid is None else str(rid))
            self.Table.setItem(i, 2, it)
            dt = QtWidgets.QTableWidgetItem(str(r.get('dtype') or 'H'))
            self.Table.setItem(i, 3, dt)
            pi = QtWidgets.QTableWidgetItem('')
            pi.setFlags(pi.flags() & ~QtCore.Qt.ItemIsEditable)
            pi.setForeground(QtGui.QBrush(QtGui.QColor(GREY)))
            self.Table.setItem(i, 4, pi)
        self.Table.blockSignals(False)
        self._recompute()

    def _changed(self, item):
        if item.column() not in (2, 3):
            return
        r = self._rows[item.row()]
        if item.column() == 2:
            t = item.text().strip()
            try:
                r['rindex'] = int(t) if t else None
            except ValueError:
                r['rindex'] = None
        else:
            t = item.text().strip().upper()
            r['dtype'] = t if t in DTYPES else 'H'
        self._recompute()

    def _fill_channel(self):
        """Number one channel's rows in hybe order, from the given start.

        H rows only: an R or T round re-images a bin that already has an
        id, and renumbering it would make the repeat distance compare
        two different loci.
        """
        try:
            ch = int(self.FillChannelComboBox.currentText())
        except (TypeError, ValueError):
            return
        start = int(self.FillStartSpinBox.value())
        step = int(self.FillStepSpinBox.value())
        blank_only = self.FillOnlyBlankCheckBox.isChecked()
        targets = [r for r in self._rows
                   if int(r['channel']) == ch
                   and str(r.get('dtype') or '').upper() == 'H'
                   and (not blank_only or r.get('rindex') is None)]
        targets.sort(key=lambda r: str(r['hybe']))
        for i, r in enumerate(targets):
            r['rindex'] = start + i * step
        self._fill()

    def _reset(self):
        self._rows = [dict(r) for r in self._seed]
        self._fill()

    def _recompute(self):
        """Re-derive every polymer index and say what is still wrong."""
        rows, n_slots, complaints = ReadoutMap.place_rows(
            self._rows, self.RangeLineEdit.text().strip() or None)
        self._placed, self._n_slots = rows, n_slots
        self.Table.blockSignals(True)
        for i, r in enumerate(rows):
            it = self.Table.item(i, 4)
            if it is not None:
                it.setText('' if r['pindex'] == NOT_A_BIN else str(r['pindex']))
            rit = self.Table.item(i, 2)
            if rit is not None:
                bad = (str(r.get('dtype', '')).upper() == 'H'
                       and r.get('rindex') is None)
                rit.setBackground(QtGui.QBrush(
                    QtGui.QColor('#ffe9e9' if bad else '#ffffff')))
        self.Table.blockSignals(False)
        n_bins = sum(1 for r in rows if r['pindex'] != NOT_A_BIN)
        if complaints:
            self.Complaints.setStyleSheet(f'color: {WARN};')
            self.Complaints.setText(
                '%d still to settle — %s' % (len(complaints),
                                             '; '.join(complaints[:3])
                                             + (' …' if len(complaints) > 3
                                                else '')))
        else:
            self.Complaints.setStyleSheet(f'color: {GREY};')
            self.Complaints.setText(
                '%d polymer bin(s) on a %d-slot axis, %d slot(s) nothing '
                'imaged.' % (n_bins, n_slots, max(0, n_slots - n_bins)))
        ok = self.Buttons.button(QtWidgets.QDialogButtonBox.Ok)
        if ok is not None:
            ok.setEnabled(not complaints)

    # -- result ----------------------------------------------------------

    def value(self):
        """The settled map as one config attribute, or '' for the seed.

        Empty means "the layout as it stands", so a round added later is
        picked up without anyone reopening this -- the same reason the
        localization rounds store empty for the expectation.
        """
        rows, n_slots, complaints = ReadoutMap.place_rows(
            self._rows, self.RangeLineEdit.text().strip() or None)
        if complaints:
            raise ValueError('; '.join(complaints[:4]))
        seeded, seed_slots, seed_bad = ReadoutMap.place_rows(self._seed, None)
        if not seed_bad and rows == seeded and n_slots == seed_slots:
            return ''
        return ReadoutMap.from_rows(rows, slots=n_slots).to_text()

    def declared(self):
        return self.RangeLineEdit.text().strip()
