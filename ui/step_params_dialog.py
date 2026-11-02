"""One pipeline step's parameters, in a dialog generated from the config map.

THERE IS NO SECOND PARAMETER MODEL HERE, and that is the point.
`MainWindow._CONFIG_PARAM_MAP` already names every analysis parameter as
{section: {param: (panel, widget)}} and already drives BOTH Save Config
and Load Config, so a hand-written form per step would be a second list
of what a step's parameters are -- exactly the drift that map exists to
prevent. This dialog instead:

  - reads `status.STEP_SECTIONS` for the step's config sections,
  - builds a COPY of each named widget (same class, same items, same
    range, same current value), and
  - on OK hands the values back through `_apply_config_params`, the same
    door a loaded config comes through, so the dialog and a config file
    cannot apply a value differently.

WHERE THE VALUE LIVES: the tab's own widget (the user's call,
2026-11-02). OK really does change the Spot Localization tab. That keeps
one truth -- "a run uses each tab's current settings" stays literally
true, and Save Config records what ran. Cancel changes nothing, and
"Restore" puts back the values as they were when the dialog opened.

THE ROUNDS LIST is the one control with no widget on another tab,
because until now there was none: the status module derives the expected
localization rounds from the config (29 on the SG store) and nobody
could choose a subset. Its home is a widget on the Pipeline panel that
IS in the config map, so the choice is saved and loaded like every other
parameter. Empty means every target -- see status.localization_run_targets.
"""
import os

from PyQt5 import QtCore, QtWidgets

from codelab_pipeline.pipeline import status as S

#: extra widgets a step runs with that are NOT config: how many workers
#: is a property of this machine, not of the experiment, which is why
#: the config map leaves it out. The step still runs with it, so the
#: dialog shows it -- in its own box, labelled as not saved.
MACHINE = {
    'ingestion': (('workers', 'IngestionPanel', 'IngestWorkersSpinBox'),),
    'localization': (('workers', 'IngestionPanel', 'IngestWorkersSpinBox'),),
}

#: the parameters worth putting on the step's one-line summary in the
#: table. Not all of them: the line is a reminder, not the dialog.
SUMMARY_KEYS = {
    'ingestion': ('voxel_xy_um', 'voxel_z_um'),
    'segmentation': ('method', 'diameter', 'projection', 'gate_drop_flags'),
    'fov_alignment': ('border_trim', 'max_shift'),
    'cross_modal': ('channel_type', 'max_shift'),
    'cell_alignment': ('channel_type', 'pad', 'max_z_shift'),
    'localization': ('model', 'min_p_exist', 'pad'),
    'celltype': ('method', 'fov_ranges'),
    'cellcycle': ('proxy', 'min_total', 'origin'),
    'tracing': ('engine', 'readout_channel'),
}

#: parameters the dialog deliberately does NOT offer, and why. These are
#: not "advanced": they are values a step MEASURES per FOV, and offering
#: them as settings invites fixing by hand what the run is about to
#: detect. The segmentation step profiles each FOV's own focus before it
#: segments (pipeline_wiring._run_segmentation, phase 'focus'), and the
#: focal plane really does move -- 44 to 59 across six fields of one
#: experiment -- so one number here cannot be right for a batch.
MEASURED_NOT_SET = {
    'cell_segmentation': {
        'z_plane': 'detected per FOV before each segmentation',
        'z_start': 'the detected plateau, per FOV',
        'z_end': 'the detected plateau, per FOV',
    },
}

#: the rounds picker edits this one, so the generated form must not also
#: offer it as a line of text
PICKER_PARAMS = {('localization_run', 'rounds')}

#: section -> (the version parameter, the panel attribute holding that
#: panel's OWN per-engine QStackedWidget). Which parameter belongs to
#: which engine is read from that stack, not from the parameter names:
#: grouping by prefix put chromatin tracing's `readout_*` and
#: `fiducial_*` -- the v1 page's own fit columns -- in boxes of their
#: own that stayed visible under v3, where they do nothing. The panel
#: already answers this question correctly for its own widgets, so the
#: dialog asks it rather than keeping a second answer.
VERSION_STACK = {
    'chromatin_tracing': ('engine', 'ChromatinTracingPanel',
                          'FitParamsStackedWidget'),
}

#: a readable name for the tab a section writes to
PANEL_TITLES = {
    'IngestionPanel': 'the Ingestion tab',
    'CellSegmentPanel': 'the Cell Segmentation tab',
    'AlignmentPanel': 'the Alignment tab',
    'SpotLocalizationPanel': 'the Spot Localization tab',
    'CelltypeDeterminationPanel': 'the Celltype Determination tab',
    'CellCyclePanel': 'the Cell Cycle tab',
    'ChromatinTracingPanel': 'the Chromatin Tracing tab',
    'PipelinePanel': 'the Pipeline tab',
    'localize_3d_displayer': 'the 3D localization window',
}
GREY = '#464646'


def _editor_for(src):
    """A copy of `src`: same class, same choices, same current value.

    Returns None for a widget class the config map should not contain,
    rather than guessing -- an unknown widget is a map change, and a
    guess here would silently write the wrong kind of value.
    """
    if isinstance(src, QtWidgets.QComboBox):
        w = QtWidgets.QComboBox()
        for i in range(src.count()):
            w.addItem(src.itemText(i), src.itemData(i))
        w.setEditable(src.isEditable())
        w.setCurrentIndex(src.currentIndex())
        if src.isEditable():
            w.setCurrentText(src.currentText())
        return w
    if isinstance(src, QtWidgets.QCheckBox):
        w = QtWidgets.QCheckBox(src.text())
        w.setChecked(src.isChecked())
        return w
    if isinstance(src, QtWidgets.QLineEdit):
        w = QtWidgets.QLineEdit(src.text())
        w.setPlaceholderText(src.placeholderText())
        return w
    if isinstance(src, QtWidgets.QDoubleSpinBox):
        w = QtWidgets.QDoubleSpinBox()
        w.setDecimals(src.decimals())
        w.setRange(src.minimum(), src.maximum())
        w.setSingleStep(src.singleStep())
        w.setSuffix(src.suffix())
        w.setValue(src.value())
        return w
    if isinstance(src, QtWidgets.QSpinBox):
        w = QtWidgets.QSpinBox()
        w.setRange(src.minimum(), src.maximum())
        w.setSingleStep(src.singleStep())
        w.setSuffix(src.suffix())
        w.setValue(src.value())
        return w
    return None


def _set_editor(w, value):
    """Put a captured string back into an editor (the dialog's own copy).

    The real widgets are never written this way -- they go through
    _apply_config_params, which has the retry pass a dependent combo
    needs. An editor has no dependants, so a direct set is enough.
    """
    from windows.main_window import MainWindow
    try:
        MainWindow._apply_widget_value(w, value)
    except Exception:                                       # noqa: BLE001
        pass


#: the summary line is a reminder in a table cell, not the dialog
SUMMARY_CHARS = 92


def summary_line(mw, step):
    """One line of the step's current parameters, for the steps table."""
    from PyQt5 import QtWidgets as W
    from windows.main_window import MainWindow
    bits = []
    keys = SUMMARY_KEYS.get(step, ())
    for section in S.STEP_SECTIONS.get(step, ()):
        entries = mw._CONFIG_PARAM_MAP.get(section, {})
        for param in keys:
            if param not in entries:
                continue
            panel, widget = entries[param]
            try:
                src = mw._config_widget(panel, widget)
                value = MainWindow._widget_value(src)
            except Exception:                               # noqa: BLE001
                continue
            if isinstance(src, W.QCheckBox):
                # a flag reads as its own name or not at all: 'remove
                # edge cells 1' says less than 'remove edge cells'
                if value == '1':
                    bits.append(param.replace('_', ' '))
                continue
            if value == '':
                continue
            if ('/' in value or '\\' in value) and len(value) > 24:
                # a model directory: the line has room for its name, not
                # for the path to it
                value = os.path.basename(value.rstrip('/\\'))
            # NOT `value in ('', '0')`: a border trim of 0 is a setting
            # someone chose, and hiding it made the FOV alignment row
            # look like it had no parameters at all
            bits.append(f'{param} {value}')
    for label, panel, widget in MACHINE.get(step, ()):
        try:
            bits.append(f'{label} {mw._config_widget(panel, widget).value()}')
        except Exception:                                   # noqa: BLE001
            pass
    if step == 'localization':
        try:
            chosen = mw.ui.PipelinePanel.LocalizationRoundsLineEdit.text().strip()
            n = len([c for c in chosen.split(',') if c.strip()])
            bits.append(f'{n} round(s)' if chosen else 'all rounds')
        except Exception:                                   # noqa: BLE001
            pass
    line = ', '.join(bits)
    if len(line) > SUMMARY_CHARS:
        line = line[:SUMMARY_CHARS - 1].rstrip(' ,') + '…'
    return line


class RoundsPicker(QtWidgets.QWidget):
    """Which localization rounds the next run touches.

    Shows every expected target (status.localization_targets), grouped by
    modality. ALL CHECKED IS STORED AS AN EMPTY STRING, not as the 29
    names: empty keeps following the config, so a tracing round added
    later is localized without anyone opening this dialog again.
    """

    def __init__(self, targets, chosen, parent=None):
        super().__init__(parent)
        self.boxes = []
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        keep = set()
        for part in (chosen or '').split(','):
            bits = [b.strip() for b in part.split('|')]
            if len(bits) >= 3 and bits[2].isdigit():
                keep.add((bits[0], bits[1], int(bits[2])))

        inner = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(inner)
        grid.setContentsMargins(2, 2, 2, 2)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(2)
        # THREE columns, not four: a fourth fitted only by pushing a
        # horizontal scrollbar under the list and clipping 'ch635' to
        # 'ch63'. A round whose name cannot be read is not a choice.
        ncol = 3
        row = col = 0
        for modality in sorted(targets):
            rounds = sorted(targets[modality])
            if not rounds:
                continue
            if col:
                row, col = row + 1, 0
            head = QtWidgets.QLabel(f'{modality}  ({len(rounds)})')
            head.setStyleSheet(f'color: {GREY};')
            grid.addWidget(head, row, 0, 1, ncol)
            row += 1
            for hybe, ch in rounds:
                cb = QtWidgets.QCheckBox(f'{hybe} · ch{ch}')
                cb.setChecked(not keep or (modality, hybe, int(ch)) in keep)
                cb.key = (modality, hybe, int(ch))
                self.boxes.append(cb)
                grid.addWidget(cb, row, col)
                col += 1
                if col == ncol:
                    row, col = row + 1, 0
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(210)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        bar = QtWidgets.QHBoxLayout()
        for text, fn in (('All', lambda: self._set_all(True)),
                         ('None', lambda: self._set_all(False)),
                         ('Clear toeholds', self._clear_toeholds)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            bar.addWidget(b)
        bar.addStretch(1)
        self.CountLabel = QtWidgets.QLabel('')
        self.CountLabel.setStyleSheet(f'color: {GREY};')
        bar.addWidget(self.CountLabel)
        outer.addLayout(bar)
        for cb in self.boxes:
            cb.toggled.connect(self._count)
        self._count()

    def _set_all(self, on):
        for cb in self.boxes:
            cb.setChecked(on)

    def _clear_toeholds(self):
        for cb in self.boxes:
            if cb.key[1].lower().startswith('toe'):
                cb.setChecked(False)

    def _count(self):
        n = sum(1 for cb in self.boxes if cb.isChecked())
        total = len(self.boxes)
        self.CountLabel.setText(
            f'{n} of {total} — all rounds, and the list follows the config'
            if n == total else f'{n} of {total} selected')

    def value(self):
        """'' when everything is checked, else 'MOD|HYBE|CH,...'."""
        picked = [cb for cb in self.boxes if cb.isChecked()]
        if len(picked) == len(self.boxes):
            return ''
        return ','.join('%s|%s|%d' % cb.key for cb in picked)


class StepParamsDialog(QtWidgets.QDialog):
    """The parameters of one step, applied to the tabs they belong to."""

    def __init__(self, mw, step, note='', plan_line='', targets=None,
                 parent=None):
        super().__init__(parent or mw)
        from ui.pipeline_panel import STEP_LABELS
        self.mw, self.step = mw, step
        self.setWindowTitle(f'{STEP_LABELS.get(step, step)} — parameters')
        self.setMinimumWidth(640)
        self._fields = []          # (section, param, editor, original)
        self._machine = []         # (src widget, editor)
        self.rounds = None

        outer = QtWidgets.QVBoxLayout(self)
        panels = sorted({PANEL_TITLES.get(p, p)
                         for section in S.STEP_SECTIONS.get(step, ())
                         for p, _w in mw._CONFIG_PARAM_MAP.get(section, {}).values()})
        where = QtWidgets.QLabel(
            ('These values live on %s. OK writes them there.'
             % ' and '.join(panels)) if panels
            else 'This step has no parameters of its own.')
        where.setWordWrap(True)
        where.setStyleSheet(f'color: {GREY};')
        outer.addWidget(where)

        # The sections scroll, the buttons and the plan line do not.
        # Chromatin tracing declares nearly forty parameters; laid out
        # straight into the dialog it is taller than the screen, and a
        # dialog whose OK button is off the bottom cannot be used at all.
        body = QtWidgets.QWidget()
        inner = QtWidgets.QVBoxLayout(body)
        inner.setContentsMargins(0, 0, 0, 0)
        for section in S.STEP_SECTIONS.get(step, ()):
            box = self._section_box(section)
            if box is not None:
                inner.addWidget(box)

        if step == 'localization':
            box = QtWidgets.QGroupBox('Rounds to localize')
            lay = QtWidgets.QVBoxLayout(box)
            flat = {m: list(v) for m, v in (targets or {}).items() if v}
            if flat:
                field = mw.ui.PipelinePanel.LocalizationRoundsLineEdit
                self.rounds = RoundsPicker(flat, field.text())
                lay.addWidget(self.rounds)
            else:
                msg = QtWidgets.QLabel(
                    'No targets yet — press Refresh status first, or the '
                    'config names no round to localize.')
                msg.setWordWrap(True)
                msg.setStyleSheet(f'color: {GREY};')
                lay.addWidget(msg)
            inner.addWidget(box)

        if MACHINE.get(step):
            box = QtWidgets.QGroupBox('This machine — not saved in the config')
            form = QtWidgets.QFormLayout(box)
            for label, panel, widget in MACHINE[step]:
                try:
                    src = mw._config_widget(panel, widget)
                except Exception:                           # noqa: BLE001
                    continue
                ed = _editor_for(src)
                if ed is None:
                    continue
                self._machine.append((src, ed))
                form.addRow(label, ed)
            inner.addWidget(box)

        inner.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(body)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        screen = QtWidgets.QApplication.primaryScreen()
        cap = int(screen.availableGeometry().height() * 0.70) if screen else 620
        # the scroll area's own sizeHint is small, so without this every
        # dialog came out the same short height with its contents
        # scrolling inside -- the cap must bite only when the content
        # really is taller than the screen allows
        body.adjustSize()
        scroll.setMinimumHeight(min(body.sizeHint().height() + 4, cap))
        scroll.setMaximumHeight(cap)
        outer.addWidget(scroll, 1)

        if note:
            lab = QtWidgets.QLabel(note)
            lab.setWordWrap(True)
            lab.setStyleSheet(f'color: {GREY};')
            outer.addWidget(lab)
        if plan_line:
            lab = QtWidgets.QLabel(plan_line)
            lab.setWordWrap(True)
            f = lab.font()
            f.setBold(True)
            lab.setFont(f)
            outer.addWidget(lab)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
            | QtWidgets.QDialogButtonBox.RestoreDefaults)
        buttons.button(QtWidgets.QDialogButtonBox.RestoreDefaults).setText(
            'Restore')
        buttons.button(QtWidgets.QDialogButtonBox.RestoreDefaults).setToolTip(
            'The values as they were when this dialog opened.')
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QtWidgets.QDialogButtonBox.RestoreDefaults
                       ).clicked.connect(self._restore)
        outer.addWidget(buttons)

    def _section_box(self, section):
        """One box per config section, with its families folded in.

        A FAMILY IS READ OFF THE NAMES, not off a list kept here: three
        or more parameters sharing a leading token (classical_*, v2_*,
        v3_*) get their own sub-box. Chromatin tracing declares nearly
        forty parameters and segmentation sixteen, and as one flat
        column the engine-specific ones sit among the general ones
        looking equally relevant. Grouping by prefix is generic, so a
        parameter added to the map joins its family without an edit here.
        """
        from windows.main_window import MainWindow
        entries = self.mw._CONFIG_PARAM_MAP.get(section, {})
        if not entries:
            return None
        rows, before = [], len(self._fields)
        measured = MEASURED_NOT_SET.get(section, {})
        for param, (panel, widget) in entries.items():
            if param in measured or (section, param) in PICKER_PARAMS:
                continue
            try:
                src = self.mw._config_widget(panel, widget)
            except Exception:                               # noqa: BLE001
                # a map entry naming a window that is not built yet (the
                # 3D localization popup). Skipped, not guessed at.
                continue
            ed = _editor_for(src)
            if ed is None:
                continue
            ed.setToolTip(f'{panel}.{widget}')
            self._fields.append((section, param, ed,
                                 MainWindow._widget_value(src)))
            rows.append((param, ed, panel, widget))
        if len(self._fields) == before:
            return None

        pages = self._engine_pages(section, rows)
        if pages:
            families, titles = pages
        else:
            counts = {}
            for param, _ed, _p, _w in rows:
                head = param.split('_')[0]
                counts[head] = counts.get(head, 0) + 1
            families = {param: param.split('_')[0] for param, _e, _p, _w in rows
                        if counts[param.split('_')[0]] >= 3 and len(rows) > 6}
            titles = {}

        box = QtWidgets.QGroupBox(section.replace('_', ' '))
        lay = QtWidgets.QVBoxLayout(box)
        main = QtWidgets.QFormLayout()
        lay.addLayout(main)
        sub, group_boxes = {}, {}
        for param, ed, panel, widget in rows:
            head = families.get(param)
            if head is not None:
                if head not in sub:
                    g = QtWidgets.QGroupBox(titles.get(head, str(head)))
                    group_boxes[head] = g
                    sub[head] = QtWidgets.QFormLayout(g)
                    lay.addWidget(g)
                prefix = str(head) + '_'
                text = param[len(prefix):] if param.startswith(prefix) else param
                form = sub[head]
            else:
                form, text = main, param
            label = QtWidgets.QLabel(text.replace('_', ' '))
            label.setToolTip(f'config: {section}/{param}')
            form.addRow(label, ed)

        if section in VERSION_STACK and group_boxes:
            version = VERSION_STACK[section][0]
            combo = next((ed for p, ed, _pa, _w in rows if p == version), None)
            if isinstance(combo, QtWidgets.QComboBox):
                self._bind_version(combo, group_boxes)
        if measured:
            note = QtWidgets.QLabel(
                'Measured per FOV, not set here: '
                + '; '.join('%s (%s)' % (k.replace('_', ' '), why)
                            for k, why in measured.items()) + '.')
            note.setWordWrap(True)
            note.setStyleSheet(f'color: {GREY};')
            lay.addWidget(note)
        return box

    def _engine_pages(self, section, rows):
        """({param: page index}, {page index: that engine's name}), from
        the panel's own per-engine QStackedWidget, or None.

        A parameter belongs to the engine whose PAGE its widget sits on.
        Walking up from the widget is the only way to ask that question
        of the panel itself; a list here would be the second answer that
        goes stale the next time a page gains a row.
        """
        entry = VERSION_STACK.get(section)
        if entry is None:
            return None
        _version, panel, attr = entry
        try:
            stack = self.mw._config_widget(panel, attr)
        except Exception:                                   # noqa: BLE001
            return None
        pages = [stack.widget(i) for i in range(stack.count())]
        families = {}
        for param, _ed, pname, wname in rows:
            try:
                w = self.mw._config_widget(pname, wname)
            except Exception:                               # noqa: BLE001
                continue
            node = w
            while node is not None:
                if node in pages:
                    families[param] = pages.index(node)
                    break
                node = node.parentWidget()
        if not families:
            return None
        titles = {}
        combo = next((ed for p, ed, _pa, _w in rows if p == _version), None)
        for i in set(families.values()):
            titles[i] = (combo.itemText(i) if combo is not None
                         and i < combo.count() else 'engine %d' % i)
        return families, titles

    def _bind_version(self, combo, group_boxes):
        """Show only the chosen engine's page of parameters.

        The keys of `group_boxes` are page INDICES of the panel's own
        stack, and the combo's index selects the page -- the panel wires
        the same two together (ChromatinTracingPanel line 1236), so the
        dialog shows exactly what the tab would show.
        """
        def _apply():
            current = combo.currentIndex()
            for index, g in group_boxes.items():
                g.setVisible(index == current)
        combo.currentIndexChanged.connect(lambda _i: _apply())
        _apply()

    def _restore(self):
        for _section, _param, ed, original in self._fields:
            _set_editor(ed, original)

    def values(self):
        """{section: {param: str}} in the config's own spelling."""
        from windows.main_window import MainWindow
        out = {}
        for section, param, ed, _original in self._fields:
            out.setdefault(section, {})[param] = MainWindow._widget_value(ed)
        return out

    def apply(self):
        """Write the dialog back to the tabs. Returns a log line."""
        from windows.main_window import MainWindow
        changed = [f'{s}/{p}' for s, p, ed, orig in self._fields
                   if MainWindow._widget_value(ed) != orig]
        self.mw._apply_config_params(self.values())
        for src, ed in self._machine:
            src.setValue(ed.value())
        if self.rounds is not None:
            field = self.mw.ui.PipelinePanel.LocalizationRoundsLineEdit
            value = self.rounds.value()
            if value != field.text():
                field.setText(value)
                changed.append('pipeline/localization_rounds')
        return ', '.join(changed)


def open_for(mw, step, note='', plan_line='', targets=None):
    """Show the dialog; returns the log line of what changed, or None."""
    dlg = StepParamsDialog(mw, step, note=note, plan_line=plan_line,
                           targets=targets)
    if dlg.exec_() != QtWidgets.QDialog.Accepted:
        return None
    return dlg.apply()
