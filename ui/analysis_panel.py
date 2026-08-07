from PyQt5 import QtWidgets, QtGui, QtCore

class AnalysisPanelUI(object):
    """
    Interactive analysis tab: a lightweight embedded Python console that
    executes arbitrary code against a live namespace containing all of the
    pipeline objects currently held by the main window.

    The namespace is populated by MainWindow._build_analysis_namespace() and
    refreshed every time the "Run" button is pressed, so newly computed
    objects (cell_container, fov_matrices, …) are always visible without
    restarting the app.

    Keyboard shortcut: Ctrl+Return runs the selected text (or the whole
    editor when nothing is selected), matching the behaviour of most
    notebook/IDE environments.
    """

    def setupUi(self, Widget):
        Widget.setObjectName('AnalysisPanel')
        layout = QtWidgets.QVBoxLayout(Widget)

        # -- Help / namespace summary --
        helpLabel = QtWidgets.QLabel(
            'Run Python code against the live pipeline state.  '
            'Press <b>Run</b> or <b>Ctrl+Return</b> to execute.  '
            'Available names: <tt>window</tt>, <tt>cell_container</tt>, '
            '<tt>cell_container_permanent</tt>, <tt>fov_matrices</tt>, '
            '<tt>modality_data</tt>, <tt>modality_names</tt>, '
            '<tt>hybe_records</tt>, <tt>cross_modal_result</tt>, '
            '<tt>np</tt>, <tt>pd</tt>, <tt>plt</tt>.'
        )
        helpLabel.setWordWrap(True)
        layout.addWidget(helpLabel)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        layout.addWidget(splitter, stretch=1)

        # -- Code editor --
        editorGroup = QtWidgets.QGroupBox('Code')
        editorLayout = QtWidgets.QVBoxLayout(editorGroup)
        self.CodeEditor = QtWidgets.QPlainTextEdit()
        self.CodeEditor.setFont(QtGui.QFont('Courier', 10))
        self.CodeEditor.setPlaceholderText(
            '# Example:\n'
            'print(cell_container)\n'
            'print(list(fov_matrices.keys())[:5])\n'
        )
        self.CodeEditor.setMinimumHeight(120)
        editorLayout.addWidget(self.CodeEditor)
        splitter.addWidget(editorGroup)

        # -- Output display --
        outputGroup = QtWidgets.QGroupBox('Output')
        outputLayout = QtWidgets.QVBoxLayout(outputGroup)
        self.OutputDisplay = QtWidgets.QPlainTextEdit()
        self.OutputDisplay.setFont(QtGui.QFont('Courier', 10))
        self.OutputDisplay.setReadOnly(True)
        self.OutputDisplay.setMinimumHeight(80)
        outputLayout.addWidget(self.OutputDisplay)
        splitter.addWidget(outputGroup)

        # -- Buttons --
        btnRow = QtWidgets.QWidget()
        btnLayout = QtWidgets.QHBoxLayout(btnRow)
        btnLayout.setContentsMargins(0, 0, 0, 0)

        self.RunPushButton = QtWidgets.QPushButton('Run  (Ctrl+Return)')
        self.RunPushButton.setDefault(True)

        self.ClearOutputPushButton = QtWidgets.QPushButton('Clear Output')
        self.ClearCodePushButton = QtWidgets.QPushButton('Clear Code')

        btnLayout.addWidget(self.RunPushButton)
        btnLayout.addWidget(self.ClearOutputPushButton)
        btnLayout.addWidget(self.ClearCodePushButton)
        btnLayout.addStretch()
        layout.addWidget(btnRow)

        # Ctrl+Return shortcut for the editor
        shortcut = QtWidgets.QShortcut(
            QtGui.QKeySequence('Ctrl+Return'), Widget)
        shortcut.activated.connect(self.RunPushButton.click)
