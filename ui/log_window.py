"""
The one combined session log.

Every panel used to carry its own little log box, so each message landed in
exactly one tab and following a multi-step run meant hopping between them.
This window replaces all of those: MainWindow.log() is the single sink for
every panel's messages, and each line is timestamped here
('2026-08-24-11:14:25-Segmenting cells...') so a session leaves a readable
trace of when things happened, not just that they did.

Comes up together with the app (main.py shows it right before the main
window) and can be re-opened any time with the main window's 'Show Log'
corner button; closing it merely hides it. Quitting the app is the MAIN
window's close button, which asks first -- see MainWindow.closeEvent.
"""
import time

from PyQt5 import QtGui, QtWidgets


class LogWindow(QtWidgets.QMainWindow):
    # Keeps the log bounded on a days-long session: QPlainTextEdit drops the
    # oldest blocks past this, and 20k lines is far more history than any
    # single run of the pipeline produces.
    MAX_LINES = 20000

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Log')
        self.resize(760, 420)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.LogTextEdit = QtWidgets.QPlainTextEdit()
        self.LogTextEdit.setReadOnly(True)
        self.LogTextEdit.setMaximumBlockCount(self.MAX_LINES)
        # Fixed-width font so the timestamp prefix stays a visual column.
        self.LogTextEdit.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        layout.addWidget(self.LogTextEdit)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        self.ClearPushButton = QtWidgets.QPushButton('Clear')
        self.ClearPushButton.clicked.connect(self.LogTextEdit.clear)
        row.addWidget(self.ClearPushButton)
        layout.addLayout(row)

    def append(self, message):
        stamp = time.strftime('%Y-%m-%d-%H:%M:%S')
        # A multi-line message gets the stamp on every line -- continuation
        # lines are appended as separate entries all over the app, so
        # stamping uniformly keeps the left edge parseable.
        for line in (str(message).splitlines() or ['']):
            self.LogTextEdit.appendPlainText(f'{stamp}-{line}')
