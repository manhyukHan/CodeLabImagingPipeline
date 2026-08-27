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
import os
import time

from PyQt5 import QtGui, QtWidgets

# Both things that write a log write it here: this window's Save button and
# launch_codelab.vbs. The folder is committed (log/.gitkeep) with its
# contents gitignored, so it exists in a fresh checkout and nothing in it
# is ever committed by accident.
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'log')


def default_log_dir():
    """The log folder, creating it if needed; falls back to the cwd.

    A save dialog that cannot open where it was told to is worse than one
    that opens somewhere plausible, so an unwritable install path degrades
    rather than raising in a button handler.
    """
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        return LOG_DIR
    except OSError:
        return os.getcwd()


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
        self.SavePushButton = QtWidgets.QPushButton('Save...')
        self.SavePushButton.clicked.connect(self._save_log)
        row.addWidget(self.SavePushButton)
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

    def _save_log(self):
        # The suggested name reuses the line-stamp date format but with
        # dashes for the time -- colons are illegal in Windows filenames.
        # Suggested as a FULL path so the dialog opens in the log folder;
        # a bare name would open wherever the process happens to be.
        suggested = os.path.join(default_log_dir(),
                                 time.strftime('codelab_log_%Y-%m-%d-%H-%M-%S.txt'))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save Log', suggested, 'Text files (*.txt);;All files (*)')
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.LogTextEdit.toPlainText() + '\n')
        except OSError as e:
            QtWidgets.QMessageBox.critical(self, 'Save Log', f"Can't write {path}: {e}")
            return
        # confirmation lands in the log itself, stamped like everything else
        self.append(f'Log saved to {path}')
