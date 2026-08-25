"""
Analysis stays usable WHILE ingestion runs.

The confirmed real failure this pins: with an ingestion live, FOV
alignment offered only the reference hybe and cell alignment "aligned"
only the two reference hybes -- even though FOV001 was already fully
ingested on disk. Cause: the mid-ingestion gate skipped every readiness
scan, so `active_hybe_list` held only what THIS session's task_done had
reported, and every downstream list (results, per-cell/per-hybe,
overlay specs) is filtered by it.

That gate is gone. Readiness is one directory listing per (modality,
FOV) -- measured 7 ms for a full 2-modality x 40-FOV sweep of the real
store -- so there is nothing an ingestion needs protecting from, and
analysis during ingestion is the whole point of the per-FOV layout.

Run: python tests/test_analysis_during_ingestion.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np                                    # noqa: E402
import h5py                                           # noqa: E402
from PyQt5 import QtWidgets                           # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

from codelab_pipeline.io import paths                 # noqa: E402
from windows.main_window import MainWindow            # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(('  ok ' if cond else 'FAIL ') + name + (f' -- {detail}' if detail and not cond else ''))


HYBES = ['Hyb_001', 'Hyb_002', 'Hyb_003', 'Hyb_004']


def build_store(root, tag):
    """A tiny project with FOV1 fully ingested (manifest + mips/fov001)."""
    dp = os.path.join(root, f'proj_{tag}')
    paths.write_manifest(dp, ['DNA'])
    sp = os.path.join(dp, 'DNA')
    mips = paths.mips_dir(sp, 1)
    os.makedirs(mips, exist_ok=True)
    for h in HYBES:
        with h5py.File(os.path.join(mips, f'{h}.h5'), 'w') as f:
            f.attrs['coordinate_order'] = 'yx'
            f.attrs['fiducial_channel'] = 555
            f.create_dataset('ch555', data=np.zeros((8, 8), dtype='uint16'))
    return sp


class FakeWorker:
    """Stands in for a live IngestionWorker -- _ingestion_is_running()
    reads _active_ingestions."""
    def __init__(self, storage_paths):
        self.storage_paths = list(storage_paths)

    def isRunning(self):
        return True


def records():
    return [{'folder': h, 'channels': [555], 'fiducial_channel': 555,
             'datatype': 'H', 'readout_id': 0, 'readout_name': '',
             'total_frames': 2} for h in HYBES]


def main():
    root = tempfile.mkdtemp(prefix='during_ingest_')
    try:
        mw = MainWindow()
        ip = mw.ui.IngestionPanel

        for tag in ('a', 'b'):
            sp = build_store(root, tag)
            ip.modality_names = ['DNA']
            ip.modality_data = {'DNA': {'storage_path': sp, 'layout_path': '',
                                        'dax_directory': '', 'active_hybe_list': []}}
            ip.FovListLineEdit.setText('1')
            mw.hybe_records_by_modality = {'DNA': records()}
            mw.fov_ready_hybes = {}
            mw._active_hybe_records_cache = {}

            # idle: the scan runs and sees all four ingested hybes
            mw._active_ingestions = []
            idle = mw._active_hybe_records_for_modality('DNA')
            check(f'[{tag}] idle scan sees the ingested FOV',
                  len(idle) == len(HYBES), f'{len(idle)}/{len(HYBES)}')

            # now an ingestion starts, and the session has NO memory of
            # what is already on disk (fresh app / earlier session)
            ip.modality_data['DNA']['active_hybe_list'] = []
            mw.fov_ready_hybes = {}
            mw._active_hybe_records_cache = {}
            mw._active_ingestions = [FakeWorker([sp])]

            during = mw._active_hybe_records_for_modality('DNA')
            check(f'[{tag}] an already-ingested FOV stays visible DURING ingestion',
                  len(during) == len(HYBES),
                  f'{len(during)}/{len(HYBES)} -- this is the reported bug')
            check(f'[{tag}] _ready_hybes answers from disk during ingestion',
                  mw._ready_hybes('DNA', 1) == set(HYBES), str(mw._ready_hybes('DNA', 1)))
            mw._active_ingestions = []

        # the memo must not pin a stale answer mid-run: a hybe that lands
        # during the run has to become usable without waiting for the end
        sp = build_store(root, 'c')
        ip.modality_data = {'DNA': {'storage_path': sp, 'layout_path': '',
                                    'dax_directory': '', 'active_hybe_list': []}}
        mw.hybe_records_by_modality = {'DNA': records() + [
            {'folder': 'Hyb_005', 'channels': [555], 'fiducial_channel': 555,
             'datatype': 'H', 'readout_id': 0, 'readout_name': '', 'total_frames': 2}]}
        mw._active_hybe_records_cache = {}
        mw._active_ingestions = [FakeWorker([sp])]
        before = len(mw._active_hybe_records_for_modality('DNA'))
        with h5py.File(os.path.join(paths.mips_dir(sp, 1), 'Hyb_005.h5'), 'w') as f:
            f.attrs['coordinate_order'] = 'yx'
            f.attrs['fiducial_channel'] = 555
            f.create_dataset('ch555', data=np.zeros((8, 8), dtype='uint16'))
        after = len(mw._active_hybe_records_for_modality('DNA'))
        check('a hybe landing mid-run appears without waiting for the run to end',
              (before, after) == (4, 5), f'{before} -> {after}')
        mw._active_ingestions = []

        print()
        print(f'{len(PASS)} passed, {len(FAIL)} failed')
        if FAIL:
            raise SystemExit('FAILURES: ' + ', '.join(FAIL))
        print('ALL GOOD')
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == '__main__':
    main()
