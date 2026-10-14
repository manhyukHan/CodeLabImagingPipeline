"""
DAX ingestion with buffer frames and a FovMap sheet, end to end on
fabricated data.

Pins: the optional bufferFrames / bufferFramesEnd layout columns (absent
-> 0, so every old layout parses as before), the frame slice the worker
takes before de-interleaving (depth = scan frames / channels, and plane
0 of every channel is the first SCAN frame, not the first buffer frame),
the FovMap sheet (folder-specific rows beat `*` rows beat `fov - 1`;
no sheet -> {}), the ConvZscan index the worker actually opens under a
mapping, and the loud refusal of a frame count that no longer divides
by the channel count once the buffers are removed.

Run:  python tests/test_dax_buffer_fovmap.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py                                                    # noqa: E402
import numpy as np                                             # noqa: E402
import pandas as pd                                            # noqa: E402

from codelab_pipeline.io import paths, preprocess              # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}'
          + (f'   [{detail}]' if detail and not cond else ''))


H = W = 16


def write_dax(path, frame_values):
    """One 16x16 movie whose frame f is the constant frame_values[f]
    (uint16), in the .dax layout read_dax expects, plus its .inf."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frames = np.stack([np.full((H, W), v, dtype=np.uint16) for v in frame_values])
    frames.tofile(path)
    with open(path[:-4] + '.inf', 'w') as f:
        f.write(f'frame dimensions = {W} x {H}\n'
                f'number of frames = {len(frame_values)}\n'
                f'data type = 16 bit integers (binary, little endian)\n')


def layout_row(folder, hybe_num, total, buf0=None, buf1=None):
    row = {'FolderName': folder, 'Readouts': hybe_num, 'DataType': 'H', 'HybNum': hybe_num,
           'channels': '[750, 561]', 'fiducialChannel': 561, 'channelLayout': 'interleaved',
           'totalFrames': total}
    if buf0 is not None:
        row['bufferFrames'] = buf0
    if buf1 is not None:
        row['bufferFramesEnd'] = buf1
    return row


root = tempfile.mkdtemp(prefix='dax_buffer_fovmap_')
try:
    dax_dir = os.path.join(root, 'raw')
    store = os.path.join(root, 'proj', 'DNA')
    paths.write_manifest(os.path.join(root, 'proj'), ['DNA'])

    print('layout columns')
    plain = os.path.join(root, 'plain.xlsx')
    pd.DataFrame([layout_row('Hyb_001', 1, 8)]).to_excel(plain, index=False)
    rec = preprocess.parse_experiment_layout(plain)[0]
    check('no buffer columns -> 0 / 0', (rec['buffer_frames'], rec['buffer_frames_end']) == (0, 0))
    check('no FovMap sheet -> {}', preprocess.parse_fov_map(plain) == {})
    check('missing workbook -> {}', preprocess.parse_fov_map(os.path.join(root, 'nope.xlsx')) == {})

    buffered = os.path.join(root, 'buffered.xlsx')
    with pd.ExcelWriter(buffered) as xw:
        pd.DataFrame([layout_row('Hyb_001', 1, 14, 3, 1), layout_row('Hyb_002', 2, 14, 3, 1),
                      layout_row('Hyb_003', 3, 14, None, None)]).to_excel(xw, index=False)
        pd.DataFrame([{'FolderName': '*', 'fov': 2, 'daxIndex': 5},
                      {'FolderName': 'Hyb_002', 'fov': 2, 'daxIndex': 7},
                      {'FolderName': None, 'fov': 3, 'daxIndex': 9}]).to_excel(xw, sheet_name='FovMap', index=False)
    recs = preprocess.parse_experiment_layout(buffered)
    check('bufferFrames / bufferFramesEnd read per row',
          [(r['buffer_frames'], r['buffer_frames_end']) for r in recs] == [(3, 1), (3, 1), (0, 0)])
    fov_map = preprocess.parse_fov_map(buffered)
    check('FovMap sheet: blank FolderName folds into *',
          fov_map == {'*': {2: 5, 3: 9}, 'Hyb_002': {2: 7}}, str(fov_map))
    check('resolver: folder row beats * row beats fov-1',
          [preprocess.dax_index_for(fov_map, f, v) for f, v in
           (('Hyb_002', 2), ('Hyb_001', 2), ('Hyb_001', 3), ('Hyb_001', 1))] == [7, 5, 9, 0])
    check('resolver: no map -> fov-1', preprocess.dax_index_for({}, 'Hyb_001', 4) == 3
          and preprocess.dax_index_for(None, 'Hyb_001', 4) == 3)
    bad = os.path.join(root, 'bad.xlsx')
    with pd.ExcelWriter(bad) as xw:
        pd.DataFrame([layout_row('Hyb_001', 1, 8)]).to_excel(xw, index=False)
        pd.DataFrame([{'fov': 1, 'index': 0}]).to_excel(xw, sheet_name='FovMap', index=False)
    try:
        preprocess.parse_fov_map(bad)
        check('FovMap without daxIndex rejected', False)
    except ValueError:
        check('FovMap without daxIndex rejected', True)

    print('worker: buffer frames come off the frame axis before de-interleaving')
    # 14 frames: 3 buffer (values 100..102), 10 scan frames alternating
    # ch750 / ch561 (values 1..10), 1 end buffer (value 200)
    values = [100, 101, 102] + list(range(1, 11)) + [200]
    write_dax(os.path.join(dax_dir, 'Hyb_001', 'ConvZscan_00.dax'), values)
    fov_r, hybe_r, err = preprocess.convert_dax_to_h5_worker(1, recs[0], dax_dir, store, 'DNA')
    check('buffered movie ingests', err is None, str(err))
    with h5py.File(paths.stack_path(store, 1, 'Hyb_001'), 'r') as f:
        d750, d561 = f['/stack/ch750'][:], f['/stack/ch561'][:]
        attrs = dict(f.attrs)
    check('depth = (14 - 3 - 1) / 2 = 5 planes per channel', d750.shape[-1] == 5 and d561.shape[-1] == 5,
          str((d750.shape, d561.shape)))
    check('ch750 planes are scan frames 0,2,4,.. (values 1,3,5,7,9) -- no buffer frame leaked in',
          [int(d750[0, 0, z]) for z in range(5)] == [1, 3, 5, 7, 9], str(d750[0, 0, :]))
    check('ch561 planes are scan frames 1,3,5,.. (values 2,4,6,8,10)',
          [int(d561[0, 0, z]) for z in range(5)] == [2, 4, 6, 8, 10], str(d561[0, 0, :]))
    check('provenance attrs: buffers, dax index, depth',
          (attrs['buffer_frames'], attrs['buffer_frames_end'], attrs['dax_index'], attrs['expected_depth'])
          == (3, 1, 0, 5), str(attrs))

    print('worker: FovMap decides which ConvZscan the FOV reads')
    write_dax(os.path.join(dax_dir, 'Hyb_002', 'ConvZscan_07.dax'), values)
    idx = preprocess.dax_index_for(fov_map, 'Hyb_002', 2)
    fov_r, hybe_r, err = preprocess.convert_dax_to_h5_worker(2, recs[1], dax_dir, store, 'DNA', dax_index=idx)
    check('FOV 2 of Hyb_002 read ConvZscan_07 (no ConvZscan_01 exists)', err is None, str(err))
    with h5py.File(paths.stack_path(store, 2, 'Hyb_002'), 'r') as f:
        check('stored path and dax_index name the mapped movie',
              f.attrs['path'].endswith('ConvZscan_07.dax') and f.attrs['dax_index'] == 7, f.attrs['path'])
    # overwrite=True: in append mode the complete stack just published above
    # returns early before any DAX path is formed (that early return is the
    # append contract, not what this check is about)
    fov_r, hybe_r, err = preprocess.convert_dax_to_h5_worker(2, recs[1], dax_dir, store, 'DNA', overwrite=True,
                                                            dax_index=1)
    check('the unmapped index is a missing file, reported not raised', err == 'FileNotFoundError', str(err))

    print('worker: unbuffered layout row is byte-for-byte the old behaviour')
    write_dax(os.path.join(dax_dir, 'Hyb_003', 'ConvZscan_00.dax'), list(range(1, 15)))
    fov_r, hybe_r, err = preprocess.convert_dax_to_h5_worker(1, recs[2], dax_dir, store, 'DNA')
    with h5py.File(paths.stack_path(store, 1, 'Hyb_003'), 'r') as f:
        check('14 frames / 2 channels = 7 planes, plane 0 = frame 0',
              err is None and f['/stack/ch750'].shape[-1] == 7 and int(f['/stack/ch750'][0, 0, 0]) == 1)

    print('worker: a frame count the buffers leave odd is refused, not truncated')
    odd = dict(recs[0]); odd['folder'] = 'Hyb_004'; odd['buffer_frames_end'] = 2    # 14 - 3 - 2 = 9
    write_dax(os.path.join(dax_dir, 'Hyb_004', 'ConvZscan_00.dax'), values)
    fov_r, hybe_r, err = preprocess.convert_dax_to_h5_worker(1, odd, dax_dir, store, 'DNA')
    check('error names the arithmetic', err is not None and 'not a multiple' in err, str(err))
    check('nothing published for it', not os.path.exists(paths.stack_path(store, 1, 'Hyb_004')))
finally:
    shutil.rmtree(root, ignore_errors=True)

print(f'\n{len(PASS)} passed, {len(FAIL)} failed' + (f': {FAIL}' if FAIL else ''))
sys.exit(1 if FAIL else 0)
