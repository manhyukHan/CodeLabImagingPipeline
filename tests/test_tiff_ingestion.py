"""
From-TIFF ingestion, end to end on fabricated data -- the CURRENT
microscope convention: one file per (FOV, channel) per trial,
{opener}_Pos{fov}__{job}_{index}_RAW_ch{cc:02d}.tif, pages round-major.

Pins: the readout-code grammar (numeric AND letter forms, strict
edges), repo-convention folder naming, filename DISCOVERY (the
acquisition index is matched, never constructed; duplicates raise),
the page-slicing math, z-LAST store orientation, the shared atomic
publish (stack + MIP flag), completeness-gated append, loud refusal of
short files, the generated ExperimentLayout XLSX round-tripping through
parse_experiment_layout, 3-channel handling, and layout-ordered
readout-channel resolution.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_tiff_ingestion.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py                                                    # noqa: E402
import numpy as np                                             # noqa: E402
import tifffile as tf                                          # noqa: E402

from codelab_pipeline.io import paths, preprocess              # noqa: E402
from codelab_pipeline.io import tiff_ingestion as ti           # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}'
          + (f'   [{detail}]' if detail and not cond else ''))


print('the readout-code grammar')
check('n -> Hyb, -n -> Rep, 1000+n -> Toe, 2000+n -> barcode Hyb',
      [ti.folder_for(c) for c in (7, -104, 1018, 2130)] == [
          ('Hyb_007', 'H', 7), ('Rep_104', 'R', 104),
          ('Toe_018', 'T', 18), ('Hyb_130', 'B', 130)])
check('0 is the skip sentinel', ti.folder_for(0) is None)
for bad in (1000, 2000, 3000):
    try:
        ti.parse_readout_code(bad)
        check(f'{bad} rejected', False)
    except ValueError:
        check(f'{bad} rejected (the SG parser quietly mangled it)', True)
check('letters and numbers mix freely: "1-3 r5 R10 t10 B130 b131 h7 0"',
      ti.parse_readout_codes('1-3 r5 R10 t10 B130 b131 h7 0')
      == [1, 2, 3, -5, -10, 1010, 2130, 2131, 7, 0])
check('letter forms name the same folders as numeric ones',
      [ti.folder_for(c) for c in ti.parse_readout_codes('r5 t10 b130')]
      == [('Rep_005', 'R', 5), ('Toe_010', 'T', 10), ('Hyb_130', 'B', 130)])
try:
    ti.parse_readout_codes('t1001')
    check('letter with out-of-range number rejected', False)
except ValueError:
    check('letter with out-of-range number rejected', True)
check('letter ranges expand: r104-110 == r104-r110 == R104..R110',
      ti.parse_readout_codes('r104-110')
      == ti.parse_readout_codes('r104-r110')
      == [-n for n in range(104, 111)]
      and ti.parse_readout_codes('t10-12') == [1010, 1011, 1012])
try:
    ti.parse_readout_codes('-104-110')
    check('leading-minus range rejected with the r-form hint', False)
except ValueError as e:
    check('leading-minus range rejected with the r-form hint',
          'r104-110' in str(e))
try:
    ti.parse_readout_codes('r104-t110')
    check('mixed-letter range endpoints rejected', False)
except ValueError:
    check('mixed-letter range endpoints rejected', True)
check('bare NAMES are hybe rounds: DAPI -> Hyb_DAPI, DataType H',
      ti.parse_readout_codes('1-2 DAPI SRRM1') == [1, 2, 'DAPI', 'SRRM1']
      and ti.folder_for('DAPI') == ('Hyb_DAPI', 'H', 0)
      and ti.folder_for('DAPI', {'Hyb_DAPI': 201}) == ('Hyb_DAPI', 'H', 201))
try:
    ti.parse_readout_codes('DA__PI')
    check('a name unusable as a folder is rejected', False)
except ValueError:
    check('a name unusable as a folder is rejected', True)

print('\nfabricated channel-decomposed TIFFs')
root = tempfile.mkdtemp(prefix='tiffingest_')
try:
    DEPTH, CH = 5, [555, 647, 488]
    C, Z, H, W = len(CH), DEPTH, 16, 16
    OPENER, JOB = 'M_and_F', 'Job 2'
    exp_dir = os.path.join(root, 'experiment')
    trials = [('trialA', [200, 1, 2]), ('trialB', [0, 3, -1])]
    rng = np.random.default_rng(0)
    truth = {}          # (trial, fov, round, cc) -> (Z, H, W) stack
    idx = 150           # the microscope's own acquisition counter
    for tname, codes in trials:
        tdir = os.path.join(exp_dir, tname)
        os.makedirs(tdir)
        for fov in (1, 2):
            idx += 1
            for cc in range(C):
                arr = rng.integers(0, 60000, (len(codes) * Z, H, W),
                                   dtype=np.uint16)
                for r in range(len(codes)):
                    truth[(tname, fov, r, cc)] = arr[r * Z:(r + 1) * Z]
                tf.imwrite(os.path.join(
                    tdir, f'{OPENER}_Pos{fov:02d}__{JOB}_{idx}_RAW_'
                          f'ch{cc:02d}.tif'), arr,
                    photometric='minisblack')
    check('trials discovered', ti.discover_trials(exp_dir)
          == ['trialA', 'trialB'])
    # content-driven discovery (the reported case): junk subdirs are
    # filtered, and a TRIAL dir pointed at directly offers ITSELF
    os.makedirs(os.path.join(exp_dir, 'trialA', 'Bleach'))
    check('with opener+job, only match-bearing dirs are offered',
          ti.discover_trials(exp_dir, OPENER, JOB) == ['trialA', 'trialB'])
    check('a trial dir pointed at directly offers itself, junk filtered',
          ti.discover_trials(os.path.join(exp_dir, 'trialA'), OPENER, JOB)
          == ['.'])
    check('FOVs discovered by pattern (index matched, never constructed)',
          ti.discover_fovs(os.path.join(exp_dir, 'trialA'), OPENER, JOB)
          == [1, 2])
    files = ti.discover_files(os.path.join(exp_dir, 'trialA'), OPENER, JOB)
    check('per-FOV channel slots complete', sorted(files[1]) == [0, 1, 2])
    # a duplicate (fov, channel) claim must raise, not pick silently
    dup = os.path.join(exp_dir, 'trialA',
                       f'{OPENER}_Pos01__{JOB}_999_RAW_ch00.tif')
    shutil.copyfile(os.path.join(exp_dir, 'trialA', files[1][0]), dup)
    try:
        ti.discover_files(os.path.join(exp_dir, 'trialA'), OPENER, JOB)
        check('ambiguous acquisition index raises', False)
    except ValueError:
        check('ambiguous acquisition index raises', True)
    os.remove(dup)

    print('\nlayout synthesis + XLSX round-trip')

    def spec_for(t, c, **kw):
        base = {'path': os.path.join(exp_dir, t), 'codes': c,
                'modality': 'DNA', 'opener': OPENER, 'job_name': JOB,
                'channels': CH, 'depth': DEPTH, 'fiducial_channel': 555}
        base.update(kw)
        return base

    specs = [spec_for(t, c) for t, c in trials]
    by_mod = ti.synthesize_hybe_records(specs)
    records = by_mod['DNA']
    check('one record per non-skip round, acquisition order',
          list(by_mod) == ['DNA'] and [r['folder'] for r in records]
          == ['Hyb_200', 'Hyb_001', 'Hyb_002', 'Hyb_003', 'Rep_001'])
    check('3 channels carried, total_frames = depth * channels',
          records[0]['channels'] == [555, 647, 488]
          and records[0]['total_frames'] == DEPTH * 3)
    try:
        ti.synthesize_hybe_records([spec_for('trialA', [200, 1, 2]),
                                    spec_for('x', [1])])
        check('duplicate round across trials rejected', False)
    except ValueError:
        check('duplicate round across trials rejected', True)
    # per-trial MODALITY: automatic grouping into one layout per
    # modality; the SAME folder in different modalities is legitimate
    # (the cross-modal bridge hybe), with independent HybNum sequences
    mixed = ti.synthesize_hybe_records(
        [spec_for('trialA', [200, 1, 2]),
         spec_for('trialB', [1, 5], modality='RNA', channels=[555, 635],
                  fiducial_channel=555)])
    check('per-trial modality groups into separate layouts',
          sorted(mixed) == ['DNA', 'RNA']
          and [r['folder'] for r in mixed['RNA']] == ['Hyb_001', 'Hyb_005']
          and mixed['RNA'][0]['hybe_num'] == 1
          and mixed['RNA'][0]['channels'] == [555, 635]
          and [r['folder'] for r in mixed['DNA']]
          == ['Hyb_200', 'Hyb_001', 'Hyb_002'])
    # named rounds: Readouts auto-assigned LAST (max numeric + 1, +2 in
    # acquisition order), name kept in rnaNames; numeric Readouts stay
    # the TYPED numbers verbatim -- never sorted or re-derived
    named_mod = ti.synthesize_hybe_records(
        [spec_for('trialA', [200, 'DAPI', 3, 'SRRM1'])])['DNA']
    check('named rounds get auto-last Readouts, names kept',
          [(r['folder'], r['readout_id'], r['readout_name'])
           for r in named_mod]
          == [('Hyb_200', 200, None), ('Hyb_DAPI', 201, 'DAPI'),
              ('Hyb_003', 3, None), ('Hyb_SRRM1', 202, 'SRRM1')])
    named_xlsx = ti.write_layout_xlsx(
        named_mod, os.path.join(root, 'named_layout.xlsx'))
    named_parsed = preprocess.parse_experiment_layout(named_xlsx)
    check('named rounds round-trip through the XLSX with rnaNames',
          [(p['folder'], p['readout_id'], p['readout_name'])
           for p in named_parsed]
          == [('Hyb_200', 200, None), ('Hyb_DAPI', 201, 'DAPI'),
              ('Hyb_003', 3, None), ('Hyb_SRRM1', 202, 'SRRM1')])
    dp = os.path.join(root, 'proj')
    paths.write_manifest(dp, ['DNA'])
    xlsx = ti.write_layout_xlsx(records,
                                os.path.join(dp, 'DNA_ExperimentLayout.xlsx'))
    parsed = preprocess.parse_experiment_layout(xlsx)
    check('generated XLSX parses through parse_experiment_layout',
          [p['folder'] for p in parsed] == [r['folder'] for r in records]
          and parsed[0]['channels'] == [555, 647, 488]
          and parsed[4]['datatype'] == 'R'
          and parsed[0]['total_frames'] == DEPTH * 3)

    print('\nconversion: slicing math, z-last, atomic publish')
    sp = os.path.join(dp, 'DNA')
    for fov in (1, 2):
        for spec in specs:
            res = ti.convert_tiff_trial_fov_worker(fov, spec, sp, 'DNA')
            assert all(e is None for _f, _h, e in res), res
    with h5py.File(paths.stack_path(sp, 1, 'Hyb_001'), 'r') as f:
        d = f['/stack/ch647'][:]
        check('stack is z-LAST (H, W, depth)', d.shape == (H, W, Z))
        # trialA codes [200, 1, 2] -> Hyb_001 is round index 1; ch647 slot 1
        want = truth[('trialA', 1, 1, 1)].transpose(1, 2, 0)
        check('page-slicing math reproduces the exact frames',
              np.array_equal(d, want))
        check('attrs carry the record identity',
              f.attrs['datatype'] == 'H' and f.attrs['readout_id'] == 1
              and f.attrs['expected_depth'] == Z)
        check('MIP matches the stack max',
              np.array_equal(f['/mip/ch488'][:],
                             f['/stack/ch488'][:].max(axis=-1)))
    check('skip round wrote nothing',
          not os.path.exists(paths.stack_path(sp, 1, 'Hyb_000')))
    check('repeat round landed as Rep_001',
          os.path.exists(paths.stack_path(sp, 1, 'Rep_001')))
    check('MIP flag files present (ingestion-complete flags)',
          {'Hyb_200', 'Hyb_001', 'Hyb_002'} <= paths.mips_present(sp, 1))
    from codelab_pipeline.io import analysis_store
    rmip = analysis_store.readout_channel_mip(sp, 1, 'Hyb_001')
    with h5py.File(paths.mip_path(sp, 1, 'Hyb_001'), 'r') as f:
        check('readout_channel_mip follows layout order, not ASCII',
              np.array_equal(rmip, f['ch647'][:])
              and not np.array_equal(rmip, f['ch488'][:]))
        # channel_mip: the ONE resolver behind every alignment layer's
        # channel choice -- roles and concrete channels alike
        check('channel_mip resolves roles and concrete channels',
              np.array_equal(analysis_store.channel_mip(sp, 1, 'Hyb_001',
                                                        'fiducial'),
                             f['ch555'][:])
              and np.array_equal(analysis_store.channel_mip(sp, 1, 'Hyb_001',
                                                            '488'),
                                 f['ch488'][:])
              and np.array_equal(analysis_store.channel_mip(sp, 1, 'Hyb_001',
                                                            '405'),
                                 f['ch647'][:]))   # absent -> readout rule

    print('\nappend mode: completeness-gated, per round')
    reads_before = os.path.getmtime(paths.stack_path(sp, 1, 'Hyb_001'))
    res = ti.convert_tiff_trial_fov_worker(1, specs[0], sp, 'DNA',
                                           overwrite=False)
    check('append run skips complete rounds (no rewrite)',
          all(e is None for _f, _h, e in res)
          and os.path.getmtime(paths.stack_path(sp, 1, 'Hyb_001'))
          == reads_before)
    victim = paths.stack_path(sp, 1, 'Hyb_002')
    raw = open(victim, 'rb').read()
    open(victim, 'wb').write(raw[:len(raw) // 2])
    res = ti.convert_tiff_trial_fov_worker(1, specs[0], sp, 'DNA',
                                           overwrite=False,
                                           rounds=[2])   # per-round subset
    with h5py.File(victim, 'r') as f:
        healed = np.array_equal(
            f['/stack/ch555'][:], truth[('trialA', 1, 2, 0)].transpose(1, 2, 0))
    check('a truncated stack self-heals on append', healed)

    print('\nshort files refuse loudly')
    short_spec = dict(specs[0], codes=[200, 1, 2, 99])  # one round extra
    res = ti.convert_tiff_trial_fov_worker(1, short_spec, sp, 'DNA',
                                           overwrite=True)
    check('declared rounds beyond the file fail EVERY round, silently '
          'truncating nothing',
          all(e is not None and 'pages' in e for _f, _h, e in res))
    problems, note = ti.validate_trial(short_spec, [1, 2])
    check('validate reports the short file and stops early',
          len(problems) == 1 and 'pages' in problems[0][1]
          and 'FAILED' in note)
    problems, note = ti.validate_trial(specs[0], [1, 2, 7])
    check('validate: one page count + size checks, missing FOV named',
          problems == [(7, 'no files for this FOV')]
          and 'on FOV001 ch00' in note and '15 pages OK' in note)
    # tifffile writes its IFD chain at the END (non-interleaved), so the
    # arithmetic shortcut must REFUSE (None) rather than guess -- the
    # microscope's interleaved [data][IFD] layout is where it applies
    # (verified on the real store: 903 pages in ~1 s); the refusal is
    # what routed the note above through the honest full walk
    check('fast_page_count refuses non-interleaved layouts',
          ti.fast_page_count(os.path.join(
              exp_dir, 'trialA',
              ti.discover_files(os.path.join(exp_dir, 'trialA'),
                                OPENER, JOB)[1][0])) is None
          and 'full page walk' in note)
finally:
    shutil.rmtree(root, ignore_errors=True)

print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
if FAIL:
    for n in FAIL:
        print('  FAILED:', n)
    sys.exit(1)
print('ALL GOOD')
