"""
From-TIFF ingestion: round-based, channel-decomposed TIFFs -> the same
v2 store the DAX path writes.

The raw data model (the CURRENT microscope saving convention -- the
older channel-interleaved single-file convention is gone and stays
gone, per explicit decision):

  <experiment dir>/
    <trial dir A>/  M_and_F_Pos01__Job 2_152_RAW_ch00.tif
                    M_and_F_Pos01__Job 2_152_RAW_ch01.tif
                    M_and_F_Pos01__Job 2_152_RAW_ch02.tif
                    M_and_F_Pos02__Job 2_153_RAW_ch00.tif ...

One file per (FOV, channel) per trial:
  {opener}_Pos{fov:0Nd}__{job}_{index}_RAW_ch{cc:02d}.tif
where `index` is the microscope's own acquisition counter -- it varies
per FOV and per session, so filenames are DISCOVERED by pattern, never
constructed. `cc` is the 00-based acquisition-order channel slot; the
operator supplies the channel names in that order ([555, 647, 488] ->
ch00=555). Pages inside one file are round-major: page = round*Z + z.

Which round is which lives nowhere in the file; the operator supplies
an ORDERED code list per trial, in a grammar that converts either
numbers or letter-prefixed names into this app's hybe conventions:

    n   or  H7/h7      -> Hyb_{n:03d}   DataType H
   -n   or  R5/r5      -> Rep_{n:03d}   DataType R
  1000+n or T10/t10    -> Toe_{n:03d}   DataType T
  2000+n or B130/b130  -> Hyb_{n:03d}   DataType B   (barcode)
    0                  -> skip this round (in the file, not wanted)
  a-b                  -> Hyb_a .. Hyb_b (ranges, plain hybes only)

Everything downstream is THIS app's convention: repo folder names,
z-LAST (height, width, depth) stacks, the shared publish_stack atomic
door, completeness-gated append, and a generated ExperimentLayout XLSX
so the rest of the app parses this store exactly like a DAX one. The
generated layout is the durable artifact; the raw TIFF directory, like
a DAX directory, has no role after ingestion.
"""
import logging
import os
import re

import numpy as np
import pandas as pd

from . import paths
from . import preprocess

_DTYPE_FOLDER = {'H': 'Hyb_{:03d}', 'R': 'Rep_{:03d}',
                 'T': 'Toe_{:03d}', 'B': 'Hyb_{:03d}'}


def parse_readout_code(code):
    """One numeric readout code -> ('H'|'R'|'T'|'B', number) or None
    (skip). Strict on the grammar's edges: the SG parser quietly
    produced H1000 for 1000 and B0 for 2000; both are errors here."""
    code = int(code)
    if code == 0:
        return None
    if code < 0:
        return ('R', -code)
    if 0 < code < 1000:
        return ('H', code)
    if 1000 < code < 2000:
        return ('T', code - 1000)
    if 2000 < code < 3000:
        return ('B', code - 2000)
    raise ValueError(f'readout code {code} is outside the grammar: '
                     f'n (hybe), -n (repeat), 1000+n (toe), 2000+n '
                     f'(barcode), 0 (skip); 1000/2000/>=3000 are invalid')


_LETTER_OFFSET = {'H': 0, 'R': None, 'T': 1000, 'B': 2000}


def parse_readout_codes(text):
    """A trial's ordered rounds -- numbers, letters, and NAMES freely
    mixed: '1-10 r5 R10 t10 11-30 B130 DAPI 0' -> canonical codes.
    Letters are case-insensitive (r5 == R5 -> Rep_005); ranges expand
    only between plain positive numbers; a bare NAME (starts with a
    letter, e.g. DAPI, SRRM1) is a hybe round named Hyb_{NAME} whose
    Readouts index is auto-assigned LAST at layout synthesis (per
    request: automatic, without meaning). Every code is validated."""
    def letter_code(letter, n):
        if not 0 < n < 1000:
            raise ValueError(f'{letter}{n}: the number must be 1..999')
        return -n if letter == 'R' else _LETTER_OFFSET[letter] + n

    codes = []
    for token in re.split(r'[,\s]+', str(text).strip()):
        if not token:
            continue
        # ranges, with an optional letter prefix on either end:
        # 104-110 / r104-110 / r104-r110 / t10-15 -- prefixes must agree
        m = re.fullmatch(r'([hHrRtTbB]?)(\d+)-([hHrRtTbB]?)(\d+)', token)
        if m:
            la, a, lb, b = (m.group(1).upper(), int(m.group(2)),
                            m.group(3).upper(), int(m.group(4)))
            if lb and la and lb != la:
                raise ValueError(f'{token}: range endpoints disagree '
                                 f'({la} vs {lb})')
            letter = la or lb or 'H'
            if b < a:
                raise ValueError(f'descending range {token}')
            codes.extend(letter_code(letter, n) if letter != 'H' else n
                         for n in range(a, b + 1))
            continue
        m = re.fullmatch(r'([hHrRtTbB])(\d+)', token)
        if m:
            codes.append(letter_code(m.group(1).upper(), int(m.group(2))))
            continue
        if re.fullmatch(r'-?\d+', token):
            codes.append(int(token))
            continue
        if token.startswith('-') and token.count('-') > 1:
            raise ValueError(f'{token}: a repeat RANGE is written '
                             f'r{token.lstrip("-").split("-")[0]}-'
                             f'{token.rsplit("-", 1)[1]}, not with a '
                             f'leading minus')
        # NO hyphens in names -- anything range-shaped must never be
        # silently swallowed as a name (r104-110 once was; reported)
        if re.fullmatch(r'[A-Za-z][A-Za-z0-9]*', token):
            codes.append(token)        # a named hybe round
            continue
        raise ValueError(f'{token!r} is neither a readout code nor a name')
    for c in codes:
        parse_readout_code(c) if isinstance(c, int) else None
    return codes


def folder_for(code, auto_ids=None):
    """Code -> (folder, datatype, readout_id), None for skip. A NAMED
    round (str code) is Hyb_{NAME}, DataType H; its auto-assigned
    Readouts index comes from auto_ids ({folder: id}, produced by
    synthesize_hybe_records) when given, else 0 as a placeholder."""
    if isinstance(code, str):
        folder = f'Hyb_{code}'
        return folder, 'H', (auto_ids or {}).get(folder, 0)
    parsed = parse_readout_code(code)
    if parsed is None:
        return None
    datatype, num = parsed
    return _DTYPE_FOLDER[datatype].format(num), datatype, num


def _file_pattern(opener, job_name):
    """The fixed filename shape, opener and job code gating it:
    {opener}_Pos{fov}__{job}_{index}_RAW_ch{cc}.tif"""
    return re.compile(
        re.escape(opener) + r'_Pos(\d+)__' + re.escape(job_name)
        + r'_(\d+)_RAW_ch(\d+)\.tiff?$', re.IGNORECASE)


def _has_matches(trial_dir, opener, job_name):
    pat = _file_pattern(opener, job_name)
    try:
        return any(pat.fullmatch(n) for n in os.listdir(trial_dir))
    except OSError:
        return False


def _has_tiffs(trial_dir):
    try:
        return any(n.lower().endswith(('.tif', '.tiff'))
                   for n in os.listdir(trial_dir))
    except OSError:
        return False


def discover_trials(experiment_dir, opener=None, job_name=None):
    """The trial rows the dialog offers: subdirectories holding TIFFs,
    plus '.' when the chosen directory itself does -- users point at
    either the experiment parent OR a single trial dir (reported: a
    trial dir's junk subfolders -- Bleach, MetaData -- were offered
    while its own TIFFs were not). With opener/job given, the filter
    tightens to files matching the full pattern; without them (the
    assign-parameters-later flow, where opener/job are PER TRIAL), any
    .tif at all keeps the row."""
    subdirs = sorted(d for d in os.listdir(experiment_dir)
                     if os.path.isdir(os.path.join(experiment_dir, d)))
    if opener is not None and job_name is not None:
        def keep(d):
            return _has_matches(d, opener, job_name)
    else:
        keep = _has_tiffs
    out = []
    if keep(experiment_dir):
        out.append('.')
    out.extend(d for d in subdirs
               if keep(os.path.join(experiment_dir, d)))
    return out


def discover_files(trial_dir, opener, job_name):
    """{fov: {channel_slot: filename}} for one trial, by pattern -- the
    acquisition index in the name is the microscope's own counter and
    is never constructed, only matched. Two files claiming the same
    (fov, channel slot) is a real ambiguity and raises."""
    pat = _file_pattern(opener, job_name)
    out = {}
    for name in os.listdir(trial_dir):
        m = pat.fullmatch(name)
        if not m:
            continue
        fov, cc = int(m.group(1)), int(m.group(3))
        slot = out.setdefault(fov, {})
        if cc in slot:
            raise ValueError(f'{trial_dir}: FOV {fov} ch{cc:02d} matched by '
                             f'BOTH {slot[cc]} and {name} -- ambiguous '
                             f'acquisition index; move one aside')
        slot[cc] = name
    return out


def discover_fovs(trial_dir, opener, job_name):
    return sorted(discover_files(trial_dir, opener, job_name))



# -- trial specs ---------------------------------------------------------
# A trial spec is SELF-CONTAINED (per explicit redesign: every parameter
# -- modality included -- is assignable per trial, the dialog's global
# fields being only an assigner):
#   {'path', 'codes', 'modality', 'opener', 'job_name',
#    'channels' (names for slots ch00..), 'depth', 'fiducial_channel'}

_SPEC_KEYS = ('modality', 'opener', 'job_name', 'channels', 'depth',
              'fiducial_channel')


def check_spec(spec):
    """Raise with the trial's name on anything missing/unparseable."""
    name = os.path.basename(str(spec.get('path', '?')))
    for key in _SPEC_KEYS:
        if not spec.get(key):
            raise ValueError(f'{name}: {key} is not set -- assign the '
                             f'global parameters (or fill the cell)')
    if not spec.get('codes'):
        raise ValueError(f'{name}: readout codes are empty')
    for ch in spec['channels']:
        int(ch)         # numeric channels only -- 'BF' has no store slot yet
    int(spec['fiducial_channel'])
    int(spec['depth'])
    return spec


def synthesize_hybe_records(trial_specs):
    """The layouts, from the trial table alone -- one record per
    non-skip round, grouped BY MODALITY (each modality gets its own
    generated ExperimentLayout), HybNum sequential within a modality,
    in EXACTLY the shape parse_experiment_layout produces.

    Returns {modality: [records]}. A code that names the same
    (modality, folder) twice is an error -- re-images are what R codes
    are for. The same folder in DIFFERENT modalities is legitimate (the
    cross-modal bridge hybe exists in both).

    NAMED rounds (str codes, e.g. 'DAPI' -> Hyb_DAPI) get their
    Readouts index auto-assigned LAST: max(numeric ids in the modality)
    + 1, + 2, ... in acquisition order -- deterministic, collision-free,
    and deliberately meaningless (per request); the name itself is kept
    in readout_name (the rnaNames column).
    """
    by_modality, seen = {}, {}
    for spec in trial_specs:
        check_spec(spec)
        modality = str(spec['modality'])
        records = by_modality.setdefault(modality, [])
        trial_channels = [int(ch) for ch in spec['channels']]
        for code in spec['codes']:
            named = folder_for(code)
            if named is None:
                continue
            folder, datatype, readout_id = named
            key = (modality, folder)
            if key in seen:
                raise ValueError(
                    f"round {folder} ({modality}) claimed by both "
                    f"'{seen[key]}' and '{spec['path']}' -- a re-image is "
                    f'a REPEAT (r-code), not a duplicate')
            seen[key] = spec['path']
            records.append({
                'folder': folder,
                'readout_id': None if isinstance(code, str) else readout_id,
                'datatype': datatype,
                'hybe_num': len(records) + 1,
                'channels': trial_channels,
                'fiducial_channel': int(spec['fiducial_channel']),
                'channel_layout': 'decomposed',
                'total_frames': int(spec['depth']) * len(trial_channels),
                'readout_name': code if isinstance(code, str) else None,
            })
    for records in by_modality.values():
        next_id = max([r['readout_id'] for r in records
                       if r['readout_id'] is not None] or [0]) + 1
        for r in records:
            if r['readout_id'] is None:
                r['readout_id'] = next_id
                next_id += 1
    return by_modality


def auto_ids_for(records):
    """{folder: readout_id} for one modality's records -- what the
    convert worker needs to stamp named rounds' auto-assigned indices
    into stack attrs (attach as spec['auto_ids'])."""
    return {r['folder']: r['readout_id'] for r in records}


def write_layout_xlsx(records, xlsx_path):
    """One modality's generated ExperimentLayout -- the durable
    description of this store, column-for-column what the real layouts
    carry, so parse_experiment_layout reads it with zero special-casing."""
    rows = []
    for r in records:
        rows.append({
            'FolderName': r['folder'],
            'Readouts': r['readout_id'],
            'DataType': r['datatype'],
            'HybNum': r['hybe_num'],
            'channels': '[' + ', '.join(str(c) for c in r['channels']) + ']',
            'fiducialChannel': r['fiducial_channel'],
            'channelLayout': r['channel_layout'],
            'bufferFrames': 0,
            'totalFrames': r['total_frames'],
            'FirstFrameActive': True,
        })
    df = pd.DataFrame(rows)
    if any(r.get('readout_name') for r in records):
        # None (not '') for unnamed rounds: parse_experiment_layout maps
        # NaN back to None, while an empty STRING would survive as ''
        df['rnaNames'] = [r.get('readout_name') or None for r in records]
    tmp = xlsx_path + '.part.xlsx'      # pandas needs a real xlsx suffix
    df.to_excel(tmp, index=False)
    os.replace(tmp, xlsx_path)
    return xlsx_path


def fast_page_count(path):
    """Page count WITHOUT walking the IFD chain: these microscope files
    are uniform uncompressed pages, so two IFD offsets give the
    per-page byte stride and (file_size - first_offset)/stride + 1 IS
    the page count (data precedes its IFD, hence the +1 -- verified
    against a real file: 903.0006 -> 903). Measured ~1 s over the NAS
    vs ~60 s for the full chain walk. Returns int, or None when the
    file is compressed or non-uniform (fractional estimate) -- callers
    fall back to the honest walk then.
    """
    import tifffile as tf
    with tf.TiffFile(path) as h:
        p0 = h.pages[0]
        if p0.compression != 1:
            return None
        try:
            p1 = h.pages[1]
        except IndexError:
            return 1
        stride = p1.offset - p0.offset
    if stride <= 0:
        return None
    n = (os.path.getsize(path) - p0.offset) / stride + 1
    return round(n) if abs(n - round(n)) < 0.1 else None


def validate_trial(spec, fovs, progress=None):
    """The dry run for ONE trial across many FOVs, at listdir cost.

    File presence per (FOV, channel) is pure discovery (one listdir).
    The rounds-x-depth declaration is proven against ONE reference file
    via fast_page_count (~1 s; the full IFD walk it replaces was ~60 s
    and an earlier draft paid it per FOV -- reported, rightly, as a
    complete waste). Every other file is then checked by byte size
    against the reference: pages are uniform, so a file short by even
    one round differs by ~1/rounds of its size, far beyond the 2%
    tolerance. Returns (problems, note).
    """
    import tifffile as tf
    check_spec(spec)
    C, Z = len(spec['channels']), int(spec['depth'])
    expected = len(spec['codes']) * Z
    by_fov = discover_files(spec['path'], spec['opener'], spec['job_name'])
    problems = []
    ref_size = None
    note = 'no FOV had a complete file set -- nothing to count pages on'
    for fov in fovs:
        slots = by_fov.get(fov, {})
        missing = [cc for cc in range(C) if cc not in slots]
        if missing:
            problems.append((fov, 'no files for this FOV'
                             if len(missing) == C else
                             f'missing channel file(s) '
                             f'{["ch%02d" % c for c in missing]}'))
            continue
        first = os.path.join(spec['path'], slots[0])
        if ref_size is None:
            n_pages = fast_page_count(first)
            how = 'fast page-count'
            if n_pages is None:         # compressed/non-uniform: be honest
                with tf.TiffFile(first) as handle:
                    n_pages = len(handle.pages)
                how = 'full page walk (file not uniform)'
            if n_pages != expected:
                inferred = (f'; pages/rounds = '
                            f'{n_pages / len(spec["codes"]):g}'
                            if n_pages % len(spec['codes']) == 0 else '')
                problems.append((fov, f'{slots[0]}: {n_pages} pages != '
                                      f'{expected} expected '
                                      f'({len(spec["codes"])} rounds '
                                      f'x {Z} z){inferred}'))
                return problems, (f'{how} on FOV{fov:03d} ch00 FAILED -- '
                                  f'fix the codes/depth before anything '
                                  f'else is worth checking')
            ref_size = os.path.getsize(first)
            note = (f'{how} on FOV{fov:03d} ch00: {n_pages} pages OK; '
                    f'every other file size-checked against it')
        for cc in range(C):
            path = os.path.join(spec['path'], slots[cc])
            size = os.path.getsize(path)
            if abs(size - ref_size) > 0.02 * ref_size:
                problems.append((fov, f'{slots[cc]}: byte size differs from '
                                      f'the reference by more than 2% '
                                      f'({size} vs {ref_size}) -- likely '
                                      f'short or truncated'))
        if progress is not None:
            progress(fov)
    return problems, note


def convert_tiff_trial_fov_worker(fov, spec, storage_path, modality,
                                  overwrite=False, rounds=None):
    """Rounds of one trial for one FOV. rounds: round INDICES within the
    trial's code list to process (None = all) -- the dialog submits
    PER-ROUND tasks so progress flows continuously instead of arriving
    in whole-trial bursts minutes apart (reported: 'preparation is
    really huge and I cannot notice the progression'); the extra file
    open per round is ~1 s against a minute-scale round read.

    Returns [(fov, folder, err)] per processed non-skip round, the same
    per-round contract as the DAX worker, so completeness/append/GUI-
    readiness all behave identically. Completeness is checked per round
    BEFORE its pages are read (tifffile pages are lazy).
    """
    import tifffile as tf
    trial_channels = [int(ch) for ch in spec['channels']]
    C, Z = len(trial_channels), int(spec['depth'])
    fiducial_channel = int(spec['fiducial_channel'])
    auto_ids = spec.get('auto_ids')
    named = [(r, folder_for(code, auto_ids))
             for r, code in enumerate(spec['codes'])
             if (rounds is None or r in rounds)]
    named = [(r, n) for r, n in named if n is not None]
    try:
        slots = discover_files(spec['path'], spec['opener'],
                               spec['job_name']).get(fov, {})
    except ValueError as e:
        return [(fov, folder, str(e)) for _r, (folder, _dt, _rid) in named]
    missing = [cc for cc in range(C) if cc not in slots]
    if missing:
        return [(fov, folder,
                 f'missing channel file(s) {["ch%02d" % c for c in missing]}')
                for _r, (folder, _dt, _rid) in named]

    results = []
    handles = {}
    try:
        for r, (folder, datatype, readout_id) in named:
            stack_h5name = paths.stack_path(storage_path, fov, folder)
            os.makedirs(os.path.dirname(stack_h5name), exist_ok=True)
            if os.path.exists(stack_h5name) and not overwrite:
                if preprocess.stack_is_complete(stack_h5name, trial_channels, Z):
                    if preprocess._restore_missing_mip(
                            storage_path, fov, folder, trial_channels,
                            fiducial_channel):
                        results.append((fov, folder, None))
                        continue
                    logging.warning(f'FOV {fov} {folder}: complete stack but '
                                    f'its MIP could not be restored -- '
                                    f'rebuilding despite append mode')
                else:
                    logging.warning(f'FOV {fov} {folder}: existing stack is '
                                    f'incomplete or unreadable -- rebuilding')
            channel_arrays = {}
            for cc, ch in enumerate(trial_channels):
                if cc not in handles:
                    handles[cc] = tf.TiffFile(
                        os.path.join(spec['path'], slots[cc]))
                    n_pages = len(handles[cc].pages)
                    if n_pages < len(spec['codes']) * Z:
                        # refuse the whole file rather than silently
                        # slicing short (the SG engine's failure mode)
                        raise ValueError(
                            f'{slots[cc]}: {n_pages} pages, but the trial '
                            f'declares {len(spec["codes"])} rounds x '
                            f'{Z} z = {len(spec["codes"]) * Z}')
                pages = handles[cc].pages[r * Z:(r + 1) * Z]
                stack = np.stack([p.asarray() for p in pages], axis=0)
                # z-FIRST off the TIFF; this store is z-LAST everywhere
                channel_arrays[ch] = np.ascontiguousarray(
                    stack.transpose(1, 2, 0))
            attributes = {
                'hybe': folder,
                'fov': fov,
                'readout_id': readout_id,
                'readout_name': '',
                'datatype': datatype,
                'modality': modality,
                'fiducial_channel': fiducial_channel,
                'channel_list': np.array([str(c) for c in trial_channels],
                                         dtype='S'),
                'total_frames': C * Z,
                'expected_depth': Z,
                'shape': next(iter(channel_arrays.values())).shape,
                'path': os.path.join(spec['path'], slots[0]),
            }
            err = preprocess.publish_stack(
                stack_h5name, attributes, channel_arrays, storage_path,
                fov, folder, fiducial_channel)
            results.append((fov, folder, err))
    except BaseException as e:
        logging.error(f'Error processing FOV {fov} trial '
                      f'{spec["path"]}: {e}')
        if not isinstance(e, Exception):
            raise
        done = {f for _fov, f, _e in results}
        results.extend((fov, folder, str(e))
                       for _r, (folder, _dt, _rid) in named
                       if folder not in done)
    finally:
        for h in handles.values():
            h.close()
    return results
