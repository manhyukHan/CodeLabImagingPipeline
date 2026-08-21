import os
import re
import numpy as np
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
import h5py

from . import paths
import cv2
from scipy.optimize import minimize
import xml.etree.ElementTree as ET
from xml.sax.saxutils import quoteattr

# One convert_dax_to_h5_worker holds a whole DAX in RAM at once (read_dax
# does a single np.fromfile of the entire file) plus the channel slice it
# copies out of it -- ~4.5 GB for this project's real 2048x2048x354 uint16
# stacks. On a big box that memory, not the core count, is what actually
# bounds a pooled ingestion, so max_ingestion_workers() takes the smaller
# of the two.
DAX_WORKER_PEAK_BYTES = 5 * 1024 ** 3

# Never offer fewer than this many, whatever the arithmetic below says. The
# work is I/O-bound, so a few conversions in flight is useful even on a
# 2-core laptop, and 4 is what the spinbox already defaulted to for
# everyone -- a spec-derived ceiling that took a small machine BELOW its
# own current default would be a regression dressed up as a fix.
MIN_WORKER_CEILING = 4

# Fallback memory limit for a machine whose RAM we cannot measure. Same
# number the spinbox was hard-coded to before it became spec-derived, so an
# unmeasurable machine keeps exactly the old bound rather than a cap
# nothing has verified it can survive.
UNKNOWN_RAM_WORKER_CEILING = 16


def total_ram_bytes():
    """
    Physical RAM in bytes, or None if this machine will not say.

    psutil first (present in the usual conda env but NOT in
    requirements.txt, so it has to stay optional), then the POSIX sysconf
    pair, then the Win32 call. Every path is read-only and cheap.
    """
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    try:
        return int(os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE'))
    except (ValueError, AttributeError, OSError):
        pass
    try:
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong),
                        ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong),
                        ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong),
                        ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong),
                        ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
    except Exception:
        pass
    return None


def max_ingestion_workers(hard_ceiling=64):
    """
    How many DAX->H5 conversion processes this machine can actually host.

    Was a flat 16, which quietly wasted a workstation -- the 64-logical-core
    / 352 GB box this was found on could not be asked for a 17th worker --
    and was equally happy to offer a 4-core laptop more workers than it has
    cores. Two independent limits, whichever binds first:

      cores  -- os.cpu_count() minus two, leaving room for the coordinator
                QThread that does the vlinks.h5 MIP writes and for the GUI
                to stay responsive while a long ingestion runs.
      memory -- 60% of physical RAM divided by DAX_WORKER_PEAK_BYTES. The
                40% held back is for the OS file cache, which is doing real
                work during an ingestion, and for the rest of the app
                (a loaded cell container is not small).

    then floored at MIN_WORKER_CEILING and capped at hard_ceiling, so no
    machine is offered less than the old default nor a number so large the
    spinbox stops meaning anything.

    This is the ceiling the user may dial up to, not a recommendation: the
    work is I/O-bound, so the useful setting is usually well below it and
    depends on the storage -- a network share saturates long before a local
    NVMe does.
    """
    cores = os.cpu_count() or 4
    cap = max(1, cores - 2)

    ram = total_ram_bytes()
    if ram:
        cap = min(cap, int(ram * 0.6) // DAX_WORKER_PEAK_BYTES)
    else:
        cap = min(cap, UNKNOWN_RAM_WORKER_CEILING)

    return min(max(cap, MIN_WORKER_CEILING), hard_ceiling)


def create_or_replace_dataset(group, name, data, dtype):
    """
    Create or replace a dataset in an HDF5 group.
     If the dataset with the given name already exists, it will be deleted and replaced.
     
     Args:
        group (h5py.Group): The HDF5 group where the dataset will be created.
        name (str): The name of the dataset.
        data (array-like): The data to be stored in the dataset.
        dtype (str or numpy.dtype): The data type of the dataset.
    """
    if name in group:
        del group[name]
    group.create_dataset(name, data=data, dtype=dtype)


_LAYOUT_CACHE = {}


def parse_experiment_layout(xlsx_path):
    """
    Parse an ExperimentLayout.xlsx sheet into one record per hybe -- the
    authoritative source for which channels exist per hybe (never a fixed
    405/488/555/635 set), the fiducial channel, readout identity/datatype,
    and ingestion order (hybe_num). datatype/readout_id are informational
    only: alignment logic must never branch on them (the alignment reference
    is always an explicit input, not inferred from e.g. a 'barcode' datatype).

    Returns a list of dicts: folder, readout_id, datatype, hybe_num,
    channels (list[int]), fiducial_channel (int), channel_layout (str),
    total_frames (int), readout_name (str or None -- DNA layouts have no
    rnaNames column, so target rounds are anonymous).
    """
    # mtime-keyed cache: the SAME layout is re-parsed inside several
    # doors per click (~46 ms at 16 hybes, ~300 ms at the real 111 --
    # per call, over NAS). Copies out so a caller mutating its records
    # cannot poison later parses.
    try:
        key = (os.path.abspath(xlsx_path), os.stat(xlsx_path).st_mtime_ns)
    except OSError:
        key = None
    if key is not None and key in _LAYOUT_CACHE:
        return [dict(r) for r in _LAYOUT_CACHE[key]]
    df = pd.read_excel(xlsx_path)
    has_names = 'rnaNames' in df.columns
    records = []
    for _, row in df.iterrows():
        channels = [int(c.strip()) for c in str(row['channels']).strip('[]').split(',')]
        readout_name = str(row['rnaNames']) if has_names and pd.notna(row['rnaNames']) else None
        records.append({
            'folder': str(row['FolderName']),
            'readout_id': int(row['Readouts']),
            'datatype': str(row['DataType']),
            'hybe_num': int(row['HybNum']),
            'channels': channels,
            'fiducial_channel': int(row['fiducialChannel']),
            'channel_layout': str(row['channelLayout']),
            'total_frames': int(row['totalFrames']),
            'readout_name': readout_name,
        })
    if key is not None:
        _LAYOUT_CACHE.clear()          # one layout per modality in practice; keep it tiny
        _LAYOUT_CACHE[key] = records
        return [dict(r) for r in records]
    return records

def make_xml_file(config, save_path):
    """
    Persist this app's config as a <settings> root with one <modality>
    child per modality -- a real multi-layer structure (not hardcoded to
    exactly RNA/DNA) since some of what a modality carries is genuinely
    per-modality (layout_path/dax_directory/storage_path, its own within-
    experiment reference_hybe + same_modality_channel_type, and its own
    cross_modality_reference_hybe -- the hybe THIS modality uses as the
    cross-modal bridge point), while other settings are genuinely global
    (num_modalities, fov_list, cross_modality_channel_type, cell_align_reference_hybe,
    cell_align_channel_type, cell_seg_fov). Deliberately excludes
    cell_seg_reference_hybe/cell_seg_channel/cell_seg_method -- those
    describe whatever a real segmentation run actually did (persisted in
    vlinks.h5), not something an external config should be dictating.

    config: {'global': {key: value}, 'modalities': {name: {key: value}}}.
    list/tuple values are comma-joined; everything else is str()'d.
    """
    root = ET.Element('settings')
    for key, value in config.get('global', {}).items():
        if isinstance(value, (list, tuple)):
            root.set(key, ','.join(str(v) for v in value))
        else:
            root.set(key, str(value))
    for name, fields in config.get('modalities', {}).items():
        elem = ET.SubElement(root, 'modality')
        elem.set('name', str(name))
        for key, value in fields.items():
            elem.set(key, str(value))
    with open(save_path, 'w') as f:
        f.write('\n'.join(_render_element(root)) + '\n')

def _render_element(elem, depth=0):
    """
    Serialize `elem` with one attribute per line (readable diffs/manual
    editing for wide elements like <modality ...>) instead of
    ElementTree's default single-line-per-tag output. Still plain,
    parseable XML -- whitespace between attributes is legal XML, so
    load_xml_file's ET.parse-based reading is unaffected.
    """
    indent = '  ' * depth
    tag_indent = '  ' * (depth + 1)
    items = list(elem.attrib.items())
    children = list(elem)
    lines = [f'{indent}<{elem.tag}']
    for i, (key, value) in enumerate(items):
        line = f'{tag_indent}{key}={quoteattr(str(value))}'
        if i == len(items) - 1:
            line += '>' if children else ' />'
        lines.append(line)
    if not items:
        lines[0] += '>' if children else ' />'
    if children:
        for child in children:
            lines.extend(_render_element(child, depth + 1))
        lines.append(f'{indent}</{elem.tag}>')
    return lines

def load_xml_file(file_path):
    """
    Inverse of make_xml_file -- {'global': {key: str}, 'modalities':
    {name: {key: str}}}, whatever keys/modalities the file happens to
    have (older/narrower files just come back with fewer of them; callers
    use .get(key, default)). global['fov_list'], if present, is parsed
    into list[int] -- comma AND/OR whitespace separated, matching
    windows/main_window.py's own _parse_fov_list convention (a config
    field shouldn't be pickier about separators than the UI field it
    feeds).
    """
    root = ET.parse(file_path).getroot()
    cfg = {'global': dict(root.attrib), 'modalities': {}}
    if 'fov_list' in cfg['global']:
        cfg['global']['fov_list'] = [int(f) for f in re.split(r'[,\s]+', cfg['global']['fov_list'].strip()) if f.strip()]
    for elem in root.findall('modality'):
        name = elem.get('name')
        fields = dict(elem.attrib)
        fields.pop('name', None)
        cfg['modalities'][name] = fields
    return cfg

def read_dax(filename, matlab=False):
    """
    Parse a .dax + .inf pair into a (height, width, frames) array. Ported
    from scripts/utils.py -- validated against real RNA_Expt/DNA_Expt .dax
    files (shape and non-degenerate signal confirmed against
    ExperimentLayout.xlsx's totalFrames/channels for the same hybes).
    """
    if filename.endswith('.dax'):
        daxname = filename
        infofile = filename.replace('.dax', '.inf')
    elif filename.endswith('.inf'):
        infofile = filename
        daxname = filename.replace('.inf', '.dax')
    else:
        daxname = filename + '.dax'
        infofile = filename + '.inf'

    width, height, frames = 0, 0, 0
    dtype = np.uint16

    with open(infofile, 'r') as f:
        for line in f:
            if 'frame dimensions' in line:
                width, height = map(int, line.split('=')[-1].split('x'))
            elif 'number of frames' in line:
                frames = int(line.split('=')[-1])
            elif 'data type' in line:
                if '16 bit integers' in line:
                    dtype = np.uint16
                elif '8 bit integers' in line:
                    dtype = np.uint8
                else:
                    dtype = np.float32

    if matlab:
        with open(daxname, 'rb') as f:
            data = np.fromfile(f, dtype=dtype).reshape((height, width, frames), order='F').transpose((1, 0, 2)).squeeze()
    else:
        with open(daxname, 'rb') as f:
            data = np.fromfile(f, dtype=dtype).reshape((height, width, frames), order='F').squeeze()
    return data

def convert_dax_to_h5_worker(fov, hybe_record, dax_directory, storage_path, modality, overwrite=False):
    """
    Convert one FOV's raw .dax for one hybe into a per-(fov,hybe) H5 file,
    channel/readout/datatype-aware per parse_experiment_layout. Only the
    channels actually listed for this hybe get a dataset -- e.g. this real
    dataset is always [555, 635], so no empty 405/488 containers are made.
    Note: DAX-sourced /stack/ch{ch} is (height, width, depth) -- depth last,
    since that's what read_dax naturally produces -- unlike the TIFF path
    above, which is (depth, height, width). Any future unified reader needs
    to know which ingestion path produced a given file.
    """
    folder = hybe_record['folder']
    channels = hybe_record['channels']
    os.makedirs(os.path.dirname(paths.stack_path(storage_path, fov, folder)), exist_ok=True)
    stack_h5name = paths.stack_path(storage_path, fov, folder)

    if os.path.exists(stack_h5name):
        if not overwrite:
            return fov, folder, None
        os.remove(stack_h5name)

    # ExperimentLayout's totalFrames is the authoritative source for depth (z-plane
    # count) per hybe -- e.g. barcode hybes here have totalFrames=354 (177 z-planes
    # per channel) vs. 260 (130 z-planes) for regular hybes, so this must be read
    # per-hybe from the layout, never assumed uniform across a dataset.
    expected_depth = hybe_record['total_frames'] // len(channels)

    dax_path = os.path.join(dax_directory, folder, f'ConvZscan_{fov-1:02d}.dax')
    try:
        dax = read_dax(dax_path)
        attributes = {
            'hybe': folder,
            'fov': fov,
            'readout_id': hybe_record['readout_id'],
            'readout_name': hybe_record['readout_name'] or '',
            'datatype': hybe_record['datatype'],
            'modality': modality,
            'fiducial_channel': hybe_record['fiducial_channel'],
            'channel_list': np.array([str(ch) for ch in channels], dtype='S'),
            'total_frames': hybe_record['total_frames'],
            'expected_depth': expected_depth,
            'shape': (),
            'path': dax_path,
        }

        with h5py.File(stack_h5name, 'w') as f:
            # No /matrix group: alignment matrices are metadata and live in
            # vlinks.h5 alone (see chain.write_same_modality_matrices). A
            # stack file holds raw data plus the MIP derived from it in this
            # same pass -- nothing mutable, so nothing here can go stale.
            stack_group = f.create_group('/stack')
            mip_group = f.create_group('/mip')
            f.attrs.update(attributes)

            dat = None
            for cid, ch in enumerate(channels):
                dat = dax[:, :, cid::len(channels)]
                if dat.shape[-1] != expected_depth:
                    # Layout and actual DAX content disagree -- surface it loudly
                    # rather than silently ingesting a shape the layout didn't predict.
                    raise ValueError(f'depth mismatch for {folder} ch{ch}: DAX has '
                                     f'{dat.shape[-1]} z-planes, ExperimentLayout '
                                     f'totalFrames predicts {expected_depth}')
                # Chunked + lightly compressed, sized for the pipeline's
                # real access pattern: small-XY x deep-Z crops (3D
                # localization, chromatin tracing). On the measured NAS
                # (12.6 ms per scattered read) a contiguous stack made one
                # 17x17xZ crop ~3.6 s (289 scattered runs); (32, 32,
                # z-slab) chunks make it a handful of contiguous chunk
                # reads (~0.1 s), and partial-Z access stays partial --
                # chunks decompress independently, so only the z-slabs
                # overlapping a request are touched.
                zslab = min(dat.shape[-1], 64)
                stack_group.create_dataset(f'ch{ch}', data=dat, dtype='uint16',
                                           chunks=(32, 32, zslab),
                                           compression='gzip', compression_opts=1,
                                           shuffle=True)
                create_or_replace_dataset(mip_group, f'ch{ch}', np.max(dat, axis=-1), 'uint16')
            attributes['shape'] = dat.shape
            f.attrs.update(attributes)
        if paths.is_v2(storage_path):
            # v2: this worker writes the per-hybe MIP file itself
            # (atomically -- see vlinks_store.write_hybe_mip's v2 branch),
            # so the coordinator never touches MIPs and vlinks.h5 sees no
            # ingestion traffic at all: the UI stays live mid-ingestion
            # and each hybe becomes browsable the moment ITS file lands.
            from . import vlinks_store
            with h5py.File(stack_h5name, 'r') as f:
                channel_mips = {ch: f[f'/mip/ch{ch}'][:] for ch in channels}
            vlinks_store.write_hybe_mip(storage_path, fov, folder, channel_mips,
                                        fiducial_channel=hybe_record['fiducial_channel'])
        logging.info(f'Converted FOV {fov}, hybe {folder} ({modality})')
        return fov, folder, None
    except FileNotFoundError:
        logging.error(f'DAX file not found: {dax_path}')
        return fov, folder, 'FileNotFoundError'
    except Exception as e:
        logging.error(f'Error processing FOV {fov}, hybe {folder}: {e}')
        return fov, folder, str(e)


def normalize_to_uint8(img: np.ndarray, lb=.1, ub=.9999):
    img = img.astype(np.float32)
    lbq = np.nanquantile(img,lb) if lb < 1 else lb
    ubq = np.nanquantile(img, ub) if ub < 1 else ub    
    img = np.clip(img, lbq, ubq)
    img = (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img)) * 255
    return img.astype(np.uint8)

def pad_to_same_size(img1, img2, pad_value=0):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    H, W = max(h1, h2), max(w1, w2)

    def pad(img, target_shape):
        h, w = img.shape[:2]
        top = (target_shape[0] - h) // 2
        bottom = target_shape[0] - h - top
        left = (target_shape[1] - w) // 2
        right = target_shape[1] - w - left
        return np.pad(img, ((top, bottom), (left, right)), constant_values=pad_value), (top, left)

    padded1, offset1 = pad(img1, (H, W))
    padded2, offset2 = pad(img2, (H, W))

    return padded1, padded2, offset1, offset2


def msd_cost_function(params, moving_image, reference_image, fixed_scale=1.0, fixed_angle=False):
    """
    fixed_angle: False (default) -- angle is a free Powell parameter, same
    as always. True -- angle fixed at 0 (translation-only), the original
    behavior. A number -- angle fixed at that exact degree value (Powell
    still optimizes dx/dy under it); used to independently confirm a
    translation under a rotation estimated elsewhere (e.g. ORB's own
    angle), without letting Powell re-guess rotation (which it can't
    reliably do anyway -- see compute_msd_homography_matrix's docstring).
    """
    dx, dy, angle = params

    # Pad images
    moving_padded, reference_padded, _, _ = pad_to_same_size(moving_image, reference_image)
    h, w = moving_padded.shape[:2]
    center = (w // 2, h // 2)

    angle_to_use = angle if fixed_angle is False else (0.0 if fixed_angle is True else float(fixed_angle))
    M = cv2.getRotationMatrix2D(center, angle_to_use, fixed_scale)
    M[0, 2] += dx
    M[1, 2] += dy

    # Warp image and mask
    transformed_image = cv2.warpAffine(moving_padded, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid_mask = cv2.warpAffine(np.ones_like(moving_padded, dtype=np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Compute MSD where both mask and reference are valid
    overlap_mask = (valid_mask > 0) & (reference_padded > 0)
    if np.count_nonzero(overlap_mask) == 0:
        return np.inf

    diff = (transformed_image.astype(np.float32) - reference_padded.astype(np.float32)) ** 2
    msd = diff[overlap_mask].mean()

    return msd

def find_best_alignment(moving_image, reference_image, fixed_scale=1.0,
                        fixed_angle=False, initial_guess=[0,0,0], method='Powell', verbose=False, bounds=None):
    # Initial guess: [dx, dy, angle]
    # bounds (default None, no behavior change): [(dx_min, dx_max), (dy_min,
    # dy_max), (angle_min, angle_max)] -- scipy's Powell implementation
    # supports bounds natively (clips each line search to stay within them),
    # so passing this constrains the SEARCH itself rather than checking the
    # result after the fact. The angle bound is a no-op whenever fixed_angle
    # isn't False (msd_cost_function ignores that parameter entirely then,
    # and find_best_alignment itself never reads the optimizer's own angle
    # value in that case either -- see angle_to_use below), so it's always
    # safe to pass the same 3-tuple regardless of fixed_angle.

    # Minimize the cost function
    result = minimize(msd_cost_function, initial_guess, args=(moving_image, reference_image, fixed_scale, fixed_angle),
                      method=method, bounds=bounds)
    if verbose:
        print(f"Optimization Result: {result}")
        print(f"Success: {result.success}, Message: {result.message}")

    # Extract optimal parameters
    dx, dy, angle = result.x

    # Compute final transformation matrix with fixed scale. angle here is
    # whatever msd_cost_function actually used -- Powell's free-angle guess,
    # 0 (fixed_angle=True), or the caller's fixed numeric angle -- not the
    # raw (possibly-unused) optimizer parameter.
    angle_to_use = angle if fixed_angle is False else (0.0 if fixed_angle is True else float(fixed_angle))
    if abs(angle_to_use) > 1/2:
        h, w = moving_image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle_to_use, fixed_scale)
    else:
        M = np.eye(3)[:2].astype(float)
    M[0, 2] += dx
    M[1, 2] += dy

    return M

def find_translation_via_phase_correlation(img1, img2):
    # Ensure float32 format
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    pad = int(np.max(img1.shape) * 0.2)
    img1_padded, img2_padded,_,_ = pad_to_same_size(img1, img2, pad_value=0)
    #img1 = cv2.copyMakeBorder(img1, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    #img2 = cv2.copyMakeBorder(img2, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    # Use phase correlation to estimate shift
    shift, response = cv2.phaseCorrelate(img1_padded, img2_padded)

    # Build translation-only affine matrix
    affine_matrix = np.array([
        [1, 0, shift[0]],
        [0, 1, shift[1]]
    ], dtype=np.float32)

    return affine_matrix

def compute_msd_homography_matrix(moving_image, reference_image, fixed_scale=1.0, fixed_angle=False,
                                  initial_guess=[0,0,0], method='Powell', verbose=False, bounds=None):
    # Find best alignment
    affine_matrix = find_best_alignment(moving_image, reference_image, fixed_scale, fixed_angle, initial_guess,
                                        method, verbose, bounds=bounds)
    
    # Convert affine transformation to homography
    homography_matrix = np.array([
        [affine_matrix[0, 0], affine_matrix[0, 1], affine_matrix[0, 2]],
        [affine_matrix[1, 0], affine_matrix[1, 1], affine_matrix[1, 2]],
        [0, 0, 1]
    ])
    
    #print(f"Homography Matrix:\n{homography_matrix}")
    return homography_matrix

def compute_features_affinelike_matrix(moving_image, reference_image):
    """
    Ported from scripts/utils.py. ORB feature matching + RANSAC affine
    estimation, with the estimated 2x2 linear part re-orthogonalized via SVD
    (U @ Vt) so the result is a pure rotation+translation (no shear/skew) --
    an "affine-like" (rigid) matrix.

    Confirmed via a synthetic ground-truth test (apply a known rotation +
    translation to a real MIP, recover it) to correctly detect rotation --
    residual MSD ~280-300 after correction at both 3deg and 8deg synthetic
    rotation, vs. thousands uncorrected. compute_msd_homography_matrix above
    (Powell optimization over dx/dy/angle) was tested the same way and does
    NOT recover rotation at all, even at 8deg -- it isn't just the small-
    angle threshold in find_best_alignment, the optimizer itself converges
    to angle=0 regardless of true rotation for this kind of image. Falls
    back to the MSD method if feature matching fails (e.g. too few
    keypoints/matches for RANSAC).
    """
    try:
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(moving_image, None)
        kp2, des2 = orb.detectAndCompute(reference_image, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        H = np.eye(3)
        A, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
        U, _, Vt = np.linalg.svd(A[:2, :2])
        Afixed = U @ Vt
        H[:2, :2] = Afixed
        H[:2, 2] = A[:2, 2]
        return H
    except Exception:
        return compute_msd_homography_matrix(moving_image, reference_image)

