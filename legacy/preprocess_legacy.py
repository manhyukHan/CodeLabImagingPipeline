"""
Dead code -- see legacy/README.md. Batch/CLI-era ingestion and alignment
entry points from codelab_pipeline/io/preprocess.py, superseded by either
the GUI's own per-item QThread worker loops (windows/main_window.py) or by
compute_features_affinelike_matrix (still live in preprocess.py). Moved
here verbatim (unmodified) on 2026-08-07.

Includes the entire TIFF-ingestion path -- this pipeline only ever ingests
DAX in practice (see preprocess.convert_dax_to_h5_worker, still live).
"""
import os
import sys
import logging
logging.basicConfig(level=logging.INFO)
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

import numpy as np
import tifffile as tf
import h5py
import cv2

import matplotlib.pyplot as plt

from codelab_pipeline.io.preprocess import (
    create_or_replace_dataset,
    normalize_to_uint8,
    pad_to_same_size,
    compute_msd_homography_matrix,
)

fontdict_label = {'fontname': 'arial',
                  'fontsize':16,
                  }


def parse_readout(ro):
    """
    Parse readout code into string format.
    T: Toe readout (e.g., T1, T2, ...)
    R: Repeat readout (e.g., R1, R2, ...)
    H: Hybridization readout (e.g., H1, H2, ...)
    B: Barcode readout (e.g., B1, B2, ...)
    """
    if (ro > 1000) and (ro < 2000): return f'T{ro-1000}'
    elif (ro >= 2000) and (ro < 3000): return f'B{ro-2000}'
    elif ro < 0 : return f'R{-1*ro}'
    elif ro > 0 : return f'H{ro}'
    else: return None

def reorder_hybe_list(raw_hybe_list):
    hybe_list = []
    for h in ['H','R','T','B']:
        sub_hybe_list = [int(rh[1:]) for rh in raw_hybe_list if rh.startswith(h)]
        sub_hybe_list.sort()
        hybe_list += [f'{h}{sh}' for sh in sub_hybe_list]
    return hybe_list

def reorder_hybe_list_frompath(path):
    raw_hybe_list = [l.split('_')[0] for l in os.listdir(os.path.join(path))]
    return reorder_hybe_list(raw_hybe_list)

def convert_tiff_to_h5_worker(trial_path, readouts, fov, channel_list, fiducial_channel,
                               depth, storage_path, template, job_name,):
    """
    Convert a single TIFF file to HDF5 format for a specific FOV and readouts.

    Args:
        trial_path (str): Path to the trial directory containing TIFF files.
        readouts (list): List of readout codes for the current trial.
        fov (int): Field of view number to process.
        channel_list (list): List of channel indices.
        fiducial_channel (int): Index of the fiducial channel.
        depth (int): Depth of the image stack.
        storage_path (str): Path to store the converted HDF5 files.
        template (str): Template string for TIFF file naming.
        job_name (str): Job code/name for the current trial.
    Returns:
        """
    os.makedirs(os.path.join(storage_path, f'FOV{fov:02d}'), exist_ok=True)
    imagepath = os.path.join(trial_path, template.replace('++FOV++', f'{fov:02d}').replace('++JobCode++', job_name))
    num_channels = len(channel_list)

    try:
        with tf.TiffFile(imagepath) as handle:
            for r,readout in enumerate(readouts):
                if readout is None: continue

                stack_h5name = os.path.join(storage_path, f'FOV{fov:02d}', f'{readout}_stack.h5')
                attributes = {
                    'hybe': readout,
                    'fov': fov,
                    'type': readout[0],
                    'num_channels': num_channels,
                    'fiducial_channel': fiducial_channel,
                    'reference_channel': fiducial_channel,
                    'reference_hybe': readouts[0],
                    'channel_list': np.array([str(ch) for ch in channel_list], dtype='S'),
                    'shape': (),
                    'path': imagepath
                    }

                with h5py.File(stack_h5name, 'w') as f:
                    stack_group = f.create_group('/stack')
                    mip_group = f.create_group('/mip')
                    matrix_group = f.create_group('/matrix')
                    f.attrs.update(attributes)

                    for cid,ch in enumerate(channel_list):
                        st,end,step = cid+r*(num_channels*depth), cid+(r+1)*num_channels*depth, num_channels
                        stacks = np.stack([p.asarray() for p in handle.pages[st:end:step]], axis=0)
                        create_or_replace_dataset(stack_group, f'ch{ch}', stacks, 'uint16')
                        create_or_replace_dataset(mip_group, f'ch{ch}', np.max(stacks, axis=0), 'uint16')
                        create_or_replace_dataset(matrix_group, f'{readout}', np.eye(3), 'float32')
                    attributes['shape'] = stacks.shape
                    f.attrs.update(attributes)
                logging.info(f'Converted FOV {fov} in trial {trial_path} for readout {readout}')
    except FileNotFoundError:
        logging.error(f'TIFF file not found: {imagepath}')
    except Exception as e:
        logging.error(f'Error processing FOV {fov} in trial {trial_path}: {e}')

def convert_dax_to_h5_main(fov_list, hybe_records, dax_directory, storage_path, modality, overwrite=False, num_workers=4):
    from codelab_pipeline.io.preprocess import convert_dax_to_h5_worker
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for fov, hybe_record in product(fov_list, hybe_records):
            futures.append(executor.submit(convert_dax_to_h5_worker, fov, hybe_record, dax_directory, storage_path, modality, overwrite))
        for future in as_completed(futures):
            fov, hybe, error = future.result()
            if error is not None:
                print(f'Error processing FOV {fov}, hybe {hybe}: {error}', file=sys.stderr)

def dax_vlinks_h5(storage_path, fov_list, hybe_records, filename='vlinks.h5', overwrite=False):
    """
    Like vlinks_h5 below, but driven directly by ExperimentLayout records
    (ordered by hybe_num) instead of re-parsing hybe codes out of filenames
    -- convert_dax_to_h5_worker names per-hybe files after the raw source
    folder (e.g. 'Hyb_101_stack.h5'), which reorder_hybe_list_frompath's
    single-underscore-split assumption doesn't parse correctly.

    Depth (z-plane count) is read per-hybe, not assumed uniform: this real
    dataset's barcode hybes have a different totalFrames than regular hybes
    (e.g. 177 vs 130 z-planes), so a single shared stack_shape silently
    truncated the deeper hybes when a virtual dataset used the shallowest
    hybe's depth for all of them. The virtual stack dataset is sized to the
    max depth across hybes, and each hybe's VirtualSource is only assigned
    into its own actual depth -- shallower hybes leave the remainder at the
    HDF5 fill value, matching scripts/utils.py's combine_and_initialize_h5_files.
    """
    def create_fov_groups(vlink_file, fov):
        fov_group = vlink_file.create_group(f'FOV{fov:02d}')
        fov_group.create_group('mip')
        fov_group.create_group('stack')
        fov_group.create_group('matrix')
        fov_group.create_group('cells')
        return fov_group

    def create_virtual_datasets(fov_path, hybe_list, channel_list, height, width, depth_by_hybe):
        max_depth = max(depth_by_hybe.values())
        stack_layouts = {}
        mip_layouts = {}
        for ch in channel_list:
            stack_layout = h5py.VirtualLayout(shape=(len(hybe_list), height, width, max_depth), dtype='uint16')
            mip_layout = h5py.VirtualLayout(shape=(len(hybe_list), height, width), dtype='uint16')
            for hid, hybe in enumerate(hybe_list):
                source_path = os.path.join(fov_path, f'{hybe}_stack.h5')
                with h5py.File(source_path, 'r') as sample_h5:
                    if f'ch{ch}' not in sample_h5['/stack']:
                        continue
                depth = depth_by_hybe[hybe]
                vsource_stack = h5py.VirtualSource(source_path, f'/stack/ch{ch}', shape=(height, width, depth), dtype='uint16')
                vsource_mip = h5py.VirtualSource(source_path, f'/mip/ch{ch}', shape=(height, width), dtype='uint16')
                stack_layout[hid, :, :, :depth] = vsource_stack
                mip_layout[hid, ...] = vsource_mip
            stack_layouts[ch] = stack_layout
            mip_layouts[ch] = mip_layout
        return stack_layouts, mip_layouts

    hybe_records = sorted(hybe_records, key=lambda r: r['hybe_num'])
    hybe_list = [r['folder'] for r in hybe_records]
    fov_dirs = [f'FOV{fov:02d}' for fov in fov_list if os.path.exists(os.path.join(storage_path, f'FOV{fov:02d}'))]
    vlink_path = os.path.join(storage_path, filename)

    if os.path.exists(vlink_path) and not overwrite:
        vlink_file = h5py.File(vlink_path, 'r+')
    else:
        vlink_file = h5py.File(vlink_path, 'w')
        overwrite = True

    try:
        vlink_file.attrs['fov_list'] = np.array(fov_list, dtype='i4')
        vlink_file.attrs['hybe_list'] = np.array(hybe_list, dtype='S')
        vlink_file.attrs['readout_id_list'] = np.array([r['readout_id'] for r in hybe_records], dtype='i4')
        vlink_file.attrs['datatype_list'] = np.array([r['datatype'] for r in hybe_records], dtype='S')

        total_channel_list = sorted(set(ch for r in hybe_records for ch in r['channels']))
        vlink_file.attrs['channels_list'] = total_channel_list

        # Depth is derived straight from ExperimentLayout's totalFrames (already
        # validated per-hybe against the actual ingested DAX content at ingestion
        # time -- see convert_dax_to_h5_worker) rather than re-opening every
        # per-hybe file just to read its shape attr back.
        depth_by_hybe = {r['folder']: r['total_frames'] // len(r['channels']) for r in hybe_records}
        vlink_file.attrs['depth_by_hybe'] = np.array([depth_by_hybe[h] for h in hybe_list], dtype='i4')

        height = width = None
        for fov_dir in fov_dirs:
            fov = int(fov_dir.replace('FOV', ''))
            fov_group = create_fov_groups(vlink_file, fov) if overwrite else vlink_file[f'FOV{fov:02d}']
            fov_path = os.path.join(storage_path, fov_dir)

            if height is None:
                # height/width aren't in ExperimentLayout -- read once from any
                # already-ingested hybe file (uniform across a dataset in practice).
                with h5py.File(os.path.join(fov_path, f'{hybe_list[0]}_stack.h5'), 'r') as sample_h5:
                    h, w, _ = tuple(sample_h5.attrs['shape'])
                    height, width = int(h), int(w)
                vlink_file.attrs['stack_shape'] = np.array((height, width, max(depth_by_hybe.values())), dtype='i4')

            stack_layouts, mip_layouts = create_virtual_datasets(fov_path, hybe_list, total_channel_list, height, width, depth_by_hybe)
            for ch, stack_layout in stack_layouts.items():
                if f'ch{ch}' in fov_group['stack']: del fov_group['stack'][f'ch{ch}']
                fov_group['stack'].create_virtual_dataset(f'ch{ch}', stack_layout)
            for ch, mip_layout in mip_layouts.items():
                if f'ch{ch}' in fov_group['mip']: del fov_group['mip'][f'ch{ch}']
                fov_group['mip'].create_virtual_dataset(f'ch{ch}', mip_layout)

            for hybe in hybe_list:
                with h5py.File(os.path.join(fov_path, f'{hybe}_stack.h5'), 'r') as hybe_h5:
                    matrix = hybe_h5['/matrix'][hybe][:]
                if hybe in fov_group['matrix']: del fov_group['matrix'][hybe]
                fov_group['matrix'].create_dataset(hybe, data=matrix, dtype='float32')
    finally:
        vlink_file.flush()
        vlink_file.close()

    return vlink_path

def convert_tiff_to_h5_main(trial_path_dict, fov_list,
                            channel_list, fiducial_channel, depth, storage_path, template, job_name,
                            num_workers=4):

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        for (trial_path, readout_codes), fov in product(trial_path_dict.items(), fov_list):
            readouts = [parse_readout(ro) for ro in readout_codes]
            futures.append(executor.submit(convert_tiff_to_h5_worker, trial_path, readouts, fov,
                                           channel_list, fiducial_channel, depth, storage_path, template, job_name,))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f'Error processing FOV: {e}', file=sys.stderr)

def vlinks_h5(storage_path, fov_list, filename='vlinks.h5', overwrite=False):
    """
    Create a virtual HDF5 file linking all FOVs and hybes.

    Args:
        storage_path (str): Path where the FOV directories are stored.
        fov_list (list): List of field of view numbers to include.
        filename (str): Base name for the virtual HDF5 file.
        overwrite (bool): Whether to overwrite existing virtual file.
    """
    def create_fov_groups(vlink_file, fov):
        """Create groups for a specific FOV."""
        fov_group = vlink_file.create_group(f'FOV{fov:02d}')
        fov_group.create_group('mip')
        fov_group.create_group('stack')
        fov_group.create_group('matrix')
        fov_group.create_group('cells')
        return fov_group

    def create_virtual_datasets(fov_path, h5_files, channel_list, hybe_list, stack_shape):
        """Create virtual datasets for stack and mip."""
        stack_layouts = {}
        mip_layouts = {}

        for ch in channel_list:
            stack_layout = h5py.VirtualLayout(shape=(len(hybe_list), *stack_shape), dtype='uint16')
            mip_layout = h5py.VirtualLayout(shape=(len(hybe_list), *stack_shape[1:]), dtype='uint16')

            for h5_file in h5_files:
                hid = hybe_list.index(h5_file.replace('_stack.h5', ''))
                source_path = os.path.join(fov_path, h5_file)
                vsource_stack = h5py.VirtualSource(source_path, f'/stack/ch{ch}', shape=stack_shape, dtype='uint16')
                vsource_mip = h5py.VirtualSource(source_path, f'/mip/ch{ch}', shape=stack_shape[1:], dtype='uint16')
                stack_layout[hid, ...] = vsource_stack
                mip_layout[hid, ...] = vsource_mip

            stack_layouts[ch] = stack_layout
            mip_layouts[ch] = mip_layout

        return stack_layouts, mip_layouts

    fov_dirs = [f'FOV{fov:02d}' for fov in fov_list if os.path.exists(os.path.join(storage_path, f'FOV{fov:02d}'))]
    vlink_path = os.path.join(storage_path, filename)
    hybe_list = reorder_hybe_list_frompath(os.path.join(storage_path, fov_dirs[0]))

    try:
        if os.path.exists(vlink_path) and not overwrite:
            vlink_file = h5py.File(vlink_path, 'r+')
        else:
            vlink_file = h5py.File(vlink_path, 'w')
            overwrite = True  # Ensure overwrite is True for new file

        vlink_file.attrs['fov_list'] = np.array(fov_list, dtype='i4')
        vlink_file.attrs['hybe_list'] = np.array(hybe_list, dtype='S')
        vlink_file.attrs['reference_channel'] = ''
        vlink_file.attrs['reference_hybe'] = ''
        vlink_file.attrs['stack_shape'] = ()

        for i,fov_dir in enumerate(fov_dirs):
            fov = int(fov_dir.replace('FOV',''))
            fov_group = create_fov_groups(vlink_file, fov) if overwrite else vlink_file[f'FOV{fov:02d}']
            fov_path = os.path.join(storage_path, fov_dir)
            h5_files = [f'{h}_stack.h5' for h in reorder_hybe_list_frompath(fov_path)]

            # Read sample HDF5 to get attributes
            if i == 0:
                channels_list = []
                for j,h5_file in enumerate(h5_files):
                    with h5py.File(os.path.join(fov_path, h5_file), 'r') as sample_h5:
                        channels_list.append([str(ch.decode('utf-8')) if isinstance(ch, bytes) else str(ch)
                                               for ch in sample_h5.attrs['channel_list']])
                        if j == 0: stack_shape = tuple([int(s.decode('utf-8')) if isinstance(s, bytes) else int(s) for s in sample_h5.attrs['shape']])
                total_channel_list = np.unique(np.ravel(channels_list)).tolist()
                vlink_file.attrs['stack_shape'] = np.array(stack_shape, dtype='i4')
                vlink_file.attrs['channels_list'] = channels_list

            stack_layouts, mip_layouts = create_virtual_datasets(fov_path, h5_files, total_channel_list, hybe_list, stack_shape)

            # Add virtual datasets to the HDF5 file
            for ch, stack_layout in stack_layouts.items():
                if f'ch{ch}' in fov_group['stack']:
                    del fov_group['stack'][f'ch{ch}']
                fov_group['stack'].create_virtual_dataset(f'ch{ch}', stack_layout)
            for ch, mip_layout in mip_layouts.items():
                if f'ch{ch}' in fov_group['mip']:
                    del fov_group['mip'][f'ch{ch}']
                fov_group['mip'].create_virtual_dataset(f'ch{ch}', mip_layout)

            for rd in hybe_list:
                with h5py.File(os.path.join(fov_path, f'{rd}_stack.h5'), 'r') as hybe_h5:
                    matrix = hybe_h5[f'/matrix'][rd][:]
                if rd in fov_group['matrix']:
                    del fov_group['matrix'][rd]
                fov_group['matrix'].create_dataset(f'{rd}', data=matrix, dtype='float32')

            if overwrite:
                fov_group['cells'].attrs['defined_hybe'] = hybe_list[0]
                fov_group['cells'].attrs['defined_channel'] = ''
                fov_group['cells'].attrs['pad'] = 10
                fov_group['cells'].attrs['good'] = []
                dt = h5py.string_dtype(encoding='utf-8')
                fov_group['cells'].create_dataset('cellbarcodes', shape=(0,), maxshape=(None,), dtype=dt)
                fov_group['cells'].create_dataset(f'mask', shape=stack_shape[1:], dtype='uint16')
                fov_group['cells'].create_dataset(f'spots', shape=(0,7), maxshape=(None,7), dtype='float32')
                fov_group['cells'].create_dataset(f'matrix/yx', shape=(0,len(hybe_list),3,3), maxshape=(None,len(hybe_list),3,3), dtype='float32')
                fov_group['cells'].create_dataset(f'matrix/zx', shape=(0,len(hybe_list),3,3), maxshape=(None,len(hybe_list),3,3), dtype='float32')
    finally:
        vlink_file.flush()
        vlink_file.close()

    return vlink_path

def align_mips_worker(storage_path, fov, hybe, reference_hybe, reference_mip, reference_channel, lb=0.3, up=0.9999):
    """
    Align MIPs for a specific FOV using a reference hybe and channel.

    Args:
        storage_path (str): Path where the FOV directories are stored.
        fov (int): Field of view number to process.
        reference_hybe (str): Reference hybridization readout code.
        reference_channel (int): Index of the reference channel.
        lb (float): Lower bound for intensity normalization.
        up (float): Upper bound for intensity normalization.
    """
    h5file_path = os.path.join(storage_path, f'FOV{fov:02d}', f'{hybe}_stack.h5')
    savepath = os.path.join(storage_path, f'FOV{fov:02d}', 'Aligned_MIPs')
    os.makedirs(savepath, exist_ok=True)
    try:
        with h5py.File(h5file_path, 'r+') as h5_file:
            # update attributes
            current_reference_hybe = h5_file.attrs.get('reference_hybe', '')
            current_reference_channel = h5_file.attrs.get('reference_channel', '')
            h5_file.attrs['reference_hybe'] = '_'.join([str(current_reference_hybe), str(reference_hybe)]) if '_' in str(current_reference_hybe) else str(reference_hybe)
            h5_file.attrs['reference_channel'] = '_'.join([str(current_reference_channel), str(reference_channel)]) if '_' in str(current_reference_channel) else str(reference_channel)
            reference_mip_norm = normalize_to_uint8(reference_mip, lb, up)

            # Compute homography matrix
            if hybe == reference_hybe:
                homography_matrix = np.eye(3)
                moving_mip = reference_mip
                moving_mip_norm = reference_mip_norm
            else:
                moving_mip = h5_file[f'/mip/ch{reference_channel}'][:]
                H = h5_file[f'/matrix/{reference_hybe}'][:]
                moving_mip = cv2.warpPerspective(moving_mip, H, (moving_mip.shape[1], moving_mip.shape[0]),
                                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                moving_mip_norm = normalize_to_uint8(moving_mip, lb, up)

                homography_matrix = compute_msd_homography_matrix(
                    moving_mip_norm, reference_mip_norm, fixed_scale=1.0, fixed_angle=False,
                    initial_guess=[0,0,0], method='Powell', verbose=False)

            # Save homography matrix
            if not 'matrix' in h5_file:
                h5_file.create_group('matrix')
            if f'{hybe}' in h5_file['matrix']:
                H0 = h5_file[f'matrix/{hybe}'][:]
                H = homography_matrix @ H0
                h5_file[f'matrix/{hybe}'][:] = H
            else:
                h5_file['matrix'].create_dataset(f'{hybe}', data=homography_matrix, dtype='float32')

            # Apply transformation to all channels and save aligned MIPs
            h, w = moving_mip.shape[:2]
            aligned_mip = cv2.warpPerspective(moving_mip, homography_matrix, (w, h),
                                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            aligned_mip_norm = normalize_to_uint8(aligned_mip, lb, up)

            fig,ax = plt.subplots(1,2,figsize=(20,10), facecolor='none')
            composite = np.zeros((h,w,3), dtype=float)
            composite[...,0] = reference_mip_norm
            composite[...,1] = moving_mip_norm
            composite[...,2] = moving_mip_norm
            ax[0].imshow(composite /255)
            ax[0].axis('off')
            ax[0].set_title(f'Original Image\nChannel: {reference_channel} | FOV: {fov}\nRED: {reference_hybe} | GREEN/BLUE: {hybe}',**fontdict_label)

            composite = np.zeros((h,w,3), dtype=float)
            composite[...,0] = reference_mip_norm
            composite[...,1] = aligned_mip_norm
            composite[...,2] = aligned_mip_norm
            ax[1].imshow(composite /255)
            ax[1].axis('off')
            ax[1].set_title(f'Aligned Image\nChannel: {reference_channel} | FOV: {fov}\nRED: {reference_hybe} | GREEN/BLUE: {hybe}',**fontdict_label)
            fig.tight_layout()
            image_path = os.path.join(savepath, f'{hybe}_{reference_hybe}_alignment_check_channel{reference_channel}.png')
            if os.path.exists(image_path):
                os.remove(image_path)
            fig.savefig(os.path.join(savepath, f'{hybe}_{reference_hybe}_alignment_check_channel{reference_channel}.png'), dpi=150, )
            plt.close(fig)

            logging.info(f'Aligned MIPs for FOV {fov} using reference hybe {reference_hybe} and channel {reference_channel}')
    except FileNotFoundError:
        logging.error(f'HDF5 file not found: {h5file_path}')
    except Exception as e:
        logging.error(f'Error aligning MIPs for FOV {fov}, hybe {hybe}: {e}')

    return fov

def align_mips_main(storage_path, fov_list, hybe_list, reference_hybe, reference_channel, lb=0.3, up=0.9999, num_workers=4):
    reference_mip_dict = {fov: None for fov in fov_list}
    for fov in fov_list:
        with h5py.File(os.path.join(storage_path, f'FOV{fov:02d}', f'{reference_hybe}_stack.h5'), 'r') as ref_h5:
            H = ref_h5[f'/matrix/{reference_hybe}'][:]
            reference_mip = ref_h5[f'/mip/ch{reference_channel}'][:]
            reference_mip = cv2.warpPerspective(reference_mip, H, (reference_mip.shape[1], reference_mip.shape[0]),
                                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            reference_mip_dict[fov] = reference_mip

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        for fov, hybe in product(fov_list, hybe_list):
            futures.append(executor.submit(align_mips_worker, storage_path, fov, hybe, reference_hybe, reference_mip_dict[fov], reference_channel, lb, up))

        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                print(f'Error aligning MIPs: {e}', file=sys.stderr)
