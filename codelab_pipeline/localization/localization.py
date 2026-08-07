import os
import numpy as np
import numpy.linalg as la
from scipy.optimize import minimize
from skimage.feature import peak_local_max
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py

from ..io import preprocess
import cv2

from ..alignment import chain as alignment

def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo, yo = float(xo), float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    return offset + amplitude * np.exp(- (a*(x-xo)**2 + 2*b*(x-xo)*(y-yo) + c*(y-yo)**2))

def cost_function(params, x, y, z):
    amp, xo, yo, sigma_x, sigma_y, theta, offset = params
    model = gaussian_2d((x, y), amp, xo, yo, sigma_x, sigma_y, theta, offset)
    residuals = (z - model)**2
    return np.nansum(residuals) 

def fit_gaussian_2d(img, x0, y0,):
    x, y = np.arange(img.shape[1]), np.arange(img.shape[0])
    x, y = np.meshgrid(x, y)
    mask = ~np.isnan(img)
    x, y, z = x[mask], y[mask], img[mask]
    initial_guess = (img[mask].max(), x0, y0, 1, 1, 0, img[mask].min())
    result = minimize(cost_function, initial_guess, args=(x, y, z), method='Powell')
    if result.success:
        return result.x
    else:
        return None

def gaussian_3d(xyz, amplitude, x0, y0, z0, sigma_x, sigma_y, sigma_z, offset):
    x, y, z = xyz
    return offset + amplitude * np.exp(
        -(((x - x0)**2) / (2 * sigma_x**2)
          + ((y - y0)**2) / (2 * sigma_y**2)
          + ((z - z0)**2) / (2 * sigma_z**2))
    )

def cost_function_3d(params, x, y, z, values):
    amp, x0, y0, z0, sigma_x, sigma_y, sigma_z, offset = params
    model = gaussian_3d((x, y, z), amp, x0, y0, z0, sigma_x, sigma_y, sigma_z, offset)
    residuals = (values - model)**2
    return np.nansum(residuals)

def fit_gaussian_3d(zyx, x0, y0, z0):
    z, y, x = np.indices(zyx.shape)
    mask = np.isfinite(zyx)
    x, y, z, values = x[mask], y[mask], z[mask], zyx[mask]

    initial_guess = (
        np.nanmax(values),  # amplitude
        x0, y0, z0,          # centers
        1, 1, 1,             # sigma_x/y/z
        np.nanmin(values)   # offset
    )

    result = minimize(cost_function_3d, initial_guess, args=(x, y, z, values), method='Powell')
    if result.success:
        return result.x  # [amp, x0, y0, z0, sigma_x, sigma_y, sigma_z, offset]
    else:
        return None

def localize_spots_worker(fov, hybe, hybe_list, cell_parameter_dict,
                           max_to_background, max_to_average, frac, absolute_threshold, cpad, spad, min_distance, channel, including_z,
                           max_num_alleles, max_sigma, max_deviation, h5_save_path):
    cid = cell_parameter_dict['cid']
    cb = cell_parameter_dict['cb']
    matrix_yx = cell_parameter_dict['matrix_yx']
    matrix_zx = cell_parameter_dict['matrix_zx']
    cy,cx = cell_parameter_dict['cy'], cell_parameter_dict['cx']
    xyzs = []
    xyzs_crop = []
    with h5py.File(os.path.join(h5_save_path, f'FOV{fov:02d}', f'{hybe}_stack.h5'), 'r') as f:
        depth, height, width = f[f'stack/ch{channel}'].shape
        H1 = f[f'/matrix/{hybe}'][:]
        if including_z:
            H2 = matrix_yx
            H3 = matrix_zx
            H2[0,2] -= H3[0,2]
        else:
            H2 = matrix_yx
        H = H2@H1
        Hinv =  la.inv(H)
        ry,rx = alignment.align_cell((cy, cx), Hinv, (height, width))
        if len(ry) == 0: return np.zeros((0,7), dtype=float), None, hybe

        rymin,rymax,rxmin,rxmax = max(0, ry.min()-cpad), min(height, ry.max()+cpad+1), max(0, rx.min()-cpad), min(width, rx.max()+cpad+1)
        img = np.full((rymax-rymin, rxmax-rxmin), np.nan, dtype=float)
        img[ry-rymin,rx-rxmin] = f[f'/mip/ch{channel}'][:][ry,rx]
        
        stacks = np.full((depth, rymax-rymin, rxmax-rxmin), np.nan, dtype=float)
        stacks_value = f[f'/stack/ch{channel}'][:,rymin:rymax,rxmin:rxmax]
        stacks[:,ry-rymin,rx-rxmin] = stacks_value[:,ry-rymin,rx-rxmin]
        bimg = np.nanmax(stacks,axis=1)

        v,c = np.unique(img[~np.isnan(img)], return_counts=True)
        cutoff = max_to_background * v[c.argmax()]
        yx = peak_local_max(img, min_distance=min_distance,
                            exclude_border=1,threshold_abs=cutoff)
        brightness = img[yx[:,0], yx[:,1]]
        for j in brightness.argsort()[::-1][:max_num_alleles]:
            if (brightness[j] < brightness.max() * frac) or (brightness[j] < absolute_threshold): continue
            y,x = yx[j]
            if including_z:
                zs = peak_local_max(bimg[:,x], exclude_border=1, threshold_abs=max(brightness[j]*.9,cutoff,absolute_threshold))
            else:
                zs = [np.nan]
            if len(zs) == 0: continue
            z = zs[0]
            x1,y1,_ = H@np.array([x+rxmin,y+rymin,1]).reshape(3,1)
            xyzs.append([fov,cid,hybe_list.index(hybe),cb,float(x1),float(y1),float(z+H3[1,2])])
            xyzs_crop.append([fov,cid,hybe_list.index(hybe),cb,float(x),float(y),float(z)])
                    
            """
            symin,symax,sxmin,sxmax = max(0, y-spad), min(img.shape[0], y+spad+1), max(0, x-spad), min(img.shape[1], x+spad+1)
            params2d = fit_gaussian_2d(img[symin:symax,sxmin:sxmax], x-sxmin, y-symin)
            if params2d is not None:
                amp, x0, y0, sigma_x, sigma_y, theta, offset = params2d
                sigma = abs(sigma_x*sigma_y)**0.5
                if (abs(sigma_x) > .5 and abs(sigma_y) > .5 and sigma < max_sigma and 
                        ((x0+sxmin - x)**2 + (y0+symin - y)**2)**.5 < max_deviation and 
                        amp > (max_to_average-1) * offset and brightness[j] > brightness.max() * frac):
                    if including_z:
                        zs = peak_local_max(bimg[:,x], exclude_border=1, threshold_abs=cutoff)
                        if len(zs) == 0: continue
                        zs = zs[np.isclose(bimg[zs,x], brightness[j], atol=10)].flatten()
                    else:
                        zs = [np.nan]

                    for z in zs:
                        x1,y1,_ = H@np.array([x+rxmin,y+rymin,1]).reshape(3,1)
                        xyzs.append([fov,cid,hybe_list.index(hybe),cb,float(x1),float(y1),float(z+H3[1,2])])
                        xyzs_crop.append([fov,cid,hybe_list.index(hybe),cb,float(x),float(y),float(z)])
            
                    szmin,szmax = max(0, z-spad), min(depth, z+spad+1)
                    zyx = stacks[szmin:szmax,symin:symax,sxmin:sxmax]
                    params3d = fit_gaussian_3d(zyx, x-sxmin, y-symin, z-szmin)
                    if params3d is not None:
                        amp, x0, y0, z0, sigma_x, sigma_y, sigma_z, offset = params3d
                        sigma = abs(sigma_x*sigma_y*sigma_z)**(1/3)
                        if abs(sigma_x) > .5 and abs(sigma_y) > .5 and abs(sigma_z) > .5 and sigma < max_sigma:
                            if ((x0+sxmin - x)**2 + (y0+symin - y)**2 + (z0+szmin - z)**2)**.5 < max_deviation:
                                if amp > (max_to_average-1) * offset and brightness[j] > brightness.max() * frac:
                                    x1,y1,_ = H@np.array([x+rxmin,y+rymin,1]).reshape(3,1)
                                    xyzs.append([fov,cid,hybe_list.index(hybe),cb,float(x1),float(y1),float(z+H3[1,2])])
                                    xyzs_crop.append([fov,cid,hybe_list.index(hybe),cb,float(x),float(y),float(z)])"""

    if len(xyzs) > 0:
        return np.array(xyzs, dtype=float), (img,bimg,np.array(xyzs_crop, dtype=float)), hybe
    else:
        return np.zeros((0,7), dtype=float), (img,bimg,np.zeros((0,7), dtype=float)), hybe

def _build_cell_crop(cell, hybe, channel, storage_path, fov, pad, modality=None):
    """
    Shared crop-building logic for localize_cell_2d_worker/3d_worker AND
    the interactive spot localization panel's "Current Cell" scope --
    factored out so the bulk workers and the interactive panel can't drift
    apart. Transforms this cell's own mask area into `hybe`'s native frame
    (cell.get_area_in_readout, itself never resampling raw pixels -- only
    coordinates move), reads a padded bbox crop of both the MIP and full
    Z-stack, and NaNs out every pixel outside the cell's own mask within
    that bbox (so background/neighboring-cell pixels never contaminate a
    per-cell peak search). h5py fancy-indexing requires ascending-order
    indices (unlike numpy); y_area/x_area come from np.where and aren't
    sorted, so the rectangular crop is sliced first (always
    contiguous/ascending) and the cell-mask fancy indexing done on the
    resulting in-memory array instead.

    Returns None only if the transformed area is genuinely empty (e.g.
    the cell has no mask pixels at all) -- get_area_in_readout itself
    never raises: no alignment matrix yet for (hybe, modality) means
    identity (no correction), never an error, per the same "no no-
    alignment case" principle compute_cell_alignment's own fallbacks
    already implement. Otherwise returns a dict: {'img': (h,w) MIP crop
    (NaN outside cell), 'stacks': (h,w,depth) Z-stack crop (NaN outside
    cell), 'bimg': (w,depth) Z-profile per column (nanmax over y),
    'rxmin': int, 'rymin': int, 'H': cell's yx matrix for this hybe
    (identity if none), 'Hz': cell's zx matrix for this hybe (identity
    if none)}.
    """
    modality = modality if modality is not None else cell.modality
    x_area, y_area = cell.get_area_in_readout(hybe, modality)
    if len(x_area) == 0:
        return None
    x_area, y_area = x_area.astype(int), y_area.astype(int)

    height, width = cell.frame_shape
    rymin, rymax = max(0, y_area.min() - pad), min(height, y_area.max() + pad + 1)
    rxmin, rxmax = max(0, x_area.min() - pad), min(width, x_area.max() + pad + 1)

    h5path = os.path.join(storage_path, f'FOV{fov:02d}', f'{hybe}_stack.h5')
    with h5py.File(h5path, 'r') as f:
        mip_crop = f[f'/mip/ch{channel}'][rymin:rymax, rxmin:rxmax]
        stacks_value = f[f'/stack/ch{channel}'][rymin:rymax, rxmin:rxmax, :]
        img = np.full((rymax - rymin, rxmax - rxmin), np.nan, dtype=float)
        img[y_area - rymin, x_area - rxmin] = mip_crop[y_area - rymin, x_area - rxmin]

    depth = stacks_value.shape[-1]
    stacks = np.full((rymax - rymin, rxmax - rxmin, depth), np.nan, dtype=float)
    stacks[y_area - rymin, x_area - rxmin] = stacks_value[y_area - rymin, x_area - rxmin]
    bimg = np.nanmax(stacks, axis=0)  # (width_crop, depth) -- Z-profile per column

    # matrix_to (not a direct cell.matrices lookup): cell.matrices entries
    # target the shared FOV frame, not cell.reference_hybe's frame -- see
    # compute_cell_alignment's docstring -- and the caller below applies H
    # forward to a raw point to land in cell.reference_hybe's frame
    # (matching spot_mapper.raw_to_reference's own convention), which
    # matrix_to is what actually resolves.
    H = cell.matrix_to(hybe, modality)
    Hz = cell.matrices.get((hybe, modality), {}).get('zx', np.eye(3))

    return {'img': img, 'stacks': stacks, 'bimg': bimg, 'rxmin': rxmin, 'rymin': rymin, 'H': H, 'Hz': Hz}


def localize_cell_2d_worker(cell, hybe, channel, storage_path, fov,
                            max_to_background, max_to_average, absolute_threshold,
                            min_distance, frac, max_num_alleles, pad):
    """
    2D localization for one cell in one hybe, with sub-pixel gaussian
    refinement (the fit_gaussian_2d step that exists but is commented out
    in localize_spots_worker above, reactivated here). Ports
    scripts/utils.py's localize_2d_spots_worker, adapted to this project's
    ACell/composed-matrix model instead of the old flat H5 /cells/matrix
    arrays. Returns (cell.id, hybe, [ASpot, ...]) -- run inside a
    ProcessPoolExecutor, so results are returned rather than mutating cell
    in place (a separate-process copy wouldn't be visible to the caller).
    """
    from ..models.spot import ASpot
    spots = []
    crop = _build_cell_crop(cell, hybe, channel, storage_path, fov, pad)
    if crop is None:
        return cell.id, hybe, spots
    img, stacks, bimg = crop['img'], crop['stacks'], crop['bimg']
    rxmin, rymin, H, Hz = crop['rxmin'], crop['rymin'], crop['H'], crop['Hz']

    cutoff = max_to_background * np.nanquantile(img, 0.5)
    yx = peak_local_max(img, min_distance=min_distance, exclude_border=1,
                        threshold_abs=max(cutoff, absolute_threshold))
    if len(yx) == 0:
        return cell.id, hybe, spots
    brightness = img[yx[:, 0], yx[:, 1]]

    for j in brightness.argsort()[::-1][:max_num_alleles]:
        y, x = yx[j]
        z_candidates = peak_local_max(bimg[x], exclude_border=1,
                                      threshold_abs=max(bimg[x].max() * .9, absolute_threshold, cutoff))
        if len(z_candidates) == 0:
            continue
        z = int(z_candidates[bimg[x, z_candidates].argmax()])

        symin, symax = max(0, y - pad), min(img.shape[0], y + pad + 1)
        sxmin, sxmax = max(0, x - pad), min(img.shape[1], x + pad + 1)
        params = fit_gaussian_2d(img[symin:symax, sxmin:sxmax], x - sxmin, y - symin)
        if params is None:
            continue
        amp, xo, yo, sigma_x, sigma_y, theta, offset = params
        if not (abs(sigma_x) > .5 and abs(sigma_y) > .5):
            continue
        if ((xo + sxmin - x) ** 2 + (yo + symin - y) ** 2) ** .5 >= 3:
            continue
        if not (amp + offset > max_to_average * offset and brightness[j] > brightness.max() * frac):
            continue

        raw_x, raw_y = x + rxmin, y + rymin
        x1, y1, _ = H @ np.array([raw_x, raw_y, 1]).reshape(3, 1)
        z1 = z + Hz[1, 2]

        spot = ASpot()
        spot.set_metadata(fov=fov, hybe=hybe, channel=channel, cell=cell.id,
                          coordinate=(float(x1), float(y1), float(z1)),
                          raw_coordinate=(float(raw_x), float(raw_y), float(z)),
                          brightness=float(brightness[j]))
        spots.append(spot)

    return cell.id, hybe, spots

def localize_cells_2d(cell_container, fov, hybe_records, channel,
                      max_to_background=1.25, max_to_average=1.25, absolute_threshold=450.0,
                      min_distance=3, frac=0.8, max_num_alleles=2, pad=5,
                      storage_path=None, n_procs=4):
    """
    Bulk (non-interactive) 2D localization over every cell in
    cell_container.get_cells(fov), across every hybe in hybe_records.
    Parameters are an already-confirmed set -- tune interactively via
    LocalizationWidget first, then run this in bulk; not re-tuned per call.
    Writes results directly into each cell's .spots/.num_spots/.total_num_spots.
    """
    cells = cell_container.get_cells(fov)
    tasks = [(cell, record['folder']) for cell in cells for record in hybe_records]

    with ProcessPoolExecutor(max_workers=n_procs) as executor:
        futures = [executor.submit(localize_cell_2d_worker, cell, hybe, channel, storage_path, fov,
                                   max_to_background, max_to_average, absolute_threshold,
                                   min_distance, frac, max_num_alleles, pad)
                  for cell, hybe in tasks]
        cells_by_id = {c.id: c for c in cells}
        for future in as_completed(futures):
            cell_id, hybe, spots = future.result()
            if len(spots) == 0:
                continue
            cell = cells_by_id[cell_id]
            cell.spots.extend(spots)
            cell.num_spots[hybe] = cell.num_spots.get(hybe, 0) + len(spots)
            cell.total_num_spots += len(spots)

def localize_cell_3d_worker(cell, hybe, channel, storage_path, fov,
                            max_to_background, max_to_average, absolute_threshold,
                            min_distance, frac, max_num_alleles, max_sigma, max_deviation, pad, spad):
    """
    3D localization for one cell in one hybe, with sub-pixel 2D+3D gaussian
    refinement -- ports scripts/utils.py's localize_3d_spots_worker, same
    adaptation as localize_cell_2d_worker above.
    """
    from ..models.spot import ASpot
    spots = []
    crop = _build_cell_crop(cell, hybe, channel, storage_path, fov, pad)
    if crop is None:
        return cell.id, hybe, spots
    img, stacks, bimg = crop['img'], crop['stacks'], crop['bimg']
    rxmin, rymin, H, Hz = crop['rxmin'], crop['rymin'], crop['H'], crop['Hz']
    depth = stacks.shape[-1]

    cutoff = max_to_background * np.nanquantile(img, 0.5)
    yx = peak_local_max(img, min_distance=min_distance, exclude_border=1,
                        threshold_abs=max(cutoff, absolute_threshold))
    if len(yx) == 0:
        return cell.id, hybe, spots
    brightness = img[yx[:, 0], yx[:, 1]]

    for j in brightness.argsort()[::-1][:max_num_alleles]:
        y, x = yx[j]
        z_candidates = peak_local_max(bimg[x], exclude_border=1,
                                      threshold_abs=max(bimg[x].max() * .9, absolute_threshold, cutoff))
        if len(z_candidates) == 0:
            continue
        z = int(z_candidates[bimg[x, z_candidates].argmax()])

        symin, symax = max(0, y - pad), min(img.shape[0], y + pad + 1)
        sxmin, sxmax = max(0, x - pad), min(img.shape[1], x + pad + 1)
        params2d = fit_gaussian_2d(img[symin:symax, sxmin:sxmax], x - sxmin, y - symin)
        if params2d is None:
            continue
        amp, xo, yo, sigma_x, sigma_y, theta, offset = params2d
        if not (abs(sigma_x) > .5 and abs(sigma_y) > .5 and (sigma_x * sigma_y) ** .5 < max_sigma):
            continue
        if ((xo + sxmin - x) ** 2 + (yo + symin - y) ** 2) ** .5 >= max_deviation:
            continue
        if not (amp > (max_to_average - 1) * offset and brightness[j] > brightness.max() * frac):
            continue

        szmin, szmax = max(0, z - spad), min(depth, z + spad + 1)
        zyx = stacks[symin:symax, sxmin:sxmax, szmin:szmax]
        params3d = fit_gaussian_3d(zyx, x - sxmin, y - symin, z - szmin)
        if params3d is None:
            continue
        amp3, x0, y0, z0, sigma_x3, sigma_y3, sigma_z3, offset3 = params3d
        sigma3 = abs(sigma_x3 * sigma_y3 * sigma_z3) ** (1 / 3)
        if not (abs(sigma_x3) > .5 and abs(sigma_y3) > .5 and abs(sigma_z3) > .5 and sigma3 < max_sigma):
            continue
        if ((x0 + sxmin - x) ** 2 + (y0 + symin - y) ** 2 + (z0 + szmin - z) ** 2) ** .5 >= max_deviation:
            continue
        if not (amp3 > (max_to_average - 1) * offset3 and brightness[j] > brightness.max() * frac):
            continue

        raw_x, raw_y, raw_z = x + rxmin, y + rymin, z0 + szmin
        x1, y1, _ = H @ np.array([raw_x, raw_y, 1]).reshape(3, 1)
        z1 = raw_z + Hz[1, 2]

        spot = ASpot()
        spot.set_metadata(fov=fov, hybe=hybe, channel=channel, cell=cell.id,
                          coordinate=(float(x1), float(y1), float(z1)),
                          raw_coordinate=(float(raw_x), float(raw_y), float(raw_z)),
                          brightness=float(brightness[j]))
        spots.append(spot)

    return cell.id, hybe, spots

def localize_cells_3d(cell_container, fov, hybe_records, channel,
                      max_to_background=1.25, max_to_average=1.25, absolute_threshold=450.0,
                      min_distance=3, frac=0.8, max_num_alleles=2, max_sigma=3.0, max_deviation=3.0,
                      pad=5, spad=5, storage_path=None, n_procs=4):
    """3D counterpart of localize_cells_2d -- see localize_cell_3d_worker."""
    cells = cell_container.get_cells(fov)
    tasks = [(cell, record['folder']) for cell in cells for record in hybe_records]

    with ProcessPoolExecutor(max_workers=n_procs) as executor:
        futures = [executor.submit(localize_cell_3d_worker, cell, hybe, channel, storage_path, fov,
                                   max_to_background, max_to_average, absolute_threshold,
                                   min_distance, frac, max_num_alleles, max_sigma, max_deviation, pad, spad)
                  for cell, hybe in tasks]
        cells_by_id = {c.id: c for c in cells}
        for future in as_completed(futures):
            cell_id, hybe, spots = future.result()
            if len(spots) == 0:
                continue
            cell = cells_by_id[cell_id]
            cell.spots.extend(spots)
            cell.num_spots[hybe] = cell.num_spots.get(hybe, 0) + len(spots)
            cell.total_num_spots += len(spots)