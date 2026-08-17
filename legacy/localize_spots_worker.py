"""
Dead code -- see legacy/README.md. Moved out of
codelab_pipeline/localization/localization.py on 2026-08-17.

DO NOT REACTIVATE AS WRITTEN. localization.py's own docstrings point here as
a porting reference ("in localize_spots_worker above, reactivated here"), and
the trap is line 331 of the original:

    H1 = f[f'/matrix/{hybe}'][:]

That reads the same-modality alignment matrix out of the raw {hybe}_stack.h5
file. That store is obsolete: write_same_modality_matrices was migrated to
vlinks.h5 (chain.py), so nothing has written to the stack copy since, and it
was measured to disagree with vlinks on 16 of 18 hybes in
data/chr19_downstream_new -- the stack files still held a previous alignment
run's translations while vlinks correctly held identity.

Reading a matrix from a raw stack file is exactly the failure that rotated
every projected cell mask ~100 degrees (see the /matrix_across removal in the
same pass). Any port must take its matrices from vlinks_store, ideally via
FrameResolver rather than by composing them by hand.

The live equivalents are localize_cell_2d_worker / localize_cell_3d_worker,
which share crop construction through _build_cell_crop.
"""
import os

import h5py
import numpy as np
import numpy.linalg as la

from codelab_pipeline.alignment import chain as alignment


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
