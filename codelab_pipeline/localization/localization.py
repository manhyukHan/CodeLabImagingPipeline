import os
import numpy as np
import numpy.linalg as la
from scipy.optimize import minimize
from skimage.feature import peak_local_max
from concurrent.futures import ProcessPoolExecutor, as_completed

import ipywidgets as widgets
import time
import h5py

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from ..io import preprocess
import cv2

from ..alignment import chain as alignment

from IPython.display import display, clear_output
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
          '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
          '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
          '#000075', '#808080', '#000000']

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

class LocalizationWidget(object):
    """
    A widget for setting localization parameters.
    """
    def __init__(self, h5_save_path, filename='vlinks.h5', cellbarcode_list=['NA'], n_cpus=4):
        self.h5_save_path = h5_save_path
        self.filename = filename
        self.cellbarcode_list = cellbarcode_list
        self.n_cpus = n_cpus
        
        with h5py.File(os.path.join(h5_save_path, filename), 'r') as f:
            self.fov_list = f.attrs['fov_list'][:].tolist()
            self.channels_list = f.attrs['channels_list'][:].tolist()
            self.hybe_list = [str(h.decode()) for h in f.attrs['hybe_list']]

        self.total_channel_list = np.unique(np.ravel(self.channels_list)).tolist()

        self.widgets = {}
        self._create_widgets()
        self._set_events()
        
        self.output = widgets.Output()
        self.ui = widgets.VBox([
            widgets.HBox([
                self.widgets['including_z'],
                self.widgets['overwrite'],
            ]),
            widgets.HBox([
                self.widgets['fov'],
                self.widgets['channel'],
                self.widgets['sub_hybe_list'],
                self.widgets['cell_id'],
            ]),
            widgets.HBox([
                self.widgets['lb'],
                self.widgets['up'],
                self.widgets['Max/Background'],
                self.widgets['max/avg'],
            ]),
            widgets.HBox([
                self.widgets['cpad'],
                self.widgets['spad'],
                self.widgets['max_sigma'],
                self.widgets['max_deviation'],
            ]), 
            widgets.HBox([
                self.widgets['max_num_alleles'],
                self.widgets['brightness_fraction'],
                self.widgets['absolute_threshold'],
            ]),
            widgets.HBox([
                self.widgets['run_button'],
                self.widgets['save_button'],
                self.widgets['discard_button'],
            ]),
            self.widgets['loading_label'],
            self.output,
        ])
        
    def _create_widgets(self):
        self.widgets['including_z'] = widgets.Checkbox(
            value=True,
            description='Include Z in localization',
        )
        self.widgets['overwrite'] = widgets.Checkbox(
            value=False,
            description='Overwrite existing localization',
        )
        self.widgets['fov'] = widgets.Dropdown(value=self.fov_list[0],
                                               options=self.fov_list,
                                                  description='FOV:')
        self.widgets['channel'] = widgets.Dropdown(value=self.total_channel_list[0],
                                                   options=self.total_channel_list,
                                                    description='Channel:')
        self.widgets['sub_hybe_list'] = widgets.SelectMultiple(
            options=self.hybe_list,
            value=[h for h in self.hybe_list if h.startswith('H')],
            description='Hybes:',
        )
        self.widgets['cell_id'] = widgets.BoundedIntText(
            value=1,
            min=1,
            max=10000,
            step=1,
            description='Cell ID:',
        )
        self.widgets['lb'] = widgets.BoundedFloatText(
            value=35.0,
            min=0,
            max=90,
            step=.01,
            description='Lower bound:',
        )
        self.widgets['up'] = widgets.BoundedFloatText(
            value=99.9,
            min=91,
            max=100,
            step=.01,
            description='Upper bound:',
        )
        self.widgets['Max/Background'] = widgets.BoundedFloatText(
            value=1.25,
            min=1.0,
            max=100.0,
            step=0.01,
            description='Max/Background:',
        )
        self.widgets['max/avg'] = widgets.BoundedFloatText(
            value=1.25,
            min=1.0,
            max=100.0,
            step=0.01,
            description='Max/Avg:',
        )
        self.widgets['max_num_alleles'] = widgets.BoundedIntText(
            value=2,
            min=1,
            max=20,
            step=1,
            description='Max # Alleles:',
        )
        self.widgets['brightness_fraction'] = widgets.BoundedFloatText(
            value=0.8,
            min=0.5,
            max=1.0,
            step=0.01,
            description='Brightness Fraction:',
        )
        self.widgets['absolute_threshold'] = widgets.BoundedFloatText(
            value=450.0,
            min=0.0,
            max=10000.0,
            step=10,
            description='Absolute Threshold:',
        )
        self.widgets['max_sigma'] = widgets.BoundedFloatText(
            value=3.0,
            min=0.5,
            max=10.0,
            step=0.1,
            description='Max Sigma:',
        )
        self.widgets['max_deviation'] = widgets.BoundedFloatText(
            value=3.0,
            min=0.5,
            max=10.0,
            step=0.1,
            description='Max Deviation:',
        )
        self.widgets['cpad'] = widgets.BoundedIntText(
            value=5,
            min=1,
            max=20,
            step=1,
            description='Cell Padding:',
        )
        self.widgets['spad'] = widgets.BoundedIntText(
            value=5,
            min=1,
            max=20,
            step=1,
            description='Spot Padding:',
        )
        self.widgets['run_button'] = widgets.Button(
            description='Run Localization',
            button_style='success',
            tooltip='Click to run localization',
            icon='check'
        )
        self.widgets['save_button'] = widgets.Button(
            description='Save Localization',
            button_style='info',
            tooltip='Click to save localization',
            icon='save'
        )
        self.widgets['discard_button'] = widgets.Button(
            description='Discard Localization',
            button_style='danger',
            tooltip='Click to discard localization',
            icon='trash'
        )
        self.widgets['loading_label'] = widgets.Label(
            value='Push "Run Localization" to start the process.'
        )

    def _set_events(self):
        self.widgets['run_button'].on_click(lambda x: self.run_localization())
        self.widgets['discard_button'].on_click(lambda x: self.run_discard())
        self.widgets['save_button'].on_click(lambda x: self.run_save())

    def show(self):
        display(self.ui)

    def run_localization(self):
        with self.output:
            clear_output(wait=True)     
            sub_hybe_list = list(self.widgets['sub_hybe_list'].value)
            lb = self.widgets['lb'].value
            up = self.widgets['up'].value
            max_to_background = self.widgets['Max/Background'].value
            max_to_average = self.widgets['max/avg'].value
            frac = self.widgets['brightness_fraction'].value
            cpad = self.widgets['cpad'].value
            spad = self.widgets['spad'].value
            including_z = self.widgets['including_z'].value
            fov = self.widgets['fov'].value
            min_distance = 3
            max_num_alleles = self.widgets['max_num_alleles'].value
            max_sigma = self.widgets['max_sigma'].value
            max_deviation = self.widgets['max_deviation'].value
            absolute_threshold = self.widgets['absolute_threshold'].value
            channel = self.widgets['channel'].value
            n_cpus = self.n_cpus if hasattr(self, 'n_cpus') else os.cpu_count() - 1
            cell_id = self.widgets['cell_id'].value
            overwrite = self.widgets['overwrite'].value
            
            self.widgets['loading_label'].value = 'Localization in progress... Please wait.'
            start_time = time.time()

            # load cell parameters
            with h5py.File(os.path.join(self.h5_save_path, self.filename), 'r') as f:
                hybe_list = [str(h.decode()) for h in f.attrs['hybe_list']]
                cell_mask = f[f'FOV{fov:02d}/cells/mask'][:]
                cell_id_list = np.unique(cell_mask)[1:]
                if cell_id not in cell_id_list:
                    print(f'Cell ID {cell_id} not found in FOV {fov}. Available IDs: {cell_id_list}')
                    self.widgets['loading_label'].value = 'Localization process terminated.'
                    self.widgets['cell_id'].value = cell_id_list[cell_id_list > cell_id][0] if np.any(cell_id_list > cell_id) else cell_id_list[0]
                    return
                
                cid = np.where(cell_id_list == cell_id)[0][0]
                cy,cx = np.where(cell_mask == cell_id)
                cb = f[f'FOV{fov:02d}/cells/cellbarcodes'][:][cid]
                cbid = self.cellbarcode_list.index(cb.decode()) if cb.decode() in self.cellbarcode_list else -1
                matrix_yx_hybes = f[f'FOV{fov:02d}/cells/matrix/yx'][cid]
                matrix_zx_hybes = f[f'FOV{fov:02d}/cells/matrix/zx'][cid]
                cellboundaries = (cell_mask == cell_id).astype(np.uint8) - cv2.erode((cell_mask == cell_id).astype(np.uint8), np.ones((3,3),np.uint8), iterations=1)
            
            ncols = 8
            nrows = int(np.ceil(len(sub_hybe_list)/ncols))
            os.makedirs(os.path.join(self.h5_save_path, 'SpotAnalysis'), exist_ok=True)

            # Parallel localization

            with ProcessPoolExecutor(n_cpus) as executor:
                futures = []
                for hybe in sub_hybe_list:
                    cell_parameter_dict = {
                        'cid': cid,
                        'cb': cbid,
                        'matrix_yx': matrix_yx_hybes[hybe_list.index(hybe)],
                        'matrix_zx': matrix_zx_hybes[hybe_list.index(hybe)],
                        'cy': cy,
                        'cx': cx,
                    }
                    futures.append(executor.submit(
                        localize_spots_worker,
                        fov, hybe, hybe_list, cell_parameter_dict,
                        max_to_background, max_to_average, frac, absolute_threshold, cpad, spad, min_distance, channel, including_z,
                        max_num_alleles, max_sigma, max_deviation,
                        self.h5_save_path,
                    ))
                results = []
                images_dict = {}
                for future in as_completed(futures):
                    xyz,img,hybe = future.result()
                    results.append(xyz)
                    images_dict[hybe] = img

            all_xyzs = np.vstack(results) if len(results) > 0 else np.zeros((0,7), dtype=float)
            if overwrite:
                np.savetxt(os.path.join(self.h5_save_path, 'SpotAnalysis', f'FOV{fov:02d}_Cell{cell_id}_Localization.txt'),
                           all_xyzs, fmt=['%d','%d','%d','%d','%.3f','%.3f','%.3f'],delimiter='\t',
                           header='FOV\tCellID\tHybeIndex\tCellBarcode\tX\tY\tZ', comments='')
            else:
                save_path = os.path.join(self.h5_save_path, 'SpotAnalysis', f'FOV{fov:02d}_Cell{cell_id}_Localization.txt')
                if os.path.exists(save_path):
                    existing_data = np.loadtxt(save_path, dtype=float, delimiter='\t', skiprows=1)
                    all_xyzs = np.vstack([existing_data, all_xyzs])
                np.savetxt(save_path,
                           all_xyzs, fmt=['%d','%d','%d','%d','%.3f','%.3f','%.3f'],delimiter='\t',
                           header='FOV\tCellID\tHybeIndex\tCellBarcode\tX\tY\tZ', comments='')

            # Plotting
            self.widgets['loading_label'].value = 'Generating plots... Please wait.'
            fig,ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*4, nrows*6), squeeze=False)
            for r in range(nrows):
                for c in range(ncols):
                    ax[r,c].axis('off')
            for i,hybe in enumerate(sub_hybe_list):
                r,c = i//ncols, i%ncols
                ax[r,c].axis('on')
                divider = make_axes_locatable(ax[r,c])
                bax = divider.append_axes("bottom", size="30%", pad=0.1)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                img,bimg,xyzs_crop = images_dict[hybe]
                nimg,nbimg = preprocess.normalize_to_uint8(img, lb/100, up/100), preprocess.normalize_to_uint8(bimg, lb/100, up/100)
                im1 = ax[r,c].imshow(nimg, cmap='gray', aspect='auto')
                fig.colorbar(cm.ScalarMappable(cmap=cm.gray, norm=mcolors.Normalize(vmin=np.nanquantile(img,lb/100),
                                                                                    vmax=np.nanquantile(img,up/100))), cax=cax)
                ax[r,c].set_title(f'Hybe: {hybe}', fontsize=12)
                im2 = bax.imshow(nbimg, cmap='gray', aspect='auto')
                ax[r,c].set_xticks([])
                ax[r,c].set_yticks([])
                bax.set_xticks([])
                bax.set_yticks([])
                
                hid = hybe_list.index(hybe)
                H = matrix_yx_hybes[hid].copy()
                matrix_zx = matrix_zx_hybes[hid].copy()
                H[0,2] -= matrix_zx[0,2]
                ry,rx = alignment.align_cell(np.where(cellboundaries > 0), la.inv(H), (cell_mask.shape[0], cell_mask.shape[1]))
                if len(ry) > 0:
                    rymin,rxmin = max(0,ry.min()-cpad), max(0, rx.min()-cpad)        
                    ax[r,c].scatter(rx - rxmin, ry - rymin, c='y', s=1)

                if len(xyzs_crop) > 0:
                    crop_xyz = xyzs_crop[xyzs_crop[:,2] == hybe_list.index(hybe)]
                    if len(crop_xyz) > 0:
                        ax[r,c].scatter(crop_xyz[:,4], crop_xyz[:,5], s=30, edgecolor='red', facecolor='none', label='Localized Spots')
                        bax.scatter(crop_xyz[:,4], crop_xyz[:,6], s=30, edgecolor='red', facecolor='none', label='Localized Spots (Z)')

            plt.tight_layout()
            plt.show()
            plot_end_time = time.time()
            self.widgets['loading_label'].value = f'Plotting completed in {plot_end_time - start_time:.2f} seconds.'
            fig.savefig(os.path.join(self.h5_save_path, 'SpotAnalysis', f'FOV{fov:02d}_Cell{cell_id}_Localization.png'), dpi=300)

    def run_discard(self):
        with self.output:
            clear_output(wait=True)
            self.widgets['loading_label'].value = 'Localization results discarded.'

    def run_save(self):
        with self.output:
            clear_output(wait=True)
            self.widgets['loading_label'].value = 'Localization results saved.'

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

def _build_cell_crop(cell, hybe, channel, storage_path, fov, pad):
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

    Returns None if the cell has no area overlapping this hybe's frame
    (either get_area_in_readout raised KeyError -- no alignment matrix for
    this hybe yet -- or the transformed area is simply empty), otherwise
    a dict: {'img': (h,w) MIP crop (NaN outside cell), 'stacks': (h,w,depth)
    Z-stack crop (NaN outside cell), 'bimg': (w,depth) Z-profile per column
    (nanmax over y), 'rxmin': int, 'rymin': int, 'H': cell's yx matrix for
    this hybe (identity if none), 'Hz': cell's zx matrix for this hybe
    (identity if none)}.
    """
    try:
        x_area, y_area = cell.get_area_in_readout(hybe)
    except KeyError:
        return None
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

    H = cell.matrices.get(hybe, {}).get('yx', np.eye(3))
    Hz = cell.matrices.get(hybe, {}).get('zx', np.eye(3))

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