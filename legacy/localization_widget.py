"""
Dead code -- see legacy/README.md. ipywidgets/Jupyter-notebook UI, never
instantiated by this project's PyQt5 GUI. Superseded by localize_cells_2d /
localize_cells_3d in codelab_pipeline/localization/localization.py, which
ui/spot_localization_panel.py drives instead. Moved here verbatim
(unmodified) on 2026-08-07.
"""
import os
import time

import numpy as np
import numpy.linalg as la
from concurrent.futures import ProcessPoolExecutor, as_completed

import ipywidgets as widgets
import h5py
import cv2

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from IPython.display import display, clear_output
from mpl_toolkits.axes_grid1 import make_axes_locatable

from codelab_pipeline.io import preprocess
from codelab_pipeline.alignment import chain as alignment
from codelab_pipeline.localization.localization import localize_spots_worker


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
