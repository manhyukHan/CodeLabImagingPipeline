import os
import numpy as np
import numpy.linalg as la
import ipywidgets as widgets
import time
import cellpose.models
import h5py

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from . import preprocess
import cv2

from IPython.display import display, clear_output
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
model_cyto = cellpose.models.Cellpose(model_type='cyto3')

colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
          '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
          '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
          '#000075', '#808080', '#000000']

plt.rcParams['font.family'] = 'arial'
plt.rcParams['font.size'] = 14
plt.rcParams['pdf.fonttype'] = 42

fontdict_label = {'fontname': 'arial',
                  'fontsize':16,
                  }
fontdict_ticks = {'fontname': 'arial',
                  'fontsize': 14,
                  }
fontdict_title = {'fontname': 'arial',
                  'fontsize': 24,
                  'fontweight': 'bold',
                  }

red_to_cyan_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['#FF0000', '#FFFFFF', '#00FFFF'], N=256)

def align_cell(yx, H, shape):
    y,x = yx
    height,width = shape
    cx,cy = (H[:2]@np.array([x,y,np.ones_like(x)])).astype(int)
    bad = (cx < 0) | ( cy < 0) | (cx >= width) | (cy >= height)
    cx,cy = cx[~bad],cy[~bad]
    adjusted_mask = np.zeros((height,width))
    adjusted_mask[cy,cx] = 1
    closed = cv2.morphologyEx(adjusted_mask, cv2.MORPH_CLOSE, kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    cy,cx = np.where(closed > 0)
    return cy,cx

def crop_or_pad_to_shape(img, target_shape, pad_value=0):
    h, w = img.shape[:2]
    H, W = target_shape
    dh, dw = H - h, W - w

    # Compute crop or pad for height
    if dh < 0:
        top = (-dh) // 2
        bottom = top + H
        img = img[top:bottom, :]
    else:
        top_pad = dh // 2
        bottom_pad = dh - top_pad
        img = np.pad(img, ((top_pad, bottom_pad), (0, 0)), constant_values=pad_value)

    # Compute crop or pad for width
    if dw < 0:
        left = (-dw) // 2
        right = left + W
        img = img[:, left:right]
    else:
        left_pad = dw // 2
        right_pad = dw - left_pad
        img = np.pad(img, ((0, 0), (left_pad, right_pad)), constant_values=pad_value)

    return img

class AlignmentWidget(object):
    """
    A widget for aligning multispot microscopy images using Cellpose segmentation.
    """
    def __init__(self, reference_hybe, h5_save_path, filename='vlinks.h5'):
        self.h5_save_path = h5_save_path
        self.filename = filename
        with h5py.File(os.path.join(h5_save_path,self.filename), 'r') as f:
            self.fov_list = f.attrs['fov_list'][:].tolist()
            self.channels_list = f.attrs['channels_list'][:].tolist()
            self.hybe_list = [str(h.decode()) for h in f.attrs['hybe_list']]

        self.total_channel_list = np.unique(np.ravel(self.channels_list)).tolist()
        self.reference_hybe = reference_hybe


        self.create_widgets()

    def generate_colormap(self, sub_hybe_list):
        colormaps = {
            hybe: mcolors.LinearSegmentedColormap.from_list(
                f'{hybe}',
                [(0, 0, 0, 1), red_to_cyan_cmap(int(hid / len(sub_hybe_list) * 256))],
                256,
                gamma=1.8
            )
            for hid, hybe in enumerate(sub_hybe_list)
        }
        return colormaps

    def create_widgets(self):
        self.widgets = {}
        self.widgets['including_z'] = widgets.Checkbox(value=True, description='Including Z:',)
        self.widgets['reference_hybe'] = widgets.Dropdown(value=self.reference_hybe, options=self.hybe_list, description='Reference Hybe:',)
        self.widgets['sub_hybe_list'] = widgets.SelectMultiple(options=self.hybe_list, value=tuple(self.hybe_list), description='Hybes to Align:',)

        self.widgets['reference_channel'] = widgets.Dropdown(value=self.total_channel_list[0], options=self.total_channel_list, description='reference Channel:',)
        self.widgets['fov_list'] = widgets.BoundedIntText(value=self.fov_list[0], min=1, max=max(self.fov_list)+1, step=1, description='FOV:',)
        self.widgets['pad'] = widgets.BoundedIntText(value=0, min=0, max=100, step=5, description='Pad:',)
        self.widgets['id_spinbox'] = widgets.BoundedIntText(value=1, min=1, max=10000, step=1, description='Cell ID:',)
        self.widgets['lb'] = widgets.BoundedFloatText(value=85, min=0, max=90, step=0.01, description='Lower Bound:',)
        self.widgets['ub'] = widgets.BoundedFloatText(value=99.99, min=90, max=100, step=0.01, description='Upper Bound:',)
        
        self.widgets['align_button'] = widgets.Button(description='Align', button_style='success',)
        self.widgets['save_and_pass_button'] = widgets.Button(description='Save and Pass', button_style='success',)
        self.widgets['discard_button'] = widgets.Button(description='Discard', button_style='danger',)
        self.widgets['spinbox_align_buttons'] = widgets.HBox([self.widgets['id_spinbox'], self.widgets['align_button']])
        
        self.widgets['loading_label'] = widgets.Label(value='Push "Align" to start aligning cell... please wait.',)
        
        self.output = widgets.Output()

        self.mips_by_hybe = {hybe:None for hybe in self.hybe_list}

        self.figure = None
        self.axes = None
        self.current_image_parameters = {'lb':0, 'ub':0, 'fov':0, 'reference_channel':0}

        self.ui = widgets.VBox([
            widgets.Text(value=self.h5_save_path, description='Save Directory:',),
            self.widgets['including_z'],
            self.widgets['reference_hybe'],
            self.widgets['sub_hybe_list'],
            self.widgets['fov_list'],
            self.widgets['reference_channel'],
            self.widgets['pad'],
            self.widgets['lb'],
            self.widgets['ub'],
            self.widgets['spinbox_align_buttons'],
            widgets.HBox([self.widgets['save_and_pass_button'], self.widgets['discard_button']]),
            self.widgets['loading_label'],
            self.output
        ])

        self.widgets['save_and_pass_button'].on_click(lambda b: self.run_save_and_pass())
        self.widgets['discard_button'].on_click(lambda b: self.run_discard())
        self.widgets['align_button'].on_click(lambda b: self.run_cell_based_align())

    def show(self):
        display(self.ui)

    def run_cell_based_align(self,):
        with self.output:
            clear_output(wait=True)
            self.widgets['loading_label'].value = "🔄 Aligning cell... please wait."
            start = time.time()
            try:
                lb = self.widgets['lb'].value/100
                ub = self.widgets['ub'].value/100
                
                fov = self.widgets['fov_list'].value
                reference_channel = self.widgets['reference_channel'].value
                id = self.widgets['id_spinbox'].value

                cell_align_dir = os.path.join(self.h5_save_path, f'FOV{fov:02d}/Aligned_Cells')
                os.makedirs(cell_align_dir, exist_ok=True)
                os.makedirs(os.path.join(cell_align_dir, 'Pass'), exist_ok=True)
                os.makedirs(os.path.join(cell_align_dir, 'Discard'), exist_ok=True)
                
                t0 = time.time()
                with h5py.File(os.path.join(self.h5_save_path, self.filename), 'r') as f:
                    mask = f[f'FOV{fov:02d}/cells/mask'][:]
                    id_list = np.unique(mask[mask > 0]).tolist()
                    if (id == 1) and (id not in id_list):
                        id = id_list[0]
                        self.widgets['id_spinbox'].value = id
                    elif id not in id_list:
                        id = np.array(id_list)[np.array(id_list) > id][0]
                        self.widgets['loading_label'].value = f'⚠️ Cell ID {self.widgets["id_spinbox"].value} not found in FOV {fov}. Loading next available ID {id}.'
                        self.widgets['id_spinbox'].value = id
                    condition = f[f'FOV{fov:02d}/cells/cellbarcodes'][id_list.index(id)].decode('utf-8')

                    matrices_fov = np.zeros((len(self.hybe_list),3,3))
                    for hid,hybe in enumerate(self.hybe_list): matrices_fov[hid] = f[f'FOV{fov:02d}/matrix/{hybe}'][:]
                        
                    if (fov != self.current_image_parameters['fov']) or (reference_channel != self.current_image_parameters['reference_channel']):  
                        self.current_image_parameters['lb'] = lb
                        self.current_image_parameters['ub'] = ub
                        self.current_image_parameters['fov'] = fov
                        self.current_image_parameters['reference_channel'] = reference_channel
                        
                        for hid,hybe in enumerate(self.hybe_list): self.mips_by_hybe[hybe] = preprocess.normalize_to_uint8(f[f'FOV{fov:02d}/mip/ch{reference_channel}'][hid,...],lb,ub)
                    print(f'Loading time: {time.time()-t0:.2f} seconds')
                        
                if id == 1 and id not in id_list:
                    id = id_list[0]
                    self.widgets['id_spinbox'].value = id
                if np.sum(mask == id) == 0:
                    self.widgets['loading_label'].value = f'❌ No cells found with ID {id} in FOV {fov}.'
                    return
                self.cell_based_align(id, condition, mask, matrices_fov, )
            finally:
                self.widgets['loading_label'].value = f"✅ Cell aligned successfully. {time.time()-start:.2f} seconds."
                display(self.figure)

    def cell_based_align(self, id, condition, mask, matrices_fov, ):
        # parameters
        lb,ub = self.current_image_parameters['lb'], self.current_image_parameters['ub']
        pad = self.widgets['pad'].value
        including_z = self.widgets['including_z'].value
        reference_hybe = str(self.widgets['reference_hybe'].value)
        id_list = np.unique(mask[mask > 0])
        id_of_mask = np.where(id_list == id)[0][0]
        height,width = mask.shape
        reference_channel = self.widgets['reference_channel'].value
        fov = self.widgets['fov_list'].value
        sub_hybe_list = list(self.widgets['sub_hybe_list'].value)
        colormaps = self.generate_colormap(sub_hybe_list)

        H1 = matrices_fov[self.hybe_list.index(reference_hybe)]
        y,x = np.where(mask == id)
        ry,rx = align_cell((y,x), la.inv(H1), (height,width))
        rymin,rymax,rxmin,rxmax = max(0,ry.min()-pad), min(height,ry.max()+pad+1), max(0,rx.min()-pad), min(width,rx.max()+pad+1)
        reference_image_norm = self.mips_by_hybe[reference_hybe]
        reference_image_crop = reference_image_norm[rymin:rymax,rxmin:rxmax]

        ycomp1,ycomp2,ycomp3 = [np.zeros((rymax-rymin,rxmax-rxmin,4,len(sub_hybe_list)),dtype=float) for _ in range(3)]

        with h5py.File(os.path.join(self.h5_save_path,self.filename ), 'r') as f:
            matrices_yx = f[f'FOV{fov:02d}/cells/matrix/yx'][id_of_mask,:]
            matrices_zx = f[f'FOV{fov:02d}/cells/matrix/zx'][id_of_mask,:]

        for shid,hybe in enumerate(sub_hybe_list):
            hid = self.hybe_list.index(hybe)
            cmap = colormaps[hybe]
            if hybe != reference_hybe:
                target_image_norm = self.mips_by_hybe[hybe]
                H1 = matrices_fov[hid]
                cy,cx = align_cell((ry,rx), la.inv(H1), (height,width))
                cymin,cymax,cxmin,cxmax = max(0,cy.min()-pad), min(height,cy.max()+pad+1), max(0,cx.min()-pad), min(width,cx.max()+pad+1)
                target_image_crop = target_image_norm[rymin:rymax,rxmin:rxmax]
                target_image_crop_first = target_image_norm[cymin:cymax,cxmin:cxmax]

                H2 = np.vstack([preprocess.find_translation_via_phase_correlation(target_image_crop_first,
                                                                                  reference_image_crop),
                                                                                  np.array([0,0,1])])
                matrices_yx[hid] = H2
                target_image_crop_final = cv2.warpAffine(target_image_crop_first, H2[:2], (cxmax-cxmin,cymax-cymin), )

                target_image_crop_ = target_image_crop.astype(float) / target_image_crop.max() * 255
                target_image_crop_first_ = target_image_crop_first.astype(float) / target_image_crop_first.max() * 255
                target_image_crop_final_ = target_image_crop_final.astype(float) / target_image_crop_final.max() * 255
                ycomp1[...,shid] = cmap(target_image_crop_.astype(np.uint8))
                ycomp2[...,shid] = cmap(crop_or_pad_to_shape(target_image_crop_first_.astype(np.uint8),(rymax-rymin,rxmax-rxmin)))
                ycomp3[...,shid] = cmap(crop_or_pad_to_shape(target_image_crop_final_.astype(np.uint8),(rymax-rymin,rxmax-rxmin)))
            else:
                reference_image_crop_ = reference_image_crop.astype(float) / reference_image_crop.max() * 255
                ycomp1[...,shid] = cmap(reference_image_crop_.astype(np.uint8))
                ycomp2[...,shid] = ycomp1[...,shid]
                ycomp3[...,shid] = ycomp1[...,shid]
            
        if including_z:
            with h5py.File(os.path.join(self.h5_save_path,self.filename), 'r') as f:
                depth = f.attrs['stack_shape'][0]
                hid = self.hybe_list.index(reference_hybe)
                stacks = f[f'FOV{fov:02d}/stack/ch{reference_channel}'][hid,:,rymin:rymax,rxmin:rxmax].squeeze()
                zcomp1,zcomp2,zcomp3 = [np.zeros((depth,rxmax-rxmin,4,len(sub_hybe_list)),dtype=float) for _ in range(3)]
                stack_ref_zx = preprocess.normalize_to_uint8(stacks.max(1),lb,ub)
                
                for shid,hybe in enumerate(sub_hybe_list):
                    hid = self.hybe_list.index(hybe)
                    cmap = colormaps[hybe]
                    
                    if hybe != reference_hybe:
                        stacks = f[f'FOV{fov:02d}/stack/ch{reference_channel}'][hid,:,rymin:rymax,rxmin:rxmax].squeeze()
                        stack_hyb_zx = preprocess.normalize_to_uint8(stacks.max(1),lb,ub)
                        H1 = matrices_fov[hid]
                        cy,cx = align_cell((ry,rx), la.inv(H1), (height,width))
                        cymin,cymax,cxmin,cxmax = max(0,cy.min()-pad), min(height,cy.max()+pad+1), max(0,cx.min()-pad), min(width,cx.max()+pad+1)

                        stacks = f[f'FOV{fov:02d}/stack/ch{reference_channel}'][hid,:,cymin:cymax,cxmin:cxmax].squeeze()
                        stack_fov_zx = preprocess.normalize_to_uint8(stacks.max(1),lb,ub)

                        H2 = matrices_yx[hid]
                        stack_mip_zx = preprocess.normalize_to_uint8(np.concatenate([cv2.warpAffine(stacks[j], H2[:2],
                                                                                                    (cxmax-cxmin,cymax-cymin),)[None,...]
                                                                                                      for j in range(depth)], axis=0).max(1),lb,ub)

                        A3 = preprocess.find_translation_via_phase_correlation(stack_mip_zx, stack_ref_zx)
                        zcomp1[...,shid] = cmap(stack_hyb_zx)
                        zcomp2[...,shid] = cmap(crop_or_pad_to_shape(stack_fov_zx, (depth,rxmax-rxmin)))
                        zcomp3[...,shid] = cmap(crop_or_pad_to_shape(cv2.warpAffine(stack_mip_zx, A3, (cxmax-cxmin,depth)),(depth,rxmax-rxmin)))
                        H3 = np.vstack([A3[:2],np.array([0,0,1])])
                        matrices_zx[hid] = H3
                    else:
                        zcomp1[...,shid] = cmap(stack_ref_zx)
                        zcomp2[...,shid] = zcomp1[...,shid]
                        zcomp3[...,shid] = zcomp1[...,shid]
                
        with h5py.File(os.path.join(self.h5_save_path,self.filename), 'r+') as f:
            f[f'/FOV{fov:02d}/cells/matrix/yx'][id_of_mask,:] = matrices_yx
            f[f'/FOV{fov:02d}/cells/matrix/zx'][id_of_mask,:] = matrices_zx

        if not including_z:
            fig,ax = plt.subplots(1,3,figsize=(18,6))
            ax[0].imshow(ycomp1.max(-1),)
            ax[1].imshow(ycomp2.max(-1),)
            ax[2].imshow(ycomp3.max(-1),)
            cell = np.zeros((rymax-rymin,rxmax-rxmin),dtype=np.uint8)
            cell[ry-rymin,rx-rxmin] = 1
            boundaries = (cell).astype(np.uint8) - cv2.erode((cell).astype(np.uint8), np.ones((3,3),np.uint8), iterations=1)
            y,x = np.where(boundaries > 0)
            ax[0].scatter(x,y, color='yellow', s=10, marker='s', alpha=.5)
            ax[1].scatter(x,y, color='yellow', s=10, marker='s', alpha=.5)
            ax[2].scatter(x,y, color='yellow', s=10, marker='s', alpha=.5)
            ax[0].axis('off')
            ax[1].axis('off')
            ax[2].axis('off')
            ax[0].set_title(f'Original Image\nRED: {sub_hybe_list[0]} CYAN: {sub_hybe_list[-1]}\nFOV: {fov} Cell ID: {id} Barcode {condition}', fontdict=fontdict_label)
            ax[1].set_title(f'FOV Aligned Image\nRED: {sub_hybe_list[0]} CYAN: {sub_hybe_list[-1]}\nFOV: {fov} Cell ID: {id} Barcode {condition}', fontdict=fontdict_label)
            ax[2].set_title(f'Final Image\nRED: {sub_hybe_list[0]} CYAN: {sub_hybe_list[-1]}\nFOV: {fov} Cell ID: {id} Barcode {condition}', fontdict=fontdict_label)
        else:
            fig,ax = plt.subplots(2,3,figsize=(18,12))
            ax[0,0].imshow(ycomp1.max(-1),aspect='auto')
            ax[0,1].imshow(ycomp2.max(-1),aspect='auto')
            ax[0,2].imshow(ycomp3.max(-1),aspect='auto')
            cell = np.zeros((rymax-rymin,rxmax-rxmin),dtype=np.uint8)
            cell[ry-rymin,rx-rxmin] = 1
            boundaries = (cell).astype(np.uint8) - cv2.erode((cell).astype(np.uint8), np.ones((3,3),np.uint8), iterations=1)
            y,x = np.where(boundaries > 0)
            ax[0,0].scatter(x,y, color='yellow', s=10, marker='s', alpha=.5)
            ax[0,1].scatter(x,y, color='yellow', s=10, marker='s', alpha=.5)
            ax[0,2].scatter(x,y, color='yellow', s=10, marker='s', alpha=.5)
            ax[0,0].set_title(f'Original Image\nRED: {sub_hybe_list[0]} CYAN: {sub_hybe_list[-1]}\nFOV: {fov} Cell ID: {id} Barcode {condition}', fontdict=fontdict_label)
            ax[0,1].set_title(f'FOV Aligned Image\nRED: {sub_hybe_list[0]} CYAN: {sub_hybe_list[-1]}\nFOV: {fov} Cell ID: {id} Barcode {condition}', fontdict=fontdict_label)
            ax[0,2].set_title(f'Final Image\nRED: {sub_hybe_list[0]} CYAN: {sub_hybe_list[-1]}\nFOV: {fov} Cell ID: {id} Barcode {condition}', fontdict=fontdict_label)
            ax[1,0].imshow(zcomp1.max(-1),aspect='auto')
            ax[1,1].imshow(zcomp2.max(-1),aspect='auto')
            ax[1,2].imshow(zcomp3.max(-1),aspect='auto')
        
            ax[0,0].set_xticks([])
            ax[0,1].set_xticks([])
            ax[0,2].set_xticks([])
            ax[0,0].set_yticks([])
            ax[0,1].set_yticks([])
            ax[0,2].set_yticks([])
            ax[1,0].set_xticks([])
            ax[1,1].set_xticks([])
            ax[1,2].set_xticks([])
            ax[1,0].set_yticks([])
            ax[1,1].set_yticks([])
            ax[1,2].set_yticks([])
            ax[0,0].set_ylabel('Y',rotation=0,ha='right',**fontdict_label)
            ax[0,1].set_ylabel('Y',rotation=0,ha='right',**fontdict_label)
            ax[0,2].set_ylabel('Y',rotation=0,ha='right',**fontdict_label)
            ax[1,0].set_ylabel('Z',rotation=0,ha='right',**fontdict_label)
            ax[1,1].set_ylabel('Z',rotation=0,ha='right',**fontdict_label)
            ax[1,2].set_ylabel('Z',rotation=0,ha='right',**fontdict_label)
            ax[1,0].set_xlabel('X',**fontdict_label)
            ax[1,1].set_xlabel('X',**fontdict_label)
            ax[1,2].set_xlabel('X',**fontdict_label)

        plt.tight_layout()
        self.figure = fig
        self.axes = ax
        plt.close(fig)

    def run_save_and_pass(self):
        with self.output:
            clear_output(wait=True)
            start = time.time()
            fov = self.widgets['fov_list'].value
            id = self.widgets['id_spinbox'].value
            cell_align_dir = os.path.join(self.h5_save_path, f'FOV{fov:02d}/Aligned_Cells')

            with h5py.File(os.path.join(self.h5_save_path,self.filename), 'r+') as f:
                mask = f[f'FOV{fov:02d}/cells/mask'][:]
                id_list = np.unique(mask)[1:].tolist()
                if id not in f[f'FOV{fov:02d}/cells'].attrs['good'] and id in id_list:
                    f[f'FOV{fov:02d}/cells'].attrs['good'] = np.unique(np.concatenate((f[f'FOV{fov:02d}/cells'].attrs['good'], [id])))
                
                self.figure.savefig(os.path.join(cell_align_dir,'Pass',
                                                 f'aligned_reference_{self.widgets["reference_hybe"].value}_FOV{fov:02d}_ID{id}.png'))
            if id_list.index(id)+1 < len(id_list):
                new_id = id_list[id_list.index(id)+1]
                self.widgets['id_spinbox'].value = new_id
                self.run_cell_based_align()
            else:
                self.widgets['loading_label'].value = f'✅ All cells aligned in FOV {fov}. {time.time()-start:.2f} seconds.'
                return
            
    def run_discard(self):
        with self.output:
            clear_output(wait=True)
            start = time.time()
            fov = self.widgets['fov_list'].value
            id = self.widgets['id_spinbox'].value
            cell_align_dir = os.path.join(self.h5_save_path, f'FOV{fov:02d}/Aligned_Cells')

            with h5py.File(os.path.join(self.h5_save_path,self.filename), 'r+') as f:
                mask = f[f'FOV{fov:02d}/cells/mask'][:]
                id_list = np.unique(mask)[1:].tolist()
                if id in id_list:
                    f[f'FOV{fov:02d}/cells'].attrs['good'] = np.unique(np.array([i for i in f[f'FOV{fov:02d}/cells'].attrs['good'] if i != id]))
                self.figure.savefig(os.path.join(cell_align_dir,'Discard',
                                                 f'aligned_reference_{self.widgets["reference_hybe"].value}_FOV{fov:02d}_ID{id}.png'))
            if id_list.index(id)+1 < len(id_list):
                new_id = id_list[id_list.index(id)+1]
                self.widgets['id_spinbox'].value = new_id
                self.run_cell_based_align()
            else:
                print(f'No more cells to align in FOV {fov}.')
                return
