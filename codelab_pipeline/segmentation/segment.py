import os
import sys
import numpy as np
import ipywidgets as widgets
import h5py
from skimage import filters as skimage_filters, morphology as skimage_morphology, segmentation as skimage_segmentation
from skimage.feature import peak_local_max
from scipy import ndimage as scind

from IPython.display import display, clear_output, HTML
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="cellpose")

_model_cyto = None

def get_model_cyto():
    """Lazily load the Cellpose cyto3 model on first use, not at import time."""
    global _model_cyto
    if _model_cyto is None:
        import cellpose.models
        _model_cyto = cellpose.models.Cellpose(gpu=True, model_type='cyto3')
    return _model_cyto

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..io import preprocess

import time
import cv2

red_to_cyan_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['#FF0000', '#FFFFFF', '#00FFFF'], N=256)

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
colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
          '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
          '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
          '#000075', '#808080', '#000000']

def segment_fov(storage_path, fov, reference_hybe, channel, diameter=40, min_size=1000, max_size=10000):
    """
    Bulk (non-interactive) cell segmentation for one FOV -- reads the
    reference MIP straight from that hybe's per-hybe H5 file (the current
    ingestion convention: {storage_path}/FOV{fov:02d}/{hybe}_stack.h5,
    /mip/ch{channel}), not the old vlinks.h5-based layout SegmentWidget used.
    Returns (mask, reference_image); doesn't display or save anything itself
    -- matches localize_cells_2d's separation of computation from I/O, so the
    GUI can run this off the main thread and review the result before saving.

    Core Cellpose-call + size-filter + relabel logic mirrors
    SegmentWidget.create_mask_in_reference_hybe below, minus its
    Jupyter/plotting/H5-write scaffolding.
    """
    h5path = os.path.join(storage_path, f'FOV{fov:02d}', f'{reference_hybe}_stack.h5')
    with h5py.File(h5path, 'r') as f:
        reference_image = f[f'/mip/ch{channel}'][:]

    # eval()'s return tuple length varies by cellpose version/model class
    # (3-tuple for CellposeModel/cpsam, 4-tuple for the classical
    # Cellpose/cyto3 class used here) -- take masks by position only,
    # matching CellClassifier/utils/cellpose_segmentation.py's approach.
    try:
        result = get_model_cyto().eval([reference_image], diameter=diameter, channels=[0, 0], do_3D=False)
    except Exception:
        # masks_to_flows_gpu's boundary IndexError (MouseLand/cellpose#1004
        # and others) -- MPS in particular is a much less mature PyTorch
        # backend than CUDA, already hit here and independently in
        # CellClassifier/utils/cellpose_segmentation.py. Retry on CPU with a
        # fresh model instance rather than propagating; doesn't touch the
        # cached GPU singleton, so later calls still try GPU first.
        import cellpose.models
        cpu_model = cellpose.models.Cellpose(gpu=False, model_type='cyto3')
        result = cpu_model.eval([reference_image], diameter=diameter, channels=[0, 0], do_3D=False)
    masks = result[0]
    mask = masks[0].astype(float)
    mask = _filter_and_relabel(mask, min_size, max_size)

    return mask, reference_image


def _filter_and_relabel(mask, min_size, max_size):
    """
    Shared by every segmentation method (Cellpose, classical, and manual's
    own additive commits use the same convention): drop labels outside
    [min_size, max_size] by area, then relabel sequentially starting at 1
    (0 stays background). Starting enumerate() at 0 (as the legacy
    SegmentWidget.create_mask_in_reference_hybe below does) maps the first
    valid cell to -0 == 0 and silently drops it into the background --
    fixed here via start=1, not propagated.
    """
    v, c = np.unique(mask, return_counts=True)
    mask = mask.copy()
    mask[np.isin(mask, v[c < min_size])] = 0
    mask[np.isin(mask, v[c > max_size])] = 0
    for i, id in enumerate(np.unique(mask)[1:], start=1):
        mask[mask == id] = -i
    return (-1 * mask).astype(np.uint8)


def segment_fov_classical(storage_path, fov, reference_hybe, channel, method='otsu',
                          absolute_cutoff=None, min_distance=7, min_size=1000, max_size=10000):
    """
    Bulk (non-interactive) classical threshold+watershed cell segmentation
    for one FOV -- same I/O contract as segment_fov (reads
    {storage_path}/FOV{fov:02d}/{reference_hybe}_stack.h5's /mip/ch{channel},
    returns (mask, reference_image)). Ports
    CellClassifier/canvas/main_image_canvas.py::_runCellSegment's classical
    branch (method in {'otsu','yen','li','triangle','manual'}), minus its
    pyqtgraph/live-canvas coupling. 'manual' there meant "type an absolute
    intensity cutoff by hand" (unrelated to this project's own polygon-draw
    "Manual" segmentation method) -- renamed here to method='absolute' with
    an explicit absolute_cutoff value, to avoid that name collision.

    CellClassifier's original call used
    peak_local_max(distance_transformed, indices=False, min_distance=...),
    an indices=False boolean-mask return mode removed from the skimage
    version already used elsewhere in this project (localization.py never
    passes indices=). The boolean marker mask is rebuilt manually from the
    returned peak coordinates instead.
    """
    h5path = os.path.join(storage_path, f'FOV{fov:02d}', f'{reference_hybe}_stack.h5')
    with h5py.File(h5path, 'r') as f:
        reference_image = f[f'/mip/ch{channel}'][:]

    if method == 'absolute':
        cutoff = float(absolute_cutoff)
    elif method == 'yen':
        cutoff = skimage_filters.threshold_yen(reference_image)
    elif method == 'otsu':
        cutoff = skimage_filters.threshold_otsu(reference_image)
    elif method == 'triangle':
        cutoff = skimage_filters.threshold_triangle(reference_image)
    elif method == 'li':
        cutoff = skimage_filters.threshold_li(reference_image)
    else:
        raise ValueError(f"Unknown classical segmentation method '{method}'")

    binary = (reference_image > cutoff).astype(np.uint8)
    opened_binary = skimage_morphology.opening(binary)
    distance_transformed = scind.distance_transform_edt(opened_binary)

    peak_coords = peak_local_max(distance_transformed, min_distance=min_distance, labels=opened_binary)
    local_max = np.zeros_like(distance_transformed, dtype=bool)
    local_max[tuple(peak_coords.T)] = True
    markers, _ = scind.label(local_max, structure=np.ones((3, 3)))
    mask = skimage_segmentation.watershed(-distance_transformed, markers, mask=opened_binary).astype(float)
    mask = _filter_and_relabel(mask, min_size, max_size)

    return mask, reference_image


class SegmentWidget(object):
    """Base class for creating widgets in Jupyter Notebooks.
    """
    def __init__(self, fov_list, hybe_list, channel_list, reference_hybe, h5_save_path, filename='vlinks.h5'):
        self.fov_list = fov_list
        self.hybe_list = hybe_list
        self.channel_list = channel_list
        self.reference_hybe = reference_hybe
        self.h5_save_path = h5_save_path
        self.filename = filename
        self.create_widgets()

    def create_widgets(self):
        self.widgets = {}
        self.widgets['fov_list'] = widgets.BoundedIntText(value=min(self.fov_list),min=min(self.fov_list),max=max(self.fov_list)+1,step=1,description='FOV:',)
        self.widgets['reference_hybe'] = widgets.Dropdown(value=self.reference_hybe,options=self.hybe_list,description='Reference Hybe:',)
        self.widgets['channel_list'] = widgets.Dropdown(value=self.channel_list[0],options=self.channel_list,description='Channel:',)
        self.widgets['diameter'] = widgets.IntSlider(value=40,min=10,max=100,step=5,description='Diameter:',)
        self.widgets['min_size'] = widgets.IntSlider(value=1000,min=500,max=10000,step=100,description='Min Size:',)
        self.widgets['max_size'] = widgets.IntSlider(value=10000,min=10000,max=20000,step=1000,description='Max Size:',)
        self.widgets['bad_indices'] = widgets.Text(value='',description='Bad Indices:',)
        self.widgets['create_mask_button'] = widgets.Button(description='Create Cells', button_style='success',)
        self.widgets['update_mask_button'] = widgets.Button(description='Update Cells', button_style='success',)
        self.widgets['save_mask_button'] = widgets.Button(description='Save Cells', button_style='success',)
        self.widgets['loading_label'] = widgets.Label(value='Press "Create Cells" value')
        
        self.output = widgets.Output()

        self.figure = None
        self.axes = None
        self.cells_and_align_matrices_fov = {fov:{'image':None, 'mask':None, 'hybe': None, 'channel': None,} for fov in self.fov_list}
        self.reference_image = np.zeros((1024,1024),dtype=np.uint16)

        self.ui = widgets.VBox([self.widgets['fov_list'],self.widgets['reference_hybe'],self.widgets['channel_list'],self.widgets['diameter'],self.widgets['min_size'],self.widgets['max_size'],self.widgets['bad_indices'],
                                widgets.HBox([self.widgets['create_mask_button'], self.widgets['update_mask_button'], self.widgets['save_mask_button']]),
                                self.widgets['loading_label'],
                                self.output])

        
        self.widgets['create_mask_button'].on_click(lambda b: self.run_segmentation())
        self.widgets['update_mask_button'].on_click(lambda b: self.run_update_mask())
        self.widgets['save_mask_button'].on_click(lambda b: self.run_save_mask())
        
    def show(self):
        display(self.ui)

    def run_segmentation(self,):
        with self.output:
            clear_output(wait=True)
            self.widgets['loading_label'].value = "🔄 Segmenting cell... please wait."
            start = time.time()
            
            try:
                fig,ax = plt.subplots(1,2,figsize=(22,10))
                fov = self.widgets['fov_list'].value
                reference_hybe = self.widgets['reference_hybe'].value
                channel = self.widgets['channel_list'].value
                diameter = self.widgets['diameter'].value
                min_size = self.widgets['min_size'].value
                max_size = self.widgets['max_size'].value
                
                mask,reference_image = self.create_mask_in_reference_hybe(ax,fov, reference_hybe, channel, self.h5_save_path, diameter, min_size, max_size)
                self.cells_and_align_matrices_fov[fov]['image'] = reference_image
                self.cells_and_align_matrices_fov[fov]['mask'] = mask
                self.cells_and_align_matrices_fov[fov]['hybe'] = reference_hybe
                self.cells_and_align_matrices_fov[fov]['channel'] = channel
                plt.show()
                self.figure = fig
                self.axes = ax
            finally:
                self.widgets['loading_label'].value = f"✅ Done.  {time.time()-start:.2f} seconds."

    def create_mask_in_reference_hybe(self, ax, fov, reference_hybe, channel, h5path, diameter, min_size, max_size,):
        with h5py.File(os.path.join(h5path,f'vlinks.h5'),'r') as f:
            hybe_list = [str(h.decode()) for h in f.attrs['hybe_list']]
            reference_image = f[f'/FOV{fov:02d}/mip/ch{channel}'][hybe_list.index(reference_hybe),...]

        ax[0].set_title('Reference Image', **fontdict_label)
        ax[1].set_title('Cell Mask', **fontdict_label)
        ax[0].axis('off')
        ax[1].axis('off')
        ax[0].imshow(reference_image, cmap=cm.gray,
                      vmin=np.quantile(reference_image,0.3),
                      vmax=np.quantile(reference_image,0.999), )
        ax[1].imshow(reference_image, cmap=cm.gray,
                    vmin=np.quantile(reference_image,0.3),
                    vmax=np.quantile(reference_image,0.999), )

        # eval()'s return tuple length varies by cellpose version/model class
        # (3-tuple for CellposeModel/cpsam, 4-tuple for the classical
        # Cellpose/cyto3 class used here) -- take masks by position only,
        # matching CellClassifier/utils/cellpose_segmentation.py's approach,
        # so this doesn't break if that changes.
        result = get_model_cyto().eval([reference_image], diameter=diameter, channels=[0,0], do_3D=False)
        masks = result[0]
        mask = masks[0].astype(float)
        v,c = np.unique(mask, return_counts=True)
        mask[np.isin(mask, v[c < min_size])] = 0
        mask[np.isin(mask, v[c > max_size])] = 0
        for i,id in enumerate(np.unique(mask)[1:]):
            mask[mask == id] = -i
        mask = (-1*mask).astype(np.uint8)
        
        for id in np.unique(mask)[1:]:
            cb = (mask == id).astype(np.uint8) - cv2.erode((mask == id).astype(np.uint8), np.ones((3,3),np.uint8), iterations=1)
            y,x = np.where(cb > 0)
            ax[1].scatter(x,y, color='yellow', s=.5, alpha=.5)
            ax[1].text(x.mean(),y.mean(), str(id), color='red', fontsize=12, ha='center', va='center')

        return mask, reference_image
                
    def run_update_mask(self,):
        with self.output:
            clear_output(wait=True)
            self.widgets['loading_label'].value = "🔄 Updating cell mask... please wait."
            start = time.time()

            try:
                fig,ax = plt.subplots(1,2,figsize=(22,10))
                fov = self.widgets['fov_list'].value
                bad_indices = [int(i) for i in self.widgets['bad_indices'].value.split() if i.strip()]
                mask = self.cells_and_align_matrices_fov[fov]['mask']
                reference_image = self.cells_and_align_matrices_fov[fov]['image']
                mask = self.update_mask(ax, reference_image, mask, bad_indices)
                self.cells_and_align_matrices_fov[fov]['mask'] = mask
                plt.show()
                self.figure = fig
                self.axes = ax

            finally:
                self.widgets['loading_label'].value = f"✅ Done. {time.time()-start:.2f} seconds."

    def update_mask(self, ax, reference_image, mask, bad_indices):
        ax[0].set_title('Reference Image', **fontdict_label)
        ax[1].set_title('Cell Mask', **fontdict_label)
        ax[0].axis('off')
        ax[1].axis('off')
        ax[0].imshow(reference_image, cmap=cm.gray,
                    vmin=np.quantile(reference_image,0.3),
                    vmax=np.quantile(reference_image,0.999), )
        # Replot the updated mask
        ax[1].imshow(reference_image, cmap=cm.gray,
                    vmin=np.quantile(reference_image,0.3),
                    vmax=np.quantile(reference_image,0.999), )

        # Update the mask with bad indices
        for id in bad_indices:
            mask[mask == id] = 0

        for id in np.unique(mask)[1:]:
            cb = (mask == id).astype(np.uint8) - cv2.erode((mask == id).astype(np.uint8), np.ones((3,3),np.uint8), iterations=1)
            y,x = np.where(cb > 0)
            ax[1].scatter(x,y, color='yellow', s=.5, alpha=.5)
            ax[1].text(x.mean(),y.mean(), str(id), color='red', fontsize=12, ha='center', va='center')

        return mask

    def run_save_mask(self,):
        with self.output:
            clear_output(wait=True)
            self.widgets['loading_label'].value = "🔄 Saving cell mask... please wait."
            start = time.time()
            fov = self.widgets['fov_list'].value
            reference_hybe = self.widgets['reference_hybe'].value
            channel = self.widgets['channel_list'].value
            h5_save_path = self.h5_save_path
            mask = self.cells_and_align_matrices_fov[fov]['mask']
            
            with h5py.File(os.path.normpath(os.path.join(h5_save_path, self.filename)), 'r+') as f:
                cell_group = f[f'/FOV{fov:02d}/cells']
                cell_group.attrs['defined_hybe'] = reference_hybe
                cell_group.attrs['defined_channel'] = channel
                cell_group['mask'][:] = mask
                id_list = np.unique(mask[mask > 0])

                cell_group['matrix/yx'].resize((len(id_list),len(self.hybe_list),3,3))
                cell_group['matrix/zx'].resize((len(id_list),len(self.hybe_list),3,3))
                cell_group['cellbarcodes'].resize((len(id_list),))

                for i in range(len(id_list)):
                    cell_group['cellbarcodes'][i] = 'NA'
                    for j in range(len(self.hybe_list)):
                        cell_group['matrix/yx'][i,j] = np.eye(3)
                        cell_group['matrix/zx'][i,j] = np.eye(3)

            self.widgets['loading_label'].value = f"✅ Mask saved. {time.time()-start:.2f} seconds."
            fig = self.figure
            
            fig.savefig(os.path.normpath(os.path.join(h5_save_path,f'FOV{fov:02d}','CellSegmented.png')),)

            if fov == max(self.fov_list):
                self.widgets['loading_label'].value = "✅ All done."
            else:
                next_fov = fov + 1
                self.widgets['fov_list'].value = next_fov
                self.widgets['bad_indices'].value = ''
                self.run_segmentation()

class CellbarcodeWidget(object):
    """Base class for creating widgets in Jupyter Notebooks.
    """
    def __init__(self, fov_list, hybe_list, channel_list, datatype_list, h5_save_path, filename='vlinks.h5'):
        self.fov_list = fov_list
        self.hybe_list = hybe_list
        self.channel_list = channel_list
        self.datatype_list = datatype_list
        self.h5_save_path = h5_save_path
        self.filename = filename
        self.create_widgets()
        self.hex2rgba = lambda hex: tuple(int(hex[i:i+2], 16)/255 for i in (1, 3, 5)) + (1,)

    def create_widgets(self):
        self.cellbarcodes = np.array(self.hybe_list)[np.isin(self.datatype_list, 'B')]
        self.output = widgets.Output()
        self.widgets = {}
        self.widgets['if_fov_barcode'] = widgets.Dropdown(value='Barcode',options=['Barcode','FOV'], description='From:',)
        self.widgets['fov_list'] = widgets.BoundedIntText(value=min(self.fov_list), min=min(self.fov_list), max=max(self.fov_list)+1, step=1, description='FOV:',)
        self.widgets['channel_list'] = widgets.Dropdown(value=self.channel_list[0], options=self.channel_list, description='Channel:',)
        self.widgets['text_fov_list'] = widgets.Text(value='', description='FOV elements: ')
        self.widgets['current_barcode'] = widgets.Dropdown(value=self.cellbarcodes[0], options=self.cellbarcodes, description='Current Barcode:',)
        self.widgets['min_slider'] = widgets.IntText(value=1000, min=0, max=65535, step=1, description='Min:',)
        self.widgets['max_slider'] = widgets.IntText(value=40000, min=0, max=65535, step=1, description='Max:',)
        self.widgets['rescale_button'] = widgets.Button(description='Rescale', button_style='success',)
        self.widgets['cellbarcodes_list'] = widgets.SelectMultiple(value=self.cellbarcodes, options=self.cellbarcodes, description='cellbarcodes:',)
        self.widgets['bad_indices'] = widgets.Text(value='', description='Bad Indices:',)
        self.widgets['initialize_images_button'] = widgets.Button(description='Initialize Images', button_style='success',)
        self.widgets['determine_cellbarcodes_button'] = widgets.Button(description='Determine cellbarcodes', button_style='success',)
        self.widgets['save_current_cellbarcode_button'] = widgets.Button(description='Save cellbarcodes', button_style='success',)
        self.widgets['loading_label'] = widgets.Label(value='Press "Initialize Images" value')
        self.ui = widgets.VBox([self.widgets['if_fov_barcode'], self.widgets['fov_list'], 
                                widgets.HBox([self.widgets['current_barcode'], self.widgets['text_fov_list']]),
                                widgets.HBox([self.widgets['min_slider'], self.widgets['max_slider'], self.widgets['rescale_button']]),
                                self.widgets['bad_indices'], self.widgets['cellbarcodes_list'],
                                widgets.HBox([self.widgets['initialize_images_button'], self.widgets['determine_cellbarcodes_button'], self.widgets['save_current_cellbarcode_button']]),
                                self.widgets['loading_label'],
                                widgets.HTML(value=self.display_color_legend().data),self.output])
        
        self.figure = None
        self.axes = None
        self.initialize_images()
        self.widgets['rescale_button'].on_click(lambda b: self.rescale_and_update(self.widgets['min_slider'].value, 
                                                                                  self.widgets['max_slider'].value))
        self.widgets['initialize_images_button'].on_click(lambda b: self.initialize_images())
        self.widgets['determine_cellbarcodes_button'].on_click(lambda b: self.determine_cellbarcode())
        self.widgets['current_barcode'].observe(lambda change: self.on_dropdown_change(change), names='value')
        self.widgets['save_current_cellbarcode_button'].on_click(lambda b: self.save_current_cellbarcode())

    def initialize_images(self,):
        self.current_image_parameters = {'raw_images':[], 'images': [], 'bounds':[]}

    def show(self):
        display(self.ui)

    def rescale_and_update(self,lb,ub):
        with self.output:
            clear_output(wait=True)
            self.widgets['loading_label'].value = "🔄 Rescaling image... please wait."
            if lb > ub:
                self.widgets['input_min_slider'].value = ub
            current_barcode = self.widgets['input_current_barcode'].value
            barcode_index = self.cellbarcodes.tolist().index(current_barcode)
            self.current_image_parameters['bounds'][barcode_index] = [lb, ub]
            image_rescaled = preprocess.normalize_to_uint8(self.current_image_parameters['raw_images'][barcode_index], lb, ub)
            self.current_image_parameters['images'][barcode_index] = image_rescaled
            self.update_image()

    def initialize_images(self,):
        with self.output:
            clear_output(wait=True)
            self.widgets['loading_label'].value = "🔄 Initializing images... please wait."

            try:
                self.initialize_images()
                for barcode in self.cellbarcodes:
                    lb,ub = self.current_image_parameters['bounds'][self.cellbarcodes.tolist().index(barcode)]
                    with h5py.File(os.path.join(self.h5_save_path, self.filename), 'r') as f:
                        self.current_image_parameters['raw_images'].append(f[f'/FOV{self.widgets["fov_list"].value:02d}/mip/{barcode}/ch{self.widgets["channel_list"].value}'][:])
                        self.current_image_parameters['images'].append(preprocess.normalize_to_uint8(self.current_image_parameters['raw_images'][-1],lb,ub))
                if self.figure is None:
                    fig,ax = plt.subplots(1,1,figsize=(20,20))
                    divider = make_axes_locatable(ax)
                    caxes = []

                    for i in range(len(self.cellbarcodes)):
                        if i == 0: caxes.append(divider.append_axes('right', size='5%', pad=0.15))
                        else: caxes.append(divider.append_axes('right', size='5%', pad=0.75))
                        cmap = mcolors.LinearSegmentedColormap.from_list(f'{i}',[(0,0,0,1),self.hex2rgba(colors[i])],256,gamma=1.5)
                        cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0,255)), cax=caxes[i], orientation='vertical')
                    rax = divider.append_axes('right', size='100%', pad=1.5)
                    ax.axis('off')
                    rax.axis('off')
                    ax.set_title('Composite Image', fontdict=fontdict_label)
                    self.figure = fig
                    self.axes = [ax,caxes,rax]
                    plt.close(fig)
                else:
                    ax,caxes,rax = self.axes
                    ax.cla()
                    rax.cla()
                self.update_image()
            finally:
                self.widgets['loading_label'].value = "✅ Images initialized."

    def update_image(self):
        with self.output:
            clear_output(wait=True)
            self.widgets['loading_label'].value = "🔄 Updating image... please wait."
            try:
                height, width = self.current_image_parameters['images'][-1].shape
                composites = np.zeros((height, width, 4, len(self.current_image_parameters['images'])), dtype=np.uint8)
                for i in range(len(self.current_image_parameters['images'])):
                    cmap = mcolors.LinearSegmentedColormap.from_list(f'{i}',[(0,0,0,1),self.hex2rgba(colors[i])],256,gamma=1.5)
                    composites[...,i] = cmap(self.current_image_parameters['images'][i]) * 255
                ax,caxes,rax = self.axes
                ax.cla()
                ax.imshow(composites.max(-1))
                for i in range(len(self.current_image_parameters['bounds'])):
                    lb,ub = self.current_image_parameters['bounds'][i]
                    caxes[i].set_yticks([0,255],[f'{lb}', f'{ub}'])
                ax.axis('off')
                rax.axis('off')
            finally:
                self.widgets['loading_label'].value = "✅ Image updated."
                display(self.current_image_parameters['figure'])

    def determine_cellbarcode(self,):
        with self.output:
            clear_output(wait=True)
            self.widgets['loading_label'].value = "🔄 Determining cellbarcodes... please wait."

            try:
                fov = self.widgets['fov_list'].value
                cellbarcodes = self.widgets['cellbarcodes_list'].value
                bad_indices = [int(i) for i in self.widgets['bad_indices'].value.split() if i.strip()]
                with h5py.File(os.path.normpath(os.path.normpath(os.path.join(self.h5_save_path, self.filename))), 'r+') as f:
                    mask = f[f'/cells/mask'][:]
                    id_list = np.unique(mask)[1:]
                    good_id = np.isin(id_list, bad_indices, invert=True)
                    
                    ct_list = []
                    for j,id in enumerate(id_list):
                        barcode_channel_values = [np.median(self.current_image_parameters['images'][i][mask == id])
                                                   for i in range(len(self.current_image_parameters['images']))]
                        if max(barcode_channel_values) == 0:
                            ct = 'NA'
                        else:
                            ct = cellbarcodes[np.argmax(barcode_channel_values)]                
                        ct_list.append(ct)
                    ct_array = np.array(ct_list, dtype=h5py.string_dtype(encoding='utf-8'))
                    height, width = self.current_image_parameters['images'][0].shape

                    composite = np.zeros((height, width, 4, len(self.current_image_parameters['images'])), dtype=np.uint8)
                    for i in range(len(self.current_image_parameters['images'])):
                        cmap = mcolors.LinearSegmentedColormap.from_list(f'{i}',[(0,0,0,1),self.hex2rgba(colors[i])],256,gamma=1.5)
                        which_id = id_list[(ct_array == cellbarcodes[i]) & good_id]
                        composite[...,i] = cmap(np.isin(mask, which_id).astype(int) * 255) * 255

                fig = self.figure
                ax,caxes,rax = self.axes
                rax.cla()
                rax.imshow(composite.max(-1))
                for id in id_list:
                    if id in bad_indices:
                        continue
                    y,x = np.where(mask == id)
                    rax.text(x.mean(),y.mean(), str(id), color='white', fontsize=14, ha='center', va='center')
                rax.axis('off')
            finally:
                self.widgets['loading_label'].value = "✅ cellbarcodes determined."
                display(self.figure)

    def save_current_cellbarcode(self):
        with self.output:
            clear_output(wait=True)
            self.widgets['loading_label'].value = "🔄 Saving cellbarcodes... please wait."

            try:
                fov = self.widgets['fov_list'].value
                cellbarcodes = self.widgets['cellbarcodes_list'].value
                bad_indices = [int(i) for i in self.widgets['bad_indices'].value.split() if i.strip()]
                with h5py.File(os.path.normpath(os.path.normpath(os.path.join(self.h5_save_path,self.filename))), 'r+') as f:
                    mask = f[f'/cells/mask'][:]
                    mask[np.isin(mask, bad_indices)] = 0
                    f[f'/cells/mask'][:] = mask
                    id_list = np.unique(mask)[1:]
                    hybe_list = f.attrs['hybe_list']

                    f[f'/cells/matrix/yx'].resize((len(id_list),len(hybe_list),3,3))
                    f[f'/cells/matrix/xz'].resize((len(id_list),len(hybe_list),3,3))

                    for j in range(len(hybe_list)):
                        for i in range(len(id_list)):
                            f[f'/cells/matrix/yx'][i,j] = np.eye(3)
                            f[f'/cells/matrix/xz'][i,j] = np.eye(3)

                    ct_list = []
                    for j,id in enumerate(id_list):
                        barcode_channel_values = [np.median(self.current_image_parameters['images'][i][mask == id]) for i in range(len(self.current_image_parameters['images']))]
                        if max(barcode_channel_values) == 0:
                            ct = 'NA'
                        else:
                            ct = cellbarcodes[np.argmax(barcode_channel_values)]                
                        ct_list.append(ct)
                    f[f'/cells/cellbarcodes'].resize((len(id_list),))
                    f[f'/cells/cellbarcodes'][:] = ct_list#np.array(ct_list, dtype=h5py.string_dtype(encoding='utf-8'))
                    height, width = self.current_image_parameters['images'][0].shape

                    composite = np.zeros((height, width, 4, len(self.current_image_parameters['images'])), dtype=np.uint8)
                    for i in range(len(self.current_image_parameters['images'])):
                        cmap = mcolors.LinearSegmentedColormap.from_list(f'{i}',[(0,0,0,1),self.hex2rgba(colors[i])],256,gamma=1.5)
                        which_id = id_list[(f[f'/cells/cellbarcodes'][:] == cellbarcodes[i])]
                        composite[...,i] = cmap(np.isin(mask, which_id).astype(int) * 255) * 255

                fig = self.figure
                ax,caxes,rax = self.axes
                rax.cla()
                rax.imshow(composite.max(-1))
                for id in id_list:
                    if id in bad_indices:
                        continue
                    y,x = np.where(mask == id)
                    rax.text(x.mean(),y.mean(), str(id), color='white', fontsize=14, ha='center', va='center')
                rax.axis('off')
                fig.savefig(os.path.normpath(os.path.join(self.h5_save_path, f'H5',f'FOV{fov:02d}',f'cellbarcodes.png')), bbox_inches='tight')
                
            finally:
                self.widgets['loading_label'].value = "✅ cellbarcodes determined."
                display(self.figure)
                if fov < max(self.fov_list):
                    next_fov = fov + 1
                    self.widgets['fov_list'].value = next_fov
                    self.widgets['bad_indices'].value = ''
                    self.widgets['current_barcode'].value = self.cellbarcodes[0]
                    self.initialize_images()
                    self.determine_cellbarcode()
                else:
                    self.widgets['loading_label'].value = "✅ All done."

    def display_color_legend(self):
        legend_html = "<div style='display:flex; flex-wrap:wrap; gap:8px;'>"
        for i, barcode in enumerate(self.cellbarcodes):
            rgba = self.hex2rgba(colors[i])
            rgba_str = f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, 1.0)"
            legend_html += f"""
            <div style='display:flex; align-items:center; gap:4px;'>
                <div style='width:16px; height:16px; background:{rgba_str}; border:1px solid #ccc;'></div>
                <span style='font-family:monospace;'>{barcode}</span>
            </div>
            """
        legend_html += "</div>"
        return HTML(legend_html)

    def on_dropdown_change(self, change):
        current_barcode = change['new']
        barcode_index = self.cellbarcodes.tolist().index(current_barcode)
        self.widgets['input_min_slider'].value = self.current_image_parameters['bounds'][barcode_index][0]
        self.widgets['input_max_slider'].value = self.current_image_parameters['bounds'][barcode_index][1]

