import os
import abc
import copy

import numpy as np
import pyvista as pv
import nibabel as nib

import matplotlib.pyplot as plt
import matplotlib as mpl

import numbers

def get_face_label(labels):
    n_parcels = len(set(labels))
    res = labels[0] if (n_parcels == 1) else -1
    return res

def get_triangle_stats(vertex_stats, triangles, func=np.mean):
    res = np.zeros(len(triangles), dtype=vertex_stats.dtype)
    for face_idx, face_stats in enumerate(vertex_stats[triangles]):
        res[face_idx] = func(face_stats)
    return res


def _get_subsystem(name):
    try:
        return name.split('_')[2]
    except:
        return ''
    
def get_border(brain):
    label_subsystems = [_get_subsystem(p) for p in brain.parcel_names]

    subsystem_to_index = {}
    for subsystem in label_subsystems:
        if not(subsystem in subsystem_to_index):
            subsystem_to_index[subsystem] = len(subsystem_to_index)

    label_to_index = np.array([subsystem_to_index[subsystem] for subsystem in label_subsystems])

    is_border = np.zeros(brain.triangles.shape[0], dtype=bool)
    
    for idx, triangle in enumerate(brain.triangles[:, 1:]):
#         triangle_subsystems = set(label_to_index[brain.vertex_labels[triangle]])
        triangle_subsystems = set(brain.vertex_labels[triangle])
        significant_overlap = brain.significant_parcels & triangle_subsystems

        is_border[idx] = (len(significant_overlap) > 0) & (len(triangle_subsystems) > 1)
        
    return is_border

def _is_scalar(data):
    return (len(data) > 0) and isinstance(next(iter(data.values())), numbers.Number)


class BrainSurface:
    def __init__(self, subject_path, hemis=None, surface='pial',
                 parcellation='Schaefer2018_100Parcels_17Networks'):
        self.subject_path = subject_path
        self.parcellation = parcellation
        self.surface = surface
        
        if hemis is None:
            self.hemis = ['lh', 'rh']
        else:
            self.hemis = list(hemis)
            
        self._load_hemis()
        self._load_annotations()
        self.data = dict()
        self.plotter = None

    def _load_hemis(self):
        self.surfaces = dict()
        self.triangles = list()
        
        index_offset = 0
        
        coords = list()
        curviture = list()
        
        for hemi in self.hemis:
            surf_path = os.path.join(self.subject_path, 'surf', f'{hemi}.{self.surface}')
            curv_path = os.path.join(self.subject_path, 'surf', f'{hemi}.curv')
            
            surf_coords, surf_triangles = nib.freesurfer.io.read_geometry(surf_path)
            
            surf_triangles += index_offset
            surf_triangles = np.hstack([np.full_like(surf_triangles[:,:1], 3), surf_triangles])
            
            surf_curv = nib.freesurfer.read_morph_data(curv_path)
            
            coords.append(surf_coords)
            curviture.append(surf_curv)
            self.triangles.append(surf_triangles)
            
            index_offset += surf_coords.shape[0]
            
        self.triangles = np.vstack(self.triangles)
        self.vertex_curviture = np.hstack(curviture)
        
        self.face_curviture = get_triangle_stats(self.vertex_curviture, self.triangles[:, 1:])
        
        coords = np.vstack(coords)
        self.brain_mesh = pv.PolyData(coords, self.triangles)

    def _load_annotations(self):
        self.annotations = dict()
        self.parcel_names = list()
        vertex_labels = list()
        
        for hemi in self.hemis:
            annot_path = os.path.join(self.subject_path, 'label', f'{hemi}.{self.parcellation}.annot')
            
            labels_orig, _, annot_ch_names = nib.freesurfer.io.read_annot(annot_path)   
            labels_orig += len(self.parcel_names)
            
            annot_ch_names = [n.decode() for n in annot_ch_names]
            
            vertex_labels += labels_orig.tolist()
            self.parcel_names += annot_ch_names
            
        self.vertex_labels = np.array(vertex_labels)
        self.face_labels = get_triangle_stats(self.vertex_labels, self.triangles[:, 1:], func=get_face_label)
    
    def set_data(self, data, significant_parcels=None):
        missing_keys = [key for key in data.keys() if not(key in self.parcel_names)]
        self.data = {key:value for (key,value) in data.items() if key in self.parcel_names}
        # self.data = copy.deepcopy(data)
        
        if (significant_parcels is None):
            self.significant_parcels = set()  
        else:
            significant_filtered = [p for p in significant_parcels if (p in self.parcel_names)]
           
            self.significant_parcels = set([self.parcel_names.index(p) for p in significant_filtered])

    def plot(self, cmap='viridis', camera_position=None, show=True, zoom=1.0, vmin=None, vmax=None, 
             draw_borders=False, border_color=(0,1,0,1), lightning='light_kit', norm=None, silhouette=False, **kwargs):
        is_scalar = _is_scalar(self.data)
        
        cmap_obj = plt.get_cmap(cmap) if (type(cmap) is str) else cmap
        
        # if len(self.data) == 0:
        #     return
    
        if norm is None:
            norm = 'linear'

        cmap_obj = plt.get_cmap(cmap) if (type(cmap) is str) else cmap
        
        if is_scalar:
            min_data_value = np.nanmin(list(self.data.values())) if vmin is None else vmin
            max_data_value = np.nanmax(list(self.data.values())) if vmax is None else vmax

            if norm == 'none':
                norm_obj = mpl.colors.NoNorm(min_data_value, max_data_value)
            elif norm == 'log':
                norm_obj = mpl.colors.LogNorm(min_data_value, max_data_value)
            elif norm == 'discrete':
                bounds = np.linspace(min_data_value, max_data_value, 10)
                norm_obj = mpl.colors.BoundaryNorm(bounds, cmap_obj.N)
            elif norm == 'log_discrete':
                bounds = np.geomspace(min_data_value, max_data_value, 10)
                norm_obj = mpl.colors.BoundaryNorm(bounds, cmap_obj.N)
            else:
                norm_obj = mpl.colors.Normalize(min_data_value, max_data_value)
            norm_obj._norm_type = norm
        
        colormap = list([(0.5,0.5,0.5, 1.0)])
        
        scalars = np.full(len(self.face_labels), np.nan)
        border_mask = (self.face_labels == 0) | (self.face_labels == -1)
        scalars[border_mask] = 0
        
        for item_index, (key, value) in enumerate(self.data.items(), start=1):
            if not(key in self.parcel_names):
                print(f'wut {key}')
                color = (0.5, 0.5, 0.5, 1.0)
            else:
                label = self.parcel_names.index(key)
                
                alpha = 1.0 if (label in self.significant_parcels) else 0.5

                mask = (self.face_labels == label)
                
                scalars[mask] = item_index

                if is_scalar:
                    color = norm_obj(value)
                else:
                    color = value
                    if type(value) is tuple: 
                        color = value if len(value) == 4 else tuple(value) + (alpha,)
                    
            colormap.append(color)
        
        self.scalars = scalars.copy()
                
        # curviture_bin = (self.face_curviture > 0)
        # no_data_mask = np.isnan(scalars)
        
        # scalars[no_data_mask] = curviture_bin[no_data_mask] + len(colormap)
        no_data_mask = np.isnan(scalars)
        
        scalars[no_data_mask] = len(colormap)
        
        if no_data_mask.sum() > 0:
            colormap.append((0.8,0.8,0.8, 1.0))
            
        if draw_borders:
            # border_mask = get_border(self)
            
            if border_mask.sum() > 0:
                scalars[border_mask] = len(colormap)
                colormap.append(border_color)
            
        colormap = mpl.colors.ListedColormap(colormap)
            
        self.plotter = pv.Plotter(lighting=lightning)
        self.plotter.add_mesh(self.brain_mesh, scalars=scalars, cmap=colormap, categories=not(is_scalar), show_scalar_bar=False, silhouette=silhouette)
        
        if not(camera_position is None):
            self.plotter.camera_position = camera_position
            
        self.plotter.set_background('white')
        self.plotter.camera.Zoom(zoom)
        
        if show:
            self.plotter.show(**kwargs)
        
    def show_on_axis(self, ax):
        img = self.plotter.screenshot(return_img=True)
        ax.imshow(img)
        ax.set_axis_off()
        
    def save_to_image(self, img_path, **kwargs):
        fig, ax = plt.subplots(**kwargs)
        
        img = self.plotter.screenshot(return_img=True)
        ax.imshow(img)
        
        ax.set_axis_off()
        fig.savefig(img_path, **kwargs)
        
        plt.close(fig)