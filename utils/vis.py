import os

import numpy as np

import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import nibabel as nib

from copy import deepcopy

import struct

def parse_patch(filename):
    """
    """
    with open(filename, 'rb') as fp:
        header, = struct.unpack('>i', fp.read(4))
        nverts, = struct.unpack('>i', fp.read(4))
        data = np.fromstring(fp.read(), dtype=[('vert', '>i4'), ('x', '>f4'),
                                               ('y', '>f4'), ('z', '>f4')])
        assert len(data) == nverts
        return data

def _move_disconnect_points_to_zero(pts, polys):
    """Change coordinates of points not in polygons to zero.
    
    This cleaning step is useful after _remove_disconnected_polys, to
    avoid using this points in boundaries computations (through pts.max(axis=0)
    here and there).
    """
    mask = np.zeros(len(pts), dtype=bool)
    mask[np.unique(polys)] = True
    pts[~mask] = 0
    return pts

def _remove_disconnected_polys(polys):
    """Remove polygons that are not in the main connected component.
    
    This function creates a sparse graph based on edges in the input.
    Then it computes the connected components, and returns only the polygons
    that are in the largest component.
    
    This filtering is useful to remove disconnected vertices resulting from a
    poor surface cut.
    """
    n_points = np.max(polys) + 1
    import scipy.sparse as sp

    # create the sparse graph
    row = np.concatenate([
        polys[:, 0], polys[:, 1], polys[:, 0],
        polys[:, 2], polys[:, 1], polys[:, 2]
    ])
    col = np.concatenate([
        polys[:, 1], polys[:, 0], polys[:, 2],
        polys[:, 0], polys[:, 2], polys[:, 1]
    ])
    data = np.ones(len(col), dtype=bool)
    graph = sp.coo_matrix((data, (row, col)), shape=(n_points, n_points),
                          dtype=bool)
    
    # compute connected components
    n_components, labels = sp.csgraph.connected_components(graph)
    unique_labels, counts = np.unique(labels, return_counts=True)
    non_trivial_components = unique_labels[np.where(counts > 1)[0]]
    main_component = unique_labels[np.argmax(counts)]
    extra_components = non_trivial_components[non_trivial_components != main_component]

    # filter all components not in the largest component
    disconnected_pts = np.where(np.isin(labels, extra_components))[0]
    disconnected_polys_mask = np.isin(polys[:, 0], disconnected_pts)
    return polys[~disconnected_polys_mask]



def get_rotation_matrix(degrees):
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    
    return R

def rotate_coords(coords, degree):
    return np.dot(coords, get_rotation_matrix(degree))

def parse_surf(filename):
    import struct
    """
    """
    with open(filename, 'rb') as fp:
        #skip magic
        fp.seek(3)
        _ = fp.readline()
        fp.readline()
        verts, faces = struct.unpack('>2I', fp.read(8))
        pts = np.frombuffer(fp.read(4*3*verts), dtype='f4').byteswap()
        polys = np.frombuffer(fp.read(4*3*faces), dtype='i4').byteswap()

        return pts.reshape(-1, 3), polys.reshape(-1, 3)

def get_surf_data(surf_fname, patch_fname):
    pts, polys = parse_surf(surf_fname)

    patch = parse_patch(patch_fname)
    verts = patch[patch['vert'] > 0]['vert'] - 1
    edges = -patch[patch['vert'] < 0]['vert'] - 1

    idx = np.zeros((len(pts),), dtype=bool)
    idx[verts] = True
    idx[edges] = True
    valid = idx[polys.ravel()].reshape(-1, 3).all(1)
    polys = polys[valid]
    idx = np.zeros((len(pts),))
    idx[verts] = 1
    idx[edges] = -1

    for i, x in enumerate(['x', 'y', 'z']):
        pts[verts, i] = patch[patch['vert'] > 0][x]
        pts[edges, i] = patch[patch['vert'] < 0][x]
        
    return pts, polys, idx


def get_subsystem(name):
    if len(name) > 0:
        return name.split('_')[2]
    
    return 'Undefined'


def get_triangle_labels(labels, triangles, cortical_subsystems, parcel_edge_value=-1, subsystem_edge_value=-2):
    res = np.zeros(shape=len(triangles), dtype=labels.dtype)
    
    for idx, coords in enumerate(triangles):
        triangle_labels = labels[coords]
        labels_set = set(triangle_labels)

        if len(labels_set) == 1:
            res[idx] = triangle_labels[0]
        elif len(labels_set) == 2:
            x, y = labels_set
            
            res[idx] = parcel_edge_value if cortical_subsystems[x] == cortical_subsystems[y] else subsystem_edge_value
            # res[idx] = parcel_edge_value
        else:
            res[idx] = parcel_edge_value
    
    return res


def create_hb_df(surf, connectome, threshold):
    parc_coords = np.concatenate([surf.annotations['lh']['coords'][1:], surf.annotations['rh']['coords'][1:]])
    
    coords_df = pd.DataFrame(parc_coords, columns=['x', 'y'])
    coords_df['name'] = surf.annotations['lh']['parcel_names'][1:] + surf.annotations['rh']['parcel_names'][1:]
    
    edges_df = {'source': [], 'target': [], 'weight': []}
    for i, j in zip(*np.triu_indices_from(connectome, 1)):
        value = connectome[i,j]
        
        if value >= threshold:
            edges_df['source'].append(i)
            edges_df['target'].append(j)
            edges_df['weight'].append(value)
        
    edges_df = pd.DataFrame(edges_df)
    
    return coords_df, edges_df


class FlatSurface:
    def __init__(self, subject_path, parcellation='Schaefer2018_100Parcels_17Networks'):
        self.subject_path = subject_path
        self.parcellation = parcellation
        
        self._load_hemis()
        self._load_annotations()
        
    def _load_hemis(self):
        self.surfaces = dict()
        
        for hemi in ['lh', 'rh']:
            wm_path = os.path.join(self.subject_path, 'surf', f'{hemi}.smoothwm')
            patch_path = os.path.join(self.subject_path, 'surf', f'{hemi}.cortex.patch.flat')

            hemi_coords, hemi_faces, _ = get_surf_data(wm_path, patch_path)
            
            hemi_coords = hemi_coords[:, [1, 0, 2]]
            # Flip Y axis upside down
            hemi_coords[:, 1] = -hemi_coords[:, 1]
            
            hemi_faces = _remove_disconnected_polys(hemi_faces)
            hemi_coords = _move_disconnect_points_to_zero(hemi_coords, hemi_faces)[:, :2]
            
            degree = 90 if hemi == 'lh'  else -90
            hemi_coords = rotate_coords(hemi_coords, degree)

            if hemi == 'rh':
                hemi_coords[:, 0] += self.surfaces['lh']['coords'][:,0].max() - hemi_coords[:,0].min() + 10
            
            self.surfaces[hemi] = {'coords': hemi_coords, 'faces': hemi_faces}
            
    def _load_annotations(self):
        self.annotations = dict()
        self.parcel_coords = dict()
        
        self.n_parcels = 0
        
        for hemi in ['lh', 'rh']:
            annot_path = os.path.join(self.subject_path, 'label', f'{hemi}.{self.parcellation}_order.annot')
            
            labels_orig, _, annot_ch_names = nib.freesurfer.io.read_annot(annot_path)   
            annot_ch_names = [n.decode() for n in annot_ch_names]
            annot_ch_subsystems = [get_subsystem(n) for n in annot_ch_names]
            
            labels_faces = get_triangle_labels(labels_orig, self.surfaces[hemi]['faces'], annot_ch_subsystems)
            
            parcel_coords = np.zeros((len(set(labels_orig)), 2))
            for label in set(labels_orig):
                label_indices = (labels_orig == label)
                parcel_coords[label] = np.median(self.surfaces[hemi]['coords'][label_indices], axis=0)
            
            self.annotations[hemi] = {'vertex_labels': labels_orig, 'face_labels': labels_faces,
                                      'parcel_names': annot_ch_names, 'coords': parcel_coords}
            
            self.n_parcels += len(annot_ch_names)
            
        self.connectome = None
        self.data = dict()
                        
    def add_connectome(self, connectome):
        if connectome.shape[0] != self.n_parcels - 2: # -2 unknown
            raise RuntimeError(f'Amount of connectome parcels ({connectome.shape[0]}) is not equal to data ({self.n_parcels})')
        
        if connectome.shape[0] != connectome.shape[1]:
            raise RuntimeError(f'Connectome is not square! Shape: {connectome.shape}')
        
        self.connectome = connectome.copy()
        
    def add_data(self, data):
        # data should be a dict with mapping parcel_name -> value
        self.data = deepcopy(data)
    
    def plot(self, ax=None, **kwargs):
    
        if (ax is None):
            _, ax_plot = plt.subplots(figsize=(4*3, 3*3))
        else:
            ax_plot = ax
        
        self._plot_data(ax=ax_plot, **kwargs)
        
        # if not(self.connectome is None):
        #     self._plot_connectome(cmap=connectome_cmap, ax=ax_plot, threshold=connectome_threshold)
        
        ax_plot.set_axis_off()

        return ax_plot
    
    def _plot_connectome(self, cmap, ax, threshold):
        fig = ax.get_figure()
        cmap_obj = plt.get_cmap(cmap) if (type(cmap) is str) else cmap

        ds_nodes, ds_edges = create_hb_df(self, self.connectome, threshold=threshold)
        hb = hammer_bundle(ds_nodes, ds_edges, initial_bandwidth=0.125, decay=0.125)
        
        segments = np.array_split(hb.values, np.where(np.isnan(hb.values[:,0]))[0])
        
        norm = mpl.colors.Normalize(ds_edges['weight'].min(), ds_edges['weight'].max())
        
        for s in segments[:-1]:
            width = s[1,2]
            ax.plot(s[:,0], s[:,1], color=cmap_obj(norm(width)), lw=width*5)
            
        pos =  ax.get_position()

        cbar_ax = fig.add_axes([pos.x1, pos.y0 + pos.height*3/4, 0.01, pos.height*1/4])
        mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap_obj, orientation='vertical', norm=norm)
        
    def _plot_data(self, ax, cmap='viridis', draw_colorbar=False, norm=None, min_value=None, max_value=None, round_ticks=False, n_round=2):

        if len(self.data) == 0:
            return 

        if norm is None:
            norm = 'linear'

        cmap_obj = plt.get_cmap(cmap) if (type(cmap) is str) else cmap
        min_data_value = np.nanmin(list(self.data.values())) if min_value is None else min_value
        max_data_value = np.nanmax(list(self.data.values())) if max_value is None else max_value

        if norm is 'none':
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
                
        for hemi in ['lh', 'rh']:
            colors_to_use = [(0.0, 0.0, 0.0, 1.0), (0.0,0.0,0.0, 1.0) ]
            
            face_data = np.zeros_like(self.annotations[hemi]['face_labels'], dtype=float)
            face_mask = (self.annotations[hemi]['face_labels'] >= 0)

            parcel_edges = (self.annotations[hemi]['face_labels'] == -1)
            subsystem_edges = (self.annotations[hemi]['face_labels'] == -2)
            
            face_data[parcel_edges] = 1
            face_data[subsystem_edges] = 2

            for parcel_idx, parcel_name in enumerate(self.annotations[hemi]['parcel_names']):
                parcel_value = self.data.get(parcel_name)

                if parcel_value is None:
                    continue
                
                if np.isscalar(parcel_value):
                    parcel_value = norm_obj(parcel_value)
                    parcel_color = cmap_obj(parcel_value)
                else:
                    parcel_color = parcel_value

                colors_to_use.append(parcel_color)
                
                face_indices = (self.annotations[hemi]['face_labels'] == parcel_idx)
                face_data[face_indices] = len(colors_to_use)
                face_mask[face_indices] = False                
            
            plot_cmap = mpl.colors.ListedColormap(colors_to_use)

            ax.tripcolor(*self.surfaces[hemi]['coords'].T, self.surfaces[hemi]['faces'], 
                 facecolors=face_data, mask=face_mask, cmap=plot_cmap, alpha=None)  
        
        if draw_colorbar:
            self._plot_colorbar(ax, cmap, norm_obj, round_ticks=round_ticks, n_round=n_round)
            
    def _plot_colorbar(self, ax, cmap, norm, round_ticks=False, n_round=2):
        cmap_obj = plt.get_cmap(cmap) if (type(cmap) is str) else cmap

        vmin = norm.vmin
        vmax= norm.vmax

        space_func = np.geomspace if 'log' in norm._norm_type else np.linspace

        if round_ticks:
            ticks = space_func(np.ceil(vmin), np.floor(vmax), 4).round().astype(int)
        else:
            dist = vmax - vmin
            ticks = space_func(vmin + dist * 0.025, vmax - dist*0.025, 4)

            if not(n_round is None):
                ticks = ticks.round(n_round)

        self.cbar_ax = inset_axes(ax, width="5%", height="40%", loc=2) 
        mpl.colorbar.ColorbarBase(self.cbar_ax, cmap=cmap_obj, orientation='vertical', norm=norm, ticks=ticks)
        self.cbar_ax.yaxis.set_ticks_position('left')


