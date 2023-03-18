import os

import numpy as np

from cortex.freesurfer import _remove_disconnected_polys, _move_disconnect_points_to_zero

import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import matplotlib as mpl

import nibabel as nib

from copy import deepcopy

from datashader.bundling import hammer_bundle

from collections import Counter, defaultdict

import itertools

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
        comment = fp.readline()
        fp.readline()
        verts, faces = struct.unpack('>2I', fp.read(8))
        pts = np.frombuffer(fp.read(4*3*verts), dtype='f4').byteswap()
        polys = np.frombuffer(fp.read(4*3*faces), dtype='i4').byteswap()

        return pts.reshape(-1, 3), polys.reshape(-1, 3)

def get_surf_data(surf_fname, patch_fname):
    from cortex.freesurfer import get_paths, parse_patch

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

# def get_triangle_labels(labels, triangles, edge_value=-1):
#     res = np.zeros(shape=len(triangles), dtype=labels.dtype)
    
#     for idx, coords in enumerate(triangles):
#         triangle_labels = labels[coords]
#         if len(set(triangle_labels)) == 1:
#             res[idx] = triangle_labels[0]
#         else:
#             res[idx] = edge_value
    
#     return res

def get_triangle_labels(labels, triangles, cortical_subsystems, edge_value=-1, depth=1):
    res = np.zeros(shape=len(triangles), dtype=object)
    edge_faces = np.zeros(shape=len(triangles), dtype=bool)
    face_graph = build_face_graph(triangles)
    
    for idx, coords in enumerate(triangles):
        triangle_labels = labels[coords]
        labels_set = set(triangle_labels)
        
        if len(labels_set) == 1:
            res[idx] = triangle_labels[0]
        elif len(labels_set) == 3: #intersection of 3 areas
            res[idx] = edge_value
        else: # an edge
            x, y = labels_set
            top_label = Counter(triangle_labels).most_common()[0][0]
            res[idx] = top_label
            
            if cortical_subsystems[x] != cortical_subsystems[y]:
                edge_faces[idx] = True
    
    to_visit = np.where(edge_faces)[0]
    to_visit_updated = list()
    
    for _ in range(depth):
        for face_idx in to_visit:
            for adj_idx in face_graph[face_idx]:
                if edge_faces[adj_idx] == True:
                    continue
                
                edge_faces[adj_idx] = True
                to_visit_updated.append(adj_idx)
        
        to_visit = to_visit_updated.copy()
        to_visit_updated = list()
    
    
    res[edge_faces] = np.take(cortical_subsystems, res[edge_faces].astype(int))
    
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

def get_subsystem(name):
    if len(name) == 0:
        return 'Undefined'
    
    return name.split('_')[2]


def build_vertex_to_face(faces):    
    res = defaultdict(list)
    
    for face_idx, coords in enumerate(faces):
        for c in coords:
            res[c].append(face_idx)
    
    return res

def build_face_graph(faces):
    vertex_to_faces = build_vertex_to_face(faces)
    
    res = defaultdict(set)
    
    for node_idx, adjacent in vertex_to_faces.items():
        for i, j in itertools.combinations(adjacent, 2):
            res[i].add(j)
            res[j].add(i)

    return res


class FlatSurface:
    def __init__(self, subject_path, substem_to_color, parcellation='Schaefer2018_100Parcels_17Networks'):
        self.subject_path = subject_path
        self.parcellation = parcellation
        self.subsystem_to_color = substem_to_color

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
            annot_ch_subsystem = [get_subsystem(n) for n in annot_ch_names]
            
            labels_faces = get_triangle_labels(labels_orig, self.surfaces[hemi]['faces'], annot_ch_subsystem, depth=3)
            
            parcel_coords = np.zeros((len(set(labels_orig)), 2))
            for label in set(labels_orig):
                label_indices = (labels_orig == label)
                parcel_coords[label] = np.median(self.surfaces[hemi]['coords'][label_indices], axis=0)
#             parcel_coords = parcel_coords[1:]
#             parcel_coords = npg.aggregate(labels_orig, self.surfaces[hemi]['coords'], func='median', axis=0)[1:]
            
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
    
    def plot(self, data_cmap='viridis', connectome_cmap='jet', connectome_threshold=0.0, ax=None, 
             use_norm=True, draw_colorbar=False, alpha=1, min_value=None, max_value=None):
        if (ax is None):
            fig, ax_plot = plt.subplots(figsize=(4*3, 3*3))
        else:
            ax_plot = ax
        
        self._plot_data(cmap=data_cmap, ax=ax_plot, use_norm=use_norm, draw_colorbar=draw_colorbar, alpha=alpha, min_value=min_value, max_value=max_value)
        
        if not(self.connectome is None):
            self._plot_connectome(cmap=connectome_cmap, ax=ax_plot, threshold=connectome_threshold)
        
        ax_plot.set_axis_off()
    
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
        
    def _plot_data(self, cmap, ax, draw_colorbar=False, use_norm=True, alpha=1.0, min_value=None, max_value=None):
        fig = ax.get_figure()
        cmap_obj = plt.get_cmap(cmap) if (type(cmap) is str) else cmap
        
        if len(self.data) > 0:
            min_data_value = np.nanmin(list(self.data.values())) if min_value is None else min_value
            max_data_value = np.nanmax(list(self.data.values())) if max_value is None else max_value
            norm = mpl.colors.Normalize(min_data_value, max_data_value)
                
        for hemi in ['lh', 'rh']:
            colors_to_use = [(0, 0, 0)]
            
            face_data = np.zeros_like(self.annotations[hemi]['face_labels'], dtype=float)
            face_mask = (self.annotations[hemi]['face_labels'] != -1)
            face_data[~face_mask] = len(colors_to_use)

            for parcel_idx, parcel_name in enumerate(self.annotations[hemi]['parcel_names']):
                parcel_value = self.data.get(parcel_name)

                if parcel_value is None:
                    continue
                
                if use_norm:
                    parcel_value = norm(parcel_value)

                colors_to_use.append(cmap_obj(parcel_value))
                
                face_indices = (self.annotations[hemi]['face_labels'] == parcel_idx)
                face_data[face_indices] = len(colors_to_use)
                face_mask[face_indices] = False                
            

            for subsystem_name, subsystem_color in self.subsystem_to_color.items():
                colors_to_use.append(subsystem_color)

                face_indices = (self.annotations[hemi]['face_labels'] == subsystem_name)
                face_data[face_indices] = len(colors_to_use)
                face_mask[face_indices] = False       

            plot_cmap = mpl.colors.ListedColormap(colors_to_use)

            ax.tripcolor(*self.surfaces[hemi]['coords'].T, self.surfaces[hemi]['faces'], 
                 facecolors=face_data, mask=face_mask, cmap=plot_cmap, alpha=alpha)  
        
        if len(self.data) > 0 and draw_colorbar:
            self._plot_colorbar(ax, min_data_value, max_data_value, cmap)
            
    def _plot_colorbar(self, ax, vmin, vmax, cmap):
        fig = ax.get_figure()
        norm = mpl.colors.Normalize(vmin, vmax)
        cmap_obj = plt.get_cmap(cmap) if (type(cmap) is str) else cmap
        
        pos =  ax.get_position()
        cbar_ax = fig.add_axes([pos.x0, pos.y0 + pos.height*3/4, 0.01, pos.height*1/4])
        mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap_obj, orientation='vertical', norm=norm)