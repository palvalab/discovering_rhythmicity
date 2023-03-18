import numpy as np
import scipy as sp

import nibabel as nib

import pyvista as pv

from sklearn.cluster import KMeans

from joblib import Parallel, delayed, wrap_non_picklable_objects
from joblib.externals.loky import set_loky_pickler

import networkx as nx

import community as community_louvain


def create_adjacency(surf_fname, annot_fname, offset=0, n_parcels=400):
    coords, triangles = nib.freesurfer.io.read_geometry(surf_fname)
    labels, _, _ = nib.freesurfer.io.read_annot(annot_fname)
    
    adjacency = np.zeros((n_parcels, n_parcels), dtype=bool)
    
    for row in labels[triangles]:
        if any(row == 0):
            continue

        a,b,c = row - 1 + offset
        adjacency[a,b] = True
        adjacency[a,c] = True
        adjacency[b,c] = True
        
        adjacency[b,a] = True
        adjacency[c,a] = True
        adjacency[c,b] = True
    
    return adjacency


def find_path(coords, triangles, centroids, indices):
    brain = pv.PolyData(coords, triangles)
    
    res = list()
    
    for i, j in indices:
        i_coord = centroids[i]
        j_coord = centroids[j]

        i_point = brain.find_closest_point(i_coord)
        j_point = brain.find_closest_point(j_coord)

        res.append(brain.geodesic_distance(i_point, j_point))
    
    return res


def compute_centroids(surf_fname, annot_fname, surf_parcels):
    coords, _ = nib.freesurfer.io.read_geometry(surf_fname)
    
    labels, _, _ = nib.freesurfer.io.read_annot(annot_fname)
    labels -= 1
    
    centroids = np.zeros((surf_parcels, 3))
    
    for i in range(surf_parcels):
        centroids[i] = coords[labels == i].mean(axis=0)

    return centroids


def create_brain(surf_fname):
    coords, triangles = nib.freesurfer.io.read_geometry(surf_fname)
    triangles = np.hstack([np.full_like(triangles[:,:1], 3), triangles])
    
    return pv.PolyData(coords, triangles)


def create_distance_matrix(surf_fname, annot_fname, offset=0, total_parcels=400, n_jobs=32):
    surf_parcels = total_parcels // 2
    
    coords, triangles = nib.freesurfer.io.read_geometry(surf_fname)
    triangles = np.hstack([np.full_like(triangles[:,:1], 3), triangles])
    
    labels, _, _ = nib.freesurfer.io.read_annot(annot_fname)
    labels -= 1
    
    centroids = np.zeros((surf_parcels, 3))
    
    for i in range(surf_parcels):
        centroids[i] = coords[labels == i].mean(axis=0)

    dist = np.zeros((total_parcels, total_parcels))
    
    indices = list(zip(*np.triu_indices_from(np.ones((surf_parcels, surf_parcels)), k=1)))
    indices_split = np.array_split(indices, n_jobs)
    
    values = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(find_path)(coords, triangles, centroids, job_indices) 
                                                            for job_indices in indices_split)
    
    values_lst = sum(values, [])
    for (i,j), val in zip(indices, values_lst):
        dist[j+offset,i+offset] = dist[i+offset,j+offset] = val
        
    return dist


def smooth_adjacent_by_distance(arr, dist_matrix, n_neighbours=5):
    res = arr.copy()
    n_samples = n_neighbours + 1 # self + adjacent
    
    for i in range(arr.shape[0]):
        adjacent_idx = dist_matrix[i].argsort()[:n_samples]
        adjacent_idx = adjacent_idx[np.isfinite(dist_matrix[i, adjacent_idx])]
        
        res[i] = arr[adjacent_idx].mean()
    
    return res

def smooth_adjacent_values(values, counter, adjacency_matrix, replace_na=True):
    res = values.copy()
    
    for i in range(len(values)): 
        if not(replace_na) and np.isnan(values[i]):
            continue
    
        adj_indices = np.where(adjacency_matrix[i] & np.isfinite(values))[0]
        adj_counter = counter[adj_indices]
        
        if (adj_counter == 0).all():
            res[i] = np.nan
        else:
            res[i] = np.average(values[adj_indices], weights=np.sqrt(adj_counter))
    
    return res


def impute_adjacent_values(values, counter, adjacency_matrix, replace_na=True):
    res = values.copy()
    
    for i in range(len(values)): 
        if np.isnan(values[i]):
            adj_indices = np.where(adjacency_matrix[i] & np.isfinite(values))[0]
            adj_counter = counter[adj_indices]

            if (adj_counter == 0).all():
                res[i] = np.nan
            else:
                res[i] = np.average(values[adj_indices], weights=np.sqrt(adj_counter))
    
    return res


def smooth_adjacent_by_clusters(arr, dist_matrix, cluster_sizes, valid_mask):
    res = arr.copy()
    
    for i in range(arr.shape[0]):
        adjacent_idx = dist_matrix[i].argsort()[:cluster_sizes[i]]
        adjacent_idx = adjacent_idx[np.isfinite(dist_matrix[i, adjacent_idx]) & valid_mask[adjacent_idx]]
        
        res[i] = arr[adjacent_idx].mean()
    
    return res

def pac_to_dist_graph(arr,  valid_mask, adjacency_mask):
    arr_imputed = arr.copy()
    arr_imputed[~valid_mask] = np.nan

    pac_diff = np.abs(arr_imputed[:, None] - arr_imputed[None])
    
    zero_indices = np.isnan(pac_diff) | np.logical_not(adjacency_mask)

    # pac_diff = 1 - pac_diff/np.nanmax(pac_diff) + 0.1 # lets have some weight even for max distances
    pac_diff = 1/(pac_diff + 1.0)
    pac_diff[zero_indices] = 0.0
    
    G = nx.Graph(pac_diff)

    return G

def kmeans_smooth_resolution(arr, valid_mask, adjacency_mask, r=0.1):
    G = pac_to_dist_graph(arr, valid_mask, adjacency_mask)
    clusters = np.array(list(community_louvain.best_partition(G, resolution=r, random_state=42).values()))
    
    n_clusters = len(set(clusters))

    res = arr.copy()
    resolution = np.zeros(n_clusters)
    
    for i in range(n_clusters):
        res[clusters == i] = arr[(clusters == i) & valid_mask].mean()
        resolution[i] = ((clusters == i) & valid_mask).sum()
    
    res[~valid_mask] = np.nan
    
    return res, resolution, clusters, resolution[clusters].astype(int)


def create_labels_map(annot_fname, data_ch_names, offset=0):
    n_parcels = len(data_ch_names)
    labels_orig, _, annot_ch_names = nib.freesurfer.io.read_annot(annot_fname)
    
    annot_ch_names = [n.decode() for n in annot_ch_names[1:]]
    labels_orig -= 1

    labels = labels_orig.copy()
    
    orig_to_data = dict()
        
    for ch_idx, name in enumerate(data_ch_names):
        orig_idx = ch_idx + offset
        labels[labels_orig == orig_idx] = ch_idx
        orig_to_data[orig_idx] = ch_idx
        
    new_indices = np.arange(n_parcels)
    for old, new in orig_to_data.items():
        new_indices[new] = old
        
    return new_indices


def convert_matrix_indices(arr, annot_fname, data_ch_names, offset=0):
    n_parcels = len(data_ch_names)
    end = n_parcels//2 + offset
    
    res = np.zeros_like(arr)

    res[offset:end, offset:end] = arr[:n_parcels//2, :n_parcels//2].copy()
    
    return res
