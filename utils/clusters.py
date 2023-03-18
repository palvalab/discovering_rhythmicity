import numpy as np
import networkx as nx

import scipy
import scipy.cluster.vq
import scipy.spatial.distance

from itertools import combinations
import skimage

dst = scipy.spatial.distance.euclidean

def hclust(data, k):
    M = scipy.cluster.hierarchy.linkage(data, method='ward')
    kml = scipy.cluster.hierarchy.fcluster(M, t=k, criterion='maxclust') - 1
    kmc = [data[kml == l].mean(axis=0) for l in range(k)]
    
    return np.array(kmc), np.array(kml)
    
def compare_profiles(x, y):
    def _compare(a,b):
        if a < b:
            return -1
        elif a > b:
            return 1
        else:
            return 0

    _, x_val = x
    _, y_val = y
    
    if x_val[0] == y_val[0]:
        return _compare(x_val[1].argmax(), y_val[1].argmax())
    else:
        return _compare(x_val[0], y_val[0])



def get_percolated_cliques(G, k):
    perc_graph = nx.Graph()
    cliques = list(frozenset(c) for c in nx.find_cliques(G) if len(c) >= k)
    perc_graph.add_nodes_from(cliques)
    
    # Add an edge in the clique graph for each pair of cliques that percolate
    for c1, c2 in combinations(cliques, 2):
        if len(c1.intersection(c2)) >= (k - 1):
            perc_graph.add_edge(c1, c2)

    for component in nx.connected_components(perc_graph):
        yield(frozenset.union(*component))

def compute_surr_cluster_sizes(X, n_rounds=1000):
    X_shuf = X.copy()
    
    i_upper = np.triu_indices(X.shape[0], 1)
    i_lower = np.tril_indices(X.shape[0], -1)
    
    res = np.zeros(n_rounds)
    
    for i in range(n_rounds):
        vals_upper = X_shuf[i_upper]
        np.random.shuffle(vals_upper)
        
        X_shuf[i_upper] = X_shuf[i_lower] = vals_upper
        
        labels = measure.label(X_shuf, connectivity=2)
        _, cnts = np.unique(labels, return_counts=True)
        
        res[i] = cnts[1:].max() # exclude 0
        
    return res

def remove_small_clusters(X, threshold):
    image_labeled = skimage.measure.label(X, connectivity=2)
    labels, counts = np.unique(image_labeled, return_counts=True)
    
    to_remove = labels[counts < threshold]
    mask = np.isin(image_labeled, to_remove)
    
    res = X.copy()
    res[mask] = False
    
    return res