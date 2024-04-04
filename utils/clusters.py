import numpy as np
import networkx as nx

from cdlib import NodeClustering
from cdlib.utils import convert_graph_formats
import karateclub


import scipy as sp
import scipy.cluster.vq
import scipy.spatial.distance

from itertools import combinations
import skimage

from collections import defaultdict, Counter


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
        
        labels = skimage.measure.label(X_shuf, connectivity=2)
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

def compute_surr_gain(data, n_rounds=1000, n_clusters = 9):
    noise_gain = np.zeros((n_rounds, n_clusters))

    for i in range(noise_gain.shape[0]):
        surr_data = data.copy()

        for j in range(surr_data.shape[0]):
            np.random.shuffle(surr_data[j])
            
        M_noise = sp.cluster.hierarchy.linkage(surr_data, method='ward')

        noise_cost = M_noise[-10:,2][::-1]
        noise_gain[i] = (noise_cost[:-1] - noise_cost[1:])/noise_cost[0]

    return noise_gain




# copied from cdlib because for some reason updated versions do not contain DANMF and older versions are not compatible with other libraries...
def danmf(
    g_original: object,
    layers: tuple = (32, 8),
    pre_iterations: int = 100,
    iterations: int = 100,
    seed: int = 42,
    lamb: float = 0.01,
) -> NodeClustering:
    """
    The procedure uses telescopic non-negative matrix factorization in order to learn a cluster memmbership distribution over nodes. The method can be used in an overlapping and non-overlapping way.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       Yes
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param layers: Autoencoder layer sizes in a list of integers. Default [32, 8].
    :param pre_iterations: Number of pre-training epochs. Default 100.
    :param iterations: Number of training epochs. Default 100.
    :param seed: Random seed for weight initializations. Default 42.
    :param lamb: Regularization parameter. Default 0.01.
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.danmf(G)

    :References:

    Ye, Fanghua, Chuan Chen, and Zibin Zheng. "Deep autoencoder-like nonnegative matrix factorization for community detection." Proceedings of the 27th ACM International Conference on Information and Knowledge Management. 2018.

    .. note:: Reference implementation: https://karateclub.readthedocs.io/
    """

    g = convert_graph_formats(g_original, nx.Graph)
    model = karateclub.DANMF(layers, pre_iterations, iterations, seed, lamb)

    mapping = {node: i for i, node in enumerate(g.nodes())}
    rev = {i: node for node, i in mapping.items()}
    H = nx.relabel_nodes(g, mapping)

    model.fit(H)
    members = model.get_memberships()

    # Reshaping the results
    coms_to_node = defaultdict(list)
    for n, c in members.items():
        coms_to_node[c].append(rev[n])

    coms = [list(c) for c in coms_to_node.values()]

    return NodeClustering(
        coms,
        g_original,
        "DANMF",
        method_parameters={
            "layers": layers,
            "pre_iteration": pre_iterations,
            "iterations": iterations,
            "seed": seed,
            "lamb": lamb,
            "fitted_model": model,
        },
        overlap=True,
    )


def get_communities_prob_threshold(mdl, threshold=0.1):
    res = list()

    for i in range(mdl._P.shape[-1]):
        res.append(np.where(mdl._P[:, i] >= threshold)[0].tolist())
    
    return res

def filter_small_communities(communities, significant_ratio, sign_threshold=0.05, min_size=3):
    res = [c for c in communities if (significant_ratio[c].mean() > sign_threshold) and (len(c) > min_size) and np.all(np.diff(c) == 1)]

    return res

def remove_small_communities(mask, min_size=5):
    res = np.zeros_like(mask)

    labels = skimage.measure.label(mask)
    counts = Counter(labels.flatten())
    counts.pop(0)


    for lbl, cnt in counts.items():
        if cnt >= min_size:
            res[labels == lbl] = True

    return res
    

def filter_out_distance_matrix(distance, threshold, freq_distance=10, min_size=5):
    distant_freqs_mask = np.abs(np.arange(81).reshape(1,-1) - np.arange(81).reshape(-1,1)) > 10
    similar_freqs_mask = distance > threshold

    similarity_mask_combined = distant_freqs_mask | similar_freqs_mask
    similarity_mask_cleared = ~remove_small_communities(~similarity_mask_combined, min_size=min_size)

    pac_val_corr_distance_filtered = 1 - distance.copy()
    pac_val_corr_distance_filtered[similarity_mask_cleared] = 0.0

    return pac_val_corr_distance_filtered