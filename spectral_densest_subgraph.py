import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import networkx as nx
from scipy.sparse import csr_matrix


def gsa_densest_subgraph(adjacency, use_abs=True, assume_undirected=True):
    A = adjacency.tocsr().astype(float)
    n = A.shape[0]

    
    if A.diagonal().sum() != 0:
        A = A.copy()
        A.setdiag(0)
        A.eliminate_zeros()

    if assume_undirected:

        vals, vecs = eigsh(A, k=1, which="LA")
        v1 = vecs[:, 0]
    else:

        A_sym = (A + A.T) * 0.5
        vals, vecs = eigsh(A_sym, k=1, which="LA")
        v1 = vecs[:, 0]

    scores = np.abs(v1) if use_abs else v1


    order = np.argsort(scores)[::-1]

    in_set = np.zeros(n, dtype=bool)
    internal_edge_weight = 0.0  
    best_density = -np.inf
    best_size = 0

    indptr = A.indptr
    indices = A.indices
    data = A.data

    for s, u in enumerate(order, start=1):
        start = indptr[u]
        end = indptr[u + 1]
        nbrs = indices[start:end]
        wts = data[start:end]

        if nbrs.size > 0:
            internal_edge_weight += wts[in_set[nbrs]].sum()

        in_set[u] = True

        density = internal_edge_weight / s 
        if density > best_density:
            best_density = density
            best_size = s

    best_nodes_idx = order[:best_size]
    return best_nodes_idx, float(best_density), order, scores





def spectral_densest_subgraph_from_adjacency(adjacency, node_attributes_dict=None):
    G = nx.Graph(adjacency)
    nodelist = list(G)
    nlen = len(nodelist)
    indexNodes = dict(zip(range(nlen), nodelist))
    A = nx.adjacency_matrix(G, nodelist=nodelist)
    A = csr_matrix(A) 

    best_idx, best_density, order, scores = gsa_densest_subgraph(A, use_abs=True)

    best_nodes = [indexNodes[i] for i in best_idx]
    return  best_density, best_nodes
