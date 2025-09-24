import numpy as np
import networkx as nx
from typing import Dict, List, Tuple

def _kmeans(X: np.ndarray, k: int, max_iter: int = 50, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, d = X.shape
    centroids = X[rng.choice(n, size=k, replace=False)]
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # E-step
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        # M-step
        for j in range(k):
            pts = X[labels == j]
            if len(pts) > 0:
                centroids[j] = pts.mean(axis=0)
    return labels, centroids

def featureize_graph(G: nx.Graph, traffic_in: np.ndarray, traffic_out: np.ndarray, svc_share: np.ndarray) -> np.ndarray:
    """Build per-node feature: [degree, betweenness, in, out, svc_share...]"""
    n = G.number_of_nodes()
    deg = np.array([G.degree(i) for i in range(n)], dtype=float)
    if n <= 200:  # betweenness is expensive; fallback for large graphs
        btw = np.array(list(nx.betweenness_centrality(G, k=min(n, 50), normalized=True, seed=42).values()), dtype=float)
    else:
        btw = np.zeros(n, dtype=float)
    deg = (deg - deg.mean()) / (deg.std() + 1e-6)
    btw = (btw - btw.mean()) / (btw.std() + 1e-6)
    tin = (traffic_in - traffic_in.mean()) / (traffic_in.std() + 1e-6)
    tout = (traffic_out - traffic_out.mean()) / (traffic_out.std() + 1e-6)
    X = np.column_stack([deg, btw, tin, tout, svc_share])
    return X

def dynamic_clustering(G: nx.Graph, traffic_matrix: np.ndarray, svc_class_share: np.ndarray, k: int = 3, seed: int = 42) -> Dict[int, int]:
    """
    Returns a mapping node_id -> cluster_id (0..k-1).
    - traffic_matrix: n x n demand volumes
    - svc_class_share: n x C per-node distribution of classes (row-normalized)
    """
    n = G.number_of_nodes()
    traffic_in = traffic_matrix.sum(axis=0)
    traffic_out = traffic_matrix.sum(axis=1)
    X = featureize_graph(G, traffic_in, traffic_out, svc_class_share)
    labels, _ = _kmeans(X, k=k, seed=seed)
    return {i: int(labels[i]) for i in range(n)}