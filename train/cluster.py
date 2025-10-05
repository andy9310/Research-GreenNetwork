import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

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

def determine_optimal_k_silhouette(X: np.ndarray, k_range: Optional[Tuple[int, int]] = None, 
                                   seed: int = 42, max_k: int = 15) -> int:
    """
    Find optimal k using silhouette score.
    Tests k values in range and returns the k with highest silhouette score.
    """
    n = X.shape[0]
    if k_range is None:
        # Default: test from 2 to sqrt(n/2), capped at max_k
        k_min = 2
        k_max = min(int(np.sqrt(n / 2)), max_k, n - 1)
        k_range = (k_min, k_max)
    
    best_k = k_range[0]
    best_score = -1
    
    for k in range(k_range[0], k_range[1] + 1):
        if k >= n:
            break
        labels, _ = _kmeans(X, k=k, seed=seed)
        # Need at least 2 clusters for silhouette
        if len(np.unique(labels)) < 2:
            continue
        try:
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except:
            continue
    
    return best_k

def determine_optimal_k_elbow(X: np.ndarray, k_range: Optional[Tuple[int, int]] = None, 
                              seed: int = 42, max_k: int = 15) -> int:
    """
    Find optimal k using elbow method (within-cluster sum of squares).
    """
    n = X.shape[0]
    if k_range is None:
        k_min = 2
        k_max = min(int(np.sqrt(n / 2)), max_k, n - 1)
        k_range = (k_min, k_max)
    
    wcss_list = []
    k_values = []
    
    for k in range(k_range[0], k_range[1] + 1):
        if k >= n:
            break
        labels, centroids = _kmeans(X, k=k, seed=seed)
        # Calculate WCSS
        wcss = 0
        for j in range(k):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                wcss += np.sum((cluster_points - centroids[j]) ** 2)
        wcss_list.append(wcss)
        k_values.append(k)
    
    if len(wcss_list) < 3:
        return k_range[0]
    
    # Find elbow using rate of change
    wcss_arr = np.array(wcss_list)
    # Normalize
    wcss_norm = (wcss_arr - wcss_arr.min()) / (wcss_arr.max() - wcss_arr.min() + 1e-10)
    
    # Find point of maximum curvature
    k_arr = np.array(k_values)
    # Calculate second derivative approximation
    if len(wcss_norm) >= 3:
        diffs = np.diff(wcss_norm)
        second_diffs = np.diff(diffs)
        # Elbow is where second derivative is maximum
        elbow_idx = np.argmax(second_diffs) + 1
        return k_values[min(elbow_idx, len(k_values) - 1)]
    
    return k_range[0]

def determine_optimal_k_network_heuristic(G: nx.Graph, traffic_matrix: np.ndarray, 
                                          min_cluster_size: int = 10) -> int:
    """
    Network-aware heuristic for determining k.
    Considers: network connectivity, traffic load variation, network size.
    """
    n = G.number_of_nodes()
    
    # Factor 1: Network size - use sqrt heuristic with bounds
    size_k = max(2, min(int(np.sqrt(n / 2)), 15))
    
    # Factor 2: Network connectivity - modularity suggests natural partitions
    try:
        communities = nx.community.greedy_modularity_communities(G, weight=None)
        modularity_k = len(communities)
        # Ensure reasonable bounds
        modularity_k = max(2, min(modularity_k, 15))
    except:
        modularity_k = size_k
    
    # Factor 3: Traffic heterogeneity
    traffic_variance = np.var(traffic_matrix.sum(axis=0) + traffic_matrix.sum(axis=1))
    traffic_mean = np.mean(traffic_matrix.sum(axis=0) + traffic_matrix.sum(axis=1))
    cv = traffic_variance / (traffic_mean + 1e-10)  # coefficient of variation
    
    # Higher variance -> more clusters needed
    if cv > 1.0:
        traffic_k = min(int(size_k * 1.5), 15)
    elif cv > 0.5:
        traffic_k = size_k
    else:
        traffic_k = max(2, int(size_k * 0.75))
    
    # Combine factors (weighted average)
    k = int(0.4 * size_k + 0.3 * modularity_k + 0.3 * traffic_k)
    
    # Ensure minimum cluster size constraint
    k = min(k, n // min_cluster_size)
    
    return max(2, min(k, 15))

def dbscan_clustering(X: np.ndarray, eps: Optional[float] = None, min_samples: int = 3) -> Dict[int, int]:
    """
    DBSCAN clustering - automatically determines number of clusters.
    Returns mapping node_id -> cluster_id.
    Noise points (label -1) are assigned to nearest cluster.
    """
    # Auto-determine eps if not provided
    if eps is None:
        # Use median distance to k-th nearest neighbor as eps
        from sklearn.neighbors import NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors.fit(X)
        distances, _ = neighbors.kneighbors(X)
        eps = np.median(distances[:, -1]) * 1.5
    
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clusterer.fit_predict(X)
    
    # Handle noise points - assign to nearest cluster
    unique_labels = set(labels)
    if -1 in unique_labels:
        noise_mask = labels == -1
        if np.any(~noise_mask):  # If there are non-noise points
            from sklearn.neighbors import NearestNeighbors
            non_noise_X = X[~noise_mask]
            non_noise_labels = labels[~noise_mask]
            
            if len(non_noise_X) > 0:
                nn = NearestNeighbors(n_neighbors=1)
                nn.fit(non_noise_X)
                _, indices = nn.kneighbors(X[noise_mask])
                labels[noise_mask] = non_noise_labels[indices.flatten()]
    
    # Ensure labels are 0-indexed
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels])
    
    return {i: int(labels[i]) for i in range(len(labels))}

def hierarchical_clustering(X: np.ndarray, max_k: int = 15, method: str = 'ward') -> Tuple[np.ndarray, int]:
    """
    Hierarchical clustering with dynamic k selection.
    Uses inconsistency criterion to determine optimal cut.
    """
    n = X.shape[0]
    if n <= 2:
        return np.zeros(n, dtype=int), 1
    
    # Compute linkage
    try:
        if method == 'ward':
            Z = linkage(X, method='ward')
        else:
            distances = pdist(X)
            Z = linkage(distances, method=method)
    except:
        # Fallback to simple k-means
        labels, _ = _kmeans(X, k=min(3, n-1))
        return labels, len(np.unique(labels))
    
    # Try different cuts and choose based on inconsistency
    best_k = 2
    best_score = -np.inf
    
    for k in range(2, min(max_k + 1, n)):
        labels = fcluster(Z, k, criterion='maxclust')
        if len(np.unique(labels)) < 2:
            continue
        try:
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except:
            continue
    
    labels = fcluster(Z, best_k, criterion='maxclust')
    # Convert to 0-indexed
    labels = labels - 1
    
    return labels, best_k

def dp_means_clustering(X: np.ndarray, lambda_param: Optional[float] = None, 
                       max_iter: int = 100, seed: int = 42) -> Tuple[np.ndarray, int]:
    """
    DP-means clustering algorithm - dynamic k-means without fixed k.
    
    DP-means is a non-parametric extension of k-means that automatically 
    determines the number of clusters based on a distance threshold λ.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        lambda_param: Distance threshold for creating new clusters
                     If None, auto-estimated based on data scale
        max_iter: Maximum iterations
        seed: Random seed
    
    Returns:
        Tuple of (labels, num_clusters)
    """
    n, d = X.shape
    if n <= 1:
        return np.zeros(n, dtype=int), 1
    
    rng = np.random.default_rng(seed)
    
    # Auto-estimate lambda if not provided
    if lambda_param is None:
        # Use median pairwise distance as base, scale by sqrt(d) for dimensionality
        from sklearn.metrics import pairwise_distances
        if n <= 1000:  # For small datasets, compute all pairwise distances
            distances = pairwise_distances(X)
            # Use upper triangle (excluding diagonal)
            upper_tri = distances[np.triu_indices(n, k=1)]
            lambda_param = np.median(upper_tri) * 0.5  # Conservative estimate
        else:  # For large datasets, sample
            sample_size = min(1000, n)
            sample_indices = rng.choice(n, size=sample_size, replace=False)
            sample_X = X[sample_indices]
            distances = pairwise_distances(sample_X)
            upper_tri = distances[np.triu_indices(sample_size, k=1)]
            lambda_param = np.median(upper_tri) * 0.5
    
    # Initialize with first point as first centroid
    centroids = [X[0].copy()]
    labels = np.zeros(n, dtype=int)
    
    for iteration in range(max_iter):
        labels_old = labels.copy()
        
        # Assignment step
        for i in range(n):
            point = X[i]
            
            # Calculate distances to all existing centroids
            distances_to_centroids = [np.linalg.norm(point - centroid) for centroid in centroids]
            min_distance = min(distances_to_centroids)
            closest_centroid_idx = distances_to_centroids.index(min_distance)
            
            # If distance > lambda, create new cluster
            if min_distance > lambda_param:
                centroids.append(point.copy())
                labels[i] = len(centroids) - 1
            else:
                labels[i] = closest_centroid_idx
        
        # Update centroids
        for k in range(len(centroids)):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = cluster_points.mean(axis=0)
        
        # Check convergence
        if np.array_equal(labels, labels_old):
            break
    
    # Ensure labels are 0-indexed and consecutive
    unique_labels = np.unique(labels)
    if len(unique_labels) > 0:
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels])
    
    num_clusters = len(unique_labels)
    return labels, num_clusters

def dp_means_adaptive(X: np.ndarray, lambda_range: Tuple[float, float] = (1.0, 5.0),
                      n_lambda_tests: int = 10, k_range: Optional[Tuple[int, int]] = None, seed: int = 42) -> Tuple[np.ndarray, int, float]:
    """
    DP-means with adaptive lambda selection using silhouette score.
    
    Tests multiple lambda values and selects the one with best silhouette score.
    
    Args:
        X: Feature matrix
        lambda_range: (min_lambda, max_lambda) to test
        n_lambda_tests: Number of lambda values to test
        k_range: (min_k, max_k) constraint for number of clusters
        seed: Random seed
    
    Returns:
        Tuple of (labels, num_clusters, best_lambda)
    """
    n = X.shape[0]
    if n <= 2:
        return np.zeros(n, dtype=int), 1, lambda_range[0]
    
    # Generate lambda values to test
    lambda_values = np.linspace(lambda_range[0], lambda_range[1], n_lambda_tests)
    
    best_score = -np.inf
    best_labels = None
    best_k = 1
    best_lambda = lambda_range[0]
    
    for lambda_val in lambda_values:
        try:
            labels, k = dp_means_clustering(X, lambda_param=lambda_val, seed=seed)
            
            # Apply k_range constraint if provided
            if k_range is not None:
                if k < k_range[0] or k > k_range[1]:
                    continue
            
            # Skip if too few or too many clusters (general constraints)
            if k < 2 or k >= n:
                continue
            
            # Calculate silhouette score
            if len(np.unique(labels)) >= 2:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_labels = labels
                    best_k = k
                    best_lambda = lambda_val
        except:
            continue
    
    # Fallback if no good clustering found
    if best_labels is None:
        best_labels, best_k = dp_means_clustering(X, lambda_param=lambda_range[0], seed=seed)
        best_lambda = lambda_range[0]
    
    return best_labels, best_k, best_lambda

def dynamic_clustering(G: nx.Graph, traffic_matrix: np.ndarray, svc_class_share: np.ndarray, 
                      k: Optional[int] = None, 
                      method: str = 'silhouette',
                      k_range: Optional[Tuple[int, int]] = None,
                      lambda_param: Optional[float] = None,
                      seed: int = 42) -> Dict[int, int]:
    """
    Returns a mapping node_id -> cluster_id (0..k-1).
    
    Args:
        G: Network graph
        traffic_matrix: n x n demand volumes
        svc_class_share: n x C per-node distribution of classes (row-normalized)
        k: Number of clusters (if None, auto-determined)
        method: Method to determine k if k=None
                - 'silhouette': Use silhouette score optimization
                - 'elbow': Use elbow method
                - 'network_heuristic': Network-aware heuristic
                - 'dbscan': Density-based clustering
                - 'hierarchical': Hierarchical clustering
                - 'dp_means': DP-means clustering (dynamic k)
                - 'dp_means_adaptive': DP-means with adaptive lambda
        k_range: Optional (min_k, max_k) tuple for search range
        lambda_param: Lambda parameter for DP-means (distance threshold)
        seed: Random seed
    
    Returns:
        Dictionary mapping node_id -> cluster_id
    """
    n = G.number_of_nodes()
    traffic_in = traffic_matrix.sum(axis=0)
    traffic_out = traffic_matrix.sum(axis=1)
    X = featureize_graph(G, traffic_in, traffic_out, svc_class_share)
    
    # Auto-determine k if not provided
    if k is None:
        if method == 'silhouette':
            k = determine_optimal_k_silhouette(X, k_range=k_range, seed=seed)
        elif method == 'elbow':
            k = determine_optimal_k_elbow(X, k_range=k_range, seed=seed)
        elif method == 'network_heuristic':
            k = determine_optimal_k_network_heuristic(G, traffic_matrix)
        elif method == 'dbscan':
            return dbscan_clustering(X)
        elif method == 'hierarchical':
            labels, k = hierarchical_clustering(X, max_k=k_range[1] if k_range else 15)
            return {i: int(labels[i]) for i in range(n)}
        elif method == 'dp_means':
            labels, k = dp_means_clustering(X, lambda_param=lambda_param, seed=seed)
            return {i: int(labels[i]) for i in range(n)}
        elif method == 'dp_means_adaptive':
            labels, k, best_lambda = dp_means_adaptive(X, k_range=k_range, seed=seed)
            return {i: int(labels[i]) for i in range(n)}
        else:
            # Default fallback
            k = max(2, min(int(np.sqrt(n / 2)), 10))
    
    # Ensure k is valid
    k = max(2, min(k, n - 1))
    
    labels, _ = _kmeans(X, k=k, seed=seed)
    return {i: int(labels[i]) for i in range(n)}