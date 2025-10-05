# Adaptive Clustering Guide

## Overview

The clustering module now supports **adaptive/dynamic cluster determination** instead of using a fixed number of clusters. This allows the system to automatically adjust the number of clusters based on network topology, traffic patterns, and service class distribution.

## Configuration

Add these parameters to your `config.json`:

```json
{
  "adaptive_clustering": true,           // Enable adaptive clustering
  "clustering_method": "silhouette",     // Method to determine optimal k
  "clustering_k_range": [2, 10],        // Min and max clusters to test
  "num_clusters": 3                      // Used only if adaptive_clustering=false
}
```

## Available Clustering Methods

### 1. **Silhouette Score Method** (Recommended)
- **Method**: `"silhouette"`
- **Best for**: General purpose, balanced clusters
- **How it works**: Tests different k values and chooses the one with highest silhouette score (measures cluster cohesion and separation)
- **Pros**: Reliable, well-established metric
- **Cons**: Can be computationally expensive for large networks

```json
{
  "adaptive_clustering": true,
  "clustering_method": "silhouette",
  "clustering_k_range": [2, 10]
}
```

### 2. **Elbow Method**
- **Method**: `"elbow"`
- **Best for**: Cost-sensitive applications
- **How it works**: Finds the "elbow point" in within-cluster sum of squares (WCSS) curve
- **Pros**: Fast, intuitive
- **Cons**: Elbow point may not always be clear

```json
{
  "adaptive_clustering": true,
  "clustering_method": "elbow",
  "clustering_k_range": [2, 15]
}
```

### 3. **Network Heuristic** (SDN-Optimized)
- **Method**: `"network_heuristic"`
- **Best for**: Large-scale SDN networks, production environments
- **How it works**: Combines network size, connectivity (modularity), and traffic heterogeneity
  - Network size: Uses √(n/2) heuristic
  - Connectivity: Uses modularity-based community detection
  - Traffic variance: Adjusts k based on traffic load heterogeneity
- **Pros**: Very fast, network-aware, considers topology
- **Cons**: May not be as precise as optimization-based methods

```json
{
  "adaptive_clustering": true,
  "clustering_method": "network_heuristic",
  "clustering_k_range": [2, 15]
}
```

**Algorithm Details**:
- High traffic variance (CV > 1.0) → More clusters
- Moderate variance (CV > 0.5) → Standard number
- Low variance (CV ≤ 0.5) → Fewer clusters
- Ensures minimum cluster size of 10 nodes

### 4. **DBSCAN** (Density-Based)
- **Method**: `"dbscan"`
- **Best for**: Networks with varying density, irregular cluster shapes
- **How it works**: Automatically discovers clusters based on density (doesn't require k)
- **Pros**: Handles irregular shapes, auto-determines k, identifies outliers
- **Cons**: May create many small clusters, sensitive to parameters

```json
{
  "adaptive_clustering": true,
  "clustering_method": "dbscan"
}
```

Note: DBSCAN doesn't use `clustering_k_range` - it determines k automatically.

### 5. **Hierarchical Clustering**
- **Method**: `"hierarchical"`
- **Best for**: Understanding cluster hierarchy, dendrogram analysis
- **How it works**: Builds dendrogram, dynamically cuts based on silhouette scores
- **Pros**: Shows hierarchical structure, flexible cutting
- **Cons**: Computationally expensive for large networks

```json
{
  "adaptive_clustering": true,
  "clustering_method": "hierarchical",
  "clustering_k_range": [2, 15]
}
```

### 6. **DP-means** (Dynamic k-means) ⭐ **NEW & RECOMMENDED**
- **Method**: `"dp_means"`
- **Best for**: Dynamic environments, traffic-aware clustering, anomaly detection
- **How it works**: Non-parametric k-means that creates new clusters when distance > λ threshold
- **Pros**: No fixed k, adapts to data structure, handles outliers, fast
- **Cons**: Requires λ parameter tuning

```json
{
  "adaptive_clustering": true,
  "clustering_method": "dp_means",
  "dp_means_lambda": 0.5
}
```

### 7. **DP-means Adaptive** (Auto λ) ⭐ **BEST FOR SDN**
- **Method**: `"dp_means_adaptive"`
- **Best for**: Production SDN, automatic parameter optimization
- **How it works**: Tests multiple λ values, selects best using silhouette score
- **Pros**: No parameter tuning, optimal λ selection, network-aware
- **Cons**: Slightly slower than fixed λ

```json
{
  "adaptive_clustering": true,
  "clustering_method": "dp_means_adaptive"
}
```

## Comparison Table

| Method | Speed | Accuracy | Network-Aware | Auto k | Best Use Case |
|--------|-------|----------|---------------|--------|---------------|
| Silhouette | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ✅ | General purpose |
| Elbow | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ✅ | Quick deployment |
| Network Heuristic | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | Production SDN |
| DBSCAN | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ✅ | Irregular shapes |
| Hierarchical | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ✅ | Analysis |
| **DP-means** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | **Dynamic SDN** |
| **DP-means Adaptive** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | **Best Overall** |

## Recommendation for Your Use Case

For **200 nodes, 2000 edges, 3 regions** with **dynamic traffic patterns**:

### Option A: DP-means Adaptive (BEST FOR SDN) ⭐
```json
{
  "adaptive_clustering": true,
  "clustering_method": "dp_means_adaptive"
}
```

### Option B: DP-means (Manual λ)
```json
{
  "adaptive_clustering": true,
  "clustering_method": "dp_means",
  "dp_means_lambda": 0.5
}
```

### Option C: Network Heuristic (Fast + Network-Aware)
```json
{
  "adaptive_clustering": true,
  "clustering_method": "network_heuristic",
  "clustering_k_range": [3, 12]
}
```

### Option D: Research (Accurate + Validated)
```json
{
  "adaptive_clustering": true,
  "clustering_method": "silhouette",
  "clustering_k_range": [3, 10]
}
```

### Option E: Fixed (Baseline Comparison)
```json
{
  "adaptive_clustering": false,
  "num_clusters": 3
}
```

## Dynamic Reclustering

Clusters are recomputed every `recluster_every_steps` (default: 30) based on:
1. **Current traffic matrix** from active flows
2. **Service class distribution** from flow priorities
3. **Network topology** (degree, betweenness centrality)

This allows clusters to adapt as traffic patterns shift (peak vs off-peak).

## Feature Engineering

Clustering uses these per-node features:
- **Topology**: Node degree (normalized)
- **Centrality**: Betweenness centrality (for n ≤ 200)
- **Traffic**: Incoming traffic volume (normalized)
- **Traffic**: Outgoing traffic volume (normalized)
- **Service**: Service class distribution (6 priority levels)

## Monitoring Cluster Count

To track how many clusters are being used:

```python
# In your training loop
info = env.step(action)
actual_clusters = env._actual_num_clusters
print(f"Current clusters: {actual_clusters}")
```

## Tips for Tuning

1. **Too many clusters** (>15):
   - Reduce `clustering_k_range[1]`
   - Switch to "network_heuristic" method
   
2. **Too few clusters** (<2):
   - Increase `clustering_k_range[1]`
   - Check if network has sufficient diversity
   
3. **Unstable clustering**:
   - Increase `recluster_every_steps`
   - Use fixed clustering for debugging
   
4. **Performance issues**:
   - Use "network_heuristic" or "elbow"
   - Reduce `clustering_k_range` size
   - Increase `recluster_every_steps`

## Example: Comparing Methods

```python
import json

methods = ["silhouette", "elbow", "network_heuristic", "dbscan"]
results = {}

for method in methods:
    config = json.load(open("config.json"))
    config["clustering_method"] = method
    config["adaptive_clustering"] = True
    
    env = SDNEnv(config)
    obs = env.reset()
    
    results[method] = {
        "clusters": env._actual_num_clusters,
        "obs_dim": len(obs)
    }
    
print(results)
```

## Research Evaluation

For your paper comparison with ILP/MIP, Heuristic, and DQN:

1. **Baseline**: Fixed k=3 (regions)
2. **Adaptive**: Silhouette (k=2-10)
3. **Production**: Network heuristic (k=3-12)

Compare:
- Energy efficiency
- Latency/SLA violations
- Computational overhead
- Adaptability to traffic changes

## Backward Compatibility

To use **fixed clustering** (legacy behavior):

```json
{
  "adaptive_clustering": false,
  "num_clusters": 3
}
```

This maintains the original behavior where k = num_regions. 