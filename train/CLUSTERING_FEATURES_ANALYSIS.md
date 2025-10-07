# Clustering Features Analysis

## Overview

Your clustering algorithm uses **5 main features** to group network nodes into clusters. These features capture both network topology and traffic characteristics.

---

## The 5 Clustering Features

### Feature 1: Node Degree (deg)
**Code:** `cluster.py`, line 31
```python
deg = np.array([G.degree(i) for i in range(n)], dtype=float)
```

**What it measures:**
- Number of edges connected to each node
- Indicates how well-connected a node is

**Example:**
- Node A has 15 edges → deg = 15
- Node B has 5 edges → deg = 5
- Nodes with similar degrees tend to cluster together

**Why it matters:**
- High-degree nodes are often core/hub nodes
- Low-degree nodes are often edge/peripheral nodes
- Grouping nodes with similar connectivity patterns

---

### Feature 2: Betweenness Centrality (btw)
**Code:** `cluster.py`, lines 32-35
```python
btw = nx.betweenness_centrality(G, k=min(n, 50), normalized=True, seed=42)
```

**What it measures:**
- How often a node appears on shortest paths between other nodes
- Indicates how "central" or "important" a node is for routing

**Example:**
- Node A is on 80% of shortest paths → btw = 0.8 (high)
- Node B is on 10% of shortest paths → btw = 0.1 (low)

**Why it matters:**
- High betweenness = critical routing node (bottleneck)
- Low betweenness = peripheral node
- Helps identify core vs edge nodes

**Note:** Only calculated for networks ≤ 200 nodes (expensive to compute)

---

### Feature 3: Incoming Traffic (tin)
**Code:** `cluster.py`, line 38 + `cluster.py`, line 420
```python
traffic_in = traffic_matrix.sum(axis=0)  # Sum over sources
tin = (traffic_in - traffic_in.mean()) / (traffic_in.std() + 1e-6)
```

**What it measures:**
- Total traffic flowing INTO each node
- Aggregated from all source nodes

**Example:**
- Node A receives 1000 bytes from all sources → traffic_in[A] = 1000
- Node B receives 200 bytes from all sources → traffic_in[B] = 200

**Why it matters:**
- High incoming traffic = popular destination (server, data center)
- Low incoming traffic = client node
- Nodes with similar traffic patterns cluster together

---

### Feature 4: Outgoing Traffic (tout)
**Code:** `cluster.py`, line 39 + `cluster.py`, line 421
```python
traffic_out = traffic_matrix.sum(axis=1)  # Sum over destinations
tout = (traffic_out - traffic_out.mean()) / (traffic_out.std() + 1e-6)
```

**What it measures:**
- Total traffic flowing OUT OF each node
- Aggregated to all destination nodes

**Example:**
- Node A sends 800 bytes to all destinations → traffic_out[A] = 800
- Node B sends 100 bytes to all destinations → traffic_out[B] = 100

**Why it matters:**
- High outgoing traffic = active client/sender
- Low outgoing traffic = passive receiver
- Helps distinguish client vs server nodes

---

### Feature 5: Service Class Share (svc_share)
**Code:** `cluster.py`, line 40 + `env.py`, lines 176-184
```python
# 6-dimensional vector per node (one per priority class)
svc_share = [p1_ratio, p2_ratio, p3_ratio, p4_ratio, p5_ratio, p6_ratio]
```

**What it measures:**
- Distribution of traffic priorities (1-6) for each node
- Shows what types of services each node handles

**Example:**
- Node A: [0.8, 0.1, 0.05, 0.05, 0, 0] → Mostly priority 1 (high priority)
- Node B: [0.1, 0.1, 0.2, 0.3, 0.2, 0.1] → Mixed priorities

**Why it matters:**
- Nodes handling similar service types cluster together
- High-priority nodes (p1, p2) vs low-priority nodes (p5, p6)
- Helps create QoS-aware clusters

---

## Feature Vector Structure

Each node is represented by an **11-dimensional feature vector**:

```python
X = [deg, btw, tin, tout, svc1, svc2, svc3, svc4, svc5, svc6]
#    [1]  [2]  [3]  [4]   [5]   [6]   [7]   [8]   [9]   [10]
```

**Dimensions:**
- 1 dimension: Node degree
- 1 dimension: Betweenness centrality
- 1 dimension: Incoming traffic
- 1 dimension: Outgoing traffic
- 6 dimensions: Service class distribution (priority 1-6)

**Total: 11 features per node**

---

## Feature Normalization

All features are **normalized** to have mean=0 and std=1:

```python
deg = (deg - deg.mean()) / (deg.std() + 1e-6)
btw = (btw - btw.mean()) / (btw.std() + 1e-6)
tin = (traffic_in - traffic_in.mean()) / (traffic_in.std() + 1e-6)
tout = (traffic_out - traffic_out.mean()) / (traffic_out.std() + 1e-6)
```

**Why normalize?**
- Prevents features with large values from dominating
- Ensures all features contribute equally to clustering
- Standard practice in machine learning

---

## How Features Are Used in Clustering

### Step 1: Extract Features
```python
X = featureize_graph(G, traffic_in, traffic_out, svc_share)
# X is now a (n x 11) matrix, where n = number of nodes
```

### Step 2: Run Clustering Algorithm
```python
# K-means clustering
labels, centroids = _kmeans(X, k=3, seed=42)

# Or DP-means (adaptive k)
labels, k = dp_means_clustering(X, lambda_param=2.0)

# Or Silhouette-based (optimal k)
k = determine_optimal_k_silhouette(X, k_range=(2, 10))
labels, _ = _kmeans(X, k=k)
```

### Step 3: Assign Nodes to Clusters
```python
cluster_map = {i: int(labels[i]) for i in range(n)}
# cluster_map[node_id] = cluster_id
```

---

## Feature Importance Analysis

### Most Important Features:

1. **Betweenness Centrality (btw)** - Distinguishes core vs edge nodes
2. **Service Class Share (svc_share)** - Groups nodes by traffic type
3. **Traffic In/Out (tin, tout)** - Identifies traffic patterns

### Less Important Features:

4. **Node Degree (deg)** - Correlated with betweenness
5. **Individual service classes** - Often sparse (many zeros)

---

## Example Clustering Scenario

### Network with 200 nodes:

**Cluster 0: Core Routers**
- High betweenness (0.8-1.0)
- High degree (15-20 edges)
- Moderate traffic in/out
- Mixed service classes

**Cluster 1: Data Center Servers**
- Low betweenness (0.1-0.3)
- Moderate degree (8-12 edges)
- **High incoming traffic** (receiving data)
- Low outgoing traffic
- Mostly high-priority services (p1, p2)

**Cluster 2: Client Nodes**
- Low betweenness (0.0-0.2)
- Low degree (3-6 edges)
- Low incoming traffic
- **High outgoing traffic** (sending requests)
- Mixed service classes

---

## Dynamic Updates

Features are **recalculated every 30 steps** (default):

```python
# config.json
"recluster_every_steps": 30
```

**What happens:**
1. Current flows are analyzed
2. Traffic matrix is rebuilt
3. Service class distribution is updated
4. Features are re-extracted
5. Clustering is re-run
6. Cluster assignments may change

**Why dynamic?**
- Traffic patterns change over time (peak vs off-peak)
- Network conditions evolve
- Adaptive clustering responds to changes

---

## Feature Summary Table

| Feature | Dimension | Source | Purpose |
|---------|-----------|--------|---------|
| **deg** | 1 | Network topology | Node connectivity |
| **btw** | 1 | Network topology | Routing importance |
| **tin** | 1 | Traffic matrix | Incoming demand |
| **tout** | 1 | Traffic matrix | Outgoing demand |
| **svc_share** | 6 | Flow priorities | Service type distribution |
| **Total** | **11** | - | - |

---

## Code Flow Summary

```
env.py: _clusterize()
    ↓
1. Build traffic matrix from current flows
    ↓
2. Calculate service class distribution
    ↓
cluster.py: dynamic_clustering()
    ↓
3. Extract features: featureize_graph()
    ↓
4. Run clustering algorithm (k-means, DP-means, etc.)
    ↓
5. Return cluster assignments
    ↓
env.py: Update _cluster_map
```

---

## Key Insights

### Why These Features?

1. **Topology features (deg, btw)** - Capture network structure
2. **Traffic features (tin, tout)** - Capture demand patterns
3. **Service features (svc_share)** - Capture QoS requirements

### What Makes a Good Cluster?

- Nodes with **similar roles** (core, edge, server, client)
- Nodes with **similar traffic patterns** (high/low demand)
- Nodes with **similar service requirements** (priority levels)

### Benefits:

- **Energy efficiency** - Turn off links within low-traffic clusters
- **QoS management** - Prioritize high-priority clusters
- **Load balancing** - Distribute traffic across clusters
- **Scalability** - Reduce action space for RL agent

---

## Configuration Options

You can control clustering behavior in `config.json`:

```json
{
  "clustering_method": "dp_means_adaptive",  // Algorithm choice
  "clustering_k_range": [2, 10],            // Search range for k
  "recluster_every_steps": 30,              // How often to recluster
  "adaptive_clustering": true,              // Enable dynamic k
  "dp_means_lambda": 2.0                    // Distance threshold
}
```

---

## Conclusion

Your clustering uses **11 features** across **5 categories** to intelligently group network nodes based on:
- Network topology (degree, betweenness)
- Traffic patterns (incoming, outgoing)
- Service requirements (priority distribution)

This creates meaningful clusters that help the RL agent make better energy-saving decisions while maintaining network performance.
