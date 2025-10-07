# Reinforcement Learning State and Reward Analysis

## Overview
This document provides a detailed analysis of the state representation and reward mechanism in the SDN environment (`train/env.py`).

---

## 1. State Representation (Observation Space)

### 1.1 State Dimensions
The observation space has **dynamic dimensions** based on clustering configuration:
- **Formula**: `obs_dim = max_clusters × 6 + 3`
- **Default**: With `max_clusters = 10`, the observation dimension is **63 features**

### 1.2 State Components

#### **A. Per-Cluster Features (6 features × K clusters)**
For each cluster `c` in the network, the following 6 features are computed:

1. **`traffic_in` (tin)**: Number of flows with destination nodes in cluster `c`
   - Measures incoming traffic demand to the cluster
   - Computed as: `sum(1 for f in flows if f.t in cluster_nodes)`

2. **`traffic_out` (tout)**: Number of flows with source nodes in cluster `c`
   - Measures outgoing traffic demand from the cluster
   - Computed as: `sum(1 for f in flows if f.s in cluster_nodes)`

3. **`active_links`**: Number of currently active (ON) links within the cluster
   - Indicates how many links are available for routing within the cluster
   - Computed from subgraph edges where `active == 1`

4. **`mean_util`**: Average link utilization within the cluster
   - Represents congestion level (0.0 = no traffic, 1.0 = at capacity, >1.0 = overloaded)
   - Computed as: `mean(utilization[(u,v)] for all edges in cluster)`
   - **Critical for energy-performance trade-off decisions**

5. **`svc_hi_share`**: Count of high-priority flows (priority ≤ 2) involving cluster nodes
   - Indicates presence of latency-sensitive traffic
   - Used to protect critical flows from link deactivation

6. **`svc_lo_share`**: Count of low-priority flows (priority ≥ 5) involving cluster nodes
   - Indicates presence of best-effort traffic
   - Can tolerate higher latency and link deactivation

**Adaptive Clustering Note**: When using adaptive clustering (e.g., DP-means), if the actual number of clusters is less than `max_clusters`, the remaining cluster slots are **zero-padded** to maintain consistent observation dimensions.

#### **B. Global Inter-Cluster Features (3 features)**

7. **`inter_ratio`**: Ratio of active inter-cluster links to total inter-cluster links
   - Formula: `inter_active / inter_total`
   - Measures connectivity between clusters
   - Range: [0.0, 1.0] where 1.0 = all inter-cluster links are active

8. **`phase`**: Time-of-day encoding (normalized episode progress)
   - Formula: `current_time / (max_steps_per_episode - 1)`
   - Range: [0.0, 1.0]
   - Helps agent learn time-dependent traffic patterns (peak vs off-peak)

9. **`last_a`**: Normalized previous action
   - Formula: `last_action / (action_n - 1)`
   - Range: [0.0, 1.0]
   - Provides temporal context for sequential decision-making

### 1.3 State Construction Code
```python
def _observe(self, utilization: Dict[Tuple[int,int], float] = None) -> np.ndarray:
    K_actual = self._actual_num_clusters
    K_max = self.max_clusters
    feats = []
    
    # Per-cluster features
    for c in range(K_actual):
        nodes = [i for i in range(self.n) if self._cluster_map.get(i, 0) == c]
        # ... compute 6 features per cluster ...
        feats.extend([tin, tout, active_links, mean_util, svc_hi, svc_lo])
    
    # Zero-padding for adaptive clustering
    if self.adaptive_clustering and K_actual < K_max:
        padding = [0.0] * 6 * (K_max - K_actual)
        feats.extend(padding)
    
    # Global features
    feats.extend([inter_ratio, phase, last_a])
    return np.array(feats, dtype=np.float32)
```

---

## 2. Action Space

### 2.1 Action Dimensions
- **Formula**: `action_n = max_clusters × len(cluster_bins) + len(inter_keep_opts)`
- **Default**: `10 × 9 + 4 = 94 discrete actions`

### 2.2 Action Components

#### **A. Per-Cluster Utilization Thresholds**
- **Purpose**: Determine which links to deactivate within each cluster based on utilization
- **Options**: 9 threshold bins `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`
- **Mechanism**: Links with utilization below the threshold are candidates for deactivation
- **Encoding**: Action is decoded in base-`len(cluster_bins)` to extract per-cluster thresholds

#### **B. Inter-Cluster Minimum Edges**
- **Purpose**: Ensure connectivity between clusters
- **Options**: `[2, 3, 4, 5]` minimum inter-cluster links to keep active
- **Mechanism**: Forces at least N links between different clusters to remain ON

### 2.3 Action Decoding
```python
def _decode_action(self, a: int):
    binsK = len(self.cluster_bins)
    thresholds_idx = []
    
    # Extract per-cluster thresholds (base-binsK encoding)
    for _ in range(self.max_clusters):
        thresholds_idx.append(a % binsK)
        a //= binsK
    
    # Extract inter-cluster keep option
    inter_idx = a % len(self.inter_keep_opts)
    
    thresholds = [float(self.cluster_bins[i]) for i in thresholds_idx]
    return thresholds, self.inter_keep_opts[inter_idx]
```

---

## 3. Reward Function

### 3.1 Reward Formula
```python
reward = energy_saving - (latency_penalty + sla_penalty + overload_penalty)
```

### 3.2 Reward Components

#### **A. Energy Saving (Positive Reward)**
```python
base_all_on = energy_on × num_edges  # Energy if all links are ON
energy = active_links × energy_on + inactive_links × energy_sleep
energy_saving = (base_all_on - energy) / base_all_on
```
- **Range**: [0.0, 1.0]
- **Interpretation**: 
  - 0.0 = All links ON (no savings)
  - 1.0 = Maximum savings (theoretical, would disconnect network)
- **Configuration**:
  - `energy_on = 1.0` (power consumption of active link)
  - `energy_sleep = 0.1` (power consumption of sleeping link)

#### **B. Latency Penalty (Negative Reward)**
```python
latency_penalty = 0.001 × avg_latency_ms
```
- **Purpose**: Penalize high average flow latency
- **Weight**: 0.001 (relatively small to allow trade-offs)
- **Latency Calculation**:
  - Base latency: Sum of edge delays on path
  - Congestion delay: `congestion_factor × (utilization - 1.0) × edge_delay` if utilization > 1.0
  - Unroutable flows: Penalized with 50ms latency

#### **C. SLA Violation Penalty (Negative Reward)**
```python
sla_penalty = 0.05 × sla_violation_percentage
```
- **Purpose**: Strongly penalize flows exceeding their priority-based SLA latency
- **Weight**: 0.05 (5× stronger than latency penalty)
- **SLA Thresholds** (from config):
  - Priority 1: 1.0 ms (ultra-low latency)
  - Priority 2: 2.0 ms
  - Priority 3: 4.0 ms
  - Priority 4: 8.0 ms
  - Priority 5: 16.0 ms
  - Priority 6: 32.0 ms (best effort)

#### **D. Overload Penalty (Negative Reward)**
```python
avg_util_pct = average_utilization × 100  # ACTUAL value, not capped

if avg_util_pct > 100.0:
    # Catastrophic overload
    excess = avg_util_pct - 100.0
    overload_penalty = 10.0 + 0.1 × (excess²)  # Quadratic growth
    
elif avg_util_pct > 80.0:
    # High utilization warning zone
    excess = avg_util_pct - 80.0
    overload_penalty = 0.1 × (excess^1.5)  # Exponential growth
    
else:
    overload_penalty = 0.0
```

**Penalty Breakdown**:
- **80-100% utilization**: Exponential penalty (0.2 at 84%, 0.9 at 90%, 2.8 at 100%)
- **>100% utilization**: Catastrophic penalty (base 10 + quadratic growth)
- **Purpose**: Prevent network overload which would cause packet loss in real systems

### 3.3 Reward Weights Summary
| Component | Weight | Purpose |
|-----------|--------|---------|
| Energy Saving | +1.0 | Maximize energy efficiency |
| Latency Penalty | -0.001 | Minimize average latency |
| SLA Violation | -0.05 | Strongly protect QoS |
| Overload (80-100%) | -0.1×(excess^1.5) | Prevent congestion |
| Overload (>100%) | -10.0 - 0.1×(excess²) | Catastrophic prevention |

---

## 4. State-Action-Reward Dynamics

### 4.1 Decision Flow
```
1. Observe State (63 features)
   ├─ Per-cluster: traffic, utilization, priorities
   ├─ Inter-cluster: connectivity ratio
   └─ Temporal: time-of-day, previous action

2. Select Action (1 of 94)
   ├─ Per-cluster thresholds (9 options × 10 clusters)
   └─ Inter-cluster min edges (4 options)

3. Apply Action
   ├─ Deterministic link deactivation (priority-aware)
   ├─ Protect high-priority flow paths
   └─ Ensure minimum inter-cluster connectivity

4. Generate Flows
   ├─ Traffic load mode (low/high)
   ├─ Peak/off-peak patterns
   └─ Priority assignment

5. Route & Measure
   ├─ Shortest path routing on active links
   ├─ Calculate utilization & latency
   └─ Check SLA violations

6. Compute Reward
   ├─ Energy savings (positive)
   └─ Penalties (latency, SLA, overload)

7. Return Next State
```

### 4.2 Key State-Reward Relationships

#### **High Utilization → Low Threshold**
- **State**: `mean_util` is high (e.g., 0.8)
- **Good Action**: Choose high threshold (e.g., 0.7-0.9) to keep more links active
- **Reward**: Avoid overload penalty, maintain low latency

#### **Low Utilization → High Threshold**
- **State**: `mean_util` is low (e.g., 0.2)
- **Good Action**: Choose low threshold (e.g., 0.1-0.3) to deactivate more links
- **Reward**: Maximize energy savings, minimal performance impact

#### **High Priority Traffic → Conservative Deactivation**
- **State**: `svc_hi_share` is high
- **Good Action**: Higher thresholds + higher inter-cluster connectivity
- **Reward**: Avoid SLA violations (strong penalty)

#### **Peak Time → More Active Links**
- **State**: `phase` indicates peak period (e.g., 0.2-0.4 for region 0)
- **Good Action**: Higher thresholds to handle increased traffic
- **Reward**: Balance energy vs. performance during high demand

---

## 5. Traffic Generation & Flow Characteristics

### 5.1 Traffic Load Modes
From `config.json`:

| Mode | Target Utilization | Flow Intensity | Peak Prob | Off-Peak Prob |
|------|-------------------|----------------|-----------|---------------|
| **Low** | 20-40% | 2.0× | 0.7 | 0.4 |
| **High** | 60-80% | 2.5× | 0.8 | 0.4 |

### 5.2 Flow Generation Process
```python
def _generate_new_flows(self):
    for region in regions:
        in_peak = current_step in peak_range[region]
        
        # Determine flow probability
        if in_peak:
            flow_prob = traffic_config["peak_flow_probability"]
        else:
            flow_prob = traffic_config["offpeak_flow_probability"]
        
        # Generate flows
        for _ in range(num_potential_flows):
            if random() < flow_prob:
                s, t = random_hosts_in_region()
                flow_size = calculate_adaptive_flow_size(s, t, in_peak)
                priority = assign_priority_by_size(flow_size, in_peak)
                ttl = random_int(5, 15)  # Flow lifetime
                
                flows.append(Flow(s, t, flow_size, priority, ttl))
    
    # Remove expired flows
    flows = [f for f in flows if f.ttl > 0]
```

### 5.3 Flow Priority Assignment
Based on flow size and peak status:

| Flow Size (bytes) | Peak Priority | Off-Peak Priority |
|-------------------|---------------|-------------------|
| > 30,000 | 1-2 | 1-3 |
| 15,000-30,000 | 2-4 | 3-4 |
| 5,000-15,000 | 3-5 | 3-5 |
| < 5,000 | 4-6 | 4-6 |

**Rationale**: Larger flows (e.g., video streaming, file transfers) get higher priority to ensure smooth transmission.

---

## 6. Routing & Latency Calculation

### 6.1 Routing Algorithm
- **Method**: Shortest path on active links (Dijkstra's algorithm)
- **Weight**: Edge delay (ms)
- **Fallback**: If no path exists, flow is unrouted (50ms penalty + SLA violation)

### 6.2 Latency Components
```python
flow_latency = base_latency + congestion_delay

# Base latency: sum of edge delays on path
base_latency = sum(edge["delay_ms"] for edge in path)

# Congestion delay: per-edge penalty if overloaded
for edge in path:
    utilization_fraction = edge_load / edge_capacity
    if utilization_fraction > 1.0:
        congestion_delay += congestion_factor × (utilization_fraction - 1.0) × edge_delay
```

**Configuration**:
- `edge_base_delay_ms = 0.3` ms (fiber propagation delay)
- `congestion_delay_factor = 2.0` (multiplier for overload penalty)

### 6.3 Utilization Calculation
```python
# For each flow routed on path
for edge in path:
    edge_utilization[edge] += flow_size / 1000.0  # Simplified load units

# Normalize by capacity
utilization_fraction = edge_utilization / edge_capacity
```

**Note**: Utilization > 1.0 indicates overload (more traffic than capacity).

---

## 7. Clustering & Hierarchical Control

### 7.1 Clustering Methods
From `config.json`:
- **Adaptive Clustering**: Enabled (`adaptive_clustering: true`)
- **Method**: `dp_means_adaptive` (DP-means with adaptive K)
- **K Range**: [2, 10] clusters
- **Lambda**: 2.0 (distance threshold for creating new clusters)

### 7.2 Clustering Features
The clustering algorithm uses:
1. **Traffic Matrix (TM)**: Flow sizes between node pairs
2. **Service Class Share**: Distribution of flow priorities per node

```python
def _clusterize(self):
    # Build traffic matrix from current flows
    tm = zeros(n, n)
    for flow in flows:
        tm[flow.s, flow.t] += flow.size
    
    # Build service class distribution
    svc_share = zeros(n, 6)  # 6 priority classes
    for flow in flows:
        svc_share[flow.s, flow.prio-1] += 1
        svc_share[flow.t, flow.prio-1] += 1
    
    # Normalize
    svc_share /= sum(svc_share, axis=1)
    
    # Perform clustering
    cluster_map = dynamic_clustering(G, tm, svc_share, method="dp_means_adaptive")
```

### 7.3 Reclustering
- **Frequency**: Every 30 steps (`recluster_every_steps: 30`)
- **Purpose**: Adapt cluster boundaries to changing traffic patterns
- **Statistics Tracked**:
  - Cluster count history
  - Reclustering events
  - Total reclustering time

---

## 8. Deterministic Link Deactivation

### 8.1 Algorithm Types
From `config.json`:
- **Current**: `"deactivation_algorithm": "priority"`
- **Options**: `"greedy"`, `"priority"`, `"balanced"`

### 8.2 Priority-Aware Deactivation
```python
def apply_complete_deactivation_strategy(graph, thresholds, flows):
    # 1. Identify high-priority flow paths
    protected_edges = set()
    for flow in flows:
        if flow.prio <= 2:  # High priority
            path = shortest_path(flow.s, flow.t)
            protected_edges.update(path_edges)
    
    # 2. Apply per-cluster thresholds
    for cluster_id, threshold in enumerate(thresholds):
        for edge in cluster_edges[cluster_id]:
            if edge not in protected_edges:
                if edge_utilization < threshold:
                    edge["active"] = 0  # Deactivate
    
    # 3. Ensure inter-cluster connectivity
    ensure_min_inter_cluster_edges(inter_keep_min)
```

**Key Features**:
- **Protection**: High-priority flows (priority ≤ 2) have their paths protected
- **Threshold-Based**: Low-utilization links are deactivated
- **Connectivity**: Minimum inter-cluster links are always kept active

---

## 9. Key Insights & Design Rationale

### 9.1 Multi-Objective Optimization
The RL agent must balance:
1. **Energy Efficiency**: Deactivate as many links as possible
2. **QoS Maintenance**: Keep latency low and avoid SLA violations
3. **Network Resilience**: Prevent overload and maintain connectivity

### 9.2 Hierarchical Approach
- **Clustering**: Reduces action space complexity (10 clusters vs 200 nodes)
- **Per-Cluster Control**: Allows localized optimization
- **Inter-Cluster Coordination**: Ensures global connectivity

### 9.3 Temporal Dynamics
- **Peak/Off-Peak**: Different traffic patterns require different strategies
- **Flow TTL**: Flows expire after 5-15 steps, creating dynamic traffic
- **Reclustering**: Adapts to changing traffic patterns every 30 steps

### 9.4 Priority-Aware Routing
- **SLA Differentiation**: 6 priority classes with different latency SLAs
- **Protected Paths**: High-priority flows are protected from link deactivation
- **Adaptive Sizing**: Flow sizes are calculated to achieve target utilization

---

## 10. Example Scenario Walkthrough

### Scenario: Low Traffic, Off-Peak Period

**Initial State**:
```
Cluster 0: tin=5, tout=4, active_links=180, mean_util=0.15, svc_hi=1, svc_lo=3
Cluster 1: tin=3, tout=5, active_links=150, mean_util=0.12, svc_hi=0, svc_lo=4
...
inter_ratio=0.95, phase=0.15, last_a=0.3
```

**Agent Decision**:
- **Interpretation**: Low utilization (0.15, 0.12), mostly low-priority traffic
- **Action**: Choose low thresholds (e.g., 0.2) for aggressive deactivation
- **Result**: Many links deactivated, high energy savings

**Outcome**:
```
Energy Saving: 0.65 (65% reduction)
Latency Penalty: 0.003 (3ms average)
SLA Penalty: 0.0 (no violations)
Overload Penalty: 0.0 (utilization still < 80%)
Reward: 0.65 - 0.003 = 0.647 (excellent!)
```

### Scenario: High Traffic, Peak Period

**Initial State**:
```
Cluster 0: tin=25, tout=28, active_links=180, mean_util=0.75, svc_hi=12, svc_lo=8
Cluster 1: tin=22, tout=24, active_links=150, mean_util=0.82, svc_hi=10, svc_lo=6
...
inter_ratio=1.0, phase=0.35, last_a=0.7
```

**Agent Decision**:
- **Interpretation**: High utilization (0.75, 0.82), many high-priority flows
- **Action**: Choose high thresholds (e.g., 0.8-0.9) to keep most links active
- **Result**: Few links deactivated, prioritize performance

**Outcome**:
```
Energy Saving: 0.15 (15% reduction)
Latency Penalty: 0.005 (5ms average)
SLA Penalty: 0.05 (1% violations)
Overload Penalty: 0.3 (utilization at 85%)
Reward: 0.15 - 0.005 - 0.05 - 0.3 = -0.205 (challenging trade-off)
```

**Learning**: Agent learns to be more conservative during peak periods with high-priority traffic.

---

## 11. Comparison with DQN_train Environment

### Key Differences

| Aspect | `train/env.py` (Hierarchical MARL) | `DQN_train/env.py` (Sequential DQN) |
|--------|-------------------------------------|-------------------------------------|
| **State Space** | 63 features (cluster-based) | 2×max_edges + 1 (link-based) |
| **Action Space** | 94 discrete (thresholds + inter-keep) | 2 discrete (close/keep per link) |
| **Decision Granularity** | Cluster-level (hierarchical) | Link-level (sequential) |
| **Scalability** | Scales to large networks (200 nodes) | Limited by sequential decisions |
| **Reward** | Energy - (latency + SLA + overload) | Energy per closed link - violations |
| **Clustering** | Dynamic (DP-means adaptive) | None |
| **Priority Handling** | Protected paths for high-priority | Implicit through routing |
| **Episode Termination** | Fixed steps (200) | After all links decided or violation |

### Advantages of Hierarchical Approach
1. **Scalability**: O(K) actions vs O(E) sequential decisions
2. **Global Optimization**: Considers entire network state simultaneously
3. **Adaptive Clustering**: Adjusts to traffic patterns
4. **Priority Protection**: Explicitly protects critical flows

---

## 12. Configuration Parameters Summary

### Network Topology
- **Nodes**: 200
- **Edges**: 2000
- **Regions**: 3
- **Hosts**: 80 (40% of nodes)
- **Edge Capacity**: Mean 5.0 Mbps, Std 1.0 Mbps
- **Base Delay**: 0.3 ms

### RL Training
- **Episodes**: 2000
- **Steps per Episode**: 200
- **Gamma**: 0.95 (discount factor)
- **Learning Rate**: 0.0003
- **Epsilon Decay**: 5000 steps (1.0 → 0.05)
- **Batch Size**: 256
- **Buffer Size**: 100,000

### Energy & Performance
- **Energy ON**: 1.0 W
- **Energy Sleep**: 0.1 W
- **Congestion Factor**: 2.0×
- **Routing K-Paths**: 3

### Clustering
- **Method**: DP-means adaptive
- **K Range**: [2, 10]
- **Lambda**: 2.0
- **Recluster Every**: 30 steps

---

## 13. Conclusion

The reinforcement learning environment implements a sophisticated **hierarchical, multi-objective optimization** problem:

### State (63 features)
- **Cluster-level**: Traffic, utilization, priorities (6 × 10 clusters)
- **Global**: Inter-cluster connectivity, time-of-day, action history

### Action (94 options)
- **Per-cluster thresholds**: 9 utilization levels × 10 clusters
- **Inter-cluster connectivity**: 4 minimum edge options

### Reward (multi-component)
- **Maximize**: Energy savings (up to +1.0)
- **Minimize**: Latency (-0.001×), SLA violations (-0.05×), overload (-10.0+)

### Key Mechanisms
1. **Dynamic Clustering**: Adapts network partitioning to traffic
2. **Priority-Aware Deactivation**: Protects critical flows
3. **Temporal Patterns**: Learns peak/off-peak strategies
4. **Hierarchical Control**: Scales to large networks

This design enables the RL agent to learn **energy-efficient routing policies** while maintaining **Quality of Service** guarantees in dynamic network conditions.
