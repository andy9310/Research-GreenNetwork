# Comparative Experiment: Energy-Aware SDN Routing

## Overview

This experiment compares three energy-aware routing approaches across different network scales to evaluate their performance, scalability, and efficiency.

## Methods Compared

### 1. **Our Method: DQN with Adaptive Clustering**
- **Description**: Deep Q-Network with DP-means adaptive clustering
- **Key Features**:
  - Dynamic cluster-based link management
  - Reinforcement learning for optimal policy
  - Adaptive to traffic patterns
- **Implementation**: `../train.py` with `config.json`

### 2. **Baseline 1: Energy-Aware Routing (EAR)**
- **Paper**: "Energy-aware routing algorithms in Software-Defined Networks"
- **Description**: Heuristic-based approach using link utilization thresholds
- **Key Features**:
  - Greedy link deactivation based on utilization
  - Static threshold-based decisions
  - Fast computation
- **Implementation**: `baselines/energy_aware_routing.py`

### 3. **Baseline 2: RL-Based Energy Routing (RL-ER)**
- **Paper**: "Reinforcement Learning and Energy-Aware Routing"
- **Description**: Basic Q-learning without clustering
- **Key Features**:
  - Tabular Q-learning
  - Global state representation
  - No network decomposition
- **Implementation**: `baselines/rl_energy_routing.py`

---

## Experimental Setup

### Network Topologies

We evaluate all methods on **5 different network scales**:

| Topology | Nodes | Links | Hosts | Description |
|----------|-------|-------|-------|-------------|
| **Tiny** | 20 | 20 | 8 | Small enterprise network |
| **Small** | 50 | 100 | 20 | Campus network |
| **Medium** | 100 | 500 | 40 | Regional ISP |
| **Large** | 150 | 1000 | 60 | Metropolitan network |
| **X-Large** | 200 | 2000 | 80 | Large-scale datacenter |

### Evaluation Metrics

1. **Energy Saving (%)**
   - Percentage of links deactivated compared to all-on baseline
   - Formula: `(links_off / total_links) × 100`

2. **Latency (ms)**
   - Average end-to-end latency for all flows
   - Compared to baseline (all links active)

3. **SLA Violations (%)**
   - Percentage of flows exceeding latency threshold
   - Formula: `(violated_flows / total_flows) × 100`

4. **Computation Time (seconds)**
   - Time to compute link activation decisions (how much time it cost in one action)
   - Measured per episode/iteration

5. **Throughput**
   - Successfully routed flows per second
   - Network capacity utilization

---

## Directory Structure

```
experiment/
├── README.md                          # This file
├── baselines/
│   ├── __init__.py
│   ├── energy_aware_routing.py       # EAR implementation
│   ├── rl_energy_routing.py          # RL-ER implementation
│   └── baseline_utils.py             # Shared utilities
├── configs/
│   ├── topology_20.json              # 20 links config
│   ├── topology_100.json             # 100 links config
│   ├── topology_500.json             # 500 links config
│   ├── topology_1000.json            # 1000 links config
│   └── topology_2000.json            # 2000 links config
├── run_experiments.py                # Main experiment runner
├── analyze_results.py                # Results analysis script
└── results/
    ├── raw/                          # Raw experiment data
    ├── plots/                        # Generated plots
    └── comparison_table.csv          # Summary table
```

---

## Quick Start

### 1. Run All Experiments

```bash
cd experiment

# Run complete comparison (all methods, all topologies)
python run_experiments.py --all

# Run specific method
python run_experiments.py --method dqn_clustering
python run_experiments.py --method energy_aware
python run_experiments.py --method rl_basic

# Run specific topology
python run_experiments.py --topology 500
```

### 2. Analyze Results

```bash
# Generate comparison plots and tables
python analyze_results.py

# Generate LaTeX table for paper
python analyze_results.py --latex
```

---

## Running Individual Experiments

### Example 1: Compare on 500-link topology

```bash
# Run our method
python run_experiments.py --method dqn_clustering --topology 500 --episodes 2000

# Run EAR baseline
python run_experiments.py --method energy_aware --topology 500 --episodes 100

# Run RL-ER baseline
python run_experiments.py --method rl_basic --topology 500 --episodes 1000
```

### Example 2: Quick test on small topology

```bash
# Fast comparison on 20-link network
python run_experiments.py --topology 20 --episodes 100 --all
```

---

## Expected Results

### Hypothesis

| Metric | Our Method (DQN+Clustering) | EAR (Heuristic) | RL-ER (Basic RL) |
|--------|----------------------------|-----------------|------------------|
| **Energy Saving** | 70-80% ✅ | 50-60% | 60-70% |
| **Latency** | Low (10-15ms) ✅ | Medium (15-25ms) | Low (10-20ms) |
| **SLA Violations** | 5-10% ✅ | 15-25% | 10-15% |
| **Computation Time** | Medium (0.1-1s) | Fast (<0.01s) ✅ | Slow (1-10s) |
| **Scalability** | Good ✅ | Excellent ✅ | Poor |

---

## Next Steps

1. ✅ Implement baseline algorithms (`baselines/`)
2. ✅ Create topology configs (`configs/`)
3. ✅ Implement experiment runner (`run_experiments.py`)
4. ✅ Run experiments
5. ✅ Analyze results
6. ✅ Generate plots for paper

---

**Last Updated**: 2025-10-06

