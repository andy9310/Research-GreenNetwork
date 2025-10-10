# Standalone Baseline Tests

This directory contains standalone test scripts to run individual baseline algorithms and get their performance metrics.

## Prerequisites

Configuration files must exist in `configs/` directory:
- `configs/topology_20.json` - 20 links network
- `configs/topology_100.json` - 100 links network
- `configs/topology_500.json` - 500 links network
- `configs/topology_1000.json` - 1000 links network
- `configs/topology_2000.json` - 2000 links network

These files are pre-created and ready to use.

## Available Scripts

### 1. Energy-Aware Routing (EAR)
**File**: `test_ear_standalone.py`

**Quick Run**:
```bash
# Default: 100 links, 100 episodes
python test_ear_standalone.py

# Custom configuration
python test_ear_standalone.py --links 500 --episodes 200 --threshold 0.3 --min-active 0.2
```

**Arguments**:
- `--links`: Number of links (20, 100, 500, 1000, 2000) [default: 100]
- `--episodes`: Number of test episodes [default: 100]
- `--threshold`: Utilization threshold for link deactivation [default: 0.3]
- `--min-active`: Minimum fraction of links to keep active [default: 0.2]

**Output Metrics**:
- ✅ **Energy Saving (%)**: Percentage of energy saved
- ✅ **Latency (ms)**: Average end-to-end latency
- ✅ **SLA Violation Rate (%)**: Percentage of flows violating SLA
- ✅ **Computation Time (s)**: Average time per decision

---

### 2. RL-Based Energy Routing (RL-ER)
**File**: `test_rler_standalone.py`

**Quick Run**:
```bash
# Default: 100 links, 500 episodes
python test_rler_standalone.py

# Custom configuration
python test_rler_standalone.py --links 500 --episodes 1000 --lr 0.1 --gamma 0.95 --epsilon 0.1
```

**Arguments**:
- `--links`: Number of links (20, 100, 500, 1000, 2000) [default: 100]
- `--episodes`: Number of training episodes [default: 500]
- `--lr`: Learning rate (alpha) [default: 0.1]
- `--gamma`: Discount factor (gamma) [default: 0.95]
- `--epsilon`: Exploration rate [default: 0.1]

**Output Metrics**:
- ✅ **Reward**: Average cumulative reward
- ✅ **Energy Saving (%)**: Percentage of energy saved
- ✅ **Latency (ms)**: Average end-to-end latency
- ✅ **SLA Violation Rate (%)**: Percentage of flows violating SLA
- ✅ **Computation Time (s)**: Average time per decision
- **Q-Table Size**: Number of learned states

---

## Examples

### Test EAR on small network (quick test)
```bash
python test_ear_standalone.py --links 20 --episodes 50
```

### Test EAR on medium network
```bash
python test_ear_standalone.py --links 500 --episodes 100
```

### Test RL-ER on small network (quick test)
```bash
python test_rler_standalone.py --links 20 --episodes 200
```

### Test RL-ER on medium network (longer training)
```bash
python test_rler_standalone.py --links 500 --episodes 1000
```

---

## Output

### Console Output
Both scripts provide:
1. **Progress updates** during execution
2. **Final metrics** with mean ± standard deviation
3. **Summary statistics** (computation time, active links, etc.)

### Saved Results
Results are automatically saved to JSON files:
- `results/ear_standalone_{n_links}links.json`
- `results/rler_standalone_{n_links}links.json`

---

## Expected Performance

Based on the README.md hypothesis:

| Metric | EAR (Heuristic) | RL-ER (Basic RL) |
|--------|-----------------|------------------|
| **Energy Saving** | 50-60% | 60-70% |
| **Latency** | 15-25ms | 10-20ms |
| **SLA Violations** | 15-25% | 10-15% |
| **Computation Time** | <0.01s ✅ | 1-10s |
| **Scalability** | Excellent ✅ | Poor |

---

## Comparison with Full Experiment

These standalone scripts are useful for:
- ✅ **Quick testing** of individual algorithms
- ✅ **Parameter tuning** (threshold, learning rate, etc.)
- ✅ **Debugging** algorithm behavior
- ✅ **Getting baseline metrics** without running full comparison

For complete comparison across all methods, use:
```bash
python run_experiments.py --all
```

---

## Notes

1. **EAR** is deterministic and converges quickly (100 episodes sufficient)
2. **RL-ER** requires training and more episodes (500-1000 recommended)
3. Results are averaged over the **last 20% of episodes** for stability
4. Larger networks require more episodes for RL-ER to converge
5. Results are saved automatically to `results/` directory
