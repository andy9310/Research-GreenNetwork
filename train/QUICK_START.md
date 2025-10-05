# Quick Start: Adaptive Clustering

## What Changed?

Your clustering system now supports **adaptive/dynamic cluster selection** instead of fixed k=3. The system can automatically determine the optimal number of clusters based on:
- Network topology (degree, betweenness centrality)
- Traffic patterns (incoming/outgoing flows)
- Service class distribution (priority levels)

## Quick Configuration

### Option 1: Use DP-means Adaptive (BEST FOR SDN) ⭐

Edit `config.json`:
```json
{
  "adaptive_clustering": true,
  "clustering_method": "dp_means_adaptive"
}
```

### Option 2: Use DP-means (Manual λ)

```json
{
  "adaptive_clustering": true,
  "clustering_method": "dp_means",
  "dp_means_lambda": 0.5
}
```

### Option 3: Keep Fixed Clustering (Baseline)

```json
{
  "adaptive_clustering": false,
  "num_clusters": 3
}
```

## Available Methods

| Method | Speed | Best For |
|--------|-------|----------|
| `dp_means_adaptive` | Medium | **BEST: Dynamic SDN, auto-optimized** |
| `dp_means` | Fast | Dynamic SDN, manual tuning |
| `network_heuristic` | **Fastest** | Production SDN, large scale |
| `silhouette` | Medium | General purpose, reliable |
| `elbow` | Fast | Quick deployment |
| `dbscan` | Fast | Irregular cluster shapes |
| `hierarchical` | Slow | Research analysis |

## Test It Out

Run the test script to compare methods:

```bash
cd Research-GreenNetwork/train
python test_clustering.py
```

This will show you:
- Number of clusters each method produces
- Cluster size distribution
- Performance comparison
- Recommendations for your network

## Training Example

```python
import json
from env import SDNEnv

# Load config with adaptive clustering
with open("config.json") as f:
    config = json.load(f)["config"]

# Create environment
env = SDNEnv(config)
obs = env.reset()

# Check how many clusters were created
print(f"Actual clusters: {env._actual_num_clusters}")

# Normal training loop
for episode in range(10):
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # Your policy here
        obs, reward, done, info = env.step(action)
        
        # Monitor cluster count (changes every recluster_every_steps)
        if env._time % env.recluster_every == 0:
            print(f"Reclustered: {env._actual_num_clusters} clusters")
```

## Comparing Methods for Your Paper

For your research comparison with ILP/MIP, Heuristic, and DQN:

**Experiment 1: Baseline**
```json
{"adaptive_clustering": false, "num_clusters": 3}
```

**Experiment 2: DP-means Adaptive (RECOMMENDED)**
```json
{"adaptive_clustering": true, "clustering_method": "dp_means_adaptive"}
```

**Experiment 3: DP-means (Manual λ)**
```json
{"adaptive_clustering": true, "clustering_method": "dp_means", "dp_means_lambda": 0.5}
```

**Experiment 4: Network Heuristic**
```json
{"adaptive_clustering": true, "clustering_method": "network_heuristic", "clustering_k_range": [3, 12]}
```

Then compare:
- Energy efficiency
- Latency/SLA violations  
- Computational time
- Adaptability to traffic changes

## Files Modified

1. **`cluster.py`**: Added 5 adaptive clustering methods
2. **`env.py`**: Updated to support adaptive clustering
3. **`config.json`**: Added clustering parameters
4. **`requirements.txt`**: Added scikit-learn and scipy

## New Files

1. **`CLUSTERING_GUIDE.md`**: Detailed documentation
2. **`test_clustering.py`**: Comparison script
3. **`QUICK_START.md`**: This file

## Troubleshooting

**Issue**: Too many clusters (>15)
- **Solution**: Reduce `clustering_k_range[1]` or use `network_heuristic`

**Issue**: Clustering is slow
- **Solution**: Use `network_heuristic` or `elbow` method

**Issue**: Unstable cluster count
- **Solution**: Increase `recluster_every_steps` in config

**Issue**: ImportError for sklearn/scipy
- **Solution**: `pip install -r requirements.txt`

## Next Steps

1. ✅ Run `python test_clustering.py` to see method comparison
2. ✅ Choose a clustering method based on results
3. ✅ Update `config.json` with your chosen method
4. ✅ Run training: `python train.py`
5. ✅ Compare results with fixed k=3 baseline

## References

- **Full Documentation**: See `CLUSTERING_GUIDE.md`
- **Original Code**: `cluster.py`, `env.py`
- **Config**: `config.json`

---

**Recommendation**: Start with `"dp_means_adaptive"` for best results, or `"network_heuristic"` for speed. 