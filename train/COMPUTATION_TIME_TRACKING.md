# Computation Time Tracking Implementation

## Overview
This document explains how computation time (action execution time) is tracked and plotted in the comparison experiments.

## What is Computation Time?

**Computation Time** measures the time taken to execute one complete action in the environment, from the moment `env.step(action)` is called until it returns. This includes:

1. **Action Decoding** - Converting the action index to thresholds and parameters
2. **Link Deactivation** - Applying the deactivation algorithm to turn off links
3. **Clustering Operations** - If using clustering methods, this includes:
   - Running the clustering algorithm (DP-means, Silhouette, etc.)
   - Assigning nodes to clusters
   - Reclustering when triggered
4. **Flow Generation** - Creating new network flows
5. **Routing & Measurement** - Computing paths, latency, and SLA violations
6. **Energy Calculation** - Computing energy consumption

## Implementation Details

### 1. Training Script (`train.py`)

**Added computation time tracking:**
```python
# Measure computation time for the action (including clustering)
action_start_time = time.time()
obs2, r, done, info = env.step(a)
action_end_time = time.time()
action_computation_time = (action_end_time - action_start_time) * 1000  # Convert to milliseconds
step_computation_times.append(action_computation_time)
```

**Updated TrainingVisualizer class:**
- Added `episode_computation_times` list to store per-episode average computation times
- Updated `log_episode()` to accept `computation_time` parameter
- Updated `save_metrics()` to include `computation_time` column in CSV

**Episode-level aggregation:**
```python
# Calculate average computation time per step
avg_computation_time = np.mean(step_computation_times) if step_computation_times else 0.0
```

### 2. Comparison Plot (`run_comparison.py`)

**Updated metrics to show only 4 key metrics:**
1. **Energy Saving (%)** - Percentage of energy saved compared to all links on
2. **Latency (ms)** - Average network latency in milliseconds
3. **SLA Violations (%)** - Percentage of flows violating SLA requirements
4. **Computation Time (ms)** - Average time to execute one action (in milliseconds)

**Plot layout:**
- Changed from 2x3 to 2x2 grid (4 metrics only)
- Uses actual data from `training_metrics.csv`
- Shows 3 decimal places for precision

### 3. CSV Output Format

The `training_metrics.csv` now includes:
```csv
episode,reward,loss,energy_saving,latency,sla_violations,active_links,cluster_count,computation_time
1,-1658.59,0.0,0.8253,20.01,37.28,152,4,0.523
2,-1617.58,2.306,0.8252,20.68,52.97,155,2,0.487
...
```

## Why Computation Time Matters

### No-Clustering Mode
- **Lower computation time** - No clustering algorithm overhead
- Single cluster means simpler action space
- Faster decision-making but potentially less energy efficient

### With-Clustering Mode
- **Higher computation time** - Includes clustering algorithm execution
- DP-means adaptive clustering adds overhead during reclustering events
- More complex decision-making but potentially better energy savings

## Expected Results

**Typical computation time ranges:**

| Method | Computation Time | Notes |
|--------|-----------------|-------|
| No Clustering | 0.1 - 0.5 ms | Minimal overhead, single cluster |
| Fixed k=3 Clustering | 0.3 - 1.0 ms | Fixed clustering, no adaptation |
| DP-means Adaptive | 0.5 - 2.0 ms | Higher during reclustering events |
| Silhouette Clustering | 0.8 - 3.0 ms | Most expensive due to score calculation |

**Note:** Actual times depend on:
- Network size (number of nodes/edges)
- Number of active flows
- Hardware performance
- Clustering frequency (reclustering interval)

## Usage

### Running Comparison with Computation Time Tracking

```bash
# Run comparison experiment
python run_comparison.py config.json low 50

# Results will be saved to:
# - comparison_results_temp/no_clustering_metrics.csv
# - comparison_results_temp/with_clustering_metrics.csv
# - clustering_comparison.png (plot with 4 metrics)
```

### Analyzing Computation Time

The computation time in the CSV represents the **average per-step computation time** for each episode. To analyze:

1. **Check CSV files** - Look at the `computation_time` column
2. **Compare methods** - No-clustering should be faster than clustering methods
3. **Identify bottlenecks** - High computation times may indicate:
   - Frequent reclustering
   - Large number of flows
   - Complex routing calculations

## Technical Notes

- **Time unit**: Milliseconds (ms)
- **Measurement**: Wall-clock time using `time.time()`
- **Scope**: Only measures `env.step()` execution, not agent decision time
- **Precision**: 3 decimal places in plots (microsecond precision)

## Future Improvements

Potential enhancements:
1. Break down computation time by component (clustering, routing, etc.)
2. Track peak computation times vs average
3. Measure agent inference time separately
4. Add computation time to reward function for efficiency optimization
