# Quick Start Guide

## Setup

1. **Install dependencies** (if not already installed):
```bash
pip install numpy networkx matplotlib pandas torch
```

2. **Navigate to experiment directory**:
```bash
cd train/experiment
```

## Running Experiments

### Option 1: Run All Experiments (Recommended for Paper)

```bash
# Run complete comparison (takes ~6-12 hours)
python run_experiments.py --all
```

This will:
- Test all 3 methods (DQN+Clustering, Energy-Aware, RL-Basic)
- On all 5 topologies (20, 100, 500, 1000, 2000 links)
- Save results to `results/` directory

### Option 2: Quick Test (5 minutes)

```bash
# Test on smallest topology only
python run_experiments.py --topology 20 --method dqn_clustering --episodes 100
python run_experiments.py --topology 20 --method energy_aware --episodes 50
python run_experiments.py --topology 20 --method rl_basic --episodes 100
```

### Option 3: Single Method on Single Topology

```bash
# Test your method on 500-link network
python run_experiments.py --method dqn_clustering --topology 500 --episodes 2000

# Test baseline on same topology
python run_experiments.py --method energy_aware --topology 500 --episodes 100
```

## Analyzing Results

### Generate All Plots

```bash
python analyze_results.py
```

This creates:
- `results/plots/energy_saving_comparison.png`
- `results/plots/latency_comparison.png`
- `results/plots/computation_time_scalability.png`
- `results/plots/energy_latency_tradeoff.png`

### Generate LaTeX Table for Paper

```bash
python analyze_results.py --latex
```

Copy the output directly into your paper!

## Expected Timeline

| Task | Time | Output |
|------|------|--------|
| **Quick test (20 links)** | 5-10 min | Verify setup works |
| **Single topology (500 links)** | 1-2 hours | One data point |
| **Full experiment (all)** | 6-12 hours | Complete comparison |
| **Analysis** | 1 min | All plots + tables |

## Troubleshooting

### Error: "No module named 'env'"
**Solution**: Make sure you're in the `experiment/` directory

### Error: "CUDA out of memory"
**Solution**: Edit configs to use `"device": "cpu"`

### Results look wrong
**Solution**: Check that:
1. All methods completed successfully
2. Episode counts are sufficient (DQN needs 2000+)
3. Results CSV exists in `results/comparison_table.csv`

## Next Steps

1. ✅ Run quick test to verify setup
2. ✅ Run full experiments (leave overnight)
3. ✅ Generate plots and tables
4. ✅ Include in your paper
5. ✅ Celebrate! 🎉

## File Structure After Running

```
experiment/
├── results/
│   ├── raw/
│   │   ├── dqn_clustering_20.json
│   │   ├── dqn_clustering_100.json
│   │   ├── ...
│   ├── plots/
│   │   ├── energy_saving_comparison.png
│   │   ├── latency_comparison.png
│   │   ├── computation_time_scalability.png
│   │   └── energy_latency_tradeoff.png
│   └── comparison_table.csv
```

## Tips for Paper

1. **Use the LaTeX table** - directly copy from `analyze_results.py --latex`
2. **Include all 4 plots** - they show different aspects
3. **Highlight your advantages**:
   - Better energy-latency trade-off
   - Good scalability
   - Lower SLA violations
4. **Discuss trade-offs**:
   - Slightly higher computation time than heuristic
   - But much better performance

Good luck with your research! 🚀
