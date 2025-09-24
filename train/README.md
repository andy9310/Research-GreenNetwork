This is a lightweight, Colab-friendly research scaffold for:
- **Dynamic clustering** of a large SDN topology (by topology/traffic/service mix).
- **Hierarchical multi-agent DQN** actioning cluster thresholds & inter-cluster budgets.
- **Rule-based refinement** to protect high-priority (EF/AF4) flows.
- **Baselines**: Heuristic thresholding + a simplified ILP (CBC) energy minimization.

## Files
- `config.json` – knobs for topology size, traffic peaks, RL hyper-params.
- `env.py` – SDN environment with flow generation, routing, latency & energy modeling.
- `cluster.py` – numpy K-means for dynamic clustering on graph features.
- `algorithm.py` – HierarchicalDQN + GlobalThresholdDQN baseline.
- `train.py` – trains HMARL.
- `compare/compare.py` – quick head-to-head comparison (HMARL vs Heuristic vs ILP).

## Quickstart (in Colab)
```bash
%cd /content
!pip -q install networkx pulp torch --progress-bar off
!unzip -o /content/sdn_hmarl_demo.zip -d sdn_hmarl_demo
%cd sdn_hmarl_demo
!python train.py
!python compare/compare.py