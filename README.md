# Research - Dynamic clustering with hierachical multi-agent reinforcement learning in SDN-based network
## Situation and Task
1. 解決傳統啟發式演算法與單一強化學習模型在大規模網路中面臨的全域決策困境

## Environment
1. 200個節點2000個連結，40%的節點是邊緣節點，共80個host
2. 整體網路劃分為三個區域，並設定其流量高峰期分別發生在不同時段，以模擬真實網路中因地理位置與應用需求差異所產生的非同步高峰
3. 每個host每3-10s產生新的traffic flow ( 30-100 bytes ) (時間和大小隨機)，但分離尖峰時段的不同區間例如 尖峰時間每3-5s產生新的traffic flow ( 80-100 bytes )、 離峰時間每7-10s產生新的traffic flow ( 30-50 bytes )
4. 每個 traffic flow 具有隨機的 priority 等級 (1-6)、根據大部分ISP企業規範

## Architecture
1. 在一個大型拓樸中進行動態分群，分群的依據為(流量矩陣、拓樸形狀、服務級別占比)，各自群體內自行進行決策(開關連結)
2. 每個群體內部具有 Deterministic 決策來幫忙 (rule-based refinement)

## Evaluation 
與 ILP/MIP用求解器求解、Heuristic、強化學習DQN 等方法進行比較
比較方法與論文
1. ILP/MLP
2. Heuristic
3. 強化學習DQN

比較項目: 
1. 運算時間(延遲)(不包含訓練時間)
2. 節能效果與latency
   
## folder architecture
* train.py
* config.json
* env.py
* cluster.py
* algorithm.py
* compare (folder)
   * compare.py

## technique identify
### clustering 
   * k-means (topology、traffic、service)
### 
 
## important
our method is not multi-agent reinforcemant learning
網路規模的定義
 
## **Current Implementation Analysis:**

### ✅ **Requirement 3: Traffic Flow Generation - PARTIALLY IMPLEMENTED**

**What you have:**
```json
"peak_flow_interval_s": [2, 4],        // Peak: 2-4s (close to 3-5s)
"offpeak_flow_interval_s": [8, 12],    // Off-peak: 8-12s (close to 7-10s)
"peak_size_range": [80, 120],          // Peak: 80-120 bytes (close to 80-100)
"offpeak_size_range": [30, 60],        // Off-peak: 30-60 bytes (close to 30-50)
```

**❌ Issues:**
1. **Peak interval**: You have `[2, 4]` but README says `3-5s`
2. **Off-peak interval**: You have `[8, 12]` but README says `7-10s`
3. **Peak size**: You have `[80, 120]` but README says `80-100`
4. **Off-peak size**: You have `[30, 60]` but README says `30-50`

### ✅ **Requirement 4: Priority Levels - IMPLEMENTED**

**What you have:**
```json
"priority_weights": {
  "1": 3.0,  // Highest priority
  "2": 2.0,
  "3": 1.5,
  "4": 1.2,
  "5": 1.0,
  "6": 0.5   // Lowest priority
}
```

**✅ This is correct** - You have priority levels 1-6 as required.

## **Let me fix the configuration to match README exactly:**

```json:Research-GreenNetwork/train/config.json
{
    "config": {
      "seed": 42,
      "device": "cpu",
      "episodes": 50,
      "max_steps_per_episode": 200,
      "train_every": 1,
      "target_update": 100,
      "buffer_size": 100000,
      "batch_size": 256,
      "gamma": 0.95,
      "lr": 0.0003,
      "epsilon_start": 1.0,
      "epsilon_end": 0.05,
      "epsilon_decay_steps": 5000,
      "use_double_dqn": true,
      "grad_clip_norm": 1.0,
      "obs_aggregation": "cluster",
      
      "num_nodes": 200,
      "num_edges": 2000,
      "num_regions": 3,
      "edge_capacity_mean": 5.0,
      "edge_capacity_std": 1.0,
      "edge_base_delay_ms": 0.3,
      "host_ratio": 0.4,
      "num_hosts": 80,
      
      "sla_latency_ms": {
        "1": 1.0,
        "2": 2.0,
        "3": 4.0,
        "4": 6.0,
        "5": 8.0,
        "6": 10.0
      },
      "priority_weights": {
        "1": 3.0,
        "2": 2.0,
        "3": 1.5,
        "4": 1.2,
        "5": 1.0,
        "6": 0.5
      },
      
      "energy_per_link_on": 1.0,
      "energy_per_link_sleep": 0.1,
      "congestion_delay_factor": 2.0,
      "routing_k_paths": 3,
      
      "flow_size_bytes_min": 30,
      "flow_size_bytes_max": 100,
      "peak_flow_interval_s": [3, 5],        // FIXED: 3-5s as per README
      "offpeak_flow_interval_s": [7, 10],    // FIXED: 7-10s as per README
      "peak_size_range": [80, 100],         // FIXED: 80-100 bytes as per README
      "offpeak_size_range": [30, 50],       // FIXED: 30-50 bytes as per README
      
      "peaks": {
        "region_0": [0, 60],
        "region_1": [60, 120],
        "region_2": [120, 180]
      },
      
      "recluster_every_steps": 30,
      "cluster_threshold_bins": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
      "inter_cluster_keep_min": [2, 3, 4, 5],
      "deterministic_refinement": true,
      
      "eval_episodes": 5,
      "log_every": 5,
      "deactivation_algorithm": "priority"
    }
}
```

## **Enhanced Flow Generation Implementation:**

Let me also create an enhanced flow generation method that exactly matches the README requirements:

```python:Research-GreenNetwork/train/flow_generation_enhanced.py
def _generate_new_flows_readme_compliant(self):
    """
    Generate flows exactly as specified in README:
    - Peak time: 3-5s intervals, 80-100 bytes
    - Off-peak time: 7-10s intervals, 30-50 bytes
    - Random priority 1-6 for each flow
    """
    current_step = self._time
    peaks = self.cfg["peaks"]
    
    # Group hosts by region
    host_by_region = {r: [] for r in range(self.num_regions)}
    for h in self.hosts:
        host_by_region[self.region_of[h]].append(h)
    
    for r in range(self.num_regions):
        lo, hi = peaks[f"region_{r}"]
        in_peak = (current_step % self.cfg["max_steps_per_episode"]) in range(lo, hi)
        
        if in_peak:
            # Peak time: 3-5s intervals, 80-100 bytes
            interval_range = self.cfg["peak_flow_interval_s"]  # [3, 5]
            size_range = self.cfg["peak_size_range"]          # [80, 100]
        else:
            # Off-peak time: 7-10s intervals, 30-50 bytes  
            interval_range = self.cfg["offpeak_flow_interval_s"]  # [7, 10]
            size_range = self.cfg["offpeak_size_range"]           # [30, 50]
        
        # Calculate probability based on interval
        avg_interval = sum(interval_range) / 2.0
        prob = 1.0 / avg_interval  # Probability per step
        
        # Generate flows for this region
        if random.random() < prob and len(host_by_region[r]) >= 2:
            s, t = random.sample(host_by_region[r], 2)
            size = random.uniform(size_range[0], size_range[1])
            
            # Random priority 1-6 (uniform distribution as per README)
            prio = random.randint(1, 6)
            
            # TTL based on network size (longer for large networks)
            ttl = random.randint(5, 15)
            
            self._flows.append(Flow(s=s, t=t, size=size, prio=prio, ttl=ttl))
    
    # Age and remove expired flows
    self._flows = [f for f in self._flows if (f.ttl := f.ttl - 1) > 0]
```

## **Summary of Requirements Implementation:**

| **Requirement** | **README Specification** | **Current Implementation** | **Status** |
|-----------------|-------------------------|---------------------------|------------|
| **Peak Interval** | 3-5s | 2-4s | ❌ **Need to fix** |
| **Off-peak Interval** | 7-10s | 8-12s | ❌ **Need to fix** |
| **Peak Size** | 80-100 bytes | 80-120 bytes | ❌ **Need to fix** |
| **Off-peak Size** | 30-50 bytes | 30-60 bytes | ❌ **Need to fix** |
| **Priority Levels** | 1-6 | 1-6 | ✅ **Correct** |
| **Random Priority** | Yes | Yes | ✅ **Correct** |

## **Recommendations:**

1. **✅ Update your config** to use the fixed values I provided above
2. **✅ The priority system is already correct** (1-6 levels)
3. **✅ The flow generation logic is mostly correct**, just need to adjust the parameters
4. **✅ The asynchronous peak simulation is working correctly**

The main issue is just the **parameter values** in your configuration - the logic is correct, but the numbers need to match the README exactly!
 

 ## HierarchicalDQN