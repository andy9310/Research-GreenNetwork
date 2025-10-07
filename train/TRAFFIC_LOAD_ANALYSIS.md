# Traffic Load Analysis - Critical Issues Found

## Problem Summary

You correctly identified that **82.5% energy saving with high traffic load is unrealistic**. After analyzing the code, I found **multiple critical issues**:

---

## Issue 1: Energy Saving Calculation is Misleading ⚠️

### Current Implementation (env.py, line 248-249):
```python
base_all_on = self.energy_on * self.G_full.number_of_edges()  # Total edges = 2000
energy_saving = (base_all_on - energy) / base_all_on
```

### The Problem:
**Energy saving is calculated against ALL 2000 edges, but the network only needs ~150-160 active links to function!**

### Example:
- Total edges: 2000
- Active links needed: 155
- Energy baseline (all on): 2000 * 1.0 = 2000
- Energy used: 155 * 1.0 + 1845 * 0.1 = 339.5
- **Energy saving: (2000 - 339.5) / 2000 = 83%** ✅ Matches your output!

### Why This is Wrong:
The network is **highly redundant** by design. You can't turn off 90% of links and still route traffic - the graph would be disconnected! The baseline should be the **minimum links needed for connectivity**, not all 2000 edges.

---

## Issue 2: Utilization Calculation Caps at 100% 🚨

### Current Implementation (env.py, lines 619-622):
```python
# Utilization is already a fraction (0-1), convert to percentage
# Cap at 100% for display (overload is shown separately)
avg_util = min(100.0, (total_util / max(1, active_edges)) * 100)
max_util_pct = min(100.0, max_util * 100)
```

### The Problem:
**The code CAPS utilization at 100% for display!** This means:
- If actual utilization is 150% (overloaded), it shows as 100%
- If actual utilization is 200% (severely overloaded), it shows as 100%
- You see "99.1% 🔥" but the real utilization could be 300%!

### Evidence from Your Output:
```
Flows: 364 | Util: 99.1% 🔥
```
The 🔥 emoji indicates overload (line 631: `"overloaded": max_util > 1.0`), but the display is capped.

---

## Issue 3: Flow Size Calculation is Flawed 🔧

### Current Implementation (env.py, lines 561-565):
```python
expected_concurrent_flows = len(self._flows) / max(1, self.G_full.number_of_edges())
target_util_per_flow = np.random.uniform(target_min, target_max) / max(1, expected_concurrent_flows)
base_size = target_util_per_flow * self.avg_edge_capacity * 1000
```

### The Problem:
1. **Divides by total edges (2000), not active edges (~155)**
   - This makes flows WAY too small
   - Expected flows per edge = 364 / 2000 = 0.18 flows/edge
   - Should be: 364 / 155 = 2.35 flows/edge

2. **Uses `len(self._flows)` which includes old flows**
   - Flows have TTL 5-15 steps
   - Old flows accumulate, making calculation even worse

3. **Target utilization is divided by expected flows**
   - For high traffic: target = 0.6-0.8
   - Divided by 0.18 = 3.3-4.4 (makes flows HUGE)
   - But then divided by 2000 edges = tiny flows again

### Result:
Flow sizes don't actually achieve the target utilization because the calculation is based on wrong assumptions.

---

## Issue 4: High Traffic Config May Not Generate Enough Flows 📊

### Config (config.json):
```json
"high": {
  "target_utilization_range": [0.6, 0.8],
  "flow_intensity_multiplier": 2.5,
  "peak_flow_probability": 0.8,
  "offpeak_flow_probability": 0.4
}
```

### Flow Generation (env.py, lines 344-348):
```python
flow_prob = base_prob * self.flow_intensity_multiplier
num_potential_flows = max(1, int(flow_prob * len(host_by_region[r]) / 2))
```

### Calculation:
- Hosts per region: 80 / 3 = ~27 hosts
- Peak: flow_prob = 0.8 * 2.5 = 2.0
- num_potential_flows = int(2.0 * 27 / 2) = 27 flows per region
- Total: ~81 flows per step

But your output shows **364 flows** - this is because flows have TTL 5-15, so they accumulate over multiple steps.

---

## Issue 5: Overload Penalty Doesn't Prevent Overload ⚡

### Current Implementation (env.py, lines 261-263):
```python
if avg_util_pct > 100.0:
    overload_penalty = 10.0  # Massive penalty
```

### The Problem:
The penalty uses **avg_util_pct which is capped at 100%** (from Issue 2)!

So even if real utilization is 300%, the penalty calculation sees it as 100% and applies a fixed 10.0 penalty.

---

## Recommendations

### Fix 1: Correct Energy Saving Baseline
```python
# Use minimum spanning tree or actual minimum connectivity as baseline
min_edges_needed = self.n - 1  # At minimum, need n-1 edges for connectivity
# Or use a more realistic baseline like 30% of edges
realistic_baseline = self.G_full.number_of_edges() * 0.3
base_all_on = self.energy_on * realistic_baseline
energy_saving = (base_all_on - energy) / base_all_on
```

### Fix 2: Remove Utilization Cap for Calculations
```python
# For display, cap at 100%
avg_util_display = min(100.0, (total_util / max(1, active_edges)) * 100)

# For calculations (penalties, rewards), use ACTUAL utilization
avg_util_actual = (total_util / max(1, active_edges)) * 100  # No cap!
```

### Fix 3: Fix Flow Size Calculation
```python
# Use ACTIVE edges, not total edges
active_edges = sum(1 for u, v, d in self.G_full.edges(data=True) if d["active"] == 1)
expected_concurrent_flows = len(self._flows) / max(1, active_edges)

# Or better: calculate based on target total network load
target_total_load = target_util * active_edges * self.avg_edge_capacity
flow_size = target_total_load / max(1, expected_new_flows_per_step)
```

### Fix 4: Increase Flow Generation for High Traffic
```json
"high": {
  "target_utilization_range": [0.6, 0.8],
  "flow_intensity_multiplier": 4.0,  // Increase from 2.5
  "peak_flow_probability": 0.9,      // Increase from 0.8
  "offpeak_flow_probability": 0.6    // Increase from 0.4
}
```

### Fix 5: Use Actual Utilization for Penalties
```python
# Calculate actual utilization WITHOUT capping
actual_avg_util = (total_util / max(1, active_edges)) * 100

if actual_avg_util > 100.0:
    # Penalty based on ACTUAL overload amount
    excess = actual_avg_util - 100.0
    overload_penalty = 0.1 * (excess ** 2)  # Quadratic penalty
```

---

## Why You're Seeing These Results

Your output shows:
- **Energy Saving: 82.5%** - Misleading because baseline is 2000 edges, not realistic minimum
- **Util: 99.1% 🔥** - Capped display, actual utilization is likely 150-200%
- **Links: 154/2000** - Only 7.7% of edges active (this is actually reasonable!)
- **Flows: 364** - Accumulated flows with TTL, seems reasonable

The agent has learned to:
1. Turn off as many links as possible (good for energy)
2. Overload the remaining links (bad for network)
3. Get rewarded because energy saving is calculated wrong
4. Not penalized enough because utilization is capped

---

## Verification Steps

1. **Check actual utilization** - Remove the `min(100.0, ...)` cap and log real values
2. **Check flow sizes** - Log actual flow sizes and compare to edge capacities
3. **Check routing failures** - Count how many flows can't find paths
4. **Recalculate energy baseline** - Use realistic minimum connectivity

This explains why high traffic still shows 82% energy saving - the calculation is fundamentally flawed!
