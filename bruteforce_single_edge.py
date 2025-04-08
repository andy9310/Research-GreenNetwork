import numpy as np
import networkx as nx
from env import NetworkEnv
import time
import json
from tqdm import tqdm
import argparse

# --- Load Configuration from JSON ---
def load_config(config_path="config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Single-edge bruteforce search for network optimization')
parser.add_argument('--config', type=str, default='config.json', help='Path to configuration JSON file')
parser.add_argument('--tm-index', type=int, default=0, help='Index of traffic matrix to use from tm_list (default: 0)')
args = parser.parse_args()

# Load config from specified file
config_path = args.config
print(f"Starting Single-Edge Optimization Search using {config_path}...")
config = load_config(config_path)

# --- Environment Setup (Load from config) ---
num_nodes = config["num_nodes"]
adj_matrix = config["adj_matrix"]
edge_list = config["edge_list"]
node_props = config["node_props"]
tm_list = config["tm_list"]
link_capacity = config["link_capacity"]
max_edges = config["max_edges"]
energy_unit_reward = config["energy_unit_reward"]

# --- Create environment instance --- 
env = NetworkEnv(
    adj_matrix=adj_matrix,
    edge_list=edge_list,
    tm_list=tm_list,
    node_props=node_props,
    num_nodes=num_nodes,
    link_capacity=link_capacity,
    max_edges=max_edges,
    seed=int(time.time())
)

# Set the specified traffic matrix index
tm_index = args.tm_index
print(tm_index)
# Validate the tm_index
if tm_index < 0 or tm_index >= len(tm_list):
    print(f"Error: Traffic matrix index {tm_index} is out of range (0-{len(tm_list)-1})")
    print(f"Defaulting to index 0")
    tm_index = 0

env.current_tm_idx = tm_index
print(f"Using traffic matrix index {tm_index} (of {len(tm_list)} matrices) for single-edge optimization")

# Ensure the traffic matrix is set
env.traffic = np.array(env.tm_list[tm_index])

num_edges = env.num_edges
print(f"Network has {num_nodes} nodes and {num_edges} edges.")
print(f"Checking {num_edges} possible configurations (closing one edge at a time)...")

# --- Single-Edge Closure Search --- 
valid_closures = []
rewards = []

# Start with all links open
base_config = np.ones(num_edges, dtype=int)

# Try closing one edge at a time
with tqdm(total=num_edges, desc="Testing Single-Edge Closures", unit="edge") as pbar:
    for edge_idx in range(num_edges):
        # Create configuration with just this edge closed
        test_config = base_config.copy()
        test_config[edge_idx] = 0  # Close this edge
        
        # Set the link status in the environment
        env.link_open = test_config
        
        # Check if this configuration violates any constraints
        routing_successful, G_open = env._update_link_usage()
        isolated, overloaded, num_overloaded = env._check_violations(routing_successful, G_open)
        
        # If valid (no violations), record this edge as a candidate for closure
        if not isolated and not overloaded:
            closed_edge = edge_list[edge_idx]
            valid_closures.append(edge_idx)
            rewards.append(energy_unit_reward)  # Reward is fixed at energy_unit_reward
            pbar.set_postfix({"Valid Closures": len(valid_closures)})
        
        pbar.update(1)

# Print results
print(f"\nResults: Found {len(valid_closures)} edges that can be safely closed")

if valid_closures:
    print("\n--- Safe Edges to Close ---")
    print("Edge Index | Edge (Nodes) | Energy Saving")
    print("-" * 40)
    
    for i, edge_idx in enumerate(valid_closures):
        edge = edge_list[edge_idx]
        print(f"{edge_idx:10} | ({edge[0]},{edge[1]})    | {rewards[i]}")
    
    print("\nAny of these edges can be closed individually without violating network constraints.")
    print(f"Each closure provides a reward of {energy_unit_reward} energy units.")
    
    # Demonstrate using the first valid edge as an example
    if valid_closures:
        example_edge_idx = valid_closures[0]
        example_edge = edge_list[example_edge_idx]
        print(f"\nFor example, you could safely close edge {example_edge_idx} between nodes {example_edge[0]} and {example_edge[1]}")
else:
    print("\nNo edges can be safely closed in this network configuration.")
    print("The network is likely operating at minimal connectivity already.")
