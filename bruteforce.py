import numpy as np
import networkx as nx
from env import NetworkEnv # Reuse the environment definition
import itertools
import time
import json # Import json

print("Starting Brute-Force Search for Optimal Configuration...")

# --- Load Configuration from JSON ---
def load_config(config_path="config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

config = load_config()

# --- Environment Setup (Load from config) ---
num_nodes = config["num_nodes"]
adj_matrix = config["adj_matrix"]
edge_list = config["edge_list"]
node_props = config["node_props"]
tm_list = config["tm_list"]
link_capacity = config["link_capacity"]
# energy_unit_reward is loaded within the env, but we can get it if needed
# energy_unit_reward = config["energy_unit_reward"]

# --- Create a temporary environment instance --- 
env = NetworkEnv(
    adj_matrix=adj_matrix,
    edge_list=edge_list,
    tm_list=tm_list, 
    node_props=node_props,
    num_nodes=num_nodes,
    link_capacity=link_capacity,
    seed=int(time.time()) # Seed doesn't matter for brute force logic
)
# Ensure the traffic matrix is set and is a numpy array for checks
env.traffic = np.array(env.tm_list[0])

num_edges = env.num_edges
print(f"Network has {num_nodes} nodes and {num_edges} edges.")
print(f"Checking {2**num_edges} possible link configurations...")

# --- Brute-Force Search --- 
best_config = None
best_score = -float('inf') # Initialize with a very low score
checked_configs = 0
valid_configs_found = 0

# Generate all possible binary configurations (0=closed, 1=open)
# Itertools.product([0, 1], repeat=num_edges) generates tuples, convert to numpy array
for config_tuple in itertools.product([0, 1], repeat=num_edges):
    config_vector = np.array(config_tuple, dtype=int)
    checked_configs += 1

    # Manually set the link status in the environment
    env.link_open = config_vector

    # Manually call the internal checks from the environment
    routing_successful, G_open = env._update_link_usage() # Calculates usage based on env.link_open
    isolated, overloaded, num_overloaded = env._check_violations(routing_successful, G_open)

    # Check if the configuration is valid (no violations)
    if not isolated and not overloaded:
        valid_configs_found += 1
        # Calculate score: Energy savings from closed links
        num_closed_links = np.sum(config_vector == 0)
        # Use the same reward unit as the environment for direct comparison
        current_score = num_closed_links * env.energy_unit_reward

        # Update best score if this config is better
        if current_score > best_score:
            best_score = current_score
            best_config = config_vector

    if checked_configs % 100 == 0: # Progress indicator for larger networks
         print(f"... checked {checked_configs}/{2**num_edges} configurations ...")


print(f"\nFinished checking {checked_configs} configurations.")
print(f"Found {valid_configs_found} valid configurations (no violations).")

# --- Report Results --- 
if best_config is not None:
    print("\n--- Optimal Configuration Found (Brute-Force) --- ")
    print(f" Best Configuration (0=closed, 1=open): {best_config}")
    print(f" Number of Links Closed: {np.sum(best_config == 0)} / {num_edges}")
    print(f" Maximum Achievable Score (Energy Saved Reward): {best_score}")
    # You could optionally recalculate usage/etc here for more details if needed
else:
    print("\nNo valid configuration found that satisfies the constraints.")
    print("(This might indicate an issue with the topology, TM, or capacity definitions)")
