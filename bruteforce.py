import numpy as np
import networkx as nx
from env import NetworkEnv # Reuse the environment definition
import itertools
import time
import json # Import json
from tqdm import tqdm # For progress bar
import argparse # For command-line arguments

# --- Load Configuration from JSON ---
def load_config(config_path="config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Brute-force search for optimal network configuration')
parser.add_argument('--config', type=str, default='config.json', help='Path to configuration JSON file')
parser.add_argument('--tm-index', type=int, default=0, help='Index of traffic matrix to use from tm_list (default: 0)')
args = parser.parse_args()

# Load config from specified file
config_path = args.config
print(f"Starting Brute-Force Search for Optimal Configuration using {config_path}...")
config = load_config(config_path)

# --- Environment Setup (Load from config) ---
num_nodes = config["num_nodes"]
adj_matrix = config["adj_matrix"]
edge_list = config["edge_list"]
node_props = config["node_props"]
tm_list = config["tm_list"]
link_capacity = config["link_capacity"]
max_edges = config["max_edges"]
# energy_unit_reward is loaded within the env, but we can get it if needed
# energy_unit_reward = config["energy_unit_reward"]

# --- Create a temporary environment instance --- 
env = NetworkEnv(
    adj_matrix=adj_matrix,
    edge_list=edge_list,
    tm_list=tm_list,  # We'll set current_tm_idx=0 below
    node_props=node_props,
    num_nodes=num_nodes,
    link_capacity=link_capacity,
    max_edges=max_edges, # Pass max_edges to env
    seed=int(time.time()) # Seed doesn't matter for brute force logic
)

# Set the specified traffic matrix index
tm_index = args.tm_index

# Validate the tm_index
if tm_index < 0 or tm_index >= len(tm_list):
    print(f"Error: Traffic matrix index {tm_index} is out of range (0-{len(tm_list)-1})")
    print(f"Defaulting to index 0")
    tm_index = 0

env.current_tm_idx = tm_index
print(f"Using traffic matrix index {tm_index} (of {len(tm_list)} matrices) for brute force search")

# Ensure the traffic matrix is set and is a numpy array for checks
env.traffic = np.array(env.tm_list[tm_index])

num_edges = env.num_edges
print(f"Network has {num_nodes} nodes and {num_edges} edges.")
print(f"Checking {2**num_edges} possible link configurations...")

# --- Brute-Force Search --- 
best_config = None
best_score = -float('inf') # Initialize with a very low score
valid_configs_found = 0

# Calculate total number of configurations
total_configs = 2**num_edges

# Limit the total number of configurations to check if it's too large
# For very large networks, we'll do sampling rather than exhaustive search
max_configs_to_check = 10000000  # 10 million is a reasonable limit
do_sampling = total_configs > max_configs_to_check

if do_sampling:
    print(f"Warning: Total configurations ({total_configs:,}) exceeds limit of {max_configs_to_check:,}")
    print(f"Will sample {max_configs_to_check:,} random configurations instead of exhaustive search")
    # Generate random configurations instead of exhaustive search
    all_configs = np.random.randint(0, 2, size=(max_configs_to_check, num_edges))
    total_configs = max_configs_to_check
else:
    # Generate all possible binary configurations (0=closed, 1=open)
    # Convert to a list that tqdm can count
    all_configs = list(itertools.product([0, 1], repeat=num_edges))

# Use tqdm for progress tracking
with tqdm(total=total_configs, desc="Checking Configurations", unit="config") as pbar:
    # Track best configuration seen so far
    for i, config_data in enumerate(all_configs):
        if do_sampling:
            # Already a numpy array from random generation
            config_vector = config_data
        else:
            # Convert tuple to numpy array
            config_vector = np.array(config_data, dtype=int)
        
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
                # Update progress bar description with current best
                pbar.set_postfix({"Valid Found": valid_configs_found, "Best Score": best_score})
        
        # Update progress bar
        pbar.update(1)


print(f"\nFinished checking {total_configs:,} configurations.")
print(f"Found {valid_configs_found:,} valid configurations (no violations).")

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
