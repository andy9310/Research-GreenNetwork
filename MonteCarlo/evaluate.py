import torch
import numpy as np
from env import NetworkEnv
from agent import MonteCarloAgent
import json
import argparse
import os
import time

# --- Load topology configuration from JSON ---
def load_config(config_path="config.json"):
    # Check if the path is a relative path without directory
    if '/' not in config_path and '\\' not in config_path:
        # Prepend configs directory path
        config_path = f"../configs/{config_path}"
    
    # Now open and load the config file
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Evaluate a Monte Carlo agent for network topology optimization')
parser.add_argument('--config', type=str, default='config_5node.json', help='Path to configuration JSON file')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
parser.add_argument('--gpu-device', type=int, default=0, help='GPU device index to use')
parser.add_argument('--tm-index', type=int, default=None, help='Index of traffic matrix to use (default: evaluate all)')
parser.add_argument('--architecture', type=str, choices=['mlp', 'fat_mlp', 'transformer'], default='transformer', 
                   help='Neural network architecture to use: mlp (standard), fat_mlp (wider/deeper), or transformer')
parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension size for the network')
parser.add_argument('--model-path', type=str, default=None, help='Path to specific model to use (overrides auto detection)')
parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
args = parser.parse_args()

# Load config from specified file
config_path = args.config
print(f"Loading configuration from {config_path}")
config = load_config(config_path)

# --- Environment Setup (Load from config) ---
num_nodes = config["num_nodes"]
adj_matrix = config["adj_matrix"]
edge_list = config["edge_list"]
node_props = config["node_props"]
tm_list = config["tm_list"]
link_capacity = config["link_capacity"]
max_edges = config["max_edges"]

print(f"Loaded configuration with {num_nodes} nodes and {len(edge_list)} edges")
print(f"Number of traffic matrices: {len(tm_list)}")

# --- Environment Instantiation with Capacity Adjustment ---
seed = 42  # For reproducibility

# Function to compute adjusted capacity for traffic matrices with high demands
def get_adjusted_capacity(traffic_matrix, edge_list, default_capacity):
    """Calculate adjusted capacity for traffic matrices with high demands.
    Uses the same logic as the bruteforce algorithm."""
    tm = np.array(traffic_matrix)
    total_traffic = np.sum(tm)
    avg_traffic_per_edge = total_traffic / len(edge_list) * 2  # Conservative estimate
    
    if avg_traffic_per_edge > default_capacity:
        suggested_capacity = int(avg_traffic_per_edge * 1.5)  # Add 50% margin
        print(f"WARNING: Traffic matrix might require higher capacity.")
        print(f"  - Current capacity: {default_capacity}")
        print(f"  - Suggested minimum capacity: {suggested_capacity}")
        print(f"  - Using adjusted capacity: {suggested_capacity}")
        return suggested_capacity
    else:
        return default_capacity

# Initialize environment with base capacity first
env = NetworkEnv(
    adj_matrix=adj_matrix,
    edge_list=edge_list,
    tm_list=tm_list,
    node_props=node_props,
    num_nodes=num_nodes,
    link_capacity=link_capacity,  # Will be adjusted per traffic matrix as needed
    max_edges=max_edges,
    seed=seed
)

# --- Agent Setup ---
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Device configuration
if args.gpu and torch.cuda.is_available():
    device_idx = args.gpu_device
    if device_idx >= torch.cuda.device_count():
        print(f"Warning: GPU device index {device_idx} out of range. Using device 0 instead.")
        device_idx = 0
    device = f"cuda:{device_idx}"
    print(f"Using GPU: {torch.cuda.get_device_name(device_idx)}")
else:
    device = "cpu"
    print("Using CPU")

# Initialize the agent
agent = MonteCarloAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dim=args.hidden_dim,
    device=device,
    network_type=args.architecture,
    nhead=4,
    num_layers=2
)

# Load the trained model
if args.model_path:
    # Use explicitly specified model path
    model_path = args.model_path
else:
    # Auto-detect model path based on config name
    if '/' in config_path:
        config_file = config_path.split('/')[-1]
    else:
        config_file = config_path.split('\\')[-1] if '\\' in config_path else config_path
    
    config_name = config_file.split('.')[0]  # Remove .json extension
    model_path = f"models/monte_carlo_{args.architecture}_{config_name}.pth"

if os.path.exists(model_path):
    print(f"Loading model from {model_path}")
    agent.load_model(model_path)
else:
    print(f"Model not found at {model_path}. Evaluating with untrained model.")

# Set the seed for reproducibility
eval_seed = int(time.time())
# Note: NetworkEnv doesn't have a direct seed method, seed is set during initialization

# Determine which traffic matrices to evaluate
tm_indices = [args.tm_index] if args.tm_index is not None else range(len(tm_list))

print(f"\nEvaluating model on {len(tm_indices)} traffic matrices...")

# Results storage
results = {}

# Evaluate on each traffic matrix
for index in tm_indices:
    if index < 0 or index >= len(tm_list):
        print(f"Error: Traffic matrix index {index} is out of range (0-{len(tm_list)-1})")
        continue
    
    # Adjust capacity for this traffic matrix if needed
    adjusted_capacity = get_adjusted_capacity(tm_list[index], edge_list, link_capacity)
    
    # Update environment's link capacity
    for i, (u, v) in enumerate(env.edge_list):
        env.graph[u][v]['capacity'] = adjusted_capacity
        
    env.current_tm_idx = index
    
    print(f"\n--- Evaluating on Traffic Matrix {index+1}/{len(tm_list)} ---")
    print(f"  Using capacity: {adjusted_capacity}")
    
    # Single evaluation run
    state, _, _, _, _ = env.reset()
    episode_reward = 0
    done = False
    step = 0
    violations = {'isolation': 0, 'overload': 0}
    
    while not done:
        step += 1
        # Select action (no exploration)
        with torch.no_grad():
            action = agent.select_action(state, epsilon=0)
        
        # Execute action
        next_state, reward, done, _, info = env.step(action)
        
        # Track reward and violations
        episode_reward += reward
        if info.get('violation') == 'isolation':
            violations['isolation'] += 1
        elif info.get('violation') == 'overload':
            violations['overload'] += info.get('num_overloaded', 1)
        
        # Move to next state
        state = next_state
    
    # Store results for this traffic matrix
    open_links = np.sum(env.link_open)
    closed_links = env.num_edges - open_links
    total_violations = sum(violations.values())
    
    # Print detailed results for this traffic matrix
    print(f"  Final Reward: {episode_reward:.2f}")
    print(f"  Links Closed: {closed_links}/{env.num_edges} ({closed_links/env.num_edges*100:.1f}%)")
    print(f"  Link Configuration: {env.link_open}")
    print(f"  Violations: {violations}")
    
    # Save results
    results[index] = {
        "reward": float(episode_reward),
        "link_config": env.link_open.tolist(),
        "links_closed": int(closed_links),
        "total_links": int(env.num_edges),
        "violations": violations,
        "steps": step
    }
    
    # No need for statistics since we're only running once

# Save results to file
config_basename = os.path.basename(config_path).replace('.json', '')
results_file = f"mc_eval_results_{config_basename}.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {results_file}")
print("\nEvaluation complete.")
