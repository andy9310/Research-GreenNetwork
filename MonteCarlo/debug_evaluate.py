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
parser = argparse.ArgumentParser(description='Debug evaluation of a Monte Carlo agent for network topology optimization')
parser.add_argument('--config', type=str, default='config_5node.json', help='Path to configuration JSON file')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
parser.add_argument('--gpu-device', type=int, default=0, help='GPU device index to use')
parser.add_argument('--tm-index', type=int, default=2, help='Index of traffic matrix to use (default: 2)')
parser.add_argument('--architecture', type=str, choices=['mlp', 'fat_mlp', 'transformer'], default='transformer', 
                   help='Neural network architecture to use: mlp (standard), fat_mlp (wider/deeper), or transformer')
parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension size for the network')
parser.add_argument('--model-path', type=str, default="models/monte_carlo_transformer_config_5node.pth", 
                   help='Path to specific model to use (overrides auto detection)')
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

# --- Environment Instantiation ---
seed = 42  # For reproducibility
env = NetworkEnv(
    adj_matrix=adj_matrix,
    edge_list=edge_list,
    tm_list=tm_list,
    node_props=node_props,
    num_nodes=num_nodes,
    link_capacity=link_capacity,
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
model_path = args.model_path

if os.path.exists(model_path):
    print(f"Loading model from {model_path}")
    agent.load_model(model_path)
else:
    print(f"Model not found at {model_path}. Evaluating with untrained model.")

# --- ADD DEBUG FUNCTIONS TO TRACK REWARDS ---
# Initialize global cumulative reward
cumulative_reward = 0

# Define debug function to wrap the step method
def debug_step_wrapper(action):
    """Wraps the environment step method to provide detailed debugging output."""
    edge_idx = env.current_edge_idx
    print(f"\nDEBUG Step {edge_idx+1}/{env.num_edges} - Edge: {env.edge_list[edge_idx]}")
    print(f"  Action: {'OPEN' if action == 1 else 'CLOSE'} (value: {action})")
    
    # Get link status before the action
    link_status_before = env.link_open.copy()
    
    # Call the original step method
    obs, reward, done, truncated, info = env.step(action)
    
    # Get link status after the action
    link_status_after = env.link_open.copy()
    
    # Analysis of what happened
    print(f"  Link Status Before: {link_status_before}")
    print(f"  Link Status After:  {link_status_after}")
    print(f"  Reward: {reward}")
    print(f"  Violations: {info.get('violation', 'None')}")
    
    # Check for overloads without violations being reported
    overload_count = 0
    for i, (u, v) in enumerate(env.edge_list):
        if env.link_open[i] == 1:  # Only check open links
            capacity = env.graph[u][v]['capacity']
            if capacity > 0 and env.usage[i] > capacity:
                overload_count += 1
                print(f"  WARNING: Unreported overload on link {i}: {env.edge_list[i]} - Usage: {env.usage[i]:.2f}/{capacity}")
    
    # Detailed usage information
    if args.verbose:
        print("  Link Usage:")
        for i, (u, v) in enumerate(env.edge_list):
            if env.link_open[i] == 1:  # Only show open links
                capacity = env.graph[u][v]['capacity']
                usage_pct = (env.usage[i] / capacity * 100) if capacity > 0 else 0
                print(f"    Link {i} ({u}-{v}): {env.usage[i]:.2f}/{capacity} ({usage_pct:.1f}%)")
    
    # Report cumulative reward
    global cumulative_reward
    cumulative_reward += reward
    print(f"  Cumulative Reward: {cumulative_reward}")
    
    return obs, reward, done, truncated, info

# --- Evaluation ---
# Set the specific traffic matrix to evaluate
if args.tm_index is not None:
    if args.tm_index < 0 or args.tm_index >= len(tm_list):
        print(f"Error: Traffic matrix index {args.tm_index} is out of range (0-{len(tm_list)-1})")
        exit(1)
    env.current_tm_idx = args.tm_index
    print(f"Evaluating on traffic matrix {args.tm_index+1}/{len(tm_list)}")
else:
    print("No traffic matrix specified. Please use --tm-index.")
    exit(1)

# Reset cumulative reward at the start of evaluation
cumulative_reward = 0

# Run a single evaluation
print("\n=== Starting Debug Evaluation ===")
state, _, _, _, _ = env.reset()
done = False
episode_reward = 0
step = 0
violations = {'isolation': 0, 'overload': 0}

# Print initial state information
print("\nInitial environment state:")
print(f"Traffic Matrix Index: {env.current_tm_idx}")
print(f"Initial Links: {env.link_open}")

print("\n--- Step-by-Step Execution ---")
while not done:
    step += 1
    # Select action (no exploration)
    with torch.no_grad():
        action = agent.select_action(state, epsilon=0)
        
        # Print the raw action values
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = agent.policy_network(state_tensor).cpu().numpy()[0]
        print(f"Q-values: CLOSE={q_values[0]:.3f}, OPEN={q_values[1]:.3f}")
    
    # Execute action with our debug wrapper
    next_state, reward, done, _, info = debug_step_wrapper(action)
    
    # Track reward and violations
    episode_reward += reward
    if info.get('violation') == 'isolation':
        violations['isolation'] += 1
    elif info.get('violation') == 'overload':
        violations['overload'] += info.get('num_overloaded', 1)
    
    # Move to next state
    state = next_state

# Print final results
print("\n=== Final Results ===")
print(f"Total Reward: {episode_reward}")
print(f"Final Link Configuration: {env.link_open}")
print(f"Number of Links Closed: {np.sum(env.link_open == 0)}/{env.num_edges}")
print(f"Violations: {violations}")

print("\nDebug evaluation complete.")
