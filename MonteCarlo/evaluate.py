import torch
import numpy as np
from env import NetworkEnv
from agent import MonteCarloAgent
import json
import argparse
import os
from tqdm import tqdm

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
parser.add_argument('--episodes', type=int, default=100, help='Number of evaluation episodes')
parser.add_argument('--architecture', type=str, choices=['mlp', 'fat_mlp', 'transformer'], default='transformer', 
                   help='Neural network architecture to use: mlp (standard), fat_mlp (wider/deeper), or transformer')
parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension size for the network')
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
# Extract config name for model path
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

# Evaluation
print(f"\nEvaluating model for {args.episodes} episodes per traffic matrix...")

# Track metrics
all_rewards = []
all_open_links = []
all_violations = []

# Evaluate on each traffic matrix
for tm_idx, _ in enumerate(tm_list):
    env.current_tm_idx = tm_idx
    
    tm_rewards = []
    tm_open_links = []
    tm_violations = []
    
    print(f"\nEvaluating on traffic matrix {tm_idx+1}/{len(tm_list)}")
    
    # Run evaluation episodes
    for episode in tqdm(range(args.episodes), desc=f"TM {tm_idx+1}"):
        state, _, _, _, _ = env.reset()
        episode_reward = 0
        done = False
        violations = {'isolation': 0, 'overload': 0}
        
        while not done:
            # Select action (no exploration)
            with torch.no_grad():
                action = agent.select_action(state, epsilon=0)
            
            # Execute action
            next_state, reward, done, _, info = env.step(action)
            
            # Track reward and violations
            episode_reward += reward
            if info.get('violation'):
                violations[info['violation']] += 1
            
            # Move to next state
            state = next_state
        
        # Record episode metrics
        tm_rewards.append(episode_reward)
        
        # Count open links at the end of the episode
        open_links = np.sum(env.link_open)
        tm_open_links.append(open_links)
        
        # Count total violations
        total_violations = sum(violations.values())
        tm_violations.append(total_violations)
    
    # Calculate statistics for this traffic matrix
    avg_reward = np.mean(tm_rewards)
    avg_open_links = np.mean(tm_open_links)
    avg_violations = np.mean(tm_violations)
    
    print(f"  Avg Reward: {avg_reward:.2f}")
    print(f"  Avg Open Links: {avg_open_links:.2f}/{env.num_edges} ({avg_open_links/env.num_edges*100:.1f}%)")
    print(f"  Avg Violations: {avg_violations:.2f}")
    
    # Add to overall metrics
    all_rewards.extend(tm_rewards)
    all_open_links.extend(tm_open_links)
    all_violations.extend(tm_violations)

# Calculate overall statistics
overall_avg_reward = np.mean(all_rewards)
overall_avg_open_links = np.mean(all_open_links)
overall_avg_violations = np.mean(all_violations)

print("\nOverall Evaluation Results:")
print(f"  Overall Avg Reward: {overall_avg_reward:.2f}")
print(f"  Overall Avg Open Links: {overall_avg_open_links:.2f}/{env.num_edges} ({overall_avg_open_links/env.num_edges*100:.1f}%)")
print(f"  Overall Avg Violations: {overall_avg_violations:.2f}")

print("\nEvaluation complete.")
