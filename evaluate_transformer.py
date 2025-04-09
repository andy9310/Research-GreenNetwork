import torch
import numpy as np
from env import NetworkEnv
from agent_transformer import SequentialDQNAgent, SequenceTracker
import json
import time
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# --- Load Configuration from JSON ---
def load_config(config_path="config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Evaluate a trained Transformer DQN agent for network topology optimization')
parser.add_argument('--config', type=str, default='config.json', help='Path to configuration JSON file')
parser.add_argument('--tm-index', type=int, default=0, help='Index of traffic matrix to evaluate (default: 0)')
parser.add_argument('--gpu', action='store_true', help='Force using GPU if available')
parser.add_argument('--gpu-device', type=int, default=0, help='GPU device index to use when multiple GPUs are available')
parser.add_argument('--seq-length', type=int, default=10, help='Sequence length for transformer evaluation')
parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension for transformer model')
parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to evaluate')
parser.add_argument('--render', action='store_true', help='Render the environment during evaluation (if supported)')
parser.add_argument('--save-plot', action='store_true', help='Save evaluation results as plots')
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

# --- Environment Instantiation ---
seed = 42  # Keep a fixed seed for reproducibility
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

# Get actual number of edges in this topology
num_actual_edges = env.num_edges

# --- Agent Setup ---
# State and action dimensions
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n  # Should be 2

print(f"Topology: {num_actual_edges} actual edges (max_edges={max_edges})")
print(f"State Dimension (Padded): {state_dim}")
print(f"Action Dimension: {action_dim}")
print(f"Using sequence length: {args.seq_length}")

# --- Device configuration ---
if args.gpu and torch.cuda.is_available():
    if args.gpu_device >= 0 and args.gpu_device < torch.cuda.device_count():
        device = torch.device(f"cuda:{args.gpu_device}")
    else:
        device = torch.device("cuda:0")
    print(f"Using GPU: {device}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# --- Create Transformer DQN Agent ---
agent = SequentialDQNAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dim=args.hidden_dim,
    nhead=4,
    num_layers=3,
    dropout=0.1,
    sequence_length=args.seq_length,
    prediction_horizon=5,  # Number of steps to predict ahead
    learning_rate=0.0001,  # Not used during evaluation
    gamma=0.99,           # Not used during evaluation
    device=device
)

# Create sequence tracker
sequence_tracker = SequenceTracker(args.seq_length, state_dim)

# --- Load Trained Model ---
config_name = config_path.split('.')[0]  # Remove .json extension
model_path = f"transformer_dqn_model_{config_name}.pth"
print(f"Loading model from {model_path}...")
try:
    agent.load(model_path)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file {model_path} not found.")
    print("Please train the model first using train_transformer.py.")
    exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# --- Evaluate Model ---
# Set traffic matrix to evaluate on
tm_idx = args.tm_index
if tm_idx < 0 or tm_idx >= len(tm_list):
    print(f"Error: Traffic matrix index {tm_idx} out of range (0-{len(tm_list)-1}).")
    exit(1)
    
env.current_tm_idx = tm_idx
print(f"Evaluating on traffic matrix {tm_idx} of {len(tm_list)}...")

# Track rewards and actions
episode_rewards = []
episode_actions = []
episode_violations = []
episode_final_utilizations = []

# Run evaluation episodes
for episode in range(args.episodes):
    print(f"\nEpisode {episode+1}/{args.episodes}")
    
    # Reset environment and agent
    state, _ = env.reset()
    agent.reset()
    sequence_tracker.reset()
    sequence_tracker.update(state)
    
    # Track episode data
    episode_reward = 0
    actions_taken = []
    violations = 0
    
    # Create progress bar for this episode
    max_steps = env.num_edges  # Maximum steps should be number of edges
    progress = tqdm(total=max_steps, desc=f"Episode {episode+1} Progress")
    
    done = False
    step = 0
    
    while not done:
        # Get action from agent (no exploration)
        action = agent.select_action(state, epsilon=0.0)
        actions_taken.append(int(action))
        
        # Execute action
        next_state, reward, done, info = env.step(action)
        
        # Update sequence tracker
        sequence_tracker.update(next_state, action)
        
        # Update stats
        episode_reward += reward
        step += 1
        progress.update(1)
        
        # Check for violations
        if 'violation' in info and info['violation']:
            violations += 1
        
        # Print step info
        edge_idx = env.current_edge_idx - 1  # Previous edge we just decided on
        if edge_idx >= 0 and edge_idx < len(env.edge_list):
            edge = env.edge_list[edge_idx]
            action_name = "Keep" if action == 1 else "Remove"
            progress.write(f"  Step {step}: Edge {edge_idx} ({edge[0]}->{edge[1]}): {action_name}, Reward: {reward:.2f}")
        
        # Move to next state
        state = next_state
        
        # Break if we've exceeded the expected number of steps
        if step > max_steps + 10:  # Add some buffer
            print(f"Warning: Episode exceeded expected number of steps ({max_steps}).")
            break
    
    progress.close()
    
    # Record final results
    episode_rewards.append(episode_reward)
    episode_actions.append(actions_taken)
    episode_violations.append(violations)
    
    # Get final network utilization
    if hasattr(env, 'network_utilization'):
        final_utilization = env.network_utilization
        episode_final_utilizations.append(final_utilization)
        print(f"Final network utilization: {final_utilization:.2f}%")
    
    print(f"Episode {episode+1} Reward: {episode_reward:.2f}, Violations: {violations}")
    
    # Print action distribution
    actions = np.array(actions_taken)
    num_kept = np.sum(actions == 1)
    num_removed = np.sum(actions == 0)
    print(f"Actions: Kept {num_kept}/{len(actions)} edges ({num_kept/len(actions)*100:.1f}%), "
          f"Removed {num_removed}/{len(actions)} edges ({num_removed/len(actions)*100:.1f}%)")

# --- Print Evaluation Results ---
print("\nEvaluation Results:")
print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
print(f"Average Violations: {np.mean(episode_violations):.2f} ± {np.std(episode_violations):.2f}")

if episode_final_utilizations:
    print(f"Average Network Utilization: {np.mean(episode_final_utilizations):.2f}% ± {np.std(episode_final_utilizations):.2f}%")

# --- Plot Results (if matplotlib is available and save_plot is True) ---
if args.save_plot:
    try:
        # Create plots directory if it doesn't exist
        os.makedirs("plots", exist_ok=True)
        
        # Plot rewards
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(episode_rewards)), episode_rewards)
        plt.axhline(y=np.mean(episode_rewards), color='r', linestyle='-', label=f'Mean: {np.mean(episode_rewards):.2f}')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(f'Transformer DQN Evaluation - Rewards (TM {tm_idx})')
        plt.legend()
        plt.savefig(f"plots/transformer_eval_{config_name}_tm{tm_idx}_rewards.png")
        
        # Plot action distributions for each episode
        plt.figure(figsize=(12, 6))
        width = 0.35
        for i, actions in enumerate(episode_actions):
            actions = np.array(actions)
            keeps = np.sum(actions == 1)
            removes = np.sum(actions == 0)
            plt.bar(i-width/2, keeps, width, label='Keep' if i == 0 else None, color='green')
            plt.bar(i+width/2, removes, width, label='Remove' if i == 0 else None, color='red')
        
        plt.xlabel('Episode')
        plt.ylabel('Count')
        plt.title(f'Transformer DQN Evaluation - Action Distribution (TM {tm_idx})')
        plt.legend()
        plt.savefig(f"plots/transformer_eval_{config_name}_tm{tm_idx}_actions.png")
        
        print(f"Plots saved to plots/transformer_eval_{config_name}_tm{tm_idx}_*.png")
    except ImportError:
        print("Matplotlib not available. Skipping plot generation.")
