import torch
import numpy as np
from env import NetworkEnv
from agent import MonteCarloAgent, EpisodeBuffer
import json
import time
from tqdm import tqdm
import argparse
import os

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

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
parser = argparse.ArgumentParser(description='Train a Monte Carlo agent for network topology optimization')
parser.add_argument('--config', type=str, default='config_5node.json', help='Path to configuration JSON file (stored in configs directory)')
parser.add_argument('--gpu', action='store_true', help='Force using GPU if available')
parser.add_argument('--gpu-device', type=int, default=0, help='GPU device index to use when multiple GPUs are available')
parser.add_argument('--load-model', action='store_true', help='Load and continue training from an existing model')
parser.add_argument('--tm-index', type=int, default=None, help='Start training from a specific traffic matrix index')
parser.add_argument('--episodes', type=int, default=3000, help='Number of episodes to train per traffic matrix')
parser.add_argument('--tm-subset', type=int, default=None, help='Use only a subset of traffic matrices (specify count)')
parser.add_argument('--architecture', type=str, choices=['mlp', 'fat_mlp', 'transformer'], default='transformer', 
                   help='Neural network architecture to use: mlp (standard), fat_mlp (wider/deeper), or transformer')
parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension size for the network')
parser.add_argument('--epsilon-start', type=float, default=1.0, help='Starting epsilon value for exploration')
parser.add_argument('--epsilon-min', type=float, default=0.05, help='Minimum epsilon value for exploration')
parser.add_argument('--epsilon-decay-steps', type=int, default=10000, help='Number of steps for epsilon to decay from start to min')
parser.add_argument('--verbose', action='store_true', help='Enable verbose output during training')
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

num_actual_edges = env.num_edges

# --- Agent Setup ---
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print(f"Topology: {num_actual_edges} actual edges (max_edges={max_edges})")
print(f"State Dimension (Padded): {state_dim}")
print(f"Action Dimension: {action_dim}")

# Hyperparameters
learning_rate = 1e-4  # Typically lower for MC methods
gamma = 0.99  # Discount factor
episode_buffer_size = 1000  # Number of episodes to store
batch_size = 32  # Number of episodes to sample for learning

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

# Initialize agent and buffer
agent = MonteCarloAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dim=args.hidden_dim,
    lr=learning_rate,
    gamma=gamma,
    device=device,
    network_type=args.architecture,
    nhead=4,
    num_layers=2
)

episode_buffer = EpisodeBuffer(capacity=episode_buffer_size)

if args.verbose:
    # Print model summary
    if args.architecture == 'fat_mlp':
        print("\nFat MLP Model Structure:")
        print(f"  Input Size: {state_dim}")
        print(f"  Layer 1: {args.hidden_dim} neurons with LayerNorm and Dropout(0.2)")
        print(f"  Layer 2: {args.hidden_dim*2} neurons with LayerNorm and Dropout(0.2)")
        print(f"  Layer 3: {args.hidden_dim*2} neurons with LayerNorm and Dropout(0.2)")
        print(f"  Layer 4: {args.hidden_dim} neurons with LayerNorm and Dropout(0.2)")
        print(f"  Output: {action_dim} neurons (action values)")
    elif args.architecture == 'transformer':
        print("\nTransformer Model Structure:")
        print(f"  Attention heads: 4")
        print(f"  Transformer layers: 2")
        print(f"  Hidden dimension: {args.hidden_dim}")
    elif args.architecture == 'mlp':
        print("\nStandard MLP Model Structure:")
        print(f"  Input Size: {state_dim}")
        print(f"  Layer 1: {args.hidden_dim} neurons with LayerNorm and Dropout(0.2)")
        print(f"  Layer 2: {args.hidden_dim*2} neurons with LayerNorm and Dropout(0.2)")
        print(f"  Layer 3: {args.hidden_dim} neurons with LayerNorm and Dropout(0.2)")
        print(f"  Output: {action_dim} neurons (action values)")

# Load pre-trained model if requested
if args.load_model:
    # Extract just the filename without path and extension for model naming
    if '/' in config_path:
        config_file = config_path.split('/')[-1]
    else:
        config_file = config_path.split('\\')[-1] if '\\' in config_path else config_path
    
    config_name = config_file.split('.')[0]
    model_load_path = f"models/monte_carlo_{args.architecture}_{config_name}.pth"
    
    if os.path.exists(model_load_path):
        print(f"Loading pre-trained model from {model_load_path}...")
        agent.load_model(model_load_path)
        print("Model loaded successfully.")
    else:
        print(f"Pre-trained model {model_load_path} not found. Starting with a new model.")

# Training settings
num_episodes_per_tm = args.episodes
episode_rewards = []
tm_episode_rewards = []
print_interval = 100

# Determine starting traffic matrix index
start_tm_idx = 0
if args.tm_index is not None:
    if 0 <= args.tm_index < len(tm_list):
        start_tm_idx = args.tm_index
        print(f"Starting training from traffic matrix index {start_tm_idx}")
    else:
        print(f"Warning: Requested TM index {args.tm_index} out of range (0-{len(tm_list)-1}). Starting from 0.")

# Calculate total episodes for progress tracking
total_num_episodes = (len(tm_list) - start_tm_idx) * num_episodes_per_tm

# Create a progress bar for traffic matrices
tm_progress = tqdm(total=total_num_episodes, desc="Total Training Progress")

# Limit the number of traffic matrices if subset option is used
training_tm_list = tm_list[start_tm_idx:]
if args.tm_subset is not None and args.tm_subset > 0 and args.tm_subset < len(training_tm_list):
    # Take a representative subset evenly distributed across the list
    if args.tm_subset >= 3:
        # Get indices approximately evenly spaced
        step = len(training_tm_list) / args.tm_subset
        indices = [int(i * step) for i in range(args.tm_subset)]
        training_tm_list = [training_tm_list[i] for i in indices]
    else:
        # For very small subsets, just take first few
        training_tm_list = training_tm_list[:args.tm_subset]
    
    # Recalculate total episodes based on the subset
    total_num_episodes = len(training_tm_list) * num_episodes_per_tm
    tm_progress.total = total_num_episodes
    tm_progress.refresh()

# Train on each traffic matrix
for tm_idx, traffic_matrix in enumerate(training_tm_list):
    # Set this traffic matrix in the environment
    env.current_tm_idx = tm_idx + start_tm_idx
    
    print(f"\nTraining on traffic matrix {tm_idx+1}/{len(training_tm_list)}...")
    if args.verbose:
        # Print basic info about this traffic matrix
        tm_sum = np.sum(traffic_matrix)
        tm_max = np.max(traffic_matrix)
        print(f"  Traffic matrix sum: {tm_sum:.2f}, max: {tm_max:.2f}")
    
    # Reset episode rewards for this traffic matrix
    tm_episode_rewards = []
    
    # Training loop for this traffic matrix
    for episode in range(num_episodes_per_tm):
        state, _, _, _, _ = env.reset()  # Start a new episode
        episode_reward = 0
        done = False
        
        # Collect the episode
        while not done:
            # Calculate epsilon based on cumulative steps
            agent.increment_step()  # Increment step counter
            epsilon = max(
                args.epsilon_min,  # Minimum epsilon
                args.epsilon_start - (agent.total_steps / args.epsilon_decay_steps) * (args.epsilon_start - args.epsilon_min)  # Linear decay
            )
            
            # Select and execute action
            action = agent.select_action(state, epsilon)
            next_state, reward, done, truncated, info = env.step(action)
            
            # Add this step to the current episode
            episode_buffer.add_experience(state, action, reward, next_state, done)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            if done:
                # End of episode, add to buffer
                episode_buffer.end_episode()
                
                # Log rewards
                episode_rewards.append(episode_reward)
                tm_episode_rewards.append(episode_reward)
                
                # Perform Monte Carlo learning
                if len(episode_buffer) > 0:
                    loss = agent.learn(episode_buffer, batch_size=min(batch_size, len(episode_buffer)))
        
        # Update progress bar
        tm_progress.update(1)
        
        # Print training progress
        if (episode + 1) % print_interval == 0:
            avg_reward = np.mean(tm_episode_rewards[-min(print_interval, len(tm_episode_rewards)):])
            avg_overall = np.mean(episode_rewards[-min(print_interval, len(episode_rewards)):])
            
            tm_progress.write(f"TM {tm_idx+1}/{len(training_tm_list)}, Episode {episode+1}/{num_episodes_per_tm}, "
                             f"TM Avg Reward: {avg_reward:.2f}, Overall Avg: {avg_overall:.2f}, "
                             f"Epsilon: {epsilon:.3f}, Total Steps: {agent.total_steps}")
            
            # Additional verbose output
            if args.verbose and (episode + 1) % (print_interval * 5) == 0:
                # Print sample action values to monitor learning progress
                with torch.no_grad():
                    if len(episode_buffer.episodes) > 0 and len(episode_buffer.episodes[-1]) > 0:
                        sample_state = episode_buffer.episodes[-1][0].state
                    else:
                        sample_state = env._get_observation()
                    
                    sample_state = torch.FloatTensor(sample_state).to(device)
                    action_vals = agent.policy_network(sample_state)
                    tm_progress.write(f"  Sample action values: close={action_vals[0]:.3f}, open={action_vals[1]:.3f}, "
                                     f"diff={action_vals[1]-action_vals[0]:.3f}")

print("\nTraining finished.")

# --- Save Model ---
# Extract just the filename without path and extension for model naming
if '/' in config_path:
    config_file = config_path.split('/')[-1]
else:
    config_file = config_path.split('\\')[-1] if '\\' in config_path else config_path

config_name = config_file.split('.')[0]  # Remove .json extension
model_save_path = f"models/monte_carlo_{args.architecture}_{config_name}.pth"

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

print(f"Saving trained model to {model_save_path}...")
agent.save_model(model_save_path)
print("Model saved.")
