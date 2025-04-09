import torch
import numpy as np
from env import NetworkEnv # Assuming env.py is in the same directory
from agent import DQN, ReplayBuffer # Assuming agent.py contains DQN and ReplayBuffer
import json # Import the json library
import time # Import time for seeding if needed
from tqdm import tqdm # Import tqdm for progress bar
import argparse # For command-line arguments
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
# --- Load topology configuration from JSON ---
def load_config(config_path="config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a DQN agent for network topology optimization')
parser.add_argument('--config', type=str, default='config.json', help='Path to configuration JSON file')
parser.add_argument('--gpu', action='store_true', help='Force using GPU if available')
parser.add_argument('--gpu-device', type=int, default=0, help='GPU device index to use when multiple GPUs are available')
parser.add_argument('--load-model', action='store_true', help='Load and continue training from an existing model')
parser.add_argument('--tm-index', type=int, default=None, help='Start training from a specific traffic matrix index')
parser.add_argument('--episodes', type=int, default=5000, help='Number of episodes to train per traffic matrix')
parser.add_argument('--tm-subset', type=int, default=None, help='Use only a subset of traffic matrices (specify count)')
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
max_edges = config["max_edges"] # Load max_edges

# --- Environment Instantiation ---
seed = 42 # Keep a fixed seed for training reproducibility
env = NetworkEnv(
    adj_matrix=adj_matrix,
    edge_list=edge_list,
    tm_list=tm_list,
    node_props=node_props,
    num_nodes=num_nodes,
    link_capacity=link_capacity,
    max_edges=max_edges, # Pass max_edges to env
    seed=seed
)

num_actual_edges = env.num_edges # Actual edges in this specific config

# --- Agent Setup ---
# State dimension based on padded observation space
state_dim = env.observation_space.shape[0] # Get state dim from env
action_dim = env.action_space.n # Should be 2

print(f"Topology: {num_actual_edges} actual edges (max_edges={max_edges})")
print(f"State Dimension (Padded): {state_dim}")
print(f"Action Dimension: {action_dim}")

# Hyperparameters
hidden_dim = 256  # Match the enhanced architecture
learning_rate = 5e-5  # Reduced learning rate for stability
gamma = 0.99
buffer_size = 100000 # Increased buffer size
batch_size = 128  # Larger batch size for better training

epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay_steps = 20000 # Adjust decay steps based on expected total steps
target_update_freq = 1000  # Update target net less frequently (in steps)

# Device configuration
if args.gpu and torch.cuda.is_available():
    if args.gpu_device >= 0 and args.gpu_device < torch.cuda.device_count():
        device = torch.device(f"cuda:{args.gpu_device}")
    else:
        device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True  # Optimize CUDNN
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster computation
    torch.backends.cudnn.allow_tf32 = True       # Allow TF32 for faster computation
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    print(f"CUDA Memory Available: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
else:
    device = torch.device("cpu")
    if args.gpu and not torch.cuda.is_available():
        print("Warning: GPU requested but CUDA is not available. Using CPU instead.")
    else:
        print("Using CPU for training")


# Initialize the DQN agent with a transformer-based network architecture
agent = DQN(state_dim, action_dim, hidden_dim=hidden_dim, lr=learning_rate, gamma=gamma, device=device, 
         network_type='transformer', nhead=4, num_layers=2)

# Load pre-trained model if requested
config_name = config_path.split('.')[0]  # Remove .json extension
model_path = f"dqn_model_{config_name}.pth"
if args.load_model:
    print(f"Loading pre-trained model from {model_path}...")
    try:
        agent.qnetwork_local.load_state_dict(torch.load(model_path, map_location=device))
        agent.update_target_network()  # Make sure target network is updated with loaded weights
        print("Pre-trained model loaded successfully. Continuing training...")
    except FileNotFoundError:
        print(f"Warning: Model file {model_path} not found. Starting with a new model.")
    except Exception as e:
        print(f"Error loading model: {e}. Starting with a new model.")

replay_buffer = ReplayBuffer(buffer_size)

# --- Training Loop ---
num_episodes_per_tm = args.episodes # Episodes per traffic matrix
total_steps = 0
episode_rewards = []
tm_episode_rewards = []  # Track rewards for each traffic matrix
print_interval = 100

# Determine starting traffic matrix index
start_tm_idx = 0
if args.tm_index is not None:
    if 0 <= args.tm_index < len(tm_list):
        start_tm_idx = args.tm_index
        print(f"Starting training from traffic matrix index {start_tm_idx}")
    else:
        print(f"Warning: Traffic matrix index {args.tm_index} is out of range (0-{len(tm_list)-1})")
        print(f"Starting from the first traffic matrix")

# Calculate total episodes based on starting matrix
training_matrices = len(tm_list) - start_tm_idx
total_num_episodes = training_matrices * num_episodes_per_tm

print(f"\nStarting Training on {training_matrices} traffic matrices, {num_episodes_per_tm} episodes each...")

# Create a progress bar for traffic matrices
tm_progress = tqdm(total=total_num_episodes, desc="Total Training Progress")

# Limit the number of traffic matrices if subset option is used
training_tm_list = tm_list[start_tm_idx:]
if args.tm_subset is not None and args.tm_subset > 0 and args.tm_subset < len(training_tm_list):
    # Take a representative subset evenly distributed across the list
    if args.tm_subset >= 3:
        # Try to get representatives from start, middle, and end
        indices = np.linspace(0, len(training_tm_list)-1, args.tm_subset, dtype=int)
        training_tm_list = [training_tm_list[i] for i in indices]
        print(f"Using {len(training_tm_list)} traffic matrices (subset) from indices: {indices}")
    else:
        training_tm_list = training_tm_list[:args.tm_subset]
        print(f"Using first {len(training_tm_list)} traffic matrices")
    
    # Recalculate total episodes based on the subset
    total_num_episodes = len(training_tm_list) * num_episodes_per_tm
    tm_progress.total = total_num_episodes
    tm_progress.refresh()

# Train on each traffic matrix
for tm_idx, traffic_matrix in enumerate(training_tm_list):
    # Set this traffic matrix in the environment
    env.current_tm_idx = tm_idx
    
    print(f"\nTraining on traffic matrix {tm_idx+1}/{len(tm_list)}...")
    
    # Reset epsilon for each traffic matrix
    epsilon = epsilon_start
    tm_steps = 0
    
    # Reset episode rewards for this traffic matrix
    tm_episode_rewards = []
    
    # Training loop for this traffic matrix
    for episode in range(num_episodes_per_tm):
        state, _ = env.reset() # Env reset returns (obs, info)
        episode_reward = 0
        done = False

        while not done:
            total_steps += 1
            tm_steps += 1

            # Calculate current epsilon - reset for each traffic matrix
            # Linear decay: max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (tm_steps / epsilon_decay_steps))
            # Exponential decay: epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * tm_steps / epsilon_decay_steps)
            epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (tm_steps / epsilon_decay_steps))

            # Use torch.no_grad() for action selection to reduce memory usage and speed up training
            with torch.no_grad():
                # Select action using epsilon-greedy (no mask needed for action_dim=2)
                action = agent.select_action(state, epsilon)

            # Execute action in environment
            next_state, reward, done, info = env.step(action)
            
            # Debug severe penalties
            # if reward < -1000:
            #     violation_type = info.get('violation', 'unknown')
            #     current_edge = env.current_edge_idx - 1  # The edge we just decided on
            #     if (episode + 1) % print_interval == 0:  # To avoid too much output
            #         tqdm.write(f"  Step {total_steps}: Large penalty {reward:.0f} due to {violation_type} violation on edge {current_edge}")

            # Store experience in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)

            # Move to the next state
            state = next_state
            episode_reward += reward

            # Perform learning step if buffer has enough samples
            if len(replay_buffer) > batch_size:
                loss = agent.learn(replay_buffer, batch_size)
                # Optionally log loss

            # Update target network periodically
            if total_steps % target_update_freq == 0:
                agent.update_target_network()

            # Episode finished (either done or max steps per episode if implemented)
            if done:
                break

        # Add to both overall and traffic matrix specific rewards
        episode_rewards.append(episode_reward)
        tm_episode_rewards.append(episode_reward)
        
        # Update the progress bar
        tm_progress.update(1)
        
        # Print progress at intervals
        if (episode + 1) % print_interval == 0:
            avg_reward = np.mean(tm_episode_rewards[-min(print_interval, len(tm_episode_rewards)):]) 
            avg_overall = np.mean(episode_rewards[-min(print_interval, len(episode_rewards)):])
            
            # Print progress information
            tm_progress.write(f"TM {tm_idx+1}/{len(tm_list)}, Episode {episode + 1}/{num_episodes_per_tm}, "
                             f"TM Avg Reward: {avg_reward:.2f}, Overall Avg: {avg_overall:.2f}, "
                             f"Epsilon: {epsilon:.3f}, Total Steps: {total_steps}")


print("\nTraining finished.")

# --- Save Model --- 
# Use config name in the model filename to avoid overwriting models from different configs
config_name = config_path.split('.')[0]  # Remove .json extension
model_save_path = f"dqn_model_{config_name}.pth"
print(f"Saving trained model to {model_save_path}...")
torch.save(agent.qnetwork_local.state_dict(), model_save_path)
print("Model saved.")
