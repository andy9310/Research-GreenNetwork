import torch
import numpy as np
from env import NetworkEnv 
from agent_transformer import SequentialDQNAgent, SequentialReplayBuffer, SequenceTracker
import json
import time
from tqdm import tqdm
import argparse
import os
import sys
import matplotlib.pyplot as plt
import traceback

# --- Print PyTorch info ---
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

# --- Parse command-line arguments ---
parser = argparse.ArgumentParser(description='Train a Sequential Transformer DQN agent for network topology optimization')
parser.add_argument('--config', type=str, default='config.json', help='Path to configuration JSON file')
parser.add_argument('--gpu', action='store_true', help='Force using GPU if available')
parser.add_argument('--gpu-device', type=int, default=0, help='GPU device index to use when multiple GPUs are available')
parser.add_argument('--load-model', action='store_true', help='Load and continue training from an existing model')
parser.add_argument('--tm-index', type=int, default=None, help='Start training from a specific traffic matrix index')
parser.add_argument('--episodes', type=int, default=5000, help='Number of episodes to train per traffic matrix')
parser.add_argument('--tm-subset', type=int, default=None, help='Use only a subset of traffic matrices (specify count)')
parser.add_argument('--seq-length', type=int, default=10, help='Sequence length for transformer training')
parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension for transformer model (note: currently fixed at 256 for compatibility)')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
args = parser.parse_args()

# --- Load config from specified file ---
config_path = args.config
print(f"Loading configuration from {config_path}")
try:
    config = load_config(config_path)
except FileNotFoundError:
    # Try to look for the config file in the parent directory
    parent_config_path = os.path.join("..", config_path)
    print(f"Config not found, trying parent directory: {parent_config_path}")
    config = load_config(parent_config_path)

# --- Environment Setup (Load from config) ---
num_nodes = config["num_nodes"]
adj_matrix = config["adj_matrix"]
edge_list = config["edge_list"]
node_props = config["node_props"]
tm_list = config["tm_list"]
link_capacity = config["link_capacity"]
max_edges = config["max_edges"]

# --- Environment Instantiation ---
seed = 42  # Fixed seed for reproducibility
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

# Ensure plots directory exists for storing training results
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)  # Also create a models directory for saved models

num_actual_edges = env.num_edges  # Actual edges in this specific config

# --- Agent Setup ---
# State dimension based on padded observation space
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n  # Should be 2

print(f"Topology: {num_actual_edges} actual edges (max_edges={max_edges})")
print(f"State Dimension (Padded): {state_dim}")
print(f"Action Dimension: {action_dim}")
print(f"Using sequence length: {args.seq_length}")

# --- Hyperparameters ---
# Force hidden_dim to be 256 to ensure compatibility with the transformer architecture
hidden_dim = 256  # Fixed to avoid dimension mismatch
print(f"Using fixed hidden_dim: {hidden_dim} for transformer compatibility")
learning_rate = args.lr
gamma = 0.99
buffer_size = 100000
batch_size = args.batch_size
sequence_length = args.seq_length
prediction_horizon = 5  # How many steps ahead to predict

epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay_steps = 5000
target_update_freq = 500

# --- Device configuration ---
if args.gpu and torch.cuda.is_available():
    if args.gpu_device >= 0 and args.gpu_device < torch.cuda.device_count():
        device = torch.device(f"cuda:{args.gpu_device}")
    else:
        device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"Using GPU: {device}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# --- Create Transformer DQN Agent ---
agent = SequentialDQNAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dim=hidden_dim,
    nhead=4,  # Number of attention heads
    num_layers=3,  # Number of transformer layers
    dropout=0.1,
    sequence_length=sequence_length,
    prediction_horizon=prediction_horizon,
    learning_rate=learning_rate,
    gamma=gamma,
    device=device
)

# --- Create Replay Buffer ---
replay_buffer = SequentialReplayBuffer(buffer_size, sequence_length, batch_size, device)

# --- Reset state sequences and prepare for training ---
try:
    state, info = env.reset()
except ValueError as e:
    print(f"Error during environment reset: {e}")
    print("This could be due to the environment not returning the correct tuple structure.")
    print("Attempting to fix by unpacking a 5-element tuple...")
    # Try different unpacking approaches
    state, _, _, _, info = env.reset()

# Initialize sequence tracker
sequence_tracker = SequenceTracker(sequence_length, state_dim)

# --- Model Path ---
config_name = os.path.basename(config_path).split('.')[0]  # Remove .json extension
model_path = f"models/transformer_dqn_model_{config_name}.pth"

# --- Load pre-trained model if requested ---
if args.load_model:
    print(f"Loading pre-trained model from {model_path}...")
    try:
        agent.load(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Starting with a new model.")

# --- Initialize tracking variables ---
total_steps = 0
episode_rewards = []

# Optional: start from specific traffic matrix
start_tm_idx = args.tm_index if args.tm_index is not None else 0

# Limit the number of traffic matrices used for training if specified
if args.tm_subset is not None and args.tm_subset < len(tm_list):
    training_tm_list = tm_list[start_tm_idx:start_tm_idx + args.tm_subset]
    training_matrices = args.tm_subset
else:
    training_tm_list = tm_list[start_tm_idx:]
    training_matrices = len(training_tm_list)
    
# Number of episodes per traffic matrix
num_episodes_per_tm = args.episodes

# Calculate total number of episodes for progress tracking
total_num_episodes = training_matrices * num_episodes_per_tm

print(f"\nStarting Training on {training_matrices} traffic matrices, {num_episodes_per_tm} episodes each...")

# Create a progress bar for total training
tm_progress = tqdm(total=total_num_episodes, desc="Total Training Progress")

# Define print interval for logging (print progress every N episodes)
print_interval = 100

# --- Training Loop ---
for tm_idx, traffic_matrix in enumerate(training_tm_list):
    # Set this traffic matrix in the environment
    env.current_tm_idx = tm_idx + start_tm_idx
    
    print(f"\nTraining on traffic matrix {tm_idx+start_tm_idx+1}/{len(tm_list)}...")
    
    # Reset epsilon for each traffic matrix
    epsilon = epsilon_start
    tm_steps = 0
    
    # Reset episode rewards for this traffic matrix
    tm_episode_rewards = []
    
    # Training loop for this traffic matrix
    for episode in range(num_episodes_per_tm):
        # Reset environment and sequence tracker
        state, _ = env.reset()
        agent.reset()  # Reset agent's internal buffers
        sequence_tracker.reset()  # Reset sequence tracker
        
        # Initialize sequence tracker with the initial state
        sequence_tracker.update(state)
        
        episode_reward = 0.0
        done = False
        
        # Print state shape information for debugging
        if episode == 0 and tm_idx == 0:
            print(f"Initial state shape: {np.array(state).shape}")
        
        # Episode loop
        step_count = 0
        while not done:
            total_steps += 1
            tm_steps += 1
            step_count += 1
            
            # Calculate current epsilon with linear decay
            epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (tm_steps / epsilon_decay_steps))
            
            # Debug state at beginning of training
            if episode == 0 and tm_idx == 0 and step_count <= 3:
                print(f"Step {step_count} - State shape: {np.array(state).shape}")
            
            try:
                # Select action using epsilon-greedy with robust error handling
                action = 0  # Default action in case of errors
                
                try:
                    with torch.no_grad():
                        action = agent.select_action(state, epsilon)
                        
                    # Validate action before passing to environment
                    if isinstance(action, torch.Tensor):
                        try:
                            action = action.item()
                        except Exception as e:
                            print(f"Error converting tensor action to item: {e}")
                            action = 0
                        
                    # Execute action in environment with proper error handling for API compatibility
                    try:
                        step_result = env.step(action)
                        
                        # Handle different return value patterns (4 or 5 elements)
                        if len(step_result) == 5:
                            next_state, reward, done, truncated, info = step_result
                        elif len(step_result) == 4:
                            next_state, reward, done, info = step_result
                            truncated = False
                        else:
                            raise ValueError(f"Unexpected number of return values from env.step(): {len(step_result)}")
                            
                    except ValueError as e:
                        print(f"Warning: step() function call error: {e}")
                        print("Attempting alternative unpacking...")
                        try:
                            next_state, reward, done, truncated, info = env.step(action)
                        except Exception as inner_e:
                            print(f"Alternative unpacking failed: {inner_e}")
                            # Ultimate fallback
                            action = 0
                            next_state, reward, done, truncated, info = env.step(action)
                except Exception as action_e:
                    print(f"Error during action selection or execution: {action_e}")
                    # Fallback to a safe action
                    action = 0
                    next_state, reward, done, truncated, info = env.step(action)
                
                # Update sequence tracker with the new state and action
                sequence_tracker.update(next_state, action)
                
                # Store transition in replay buffer if we have enough history
                if step_count >= sequence_length - 1:
                    # Get current sequences before the update
                    state_seq = sequence_tracker.get_state_sequence()
                    action_seq = sequence_tracker.get_action_sequence()
                    
                    # Create next state sequence by shifting window
                    next_state_seq = np.copy(state_seq)
                    if len(next_state_seq) > 0:
                        # Shift sequence and add new state at the end
                        next_state_seq = np.roll(next_state_seq, -1, axis=0)
                        next_state_seq[-1] = next_state
                    
                    # Only store experiences with valid sequences
                    if action_seq is not None and len(action_seq) > 0:
                        # Store in replay buffer
                        replay_buffer.push(state_seq, action_seq, reward, next_state_seq, done)
                
                # Move to the next state
                state = next_state
                episode_reward += reward
                
            except Exception as e:
                tm_progress.write(f"Error at step {step_count}: {str(e)}")
                if step_count <= 5:  # Only show detailed error for early steps
                    tm_progress.write(traceback.format_exc())
                # Continue with next step
                if done:
                    break
                continue
            
            # Perform learning step if buffer has enough samples
            if len(replay_buffer) > batch_size and total_steps % 4 == 0:  # Learn every 4 steps
                loss = agent.learn(replay_buffer, batch_size)
                
            # Update target network periodically
            if total_steps % target_update_freq == 0:
                agent.update_target_network()
                
            # Episode finished
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
            tm_progress.write(f"TM {tm_idx+start_tm_idx+1}/{len(tm_list)}, Episode {episode + 1}/{num_episodes_per_tm}, "
                             f"TM Avg Reward: {avg_reward:.2f}, Overall Avg: {avg_overall:.2f}, "
                             f"Epsilon: {epsilon:.3f}, Total Steps: {total_steps}")
                
        # Save the model periodically (every 500 episodes)
        if (episode + 1) % 500 == 0:
            checkpoint_path = f"models/transformer_dqn_checkpoint_{config_name}_tm{tm_idx+start_tm_idx}_ep{episode+1}.pth"
            agent.save(checkpoint_path)
            tm_progress.write(f"Checkpoint saved to {checkpoint_path}")

# --- Save Final Model ---
print("\nTraining finished.")
print(f"Saving trained model to {model_path}...")
agent.save(model_path)
print("Model saved.")

# --- Training Statistics ---
print("\nTraining Statistics:")
print(f"Total steps: {total_steps}")
print(f"Final average reward (last {min(100, len(episode_rewards))} episodes): {np.mean(episode_rewards[-min(100, len(episode_rewards))]):0.2f}")

# --- Plot Training Curve ---
try:
    # Plot episode rewards
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(episode_rewards)), episode_rewards)
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.title(f'Transformer DQN Training Curve - {config_name}')
    
    # Add smoothed moving average
    window_size = min(100, len(episode_rewards))
    if len(episode_rewards) >= window_size:
        smoothed_rewards = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(np.arange(len(smoothed_rewards)) + window_size-1, smoothed_rewards, 'r-', linewidth=2)
    
    plt.savefig(f"plots/transformer_dqn_{config_name}_rewards.png")
    print(f"Training curve saved to plots/transformer_dqn_{config_name}_rewards.png")
except Exception as e:
    print(f"Error generating plot: {str(e)}")
