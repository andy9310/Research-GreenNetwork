import torch
import numpy as np
from env import NetworkEnv # Assuming env.py is in the same directory
from agent import DQN, ReplayBuffer # Assuming agent.py contains DQN and ReplayBuffer
import json # Import the json library
import time # Import time for seeding if needed
from tqdm import tqdm # Import tqdm for progress bar

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
hidden_dim = 128
learning_rate = 1e-4
gamma = 0.99
buffer_size = 100000 # Increased buffer size
batch_size = 128     # Increased batch size

epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay_steps = 20000 # Adjust decay steps based on expected total steps
target_update_freq = 1000  # Update target net less frequently (in steps)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

agent = DQN(state_dim, action_dim, hidden_dim=hidden_dim, lr=learning_rate, gamma=gamma, device=device)
replay_buffer = ReplayBuffer(buffer_size)

# --- Training Loop ---
num_episodes = 5000 # Adjust as needed
epsilon = epsilon_start
total_steps = 0
episode_rewards = []
print_interval = 100

print("\nStarting Training...")
for episode in tqdm(range(num_episodes), desc="Training Episodes"): # Use tqdm
    state = env.reset() # Env reset now returns only obs
    episode_reward = 0
    done = False

    while not done:
        total_steps += 1

        # Calculate current epsilon
        # Linear decay: max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (total_steps / epsilon_decay_steps))
        # Exponential decay: epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * total_steps / epsilon_decay_steps)
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (total_steps / epsilon_decay_steps))

        # Select action using epsilon-greedy (no mask needed for action_dim=2)
        action = agent.select_action(state, epsilon)

        # Execute action in environment
        next_state, reward, done, info = env.step(action)

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
            agent.update_target_net()

        # Episode finished (either done or max steps per episode if implemented)
        if done:
            break

    episode_rewards.append(episode_reward)
    if (episode + 1) % print_interval == 0:
        avg_reward = np.mean(episode_rewards[-print_interval:])
        # Clearer progress print using tqdm's position
        tqdm.write(f"Episode {episode + 1}/{num_episodes}, Avg Reward (Last {print_interval}): {avg_reward:.2f}, Epsilon: {epsilon:.3f}, Total Steps: {total_steps}")


print("\nTraining finished.")

# --- Save Model --- 
model_save_path = "dqn_network_model_padded.pth"
print(f"Saving trained model to {model_save_path}...")
torch.save(agent.policy_net.state_dict(), model_save_path)
print("Model saved.")
