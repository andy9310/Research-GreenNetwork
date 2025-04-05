import torch
import numpy as np
from env import NetworkEnv
from agent import DQN, QNetwork # Import QNetwork too for model loading
import time
import json # Import json

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

# Use a different seed for evaluation if desired
eval_seed = int(time.time())
env = NetworkEnv(
    adj_matrix=adj_matrix,
    edge_list=edge_list,
    tm_list=tm_list,
    node_props=node_props,
    num_nodes=num_nodes,
    link_capacity=link_capacity,
    max_edges=max_edges, # Pass max_edges to env
    seed=eval_seed
)

num_actual_edges = env.num_edges # Actual edges in this specific config

# --- Agent Setup ---
# State dimension based on padded observation space
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n # Should be 2
hidden_dim = 128 # Should match the hidden_dim used during training

print(f"Evaluating Topology: {num_actual_edges} actual edges (max_edges={max_edges})")
print(f"State Dimension (Padded): {state_dim}")
print(f"Action Dimension: {action_dim}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the QNetwork structure (needed to load state_dict)
policy_net = QNetwork(state_dim, action_dim, hidden_dim).to(device)

# --- Load Trained Model --- 
model_load_path = "dqn_network_model_padded.pth" # Use the new padded model name
print(f"\nLoading trained model from {model_load_path}...")
try:
    policy_net.load_state_dict(torch.load(model_load_path, map_location=device))
    policy_net.eval() # Set the network to evaluation mode
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_load_path}")
    print("Please train the model first using train.py with the same max_edges setting.")
    exit()
except RuntimeError as e:
    print(f"Error loading model: {e}")
    print("This might happen if the saved model's architecture (state/action/hidden dims)")
    print("doesn't match the current model structure based on config.json (max_edges).")
    print("Ensure max_edges in config.json is the same as during training.")
    exit()

# --- Evaluation Loop --- 
num_eval_episodes = 5 # Evaluate over a few episodes for stability
total_rewards = []
final_link_configs = []
violations_occurred = []

print(f"\nRunning evaluation for {num_eval_episodes} episodes...")
for episode in range(num_eval_episodes):
    state = env.reset() # Reset returns only obs
    episode_reward = 0
    done = False
    step = 0
    episode_violations = {'isolated': 0, 'overloaded': 0}

    while not done:
        step += 1
        # Select action greedily (epsilon=0.0) using the loaded policy net
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            action = q_values.argmax(dim=1).item()

        # Execute action in environment
        next_state, reward, done, info = env.step(action)

        episode_reward += reward
        state = next_state

        # Track violations if they occur
        if info.get('violation') == 'isolated':
            episode_violations['isolated'] += 1
        elif info.get('violation') == 'overloaded':
             episode_violations['overloaded'] += info.get('num_overloaded', 1)

        if done:
            total_rewards.append(episode_reward)
            final_link_configs.append(env.link_open.copy()) # Store copy of final state
            violations_occurred.append(episode_violations)
            print(f" Episode {episode + 1}: Reward={episode_reward:.2f}, Final Links={env.link_open}, Violations={episode_violations}")
            break
        # Optional: Add max steps per episode safeguard
        # if step > env.num_edges * 2: # Example safeguard
        #     print(f" Episode {episode + 1}: Exceeded max steps, terminating early.")
        #     break

# --- Report Results --- 
print("\n--- Evaluation Summary ---")
if total_rewards:
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f" Average Reward: {avg_reward:.2f} +/- {std_reward:.2f}")

    # Analyze final configurations (e.g., show the most common one)
    # Convert configurations to tuples for easier comparison/counting
    config_tuples = [tuple(cfg) for cfg in final_link_configs]
    if config_tuples:
        from collections import Counter
        most_common_config, count = Counter(config_tuples).most_common(1)[0]
        print(f" Most Common Final Configuration ({count}/{num_eval_episodes} times): {np.array(most_common_config)}")

    # Summarize violations
    total_iso = sum(v['isolated'] for v in violations_occurred)
    total_ovl = sum(v['overloaded'] for v in violations_occurred)
    episodes_with_violations = sum(1 for v in violations_occurred if v['isolated'] > 0 or v['overloaded'] > 0)
    print(f" Total Isolation Violations across episodes: {total_iso}")
    print(f" Total Overload Violations across episodes: {total_ovl}")
    print(f" Episodes with any violation: {episodes_with_violations}/{num_eval_episodes}")
else:
    print("No episodes completed successfully.")
