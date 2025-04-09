import torch
import numpy as np
from env import NetworkEnv
from agent_transformer import SequentialDQNAgent, SequenceTracker
import json
import argparse

# Load configuration from JSON
def load_config(config_path="config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Test the Transformer DQN implementation')
parser.add_argument('--config', type=str, default='config_5node.json', help='Path to configuration JSON file')
args = parser.parse_args()

# Load config from specified file
config_path = args.config
print(f"Loading configuration from {config_path}")
config = load_config(config_path)

# Environment setup
num_nodes = config["num_nodes"]
adj_matrix = config["adj_matrix"]
edge_list = config["edge_list"]
node_props = config["node_props"]
tm_list = config["tm_list"]
link_capacity = config["link_capacity"]
max_edges = config["max_edges"]

# Environment instantiation
seed = 42
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

# State and action dimensions
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print(f"Topology: {env.num_edges} actual edges (max_edges={max_edges})")
print(f"State Dimension (Padded): {state_dim}")
print(f"Action Dimension: {action_dim}")

# Create transformer agent
device = torch.device("cpu")
sequence_length = 10
hidden_dim = 128

# Debug the input state
print("\n--- Initial State Information ---")
state, _ = env.reset()
print(f"State type: {type(state)}")
print(f"State shape: {np.array(state).shape}")
print(f"State sample (first 5 values): {np.array(state)[:5]}")

# Create agent and sequence tracker
print("\n--- Testing Sequence Tracker ---")
agent = SequentialDQNAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dim=hidden_dim,
    sequence_length=sequence_length,
    device=device
)

sequence_tracker = SequenceTracker(sequence_length, state_dim)
sequence_tracker.update(state)

# Get sequence
state_seq = sequence_tracker.get_state_sequence()
print(f"Sequence shape: {state_seq.shape}")

# Test action selection
print("\n--- Testing Action Selection ---")
action = agent.select_action(state, epsilon=0.5)
print(f"Selected action: {action}")

# Update with next state
print("\n--- Testing Environment Step ---")
next_state, reward, done, info = env.step(action)
print(f"Reward: {reward}")
print(f"Done: {done}")
print(f"Next state shape: {np.array(next_state).shape}")

# Update sequence
sequence_tracker.update(next_state, action)
next_seq = sequence_tracker.get_state_sequence()
action_seq = sequence_tracker.get_action_sequence()

print(f"Updated sequence shape: {next_seq.shape}")
if action_seq is not None:
    print(f"Action sequence: {action_seq}")
else:
    print("Action sequence is None")

# Test multiple steps
print("\n--- Testing Multiple Environment Steps ---")
for i in range(5):
    action = agent.select_action(next_state, epsilon=0.1)
    next_state, reward, done, info = env.step(action)
    sequence_tracker.update(next_state, action)
    
    print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Done={done}")
    
    if done:
        print("Environment completed early")
        break

# Test sequence predictions
print("\n--- Testing Batch Processing ---")
try:
    state_seq = sequence_tracker.get_state_sequence()
    action_seq = sequence_tracker.get_action_sequence()
    
    # Convert to tensors and add batch dimension
    state_batch = torch.FloatTensor(state_seq).unsqueeze(0).to(device)
    if action_seq is not None:
        action_batch = torch.LongTensor(action_seq).unsqueeze(0).to(device)
        print(f"State batch shape: {state_batch.shape}")
        print(f"Action batch shape: {action_batch.shape}")
        
        # Forward pass through network
        with torch.no_grad():
            q_values = agent.qnetwork_local(state_batch, action_batch)
            print(f"Q-values shape: {q_values.shape}")
            print(f"Q-values: {q_values.squeeze().detach().numpy()}")
    else:
        print("Not enough actions for batch processing test")
except Exception as e:
    print(f"Error in batch processing: {str(e)}")
    import traceback
    traceback.print_exc()

print("\nTransformer DQN test completed successfully")
