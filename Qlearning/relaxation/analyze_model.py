import os
import json
import torch
import numpy as np
import argparse
from Qlearning.relaxation.env import RelaxedNetworkEnv
from Qlearning.relaxation.ddpg_agent import DDPGAgent

def load_config(config_path):
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def find_latest_model_files(config_name):
    """Find the latest model files for a given configuration."""
    model_dir = os.path.join(os.getcwd(), "models")
    
    # List all files in the model directory matching the pattern
    pattern = os.path.join(model_dir, f"relaxed_{config_name}")
    matching_files = [f for f in os.listdir(model_dir) if pattern in os.path.join(model_dir, f)]
    
    # Sort to find the latest episode
    episodes = []
    for file in matching_files:
        if "episode" in file:
            try:
                episode = file.split("episode")[1].split("_")[0]
                if episode == "final":
                    episodes.append(("final", float('inf')))
                else:
                    episodes.append((episode, int(episode)))
            except:
                pass
    
    # If no models with episode numbers found, return None
    if not episodes:
        print("No model files found.")
        return None, None, None
    
    # Find the latest episode
    latest_episode = max(episodes, key=lambda x: x[1])[0]
    
    # Find corresponding model files
    actor_path = None
    critic_path = None
    predictor_path = None
    
    for file in matching_files:
        if f"episode{latest_episode}_actor" in file:
            actor_path = os.path.join(model_dir, file)
        elif f"episode{latest_episode}_critic" in file:
            critic_path = os.path.join(model_dir, file)
        elif f"episode{latest_episode}_predictor" in file:
            predictor_path = os.path.join(model_dir, file)
    
    print(f"Found latest model files from episode {latest_episode}:")
    print(f"Actor: {actor_path}")
    print(f"Critic: {critic_path}")
    print(f"Predictor: {predictor_path}")
    
    return actor_path, critic_path, predictor_path

def print_capacity_decisions(env, factors, tm_idx):
    """
    Print detailed information about capacity decisions made by the model.
    """
    print(f"\n--- Detailed Capacity Decisions for TM {tm_idx} ---")
    print(f"{'Link':>10} | {'From':>5} | {'To':>5} | {'Capacity':>10} | {'Factor':>10} | {'Effective Cap':>15} | {'Usage':>10} | {'Utilization':>15}")
    print("-" * 90)
    
    for i, (u, v) in enumerate(env.edge_list):
        capacity = env.link_capacity[i] if isinstance(env.link_capacity, list) else env.link_capacity
        factor = factors[i]
        effective_cap = capacity * factor
        usage = env.link_usage[i]
        utilization = usage / effective_cap if effective_cap > 0 else float('inf')
        
        status = "OPEN" if factor > 0.01 else "CLOSED"
        
        print(f"{i:>10} | {u:>5} | {v:>5} | {capacity:>10.2f} | {factor:>10.4f} | {effective_cap:>15.2f} | {usage:>10.2f} | {utilization:>15.4f} | {status}")
    
    # Calculate energy savings
    total_capacity = sum(env.link_capacity) if isinstance(env.link_capacity, list) else env.link_capacity * env.num_edges
    used_capacity = sum(env.link_capacity[i] * factors[i] for i in range(env.num_edges)) if isinstance(env.link_capacity, list) else sum(env.link_capacity * factors[i] for i in range(env.num_edges))
    saved_capacity = total_capacity - used_capacity
    saved_percent = (saved_capacity / total_capacity) * 100 if total_capacity > 0 else 0
    
    print(f"\nTotal Capacity: {total_capacity:.2f}")
    print(f"Used Capacity: {used_capacity:.2f}")
    print(f"Saved Capacity: {saved_capacity:.2f} ({saved_percent:.2f}%)")
    
    # Decision summary
    fully_open = sum(1 for f in factors if f > 0.99)
    partial = sum(1 for f in factors if 0.01 < f < 0.99)
    closed = sum(1 for f in factors if f <= 0.01)
    
    print(f"\nDecision Summary:")
    print(f"Fully Open Links: {fully_open}/{env.num_edges}")
    print(f"Partially Open Links: {partial}/{env.num_edges}")
    print(f"Closed Links: {closed}/{env.num_edges}")

def main():
    parser = argparse.ArgumentParser(description='Analyze relaxation model capacity decisions')
    parser.add_argument('--config', default='configs/config_5node_small.json', help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config_path = args.config
    config = load_config(config_path)
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    print(f"Loaded configuration from {config_path}")
    
    # Create environment
    env = RelaxedNetworkEnv(
        adj_matrix=config['adj_matrix'],
        edge_list=config['edge_list'],
        tm_list=config['tm_list'],
        node_props=config['node_props'],
        num_nodes=config['num_nodes'],
        link_capacity=config.get('link_capacity', 1.0),
        seed=config.get('seed', None),
        random_edge_order=config.get('random_edge_order', False)
    )
    print(f"Created environment with {env.num_nodes} nodes, {env.num_edges} edges, and {len(env.tm_list)} traffic matrices")
    
    # Find latest model files
    actor_path, critic_path, predictor_path = find_latest_model_files(config_name)
    if not actor_path or not critic_path:
        print("Missing model files. Cannot analyze model decisions.")
        return
    
    # Initialize device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize agent
    state_dim = env.observation_space.shape[0]
    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=1,  # One action per edge
        latent_dim=64,
        hidden_dim=256,
        device=device
    )
    print(f"State dimension: {state_dim}")
    
    # Load trained model
    agent.load(actor_path, critic_path, predictor_path)
    print("Model loaded successfully.")
    
    # Analyze each traffic matrix
    for tm_idx in range(env.num_tm):
        print(f"\n\n{'='*80}\nAnalyzing Traffic Matrix {tm_idx}\n{'='*80}")
        
        # Reset environment with the traffic matrix
        env.current_tm_idx = tm_idx
        state = env.reset()
        
        # Record link decisions
        capacity_factors = []
        done = False
        
        # Get decisions from the model
        while not done:
            # Select action without noise (deterministic policy)
            action = agent.act(state, add_noise=False)
            
            # Store capacity factor
            capacity_factors.append(float(action[0]))
            
            # Take action in environment
            next_state, reward, done, truncated, info = env.step(action)
            
            # Update state
            state = next_state
            
            # Break if done for any reason (e.g., violation)
            if done:
                break
        
        # Print decisions
        print_capacity_decisions(env, env.link_factors, tm_idx)

if __name__ == "__main__":
    main()
