"""
Simple bruteforce algorithm to find optimal network configurations.
"""

import os
import sys
import json
import argparse
import numpy as np
import time
import itertools
from tqdm import tqdm

# Add parent directory to path to import modules
sys.path.append('../')
from env import NetworkEnv

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='Simple bruteforce algorithm')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--tm-index', type=int, default=0, help='Traffic matrix index to evaluate')
    parser.add_argument('--max-depth', type=int, default=4, help='Maximum number of links to close')
    args = parser.parse_args()
    
    # Load configuration
    config_path = args.config
    config = load_config(config_path)
    
    # Extract parameters from config
    num_nodes = config["num_nodes"]
    edge_list = config["edge_list"]
    tm_list = config["tm_list"]
    link_capacity = config.get("link_capacity", [1.0] * len(edge_list))
    node_props = config.get("node_props", {})
    
    # Create environment
    env = NetworkEnv(
        edge_list=edge_list,
        tm_list=tm_list,
        link_capacity=link_capacity,
        node_props=node_props,
        num_nodes=num_nodes
    )
    
    # Set traffic matrix
    tm_idx = args.tm_index
    if tm_idx >= len(tm_list):
        print(f"Error: Traffic matrix index {tm_idx} out of range (0-{len(tm_list)-1})")
        return
    
    env.current_tm_idx = tm_idx
    
    # Maximum number of links to close
    max_depth = min(args.max_depth, len(edge_list))
    
    # Bruteforce search
    best_reward = -float('inf')
    best_actions = []
    
    print(f"Bruteforce search for traffic matrix {tm_idx}...")
    print(f"Trying combinations with up to {max_depth} closed links out of {len(edge_list)} total links")
    
    # Try closing different numbers of links
    for depth in range(max_depth + 1):
        print(f"Testing combinations with {depth} closed links...")
        
        # Generate all combinations of actions
        for action_sequence in tqdm(itertools.combinations(range(len(edge_list)), depth)):
            # Reset environment
            state = env.reset()
            done = False
            total_reward = 0
            
            # Apply actions
            for action in action_sequence:
                if done:
                    break
                next_state, reward, done, truncated, info = env.step(action)
                total_reward += reward
            
            # Check if valid (no isolations or overloaded links)
            valid = not done and not any(usage > capacity for usage, capacity in zip(env.link_usage, env.link_capacity))
            
            # Update best result
            if valid and total_reward > best_reward:
                best_reward = total_reward
                best_actions = list(action_sequence)
    
    # Print results
    print(f"\nBest solution for traffic matrix {tm_idx}:")
    print(f"Actions: {best_actions}")
    print(f"Closed links: {len(best_actions)}/{len(edge_list)}")
    print(f"Reward: {best_reward}")
    
    # Check if a valid solution was found
    if best_reward > -float('inf'):
        closed_edges = [edge_list[i] for i in best_actions]
        print(f"Closed edges: {closed_edges}")
        print("Status: Success ✅")
    else:
        print("No valid solution found.")
        print("Status: Failed ❌")

if __name__ == "__main__":
    main()
