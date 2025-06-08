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
from latent_env import NetworkEnv

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
        adj_matrix=config["adj_matrix"],
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
            reset_result = env.reset()
            # Handle both old and new gym interfaces
            if isinstance(reset_result, tuple):
                state = reset_result[0]  # The observation is the first element
                # If the reset method returns (obs, 0, False, False, info)
                done = reset_result[2] if len(reset_result) > 2 else False
            else:
                state = reset_result
                done = False
            total_reward = 0
            
            # Apply actions to close specific edges
            # First, iterate through all edges
            for edge_idx in range(len(edge_list)):
                if done:
                    break
                    
                # Only close this edge if it's in our selected edges to close
                action = 0 if edge_idx in action_sequence else 1  # 0 = close, 1 = keep open
                
                # Set the current edge we're making a decision on
                env.current_edge_idx = edge_idx
                
                # Take action
                next_state, reward, done, truncated, info = env.step(action)
                total_reward += reward
                
                # Debug print for first few combinations to see what's happening
                if len(best_actions) == 0 and depth <= 1 and edge_idx == len(edge_list) - 1:  # Only for first few tests
                    print(f"  Debug - Trying {'closing' if action == 0 else 'keeping'} edge {edge_idx}: Reward: {reward}, Done: {done}, Info: {info}")
            
            # Check if valid (no isolations or overloaded links)
            # In your environment, 'done' already indicates a violation occurred
            valid = not done
            
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
        
        # Try just keeping all edges open to verify if that works
        print("\nTesting baseline (all links open):")
        reset_result = env.reset()
        # Handle both old and new gym interfaces
        if isinstance(reset_result, tuple):
            state = reset_result[0]  # The observation is the first element
            # If the reset method returns (obs, 0, False, False, info)
            done = reset_result[2] if len(reset_result) > 2 else False
        else:
            state = reset_result
            done = False
        total_reward = 0
        for edge_idx in range(len(edge_list)):
            env.current_edge_idx = edge_idx
            next_state, reward, done, truncated, info = env.step(1)  # 1 = keep open
            total_reward += reward
            if done:
                print(f"  Error at edge {edge_idx}: {info}")
                break
        print(f"  Baseline reward: {total_reward}, Done: {done}")

if __name__ == "__main__":
    main()
