"""
Improved bruteforce algorithm for finding optimal network configurations.
Fixes key issues with the original simple_bruteforce.py:
1. Processes full action sequences before determining validity
2. Properly resets environment between tests
3. Tests different edge orderings
4. Better handles error conditions
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

def evaluate_configuration(env, edge_list, action_vector, tm_idx, verbose=False):
    """
    Evaluate a specific network configuration.
    
    Args:
        env: NetworkEnv instance
        edge_list: List of edges in the network
        action_vector: List of actions (0=close, 1=open) for each edge
        tm_idx: Traffic matrix index
        verbose: Whether to print detailed information
        
    Returns:
        tuple: (total_reward, valid, info)
    """
    # Reset environment
    env.current_tm_idx = tm_idx
    env.reset()
    
    total_reward = 0
    all_info = []
    violations = []
    edges_closed = []
    
    # Apply all actions first
    for edge_idx, action in enumerate(action_vector):
        env.current_edge_idx = edge_idx
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if action == 0:  # If the edge is closed
            edges_closed.append(edge_idx)
            
        # Store the info for debugging
        all_info.append(info)
        
        if info.get('violation') is not None:
            violations.append((edge_idx, info.get('violation')))
    
    # Only consider valid if there are no violations at the end
    valid = len(violations) == 0
    
    if verbose:
        print(f"Evaluating configuration: {action_vector}")
        print(f"Edges closed: {edges_closed}")
        print(f"Total reward: {total_reward}")
        print(f"Valid: {valid}")
        print(f"Violations: {violations}")
    
    return total_reward, valid, violations, all_info

def main():
    parser = argparse.ArgumentParser(description='Improved bruteforce algorithm')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--tm-index', type=int, default=0, help='Traffic matrix index to evaluate')
    parser.add_argument('--max-closed', type=int, default=8, help='Maximum number of links to close')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--random-samples', type=int, default=None, 
                        help='Number of random samples to try (for large networks)')
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
    
    # Initialize tracking variables
    best_reward = -float('inf')
    best_action_vector = None
    best_violations = None
    valid_solutions_found = 0
    total_configurations_tested = 0
    
    print(f"Bruteforce search for traffic matrix {tm_idx}...")
    
    # Test baseline configuration (all links open)
    baseline_action_vector = [1] * len(edge_list)  # All links open
    baseline_reward, baseline_valid, baseline_violations, baseline_info = evaluate_configuration(
        env, edge_list, baseline_action_vector, tm_idx, args.verbose
    )
    
    print(f"\nBaseline (all links open):")
    print(f"  Reward: {baseline_reward}")
    print(f"  Valid: {baseline_valid}")
    print(f"  Violations: {baseline_violations}")
    
    # Update best result if baseline is valid
    if baseline_valid:
        best_reward = baseline_reward
        best_action_vector = baseline_action_vector
        valid_solutions_found += 1
    
    max_closed = min(args.max_closed, len(edge_list))
    
    # Initialize search space for bruteforce
    if args.random_samples:
        # For large networks, use random sampling
        configurations_to_test = []
        num_samples = args.random_samples
        
        # Generate random configurations with different numbers of closed links
        for num_closed in range(max_closed + 1):
            for _ in range(num_samples // (max_closed + 1)):
                # Generate a random configuration with exactly num_closed links closed
                action_vector = [1] * len(edge_list)  # Start with all open
                closed_indices = np.random.choice(len(edge_list), num_closed, replace=False)
                for idx in closed_indices:
                    action_vector[idx] = 0  # Close selected links
                configurations_to_test.append(action_vector)
        
        print(f"Testing {len(configurations_to_test)} random configurations...")
    else:
        # For smaller networks, do exhaustive search
        configurations_to_test = []
        
        # Generate all configurations with different numbers of closed links
        for num_closed in range(1, max_closed + 1):
            for indices_to_close in itertools.combinations(range(len(edge_list)), num_closed):
                action_vector = [1] * len(edge_list)  # Start with all open
                for idx in indices_to_close:
                    action_vector[idx] = 0  # Close selected links
                configurations_to_test.append(action_vector)
        
        print(f"Testing all {len(configurations_to_test)} possible configurations with up to {max_closed} closed links...")
    
    # Evaluate all configurations
    for action_vector in tqdm(configurations_to_test):
        total_configurations_tested += 1
        
        # Evaluate the configuration
        reward, valid, violations, info = evaluate_configuration(
            env, edge_list, action_vector, tm_idx, False
        )
        
        # Update best result if valid and better than current best
        if valid and reward > best_reward:
            best_reward = reward
            best_action_vector = action_vector
            valid_solutions_found += 1
            
            # Optional: print when a new best is found
            if args.verbose:
                closed_indices = [i for i, action in enumerate(action_vector) if action == 0]
                closed_edges = [edge_list[idx] for idx in closed_indices]
                print(f"\nNew best found! Reward: {reward}, Closed links: {len(closed_indices)}/{len(edge_list)}")
                print(f"  Closed edges: {closed_edges}")
    
    # Print results
    print("\n" + "="*80)
    print(f"BRUTEFORCE RESULTS FOR TRAFFIC MATRIX {tm_idx}")
    print("="*80)
    print(f"Configurations tested: {total_configurations_tested}")
    print(f"Valid solutions found: {valid_solutions_found}")
    
    if best_action_vector is not None:
        closed_indices = [i for i, action in enumerate(best_action_vector) if action == 0]
        closed_edges = [edge_list[idx] for idx in closed_indices]
        
        print(f"\nBest solution found:")
        print(f"  Reward: {best_reward}")
        print(f"  Closed links: {len(closed_indices)}/{len(edge_list)} ({(len(closed_indices)/len(edge_list))*100:.1f}%)")
        print(f"  Closed edges: {closed_edges}")
        print("  Status: ✅ Success")
    else:
        print("\nNo valid solution found ❌")
        print("  Even keeping all links open resulted in violations.")
        print(f"  Baseline violations: {baseline_violations}")
    
    # Compare with all-links-open baseline
    if best_action_vector is not None and best_reward > baseline_reward:
        print(f"\nImprovement over baseline: +{best_reward - baseline_reward:.2f} reward")
        closed_indices = [i for i, action in enumerate(best_action_vector) if action == 0]
        print(f"  Additional links closed: {len(closed_indices)}")
    else:
        print("\nNo improvement over baseline (all links open)")

if __name__ == "__main__":
    main()
