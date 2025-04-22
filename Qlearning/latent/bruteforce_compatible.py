"""
Bruteforce algorithm to find optimal network configurations
Compatible with the existing NetworkEnv implementation
"""

import os
import sys
import json
import argparse
import numpy as np
import time
from tqdm import tqdm
import itertools

# Add parent directory to path to import modules
sys.path.append('../')
from env import NetworkEnv

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def evaluate_configuration(env, link_status, tm_idx=0):
    """
    Evaluate a given link configuration.
    
    Args:
        env: The network environment
        link_status: List of binary values (0/1) for each link
        tm_idx: Index of traffic matrix to evaluate
        
    Returns:
        reward: The reward received
        isolated: Whether any node is isolated
        overloaded: Whether any link is overloaded
        closed_count: Number of closed links
    """
    # Set traffic matrix
    env.current_tm_idx = tm_idx
    
    # Reset environment
    state = env.reset()
    
    # Close links according to link_status
    closed_count = 0
    for i, status in enumerate(link_status):
        if status == 0:  # Close link
            closed_count += 1
            next_state, reward, done, truncated, info = env.step(i)
            if done:
                # If closing this link would make network invalid
                return -float('inf'), True, False, closed_count
    
    # Check network state
    has_isolated = not env.is_connected()
    has_overloaded = any(u > c for u, c in zip(env.link_usage, env.link_capacity))
    
    # Calculate reward
    reward = closed_count * 10  # 10 reward per closed link
    
    if has_isolated:
        reward -= 100  # Large penalty for isolated nodes
    
    if has_overloaded:
        reward -= 20  # Penalty for overloaded links
    
    return reward, has_isolated, has_overloaded, closed_count

def main():
    parser = argparse.ArgumentParser(description='Bruteforce algorithm for finding optimal network configurations')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--tm-indices', type=str, help='Comma-separated list of traffic matrix indices to evaluate')
    parser.add_argument('--max-links-to-close', type=int, help='Maximum number of links to consider closing')
    parser.add_argument('--random-sampling', action='store_true', help='Use random sampling instead of exhaustive search')
    parser.add_argument('--sample-size', type=int, default=5000, help='Number of samples for random sampling')
    args = parser.parse_args()
    
    # Load configuration
    config_path = args.config
    config = load_config(config_path)
    config_name = os.path.basename(config_path).split('.')[0]
    
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
    
    # Specific traffic matrices to evaluate
    if args.tm_indices:
        tm_indices = [int(idx) for idx in args.tm_indices.split(',')]
    else:
        tm_indices = list(range(len(tm_list)))
    
    print(f"Loaded configuration with {num_nodes} nodes, {len(edge_list)} edges, and {len(tm_list)} traffic matrices")
    
    # Max number of links to close
    max_edges = len(edge_list)
    if args.max_links_to_close:
        max_links_to_close = min(args.max_links_to_close, max_edges)
    else:
        max_links_to_close = max_edges // 2  # Default: try closing up to half the links
    
    print(f"Will try closing up to {max_links_to_close} links out of {max_edges} total links")
    
    # Store results for each traffic matrix
    results = []
    
    # Evaluate each traffic matrix
    for tm_idx in tm_indices:
        print(f"\n--- Evaluating Traffic Matrix {tm_idx} ---")
        start_time = time.time()
        
        best_reward = -float('inf')
        best_config = None
        best_isolated = True
        best_overloaded = True
        best_closed_count = 0
        total_configs = 0
        
        if args.random_sampling:
            # Random sampling approach
            print(f"Using random sampling with {args.sample_size} samples...")
            
            for _ in tqdm(range(args.sample_size)):
                # Generate random configuration
                link_status = np.random.randint(0, 2, max_edges)
                
                # Ensure we don't close too many links
                if sum(1 for s in link_status if s == 0) > max_links_to_close:
                    indices_to_close = np.random.choice(
                        [i for i, s in enumerate(link_status) if s == 0],
                        sum(1 for s in link_status if s == 0) - max_links_to_close,
                        replace=False
                    )
                    for idx in indices_to_close:
                        link_status[idx] = 1
                
                # Evaluate this configuration
                reward, isolated, overloaded, closed_count = evaluate_configuration(env, link_status, tm_idx)
                total_configs += 1
                
                # Update best configuration if better
                if reward > best_reward and not isolated:
                    best_reward = reward
                    best_config = link_status.copy()
                    best_isolated = isolated
                    best_overloaded = overloaded
                    best_closed_count = closed_count
        
        else:
            # Exhaustive search approach
            print("Using exhaustive search...")
            
            # Try closing different numbers of links
            for num_to_close in range(max_links_to_close + 1):
                print(f"Trying configurations with {num_to_close} closed links...")
                
                # Generate all combinations of links to close
                for indices_to_close in tqdm(itertools.combinations(range(max_edges), num_to_close)):
                    # Create configuration with these links closed
                    link_status = np.ones(max_edges, dtype=int)
                    for idx in indices_to_close:
                        link_status[idx] = 0
                    
                    # Evaluate this configuration
                    reward, isolated, overloaded, closed_count = evaluate_configuration(env, link_status, tm_idx)
                    total_configs += 1
                    
                    # Update best configuration if better
                    if reward > best_reward and not isolated:
                        best_reward = reward
                        best_config = link_status.copy()
                        best_isolated = isolated
                        best_overloaded = overloaded
                        best_closed_count = closed_count
        
        elapsed_time = time.time() - start_time
        
        # Store results for this traffic matrix
        result = {
            "tm_idx": tm_idx,
            "reward": best_reward,
            "closed_links": best_closed_count,
            "total_links": max_edges,
            "isolated": best_isolated,
            "overloaded": best_overloaded,
            "elapsed_time": elapsed_time,
            "configs_evaluated": total_configs
        }
        results.append(result)
        
        # Print results for this traffic matrix
        print(f"\nTraffic Matrix {tm_idx} Results:")
        print(f"  Best Reward: {best_reward}")
        
        if best_config is not None:
            closed_indices = [i for i, status in enumerate(best_config) if status == 0]
            closed_edges = [edge_list[i] for i in closed_indices]
            print(f"  Closed Links: {best_closed_count}/{max_edges}")
            print(f"  Closed Link Indices: {closed_indices}")
            print(f"  Closed Edges: {closed_edges}")
            print(f"  Has Violations: {best_overloaded}")
        else:
            print("  No valid configuration found")
        
        print(f"  Evaluation Time: {elapsed_time:.2f} seconds")
        print(f"  Configurations Evaluated: {total_configs}")
    
    # Print overall results
    if results:
        print("\n=== Overall Bruteforce Results ===")
        avg_reward = sum(r["reward"] for r in results) / len(results)
        avg_links_closed = sum(r["closed_links"] for r in results) / len(results)
        violation_count = sum(1 for r in results if r["overloaded"])
        success_rate = 100 * (len(results) - violation_count) / len(results)
        
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Links Closed: {avg_links_closed:.2f} / {max_edges}")
        print(f"Traffic Matrices with Violations: {violation_count}/{len(results)} ({100-success_rate:.1f}%)")
        print(f"Success Rate: {success_rate:.1f}%")
        
        # Save results to file
        results_file = f"bruteforce_results_{config_name}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()
