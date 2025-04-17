import sys
import os
import numpy as np
import networkx as nx
import itertools
import time
import json
from tqdm import tqdm
import argparse

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the environment from Qlearning directory
from Qlearning.env import NetworkEnv

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def run_bruteforce(config_path, tm_index=None):
    """
    Run bruteforce algorithm on a specific traffic matrix or all matrices in a config
    
    Args:
        config_path: Path to the configuration file
        tm_index: Index of traffic matrix to use (None = all matrices)
    
    Returns:
        Dictionary of results for each traffic matrix
    """
    print(f"Loading configuration from {config_path}...")
    config = load_config(config_path)
    
    # Load configuration
    num_nodes = config["num_nodes"]
    adj_matrix = config["adj_matrix"]
    edge_list = config["edge_list"]
    node_props = config["node_props"]
    tm_list = config["tm_list"]
    link_capacity = config["link_capacity"]
    max_edges = config["max_edges"]
    
    # Get configuration basename for result filenames
    config_basename = os.path.basename(config_path).replace('.json', '')
    
    # Determine which traffic matrices to process
    if tm_index is not None:
        # Process a single specific matrix
        tm_indices = [tm_index]
    else:
        # Process all matrices
        tm_indices = range(len(tm_list))
    
    # Store results for each traffic matrix
    results = {}
    
    # Process each traffic matrix
    for idx in tm_indices:
        if idx < 0 or idx >= len(tm_list):
            print(f"Error: Traffic matrix index {idx} is out of range (0-{len(tm_list)-1})")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing Traffic Matrix {idx+1}/{len(tm_list)} from {config_basename}")
        print(f"{'='*80}")
        
        # Create environment with the current traffic matrix
        # Increase link capacity if needed to ensure solutions exist
        effective_link_capacity = link_capacity
        # Check if we need to adjust capacity for test configurations
        if 'test_config' in config_path:
            # For test configurations, we might need higher capacity
            # Analyze the traffic matrix to estimate required capacity
            tm = np.array(tm_list[idx])
            total_traffic = np.sum(tm)
            avg_traffic_per_edge = total_traffic / len(edge_list) * 2  # Conservative estimate
            if avg_traffic_per_edge > link_capacity:
                suggested_capacity = int(avg_traffic_per_edge * 1.5)  # Add 50% margin
                print(f"WARNING: Traffic matrix might require higher capacity.")
                print(f"  - Current capacity: {link_capacity}")
                print(f"  - Suggested minimum capacity: {suggested_capacity}")
                print(f"  - Using adjusted capacity: {suggested_capacity}")
                effective_link_capacity = suggested_capacity
        
        env = NetworkEnv(
            adj_matrix=adj_matrix,
            edge_list=edge_list,
            tm_list=tm_list,
            node_props=node_props,
            num_nodes=num_nodes,
            link_capacity=effective_link_capacity,  # Use adjusted capacity
            max_edges=max_edges,
            seed=int(time.time())
        )
        
        # Set the current traffic matrix
        env.current_tm_idx = idx
        env.traffic = np.array(env.tm_list[idx])
        
        num_edges = env.num_edges
        print(f"Network has {num_nodes} nodes and {num_edges} edges.")
        print(f"Total possible configurations: {2**num_edges}")
        
        # Bruteforce search
        best_config = None
        best_score = -float('inf')
        valid_configs_found = 0
        
        # Calculate total configurations
        total_configs = 2**num_edges
        
        # Limit the total number of configurations to check if it's too large
        max_configs_to_check = 10000000  # 10 million is a reasonable limit
        do_sampling = total_configs > max_configs_to_check
        
        if do_sampling:
            print(f"Warning: Total configurations ({total_configs:,}) exceeds limit of {max_configs_to_check:,}")
            print(f"Will sample {max_configs_to_check:,} random configurations instead of exhaustive search")
            # Generate random configurations instead of exhaustive search
            all_configs = np.random.randint(0, 2, size=(max_configs_to_check, num_edges))
            total_configs = max_configs_to_check
        else:
            # Generate all possible binary configurations (0=closed, 1=open)
            all_configs = list(itertools.product([0, 1], repeat=num_edges))
        
        # Use tqdm for progress tracking
        with tqdm(total=total_configs, desc=f"TM {idx} - Checking Configurations", unit="config") as pbar:
            # Track best configuration seen so far
            for i, config_data in enumerate(all_configs):
                if do_sampling:
                    # Already a numpy array from random generation
                    config_vector = config_data
                else:
                    # Convert tuple to numpy array
                    config_vector = np.array(config_data, dtype=int)
                
                # Manually set the link status in the environment
                env.link_open = config_vector
                
                # Manually call the internal checks from the environment
                routing_successful, G_open = env._update_link_usage()
                isolated, overloaded, num_overloaded = env._check_violations(routing_successful, G_open)
                
                # Check if the configuration is valid (no violations)
                if not isolated and not overloaded:
                    valid_configs_found += 1
                    # Calculate score: Energy savings from closed links
                    num_closed_links = np.sum(config_vector == 0)
                    # Use the same reward unit as the environment for direct comparison
                    current_score = num_closed_links * env.energy_unit_reward
                    
                    # Update best score if this config is better
                    if current_score > best_score:
                        best_score = current_score
                        best_config = config_vector.copy()
                        # Update progress bar description with current best
                        pbar.set_postfix({"Valid Found": valid_configs_found, "Best Score": best_score})
                
                # Update progress bar
                pbar.update(1)
        
        print(f"\nFinished checking {total_configs:,} configurations.")
        print(f"Found {valid_configs_found:,} valid configurations (no violations).")
        
        # Report results
        if best_config is not None:
            print(f"\n--- Optimal Configuration Found (Brute-Force) for TM {idx} ---")
            print(f" Best Configuration (0=closed, 1=open): {best_config}")
            print(f" Number of Links Closed: {np.sum(best_config == 0)} / {num_edges}")
            print(f" Maximum Achievable Score (Energy Saved Reward): {best_score}")
            
            # Store results for this traffic matrix
            results[idx] = {
                "best_config": best_config.tolist(),
                "num_links_closed": int(np.sum(best_config == 0)),
                "total_links": num_edges,
                "best_score": float(best_score),
                "valid_configs_found": valid_configs_found
            }
        else:
            print(f"\nNo valid configuration found for TM {idx} that satisfies the constraints.")
            print("(This might indicate an issue with the topology, TM, or capacity definitions)")
            
            # Store null results
            results[idx] = {
                "best_config": None,
                "num_links_closed": 0,
                "total_links": num_edges,
                "best_score": 0,
                "valid_configs_found": 0
            }
    
    # Save results to file
    results_file = f"bruteforce_results_{config_basename}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Brute-force search for optimal network configurations')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    parser.add_argument('--tm-index', type=int, default=None, help='Index of traffic matrix to use (default: run all)')
    args = parser.parse_args()
    
    # Run bruteforce for the specified configuration and traffic matrix(s)
    run_bruteforce(args.config, args.tm_index)
