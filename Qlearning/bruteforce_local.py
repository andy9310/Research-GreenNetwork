import networkx as nx
import itertools
import time
import json
from tqdm import tqdm
import argparse
import sys
import os
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env import NetworkEnv

# --- Load Configuration from JSON ---
def load_config(config_path="config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def evaluate_configuration(env, link_status):
    """
    Evaluate a given link configuration.
    
    Args:
        env: The network environment
        link_status: List of binary values (0/1) for each link
        
    Returns:
        reward: The reward received
        isolated: Whether any node is isolated
        overloaded: Whether any link is overloaded
    """
    # Apply link status to environment
    env.reset()  # Reset environment
    
    # Set link status one by one
    for i, status in enumerate(link_status):
        if status == 0:  # Close link
            _, _, done, _, _ = env.step(i)
            if done:
                # If closing this link would make network invalid, keep it open
                return -float('inf'), True, False
    
    # Evaluate configuration
    reward = 0
    for i, status in enumerate(link_status):
        if status == 0:  # Closed link
            reward += 10  # Reward for each closed link
    
    # Check if network is valid
    isolated = not env.is_connected()
    overloaded = any(usage > capacity for usage, capacity in zip(env.link_usage, env.link_capacity))
    
    if isolated:
        reward -= 100  # Large penalty for isolated nodes
    
    if overloaded:
        reward -= 100  # Large penalty for overloaded links
    
    return reward, isolated, overloaded

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Brute-force search for optimal network configuration')
    parser.add_argument('--config', type=str, default='config.json', help='Path to configuration JSON file')
    parser.add_argument('--tm-index', type=int, default=None, help='Index of traffic matrix to use (default: all)')
    parser.add_argument('--max-links-to-close', type=int, default=None, help='Maximum number of links to consider closing')
    parser.add_argument('--random-sampling', action='store_true', help='Use random sampling instead of exhaustive search')
    parser.add_argument('--sample-size', type=int, default=10000, help='Number of samples for random sampling')
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
    
    # Determine which traffic matrices to evaluate
    if args.tm_index is not None:
        tm_indices = [args.tm_index]
    else:
        tm_indices = list(range(len(tm_list)))
    
    print(f"Loaded configuration with {num_nodes} nodes, {len(edge_list)} edges, and {len(tm_list)} traffic matrices.")
    
    # Max number of links to close
    max_edges = len(edge_list)
    if args.max_links_to_close:
        max_links_to_close = min(args.max_links_to_close, max_edges)
    else:
        max_links_to_close = max_edges // 2  # Default: try closing up to half the links
    
    print(f"Will try closing up to {max_links_to_close} links out of {max_edges} total links.")
    
    # Store results for each traffic matrix
    results = []
    
    # Evaluate each traffic matrix
    for tm_idx in tm_indices:
        if tm_idx >= len(tm_list):
            print(f"Traffic matrix index {tm_idx} out of range (0-{len(tm_list)-1}).")
            continue
            
        print(f"\n--- Traffic Matrix {tm_idx} ---")
        
        # Create environment with this traffic matrix
        env = NetworkEnv(
            edge_list=edge_list,
            tm=tm_list[tm_idx],
            link_capacity=link_capacity,
            node_props=node_props,
            num_nodes=num_nodes
        )
        
        best_reward = -float('inf')
        best_config = None
        best_isolated = True
        best_overloaded = True
        best_closed_count = 0
        
        # Base configuration: all links open
        base_link_status = [1] * max_edges
        base_reward, base_isolated, base_overloaded = evaluate_configuration(env, base_link_status)
        
        start_time = time.time()
        total_configs = 0
        
        if args.random_sampling:
            # Random sampling approach for very large networks
            print(f"Using random sampling with {args.sample_size} samples...")
            
            for _ in tqdm(range(args.sample_size)):
                # Generate random configuration (decide how many links to close)
                num_to_close = np.random.randint(0, max_links_to_close + 1)
                
                # Randomly choose which links to close
                indices_to_close = np.random.choice(max_edges, num_to_close, replace=False)
                link_status = base_link_status.copy()
                for idx in indices_to_close:
                    link_status[idx] = 0
                
                # Evaluate this configuration
                reward, isolated, overloaded = evaluate_configuration(env, link_status)
                total_configs += 1
                
                # Update best configuration if better
                if reward > best_reward and not isolated and not overloaded:
                    best_reward = reward
                    best_config = link_status.copy()
                    best_isolated = isolated
                    best_overloaded = overloaded
                    best_closed_count = num_to_close
                
        else:
            # Exhaustive search approach
            print("Using exhaustive search...")
            
            # Try closing different numbers of links, from 0 to max_links_to_close
            for num_to_close in range(max_links_to_close + 1):
                print(f"Trying configurations with {num_to_close} closed links...")
                
                # Generate all combinations of links to close
                for indices_to_close in tqdm(itertools.combinations(range(max_edges), num_to_close)):
                    # Create configuration with these links closed
                    link_status = base_link_status.copy()
                    for idx in indices_to_close:
                        link_status[idx] = 0
                    
                    # Evaluate this configuration
                    reward, isolated, overloaded = evaluate_configuration(env, link_status)
                    total_configs += 1
                    
                    # Update best configuration if better
                    if reward > best_reward and not isolated and not overloaded:
                        best_reward = reward
                        best_config = link_status.copy()
                        best_isolated = isolated
                        best_overloaded = overloaded
                        best_closed_count = num_to_close
        
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
        
        if best_config:
            closed_indices = [i for i, status in enumerate(best_config) if status == 0]
            closed_edges = [edge_list[i] for i in closed_indices]
            print(f"  Closed Links: {best_closed_count}/{max_edges}")
            print(f"  Closed Link Indices: {closed_indices}")
            print(f"  Closed Edges: {closed_edges}")
        else:
            print("  No valid configuration found.")
        
        print(f"  Evaluation Time: {elapsed_time:.2f} seconds")
        print(f"  Configurations Evaluated: {total_configs}")
    
    # Print overall results
    if results:
        print("\n=== Overall Results ===")
        avg_reward = sum(r["reward"] for r in results) / len(results)
        avg_links_closed = sum(r["closed_links"] for r in results) / len(results)
        success_count = sum(1 for r in results if not r["isolated"] and not r["overloaded"])
        
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Links Closed: {avg_links_closed:.2f} / {max_edges}")
        print(f"Successful Traffic Matrices: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
        
        # Save results to file
        results_file = f"bruteforce_results_{config_name}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()
