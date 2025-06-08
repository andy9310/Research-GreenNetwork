"""
Comprehensive bruteforce algorithm for finding optimal network configurations.
This script evaluates all traffic matrices in a configuration file and
compares results with the DQN's performance.
"""

import os
import sys
import json
import argparse
import numpy as np
import time
import itertools
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

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
        tuple: (total_reward, valid, violations, all_info)
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

def analyze_traffic_matrix(env, edge_list, tm_idx, max_closed, random_samples=None, verbose=False):
    """
    Analyze a single traffic matrix using bruteforce.
    
    Args:
        env: NetworkEnv instance
        edge_list: List of edges in the network
        tm_idx: Traffic matrix index
        max_closed: Maximum number of links to close
        random_samples: Number of random samples to use (for large networks)
        verbose: Whether to print detailed information
        
    Returns:
        dict: Results for this traffic matrix
    """
    # Initialize tracking variables
    best_reward = -float('inf')
    best_action_vector = None
    valid_solutions_found = 0
    total_configurations_tested = 0
    
    # Test baseline configuration (all links open)
    baseline_action_vector = [1] * len(edge_list)  # All links open
    baseline_reward, baseline_valid, baseline_violations, baseline_info = evaluate_configuration(
        env, edge_list, baseline_action_vector, tm_idx, verbose
    )
    
    # Update best result if baseline is valid
    if baseline_valid:
        best_reward = baseline_reward
        best_action_vector = baseline_action_vector
        valid_solutions_found += 1
    
    max_closed = min(max_closed, len(edge_list))
    
    # Initialize search space for bruteforce
    if random_samples:
        # For large networks, use random sampling
        configurations_to_test = []
        num_samples = random_samples
        
        # Generate random configurations with different numbers of closed links
        for num_closed in range(max_closed + 1):
            for _ in range(num_samples // (max_closed + 1)):
                # Generate a random configuration with exactly num_closed links closed
                action_vector = [1] * len(edge_list)  # Start with all open
                closed_indices = np.random.choice(len(edge_list), num_closed, replace=False)
                for idx in closed_indices:
                    action_vector[idx] = 0  # Close selected links
                configurations_to_test.append(action_vector)
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
    
    # Evaluate all configurations
    for action_vector in configurations_to_test:
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
    
    # Prepare results
    if best_action_vector is not None:
        closed_indices = [i for i, action in enumerate(best_action_vector) if action == 0]
        closed_edges = [edge_list[idx] for idx in closed_indices]
        improvement = best_reward - baseline_reward if baseline_valid else "N/A"
        status = "Success"
    else:
        closed_indices = []
        closed_edges = []
        improvement = "N/A"
        status = "Failed"
    
    return {
        "tm_idx": tm_idx,
        "configurations_tested": total_configurations_tested,
        "valid_solutions": valid_solutions_found,
        "best_reward": best_reward if best_reward > -float('inf') else None,
        "baseline_reward": baseline_reward if baseline_valid else None,
        "baseline_valid": baseline_valid,
        "num_links_closed": len(closed_indices),
        "total_links": len(edge_list),
        "closed_edges": closed_edges,
        "closed_indices": closed_indices,
        "status": status,
        "improvement": improvement
    }

def load_dqn_results(eval_file=None):
    """
    Load DQN evaluation results if available.
    
    Args:
        eval_file: Path to DQN evaluation results file
        
    Returns:
        dict: DQN results by traffic matrix, or None if not available
    """
    if eval_file and os.path.exists(eval_file):
        try:
            with open(eval_file, 'r') as f:
                return json.load(f)
        except:
            print(f"Warning: Could not load DQN results from {eval_file}")
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Comprehensive bruteforce algorithm')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--tm-indices', type=str, default=None, 
                        help='Comma-separated list of traffic matrix indices to evaluate (default: all)')
    parser.add_argument('--max-closed', type=int, default=8, help='Maximum number of links to close')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--random-samples', type=int, default=None, 
                        help='Number of random samples to try (for large networks)')
    parser.add_argument('--dqn-results', type=str, default=None,
                        help='Path to DQN evaluation results file (for comparison)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results (default: bruteforce_results_<timestamp>.json)')
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
        adj_matrix=config["adj_matrix"],
        edge_list=edge_list,
        tm_list=tm_list,
        link_capacity=link_capacity,
        node_props=node_props,
        num_nodes=num_nodes
    )
    
    # Determine which traffic matrices to evaluate
    if args.tm_indices:
        tm_indices = [int(idx) for idx in args.tm_indices.split(',')]
    else:
        tm_indices = list(range(len(tm_list)))
    
    # Load DQN results if available
    dqn_results = load_dqn_results(args.dqn_results)
    
    # Initialize results
    bruteforce_results = {
        "config": config_path,
        "config_name": config_name,
        "num_nodes": num_nodes,
        "num_edges": len(edge_list),
        "num_tms": len(tm_list),
        "max_closed_links": args.max_closed,
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "traffic_matrices": {}
    }
    
    # Process each traffic matrix
    print(f"Analyzing {len(tm_indices)} traffic matrices from {config_path}")
    print(f"Network: {num_nodes} nodes, {len(edge_list)} edges")
    
    for i, tm_idx in enumerate(tm_indices):
        if tm_idx >= len(tm_list):
            print(f"Error: Traffic matrix index {tm_idx} out of range (0-{len(tm_list)-1})")
            continue
        
        print(f"\n--- Processing Traffic Matrix {tm_idx} ({i+1}/{len(tm_indices)}) ---")
        
        # Run bruteforce analysis
        start_time = time.time()
        results = analyze_traffic_matrix(
            env, edge_list, tm_idx, args.max_closed, 
            args.random_samples, args.verbose
        )
        elapsed_time = time.time() - start_time
        
        # Add timing information
        results["analysis_time"] = elapsed_time
        
        # Add DQN comparison if available
        if dqn_results and str(tm_idx) in dqn_results:
            dqn_tm_results = dqn_results[str(tm_idx)]
            results["dqn_comparison"] = {
                "dqn_reward": dqn_tm_results.get("reward"),
                "dqn_links_closed": dqn_tm_results.get("links_closed"),
                "dqn_status": dqn_tm_results.get("status"),
                "comparison": "better" if results["best_reward"] and dqn_tm_results.get("reward") and 
                              results["best_reward"] > dqn_tm_results.get("reward") else
                              "worse" if results["best_reward"] and dqn_tm_results.get("reward") and 
                              results["best_reward"] < dqn_tm_results.get("reward") else
                              "equal" if results["best_reward"] and dqn_tm_results.get("reward") else
                              "unknown"
            }
        
        # Add to overall results
        bruteforce_results["traffic_matrices"][tm_idx] = results
        
        # Print summary for this traffic matrix
        print(f"Completed in {elapsed_time:.2f} seconds")
        print(f"Configurations tested: {results['configurations_tested']}")
        print(f"Valid solutions found: {results['valid_solutions']}")
        
        if results["best_reward"] is not None:
            print(f"Best solution: Reward {results['best_reward']}, " +
                  f"{results['num_links_closed']}/{results['total_links']} links closed")
            if results.get("dqn_comparison"):
                comp = results["dqn_comparison"]
                print(f"DQN comparison: DQN achieved {comp['dqn_reward']} reward with " +
                      f"{comp['dqn_links_closed']} links closed")
                print(f"Bruteforce is {comp['comparison']} than DQN")
        else:
            print("No valid solution found")
    
    # Calculate overall statistics
    tm_results = list(bruteforce_results["traffic_matrices"].values())
    
    # Filter out None values for reward calculations
    valid_rewards = [r["best_reward"] for r in tm_results if r["best_reward"] is not None]
    links_closed = [r["num_links_closed"] for r in tm_results if r["best_reward"] is not None]
    success_count = sum(1 for r in tm_results if r["status"] == "Success")
    
    # Only calculate statistics if there are valid solutions
    if valid_rewards:
        bruteforce_results["summary"] = {
            "avg_reward": sum(valid_rewards) / len(valid_rewards) if valid_rewards else 0,
            "avg_links_closed": sum(links_closed) / len(links_closed) if links_closed else 0,
            "success_rate": success_count / len(tm_results) * 100
        }
        
        # Add DQN comparison summary if available
        if dqn_results:
            dqn_comparisons = [r.get("dqn_comparison", {}) for r in tm_results if r.get("dqn_comparison")]
            better_count = sum(1 for c in dqn_comparisons if c.get("comparison") == "better")
            equal_count = sum(1 for c in dqn_comparisons if c.get("comparison") == "equal")
            worse_count = sum(1 for c in dqn_comparisons if c.get("comparison") == "worse")
            
            bruteforce_results["dqn_summary"] = {
                "better_than_dqn": better_count,
                "equal_to_dqn": equal_count,
                "worse_than_dqn": worse_count,
                "better_percentage": better_count / len(dqn_comparisons) * 100 if dqn_comparisons else 0
            }
    
    # Save results to file
    output_file = args.output or f"bruteforce_results_{config_name}_{bruteforce_results['timestamp']}.json"
    with open(output_file, 'w') as f:
        json.dump(bruteforce_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Generate summary CSV file
    csv_file = output_file.replace(".json", ".csv")
    
    # Prepare data for CSV
    data = []
    for tm_idx, result in bruteforce_results["traffic_matrices"].items():
        row = {
            "TM": tm_idx,
            "Bruteforce_Reward": result.get("best_reward"),
            "Bruteforce_Links_Closed": result.get("num_links_closed"),
            "Bruteforce_Status": result.get("status")
        }
        
        # Add DQN comparison if available
        if result.get("dqn_comparison"):
            row.update({
                "DQN_Reward": result["dqn_comparison"].get("dqn_reward"),
                "DQN_Links_Closed": result["dqn_comparison"].get("dqn_links_closed"),
                "DQN_Status": result["dqn_comparison"].get("dqn_status"),
                "Comparison": result["dqn_comparison"].get("comparison")
            })
        
        data.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    print(f"CSV summary saved to {csv_file}")
    
    # Print overall summary
    print("\n" + "="*80)
    print(f"OVERALL BRUTEFORCE RESULTS FOR {config_name}")
    print("="*80)
    
    if "summary" in bruteforce_results:
        print(f"Traffic matrices analyzed: {len(tm_results)}")
        print(f"Success rate: {bruteforce_results['summary']['success_rate']:.1f}%")
        print(f"Average reward: {bruteforce_results['summary']['avg_reward']:.2f}")
        print(f"Average links closed: {bruteforce_results['summary']['avg_links_closed']:.2f}/{len(edge_list)}")
        
        if "dqn_summary" in bruteforce_results:
            print("\nComparison with DQN:")
            print(f"  Better than DQN: {bruteforce_results['dqn_summary']['better_than_dqn']} " +
                  f"({bruteforce_results['dqn_summary']['better_percentage']:.1f}%)")
            print(f"  Equal to DQN: {bruteforce_results['dqn_summary']['equal_to_dqn']}")
            print(f"  Worse than DQN: {bruteforce_results['dqn_summary']['worse_than_dqn']}")
    else:
        print("No valid solutions found in any traffic matrix")
    
    # Generate plots if pandas and matplotlib are available
    try:
        # Prepare DQN comparison data if available
        if "dqn_summary" in bruteforce_results:
            # Plot comparison of rewards
            plt.figure(figsize=(12, 8))
            tm_indices = []
            bf_rewards = []
            dqn_rewards = []
            
            for tm_idx, result in bruteforce_results["traffic_matrices"].items():
                if result.get("dqn_comparison") and result["best_reward"] is not None:
                    tm_indices.append(int(tm_idx))
                    bf_rewards.append(result["best_reward"])
                    dqn_rewards.append(result["dqn_comparison"]["dqn_reward"])
            
            # Sort by TM index
            sorted_indices = sorted(range(len(tm_indices)), key=lambda i: tm_indices[i])
            tm_indices = [tm_indices[i] for i in sorted_indices]
            bf_rewards = [bf_rewards[i] for i in sorted_indices]
            dqn_rewards = [dqn_rewards[i] for i in sorted_indices]
            
            plt.bar(np.array(tm_indices) - 0.2, bf_rewards, width=0.4, label='Bruteforce', color='blue')
            plt.bar(np.array(tm_indices) + 0.2, dqn_rewards, width=0.4, label='DQN', color='green')
            plt.xlabel('Traffic Matrix Index')
            plt.ylabel('Reward')
            plt.title(f'Bruteforce vs DQN Rewards ({config_name})')
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(tm_indices)
            plt.savefig(output_file.replace(".json", "_reward_comparison.png"))
            print(f"Reward comparison plot saved to {output_file.replace('.json', '_reward_comparison.png')}")
            
            # Plot comparison of links closed
            plt.figure(figsize=(12, 8))
            tm_indices = []
            bf_links = []
            dqn_links = []
            
            for tm_idx, result in bruteforce_results["traffic_matrices"].items():
                if result.get("dqn_comparison") and result["best_reward"] is not None:
                    tm_indices.append(int(tm_idx))
                    bf_links.append(result["num_links_closed"])
                    dqn_links.append(result["dqn_comparison"]["dqn_links_closed"])
            
            # Sort by TM index
            sorted_indices = sorted(range(len(tm_indices)), key=lambda i: tm_indices[i])
            tm_indices = [tm_indices[i] for i in sorted_indices]
            bf_links = [bf_links[i] for i in sorted_indices]
            dqn_links = [dqn_links[i] for i in sorted_indices]
            
            plt.bar(np.array(tm_indices) - 0.2, bf_links, width=0.4, label='Bruteforce', color='blue')
            plt.bar(np.array(tm_indices) + 0.2, dqn_links, width=0.4, label='DQN', color='green')
            plt.xlabel('Traffic Matrix Index')
            plt.ylabel('Links Closed')
            plt.title(f'Bruteforce vs DQN Links Closed ({config_name})')
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(tm_indices)
            plt.savefig(output_file.replace(".json", "_links_comparison.png"))
            print(f"Links closed comparison plot saved to {output_file.replace('.json', '_links_comparison.png')}")
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")

if __name__ == "__main__":
    main()
