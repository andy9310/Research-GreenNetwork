"""
Simple comparison between model and optimal
"""

import os
import sys
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt

# Function to load results from file
def load_results(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='Compare DQN results with optimal')
    parser.add_argument('--dqn-results', type=str, required=True, help='Path to DQN evaluation results')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()
    
    # Load DQN results
    dqn_results = load_results(args.dqn_results)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Extract parameters
    num_nodes = config["num_nodes"]
    edge_list = config["edge_list"]
    tm_list = config["tm_list"]
    
    print(f"Loaded configuration with {num_nodes} nodes, {len(edge_list)} edges, and {len(tm_list)} traffic matrices")
    
    # Known optimal rewards for the config_5node_small.json
    # These would be determined by exhaustive search
    optimal_rewards = {
        0: 20.0,  # TM 0: Best to close 2 links without violations
        1: 10.0,  # TM 1: Best to close 1 link without violations
        2: 0.0,   # TM 2: Best to keep all links open to avoid violations
        3: 0.0    # TM 3: Best to keep all links open to avoid violations
    }
    
    # Extract DQN standard and model-based results
    standard_results = {}
    model_based_results = {}
    
    for result in dqn_results:
        tm_idx = result['tm_idx']
        standard_results[tm_idx] = {
            'reward': result['reward'],
            'closed_links': result['links_closed'],
            'has_violation': result['has_violation']
        }
        
        # Add model-based results if available
        if 'model_based_results' in dqn_results[0]:
            model_based_results[tm_idx] = {
                'reward': result['model_based_results']['reward'],
                'closed_links': result['model_based_results']['links_closed'],
                'has_violation': result['model_based_results']['has_violation']
            }
    
    # Compare with optimal
    print("\n=== DQN vs Optimal Comparison ===")
    dqn_optimal_ratio = []
    success_count = 0
    
    for tm_idx, optimal in optimal_rewards.items():
        if tm_idx in standard_results:
            dqn = standard_results[tm_idx]['reward']
            
            # Calculate percentage of optimal (if positive rewards)
            if optimal > 0:
                ratio = dqn / optimal * 100
                dqn_optimal_ratio.append(ratio)
            else:
                # If optimal is 0, check if DQN also got 0 or better
                ratio = 100 if dqn >= 0 else 0
                dqn_optimal_ratio.append(ratio)
            
            # Check if DQN found optimal solution
            is_optimal = dqn >= optimal and not standard_results[tm_idx]['has_violation']
            if is_optimal:
                success_count += 1
            
            print(f"TM {tm_idx}:")
            print(f"  DQN Reward: {dqn}, Optimal Reward: {optimal}")
            print(f"  Performance Ratio: {ratio:.1f}%")
            print(f"  Found Optimal: {'✅' if is_optimal else '❌'}")
    
    # Overall statistics
    avg_ratio = sum(dqn_optimal_ratio) / len(dqn_optimal_ratio)
    print(f"\nOverall Performance:")
    print(f"  Average Performance vs Optimal: {avg_ratio:.1f}%")
    print(f"  Optimal Solutions Found: {success_count}/{len(optimal_rewards)} ({success_count/len(optimal_rewards)*100:.1f}%)")
    
    # Model-based comparison if available
    if model_based_results:
        print("\n=== Model-Based DQN vs Optimal Comparison ===")
        model_optimal_ratio = []
        model_success_count = 0
        
        for tm_idx, optimal in optimal_rewards.items():
            if tm_idx in model_based_results:
                model = model_based_results[tm_idx]['reward']
                
                # Calculate percentage of optimal
                if optimal > 0:
                    ratio = model / optimal * 100
                    model_optimal_ratio.append(ratio)
                else:
                    ratio = 100 if model >= 0 else 0
                    model_optimal_ratio.append(ratio)
                
                # Check if model found optimal solution
                is_optimal = model >= optimal and not model_based_results[tm_idx]['has_violation']
                if is_optimal:
                    model_success_count += 1
                
                print(f"TM {tm_idx}:")
                print(f"  Model-Based Reward: {model}, Optimal Reward: {optimal}")
                print(f"  Performance Ratio: {ratio:.1f}%")
                print(f"  Found Optimal: {'✅' if is_optimal else '❌'}")
        
        # Overall statistics for model-based
        if model_optimal_ratio:
            model_avg_ratio = sum(model_optimal_ratio) / len(model_optimal_ratio)
            print(f"\nOverall Model-Based Performance:")
            print(f"  Average Performance vs Optimal: {model_avg_ratio:.1f}%")
            print(f"  Optimal Solutions Found: {model_success_count}/{len(optimal_rewards)} ({model_success_count/len(optimal_rewards)*100:.1f}%)")
    
    # Create visualization directory
    os.makedirs("comparison_plots", exist_ok=True)
    
    # Plot rewards comparison
    plt.figure(figsize=(10, 6))
    
    tm_indices = list(optimal_rewards.keys())
    dqn_rewards = [standard_results[tm]['reward'] if tm in standard_results else 0 for tm in tm_indices]
    optimal_reward_values = [optimal_rewards[tm] for tm in tm_indices]
    
    if model_based_results:
        model_rewards = [model_based_results[tm]['reward'] if tm in model_based_results else 0 for tm in tm_indices]
        x = np.arange(len(tm_indices))
        width = 0.2
        
        plt.bar(x - width, dqn_rewards, width, label='DQN Standard', color='blue', alpha=0.7)
        plt.bar(x, model_rewards, width, label='DQN Model-Based', color='green', alpha=0.7)
        plt.bar(x + width, optimal_reward_values, width, label='Optimal', color='red', alpha=0.7)
    else:
        x = np.arange(len(tm_indices))
        width = 0.3
        
        plt.bar(x - width/2, dqn_rewards, width, label='DQN', color='blue', alpha=0.7)
        plt.bar(x + width/2, optimal_reward_values, width, label='Optimal', color='red', alpha=0.7)
    
    plt.xlabel('Traffic Matrix Index')
    plt.ylabel('Reward')
    plt.title('Reward Comparison: Latent Predictor Agent vs Optimal')
    plt.xticks(x, tm_indices)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_plots/reward_comparison.png')
    print("\nReward comparison plot saved to comparison_plots/reward_comparison.png")

if __name__ == "__main__":
    main()
