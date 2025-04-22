"""
Compare test script to evaluate individual traffic matrices and validate optimal solutions
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
import itertools
import time
import matplotlib.pyplot as plt

# Add parent directory to path to import modules
sys.path.append('../')
from env import NetworkEnv
from latent_predictor_agent import LatentPredictorAgent

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def exhaustive_search(env, tm_idx, max_links_to_close=4):
    """
    Perform exhaustive search to find optimal configuration.
    
    Args:
        env: The network environment
        tm_idx: Index of traffic matrix to evaluate
        max_links_to_close: Maximum number of links to close
        
    Returns:
        best_actions: List of actions (links to close)
        best_reward: Reward of the best configuration
        stats: Dictionary with search statistics
    """
    # Set traffic matrix
    env.current_tm_idx = tm_idx
    
    # Initialize search
    max_edges = len(env.edge_list)
    best_reward = -float('inf')
    best_actions = []
    total_configs = 0
    start_time = time.time()
    
    # Try closing different numbers of links
    for num_to_close in range(max_links_to_close + 1):
        print(f"Trying {num_to_close} closed links...")
        
        # Generate all combinations of links to close
        for actions in tqdm(itertools.combinations(range(max_edges), num_to_close)):
            # Reset environment
            state = env.reset()
            total_reward = 0
            done = False
            
            # Apply actions
            for action in actions:
                if done:
                    break
                next_state, reward, done, truncated, info = env.step(action)
                total_reward += reward
            
            total_configs += 1
            
            # Check if valid (no isolations or overloaded links)
            valid = not done and not any(u > c for u, c in zip(env.link_usage, env.link_capacity))
            
            # Update best configuration if better
            if valid and total_reward > best_reward:
                best_reward = total_reward
                best_actions = list(actions)
    
    # Calculate stats
    elapsed_time = time.time() - start_time
    stats = {
        "total_configs": total_configs,
        "elapsed_time": elapsed_time,
        "closed_links": len(best_actions),
        "total_links": max_edges
    }
    
    return best_actions, best_reward, stats

def agent_evaluation(env, agent, tm_idx, model_based=False):
    """
    Evaluate agent performance on a specific traffic matrix.
    
    Args:
        env: The network environment
        agent: The trained LatentPredictorAgent
        tm_idx: Index of traffic matrix to evaluate
        model_based: Whether to use model-based evaluation
        
    Returns:
        reward: Total reward achieved
        stats: Dictionary with evaluation statistics
    """
    # Set traffic matrix
    env.current_tm_idx = tm_idx
    
    # Reset environment
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    
    total_reward = 0
    done = False
    actions_taken = []
    
    # Model-based evaluation
    if model_based:
        # Get initial state
        current_state = state.copy()
        steps = 0
        
        while not done and steps < 100:  # Limit steps to avoid infinite loops
            # Select action using trained policy
            action = agent.act(current_state, epsilon=0.0)
            actions_taken.append(action)
            
            # Predict next state instead of environment interaction
            next_state = agent.predict_next_state(current_state, action)
            reward = 10 if action > 0 else 0  # Simplified reward for prediction
            
            # Update tracking variables
            total_reward += reward
            current_state = next_state
            steps += 1
            
            # Check if done based on link status in predicted state
            # This is a simplification as we can't directly check constraints
            if steps >= len(env.edge_list):
                done = True
        
        # Validate final configuration in actual environment
        env.reset()
        done = False
        validation_reward = 0
        
        for action in actions_taken:
            if done:
                break
            next_state, reward, done, truncated, info = env.step(action)
            validation_reward += reward
        
        # Check final state
        has_isolated = not env.is_connected()
        has_overloaded = any(u > c for u, c in zip(env.link_usage, env.link_capacity))
        
    # Standard evaluation
    else:
        while not done:
            # Select action using trained policy
            action = agent.act(state, epsilon=0.0)
            actions_taken.append(action)
            
            # Take action in environment
            next_state, reward, done, truncated, info = env.step(action)
            
            # Update tracking variables
            total_reward += reward
            state = next_state
        
        # Check final state
        has_isolated = not env.is_connected()
        has_overloaded = any(u > c for u, c in zip(env.link_usage, env.link_capacity))
        validation_reward = total_reward
    
    # Calculate statistics
    stats = {
        "actions": actions_taken,
        "closed_links": len(actions_taken),
        "isolated": has_isolated,
        "overloaded": has_overloaded,
        "validation_reward": validation_reward
    }
    
    return total_reward, stats

def main():
    parser = argparse.ArgumentParser(description='Compare DQN with optimal solutions')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--predictor-model', type=str, required=False, help='Path to predictor model')
    parser.add_argument('--latent-dim', type=int, default=16, help='Dimension of latent representation')
    parser.add_argument('--architecture', type=str, default='mlp', choices=['mlp', 'fatmlp'], help='Network architecture')
    args = parser.parse_args()
    
    # Load configuration
    config_path = args.config
    config = load_config(config_path)
    
    # Extract parameters
    num_nodes = config["num_nodes"]
    edge_list = config["edge_list"]
    tm_list = config["tm_list"]
    link_capacity = config.get("link_capacity", [1.0] * len(edge_list))
    node_props = config.get("node_props", {})
    
    print(f"Loaded configuration with {num_nodes} nodes, {len(edge_list)} edges, and {len(tm_list)} traffic matrices")
    
    # Create environment
    env = NetworkEnv(
        edge_list=edge_list,
        tm_list=tm_list,
        link_capacity=link_capacity,
        node_props=node_props,
        num_nodes=num_nodes
    )
    
    # Verify state dimensions
    initial_state = env.reset()
    if isinstance(initial_state, tuple):
        initial_state = initial_state[0]
    state_dim = initial_state.shape[0]
    action_dim = env.action_space.n
    
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize agent
    agent = LatentPredictorAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=args.latent_dim,
        enable_predictor=args.predictor_model is not None,
        architecture=args.architecture,
        device=device
    )
    
    # Load model weights
    if args.predictor_model:
        agent.load(args.model, args.predictor_model)
        print(f"Loaded model from {args.model} and predictor from {args.predictor_model}")
    else:
        agent.load(args.model)
        print(f"Loaded model from {args.model}")
    
    # Comparative evaluation results
    results = []
    
    # Evaluate each traffic matrix
    for tm_idx in range(len(tm_list)):
        print(f"\n--- Evaluating Traffic Matrix {tm_idx} ---")
        
        # DQN standard evaluation
        dqn_reward, dqn_stats = agent_evaluation(env, agent, tm_idx, model_based=False)
        print("\nStandard DQN Evaluation:")
        print(f"Reward: {dqn_reward}")
        print(f"Closed Links: {dqn_stats['closed_links']}/{len(edge_list)}")
        status = "Success ✅" if not dqn_stats["isolated"] and not dqn_stats["overloaded"] else "Violation ❌"
        print(f"Status: {status}")
        
        # DQN model-based evaluation (if predictor available)
        if args.predictor_model:
            model_reward, model_stats = agent_evaluation(env, agent, tm_idx, model_based=True)
            print("\nModel-Based DQN Evaluation:")
            print(f"Reward: {model_reward}")
            print(f"Closed Links: {model_stats['closed_links']}/{len(edge_list)}")
            print(f"Validation Reward: {model_stats['validation_reward']}")
            status = "Success ✅" if not model_stats["isolated"] and not model_stats["overloaded"] else "Violation ❌"
            print(f"Status: {status}")
        else:
            model_reward = None
            model_stats = None
        
        # Optimal solution using exhaustive search
        print("\nFinding optimal solution using exhaustive search...")
        opt_actions, opt_reward, opt_stats = exhaustive_search(env, tm_idx, max_links_to_close=4)
        print("\nExhaustive Search Results:")
        print(f"Best Reward: {opt_reward}")
        print(f"Closed Links: {opt_stats['closed_links']}/{opt_stats['total_links']}")
        print(f"Actions: {opt_actions}")
        print(f"Evaluation Time: {opt_stats['elapsed_time']:.2f} seconds")
        print(f"Configurations Evaluated: {opt_stats['total_configs']}")
        
        # Store results
        result = {
            "tm_idx": tm_idx,
            "dqn": {
                "reward": dqn_reward,
                "closed_links": dqn_stats["closed_links"],
                "isolated": dqn_stats["isolated"],
                "overloaded": dqn_stats["overloaded"],
                "actions": dqn_stats["actions"]
            },
            "optimal": {
                "reward": opt_reward,
                "closed_links": opt_stats["closed_links"],
                "actions": opt_actions
            }
        }
        
        if model_stats:
            result["model_based"] = {
                "reward": model_reward,
                "closed_links": model_stats["closed_links"],
                "isolated": model_stats["isolated"],
                "overloaded": model_stats["overloaded"],
                "validation_reward": model_stats["validation_reward"],
                "actions": model_stats["actions"]
            }
        
        results.append(result)
    
    # Overall comparison
    print("\n=== Overall Comparison ===")
    
    # Calculate metrics
    dqn_vs_opt = [r["dqn"]["reward"] / max(1, r["optimal"]["reward"]) * 100 if r["optimal"]["reward"] > 0 else 0 for r in results]
    dqn_success = sum(1 for r in results if not r["dqn"]["isolated"] and not r["dqn"]["overloaded"])
    opt_success = sum(1 for r in results if r["optimal"]["reward"] > -float('inf'))
    
    if args.predictor_model:
        model_vs_opt = [r["model_based"]["reward"] / max(1, r["optimal"]["reward"]) * 100 if r["optimal"]["reward"] > 0 else 0 for r in results]
        model_success = sum(1 for r in results if not r["model_based"]["isolated"] and not r["model_based"]["overloaded"])
    
    print(f"DQN Success Rate: {dqn_success}/{len(results)} ({dqn_success/len(results)*100:.1f}%)")
    print(f"DQN vs Optimal Performance: {sum(dqn_vs_opt)/len(dqn_vs_opt):.1f}%")
    
    if args.predictor_model:
        print(f"Model-Based Success Rate: {model_success}/{len(results)} ({model_success/len(results)*100:.1f}%)")
        print(f"Model-Based vs Optimal Performance: {sum(model_vs_opt)/len(model_vs_opt):.1f}%")
    
    print(f"Optimal Success Rate: {opt_success}/{len(results)} ({opt_success/len(results)*100:.1f}%)")
    
    # Save results
    config_name = os.path.basename(config_path).split('.')[0]
    results_file = f"comparison_results_{config_name}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed comparison results saved to {results_file}")
    
    # Create visualization directory
    os.makedirs("comparison_plots", exist_ok=True)
    
    # Plot reward comparison
    plt.figure(figsize=(12, 6))
    tm_indices = [r["tm_idx"] for r in results]
    dqn_rewards = [r["dqn"]["reward"] for r in results]
    opt_rewards = [r["optimal"]["reward"] for r in results]
    
    plt.bar(np.array(tm_indices) - 0.2, dqn_rewards, width=0.4, label='DQN', color='blue', alpha=0.7)
    if args.predictor_model:
        model_rewards = [r["model_based"]["validation_reward"] for r in results]
        plt.bar(np.array(tm_indices), model_rewards, width=0.4, label='Model-Based', color='green', alpha=0.7)
    plt.bar(np.array(tm_indices) + 0.2, opt_rewards, width=0.4, label='Optimal', color='red', alpha=0.7)
    
    plt.xlabel('Traffic Matrix Index')
    plt.ylabel('Reward')
    plt.title('Reward Comparison: DQN vs Optimal')
    plt.xticks(tm_indices)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = f"comparison_plots/reward_comparison_{config_name}.png"
    plt.savefig(plot_path)
    print(f"Reward comparison plot saved to {plot_path}")
    
    # Plot closed links comparison
    plt.figure(figsize=(12, 6))
    dqn_closed = [r["dqn"]["closed_links"] for r in results]
    opt_closed = [r["optimal"]["closed_links"] for r in results]
    
    plt.bar(np.array(tm_indices) - 0.2, dqn_closed, width=0.4, label='DQN', color='blue', alpha=0.7)
    if args.predictor_model:
        model_closed = [r["model_based"]["closed_links"] for r in results]
        plt.bar(np.array(tm_indices), model_closed, width=0.4, label='Model-Based', color='green', alpha=0.7)
    plt.bar(np.array(tm_indices) + 0.2, opt_closed, width=0.4, label='Optimal', color='red', alpha=0.7)
    
    plt.xlabel('Traffic Matrix Index')
    plt.ylabel('Number of Closed Links')
    plt.title('Closed Links Comparison: DQN vs Optimal')
    plt.xticks(tm_indices)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = f"comparison_plots/closed_links_comparison_{config_name}.png"
    plt.savefig(plot_path)
    print(f"Closed links comparison plot saved to {plot_path}")

if __name__ == "__main__":
    main()
