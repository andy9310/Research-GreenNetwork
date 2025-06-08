"""
Compare Relaxation Approach with Bruteforce

This script evaluates the relaxation approach against bruteforce optimal solutions,
comparing performance on network optimization tasks.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import itertools
import networkx as nx

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relaxation.env import RelaxedNetworkEnv
from relaxation.ddpg_agent import DDPGAgent

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def bruteforce_evaluation(env, tm_idx, max_combinations=10000):
    """
    Find the optimal solution by exhaustively searching all possible configurations.
    For large networks, use random sampling if combinations exceed max_combinations.
    
    Args:
        env: RelaxedNetworkEnv environment
        tm_idx: Traffic matrix index
        max_combinations: Maximum number of combinations to evaluate
        
    Returns:
        best_config: Best configuration of links
        best_reward: Best reward achieved
        best_link_usage: Link usage for best configuration
        evaluated_configs: Number of configurations evaluated
    """
    print(f"Starting bruteforce evaluation for traffic matrix {tm_idx}...")
    
    # Set traffic matrix
    env.current_tm_idx = tm_idx
    
    # Calculate total number of combinations (2^num_edges)
    num_edges = env.num_edges
    total_combinations = 2**num_edges
    
    # Determine if we need to use sampling
    use_sampling = total_combinations > max_combinations
    
    if use_sampling:
        print(f"Total combinations ({total_combinations}) exceeds max ({max_combinations}), using random sampling")
        # Generate random configurations
        configs_to_evaluate = []
        for _ in range(max_combinations):
            # Random binary configuration
            config = [np.random.randint(0, 2) for _ in range(num_edges)]
            configs_to_evaluate.append(config)
    else:
        print(f"Evaluating all {total_combinations} combinations")
        # Generate all possible configurations
        configs_to_evaluate = list(itertools.product([0, 1], repeat=num_edges))
    
    # Initialize tracking variables
    best_reward = float('-inf')
    best_config = None
    best_link_usage = None
    best_factors = None
    evaluated_configs = 0
    
    # Progress bar
    pbar = tqdm(total=len(configs_to_evaluate))
    
    # Evaluate each configuration
    for config in configs_to_evaluate:
        # Reset environment
        state = env.reset()
        done = False
        episode_reward = 0
        
        # Apply configuration
        for i, action_value in enumerate(config):
            # Convert binary value to capacity factor (0 or 1)
            action = np.array([float(action_value)])
            
            # Take step
            next_state, reward, done, truncated, info = env.step(action)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            # Break if episode is done
            if done:
                break
        
        # Check if this is the best configuration so far
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_config = config
            best_link_usage = env.link_usage.copy()
            best_factors = env.link_factors.copy()
        
        evaluated_configs += 1
        pbar.update(1)
    
    pbar.close()
    
    print(f"Bruteforce evaluation completed. Best reward: {best_reward:.2f}")
    print(f"Best configuration: {best_config}")
    
    return best_config, best_reward, best_link_usage, best_factors, evaluated_configs

def relaxation_evaluation(agent, env, tm_idx, num_episodes=10, use_state_predictor=True):
    """
    Evaluate the relaxation agent on a traffic matrix.
    
    Args:
        agent: DDPG agent
        env: RelaxedNetworkEnv environment
        tm_idx: Traffic matrix index
        num_episodes: Number of evaluation episodes
        use_state_predictor: Whether to use the state predictor for evaluation
        
    Returns:
        best_config: Best configuration of links
        best_reward: Best reward achieved
        best_link_usage: Link usage for best configuration
        best_factors: Best capacity factors
    """
    print(f"Evaluating relaxation approach on traffic matrix {tm_idx}...")
    
    # Set traffic matrix
    env.current_tm_idx = tm_idx
    
    # Initialize tracking variables
    best_reward = float('-inf')
    best_config = None
    best_link_usage = None
    best_factors = None
    
    for i in range(num_episodes):
        if use_state_predictor:
            # Model-based evaluation
            state = env.reset()
            done = False
            
            # Track decisions
            actions = []
            
            # Use state predictor for simulation
            while not done:
                # Select action without noise
                action = agent.act(state, add_noise=False)
                actions.append(action[0])
                
                # Predict next state
                next_state = agent.predict_next_state(state, action)
                
                # Check if episode is done
                current_edge_idx = int(state[-1])
                if current_edge_idx >= env.num_edges - 1:
                    done = True
                else:
                    # Update current edge index in predicted state
                    if len(next_state.shape) > 1:
                        next_state[0, -1] = current_edge_idx + 1
                    else:
                        next_state[-1] = current_edge_idx + 1
                
                # Update state
                state = next_state
            
            # Apply the actions to the environment to get the actual reward
            state = env.reset()
            episode_reward = 0
            
            for action_value in actions:
                action = np.array([action_value])
                next_state, reward, done, truncated, info = env.step(action)
                state = next_state
                episode_reward += reward
        else:
            # Standard evaluation
            state = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Select action without noise
                action = agent.act(state, add_noise=False)
                
                # Take step
                next_state, reward, done, truncated, info = env.step(action)
                
                # Update state and reward
                state = next_state
                episode_reward += reward
        
        # Check if this is the best configuration
        if episode_reward > best_reward:
            best_reward = episode_reward
            
            # Convert capacity factors to binary config for comparison
            binary_config = [1 if factor >= 0.01 else 0 for factor in env.link_factors]
            best_config = binary_config
            
            best_link_usage = env.link_usage.copy()
            best_factors = env.link_factors.copy()
    
    print(f"Relaxation evaluation completed. Best reward: {best_reward:.2f}")
    
    return best_config, best_reward, best_link_usage, best_factors

def compare_approaches(args):
    """
    Compare relaxation approach with bruteforce.
    
    Args:
        args: Command-line arguments
    """
    # --- Load configuration ---
    config_path = args.config
    config = load_config(config_path)
    config_name = os.path.basename(config_path).split('.')[0]
    print(f"Loaded configuration from {config_path}")
    
    # Extract parameters from config
    num_nodes = config["num_nodes"]
    edge_list = config["edge_list"]
    tm_list = config["tm_list"]
    link_capacity = config.get("link_capacity", [1.0] * len(edge_list))
    node_props = config.get("node_props", {})
    max_edges = len(edge_list)
    
    # Create environment
    env = RelaxedNetworkEnv(
        edge_list=edge_list,
        tm_list=tm_list,
        link_capacity=link_capacity,
        node_props=node_props,
        num_nodes=num_nodes,
        max_edges=max_edges,
        random_edge_order=False,  # Fixed order for comparison
        seed=args.seed
    )
    print(f"Created environment with {num_nodes} nodes, {len(edge_list)} edges, and {len(tm_list)} traffic matrices")
    
    # Get state dimensions
    initial_state = env.reset()
    state_dim = initial_state.shape[0]
    print(f"State dimension: {state_dim}")
    
    # --- Setup agent ---
    # Set up device
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize agent
    agent = DDPGAgent(
        state_dim=state_dim,
        latent_dim=args.latent_dim,
        predictor_hidden_dim=args.predictor_hidden_dim,
        actor_hidden_dim=args.actor_hidden_dim,
        critic_hidden_dim=args.critic_hidden_dim,
        enable_predictor=True,  # Enable predictor for model-based evaluation
        device=device
    )
    
    # Load trained models
    agent.load(args.actor_model, args.critic_model, args.predictor_model)
    print(f"Loaded models from {args.actor_model}, {args.critic_model}, {args.predictor_model}")
    
    # Create directories for results
    os.makedirs("results", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    
    # --- Comparison ---
    print("\nStarting comparison...")
    
    # Results storage
    results = {
        "bruteforce": {},
        "relaxation": {},
        "comparison": {}
    }
    
    # Evaluate on specified traffic matrices
    tm_indices = args.tm_indices if args.tm_indices else list(range(len(tm_list)))
    
    for tm_idx in tm_indices:
        print(f"\n--- Traffic Matrix {tm_idx} ---")
        
        # Run bruteforce evaluation
        if args.run_bruteforce:
            bf_config, bf_reward, bf_usage, bf_factors, bf_evaluated = bruteforce_evaluation(
                env, tm_idx, args.max_combinations
            )
            
            results["bruteforce"][tm_idx] = {
                "config": bf_config,
                "reward": float(bf_reward),
                "usage": bf_usage.tolist(),
                "factors": bf_factors.tolist(),
                "evaluated_configs": bf_evaluated
            }
        
        # Run relaxation evaluation
        rel_config, rel_reward, rel_usage, rel_factors = relaxation_evaluation(
            agent, env, tm_idx, args.eval_episodes, args.use_state_predictor
        )
        
        results["relaxation"][tm_idx] = {
            "config": rel_config,
            "reward": float(rel_reward),
            "usage": rel_usage.tolist(),
            "factors": rel_factors.tolist()
        }
        
        # Compare if bruteforce was run
        if args.run_bruteforce:
            # Calculate difference in reward
            reward_diff = rel_reward - bf_reward
            reward_diff_percent = (reward_diff / abs(bf_reward)) * 100 if bf_reward != 0 else float('inf')
            
            # Calculate configuration difference
            config_diff = sum(1 for a, b in zip(rel_config, bf_config) if a != b)
            config_diff_percent = (config_diff / len(rel_config)) * 100
            
            results["comparison"][tm_idx] = {
                "reward_diff": float(reward_diff),
                "reward_diff_percent": float(reward_diff_percent),
                "config_diff": config_diff,
                "config_diff_percent": float(config_diff_percent)
            }
            
            print(f"Reward Comparison: Relaxation {rel_reward:.2f} vs Bruteforce {bf_reward:.2f}")
            print(f"Difference: {reward_diff:.2f} ({reward_diff_percent:.2f}%)")
            print(f"Configuration Difference: {config_diff}/{len(rel_config)} links ({config_diff_percent:.2f}%)")
        
        # Visualize relaxation solution
        visualize_network(
            num_nodes, edge_list, rel_config, rel_usage, rel_factors,
            args.threshold, tm_idx, config_name, "relaxation"
        )
        
        # Visualize bruteforce solution if available
        if args.run_bruteforce:
            visualize_network(
                num_nodes, edge_list, bf_config, bf_usage, bf_factors,
                args.threshold, tm_idx, config_name, "bruteforce"
            )
    
    # Save results
    results_path = f"results/comparison_{config_name}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nComparison results saved to {results_path}")
    
    # Print overall summary
    print("\n--- Comparison Summary ---")
    
    if args.run_bruteforce and results["comparison"]:
        avg_reward_diff = np.mean([results["comparison"][tm_idx]["reward_diff"] 
                                 for tm_idx in results["comparison"]])
        avg_reward_diff_percent = np.mean([results["comparison"][tm_idx]["reward_diff_percent"] 
                                        for tm_idx in results["comparison"] 
                                        if results["comparison"][tm_idx]["reward_diff_percent"] != float('inf')])
        avg_config_diff_percent = np.mean([results["comparison"][tm_idx]["config_diff_percent"] 
                                        for tm_idx in results["comparison"]])
        
        print(f"Average Reward Difference: {avg_reward_diff:.2f} ({avg_reward_diff_percent:.2f}%)")
        print(f"Average Configuration Difference: {avg_config_diff_percent:.2f}%")
        
        # Create comparison bar chart
        create_comparison_chart(results, tm_indices, config_name)
    
    return results

def visualize_network(num_nodes, edge_list, link_state, link_usage, capacity_factors, 
                     threshold, tm_idx, config_name, method_name):
    """
    Visualize the network topology with link usage information.
    
    Args:
        num_nodes: Number of nodes
        edge_list: List of edges
        link_state: List of link states (0=closed, 1=open)
        link_usage: List of link usages
        capacity_factors: List of capacity scaling factors
        threshold: Threshold for displaying capacity factors
        tm_idx: Traffic matrix index
        config_name: Configuration name
        method_name: Method name ('relaxation' or 'bruteforce')
    """
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for i in range(num_nodes):
        G.add_node(i)
    
    # Add edges with attributes
    for i, (u, v) in enumerate(edge_list):
        # Check if link is open
        if link_state[i] == 1:
            # Calculate utilization
            capacity = capacity_factors[i] if isinstance(capacity_factors[i], (int, float)) else 1.0
            utilization = link_usage[i] / capacity if capacity > 0 else float('inf')
            
            # Add edge with attributes
            G.add_edge(u, v, 
                      capacity=capacity,
                      usage=link_usage[i],
                      utilization=utilization)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Position nodes using spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    # Edge widths and colors based on utilization
    edge_colors = []
    edge_widths = []
    for (u, v, data) in G.edges(data=True):
        # Determine color based on utilization
        if data.get('utilization', 0) < 0.3:
            edge_colors.append('green')  # Low utilization
        elif data.get('utilization', 0) < 0.7:
            edge_colors.append('orange')  # Medium utilization
        else:
            edge_colors.append('red')  # High utilization
        
        # Edge width based on capacity
        edge_widths.append(1 + 3 * data.get('capacity', 1.0))
    
    # Draw edges with colors
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.7)
    
    # Edge labels with utilization
    edge_labels = {(u, v): f"{data.get('utilization', 0):.2f}" for u, v, data in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    # Title and legend
    plt.title(f"{method_name.title()} Approach - TM {tm_idx} ({config_name})")
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=4, label='Low Utilization (<30%)'),
        Line2D([0], [0], color='orange', lw=4, label='Medium Utilization (30-70%)'),
        Line2D([0], [0], color='red', lw=4, label='High Utilization (>70%)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Save figure
    plt.savefig(f"visualizations/{method_name}_{config_name}_tm{tm_idx}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"{method_name.title()} visualization saved to visualizations/{method_name}_{config_name}_tm{tm_idx}.png")

def create_comparison_chart(results, tm_indices, config_name):
    """
    Create a bar chart comparing relaxation and bruteforce approaches.
    
    Args:
        results: Dictionary with results
        tm_indices: List of traffic matrix indices
        config_name: Configuration name
    """
    # Extract rewards
    bf_rewards = [results["bruteforce"][tm_idx]["reward"] for tm_idx in tm_indices]
    rel_rewards = [results["relaxation"][tm_idx]["reward"] for tm_idx in tm_indices]
    
    # Setup the chart
    x = np.arange(len(tm_indices))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, bf_rewards, width, label='Bruteforce')
    rects2 = ax.bar(x + width/2, rel_rewards, width, label='Relaxation')
    
    # Add labels and title
    ax.set_xlabel('Traffic Matrix Index')
    ax.set_ylabel('Reward')
    ax.set_title(f'Bruteforce vs Relaxation Approach ({config_name})')
    ax.set_xticks(x)
    ax.set_xticklabels(tm_indices)
    ax.legend()
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(rect.get_x() + rect.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    
    # Save figure
    plt.savefig(f"visualizations/reward_comparison_{config_name}.png", dpi=300)
    plt.close()
    
    print(f"Comparison chart saved to visualizations/reward_comparison_{config_name}.png")

def main():
    parser = argparse.ArgumentParser(description='Compare relaxation approach with bruteforce')
    
    # Configuration
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    
    # Model paths
    parser.add_argument('--actor-model', type=str, required=True, help='Path to actor model')
    parser.add_argument('--critic-model', type=str, required=True, help='Path to critic model')
    parser.add_argument('--predictor-model', type=str, required=True, help='Path to state predictor model')
    
    # Network architecture
    parser.add_argument('--latent-dim', type=int, default=64, help='Dimension of latent state representation')
    parser.add_argument('--predictor-hidden-dim', type=int, default=256, help='Hidden dimension of state predictor')
    parser.add_argument('--actor-hidden-dim', type=int, default=256, help='Hidden dimension of actor network')
    parser.add_argument('--critic-hidden-dim', type=int, default=256, help='Hidden dimension of critic network')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage instead of CUDA')
    
    # Evaluation settings
    parser.add_argument('--eval-episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--use-state-predictor', action='store_true', help='Use state predictor for evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--threshold', type=float, default=0.1, help='Capacity factor threshold for visualization')
    parser.add_argument('--tm-indices', type=int, nargs='+', help='Traffic matrix indices to evaluate')
    
    # Bruteforce settings
    parser.add_argument('--run-bruteforce', action='store_true', help='Run bruteforce evaluation')
    parser.add_argument('--max-combinations', type=int, default=10000, help='Maximum number of combinations for bruteforce')
    
    args = parser.parse_args()
    compare_approaches(args)

if __name__ == "__main__":
    main()
