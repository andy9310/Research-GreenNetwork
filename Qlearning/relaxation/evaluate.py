"""
Evaluation Script for Relaxed Network Optimization

This script evaluates a trained DDPG agent on the relaxed network optimization task,
comparing standard evaluation with model-based evaluation using the state predictor.
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
import networkx as nx

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relaxation.env import RelaxedNetworkEnv
from relaxation.ddpg_agent import DDPGAgent

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def find_latest_model_files(config_name):
    """
    Find the most recent model files for the given configuration.
    
    Args:
        config_name: Configuration name
        
    Returns:
        Tuple of (actor_path, critic_path, predictor_path)
    """
    import glob
    import os
    
    # Get the model directory
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
    
    # Find all model files for this configuration
    pattern = os.path.join(model_dir, f"relaxed_{config_name}*")
    model_files = glob.glob(pattern)
    
    # Group files by episode
    file_groups = {}
    for file in model_files:
        basename = os.path.basename(file)
        if "actor" in basename:
            episode = basename.split("episode")[1].split("_")[0] if "episode" in basename else "final"
            if episode not in file_groups:
                file_groups[episode] = {}
            file_groups[episode]["actor"] = file
        elif "critic" in basename:
            episode = basename.split("episode")[1].split("_")[0] if "episode" in basename else "final"
            if episode not in file_groups:
                file_groups[episode] = {}
            file_groups[episode]["critic"] = file
        elif "predictor" in basename:
            episode = basename.split("episode")[1].split("_")[0] if "episode" in basename else "final"
            if episode not in file_groups:
                file_groups[episode] = {}
            file_groups[episode]["predictor"] = file
    
    # If no complete file groups found, try more flexible matching
    if not file_groups or not any(len(group) >= 2 for group in file_groups.values()):
        actor_files = glob.glob(os.path.join(model_dir, f"*actor*"))
        critic_files = glob.glob(os.path.join(model_dir, f"*critic*"))
        predictor_files = glob.glob(os.path.join(model_dir, f"*predictor*"))
        
        actor_path = actor_files[0] if actor_files else None
        critic_path = critic_files[0] if critic_files else None
        predictor_path = predictor_files[0] if predictor_files else None
        
        if actor_path and critic_path:
            print(f"Found model files using flexible matching:")
            print(f"Actor: {actor_path}")
            print(f"Critic: {critic_path}")
            print(f"Predictor: {predictor_path}")
            return actor_path, critic_path, predictor_path
    
    # Find the most recent episode (highest number or "final")
    if "final" in file_groups and len(file_groups["final"]) >= 2:
        latest_episode = "final"
    else:
        numeric_episodes = [int(ep) for ep in file_groups.keys() if ep != "final" and ep.isdigit()]
        latest_episode = str(max(numeric_episodes)) if numeric_episodes else None
    
    if not latest_episode:
        return None, None, None
    
    actor_path = file_groups[latest_episode].get("actor")
    critic_path = file_groups[latest_episode].get("critic")
    predictor_path = file_groups[latest_episode].get("predictor")
    
    print(f"Found latest model files from episode {latest_episode}:")
    print(f"Actor: {actor_path}")
    print(f"Critic: {critic_path}")
    print(f"Predictor: {predictor_path}")
    
    return actor_path, critic_path, predictor_path

def print_capacity_decisions(env, factors, tm_idx):
    """
    Print detailed information about capacity decisions made by the model.
    
    Args:
        env: Environment
        factors: List of capacity factors for each link
        tm_idx: Traffic matrix index
    """
    print(f"\n--- Detailed Capacity Decisions for TM {tm_idx} ---")
    print(f"{'Link':>10} | {'From':>5} | {'To':>5} | {'Capacity':>10} | {'Factor':>10} | {'Effective Cap':>15} | {'Usage':>10} | {'Utilization':>15}")
    print("-" * 90)
    
    for i, (u, v) in enumerate(env.edge_list):
        capacity = env.link_capacity[i] if isinstance(env.link_capacity, list) else env.link_capacity
        factor = factors[i]
        effective_cap = capacity * factor
        usage = env.link_usage[i]
        utilization = usage / effective_cap if effective_cap > 0 else float('inf')
        
        status = "OPEN" if factor > 0.01 else "CLOSED"
        
        print(f"{i:>10} | {u:>5} | {v:>5} | {capacity:>10.2f} | {factor:>10.4f} | {effective_cap:>15.2f} | {usage:>10.2f} | {utilization:>15.4f} | {status}")
    
    # Calculate energy savings
    total_capacity = sum(env.link_capacity) if isinstance(env.link_capacity, list) else env.link_capacity * env.num_edges
    used_capacity = sum(env.link_capacity[i] * factors[i] for i in range(env.num_edges)) if isinstance(env.link_capacity, list) else sum(env.link_capacity * factors[i] for i in range(env.num_edges))
    saved_capacity = total_capacity - used_capacity
    saved_percent = (saved_capacity / total_capacity) * 100 if total_capacity > 0 else 0
    
    print(f"\nTotal Capacity: {total_capacity:.2f}")
    print(f"Used Capacity: {used_capacity:.2f}")
    print(f"Saved Capacity: {saved_capacity:.2f} ({saved_percent:.2f}%)")
    
    # Decision summary
    fully_open = sum(1 for f in factors if f > 0.99)
    partial = sum(1 for f in factors if 0.01 < f < 0.99)
    closed = sum(1 for f in factors if f <= 0.01)
    
    print(f"\nDecision Summary:")
    print(f"Fully Open Links: {fully_open}/{env.num_edges}")
    print(f"Partially Open Links: {partial}/{env.num_edges}")
    print(f"Closed Links: {closed}/{env.num_edges}")

def evaluate_agent(args):
    """
    Evaluate the DDPG agent performance on the test environment.
    
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
        adj_matrix=config["adj_matrix"],
        edge_list=edge_list,
        tm_list=tm_list,
        node_props=node_props,
        num_nodes=num_nodes,
        link_capacity=link_capacity,
        max_edges=max_edges,
        random_edge_order=args.random_edge_order,
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
        enable_predictor=True,  # Always enable predictor for evaluation
        device=device
    )
    
    # Load trained models
    if args.auto_find_models or not (args.actor_model and args.critic_model):
        # Auto-find model files
        config_name = os.path.basename(config_path).split('.')[0]
        actor_path, critic_path, predictor_path = find_latest_model_files(config_name)
        
        if not (actor_path and critic_path):
            raise ValueError("Could not find valid model files automatically. Please specify them manually.")
    else:
        # Use specified model paths
        actor_path = args.actor_model
        critic_path = args.critic_model
        predictor_path = args.predictor_model
    
    # Load the models
    agent.load(actor_path, critic_path, predictor_path)
    print(f"Loaded models from:\n{actor_path}\n{critic_path}" + (f"\n{predictor_path}" if predictor_path else ""))
    
    # Create directories for results
    os.makedirs("results", exist_ok=True)
    
    # Determine evaluation method
    if args.model_based:
        print("Using model-based evaluation with state predictor")
    else:
        print("Using standard environment-based evaluation")
    
    # --- Evaluation ---
    print("\nStarting evaluation...")
    
    # Evaluate on each traffic matrix
    results = {
        "standard": {},
        "model_based": {},
        "comparison": {}
    }
    
    for tm_idx in range(len(tm_list)):
        print(f"\nEvaluating on Traffic Matrix {tm_idx}/{len(tm_list) - 1}")
        
        # Standard evaluation (environment interaction)
        std_rewards, std_links, std_usages, std_factors = standard_evaluation(
            agent, env, tm_idx, args.eval_episodes
        )
        
        results["standard"][tm_idx] = {
            "rewards": std_rewards,
            "links": std_links,
            "usages": std_usages,
            "factors": std_factors,
            "avg_reward": np.mean(std_rewards),
            "max_reward": np.max(std_rewards),
            "min_reward": np.min(std_rewards)
        }
        
        # Find best performing run
        best_idx = np.argmax(std_rewards)
        best_factors = std_factors[best_idx]
        
        # Print detailed decision information for the best run
        print(f"\nDetailed Analysis for Traffic Matrix {tm_idx}:")
        # Run a fresh evaluation to get updated usage information
        env.current_tm_idx = tm_idx
        state = env.reset()
        for edge_idx, factor in enumerate(best_factors):
            action = np.array([factor], dtype=np.float32)
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
        
        # Print detailed capacity decisions
        print_capacity_decisions(env, best_factors, tm_idx)
        
        print(f"Standard Evaluation: Avg Reward = {np.mean(std_rewards):.2f}, "
             f"Max = {np.max(std_rewards):.2f}, Min = {np.min(std_rewards):.2f}")
        
        # Model-based evaluation (state predictor)
        if args.model_based:
            mb_rewards, mb_links, mb_usages, mb_factors = model_based_evaluation(
                agent, env, tm_idx, args.eval_episodes
            )
            
            results["model_based"][tm_idx] = {
                "rewards": mb_rewards,
                "links": mb_links,
                "usages": mb_usages,
                "factors": mb_factors,
                "avg_reward": np.mean(mb_rewards),
                "max_reward": np.max(mb_rewards),
                "min_reward": np.min(mb_rewards)
            }
            
            # Compare with standard evaluation
            diff_reward = np.mean(mb_rewards) - np.mean(std_rewards)
            
            results["comparison"][tm_idx] = {
                "reward_diff": diff_reward,
                "reward_diff_percent": (diff_reward / abs(np.mean(std_rewards))) * 100 if np.mean(std_rewards) != 0 else float('inf')
            }
            
            print(f"Model-based Evaluation: Avg Reward = {np.mean(mb_rewards):.2f}, "
                 f"Max = {np.max(mb_rewards):.2f}, Min = {np.min(mb_rewards):.2f}")
            print(f"Difference: {diff_reward:.2f} ({results['comparison'][tm_idx]['reward_diff_percent']:.2f}%)")
        
        # Visualize network for best run
        best_idx = np.argmax(std_rewards)
        best_links = std_links[best_idx]
        best_usages = std_usages[best_idx]
        best_factors = std_factors[best_idx]
        
        # Generate network visualization
        visualize_network(
            num_nodes, edge_list, best_links, best_usages, best_factors,
            args.threshold, tm_idx, config_name, link_capacity
        )
    
    # Save results
    results_path = f"results/relaxed_{config_name}_eval_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nEvaluation results saved to {results_path}")
    
    # Print overall summary
    print("\n--- Evaluation Summary ---")
    
    avg_std_reward = np.mean([results["standard"][tm_idx]["avg_reward"] for tm_idx in results["standard"]])
    print(f"Standard Evaluation Average Reward: {avg_std_reward:.2f}")
    
    if args.model_based and results["model_based"]:
        avg_mb_reward = np.mean([results["model_based"][tm_idx]["avg_reward"] for tm_idx in results["model_based"]])
        avg_diff = np.mean([results["comparison"][tm_idx]["reward_diff"] for tm_idx in results["comparison"]])
        avg_diff_percent = np.mean([results["comparison"][tm_idx]["reward_diff_percent"] 
                                  for tm_idx in results["comparison"] 
                                  if results["comparison"][tm_idx]["reward_diff_percent"] != float('inf')])
        
        print(f"Model-based Evaluation Average Reward: {avg_mb_reward:.2f}")
        print(f"Average Difference: {avg_diff:.2f} ({avg_diff_percent:.2f}%)")
    
    return results

def standard_evaluation(agent, env, tm_idx, num_episodes):
    """
    Standard evaluation with environment interaction.
    
    Args:
        agent: DDPG agent
        env: Environment
        tm_idx: Traffic matrix index
        num_episodes: Number of evaluation episodes
    
    Returns:
        rewards: List of episode rewards
        links: List of final link states
        usages: List of final link usages
        factors: List of final capacity scaling factors
    """
    rewards = []
    links = []
    usages = []
    factors = []
    
    # Set traffic matrix
    env.current_tm_idx = tm_idx
    
    for i in range(num_episodes):
        # Reset environment
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action without noise (deterministic policy)
            action = agent.act(state, add_noise=False)
            
            # Take action in environment
            next_state, reward, done, truncated, info = env.step(action)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            # Store final information if done
            if done:
                link_state = []
                for idx, (u, v) in enumerate(env.edge_list):
                    if env.G.has_edge(u, v):
                        link_state.append(1)  # Link is open
                    else:
                        link_state.append(0)  # Link is closed
                
                links.append(link_state)
                usages.append(env.link_usage.tolist())
                factors.append(env.link_factors.tolist())
        
        rewards.append(episode_reward)
    
    return rewards, links, usages, factors

def model_based_evaluation(agent, env, tm_idx, num_episodes):
    """
    Model-based evaluation using state predictor.
    
    Args:
        agent: DDPG agent with state predictor
        env: Environment
        tm_idx: Traffic matrix index
        num_episodes: Number of evaluation episodes
    
    Returns:
        rewards: List of episode rewards
        links: List of final link states
        usages: List of final link usages
        factors: List of final capacity scaling factors
    """
    rewards = []
    links = []
    usages = []
    factors = []
    
    # Set traffic matrix
    env.current_tm_idx = tm_idx
    
    # Using standard evaluation as a fallback if model-based fails
    try:
        for i in range(num_episodes):
            try:
                # Reset environment to get initial state
                state = env.reset()
                done = False
                
                # Track link decisions
                link_decisions = []
                capacity_factors = []
                
                # Use state predictor for simulation
                while not done:
                    # Select action without noise (deterministic policy)
                    action = agent.act(state, add_noise=False)
                    
                    # Remember action
                    capacity_factors.append(float(action[0]))
                    
                    # Predict next state
                    next_state = agent.predict_next_state(state, action)
                    
                    # Mark current edge as decided
                    if isinstance(state, np.ndarray):
                        # Extract as scalar using numpy's item() method
                        if state.ndim > 1:
                            current_edge_idx = int(state[0, -1])
                        else:
                            current_edge_idx = int(state[-1])
                    else:
                        # Handle tensor case
                        if state.dim() > 1:
                            current_edge_idx = int(state[0, -1].item())
                        else:
                            current_edge_idx = int(state[-1].item())
                    
                    # Determine if link is open or closed based on capacity factor
                    if capacity_factors[-1] < 0.01:  # Very small capacity factor means link is closed
                        link_decisions.append(0)  # Link is closed
                    else:
                        link_decisions.append(1)  # Link is open
                    
                    # Check if all edges have been processed
                    if current_edge_idx >= env.num_edges - 1:
                        done = True
                    else:
                        # Update current edge index in predicted state
                        new_edge_idx = current_edge_idx + 1
                        if isinstance(next_state, np.ndarray):
                            if next_state.ndim > 1:
                                next_state[0, -1] = new_edge_idx
                            else:
                                next_state[-1] = new_edge_idx
                        else:
                            # Handle tensor case
                            if next_state.dim() > 1:
                                next_state[0, -1] = new_edge_idx
                            else:
                                next_state[-1] = new_edge_idx
                    
                    # Update state
                    state = next_state
                
                # Now, apply all decisions to a fresh environment and get the actual reward
                new_state = env.reset()
                new_done = False
                episode_reward = 0
                
                # Ensure we have the right number of decisions
                link_decisions = link_decisions[:env.num_edges]
                capacity_factors = capacity_factors[:env.num_edges]
                
                for edge_idx, (decision, factor) in enumerate(zip(link_decisions, capacity_factors)):
                    # Apply the pre-computed capacity factor
                    action = np.array([factor], dtype=np.float32)
                    
                    # Take action in environment
                    new_next_state, reward, new_done, truncated, info = env.step(action)
                    
                    # Update state and reward
                    new_state = new_next_state
                    episode_reward += reward
                    
                    # Break if done (e.g., due to violation)
                    if new_done:
                        break
                
                # Store results
                rewards.append(episode_reward)
                
                # Get final link states
                link_state = []
                for idx, (u, v) in enumerate(env.edge_list):
                    if env.G.has_edge(u, v):
                        link_state.append(1)  # Link is open
                    else:
                        link_state.append(0)  # Link is closed
                
                links.append(link_state)
                usages.append(env.link_usage.tolist())
                factors.append(env.link_factors.tolist())
                
            except Exception as e:
                print(f"Error in episode {i}: {e}")
                # Use default values for this episode
                rewards.append(0)
                links.append([1] * env.num_edges)  # All links open
                usages.append([0] * env.num_edges)  # No usage
                factors.append([1.0] * env.num_edges)  # All factors 1.0
    
    except Exception as e:
        print(f"Model-based evaluation failed: {e}")
        print("Falling back to standard evaluation...")
        return standard_evaluation(agent, env, tm_idx, num_episodes)
    
    return rewards, links, usages, factors

def visualize_network(num_nodes, edge_list, link_state, link_usage, capacity_factors, 
                     threshold, tm_idx, config_name, link_capacity=None):
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
        link_capacity: Original link capacities
    """
    # Create directory for visualizations
    os.makedirs("visualizations", exist_ok=True)
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for i in range(num_nodes):
        G.add_node(i)
    
    # Default link capacity
    if link_capacity is None:
        link_capacity = [1.0] * len(edge_list)
    elif isinstance(link_capacity, (int, float)):
        # If link_capacity is a scalar, convert to a list
        link_capacity = [float(link_capacity)] * len(edge_list)
    
    # Add edges with attributes
    for i, (u, v) in enumerate(edge_list):
        # Check if link is open
        if link_state[i] == 1:
            # Calculate effective capacity
            capacity = link_capacity[i] if i < len(link_capacity) else 1.0
            effective_capacity = capacity * capacity_factors[i]
            
            # Calculate utilization
            utilization = link_usage[i] / effective_capacity if effective_capacity > 0 else float('inf')
            
            # Add edge with attributes
            G.add_edge(u, v, 
                      capacity=effective_capacity,
                      usage=link_usage[i],
                      utilization=utilization,
                      capacity_factor=capacity_factors[i])
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Position nodes using spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    # Draw edges with color based on capacity factor
    for (u, v, data) in G.edges(data=True):
        # Skip edges with capacity factors below threshold
        if data.get('capacity_factor', 1.0) < threshold:
            continue
        
        # Determine color based on utilization
        if data.get('utilization', 0) < 0.3:
            color = 'green'  # Low utilization
        elif data.get('utilization', 0) < 0.7:
            color = 'orange'  # Medium utilization
        else:
            color = 'red'  # High utilization
        
        # Edge width based on capacity factor
        width = 1 + 3 * data.get('capacity_factor', 1.0)
        
        # Draw edge
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, edge_color=color, alpha=0.7)
        
        # Edge label with capacity factor
        label = f"{data.get('capacity_factor', 1.0):.2f}"
        nx.draw_networkx_edge_labels(G, pos, edge_labels={
            (u, v): label
        }, font_size=8)
    
    # Title and legend
    plt.title(f"Network Topology (TM {tm_idx}) - Capacity Factors >= {threshold}")
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=4, label='Low Utilization (<30%)'),
        Line2D([0], [0], color='orange', lw=4, label='Medium Utilization (30-70%)'),
        Line2D([0], [0], color='red', lw=4, label='High Utilization (>70%)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Save figure
    plt.savefig(f"visualizations/relaxed_{config_name}_tm{tm_idx}_thresh{threshold}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Network visualization saved to visualizations/relaxed_{config_name}_tm{tm_idx}_thresh{threshold}.png")

def main():
    parser = argparse.ArgumentParser(description='Evaluate a DDPG agent for relaxed network optimization')
    
    # Configuration
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    
    # Model paths
    parser.add_argument('--actor-model', type=str, help='Path to actor model')
    parser.add_argument('--critic-model', type=str, help='Path to critic model')
    parser.add_argument('--predictor-model', type=str, help='Path to state predictor model')
    parser.add_argument('--auto-find-models', action='store_true', help='Automatically find the most recent model files')
    
    # Network architecture
    parser.add_argument('--latent-dim', type=int, default=64, help='Dimension of latent state representation')
    parser.add_argument('--predictor-hidden-dim', type=int, default=256, help='Hidden dimension of state predictor')
    parser.add_argument('--actor-hidden-dim', type=int, default=256, help='Hidden dimension of actor network')
    parser.add_argument('--critic-hidden-dim', type=int, default=256, help='Hidden dimension of critic network')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage instead of CUDA')
    
    # Evaluation settings
    parser.add_argument('--eval-episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--model-based', action='store_true', help='Use model-based evaluation with state predictor')
    parser.add_argument('--random-edge-order', action='store_true', help='Use random edge ordering in environment')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--threshold', type=float, default=0.1, help='Capacity factor threshold for visualization')
    
    args = parser.parse_args()
    evaluate_agent(args)

if __name__ == "__main__":
    main()
