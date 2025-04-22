"""
Evaluation Script for Latent Predictor Agent

This script evaluates a latent predictor agent in two modes:
1. Standard mode: Using environment interaction to get true next states
2. Model-based mode: Using state prediction to simulate transitions without environment interaction

It compares the performance between both modes and visualizes the results.
"""

import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx

import sys
import os
# Add parent directory to path to import modules from there
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import NetworkEnv
from latent_predictor_agent import LatentPredictorAgent

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def plot_network_state(G, pos, edge_list, link_open, usage, capacity, title, save_path=None):
    """Plot the network state with link status and utilization."""
    plt.figure(figsize=(10, 8))
    
    # Nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, font_size=14)
    
    # Edges with color based on status and width based on utilization
    for i, (u, v) in enumerate(edge_list):
        if link_open[i] == 1:  # Open link
            utilization = usage[i] / capacity if capacity > 0 else 0
            if utilization > 1.0:  # Overloaded
                color = 'red'
                width = 3.0
            else:
                color = 'green'
                width = 1.0 + 2.0 * utilization
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, edge_color=color)
        else:  # Closed link
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=1.0, edge_color='gray', style='dashed')
    
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_traffic_matrix(traffic_matrix, title, save_path=None):
    """Plot traffic matrix as a heatmap."""
    plt.figure(figsize=(8, 6))
    plt.imshow(traffic_matrix, cmap='viridis')
    plt.colorbar(label='Traffic Demand')
    plt.title(title)
    plt.xlabel('Destination Node')
    plt.ylabel('Source Node')
    
    # Add text annotations
    for i in range(traffic_matrix.shape[0]):
        for j in range(traffic_matrix.shape[1]):
            plt.text(j, i, f'{traffic_matrix[i, j]:.1f}', 
                     ha='center', va='center', 
                     color='white' if traffic_matrix[i, j] > np.mean(traffic_matrix) else 'black')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_prediction_accuracy(real_states, predicted_states, title, save_path=None):
    """Plot comparison between real and predicted states."""
    plt.figure(figsize=(12, 6))
    
    # Calculate mean squared error per state component
    mse_per_component = np.mean((real_states - predicted_states) ** 2, axis=0)
    
    # Plot the MSE for each component
    plt.bar(range(len(mse_per_component)), mse_per_component)
    plt.xlabel('State Component Index')
    plt.ylabel('Mean Squared Error')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def evaluate_agent(args):
    """Evaluate a DQN agent with latent state encoding and prediction capabilities."""
    # --- Load configuration ---
    config_path = args.config
    config = load_config(config_path)
    config_name = os.path.basename(config_path).split('.')[0]
    print(f"Loaded configuration from {config_path}")
    
    # Extract parameters from config
    num_nodes = config["num_nodes"]
    adj_matrix = config["adj_matrix"]
    edge_list = config["edge_list"]
    tm_list = config["tm_list"]
    node_props = config.get("node_props", {})
    link_capacity = config["link_capacity"]
    max_edges = config.get("max_edges", len(edge_list))
    
    # --- Setup environment ---
    env = NetworkEnv(
        adj_matrix=adj_matrix,
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
    
    # Create a shadow environment for comparison
    shadow_env = NetworkEnv(
        adj_matrix=adj_matrix,
        edge_list=edge_list,
        tm_list=tm_list,
        node_props=node_props,
        num_nodes=num_nodes,
        link_capacity=link_capacity,
        max_edges=max_edges,
        random_edge_order=args.random_edge_order,
        seed=args.seed
    )
    
    # Create a networkx graph for visualization
    G = nx.Graph()
    for i in range(num_nodes):
        G.add_node(i)
    for i, (u, v) in enumerate(edge_list):
        G.add_edge(u, v)
    
    # Get node positions for visualization
    pos = nx.spring_layout(G, seed=42)  # Use seed for consistent layout
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Verify the environment is providing states with the expected dimensions
    initial_state = env.reset()
    if isinstance(initial_state, tuple):
        # Newer gym versions return (state, info)
        initial_state = initial_state[0]
    print(f"Initial state shape from environment: {initial_state.shape}")
    if initial_state.shape[0] != state_dim:
        print(f"WARNING: Environment state dimension ({initial_state.shape[0]}) doesn't match expected dimension ({state_dim})")
        print(f"Adjusting state_dim to match environment")
        state_dim = initial_state.shape[0]
    
    # --- Setup agent ---
    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize agent
    agent = LatentPredictorAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=args.latent_dim,
        predictor_hidden_dim=args.predictor_hidden_dim,
        enable_predictor=args.model_based,
        architecture=args.architecture,
        device=device
    )
    
    # Load model weights
    if args.model_based and args.predictor_model:
        agent.load(args.model, args.predictor_model)
        print(f"Loaded model from {args.model} and predictor from {args.predictor_model}")
    else:
        agent.load(args.model)
        print(f"Loaded model from {args.model}")
    
    # Create visualizations directory
    os.makedirs("visualizations", exist_ok=True)
    
    # Specific traffic matrices to evaluate
    if args.tm_indices:
        tm_indices = [int(idx) for idx in args.tm_indices.split(',')]
    else:
        tm_indices = list(range(len(tm_list)))
    
    # Evaluation results
    standard_results = []
    model_based_results = []
    prediction_accuracy = []
    
    # Process each traffic matrix
    for tm_idx in tm_indices:
        if tm_idx >= len(tm_list):
            print(f"Error: Traffic matrix index {tm_idx} out of range (0-{len(tm_list)-1})")
            continue
        
        env.current_tm_idx = tm_idx
        shadow_env.current_tm_idx = tm_idx
        current_tm = np.array(tm_list[tm_idx])
        
        print(f"\n--- Evaluating Traffic Matrix {tm_idx} ---")
        
        # Visualization of traffic matrix
        plot_traffic_matrix(
            current_tm,
            title=f"Traffic Matrix {tm_idx}",
            save_path=f"visualizations/tm_{tm_idx}.png"
        )
        
        # ----------- Standard Evaluation (with environment interaction) -----------
        print("Running standard evaluation...")
        
        state, _, _, _, _ = env.reset()
        episode_reward = 0
        done = False
        episode_violations = {'isolated': 0, 'overloaded': 0}
        
        # Track action history and states
        action_history = []
        standard_states = [state.copy()]
        
        # Standard episode loop
        while not done:
            # Get action from the agent
            with torch.no_grad():
                action = agent.act(state, epsilon=0.0)  # No exploration during evaluation
            
            # Take action in environment
            next_state, reward, done, truncated, info = env.step(action)
            
            # Record action
            action_history.append((env.current_edge_idx, action))
            
            # Check for violations
            if done and 'violation' in info:
                if info['violation'] == 'isolated':
                    episode_violations['isolated'] += 1
                elif info['violation'] == 'overloaded':
                    episode_violations['overloaded'] += 1
            
            # Update state and reward
            state = next_state
            standard_states.append(state.copy())
            episode_reward += reward
            
            # End of episode
            if done:
                break
        
        # Record results for standard evaluation
        standard_result = {
            'tm_idx': tm_idx,
            'reward': episode_reward,
            'links_closed': np.sum(env.link_open == 0),
            'total_links': env.num_edges,
            'isolated_violations': episode_violations['isolated'],
            'overloaded_violations': episode_violations['overloaded'],
            'has_violation': any(episode_violations.values()),
            'steps': len(action_history)
        }
        standard_results.append(standard_result)
        
        # Visualize final network state
        standard_title = (
            f"TM {tm_idx}: Standard Evaluation "
            f"(Reward={episode_reward:.2f}, "
            f"Closed={np.sum(env.link_open == 0)}/{env.num_edges})"
        )
        if any(episode_violations.values()):
            violations_str = ", ".join([f"{v} {k}" for k, v in episode_violations.items() if v > 0])
            standard_title += f"\nViolations: {violations_str}"
        
        plot_network_state(
            G, pos, env.edge_list, env.link_open, env.usage, env.link_capacity, 
            standard_title,
            save_path=f"visualizations/standard_tm{tm_idx}_final.png"
        )
        
        # Skip model-based evaluation if not enabled
        if not args.model_based or not args.predictor_model:
            print("Model-based evaluation skipped (not enabled)")
            continue
        
        # ----------- Model-Based Evaluation (using state predictor) -----------
        print("Running model-based evaluation...")
        
        # Reset environment to get initial state
        initial_state, _, _, _, _ = shadow_env.reset()
        
        # Use initial state as the starting point
        predicted_state = initial_state.copy()
        model_reward = 0
        model_done = False
        model_violations = {'isolated': 0, 'overloaded': 0}
        
        # Track states for comparison
        predicted_states = [predicted_state.copy()]
        model_actions = []
        model_steps = 0
        
        # In model-based evaluation, we only interact with the environment
        # to get the initial state, then use the predictor for future states
        while not model_done and model_steps < args.max_steps:
            model_steps += 1
            
            # Get action from the agent based on predicted state
            with torch.no_grad():
                action = agent.act(predicted_state, epsilon=0.0)
            
            model_actions.append((model_steps, action))
            
            # For validation: Take the same action in shadow environment
            # (this wouldn't happen in a real deployment, just for comparison)
            shadow_next_state, shadow_reward, shadow_done, shadow_truncated, shadow_info = shadow_env.step(action)
            
            # Predict next state using the state predictor
            next_predicted_state = agent.predict_next_state(predicted_state, action)
            
            # Track prediction accuracy
            prediction_error = np.mean((next_predicted_state - shadow_next_state) ** 2)
            prediction_accuracy.append((predicted_state.copy(), next_predicted_state.copy(), shadow_next_state.copy(), prediction_error))
            
            # Check for terminal conditions in shadow environment
            if shadow_done:
                model_done = True
                model_reward += shadow_reward  # Use real reward for accurate comparison
                
                if 'violation' in shadow_info:
                    if shadow_info['violation'] == 'isolated':
                        model_violations['isolated'] += 1
                    elif shadow_info['violation'] == 'overloaded':
                        model_violations['overloaded'] += 1
            else:
                model_reward += shadow_reward
            
            # Update state
            predicted_state = next_predicted_state
            predicted_states.append(predicted_state.copy())
            
            # End conditions
            if model_steps >= env.num_edges:
                model_done = True
        
        # Record results for model-based evaluation
        model_result = {
            'tm_idx': tm_idx,
            'reward': model_reward,
            'links_closed': np.sum(shadow_env.link_open == 0),
            'total_links': shadow_env.num_edges,
            'isolated_violations': model_violations['isolated'],
            'overloaded_violations': model_violations['overloaded'],
            'has_violation': any(model_violations.values()),
            'steps': model_steps
        }
        model_based_results.append(model_result)
        
        # Convert lists for visualization
        predicted_states_array = np.array(predicted_states)
        real_states_array = np.array(standard_states[:len(predicted_states)])
        
        # Visualize prediction accuracy if we have enough states
        if len(predicted_states) > 1 and len(standard_states) > 1:
            min_length = min(len(predicted_states), len(standard_states))
            plot_prediction_accuracy(
                real_states_array[:min_length], 
                predicted_states_array[:min_length],
                title=f"TM {tm_idx}: State Prediction Accuracy",
                save_path=f"visualizations/prediction_accuracy_tm{tm_idx}.png"
            )
        
        # Visualize final network state from model-based evaluation
        model_title = (
            f"TM {tm_idx}: Model-Based Evaluation "
            f"(Reward={model_reward:.2f}, "
            f"Closed={np.sum(shadow_env.link_open == 0)}/{shadow_env.num_edges})"
        )
        if any(model_violations.values()):
            violations_str = ", ".join([f"{v} {k}" for k, v in model_violations.items() if v > 0])
            model_title += f"\nViolations: {violations_str}"
        
        plot_network_state(
            G, pos, shadow_env.edge_list, shadow_env.link_open, shadow_env.usage, shadow_env.link_capacity, 
            model_title,
            save_path=f"visualizations/model_based_tm{tm_idx}_final.png"
        )
        
        # Print results for this traffic matrix
        print("\nStandard Evaluation Results:")
        standard_status = "❌ (Violation)" if standard_result['has_violation'] else "✅ (Success)"
        print(f"TM {tm_idx}: Reward={standard_result['reward']:.2f}, Closed={standard_result['links_closed']}/{standard_result['total_links']}, Status={standard_status}")
        if standard_result['has_violation']:
            violations = []
            if standard_result['isolated_violations'] > 0:
                violations.append(f"{standard_result['isolated_violations']} isolated")
            if standard_result['overloaded_violations'] > 0:
                violations.append(f"{standard_result['overloaded_violations']} overloaded")
            print(f"  Violations: {', '.join(violations)}")
        
        if args.model_based:
            print("\nModel-Based Evaluation Results:")
            model_status = "❌ (Violation)" if model_result['has_violation'] else "✅ (Success)"
            print(f"TM {tm_idx}: Reward={model_result['reward']:.2f}, Closed={model_result['links_closed']}/{model_result['total_links']}, Status={model_status}")
            if model_result['has_violation']:
                violations = []
                if model_result['isolated_violations'] > 0:
                    violations.append(f"{model_result['isolated_violations']} isolated")
                if model_result['overloaded_violations'] > 0:
                    violations.append(f"{model_result['overloaded_violations']} overloaded")
                print(f"  Violations: {', '.join(violations)}")
            
            # Calculate average prediction error
            if prediction_accuracy:
                avg_pred_error = np.mean([p[3] for p in prediction_accuracy])
                print(f"Average prediction error: {avg_pred_error:.6f}")
    
    # Compute and print overall statistics
    if standard_results:
        print("\n=== Overall Standard Evaluation Results ===")
        avg_reward = np.mean([r['reward'] for r in standard_results])
        avg_links_closed = np.mean([r['links_closed'] for r in standard_results])
        violations_count = sum(1 for r in standard_results if r['has_violation'])
        
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Links Closed: {avg_links_closed:.2f} / {standard_results[0]['total_links']}")
        print(f"Traffic Matrices with Violations: {violations_count}/{len(standard_results)} ({violations_count/len(standard_results)*100:.1f}%)")
    
    if model_based_results:
        print("\n=== Overall Model-Based Evaluation Results ===")
        avg_reward = np.mean([r['reward'] for r in model_based_results])
        avg_links_closed = np.mean([r['links_closed'] for r in model_based_results])
        violations_count = sum(1 for r in model_based_results if r['has_violation'])
        
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Links Closed: {avg_links_closed:.2f} / {model_based_results[0]['total_links']}")
        print(f"Traffic Matrices with Violations: {violations_count}/{len(model_based_results)} ({violations_count/len(model_based_results)*100:.1f}%)")
        
        # Compare prediction error
        if prediction_accuracy:
            avg_total_pred_error = np.mean([p[3] for p in prediction_accuracy])
            print(f"\nAverage State Prediction Error: {avg_total_pred_error:.6f}")
    
    print(f"\nVisualizations saved to the 'visualizations/' directory")
    
    return standard_results, model_based_results

def main():
    parser = argparse.ArgumentParser(description='Evaluate Latent Predictor Agent')
    
    # Configuration
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True, help='Path to saved model')
    parser.add_argument('--predictor-model', type=str, help='Path to saved predictor model (required for model-based evaluation)')
    
    # Model parameters
    parser.add_argument('--latent-dim', type=int, default=64, help='Dimension of latent state representation')
    parser.add_argument('--predictor-hidden-dim', type=int, default=256, help='Hidden dimension of state predictor')
    parser.add_argument('--architecture', type=str, default='mlp', choices=['mlp', 'fatmlp'], help='Network architecture type')
    parser.add_argument('--model-based', action='store_true', help='Use model-based evaluation')
    
    # Evaluation parameters
    parser.add_argument('--tm-indices', type=str, help='Comma-separated list of traffic matrix indices to evaluate')
    parser.add_argument('--max-steps', type=int, default=100, help='Maximum steps to simulate in model-based evaluation')
    parser.add_argument('--random-edge-order', action='store_true', help='Use random edge ordering in environment')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    evaluate_agent(args)

if __name__ == "__main__":
    main()
