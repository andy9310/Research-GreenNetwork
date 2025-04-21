"""
Evaluation script for the Traffic Matrix Encoder model

This script evaluates the performance of a model trained with
Traffic Matrix Representation Learning on various traffic matrices.
"""

import os
import sys
import json
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter

from tm_agent import TMEnhancedDQNAgent
from env import NetworkEnv
from visualization_utils import visualize_network_decisions, visualize_traffic_matrix, visualize_evaluation_results

def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def evaluate_model(args):
    """Evaluate the TM-enhanced model on specific traffic matrices."""
    # Load configuration
    config_path = args.config
    print(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    # Extract parameters
    num_nodes = config["num_nodes"]
    adj_matrix = config["adj_matrix"]
    edge_list = config["edge_list"]
    tm_list = config["tm_list"]
    node_props = config.get("node_props", {})
    link_capacity = config["link_capacity"]
    max_edges = config.get("max_edges", len(edge_list))
    
    # Create environment
    env = NetworkEnv(
        adj_matrix=adj_matrix,
        edge_list=edge_list,
        tm_list=tm_list,
        node_props=node_props,
        num_nodes=num_nodes,
        link_capacity=link_capacity,
        max_edges=max_edges,
        seed=args.seed
    )
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using {device} for evaluation")
    
    # Get state and action dimensions
    state_dim = len(env._get_observation())
    action_dim = env.action_space.n
    
    # Load the model
    try:
        # Load model
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # Create an agent instance
        agent = TMEnhancedDQNAgent(
            state_dim=checkpoint.get('state_dim', state_dim),
            action_dim=checkpoint.get('action_dim', action_dim),
            num_nodes=checkpoint.get('num_nodes', num_nodes),
            tm_embedding_dim=checkpoint.get('tm_embedding_dim', 64),
            device=device
        )
        
        # Load the saved weights
        agent.qnetwork_local.load_state_dict(checkpoint['qnetwork_state_dict'])
        agent.qnetwork_target.load_state_dict(checkpoint['qnetwork_state_dict'])
        
        print(f"Model successfully loaded from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create visualization directory if needed
    if args.visualize:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"Visualizations will be saved to {args.save_dir}/")
    
    # Set to evaluation mode
    agent.qnetwork_local.eval()
    
    # Specify which traffic matrix to evaluate on
    if args.tm_index is not None:
        tm_indices = [args.tm_index]
    else:
        tm_indices = list(range(len(tm_list)))
    
    # Storage for all results
    all_tm_results = []
    
    # Process each traffic matrix
    for tm_idx in tm_indices:
        if tm_idx >= len(tm_list):
            print(f"Error: Traffic matrix index {tm_idx} out of range (0-{len(tm_list)-1})")
            continue
        
        env.current_tm_idx = tm_idx
        current_tm = np.array(tm_list[tm_idx])
        
        print(f"\n--- Evaluating Traffic Matrix {tm_idx} ---")
        
        # Run a single evaluation episode per traffic matrix
        state, _, _, _, _ = env.reset()
        episode_reward = 0
        done = False
        episode_violations = {'isolated': 0, 'overloaded': 0}
        
        # Track action history and states for visualization
        action_history = []
        state_history = [state.copy()]
        link_utilization = None
        
        while not done:
            # Get action from the agent
            with torch.no_grad():
                action = agent.act(state, current_tm, epsilon=0.0)  # No exploration during evaluation
            
            # Record action
            action_history.append(action)
            
            # Take action in environment
            next_state, reward, done, _, info = env.step(action)
            
            # Record state
            state_history.append(next_state.copy())
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            # Track violations
            if info.get('violation') == 'isolated':
                episode_violations['isolated'] += 1
            elif info.get('violation') == 'overloaded':
                episode_violations['overloaded'] += 1
            
            # Get link utilization for visualization
            if 'link_utilization' in info:
                link_utilization = info['link_utilization']
        
        # Calculate final configuration
        final_config = env.link_open.copy()
        num_closed = sum(1 for link in final_config if link == 0)
        
        # If link utilization not in info, get it from the environment
        if link_utilization is None:
            # The network environment stores link usage in self.usage
            # We need to normalize by the link capacity to get utilization
            link_utilization = np.zeros(env.num_edges)
            
            # Force a recalculation of link usage based on final configuration
            # First, save the current state
            original_link_open = env.link_open.copy()
            
            # Set the link configuration to our final configuration
            env.link_open = final_config.copy()
            
            # Update link usage with this configuration
            env._update_link_usage()
            
            # Get the updated usage values and normalize by capacity
            for i in range(env.num_edges):
                if env.link_open[i] == 1:  # Only consider open links
                    link_utilization[i] = env.usage[i] / env.link_capacity
                    
            # Restore the original state
            env.link_open = original_link_open
        
        # Print summary for this traffic matrix
        print(f"Traffic Matrix {tm_idx}: Reward={episode_reward:.2f}, Links Closed={num_closed}/{env.num_edges}")
        print(f"Violations: {episode_violations}")
        
        # Store results for this traffic matrix
        tm_result = {
            'tm_idx': tm_idx,
            'reward': episode_reward,
            'final_config': final_config,
            'link_utilization': link_utilization,
            'violations': episode_violations,
            'num_closed': num_closed
        }
        all_tm_results.append(tm_result)
        
        # Generate visualization if requested
        if args.visualize:
            model_name = "TM_Encoder"
            output_path = os.path.join(args.save_dir, f"{model_name}_tm{tm_idx}.png")
            
            # Create visualization
            fig = plt.figure(figsize=(18, 12))
            plt.subplots_adjust(wspace=0.3, hspace=0.3)
            
            # Traffic Matrix plot
            ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=1, rowspan=1)
            visualize_traffic_matrix(current_tm, title=f"Traffic Matrix {tm_idx}")
            
            # Network plot
            ax2 = plt.subplot2grid((2, 3), (0, 1), colspan=2, rowspan=2)
            
            # Identify violated links
            violated_links = []
            for i, util in enumerate(link_utilization):
                if util > 1.0 and final_config[i] == 1:  # Open link that's overloaded
                    violated_links.append(i)
            
            # Create network visualization title
            title = f"{model_name} - Traffic Matrix {tm_idx} - Reward: {episode_reward:.2f}"
            if episode_violations.get('overloaded', 0) > 0:
                title += f" - {episode_violations['overloaded']} Overloaded"
            if episode_violations.get('isolated', 0) > 0:
                title += f" - {episode_violations['isolated']} Isolated"
            
            # Draw network visualization
            visualize_network_decisions(
                edge_list=edge_list,
                link_open=final_config,
                link_utilization=link_utilization,
                violated_links=violated_links,
                title=title
            )
            
            # Link utilization plot
            ax3 = plt.subplot2grid((2, 3), (1, 0), colspan=1, rowspan=1)
            
            # Filter to only show open links
            open_links = [i for i, is_open in enumerate(final_config) if is_open == 1]
            if open_links:  # Only if there are open links
                open_link_utils = [link_utilization[i] for i in open_links]
                open_link_labels = [f"{edge_list[i][0]}->{edge_list[i][1]}" for i in open_links]
                
                # Sort by utilization
                sorted_indices = np.argsort(open_link_utils)
                sorted_utils = [open_link_utils[i] for i in sorted_indices]
                sorted_labels = [open_link_labels[i] for i in sorted_indices]
                
                # Use color mapping based on utilization
                colors = plt.cm.RdYlGn_r(np.array(sorted_utils))
                
                # Create horizontal bar chart
                bars = ax3.barh(range(len(sorted_utils)), sorted_utils, color=colors)
                ax3.set_title("Link Utilization (Open Links)")
                ax3.set_xlabel("Utilization")
                ax3.axvline(x=1.0, color='red', linestyle='--', label="Capacity Threshold")
                ax3.set_yticks(range(len(sorted_labels)))
                ax3.set_yticklabels(sorted_labels)
                ax3.legend()
            else:
                ax3.set_title("No Open Links")
            
            # Add metadata text
            metadata_text = (
                f"Model: {model_name}\n"
                f"Traffic Matrix: {tm_idx}\n"
                f"Traffic Matrix Embedding Size: {agent.tm_embedding_dim}\n"
                f"Final Reward: {episode_reward:.2f}\n"
                f"Links Closed: {num_closed}/{env.num_edges}\n"
                f"Overload Violations: {episode_violations.get('overloaded', 0)}\n"
                f"Isolation Violations: {episode_violations.get('isolated', 0)}"
            )
            
            # Add text box with metadata
            fig.text(0.02, 0.02, metadata_text, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.8))
            
            # Save figure
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Visualization saved to {output_path}")
    
    # Print overall summary
    print("\n=== Overall Evaluation Summary ===\n")
    if all_tm_results:
        # Compile aggregated statistics
        avg_reward = np.mean([r['reward'] for r in all_tm_results])
        avg_closed = np.mean([r['num_closed'] for r in all_tm_results])
        tm_with_violations = sum(1 for r in all_tm_results if r['violations']['isolated'] > 0 or r['violations']['overloaded'] > 0)
        total_isolated = sum(r['violations']['isolated'] for r in all_tm_results)
        total_overloaded = sum(r['violations']['overloaded'] for r in all_tm_results)
        
        # Print summary
        print(f"Evaluated on {len(all_tm_results)} traffic matrices")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Links Closed: {avg_closed:.2f}/{env.num_edges}")
        print(f"Traffic Matrices with Violations: {tm_with_violations}/{len(all_tm_results)}")
        print(f"Total Isolation Violations: {total_isolated}")
        print(f"Total Overload Violations: {total_overloaded}")
        
        # Print individual TM results
        print("\nResults by Traffic Matrix:")
        for result in all_tm_results:
            tm_idx = result['tm_idx']
            reward = result['reward']
            num_closed = result['num_closed']
            violations = result['violations']
            has_violation = violations['isolated'] > 0 or violations['overloaded'] > 0
            
            status = "❌ (Violation)" if has_violation else "✅ (Valid)"
            print(f"TM {tm_idx}: Reward={reward:.2f}, Closed={num_closed}/{env.num_edges}, Status={status}")
            
            if has_violation:
                violation_details = []
                if violations['isolated'] > 0:
                    violation_details.append(f"{violations['isolated']} isolated")
                if violations['overloaded'] > 0:
                    violation_details.append(f"{violations['overloaded']} overloaded")
                print(f"  Violations: {', '.join(violation_details)}")
        
        if args.visualize:
            print(f"\nVisualizations saved to {args.save_dir}/")
    else:
        print("No traffic matrices were evaluated successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Traffic Matrix Encoder model')
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON')
    parser.add_argument('--model-path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--tm-index', type=int, default=None, help='Specific traffic matrix index to evaluate')
    parser.add_argument('--episodes', type=int, default=1, help='Number of evaluation episodes per traffic matrix')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations of model decisions')
    parser.add_argument('--save-dir', type=str, default='visualizations', help='Directory to save visualizations')
    
    args = parser.parse_args()
    evaluate_model(args)
