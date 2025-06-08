#!/usr/bin/env python
"""
Edge Importance Visualization Script
------------------------------------
This script visualizes the edge importance data collected during training.
It shows reward contributions, violations, and decision patterns for each edge.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

def load_data(filepath):
    """Load edge importance data from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def create_edge_labels(edge_list):
    """Create readable edge labels from edge list."""
    return [f"({u},{v})" for u, v in edge_list]

def plot_rewards(ax, rewards, edge_labels):
    """Plot rewards associated with each edge."""
    y_pos = np.arange(len(edge_labels))
    
    # Sort by reward value
    sorted_indices = np.argsort(rewards)
    sorted_rewards = [rewards[i] for i in sorted_indices]
    sorted_labels = [edge_labels[i] for i in sorted_indices]
    
    colors = ['#3498db' if r >= 0 else '#e74c3c' for r in sorted_rewards]
    
    bars = ax.barh(y_pos, sorted_rewards, align='center', color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_labels)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Total Reward Contribution')
    ax.set_title('Edge Reward Contribution')
    
    # Add value labels to the bars
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width if width >= 0 else width - 1000
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                f'{width:.0f}', va='center')
    
    return sorted_indices

def plot_violations(ax, violations, edge_labels, sorted_indices=None):
    """Plot violations for each edge."""
    isolation = np.array(violations["isolation"])
    overloaded = np.array(violations["overloaded"])
    
    # Ensure we use the same order as in rewards plot if provided
    if sorted_indices is not None:
        edge_labels = [edge_labels[i] for i in sorted_indices]
        isolation = isolation[sorted_indices]
        overloaded = overloaded[sorted_indices]
    
    x = np.arange(len(edge_labels))
    width = 0.35
    
    ax.bar(x - width/2, isolation, width, label='Isolation', color='#3498db')
    ax.bar(x + width/2, overloaded, width, label='Overloaded', color='#e74c3c')
    
    ax.set_ylabel('Number of Violations')
    ax.set_title('Edge Violations')
    ax.set_xticks(x)
    ax.set_xticklabels(edge_labels, rotation=45, ha='right')
    ax.legend()
    
    # Add gridlines
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

def plot_decisions(ax, decisions, edge_labels, sorted_indices=None):
    """Plot decision counts for each edge."""
    # Extract decision data
    open_counts = np.array(decisions["open"])
    close_success = np.array(decisions["close_success"])
    close_failure = np.array(decisions["close_failure"])
    
    # Ensure we use the same order as in rewards plot if provided
    if sorted_indices is not None:
        edge_labels = [edge_labels[i] for i in sorted_indices]
        open_counts = open_counts[sorted_indices]
        close_success = close_success[sorted_indices]
        close_failure = close_failure[sorted_indices]
    
    x = np.arange(len(edge_labels))
    width = 0.25
    
    # Plot bars
    ax.bar(x - width, open_counts, width, label='Keep Open', color='#2ecc71')
    ax.bar(x, close_success, width, label='Close Success', color='#3498db')
    ax.bar(x + width, close_failure, width, label='Close Failure', color='#e74c3c')
    
    ax.set_ylabel('Count')
    ax.set_title('Edge Decisions')
    ax.set_xticks(x)
    ax.set_xticklabels(edge_labels, rotation=45, ha='right')
    ax.legend()
    
    # Add gridlines
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

def calculate_importance_score(data):
    """Calculate a composite importance score for each edge."""
    rewards = np.array(data["edge_rewards"])
    isolation = np.array(data["edge_violations"]["isolation"])
    overloaded = np.array(data["edge_violations"]["overloaded"])
    
    # Normalize each metric to [0,1] range
    if np.max(rewards) - np.min(rewards) > 0:
        norm_rewards = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards))
    else:
        norm_rewards = np.zeros_like(rewards)
    
    if np.max(isolation) > 0:
        norm_isolation = isolation / np.max(isolation) if np.max(isolation) > 0 else np.zeros_like(isolation)
    else:
        norm_isolation = np.zeros_like(isolation)
        
    if np.max(overloaded) > 0:
        norm_overloaded = overloaded / np.max(overloaded) if np.max(overloaded) > 0 else np.zeros_like(overloaded)
    else:
        norm_overloaded = np.zeros_like(overloaded)
    
    # Calculate importance score (rewards are positive, violations are negative)
    # We weigh isolation violations higher as they're usually more critical
    importance_score = norm_rewards - (0.7 * norm_isolation + 0.3 * norm_overloaded)
    
    return importance_score

def plot_importance_score(ax, importance_score, edge_labels, sorted_indices=None):
    """Plot composite importance score."""
    # Use provided sorting or sort by importance score
    if sorted_indices is None:
        sorted_indices = np.argsort(importance_score)
        
    sorted_scores = importance_score[sorted_indices]
    sorted_labels = [edge_labels[i] for i in sorted_indices]
    
    y_pos = np.arange(len(sorted_labels))
    colors = ['#2ecc71' if s >= 0 else '#e74c3c' for s in sorted_scores]
    
    bars = ax.barh(y_pos, sorted_scores, align='center', color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_labels)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Importance Score')
    ax.set_title('Overall Edge Importance')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 0.01 if width >= 0 else width - 0.05
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}', va='center')
    
    # Add a vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    return sorted_indices

def visualize_data(data, save_path=None, show_plot=True):
    """Visualize edge importance data with multiple plots."""
    edge_labels = create_edge_labels(data["edge_list"])
    
    # Calculate importance score
    importance_score = calculate_importance_score(data)
    
    # Set up the figure with GridSpec for better control
    plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 2, height_ratios=[1, 1, 1])
    
    # Reward contribution plot
    ax_rewards = plt.subplot(gs[0, 0])
    sorted_indices = plot_rewards(ax_rewards, data["edge_rewards"], edge_labels)
    
    # Importance score plot
    ax_importance = plt.subplot(gs[0, 1])
    plot_importance_score(ax_importance, importance_score, edge_labels, sorted_indices)
    
    # Violations plot
    ax_violations = plt.subplot(gs[1, :])
    plot_violations(ax_violations, data["edge_violations"], edge_labels, sorted_indices)
    
    # Decisions plot
    ax_decisions = plt.subplot(gs[2, :])
    plot_decisions(ax_decisions, data["edge_decisions"], edge_labels, sorted_indices)
    
    plt.tight_layout()
    
    # Add a title with data file info
    plt.suptitle(f"Edge Importance Analysis", fontsize=16, y=0.995)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

def compare_random_vs_fixed(random_file, fixed_file, save_path=None, show_plot=True):
    """Compare edge importance between random and fixed ordering."""
    # Load both datasets
    random_data = load_data(random_file)
    fixed_data = load_data(fixed_file)
    
    # Get edge labels
    edge_labels = create_edge_labels(random_data["edge_list"])
    
    # Calculate importance scores
    random_score = calculate_importance_score(random_data)
    fixed_score = calculate_importance_score(fixed_data)
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Plot comparison of importance scores
    ax1 = plt.subplot(211)
    x = np.arange(len(edge_labels))
    width = 0.35
    
    ax1.bar(x - width/2, random_score, width, label='Random Order', color='#3498db')
    ax1.bar(x + width/2, fixed_score, width, label='Fixed Order', color='#e74c3c')
    
    ax1.set_ylabel('Importance Score')
    ax1.set_title('Edge Importance Comparison: Random vs Fixed Order')
    ax1.set_xticks(x)
    ax1.set_xticklabels(edge_labels, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Plot correlation
    ax2 = plt.subplot(212)
    ax2.scatter(random_score, fixed_score, alpha=0.7, s=100)
    
    # Add edge labels to points
    for i, label in enumerate(edge_labels):
        ax2.annotate(label, (random_score[i], fixed_score[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    # Add diagonal line for reference
    min_val = min(np.min(random_score), np.min(fixed_score))
    max_val = max(np.max(random_score), np.max(fixed_score))
    padding = (max_val - min_val) * 0.1
    ax2.plot([min_val-padding, max_val+padding], [min_val-padding, max_val+padding], 
             'k--', alpha=0.5, label='Equal Importance')
    
    ax2.set_xlabel('Importance Score (Random Order)')
    ax2.set_ylabel('Importance Score (Fixed Order)')
    ax2.set_title('Correlation of Edge Importance')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    
    # Add a title
    plt.suptitle(f"Random vs Fixed Edge Order Comparison", fontsize=16, y=0.995)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison visualization saved to {save_path}")
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

def main():
    """Main function to run the visualization."""
    parser = argparse.ArgumentParser(description="Visualize edge importance data")
    parser.add_argument("--file", type=str, help="Path to edge importance JSON file")
    parser.add_argument("--compare", action="store_true", help="Compare random vs fixed edge order")
    parser.add_argument("--random-file", type=str, help="Path to random edge order JSON file")
    parser.add_argument("--fixed-file", type=str, help="Path to fixed edge order JSON file")
    parser.add_argument("--save", type=str, help="Path to save visualization (e.g., 'viz.png')")
    parser.add_argument("--no-show", action="store_true", help="Don't show the plot (just save)")
    args = parser.parse_args()
    
    # Default paths
    edge_importance_dir = "edge_importance"
    
    # If no file is specified, try to find JSON files in the edge_importance directory
    if not args.file and not args.compare:
        if os.path.exists(edge_importance_dir):
            json_files = [f for f in os.listdir(edge_importance_dir) if f.endswith('.json')]
            if json_files:
                args.file = os.path.join(edge_importance_dir, json_files[0])
                print(f"No file specified, using: {args.file}")
            else:
                print(f"No JSON files found in {edge_importance_dir}")
                return
        else:
            print(f"Edge importance directory {edge_importance_dir} not found")
            return
    
    # For compare mode, find random and fixed files if not specified
    if args.compare and not (args.random_file and args.fixed_file):
        if os.path.exists(edge_importance_dir):
            json_files = [f for f in os.listdir(edge_importance_dir) if f.endswith('.json')]
            random_files = [f for f in json_files if 'randomTrue' in f]
            fixed_files = [f for f in json_files if 'randomFalse' in f]
            
            if random_files and not args.random_file:
                args.random_file = os.path.join(edge_importance_dir, random_files[0])
                print(f"Using random edge order file: {args.random_file}")
            
            if fixed_files and not args.fixed_file:
                args.fixed_file = os.path.join(edge_importance_dir, fixed_files[0])
                print(f"Using fixed edge order file: {args.fixed_file}")
    
    # Determine save path if not provided
    if args.save is None:
        if args.compare:
            args.save = "edge_importance_comparison.png"
        elif args.file:
            base_name = os.path.basename(args.file).replace('.json', '')
            args.save = f"{base_name}_visualization.png"
    
    # Run visualization
    if args.compare and args.random_file and args.fixed_file:
        compare_random_vs_fixed(args.random_file, args.fixed_file, args.save, not args.no_show)
    elif args.file:
        data = load_data(args.file)
        visualize_data(data, args.save, not args.no_show)
    else:
        print("Error: Please provide either a file to visualize or both random and fixed files to compare")

if __name__ == "__main__":
    main()
