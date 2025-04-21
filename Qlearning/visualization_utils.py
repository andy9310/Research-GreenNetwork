"""
Visualization utilities for network optimization models.

This module provides visualization tools for network topology and traffic matrices,
helping to understand model decisions and network behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from matplotlib.lines import Line2D
import torch

def visualize_network_decisions(edge_list, link_open, link_utilization, node_positions=None, 
                               violated_links=None, title=None, figsize=(10, 8), 
                               show_utilization=True, threshold=1.0):
    """
    Visualize network topology with link decisions and utilization.
    
    Args:
        edge_list: List of edges [(src, dst), ...] 
        link_open: Binary array indicating if links are open (1) or closed (0)
        link_utilization: Array of link utilization values
        node_positions: Dict of node positions for layout (optional)
        violated_links: List of link indices with violations
        title: Plot title
        figsize: Figure size as tuple
        show_utilization: Whether to show utilization values
        threshold: Utilization threshold for coloring (default 1.0)
    
    Returns:
        Figure and axes objects
    """
    # Create graph
    G = nx.Graph()
    
    # Add all nodes first
    all_nodes = set()
    for src, dst in edge_list:
        all_nodes.add(src)
        all_nodes.add(dst)
    
    for node in all_nodes:
        G.add_node(node)
    
    # Get unique nodes for position generation if needed
    if node_positions is None:
        node_positions = nx.spring_layout(G, seed=42)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Setup node drawing
    nx.draw_networkx_nodes(G, node_positions, node_size=700, node_color='lightblue', 
                         edgecolors='black', ax=ax)
    
    # Setup labels
    nx.draw_networkx_labels(G, node_positions, font_size=12, font_weight='bold', ax=ax)
    
    # Draw edges with appropriate colors and styles
    if violated_links is None:
        violated_links = []
    
    # Colors for utilization (from green to red)
    cmap = plt.cm.RdYlGn_r
    
    # Draw each edge with appropriate color and style
    for i, (src, dst) in enumerate(edge_list):
        # Skip if nodes aren't in the positions (shouldn't happen if we generated positions)
        if src not in node_positions or dst not in node_positions:
            continue
        
        is_open = link_open[i] == 1
        is_violated = i in violated_links
        utilization = link_utilization[i] if is_open else 0
        
        # Determine edge color based on utilization and status
        if not is_open:
            edge_color = 'lightgray'  # Closed links
            width = 1.5
            style = 'dashed'
        elif is_violated:
            edge_color = 'red'  # Violated links
            width = 3.0
            style = 'solid'
        else:
            # Color based on utilization (green->yellow->red)
            if show_utilization:
                # Normalize utilization: 0->green, threshold->red
                norm_util = min(utilization / threshold, 1.0)
                edge_color = cmap(norm_util)
            else:
                edge_color = 'blue'  # Open links
            width = 2.0
            style = 'solid'
        
        # Draw edge
        nx.draw_networkx_edges(G, node_positions, edgelist=[(src, dst)], 
                             width=width, edge_color=[edge_color], 
                             style=style, ax=ax)
        
        # Add utilization text if requested
        if show_utilization and is_open:
            # Calculate edge midpoint
            x1, y1 = node_positions[src]
            x2, y2 = node_positions[dst]
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            
            # Add offset to avoid overlapping with edge
            offset = 0.05
            dx = y2 - y1  # perpendicular direction
            dy = x1 - x2  # perpendicular direction
            norm = np.sqrt(dx**2 + dy**2)
            if norm > 0:
                dx, dy = dx/norm * offset, dy/norm * offset
            
            # Position and add text
            text_x, text_y = mid_x + dx, mid_y + dy
            utilization_text = f"{utilization:.2f}"
            text_color = 'black'
            
            # Add background to text for better visibility
            ax.text(text_x, text_y, utilization_text, 
                  color=text_color, fontsize=9, fontweight='bold',
                  ha='center', va='center', 
                  bbox=dict(boxstyle="round,pad=0.3", 
                           facecolor='white', alpha=0.7))
    
    # Set title if provided
    if title:
        ax.set_title(title, fontsize=14)
    
    # Create custom legend
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Open Link'),
        Line2D([0], [0], color='lightgray', lw=1.5, linestyle='dashed', label='Closed Link'),
        Line2D([0], [0], color='red', lw=3, label='Violated Link')
    ]
    
    if show_utilization:
        # Add utilization color bar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, threshold))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.1, 
                          label='Link Utilization')
    
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Remove axes
    ax.set_axis_off()
    
    return fig, ax

def visualize_traffic_matrix(tm, node_names=None, title="Traffic Matrix", cmap='viridis', 
                           figsize=(10, 8), annotate=True):
    """
    Visualize a traffic matrix as a heatmap.
    
    Args:
        tm: Traffic matrix (2D array)
        node_names: List of node names (optional)
        title: Plot title
        cmap: Colormap name
        figsize: Figure size tuple
        annotate: Whether to show values in cells
    
    Returns:
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to numpy array if it's a torch tensor
    if isinstance(tm, torch.Tensor):
        tm = tm.detach().cpu().numpy()
    
    # Create heatmap
    im = ax.imshow(tm, cmap=cmap)
    
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Traffic Demand", rotation=-90, va="bottom")
    
    # We want to show all ticks and label them with the respective list entries
    num_nodes = tm.shape[0]
    if node_names is None:
        node_names = [str(i) for i in range(num_nodes)]
    
    # Set tick labels
    ax.set_xticks(np.arange(num_nodes))
    ax.set_yticks(np.arange(num_nodes))
    ax.set_xticklabels(node_names)
    ax.set_yticklabels(node_names)
    
    # Rotate tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Annotate with values
    if annotate:
        # Loop over data dimensions and create text annotations
        for i in range(num_nodes):
            for j in range(num_nodes):
                text = ax.text(j, i, f"{tm[i, j]:.1f}",
                             ha="center", va="center", 
                             color="white" if tm[i, j] > np.mean(tm) else "black")
    
    ax.set_title(title)
    fig.tight_layout()
    
    return fig, ax

def visualize_evaluation_results(model_name, tm_index, reward, edge_list, final_config, 
                               link_utilization, violations, config):
    """
    Create and save visualizations for model evaluation results.
    
    Args:
        model_name: Name of the model being evaluated
        tm_index: Index of the traffic matrix being evaluated
        reward: Final reward achieved
        edge_list: List of edges in the network
        final_config: Final link configuration (open/closed)
        link_utilization: Link utilization values
        violations: Dict with violation information
        config: Network configuration dictionary
    
    Returns:
        Path to the saved visualization file
    """
    # Create output directory if it doesn't exist
    import os
    os.makedirs("visualizations", exist_ok=True)
    
    # Get current traffic matrix
    tm = np.array(config["tm_list"][tm_index])
    
    # Create figure with subplots (2 row, 2 columns)
    fig = plt.figure(figsize=(18, 12))
    
    # Create subplot for traffic matrix
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=1, rowspan=1)
    # Show traffic matrix
    im = ax1.imshow(tm, cmap='viridis')
    plt.colorbar(im, ax=ax1, label="Traffic Demand")
    
    # Set tick labels
    num_nodes = tm.shape[0]
    node_names = [str(i) for i in range(num_nodes)]
    ax1.set_xticks(np.arange(num_nodes))
    ax1.set_yticks(np.arange(num_nodes))
    ax1.set_xticklabels(node_names)
    ax1.set_yticklabels(node_names)
    ax1.set_title(f"Traffic Matrix {tm_index}")
    
    # Subplot for network visualization
    ax2 = plt.subplot2grid((2, 3), (0, 1), colspan=2, rowspan=2)
    
    # Identify violated links
    violated_links = []
    if violations.get('overloaded', 0) > 0 or violations.get('isolated', 0) > 0:
        for i, util in enumerate(link_utilization):
            if util > 1.0 and final_config[i] == 1:  # Open link that's overloaded
                violated_links.append(i)
    
    # Create network graph to get node positions
    G = nx.Graph()
    for src, dst in edge_list:
        G.add_node(src)
        G.add_node(dst)
        G.add_edge(src, dst)
    
    node_pos = nx.spring_layout(G, seed=42)
    
    # Draw network with decisions
    network_title = f"{model_name} - TM {tm_index} - Reward: {reward:.2f}"
    if violations.get('overloaded', 0) > 0:
        network_title += f" - {violations['overloaded']} Overloaded"
    if violations.get('isolated', 0) > 0:
        network_title += f" - {violations['isolated']} Isolated"
    
    # Visualize network with decisions
    visualize_network_decisions(edge_list, final_config, link_utilization, 
                              node_positions=node_pos,
                              violated_links=violated_links, 
                              title=network_title, 
                              ax=ax2, fig=fig)
    
    # Subplot for link utilization bar chart
    ax3 = plt.subplot2grid((2, 3), (1, 0), colspan=1, rowspan=1)
    
    # Filter to only show open links
    open_links = [i for i, is_open in enumerate(final_config) if is_open == 1]
    open_link_utils = [link_utilization[i] for i in open_links]
    open_link_labels = [f"{edge_list[i][0]}->{edge_list[i][1]}" for i in open_links]
    
    # Sort by utilization for better visualization
    sorted_indices = np.argsort(open_link_utils)
    sorted_utils = [open_link_utils[i] for i in sorted_indices]
    sorted_labels = [open_link_labels[i] for i in sorted_indices]
    
    # Use color mapping (green to red) based on utilization
    colors = plt.cm.RdYlGn_r(np.array(sorted_utils) / config["link_capacity"])
    
    bars = ax3.barh(range(len(sorted_utils)), sorted_utils, color=colors)
    ax3.set_title("Link Utilization (Open Links)")
    ax3.set_xlabel("Utilization")
    ax3.axvline(x=config["link_capacity"], color='red', linestyle='--', 
               label=f"Capacity ({config['link_capacity']})")
    ax3.set_yticks(range(len(sorted_labels)))
    ax3.set_yticklabels(sorted_labels)
    ax3.legend()
    
    # Set main title
    fig.suptitle(f"Evaluation of {model_name} on Traffic Matrix {tm_index}", fontsize=16)
    
    # Add metadata text
    closed_links = sum(1 for link in final_config if link == 0)
    total_links = len(final_config)
    metadata_text = (
        f"Model: {model_name}\n"
        f"Traffic Matrix: {tm_index}\n"
        f"Final Reward: {reward:.2f}\n"
        f"Links Closed: {closed_links}/{total_links}\n"
        f"Overload Violations: {violations.get('overloaded', 0)}\n"
        f"Isolation Violations: {violations.get('isolated', 0)}"
    )
    
    # Add text box with metadata
    fig.text(0.02, 0.02, metadata_text, fontsize=10,
           bbox=dict(facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    output_path = f"visualizations/{model_name}_tm{tm_index}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return output_path
