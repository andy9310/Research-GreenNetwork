"""
Generate additional traffic matrices for the 5-node network configuration.
This script generates traffic matrices that don't cause any link overloads
when all links are open.
"""

import os
import sys
import json
import numpy as np
import networkx as nx
import random
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Qlearning.env import NetworkEnv  # Import environment for routing check

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def save_config(config, config_path):
    """Save configuration to JSON file."""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def check_traffic_matrix_feasibility(tm, env):
    """Check if a traffic matrix is feasible (no overloads when all links are open)."""
    # Set traffic matrix
    env.traffic = np.array(tm)
    
    # Make sure all links are open
    env.link_open = np.ones(env.num_edges, dtype=int)
    
    # Update link usage and check if there are violations
    routing_successful, G_open = env._update_link_usage()
    isolated, overloaded, num_overloaded = env._check_violations(routing_successful, G_open, epsilon=0.0)
    
    # Get details about the highest loaded link
    max_usage = 0
    max_usage_link = None
    if env.usage.size > 0:
        max_usage = np.max(env.usage)
        max_usage_idx = np.argmax(env.usage)
        max_usage_link = env.edge_list[max_usage_idx]
        
    feasible = not (isolated or overloaded)
    
    max_usage_percent = 0
    if max_usage_link is not None:
        edge_u, edge_v = max_usage_link
        capacity = env.graph[edge_u][edge_v]['capacity']
        if capacity > 0:
            max_usage_percent = max_usage / capacity * 100
    
    return {
        "feasible": feasible,
        "max_usage": max_usage,
        "max_usage_link": max_usage_link,
        "max_usage_percent": max_usage_percent,
        "isolated": isolated,
        "overloaded": overloaded,
        "num_overloaded": num_overloaded
    }

def generate_traffic_matrix(num_nodes, min_traffic=5, max_traffic=50, pattern_type="random"):
    """Generate a random traffic matrix."""
    tm = np.zeros((num_nodes, num_nodes))
    
    # Simple traffic generation patterns:
    if pattern_type == "random":
        # Completely random traffic
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    tm[i, j] = random.uniform(min_traffic, max_traffic)
    
    elif pattern_type == "clustered":
        # Create traffic clusters - nodes 0,1,2 and nodes 2,3,4 form clusters
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    # Determine if in same cluster
                    if (i < 3 and j < 3) or (i >= 2 and j >= 2):
                        # Higher traffic within cluster
                        tm[i, j] = random.uniform(max_traffic * 0.6, max_traffic)
                    else:
                        # Lower traffic between clusters
                        tm[i, j] = random.uniform(min_traffic, max_traffic * 0.4)
    
    elif pattern_type == "hub_spoke":
        # Hub and spoke pattern - node 0 is the hub
        hub = 0
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    if i == hub or j == hub:
                        # Higher traffic to/from hub
                        tm[i, j] = random.uniform(max_traffic * 0.7, max_traffic)
                    else:
                        # Lower traffic between spokes
                        tm[i, j] = random.uniform(min_traffic, max_traffic * 0.3)
    
    else:  # Default pattern if not recognized
        # Uniform traffic distribution
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    tm[i, j] = random.uniform(min_traffic, max_traffic)
    
    # Round to 2 decimal places
    return np.round(tm, 2).tolist()

def scale_traffic_matrix(tm, target_load=0.8, env=None):
    """Scale a traffic matrix to achieve a target maximum link load."""
    if env is None:
        return tm
    
    # Make a copy of tm to avoid modifying the original
    tm_array = np.array(tm)
    
    # Set the traffic matrix
    env.traffic = tm_array
    
    # Make sure all links are open
    env.link_open = np.ones(env.num_edges, dtype=int)
    
    # Update link usage
    env._update_link_usage()
    
    # Get the current maximum load as a fraction of capacity
    if np.any(env.usage):
        max_usage = np.max(env.usage)
        max_usage_idx = np.argmax(env.usage)
        edge = env.edge_list[max_usage_idx]
        capacity = env.graph[edge[0]][edge[1]]['capacity']
        
        # Calculate scaling factor to achieve target load
        if capacity > 0 and max_usage > 0:
            current_load_fraction = max_usage / capacity
            scale_factor = target_load / current_load_fraction
            
            # Apply scaling to all traffic demands
            scaled_tm = tm_array * scale_factor
            
            # Round to 2 decimal places
            return np.round(scaled_tm, 2).tolist()
    
    # If we can't scale properly, return the original
    return tm

def generate_and_test_matrices(config_path, num_new_matrices=10, safety_margin=0.85):
    """Generate and test new traffic matrices, then add them to the config."""
    # Load configuration
    config = load_config(config_path)
    
    # Extract parameters
    num_nodes = config["num_nodes"]
    adj_matrix = config["adj_matrix"]
    edge_list = config["edge_list"]
    tm_list = config["tm_list"]
    node_props = config.get("node_props", {})
    link_capacity = config["link_capacity"]
    max_edges = config.get("max_edges", len(edge_list))
    
    # Create environment for testing
    env = NetworkEnv(
        adj_matrix=adj_matrix,
        edge_list=edge_list,
        tm_list=tm_list,
        node_props=node_props,
        num_nodes=num_nodes,
        link_capacity=link_capacity,
        max_edges=max_edges,
        seed=42
    )
    
    # Types of traffic patterns to generate
    pattern_types = ["random", "clustered", "hub_spoke"]
    
    # Generate new matrices
    new_matrices = []
    print(f"Generating {num_new_matrices} new traffic matrices...")
    
    pbar = tqdm(total=num_new_matrices)
    count = 0
    
    for i in range(num_new_matrices * 3):  # Try up to 3x as many attempts
        if count >= num_new_matrices:
            break
            
        try:
            # Select pattern type (cycle through them)
            pattern_type = pattern_types[i % len(pattern_types)]
            
            # Generate raw traffic matrix
            raw_tm = generate_traffic_matrix(
                num_nodes=num_nodes, 
                min_traffic=5, 
                max_traffic=50,
                pattern_type=pattern_type
            )
            
            # Scale to desired load level
            scaled_tm = scale_traffic_matrix(
                tm=raw_tm,
                target_load=safety_margin,  # Target load as a safety margin
                env=env
            )
            
            # Test feasibility
            result = check_traffic_matrix_feasibility(scaled_tm, env)
            
            # Only add if feasible
            if result["feasible"]:
                new_matrices.append(scaled_tm)
                count += 1
                pbar.update(1)
                print(f"Added matrix #{count}: {pattern_type.upper()} pattern, max usage: {result['max_usage_percent']:.1f}%")
        except Exception as e:
            print(f"Error generating matrix: {e}")
            continue
    
    pbar.close()
    
    # Add new matrices to config
    config["tm_list"].extend(new_matrices)
    print(f"Added {len(new_matrices)} new traffic matrices to config")
    print(f"Total traffic matrices in config: {len(config['tm_list'])}")
    
    # Save updated config
    config_basename = os.path.basename(config_path)
    new_config_path = f"configs/extended_{config_basename}"
    save_config(config, new_config_path)
    print(f"Saved extended configuration to {new_config_path}")
    
    return new_config_path, len(new_matrices)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate additional traffic matrices")
    parser.add_argument('--config', type=str, default="configs/config_5node.json", help='Path to config JSON file')
    parser.add_argument('--num-matrices', type=int, default=10, help='Number of new matrices to generate')
    parser.add_argument('--margin', type=float, default=0.85, help='Safety margin for link capacity (0.0-1.0)')
    
    args = parser.parse_args()
    
    # Generate and test matrices
    new_config_path, count = generate_and_test_matrices(
        config_path=args.config,
        num_new_matrices=args.num_matrices,
        safety_margin=args.margin
    )
    
    print(f"Successfully generated {count} new traffic matrices")
    print(f"Extended configuration saved to: {new_config_path}")
