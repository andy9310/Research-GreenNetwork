"""
Script to scale down traffic matrix values to make them feasible
within the network capacity constraints.
"""

import json
import argparse
import numpy as np
import networkx as nx
import copy
from tqdm import tqdm

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def save_config(config, output_path):
    """Save configuration to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

def is_traffic_matrix_valid(traffic_matrix, edge_list, link_capacity, num_nodes):
    """Check if a traffic matrix is valid within capacity constraints."""
    # Create a graph for routing
    graph = nx.Graph()
    for i in range(num_nodes):
        graph.add_node(i)
    for i, (u, v) in enumerate(edge_list):
        graph.add_edge(u, v, capacity=link_capacity, weight=1, id=i)
    
    # Check graph connectivity
    if not nx.is_connected(graph):
        return False, "disconnected", []
    
    # Calculate traffic routing and usage
    edge_usage = np.zeros(len(edge_list))
    routing_successful = True
    
    try:
        for src in range(num_nodes):
            for dst in range(num_nodes):
                if src != dst and traffic_matrix[src][dst] > 0:
                    try:
                        path = nx.shortest_path(graph, source=src, target=dst, weight='weight')
                        path_edges = list(zip(path[:-1], path[1:]))
                        
                        for u, v in path_edges:
                            # Find the edge id
                            for edge_idx, (edge_u, edge_v) in enumerate(edge_list):
                                if (u == edge_u and v == edge_v) or (u == edge_v and v == edge_u):
                                    edge_usage[edge_idx] += traffic_matrix[src][dst]
                                    break
                    except nx.NetworkXNoPath:
                        routing_successful = False
                        return False, "routing_failed", []
    except Exception as e:
        return False, f"error: {str(e)}", []
    
    # Check for overloads
    overloaded_links = []
    for edge_idx, usage in enumerate(edge_usage):
        if usage > link_capacity:
            u, v = edge_list[edge_idx]
            overloaded_links.append({
                "edge_idx": edge_idx,
                "endpoints": (u, v),
                "usage": usage,
                "capacity": link_capacity,
                "ratio": usage / link_capacity
            })
    
    if overloaded_links:
        return False, "overloaded", overloaded_links
    
    return True, "valid", []

def scale_traffic_matrices(config, scale_factor=0.5, target_capacity=None):
    """Scale down all traffic matrices by the specified factor."""
    new_config = copy.deepcopy(config)
    
    # If target_capacity is specified, update it in the config
    if target_capacity:
        new_config["link_capacity"] = target_capacity
    
    # Scale down all traffic matrices
    for i in range(len(new_config["tm_list"])):
        for j in range(len(new_config["tm_list"][i])):
            for k in range(len(new_config["tm_list"][i][j])):
                new_config["tm_list"][i][j][k] = int(new_config["tm_list"][i][j][k] * scale_factor)
    
    return new_config

def find_optimal_scaling(config_path, output_path, target_capacity=300, max_iterations=10, debug=False):
    """Find the optimal scaling factor to make all traffic matrices feasible."""
    # Load configuration
    config = load_config(config_path)
    
    # Extract parameters
    num_nodes = config["num_nodes"]
    edge_list = config["edge_list"]
    tm_list = config["tm_list"]
    
    # Set link capacity
    original_capacity = config["link_capacity"]
    link_capacity = target_capacity
    
    print(f"Original Configuration:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {len(edge_list)}")
    print(f"  Traffic Matrices: {len(tm_list)}")
    print(f"  Original Link Capacity: {original_capacity}")
    print(f"  Target Link Capacity: {link_capacity}")
    
    # Start with a scaling factor of 1.0 (no scaling)
    upper_bound = 1.0
    lower_bound = 0.01
    current_scale = upper_bound
    
    # Binary search to find optimal scaling factor
    best_scale = None
    best_config = None
    
    for iteration in range(max_iterations):
        print(f"\nIteration {iteration+1}, Testing scale factor: {current_scale:.4f}")
        
        # Scale traffic matrices with current factor
        scaled_config = scale_traffic_matrices(config, current_scale, target_capacity)
        
        # Check if all matrices are valid with this scaling
        all_valid = True
        max_ratio = 0
        
        for tm_idx, tm in enumerate(tqdm(scaled_config["tm_list"], desc="Checking Matrices")):
            is_valid, reason, overloaded = is_traffic_matrix_valid(
                tm, edge_list, link_capacity, num_nodes
            )
            
            if not is_valid:
                all_valid = False
                if reason == "overloaded" and overloaded:
                    max_ratio = max(max_ratio, max(o["ratio"] for o in overloaded))
                    if debug:
                        print(f"  TM {tm_idx}: ❌ - Max overload ratio: {max_ratio:.2f}")
                else:
                    if debug:
                        print(f"  TM {tm_idx}: ❌ - {reason}")
            elif debug:
                print(f"  TM {tm_idx}: ✅")
        
        if all_valid:
            # All matrices valid with this scale - try a larger scale
            best_scale = current_scale
            best_config = scaled_config
            lower_bound = current_scale
            current_scale = (lower_bound + upper_bound) / 2
            print(f"  Success - All matrices valid at scale {best_scale:.4f}")
            print(f"  Trying larger scale: {current_scale:.4f}")
        else:
            # Some matrices invalid - try a smaller scale
            upper_bound = current_scale
            current_scale = (lower_bound + upper_bound) / 2
            if max_ratio > 0:
                print(f"  Failed - Max overload ratio: {max_ratio:.2f}")
            else:
                print(f"  Failed - Some matrices invalid")
            print(f"  Trying smaller scale: {current_scale:.4f}")
        
        # Check if we've converged (bounds are very close)
        if upper_bound - lower_bound < 0.01:
            break
    
    if best_scale:
        print(f"\nFound optimal scaling factor: {best_scale:.4f}")
        
        # Verify all matrices again with the best scale
        scaled_config = scale_traffic_matrices(config, best_scale, target_capacity)
        valid_count = 0
        
        print("\nVerifying final scaled matrices:")
        for tm_idx, tm in enumerate(scaled_config["tm_list"]):
            is_valid, reason, _ = is_traffic_matrix_valid(
                tm, edge_list, link_capacity, num_nodes
            )
            if is_valid:
                valid_count += 1
                print(f"  TM {tm_idx}: ✅ Valid")
            else:
                print(f"  TM {tm_idx}: ❌ Invalid - {reason}")
        
        print(f"\nValidation results: {valid_count}/{len(scaled_config['tm_list'])} matrices valid")
        
        # Save modified configuration
        save_config(scaled_config, output_path)
        print(f"Saved scaled configuration to {output_path}")
        
        return True, best_scale
    else:
        print("\nFailed to find a valid scaling factor.")
        return False, None

def generate_new_traffic_matrices(config_path, output_path, num_matrices=24, density=0.4, 
                                 max_demand=100, target_capacity=300):
    """Generate new traffic matrices that respect capacity constraints."""
    # Load configuration
    config = load_config(config_path)
    
    # Extract parameters
    num_nodes = config["num_nodes"]
    edge_list = config["edge_list"]
    
    # Create a copy of the config
    new_config = copy.deepcopy(config)
    new_config["link_capacity"] = target_capacity
    
    # Create empty list for new matrices
    new_config["tm_list"] = []
    
    print(f"Generating {num_matrices} new traffic matrices...")
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {len(edge_list)}")
    print(f"  Target Capacity: {target_capacity}")
    print(f"  Max Demand: {max_demand}")
    print(f"  Density: {density}")
    
    # Create a graph for testing routing
    graph = nx.Graph()
    for i in range(num_nodes):
        graph.add_node(i)
    for i, (u, v) in enumerate(edge_list):
        graph.add_edge(u, v, capacity=target_capacity, weight=1, id=i)
    
    # Generate matrices
    valid_matrices = 0
    attempts = 0
    max_attempts = 100
    
    # Progressive scaling for matrix generation
    scale_start = 0.2
    scale_increment = 0.05
    current_scale = scale_start
    
    with tqdm(total=num_matrices, desc="Generating Matrices") as pbar:
        while valid_matrices < num_matrices and attempts < max_attempts:
            # Generate a random traffic matrix
            tm = np.zeros((num_nodes, num_nodes))
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j and np.random.random() < density:
                        # Generate a random demand but scale it
                        demand = np.random.randint(1, max_demand + 1)
                        tm[i, j] = int(demand * current_scale)
            
            # Check if the matrix is valid
            is_valid, reason, _ = is_traffic_matrix_valid(tm.tolist(), edge_list, target_capacity, num_nodes)
            
            if is_valid:
                new_config["tm_list"].append(tm.tolist())
                valid_matrices += 1
                pbar.update(1)
                
                # Potentially increase scale for more aggressive matrices
                if valid_matrices % 5 == 0 and current_scale < 1.0:
                    current_scale += scale_increment
                    print(f"  Increasing scale to {current_scale:.2f}")
            else:
                # If failed due to overload, decrease scale
                if reason == "overloaded" and current_scale > 0.1:
                    current_scale -= 0.02
                    print(f"  Decreasing scale to {current_scale:.2f}")
            
            attempts += 1
    
    if valid_matrices < num_matrices:
        print(f"\nWarning: Could only generate {valid_matrices} valid matrices out of {num_matrices} requested.")
    
    # Save modified configuration
    save_config(new_config, output_path)
    print(f"Saved new configuration with {valid_matrices} matrices to {output_path}")
    
    return valid_matrices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fix traffic matrices in a configuration file')
    parser.add_argument('--config', type=str, default='../../train_configs/config_17node_25edges.json', 
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='../../train_configs/config_17node_25edges_fixed.json', 
                       help='Path to output configuration file')
    parser.add_argument('--method', type=str, choices=['scale', 'generate'], default='scale',
                       help='Method to fix matrices: scale down existing or generate new ones')
    parser.add_argument('--capacity', type=int, default=300, 
                       help='Target link capacity')
    parser.add_argument('--debug', action='store_true',
                       help='Show detailed debugging information')
    args = parser.parse_args()
    
    if args.method == 'scale':
        success, scale = find_optimal_scaling(
            args.config, args.output, args.capacity, debug=args.debug
        )
        if success:
            print(f"Successfully scaled traffic matrices by factor {scale:.4f}")
        else:
            print("Failed to scale traffic matrices to be feasible")
    else:
        count = generate_new_traffic_matrices(
            args.config, args.output, target_capacity=args.capacity
        )
        print(f"Generated {count} new valid traffic matrices")
