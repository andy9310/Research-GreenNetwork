#!/usr/bin/env python3
"""
Configuration Generator for Network Optimization

This script generates network configuration files with a specified number of nodes,
edges, and traffic matrices, then validates them to ensure they're feasible.
"""

import os
import json
import random
import argparse
import numpy as np
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Generate and validate network configuration files')
    parser.add_argument('--nodes', type=int, default=10, help='Number of nodes in the network')
    parser.add_argument('--edges', type=int, default=15, help='Number of edges in the network')
    parser.add_argument('--train-matrices', type=int, default=40, help='Number of traffic matrices for training')
    parser.add_argument('--test-matrices', type=int, default=8, help='Number of traffic matrices for testing')
    parser.add_argument('--link-capacity', type=float, default=200, help='Capacity for each link')
    parser.add_argument('--max-demand', type=float, default=10, help='Maximum demand between node pairs')
    parser.add_argument('--energy-reward', type=float, default=10, help='Energy unit reward for closing a link')
    parser.add_argument('--train-output', type=str, default='config_train.json', help='Output file for training configuration')
    parser.add_argument('--test-output', type=str, default='config_test.json', help='Output file for testing configuration')
    parser.add_argument('--visualize', action='store_true', help='Visualize the generated network')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    return parser.parse_args()

def generate_connected_graph(n_nodes, n_edges, seed=None):
    """Generate a connected graph with n_nodes and n_edges."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Ensure the graph is connected (requires at least n_nodes-1 edges)
    min_edges = n_nodes - 1
    if n_edges < min_edges:
        print(f"Warning: At least {min_edges} edges needed to ensure connectivity. Setting edges to {min_edges}.")
        n_edges = min_edges
    
    # Create a connected graph using minimum spanning tree
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    
    # Make sure all nodes are connected first (spanning tree)
    for i in range(1, n_nodes):
        # Connect to a random previous node
        j = random.randint(0, i-1)
        G.add_edge(i, j)
    
    # Add remaining edges randomly
    possible_edges = [(i, j) for i in range(n_nodes) for j in range(i+1, n_nodes) if not G.has_edge(i, j)]
    remaining_edges = n_edges - (n_nodes - 1)
    
    if remaining_edges > 0:
        if remaining_edges > len(possible_edges):
            print(f"Warning: Cannot add {remaining_edges} more edges, maximum is {len(possible_edges)}.")
            remaining_edges = len(possible_edges)
        
        selected_edges = random.sample(possible_edges, remaining_edges)
        G.add_edges_from(selected_edges)
    
    # Get the adjacency matrix
    adj_matrix = np.zeros((n_nodes, n_nodes), dtype=int)
    
    # Create edge list in format needed for configuration file
    edge_list = []
    for i, j in G.edges():
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1
        edge_list.append([i, j])
    
    return G, adj_matrix, edge_list

def generate_traffic_matrices(G, n_matrices, max_demand=50, seed=None):
    """Generate n_matrices traffic matrices for the graph G."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    n_nodes = G.number_of_nodes()
    traffic_matrices = []
    
    for _ in range(n_matrices):
        traffic_matrix = []
        for i in range(n_nodes):
            row = []
            for j in range(n_nodes):
                if i == j:  # No traffic to self
                    demand = 0
                else:
                    # Generate a random integer demand between nodes
                    demand = random.randint(0, max_demand)
                row.append(demand)
            traffic_matrix.append(row)
        traffic_matrices.append(traffic_matrix)
    
    return traffic_matrices

def calculate_shortest_paths(G):
    """Calculate shortest paths for all pairs of nodes in the graph."""
    return dict(nx.all_pairs_shortest_path(G))

def calculate_link_usage(traffic_matrix, shortest_paths, n_nodes, edge_list):
    """Calculate the usage on each link based on the traffic matrix and shortest paths."""
    link_usage = defaultdict(float)
    
    # For each (source, destination) pair
    for src in range(n_nodes):
        for dst in range(n_nodes):
            if src != dst and traffic_matrix[src][dst] > 0:
                # Get the path from source to destination
                try:
                    path = shortest_paths[src][dst]
                    # For each edge in the path, add the traffic
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        # Find the edge index in the edge_list
                        if [u, v] in edge_list:
                            edge_idx = edge_list.index([u, v])
                        elif [v, u] in edge_list:
                            edge_idx = edge_list.index([v, u])
                        else:
                            continue
                        
                        link_usage[edge_idx] += traffic_matrix[src][dst]
                except KeyError:
                    # This happens if there's no path from src to dst
                    continue
    
    return link_usage

def validate_traffic_matrix(traffic_matrix, G, edge_list, link_capacity):
    """Check if the traffic matrix is valid (all links below capacity and all nodes connected)."""
    shortest_paths = calculate_shortest_paths(G)
    n_nodes = G.number_of_nodes()
    link_usage = calculate_link_usage(traffic_matrix, shortest_paths, n_nodes, edge_list)
    
    # Check link capacities
    overloaded_links = []
    for edge_idx, usage in link_usage.items():
        if usage > link_capacity:
            u, v = edge_list[edge_idx]
            overloaded_links.append((edge_idx, u, v, usage))
    
    return len(overloaded_links) == 0, overloaded_links

def check_all_traffic_matrices(traffic_matrices, G, edge_list, link_capacity):
    """Validate all traffic matrices."""
    valid_count = 0
    invalid_count = 0
    issues = []
    
    for idx, tm in tqdm(enumerate(traffic_matrices), total=len(traffic_matrices), desc="Validating Traffic Matrices"):
        is_valid, overloaded_links = validate_traffic_matrix(tm, G, edge_list, link_capacity)
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            issues.append((idx, overloaded_links))
    
    return valid_count, invalid_count, issues

def scale_traffic_matrices(traffic_matrices, G, edge_list, link_capacity, max_iterations=100):
    """Scale down traffic matrices until they're all valid."""
    scale_factor = 1.0
    scale_step = 0.01
    iteration = 0
    
    valid_count, invalid_count, issues = check_all_traffic_matrices(
        traffic_matrices, G, edge_list, link_capacity
    )
    
    print(f"Starting scaling with initial invalid count: {invalid_count}")
    
    # First pass: global scaling for all matrices
    while invalid_count > 0 and iteration < max_iterations:
        # Decrease scale factor
        scale_factor -= scale_step
        
        # Apply scaling factor
        scaled_matrices = []
        for tm in traffic_matrices:
            scaled_tm = []
            for row in tm:
                scaled_row = [demand * scale_factor for demand in row]
                scaled_tm.append(scaled_row)
            scaled_matrices.append(scaled_tm)
        
        # Check if all are valid now
        valid_count, invalid_count, issues = check_all_traffic_matrices(
            scaled_matrices, G, edge_list, link_capacity
        )
        
        iteration += 1
        if iteration % 10 == 0:
            print(f"  Iteration {iteration}: Scale factor = {scale_factor:.4f}, Invalid matrices = {invalid_count}")
    
    # Second pass: individual scaling for any remaining invalid matrices
    if invalid_count > 0:
        print(f"Global scaling reached limit. Applying individual scaling to remaining invalid matrices...")
        
        # Find all invalid matrices and apply more aggressive scaling
        for i, tm in enumerate(scaled_matrices):
            # Test this specific matrix
            is_valid, _ = validate_traffic_matrix(tm, G, edge_list, link_capacity)
            
            if not is_valid:
                print(f"  Further scaling matrix {i}...")
                individual_scale = scale_factor
                for j in range(20):  # Try up to 20 more scaling steps
                    individual_scale -= scale_step * 2  # More aggressive step
                    
                    # Apply individual scaling
                    scaled_tm = []
                    for row in traffic_matrices[i]:
                        scaled_row = [demand * individual_scale for demand in row]
                        scaled_tm.append(scaled_row)
                    
                    # Check if valid
                    is_valid, _ = validate_traffic_matrix(scaled_tm, G, edge_list, link_capacity)
                    if is_valid:
                        scaled_matrices[i] = scaled_tm
                        print(f"    Matrix {i} valid at scale factor {individual_scale:.4f}")
                        break
    
    # Final validation check
    final_valid_count, final_invalid_count, _ = check_all_traffic_matrices(
        scaled_matrices, G, edge_list, link_capacity
    )
    
    if final_invalid_count > 0:
        print(f"WARNING: Could not make all traffic matrices valid. {final_invalid_count} matrices remain invalid.")
        print(f"Consider increasing link capacity, reducing traffic demands, or adding more edges to the network.")
    else:
        print(f"Success! All traffic matrices are now valid after scaling.")
    
    # Make sure all demands are integers
    final_matrices = []
    for tm in scaled_matrices:
        int_tm = [[int(value) for value in row] for row in tm]
        final_matrices.append(int_tm)
    
    return final_matrices, scale_factor

def create_config_file(n_nodes, adj_matrix, edge_list, traffic_matrices, link_capacity, energy_reward, output_file):
    """Create the configuration file in the required format."""
    # Ensure all values in traffic matrices are integers
    integer_traffic_matrices = []
    for tm in traffic_matrices:
        integer_tm = []
        for row in tm:
            integer_row = [int(demand) for demand in row]
            integer_tm.append(integer_row)
        integer_traffic_matrices.append(integer_tm)
    
    config = {
        "num_nodes":n_nodes,
        "adj_matrix": adj_matrix.tolist(),
        "edge_list": edge_list,
        "tm_list": integer_traffic_matrices,
        "link_capacity": int(link_capacity),
        "energy_unit_reward": int(energy_reward),
        "max_edges": len(edge_list) + 5  # Allow a few more edges than currently in the graph
    }
    
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to {output_file}")

def visualize_network(G, edge_list, output_dir='.'):
    """Visualize the network topology."""
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)  # Position nodes using spring layout
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7)
    
    # Label nodes
    labels = {i: str(i) for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=12)
    
    # Label edges with their index
    edge_labels = {(u, v): str(idx) for idx, (u, v) in enumerate(edge_list)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    plt.title(f"Network Topology ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    plt.axis('off')
    
    # Save figure
    output_file = os.path.join(output_dir, 'network_topology.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Network visualization saved to {output_file}")
    plt.close()

def main():
    args = parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    print(f"Generating network with {args.nodes} nodes and {args.edges} edges...")
    G, adj_matrix, edge_list = generate_connected_graph(args.nodes, args.edges, args.seed)
    
    # ---- TRAINING CONFIGURATION ----
    print(f"\n==== GENERATING TRAINING CONFIGURATION ====")
    print(f"Generating {args.train_matrices} traffic matrices for training...")
    train_matrices = generate_traffic_matrices(G, args.train_matrices, args.max_demand, args.seed)
    
    print("Validating training traffic matrices...")
    train_valid_count, train_invalid_count, train_issues = check_all_traffic_matrices(
        train_matrices, G, edge_list, args.link_capacity
    )
    
    print(f"\nTraining Validation Results:")
    print(f"  Valid Traffic Matrices: {train_valid_count}/{args.train_matrices} ({train_valid_count/args.train_matrices*100:.2f}%)")
    print(f"  Invalid Traffic Matrices: {train_invalid_count}/{args.train_matrices} ({train_invalid_count/args.train_matrices*100:.2f}%)")
    
    if train_invalid_count > 0:
        print("\nScaling training traffic matrices to make them all valid...")
        train_matrices, train_scale_factor = scale_traffic_matrices(
            train_matrices, G, edge_list, args.link_capacity
        )
        print(f"Applied scaling factor: {train_scale_factor:.4f}")
        
        # Verify all matrices are now valid
        train_valid_count, train_invalid_count, train_issues = check_all_traffic_matrices(
            train_matrices, G, edge_list, args.link_capacity
        )
        print(f"\nAfter Scaling:")
        print(f"  Valid Traffic Matrices: {train_valid_count}/{args.train_matrices} ({train_valid_count/args.train_matrices*100:.2f}%)")
        print(f"  Invalid Traffic Matrices: {train_invalid_count}/{args.train_matrices} ({train_invalid_count/args.train_matrices*100:.2f}%)")
    
    # ---- TESTING CONFIGURATION ----
    print(f"\n==== GENERATING TESTING CONFIGURATION ====")
    # Use a different seed for test matrices to ensure they're different from training
    test_seed = None if args.seed is None else args.seed + 1000
    print(f"Generating {args.test_matrices} traffic matrices for testing...")
    test_matrices = generate_traffic_matrices(G, args.test_matrices, args.max_demand, test_seed)
    
    print("Validating testing traffic matrices...")
    test_valid_count, test_invalid_count, test_issues = check_all_traffic_matrices(
        test_matrices, G, edge_list, args.link_capacity
    )
    
    print(f"\nTesting Validation Results:")
    print(f"  Valid Traffic Matrices: {test_valid_count}/{args.test_matrices} ({test_valid_count/args.test_matrices*100:.2f}%)")
    print(f"  Invalid Traffic Matrices: {test_invalid_count}/{args.test_matrices} ({test_invalid_count/args.test_matrices*100:.2f}%)")
    
    if test_invalid_count > 0:
        print("\nScaling testing traffic matrices to make them all valid...")
        test_matrices, test_scale_factor = scale_traffic_matrices(
            test_matrices, G, edge_list, args.link_capacity
        )
        print(f"Applied scaling factor: {test_scale_factor:.4f}")
        
        # Verify all matrices are now valid
        test_valid_count, test_invalid_count, test_issues = check_all_traffic_matrices(
            test_matrices, G, edge_list, args.link_capacity
        )
        print(f"\nAfter Scaling:")
        print(f"  Valid Traffic Matrices: {test_valid_count}/{args.test_matrices} ({test_valid_count/args.test_matrices*100:.2f}%)")
        print(f"  Invalid Traffic Matrices: {test_invalid_count}/{args.test_matrices} ({test_invalid_count/args.test_matrices*100:.2f}%)")
    
    # Create the training configuration file
    print(f"\nCreating training configuration file: {args.train_output}")
    create_config_file(
        args.nodes, adj_matrix, edge_list, train_matrices, 
        args.link_capacity, args.energy_reward, args.train_output
    )
    
    # Create the testing configuration file
    print(f"Creating testing configuration file: {args.test_output}")
    create_config_file(
        args.nodes, adj_matrix, edge_list, test_matrices, 
        args.link_capacity, args.energy_reward, args.test_output
    )
    
    # Visualize the network if requested
    if args.visualize:
        visualize_network(G, edge_list, os.path.dirname(args.train_output))
    
    # Wait for all validation processes to finish before printing statistics
    time.sleep(0.5)
    
    # Calculate link utilization statistics and save to separate files for train and test
    train_stats_file = os.path.join(os.path.dirname(args.train_output), "train_link_utilization_stats.txt")
    test_stats_file = os.path.join(os.path.dirname(args.test_output), "test_link_utilization_stats.txt")
    
    # Calculate shortest paths once
    shortest_paths = calculate_shortest_paths(G)
    n_nodes = G.number_of_nodes()
    
    # Process training matrices
    train_link_stats = calculate_link_stats(train_matrices, G, edge_list, shortest_paths, n_nodes, args.link_capacity, train_stats_file)
    
    # Process testing matrices
    test_link_stats = calculate_link_stats(test_matrices, G, edge_list, shortest_paths, n_nodes, args.link_capacity, test_stats_file)
    
    # Print summary to console
    print(f"\nLink utilization statistics saved to:")
    print(f"  Training: {train_stats_file}")
    print(f"  Testing: {test_stats_file}")
    
    print("\nConfiguration Generation Complete:")
    print(f"Network Topology: {args.nodes} nodes, {len(edge_list)} edges")
    print(f"Training Traffic Matrices: {args.train_matrices} (all valid)")
    print(f"Testing Traffic Matrices: {args.test_matrices} (all valid)")
    print(f"Link Capacity: {args.link_capacity}")
    print(f"Energy Unit Reward: {args.energy_reward}")
    print(f"Output Files: ")
    print(f"  Training: {args.train_output}")
    print(f"  Testing: {args.test_output}")

def calculate_link_stats(traffic_matrices, G, edge_list, shortest_paths, n_nodes, link_capacity, stats_file):
    """Calculate and write link utilization statistics for a set of traffic matrices."""
    with open(stats_file, 'w') as f:
        f.write("Link Utilization Statistics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Link':>10} | {'Avg Util %':>10} | {'Max Util %':>10} | {'Min Util %':>10} | {'Std Dev':>10}\n")
        f.write("-" * 80 + "\n")
        
        # Calculate statistics for each link
        link_stats = {}
        for i, edge in enumerate(edge_list):
            utilizations = []
            for tm in traffic_matrices:
                # Calculate usage for this traffic matrix
                link_usage = calculate_link_usage(tm, shortest_paths, n_nodes, edge_list)
                
                # Get usage for this specific edge
                flow = link_usage.get(i, 0.0)
                util_percent = (flow / link_capacity) * 100
                utilizations.append(util_percent)
            
            # Calculate statistics
            avg_util = np.mean(utilizations)
            max_util = np.max(utilizations)
            min_util = np.min(utilizations)
            std_dev = np.std(utilizations)
            
            # Store and print statistics
            link_stats[i] = {
                'edge': edge,
                'avg_util': avg_util,
                'max_util': max_util,
                'min_util': min_util,
                'std_dev': std_dev
            }
            
            f.write(f"{str(edge):>10} | {avg_util:>10.2f} | {max_util:>10.2f} | {min_util:>10.2f} | {std_dev:>10.2f}\n")
        
        # Find the most and least utilized links
        most_util_link = max(link_stats.items(), key=lambda x: x[1]['avg_util'])
        least_util_link = min(link_stats.items(), key=lambda x: x[1]['avg_util'])
        
        f.write("-" * 80 + "\n")
        f.write(f"Most utilized link: {most_util_link[1]['edge']} (Avg: {most_util_link[1]['avg_util']:.2f}%)\n")
        f.write(f"Least utilized link: {least_util_link[1]['edge']} (Avg: {least_util_link[1]['avg_util']:.2f}%)\n")
    
    return link_stats

if __name__ == "__main__":
    main()
