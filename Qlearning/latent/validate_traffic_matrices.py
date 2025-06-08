"""
Validation script to check if all traffic matrices in a configuration file
are valid when all links are open.
"""

import json
import argparse
import numpy as np
import networkx as nx
from latent_env import NetworkEnv

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def validate_traffic_matrices(config_path):
    """
    Check if all traffic matrices in the configuration are valid
    when all links are open.
    """
    # Load config
    config = load_config(config_path)
    
    # Extract parameters
    num_nodes = config["num_nodes"]
    adj_matrix = config["adj_matrix"]
    edge_list = config["edge_list"]
    tm_list = config["tm_list"]
    node_props = config.get("node_props", {})
    link_capacity = config["link_capacity"]
    max_edges = config.get("max_edges", len(edge_list))
    
    print(f"Configuration: {config_path}")
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {len(edge_list)}")
    print(f"  Traffic Matrices: {len(tm_list)}")
    print(f"  Link Capacity: {link_capacity}")
    
    # Create a NetworkX graph to test routing directly
    graph = nx.Graph()
    for i in range(num_nodes):
        graph.add_node(i, **node_props.get(i, {}))
    for i, (u, v) in enumerate(edge_list):
        graph.add_edge(u, v, capacity=link_capacity, weight=1, id=i)
    
    results = []
    
    # Test each traffic matrix
    print("\nValidating Traffic Matrices...")
    print("=" * 80)
    
    for tm_idx, tm in enumerate(tm_list):
        # Initialize environment with this traffic matrix
        traffic = np.array(tm)
        
        # Check graph connectivity
        is_connected = nx.is_connected(graph)
        
        # Manually calculate routing
        edge_usage = np.zeros(len(edge_list))
        routing_successful = True
        
        # Try to route traffic between all node pairs with demand
        try:
            for src in range(num_nodes):
                for dst in range(num_nodes):
                    if src != dst and traffic[src, dst] > 0:
                        try:
                            path = nx.shortest_path(graph, source=src, target=dst, weight='weight')
                            path_edges = list(zip(path[:-1], path[1:]))
                            
                            for u, v in path_edges:
                                # Find the edge id
                                for edge_idx, (edge_u, edge_v) in enumerate(edge_list):
                                    if (u == edge_u and v == edge_v) or (u == edge_v and v == edge_u):
                                        edge_usage[edge_idx] += traffic[src, dst]
                                        break
                        except nx.NetworkXNoPath:
                            routing_successful = False
                            break
                
                if not routing_successful:
                    break
        except Exception as e:
            routing_successful = False
            print(f"Error during routing for TM {tm_idx}: {e}")
        
        # Check for overloads
        overloaded = False
        num_overloaded = 0
        overloaded_links = []
        
        for edge_idx, usage in enumerate(edge_usage):
            if usage > link_capacity:
                overloaded = True
                num_overloaded += 1
                u, v = edge_list[edge_idx]
                overloaded_links.append((edge_idx, u, v, usage, link_capacity, usage/link_capacity))
        
        # Record result
        status = "✅ Valid" if is_connected and routing_successful and not overloaded else "❌ Invalid"
        
        issues = []
        if not is_connected:
            issues.append("disconnected")
        if not routing_successful:
            issues.append("routing failed")
        if overloaded:
            issues.append(f"{num_overloaded} overloaded links")
        
        results.append({
            "tm_idx": tm_idx,
            "valid": is_connected and routing_successful and not overloaded,
            "connected": is_connected,
            "routing_successful": routing_successful,
            "overloaded": overloaded,
            "num_overloaded": num_overloaded,
            "overloaded_links": overloaded_links if overloaded else []
        })
        
        print(f"TM {tm_idx}: {status}" + (f" - Issues: {', '.join(issues)}" if issues else ""))
        
        # Print detailed information for invalid matrices
        if overloaded:
            print("  Overloaded Links:")
            for edge_idx, u, v, usage, capacity, ratio in overloaded_links:
                print(f"    Edge {edge_idx} ({u}-{v}): Usage={usage:.2f}, Capacity={capacity}, Ratio={ratio:.2f}")
    
    # Summarize results
    valid_count = sum(1 for r in results if r["valid"])
    print("\nSummary:")
    print(f"  Valid Traffic Matrices: {valid_count}/{len(tm_list)} ({valid_count/len(tm_list)*100:.2f}%)")
    print(f"  Invalid Traffic Matrices: {len(tm_list) - valid_count}/{len(tm_list)} ({(len(tm_list) - valid_count)/len(tm_list)*100:.2f}%)")
    
    # Return invalid matrices
    return [r for r in results if not r["valid"]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate traffic matrices in a configuration file')
    parser.add_argument('--config', type=str, default='../../train_configs/config_17node_25edges.json', 
                       help='Path to configuration file')
    args = parser.parse_args()
    
    invalid_matrices = validate_traffic_matrices(args.config)
    
    if not invalid_matrices:
        print("\nAll traffic matrices are valid!")
    else:
        print(f"\nFound {len(invalid_matrices)} invalid traffic matrices that need fixing.")
