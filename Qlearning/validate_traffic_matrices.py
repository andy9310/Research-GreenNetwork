"""
Traffic Matrix Validation Script

This script validates traffic matrices in a configuration file to ensure they can be
routed without violations when all links are open. It identifies problematic traffic
matrices and optionally removes them or scales them down.
"""

import os
import sys
import json
import numpy as np
import networkx as nx
import argparse
from tqdm import tqdm

from env import NetworkEnv

def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def save_config(config, output_path):
    """Save configuration to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {output_path}")

def validate_traffic_matrices(config_path, epsilon=0.02, action='report', scale_factor=0.95, output_path=None):
    """
    Validate traffic matrices in a configuration file.
    
    Args:
        config_path: Path to configuration file
        epsilon: Tolerance for capacity violations (default: 0.02 or 2%)
        action: What to do with invalid traffic matrices:
                'report' (default) - Just report the problems
                'remove' - Remove invalid matrices from the configuration
                'scale' - Scale down invalid matrices by scale_factor
        scale_factor: Factor to scale down invalid matrices (default: 0.95)
        output_path: Path to save the modified configuration file (if action is 'remove' or 'scale')
                     If None, will use the original filename with '_validated' appended
    
    Returns:
        dict: Statistics about the validation process
    """
    # Load configuration
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
        seed=42
    )
    
    # Track validation results
    validation_results = []
    valid_tm_indices = []
    invalid_tm_indices = []
    
    # Process each traffic matrix
    print(f"Validating {len(tm_list)} traffic matrices...")
    for tm_idx in tqdm(range(len(tm_list))):
        env.current_tm_idx = tm_idx
        
        # Reset environment to ensure all links are open
        env.reset()
        
        # Make sure all links are open
        env.link_open = np.ones(env.num_edges, dtype=int)
        
        # Update link usage
        routing_successful, G_open = env._update_link_usage()
        
        # Check for violations with the specified epsilon tolerance
        isolated, overloaded, num_overloaded = env._check_violations(routing_successful, G_open, epsilon=epsilon)
        
        # Record result
        is_valid = not (isolated or overloaded)
        validation_results.append({
            'tm_idx': tm_idx,
            'valid': is_valid,
            'isolated': isolated,
            'overloaded': overloaded,
            'num_overloaded': num_overloaded,
            'violation_details': []
        })
        
        # If there are violations, get details
        if not is_valid:
            invalid_tm_indices.append(tm_idx)
            
            # Collect details about overloaded links
            if overloaded:
                for i, usage in enumerate(env.usage):
                    if env.link_open[i] == 1:  # Only check open links
                        u, v = env.edge_list[i]
                        capacity = env.graph[u][v]['capacity']
                        
                        # Calculate percent of capacity used
                        percent_used = usage / capacity * 100.0
                        
                        # Check if this link is overloaded
                        if usage > capacity * (1.0 + epsilon):
                            validation_results[-1]['violation_details'].append({
                                'link_idx': i,
                                'link': (u, v),
                                'usage': float(usage),
                                'capacity': float(capacity),
                                'utilization': float(usage / capacity),
                                'percent_used': float(percent_used)
                            })
        else:
            valid_tm_indices.append(tm_idx)
    
    # Process validation results based on the selected action
    if action in ['remove', 'scale'] and (len(invalid_tm_indices) > 0):
        # Create output path if not provided
        if output_path is None:
            filename, ext = os.path.splitext(config_path)
            output_path = f"{filename}_validated{ext}"
        
        if action == 'remove':
            # Remove invalid traffic matrices
            config['tm_list'] = [tm_list[i] for i in valid_tm_indices]
            print(f"Removed {len(invalid_tm_indices)} invalid traffic matrices out of {len(tm_list)}")
            
        elif action == 'scale':
            # Scale down invalid traffic matrices
            for idx in invalid_tm_indices:
                # Scale entire matrix down by the scale factor
                config['tm_list'][idx] = [[cell * scale_factor for cell in row] for row in config['tm_list'][idx]]
            print(f"Scaled down {len(invalid_tm_indices)} invalid traffic matrices by factor {scale_factor}")
        
        # Save the modified configuration
        save_config(config, output_path)
    
    # Print validation summary
    print("\n=== Traffic Matrix Validation Summary ===")
    print(f"Total Traffic Matrices: {len(tm_list)}")
    print(f"Valid Matrices: {len(valid_tm_indices)} ({len(valid_tm_indices)/len(tm_list)*100:.1f}%)")
    print(f"Invalid Matrices: {len(invalid_tm_indices)} ({len(invalid_tm_indices)/len(tm_list)*100:.1f}%)")
    
    if len(invalid_tm_indices) > 0:
        print("\nInvalid Traffic Matrices:")
        for idx in invalid_tm_indices:
            result = validation_results[idx]
            violation_type = "isolated nodes" if result['isolated'] else "overloaded links"
            print(f"  TM {idx}: {violation_type}, {result['num_overloaded']} overloaded links")
            
            # Print detailed violation information
            if result['violation_details']:
                for i, violation in enumerate(result['violation_details']):
                    if i >= 3 and len(result['violation_details']) > 4:  # Show at most 3 violations if there are many
                        print(f"    ... and {len(result['violation_details']) - 3} more overloaded links")
                        break
                    link = violation['link']
                    util = violation['utilization']
                    print(f"    Link {link[0]}->{link[1]}: Utilization {util:.2f} ({util*100:.1f}% of capacity)")
    
    return {
        'config_path': config_path,
        'tm_count': len(tm_list),
        'valid_count': len(valid_tm_indices),
        'invalid_count': len(invalid_tm_indices),
        'invalid_indices': invalid_tm_indices,
        'validation_results': validation_results,
        'action_taken': action if len(invalid_tm_indices) > 0 else 'none',
        'output_path': output_path if action in ['remove', 'scale'] and len(invalid_tm_indices) > 0 else None
    }

def main():
    parser = argparse.ArgumentParser(description='Validate traffic matrices in a configuration file')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    parser.add_argument('--epsilon', type=float, default=0.02, help='Tolerance for capacity violations (default: 0.02 or 2%%)')
    parser.add_argument('--action', type=str, choices=['report', 'remove', 'scale'], default='report',
                        help='Action to take for invalid matrices: report, remove, or scale (default: report)')
    parser.add_argument('--scale-factor', type=float, default=0.95,
                        help='Factor to scale down invalid matrices (default: 0.95, only used with --action=scale)')
    parser.add_argument('--output', type=str, default=None, 
                        help='Path to save modified configuration (default: adds "_validated" to original filename)')
    
    args = parser.parse_args()
    
    # Run validation
    validate_traffic_matrices(
        config_path=args.config,
        epsilon=args.epsilon,
        action=args.action,
        scale_factor=args.scale_factor,
        output_path=args.output
    )

if __name__ == "__main__":
    main()
