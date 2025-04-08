import json
import numpy as np
import random
import copy

# Load existing configuration
config_path = "config_17node.json"
with open(config_path, 'r') as f:
    config = json.load(f)

# Get current adjacency matrix
adj_matrix = config["adj_matrix"]
num_nodes = config["num_nodes"]

# Count current edges
current_edges = 0
for i in range(num_nodes):
    for j in range(i+1, num_nodes):  # Only count upper triangle to avoid counting twice
        if adj_matrix[i][j] == 1:
            current_edges += 1

print(f"Current number of edges: {current_edges}")

# Target number of edges
target_edges = 60

if current_edges <= target_edges:
    print("Current edges are already at or below target. No reduction needed.")
    exit()

# Create a list of all edges (as tuples)
all_edges = []
for i in range(num_nodes):
    for j in range(i+1, num_nodes):
        if adj_matrix[i][j] == 1:
            all_edges.append((i, j))

# Shuffle the edge list to randomize which edges we remove
random.seed(42)  # For reproducibility
random.shuffle(all_edges)

# Keep only the first 'target_edges' edges
edges_to_keep = all_edges[:target_edges]
edges_to_remove = all_edges[target_edges:]

print(f"Removing {len(edges_to_remove)} edges")

# Create a new adjacency matrix with reduced edges
new_adj_matrix = copy.deepcopy(adj_matrix)
for i, j in edges_to_remove:
    new_adj_matrix[i][j] = 0
    new_adj_matrix[j][i] = 0  # Also update the symmetric entry

# Create a new edge list
new_edge_list = []
for i in range(num_nodes):
    for j in range(num_nodes):
        if new_adj_matrix[i][j] == 1 and i < j:  # Only count each edge once
            new_edge_list.append([i, j])

# Update the config
config["adj_matrix"] = new_adj_matrix
config["edge_list"] = new_edge_list
config["max_edges"] = 60  # Update max_edges as well

# Save to a new file
output_path = "config_17node_60edges.json"
with open(output_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"New configuration with {len(new_edge_list)} edges saved to {output_path}")
print(f"Max edges set to {config['max_edges']}")
