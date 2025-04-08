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

# Target number of edges
target_edges = 25

# Create a new empty adjacency matrix (all zeros)
new_adj_matrix = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]

# First, ensure the graph is connected by creating a minimum spanning tree
# This requires at least (num_nodes - 1) edges to ensure all nodes are connected
edges_for_mst = num_nodes - 1  # 16 edges for a 17-node MST

# Create a list of all possible edges
all_possible_edges = []
for i in range(num_nodes):
    for j in range(i+1, num_nodes):
        all_possible_edges.append((i, j))

# Randomly select edges for MST (ensuring connectivity)
random.seed(42)  # For reproducibility
random.shuffle(all_possible_edges)

# Function to check if adding an edge would create a cycle
def find_set(parent, i):
    if parent[i] != i:
        parent[i] = find_set(parent, parent[i])
    return parent[i]

def union(parent, rank, x, y):
    root_x = find_set(parent, x)
    root_y = find_set(parent, y)
    if root_x != root_y:
        if rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        elif rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        else:
            parent[root_y] = root_x
            rank[root_x] += 1

# Create MST using Kruskal's algorithm
mst_edges = []
parent = [i for i in range(num_nodes)]
rank = [0] * num_nodes
edge_count = 0

for u, v in all_possible_edges:
    if edge_count >= edges_for_mst:
        break
    
    root_u = find_set(parent, u)
    root_v = find_set(parent, v)
    
    if root_u != root_v:
        mst_edges.append((u, v))
        union(parent, rank, root_u, root_v)
        edge_count += 1

# Add MST edges to the adjacency matrix
for u, v in mst_edges:
    new_adj_matrix[u][v] = 1
    new_adj_matrix[v][u] = 1  # Undirected graph

# Calculate remaining edges to add
remaining_edges = target_edges - len(mst_edges)
print(f"MST edges: {len(mst_edges)}, Remaining to add: {remaining_edges}")

# Remove MST edges from possible edges to avoid duplicates
possible_edges = []
for i in range(num_nodes):
    for j in range(i+1, num_nodes):
        if new_adj_matrix[i][j] == 0:
            possible_edges.append((i, j))

# Randomly choose the remaining edges
random.shuffle(possible_edges)
additional_edges = possible_edges[:remaining_edges]

# Add additional edges to adjacency matrix
for u, v in additional_edges:
    new_adj_matrix[u][v] = 1
    new_adj_matrix[v][u] = 1

# Create a new edge list from the adjacency matrix
edge_list = []
for i in range(num_nodes):
    for j in range(num_nodes):
        if new_adj_matrix[i][j] == 1 and i < j:  # Only count each edge once
            edge_list.append([i, j])

# Count final edges
final_edge_count = 0
for i in range(num_nodes):
    for j in range(i+1, num_nodes):
        if new_adj_matrix[i][j] == 1:
            final_edge_count += 1

print(f"Final edge count: {final_edge_count}")

# Update the config
config["adj_matrix"] = new_adj_matrix
config["edge_list"] = edge_list
config["max_edges"] = 30  # Set max edges slightly higher than our target

# Save to a new file
output_path = "config_17node_25edges.json"
with open(output_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"New configuration with {len(edge_list)} edges saved to {output_path}")
print(f"Max edges set to {config['max_edges']}")

# Print network properties
print("\nNetwork properties:")
print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {final_edge_count}")
print(f"Connectivity ratio: {final_edge_count / (num_nodes * (num_nodes - 1) / 2) * 100:.2f}%")
