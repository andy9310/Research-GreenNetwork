"""
Script to update the config_17node.json file with 24 traffic matrices:
- 8 low-traffic matrices
- 8 medium-traffic matrices
- 8 high-traffic matrices
"""

import json
import numpy as np
import copy

# Load the existing config file
config_file = "config_17node.json"
with open(config_file, 'r') as f:
    config = json.load(f)

# Get existing traffic matrices
tm_list = config["tm_list"]
num_nodes = config["num_nodes"]

print(f"Current number of traffic matrices: {len(tm_list)}")

# Classify existing matrices
# The last matrix is low-traffic (matrix 6)
low_tm = tm_list[-1]  # Matrix with values around 1-5
medium_tms = tm_list[:-1]  # First 6 matrices (medium traffic)

print(f"Existing: {len(medium_tms)} medium-traffic and 1 low-traffic matrices")

# Create 7 more low-traffic matrices (for a total of 8)
new_low_tms = []
for i in range(7):
    # Create a new matrix based on the existing low-traffic matrix
    # Add some randomness to make it different but similar
    new_tm = copy.deepcopy(low_tm)
    for row in range(num_nodes):
        for col in range(num_nodes):
            if row != col:  # No traffic from a node to itself
                # Vary by ±30%
                variation = 0.7 + np.random.random() * 0.6  # 0.7 to 1.3
                new_tm[row][col] = max(1, round(new_tm[row][col] * variation))
    new_low_tms.append(new_tm)

print(f"Created {len(new_low_tms)} additional low-traffic matrices")

# Create 2 more medium-traffic matrices (for a total of 8)
new_medium_tms = []
for i in range(2):
    # Create a new matrix based on existing medium-traffic matrices
    # Choose a random medium matrix as base
    base_tm = medium_tms[np.random.randint(0, len(medium_tms))]
    new_tm = copy.deepcopy(base_tm)
    for row in range(num_nodes):
        for col in range(num_nodes):
            if row != col:
                # Vary by ±20%
                variation = 0.8 + np.random.random() * 0.4  # 0.8 to 1.2
                new_tm[row][col] = max(1, round(new_tm[row][col] * variation))
    new_medium_tms.append(new_tm)

print(f"Created {len(new_medium_tms)} additional medium-traffic matrices")

# Create 8 high-traffic matrices
high_tms = []
for i in range(8):
    # Create a high-traffic matrix based on medium-traffic matrices
    base_tm = medium_tms[np.random.randint(0, len(medium_tms))]
    new_tm = copy.deepcopy(base_tm)
    for row in range(num_nodes):
        for col in range(num_nodes):
            if row != col:
                # Scale up by 1.5-2.0x
                scale = 1.5 + np.random.random() * 0.5  # 1.5 to 2.0
                new_tm[row][col] = max(1, round(new_tm[row][col] * scale))
    high_tms.append(new_tm)

print(f"Created {len(high_tms)} high-traffic matrices")

# Combine all matrices in the desired order
# Low, Medium, High
new_tm_list = []
new_tm_list.extend([low_tm])  # Original low traffic matrix
new_tm_list.extend(new_low_tms)  # 7 new low traffic matrices
new_tm_list.extend(medium_tms)  # Original medium traffic matrices (6)
new_tm_list.extend(new_medium_tms)  # 2 new medium traffic matrices
new_tm_list.extend(high_tms)  # 8 new high traffic matrices

print(f"Total number of traffic matrices: {len(new_tm_list)}")
print(f"Distribution: 8 low, 8 medium, 8 high")

# Update the config
config["tm_list"] = new_tm_list

# Save the updated config
output_file = "config_17node_updated.json"
with open(output_file, 'w') as f:
    json.dump(config, f, indent=2)

print(f"Updated config saved to {output_file}")
print("Please review the file and then rename it to config_17node.json if satisfied")
