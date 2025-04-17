import json
import random
import os

# Path to the test config file
test_config_path = os.path.join(os.path.dirname(__file__), "test_config_17node.json")

# Load the existing config file
with open(test_config_path, 'r') as f:
    config = json.load(f)

# Number of nodes
num_nodes = config["num_nodes"]

# Generate 10 traffic matrices
tm_list = []
for _ in range(10):
    tm = []
    for i in range(num_nodes):
        row = []
        for j in range(num_nodes):
            if i == j:
                # No traffic to self
                row.append(0)
            else:
                # Random traffic between 10-35
                row.append(random.randint(10, 35))
        tm.append(row)
    tm_list.append(tm)

# Update the config with the traffic matrices
config["tm_list"] = tm_list

# Save the updated config
with open(test_config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"Successfully generated 10 traffic matrices for {num_nodes} nodes")
print(f"Updated file: {test_config_path}")
