import json

with open("config_17node_updated.json", 'r') as f:
    config = json.load(f)

num_edges = sum(sum(row) for row in config["adj_matrix"]) // 2
num_tms = len(config["tm_list"])
edge_list_count = len(config["edge_list"])

print(f"Number of edges in adjacency matrix: {num_edges}")
print(f"Number of edges in edge list: {edge_list_count}")
print(f"Number of traffic matrices: {num_tms}")

# Analyze traffic matrices
tm_list = config["tm_list"]
tm_averages = []

for i, tm in enumerate(tm_list):
    # Calculate average demand (excluding self-links)
    total = 0
    count = 0
    for row in range(len(tm)):
        for col in range(len(tm[0])):
            if row != col:
                total += tm[row][col]
                count += 1
    avg = total / count
    tm_averages.append(avg)
    print(f"TM {i}: Average traffic demand = {avg:.2f}")

# Categorize matrices
print("\nTraffic Matrix Categories:")
print(f"Low traffic (first 8): Avg = {sum(tm_averages[:8])/8:.2f}")
print(f"Medium traffic (next 8): Avg = {sum(tm_averages[8:16])/8:.2f}")
print(f"High traffic (last 8): Avg = {sum(tm_averages[16:])/8:.2f}")
