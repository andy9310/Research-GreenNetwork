import json
import sys

def count_traffic_matrices(config_path):
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    if 'tm_list' in data:
        print(f"Number of traffic matrices in {config_path}: {len(data['tm_list'])}")
    else:
        print(f"No 'tm_list' found in {config_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "../../train_configs/config_17node_25edges.json"
    
    count_traffic_matrices(config_path)
