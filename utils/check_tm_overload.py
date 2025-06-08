import json
import numpy as np
import networkx as nx
import argparse


def check_overload(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    edge_list = config["edge_list"]
    tm_list = config["tm_list"]
    num_nodes = config["num_nodes"]
    link_capacity = config["link_capacity"]
    
    # Build the networkx graph with all links open
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for idx, (u, v) in enumerate(edge_list):
        G.add_edge(u, v, id=idx, capacity=link_capacity)

    overloaded_any = False
    for tm_idx, tm in enumerate(tm_list):
        usage = np.zeros(len(edge_list), dtype=float)
        tm = np.array(tm)
        overloaded_edges = set()
        for src in range(num_nodes):
            for dst in range(num_nodes):
                if src != dst and tm[src][dst] > 0:
                    try:
                        path = nx.shortest_path(G, source=src, target=dst)
                        path_edges = list(zip(path[:-1], path[1:]))
                        for u, v in path_edges:
                            edge_id = G[u][v]['id']
                            usage[edge_id] += tm[src][dst]
                    except nx.NetworkXNoPath:
                        print(f"No path between {src} and {dst} in TM {tm_idx}")
        # Check for overloads
        for idx, load in enumerate(usage):
            if load > link_capacity:
                overloaded_edges.add(idx)
        if overloaded_edges:
            overloaded_any = True
            print(f"[!] Traffic Matrix {tm_idx}: OVERLOADED edges: {sorted(list(overloaded_edges))}")
            for idx in overloaded_edges:
                u, v = edge_list[idx]
                print(f"    Edge ({u},{v}): load={usage[idx]}, capacity={link_capacity}")
        else:
            print(f"[OK] Traffic Matrix {tm_idx}: No overloads.")
    if not overloaded_any:
        print("All traffic matrices are safe (no overloads with all links open).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for overloaded edges in traffic matrices with all links open.")
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON')
    args = parser.parse_args()
    check_overload(args.config)
