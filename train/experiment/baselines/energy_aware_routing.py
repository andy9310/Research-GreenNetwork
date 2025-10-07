"""
Energy-Aware Routing (EAR) Baseline
Based on: "Energy-aware routing algorithms in Software-Defined Networks"

Heuristic approach: Deactivate links with low utilization while maintaining connectivity
"""

import networkx as nx
import numpy as np
import time


class EnergyAwareRouting:
    """
    Heuristic-based energy-aware routing algorithm
    """
    
    def __init__(self, utilization_threshold=0.3, min_active_ratio=0.2):
        """
        Args:
            utilization_threshold: Links below this utilization can be deactivated
            min_active_ratio: Minimum fraction of links that must stay active
        """
        self.utilization_threshold = utilization_threshold
        self.min_active_ratio = min_active_ratio
        self.computation_times = []
        
    def select_links(self, graph, flows, traffic_matrix):
        """
        Select which links to keep active
        
        Args:
            graph: NetworkX graph with link attributes
            flows: List of active flows
            traffic_matrix: n x n traffic demand matrix
            
        Returns:
            active_links: Set of (u, v) tuples for active links
            metrics: Dict of performance metrics
        """
        start_time = time.time()
        
        n_edges = graph.number_of_edges()
        min_active = max(int(n_edges * self.min_active_ratio), 20)
        
        # Step 1: Calculate link utilization
        link_util = self._calculate_utilization(graph, flows)
        
        # Step 2: Rank links by importance (degree centrality + utilization)
        link_importance = self._calculate_importance(graph, link_util)
        
        # Step 3: Greedy selection
        active_links = set()
        sorted_links = sorted(link_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Always keep most important links
        for (u, v), importance in sorted_links[:min_active]:
            active_links.add((u, v))
        
        # Add links with high utilization
        for (u, v), util in link_util.items():
            if util > self.utilization_threshold:
                active_links.add((u, v))
        
        # Step 4: Ensure connectivity
        active_links = self._ensure_connectivity(graph, active_links)
        
        comp_time = time.time() - start_time
        self.computation_times.append(comp_time)
        
        # Calculate metrics
        metrics = {
            'active_links': len(active_links),
            'energy_saving': (n_edges - len(active_links)) / n_edges * 100,
            'computation_time': comp_time
        }
        
        return active_links, metrics
    
    def _calculate_utilization(self, graph, flows):
        """Calculate utilization for each link"""
        util = {(u, v): 0.0 for u, v in graph.edges()}
        
        if not flows:
            return util
        
        # Simple utilization: count flows using each link
        for flow in flows:
            try:
                path = nx.shortest_path(graph, flow.s, flow.t)
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    if u > v:
                        u, v = v, u
                    if (u, v) in util:
                        util[(u, v)] += flow.size / 1000.0
            except:
                continue
        
        # Normalize by capacity
        for (u, v) in util:
            cap = graph[u][v].get('capacity', 5.0)
            util[(u, v)] = util[(u, v)] / cap
        
        return util
    
    def _calculate_importance(self, graph, link_util):
        """Calculate importance score for each link"""
        importance = {}
        node_degrees = dict(graph.degree())
        
        for u, v in graph.edges():
            # Importance = degree centrality + utilization
            degree_score = (node_degrees[u] + node_degrees[v]) / (2 * graph.number_of_nodes())
            util_score = link_util.get((u, v), 0.0)
            importance[(u, v)] = 0.6 * degree_score + 0.4 * util_score
        
        return importance
    
    def _ensure_connectivity(self, graph, active_links):
        """Ensure network remains connected"""
        # Build active subgraph
        H = nx.Graph()
        H.add_nodes_from(graph.nodes())
        H.add_edges_from(active_links)
        
        # If not connected, add bridges
        if not nx.is_connected(H):
            components = list(nx.connected_components(H))
            
            # Connect components
            for i in range(len(components) - 1):
                comp1 = components[i]
                comp2 = components[i + 1]
                
                # Find shortest bridge
                min_path = None
                min_len = float('inf')
                
                for n1 in comp1:
                    for n2 in comp2:
                        if graph.has_edge(n1, n2):
                            active_links.add((n1, n2) if n1 < n2 else (n2, n1))
                            break
                    else:
                        continue
                    break
        
        return active_links
    
    def get_stats(self):
        """Get algorithm statistics"""
        return {
            'avg_computation_time': np.mean(self.computation_times) if self.computation_times else 0,
            'total_decisions': len(self.computation_times)
        }
