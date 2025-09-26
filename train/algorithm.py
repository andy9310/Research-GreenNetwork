import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class LinkState:
    """Represents the state of a network link"""
    u: int
    v: int
    active: bool
    utilization: float
    importance: float
    criticality: float
    capacity: float
    delay: float

@dataclass
class ClusterInfo:
    """Information about a network cluster"""
    cluster_id: int
    nodes: List[int]
    edges: List[Tuple[int, int]]
    active_edges: List[Tuple[int, int]]
    utilization: float
    energy_cost: float
    latency: float

class LinkDeactivationAlgorithm(ABC):
    """Abstract base class for link deactivation algorithms"""
    
    @abstractmethod
    def deactivate_links(self, graph, cluster_info, threshold, **kwargs):
        """Deactivate links based on algorithm strategy"""
        pass

class GreedyEnergySaving(LinkDeactivationAlgorithm):
    """Greedy algorithm that deactivates links with low utilization"""
    
    def deactivate_links(self, graph, cluster_info, threshold, **kwargs):
        """Deactivate links with utilization below threshold"""
        deactivated = []
        
        for u, v in cluster_info.edges:
            if graph.has_edge(u, v):
                edge_data = graph[u][v]
                utilization = edge_data.get("utilization", 0.0)
                
                if utilization < threshold and edge_data.get("active", 1) == 1:
                    graph[u][v]["active"] = 0
                    deactivated.append((u, v))
        
        return deactivated

class PriorityBasedDeactivation(LinkDeactivationAlgorithm):
    """Priority-based algorithm considering flow priorities"""
    
    def deactivate_links(self, graph, cluster_info, threshold, flows=None, **kwargs):
        """Deactivate links considering flow priorities"""
        deactivated = []
        
        # Calculate link importance based on flows
        link_importance = {}
        for u, v in cluster_info.edges:
            if graph.has_edge(u, v):
                importance = 0.0
                for flow in flows or []:
                    if (flow.s == u and flow.t == v) or (flow.s == v and flow.t == u):
                        # Higher priority flows get more weight
                        priority_weight = {1: 3.0, 2: 2.0, 3: 1.5, 4: 1.2, 5: 1.0, 6: 0.5}
                        importance += flow.size * priority_weight.get(flow.prio, 1.0)
                
                link_importance[(u, v)] = importance
        
        # Sort links by importance (least important first)
        sorted_links = sorted(link_importance.items(), key=lambda x: x[1])
        
        for (u, v), importance in sorted_links:
            if graph.has_edge(u, v):
                edge_data = graph[u][v]
                utilization = edge_data.get("utilization", 0.0)
                
                if utilization < threshold and edge_data.get("active", 1) == 1:
                    graph[u][v]["active"] = 0
                    deactivated.append((u, v))
        
        return deactivated

class DeterministicLinkManager:
    """Manages deterministic link deactivation strategies"""
    
    def __init__(self, algorithm_type: str = "greedy"):
        self.algorithm_type = algorithm_type
        self.algorithms = {
            "greedy": GreedyEnergySaving(),
            "priority": PriorityBasedDeactivation()
        }
        self.algorithm = self.algorithms.get(algorithm_type, GreedyEnergySaving())
    
    def apply_complete_deactivation_strategy(self, graph, region_of, thresholds, inter_keep_min, flows=None):
        """Apply complete deactivation strategy"""
        # Step 1: Apply per-cluster deactivation
        self._apply_per_cluster_deactivation(graph, region_of, thresholds, flows)
        
        # Step 2: Apply inter-cluster deactivation
        self._apply_inter_cluster_deactivation(graph, region_of, inter_keep_min)
        
        # Step 3: Ensure connectivity
        self._ensure_connectivity(graph, region_of)
    
    def _apply_per_cluster_deactivation(self, graph, region_of, thresholds, flows):
        """Apply deactivation within each cluster"""
        clusters = {}
        
        # Group nodes by cluster
        for node in graph.nodes():
            cluster_id = region_of.get(node, 0)
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(node)
        
        # Apply deactivation for each cluster
        for cluster_id, nodes in clusters.items():
            if cluster_id < len(thresholds):
                threshold = thresholds[cluster_id]
                
                # Get cluster edges
                cluster_edges = []
                for u in nodes:
                    for v in graph.neighbors(u):
                        if region_of.get(v, 0) == cluster_id:
                            cluster_edges.append((u, v))
                
                # Create cluster info
                cluster_info = ClusterInfo(
                    cluster_id=cluster_id,
                    nodes=nodes,
                    edges=cluster_edges,
                    active_edges=[e for e in cluster_edges if graph[e[0]][e[1]].get("active", 1) == 1],
                    utilization=0.0,
                    energy_cost=0.0,
                    latency=0.0
                )
                
                # Apply deactivation algorithm
                self.algorithm.deactivate_links(graph, cluster_info, threshold, flows=flows)
    
    def _apply_inter_cluster_deactivation(self, graph, region_of, inter_keep_min):
        """Apply deactivation for inter-cluster links"""
        # Get all inter-cluster edges
        inter_edges = []
        for u, v in graph.edges():
            if region_of.get(u, 0) != region_of.get(v, 0):
                inter_edges.append((u, v))
        
        # Sort by importance (degree-based)
        node_degrees = dict(graph.degree())
        edge_scores = []
        for u, v in inter_edges:
            score = node_degrees.get(u, 0) + node_degrees.get(v, 0)
            edge_scores.append(((u, v), score))
        
        # Sort by score (highest first)
        edge_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top inter_keep_min edges, deactivate others
        edges_to_keep = [edge for edge, _ in edge_scores[:inter_keep_min]]
        
        for u, v in inter_edges:
            if (u, v) not in edges_to_keep and (v, u) not in edges_to_keep:
                if graph.has_edge(u, v):
                    graph[u][v]["active"] = 0
    
    def _ensure_connectivity(self, graph, region_of):
        """Ensure network connectivity by reactivating critical links"""
        # Check if graph is connected
        active_graph = nx.Graph()
        for u, v, data in graph.edges(data=True):
            if data.get("active", 1) == 1:
                active_graph.add_edge(u, v)
        
        if nx.is_connected(active_graph):
            return  # Already connected
        
        # Find disconnected components
        components = list(nx.connected_components(active_graph))
        
        # Reactivate edges to connect components
        for i in range(len(components) - 1):
            comp1 = components[i]
            comp2 = components[i + 1]
            
            # Find shortest path in full graph
            try:
                # Get representative nodes from each component
                node1 = list(comp1)[0]
                node2 = list(comp2)[0]
                
                # Find path in full graph
                if graph.has_node(node1) and graph.has_node(node2):
                    path = nx.shortest_path(graph, node1, node2)
                    
                    # Reactivate edges along the path
                    for j in range(len(path) - 1):
                        u, v = path[j], path[j + 1]
                        if graph.has_edge(u, v):
                            graph[u][v]["active"] = 1
                            
            except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError):
                # If no path exists, reactivate a random edge between components
                for u in comp1:
                    for v in comp2:
                        if graph.has_edge(u, v):
                            graph[u][v]["active"] = 1
                            break
                    else:
                        continue
                    break
    
    def apply_deterministic_refinement(self, graph, flows, region_of):
        """Apply deterministic refinement to ensure flow connectivity"""
        if not flows:
            return
        
        # Build active graph
        active_graph = nx.Graph()
        for u, v, data in graph.edges(data=True):
            if data.get("active", 1) == 1:
                active_graph.add_edge(u, v)
        
        # Check each flow for connectivity
        for flow in flows:
            s, t = flow.s, flow.t
            
            # Skip if nodes don't exist in graph
            if not graph.has_node(s) or not graph.has_node(t):
                continue
            
            # Check if path exists in active graph
            if not active_graph.has_node(s) or not active_graph.has_node(t):
                continue
            
            try:
                # Try to find path in active graph
                path = nx.shortest_path(active_graph, s, t)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # No path exists, try to find one in full graph
                try:
                    if graph.has_node(s) and graph.has_node(t):
                        path = nx.shortest_path(graph, s, t)
                        
                        # Reactivate edges along the path
                        for i in range(len(path) - 1):
                            u, v = path[i], path[i + 1]
                            if graph.has_edge(u, v):
                                graph[u][v]["active"] = 1
                                # Update active graph
                                active_graph.add_edge(u, v)
                                
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    # If still no path, skip this flow
                    continue