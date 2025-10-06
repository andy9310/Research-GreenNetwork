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

class SLAEnhancedHeuristic(LinkDeactivationAlgorithm):
    """
    增強版啟發式演算法，包含SLA考量
    輸出5個門檻值：buffer_threshold, growth_rate_threshold, link_usage_threshold, 
    sla_safety_margin, overload_threshold
    """
    
    def __init__(self):
        self.buffer_history = {}  # 記錄每個節點的buffer歷史
        self.growth_rates = {}   # 記錄成長率
        self.sla_violations = 0  # SLA違反計數
        
    def deactivate_links(self, graph, cluster_info, threshold, flows=None, 
                        buffer_threshold=0.3, growth_rate_threshold=0.3,
                        link_usage_threshold=0.5, sla_safety_margin=0.1,
                        overload_threshold=0.9, **kwargs):
        """
        執行SLA增強的啟發式演算法
        
        Args:
            buffer_threshold: buffer長度門檻 (0-1)
            growth_rate_threshold: 成長率門檻 (0-1)
            link_usage_threshold: 連結使用率門檻 (0-1)
            sla_safety_margin: SLA安全邊際 (0-1)
            overload_threshold: 過載門檻 (0-1)
        """
        deactivated = []
        
        # 1. 識別低負載節點
        low_load_nodes = self._identify_low_load_nodes(
            graph, cluster_info, buffer_threshold, growth_rate_threshold
        )
        
        # 2. 找出可關閉的連結
        candidate_links = self._find_candidate_links_to_close(
            graph, cluster_info, low_load_nodes, link_usage_threshold
        )
        
        # 3. SLA約束檢查
        safe_links_to_close = []
        for link in candidate_links:
            if self._is_sla_safe_to_close(graph, link, flows, sla_safety_margin):
                safe_links_to_close.append(link)
        
        # 4. 執行關閉並重導向流量
        for link in safe_links_to_close:
            if self._redirect_traffic_safely(graph, link, overload_threshold):
                graph[link[0]][link[1]]["active"] = 0
                deactivated.append(link)
        
        return deactivated
    
    def _identify_low_load_nodes(self, graph, cluster_info, buffer_threshold, growth_rate_threshold):
        """識別低負載節點"""
        low_load_nodes = []
        
        for node in cluster_info.nodes:
            # 計算buffer長度 (模擬)
            buffer_length = self._calculate_buffer_length(graph, node)
            
            # 計算成長率
            growth_rate = self._calculate_growth_rate(node)
            
            # 檢查是否為低負載
            if buffer_length < buffer_threshold and growth_rate < growth_rate_threshold:
                low_load_nodes.append(node)
        
        return low_load_nodes
    
    def _calculate_buffer_length(self, graph, node):
        """計算節點的buffer長度 (模擬)"""
        # 基於節點度數和鄰居使用率估算
        degree = graph.degree(node)
        if degree == 0:
            return 0.0
        
        # 計算鄰居的平均使用率
        neighbor_utilization = 0.0
        neighbor_count = 0
        
        for neighbor in graph.neighbors(node):
            if graph.has_edge(node, neighbor):
                edge_data = graph[node][neighbor]
                utilization = edge_data.get("utilization", 0.0)
                neighbor_utilization += utilization
                neighbor_count += 1
        
        avg_neighbor_util = neighbor_utilization / max(1, neighbor_count)
        
        # Buffer長度與使用率成反比
        buffer_length = max(0.0, 1.0 - avg_neighbor_util)
        return buffer_length
    
    def _calculate_growth_rate(self, node):
        """計算節點的成長率"""
        if node not in self.buffer_history:
            self.buffer_history[node] = []
            self.growth_rates[node] = 0.0
            return 0.0
        
        # 計算過去5秒的成長率
        history = self.buffer_history[node]
        if len(history) < 2:
            return 0.0
        
        # 簡單的線性成長率計算
        recent_values = history[-5:] if len(history) >= 5 else history
        if len(recent_values) < 2:
            return 0.0
        
        growth_rate = (recent_values[-1] - recent_values[0]) / max(1, len(recent_values) - 1)
        self.growth_rates[node] = growth_rate
        return growth_rate
    
    def _find_candidate_links_to_close(self, graph, cluster_info, low_load_nodes, link_usage_threshold):
        """找出可關閉的連結候選"""
        candidate_links = []
        
        for node in low_load_nodes:
            for neighbor in graph.neighbors(node):
                if neighbor in cluster_info.nodes:  # 只考慮cluster內的連結
                    edge_data = graph[node][neighbor]
                    utilization = edge_data.get("utilization", 0.0)
                    
                    # 保留高使用率的連結
                    if utilization < link_usage_threshold:
                        candidate_links.append((node, neighbor))
        
        return candidate_links
    
    def _is_sla_safe_to_close(self, graph, link, flows, sla_safety_margin):
        """檢查關閉連結是否安全 (不會違反SLA)"""
        if not flows:
            return True
        
        # 模擬關閉連結
        original_state = graph[link[0]][link[1]].get("active", 1)
        graph[link[0]][link[1]]["active"] = 0
        
        # 檢查每個flow是否還能滿足SLA
        sla_violations = 0
        for flow in flows:
            if self._would_violate_sla(flow, graph):
                sla_violations += 1
        
        # 恢復原始狀態
        graph[link[0]][link[1]]["active"] = original_state
        
        # 如果違反率超過安全邊際，則不安全
        violation_rate = sla_violations / max(1, len(flows))
        return violation_rate <= sla_safety_margin
    
    def _would_violate_sla(self, flow, graph):
        """檢查特定flow是否會違反SLA"""
        try:
            # 計算新路徑的延遲
            path = nx.shortest_path(graph, flow.s, flow.t, weight="delay_ms")
            total_delay = 0.0
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if graph.has_edge(u, v):
                    edge_data = graph[u][v]
                    total_delay += edge_data.get("delay_ms", 0.0)
            
            # 檢查是否超過SLA門檻
            sla_threshold = self._get_sla_threshold(flow.prio)
            return total_delay > sla_threshold
            
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return True  # 無法找到路徑視為違反SLA
    
    def _get_sla_threshold(self, priority):
        """根據優先級獲取SLA門檻"""
        sla_thresholds = {1: 1.0, 2: 2.0, 3: 4.0, 4: 6.0, 5: 8.0, 6: 10.0}
        return sla_thresholds.get(priority, 10.0)
    
    def _redirect_traffic_safely(self, graph, link, overload_threshold):
        """安全地重導向流量"""
        # 檢查重導向後是否會造成過載
        affected_links = self._get_affected_links(graph, link)
        
        for affected_link in affected_links:
            u, v = affected_link
            if graph.has_edge(u, v):
                current_util = graph[u][v].get("utilization", 0.0)
                # 模擬重導向後的負載
                new_util = current_util + 0.1  # 假設增加10%負載
                
                if new_util > overload_threshold:
                    return False  # 會造成過載，不關閉
        
        return True
    
    def _get_affected_links(self, graph, closed_link):
        """獲取受影響的連結"""
        # 簡化實現：返回關閉連結的鄰居
        affected = []
        u, v = closed_link
        
        for neighbor in graph.neighbors(u):
            if neighbor != v:
                affected.append((u, neighbor))
        
        for neighbor in graph.neighbors(v):
            if neighbor != u:
                affected.append((v, neighbor))
        
        return affected

class HeuristicThresholdModel:
    """輸出啟發式演算法門檻值的模型"""
    
    def __init__(self, obs_dim, action_dim=5):
        """
        Args:
            obs_dim: 觀察空間維度
            action_dim: 動作空間維度 (5個門檻值)
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 門檻值範圍
        self.threshold_ranges = {
            'buffer_threshold': (0.1, 0.8),
            'growth_rate_threshold': (0.1, 0.8), 
            'link_usage_threshold': (0.2, 0.9),
            'sla_safety_margin': (0.05, 0.3),
            'overload_threshold': (0.7, 0.95)
        }
    
    def predict_thresholds(self, network_state):
        """根據網路狀態預測最佳門檻值"""
        # 這裡可以是用神經網路或其他ML模型
        # 目前使用簡單的規則基礎方法
        
        thresholds = {}
        
        # 根據網路負載調整門檻值
        avg_utilization = network_state.get('avg_utilization', 0.5)
        
        if avg_utilization < 0.3:  # 低負載
            thresholds['buffer_threshold'] = 0.2
            thresholds['growth_rate_threshold'] = 0.2
            thresholds['link_usage_threshold'] = 0.3
            thresholds['sla_safety_margin'] = 0.1
            thresholds['overload_threshold'] = 0.8
        elif avg_utilization < 0.6:  # 中負載
            thresholds['buffer_threshold'] = 0.4
            thresholds['growth_rate_threshold'] = 0.4
            thresholds['link_usage_threshold'] = 0.5
            thresholds['sla_safety_margin'] = 0.15
            thresholds['overload_threshold'] = 0.85
        else:  # 高負載
            thresholds['buffer_threshold'] = 0.6
            thresholds['growth_rate_threshold'] = 0.6
            thresholds['link_usage_threshold'] = 0.7
            thresholds['sla_safety_margin'] = 0.2
            thresholds['overload_threshold'] = 0.9
        
        return thresholds
    
    def get_action_space_size(self):
        """獲取動作空間大小"""
        return self.action_dim

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
        
        # Ensure minimum number of active links (at least 10% of total for adequate capacity)
        # Reduced from 200 to 150 to allow agent more flexibility
        total_edges = graph.number_of_edges()
        min_active_links = max(150, int(0.075 * total_edges))  # At least 150 or 7.5% of edges
        current_active = active_graph.number_of_edges()
        
        if current_active < min_active_links:
            # Activate more links to ensure good connectivity
            # Strategy: Use betweenness centrality to activate important bridge edges
            import random
            inactive_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("active", 1) == 0]
            
            # Calculate edge betweenness on full graph (expensive but important)
            try:
                edge_betweenness = nx.edge_betweenness_centrality(graph, k=min(50, graph.number_of_nodes()))
                # Sort by betweenness (most important bridges first)
                inactive_edges.sort(key=lambda e: edge_betweenness.get((e[0], e[1]), edge_betweenness.get((e[1], e[0]), 0)), reverse=True)
            except:
                # Fallback: sort by degree
                node_degrees = dict(graph.degree())
                inactive_edges.sort(key=lambda e: node_degrees[e[0]] + node_degrees[e[1]], reverse=True)
            
            # Activate top edges until we reach minimum
            needed = min_active_links - current_active
            for u, v in inactive_edges[:needed]:
                graph[u][v]["active"] = 1
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