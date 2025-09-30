import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random

@dataclass
class HeuristicThresholds:
    """啟發式演算法的門檻值"""
    buffer_threshold: float = 0.3      # 30% buffer長度門檻
    growth_rate_threshold: float = 0.3  # 30% 成長率門檻  
    link_usage_threshold: float = 0.5   # 50% 連結使用率門檻
    sla_safety_margin: float = 0.1     # 10% SLA安全邊際
    overload_threshold: float = 0.9     # 90% 過載門檻

class SLAEnhancedHeuristic:
    """
    增強版啟發式演算法，包含SLA考量
    
    動作空間：5個連續門檻值
    - buffer_threshold: 0.1-0.8
    - growth_rate_threshold: 0.1-0.8  
    - link_usage_threshold: 0.2-0.9
    - sla_safety_margin: 0.05-0.3
    - overload_threshold: 0.7-0.95
    """
    
    def __init__(self):
        self.buffer_history = {}  # 記錄每個節點的buffer歷史
        self.growth_rates = {}   # 記錄成長率
        self.sla_violations = 0  # SLA違反計數
        
    def execute_heuristic(self, graph, region_of, thresholds: HeuristicThresholds, flows=None):
        """
        執行SLA增強的啟發式演算法
        
        Args:
            graph: 網路圖
            region_of: 節點到區域的映射
            thresholds: 啟發式門檻值
            flows: 當前流量列表
            
        Returns:
            deactivated_links: 被關閉的連結列表
        """
        deactivated_links = []
        
        # 按區域執行演算法
        regions = self._group_nodes_by_region(region_of)
        
        for region_id, nodes in regions.items():
            region_deactivated = self._execute_region_heuristic(
                graph, nodes, thresholds, flows
            )
            deactivated_links.extend(region_deactivated)
        
        # 執行inter-cluster連結管理
        self._manage_inter_cluster_links(graph, region_of, thresholds)
        
        # 確保連通性
        self._ensure_connectivity(graph)
        
        return deactivated_links
    
    def _execute_region_heuristic(self, graph, nodes, thresholds, flows):
        """在單一區域內執行啟發式演算法"""
        deactivated = []
        
        # 1. 識別低負載節點
        low_load_nodes = self._identify_low_load_nodes(
            graph, nodes, thresholds.buffer_threshold, 
            thresholds.growth_rate_threshold
        )
        
        # 2. 找出可關閉的連結
        candidate_links = self._find_candidate_links_to_close(
            graph, nodes, low_load_nodes, thresholds.link_usage_threshold
        )
        
        # 3. SLA約束檢查
        safe_links_to_close = []
        for link in candidate_links:
            if self._is_sla_safe_to_close(graph, link, flows, thresholds.sla_safety_margin):
                safe_links_to_close.append(link)
        
        # 4. 執行關閉並重導向流量
        for link in safe_links_to_close:
            if self._redirect_traffic_safely(graph, link, thresholds.overload_threshold):
                graph[link[0]][link[1]]["active"] = 0
                deactivated.append(link)
        
        return deactivated
    
    def _identify_low_load_nodes(self, graph, nodes, buffer_threshold, growth_rate_threshold):
        """識別低負載節點"""
        low_load_nodes = []
        
        for node in nodes:
            # 計算buffer長度
            buffer_length = self._calculate_buffer_length(graph, node)
            
            # 計算成長率
            growth_rate = self._calculate_growth_rate(node)
            
            # 檢查是否為低負載
            if buffer_length < buffer_threshold and growth_rate < growth_rate_threshold:
                low_load_nodes.append(node)
        
        return low_load_nodes
    
    def _calculate_buffer_length(self, graph, node):
        """計算節點的buffer長度"""
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
        
        # 更新歷史記錄
        if node not in self.buffer_history:
            self.buffer_history[node] = []
        self.buffer_history[node].append(buffer_length)
        
        # 保持歷史記錄長度
        if len(self.buffer_history[node]) > 10:
            self.buffer_history[node] = self.buffer_history[node][-10:]
        
        return buffer_length
    
    def _calculate_growth_rate(self, node):
        """計算節點的成長率"""
        if node not in self.buffer_history:
            self.buffer_history[node] = []
            return 0.0
        
        history = self.buffer_history[node]
        if len(history) < 2:
            return 0.0
        
        # 計算過去5個時間步的成長率
        recent_values = history[-5:] if len(history) >= 5 else history
        if len(recent_values) < 2:
            return 0.0
        
        growth_rate = (recent_values[-1] - recent_values[0]) / max(1, len(recent_values) - 1)
        self.growth_rates[node] = growth_rate
        return growth_rate
    
    def _find_candidate_links_to_close(self, graph, nodes, low_load_nodes, link_usage_threshold):
        """找出可關閉的連結候選"""
        candidate_links = []
        
        for node in low_load_nodes:
            for neighbor in graph.neighbors(node):
                if neighbor in nodes:  # 只考慮區域內的連結
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
        affected = []
        u, v = closed_link
        
        for neighbor in graph.neighbors(u):
            if neighbor != v:
                affected.append((u, neighbor))
        
        for neighbor in graph.neighbors(v):
            if neighbor != u:
                affected.append((v, neighbor))
        
        return affected
    
    def _group_nodes_by_region(self, region_of):
        """按區域分組節點"""
        regions = {}
        for node, region_id in region_of.items():
            if region_id not in regions:
                regions[region_id] = []
            regions[region_id].append(node)
        return regions
    
    def _manage_inter_cluster_links(self, graph, region_of, thresholds):
        """管理inter-cluster連結"""
        # 獲取所有inter-cluster邊
        inter_edges = []
        for u, v in graph.edges():
            if region_of.get(u, 0) != region_of.get(v, 0):
                inter_edges.append((u, v))
        
        # 根據使用率決定是否關閉
        for u, v in inter_edges:
            edge_data = graph[u][v]
            utilization = edge_data.get("utilization", 0.0)
            
            # 如果使用率低於門檻，考慮關閉
            if utilization < thresholds.link_usage_threshold:
                # 檢查是否安全關閉
                if self._is_sla_safe_to_close(graph, (u, v), [], thresholds.sla_safety_margin):
                    graph[u][v]["active"] = 0
    
    def _ensure_connectivity(self, graph):
        """確保網路連通性"""
        # 檢查是否連通
        active_graph = nx.Graph()
        for u, v, data in graph.edges(data=True):
            if data.get("active", 1) == 1:
                active_graph.add_edge(u, v)
        
        if nx.is_connected(active_graph):
            return  # 已經連通
        
        # 找到斷開的組件並重新連接
        components = list(nx.connected_components(active_graph))
        
        for i in range(len(components) - 1):
            comp1 = components[i]
            comp2 = components[i + 1]
            
            # 嘗試找到連接路徑
            for u in comp1:
                for v in comp2:
                    if graph.has_edge(u, v):
                        graph[u][v]["active"] = 1
                        break
                else:
                    continue
                break

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
        sla_violation_rate = network_state.get('sla_violation_rate', 0.0)
        
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
        
        # 根據SLA違反率調整安全邊際
        if sla_violation_rate > 0.1:  # 高違反率
            thresholds['sla_safety_margin'] = min(0.3, thresholds['sla_safety_margin'] + 0.05)
        
        return HeuristicThresholds(**thresholds)
    
    def get_action_space_size(self):
        """獲取動作空間大小"""
        return self.action_dim
    
    def decode_action(self, action_vector):
        """將動作向量解碼為門檻值"""
        if len(action_vector) != self.action_dim:
            raise ValueError(f"Action vector length {len(action_vector)} != {self.action_dim}")
        
        # 將[0,1]範圍映射到實際門檻值範圍
        thresholds = {}
        for i, (key, (min_val, max_val)) in enumerate(self.threshold_ranges.items()):
            normalized_val = action_vector[i]
            thresholds[key] = min_val + normalized_val * (max_val - min_val)
        
        return HeuristicThresholds(**thresholds) 