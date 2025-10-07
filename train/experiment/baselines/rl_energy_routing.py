"""
RL-Based Energy Routing (RL-ER) Baseline
Based on: "Reinforcement Learning and Energy-Aware Routing"

Basic Q-learning without clustering - global state representation
"""

import numpy as np
import networkx as nx
import time
from collections import defaultdict


class RLEnergyRouting:
    """
    Basic Q-learning for energy-aware routing
    """
    
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        """
        Args:
            learning_rate: Q-learning alpha
            discount_factor: Q-learning gamma
            epsilon: Exploration rate
        """
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        # Q-table: state -> action -> value
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.computation_times = []
        
    def select_action(self, state, graph):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Discretized network state
            graph: NetworkX graph
            
        Returns:
            action: Link to deactivate (u, v) or None
        """
        start_time = time.time()
        
        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            # Random action: select random active link
            active_links = [(u, v) for u, v, d in graph.edges(data=True) if d.get('active', 1) == 1]
            action = active_links[np.random.randint(len(active_links))] if active_links else None
        else:
            # Greedy action: select best from Q-table
            state_key = self._state_to_key(state)
            q_values = self.q_table[state_key]
            
            if q_values:
                action = max(q_values.items(), key=lambda x: x[1])[0]
            else:
                # No Q-values yet, random
                active_links = [(u, v) for u, v, d in graph.edges(data=True) if d.get('active', 1) == 1]
                action = active_links[np.random.randint(len(active_links))] if active_links else None
        
        comp_time = time.time() - start_time
        self.computation_times.append(comp_time)
        
        return action
    
    def update(self, state, action, reward, next_state):
        """
        Update Q-table using Q-learning update rule
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        # Q-learning update
        current_q = self.q_table[state_key][action]
        max_next_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0.0
        
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action] = new_q
    
    def get_state(self, graph, flows):
        """
        Get discretized state representation
        
        Args:
            graph: NetworkX graph
            flows: List of active flows
            
        Returns:
            state: Tuple of state features
        """
        # Simple state: (avg_utilization_bin, num_flows_bin, num_active_links_bin)
        active_links = sum(1 for _, _, d in graph.edges(data=True) if d.get('active', 1) == 1)
        total_links = graph.number_of_edges()
        
        # Discretize
        util_bin = min(int(active_links / total_links * 10), 9)  # 0-9
        flow_bin = min(int(len(flows) / 50), 9)  # 0-9
        link_bin = min(int(active_links / 100), 9)  # 0-9
        
        return (util_bin, flow_bin, link_bin)
    
    def _state_to_key(self, state):
        """Convert state tuple to hashable key"""
        return str(state)
    
    def get_stats(self):
        """Get algorithm statistics"""
        return {
            'avg_computation_time': np.mean(self.computation_times) if self.computation_times else 0,
            'q_table_size': len(self.q_table),
            'total_updates': len(self.computation_times)
        }
