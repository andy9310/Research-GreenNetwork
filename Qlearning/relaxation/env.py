"""
Relaxed Network Environment

This environment allows for continuous link capacity scaling factors (0-1)
instead of binary open/close decisions, enabling relaxation techniques.
"""

import sys
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import copy
import random

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class RelaxedNetworkEnv(gym.Env):
    """
    Network environment with relaxed (continuous) link capacity factors.
    
    This environment allows setting continuous capacity scaling factors (0-1)
    for each link instead of binary open/close decisions.
    """
    
    def __init__(self, adj_matrix, edge_list, tm_list, node_props, num_nodes, link_capacity=1.0, seed=None, max_edges=None, random_edge_order=False):
        """
        Initialize the relaxed network environment.
        
        Args:
            edge_list: List of edges [(u,v), ...] 
            tm_list: List of traffic matrices
            link_capacity: Capacity of each link (default: all 1.0)
            node_props: Properties of nodes
            num_nodes: Number of nodes (if not provided, inferred from edge_list)
            max_edges: Maximum number of edges (if not provided, inferred from edge_list)
            seed: Random seed
            random_edge_order: Whether to randomize edge order
        """
        super(RelaxedNetworkEnv, self).__init__()
        
        # Store adjacency matrix, edge list and traffic matrices
        self.adj_matrix = np.array(adj_matrix)
        self.edge_list = edge_list
        self.tm_list = [np.array(tm) for tm in tm_list]  # Ensure TMs are numpy arrays
        self.node_props = node_props
        self.num_nodes = num_nodes
        
        # Number of edges and links
        self.num_edges = len(self.edge_list)
        
        # Maximum edges for observation space
        if max_edges is None:
            self.max_edges = self.num_edges
        else:
            self.max_edges = max_edges
        
        # Link capacity - can be a single value for all links or a list
        if isinstance(link_capacity, (int, float)):
            self.link_capacity = [link_capacity] * self.num_edges
        else:
            self.link_capacity = link_capacity
        
        # Node properties
        self.node_props = node_props if node_props else {}
        
        # Current traffic matrix index
        self.current_tm_idx = 0
        
        # Random edge ordering
        self.random_edge_order = random_edge_order
        self.edge_order = list(range(self.num_edges))
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Network state: link capacity scaling factors (continuous 0-1)
        self.link_factors = np.ones(self.num_edges)
        
        # Link usage
        self.link_usage = np.zeros(self.num_edges)
        
        # Current edge index
        self.current_edge_idx = 0
        
        # Network graph
        self.G = nx.Graph()
        for i in range(self.num_nodes):
            self.G.add_node(i)
        for i, (u, v) in enumerate(self.edge_list):
            self.G.add_edge(u, v, capacity=self.link_capacity[i], weight=1.0)
        
        # Action space: continuous [0, 1] for link capacity scaling factor
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Observation space: link usage + current edge index
        obs_dim = self.max_edges + 1
        self.observation_space = spaces.Box(
            low=0.0, high=float('inf'), shape=(obs_dim,), dtype=np.float32
        )
    
    def reset(self):
        """Reset the environment to initial state."""
        # Reset link capacity scaling factors
        self.link_factors = np.ones(self.num_edges)
        
        # Reset link usage
        self.link_usage = np.zeros(self.num_edges)
        
        # Reset edge index
        self.current_edge_idx = 0
        
        # Reset graph
        self.G = nx.Graph()
        for i in range(self.num_nodes):
            self.G.add_node(i)
        for i, (u, v) in enumerate(self.edge_list):
            self.G.add_edge(u, v, capacity=self.link_capacity[i], weight=1.0)
        
        # Random edge ordering if enabled
        if self.random_edge_order:
            self.edge_order = np.random.permutation(self.num_edges).tolist()
        else:
            self.edge_order = list(range(self.num_edges))
        
        # Calculate initial routing
        self._calculate_routing()
        
        return self._get_observation()
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Continuous value [0, 1] representing the capacity scaling factor
                   for the current link
        
        Returns:
            observation: The new state
            reward: The reward
            done: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        # Initialize info dictionary
        info = {}
        done = False
        
        # Get the current edge index from the edge order
        edge_idx = self.edge_order[self.current_edge_idx]
        
        # Apply the action: set the capacity scaling factor for the current edge
        # Action is a continuous value between 0 and 1
        scaling_factor = float(action[0])  # Extract scalar value
        self.link_factors[edge_idx] = scaling_factor
        
        # If scaling factor is very low, remove the edge from the graph
        edge = self.edge_list[edge_idx]
        u, v = edge
        if scaling_factor < 0.01:  # Effectively closed
            if self.G.has_edge(u, v):
                self.G.remove_edge(u, v)
        else:  # Adjust the capacity
            if not self.G.has_edge(u, v):
                self.G.add_edge(u, v)
            # Update edge capacity
            self.G[u][v]['capacity'] = self.link_capacity[edge_idx] * scaling_factor
        
        # Strong incentive for fully closing links (capacity_factor < 0.01)
        if scaling_factor < 0.01:
            # Bonus reward for fully closing a link
            reward = 5.0
        else:
            # Continuous reward that increases significantly as capacity_factor approaches 0
            # Using a non-linear reward curve that accelerates as factor decreases
            reward = 2.0 * (1.0 - scaling_factor**0.5)  # Square root makes reward curve steeper at lower values
        
        # Check for network connectivity issues
        if not self.is_connected():
            # Get number of connected components and isolated nodes
            components = list(nx.connected_components(self.G))
            num_components = len(components)
            num_isolated = sum(1 for comp in components if len(comp) == 1)
            
            # Calculate severity-based penalty - using more moderate values
            disconnection_penalty = -20 * (num_components + num_isolated - 1)
            reward += disconnection_penalty
            info['violation'] = 'disconnection'
            info['link_idx'] = edge_idx
            info['num_components'] = num_components
            info['isolated_nodes'] = num_isolated
            done = True
            
            # Return early if disconnected
            return self._get_observation(), reward, done, False, info
        
        # Calculate routing if network is still connected
        self._calculate_routing()
        
        # Check for link overloads
        overloaded = False
        overload_amount = 0
        overloaded_links = []
        
        for i, usage in enumerate(self.link_usage):
            cap = self.link_capacity[i] * self.link_factors[i]
            if usage > cap and cap > 0:
                overloaded = True
                # Calculate how much the link is overloaded
                overload_ratio = usage / cap
                overload_amount += (overload_ratio - 1) * 100  # Scale for better gradient
                overloaded_links.append(i)
        
        if overloaded:
            # Apply overload penalty proportional to severity - using more moderate values
            reward += -15 * overload_amount
            info['violation'] = 'overload'
            info['link_idx'] = edge_idx
            info['overloaded_links'] = overloaded_links
            info['overload_amount'] = overload_amount
            done = True
            
            # Return early if overloaded
            return self._get_observation(), reward, done, False, info
        
        # Move to the next edge
        self.current_edge_idx += 1
        
        # Check if this is the last edge
        if self.current_edge_idx >= self.num_edges:
            done = True
        
        # Add the link factors and usage to info
        info['link_factors'] = self.link_factors.copy()
        info['link_usage'] = self.link_usage.copy()
        
        # Check if episode is done
        if self.current_edge_idx >= self.num_edges:
            self.current_edge_idx = self.num_edges - 1  # Keep at last edge
            done = True
        
        return self._get_observation(), reward, done, False, info
    
    def _get_observation(self):
        """Get the current observation."""
        # Create observation: link usage + current edge index
        obs = np.zeros(self.observation_space.shape[0])
        
        # Fill in link usage - this reveals whether links are open or closed
        for i in range(min(self.num_edges, self.max_edges)):
            obs[i] = self.link_usage[i]
        
        # Current edge index
        obs[-1] = self.current_edge_idx
        
        return obs
    
    def _calculate_routing(self):
        """Calculate routing based on current topology and traffic matrix."""
        # Get current traffic matrix
        tm = self.tm_list[self.current_tm_idx]
        
        # Reset link usage
        self.link_usage = np.zeros(self.num_edges)
        
        # Create a mapping from edge (u,v) to index
        edge_to_idx = {(u, v): i for i, (u, v) in enumerate(self.edge_list)}
        edge_to_idx.update({(v, u): i for i, (u, v) in enumerate(self.edge_list)})  # Both directions
        
        # For each source-destination pair, find shortest path and add to link usage
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                if src != dst and tm[src][dst] > 0:
                    try:
                        # Find shortest path
                        path = nx.shortest_path(self.G, source=src, target=dst, weight='weight')
                        
                        # Add traffic to each link in the path
                        for i in range(len(path) - 1):
                            u, v = path[i], path[i + 1]
                            if (u, v) in edge_to_idx:
                                edge_index = edge_to_idx[(u, v)]
                            else:
                                edge_index = edge_to_idx[(v, u)]
                            
                            self.link_usage[edge_index] += tm[src][dst]
                    except nx.NetworkXNoPath:
                        # No path exists, network is disconnected
                        pass
    
    def is_connected(self):
        """Check if the graph is connected."""
        return nx.is_connected(self.G)
    
    def get_effective_capacities(self):
        """Get effective capacities after applying scaling factors."""
        return [cap * factor for cap, factor in zip(self.link_capacity, self.link_factors)]
    
    def get_utilization(self):
        """Get link utilization as usage/effective_capacity."""
        effective_capacities = self.get_effective_capacities()
        return [usage / max(cap, 1e-10) for usage, cap in zip(self.link_usage, effective_capacities)]
