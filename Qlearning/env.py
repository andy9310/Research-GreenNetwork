import numpy as np
import networkx as nx
import random
import gymnasium as gym
from gymnasium import spaces

class NetworkEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, adj_matrix, edge_list, tm_list, node_props, num_nodes, link_capacity, seed=None, max_edges=10, random_edge_order=False):
        super(NetworkEnv, self).__init__()
        np.random.seed(seed)
        random.seed(seed)

        self.adj_matrix = np.array(adj_matrix)
        self.edge_list = edge_list
        self.tm_list = [np.array(tm) for tm in tm_list] # Ensure TMs are numpy arrays
        self.node_props = node_props
        self.num_nodes = num_nodes
        self.link_capacity = link_capacity
        self.num_edges = len(self.edge_list)
        self.max_edges = max_edges # Maximum possible edges
        self.random_edge_order = random_edge_order
        
        # For tracking edge importance (contribution to reward and violations)
        self.edge_rewards = np.zeros(self.num_edges, dtype=float)  # Rewards per edge
        self.edge_violations = {"isolation": np.zeros(self.num_edges, dtype=int),
                               "overloaded": np.zeros(self.num_edges, dtype=int)}
        self.edge_decisions = {"open": np.zeros(self.num_edges, dtype=int),
                              "close_success": np.zeros(self.num_edges, dtype=int),
                              "close_failure": np.zeros(self.num_edges, dtype=int)}
        self.edge_perm = np.arange(self.num_edges)  # Identity permutation by default
        self.inv_edge_perm = np.arange(self.num_edges)

        if self.num_edges > self.max_edges:
             raise ValueError(f"Number of edges ({self.num_edges}) exceeds max_edges ({self.max_edges})")

        # Define graph using networkx for path calculations
        self.graph = nx.Graph()
        for i in range(self.num_nodes):
            self.graph.add_node(i, **self.node_props.get(i, {}))
        for i, (u, v) in enumerate(self.edge_list):
            self.graph.add_edge(u, v, capacity=self.link_capacity, weight=1, id=i) # Add edge index 'id'

        # --- Action and Observation Space (Based on max_edges) ---
        # Action: Decide whether to close (0) or keep open (1) the current link
        self.action_space = spaces.Discrete(2)

        # Observation: [link_open_status (max_edges), link_usage (max_edges), current_edge_idx (1), current_usage_ratio (1)]
        # Pad with a distinct value, e.g., -1, if needed, or use masking carefully.
        # Here we assume 0 padding is acceptable for link status/usage.
        obs_dim = self.max_edges + self.max_edges + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Environment state variables
        self.current_edge_idx = 0
        self.link_open = np.ones(self.num_edges, dtype=int) # Initially all links are open
        self.usage = np.zeros(self.num_edges, dtype=float)
        self.traffic = None
        self.current_tm_idx = 0  # Default to first traffic matrix

        # Reward structure (can be loaded from config if needed)
        self.energy_unit_reward = 10 # Reward for closing one link without violation
        self.isolated_penalty = 1000
        self.overloaded_penalty = 10

    def _get_observation(self):
        """Constructs an improved observation vector that's more generalizable."""
        # Calculate normalized usage-to-capacity ratio instead of raw usage
        usage_to_capacity = np.zeros(self.num_edges, dtype=float)
        for i in range(self.num_edges):
            if self.link_open[i] == 1:  # Only for open links
                u, v = self.edge_list[i]
                capacity = self.graph[u][v]['capacity']
                if capacity > 0:
                    usage_to_capacity[i] = self.usage[i] / capacity  # Normalized [0-1+]
        
        # If using random edge order, rearrange link_open and usage_ratio according to permutation
        # So the agent sees edges in the order it expects to make decisions
        if self.random_edge_order:
            permuted_link_open = self.link_open[self.inv_edge_perm]
            permuted_usage_ratio = usage_to_capacity[self.inv_edge_perm]
            padded_link_open = np.pad(permuted_link_open, (0, self.max_edges - self.num_edges), 'constant', constant_values=0)
            padded_usage_ratio = np.pad(permuted_usage_ratio, (0, self.max_edges - self.num_edges), 'constant', constant_values=0.0)
        else:
            padded_link_open = np.pad(self.link_open, (0, self.max_edges - self.num_edges), 'constant', constant_values=0)
            padded_usage_ratio = np.pad(usage_to_capacity, (0, self.max_edges - self.num_edges), 'constant', constant_values=0.0)
        
        # Include the current edge's local features
        if self.current_edge_idx < self.num_edges:
            # Get the actual edge index after permutation
            actual_edge_idx = self.edge_perm[self.current_edge_idx]
            current_edge = self.edge_list[actual_edge_idx]
            current_capacity = self.graph[current_edge[0]][current_edge[1]]['capacity']
            current_usage = self.usage[actual_edge_idx]
            current_ratio = current_usage / current_capacity if current_capacity > 0 else 0
        else:
            current_ratio = 0
            
        # Create observation vector
        obs = np.concatenate([
            padded_link_open,
            padded_usage_ratio,
            [self.current_edge_idx / self.num_edges],  # Normalized index
            [current_ratio]  # Current edge's usage ratio
        ]).astype(np.float32)
        
        return obs

    def _get_action_mask(self):
        """Creates a mask for valid actions (always [True, True] since both close/open are valid)."""
        return np.array([True, True], dtype=bool)  # Both actions (close/open) are always valid

    def reset(self):
        """Resets the environment for a new episode."""
        # Use the current traffic matrix index if available
        # This allows for cycling through matrices during training
        if hasattr(self, 'current_tm_idx') and 0 <= self.current_tm_idx < len(self.tm_list):
            self.traffic = self.tm_list[self.current_tm_idx].copy()
        else:
            # Fall back to random selection if index is invalid
            self.traffic = random.choice(self.tm_list).copy() # Use a copy

        # Generate random permutation of edge indices if enabled
        if self.random_edge_order:
            self.edge_perm = np.random.permutation(self.num_edges)
            self.inv_edge_perm = np.argsort(self.edge_perm)
        else:
            self.edge_perm = np.arange(self.num_edges)  # Identity permutation
            self.inv_edge_perm = np.arange(self.num_edges)
            
        # Reset state variables
        self.current_edge_idx = 0
        self.link_open = np.ones(self.num_edges, dtype=int) # Start with all links open
        self.usage = np.zeros(self.num_edges, dtype=float)
        self._update_link_usage() # Calculate initial usage

        obs = self._get_observation()
        action_mask = self._get_action_mask()
        info = {'action_mask': action_mask}
        
        # Return 5 values: observation, reward, done, truncated, info
        return obs, 0, False, False, info

    def step(self, action):
        """Executes one time step within the environment."""
        reward = 0
        done = False
        info = {}
        
        # Check if we've reached the end of the episode (all edges processed)
        if self.current_edge_idx >= self.num_edges:
            print(f"Warning: current_edge_idx {self.current_edge_idx} >= num_edges {self.num_edges}")
            done = True
            obs = self._get_observation()
            action_mask = self._get_action_mask()
            info = {'action_mask': action_mask, 'violation': 'episode_complete'}
            # Return 5 values: observation, reward, done, truncated, info
            return obs, reward, done, False, info

        # Action should be 0 (close) or 1 (keep open)
        if action not in [0, 1]:
            print(f"Warning: Invalid action {action}. Must be 0 (close) or 1 (keep open). Defaulting to 1.")
            action = 1  # Default to keeping the link open rather than raising an error

        # Action 0 means try to close the current link, 1 means keep it open
        chosen_action_for_current_edge = action  # 0=close, 1=keep open
        
        # Map current_edge_idx to actual edge index using permutation
        permuted_edge_idx = self.edge_perm[self.current_edge_idx]
        edge_to_modify = permuted_edge_idx  # This is the actual edge in edge_list to modify
        
        # Additional safety check
        if edge_to_modify >= len(self.link_open):
            print(f"Error: edge_to_modify {edge_to_modify} >= link_open length {len(self.link_open)}")
            done = True
            obs = self._get_observation()
            action_mask = self._get_action_mask()
            info = {'action_mask': action_mask, 'violation': 'edge_index_out_of_bounds'}
            # Return 5 values: observation, reward, done, truncated, info
            return obs, -1, done, False, info

        if chosen_action_for_current_edge == 0: # Try to close the link
            # Tentatively close the link
            original_state = self.link_open[edge_to_modify]
            self.link_open[edge_to_modify] = 0

            # Reroute traffic and check for violations
            routing_successful, G_open = self._update_link_usage()
            isolated, overloaded, num_overloaded = self._check_violations(routing_successful, G_open)

            if isolated or overloaded:
                # Calculate negative reward for violations
                reward = 0
                if isolated:
                    reward = -self.isolated_penalty
                    # Track edge violation for importance analysis
                    self.edge_violations["isolation"][edge_to_modify] += 1
                elif overloaded:
                    reward = -self.overloaded_penalty * num_overloaded
                    # Track edge violation for importance analysis
                    self.edge_violations["overloaded"][edge_to_modify] += 1
                info['violation'] = 'isolated' if isolated else 'overloaded'
                done = True  # Terminate episode on violation
                # Track failed closing attempt
                self.edge_decisions["close_failure"][edge_to_modify] += 1
            else:
                # No violation - success! Keep link closed and give reward.
                reward = self.energy_unit_reward
                # Track reward contribution for importance analysis
                self.edge_rewards[edge_to_modify] += reward
                # Track successful closing
                self.edge_decisions["close_success"][edge_to_modify] += 1

        else: # Action 1: Keep the link open
            # No change in link state, no energy reward, no penalty
            self.link_open[edge_to_modify] = 1 # Ensure it stays open
            
            # IMPORTANT FIX: Check for violations even when keeping links open
            # This is crucial to detect cascading effects of sequential decisions
            routing_successful, G_open = self._update_link_usage()
            isolated, overloaded, num_overloaded = self._check_violations(routing_successful, G_open)
            
            if isolated or overloaded:
                # Violation occurred - penalize and terminate episode
                reward = 0
                if isolated:
                    reward = -self.isolated_penalty
                    # Track edge violation for importance analysis
                    self.edge_violations["isolation"][edge_to_modify] += 1
                elif overloaded:
                    reward = -self.overloaded_penalty * num_overloaded
                    # Track edge violation for importance analysis
                    self.edge_violations["overloaded"][edge_to_modify] += 1
                info['violation'] = 'isolated' if isolated else 'overloaded'
                info['num_overloaded'] = num_overloaded if overloaded else 0
                done = True  # Terminate episode on violation
            else:
                # No violations
                reward = 0
                info['violation'] = None
            
            # Track decision to keep open
            self.edge_decisions["open"][edge_to_modify] += 1

        # Move to the next edge decision
        self.current_edge_idx += 1
        
        # Add real edge index to info
        info['real_edge_idx'] = edge_to_modify

        # --- Check if episode is done --- 
        if self.current_edge_idx >= self.num_edges:
            # reward += 10 # give a final reward
            done = True
            # No final reward, rewards are per-step based on closing links

        obs = self._get_observation()
        action_mask = self._get_action_mask()
        # Always ensure 'violation' is present in info
        if 'violation' not in info:
            info['violation'] = None
        info['action_mask'] = action_mask # Add mask to info

        # Return 5 values: observation, reward, done, truncated, info
        return obs, reward, done, False, info


    def _update_link_usage(self):
        """Calculates traffic flow based on currently open links."""
        self.usage = np.zeros(self.num_edges, dtype=float)
        G_open = nx.Graph()
        G_open.add_nodes_from(range(self.num_nodes))
        open_edge_indices = []
        for i, (u, v) in enumerate(self.edge_list):
            if self.link_open[i] == 1:
                G_open.add_edge(u, v, weight=1, id=i)
                open_edge_indices.append(i)

        routing_successful = True
        if not nx.is_connected(G_open) and G_open.number_of_nodes() > 0:
             # Check if the disconnect matters for the current TM
             nodes_with_demand = set(np.where(self.traffic > 0)[0]) | set(np.where(self.traffic > 0)[1])
             if nodes_with_demand:
                 # Find the component containing an arbitrary node with demand (if any)
                 start_node = next(iter(nodes_with_demand))
                 if start_node in G_open:
                     component_with_start = nx.node_connected_component(G_open, start_node)
                     # If not all demanding nodes are in the same component, routing will fail for some pairs
                     if not nodes_with_demand.issubset(component_with_start):
                         routing_successful = False
                 else: # Start node isn't even in the open graph (shouldn't happen if start node has demand)
                     routing_successful = False

        if routing_successful:
            try:
                for src in range(self.num_nodes):
                    for dst in range(self.num_nodes):
                        if src != dst and self.traffic[src, dst] > 0:
                            path = nx.shortest_path(G_open, source=src, target=dst, weight='weight')
                            path_edges = zip(path[:-1], path[1:])
                            for u, v in path_edges:
                                # Find the edge id corresponding to (u,v) or (v,u)
                                edge_id = G_open[u][v]['id']
                                self.usage[edge_id] += self.traffic[src, dst]
            except nx.NetworkXNoPath:
                # This exception means a path doesn't exist between required nodes
                routing_successful = False
                # self.usage remains potentially incomplete, but _check_violations will catch the isolation
            except Exception as e:
                 print(f"Error during routing: {e}")
                 routing_successful = False

        return routing_successful, G_open


    def _check_violations(self, routing_successful, G_open, epsilon=0.0):
        """Check for network violations: isolated nodes and overloaded links.
        
        Args:
            routing_successful: Boolean indicating if routing was successful
            G_open: NetworkX graph with only open links
            epsilon: Tolerance for capacity violation (default: 0.02, 2%)
                     This provides a safety margin to prevent false violations due to floating point errors
        """
        # Check if any nodes are isolated (when they shouldn't be)
        isolated = not routing_successful
        
        # Check if any links are overloaded (usage > capacity)
        overloaded = False
        num_overloaded = 0
        for i, usage in enumerate(self.usage):
            if self.link_open[i] == 1:  # Only check open links
                u, v = self.edge_list[i]
                capacity = self.graph[u][v]['capacity']
                
                # Calculate percent of capacity used
                percent_used = usage / capacity * 100.0
                
                # Add epsilon tolerance to prevent tiny floating point issues
                # Only flag as overloaded if usage clearly exceeds capacity beyond floating point error range
                if usage > capacity * (1.0 + epsilon):
                    # This is a significant overload, not just a rounding error
                    overloaded = True
                    num_overloaded += 1
        
        return isolated, overloaded, num_overloaded
        
    def get_edge_importance_data(self):
        """Return collected data about edge importance."""
        return {
            "edge_rewards": self.edge_rewards.tolist(),
            "edge_violations": {
                "isolation": self.edge_violations["isolation"].tolist(),
                "overloaded": self.edge_violations["overloaded"].tolist()
            },
            "edge_decisions": {
                "open": self.edge_decisions["open"].tolist(),
                "close_success": self.edge_decisions["close_success"].tolist(),
                "close_failure": self.edge_decisions["close_failure"].tolist()
            },
            "edge_list": self.edge_list,
            "num_edges": self.num_edges
        }
        
    def save_edge_importance_data(self, filepath):
        """Save edge importance data to a file."""
        import json
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare data
        data = self.get_edge_importance_data()
        
        # Convert edge_list to serializable format
        data["edge_list"] = [list(edge) for edge in data["edge_list"]]
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
        
    def render(self, mode='human', close=False):
        pass # Optional: Implement visualization if needed

    def close(self):
        pass # Optional: Cleanup resources