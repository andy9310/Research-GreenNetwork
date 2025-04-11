import numpy as np
import networkx as nx
import random
import gymnasium as gym
from gymnasium import spaces

class NetworkEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, adj_matrix, edge_list, tm_list, node_props, num_nodes, link_capacity, seed=None, max_edges=10):
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

        # Observation: [link_open_status (max_edges), link_usage (max_edges), current_edge_idx (1)]
        # Pad with a distinct value, e.g., -1, if needed, or use masking carefully.
        # Here we assume 0 padding is acceptable for link status/usage.
        obs_dim = self.max_edges + self.max_edges + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Environment state variables
        self.current_edge_idx = 0
        self.link_open = np.ones(self.num_edges, dtype=int) # Initially all links are open
        self.usage = np.zeros(self.num_edges, dtype=float)
        self.traffic = None
        self.current_tm_idx = 0  # Default to first traffic matrix

        # Reward structure (can be loaded from config if needed)
        self.energy_unit_reward = 10  # Reward for successfully closing a link
        self.overloaded_penalty = 50  # Penalty for overloaded links
        self.isolated_penalty = 1000   # Penalty for isolated nodes

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        # Optional seed setting
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Reset internal state
        self.current_edge_idx = 0
        self.link_open = np.ones(self.num_edges, dtype=int) # All links start open
        self.usage = np.zeros(self.num_edges, dtype=float)

        # Recalculate link usage for the current traffic matrix
        self._update_link_usage()

        # Generate initial observation
        obs = self._get_observation()
        action_mask = self._get_action_mask()
        info = {'action_mask': action_mask}
        
        # Return 5 values: observation, reward, done, truncated, info
        return obs, info

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
        edge_to_modify = self.current_edge_idx
        
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
                # Violation occurred - penalize and revert the change
                self.link_open[edge_to_modify] = original_state # Revert
                self._update_link_usage() # Recalculate usage with link open again
                reward = 0 # No reward for trying to close if it causes violation
                if isolated:
                    reward = -self.isolated_penalty
                elif overloaded:
                    reward = -self.overloaded_penalty * num_overloaded
                info['violation'] = 'isolated' if isolated else 'overloaded'
                done = True  # Terminate episode on violation
            else:
                # No violation - success! Keep link closed and give reward.
                reward = self.energy_unit_reward
                info['violation'] = None
        else: # Action 1: Keep the link open
            # No change in link state, no energy reward, no penalty
            self.link_open[edge_to_modify] = 1 # Ensure it stays open
            reward = 0
            info['violation'] = None
            # No need to re-route if link state didn't change from open to open

        # Move to the next edge decision
        self.current_edge_idx += 1

        # --- Check if episode is done --- 
        if self.current_edge_idx >= self.num_edges:
            # reward += 10 # give a final reward
            done = True
            # No final reward, rewards are per-step based on closing links

        obs = self._get_observation()
        action_mask = self._get_action_mask()
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
             components = list(nx.connected_components(G_open))
             # This is a tentative solution that may not be accurate for all network topologies
             # If any two nodes with non-zero traffic are disconnected, it's a violation
             traffic_matrix = self.tm_list[self.current_tm_idx]
             for component in components:
                 for i in range(self.num_nodes):
                     if i not in component:
                         for j in component:
                             if traffic_matrix[i][j] > 0 or traffic_matrix[j][i] > 0:
                                 # There's traffic between disconnected components
                                 return False, G_open
        
        # For each source-destination pair with traffic
        traffic_matrix = self.tm_list[self.current_tm_idx]
        
        # Function to increment usage for a path
        def increment_path_usage(path, traffic):
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                # Find edge index in original graph
                for edge_idx, (edge_u, edge_v) in enumerate(self.edge_list):
                    if (u == edge_u and v == edge_v) or (u == edge_v and v == edge_u):
                        # Add traffic to usage
                        self.usage[edge_idx] += traffic
                        break
        
        # Route traffic between all node pairs with demand
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                if src != dst and traffic_matrix[src][dst] > 0:
                    try:
                        # Find shortest path between src and dst in G_open
                        path = nx.shortest_path(G_open, source=src, target=dst, weight='weight')
                        # Increment usage along the path
                        increment_path_usage(path, traffic_matrix[src][dst])
                    except nx.NetworkXNoPath:
                        # No path found between src and dst
                        routing_successful = False
        
        return routing_successful, G_open
        
    def _check_violations(self, routing_successful, G_open):
        """Check for isolated nodes or overloaded links."""
        # Check for network isolation
        if not routing_successful or not nx.is_connected(G_open):
            # For now, any disconnection is considered a violation
            isolated = True
        else:
            isolated = False
        
        # Check for link overload
        overloaded = False
        num_overloaded = 0
        for i, usage in enumerate(self.usage):
            if usage > self.link_capacity and self.link_open[i] == 1:
                overloaded = True
                num_overloaded += 1
        
        return isolated, overloaded, num_overloaded
                
    def _get_observation(self):
        """Get the current state observation."""
        # Pad arrays if necessary to match observation space dimension
        padded_link_open = np.pad(self.link_open, (0, self.max_edges - self.num_edges), 'constant', constant_values=0) # Pad with 0
        padded_usage = np.pad(self.usage, (0, self.max_edges - self.num_edges), 'constant', constant_values=0) # Pad with 0
        
        # Normalize usage by link capacity
        normalized_usage = padded_usage / self.link_capacity if self.link_capacity > 0 else padded_usage
        
        # Combine components into single observation array
        obs = np.concatenate([padded_link_open, normalized_usage, [self.current_edge_idx]])
        
        return obs.astype(np.float32)
    
    def _get_action_mask(self):
        """Get a binary mask of valid actions: [0] can close, [1] can keep open."""
        # Both actions are always valid in this implementation
        return np.ones(self.action_space.n, dtype=np.int8)
    
    def render(self, mode='human'):
        """Renders the environment."""
        if mode != 'human':
            raise NotImplementedError(f"Render mode {mode} is not implemented")
        
        # Print current state
        print(f"Current edge index: {self.current_edge_idx}")
        print(f"Link open status: {self.link_open}")
        print(f"Link usage: {self.usage}")
        
        # We could visualize the network using networkx and matplotlib
        # This would be implemented if needed
        return 
        
    def close(self):
        """Clean up resources."""
        pass # No specific resources to clean up
