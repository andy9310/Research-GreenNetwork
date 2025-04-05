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
        # Action: Choose which link's status to decide (0 to max_edges-1)
        self.action_space = spaces.Discrete(self.max_edges)

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

        # Reward structure (can be loaded from config if needed)
        self.energy_unit_reward = 2 # Reward for closing one link without violation
        self.isolated_penalty = 100
        self.overloaded_penalty = 50

    def _get_observation(self):
        """Constructs the observation vector, padded to max_edges."""
        # Pad link_open and usage arrays
        padded_link_open = np.pad(self.link_open, (0, self.max_edges - self.num_edges), 'constant', constant_values=0) # Pad with 0
        padded_usage = np.pad(self.usage, (0, self.max_edges - self.num_edges), 'constant', constant_values=0.0)

        obs = np.concatenate([
            padded_link_open,
            padded_usage,
            [self.current_edge_idx]
        ]).astype(np.float32)
        return obs

    def _get_action_mask(self):
        """Creates a mask for valid actions (edges in the current topology)."""
        mask = np.zeros(self.max_edges, dtype=bool)
        mask[:self.num_edges] = True # Only the first num_edges actions are valid
        return mask

    def reset(self):
        """Resets the environment for a new episode."""
        # Choose a traffic matrix for the episode
        self.traffic = random.choice(self.tm_list).copy() # Use a copy

        # Reset state variables
        self.current_edge_idx = 0
        self.link_open = np.ones(self.num_edges, dtype=int) # Start with all links open
        self.usage = np.zeros(self.num_edges, dtype=float)
        self._update_link_usage() # Calculate initial usage

        obs = self._get_observation()
        action_mask = self._get_action_mask()
        info = {'action_mask': action_mask}

        return obs, info # Return obs and info dict containing mask

    def step(self, action):
        """Executes one time step within the environment."""
        reward = 0
        done = False
        info = {}

        # The action directly corresponds to the edge index whose state we are deciding
        # Note: The agent should only select actions where mask is True.
        # We assume the agent respects the mask, so action < self.num_edges
        if action >= self.num_edges:
            # This case should ideally not happen if the agent uses the mask correctly
            # Penalize heavily or raise an error if an invalid action is chosen
            reward = -200 # Heavy penalty for trying to act on a non-existent edge
            print(f"Warning: Agent chose invalid action {action} >= num_edges {self.num_edges}. Masking failed? Current Edge Idx: {self.current_edge_idx}")
            # State doesn't change, but we move to the next decision point
            self.current_edge_idx += 1

        else:
            edge_to_consider = action # The action *is* the edge index

            # For simplicity in this step-by-step decision, let's assume the agent decides
            # the state of link `edge_to_consider`. Let's say 1 means keep open, 0 means try to close.
            # A more standard RL approach would have action 0 = 'close edge k', action 1 = 'keep edge k open',
            # where k is self.current_edge_idx. Let's adapt to that.

            # --- Corrected Logic: Action decides state of current_edge_idx --- 
            chosen_action_for_current_edge = action # Let's redefine: action 0=close, 1=keep open
            edge_to_modify = self.current_edge_idx

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
                        reward -= self.isolated_penalty
                    if overloaded:
                        reward -= self.overloaded_penalty * num_overloaded
                    info['violation'] = 'isolated' if isolated else 'overloaded'
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
            done = True
            # No final reward, rewards are per-step based on closing links

        obs = self._get_observation()
        action_mask = self._get_action_mask()
        info['action_mask'] = action_mask # Add mask to info

        return obs, reward, done, info # Return obs, reward, done, info


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


    def _check_violations(self, routing_successful, G_open):
        """Checks for node isolation and link overload.

        Args:
            routing_successful (bool): Whether routing could be performed.
            G_open (nx.Graph): The graph containing only currently open links.

        Returns:
            tuple: (isolated, overloaded, num_overloaded)
        """
        isolated = False
        overloaded = False
        num_overloaded = 0

        # 1. Check for Node Isolation
        if not routing_successful:
            isolated = True
        # Optional: Add check for disconnected G_open even if routing_successful was True initially
        # This can happen if demands are zero, but the graph is still split.
        # elif G_open.number_of_nodes() > 0 and not nx.is_connected(G_open):
        #    isolated = True # Consider any disconnection an isolation for simplicity

        # 2. Check for Link Overload
        for i, (u, v) in enumerate(self.edge_list):
            if self.link_open[i] == 1: # Only check open links
                capacity = self.graph[u][v]['capacity']
                if capacity > 0 and self.usage[i] > capacity:
                    overloaded = True
                    num_overloaded += 1

        return isolated, overloaded, num_overloaded

    def render(self, mode='human', close=False):
        pass # Optional: Implement visualization if needed

    def close(self):
        pass # Optional: Cleanup resources