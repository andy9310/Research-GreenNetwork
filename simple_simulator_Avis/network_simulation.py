import networkx as nx
from itertools import permutations
from collections import defaultdict
import numpy as np
import math

class NetworkEnvironment:
    def __init__(self, switches, hosts, links, port_info, max_ports_per_linecard, fixed_throughput, target_switches):
        # Initialize the network environment
        self.switches = switches
        self.hosts = hosts
        self.links = links
        self.port_info = port_info
        self.max_ports_per_linecard = max_ports_per_linecard
        self.fixed_throughput = fixed_throughput
        self.target_switches = target_switches
        self.num_actions = 2**11
        # Create network and initial states
        self.network = self.create_network_with_weights()
        self.linecard_mapping = self.create_linecard()
        self.linecard_status = self.initialize_linecard_status()
        self.traffic_volume_dict = self.create_traffic_volume_dict()
        self.shortest_paths = self.find_shortest_paths()
        self.flows_per_switch = self.count_flows_per_switch()
        self.flows_per_link = self.count_flows_per_link()
        self.switch_degrees = self.calculate_switch_degrees()
        
        # Initial states
        self.state_linecard_status = self.get_linecard_status_vector()
        self.state_switch_status = self.get_switch_status_vector()
        self.state_switch_degree = self.get_switch_degree_vector()
        self.state = self.state_linecard_status + self.state_switch_status + self.state_switch_degree

        # Track initial conditions
        self.target_switches = [4, 9, 10, 15]
        self.steps = 0  # Step counter

        # self.action_map = [np.array([1, 1, 1]), np.array([1, 1, 0]), np.array([1, 0, 0]), np.array([0, 0, 0]), np.array([0, 1, 1]), np.array([0, 1, 0]),np.array([0, 0, 0]), np.array([0, 0, 1])]
        self.action_map = self._create_action_map()

    def _create_action_map(self):
        """Map indices (0-7) to their corresponding binary action vectors."""
        actions = []
        for i in range(self.num_actions):
            actions.append(np.array([int(x) for x in f"{i:011b}"]))  # Convert index to binary
        # for idx, action in enumerate(actions):
        #     print(f"Action Index: {idx}, Action Vector: {action}")  # Debugging print
        return actions
        
    def reset(self):
        """
        Reset the environment to its initial state.
        """
        # Reinitialize linecard statuses
        self.linecard_status = self.initialize_linecard_status()
        
        # Recreate the network graph
        self.network = self.create_network_with_weights()

        # Recalculate all metrics and state components
        self.traffic_volume_dict = self.create_traffic_volume_dict()
        self.shortest_paths = self.find_shortest_paths()
        self.flows_per_switch = self.count_flows_per_switch()
        self.flows_per_link = self.count_flows_per_link()
        self.switch_degrees = self.calculate_switch_degrees()

        # Reinitialize the state
        self.state_linecard_status = self.get_linecard_status_vector()
        self.state_switch_status = self.get_switch_status_vector()
        self.state_switch_degree = self.get_switch_degree_vector()
        self.state = self.state_linecard_status + self.state_switch_status + self.state_switch_degree

        # Reset step counter
        self.steps = 0

        return self.state


    def update_network_on_linecard_status(self, changes):
        # print(changes)
        all_modified_links = []
        for switch, linecard_id, desired_status in changes:
            current_status = self.linecard_status.get((switch, linecard_id))

            # If the desired status matches the current status, do nothing
            if current_status == desired_status:
                print(f"Linecard {linecard_id} on Switch {switch} is already {'on' if desired_status else 'off'}.")
                continue

            # Update the status
            self.linecard_status[(switch, linecard_id)] = desired_status

            # Get affected ports
            if switch in self.linecard_mapping and linecard_id in self.linecard_mapping[switch]:
                affected_ports = self.linecard_mapping[switch][linecard_id]
                modified_links = []

                for (switch_from, switch_to), port in self.port_info.items():
                    if switch_from == switch and port in affected_ports:
                        if desired_status:  # Turning the linecard ON
                            # Add the link back if it doesn't already exist
                            if not self.network.has_edge(switch_from, switch_to):
                                self.network.add_edge(switch_from, switch_to, weight=1 / self.links[(switch_from, switch_to)])
                                modified_links.append((switch_from, switch_to))
                                # print(f"Added link: {switch_from} -> {switch_to}")

                            # Add duplex link if not present
                            if not self.network.has_edge(switch_to, switch_from):
                                self.network.add_edge(switch_to, switch_from, weight=1 / self.links[(switch_to, switch_from)])
                                modified_links.append((switch_to, switch_from))
                                # print(f"Added duplex link: {switch_to} -> {switch_from}")
                        
                        else:  # Turning the linecard OFF
                            # Remove the link if it exists
                            if self.network.has_edge(switch_from, switch_to):
                                self.network.remove_edge(switch_from, switch_to)
                                modified_links.append((switch_from, switch_to))
                                # print(f"Removed link: {switch_from} -> {switch_to}")

                            # Remove duplex link if it exists
                            if self.network.has_edge(switch_to, switch_from):
                                self.network.remove_edge(switch_to, switch_from)
                                modified_links.append((switch_to, switch_from))
                                # print(f"Removed duplex link: {switch_to} -> {switch_from}")

                # print(f"Links modified for Switch {switch}, Linecard {linecard_id}: {modified_links}")
                all_modified_links.extend(modified_links)

        # print(f"Total links modified: {all_modified_links}")

        # Refresh the environment after topology changes
        self.refresh_environment()
    

    def create_network_with_weights(self):
        G = nx.DiGraph()
        for (u, v), bandwidth in self.links.items():
            G.add_edge(u, v, weight=1 / bandwidth)
        for host, switch in zip(self.hosts, self.switches):
            G.add_edge(host, switch, weight=1 / 10000)
            G.add_edge(switch, host, weight=1 / 10000)
        return G
    


    def create_linecard(self):
        linecard = defaultdict(lambda: defaultdict(list))
        ports_per_switch = defaultdict(set)
        for (switch_from, switch_to), port in self.port_info.items():
            ports_per_switch[switch_from].add(port)
        for switch, ports in ports_per_switch.items():
            ports = sorted(ports)
            for i, port in enumerate(ports):
                linecard_id = i // self.max_ports_per_linecard + 1
                linecard[switch][linecard_id].append(port)
        

        # change the switch we control
        result = {switch: dict(linecards) for switch, linecards in linecard.items()}
        filtered_data = {k: v for k, v in result.items() if k in self.target_switches}
        
        return filtered_data
    



    
    def initialize_linecard_status(self):
        return {(switch, linecard_id): True for switch, linecards in self.linecard_mapping.items() for linecard_id in linecards}
    
    def create_traffic_volume_dict(self):
        return {(host1, host2): self.fixed_throughput for host1, host2 in permutations(self.hosts, 2)}
    
    def find_shortest_paths(self):
        shortest_paths = {}
        for host1, host2 in permutations(self.hosts, 2):
            try:
                shortest_paths[(host1, host2)] = nx.shortest_path(self.network, source=host1, target=host2, weight="weight")
            except nx.NetworkXNoPath:
                shortest_paths[(host1, host2)] = None
        return shortest_paths
    
    def count_flows_per_switch(self):
        flow_count = defaultdict(int)
        for path in self.shortest_paths.values():
            if path:
                for node in path:
                    if isinstance(node, int):
                        flow_count[node] += 1

        for switch in self.switches:
            if switch not in flow_count:
                flow_count[switch] = 0
        return dict(flow_count)
    
    def count_flows_per_link(self):
        flow_count = defaultdict(int)
        for path in self.shortest_paths.values():
            if path:
                for i in range(len(path) - 1):
                    if path[i] not in self.hosts and path[i + 1] not in self.hosts:
                        flow_count[(path[i], path[i + 1])] += 1
        return dict(flow_count)
    
    def calculate_switch_degrees(self):
        return {switch: (self.network.in_degree(switch), self.network.out_degree(switch),
                         self.network.in_degree(switch) + self.network.out_degree(switch)) for switch in self.switches}
    
    def get_linecard_status_vector(self):
        sorted_keys = sorted(self.linecard_status.keys())
        return [1 if self.linecard_status[key] else 0 for key in sorted_keys]
    
    def get_switch_status_vector(self):
        return self.vectorize_flows(self.flows_per_switch, normalize=True)
    
    def get_switch_degree_vector(self):
        return self.vectorize_degrees(self.switch_degrees, normalize=True)
    
    @staticmethod
    def vectorize_flows(flows_per_switch, normalize=False):
        sorted_keys = sorted(flows_per_switch.keys())
        vector = [flows_per_switch[key] for key in sorted_keys]
        if normalize:
            max_value = max(vector)
            if max_value > 0:
                vector = [v / max_value for v in vector]
        return vector
    
    @staticmethod
    def vectorize_degrees(switch_degrees, normalize=False):
        sorted_keys = sorted(switch_degrees.keys())
        vector = []
        for key in sorted_keys:
            vector.extend(switch_degrees[key])
        if normalize:
            max_value = max(vector)
            if max_value > 0:
                vector = [v / max_value for v in vector]
        return vector
    

    def refresh_environment(self):
        """Refresh all computed properties of the environment."""
        self.traffic_volume_dict = self.create_traffic_volume_dict()
        self.shortest_paths = self.find_shortest_paths()
        self.flows_per_switch = self.count_flows_per_switch()
        self.flows_per_link = self.count_flows_per_link()
        self.switch_degrees = self.calculate_switch_degrees()
        self.state_linecard_status = self.get_linecard_status_vector()
        self.state_switch_status = self.get_switch_status_vector()
        self.state_switch_degree = self.get_switch_degree_vector()
        self.state = self.state_linecard_status + self.state_switch_status + self.state_switch_degree


    @property
    def action_size(self):
        """Calculate total number of linecards in target switches."""
        return sum(len(linecards) for switch, linecards in self.linecard_mapping.items() if switch in self.target_switches)
    


    def apply_action(self, action_index):
        """
        Apply the binary action vector to update linecard statuses.
        Args:
            action: Binary vector representing the desired linecard statuses.
        """
        # print("apply action", action)

        # Flatten the action array to ensure it's 1D
        # print("====================")
        # print(action)
        # print("====================")
        # action = action[1]
        
        action = self.action_map[action_index]
        # Flatten the linecard mapping for target switches
        linecards = [(switch, linecard_id) for switch in self.target_switches
                    for linecard_id in sorted(self.linecard_mapping[switch].keys())]

        # Apply the action by comparing with current statuses
        changes = []
        
        for idx, (switch, linecard_id) in enumerate(linecards):
            current_status = self.linecard_status[(switch, linecard_id)]
            desired_status = bool(action[idx])  # Ensure action[idx] is a scalar
            if current_status != desired_status:
                changes.append((switch, linecard_id, desired_status))

        # Update network based on changes
        self.update_network_on_linecard_status(changes)


    def step(self, action_index):
        """
        Execute the given action and update the environment.
        Args:
            action: Binary vector representing the desired linecard statuses.
        """
        

        # Apply the action
        self.apply_action(action_index)
        a1 = 0.5  # Increase emphasis on energy savings
        a2 = 0.4  # Keep connectivity penalty significant
        a3 = 0.1  # Reduce emphasis on bandwidth penalty (if it's rare)
       
        # Reward Function
        basic_energy = (542 + 260) * len(self.target_switches)              # four switches
        card_energy =  np.sum(self.action_map[action_index] == 1) * 50      
        energy = basic_energy +  card_energy
        # print("action",action)
      
        max_possible_card_energy = self.action_size * 50  # All linecards on
        max_energy = basic_energy + max_possible_card_energy

        # print(energy)
        # print(self.action_map[action_index])
        # print(len(self.target_switches)   )
        # connectivity_penalty = 0
        # if not nx.is_connected(self.network.to_undirected()):
        #     connectivity_penalty = 500
        # bandwidth_exceed_penalty = 0 
        # for (u, v), bandwidth in self.links.items():
        #     link_throughput = self.flows_per_link.get((u, v), 0) * self.fixed_throughput
        #     # print(link_throughput)
        #     if link_throughput > bandwidth:
        #         bandwidth_exceed_penalty += (link_throughput - bandwidth) * 0.1  # Example penalty factor
        # if bandwidth_exceed_penalty > 0:
        #     bandwidth_exceed_penalty = 500

        # energy_scaled = energy / max_energy  # Scale to range [0, 1]
        # connectivity_penalty_scaled = connectivity_penalty / 500  # Scale to [0, 1]
        # bandwidth_exceed_penalty_scaled = bandwidth_exceed_penalty / 500  # Scale to [0, 1]



        # Penalties
        connectivity_penalty = 0
        if not nx.is_connected(self.network.to_undirected()):
            connectivity_penalty = 3000  # Larger penalty for disconnection

        bandwidth_exceed_penalty = 0
        for (u, v), bandwidth in self.links.items():
            link_throughput = self.flows_per_link.get((u, v), 0) * self.fixed_throughput
            if link_throughput > bandwidth:
                bandwidth_exceed_penalty += (link_throughput - bandwidth) * 0.1
        bandwidth_exceed_penalty = np.log1p(bandwidth_exceed_penalty) * 300  # Smooth scaling
        # bandwidth_exceed_penalty = bandwidth_exceed_penalty * 100
        # Energy Scaling
        # energy_scaled = (max_energy - energy) / max_energy  # Higher is better
        # connectivity_penalty_scaled = connectivity_penalty / 500
        # bandwidth_exceed_penalty_scaled = bandwidth_exceed_penalty / 500

        # reward = a1 * energy_scaled - (a2 * connectivity_penalty_scaled + a3 * bandwidth_exceed_penalty_scaled)
        # print(energy, connectivity_penalty, bandwidth_exceed_penalty)
        reward = (-1)* (energy + connectivity_penalty + bandwidth_exceed_penalty)
        reward = reward/10000

        
        # basic_energy = (542 + 260) * len(self.target_switches)              # four switches
        # card_energy = np.sum(action == 1) * 50
        # energy = basic_energy +  card_energy
        # connectivity_penalty = 1000
        # bandwidth_exceed_penalty = 1000
        # if connectivity_penalty > 0 or bandwidth_exceed_penalty > 0:
        #     reward = (-1)* (connectivity_penalty + bandwidth_exceed_penalty)
        
        # else: 
        #     reward = (-1)* (energy)
        
        # reward = (-1) * (a1 * energy_scaled + a2 * connectivity_penalty_scaled + a3 * bandwidth_exceed_penalty_scaled)
      
        


        if self.steps + 1 >= 100:
            done = True
        else:
            done = False

        # Update state
        self.state = self.state_linecard_status + self.state_switch_status + self.state_switch_degree
        # print("self.state_linecard_status", len(self.state_linecard_status))
        # print("self.state_switch_status", len(self.state_switch_status))
        # print("self.state_switch_degree", len(self.state_switch_degree))

        flow_data_rates = {}


        for flow, path in self.shortest_paths.items():
            # Check if the path has at least two elements to form a valid link.
            if not path or not isinstance(path, list) or len(path) < 2:
                # print(f"Flow session {flow} does not contain a valid path; skipping...")
                flow_data_rates[flow] = 0
                continue

            link_shares = []
            for i in range(len(path) - 1):
                # Process only adjacent elements that are integers (switch identifiers)
                if isinstance(path[i], int) and isinstance(path[i+1], int):
                    link = (path[i], path[i+1])
                    # Try both possible orientations for the link.
                    if link in self.flows_per_link:
                        count = self.flows_per_link[link]
                    elif (link[1], link[0]) in self.flows_per_link:
                        count = self.flows_per_link[(link[1], link[0])]
                    else:
                        # print(f"Warning: Link {link} not found in flows_per_link!")
                        continue
                        
                    # Scale down only if the count is greater than 70; otherwise, use a full share (1).
                    if count > 70:
                        share = (70 / count) * 100
                    else:
                        share = 100
                
                    link_shares.append(share)

            # If no valid link shares were found, mark the effective rate as None.
            if link_shares:
                effective_rate = min(link_shares)
            else:
                effective_rate = 100
            flow_data_rates[flow] = effective_rate

            # Display the effective data rate for each flow session.
            # for flow, rate in flow_data_rates.items():
                # print(f"Flow session {flow} has an effective data rate: {rate}")
            
        # Calculate and display the average effective data rate across all flow sessions.
        # print(flow_data_rates)
        # valid_rates = [rate for rate in flow_data_rates.values() if rate is not None]
        # if valid_rates:
        #     temp = sum(valid_rates) / len(valid_rates)
        #     temp = temp * 100
        #     average_rate = temp
            # print(f"\nAverage effective data rate for all sessions: {average_rate}")
            # print("==") 
        # else:
        #     pass
            # print("\nNo valid flow sessions found to compute an average data rate.")    

        total_value = sum(flow_data_rates.values())
        num_entries = len(flow_data_rates)

        # print(flow_data_rates)
        # print(self.traffic_volume_dict)

        # Compute the average value (make sure to avoid division by zero)
        throughput = total_value / num_entries if num_entries > 0 else 0

            
            

        self.steps += 1

        return self.state, reward, done, {"energy": energy, "connectivity_penalty": connectivity_penalty,
                                      "bandwidth_exceed_penalty": bandwidth_exceed_penalty}, throughput

"""
if __name__ == '__main__':

    # Define switches, hosts, and links
    switches = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    hosts = [f"h{h}" for h in range(1, 18)]
    links = {
        (1, 2): 1000, (2, 1): 1000, (1, 3): 1000, (3, 1): 1000, (1, 4): 1000, (4, 1): 1000, (1, 9): 1000, (9, 1): 1000, (2, 4): 1000, (4, 2): 1000,
        (2, 9): 1000, (9, 2): 1000, (3, 4): 1000, (4, 3): 1000, (3, 9): 1000, (9, 3): 1000, (4, 5): 1000, (5, 4): 1000, (4, 6): 1000, (6, 4): 1000,
        (4, 7): 1000, (7, 4): 1000, (4, 8): 1000, (8, 4): 1000, (4, 9): 1000, (9, 4): 1000, (4, 10): 1000, (10, 4): 1000, (4, 11): 1000, (11, 4): 1000,
        (4, 15): 1000, (15, 4): 1000, (5, 9): 1000, (9, 5): 1000, (6, 15): 1000, (15, 6): 1000, (7, 9): 1000, (9, 7): 1000, (8, 9): 1000, (9, 8): 1000,
        (9, 10): 1000, (10, 9): 1000, (9, 15): 1000, (15, 9): 1000, (10, 12): 1000, (12, 10): 1000, (10, 13): 1000, (13, 10): 1000, (10, 14): 1000,
        (14, 10): 1000, (10, 15): 1000, (15, 10): 1000, (10, 16): 1000, (16, 10): 1000, (10, 17): 1000, (17, 10): 1000, (11, 15): 1000, (15, 11): 1000,
        (12, 15): 1000, (15, 12): 1000, (13, 15): 1000, (15, 13): 1000, (14, 15): 1000, (15, 14): 1000, (15, 16): 1000, (16, 15): 1000,
        (15, 17): 1000, (17, 15): 1000
    }
    port_info = {
        (1, 2): 3, (2, 1): 2,(1, 3): 4, (3, 1): 2,(1, 4): 5, (4, 1): 2,(1, 9): 10, (9, 1): 2,(2, 4): 5, (4, 2): 3,(2, 9): 10, (9, 2): 3,(3, 4): 5, 
        (4, 3): 4,(3, 9): 10, (9, 3): 4,(4, 5): 6, (5, 4): 5,(4, 6): 7, (6, 4): 5,(4, 7): 8, (7, 4): 5,(4, 8): 9, (8, 4): 5,(4, 9): 10, (9, 4): 5,
        (4, 10): 11, (10, 4): 5,(4, 11): 12, (11, 4): 5,(4, 15): 16, (15, 4): 5,(5, 9): 10, (9, 5): 6,(6, 15): 16, (15, 6): 7,(7, 9): 10, (9, 7): 8,
        (8, 9): 10, (9, 8): 9,(9, 10): 11, (10, 9): 10,(9, 15): 16, (15, 9): 10,(10, 12): 13, (12, 10): 11,(10, 13): 14, (13, 10): 11,(10, 14): 15, 
        (14, 10): 11,(10, 15): 16, (15, 10): 11,(10, 16): 17, (16, 10): 11,(10, 17): 18, (17, 10): 11,(11, 15): 16, (15, 11): 12,(12, 15): 16, 
        (15, 12): 13,(13, 15): 16, (15, 13): 14,(14, 15): 16, (15, 14): 15,(15, 16): 17, (16, 15): 16,(15, 17): 18, (17, 15): 16,
    }
    target_switches_ = [1,2,3, 4,9,10,15]
    env = NetworkEnvironment(switches, hosts, links, port_info, max_ports_per_linecard=4, fixed_throughput=200, target_switches = target_switches_)
    
    # print("Initial linecard status vector:", env.state_linecard_status)
    # print("Initial switch status vector:", env.state_switch_status)
    # print("Initial switch degree vector:", env.state_switch_degree)   

    print("\n=== network ===")
    print(env.network)
    print("\n=== linecard_mapping ===")
    print(env.linecard_mapping)
    print("\n=== linecard_status ===")
    print(env.linecard_status) 
    print("\n=== traffic_volume_dict ===")
    print(env.traffic_volume_dict)
    print("\n=== shortest_paths ===")
    print(env.shortest_paths)
    print("\n=== flows_per_switch ===")
    print(env.flows_per_switch)
    print("\n=== flows_per_link ===")
    print(env.flows_per_link)
    print("\n=== switch_degrees ===")
    print(env.switch_degrees)
    print("\n=== state_linecard_status ===")
    print(env.state_linecard_status )
    print("\n=== state_switch_status ===")
    print(env.state_switch_status )
    print("\n=== state_switch_degree ===")
    print(env.state_switch_degree ) 





    # ini_state_ = state_linecard_status + state_switch_status + state_switch_degree

    # # extracted_linecards = {switch: linecard_mapping[switch] for switch in target_switches if switch in linecard_mapping
    # filtered_linecard_status  = {key: value for key, value in linecard_status.items() if key[0] in target_switches}
    # sorted_keys = sorted(filtered_linecard_status.keys())
    # ini_action_ = [1 if filtered_linecard_status[key] else 0 for key in sorted_keys]
    # # print(list(filtered_linecard_status.items())[0])
    # print(ini_action_)
    # reward = ini_action_.count(0) * 50 
    # print(reward)
    # print(flows_per_link)

    
      

    # print("\n=== Linecard Status After Failure ===")
    # print(env.state_switch_degree)   
    # failures = [(4, 1, False)]         # Linecard 1 on Switch 1 and Linecard 1 on Switch 4 fail
    # updated_network = env.update_network_on_linecard_status(failures)
    # print("\n=== after failed ===")
    # print(env.state_switch_degree)   

    # print("\n=== after recovery ===")
    # failures = [(4, 1, False)]         # Linecard 1 on Switch 1 and Linecard 1 on Switch 4 fail
    # updated_network = env.update_network_on_linecard_status(failures)
    # print(env.state_switch_degree)   

    print()
    print()
    print()
    action = [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    # Flatten the linecard mapping for target switches
    linecards = [(switch, linecard_id) for switch in env.target_switches
                 for linecard_id in sorted(env.linecard_mapping[switch].keys())]

    print(linecards)
    # Apply the action by comparing with current statuses
    changes = []
    for idx, (switch, linecard_id) in enumerate(linecards):
        current_status = env.linecard_status[(switch, linecard_id)]
        desired_status = bool(action[idx])
        if current_status != desired_status:
            changes.append((switch, linecard_id, desired_status))

    print(changes)
    env.update_network_on_linecard_status(changes)




    print("====")
    action = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    # Flatten the linecard mapping for target switches
    linecards = [(switch, linecard_id) for switch in env.target_switches
                 for linecard_id in sorted(env.linecard_mapping[switch].keys())]

    print(linecards)
    # Apply the action by comparing with current statuses
    changes = []
    for idx, (switch, linecard_id) in enumerate(linecards):
        current_status = env.linecard_status[(switch, linecard_id)]
        desired_status = bool(action[idx])
        if current_status != desired_status:
            changes.append((switch, linecard_id, desired_status))

    print(changes)
    env.update_network_on_linecard_status(changes)

"""






    # basic_energy = 542 + 260
    # energy = basic_energy + np.sum(action == 0) * 50
    # connectivity_penalty = 0
    # if not nx.is_connected(self.network.to_undirected()):
    #     connectivity_penalty = 500
    # bandwidth_exceed_penalty = 0
    # for (u, v), bandwidth in self.links.items():
    #     link_throughput = self.flows_per_link.get((u, v), 0) * self.fixed_throughput
    #     if link_throughput > bandwidth:
    #         bandwidth_exceed_penalty += (link_throughput - bandwidth) * 0.1  # Example penalty factor

    # # Normalize components for reward calculation
    # energy_scaled = energy / 1000  # Scale to range [0, 1]
    # connectivity_penalty_scaled = connectivity_penalty / 500  # Scale to [0, 1]
    # bandwidth_exceed_penalty_scaled = bandwidth_exceed_penalty / 1000  # Scale to [0, 1]
    # reward = (-1) * (a1 * energy_scaled + a2 * connectivity_penalty_scaled + a3 * bandwidth_exceed_penalty_scaled)
