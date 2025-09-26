import gym
import numpy as np
from gym import spaces

# Mininet-related imports
from mininet.net import Mininet
from mininet.node import OVSController
from mininet.link import TCLink
from mininet.topo import Topo
from mininet.log import setLogLevel, info
import time

# ------------------------------------------------------
# 1) Define a custom Mininet Topology
# ------------------------------------------------------
class MyCustomTopo(Topo):
    """
    Example topology: N routers (hosts or switches), each
    with up to 4 connections. For real RL tasks, you might
    define your own or build a ring/mesh/star, etc.
    """
    def build(self, num_nodes=4, max_capacity=100):
        # Simply create num_nodes "switches" (or hosts).
        # Then link them in some pattern, e.g. linear chain or partial mesh.
        
        # Add N switches:
        switches = []
        for i in range(num_nodes):
            sw = self.addSwitch(f's{i}')
            switches.append(sw)
        
        # Example: link them in a chain with capacity = max_capacity
        # In Mininet, you can define link parameters (bw, delay, loss, etc.) with TCLink
        for i in range(num_nodes - 1):
            self.addLink(
                switches[i],
                switches[i+1],
                cls=TCLink,
                bw=max_capacity,  # This is "bandwidth" in Mb/s
            )

# ------------------------------------------------------
# 2) Define the RL Environment
# ------------------------------------------------------
class MininetNetworkEnv(gym.Env):
    """
    A custom environment that uses Mininet to emulate a network.
    
    Observations:
      - For each link, usage ratio plus open/closed state
      - Alternatively, you can keep it simpler or more complex depending on your needs
    
    Actions:
      - A binary vector indicating which links are open (1) or closed (0)
    
    Reward:
      - Negative of the number of overloaded links, or something similar
    """
    
    def __init__(
        self,
        num_nodes=4,
        max_capacity=10,  # in Mb/s for Mininet's TCLink
        max_steps=5
    ):
        super(MininetNetworkEnv, self).__init__()
        
        self.num_nodes = num_nodes
        self.max_capacity = max_capacity
        self.max_steps = max_steps
        self.current_step = 0
        
        # Build the topology
        self.topo = MyCustomTopo(num_nodes=self.num_nodes, max_capacity=self.max_capacity)
        
        # Create the Mininet instance
        self.net = Mininet(
            topo=self.topo,
            controller=OVSController,
            link=TCLink,  # allow us to set link bandwidth
            autoSetMacs=True,
            autoStaticArp=True
        )
        
        # Once the net is built, we can retrieve the actual links
        self.link_list = self._get_link_list()
        self.num_links = len(self.link_list)
        
        # Define action space (binary for each link: open=1, close=0)
        self.action_space = spaces.MultiBinary(self.num_links)
        
        # Define observation space, e.g. for each link: (usage_ratio, is_open)
        # usage_ratio ∈ [0, ∞), is_open ∈ [0,1]
        low = np.array([0.0, 0.0] * self.num_links, dtype=np.float32)
        high = np.array([np.inf, 1.0] * self.num_links, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Track open/closed state
        self.link_open = np.ones(self.num_links, dtype=int)
        
    def _get_link_list(self):
        """
        Retrieve the list of links from the Mininet object.
        Each link is typically (node1, node2).
        """
        # But we have to be mindful that Mininet might store them differently
        # We can do something like:
        link_list = []
        for link in self.topo.links(withInfo=False):
            link_list.append(link)
        return link_list
        
    def _set_link_state(self, idx, state):
        """
        Use Mininet commands or config to bring links up or down.
        - idx is which link in self.link_list
        - state is 1 for up, 0 for down
        """
        (node1, node2) = self.link_list[idx]
        # We can identify the interfaces, e.g.:
        intf1 = self.net.get(node1).connectionsTo(self.net.get(node2))[0][0]
        intf2 = self.net.get(node1).connectionsTo(self.net.get(node2))[0][1]
        
        if state == 1:
            # Bring interfaces up
            intf1.config(bw=self.max_capacity, enable_groot=True)
            intf2.config(bw=self.max_capacity, enable_groot=True)
        else:
            # Bring interfaces down
            intf1.ifconfig('down')
            intf2.ifconfig('down')
    
    def _measure_link_usage(self):
        """
        In a real scenario, you'd run iperf or measure traffic on each link.
        For demonstration, we do a dummy "usage" step.
        
        - E.g., start an iperf server on one host, client on another,
          measure throughput, parse the output, and see how close
          it is to the link capacity.
          
        - We'll do a simplistic placeholder that assigns random usage
          for demonstration. Replace with real traffic measurement logic.
        """
        usage = np.random.uniform(0, self.max_capacity, size=self.num_links)
        return usage
    
    def reset(self):
        # Start the Mininet network fresh
        self.current_step = 0
        
        info("*** Starting Mininet ***\n")
        setLogLevel('warning')  # or 'info' for more debug output
        self.net.start()
        
        # By default, set all links to "up"
        self.link_open = np.ones(self.num_links, dtype=int)
        for i in range(self.num_links):
            self._set_link_state(i, 1)
        
        # (Optionally) run some baseline or wait a bit
        time.sleep(1)
        
        # Initial measure
        self.usage = self._measure_link_usage()
        
        return self._get_observation()
    
    def _get_observation(self):
        obs = []
        for i in range(self.num_links):
            usage_ratio = self.usage[i] / self.max_capacity if self.max_capacity else 0.0
            is_open = self.link_open[i]
            obs.append(usage_ratio)
            obs.append(is_open)
        return np.array(obs, dtype=np.float32)
    
    def step(self, action):
        # Apply the action: open/close each link
        self.link_open = action
        for i in range(self.num_links):
            self._set_link_state(i, self.link_open[i])
        
        # Wait a moment for link changes to take effect
        time.sleep(0.5)
        
        # Measure usage
        self.usage = self._measure_link_usage()
        
        # Calculate how many links are overloaded
        overloaded_links = self._count_overloaded_links()
        
        # Reward example: negative of number of overloaded links
        reward = -float(overloaded_links)
        
        # Check if done
        self.current_step += 1
        done = (self.current_step >= self.max_steps)
        
        info = {
            'overloaded_links': overloaded_links
        }
        
        return self._get_observation(), reward, done, info
    
    def _count_overloaded_links(self):
        """
        Count how many links exceed capacity in usage measurement.
        """
        overloaded = 0
        for i in range(self.num_links):
            if self.usage[i] > self.max_capacity:
                overloaded += 1
        return overloaded
    
    def render(self, mode='human'):
        print("Current step:", self.current_step)
        for i, (n1, n2) in enumerate(self.link_list):
            print(f" Link {n1}-{n2} | is_open={self.link_open[i]} | usage={self.usage[i]:.2f}/{self.max_capacity} Mb/s")
    
    def close(self):
        # Stop the Mininet network
        if self.net:
            self.net.stop()
        super().close()

# ------------------------------------------------------
# 3) Simple Testing of the Environment
# ------------------------------------------------------
if __name__ == "__main__":
    env = MininetNetworkEnv(num_nodes=4, max_capacity=10, max_steps=3)
    obs = env.reset()
    
    done = False
    while not done:
        # For demonstration, pick a random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        print(f"Reward: {reward}, Info: {info}\n")
    
    env.close()
