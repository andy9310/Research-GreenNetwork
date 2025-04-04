import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from env import NetworkEnv  # <-- import your custom env
# from int_to_bin_action import int_to_binary_action, binary_action_to_int
# (Or just define them inline if not in separate files)

def int_to_binary_action(index, num_edges):
    """
    Convert an integer index in [0, 2^num_edges - 1] 
    to a binary vector of length num_edges.
    """
    return np.array([int(x) for x in np.binary_repr(index, width=num_edges)], dtype=int)

def binary_action_to_int(action_array):
    """
    Convert a binary vector (shape [num_edges]) to an integer in [0, 2^num_edges - 1].
    """
    # E.g. [1,0,1] => '101' => int=5
    bits_str = ''.join(str(x) for x in action_array)
    return int(bits_str, 2)
# -------------------------
# 1) Create the environment
# -------------------------
env = NetworkEnv(
    num_nodes=6,       # Adjust to keep edges small enough
    max_interfaces=4,
    max_capacity=100,
    max_steps=20,
    seed=42
)

num_edges = env.num_edges
# Observation is 2 * num_edges in shape
state_dim = 2 * num_edges
# We treat each possible MultiBinary action as a discrete action => action_dim = 2^num_edges
action_dim = 2 ** num_edges

print(f"Number of edges: {num_edges}")
print(f"Discrete action space size = {action_dim}")
print(f"Observation space dim = {state_dim}")

# -------------------------
# 2) Define a Q-network
# -------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# -------------------------
# 3) Replay Buffer
# -------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        state, next_state: np.ndarray of shape [state_dim]
        action: int in [0, 2^num_edges - 1]
        reward: float
        done: bool
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# -------------------------
# 4) Define Training Params
# -------------------------
lr = 1e-3
gamma = 0.99
batch_size = 64
buffer_size = 10000
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
target_update_freq = 10
episodes = 300

q_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=lr)
replay_buffer = ReplayBuffer(buffer_size)
criterion = nn.MSELoss()

epsilon = epsilon_start

# -------------------------
# 5) Training Loop
# -------------------------
for time in range(24):
    # train episodes every hour in a day 
    for episode in range(episodes):
        state = env.reset(time)  # shape = [2*num_edges]
        total_reward = 0.0
        done = False
        
        while not done:
            # Epsilon-greedy for discrete actions in [0, 2^num_edges - 1]
            if random.random() < epsilon:
                action_index = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)  # shape [1, state_dim]
                    q_values = q_net(state_tensor)                        # shape [1, action_dim]
                    action_index = q_values.argmax(dim=1).item()
            
            # Convert discrete action index -> MultiBinary edge vector
            bin_action = int_to_binary_action(action_index, num_edges)

            # Step the environment
            next_state, reward, done, info = env.step(bin_action)
            
            # Store in replay buffer
            replay_buffer.push(state, action_index, reward, next_state, done)
            
            state = next_state
            total_reward += reward

            # Train if we have enough samples
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                states_tensor = torch.FloatTensor(states)               # [batch_size, state_dim]
                actions_tensor = torch.LongTensor(actions).unsqueeze(1) # [batch_size, 1]
                rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)# [batch_size, 1]
                next_states_tensor = torch.FloatTensor(next_states)      # [batch_size, state_dim]
                dones_tensor = torch.FloatTensor(dones).unsqueeze(1)     # [batch_size, 1]

                # Q(s,a)
                current_q = q_net(states_tensor).gather(1, actions_tensor)  # shape [batch_size, 1]

                # Q_target(s', a') using target_net
                with torch.no_grad():
                    max_next_q = target_net(next_states_tensor).max(dim=1)[0].unsqueeze(1)  # [batch_size, 1]
                    target_q = rewards_tensor + gamma * max_next_q * (1 - dones_tensor)

                # Compute loss
                loss = criterion(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Epsilon decay
        epsilon = max(epsilon_end, epsilon_decay * epsilon)

        # Update target network
        if episode % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        print(f"Episode: {episode}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

env.close()
