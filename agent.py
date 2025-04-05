import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple

# Experience tuple for Replay Buffer
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, capacity):
        """
        Initialize a ReplayBuffer.

        Params
        ======
            capacity (int): maximum size of buffer
        """
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # Convert tensors to numpy arrays or detach them for storage if they require grads
        state_np = state.cpu().detach().numpy() if torch.is_tensor(state) else state
        next_state_np = next_state.cpu().detach().numpy() if torch.is_tensor(next_state) else next_state
        action_val = action.item() if torch.is_tensor(action) else action
        reward_val = reward.item() if torch.is_tensor(reward) else reward
        done_val = float(done.item()) if torch.is_tensor(done) else float(done) # Ensure done is float

        e = Experience(state_np, action_val, reward_val, next_state_np, done_val)
        self.memory.append(e)

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=batch_size)

        # Convert batch of Experiences to Experience of batches
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class QNetwork(nn.Module):
    """Simple MLP Q-Network."""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQN:
    """Deep Q-Network Agent."""

    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-4, gamma=0.99, device='cpu'):
        """
        Initialize an Agent object.

        Params
        ======
            state_dim (int): dimension of each state
            action_dim (int): dimension of each action
            hidden_dim (int): number of nodes in hidden layers
            lr (float): learning rate
            gamma (float): discount factor
            device (torch.device): device to use for tensors (cpu or cuda)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim # Should be 2 (close/open)
        self.gamma = gamma
        self.device = device

        # Q-Network
        self.qnetwork_local = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.qnetwork_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Initialize target network with local network's weights
        self.update_target_network()

        self.qnetwork_target.eval() # Target network in eval mode


    def select_action(self, state, epsilon=0.0):
        """Selects an action using epsilon-greedy strategy."""
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim) # Explore: choose 0 or 1
        else:
            # Exploit: choose the action with the highest Q-value
            with torch.no_grad():
                # Ensure state is a tensor on the correct device
                if not isinstance(state, torch.Tensor):
                    state = torch.FloatTensor(state).to(self.device)
                if state.dim() == 1:
                    state = state.unsqueeze(0) # Add batch dimension if missing

                q_values = self.qnetwork_local(state)
                action = q_values.argmax(dim=1).item() # Get the index (0 or 1)
                return action

    def learn(self, replay_buffer, batch_size):
        """Update value parameters using given batch of experience tuples."""
        if len(replay_buffer) < batch_size:
            return # Not enough samples yet

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # Move batch to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        # Q_targets = reward + (gamma * Q_targets_next * (1 - done))
        # Important: dones is 0 if terminal, 1 if not (convention in some libs)
        # If your env sets done=True (1.0) for terminal state, use (1 - dones)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        # We need to gather the Q-values corresponding to the actions taken
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()

    def update_target_network(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Alternatively, do a hard update (copy weights directly):
        """
        # Hard update:
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        # print("Target network updated.")

# Initialize QNetwork and DQN with state_dim based on max_edges and action_dim=2
max_edges = 10
state_dim = max_edges
action_dim = 2
hidden_dim = 128
lr = 1e-4
gamma = 0.99
device = 'cpu'

qnetwork = QNetwork(state_dim, action_dim, hidden_dim)
dqn = DQN(state_dim, action_dim, hidden_dim, lr, gamma, device)
