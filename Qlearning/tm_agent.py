"""
Enhanced DQN Agent with Traffic Matrix Representation Learning

This agent includes traffic matrix encoding to improve generalization
across different traffic patterns.
"""

import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque

from tm_encoder import EnhancedQNetwork

# Define experience tuple structure
Experience = namedtuple('Experience', 
    field_names=['state', 'action', 'reward', 'next_state', 'done', 'traffic_matrix'])


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples with traffic matrices."""
    
    def __init__(self, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.
        
        Args:
            buffer_size (int): Maximum size of buffer
            batch_size (int): Size of each training batch
            device: torch device
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device
    
    def add(self, state, action, reward, next_state, done, traffic_matrix):
        """Add a new experience to memory."""
        e = Experience(state, action, reward, next_state, done, traffic_matrix)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=min(len(self.memory), self.batch_size))
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        traffic_matrices = torch.from_numpy(np.array([e.traffic_matrix for e in experiences if e is not None])).float().to(self.device)
        
        return (states, actions, rewards, next_states, dones, traffic_matrices)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class TMEnhancedDQNAgent:
    """DQN Agent with Traffic Matrix Representation Learning."""
    
    def __init__(self, state_dim, action_dim, num_nodes, seed=0,
                 lr=1e-3, gamma=0.99, tau=1e-3, buffer_size=100000, 
                 batch_size=64, hidden_dim=256, update_every=4,
                 tm_embedding_dim=64, device='cpu'):
        """Initialize an Agent object.
        
        Args:
            state_dim (int): Dimension of each state
            action_dim (int): Dimension of each action
            num_nodes (int): Number of nodes in the network
            seed (int): Random seed
            lr (float): Learning rate
            gamma (float): Discount factor
            tau (float): Soft update parameter
            buffer_size (int): Replay buffer size
            batch_size (int): Minibatch size
            hidden_dim (int): Hidden dimension of Q-network
            update_every (int): How often to update the network
            tm_embedding_dim (int): Traffic matrix embedding dimension
            device (str): Device to run on ('cpu' or 'cuda')
        """
        random.seed(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.device = torch.device(device)
        self.tm_embedding_dim = tm_embedding_dim
        
        # Q-Networks (with Traffic Matrix Encoder)
        self.qnetwork_local = EnhancedQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            num_nodes=num_nodes,
            hidden_dim=hidden_dim,
            tm_embedding_dim=tm_embedding_dim
        ).to(self.device)
        
        self.qnetwork_target = EnhancedQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            num_nodes=num_nodes,
            hidden_dim=hidden_dim,
            tm_embedding_dim=tm_embedding_dim
        ).to(self.device)
        
        # Initialize target network weights to match local network
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size, batch_size, self.device)
        
        # Initialize step counter
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done, traffic_matrix):
        """Save experience in replay memory, and occasionally update the target network."""
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done, traffic_matrix)
        
        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            # Get random batch of experiences from memory
            experiences = self.memory.sample()
            # Update networks
            self.learn(experiences)
            
    def act(self, state, traffic_matrix, epsilon=0.0):
        """Returns actions for given state as per current policy.
        
        Args:
            state: Current state array
            traffic_matrix: Current traffic matrix array
            epsilon (float): Epsilon for epsilon-greedy action selection
        
        Returns:
            Action index
        """
        # Convert to tensors
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        traffic_matrix = torch.from_numpy(traffic_matrix).float().to(self.device)
        
        # Set to evaluation mode
        self.qnetwork_local.eval()
        
        with torch.no_grad():
            # Get action values (pass both state and traffic matrix)
            action_values = self.qnetwork_local(state, traffic_matrix)
            
        # Back to training mode
        self.qnetwork_local.train()
        
        # Epsilon-greedy action selection
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_dim))
        
    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        
        Args:
            experiences: Tuple of (s, a, r, s', done, tm) tuples
        """
        states, actions, rewards, next_states, dones, traffic_matrices = experiences
        
        # Get max predicted Q values for next states from target model
        with torch.no_grad():
            Q_targets_next = self.qnetwork_target(next_states, traffic_matrices).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states, traffic_matrices).gather(1, actions)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        
    def soft_update(self, local_model, target_model):
        """Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target
        
        Args:
            local_model: Model whose weights will be copied
            target_model: Model to receive weights
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
            
    def save(self, filepath):
        """Save the trained model."""
        torch.save({
            'qnetwork_state_dict': self.qnetwork_local.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_nodes': self.num_nodes,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'tm_embedding_dim': self.tm_embedding_dim
        }, filepath)
        
    @classmethod
    def load(cls, filepath, device='cpu'):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=torch.device(device))
        
        # Create an instance of the agent
        agent = cls(
            state_dim=checkpoint['state_dim'],
            action_dim=checkpoint['action_dim'],
            num_nodes=checkpoint['num_nodes'],
            tm_embedding_dim=checkpoint.get('tm_embedding_dim', 64),
            device=device
        )
        
        # Load the saved weights
        agent.qnetwork_local.load_state_dict(checkpoint['qnetwork_state_dict'])
        agent.qnetwork_target.load_state_dict(checkpoint['qnetwork_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return agent
