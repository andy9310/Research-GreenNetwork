"""
DDPG Agent with State Prediction for Relaxed Network Optimization

This module implements a DDPG (Deep Deterministic Policy Gradient) agent 
with state prediction capabilities, suitable for continuous control of 
network link capacities.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import copy
import os

# Define experience tuple types
Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
Transition = namedtuple("Transition", field_names=["state", "action", "next_state"])

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object."""
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=min(len(self.memory), self.batch_size))
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PredictorMemory:
    """Memory buffer for storing state transitions for training the state predictor."""
    
    def __init__(self, buffer_size, batch_size, device):
        """Initialize the predictor memory buffer."""
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device
        
    def add(self, state, action, next_state):
        """Add a transition to the buffer."""
        self.memory.append((state, action, next_state))
        
    def sample(self):
        """Sample a batch of transitions."""
        transitions = random.sample(self.memory, k=min(self.batch_size, len(self.memory)))
        
        states = torch.from_numpy(np.vstack([t[0] for t in transitions])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([t[1] for t in transitions])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([t[2] for t in transitions])).float().to(self.device)
        
        return states, actions, next_states
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.memory)


class StateEncoder(nn.Module):
    """Encodes the network state into a latent representation."""
    
    def __init__(self, state_dim, max_edges, latent_dim=64, dropout_rate=0.2):
        """Initialize the State Encoder network."""
        super(StateEncoder, self).__init__()
        
        # Store dimensions
        self.state_dim = state_dim
        self.max_edges = max_edges
        
        # Simple encoder network
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, latent_dim)
        )
    
    def forward(self, state):
        """Forward pass through the encoder."""
        # Ensure state has batch dimension
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        # Handle dimension issues
        if state.shape[1] > self.state_dim:
            state = state[:, :self.state_dim]
        elif state.shape[1] < self.state_dim:
            padding = torch.zeros(state.shape[0], self.state_dim - state.shape[1], device=state.device)
            state = torch.cat([state, padding], dim=1)
        
        return self.encoder(state)


class StatePredictor(nn.Module):
    """Neural network that predicts the next state given the current state and action."""
    
    def __init__(self, state_dim, latent_dim, hidden_dim=256, dropout_rate=0.1):
        """Initialize the State Predictor network."""
        super(StatePredictor, self).__init__()
        
        # Input is state and action
        input_dim = latent_dim + 1  # Latent state + continuous action
        
        # Prediction network
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, state_dim)  # Output has same dimension as state
        )
    
    def forward(self, latent_state, action):
        """Forward pass to predict next state."""
        # Ensure proper dimensions
        if len(latent_state.shape) == 1:
            latent_state = latent_state.unsqueeze(0)
        
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        elif len(action.shape) == 0:
            action = action.unsqueeze(0).unsqueeze(1)
        
        # If action is not 1-dimensional (e.g., [batch_size, 1]), reshape it
        if len(action.shape) > 1 and action.shape[1] == 1:
            action = action.squeeze(1)
        
        # Concatenate latent state and action
        combined = torch.cat([latent_state, action.unsqueeze(1)], dim=1)
        
        # Predict next state
        return self.network(combined)


class Actor(nn.Module):
    """Actor (Policy) Network for DDPG."""
    
    def __init__(self, state_dim, latent_dim=64, hidden_size=256, dropout_rate=0.1):
        """Initialize the Actor network."""
        super(Actor, self).__init__()
        
        # State encoder
        self.state_encoder = StateEncoder(
            state_dim=state_dim,
            max_edges=(state_dim-1)//2,  # Calculate max_edges from state_dim
            latent_dim=latent_dim,
            dropout_rate=dropout_rate
        )
        
        # Actor network (policy)
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.LayerNorm(hidden_size//2),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid()  # Output in [0,1] for capacity scaling factor
        )
    
    def forward(self, state):
        """Forward pass of the actor network."""
        # Encode state to latent representation
        latent_state = self.state_encoder(state)
        
        # Compute action
        return self.network(latent_state)


class Critic(nn.Module):
    """Critic (Value) Network for DDPG."""
    
    def __init__(self, state_dim, latent_dim=64, hidden_size=256, dropout_rate=0.1):
        """Initialize the Critic network."""
        super(Critic, self).__init__()
        
        # State encoder
        self.state_encoder = StateEncoder(
            state_dim=state_dim,
            max_edges=(state_dim-1)//2,  # Calculate max_edges from state_dim
            latent_dim=latent_dim,
            dropout_rate=dropout_rate
        )
        
        # Critic network (Q-value)
        self.network = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_size),  # Latent state + action
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.LayerNorm(hidden_size//2),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size//2, 1)  # Q-value
        )
    
    def forward(self, state, action):
        """Forward pass of the critic network."""
        # Encode state to latent representation
        latent_state = self.state_encoder(state)
        
        # Ensure action has the right shape
        if len(action.shape) == 1:
            action = action.unsqueeze(1)
        
        # Concatenate latent state and action
        x = torch.cat([latent_state, action], dim=1)
        
        # Compute Q-value
        return self.network(x)


class DDPGAgent:
    """Deep Deterministic Policy Gradient agent with state prediction capabilities."""
    
    def __init__(self, state_dim, latent_dim=64, predictor_hidden_dim=256,
                 actor_hidden_dim=256, critic_hidden_dim=256,
                 buffer_size=100000, batch_size=64, predictor_batch_size=128,
                 gamma=0.99, tau=1e-3, actor_lr=1e-4, critic_lr=1e-3, predictor_lr=1e-3,
                 enable_predictor=True, device=None):
        """
        Initialize the DDPG agent.
        
        Args:
            state_dim: Dimension of the state
            latent_dim: Dimension of latent state representation
            predictor_hidden_dim: Hidden dimension of state predictor
            actor_hidden_dim: Hidden dimension of actor network
            critic_hidden_dim: Hidden dimension of critic network
            buffer_size: Replay buffer size
            batch_size: Batch size for DDPG updates
            predictor_batch_size: Batch size for predictor updates
            gamma: Discount factor
            tau: Soft update parameter
            actor_lr: Learning rate for actor
            critic_lr: Learning rate for critic
            predictor_lr: Learning rate for state predictor
            enable_predictor: Whether to enable state prediction
            device: Device to use for computation
        """
        # Store parameters
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.predictor_batch_size = predictor_batch_size
        self.enable_predictor = enable_predictor
        
        # Determine device
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Actor networks
        self.actor_local = Actor(state_dim, latent_dim, actor_hidden_dim).to(self.device)
        self.actor_target = Actor(state_dim, latent_dim, actor_hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=actor_lr)
        
        # Critic networks
        self.critic_local = Critic(state_dim, latent_dim, critic_hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, latent_dim, critic_hidden_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=critic_lr)
        
        # Hard update target networks to match local networks
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)
        
        # State predictor (optional)
        if self.enable_predictor:
            self.state_predictor = StatePredictor(
                state_dim=state_dim, 
                latent_dim=latent_dim,
                hidden_dim=predictor_hidden_dim
            ).to(self.device)
            self.predictor_optimizer = optim.Adam(self.state_predictor.parameters(), lr=predictor_lr)
            self.predictor_memory = PredictorMemory(buffer_size, predictor_batch_size, self.device)
            self.prediction_errors = []  # Track prediction errors
        
        # Experience replay
        self.memory = ReplayBuffer(buffer_size, batch_size, self.device)
        
        # Noise process for exploration (Ornstein-Uhlenbeck process)
        self.noise = OUNoise(1)  # Action dimension is 1 (capacity scaling factor)
        
        # Training step counter
        self.train_step_count = 0
    
    def step(self, state, action, reward, next_state, done):
        """
        Save experience in replay memory and train networks.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Save experience
        self.memory.add(state, action, reward, next_state, done)
        
        # Save state transition for predictor
        if self.enable_predictor:
            self.predictor_memory.add(state, action, next_state)
        
        # Learn, if enough samples in memory
        if len(self.memory) >= self.batch_size:
            self.train_step_count += 1
            experiences = self.memory.sample()
            self.learn(experiences)
            
            # Train state predictor periodically
            if self.enable_predictor and len(self.predictor_memory) >= self.predictor_batch_size:
                states, actions, next_states = self.predictor_memory.sample()
                self.learn_predictor(states, actions, next_states)
    
    def act(self, state, add_noise=True):
        """
        Return action for given state based on current policy.
        
        Args:
            state: Current state
            add_noise: Whether to add noise for exploration
            
        Returns:
            Action as a capacity scaling factor [0,1]
        """
        # Convert state to tensor
        state = torch.from_numpy(state).float().to(self.device)
        
        # Set actor to evaluation mode
        self.actor_local.eval()
        
        with torch.no_grad():
            # Get action from actor network
            action = self.actor_local(state).cpu().data.numpy()
        
        # Set actor back to training mode
        self.actor_local.train()
        
        # Add noise for exploration
        if add_noise:
            action += self.noise.sample()
        
        # Ensure action is in [0,1]
        return np.clip(action, 0, 1)
    
    def predict_next_state(self, state, action):
        """
        Predict the next state using the state predictor.
        
        Args:
            state: Current state
            action: Action to take
            
        Returns:
            Predicted next state
        """
        if not self.enable_predictor:
            raise ValueError("State predictor is not enabled")
        
        # Convert inputs to tensors
        state_tensor = torch.from_numpy(state).float().to(self.device)
        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)
        
        # Encode state to latent representation
        latent_state = self.actor_local.state_encoder(state_tensor)
        
        # Set predictor to evaluation mode
        self.state_predictor.eval()
        
        with torch.no_grad():
            # Predict next state
            next_state_pred = self.state_predictor(latent_state, action_tensor)
        
        # Set predictor back to training mode
        self.state_predictor.train()
        
        return next_state_pred.cpu().numpy()
    
    def reset(self):
        """Reset the noise process."""
        self.noise.reset()
    
    def learn(self, experiences):
        """
        Update actor and critic networks.
        
        Args:
            experiences: Tuple of (states, actions, rewards, next_states, dones)
        """
        states, actions, rewards, next_states, dones = experiences
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next actions
        next_actions = self.actor_target(next_states)
        
        # Get Q values from target models for next states
        Q_targets_next = self.critic_target(next_states, next_actions)
        
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize critic loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)  # Gradient clipping
        self.critic_optimizer.step()
        
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # Minimize actor loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)  # Gradient clipping
        self.actor_optimizer.step()
        
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)
    
    def learn_predictor(self, states, actions, next_states):
        """
        Update state predictor.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            next_states: Batch of next states
        """
        if not self.enable_predictor:
            return
        
        # Encode states to latent representation
        latent_states = self.actor_local.state_encoder(states)
        
        # Predict next states
        next_states_pred = self.state_predictor(latent_states, actions)
        
        # Compute loss
        predictor_loss = F.mse_loss(next_states_pred, next_states)
        
        # Track prediction error
        self.prediction_errors.append(predictor_loss.item())
        
        # Minimize loss
        self.predictor_optimizer.zero_grad()
        predictor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.state_predictor.parameters(), 1)  # Gradient clipping
        self.predictor_optimizer.step()
        
        return predictor_loss.item()
    
    def soft_update(self, local_model, target_model):
        """
        Soft update target network parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        
        Args:
            local_model: Source model
            target_model: Target model
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def hard_update(self, target_model, source_model):
        """
        Copy network parameters from source to target.
        θ_target = θ_source
        
        Args:
            target_model: Target model
            source_model: Source model
        """
        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.copy_(source_param.data)
    
    def save(self, actor_path, critic_path, predictor_path=None):
        """
        Save model weights.
        
        Args:
            actor_path: Path to save actor weights
            critic_path: Path to save critic weights
            predictor_path: Path to save predictor weights
        """
        try:
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(actor_path), exist_ok=True)
            os.makedirs(os.path.dirname(critic_path), exist_ok=True)
            if predictor_path:
                os.makedirs(os.path.dirname(predictor_path), exist_ok=True)
                
            # Save models
            torch.save(self.actor_local.state_dict(), actor_path)
            print(f"Actor saved to {actor_path}")
            
            torch.save(self.critic_local.state_dict(), critic_path)
            print(f"Critic saved to {critic_path}")
            
            if self.enable_predictor and predictor_path:
                torch.save(self.state_predictor.state_dict(), predictor_path)
                print(f"Predictor saved to {predictor_path}")
                
            return True
        except Exception as e:
            print(f"Error saving models: {e}")
            return False
    
    def load(self, actor_path, critic_path, predictor_path=None):
        """
        Load model weights.
        
        Args:
            actor_path: Path to load actor weights
            critic_path: Path to load critic weights
            predictor_path: Path to load predictor weights
        """
        self.actor_local.load_state_dict(torch.load(actor_path))
        self.actor_target.load_state_dict(torch.load(actor_path))
        
        self.critic_local.load_state_dict(torch.load(critic_path))
        self.critic_target.load_state_dict(torch.load(critic_path))
        
        if self.enable_predictor and predictor_path:
            self.state_predictor.load_state_dict(torch.load(predictor_path))


class OUNoise:
    """Ornstein-Uhlenbeck process for action exploration."""
    
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()
    
    def reset(self):
        """Reset the internal state."""
        self.state = copy.copy(self.mu)
    
    def sample(self):
        """Update internal state and return a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
