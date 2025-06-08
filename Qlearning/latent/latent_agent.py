"""
Latent Predictor Agent Module

This module combines latent state encoding with state prediction capabilities,
providing a comprehensive agent that can both encode states into a latent space
and predict future states based on actions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque
import random

# Define experience tuple type
Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.
        
        Args:
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            device (torch.device): device for tensor operations
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
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
        """
        Initialize the memory buffer.
        
        Args:
            buffer_size (int): Maximum size of the buffer
            batch_size (int): Size of batches to sample
            device (torch.device): Device for tensor operations
        """
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
        actions = torch.from_numpy(np.vstack([t[1] for t in transitions])).long().to(self.device)
        next_states = torch.from_numpy(np.vstack([t[2] for t in transitions])).float().to(self.device)
        
        return states, actions, next_states
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.memory)


class StateEncoder(nn.Module):
    """
    Encodes the network state into a latent representation.
    
    The state includes:
    - Link open status (binary)
    - Link usage-to-capacity ratios
    - Current edge index
    
    This encoder compresses this information into a fixed-size latent representation.
    """
    
    def __init__(self, state_dim, max_edges=10, latent_dim=64, dropout_rate=0.2):
        """
        Initialize the State Encoder network.
        
        Args:
            state_dim (int): Dimension of the state vector
            max_edges (int): Maximum number of edges in the network
            latent_dim (int): Dimension of the latent representation
            dropout_rate (float): Dropout rate for regularization
        """
        super(StateEncoder, self).__init__()
        self.state_dim = state_dim
        self.max_edges = max_edges
        self.latent_dim = latent_dim
        
        # Simple MLP encoder with fixed input dimension
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, latent_dim)
        )
    
    def forward(self, state):
        """
        Forward pass through the encoder.
        
        Args:
            state (torch.Tensor): The state tensor of shape [batch_size, state_dim]
            
        Returns:
            torch.Tensor: Latent representation of shape [batch_size, latent_dim]
        """
        # Ensure state has batch dimension
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        # Get the first self.state_dim elements if state is larger
        if state.shape[1] > self.state_dim:
            state = state[:, :self.state_dim]
            
        # Pad with zeros if state is smaller than expected
        if state.shape[1] < self.state_dim:
            padding = torch.zeros(state.shape[0], self.state_dim - state.shape[1], device=state.device)
            state = torch.cat([state, padding], dim=1)
        
        # Pass through encoder
        return self.encoder(state)


class StatePredictor(nn.Module):
    """
    Neural network that predicts the next state given the current state and action.
    
    This model serves as a learned dynamics model of the environment.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, dropout_rate=0.1):
        """
        Initialize the State Predictor network.
        
        Args:
            state_dim (int): Dimension of the state vector
            action_dim (int): Dimension of the action space (used for one-hot encoding)
            hidden_dim (int): Size of hidden layers
            dropout_rate (float): Dropout rate for regularization
        """
        super(StatePredictor, self).__init__()
        
        # Action will be one-hot encoded and concatenated with state
        input_dim = state_dim + action_dim
        
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
    
    def forward(self, state, action):
        """
        Forward pass to predict next state.
        
        Args:
            state (torch.Tensor): Current state tensor [batch_size, state_dim]
            action (torch.Tensor): Action indices [batch_size]
            
        Returns:
            torch.Tensor: Predicted next state [batch_size, state_dim]
        """
        # Ensure state has batch dimension
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        # Ensure action has batch dimension
        if len(action.shape) == 0:
            action = action.unsqueeze(0)
        
        batch_size = state.shape[0]
        action_dim = self.network[0].in_features - state.shape[1]
        
        # Handle the action - ensure it's within bounds
        # If action is a scalar or has shape [batch_size], make sure it's properly bounded
        if action_dim > 0:
            # Convert action to long type to use as indices
            action = action.long()
            
            # Clamp action to be within valid range of action_dim
            action = torch.clamp(action, 0, action_dim - 1)
            
            # Create one-hot encoding
            action_one_hot = torch.zeros(batch_size, action_dim, device=state.device)
            action_one_hot.scatter_(1, action.unsqueeze(1), 1)
            
            # Concatenate state and action
            state_action = torch.cat([state, action_one_hot], dim=1)
        else:
            # If for some reason action_dim is 0 or negative, just use state
            print(f"Warning: Invalid action dimension {action_dim}. Using only state for prediction.")
            state_action = state
        
        # Predict next state
        next_state_pred = self.network(state_action)
        
        return next_state_pred


class EnhancedQNetwork(nn.Module):
    """
    Enhanced Q-Network with latent state encoding.
    
    This network has two parts:
    1. State encoder: Transforms network state into latent representation
    2. Q-value network: Maps latent representation to Q-values for actions
    
    Available architectures:
    - 'mlp': Standard MLP architecture
    - 'fatmlp': Larger MLP with increased capacity
    - 'advanced': Advanced architecture with residual connections
    """
    
    def __init__(self, state_dim, action_dim, latent_dim=64, dropout_rate=0.2, architecture='mlp', num_nodes=None):
        """
        Initialize the Enhanced Q-Network.
        
        Args:
            state_dim (int): Dimension of the input state
            action_dim (int): Number of possible actions
            latent_dim (int): Dimension of the latent state representation
            dropout_rate (float): Dropout rate for regularization
            architecture (str): Network architecture type ('mlp', 'fatmlp', 'advanced', or 'noisy')
            num_nodes (int, optional): Number of nodes in the network, for adjacency matrix representation
        """
        super(EnhancedQNetwork, self).__init__()
        
        # State encoder
        # Calculate max_edges from state_dim for backward compatibility
        max_edges = (state_dim-1)
        self.state_encoder = StateEncoder(state_dim=state_dim, max_edges=max_edges, latent_dim=latent_dim)
        
        # Select architecture type
        self.architecture = architecture
        
        if architecture == 'mlp':
            # Standard MLP architecture
            self.q_network = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                nn.Linear(64, action_dim)
            )
        elif architecture == 'fatmlp':
            # Larger MLP with increased capacity
            self.q_network = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                nn.Linear(128, action_dim)
            )
        elif architecture == 'advanced':
            # Advanced architecture with residual connections and more capacity
            # This encoder is specifically for processing traffic patterns
            self.traffic_encoder = nn.Sequential(
                nn.Linear(latent_dim, 384),
                nn.GELU(),  # GELU activation often performs better than ReLU for deep networks
                nn.LayerNorm(384)
            )
            
            # Residual blocks for better gradient flow and information preservation
            self.res_block1 = nn.Sequential(
                nn.Linear(384, 384),
                nn.GELU(),
                nn.LayerNorm(384),
                nn.Dropout(dropout_rate)
            )
            
            self.res_block2 = nn.Sequential(
                nn.Linear(384, 384),
                nn.GELU(),
                nn.LayerNorm(384),
                nn.Dropout(dropout_rate)
            )
            
            # Final layers for action prediction
            self.output_layers = nn.Sequential(
                nn.Linear(384, 192),
                nn.GELU(),
                nn.LayerNorm(192),
                nn.Dropout(dropout_rate),
                nn.Linear(192, action_dim)
            )
        else:
            raise ValueError(f"Unknown architecture type: {architecture}. Use 'mlp', 'fatmlp', or 'advanced'")
    


    def forward(self, state):
        """
        Forward pass through the enhanced Q-network.
        
        Args:
            state: Input state
        
        Returns:
            Q-values for each action
        """
        # Encode state to latent representation
        x = self.state_encoder(state)
        
        # Handle different architectures
        if self.architecture == 'advanced':
            # Advanced architecture with residual connections
            x = self.traffic_encoder(x)
            
            # Apply residual connections
            residual = x
            x = self.res_block1(x) + residual
            
            residual = x
            x = self.res_block2(x) + residual
            
            return self.output_layers(x)
        else:
            # Standard or fatmlp architectures
            return self.q_network(x)


class LatentPredictorAgent:
    """
    Agent that combines latent state encoding with state prediction capabilities.
    
    This agent can:
    1. Encode states into a latent space for more efficient DQN learning
    2. Predict future states based on current state and action
    3. Operate in model-based mode during evaluation, using predicted states
    """
    
    def __init__(self, state_dim, action_dim, latent_dim=64, predictor_hidden_dim=256,
             buffer_size=100000, batch_size=64, predictor_batch_size=128,
             gamma=0.99, tau=1e-3, lr=5e-4, predictor_lr=1e-3,
             update_every=4, enable_predictor=True, architecture='mlp', num_nodes=None, device=None):
        """
        Initialize the Latent Predictor Agent.
        
        Args:
            state_dim (int): Dimension of the state
            action_dim (int): Dimension of action space
            latent_dim (int): Dimension of latent state representation
            predictor_hidden_dim (int): Hidden dimension of state predictor
            buffer_size (int): Replay buffer size for DQN
            batch_size (int): Batch size for DQN updates
            predictor_batch_size (int): Batch size for state predictor updates
            gamma (float): Discount factor
            tau (float): Soft update parameter
            lr (float): Learning rate for DQN
            predictor_lr (float): Learning rate for state predictor
            update_every (int): How often to update networks
            enable_predictor (bool): Whether to enable state prediction
            device (torch.device): Device to use
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.predictor_batch_size = predictor_batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.enable_predictor = enable_predictor
        
        # Determine device
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Q-Networks with specified architecture
        self.architecture = architecture
        self.num_nodes = num_nodes
        self.qnetwork_local = EnhancedQNetwork(state_dim, action_dim, latent_dim, architecture=architecture, num_nodes=num_nodes).to(self.device)
        self.qnetwork_target = EnhancedQNetwork(state_dim, action_dim, latent_dim, architecture=architecture, num_nodes=num_nodes).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        
        # For tracking learning metrics
        self.q_losses = []
        
        # State Predictor (optional)
        if enable_predictor:
            self.state_predictor = StatePredictor(state_dim, action_dim, predictor_hidden_dim).to(self.device)
            self.predictor_optimizer = optim.Adam(self.state_predictor.parameters(), lr=predictor_lr)
            self.predictor_memory = PredictorMemory(buffer_size, predictor_batch_size, self.device)
            self.prediction_errors = []
        
        # Replay memory for DQN
        self.memory = ReplayBuffer(buffer_size, batch_size, self.device)
        
        # Initialize time step (for updating every update_every steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        """Add experience to memory and learn if it's time."""
        # Add experience to DQN replay buffer
        self.memory.add(state, action, reward, next_state, done)
        
        # Add state transition to predictor memory (if enabled)
        if self.enable_predictor:
            self.predictor_memory.add(state, action, next_state)
        
        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # Update DQN if enough samples
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn_dqn(experiences)
            
            # Update state predictor if enabled and enough samples
            if self.enable_predictor and len(self.predictor_memory) > self.predictor_batch_size:
                states, actions, next_states = self.predictor_memory.sample()
                self.learn_predictor(states, actions, next_states)
    
    def act(self, state, epsilon=0.0):
        """Returns actions for given state as per current policy.
        
        Args:
            state (np.array): Current state
            epsilon (float): Epsilon for epsilon-greedy action selection
        """
        # Ensure state is a numpy array
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
            
        # Convert state to tensor
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # Set network to evaluation mode
        self.qnetwork_local.eval()
        
        with torch.no_grad():
            try:
                # Get action values
                action_values = self.qnetwork_local(state_tensor)
            except Exception as e:
                print(f"Error in forward pass: {e}")
                print(f"State tensor shape: {state_tensor.shape}")
                raise
            
        # Back to training mode
        self.qnetwork_local.train()
        
        # Epsilon-greedy action selection
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_dim))
    
    def predict_next_state(self, state, action):
        """Predict the next state using the state predictor."""
        if not self.enable_predictor:
            raise ValueError("State predictor is not enabled for this agent")
            
        # Convert inputs to tensors
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action_tensor = torch.tensor([action], device=self.device)
        
        # Set to evaluation mode
        self.state_predictor.eval()
        
        with torch.no_grad():
            # Predict next state
            next_state_pred = self.state_predictor(state_tensor, action_tensor)
            
        # Back to training mode
        self.state_predictor.train()
        
        return next_state_pred.cpu().numpy().squeeze()
    
    def learn_dqn(self, experiences):
        """Update DQN value parameters using given batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values for next states from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Store the loss value for visualization
        self.q_losses.append(loss.item())
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        
        return loss.item()
    
    def learn_predictor(self, states, actions, next_states):
        """Update state predictor using given batch of transitions."""
        if not self.enable_predictor:
            return 0.0
            
        # Predict next states
        next_states_pred = self.state_predictor(states, actions.squeeze(1))
        
        # Compute loss (MSE between predicted and actual next states)
        loss = F.mse_loss(next_states_pred, next_states)
        
        # Track prediction error
        self.prediction_errors.append(loss.item())
        
        # Minimize the loss
        self.predictor_optimizer.zero_grad()
        loss.backward()
        self.predictor_optimizer.step()
        
        return loss.item()
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update target network parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def save(self, model_path, predictor_path=None):
        """Save model weights to files."""
        torch.save(self.qnetwork_local.state_dict(), model_path)
        
        if self.enable_predictor and predictor_path:
            torch.save(self.state_predictor.state_dict(), predictor_path)
    
    def load(self, model_path, predictor_path=None):
        """Load model weights from files with backward compatibility."""
        try:
            # Try to load with standard approach
            self.qnetwork_local.load_state_dict(torch.load(model_path, map_location=self.device))
        except RuntimeError as e:
            print("\nDetected model architecture mismatch. Using compatibility mode...")
            # Load the state dict but don't enforce strict matching
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Filter out incompatible keys
            compatible_state_dict = {}
            for name, param in self.qnetwork_local.named_parameters():
                # If the parameter exists in the saved model, use it
                if name in state_dict:
                    compatible_state_dict[name] = state_dict[name]
            
            # Load the compatible parameters only
            self.qnetwork_local.load_state_dict(compatible_state_dict, strict=False)
            print("Loaded compatible parameters successfully. Some weights may be initialized randomly.")
        
        # Copy weights to target network
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        
        # Load predictor if available
        if self.enable_predictor and predictor_path:
            try:
                self.state_predictor.load_state_dict(torch.load(predictor_path, map_location=self.device))
            except RuntimeError:
                print("\nPredictor model architecture mismatch. Using compatibility mode...")
                predictor_state_dict = torch.load(predictor_path, map_location=self.device)
                compatible_predictor_dict = {}
                for name, param in self.state_predictor.named_parameters():
                    if name in predictor_state_dict:
                        compatible_predictor_dict[name] = predictor_state_dict[name]
                self.state_predictor.load_state_dict(compatible_predictor_dict, strict=False)
                print("Loaded compatible predictor parameters successfully.")
