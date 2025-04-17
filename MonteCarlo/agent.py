import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import math
from collections import defaultdict, deque, namedtuple

# Experience tuple for episode storage
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class EpisodeBuffer:
    """Buffer to store entire episodes for Monte Carlo methods."""
    
    def __init__(self, capacity=1000):
        """
        Initialize an EpisodeBuffer.
        
        Params
        ======
            capacity (int): maximum number of episodes to store
        """
        self.episodes = deque(maxlen=capacity)
        self.current_episode = []
    
    def add_experience(self, state, action, reward, next_state, done):
        """Add a new step to the current episode."""
        # Convert tensors to numpy arrays for storage if needed
        state_np = state.cpu().detach().numpy() if torch.is_tensor(state) else state
        next_state_np = next_state.cpu().detach().numpy() if torch.is_tensor(next_state) else next_state
        action_val = action.item() if torch.is_tensor(action) else action
        reward_val = reward.item() if torch.is_tensor(reward) else reward
        done_val = float(done.item()) if torch.is_tensor(done) else float(done)
        
        e = Experience(state_np, action_val, reward_val, next_state_np, done_val)
        self.current_episode.append(e)
    
    def end_episode(self):
        """End the current episode and store it."""
        if self.current_episode:
            self.episodes.append(list(self.current_episode))
            self.current_episode = []
    
    def sample_episodes(self, batch_size):
        """Sample a batch of episodes."""
        if len(self.episodes) < batch_size:
            return self.episodes  # Return all if we don't have enough
        return random.sample(self.episodes, k=batch_size)
    
    def __len__(self):
        """Return the number of stored complete episodes."""
        return len(self.episodes)


class MLPNetwork(nn.Module):
    """Standard MLP architecture for value function approximation."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(MLPNetwork, self).__init__()
        # Layer normalization instead of batch normalization
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.2)
        
        # Multi-layer architecture
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.ln2 = nn.LayerNorm(hidden_dim*2)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.dropout3 = nn.Dropout(0.2)
        
        # Output layer (state-action values)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        """Map state to action values."""
        # Convert to tensor if needed
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
            
        # Handle single state or batch
        if state.dim() == 1:
            state = state.unsqueeze(0)
            is_single = True
        else:
            is_single = False
            
        # Forward pass with activations, normalization, and dropout
        x = F.relu(self.ln1(self.fc1(state)))
        x = self.dropout1(x)
        
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.dropout3(x)
        
        # Final output without activation (raw values)
        action_values = self.fc4(x)
        
        if is_single:
            return action_values.squeeze(0)
        return action_values


class FatMLPNetwork(nn.Module):
    """Fat MLP architecture with more layers and wider hidden dimensions."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(FatMLPNetwork, self).__init__()
        # First layer with large hidden dimension
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.2)
        
        # Wider hidden layers
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.ln2 = nn.LayerNorm(hidden_dim*2)
        self.dropout2 = nn.Dropout(0.2)
        
        # Extra middle layer for more depth
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.ln3 = nn.LayerNorm(hidden_dim*2)
        self.dropout3 = nn.Dropout(0.2)
        
        # Another hidden layer
        self.fc4 = nn.Linear(hidden_dim*2, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)
        self.dropout4 = nn.Dropout(0.2)
        
        # Output layer for action values
        self.fc5 = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights for better gradient flow
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        """Map state to action values."""
        # Convert to tensor if needed
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
            
        # Handle single state or batch
        if state.dim() == 1:
            state = state.unsqueeze(0)
            is_single = True
        else:
            is_single = False
            
        # Forward pass with activations, normalization, and dropout
        x = F.relu(self.ln1(self.fc1(state)))
        x = self.dropout1(x)
        
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = F.relu(self.ln4(self.fc4(x)))
        x = self.dropout4(x)
        
        # Output layer (raw values)
        action_values = self.fc5(x)
        
        if is_single:
            return action_values.squeeze(0)
        return action_values


class TransformerNetwork(nn.Module):
    """Transformer-based network for value function approximation."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerNetwork, self).__init__()
        
        # Embedding dimension must be divisible by number of heads
        d_model = hidden_dim if hidden_dim % nhead == 0 else nhead * (hidden_dim // nhead + 1)
        
        # Input projection to match transformer dimensions
        self.input_projection = nn.Linear(state_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = self.PositionalEncoding(d_model, dropout)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output projection for action values
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.d_model = d_model
        self.seq_len = 1  # Treat the state as a sequence of length 1
        
    class PositionalEncoding(nn.Module):
        """Positional encoding for the transformer model."""
        
        def __init__(self, d_model, dropout=0.1, max_len=200):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            
            # Compute positional encodings
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            pe = pe.transpose(0, 1)  # [1, max_len, d_model]
            self.register_buffer('pe', pe)
            
        def forward(self, x):
            """Add positional encoding to the input tensor."""
            # x shape: [batch_size, seq_len, d_model]
            x = x + self.pe[:, :x.size(1)]
            return self.dropout(x)
    
    def forward(self, state):
        """Process state through transformer architecture."""
        # Convert to tensor if needed
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
            
        # Handle single state or batch
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension
            is_single = True
        else:
            is_single = False
            
        # Reshape for transformer: [batch_size, seq_len=1, feature_dim]
        x = state.unsqueeze(1)
        
        # Project input to transformer dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Average across sequence dimension (though it's just 1 in this case)
        x = x.mean(dim=1)
        
        # Final projection to action values
        action_values = self.output_projection(x)
        
        if is_single:
            return action_values.squeeze(0)
        return action_values


class MonteCarloAgent:
    """Monte Carlo agent for reinforcement learning."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=1e-4, gamma=0.99, 
                 device='cpu', network_type='mlp', nhead=4, num_layers=2):
        """
        Initialize a Monte Carlo Agent.
        
        Params
        ======
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            hidden_dim (int): Hidden dimension size for neural networks
            lr (float): Learning rate
            gamma (float): Discount factor
            device (str): Device to use for training ('cpu' or 'cuda')
            network_type (str): Network architecture ('mlp', 'fat_mlp', or 'transformer')
            nhead (int): Number of attention heads for transformer
            num_layers (int): Number of transformer layers
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = device
        
        # Initialize policy network based on specified architecture
        if network_type == 'transformer':
            print(f"Using Transformer Network with hidden dim {hidden_dim}")
            self.policy_network = TransformerNetwork(
                state_dim, action_dim, hidden_dim, nhead, num_layers
            ).to(self.device)
        elif network_type == 'fat_mlp':
            print(f"Using Fat MLP Network with hidden dim {hidden_dim}")
            self.policy_network = FatMLPNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        else:  # Default to standard MLP
            print(f"Using Standard MLP Network with hidden dim {hidden_dim}")
            self.policy_network = MLPNetwork(state_dim, action_dim, hidden_dim).to(self.device)
            
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        
        # For Monte Carlo exploration
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = 10000
        self.total_steps = 0
        
    def select_action(self, state, epsilon=None):
        """Select action using epsilon-greedy policy."""
        # Use provided epsilon or calculate it based on decay
        if epsilon is None:
            epsilon = max(
                self.epsilon_end, 
                self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (self.total_steps / self.epsilon_decay_steps)
            )
            
        # Exploration
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
            
        # Exploitation
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(self.device)
            if state.dim() == 1:
                state = state.unsqueeze(0)
                
            action_values = self.policy_network(state)
            return action_values.argmax(dim=1).item()
    
    def _calculate_returns(self, episode):
        """Calculate discounted returns for an episode."""
        returns = []
        G = 0
        # Calculate returns from the end of the episode
        for experience in reversed(episode):
            reward = experience.reward
            G = reward + self.gamma * G
            returns.insert(0, G)  # Insert at the beginning
        return returns
    
    def learn(self, episode_buffer, batch_size=32):
        """Update policy network using Monte Carlo returns."""
        if len(episode_buffer) < 1:
            return 0  # No episodes to learn from
            
        # Sample episodes and calculate returns
        sampled_episodes = episode_buffer.sample_episodes(batch_size)
        
        all_states = []
        all_actions = []
        all_returns = []
        
        # Process each episode
        for episode in sampled_episodes:
            # Skip empty episodes
            if not episode:
                continue
                
            # Calculate returns for the episode
            returns = self._calculate_returns(episode)
            
            # Store states, actions, and returns
            for i, experience in enumerate(episode):
                all_states.append(experience.state)
                all_actions.append(experience.action)
                all_returns.append(returns[i])
        
        if not all_states:  # No valid data after processing
            return 0
            
        # Convert to tensors
        states = torch.FloatTensor(np.vstack(all_states)).to(self.device)
        actions = torch.LongTensor(all_actions).unsqueeze(1).to(self.device)
        returns = torch.FloatTensor(all_returns).unsqueeze(1).to(self.device)
        
        # Get predicted action values
        predicted_values = self.policy_network(states).gather(1, actions)
        
        # Calculate loss (MSE between predicted values and returns)
        loss = F.mse_loss(predicted_values, returns)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def increment_step(self):
        """Increment step counter for epsilon decay."""
        self.total_steps += 1
    
    def save_model(self, path):
        """Save model to disk."""
        torch.save(self.policy_network.state_dict(), path)
        
    def load_model(self, path):
        """Load model from disk."""
        self.policy_network.load_state_dict(torch.load(path, map_location=self.device))


# Agent module exports
__all__ = ['MonteCarloAgent', 'MLPNetwork', 'FatMLPNetwork', 'TransformerNetwork', 'EpisodeBuffer', 'Experience']
