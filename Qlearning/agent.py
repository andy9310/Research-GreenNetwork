import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import math

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
    """Enhanced MLP Q-Network for larger topologies."""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        # Use layer normalization instead of batch normalization
        # Layer norm works fine with batch size of 1
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.2)
        
        # Deeper architecture with multiple hidden layers
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)  # Larger middle layer
        self.ln2 = nn.LayerNorm(hidden_dim*2)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.dropout3 = nn.Dropout(0.2)
        
        # Output layer
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        # Convert to tensor if it's not already
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
            
        # Check if we're dealing with a single state or a batch
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension
            is_single = True
        else:
            is_single = False
            
        # Apply layers with activations, layer norm, and dropout
        x = F.relu(self.ln1(self.fc1(state)))
        x = self.dropout1(x)
        
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.dropout3(x)
        
        # Output layer without activation (Q-values can be any real number)
        x = self.fc4(x)
        
        # Remove batch dimension if input was a single state
        if is_single:
            x = x.squeeze(0)
            
        return x


class FatMLP(nn.Module):
    """Fat MLP architecture with more layers and wider hidden dimensions."""
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(FatMLP, self).__init__()
        # First layer with large hidden dimension
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.2)
        
        # Extremely wide middle layers
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)  # 1024 neurons
        self.ln2 = nn.LayerNorm(hidden_dim*2)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim*4)  # 2048 neurons
        self.ln3 = nn.LayerNorm(hidden_dim*4)
        self.dropout3 = nn.Dropout(0.3)  # Higher dropout for wider layer
        
        self.fc4 = nn.Linear(hidden_dim*4, hidden_dim*2)  # 1024 neurons
        self.ln4 = nn.LayerNorm(hidden_dim*2)
        self.dropout4 = nn.Dropout(0.2)
        
        self.fc5 = nn.Linear(hidden_dim*2, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim)
        self.dropout5 = nn.Dropout(0.1)
        
        # Output layer
        self.fc6 = nn.Linear(hidden_dim, action_dim)
        
        # Optional: Apply weight initialization for better learning
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization for better gradient flow."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        """Build a network that maps state -> action values."""
        # Convert to tensor if it's not already
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
            
        # Check if we're dealing with a single state or a batch
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension
            is_single = True
        else:
            is_single = False
            
        # Apply layers with activations, layer norm, and dropout
        x = F.relu(self.ln1(self.fc1(state)))
        x = self.dropout1(x)
        
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = F.relu(self.ln4(self.fc4(x)))
        x = self.dropout4(x)
        
        x = F.relu(self.ln5(self.fc5(x)))
        x = self.dropout5(x)
        
        # No activation on the output layer (raw Q-values)
        q_values = self.fc6(x)
        
        # Return a single value if input was a single state
        if is_single:
            return q_values.squeeze(0)
        else:
            return q_values


class TransformerQNetwork(nn.Module):
    """Transformer-based Q-Network for network optimization tasks.
    Leverages self-attention mechanisms to capture relationships between network elements.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerQNetwork, self).__init__()
        
        # Input embedding layer to convert input features to hidden dimension
        self.input_embedding = nn.Linear(1, hidden_dim)  # Each state element gets embedded
        
        # Positional encoding for transformer
        self.pos_encoder = self.PositionalEncoding(hidden_dim, dropout)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim*2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output projection layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # Final output layer
        self.output = nn.Linear(hidden_dim, action_dim)
        
    class PositionalEncoding(nn.Module):
        """Positional encoding for the transformer model."""
        def __init__(self, d_model, dropout=0.1, max_len=200):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            pe = pe.transpose(0, 1)  # [1, max_len, d_model]
            self.register_buffer('pe', pe)

        def forward(self, x):
            """Add positional encoding to the input tensor."""
            # x: [batch_size, seq_len, d_model]
            x = x + self.pe[:, :x.size(1)]
            return self.dropout(x)
        
    def forward(self, state):
        """Process input state through transformer architecture."""
        # Convert to tensor if it's not already
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
            
        # Check if we're dealing with a single state or a batch
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension
            is_single = True
        else:
            is_single = False
            
        # Reshape to [batch_size, seq_len, 1] where seq_len = state_dim
        x = state.unsqueeze(-1)  # Add feature dimension
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Input embedding [batch_size, seq_len, hidden_dim]
        x = self.input_embedding(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Global pooling across sequence dimension to get a single vector per batch item
        x = x.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Apply final layers
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.output(x)
        
        # Remove batch dimension if input was a single state
        if is_single:
            x = x.squeeze(0)
            
        return x


class DQN:
    """Deep Q-Network Agent."""

    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=5e-5, gamma=0.99, device='cpu', network_type='mlp', nhead=4, num_layers=2):
        """
        Initialize an Agent object.

        Params
        ======
            state_dim (int): dimension of each state
            action_dim (int): dimension of each action
            hidden_dim (int): number of nodes in hidden layers
            lr (float): learning rate
            gamma (float): discount factor
            device (string): 'cpu' or 'cuda:0' etc
            network_type (string): 'mlp', 'fat_mlp', or 'transformer'
            nhead (int): number of attention heads (only used if network_type='transformer')
            num_layers (int): number of transformer layers (only used if network_type='transformer')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.device = device
        self.network_type = network_type

        # Initialize Q-Networks (local and target)
        if network_type == 'transformer':
            print(f"Using Transformer Q-Network with {nhead} attention heads and {num_layers} layers")
            self.qnetwork_local = TransformerQNetwork(
                state_dim, action_dim, hidden_dim, nhead, num_layers
            ).to(self.device)
            self.qnetwork_target = TransformerQNetwork(
                state_dim, action_dim, hidden_dim, nhead, num_layers
            ).to(self.device)
        elif network_type == 'fat_mlp':
            print(f"Using Fat MLP Network with hidden dim {hidden_dim}")
            self.qnetwork_local = FatMLP(state_dim, action_dim, hidden_dim).to(self.device)
            self.qnetwork_target = FatMLP(state_dim, action_dim, hidden_dim).to(self.device)
        else:  # Default to standard MLP
            print(f"Using Standard MLP Q-Network with hidden dim {hidden_dim}")
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

                q_values = self.qnetwork_local(state) ## time bottleneck
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

# Agent module exports these classes
__all__ = ['DQN', 'QNetwork', 'FatMLP', 'TransformerQNetwork', 'ReplayBuffer', 'Experience']
