import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import math
from collections import deque, namedtuple

# Define Experience tuple for replay buffer
Experience = namedtuple('Experience', 
                        field_names=['state_seq', 'action_seq', 'reward', 'next_state_seq', 'done'])

class PositionalEncoding(nn.Module):
    """Positional encoding for the transformer model."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
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

def generate_square_subsequent_mask(sz):
    """Generate a causal mask for the transformer decoder.
    
    Each position can only attend to previous positions.
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class SequentialTransformerQNetwork(nn.Module):
    """Transformer-based Q-Network that processes sequences of states and predicts Q-values 
    for the next k actions in the sequence.
    
    This adapts the next-token prediction paradigm from language models to action prediction
    in network optimization.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, nhead=4, num_layers=3, 
                 dropout=0.1, max_seq_length=100, prediction_horizon=5):
        super().__init__()
        
        # Force hidden_dim to be 256 to avoid dimension mismatches
        hidden_dim = 256
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.prediction_horizon = prediction_horizon  # How many steps ahead to predict
        self.max_seq_length = max_seq_length
        
        # Input embeddings - encoding state features and action history
        self.state_embedding = nn.Linear(1, hidden_dim // 2)  # For state features
        self.action_embedding = nn.Embedding(action_dim, hidden_dim // 2)  # For action history
        
        # Merge state and action embeddings
        self.fusion_layer = nn.Linear(hidden_dim, hidden_dim)
        
        # Position encodings
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output projection for Q-values of next actions
        self.output = nn.Linear(hidden_dim, action_dim)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, state_sequence, action_history=None):
        """
        Process a sequence of states and past actions to output Q-values for the next action
        
        Args:
            state_sequence: tensor of shape [batch_size, seq_len, state_dim]
                           containing sequence of environment states
            action_history: tensor of shape [batch_size, seq_len-1]
                           containing previous actions (can be None if this is the first step)
        
        Returns:
            Q-values for the next action: tensor of shape [batch_size, action_dim]
        """
        try:
            # Convert to tensor if not already
            if not isinstance(state_sequence, torch.Tensor):
                state_sequence = torch.FloatTensor(state_sequence)
            
            # Check dimensions
            if state_sequence.dim() == 2:  # [batch_size, state_dim]
                state_sequence = state_sequence.unsqueeze(1)  # Add sequence dimension [batch_size, 1, state_dim]
                if action_history is not None and action_history.dim() == 1:
                    action_history = action_history.unsqueeze(1)  # [batch_size, 1]
            
            # Get batch_size and seq_len safely with explicit checks
            if state_sequence.dim() < 3:
                print(f"Error: state_sequence dimensions are invalid: {state_sequence.shape}")
                return torch.zeros(1, self.action_dim, device=state_sequence.device)
                
            batch_size, seq_len = state_sequence.shape[0], state_sequence.shape[1]
            
            # Safety check on state_sequence dimensions
            if state_sequence.shape[2] > 200:
                print(f"Warning: Unusually large state dimension: {state_sequence.shape[2]}, clipping to 200")
                # Clip to first 200 dimensions to avoid memory issues
                state_sequence = state_sequence[:, :, :200]
                
            # Check if action_history is valid when provided
            if action_history is not None:
                if isinstance(action_history, np.ndarray):
                    action_history = torch.tensor(action_history, dtype=torch.long, device=state_sequence.device)
                # Ensure action indices are within valid range
                if torch.any(action_history >= self.action_dim):
                    print(f"Warning: Action history contains invalid actions, clipping to valid range")
                    action_history = torch.clamp(action_history, 0, self.action_dim - 1)
            
            # Reshape state sequence to have feature dimension of 1
            # [batch_size, seq_len, state_dim] -> [batch_size, seq_len, state_dim, 1]
            # Then process each state element separately
            embedded_states = []
            
            # Use a fixed-size hidden dimension to ensure consistency
            embedding_dim = self.hidden_dim // 2
            
            # Process each state feature and embed it
            state_dim = state_sequence.shape[2]
            for i in range(state_dim):
                try:
                    feature = state_sequence[:, :, i].unsqueeze(-1)  # [batch_size, seq_len, 1]
                    embedded_feature = self.state_embedding(feature)  # [batch_size, seq_len, embedding_dim]
                    embedded_states.append(embedded_feature)
                except Exception as e:
                    print(f"Error processing feature {i}: {str(e)}")
            
            # Safety check - make sure we have at least one embedded state
            if not embedded_states:
                # If we have no valid embedded states, create a zero tensor as fallback
                print("Warning: No valid embedded states found, using zeros")
                return torch.zeros(batch_size, self.action_dim, device=state_sequence.device)
            
            # Combine embedded state features (average them)
            state_embeddings = torch.stack(embedded_states, dim=0).mean(dim=0)  # [batch_size, seq_len, embedding_dim]
            
            # Create zero tensor of same size as state embeddings in case action history is None
            zero_embeddings = torch.zeros(
                batch_size, seq_len, embedding_dim,
                device=state_sequence.device
            )
            
            if action_history is None or seq_len == 1:
                # Just use state embeddings, expanded to match the expected hidden dim
                combined = torch.cat([state_embeddings, zero_embeddings], dim=-1)  # [batch_size, seq_len, hidden_dim]
            else:
                # Convert to tensor if not already
                if not isinstance(action_history, torch.Tensor):
                    action_history = torch.LongTensor(action_history).to(state_sequence.device)
                    
                # Embed action history
                action_embeddings = self.action_embedding(action_history)  # [batch_size, seq_len-1, embedding_dim]
                
                # Pad action embeddings to match state sequence length
                padded_action_embeddings = torch.zeros(
                    batch_size, seq_len, embedding_dim, 
                    device=action_embeddings.device
                )
                
                # Shift action embeddings by 1 to align with next states
                padded_action_embeddings[:, 1:, :] = action_embeddings
                
                # Make sure the dimensions match exactly before concatenating
                if padded_action_embeddings.shape[-1] != state_embeddings.shape[-1]:
                    print(f"Warning: Dimension mismatch between state_embeddings ({state_embeddings.shape[-1]}) and "
                          f"action_embeddings ({padded_action_embeddings.shape[-1]})")
                    
                    # Force both tensors to have exactly embedding_dim size
                    # For state embeddings:
                    state_embeddings_resized = torch.zeros(
                        batch_size, seq_len, embedding_dim,
                        device=state_embeddings.device
                    )
                    # Copy as much data as possible from original tensor
                    min_dim = min(state_embeddings.shape[-1], embedding_dim)
                    state_embeddings_resized[:, :, :min_dim] = state_embeddings[:, :, :min_dim]
                    state_embeddings = state_embeddings_resized
                    
                    # For action embeddings:
                    action_embeddings_resized = torch.zeros(
                        batch_size, seq_len, embedding_dim,
                        device=padded_action_embeddings.device
                    )
                    # Copy as much data as possible from original tensor
                    min_dim = min(padded_action_embeddings.shape[-1], embedding_dim)
                    action_embeddings_resized[:, :, :min_dim] = padded_action_embeddings[:, :, :min_dim]
                    padded_action_embeddings = action_embeddings_resized
                
                # Combined state and action embeddings
                combined = torch.cat([state_embeddings, padded_action_embeddings], dim=-1)  # [batch_size, seq_len, hidden_dim]
            
            # Fuse the state and action information
            x = self.fusion_layer(combined)
            
            # Add positional encoding
            x = self.pos_encoder(x)
            
            # Create causal self-attention mask (each position can only attend to previous positions)
            mask = generate_square_subsequent_mask(seq_len).to(x.device)
            
            # Apply transformer encoder with causal mask
            x = self.transformer_encoder(x, mask=mask)
            
            # Extract the last sequence position for predicting the next action
            x = x[:, -1, :]  # [batch_size, hidden_dim]
            
            # Apply layer normalization and dropout
            x = self.layer_norm(x)
            x = self.dropout(x)
            
            # Project to get Q-values for the next action
            q_values = self.output(x)  # [batch_size, action_dim]
            
            return q_values
            
        except Exception as e:
            # Catch any unexpected errors and return a safe default output
            print(f"Error in forward pass: {str(e)}")
            device = state_sequence.device if isinstance(state_sequence, torch.Tensor) else torch.device('cpu')
            if state_sequence is not None and isinstance(state_sequence, torch.Tensor) and state_sequence.dim() > 0:
                batch_size = state_sequence.shape[0]
                return torch.zeros(batch_size, self.action_dim, device=device)
            else:
                # Fallback when we can't determine the batch size
                return torch.zeros(1, self.action_dim, device=device)
    
    def predict_multiple_steps(self, state_sequence, action_history=None, steps=None):
        """
        Predict Q-values for multiple steps ahead using autoregressive prediction.
        
        This simulates the model making sequential decisions, using its own predictions.
        
        Args:
            state_sequence: Initial state sequence [batch_size, seq_len, state_dim]
            action_history: Initial action history [batch_size, seq_len-1] or None
            steps: Number of steps to predict (defaults to self.prediction_horizon)
            
        Returns:
            List of tensors containing Q-values for each of the predicted steps
        """
        if steps is None:
            steps = self.prediction_horizon
            
        predicted_q_values = []
        current_state_seq = state_sequence
        current_action_history = action_history
        
        # Make autoregressive predictions
        for _ in range(steps):
            # Predict Q-values for the next action
            q_values = self.forward(current_state_seq, current_action_history)
            predicted_q_values.append(q_values)
            
            # Determine the greedy action (for simulation purposes)
            predicted_action = q_values.argmax(dim=-1)  # [batch_size]
            
            if current_action_history is None:
                current_action_history = predicted_action.unsqueeze(-1)  # [batch_size, 1]
            else:
                current_action_history = torch.cat(
                    [current_action_history, predicted_action.unsqueeze(-1)], 
                    dim=1
                )
                
            # In a real environment, we would get the next state here
            # For simulation/planning purposes, we would need an environment model
                
        return predicted_q_values

class SequentialReplayBuffer:
    """Replay buffer for storing and sampling sequences of experiences."""
    
    def __init__(self, buffer_size, sequence_length, batch_size, device):
        """Initialize a ReplayBuffer object.
        
        Args:
            buffer_size (int): maximum size of buffer
            sequence_length (int): length of each sequence to store
            batch_size (int): size of each training batch
            device: torch device
        """
        self.memory = deque(maxlen=buffer_size)
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.device = device
        self.experience = Experience
        
    def push(self, state_seq, action_seq, reward, next_state_seq, done):
        """Add a new experience to memory."""
        e = self.experience(state_seq, action_seq, reward, next_state_seq, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=min(self.batch_size, len(self)))
        
        # Process states (with consistent shapes)
        try:
            states = torch.FloatTensor(np.array([e.state_seq for e in experiences if e is not None])).to(self.device)
        except ValueError as e:
            print(f"Error processing state sequences: {e}")
            # Process them individually and pad to consistent shape
            state_seqs = [e.state_seq for e in experiences if e is not None]
            max_len = max(len(seq) for seq in state_seqs)
            # Find the first valid shape
            first_valid_shape = None
            for seq in state_seqs:
                if isinstance(seq[0], np.ndarray):
                    first_valid_shape = seq[0].shape
                    break
            
            if first_valid_shape is None:
                raise ValueError("Could not find any valid state shapes in batch")
            
            # Create a padded batch with consistent shapes
            padded_states = np.zeros((len(state_seqs), max_len) + first_valid_shape)
            for i, seq in enumerate(state_seqs):
                for j, state in enumerate(seq[:max_len]):
                    if isinstance(state, np.ndarray) and state.shape == first_valid_shape:
                        padded_states[i, j] = state
                    else:
                        # Resize to match first valid shape
                        try:
                            padded_states[i, j] = np.resize(np.array(state), first_valid_shape)
                        except:
                            # Just use zeros if resize fails
                            pass
            
            states = torch.FloatTensor(padded_states).to(self.device)
        
        # Process action sequences (with consistent lengths)
        try:
            # Get action sequences and find the maximum length
            action_seqs = [e.action_seq for e in experiences if e is not None]
            if not action_seqs or len(action_seqs) == 0:
                return None  # No valid actions in batch
                
            # Pad all sequences to the same length
            max_action_len = max(len(seq) for seq in action_seqs)
            padded_actions = np.zeros((len(action_seqs), max_action_len), dtype=np.int64)
            
            for i, seq in enumerate(action_seqs):
                padded_actions[i, :len(seq)] = seq[:max_action_len]
                
            actions = torch.LongTensor(padded_actions).to(self.device)
        except Exception as e:
            print(f"Error processing action sequences: {e}")
            # Simple fallback - create a batch of zeros (or return None)
            return None
        
        # Other tensors are simpler
        rewards = torch.FloatTensor(np.array([e.reward for e in experiences if e is not None])).to(self.device)
        
        # Process next states (similar to states)
        try:
            next_states = torch.FloatTensor(np.array([e.next_state_seq for e in experiences if e is not None])).to(self.device)
        except ValueError:
            # Use the same approach as for states
            next_state_seqs = [e.next_state_seq for e in experiences if e is not None]
            # Using the same shape as states for consistency
            padded_next_states = np.zeros_like(padded_states)
            for i, seq in enumerate(next_state_seqs):
                for j, state in enumerate(seq[:max_len]):
                    if isinstance(state, np.ndarray) and state.shape == first_valid_shape:
                        padded_next_states[i, j] = state
                    else:
                        try:
                            padded_next_states[i, j] = np.resize(np.array(state), first_valid_shape)
                        except:
                            pass
            
            next_states = torch.FloatTensor(padded_next_states).to(self.device)
            
        dones = torch.FloatTensor(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).to(self.device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class SequentialDQNAgent:
    """DQN Agent that uses a sequential transformer for decision making."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, nhead=4, num_layers=3,
                 dropout=0.1, sequence_length=10, prediction_horizon=5,
                 learning_rate=1e-4, gamma=0.99, device="cpu"):
        """Initialize an Agent object.
        
        Args:
            state_dim (int): Dimension of each state
            action_dim (int): Dimension of each action
            hidden_dim (int): Hidden dimension of the transformer
            nhead (int): Number of heads in multi-head attention
            num_layers (int): Number of transformer layers
            dropout (float): Dropout rate
            sequence_length (int): Length of sequences to process
            prediction_horizon (int): Number of steps ahead to predict
            learning_rate (float): Learning rate for optimizer
            gamma (float): Discount factor
            device (str): Device to run the model on ('cpu' or 'cuda')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.gamma = gamma
        self.device = device
        
        # Q-Networks
        self.qnetwork_local = SequentialTransformerQNetwork(
            state_dim, action_dim, hidden_dim, nhead, num_layers, 
            dropout, sequence_length, prediction_horizon
        ).to(device)
        
        self.qnetwork_target = SequentialTransformerQNetwork(
            state_dim, action_dim, hidden_dim, nhead, num_layers, 
            dropout, sequence_length, prediction_horizon
        ).to(device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        
        # Initialize sequence buffers
        self.state_buffer = deque(maxlen=sequence_length)
        self.action_buffer = deque(maxlen=sequence_length-1)
        self.initialized = False
        
    def update_buffers(self, state, action=None):
        """Update the internal state and action buffers with new data."""
        self.state_buffer.append(state)
        if action is not None:
            self.action_buffer.append(action)
            
        # Check if buffers are initialized
        if len(self.state_buffer) == self.sequence_length:
            self.initialized = True
            
    def get_current_sequences(self):
        """Get the current state sequence and action history from buffers."""
        if not self.initialized:
            # If not initialized, pad with zeros
            state_seq = list(self.state_buffer)
            if not state_seq:  # If buffer is empty
                return None, None
            
            # Find first valid state to use as a template
            valid_state = None
            for s in state_seq:
                if isinstance(s, (np.ndarray, list)) and len(s) > 0:
                    valid_state = np.array(s, dtype=np.float32)
                    break
            
            # If no valid state found, create a zero state with proper dimensions
            if valid_state is None:
                valid_state = np.zeros(self.state_dim, dtype=np.float32)
                
            # Pad with zeros based on valid state
            padding = [np.zeros_like(valid_state)] * (self.sequence_length - len(state_seq))
            state_seq = padding + state_seq
            
            action_history = None  # No actions yet
        else:
            state_seq = list(self.state_buffer)
            action_history = list(self.action_buffer) if self.action_buffer else None
            
            if action_history:
                action_history = torch.LongTensor(action_history).to(self.device)
        
        # Convert all states to numpy arrays of the same shape and ensure none are empty
        for i in range(len(state_seq)):
            if not isinstance(state_seq[i], np.ndarray) or len(state_seq[i]) == 0:
                # Replace empty or invalid states with zeros of correct dimension
                state_seq[i] = np.zeros(self.state_dim, dtype=np.float32)
        
        # Stack states manually to ensure consistent shape
        try:
            # Ensure all states have the correct dimension
            for i in range(len(state_seq)):
                if not isinstance(state_seq[i], np.ndarray) or state_seq[i].shape != (self.state_dim,):
                    # Fix any state with wrong shape
                    state_seq[i] = np.zeros(self.state_dim, dtype=np.float32)
            
            state_array = np.stack(state_seq)
            state_seq = torch.FloatTensor(state_array).to(self.device)
        except ValueError as e:
            # If shapes are still inconsistent, handle more aggressively
            shapes = [getattr(s, 'shape', None) for s in state_seq]
            print(f"Warning: Inconsistent state shapes in buffer: {shapes}")
            print(f"Error details: {str(e)}")
            
            # Create a completely new array with correct dimensions
            state_array = np.zeros((len(state_seq), self.state_dim), dtype=np.float32)
            # Copy what data we can
            for i, state in enumerate(state_seq):
                if isinstance(state, np.ndarray) and state.size > 0:
                    # Copy as many elements as we can, up to state_dim
                    state_flat = state.flatten()
                    copy_len = min(len(state_flat), self.state_dim)
                    state_array[i, :copy_len] = state_flat[:copy_len]
            
            state_seq = torch.FloatTensor(state_array).to(self.device)
        
        # Add batch dimension if not present
        if state_seq.dim() == 2:
            state_seq = state_seq.unsqueeze(0)
        if action_history is not None and action_history.dim() == 1:
            action_history = action_history.unsqueeze(0)
        return state_seq, action_history

    def select_action(self, state, epsilon=0.0):
        """Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            epsilon: Probability of selecting a random action
            
        Returns:
            Selected action
        """
        # Update state buffer with new state
        self.update_buffers(state)
        
        # Get current sequences
        state_seq, action_history = self.get_current_sequences()
        
        # Epsilon-greedy action selection
        if random.random() > epsilon:
            try:
                # Get Q-values from network
                with torch.no_grad():
                    q_values = self.qnetwork_local(state_seq, action_history)
                    
                # Get Q-values and validate they're not NaN or infinite
                q_values_np = q_values.cpu().data.numpy()
                if np.any(np.isnan(q_values_np)) or np.any(np.isinf(q_values_np)):
                    print(f"Warning: Q-values contain NaN or Inf, using random action")
                    action = random.randint(0, self.action_dim - 1)
                else:
                    # Make sure argmax is safe by clipping within bounds of action dimension
                    max_index = np.argmax(q_values_np)
                    action = int(max_index) if max_index < self.action_dim else 0
                    
                # Double-check action is within valid range
                if action >= self.action_dim or action < 0:
                    print(f"Warning: Invalid action {action}, clipping to valid range")
                    action = max(0, min(action, self.action_dim - 1))  # Clip to valid range
            except Exception as e:
                # Fallback to random action if there's any error
                print(f"Error in action selection: {str(e)}, using random action")
                action = random.randint(0, self.action_dim - 1)
        else:
            # Select random action
            action = random.randint(0, self.action_dim - 1)
            
        # Update action buffer with selected action
        self.update_buffers(None, action)
        
        return action
        
    def learn(self, replay_buffer, batch_size):
        """Update value parameters using given batch of experience tuples.

        Args:
            replay_buffer: Replay buffer containing experiences
            batch_size: Batch size for training

        Returns:
            Loss value or 0.0 if learning is skipped
        """
        # Sample a batch of experiences
        if len(replay_buffer) < batch_size:
            return 0.0
        
        try:    
            # Get batch of experiences
            batch = replay_buffer.sample()
            if batch is None:
                # Skip this learning step if we couldn't get a valid batch
                return 0.0
                
            states, actions, rewards, next_states, dones = batch
            
            # Safety check on tensor shapes
            if actions.size(0) == 0 or actions.size(1) <= 1:
                # Need at least one past action and one current action
                return 0.0
            
            # Get Q-values for current states and actions
            # Use all but the last action as history for current states
            try:
                q_values = self.qnetwork_local(states, actions[:, :-1])
            except Exception as e:
                print(f"Error in qnetwork_local forward pass: {str(e)}")
                return 0.0
                
            # Use all but the first action as history for next states
            try:    
                q_targets_next = self.qnetwork_target(next_states, actions[:, 1:])
            except Exception as e:
                print(f"Error in qnetwork_target forward pass: {str(e)}")
                return 0.0
            
            # Get max predicted Q-values for next states
            try:
                # Handle potential dimension issues in Q-value calculation
                q_targets_next_max = q_targets_next.detach().max(1)[0].unsqueeze(1)
                
                # Compute Q targets for current states
                q_targets = rewards.unsqueeze(1) + (self.gamma * q_targets_next_max * (1 - dones.unsqueeze(1)))
                
                # Get expected Q-values for chosen actions (use last action in sequence)
                # Ensure actions are within valid range
                last_actions = actions[:, -1].clone()
                # Clip any out-of-range actions
                if (last_actions >= self.action_dim).any():
                    print(f"Warning: Clipping out-of-range actions for loss calculation")
                    last_actions = torch.clamp(last_actions, 0, self.action_dim - 1)
                
                q_expected = q_values.gather(1, last_actions.unsqueeze(1))
                
                # Compute loss
                loss = F.mse_loss(q_expected, q_targets)
                
                # Minimize the loss
                self.optimizer.zero_grad()
                loss.backward()
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
                self.optimizer.step()
                
                return loss.item()
            except Exception as e:
                print(f"Error in Q-value calculation: {str(e)}")
                return 0.0
                
        except Exception as e:
            print(f"Error during learning step: {str(e)}")
            return 0.0
    
    def update_target_network(self):
        """Update target network with weights from local network."""
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        
    def reset(self):
        """Reset the agent's internal buffers."""
        self.state_buffer.clear()
        self.action_buffer.clear()
        self.initialized = False
        
    def save(self, path):
        """Save the local Q-network to a file."""
        torch.save(self.qnetwork_local.state_dict(), path)
        
    def load(self, path):
        """Load a saved model into the local Q-network and copy to target."""
        self.qnetwork_local.load_state_dict(torch.load(path, map_location=self.device))
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        
# Helper function to maintain a sequence of states for environment interaction
class SequenceTracker:
    """Helper class to maintain sequences of states and actions for environment interaction."""
    
    def __init__(self, sequence_length, state_dim):
        """Initialize a SequenceTracker object.
        
        Args:
            sequence_length (int): Length of sequences to maintain
            state_dim (int): Dimension of each state
        """
        self.sequence_length = sequence_length
        self.state_dim = state_dim
        self.reset()
        
    def reset(self):
        """Reset the sequence buffers."""
        self.state_buffer = deque(maxlen=self.sequence_length)
        self.action_buffer = deque(maxlen=self.sequence_length-1)
        
    def update(self, state, action=None):
        """Update buffers with new state and action."""
        if state is not None:
            self.state_buffer.append(state)
        if action is not None:
            self.action_buffer.append(action)
            
    def get_state_sequence(self):
        """Get the current state sequence, padded if necessary."""
        state_seq = list(self.state_buffer)
        
        if not state_seq:
            # Return empty array with correct shape if buffer is empty
            return np.zeros((self.sequence_length, self.state_dim))
        
        # Find first valid state with proper dimensions
        valid_state = None
        for s in state_seq:
            if isinstance(s, (np.ndarray, list)) and len(s) > 0:
                try:
                    valid_state = np.array(s, dtype=np.float32)
                    if valid_state.size == self.state_dim:
                        break
                except:
                    pass
        
        # If no valid state found, create a zero state with proper dimensions
        if valid_state is None or valid_state.shape != (self.state_dim,):
            valid_state = np.zeros(self.state_dim, dtype=np.float32)
            
        # Ensure all states are numpy arrays with consistent shape
        for i in range(len(state_seq)):
            if not isinstance(state_seq[i], np.ndarray) or state_seq[i].shape != (self.state_dim,):
                # Replace invalid states with valid template
                state_seq[i] = valid_state.copy()
        
        # Pad with zeros if buffer is not full
        if len(state_seq) < self.sequence_length:
            padding = [valid_state.copy() for _ in range(self.sequence_length - len(state_seq))]
            state_seq = padding + state_seq
        
        try:
            # Try stacking the states with the guaranteed correct shapes
            return np.stack(state_seq)
        except ValueError as e:
            # If there's still an issue (which shouldn't happen now), create a clean array
            print(f"Warning: Failed to stack states. Error: {e}")
            
            # Create a completely new array with correct dimensions
            result = np.zeros((self.sequence_length, self.state_dim), dtype=np.float32)
            # Copy what data we can from the original states
            for i, state in enumerate(state_seq):
                if isinstance(state, np.ndarray) and state.size > 0:
                    # Safely copy data
                    state_flat = state.flatten()
                    copy_len = min(len(state_flat), self.state_dim)
                    result[i, :copy_len] = state_flat[:copy_len]
            
            return result
    
    def get_action_sequence(self):
        """Get the current action sequence, or None if no actions."""
        if not self.action_buffer:
            return None
        return np.array(list(self.action_buffer))
    
    def get_current_sequences(self):
        """Get both state and action sequences."""
        return self.get_state_sequence(), self.get_action_sequence()
