"""
Traffic Matrix Representation Learning Module

This module implements a neural network encoder for traffic matrices
that helps the model learn to understand the "structure" of traffic patterns
rather than treating each traffic matrix as unique.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TrafficMatrixEncoder(nn.Module):
    """Encoder network that extracts meaningful features from traffic matrices"""
    def __init__(self, num_nodes, embedding_dim=64):
        super(TrafficMatrixEncoder, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        
        # Input dimension is the flattened traffic matrix (num_nodes x num_nodes)
        input_dim = num_nodes * num_nodes
        
        # Encoder layers
        self.fc1 = nn.Linear(input_dim, embedding_dim * 2)
        self.ln1 = nn.LayerNorm(embedding_dim * 2)
        self.dropout1 = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        
    def forward(self, traffic_matrix):
        """
        Encode a traffic matrix into a fixed-size embedding vector
        
        Args:
            traffic_matrix: Traffic matrix of shape [batch_size, num_nodes, num_nodes]
                            or [num_nodes, num_nodes] for a single matrix
        
        Returns:
            embedding: Encoded representation of the traffic matrix
        """
        # Convert to tensor if needed
        if not isinstance(traffic_matrix, torch.Tensor):
            traffic_matrix = torch.FloatTensor(traffic_matrix)
            
        # Handle both batched and unbatched inputs
        original_shape = traffic_matrix.shape
        is_batched = len(original_shape) == 3
        
        if not is_batched:
            # Add batch dimension if not already present
            traffic_matrix = traffic_matrix.unsqueeze(0)
        
        batch_size = traffic_matrix.shape[0]
        
        # Flatten the traffic matrix
        x = traffic_matrix.view(batch_size, -1)
        
        # Apply encoding layers
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.ln2(self.fc2(x)))
        
        # Remove batch dimension if input wasn't batched
        if not is_batched:
            x = x.squeeze(0)
            
        return x

class EnhancedQNetwork(nn.Module):
    """
    Enhanced Q-Network with traffic matrix encoding capabilities
    This model takes both the network state and traffic matrix as inputs
    """
    def __init__(self, state_dim, action_dim, num_nodes, hidden_dim=256, tm_embedding_dim=64):
        super(EnhancedQNetwork, self).__init__()
        
        # Traffic matrix encoder
        self.tm_encoder = TrafficMatrixEncoder(num_nodes, tm_embedding_dim)
        
        # Main network layers with layer normalization and dropout
        self.fc1 = nn.Linear(state_dim + tm_embedding_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.2)
        
        # Deeper architecture with multiple hidden layers
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.ln2 = nn.LayerNorm(hidden_dim*2)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.dropout3 = nn.Dropout(0.2)
        
        # Output layer
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state, traffic_matrix):
        """
        Build a network that maps (state, traffic_matrix) -> action values
        
        Args:
            state: Network state tensor of shape [batch_size, state_dim] or [state_dim]
            traffic_matrix: Traffic matrix tensor of shape [batch_size, num_nodes, num_nodes]
                           or [num_nodes, num_nodes]
        
        Returns:
            Q-values for each action
        """
        # Convert inputs to tensors if they're not already
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if not isinstance(traffic_matrix, torch.Tensor):
            traffic_matrix = torch.FloatTensor(traffic_matrix)
            
        # Handle both batched and unbatched inputs
        is_single_state = state.dim() == 1
        is_single_tm = traffic_matrix.dim() == 2
        
        # Add batch dimension if needed
        if is_single_state:
            state = state.unsqueeze(0)  # Add batch dimension
        if is_single_tm:
            traffic_matrix = traffic_matrix.unsqueeze(0)
                
        # Encode the traffic matrix
        tm_embedding = self.tm_encoder(traffic_matrix)
        
        # Ensure tm_embedding has batch dimension
        if tm_embedding.dim() == 1:
            tm_embedding = tm_embedding.unsqueeze(0)
        
        # Concatenate state and traffic matrix embedding
        combined_input = torch.cat([state, tm_embedding], dim=1)
        
        # Apply layers with activations, layer norm, and dropout
        x = F.relu(self.ln1(self.fc1(combined_input)))
        x = self.dropout1(x)
        
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.dropout3(x)
        
        # Output layer without activation (Q-values can be any real number)
        x = self.fc4(x)
        
        # Remove batch dimension if input was a single state
        if is_single_state:
            x = x.squeeze(0)
            
        return x
