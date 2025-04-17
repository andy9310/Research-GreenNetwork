import sys
import os
import torch
import numpy as np
import json
import argparse

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import directly from Qlearning directory
from Qlearning.env import NetworkEnv
from Qlearning.agent import TransformerQNetwork

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='Debug the NetworkEnv state representation')
    parser.add_argument('--config', type=str, required=True, help='Path to testing configuration JSON file')
    parser.add_argument('--tm-index', type=int, default=0, help='Index of traffic matrix to use (default: 0)')
    args = parser.parse_args()
    
    # Load testing configuration
    config_path = args.config
    print(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    # Load environment parameters from testing config
    num_nodes = config["num_nodes"]
    adj_matrix = config["adj_matrix"]
    edge_list = config["edge_list"]
    node_props = config["node_props"]
    tm_list = config["tm_list"]
    link_capacity = config["link_capacity"]
    max_edges = config["max_edges"]
    
    # Analyze traffic matrix for potential capacity adjustment
    tm_index = args.tm_index
    if tm_index < 0 or tm_index >= len(tm_list):
        print(f"Error: Traffic matrix index {tm_index} is out of range (0-{len(tm_list)-1})")
        tm_index = 0
    
    tm = np.array(tm_list[tm_index])
    total_traffic = np.sum(tm)
    avg_traffic_per_edge = total_traffic / len(edge_list) * 2
    
    effective_link_capacity = link_capacity
    if avg_traffic_per_edge > link_capacity:
        suggested_capacity = int(avg_traffic_per_edge * 1.5)
        print(f"WARNING: Traffic matrix requires higher capacity: using {suggested_capacity}")
        effective_link_capacity = suggested_capacity
    
    # Create environment with the test configuration
    env = NetworkEnv(
        adj_matrix=adj_matrix,
        edge_list=edge_list,
        tm_list=tm_list,
        node_props=node_props,
        num_nodes=num_nodes,
        link_capacity=effective_link_capacity,
        max_edges=max_edges,
        seed=42
    )
    
    env.current_tm_idx = tm_index
    print(f"Using traffic matrix index {tm_index} (of {len(tm_list)} matrices)")
    
    # Debug NetworkEnv state representation
    state = env.reset()
    print(f"\nState type: {type(state)}")
    
    # Handle tuple state format
    if isinstance(state, tuple):
        print(f"Tuple length: {len(state)}")
        for i, item in enumerate(state):
            print(f"  Item {i} type: {type(item)}")
            print(f"  Item {i} value: {item}")
            if isinstance(item, np.ndarray):
                print(f"  Item {i} shape: {item.shape}")
    elif hasattr(state, '__dict__'):
        print(f"State attributes: {dir(state)}")
    else:
        print(f"State value: {state}")
    
    print("\nEnvironment attributes:")
    env_attrs = [attr for attr in dir(env) if not attr.startswith('__')]
    print(f"Available attributes: {env_attrs}")
    
    # Check for common attributes
    common_attrs = ['link_open', 'link_loads', 'num_edges', 'action_space', 'observation_space']
    for attr in common_attrs:
        if hasattr(env, attr):
            value = getattr(env, attr)
            print(f"{attr}: {type(value)}")
            if isinstance(value, np.ndarray):
                print(f"  Shape: {value.shape}")
                print(f"  Value: {value}")
    
    # Create feature vector manually to see what the trained model expects
    try:
        print("\nTrying to create feature vector from environment state variables:")
        # If state is a tuple and first element is a numpy array,
        # use that directly as our feature vector
        if isinstance(state, tuple) and len(state) > 0 and isinstance(state[0], np.ndarray):
            feature_vector1 = state[0]
            print(f"Feature vector from state[0] shape: {feature_vector1.shape} - {feature_vector1}")
        
        # If env has link_open and other necessary attributes, try to create
        # a feature vector from those
        if hasattr(env, 'link_open'):
            try:
                if hasattr(env, 'link_loads') and hasattr(env, 'link_capacity'):
                    feature_vector2 = np.concatenate([
                        env.link_open,
                        env.link_loads.flatten() / env.link_capacity
                    ])
                    print(f"Feature vector from env attrs shape: {feature_vector2.shape} - {feature_vector2}")
                else:
                    print("Missing link_loads or link_capacity attributes")
                    
                # Another possible format with traffic matrix
                if hasattr(env, 'current_tm'):
                    tm_flatten = env.current_tm.flatten()
                    feature_vector3 = np.concatenate([
                        env.link_open,
                        tm_flatten[:min(len(tm_flatten), 5)]  # Include some TM elements (limit to 5)
                    ])
                    print(f"Feature vector with TM shape: {feature_vector3.shape} - {feature_vector3}")
            except Exception as e:
                print(f"Error creating feature vector from env attributes: {e}")
    except Exception as e:
        print(f"Error creating feature vector: {e}")
    
    # Try loading a model to see what input it expects
    try:
        # Find available models
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                "Qlearning", "models")
        print("\nAvailable models in Qlearning/models directory:")
        if os.path.exists(models_dir):
            models = [f for f in os.listdir(models_dir) if f.endswith(".pth")]
            for i, model_file in enumerate(models):
                print(f"  {i+1}. {model_file}")
            
            if models:
                # Try loading the first model
                model_path = os.path.join(models_dir, models[0])
                print(f"\nLoading model: {model_path}")
                
                # Get state dimensionality from env
                state_dim = env.observation_space.shape[0]
                action_dim = env.action_space.n
                hidden_dim = 256
                
                device = torch.device("cpu")
                policy_net = TransformerQNetwork(state_dim, action_dim, hidden_dim=hidden_dim, 
                                                nhead=4, num_layers=2).to(device)
                
                policy_net.load_state_dict(torch.load(model_path, map_location=device))
                policy_net.eval()
                print("Model loaded successfully")
                
                # Try different formats for getting action from model
                with torch.no_grad():
                    try:
                        print("\nTrying different state formats for model inference:")
                        
                        # Try all available feature vectors
                        available_vectors = [(name, var) for name, var in locals().items() 
                                           if name.startswith('feature_vector') and isinstance(var, np.ndarray)]
                        
                        for name, vector in available_vectors:
                            try:
                                print(f"\nTrying {name} for model inference:")
                                state_tensor = torch.FloatTensor(vector).unsqueeze(0).to(device)
                                print(f"Tensor shape: {state_tensor.shape}")
                                q_values = policy_net(state_tensor)
                                action = q_values.argmax(dim=1).item()
                                print(f"Action: {action}, Q-values: {q_values}")
                            except Exception as e:
                                print(f"Error with {name}: {e}")
                        
                        # If state is a tuple, try using state[0] directly
                        if isinstance(state, tuple) and len(state) > 0:
                            try:
                                print("\nTrying state[0] directly for model inference:")
                                first_item = state[0]
                                if isinstance(first_item, np.ndarray):
                                    state_tensor = torch.FloatTensor(first_item).unsqueeze(0).to(device)
                                    print(f"State tensor shape: {state_tensor.shape}")
                                    q_values = policy_net(state_tensor)
                                    action = q_values.argmax(dim=1).item()
                                    print(f"Action: {action}, Q-values: {q_values}")
                            except Exception as e:
                                print(f"Error with state[0]: {e}")
                        
                    except Exception as e:
                        print(f"Error during model inference: {e}")
        else:
            print("Models directory not found")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    print("\nDebug Complete")

if __name__ == "__main__":
    main()
