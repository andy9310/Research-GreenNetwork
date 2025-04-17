import sys
import os
import torch
import numpy as np
import json
import argparse
import time

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import directly from Qlearning directory
from Qlearning.env import NetworkEnv
from Qlearning.agent import DQN, TransformerQNetwork, QNetwork, FatMLP  # Import all model types

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained DQN agent on testing configurations')
    parser.add_argument('--config', type=str, required=True, help='Path to testing configuration JSON file')
    parser.add_argument('--tm-index', type=int, default=0, help='Index of traffic matrix to use (default: 0)')
    parser.add_argument('--gpu', action='store_true', help='Force using GPU if available')
    parser.add_argument('--model-path', type=str, default=None, help='Path to the trained model')
    parser.add_argument('--model-type', type=str, default='mlp', choices=['mlp', 'transformer', 'fat_mlp'],
                        help='Model architecture type (default: mlp)')
    parser.add_argument('--episodes', type=int, default=5, help='Number of evaluation episodes')
    args = parser.parse_args()
    
    # Load testing configuration
    config_path = args.config
    print(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    # Load environment parameters from testing config
    num_nodes = config["num_nodes"]
    adj_matrix = config["adj_matrix"]
    edge_list = config["edge_list"]
    node_props = config["node_props"] if "node_props" in config else None
    tm_list = config["tm_list"]
    link_capacity = config["link_capacity"]
    max_edges = config["max_edges"]
    
    # Analyze traffic matrix for potential capacity adjustment
    tm_index = args.tm_index
    if tm_index < 0 or tm_index >= len(tm_list):
        print(f"Error: Traffic matrix index {tm_index} is out of range (0-{len(tm_list)-1})")
        print(f"Defaulting to index 0")
        tm_index = 0
    
    tm = np.array(tm_list[tm_index])
    total_traffic = np.sum(tm)
    avg_traffic_per_edge = total_traffic / len(edge_list) * 2  # Conservative estimate
    
    effective_link_capacity = link_capacity
    if avg_traffic_per_edge > link_capacity:
        suggested_capacity = int(avg_traffic_per_edge * 1.5)  # Add 50% margin
        print(f"WARNING: Traffic matrix might require higher capacity.")
        print(f"  - Current capacity: {link_capacity}")
        print(f"  - Suggested minimum capacity: {suggested_capacity}")
        print(f"  - Using adjusted capacity: {suggested_capacity}")
        effective_link_capacity = suggested_capacity
    
    # Use a different seed for evaluation
    eval_seed = int(time.time())
    
    # Create environment with the test configuration
    env_kwargs = {
        "adj_matrix": adj_matrix,
        "edge_list": edge_list,
        "tm_list": tm_list,
        "num_nodes": num_nodes,
        "link_capacity": effective_link_capacity,
        "max_edges": max_edges,
        "seed": eval_seed
    }
    
    if node_props is not None:
        env_kwargs["node_props"] = node_props
        
    env = NetworkEnv(**env_kwargs)
    
    # Set current traffic matrix index
    try:
        env.current_tm_idx = tm_index
    except:
        print("Note: Could not set tm_index directly, will be set during reset")
    
    print(f"Evaluating using traffic matrix index {tm_index} (of {len(tm_list)} matrices)")
    
    # Device configuration
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU for evaluation")
    
    # Get state and action dimensions from environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_dim = 256  # Standard hidden dimension
    
    print(f"Environment setup:")
    print(f"- Topology: {len(edge_list)} edges (max_edges={max_edges})")
    print(f"- State Dimension: {state_dim}")
    print(f"- Action Dimension: {action_dim}")
    
    # Determine model path based on config
    if args.model_path:
        model_load_path = args.model_path
    else:
        # Extract config name from path
        config_name = os.path.basename(config_path).replace('.json', '')
        if config_name.startswith("test_"):
            training_config = config_name.replace("test_", "", 1)
        else:
            training_config = config_name
            
        # Look for models with this config name
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 "Qlearning", "models")
        
        # Try different model name patterns
        model_patterns = [
            f"dqn_{args.model_type}_{training_config}.pth",  # e.g., dqn_mlp_config_5node.pth
            f"dqn_model_{training_config}.pth",             # e.g., dqn_model_config_5node.pth
        ]
        
        model_load_path = None
        for pattern in model_patterns:
            potential_path = os.path.join(models_dir, pattern)
            if os.path.exists(potential_path):
                model_load_path = potential_path
                print(f"Found model at {model_load_path}")
                break
                
        if model_load_path is None:
            print("No matching model found. Available models:")
            if os.path.exists(models_dir):
                for file in os.listdir(models_dir):
                    if file.endswith(".pth"):
                        print(f"  - {file}")
                # Default to first .pth file
                pth_files = [f for f in os.listdir(models_dir) if f.endswith(".pth")]
                if pth_files:
                    model_load_path = os.path.join(models_dir, pth_files[0])
                    print(f"Using first available model: {pth_files[0]}")
                else:
                    print("Error: No .pth model files found")
                    return
            else:
                print("Error: Models directory not found")
                return
    
    # Initialize the network based on model type
    if "mlp" in model_load_path.lower():
        if "fat" in model_load_path.lower():
            hidden_dim = 512  # Larger for fat_mlp
            policy_net = FatMLP(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
            print(f"Using Fat MLP network with hidden_dim={hidden_dim}")
        else:
            hidden_dim = 256  # Standard hidden dimension
            policy_net = QNetwork(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
            print(f"Using standard MLP network with hidden_dim={hidden_dim}")
    else:  # Default to transformer
        policy_net = TransformerQNetwork(state_dim, action_dim, hidden_dim=hidden_dim, 
                                        nhead=4, num_layers=2).to(device)
        print("Using Transformer network")
    
    print(f"\nLoading trained model from {model_load_path}...")
    try:
        policy_net.load_state_dict(torch.load(model_load_path, map_location=device))
        policy_net.eval()  # Set to evaluation mode
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model file exists and matches the network architecture.")
        if "mlp" in args.model_type and "transformer" in model_load_path.lower():
            print("Try specifying --model-type transformer instead")
        elif "transformer" in args.model_type and "mlp" in model_load_path.lower():
            print("Try specifying --model-type mlp instead")
        return
    
    # --- Evaluation Loop ---
    num_eval_episodes = args.episodes
    total_rewards = []
    final_link_configs = []
    violations_occurred = []
    
    print(f"\nRunning evaluation for {num_eval_episodes} episodes...")
    for episode in range(num_eval_episodes):
        # Reset environment
        state_result = env.reset()
        
        # Handle different return formats
        if isinstance(state_result, tuple):
            # If reset returns a tuple, the first element is usually the state
            state = state_result[0]
        else:
            state = state_result
            
        episode_reward = 0
        done = False
        step = 0
        episode_violations = {'isolated': 0, 'overloaded': 0}
        
        while not done and step < 50:  # Limit to 50 steps maximum
            step += 1
            
            # Convert state to tensor and get action from policy
            with torch.no_grad():
                if isinstance(state, np.ndarray):
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                else:
                    # Try to handle other state formats
                    try:
                        state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device)
                    except:
                        print(f"Error: Could not convert state to tensor. State type: {type(state)}")
                        break
                        
                q_values = policy_net(state_tensor)
                action = q_values.argmax(dim=1).item()
            
            # Execute action in environment
            step_result = env.step(action)
            
            # Handle different return formats
            if isinstance(step_result, tuple):
                if len(step_result) == 4:  # Standard gym format
                    next_state, reward, done, info = step_result
                elif len(step_result) == 5:  # New gymnasium format
                    next_state, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    print(f"Warning: Unexpected step result format with {len(step_result)} elements")
                    next_state, reward = step_result[0], step_result[1]
                    done = step > 10  # Assume done after 10 steps if format is unknown
                    info = {}
            else:
                next_state, reward = step_result, 0
                done = step > 10
                info = {}
            
            episode_reward += reward
            state = next_state
            
            # Track violations if they occur
            if isinstance(info, dict):
                if info.get('violation') == 'isolated':
                    episode_violations['isolated'] += 1
                elif info.get('violation') == 'overloaded':
                    episode_violations['overloaded'] += info.get('num_overloaded', 1)
        
        # Get final configuration
        try:
            final_config = env.link_open.copy() if hasattr(env, 'link_open') else None
            num_closed = sum(1 for link in final_config if link == 0) if final_config is not None else "unknown"
        except:
            final_config = None
            num_closed = "unknown"
            
        total_rewards.append(episode_reward)
        if final_config is not None:
            final_link_configs.append(final_config)
        violations_occurred.append(episode_violations)
        
        print(f" Episode {episode + 1}: Reward={episode_reward:.2f}, " +
              f"Links Closed={num_closed}/{len(edge_list) if final_config is not None else '?'}, " +
              f"Violations={episode_violations}")
    
    # --- Report Results ---
    print("\n--- Evaluation Summary ---")
    if total_rewards:
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        print(f" Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        
        # Analyze final configurations
        if final_link_configs:
            most_common = np.zeros(len(edge_list))
            for config in final_link_configs:
                most_common += config
            most_common = np.round(most_common / len(final_link_configs))
            print(f" Average Configuration (0=closed, 1=open): {most_common}")
            
            # Calculate the number of closed links
            avg_closed = len(edge_list) - np.sum(most_common)
            print(f" Average Links Closed: {avg_closed:.1f}/{len(edge_list)}")
        
        # Summarize violations
        total_iso = sum(v['isolated'] for v in violations_occurred)
        total_ovl = sum(v['overloaded'] for v in violations_occurred)
        episodes_with_violations = sum(1 for v in violations_occurred if v['isolated'] > 0 or v['overloaded'] > 0)
        print(f" Total Isolation Violations: {total_iso}")
        print(f" Total Overload Violations: {total_ovl}")
        print(f" Episodes with Violations: {episodes_with_violations}/{num_eval_episodes}")
        
        # Save evaluation results
        results = {
            "config": config_path,
            "tm_index": tm_index,
            "avg_reward": float(avg_reward),
            "std_reward": float(std_reward),
            "episodes_with_violations": episodes_with_violations
        }
        
        if final_link_configs:
            results["avg_links_closed"] = float(avg_closed)
            results["total_links"] = len(edge_list)
            results["isolation_violations"] = total_iso
            results["overload_violations"] = total_ovl
        
        # Compare with bruteforce results if available
        config_basename = os.path.basename(config_path).replace('.json', '')
        bruteforce_results_path = f"bruteforce_results_{config_basename}.json"
        
        if os.path.exists(bruteforce_results_path):
            try:
                with open(bruteforce_results_path, 'r') as f:
                    bruteforce_results = json.load(f)
                
                bf_result = bruteforce_results.get(str(tm_index))
                if bf_result:
                    print("\n--- Comparison with Bruteforce Optimal Solution ---")
                    print(f" Bruteforce Best Score: {bf_result['best_score']}")
                    print(f" Bruteforce Optimal Links Closed: {bf_result['num_links_closed']}/{bf_result['total_links']}")
                    
                    # Calculate relative performance
                    if bf_result['best_score'] > 0:
                        relative_performance = (avg_reward / bf_result['best_score']) * 100.0
                        print(f" Model Performance vs Optimal: {relative_performance:.2f}%")
                        results["optimal_score"] = bf_result['best_score']
                        results["optimal_links_closed"] = bf_result['num_links_closed']
                        results["relative_performance"] = float(relative_performance)
            except Exception as e:
                print(f"Error comparing with bruteforce results: {e}")
        
        # Save evaluation results
        results_file = f"evaluation_results_{config_basename}_tm{tm_index}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nEvaluation results saved to {results_file}")
    else:
        print("No episodes completed successfully.")

if __name__ == "__main__":
    main()
