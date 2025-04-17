import sys
import os
import torch
import numpy as np
import time
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
    parser = argparse.ArgumentParser(description='Evaluate a trained DQN agent on testing configurations')
    parser.add_argument('--config', type=str, required=True, help='Path to testing configuration JSON file')
    parser.add_argument('--tm-index', type=int, default=0, help='Index of traffic matrix to use (default: 0)')
    parser.add_argument('--gpu', action='store_true', help='Force using GPU if available')
    parser.add_argument('--model-path', type=str, default=None, help='Path to the trained model')
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
    env = NetworkEnv(
        adj_matrix=adj_matrix,
        edge_list=edge_list,
        tm_list=tm_list,
        node_props=node_props,
        num_nodes=num_nodes,
        link_capacity=effective_link_capacity,
        max_edges=max_edges,
        seed=eval_seed
    )
    
    env.current_tm_idx = tm_index
    print(f"Evaluating using traffic matrix index {tm_index} (of {len(tm_list)} matrices)")
    
    # Device configuration
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU for evaluation")
    
    # State and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_dim = 256
    
    print(f"Evaluating Topology: {env.num_edges} actual edges (max_edges={max_edges})")
    print(f"State Dimension: {state_dim}")
    print(f"Action Dimension: {action_dim}")
    
    # Initialize the network with transformer architecture
    policy_net = TransformerQNetwork(state_dim, action_dim, hidden_dim=hidden_dim, nhead=4, num_layers=2).to(device)
    
    # Determine model path
    if args.model_path:
        model_load_path = args.model_path
    else:
        # Extract config name from path
        config_name = os.path.basename(config_path).replace('.json', '')
        if config_name.startswith("test_"):
            model_config_name = config_name.replace("test_", "", 1)
        else:
            model_config_name = config_name
            
        # Default model path
        model_load_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "Qlearning", "models", f"dqn_model_{model_config_name}.pth"
        )
    
    print(f"\nLoading trained model from {model_load_path}...")
    try:
        policy_net.load_state_dict(torch.load(model_load_path, map_location=device))
        policy_net.eval()  # Set to evaluation mode
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model file exists and matches the network architecture.")
        print("Available models in Qlearning/models directory:")
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Qlearning", "models")
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith(".pth"):
                    print(f"  - {file}")
        return
    
    # --- Evaluation Loop ---
    num_eval_episodes = 5
    total_rewards = []
    final_link_configs = []
    violations_occurred = []
    
    print(f"\nRunning evaluation for {num_eval_episodes} episodes...")
    for episode in range(num_eval_episodes):
        state = env.reset()
        # Debug state format
        print(f"\nState type: {type(state)}")
        print(f"State value: {state}")
        print(f"State structure: {dir(state) if hasattr(state, '__dict__') else 'N/A'}")
        
        episode_reward = 0
        done = False
        step = 0
        episode_violations = {'isolated': 0, 'overloaded': 0}
        
        while not done:
            step += 1
            # Convert state to tensor and get action from policy
            with torch.no_grad():
                # Handle different state formats
                if isinstance(state, np.ndarray):
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                elif hasattr(state, 'observation'):
                    # Gym observation wrapper format
                    state_tensor = torch.FloatTensor(state.observation).unsqueeze(0).to(device)
                elif hasattr(state, 'astype'):
                    # Try to convert directly
                    state_tensor = torch.FloatTensor(state.astype(np.float32)).unsqueeze(0).to(device)
                else:
                    # If state is a dictionary or other complex type, print it and extract the observation
                    print(f"Complex state: {type(state)}, {state}")
                    if isinstance(state, dict) and 'observation' in state:
                        state_tensor = torch.FloatTensor(state['observation']).unsqueeze(0).to(device)
                    else:
                        # Last resort - try to create a feature vector from what we know about the environment
                        try:
                            # Use network state variables directly
                            feature_vector = np.concatenate([
                                env.link_open,
                                env.link_loads.flatten() / env.link_capacity
                            ])
                            state_tensor = torch.FloatTensor(feature_vector).unsqueeze(0).to(device)
                            print(f"Created custom feature vector with shape: {feature_vector.shape}")
                        except Exception as e:
                            print(f"Failed to create custom feature vector: {e}")
                            # If we can't create a feature vector, we'll use a dummy action
                            # Just select action 0 (close link) by default
                            action = 0
                            return action
                
                q_values = policy_net(state_tensor)
                action = q_values.argmax(dim=1).item()
            
            # Execute action in environment
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            state = next_state
            
            # Track violations if they occur
            if isinstance(info, dict):
                if info.get('violation') == 'isolated':
                    episode_violations['isolated'] += 1
                elif info.get('violation') == 'overloaded':
                    episode_violations['overloaded'] += info.get('num_overloaded', 1)
            
            if done:
                total_rewards.append(episode_reward)
                final_link_configs.append(env.link_open.copy())
                violations_occurred.append(episode_violations)
                # Count closed links
                num_closed = sum(1 for link in env.link_open if link == 0)
                print(f" Episode {episode + 1}: Reward={episode_reward:.2f}, Links Closed={num_closed}/{env.num_edges}, Violations={episode_violations}")
                break
            
            # Safety check for max steps
            if step > env.num_edges * 2:
                print(f" Episode {episode + 1}: Exceeded max steps, terminating early.")
                break
    
    # --- Report Results ---
    print("\n--- Evaluation Summary ---")
    if total_rewards:
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        print(f" Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        
        # Analyze final configurations
        if final_link_configs:
            most_common = np.zeros(env.num_edges)
            for config in final_link_configs:
                most_common += config
            most_common = np.round(most_common / len(final_link_configs))
            print(f" Average Configuration (0=closed, 1=open): {most_common}")
            
            # Calculate the number of closed links
            avg_closed = env.num_edges - np.sum(most_common)
            print(f" Average Links Closed: {avg_closed:.1f}/{env.num_edges}")
        
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
            "avg_links_closed": float(avg_closed),
            "total_links": env.num_edges,
            "isolation_violations": total_iso,
            "overload_violations": total_ovl,
            "episodes_with_violations": episodes_with_violations
        }
        
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
