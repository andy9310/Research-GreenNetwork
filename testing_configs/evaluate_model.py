import sys
import os
import torch
import numpy as np
import time
import json
import argparse
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from Qlearning directory
from Qlearning.env import NetworkEnv
from Qlearning.agent import DQN, QNetwork, TransformerQNetwork

# --- Load Configuration from JSON ---
def load_config(config_path="config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def evaluate_model_on_test_config(test_config_path, model_config_path=None, tm_index=0, 
                                  use_gpu=False, gpu_device=0, adjust_capacity=True):
    """
    Evaluate a trained model on a testing configuration.
    
    Args:
        test_config_path: Path to the testing configuration file
        model_config_path: Path to the configuration used to train the model (optional)
        tm_index: Index of traffic matrix to evaluate on
        use_gpu: Whether to use GPU for evaluation
        gpu_device: GPU device index to use
        adjust_capacity: Whether to adjust link capacity for testing
    """
    # Load testing configuration
    print(f"Loading testing configuration from {test_config_path}")
    test_config = load_config(test_config_path)
    
    # If model config not specified, derive it from test config
    if model_config_path is None:
        # Extract the base config name (e.g., "test_config_5node" -> "config_5node")
        test_config_name = os.path.basename(test_config_path)
        if test_config_name.startswith("test_"):
            model_config_name = test_config_name.replace("test_", "", 1)
        else:
            model_config_name = test_config_name
            
        # Look for the corresponding config in the configs directory
        model_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                      "configs", model_config_name)
        
        print(f"Model config path not specified, derived as: {model_config_path}")
    
    # Load environment parameters from testing config
    num_nodes = test_config["num_nodes"]
    adj_matrix = test_config["adj_matrix"]
    edge_list = test_config["edge_list"]
    node_props = test_config["node_props"]
    tm_list = test_config["tm_list"]
    link_capacity = test_config["link_capacity"]
    max_edges = test_config["max_edges"]
    
    # Adjust link capacity if needed and requested
    effective_link_capacity = link_capacity
    if adjust_capacity:
        # Analyze the traffic matrix to estimate required capacity
        tm = np.array(tm_list[tm_index])
        total_traffic = np.sum(tm)
        avg_traffic_per_edge = total_traffic / len(edge_list) * 2  # Conservative estimate
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
        link_capacity=effective_link_capacity,  # Use adjusted capacity
        max_edges=max_edges,
        seed=eval_seed
    )
    
    num_actual_edges = env.num_edges
    
    # Set up state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_dim = 256  # Match the architecture used during training
    
    print(f"Evaluating Topology: {num_actual_edges} actual edges (max_edges={max_edges})")
    print(f"State Dimension (Padded): {state_dim}")
    print(f"Action Dimension: {action_dim}")
    
    # Device configuration
    if use_gpu and torch.cuda.is_available():
        if gpu_device >= 0 and gpu_device < torch.cuda.device_count():
            device = torch.device(f"cuda:{gpu_device}")
        else:
            device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True  # Optimize CUDNN
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        print(f"CUDA Memory Available: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        if use_gpu and not torch.cuda.is_available():
            print("Warning: GPU requested but CUDA is not available. Using CPU instead.")
        else:
            print("Using CPU for evaluation")
    
    # Initialize the network with transformer architecture
    policy_net = TransformerQNetwork(state_dim, action_dim, hidden_dim=hidden_dim, nhead=4, num_layers=2).to(device)
    
    # Determine model filename based on the model config
    if model_config_path.endswith('.json'):
        model_config_path = model_config_path[:-5]  # Remove .json extension
    
    # Extract just the base name for the model file
    base_name = os.path.basename(model_config_path)
    
    # Try all possible model naming patterns and locations
    project_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    possible_paths = [
        # Models in Qlearning/models directory
        os.path.join(project_dir, "Qlearning", "models", f"dqn_model_{base_name}.pth"),
        os.path.join(project_dir, "Qlearning", "models", f"dqn_mlp_{base_name}.pth"),
        os.path.join(project_dir, "Qlearning", "models", f"dqn_fat_mlp_{base_name}.pth"),
        
        # Models in root project directory
        os.path.join(project_dir, f"dqn_model_{base_name}.pth"),
        os.path.join(project_dir, f"dqn_mlp_{base_name}.pth"),
        
        # Models in project models directory
        os.path.join(project_dir, "models", f"dqn_model_{base_name}.pth"),
        os.path.join(project_dir, "models", f"dqn_mlp_{base_name}.pth")
    ]
    
    # Find the first path that exists
    model_load_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_load_path = path
            break
    
    # If no model found, default to the first path for error reporting
    if model_load_path is None:
        model_load_path = possible_paths[0]
    
    print(f"\nLoading trained model from {model_load_path}...")
    try:
        policy_net.load_state_dict(torch.load(model_load_path, map_location=device))
        policy_net.eval()  # Set the network to evaluation mode
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_load_path}")
        print("Please train the model first using train.py with the same config.")
        return None
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        print("This might happen if the saved model's architecture doesn't match")
        print("the current model structure based on the configuration.")
        return None
    
    # Validate the tm_index
    if tm_index < 0 or tm_index >= len(tm_list):
        print(f"Error: Traffic matrix index {tm_index} is out of range (0-{len(tm_list)-1})")
        print(f"Defaulting to index 0")
        tm_index = 0
    
    env.current_tm_idx = tm_index
    print(f"Evaluating using traffic matrix index {tm_index} (of {len(tm_list)} matrices)")
    
    # --- Evaluation Loop --- 
    num_eval_episodes = 5  # Evaluate over a few episodes for stability
    total_rewards = []
    final_link_configs = []
    violations_occurred = []
    
    print(f"\nRunning evaluation for {num_eval_episodes} episodes...")
    for episode in range(num_eval_episodes):
        # Handle different return formats of env.reset()
        reset_result = env.reset()
        # Check if reset returns a tuple with info or just the state
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            state, info = reset_result
        else:
            state = reset_result
            info = {}
        episode_reward = 0
        done = False
        step = 0
        episode_violations = {'isolated': 0, 'overloaded': 0}
        
        while not done:
            step += 1
            # Select action greedily (epsilon=0.0) using the loaded policy net
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = policy_net(state_tensor)
                action = q_values.argmax(dim=1).item()
            
            # Execute action in environment
            # Handle different return formats for env.step()
            step_result = env.step(action)
            
            # Check if step returns a tuple with 4 or 5 values
            if isinstance(step_result, tuple):
                if len(step_result) == 4:  # Old gym format: next_state, reward, done, info
                    next_state, reward, done, info = step_result
                elif len(step_result) == 5:  # New gymnasium format: next_state, reward, terminated, truncated, info
                    next_state, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    raise ValueError(f"Unexpected return format from env.step(): {len(step_result)} values")
            else:
                raise ValueError("env.step() did not return a tuple")
            
            episode_reward += reward
            state = next_state
            
            # Track violations if they occur
            if info.get('violation') == 'isolated':
                episode_violations['isolated'] += 1
            elif info.get('violation') == 'overloaded':
                episode_violations['overloaded'] += info.get('num_overloaded', 1)
            
            if done:
                total_rewards.append(episode_reward)
                final_link_configs.append(env.link_open.copy())  # Store copy of final state
                violations_occurred.append(episode_violations)
                # Count how many interfaces were closed
                num_closed = sum(1 for link in env.link_open if link == 0)
                print(f" Episode {episode + 1}: Reward={episode_reward:.2f}, Links Closed={num_closed}/{env.num_edges}, Violations={episode_violations}")
                break
    
    # --- Report Results --- 
    print("\n--- Evaluation Summary ---")
    results = {}
    
    if total_rewards:
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        print(f" Average Reward: {avg_reward:.2f} +/- {std_reward:.2f}")
        
        # Analyze final configurations
        config_tuples = [tuple(cfg) for cfg in final_link_configs]
        if config_tuples:
            from collections import Counter
            most_common_config, count = Counter(config_tuples).most_common(1)[0]
            print(f" Most Common Final Configuration ({count}/{num_eval_episodes} times): {np.array(most_common_config)}")
            
            # Calculate the number of closed links in the most common configuration
            most_common_closed = sum(1 for link in most_common_config if link == 0)
            print(f" Links Closed in Most Common Config: {most_common_closed}/{env.num_edges}")
        
        # Summarize violations
        total_iso = sum(v['isolated'] for v in violations_occurred)
        total_ovl = sum(v['overloaded'] for v in violations_occurred)
        episodes_with_violations = sum(1 for v in violations_occurred if v['isolated'] > 0 or v['overloaded'] > 0)
        print(f" Total Isolation Violations across episodes: {total_iso}")
        print(f" Total Overload Violations across episodes: {total_ovl}")
        print(f" Episodes with any violation: {episodes_with_violations}/{num_eval_episodes}")
        
        # Store results
        results = {
            "average_reward": float(avg_reward),
            "std_reward": float(std_reward),
            "most_common_config": np.array(most_common_config).tolist(),
            "links_closed": most_common_closed,
            "total_links": env.num_edges,
            "isolation_violations": total_iso,
            "overload_violations": total_ovl,
            "episodes_with_violations": episodes_with_violations
        }
    else:
        print("No episodes completed successfully.")
    
    # Compare with bruteforce results if available
    test_config_basename = os.path.basename(test_config_path).replace('.json', '')
    bruteforce_results_path = f"bruteforce_results_{test_config_basename}.json"
    
    if os.path.exists(bruteforce_results_path):
        try:
            with open(bruteforce_results_path, 'r') as f:
                bruteforce_results = json.load(f)
            
            bf_result = bruteforce_results.get(str(tm_index))
            if bf_result:
                print("\n--- Comparison with Bruteforce Optimal Solution ---")
                print(f" Bruteforce Best Score: {bf_result['best_score']}")
                print(f" Bruteforce Optimal Links Closed: {bf_result['num_links_closed']}/{bf_result['total_links']}")
                
                if results:
                    # Calculate relative performance
                    relative_performance = 100.0
                    if bf_result['best_score'] > 0:
                        relative_performance = (avg_reward / bf_result['best_score']) * 100.0
                    
                    print(f" Model Performance vs Optimal: {relative_performance:.2f}%")
                    results["optimal_score"] = bf_result['best_score']
                    results["optimal_links_closed"] = bf_result['num_links_closed']
                    results["relative_performance"] = float(relative_performance)
        except Exception as e:
            print(f"Error comparing with bruteforce results: {e}")
    
    # Save evaluation results
    results_file = f"evaluation_results_{test_config_basename}_tm{tm_index}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation results saved to {results_file}")
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained DQN agent on testing configurations')
    parser.add_argument('--test-config', type=str, required=True, 
                      help='Path to testing configuration JSON file')
    parser.add_argument('--model-config', type=str, default=None,
                      help='Path to the configuration used to train the model (optional)')
    parser.add_argument('--tm-index', type=int, default=0, 
                      help='Index of traffic matrix to use (default: 0)')
    parser.add_argument('--gpu', action='store_true', 
                      help='Force using GPU if available')
    parser.add_argument('--gpu-device', type=int, default=0, 
                      help='GPU device index to use when multiple GPUs are available')
    parser.add_argument('--no-adjust-capacity', action='store_true',
                      help='Disable automatic capacity adjustment for testing configs')
    
    args = parser.parse_args()
    
    evaluate_model_on_test_config(
        test_config_path=args.test_config,
        model_config_path=args.model_config,
        tm_index=args.tm_index,
        use_gpu=args.gpu,
        gpu_device=args.gpu_device,
        adjust_capacity=not args.no_adjust_capacity
    )

if __name__ == "__main__":
    main()
