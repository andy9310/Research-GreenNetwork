import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from env import NetworkEnv
from agent import DQN, QNetwork, TransformerQNetwork # Import TransformerQNetwork for model loading
import time
import json # Import json
import argparse # For command-line arguments
from collections import Counter # For counting configurations
import logging 
from visualization_utils import visualize_network_decisions, visualize_traffic_matrix, visualize_evaluation_results


logging.basicConfig(
    level=logging.DEBUG if __debug__ else logging.INFO,
    # ↑ Python 以 -O(Optimize) 執行時 __debug__ 變 False → 自動調整等級
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)  # 建立模組專屬 logger

# --- Load Configuration from JSON ---
def load_config(config_path="config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Evaluate a trained DQN agent for network topology optimization')
parser.add_argument('--config', type=str, default='config.json', help='Path to configuration JSON file')
parser.add_argument('--tm-index', type=int, default=None, help='Index of traffic matrix to use from tm_list (if None, evaluates all)')
parser.add_argument('--gpu', action='store_true', help='Force using GPU if available')
parser.add_argument('--gpu-device', type=int, default=0, help='GPU device index to use when multiple GPUs are available')
parser.add_argument('--architecture', type=str, choices=['mlp', 'fat_mlp', 'transformer'], default='mlp', help='Network architecture for evaluation')
parser.add_argument('--visualize', action='store_true', help='Generate visualizations of model decisions')
parser.add_argument('--save-dir', type=str, default='visualizations', help='Directory to save visualizations')
args = parser.parse_args()

# Load config from specified file
config_path = args.config
print(f"Loading configuration from {config_path}")
config = load_config(config_path)

# --- Environment Setup (Load from config) ---
num_nodes = config["num_nodes"]
adj_matrix = config["adj_matrix"]
edge_list = config["edge_list"]
node_props = config["node_props"]
tm_list = config["tm_list"]
link_capacity = config["link_capacity"]
max_edges = config["max_edges"] # Load max_edges

# Use a different seed for evaluation if desired
eval_seed = int(time.time())
    
# Create visualization directory if needed
if args.visualize:
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Visualizations will be saved to {args.save_dir}/")
env = NetworkEnv(
    adj_matrix=adj_matrix,
    edge_list=edge_list,
    tm_list=tm_list,  # We'll set current_tm_idx=0 below
    node_props=node_props,
    num_nodes=num_nodes,
    link_capacity=link_capacity,
    max_edges=max_edges, # Pass max_edges to env
    seed=eval_seed
)

num_actual_edges = env.num_edges # Actual edges in this specific config

# --- Agent Setup ---
# State dimension based on padded observation space
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n # Should be 2
hidden_dim = 256 # Match the enhanced architecture used during training

print(f"Evaluating Topology: {num_actual_edges} actual edges (max_edges={max_edges})")
print(f"State Dimension (Padded): {state_dim}")
print(f"Action Dimension: {action_dim}")

# Device configuration
if args.gpu and torch.cuda.is_available():
    if args.gpu_device >= 0 and args.gpu_device < torch.cuda.device_count():
        device = torch.device(f"cuda:{args.gpu_device}")
    else:
        device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True  # Optimize CUDNN
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    print(f"CUDA Memory Available: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
else:
    device = torch.device("cpu")
    if args.gpu and not torch.cuda.is_available():
        print("Warning: GPU requested but CUDA is not available. Using CPU instead.")
    else:
        print("Using CPU for evaluation")

# Initialize the QNetwork structure based on selected architecture
if args.architecture == 'mlp':
    policy_net = QNetwork(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
    print(f"Using Standard MLP Q-Network with hidden dim {hidden_dim}")
elif args.architecture == 'fat_mlp':
    # Fat MLP uses bigger hidden dimensions
    fat_hidden_dim = hidden_dim * 4  # Typically 4x larger
    policy_net = QNetwork(state_dim, action_dim, hidden_dim=fat_hidden_dim).to(device)
    print(f"Using Fat MLP Q-Network with hidden dim {fat_hidden_dim}")
else:
    policy_net = TransformerQNetwork(state_dim, action_dim, hidden_dim=hidden_dim, nhead=4, num_layers=2).to(device)
    print(f"Using Transformer Q-Network with hidden dim {hidden_dim}")

# --- Load Trained Model --- 
# Use config name in the model filename to match the naming from training
config_name = os.path.basename(config_path).split('.')[0]  # Extract base name without extension

# Build model path based on architecture - match train.py's path convention
if args.architecture == 'mlp':
    # model_load_path = f"models/dqn_mlp_{config_name}.pth"
    model_load_path = f"models/dqn_mlp_config_5node.pth"
elif args.architecture == 'fat_mlp':
    model_load_path = f"models/dqn_fat_mlp_{config_name}.pth"
elif args.architecture == 'transformer':
    model_load_path = f"models/dqn_transformer_{config_name}.pth"
else:
    # Generic fallback
    model_load_path = f"models/dqn_model_{config_name}.pth"
print(f"\nLoading trained model from {model_load_path}...")
try:
    policy_net.load_state_dict(torch.load(model_load_path, map_location=device))
    policy_net.eval() # Set the network to evaluation mode
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_load_path}")
    print("Please train the model first using train.py with the same max_edges setting.")
    exit()
except RuntimeError as e:
    print(f"Error loading model: {e}")
    print("This might happen if the saved model's architecture (state/action/hidden dims)")
    print("doesn't match the current model structure based on config.json (max_edges).")
    print("Ensure max_edges in config.json is the same as during training.")
    exit()

# Determine which traffic matrices to evaluate
if args.tm_index is not None:
    tm_indices = [args.tm_index]
    if args.tm_index >= len(tm_list):
        print(f"Warning: TM index {args.tm_index} out of range. Using the last available TM.")
        tm_indices = [len(tm_list) - 1]
    print(f"Evaluating only on traffic matrix index {tm_indices[0]}")
else:
    tm_indices = list(range(len(tm_list)))
    print(f"Evaluating on all {len(tm_indices)} traffic matrices")

# --- Process each traffic matrix ---
all_tm_results = []
for tm_idx in tm_indices:
    print(f"\n--- Evaluating Traffic Matrix {tm_idx} ---")
    env.current_tm_idx = tm_idx
    current_tm = np.array(tm_list[tm_idx])
    
    # --- Run a single episode for this traffic matrix ---
    state, _, _, _, info = env.reset() # Env reset returns (obs, reward, done, truncated, info)
    episode_reward = 0
    done = False
    step = 0
    episode_violations = {'isolated': 0, 'overloaded': 0}
    
    # Track action history for visualization
    action_history = []
    state_history = [state.copy()]
    link_utilization = None
    
    while not done:
        step += 1
        # Select action greedily (epsilon=0.0) using the loaded policy net
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            action = q_values.argmax(dim=1).item()
        
        # Record action
        action_history.append(action)
        
        # Execute action in environment
        next_state, reward, done, truncated, info = env.step(action)
        
        # Record state
        state_history.append(next_state.copy())
        
        # Update running values
        episode_reward += reward
        state = next_state
        
        # Track violations if they occur
        if info.get('violation') == 'isolated':
            episode_violations['isolated'] += 1
        elif info.get('violation') == 'overloaded':
            episode_violations['overloaded'] += info.get('num_overloaded', 1)
            
        # Get link utilization for visualization
        if 'link_utilization' in info:
            link_utilization = info['link_utilization']
        
        # Log violations
        if info.get('violation'):
            log.debug(f"TM {tm_idx}, Step {step}, Violation: {info.get('violation')}, Reward: {reward}")
    
    # Calculate final configuration
    final_config = env.link_open.copy()
    num_closed = sum(1 for link in final_config if link == 0)
    
    # If link utilization not in info, calculate it
    if link_utilization is None:
        # We need to compute the link utilization based on final config
        link_utilization = env.get_link_utilization()
    
    # Print summary for this traffic matrix
    print(f"Traffic Matrix {tm_idx}: Reward={episode_reward:.2f}, Links Closed={num_closed}/{env.num_edges}")
    print(f"Violations: {episode_violations}")
    
    # Store results for this traffic matrix
    tm_result = {
        'tm_idx': tm_idx,
        'reward': episode_reward,
        'final_config': final_config,
        'link_utilization': link_utilization,
        'violations': episode_violations,
        'num_closed': num_closed
    }
    all_tm_results.append(tm_result)
    
    # Generate visualization if requested
    if args.visualize:
        model_name = f"DQN_{args.architecture}"
        output_path = os.path.join(args.save_dir, f"{model_name}_tm{tm_idx}.png")
        
        # Create visualization
        fig = plt.figure(figsize=(18, 12))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        
        # Traffic Matrix plot
        ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=1, rowspan=1)
        visualize_traffic_matrix(current_tm, title=f"Traffic Matrix {tm_idx}")
        
        # Network plot
        ax2 = plt.subplot2grid((2, 3), (0, 1), colspan=2, rowspan=2)
        
        # Identify violated links
        violated_links = []
        for i, util in enumerate(link_utilization):
            if util > 1.0 and final_config[i] == 1:  # Open link that's overloaded
                violated_links.append(i)
        
        # Create network visualization title
        title = f"{model_name} - Traffic Matrix {tm_idx} - Reward: {episode_reward:.2f}"
        if episode_violations.get('overloaded', 0) > 0:
            title += f" - {episode_violations['overloaded']} Overloaded"
        if episode_violations.get('isolated', 0) > 0:
            title += f" - {episode_violations['isolated']} Isolated"
        
        # Draw network visualization
        visualize_network_decisions(
            edge_list=edge_list,
            link_open=final_config,
            link_utilization=link_utilization,
            violated_links=violated_links,
            title=title
        )
        
        # Link utilization plot
        ax3 = plt.subplot2grid((2, 3), (1, 0), colspan=1, rowspan=1)
        
        # Filter to only show open links
        open_links = [i for i, is_open in enumerate(final_config) if is_open == 1]
        if open_links:  # Only if there are open links
            open_link_utils = [link_utilization[i] for i in open_links]
            open_link_labels = [f"{edge_list[i][0]}->{edge_list[i][1]}" for i in open_links]
            
            # Sort by utilization
            sorted_indices = np.argsort(open_link_utils)
            sorted_utils = [open_link_utils[i] for i in sorted_indices]
            sorted_labels = [open_link_labels[i] for i in sorted_indices]
            
            # Use color mapping based on utilization
            colors = plt.cm.RdYlGn_r(np.array(sorted_utils))
            
            # Create horizontal bar chart
            bars = ax3.barh(range(len(sorted_utils)), sorted_utils, color=colors)
            ax3.set_title("Link Utilization (Open Links)")
            ax3.set_xlabel("Utilization")
            ax3.axvline(x=1.0, color='red', linestyle='--', label="Capacity Threshold")
            ax3.set_yticks(range(len(sorted_labels)))
            ax3.set_yticklabels(sorted_labels)
            ax3.legend()
        else:
            ax3.set_title("No Open Links")
        
        # Add metadata text
        metadata_text = (
            f"Model: {model_name}\n"
            f"Traffic Matrix: {tm_idx}\n"
            f"Final Reward: {episode_reward:.2f}\n"
            f"Links Closed: {num_closed}/{env.num_edges}\n"
            f"Overload Violations: {episode_violations.get('overloaded', 0)}\n"
            f"Isolation Violations: {episode_violations.get('isolated', 0)}"
        )
        
        # Add text box with metadata
        fig.text(0.02, 0.02, metadata_text, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Save figure
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Visualization saved to {output_path}")

# --- Report Overall Results --- 
print("\n=== Overall Evaluation Summary ===\n")
if all_tm_results:
    # Compile aggregated statistics
    avg_reward = np.mean([r['reward'] for r in all_tm_results])
    avg_closed = np.mean([r['num_closed'] for r in all_tm_results])
    tm_with_violations = sum(1 for r in all_tm_results if r['violations']['isolated'] > 0 or r['violations']['overloaded'] > 0)
    total_isolated = sum(r['violations']['isolated'] for r in all_tm_results)
    total_overloaded = sum(r['violations']['overloaded'] for r in all_tm_results)
    
    # Print summary
    print(f"Evaluated on {len(all_tm_results)} traffic matrices")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Links Closed: {avg_closed:.2f}/{env.num_edges}")
    print(f"Traffic Matrices with Violations: {tm_with_violations}/{len(all_tm_results)}")
    print(f"Total Isolation Violations: {total_isolated}")
    print(f"Total Overload Violations: {total_overloaded}")
    
    # Print individual TM results
    print("\nResults by Traffic Matrix:")
    for result in all_tm_results:
        tm_idx = result['tm_idx']
        reward = result['reward']
        num_closed = result['num_closed']
        violations = result['violations']
        has_violation = violations['isolated'] > 0 or violations['overloaded'] > 0
        
        status = "❌ (Violation)" if has_violation else "✅ (Valid)"
        print(f"TM {tm_idx}: Reward={reward:.2f}, Closed={num_closed}/{env.num_edges}, Status={status}")
        
        if has_violation:
            violation_details = []
            if violations['isolated'] > 0:
                violation_details.append(f"{violations['isolated']} isolated")
            if violations['overloaded'] > 0:
                violation_details.append(f"{violations['overloaded']} overloaded")
            print(f"  Violations: {', '.join(violation_details)}")
    
    if args.visualize:
        print(f"\nVisualizations saved to {args.save_dir}/")
else:
    print("No traffic matrices were evaluated successfully.")
