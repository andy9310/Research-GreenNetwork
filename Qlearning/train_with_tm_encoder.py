"""
Training Script with Traffic Matrix Representation Learning

This script trains a DQN agent with traffic matrix encoding capabilities,
allowing it to better generalize across different traffic patterns.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import random
import torch
from tqdm import tqdm

from tm_agent import TMEnhancedDQNAgent
from env import NetworkEnv

def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def train_agent_with_tm_encoder(args):
    """Train DQN agent with traffic matrix representation learning."""
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load config from specified file
    config_path = args.config
    print(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    # --- Environment Setup (Load from config) ---
    num_nodes = config["num_nodes"]
    adj_matrix = config["adj_matrix"]
    edge_list = config["edge_list"]
    node_props = config.get("node_props", {})
    tm_list = config["tm_list"]
    link_capacity = config["link_capacity"]
    max_edges = config.get("max_edges", len(edge_list))
    
    # If tm_subset is specified, only use a subset of traffic matrices
    if args.tm_subset is not None and args.tm_subset < len(tm_list):
        tm_list = tm_list[:args.tm_subset]
        print(f"Using {len(tm_list)} traffic matrices (subset)")
    
    # Initialize environment
    env = NetworkEnv(
        adj_matrix=adj_matrix,
        edge_list=edge_list,
        tm_list=tm_list,
        node_props=node_props,
        num_nodes=num_nodes,
        link_capacity=link_capacity,
        max_edges=max_edges,
        seed=args.seed,
        random_edge_order=args.random_edge_order
    )
    
    # Get state and action dimensions
    state_dim = len(env._get_observation())
    action_dim = env.action_space.n
    print(f"State Dimension: {state_dim}")
    print(f"Action Dimension: {action_dim}")
    
    # Device for training
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using {device} for training")
    
    # Initialize agent with traffic matrix encoder
    agent = TMEnhancedDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_nodes=num_nodes,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        lr=args.learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        update_every=args.update_every,
        tm_embedding_dim=args.tm_embedding_dim,
        device=device
    )
    print(f"Initialized agent with traffic matrix encoder (embedding dim: {args.tm_embedding_dim})")
    
    # --- Training parameters ---
    num_episodes_per_tm = args.episodes
    total_episodes = num_episodes_per_tm * len(tm_list)
    print(f"Training for {total_episodes} episodes ({num_episodes_per_tm} per traffic matrix)")
    
    # Epsilon-greedy parameters
    epsilon_start = args.epsilon_start
    epsilon_end = args.epsilon_end
    epsilon_decay = args.epsilon_decay
    
    # Other parameters
    target_update_freq = args.target_update
    print_interval = 100
    save_interval = 500
    
    # --- Training loop ---
    print("\nStarting training...")
    total_steps = 0
    episode_rewards = []
    
    # Create directory for models if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Get config name for saving models
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    
    # Main training loop - iterate through traffic matrices
    for tm_round in range(args.tm_rounds):
        print(f"\nTraining round {tm_round+1}/{args.tm_rounds}")
        
        # Shuffle traffic matrices each round for better generalization
        tm_indices = list(range(len(tm_list)))
        random.shuffle(tm_indices)
        
        for tm_idx_pos, tm_idx in enumerate(tm_indices):
            # Set current traffic matrix
            env.current_tm_idx = tm_idx
            current_tm = np.array(tm_list[tm_idx])
            
            print(f"\nTraining on Traffic Matrix {tm_idx+1}/{len(tm_list)}")
            tm_episode_rewards = []
            
            # Create progress bar
            tm_progress = tqdm(total=num_episodes_per_tm, desc=f"TM {tm_idx+1}/{len(tm_list)}")
            
            # Train on this traffic matrix for specified number of episodes
            for episode in range(num_episodes_per_tm):
                # Reset environment
                state, _, _, _, _ = env.reset()
                episode_reward = 0
                done = False
                
                # Calculate epsilon for this episode (decaying over time)
                total_ep = tm_round * len(tm_list) * num_episodes_per_tm + tm_idx_pos * num_episodes_per_tm + episode
                epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** (total_ep // 100)))
                
                # Episode loop
                while not done:
                    # Select action using epsilon-greedy policy
                    action = agent.act(state, current_tm, epsilon)
                    
                    # Take action in environment
                    next_state, reward, done, truncated, info = env.step(action)
                    
                    # Store transition in replay buffer with traffic matrix
                    agent.step(state, action, reward, next_state, done, current_tm)
                    
                    # Update state and reward
                    state = next_state
                    episode_reward += reward
                    total_steps += 1
                    
                    # Update target network periodically
                    if total_steps % target_update_freq == 0:
                        agent.qnetwork_target.load_state_dict(agent.qnetwork_local.state_dict())
                    
                    # Episode finished
                    if done:
                        break
                
                # Add to both overall and traffic matrix specific rewards
                episode_rewards.append(episode_reward)
                tm_episode_rewards.append(episode_reward)
                
                # Update the progress bar
                tm_progress.update(1)
                
                # Print progress at intervals
                if (episode + 1) % print_interval == 0:
                    avg_reward = np.mean(tm_episode_rewards[-min(print_interval, len(tm_episode_rewards)):]) 
                    max_reward = np.max(tm_episode_rewards) if tm_episode_rewards else 0
                    
                    # Print progress information
                    tm_progress.write(f"TM {tm_idx+1}/{len(tm_list)}, Episode {episode + 1}/{num_episodes_per_tm}, "
                                    f"TM Avg Reward: {avg_reward:.2f}, Max Reward: {max_reward:.2f}, "
                                    f"Epsilon: {epsilon:.3f}, Total Steps: {total_steps}")
                
                # Save model at intervals
                if (tm_round * len(tm_list) * num_episodes_per_tm + tm_idx_pos * num_episodes_per_tm + episode + 1) % save_interval == 0:
                    model_path = f"models/tm_dqn_{args.architecture}_{config_name}_checkpoint.pth"
                    agent.save(model_path)
                    print(f"\nCheckpoint saved to {model_path}")
            
            # Close progress bar
            tm_progress.close()
        
        # Save model at the end of each round
        model_path = f"models/tm_dqn_{args.architecture}_{config_name}_round{tm_round+1}.pth"
        agent.save(model_path)
        print(f"\nModel saved after round {tm_round+1} to {model_path}")
    
    # --- Training complete ---
    # Save final model
    final_model_path = f"models/tm_dqn_{args.architecture}_{config_name}.pth"
    agent.save(final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    # Print training summary
    avg_reward = np.mean(episode_rewards[-min(1000, len(episode_rewards)):])
    print(f"\nTraining Complete!")
    print(f"Final Average Reward (last 1000 episodes): {avg_reward:.2f}")
    print(f"Total Episodes: {total_episodes}")
    print(f"Total Steps: {total_steps}")
    
    return agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DQN agent with traffic matrix encoding')
    parser.add_argument('--config', type=str, default="configs/config_5node.json", help='Path to config JSON')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--tm-subset', type=int, default=None, help='Use only a subset of traffic matrices (specify count)')
    parser.add_argument('--tm-rounds', type=int, default=5, help='Number of rounds to cycle through all traffic matrices')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes per traffic matrix per round')
    parser.add_argument('--architecture', type=str, default='mlp', help='Architecture type for model saving')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension for Q-network')
    parser.add_argument('--tm-embedding-dim', type=int, default=64, help='Embedding dimension for traffic matrix encoder')
    parser.add_argument('--learning-rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--tau', type=float, default=1e-3, help='Soft update parameter')
    parser.add_argument('--buffer-size', type=int, default=100000, help='Replay buffer size')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--update-every', type=int, default=4, help='Update frequency')
    parser.add_argument('--target-update', type=int, default=1000, help='Target network update frequency')
    parser.add_argument('--epsilon-start', type=float, default=1.0, help='Starting epsilon for epsilon-greedy')
    parser.add_argument('--epsilon-end', type=float, default=0.01, help='Minimum epsilon for epsilon-greedy')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Epsilon decay rate')
    parser.add_argument('--random-edge-order', action='store_true', help='Randomize edge order each episode')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    args = parser.parse_args()
    train_agent_with_tm_encoder(args)
