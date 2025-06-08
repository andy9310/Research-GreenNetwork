"""
Training Script with Latent State Representation and Prediction Learning

This script trains a DQN agent with both:
1. Latent state encoding - for better state representation
2. State prediction capabilities - for model-based reinforcement learning

The agent can operate in two modes:
- Standard mode: Uses the environment for transitions
- Model-based mode: Uses the learned dynamics model to predict next states
"""

import os
import json
import argparse
import numpy as np
import torch
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import sys
import os
# Add parent directory to path to import modules from there
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent_env import NetworkEnv
from latent_agent import LatentPredictorAgent

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def train_agent(args):
    """Train a DQN agent with latent state encoding and state prediction."""
    # Record start time
    start_time = time.time()
    
    # --- Load configuration ---
    config_path = args.config
    config = load_config(config_path)
    config_name = os.path.basename(config_path).split('.')[0]
    print(f"Loaded configuration from {config_path}")
    
    # Extract parameters from config
    num_nodes = config["num_nodes"]
    adj_matrix = config["adj_matrix"]
    edge_list = config["edge_list"]
    tm_list = config["tm_list"]
    node_props = config.get("node_props", {})
    link_capacity = config["link_capacity"]
    max_edges = config.get("max_edges", len(edge_list))
    
    # --- Setup environment ---
    env = NetworkEnv(
        adj_matrix=adj_matrix,
        edge_list=edge_list,
        tm_list=tm_list,
        node_props=node_props,
        num_nodes=num_nodes,
        link_capacity=link_capacity,
        max_edges=max_edges,
        random_edge_order=args.random_edge_order,
        seed=args.seed
    )
    print(f"Created environment with {num_nodes} nodes, {len(edge_list)} edges, and {len(tm_list)} traffic matrices")
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    # Verify the environment is providing states with the expected dimensions
    initial_state = env.reset()
    if isinstance(initial_state, tuple):
        # Newer gym versions return (state, info)
        initial_state = initial_state[0]
    print(f"Initial state shape from environment: {initial_state.shape}")
    if initial_state.shape[0] != state_dim:
        print(f"WARNING: Environment state dimension ({initial_state.shape[0]}) doesn't match expected dimension ({state_dim})")
        print(f"Adjusting state_dim to match environment")
        state_dim = initial_state.shape[0]
    
    # --- Setup agent ---
    # Set the device
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize agent with num_nodes for adjacency matrix representation
    agent = LatentPredictorAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=args.latent_dim,
        predictor_hidden_dim=args.predictor_hidden_dim,
        buffer_size=100000,
        batch_size=64,
        predictor_batch_size=128,
        gamma=0.99,
        tau=1e-3,
        lr=args.lr,
        predictor_lr=args.predictor_lr,
        update_every=4,
        enable_predictor=args.enable_predictor,
        architecture=args.architecture,
        num_nodes=num_nodes,  # Pass number of nodes for adjacency matrix processing
        device=device
    )
    
    # Print agent configuration
    print(f"Initialized agent with latent state encoder (latent dim: {args.latent_dim}, architecture: {args.architecture})")
    if args.enable_predictor:
        print(f"State predictor enabled (hidden dim: {args.predictor_hidden_dim})")
    else:
        print("State predictor disabled")
    
    # --- Training parameters ---
    num_episodes_per_tm = args.episodes
    total_episodes = num_episodes_per_tm * len(tm_list) * args.tm_rounds
    print(f"Training for {total_episodes} episodes ({num_episodes_per_tm} per traffic matrix, {args.tm_rounds} rounds)")
    
    # Epsilon-greedy parameters
    epsilon_start = args.epsilon_start
    epsilon_end = args.epsilon_end
    epsilon_decay = args.epsilon_decay
    
    # Other parameters
    target_update_freq = args.target_update
    print_interval = 1000
    save_interval = 5000
    
    # --- Training loop ---
    print("\nStarting training...")
    total_steps = 0
    episode_rewards = []
    prediction_errors = []
    
    # Create directory for models if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Get config name for saving models
    model_base_path = f"models/latent_predictor_{config_name}"
    if args.random_edge_order:
        model_base_path += "_random_edge"
    model_base_path += f"_latent{args.latent_dim}_{args.architecture}"
    if args.enable_predictor:
        model_base_path += f"_pred{args.predictor_hidden_dim}"
    
    # Initialize rewards tracking
    last_100_rewards = []
    tm_indices = list(range(len(tm_list)))
    
    # Multiple rounds of training over all traffic matrices
    for tm_round in range(args.tm_rounds):
        print(f"\n--- Round {tm_round+1}/{args.tm_rounds} ---")
        
        # Shuffle traffic matrix order for this round
        random.shuffle(tm_indices)
        
        # Process each traffic matrix
        for tm_idx_pos, tm_idx in enumerate(tm_indices):
            # Set the current traffic matrix
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
                
                # If using model-based mode for validation (every 100 episodes)
                use_model = args.enable_predictor and args.validate_predictor and episode % 100 == 0 and episode > 0
                
                if use_model:
                    # Model-based episode (for validation)
                    predicted_state = state.copy()
                    true_state = state.copy()
                    model_steps = 0
                    prediction_error_sum = 0
                    
                    # Take actions using the model for prediction
                    while not done and model_steps < env.num_edges * 2:
                        # Select action based on predicted state
                        action = agent.act(predicted_state, epsilon)
                        
                        # Get the true next state from environment (for validation only)
                        next_true_state, reward, done, _, _ = env.step(action)
                        
                        # Predict the next state
                        next_predicted_state = agent.predict_next_state(predicted_state, action)
                        
                        # Calculate prediction error
                        pred_error = np.mean((next_predicted_state - next_true_state)**2)
                        prediction_error_sum += pred_error
                        
                        # Update states
                        predicted_state = next_predicted_state
                        true_state = next_true_state
                        
                        # Update rewards and counters
                        episode_reward += reward
                        model_steps += 1
                        
                        if done:
                            break
                    
                    # Track prediction error
                    if model_steps > 0:
                        avg_pred_error = prediction_error_sum / model_steps
                        prediction_errors.append(avg_pred_error)
                        print(f"\nModel-based episode {episode}: Reward={episode_reward:.2f}, Avg Pred Error={avg_pred_error:.6f}, ModelSteps:{model_steps}")
                else:
                    # Standard DQN episode with environment interaction
                    while not done:
                        # Print state shape for debugging (only first episode)
                        if total_ep == 0 and episode_reward == 0:
                            print(f"\nDEBUG - State shape: {state.shape}")
                            print(f"State content sample: {state[:10]}")

                        # Select action using epsilon-greedy policy
                        try:
                            action = agent.act(state, epsilon)
                        except Exception as e:
                            print(f"\nERROR during agent.act() - {str(e)}")
                            print(f"State shape: {state.shape}")
                            raise
                        
                        # Take action in environment
                        next_state, reward, done, truncated, info = env.step(action)
                        
                        # Store transition in replay buffer
                        agent.step(state, action, reward, next_state, done)
                        
                        # Update state and reward
                        state = next_state
                        episode_reward += reward
                        total_steps += 1
                        
                        # Episode finished
                        if done:
                            break
                
                # Add to both overall and traffic matrix specific rewards
                episode_rewards.append(episode_reward)
                tm_episode_rewards.append(episode_reward)
                last_100_rewards.append(episode_reward)
                if len(last_100_rewards) > 100:
                    last_100_rewards.pop(0)
                
                # Update the progress bar
                tm_progress.update(1)
                
                # Print progress periodically
                if (total_ep+1) % print_interval == 0:
                    avg_reward = np.mean(last_100_rewards)
                    avg_pred_error = np.mean(prediction_errors[-10:]) if prediction_errors else 0
                    print(f"Episode {total_ep+1}/{total_episodes} | Avg Reward: {avg_reward:.2f} | Epsilon: {epsilon:.4f} | Pred Error: {avg_pred_error:.6f}")
                
                # Save model periodically
                if (total_ep+1) % save_interval == 0:
                    if args.enable_predictor:
                        model_path = f"{model_base_path}_episode{total_ep+1}_dqn.pth"
                        predictor_path = f"{model_base_path}_episode{total_ep+1}_predictor.pth"
                        agent.save(model_path, predictor_path)
                        print(f"Models saved to {model_path} and {predictor_path}")
                    else:
                        model_path = f"{model_base_path}_episode{total_ep+1}.pth"
                        agent.save(model_path)
                        print(f"Model saved to {model_path}")
            
            # Close progress bar
            tm_progress.close()
            
            # Print traffic matrix specific results
            avg_tm_reward = np.mean(tm_episode_rewards)
            print(f"Traffic Matrix {tm_idx+1} | Avg Reward: {avg_tm_reward:.2f}")
    
    # Save final model
    if args.enable_predictor:
        final_model_path = f"{model_base_path}_final_dqn.pth"
        final_predictor_path = f"{model_base_path}_final_predictor.pth"
        agent.save(final_model_path, final_predictor_path)
        print(f"Final models saved to {final_model_path} and {final_predictor_path}")
    else:
        final_model_path = f"{model_base_path}_final.pth"
        agent.save(final_model_path)
        print(f"Final model saved to {final_model_path}")
    
    # Plot training rewards
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    rewards_plot_path = f"models/latent_predictor_rewards_{config_name}.png"
    plt.savefig(rewards_plot_path)
    print(f"Training rewards plot saved to {rewards_plot_path}")
    
    # Plot prediction errors if available
    if prediction_errors:
        plt.figure(figsize=(10, 6))
        plt.plot(prediction_errors)
        plt.xlabel('Training Steps')
        plt.ylabel('Prediction Error (MSE)')
        plt.title('State Prediction Errors')
        pred_plot_path = f"models/latent_predictor_errors_{config_name}.png"
        plt.savefig(pred_plot_path)
        print(f"Prediction errors plot saved to {pred_plot_path}")
    
    # Plot Q-learning losses
    if agent.q_losses:
        plt.figure(figsize=(10, 6))
        plt.plot(agent.q_losses)
        plt.xlabel('Training Steps')
        plt.ylabel('Q-Network Loss (MSE)')
        plt.title('DQN Training Loss')
        # Add moving average to smooth the curve
        window_size = min(100, len(agent.q_losses))
        if window_size > 0:
            moving_avg = np.convolve(agent.q_losses, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(agent.q_losses)), moving_avg, 'r-', linewidth=2, label='Moving Average')
            plt.legend()
        q_loss_plot_path = f"models/latent_q_losses_{config_name}.png"
        plt.savefig(q_loss_plot_path)
        print(f"Q-learning loss plot saved to {q_loss_plot_path}")
    
    # Calculate and display total training time
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal training time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d} (HH:MM:SS)")
    
    return final_model_path

def main():
    parser = argparse.ArgumentParser(description='Train DQN with latent state encoding and prediction')
    
    # Configuration
    parser.add_argument('--config', type=str, default='../configs/config.json', help='Path to configuration file')
    
    # Network architecture
    parser.add_argument('--latent-dim', type=int, default=64, help='Dimension of latent state representation')
    parser.add_argument('--predictor-hidden-dim', type=int, default=256, help='Hidden dimension of state predictor')
    parser.add_argument('--architecture', type=str, default='mlp', choices=['mlp', 'fatmlp', 'advanced'], help='Network architecture type')
    parser.add_argument('--enable-predictor', action='store_true', help='Enable state predictor training')
    parser.add_argument('--validate-predictor', action='store_true', help='Periodically validate predictor during training')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage instead of CUDA')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes per traffic matrix')
    parser.add_argument('--tm-rounds', type=int, default=3, help='Number of rounds to cycle through all traffic matrices')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for DQN')
    parser.add_argument('--predictor-lr', type=float, default=1e-3, help='Learning rate for state predictor')
    parser.add_argument('--epsilon-start', type=float, default=1.0, help='Initial epsilon for epsilon-greedy')
    parser.add_argument('--epsilon-end', type=float, default=0.01, help='Final epsilon for epsilon-greedy')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Epsilon decay rate')
    parser.add_argument('--target-update', type=int, default=1, help='Steps between target network updates')
    
    # Environment settings
    parser.add_argument('--random-edge-order', action='store_true', help='Use random edge ordering in environment')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    train_agent(args)

if __name__ == "__main__":
    main()
