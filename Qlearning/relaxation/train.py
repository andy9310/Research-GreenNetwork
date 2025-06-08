"""
Training Script for Relaxed Network Optimization

This script trains a DDPG agent with state prediction for relaxed network optimization.
The agent learns to set continuous capacity scaling factors for network links.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Qlearning.relaxation.env import RelaxedNetworkEnv
from Qlearning.relaxation.ddpg_agent import DDPGAgent

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def train_agent(args):
    """
    Train a DDPG agent on the relaxed network optimization task.
    
    Args:
        args: Command-line arguments
    """
    # --- Load configuration ---
    config_path = args.config
    config = load_config(config_path)
    config_name = os.path.basename(config_path).split('.')[0]
    print(f"Loaded configuration from {config_path}")
    
    # Extract parameters from config
    num_nodes = config["num_nodes"]
    edge_list = config["edge_list"]
    tm_list = config["tm_list"]
    link_capacity = config.get("link_capacity", [1.0] * len(edge_list))
    node_props = config.get("node_props", {})
    max_edges = len(edge_list)
    
    # Create environment
    env = RelaxedNetworkEnv(
        adj_matrix=config["adj_matrix"],
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
    
    # Get state dimensions
    initial_state = env.reset()
    state_dim = initial_state.shape[0]
    print(f"State dimension: {state_dim}")
    
    # --- Setup agent ---
    # Set up device
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize agent
    agent = DDPGAgent(
        state_dim=state_dim,
        latent_dim=args.latent_dim,
        predictor_hidden_dim=args.predictor_hidden_dim,
        actor_hidden_dim=args.actor_hidden_dim,
        critic_hidden_dim=args.critic_hidden_dim,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        predictor_batch_size=args.predictor_batch_size,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        predictor_lr=args.predictor_lr,
        enable_predictor=args.enable_predictor,
        device=device
    )
    
    # Print agent configuration
    print(f"Initialized agent with latent state encoder (latent dim: {args.latent_dim})")
    if args.enable_predictor:
        print(f"State predictor enabled (hidden dim: {args.predictor_hidden_dim})")
    else:
        print("State predictor disabled")
    
    # Create directories for saving models and plots
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Get config name for saving models
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
    model_base_path = os.path.join(model_dir, f"relaxed_{config_name}")
    if args.random_edge_order:
        model_base_path += "_random_edge"
    model_base_path += f"_latent{args.latent_dim}"
    if args.enable_predictor:
        model_base_path += f"_pred{args.predictor_hidden_dim}"
    
    # Set training parameters
    max_episodes = args.episodes * len(tm_list) * args.tm_rounds
    print(f"Training for {max_episodes} episodes ({args.episodes} per traffic matrix, {args.tm_rounds} rounds)")
    
    # --- Training ---
    print("\nStarting training...")
    
    # Lists to store rewards and prediction errors
    rewards = []
    avg_rewards = []
    prediction_errors = []
    prediction_valid_tm_indices = []
    
    # Training loop
    total_ep = 0
    noise_scale = args.noise_scale
    
    # Track validation metrics
    best_total_reward = -float('inf')
    
    # Training rounds
    for round_idx in range(args.tm_rounds):
        print(f"\n--- Round {round_idx+1}/{args.tm_rounds} ---\n")
        
        # Train on each traffic matrix
        for tm_idx in range(len(tm_list)):
            env.current_tm_idx = tm_idx
            
            print(f"Training on Traffic Matrix {tm_idx}/{len(tm_list) - 1}")
            
            # Train for specified number of episodes per traffic matrix
            pbar = tqdm(range(args.episodes))
            for episode in pbar:
                # Reset environment
                state = env.reset()
                episode_reward = 0
                
                # Reset noise process
                agent.reset()
                
                # Decreasing noise over time
                current_noise = noise_scale * max(0.1, 1.0 - total_ep / max_episodes)
                
                # Adjust current_tm_idx if random_edge_order is True to ensure we train on the right TM
                env.current_tm_idx = tm_idx
                
                done = False
                while not done:
                    # Select action with noise
                    action = agent.act(state, add_noise=True)
                    
                    # Take action in environment
                    next_state, reward, done, truncated, info = env.step(action)
                    
                    # Save experience and learn
                    agent.step(state, action, reward, next_state, done)
                    
                    # Update state and reward
                    state = next_state
                    episode_reward += reward
                
                # Store reward
                rewards.append(episode_reward)
                
                # Update counters
                total_ep += 1
                
                # Compute average reward over last 100 episodes
                if len(rewards) >= 100:
                    avg_reward = np.mean(rewards[-100:])
                else:
                    avg_reward = np.mean(rewards)
                avg_rewards.append(avg_reward)
                
                # Get average prediction error if predictor is enabled
                if args.enable_predictor and agent.prediction_errors:
                    avg_pred_error = np.mean(agent.prediction_errors[-100:]) if len(agent.prediction_errors) >= 100 else np.mean(agent.prediction_errors)
                    prediction_errors.append(avg_pred_error)
                    prediction_valid_tm_indices.append(tm_idx)
                
                # Update progress bar
                pbar.set_description(f"TM {tm_idx}/{len(tm_list) - 1}: Episode {episode+1}/{args.episodes} | Reward: {episode_reward:.2f} | Avg: {avg_reward:.2f} | Noise: {current_noise:.3f}")
                
                # Validate state predictor periodically
                if args.enable_predictor and args.validate_predictor and total_ep % args.validation_freq == 0:
                    # Validate on current TM: compare predicted vs actual trajectory
                    if args.enable_predictor:
                        validation_reward, prediction_accuracy = validate_predictor(agent, env, tm_idx)
                        print(f"\nValidation TM {tm_idx}: Reward={validation_reward:.2f}, Pred Accuracy={prediction_accuracy:.6f}")
                
                # Save model periodically
                if total_ep % args.save_freq == 0:
                    # Create model directory if it doesn't exist
                    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
                    os.makedirs(model_dir, exist_ok=True)
                    
                    # Use absolute paths for saving models
                    actor_path = os.path.join(model_dir, f"relaxed_{config_name}_episode{total_ep}_actor.pth")
                    critic_path = os.path.join(model_dir, f"relaxed_{config_name}_episode{total_ep}_critic.pth")
                    predictor_path = os.path.join(model_dir, f"relaxed_{config_name}_episode{total_ep}_predictor.pth") if args.enable_predictor else None
                    
                    # Save the models
                    agent.save(actor_path, critic_path, predictor_path)
                    print(f"\nModels saved to:\n{actor_path}\n{critic_path}" + (f"\n{predictor_path}" if predictor_path else ""))
                    
                    # Save training curves
                    save_training_curves(rewards, avg_rewards, prediction_errors, prediction_valid_tm_indices, config_name)
    
    # Save final model
    actor_path = f"{model_base_path}_final_actor.pth"
    critic_path = f"{model_base_path}_final_critic.pth"
    predictor_path = f"{model_base_path}_final_predictor.pth" if args.enable_predictor else None
    
    agent.save(actor_path, critic_path, predictor_path)
    print(f"Final models saved to {actor_path}, {critic_path}" + (f", {predictor_path}" if predictor_path else ""))
    
    # Save training curves
    save_training_curves(rewards, avg_rewards, prediction_errors, prediction_valid_tm_indices, config_name)
    
    return agent

def validate_predictor(agent, env, tm_idx, num_steps=10):
    """
    Validate state predictor by comparing predicted vs actual trajectory.
    
    Args:
        agent: DDPG agent with state predictor
        env: Environment
        tm_idx: Traffic matrix index
        num_steps: Number of steps to simulate
    
    Returns:
        validation_reward: Total reward from actual environment
        prediction_accuracy: Mean squared error between predicted and actual states
    """
    # Set environment to specified traffic matrix
    env.current_tm_idx = tm_idx
    
    # Reset environment
    state = env.reset()
    done = False
    
    # Tracking
    actual_states = []
    predicted_states = []
    total_reward = 0
    step_count = 0
    
    # Run trajectory
    while not done and step_count < num_steps:
        # Select action without noise (deterministic policy)
        action = agent.act(state, add_noise=False)
        
        # Get actual next state from environment
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        # Predict next state using state predictor
        if agent.enable_predictor:
            predicted_next_state = agent.predict_next_state(state, action)
            
            # Store states for comparison
            actual_states.append(next_state)
            predicted_states.append(predicted_next_state)
        
        # Update state
        state = next_state
        step_count += 1
    
    # Calculate prediction accuracy
    if agent.enable_predictor and actual_states and predicted_states:
        actual_array = np.array(actual_states)
        predicted_array = np.array(predicted_states)
        
        # Mean squared error
        mse = np.mean((actual_array - predicted_array) ** 2)
        return total_reward, mse
    
    return total_reward, float('inf')

def save_training_curves(rewards, avg_rewards, prediction_errors, prediction_valid_tm_indices, config_name):
    """
    Save training curves as plots.
    
    Args:
        rewards: List of episode rewards
        avg_rewards: List of average rewards
        prediction_errors: List of prediction errors
        prediction_valid_tm_indices: Traffic matrix indices for prediction errors
        config_name: Configuration name
    """
    # Create plots directory and get absolute path
    os.makedirs("plots", exist_ok=True)
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "plots")
    
    # Plot rewards
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, alpha=0.3, color='blue', label='Episode Reward')
    plt.plot(avg_rewards, color='red', linewidth=2, label='Average Reward (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Training Rewards ({config_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, f"rewards_{config_name}.png"))
    plt.close()
    
    # Plot prediction errors if available
    if prediction_errors:
        plt.figure(figsize=(12, 6))
        
        # Create colormap based on traffic matrix
        unique_tm_indices = sorted(list(set(prediction_valid_tm_indices)))
        cmap = plt.cm.get_cmap('tab10', len(unique_tm_indices))
        colors = [cmap(unique_tm_indices.index(tm_idx)) for tm_idx in prediction_valid_tm_indices]
        
        # Plot scatter with color-coded points
        plt.scatter(range(len(prediction_errors)), prediction_errors, c=colors, alpha=0.7, s=10)
        
        # Add smoothed line
        window_size = min(100, len(prediction_errors)//10) if len(prediction_errors) > 100 else len(prediction_errors)
        if window_size > 0:
            smoothed = np.convolve(prediction_errors, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(prediction_errors)), smoothed, color='red', linewidth=2, label=f'Moving Average (window={window_size})')
        
        plt.xlabel('Training Step')
        plt.ylabel('Prediction Error (MSE)')
        plt.title(f'State Prediction Error ({config_name})')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, f"prediction_errors_{config_name}.png"))
        
        # Create color legend for traffic matrices
        legend_fig = plt.figure(figsize=(8, 2))
        ax = legend_fig.add_subplot(111)
        for i, tm_idx in enumerate(unique_tm_indices):
            ax.scatter([], [], c=[cmap(i)], label=f'TM {tm_idx}')
        ax.legend(ncol=min(10, len(unique_tm_indices)), loc='center')
        ax.axis('off')
        legend_fig.savefig(os.path.join(plots_dir, f"prediction_errors_legend_{config_name}.png"))
        plt.close('all')
    
    plot_path = os.path.join(plots_dir, f"rewards_{config_name}.png")
    pred_plot_path = os.path.join(plots_dir, f"prediction_errors_{config_name}.png")
    print(f"Training plots saved to {plot_path}" + (f" and {pred_plot_path}" if prediction_errors else ""))

def main():
    parser = argparse.ArgumentParser(description='Train a DDPG agent for relaxed network optimization')
    
    # Configuration
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    
    # Network architecture
    parser.add_argument('--latent-dim', type=int, default=64, help='Dimension of latent state representation')
    parser.add_argument('--predictor-hidden-dim', type=int, default=256, help='Hidden dimension of state predictor')
    parser.add_argument('--actor-hidden-dim', type=int, default=256, help='Hidden dimension of actor network')
    parser.add_argument('--critic-hidden-dim', type=int, default=256, help='Hidden dimension of critic network')
    parser.add_argument('--enable-predictor', action='store_true', help='Enable state predictor training')
    parser.add_argument('--validate-predictor', action='store_true', help='Periodically validate predictor during training')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage instead of CUDA')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes per traffic matrix')
    parser.add_argument('--tm-rounds', type=int, default=3, help='Number of rounds to cycle through all traffic matrices')
    parser.add_argument('--actor-lr', type=float, default=1e-4, help='Learning rate for actor network')
    parser.add_argument('--critic-lr', type=float, default=1e-3, help='Learning rate for critic network')
    parser.add_argument('--predictor-lr', type=float, default=1e-3, help='Learning rate for state predictor')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--tau', type=float, default=1e-3, help='Soft update parameter')
    parser.add_argument('--noise-scale', type=float, default=0.2, help='Scale of exploration noise')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--predictor-batch-size', type=int, default=128, help='Batch size for predictor training')
    parser.add_argument('--buffer-size', type=int, default=100000, help='Replay buffer size')
    
    # Environment settings
    parser.add_argument('--random-edge-order', action='store_true', help='Use random edge ordering in environment')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Saving and validation
    parser.add_argument('--save-freq', type=int, default=10000, help='Save model every N episodes')
    parser.add_argument('--validation-freq', type=int, default=5000, help='Validate predictor every N episodes')
    
    args = parser.parse_args()
    train_agent(args)

if __name__ == "__main__":
    main()
