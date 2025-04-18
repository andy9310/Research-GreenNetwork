#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comparison of Monte Carlo and Q-Learning methods for network optimization.
This script visualizes the learning process of both methods with identical parameters.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import torch
from tqdm import tqdm
import time
import copy
import sys

# Add paths to both implementation directories
sys.path.append('MonteCarlo')
sys.path.append('Qlearning')

# Import both implementations (ensure naming doesn't conflict)
from MonteCarlo.env import NetworkEnv as MCEnv
from MonteCarlo.agent import MonteCarloAgent, EpisodeBuffer

from Qlearning.env import NetworkEnv as DQNEnv  
from Qlearning.agent import DQN as DQNAgent, ReplayBuffer

# Create a shared base environment for fair comparison
def load_config(config_path):
    """Load configuration from file."""
    # Check if the path is a relative path without directory
    if '/' not in config_path and '\\' not in config_path:
        # Prepend configs directory path
        config_path = f"configs/{config_path}"
    
    # Now open and load the config file
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def smooth_rewards(rewards, window_size=100):
    """Apply smoothing to the rewards for clearer visualization."""
    if len(rewards) < window_size:
        return rewards
    
    smoothed = []
    for i in range(len(rewards)):
        start_idx = max(0, i - window_size + 1)
        smoothed.append(np.mean(rewards[start_idx:i+1]))
    return smoothed

class TrainingMetrics:
    """Track and store metrics during training."""
    
    def __init__(self):
        self.episode_rewards = []
        self.smoothed_rewards = []
        self.violations_per_episode = []  # Count of violations per episode
        self.violation_types = []  # Type of violations (isolation, overload)
        self.episode_steps = []  # Number of steps per episode
        self.successful_episodes = []  # Episodes with no violations
        self.early_terminations = []  # Episodes that terminated early
        self.links_closed = []  # Number of links closed in each episode
        self.training_times = []  # Time taken for each batch of episodes
        
    def add_episode_metrics(self, reward, steps, violations, links_closed, violation_type=None):
        """Add metrics for a single episode."""
        self.episode_rewards.append(reward)
        self.episode_steps.append(steps)
        self.violations_per_episode.append(violations)
        self.successful_episodes.append(violations == 0)
        self.early_terminations.append(steps < 8)  # Assuming 8 links in the network
        self.links_closed.append(links_closed)
        self.violation_types.append(violation_type)
        
    def update_smoothed_rewards(self, window_size=100):
        """Update smoothed reward calculations."""
        self.smoothed_rewards = smooth_rewards(self.episode_rewards, window_size)
        
    def add_training_time(self, time):
        """Record training time."""
        self.training_times.append(time)
        
    def get_success_rate(self, window=100):
        """Calculate success rate over recent episodes."""
        if not self.successful_episodes:
            return 0
        
        recent = self.successful_episodes[-window:]
        return sum(recent) / len(recent)
    
    def get_avg_links_closed(self, window=100):
        """Calculate average links closed over recent successful episodes."""
        if not self.successful_episodes:
            return 0
        
        recent_success_idx = [i for i, success in enumerate(self.successful_episodes[-window:]) if success]
        if not recent_success_idx:
            return 0
        
        recent_closed = [self.links_closed[-window:][i] for i in recent_success_idx]
        return np.mean(recent_closed) if recent_closed else 0
        
def train_monte_carlo(config, args):
    """Train a Monte Carlo agent and track metrics."""
    
    # Environment setup
    env = MCEnv(
        adj_matrix=config["adj_matrix"],
        edge_list=config["edge_list"],
        tm_list=config["tm_list"],
        node_props=config["node_props"],
        num_nodes=config["num_nodes"],
        link_capacity=config["link_capacity"],
        max_edges=config["max_edges"],
        seed=args.seed
    )
    
    # Get dimensions for the agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create agent
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    agent = MonteCarloAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        lr=args.learning_rate,
        gamma=args.gamma,
        device=device,
        network_type=args.architecture,
        nhead=4,  # For transformer
        num_layers=2  # For transformer
    )
    
    # Episode buffer
    episode_buffer = EpisodeBuffer(capacity=1000)
    
    # Metrics tracking
    metrics = TrainingMetrics()
    
    print("\nStarting Monte Carlo Training...")
    
    # Track total steps for epsilon calculation
    total_steps = 0
    
    # Training loop
    for tm_idx, traffic_matrix in enumerate(config["tm_list"]):
        # Set current traffic matrix
        env.current_tm_idx = tm_idx
        
        # Training on this traffic matrix
        print(f"\nTraining on traffic matrix {tm_idx+1}/{len(config['tm_list'])}...")
        
        for episode in tqdm(range(args.episodes_per_tm), desc=f"MC-TM {tm_idx+1}"):
            # Start episode
            start_time = time.time()
            state, _, _, _, _ = env.reset()
            episode_reward = 0
            done = False
            steps = 0
            violations = 0
            violation_type = None
            
            # Track link statuses for this episode
            link_statuses = np.ones(env.num_edges, dtype=int)  # Start with all open
            
            # Episode loop
            while not done:
                # Calculate epsilon based on total steps
                total_steps += 1
                epsilon = max(
                    args.epsilon_min,
                    args.epsilon_start - (total_steps / args.epsilon_decay_steps) * 
                    (args.epsilon_start - args.epsilon_min)
                )
                
                # Select and execute action
                action = agent.select_action(state, epsilon)
                next_state, reward, done, _, info = env.step(action)
                
                # Track link status (0=closed, 1=open)
                if not done:  # Only update if not terminating
                    link_statuses[env.current_edge_idx-1] = action
                
                # Add to episode buffer
                episode_buffer.add_experience(state, action, reward, next_state, done)
                
                # Update state and counters
                state = next_state
                episode_reward += reward
                steps += 1
                
                # Check for violations
                if info['violation']:
                    violations += 1
                    violation_type = info['violation']
                
                # If done, end episode
                if done:
                    episode_buffer.end_episode()
                    
                    # Perform learning if we have episodes
                    if len(episode_buffer) > 0:
                        agent.learn(episode_buffer, batch_size=min(32, len(episode_buffer)))
            
            # Calculate links closed at end of episode
            links_closed = env.num_edges - np.sum(link_statuses)
            
            # Record episode metrics
            metrics.add_episode_metrics(
                reward=episode_reward,
                steps=steps,
                violations=violations,
                links_closed=links_closed,
                violation_type=violation_type
            )
            
            # Record training time
            metrics.add_training_time(time.time() - start_time)
            
            # Update smoothed rewards
            if (episode + 1) % 10 == 0:
                metrics.update_smoothed_rewards()
    
    return metrics

def train_q_learning(config, args):
    """Train a Q-learning agent and track metrics."""
    
    # Environment setup
    env = DQNEnv(
        adj_matrix=config["adj_matrix"],
        edge_list=config["edge_list"],
        tm_list=config["tm_list"],
        node_props=config["node_props"],
        num_nodes=config["num_nodes"],
        link_capacity=config["link_capacity"],
        max_edges=config["max_edges"],
        seed=args.seed
    )
    
    # Get dimensions for the agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create agent
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        lr=args.learning_rate,
        gamma=args.gamma,
        device=device,
        network_type=args.architecture
    )
    
    # Create replay buffer separately
    replay_buffer = ReplayBuffer(capacity=10000)
    
    # Metrics tracking
    metrics = TrainingMetrics()
    
    print("\nStarting Q-Learning Training...")
    
    # Track total steps for epsilon calculation
    total_steps = 0
    
    # Training loop
    for tm_idx, traffic_matrix in enumerate(config["tm_list"]):
        # Set current traffic matrix
        env.current_tm_idx = tm_idx
        
        # Training on this traffic matrix
        print(f"\nTraining on traffic matrix {tm_idx+1}/{len(config['tm_list'])}...")
        
        for episode in tqdm(range(args.episodes_per_tm), desc=f"DQN-TM {tm_idx+1}"):
            # Start episode
            start_time = time.time()
            state, _, _, _, _ = env.reset()
            episode_reward = 0
            done = False
            steps = 0
            violations = 0
            violation_type = None
            
            # Track link statuses for this episode
            link_statuses = np.ones(env.num_edges, dtype=int)  # Start with all open
            
            # Episode loop
            while not done:
                # Calculate epsilon based on total steps
                total_steps += 1
                epsilon = max(
                    args.epsilon_min,
                    args.epsilon_start - (total_steps / args.epsilon_decay_steps) * 
                    (args.epsilon_start - args.epsilon_min)
                )
                
                # Select and execute action
                action = agent.select_action(state, epsilon)
                next_state, reward, done, _, info = env.step(action)
                
                # Track link status (0=closed, 1=open)
                if not done:  # Only update if not terminating
                    link_statuses[env.current_edge_idx-1] = action
                
                # Add to replay buffer
                replay_buffer.push(state, action, reward, next_state, done)
                
                # Update state and counters
                state = next_state
                episode_reward += reward
                steps += 1
                
                # Check for violations
                if info['violation']:
                    violations += 1
                    violation_type = info['violation']
                
                # Perform learning every step
                if len(replay_buffer) > 32:  # Minimum batch size
                    agent.learn(replay_buffer, 32)
            
            # Calculate links closed at end of episode
            links_closed = env.num_edges - np.sum(link_statuses)
            
            # Record episode metrics
            metrics.add_episode_metrics(
                reward=episode_reward,
                steps=steps,
                violations=violations,
                links_closed=links_closed,
                violation_type=violation_type
            )
            
            # Record training time
            metrics.add_training_time(time.time() - start_time)
            
            # Update smoothed rewards
            if (episode + 1) % 10 == 0:
                metrics.update_smoothed_rewards()
    
    return metrics

def plot_comparison(mc_metrics, dqn_metrics, args):
    """Generate comparison plots."""
    
    # Create visualization directory if it doesn't exist
    os.makedirs("visualizations", exist_ok=True)
    prefix = f"{args.architecture}_{os.path.splitext(os.path.basename(args.config))[0]}"
    
    # Create figure for rewards comparison
    plt.figure(figsize=(12, 7))
    plt.plot(mc_metrics.smoothed_rewards, label='Monte Carlo', color='blue')
    plt.plot(dqn_metrics.smoothed_rewards, label='Q-Learning', color='red')
    plt.title('Average Reward During Training')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'visualizations/{prefix}_rewards_comparison.png', dpi=300)
    
    # Create figure for success rate
    window = 100
    mc_success = []
    dqn_success = []
    
    for i in range(window, len(mc_metrics.successful_episodes), window):
        mc_success.append(sum(mc_metrics.successful_episodes[i-window:i]) / window * 100)
        
    for i in range(window, len(dqn_metrics.successful_episodes), window):
        dqn_success.append(sum(dqn_metrics.successful_episodes[i-window:i]) / window * 100)
    
    plt.figure(figsize=(12, 7))
    plt.plot(range(window, len(mc_success)*window + 1, window), mc_success, label='Monte Carlo', color='blue')
    plt.plot(range(window, len(dqn_success)*window + 1, window), dqn_success, label='Q-Learning', color='red')
    plt.title('Success Rate (Episodes Without Violations)')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'visualizations/{prefix}_success_rate_comparison.png', dpi=300)
    
    # Create figure for links closed in successful episodes
    mc_links = []
    dqn_links = []
    
    for i in range(window, len(mc_metrics.successful_episodes), window):
        # Get indices of successful episodes in this window
        success_indices = [j for j in range(i-window, i) if mc_metrics.successful_episodes[j]]
        if success_indices:
            # Calculate average links closed in successful episodes
            avg_links = sum(mc_metrics.links_closed[j] for j in success_indices) / len(success_indices)
            mc_links.append(avg_links)
        else:
            mc_links.append(0)
            
    for i in range(window, len(dqn_metrics.successful_episodes), window):
        # Get indices of successful episodes in this window
        success_indices = [j for j in range(i-window, i) if dqn_metrics.successful_episodes[j]]
        if success_indices:
            # Calculate average links closed in successful episodes
            avg_links = sum(dqn_metrics.links_closed[j] for j in success_indices) / len(success_indices)
            dqn_links.append(avg_links)
        else:
            dqn_links.append(0)
    
    plt.figure(figsize=(12, 7))
    plt.plot(range(window, len(mc_links)*window + 1, window), mc_links, label='Monte Carlo', color='blue')
    plt.plot(range(window, len(dqn_links)*window + 1, window), dqn_links, label='Q-Learning', color='red')
    plt.title('Average Links Closed in Successful Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Average Links Closed')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'visualizations/{prefix}_links_closed_comparison.png', dpi=300)
    
    # Create figure for early termination rate
    mc_early = []
    dqn_early = []
    
    for i in range(window, len(mc_metrics.early_terminations), window):
        mc_early.append(sum(mc_metrics.early_terminations[i-window:i]) / window * 100)
        
    for i in range(window, len(dqn_metrics.early_terminations), window):
        dqn_early.append(sum(dqn_metrics.early_terminations[i-window:i]) / window * 100)
    
    plt.figure(figsize=(12, 7))
    plt.plot(range(window, len(mc_early)*window + 1, window), mc_early, label='Monte Carlo', color='blue')
    plt.plot(range(window, len(dqn_early)*window + 1, window), dqn_early, label='Q-Learning', color='red')
    plt.title('Early Termination Rate (Due to Violations)')
    plt.xlabel('Episode')
    plt.ylabel('Early Termination Rate (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'visualizations/{prefix}_early_termination_comparison.png', dpi=300)
    
    # Save metrics data for later analysis
    metrics_data = {
        'monte_carlo': {
            'rewards': mc_metrics.episode_rewards,
            'smoothed_rewards': mc_metrics.smoothed_rewards,
            'success_rate': mc_success,
            'links_closed': mc_links,
            'early_termination_rate': mc_early,
            'training_times': mc_metrics.training_times
        },
        'q_learning': {
            'rewards': dqn_metrics.episode_rewards,
            'smoothed_rewards': dqn_metrics.smoothed_rewards,
            'success_rate': dqn_success,
            'links_closed': dqn_links,
            'early_termination_rate': dqn_early,
            'training_times': dqn_metrics.training_times
        },
        'parameters': {
            'episodes_per_tm': args.episodes_per_tm,
            'hidden_dim': args.hidden_dim,
            'architecture': args.architecture,
            'learning_rate': args.learning_rate,
            'gamma': args.gamma,
            'epsilon_start': args.epsilon_start,
            'epsilon_min': args.epsilon_min,
            'epsilon_decay_steps': args.epsilon_decay_steps,
            'seed': args.seed
        }
    }
    
    with open(f'visualizations/{prefix}_comparison_metrics.json', 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    # Print summary statistics
    print("\n=== Comparison Summary ===")
    print(f"Monte Carlo final avg reward: {np.mean(mc_metrics.episode_rewards[-100:]):.2f}")
    print(f"Q-Learning final avg reward: {np.mean(dqn_metrics.episode_rewards[-100:]):.2f}")
    print(f"Monte Carlo final success rate: {mc_success[-1]:.2f}%")
    print(f"Q-Learning final success rate: {dqn_success[-1]:.2f}%")
    print(f"Monte Carlo avg links closed: {mc_links[-1]:.2f}")
    print(f"Q-Learning avg links closed: {dqn_links[-1]:.2f}")
    print(f"Monte Carlo total training time: {sum(mc_metrics.training_times):.2f}s")
    print(f"Q-Learning total training time: {sum(dqn_metrics.training_times):.2f}s")
    
    # Create overall performance radar chart
    mc_metrics_final = [
        np.mean(mc_metrics.episode_rewards[-100:]) / 100,  # Normalize reward
        mc_success[-1] / 100,  # Already as percentage
        mc_links[-1] / 8,  # Normalize by max links
        (100 - mc_early[-1]) / 100,  # Convert termination to completion rate
        1 - (sum(mc_metrics.training_times) / (sum(mc_metrics.training_times) + sum(dqn_metrics.training_times)))  # Normalize time
    ]
    
    dqn_metrics_final = [
        np.mean(dqn_metrics.episode_rewards[-100:]) / 100,  # Normalize reward
        dqn_success[-1] / 100,  # Already as percentage
        dqn_links[-1] / 8,  # Normalize by max links
        (100 - dqn_early[-1]) / 100,  # Convert termination to completion rate
        1 - (sum(dqn_metrics.training_times) / (sum(mc_metrics.training_times) + sum(dqn_metrics.training_times)))  # Normalize time
    ]
    
    # Create radar chart
    labels = ['Reward', 'Success Rate', 'Links Closed', 'Completion Rate', 'Training Efficiency']
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    mc_metrics_final += mc_metrics_final[:1]  # Close the loop
    dqn_metrics_final += dqn_metrics_final[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.plot(angles, mc_metrics_final, 'b-', linewidth=2, label='Monte Carlo')
    ax.fill(angles, mc_metrics_final, 'b', alpha=0.1)
    ax.plot(angles, dqn_metrics_final, 'r-', linewidth=2, label='Q-Learning')
    ax.fill(angles, dqn_metrics_final, 'r', alpha=0.1)
    
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)
    ax.grid(True)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Method Comparison: Overall Performance')
    plt.tight_layout()
    plt.savefig(f'visualizations/{prefix}_radar_comparison.png', dpi=300)
    
    print("\nVisualization complete. Results saved to 'visualizations/' directory.")

def main():
    """Main function to run the comparison."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Compare Monte Carlo and Q-Learning for network optimization')
    parser.add_argument('--config', type=str, default='config_5node.json', help='Configuration file')
    parser.add_argument('--architecture', type=str, choices=['mlp', 'fat_mlp', 'transformer'], default='mlp', 
                        help='Neural network architecture')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden layer dimension')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--episodes-per-tm', type=int, default=1000, help='Episodes per traffic matrix')
    parser.add_argument('--epsilon-start', type=float, default=1.0, help='Starting epsilon value')
    parser.add_argument('--epsilon-min', type=float, default=0.1, help='Minimum epsilon value')
    parser.add_argument('--epsilon-decay-steps', type=int, default=10000, help='Steps to decay epsilon')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Train Monte Carlo agent
    mc_metrics = train_monte_carlo(config, args)
    
    # Train Q-Learning agent
    dqn_metrics = train_q_learning(config, args)
    
    # Generate comparison plots
    plot_comparison(mc_metrics, dqn_metrics, args)

if __name__ == "__main__":
    main()
