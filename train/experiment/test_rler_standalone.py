"""
Standalone test script for RL-Based Energy Routing (RL-ER) baseline
Run this to get individual metrics for RL-ER algorithm
"""

import sys
import json
import numpy as np
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from env import SDNEnv
from baselines.rl_energy_routing import RLEnergyRouting


def load_config(n_links=100):
    """Load configuration from JSON file"""
    config_file = Path(__file__).parent / f"configs/topology_{n_links}.json"
    
    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_file}\n"
            f"Available configs: 20, 100, 500, 1000, 2000 links"
        )
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return config


def test_rler(n_links=100, episodes=500, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
    """
    Test RL-Based Energy Routing algorithm
    
    Args:
        n_links: Network size (number of links)
        episodes: Number of training/test episodes
        learning_rate: Q-learning alpha
        discount_factor: Q-learning gamma
        epsilon: Exploration rate
    
    Returns:
        dict: Performance metrics
    """
    print("=" * 80)
    print("RL-BASED ENERGY ROUTING (RL-ER) - STANDALONE TEST")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Network Size: {n_links} links")
    print(f"  Episodes: {episodes}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Discount Factor: {discount_factor}")
    print(f"  Epsilon: {epsilon}")
    print("\n" + "-" * 80)
    
    # Initialize
    config = load_config(n_links)
    env = SDNEnv(config)
    rl_er = RLEnergyRouting(
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon
    )
    
    # Metrics storage
    rewards = []
    energy_savings = []
    latencies = []
    sla_violations = []
    
    print(f"\n🚀 Running {episodes} episodes (training + evaluation)...\n")
    
    for ep in range(episodes):
        obs = env.reset()
        state = rl_er.get_state(env.G_full, env._flows)
        total_r = 0.0
        
        ep_energy = []
        ep_latency = []
        ep_sla = []
        
        for t in range(config['max_steps_per_episode']):
            # Select action (which link to deactivate)
            action = rl_er.select_action(state, env.G_full)
            
            # Apply action
            if action:
                u, v = action
                env.G_full[u][v]['active'] = 0
            
            # Measure performance
            latency, sla_viol, _ = env._route_and_measure()
            energy = env._energy_cost()
            
            # Calculate energy saving
            base_all_on = env.energy_on * env.G_full.number_of_edges()
            energy_saving = (base_all_on - energy) / base_all_on if base_all_on > 0 else 0
            
            # Calculate reward
            reward = energy_saving - 0.001 * latency - 0.05 * sla_viol
            total_r += reward
            
            # Get next state
            next_state = rl_er.get_state(env.G_full, env._flows)
            
            # Update Q-table
            if action:
                rl_er.update(state, action, reward, next_state)
            
            state = next_state
            
            ep_energy.append(energy_saving * 100)
            ep_latency.append(latency)
            ep_sla.append(sla_viol)  # Already in percentage from _route_and_measure()
            
            # Generate new flows
            env._generate_new_flows()
        
        # Store episode metrics
        rewards.append(total_r)
        energy_savings.append(np.mean(ep_energy))
        latencies.append(np.mean(ep_latency))
        sla_violations.append(np.mean(ep_sla))
        
        # Progress update
        if (ep + 1) % 50 == 0 or ep == 0:
            recent_reward = np.mean(rewards[-50:])
            recent_energy = np.mean(energy_savings[-50:])
            recent_latency = np.mean(latencies[-50:])
            recent_sla = np.mean(sla_violations[-50:])
            print(f"  Episode {ep+1:4d}/{episodes}: "
                  f"Reward={recent_reward:6.3f}, "
                  f"Energy={recent_energy:5.1f}%, "
                  f"Latency={recent_latency:5.2f}ms, "
                  f"SLA Viol={recent_sla:5.2f}%")
    
    # Get algorithm statistics
    stats = rl_er.get_stats()
    
    # Calculate final metrics (average of last 20% episodes)
    last_n = max(int(episodes * 0.2), 20)
    
    results = {
        'reward': np.mean(rewards[-last_n:]),
        'reward_std': np.std(rewards[-last_n:]),
        'energy_saving': np.mean(energy_savings[-last_n:]),
        'energy_saving_std': np.std(energy_savings[-last_n:]),
        'latency': np.mean(latencies[-last_n:]),
        'latency_std': np.std(latencies[-last_n:]),
        'sla_violations': np.mean(sla_violations[-last_n:]),
        'sla_violations_std': np.std(sla_violations[-last_n:]),
        'computation_time': stats['avg_computation_time'],
        'q_table_size': stats['q_table_size'],
        'total_updates': stats['total_updates']
    }
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS (averaged over last 20% episodes)")
    print("=" * 80)
    print(f"\n📊 Performance Metrics:")
    print(f"  Reward:              {results['reward']:.4f} ± {results['reward_std']:.4f}")
    print(f"  Energy Saving:       {results['energy_saving']:.2f}% ± {results['energy_saving_std']:.2f}%")
    print(f"  Latency:             {results['latency']:.2f} ± {results['latency_std']:.2f} ms")
    print(f"  SLA Violation Rate:  {results['sla_violations']:.2f}% ± {results['sla_violations_std']:.2f}%")
    print(f"  Computation Time:    {results['computation_time']:.6f} seconds")
    print(f"\n🧠 Learning Statistics:")
    print(f"  Q-Table Size:        {results['q_table_size']} states")
    print(f"  Total Updates:       {results['total_updates']}")
    
    print("\n" + "=" * 80)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test RL-Based Energy Routing baseline')
    parser.add_argument('--links', type=int, default=100, 
                        help='Number of links (20, 100, 500, 1000, 2000)')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate (alpha)')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='Discount factor (gamma)')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Exploration rate')
    
    args = parser.parse_args()
    
    results = test_rler(
        n_links=args.links,
        episodes=args.episodes,
        learning_rate=args.lr,
        discount_factor=args.gamma,
        epsilon=args.epsilon
    )
    
    # Save results to file
    import json
    output_file = f"results/rler_standalone_{args.links}links.json"
    Path("results").mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")
