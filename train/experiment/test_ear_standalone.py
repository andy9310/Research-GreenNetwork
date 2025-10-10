"""
Standalone test script for Energy-Aware Routing (EAR) baseline
Run this to get individual metrics for EAR algorithm
"""

import sys
import json
import numpy as np
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from env import SDNEnv
from baselines.energy_aware_routing import EnergyAwareRouting


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


def test_ear(n_links=100, episodes=100, utilization_threshold=0.3, min_active_ratio=0.2):
    """
    Test Energy-Aware Routing algorithm
    
    Args:
        n_links: Network size (number of links)
        episodes: Number of test episodes
        utilization_threshold: Threshold for link deactivation
        min_active_ratio: Minimum fraction of links to keep active
    
    Returns:
        dict: Performance metrics
    """
    print("=" * 80)
    print("ENERGY-AWARE ROUTING (EAR) - STANDALONE TEST")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Network Size: {n_links} links")
    print(f"  Episodes: {episodes}")
    print(f"  Utilization Threshold: {utilization_threshold}")
    print(f"  Min Active Ratio: {min_active_ratio}")
    print("\n" + "-" * 80)
    
    # Initialize
    config = load_config(n_links)
    env = SDNEnv(config)
    ear = EnergyAwareRouting(
        utilization_threshold=utilization_threshold,
        min_active_ratio=min_active_ratio
    )
    
    # Metrics storage
    energy_savings = []
    latencies = []
    sla_violations = []
    active_links_list = []
    
    print(f"\n🚀 Running {episodes} episodes...\n")
    
    for ep in range(episodes):
        obs = env.reset()
        
        ep_energy = []
        ep_latency = []
        ep_sla = []
        ep_active = []
        
        for t in range(config['max_steps_per_episode']):
            # Get active links from EAR algorithm
            active_links, metrics = ear.select_links(env.G_full, env._flows, None)
            
            # Apply link activation to environment
            for u, v in env.G_full.edges():
                is_active = (u, v) in active_links or (v, u) in active_links
                env.G_full[u][v]['active'] = 1 if is_active else 0
            
            # Measure performance
            latency, sla_viol, _ = env._route_and_measure()
            energy = env._energy_cost()
            
            # Calculate energy saving
            base_all_on = env.energy_on * env.G_full.number_of_edges()
            energy_saving = (base_all_on - energy) / base_all_on * 100 if base_all_on > 0 else 0
            
            ep_energy.append(energy_saving)
            ep_latency.append(latency)
            ep_sla.append(sla_viol)  # Already in percentage from _route_and_measure()
            ep_active.append(len(active_links))
            
            # Generate new flows for next step
            env._generate_new_flows()
        
        # Store episode averages
        energy_savings.append(np.mean(ep_energy))
        latencies.append(np.mean(ep_latency))
        sla_violations.append(np.mean(ep_sla))
        active_links_list.append(np.mean(ep_active))
        
        # Progress update
        if (ep + 1) % 10 == 0 or ep == 0:
            recent_energy = np.mean(energy_savings[-10:])
            recent_latency = np.mean(latencies[-10:])
            recent_sla = np.mean(sla_violations[-10:])
            print(f"  Episode {ep+1:3d}/{episodes}: "
                  f"Energy={recent_energy:5.1f}%, "
                  f"Latency={recent_latency:5.2f}ms, "
                  f"SLA Viol={recent_sla:5.2f}%")
    
    # Get algorithm statistics
    stats = ear.get_stats()
    
    # Calculate final metrics (average of last 20% episodes)
    last_n = max(int(episodes * 0.2), 10)
    
    results = {
        'energy_saving': np.mean(energy_savings[-last_n:]),
        'energy_saving_std': np.std(energy_savings[-last_n:]),
        'latency': np.mean(latencies[-last_n:]),
        'latency_std': np.std(latencies[-last_n:]),
        'sla_violations': np.mean(sla_violations[-last_n:]),
        'sla_violations_std': np.std(sla_violations[-last_n:]),
        'computation_time': stats['avg_computation_time'],
        'avg_active_links': np.mean(active_links_list[-last_n:]),
        'total_links': env.G_full.number_of_edges()
    }
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS (averaged over last 20% episodes)")
    print("=" * 80)
    print(f"\n📊 Performance Metrics:")
    print(f"  Energy Saving:       {results['energy_saving']:.2f}% ± {results['energy_saving_std']:.2f}%")
    print(f"  Latency:             {results['latency']:.2f} ± {results['latency_std']:.2f} ms")
    print(f"  SLA Violation Rate:  {results['sla_violations']:.2f}% ± {results['sla_violations_std']:.2f}%")
    print(f"  Computation Time:    {results['computation_time']:.6f} seconds")
    print(f"\n🔗 Link Statistics:")
    print(f"  Total Links:         {results['total_links']}")
    print(f"  Avg Active Links:    {results['avg_active_links']:.1f}")
    print(f"  Avg Deactivated:     {results['total_links'] - results['avg_active_links']:.1f} "
          f"({(1 - results['avg_active_links']/results['total_links'])*100:.1f}%)")
    
    print("\n" + "=" * 80)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Energy-Aware Routing baseline')
    parser.add_argument('--links', type=int, default=100, 
                        help='Number of links (20, 100, 500, 1000, 2000)')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of test episodes')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Utilization threshold for link deactivation')
    parser.add_argument('--min-active', type=float, default=0.2,
                        help='Minimum fraction of links to keep active')
    
    args = parser.parse_args()
    
    results = test_ear(
        n_links=args.links,
        episodes=args.episodes,
        utilization_threshold=args.threshold,
        min_active_ratio=args.min_active
    )
    
    # Save results to file
    import json
    output_file = f"results/ear_standalone_{args.links}links.json"
    Path("results").mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")
