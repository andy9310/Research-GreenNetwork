"""
Main Experiment Runner
Compares three methods across different network topologies
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from env import SDNEnv
from agent import HierarchicalDQN
from baselines.energy_aware_routing import EnergyAwareRouting
from baselines.rl_energy_routing import RLEnergyRouting


class ExperimentRunner:
    """
    Runs comparative experiments across methods and topologies
    """
    
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(f"{output_dir}/raw", exist_ok=True)
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        
        self.results = []
        
    def run_all_experiments(self, topologies=[20, 100, 500, 1000, 2000], 
                           methods=['dqn_clustering', 'energy_aware', 'rl_basic'],
                           episodes_per_method={'dqn_clustering': 2000, 'energy_aware': 100, 'rl_basic': 1000}):
        """
        Run all experiments
        
        Args:
            topologies: List of network sizes (number of links)
            methods: List of methods to compare
            episodes_per_method: Dict of episodes for each method
        """
        print("=" * 80)
        print("COMPARATIVE EXPERIMENT: ENERGY-AWARE SDN ROUTING")
        print("=" * 80)
        
        for topology in topologies:
            print(f"\n{'='*80}")
            print(f"TOPOLOGY: {topology} links")
            print(f"{'='*80}")
            
            config = self._generate_config(topology)
            
            for method in methods:
                print(f"\n{'-'*80}")
                print(f"METHOD: {method}")
                print(f"{'-'*80}")
                
                episodes = episodes_per_method.get(method, 100)
                result = self.run_single_experiment(method, config, episodes)
                
                result['topology'] = topology
                result['method'] = method
                self.results.append(result)
                
                self._save_result(result)
        
        # Generate summary
        self._generate_summary()
        
    def run_single_experiment(self, method, config, episodes):
        """
        Run single experiment for one method
        
        Args:
            method: 'dqn_clustering', 'energy_aware', or 'rl_basic'
            config: Configuration dict
            episodes: Number of episodes
            
        Returns:
            result: Dict of metrics
        """
        print(f"\n🚀 Running {method} for {episodes} episodes...")
        
        start_time = time.time()
        
        if method == 'dqn_clustering':
            result = self._run_dqn(config, episodes)
        elif method == 'energy_aware':
            result = self._run_energy_aware(config, episodes)
        elif method == 'rl_basic':
            result = self._run_rl_basic(config, episodes)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        result['training_time'] = time.time() - start_time
        
        print(f"\n✅ Completed in {result['training_time']:.2f}s")
        print(f"   Energy Saving: {result['energy_saving']:.1f}%")
        print(f"   Latency: {result['latency']:.2f}ms")
        print(f"   SLA Violations: {result['sla_violations']:.1f}%")
        print(f"   Computation Time: {result['computation_time']:.4f}s")
        
        return result
    
    def _run_dqn(self, config, episodes):
        """Run DQN with clustering"""
        env = SDNEnv(config)
        obs = env.reset()
        agent = HierarchicalDQN(obs_dim=obs.shape[0], action_n=env.action_n, cfg=config, device='cpu')
        
        rewards = []
        energy_savings = []
        latencies = []
        sla_violations = []
        comp_times = []
        
        for ep in range(episodes):
            obs = env.reset()
            total_r = 0.0
            
            for t in range(config['max_steps_per_episode']):
                start = time.time()
                a = agent.act(obs)
                comp_times.append(time.time() - start)
                
                obs2, r, done, info = env.step(a)
                agent.push(obs, a, r, obs2, float(done))
                obs = obs2
                total_r += r
                
                if len(agent.rb) >= config['batch_size']:
                    agent.train_step()
                
                if t % 100 == 0:
                    agent.update_target()
                
                if done:
                    break
            
            rewards.append(total_r)
            energy_savings.append(info.get('energy_saving', 0) * 100)
            latencies.append(info.get('latency_ms', 0))
            sla_violations.append(info.get('sla_viol', 0))
            
            if (ep + 1) % 100 == 0:
                print(f"  Episode {ep+1}/{episodes}: Reward={total_r:.2f}, Energy={energy_savings[-1]:.1f}%")
        
        return {
            'reward': np.mean(rewards[-100:]),
            'energy_saving': np.mean(energy_savings[-100:]),
            'latency': np.mean(latencies[-100:]),
            'sla_violations': np.mean(sla_violations[-100:]),
            'computation_time': np.mean(comp_times)
        }
    
    def _run_energy_aware(self, config, episodes):
        """Run Energy-Aware Routing heuristic"""
        env = SDNEnv(config)
        ear = EnergyAwareRouting(utilization_threshold=0.3, min_active_ratio=0.2)
        
        energy_savings = []
        latencies = []
        sla_violations = []
        
        for ep in range(episodes):
            obs = env.reset()
            
            for t in range(config['max_steps_per_episode']):
                # Get active links from EAR
                active_links, metrics = ear.select_links(env.G_full, env._flows, None)
                
                # Apply to environment
                for u, v in env.G_full.edges():
                    env.G_full[u][v]['active'] = 1 if (u, v) in active_links or (v, u) in active_links else 0
                
                # Measure performance
                latency, sla_viol, _ = env._route_and_measure()
                energy = env._energy_cost()
                base_all_on = env.energy_on * env.G_full.number_of_edges()
                energy_saving = (base_all_on - energy) / base_all_on * 100
                
                energy_savings.append(energy_saving)
                latencies.append(latency)
                sla_violations.append(sla_viol)
                
                # Generate new flows
                env._generate_new_flows()
            
            if (ep + 1) % 20 == 0:
                print(f"  Episode {ep+1}/{episodes}: Energy={np.mean(energy_savings[-100:]):.1f}%")
        
        stats = ear.get_stats()
        
        return {
            'reward': 0,  # Not applicable
            'energy_saving': np.mean(energy_savings[-100:]),
            'latency': np.mean(latencies[-100:]),
            'sla_violations': np.mean(sla_violations[-100:]),
            'computation_time': stats['avg_computation_time']
        }
    
    def _run_rl_basic(self, config, episodes):
        """Run basic RL without clustering"""
        env = SDNEnv(config)
        rl_er = RLEnergyRouting(learning_rate=0.1, discount_factor=0.95, epsilon=0.1)
        
        rewards = []
        energy_savings = []
        latencies = []
        sla_violations = []
        
        for ep in range(episodes):
            obs = env.reset()
            state = rl_er.get_state(env.G_full, env._flows)
            total_r = 0.0
            
            for t in range(config['max_steps_per_episode']):
                # Select action
                action = rl_er.select_action(state, env.G_full)
                
                # Apply action (deactivate link)
                if action:
                    u, v = action
                    env.G_full[u][v]['active'] = 0
                
                # Measure performance
                latency, sla_viol, _ = env._route_and_measure()
                energy = env._energy_cost()
                base_all_on = env.energy_on * env.G_full.number_of_edges()
                energy_saving = (base_all_on - energy) / base_all_on
                
                reward = energy_saving - 0.001 * latency - 0.05 * sla_viol
                total_r += reward
                
                # Get next state
                next_state = rl_er.get_state(env.G_full, env._flows)
                
                # Update Q-table
                rl_er.update(state, action, reward, next_state)
                state = next_state
                
                energy_savings.append(energy_saving * 100)
                latencies.append(latency)
                sla_violations.append(sla_viol)
                
                # Generate new flows
                env._generate_new_flows()
            
            rewards.append(total_r)
            
            if (ep + 1) % 100 == 0:
                print(f"  Episode {ep+1}/{episodes}: Reward={total_r:.2f}, Energy={np.mean(energy_savings[-100:]):.1f}%")
        
        stats = rl_er.get_stats()
        
        return {
            'reward': np.mean(rewards[-100:]),
            'energy_saving': np.mean(energy_savings[-100:]),
            'latency': np.mean(latencies[-100:]),
            'sla_violations': np.mean(sla_violations[-100:]),
            'computation_time': stats['avg_computation_time']
        }
    
    def _generate_config(self, num_links):
        """Generate config for given topology size"""
        # Calculate nodes from links (assuming ~10 links per node)
        num_nodes = max(20, int(num_links / 10))
        num_hosts = int(num_nodes * 0.4)
        
        return {
            'seed': 42,
            'device': 'cpu',
            'episodes': 100,
            'max_steps_per_episode': 200,
            'train_every': 1,
            'target_update': 100,
            'buffer_size': 100000,
            'batch_size': 256,
            'gamma': 0.95,
            'lr': 0.0003,
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay_steps': 5000,
            'use_double_dqn': True,
            'grad_clip_norm': 1.0,
            'obs_aggregation': 'cluster',
            'energy_per_link_on': 1.0,
            'energy_per_link_sleep': 0.1,
            'congestion_delay_factor': 2.0,
            'routing_k_paths': 3,
            'num_nodes': num_nodes,
            'num_edges': num_links,
            'num_regions': 3,
            'edge_capacity_mean': 5.0,
            'edge_capacity_std': 1.0,
            'edge_base_delay_ms': 0.3,
            'host_ratio': 0.4,
            'num_hosts': num_hosts,
            'sla_latency_ms': {1: 1.0, 2: 2.0, 3: 4.0, 4: 8.0, 5: 16.0, 6: 32.0},
            'traffic_load_mode': 'low',
            'traffic_modes': {
                'low': {
                    'target_utilization_range': [0.2, 0.4],
                    'flow_intensity_multiplier': 2.0,
                    'peak_flow_probability': 0.7,
                    'offpeak_flow_probability': 0.4,
                    'description': '20-40% link capacity utilization'
                }
            },
            'flow_size_bytes_min': 1000,
            'flow_size_bytes_max': 10000,
            'peak_flow_interval_s': [1, 3],
            'offpeak_flow_interval_s': [3, 7],
            'peak_size_range': [5000, 10000],
            'offpeak_size_range': [1000, 5000],
            'peaks': {'region_0': [0, 60], 'region_1': [60, 120], 'region_2': [120, 180]},
            'recluster_every_steps': 30,
            'cluster_threshold_bins': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'inter_cluster_keep_min': [2, 3, 4, 5],
            'deterministic_refinement': True,
            'adaptive_clustering': True,
            'clustering_method': 'dp_means_adaptive',
            'clustering_k_range': [2, 10],
            'dp_means_lambda': None,
            'num_clusters': 3,
            'no_clustering': False,
            'eval_episodes': 5,
            'log_every': 5,
            'deactivation_algorithm': 'priority'
        }
    
    def _save_result(self, result):
        """Save individual result"""
        filename = f"{self.output_dir}/raw/{result['method']}_{result['topology']}.json"
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
    
    def _generate_summary(self):
        """Generate summary CSV"""
        import csv
        
        filename = f"{self.output_dir}/comparison_table.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Method', 'Topology', 'Links', 'Energy_Saving_%', 
                                                   'Latency_ms', 'SLA_Violations_%', 'Comp_Time_s', 'Training_Time_s'])
            writer.writeheader()
            
            for result in self.results:
                writer.writerow({
                    'Method': result['method'],
                    'Topology': result['topology'],
                    'Links': result['topology'],
                    'Energy_Saving_%': f"{result['energy_saving']:.1f}",
                    'Latency_ms': f"{result['latency']:.2f}",
                    'SLA_Violations_%': f"{result['sla_violations']:.1f}",
                    'Comp_Time_s': f"{result['computation_time']:.4f}",
                    'Training_Time_s': f"{result['training_time']:.2f}"
                })
        
        print(f"\n✅ Summary saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(description='Run comparative experiments')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--method', type=str, choices=['dqn_clustering', 'energy_aware', 'rl_basic'], 
                       help='Run specific method')
    parser.add_argument('--topology', type=int, choices=[20, 100, 500, 1000, 2000], 
                       help='Run specific topology')
    parser.add_argument('--episodes', type=int, default=None, help='Override episode count')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner()
    
    if args.all:
        runner.run_all_experiments()
    elif args.method and args.topology:
        config = runner._generate_config(args.topology)
        episodes = args.episodes or 100
        result = runner.run_single_experiment(args.method, config, episodes)
        result['topology'] = args.topology
        result['method'] = args.method
        runner._save_result(result)
    else:
        print("Please specify --all or both --method and --topology")
        parser.print_help()


if __name__ == "__main__":
    main()
