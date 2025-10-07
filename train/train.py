import json, os, time, numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any
import pandas as pd
import signal
import sys

from env import SDNEnv
from agent import HierarchicalDQN

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown_requested
    print("\n\n  Ctrl+C detected! Saving progress and shutting down gracefully...")
    print("  Press Ctrl+C again to force quit (may lose data)")
    shutdown_requested = True

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

class TrainingVisualizer:
    """Class to handle training visualization and metrics collection"""
    
    def __init__(self, save_dir="training_results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Metrics storage
        self.episode_rewards = []
        self.episode_losses = []
        self.episode_energy_savings = []
        self.episode_latencies = []
        self.episode_sla_violations = []
        self.episode_active_links = []
        self.episode_cluster_counts = []
        self.episode_computation_times = []
        
        # Step-by-step metrics
        self.step_rewards = []
        self.step_losses = []
        self.step_energy_savings = []
        self.step_latencies = []
        
    def log_episode(self, episode, reward, loss, energy_saving, latency, sla_viol, active_links, cluster_count=1, computation_time=0.0):
        """Log episode-level metrics"""
        self.episode_rewards.append(reward)
        self.episode_losses.append(loss if loss is not None else 0.0)
        self.episode_energy_savings.append(energy_saving)
        self.episode_latencies.append(latency)
        self.episode_sla_violations.append(sla_viol)
        self.episode_active_links.append(active_links)
        self.episode_cluster_counts.append(cluster_count)
        self.episode_computation_times.append(computation_time)
        
    def log_step(self, reward, loss, energy_saving, latency):
        """Log step-level metrics"""
        self.step_rewards.append(reward)
        self.step_losses.append(loss if loss is not None else 0.0)
        self.step_energy_savings.append(energy_saving)
        self.step_latencies.append(latency)
        
    def plot_training_curves(self, episode):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Training Progress - Episode {episode}', fontsize=16)
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards, 'b-', alpha=0.7)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Episode losses
        if any(l > 0 for l in self.episode_losses):
            axes[0, 1].plot(self.episode_losses, 'r-', alpha=0.7)
            axes[0, 1].set_title('Episode Losses')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True)
        
        # Energy savings
        axes[0, 2].plot(self.episode_energy_savings, 'g-', alpha=0.7)
        axes[0, 2].set_title('Energy Savings (%)')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Energy Saving %')
        axes[0, 2].grid(True)
        
        # Latency
        axes[1, 0].plot(self.episode_latencies, 'm-', alpha=0.7)
        axes[1, 0].set_title('Average Latency (ms)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Latency (ms)')
        axes[1, 0].grid(True)
        
        # SLA violations
        axes[1, 1].plot(self.episode_sla_violations, 'orange', alpha=0.7)
        axes[1, 1].set_title('SLA Violations (%)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('SLA Violation %')
        axes[1, 1].grid(True)
        
        # Active links
        axes[1, 2].plot(self.episode_active_links, 'purple', alpha=0.7)
        axes[1, 2].set_title('Active Links')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Number of Active Links')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/training_curves_episode_{episode}.png', dpi=150, bbox_inches='tight')
        plt.close()  # Close the figure instead of showing it
        
    def save_metrics(self, episode):
        """Save metrics to CSV"""
        metrics_df = pd.DataFrame({
            'episode': range(1, episode + 1),
            'reward': self.episode_rewards,
            'loss': self.episode_losses,
            'energy_saving': self.episode_energy_savings,
            'latency': self.episode_latencies,
            'sla_violations': self.episode_sla_violations,
            'active_links': self.episode_active_links,
            'cluster_count': self.episode_cluster_counts,
            'computation_time': self.episode_computation_times
        })
        metrics_df.to_csv(f'{self.save_dir}/training_metrics.csv', index=False)
        
    def get_latest_metrics(self):
        """Get latest episode metrics"""
        if not self.episode_rewards:
            return None
        return {
            'reward': self.episode_rewards[-1],
            'loss': self.episode_losses[-1] if self.episode_losses else 0.0,
            'energy_saving': self.episode_energy_savings[-1],
            'latency': self.episode_latencies[-1],
            'sla_violations': self.episode_sla_violations[-1],
            'active_links': self.episode_active_links[-1],
            'cluster_count': self.episode_cluster_counts[-1] if self.episode_cluster_counts else 1
        }

def run_training(cfg_path="config.json", traffic_mode=None):
    """Run enhanced training with traffic load mode selection"""
    print("Starting Training with Traffic Load Management ...")
    print("=" * 60)
    
    # Load configuration
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)['config']
    
    # Override traffic mode if specified
    if traffic_mode and traffic_mode in cfg.get("traffic_modes", {}):
        cfg["traffic_load_mode"] = traffic_mode
        print(f"🔧 Traffic mode overridden to: {traffic_mode}")
    
    current_mode = cfg.get("traffic_load_mode", "low")
    traffic_desc = cfg.get("traffic_modes", {}).get(current_mode, {}).get("description", "Unknown")
    
    print(f"📋 Configuration loaded from: {cfg_path}")
    print(f"🚦 Traffic Load Mode: {current_mode} ({traffic_desc})")
    print(f"🎯 Episodes: {cfg['episodes']}, Steps: {cfg['max_steps_per_episode']}")
    print(f"🌐 Network: {cfg['num_nodes']} nodes, {cfg['num_edges']} edges")
    print(f"🏠 Hosts: {cfg['num_hosts']} ({cfg.get('host_ratio', 0.4)*100:.1f}%)")
    print(f"🌍 Regions: {cfg['num_regions']}")
    
    # Initialize environment and agent
    env = SDNEnv(cfg)
    obs = env.reset()
    agent = HierarchicalDQN(obs_dim=obs.shape[0], action_n=env.action_n, cfg=cfg, device=cfg.get("device","cpu"))
    
    # Initialize visualizer
    visualizer = TrainingVisualizer()
    
    # Training loop
    ep_rewards = []
    t_global = 0
    best_reward = float('-inf')
    
    print(f"\n🎬 Starting training for {cfg['episodes']} episodes...")
    print("-" * 60)
    
    for ep in range(cfg["episodes"]):
        # Check for shutdown request
        if shutdown_requested:
            print("\n🛑 Shutdown requested. Saving current progress...")
            break
            
        obs = env.reset()
        total_r = 0.0
        step_losses = []
        step_energy_savings = []
        step_latencies = []
        step_utilizations = []
        step_computation_times = []
        
        for t in range(cfg["max_steps_per_episode"]):
            # Check for shutdown request in inner loop too
            if shutdown_requested:
                break
            # Agent action
            a = agent.act(obs)
            
            # Measure computation time for the action (including clustering)
            action_start_time = time.time()
            obs2, r, done, info = env.step(a)
            action_end_time = time.time()
            action_computation_time = (action_end_time - action_start_time) * 1000  # Convert to milliseconds
            step_computation_times.append(action_computation_time)
            
            # Store experience
            agent.push(obs, a, r, obs2, float(done))
            obs = obs2
            total_r += r
            
            # Training
            if len(agent.rb) >= cfg["batch_size"] and t_global % cfg["train_every"] == 0:
                loss = agent.train_step()
                if loss is not None:
                    step_losses.append(loss)
            
            # Update target network every 100 steps
            if t_global % 100 == 0:
                agent.update_target()
            
            # Log step metrics
            energy_saving = info.get('energy_saving', 0.0)
            latency = info.get('avg_latency', info.get('latency_ms', 0.0))
            
            # Get utilization stats
            util_stats = env.get_current_utilization_stats()
            current_util = util_stats["average_utilization"]
            step_utilizations.append(current_util)
            
            visualizer.log_step(r, step_losses[-1] if step_losses else None, energy_saving, latency)
            step_energy_savings.append(energy_saving)
            step_latencies.append(latency)
            
            t_global += 1
            
            if done:
                break
        
        # Episode metrics
        avg_loss = np.mean(step_losses) if step_losses else 0.0
        avg_energy_saving = np.mean(step_energy_savings) if step_energy_savings else 0.0
        avg_latency = np.mean(step_latencies) if step_latencies else 0.0
        avg_utilization = np.mean(step_utilizations) if step_utilizations else 0.0
        sla_violations = info.get('sla_violations', info.get('sla_viol', 0.0))
        active_links = info.get('active_links', 0)
        
        # Clustering statistics
        clustering_stats = env.get_clustering_statistics()
        
        # Calculate average computation time per step
        avg_computation_time = np.mean(step_computation_times) if step_computation_times else 0.0
        
        # Log episode metrics
        visualizer.log_episode(ep + 1, total_r, avg_loss, avg_energy_saving, avg_latency, sla_violations, active_links, clustering_stats['current_cluster_count'], avg_computation_time)
        ep_rewards.append(total_r)
        
        # Save best model
        if total_r > best_reward:
            best_reward = total_r
            model_name = f"best_model_{current_mode}_episode_{ep}.pth"
            torch.save(agent.q.state_dict(), model_name)
            print(f"💾 New best model saved! Reward: {total_r:.2f}")
        
        # Save model every 10 episodes
        if (ep + 1) % 10 == 0:
            checkpoint_name = f"checkpoint_{current_mode}_episode_{ep + 1}.pth"
            torch.save(agent.q.state_dict(), checkpoint_name)
            print(f"💾 Checkpoint saved at episode {ep + 1}")
        
        # Enhanced progress logging with utilization and clustering
        target_min, target_max = env.target_util_range
        # Convert target range from fraction to percentage for comparison
        target_min_pct = target_min * 100
        target_max_pct = target_max * 100
        
        # Get actual utilization (uncapped) and display version
        actual_util = util_stats.get("average_utilization", avg_utilization)
        overloaded_edges = util_stats.get("overloaded_edges", 0)
        
        # Check if network is overloaded
        is_overloaded = util_stats.get("overloaded", False)
        if is_overloaded:
            util_status = f"🔥({overloaded_edges})"  # Show number of overloaded edges
        elif target_min_pct <= actual_util <= target_max_pct:
            util_status = "✅"  # Within target
        else:
            util_status = "⚠️"  # Outside target
        
        # Clustering info
        clustering_mode = "NO-CLUST" if env.no_clustering else f"CLUST-{env._actual_num_clusters}"
        clustering_method = clustering_stats['clustering_method_used']
        
        # Calculate total links for display
        total_links = env.G_full.number_of_edges()
        
        print(f"Episode {ep + 1:3d}/{cfg['episodes']} | "
              f"Reward: {total_r:8.2f} | "
              f"Loss: {avg_loss:6.4f} | "
              f"EnergySave: {avg_energy_saving*100:5.1f}% | "
              f"Latency: {avg_latency:6.2f}ms | "
              f"SLA: {sla_violations:5.1f}% | "
              f"Links: {active_links}/{total_links} | "
              f"Flows: {len(env._flows):3d} | "
              f"Util: {actual_util:6.1f}% {util_status} | "
              f"Mode: {clustering_mode} ({clustering_method})")
        
        # Plot every 10 episodes
        if (ep + 1) % 10 == 0:
            visualizer.plot_training_curves(ep + 1)
            visualizer.save_metrics(ep + 1)
    
    # Final model save
    final_model_name = f"final_model_{current_mode}.pth"
    torch.save(agent.q.state_dict(), final_model_name)
    print(f"\n Final model saved: {final_model_name}")
    
    # Final analysis
    final_util_stats = env.get_current_utilization_stats()
    final_clustering_stats = env.get_clustering_statistics()
    visualizer.save_metrics(len(ep_rewards))  # Use actual episode count
    
    if shutdown_requested:
        print(f"\n  Training interrupted at episode {ep + 1}/{cfg['episodes']}")
    print(f"📈 Target utilization: {final_util_stats['target_range'][0]*100:.0f}-{final_util_stats['target_range'][1]*100:.0f}%")
    
    # Clustering analysis
    print(f"\n🔗 Clustering Analysis:")
    print(f"📊 Clustering method: {final_clustering_stats['clustering_method_used']}")
    print(f"📊 Final cluster count: {final_clustering_stats['current_cluster_count']}")
    print(f"📊 Average cluster count: {final_clustering_stats['avg_cluster_count']:.1f}")
    print(f"📊 Cluster count std: {final_clustering_stats['cluster_count_std']:.2f}")
    print(f"📊 Reclustering events: {final_clustering_stats['reclustering_events']}")
    print(f"📊 Total clustering time: {final_clustering_stats['total_reclustering_time']:.3f}s")
    print(f"📊 Nodes per cluster: {final_clustering_stats['nodes_per_cluster']:.1f}")
    
    return {
        'best_reward': best_reward,
        'avg_reward': np.mean(ep_rewards),
        'final_energy_saving': avg_energy_saving,
        'final_latency': avg_latency,
        'final_sla_violations': sla_violations,
        'final_utilization': final_util_stats['average_utilization'],
        'traffic_mode': current_mode,
        'final_path': final_model_name,
        'clustering_stats': final_clustering_stats,
        'no_clustering': env.no_clustering
    }

if __name__ == "__main__":
    import sys
    
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    traffic_mode = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Validate traffic mode
    if traffic_mode:
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)['config']
        available_modes = list(cfg.get("traffic_modes", {}).keys())
        if traffic_mode not in available_modes:
            print(f"❌ Invalid traffic mode: {traffic_mode}")
            print(f"Available modes: {available_modes}")
            sys.exit(1)
    
    results = run_training(cfg_path, traffic_mode)
    
    if results:
        print("\n✅ Training completed successfully!")
        print(f"📊 Traffic Mode: {results['traffic_mode']}")
        print(f"📊 Final Utilization: {results['final_utilization']:.1f}%")
        print(f"📊 Check results in: training_results/")
        print(f"📈 Model file: {results['final_path']}")
    else:
        print("\n❌ Training failed!")
        sys.exit(1) 