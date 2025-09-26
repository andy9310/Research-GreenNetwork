import json, os, time, numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any
import pandas as pd

from env import SDNEnv
from agent import HierarchicalDQN

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
        
        # Step-by-step metrics
        self.step_rewards = []
        self.step_losses = []
        self.step_energy_savings = []
        self.step_latencies = []
        
    def log_episode(self, episode, reward, loss, energy_saving, latency, sla_viol, active_links):
        """Log episode-level metrics"""
        self.episode_rewards.append(reward)
        self.episode_losses.append(loss if loss is not None else 0.0)
        self.episode_energy_savings.append(energy_saving)
        self.episode_latencies.append(latency)
        self.episode_sla_violations.append(sla_viol)
        self.episode_active_links.append(active_links)
        
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
        axes[0, 0].plot(pd.Series(self.episode_rewards).rolling(10).mean(), 'r-', linewidth=2)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Episode losses
        axes[0, 1].plot(self.episode_losses, 'g-', alpha=0.7)
        axes[0, 1].plot(pd.Series(self.episode_losses).rolling(10).mean(), 'r-', linewidth=2)
        axes[0, 1].set_title('Episode Losses')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # Energy savings
        axes[0, 2].plot(self.episode_energy_savings, 'purple', alpha=0.7)
        axes[0, 2].plot(pd.Series(self.episode_energy_savings).rolling(10).mean(), 'r-', linewidth=2)
        axes[0, 2].set_title('Energy Savings')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Energy Saving (%)')
        axes[0, 2].grid(True)
        
        # Latency
        axes[1, 0].plot(self.episode_latencies, 'orange', alpha=0.7)
        axes[1, 0].plot(pd.Series(self.episode_latencies).rolling(10).mean(), 'r-', linewidth=2)
        axes[1, 0].set_title('Average Latency')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Latency (ms)')
        axes[1, 0].grid(True)
        
        # SLA violations
        axes[1, 1].plot(self.episode_sla_violations, 'red', alpha=0.7)
        axes[1, 1].plot(pd.Series(self.episode_sla_violations).rolling(10).mean(), 'r-', linewidth=2)
        axes[1, 1].set_title('SLA Violations')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Violations')
        axes[1, 1].grid(True)
        
        # Active links
        axes[1, 2].plot(self.episode_active_links, 'brown', alpha=0.7)
        axes[1, 2].plot(pd.Series(self.episode_active_links).rolling(10).mean(), 'r-', linewidth=2)
        axes[1, 2].set_title('Active Links')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Number of Links')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/training_curves_episode_{episode}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_step_metrics(self, episode):
        """Plot step-by-step metrics for the last episode"""
        if len(self.step_rewards) == 0:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Step-by-Step Metrics - Episode {episode}', fontsize=16)
        
        # Step rewards
        axes[0, 0].plot(self.step_rewards, 'b-')
        axes[0, 0].set_title('Step Rewards')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Step losses
        axes[0, 1].plot(self.step_losses, 'g-')
        axes[0, 1].set_title('Step Losses')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # Step energy savings
        axes[1, 0].plot(self.step_energy_savings, 'purple')
        axes[1, 0].set_title('Step Energy Savings')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Energy Saving (%)')
        axes[1, 0].grid(True)
        
        # Step latency
        axes[1, 1].plot(self.step_latencies, 'orange')
        axes[1, 1].set_title('Step Latency')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Latency (ms)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/step_metrics_episode_{episode}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_metrics(self, episode):
        """Save metrics to CSV files"""
        # Episode metrics
        episode_df = pd.DataFrame({
            'episode': range(len(self.episode_rewards)),
            'reward': self.episode_rewards,
            'loss': self.episode_losses,
            'energy_saving': self.episode_energy_savings,
            'latency': self.episode_latencies,
            'sla_violations': self.episode_sla_violations,
            'active_links': self.episode_active_links
        })
        episode_df.to_csv(f'{self.save_dir}/episode_metrics.csv', index=False)
        
        # Step metrics
        if len(self.step_rewards) > 0:
            step_df = pd.DataFrame({
                'step': range(len(self.step_rewards)),
                'reward': self.step_rewards,
                'loss': self.step_losses,
                'energy_saving': self.step_energy_savings,
                'latency': self.step_latencies
            })
            step_df.to_csv(f'{self.save_dir}/step_metrics.csv', index=False)
            
    def print_summary(self, episode):
        """Print training summary"""
        if len(self.episode_rewards) == 0:
            return
            
        print(f"\n{'='*60}")
        print(f"TRAINING SUMMARY - Episode {episode}")
        print(f"{'='*60}")
        print(f"Average Reward (last 10): {np.mean(self.episode_rewards[-10:]):.4f}")
        print(f"Average Energy Saving (last 10): {np.mean(self.episode_energy_savings[-10:]):.2%}")
        print(f"Average Latency (last 10): {np.mean(self.episode_latencies[-10:]):.2f} ms")
        print(f"Average SLA Violations (last 10): {np.mean(self.episode_sla_violations[-10:]):.2f}")
        print(f"Average Active Links (last 10): {np.mean(self.episode_active_links[-10:]):.0f}")
        print(f"Best Reward: {max(self.episode_rewards):.4f}")
        print(f"Best Energy Saving: {max(self.episode_energy_savings):.2%}")
        print(f"{'='*60}\n")

def run_training(cfg_path: str = "config.json"):
    """training with comprehensive visualization"""
    print("Starting Training with Visualization")
    print("=" * 60)
    
    # Load configuration
    try:
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)['config']
        print(f"✅ Configuration loaded from {cfg_path}")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return None
    
    # Initialize environment and agent
    env = SDNEnv(cfg)
    obs = env.reset()
    agent = HierarchicalDQN(
        obs_dim=obs.shape[0],
        action_n=env.action_n,
        cfg=cfg,
        device=cfg.get("device", "cpu")
    )
    
    # Initialize visualizer
    visualizer = TrainingVisualizer()
    
    # Training metrics
    episode_rewards = []
    best_reward = float('-inf')
    start_time = time.time()
    
    print(f"🌐 Network: {cfg['num_nodes']} nodes, {cfg['num_edges']} edges, {cfg['num_hosts']} hosts")
    print(f"🎯 Algorithm: {cfg.get('deactivation_algorithm', 'greedy')}")
    print(f"🔄 Episodes: {cfg['episodes']}, Steps per episode: {cfg['max_steps_per_episode']}")
    print()
    
    try:
        for episode in range(cfg["episodes"]):
            obs = env.reset()
            episode_reward = 0.0
            episode_loss = 0.0
            episode_energy_saving = 0.0
            episode_latency = 0.0
            episode_sla_viol = 0
            episode_active_links = 0
            step_count = 0
            
            # Clear step metrics for new episode
            visualizer.step_rewards = []
            visualizer.step_losses = []
            visualizer.step_energy_savings = []
            visualizer.step_latencies = []
            
            # Episode loop
            for step in range(cfg["max_steps_per_episode"]):
                # Agent action
                action = agent.act(obs)
                obs_next, reward, done, info = env.step(action)
                
                # Store experience
                agent.push(obs, action, reward, obs_next, float(done))
                
                # Training
                loss = None
                if len(agent.rb) >= cfg["batch_size"]:
                    loss = agent.train_step()
                
                # Update metrics
                episode_reward += reward
                episode_loss += loss if loss is not None else 0.0
                episode_energy_saving += info.get('energy_saving', 0.0)
                episode_latency += info.get('latency_ms', 0.0)
                episode_sla_viol += info.get('sla_viol', 0)
                episode_active_links += info.get('active_links', 0)
                step_count += 1
                
                # Log step metrics
                visualizer.log_step(
                    reward, loss, 
                    info.get('energy_saving', 0.0), 
                    info.get('latency_ms', 0.0)
                )
                
                obs = obs_next
                
                # Target network update
                if step % cfg["target_update"] == 0:
                    agent.update_target()
                
                if done:
                    break
            
            # Calculate episode averages
            avg_energy_saving = episode_energy_saving / max(1, step_count)
            avg_latency = episode_latency / max(1, step_count)
            avg_active_links = episode_active_links / max(1, step_count)
            avg_loss = episode_loss / max(1, step_count)
            
            # Log episode metrics
            visualizer.log_episode(
                episode, episode_reward, avg_loss,
                avg_energy_saving, avg_latency, 
                episode_sla_viol, avg_active_links
            )
            
            episode_rewards.append(episode_reward)
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                model_path = f"best_model_episode_{episode}.pth"
                agent.save(model_path)
                print(f"🏆 New best model saved: {model_path}")
            
            # Logging and visualization
            if (episode + 1) % cfg.get("log_every", 5) == 0 or episode == 0:
                elapsed = time.time() - start_time
                print(f"Episode {episode+1:3d}/{cfg['episodes']:3d} | "
                      f"Reward: {episode_reward:7.3f} | "
                      f"Energy↓: {avg_energy_saving:5.1%} | "
                      f"Latency: {avg_latency:5.1f}ms | "
                      f"SLA❌: {episode_sla_viol:3d} | "
                      f"Links: {avg_active_links:4.0f} | "
                      f"ε: {agent.eps:4.2f} | "
                      f"Time: {elapsed:5.1f}s")
                
                # Generate plots
                visualizer.plot_training_curves(episode + 1)
                visualizer.plot_step_metrics(episode + 1)
                visualizer.save_metrics(episode + 1)
                visualizer.print_summary(episode + 1)
    
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Final save
    final_path = "final_model.pth"
    agent.save(final_path)
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("🎉 Training Completed Successfully!")
    print(f"🌐 Network: {cfg['num_nodes']} nodes, {cfg['num_edges']} edges, {cfg['num_hosts']} hosts")
    print(f"🎯 Algorithm: {cfg.get('deactivation_algorithm', 'greedy')}")
    print(f"🔄 Episodes: {cfg['episodes']}, Steps per episode: {cfg['max_steps_per_episode']}")
    print(f"⏰ Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"🏆 Best reward: {best_reward:.3f}")
    print(f"📈 Final 10-episode avg: {np.mean(episode_rewards[-10:]):.3f}")
    print(f"💾 Model saved to: {final_path}")
    print(f"📊 Results saved to: training_results/")
    print("="*60)
    
    return {
        'episode_rewards': episode_rewards,
        'best_reward': best_reward,
        'final_path': final_path,
        'total_time': total_time
    }

if __name__ == "__main__":
    import sys
    
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    results = run_enhanced_training(cfg_path)
    
    if results:
        print("\n✅ Training completed successfully!")
        print(f"📊 Check results in: training_results/")
        print(f"📈 Model file: {results['final_path']}")
    else:
        print("\n❌ Training failed!")
        sys.exit(1) 