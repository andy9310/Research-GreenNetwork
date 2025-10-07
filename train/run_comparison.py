"""
Simple script to compare training with and without clustering
"""
import json
import subprocess
import sys

def run_comparison(config_path="config.json", traffic_mode="low", episodes=50):
    """Run comparison between clustering and no-clustering modes"""
    import os
    import shutil
    
    print("=" * 80)
    print("CLUSTERING COMPARISON EXPERIMENT")
    print("=" * 80)
    
    # Load base config
    with open(config_path, 'r') as f:
        base_config = json.load(f)
    
    results = {}
    
    # Experiment 1: No Clustering (Baseline)
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: NO CLUSTERING (Single Big Cluster)")
    print("=" * 80)
    
    config_no_clust = base_config.copy()
    config_no_clust['config']['no_clustering'] = True
    config_no_clust['config']['adaptive_clustering'] = False
    config_no_clust['config']['episodes'] = episodes
    
    with open('temp_no_clustering.json', 'w') as f:
        json.dump(config_no_clust, f, indent=2)
    
    print("\n🚀 Running training WITHOUT clustering...")
    subprocess.run([sys.executable, 'train.py', 'temp_no_clustering.json', traffic_mode])
    
    # Save results from first experiment
    if os.path.exists('training_results/training_metrics.csv'):
        os.makedirs('comparison_results_temp', exist_ok=True)
        shutil.copy('training_results/training_metrics.csv', 'comparison_results_temp/no_clustering_metrics.csv')
        print("✅ Saved no-clustering results")
    
    # Experiment 2: With Clustering (DP-means Adaptive)
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: WITH CLUSTERING (DP-means Adaptive)")
    print("=" * 80)
    
    config_with_clust = base_config.copy()
    config_with_clust['config']['no_clustering'] = False
    config_with_clust['config']['adaptive_clustering'] = True
    config_with_clust['config']['clustering_method'] = 'dp_means_adaptive'
    config_with_clust['config']['episodes'] = episodes
    
    with open('temp_with_clustering.json', 'w') as f:
        json.dump(config_with_clust, f, indent=2)
    
    print("\n🚀 Running training WITH clustering...")
    subprocess.run([sys.executable, 'train.py', 'temp_with_clustering.json', traffic_mode])
    
    # Save results from second experiment
    if os.path.exists('training_results/training_metrics.csv'):
        shutil.copy('training_results/training_metrics.csv', 'comparison_results_temp/with_clustering_metrics.csv')
        print("✅ Saved with-clustering results")
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE!")
    print("=" * 80)
    print("\n📊 Check the terminal output above to compare:")
    print("   - Final rewards")
    print("   - Energy savings")
    print("   - SLA violations")
    print("   - Latency")
    print("\n📁 Models saved:")
    print("   - final_model_low.pth (from last experiment)")
    print("\n💡 Tip: Run with more episodes for better comparison:")
    print("   python run_comparison.py config.json low 100")
    
    # Generate comparison plot
    print("\n📊 Generating comparison plot...")
    try:
        generate_comparison_plot(traffic_mode)
        print("✅ Plot saved: clustering_comparison.png")
    except Exception as e:
        print(f"⚠️ Could not generate plot: {e}")


def generate_comparison_plot(traffic_mode):
    """Generate comparison plot from actual training results"""
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    
    # Check for saved comparison results
    no_clust_file = "comparison_results_temp/no_clustering_metrics.csv"
    with_clust_file = "comparison_results_temp/with_clustering_metrics.csv"
    
    if not os.path.exists(no_clust_file) or not os.path.exists(with_clust_file):
        print("⚠️ No comparison results found. Run the comparison first.")
        return
    
    # Load actual training metrics
    no_clust_df = pd.read_csv(no_clust_file)
    with_clust_df = pd.read_csv(with_clust_file)
    
    # Extract final episode metrics (last row)
    no_clust_final = no_clust_df.iloc[-1]
    with_clust_final = with_clust_df.iloc[-1]
    
    # Calculate averages
    no_clust_avg_reward = no_clust_df['reward'].mean()
    with_clust_avg_reward = with_clust_df['reward'].mean()
    
    # Create comparison plot with actual data - 2x2 grid for 4 metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Clustering vs No-Clustering Comparison ({traffic_mode} traffic)', 
                 fontsize=16, fontweight='bold')
    
    methods = ['No Clustering', 'With Clustering']
    colors = ['#ff7f0e', '#2ca02c']
    
    # Extract actual metrics from training data - only the 4 requested metrics
    metrics = {
        'Energy Saving (%)': [
            no_clust_final['energy_saving'] * 100,
            with_clust_final['energy_saving'] * 100
        ],
        'Latency (ms)': [
            no_clust_final['latency'],
            with_clust_final['latency']
        ],
        'SLA Violations (%)': [
            no_clust_final['sla_violations'],
            with_clust_final['sla_violations']
        ],
        'Computation Time (ms)': [
            no_clust_final['computation_time'],
            with_clust_final['computation_time']
        ]
    }
    
    for idx, (metric, values) in enumerate(metrics.items()):
        ax = axes[idx // 2, idx % 2]
        bars = ax.bar(methods, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(metric, fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('clustering_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: clustering_comparison.png")

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    traffic_mode = sys.argv[2] if len(sys.argv) > 2 else "low"
    episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    
    run_comparison(config_path, traffic_mode, episodes)
