"""
Simple script to compare training with and without clustering
"""
import json
import subprocess
import sys

def run_comparison(config_path="config.json", traffic_mode="low", episodes=50):
    """Run comparison between clustering and no-clustering modes"""
    
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

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    traffic_mode = sys.argv[2] if len(sys.argv) > 2 else "low"
    episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    
    run_comparison(config_path, traffic_mode, episodes)
