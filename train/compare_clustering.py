#!/usr/bin/env python3
"""
Clustering Comparison Script

This script compares different clustering approaches:
1. No clustering (single big cluster)
2. DP-means adaptive clustering
3. Fixed k=3 clustering

Usage:
    python compare_clustering.py [config.json] [traffic_mode]
"""

import json
import os
import sys
import time
import pandas as pd
import numpy as np
from train import run_training

def create_config_variants(base_config_path, output_dir="comparison_configs"):
    """Create different config variants for comparison"""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(base_config_path, 'r') as f:
        base_config = json.load(f)
    
    variants = [
        {
            "name": "no_clustering",
            "description": "No Clustering (Single Big Cluster)",
            "config": {
                **base_config["config"],
                "no_clustering": True,
                "adaptive_clustering": False
            }
        },
        {
            "name": "dp_means_adaptive",
            "description": "DP-means Adaptive Clustering",
            "config": {
                **base_config["config"],
                "no_clustering": False,
                "adaptive_clustering": True,
                "clustering_method": "dp_means_adaptive"
            }
        },
        {
            "name": "fixed_k3",
            "description": "Fixed k=3 Clustering",
            "config": {
                **base_config["config"],
                "no_clustering": False,
                "adaptive_clustering": False,
                "num_clusters": 3
            }
        },
        {
            "name": "silhouette",
            "description": "Silhouette Score Clustering",
            "config": {
                **base_config["config"],
                "no_clustering": False,
                "adaptive_clustering": True,
                "clustering_method": "silhouette"
            }
        }
    ]
    
    # Save config variants
    config_paths = []
    for variant in variants:
        config_path = f"{output_dir}/{variant['name']}_config.json"
        with open(config_path, 'w') as f:
            json.dump({"config": variant["config"]}, f, indent=2)
        config_paths.append((config_path, variant))
    
    return config_paths

def run_comparison(base_config_path="config.json", traffic_mode=None, episodes=50):
    """Run comparison between different clustering methods"""
    
    print("🔬 CLUSTERING COMPARISON EXPERIMENT")
    print("=" * 80)
    print(f"📋 Base config: {base_config_path}")
    print(f"🚦 Traffic mode: {traffic_mode or 'default'}")
    print(f"🎯 Episodes per variant: {episodes}")
    print()
    
    # Create config variants
    config_variants = create_config_variants(base_config_path)
    
    results = []
    start_time = time.time()
    
    for i, (config_path, variant) in enumerate(config_variants, 1):
        print(f"\n🔄 Running variant {i}/{len(config_variants)}: {variant['description']}")
        print("-" * 60)
        
        # Temporarily modify episodes for faster comparison
        with open(config_path, 'r') as f:
            config = json.load(f)
        config['config']['episodes'] = episodes
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Run training
        variant_start = time.time()
        try:
            result = run_training(config_path, traffic_mode)
            variant_time = time.time() - variant_start
            
            # Add variant info to result
            result.update({
                'variant_name': variant['name'],
                'variant_description': variant['description'],
                'training_time': variant_time,
                'config_path': config_path
            })
            
            results.append(result)
            
            print(f"✅ {variant['description']} completed in {variant_time:.1f}s")
            
        except Exception as e:
            print(f"❌ {variant['description']} failed: {e}")
            continue
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Total comparison time: {total_time:.1f}s")
    
    return results

def analyze_results(results):
    """Analyze and compare results"""
    if not results:
        print("❌ No results to analyze!")
        return
    
    print("\n📊 RESULTS ANALYSIS")
    print("=" * 80)
    
    # Create comparison table
    comparison_data = []
    for result in results:
        clustering_stats = result.get('clustering_stats', {})
        comparison_data.append({
            'Method': result['variant_description'],
            'No Clustering': result.get('no_clustering', False),
            'Clustering Method': clustering_stats.get('clustering_method_used', 'unknown'),
            'Avg Cluster Count': f"{clustering_stats.get('avg_cluster_count', 1):.1f}",
            'Final Cluster Count': clustering_stats.get('current_cluster_count', 1),
            'Reclustering Events': clustering_stats.get('reclustering_events', 0),
            'Clustering Time (s)': f"{clustering_stats.get('total_reclustering_time', 0):.3f}",
            'Best Reward': f"{result['best_reward']:.2f}",
            'Avg Reward': f"{result['avg_reward']:.2f}",
            'Energy Saving (%)': f"{result['final_energy_saving']:.1f}",
            'Latency (ms)': f"{result['final_latency']:.2f}",
            'SLA Violations (%)': f"{result['final_sla_violations']:.1f}",
            'Training Time (s)': f"{result['training_time']:.1f}"
        })
    
    # Display comparison table
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    # Find best performers
    print(f"\n🏆 BEST PERFORMERS:")
    
    # Best reward
    best_reward_idx = df['Best Reward'].str.replace('.', '').astype(float).idxmax()
    print(f"🥇 Best Reward: {df.iloc[best_reward_idx]['Method']} ({df.iloc[best_reward_idx]['Best Reward']})")
    
    # Best energy saving
    best_energy_idx = df['Energy Saving (%)'].str.replace('.', '').astype(float).idxmax()
    print(f"⚡ Best Energy Saving: {df.iloc[best_energy_idx]['Method']} ({df.iloc[best_energy_idx]['Energy Saving (%)']}%)")
    
    # Best latency
    best_latency_idx = df['Latency (ms)'].str.replace('.', '').astype(float).idxmin()
    print(f"Best Latency: {df.iloc[best_latency_idx]['Method']} ( {df.iloc[best_latency_idx]['Latency (ms)']} ms)")
    
    # Lowest SLA violations
    best_sla_idx = df['SLA Violations (%)'].str.replace('.', '').astype(float).idxmin()
    print(f"📋 Lowest SLA Violations: {df.iloc[best_sla_idx]['Method']} ({df.iloc[best_sla_idx]['SLA Violations (%)']}%)")
    
    # Fastest training
    fastest_idx = df['Training Time (s)'].str.replace('.', '').astype(float).idxmin()
    print(f"⚡ Fastest Training: {df.iloc[fastest_idx]['Method']} ({df.iloc[fastest_idx]['Training Time (s)']}s)")
    
    return df

def save_comparison_results(results, df, output_dir="comparison_results"):
    """Save comparison results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_file = f"{output_dir}/detailed_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save comparison table
    table_file = f"{output_dir}/comparison_table.csv"
    df.to_csv(table_file, index=False)
    
    print(f"\n💾 Results saved to {output_dir}/")
    print(f"📄 Detailed results: {results_file}")
    print(f"📊 Comparison table: {table_file}")

def print_recommendations(df):
    """Print recommendations based on results"""
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 40)
    
    # Analyze clustering vs no-clustering
    no_clust_results = df[df['No Clustering'] == True]
    clust_results = df[df['No Clustering'] == False]
    
    if len(no_clust_results) > 0 and len(clust_results) > 0:
        no_clust_reward = float(no_clust_results.iloc[0]['Best Reward'])
        best_clust_reward = float(clust_results['Best Reward'].str.replace('.', '').astype(float).max())
        
        if best_clust_reward > no_clust_reward * 1.05:  # 5% improvement threshold
            print("✅ Clustering shows significant improvement over no-clustering")
            best_clust_method = clust_results.loc[clust_results['Best Reward'].str.replace('.', '').astype(float).idxmax(), 'Method']
            print(f"   Best clustering method: {best_clust_method}")
        elif best_clust_reward < no_clust_reward * 0.95:  # 5% degradation threshold
            print("⚠️  Clustering shows degradation compared to no-clustering")
            print("   Consider using no-clustering for this network configuration")
        else:
            print("📊 Clustering and no-clustering perform similarly")
            print("   Choose based on computational efficiency and complexity")
    
    # DP-means analysis
    dp_means_results = df[df['Clustering Method'].str.contains('dp_means', na=False)]
    if len(dp_means_results) > 0:
        print(f"\n🔗 DP-means Analysis:")
        dp_means_row = dp_means_results.iloc[0]
        avg_clusters = float(dp_means_row['Avg Cluster Count'])
        reclustering_events = int(dp_means_row['Reclustering Events'])
        
        if avg_clusters > 5:
            print(f"   📈 High cluster count ({avg_clusters:.1f}) - good adaptation to network complexity")
        elif avg_clusters < 3:
            print(f"   📉 Low cluster count ({avg_clusters:.1f}) - may need parameter tuning")
        
        if reclustering_events > 10:
            print(f"   🔄 High reclustering activity ({reclustering_events} events) - good adaptation")
        else:
            print(f"   🔒 Stable clustering ({reclustering_events} events) - consistent structure")

def main():
    """Main comparison function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare clustering methods for SDN networks')
    parser.add_argument('config', nargs='?', default='config.json', help='Base config file')
    parser.add_argument('traffic_mode', nargs='?', help='Traffic mode (low/high)')
    parser.add_argument('--episodes', type=int, default=50, help='Episodes per variant (default: 50)')
    
    args = parser.parse_args()
    
    # Validate config file
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    # Validate traffic mode if provided
    if args.traffic_mode:
        with open(args.config, 'r') as f:
            config = json.load(f)
        available_modes = list(config.get("config", {}).get("traffic_modes", {}).keys())
        if args.traffic_mode not in available_modes:
            print(f"❌ Invalid traffic mode: {args.traffic_mode}")
            print(f"Available modes: {available_modes}")
            sys.exit(1)
    
    # Run comparison
    results = run_comparison(args.config, args.traffic_mode, args.episodes)
    
    if results:
        df = analyze_results(results)
        save_comparison_results(results, df)
        print_recommendations(df)
        
        print(f"\n✅ Comparison completed successfully!")
        print(f"📊 {len(results)} variants tested")
    else:
        print(f"\n❌ Comparison failed - no results generated")
        sys.exit(1)

if __name__ == "__main__":
    main()
