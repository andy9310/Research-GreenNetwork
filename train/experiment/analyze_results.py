"""
Results Analysis Script
Generates plots and tables from experimental results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


def load_results(results_dir="results"):
    """Load results from CSV"""
    csv_file = f"{results_dir}/comparison_table.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run experiments first.")
        return None
    
    df = pd.read_csv(csv_file)
    return df


def plot_energy_saving(df, output_dir="results/plots"):
    """Plot energy saving comparison"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = df['Method'].unique()
    topologies = sorted(df['Topology'].unique())
    x = np.arange(len(topologies))
    width = 0.25
    
    for i, method in enumerate(methods):
        method_data = df[df['Method'] == method]
        values = [method_data[method_data['Topology'] == t]['Energy_Saving_%'].values[0] 
                 if len(method_data[method_data['Topology'] == t]) > 0 else 0 
                 for t in topologies]
        ax.bar(x + i * width, values, width, label=method)
    
    ax.set_xlabel('Network Size (Links)', fontsize=12)
    ax.set_ylabel('Energy Saving (%)', fontsize=12)
    ax.set_title('Energy Saving Comparison Across Network Sizes', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(topologies)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/energy_saving_comparison.png", dpi=300)
    print(f"✅ Saved: {output_dir}/energy_saving_comparison.png")
    plt.close()


def plot_latency_comparison(df, output_dir="results/plots"):
    """Plot latency comparison"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = df['Method'].unique()
    topologies = sorted(df['Topology'].unique())
    x = np.arange(len(topologies))
    width = 0.25
    
    for i, method in enumerate(methods):
        method_data = df[df['Method'] == method]
        values = [method_data[method_data['Topology'] == t]['Latency_ms'].values[0] 
                 if len(method_data[method_data['Topology'] == t]) > 0 else 0 
                 for t in topologies]
        ax.bar(x + i * width, values, width, label=method)
    
    ax.set_xlabel('Network Size (Links)', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Latency Comparison Across Network Sizes', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(topologies)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/latency_comparison.png", dpi=300)
    print(f"✅ Saved: {output_dir}/latency_comparison.png")
    plt.close()


def plot_computation_time(df, output_dir="results/plots"):
    """Plot computation time scalability"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = df['Method'].unique()
    
    for method in methods:
        method_data = df[df['Method'] == method].sort_values('Topology')
        ax.plot(method_data['Topology'], method_data['Comp_Time_s'], 
               marker='o', linewidth=2, markersize=8, label=method)
    
    ax.set_xlabel('Network Size (Links)', fontsize=12)
    ax.set_ylabel('Computation Time (seconds)', fontsize=12)
    ax.set_title('Computation Time Scalability', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/computation_time_scalability.png", dpi=300)
    print(f"✅ Saved: {output_dir}/computation_time_scalability.png")
    plt.close()


def plot_tradeoff(df, output_dir="results/plots"):
    """Plot energy-latency trade-off"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    methods = df['Method'].unique()
    colors = {'dqn_clustering': 'blue', 'energy_aware': 'green', 'rl_basic': 'red'}
    
    for method in methods:
        method_data = df[df['Method'] == method]
        ax.scatter(method_data['Energy_Saving_%'], method_data['Latency_ms'], 
                  s=200, alpha=0.6, c=colors.get(method, 'gray'), label=method)
        
        # Add topology labels
        for _, row in method_data.iterrows():
            ax.annotate(f"{int(row['Topology'])}", 
                       (row['Energy_Saving_%'], row['Latency_ms']),
                       fontsize=8, ha='center')
    
    ax.set_xlabel('Energy Saving (%)', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Energy-Latency Trade-off', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/energy_latency_tradeoff.png", dpi=300)
    print(f"✅ Saved: {output_dir}/energy_latency_tradeoff.png")
    plt.close()


def generate_latex_table(df):
    """Generate LaTeX table for paper"""
    print("\n" + "="*80)
    print("LATEX TABLE (copy to your paper)")
    print("="*80 + "\n")
    
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Comparison of Energy-Aware Routing Methods}")
    print("\\label{tab:comparison}")
    print("\\begin{tabular}{|l|c|c|c|c|c|}")
    print("\\hline")
    print("\\textbf{Method} & \\textbf{Topology} & \\textbf{Energy (\\%)} & \\textbf{Latency (ms)} & \\textbf{SLA (\\%)} & \\textbf{Time (s)} \\\\")
    print("\\hline")
    
    for _, row in df.iterrows():
        print(f"{row['Method']} & {int(row['Topology'])} & {row['Energy_Saving_%']:.1f} & {row['Latency_ms']:.2f} & {row['SLA_Violations_%']:.1f} & {row['Comp_Time_s']:.4f} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    print()


def print_summary(df):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80 + "\n")
    
    for method in df['Method'].unique():
        method_data = df[df['Method'] == method]
        print(f"\n{method.upper()}:")
        print(f"  Avg Energy Saving: {method_data['Energy_Saving_%'].mean():.1f}% ± {method_data['Energy_Saving_%'].std():.1f}%")
        print(f"  Avg Latency: {method_data['Latency_ms'].mean():.2f}ms ± {method_data['Latency_ms'].std():.2f}ms")
        print(f"  Avg SLA Violations: {method_data['SLA_Violations_%'].mean():.1f}% ± {method_data['SLA_Violations_%'].std():.1f}%")
        print(f"  Avg Computation Time: {method_data['Comp_Time_s'].mean():.4f}s ± {method_data['Comp_Time_s'].std():.4f}s")


def main():
    parser = argparse.ArgumentParser(description='Analyze experimental results')
    parser.add_argument('--latex', action='store_true', help='Generate LaTeX table')
    parser.add_argument('--results-dir', type=str, default='results', help='Results directory')
    
    args = parser.parse_args()
    
    # Load results
    df = load_results(args.results_dir)
    if df is None:
        return
    
    print(f"\n✅ Loaded {len(df)} results")
    
    # Generate plots
    print("\n📊 Generating plots...")
    plot_energy_saving(df, f"{args.results_dir}/plots")
    plot_latency_comparison(df, f"{args.results_dir}/plots")
    plot_computation_time(df, f"{args.results_dir}/plots")
    plot_tradeoff(df, f"{args.results_dir}/plots")
    
    # Print summary
    print_summary(df)
    
    # Generate LaTeX table
    if args.latex:
        generate_latex_table(df)
    
    print("\n✅ Analysis complete!")
    print(f"📁 Results saved in: {args.results_dir}/plots/")


if __name__ == "__main__":
    main()
