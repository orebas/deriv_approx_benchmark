#!/usr/bin/env python3
"""
Comprehensive Derivative Approximation Benchmark Report
Generate analysis and visualizations from CSV results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from datetime import datetime

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def main():
    print("📊 DERIVATIVE APPROXIMATION BENCHMARK REPORT")
    print("=" * 60)
    
    # Create report directory
    report_dir = f"report_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    Path(report_dir).mkdir(exist_ok=True)
    Path(f"{report_dir}/figures").mkdir(exist_ok=True)
    
    print(f"📁 Report directory: {report_dir}")
    
    # Load all sweep data files
    data_files = glob.glob("results/sweep_lv_periodic_n*_d101.csv")
    print(f"\n📥 Found {len(data_files)} data files:")
    
    all_data = []
    for file in data_files:
        print(f"  ✓ Loading {file}")
        df = pd.read_csv(file)
        all_data.append(df)
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Combined dataset: {len(combined_data):,} rows × {len(combined_data.columns)} columns")
    
    # Get unique summary statistics (RMSE is constant per method/derivative/noise combo)
    summary_data = combined_data.groupby(['method', 'derivative_order', 'noise_level', 'observable']).agg({
        'rmse': 'first',
        'mae': 'first', 
        'max_error': 'first',
        'data_size': 'first'
    }).reset_index()
    
    print(f"📈 Summary dataset: {len(summary_data):,} unique combinations")
    print(f"Methods: {sorted(summary_data['method'].unique())}")
    print(f"Noise levels: {sorted(summary_data['noise_level'].unique())}")
    print(f"Derivative orders: {sorted(summary_data['derivative_order'].unique())}")
    print(f"Observables: {sorted(summary_data['observable'].unique())}")
    
    # Generate comprehensive analysis and plots
    print("\n🎨 Generating visualizations...")
    
    # 1. RMSE by Method and Derivative Order (Heatmap)
    create_method_derivative_heatmap(summary_data, report_dir)
    
    # 2. RMSE vs Noise Level (Log scale)
    create_noise_performance_plot(summary_data, report_dir)
    
    # 3. Method Performance Rankings
    create_method_rankings(summary_data, report_dir)
    
    # 4. Derivative Order Performance Degradation
    create_derivative_degradation_plot(summary_data, report_dir)
    
    # 5. Comprehensive Summary Table
    create_summary_tables(summary_data, report_dir)
    
    # 6. Generate Markdown Report
    create_markdown_report(summary_data, report_dir)
    
    print(f"\n✅ Report complete! Check {report_dir}/ for results")
    print(f"📊 Key findings saved to {report_dir}/README.md")

def create_method_derivative_heatmap(data, report_dir):
    """Create heatmap of RMSE by method and derivative order"""
    print("  📊 Creating method-derivative heatmap...")
    
    # Average RMSE across noise levels and observables for each method/derivative combo
    heatmap_data = data.groupby(['method', 'derivative_order'])['rmse'].mean().unstack()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='.2e', cmap='YlOrRd', 
                cbar_kws={'label': 'RMSE'})
    plt.title('RMSE by Method and Derivative Order\n(Lower is Better)', fontsize=14, fontweight='bold')
    plt.xlabel('Derivative Order')
    plt.ylabel('Method')
    plt.tight_layout()
    plt.savefig(f'{report_dir}/figures/method_derivative_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_noise_performance_plot(data, report_dir):
    """Create plots showing performance vs noise level"""
    print("  📈 Creating noise performance plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, deriv_order in enumerate(sorted(data['derivative_order'].unique())):
        if i >= 6:
            break
            
        ax = axes[i]
        deriv_data = data[data['derivative_order'] == deriv_order]
        
        for method in sorted(deriv_data['method'].unique()):
            method_data = deriv_data[deriv_data['method'] == method]
            # Average across observables
            noise_rmse = method_data.groupby('noise_level')['rmse'].mean()
            
            ax.loglog(noise_rmse.index, noise_rmse.values, 'o-', label=method, markersize=4)
        
        ax.set_title(f'Derivative Order {deriv_order}')
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('RMSE')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle('RMSE vs Noise Level by Derivative Order\n(Log-Log Scale)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{report_dir}/figures/noise_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_method_rankings(data, report_dir):
    """Create method performance rankings"""
    print("  🏆 Creating method rankings...")
    
    # Calculate average RMSE across all conditions for each method
    method_performance = data.groupby('method').agg({
        'rmse': ['mean', 'median', 'std', 'count']
    }).round(4)
    
    method_performance.columns = ['Mean_RMSE', 'Median_RMSE', 'Std_RMSE', 'Count']
    method_performance = method_performance.sort_values('Mean_RMSE')
    
    # Plot rankings
    plt.figure(figsize=(10, 6))
    y_pos = range(len(method_performance))
    
    plt.barh(y_pos, method_performance['Mean_RMSE'], alpha=0.7)
    plt.yticks(y_pos, method_performance.index)
    plt.xlabel('Average RMSE (across all conditions)')
    plt.title('Method Performance Rankings\n(Lower RMSE is Better)', fontweight='bold')
    plt.xscale('log')
    
    # Add value labels
    for i, v in enumerate(method_performance['Mean_RMSE']):
        plt.text(v, i, f'  {v:.2e}', va='center')
    
    plt.tight_layout()
    plt.savefig(f'{report_dir}/figures/method_rankings.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return method_performance

def create_derivative_degradation_plot(data, report_dir):
    """Show how performance degrades with derivative order"""
    print("  📉 Creating derivative degradation plot...")
    
    # Average RMSE by derivative order across all methods/conditions
    deriv_performance = data.groupby(['derivative_order', 'method'])['rmse'].mean().unstack()
    
    plt.figure(figsize=(12, 6))
    
    for method in deriv_performance.columns:
        plt.semilogy(deriv_performance.index, deriv_performance[method], 'o-', 
                    label=method, markersize=6, linewidth=2)
    
    plt.xlabel('Derivative Order')
    plt.ylabel('RMSE (log scale)')
    plt.title('Performance Degradation with Derivative Order', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(6))
    plt.tight_layout()
    plt.savefig(f'{report_dir}/figures/derivative_degradation.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_tables(data, report_dir):
    """Create summary statistics tables"""
    print("  📋 Creating summary tables...")
    
    # Best method per derivative order
    best_methods = []
    for deriv_order in sorted(data['derivative_order'].unique()):
        deriv_data = data[data['derivative_order'] == deriv_order]
        best_method = deriv_data.groupby('method')['rmse'].mean().idxmin()
        best_rmse = deriv_data.groupby('method')['rmse'].mean().min()
        
        best_methods.append({
            'derivative_order': deriv_order,
            'best_method': best_method,
            'rmse': best_rmse
        })
    
    best_methods_df = pd.DataFrame(best_methods)
    best_methods_df.to_csv(f'{report_dir}/best_methods_per_derivative.csv', index=False)
    
    # Overall summary statistics
    summary_stats = data.groupby(['method', 'derivative_order']).agg({
        'rmse': ['mean', 'std', 'min', 'max'],
        'mae': ['mean', 'std'],
        'max_error': ['mean', 'std']
    }).round(6)
    
    summary_stats.to_csv(f'{report_dir}/comprehensive_summary_statistics.csv')
    
    return best_methods_df, summary_stats

def create_markdown_report(data, report_dir):
    """Generate comprehensive markdown report"""
    print("  📄 Creating markdown report...")
    
    best_methods_df, _ = create_summary_tables(data, report_dir)
    
    report_content = f"""# Derivative Approximation Benchmark Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report analyzes the performance of **{len(data['method'].unique())} approximation methods** across **{len(data['derivative_order'].unique())} derivative orders** and **{len(data['noise_level'].unique())} noise levels** using the Lotka-Volterra periodic system.

### Key Findings

#### 🏆 Best Methods by Derivative Order

| Derivative Order | Best Method | RMSE |
|-----------------|-------------|------|
"""
    
    for _, row in best_methods_df.iterrows():
        report_content += f"| {int(row['derivative_order'])} | {row['best_method']} | {row['rmse']:.2e} |\n"
    
    report_content += f"""

#### 📊 Study Parameters
- **Methods Tested**: {', '.join(sorted(data['method'].unique()))}
- **Noise Levels**: {', '.join([f'{x:.1e}' for x in sorted(data['noise_level'].unique())])}
- **Derivative Orders**: {', '.join([str(x) for x in sorted(data['derivative_order'].unique())])}
- **Data Size**: {data['data_size'].iloc[0]} points per experiment
- **Total Experiments**: {len(data):,} combinations

#### 🎯 Summary Statistics

**Overall Method Rankings** (by average RMSE across all conditions):
"""
    
    method_rankings = data.groupby('method')['rmse'].mean().sort_values()
    for i, (method, rmse) in enumerate(method_rankings.items(), 1):
        report_content += f"{i}. **{method}**: {rmse:.2e}\n"
    
    report_content += f"""

#### 🔍 Key Insights

1. **Performance Degradation**: All methods show increasing RMSE with higher derivative orders
2. **Noise Sensitivity**: Performance varies significantly with noise level
3. **Method Specialization**: Different methods excel at different derivative orders

## Visualizations

- `figures/method_derivative_heatmap.png`: RMSE heatmap by method and derivative order
- `figures/noise_performance.png`: Performance vs noise level (log-log plots)
- `figures/method_rankings.png`: Overall method performance rankings
- `figures/derivative_degradation.png`: Performance degradation with derivative order

## Data Files

- `best_methods_per_derivative.csv`: Best performing method for each derivative order
- `comprehensive_summary_statistics.csv`: Detailed statistics for all method/derivative combinations

## Methodology

- **Benchmark System**: Lotka-Volterra periodic ODE system
- **Summary Statistic**: Root Mean Square Error (RMSE)
- **Evaluation**: Function values and derivatives (orders 0-5)
- **Cross-validation**: Multiple noise levels and observables tested

---

*Report generated using Python with pandas, matplotlib, and seaborn*
"""
    
    with open(f'{report_dir}/README.md', 'w') as f:
        f.write(report_content)

if __name__ == "__main__":
    main()