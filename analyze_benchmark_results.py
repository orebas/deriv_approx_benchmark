#!/usr/bin/env python3
"""
Analyze benchmark results by creating pivot tables and summary statistics.
"""

import pandas as pd
import numpy as np

def load_and_summarize_data():
    """Load the raw benchmark data and create summary pivot tables."""
    
    # Load the data
    print("Loading benchmark data...")
    df = pd.read_csv('unified_analysis/RAW_MASTER_TABLE.csv')
    
    print(f"Total records: {len(df)}")
    print(f"Unique methods: {df['method'].nunique()}")
    print(f"Test cases: {df['test_case'].unique()}")
    print(f"Noise levels: {sorted(df['noise_level'].unique())}")
    print(f"Derivative orders: {sorted(df['derivative_order'].unique())}")
    
    # Create pivot table: average RMSE by derivative level, method, and noise level
    print("\nCreating pivot table...")
    pivot_rmse = df.groupby(['derivative_order', 'method', 'noise_level'])['rmse'].mean().reset_index()
    
    # Also create a wider format for easier analysis
    pivot_wide = pivot_rmse.pivot_table(
        index=['method', 'noise_level'], 
        columns='derivative_order', 
        values='rmse'
    ).reset_index()
    
    # Save the pivot tables
    pivot_rmse.to_csv('unified_analysis/pivot_rmse_by_method_noise_deriv.csv', index=False)
    pivot_wide.to_csv('unified_analysis/pivot_rmse_wide_format.csv', index=False)
    
    print("Pivot tables saved to:")
    print("- unified_analysis/pivot_rmse_by_method_noise_deriv.csv")
    print("- unified_analysis/pivot_rmse_wide_format.csv")
    
    # Create summary statistics
    print("\nGenerating summary statistics...")
    
    # Performance by method (average across all conditions)
    method_performance = df.groupby('method').agg({
        'rmse': ['mean', 'median', 'std'],
        'eval_time': ['mean', 'median'],
        'fit_time': ['mean', 'median']
    }).round(6)
    method_performance.columns = ['_'.join(col).strip() for col in method_performance.columns]
    method_performance = method_performance.reset_index()
    method_performance.to_csv('unified_analysis/method_performance_summary.csv', index=False)
    
    # Performance by derivative order
    deriv_performance = df.groupby('derivative_order').agg({
        'rmse': ['mean', 'median', 'std', 'min', 'max']
    }).round(6)
    deriv_performance.columns = ['_'.join(col).strip() for col in deriv_performance.columns]
    deriv_performance = deriv_performance.reset_index()
    deriv_performance.to_csv('unified_analysis/derivative_order_analysis.csv', index=False)
    
    # Performance by noise level
    noise_performance = df.groupby('noise_level').agg({
        'rmse': ['mean', 'median', 'std', 'min', 'max']
    }).round(6)
    noise_performance.columns = ['_'.join(col).strip() for col in noise_performance.columns]
    noise_performance = noise_performance.reset_index()
    noise_performance.to_csv('unified_analysis/noise_level_analysis.csv', index=False)
    
    # Julia vs Python comparison
    julia_methods = df[df['method'].str.contains('Julia|GPR|AAA|BSpline|LOESS|TVDiff', case=False, na=False)]
    python_methods = df[df['method'].str.contains('Python', case=False, na=False)]
    
    julia_summary = julia_methods.groupby('method').agg({
        'rmse': ['mean', 'median', 'std'],
        'eval_time': ['mean', 'median']
    }).round(6)
    julia_summary.columns = ['_'.join(col).strip() for col in julia_summary.columns]
    julia_summary['implementation'] = 'Julia'
    
    python_summary = python_methods.groupby('method').agg({
        'rmse': ['mean', 'median', 'std'],
        'eval_time': ['mean', 'median']
    }).round(6)
    python_summary.columns = ['_'.join(col).strip() for col in python_summary.columns]
    python_summary['implementation'] = 'Python'
    
    implementation_comparison = pd.concat([julia_summary, python_summary]).reset_index()
    implementation_comparison.to_csv('unified_analysis/julia_vs_python_comparison.csv', index=False)
    
    print("Summary files generated:")
    print("- unified_analysis/method_performance_summary.csv")
    print("- unified_analysis/derivative_order_analysis.csv") 
    print("- unified_analysis/noise_level_analysis.csv")
    print("- unified_analysis/julia_vs_python_comparison.csv")
    
    return pivot_rmse, pivot_wide, df

def analyze_best_performers(df):
    """Find the best performing methods for different scenarios."""
    
    print("\n" + "="*60)
    print("BEST PERFORMER ANALYSIS")
    print("="*60)
    
    # Best overall performers (lowest average RMSE)
    overall_best = df.groupby('method')['rmse'].mean().sort_values().head(10)
    print("\nTop 10 methods by overall average RMSE:")
    for method, rmse in overall_best.items():
        print(f"  {method:<30} {rmse:.2e}")
    
    # Best performers by derivative order
    print(f"\nBest performers by derivative order:")
    for deriv_order in sorted(df['derivative_order'].unique()):
        deriv_data = df[df['derivative_order'] == deriv_order]
        best = deriv_data.groupby('method')['rmse'].mean().sort_values().head(3)
        print(f"\n  Derivative order {deriv_order}:")
        for i, (method, rmse) in enumerate(best.items(), 1):
            print(f"    {i}. {method:<25} {rmse:.2e}")
    
    # Best performers by noise level
    print(f"\nBest performers by noise level:")
    for noise in sorted(df['noise_level'].unique()):
        noise_data = df[df['noise_level'] == noise]
        best = noise_data.groupby('method')['rmse'].mean().sort_values().head(3)
        print(f"\n  Noise level {noise}:")
        for i, (method, rmse) in enumerate(best.items(), 1):
            print(f"    {i}. {method:<25} {rmse:.2e}")

def analyze_gpr_performance(df):
    """Specifically analyze GPR performance after the fix."""
    
    print("\n" + "="*60)
    print("GPR PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Find GPR methods
    gpr_methods = df[df['method'].str.contains('GPR', case=False, na=False)]['method'].unique()
    print(f"GPR methods found: {list(gpr_methods)}")
    
    if len(gpr_methods) > 0:
        gpr_data = df[df['method'].isin(gpr_methods)]
        
        print(f"\nGPR Performance Summary:")
        gpr_summary = gpr_data.groupby('method').agg({
            'rmse': ['mean', 'median', 'std', 'min', 'max'],
            'eval_time': ['mean', 'median']
        }).round(6)
        
        for method in gpr_methods:
            method_data = gpr_data[gpr_data['method'] == method]
            print(f"\n  {method}:")
            print(f"    Average RMSE: {method_data['rmse'].mean():.2e}")
            print(f"    Median RMSE:  {method_data['rmse'].median():.2e}")
            print(f"    Min RMSE:     {method_data['rmse'].min():.2e}")
            print(f"    Max RMSE:     {method_data['rmse'].max():.2e}")
            print(f"    Avg eval time: {method_data['eval_time'].mean():.4f}s")

if __name__ == "__main__":
    pivot_rmse, pivot_wide, df = load_and_summarize_data()
    analyze_best_performers(df)
    analyze_gpr_performance(df)
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Check the generated CSV files in unified_analysis/ for detailed pivot tables.")