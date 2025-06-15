#!/usr/bin/env python3
"""
Unified Comparison of ALL Methods
Combines raw Python and Julia benchmark results, formats them for clarity,
and generates a master data table, analysis, and plots.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Attempt to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_ENABLED = True
    plt.style.use('seaborn-v0_8-whitegrid')
except ImportError:
    PLOTTING_ENABLED = False
    print("WARNING: matplotlib or seaborn not found. Skipping plot generation.")

def load_and_format_results():
    """
    Load raw benchmark results and format method names with implementation suffixes.
    """
    
    print("🔍 LOADING & FORMATTING RAW BENCHMARK RESULTS")
    print("="*50)
    
    all_data = []

    # 1. Load Raw Python Data
    python_raw_file = "results/python_raw_benchmark.csv"
    if Path(python_raw_file).exists():
        print(f"Loading Python results from: {python_raw_file}")
        python_df = pd.read_csv(python_raw_file)
        python_df['method'] = python_df['method'] + '_Python'
        python_df['implementation'] = 'Python'
        all_data.append(python_df)
    else:
        print(f"FATAL: Python raw data file not found at {python_raw_file}")
        return None

    # 2. Load Raw Julia Data
    julia_raw_file = "results/julia_raw_benchmark.csv"
    if Path(julia_raw_file).exists():
        print(f"Loading Julia results from: {julia_raw_file}")
        julia_df = pd.read_csv(julia_raw_file)
        julia_df['method'] = julia_df['method'] + '_Julia'
        julia_df['implementation'] = 'Julia'
        all_data.append(julia_df)
    else:
        print(f"FATAL: Julia raw data file not found at {julia_raw_file}")
        return None
    
    # Combine the data
    final_df = pd.concat(all_data, ignore_index=True)
    
    print(f"\nFormatted dataset: {len(final_df)} rows")
    print(f"Methods: {sorted(final_df['method'].unique())}")
    
    return final_df

def create_unified_analysis(df):
    """Create comprehensive analysis from the formatted raw data."""
    
    print("\n📊 CREATING UNIFIED ANALYSIS")
    print("="*50)
    
    output_dir = Path("unified_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # 1. SAVE THE MASTER TABLE
    master_table_path = output_dir / "RAW_MASTER_TABLE.csv"
    print(f"📋 Saving master data table to: {master_table_path}")
    df.to_csv(master_table_path, index=False)
    
    # 2. CREATE VISUALIZATIONS
    if not PLOTTING_ENABLED:
        print("\nSkipping plot generation.")
        return

    print("📈 Creating visualizations...")
    
    plot_df = df[df['success'] == True].copy()
    
    # Sort methods for consistent plot coloring
    method_order = sorted(plot_df['method'].unique())
    
    g = sns.FacetGrid(plot_df, col="derivative_order", hue="method",
                      col_wrap=2, sharey=False, height=6, aspect=1.5,
                      col_order=sorted(plot_df['derivative_order'].unique()),
                      hue_order=method_order,
                      palette='tab20')
    
    g.map(sns.lineplot, "noise_level", "rmse", marker='o', alpha=0.8)
    g.set(xscale='log', yscale='log')
    g.add_legend(title='Method',
                 bbox_to_anchor=(1.01, 0.5),
                 loc='center left',
                 borderaxespad=0)
    g.set_axis_labels("Noise Level", "RMSE")
    g.set_titles("Derivative Order: {col_name}")
    g.fig.suptitle("RMSE vs. Noise Level by Method", y=1.02, fontsize=16)
    g.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    
    plot_path = output_dir / "rmse_by_derivative_plots.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"🖼️ Saved derivative performance plot to: {plot_path}")

    # 3. GENERATE SUMMARY REPORT
    print("📝 Generating summary report...")
    
    summary_df = plot_df.groupby(['method', 'implementation', 'derivative_order']).agg(
        avg_rmse=('rmse', 'mean')
    ).reset_index()

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report = f"""
# UNIFIED DERIVATIVE BENCHMARK REPORT
Generated: {timestamp}

## EXECUTIVE SUMMARY

This report analyzes performance data from all Python and Julia methods across multiple derivative orders and noise levels. The full, granular dataset is available in `RAW_MASTER_TABLE.csv`.

- **Implementations**: {', '.join(df['implementation'].unique())}
- **Methods Compared**: {len(df['method'].unique())} total methods
- **Derivatives Tested**: {', '.join(map(str, sorted(df['derivative_order'].unique())))}

---

## TOP PERFORMERS BY DERIVATIVE ORDER

Top 5 methods for each derivative order, based on average RMSE across all noise levels.

"""
    
    for order in sorted(summary_df['derivative_order'].unique()):
        report += f"\n### Derivative Order {order}\n\n"
        report += "| Rank | Method | Avg RMSE |\n"
        report += "|------|--------|----------|\n"
        
        top_performers = summary_df[summary_df['derivative_order'] == order].sort_values('avg_rmse').head(5)
        
        for i, (_, row) in enumerate(top_performers.iterrows(), 1):
            report += f"| {i} | {row['method']} | {row['avg_rmse']:.2e} |\n"

    report += """
---

## FILES GENERATED

- `RAW_MASTER_TABLE.csv`: The complete, raw data from both benchmarks. **This is your single source of raw data.**
- `rmse_by_derivative_plots.png`: A plot showing RMSE vs. Noise Level for each derivative order.

## DATA SOURCES

- Python Raw Data: `results/python_raw_benchmark.csv`
- Julia Raw Data: `results/julia_raw_benchmark.csv`
"""
    
    report_path = output_dir / "UNIFIED_ANALYSIS_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"📄 Report saved to: {report_path}")


def main():
    """Main execution function"""
    
    df = load_and_format_results()
    
    if df is not None and not df.empty:
        create_unified_analysis(df)
        
        print(f"\n🎉 UNIFIED ANALYSIS COMPLETE!")
        print(f"📁 All results in: unified_analysis/")
        print(f"📊 The main file you requested is: unified_analysis/RAW_MASTER_TABLE.csv")
    else:
        print("❌ No raw data found to analyze. Please run the Python and Julia benchmarks first.")

if __name__ == "__main__":
    main()