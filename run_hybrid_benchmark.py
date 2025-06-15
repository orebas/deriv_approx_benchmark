#!/usr/bin/env python3
"""
Hybrid benchmark: Run Python methods on same data as Julia methods
Then combine results for comprehensive analysis
"""

import pandas as pd
import numpy as np
import glob
import json
from pathlib import Path
from python_methods_standalone import *

def extract_data_from_julia_results(julia_csv_file):
    """Extract the original time series data from Julia results"""
    
    # Load Julia results
    df = pd.read_csv(julia_csv_file)
    
    # Get unique parameter combinations
    param_combinations = df[['noise_level', 'data_size', 'observable']].drop_duplicates()
    
    extracted_data = []
    
    for _, params in param_combinations.iterrows():
        # Filter to this parameter combination
        param_data = df[
            (df['noise_level'] == params['noise_level']) & 
            (df['data_size'] == params['data_size']) & 
            (df['observable'] == params['observable'])
        ]
        
        if len(param_data) > 0:
            # Extract time and value series
            t = param_data['time'].values
            y_true = param_data['true_value'].values
            
            # Sort by time
            sort_idx = np.argsort(t)
            t_sorted = t[sort_idx]
            y_sorted = y_true[sort_idx]
            
            extracted_data.append({
                'noise_level': params['noise_level'],
                'data_size': params['data_size'],
                'observable': params['observable'],
                't': t_sorted,
                'y': y_sorted
            })
    
    return extracted_data

def run_python_methods_on_julia_data():
    """Run Python methods on the same data used by Julia benchmark"""
    
    print("🐍 RUNNING PYTHON METHODS ON JULIA BENCHMARK DATA")
    print("=" * 60)
    
    # Find Julia result files
    julia_files = glob.glob("results/sweep_lv_periodic_n*.csv")
    print(f"Found {len(julia_files)} Julia result files")
    
    all_python_results = []
    
    for julia_file in julia_files:
        print(f"\n📊 Processing {julia_file}")
        
        # Extract original data
        data_sets = extract_data_from_julia_results(julia_file)
        print(f"  Extracted {len(data_sets)} parameter combinations")
        
        for i, data_set in enumerate(data_sets):
            print(f"  Running Python methods on dataset {i+1}/{len(data_sets)}...")
            
            t = data_set['t']
            y = data_set['y']
            
            # Skip if data is too small
            if len(t) < 5:
                continue
            
            # Run Python methods
            python_methods = ['savgol', 'fourier', 'chebyshev', 'finitediff']
            
            try:
                results = run_method_comparison(t, y, t, python_methods)
                
                # Convert to benchmark format
                for method_name, method_result in results.items():
                    if method_result is not None:
                        
                        # Calculate errors for each derivative order
                        for deriv_order in range(4):  # 0-3 derivatives
                            
                            # Get true derivatives from Julia data (using GPR as reference)
                            julia_data = pd.read_csv(julia_file)
                            reference_data = julia_data[
                                (julia_data['noise_level'] == data_set['noise_level']) &
                                (julia_data['data_size'] == data_set['data_size']) &
                                (julia_data['observable'] == data_set['observable']) &
                                (julia_data['derivative_order'] == deriv_order) &
                                (julia_data['method'] == 'GPR')  # Use GPR as "truth"
                            ]
                            
                            if len(reference_data) > 0:
                                # Get predictions
                                if deriv_order == 0:
                                    y_pred = method_result['y']
                                else:
                                    y_pred = method_result[f'd{deriv_order}']
                                
                                y_true = reference_data['true_value'].values
                                
                                # Ensure same length
                                min_len = min(len(y_pred), len(y_true))
                                y_pred = y_pred[:min_len]
                                y_true = y_true[:min_len]
                                
                                # Calculate errors
                                errors = y_pred - y_true
                                rmse = np.sqrt(np.mean(errors**2))
                                mae = np.mean(np.abs(errors))
                                max_error = np.max(np.abs(errors))
                                
                                # Store result in same format as Julia
                                all_python_results.append({
                                    'method': f'Python_{method_name}',
                                    'noise_level': data_set['noise_level'],
                                    'data_size': data_set['data_size'],
                                    'observable': data_set['observable'],
                                    'derivative_order': deriv_order,
                                    'rmse': rmse,
                                    'mae': mae,
                                    'max_error': max_error,
                                    'source_file': julia_file
                                })
                
            except Exception as e:
                print(f"    ❌ Error processing dataset: {e}")
                continue
    
    # Convert to DataFrame and save
    if all_python_results:
        python_df = pd.DataFrame(all_python_results)
        output_file = f"results/python_methods_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        python_df.to_csv(output_file, index=False)
        
        print(f"\n✅ Python method results saved to: {output_file}")
        print(f"📊 Total Python method evaluations: {len(python_df)}")
        
        # Show summary
        print(f"\nPython methods tested: {sorted(python_df['method'].unique())}")
        print(f"Parameter combinations: {len(python_df[['noise_level', 'data_size', 'observable']].drop_duplicates())}")
        
        return output_file
    else:
        print("❌ No Python results generated")
        return None

def combine_julia_and_python_results(python_results_file):
    """Combine Julia and Python results for unified analysis"""
    
    print(f"\n🔗 COMBINING JULIA AND PYTHON RESULTS")
    print("=" * 50)
    
    # Load Python results
    python_df = pd.read_csv(python_results_file)
    
    # Load Julia results
    julia_files = glob.glob("results/sweep_lv_periodic_n*.csv")
    julia_data = []
    
    for file in julia_files:
        df = pd.read_csv(file)
        # Extract summary data
        summary = df.groupby(['method', 'derivative_order', 'noise_level', 'observable', 'data_size']).agg({
            'rmse': 'first',
            'mae': 'first',
            'max_error': 'first'
        }).reset_index()
        julia_data.append(summary)
    
    julia_df = pd.concat(julia_data, ignore_index=True)
    
    # Combine datasets
    combined_df = pd.concat([julia_df, python_df], ignore_index=True)
    
    # Save combined results
    combined_file = f"results/combined_julia_python_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    combined_df.to_csv(combined_file, index=False)
    
    print(f"✅ Combined results saved to: {combined_file}")
    print(f"📊 Total methods: {len(combined_df['method'].unique())}")
    print(f"   Julia methods: {sorted([m for m in combined_df['method'].unique() if not m.startswith('Python_')])}")
    print(f"   Python methods: {sorted([m for m in combined_df['method'].unique() if m.startswith('Python_')])}")
    
    return combined_file

def main():
    """Run the hybrid benchmark"""
    
    # Step 1: Run Python methods on Julia data
    python_results_file = run_python_methods_on_julia_data()
    
    if python_results_file:
        # Step 2: Combine results
        combined_file = combine_julia_and_python_results(python_results_file)
        
        print(f"\n🎉 HYBRID BENCHMARK COMPLETE!")
        print(f"📄 Combined results: {combined_file}")
        print(f"📊 Ready for comprehensive analysis with all methods!")
        
        return combined_file
    else:
        print("❌ Hybrid benchmark failed - no Python results generated")
        return None

if __name__ == "__main__":
    main()