#!/usr/bin/env python3
"""
Massive Derivative Approximation Benchmark
Tests 14+ methods from comprehensive_methods_library
"""

import pandas as pd
import numpy as np
import glob
import time
from pathlib import Path
from datetime import datetime
from comprehensive_methods_library import create_all_methods, get_method_categories

def extract_julia_data_points(julia_csv_file, max_files=5):
    """Extract original time series data from Julia results"""
    
    df = pd.read_csv(julia_csv_file)
    
    # Get parameter combinations
    param_combinations = df[['noise_level', 'data_size', 'observable']].drop_duplicates()
    
    extracted_data = []
    count = 0
    
    for _, params in param_combinations.iterrows():
        if count >= max_files:
            break
            
        # Filter to this parameter combination and get original time series
        param_data = df[
            (df['noise_level'] == params['noise_level']) & 
            (df['data_size'] == params['data_size']) & 
            (df['observable'] == params['observable'])
        ]
        
        if len(param_data) > 0:
            # Extract time and true values
            t_data = param_data['time'].values
            y_data = param_data['true_value'].values
            
            # Sort by time and remove duplicates
            sort_idx = np.argsort(t_data)
            t_sorted = t_data[sort_idx]
            y_sorted = y_data[sort_idx]
            
            # Remove duplicate time points
            unique_idx = np.unique(t_sorted, return_index=True)[1]
            t_unique = t_sorted[unique_idx]
            y_unique = y_sorted[unique_idx]
            
            if len(t_unique) >= 10:  # Minimum data points
                extracted_data.append({
                    'noise_level': params['noise_level'],
                    'data_size': params['data_size'],
                    'observable': params['observable'],
                    't': t_unique,
                    'y': y_unique,
                    'source_file': julia_csv_file
                })
                count += 1
    
    return extracted_data

def run_massive_benchmark():
    """Run comprehensive benchmark with all available methods"""
    
    print("🚀 MASSIVE DERIVATIVE APPROXIMATION BENCHMARK")
    print("=" * 60)
    
    # Load available Julia data
    julia_files = glob.glob("results/sweep_lv_periodic_n*.csv")
    print(f"Found {len(julia_files)} Julia result files")
    
    # Extract test datasets (limit to reasonable number for comprehensive testing)
    all_datasets = []
    for julia_file in julia_files[:3]:  # Use first 3 files to keep runtime reasonable
        datasets = extract_julia_data_points(julia_file, max_files=2)
        all_datasets.extend(datasets)
    
    print(f"Extracted {len(all_datasets)} test datasets")
    
    # Get all available methods
    if len(all_datasets) > 0:
        sample_data = all_datasets[0]
        all_methods = create_all_methods(sample_data['t'], sample_data['y'])
        method_categories = get_method_categories()
        
        print(f"\\n📊 Testing {len(all_methods)} methods:")
        for category, methods in method_categories.items():
            print(f"  {category}: {methods}")
    
    # Run comprehensive benchmark
    all_results = []
    total_combinations = len(all_datasets) * len(all_methods)
    combination_count = 0
    
    print(f"\\n🏃 Starting benchmark: {total_combinations} total combinations")
    print("-" * 60)
    
    for dataset_idx, dataset in enumerate(all_datasets):
        print(f"\\nDataset {dataset_idx+1}/{len(all_datasets)}: noise={dataset['noise_level']}, size={dataset['data_size']}, obs={dataset['observable']}")
        
        t = dataset['t']
        y = dataset['y']
        
        # Create methods for this dataset
        methods = create_all_methods(t, y)
        
        # Test each method
        for method_name, method in methods.items():
            combination_count += 1
            
            if combination_count % 10 == 0:
                print(f"  Progress: {combination_count}/{total_combinations} ({100*combination_count/total_combinations:.1f}%)")
            
            try:
                # Fit and evaluate method
                start_time = time.time()
                results = method.evaluate(t, max_derivative=3)  # Test up to 3rd derivative
                eval_time = time.time() - start_time
                
                if results['success']:
                    # Calculate errors vs "truth" (using original data as ground truth for function)
                    for deriv_order in range(4):  # 0, 1, 2, 3
                        if deriv_order == 0:
                            y_pred = results['y']
                            y_true = y  # Original data is "truth" for function values
                        else:
                            y_pred = results[f'd{deriv_order}']
                            # For derivatives, we need to calculate "true" derivatives
                            # Use high-resolution spline as reference
                            from scipy.interpolate import CubicSpline
                            ref_spline = CubicSpline(t, y)
                            y_true = ref_spline.derivative(deriv_order)(t)
                        
                        # Calculate errors
                        errors = y_pred - y_true
                        rmse = np.sqrt(np.mean(errors**2))
                        mae = np.mean(np.abs(errors))
                        max_error = np.max(np.abs(errors))
                        
                        # Store result
                        all_results.append({
                            'method': method_name,
                            'category': get_method_category(method_name, method_categories),
                            'noise_level': dataset['noise_level'],
                            'data_size': dataset['data_size'],
                            'observable': dataset['observable'],
                            'derivative_order': deriv_order,
                            'rmse': rmse,
                            'mae': mae,
                            'max_error': max_error,
                            'eval_time': eval_time,
                            'fit_time': method.fit_time,
                            'success': True,
                            'source_file': dataset['source_file']
                        })
                else:
                    # Method failed
                    for deriv_order in range(4):
                        all_results.append({
                            'method': method_name,
                            'category': get_method_category(method_name, method_categories),
                            'noise_level': dataset['noise_level'],
                            'data_size': dataset['data_size'],
                            'observable': dataset['observable'],
                            'derivative_order': deriv_order,
                            'rmse': np.nan,
                            'mae': np.nan,
                            'max_error': np.nan,
                            'eval_time': eval_time,
                            'fit_time': method.fit_time,
                            'success': False,
                            'error': results.get('error', 'Unknown error'),
                            'source_file': dataset['source_file']
                        })
                        
            except Exception as e:
                print(f"    ❌ {method_name} failed: {str(e)[:100]}")
                
                # Store failure
                for deriv_order in range(4):
                    all_results.append({
                        'method': method_name,
                        'category': get_method_category(method_name, method_categories),
                        'noise_level': dataset['noise_level'],
                        'data_size': dataset['data_size'],
                        'observable': dataset['observable'],
                        'derivative_order': deriv_order,
                        'rmse': np.nan,
                        'mae': np.nan,
                        'max_error': np.nan,
                        'eval_time': np.nan,
                        'fit_time': np.nan,
                        'success': False,
                        'error': str(e),
                        'source_file': dataset['source_file']
                    })
    
    # Save results
    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"results/massive_benchmark_{timestamp}.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"\\n🎉 MASSIVE BENCHMARK COMPLETE!")
    print(f"📊 Results saved to: {output_file}")
    print(f"📈 Total evaluations: {len(results_df)}")
    
    # Quick summary
    print(f"\\n📋 QUICK SUMMARY:")
    print(f"Methods tested: {results_df['method'].nunique()}")
    print(f"Successful evaluations: {results_df['success'].sum()}/{len(results_df)}")
    
    success_by_method = results_df.groupby('method')['success'].agg(['sum', 'count', 'mean']).sort_values('mean', ascending=False)
    print(f"\\n🏆 TOP METHODS BY SUCCESS RATE:")
    for method, stats in success_by_method.head(10).iterrows():
        print(f"  {method}: {stats['mean']*100:.1f}% ({stats['sum']}/{stats['count']})")
    
    print(f"\\n📊 PERFORMANCE BY CATEGORY:")
    if 'category' in results_df.columns:
        category_success = results_df.groupby('category')['success'].mean().sort_values(ascending=False)
        for category, success_rate in category_success.items():
            print(f"  {category}: {success_rate*100:.1f}% success rate")
    
    return output_file

def get_method_category(method_name, categories):
    """Get category for a method"""
    for category, methods in categories.items():
        if method_name in methods:
            return category
    return 'Unknown'

def install_required_packages():
    """Check and install required packages"""
    import subprocess
    import sys
    
    required_packages = [
        'scikit-learn>=1.0.0',
        'scipy>=1.7.0',
        'numpy>=1.20.0',
        'pandas>=1.3.0'
    ]
    
    print("📦 Checking required packages...")
    
    for package in required_packages:
        try:
            pkg_name = package.split('>=')[0]
            __import__(pkg_name.replace('-', '_'))
            print(f"  ✓ {pkg_name}")
        except ImportError:
            print(f"  ❌ {pkg_name} not found, installing...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"  ✓ {pkg_name} installed")

if __name__ == "__main__":
    # Check dependencies
    install_required_packages()
    
    # Run massive benchmark
    output_file = run_massive_benchmark()
    
    print(f"\\n🎯 Next steps:")
    print(f"1. Analyze results: python analyze_massive_results.py {output_file}")
    print(f"2. Generate report: python create_massive_report.py {output_file}")
    print(f"3. Compare with Julia methods: python compare_all_methods.py")