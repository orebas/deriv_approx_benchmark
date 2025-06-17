#!/usr/bin/env python3
"""
PROPER Noisy Data Benchmark - Test methods on actually noisy data
"""

import pandas as pd
import numpy as np
import glob
import time
from datetime import datetime
from comprehensive_methods_library import create_all_methods, get_method_categories
from scipy.interpolate import CubicSpline

def generate_noisy_test_cases():
    """Generate proper noisy test cases from clean Julia data"""
    
    print("🎯 GENERATING PROPER NOISY TEST CASES")
    print("="*50)
    
    # Load one clean dataset to get the underlying ODE solution
    julia_files = glob.glob("results/sweep_lv_periodic_n*.csv")
    
    # Use the lowest noise file as our "clean" reference
    clean_file = None
    for file in julia_files:
        if "n0.0001" in file:  # Lowest noise level
            clean_file = file
            break
    
    if not clean_file:
        clean_file = julia_files[0]
    
    print(f"Using {clean_file} as clean reference")
    
    df = pd.read_csv(clean_file)
    
    # Get clean time series data
    clean_data = df[
        (df['derivative_order'] == 0) & 
        (df['observable'] == 'x1(t)')
    ].head(101)  # Use decent amount of data
    
    t_clean = clean_data['time'].values
    y_clean = clean_data['true_value'].values
    
    # Sort and remove duplicates
    sort_idx = np.argsort(t_clean)
    t_clean = t_clean[sort_idx]
    y_clean = y_clean[sort_idx]
    
    unique_idx = np.unique(t_clean, return_index=True)[1]
    t_clean = t_clean[unique_idx]
    y_clean = y_clean[unique_idx]
    
    print(f"Clean reference data: {len(t_clean)} points, t=[{t_clean.min():.2f}, {t_clean.max():.2f}]")
    
    # Generate test cases with controlled noise levels
    noise_levels = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    test_cases = []
    
    np.random.seed(42)  # Reproducible noise
    
    for noise_level in noise_levels:
        if noise_level == 0.0:
            # Perfect data
            y_noisy = y_clean.copy()
        else:
            # Add Gaussian noise
            noise_std = noise_level * np.std(y_clean)
            noise = np.random.normal(0, noise_std, len(y_clean))
            y_noisy = y_clean + noise
        
        test_cases.append({
            'noise_level': noise_level,
            't': t_clean.copy(),
            'y_clean': y_clean.copy(),
            'y_noisy': y_noisy.copy(),
            'noise_std': noise_level * np.std(y_clean) if noise_level > 0 else 0.0
        })
        
        actual_noise_std = np.std(y_noisy - y_clean)
        print(f"  Noise level {noise_level:.1e}: actual std = {actual_noise_std:.2e}")
    
    return test_cases

def run_proper_noisy_benchmark():
    """Run benchmark on properly generated noisy data"""
    
    print("\\n🚀 PROPER NOISY DATA BENCHMARK")
    print("="*60)
    
    # Generate noisy test cases
    test_cases = generate_noisy_test_cases()
    
    # Get methods
    sample_case = test_cases[0]
    methods = create_all_methods(sample_case['t'], sample_case['y_noisy'])
    categories = get_method_categories()
    
    print(f"\\nTesting {len(methods)} methods on {len(test_cases)} noise levels")
    
    all_results = []
    
    for case_idx, test_case in enumerate(test_cases):
        noise_level = test_case['noise_level']
        t = test_case['t']
        y_clean = test_case['y_clean']
        y_noisy = test_case['y_noisy']
        
        print(f"\\nNoise level {noise_level:.1e} (case {case_idx+1}/{len(test_cases)}):")
        
        # Create reference spline from CLEAN data for derivative truth
        ref_spline = CubicSpline(t, y_clean)
        
        # Compute normalization factor (range of clean data)
        y_range = y_clean.max() - y_clean.min()
        if y_range == 0:
            y_range = 1.0  # Avoid division by zero
        
        # Test each method on NOISY data
        methods = create_all_methods(t, y_noisy)  # Fit to noisy data!
        
        for method_name, method in methods.items():
            try:
                start_time = time.time()
                results = method.evaluate(t, max_derivative=3)
                eval_time = time.time() - start_time
                
                # Get category
                category = 'Unknown'
                for cat, method_list in categories.items():
                    if method_name in method_list:
                        category = cat
                        break
                
                if results['success']:
                    # Test each derivative order
                    for deriv_order in range(4):  # 0, 1, 2, 3
                        
                        # Get predictions (from fitting noisy data)
                        if deriv_order == 0:
                            y_pred = results['y']
                        else:
                            y_pred = results[f'd{deriv_order}']
                        
                        # Get TRUE derivatives (from clean reference)
                        if deriv_order == 0:
                            y_true = y_clean  # True function values
                        else:
                            y_true = ref_spline.derivative(deriv_order)(t)
                        
                        # Calculate errors vs TRUE values
                        errors = y_pred - y_true
                        rmse = np.sqrt(np.mean(errors**2))
                        mae = np.mean(np.abs(errors))
                        max_error = np.max(np.abs(errors))
                        
                        # Calculate normalized errors (by range of observable)
                        rmse_normalized = rmse / y_range
                        mae_normalized = mae / y_range
                        max_error_normalized = max_error / y_range
                        
                        # Store result
                        all_results.append({
                            'method': method_name,
                            'category': category,
                            'noise_level': noise_level,
                            'derivative_order': deriv_order,
                            'rmse': rmse,
                            'mae': mae,
                            'max_error': max_error,
                            'rmse_normalized': rmse_normalized,
                            'mae_normalized': mae_normalized,
                            'max_error_normalized': max_error_normalized,
                            'eval_time': eval_time,
                            'fit_time': method.fit_time,
                            'success': True,
                            'data_type': 'properly_noisy'
                        })
                        
                        if deriv_order == 0 and noise_level in [0.0, 1e-3, 1e-2]:
                            print(f"    {method_name:15s}: RMSE={rmse:.2e}")
                
                else:
                    # Method failed
                    category = 'Unknown'
                    for cat, method_list in categories.items():
                        if method_name in method_list:
                            category = cat
                            break
                    
                    for deriv_order in range(4):
                        all_results.append({
                            'method': method_name,
                            'category': category,
                            'noise_level': noise_level,
                            'derivative_order': deriv_order,
                            'rmse': np.nan,
                            'mae': np.nan,
                            'max_error': np.nan,
                            'rmse_normalized': np.nan,
                            'mae_normalized': np.nan,
                            'max_error_normalized': np.nan,
                            'eval_time': eval_time,
                            'fit_time': method.fit_time,
                            'success': False,
                            'error': results.get('error', 'Unknown'),
                            'data_type': 'properly_noisy'
                        })
            
            except Exception as e:
                # Exception during method execution
                for deriv_order in range(4):
                    all_results.append({
                        'method': method_name,
                        'category': 'Unknown',
                        'noise_level': noise_level,
                        'derivative_order': deriv_order,
                        'rmse': np.nan,
                        'mae': np.nan,
                        'max_error': np.nan,
                        'rmse_normalized': np.nan,
                        'mae_normalized': np.nan,
                        'max_error_normalized': np.nan,
                        'eval_time': np.nan,
                        'fit_time': np.nan,
                        'success': False,
                        'error': str(e)[:200],
                        'data_type': 'properly_noisy'
                    })
    
    # Save results
    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"results/proper_noisy_benchmark_{timestamp}.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"\\n🎉 PROPER NOISY BENCHMARK COMPLETE!")
    print(f"📁 Results saved to: {output_file}")
    
    # Analysis
    analyze_proper_results(results_df)
    
    return output_file

def analyze_proper_results(df):
    """Analyze results from proper noisy benchmark"""
    
    print(f"\\n📊 PROPER NOISY BENCHMARK ANALYSIS")
    print("="*50)
    
    successful_df = df[df['success'] == True]
    
    print(f"Total evaluations: {len(df)}")
    print(f"Successful evaluations: {len(successful_df)} ({100*len(successful_df)/len(df):.1f}%)")
    
    # Performance by noise level (function approximation only)
    print(f"\\n🎯 FUNCTION APPROXIMATION PERFORMANCE BY NOISE LEVEL:")
    func_data = successful_df[successful_df['derivative_order'] == 0]
    
    for noise_level in sorted(func_data['noise_level'].unique()):
        noise_data = func_data[func_data['noise_level'] == noise_level]
        print(f"\\n  Noise level {noise_level:.1e}:")
        
        method_rmse = noise_data.groupby('method')['rmse'].mean().sort_values()
        for method, rmse in method_rmse.head(5).items():
            print(f"    {method:15s}: RMSE = {rmse:.2e}")
    
    # Noise robustness analysis
    print(f"\\n🛡️  NOISE ROBUSTNESS ANALYSIS:")
    print("(Methods that maintain low RMSE across noise levels)")
    
    # Calculate robustness metric: ratio of noisy to clean performance
    robustness_data = []
    for method in func_data['method'].unique():
        method_data = func_data[func_data['method'] == method]
        
        clean_rmse = method_data[method_data['noise_level'] == 0.0]['rmse'].mean()
        high_noise_rmse = method_data[method_data['noise_level'] == 0.05]['rmse'].mean()
        
        if not np.isnan(clean_rmse) and not np.isnan(high_noise_rmse) and clean_rmse > 0:
            robustness_ratio = high_noise_rmse / clean_rmse
            robustness_data.append({
                'method': method,
                'clean_rmse': clean_rmse,
                'noisy_rmse': high_noise_rmse,
                'robustness_ratio': robustness_ratio
            })
    
    robustness_df = pd.DataFrame(robustness_data).sort_values('robustness_ratio')
    
    print("\\n🏆 MOST ROBUST METHODS (lowest degradation with noise):")
    for _, row in robustness_df.head(10).iterrows():
        method = row['method']
        ratio = row['robustness_ratio']
        clean = row['clean_rmse']
        noisy = row['noisy_rmse']
        print(f"  {method:15s}: {ratio:6.1f}x degradation (clean:{clean:.2e} → noisy:{noisy:.2e})")

if __name__ == "__main__":
    output_file = run_proper_noisy_benchmark()
    
    print(f"\\n🎯 NOW WE HAVE PROPER NOISY DATA RESULTS!")
    print(f"Previous benchmark was testing on clean data (explains perfect CubicSpline)")
    print(f"This benchmark tests on actual noisy data vs clean truth")
    print(f"Results: {output_file}")