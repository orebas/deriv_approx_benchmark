#!/usr/bin/env python3
"""
Focused Gaussian Process Comparison
Test different GP kernels and Matérn smoothness parameters on noisy data
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime
from enhanced_gp_methods import create_enhanced_gp_methods
from scipy.interpolate import CubicSpline

def generate_test_data():
    """Generate clean and noisy test data for GP comparison"""
    
    print("🎯 GENERATING TEST DATA FOR GP COMPARISON")
    print("="*50)
    
    # Create a smooth test function (similar to ODE solution)
    t = np.linspace(0, 4*np.pi, 101)
    
    # Lotka-Volterra-like function
    y_clean = 2 + 3*np.sin(t) + 1.5*np.cos(2*t) + 0.5*np.sin(3*t)
    
    # Multiple noise levels
    noise_levels = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    
    test_cases = []
    np.random.seed(42)  # Reproducible
    
    for noise_level in noise_levels:
        if noise_level == 0.0:
            y_noisy = y_clean.copy()
        else:
            noise_std = noise_level * np.std(y_clean)
            noise = np.random.normal(0, noise_std, len(y_clean))
            y_noisy = y_clean + noise
        
        test_cases.append({
            'noise_level': noise_level,
            't': t.copy(),
            'y_clean': y_clean.copy(),
            'y_noisy': y_noisy.copy()
        })
        
        actual_noise = np.std(y_noisy - y_clean)
        print(f"  Noise {noise_level:.1e}: actual std = {actual_noise:.2e}")
    
    return test_cases

def run_gp_comparison():
    """Run comprehensive GP kernel comparison"""
    
    print("\\n🚀 GAUSSIAN PROCESS KERNEL COMPARISON")
    print("="*60)
    
    test_cases = generate_test_data()
    
    # Get sample for method creation
    sample_case = test_cases[0]
    enhanced_gp_methods = create_enhanced_gp_methods(sample_case['t'], sample_case['y_noisy'])
    
    print(f"\\nTesting {len(enhanced_gp_methods)} GP variants:")
    for name in enhanced_gp_methods:
        print(f"  - {name}")
    
    all_results = []
    
    for case_idx, test_case in enumerate(test_cases):
        noise_level = test_case['noise_level']
        t = test_case['t']
        y_clean = test_case['y_clean']
        y_noisy = test_case['y_noisy']
        
        print(f"\\nNoise level {noise_level:.1e} (case {case_idx+1}/{len(test_cases)}):")
        
        # Create reference for derivatives
        ref_spline = CubicSpline(t, y_clean)
        
        # Compute normalization factor (range of clean data)
        y_range = y_clean.max() - y_clean.min()
        if y_range == 0:
            y_range = 1.0  # Avoid division by zero
        
        # Test each GP variant
        gp_methods = create_enhanced_gp_methods(t, y_noisy)
        
        for method_name, method in gp_methods.items():
            print(f"  Testing {method_name}...", end="")
            
            try:
                start_time = time.time()
                results = method.evaluate(t, max_derivative=3)
                eval_time = time.time() - start_time
                
                if results['success']:
                    print(" ✓")
                    
                    # Evaluate each derivative order
                    for deriv_order in range(4):  # 0, 1, 2, 3
                        
                        # Get predictions and truth
                        if deriv_order == 0:
                            y_pred = results['y']
                            y_true = y_clean
                        else:
                            y_pred = results[f'd{deriv_order}']
                            y_true = ref_spline.derivative(deriv_order)(t)
                        
                        # Calculate errors
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
                            'kernel_type': method.kernel_type,
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
                            'success': True
                        })
                else:
                    print(" ❌ Failed")
                    
                    for deriv_order in range(4):
                        all_results.append({
                            'method': method_name,
                            'kernel_type': method.kernel_type,
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
                            'success': False
                        })
            
            except Exception as e:
                print(f" ❌ Error: {str(e)[:30]}")
                
                for deriv_order in range(4):
                    all_results.append({
                        'method': method_name,
                        'kernel_type': getattr(method, 'kernel_type', 'unknown'),
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
                        'error': str(e)[:100]
                    })
    
    # Save results
    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"results/gp_comparison_{timestamp}.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"\\n🎉 GP COMPARISON COMPLETE!")
    print(f"📁 Results saved to: {output_file}")
    
    # Analyze results
    analyze_gp_results(results_df)
    
    return output_file

def analyze_gp_results(df):
    """Analyze GP comparison results"""
    
    print(f"\\n📊 GP KERNEL COMPARISON ANALYSIS")
    print("="*50)
    
    successful_df = df[df['success'] == True]
    
    print(f"Total evaluations: {len(df)}")
    print(f"Successful evaluations: {len(successful_df)} ({100*len(successful_df)/len(df):.1f}%)")
    
    # Function approximation performance by noise level
    print(f"\\n🎯 FUNCTION APPROXIMATION BY NOISE LEVEL:")
    func_data = successful_df[successful_df['derivative_order'] == 0]
    
    noise_levels = sorted(func_data['noise_level'].unique())
    
    for noise_level in noise_levels:
        print(f"\\n  📈 Noise level {noise_level:.1e}:")
        noise_data = func_data[func_data['noise_level'] == noise_level]
        
        method_rmse = noise_data.groupby('method')['rmse'].mean().sort_values()
        for method, rmse in method_rmse.items():
            kernel_info = noise_data[noise_data['method'] == method]['kernel_type'].iloc[0]
            print(f"    {method:15s} ({kernel_info:12s}): RMSE = {rmse:.2e}")
    
    # Robustness analysis
    print(f"\\n🛡️  NOISE ROBUSTNESS RANKING:")
    print("(Performance degradation from clean to highest noise)")
    
    # Create a dictionary to hold results by method
    method_perf = {}
    for method in func_data['method'].unique():
        method_perf[method] = {'kernel_type': func_data[func_data['method'] == method]['kernel_type'].iloc[0]}

    # Populate with clean and noisy results
    clean_data = func_data[func_data['noise_level'] == 0.0]
    for _, row in clean_data.iterrows():
        if row['method'] in method_perf:
            method_perf[row['method']]['clean_rmse'] = row['rmse']

    high_noise_data = func_data[func_data['noise_level'] == 0.1]
    for _, row in high_noise_data.iterrows():
        if row['method'] in method_perf:
            method_perf[row['method']]['noisy_rmse'] = row['rmse']

    # Build final robustness data list
    robustness_data = []
    for method, data in method_perf.items():
        if 'clean_rmse' in data and 'noisy_rmse' in data and data['clean_rmse'] > 0:
            ratio = data['noisy_rmse'] / data['clean_rmse']
            robustness_data.append({
                'method': method,
                'kernel_type': data['kernel_type'],
                'clean_rmse': data['clean_rmse'],
                'noisy_rmse': data['noisy_rmse'],
                'robustness_ratio': ratio
            })
    
    if not robustness_data:
        print("\nCould not compute robustness analysis; missing clean or high-noise results.")
        robustness_df = pd.DataFrame()
    else:
        robustness_df = pd.DataFrame(robustness_data).sort_values('robustness_ratio')
    
    print("\n🏆 MOST ROBUST GP KERNELS:")
    for _, row in robustness_df.iterrows():
        method = row['method']
        kernel = row['kernel_type']
        ratio = row['robustness_ratio']
        clean = row['clean_rmse']
        noisy = row['noisy_rmse']
        print(f"  {method:15s} ({kernel:12s}): {ratio:6.1f}x ({clean:.2e} → {noisy:.2e})")
    
    # Best kernel for each derivative order
    print(f"\\n🎖️  BEST GP KERNEL BY DERIVATIVE ORDER:")
    for deriv_order in sorted(successful_df['derivative_order'].unique()):
        deriv_data = successful_df[successful_df['derivative_order'] == deriv_order]
        
        # Average across all noise levels
        method_performance = deriv_data.groupby('method')['rmse'].mean().sort_values()
        
        if len(method_performance) > 0:
            best_method = method_performance.index[0]
            best_rmse = method_performance.iloc[0]
            kernel_type = deriv_data[deriv_data['method'] == best_method]['kernel_type'].iloc[0]
            
            print(f"  Order {deriv_order}: {best_method:15s} ({kernel_type:12s}) - RMSE = {best_rmse:.2e}")
    
    # Matérn smoothness comparison
    print(f"\\n🔬 MATÉRN SMOOTHNESS PARAMETER COMPARISON:")
    matern_methods = successful_df[successful_df['method'].str.contains('Matern')]
    
    if len(matern_methods) > 0:
        matern_func = matern_methods[matern_methods['derivative_order'] == 0]
        matern_performance = matern_func.groupby('method')['rmse'].mean().sort_values()
        
        print("  Function approximation (average across noise levels):")
        for method, rmse in matern_performance.items():
            nu_value = method.split('_')[-1]
            print(f"    Matérn ν={nu_value:3s}: RMSE = {rmse:.2e}")

if __name__ == "__main__":
    output_file = run_gp_comparison()
    
    print(f"\\n🎯 GP KERNEL COMPARISON COMPLETE!")
    print(f"This should show whether RBF (isotropic) beats Matérn")
    print(f"and which Matérn smoothness parameter works best")
    print(f"Results: {output_file}")