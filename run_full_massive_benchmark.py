#!/usr/bin/env python3
"""
Full Massive Benchmark - Comprehensive test of all methods
"""

import pandas as pd
import numpy as np
import glob
import time
from datetime import datetime
from comprehensive_methods_library import create_all_methods, get_method_categories
from scipy.interpolate import CubicSpline

def run_full_benchmark():
    """Run comprehensive benchmark across multiple datasets and conditions"""
    
    print("🚀 FULL MASSIVE DERIVATIVE APPROXIMATION BENCHMARK")
    print("=" * 70)
    
    # Load Julia data files
    julia_files = glob.glob("results/sweep_lv_periodic_n*.csv")[:5]  # Limit for runtime
    print(f"Using {len(julia_files)} data files")
    
    # Extract test cases
    test_cases = []
    for julia_file in julia_files:
        df = pd.read_csv(julia_file)
        
        # Get unique parameter combinations
        params = df[['noise_level', 'data_size', 'observable']].drop_duplicates()
        
        for _, param_row in params.head(2).iterrows():  # 2 cases per file
            # Get data for this parameter combination
            subset = df[
                (df['noise_level'] == param_row['noise_level']) &
                (df['data_size'] == param_row['data_size']) &
                (df['observable'] == param_row['observable']) &
                (df['derivative_order'] == 0)  # Just function values for original data
            ]
            
            if len(subset) >= 20:  # Minimum data points
                t = subset['time'].values
                y = subset['true_value'].values
                
                # Sort and clean
                sort_idx = np.argsort(t)
                t = t[sort_idx]
                y = y[sort_idx]
                
                # Remove duplicates
                unique_idx = np.unique(t, return_index=True)[1]
                t = t[unique_idx]
                y = y[unique_idx]
                
                if len(t) >= 15:
                    test_cases.append({
                        'noise_level': param_row['noise_level'],
                        'data_size': param_row['data_size'],
                        'observable': param_row['observable'],
                        't': t,
                        'y': y,
                        'source': julia_file
                    })
    
    print(f"Extracted {len(test_cases)} test cases")
    
    # Initialize results
    all_results = []
    
    # Get method info
    if test_cases:
        methods = create_all_methods(test_cases[0]['t'], test_cases[0]['y'])
        categories = get_method_categories()
        
        print(f"\\nTesting {len(methods)} methods across {len(test_cases)} datasets")
        print("Categories:", list(categories.keys()))
        
        total_combinations = len(test_cases) * len(methods) * 4  # 4 derivative orders
        combo_count = 0
        
        # Run benchmark
        for case_idx, test_case in enumerate(test_cases):
            print(f"\\nDataset {case_idx+1}/{len(test_cases)}: noise={test_case['noise_level']:.1e}, size={len(test_case['t'])}")
            
            t = test_case['t']
            y = test_case['y']
            
            # Create reference spline for derivative truth
            ref_spline = CubicSpline(t, y)
            
            # Test each method
            methods = create_all_methods(t, y)
            
            for method_name, method in methods.items():
                
                try:
                    # Fit and evaluate
                    start_time = time.time()
                    results = method.evaluate(t, max_derivative=3)
                    eval_time = time.time() - start_time
                    
                    # Get method category
                    category = 'Unknown'
                    for cat, method_list in categories.items():
                        if method_name in method_list:
                            category = cat
                            break
                    
                    if results['success']:
                        # Test each derivative order
                        for deriv_order in range(4):  # 0, 1, 2, 3
                            combo_count += 1
                            
                            if combo_count % 50 == 0:
                                progress = 100 * combo_count / total_combinations
                                print(f"    Progress: {combo_count}/{total_combinations} ({progress:.1f}%)")
                            
                            # Get predictions and truth
                            if deriv_order == 0:
                                y_pred = results['y']
                                y_true = y
                            else:
                                y_pred = results[f'd{deriv_order}']
                                y_true = ref_spline.derivative(deriv_order)(t)
                            
                            # Calculate errors
                            errors = y_pred - y_true
                            rmse = np.sqrt(np.mean(errors**2))
                            mae = np.mean(np.abs(errors))
                            max_error = np.max(np.abs(errors))
                            
                            # Store result
                            all_results.append({
                                'method': method_name,
                                'category': category,
                                'noise_level': test_case['noise_level'],
                                'data_size': len(t),
                                'observable': test_case['observable'],
                                'derivative_order': deriv_order,
                                'rmse': rmse,
                                'mae': mae,
                                'max_error': max_error,
                                'eval_time': eval_time,
                                'fit_time': method.fit_time,
                                'success': True,
                                'case_id': case_idx
                            })
                    else:
                        # Failed - add NaN entries
                        for deriv_order in range(4):
                            combo_count += 1
                            all_results.append({
                                'method': method_name,
                                'category': category,
                                'noise_level': test_case['noise_level'],
                                'data_size': len(t),
                                'observable': test_case['observable'],
                                'derivative_order': deriv_order,
                                'rmse': np.nan,
                                'mae': np.nan,
                                'max_error': np.nan,
                                'eval_time': eval_time,
                                'fit_time': method.fit_time,
                                'success': False,
                                'error': results.get('error', 'Unknown'),
                                'case_id': case_idx
                            })
                
                except Exception as e:
                    # Exception - add failure entries
                    category = 'Unknown'
                    for cat, method_list in categories.items():
                        if method_name in method_list:
                            category = cat
                            break
                    
                    for deriv_order in range(4):
                        combo_count += 1
                        all_results.append({
                            'method': method_name,
                            'category': category,
                            'noise_level': test_case['noise_level'],
                            'data_size': len(t),
                            'observable': test_case['observable'],
                            'derivative_order': deriv_order,
                            'rmse': np.nan,
                            'mae': np.nan,
                            'max_error': np.nan,
                            'eval_time': np.nan,
                            'fit_time': np.nan,
                            'success': False,
                            'error': str(e)[:200],
                            'case_id': case_idx
                        })
    
    # Save results
    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"results/massive_benchmark_{timestamp}.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"\\n🎉 FULL MASSIVE BENCHMARK COMPLETE!")
    print(f"📁 Results saved to: {output_file}")
    print(f"📊 Total evaluations: {len(results_df)}")
    
    # Comprehensive summary
    print(f"\\n📋 COMPREHENSIVE SUMMARY:")
    print(f"Methods tested: {results_df['method'].nunique()}")
    print(f"Test cases: {results_df['case_id'].nunique()}")
    print(f"Successful evaluations: {results_df['success'].sum()}/{len(results_df)} ({100*results_df['success'].mean():.1f}%)")
    
    # Success by method
    print(f"\\n🏆 SUCCESS RATE BY METHOD:")
    method_success = results_df.groupby('method')['success'].agg(['sum', 'count', 'mean']).sort_values('mean', ascending=False)
    for method, stats in method_success.iterrows():
        print(f"  {method:15s}: {stats['mean']*100:5.1f}% ({stats['sum']:3d}/{stats['count']:3d})")
    
    # Performance by category
    print(f"\\n📊 PERFORMANCE BY CATEGORY:")
    category_stats = results_df.groupby('category').agg({
        'success': 'mean',
        'rmse': lambda x: np.nanmean(x),
        'eval_time': lambda x: np.nanmean(x)
    }).sort_values('success', ascending=False)
    
    for category, stats in category_stats.iterrows():
        print(f"  {category:15s}: {stats['success']*100:5.1f}% success, RMSE={stats['rmse']:.2e}, Time={stats['eval_time']:.3f}s")
    
    # Top performers overall
    successful_results = results_df[results_df['success'] & (results_df['derivative_order'] == 0)]
    if len(successful_results) > 0:
        print(f"\\n🥇 TOP PERFORMERS (Function approximation - derivative order 0):")
        top_performers = successful_results.groupby('method')['rmse'].mean().sort_values().head(10)
        for method, rmse in top_performers.items():
            print(f"  {method:15s}: RMSE = {rmse:.2e}")
    
    return output_file

if __name__ == "__main__":
    output_file = run_full_benchmark()
    
    print(f"\\n🎯 NEXT STEPS:")
    print(f"1. Create comprehensive report with all methods")
    print(f"2. Compare with original Julia methods") 
    print(f"3. Identify best methods for each use case")
    print(f"4. Results file: {output_file}")