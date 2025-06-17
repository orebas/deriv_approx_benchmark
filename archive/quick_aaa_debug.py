#!/usr/bin/env python3
"""
Quick debug script for AAA methods - tests a few specific cases first
"""

import numpy as np
import pandas as pd
import time
import traceback
import warnings
warnings.filterwarnings('ignore')

from comprehensive_methods_library import AAALeastSquaresApproximator, AAA_FullOpt_Approximator

def test_single_case(method_class, method_name, t, y_noisy, y_truth, case_name):
    """Test a single method on one case"""
    print(f"Testing {method_name} on {case_name}...")
    
    try:
        start_time = time.time()
        method = method_class(t, y_noisy, method_name)
        method.fit()
        fit_time = time.time() - start_time
        
        print(f"  Fit time: {fit_time:.3f}s")
        print(f"  Success: {method.success}")
        
        if method.success:
            # Test function evaluation
            y_pred = method.evaluate_function(t)
            if not np.all(np.isnan(y_pred)):
                function_rmse = np.sqrt(np.mean((y_pred - y_truth)**2))
                print(f"  Function RMSE: {function_rmse:.6f}")
                
                # Test derivatives
                for deriv_order in [1, 2, 3]:
                    try:
                        deriv_pred = method.evaluate_derivative(t, deriv_order)
                        if not np.all(np.isnan(deriv_pred)):
                            print(f"  Derivative {deriv_order}: ✓")
                        else:
                            print(f"  Derivative {deriv_order}: NaN")
                    except Exception as e:
                        print(f"  Derivative {deriv_order}: Error - {e}")
                        
                return True, function_rmse, fit_time
            else:
                print(f"  Function evaluation returned NaN")
                return False, np.nan, fit_time
        else:
            print(f"  Method reported failure")
            return False, np.nan, fit_time
            
    except Exception as e:
        print(f"  Exception: {e}")
        traceback.print_exc()
        return False, np.nan, 0.0

def load_test_case(case_path):
    """Load a specific test case"""
    truth_file = f"test_data/{case_path}/truth_data.csv"
    noisy_file = f"test_data/{case_path}/noisy_data.csv"
    
    try:
        truth_df = pd.read_csv(truth_file)
        noisy_df = pd.read_csv(noisy_file)
        
        t = truth_df['t'].values
        
        # Get first variable (x1(t))
        var_name = 'x1(t)'
        y_truth = truth_df[var_name].values
        y_noisy = noisy_df[var_name].values
        
        return t, y_truth, y_noisy, var_name
        
    except Exception as e:
        print(f"Error loading {case_path}: {e}")
        return None, None, None, None

def main():
    print("=== Quick AAA Debug Test ===")
    print()
    
    # Test a few specific cases to identify patterns
    test_cases = [
        "lv_periodic/noise_0.0",      # Clean data
        "lv_periodic/noise_1.0e-8",   # Very low noise
        "lv_periodic/noise_1.0e-5",   # Low noise
        "lv_periodic/noise_0.001",    # Medium noise
        "lv_periodic/noise_0.01",     # Higher noise
    ]
    
    methods = [
        (AAALeastSquaresApproximator, "AAA_LS"),
        (AAA_FullOpt_Approximator, "AAA_FullOpt")
    ]
    
    results_summary = []
    
    for case_path in test_cases:
        print(f"\n--- Testing case: {case_path} ---")
        
        t, y_truth, y_noisy, var_name = load_test_case(case_path)
        if t is None:
            continue
            
        print(f"Data loaded: {len(t)} points, variable: {var_name}")
        print(f"Data range: [{y_truth.min():.3f}, {y_truth.max():.3f}]")
        
        noise_level = np.std(y_noisy - y_truth) if not np.allclose(y_noisy, y_truth) else 0.0
        print(f"Estimated noise std: {noise_level:.8f}")
        
        for method_class, method_name in methods:
            success, rmse, fit_time = test_single_case(
                method_class, method_name, t, y_noisy, y_truth, case_path
            )
            
            results_summary.append({
                'case': case_path,
                'method': method_name,
                'success': success,
                'rmse': rmse,
                'fit_time': fit_time,
                'noise_std': noise_level
            })
    
    # Summary analysis
    print("\n=== Summary ===")
    results_df = pd.DataFrame(results_summary)
    
    print("\nSuccess rates by method:")
    success_rates = results_df.groupby('method')['success'].agg(['count', 'sum'])
    success_rates['success_rate'] = success_rates['sum'] / success_rates['count']
    print(success_rates)
    
    print("\nFailed cases:")
    failed = results_df[~results_df['success']]
    for _, row in failed.iterrows():
        print(f"  {row['method']} failed on {row['case']}")
    
    print("\nSuccessful cases RMSE:")
    successful = results_df[results_df['success']]
    if len(successful) > 0:
        rmse_stats = successful.groupby('method')['rmse'].agg(['mean', 'std', 'min', 'max'])
        rmse_stats = rmse_stats.round(6)
        print(rmse_stats)
    
    # Save results
    results_df.to_csv('quick_aaa_debug_results.csv', index=False)
    print("\nResults saved to quick_aaa_debug_results.csv")

if __name__ == "__main__":
    main()