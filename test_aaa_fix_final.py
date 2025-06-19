#!/usr/bin/env python3
"""Test that the AAA fix works correctly"""

import numpy as np
from comprehensive_methods_library import AAA_SmoothBarycentric_Approximator

def test_aaa_fix():
    # Generate test data
    t = np.linspace(0, 2*np.pi, 50)
    y = np.sin(t) + 0.01 * np.random.randn(len(t))
    
    print("Testing AAA_SmoothBarycentric with fixed smooth_barycentric_eval:")
    print("="*70)
    
    # Create and fit the method
    method = AAA_SmoothBarycentric_Approximator(t, y, "AAA_SmoothBary_Fixed")
    method.fit()
    
    print(f"Fit successful: {method.fitted}")
    print(f"Success flag: {method.success}")
    
    # Evaluate derivatives
    results = method.evaluate(t, max_derivative=4)
    
    print(f"\nResults for evaluation at {len(t)} points:")
    
    for order in range(5):
        key = 'y' if order == 0 else f'd{order}'
        values = results.get(key)
        
        if values is not None:
            nan_count = np.sum(np.isnan(values))
            print(f"\nDerivative order {order}:")
            print(f"  NaN count: {nan_count} / {len(values)}")
            
            if nan_count == 0:
                print(f"  ✅ No NaN values!")
                print(f"  Min: {np.min(values):.6f}")
                print(f"  Max: {np.max(values):.6f}")
                print(f"  Mean: {np.mean(values):.6f}")
            else:
                valid_values = values[~np.isnan(values)]
                if len(valid_values) > 0:
                    print(f"  Valid values: {len(valid_values)}")
                    print(f"  Min (valid): {np.min(valid_values):.6f}")
                    print(f"  Max (valid): {np.max(valid_values):.6f}")
    
    # Check error metrics
    print("\n" + "="*70)
    print("Checking error calculation with fixed derivatives:")
    
    # Simulate what happens in the benchmark
    y_true = np.sin(t)  # True function values
    y_pred = results['y']
    
    errors = y_pred - y_true
    rmse = np.sqrt(np.mean(errors**2))
    
    print(f"\nFunction approximation error:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  RMSE is NaN: {np.isnan(rmse)}")
    
    # Check first derivative
    d1_true = np.cos(t)
    d1_pred = results['d1']
    
    d1_errors = d1_pred - d1_true
    d1_rmse = np.sqrt(np.mean(d1_errors**2))
    
    print(f"\nFirst derivative error:")
    print(f"  RMSE: {d1_rmse:.6f}")
    print(f"  RMSE is NaN: {np.isnan(d1_rmse)}")
    
    if np.isnan(d1_rmse):
        print("  ❌ Still getting NaN in error metrics!")
    else:
        print("  ✅ Error metrics are valid!")

if __name__ == "__main__":
    test_aaa_fix()