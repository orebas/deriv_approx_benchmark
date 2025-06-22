#!/usr/bin/env python3
"""Test the final AAA fix with rational approximation"""

import numpy as np
from comprehensive_methods_library import AAA_SmoothBarycentric_Approximator

def test_final_fix():
    # Generate test data
    t = np.linspace(0, 2*np.pi, 50)
    y = np.sin(t) + 0.01 * np.random.randn(len(t))
    
    print("Testing AAA_SmoothBarycentric with rational approximation fix:")
    print("="*70)
    
    # Create and fit the method
    method = AAA_SmoothBarycentric_Approximator(t, y, "AAA_SmoothBary_Final")
    method.fit()
    
    print(f"Fit successful: {method.fitted}")
    print(f"Success flag: {method.success}")
    
    # Evaluate derivatives at all points including support points
    results = method.evaluate(t, max_derivative=4)
    
    print(f"\nResults for evaluation at {len(t)} points:")
    
    # Check each derivative order
    all_good = True
    for order in range(5):
        key = 'y' if order == 0 else f'd{order}'
        values = results.get(key)
        
        if values is not None:
            nan_count = np.sum(np.isnan(values))
            print(f"\nDerivative order {order}:")
            print(f"  NaN count: {nan_count} / {len(values)}")
            
            if nan_count > 0:
                all_good = False
                print(f"  ❌ Still has NaN values!")
                # Find which points have NaN
                nan_indices = np.where(np.isnan(values))[0]
                print(f"  NaN at indices: {nan_indices[:5].tolist()}...")
                if hasattr(method, 'zj') and method.zj is not None:
                    # Check if these are support points
                    for idx in nan_indices[:3]:
                        t_val = t[idx]
                        min_dist = np.min(np.abs(t_val - np.array(method.zj)))
                        print(f"    t[{idx}] = {t_val:.6f}, min dist to support: {min_dist:.2e}")
            else:
                print(f"  ✅ No NaN values!")
                # Show statistics
                print(f"  Min: {np.min(values):.6f}")
                print(f"  Max: {np.max(values):.6f}")
                print(f"  Mean: {np.mean(values):.6f}")
                print(f"  Std: {np.std(values):.6f}")
    
    # Final verdict
    print("\n" + "="*70)
    if all_good:
        print("✅ SUCCESS! All derivatives are finite - no NaN values!")
        
        # Test error calculation
        y_true = np.sin(t)
        y_pred = results['y']
        errors = y_pred - y_true
        rmse = np.sqrt(np.mean(errors**2))
        print(f"\nFunction approximation RMSE: {rmse:.6f}")
        
        d1_true = np.cos(t)
        d1_pred = results['d1']
        d1_errors = d1_pred - d1_true
        d1_rmse = np.sqrt(np.mean(d1_errors**2))
        print(f"First derivative RMSE: {d1_rmse:.6f}")
        
    else:
        print("❌ FAILED! Still getting NaN values in derivatives.")

if __name__ == "__main__":
    test_final_fix()