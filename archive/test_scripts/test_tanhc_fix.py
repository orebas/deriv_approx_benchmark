#!/usr/bin/env python3
"""Test the tanhc algebraic reformulation fix"""

import numpy as np
import jax
import jax.numpy as jnp
from comprehensive_methods_library import AAA_SmoothBarycentric_Approximator, smooth_barycentric_eval
from scipy.interpolate import AAA

def test_tanhc_fix():
    print("Testing tanhc algebraic reformulation fix:")
    print("="*70)
    
    # 1. Test derivatives at problematic points
    print("\n1. Testing derivatives at support points:")
    
    # Create simple test case
    zj = jnp.array([0.0, 1.0, 2.0])
    fj = jnp.array([0.0, 1.0, 4.0])  # f(x) = x²
    wj = jnp.array([1.0, -1.0, 1.0])  # Proper barycentric weights for x²
    
    # Create derivative functions
    f = lambda x: smooth_barycentric_eval(x, zj, fj, wj)
    f1 = jax.grad(f)
    f2 = jax.grad(f1)
    
    test_points = [
        (0.0, "At support point 0"),
        (1.0, "At support point 1"), 
        (2.0, "At support point 2"),
        (0.5, "Between supports"),
        (1.5, "Between supports"),
    ]
    
    for x, desc in test_points:
        print(f"\n  {desc}: x = {x}")
        v0 = float(f(x))
        v1 = float(f1(x))
        v2 = float(f2(x))
        
        print(f"    f(x)   = {v0:.6f} (true: {x**2:.6f})")
        print(f"    f'(x)  = {v1:.6f} (true: {2*x:.6f})")
        print(f"    f''(x) = {v2:.6f} (true: 2.0)")
        
        if np.isnan(v1) or np.isnan(v2):
            print("    ❌ NaN in derivatives!")
        else:
            print("    ✅ No NaN!")
    
    # 2. Test with actual AAA method
    print("\n" + "="*70)
    print("2. Testing with AAA_SmoothBarycentric method:")
    
    # Generate noisy sine data
    t = np.linspace(0, 2*np.pi, 50)
    y = np.sin(t) + 0.01 * np.random.randn(len(t))
    
    method = AAA_SmoothBarycentric_Approximator(t, y, "AAA_TanhC_Fix")
    method.fit()
    
    print(f"  Fit successful: {method.fitted}")
    print(f"  Success flag: {method.success}")
    
    if method.success:
        # Check for NaN in derivatives
        results = method.evaluate(t, max_derivative=4)
        
        all_clean = True
        for order in range(5):
            key = 'y' if order == 0 else f'd{order}'
            values = results.get(key)
            
            if values is not None:
                nan_count = np.sum(np.isnan(values))
                print(f"  Derivative order {order}: {nan_count} NaN out of {len(values)}")
                if nan_count > 0:
                    all_clean = False
        
        if all_clean:
            print("  ✅ All derivatives are finite!")
        else:
            print("  ❌ Still getting NaN values!")
    
    # 3. Accuracy test with proper AAA weights
    print("\n" + "="*70)
    print("3. Accuracy test with proper AAA weights:")
    
    # Test on a known function
    x_test = np.linspace(0, 2*np.pi, 20)
    y_test = np.sin(x_test)
    
    # Get AAA approximation
    aaa = AAA(x_test, y_test, max_terms=15)
    zj_aaa = jnp.array(aaa.support_points)
    fj_aaa = jnp.array(aaa.support_values)
    wj_aaa = jnp.array(aaa.weights)
    
    print(f"  AAA found {len(zj_aaa)} support points")
    
    # Test at various points
    x_eval = np.linspace(0, 2*np.pi, 100)
    
    # Evaluate with our smooth function
    smooth_vals = np.array([float(smooth_barycentric_eval(x, zj_aaa, fj_aaa, wj_aaa)) 
                           for x in x_eval])
    
    # Compare with true values
    true_vals = np.sin(x_eval)
    errors = np.abs(smooth_vals - true_vals)
    
    print(f"  Mean absolute error: {np.mean(errors):.2e}")
    print(f"  Max absolute error: {np.max(errors):.2e}")
    print(f"  RMSE: {np.sqrt(np.mean(errors**2)):.2e}")
    
    # Compare with direct AAA evaluation
    aaa_vals = aaa(x_eval)
    aaa_errors = np.abs(aaa_vals - true_vals)
    
    print(f"  Direct AAA RMSE: {np.sqrt(np.mean(aaa_errors**2)):.2e}")
    print(f"  Ratio (smooth/direct): {np.sqrt(np.mean(errors**2))/np.sqrt(np.mean(aaa_errors**2)):.3f}")

if __name__ == "__main__":
    test_tanhc_fix()