#!/usr/bin/env python3
"""Debug why derivatives produce NaN values"""

import numpy as np
import jax
import jax.numpy as jnp
from comprehensive_methods_library import smooth_barycentric_eval, AAA_SmoothBarycentric_Approximator
from scipy.interpolate import AAA

def test_derivative_nans():
    """Test where derivative NaN values occur"""
    
    # Create test data
    t = np.linspace(0, 2*np.pi, 50)
    y = np.sin(t) + 0.01 * np.random.randn(len(t))
    
    # Create AAA method
    method = AAA_SmoothBarycentric_Approximator(t, y)
    method.fit()
    
    print("Testing derivative evaluations:")
    print("="*60)
    
    # Get the derivative functions
    print(f"Number of derivative functions: {len(method.ad_derivatives)}")
    
    # Test derivative at different points
    test_points = [t[0], t[1], t[len(t)//2], t[-1]]
    
    for i, x in enumerate(test_points):
        print(f"\nTest point {i+1}: x = {x:.6f}")
        
        # Try each derivative order
        for order in range(min(5, len(method.ad_derivatives))):
            try:
                deriv_func = method.ad_derivatives[order]
                result = float(deriv_func(x))
                print(f"  Order {order}: {result:.6f}")
            except Exception as e:
                print(f"  Order {order}: Error - {type(e).__name__}: {str(e)}")
    
    # Now test on the full array to see where NaN occurs
    print("\nTesting full array evaluation:")
    print("="*60)
    
    for order in range(5):
        print(f"\nDerivative order {order}:")
        
        if order == 0:
            values = method._evaluate_function(t)
        else:
            values = method._evaluate_derivative(t, order)
        
        nan_mask = np.isnan(values)
        nan_indices = np.where(nan_mask)[0]
        
        if len(nan_indices) > 0:
            print(f"  NaN at {len(nan_indices)} positions")
            print(f"  First 5 NaN indices: {nan_indices[:5].tolist()}")
            print(f"  Corresponding t values: {t[nan_indices[:5]].tolist()}")
            
            # Check if these correspond to support points
            if hasattr(method, 'zj') and method.zj is not None:
                for idx in nan_indices[:3]:
                    t_val = t[idx]
                    min_dist = np.min(np.abs(t_val - np.array(method.zj)))
                    print(f"    t[{idx}] = {t_val:.6f}, min dist to support: {min_dist:.2e}")
        else:
            print(f"  No NaN values found")
    
    # Check if the issue is with vmap
    print("\nTesting vmap vs individual evaluation:")
    print("="*60)
    
    # Test a specific point that might be problematic
    test_idx = 0  # First point often has issues
    x_test = t[test_idx]
    
    print(f"Testing at t[{test_idx}] = {x_test:.6f}")
    
    # Individual evaluation
    for order in range(3):
        individual_result = float(method.ad_derivatives[order](x_test))
        print(f"  Individual eval, order {order}: {individual_result}")
    
    # Vmap evaluation
    for order in range(3):
        vmap_func = jax.vmap(method.ad_derivatives[order])
        vmap_result = vmap_func(jnp.array([x_test]))[0]
        print(f"  Vmap eval, order {order}: {vmap_result}")

if __name__ == "__main__":
    test_derivative_nans()