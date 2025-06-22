#!/usr/bin/env python3
"""
Debug why barycentric interpolation has such high errors
"""

import numpy as np
import jax.numpy as jnp
from scipy.interpolate import AAA, BarycentricInterpolator

def naive_barycentric(x, zj, fj, wj):
    """Standard barycentric formula"""
    diffs = x - zj
    # Avoid exact zeros
    diffs = jnp.where(diffs == 0, 1e-15, diffs)
    weights = wj / diffs
    return jnp.sum(weights * fj) / jnp.sum(weights)

def test_barycentric_basics():
    """Test basic barycentric interpolation"""
    
    print("Testing barycentric interpolation basics")
    print("="*60)
    
    # Simple test: interpolate x²
    x_support = np.array([0.0, 1.0, 2.0, 3.0])
    y_support = x_support**2
    
    # Get proper barycentric weights from AAA
    print("\n1. Using AAA to get barycentric weights:")
    aaa = AAA(x_support, y_support, max_terms=len(x_support))
    
    print(f"   Support points: {aaa.support_points}")
    print(f"   Support values: {aaa.support_values}")
    print(f"   Weights: {aaa.weights}")
    
    # Test at various points
    test_points = [0.5, 1.5, 2.5]
    
    print("\n2. Testing interpolation accuracy:")
    print(f"{'x':>6} {'True':>10} {'AAA':>10} {'Error':>10}")
    print("-"*40)
    
    for x in test_points:
        true_val = x**2
        aaa_val = aaa(x)
        error = abs(aaa_val - true_val)
        print(f"{x:6.1f} {true_val:10.4f} {aaa_val:10.4f} {error:10.2e}")
    
    # Now test with uniform weights (wrong!)
    print("\n3. Testing with uniform weights (incorrect):")
    uniform_weights = np.ones(len(x_support))
    
    print(f"{'x':>6} {'True':>10} {'Uniform':>10} {'Error':>10}")
    print("-"*40)
    
    for x in test_points:
        true_val = x**2
        unif_val = float(naive_barycentric(x, x_support, y_support, uniform_weights))
        error = abs(unif_val - true_val)
        print(f"{x:6.1f} {true_val:10.4f} {unif_val:10.4f} {error:10.2e}")
    
    # Compare with scipy's BarycentricInterpolator
    print("\n4. Using scipy BarycentricInterpolator:")
    bary = BarycentricInterpolator(x_support, y_support)
    
    print(f"{'x':>6} {'True':>10} {'Scipy':>10} {'Error':>10}")
    print("-"*40)
    
    for x in test_points:
        true_val = x**2
        scipy_val = bary(x)
        error = abs(scipy_val - true_val)
        print(f"{x:6.1f} {true_val:10.4f} {scipy_val:10.4f} {error:10.2e}")
    
    # The issue: our smooth_barycentric_eval is using uniform weights!
    print("\n" + "="*60)
    print("INSIGHT: The high errors are because we're using uniform weights!")
    print("AAA should provide the proper barycentric weights, not ones.")
    print("="*60)

if __name__ == "__main__":
    test_barycentric_basics()