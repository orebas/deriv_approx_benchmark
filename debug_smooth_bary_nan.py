#!/usr/bin/env python3
"""Debug why smooth_barycentric_eval produces NaN values"""

import numpy as np
import jax.numpy as jnp
from comprehensive_methods_library import smooth_barycentric_eval, AAA_SmoothBarycentric_Approximator
from scipy.interpolate import AAA

def test_nan_locations():
    """Find where NaN values occur in AAA evaluation"""
    
    # Create test data
    t = np.linspace(0, 2*np.pi, 50)
    y = np.sin(t) + 0.01 * np.random.randn(len(t))
    
    # Get AAA parameters
    aaa_obj = AAA(t, y, max_terms=15)
    zj = jnp.array(aaa_obj.support_points)
    fj = jnp.array(aaa_obj.support_values)
    wj = jnp.array(aaa_obj.weights)
    
    print("AAA Support points and evaluation:")
    print("="*60)
    print(f"Number of support points: {len(zj)}")
    print(f"Support points (zj): {zj[:5]}... (showing first 5)")
    print(f"Weights (wj): {wj[:5]}... (showing first 5)")
    
    # Test evaluation at different points
    print("\nTesting evaluation at key points:")
    
    # 1. At support points
    print("\n1. At support points (should be exact):")
    for i in range(min(3, len(zj))):
        x = zj[i]
        result = smooth_barycentric_eval(x, zj, fj, wj)
        print(f"   x = {x:.6f} (support point {i}): result = {result}")
    
    # 2. Near support points
    print("\n2. Near support points:")
    for i in range(min(3, len(zj))):
        x = float(zj[i]) + 1e-10  # Very close but not exact
        result = smooth_barycentric_eval(x, zj, fj, wj)
        print(f"   x = {x:.6f} (near support {i}): result = {result}")
    
    # 3. At original data points
    print("\n3. At original data points:")
    nan_count = 0
    for i in range(len(t)):
        result = smooth_barycentric_eval(t[i], zj, fj, wj)
        if np.isnan(result):
            nan_count += 1
            if nan_count <= 5:  # Show first 5 NaN occurrences
                print(f"   x = {t[i]:.6f}: NaN")
                # Check if it's a support point
                min_dist = np.min(np.abs(t[i] - zj))
                print(f"      Min distance to support point: {min_dist}")
    
    print(f"\nTotal NaN values: {nan_count} out of {len(t)}")
    
    # 4. Check if NaN occurs exactly at support points
    print("\n4. Checking if NaN occurs at exact support points in t:")
    for i, ti in enumerate(t):
        for j, zj_val in enumerate(zj):
            if np.abs(ti - zj_val) < 1e-14:  # Machine epsilon comparison
                result = smooth_barycentric_eval(ti, zj, fj, wj)
                print(f"   t[{i}] = {ti} matches zj[{j}] = {zj_val}, result = {result}")

if __name__ == "__main__":
    test_nan_locations()