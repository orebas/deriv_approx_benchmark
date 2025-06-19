#!/usr/bin/env python3
"""
Test Gemini's v3 smooth barycentric evaluation - the definitive version!
Fixes: proper W scale, gamma factor for "all far" case
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

def naive_barycentric_eval(z, x, f, w):
    """Standard barycentric evaluation - can fail when z ≈ x[j]"""
    diffs = z - x
    weights = w / diffs
    num = jnp.sum(weights * f)
    den = jnp.sum(weights)
    return num / den

def smooth_barycentric_eval_v3(z, x, f, w, W=0.1):
    """Gemini's v3 smooth barycentric evaluation - robust for all cases"""
    d = z - x
    d_sq = d**2
    alpha = jnp.tanh(d_sq / W)  # alpha=1 is far, alpha=0 is close
    
    # --- Shared components ---
    safe_far_term = jnp.nan_to_num(alpha / d, nan=0.0)
    N_far_unscaled = jnp.sum(safe_far_term * w * f)
    D_far_unscaled = jnp.sum(safe_far_term * w)
    
    one_minus_alpha = 1.0 - alpha
    N_close = jnp.sum(one_minus_alpha * w * f)
    D_close = jnp.sum(one_minus_alpha * w)
    
    d_scale = jnp.sum(one_minus_alpha * d)
    
    # --- New "farness" factor ---
    gamma = jnp.prod(alpha)
    
    # --- Final Assembly (v3) ---
    # The gamma terms restore the naive formula when all points are far
    N_final = N_close + d_scale * N_far_unscaled + gamma * N_far_unscaled
    D_final = D_close + d_scale * D_far_unscaled + gamma * D_far_unscaled
    
    # Add small epsilon for absolute safety
    return N_final / (D_final + 1e-30)

def test_numerical_equivalence_v3():
    """Test v3 numerical equivalence across different scales"""
    print("Testing numerical equivalence (v3)...")
    
    x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    f = jnp.array([0.0, 1.0, 4.0, 9.0, 16.0])  # x^2
    w = jnp.ones(5)
    
    # Test both near and far points
    z_test = jnp.array([0.5, 1.5, 2.5, 3.5, -0.5, 4.5, -2.0, 6.0])  # Added far points
    
    print("Support points:", x)
    print("Function values:", f)
    print()
    
    for z in z_test:
        try:
            naive_val = naive_barycentric_eval(z, x, f, w)
            smooth_val = smooth_barycentric_eval_v3(z, x, f, w)
            diff = abs(naive_val - smooth_val)
            
            print(f"z = {z:5.1f}: naive = {naive_val:12.8f}, smooth = {smooth_val:12.8f}, diff = {diff:.2e}")
            
            if diff > 1e-8:  # Allow some tolerance for approximation
                if diff > 1e-4:
                    print(f"  ⚠️  Large difference!")
                else:
                    print(f"  ⚠️  Small approximation error (okay)")
            else:
                print(f"  ✅ Excellent agreement")
                
        except Exception as e:
            smooth_val = smooth_barycentric_eval_v3(z, x, f, w)
            print(f"z = {z:5.1f}: naive failed ({e}), smooth = {smooth_val:12.8f}")
            print(f"  ✅ Smooth method handled problematic case")
    
    print()

def test_derivatives_comprehensive():
    """Test derivatives across all regions"""
    print("Testing derivatives comprehensively (v3)...")
    
    x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    f = jnp.array([0.0, 1.0, 4.0, 9.0, 16.0])
    w = jnp.ones(5)
    
    def eval_at_z(z):
        return smooth_barycentric_eval_v3(z, x, f, w)
    
    # Test points in different regions
    z_test = jnp.array([
        0.5,            # Between points
        1.0 + 1e-8,     # Very close to support point
        1.5,            # Between points
        2.0 + 1e-8,     # Very close to support point
        -1.0,           # Far outside range
        5.0,            # Far outside range
        0.1,            # Near first point
        3.9             # Near last point
    ])
    
    print("Testing first and second derivatives:")
    all_finite = True
    
    for z in z_test:
        try:
            val = eval_at_z(z)
            grad1 = jax.grad(eval_at_z)(z)
            grad2 = jax.grad(jax.grad(eval_at_z))(z)
            
            finite_val = jnp.isfinite(val)
            finite_grad1 = jnp.isfinite(grad1)
            finite_grad2 = jnp.isfinite(grad2)
            
            print(f"z = {z:10.6f}: f = {val:12.8f}, f' = {grad1:12.8f}, f'' = {grad2:12.8f}")
            
            if finite_val and finite_grad1 and finite_grad2:
                print(f"  ✅ All derivatives finite")
            else:
                print(f"  ❌ Non-finite: val={finite_val}, grad1={finite_grad1}, grad2={finite_grad2}")
                all_finite = False
                
        except Exception as e:
            print(f"z = {z}: Derivative computation failed: {e}")
            all_finite = False
    
    if all_finite:
        print("\n🎉 ALL DERIVATIVES ARE FINITE IN ALL REGIONS!")
    else:
        print("\n❌ Some derivatives failed")
    
    print()

def test_limit_behavior():
    """Test approaching support points"""
    print("Testing limit behavior (v3)...")
    
    x = jnp.array([0.0, 1.0, 2.0])
    f = jnp.array([1.0, 5.0, 9.0])  # Clear distinct values
    w = jnp.array([1.0, 1.0, 1.0])
    
    # Test approaching x[1] = 1.0, f[1] = 5.0
    distances = [1e-1, 1e-3, 1e-6, 1e-9, 1e-12]
    
    print("Approaching x[1] = 1.0 (f[1] = 5.0)")
    for dist in distances:
        z = 1.0 + dist
        
        smooth_val = smooth_barycentric_eval_v3(z, x, f, w)
        
        # Test derivative
        def eval_at_z(z_val):
            return smooth_barycentric_eval_v3(z_val, x, f, w)
        
        try:
            grad = jax.grad(eval_at_z)(z)
            grad_finite = jnp.isfinite(grad)
        except:
            grad = float('nan')
            grad_finite = False
        
        error_from_limit = abs(smooth_val - 5.0)
        print(f"dist = {dist:.0e}: f = {smooth_val:12.8f}, f' = {grad:12.8f}, error = {error_from_limit:.2e}, finite = {grad_finite}")
        
        if error_from_limit < max(1e-6, dist * 10):
            print(f"  ✅ Good convergence to f[1] = 5.0")
        else:
            print(f"  ⚠️  Slow convergence")
    
    print()

def test_edge_cases():
    """Test various edge cases"""
    print("Testing edge cases (v3)...")
    
    # Case 1: Very far evaluation point
    print("Case 1: Very far evaluation point")
    x = jnp.array([0.0, 1.0])
    f = jnp.array([0.0, 1.0])
    w = jnp.array([1.0, 1.0])
    z = 100.0  # Very far
    
    result = smooth_barycentric_eval_v3(z, x, f, w)
    print(f"z = 100.0: result = {result}")
    
    # Compare with naive (should work for far points)
    naive_result = naive_barycentric_eval(z, x, f, w)
    print(f"Naive result: {naive_result}")
    print(f"Difference: {abs(result - naive_result):.2e}")
    print()
    
    # Case 2: Clustered support points
    print("Case 2: Clustered support points")
    x = jnp.array([1.0, 1.001, 1.002])
    f = jnp.array([2.0, 3.0, 4.0])
    w = jnp.array([1.0, 1.0, 1.0])
    z = 1.0005  # In the middle of cluster
    
    result = smooth_barycentric_eval_v3(z, x, f, w)
    
    def eval_at_z(z_val):
        return smooth_barycentric_eval_v3(z_val, x, f, w)
    
    grad = jax.grad(eval_at_z)(z)
    
    print(f"Clustered test: f = {result}, f' = {grad}")
    print(f"Gradient finite: {jnp.isfinite(grad)}")
    print()

if __name__ == "__main__":
    print("="*80)
    print("TESTING GEMINI'S V3 SMOOTH BARYCENTRIC EVALUATION - DEFINITIVE VERSION!")
    print("="*80)
    print()
    
    test_numerical_equivalence_v3()
    test_derivatives_comprehensive()
    test_limit_behavior()
    test_edge_cases()
    
    print("="*80)
    print("V3 TESTING COMPLETE")
    print("="*80)