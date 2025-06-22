#!/usr/bin/env python3
"""
Comprehensive test of smooth barycentric derivatives at various points
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

# Original smooth barycentric eval (before my fix)
def smooth_barycentric_eval_original(x, zj, fj, wj, W=1e-7):
    """Original implementation with tanh transition"""
    d = x - zj
    d_sq = d**2
    alpha = jnp.tanh(d_sq / W)  # alpha=1 is far, alpha=0 is close
    
    # Handle potentially problematic alpha/d term
    safe_far_term = jnp.nan_to_num(alpha / d, nan=0.0)
    N_far_unscaled = jnp.sum(safe_far_term * wj * fj)
    D_far_unscaled = jnp.sum(safe_far_term * wj)
    
    # Close region contributions
    one_minus_alpha = 1.0 - alpha
    N_close = jnp.sum(one_minus_alpha * wj * fj)
    D_close = jnp.sum(one_minus_alpha * wj)
    
    # Smooth scaling factor (weighted average of distances)
    d_scale = jnp.sum(one_minus_alpha * d)
    
    # Farness factor to restore naive formula when all points are far
    gamma = jnp.prod(alpha)
    
    # Final assembly combining all terms
    N_final = N_close + d_scale * N_far_unscaled + gamma * N_far_unscaled
    D_final = D_close + d_scale * D_far_unscaled + gamma * D_far_unscaled
    
    return N_final / (D_final + 1e-30)

# Gemini's suggested fix using jnp.where
def smooth_barycentric_eval_fixed(x, zj, fj, wj, W=1e-7):
    """Fixed implementation using jnp.where for the problematic term"""
    d = x - zj
    d_sq = d**2
    alpha = jnp.tanh(d_sq / W)  # alpha=1 is far, alpha=0 is close
    
    # Use jnp.where to handle the singularity properly
    # Near d=0, use Taylor expansion: tanh(d²/W)/d ≈ d/W
    threshold = 1e-6
    safe_far_term = jnp.where(
        jnp.abs(d) < threshold,
        d / W,  # Taylor approximation near singularity
        alpha / d  # Original formula away from singularity
    )
    
    N_far_unscaled = jnp.sum(safe_far_term * wj * fj)
    D_far_unscaled = jnp.sum(safe_far_term * wj)
    
    # Close region contributions
    one_minus_alpha = 1.0 - alpha
    N_close = jnp.sum(one_minus_alpha * wj * fj)
    D_close = jnp.sum(one_minus_alpha * wj)
    
    # Smooth scaling factor (weighted average of distances)
    d_scale = jnp.sum(one_minus_alpha * d)
    
    # Farness factor to restore naive formula when all points are far
    gamma = jnp.prod(alpha)
    
    # Final assembly combining all terms
    N_final = N_close + d_scale * N_far_unscaled + gamma * N_far_unscaled
    D_final = D_close + d_scale * D_far_unscaled + gamma * D_far_unscaled
    
    return N_final / (D_final + 1e-30)

def test_derivatives():
    """Test derivatives at support points, near, and far"""
    
    # Simple test case
    zj = jnp.array([0.0, 1.0, 2.0, 3.0])
    fj = jnp.array([0.0, 1.0, 4.0, 9.0])  # f(x) = x²
    wj = jnp.ones(4)
    
    # Test points: at support, very close, and far
    test_points = [
        (0.0, "At support point 0"),
        (1e-8, "Very close to support 0"),
        (0.5, "Between supports"),
        (1.0, "At support point 1"),
        (1.0 + 1e-8, "Very close to support 1"),
        (2.5, "Far from supports"),
    ]
    
    print("="*80)
    print("TESTING ORIGINAL IMPLEMENTATION")
    print("="*80)
    
    # Create derivative functions for original
    f0_orig = lambda x: smooth_barycentric_eval_original(x, zj, fj, wj)
    f1_orig = jax.grad(f0_orig)
    f2_orig = jax.grad(f1_orig)
    
    for x_val, desc in test_points:
        print(f"\n{desc}: x = {x_val}")
        try:
            v0 = f0_orig(x_val)
            v1 = f1_orig(x_val)
            v2 = f2_orig(x_val)
            print(f"  f(x)   = {v0:.6f}")
            print(f"  f'(x)  = {v1:.6f}")
            print(f"  f''(x) = {v2:.6f}")
            
            if np.isnan(v1) or np.isnan(v2):
                print("  ❌ NaN detected in derivatives!")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "="*80)
    print("TESTING FIXED IMPLEMENTATION (jnp.where)")
    print("="*80)
    
    # Create derivative functions for fixed version
    f0_fixed = lambda x: smooth_barycentric_eval_fixed(x, zj, fj, wj)
    f1_fixed = jax.grad(f0_fixed)
    f2_fixed = jax.grad(f1_fixed)
    
    for x_val, desc in test_points:
        print(f"\n{desc}: x = {x_val}")
        try:
            v0 = f0_fixed(x_val)
            v1 = f1_fixed(x_val)
            v2 = f2_fixed(x_val)
            print(f"  f(x)   = {v0:.6f}")
            print(f"  f'(x)  = {v1:.6f}")
            print(f"  f''(x) = {v2:.6f}")
            
            if np.isnan(v1) or np.isnan(v2):
                print("  ❌ NaN detected in derivatives!")
            else:
                print("  ✅ All derivatives finite!")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Compare accuracy
    print("\n" + "="*80)
    print("ACCURACY COMPARISON")
    print("="*80)
    
    # Test on a finer grid
    x_test = jnp.linspace(-0.5, 3.5, 100)
    
    # True function is x²
    true_f = x_test**2
    true_f1 = 2*x_test
    true_f2 = 2*jnp.ones_like(x_test)
    
    # Evaluate both versions
    pred_orig = jax.vmap(f0_orig)(x_test)
    pred_fixed = jax.vmap(f0_fixed)(x_test)
    
    # Compute errors (ignoring NaN)
    err_orig = jnp.nanmean(jnp.abs(pred_orig - true_f))
    err_fixed = jnp.nanmean(jnp.abs(pred_fixed - true_f))
    
    print(f"Mean absolute error (function values):")
    print(f"  Original: {err_orig:.2e}")
    print(f"  Fixed:    {err_fixed:.2e}")
    
    # Count NaN occurrences in derivatives
    d1_orig_vals = jax.vmap(f1_orig)(x_test)
    d1_fixed_vals = jax.vmap(f1_fixed)(x_test)
    
    nan_count_orig = jnp.sum(jnp.isnan(d1_orig_vals))
    nan_count_fixed = jnp.sum(jnp.isnan(d1_fixed_vals))
    
    print(f"\nNaN count in first derivatives (out of {len(x_test)} points):")
    print(f"  Original: {nan_count_orig}")
    print(f"  Fixed:    {nan_count_fixed}")

if __name__ == "__main__":
    test_derivatives()