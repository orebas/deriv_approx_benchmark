#!/usr/bin/env python3
"""
Robust smooth barycentric implementation that avoids NaN completely
"""

import numpy as np
import jax.numpy as jnp
import jax

def smooth_barycentric_eval_robust(x, zj, fj, wj, W=1e-7):
    """
    Robust smooth barycentric evaluation without NaN in derivatives.
    
    Key insight: Completely reformulate to avoid problematic terms.
    Instead of alpha/d, we use a different smooth transition.
    """
    
    # Compute distances
    d = x - zj
    d_sq = d**2
    
    # Smooth transition function (0 when close, 1 when far)
    # Using exponential transition which is infinitely differentiable
    transition = 1.0 - jnp.exp(-d_sq / W)
    
    # Weights that smoothly transition from wj (at support) to wj/d (far away)
    # Key: we never explicitly divide by d
    # Instead, we use the fact that for small d:
    # (1 - exp(-d²/W))/d ≈ d/W
    
    # Safe computation of transition/d using L'Hôpital's rule result
    # For small d: (1 - exp(-d²/W))/d ≈ 2d/W * exp(-d²/W) ≈ 2d/W
    eps = 1e-10
    d_safe = jnp.where(jnp.abs(d) > eps, d, eps)
    
    # Compute the weight factors
    weight_factors = transition / d_safe * wj
    
    # Standard barycentric formula
    numerator = jnp.sum(weight_factors * fj)
    denominator = jnp.sum(weight_factors)
    
    # Add contribution from "close" points
    # When d is small, transition ≈ 0, so we need the direct contribution
    close_weights = (1.0 - transition) * wj
    close_numerator = jnp.sum(close_weights * fj)
    close_denominator = jnp.sum(close_weights)
    
    # Combine both contributions
    total_numerator = numerator + close_numerator
    total_denominator = denominator + close_denominator
    
    return total_numerator / (total_denominator + 1e-30)


def smooth_barycentric_eval_simplest(x, zj, fj, wj, epsilon=1e-8):
    """
    Simplest robust implementation: just regularize the distance.
    This is what I implemented earlier as a "fix", but let's test it properly.
    """
    
    # Compute regularized distances - never exactly zero
    d = x - zj
    d_reg = jnp.sqrt(d**2 + epsilon**2)
    
    # Standard barycentric weights with regularized distance
    weights = wj / d_reg
    
    # Standard barycentric formula
    numerator = jnp.sum(weights * fj)
    denominator = jnp.sum(weights)
    
    return numerator / (denominator + 1e-30)


def test_implementations():
    """Test all implementations thoroughly"""
    
    # Test case: simple quadratic
    zj = jnp.array([0.0, 1.0, 2.0])
    fj = jnp.array([0.0, 1.0, 4.0])  # f(x) = x²
    wj = jnp.ones(3)
    
    print("="*80)
    print("TESTING ROBUST EXPONENTIAL IMPLEMENTATION")
    print("="*80)
    
    f_robust = lambda x: smooth_barycentric_eval_robust(x, zj, fj, wj)
    f1_robust = jax.grad(f_robust)
    f2_robust = jax.grad(f1_robust)
    
    test_points = [
        (0.0, "At support 0"),
        (1e-10, "Very close to 0"),
        (0.5, "Between supports"),
        (1.0, "At support 1"),
        (2.0, "At support 2"),
    ]
    
    for x_val, desc in test_points:
        print(f"\n{desc}: x = {x_val}")
        v0 = f_robust(x_val)
        v1 = f1_robust(x_val)
        v2 = f2_robust(x_val)
        
        print(f"  f(x)   = {v0:.6f} (true: {x_val**2:.6f})")
        print(f"  f'(x)  = {v1:.6f} (true: {2*x_val:.6f})")
        print(f"  f''(x) = {v2:.6f} (true: 2.0)")
        
        if np.isnan(v1) or np.isnan(v2):
            print("  ❌ NaN in derivatives!")
        else:
            print("  ✅ No NaN!")
    
    print("\n" + "="*80)
    print("TESTING SIMPLE REGULARIZED IMPLEMENTATION")
    print("="*80)
    
    f_simple = lambda x: smooth_barycentric_eval_simplest(x, zj, fj, wj)
    f1_simple = jax.grad(f_simple)
    f2_simple = jax.grad(f1_simple)
    
    for x_val, desc in test_points:
        print(f"\n{desc}: x = {x_val}")
        v0 = f_simple(x_val)
        v1 = f1_simple(x_val)
        v2 = f2_simple(x_val)
        
        print(f"  f(x)   = {v0:.6f} (true: {x_val**2:.6f})")
        print(f"  f'(x)  = {v1:.6f} (true: {2*x_val:.6f})")
        print(f"  f''(x) = {v2:.6f} (true: 2.0)")
        
        if np.isnan(v1) or np.isnan(v2):
            print("  ❌ NaN in derivatives!")
        else:
            print("  ✅ No NaN!")
    
    # Test accuracy on a grid
    print("\n" + "="*80)
    print("ACCURACY COMPARISON")
    print("="*80)
    
    x_test = jnp.linspace(-0.5, 2.5, 100)
    true_vals = x_test**2
    
    robust_vals = jax.vmap(f_robust)(x_test)
    simple_vals = jax.vmap(f_simple)(x_test)
    
    err_robust = jnp.mean(jnp.abs(robust_vals - true_vals))
    err_simple = jnp.mean(jnp.abs(simple_vals - true_vals))
    
    print(f"Mean absolute error:")
    print(f"  Robust exponential: {err_robust:.2e}")
    print(f"  Simple regularized: {err_simple:.2e}")
    
    # Check for NaN in derivatives
    d1_robust = jax.vmap(f1_robust)(x_test)
    d1_simple = jax.vmap(f1_simple)(x_test)
    
    nan_robust = jnp.sum(jnp.isnan(d1_robust))
    nan_simple = jnp.sum(jnp.isnan(d1_simple))
    
    print(f"\nNaN count in derivatives:")
    print(f"  Robust exponential: {nan_robust}")
    print(f"  Simple regularized: {nan_simple}")

if __name__ == "__main__":
    test_implementations()