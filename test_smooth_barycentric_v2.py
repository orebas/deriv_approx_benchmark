#!/usr/bin/env python3
"""
Test Gemini's revised smooth barycentric evaluation (v2)
Fixes: W=1e-8, remove jnp.where, use jnp.nan_to_num for 0/0 cases
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

def smooth_barycentric_eval_v2(z, x, f, w, W=1e-8):
    """Gemini's revised smooth barycentric evaluation - no conditionals, proper W"""
    # Step 1: Pre-computation
    d = z - x
    d_sq = d**2
    alpha = jnp.tanh(d_sq / W)  # alpha=1 is far, alpha=0 is close
    
    # Step 2: Decomposed Sums
    # Critical fix: handle 0/0 case with nan_to_num
    raw_far_term = alpha / d
    safe_far_term = jnp.nan_to_num(raw_far_term, nan=0.0)
    
    N_far_unscaled = jnp.sum(safe_far_term * w * f)
    D_far_unscaled = jnp.sum(safe_far_term * w)
    
    one_minus_alpha = 1.0 - alpha
    N_close = jnp.sum(one_minus_alpha * w * f)
    D_close = jnp.sum(one_minus_alpha * w)
    
    # Step 3: Smooth Scaling Factor
    d_scale = jnp.sum(one_minus_alpha * d)
    
    # Step 4: Final Assembly (NO CONDITIONAL)
    N_final = N_close + d_scale * N_far_unscaled
    D_final = D_close + d_scale * D_far_unscaled
    
    return N_final / D_final

def test_numerical_equivalence_v2():
    """Test v2 numerical equivalence"""
    print("Testing numerical equivalence (v2)...")
    
    x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    f = jnp.array([0.0, 1.0, 4.0, 9.0, 16.0])  # x^2
    w = jnp.ones(5)
    
    z_test = jnp.array([0.5, 1.5, 2.5, 3.5, -0.5, 4.5])
    
    print("Support points:", x)
    print("Function values:", f)
    print()
    
    for z in z_test:
        try:
            naive_val = naive_barycentric_eval(z, x, f, w)
            smooth_val = smooth_barycentric_eval_v2(z, x, f, w)
            diff = abs(naive_val - smooth_val)
            
            print(f"z = {z:5.1f}: naive = {naive_val:12.8f}, smooth = {smooth_val:12.8f}, diff = {diff:.2e}")
            
            if diff > 1e-10:  # Slightly relaxed tolerance
                print(f"  ⚠️  WARNING: Large difference!")
            else:
                print(f"  ✅ Good agreement")
                
        except Exception as e:
            smooth_val = smooth_barycentric_eval_v2(z, x, f, w)
            print(f"z = {z:5.1f}: naive failed ({e}), smooth = {smooth_val:12.8f}")
    
    print()

def test_derivatives_v2():
    """Test that derivatives are finite everywhere"""
    print("Testing derivative computation (v2)...")
    
    x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    f = jnp.array([0.0, 1.0, 4.0, 9.0, 16.0])
    w = jnp.ones(5)
    
    def eval_at_z(z):
        return smooth_barycentric_eval_v2(z, x, f, w)
    
    # Test derivative at various points including near support points
    z_test = jnp.array([0.5, 1.0 + 1e-8, 1.5, 2.0 + 1e-8, 0.1, 3.9])
    
    print("Testing first derivatives:")
    all_finite = True
    for z in z_test:
        try:
            val = eval_at_z(z)
            grad = jax.grad(eval_at_z)(z)
            
            print(f"z = {z:12.8f}: f = {val:12.8f}, f' = {grad:12.8f}")
            
            if jnp.isfinite(grad):
                print(f"  ✅ Finite derivative")
            else:
                print(f"  ❌ Non-finite derivative!")
                all_finite = False
                
        except Exception as e:
            print(f"z = {z}: Gradient computation failed: {e}")
            all_finite = False
    
    if all_finite:
        print("\n🎉 ALL DERIVATIVES ARE FINITE!")
    else:
        print("\n❌ Some derivatives failed")
    
    print()

def test_near_support_points_v2():
    """Test behavior very close to support points"""
    print("Testing very close to support points (v2)...")
    
    x = jnp.array([0.0, 1.0, 2.0])
    f = jnp.array([1.0, 2.0, 4.0])
    w = jnp.array([1.0, 1.0, 1.0])
    
    # Test extremely close to x[1] = 1.0
    distances = [1e-3, 1e-6, 1e-9, 1e-12, 1e-15]
    
    print("Approaching x[1] = 1.0 (f[1] = 2.0)")
    for dist in distances:
        z = 1.0 + dist
        
        smooth_val = smooth_barycentric_eval_v2(z, x, f, w)
        
        # Test derivative too
        def eval_at_z(z_val):
            return smooth_barycentric_eval_v2(z_val, x, f, w)
        
        try:
            grad = jax.grad(eval_at_z)(z)
            grad_finite = jnp.isfinite(grad)
        except:
            grad = float('nan')
            grad_finite = False
        
        print(f"z = 1 + {dist:.0e}: f = {smooth_val:12.8f}, f' = {grad:12.8f}, finite = {grad_finite}")
        
        # Should approach f[1] = 2.0
        error_from_limit = abs(smooth_val - 2.0)
        if error_from_limit < 1e-6:
            print(f"  ✅ Good limit behavior")
        else:
            print(f"  ⚠️  Error from limit: {error_from_limit:.2e}")
    
    print()

def test_transition_behavior():
    """Test the transition region behavior"""
    print("Testing transition region behavior...")
    
    x = jnp.array([0.0, 1.0])
    f = jnp.array([0.0, 1.0])
    w = jnp.array([1.0, 1.0])
    
    # Test W parameter values
    W_values = [1e-6, 1e-8, 1e-10]
    z = 0.5  # Midpoint
    
    print("Testing different W values at z = 0.5:")
    for W in W_values:
        val = smooth_barycentric_eval_v2(z, x, f, w, W=W)
        
        def eval_at_z(z_val):
            return smooth_barycentric_eval_v2(z_val, x, f, w, W=W)
        
        grad = jax.grad(eval_at_z)(z)
        
        print(f"W = {W:.0e}: f = {val:12.8f}, f' = {grad:12.8f}")
    
    print()

if __name__ == "__main__":
    print("="*70)
    print("TESTING GEMINI'S REVISED SMOOTH BARYCENTRIC EVALUATION (V2)")
    print("="*70)
    print()
    
    test_numerical_equivalence_v2()
    test_derivatives_v2()
    test_near_support_points_v2()
    test_transition_behavior()
    
    print("="*70)
    print("TESTING COMPLETE")
    print("="*70)