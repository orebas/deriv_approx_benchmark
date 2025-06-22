#!/usr/bin/env python3
"""
Precise analysis of smooth barycentric evaluation errors at machine precision
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

def naive_barycentric_eval(z, x, f, w):
    """Standard barycentric evaluation"""
    diffs = z - x
    weights = w / diffs
    num = jnp.sum(weights * f)
    den = jnp.sum(weights)
    return num / den

def smooth_barycentric_eval(z, x, f, w, W):
    """Smooth barycentric evaluation with configurable W"""
    d = z - x
    d_sq = d**2
    alpha = jnp.tanh(d_sq / W)
    
    safe_far_term = jnp.nan_to_num(alpha / d, nan=0.0)
    N_far_unscaled = jnp.sum(safe_far_term * w * f)
    D_far_unscaled = jnp.sum(safe_far_term * w)
    
    one_minus_alpha = 1.0 - alpha
    N_close = jnp.sum(one_minus_alpha * w * f)
    D_close = jnp.sum(one_minus_alpha * w)
    
    d_scale = jnp.sum(one_minus_alpha * d)
    gamma = jnp.prod(alpha)
    
    N_final = N_close + d_scale * N_far_unscaled + gamma * N_far_unscaled
    D_final = D_close + d_scale * D_far_unscaled + gamma * D_far_unscaled
    
    return N_final / (D_final + 1e-30)

def test_machine_precision():
    """Test precision with extremely small W values"""
    print("MACHINE PRECISION ANALYSIS")
    print("="*60)
    
    # Simple test case
    x = jnp.array([0.0, 1.0, 2.0])
    f = jnp.array([1.0, 2.0, 4.0])
    w = jnp.array([1.0, 1.0, 1.0])
    z = 0.5  # Far from all support points
    
    naive_val = naive_barycentric_eval(z, x, f, w)
    print(f"Naive (reference) result: {naive_val:.16e}")
    print()
    
    # Test with progressively smaller W values
    W_values = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16]
    
    print("W value sweep for machine precision:")
    print("W           Smooth Result              Absolute Error        Relative Error")
    print("-" * 75)
    
    for W in W_values:
        try:
            smooth_val = smooth_barycentric_eval(z, x, f, w, W)
            abs_error = abs(smooth_val - naive_val)
            rel_error = abs_error / abs(naive_val) if naive_val != 0 else 0
            
            print(f"{W:.0e}    {smooth_val:.16e}    {abs_error:.2e}    {rel_error:.2e}")
            
            # Test derivative too
            def eval_func(z_val):
                return smooth_barycentric_eval(z_val, x, f, w, W)
            
            grad = jax.grad(eval_func)(z)
            if not jnp.isfinite(grad):
                print(f"         ⚠️ Gradient is not finite!")
                
        except Exception as e:
            print(f"{W:.0e}    FAILED: {e}")
    
    print()

def test_near_support_point_precision():
    """Test precision when very close to support points"""
    print("NEAR SUPPORT POINT PRECISION")
    print("="*60)
    
    x = jnp.array([0.0, 1.0, 2.0])
    f = jnp.array([1.0, 5.0, 9.0])  # Clear distinct values
    w = jnp.array([1.0, 1.0, 1.0])
    
    # Test very close to x[1] = 1.0, should approach f[1] = 5.0
    distances = [1e-6, 1e-9, 1e-12, 1e-15]
    W_test = 1e-14  # Very small W
    
    print(f"Testing approach to x[1] = 1.0 (f[1] = 5.0) with W = {W_test:.0e}")
    print("Distance    Smooth Result              Error from Limit")
    print("-" * 55)
    
    for dist in distances:
        z = 1.0 + dist
        
        try:
            smooth_val = smooth_barycentric_eval(z, x, f, w, W_test)
            error_from_limit = abs(smooth_val - 5.0)
            
            print(f"{dist:.0e}     {smooth_val:.16e}    {error_from_limit:.2e}")
            
        except Exception as e:
            print(f"{dist:.0e}     FAILED: {e}")
    
    print()

def test_aaa_level_precision():
    """Test if we can achieve 1e-14 level precision like vanilla AAA"""
    print("AAA-LEVEL PRECISION TEST")
    print("="*60)
    
    # Use a more realistic AAA-like setup
    x = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    f = jnp.sin(x)  # Smooth function
    w = jnp.ones(5)
    
    # Test points between support points
    z_test = jnp.array([-0.75, -0.25, 0.25, 0.75])
    
    print("Support points:", x)
    print("Function values:", f)
    print()
    
    # Find W that gives machine precision
    target_precision = 1e-14
    
    for z in z_test:
        naive_val = naive_barycentric_eval(z, x, f, w)
        
        print(f"Test point z = {z:6.2f}:")
        print(f"  Naive result: {naive_val:.16e}")
        
        # Try to find W that achieves target precision
        best_W = None
        best_error = float('inf')
        
        for W in [1e-12, 1e-14, 1e-16, 1e-18]:
            try:
                smooth_val = smooth_barycentric_eval(z, x, f, w, W)
                error = abs(smooth_val - naive_val)
                
                print(f"  W = {W:.0e}: error = {error:.2e}")
                
                if error < best_error:
                    best_error = error
                    best_W = W
                    
                # Test gradient
                def eval_func(z_val):
                    return smooth_barycentric_eval(z_val, x, f, w, W)
                
                grad = jax.grad(eval_func)(z)
                if not jnp.isfinite(grad):
                    print(f"            ⚠️ Gradient not finite")
                    
            except Exception as e:
                print(f"  W = {W:.0e}: FAILED - {e}")
        
        print(f"  Best: W = {best_W:.0e}, error = {best_error:.2e}")
        
        if best_error < target_precision:
            print(f"  ✅ Achieved AAA-level precision!")
        else:
            print(f"  ⚠️ Still {best_error/target_precision:.1f}x above target")
        
        print()

if __name__ == "__main__":
    test_machine_precision()
    test_near_support_point_precision()
    test_aaa_level_precision()