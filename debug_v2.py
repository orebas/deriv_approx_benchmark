#!/usr/bin/env python3
"""
Debug the v2 implementation step by step
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

def debug_smooth_barycentric_v2(z, x, f, w, W=1e-8):
    """Debug version with detailed output"""
    print(f"\n=== DEBUGGING z = {z} ===")
    
    # Step 1: Pre-computation
    d = z - x
    d_sq = d**2
    alpha = jnp.tanh(d_sq / W)
    
    print(f"d = {d}")
    print(f"d_sq = {d_sq}")
    print(f"W = {W}")
    print(f"d_sq/W = {d_sq/W}")
    print(f"alpha = {alpha}")
    
    # Step 2: Check the problematic division
    raw_far_term = alpha / d
    print(f"raw_far_term (alpha/d) = {raw_far_term}")
    
    # Check for NaN
    has_nan = jnp.any(jnp.isnan(raw_far_term))
    print(f"Has NaN in raw_far_term: {has_nan}")
    
    safe_far_term = jnp.nan_to_num(raw_far_term, nan=0.0)
    print(f"safe_far_term = {safe_far_term}")
    
    # Check individual computations
    N_far_unscaled = jnp.sum(safe_far_term * w * f)
    D_far_unscaled = jnp.sum(safe_far_term * w)
    
    print(f"safe_far_term * w * f = {safe_far_term * w * f}")
    print(f"N_far_unscaled = {N_far_unscaled}")
    print(f"D_far_unscaled = {D_far_unscaled}")
    
    one_minus_alpha = 1.0 - alpha
    N_close = jnp.sum(one_minus_alpha * w * f)
    D_close = jnp.sum(one_minus_alpha * w)
    
    print(f"one_minus_alpha = {one_minus_alpha}")
    print(f"N_close = {N_close}")
    print(f"D_close = {D_close}")
    
    # Step 3: Smooth Scaling Factor
    d_scale = jnp.sum(one_minus_alpha * d)
    print(f"d_scale = {d_scale}")
    
    # Step 4: Final Assembly
    N_final = N_close + d_scale * N_far_unscaled
    D_final = D_close + d_scale * D_far_unscaled
    
    print(f"N_final = {N_final}")
    print(f"D_final = {D_final}")
    
    # Check for issues
    if abs(D_final) < 1e-15:
        print("⚠️ WARNING: D_final is nearly zero!")
    
    result = N_final / D_final
    print(f"result = {result}")
    
    return result

def test_simple_case():
    """Test the simplest possible case"""
    print("Testing simplest case: 2 points")
    
    x = jnp.array([0.0, 1.0])
    f = jnp.array([0.0, 1.0])
    w = jnp.array([1.0, 1.0])
    z = 0.5
    
    result = debug_smooth_barycentric_v2(z, x, f, w, W=1e-6)
    
def test_near_zero():
    """Test what happens near x=0"""
    print("\n" + "="*50)
    print("Testing near x=0")
    
    x = jnp.array([0.0, 1.0])
    f = jnp.array([1.0, 2.0])
    w = jnp.array([1.0, 1.0])
    z = 1e-6  # Very close to x[0]=0
    
    result = debug_smooth_barycentric_v2(z, x, f, w, W=1e-6)

def test_larger_W():
    """Test with much larger W"""
    print("\n" + "="*50)
    print("Testing with larger W")
    
    x = jnp.array([0.0, 1.0, 2.0])
    f = jnp.array([0.0, 1.0, 4.0])
    w = jnp.array([1.0, 1.0, 1.0])
    z = 0.5
    
    for W in [1e-3, 1e-2, 1e-1]:
        print(f"\n--- Testing W = {W} ---")
        result = debug_smooth_barycentric_v2(z, x, f, w, W=W)

if __name__ == "__main__":
    test_simple_case()
    test_near_zero()
    test_larger_W()