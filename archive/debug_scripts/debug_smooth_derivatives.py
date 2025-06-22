#!/usr/bin/env python3
"""
Debug why smooth_barycentric_eval produces NaN derivatives
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

def smooth_barycentric_eval_debug(z, x, f, w, W=1e-14):
    """Gemini's smooth barycentric evaluation with debug output"""
    # Step 1: Pre-computation
    d = z - x
    d_sq = d**2
    alpha = jnp.tanh(d_sq / W)  # alpha=1 is far, alpha=0 is close
    
    print(f"z = {z}")
    print(f"d = {d}")
    print(f"d_sq = {d_sq}")
    print(f"alpha = {alpha}")
    
    # Step 2: Decomposed Sums
    weights_for_far = (alpha * w) / d
    print(f"weights_for_far = {weights_for_far}")
    print(f"Are weights_for_far finite? {jnp.all(jnp.isfinite(weights_for_far))}")
    
    N_far_unscaled = jnp.sum((alpha * w * f) / d)
    D_far_unscaled = jnp.sum(weights_for_far)
    
    N_close = jnp.sum((1.0 - alpha) * w * f)
    D_close = jnp.sum((1.0 - alpha) * w)
    
    print(f"N_far_unscaled = {N_far_unscaled}")
    print(f"D_far_unscaled = {D_far_unscaled}")
    print(f"N_close = {N_close}")
    print(f"D_close = {D_close}")
    
    # Step 3: Smooth Scaling Factor
    d_scale = jnp.sum((1.0 - alpha) * d)
    print(f"d_scale = {d_scale}")
    
    # Step 4: Final Assembly
    N_final = N_close + d_scale * N_far_unscaled
    D_final = D_close + d_scale * D_far_unscaled
    
    print(f"N_final = {N_final}")
    print(f"D_final = {D_final}")
    
    # Handle the case where z is far from all points
    far_from_all = D_final < 1e-15
    print(f"far_from_all = {far_from_all}")
    
    if far_from_all:
        naive_result = jnp.sum(w * f / d) / jnp.sum(w / d)
        print(f"Using naive result: {naive_result}")
        return naive_result
    else:
        result = N_final / D_final
        print(f"Using smooth result: {result}")
        return result

def investigate_nan_derivative():
    """Investigate why derivative is NaN at z=0.5"""
    print("="*50)
    print("INVESTIGATING NaN DERIVATIVE AT z=0.5")
    print("="*50)
    
    x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    f = jnp.array([0.0, 1.0, 4.0, 9.0, 16.0])
    w = jnp.ones(5)
    z = 0.5
    
    print("Support points:", x)
    print("Function values:", f)
    print("Test point z =", z)
    print()
    
    # Direct evaluation
    result = smooth_barycentric_eval_debug(z, x, f, w)
    print(f"Final result: {result}")
    print()
    
    # Try computing gradient
    def eval_func(z_val):
        return smooth_barycentric_eval_debug(z_val, x, f, w)
    
    print("Computing gradient...")
    try:
        grad = jax.grad(eval_func)(z)
        print(f"Gradient: {grad}")
    except Exception as e:
        print(f"Gradient computation failed: {e}")
    
    print()

def test_simpler_case():
    """Test with a simpler case to isolate the issue"""
    print("="*50)
    print("TESTING SIMPLER CASE")
    print("="*50)
    
    # Just 3 points
    x = jnp.array([0.0, 1.0, 2.0])
    f = jnp.array([1.0, 2.0, 3.0])
    w = jnp.array([1.0, 1.0, 1.0])
    z = 0.5
    
    print("Support points:", x)
    print("Function values:", f)
    print("Test point z =", z)
    print()
    
    # Direct evaluation with detailed output
    result = smooth_barycentric_eval_debug(z, x, f, w)
    print()
    
def check_problematic_terms():
    """Check if the issue is in specific terms"""
    print("="*50)
    print("CHECKING PROBLEMATIC TERMS")
    print("="*50)
    
    x = jnp.array([0.0, 1.0, 2.0])
    f = jnp.array([1.0, 2.0, 3.0])
    w = jnp.array([1.0, 1.0, 1.0])
    z = 0.5
    W = 1e-14
    
    d = z - x
    d_sq = d**2
    alpha = jnp.tanh(d_sq / W)
    
    print(f"d = {d}")
    print(f"d_sq = {d_sq}")
    print(f"W = {W}")
    print(f"d_sq/W = {d_sq/W}")
    print(f"alpha = {alpha}")
    print()
    
    # Check individual terms in the sum
    for i in range(len(x)):
        term = (alpha[i] * w[i]) / d[i]
        print(f"Point {i}: alpha={alpha[i]:.6e}, d={d[i]:.6e}, term={term:.6e}")
    
    print()
    
    # Check if the issue is the tanh function for large arguments
    print("Testing tanh behavior:")
    for exp in [10, 14, 18, 22]:
        arg = 10**exp
        tanh_val = jnp.tanh(arg)
        print(f"tanh(1e{exp}) = {tanh_val}")

if __name__ == "__main__":
    check_problematic_terms()
    test_simpler_case()
    investigate_nan_derivative()