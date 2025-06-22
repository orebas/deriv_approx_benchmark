#!/usr/bin/env python3
"""
Test Gemini's smooth barycentric evaluation approach.
First verify numerical equivalence with naive formula, then test AD compatibility.
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

def smooth_barycentric_eval(z, x, f, w, W=1e-14):
    """Gemini's smooth barycentric evaluation"""
    # Step 1: Pre-computation
    d = z - x
    d_sq = d**2
    alpha = jnp.tanh(d_sq / W)  # alpha=1 is far, alpha=0 is close
    
    # Step 2: Decomposed Sums
    # Note: The (alpha / d) term is stable as d -> 0
    N_far_unscaled = jnp.sum((alpha * w * f) / d)
    D_far_unscaled = jnp.sum((alpha * w) / d)
    
    N_close = jnp.sum((1.0 - alpha) * w * f)
    D_close = jnp.sum((1.0 - alpha) * w)
    
    # Step 3: Smooth Scaling Factor
    d_scale = jnp.sum((1.0 - alpha) * d)
    
    # Step 4: Final Assembly
    N_final = N_close + d_scale * N_far_unscaled
    D_final = D_close + d_scale * D_far_unscaled
    
    # Handle the case where z is far from all points
    far_from_all = D_final < 1e-15
    naive_result = jnp.sum(w * f / d) / jnp.sum(w / d)
    
    return jnp.where(far_from_all, naive_result, N_final / D_final)

def test_numerical_equivalence():
    """Test that smooth formula equals naive formula when z is not too close to any x[j]"""
    print("Testing numerical equivalence...")
    
    # Test case 1: Simple polynomial-like data
    x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    f = jnp.array([0.0, 1.0, 4.0, 9.0, 16.0])  # x^2
    w = jnp.ones(5)
    
    # Test points far from support points
    z_test = jnp.array([0.5, 1.5, 2.5, 3.5, -0.5, 4.5])
    
    print("Test case 1: Simple polynomial data")
    print("Support points:", x)
    print("Function values:", f)
    print("Weights:", w)
    print()
    
    for z in z_test:
        try:
            naive_val = naive_barycentric_eval(z, x, f, w)
            smooth_val = smooth_barycentric_eval(z, x, f, w)
            diff = abs(naive_val - smooth_val)
            
            print(f"z = {z:5.1f}: naive = {naive_val:12.8f}, smooth = {smooth_val:12.8f}, diff = {diff:.2e}")
            
            # Check they're nearly equal (within numerical precision)
            if diff > 1e-12:
                print(f"  ⚠️  WARNING: Large difference!")
            else:
                print(f"  ✅ Good agreement")
                
        except Exception as e:
            print(f"z = {z:5.1f}: naive failed ({e}), smooth = {smooth_barycentric_eval(z, x, f, w)}")
    
    print()

def test_near_support_points():
    """Test behavior when z is very close to support points"""
    print("Testing near support point behavior...")
    
    x = jnp.array([0.0, 1.0, 2.0])
    f = jnp.array([1.0, 2.0, 4.0])
    w = jnp.array([1.0, 1.0, 1.0])
    
    # Test very close to x[1] = 1.0
    distances = [1e-3, 1e-6, 1e-9, 1e-12, 1e-15]
    
    print("Test case 2: Approaching x[1] = 1.0 (f[1] = 2.0)")
    for dist in distances:
        z = 1.0 + dist
        
        try:
            naive_val = naive_barycentric_eval(z, x, f, w)
        except:
            naive_val = float('nan')
            
        smooth_val = smooth_barycentric_eval(z, x, f, w)
        
        print(f"z = 1 + {dist:.0e}: naive = {naive_val:12.8f}, smooth = {smooth_val:12.8f}")
        
        # Smooth value should approach f[1] = 2.0
        if abs(smooth_val - 2.0) < dist * 10:
            print(f"  ✅ Approaching correct limit f[1] = 2.0")
        else:
            print(f"  ⚠️  Not approaching f[1] = 2.0")
    
    print()

def test_derivatives():
    """Test that derivatives are finite and reasonable"""
    print("Testing derivative computation...")
    
    x = jnp.array([0.0, 1.0, 2.0])
    f = jnp.array([1.0, 2.0, 4.0])  
    w = jnp.array([1.0, 1.0, 1.0])
    
    # Define function for AD
    def eval_at_z(z):
        return smooth_barycentric_eval(z, x, f, w)
    
    # Test derivative at various points
    z_test = jnp.array([0.5, 1.0 + 1e-8, 1.5, 2.0 + 1e-8])
    
    print("Testing first derivatives:")
    for z in z_test:
        try:
            val = eval_at_z(z)
            grad = jax.grad(eval_at_z)(z)
            
            print(f"z = {z:12.8f}: f = {val:12.8f}, f' = {grad:12.8f}")
            
            if jnp.isfinite(grad):
                print(f"  ✅ Finite derivative")
            else:
                print(f"  ❌ Non-finite derivative!")
                
        except Exception as e:
            print(f"z = {z}: Gradient computation failed: {e}")
    
    print()

def test_multiple_close_points():
    """Test case where z is close to multiple support points"""
    print("Testing multiple close points...")
    
    # Three points clustered together
    x = jnp.array([1.0, 1.001, 1.002])
    f = jnp.array([2.0, 3.0, 4.0])
    w = jnp.array([1.0, 1.0, 1.0])
    
    # Test point near the cluster
    z = 1.0005
    
    try:
        naive_val = naive_barycentric_eval(z, x, f, w)
    except:
        naive_val = float('nan')
        
    smooth_val = smooth_barycentric_eval(z, x, f, w)
    
    print(f"Clustered points: x = {x}")
    print(f"Function values: f = {f}")
    print(f"Test point z = {z}")
    print(f"Naive result: {naive_val}")
    print(f"Smooth result: {smooth_val}")
    
    # Check derivative
    def eval_at_z(z):
        return smooth_barycentric_eval(z, x, f, w)
    
    try:
        grad = jax.grad(eval_at_z)(z)
        print(f"Gradient: {grad}")
        print("✅ Handled multiple close points successfully")
    except Exception as e:
        print(f"❌ Gradient failed: {e}")
    
    print()

if __name__ == "__main__":
    print("="*60)
    print("TESTING GEMINI'S SMOOTH BARYCENTRIC EVALUATION")
    print("="*60)
    print()
    
    test_numerical_equivalence()
    test_near_support_points()
    test_derivatives()
    test_multiple_close_points()
    
    print("="*60)
    print("TESTING COMPLETE")
    print("="*60)