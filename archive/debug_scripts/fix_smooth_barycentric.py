#!/usr/bin/env python3
"""
Fixed smooth barycentric evaluation that avoids NaN in derivatives
"""

import jax.numpy as jnp
import jax

def smooth_barycentric_eval_fixed(x, zj, fj, wj, epsilon=1e-8):
    """
    Fixed smooth barycentric evaluation that avoids NaN in derivatives.
    
    Uses a simple regularization approach: add small epsilon to denominators
    to avoid division by zero, which prevents NaN in derivatives.
    
    Args:
        x: Evaluation point
        zj: Support points  
        fj: Function values at support points
        wj: Barycentric weights
        epsilon: Small regularization parameter (default 1e-8)
        
    Returns:
        Barycentric interpolation result with smooth derivatives
    """
    
    # Compute distances with regularization
    d = x - zj
    d_reg = d**2 + epsilon**2  # Regularized squared distance
    
    # Compute weights using regularized distance
    # This avoids division by zero and keeps derivatives smooth
    weights = wj / jnp.sqrt(d_reg)
    
    # Standard barycentric formula with regularized weights
    numerator = jnp.sum(weights * fj)
    denominator = jnp.sum(weights)
    
    return numerator / (denominator + 1e-30)  # Extra safety in denominator


def test_fixed_implementation():
    """Test the fixed implementation"""
    import numpy as np
    
    # Test data
    t = np.linspace(0, 2*np.pi, 20)
    y = np.sin(t)
    
    # Simple AAA-like setup
    indices = [0, 5, 10, 15, 19]
    zj = jnp.array(t[indices])
    fj = jnp.array(y[indices])
    wj = jnp.ones(len(indices))
    
    print("Testing fixed smooth barycentric evaluation:")
    print("="*60)
    
    # Test function evaluation
    print("\n1. Function evaluation:")
    for i in range(5):
        x = t[i*4]
        val = smooth_barycentric_eval_fixed(x, zj, fj, wj)
        print(f"  x = {x:.3f}: f(x) = {val:.6f}, true = {np.sin(x):.6f}")
    
    # Test derivatives
    print("\n2. Derivative evaluation:")
    
    # Create derivative functions
    eval_func = lambda x: smooth_barycentric_eval_fixed(x, zj, fj, wj)
    d1_func = jax.grad(eval_func)
    d2_func = jax.grad(d1_func)
    
    for i in range(5):
        x = t[i*4]
        f_val = eval_func(x)
        d1_val = d1_func(x)
        d2_val = d2_func(x)
        
        print(f"  x = {x:.3f}:")
        print(f"    f(x)  = {f_val:.6f} (true: {np.sin(x):.6f})")
        print(f"    f'(x) = {d1_val:.6f} (true: {np.cos(x):.6f})")
        print(f"    f''(x) = {d2_val:.6f} (true: {-np.sin(x):.6f})")
        
        # Check for NaN
        if np.isnan(d1_val) or np.isnan(d2_val):
            print("    ❌ NaN detected!")
        else:
            print("    ✅ No NaN")
    
    # Test at exact support points
    print("\n3. Evaluation at support points:")
    for i, z in enumerate(zj):
        f_val = eval_func(z)
        d1_val = d1_func(z)
        d2_val = d2_func(z)
        
        print(f"  z[{i}] = {z:.3f}:")
        print(f"    f = {f_val:.6f}, f' = {d1_val:.6f}, f'' = {d2_val:.6f}")
        
        if np.isnan(d1_val) or np.isnan(d2_val):
            print("    ❌ NaN at support point!")
        else:
            print("    ✅ No NaN at support point")

if __name__ == "__main__":
    test_fixed_implementation()