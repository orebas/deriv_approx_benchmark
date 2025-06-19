#!/usr/bin/env python3
"""
Debug why jnp.where still produces NaN
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

def test_where_gradient():
    """Test gradient of jnp.where at branch point"""
    
    print("Testing why jnp.where still produces NaN")
    print("="*60)
    
    # Simple test of jnp.where gradient behavior
    def f_where(x):
        # Similar structure to our problematic term
        return jnp.where(
            jnp.abs(x) < 1e-6,
            x / 1e-7,  # Branch 1: x/W
            jnp.tanh(x**2 / 1e-7) / x  # Branch 2: tanh(x²/W)/x
        )
    
    # Test at x=0
    print("\nTesting simple where function at x=0:")
    try:
        val = f_where(0.0)
        grad_val = jax.grad(f_where)(0.0)
        print(f"f(0) = {val}")
        print(f"f'(0) = {grad_val}")
    except Exception as e:
        print(f"Error: {e}")
    
    # The issue is that JAX still evaluates BOTH branches
    # Let's verify this
    print("\n\nUnderstanding JAX's behavior:")
    print("-"*60)
    
    def debug_where(x):
        threshold = 1e-6
        cond = jnp.abs(x) < threshold
        
        # Branch 1: Taylor approximation
        branch1 = x / 1e-7
        
        # Branch 2: Original formula (problematic at x=0)
        # This still gets evaluated even if not selected!
        branch2 = jnp.tanh(x**2 / 1e-7) / x
        
        return jnp.where(cond, branch1, branch2)
    
    print("JAX evaluates BOTH branches of jnp.where, then selects.")
    print("So at x=0, it still computes 0/0 in branch2, causing NaN.")
    
    # The real fix: restructure to avoid 0/0 entirely
    print("\n\nProper fix: Restructure the formula")
    print("-"*60)
    
    def proper_smooth_term(x, W=1e-7):
        """Properly handle the singularity without creating 0/0"""
        d_sq = x**2
        
        # For small x, use Taylor series of the entire expression
        # tanh(x²/W)/x ≈ x/W - x³/(3W²) + ...
        # We'll use just the first term for simplicity
        
        # But more robust: factor out the division differently
        # tanh(d²/W) = d² * (tanh(d²/W)/(d²/W)) / W
        # So tanh(d²/W)/d = d * (tanh(d²/W)/(d²/W)) / W
        
        # The function tanh(u)/u is smooth at u=0 with value 1
        u = d_sq / W
        tanh_u_over_u = jnp.where(
            u < 1e-10,
            1.0 - u/3.0,  # Taylor series: tanh(u)/u ≈ 1 - u/3 + ...
            jnp.tanh(u) / u
        )
        
        return x / W * tanh_u_over_u
    
    print("Testing proper implementation:")
    test_points = [0.0, 1e-8, 1e-6, 1e-4, 0.1]
    
    for x_val in test_points:
        val = proper_smooth_term(x_val)
        grad_val = jax.grad(proper_smooth_term)(x_val)
        print(f"x = {x_val:g}: f(x) = {val:.6f}, f'(x) = {grad_val:.6f}")
        
        if np.isnan(grad_val):
            print("  ❌ Still NaN!")
        else:
            print("  ✅ No NaN!")

if __name__ == "__main__":
    test_where_gradient()