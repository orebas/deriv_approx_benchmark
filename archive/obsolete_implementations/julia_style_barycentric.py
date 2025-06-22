#!/usr/bin/env python3
"""
Implementation of the user's Julia barycentric algorithm with proper custom VJP
"""

import jax
import jax.numpy as jnp
import numpy as np

@jax.custom_vjp
def julia_barycentric_eval(x, zj, fj, wj, tol=1e-13):
    """
    Barycentric evaluation following the user's Julia algorithm.
    
    This handles singularities by excluding the problematic term and blending
    it back in linearly, which gives exact interpolation at support points
    with smooth derivatives everywhere.
    
    Args:
        x: Evaluation point (scalar)
        zj: Support points array
        fj: Function values at support points  
        wj: Barycentric weights
        tol: Tolerance for detecting near-support points
    
    Returns:
        Interpolated value
    """
    return _julia_bary_forward(x, zj, fj, wj, tol)

def _julia_bary_forward(x, zj, fj, wj, tol=1e-13):
    """Forward pass implementing the Julia algorithm"""
    d = x - zj
    
    # Check for exact hits first (machine precision)
    exact_indices = jnp.abs(d) < 1e-15
    if jnp.any(exact_indices):
        # Return exact value at support point
        idx = jnp.argmax(exact_indices)
        return fj[idx]
    
    # Check if close to any support point  
    close_indices = jnp.abs(d)**2 < jnp.sqrt(tol)
    
    if jnp.any(close_indices):
        # Find the closest support point (breakindex in Julia code)
        breakindex = jnp.argmin(jnp.abs(d))
        m = d[breakindex]  # Small distance to support point
        
        # Compute partial sums excluding the problematic term
        # This is the key insight: remove the singular term
        mask = jnp.arange(len(zj)) != breakindex
        
        # Only include terms that are not the problematic one
        safe_d = jnp.where(mask, d, 1.0)  # Avoid 0/0
        safe_weights = jnp.where(mask, wj / safe_d, 0.0)
        
        num_partial = jnp.sum(safe_weights * fj)
        den_partial = jnp.sum(safe_weights)
        
        # Linear blending formula from Julia code:
        # fz = (w[breakindex] * f[breakindex] + m * num) / (w[breakindex] + m * den)
        numerator = wj[breakindex] * fj[breakindex] + m * num_partial
        denominator = wj[breakindex] + m * den_partial
        
        return numerator / denominator
    else:
        # Standard barycentric formula when far from all support points
        weights = wj / d
        return jnp.sum(weights * fj) / jnp.sum(weights)

def _julia_bary_fwd_vjp(x, zj, fj, wj, tol=1e-13):
    """Forward pass for VJP"""
    value = _julia_bary_forward(x, zj, fj, wj, tol)
    return value, (x, zj, fj, wj, tol)

def _julia_bary_bwd_vjp(res, g):
    """
    Backward pass - compute derivative using small perturbation method
    This is more robust than analytical formulas for the complex blended case
    """
    x, zj, fj, wj, tol = res
    
    # Use symmetric finite differences with adaptive step size
    # This avoids the complex analytical derivatives while maintaining accuracy
    
    # Choose step size based on distance to nearest support point
    d = x - zj
    min_dist = jnp.min(jnp.abs(d))
    
    # Adaptive step size: smaller when close to support points
    h = jnp.maximum(1e-8, min_dist * 1e-3)
    
    # Symmetric finite difference
    f_plus = _julia_bary_forward(x + h, zj, fj, wj, tol)
    f_minus = _julia_bary_forward(x - h, zj, fj, wj, tol)
    
    derivative = (f_plus - f_minus) / (2 * h)
    
    return (g * derivative, None, None, None, None)

# Register the custom VJP
julia_barycentric_eval.defvjp(_julia_bary_fwd_vjp, _julia_bary_bwd_vjp)

def test_julia_style_barycentric():
    """Test the Julia-style barycentric implementation"""
    print("Testing Julia-style barycentric evaluation:")
    print("="*60)
    
    # Test case 1: Simple polynomial with known weights
    print("\n1. Testing with x² and proper polynomial weights:")
    zj = jnp.array([0.0, 1.0, 2.0])
    fj = jnp.array([0.0, 1.0, 4.0])  # f(x) = x²
    # For polynomial interpolation of x² at [0,1,2], the barycentric weights are:
    # These can be computed from the formula w_j = 1/∏(z_j - z_k) for k≠j
    wj = jnp.array([0.5, -1.0, 0.5])  # Proper barycentric weights
    
    print(f"Support points: {zj}")
    print(f"Function values: {fj}")  
    print(f"Barycentric weights: {wj}")
    
    # Create derivative functions
    f = lambda x: julia_barycentric_eval(x, zj, fj, wj)
    f_prime = jax.grad(f)
    f_double_prime = jax.grad(f_prime)
    
    # Test points
    test_points = [
        (0.0, "At support point 0"),
        (1.0, "At support point 1"),
        (2.0, "At support point 2"),
        (0.5, "Between supports"),
        (1.5, "Between supports"),
        (1e-14, "Very close to support"),
    ]
    
    print(f"\n{'Description':>20} {'x':>8} {'f(x)':>10} {'True':>10} {'f\'(x)':>10} {'True':>10} {'f\'\'(x)':>10} {'True':>10}")
    print("-" * 95)
    
    for x, desc in test_points:
        try:
            val = float(f(x))
            deriv1 = float(f_prime(x))
            deriv2 = float(f_double_prime(x))
            
            true_val = x**2
            true_deriv1 = 2*x
            true_deriv2 = 2.0
            
            print(f"{desc:>20} {x:8.0e} {val:10.6f} {true_val:10.6f} {deriv1:10.6f} {true_deriv1:10.6f} {deriv2:10.6f} {true_deriv2:10.6f}")
            
        except Exception as e:
            print(f"{desc:>20} {x:8.0e} ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_julia_style_barycentric()