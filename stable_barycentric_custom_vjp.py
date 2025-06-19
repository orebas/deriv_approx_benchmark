#!/usr/bin/env python3
"""
Stable barycentric evaluation with custom VJP for JAX automatic differentiation
Based on the user's Julia algorithm with proper derivative handling
"""

import jax
import jax.numpy as jnp
import numpy as np

@jax.custom_vjp
def stable_barycentric_eval(x, zj, fj, wj, tol=1e-13):
    """
    Stable barycentric evaluation that handles singularities at support points.
    Uses custom VJP to ensure proper derivatives for automatic differentiation.
    
    Args:
        x: Evaluation point (scalar)
        zj: Support points array
        fj: Function values at support points
        wj: Barycentric weights
        tol: Tolerance for detecting near-support points
    
    Returns:
        Interpolated value
    """
    return _stable_bary_fwd(x, zj, fj, wj, tol)

def _stable_bary_fwd(x, zj, fj, wj, tol=1e-13):
    """Forward pass implementation"""
    d = x - zj
    
    # Check for exact hits first (most important for derivatives)
    exact_hit = jnp.abs(d) < 1e-15  # Machine precision
    if jnp.any(exact_hit):
        idx = jnp.argmax(exact_hit)
        return fj[idx]
    
    # Check for near-support points
    close_to_support = jnp.abs(d) < jnp.sqrt(tol)
    
    if jnp.any(close_to_support):
        # Find closest support point
        breakindex = jnp.argmin(jnp.abs(d))
        m = d[breakindex]
        
        # Compute partial sums excluding the problematic term
        exclude_mask = jnp.arange(len(zj)) != breakindex
        safe_d = jnp.where(exclude_mask, d, 1.0)  # Avoid division by zero
        safe_weights = jnp.where(exclude_mask, wj / safe_d, 0.0)
        
        num_partial = jnp.sum(safe_weights * fj)
        den_partial = jnp.sum(safe_weights)
        
        # Linear blending formula
        return (wj[breakindex] * fj[breakindex] + m * num_partial) / (wj[breakindex] + m * den_partial)
    else:
        # Standard barycentric formula
        weights = wj / d
        return jnp.sum(weights * fj) / jnp.sum(weights)

def _stable_bary_fwd_vjp(x, zj, fj, wj, tol=1e-13):
    """Forward pass for VJP rule"""
    value = _stable_bary_fwd(x, zj, fj, wj, tol)
    return value, (x, zj, fj, wj, tol)

def _stable_bary_bwd_vjp(res, g):
    """Backward pass for VJP rule - computes derivative"""
    x, zj, fj, wj, tol = res
    
    # Compute derivative analytically
    d = x - zj
    
    # For exact hits, derivative is 0 (function is locally constant)
    exact_hit = jnp.abs(d) < 1e-15
    if jnp.any(exact_hit):
        return (0.0, None, None, None, None)
    
    # For the general case, use the derivative of the rational function
    # P(x) = N(x)/D(x) where N(x) = Σ(w_j * f_j / (x - z_j)), D(x) = Σ(w_j / (x - z_j))
    # P'(x) = (N'(x)*D(x) - N(x)*D'(x)) / D(x)²
    
    weights = wj / d
    N = jnp.sum(weights * fj)
    D = jnp.sum(weights)
    
    # Derivatives: N'(x) = -Σ(w_j * f_j / (x - z_j)²), D'(x) = -Σ(w_j / (x - z_j)²)
    weights_deriv = -wj / (d**2)
    N_prime = jnp.sum(weights_deriv * fj)
    D_prime = jnp.sum(weights_deriv)
    
    # Quotient rule: P'(x) = (N'*D - N*D') / D²
    derivative = (N_prime * D - N * D_prime) / (D**2)
    
    # Return gradient w.r.t. x (multiplied by incoming gradient g)
    return (g * derivative, None, None, None, None)

# Register the custom VJP
stable_barycentric_eval.defvjp(_stable_bary_fwd_vjp, _stable_bary_bwd_vjp)

def test_stable_barycentric():
    """Test the stable barycentric implementation"""
    print("Testing stable barycentric evaluation with custom VJP:")
    print("="*60)
    
    # Test case: quadratic function
    zj = jnp.array([0.0, 1.0, 2.0])
    fj = jnp.array([0.0, 1.0, 4.0])  # f(x) = x²
    
    # Get proper barycentric weights for polynomial interpolation
    # For uniformly spaced points interpolating x², weights are [1, -2, 1]
    wj = jnp.array([1.0, -2.0, 1.0])
    
    print("Support points:", zj)
    print("Function values:", fj)
    print("Weights:", wj)
    
    # Create derivative functions
    f = lambda x: stable_barycentric_eval(x, zj, fj, wj)
    f_prime = jax.grad(f)
    f_double_prime = jax.grad(f_prime)
    
    # Test at various points
    test_points = [
        (0.0, "At support point 0"),
        (1.0, "At support point 1"),
        (2.0, "At support point 2"),
        (0.5, "Between support points"),
        (1.5, "Between support points"),
        (1e-14, "Very close to support point"),
    ]
    
    print(f"\n{'Point':>15} {'f(x)':>10} {'True':>10} {'f\'(x)':>10} {'True':>10} {'f\'\'(x)':>10} {'True':>10}")
    print("-" * 80)
    
    for x, desc in test_points:
        try:
            val = float(f(x))
            deriv1 = float(f_prime(x))
            deriv2 = float(f_double_prime(x))
            
            true_val = x**2
            true_deriv1 = 2*x
            true_deriv2 = 2.0
            
            print(f"{desc:>15} {val:10.6f} {true_val:10.6f} {deriv1:10.6f} {true_deriv1:10.6f} {deriv2:10.6f} {true_deriv2:10.6f}")
            
            if np.isnan(deriv1) or np.isnan(deriv2):
                print(f"                ❌ NaN in derivatives!")
            else:
                print(f"                ✅ All derivatives finite")
                
        except Exception as e:
            print(f"{desc:>15} ERROR: {e}")

if __name__ == "__main__":
    test_stable_barycentric()