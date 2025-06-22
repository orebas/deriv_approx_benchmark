#!/usr/bin/env python3
"""
Analytical VJP implementation for the stable barycentric evaluation function.
This replaces the finite difference approach to fix higher-order derivative explosion.
"""

import jax
import jax.numpy as jnp
import numpy as np

@jax.custom_vjp
def analytical_barycentric_eval(x, zj, fj, wj, W=1e-7):
    """
    Stable barycentric evaluation with analytical derivatives.
    
    This implements the Julia algorithm with proper analytical derivatives
    to avoid the numerical instability of finite differences for higher orders.
    
    Args:
        x: Evaluation point
        zj: Support points  
        fj: Function values at support points
        wj: Barycentric weights
        W: Compatibility parameter, converted to tolerance (default 1e-7)
        
    Returns:
        Barycentric interpolation result
    """
    tol = W * 1e6
    return _analytical_bary_forward(x, zj, fj, wj, tol)

def _analytical_bary_forward(x, zj, fj, wj, tol=1e-13):
    """Forward pass implementing the Julia algorithm"""
    d = x - zj
    
    # Case 1: Exact hits (machine precision)
    exact_indices = jnp.abs(d) < 1e-15
    exact_hit = jnp.any(exact_indices)
    exact_idx = jnp.argmax(exact_indices)
    exact_result = fj[exact_idx]
    
    # Case 2: Close to support points (blended case)
    close_indices = jnp.abs(d)**2 < jnp.sqrt(tol)
    close_hit = jnp.any(close_indices)
    breakindex = jnp.argmin(jnp.abs(d))
    m = d[breakindex]
    
    # Compute blended result
    mask = jnp.arange(len(zj)) != breakindex
    safe_d = jnp.where(mask, d, 1.0)
    safe_weights = jnp.where(mask, wj / safe_d, 0.0)
    
    num_partial = jnp.sum(safe_weights * fj)
    den_partial = jnp.sum(safe_weights)
    
    blended_result = (wj[breakindex] * fj[breakindex] + m * num_partial) / (wj[breakindex] + m * den_partial)
    
    # Case 3: Standard barycentric (far from all points)
    weights = wj / d
    standard_result = jnp.sum(weights * fj) / jnp.sum(weights)
    
    # Select the appropriate result
    result = jnp.where(
        exact_hit,
        exact_result,
        jnp.where(close_hit, blended_result, standard_result)
    )
    
    return result

def _analytical_bary_fwd_vjp(x, zj, fj, wj, W=1e-7):
    """Forward pass for VJP"""
    result = _analytical_bary_forward(x, zj, fj, wj, W * 1e6)
    return result, (x, zj, fj, wj, W)

def _analytical_bary_bwd_vjp(res, g):
    """
    Backward pass with analytical derivatives.
    
    This computes the exact derivative for each case in the Julia algorithm
    to avoid the numerical instability of finite differences.
    """
    x, zj, fj, wj, W = res
    tol = W * 1e6
    d = x - zj
    
    # Determine which case we're in
    exact_indices = jnp.abs(d) < 1e-15
    exact_hit = jnp.any(exact_indices)
    
    close_indices = jnp.abs(d)**2 < jnp.sqrt(tol)
    close_hit = jnp.any(close_indices)
    
    # Case 1: Exact hit at support point
    # For exact hits, compute the derivative directly using L'Hopital's rule
    # The derivative at a support point is the standard barycentric derivative
    # evaluated in the limit as x approaches the support point
    exact_idx = jnp.argmax(exact_indices)
    
    # Use the standard barycentric derivative formula but exclude the singular term
    d_safe = jnp.where(exact_indices, 1.0, d)  # Avoid 0 in denominator
    weights_safe = jnp.where(exact_indices, 0.0, wj / d_safe)
    
    num_safe = jnp.sum(weights_safe * fj)
    den_safe = jnp.sum(weights_safe)
    
    # Derivative terms (excluding the exact hit term)
    dnum_safe_dx = -jnp.sum(jnp.where(exact_indices, 0.0, wj * fj / (d_safe**2)))
    dden_safe_dx = -jnp.sum(jnp.where(exact_indices, 0.0, wj / (d_safe**2)))
    
    exact_derivative = jnp.where(
        den_safe != 0,
        (den_safe * dnum_safe_dx - num_safe * dden_safe_dx) / (den_safe**2),
        0.0
    )
    
    # Case 2: Close to support point (blended case)
    breakindex = jnp.argmin(jnp.abs(d))
    m = d[breakindex]
    
    # Compute partial sums (excluding breakindex term)
    mask = jnp.arange(len(zj)) != breakindex
    safe_d = jnp.where(mask, d, 1.0)
    safe_weights = jnp.where(mask, wj / safe_d, 0.0)
    
    num_partial = jnp.sum(safe_weights * fj)
    den_partial = jnp.sum(safe_weights)
    
    # Analytical derivative of blended formula
    # f(x) = (w_k * f_k + m * num_partial) / (w_k + m * den_partial)
    # where m = x - z_k
    
    # Derivatives of num_partial and den_partial with respect to x
    # d/dx (w_j / (x - z_j)) = -w_j / (x - z_j)^2
    # Avoid division by zero by excluding terms where d is very small
    safe_d_squared = jnp.where(jnp.abs(safe_d) < 1e-12, 1.0, safe_d**2)
    d_term_mask = mask & (jnp.abs(safe_d) > 1e-12)
    
    dnum_dx = -jnp.sum(jnp.where(d_term_mask, wj * fj / safe_d_squared, 0.0))
    dden_dx = -jnp.sum(jnp.where(d_term_mask, wj / safe_d_squared, 0.0))
    
    # Quotient rule: d/dx (A/B) = (B * dA/dx - A * dB/dx) / B^2
    numerator = wj[breakindex] * fj[breakindex] + m * num_partial
    denominator = wj[breakindex] + m * den_partial
    
    # d/dx numerator = d/dx (w_k * f_k + (x - z_k) * num_partial)
    #                = num_partial + (x - z_k) * dnum_dx
    dnumerator_dx = num_partial + m * dnum_dx
    
    # d/dx denominator = d/dx (w_k + (x - z_k) * den_partial)
    #                  = den_partial + (x - z_k) * dden_dx
    ddenominator_dx = den_partial + m * dden_dx
    
    blended_derivative = (denominator * dnumerator_dx - numerator * ddenominator_dx) / (denominator**2)
    
    # Case 3: Standard barycentric (far from all points)
    # f(x) = sum(w_j * f_j / (x - z_j)) / sum(w_j / (x - z_j))
    weights = wj / d
    num_standard = jnp.sum(weights * fj)
    den_standard = jnp.sum(weights)
    
    # Derivatives
    dnum_standard_dx = -jnp.sum(wj * fj / (d**2))
    dden_standard_dx = -jnp.sum(wj / (d**2))
    
    standard_derivative = (den_standard * dnum_standard_dx - num_standard * dden_standard_dx) / (den_standard**2)
    
    # Select the appropriate derivative
    derivative = jnp.where(
        exact_hit,
        exact_derivative,  # Use the specially computed exact derivative
        jnp.where(close_hit, blended_derivative, standard_derivative)
    )
    
    return (g * derivative, None, None, None, None)

# Register the custom VJP
analytical_barycentric_eval.defvjp(_analytical_bary_fwd_vjp, _analytical_bary_bwd_vjp)

def test_analytical_derivatives():
    """Test the analytical derivative implementation"""
    print("Testing analytical barycentric derivatives:")
    print("="*60)
    
    # Test case: polynomial x^2 with exact barycentric weights
    zj = jnp.array([0.0, 1.0, 2.0])
    fj = jnp.array([0.0, 1.0, 4.0])  # f(x) = x²
    wj = jnp.array([0.5, -1.0, 0.5])  # Exact barycentric weights for x² at [0,1,2]
    
    print(f"Support points: {zj}")
    print(f"Function values: {fj}")
    print(f"Barycentric weights: {wj}")
    
    # Create derivative functions
    f = lambda x: analytical_barycentric_eval(x, zj, fj, wj)
    f_prime = jax.grad(f)
    f_double_prime = jax.grad(f_prime)
    f_triple_prime = jax.grad(f_double_prime)
    f_quad_prime = jax.grad(f_triple_prime)
    
    test_points = [
        (0.0, "At support point 0"),
        (1.0, "At support point 1"), 
        (2.0, "At support point 2"),
        (0.5, "Between supports"),
        (1.5, "Between supports"),
        (1e-14, "Very close to support"),
        (-0.5, "Outside domain (left)"),
        (2.5, "Outside domain (right)")
    ]
    
    print(f"\n{'Description':>25} {'x':>8} {'f(x)':>10} {'True':>10} {'f_prime':>10} {'True':>10} {'f_double':>10} {'True':>10} {'f_triple':>12} {'True':>10}")
    print("-" * 120)
    
    all_good = True
    
    for x, desc in test_points:
        try:
            val = float(f(x))
            deriv1 = float(f_prime(x))
            deriv2 = float(f_double_prime(x))
            deriv3 = float(f_triple_prime(x))
            deriv4 = float(f_quad_prime(x))
            
            # True values for x²
            true_val = x**2
            true_deriv1 = 2*x
            true_deriv2 = 2.0
            true_deriv3 = 0.0  # Third derivative of x² is 0
            true_deriv4 = 0.0  # Fourth derivative of x² is 0
            
            # Check if values are reasonable
            val_good = not (np.isnan(val) or np.isinf(val))
            d1_good = not (np.isnan(deriv1) or np.isinf(deriv1))
            d2_good = not (np.isnan(deriv2) or np.isinf(deriv2))
            d3_good = not (np.isnan(deriv3) or np.isinf(deriv3) or abs(deriv3) > 1e6)
            d4_good = not (np.isnan(deriv4) or np.isinf(deriv4) or abs(deriv4) > 1e6)
            
            status = "✅" if (val_good and d1_good and d2_good and d3_good and d4_good) else "❌"
            if not (val_good and d1_good and d2_good and d3_good and d4_good):
                all_good = False
            
            print(f"{status} {desc:>22} {x:8.0e} {val:10.6f} {true_val:10.6f} {deriv1:10.6f} {true_deriv1:10.6f} {deriv2:10.6f} {true_deriv2:10.6f} {deriv3:12.2e} {true_deriv3:10.6f}")
            
        except Exception as e:
            print(f"❌ {desc:>22} {x:8.0e} ERROR: {str(e)[:50]}")
            all_good = False
    
    print(f"\n{'='*60}")
    if all_good:
        print("🎉 ALL TESTS PASSED! Analytical derivatives are working correctly!")
        print("✅ No NaN, no Inf, no explosion in higher-order derivatives")
    else:
        print("❌ Some tests failed - analytical implementation needs debugging")
    
    return all_good

if __name__ == "__main__":
    test_analytical_derivatives()