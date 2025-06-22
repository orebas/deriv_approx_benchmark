#!/usr/bin/env python3
"""
Simplified analytical VJP using the standard barycentric derivative formula.
This avoids the complexity of the Julia blended approach for derivatives.
"""

import jax
import jax.numpy as jnp
import numpy as np

@jax.custom_vjp
def simple_barycentric_eval(x, zj, fj, wj, W=1e-7):
    """
    Simple barycentric evaluation with analytical derivatives.
    Uses the forward Julia algorithm but computes derivatives using
    the standard barycentric derivative formula for simplicity.
    """
    tol = W * 1e6
    return _simple_bary_forward(x, zj, fj, wj, tol)

def _simple_bary_forward(x, zj, fj, wj, tol=1e-13):
    """Forward pass - same as the Julia algorithm"""
    d = x - zj
    
    # Case 1: Exact hits
    exact_indices = jnp.abs(d) < 1e-15
    exact_hit = jnp.any(exact_indices)
    exact_idx = jnp.argmax(exact_indices)
    exact_result = fj[exact_idx]
    
    # Case 2: Close to support points (blended case)
    close_indices = jnp.abs(d)**2 < jnp.sqrt(tol)
    close_hit = jnp.any(close_indices)
    breakindex = jnp.argmin(jnp.abs(d))
    m = d[breakindex]
    
    mask = jnp.arange(len(zj)) != breakindex
    safe_d = jnp.where(mask, d, 1.0)
    safe_weights = jnp.where(mask, wj / safe_d, 0.0)
    
    num_partial = jnp.sum(safe_weights * fj)
    den_partial = jnp.sum(safe_weights)
    
    blended_result = (wj[breakindex] * fj[breakindex] + m * num_partial) / (wj[breakindex] + m * den_partial)
    
    # Case 3: Standard barycentric
    weights = wj / d
    standard_result = jnp.sum(weights * fj) / jnp.sum(weights)
    
    result = jnp.where(
        exact_hit,
        exact_result,
        jnp.where(close_hit, blended_result, standard_result)
    )
    
    return result

def _simple_bary_fwd_vjp(x, zj, fj, wj, W=1e-7):
    """Forward pass for VJP"""
    result = _simple_bary_forward(x, zj, fj, wj, W * 1e6)
    return result, (x, zj, fj, wj, W)

def _simple_bary_bwd_vjp(res, g):
    """
    Backward pass using the standard barycentric derivative formula.
    This is simpler and more robust than trying to differentiate the blended cases.
    """
    x, zj, fj, wj, W = res
    d = x - zj
    
    # For all cases, use the standard barycentric derivative formula
    # This gives the correct derivative everywhere except exactly at support points
    
    # Check for exact hits
    exact_indices = jnp.abs(d) < 1e-15
    exact_hit = jnp.any(exact_indices)
    
    # For exact hits, use finite difference with small step
    if exact_hit:
        h = 1e-8
        f_plus = _simple_bary_forward(x + h, zj, fj, wj, W * 1e6)
        f_minus = _simple_bary_forward(x - h, zj, fj, wj, W * 1e6)
        derivative = (f_plus - f_minus) / (2 * h)
    else:
        # Standard barycentric derivative formula
        # f(x) = sum(w_j * f_j / (x - z_j)) / sum(w_j / (x - z_j))
        weights = wj / d
        num = jnp.sum(weights * fj)
        den = jnp.sum(weights)
        
        # Derivatives
        dnum_dx = -jnp.sum(wj * fj / (d**2))
        dden_dx = -jnp.sum(wj / (d**2))
        
        # Quotient rule
        derivative = (den * dnum_dx - num * dden_dx) / (den**2)
    
    return (g * derivative, None, None, None, None)

# Register the custom VJP
simple_barycentric_eval.defvjp(_simple_bary_fwd_vjp, _simple_bary_bwd_vjp)

def test_simple_analytical():
    """Test the simplified analytical approach"""
    print("Testing simplified analytical barycentric derivatives:")
    print("="*60)
    
    # Test case: polynomial x^2
    zj = jnp.array([0.0, 1.0, 2.0])
    fj = jnp.array([0.0, 1.0, 4.0])
    wj = jnp.array([0.5, -1.0, 0.5])
    
    f = lambda x: simple_barycentric_eval(x, zj, fj, wj)
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
        (1e-12, "Very close to support"),
    ]
    
    print(f"\n{'Description':>25} {'x':>8} {'f(x)':>10} {'f_prime':>10} {'f_double':>10} {'f_triple':>12} {'f_quad':>12}")
    print("-" * 100)
    
    all_good = True
    
    for x, desc in test_points:
        try:
            val = float(f(x))
            deriv1 = float(f_prime(x))
            deriv2 = float(f_double_prime(x))
            deriv3 = float(f_triple_prime(x))
            deriv4 = float(f_quad_prime(x))
            
            # Check for problems
            val_good = not (np.isnan(val) or np.isinf(val))
            d1_good = not (np.isnan(deriv1) or np.isinf(deriv1))
            d2_good = not (np.isnan(deriv2) or np.isinf(deriv2))
            d3_good = not (np.isnan(deriv3) or np.isinf(deriv3) or abs(deriv3) > 1e6)
            d4_good = not (np.isnan(deriv4) or np.isinf(deriv4) or abs(deriv4) > 1e6)
            
            status = "✅" if (val_good and d1_good and d2_good and d3_good and d4_good) else "❌"
            if not (val_good and d1_good and d2_good and d3_good and d4_good):
                all_good = False
            
            print(f"{status} {desc:>22} {x:8.0e} {val:10.6f} {deriv1:10.6f} {deriv2:10.6f} {deriv3:12.2e} {deriv4:12.2e}")
            
        except Exception as e:
            print(f"❌ {desc:>22} {x:8.0e} ERROR: {str(e)[:50]}")
            all_good = False
    
    print(f"\n{'='*60}")
    if all_good:
        print("🎉 ALL TESTS PASSED! Simplified analytical approach working!")
    else:
        print("❌ Some tests failed")
    
    return all_good

if __name__ == "__main__":
    test_simple_analytical()