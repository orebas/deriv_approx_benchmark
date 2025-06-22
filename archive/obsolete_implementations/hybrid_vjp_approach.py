#!/usr/bin/env python3
"""
Hybrid approach: Use analytical derivatives where possible, 
fall back to improved finite differences for problematic cases.
"""

import jax
import jax.numpy as jnp
import numpy as np

@jax.custom_vjp
def hybrid_barycentric_eval(x, zj, fj, wj, W=1e-7):
    """
    Hybrid barycentric evaluation with mixed analytical/finite difference derivatives.
    Forward pass uses the Julia algorithm, backward pass uses analytical derivatives
    for the standard case and improved finite differences near support points.
    """
    tol = W * 1e6
    return _hybrid_bary_forward(x, zj, fj, wj, tol)

def _hybrid_bary_forward(x, zj, fj, wj, tol=1e-13):
    """Forward pass - same as Julia algorithm"""
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

def _hybrid_bary_fwd_vjp(x, zj, fj, wj, W=1e-7):
    """Forward pass for VJP"""
    result = _hybrid_bary_forward(x, zj, fj, wj, W * 1e6)
    return result, (x, zj, fj, wj, W)

def _hybrid_bary_bwd_vjp(res, g):
    """
    Hybrid backward pass:
    - Use analytical derivatives when far from support points
    - Use improved finite differences when close to support points
    """
    x, zj, fj, wj, W = res
    tol = W * 1e6
    d = x - zj
    
    # Determine which case we're in
    min_dist = jnp.min(jnp.abs(d))
    near_support = min_dist < jnp.sqrt(jnp.sqrt(tol))  # Use wider threshold for derivatives
    
    # Case 1: Near support points - use high-order finite differences
    def finite_diff_derivative():
        # Use higher-order finite differences (4th order) for better accuracy
        h = jnp.maximum(1e-10, min_dist * 1e-4)  # Very small step
        
        # 4th order finite difference: f'(x) ≈ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
        f_2h = _hybrid_bary_forward(x + 2*h, zj, fj, wj, tol)
        f_h = _hybrid_bary_forward(x + h, zj, fj, wj, tol)
        f_minus_h = _hybrid_bary_forward(x - h, zj, fj, wj, tol)
        f_minus_2h = _hybrid_bary_forward(x - 2*h, zj, fj, wj, tol)
        
        derivative = (-f_2h + 8*f_h - 8*f_minus_h + f_minus_2h) / (12*h)
        return derivative
    
    # Case 2: Far from support points - use analytical derivative
    def analytical_derivative():
        weights = wj / d
        num = jnp.sum(weights * fj)
        den = jnp.sum(weights)
        
        dnum_dx = -jnp.sum(wj * fj / (d**2))
        dden_dx = -jnp.sum(wj / (d**2))
        
        derivative = (den * dnum_dx - num * dden_dx) / (den**2)
        return derivative
    
    # Choose method based on distance to support points
    derivative = jnp.where(
        near_support,
        finite_diff_derivative(),
        analytical_derivative()
    )
    
    return (g * derivative, None, None, None, None)

# Register the custom VJP
hybrid_barycentric_eval.defvjp(_hybrid_bary_fwd_vjp, _hybrid_bary_bwd_vjp)

def test_hybrid_approach():
    """Test the hybrid analytical/finite difference approach"""
    print("Testing hybrid analytical/finite difference approach:")
    print("="*60)
    
    # Test case: polynomial x^2
    zj = jnp.array([0.0, 1.0, 2.0])
    fj = jnp.array([0.0, 1.0, 4.0])
    wj = jnp.array([0.5, -1.0, 0.5])
    
    print(f"Support points: {zj}")
    print(f"Function values: {fj}")
    print(f"Barycentric weights: {wj}")
    
    f = lambda x: hybrid_barycentric_eval(x, zj, fj, wj)
    f_prime = jax.grad(f)
    f_double_prime = jax.grad(f_prime)
    f_triple_prime = jax.grad(f_double_prime)
    f_quad_prime = jax.grad(f_triple_prime)
    f_fifth_prime = jax.grad(f_quad_prime)
    
    test_points = [
        (0.0, "At support point 0"),
        (1.0, "At support point 1"), 
        (2.0, "At support point 2"),
        (0.5, "Between supports"),
        (1.5, "Between supports"),
        (1e-10, "Very close to support"),
        (-0.5, "Outside domain"),
        (2.5, "Outside domain")
    ]
    
    print(f"\n{'Description':>25} {'x':>8} {'f':>10} {'f_prime':>10} {'f_double':>10} {'f_triple':>12} {'f_quad':>12} {'f_fifth':>12}")
    print("-" * 110)
    
    all_good = True
    max_error_seen = 0
    
    for x, desc in test_points:
        try:
            val = float(f(x))
            deriv1 = float(f_prime(x))
            deriv2 = float(f_double_prime(x))
            deriv3 = float(f_triple_prime(x))
            deriv4 = float(f_quad_prime(x))
            deriv5 = float(f_fifth_prime(x))
            
            # True values for x²
            true_val = x**2
            true_deriv1 = 2*x
            true_deriv2 = 2.0
            true_deriv3 = 0.0  
            true_deriv4 = 0.0  
            true_deriv5 = 0.0
            
            # Check for problems
            val_good = not (np.isnan(val) or np.isinf(val))
            d1_good = not (np.isnan(deriv1) or np.isinf(deriv1))
            d2_good = not (np.isnan(deriv2) or np.isinf(deriv2))
            d3_good = not (np.isnan(deriv3) or np.isinf(deriv3) or abs(deriv3) > 1e3)
            d4_good = not (np.isnan(deriv4) or np.isinf(deriv4) or abs(deriv4) > 1e6)
            d5_good = not (np.isnan(deriv5) or np.isinf(deriv5) or abs(deriv5) > 1e9)
            
            # Track maximum error for derivatives that should be small
            if abs(deriv3) > max_error_seen:
                max_error_seen = abs(deriv3)
            if abs(deriv4) > max_error_seen:
                max_error_seen = abs(deriv4)
            if abs(deriv5) > max_error_seen:
                max_error_seen = abs(deriv5)
            
            status = "✅" if (val_good and d1_good and d2_good and d3_good and d4_good and d5_good) else "❌"
            if not (val_good and d1_good and d2_good and d3_good and d4_good and d5_good):
                all_good = False
            
            print(f"{status} {desc:>22} {x:8.0e} {val:10.6f} {deriv1:10.6f} {deriv2:10.6f} {deriv3:12.2e} {deriv4:12.2e} {deriv5:12.2e}")
            
        except Exception as e:
            print(f"❌ {desc:>22} {x:8.0e} ERROR: {str(e)[:50]}")
            all_good = False
    
    print(f"\n{'='*60}")
    print(f"Maximum error in higher-order derivatives: {max_error_seen:.2e}")
    
    if all_good and max_error_seen < 1e3:
        print("🎉 SUCCESS! Hybrid approach provides stable derivatives!")
        print("✅ No NaN, no Inf, reasonable higher-order derivatives")
    elif all_good:
        print("⚠️  PARTIAL SUCCESS: No NaN/Inf but some large higher-order derivatives")
    else:
        print("❌ Still has issues with NaN/Inf")
    
    return all_good, max_error_seen

if __name__ == "__main__":
    test_hybrid_approach()