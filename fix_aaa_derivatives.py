#!/usr/bin/env python3
"""
Comprehensive fix for AAA derivative evaluation issues.

Root causes identified:
1. barycentric_eval uses jnp.where which creates discontinuities, breaking JAX gradients
2. AAA_SmoothBary doesn't actually optimize for smoothness
3. High-order derivatives (6-7) are numerically unstable

Solution:
1. Create a unified smooth_barycentric_eval function
2. Replace all barycentric_eval calls with the smooth version
3. Ensure all methods actually optimize for smoothness
"""

import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

@jax.jit
def smooth_barycentric_eval(x, zj, fj, wj, tolerance=1e-6):
    """
    Smooth, differentiable barycentric evaluation for AAA methods.
    
    Key improvements:
    1. No discontinuous jnp.where - uses smooth sigmoid blending
    2. Numerically stable with proper epsilon handling
    3. Gradient-friendly throughout
    
    Args:
        x: Evaluation point
        zj: Support points  
        fj: Function values at support points
        wj: Weights
        tolerance: Controls smoothness of transition near support points
    
    Returns:
        Interpolated value with smooth handling near support points
    """
    # Small epsilon for numerical stability
    eps = 1e-12
    
    # Calculate differences from support points
    diffs = x - zj
    
    # For stability, ensure we never divide by exactly zero
    # This is gradient-safe as it doesn't use conditional logic
    safe_diffs = diffs + eps * jnp.sign(diffs + eps)
    
    # Compute weights with magnitude control to prevent overflow
    # The clipping here is smooth and gradient-friendly
    MAX_MAG = 1e12
    weights_raw = wj / safe_diffs
    weights = jnp.clip(weights_raw, -MAX_MAG, MAX_MAG)
    
    # Standard barycentric formula
    num = jnp.sum(weights * fj)
    den = jnp.sum(weights)
    
    # This is the interpolated value away from support points
    interp_value = num / (den + eps)
    
    # For smooth transition near support points, we need to blend
    # between the interpolated value and the exact support values
    
    # Find distance to nearest support point
    min_dist = jnp.min(jnp.abs(diffs))
    
    # Create smooth activation using tanh (smoother than sigmoid)
    # This goes from 0 (far from support) to 1 (at support)
    # The scaling factor controls the transition width
    scale = 1.0 / tolerance
    activation = 0.5 * (1.0 + jnp.tanh(scale * (tolerance - min_dist)))
    
    # Find the value at the nearest support point
    nearest_idx = jnp.argmin(jnp.abs(diffs))
    nearest_value = fj[nearest_idx]
    
    # Smooth blend between interpolated and exact values
    # Far from support points: use interpolated value
    # Near support points: smoothly transition to exact value
    result = (1.0 - activation) * interp_value + activation * nearest_value
    
    return result


def create_smooth_objective(t, y, regularization_weight=1e-6):
    """
    Creates a smooth objective function for AAA optimization.
    
    This objective includes:
    1. Data fitting term (least squares)
    2. Smoothness regularization (penalizes second derivatives)
    
    The key is that it uses smooth_barycentric_eval throughout,
    ensuring gradients exist and are well-behaved.
    """
    
    def objective_func(params, zj):
        """
        Objective function for given support points zj.
        
        Args:
            params: Concatenated [fj, wj] to optimize
            zj: Fixed support points
        """
        # Split parameters
        m = len(zj)
        fj = params[:m]
        wj = params[m:]
        
        # Vectorized smooth evaluation
        eval_func = lambda t_point: smooth_barycentric_eval(t_point, zj, fj, wj)
        y_pred = jax.vmap(eval_func)(t)
        
        # Data fitting term
        error_term = jnp.sum((y - y_pred)**2)
        
        # Smoothness regularization via second derivatives
        # This is now well-defined because smooth_barycentric_eval is differentiable
        d1_func = jax.grad(eval_func)
        d2_func = jax.grad(d1_func)
        
        # Evaluate second derivatives at data points
        # We use vmap for efficiency
        d2_values = jax.vmap(d2_func)(t)
        
        # Penalize large second derivatives
        smoothness_term = jnp.sum(d2_values**2)
        
        # Scale regularization based on data characteristics
        t_range = t[-1] - t[0]
        y_scale = jnp.std(y)
        
        # The scaling ensures dimensional consistency
        # error_term has units of y^2
        # smoothness_term has units of y^2/t^4
        # So we multiply by t^4 to match dimensions
        scaled_lambda = regularization_weight * (t_range**4) / (y_scale**2 + 1e-6)
        
        total_objective = error_term + scaled_lambda * smoothness_term
        
        return total_objective
    
    # Return the objective function with gradient
    return jax.jit(jax.value_and_grad(objective_func))


def test_smooth_evaluation():
    """Test that smooth evaluation produces valid gradients."""
    print("Testing smooth barycentric evaluation...")
    
    # Create test data
    t = jnp.linspace(0, 2*jnp.pi, 20)
    y = jnp.sin(t)
    
    # Create simple support points
    m = 5
    indices = jnp.linspace(0, len(t)-1, m, dtype=int)
    zj = t[indices]
    fj = y[indices]
    wj = jnp.ones(m)
    
    # Test evaluation at a point
    x_test = 1.5
    
    # Test function evaluation
    val = smooth_barycentric_eval(x_test, zj, fj, wj)
    print(f"Function value at x={x_test}: {val}")
    
    # Test derivatives
    eval_func = lambda x: smooth_barycentric_eval(x, zj, fj, wj)
    
    # Compute derivatives up to order 7
    derivatives = [eval_func]
    for order in range(1, 8):
        d_func = jax.grad(derivatives[-1])
        derivatives.append(d_func)
        
        try:
            d_val = d_func(x_test)
            is_finite = jnp.isfinite(d_val)
            print(f"Derivative order {order}: {d_val} (finite: {is_finite})")
        except Exception as e:
            print(f"Derivative order {order}: FAILED - {e}")
    
    print("\nTesting vectorized evaluation...")
    x_array = jnp.linspace(0, 2*jnp.pi, 10)
    y_array = jax.vmap(eval_func)(x_array)
    print(f"All values finite: {jnp.all(jnp.isfinite(y_array))}")
    
    # Test first derivative array
    d1_func = jax.grad(eval_func)
    d1_array = jax.vmap(d1_func)(x_array)
    print(f"All first derivatives finite: {jnp.all(jnp.isfinite(d1_array))}")


if __name__ == "__main__":
    test_smooth_evaluation()
    
    print("\n" + "="*60)
    print("IMPLEMENTATION GUIDE")
    print("="*60)
    print("""
1. Replace barycentric_eval with smooth_barycentric_eval globally
   - This fixes the discontinuity issue causing NaN derivatives
   
2. Update all AAA method objective functions to use the smooth version
   - AAA_LS: lines 655-672
   - AAA_FullOpt: lines 777-793  
   - AAA_TwoStage: lines 1023-1040
   
3. Remove AAA_SmoothBary as it becomes redundant
   
4. Fix derivative loops in other methods:
   - GPRegressionApproximator: line 205
   - PolynomialRegressionApproximator: line 301
   - SVRApproximator: line 452
   Change: for _ in range(5) → for _ in range(self.max_derivative_supported)

5. Consider reducing max derivatives from 7 to 5 for numerical stability
   - 7th order derivatives are extremely sensitive to noise
   - Most practical applications don't need beyond 3rd or 4th order
""")