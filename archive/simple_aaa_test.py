#!/usr/bin/env python3
"""
Simple test to verify AAA stable implementations work
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

# Test the smooth barycentric function
def test_smooth_barycentric():
    """Test that smooth barycentric evaluation works"""
    
    @jax.jit
    def smooth_barycentric_eval(x, zj, fj, wj, tolerance=1e-8):
        """Smooth version of barycentric evaluation"""
        # Standard barycentric formula
        num = jnp.sum(wj * fj / (x - zj))
        den = jnp.sum(wj / (x - zj))
        val_interp = num / (den + 1e-12)
        
        # Direct value at closest support point
        dists_sq = (x - zj)**2
        idx = jnp.argmin(dists_sq)
        val_direct = fj[idx]
        min_dist_sq = dists_sq[idx]
        
        # Smooth transition using tanh
        activation = 0.5 * (1.0 + jnp.tanh(-min_dist_sq / tolerance + 10))
        
        return activation * val_direct + (1 - activation) * val_interp
    
    # Create simple test case
    zj = jnp.array([0.0, 1.0, 2.0])
    fj = jnp.array([1.0, 0.0, 1.0])
    wj = jnp.array([1.0, 1.0, 1.0])
    
    # Test evaluation at various points
    test_points = jnp.array([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    
    print("Testing smooth barycentric evaluation:")
    for x in test_points:
        try:
            result = smooth_barycentric_eval(x, zj, fj, wj)
            print(f"  x={x:.1f}: result={result:.4f}")
        except Exception as e:
            print(f"  x={x:.1f}: ERROR - {e}")
    
    # Test gradient computation
    print("\nTesting gradient computation:")
    grad_func = jax.grad(lambda x: smooth_barycentric_eval(x, zj, fj, wj))
    
    for x in test_points:
        try:
            grad = grad_func(x)
            print(f"  x={x:.1f}: grad={grad:.4f}")
        except Exception as e:
            print(f"  x={x:.1f}: GRAD ERROR - {e}")
    
    # Test gradient w.r.t. support points
    print("\nTesting gradient w.r.t. support points:")
    def test_objective(params):
        zj_new, fj_new, wj_new = jnp.split(params, 3)
        x_eval = 0.5
        return smooth_barycentric_eval(x_eval, zj_new, fj_new, wj_new)
    
    params = jnp.concatenate([zj, fj, wj])
    try:
        grad_params = jax.grad(test_objective)(params)
        print(f"  Gradient w.r.t. all params: {grad_params}")
        print(f"  Support point grads: {grad_params[:3]}")
        print(f"  Function value grads: {grad_params[3:6]}")
        print(f"  Weight grads: {grad_params[6:]}")
    except Exception as e:
        print(f"  PARAM GRAD ERROR - {e}")


def test_two_stage_concept():
    """Test the two-stage concept with simple data"""
    
    # Create simple test data
    t = jnp.linspace(0, 1, 11)
    y = jnp.sin(t * 2 * jnp.pi) + 0.1 * np.random.randn(len(t))
    
    print("\nTesting two-stage concept:")
    print(f"Data: t={t}")
    print(f"Data: y={y}")
    
    # Simulate stage 1: Fixed support points
    from scipy.interpolate import AAA
    try:
        aaa_obj = AAA(t, y, max_terms=5)
        zj = jnp.array(aaa_obj.support_points)
        fj = jnp.array(aaa_obj.support_values)
        wj = jnp.array(aaa_obj.weights)
        
        print(f"AAA initialization successful:")
        print(f"  Support points: {zj}")
        print(f"  Function values: {fj}")
        print(f"  Weights: {wj}")
        
        # Test evaluation
        from comprehensive_methods_library import barycentric_eval
        test_point = 0.5
        result = barycentric_eval(test_point, zj, fj, wj)
        print(f"  Evaluation at x={test_point}: {result}")
        
    except Exception as e:
        print(f"AAA initialization failed: {e}")


if __name__ == "__main__":
    test_smooth_barycentric()
    test_two_stage_concept()