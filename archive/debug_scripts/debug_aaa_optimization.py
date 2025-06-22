#!/usr/bin/env python3
"""
Debug script to investigate why AAA optimization is failing.
"""

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from comprehensive_methods_library import barycentric_eval

# Configure JAX to use 64-bit precision
from jax import config
config.update("jax_enable_x64", True)

def test_aaa_optimization():
    """Test the AAA optimization directly to see what's failing."""
    print("🔧 DEBUGGING AAA OPTIMIZATION")
    print("=" * 50)
    
    # Create simple test data
    t = np.linspace(0, 2*np.pi, 20)
    y = np.sin(t) + 0.01 * np.random.RandomState(42).randn(len(t))
    
    print(f"Test data: {len(t)} points, y range: [{y.min():.3f}, {y.max():.3f}]")
    
    # Create fallback support points (like our fixed code does)
    m_target = 5
    indices = np.linspace(0, len(t) - 1, m_target, dtype=int)
    zj = jnp.array(t[indices])
    fj_initial = jnp.array(y[indices])
    wj_initial = jnp.ones(m_target)
    
    print(f"\\nSupport points (zj): {zj}")
    print(f"Initial values (fj): {fj_initial}")
    print(f"Initial weights (wj): {wj_initial}")
    
    # Test the objective function
    y_scale = jnp.std(y)
    dt_scale = jnp.mean(jnp.diff(t)) ** 4
    lambda_reg = 1e-4 * y_scale * dt_scale if y_scale > 1e-9 else 1e-4 * dt_scale
    
    print(f"\\nRegularization parameters:")
    print(f"  y_scale: {y_scale}")
    print(f"  dt_scale: {dt_scale}")
    print(f"  lambda_reg: {lambda_reg}")
    
    def objective_func(params):
        fj, wj = jnp.split(params, 2)
        vmap_bary_eval = jax.vmap(lambda x: barycentric_eval(x, zj, fj, wj))
        y_pred = vmap_bary_eval(t)
        error_term = jnp.sum((y - y_pred)**2)
        
        # Compute smoothness term avoiding support points where d2 is undefined
        d2_func = jax.grad(jax.grad(lambda x: barycentric_eval(x, zj, fj, wj)))
        
        # Filter out evaluation points that are too close to support points
        def safe_d2(x):
            min_dist = jnp.min(jnp.abs(x - zj))
            is_safe = min_dist > 1e-10  # Safe distance threshold
            return jnp.where(is_safe, d2_func(x)**2, 0.0)
        
        d2_squared_values = jax.vmap(safe_d2)(t)
        smoothness_term = jnp.sum(d2_squared_values)
        return error_term + lambda_reg * smoothness_term
    
    # Test initial objective evaluation
    initial_params = jnp.concatenate([fj_initial, wj_initial])
    print(f"\\nInitial params shape: {initial_params.shape}")
    
    try:
        initial_obj = objective_func(initial_params)
        print(f"Initial objective value: {initial_obj}")
        
        if jnp.isnan(initial_obj) or jnp.isinf(initial_obj):
            print("❌ Initial objective is NaN/Inf!")
            return False
            
    except Exception as e:
        print(f"❌ Error evaluating initial objective: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test gradient computation
    try:
        objective_with_grad = jax.jit(jax.value_and_grad(objective_func))
        val, grad = objective_with_grad(initial_params)
        print(f"Initial gradient - val: {val}, grad shape: {grad.shape}")
        print(f"Gradient finite: {jnp.all(jnp.isfinite(grad))}")
        
        if not jnp.all(jnp.isfinite(grad)):
            print(f"Non-finite grad elements: {jnp.sum(~jnp.isfinite(grad))}")
            print(f"Sample grad values: {grad[:5]}")
            
    except Exception as e:
        print(f"❌ Error computing gradient: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test optimization
    def scipy_objective(params_flat):
        val, grad = objective_with_grad(params_flat)
        if jnp.isnan(val) or jnp.any(jnp.isnan(grad)):
            return np.inf, np.zeros_like(params_flat)
        return np.array(val, dtype=np.float64), np.array(grad, dtype=np.float64)
    
    print("\\n🔧 Testing scipy optimization...")
    try:
        result = minimize(
            scipy_objective, 
            initial_params, 
            method='L-BFGS-B', 
            jac=True,
            options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6, 'disp': True}
        )
        
        print(f"Optimization result:")
        print(f"  Success: {result.success}")
        print(f"  Message: {result.message}")
        print(f"  Final objective: {result.fun}")
        print(f"  Iterations: {result.nit}")
        
        if result.success:
            print("✅ Optimization succeeded!")
            return True
        else:
            print("❌ Optimization failed!")
            return False
            
    except Exception as e:
        print(f"❌ Optimization exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_aaa_optimization()
    print(f"\\n{'✅ SUCCESS' if success else '❌ FAILED'}")