#!/usr/bin/env python3
"""
Detailed debug of AAA_SmoothBary
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

import traceback
from scipy.optimize import minimize
from scipy.interpolate import AAA

# Test the smooth barycentric function directly
@jax.jit
def smooth_barycentric_eval(x, zj, fj, wj, tolerance=1e-8):
    """Smooth version of barycentric evaluation using tanh transition"""
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

def test_smooth_function():
    """Test the smooth barycentric function"""
    print("Testing smooth barycentric function...")
    
    # Create test data
    t = np.linspace(0, 2*np.pi, 21)
    y = np.sin(t) + 0.01 * np.random.randn(len(t))
    
    try:
        # Get initial AAA approximation
        aaa_obj = AAA(t, y, max_terms=5)
        zj = jnp.array(aaa_obj.support_points)
        fj = jnp.array(aaa_obj.support_values)
        wj = jnp.array(aaa_obj.weights)
        
        print(f"  AAA initialization: m={len(zj)}")
        print(f"  Support points: {zj}")
        print(f"  Function values: {fj}")
        print(f"  Weights: {wj}")
        
        # Test smooth evaluation at a point
        x_test = 1.0
        result = smooth_barycentric_eval(x_test, zj, fj, wj)
        print(f"  Evaluation at x={x_test}: {result}")
        
        # Test gradient computation
        grad_func = jax.grad(lambda x: smooth_barycentric_eval(x, zj, fj, wj))
        grad_result = grad_func(x_test)
        print(f"  Gradient at x={x_test}: {grad_result}")
        
        # Test objective function
        def test_objective(params):
            zj_new, fj_new, wj_new = jnp.split(params, 3)
            
            vmap_bary_eval = jax.vmap(
                lambda x: smooth_barycentric_eval(x, zj_new, fj_new, wj_new, 1e-8)
            )
            y_pred = vmap_bary_eval(t)
            error_term = jnp.sum((y - y_pred)**2)
            
            return error_term
            
        params = jnp.concatenate([zj, fj, wj])
        obj_val = test_objective(params)
        print(f"  Objective value: {obj_val}")
        
        # Test gradient of objective
        grad_obj = jax.grad(test_objective)(params)
        print(f"  Objective gradient norm: {jnp.linalg.norm(grad_obj)}")
        print(f"  Gradient has NaN: {jnp.any(jnp.isnan(grad_obj))}")
        
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        return False

def test_simplified_smoothbary():
    """Test a simplified version of AAA_SmoothBary"""
    print("\nTesting simplified AAA_SmoothBary...")
    
    # Create test data
    t = np.linspace(0, 2*np.pi, 21)
    y = np.sin(t) + 0.01 * np.random.randn(len(t))
    
    try:
        # Get initial AAA approximation - try just one model size
        aaa_obj = AAA(t, y, max_terms=5)
        zj_initial = jnp.array(aaa_obj.support_points)
        fj_initial = jnp.array(aaa_obj.support_values)
        wj_initial = jnp.array(aaa_obj.weights)
        
        print(f"  Initial model: m={len(zj_initial)}")
        
        # Simplified objective without regularization
        def simple_objective(params):
            zj, fj, wj = jnp.split(params, 3)
            
            vmap_bary_eval = jax.vmap(
                lambda x: smooth_barycentric_eval(x, zj, fj, wj, 1e-6)
            )
            y_pred = vmap_bary_eval(t)
            error_term = jnp.sum((y - y_pred)**2)
            
            return error_term
            
        objective_with_grad = jax.jit(jax.value_and_grad(simple_objective))
        
        def scipy_objective(params_flat):
            val, grad = objective_with_grad(params_flat)
            if jnp.isnan(val) or jnp.any(jnp.isnan(grad)):
                print(f"    NaN detected: val={val}, grad_nan_count={jnp.sum(jnp.isnan(grad))}")
                return np.inf, np.zeros_like(params_flat)
            return np.array(val, dtype=np.float64), np.array(grad, dtype=np.float64)
            
        initial_params = jnp.concatenate([zj_initial, fj_initial, wj_initial])
        
        print(f"  Testing initial objective...")
        init_val, init_grad = scipy_objective(initial_params)
        print(f"  Initial objective: {init_val}")
        print(f"  Initial gradient norm: {np.linalg.norm(init_grad)}")
        
        if init_val < np.inf:
            print(f"  Running optimization...")
            result = minimize(
                scipy_objective,
                initial_params,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            print(f"  Optimization result:")
            print(f"    Success: {result.success}")
            print(f"    Message: {result.message}")
            print(f"    Iterations: {result.nit}")
            print(f"    Final objective: {result.fun}")
            
            if result.success:
                # Test evaluation
                final_params = result.x
                zj_final, fj_final, wj_final = jnp.split(final_params, 3)
                
                y_pred_final = jax.vmap(
                    lambda x: smooth_barycentric_eval(x, zj_final, fj_final, wj_final, 1e-6)
                )(t)
                
                rmse = np.sqrt(np.mean((y_pred_final - y)**2))
                print(f"    Final RMSE: {rmse:.6f}")
                return True
            else:
                print(f"    Optimization failed")
                return False
        else:
            print(f"  Initial objective invalid")
            return False
            
    except Exception as e:
        print(f"  Error: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        return False

def main():
    print("Detailed AAA_SmoothBary Debug")
    print("=" * 40)
    
    np.random.seed(42)
    
    # Test smooth function
    success1 = test_smooth_function()
    
    # Test simplified version
    success2 = test_simplified_smoothbary()
    
    print(f"\nResults:")
    print(f"  Smooth function test: {'PASS' if success1 else 'FAIL'}")
    print(f"  Simplified optimization: {'PASS' if success2 else 'FAIL'}")

if __name__ == "__main__":
    main()