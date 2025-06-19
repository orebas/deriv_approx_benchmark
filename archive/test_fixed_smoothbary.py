#!/usr/bin/env python3
"""
Test the fixed AAA_SmoothBary directly
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

from scipy.optimize import minimize
from scipy.interpolate import AAA
import traceback

@jax.jit
def smooth_barycentric_eval_fixed(x, zj, fj, wj, tolerance=1e-8):
    """Fixed smooth version of barycentric evaluation"""
    eps = 1e-12
    
    # Calculate differences and fix zero-diff issue
    diffs = x - zj
    # Fix: add eps when |diffs| < eps instead of using sign multiplication
    safe_diffs = diffs + eps * (jnp.abs(diffs) < eps)
    
    # Clip inverse weights to prevent infinity
    MAX_W = 1.0 / eps
    weights_inv = jnp.clip(wj / safe_diffs, -MAX_W, MAX_W)
    
    # Standard barycentric formula (with regularization)
    num = jnp.sum(weights_inv * fj)
    den = jnp.sum(weights_inv)
    val_interp = num / (den + eps)
    
    # Calculate proximity weights using stable sigmoid
    dists_sq = diffs**2
    from jax.scipy.special import expit  # numerically stable sigmoid
    proximity_weights = expit(-(dists_sq / tolerance - 5.0))
    
    # Stabilize normalization
    sum_pw = jnp.sum(proximity_weights)
    proximity_weights = proximity_weights / (sum_pw + eps)
    
    # Weighted combination: close points get their exact values
    val_direct = jnp.sum(proximity_weights * fj)
    
    # Final smooth combination using stable sigmoid
    min_dist_sq = jnp.min(dists_sq)
    overall_activation = expit(-(min_dist_sq / tolerance - 3.0))
    
    return overall_activation * val_direct + (1.0 - overall_activation) * val_interp

def test_fixed_version():
    """Test the fixed smooth barycentric function"""
    print("Testing fixed smooth barycentric function...")
    
    np.random.seed(42)
    
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
        
        # Test single point evaluation
        x_test = 1.0
        result = smooth_barycentric_eval_fixed(x_test, zj, fj, wj)
        print(f"  Single evaluation at x={x_test}: {result}")
        
        # Test vectorized evaluation (this was failing before)
        vmap_eval = jax.vmap(lambda x: smooth_barycentric_eval_fixed(x, zj, fj, wj))
        results_vec = vmap_eval(t)
        print(f"  Vectorized evaluation successful: {len(results_vec)} points")
        print(f"  Any NaN in results: {jnp.any(jnp.isnan(results_vec))}")
        
        if not jnp.any(jnp.isnan(results_vec)):
            rmse = np.sqrt(np.mean((results_vec - y)**2))
            print(f"  RMSE: {rmse:.6f}")
            
            # Test objective function
            def test_objective(params):
                zj_new, fj_new, wj_new = jnp.split(params, 3)
                vmap_bary_eval = jax.vmap(
                    lambda x: smooth_barycentric_eval_fixed(x, zj_new, fj_new, wj_new, 1e-6)
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
            
            if not jnp.any(jnp.isnan(grad_obj)):
                print("  ✓ All tests passed - fixed version works!")
                return True
            else:
                print("  ✗ Gradient still has NaN")
                return False
        else:
            print("  ✗ Vectorized evaluation still produces NaN")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        return False

def test_smoothbary_class():
    """Test the AAA_SmoothBary class with fixed implementation"""
    print("\nTesting AAA_SmoothBarycentric_Approximator class...")
    
    try:
        from comprehensive_methods_library import AAA_SmoothBarycentric_Approximator
        
        np.random.seed(42)
        t = np.linspace(0, 2*np.pi, 31)
        y = np.sin(t) + 0.01 * np.random.randn(len(t))
        
        approximator = AAA_SmoothBarycentric_Approximator(t, y)
        approximator.fit()
        
        print(f"  Fitting success: {approximator.success}")
        
        if approximator.success:
            result = approximator.evaluate(t, max_derivative=2)
            print(f"  Evaluation successful")
            
            # Check for NaN values
            nan_counts = {
                'y': np.sum(np.isnan(result['y'])),
                'd1': np.sum(np.isnan(result['d1'])),
                'd2': np.sum(np.isnan(result['d2']))
            }
            print(f"  NaN counts: {nan_counts}")
            
            if nan_counts['y'] == 0:
                rmse = np.sqrt(np.mean((result['y'] - y)**2))
                print(f"  RMSE: {rmse:.6f}")
                print("  ✓ Class test passed!")
                return True
            else:
                print("  ✗ Class evaluation contains NaN")
                return False
        else:
            print("  ✗ Class fitting failed")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        return False

def main():
    print("Testing Fixed AAA_SmoothBary")
    print("=" * 40)
    
    # Test fixed function directly
    success1 = test_fixed_version()
    
    # Test class implementation
    success2 = test_smoothbary_class()
    
    print(f"\nResults:")
    print(f"  Fixed function test: {'PASS' if success1 else 'FAIL'}")
    print(f"  Class test: {'PASS' if success2 else 'FAIL'}")
    
    if success1 and success2:
        print("\n✓ AAA_SmoothBary is fixed and ready to use!")
    else:
        print("\n✗ Issues remain - needs more debugging")

if __name__ == "__main__":
    main()