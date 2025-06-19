#!/usr/bin/env python3
"""
Targeted test for the convergence fix in AAA methods.
Tests the specific gradient corruption issue that was causing failures.
"""

import numpy as np
import jax.numpy as jnp
import jax
from comprehensive_methods_library import barycentric_eval, AAALeastSquaresApproximator

def test_gradient_stability():
    """Test that gradients remain finite even with extreme parameters."""
    print("Testing gradient stability with extreme parameters...")
    
    # Create a scenario that would previously cause gradient corruption
    zj = jnp.array([0.0, 1.0, 2.0])
    fj = jnp.array([1.0, 2.0, 4.0])
    wj = jnp.array([1e8, -2e8, 1e8])  # Large weights that could cause issues
    
    # Test evaluation very close to support points (would cause overflow before fix)
    x_test = 1.0 + 1e-14  # Extremely close to zj[1]
    
    try:
        # Test function evaluation
        result = barycentric_eval(x_test, zj, fj, wj)
        print(f"  Function evaluation: {result:.6f}")
        assert jnp.isfinite(result), "Function result is not finite"
        
        # Test gradient computation
        grad_func = jax.grad(lambda x: barycentric_eval(x, zj, fj, wj))
        grad_result = grad_func(x_test)
        print(f"  Gradient: {grad_result:.6f}")
        assert jnp.isfinite(grad_result), "Gradient is not finite"
        
        # Test second derivative (used in smoothness term)
        d2_func = jax.grad(jax.grad(lambda x: barycentric_eval(x, zj, fj, wj)))
        d2_result = d2_func(x_test)
        print(f"  Second derivative: {d2_result:.6f}")
        assert jnp.isfinite(d2_result), "Second derivative is not finite"
        
        print("✅ Gradient stability test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Gradient stability test FAILED: {e}")
        return False

def test_objective_function_robustness():
    """Test that the objective function in AAA_LS remains finite."""
    print("\nTesting objective function robustness...")
    
    # Simple test data
    t = jnp.linspace(0, 1, 10)
    y = jnp.sin(2 * jnp.pi * t)
    
    # Manually create what would be inside the optimization loop
    zj = jnp.array([0.0, 0.5, 1.0])
    fj = jnp.array([0.0, 1.0, 0.0])
    
    # Test with problematic weights that could cause convergence issues
    test_weights = [
        jnp.array([1.0, -2.0, 1.0]),  # Normal weights
        jnp.array([1e6, -2e6, 1e6]),  # Large weights
        jnp.array([1e-6, -2e-6, 1e-6]),  # Small weights
    ]
    
    dt_scale = jnp.mean(jnp.diff(t)) ** 4
    lambda_reg = 1e-4 * jnp.std(y) * dt_scale
    
    for i, wj in enumerate(test_weights):
        try:
            # Compute objective like in AAA_LS
            vmap_bary_eval = jax.vmap(lambda x: barycentric_eval(x, zj, fj, wj))
            y_pred = vmap_bary_eval(t)
            error_term = jnp.sum((y - y_pred)**2)
            
            d2_func = jax.grad(jax.grad(lambda x: barycentric_eval(x, zj, fj, wj)))
            d2_values = jax.vmap(d2_func)(t)
            smoothness_term = jnp.sum(d2_values**2)
            
            objective = error_term + lambda_reg * smoothness_term
            
            print(f"  Test {i+1}: objective={objective:.6e}, error={error_term:.6e}, smooth={smoothness_term:.6e}")
            
            assert jnp.isfinite(objective), f"Objective not finite for test {i+1}"
            assert jnp.isfinite(error_term), f"Error term not finite for test {i+1}"
            assert jnp.isfinite(smoothness_term), f"Smoothness term not finite for test {i+1}"
            
        except Exception as e:
            print(f"❌ Test {i+1} FAILED: {e}")
            return False
    
    print("✅ Objective function robustness test PASSED")
    return True

def test_simple_aaa_convergence():
    """Test AAA_LS with a simple, well-conditioned problem."""
    print("\nTesting simple AAA convergence...")
    
    # Very simple, well-conditioned test case
    t = np.linspace(0, 1, 15)
    y = t**2 + 0.01 * np.random.RandomState(42).randn(len(t))
    
    try:
        method = AAALeastSquaresApproximator(t, y)
        method.fit()
        
        if method.fitted and method.success:
            print(f"✅ AAA_LS converged successfully!")
            print(f"   Support points: {len(method.zj) if method.zj is not None else 0}")
            
            # Test evaluation
            t_eval = np.linspace(0, 1, 20)
            result = method.evaluate(t_eval)
            
            if result['success'] and np.all(np.isfinite(result['y'])):
                rmse = np.sqrt(np.mean((result['y'] - t_eval**2)**2))
                print(f"   RMSE: {rmse:.6f}")
                return True
            else:
                print("❌ Evaluation failed or produced non-finite values")
                return False
        else:
            print("❌ AAA_LS failed to converge")
            return False
            
    except Exception as e:
        print(f"❌ AAA_LS test failed with exception: {e}")
        return False

def main():
    """Run all convergence tests."""
    print("🔧 Testing AAA Convergence Fixes")
    print("=" * 50)
    
    test_results = [
        test_gradient_stability(),
        test_objective_function_robustness(),
        test_simple_aaa_convergence()
    ]
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All convergence fixes working correctly!")
    else:
        print("⚠️  Some issues remain")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    if not success:
        import sys
        sys.exit(1)