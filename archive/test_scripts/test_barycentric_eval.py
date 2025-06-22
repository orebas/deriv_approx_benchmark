#!/usr/bin/env python3
"""
Comprehensive tests for the safe barycentric evaluation function.
Tests numerical stability, exact support point evaluation, gradient computation,
and edge cases that were causing RMSE blowups in AAA methods.
"""

import numpy as np
import jax
import jax.numpy as jnp
# import pytest  # Not needed for direct execution
from comprehensive_methods_library import barycentric_eval

def test_exact_support_point():
    """Test that evaluation at exact support points returns the corresponding fj values."""
    zj = jnp.array([1.0, 2.0, 3.0])
    fj = jnp.array([10.0, 20.0, 30.0])
    wj = jnp.array([1.0, 1.0, 1.0])
    
    # Loop over support points
    for i in range(len(zj)):
        x = zj[i]
        result = barycentric_eval(x, zj, fj, wj)
        np.testing.assert_allclose(result, fj[i], rtol=1e-14, atol=1e-14,
            err_msg=f"Exact support point evaluation failed at index {i}")

def test_numerical_stability_near_support():
    """Test numerical stability when x is extremely close to a support point."""
    zj = jnp.array([1.0, 2.0, 3.0])
    fj = jnp.array([100.0, 200.0, 300.0])
    wj = jnp.array([1.0, 1.0, 1.0])
    
    # Test various levels of proximity to support points
    perturbations = [1e-15, 1e-16, 1e-17]
    
    for perturbation in perturbations:
        x = 2.0 + perturbation  # Very close to zj[1]
        result = barycentric_eval(x, zj, fj, wj)
        # Should return the exact support value because near condition triggers
        np.testing.assert_allclose(result, fj[1], rtol=1e-14, atol=1e-14,
            err_msg=f"Numerical stability failed for perturbation {perturbation}")

def test_gradient_computation():
    """Test that JAX autodiff works correctly without producing NaN or Inf."""
    zj = jnp.array([0.0, 1.0, 2.0])
    fj = jnp.array([0.0, 1.0, 4.0])  # Approximates f(x) = x^2
    wj = jnp.array([1.0, -2.0, 1.0])
    
    # Test gradient at multiple points
    test_points = [0.5, 1.5, 0.1, 1.9]
    
    for x in test_points:
        grad_bary = jax.grad(lambda x_val: barycentric_eval(x_val, zj, fj, wj))
        grad_value = grad_bary(x)
        
        # Check that the gradient is finite
        assert np.isfinite(grad_value), f"Gradient at x={x} is not finite: {grad_value}"
        
        # For f(x) ≈ x^2, derivative should be approximately 2x
        expected_grad = 2.0 * x
        np.testing.assert_allclose(grad_value, expected_grad, rtol=0.1,
            err_msg=f"Gradient at x={x} deviates significantly from expected")

def test_second_derivative_computation():
    """Test that second derivatives can be computed without numerical issues."""
    zj = jnp.array([0.0, 1.0, 2.0])
    fj = jnp.array([0.0, 1.0, 4.0])  # f(x) = x^2
    wj = jnp.array([1.0, -2.0, 1.0])
    
    # Test second derivative computation
    x = 1.5
    second_deriv_func = jax.grad(jax.grad(lambda x_val: barycentric_eval(x_val, zj, fj, wj)))
    second_deriv = second_deriv_func(x)
    
    assert np.isfinite(second_deriv), f"Second derivative is not finite: {second_deriv}"
    # For f(x) = x^2, second derivative should be approximately 2
    np.testing.assert_allclose(second_deriv, 2.0, rtol=0.2,
        err_msg="Second derivative deviates significantly from expected value")

def test_edge_large_and_small_weights():
    """Test robustness for extreme weight values."""
    zj = jnp.array([1.0, 2.0, 3.0])
    fj = jnp.array([5.0, 10.0, 15.0])
    # One weight extremely large, one extremely small
    wj = jnp.array([1e10, 1e-10, 1.0])
    
    # Evaluate at a point not near support points
    x = 2.5
    result = barycentric_eval(x, zj, fj, wj)
    
    # Result should be finite and reasonable
    assert np.isfinite(result), "Result with extreme weights is not finite"
    assert 0 <= result <= 20, f"Result {result} is outside reasonable range"

def test_clustered_support_points():
    """Test behavior with support points very close together."""
    base = 1.0
    spacing = 1e-12  # Extremely close spacing
    zj = jnp.array([base, base + spacing, base + 2*spacing])
    fj = jnp.array([10.0, 10.0 + spacing, 10.0 + 2*spacing])
    wj = jnp.array([1.0, 1.0, 1.0])
    
    # Evaluate away from cluster
    x = 2.0
    result = barycentric_eval(x, zj, fj, wj)
    
    # Result should be finite and within reasonable bounds
    assert np.isfinite(result), "Result with clustered support points is not finite"
    
    # Should be close to linear interpolation behavior
    expected_approx = 10.0 + (x - base)  # Linear approximation
    np.testing.assert_allclose(result, expected_approx, rtol=0.1,
        err_msg="Clustered support points produced unexpected result")

def test_regression_quadratic_behavior():
    """Regression test for known barycentric behavior with quadratic function."""
    # Support points for quadratic f(x) = x^2
    zj = jnp.array([-1.0, 0.0, 1.0])
    fj = jnp.array([1.0, 0.0, 1.0])  # f(x) = x^2 at support points
    wj = jnp.array([0.5, -1.0, 0.5])  # Typical barycentric weights
    
    # Evaluate at points away from support points
    x_eval = jnp.array([-0.8, -0.3, 0.3, 0.8])
    expected = x_eval**2
    
    # Compute barycentric evaluations
    results = jax.vmap(lambda x: barycentric_eval(x, zj, fj, wj))(x_eval)
    
    # Should approximate quadratic behavior reasonably well
    np.testing.assert_allclose(results, expected, rtol=1e-3, atol=1e-3,
        err_msg="Regression test failed: barycentric values don't match quadratic behavior")

def test_vectorized_evaluation():
    """Test that vectorized evaluation works correctly."""
    zj = jnp.array([0.0, 1.0, 2.0])
    fj = jnp.array([1.0, 2.0, 4.0])
    wj = jnp.array([1.0, -1.0, 1.0])
    
    # Vectorized evaluation
    x_vec = jnp.array([0.5, 1.5])
    vmap_eval = jax.vmap(lambda x: barycentric_eval(x, zj, fj, wj))
    results = vmap_eval(x_vec)
    
    # Compare with individual evaluations
    for i, x in enumerate(x_vec):
        individual_result = barycentric_eval(x, zj, fj, wj)
        np.testing.assert_allclose(results[i], individual_result, rtol=1e-14,
            err_msg=f"Vectorized evaluation differs from individual at x={x}")

def test_catastrophic_cancellation_prevention():
    """Test prevention of catastrophic cancellation in near-support scenarios."""
    # Create a scenario where the old implementation would fail
    zj = jnp.array([1.0, 1.000000000000001, 2.0])  # Very close first two points
    fj = jnp.array([10.0, 10.1, 20.0])
    wj = jnp.array([1e6, -1e6, 1.0])  # Large opposing weights
    
    # Evaluate between the close support points
    x = 1.0000000000000005
    result = barycentric_eval(x, zj, fj, wj)
    
    # Should not overflow or produce NaN
    assert np.isfinite(result), "Catastrophic cancellation test failed"
    assert abs(result) < 1e6, f"Result magnitude {abs(result)} suggests numerical instability"

def test_consistency_with_manual_computation():
    """Test consistency with manual barycentric computation for well-conditioned cases."""
    zj = jnp.array([0.0, 1.0, 3.0])
    fj = jnp.array([2.0, 5.0, 8.0])
    wj = jnp.array([1.0, 1.0, 1.0])
    
    x = 2.0  # Well away from support points
    
    # Manual computation
    diff = x - zj
    weights_over_diff = wj / diff
    num_manual = jnp.sum(fj * weights_over_diff)
    den_manual = jnp.sum(weights_over_diff)
    expected_manual = num_manual / den_manual
    
    # Our implementation
    result = barycentric_eval(x, zj, fj, wj)
    
    np.testing.assert_allclose(result, expected_manual, rtol=1e-12,
        err_msg="Implementation differs from manual computation")

if __name__ == "__main__":
    """Run all tests when executed directly."""
    import sys
    
    print("🔬 Running Safe Barycentric Evaluation Tests")
    print("=" * 60)
    
    test_functions = [
        test_exact_support_point,
        test_numerical_stability_near_support,
        test_gradient_computation,
        test_second_derivative_computation,
        test_edge_large_and_small_weights,
        test_clustered_support_points,
        test_regression_quadratic_behavior,
        test_vectorized_evaluation,
        test_catastrophic_cancellation_prevention,
        test_consistency_with_manual_computation
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            print(f"Running {test_func.__name__}...", end=" ")
            test_func()
            print("✅ PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        sys.exit(1)
    else:
        print("🎉 All tests passed! Safe barycentric evaluation is working correctly.")