#!/usr/bin/env python3
"""
Test script for safe barycentric evaluation to verify numerical stability
before implementing in the main codebase.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt

# Current implementation (problematic)
@jax.jit
def barycentric_eval_current(x, zj, fj, wj):
    """Current implementation that may have near-support point issues"""
    is_support_point = jnp.any(jnp.isclose(x, zj))
    
    def true_fn():
        idx = jnp.argmin(jnp.abs(x - zj))
        return fj[idx]

    def false_fn():
        num = jnp.sum(wj * fj / (x - zj))
        den = jnp.sum(wj / (x - zj))
        return num / (den + 1e-12)

    return jax.lax.cond(is_support_point, true_fn, false_fn)

# Proposed safe implementation  
@jax.jit
def barycentric_eval_safe(x, zj, fj, wj, rtol=1e-14, atol=1e-14):
    """Safe barycentric evaluation avoiding division by near-zero"""
    diff = x - zj
    # Per-element mask for points near support points
    near = jnp.abs(diff) <= (atol + rtol * jnp.abs(zj))
    
    # Avoid division by near-zero: set to 0 where near, compute 1/diff elsewhere
    inv = jnp.where(near, 0.0, 1.0 / diff)
    
    # Compute barycentric formula with safe denominators
    num = jnp.sum(wj * fj * inv)
    den = jnp.sum(wj * inv)
    
    # If any point is near a support point, return exact value
    result_exact = fj[jnp.argmax(near)]  # Value at first near support point
    
    # Scale epsilon properly for denominator safety
    eps = jnp.finfo(den.dtype).eps
    safe_den = den + eps * jnp.maximum(1.0, jnp.abs(den))
    
    return jnp.where(jnp.any(near), result_exact, num / safe_den)

def test_basic_functionality():
    """Test basic barycentric evaluation"""
    print("Testing basic functionality...")
    
    # Simple test case
    zj = jnp.array([0.0, 1.0, 2.0])
    fj = jnp.array([1.0, 4.0, 9.0])  # f(x) = x^2 + 1 at support points
    wj = jnp.array([1.0, -2.0, 1.0])  # Standard AAA weights
    
    # Test at support points
    for i, z in enumerate(zj):
        result_current = barycentric_eval_current(z, zj, fj, wj)
        result_safe = barycentric_eval_safe(z, zj, fj, wj)
        expected = fj[i]
        
        print(f"  At support point z={z}: current={result_current:.6f}, safe={result_safe:.6f}, expected={expected:.6f}")
        assert jnp.abs(result_current - expected) < 1e-12
        assert jnp.abs(result_safe - expected) < 1e-12
    
    # Test at non-support points
    test_x = jnp.array([0.5, 1.5])
    for x in test_x:
        result_current = barycentric_eval_current(x, zj, fj, wj)
        result_safe = barycentric_eval_safe(x, zj, fj, wj)
        
        print(f"  At x={x}: current={result_current:.6f}, safe={result_safe:.6f}")
        # Should be close to each other for well-conditioned cases
        assert jnp.abs(result_current - result_safe) < 1e-10

def test_near_support_catastrophe():
    """Test the critical near-support point catastrophe scenario"""
    print("\nTesting near-support point catastrophe...")
    
    # Create a scenario where points are very close to support points
    zj = jnp.array([0.0, 1.0, 2.0])
    fj = jnp.array([1.0, 4.0, 9.0]) 
    wj = jnp.array([1.0, -2.0, 1.0])
    
    # Test points that are numerically close but outside jnp.isclose tolerance
    eps_vals = [1e-16, 1e-15, 1e-14, 1e-13, 1e-12]
    
    for eps in eps_vals:
        x_near = 1.0 + eps  # Very close to support point at 1.0
        
        try:
            result_current = barycentric_eval_current(x_near, zj, fj, wj)
            result_safe = barycentric_eval_safe(x_near, zj, fj, wj)
            
            print(f"  eps={eps:1.0e}: current={result_current:.6f}, safe={result_safe:.6f}")
            
            # Check for inf/nan in current implementation
            if not jnp.isfinite(result_current):
                print(f"    ⚠️  Current implementation produced non-finite result!")
            
            # Safe implementation should always be finite and close to support value
            assert jnp.isfinite(result_safe)
            assert jnp.abs(result_safe - 4.0) < 0.1  # Should be close to f(1) = 4
            
        except Exception as e:
            print(f"    ❌ Exception at eps={eps:1.0e}: {e}")

def test_derivative_computation():
    """Test automatic differentiation through both implementations"""
    print("\nTesting automatic differentiation...")
    
    zj = jnp.array([0.0, 1.0, 2.0])
    fj = jnp.array([0.0, 1.0, 4.0])  # Approximately f(x) = x^2
    wj = jnp.array([1.0, -2.0, 1.0])
    
    # Test derivative computation at various points
    test_points = jnp.array([0.1, 0.5, 0.9, 1.1, 1.5, 1.9])
    
    for x in test_points:
        try:
            # Compute derivatives
            grad_current = jax.grad(barycentric_eval_current)(x, zj, fj, wj)
            grad_safe = jax.grad(barycentric_eval_safe)(x, zj, fj, wj)
            
            print(f"  x={x:.1f}: grad_current={grad_current:.6f}, grad_safe={grad_safe:.6f}")
            
            # Check for nan/inf in gradients
            if not jnp.isfinite(grad_current):
                print(f"    ⚠️  Current gradient is non-finite!")
            if not jnp.isfinite(grad_safe):
                print(f"    ⚠️  Safe gradient is non-finite!")
                
        except Exception as e:
            print(f"    ❌ Exception computing derivative at x={x:.1f}: {e}")

def test_extreme_cases():
    """Test extreme cases that might cause numerical issues"""
    print("\nTesting extreme cases...")
    
    # Case 1: Very large weights
    zj = jnp.array([0.0, 1.0, 2.0])
    fj = jnp.array([1.0, 2.0, 3.0])
    wj_large = jnp.array([1e10, -2e10, 1e10])
    
    x_test = 0.5
    try:
        result_current = barycentric_eval_current(x_test, zj, fj, wj_large)
        result_safe = barycentric_eval_safe(x_test, zj, fj, wj_large)
        print(f"  Large weights: current={result_current:.6f}, safe={result_safe:.6f}")
    except Exception as e:
        print(f"    ❌ Large weights failed: {e}")
    
    # Case 2: Very small weights  
    wj_small = jnp.array([1e-10, -2e-10, 1e-10])
    try:
        result_current = barycentric_eval_current(x_test, zj, fj, wj_small)
        result_safe = barycentric_eval_safe(x_test, zj, fj, wj_small)
        print(f"  Small weights: current={result_current:.6f}, safe={result_safe:.6f}")
    except Exception as e:
        print(f"    ❌ Small weights failed: {e}")

if __name__ == "__main__":
    print("🔬 Testing Safe Barycentric Evaluation Implementation")
    print("=" * 60)
    
    test_basic_functionality()
    test_near_support_catastrophe() 
    test_derivative_computation()
    test_extreme_cases()
    
    print("\n✅ All tests completed!")
    print("\nIf no errors were reported, the safe implementation is ready for deployment.")