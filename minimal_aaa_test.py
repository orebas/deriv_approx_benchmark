#!/usr/bin/env python3
"""Minimal test to debug AAA_FullOpt NaN issue."""

import numpy as np
import jax.numpy as jnp
from comprehensive_methods_library import AAA_FullOpt_Approximator, smooth_barycentric_eval

# Create simple test data
t = np.array([0., 1., 2., 3., 4.])
y = np.array([0., 1., 0., -1., 0.])

print("Test data:")
print(f"t = {t}")
print(f"y = {y}")

# Create and fit method
method = AAA_FullOpt_Approximator(t, y)
method.fit()

print(f"\nFitted: {method.fitted}")
print(f"Success: {method.success}")
print(f"zj: {method.zj}")
print(f"fj: {method.fj}")
print(f"wj: {method.wj}")

if method.success and method.zj is not None:
    # Test direct evaluation
    x_test = 1.5
    result = smooth_barycentric_eval(x_test, method.zj, method.fj, method.wj)
    print(f"\nDirect eval at {x_test}: {result}")
    print(f"Is finite: {jnp.isfinite(result)}")
    
    # Test through method
    results = method.evaluate(np.array([x_test]), max_derivative=1)
    print(f"\nMethod eval at {x_test}:")
    print(f"  y: {results.get('y', 'MISSING')}")
    print(f"  d1: {results.get('d1', 'MISSING')}")