#!/usr/bin/env python3
"""Debug AAA_FullOpt NaN issue."""

import numpy as np
import jax.numpy as jnp
from comprehensive_methods_library import AAA_FullOpt_Approximator

def debug_fullopt():
    """Debug why AAA_FullOpt produces NaN."""
    
    # Create simple test data
    t = np.linspace(0, 2*np.pi, 50)
    y = np.sin(t) + 0.001 * np.random.RandomState(42).randn(len(t))
    
    # Create and fit method
    method = AAA_FullOpt_Approximator(t, y)
    method.fit()
    
    if not method.fitted:
        print("❌ Failed to fit!")
        return
        
    print("✅ Fitted successfully")
    print(f"Support points (zj): {method.zj}")
    print(f"Support values (fj): {method.fj}")
    print(f"Weights (wj): {method.wj}")
    
    # Check for issues
    print(f"\nzj finite: {jnp.all(jnp.isfinite(method.zj))}")
    print(f"fj finite: {jnp.all(jnp.isfinite(method.fj))}")
    print(f"wj finite: {jnp.all(jnp.isfinite(method.wj))}")
    
    if not jnp.all(jnp.isfinite(method.zj)):
        print(f"Non-finite zj: {method.zj[~jnp.isfinite(method.zj)]}")
    if not jnp.all(jnp.isfinite(method.fj)):
        print(f"Non-finite fj: {method.fj[~jnp.isfinite(method.fj)]}")
    if not jnp.all(jnp.isfinite(method.wj)):
        print(f"Non-finite wj: {method.wj[~jnp.isfinite(method.wj)]}")
    
    # Test a single evaluation
    x_test = 1.0
    try:
        val = method.ad_derivatives[0](x_test)
        print(f"\nEvaluation at x={x_test}: {val}")
        print(f"Is finite: {jnp.isfinite(val)}")
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        
if __name__ == "__main__":
    debug_fullopt()