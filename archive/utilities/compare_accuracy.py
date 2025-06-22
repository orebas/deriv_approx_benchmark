#!/usr/bin/env python3
"""Compare accuracy between our implementation and vanilla scipy AAA"""

import numpy as np
from scipy.interpolate import AAA
from comprehensive_methods_library import smooth_barycentric_eval
import jax.numpy as jnp

def compare_accuracy():
    print("Comparing accuracy: Our implementation vs Vanilla AAA")
    print("="*60)
    
    # Test on clean sine data (no noise)
    t = np.linspace(0, 2*np.pi, 30)
    y_true = np.sin(t)
    
    # Get AAA approximation
    aaa = AAA(t, y_true, max_terms=10)
    zj = jnp.array(aaa.support_points)
    fj = jnp.array(aaa.support_values) 
    wj = jnp.array(aaa.weights)
    
    print(f"AAA found {len(zj)} support points")
    print(f"Max terms requested: 10")
    
    # Test on denser grid
    t_test = np.linspace(0, 2*np.pi, 100)
    y_test_true = np.sin(t_test)
    
    # Vanilla AAA evaluation
    y_vanilla = aaa(t_test)
    
    # Our implementation
    y_ours = np.array([float(smooth_barycentric_eval(x, zj, fj, wj)) for x in t_test])
    
    # Compute errors
    error_vanilla = np.abs(y_vanilla - y_test_true)
    error_ours = np.abs(y_ours - y_test_true)
    
    print(f"\nAccuracy comparison on sine function:")
    print(f"Vanilla AAA RMSE: {np.sqrt(np.mean(error_vanilla**2)):.2e}")
    print(f"Our implementation RMSE: {np.sqrt(np.mean(error_ours**2)):.2e}")
    print(f"Vanilla AAA max error: {np.max(error_vanilla):.2e}")
    print(f"Our implementation max error: {np.max(error_ours):.2e}")
    
    # Check if they're essentially the same
    diff = np.abs(y_vanilla - y_ours)
    print(f"\nDifference between implementations:")
    print(f"Max difference: {np.max(diff):.2e}")
    print(f"Mean difference: {np.mean(diff):.2e}")
    
    if np.max(diff) < 1e-10:
        print("✅ Implementations are essentially identical")
    else:
        print("⚠️  Implementations differ - investigating...")
        
        # Find where they differ most
        worst_idx = np.argmax(diff)
        print(f"Worst difference at t={t_test[worst_idx]:.4f}:")
        print(f"  Vanilla: {y_vanilla[worst_idx]:.10f}")
        print(f"  Ours:    {y_ours[worst_idx]:.10f}")
        print(f"  True:    {y_test_true[worst_idx]:.10f}")

if __name__ == "__main__":
    compare_accuracy()