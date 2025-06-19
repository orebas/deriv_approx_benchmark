#!/usr/bin/env python3
"""
Quantify the accuracy difference between original tanh formula and rational approximation
"""

import numpy as np
import jax.numpy as jnp
import jax
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# Original smooth barycentric with tanh (but avoiding NaN at d=0)
def smooth_bary_original(x, zj, fj, wj, W=1e-7):
    """Original with tanh transitions"""
    d = x - zj
    d_sq = d**2
    alpha = jnp.tanh(d_sq / W)
    
    # Original formula but handle d=0 case
    safe_far_term = jnp.where(
        jnp.abs(d) < 1e-15,
        0.0,  # At support point, this term contributes 0
        alpha / d
    )
    
    N_far_unscaled = jnp.sum(safe_far_term * wj * fj)
    D_far_unscaled = jnp.sum(safe_far_term * wj)
    
    one_minus_alpha = 1.0 - alpha
    N_close = jnp.sum(one_minus_alpha * wj * fj)
    D_close = jnp.sum(one_minus_alpha * wj)
    
    d_scale = jnp.sum(one_minus_alpha * d)
    gamma = jnp.prod(alpha)
    
    N_final = N_close + d_scale * N_far_unscaled + gamma * N_far_unscaled
    D_final = D_close + d_scale * D_far_unscaled + gamma * D_far_unscaled
    
    return N_final / (D_final + 1e-30)

# New rational approximation
def smooth_bary_rational(x, zj, fj, wj, W=1e-7):
    """Rational approximation"""
    d = x - zj
    d_sq = d**2
    alpha = jnp.tanh(d_sq / W)
    
    # Rational approximation
    safe_far_term = d / (W + d_sq)
    
    N_far_unscaled = jnp.sum(safe_far_term * wj * fj)
    D_far_unscaled = jnp.sum(safe_far_term * wj)
    
    one_minus_alpha = 1.0 - alpha
    N_close = jnp.sum(one_minus_alpha * wj * fj)
    D_close = jnp.sum(one_minus_alpha * wj)
    
    d_scale = jnp.sum(one_minus_alpha * d)
    gamma = jnp.prod(alpha)
    
    N_final = N_close + d_scale * N_far_unscaled + gamma * N_far_unscaled
    D_final = D_close + d_scale * D_far_unscaled + gamma * D_far_unscaled
    
    return N_final / (D_final + 1e-30)

def test_on_known_functions():
    """Test accuracy on various known functions"""
    
    print("="*80)
    print("ACCURACY COMPARISON: Original tanh vs Rational approximation")
    print("="*80)
    
    # Test functions
    test_cases = [
        ("sin(x)", lambda x: np.sin(x), 0, 2*np.pi, 20),
        ("exp(x)", lambda x: np.exp(x), 0, 2, 15),
        ("1/(1+x²)", lambda x: 1/(1+x**2), -3, 3, 20),
        ("x³-2x", lambda x: x**3 - 2*x, -2, 2, 15),
    ]
    
    results = []
    
    for name, func, a, b, n_points in test_cases:
        print(f"\nTest function: {name}")
        print(f"Interval: [{a}, {b}], Support points: {n_points}")
        
        # Create support points
        t_support = np.linspace(a, b, n_points)
        y_support = func(t_support)
        
        # Simple uniform weights
        weights = np.ones(n_points)
        
        # Convert to JAX arrays
        zj = jnp.array(t_support)
        fj = jnp.array(y_support)
        wj = jnp.array(weights)
        
        # Test points (dense, including near support points)
        t_test = np.linspace(a, b, 500)
        y_true = func(t_test)
        
        # Evaluate both methods
        y_orig = np.array([smooth_bary_original(t, zj, fj, wj) for t in t_test])
        y_rat = np.array([smooth_bary_rational(t, zj, fj, wj) for t in t_test])
        
        # Calculate errors
        err_orig = np.abs(y_orig - y_true)
        err_rat = np.abs(y_rat - y_true)
        
        # Metrics
        rmse_orig = np.sqrt(np.mean(err_orig**2))
        rmse_rat = np.sqrt(np.mean(err_rat**2))
        max_err_orig = np.max(err_orig)
        max_err_rat = np.max(err_rat)
        
        # Relative to function range
        y_range = np.max(y_true) - np.min(y_true)
        rel_rmse_orig = rmse_orig / y_range * 100
        rel_rmse_rat = rmse_rat / y_range * 100
        
        print(f"  RMSE:      Original: {rmse_orig:.2e}, Rational: {rmse_rat:.2e}")
        print(f"  Max Error: Original: {max_err_orig:.2e}, Rational: {max_err_rat:.2e}")
        print(f"  Relative RMSE: Original: {rel_rmse_orig:.2f}%, Rational: {rel_rmse_rat:.2f}%")
        print(f"  Accuracy ratio (Rational/Original): {rmse_rat/rmse_orig:.3f}")
        
        results.append({
            'name': name,
            'rmse_orig': rmse_orig,
            'rmse_rat': rmse_rat,
            'ratio': rmse_rat/rmse_orig
        })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    avg_ratio = np.mean([r['ratio'] for r in results])
    print(f"Average accuracy ratio (Rational/Original): {avg_ratio:.3f}")
    
    if avg_ratio < 1.5:
        print("✅ Rational approximation maintains good accuracy (within 50% of original)")
    elif avg_ratio < 2.0:
        print("⚠️  Rational approximation is less accurate but still reasonable")
    else:
        print("❌ Rational approximation loses significant accuracy")
    
    # Test near support points specifically
    print("\n" + "="*80)
    print("ACCURACY NEAR SUPPORT POINTS")
    print("="*80)
    
    # Use a simple test case
    zj = jnp.array([0.0, 1.0, 2.0, 3.0])
    fj = jnp.array([0.0, 1.0, 4.0, 9.0])  # f(x) = x²
    wj = jnp.ones(4)
    
    # Test very close to support points
    for i, z in enumerate(zj):
        print(f"\nNear support point z[{i}] = {z}:")
        
        offsets = [0, 1e-10, 1e-8, 1e-6, 1e-4]
        for offset in offsets:
            x = float(z) + offset
            true_val = x**2
            orig_val = float(smooth_bary_original(x, zj, fj, wj))
            rat_val = float(smooth_bary_rational(x, zj, fj, wj))
            
            err_orig = abs(orig_val - true_val)
            err_rat = abs(rat_val - true_val)
            
            print(f"  Offset {offset:1.0e}: Orig err: {err_orig:1.2e}, Rat err: {err_rat:1.2e}, Ratio: {err_rat/max(err_orig, 1e-15):.2f}")

if __name__ == "__main__":
    test_on_known_functions()