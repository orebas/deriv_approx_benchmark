#!/usr/bin/env python3
"""
Analyze the errors in v3 more carefully - are they actually problematic?
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

def naive_barycentric_eval(z, x, f, w):
    """Standard barycentric evaluation"""
    diffs = z - x
    weights = w / diffs
    num = jnp.sum(weights * f)
    den = jnp.sum(weights)
    return num / den

def smooth_barycentric_eval_v3(z, x, f, w, W=0.1):
    """Gemini's v3 smooth barycentric evaluation"""
    d = z - x
    d_sq = d**2
    alpha = jnp.tanh(d_sq / W)
    
    safe_far_term = jnp.nan_to_num(alpha / d, nan=0.0)
    N_far_unscaled = jnp.sum(safe_far_term * w * f)
    D_far_unscaled = jnp.sum(safe_far_term * w)
    
    one_minus_alpha = 1.0 - alpha
    N_close = jnp.sum(one_minus_alpha * w * f)
    D_close = jnp.sum(one_minus_alpha * w)
    
    d_scale = jnp.sum(one_minus_alpha * d)
    gamma = jnp.prod(alpha)
    
    N_final = N_close + d_scale * N_far_unscaled + gamma * N_far_unscaled
    D_final = D_close + d_scale * D_far_unscaled + gamma * D_far_unscaled
    
    return N_final / (D_final + 1e-30)

def analyze_error_context():
    """Put the errors in context - what kind of function are we approximating?"""
    print("ANALYZING ERROR CONTEXT")
    print("="*50)
    
    x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    f = jnp.array([0.0, 1.0, 4.0, 9.0, 16.0])  # x^2
    w = jnp.ones(5)
    
    print("Support points:", x)
    print("Function values:", f, "(x^2)")
    print()
    
    # The function being approximated has HUGE dynamic range
    z_test = jnp.array([0.5, 1.5, 2.5, 3.5])
    
    print("Understanding the function scale:")
    for z in z_test:
        naive_val = naive_barycentric_eval(z, x, f, w)
        smooth_val = smooth_barycentric_eval_v3(z, x, f, w)
        abs_error = abs(naive_val - smooth_val)
        rel_error = abs_error / abs(naive_val) if naive_val != 0 else 0
        
        print(f"z = {z}: naive = {naive_val:8.1f}, smooth = {smooth_val:8.1f}")
        print(f"         abs_error = {abs_error:6.2f}, rel_error = {rel_error*100:5.1f}%")
        print()

def test_smoother_interpolation():
    """Test with a smoother function to see if errors are due to the harsh function being approximated"""
    print("TESTING WITH SMOOTHER FUNCTION")
    print("="*50)
    
    x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    f = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Linear: f = x + 1
    w = jnp.ones(5)
    
    print("Support points:", x)
    print("Function values:", f, "(linear: x + 1)")
    print()
    
    z_test = jnp.array([0.5, 1.5, 2.5, 3.5])
    
    print("Testing with linear function:")
    max_error = 0
    for z in z_test:
        naive_val = naive_barycentric_eval(z, x, f, w)
        smooth_val = smooth_barycentric_eval_v3(z, x, f, w)
        abs_error = abs(naive_val - smooth_val)
        rel_error = abs_error / abs(naive_val) if naive_val != 0 else 0
        max_error = max(max_error, abs_error)
        
        print(f"z = {z}: naive = {naive_val:8.4f}, smooth = {smooth_val:8.4f}")
        print(f"         abs_error = {abs_error:8.6f}, rel_error = {rel_error*100:6.3f}%")
        print()
    
    print(f"Maximum absolute error: {max_error:.6f}")
    print()

def test_different_W_values():
    """See how error varies with W parameter"""
    print("TESTING DIFFERENT W VALUES")
    print("="*50)
    
    x = jnp.array([0.0, 1.0, 2.0])
    f = jnp.array([0.0, 1.0, 4.0])  # Simple quadratic
    w = jnp.ones(3)
    z = 0.5
    
    naive_val = naive_barycentric_eval(z, x, f, w)
    print(f"Naive result at z = {z}: {naive_val}")
    print()
    
    W_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    
    print("W value effects:")
    for W in W_values:
        smooth_val = smooth_barycentric_eval_v3(z, x, f, w, W=W)
        error = abs(smooth_val - naive_val)
        rel_error = error / abs(naive_val) * 100
        
        print(f"W = {W:4.2f}: smooth = {smooth_val:8.4f}, error = {error:6.4f} ({rel_error:5.2f}%)")
    
    print()

def compare_to_typical_noise():
    """Compare approximation error to typical noise levels in data"""
    print("COMPARING TO TYPICAL NOISE LEVELS")
    print("="*50)
    
    # Simulate what happens with noisy data
    x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    f_clean = jnp.array([0.0, 1.0, 4.0, 9.0, 16.0])
    
    # Add 1% noise (typical benchmark noise level)
    np.random.seed(42)
    noise = 0.01 * np.mean(np.abs(f_clean)) * np.random.randn(len(f_clean))
    f_noisy = f_clean + noise
    
    w = jnp.ones(5)
    z = 0.5
    
    print("Clean function values:", f_clean)
    print("Noisy function values:", f_noisy)
    print("Noise level:", np.abs(noise))
    print()
    
    # Compare different sources of error
    clean_naive = naive_barycentric_eval(z, x, f_clean, w)
    clean_smooth = smooth_barycentric_eval_v3(z, x, f_clean, w)
    noisy_naive = naive_barycentric_eval(z, x, f_noisy, w)
    
    approx_error = abs(clean_smooth - clean_naive)
    noise_error = abs(noisy_naive - clean_naive)
    
    print(f"Clean naive result: {clean_naive:.4f}")
    print(f"Clean smooth result: {clean_smooth:.4f}")
    print(f"Noisy naive result: {noisy_naive:.4f}")
    print()
    print(f"Approximation error (smooth vs naive): {approx_error:.4f}")
    print(f"Noise-induced error: {noise_error:.4f}")
    print()
    print(f"Approximation error is {approx_error/noise_error:.1f}x the noise error")

if __name__ == "__main__":
    analyze_error_context()
    test_smoother_interpolation()
    test_different_W_values()
    compare_to_typical_noise()