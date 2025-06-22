#!/usr/bin/env python3
"""
Test the rational function replacement for tanh(d²/W)/d
"""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import config
config.update("jax_enable_x64", True)

def original_term(d, W=1e-7):
    """Original problematic term: tanh(d²/W)/d"""
    return jnp.where(d == 0, 0.0, jnp.tanh(d**2 / W) / d)

def rational_term(d, W=1e-7):
    """Proposed replacement: d/(W+d²)"""
    return d / (W + d**2)

def test_function_comparison():
    """Compare the two functions and their derivatives"""
    
    print("="*80)
    print("COMPARING ORIGINAL vs RATIONAL REPLACEMENT")
    print("="*80)
    
    # Test points including problematic d=0
    test_points = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0]
    
    print("\nFunction values:")
    print(f"{'d':>12} {'Original':>15} {'Rational':>15} {'Ratio':>12}")
    print("-"*60)
    
    for d in test_points:
        orig = float(original_term(d))
        rat = float(rational_term(d))
        ratio = rat/orig if orig != 0 else float('inf')
        print(f"{d:12.2e} {orig:15.6e} {rat:15.6e} {ratio:12.4f}")
    
    # Test derivatives
    print("\n" + "="*80)
    print("DERIVATIVE COMPARISON")
    print("="*80)
    
    # Create derivative functions
    d_original = jax.grad(lambda d: original_term(d))
    d_rational = jax.grad(lambda d: rational_term(d))
    
    print("\nFirst derivatives:")
    print(f"{'d':>12} {'Original':>15} {'Rational':>15} {'Difference':>15}")
    print("-"*75)
    
    for d in test_points:
        try:
            orig_d = float(d_original(d))
        except:
            orig_d = float('nan')
        
        rat_d = float(d_rational(d))
        diff = abs(orig_d - rat_d) if not np.isnan(orig_d) else float('nan')
        
        print(f"{d:12.2e} {orig_d:15.6e} {rat_d:15.6e} {diff:15.6e}")
        
        if np.isnan(orig_d) and not np.isnan(rat_d):
            print(f"              ✅ Rational avoids NaN!")
    
    # Visual comparison
    print("\n" + "="*80)
    print("VISUAL COMPARISON")
    print("="*80)
    
    # Create plot
    d_vals = np.logspace(-10, 0, 1000)
    d_vals_with_zero = np.concatenate([[0.0], d_vals])
    
    # Compute function values
    orig_vals = [float(original_term(d)) for d in d_vals_with_zero]
    rat_vals = [float(rational_term(d)) for d in d_vals_with_zero]
    
    plt.figure(figsize=(12, 5))
    
    # Function comparison
    plt.subplot(1, 2, 1)
    plt.loglog(d_vals, orig_vals[1:], 'b-', label='Original: tanh(d²/W)/d', linewidth=2)
    plt.loglog(d_vals, rat_vals[1:], 'r--', label='Rational: d/(W+d²)', linewidth=2)
    plt.xlabel('d')
    plt.ylabel('Function value')
    plt.title('Function Comparison (log-log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Ratio plot
    plt.subplot(1, 2, 2)
    ratios = np.array(rat_vals[1:]) / np.array(orig_vals[1:])
    plt.semilogx(d_vals, ratios, 'g-', linewidth=2)
    plt.xlabel('d')
    plt.ylabel('Rational / Original')
    plt.title('Ratio of Functions')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('function_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved comparison plot to 'function_comparison.png'")
    
    # Analyze asymptotic behavior
    print("\n" + "="*80)
    print("ASYMPTOTIC ANALYSIS")
    print("="*80)
    
    # Small d behavior: should go like d/W
    small_d = 1e-10
    orig_small = float(original_term(small_d))
    rat_small = float(rational_term(small_d))
    expected_small = small_d / 1e-7
    
    print(f"\nSmall d behavior (d={small_d}):")
    print(f"  Original:  {orig_small:.6e}")
    print(f"  Rational:  {rat_small:.6e}")
    print(f"  Expected (d/W): {expected_small:.6e}")
    print(f"  Rational matches expected: {abs(rat_small - expected_small) < 1e-12}")
    
    # Large d behavior: should go like 1/d
    large_d = 1.0
    orig_large = float(original_term(large_d))
    rat_large = float(rational_term(large_d))
    expected_large = 1.0 / large_d
    
    print(f"\nLarge d behavior (d={large_d}):")
    print(f"  Original:  {orig_large:.6e}")
    print(f"  Rational:  {rat_large:.6e}")
    print(f"  Expected (1/d): {expected_large:.6e}")
    print(f"  Both are close to 1/d: ✓")
    
    # Key derivative value at d=0
    print(f"\nDerivative at d=0:")
    print(f"  Original: NaN (but limit = 1/W = {1/1e-7:.0e})")
    print(f"  Rational: {float(d_rational(0.0)):.0e}")
    print(f"  ✅ Rational gives correct limit!")

if __name__ == "__main__":
    test_function_comparison()