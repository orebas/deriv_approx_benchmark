#!/usr/bin/env python3
"""
Comprehensive test of derivatives: higher orders, support points, very close to support points
"""

import numpy as np
import jax
import jax.numpy as jnp
from comprehensive_methods_library import smooth_barycentric_eval
from scipy.interpolate import AAA

def test_derivatives_comprehensive():
    print("COMPREHENSIVE DERIVATIVE TESTING")
    print("="*80)
    
    # Test 1: Higher derivatives up to order 8
    print("\n1. TESTING HIGHER DERIVATIVES (up to order 8)")
    print("-"*60)
    
    # Simple test case with known derivatives
    zj = jnp.array([0.0, 1.0, 2.0, 3.0])
    fj = jnp.array([0.0, 1.0, 4.0, 9.0])  # f(x) = x²
    wj = jnp.array([1.0, -1.0, 1.0, -1.0])  # Approximate barycentric weights
    
    # Create derivative functions up to 8th order
    f = lambda x: smooth_barycentric_eval(x, zj, fj, wj)
    derivatives = [f]
    for i in range(8):
        derivatives.append(jax.grad(derivatives[-1]))
    
    # Test at a point between supports
    x_test = 1.5
    print(f"\nTesting at x = {x_test} (between support points):")
    print(f"{'Order':>5} {'Value':>15} {'Expected':>15} {'Status':>10}")
    print("-"*50)
    
    expected = [x_test**2, 2*x_test, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    for order in range(9):
        try:
            val = float(derivatives[order](x_test))
            exp_val = expected[order]
            status = "✅" if not np.isnan(val) else "❌ NaN"
            print(f"{order:5d} {val:15.6f} {exp_val:15.6f} {status:>10}")
        except Exception as e:
            print(f"{order:5d} {'ERROR':>15} {expected[order]:15.6f} {'❌ Error':>10}")
    
    # Test 2: Exact support points
    print("\n\n2. TESTING AT EXACT SUPPORT POINTS")
    print("-"*60)
    
    support_points = [0.0, 1.0, 2.0, 3.0]
    
    for i, x_support in enumerate(support_points):
        print(f"\nAt support point z[{i}] = {x_support}:")
        print(f"{'Order':>5} {'Value':>15} {'Expected':>15} {'Status':>10}")
        print("-"*50)
        
        expected_at_support = [x_support**2, 2*x_support, 2.0, 0.0, 0.0, 0.0]
        
        for order in range(6):  # Test up to 5th derivative
            try:
                val = float(derivatives[order](x_support))
                exp_val = expected_at_support[order]
                status = "✅" if not np.isnan(val) else "❌ NaN"
                print(f"{order:5d} {val:15.6f} {exp_val:15.6f} {status:>10}")
            except Exception as e:
                print(f"{order:5d} {'ERROR':>15} {expected_at_support[order]:15.6f} {'❌ Error':>10}")
    
    # Test 3: Very close to support points
    print("\n\n3. TESTING VERY CLOSE TO SUPPORT POINTS")
    print("-"*60)
    
    offsets = [1e-15, 1e-12, 1e-10, 1e-8, 1e-6]
    support_point = 1.0  # Test around x = 1
    
    for offset in offsets:
        x_close = support_point + offset
        print(f"\nAt x = {support_point} + {offset:.0e} = {x_close}:")
        print(f"{'Order':>5} {'Value':>15} {'Status':>10}")
        print("-"*35)
        
        for order in range(5):  # Test up to 4th derivative
            try:
                val = float(derivatives[order](x_close))
                status = "✅" if not np.isnan(val) else "❌ NaN"
                print(f"{order:5d} {val:15.6f} {status:>10}")
            except Exception as e:
                print(f"{order:5d} {'ERROR':>15} {'❌ Error':>10}")

def test_with_real_aaa():
    print("\n\n" + "="*80)
    print("4. TESTING WITH REAL AAA WEIGHTS AND DATA")
    print("="*80)
    
    # Generate realistic test data
    t = np.linspace(0, 2*np.pi, 30)
    y = np.sin(t) + 0.005 * np.random.randn(len(t))  # Low noise
    
    # Get AAA approximation
    aaa = AAA(t, y, max_terms=10)
    zj = jnp.array(aaa.support_points)
    fj = jnp.array(aaa.support_values)
    wj = jnp.array(aaa.weights)
    
    print(f"AAA approximation: {len(zj)} support points")
    print(f"Support points: {zj[:5]}... (showing first 5)")
    print(f"Weights: {wj[:5]}... (showing first 5)")
    
    # Create derivative functions
    f = lambda x: smooth_barycentric_eval(x, zj, fj, wj)
    f1 = jax.grad(f)
    f2 = jax.grad(f1)
    f3 = jax.grad(f2)
    
    # Test at support points
    print(f"\nTesting derivatives at AAA support points:")
    print(f"{'Point':>6} {'f(x)':>10} {'f\'(x)':>10} {'f\'\'(x)':>10} {'f\'\'\'(x)':>10} {'Status':>10}")
    print("-"*70)
    
    for i, z in enumerate(zj[:5]):  # Test first 5 support points
        try:
            v0 = float(f(z))
            v1 = float(f1(z))
            v2 = float(f2(z))
            v3 = float(f3(z))
            
            status = "✅" if not any(np.isnan([v1, v2, v3])) else "❌ NaN"
            print(f"{i:6d} {v0:10.6f} {v1:10.6f} {v2:10.6f} {v3:10.6f} {status:>10}")
            
        except Exception as e:
            print(f"{i:6d} {'ERROR':>40} {'❌ Error':>10}")
    
    # Test very close to support points
    print(f"\nTesting very close to AAA support points:")
    print(f"{'Offset':>8} {'f\'(x)':>10} {'f\'\'(x)':>10} {'Status':>10}")
    print("-"*40)
    
    # Pick one support point and test nearby
    test_support = float(zj[0])
    for offset in [1e-14, 1e-12, 1e-10, 1e-8]:
        x_test = test_support + offset
        try:
            v1 = float(f1(x_test))
            v2 = float(f2(x_test))
            
            status = "✅" if not any(np.isnan([v1, v2])) else "❌ NaN"
            print(f"{offset:8.0e} {v1:10.6f} {v2:10.6f} {status:>10}")
            
        except Exception as e:
            print(f"{offset:8.0e} {'ERROR':>20} {'❌ Error':>10}")

if __name__ == "__main__":
    test_derivatives_comprehensive()
    test_with_real_aaa()