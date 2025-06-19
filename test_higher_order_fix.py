#!/usr/bin/env python3
"""
Quick test to see if the higher-order finite differences help with derivative explosion
"""

import numpy as np
import jax
import jax.numpy as jnp
from comprehensive_methods_library import AAALeastSquaresApproximator

def test_higher_order_derivatives():
    print("Testing higher-order derivatives with improved finite differences:")
    print("="*70)
    
    # Simple test data - clean sine wave
    t = np.linspace(0, 2*np.pi, 30)
    y = np.sin(t)
    
    print(f"Test data: {len(t)} points, clean sin(t)")
    
    # Test just one AAA method
    try:
        method = AAALeastSquaresApproximator(t, y, "AAA_LS_Test")
        method.fit()
        
        if not method.success:
            print("❌ Method failed to fit")
            return False
            
        print("✅ Method fitted successfully")
        
        # Test evaluation at a few points
        t_test = np.array([0.5, 1.0, 1.5])
        results = method.evaluate(t_test, max_derivative=5)
        
        print("\nResults at test points:")
        print(f"{'Point':>8} {'Function':>12} {'d1':>12} {'d2':>12} {'d3':>12} {'d4':>12} {'d5':>12}")
        print("-" * 80)
        
        max_higher_order = 0
        all_finite = True
        
        for i, t_val in enumerate(t_test):
            f_val = results['y'][i]
            d1_val = results['d1'][i]
            d2_val = results['d2'][i] 
            d3_val = results['d3'][i]
            d4_val = results['d4'][i]
            d5_val = results['d5'][i]
            
            # Check for NaN/Inf
            values = [f_val, d1_val, d2_val, d3_val, d4_val, d5_val]
            finite_check = [np.isfinite(v) for v in values]
            
            if not all(finite_check):
                all_finite = False
                status = "❌"
            else:
                status = "✅"
                
            # Track maximum higher-order derivative magnitude
            for val in [d3_val, d4_val, d5_val]:
                if np.isfinite(val):
                    max_higher_order = max(max_higher_order, abs(val))
            
            print(f"{status} {t_val:8.3f} {f_val:12.6f} {d1_val:12.6f} {d2_val:12.6f} {d3_val:12.2e} {d4_val:12.2e} {d5_val:12.2e}")
        
        print(f"\n📊 Summary:")
        print(f"   All derivatives finite: {'✅ Yes' if all_finite else '❌ No'}")
        print(f"   Max higher-order magnitude: {max_higher_order:.2e}")
        
        # Comparison with previous issue
        if all_finite and max_higher_order < 1e6:
            print("\n🎉 MAJOR IMPROVEMENT!")
            print("   ✅ No more NaN values")
            print("   ✅ Higher-order derivatives under control") 
            if max_higher_order < 1e3:
                print("   🌟 EXCELLENT: Very stable higher-order derivatives")
            return True
        elif all_finite:
            print("\n🔶 PARTIAL IMPROVEMENT:")
            print("   ✅ No more NaN values")
            print("   ⚠️  Higher-order derivatives still large but finite")
            return True
        else:
            print("\n❌ Still has issues with NaN/Inf values")
            return False
            
    except Exception as e:
        print(f"❌ Exception during test: {e}")
        return False

if __name__ == "__main__":
    success = test_higher_order_derivatives()
    if success:
        print("\n✅ Test completed successfully")
    else:
        print("\n❌ Test failed")