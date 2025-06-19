#!/usr/bin/env python3
"""Test if AAA methods now produce valid derivatives."""

import numpy as np
from comprehensive_methods_library import (
    AAALeastSquaresApproximator,
    AAA_FullOpt_Approximator
)

def test_aaa_derivatives():
    """Test AAA methods with the smooth evaluation fix."""
    
    # Create simple test data
    t = np.linspace(0, 2*np.pi, 50)
    y = np.sin(t) + 0.001 * np.random.RandomState(42).randn(len(t))
    
    methods = [
        (AAALeastSquaresApproximator, "AAA_LS"),
        (AAA_FullOpt_Approximator, "AAA_FullOpt")
    ]
    
    for method_class, name in methods:
        print(f"\n{'='*50}")
        print(f"Testing {name}")
        print(f"{'='*50}")
        
        # Create and fit method
        method = method_class(t, y)
        method.fit()
        
        if not method.fitted:
            print(f"❌ {name} failed to fit!")
            continue
            
        print(f"✅ Fitted successfully")
        
        # Test evaluation
        t_test = np.linspace(0.5, 5.5, 10)
        results = method.evaluate(t_test, max_derivative=7)
        
        print(f"\nDerivative results:")
        for order in range(8):
            key = 'y' if order == 0 else f'd{order}'
            if key in results:
                values = results[key]
                finite_count = np.sum(np.isfinite(values))
                all_finite = np.all(np.isfinite(values))
                
                if all_finite:
                    max_val = np.max(np.abs(values))
                    print(f"  Order {order}: ✅ All finite (max abs: {max_val:.2e})")
                else:
                    print(f"  Order {order}: ❌ {finite_count}/{len(values)} finite")
                    # Show where NaNs occur
                    nan_indices = np.where(~np.isfinite(values))[0]
                    print(f"    NaN at indices: {nan_indices}")
                    print(f"    t values at NaN: {t_test[nan_indices]}")
            else:
                print(f"  Order {order}: ❌ MISSING from results")

if __name__ == "__main__":
    test_aaa_derivatives()