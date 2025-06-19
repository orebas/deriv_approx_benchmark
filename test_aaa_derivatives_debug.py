#!/usr/bin/env python3
"""Test script to debug AAA derivative computation issues"""

import numpy as np
import sys
import traceback
from comprehensive_methods_library import (
    AAA_SmoothBarycentric_Approximator,
    AAA_FullOpt_Approximator,
    AAA_TwoStage_Approximator,
    AAALeastSquaresApproximator
)

def test_aaa_method(method_class, t, y, method_name):
    """Test a single AAA method"""
    print(f"\n{'='*60}")
    print(f"Testing {method_name}")
    print(f"{'='*60}")
    
    try:
        # Create and fit the method
        method = method_class(t, y, method_name)
        method.fit()
        
        print(f"Fit successful: {method.fitted}")
        print(f"Fit time: {method.fit_time:.3f} seconds")
        
        # Test evaluation
        t_test = np.linspace(t[0], t[-1], 10)
        results = method.evaluate(t_test, max_derivative=4)
        
        print(f"\nEvaluation results:")
        print(f"Success: {results.get('success', False)}")
        
        for key in ['y', 'd1', 'd2', 'd3', 'd4']:
            if key in results:
                values = results[key]
                if values is not None and not np.all(np.isnan(values)):
                    print(f"{key}: min={np.nanmin(values):.6f}, max={np.nanmax(values):.6f}, mean={np.nanmean(values):.6f}")
                else:
                    print(f"{key}: All NaN or None")
        
        # Check internal state for AAA methods
        if hasattr(method, 'zj') and method.zj is not None:
            print(f"\nAAA parameters:")
            print(f"Support points (zj): {len(method.zj) if method.zj is not None else 'None'}")
            print(f"Success flag: {getattr(method, 'success', 'Not set')}")
            print(f"AD derivatives built: {len(getattr(method, 'ad_derivatives', []))}")
            
            # Try to evaluate derivatives directly
            if hasattr(method, 'ad_derivatives') and method.ad_derivatives:
                print(f"\nDirect derivative evaluation test:")
                test_point = t[len(t)//2]
                for i, deriv_func in enumerate(method.ad_derivatives[:3]):
                    try:
                        val = float(deriv_func(test_point))
                        print(f"  Derivative order {i}: {val:.6f}")
                    except Exception as e:
                        print(f"  Derivative order {i}: Error - {str(e)}")
        
    except Exception as e:
        print(f"\nERROR in {method_name}:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        traceback.print_exc()

def main():
    # Generate test data
    t = np.linspace(0, 2*np.pi, 50)
    y_clean = np.sin(t)
    noise = 0.01 * np.random.randn(len(t))
    y = y_clean + noise
    
    print(f"Test data: {len(t)} points from {t[0]:.2f} to {t[-1]:.2f}")
    print(f"Signal: sin(t) with noise level 0.01")
    
    # Test each AAA method
    methods_to_test = [
        (AAALeastSquaresApproximator, "AAA_LS"),
        (AAA_SmoothBarycentric_Approximator, "AAA_SmoothBary"),
        (AAA_FullOpt_Approximator, "AAA_FullOpt"),
        (AAA_TwoStage_Approximator, "AAA_TwoStage")
    ]
    
    for method_class, name in methods_to_test:
        test_aaa_method(method_class, t, y, name)
    
    print("\n" + "="*60)
    print("Testing complete")

if __name__ == "__main__":
    main()