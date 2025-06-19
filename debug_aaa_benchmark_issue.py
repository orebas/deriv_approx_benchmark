#!/usr/bin/env python3
"""Debug script to trace the exact issue with AAA derivatives in the benchmark"""

import numpy as np
import pandas as pd
from comprehensive_methods_library import AAA_SmoothBarycentric_Approximator

# Simulate what happens in run_full_benchmark.py
def debug_benchmark_flow():
    # Create test data similar to benchmark
    t = np.linspace(0, 2*np.pi, 50)
    y_noisy = np.sin(t) + 0.01 * np.random.randn(len(t))
    
    print("="*60)
    print("DEBUGGING AAA METHOD IN BENCHMARK FLOW")
    print("="*60)
    
    # Create method instance (like line 97-98 in run_full_benchmark.py)
    method = AAA_SmoothBarycentric_Approximator(t, y_noisy, "AAA_SmoothBary")
    
    # Evaluate with max_derivative=4 (like line 143)
    max_deriv = 4
    print(f"\nCalling method.evaluate(t, max_derivative={max_deriv})")
    results = method.evaluate(t, max_derivative=max_deriv)
    
    print(f"\nResults dict keys: {list(results.keys())}")
    print(f"Results['success']: {results.get('success', 'NOT FOUND')}")
    
    # Check what's in the results for each derivative order (like lines 148-149)
    for deriv_order in range(max_deriv + 1):
        key = 'y' if deriv_order == 0 else f'd{deriv_order}'
        y_pred = results.get(key)
        
        print(f"\nDerivative order {deriv_order}:")
        print(f"  Key: '{key}'")
        print(f"  Type: {type(y_pred)}")
        if y_pred is not None:
            print(f"  Shape: {y_pred.shape if hasattr(y_pred, 'shape') else 'N/A'}")
            print(f"  First 5 values: {y_pred[:5] if hasattr(y_pred, '__getitem__') else y_pred}")
            print(f"  Contains NaN: {np.any(np.isnan(y_pred))}")
        else:
            print(f"  Value: None")
    
    # Check if the issue is with None values or NaN values
    print("\n" + "="*60)
    print("ANALYSIS:")
    print("="*60)
    
    for deriv_order in range(max_deriv + 1):
        key = 'y' if deriv_order == 0 else f'd{deriv_order}'
        y_pred = results.get(key)
        
        if y_pred is None:
            print(f"❌ Derivative order {deriv_order}: Key '{key}' returns None")
        elif np.all(np.isnan(y_pred)):
            print(f"❌ Derivative order {deriv_order}: All values are NaN")
        else:
            print(f"✓ Derivative order {deriv_order}: Valid data present")

if __name__ == "__main__":
    debug_benchmark_flow()