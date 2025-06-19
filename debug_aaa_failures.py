#!/usr/bin/env python3
"""
Debug script to investigate AAA algorithm failures.
Tests each AAA method step by step to identify where things are breaking.
"""

import numpy as np
import sys
import traceback
from comprehensive_methods_library import (
    AAALeastSquaresApproximator,
    AAA_FullOpt_Approximator, 
    AAA_TwoStage_Approximator,
    AAA_SmoothBarycentric_Approximator
)

def create_simple_test_data():
    """Create simple, well-conditioned test data."""
    t = np.linspace(0, 2*np.pi, 20)
    y = np.sin(t) + 0.01 * np.random.RandomState(42).randn(len(t))
    return t, y

def debug_aaa_method(method_class, method_name, t, y):
    """Debug a single AAA method step by step."""
    print(f"\n{'='*60}")
    print(f"🔍 DEBUGGING {method_name}")
    print(f"{'='*60}")
    
    try:
        # Step 1: Create method
        print("Step 1: Creating method instance...", end=" ")
        method = method_class(t, y)
        print("✅ SUCCESS")
        
        # Step 2: Fit method
        print("Step 2: Fitting method...", end=" ")
        method.fit()
        print(f"✅ SUCCESS (fitted: {method.fitted})")
        
        if not method.fitted:
            print("❌ Method didn't fit successfully")
            return False
        
        # Step 3: Check fitted parameters
        print("Step 3: Checking fitted parameters...")
        if hasattr(method, 'zj') and method.zj is not None:
            print(f"  Support points (zj): {len(method.zj)} points")
            print(f"  Support values (fj): shape {method.fj.shape if hasattr(method, 'fj') and method.fj is not None else 'None'}")
            print(f"  Weights (wj): shape {method.wj.shape if hasattr(method, 'wj') and method.wj is not None else 'None'}")
            
            # Check for NaN/Inf in parameters
            if hasattr(method, 'fj') and method.fj is not None:
                fj_finite = np.all(np.isfinite(method.fj))
                print(f"  fj finite: {fj_finite}")
                if not fj_finite:
                    print(f"    fj values: {method.fj}")
            
            if hasattr(method, 'wj') and method.wj is not None:
                wj_finite = np.all(np.isfinite(method.wj))
                print(f"  wj finite: {wj_finite}")
                if not wj_finite:
                    print(f"    wj values: {method.wj}")
        else:
            print("  ❌ No support points found")
            return False
        
        # Step 4: Test function evaluation
        print("Step 4: Testing function evaluation...")
        t_eval = np.linspace(t.min(), t.max(), 10)
        
        try:
            result_func = method.evaluate(t_eval, max_derivative=0)
            if result_func['success']:
                y_pred = result_func['y']
                finite_count = np.sum(np.isfinite(y_pred))
                print(f"  Function evaluation: ✅ SUCCESS ({finite_count}/{len(y_pred)} finite values)")
                if finite_count < len(y_pred):
                    print(f"    Non-finite values at indices: {np.where(~np.isfinite(y_pred))[0]}")
            else:
                print(f"  Function evaluation: ❌ FAILED")
                return False
        except Exception as e:
            print(f"  Function evaluation: ❌ EXCEPTION: {e}")
            return False
        
        # Step 5: Test derivative evaluation
        print("Step 5: Testing derivative evaluation...")
        max_deriv_test = min(3, getattr(method, 'max_derivative_supported', 3))
        
        for deriv_order in range(1, max_deriv_test + 1):
            try:
                result_deriv = method.evaluate(t_eval, max_derivative=deriv_order)
                if result_deriv['success']:
                    y_deriv = result_deriv[f'd{deriv_order}']
                    finite_count = np.sum(np.isfinite(y_deriv))
                    print(f"  Derivative order {deriv_order}: ✅ SUCCESS ({finite_count}/{len(y_deriv)} finite)")
                    if finite_count < len(y_deriv):
                        print(f"    Non-finite values at indices: {np.where(~np.isfinite(y_deriv))[0]}")
                        print(f"    Sample values: {y_deriv[:5]}")
                else:
                    print(f"  Derivative order {deriv_order}: ❌ FAILED (success=False)")
                    break
            except Exception as e:
                print(f"  Derivative order {deriv_order}: ❌ EXCEPTION: {str(e)[:100]}")
                traceback.print_exc()
                break
        
        # Step 6: Test with benchmark-style evaluation
        print("Step 6: Testing benchmark-style evaluation...")
        try:
            results = method.evaluate(t_eval, max_derivative=max_deriv_test)
            if results.get('success', True):
                print(f"  Benchmark evaluation: ✅ SUCCESS")
                for order in range(max_deriv_test + 1):
                    key = 'y' if order == 0 else f'd{order}'
                    if key in results:
                        values = results[key]
                        finite_count = np.sum(np.isfinite(values))
                        print(f"    Order {order} ({key}): {finite_count}/{len(values)} finite")
                    else:
                        print(f"    Order {order} ({key}): ❌ MISSING")
            else:
                print(f"  Benchmark evaluation: ❌ FAILED")
        except Exception as e:
            print(f"  Benchmark evaluation: ❌ EXCEPTION: {str(e)[:100]}")
            traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        traceback.print_exc()
        return False

def main():
    """Run AAA debugging for all methods."""
    print("🐛 AAA ALGORITHM FAILURE DIAGNOSIS")
    print("=" * 70)
    
    # Create test data
    t, y = create_simple_test_data()
    print(f"Test data: {len(t)} points, y range: [{y.min():.3f}, {y.max():.3f}]")
    
    # Test all AAA methods
    aaa_methods = [
        (AAALeastSquaresApproximator, "AAA_LS"),
        (AAA_FullOpt_Approximator, "AAA_FullOpt"),
        (AAA_TwoStage_Approximator, "AAA_TwoStage"), 
        (AAA_SmoothBarycentric_Approximator, "AAA_SmoothBary")
    ]
    
    results = {}
    
    for method_class, method_name in aaa_methods:
        success = debug_aaa_method(method_class, method_name, t, y)
        results[method_name] = success
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"Methods working: {passed}/{total}")
    
    for method_name, success in results.items():
        status = "✅ WORKING" if success else "❌ BROKEN"
        print(f"  {method_name:20s}: {status}")
    
    if passed < total:
        print(f"\n⚠️  {total - passed} methods are broken!")
        print("This explains why the benchmark results are empty.")
    else:
        print(f"\n🎉 All methods appear to be working in isolation.")
        print("The issue might be with the benchmark integration or data-specific.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)