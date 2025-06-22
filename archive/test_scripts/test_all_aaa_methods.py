#!/usr/bin/env python3
"""Test all AAA methods to ensure they work with the new stable implementation"""

import numpy as np
from comprehensive_methods_library import (
    AAALeastSquaresApproximator,
    AAA_FullOpt_Approximator, 
    AAA_TwoStage_Approximator,
    AAA_SmoothBarycentric_Approximator
)

def test_all_aaa_methods():
    print("Testing all AAA methods with new stable implementation:")
    print("="*70)
    
    # Generate test data
    t = np.linspace(0, 2*np.pi, 40)
    y = np.sin(t) + 0.005 * np.random.randn(len(t))  # Light noise
    
    # All AAA methods
    aaa_methods = [
        (AAALeastSquaresApproximator, "AAA_LS"),
        (AAA_FullOpt_Approximator, "AAA_FullOpt"),
        (AAA_TwoStage_Approximator, "AAA_TwoStage"),  
        (AAA_SmoothBarycentric_Approximator, "AAA_SmoothBary")
    ]
    
    print(f"Test data: {len(t)} points, sin(t) + small noise")
    print(f"Testing derivatives up to order 4")
    
    results_summary = []
    
    for method_class, method_name in aaa_methods:
        print(f"\n{'-'*50}")
        print(f"Testing {method_name}")
        print(f"{'-'*50}")
        
        try:
            # Create and fit method
            method = method_class(t, y, method_name)
            method.fit()
            
            print(f"✅ Fit successful: {method.fitted}")
            print(f"✅ Success flag: {method.success}")
            
            if method.success:
                # Test derivatives
                results = method.evaluate(t, max_derivative=4)
                
                print(f"✅ Evaluation completed")
                
                # Check for NaN in each derivative order
                all_clean = True
                for order in range(5):
                    key = 'y' if order == 0 else f'd{order}'
                    values = results.get(key)
                    
                    if values is not None:
                        nan_count = np.sum(np.isnan(values))
                        finite_count = np.sum(np.isfinite(values))
                        
                        print(f"  Order {order}: {finite_count}/{len(values)} finite, {nan_count} NaN")
                        
                        if nan_count > 0:
                            all_clean = False
                
                if all_clean:
                    print("✅ All derivatives are finite - SUCCESS!")
                    
                    # Compute accuracy
                    y_true = np.sin(t)
                    y_pred = results['y']
                    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
                    print(f"✅ Function RMSE: {rmse:.6f}")
                    
                    results_summary.append((method_name, "SUCCESS", rmse, "All finite"))
                else:
                    print("❌ Some derivatives contain NaN")
                    results_summary.append((method_name, "PARTIAL", np.nan, "Some NaN"))
            else:
                print("❌ Method failed to fit")
                results_summary.append((method_name, "FAILED", np.nan, "Fit failed"))
                
        except Exception as e:
            print(f"❌ Exception: {str(e)[:50]}...")
            results_summary.append((method_name, "ERROR", np.nan, str(e)[:30]))
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY OF ALL AAA METHODS")
    print(f"{'='*70}")
    
    print(f"{'Method':>20} {'Status':>10} {'RMSE':>12} {'Notes':>25}")
    print("-" * 70)
    
    for method_name, status, rmse, notes in results_summary:
        rmse_str = f"{rmse:.6f}" if not np.isnan(rmse) else "N/A"
        print(f"{method_name:>20} {status:>10} {rmse_str:>12} {notes:>25}")
    
    # Overall verdict
    success_count = sum(1 for _, status, _, _ in results_summary if status == "SUCCESS")
    total_count = len(results_summary)
    
    print(f"\n🎯 OVERALL RESULT: {success_count}/{total_count} AAA methods working perfectly!")
    
    if success_count == total_count:
        print("🎉 ALL AAA METHODS ARE NOW WORKING WITH STABLE DERIVATIVES!")
    else:
        print("⚠️  Some methods still have issues - need further investigation")

if __name__ == "__main__":
    test_all_aaa_methods()