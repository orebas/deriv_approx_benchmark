#!/usr/bin/env python3
"""
Integration test to verify that the safe barycentric evaluation fix
resolves RMSE blowups in AAA methods.
"""

import numpy as np
import sys
from comprehensive_methods_library import (
    AAALeastSquaresApproximator,
    AAA_FullOpt_Approximator, 
    AAA_TwoStage_Approximator,
    AAA_SmoothBarycentric_Approximator
)

def create_test_data():
    """Create test data that previously caused issues."""
    # Oscillatory function with noise - challenging for AAA methods
    t = np.linspace(0, 2*np.pi, 25)
    y_clean = np.sin(2*t) * np.exp(-0.1*t)
    noise = 0.02 * np.random.RandomState(42).randn(len(t))
    y_noisy = y_clean + noise
    return t, y_noisy, y_clean

def test_aaa_method(method_class, method_name, t, y):
    """Test an individual AAA method."""
    print(f"  Testing {method_name}...", end=" ")
    
    try:
        # Create and fit method
        method = method_class(t, y)
        method.fit()
        
        if not method.fitted:
            print("❌ FAILED - Not fitted")
            return False
        
        # Test function evaluation
        t_eval = np.linspace(t.min(), t.max(), 50)
        result = method.evaluate(t_eval)
        
        if not result['success']:
            print("❌ FAILED - Evaluation failed")
            return False
        
        # Check for NaN/Inf in function values
        if not np.all(np.isfinite(result['y'])):
            print("❌ FAILED - Non-finite function values")
            return False
        
        # Test derivative evaluation
        deriv_result = method.evaluate(t_eval, order=1)
        if not deriv_result['success']:
            print("❌ FAILED - Derivative evaluation failed")
            return False
        
        # Check for NaN/Inf in derivatives
        if not np.all(np.isfinite(deriv_result['y'])):
            print("❌ FAILED - Non-finite derivative values")
            return False
        
        # Check RMSE is reasonable (not blown up)
        rmse = np.sqrt(np.mean((result['y'] - np.sin(2*t_eval) * np.exp(-0.1*t_eval))**2))
        if rmse > 10.0:  # Reasonable threshold
            print(f"❌ FAILED - RMSE too high: {rmse:.3f}")
            return False
        
        # Get support point info if available
        n_support = len(getattr(method, 'zj', [])) if hasattr(method, 'zj') and method.zj is not None else 'N/A'
        
        print(f"✅ PASSED (RMSE: {rmse:.4f}, Support points: {n_support})")
        return True
        
    except Exception as e:
        print(f"❌ FAILED - Exception: {str(e)[:50]}")
        return False

def main():
    """Run integration tests for all AAA methods."""
    print("🔧 Testing AAA Methods with Safe Barycentric Evaluation Fix")
    print("=" * 70)
    
    # Create test data
    t, y_noisy, y_clean = create_test_data()
    print(f"Test data: {len(t)} points, noise level ~0.02")
    
    # Test all AAA methods
    aaa_methods = [
        (AAALeastSquaresApproximator, "AAA_LS"),
        (AAA_FullOpt_Approximator, "AAA_FullOpt"),
        (AAA_TwoStage_Approximator, "AAA_TwoStage"), 
        (AAA_SmoothBarycentric_Approximator, "AAA_SmoothBary")
    ]
    
    results = {}
    print("\nTesting AAA methods:")
    
    for method_class, method_name in aaa_methods:
        success = test_aaa_method(method_class, method_name, t, y_noisy)
        results[method_name] = success
    
    # Summary
    print("\n" + "=" * 70)
    passed = sum(results.values())
    total = len(results)
    
    print(f"Results: {passed}/{total} AAA methods passed")
    
    if passed == total:
        print("🎉 All AAA methods working correctly with the fix!")
        print("✅ Safe barycentric evaluation has resolved RMSE blowup issues")
    else:
        print("⚠️  Some AAA methods still have issues:")
        for method_name, success in results.items():
            if not success:
                print(f"   - {method_name}: Still failing")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)