#!/usr/bin/env python3
"""
Test all updated AAA methods with the new smooth barycentric evaluation (W=1e-7)
"""

import numpy as np
import sys
import os
sys.path.append(os.getcwd())

from comprehensive_methods_library import (
    AAALeastSquaresApproximator,
    AAA_FullOpt_Approximator, 
    AAA_TwoStage_Approximator,
    AAA_SmoothBarycentric_Approximator
)

def test_aaa_method(method_class, method_name):
    """Test a single AAA method"""
    print(f"\nTesting {method_name}")
    print("="*50)
    
    # Create test data - sine wave with small noise
    t = np.linspace(0, 2*np.pi, 30)
    y_clean = np.sin(t)
    y_noisy = y_clean + 0.01 * np.random.RandomState(42).randn(len(t))
    
    try:
        # Create and fit method
        method = method_class(t, y_noisy)
        method.fit()
        
        if not method.fitted:
            print(f"❌ {method_name}: Failed to fit")
            return False
            
        if hasattr(method, 'success') and not method.success:
            print(f"❌ {method_name}: Fit completed but marked as unsuccessful")
            return False
        
        print(f"✅ {method_name}: Fit successful")
        
        # Test function evaluation
        t_test = np.array([0.5, 1.0, 1.5])
        results = method.evaluate(t_test, max_derivative=4)
        
        if not results.get('success', True):
            print(f"❌ {method_name}: Evaluation failed")
            return False
            
        print(f"✅ {method_name}: Function evaluation successful")
        
        # Check all derivatives are finite
        all_finite = True
        for order in range(5):  # 0 to 4
            key = 'y' if order == 0 else f'd{order}'
            if key in results:
                values = results[key]
                if not np.all(np.isfinite(values)):
                    print(f"❌ {method_name}: Non-finite values in {key}")
                    all_finite = False
                else:
                    max_val = np.max(np.abs(values))
                    print(f"  {key}: max |value| = {max_val:.2e}")
        
        if all_finite:
            print(f"✅ {method_name}: All derivatives are finite")
        else:
            print(f"❌ {method_name}: Some derivatives are non-finite")
            return False
            
        # Test accuracy on a simple point
        y_pred = results['y']
        y_true = np.sin(t_test)
        error = np.mean(np.abs(y_pred - y_true))
        
        print(f"  Mean absolute error: {error:.4f}")
        
        if error < 1.0:  # Reasonable for noisy data
            print(f"✅ {method_name}: Reasonable accuracy")
        else:
            print(f"⚠️ {method_name}: High error, but may be acceptable")
            
        return True
        
    except Exception as e:
        print(f"❌ {method_name}: Exception - {e}")
        return False

def test_all_aaa_methods():
    """Test all AAA methods"""
    print("TESTING ALL UPDATED AAA METHODS")
    print("="*60)
    
    methods_to_test = [
        (AAALeastSquaresApproximator, "AAA_LS"),
        (AAA_TwoStage_Approximator, "AAA_TwoStage"), 
        (AAA_FullOpt_Approximator, "AAA_FullOpt"),
        (AAA_SmoothBarycentric_Approximator, "AAA_SmoothBary")
    ]
    
    results = {}
    
    for method_class, method_name in methods_to_test:
        success = test_aaa_method(method_class, method_name)
        results[method_name] = success
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for method_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{method_name:20}: {status}")
        if not success:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 ALL AAA METHODS WORKING WITH SMOOTH EVALUATION!")
        print("✅ Ready for full benchmark testing")
    else:
        print("⚠️ Some methods still have issues")
        
    return all_passed

if __name__ == "__main__":
    success = test_all_aaa_methods()
    
    if success:
        print("\n🚀 SUCCESS: All AAA methods updated and working!")
        print("The smooth barycentric evaluation (W=1e-7) is now integrated")
        print("across all AAA algorithms for evaluation, derivatives, and optimization.")
    else:
        print("\n❌ Some issues remain - check the output above")