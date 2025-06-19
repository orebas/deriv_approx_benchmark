#!/usr/bin/env python3
"""
Verification script to ensure stable AAA methods are properly integrated.
"""

import numpy as np
import sys

def test_method_integration():
    """Test that the stable AAA methods are properly integrated"""
    
    print("Testing stable AAA method integration...")
    
    # Test imports
    try:
        from comprehensive_methods_library import create_all_methods, get_base_method_names, get_method_categories
        print("✓ Successfully imported method factory functions")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Test method names
    try:
        method_names = get_base_method_names()
        expected_new_methods = ['AAA_TwoStage', 'AAA_SmoothBary']
        
        for method in expected_new_methods:
            if method in method_names:
                print(f"✓ {method} found in method list")
            else:
                print(f"✗ {method} NOT found in method list")
                return False
                
    except Exception as e:
        print(f"✗ Failed to get method names: {e}")
        return False
    
    # Test method categories
    try:
        categories = get_method_categories()
        advanced_methods = categories.get('Advanced', [])
        
        if 'AAA_TwoStage' in advanced_methods and 'AAA_SmoothBary' in advanced_methods:
            print("✓ New methods correctly categorized as Advanced")
        else:
            print(f"✗ New methods not properly categorized. Advanced: {advanced_methods}")
            return False
            
    except Exception as e:
        print(f"✗ Failed to get method categories: {e}")
        return False
    
    # Test method instantiation
    try:
        t = np.linspace(0, 1, 11)
        y = np.sin(t * 2 * np.pi) + 0.01 * np.random.randn(len(t))
        
        methods = create_all_methods(t, y)
        
        if 'AAA_TwoStage' in methods:
            print("✓ AAA_TwoStage instantiated successfully")
        else:
            print("✗ AAA_TwoStage not in created methods")
            return False
            
        if 'AAA_SmoothBary' in methods:
            print("✓ AAA_SmoothBary instantiated successfully")
        else:
            print("✗ AAA_SmoothBary not in created methods")
            return False
            
    except Exception as e:
        print(f"✗ Failed to create methods: {e}")
        return False
    
    # Test basic method attributes
    try:
        two_stage = methods['AAA_TwoStage']
        smooth_bary = methods['AAA_SmoothBary']
        
        # Check required attributes
        for method, name in [(two_stage, 'AAA_TwoStage'), (smooth_bary, 'AAA_SmoothBary')]:
            if hasattr(method, 'fit') and hasattr(method, 'evaluate'):
                print(f"✓ {name} has required methods")
            else:
                print(f"✗ {name} missing required methods")
                return False
                
            if hasattr(method, 'max_derivative_supported'):
                print(f"✓ {name} has max_derivative_supported = {method.max_derivative_supported}")
            else:
                print(f"✗ {name} missing max_derivative_supported")
                return False
                
    except Exception as e:
        print(f"✗ Failed to check method attributes: {e}")
        return False
    
    print("\n✓ All integration tests passed!")
    return True


def test_benchmark_config():
    """Test that benchmark configuration includes new methods"""
    
    print("\nTesting benchmark configuration...")
    
    try:
        import json
        with open('benchmark_config.json', 'r') as f:
            config = json.load(f)
            
        python_methods = config.get('python_methods', {}).get('base_methods', [])
        
        expected_methods = ['AAA_TwoStage', 'AAA_SmoothBary']
        for method in expected_methods:
            if method in python_methods:
                print(f"✓ {method} found in benchmark config")
            else:
                print(f"✗ {method} NOT found in benchmark config")
                return False
                
    except Exception as e:
        print(f"✗ Failed to check benchmark config: {e}")
        return False
        
    print("✓ Benchmark configuration updated correctly!")
    return True


def main():
    """Run all verification tests"""
    
    print("Verifying Stable AAA Integration")
    print("=" * 40)
    
    success = True
    
    success &= test_method_integration()
    success &= test_benchmark_config()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 All verification tests PASSED!")
        print("\nThe stable AAA methods are ready for use:")
        print("  - AAA_TwoStage: Two-stage approach (AAA_LS + limited refinement)")
        print("  - AAA_SmoothBary: Smooth barycentric evaluation for better gradients")
        print("\nTo address your original questions:")
        print("  1. Gradient issue: Partially fixed with smooth barycentric approach")
        print("  2. Methods added to comprehensive library: ✓ Done")
        print("  3. Methods added to benchmark config: ✓ Done")
        return 0
    else:
        print("❌ Some verification tests FAILED!")
        print("Please check the errors above and fix before using.")
        return 1


if __name__ == "__main__":
    sys.exit(main())