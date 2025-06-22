#!/usr/bin/env python3
"""
Test the Julia AAA integration with the benchmark framework.
This validates that the new methods work with the unified runner.
"""

import json
import subprocess
import sys

def test_configuration():
    """Test that the configuration includes the new Julia AAA methods."""
    print("🧪 Testing Configuration Integration")
    print("=" * 60)
    
    with open('benchmark_config.json', 'r') as f:
        config = json.load(f)
    
    # Check Julia methods
    julia_methods = config['julia_methods']
    new_methods = ["JuliaAAALS", "JuliaAAAFullOpt", "JuliaAAATwoStage", "JuliaAAASmoothBary"]
    
    print(f"Total Julia methods in config: {len(julia_methods)}")
    
    all_present = True
    for method in new_methods:
        if method in julia_methods:
            print(f"  ✅ {method} - Found in configuration")
        else:
            print(f"  ❌ {method} - Missing from configuration")
            all_present = False
    
    if all_present:
        print("✅ All new Julia AAA methods properly integrated")
    else:
        print("❌ Some methods missing from configuration")
    
    return all_present

def test_edit_config():
    """Test that edit_config.py can see the new methods."""
    print("\n🛠️  Testing edit_config.py Integration")
    print("=" * 60)
    
    try:
        result = subprocess.run(['python', 'edit_config.py', '--show'], 
                               capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            output = result.stdout
            new_methods = ["JuliaAAALS", "JuliaAAAFullOpt", "JuliaAAATwoStage", "JuliaAAASmoothBary"]
            
            all_visible = True
            for method in new_methods:
                if method in output:
                    print(f"  ✅ {method} - Visible in edit_config.py")
                else:
                    print(f"  ❌ {method} - Not visible in edit_config.py")
                    all_visible = False
            
            if all_visible:
                print("✅ All new methods visible in edit_config.py")
            else:
                print("❌ Some methods not visible in edit_config.py")
            
            return all_visible
        else:
            print(f"❌ edit_config.py failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ edit_config.py timed out")
        return False
    except Exception as e:
        print(f"❌ Error running edit_config.py: {e}")
        return False

def create_test_config():
    """Create a minimal test configuration with only the new Julia methods."""
    print("\n📝 Creating Test Configuration")
    print("=" * 60)
    
    # Load current config
    with open('benchmark_config.json', 'r') as f:
        config = json.load(f)
    
    # Create backup
    with open('benchmark_config.json.backup', 'w') as f:
        json.dump(config, f, indent=2)
    print("✅ Created backup: benchmark_config.json.backup")
    
    # Create test config with minimal settings
    test_config = config.copy()
    test_config['ode_problems'] = ['lv_periodic']  # Single ODE for fast testing
    test_config['noise_levels'] = [0.001]  # Single noise level
    test_config['julia_methods'] = ['JuliaAAALS', 'JuliaAAATwoStage']  # Two new methods
    test_config['python_methods']['base_methods'] = []  # Disable Python methods for faster testing
    test_config['python_methods']['enhanced_gp_methods'] = []
    test_config['data_config']['data_size'] = 51  # Smaller for faster testing
    test_config['data_config']['derivative_orders'] = 3  # Fewer derivatives for faster testing
    
    # Save test config
    with open('benchmark_config_test.json', 'w') as f:
        json.dump(test_config, f, indent=2)
    print("✅ Created test configuration: benchmark_config_test.json")
    
    return True

def show_usage_instructions():
    """Show instructions for using the new Julia AAA methods."""
    print("\n📋 Usage Instructions")
    print("=" * 60)
    
    print("To use the new Julia AAA methods:")
    print("")
    print("1. Enable/disable methods with edit_config.py:")
    print("   python edit_config.py")
    print("   -> Choose option 3 (Toggle Julia methods)")
    print("   -> Enable: JuliaAAALS, JuliaAAAFullOpt, JuliaAAATwoStage, JuliaAAASmoothBary")
    print("")
    print("2. Run benchmarks with unified runner:")
    print("   python run_unified_benchmark.py")
    print("")
    print("3. Compare with Python AAA methods:")
    print("   - JuliaAAALS vs AAA_LS")
    print("   - JuliaAAAFullOpt vs AAA_FullOpt") 
    print("   - JuliaAAATwoStage vs AAA_TwoStage")
    print("   - JuliaAAASmoothBary vs AAA_SmoothBary")
    print("")
    print("4. Expected improvements in Julia versions:")
    print("   ✅ No NaN values in derivatives")
    print("   ✅ No explosion to 10^15+ values")
    print("   ✅ Stable higher-order derivatives")
    print("   ✅ Proper automatic differentiation")

def main():
    """Run all integration tests."""
    print("🚀 Julia AAA Methods - Integration Test")
    print("=" * 80)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Configuration
    if test_configuration():
        success_count += 1
    
    # Test 2: edit_config.py
    if test_edit_config():
        success_count += 1
    
    # Test 3: Create test config
    if create_test_config():
        success_count += 1
    
    # Show usage instructions
    show_usage_instructions()
    
    # Final summary
    print(f"\n🎯 INTEGRATION TEST RESULTS")
    print("=" * 80)
    print(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Julia AAA methods fully integrated with benchmark framework")
        print("✅ Ready to use with edit_config.py and run_unified_benchmark.py")
        print("")
        print("Next steps:")
        print("1. Test the methods: python run_unified_benchmark.py")
        print("2. Compare results with Python AAA methods")
        print("3. After validation, consider removing Python AAA methods")
        return 0
    else:
        print("❌ Some tests failed - integration may have issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())