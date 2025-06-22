#!/usr/bin/env python3
"""
Quick script to enable only the new Julia AAA methods for testing.
This creates a minimal configuration to test the new methods against the Python ones.
"""

import json

def enable_julia_aaa_only():
    """Create a configuration with only the new Julia AAA methods enabled."""
    
    # Load current config
    with open('benchmark_config.json', 'r') as f:
        config = json.load(f)
    
    print("🔧 Configuring benchmark for Julia AAA comparison")
    print("=" * 60)
    
    # Backup original
    with open('benchmark_config.json.original', 'w') as f:
        json.dump(config, f, indent=2)
    print("✅ Backed up original config to benchmark_config.json.original")
    
    # Modify config for Julia AAA testing
    config['julia_methods'] = [
        'JuliaAAALS',
        'JuliaAAAFullOpt', 
        'JuliaAAATwoStage',
        'JuliaAAASmoothBary'
    ]
    
    # Keep corresponding Python methods for comparison
    config['python_methods']['base_methods'] = [
        'AAA_LS',
        'AAA_FullOpt',
        'AAA_TwoStage', 
        'AAA_SmoothBary'
    ]
    
    # Disable other methods to focus on AAA comparison
    config['python_methods']['enhanced_gp_methods'] = []
    
    # Use smaller dataset for faster testing
    config['ode_problems'] = ['lv_periodic']
    config['noise_levels'] = [0.0, 0.001, 0.01]  # Keep all noise levels
    config['data_config']['data_size'] = 101  # Smaller for faster testing
    
    # Save modified config
    with open('benchmark_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Updated configuration for Julia AAA testing")
    print("\nEnabled methods:")
    print("📐 Julia methods:")
    for method in config['julia_methods']:
        print(f"  - {method}")
    
    print("🐍 Python methods (for comparison):")
    for method in config['python_methods']['base_methods']:
        print(f"  - {method}")
    
    print(f"\n📊 Test configuration:")
    print(f"  - ODE problems: {config['ode_problems']}")
    print(f"  - Noise levels: {config['noise_levels']}")
    print(f"  - Data size: {config['data_config']['data_size']}")
    print(f"  - Derivative orders: {config['data_config']['derivative_orders']}")
    
    print(f"\n🚀 Ready to run!")
    print(f"  python run_unified_benchmark.py")

def restore_original():
    """Restore the original configuration."""
    try:
        with open('benchmark_config.json.original', 'r') as f:
            config = json.load(f)
        
        with open('benchmark_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✅ Restored original configuration")
        return True
    except FileNotFoundError:
        print("❌ No backup found (benchmark_config.json.original)")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "restore":
        restore_original()
    else:
        enable_julia_aaa_only()
        print("\nTo restore original config later:")
        print("  python enable_julia_aaa_only.py restore")