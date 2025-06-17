#!/usr/bin/env python3
"""
Unified benchmark runner that uses a single configuration file.
Controls Julia and Python benchmarks from one place.
"""

import json
import subprocess
import sys
import os
import shutil
from pathlib import Path

def load_config(config_file="benchmark_config.json"):
    """Load configuration from JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)

def update_julia_config(config):
    """Update Julia benchmark script with config values."""
    
    # Read the current Julia file
    julia_file = "benchmark_derivatives.jl"
    with open(julia_file, 'r') as f:
        content = f.read()
    
    # Replace ODE problems list
    ode_list_str = '[\n\t\t' + ',\n\t\t'.join([f'"{ode}"' for ode in config['ode_problems']]) + ',\n\t]'
    
    # Find and replace the ode_problems_to_test array
    import re
    pattern = r'ode_problems_to_test = \[.*?\]'
    replacement = f'ode_problems_to_test = {ode_list_str}'
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Replace noise levels
    noise_list_str = '[' + ', '.join([str(n) for n in config['noise_levels']]) + ']'
    pattern = r'noise_levels = \[.*?\]'
    replacement = f'noise_levels = {noise_list_str}'
    content = re.sub(pattern, replacement, content)
    
    # Replace data size
    pattern = r'data_size = \d+'
    replacement = f'data_size = {config["data_config"]["data_size"]}'
    content = re.sub(pattern, replacement, content)
    
    # Replace derivative orders
    pattern = r'derivative_orders = \d+'
    replacement = f'derivative_orders = {config["data_config"]["derivative_orders"]}'
    content = re.sub(pattern, replacement, content)
    
    # Replace methods in BenchmarkConfig constructor
    julia_methods_str = '[' + ', '.join([f'"{m}"' for m in config['julia_methods']]) + ']'
    pattern = r'methods = \[.*?\],'
    replacement = f'methods = {julia_methods_str},'
    content = re.sub(pattern, replacement, content)
    
    # Write back
    with open(julia_file, 'w') as f:
        f.write(content)
    
    print(f"✓ Updated {julia_file} with config values")

def update_python_config(config):
    """Create a config file that Python scripts can read."""
    
    # Create a simple Python config file
    python_config = {
        'methods': config['python_methods']['base_methods'] + config['python_methods']['enhanced_gp_methods'],
        'base_methods': config['python_methods']['base_methods'],
        'enhanced_gp_methods': config['python_methods']['enhanced_gp_methods'],
        'data_config': config['data_config'],
        'output_config': config['output_config']
    }
    
    with open('python_benchmark_config.json', 'w') as f:
        json.dump(python_config, f, indent=2)
    
    print("✓ Created python_benchmark_config.json")

def clean_results(config):
    """Clean previous results if requested."""
    if config['runtime_options']['clean_before_run']:
        dirs_to_clean = [
            config['output_config']['results_dir'],
            config['output_config']['unified_analysis_dir'], 
            config['output_config']['test_data_dir']
        ]
        
        for dir_name in dirs_to_clean:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
                print(f"✓ Cleaned {dir_name}/")

def run_julia_benchmark(config, config_file="benchmark_config.json"):
    """Run Julia benchmark."""
    if not config['runtime_options']['run_julia']:
        print("⏭️  Skipping Julia benchmark")
        return True
        
    print("\n🚀 Running Julia benchmark...")
    try:
        # Pass the config file path as a command-line argument
        cmd = ['julia', 'benchmark_derivatives.jl', '--config', config_file]
        result = subprocess.run(cmd, 
                              capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            print("✅ Julia benchmark completed successfully")
            return True
        else:
            print(f"❌ Julia benchmark failed with return code {result.returncode}")
            print("STDERR:", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("❌ Julia benchmark timed out (1 hour)")
        return False
    except Exception as e:
        print(f"❌ Julia benchmark error: {e}")
        return False

def run_python_benchmark(config):
    """Run Python benchmark."""
    if not config['runtime_options']['run_python']:
        print("⏭️  Skipping Python benchmark")
        return True
        
    print("\n🐍 Running Python benchmark...")
    try:
        result = subprocess.run(['python3', 'run_full_benchmark.py'], 
                              capture_output=True, text=True, timeout=7200)
        if result.returncode == 0:
            print("✅ Python benchmark completed successfully")
            return True
        else:
            print(f"❌ Python benchmark failed with return code {result.returncode}")
            print("STDERR:", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("❌ Python benchmark timed out (2 hours)")
        return False
    except Exception as e:
        print(f"❌ Python benchmark error: {e}")
        return False

def create_unified_analysis(config):
    """Create unified analysis."""
    if not config['runtime_options']['create_unified_analysis']:
        print("⏭️  Skipping unified analysis")
        return True
        
    print("\n📊 Creating unified analysis...")
    try:
        result = subprocess.run(['python3', 'create_unified_comparison.py'], 
                              capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print("✅ Unified analysis completed successfully")
            return True
        else:
            print(f"❌ Unified analysis failed with return code {result.returncode}")
            print("STDERR:", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("❌ Unified analysis timed out (10 minutes)")
        return False
    except Exception as e:
        print(f"❌ Unified analysis error: {e}")
        return False

def main():
    """Main function."""
    print("🎯 UNIFIED DERIVATIVE APPROXIMATION BENCHMARK")
    print("=" * 50)
    
    # Load configuration
    try:
        config = load_config()
        print(f"✓ Loaded configuration")
        print(f"  - ODE problems: {len(config['ode_problems'])}")
        print(f"  - Noise levels: {len(config['noise_levels'])}")
        print(f"  - Julia methods: {len(config['julia_methods'])}")
        print(f"  - Python methods: {len(config['python_methods']['base_methods']) + len(config['python_methods']['enhanced_gp_methods'])}")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return 1
    
    # Clean results
    clean_results(config)
    
    # Run benchmarks
    julia_success = run_julia_benchmark(config)
    python_success = run_python_benchmark(config)
    
    # Create unified analysis if both succeeded
    if julia_success and python_success:
        analysis_success = create_unified_analysis(config)
    else:
        print("⚠️  Skipping unified analysis due to benchmark failures")
        analysis_success = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 BENCHMARK SUMMARY")
    print(f"Julia benchmark: {'✅ SUCCESS' if julia_success else '❌ FAILED'}")
    print(f"Python benchmark: {'✅ SUCCESS' if python_success else '❌ FAILED'}")
    print(f"Unified analysis: {'✅ SUCCESS' if analysis_success else '❌ FAILED'}")
    
    if julia_success and python_success and analysis_success:
        print("\n🎉 ALL BENCHMARKS COMPLETED SUCCESSFULLY!")
        print(f"📁 Results available in {config['output_config']['unified_analysis_dir']}/")
        return 0
    else:
        print("\n⚠️  Some benchmarks failed. Check logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())