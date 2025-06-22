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
import datetime
import logging
from threading import Thread
import queue
import time

def setup_logging(log_dir):
    """Setup logging to both file and console."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"benchmark_run_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file

def stream_output(proc, prefix=""):
    """Stream output from subprocess in real-time."""
    def reader(pipe, queue, prefix):
        try:
            with pipe:
                for line in iter(pipe.readline, ''):
                    if line:
                        queue.put((prefix, line.rstrip()))
        finally:
            queue.put(None)
    
    q = queue.Queue()
    stdout_thread = Thread(target=reader, args=[proc.stdout, q, f"{prefix}[OUT]"])
    stderr_thread = Thread(target=reader, args=[proc.stderr, q, f"{prefix}[ERR]"])
    stdout_thread.start()
    stderr_thread.start()
    
    # Process output
    active_threads = 2
    while active_threads > 0:
        try:
            item = q.get(timeout=0.1)
            if item is None:
                active_threads -= 1
            else:
                prefix, line = item
                logging.info(f"{prefix} {line}")
        except queue.Empty:
            continue
    
    stdout_thread.join()
    stderr_thread.join()

def load_config(config_file="benchmark_config.json"):
    """Load configuration from JSON file."""
    logging.info(f"Loading configuration from {config_file}")
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
    
    logging.info(f"✓ Updated {julia_file} with config values")

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
    
    logging.info("✓ Created python_benchmark_config.json")

def clean_results(config):
    """Clean previous results if requested."""
    if config['runtime_options']['clean_before_run']:
        dirs_to_clean = [
            config['output_config']['results_dir'],
            config['output_config']['test_data_dir']
        ]
        
        # Don't clean unified_analysis_dir since we're logging there
        for dir_name in dirs_to_clean:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
                logging.info(f"✓ Cleaned {dir_name}/")

def run_julia_benchmark(config, config_file="benchmark_config.json"):
    """Run Julia benchmark."""
    if not config['runtime_options']['run_julia']:
        logging.info("⏭️  Skipping Julia benchmark")
        return True
        
    logging.info("\n🚀 Running Julia benchmark...")
    start_time = time.time()
    
    try:
        # Pass the config file path as a command-line argument
        cmd = ['julia', 'benchmark_derivatives.jl', '--config', config_file]
        logging.info(f"Command: {' '.join(cmd)}")
        
        # Run with real-time output streaming
        proc = subprocess.Popen(cmd, 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              text=True,
                              bufsize=1)
        
        # Stream output in real-time
        stream_output(proc, "JULIA")
        
        # Wait for completion
        return_code = proc.wait()
        elapsed_time = time.time() - start_time
        
        if return_code == 0:
            logging.info(f"✅ Julia benchmark completed successfully in {elapsed_time:.1f} seconds")
            return True
        else:
            logging.error(f"❌ Julia benchmark failed with return code {return_code} after {elapsed_time:.1f} seconds")
            return False
            
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        logging.error(f"❌ Julia benchmark timed out after {elapsed_time:.1f} seconds")
        proc.kill()
        return False
    except Exception as e:
        elapsed_time = time.time() - start_time
        logging.error(f"❌ Julia benchmark error after {elapsed_time:.1f} seconds: {e}")
        return False

def run_python_benchmark(config):
    """Run Python benchmark."""
    if not config['runtime_options']['run_python']:
        logging.info("⏭️  Skipping Python benchmark")
        return True
        
    logging.info("\n🐍 Running Python benchmark...")
    start_time = time.time()
    
    try:
        cmd = ['python3', 'run_full_benchmark.py']
        logging.info(f"Command: {' '.join(cmd)}")
        
        # Run with real-time output streaming
        proc = subprocess.Popen(cmd, 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              text=True,
                              bufsize=1)
        
        # Stream output in real-time
        stream_output(proc, "PYTHON")
        
        # Wait for completion
        return_code = proc.wait()
        elapsed_time = time.time() - start_time
        
        if return_code == 0:
            logging.info(f"✅ Python benchmark completed successfully in {elapsed_time:.1f} seconds")
            return True
        else:
            logging.error(f"❌ Python benchmark failed with return code {return_code} after {elapsed_time:.1f} seconds")
            return False
            
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        logging.error(f"❌ Python benchmark timed out after {elapsed_time:.1f} seconds")
        proc.kill()
        return False
    except Exception as e:
        elapsed_time = time.time() - start_time
        logging.error(f"❌ Python benchmark error after {elapsed_time:.1f} seconds: {e}")
        return False

def create_unified_analysis(config):
    """Create unified analysis."""
    if not config['runtime_options']['create_unified_analysis']:
        logging.info("⏭️  Skipping unified analysis")
        return True
        
    logging.info("\n📊 Creating unified analysis...")
    start_time = time.time()
    
    try:
        cmd = ['python3', 'create_unified_comparison.py']
        logging.info(f"Command: {' '.join(cmd)}")
        
        # Run with real-time output streaming
        proc = subprocess.Popen(cmd, 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              text=True,
                              bufsize=1)
        
        # Stream output in real-time
        stream_output(proc, "ANALYSIS")
        
        # Wait for completion
        return_code = proc.wait()
        elapsed_time = time.time() - start_time
        
        if return_code == 0:
            logging.info(f"✅ Unified analysis completed successfully in {elapsed_time:.1f} seconds")
            return True
        else:
            logging.error(f"❌ Unified analysis failed with return code {return_code} after {elapsed_time:.1f} seconds")
            return False
            
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        logging.error(f"❌ Unified analysis timed out after {elapsed_time:.1f} seconds")
        proc.kill()
        return False
    except Exception as e:
        elapsed_time = time.time() - start_time
        logging.error(f"❌ Unified analysis error after {elapsed_time:.1f} seconds: {e}")
        return False

def main():
    """Main function."""
    # Early setup for unified_analysis directory
    try:
        with open("benchmark_config.json", 'r') as f:
            config_preview = json.load(f)
            log_dir = config_preview['output_config']['unified_analysis_dir']
    except:
        log_dir = "unified_analysis"
    
    # Setup logging
    log_file = setup_logging(log_dir)
    
    logging.info("🎯 UNIFIED DERIVATIVE APPROXIMATION BENCHMARK")
    logging.info("=" * 50)
    logging.info(f"Log file: {log_file}")
    
    # Load configuration
    try:
        config = load_config()
        logging.info(f"✓ Loaded configuration")
        logging.info(f"  - ODE problems: {len(config['ode_problems'])}")
        logging.info(f"  - Noise levels: {len(config['noise_levels'])}")
        logging.info(f"  - Julia methods: {len(config['julia_methods'])}")
        logging.info(f"  - Python methods: {len(config['python_methods']['base_methods']) + len(config['python_methods']['enhanced_gp_methods'])}")
    except Exception as e:
        logging.error(f"❌ Failed to load config: {e}")
        return 1
    
    # Record total start time
    total_start_time = time.time()
    
    # Clean results
    clean_results(config)
    
    # Run benchmarks
    julia_success = run_julia_benchmark(config)
    python_success = run_python_benchmark(config)
    
    # Create unified analysis if both succeeded
    if julia_success and python_success:
        analysis_success = create_unified_analysis(config)
    else:
        logging.warning("⚠️  Skipping unified analysis due to benchmark failures")
        analysis_success = False
    
    # Calculate total time
    total_elapsed = time.time() - total_start_time
    
    # Summary
    logging.info("\n" + "=" * 50)
    logging.info("📋 BENCHMARK SUMMARY")
    logging.info(f"Julia benchmark: {'✅ SUCCESS' if julia_success else '❌ FAILED'}")
    logging.info(f"Python benchmark: {'✅ SUCCESS' if python_success else '❌ FAILED'}")
    logging.info(f"Unified analysis: {'✅ SUCCESS' if analysis_success else '❌ FAILED'}")
    logging.info(f"Total runtime: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
    
    if julia_success and python_success and analysis_success:
        logging.info("\n🎉 ALL BENCHMARKS COMPLETED SUCCESSFULLY!")
        logging.info(f"📁 Results available in {config['output_config']['unified_analysis_dir']}/")
        logging.info(f"📝 Full log: {log_file}")
        return 0
    else:
        logging.error("\n⚠️  Some benchmarks failed. Check logs above.")
        logging.info(f"📝 Full log: {log_file}")
        return 1

if __name__ == "__main__":
    sys.exit(main())