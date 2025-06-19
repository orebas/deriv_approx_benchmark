#!/usr/bin/env python3

"""
Run only AAA_SmoothBary method by temporarily modifying the config and using the existing benchmark script
"""

import json
import subprocess
import pandas as pd
import os
import shutil

# Load the current config
with open('benchmark_config.json', 'r') as f:
    config = json.load(f)

# Save original Python methods
original_methods = config['python_methods']['base_methods'].copy()

# Temporarily set to only AAA_SmoothBary
config['python_methods']['base_methods'] = ['AAA_SmoothBary']
config['python_methods']['enhanced_gp_methods'] = []  # Disable GP methods

# Backup the original config
original_config_file = 'benchmark_config.json'
backup_config_file = 'benchmark_config.json.backup'
shutil.copy2(original_config_file, backup_config_file)

# Write modified config to the main config file
with open(original_config_file, 'w') as f:
    json.dump(config, f, indent=4)

# Backup existing results
results_file = os.path.join(config['output_config']['results_dir'], 'python_raw_benchmark.csv')
backup_file = results_file + '.backup'

if os.path.exists(results_file):
    shutil.copy2(results_file, backup_file)
    existing_results = pd.read_csv(results_file)
    print(f"Backed up {len(existing_results)} existing results")
else:
    existing_results = pd.DataFrame()

# Run the benchmark with AAA_SmoothBary only
print("Running AAA_SmoothBary benchmark...")
try:
    subprocess.run(['python3', 'run_full_benchmark.py'], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error running benchmark: {e}")

# Merge results
if os.path.exists(results_file):
    new_results = pd.read_csv(results_file)
    
    # Combine with previous results (excluding any old AAA_SmoothBary results)
    if len(existing_results) > 0:
        old_without_aaa = existing_results[existing_results['method'] != 'AAA_SmoothBary']
        combined = pd.concat([old_without_aaa, new_results], ignore_index=True)
    else:
        combined = new_results
    
    # Save combined results
    combined.to_csv(results_file, index=False)
    print(f"Combined results saved. Total rows: {len(combined)}")
    
    # Clean up backup
    os.remove(backup_file)
else:
    print("No new results generated")
    # Restore backup if it exists
    if os.path.exists(backup_file):
        shutil.move(backup_file, results_file)

# Restore original config from backup
shutil.copy2(backup_config_file, original_config_file)
os.remove(backup_config_file)

print("Done!")