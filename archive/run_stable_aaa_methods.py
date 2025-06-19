#!/usr/bin/env python3
"""
Run just the stable AAA methods (AAA_TwoStage) to generate results that can be added to existing data
"""

import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
import traceback

from comprehensive_methods_library import AAA_TwoStage_Approximator

def run_stable_aaa_on_existing_data():
    """Run stable AAA methods on existing test data"""
    
    print("Running stable AAA methods on existing test data...")
    
    # Load configuration to get test parameters
    with open('benchmark_config.json', 'r') as f:
        config = json.load(f)
    
    ode_problems = config['ode_problems']
    noise_levels = config['noise_levels']
    derivative_orders = config['data_config']['derivative_orders']
    
    results = []
    
    for ode_name in ode_problems:
        for noise_level in noise_levels:
            data_path = f"test_data/{ode_name}/noise_{noise_level}"
            
            if not os.path.exists(data_path):
                print(f"  Skipping {ode_name}/{noise_level} - no test data")
                continue
                
            print(f"\n  Testing {ode_name} with noise {noise_level}...")
            
            try:
                # Load data
                noisy_df = pd.read_csv(f"{data_path}/noisy_data.csv")
                truth_df = pd.read_csv(f"{data_path}/truth_data.csv")
                
                t = noisy_df['t'].values
                
                # Test on all observables
                observables = [col for col in noisy_df.columns if col != 't']
                
                for obs in observables:
                    y = noisy_df[obs].values
                    
                    print(f"    Observable: {obs}")
                    
                    # Test AAA_TwoStage
                    method_name = "AAA_TwoStage"
                    print(f"      Testing {method_name}...")
                    
                    try:
                        approximator = AAA_TwoStage_Approximator(t, y, enable_refinement=True)
                        
                        start_time = pd.Timestamp.now()
                        approximator.fit()
                        fit_time = (pd.Timestamp.now() - start_time).total_seconds()
                        
                        if approximator.success:
                            start_time = pd.Timestamp.now()
                            result = approximator.evaluate(t, max_derivative=derivative_orders)
                            eval_time = (pd.Timestamp.now() - start_time).total_seconds()
                            
                            # Calculate errors for each derivative order
                            for d in range(derivative_orders + 1):
                                d_key = 'y' if d == 0 else f'd{d}'
                                
                                if d_key in result:
                                    pred_vals = result[d_key]
                                    
                                    # Get true values
                                    if d == 0:
                                        true_col = obs
                                    else:
                                        true_col = f"d{d}_{obs}"
                                    
                                    if true_col in truth_df.columns:
                                        true_vals = truth_df[true_col].values
                                        
                                        # Calculate errors
                                        rmse = np.sqrt(np.mean((pred_vals - true_vals)**2))
                                        mae = np.mean(np.abs(pred_vals - true_vals))
                                        max_error = np.max(np.abs(pred_vals - true_vals))
                                        
                                        # Add result
                                        results.append({
                                            'method': method_name,
                                            'noise_level': noise_level,
                                            'derivative_order': d,
                                            'rmse': rmse,
                                            'mae': mae,
                                            'max_error': max_error,
                                            'eval_time': eval_time,
                                            'fit_time': fit_time,
                                            'success': True,
                                            'category': 'Python',
                                            'observable': obs,
                                            'test_case': ode_name
                                        })
                                        
                                        print(f"        d{d}: RMSE={rmse:.6f}")
                                    else:
                                        print(f"        d{d}: No truth data")
                            
                            # Print refinement info if available
                            if hasattr(approximator, 'stage2_bic') and approximator.stage2_bic is not None:
                                print(f"        Refinement applied: BIC {approximator.stage1_bic:.1f} -> {approximator.stage2_bic:.1f}")
                            else:
                                print(f"        No refinement (Stage 1 BIC: {approximator.stage1_bic:.1f})")
                                
                        else:
                            print(f"        FAILED to fit")
                            # Add failure record
                            results.append({
                                'method': method_name,
                                'noise_level': noise_level,
                                'derivative_order': 0,
                                'rmse': np.nan,
                                'mae': np.nan,
                                'max_error': np.nan,
                                'eval_time': 0.0,
                                'fit_time': fit_time,
                                'success': False,
                                'category': 'Python',
                                'observable': obs,
                                'test_case': ode_name
                            })
                            
                    except Exception as e:
                        print(f"        ERROR: {e}")
                        # Add error record
                        results.append({
                            'method': method_name,
                            'noise_level': noise_level,
                            'derivative_order': 0,
                            'rmse': np.nan,
                            'mae': np.nan,
                            'max_error': np.nan,
                            'eval_time': 0.0,
                            'fit_time': 0.0,
                            'success': False,
                            'category': 'Python',
                            'observable': obs,
                            'test_case': ode_name
                        })
                        
            except Exception as e:
                print(f"    ERROR loading data: {e}")
                continue
    
    return results

def merge_with_existing_results(new_results):
    """Merge new results with existing Python benchmark results"""
    
    # Load existing results
    existing_file = "results/python_raw_benchmark.csv"
    if os.path.exists(existing_file):
        print(f"\nMerging with existing results from {existing_file}...")
        existing_df = pd.read_csv(existing_file)
        
        # Remove any existing AAA_TwoStage results to avoid duplicates
        existing_df = existing_df[existing_df['method'] != 'AAA_TwoStage']
        
        # Create new results DataFrame
        new_df = pd.DataFrame(new_results)
        
        # Combine
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Sort by test case, noise level, method for consistent ordering
        combined_df = combined_df.sort_values(['test_case', 'noise_level', 'method', 'derivative_order'])
        
        # Save back to the same file
        combined_df.to_csv(existing_file, index=False)
        
        print(f"✓ Merged results saved to {existing_file}")
        print(f"  Total methods: {len(combined_df['method'].unique())}")
        print(f"  AAA_TwoStage entries: {len(combined_df[combined_df['method'] == 'AAA_TwoStage'])}")
        
    else:
        print(f"\nNo existing results file found, creating new one...")
        new_df = pd.DataFrame(new_results)
        new_df.to_csv(existing_file, index=False)
        print(f"✓ New results saved to {existing_file}")

def main():
    print("Running Stable AAA Methods")
    print("=" * 40)
    
    # Run stable AAA methods
    results = run_stable_aaa_on_existing_data()
    
    if results:
        print(f"\n✓ Generated {len(results)} result entries")
        
        # Show summary
        success_count = sum(1 for r in results if r['success'])
        print(f"  Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
        
        # Merge with existing results
        merge_with_existing_results(results)
        
        print("\n✓ Results merged! You can now run the unifying script.")
        
    else:
        print("\n✗ No results generated")

if __name__ == "__main__":
    main()