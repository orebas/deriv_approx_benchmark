#!/usr/bin/env python3
"""
Debug script for AAA_SmoothBary issues
"""

import numpy as np
import traceback
import pandas as pd
import os
from comprehensive_methods_library import AAA_SmoothBarycentric_Approximator

def test_smoothbary_basic():
    """Test basic functionality of AAA_SmoothBary"""
    
    print("Testing AAA_SmoothBary basic functionality...")
    
    # Create simple test data
    t = np.linspace(0, 2*np.pi, 51)
    y_clean = np.sin(t)
    y_noisy = y_clean + 0.01 * np.random.randn(len(t))
    
    try:
        # Test instantiation
        approximator = AAA_SmoothBarycentric_Approximator(t, y_noisy)
        print("✓ Instantiation successful")
        
        # Test fitting
        approximator.fit()
        print(f"✓ Fitting successful, success={approximator.success}")
        
        if approximator.success:
            # Test evaluation
            result = approximator.evaluate(t, max_derivative=2)
            print(f"✓ Evaluation successful")
            print(f"  y shape: {result['y'].shape}")
            print(f"  d1 shape: {result['d1'].shape}")
            print(f"  d2 shape: {result['d2'].shape}")
            
            # Check for NaN values
            y_nan_count = np.sum(np.isnan(result['y']))
            d1_nan_count = np.sum(np.isnan(result['d1']))
            d2_nan_count = np.sum(np.isnan(result['d2']))
            
            print(f"  NaN counts - y: {y_nan_count}, d1: {d1_nan_count}, d2: {d2_nan_count}")
            
            if y_nan_count == 0:
                rmse = np.sqrt(np.mean((result['y'] - y_clean)**2))
                print(f"  RMSE: {rmse:.6f}")
                return True
            else:
                print("✗ Evaluation contains NaN values")
                return False
        else:
            print("✗ Fitting failed")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_smoothbary_on_real_data():
    """Test AAA_SmoothBary on real benchmark data"""
    
    print("\nTesting AAA_SmoothBary on benchmark data...")
    
    # Try to load some existing test data
    test_cases = [
        ("lv_periodic", 0.01),
        ("vanderpol", 0.001), 
        ("brusselator", 0.01)
    ]
    
    results = []
    
    for ode_name, noise_level in test_cases:
        data_path = f"test_data/{ode_name}/noise_{noise_level}"
        if not os.path.exists(data_path):
            print(f"  Skipping {ode_name}/{noise_level} - no data")
            continue
            
        try:
            print(f"\n  Testing {ode_name} with noise {noise_level}...")
            
            # Load data
            noisy_df = pd.read_csv(f"{data_path}/noisy_data.csv")
            truth_df = pd.read_csv(f"{data_path}/truth_data.csv")
            
            t = noisy_df['t'].values
            
            # Test on first observable
            observables = [col for col in noisy_df.columns if col != 't']
            obs = observables[0]
            y = noisy_df[obs].values
            
            print(f"    Observable: {obs}, data points: {len(t)}")
            
            # Test AAA_SmoothBary
            approximator = AAA_SmoothBarycentric_Approximator(t, y)
            approximator.fit()
            
            if approximator.success:
                result = approximator.evaluate(t, max_derivative=2)
                
                # Calculate errors if truth data exists
                if obs in truth_df.columns:
                    rmse_y = np.sqrt(np.mean((result['y'] - truth_df[obs].values)**2))
                    print(f"    ✓ Success - RMSE: {rmse_y:.6f}")
                    
                    results.append({
                        'ode': ode_name,
                        'noise': noise_level,
                        'observable': obs,
                        'success': True,
                        'rmse': rmse_y
                    })
                else:
                    print(f"    ✓ Success - no truth data for error calculation")
                    results.append({
                        'ode': ode_name,
                        'noise': noise_level,
                        'observable': obs,
                        'success': True,
                        'rmse': None
                    })
            else:
                print(f"    ✗ Failed to fit")
                results.append({
                    'ode': ode_name,
                    'noise': noise_level,
                    'observable': obs,
                    'success': False,
                    'rmse': None
                })
                
        except Exception as e:
            print(f"    ✗ Error: {e}")
            results.append({
                'ode': ode_name,
                'noise': noise_level,
                'observable': obs if 'obs' in locals() else 'unknown',
                'success': False,
                'rmse': None
            })
    
    return results

def main():
    print("Debugging AAA_SmoothBary")
    print("=" * 40)
    
    # Test basic functionality
    basic_success = test_smoothbary_basic()
    
    # Test on real data if basic test passes
    if basic_success:
        real_data_results = test_smoothbary_on_real_data()
        
        print(f"\nSummary of real data tests:")
        success_count = sum(1 for r in real_data_results if r['success'])
        total_count = len(real_data_results)
        print(f"  Success rate: {success_count}/{total_count}")
        
        if success_count > 0:
            successful = [r for r in real_data_results if r['success'] and r['rmse'] is not None]
            if successful:
                avg_rmse = np.mean([r['rmse'] for r in successful])
                print(f"  Average RMSE: {avg_rmse:.6f}")
                
    print("\nDebug complete.")

if __name__ == "__main__":
    np.random.seed(42)
    main()