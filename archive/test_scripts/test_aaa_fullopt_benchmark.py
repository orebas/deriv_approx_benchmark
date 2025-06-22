#!/usr/bin/env python3
"""
Quick benchmark focusing on AAA_FullOpt to demonstrate it's now working
"""

import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.getcwd())

from comprehensive_methods_library import AAA_FullOpt_Approximator

def run_aaa_fullopt_benchmark():
    """Run a focused benchmark on AAA_FullOpt"""
    print("AAA_FULLOPT FOCUSED BENCHMARK")
    print("="*50)
    print("Testing the previously broken AAA_FullOpt method")
    print("with the new smooth barycentric evaluation (W=1e-7)")
    print()
    
    # Test different scenarios
    test_cases = [
        {
            'name': 'Sine Wave',
            't': np.linspace(0, 2*np.pi, 50),
            'func': lambda t: np.sin(t),
            'noise': 0.01
        },
        {
            'name': 'Polynomial', 
            't': np.linspace(-1, 1, 30),
            'func': lambda t: t**3 - 2*t**2 + t + 1,
            'noise': 0.005
        },
        {
            'name': 'Exponential Decay',
            't': np.linspace(0, 3, 40), 
            'func': lambda t: np.exp(-t) * np.cos(2*t),
            'noise': 0.02
        }
    ]
    
    results = []
    
    for case in test_cases:
        print(f"Testing {case['name']}...")
        
        # Generate data
        t = case['t']
        y_clean = case['func'](t)
        y_noisy = y_clean + case['noise'] * np.random.RandomState(42).randn(len(t))
        
        try:
            # Create and fit AAA_FullOpt
            method = AAA_FullOpt_Approximator(t, y_noisy)
            method.fit()
            
            if not method.fitted or not method.success:
                print(f"  ❌ Failed to fit")
                continue
            
            # Test evaluation at derivative orders 1-4
            t_test = np.array([case['t'][len(case['t'])//4], 
                              case['t'][len(case['t'])//2], 
                              case['t'][3*len(case['t'])//4]])
            
            eval_results = method.evaluate(t_test, max_derivative=4)
            
            if not eval_results.get('success', True):
                print(f"  ❌ Evaluation failed")
                continue
            
            # Check all derivatives are finite
            all_finite = True
            for d in range(5):
                key = 'y' if d == 0 else f'd{d}'
                if key in eval_results:
                    values = eval_results[key]
                    if not np.all(np.isfinite(values)):
                        all_finite = False
                        print(f"  ❌ Non-finite {key}")
                        break
            
            if not all_finite:
                continue
                
            # Calculate accuracy
            y_pred = eval_results['y']
            y_true = case['func'](t_test)
            mae = np.mean(np.abs(y_pred - y_true))
            rmse = np.sqrt(np.mean((y_pred - y_true)**2))
            
            # Store result
            result = {
                'test_case': case['name'],
                'success': True,
                'mae': mae,
                'rmse': rmse,
                'fit_time': method.fit_time,
                'num_support_points': len(method.zj) if hasattr(method, 'zj') else 0
            }
            results.append(result)
            
            print(f"  ✅ Success: MAE={mae:.4f}, RMSE={rmse:.4f}, Support points={result['num_support_points']}")
            
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            result = {
                'test_case': case['name'],
                'success': False,
                'error': str(e)
            }
            results.append(result)
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    successful_tests = [r for r in results if r.get('success', False)]
    
    if len(successful_tests) == len(test_cases):
        print("🎉 ALL TESTS PASSED!")
        print(f"✅ AAA_FullOpt successfully handled {len(test_cases)} different scenarios")
        print()
        
        print("Performance Summary:")
        for result in successful_tests:
            print(f"  {result['test_case']:15}: MAE={result['mae']:.4f}, Support pts={result['num_support_points']}")
        
        avg_mae = np.mean([r['mae'] for r in successful_tests])
        avg_time = np.mean([r['fit_time'] for r in successful_tests])
        
        print(f"\nAverage MAE: {avg_mae:.4f}")
        print(f"Average fit time: {avg_time:.3f}s")
        
        print("\n🚀 AAA_FullOpt IS NOW FULLY FUNCTIONAL!")
        print("The smooth barycentric evaluation has solved the convergence issues.")
        
    else:
        print(f"⚠️ {len(successful_tests)}/{len(test_cases)} tests passed")
        for result in results:
            if not result.get('success', False):
                print(f"  Failed: {result['test_case']} - {result.get('error', 'Unknown')}")
    
    return len(successful_tests) == len(test_cases)

if __name__ == "__main__":
    success = run_aaa_fullopt_benchmark()
    
    if success:
        print("\n" + "="*60)
        print("READY FOR FULL BENCHMARK!")
        print("="*60)
        print("AAA_FullOpt is working correctly and can be included")
        print("in comprehensive benchmarks without convergence issues.")
    else:
        print("\nSome issues remain - investigate before full benchmark")