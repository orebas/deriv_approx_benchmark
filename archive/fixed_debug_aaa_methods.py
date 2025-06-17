#!/usr/bin/env python3
"""
Fixed debug script for AAA_LS and AAA_FullOpt methods
Tests both methods across all available test data to identify failure patterns
"""

import numpy as np
import pandas as pd
import os
import glob
import time
from pathlib import Path
import traceback
import warnings
warnings.filterwarnings('ignore')

# Import our methods
from comprehensive_methods_library import AAALeastSquaresApproximator, AAA_FullOpt_Approximator

class AAADebugger:
    def __init__(self, test_data_dir="test_data"):
        self.test_data_dir = test_data_dir
        self.results = []
        
    def find_all_test_cases(self):
        """Find all available test cases (ODE systems and noise levels)"""
        test_cases = []
        
        # Find all ODE system directories
        base_path = Path(self.test_data_dir)
        if not base_path.exists():
            print(f"Test data directory {self.test_data_dir} not found")
            return test_cases
            
        for ode_dir in base_path.iterdir():
            if not ode_dir.is_dir():
                continue
                
            ode_name = ode_dir.name
            print(f"Found ODE system: {ode_name}")
            
            # Find noise level directories
            for noise_dir in ode_dir.iterdir():
                if not noise_dir.is_dir() or not noise_dir.name.startswith('noise_'):
                    continue
                    
                noise_level = noise_dir.name.replace('noise_', '')
                truth_file = noise_dir / 'truth_data.csv'
                noisy_file = noise_dir / 'noisy_data.csv'
                
                if truth_file.exists() and noisy_file.exists():
                    test_cases.append({
                        'ode_system': ode_name,
                        'noise_level': noise_level,
                        'truth_file': str(truth_file),
                        'noisy_file': str(noisy_file)
                    })
                    print(f"  - Found noise level: {noise_level}")
        
        print(f"Total test cases found: {len(test_cases)}")
        return test_cases
    
    def load_test_data(self, truth_file, noisy_file):
        """Load truth and noisy data from CSV files"""
        try:
            truth_df = pd.read_csv(truth_file)
            noisy_df = pd.read_csv(noisy_file)
            
            # Extract time vector (assuming first column is 't')
            t = truth_df.iloc[:, 0].values
            
            # Get all variable columns (excluding derivatives for now)
            var_columns = [col for col in truth_df.columns if not col.startswith('d') and col != 't']
            
            datasets = {}
            for var_col in var_columns:
                datasets[var_col] = {
                    'truth': truth_df[var_col].values,
                    'noisy': noisy_df[var_col].values,
                    'derivatives': {}
                }
                
                # Extract derivative columns for this variable
                for i in range(1, 5):  # Up to 4th derivative
                    deriv_col = f'd{i}_{var_col}'
                    if deriv_col in truth_df.columns:
                        datasets[var_col]['derivatives'][i] = truth_df[deriv_col].values
            
            return t, datasets
            
        except Exception as e:
            print(f"Error loading data from {truth_file}, {noisy_file}: {e}")
            return None, None
    
    def test_method_on_variable(self, method_class, method_name, t, y_noisy, y_truth, derivatives_truth, test_info):
        """Test a single method on one variable's data"""
        result = {
            'method': method_name,
            'ode_system': test_info['ode_system'],
            'noise_level': test_info['noise_level'],
            'variable': test_info.get('variable', 'unknown'),
            'success': False,
            'fit_time': 0,
            'error_message': None,
            'function_rmse': np.nan,
            'derivative_rmses': {}
        }
        
        try:
            # Create and fit method
            start_time = time.time()
            method = method_class(t, y_noisy, method_name)
            method.fit()
            result['fit_time'] = time.time() - start_time
            result['success'] = method.success
            
            if method.success:
                # Use the evaluate method from base class
                eval_results = method.evaluate(t, max_derivative=4)
                
                # Extract function values
                y_pred = eval_results['y']
                if not np.all(np.isnan(y_pred)):
                    result['function_rmse'] = np.sqrt(np.mean((y_pred - y_truth)**2))
                
                # Extract derivatives
                for deriv_order, deriv_truth in derivatives_truth.items():
                    try:
                        deriv_pred = eval_results.get(f'd{deriv_order}')
                        if deriv_pred is not None and not np.all(np.isnan(deriv_pred)):
                            rmse = np.sqrt(np.mean((deriv_pred - deriv_truth)**2))
                            result['derivative_rmses'][f'd{deriv_order}'] = rmse
                        else:
                            result['derivative_rmses'][f'd{deriv_order}'] = np.nan
                    except Exception as e:
                        result['derivative_rmses'][f'd{deriv_order}'] = np.nan
                        print(f"    Error evaluating derivative {deriv_order}: {e}")
            else:
                result['error_message'] = "Method reported failure"
                
        except Exception as e:
            result['error_message'] = str(e)
            result['success'] = False
            print(f"    Exception in {method_name}: {e}")
            # Print traceback for debugging
            traceback.print_exc()
            
        return result
    
    def run_comprehensive_test(self):
        """Run comprehensive test of AAA methods"""
        print("=== AAA Methods Debug Analysis (Fixed Version) ===")
        print()
        
        # Find all test cases
        test_cases = self.find_all_test_cases()
        if not test_cases:
            print("No test cases found!")
            return
        
        # Define methods to test
        methods_to_test = [
            (AAALeastSquaresApproximator, "AAA_LS"),
            (AAA_FullOpt_Approximator, "AAA_FullOpt")
        ]
        
        total_tests = 0
        successful_tests = 0
        
        print("Starting comprehensive test...")
        print()
        
        for test_case in test_cases:
            print(f"Testing: {test_case['ode_system']} with noise {test_case['noise_level']}")
            
            # Load test data
            t, datasets = self.load_test_data(test_case['truth_file'], test_case['noisy_file'])
            if t is None:
                continue
            
            # Test each variable in the dataset
            for var_name, var_data in datasets.items():
                print(f"  Variable: {var_name}")
                
                y_truth = var_data['truth']
                y_noisy = var_data['noisy']
                derivatives_truth = var_data['derivatives']
                
                # Test each method
                for method_class, method_name in methods_to_test:
                    test_info = test_case.copy()
                    test_info['variable'] = var_name
                    
                    print(f"    Testing {method_name}... ", end='', flush=True)
                    
                    result = self.test_method_on_variable(
                        method_class, method_name, t, y_noisy, y_truth, derivatives_truth, test_info
                    )
                    
                    self.results.append(result)
                    total_tests += 1
                    
                    if result['success']:
                        successful_tests += 1
                        print(f"✓ (RMSE: {result['function_rmse']:.4f})")
                    else:
                        print(f"✗ ({result['error_message']})")
        
        print()
        print(f"=== Summary ===")
        print(f"Total tests run: {total_tests}")
        print(f"Successful tests: {successful_tests}")
        print(f"Success rate: {successful_tests/total_tests*100:.1f}%")
        
        return self.results
    
    def analyze_failures(self):
        """Analyze failure patterns in the results"""
        print("\n=== Failure Analysis ===")
        
        df = pd.DataFrame(self.results)
        
        # Count failures by method
        failure_counts = df.groupby('method')['success'].agg(['count', 'sum']).round(3)
        failure_counts['failure_rate'] = (failure_counts['count'] - failure_counts['sum']) / failure_counts['count']
        print("\nFailure rates by method:")
        print(failure_counts)
        
        # Analyze failed cases
        failed_cases = df[~df['success']]
        if len(failed_cases) > 0:
            print(f"\n{len(failed_cases)} failed cases found:")
            for _, case in failed_cases.iterrows():
                print(f"  {case['method']} on {case['ode_system']}/{case['variable']} "
                      f"(noise: {case['noise_level']}): {case['error_message']}")
        
        # Analyze RMSE distribution for successful cases
        successful_cases = df[df['success']]
        if len(successful_cases) > 0:
            print(f"\nRMSE statistics for successful cases:")
            rmse_stats = successful_cases.groupby('method')['function_rmse'].describe()
            print(rmse_stats)
            
            # Find cases with very high RMSE (potential soft failures)
            high_rmse_threshold = successful_cases['function_rmse'].quantile(0.9)
            high_rmse_cases = successful_cases[successful_cases['function_rmse'] > high_rmse_threshold]
            if len(high_rmse_cases) > 0:
                print(f"\nCases with high RMSE (>{high_rmse_threshold:.4f}):")
                for _, case in high_rmse_cases.iterrows():
                    print(f"  {case['method']} on {case['ode_system']}/{case['variable']} "
                          f"(noise: {case['noise_level']}): RMSE = {case['function_rmse']:.4f}")
    
    def save_results(self, filename="aaa_debug_results_fixed.csv"):
        """Save detailed results to CSV"""
        if not self.results:
            print("No results to save")
            return
            
        # Flatten derivative RMSE results
        flattened_results = []
        for result in self.results:
            base_result = {k: v for k, v in result.items() if k != 'derivative_rmses'}
            
            # Add derivative RMSEs as separate columns
            for deriv_name, rmse_val in result.get('derivative_rmses', {}).items():
                base_result[f'{deriv_name}_rmse'] = rmse_val
            
            flattened_results.append(base_result)
        
        df = pd.DataFrame(flattened_results)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

def main():
    debugger = AAADebugger()
    results = debugger.run_comprehensive_test()
    debugger.analyze_failures()
    debugger.save_results()

if __name__ == "__main__":
    main()