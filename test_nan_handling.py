#!/usr/bin/env python3
"""Test how NaN values affect the benchmark results"""

import numpy as np

# Simulate the benchmark calculation with NaN values
def test_error_calculation_with_nans():
    # Simulate predictions with some NaN values (like AAA methods produce)
    y_pred = np.array([1.0, np.nan, 2.0, np.nan, 3.0, 4.0, np.nan])
    y_true = np.array([1.1, 1.5, 2.1, 2.5, 3.1, 4.1, 4.5])
    
    print("Testing error calculations with NaN values")
    print("="*50)
    print(f"y_pred: {y_pred}")
    print(f"y_true: {y_true}")
    
    # This is what happens in run_full_benchmark.py lines 158-161
    errors = y_pred - y_true
    print(f"\nerrors: {errors}")
    
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    max_error = np.max(np.abs(errors))
    
    print(f"\nError metrics:")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"Max error: {max_error}")
    
    # Check if the results would be written to CSV
    print(f"\nWould these be written to CSV?")
    print(f"RMSE is NaN: {np.isnan(rmse)}")
    print(f"MAE is NaN: {np.isnan(mae)}")
    print(f"Max error is NaN: {np.isnan(max_error)}")

if __name__ == "__main__":
    test_error_calculation_with_nans()