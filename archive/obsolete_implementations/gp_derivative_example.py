#!/usr/bin/env python3
"""
Reusable Example: Using Enhanced GP for Derivative Estimation

This script provides a clear, minimal, and CORRECT example of how to use the 
`EnhancedGPApproximator` from `enhanced_gp_methods.py`.
"""

import numpy as np
import matplotlib.pyplot as plt
from enhanced_gp_methods import EnhancedGPApproximator

def main():
    """Main function to run the example."""
    
    # 1. GENERATE SAMPLE DATA
    print("Step 1: Generating sample noisy data...")
    np.random.seed(42)
    t = np.linspace(0, 10, 150)
    y_clean = np.sin(t) + 0.5 * np.cos(2.5 * t)
    noise = np.random.normal(0, 0.1, t.shape)
    y_noisy = y_clean + noise
    print("Data generated.")

    # 2. INITIALIZE THE GP APPROXIMATOR
    print("\nStep 2: Initializing the GP Approximator...")
    
    # The EnhancedGPApproximator takes the training data (t, y) and a `kernel_type` string.
    # The available kernel types are: 'rbf_iso', 'matern_1.5', 'matern_2.5', 'periodic'.
    # We'll use 'matern_2.5' as it was a top performer in our benchmarks.
    gp_approximator = EnhancedGPApproximator(
        t, 
        y_noisy,
        name="GP_Matern_2.5_Example", 
        kernel_type='matern_2.5'
    )
    print(f"Initialized '{gp_approximator.name}' with kernel type '{gp_approximator.kernel_type}'.")

    # 3. EVALUATE THE MODEL TO GET DERIVATIVES
    print("\nStep 3: Evaluating the model (fitting and predicting)...")
    
    # The .evaluate() method fits the model and computes the function and derivatives.
    results = gp_approximator.evaluate(t_eval=t, max_derivative=2)
    
    if results['success']:
        print("Evaluation successful!")
        print(f"Fit time: {gp_approximator.fit_time:.4f} seconds")
    else:
        print("Evaluation failed. Exiting.")
        return

    # 4. USE AND VISUALIZE THE RESULTS
    print("\nStep 4: Plotting the results...")
    
    y_pred = results['y']
    d1_pred = results['d1'] # First derivative
    d2_pred = results['d2'] # Second derivative
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot Function
    axs[0].scatter(t, y_noisy, label="Noisy Data", s=10, c='gray', alpha=0.7)
    axs[0].plot(t, y_clean, label="True Function", linestyle='--', color='k')
    axs[0].plot(t, y_pred, label="GP Fit", color='red')
    axs[0].set_title("Function Approximation")
    axs[0].legend()
    
    # Plot 1st Derivative
    d1_true = np.cos(t) - 1.25 * np.sin(2.5 * t)
    axs[1].plot(t, d1_true, label="True 1st Derivative", linestyle='--', color='k')
    axs[1].plot(t, d1_pred, label="GP 1st Derivative", color='blue')
    axs[1].set_title("First Derivative")
    axs[1].legend()
    
    # Plot 2nd Derivative
    d2_true = -np.sin(t) - 3.125 * np.cos(2.5 * t)
    axs[2].plot(t, d2_true, label="True 2nd Derivative", linestyle='--', color='k')
    axs[2].plot(t, d2_pred, label="GP 2nd Derivative", color='green')
    axs[2].set_title("Second Derivative")
    axs[2].legend()
    
    plt.tight_layout()
    output_filename = "gp_derivative_example.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved to '{output_filename}'.")


if __name__ == "__main__":
    main() 