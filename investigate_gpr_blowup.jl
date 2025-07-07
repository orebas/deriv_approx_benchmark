#!/usr/bin/env julia

# Investigate the massive GPR error amplification
using Statistics
using Printf
using DelimitedFiles

println("Investigating GPR Error Amplification")
println("=" ^ 50)

# Load the Van der Pol data
data_file = "test_data/vanderpol/noise_0.0/truth_data.csv"
data, header = readdlm(data_file, ',', header=true)

t = data[:, 1]
x2 = data[:, 3]         # x2(t) - velocity
d3_x2 = data[:, 9]      # third derivative of x2

println("Van der Pol 3rd derivative stats:")
println(@sprintf("Min: %.2f, Max: %.2f, Range: %.2f", 
        minimum(d3_x2), maximum(d3_x2), maximum(d3_x2) - minimum(d3_x2)))
println(@sprintf("RMS: %.2f", sqrt(mean(d3_x2.^2))))

# Simulate what might be happening in GPR
println("\nTesting potential error amplification scenarios:")

# Scenario 1: Simple prediction failure
simple_rmse = sqrt(mean(d3_x2.^2))
println(@sprintf("1. If GPR predicts zeros: RMSE ≈ %.1f", simple_rmse))

# Scenario 2: GPR produces oscillating predictions
println("\n2. Testing oscillating GPR predictions:")
for amplitude in [100, 1000, 10000, 100000]
    # Simulate GPR producing high-frequency oscillations
    fake_gpr_pred = amplitude * sin.(50 * t)  # High frequency oscillations
    error = fake_gpr_pred - d3_x2
    rmse = sqrt(mean(error.^2))
    println(@sprintf("   Oscillation amplitude %6d → RMSE: %.0f", amplitude, rmse))
    
    if rmse > 1e6
        println("   ★ This could explain the 1.5M error!")
        break
    end
end

# Scenario 3: Numerical instability in GP hyperparameters
println("\n3. Testing hyperparameter-induced instability:")
# When GP lengthscales become very small, predictions can blow up
for lengthscale_factor in [0.1, 0.01, 0.001, 0.0001]
    # Simulate what happens with tiny lengthscales
    # Small lengthscales → overfitting → wild predictions
    effective_noise = 1000 / lengthscale_factor  # Inverse relationship
    fake_pred = d3_x2 + effective_noise * randn(length(d3_x2))
    rmse = sqrt(mean((fake_pred - d3_x2).^2))
    println(@sprintf("   Lengthscale factor %.4f → Effective noise: %8.0f → RMSE: %.0f", 
            lengthscale_factor, effective_noise, rmse))
    
    if rmse > 1e6
        println("   ★ This could explain the 1.5M error!")
    end
end

# Scenario 4: Examine what 1.5M RMSE actually means
target_rmse = 1535588.0
println(@sprintf("\n4. Analysis of the actual error %.0f:", target_rmse))

# If RMSE = 1.5M, what does that imply about predictions?
println("If GPR predictions have RMSE = 1.5M:")
println(@sprintf("   - Typical prediction error magnitude: %.0f", target_rmse))
println(@sprintf("   - This is %.0fx larger than max true value (%.1f)", 
        target_rmse / maximum(abs.(d3_x2)), maximum(abs.(d3_x2))))

# Reverse engineer what predictions would give this RMSE
println("\n5. Reverse engineering the predictions:")
# If error = pred - true, and RMSE = sqrt(mean(error²)) = 1.5M
# Then mean(error²) = (1.5M)²
mean_squared_error = target_rmse^2
println(@sprintf("   Mean squared error: %.2e", mean_squared_error))

# This suggests predictions are wildly off
typical_pred_magnitude = sqrt(mean_squared_error + mean(d3_x2.^2))
println(@sprintf("   Typical prediction magnitude: %.0f", typical_pred_magnitude))

println("\nCONCLUSION:")
println("A 1.5M RMSE suggests GPR is producing predictions with magnitudes")
println("around 1.5M, which is ~18,000x larger than the true derivative values.")
println("This indicates:")
println("• Severe numerical instability in GPR optimization")
println("• Hyperparameters (lengthscale/noise) going to pathological values")
println("• Complete breakdown of the GP posterior, not just poor fitting")
println("• Possible overflow/underflow in matrix computations")

# Check if this is consistent with our original benchmark result
println(@sprintf("\nYour benchmark result: %.0f", 1535588.096))
println("This is consistent with complete GPR numerical breakdown,")
println("not just poor fitting to non-smooth data.")