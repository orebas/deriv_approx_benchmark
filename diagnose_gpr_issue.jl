#!/usr/bin/env julia

# Minimal script to diagnose why GPR gives 1.5M error on vanderpol x2(t)
# This replicates the exact conditions from the benchmark

# Load the same dependencies as benchmark_derivatives.jl
using GaussianProcesses
using Statistics
using LinearAlgebra
using ForwardDiff
using Optim
using LineSearches
using DelimitedFiles
using Printf
using Suppressor

println("Diagnosing GPR 1.5M Error on Van der Pol x2(t)")
println("=" ^ 60)

# Load data exactly as benchmark would
data_file = "test_data/vanderpol/noise_1.0e-6/noisy_data.csv"
data, _ = readdlm(data_file, ',', header=true)
t = vec(data[:, 1])
y = vec(data[:, 3])  # x2(t)

println("Loaded $(length(t)) data points")

# Replicate create_gpr_approximation logic
y_mean = mean(y)
y_std = std(y)
y_normalized = (y .- y_mean) ./ y_std

println(@sprintf("Normalization: mean=%.3f, std=%.3f", y_mean, y_std))

# Initial hyperparameters (same as benchmark)
initial_lengthscale = log(std(t) / 8)
initial_variance = 0.0
initial_noise = -2.0
gpr_jitter = 1e-8

# Create kernel and GP
kernel = SEIso(initial_lengthscale, initial_variance)
y_jittered = y_normalized .+ gpr_jitter * randn(length(y_normalized))

println("\nCreating GP with initial hyperparameters:")
println(@sprintf("  lengthscale: exp(%.3f) = %.3f", initial_lengthscale, exp(initial_lengthscale)))
println(@sprintf("  variance: exp(%.3f) = %.3f", initial_variance, exp(initial_variance)))
println(@sprintf("  noise: exp(%.3f) = %.3e", initial_noise, exp(initial_noise)))

gp = GP(t, y_jittered, MeanZero(), kernel, initial_noise)

# Optimize with suppression (as in benchmark)
println("\nOptimizing hyperparameters...")
@suppress begin
    try
        GaussianProcesses.optimize!(gp; 
            method = LBFGS(linesearch = LineSearches.BackTracking())
        )
    catch e
        println("Optimization error: ", e)
    end
end

# Check final hyperparameters
println("\nFinal hyperparameters:")
println(@sprintf("  lengthscale: exp(%.6f) = %.3e", gp.kernel.ℓ2, exp(gp.kernel.ℓ2)))
println(@sprintf("  variance: exp(%.6f) = %.3e", gp.kernel.σ2, exp(gp.kernel.σ2)))
println(@sprintf("  noise: %.6f", gp.logNoise))

# Check noise threshold (benchmark falls back to AAA if noise too low)
noise_level = exp(gp.logNoise)
gpr_noise_threshold = 1e-5
if noise_level < gpr_noise_threshold
    println("\n⚠️ Noise level ($(noise_level)) < threshold ($(gpr_noise_threshold))")
    println("Benchmark would fall back to AAA here!")
end

# Create approximation function
function make_approx(gp_model, mean_val, std_val)
    function approx(x)
        μ, _ = GaussianProcesses.predict_y(gp_model, [x])
        return μ[1] * std_val + mean_val
    end
    return approx
end

approx_func = make_approx(gp, y_mean, y_std)

# Test 3rd derivative using ForwardDiff (as benchmark does)
println("\nComputing 3rd derivatives with ForwardDiff...")

function compute_3rd_deriv(f, x)
    # Nested ForwardDiff as in the benchmark
    d1f(z) = ForwardDiff.derivative(f, z)
    d2f(z) = ForwardDiff.derivative(d1f, z)
    d3f(z) = ForwardDiff.derivative(d2f, z)
    return d3f(x)
end

# Test at several points
test_indices = [1, 50, 100, 114, 150, 200]  # 114 is around t=2.825 where d3 is extreme
predictions = Float64[]
errors = Float64[]

println("\nt_idx    t       pred_d³        |pred|")
println("-" ^ 45)

for idx in test_indices
    t_test = t[idx]
    try
        pred = compute_3rd_deriv(approx_func, t_test)
        push!(predictions, pred)
        
        println(@sprintf("%3d    %.3f   %12.2e   %12.2e", 
                idx, t_test, pred, abs(pred)))
        
        if abs(pred) > 1e6
            println("                              🚨 EXPLODING!")
        end
    catch e
        println(@sprintf("%3d    %.3f   FAILED: %s", idx, t_test, e))
    end
end

# Calculate RMSE if we have predictions
if !isempty(predictions)
    # Load true values
    truth_data, _ = readdlm("test_data/vanderpol/noise_0.0/truth_data.csv", ',', header=true)
    d3_true = vec(truth_data[:, 9])
    
    # Calculate errors at test points
    for (i, idx) in enumerate(test_indices[1:length(predictions)])
        push!(errors, predictions[i] - d3_true[idx])
    end
    
    test_rmse = sqrt(mean(errors.^2))
    max_pred = maximum(abs.(predictions))
    
    println("\n" * "="^60)
    println(@sprintf("Test RMSE: %.2e", test_rmse))
    println(@sprintf("Max |prediction|: %.2e", max_pred))
    println(@sprintf("Benchmark reported: %.2e", 1.5355880956940672e6))
    
    if test_rmse > 1e6 || max_pred > 1e6
        println("\n✅ REPRODUCED: Confirmed million-scale predictions!")
    else
        println("\n❓ Could not reproduce the exact error magnitude")
    end
end

# Additional diagnostics
println("\nKey diagnostics:")
println("- Effective lengthscale / data spacing: ", exp(gp.kernel.ℓ2) / mean(diff(t)))
println("- Kernel matrix size: ", length(t), " × ", length(t))

# Check condition number
try
    K = GaussianProcesses.cov(gp.kernel, t[1:10], t[1:10])  # Just first 10 for speed
    println(@sprintf("- Sample condition number: %.2e", cond(K)))
catch
    println("- Could not compute condition number")
end