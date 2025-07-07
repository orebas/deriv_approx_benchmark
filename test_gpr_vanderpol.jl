#!/usr/bin/env julia

# Test GPR on Van der Pol x2(t) to reproduce the massive error
using GaussianProcesses
using DelimitedFiles
using Statistics
using Printf
using LinearAlgebra
using Optim
using LineSearches

println("Testing GPR on Van der Pol x2(t) - 3rd derivative")
println("=" ^ 60)

# Load Van der Pol data
data_file = "test_data/vanderpol/noise_1.0e-6/noisy_data.csv"  # Using slightly noisy data
println("Loading: $data_file")

# Read CSV data
data, header = readdlm(data_file, ',', header=true)
t = vec(data[:, 1])
x2 = vec(data[:, 3])  # x2(t) - velocity component

println("Data shape: $(length(t)) points")
println(@sprintf("x2 range: [%.3f, %.3f]", minimum(x2), maximum(x2)))

# Normalize data (as done in benchmark)
y_mean = mean(x2)
y_std = std(x2)
y_normalized = (x2 .- y_mean) ./ y_std

println(@sprintf("\nNormalization: mean=%.3f, std=%.3f", y_mean, y_std))

# Set up GPR with same parameters as benchmark
initial_lengthscale = log(std(t) / 8)
initial_variance = 0.0
initial_noise = -2.0
gpr_jitter = 1e-8

println("\nInitial hyperparameters:")
println(@sprintf("  lengthscale: %.3f (exp: %.3f)", initial_lengthscale, exp(initial_lengthscale)))
println(@sprintf("  variance: %.3f (exp: %.3f)", initial_variance, exp(initial_variance)))
println(@sprintf("  noise: %.3f (exp: %.3e)", initial_noise, exp(initial_noise)))

# Create kernel and GP
kernel = SEIso(initial_lengthscale, initial_variance)
y_jittered = y_normalized .+ gpr_jitter * randn(length(y_normalized))

println("\nCreating GP...")
gp = GP(t, y_jittered, MeanZero(), kernel, initial_noise)

# Optimize hyperparameters (this is where things might go wrong)
println("\nOptimizing hyperparameters...")
println("Initial log marginal likelihood: ", GaussianProcesses.log_lik(gp))

try
    # Use the same optimizer as the benchmark
    result = GaussianProcesses.optimize!(gp; 
        method = LBFGS(linesearch = LineSearches.BackTracking()),
        iterations = 50,
        show_trace = true
    )
    
    println("\nOptimization result: ", result)
catch e
    println("\nOptimization failed!")
    println("Error: ", e)
    # Continue anyway to see what happens
end

println("\nFinal hyperparameters:")
println(@sprintf("  lengthscale: %.6f (exp: %.3e)", gp.kernel.ℓ2.value, exp(gp.kernel.ℓ2.value)))
println(@sprintf("  variance: %.6f (exp: %.3e)", gp.kernel.σ2.value, exp(gp.kernel.σ2.value)))
println(@sprintf("  noise: %.6f (exp: %.3e)", gp.logNoise.value, exp(gp.logNoise.value)))
println("Final log marginal likelihood: ", GaussianProcesses.log_lik(gp))

# Check condition number of kernel matrix
K = GaussianProcesses.cov(gp.kernel, t, t)
K_with_noise = K + exp(gp.logNoise.value) * I
cond_num = cond(K_with_noise)
println(@sprintf("\nKernel matrix condition number: %.2e", cond_num))
if cond_num > 1e12
    println("⚠️ WARNING: Matrix is poorly conditioned!")
end

# Make predictions for 3rd derivative
println("\nMaking predictions for 3rd derivative...")

# We need to compute derivatives of the GP mean function
# For a GP with SE kernel, the derivative is analytical
function gp_3rd_derivative(gp, x_pred, normalize_factor)
    # This is a simplified calculation - the actual benchmark uses ForwardDiff
    # which might behave differently
    
    try
        # Get the posterior mean at x_pred
        μ = GaussianProcesses.predict_y(gp, [x_pred])[1][1]
        
        # For demonstration, we'll compute numerical derivatives
        # (The actual code uses ForwardDiff which is more sophisticated)
        h = 1e-4
        
        # 3rd order numerical derivative using central differences
        f_plus_3h = GaussianProcesses.predict_y(gp, [x_pred + 3*h])[1][1]
        f_plus_2h = GaussianProcesses.predict_y(gp, [x_pred + 2*h])[1][1]
        f_plus_h = GaussianProcesses.predict_y(gp, [x_pred + h])[1][1]
        f_minus_h = GaussianProcesses.predict_y(gp, [x_pred - h])[1][1]
        f_minus_2h = GaussianProcesses.predict_y(gp, [x_pred - 2*h])[1][1]
        f_minus_3h = GaussianProcesses.predict_y(gp, [x_pred - 3*h])[1][1]
        
        # 3rd derivative approximation
        d3f = (-f_plus_3h + 3*f_plus_2h - 3*f_plus_h + 3*f_minus_h - 3*f_minus_2h + f_minus_3h) / (h^3)
        
        # Denormalize (derivatives scale by 1/std for each order)
        return d3f * normalize_factor / (y_std^3)
    catch e
        println("Prediction failed at x=$x_pred: ", e)
        return NaN
    end
end

# Test predictions at a few points
test_points = [0.1, 0.5, 1.0, 2.0, 2.8, 3.0, 4.0]
println("\nSample predictions:")
println("t      pred_d3")
println("-" * 30)

predictions = Float64[]
for x in test_points
    pred = gp_3rd_derivative(gp, x, 1.0)
    push!(predictions, pred)
    println(@sprintf("%.1f    %.2e", x, pred))
end

# Check if predictions are exploding
max_pred = maximum(abs.(predictions[.!isnan.(predictions)]))
println(@sprintf("\nMax absolute prediction: %.2e", max_pred))

if max_pred > 1e6
    println("🚨 PREDICTIONS ARE EXPLODING! This explains the 1.5M RMSE.")
elseif max_pred > 1e3
    println("⚠️ Predictions are very large but not at 1.5M scale.")
else
    println("✓ Predictions seem reasonable. The issue might be elsewhere.")
end

# Load true 3rd derivative for comparison
truth_file = "test_data/vanderpol/noise_0.0/truth_data.csv"
truth_data, _ = readdlm(truth_file, ',', header=true)
d3_x2_true = vec(truth_data[:, 9])
println(@sprintf("\nTrue d3x2/dt3 range: [%.2f, %.2f]", minimum(d3_x2_true), maximum(d3_x2_true)))

# Additional diagnostics
println("\nDiagnostics:")
println("- Effective lengthscale: ", exp(gp.kernel.ℓ2.value))
println("- Data spacing: ", mean(diff(t)))
println("- Lengthscale / spacing ratio: ", exp(gp.kernel.ℓ2.value) / mean(diff(t)))

if exp(gp.kernel.ℓ2.value) < 0.01 * mean(diff(t))
    println("🚨 Lengthscale is much smaller than data spacing - severe overfitting!")
end

println("\nConclusion:")
println("This test should reveal if GPR is producing million-scale predictions")
println("due to numerical instability or hyperparameter pathology.")