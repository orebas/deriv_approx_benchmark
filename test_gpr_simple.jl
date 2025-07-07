#!/usr/bin/env julia

# Simplified test to check if GPR produces exploding predictions on Van der Pol
using DelimitedFiles
using Statistics
using Printf

println("Simple GPR Behavior Test on Van der Pol x2(t)")
println("=" ^ 50)

# Check what the benchmark actually does
include("src/approximation_methods.jl")

# Load Van der Pol data
data_file = "test_data/vanderpol/noise_1.0e-6/noisy_data.csv"
data, header = readdlm(data_file, ',', header=true)
t = vec(data[:, 1])
x2 = vec(data[:, 3])  # x2(t) - velocity

# Load true derivatives
truth_file = "test_data/vanderpol/noise_0.0/truth_data.csv"
truth_data, _ = readdlm(truth_file, ',', header=true)
d3_x2_true = vec(truth_data[:, 9])  # True 3rd derivative

println("Data loaded: $(length(t)) points")
println(@sprintf("True d³x2/dt³ range: [%.1f, %.1f]", minimum(d3_x2_true), maximum(d3_x2_true)))

# Create a minimal benchmark config
config = BenchmarkConfig(
    noise_level = 1e-6,
    derivative_orders = 3,
    gpr_jitter = 1e-8,
    gpr_noise_threshold = 1e-5,
    spline_order = 5,
    aaa_tol_low = 0.1,
    aaa_tol_high = 1e-14,
    aaa_max_degree = 48
)

println("\nCreating GPR approximation...")
try
    # Create GPR approximation using the actual benchmark code
    approx_func = create_gpr_approximation(t, x2, config)
    
    println("GPR created successfully!")
    
    # Test predictions at a few points
    println("\nTesting 3rd derivative predictions:")
    println("t      pred_d³       true_d³      error")
    println("-" ^ 50)
    
    errors = Float64[]
    predictions = Float64[]
    
    for i in [1, 50, 100, 150, 200]
        t_test = t[i]
        try
            pred = nth_deriv_at(approx_func, 3, t_test)
            true_val = d3_x2_true[i]
            error = abs(pred - true_val)
            
            push!(errors, error)
            push!(predictions, pred)
            
            println(@sprintf("%.3f  %12.2e  %12.2e  %12.2e", 
                    t_test, pred, true_val, error))
            
            if abs(pred) > 1e6
                println("      🚨 EXPLODING PREDICTION DETECTED!")
            end
        catch e
            println(@sprintf("%.3f  FAILED: %s", t_test, e))
        end
    end
    
    if !isempty(errors)
        rmse = sqrt(mean(errors.^2))
        max_pred = maximum(abs.(predictions))
        
        println(@sprintf("\nRMSE: %.2e", rmse))
        println(@sprintf("Max |prediction|: %.2e", max_pred))
        
        if rmse > 1e6
            println("\n🚨 CONFIRMED: GPR is producing million-scale errors!")
            println("This matches your benchmark result of 1.5M RMSE.")
        elseif rmse > 1e3
            println("\n⚠️ Large errors detected but not at 1.5M scale.")
        else
            println("\n✓ Errors seem reasonable. Issue might be elsewhere.")
        end
    end
    
catch e
    println("\nGPR creation failed!")
    println("Error: ", e)
    println("\nThis might explain the benchmark failures.")
end

# Also test what happens if we force AAA fallback
println("\n" * "-"^50)
println("Testing AAA fallback (which GPR uses when unstable):")
try
    aaa_approx = create_aaa_approximation(t, x2, config, high_precision=true)
    
    # Test one prediction
    pred_aaa = nth_deriv_at(aaa_approx, 3, t[100])
    println(@sprintf("AAA prediction at t=%.3f: %.2e", t[100], pred_aaa))
    
catch e
    println("AAA also failed: ", e)
end