#!/usr/bin/env julia

# Simple test to verify TVDiff works
println("Testing TVDiff...")

using CSV
using DataFrames
using JSON

# Include the fixed benchmark script
include("benchmark_derivatives.jl")

# Test on one simple case
println("Loading test data...")
test_data = CSV.read("test_data/lv_periodic/noise_0.001/noisy_data.csv", DataFrame)
time_points = test_data.t
y_data = test_data[!, "r(t)"]

println("Creating BenchmarkConfig...")
config = BenchmarkConfig(
    "lv_periodic", 0.001, "gaussian", length(time_points), ["TVDiff"], 2, 
    "results", "test", 42, true,
    1e-8, 1e-5, 0.1, 1e-14, 48, 0.2, 5
)

println("Testing TVDiff approximation...")
try
    approx = create_tvdiff_approximation(time_points, y_data, config)
    println("✓ TVDiff created successfully")
    
    # Test evaluation
    test_val = approx(0.5)
    println("✓ Function evaluation: f(0.5) = $test_val")
    
    # Test if it has deriv field
    if hasfield(typeof(approx), :deriv)
        println("✓ Has deriv field")
        deriv_val = approx.deriv(0.5, 1)
        println("✓ Derivative evaluation: f'(0.5) = $deriv_val")
    else
        println("✗ Missing deriv field")
    end
    
    println("✅ TVDiff test passed!")
    
catch e
    println("❌ TVDiff test failed: $e")
    for (i, frame) in enumerate(Base.catch_backtrace())
        if i > 5  # Limit backtrace length
            break
        end
        println("  $frame")
    end
end