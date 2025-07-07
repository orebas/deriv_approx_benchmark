#!/usr/bin/env julia

# Final test: Run the actual benchmark code on Van der Pol x2(t)
println("Testing GPR on Van der Pol x2(t) using benchmark code")
println("=" ^ 60)

# Include the actual benchmark modules
include("src/DerivativeApproximationBenchmark.jl")
using .DerivativeApproximationBenchmark

# Load the exact data that caused the 1.5M error
using DelimitedFiles
data, _ = readdlm("test_data/vanderpol/noise_1.0e-6/noisy_data.csv", ',', header=true)
t = vec(data[:, 1])
y = vec(data[:, 3])  # x2(t)

# Load true 3rd derivatives
truth_data, _ = readdlm("test_data/vanderpol/noise_0.0/truth_data.csv", ',', header=true)
d3_true = vec(truth_data[:, 9])

println("Data: $(length(t)) points")
println("True d³x2/dt³ range: [$(minimum(d3_true)), $(maximum(d3_true))]")

# Create config matching benchmark
config = DerivativeApproximationBenchmark.BenchmarkConfig(
    noise_level = 1e-6,
    derivative_orders = 3,
    gpr_jitter = 1e-8,
    gpr_noise_threshold = 1e-5
)

println("\nRunning GPR approximation...")
try
    # Use the exact function from the benchmark
    result = DerivativeApproximationBenchmark.evaluate_method("GPR", t, y, t, config)
    
    if haskey(result, "d3")
        predictions = result["d3"]
        errors = predictions - d3_true
        rmse = sqrt(mean(errors.^2))
        max_pred = maximum(abs.(predictions))
        
        println("\nResults:")
        println("RMSE: $(rmse)")
        println("Max |prediction|: $(max_pred)")
        println("Expected from benchmark: 1.5355880956940672e6")
        
        # Show some sample predictions
        println("\nSample predictions at key points:")
        for i in [1, 50, 100, 114, 150, 200]
            println("t=$(t[i]): pred=$(predictions[i]), true=$(d3_true[i])")
        end
        
        if rmse > 1e6
            println("\n✅ CONFIRMED: GPR produces million-scale RMSE!")
            
            # Analyze the predictions
            println("\nPrediction analysis:")
            println("- Min prediction: $(minimum(predictions))")
            println("- Max prediction: $(maximum(predictions))")
            println("- Std of predictions: $(std(predictions))")
            println("- Number of |pred| > 1e6: $(sum(abs.(predictions) .> 1e6))")
        end
    else
        println("No d3 predictions returned!")
    end
    
catch e
    println("\nError during GPR evaluation:")
    println(e)
    
    # This might actually be informative - if GPR is falling back to AAA
    # due to numerical issues, that would explain different results
end

println("\nDone.")