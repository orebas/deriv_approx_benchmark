#!/usr/bin/env julia

"""
Simple runner for the derivative approximation benchmark.
This version loads ODEParameterEstimation from the parent directory.
"""

# Add parent directory to load path for ODEParameterEstimation
push!(LOAD_PATH, dirname(@__DIR__))

# Add src directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, "src"))

# Load the benchmark module
using DerivativeApproximationBenchmark

# Run a simple example
println("Running simple benchmark example...")

config = BenchmarkConfig(
    example_name = "lv_periodic",
    noise_level = 0.01,
    data_size = 21,  # Smaller for quick test
    methods = ["GPR", "AAA"],  # Just two methods for now
    output_format = "csv",
    output_dir = "./results",
    experiment_name = "simple_test",
    verbose = true
)

results = run_benchmark(config)

println("\nDone! Results saved to ./results/simple_test.csv")
println("You can examine the results with:")
println("  using CSV, DataFrames")
println("  df = CSV.read(\"./results/simple_test.csv\", DataFrame)")
println("  display(df)")