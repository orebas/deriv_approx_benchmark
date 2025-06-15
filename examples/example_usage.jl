# Example usage of the derivative approximation benchmark

using Pkg
Pkg.activate(dirname(@__DIR__))

push!(LOAD_PATH, joinpath(dirname(@__DIR__), "src"))
using DerivativeApproximationBenchmark
using DataFrames
using CSV

# Example 1: Basic benchmark with default settings
println("Running basic benchmark...")
config = BenchmarkConfig()
results = run_benchmark(config)

# Example 2: Custom configuration for paper figures
println("\nRunning custom benchmark for paper...")
paper_config = BenchmarkConfig(
    example_name = "lv_periodic",
    noise_level = 0.01,  # 1% noise
    noise_type = "additive",
    data_size = 101,
    methods = ["GPR", "AAA", "LOESS", "BSpline5"],
    derivative_orders = 5,
    output_format = "csv",
    output_dir = "./paper_results",
    experiment_name = "paper_figure_1",
    verbose = true
)

paper_results = run_benchmark(paper_config)

# Example 3: Noise sensitivity analysis
println("\nRunning noise sensitivity analysis...")
noise_levels = [1e-4, 1e-3, 1e-2, 5e-2]

for noise in noise_levels
    config = BenchmarkConfig(
        example_name = "lv_periodic",
        noise_level = noise,
        data_size = 51,
        methods = ["GPR", "AAA"],
        output_dir = "./sensitivity_results",
        experiment_name = "sensitivity_noise_$(noise)",
        verbose = false
    )
    
    println("  Testing noise level: $noise")
    run_benchmark(config)
end

# Example 4: Compare different ODE systems
println("\nComparing different ODE systems...")
for example in ["lv_periodic", "sir", "simple_oscillator"]
    config = BenchmarkConfig(
        example_name = example,
        noise_level = 0.01,
        data_size = 51,
        output_dir = "./system_comparison",
        experiment_name = "compare_$(example)",
        verbose = false
    )
    
    println("  Testing system: $example")
    run_benchmark(config)
end

# Example 5: Load and analyze results
println("\nAnalyzing results...")
df = CSV.read("./paper_results/paper_figure_1.csv", DataFrame)

# Get average RMSE by method and derivative order
using Statistics
summary = combine(
    groupby(df, [:method, :derivative_order]),
    :rmse => mean => :mean_rmse,
    :mae => mean => :mean_mae
)

println("\nAverage RMSE by method and derivative order:")
println(summary)

# Find best method for each derivative order
best_methods = combine(
    groupby(summary, :derivative_order),
    sdf -> sdf[argmin(sdf.mean_rmse), :]
)

println("\nBest method for each derivative order:")
println(best_methods)

println("\nExample usage complete!")