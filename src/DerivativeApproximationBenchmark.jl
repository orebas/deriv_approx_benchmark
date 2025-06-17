"""
DerivativeApproximationBenchmark.jl

Comprehensive benchmark of derivative approximation methods for noisy ODE data.
Accompanies the paper: "Using Gaussian Process Regression for ODE Parameter Estimation with Noisy Data"

This module provides tools to:
- Generate synthetic ODE data with controlled noise
- Apply various approximation methods (GPR, AAA, splines, etc.)
- Evaluate approximation quality for function values and derivatives up to order 5
- Export results in tidy format (CSV or JSON)
"""
module DerivativeApproximationBenchmark

using ODEParameterEstimation
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using Statistics
using DataFrames
using CSV
using JSON
using OrderedCollections
using Dates
using Printf
using Random

# Re-export key functions
export run_benchmark, BenchmarkConfig, load_builtin_example

# Configuration
Base.@kwdef struct BenchmarkConfig
    # ODE system configuration
    example_name::String = "lv_periodic"
    custom_example::Union{Nothing, Function} = nothing
    
    # Data generation
    noise_level::Float64 = 1e-3
    noise_type::String = "additive"  # "additive" or "multiplicative"
    data_size::Int = 51
    time_interval::Union{Nothing, Vector{Float64}} = nothing
    random_seed::Int = 42
    
    # Methods to benchmark
    methods::Vector{String} = ["GPR", "AAA", "AAA_lowpres", "LOESS", "BSpline5", "TVDiff"]
    derivative_orders::Int = 5
    
    # Output configuration
    output_format::String = "csv"  # "csv" or "json"
    output_dir::String = "./results"
    experiment_name::String = "benchmark_" * Dates.format(now(), "yyyymmdd_HHMMSS")
    
    # Method-specific parameters
    gpr_jitter::Float64 = 1e-8
    gpr_noise_threshold::Float64 = 1e-5
    aaa_tol_low::Float64 = 0.1
    aaa_tol_high::Float64 = 1e-14
    aaa_max_degree::Int = 48
    loess_span::Float64 = 0.2
    spline_order::Int = 5
    
    # Computation
    verbose::Bool = true
end

# Include additional source files
include("data_generation.jl")
include("approximation_methods.jl")
include("evaluation.jl")
include("tidy_output.jl")
include("builtin_examples.jl")

"""
    run_benchmark(config::BenchmarkConfig)

Run the complete benchmark with the specified configuration.
Returns a DataFrame with results in tidy format.
"""
function run_benchmark(config::BenchmarkConfig = BenchmarkConfig())
    # Set random seed for reproducibility
    Random.seed!(config.random_seed)
    
    # Create output directory
    mkpath(config.output_dir)
    
    # Log configuration
    if config.verbose
        println("\n" * "="^60)
        println("DERIVATIVE APPROXIMATION BENCHMARK")
        println("="^60)
        println("Configuration:")
        println("  Example: $(config.example_name)")
        println("  Noise level: $(config.noise_level)")
        println("  Data size: $(config.data_size)")
        println("  Methods: $(join(config.methods, ", "))")
        println("  Output: $(config.output_format)")
        println("="^60 * "\n")
    end
    
    # Load ODE system
    if config.verbose
        println("Loading ODE system...")
    end
    
    if !isnothing(config.custom_example)
        pep = config.custom_example()
    else
        pep = load_builtin_example(config.example_name)
    end
    
    # Generate datasets
    if config.verbose
        println("Generating datasets...")
    end
    
    datasets = generate_datasets(pep, config)
    
    # Run approximations
    if config.verbose
        println("\nRunning approximation methods...")
    end
    
    results = evaluate_all_methods(datasets, pep, config)
    
    # Convert to tidy format
    if config.verbose
        println("\nConverting to tidy format...")
    end
    
    tidy_df = results_to_tidy_dataframe(results, config)
    
    # Save results
    output_path = save_results(tidy_df, config)
    
    if config.verbose
        println("\nResults saved to: $output_path")
        println("\nBenchmark complete!")
        print_summary_statistics(tidy_df)
    end
    
    return tidy_df
end

"""
    print_summary_statistics(df::DataFrame)

Print a summary of the benchmark results.
"""
function print_summary_statistics(df::DataFrame)
    println("\n" * "="^60)
    println("SUMMARY STATISTICS")
    println("="^60)
    
    # Group by method and derivative order
    grouped = groupby(df, [:method, :derivative_order])
    summary = combine(grouped, 
        :rmse => mean => :mean_rmse,
        :rmse => std => :std_rmse,
        :mae => mean => :mean_mae,
        :max_error => mean => :mean_max_error
    )
    
    # Sort by derivative order then RMSE
    sort!(summary, [:derivative_order, :mean_rmse])
    
    # Print results by derivative order
    for d in unique(summary.derivative_order)
        println("\n$(d == 0 ? "Function values" : "Derivative order $d"):")
        println("-"^40)
        
        subset = summary[summary.derivative_order .== d, :]
        
        for row in eachrow(subset)
            @printf("  %-15s  RMSE: %.2e (±%.2e)  MAE: %.2e\n", 
                row.method, 
                row.mean_rmse, 
                row.std_rmse, 
                row.mean_mae)
        end
    end
end

end # module