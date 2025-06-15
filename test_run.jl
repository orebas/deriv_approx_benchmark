#!/usr/bin/env julia

"""
Test run of the benchmark using the parent ODEParameterEstimation environment
"""

# Load dependencies that should already be available in ODEParameterEstimation
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using Statistics
using DataFrames
using OrderedCollections
using Dates
using Printf
using Random

# Add parent directory for ODEParameterEstimation
push!(LOAD_PATH, dirname(@__DIR__))
using ODEParameterEstimation

# Simple config struct (define before includes)
Base.@kwdef struct BenchmarkConfig
    example_name::String = "lv_periodic"
    noise_level::Float64 = 0.01
    data_size::Int = 11  # Very small for test
    methods::Vector{String} = ["AAA"]  # Start with just AAA
    derivative_orders::Int = 1  # Just first derivative
    output_format::String = "csv"
    output_dir::String = "./results"
    experiment_name::String = "test_run"
    random_seed::Int = 42
    verbose::Bool = true
    # Method parameters
    gpr_jitter::Float64 = 1e-8
    gpr_noise_threshold::Float64 = 1e-5
    aaa_tol_low::Float64 = 0.1
    aaa_tol_high::Float64 = 1e-14
    aaa_max_degree::Int = 48
    loess_span::Float64 = 0.2
    spline_order::Int = 5
    noise_type::String = "additive"
    time_interval::Union{Nothing, Vector{Float64}} = nothing
    custom_example::Union{Nothing, Function} = nothing
end

# Load examples first
include("../src/examples/load_examples.jl")

# Try loading the benchmark components individually
include("src/data_generation.jl")
include("src/approximation_methods.jl")
include("src/evaluation.jl")
include("src/tidy_output.jl") 
include("src/builtin_examples.jl")

# Simple benchmark runner
function simple_benchmark()
    # Set seed
    Random.seed!(42)
    
    println("Loading ODE example...")
    pep = lv_periodic()
    
    # Create config
    config = BenchmarkConfig()
    
    println("Generating datasets...")
    datasets = generate_datasets(pep, config)
    
    println("Running approximation...")
    results = evaluate_all_methods(datasets, pep, config)
    
    println("Converting to tidy format...")
    # Simple tidy conversion without full metadata
    rows = []
    t_eval = datasets.clean["t"]
    
    for (obs_key, obs_results) in results
        if obs_key == "metadata"
            continue
        end
        
        for (method_name, method_results) in obs_results
            if !haskey(method_results, "errors")
                continue
            end
            
            for d in 0:config.derivative_orders
                d_key = d == 0 ? "y" : "d$d"
                
                if haskey(method_results["errors"], d_key)
                    error_metrics = method_results["errors"][d_key]
                    
                    row = OrderedDict(
                        "observable" => string(obs_key),
                        "method" => method_name,
                        "derivative_order" => d,
                        "rmse" => error_metrics.rmse,
                        "mae" => error_metrics.mae,
                        "max_error" => error_metrics.max_error
                    )
                    push!(rows, row)
                end
            end
        end
    end
    
    df = DataFrame(rows)
    
    # Create output directory
    mkpath(config.output_dir)
    
    # Save results
    filename = joinpath(config.output_dir, "$(config.experiment_name).csv")
    
    # Manual CSV writing to avoid CSV.jl dependency
    open(filename, "w") do io
        # Write header
        headers = collect(keys(rows[1]))
        println(io, join(headers, ","))
        
        # Write data
        for row in rows
            values = [string(row[h]) for h in headers]
            println(io, join(values, ","))
        end
    end
    
    println("Results saved to: $filename")
    println("\nSummary:")
    display(df)
    
    return df
end

# Run the test
simple_benchmark()