#!/usr/bin/env julia

"""
Standalone script to run derivative approximation benchmarks.

Usage:
    julia run_benchmark.jl [options]

Examples:
    julia run_benchmark.jl --noise 0.01 --methods GPR,AAA,LOESS
    julia run_benchmark.jl --example sir --datasize 101 --output results/sir_test
    julia run_benchmark.jl --format json --noise 0.001
"""



using ArgParse

# Add project source to load path and import the benchmark module
push!(LOAD_PATH, "src")
using DerivativeApproximationBenchmark

function parse_commandline()
    s = ArgParseSettings()
    
    @add_arg_table! s begin
        "--example"
            help = "ODE system to use (lv_periodic, sir, biomd6, simple_oscillator)"
            arg_type = String
            default = "lv_periodic"
        
        "--noise"
            help = "Noise level to add to data"
            arg_type = Float64
            default = 1e-3
        
        "--noise-type"
            help = "Type of noise (additive, multiplicative)"
            arg_type = String
            default = "additive"
        
        "--datasize"
            help = "Number of data points"
            arg_type = Int
            default = 51
        
        "--methods"
            help = "Comma-separated list of methods to benchmark"
            arg_type = String
            default = "GPR,AAA,AAA_lowpres,LOESS,BSpline5"
        
        "--derivatives"
            help = "Maximum derivative order to evaluate"
            arg_type = Int
            default = 5
        
        "--format"
            help = "Output format (csv, json)"
            arg_type = String
            default = "csv"
        
        "--output"
            help = "Output directory"
            arg_type = String
            default = "./results"
        
        "--name"
            help = "Experiment name (auto-generated if not provided)"
            arg_type = String
            default = ""
        
        "--seed"
            help = "Random seed for reproducibility"
            arg_type = Int
            default = 42
        
        "--quiet"
            help = "Suppress progress output"
            action = :store_true
    end
    
    return parse_args(s)
end

function main()
    args = parse_commandline()
    
    # Parse methods list
    methods = split(args["methods"], ",")
    
    # Create experiment name if not provided
    experiment_name = if args["name"] == ""
        "benchmark_$(args["example"])_noise$(args["noise"])_n$(args["datasize"])"
    else
        args["name"]
    end
    
    # Create configuration
    config = BenchmarkConfig(
        example_name = args["example"],
        noise_level = args["noise"],
        noise_type = args["noise-type"],
        data_size = args["datasize"],
        methods = methods,
        derivative_orders = args["derivatives"],
        output_format = args["format"],
        output_dir = args["output"],
        experiment_name = experiment_name,
        random_seed = args["seed"],
        verbose = !args["quiet"]
    )
    
    # Run benchmark
    results = run_benchmark(config)
    
    if !args["quiet"]
        println("\nBenchmark complete!")
        println("Results saved to: $(config.output_dir)/$(experiment_name).$(config.output_format)")
    end
    
    return results
end

# Run if called as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
