#!/usr/bin/env julia

"""
Comprehensive parameter sweep for derivative approximation study.

This script systematically evaluates all combinations of:
- Noise levels: [0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1]
- Data sizes: [51, 101, 201, 401, 801]
- Methods: [GPR, AAA, AAA_lowpres, LOESS, BSpline5]
- Derivative orders: [0, 1, 2, 3, 4, 5]
- Observables: [x, y] from Lotka-Volterra system

Results are saved with systematic naming for easy analysis.
"""

push!(LOAD_PATH, "src")
using DerivativeApproximationBenchmark
using DataFrames
using CSV
using Printf

function run_comprehensive_study()
    # Parameter grid
    noise_levels = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    data_sizes = [51, 101, 201, 401, 801]
    methods = ["GPR", "AAA", "AAA_lowpres", "LOESS", "BSpline5"]
    
    # Create results directory
    results_dir = "results/comprehensive_study_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
    mkpath(results_dir)
    
    # Storage for all results
    all_results = DataFrame()
    
    total_runs = length(noise_levels) * length(data_sizes)
    current_run = 0
    
    println("="^70)
    println("COMPREHENSIVE DERIVATIVE APPROXIMATION STUDY")
    println("="^70)
    println("Total parameter combinations: $total_runs")
    println("Methods: $(join(methods, ", "))")
    println("Noise levels: $(join(noise_levels, ", "))")
    println("Data sizes: $(join(data_sizes, ", "))")
    println("Output directory: $results_dir")
    println("="^70)
    
    # Loop through all combinations
    for noise in noise_levels
        for data_size in data_sizes
            current_run += 1
            
            # Create configuration
            config = BenchmarkConfig(
                example_name = "lv_periodic",
                noise_level = noise,
                data_size = data_size,
                methods = methods,
                derivative_orders = 5,
                output_format = "csv",
                output_dir = results_dir,
                experiment_name = "sweep_lv_periodic_n$(noise)_d$(data_size)",
                random_seed = 42,
                verbose = false  # Suppress individual run output
            )
            
            # Progress indicator
            progress = round(100 * current_run / total_runs, digits=1)
            println("[$current_run/$total_runs] ($progress%) Running: noise=$noise, data_size=$data_size")
            
            try
                # Run benchmark
                results = run_benchmark(config)
                
                # Add run metadata
                results[!, :run_id] = current_run
                results[!, :noise_level_actual] = noise
                results[!, :data_size_actual] = data_size
                
                # Append to master results
                if nrow(all_results) == 0
                    all_results = results
                else
                    all_results = vcat(all_results, results)
                end
                
                # Print brief summary
                mean_rmse = mean(results.rmse)
                println("    → Mean RMSE: $(round(mean_rmse, sigdigits=3))")
                
            catch e
                println("    → ERROR: $e")
                continue
            end
        end
    end
    
    # Save consolidated results
    consolidated_file = joinpath(results_dir, "consolidated_results.csv")
    CSV.write(consolidated_file, all_results)
    
    println("\n" * "="^70)
    println("STUDY COMPLETE!")
    println("="^70)
    println("Total successful runs: $(length(unique(all_results.run_id)))")
    println("Total data points: $(nrow(all_results))")
    println("Consolidated results: $consolidated_file")
    
    # Quick summary statistics
    println("\nQuick Summary by Method (mean RMSE across all conditions):")
    println("-"^50)
    method_summary = combine(groupby(all_results, :method), :rmse => mean => :mean_rmse)
    sort!(method_summary, :mean_rmse)
    
    for row in eachrow(method_summary)
        @printf("  %-15s: %.3e\n", row.method, row.mean_rmse)
    end
    
    return all_results, results_dir
end

# Run if called as script
if abspath(PROGRAM_FILE) == @__FILE__
    results, output_dir = run_comprehensive_study()
end