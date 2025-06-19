#!/usr/bin/env julia

"""
Run only TVDiff method by temporarily modifying the config and using the existing benchmark script
"""

using JSON
using CSV
using DataFrames

# Load the current config
config = JSON.parsefile("benchmark_config.json")

# Save original Julia methods
original_methods = copy(config["julia_methods"])

# Temporarily set to only TVDiff
config["julia_methods"] = ["TVDiff"]

# Write temporary config
temp_config_file = "temp_tvdiff_config.json"
open(temp_config_file, "w") do f
    JSON.print(f, config, 4)
end

# Backup existing results
results_file = joinpath(config["output_config"]["results_dir"], "julia_raw_benchmark.csv")
backup_file = results_file * ".backup"
if isfile(results_file)
    cp(results_file, backup_file, force=true)
    existing_results = CSV.read(results_file, DataFrame)
    println("Backed up $(nrow(existing_results)) existing results")
else
    existing_results = DataFrame()
end

# Run the benchmark with TVDiff only
println("Running TVDiff benchmark...")
try
    run(`julia benchmark_derivatives.jl --config $temp_config_file`)
catch e
    println("Error running benchmark: $e")
end

# Merge results
if isfile(results_file)
    new_results = CSV.read(results_file, DataFrame)
    
    # Combine with previous results (excluding any old TVDiff results)
    if nrow(existing_results) > 0
        old_without_tvdiff = filter(row -> row.method != "TVDiff", existing_results)
        combined = vcat(old_without_tvdiff, new_results)
    else
        combined = new_results
    end
    
    # Save combined results
    CSV.write(results_file, combined)
    println("Combined results saved. Total rows: $(nrow(combined))")
    
    # Clean up backup
    rm(backup_file, force=true)
else
    println("No new results generated")
    # Restore backup if it exists
    if isfile(backup_file)
        mv(backup_file, results_file, force=true)
    end
end

# Clean up temp config
rm(temp_config_file, force=true)

println("Done!")