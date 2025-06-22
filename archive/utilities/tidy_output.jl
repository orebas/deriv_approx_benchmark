# tidy_output.jl

"""
    results_to_tidy_dataframe(results, config)

Convert benchmark results to a tidy DataFrame where each row represents 
a single measurement with all relevant metadata.
"""
function results_to_tidy_dataframe(results::Dict, config::BenchmarkConfig)
    rows = []
    
    # Extract all metadata fields
    for (obs_key, obs_results) in results
        for (method_name, method_results) in obs_results
            # Skip if method failed
            if !haskey(method_results, "errors")
                continue
            end
            
            # Get computation time if available
            comp_time = get(method_results, "computation_time", missing)
            
            # For each derivative order (0 = function value)
            for d in 0:config.derivative_orders
                d_key = d == 0 ? "y" : "d$d"
                
                if haskey(method_results["errors"], d_key)
                    error_metrics = method_results["errors"][d_key]
                    
                    # For each time point
                    t_points = results["metadata"]["t_eval"]
                    true_vals = results["metadata"]["true_values"][obs_key][d_key]
                    pred_vals = method_results[d_key]
                    
                    for (i, t) in enumerate(t_points)
                        row = OrderedDict(
                            # Experiment metadata
                            "experiment_name" => config.experiment_name,
                            "timestamp" => Dates.now(),
                            
                            # Data configuration
                            "example" => config.example_name,
                            "noise_level" => config.noise_level,
                            "noise_type" => config.noise_type,
                            "data_size" => config.data_size,
                            "random_seed" => config.random_seed,
                            
                            # Observable information
                            "observable" => string(obs_key),
                            "derivative_order" => d,
                            
                            # Method information
                            "method" => method_name,
                            "computation_time" => comp_time,
                            
                            # Point-wise data
                            "time" => t,
                            "true_value" => true_vals[i],
                            "predicted_value" => pred_vals[i],
                            "error" => pred_vals[i] - true_vals[i],
                            "absolute_error" => abs(pred_vals[i] - true_vals[i]),
                            
                            # Aggregate error metrics for this derivative order
                            "rmse" => error_metrics.rmse,
                            "mae" => error_metrics.mae,
                            "max_error" => error_metrics.max_error,
                            
                            # Method-specific parameters
                            "method_params" => get_method_params(method_name, config)
                        )
                        
                        push!(rows, row)
                    end
                end
            end
        end
    end
    
    return DataFrame(rows)
end

"""
    get_method_params(method_name, config)

Extract method-specific parameters from config.
"""
function get_method_params(method_name::String, config::BenchmarkConfig)
    params = Dict{String, Any}()
    
    if method_name == "GPR"
        params["jitter"] = config.gpr_jitter
        params["noise_threshold"] = config.gpr_noise_threshold
    elseif method_name == "AAA"
        params["tolerance"] = config.aaa_tol_high
        params["max_degree"] = config.aaa_max_degree
    elseif method_name == "AAA_lowpres"
        params["tolerance"] = config.aaa_tol_low
        params["max_degree"] = config.aaa_max_degree
    elseif method_name == "LOESS"
        params["span"] = config.loess_span
    elseif method_name == "BSpline5"
        params["order"] = config.spline_order
    end
    
    return JSON.json(params)  # Convert to JSON string for storage
end

"""
    save_results(df::DataFrame, config::BenchmarkConfig)

Save results to file in the specified format.
"""
function save_results(df::DataFrame, config::BenchmarkConfig)
    filename = joinpath(
        config.output_dir, 
        config.experiment_name * "." * config.output_format
    )
    
    if config.output_format == "csv"
        CSV.write(filename, df)
    elseif config.output_format == "json"
        # Convert DataFrame to JSON-friendly format
        json_data = Dict(
            "metadata" => Dict(
                "experiment_name" => config.experiment_name,
                "timestamp" => string(Dates.now()),
                "config" => config
            ),
            "data" => [Dict(pairs(row)) for row in eachrow(df)]
        )
        
        open(filename, "w") do io
            JSON.print(io, json_data, 2)
        end
    else
        error("Unknown output format: $(config.output_format)")
    end
    
    # Also save a summary file
    save_summary(df, config)
    
    return filename
end

"""
    save_summary(df::DataFrame, config::BenchmarkConfig)

Save a summary of results grouped by method and derivative order.
"""
function save_summary(df::DataFrame, config::BenchmarkConfig)
    # Group by method, observable, and derivative order
    grouped = groupby(df, [:method, :observable, :derivative_order])
    
    summary = combine(grouped,
        :rmse => first => :rmse,
        :mae => first => :mae,
        :max_error => first => :max_error,
        :computation_time => first => :computation_time,
        nrow => :n_points
    )
    
    # Sort for readability
    sort!(summary, [:observable, :derivative_order, :rmse])
    
    # Save summary
    summary_filename = joinpath(
        config.output_dir,
        config.experiment_name * "_summary.csv"
    )
    
    CSV.write(summary_filename, summary)
    
    # Create a pivot table for easier reading
    pivot_rmse = unstack(summary, [:observable, :derivative_order], :method, :rmse)
    pivot_filename = joinpath(
        config.output_dir,
        config.experiment_name * "_pivot_rmse.csv"
    )
    CSV.write(pivot_filename, pivot_rmse)
end