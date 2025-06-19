# evaluation.jl

"""
    benchmark_timing(method_func, n_runs=10)

Benchmark the execution time of a method.

Note: This is a simple timer with warm-up run to account for JIT compilation.
For high-fidelity benchmarking results, consider using BenchmarkTools.jl.
"""
function benchmark_timing(method_func, n_runs=10)
    # Warm-up run to trigger JIT compilation
    method_func()
    
    times = Vector{UInt64}(undef, n_runs)
    
    for i in 1:n_runs
        start = time_ns()
        method_func()
        times[i] = time_ns() - start
    end
    
    # Convert from nanoseconds to seconds for the final report
    times_sec = times ./ 1e9
    
    return (
        mean = mean(times_sec),
        std = std(times_sec),
        min = minimum(times_sec),
        max = maximum(times_sec),
        median = median(times_sec)
    )
end

"""
    validate_approximation(approx_func, t_test, y_test)

Validate an approximation function on test data.
"""
function validate_approximation(approx_func, t_test, y_test)
    predictions = [approx_func(t) for t in t_test]
    
    rmse = sqrt(mean((predictions .- y_test).^2))
    mae = mean(abs.(predictions .- y_test))
    max_error = maximum(abs.(predictions .- y_test))
    
    # Relative errors
    rel_errors = abs.(predictions .- y_test) ./ (abs.(y_test) .+ 1e-10)
    mape = mean(rel_errors) * 100  # Mean absolute percentage error
    
    return (
        rmse = rmse,
        mae = mae,
        max_error = max_error,
        mape = mape,
        predictions = predictions
    )
end

"""
    cross_validate_methods(datasets, pep, config; n_folds=5)

Perform cross-validation for method comparison.
"""
function cross_validate_methods(datasets, pep, config::BenchmarkConfig; n_folds=5)
    n_points = length(datasets.clean["t"])
    fold_size = n_points ÷ n_folds
    
    cv_results = Dict{String, Vector{Dict{String,Float64}}}()
    
    for method in config.methods
        cv_results[method] = []
        
        for fold in 1:n_folds
            # Create train/test split
            test_idx = ((fold-1)*fold_size + 1):min(fold*fold_size, n_points)
            train_idx = setdiff(1:n_points, test_idx)
            
            # Get training data
            t_train = datasets.noisy["t"][train_idx]
            t_test = datasets.noisy["t"][test_idx]
            
            fold_results = Dict()
            
            for (i, mq) in enumerate(pep.measured_quantities)
                key = Num(mq.rhs)
                if key == "t"
                    continue
                end
                
                y_train = datasets.noisy[key][train_idx]
                y_test = datasets.clean[key][test_idx]
                
                try
                    # Create approximation on training data
                    approx_func = evaluate_single_method(method, t_train, y_train, t_test, config)
                    
                    # Validate on test data
                    validation = validate_approximation(approx_func["y"], t_test, y_test)
                    
                    fold_results[key] = validation
                catch e
                    @warn "CV failed for $method on fold $fold" exception=e
                end
            end
            
            push!(cv_results[method], fold_results)
        end
    end
    
    return cv_results
end

"""
    analyze_noise_sensitivity(datasets, pep, config; noise_levels=[1e-4, 1e-3, 1e-2, 1e-1])

Analyze how methods perform under different noise levels.
"""
function analyze_noise_sensitivity(datasets, pep, config::BenchmarkConfig; 
                                 noise_levels=[1e-4, 1e-3, 1e-2, 1e-1])
    
    sensitivity_results = Dict{String, Dict{Float64, Dict}}()
    
    for method in config.methods
        sensitivity_results[method] = Dict()
        
        for noise_level in noise_levels
            # Create new config with different noise
            test_config = BenchmarkConfig(
                config.example_name,
                config.custom_example,
                noise_level,  # Different noise level
                config.noise_type,
                config.data_size,
                config.time_interval,
                config.random_seed,
                [method],  # Just test this method
                config.derivative_orders,
                config.output_format,
                config.output_dir,
                config.experiment_name,
                config.verbose
            )
            
            # Generate new noisy data
            test_datasets = generate_datasets(pep, test_config)
            
            # Evaluate method
            results = evaluate_all_methods(test_datasets, pep, test_config)
            
            sensitivity_results[method][noise_level] = results
        end
    end
    
    return sensitivity_results
end