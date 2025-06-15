# approximation_methods.jl

using GaussianProcesses
using Loess
using BaryRational
using Dierckx
using LinearAlgebra
using Optim
using LineSearches
using ForwardDiff
using Suppressor
using Symbolics

"""
    evaluate_all_methods(datasets, pep, config)

Evaluate all requested approximation methods on the datasets.
"""
function evaluate_all_methods(datasets, pep, config::BenchmarkConfig)
    results = Dict{Any, Dict{String, Any}}()
    
    # Store metadata
    t_eval = datasets.clean["t"]
    results["metadata"] = Dict(
        "t_eval" => t_eval,
        "true_values" => Dict()
    )
    
    # For each observable
    for (i, mq) in enumerate(pep.measured_quantities)
        key = Num(mq.rhs)
        if key == "t"
            continue
        end
        
        if config.verbose
            println("  Processing observable: $key")
        end
        
        results[key] = Dict{String, Any}()
        
        # Store true values for comparison
        true_vals = Dict{String, Vector{Float64}}()
        true_vals["y"] = datasets.clean[key]
        for d in 1:config.derivative_orders
            true_vals["d$d"] = datasets.derivatives["d$(d)_$key"]
        end
        results["metadata"]["true_values"][key] = true_vals
        
        # Get noisy data for fitting
        t = datasets.noisy["t"]
        y = datasets.noisy[key]
        
        # Evaluate each method
        for method in config.methods
            if config.verbose
                print("    - $method...")
            end
            
            start_time = time()
            
            try
                method_result = evaluate_single_method(method, t, y, t_eval, config)
                
                # Add computation time
                method_result["computation_time"] = time() - start_time
                
                # Calculate errors
                method_result["errors"] = calculate_errors(method_result, true_vals)
                
                results[key][method] = method_result
                
                if config.verbose
                    println(" done ($(round(method_result["computation_time"], digits=3))s)")
                end
            catch e
                if config.verbose
                    println(" failed: $(typeof(e))")
                end
                @warn "Method $method failed for $key" exception=(e, catch_backtrace())
            end
        end
    end
    
    return results
end

"""
    evaluate_single_method(method_name, t, y, t_eval, config)

Evaluate a single approximation method.
"""
function evaluate_single_method(method_name::String, t, y, t_eval, config::BenchmarkConfig)
    # Create approximation function
    approx_func = if method_name == "GPR"
        create_gpr_approximation(t, y, config)
    elseif method_name == "AAA"
        create_aaa_approximation(t, y, config, high_precision=true)
    elseif method_name == "AAA_lowpres"
        create_aaa_approximation(t, y, config, high_precision=false)
    elseif method_name == "LOESS"
        create_loess_approximation(t, y, config)
    elseif method_name == "BSpline5"
        create_bspline_approximation(t, y, config)
    else
        error("Unknown method: $method_name")
    end
    
    # Evaluate function and derivatives
    result = Dict{String, Any}()
    
    # Function values
    result["y"] = [approx_func(x) for x in t_eval]
    
    # Derivatives
    for d in 1:config.derivative_orders
        result["d$d"] = [nth_deriv_at(approx_func, d, x) for x in t_eval]
    end
    
    return result
end

"""
    create_gpr_approximation(t, y, config)

Create a Gaussian Process Regression approximation.
"""
function create_gpr_approximation(t, y, config::BenchmarkConfig)
    # Normalize data
    y_mean = mean(y)
    y_std = std(y)
    y_normalized = (y .- y_mean) ./ y_std
    
    # Initial hyperparameters
    initial_lengthscale = log(std(t) / 8)
    initial_variance = 0.0
    initial_noise = -2.0
    
    # Create kernel
    kernel = SEIso(initial_lengthscale, initial_variance)
    
    # Add jitter for numerical stability
    y_jittered = y_normalized .+ config.gpr_jitter * randn(length(y))
    
    # Create and optimize GP
    gp = GP(t, y_jittered, MeanZero(), kernel, initial_noise)
    
    @suppress begin
        GaussianProcesses.optimize!(gp; 
            method = LBFGS(linesearch = LineSearches.BackTracking())
        )
    end
    
    # Check noise level
    noise_level = exp(gp.logNoise.value)
    if noise_level < config.gpr_noise_threshold
        # Fall back to AAA if noise is too low
        return create_aaa_approximation(t, y, config, high_precision=true)
    end
    
    # Create callable function
    function gpr_func(x)
        pred, _ = predict_y(gp, [x])
        return y_std * pred[1] + y_mean
    end
    
    return gpr_func
end

"""
    create_aaa_approximation(t, y, config; high_precision=true)

Create an AAA (Adaptive Antoulas-Anderson) rational approximation.
"""
function create_aaa_approximation(t, y, config::BenchmarkConfig; high_precision=true)
    if high_precision
        # Use aaad with default tolerance
        return aaad(t, y)
    else
        # Low precision version with BIC selection
        y_mean = mean(y)
        y_std = std(y)
        y_normalized = (y .- y_mean) ./ y_std
        
        # Find best approximation using BIC
        best_bic = Inf
        best_approx = nothing
        
        tol = 0.5
        for m in 1:min(config.aaa_max_degree, length(t) ÷ 2)
            tol = tol / 2.0
            
            # Create approximation
            approx = BaryRational.aaa(t, y_normalized, verbose=false, tol=tol)
            
            # Calculate BIC
            residuals = y_normalized .- [BaryRational.evaluate(approx, x) for x in t]
            ssr = sum(abs2, residuals)
            k = 2 * length(approx.x)  # Number of parameters
            n = length(t)
            bic = k * log(n) + n * log(ssr / n + 1e-100)
            
            if bic < best_bic
                best_bic = bic
                best_approx = approx
            end
        end
        
        # Create callable with denormalization
        callable_approx = AAADapprox(best_approx)
        
        function denormalized_aaa(x)
            return y_std * callable_approx(x) + y_mean
        end
        
        return denormalized_aaa
    end
end

"""
    create_loess_approximation(t, y, config)

Create a LOESS (locally weighted regression) approximation.
"""
function create_loess_approximation(t, y, config::BenchmarkConfig)
    # Create LOESS model
    model = loess(collect(t), y, span=config.loess_span)
    
    # Get predictions at data points
    predictions = Loess.predict(model, t)
    
    # Use AAA to create a differentiable function from LOESS predictions
    return aaad(t, predictions)
end

"""
    create_bspline_approximation(t, y, config)

Create a B-spline approximation.
"""
function create_bspline_approximation(t, y, config::BenchmarkConfig)
    # Estimate noise level for smoothing parameter
    n = length(t)
    mean_y = mean(abs.(y))
    expected_noise = config.noise_level * mean_y
    s = n * expected_noise^2  # Expected sum of squared residuals
    
    # Create spline with smoothing
    spl = Spline1D(t, y; k=config.spline_order, s=s)
    
    # Create callable function
    function spline_func(x)
        return evaluate(spl, x)
    end
    
    # Override nth_deriv_at for better performance with splines
    function spline_nth_deriv_at(n::Int, x::Real)
        return derivative(spl, x, nu=n)
    end
    
    # Store the derivative function as a property (hacky but works)
    spline_func.deriv = spline_nth_deriv_at
    
    return spline_func
end

"""
    calculate_errors(predictions, true_values)

Calculate error metrics for predictions vs true values.
"""
function calculate_errors(predictions::Dict, true_values::Dict)
    errors = Dict{String, NamedTuple}()
    
    for (key, true_vals) in true_values
        if haskey(predictions, key)
            pred_vals = predictions[key]
            
            rmse = sqrt(mean((pred_vals .- true_vals).^2))
            mae = mean(abs.(pred_vals .- true_vals))
            max_error = maximum(abs.(pred_vals .- true_vals))
            
            errors[key] = (rmse=rmse, mae=mae, max_error=max_error)
        end
    end
    
    return errors
end

# Override nth_deriv_at for functions with special derivative methods
function nth_deriv_at(f, n::Int, x::Real)
    if hasfield(typeof(f), :deriv)
        return f.deriv(n, x)
    else
        # Default implementation using automatic differentiation
        return ODEParameterEstimation.nth_deriv_at(f, n, x)
    end
end