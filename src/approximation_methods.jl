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
using NoiseRobustDifferentiation

# Wrapper struct for approximation functions with custom derivative methods
struct ApproximationWrapper
    func::Function
    deriv::Function
end

# Make the struct callable, so it behaves like a function for evaluation
(aw::ApproximationWrapper)(x) = aw.func(x)

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
    elseif method_name == "TVDiff"
        create_tvdiff_approximation(t, y, config)
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
        
        # Check if we found a valid approximation
        if best_approx === nothing
            @warn "AAA failed to find valid approximation for input size $(length(t))"
            throw(ArgumentError("Insufficient data points for AAA approximation"))
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
    return ApproximationWrapper(spline_func, spline_nth_deriv_at)
end

"""
    create_tvdiff_approximation(t, y, config)

Create a Total Variation Regularized Differentiation approximation.
"""
function create_tvdiff_approximation(t, y, config::BenchmarkConfig)
    # Calculate dx (grid spacing)
    dx = mean(diff(t))
    
    # TVDiff parameters (more conservative for robustness)
    iter = 25    # Fewer iterations to avoid numerical instability
    α = 0.05     # More regularization for stability
    
    # Estimate noise level for adaptive regularization
    noise_estimate = std(diff(diff(y)))  # Second difference as noise proxy
    adaptive_α = max(α, noise_estimate * 0.1)  # Adapt to noise level
    
    # Get the regularized function values and first derivative
    dy = tvdiff(y, iter, adaptive_α, dx=dx, scale="small", ε=1e-6)
    
    # For higher order derivatives, we need to apply tvdiff iteratively
    derivatives = Dict{Int, Vector{Float64}}()
    derivatives[0] = y
    derivatives[1] = dy
    
    current_deriv = dy
    max_order = min(config.derivative_orders, 3)  # Limit to 3rd order for stability
    
    for d in 2:max_order
        # Apply TVDiff to get the next derivative
        # Use progressively more regularization for higher derivatives
        deriv_α = adaptive_α * (2.0^(d-1))  # Increase regularization exponentially
        
        try
            current_deriv = tvdiff(current_deriv, iter, deriv_α, dx=dx, scale="small", ε=1e-6)
            
            # Check for numerical issues
            if any(isnan.(current_deriv)) || any(isinf.(current_deriv))
                @warn "TVDiff derivative order $d has invalid values, stopping"
                break
            end
            
            # Check for excessive values (likely numerical instability)
            max_val = maximum(abs.(current_deriv))
            if max_val > 1e6
                @warn "TVDiff derivative order $d has very large values ($max_val), stopping"
                break
            end
            
            derivatives[d] = current_deriv
            
        catch e
            @warn "TVDiff failed at derivative order $d: $e"
            break
        end
    end
    
    # Create interpolating splines for the function and its derivatives
    # This allows us to evaluate at arbitrary points
    splines = Dict{Int, Dierckx.Spline1D}()
    
    for (order, vals) in derivatives
        try
            # Ensure we have enough points for the spline degree
            k = min(3, length(t)-1, length(vals)-1)
            if k >= 1  # Need at least linear interpolation
                splines[order] = Spline1D(t, vals, k=k)
            else
                @warn "TVDiff: Not enough points for spline interpolation at order $order"
            end
        catch e
            @warn "TVDiff: Failed to create spline for order $order: $e"
        end
    end
    
    # Ensure we have at least the function (order 0)
    if !haskey(splines, 0)
        @error "TVDiff: Failed to create even the function approximation"
        # Fallback to simple linear interpolation
        try
            splines[0] = Spline1D(t, y, k=1)
        catch e
            error("TVDiff: Complete failure - cannot create any approximation: $e")
        end
    end
    
    # Create callable function with robust evaluation
    function tvdiff_func(x)
        try
            return evaluate(splines[0], x)
        catch e
            @warn "TVDiff evaluation failed at $x: $e"
            return NaN
        end
    end
    
    # Override nth_deriv_at for derivatives with robust handling
    function tvdiff_nth_deriv_at(n::Int, x::Real)
        try
            if n <= max_order && haskey(splines, n)
                return evaluate(splines[n], x)
            else
                # For orders beyond what we computed, return NaN rather than crash
                if n <= config.derivative_orders
                    @debug "TVDiff: Derivative order $n not available (max computed: $max_order)"
                end
                return NaN
            end
        catch e
            @warn "TVDiff derivative evaluation failed at order $n, x=$x: $e"
            return NaN
        end
    end
    
    return ApproximationWrapper(tvdiff_func, tvdiff_nth_deriv_at)
end

"""
    calculate_errors(predictions, true_values)

Calculate error metrics for predictions vs true values.
"""
function calculate_errors(predictions::Dict, true_values::Dict)
    errors = Dict{String, NamedTuple}()
    
    # Get the range of the function values for normalization
    y_range = 1.0
    if haskey(true_values, "y")
        y_vals = true_values["y"]
        y_range = maximum(y_vals) - minimum(y_vals)
        if y_range == 0
            y_range = 1.0  # Avoid division by zero
        end
    end
    
    for (key, true_vals) in true_values
        if haskey(predictions, key)
            pred_vals = predictions[key]
            
            rmse = sqrt(mean((pred_vals .- true_vals).^2))
            mae = mean(abs.(pred_vals .- true_vals))
            max_error = maximum(abs.(pred_vals .- true_vals))
            
            # Calculate normalized errors
            rmse_normalized = rmse / y_range
            mae_normalized = mae / y_range
            max_error_normalized = max_error / y_range
            
            errors[key] = (rmse=rmse, mae=mae, max_error=max_error,
                          rmse_normalized=rmse_normalized, 
                          mae_normalized=mae_normalized,
                          max_error_normalized=max_error_normalized)
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