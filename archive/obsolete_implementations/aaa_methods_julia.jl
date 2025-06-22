# aaa_methods_julia.jl
# Julia equivalents of Python AAA methods using proper AD and existing packages

using BaryRational
using Optim
using ForwardDiff
using LinearAlgebra
using Statistics

"""
Abstract base type for extended AAA methods that mirrors the Python implementations
but uses proper Julia AD instead of finite differences.
"""
abstract type AbstractAAAExt end

"""
    AAALeastSquaresApproximator

Julia equivalent of Python AAALeastSquaresApproximator.
Uses BaryRational.aaa for initial fit, then optimizes support values and weights
using least squares with ForwardDiff for gradients.
"""
mutable struct AAALeastSquaresApproximator <: AbstractAAAExt
    tol::Float64
    max_terms::Int
    opt_settings::Optim.Options
    # Fitted data
    initial_fit::Union{Nothing, BaryRational.AAAapprox}
    support_points::Vector{Float64}  # zj - fixed support points
    support_values::Vector{Float64}  # fj - optimized function values  
    weights::Vector{Float64}         # wj - optimized weights
    fitted::Bool
    success::Bool
end

function AAALeastSquaresApproximator(; 
    tol::Float64=1e-13, 
    max_terms::Int=100, 
    opt_settings::Optim.Options=Optim.Options(iterations=5000, f_tol=1e-12, g_tol=1e-8)
)
    return AAALeastSquaresApproximator(
        tol, max_terms, opt_settings,
        nothing, Float64[], Float64[], Float64[], false, false
    )
end

"""
    fit!(method::AAALeastSquaresApproximator, t::AbstractVector, y::AbstractVector)

Fit the AAA least squares approximator to data (t, y).
"""
function fit!(method::AAALeastSquaresApproximator, t::AbstractVector, y::AbstractVector)
    try
        # Stage 1: Initial AAA fit using BaryRational
        method.initial_fit = BaryRational.aaa(t, y; tol=method.tol, mmax=method.max_terms)
        
        # Extract initial parameters
        method.support_points = copy(method.initial_fit.x)  # Keep support points fixed
        method.support_values = copy(method.initial_fit.f)  # Initial function values
        method.weights = copy(method.initial_fit.w)         # Initial weights
        
        k = length(method.support_points)
        
        # Stage 2: Least squares optimization of support values and weights
        # Parameter vector: [fj..., wj...]
        θ0 = [method.support_values; method.weights]
        
        # Objective function: minimize ||r(θ) - y||²
        function objective(θ)
            fj = view(θ, 1:k)
            wj = view(θ, k+1:2k)
            
            # Evaluate barycentric interpolation at all data points
            residuals = zeros(length(t))
            for (i, ti) in enumerate(t)
                # Manual barycentric evaluation for stability
                d = ti .- method.support_points
                if any(abs.(d) .< 1e-15)  # Near support point
                    idx = findfirst(abs.(d) .< 1e-15)
                    residuals[i] = fj[idx] - y[i]
                else
                    # Standard barycentric formula
                    weights_eval = wj ./ d
                    interp_val = sum(weights_eval .* fj) / sum(weights_eval)
                    residuals[i] = interp_val - y[i]
                end
            end
            
            return 0.5 * sum(abs2, residuals)
        end
        
        # Use Optim with ForwardDiff for gradients
        od = OnceDifferentiable(objective, θ0; autodiff=:forward)
        result = optimize(od, θ0, BFGS(), method.opt_settings)
        
        if Optim.converged(result)
            θ_opt = Optim.minimizer(result)
            method.support_values = copy(view(θ_opt, 1:k))
            method.weights = copy(view(θ_opt, k+1:2k))
            method.success = true
        else
            @warn "Optimization did not converge for AAA_LS"
            method.success = false
        end
        
        method.fitted = true
        
    catch e
        @warn "AAA_LS fitting failed: $e"
        method.fitted = false
        method.success = false
    end
    
    return method
end

"""
    evaluate(method::AAALeastSquaresApproximator, t_eval::AbstractVector, derivative_order::Int=0)

Evaluate the fitted AAA method at points t_eval with specified derivative order.
"""
function evaluate(method::AAALeastSquaresApproximator, t_eval::AbstractVector, derivative_order::Int=0)
    if !method.fitted
        throw(ArgumentError("Method must be fitted before evaluation"))
    end
    
    if !method.success
        return fill(NaN, length(t_eval))
    end
    
    # Create a callable function using optimized parameters
    function aaa_func(x)
        d = x .- method.support_points
        
        # Handle near-support points
        if any(abs.(d) .< 1e-15)
            idx = findfirst(abs.(d) .< 1e-15)
            return method.support_values[idx]
        end
        
        # Standard barycentric evaluation
        weights_eval = method.weights ./ d
        return sum(weights_eval .* method.support_values) / sum(weights_eval)
    end
    
    # Evaluate derivatives using ForwardDiff
    if derivative_order == 0
        return [aaa_func(x) for x in t_eval]
    elseif derivative_order == 1
        return [ForwardDiff.derivative(aaa_func, x) for x in t_eval]
    elseif derivative_order == 2
        return [ForwardDiff.derivative(y -> ForwardDiff.derivative(aaa_func, y), x) for x in t_eval]
    elseif derivative_order == 3
        return [ForwardDiff.derivative(z -> ForwardDiff.derivative(y -> ForwardDiff.derivative(aaa_func, y), z), x) for x in t_eval]
    elseif derivative_order == 4
        f1(x) = ForwardDiff.derivative(aaa_func, x)
        f2(x) = ForwardDiff.derivative(f1, x)
        f3(x) = ForwardDiff.derivative(f2, x)
        return [ForwardDiff.derivative(f3, x) for x in t_eval]
    elseif derivative_order == 5
        f1(x) = ForwardDiff.derivative(aaa_func, x)
        f2(x) = ForwardDiff.derivative(f1, x)
        f3(x) = ForwardDiff.derivative(f2, x)
        f4(x) = ForwardDiff.derivative(f3, x)
        return [ForwardDiff.derivative(f4, x) for x in t_eval]
    else
        throw(ArgumentError("Derivative order $derivative_order not supported"))
    end
end

"""
    AAAFullOptApproximator

Julia equivalent of Python AAA_FullOpt_Approximator.
Optimizes support points, values, and weights simultaneously.
"""
mutable struct AAAFullOptApproximator <: AbstractAAAExt
    tol::Float64
    max_terms::Int
    opt_settings::Optim.Options
    # Fitted data
    support_points::Vector{Float64}  # zj - optimized support points
    support_values::Vector{Float64}  # fj - optimized function values
    weights::Vector{Float64}         # wj - optimized weights
    fitted::Bool
    success::Bool
end

function AAAFullOptApproximator(;
    tol::Float64=1e-13,
    max_terms::Int=100,
    opt_settings::Optim.Options=Optim.Options(iterations=1000, f_tol=1e-10, g_tol=1e-6)
)
    return AAAFullOptApproximator(
        tol, max_terms, opt_settings,
        Float64[], Float64[], Float64[], false, false
    )
end

function fit!(method::AAAFullOptApproximator, t::AbstractVector, y::AbstractVector)
    try
        # Initial AAA fit
        initial_fit = BaryRational.aaa(t, y; tol=method.tol, mmax=method.max_terms)
        
        k = length(initial_fit.x)
        
        # Parameter vector: [zj..., fj..., wj...]
        θ0 = [initial_fit.x; initial_fit.f; initial_fit.w]
        
        # Domain bounds for support points (stay within data range + small margin)
        t_min, t_max = extrema(t)
        margin = 0.1 * (t_max - t_min)
        
        # Set up bounds: support points bounded, values and weights unbounded
        lower = [fill(t_min - margin, k); fill(-Inf, 2k)]
        upper = [fill(t_max + margin, k); fill(Inf, 2k)]
        
        function objective(θ)
            zj = view(θ, 1:k)
            fj = view(θ, k+1:2k)
            wj = view(θ, 2k+1:3k)
            
            residuals = zeros(length(t))
            for (i, ti) in enumerate(t)
                d = ti .- zj
                
                # Handle near-support points with small regularization
                min_dist = minimum(abs.(d))
                if min_dist < 1e-12
                    closest_idx = argmin(abs.(d))
                    residuals[i] = fj[closest_idx] - y[i]
                else
                    weights_eval = wj ./ d
                    if any(isnan.(weights_eval)) || any(isinf.(weights_eval))
                        return Inf  # Penalty for bad configurations
                    end
                    interp_val = sum(weights_eval .* fj) / sum(weights_eval)
                    residuals[i] = interp_val - y[i]
                end
            end
            
            return 0.5 * sum(abs2, residuals)
        end
        
        # Use L-BFGS-B for bound constraints
        result = optimize(objective, lower, upper, θ0, Fminbox(BFGS()), method.opt_settings)
        
        if Optim.converged(result)
            θ_opt = Optim.minimizer(result)
            method.support_points = copy(view(θ_opt, 1:k))
            method.support_values = copy(view(θ_opt, k+1:2k))
            method.weights = copy(view(θ_opt, 2k+1:3k))
            method.success = true
        else
            @warn "Full optimization did not converge"
            method.success = false
        end
        
        method.fitted = true
        
    catch e
        @warn "AAA_FullOpt fitting failed: $e"
        method.fitted = false
        method.success = false
    end
    
    return method
end

function evaluate(method::AAAFullOptApproximator, t_eval::AbstractVector, derivative_order::Int=0)
    if !method.fitted
        throw(ArgumentError("Method must be fitted before evaluation"))
    end
    
    if !method.success
        return fill(NaN, length(t_eval))
    end
    
    function aaa_func(x)
        d = x .- method.support_points
        
        if any(abs.(d) .< 1e-15)
            idx = findfirst(abs.(d) .< 1e-15)
            return method.support_values[idx]
        end
        
        weights_eval = method.weights ./ d
        return sum(weights_eval .* method.support_values) / sum(weights_eval)
    end
    
    # Same derivative evaluation as AAALeastSquaresApproximator
    if derivative_order == 0
        return [aaa_func(x) for x in t_eval]
    elseif derivative_order == 1
        return [ForwardDiff.derivative(aaa_func, x) for x in t_eval]
    elseif derivative_order == 2
        return [ForwardDiff.derivative(y -> ForwardDiff.derivative(aaa_func, y), x) for x in t_eval]
    elseif derivative_order == 3
        return [ForwardDiff.derivative(z -> ForwardDiff.derivative(y -> ForwardDiff.derivative(aaa_func, y), z), x) for x in t_eval]
    elseif derivative_order == 4
        f1(x) = ForwardDiff.derivative(aaa_func, x)
        f2(x) = ForwardDiff.derivative(f1, x)
        f3(x) = ForwardDiff.derivative(f2, x)
        return [ForwardDiff.derivative(f3, x) for x in t_eval]
    elseif derivative_order == 5
        f1(x) = ForwardDiff.derivative(aaa_func, x)
        f2(x) = ForwardDiff.derivative(f1, x)
        f3(x) = ForwardDiff.derivative(f2, x)
        f4(x) = ForwardDiff.derivative(f3, x)
        return [ForwardDiff.derivative(f4, x) for x in t_eval]
    else
        throw(ArgumentError("Derivative order $derivative_order not supported"))
    end
end

# Export the new types and functions
export AAALeastSquaresApproximator, AAAFullOptApproximator, fit!, evaluate