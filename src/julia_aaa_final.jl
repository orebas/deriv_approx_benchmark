# julia_aaa_final.jl
# Final working Julia AAA methods using BaryRational + ForwardDiff

using BaryRational
using ForwardDiff
using LinearAlgebra
using Optim
using LineSearches

"""
Julia AAA Least Squares method using BaryRational + BIC selection.
This replaces the problematic Python AAA methods.
"""
mutable struct JuliaAAALS
    tol::Float64
    max_terms::Int
    aaa_fit::Union{Nothing, BaryRational.AAAapprox}
    fitted::Bool
    success::Bool
    
    function JuliaAAALS(tol=1e-12, max_terms=50)
        new(tol, max_terms, nothing, false, false)
    end
end

function fit!(method::JuliaAAALS, t::AbstractVector, y::AbstractVector)
    try
        # Use BaryRational's AAA with BIC model selection
        n_data = length(t)
        max_possible = min(method.max_terms, n_data ÷ 3)
        
        best_bic = Inf
        best_fit = nothing
        
        for max_terms in 3:max_possible
            try
                candidate = BaryRational.aaa(t, y; tol=method.tol, mmax=max_terms)
                
                # Calculate BIC
                residuals = [candidate(ti) - yi for (ti, yi) in zip(t, y)]
                ssr = sum(abs2, residuals)
                k = 2 * length(candidate.x)  # Parameters
                bic = k * log(n_data) + n_data * log(ssr / n_data + 1e-15)
                
                if bic < best_bic
                    best_bic = bic
                    best_fit = candidate
                end
            catch
                continue
            end
        end
        
        if best_fit !== nothing
            # Apply least-squares refinement with frozen support points
            try
                refined_fit = refine_least_squares!(best_fit, t, y)
                method.aaa_fit = refined_fit
                method.success = true
            catch e
                @warn "Refinement failed, using unrefined fit: $e"
                method.aaa_fit = best_fit
                method.success = true
            end
        else
            method.success = false
        end
        
        method.fitted = true
        
    catch e
        @warn "JuliaAAALS fitting failed: $e"
        method.fitted = true
        method.success = false
    end
    
    return method
end

function evaluate(method::JuliaAAALS, t_eval::AbstractVector, deriv_order::Int=0)
    if !method.fitted || !method.success
        return fill(NaN, length(t_eval))
    end
    
    f = method.aaa_fit
    
    if deriv_order == 0
        return [f(x) for x in t_eval]
    elseif deriv_order == 1
        return [ForwardDiff.derivative(f, x) for x in t_eval]
    elseif deriv_order == 2
        return [ForwardDiff.derivative(x -> ForwardDiff.derivative(f, x), x) for x in t_eval]
    elseif deriv_order == 3
        return [ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> ForwardDiff.derivative(f, y), x), x) for x in t_eval]
    elseif deriv_order == 4
        return [ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> ForwardDiff.derivative(z -> ForwardDiff.derivative(f, z), y), x), x) for x in t_eval]
    elseif deriv_order == 5
        return [ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> ForwardDiff.derivative(z -> ForwardDiff.derivative(w -> ForwardDiff.derivative(f, w), z), y), x), x) for x in t_eval]
    else
        error("Derivative order $deriv_order not supported")
    end
end

"""
Julia AAA Full Optimization (currently uses basic AAA).
"""
mutable struct JuliaAAAFullOpt
    tol::Float64
    max_terms::Int
    aaa_fit::Union{Nothing, BaryRational.AAAapprox}
    fitted::Bool
    success::Bool
    
    function JuliaAAAFullOpt(tol=1e-12, max_terms=25)
        new(tol, max_terms, nothing, false, false)
    end
end

function fit!(method::JuliaAAAFullOpt, t::AbstractVector, y::AbstractVector)
    try
        # For now, use standard AAA (can add full optimization later)
        method.aaa_fit = BaryRational.aaa(t, y; tol=method.tol, mmax=method.max_terms)
        method.fitted = true
        method.success = true
    catch e
        @warn "JuliaAAAFullOpt fitting failed: $e"
        method.fitted = true
        method.success = false
    end
    
    return method
end

function evaluate(method::JuliaAAAFullOpt, t_eval::AbstractVector, deriv_order::Int=0)
    if !method.fitted || !method.success
        return fill(NaN, length(t_eval))
    end
    
    f = method.aaa_fit
    
    if deriv_order == 0
        return [f(x) for x in t_eval]
    elseif deriv_order == 1
        return [ForwardDiff.derivative(f, x) for x in t_eval]
    elseif deriv_order == 2
        df(x) = ForwardDiff.derivative(f, x)
        return [ForwardDiff.derivative(df, x) for x in t_eval]
    elseif deriv_order == 3
        return [ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> ForwardDiff.derivative(f, y), x), x) for x in t_eval]
    elseif deriv_order == 4
        return [ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> ForwardDiff.derivative(z -> ForwardDiff.derivative(f, z), y), x), x) for x in t_eval]
    elseif deriv_order == 5
        return [ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> ForwardDiff.derivative(z -> ForwardDiff.derivative(w -> ForwardDiff.derivative(f, w), z), y), x), x) for x in t_eval]
    else
        error("Derivative order $deriv_order not supported")
    end
end

"""
Two-stage AAA: Use JuliaAAALS then optionally refine.
"""
mutable struct JuliaAAATwoStage
    tol::Float64
    max_terms::Int
    refine::Bool
    aaa_fit::Union{Nothing, BaryRational.AAAapprox}
    fitted::Bool
    success::Bool
    
    function JuliaAAATwoStage(tol=1e-12, max_terms=50, refine=true)
        new(tol, max_terms, refine, nothing, false, false)
    end
end

function fit!(method::JuliaAAATwoStage, t::AbstractVector, y::AbstractVector)
    try
        # Stage 1: Basic AAA
        stage1 = JuliaAAALS(method.tol, method.max_terms)
        fit!(stage1, t, y)
        
        if stage1.success
            method.aaa_fit = stage1.aaa_fit
            
            # Stage 2: Apply least-squares refinement if requested
            if method.refine
                try
                    refined_fit = refine_least_squares!(stage1.aaa_fit, t, y)
                    method.aaa_fit = refined_fit
                catch e
                    @warn "Refinement failed in JuliaAAATwoStage: $e"
                    # Keep original fit
                end
            end
            
            method.success = true
        else
            method.success = false
        end
        
        method.fitted = true
        
    catch e
        @warn "JuliaAAATwoStage fitting failed: $e"
        method.fitted = true
        method.success = false
    end
    
    return method
end

function evaluate(method::JuliaAAATwoStage, t_eval::AbstractVector, deriv_order::Int=0)
    if !method.fitted || !method.success
        return fill(NaN, length(t_eval))
    end
    
    f = method.aaa_fit
    
    if deriv_order == 0
        return [f(x) for x in t_eval]
    elseif deriv_order == 1
        return [ForwardDiff.derivative(f, x) for x in t_eval]
    elseif deriv_order == 2
        df(x) = ForwardDiff.derivative(f, x)
        return [ForwardDiff.derivative(df, x) for x in t_eval]
    elseif deriv_order == 3
        return [ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> ForwardDiff.derivative(f, y), x), x) for x in t_eval]
    elseif deriv_order == 4
        return [ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> ForwardDiff.derivative(z -> ForwardDiff.derivative(f, z), y), x), x) for x in t_eval]
    elseif deriv_order == 5
        return [ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> ForwardDiff.derivative(z -> ForwardDiff.derivative(w -> ForwardDiff.derivative(f, w), z), y), x), x) for x in t_eval]
    else
        error("Derivative order $deriv_order not supported")
    end
end

"""
Smooth Barycentric AAA using native Julia implementation.
"""
mutable struct JuliaAAASmoothBary
    tol::Float64
    max_terms::Int
    aaa_fit::Union{Nothing, BaryRational.AAAapprox}
    fitted::Bool
    success::Bool
    
    function JuliaAAASmoothBary(tol=1e-12, max_terms=50)
        new(tol, max_terms, nothing, false, false)
    end
end

function fit!(method::JuliaAAASmoothBary, t::AbstractVector, y::AbstractVector)
    try
        # Use BaryRational's native implementation which already handles singularities properly
        method.aaa_fit = BaryRational.aaa(t, y; tol=method.tol, mmax=method.max_terms)
        method.fitted = true
        method.success = true
    catch e
        @warn "JuliaAAASmoothBary fitting failed: $e"
        method.fitted = true
        method.success = false
    end
    
    return method
end

function evaluate(method::JuliaAAASmoothBary, t_eval::AbstractVector, deriv_order::Int=0)
    if !method.fitted || !method.success
        return fill(NaN, length(t_eval))
    end
    
    f = method.aaa_fit
    
    if deriv_order == 0
        return [f(x) for x in t_eval]
    elseif deriv_order == 1
        return [ForwardDiff.derivative(f, x) for x in t_eval]
    elseif deriv_order == 2
        df(x) = ForwardDiff.derivative(f, x)
        return [ForwardDiff.derivative(df, x) for x in t_eval]
    elseif deriv_order == 3
        return [ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> ForwardDiff.derivative(f, y), x), x) for x in t_eval]
    elseif deriv_order == 4
        return [ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> ForwardDiff.derivative(z -> ForwardDiff.derivative(f, z), y), x), x) for x in t_eval]
    elseif deriv_order == 5
        return [ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> ForwardDiff.derivative(z -> ForwardDiff.derivative(w -> ForwardDiff.derivative(f, w), z), y), x), x) for x in t_eval]
    else
        error("Derivative order $deriv_order not supported")
    end
end

"""
Refine an AAA approximation using least-squares optimization with frozen support points.
This implements the missing refinement step discussed in the ChatGPT conversation.
"""
function refine_least_squares!(approx::BaryRational.AAAapprox, t::AbstractVector, y::AbstractVector)
    # Extract support points (these stay fixed)
    support_points = approx.x
    n_support = length(support_points)
    
    # Create initial parameter vector from current approximation
    # Parameters are: [f_values at support points; weights]
    f_values = approx.f
    weights = approx.w
    p0 = vcat(f_values, weights)
    
    # Define optimized barycentric evaluation
    function eval_barycentric(ti, f_vals, w_vals)
        # Check if ti is at a support point
        for j in 1:n_support
            if abs(ti - support_points[j]) < 1e-14
                return f_vals[j]
            end
        end
        
        # Standard barycentric formula
        numer = 0.0
        denom = 0.0
        for j in 1:n_support
            term = w_vals[j] / (ti - support_points[j])
            numer += term * f_vals[j]
            denom += term
        end
        
        return numer / denom
    end
    
    # Define residual function for optimization
    function residual_fn(p)
        # Extract f_values and weights from parameter vector
        f_vals = p[1:n_support]
        w_vals = p[n_support+1:end]
        
        # Compute residuals at all data points
        residuals = zeros(length(t))
        for (i, ti) in enumerate(t)
            residuals[i] = eval_barycentric(ti, f_vals, w_vals) - y[i]
        end
        
        return residuals
    end
    
    # Solve least-squares problem with more robust settings
    result = optimize(
        p -> sum(abs2, residual_fn(p)), 
        p0, 
        LBFGS(
            m=10,  # Number of corrections to approximate Hessian
            linesearch=LineSearches.BackTracking()
        ),
        Optim.Options(
            iterations = 1000,
            show_trace = false
        )
    )
    
    # Extract optimized parameters
    p_opt = Optim.minimizer(result)
    f_opt = p_opt[1:n_support]
    w_opt = p_opt[n_support+1:end]
    
    # Create refined AAA approximation
    refined_approx = BaryRational.AAAapprox(support_points, f_opt, w_opt, approx.errvec)
    
    return refined_approx
end

export JuliaAAALS, JuliaAAAFullOpt, JuliaAAATwoStage, JuliaAAASmoothBary, fit!, evaluate, refine_least_squares!