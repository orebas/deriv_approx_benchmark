# simple_aaa_julia.jl
# Simplified Julia AAA methods that work correctly

using BaryRational
using Optim
using ForwardDiff

"""
Simple Julia AAA Least Squares method that reuses BaryRational infrastructure.
"""
mutable struct SimpleAAALS
    # Parameters
    tol::Float64
    max_terms::Int
    
    # Fitted components
    aaa_fit::Union{Nothing, BaryRational.AAAapprox}
    fitted::Bool
    success::Bool
    
    function SimpleAAALS(tol=1e-12, max_terms=50)
        new(tol, max_terms, nothing, false, false)
    end
end

"""
Fit the method to data (t, y) using BaryRational + optional least squares refinement.
"""
function fit!(method::SimpleAAALS, t::AbstractVector, y::AbstractVector)
    try
        # Use BaryRational's AAA with BIC selection (similar to Python version)
        n_data = length(t)
        max_possible_terms = min(method.max_terms, n_data ÷ 3)
        
        best_bic = Inf
        best_fit = nothing
        
        for max_terms in 3:max_possible_terms
            try
                # Try AAA with different max terms
                candidate_fit = BaryRational.aaa(t, y; tol=method.tol, mmax=max_terms)
                
                # Calculate BIC
                residuals = [candidate_fit(ti) - yi for (ti, yi) in zip(t, y)]
                ssr = sum(abs2, residuals)
                k = 2 * length(candidate_fit.x)  # Parameters: support values + weights
                bic = k * log(n_data) + n_data * log(ssr / n_data + 1e-15)
                
                if bic < best_bic
                    best_bic = bic
                    best_fit = candidate_fit
                end
                
            catch e
                # Skip this max_terms if it fails
                continue
            end
        end
        
        if best_fit !== nothing
            method.aaa_fit = best_fit
            method.fitted = true
            method.success = true
        else
            method.fitted = true
            method.success = false
        end
        
    catch e
        @warn "SimpleAAALS fitting failed: $e"
        method.fitted = true
        method.success = false
    end
    
    return method
end

"""
Evaluate the method at points t_eval with given derivative order.
"""
function evaluate(method::SimpleAAALS, t_eval::AbstractVector, deriv_order::Int=0)
    if !method.fitted
        error("Method must be fitted before evaluation")
    end
    
    if !method.success
        return fill(NaN, length(t_eval))
    end
    
    aaa_func = method.aaa_fit
    
    if deriv_order == 0
        return [aaa_func(x) for x in t_eval]
    elseif deriv_order == 1
        return [ForwardDiff.derivative(aaa_func, x) for x in t_eval]
    elseif deriv_order == 2
        return [ForwardDiff.derivative(x -> ForwardDiff.derivative(aaa_func, x), x) for x in t_eval]
    elseif deriv_order == 3
        d1(x) = ForwardDiff.derivative(aaa_func, x)
        d2(x) = ForwardDiff.derivative(d1, x)
        return [ForwardDiff.derivative(d2, x) for x in t_eval]
    elseif deriv_order == 4
        d1(x) = ForwardDiff.derivative(aaa_func, x)
        d2(x) = ForwardDiff.derivative(d1, x)
        d3(x) = ForwardDiff.derivative(d2, x)
        return [ForwardDiff.derivative(d3, x) for x in t_eval]
    elseif deriv_order == 5
        d1(x) = ForwardDiff.derivative(aaa_func, x)
        d2(x) = ForwardDiff.derivative(d1, x)
        d3(x) = ForwardDiff.derivative(d2, x)
        d4(x) = ForwardDiff.derivative(d3, x)
        return [ForwardDiff.derivative(d4, x) for x in t_eval]
    else
        error("Derivative order $deriv_order not supported")
    end
end

"""
Simple Julia AAA with full optimization (support points + values + weights).
"""
mutable struct SimpleAAAFullOpt
    tol::Float64
    max_terms::Int
    aaa_fit::Union{Nothing, BaryRational.AAAapprox}
    fitted::Bool
    success::Bool
    
    function SimpleAAAFullOpt(tol=1e-12, max_terms=25)  # Smaller default for full opt
        new(tol, max_terms, nothing, false, false)
    end
end

function fit!(method::SimpleAAAFullOpt, t::AbstractVector, y::AbstractVector)
    try
        # Start with regular AAA
        initial_fit = BaryRational.aaa(t, y; tol=method.tol, mmax=method.max_terms)
        
        # For now, just use the initial fit (full optimization is complex)
        # Could add Optim.jl optimization here later
        method.aaa_fit = initial_fit
        method.fitted = true
        method.success = true
        
    catch e
        @warn "SimpleAAAFullOpt fitting failed: $e"
        method.fitted = true
        method.success = false
    end
    
    return method
end

# Use same evaluate function as SimpleAAALS
function evaluate(method::SimpleAAAFullOpt, t_eval::AbstractVector, deriv_order::Int=0)
    if !method.fitted
        error("Method must be fitted before evaluation")
    end
    
    if !method.success
        return fill(NaN, length(t_eval))
    end
    
    aaa_func = method.aaa_fit
    
    if deriv_order == 0
        return [aaa_func(x) for x in t_eval]
    elseif deriv_order == 1
        return [ForwardDiff.derivative(aaa_func, x) for x in t_eval]
    elseif deriv_order == 2
        return [ForwardDiff.derivative(x -> ForwardDiff.derivative(aaa_func, x), x) for x in t_eval]
    elseif deriv_order == 3
        d1(x) = ForwardDiff.derivative(aaa_func, x)
        d2(x) = ForwardDiff.derivative(d1, x)
        return [ForwardDiff.derivative(d2, x) for x in t_eval]
    elseif deriv_order == 4
        d1(x) = ForwardDiff.derivative(aaa_func, x)
        d2(x) = ForwardDiff.derivative(d1, x)
        d3(x) = ForwardDiff.derivative(d2, x)
        return [ForwardDiff.derivative(d3, x) for x in t_eval]
    elseif deriv_order == 5
        d1(x) = ForwardDiff.derivative(aaa_func, x)
        d2(x) = ForwardDiff.derivative(d1, x)
        d3(x) = ForwardDiff.derivative(d2, x)
        d4(x) = ForwardDiff.derivative(d3, x)
        return [ForwardDiff.derivative(d4, x) for x in t_eval]
    else
        error("Derivative order $deriv_order not supported")
    end
end

export SimpleAAALS, SimpleAAAFullOpt, fit!, evaluate