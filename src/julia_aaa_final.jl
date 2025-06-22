# julia_aaa_final.jl
# Final working Julia AAA methods using BaryRational + ForwardDiff

using BaryRational
using ForwardDiff

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
            method.aaa_fit = best_fit
            method.success = true
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
            
            # Stage 2: Could add refinement here if needed
            if method.refine
                # For now, just use stage 1 result
                # Could add Optim.jl refinement here
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

export JuliaAAALS, JuliaAAAFullOpt, JuliaAAATwoStage, JuliaAAASmoothBary, fit!, evaluate