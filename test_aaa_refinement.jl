#!/usr/bin/env julia

# Test script for AAA refinement implementation
using Pkg
Pkg.activate(".")

include("src/julia_aaa_final.jl")
using .Main: JuliaAAALS, fit!, evaluate
using Printf

# Test function: Runge function
f(x) = 1 / (1 + 25 * x^2)

# Generate test data
n = 100
t = range(-1, 1, length=n)
y = f.(t)

# Add some noise to make refinement more meaningful
y_noisy = y .+ 0.001 * randn(n)

println("Testing AAA with least-squares refinement...")
println("=" ^ 60)

# Create and fit the AAA approximation
aaa = JuliaAAALS(1e-12, 30)
fit!(aaa, t, y_noisy)

if aaa.success
    println("AAA fitting successful!")
    println("Number of support points: ", length(aaa.aaa_fit.x))
    
    # Evaluate at test points
    t_test = range(-1, 1, length=200)
    y_pred = evaluate(aaa, t_test, 0)
    y_true = f.(t_test)
    
    # Calculate errors
    errors = abs.(y_pred .- y_true)
    max_error = maximum(errors)
    avg_error = sum(errors) / length(errors)
    
    println(@sprintf("Maximum error: %.2e", max_error))
    println(@sprintf("Average error: %.2e", avg_error))
    
    # Test derivatives
    println("\nTesting derivatives...")
    y_deriv1 = evaluate(aaa, t_test, 1)
    println("First derivative computed successfully")
    
    y_deriv2 = evaluate(aaa, t_test, 2)
    println("Second derivative computed successfully")
else
    println("AAA fitting failed!")
end

println("\n" * "=" ^ 60)
println("Test completed!")