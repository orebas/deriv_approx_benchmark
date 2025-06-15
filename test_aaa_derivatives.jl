#!/usr/bin/env julia

"""
Test script to investigate potential bugs in AAA approximation derivatives.
Tests two functions: e^(2x) and x^6
Compares estimated derivatives vs true values up to 5th order.
"""

using ODEParameterEstimation
using BaryRational
using ForwardDiff
using Printf
using LinearAlgebra
using Statistics
using Combinatorics

# Import nth_deriv_at from ODEParameterEstimation
import ODEParameterEstimation: nth_deriv_at

"""
	analytical_aaa_deriv(r, x::Real, n::Int)

Analytically compute the nth derivative of a Barycentric rational function `r` at point `x`.
"""
function analytical_aaa_deriv(r, x::Real, n::Int)
	# Numerator N(x) = sum(w_j * f_j / (x - z_j))
	# Denominator D(x) = sum(w_j / (x - z_j))

	# Derivatives of N(x) and D(x) can be computed exactly.
	# d^k/dx^k (1 / (x - z)) = (-1)^k * k! / (x - z)^(k+1)

	N_derivs = zeros(n + 1)
	D_derivs = zeros(n + 1)

	for k in 0:n
		N_deriv = 0.0
		D_deriv = 0.0
		for j in 1:length(r.x)
			term = (-1)^k * factorial(k) / (x - r.x[j])^(k + 1)
			N_deriv += r.w[j] * r.f[j] * term
			D_deriv += r.w[j] * term
		end
		N_derivs[k+1] = N_deriv
		D_derivs[k+1] = D_deriv
	end

	# Use General Leibniz Rule (Faà di Bruno's formula is for composition, Leibniz for product)
	# r = N * (1/D) -> r' = N'*(1/D) + N*(-D'/D^2) = (N'D - ND')/D^2
	# We need the nth derivative of r = N/D.
	# From d(N/D)/dx = (N'D - ND')/D^2, we can see this gets complicated.
	# Let's use the formula for the nth derivative of a quotient, which is:
	# d^n/dx^n (N/D) = sum_{k=0 to n} [C(n,k) * d^(n-k)N/dx^(n-k) * d^k(1/D)/dx^k]
	# where the derivative of 1/D is found using Faà di Bruno's formula.

	# This is complex. Let's compute recursively.
	# r' = (N'D - ND') / D^2
	# r'' = ...
	# Let's define C(n,k) = binomial(n,k)
	# (N/D)^(n) = (1/D) * (N^(n) - sum_{k=1 to n} C(n,k) * D^(k) * (N/D)^(n-k))

	r_derivs = zeros(n + 1)
	r_derivs[1] = N_derivs[1] / D_derivs[1] # 0th derivative (the function value)

	for i in 1:n # Compute i-th derivative
		inner_sum = 0.0
		for k in 1:i
			inner_sum += binomial(i, k) * D_derivs[k+1] * r_derivs[i-k+1]
		end
		r_derivs[i+1] = (N_derivs[i+1] - inner_sum) / D_derivs[1]
	end

	return r_derivs[n+1]
end

"""
	barycentric_eval(x, r)

AD-friendly evaluation of a barycentric rational object `r` at point `x`.
"""
function barycentric_eval(x, r)
	# Handle case where x is a support point
	for j in 1:length(r.x)
		if x == r.x[j]
			return r.f[j]
		end
	end

	num = sum(r.w[j] * r.f[j] / (x - r.x[j]) for j in 1:length(r.x))
	den = sum(r.w[j] / (x - r.x[j]) for j in 1:length(r.x))

	return num / den
end

# Define test functions and their analytical derivatives
function test_exp2x()
	f = x -> exp(2*x)
	# Analytical derivatives: d^n/dx^n e^(2x) = 2^n * e^(2x)
	derivs = [x -> 2^n * exp(2*x) for n in 0:5]
	return f, derivs, "e^(2x)"
end

function test_x6()
	f = x -> x^6
	# Analytical derivatives
	derivs = [
		x -> x^6,                    # 0th derivative
		x -> 6*x^5,                  # 1st derivative
		x -> 30*x^4,                 # 2nd derivative
		x -> 120*x^3,                # 3rd derivative
		x -> 360*x^2,                # 4th derivative
		x -> 720*x,                   # 5th derivative
	]
	return f, derivs, "x^6"
end

function test_aaa_derivatives(test_func, analytical_derivs, func_name)
	println("\n" * "="^60)
	println("Testing AAA approximation for $func_name")
	println("="^60)

	# Generate data points
	n_points = 200
	x_min, x_max = -1.0, 1.0
	x_data = collect(range(x_min, x_max, length = n_points))
	y_data = test_func.(x_data)

	# Create AAA approximations with different methods
	println("\n1. Using aaad (high precision):")
	aaa_func_old = aaad(x_data, y_data) # Keep for comparison

	println("\n2. Using BaryRational.aaa directly:")
	aaa_raw = BaryRational.aaa(x_data, y_data, verbose = false, tol = 1e-14)

	# Create a new closure for our AD-friendly evaluation
	aaa_func_new = x -> barycentric_eval(x, aaa_raw)

	# Test points: include support points and intermediate points
	test_points = Float64[]

	# Add some original data points (potential support points)
	append!(test_points, x_data[1:10:end])

	# Add intermediate points
	for i in 1:10:(length(x_data)-1)
		push!(test_points, (x_data[i] + x_data[i+1]) / 2)
	end

	# Sort and unique
	test_points = sort(unique(test_points))

	println("\n3. Testing derivatives at $(length(test_points)) points:")

	# Compute errors for each derivative order
	for deriv_order in 0:5
		println("\n  Derivative order $deriv_order:")

		errors_old = Float64[]
		errors_new = Float64[]

		for x in test_points
			# True value
			true_val = analytical_derivs[deriv_order+1](x)

			# Old AD method (on aaad object)
			try
				approx_old = nth_deriv_at(aaa_func_old, deriv_order, x)
				push!(errors_old, abs(approx_old - true_val))
			catch e
				println("    ⚠️  Error (old method) at x=$x: $e")
			end

			# New AD method (on our pure Julia function)
			try
				approx_new = nth_deriv_at(aaa_func_new, deriv_order, x)
				push!(errors_new, abs(approx_new - true_val))
			catch e
				println("    ⚠️  Error (new method) at x=$x: $e")
			end
		end

		# Summary statistics
		if !isempty(errors_old)
			println("    Old AD errors - Mean: $(@sprintf("%.2e", mean(errors_old))), Max: $(@sprintf("%.2e", maximum(errors_old)))")
		end

		if !isempty(errors_new)
			println("    New AD errors - Mean: $(@sprintf("%.2e", mean(errors_new))), Max: $(@sprintf("%.2e", maximum(errors_new)))")
		end
	end

	# Special check: evaluate at actual support points
	println("\n4. Checking derivatives at a support point:")
	x_support = aaa_raw.x[1]
	println("   Testing at support point x = $(@sprintf("%.6f", x_support))")
	for deriv_order in 0:5
		true_val = analytical_derivs[deriv_order+1](x_support)
		approx_val_new = nth_deriv_at(aaa_func_new, deriv_order, x_support)
		error = abs(approx_val_new - true_val)
		println("     d^$deriv_order: New AD error = $(@sprintf("%.2e", error))")
	end
end

# Run tests
println("AAA Derivative Testing Script")
println("Testing potential bugs in derivative calculations")

# Test 1: e^(2x)
f1, d1, name1 = test_exp2x()
test_aaa_derivatives(f1, d1, name1)

# Test 2: x^6
f2, d2, name2 = test_x6()
test_aaa_derivatives(f2, d2, name2)

println("\n" * "="^60)
println("Testing complete!")
println("="^60)