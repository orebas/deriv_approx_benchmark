#!/usr/bin/env julia

# Analysis script to understand why GPR fails on Van der Pol oscillator
using DifferentialEquations
using Plots
using ForwardDiff
using Statistics
using Printf

# Van der Pol oscillator system
function vanderpol!(du, u, p, t)
    μ = p[1]
    du[1] = u[2]  # x1' = x2
    du[2] = μ * (1 - u[1]^2) * u[2] - u[1]  # x2' = μ(1-x1²)x2 - x1
end

# Solve Van der Pol oscillator
μ = 1.0  # Standard Van der Pol parameter
u0 = [2.0, 0.0]  # Initial conditions
tspan = (0.0, 20.0)
prob = ODEProblem(vanderpol!, u0, [μ], tspan)
sol = solve(prob, Tsit5(), saveat=0.1)

# Extract time series
t = sol.t
x1 = [u[1] for u in sol.u]  # Position
x2 = [u[2] for u in sol.u]  # Velocity

println("Van der Pol Oscillator Analysis")
println("=" ^ 50)
println("Time span: $(tspan[1]) to $(tspan[2])")
println("Number of points: $(length(t))")
println("Parameter μ = $μ")

# Calculate derivatives numerically for comparison
function numerical_derivatives(t, y, max_order=3)
    derivatives = Dict{Int, Vector{Float64}}()
    derivatives[0] = y
    
    for order in 1:max_order
        if order == 1
            # First derivative using central differences
            dy = similar(y)
            for i in 2:length(y)-1
                dy[i] = (y[i+1] - y[i-1]) / (t[i+1] - t[i-1])
            end
            dy[1] = (y[2] - y[1]) / (t[2] - t[1])  # Forward difference at start
            dy[end] = (y[end] - y[end-1]) / (t[end] - t[end-1])  # Backward at end
            derivatives[order] = dy
        else
            # Higher derivatives from previous derivative
            prev_deriv = derivatives[order-1]
            dy = similar(prev_deriv)
            for i in 2:length(prev_deriv)-1
                dy[i] = (prev_deriv[i+1] - prev_deriv[i-1]) / (t[i+1] - t[i-1])
            end
            dy[1] = (prev_deriv[2] - prev_deriv[1]) / (t[2] - t[1])
            dy[end] = (prev_deriv[end] - prev_deriv[end-1]) / (t[end] - t[end-1])
            derivatives[order] = dy
        end
    end
    
    return derivatives
end

# Calculate derivatives for x2(t) - the problematic variable
x2_derivatives = numerical_derivatives(t, x2, 3)

# Analyze the derivatives
println("\nDerivative Analysis for x2(t):")
for order in 0:3
    vals = x2_derivatives[order]
    println(@sprintf("Order %d: min=%.2e, max=%.2e, std=%.2e, range=%.2e", 
            order, minimum(vals), maximum(vals), std(vals), maximum(vals) - minimum(vals)))
end

# Create plots
gr()  # Use GR backend for better performance

# Plot 1: Phase portrait showing nonlinear dynamics
p1 = plot(x1, x2, 
    title="Van der Pol Phase Portrait", 
    xlabel="x1 (position)", 
    ylabel="x2 (velocity)",
    linewidth=2,
    legend=false)
plot!(p1, x1[1:1], x2[1:1], marker=:circle, markersize=8, color=:red, label="Start")
plot!(p1, x1[end:end], x2[end:end], marker=:square, markersize=8, color=:blue, label="End")

# Plot 2: Time series of x2(t) and its derivatives
p2 = plot(t, x2, 
    title="x2(t) - Velocity Component", 
    xlabel="Time", 
    ylabel="x2(t)",
    linewidth=2,
    legend=false)

p3 = plot(t, x2_derivatives[1], 
    title="1st Derivative of x2(t)", 
    xlabel="Time", 
    ylabel="dx2/dt",
    linewidth=2,
    legend=false,
    color=:orange)

p4 = plot(t, x2_derivatives[2], 
    title="2nd Derivative of x2(t)", 
    xlabel="Time", 
    ylabel="d²x2/dt²",
    linewidth=2,
    legend=false,
    color=:green)

p5 = plot(t, x2_derivatives[3], 
    title="3rd Derivative of x2(t)", 
    xlabel="Time", 
    ylabel="d³x2/dt³",
    linewidth=2,
    legend=false,
    color=:red)

# Combine plots
main_plot = plot(p1, p2, p3, p4, p5, 
    layout=(2,3), 
    size=(1200, 800),
    plot_title="Van der Pol Oscillator: Why GPR Struggles")

# Save the plot
savefig(main_plot, "vanderpol_analysis.png")
println("\nPlot saved as 'vanderpol_analysis.png'")

# Analyze smoothness characteristics
println("\nSmoothness Analysis:")
println("=" ^ 30)

# Look at derivative magnification - how much each derivative amplifies
for order in 1:3
    prev_std = std(x2_derivatives[order-1])
    curr_std = std(x2_derivatives[order])
    amplification = curr_std / prev_std
    println(@sprintf("Order %d amplification factor: %.2f", order, amplification))
end

# Look for rapid changes (potential discontinuities)
println("\nRapid Change Analysis:")
for order in 0:3
    vals = x2_derivatives[order]
    # Find largest jumps between consecutive points
    jumps = abs.(diff(vals))
    max_jump = maximum(jumps)
    mean_jump = mean(jumps)
    jump_ratio = max_jump / mean_jump
    println(@sprintf("Order %d: max_jump=%.2e, mean_jump=%.2e, ratio=%.1f", 
            order, max_jump, mean_jump, jump_ratio))
end

# Identify problematic regions
println("\nProblematic Time Regions (high 3rd derivative):")
third_deriv = x2_derivatives[3]
threshold = std(third_deriv) * 2  # 2 standard deviations above mean
problematic_indices = findall(abs.(third_deriv) .> threshold)
if length(problematic_indices) > 0
    println("High variation times:")
    for i in problematic_indices[1:min(10, end)]  # Show first 10
        println(@sprintf("  t=%.2f: d³x2/dt³=%.2e", t[i], third_deriv[i]))
    end
end

println("\nConclusion:")
println("- Van der Pol oscillator has limit cycle behavior with sharp transitions")
println("- Higher derivatives become increasingly spiky and non-smooth")
println("- GPR assumes smooth, continuous functions - poor fit for this dynamics")
println("- The 3rd derivative shows extreme variations that GPR cannot capture well")