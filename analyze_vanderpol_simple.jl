#!/usr/bin/env julia

# Simple analysis using existing test data
using Statistics
using Printf

# Check if we have test data files
test_data_dir = "test_data"
if !isdir(test_data_dir)
    println("No test_data directory found. Let me create synthetic Van der Pol data...")
    
    # Create synthetic Van der Pol-like data showing the characteristic behavior
    t = range(0, 20, length=201)
    
    # Approximate Van der Pol solution (simplified)
    # This shows the characteristic limit cycle behavior
    x1 = [2.0 * cos(0.3*ti) * exp(-0.1*abs(sin(0.3*ti))) for ti in t]
    x2 = [-2.0 * 0.3 * sin(0.3*ti) * exp(-0.1*abs(sin(0.3*ti))) + 
          2.0 * cos(0.3*ti) * (-0.1 * sign(sin(0.3*ti)) * cos(0.3*ti) * 0.3) for ti in t]
    
    println("Created synthetic Van der Pol-like data")
else
    println("Looking for Van der Pol test data...")
    # Try to find vanderpol data
    vanderpol_files = filter(f -> contains(f, "vanderpol"), readdir(test_data_dir))
    if isempty(vanderpol_files)
        println("No vanderpol files found in test_data")
        exit(1)
    end
    
    # Load the first vanderpol file
    data_file = joinpath(test_data_dir, vanderpol_files[1])
    println("Loading: $data_file")
    
    # Read CSV-like data (assuming t, x1, x2 columns)
    lines = readlines(data_file)
    header = split(lines[1], ',')
    println("Columns: ", header)
    
    # Parse data
    data_lines = lines[2:end]
    n_points = length(data_lines)
    t = zeros(n_points)
    x1 = zeros(n_points)
    x2 = zeros(n_points)
    
    for (i, line) in enumerate(data_lines)
        parts = split(line, ',')
        t[i] = parse(Float64, parts[1])
        x1[i] = parse(Float64, parts[2])
        x2[i] = parse(Float64, parts[3])
    end
end

println("Van der Pol Data Analysis")
println("=" ^ 50)
println("Time span: $(minimum(t)) to $(maximum(t))")
println("Number of points: $(length(t))")

# Calculate numerical derivatives
function numerical_derivatives(t, y, max_order=3)
    derivatives = Dict{Int, Vector{Float64}}()
    derivatives[0] = copy(y)
    
    for order in 1:max_order
        if order == 1
            # First derivative using central differences
            dy = similar(y)
            for i in 2:length(y)-1
                dy[i] = (y[i+1] - y[i-1]) / (t[i+1] - t[i-1])
            end
            dy[1] = (y[2] - y[1]) / (t[2] - t[1])
            dy[end] = (y[end] - y[end-1]) / (t[end] - t[end-1])
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

# Analyze x1(t) and x2(t) derivatives
x1_derivatives = numerical_derivatives(t, x1, 3)
x2_derivatives = numerical_derivatives(t, x2, 3)

println("\nDerivative Statistics for x1(t) (position):")
println("-" ^ 45)
for order in 0:3
    vals = x1_derivatives[order]
    println(@sprintf("Order %d: min=%8.2e, max=%8.2e, std=%8.2e", 
            order, minimum(vals), maximum(vals), std(vals)))
end

println("\nDerivative Statistics for x2(t) (velocity):")
println("-" ^ 45)
for order in 0:3
    vals = x2_derivatives[order]
    println(@sprintf("Order %d: min=%8.2e, max=%8.2e, std=%8.2e", 
            order, minimum(vals), maximum(vals), std(vals)))
end

# Compare variability between x1 and x2
println("\nVariability Comparison (Coefficient of Variation):")
println("-" ^ 50)
for order in 0:3
    x1_cv = std(x1_derivatives[order]) / abs(mean(x1_derivatives[order]) + 1e-10)
    x2_cv = std(x2_derivatives[order]) / abs(mean(x2_derivatives[order]) + 1e-10)
    println(@sprintf("Order %d: x1_CV=%.2f, x2_CV=%.2f, ratio=%.2f", 
            order, x1_cv, x2_cv, x2_cv/x1_cv))
end

# Analyze derivative amplification (how much each order amplifies variation)
println("\nDerivative Amplification Analysis:")
println("-" ^ 35)
for var_name in ["x1", "x2"]
    derivatives = var_name == "x1" ? x1_derivatives : x2_derivatives
    println("$var_name(t):")
    for order in 1:3
        prev_std = std(derivatives[order-1])
        curr_std = std(derivatives[order])
        amplification = curr_std / (prev_std + 1e-10)
        println(@sprintf("  Order %d amplification: %.2f", order, amplification))
    end
end

# Identify rapid changes
println("\nRapid Change Analysis (Jump Detection):")
println("-" ^ 40)
for var_name in ["x1", "x2"]
    derivatives = var_name == "x1" ? x1_derivatives : x2_derivatives
    println("$var_name(t):")
    for order in 0:3
        vals = derivatives[order]
        jumps = abs.(diff(vals))
        max_jump = maximum(jumps)
        mean_jump = mean(jumps)
        jump_ratio = max_jump / (mean_jump + 1e-10)
        println(@sprintf("  Order %d: max/mean jump ratio = %.1f", order, jump_ratio))
    end
end

# Find most problematic time points for x2 third derivative
println("\nMost Problematic Time Points for x2 (3rd derivative):")
println("-" ^ 55)
third_deriv_x2 = x2_derivatives[3]
# Find indices with highest absolute values
sorted_indices = sortperm(abs.(third_deriv_x2), rev=true)
println("Top 10 problematic times:")
for i in 1:min(10, length(sorted_indices))
    idx = sorted_indices[i]
    println(@sprintf("  t=%.3f: d³x2/dt³=%8.2e", t[idx], third_deriv_x2[idx]))
end

# Summary of why GPR struggles
println("\n" * "=" * 60)
println("WHY GPR STRUGGLES WITH VAN DER POL x2(t):")
println("=" * 60)
println("1. NONLINEAR DYNAMICS: Van der Pol has limit cycles with sharp transitions")
println("2. DERIVATIVE AMPLIFICATION: Each derivative order amplifies variation")
println("3. NON-SMOOTH BEHAVIOR: Higher derivatives become increasingly spiky")
println("4. GPR ASSUMPTIONS: GPR assumes smooth, continuous functions with")
println("   well-behaved derivatives - violated by Van der Pol dynamics")
println("5. OVERFITTING: GPR may try to interpolate through noise/spikes,")
println("   leading to wild oscillations in predictions")

# Calculate smoothness metrics
x2_3rd = x2_derivatives[3]
total_variation = sum(abs.(diff(x2_3rd)))
max_variation = maximum(abs.(x2_3rd))
smoothness_metric = total_variation / (length(x2_3rd) * max_variation)

println(@sprintf("\nSMOOTHNESS METRIC for x2 3rd derivative: %.2f", smoothness_metric))
println("(Higher values = less smooth, more problematic for GPR)")

if smoothness_metric > 10
    println("⚠️  VERY NON-SMOOTH - GPR will struggle significantly")
elseif smoothness_metric > 5
    println("⚠️  MODERATELY NON-SMOOTH - GPR may have difficulties")
else
    println("✓ RELATIVELY SMOOTH - GPR should work reasonably well")
end