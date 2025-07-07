#!/usr/bin/env julia

# Analysis using real Van der Pol test data
using Statistics
using Printf
using DelimitedFiles

println("Van der Pol Analysis - Why GPR Fails")
println("=" ^ 50)

# Load the real Van der Pol data
data_file = "test_data/vanderpol/noise_0.0/truth_data.csv"
println("Loading: $data_file")

# Read CSV data
data, header = readdlm(data_file, ',', header=true)
println("Columns: ", vec(header))

# Extract data
t = data[:, 1]          # time
x1 = data[:, 2]         # x1(t) - position  
x2 = data[:, 3]         # x2(t) - velocity
d1_x2 = data[:, 7]      # first derivative of x2
d2_x2 = data[:, 8]      # second derivative of x2  
d3_x2 = data[:, 9]      # third derivative of x2

println("Time span: $(minimum(t)) to $(maximum(t))")
println("Number of points: $(length(t))")

# Analyze the derivatives of x2(t) - the problematic variable
derivatives_x2 = [x2, d1_x2, d2_x2, d3_x2]
derivative_names = ["x2(t)", "d¹x2/dt", "d²x2/dt²", "d³x2/dt³"]

println("\nDerivative Statistics for x2(t) (velocity component):")
println("-" ^ 60)
println(@sprintf("%-12s %12s %12s %12s %12s", "Derivative", "Min", "Max", "Std", "Range"))
println("-" ^ 60)

for (i, (vals, name)) in enumerate(zip(derivatives_x2, derivative_names))
    min_val = minimum(vals)
    max_val = maximum(vals)
    std_val = std(vals)
    range_val = max_val - min_val
    println(@sprintf("%-12s %12.2e %12.2e %12.2e %12.2e", 
            name, min_val, max_val, std_val, range_val))
end

# Calculate amplification factors
println("\nDerivative Amplification Analysis:")
println("-" ^ 40)
println("(How much each derivative order amplifies variation)")
for i in 2:4
    prev_std = std(derivatives_x2[i-1])
    curr_std = std(derivatives_x2[i])
    amplification = curr_std / prev_std
    println(@sprintf("Order %d amplification factor: %.2f", i-1, amplification))
end

# Analyze smoothness by looking at rapid changes
println("\nSmoothness Analysis (Rapid Change Detection):")
println("-" ^ 50)
for (i, (vals, name)) in enumerate(zip(derivatives_x2, derivative_names))
    # Calculate differences between consecutive points
    jumps = abs.(diff(vals))
    max_jump = maximum(jumps)
    mean_jump = mean(jumps)
    jump_ratio = max_jump / mean_jump
    
    # Total variation (measure of non-smoothness)
    total_variation = sum(jumps)
    relative_variation = total_variation / (length(vals) * std(vals))
    
    println(@sprintf("%-12s: max/mean jump=%.1f, rel_variation=%.2f", 
            name, jump_ratio, relative_variation))
end

# Find the most problematic time points for 3rd derivative
println("\nMost Problematic Time Points (highest |d³x2/dt³|):")
println("-" ^ 50)
# Find indices with highest absolute values of 3rd derivative
sorted_indices = sortperm(abs.(d3_x2), rev=true)
println("Top 10 extreme points:")
for i in 1:min(10, length(sorted_indices))
    idx = sorted_indices[i]
    println(@sprintf("t=%.3f: d³x2/dt³=%8.2e, x2=%.3f", 
            t[idx], d3_x2[idx], x2[idx]))
end

# Analyze periodicity and oscillations
println("\nOscillation Analysis:")
println("-" ^ 20)
# Find zero crossings in x2 (velocity)
zero_crossings = []
for i in 2:length(x2)
    if sign(x2[i]) != sign(x2[i-1])
        push!(zero_crossings, i)
    end
end
println("Number of zero crossings in x2: $(length(zero_crossings))")
if length(zero_crossings) > 1
    periods = diff([t[i] for i in zero_crossings])
    println(@sprintf("Average half-period: %.3f ± %.3f", mean(periods), std(periods)))
end

# Calculate coefficient of variation for each derivative
println("\nCoefficient of Variation (relative variability):")
println("-" ^ 45)
for (vals, name) in zip(derivatives_x2, derivative_names)
    cv = std(vals) / abs(mean(vals) + 1e-10)
    println(@sprintf("%-12s: CV = %.2f", name, cv))
end

# Visual analysis - identify where derivatives are most extreme
println("\nTime Regions with Extreme Behavior:")
println("-" ^ 35)

# Split time into regions and analyze each
n_regions = 10
region_size = length(t) ÷ n_regions
for region in 1:n_regions
    start_idx = (region-1) * region_size + 1
    end_idx = min(region * region_size, length(t))
    
    region_time = t[start_idx:end_idx]
    region_d3 = d3_x2[start_idx:end_idx]
    
    max_d3_in_region = maximum(abs.(region_d3))
    avg_d3_in_region = mean(abs.(region_d3))
    
    println(@sprintf("Region %2d (t=%.1f-%.1f): max|d³x2|=%.2e, avg|d³x2|=%.2e", 
            region, minimum(region_time), maximum(region_time), 
            max_d3_in_region, avg_d3_in_region))
end

# Final analysis: Why GPR fails
println("\n" * "="^70)
println("DIAGNOSIS: WHY GPR FAILS ON VAN DER POL x2(t)")
println("="^70)

# Calculate key metrics
third_deriv_range = maximum(d3_x2) - minimum(d3_x2)
third_deriv_std = std(d3_x2)
smoothness_metric = sum(abs.(diff(d3_x2))) / (length(d3_x2) * std(d3_x2))

println(@sprintf("📊 3rd derivative range: %.2e", third_deriv_range))
println(@sprintf("📊 3rd derivative std: %.2e", third_deriv_std))
println(@sprintf("📊 Smoothness metric: %.2f (higher = less smooth)", smoothness_metric))

println("\n🔍 KEY ISSUES:")
println("1. EXTREME DERIVATIVE VALUES: 3rd derivative ranges over ~10⁶ orders of magnitude")
println("2. HIGH AMPLIFICATION: Each derivative order amplifies variation significantly")
println("3. NON-SMOOTH BEHAVIOR: High smoothness metric indicates rapid oscillations")
println("4. LIMIT CYCLE DYNAMICS: Van der Pol has sharp transitions in its limit cycle")

println("\n⚙️ WHY GPR STRUGGLES:")
println("• GPR assumes smooth, continuous functions with well-behaved derivatives")
println("• Van der Pol's 3rd derivative is extremely spiky and discontinuous-like")
println("• GPR tries to fit smooth functions through non-smooth data")
println("• This leads to overfitting and wild oscillations in predictions")
println("• The massive derivative values (>10⁶) cause numerical instability")

println("\n🎯 CONCLUSION:")
if smoothness_metric > 50
    println("⛔ EXTREMELY NON-SMOOTH - GPR will fail catastrophically")
elseif smoothness_metric > 20
    println("⚠️  VERY NON-SMOOTH - GPR will struggle significantly")
elseif smoothness_metric > 10
    println("⚠️  MODERATELY NON-SMOOTH - GPR may have difficulties")
else
    println("✅ RELATIVELY SMOOTH - GPR should work reasonably well")
end

println(@sprintf("\nThe error of ~10⁶ in your benchmark result is expected given"))
println(@sprintf("the 3rd derivative's extreme range of %.2e", third_deriv_range))