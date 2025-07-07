#!/usr/bin/env julia

"""
Standalone script to visualize GPR vs AAA comparison for VanderPol x2(t) data.
Shows the impact of the GPR bug fix on derivative approximation quality.

Usage:
    julia visualize_vanderpol_comparison.jl
"""

using ODEParameterEstimation
using CSV
using DataFrames
using Plots
using DelimitedFiles
using Statistics
using Printf
using GaussianProcesses
using BaryRational
using Loess
using Dierckx
using LinearAlgebra
using Optim
using LineSearches
using ForwardDiff
using Suppressor
using NoiseRobustDifferentiation

# BenchmarkConfig struct (copy from benchmark_derivatives.jl to match exactly)
struct BenchmarkConfig
    example_name::String
    noise_level::Float64
    noise_type::String
    data_size::Int
    methods::Vector{String}
    derivative_orders::Int
    output_dir::String
    experiment_name::String
    random_seed::Int
    verbose::Bool
    gpr_jitter::Float64
    gpr_noise_threshold::Float64
    aaa_tol_low::Float64
    aaa_tol_high::Float64
    aaa_max_degree::Int
    loess_span::Float64
    spline_order::Int
end

# Include approximation methods module (after BenchmarkConfig is defined)  
include("src/approximation_methods.jl")

function load_vanderpol_data()
    """Load VanderPol data from the test_data directory"""
    println("Loading VanderPol data...")
    
    # Load noisy data (for fitting)
    noisy_data, _ = readdlm("test_data/vanderpol/noise_1.0e-6/noisy_data.csv", ',', header=true)
    t_noisy = noisy_data[:, 1]
    x2_noisy = noisy_data[:, 3]  # x2(t) column
    
    # Load true data (for comparison)
    truth_data, _ = readdlm("test_data/vanderpol/noise_0.0/truth_data.csv", ',', header=true) 
    t_truth = truth_data[:, 1]
    x2_truth = truth_data[:, 3]  # x2(t) column
    d3_truth = truth_data[:, 9]  # 3rd derivative of x2(t) - column 9
    
    println("  Loaded $(length(t_noisy)) noisy data points")
    println("  Loaded $(length(t_truth)) truth data points")
    println("  Loaded 3rd derivative truth data")
    
    return t_noisy, x2_noisy, t_truth, x2_truth, d3_truth
end

function create_gpr_with_diagnostics(t, y, config)
    """Create GPR with detailed diagnostics to check for AAA fallback"""
    println("  Creating GPR with diagnostics...")
    
    # Replicate the GPR creation process with diagnostics
    y_mean = mean(y)
    y_std = std(y)
    y_normalized = (y .- y_mean) ./ y_std
    
    # Initial hyperparameters
    initial_lengthscale = log(std(t) / 8)
    initial_variance = 0.0
    initial_noise = -2.0
    
    println("    Initial hyperparameters:")
    println("      lengthscale: exp($(@sprintf("%.3f", initial_lengthscale))) = $(@sprintf("%.3f", exp(initial_lengthscale)))")
    println("      variance: exp($(@sprintf("%.3f", initial_variance))) = $(@sprintf("%.3f", exp(initial_variance)))")
    println("      noise: exp($(@sprintf("%.3f", initial_noise))) = $(@sprintf("%.2e", exp(initial_noise)))")
    
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
    
    # Check noise level with diagnostics
    noise_level = exp(gp.logNoise.value)
    println("    Final hyperparameters:")
    println("      noise_level: exp($(@sprintf("%.6f", gp.logNoise.value))) = $(@sprintf("%.2e", noise_level))")
    println("      threshold: $(@sprintf("%.2e", config.gpr_noise_threshold))")
    
    if noise_level < config.gpr_noise_threshold
        println("    ⚠️  FALLBACK TRIGGERED: noise_level ($(@sprintf("%.2e", noise_level))) < threshold ($(@sprintf("%.2e", config.gpr_noise_threshold)))")
        println("    GPR will fall back to AAA!")
        return create_aaa_approximation(t, y, config, high_precision=true), true
    else
        println("    ✅ Using GPR: noise level is above threshold")
        
        # Create callable function
        function gpr_func(x)
            pred, _ = predict_y(gp, [x])
            return y_std * pred[1] + y_mean
        end
        
        return gpr_func, false
    end
end

function create_comparison_plot(t_noisy, x2_noisy, t_truth, x2_truth, d3_truth, config)
    """Create approximations and generate comparison plot - GPR only"""
    println("Creating approximations...")
    
    # Create approximations using the existing functions
    try
        println("  Fitting GPR with full diagnostics...")
        gpr_func, used_fallback = create_gpr_with_diagnostics(t_noisy, x2_noisy, config)
        x2_gpr = [gpr_func(t) for t in t_truth]
        
        println("  Computing GPR 3rd derivatives...")
        d3_gpr = [nth_deriv_at(gpr_func, 3, t) for t in t_truth]
        
        # Calculate errors for display
        gpr_rmse = sqrt(mean((x2_gpr .- x2_truth).^2))
        gpr_d3_rmse = sqrt(mean((d3_gpr .- d3_truth).^2))
        
        println("  Function RMSE - GPR: $(@sprintf("%.2e", gpr_rmse))")
        println("  3rd Deriv RMSE - GPR: $(@sprintf("%.2e", gpr_d3_rmse))")
        
        # Find worst 3rd derivative errors
        d3_errors = abs.(d3_gpr .- d3_truth)
        worst_indices = sortperm(d3_errors, rev=true)[1:5]  # Top 5 worst errors
        
        println("  Worst 3rd derivative errors:")
        for (i, idx) in enumerate(worst_indices)
            println("    $i. t=$(@sprintf("%.3f", t_truth[idx])): error=$(@sprintf("%.2e", d3_errors[idx])), pred=$(@sprintf("%.2e", d3_gpr[idx])), true=$(@sprintf("%.2e", d3_truth[idx]))")
        end
        
        # Create subplots
        println("Creating plots...")
        gr(size=(1400, 1000), dpi=300)
        
        # Function plot
        p1 = plot(
            title="VanderPol x2(t): GPR Analysis (noise=1e-6)" * (used_fallback ? " [USING AAA FALLBACK!]" : ""),
            xlabel="Time (t)",
            ylabel="x2(t)",
            legend=:bottomright,
            fontsize=10
        )
        
        # Plot true data as reference
        plot!(p1, t_truth, x2_truth,
            label="True Data",
            linewidth=2.5,
            color=:black,
            alpha=0.8
        )
        
        # Plot noisy input data
        scatter!(p1, t_noisy[1:5:end], x2_noisy[1:5:end],  # Subsample for clarity
            label="Noisy Input (1e-6)",
            markersize=2,
            markerstrokewidth=0,
            alpha=0.6,
            color=:gray
        )
        
        # Plot GPR approximation
        plot!(p1, t_truth, x2_gpr,
            label="GPR" * (used_fallback ? " (AAA fallback)" : " (pure GPR)") * " (RMSE: $(@sprintf("%.2e", gpr_rmse)))",
            linewidth=2,
            linestyle=:solid,
            color=:red
        )
        
        # 3rd derivative plot
        p2 = plot(
            title="VanderPol x2(t): 3rd Derivative Analysis",
            xlabel="Time (t)",
            ylabel="d³x2/dt³",
            legend=:topright,
            fontsize=10
        )
        
        # Plot true 3rd derivative
        plot!(p2, t_truth, d3_truth,
            label="True 3rd Derivative",
            linewidth=2.5,
            color=:black,
            alpha=0.8
        )
        
        # Plot GPR 3rd derivative
        plot!(p2, t_truth, d3_gpr,
            label="GPR 3rd Deriv (RMSE: $(@sprintf("%.2e", gpr_d3_rmse)))",
            linewidth=2,
            linestyle=:solid,
            color=:red
        )
        
        # Highlight worst error points
        scatter!(p2, t_truth[worst_indices], d3_gpr[worst_indices],
            label="Worst Errors",
            markersize=4,
            color=:orange,
            markerstrokewidth=1,
            markerstrokecolor=:black
        )
        
        # Combine plots
        p_combined = plot(p1, p2, layout=(2,1), size=(1400, 1000))
        
        return p_combined, gpr_rmse, gpr_d3_rmse, x2_gpr, d3_gpr, used_fallback, worst_indices, d3_errors
        
    catch e
        println("Error during approximation:")
        println(e)
        return nothing, NaN, NaN, [], [], false, [], []
    end
end

function create_error_plot(t_truth, x2_truth, d3_truth, x2_gpr, d3_gpr, gpr_rmse, gpr_d3_rmse, worst_indices)
    """Create error plots for GPR function and 3rd derivative"""
    if isempty(x2_gpr) || isempty(d3_gpr)
        return nothing
    end
    
    # Function errors
    gpr_errors = abs.(x2_gpr .- x2_truth)
    
    # 3rd derivative errors
    gpr_d3_errors = abs.(d3_gpr .- d3_truth)
    
    # Function error plot
    p1_error = plot(
        title="GPR Function Absolute Errors",
        xlabel="Time (t)",
        ylabel="Absolute Error",
        legend=:topright,
        yscale=:log10,
        fontsize=10
    )
    
    plot!(p1_error, t_truth, gpr_errors,
        label="GPR Function Error (RMSE: $(@sprintf("%.2e", gpr_rmse)))",
        linewidth=2,
        color=:red
    )
    
    # 3rd derivative error plot
    p2_error = plot(
        title="GPR 3rd Derivative Absolute Errors",
        xlabel="Time (t)",
        ylabel="Absolute Error",
        legend=:topright,
        yscale=:log10,
        fontsize=10
    )
    
    plot!(p2_error, t_truth, gpr_d3_errors,
        label="GPR 3rd Deriv Error (RMSE: $(@sprintf("%.2e", gpr_d3_rmse)))",
        linewidth=2,
        color=:red
    )
    
    # Highlight worst error points
    scatter!(p2_error, t_truth[worst_indices], gpr_d3_errors[worst_indices],
        label="Worst Errors",
        markersize=4,
        color=:orange,
        markerstrokewidth=1,
        markerstrokecolor=:black
    )
    
    # Combine error plots
    p_error_combined = plot(p1_error, p2_error, layout=(2,1), size=(1400, 1000))
    
    return p_error_combined
end

function save_data_tables(t_truth, x2_truth, d3_truth, x2_gpr, d3_gpr, worst_indices, d3_errors, used_fallback)
    """Save numerical results to CSV files for analysis"""
    println("Saving data tables...")
    
    # Create main data table
    data_df = DataFrame(
        t = t_truth,
        x2_true = x2_truth,
        x2_gpr = x2_gpr,
        x2_error = abs.(x2_gpr .- x2_truth),
        d3_true = d3_truth,
        d3_gpr = d3_gpr,
        d3_error = abs.(d3_gpr .- d3_truth)
    )
    
    CSV.write("vanderpol_gpr_results.csv", data_df)
    println("  Saved main results to vanderpol_gpr_results.csv")
    
    # Create worst errors table
    worst_df = DataFrame(
        rank = 1:length(worst_indices),
        t = t_truth[worst_indices],
        d3_true = d3_truth[worst_indices],
        d3_gpr = d3_gpr[worst_indices],
        error = d3_errors[worst_indices],
        relative_error = d3_errors[worst_indices] ./ abs.(d3_truth[worst_indices])
    )
    
    CSV.write("vanderpol_worst_d3_errors.csv", worst_df)
    println("  Saved worst errors to vanderpol_worst_d3_errors.csv")
    
    # Create summary table
    summary_df = DataFrame(
        metric = ["GPR_used_fallback", "function_rmse", "d3_rmse", "max_d3_error", "mean_d3_error", "std_d3_error"],
        value = [used_fallback, sqrt(mean((x2_gpr .- x2_truth).^2)), sqrt(mean((d3_gpr .- d3_truth).^2)), 
                maximum(d3_errors), mean(d3_errors), std(d3_errors)]
    )
    
    CSV.write("vanderpol_summary.csv", summary_df)
    println("  Saved summary to vanderpol_summary.csv")
end

function main()
    println("VanderPol GPR Analysis with Fallback Diagnostics")
    println("=" ^ 50)
    
    # Configuration (same values as used in benchmark)
    config = BenchmarkConfig(
        "vanderpol",      # example_name
        1e-6,             # noise_level
        "additive",       # noise_type
        201,              # data_size
        ["GPR", "AAA"],   # methods
        5,                # derivative_orders
        "./results",      # output_dir
        "comparison",     # experiment_name
        42,               # random_seed
        true,             # verbose
        1e-8,             # gpr_jitter
        1e-12,            # gpr_noise_threshold
        0.1,              # aaa_tol_low
        1e-14,            # aaa_tol_high
        48,               # aaa_max_degree
        0.2,              # loess_span
        5                 # spline_order
    )
    
    # Load data
    t_noisy, x2_noisy, t_truth, x2_truth, d3_truth = load_vanderpol_data()
    
    # Create comparison plot with diagnostics
    p_main, gpr_rmse, gpr_d3_rmse, x2_gpr, d3_gpr, used_fallback, worst_indices, d3_errors = create_comparison_plot(
        t_noisy, x2_noisy, t_truth, x2_truth, d3_truth, config
    )
    
    if p_main !== nothing
        # Save main comparison plot
        output_file = "vanderpol_gpr_analysis.png"
        println("Saving comparison plot to $output_file...")
        savefig(p_main, output_file)
        
        # Create and save error plot
        p_error = create_error_plot(t_truth, x2_truth, d3_truth, x2_gpr, d3_gpr, gpr_rmse, gpr_d3_rmse, worst_indices)
        if p_error !== nothing
            error_file = "vanderpol_gpr_errors.png"
            println("Saving error plot to $error_file...")
            savefig(p_error, error_file)
        end
        
        # Save data tables
        save_data_tables(t_truth, x2_truth, d3_truth, x2_gpr, d3_gpr, worst_indices, d3_errors, used_fallback)
        
        println("\n" * "="^60)
        println("ANALYSIS RESULTS:")
        println("="^60)
        println("GPR used AAA fallback: $(used_fallback ? "YES ⚠️" : "NO ✅")")
        println("Function RMSE:         $(@sprintf("%.2e", gpr_rmse))")
        println("3rd Deriv RMSE:        $(@sprintf("%.2e", gpr_d3_rmse))")
        println("Max 3rd deriv error:   $(@sprintf("%.2e", maximum(d3_errors)))")
        println("Mean 3rd deriv error:  $(@sprintf("%.2e", mean(d3_errors)))")
        
        if used_fallback
            println("\n🎯 DIAGNOSIS: GPR is falling back to AAA!")
            println("   This explains why both methods had identical RMSE values.")
            println("   The noise threshold may be too high, or GPR optimization")
            println("   is driving noise to artificially low values.")
        else
            println("\n🔍 DIAGNOSIS: Pure GPR is being used (no fallback)")
            if gpr_d3_rmse > 1e6
                println("   The 3rd derivative errors are still very large.")
                println("   This suggests the problem is in ForwardDiff computation")
                println("   or numerical instability in derivative calculation.")
            end
        end
        
        println("\n📁 Generated files:")
        println("   - $output_file (comparison plot)")
        println("   - $error_file (error plot)")
        println("   - vanderpol_gpr_results.csv (all data)")
        println("   - vanderpol_worst_d3_errors.csv (worst errors)")
        println("   - vanderpol_summary.csv (summary stats)")
        
    else
        println("Failed to create plots due to errors.")
    end
end

# Run the main function
main()