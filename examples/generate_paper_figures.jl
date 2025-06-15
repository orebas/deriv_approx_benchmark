# Generate figures for the paper

using Pkg
Pkg.activate(dirname(@__DIR__))

push!(LOAD_PATH, joinpath(dirname(@__DIR__), "src"))
using DerivativeApproximationBenchmark
using DataFrames
using CSV
using Plots
using StatsPlots

# Function to create comparison plot
function create_comparison_plot(results_file, output_file)
    df = CSV.read(results_file, DataFrame)
    
    # Create subplots for each derivative order
    plots = []
    
    for d in 0:5
        subset = df[df.derivative_order .== d, :]
        
        p = @df subset groupedboxplot(
            :method, 
            :absolute_error,
            ylabel = d == 0 ? "Function Error" : "$(d)th Derivative Error",
            xlabel = "",
            yscale = :log10,
            legend = false,
            outliers = false
        )
        
        push!(plots, p)
    end
    
    final_plot = plot(plots..., layout = (2, 3), size = (1200, 800))
    savefig(final_plot, output_file)
    
    return final_plot
end

# Function to create noise sensitivity plot
function create_noise_sensitivity_plot(noise_levels, output_file)
    # Collect results from different noise levels
    all_results = DataFrame()
    
    for noise in noise_levels
        filename = "./sensitivity_results/sensitivity_noise_$(noise).csv"
        if isfile(filename)
            df = CSV.read(filename, DataFrame)
            all_results = vcat(all_results, df)
        end
    end
    
    # Calculate mean RMSE by noise level and method
    summary = combine(
        groupby(all_results, [:noise_level, :method, :derivative_order]),
        :rmse => mean => :mean_rmse
    )
    
    # Create plots for each derivative order
    plots = []
    
    for d in 0:2  # Just show first 3 derivatives
        subset = summary[summary.derivative_order .== d, :]
        
        p = @df subset plot(
            :noise_level, 
            :mean_rmse,
            group = :method,
            xlabel = d == 2 ? "Noise Level" : "",
            ylabel = d == 0 ? "RMSE" : "",
            title = d == 0 ? "Function" : "$(d)th Derivative",
            xscale = :log10,
            yscale = :log10,
            marker = :circle,
            legend = d == 0 ? :topleft : false
        )
        
        push!(plots, p)
    end
    
    final_plot = plot(plots..., layout = (1, 3), size = (1200, 400))
    savefig(final_plot, output_file)
    
    return final_plot
end

# 1. Run main benchmark for paper
println("Running main benchmark...")
config = BenchmarkConfig(
    example_name = "lv_periodic",
    noise_level = 0.01,
    data_size = 101,
    methods = ["GPR", "AAA", "AAA_lowpres", "LOESS", "BSpline5"],
    output_dir = "./paper_results",
    experiment_name = "main_comparison"
)
run_benchmark(config)

# 2. Create comparison plot
println("Creating comparison plot...")
create_comparison_plot(
    "./paper_results/main_comparison.csv",
    "./paper_results/figure_1_method_comparison.pdf"
)

# 3. Run noise sensitivity analysis
println("Running noise sensitivity analysis...")
noise_levels = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]

for noise in noise_levels
    config = BenchmarkConfig(
        example_name = "lv_periodic",
        noise_level = noise,
        data_size = 51,
        methods = ["GPR", "AAA", "LOESS"],
        output_dir = "./sensitivity_results",
        experiment_name = "sensitivity_noise_$(noise)",
        verbose = false
    )
    run_benchmark(config)
end

# 4. Create noise sensitivity plot
println("Creating noise sensitivity plot...")
create_noise_sensitivity_plot(
    noise_levels,
    "./paper_results/figure_2_noise_sensitivity.pdf"
)

# 5. Generate LaTeX table
println("Generating LaTeX table...")
df = CSV.read("./paper_results/main_comparison.csv", DataFrame)

# Calculate mean errors by method and derivative order
summary = combine(
    groupby(df, [:method, :derivative_order]),
    :rmse => mean => :rmse,
    :mae => mean => :mae
)

# Pivot for LaTeX table
rmse_pivot = unstack(summary, :derivative_order, :method, :rmse)

# Write LaTeX table
open("./paper_results/table_1_rmse.tex", "w") do io
    println(io, "\\begin{table}[htbp]")
    println(io, "\\centering")
    println(io, "\\caption{Average RMSE for different methods and derivative orders}")
    println(io, "\\begin{tabular}{l" * "c"^(ncol(rmse_pivot)-1) * "}")
    println(io, "\\toprule")
    
    # Header
    print(io, "Derivative")
    for method in names(rmse_pivot)[2:end]
        print(io, " & ", method)
    end
    println(io, " \\\\")
    println(io, "\\midrule")
    
    # Data rows
    for row in eachrow(rmse_pivot)
        print(io, row[1])
        for val in row[2:end]
            print(io, " & ", @sprintf("%.2e", val))
        end
        println(io, " \\\\")
    end
    
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    println(io, "\\end{table}")
end

println("Paper figures generated in ./paper_results/")