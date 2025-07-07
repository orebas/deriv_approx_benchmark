#!/usr/bin/env julia

"""
Standalone script for benchmarking derivative approximation methods.
No package dependencies - just uses what's available in ODEParameterEstimation.

Usage:
	julia --project=.. benchmark_derivatives.jl [--noise 0.01] [--datasize 51] [--methods GPR,AAA]

This is a cleaned-up, production-ready version of study_approx.jl
"""

# Add parent directory to load ODEParameterEstimation
# push!(LOAD_PATH, dirname(@__DIR__))

using ODEParameterEstimation
using Statistics
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using DataFrames
using OrderedCollections
using OrdinaryDiffEq
using GaussianProcesses
using Loess
using BaryRational
using Printf
using Dierckx
using Random
using Dates
using LinearAlgebra
using Optim
using LineSearches
using ForwardDiff
using Suppressor
using Symbolics
using CSV
using ArgParse
using JSON
using Logging

# Try to use LoggingExtras if available, fallback to basic logging
global HAS_LOGGING_EXTRAS = false
try
    using LoggingExtras
    global HAS_LOGGING_EXTRAS = true
catch
    global HAS_LOGGING_EXTRAS = false
end

# Load the examples
# To make this script more portable, we assume the example model files
# are located relative to the project's `src` directory.
try
	include("src/examples/models/classical_systems.jl")
	include("src/examples/models/biological_systems.jl")
	include("src/examples/models/advanced_systems.jl")
catch e
	@warn "Could not include model files from default 'src/examples/models/' path."
	@warn "Please ensure these files exist or adjust the include paths if necessary."
	@warn "Original error: " e
end

# Configuration struct (must be defined before including approximation_methods.jl)
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

	# Method parameters
	gpr_jitter::Float64
	gpr_noise_threshold::Float64
	aaa_tol_low::Float64
	aaa_tol_high::Float64
	aaa_max_degree::Int
	loess_span::Float64
	spline_order::Int
end

function BenchmarkConfig(;
	example_name = "lv_periodic",
	noise_level = 1e-3,
	noise_type = "additive",
	data_size = 201,
	methods = ["GPR", "BSpline5", "TVDiff", "LOESS", "AAA_lowpres", "AAA"],
	derivative_orders = 4,
	output_dir = "./results",
	experiment_name = "benchmark_" * Dates.format(now(), "yyyymmdd_HHMMSS"),
	random_seed = 42,
	verbose = true,
	gpr_jitter = 1e-8,
	gpr_noise_threshold = 1e-12,
	aaa_tol_low = 0.1,
	aaa_tol_high = 1e-14,
	aaa_max_degree = 48,
	loess_span = 0.2,
	spline_order = 5,
)
	BenchmarkConfig(
		example_name, noise_level, noise_type, data_size, methods,
		derivative_orders, output_dir, experiment_name, random_seed, verbose,
		gpr_jitter, gpr_noise_threshold, aaa_tol_low, aaa_tol_high,
		aaa_max_degree, loess_span, spline_order,
	)
end

# Include approximation methods module (after BenchmarkConfig is defined)
include("src/approximation_methods.jl")

"""
Calculate symbolic derivatives of observables up to specified order.
"""
function calculate_observable_derivatives(equations, measured_quantities, nderivs)
	equation_dict = Dict(eq.lhs => eq.rhs for eq in equations)
	n_observables = length(measured_quantities)

	ObservableDerivatives = Symbolics.variables(:d_obs, 1:n_observables, 1:nderivs)
	SymbolicDerivs = Vector{Vector{Equation}}(undef, nderivs)

	# First derivatives
	SymbolicDerivs[1] = [
		ObservableDerivatives[i, 1] ~ substitute(
			expand_derivatives(D(measured_quantities[i].rhs)),
			equation_dict,
		) for i in 1:n_observables
	]

	# Higher order derivatives
	for j in 2:nderivs
		SymbolicDerivs[j] = [
			ObservableDerivatives[i, j] ~ substitute(
				expand_derivatives(D(SymbolicDerivs[j-1][i].rhs)),
				equation_dict,
			) for i in 1:n_observables
		]
	end

	expanded_measured_quantities = copy(measured_quantities)
	append!(expanded_measured_quantities, vcat(SymbolicDerivs...))

	return expanded_measured_quantities, ObservableDerivatives
end

"""
Generate clean and noisy datasets with true derivatives.
"""
function generate_datasets(pep, config::BenchmarkConfig)
	time_interval = pep.recommended_time_interval
	if isnothing(time_interval)
		time_interval = [0.0, 5.0]
	end

	# Calculate derivatives symbolically
	expanded_mq, obs_derivs = calculate_observable_derivatives(
		equations(pep.model.system),
		pep.measured_quantities,
		config.derivative_orders,
	)

	# Create new ODESystem with derivative observables
	@named new_sys = ODESystem(
		equations(pep.model.system),
		t;
		observed = expanded_mq,
	)

	# Solve ODE with high accuracy
	prob = ODEProblem(
		structural_simplify(new_sys),
		pep.ic,
		(time_interval[1], time_interval[2]),
		pep.p_true,
	)

	sol = solve(
		prob,
		AutoVern9(Rodas4P()),
		abstol = 1e-14,
		reltol = 1e-14,
		saveat = range(time_interval[1], time_interval[2], length = config.data_size),
	)

	# Extract clean data
	clean_data = OrderedDict{Any, Vector{Float64}}()
	clean_data["t"] = sol.t

	obs_to_key = Dict()
	for mq in pep.measured_quantities
		key = Num(mq.rhs)
		clean_data[key] = sol[mq.lhs]
		obs_to_key[mq.lhs] = key
	end

	# Extract derivatives
	derivatives = OrderedDict{Any, Vector{Float64}}()
	derivatives["t"] = sol.t

	for i in 1:length(pep.measured_quantities)
		obs_key = obs_to_key[pep.measured_quantities[i].lhs]
		for d in 1:config.derivative_orders
			derivatives["d$(d)_$obs_key"] = sol[obs_derivs[i, d]]
		end
	end

	# Generate noisy data
	noisy_data = OrderedDict{Any, Vector{Float64}}()
	for (key, values) in clean_data
		if key == "t"
			noisy_data[key] = values
		else
			if config.noise_type == "additive"
				noise_scale = config.noise_level * mean(abs.(values))
				noise = noise_scale * randn(length(values))
				noisy_data[key] = values + noise
			else
				error("Only additive noise supported")
			end
		end
	end

	return (
		clean = clean_data,
		noisy = noisy_data,
		derivatives = derivatives,
		measured_quantities = pep.measured_quantities,
		obs_derivs = obs_derivs,
		solution = sol,
	)
end

"""
Save generated datasets to CSV files for cross-language use.
"""
function save_datasets_to_csv(ode_name, noise_level, datasets)
	output_dir = "test_data/$(ode_name)/noise_$(noise_level)"
	mkpath(output_dir)

	# Convert all keys to strings to ensure consistent column names
	noisy_dict_str_keys = OrderedDict(string(k) => v for (k, v) in datasets.noisy)
	clean_dict_str_keys = OrderedDict(string(k) => v for (k, v) in datasets.clean)
	deriv_dict_str_keys = OrderedDict(string(k) => v for (k, v) in datasets.derivatives)

	# Save noisy data (input for methods)
	noisy_df = DataFrame(noisy_dict_str_keys)
	CSV.write(joinpath(output_dir, "noisy_data.csv"), noisy_df)

	# Save clean data and true derivatives (for error calculation)
	truth_df = DataFrame(clean_dict_str_keys)
	# Merge derivatives into the same DataFrame
	for (key, val) in deriv_dict_str_keys
		if key != "t"
			truth_df[!, key] .= val
		end
	end
	CSV.write(joinpath(output_dir, "truth_data.csv"), truth_df)
end

# NOTE: Commenting out duplicate functions - these are now provided by src/approximation_methods.jl
# This ensures TVDiff support and maintains consistency
#=

"""
Create GPR approximation with fallback to AAA.
"""
function create_gpr_approximation(t, y, config::BenchmarkConfig)
	y_mean = mean(y)
	y_std = std(y)
	y_normalized = (y .- y_mean) ./ y_std

	kernel = SEIso(log(std(t) / 8), 0.0)
	y_jittered = y_normalized .+ config.gpr_jitter * randn(length(y))

	gp = GP(t, y_jittered, MeanZero(), kernel, -2.0)

	@suppress begin
		GaussianProcesses.optimize!(gp;
			method = LBFGS(linesearch = LineSearches.BackTracking()),
		)
	end

	noise_level = exp(gp.logNoise.value)
	if noise_level < config.gpr_noise_threshold
		return aaad(t, y)  # Fallback to AAA
	end

	function gpr_func(x)
		pred, _ = predict_y(gp, [x])
		return y_std * pred[1] + y_mean
	end

	return gpr_func
end

"""
Create AAA approximation object.
"""
function create_aaa_approximation(t, y, config::BenchmarkConfig; high_precision = true)
	if high_precision
		return BaryRational.aaa(t, y, verbose = false, tol = config.aaa_tol_high)
	else
		# Low precision with BIC selection
		y_mean = mean(y)
		y_std = std(y)
		y_normalized = (y .- y_mean) ./ y_std

		best_bic = Inf
		best_approx = nothing

		tol = config.aaa_tol_low
		for m in 1:min(config.aaa_max_degree, length(t)÷2)
			tol = tol / 2.0

			approx = BaryRational.aaa(t, y_normalized, verbose = false, tol = tol)

			residuals = y_normalized .- [approx(x) for x in t]
			ssr = sum(abs2, residuals)
			k = 2 * length(approx.x)
			n = length(t)
			bic = k * log(n) + n * log(ssr / n + 1e-100)

			if bic < best_bic
				best_bic = bic
				best_approx = approx
			end
		end

		# Denormalize the approximation result
		best_approx.f .= y_std .* best_approx.f .+ y_mean
		return best_approx
	end
end

"""
Create LOESS approximation with AAA post-processing, returning the AAA object.
"""
function create_loess_approximation(t, y, config::BenchmarkConfig)
	model = loess(collect(t), y, span = config.loess_span)
	predictions = Loess.predict(model, t)
	return BaryRational.aaa(t, predictions, verbose = false, tol = config.aaa_tol_high)
end

"""
Create B-spline approximation.
"""
function create_bspline_approximation(t, y, config::BenchmarkConfig)
	n = length(t)
	mean_y = mean(abs.(y))
	expected_noise = config.noise_level * mean_y
	s = n * expected_noise^2

	spl = Spline1D(t, y; k = config.spline_order, s = s)

	return spl
end

"""
Evaluate all approximation methods.
"""
function evaluate_all_methods(datasets, pep, config::BenchmarkConfig)
	results = Dict{Any, Dict{String, Any}}()
	t_eval = datasets.clean["t"]

	for (i, mq) in enumerate(pep.measured_quantities)
		key = Num(mq.rhs)
		if key == "t"
			continue
		end

		if config.verbose
			@info "  Processing observable: $key"
		end

		results[key] = Dict{String, Any}()

		t = datasets.noisy["t"]
		y = datasets.noisy[key]

		for method in config.methods
			method_start_time = time()

			try
				# Create approximation function or object
				approx_obj = if method == "GPR"
					create_gpr_approximation(t, y, config)
				elseif method == "AAA"
					create_aaa_approximation(t, y, config, high_precision = true)
				elseif method == "AAA_lowpres"
					create_aaa_approximation(t, y, config, high_precision = false)
				elseif method == "LOESS"
					create_loess_approximation(t, y, config)
				elseif method == "BSpline5"
					create_bspline_approximation(t, y, config)
				else
					error("Unknown method: $method")
				end

				# Evaluate function and derivatives
				method_result = Dict{String, Any}()
				if method == "BSpline5"
					spl = approx_obj
					method_result["y"] = [evaluate(spl, x) for x in t_eval]
					for d in 1:config.derivative_orders
						method_result["d$d"] = [derivative(spl, x; nu = d) for x in t_eval]
					end
				elseif occursin("AAA", method) || method == "LOESS"
					# Use the built-in derivative for AAA objects from BaryRational
					aaa_approx = approx_obj
					method_result["y"] = [aaa_approx(x) for x in t_eval]
					for d in 1:config.derivative_orders
						method_result["d$d"] = [BaryRational.deriv(aaa_approx, x, m = d) for x in t_eval]
					end
				else # GPR is the only other method that returns a function
					approx_func = approx_obj
					method_result["y"] = [approx_func(x) for x in t_eval]

					for d in 1:config.derivative_orders
						method_result["d$d"] = [nth_deriv_at(approx_func, d, x) for x in t_eval]
					end
				end

				method_result["computation_time"] = time() - method_start_time

				# Calculate errors
				error_dict = Dict{String, NamedTuple}()

				# Function errors
				true_vals = datasets.clean[key]
				pred_vals = method_result["y"]
				rmse = sqrt(mean((pred_vals .- true_vals) .^ 2))
				mae = mean(abs.(pred_vals .- true_vals))
				max_err = maximum(abs.(pred_vals .- true_vals))
				error_dict["y"] = (rmse = rmse, mae = mae, max_error = max_err)

				# Derivative errors
				for d in 1:config.derivative_orders
					true_vals = datasets.derivatives["d$(d)_$key"]
					pred_vals = method_result["d$d"]
					rmse = sqrt(mean((pred_vals .- true_vals) .^ 2))
					mae = mean(abs.(pred_vals .- true_vals))
					max_err = maximum(abs.(pred_vals .- true_vals))
					error_dict["d$d"] = (rmse = rmse, mae = mae, max_error = max_err)
				end

				method_result["errors"] = error_dict
				results[key][method] = method_result

				if config.verbose
					@info "    ✓ $method completed in $(round(method_result["computation_time"], digits=3))s"
				end
			catch e
				if config.verbose
					@error "    ✗ $method failed: $(typeof(e))"
				end
				@warn "Method $method failed for $key" exception=e
			end
		end
	end

	return results
end

=# # End of commented duplicate functions

function results_to_summary_df(results, config::BenchmarkConfig)
	rows = []

	for (obs_key, obs_results) in results
		# Skip metadata key added by new evaluate_all_methods function
		if obs_key == "metadata"
			continue
		end
		
		for (method_name, method_results) in obs_results
			if !haskey(method_results, "errors")
				continue
			end

			comp_time = get(method_results, "computation_time", missing)

			for d in 0:config.derivative_orders
				d_key = d == 0 ? "y" : "d$(d)"

				if haskey(method_results["errors"], d_key)
					error_metrics = method_results["errors"][d_key]

					row = (
						method = method_name,
						noise_level = config.noise_level,
						derivative_order = d,
						rmse = error_metrics.rmse,
						mae = error_metrics.mae,
						max_error = error_metrics.max_error,
						eval_time = comp_time,
						fit_time = 0.0, # Julia script doesn't separate fit/eval
						success = true,
						category = "Julia",
						observable = string(obs_key),
					)
					push!(rows, row)
				end
			end
		end
	end

	return DataFrame(rows)
end

"""
Load configuration from a JSON file and merge with defaults.
"""
function load_config_from_file(file_path::String)
	@info "... Loading configuration from $file_path"
	config_dict = JSON.parsefile(file_path)

	# Extract parameters and merge them into the BenchmarkConfig constructor
	# This allows the JSON to override any of the defaults.
	return BenchmarkConfig(
		data_size = get(config_dict["data_config"], "data_size", 201),
		derivative_orders = get(config_dict["data_config"], "derivative_orders", 4),
		random_seed = get(config_dict["data_config"], "random_seed", 42),
		methods = get(config_dict, "julia_methods", ["GPR", "BSpline5"]),
		# Other parameters can be added here if they are in the JSON
	)
end

"""
Setup logging to file and console.
"""
function setup_logging()
	# Ensure unified_analysis directory exists
	mkpath("./unified_analysis")
	
	# Create log file with timestamp
	timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
	log_file = joinpath("unified_analysis", "julia_benchmark_$(timestamp).log")
	
	if HAS_LOGGING_EXTRAS
		# Create file logger
		file_logger = SimpleLogger(open(log_file, "w"))
		
		# Create console logger
		console_logger = ConsoleLogger(stdout, Logging.Info)
		
		# Combine both loggers
		global_logger(TeeLogger(console_logger, file_logger))
	else
		# Fallback to console logging only
		global_logger(ConsoleLogger(stdout, Logging.Info))
		println("Warning: LoggingExtras not available, logging to console only")
	end
	
	@info "Julia benchmark logging initialized" log_file=log_file
	
	return log_file
end

"""
Main benchmark function.
"""
function run_full_sweep(config_path::Union{String, Nothing} = nothing)
	mkpath("./results") # Ensure output directory
	mkpath("./test_data") # Ensure data export directory
	
	# Setup logging
	log_file = setup_logging()
	start_time = time()

	@info "="^60
	@info "RUNNING FULL JULIA BENCHMARK SWEEP (CONFIG-DRIVEN)"
	@info "="^60

	# 1. LOAD CONFIGURATION
	# --------------------------
	# Parse config file once and reuse throughout
	full_config_dict = Dict{String,Any}()
	
	if isnothing(config_path)
		@info "... No config file provided, using default settings."
		default_config = BenchmarkConfig()
		ode_problems_to_test = ["lv_periodic"]
		noise_levels = [0.0, 0.01]
		Random.seed!(default_config.random_seed)
		@info "... Set random seed to $(default_config.random_seed) for reproducible results"
	else
		# This part is not fully implemented for all params, but sets the stage
		base_config = load_config_from_file(config_path)
		full_config_dict = JSON.parsefile(config_path)  # Parse once here
		ode_problems_to_test = get(full_config_dict, "ode_problems", ["lv_periodic"])
		noise_levels = get(full_config_dict, "noise_levels", [0.0, 0.01])
		@info "... Loaded $(length(ode_problems_to_test)) ODEs and $(length(noise_levels)) noise levels from config."
		Random.seed!(base_config.random_seed)
		@info "... Set random seed to $(base_config.random_seed) for reproducible results"
	end

	@info "Testing against $(length(ode_problems_to_test)) ODE models: $(join(ode_problems_to_test, ", "))"

	# 2. SETUP BENCHMARK PARAMETERS
	# -------------------------------
	all_results_df = DataFrame()

	# 3. MAIN LOOP OVER ODES AND NOISE
	# ----------------------------------
	for ode_name in ode_problems_to_test
		@info "\n" * "-"^50
		@info "🚀 Starting test case: $ode_name"
		@info "-"^50

		# Load the ODE problem dynamically using the functions from the included files
		model_func = getfield(Main, Symbol(ode_name))
		pep = model_func()

		for noise_level in noise_levels
			# Use the base_config and override noise_level for this specific run
			config = if isnothing(config_path)
				BenchmarkConfig(example_name = ode_name, noise_level = noise_level)
			else
				# Re-create config for each run to correctly set noise/name
				# Using already-parsed full_config_dict from above
				BenchmarkConfig(
					example_name = ode_name,
					noise_level = noise_level,
					data_size = get(full_config_dict["data_config"], "data_size", 201),
					derivative_orders = get(full_config_dict["data_config"], "derivative_orders", 4),
					random_seed = get(full_config_dict["data_config"], "random_seed", 42),
					methods = get(full_config_dict, "julia_methods", ["GPR", "BSpline5"]),
				)
			end


			# Generate datasets for this ODE and noise level
			datasets = generate_datasets(pep, config)

			# Save datasets for Python to use
			save_datasets_to_csv(ode_name, noise_level, datasets)

			# Run methods and get raw results
			results = evaluate_all_methods(datasets, pep, config)

			# Convert to summary DataFrame for this run
			summary_df = results_to_summary_df(results, config)

			# Add the test case name to the results
			summary_df[!, :test_case] .= ode_name

			append!(all_results_df, summary_df)
		end
	end

	# 4. SAVE FINAL RESULTS
	# -----------------------
	output_file = "results/julia_raw_benchmark.csv"
	CSV.write(output_file, all_results_df)

	# Calculate total elapsed time
	elapsed_time = time() - start_time
	
	@info "\n🎉 FULL JULIA BENCHMARK SWEEP COMPLETE!"
	@info "📁 Raw Julia results for all ODEs saved to: $(output_file)"
	@info "⏱️  Total runtime: $(round(elapsed_time, digits=1)) seconds ($(round(elapsed_time/60, digits=1)) minutes)"
	@info "📝 Full log saved to: $log_file"
end

"""
Parse command-line arguments.
"""
function parse_commandline()
	s = ArgParseSettings()
	@add_arg_table! s begin
		"--config"
		help = "Path to the JSON configuration file."
		arg_type = String
	end
	return parse_args(s)
end

# This is the main entry point of the script
if abspath(PROGRAM_FILE) == @__FILE__
	parsed_args = parse_commandline()
	config_path = parsed_args["config"]
	run_full_sweep(config_path)
end
